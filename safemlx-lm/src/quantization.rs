//! Model-agnostic load-time quantization for safetensors checkpoints.
//!
//! The serialized representation follows the MLX-LM convention: a dense
//! `name.weight` tensor becomes a packed `name.weight` tensor plus
//! `name.scales` and `name.biases`. Quantization settings are stored in both
//! the `quantization` and `quantization_config` keys in `config.json` for
//! compatibility with MLX-LM and Hugging Face tooling.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

use safemlx::{ops, Array, Stream};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};

use crate::{error::Error, weights};

/// MLX affine quantization settings stored in `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineQuantization {
    /// Number of adjacent input values sharing one scale and bias.
    pub group_size: i32,
    /// Packed bit width for each weight value.
    pub bits: i32,
    /// Quantization mode. Safemlx currently supports `affine` checkpoints.
    #[serde(default = "default_affine_mode")]
    pub mode: AffineQuantizationMode,
}

impl Default for AffineQuantization {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            mode: AffineQuantizationMode::Affine,
        }
    }
}

impl AffineQuantization {
    /// Creates and validates an affine quantization configuration.
    pub fn new(group_size: i32, bits: i32) -> Result<Self, Error> {
        let config = Self {
            group_size,
            bits,
            mode: AffineQuantizationMode::Affine,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates settings supported by MLX packed affine operations.
    pub fn validate(&self) -> Result<(), Error> {
        if self.mode != AffineQuantizationMode::Affine {
            return Err(Error::Quantization(
                "only MLX affine quantization is currently supported".into(),
            ));
        }
        if self.group_size != 16 && (self.group_size <= 0 || self.group_size % 32 != 0) {
            return Err(Error::Quantization(format!(
                "group_size must be 16 or a positive multiple of 32, got {}",
                self.group_size
            )));
        }
        if !matches!(self.bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Error::Quantization(format!(
                "bits must be one of 2, 3, 4, 5, 6, or 8, got {}",
                self.bits
            )));
        }
        Ok(())
    }
}

/// Resolves an on-load request against checkpoint quantization metadata.
///
/// Returns `true` for a dense checkpoint that must be quantized and `false`
/// when a matching pre-quantized checkpoint should be loaded directly.
pub(crate) fn should_quantize_on_load(
    architecture: &str,
    existing: Option<WeightQuantization>,
    requested: WeightQuantization,
) -> Result<bool, Error> {
    requested.validate()?;
    match existing {
        None => Ok(true),
        Some(existing) if existing == requested => Ok(false),
        Some(existing) => Err(Error::Quantization(format!(
            "{architecture} checkpoint is already quantized as {existing:?}, requested {requested:?}; implicit dequantization and requantization is unsupported"
        ))),
    }
}

/// Infers the exact affine layout emitted by the native GGUF converters.
#[cfg(test)]
pub(crate) fn gguf_affine_quantization(
    weight_shape: &[i32],
    scales_shape: &[i32],
    weight_name: &str,
) -> Result<AffineQuantization, Error> {
    let Some((&packed_columns, &scale_columns)) = weight_shape.last().zip(scales_shape.last())
    else {
        return Err(Error::Quantization(format!(
            "GGUF quantized tensor {weight_name:?} has an invalid rank"
        )));
    };
    if packed_columns <= 0 || scale_columns <= 0 {
        return Err(Error::Quantization(format!(
            "GGUF quantized tensor {weight_name:?} has incompatible weight/scales shapes {weight_shape:?} and {scales_shape:?}"
        )));
    }

    for (group_size, bits) in [(16, 2), (16, 3), (16, 6), (32, 4), (32, 5), (32, 8)] {
        if i64::from(packed_columns) * 32
            == i64::from(scale_columns) * i64::from(group_size) * i64::from(bits)
        {
            return AffineQuantization::new(group_size, bits);
        }
    }

    Err(Error::Quantization(format!(
        "GGUF quantized tensor {weight_name:?} has unsupported weight/scales shapes {weight_shape:?} and {scales_shape:?}"
    )))
}

fn default_affine_mode() -> AffineQuantizationMode {
    AffineQuantizationMode::Affine
}

/// Quantization mode serialized using MLX's lowercase spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AffineQuantizationMode {
    /// Per-group scale-and-bias affine quantization.
    Affine,
}

/// Weight encoding requested for load-time quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantization {
    /// MLX affine integer quantization.
    Affine(AffineQuantization),
    /// Microscaling FP4 with E2M1 values and E8M0 scales.
    MxFp4,
}

impl WeightQuantization {
    /// MXFP4 group size fixed by the format.
    pub const MXFP4_GROUP_SIZE: i32 = 32;
    /// MXFP4 packed value width fixed by the format.
    pub const MXFP4_BITS: i32 = 4;

    /// Returns the group size passed to MLX.
    pub const fn group_size(self) -> i32 {
        match self {
            Self::Affine(config) => config.group_size,
            Self::MxFp4 => Self::MXFP4_GROUP_SIZE,
        }
    }

    /// Returns the packed value width passed to MLX.
    pub const fn bits(self) -> i32 {
        match self {
            Self::Affine(config) => config.bits,
            Self::MxFp4 => Self::MXFP4_BITS,
        }
    }

    /// Returns the corresponding typed MLX execution mode.
    pub const fn mode(self) -> ops::QuantizationMode {
        match self {
            Self::Affine(_) => ops::QuantizationMode::Affine,
            Self::MxFp4 => ops::QuantizationMode::MxFp4,
        }
    }

    /// Returns whether the encoding stores affine quantization biases.
    pub const fn has_biases(self) -> bool {
        matches!(self, Self::Affine(_))
    }

    /// Validates the selected encoding.
    pub fn validate(self) -> Result<(), Error> {
        match self {
            Self::Affine(config) => config.validate(),
            Self::MxFp4 => Ok(()),
        }
    }
}

impl From<AffineQuantization> for WeightQuantization {
    fn from(value: AffineQuantization) -> Self {
        Self::Affine(value)
    }
}

#[derive(Serialize, Deserialize)]
struct WeightQuantizationMetadata {
    group_size: i32,
    bits: i32,
    #[serde(default = "default_weight_quantization_mode")]
    mode: String,
}

fn default_weight_quantization_mode() -> String {
    "affine".to_string()
}

impl Serialize for WeightQuantization {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WeightQuantizationMetadata {
            group_size: self.group_size(),
            bits: self.bits(),
            mode: match self {
                Self::Affine(_) => "affine",
                Self::MxFp4 => "mxfp4",
            }
            .into(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeightQuantization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let metadata = WeightQuantizationMetadata::deserialize(deserializer)?;
        match metadata.mode.as_str() {
            "affine" => AffineQuantization::new(metadata.group_size, metadata.bits)
                .map(Self::Affine)
                .map_err(de::Error::custom),
            "mxfp4"
                if metadata.group_size == Self::MXFP4_GROUP_SIZE
                    && metadata.bits == Self::MXFP4_BITS =>
            {
                Ok(Self::MxFp4)
            }
            "mxfp4" => Err(de::Error::custom(format!(
                "MXFP4 requires group_size=32 and bits=4, got group_size={} bits={}",
                metadata.group_size, metadata.bits
            ))),
            mode => Err(de::Error::custom(format!(
                "unsupported quantization mode {mode:?}"
            ))),
        }
    }
}

/// Tensor selection and output-sharding options for checkpoint conversion.
#[derive(Debug, Clone)]
pub struct CheckpointQuantizationOptions {
    /// Output weight encoding.
    pub quantization: WeightQuantization,
    /// Maximum uncompressed tensor bytes accumulated before writing a shard.
    pub shard_size_bytes: usize,
    /// Only quantize tensor names containing at least one of these strings.
    /// An empty list includes every otherwise eligible tensor.
    pub include: Vec<String>,
    /// Do not quantize tensor names containing any of these strings.
    pub exclude: Vec<String>,
    /// Skip otherwise eligible matrices smaller than this many elements.
    pub minimum_elements: usize,
}

impl Default for CheckpointQuantizationOptions {
    fn default() -> Self {
        Self {
            quantization: AffineQuantization::default().into(),
            shard_size_bytes: 512 * 1024 * 1024,
            include: Vec::new(),
            exclude: Vec::new(),
            minimum_elements: 0,
        }
    }
}

impl CheckpointQuantizationOptions {
    /// Returns whether a tensor is a matrix selected for affine quantization.
    pub fn selects(&self, name: &str, tensor: &Array) -> bool {
        canonical_weight_name(name).is_some()
            && tensor.ndim() == 2
            && tensor.dtype().is_float()
            && tensor.dim(1) % self.quantization.group_size() == 0
            && tensor.dim(1) % 32 == 0
            && tensor.size() >= self.minimum_elements
            && (self.include.is_empty() || self.include.iter().any(|needle| name.contains(needle)))
            && !self.exclude.iter().any(|needle| name.contains(needle))
    }

    fn validate(&self) -> Result<(), Error> {
        self.quantization.validate()?;
        if self.shard_size_bytes == 0 {
            return Err(Error::Quantization(
                "shard_size_bytes must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

fn canonical_weight_name(name: &str) -> Option<String> {
    if name.ends_with(".weight") {
        Some(name.to_string())
    } else {
        name.strip_suffix("_weight")
            .map(|prefix| format!("{prefix}.weight"))
    }
}

/// Tensors produced from one dense quantized matrix.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed unsigned-integer weights.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group affine biases.
    pub biases: Option<Array>,
}

impl QuantizedTensor {
    /// Returns the standard MLX-LM checkpoint keys and arrays for `weight_name`.
    pub fn into_named_arrays(self, weight_name: &str) -> Result<Vec<(String, Array)>, Error> {
        let prefix = weight_name.strip_suffix(".weight").ok_or_else(|| {
            Error::Quantization(format!(
                "quantized tensor name must end in .weight: {weight_name}"
            ))
        })?;
        let mut arrays = vec![
            (weight_name.to_string(), self.weight),
            (format!("{prefix}.scales"), self.scales),
        ];
        if let Some(biases) = self.biases {
            arrays.push((format!("{prefix}.biases"), biases));
        }
        Ok(arrays)
    }
}

/// Quantizes one floating-point weight using an explicit execution stream.
///
/// The last dimension is grouped and packed. Leading dimensions, including an
/// expert-bank dimension, are retained. Both on-the-fly model loading and
/// checkpoint conversion call this function.
pub fn quantize_tensor(
    weight: &Array,
    config: impl Into<WeightQuantization>,
    stream: &Stream,
) -> Result<QuantizedTensor, Error> {
    let config = config.into();
    config.validate()?;
    if weight.ndim() < 2 || !weight.dtype().is_float() {
        return Err(Error::Quantization(format!(
            "expected a floating-point weight with at least two dimensions, got shape {:?} and dtype {:?}",
            weight.shape(),
            weight.dtype()
        )));
    }
    let input_dims = weight.dim(-1);
    if input_dims % config.group_size() != 0 || input_dims % 32 != 0 {
        return Err(Error::Quantization(format!(
            "input dimension {} must be divisible by group_size {} and 32",
            input_dims,
            config.group_size()
        )));
    }
    let original_shape = weight.shape();
    let leading_size = weight.size() as i32 / input_dims;
    let matrix = if weight.ndim() == 2 {
        weight.clone()
    } else {
        weight.reshape(&[leading_size, input_dims], stream)?
    };
    let arrays = ops::quantize_with_mode(
        &matrix,
        config.group_size(),
        config.bits(),
        config.mode(),
        stream,
    )?;
    let restore_shape = |array: Array, last_dim: i32| -> Result<Array, Error> {
        if weight.ndim() == 2 {
            Ok(array)
        } else {
            let mut shape = original_shape[..original_shape.len() - 1].to_vec();
            shape.push(last_dim);
            Ok(array.reshape(&shape, stream)?)
        }
    };
    let packed_dims = ops::quantized_packed_dimension(input_dims, config.bits());
    let group_dims = input_dims / config.group_size();
    Ok(QuantizedTensor {
        weight: restore_shape(arrays.weight, packed_dims)?,
        scales: restore_shape(arrays.scales, group_dims)?,
        biases: arrays
            .biases
            .map(|biases| restore_shape(biases, group_dims))
            .transpose()?,
    })
}

/// Summary returned after converting and saving a checkpoint directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointQuantizationReport {
    /// Number of source matrices converted to packed affine tensors.
    pub quantized_tensors: usize,
    /// Number of source tensors copied without conversion.
    pub copied_tensors: usize,
    /// Number of output safetensors shards.
    pub shards: usize,
    /// Uncompressed bytes represented by all output tensors.
    pub total_size: usize,
}

struct PendingShard {
    arrays: HashMap<String, Array>,
    bytes: usize,
}

impl PendingShard {
    fn new() -> Self {
        Self {
            arrays: HashMap::new(),
            bytes: 0,
        }
    }

    fn insert(&mut self, name: String, array: Array) {
        self.bytes += array.nbytes();
        self.arrays.insert(name, array);
    }
}

/// Quantizes a model directory tensor-by-tensor and saves an MLX-LM-compatible checkpoint.
///
/// The source directory may contain a single `model.safetensors` file or a
/// Hugging Face sharded checkpoint index. Non-weight files are copied, while
/// `config.json` is updated with both standard MLX-LM quantization keys.
pub fn quantize_checkpoint(
    source_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &CheckpointQuantizationOptions,
    stream: &Stream,
) -> Result<CheckpointQuantizationReport, Error> {
    options.validate()?;
    let source_dir = source_dir.as_ref();
    let output_dir = output_dir.as_ref();
    if !source_dir.is_dir() {
        return Err(Error::Quantization(format!(
            "source is not a directory: {}",
            source_dir.display()
        )));
    }
    if let Some(existing) = checkpoint_quantization_metadata(source_dir)? {
        return if existing == options.quantization {
            Err(Error::Quantization(format!(
                "checkpoint already uses {existing:?}; reuse it instead of quantizing it again"
            )))
        } else {
            Err(Error::Quantization(format!(
                "checkpoint is already quantized as {existing:?}, requested {:?}; implicit transcoding is unsupported",
                options.quantization
            )))
        };
    }
    fs::create_dir(output_dir).map_err(|error| {
        Error::Quantization(format!(
            "could not create empty output directory {}: {error}",
            output_dir.display()
        ))
    })?;

    let result = quantize_checkpoint_inner(source_dir, output_dir, options, stream);
    if result.is_err() {
        // The directory was created by this call and contains only partial output.
        let _ = fs::remove_dir_all(output_dir);
    }
    result
}

fn checkpoint_quantization_metadata(
    source_dir: &Path,
) -> Result<Option<WeightQuantization>, Error> {
    let path = source_dir.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let config: Value = serde_json::from_slice(&fs::read(path)?)?;
    let Some(object) = config.as_object() else {
        return Err(Error::Quantization(
            "config.json must contain a JSON object".into(),
        ));
    };
    for key in ["quantization", "quantization_config"] {
        let Some(value) = object.get(key).filter(|value| !value.is_null()) else {
            continue;
        };
        if value.get("mode").is_some() {
            return serde_json::from_value(value.clone())
                .map(Some)
                .map_err(|error| Error::Quantization(format!("invalid {key} metadata: {error}")));
        }
        return Err(Error::Quantization(format!(
            "checkpoint contains prequantized {key} metadata that is not MLX affine/MXFP4; implicit transcoding is unsupported"
        )));
    }
    Ok(None)
}

fn quantize_checkpoint_inner(
    source_dir: &Path,
    output_dir: &Path,
    options: &CheckpointQuantizationOptions,
    stream: &Stream,
) -> Result<CheckpointQuantizationReport, Error> {
    let weight_files = weights::safetensors_files(source_dir)?;
    copy_checkpoint_assets(source_dir, output_dir, &weight_files)?;
    write_quantized_config(source_dir, output_dir, options.quantization)?;

    let mut pending = PendingShard::new();
    let mut temporary_shards = Vec::new();
    let mut locations = BTreeMap::<String, usize>::new();
    let mut quantized_tensors = 0;
    let mut copied_tensors = 0;
    let mut total_size = 0;

    for file in weight_files {
        weights::for_each_safetensor_array(file, stream, |name, tensor| {
            let arrays = if options.selects(&name, &tensor) {
                quantized_tensors += 1;
                let weight_name = canonical_weight_name(&name).expect("selected weight name");
                quantize_tensor(&tensor, options.quantization, stream)?
                    .into_named_arrays(&weight_name)?
            } else {
                copied_tensors += 1;
                vec![(name, tensor)]
            };

            let incoming_bytes = arrays
                .iter()
                .map(|(_, array)| array.nbytes())
                .sum::<usize>();
            if !pending.arrays.is_empty()
                && pending.bytes.saturating_add(incoming_bytes) > options.shard_size_bytes
            {
                flush_temporary_shard(
                    output_dir,
                    &mut pending,
                    &mut temporary_shards,
                    &mut locations,
                )?;
            }
            for (name, array) in arrays {
                total_size += array.nbytes();
                pending.insert(name, array);
            }
            Ok(())
        })?;
    }
    if !pending.arrays.is_empty() {
        flush_temporary_shard(
            output_dir,
            &mut pending,
            &mut temporary_shards,
            &mut locations,
        )?;
    }
    if temporary_shards.is_empty() {
        return Err(Error::Quantization("checkpoint contains no tensors".into()));
    }

    finalize_shards(output_dir, &temporary_shards, &locations, total_size)?;
    Ok(CheckpointQuantizationReport {
        quantized_tensors,
        copied_tensors,
        shards: temporary_shards.len(),
        total_size,
    })
}

fn flush_temporary_shard(
    output_dir: &Path,
    pending: &mut PendingShard,
    temporary_shards: &mut Vec<PathBuf>,
    locations: &mut BTreeMap<String, usize>,
) -> Result<(), Error> {
    let shard_index = temporary_shards.len();
    let path = output_dir.join(format!(".quantized-{shard_index:05}.safetensors"));
    Array::save_safetensors(pending.arrays.iter(), None, &path)?;
    for name in pending.arrays.keys() {
        locations.insert(name.clone(), shard_index);
    }
    pending.arrays.clear();
    pending.bytes = 0;
    temporary_shards.push(path);
    Ok(())
}

fn finalize_shards(
    output_dir: &Path,
    temporary_shards: &[PathBuf],
    locations: &BTreeMap<String, usize>,
    total_size: usize,
) -> Result<(), Error> {
    if temporary_shards.len() == 1 {
        fs::rename(&temporary_shards[0], output_dir.join("model.safetensors"))?;
        return Ok(());
    }

    let count = temporary_shards.len();
    let mut shard_names = Vec::with_capacity(count);
    for (index, temporary) in temporary_shards.iter().enumerate() {
        let name = format!("model-{:05}-of-{count:05}.safetensors", index + 1);
        fs::rename(temporary, output_dir.join(&name))?;
        shard_names.push(name);
    }
    let weight_map = locations
        .iter()
        .map(|(name, index)| (name.clone(), Value::String(shard_names[*index].clone())))
        .collect::<serde_json::Map<_, _>>();
    let index = json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(())
}

fn copy_checkpoint_assets(
    source_dir: &Path,
    output_dir: &Path,
    weight_files: &[PathBuf],
) -> Result<(), Error> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        let file_name = entry.file_name();
        let file_name_lossy = file_name.to_string_lossy();
        if file_name_lossy == "config.json"
            || file_name_lossy == "model.safetensors.index.json"
            || weight_files.iter().any(|weight_file| weight_file == &path)
        {
            continue;
        }
        fs::copy(path, output_dir.join(file_name))?;
    }
    Ok(())
}

fn write_quantized_config(
    source_dir: &Path,
    output_dir: &Path,
    quantization: WeightQuantization,
) -> Result<(), Error> {
    let path = source_dir.join("config.json");
    let mut config = if path.exists() {
        serde_json::from_slice::<Value>(&fs::read(path)?)?
    } else {
        json!({})
    };
    let object = config
        .as_object_mut()
        .ok_or_else(|| Error::Quantization("config.json must contain a JSON object".into()))?;
    let value = serde_json::to_value(quantization)?;
    object.insert("quantization".into(), value.clone());
    object.insert("quantization_config".into(), value);
    if object.contains_key("moshi_name") {
        object.insert(
            "moshi_name".into(),
            Value::String("model.safetensors".into()),
        );
    }
    fs::write(
        output_dir.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use safemlx::{Device, DeviceType, Dtype, ExecutionContext};

    use super::*;

    #[test]
    fn affine_config_uses_mlx_spelling() {
        let value = serde_json::to_value(AffineQuantization::default()).unwrap();
        assert_eq!(value["group_size"], 64);
        assert_eq!(value["bits"], 4);
        assert_eq!(value["mode"], "affine");
    }

    #[test]
    fn mxfp4_metadata_is_fixed_and_round_trips() {
        let value = serde_json::to_value(WeightQuantization::MxFp4).unwrap();
        assert_eq!(value, json!({"group_size": 32, "bits": 4, "mode": "mxfp4"}));
        assert_eq!(
            serde_json::from_value::<WeightQuantization>(value).unwrap(),
            WeightQuantization::MxFp4
        );
        assert!(serde_json::from_value::<WeightQuantization>(
            json!({"group_size": 64, "bits": 4, "mode": "mxfp4"})
        )
        .is_err());
        assert!(serde_json::from_value::<WeightQuantization>(
            json!({"group_size": 32, "bits": 8, "mode": "mxfp4"})
        )
        .is_err());
    }

    #[test]
    fn omitted_quantization_mode_defaults_to_affine() {
        let quantization =
            serde_json::from_value::<WeightQuantization>(json!({"group_size": 64, "bits": 4}))
                .unwrap();
        assert_eq!(
            quantization,
            WeightQuantization::Affine(AffineQuantization::new(64, 4).unwrap())
        );
    }

    #[test]
    fn checkpoint_conversion_rejects_prequantized_inputs() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "safemlx-prequantized-rejection-{}-{suffix}",
            std::process::id()
        ));
        let source = root.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join("config.json"),
            serde_json::to_vec(&json!({
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
            }))
            .unwrap(),
        )
        .unwrap();
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let options = CheckpointQuantizationOptions {
            quantization: WeightQuantization::MxFp4,
            ..Default::default()
        };
        let error = quantize_checkpoint(&source, root.join("output"), &options, context.stream())
            .unwrap_err();
        assert!(error.to_string().contains("implicit transcoding"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn on_load_resolution_reuses_matching_metadata_and_rejects_mismatch() {
        let q4 = AffineQuantization::default();
        assert!(should_quantize_on_load("test", None, q4.into()).unwrap());
        assert!(!should_quantize_on_load("test", Some(q4.into()), q4.into()).unwrap());

        let q8 = AffineQuantization::new(64, 8).unwrap();
        let error = should_quantize_on_load("test", Some(q4.into()), q8.into()).unwrap_err();
        assert!(error.to_string().contains("already quantized"));
        assert!(error.to_string().contains("implicit dequantization"));
        assert!(!should_quantize_on_load(
            "test",
            Some(WeightQuantization::MxFp4),
            WeightQuantization::MxFp4
        )
        .unwrap());
    }

    #[test]
    fn affine_config_accepts_mlx_non_power_of_two_widths() {
        assert!(AffineQuantization::new(32, 3).is_ok());
        assert!(AffineQuantization::new(32, 5).is_ok());
        assert!(AffineQuantization::new(32, 6).is_ok());
        assert!(AffineQuantization::new(32, 7).is_err());
    }

    #[test]
    fn selection_is_model_agnostic_and_filterable() {
        let tensor = Array::from_slice::<f32>(&vec![0.0; 128 * 64], &[128, 64]);
        let vector = Array::from_slice::<f32>(&vec![0.0; 8192], &[8192]);
        let mut options = CheckpointQuantizationOptions::default();
        assert!(options.selects("anything.proj.weight", &tensor));
        assert!(!options.selects("anything.norm.weight", &vector));
        options.exclude.push("proj".into());
        assert!(!options.selects("anything.proj.weight", &tensor));
    }

    #[test]
    fn mxfp4_quantizes_rank_three_expert_banks() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let experts = Array::from_slice(&vec![0.25f32; 3 * 8 * 64], &[3, 8, 64]);
        let quantized =
            quantize_tensor(&experts, WeightQuantization::MxFp4, context.stream()).unwrap();
        assert_eq!(quantized.weight.shape(), &[3, 8, 8]);
        assert_eq!(quantized.scales.shape(), &[3, 8, 2]);
        assert!(quantized.biases.is_none());
    }

    #[test]
    fn saved_mxfp4_checkpoint_has_no_affine_bias_tensors() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "safemlx-mxfp4-save-test-{}-{suffix}",
            std::process::id()
        ));
        let source = root.join("source");
        let output = root.join("output");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), br#"{"model_type":"test"}"#).unwrap();
        let weight = Array::from_slice(&vec![0.25f32; 2 * 64], &[2, 64]);
        Array::save_safetensors(
            [("model.proj.weight", &weight)],
            None,
            source.join("model.safetensors"),
        )
        .unwrap();

        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let options = CheckpointQuantizationOptions {
            quantization: WeightQuantization::MxFp4,
            ..Default::default()
        };
        quantize_checkpoint(&source, &output, &options, stream).unwrap();
        let arrays =
            Array::load_safetensors(output.join("model.safetensors"), weights_context.stream())
                .unwrap();
        assert!(arrays.contains_key("model.proj.weight"));
        assert!(arrays.contains_key("model.proj.scales"));
        assert!(!arrays.contains_key("model.proj.biases"));
        let config: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(config["quantization"], config["quantization_config"]);
        assert_eq!(config["quantization"]["mode"], "mxfp4");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn saved_checkpoint_matches_direct_tensor_quantization() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "safemlx-quantization-test-{}-{suffix}",
            std::process::id()
        ));
        let source = root.join("source");
        let output = root.join("output");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), br#"{"model_type":"test"}"#).unwrap();

        let values = (0..(8 * 64))
            .map(|index| (index as f32 - 255.5) / 64.0)
            .collect::<Vec<_>>();
        let weight = Array::from_slice(&values, &[8, 64]);
        let norm = Array::from_slice(&vec![1.0f32; 64], &[64]);
        Array::save_safetensors(
            [("model.proj.weight", &weight), ("model.norm.weight", &norm)],
            None,
            source.join("model.safetensors"),
        )
        .unwrap();
        Array::save_safetensors(
            [("auxiliary.weight", &norm)],
            None,
            source.join("auxiliary.safetensors"),
        )
        .unwrap();

        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights_stream = weights_context.stream();
        let expected = quantize_tensor(&weight, AffineQuantization::default(), stream).unwrap();
        let options = CheckpointQuantizationOptions {
            shard_size_bytes: 1,
            ..Default::default()
        };
        let report = quantize_checkpoint(&source, &output, &options, stream).unwrap();
        assert_eq!(report.quantized_tensors, 1);
        assert_eq!(report.copied_tensors, 1);
        assert_eq!(report.shards, 2);

        let mut saved = HashMap::new();
        for file in weights::safetensors_files(&output).unwrap() {
            saved.extend(Array::load_safetensors(file, weights_stream).unwrap());
        }
        let saved_weight = &saved["model.proj.weight"];
        assert_eq!(saved_weight.dtype(), Dtype::Uint32);
        assert_eq!(
            saved_weight.evaluated().unwrap().as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            saved["model.proj.scales"]
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            expected.scales.evaluated().unwrap().as_slice::<f32>()
        );
        assert_eq!(
            saved["model.proj.biases"]
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            expected
                .biases
                .as_ref()
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
        );
        assert_eq!(
            saved["model.norm.weight"]
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            norm.evaluated().unwrap().as_slice::<f32>()
        );

        let config: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(config["quantization"], config["quantization_config"]);
        assert_eq!(config["quantization"]["mode"], "affine");
        assert!(output.join("auxiliary.safetensors").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
