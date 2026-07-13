//! Model-agnostic affine quantization for safetensors checkpoints.
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
use serde::{Deserialize, Serialize};
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
        if self.group_size <= 0 || self.group_size % 32 != 0 {
            return Err(Error::Quantization(format!(
                "group_size must be a positive multiple of 32, got {}",
                self.group_size
            )));
        }
        if !matches!(self.bits, 2 | 4 | 8) {
            return Err(Error::Quantization(format!(
                "bits must be one of 2, 4, or 8, got {}",
                self.bits
            )));
        }
        Ok(())
    }
}

/// Resolves an affine on-load request against checkpoint metadata.
///
/// Returns `true` for a dense checkpoint that must be quantized and `false`
/// when a matching pre-quantized checkpoint should be loaded directly.
pub(crate) fn should_quantize_on_load(
    architecture: &str,
    existing: Option<AffineQuantization>,
    requested: AffineQuantization,
) -> Result<bool, Error> {
    requested.validate()?;
    match existing {
        None => Ok(true),
        Some(existing) if existing == requested => Ok(false),
        Some(existing) => Err(Error::Quantization(format!(
            "{architecture} checkpoint is already affine-quantized as group_size={} bits={}, requested group_size={} bits={}",
            existing.group_size, existing.bits, requested.group_size, requested.bits
        ))),
    }
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

/// Tensor selection and output-sharding options for checkpoint conversion.
#[derive(Debug, Clone)]
pub struct CheckpointQuantizationOptions {
    /// Affine format settings.
    pub quantization: AffineQuantization,
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
            quantization: AffineQuantization::default(),
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
            && tensor.dim(1) % self.quantization.group_size == 0
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

/// The three tensors produced from one dense affine-quantized matrix.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed unsigned-integer weights.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group affine biases.
    pub biases: Array,
}

impl QuantizedTensor {
    /// Returns the standard MLX-LM checkpoint keys and arrays for `weight_name`.
    pub fn into_named_arrays(self, weight_name: &str) -> Result<[(String, Array); 3], Error> {
        let prefix = weight_name.strip_suffix(".weight").ok_or_else(|| {
            Error::Quantization(format!(
                "quantized tensor name must end in .weight: {weight_name}"
            ))
        })?;
        Ok([
            (weight_name.to_string(), self.weight),
            (format!("{prefix}.scales"), self.scales),
            (format!("{prefix}.biases"), self.biases),
        ])
    }
}

/// Quantizes one two-dimensional weight using an explicit execution stream.
///
/// Both on-the-fly model loading and checkpoint conversion call this function.
pub fn quantize_tensor(
    weight: &Array,
    config: AffineQuantization,
    stream: &Stream,
) -> Result<QuantizedTensor, Error> {
    config.validate()?;
    if weight.ndim() != 2 || !weight.dtype().is_float() {
        return Err(Error::Quantization(format!(
            "expected a floating-point matrix, got shape {:?} and dtype {:?}",
            weight.shape(),
            weight.dtype()
        )));
    }
    if weight.dim(1) % config.group_size != 0 || weight.dim(1) % 32 != 0 {
        return Err(Error::Quantization(format!(
            "input dimension {} must be divisible by group_size {} and 32",
            weight.dim(1),
            config.group_size
        )));
    }
    let (weight, scales, biases) = ops::quantize(weight, config.group_size, config.bits, stream)?;
    Ok(QuantizedTensor {
        weight,
        scales,
        biases,
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
                    .into_iter()
                    .collect::<Vec<_>>()
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
    quantization: AffineQuantization,
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
    fn on_load_resolution_reuses_matching_metadata_and_rejects_mismatch() {
        let q4 = AffineQuantization::default();
        assert!(should_quantize_on_load("test", None, q4).unwrap());
        assert!(!should_quantize_on_load("test", Some(q4), q4).unwrap());

        let q8 = AffineQuantization::new(64, 8).unwrap();
        let error = should_quantize_on_load("test", Some(q4), q8).unwrap_err();
        assert!(error.to_string().contains("already affine-quantized"));
        assert!(error.to_string().contains("requested group_size=64 bits=8"));
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
            expected.biases.evaluated().unwrap().as_slice::<f32>()
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
