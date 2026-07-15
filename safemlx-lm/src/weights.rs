use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;
use safemlx::{
    module::{FlattenedModuleParamMut, ModuleParameters},
    ops::{concatenate_axis, stack_axis},
    transforms::eval,
    Array, Stream,
};
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::error::Error;
use crate::quantization::{quantize_tensor, WeightQuantization};

/// Options for strict checkpoint loading.
///
/// This configuration controls how checkpoint tensor names are matched to model
/// parameters and which missing or unused names are accepted.
#[derive(Debug, Clone, Default)]
pub struct StrictLoadConfig {
    allowed_unused_prefixes: Vec<String>,
    allowed_missing_suffixes: Vec<String>,
    allowed_missing_contains: Vec<String>,
    key_prefixes_to_strip: Vec<String>,
    key_prefix_rewrites: Vec<(String, String)>,
}

impl StrictLoadConfig {
    /// Allows unused checkpoint tensors whose names start with `prefix`.
    pub fn allow_unused_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.allowed_unused_prefixes.push(prefix.into());
        self
    }

    /// Allows missing model parameters whose names end with `suffix`.
    pub fn allow_missing_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.allowed_missing_suffixes.push(suffix.into());
        self
    }

    /// Allows missing model parameters whose names contain `needle`.
    pub fn allow_missing_contains(mut self, needle: impl Into<String>) -> Self {
        self.allowed_missing_contains.push(needle.into());
        self
    }

    /// Adds a candidate key with `prefix` stripped from checkpoint tensor names.
    pub fn strip_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefixes_to_strip.push(prefix.into());
        self
    }

    /// Rewrites a checkpoint key prefix before matching it to model parameters.
    pub fn rewrite_prefix(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.key_prefix_rewrites.push((from.into(), to.into()));
        self
    }

    fn is_unused_allowed(&self, key: &str) -> bool {
        self.allowed_unused_prefixes
            .iter()
            .any(|prefix| key.starts_with(prefix))
    }

    fn is_missing_allowed(&self, key: &str) -> bool {
        self.allowed_missing_suffixes
            .iter()
            .any(|suffix| key.ends_with(suffix))
            || self
                .allowed_missing_contains
                .iter()
                .any(|needle| key.contains(needle))
    }

    pub(crate) fn candidates(&self, key: &str) -> Vec<String> {
        let mut candidates = Vec::new();
        candidates.push(key.to_string());

        for prefix in &self.key_prefixes_to_strip {
            if let Some(stripped) = key.strip_prefix(prefix) {
                candidates.push(stripped.to_string());
            }
        }

        for (from, to) in &self.key_prefix_rewrites {
            if let Some(stripped) = key.strip_prefix(from) {
                candidates.push(format!("{to}{stripped}"));
            }
        }

        let mut expanded = Vec::with_capacity(candidates.len() * 2);
        for candidate in candidates {
            expanded.push(candidate.clone());
            if let Some(inner_key) = candidate.strip_suffix(".weight") {
                expanded.push(format!("{inner_key}.inner.weight"));
            }
            if let Some(inner_key) = candidate.strip_suffix(".bias") {
                expanded.push(format!("{inner_key}.inner.bias"));
            }
            if let Some(rest) = candidate.strip_prefix("model.language_model.embed_tokens.") {
                expanded.push(format!("model.language_model.embed_tokens.inner.{rest}"));
            }
            if let Some(rest) =
                candidate.strip_prefix("model.language_model.embed_tokens_per_layer.")
            {
                expanded.push(format!(
                    "model.language_model.embed_tokens_per_layer.inner.{rest}"
                ));
            }
        }

        let mut seen = HashSet::new();
        expanded
            .into_iter()
            .filter(|candidate| seen.insert(candidate.clone()))
            .collect()
    }
}

/// Accumulates strict checkpoint-loading diagnostics across one or more files.
#[derive(Debug, Clone, Default)]
pub struct StrictLoadReport {
    loaded: HashSet<String>,
    unused: Vec<String>,
    shape_mismatches: Vec<String>,
}

impl StrictLoadReport {
    pub(crate) fn record_loaded(&mut self, key: String) {
        self.loaded.insert(key);
    }

    pub(crate) fn record_unused(&mut self, key: String) {
        self.unused.push(key);
    }

    pub(crate) fn record_shape_mismatch(
        &mut self,
        weight_key: String,
        param_key: String,
        expected_shape: Vec<i32>,
        actual_shape: Vec<i32>,
    ) {
        self.shape_mismatches.push(format!(
            "{weight_key} -> {param_key}: expected {expected_shape:?}, got {actual_shape:?}"
        ));
    }

    /// Validates the report against the model parameters and load configuration.
    pub fn finish<M: ModuleParameters>(
        self,
        model: &M,
        config: &StrictLoadConfig,
    ) -> Result<(), Error> {
        let mut missing = model
            .parameters()
            .flatten()
            .keys()
            .map(|key| key.to_string())
            .filter(|key| !self.loaded.contains(key))
            .filter(|key| !config.is_missing_allowed(key))
            .collect::<Vec<_>>();

        let mut unused = self
            .unused
            .into_iter()
            .filter(|key| !config.is_unused_allowed(key))
            .collect::<Vec<_>>();
        unused.extend(self.shape_mismatches);

        missing.sort();
        unused.sort();

        if missing.is_empty() && unused.is_empty() {
            Ok(())
        } else {
            Err(Error::StrictLoadValidation { missing, unused })
        }
    }
}

/// Loads a safetensors file into `model` and records strict-loading diagnostics.
pub fn load_safetensors_strict<M: ModuleParameters>(
    model: &mut M,
    path: impl AsRef<Path>,
    stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let mut params = model.parameters_mut().flatten();

    for_each_safetensor_array(path, stream, |key, value| {
        load_array_strict(&mut params, key, value, config, report);
        Ok(())
    })
}

pub(crate) fn load_arrays_strict<M: ModuleParameters>(
    model: &mut M,
    loaded: HashMap<String, Array>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let mut params = model.parameters_mut().flatten();

    for (key, value) in loaded {
        load_array_strict(&mut params, key, value, config, report);
    }

    Ok(())
}

/// Strict-loads and quantizes eligible tensors from an in-memory named-array
/// source such as an unquantized GGUF. The map is consumed so each dense source
/// array can be released after its packed replacement is materialized.
pub(crate) fn load_arrays_quantized_strict<M: ModuleParameters>(
    model: &mut M,
    loaded: HashMap<String, Array>,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    quantization.validate()?;
    let mut params = model.parameters_mut().flatten();
    for (key, value) in loaded {
        load_array_quantized_strict(
            &mut params,
            key,
            value,
            quantization_stream,
            quantization,
            config,
            report,
        )?;
    }
    Ok(())
}

pub(crate) fn for_each_safetensor_array<F>(
    path: impl AsRef<Path>,
    stream: &Stream,
    mut f: F,
) -> Result<(), Error>
where
    F: FnMut(String, Array) -> Result<(), Error>,
{
    let file = File::open(path)?;
    // The mmap only has to live until each TensorView is copied into an MLX-owned Array.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| Error::Other(Box::new(err)))?;

    for (key, view) in tensors.iter() {
        let value = Array::try_from(view).map_err(|err| Error::Other(Box::new(err)))?;
        let value = value.copy(stream)?;
        f(key.to_string(), value)?;
    }

    Ok(())
}

pub(crate) fn load_array_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    key: String,
    value: Array,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) {
    let mut matched = None;
    for candidate in config.candidates(&key) {
        if params.contains_key(candidate.as_str()) {
            matched = Some(candidate);
            break;
        }
    }

    if let Some(candidate) = matched {
        if let Some(param) = params.get_mut(candidate.as_str()) {
            let expected_shape = param.shape().to_vec();
            let actual_shape = value.shape().to_vec();
            if expected_shape == actual_shape {
                **param = value;
                report.record_loaded(candidate);
            } else {
                report.record_shape_mismatch(key, candidate, expected_shape, actual_shape);
            }
        }
    } else {
        report.record_unused(key);
    }
}

/// Loads a safetensors file into matching model parameters without strict validation.
///
/// This preserves the behavior of `ModuleParametersExt::load_safetensors`, but streams tensors
/// from a mmap instead of materializing the whole checkpoint file as a `HashMap`.
pub fn load_safetensors_lenient<M: ModuleParameters>(
    model: &mut M,
    path: impl AsRef<Path>,
    stream: &Stream,
) -> Result<(), Error> {
    let mut params = model.parameters_mut().flatten();
    let config = StrictLoadConfig::default();
    for_each_safetensor_array(path, stream, |key, value| {
        for candidate in config.candidates(&key) {
            if let Some(param) = params.get_mut(candidate.as_str()) {
                **param = value;
                break;
            }
        }
        Ok(())
    })
}

/// Strict-loads a dense safetensors file into a model whose selected parameters
/// use the standard MLX affine quantized layout.
///
/// Dense matrices are quantized and materialized one at a time as they are
/// read, bounding the lazy graph and active allocation peak. A target module is
/// recognized either by the standard safemlx `inner.weight` parameter plus
/// sibling `scales`/`biases`, or by a packed `weight` with those siblings.
pub fn load_safetensors_quantized_strict<M: ModuleParameters>(
    model: &mut M,
    path: impl AsRef<Path>,
    weights_stream: &Stream,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    quantization.validate()?;
    let mut params = model.parameters_mut().flatten();
    for_each_safetensor_array(path, weights_stream, |key, value| {
        load_array_quantized_strict(
            &mut params,
            key,
            value,
            quantization_stream,
            quantization,
            config,
            report,
        )
    })
}

pub(crate) fn load_array_quantized_strict(
    params: &mut FlattenedModuleParamMut<'_>,
    key: String,
    value: Array,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    {
        let target = config.candidates(&key).into_iter().find_map(|candidate| {
            let (prefix, weight_key, underscore_companions) =
                if let Some(prefix) = candidate.strip_suffix(".inner.weight") {
                    (prefix.to_string(), candidate, false)
                } else if let Some(prefix) = candidate.strip_suffix(".weight") {
                    (prefix.to_string(), candidate, false)
                } else {
                    (candidate.clone(), candidate, true)
                };
            let scales_key = if underscore_companions {
                format!("{prefix}_scales")
            } else {
                format!("{prefix}.scales")
            };
            let biases_key = if underscore_companions {
                format!("{prefix}_biases")
            } else {
                format!("{prefix}.biases")
            };
            let has_quantized_parameters = params.contains_key(weight_key.as_str())
                && params.contains_key(scales_key.as_str())
                && (!quantization.has_biases() || params.contains_key(biases_key.as_str()));
            let packed_direct_weight = !weight_key.ends_with(".inner.weight")
                && params
                    .get(weight_key.as_str())
                    .is_some_and(|target| target.shape() != value.shape());
            (has_quantized_parameters
                && (weight_key.ends_with(".inner.weight") || packed_direct_weight))
                .then_some((weight_key, scales_key, biases_key))
        });

        if let Some((weight_key, scales_key, biases_key)) = target {
            let quantized = quantize_tensor(&value, quantization, quantization_stream)?;
            // MLX quantization is lazy. Materialize this tensor before the
            // source value leaves the streaming callback so subsequent
            // weights do not accumulate a checkpoint-sized dense graph.
            let mut arrays = vec![&quantized.weight, &quantized.scales];
            if let Some(biases) = &quantized.biases {
                arrays.push(biases);
            }
            eval(arrays)?;
            quantization_stream.synchronize()?;
            load_array_strict(params, weight_key, quantized.weight, config, report);
            load_array_strict(params, scales_key, quantized.scales, config, report);
            if let Some(biases) = quantized.biases {
                load_array_strict(params, biases_key, biases, config, report);
            }
            return Ok(());
        }
    }
    load_array_strict(params, key, value, config, report);
    Ok(())
}

/// Strict-loads and quantizes every safetensors shard in a model directory.
pub fn load_safetensors_dir_quantized_strict<M: ModuleParameters>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    quantization_stream: &Stream,
    quantization: WeightQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    for file in safetensors_files(model_dir)? {
        load_safetensors_quantized_strict(
            model,
            file,
            weights_stream,
            quantization_stream,
            quantization,
            config,
            report,
        )?;
    }
    Ok(())
}

/// Loads all safetensors files from `model_dir` into matching parameters without validation.
pub fn load_safetensors_dir_lenient<M: ModuleParameters>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    stream: &Stream,
) -> Result<(), Error> {
    for file in safetensors_files(model_dir)? {
        load_safetensors_lenient(model, file, stream)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
/// Hugging Face safetensors index file.
pub struct WeightMap {
    /// Index metadata.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Mapping from tensor name to shard file name.
    pub weight_map: HashMap<String, String>,
}

/// Returns the safetensors files referenced by a Hugging Face model directory.
pub fn safetensors_files(model_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, Error> {
    let model_dir = model_dir.as_ref();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let mut files = weight_map
            .weight_map
            .values()
            .map(|file| model_dir.join(file))
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        return Ok(files);
    }

    Ok(vec![model_dir.join("model.safetensors")])
}

/// Loads all safetensors files from `model_dir` into `model` with strict validation.
pub fn load_safetensors_dir_strict<M: ModuleParameters>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    for file in safetensors_files(model_dir)? {
        load_safetensors_strict(model, file, stream, config, report)?;
    }
    Ok(())
}

/// Loads and merges all safetensors files, transforms them, then strict-loads the result.
///
/// This is useful when a checkpoint stores split per-expert tensors across shards but the runtime
/// module owns packed expert banks.
pub fn load_safetensors_dir_merged_strict_with_transform<M, F>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    transform: F,
) -> Result<(), Error>
where
    M: ModuleParameters,
    F: FnOnce(HashMap<String, Array>, &Stream) -> Result<HashMap<String, Array>, Error>,
{
    let mut loaded = HashMap::new();
    for file in safetensors_files(model_dir)? {
        loaded.extend(Array::load_safetensors(file, weights_stream)?);
    }
    let loaded = transform(loaded, transform_stream)?;
    load_arrays_strict(model, loaded, config, report)
}

/// Strict-loads a model directory while streaming and packing split ReLU2 experts.
pub fn load_safetensors_dir_strict_with_split_relu2_experts<M, F>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    num_experts: i32,
    rewrite_key: F,
) -> Result<(), Error>
where
    M: ModuleParameters,
    F: Fn(&str) -> Result<String, Error>,
{
    let mut expert_parts: HashMap<(String, i32), Relu2ExpertParts> = HashMap::new();
    let mut params = model.parameters_mut().flatten();

    for file in safetensors_files(model_dir)? {
        for_each_safetensor_array(file, weights_stream, |key, value| {
            let key = rewrite_key(&key)?;
            if let Some((prefix, expert, projection)) =
                parse_split_relu2_expert_projection_key(&key)
            {
                let parts = expert_parts.entry((prefix, expert)).or_default();
                match projection {
                    Relu2ExpertProjection::Up => parts.up = Some(value),
                    Relu2ExpertProjection::Down => parts.down = Some(value),
                }
            } else {
                load_array_strict(&mut params, key, value, config, report);
            }
            Ok(())
        })?;

        let mut complete_prefixes = expert_parts
            .keys()
            .map(|(prefix, _)| prefix.clone())
            .collect::<Vec<_>>();
        complete_prefixes.sort();
        complete_prefixes.dedup();
        for prefix in complete_prefixes {
            if split_relu2_expert_prefix_complete(&expert_parts, &prefix, num_experts) {
                let packed = pack_split_relu2_expert_prefix(
                    &mut expert_parts,
                    &prefix,
                    num_experts,
                    transform_stream,
                )?;
                for (key, value) in packed {
                    load_array_strict(&mut params, key, value, config, report);
                }
            }
        }
    }

    if let Some((prefix, _)) = expert_parts.keys().next().cloned() {
        pack_split_relu2_expert_prefix(&mut expert_parts, &prefix, num_experts, transform_stream)?;
    }

    Ok(())
}

/// Strict-loads a model directory while streaming and packing split SwiGLU experts.
///
/// Public checkpoints commonly store `w1`, `w2`, and `w3` per expert, while the
/// runtime uses one expert-major gate/up bank plus a down bank. Completed layer
/// banks are loaded immediately so all expert layers are never resident at once.
#[allow(clippy::too_many_arguments)]
pub fn load_safetensors_dir_strict_with_split_swiglu_experts<M>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    quantization: Option<WeightQuantization>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    num_experts: i32,
) -> Result<(), Error>
where
    M: ModuleParameters,
{
    load_safetensors_dir_strict_with_split_swiglu_experts_and_transform(
        model,
        model_dir,
        weights_stream,
        transform_stream,
        quantization,
        config,
        report,
        num_experts,
        |key, value| Ok(vec![(key, value)]),
    )
}

/// Strict-loads and packs split SwiGLU experts after applying a streaming key/value transform.
///
/// The transform can split or rewrite architecture-specific tensors before
/// expert detection and strict parameter matching without buffering a shard.
#[allow(clippy::too_many_arguments)]
pub fn load_safetensors_dir_strict_with_split_swiglu_experts_and_transform<M, F>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    quantization: Option<WeightQuantization>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    num_experts: i32,
    transform: F,
) -> Result<(), Error>
where
    M: ModuleParameters,
    F: Fn(String, Array) -> Result<Vec<(String, Array)>, Error>,
{
    if let Some(quantization) = quantization {
        quantization.validate()?;
    }
    let mut expert_parts: HashMap<(String, i32), SwiGluExpertParts> = HashMap::new();
    let mut params = model.parameters_mut().flatten();

    for file in safetensors_files(model_dir)? {
        for_each_safetensor_array(file, weights_stream, |key, value| {
            for (key, value) in transform(key, value)? {
                if let Some((prefix, expert, projection)) =
                    parse_split_swiglu_expert_projection_key(&key)
                {
                    {
                        let parts = expert_parts.entry((prefix.clone(), expert)).or_default();
                        match projection {
                            SwiGluExpertProjection::Gate => parts.gate = Some(value),
                            SwiGluExpertProjection::Down => parts.down = Some(value),
                            SwiGluExpertProjection::Up => parts.up = Some(value),
                        }
                    }
                    if split_swiglu_expert_prefix_complete(&expert_parts, &prefix, num_experts) {
                        for (key, value) in pack_split_swiglu_expert_prefix(
                            &mut expert_parts,
                            &prefix,
                            num_experts,
                            transform_stream,
                        )? {
                            if let Some(quantization) = quantization {
                                load_array_quantized_strict(
                                    &mut params,
                                    key,
                                    value,
                                    transform_stream,
                                    quantization,
                                    config,
                                    report,
                                )?;
                            } else {
                                load_array_strict(&mut params, key, value, config, report);
                            }
                        }
                    }
                } else if let Some(quantization) = quantization {
                    load_array_quantized_strict(
                        &mut params,
                        key,
                        value,
                        transform_stream,
                        quantization,
                        config,
                        report,
                    )?;
                } else {
                    load_array_strict(&mut params, key, value, config, report);
                }
            }
            Ok(())
        })?;
    }

    if let Some((prefix, _)) = expert_parts.keys().next().cloned() {
        pack_split_swiglu_expert_prefix(&mut expert_parts, &prefix, num_experts, transform_stream)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Projection kind in a split SwiGLU expert checkpoint.
pub enum SwiGluExpertProjection {
    /// Gate projection (`w1`).
    Gate,
    /// Down projection (`w2`).
    Down,
    /// Up projection (`w3`).
    Up,
}

#[derive(Default)]
struct SwiGluExpertParts {
    gate: Option<Array>,
    down: Option<Array>,
    up: Option<Array>,
}

/// Parses keys like `prefix.experts.17.w1.weight`.
pub fn parse_split_swiglu_expert_projection_key(
    key: &str,
) -> Option<(String, i32, SwiGluExpertProjection)> {
    let (prefix, rest) = key.split_once(".experts.")?;
    let mut parts = rest.split('.');
    let expert = parts.next()?.parse().ok()?;
    let projection = match parts.next()? {
        "w1" | "gate_proj" => SwiGluExpertProjection::Gate,
        "w2" | "down_proj" => SwiGluExpertProjection::Down,
        "w3" | "up_proj" => SwiGluExpertProjection::Up,
        _ => return None,
    };
    if parts.next()? != "weight" || parts.next().is_some() {
        return None;
    }
    Some((format!("{prefix}.experts"), expert, projection))
}

fn split_swiglu_expert_prefix_complete(
    expert_parts: &HashMap<(String, i32), SwiGluExpertParts>,
    prefix: &str,
    num_experts: i32,
) -> bool {
    (0..num_experts).all(|expert| {
        expert_parts
            .get(&(prefix.to_string(), expert))
            .is_some_and(|parts| parts.gate.is_some() && parts.down.is_some() && parts.up.is_some())
    })
}

fn pack_split_swiglu_expert_prefix(
    expert_parts: &mut HashMap<(String, i32), SwiGluExpertParts>,
    prefix: &str,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut gate_up = Vec::with_capacity(num_experts as usize);
    let mut down = Vec::with_capacity(num_experts as usize);
    for expert in 0..num_experts {
        let parts = expert_parts
            .remove(&(prefix.to_string(), expert))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "checkpoint is missing expert {expert} for '{prefix}'"
                ))
            })?;
        let gate = parts.gate.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.w1.weight"
            ))
        })?;
        let up = parts.up.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.w3.weight"
            ))
        })?;
        gate_up.push(concatenate_axis(&[gate, up], 0, stream)?);
        down.push(parts.down.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.w2.weight"
            ))
        })?);
    }
    let gate_up_proj = stack_axis(&gate_up, 0, stream)?;
    let down_proj = stack_axis(&down, 0, stream)?;
    eval([&gate_up_proj, &down_proj])?;
    Ok(HashMap::from([
        (format!("{prefix}.gate_up_proj"), gate_up_proj),
        (format!("{prefix}.down_proj"), down_proj),
    ]))
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Projection kind in a split ReLU2 expert checkpoint.
pub enum Relu2ExpertProjection {
    /// Expert up projection.
    Up,
    /// Expert down projection.
    Down,
}

#[derive(Default)]
struct Relu2ExpertParts {
    up: Option<Array>,
    down: Option<Array>,
}

/// Parses keys like `prefix.experts.17.up_proj.weight`.
pub fn parse_split_relu2_expert_projection_key(
    key: &str,
) -> Option<(String, i32, Relu2ExpertProjection)> {
    let (prefix, rest) = key.split_once(".experts.")?;
    let mut parts = rest.split('.');
    let expert = parts.next()?.parse().ok()?;
    let projection = match parts.next()? {
        "up_proj" => Relu2ExpertProjection::Up,
        "down_proj" => Relu2ExpertProjection::Down,
        _ => return None,
    };
    if parts.next()? != "weight" || parts.next().is_some() {
        return None;
    }
    Some((format!("{prefix}.experts"), expert, projection))
}

/// Packs split ReLU2 expert tensors into `prefix.experts.{up,down}_proj` banks.
pub fn transform_split_relu2_experts(
    loaded: HashMap<String, Array>,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut transformed = HashMap::with_capacity(loaded.len());
    let mut expert_parts: HashMap<(String, i32), Relu2ExpertParts> = HashMap::new();

    for (key, value) in loaded {
        if let Some((prefix, expert, projection)) = parse_split_relu2_expert_projection_key(&key) {
            let parts = expert_parts.entry((prefix, expert)).or_default();
            match projection {
                Relu2ExpertProjection::Up => parts.up = Some(value),
                Relu2ExpertProjection::Down => parts.down = Some(value),
            }
        } else {
            transformed.insert(key, value);
        }
    }

    let mut layer_prefixes = expert_parts
        .keys()
        .map(|(prefix, _)| prefix.clone())
        .collect::<Vec<_>>();
    layer_prefixes.sort();
    layer_prefixes.dedup();

    for prefix in layer_prefixes {
        transformed.extend(pack_split_relu2_expert_prefix(
            &mut expert_parts,
            &prefix,
            num_experts,
            stream,
        )?);
    }

    Ok(transformed)
}

fn split_relu2_expert_prefix_complete(
    expert_parts: &HashMap<(String, i32), Relu2ExpertParts>,
    prefix: &str,
    num_experts: i32,
) -> bool {
    (0..num_experts).all(|expert| {
        expert_parts
            .get(&(prefix.to_string(), expert))
            .is_some_and(|parts| parts.up.is_some() && parts.down.is_some())
    })
}

fn pack_split_relu2_expert_prefix(
    expert_parts: &mut HashMap<(String, i32), Relu2ExpertParts>,
    prefix: &str,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut up = Vec::with_capacity(num_experts as usize);
    let mut down = Vec::with_capacity(num_experts as usize);
    for expert in 0..num_experts {
        let parts = expert_parts
            .remove(&(prefix.to_string(), expert))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "checkpoint is missing expert {expert} for '{prefix}'"
                ))
            })?;
        up.push(parts.up.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.up_proj.weight"
            ))
        })?);
        down.push(parts.down.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "checkpoint is missing {prefix}.{expert}.down_proj.weight"
            ))
        })?);
    }

    let up_proj = stack_axis(&up, 0, stream)?;
    let down_proj = stack_axis(&down, 0, stream)?;
    eval([&up_proj, &down_proj])?;
    Ok(HashMap::from([
        (format!("{prefix}.up_proj"), up_proj),
        (format!("{prefix}.down_proj"), down_proj),
    ]))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        time::{SystemTime, UNIX_EPOCH},
    };

    use safemlx::{
        macros::ModuleParameters, module::Param, quantization::MaybeQuantized, Array, Device,
        DeviceType, Dtype, ExecutionContext,
    };

    use crate::{
        models::common::linear::unloaded_maybe_quantized_linear,
        quantization::{quantize_tensor, AffineQuantization, WeightQuantization},
    };

    use super::{
        load_arrays_quantized_strict, load_safetensors_quantized_strict,
        parse_split_swiglu_expert_projection_key, StrictLoadConfig, StrictLoadReport,
    };

    #[test]
    fn parses_split_swiglu_expert_names() {
        let (prefix, expert, projection) = parse_split_swiglu_expert_projection_key(
            "model.layers.3.feed_forward.experts.17.w3.weight",
        )
        .unwrap();
        assert_eq!(prefix, "model.layers.3.feed_forward.experts");
        assert_eq!(expert, 17);
        assert_eq!(projection, super::SwiGluExpertProjection::Up);
        let (_, _, projection) = parse_split_swiglu_expert_projection_key(
            "model.layers.3.mlp.experts.17.gate_proj.weight",
        )
        .unwrap();
        assert_eq!(projection, super::SwiGluExpertProjection::Gate);
        assert!(parse_split_swiglu_expert_projection_key(
            "model.layers.3.feed_forward.experts.17.bias"
        )
        .is_none());
    }

    #[derive(Debug, Clone, ModuleParameters)]
    struct RewrittenLinear {
        #[param]
        projection: MaybeQuantized<safemlx::nn::Linear>,
    }

    #[derive(Debug, Clone, ModuleParameters)]
    struct PackedExperts {
        #[param]
        experts: Param<Array>,
        #[param]
        experts_scales: Param<Option<Array>>,
        #[param]
        experts_biases: Param<Option<Array>>,
    }

    #[test]
    fn named_array_quantization_packs_rank_three_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = PackedExperts {
            experts: Param::<Array>::unloaded(&[3, 8, 8], Dtype::Uint32, stream).unwrap(),
            experts_scales: Param::<Option<Array>>::unloaded_some(&[3, 8, 2], Dtype::Uint8, stream)
                .unwrap(),
            experts_biases: Param::new(None),
        };
        let dense = Array::from_slice(&vec![0.25f32; 3 * 8 * 64], &[3, 8, 64]);
        let config = StrictLoadConfig::default();
        let mut report = StrictLoadReport::default();
        load_arrays_quantized_strict(
            &mut model,
            HashMap::from([("experts".into(), dense)]),
            stream,
            WeightQuantization::MxFp4,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&model, &config).unwrap();
        assert_eq!(model.experts.shape(), &[3, 8, 8]);
        assert_eq!(
            model.experts_scales.value.as_ref().unwrap().shape(),
            &[3, 8, 2]
        );
        assert!(model.experts_biases.value.is_none());
    }

    #[test]
    fn quantized_strict_load_applies_key_rewrites_before_target_selection() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights_stream = weights_context.stream();
        let quantization = AffineQuantization::default();
        let mut model = RewrittenLinear {
            projection: unloaded_maybe_quantized_linear(
                64,
                8,
                false,
                Some(quantization.into()),
                stream,
            )
            .unwrap(),
        };
        let values = (0..(8 * 64))
            .map(|index| (index as f32 - 255.5) / 64.0)
            .collect::<Vec<_>>();
        let dense = Array::from_slice(&values, &[8, 64]);
        let expected = quantize_tensor(&dense, quantization, stream).unwrap();
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "safemlx-rewritten-quantized-load-{}-{suffix}.safetensors",
            std::process::id()
        ));
        Array::save_safetensors([("checkpoint.projection.weight", &dense)], None, &path).unwrap();

        let config = StrictLoadConfig::default().rewrite_prefix("checkpoint.", "");
        let mut report = StrictLoadReport::default();
        load_safetensors_quantized_strict(
            &mut model,
            &path,
            weights_stream,
            stream,
            quantization.into(),
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&model, &config).unwrap();

        let MaybeQuantized::Quantized(projection) = model.projection else {
            panic!("target projection should use affine storage")
        };
        assert_eq!(
            projection
                .inner
                .weight
                .evaluated()
                .unwrap()
                .as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            projection.scales.evaluated().unwrap().as_slice::<f32>(),
            expected.scales.evaluated().unwrap().as_slice::<f32>()
        );
        assert_eq!(
            projection
                .biases
                .value
                .as_ref()
                .unwrap()
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
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mxfp4_strict_load_streams_weight_and_scales_without_biases() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights_stream = weights_context.stream();
        let mut model = RewrittenLinear {
            projection: unloaded_maybe_quantized_linear(
                64,
                8,
                false,
                Some(WeightQuantization::MxFp4),
                stream,
            )
            .unwrap(),
        };
        let dense = Array::from_slice(&vec![0.5f32; 8 * 64], &[8, 64]);
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "safemlx-mxfp4-strict-load-{}-{suffix}.safetensors",
            std::process::id()
        ));
        Array::save_safetensors([("projection.weight", &dense)], None, &path).unwrap();
        let config = StrictLoadConfig::default();
        let mut report = StrictLoadReport::default();
        load_safetensors_quantized_strict(
            &mut model,
            &path,
            weights_stream,
            stream,
            WeightQuantization::MxFp4,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&model, &config).unwrap();
        let MaybeQuantized::Quantized(projection) = model.projection else {
            panic!("target projection should use MXFP4 storage")
        };
        assert_eq!(projection.mode, safemlx::ops::QuantizationMode::MxFp4);
        assert!(projection.biases.value.is_none());
        std::fs::remove_file(path).unwrap();
    }
}
