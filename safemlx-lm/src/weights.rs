use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;
use safemlx::{
    module::{FlattenedModuleParamMut, ModuleParameters},
    ops::stack_axis,
    transforms::eval,
    Array, Stream,
};
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::error::Error;

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
    for_each_safetensor_array(path, stream, |key, value| {
        if let Some(param) = params.get_mut(key.as_str()) {
            **param = value;
        }
        Ok(())
    })
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
