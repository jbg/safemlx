use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{module::ModuleParameters, Array, Stream};

use crate::error::Error;

#[derive(Debug, Clone, Default)]
pub struct StrictLoadConfig {
    allowed_unused_prefixes: Vec<String>,
    allowed_missing_suffixes: Vec<String>,
    allowed_missing_contains: Vec<String>,
    key_prefixes_to_strip: Vec<String>,
    key_prefix_rewrites: Vec<(String, String)>,
}

impl StrictLoadConfig {
    pub fn allow_unused_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.allowed_unused_prefixes.push(prefix.into());
        self
    }

    pub fn allow_missing_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.allowed_missing_suffixes.push(suffix.into());
        self
    }

    pub fn allow_missing_contains(mut self, needle: impl Into<String>) -> Self {
        self.allowed_missing_contains.push(needle.into());
        self
    }

    pub fn strip_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefixes_to_strip.push(prefix.into());
        self
    }

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

#[derive(Debug, Clone, Default)]
pub struct StrictLoadReport {
    loaded: HashSet<String>,
    unused: Vec<String>,
    shape_mismatches: Vec<String>,
}

impl StrictLoadReport {
    fn record_loaded(&mut self, key: String) {
        self.loaded.insert(key);
    }

    fn record_unused(&mut self, key: String) {
        self.unused.push(key);
    }

    fn record_shape_mismatch(
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

pub fn load_safetensors_strict<M: ModuleParameters>(
    model: &mut M,
    path: impl AsRef<Path>,
    stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let loaded = Array::load_safetensors(path, stream)?;
    load_arrays_strict(model, loaded, config, report)
}

pub(crate) fn load_arrays_strict<M: ModuleParameters>(
    model: &mut M,
    loaded: HashMap<String, Array>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let mut params = model.parameters_mut().flatten();

    for (key, value) in loaded {
        let key = key.to_string();
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

    Ok(())
}
