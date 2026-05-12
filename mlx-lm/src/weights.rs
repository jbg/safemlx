use std::{collections::HashSet, path::Path};

use mlx_rs::{
    module::{ModuleParameters, ModuleParametersExt},
    Array,
};

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct StrictLoadConfig {
    allowed_unused_prefixes: Vec<String>,
    allowed_missing_suffixes: Vec<String>,
    key_prefixes_to_strip: Vec<String>,
    key_prefix_rewrites: Vec<(String, String)>,
}

impl Default for StrictLoadConfig {
    fn default() -> Self {
        Self {
            allowed_unused_prefixes: Vec::new(),
            allowed_missing_suffixes: Vec::new(),
            key_prefixes_to_strip: Vec::new(),
            key_prefix_rewrites: Vec::new(),
        }
    }
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
    }

    fn candidates(&self, key: &str) -> Vec<String> {
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
}

impl StrictLoadReport {
    fn record_loaded(&mut self, key: String) {
        self.loaded.insert(key);
    }

    fn record_unused(&mut self, key: String) {
        self.unused.push(key);
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

        missing.sort();
        unused.sort();

        if missing.is_empty() && unused.is_empty() {
            Ok(())
        } else {
            Err(Error::StrictLoadValidation { missing, unused })
        }
    }
}

pub fn load_safetensors_strict<M: ModuleParametersExt>(
    model: &mut M,
    path: impl AsRef<Path>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let loaded = Array::load_safetensors(path)?;
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
                **param = value;
                report.record_loaded(candidate);
            }
        } else {
            report.record_unused(key);
        }
    }

    model.eval()?;
    Ok(())
}
