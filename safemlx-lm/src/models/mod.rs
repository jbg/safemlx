//! Model-family implementations and high-level model loading.
//!
//! Use [`crate::models::LoadedModel`] when you want to load a model directory
//! together with its tokenizer and chat template. Use
//! [`crate::models::load_model`] and [`crate::models::load_tokenizer`] when you
//! want to manage those pieces separately.

use std::path::Path;

use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    Array, Stream,
};
use safemlx_lm_utils::tokenizer::{
    chat_template_kwargs as inspect_chat_template_kwargs, load_model_chat_template_from_file,
    ApplyChatTemplateArgs, Chat, Tokenizer as ChatTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::inspection::ActivationObserver;
use crate::models::common::CausalLm;
use crate::sampler::{DefaultSampler, Sampler};
use crate::{cache::ConcatKeyValueCache, error::Error};

/// Shared building blocks used by multiple decoder-only model families.
pub mod common;
/// Gemma 4 text model support.
pub mod gemma4;
/// Gemma 4 assistant draft-model support.
pub mod gemma4_assistant;
/// Llama decoder-only model support.
pub mod llama;
/// Nemotron-H hybrid Mamba2/attention/MoE config support.
pub mod nemotron_h;
/// Qwen3 decoder-only model support.
pub mod qwen3;
/// Qwen3.5 MoE text model support.
pub mod qwen3_5_moe;

#[derive(Debug, Clone, Deserialize)]
struct ModelMetadata {
    model_type: String,
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
    #[serde(default)]
    text_config: Option<TextModelMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
struct TextModelMetadata {
    model_type: String,
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenIdOrIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl TokenIdOrIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(id) => vec![id],
            Self::Multiple(ids) => ids,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Supported model-family dispatch target.
pub enum ModelKind {
    /// Gemma 4 text architecture.
    Gemma4,
    /// Llama decoder architecture.
    Llama,
    /// Nemotron-H hybrid Mamba2/attention/MoE architecture.
    NemotronH,
    /// Qwen3 decoder architecture.
    Qwen3,
    /// Qwen3.5 mixture-of-experts architecture.
    Qwen35Moe,
}

impl ModelKind {
    fn from_model_type(model_type: &str) -> Result<Self, Error> {
        match model_type {
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => Ok(Self::Gemma4),
            "llama" => Ok(Self::Llama),
            "nemotron_h" => Ok(Self::NemotronH),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35Moe),
            other => Err(Error::UnsupportedModelType(other.to_string())),
        }
    }
}

/// Details for a model config that this crate can load.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SupportedModelConfig {
    /// The runtime model implementation that will be used.
    pub kind: ModelKind,
    /// The top-level `model_type` from the submitted config.
    pub model_type: String,
    /// The resolved text model type used for dispatch.
    pub effective_model_type: String,
}

/// Result of checking whether a submitted model config is supported.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModelConfigSupport {
    /// The config is supported by this crate's loader.
    Supported(SupportedModelConfig),
    /// The config is not supported, with a human-readable reason.
    Unsupported {
        /// Human-readable reason the config is unsupported.
        reason: String,
    },
}

impl ModelConfigSupport {
    /// Returns true when this config is supported.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// Returns the unsupported reason, if this result is unsupported.
    pub fn unsupported_reason(&self) -> Option<&str> {
        match self {
            Self::Supported(_) => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// Checks a `config.json` string and reports whether it is supported.
pub fn check_model_config_json(config_json: &str) -> ModelConfigSupport {
    match serde_json::from_str::<Value>(config_json) {
        Ok(config) => check_model_config(&config),
        Err(error) => ModelConfigSupport::Unsupported {
            reason: format!("invalid model config JSON: {error}"),
        },
    }
}

/// Checks a parsed model config value and reports whether it is supported.
pub fn check_model_config(config: &Value) -> ModelConfigSupport {
    let metadata = match serde_json::from_value::<ModelMetadata>(config.clone()) {
        Ok(metadata) => metadata,
        Err(error) => {
            return ModelConfigSupport::Unsupported {
                reason: format!("invalid model config metadata: {error}"),
            };
        }
    };

    let effective_model_type = effective_model_type(&metadata);
    let kind = match ModelKind::from_model_type(&effective_model_type) {
        Ok(kind) => kind,
        Err(error) => {
            return ModelConfigSupport::Unsupported {
                reason: error.to_string(),
            };
        }
    };

    if let Err(error) = validate_model_config(kind, config) {
        return ModelConfigSupport::Unsupported {
            reason: error.to_string(),
        };
    }

    ModelConfigSupport::Supported(SupportedModelConfig {
        kind,
        model_type: metadata.model_type,
        effective_model_type,
    })
}

/// Reads `config.json` from a model directory and reports whether it is supported.
pub fn check_model_dir(model_dir: impl AsRef<Path>) -> ModelConfigSupport {
    let config_path = model_dir.as_ref().join("config.json");
    match std::fs::read_to_string(&config_path) {
        Ok(config_json) => check_model_config_json(&config_json),
        Err(error) => ModelConfigSupport::Unsupported {
            reason: format!("could not read {}: {error}", config_path.display()),
        },
    }
}

fn validate_model_config(kind: ModelKind, config: &Value) -> Result<(), Error> {
    match kind {
        ModelKind::Gemma4 => gemma4::validate_model_config_value(config),
        ModelKind::Llama => {
            serde_json::from_value::<llama::ModelArgs>(config.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid llama config: {error}"))
            })?;
            Ok(())
        }
        ModelKind::NemotronH => nemotron_h::validate_model_config_value(config),
        ModelKind::Qwen3 => {
            serde_json::from_value::<qwen3::ModelArgs>(config.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid qwen3 config: {error}"))
            })?;
            Ok(())
        }
        ModelKind::Qwen35Moe => qwen3_5_moe::validate_model_config_value(config),
    }
}

/// Loaded model value for any architecture supported by this crate.
pub enum Model {
    /// Gemma 4 text model.
    Gemma4(gemma4::Model),
    /// Llama model.
    Llama(llama::Model),
    /// Nemotron-H hybrid model.
    NemotronH(nemotron_h::Model),
    /// Qwen3 model.
    Qwen3(qwen3::Model),
    /// Qwen3.5 MoE text model.
    Qwen35Moe(qwen3_5_moe::Model),
}

impl Model {
    /// Returns the effective model type used for dispatch.
    pub fn model_type(&self) -> &str {
        match self {
            Self::Gemma4(model) => model.model_type(),
            Self::Llama(model) => model.model_type(),
            Self::NemotronH(model) => model.model_type(),
            Self::Qwen3(model) => model.model_type(),
            Self::Qwen35Moe(model) => model.model_type(),
        }
    }

    /// Runs a detailed instrumented forward pass for supported model families.
    ///
    /// Llama, Qwen3, Qwen3.5 MoE, and Gemma4 currently report detailed layer
    /// activations. Other families return an error until their family-specific
    /// inspection paths are wired.
    pub fn forward_with_observer(
        &mut self,
        input_tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Llama(model), ModelCache::KeyValue(cache)) => model.forward_with_observer(
                llama::ModelInput {
                    inputs: input_tokens,
                    mask,
                    cache,
                },
                stream,
                observer,
            ),
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => model.forward_with_observer(
                qwen3::ModelInput {
                    inputs: input_tokens,
                    mask,
                    cache,
                },
                stream,
                observer,
            ),
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => model.forward_with_observer(
                qwen3_5_moe::ModelInput {
                    inputs: input_tokens,
                    mask,
                    cache: Some(cache),
                },
                stream,
                observer,
            ),
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => model.forward_with_observer(
                gemma4::ModelInput {
                    inputs: input_tokens,
                    mask,
                    cache: &mut cache.kv,
                },
                stream,
                observer,
            ),
            (Self::NemotronH(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Computes initial prompt logits while reporting detailed activations.
    ///
    /// This mirrors each model family's prefill semantics and returns logits for
    /// the final prompt token with shape `[batch, vocab]`. Gemma4 uses a split
    /// prefill internally, so callers that want faithful instrumented generation
    /// should use this instead of calling [`Model::forward_with_observer`]
    /// directly on the whole prompt.
    pub fn prefill_logits_with_observer(
        &mut self,
        prompt_tokens: &Array,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                let prompt_len = prompt_tokens.shape()[1];
                if prompt_len <= 0 {
                    return Err(Exception::custom("prompt must contain at least one token"));
                }
                cache.token_ids = gemma4::token_ids_from_array(prompt_tokens, stream)?;
                cache.kv.clear();
                let logits = model.forward_with_observer(
                    gemma4::ModelInput {
                        inputs: prompt_tokens,
                        mask: None,
                        cache: &mut cache.kv,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                let logits = model.forward_with_observer(
                    llama::ModelInput {
                        inputs: prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                let logits = model.forward_with_observer(
                    qwen3::ModelInput {
                        inputs: prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                let logits = model.forward_with_observer(
                    qwen3_5_moe::ModelInput {
                        inputs: prompt_tokens,
                        mask: None,
                        cache: Some(cache),
                    },
                    stream,
                    observer,
                )?;
                let logits = final_token_logits(&logits, stream)?;
                model.adjust_prefill_logits(logits, cache, stream)
            }
            (Self::NemotronH(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates a token iterator for models that use a plain key/value cache.
    ///
    /// Qwen3.5 MoE uses a heterogeneous cache and must be driven with
    /// [`Model::generate_with_cache`] instead.
    pub fn generate<'a>(
        &'a mut self,
        cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Generate<'a> {
        self.generate_with_sampler(cache, temp, prompt_tokens, prng_key, stream, DefaultSampler)
    }

    /// Creates a token iterator using a caller-provided sampler.
    ///
    /// Qwen3.5 MoE uses a heterogeneous cache and must be driven with
    /// [`Model::generate_with_cache_sampler`] instead.
    pub fn generate_with_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> Generate<'a, S>
    where
        S: Sampler,
    {
        match self {
            Self::Gemma4(model) => Generate::Gemma4(common::Generate::with_sampler(
                model,
                cache,
                temp,
                prompt_tokens,
                prng_key,
                stream,
                sampler,
            )),
            Self::Llama(model) => Generate::Llama(llama::Generate::with_sampler(
                model,
                cache,
                temp,
                prompt_tokens,
                prng_key,
                stream,
                sampler,
            )),
            Self::Qwen3(model) => Generate::Qwen3(qwen3::Generate::with_sampler(
                model,
                cache,
                temp,
                prompt_tokens,
                prng_key,
                stream,
                sampler,
            )),
            Self::Qwen35Moe(_) => {
                panic!("qwen3_5_moe requires ModelCache; use generate_with_cache_sampler")
            }
            Self::NemotronH(_) => {
                panic!("nemotron_h requires ModelCache; use generate_with_cache_sampler")
            }
        }
    }

    /// Creates an empty cache value appropriate for this model.
    pub fn new_cache(&self) -> ModelCache {
        match self {
            Self::Gemma4(_) => ModelCache::Gemma4(gemma4::Cache::default()),
            Self::Llama(_) | Self::Qwen3(_) => ModelCache::KeyValue(Vec::new()),
            Self::NemotronH(model) => ModelCache::NemotronH(model.new_cache()),
            Self::Qwen35Moe(model) => ModelCache::Qwen35Moe(model.new_cache()),
        }
    }

    /// Computes logits for an initial prompt using a cache returned by [`Model::new_cache`].
    pub fn prefill_logits_with_cache(
        &mut self,
        prompt_tokens: &Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.prefill_logits(prompt_tokens, cache, stream)
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                model.prefill_logits(prompt_tokens, cache, stream)
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                model.prefill_logits(prompt_tokens, cache, stream)
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                model.prefill_logits(prompt_tokens, cache, stream)
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_logits(prompt_tokens, cache, stream)
            }
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Computes logits for decode tokens using a cache returned by [`Model::new_cache`].
    pub fn decode_logits_with_cache(
        &mut self,
        input_tokens: &Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.decode_logits(input_tokens, cache, stream)
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                model.decode_logits(input_tokens, cache, stream)
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                model.decode_logits(input_tokens, cache, stream)
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                model.decode_logits(input_tokens, cache, stream)
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.decode_logits(input_tokens, cache, stream)
            }
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates a token iterator using a cache returned by [`Model::new_cache`].
    pub fn generate_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        self.generate_with_cache_sampler(
            cache,
            temp,
            prompt_tokens,
            prng_key,
            stream,
            DefaultSampler,
        )
    }

    /// Creates a token iterator using a cache returned by [`Model::new_cache`] and a caller-provided sampler.
    pub fn generate_with_cache_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> ModelGenerate<'a, S>
    where
        S: Sampler,
    {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                ModelGenerate::Gemma4(gemma4::Generate::with_sampler(
                    model,
                    cache,
                    temp,
                    prompt_tokens,
                    prng_key,
                    stream,
                    sampler,
                ))
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                ModelGenerate::Llama(llama::Generate::with_sampler(
                    model,
                    cache,
                    temp,
                    prompt_tokens,
                    prng_key,
                    stream,
                    sampler,
                ))
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                ModelGenerate::Qwen3(qwen3::Generate::with_sampler(
                    model,
                    cache,
                    temp,
                    prompt_tokens,
                    prng_key,
                    stream,
                    sampler,
                ))
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                ModelGenerate::NemotronH(nemotron_h::Generate::with_sampler(
                    model,
                    cache,
                    temp,
                    prompt_tokens,
                    prng_key,
                    stream,
                    sampler,
                ))
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                ModelGenerate::Qwen35Moe(qwen3_5_moe::Generate::with_sampler(
                    model,
                    cache,
                    temp,
                    prompt_tokens,
                    prng_key,
                    stream,
                    sampler,
                ))
            }
            _ => panic!("model cache type does not match model kind"),
        }
    }
}

/// Token iterator for models backed by a vector of concatenating KV caches.
pub enum Generate<'a, S = DefaultSampler>
where
    S: Sampler,
{
    /// Gemma 4 generation iterator.
    Gemma4(common::Generate<'a, gemma4::Model, Vec<Option<ConcatKeyValueCache>>, S>),
    /// Llama generation iterator.
    Llama(llama::Generate<'a, ConcatKeyValueCache, S>),
    /// Qwen3 generation iterator.
    Qwen3(qwen3::Generate<'a, ConcatKeyValueCache, S>),
}

impl<S> Iterator for Generate<'_, S>
where
    S: Sampler,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Gemma4(generate) => generate.next(),
            Self::Llama(generate) => generate.next(),
            Self::Qwen3(generate) => generate.next(),
        }
    }
}

/// Cache value matching a [`Model`] variant.
#[derive(Clone)]
pub enum ModelCache {
    /// Gemma 4 generation cache.
    Gemma4(gemma4::Cache),
    /// Homogeneous per-layer key/value cache.
    KeyValue(Vec<Option<ConcatKeyValueCache>>),
    /// Heterogeneous Nemotron-H cache.
    NemotronH(nemotron_h::Cache),
    /// Heterogeneous Qwen3.5 MoE cache.
    Qwen35Moe(qwen3_5_moe::Cache),
}

/// Token iterator for any supported model variant.
pub enum ModelGenerate<'a, S = DefaultSampler>
where
    S: Sampler,
{
    /// Gemma 4 generation iterator.
    Gemma4(gemma4::Generate<'a, S>),
    /// Llama generation iterator.
    Llama(llama::Generate<'a, ConcatKeyValueCache, S>),
    /// Qwen3 generation iterator.
    Qwen3(qwen3::Generate<'a, ConcatKeyValueCache, S>),
    /// Nemotron-H generation iterator.
    NemotronH(nemotron_h::Generate<'a, S>),
    /// Qwen3.5 MoE generation iterator.
    Qwen35Moe(qwen3_5_moe::Generate<'a, S>),
}

impl<S> Iterator for ModelGenerate<'_, S>
where
    S: Sampler,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Gemma4(generate) => generate.next(),
            Self::Llama(generate) => generate.next(),
            Self::NemotronH(generate) => generate.next(),
            Self::Qwen3(generate) => generate.next(),
            Self::Qwen35Moe(generate) => generate.next(),
        }
    }
}

/// A model directory loaded together with its tokenizer and chat template.
///
/// This is the most convenient entry point for text generation: it owns the
/// architecture-specific [`Model`], tokenizer, optional chat template, model id
/// used by the template renderer, and EOS token ids parsed from config.
pub struct LoadedModel {
    model: Model,
    tokenizer: ChatTokenizer,
    chat_template: Option<String>,
    model_id: String,
    eos_token_ids: Vec<u32>,
}

impl LoadedModel {
    /// Loads a supported model directory, tokenizer, optional chat template, and weights.
    pub fn load(
        model_dir: impl AsRef<Path>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        let metadata = read_model_metadata(model_dir)?;
        let model_type = effective_model_type(&metadata);
        let kind = ModelKind::from_model_type(&model_type)?;
        let mut tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(model_dir)?);
        tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(model_dir)?);
        let chat_template = load_chat_template(model_dir)?;
        let model = match kind {
            ModelKind::Gemma4 => Model::Gemma4(gemma4::load_gemma4_model(
                model_dir,
                stream,
                weights_stream,
            )?),
            ModelKind::Llama => {
                Model::Llama(llama::load_llama_model(model_dir, stream, weights_stream)?)
            }
            ModelKind::NemotronH => Model::NemotronH(nemotron_h::load_nemotron_h_model(
                model_dir,
                stream,
                weights_stream,
            )?),
            ModelKind::Qwen3 => {
                Model::Qwen3(qwen3::load_qwen3_model(model_dir, stream, weights_stream)?)
            }
            ModelKind::Qwen35Moe => Model::Qwen35Moe(qwen3_5_moe::load_qwen3_5_moe_model(
                model_dir,
                stream,
                weights_stream,
            )?),
        };
        let eos_token_ids = metadata
            .eos_token_id
            .or_else(|| {
                metadata
                    .text_config
                    .and_then(|text_config| text_config.eos_token_id)
            })
            .map(TokenIdOrIds::into_vec)
            .unwrap_or_default();

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            model_id: model_type,
            eos_token_ids,
        })
    }

    /// Returns the effective runtime model type.
    pub fn model_type(&self) -> &str {
        self.model.model_type()
    }

    /// Returns the model id passed to chat-template rendering.
    pub fn model_id_for_template(&self) -> &str {
        &self.model_id
    }

    /// Returns whether a chat template is available for this model.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Returns likely user-provided kwargs referenced by the loaded chat template.
    ///
    /// This is static template analysis and does not infer value types or
    /// defaults. Standard chat-template variables supplied by this crate are
    /// excluded.
    pub fn chat_template_kwargs(&self) -> Result<Vec<String>, Error> {
        let Some(template) = &self.chat_template else {
            return Ok(Vec::new());
        };
        Ok(inspect_chat_template_kwargs(template, &self.model_id)?
            .into_iter()
            .filter(|name| !self.tokenizer.template_kwargs().contains_key(name))
            .collect())
    }

    /// Applies the loaded chat template to structured conversations.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template<'a, I, R, T>(
        &'a mut self,
        conversations: I,
        tools: Option<&'a [serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + 'a,
    {
        self.apply_chat_template_with_kwargs(conversations, tools, add_generation_prompt, None)
    }

    /// Applies the loaded chat template to structured conversations with extra template variables.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template_with_kwargs<'a, I, R, T>(
        &'a mut self,
        conversations: I,
        tools: Option<&'a [serde_json::Value]>,
        add_generation_prompt: bool,
        template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Option<String>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + 'a,
    {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };

        let rendered = self.tokenizer.apply_chat_template(
            template,
            ApplyChatTemplateArgs {
                conversations,
                tools,
                documents: None,
                model_id: &self.model_id,
                chat_template_id: None,
                add_generation_prompt: Some(add_generation_prompt),
                continue_final_message: None,
                template_kwargs,
            },
        )?;
        Ok(rendered.into_iter().next())
    }

    /// Applies the loaded chat template to JSON-valued conversations.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template_json(
        &mut self,
        conversations: impl IntoIterator<Item = Vec<serde_json::Value>>,
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, Error> {
        self.apply_chat_template_json_with_kwargs(conversations, tools, add_generation_prompt, None)
    }

    /// Applies the loaded chat template to JSON-valued conversations with extra template variables.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template_json_with_kwargs(
        &mut self,
        conversations: impl IntoIterator<Item = Vec<serde_json::Value>>,
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Option<String>, Error> {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };

        let rendered = self.tokenizer.apply_chat_template_json(
            template,
            conversations,
            tools,
            &self.model_id,
            add_generation_prompt,
            template_kwargs,
        )?;
        Ok(rendered.into_iter().next())
    }

    /// Encodes text to tokenizer ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Error> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)?
            .get_ids()
            .to_vec())
    }

    /// Encodes text and returns a `[1, len]` token-id array on `stream`.
    pub fn encode_to_array(
        &self,
        text: &str,
        add_special_tokens: bool,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = self.encode(text, add_special_tokens)?;
        Ok(Array::from(ids.as_slice()).try_index_device(NewAxis, stream)?)
    }

    /// Decodes tokenizer ids back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(Into::into)
    }

    /// Returns EOS token ids from the model config, if any.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Returns true when `id` is one of the configured EOS token ids.
    pub fn is_eos_token(&self, id: u32) -> bool {
        self.eos_token_ids.contains(&id)
    }

    /// Creates a token iterator for models that use a plain key/value cache.
    pub fn generate<'a>(
        &'a mut self,
        cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Generate<'a> {
        self.model
            .generate(cache, temp, prompt_tokens, prng_key, stream)
    }

    /// Creates a token iterator using a caller-provided sampler.
    pub fn generate_with_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> Generate<'a, S>
    where
        S: Sampler,
    {
        self.model
            .generate_with_sampler(cache, temp, prompt_tokens, prng_key, stream, sampler)
    }

    /// Creates an empty cache value appropriate for the loaded model.
    pub fn new_cache(&self) -> ModelCache {
        self.model.new_cache()
    }

    /// Computes logits for an initial prompt using a cache returned by [`LoadedModel::new_cache`].
    pub fn prefill_logits_with_cache(
        &mut self,
        prompt_tokens: &Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.model
            .prefill_logits_with_cache(prompt_tokens, cache, stream)
    }

    /// Computes initial prompt logits while reporting detailed activations.
    ///
    /// The returned logits have shape `[batch, vocab]` and match
    /// [`LoadedModel::prefill_logits_with_cache`] for the same model/cache.
    pub fn prefill_logits_with_observer(
        &mut self,
        prompt_tokens: &Array,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        self.model
            .prefill_logits_with_observer(prompt_tokens, cache, stream, observer)
    }

    /// Computes logits for decode tokens using a cache returned by [`LoadedModel::new_cache`].
    pub fn decode_logits_with_cache(
        &mut self,
        input_tokens: &Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.model
            .decode_logits_with_cache(input_tokens, cache, stream)
    }

    /// Creates a token iterator using a cache returned by [`LoadedModel::new_cache`].
    pub fn generate_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        self.model
            .generate_with_cache(cache, temp, prompt_tokens, prng_key, stream)
    }

    /// Creates a token iterator using a cache returned by [`LoadedModel::new_cache`] and a caller-provided sampler.
    pub fn generate_with_cache_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> ModelGenerate<'a, S>
    where
        S: Sampler,
    {
        self.model.generate_with_cache_sampler(
            cache,
            temp,
            prompt_tokens,
            prng_key,
            stream,
            sampler,
        )
    }

    /// Returns a mutable reference to the underlying architecture-specific model.
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
}

fn final_token_logits(logits: &Array, stream: &Stream) -> Result<Array, Exception> {
    match logits.ndim() {
        2 => Ok(logits.clone()),
        3 => logits.try_index_device((.., -1, ..), stream),
        ndim => Err(Exception::custom(format!(
            "expected 2D or 3D logits, got {ndim}D with shape {:?}",
            logits.shape()
        ))),
    }
}

/// Loads only the model weights and architecture from a model directory.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let metadata = read_model_metadata(model_dir)?;
    match ModelKind::from_model_type(&effective_model_type(&metadata))? {
        ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Llama => Ok(Model::Llama(llama::load_llama_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::NemotronH => Ok(Model::NemotronH(nemotron_h::load_nemotron_h_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen3 => Ok(Model::Qwen3(qwen3::load_qwen3_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen35Moe => Ok(Model::Qwen35Moe(qwen3_5_moe::load_qwen3_5_moe_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
    }
}

/// Loads only the tokenizer from a supported model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let model_dir = model_dir.as_ref();
    let metadata = read_model_metadata(model_dir)?;
    match ModelKind::from_model_type(&effective_model_type(&metadata))? {
        ModelKind::Gemma4 => gemma4::load_gemma4_tokenizer(model_dir),
        ModelKind::Llama => llama::load_llama_tokenizer(model_dir),
        ModelKind::NemotronH => nemotron_h::load_nemotron_h_tokenizer(model_dir),
        ModelKind::Qwen3 => qwen3::load_qwen3_tokenizer(model_dir),
        ModelKind::Qwen35Moe => qwen3_5_moe::load_qwen3_5_moe_tokenizer(model_dir),
    }
}

/// Returns likely user-provided kwargs referenced by a model directory's chat template.
///
/// This reads tokenizer/chat-template metadata only and does not load model weights.
pub fn chat_template_kwargs(model_dir: impl AsRef<Path>) -> Result<Vec<String>, Error> {
    let model_dir = model_dir.as_ref();
    let Some(template) = load_chat_template(model_dir)? else {
        return Ok(Vec::new());
    };
    let model_id = model_dir.display().to_string();
    let tokenizer_template_kwargs = load_tokenizer_template_kwargs(model_dir)?;
    Ok(inspect_chat_template_kwargs(&template, &model_id)?
        .into_iter()
        .filter(|name| !tokenizer_template_kwargs.contains_key(name))
        .collect())
}

fn read_model_metadata(model_dir: &Path) -> Result<ModelMetadata, Error> {
    let config_path = model_dir.join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

fn effective_model_type(metadata: &ModelMetadata) -> String {
    if matches!(
        metadata.model_type.as_str(),
        "gemma4" | "gemma4_unified" | "qwen3_5_moe"
    ) {
        metadata
            .text_config
            .as_ref()
            .map(|text_config| text_config.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    } else if ModelKind::from_model_type(&metadata.model_type).is_ok() {
        metadata.model_type.clone()
    } else {
        metadata
            .text_config
            .as_ref()
            .map(|text_config| text_config.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    }
}

fn load_chat_template(model_dir: &Path) -> Result<Option<String>, Error> {
    let config_path = model_dir.join("tokenizer_config.json");
    if config_path.exists() {
        if let Some(template) = load_model_chat_template_from_file(config_path)? {
            return Ok(Some(template));
        }
    }

    let jinja_path = model_dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return Ok(Some(std::fs::read_to_string(jinja_path)?));
    }

    let metadata = read_model_metadata(model_dir)?;
    if matches!(metadata.model_type.as_str(), "gemma4" | "gemma4_unified")
        || metadata.text_config.as_ref().is_some_and(|text_config| {
            matches!(
                text_config.model_type.as_str(),
                "gemma4_text" | "gemma4_unified_text"
            )
        })
    {
        return Ok(Some(GEMMA4_TEXT_TEMPLATE.to_string()));
    }

    Ok(None)
}

fn load_tokenizer_template_kwargs(model_dir: &Path) -> Result<Map<String, Value>, Error> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(Map::new());
    }

    let value: Value = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    let Some(object) = value.as_object() else {
        return Ok(Map::new());
    };

    Ok(object
        .iter()
        .filter(|(key, value)| key.ends_with("_token") && (value.is_string() || value.is_null()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

const GEMMA4_TEXT_TEMPLATE: &str = r#"<bos>{% for message in messages %}{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}<|turn>{{ role }}
{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] }}{% elif content['type'] == 'image' %}<|image>{% elif content['type'] == 'audio' %}<|audio>{% endif %}{% endfor %}{% endif %}<turn|>
{% endfor %}{% if add_generation_prompt %}<|turn>model
{% endif %}"#;

#[cfg(test)]
mod tests {
    use super::{
        chat_template_kwargs, check_model_config, check_model_config_json, check_model_dir,
        load_chat_template, load_tokenizer, load_tokenizer_template_kwargs, LoadedModel,
    };
    use crate::inspection::ActivationRecorder;
    use safemlx_lm_utils::tokenizer::Tokenizer as ChatTokenizer;
    use serde_json::json;
    use std::{
        fs,
        process::Command,
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };
    use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

    static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[test]
    #[ignore = "requires MLX runtime execution and SAFEMLX_INSPECTION_MODEL_DIR"]
    fn observer_forward_reports_attention_and_residual_hooks() {
        let model_dir = std::env::var("SAFEMLX_INSPECTION_MODEL_DIR")
            .expect("set SAFEMLX_INSPECTION_MODEL_DIR to a local model directory");
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut model = LoadedModel::load(model_dir, ctx.stream(), weights_ctx.stream()).unwrap();
        let input = model.encode_to_array("hello", true, ctx.stream()).unwrap();
        let mut cache = model.new_cache();
        let mut recorder = ActivationRecorder::new();

        model
            .model_mut()
            .forward_with_observer(&input, None, &mut cache, ctx.stream(), &mut recorder)
            .unwrap();

        let names = recorder
            .activations()
            .iter()
            .map(|activation| activation.name.as_str())
            .collect::<Vec<_>>();
        assert!(
            names.iter().any(|name| name.ends_with(".attention_probs")),
            "{names:?}"
        );
        assert!(
            names
                .iter()
                .any(|name| name.ends_with(".residual_delta_attention")),
            "{names:?}"
        );
        assert!(
            names
                .iter()
                .any(|name| name.ends_with(".residual_delta_mlp"))
                || names
                    .iter()
                    .any(|name| name.ends_with(".residual_delta_moe")),
            "{names:?}"
        );
    }

    fn temp_model_dir(config: &str) -> std::path::PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "model_metadata_test_{}_{}_{}",
            std::process::id(),
            id,
            counter
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), config).unwrap();
        Tokenizer::new(WordLevel::default())
            .save(dir.join("tokenizer.json"), false)
            .unwrap();
        dir
    }

    #[test]
    fn load_tokenizer_accepts_top_level_qwen3_5_moe_metadata() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text"
              }
            }"#,
        );
        let tokenizer = load_tokenizer(&dir).unwrap();
        assert_eq!(tokenizer.get_vocab_size(false), 0);
    }

    #[test]
    fn load_chat_template_reads_standalone_jinja_file() {
        let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
        fs::write(
            dir.join("chat_template.jinja"),
            "hello {{ messages[0].role }}",
        )
        .unwrap();

        let template = load_chat_template(&dir).unwrap().unwrap();
        assert_eq!(template, "hello {{ messages[0].role }}");
    }

    #[test]
    fn load_tokenizer_template_kwargs_reads_special_tokens() {
        let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{
              "bos_token": "<bos>",
              "eos_token": "<eos>",
              "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}{{ custom_flag }}",
              "model_max_length": 128
            }"#,
        )
        .unwrap();

        let kwargs = load_tokenizer_template_kwargs(&dir).unwrap();
        assert_eq!(kwargs.get("bos_token"), Some(&json!("<bos>")));
        assert_eq!(kwargs.get("eos_token"), Some(&json!("<eos>")));
        assert!(!kwargs.contains_key("chat_template"));
        assert!(!kwargs.contains_key("model_max_length"));
        assert_eq!(chat_template_kwargs(&dir).unwrap(), vec!["custom_flag"]);
    }

    #[test]
    #[ignore = "requires local Nemotron-H model files and Python transformers"]
    fn nemotron_chat_template_matches_transformers_on_small_prompts() {
        let model_dir = std::env::var("NEMOTRON_H_PARITY_MODEL_DIR")
            .expect("set NEMOTRON_H_PARITY_MODEL_DIR to a local Nemotron-H snapshot");
        let model_dir = std::path::PathBuf::from(model_dir);
        let template = load_chat_template(&model_dir).unwrap().unwrap();
        let mut tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(&model_dir).unwrap());
        let conversations = vec![vec![
            json!({"role": "system", "content": "You are concise."}),
            json!({"role": "user", "content": "What is 2+2?"}),
        ]];
        let prompts = ["Hello, world!", "What is 84 * 3 / 2?"];
        let local_prompt_ids = prompts
            .iter()
            .map(|prompt| tokenizer.encode(*prompt, false).unwrap().get_ids().to_vec())
            .collect::<Vec<_>>();

        let rendered = tokenizer
            .apply_chat_template_json(
                template.clone(),
                conversations.clone(),
                None,
                "nemotron_h",
                true,
                None,
            )
            .unwrap()
            .remove(0);
        let script = r#"
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
messages = json.loads(sys.argv[2])
prompts = json.loads(sys.argv[3])
rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = [tok.encode(prompt, add_special_tokens=False) for prompt in prompts]
print(json.dumps({"rendered": rendered, "ids": ids}))
"#;
        let python =
            std::env::var("NEMOTRON_H_PARITY_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let output = Command::new(&python)
            .arg("-c")
            .arg(script)
            .arg(&model_dir)
            .arg(serde_json::to_string(&conversations[0]).unwrap())
            .arg(serde_json::to_string(&prompts).unwrap())
            .output()
            .expect("failed to run Python transformers parity check");
        assert!(
            output.status.success(),
            "transformers parity script failed with {python}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(rendered, expected["rendered"].as_str().unwrap());
        let expected_ids: Vec<Vec<u32>> = serde_json::from_value(expected["ids"].clone()).unwrap();
        assert_eq!(local_prompt_ids, expected_ids);
        assert!(template.contains("<|im_start|>assistant"));
    }

    #[test]
    fn check_model_config_reports_supported_llama() {
        let support = check_model_config(&json!({
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "head_dim": 4
        }));

        assert!(support.is_supported(), "{support:?}");
    }

    #[test]
    fn check_model_config_reports_unsupported_model_type() {
        let support = check_model_config(&json!({
            "model_type": "not_a_model"
        }));

        assert!(!support.is_supported());
        assert_eq!(
            support.unsupported_reason(),
            Some("unsupported model type: not_a_model")
        );
    }

    #[test]
    fn check_model_config_json_reports_invalid_json() {
        let support = check_model_config_json("{not json");

        assert!(!support.is_supported());
        assert!(support
            .unsupported_reason()
            .unwrap()
            .starts_with("invalid model config JSON:"));
    }

    #[test]
    fn check_model_config_reports_qwen3_5_moe_missing_text_config() {
        let support = check_model_config(&json!({
            "model_type": "qwen3_5_moe"
        }));

        assert!(!support.is_supported());
        assert_eq!(
            support.unsupported_reason(),
            Some("unsupported model architecture: qwen3_5_moe config is missing text_config")
        );
    }

    #[test]
    fn check_model_config_reports_supported_qwen3_5_moe() {
        let support = check_model_config(&json!({
            "model_type": "qwen3_5_moe",
            "image_token_id": 248056,
            "video_token_id": 248057,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 128,
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128
            }
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Qwen35Moe,
                model_type: "qwen3_5_moe".to_string(),
                effective_model_type: "qwen3_5_moe_text".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_supported_nemotron_h() {
        let support = check_model_config(&json!({
            "model_type": "nemotron_h",
            "vocab_size": 131072,
            "hidden_size": 2688,
            "intermediate_size": 1856,
            "num_hidden_layers": 6,
            "hybrid_override_pattern": "MEMEM*",
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu",
            "ssm_state_size": 128,
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "conv_kernel": 4,
            "chunk_size": 128,
            "n_groups": 8,
            "n_routed_experts": 128,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "torch_dtype": "bfloat16"
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::NemotronH,
                model_type: "nemotron_h".to_string(),
                effective_model_type: "nemotron_h".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_gemma4_moe_unsupported() {
        let support = check_model_config(&json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "rms_norm_eps": 0.00001,
                "vocab_size": 32,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "head_dim": 4,
                "enable_moe_block": true
            }
        }));

        assert!(!support.is_supported());
        assert_eq!(
            support.unsupported_reason(),
            Some("unsupported model architecture: Gemma 4 MoE models are not supported yet")
        );
    }

    #[test]
    fn check_model_config_reports_supported_gemma4_unified_text() {
        let support = check_model_config(&json!({
            "model_type": "gemma4_unified",
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "rms_norm_eps": 0.00001,
                "vocab_size": 32,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "head_dim": 4,
                "enable_moe_block": false
            }
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Gemma4,
                model_type: "gemma4_unified".to_string(),
                effective_model_type: "gemma4_unified_text".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_gemma4_unified_moe_unsupported() {
        let support = check_model_config(&json!({
            "model_type": "gemma4_unified",
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "rms_norm_eps": 0.00001,
                "vocab_size": 32,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "head_dim": 4,
                "enable_moe_block": true
            }
        }));

        assert!(!support.is_supported());
        assert_eq!(
            support.unsupported_reason(),
            Some("unsupported model architecture: Gemma 4 MoE models are not supported yet")
        );
    }

    #[test]
    fn check_model_dir_reads_config_json() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "llama",
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "rms_norm_eps": 0.00001,
              "vocab_size": 32,
              "num_key_value_heads": 2,
              "max_position_embeddings": 128,
              "head_dim": 4
            }"#,
        );

        let support = check_model_dir(&dir);
        assert!(support.is_supported(), "{support:?}");
    }
}
