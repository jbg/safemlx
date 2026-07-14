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
    ops::{GgufMetadata, GgufMetadataValue},
    Array, Stream,
};
use safemlx_lm_utils::tokenizer::{
    chat_template_kwargs as inspect_chat_template_kwargs, load_model_chat_template_from_file,
    ApplyChatTemplateArgs, Chat, Tokenizer as ChatTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::gguf_tokenizer::{self, GgufTokenizer};
use crate::inspection::ActivationObserver;
use crate::models::common::CausalLm;
#[cfg(feature = "media-processing")]
use crate::processor::{load_processor, ModelProcessor, PreparedModelInput, ProcessorInput};
use crate::quantization::AffineQuantization;
use crate::sampler::{DefaultSampler, Sampler};
use crate::{
    cache::{ConcatKeyValueCache, SlidingKeyValueCache},
    error::Error,
};

/// Shared building blocks used by multiple decoder-only model families.
pub mod common;
/// Gemma 4 text model support.
pub mod gemma4;
/// Gemma 4 assistant draft-model support.
pub mod gemma4_assistant;
mod gemma4_audio;
mod gemma4_multimodal;
mod gemma4_vision;
/// Typed runtime input support.
pub mod input;
/// Llama decoder-only model support.
pub mod llama;
/// Moshi token language-model support.
///
/// This module operates on pre-tokenized Mimi streams. It intentionally does
/// not implement audio encoding, decoding, or realtime device I/O.
pub mod moshi;
/// Nemotron-H hybrid Mamba2/attention/MoE config support.
pub mod nemotron_h;
/// PersonaPlex realtime speech-to-speech token model support.
///
/// This module operates on pre-tokenized Mimi streams and hybrid prompt tokens.
/// It intentionally does not implement audio encoding, decoding, or realtime
/// device I/O.
pub mod personaplex;
/// Qwen3 decoder-only model support.
pub mod qwen3;
/// Qwen3.5 MoE text model support.
pub mod qwen3_5_moe;
/// Qwen3-VL multimodal conditional-generation support.
pub mod qwen3_vl;
mod qwen_vl;

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
    /// Llama-compatible dense decoder architecture, including Mistral.
    Llama,
    /// Nemotron-H hybrid Mamba2/attention/MoE architecture.
    NemotronH,
    /// PersonaPlex realtime speech-to-speech architecture.
    PersonaPlex,
    /// Qwen3 decoder architecture.
    Qwen3,
    /// Qwen3-VL multimodal architecture.
    Qwen3Vl,
    /// Qwen3.5 mixture-of-experts architecture.
    Qwen35Moe,
}

/// Architecture-independent options for loading model weights.
///
/// When `quantization` is set for a dense checkpoint, eligible parameters are
/// affine-quantized and materialized one tensor at a time. Checkpoints already
/// carrying matching affine metadata are loaded directly without requantizing.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ModelLoadOptions {
    /// Optional MLX affine quantization requested during dense checkpoint loading.
    pub quantization: Option<AffineQuantization>,
}

impl ModelLoadOptions {
    /// Creates load options that affine-quantize eligible dense weights on load.
    pub fn with_quantization(quantization: AffineQuantization) -> Self {
        Self {
            quantization: Some(quantization),
        }
    }
}

impl ModelKind {
    fn from_model_type(model_type: &str) -> Result<Self, Error> {
        match model_type {
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => Ok(Self::Gemma4),
            "llama" | "mistral" => Ok(Self::Llama),
            "nemotron_h" => Ok(Self::NemotronH),
            "personaplex" => Ok(Self::PersonaPlex),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_vl" | "qwen3_vl_text" => Ok(Self::Qwen3Vl),
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
        ModelKind::Llama => llama::validate_model_config_value(config),
        ModelKind::NemotronH => nemotron_h::validate_model_config_value(config),
        ModelKind::PersonaPlex => personaplex::validate_model_config_value(config),
        ModelKind::Qwen3 => {
            serde_json::from_value::<qwen3::ModelArgs>(config.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid qwen3 config: {error}"))
            })?;
            Ok(())
        }
        ModelKind::Qwen3Vl => qwen3_vl::validate_model_config_value(config),
        ModelKind::Qwen35Moe => qwen3_5_moe::validate_model_config_value(config),
    }
}

/// Loaded model value for any architecture supported by this crate.
pub enum Model {
    /// Gemma 4 text model.
    Gemma4(gemma4::Model),
    /// Llama-compatible dense model.
    Llama(llama::Model),
    /// Nemotron-H hybrid model.
    NemotronH(nemotron_h::Model),
    /// Qwen3 model.
    Qwen3(qwen3::Model),
    /// Qwen3-VL multimodal model.
    Qwen3Vl(qwen3_vl::Model),
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
            Self::Qwen3Vl(model) => model.model_type(),
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
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => model
                .forward_with_observer(
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
                    inputs_embeds: None,
                    mask,
                    cache: Some(cache),
                },
                stream,
                observer,
            ),
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => model.forward_with_observer(
                gemma4::ModelInput {
                    inputs: input_tokens,
                    inputs_embeds: None,
                    per_layer_input_ids: None,
                    mask,
                    sliding_mask: None,
                    cache: &mut cache.kv,
                },
                stream,
                observer,
            ),
            (Self::NemotronH(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            (Self::Qwen3Vl(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl yet",
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
    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    llama::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    llama::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    qwen3::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::NemotronH(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            (Self::Qwen3Vl(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl yet",
            )),
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates an empty cache value appropriate for this model.
    pub fn new_cache(&self) -> ModelCache {
        match self {
            Self::Gemma4(_) => ModelCache::Gemma4(gemma4::Cache::default()),
            Self::Llama(model) => match model.sliding_window() {
                Some(_) => ModelCache::SlidingKeyValue(model.new_sliding_cache()),
                None => ModelCache::KeyValue(Vec::new()),
            },
            Self::Qwen3(_) => ModelCache::KeyValue(Vec::new()),
            Self::Qwen3Vl(model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::NemotronH(model) => ModelCache::NemotronH(model.new_cache()),
            Self::Qwen35Moe(model) => ModelCache::Qwen35Moe(model.new_cache()),
        }
    }

    /// Computes logits for an initial typed input using a cache returned by [`Model::new_cache`].
    pub fn prefill_input_with_cache(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates a token iterator from typed input using a cache returned by [`Model::new_cache`].
    pub fn generate_input_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        self.generate_input_with_cache_sampler(cache, temp, input, prng_key, stream, DefaultSampler)
    }

    /// Creates a token iterator from typed input with a caller-provided sampler.
    pub fn generate_input_with_cache_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
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
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => ModelGenerate::Llama(
                llama::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => {
                ModelGenerate::LlamaSliding(llama::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => ModelGenerate::Qwen3(
                qwen3::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
                ModelGenerate::Qwen3Vl(qwen3_vl::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                ModelGenerate::NemotronH(nemotron_h::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                ModelGenerate::Qwen35Moe(qwen3_5_moe::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            _ => panic!("model cache type does not match model kind"),
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
    /// Qwen3-VL key/value cache and multimodal position state.
    Qwen3Vl(qwen3_vl::Cache),
    /// Homogeneous bounded cache for sliding-window attention.
    SlidingKeyValue(Vec<Option<SlidingKeyValueCache>>),
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
    /// Llama-compatible generation with bounded sliding-window caches.
    LlamaSliding(llama::Generate<'a, SlidingKeyValueCache, S>),
    /// Qwen3 generation iterator.
    Qwen3(qwen3::Generate<'a, ConcatKeyValueCache, S>),
    /// Qwen3-VL generation iterator.
    Qwen3Vl(qwen3_vl::Generate<'a, S>),
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
            Self::LlamaSliding(generate) => generate.next(),
            Self::NemotronH(generate) => generate.next(),
            Self::Qwen3(generate) => generate.next(),
            Self::Qwen3Vl(generate) => generate.next(),
            Self::Qwen35Moe(generate) => generate.next(),
        }
    }
}

/// A model directory or GGUF file loaded together with its tokenizer and chat template.
///
/// This is the most convenient entry point for text generation: it owns the
/// architecture-specific [`Model`], tokenizer, optional chat template, model id
/// used by the template renderer, and EOS token ids parsed from config.
pub struct LoadedModel {
    model: Model,
    #[cfg(feature = "media-processing")]
    processor: Option<ModelProcessor>,
    tokenizer: ChatTokenizer,
    chat_template: Option<String>,
    model_id: String,
    eos_token_ids: Vec<u32>,
}

impl LoadedModel {
    /// Loads a supported model directory or GGUF file with its tokenizer.
    ///
    /// GGUF tokenizers are reconstructed from embedded metadata. A sibling
    /// `tokenizer.json` is used only when the embedded tokenizer is absent or
    /// uses an unsupported tokenizer model.
    pub fn load(
        model_dir: impl AsRef<Path>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::load_with_options(
            model_dir,
            ModelLoadOptions::default(),
            stream,
            weights_stream,
        )
    }

    /// Loads a supported model using architecture-independent weight options.
    pub fn load_with_options(
        model_dir: impl AsRef<Path>,
        options: ModelLoadOptions,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        if is_gguf_file(model_dir) {
            let sidecar_dir = gguf_sidecar_dir(model_dir);
            let LoadedGgufModel {
                model,
                eos_token_ids,
                chat_template,
                tokenizer,
            } = load_gguf_model_data(model_dir, true, stream, weights_stream)?;
            let GgufTokenizer {
                tokenizer,
                template_kwargs,
            } = tokenizer.expect("GGUF tokenizer requested by the combined loader");
            let mut tokenizer = ChatTokenizer::from_tokenizer(tokenizer);
            tokenizer.set_template_kwargs(template_kwargs);
            let chat_template = chat_template.or(load_chat_template(sidecar_dir)?);
            return Ok(Self {
                model,
                #[cfg(feature = "media-processing")]
                processor: None,
                tokenizer,
                chat_template,
                model_id: model_dir.display().to_string(),
                eos_token_ids,
            });
        }
        let metadata = read_model_metadata(model_dir)?;
        let model_type = effective_model_type(&metadata);
        let kind = ModelKind::from_model_type(&model_type)?;
        let mut tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(model_dir)?);
        tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(model_dir)?);
        let chat_template = load_chat_template(model_dir)?;
        #[cfg(feature = "media-processing")]
        let processor = load_processor(model_dir)?;
        let model = match kind {
            ModelKind::PersonaPlex => {
                return Err(Error::UnsupportedArchitecture(
                    "PersonaPlex is a realtime speech-to-speech token model; use models::personaplex instead of LoadedModel".into(),
                ));
            }
            _ => load_model_for_kind(kind, model_dir, options, stream, weights_stream)?,
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
            #[cfg(feature = "media-processing")]
            processor,
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

    /// Returns whether this model directory includes a supported media processor.
    #[cfg(feature = "media-processing")]
    pub fn has_processor(&self) -> bool {
        self.processor.is_some()
    }

    /// Returns the loaded architecture-dispatched media processor, if available.
    #[cfg(feature = "media-processing")]
    pub fn processor(&self) -> Option<&ModelProcessor> {
        self.processor.as_ref()
    }

    /// Tokenizes and preprocesses ordered text and media segments.
    #[cfg(feature = "media-processing")]
    pub fn prepare_input(&self, input: &[ProcessorInput<'_>]) -> Result<PreparedModelInput, Error> {
        let processor = self.processor.as_ref().ok_or_else(|| {
            Error::Processor(format!(
                "model type '{}' does not have a loaded media processor",
                self.model_type()
            ))
        })?;
        processor.prepare_input(input, &mut |text| self.encode(text, false))
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

    /// Creates an empty cache value appropriate for the loaded model.
    pub fn new_cache(&self) -> ModelCache {
        self.model.new_cache()
    }

    /// Computes logits for an initial typed input using a cache returned by [`LoadedModel::new_cache`].
    pub fn prefill_input_with_cache(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.model.prefill_input_with_cache(input, cache, stream)
    }

    /// Computes initial logits from an owned processor result.
    #[cfg(feature = "media-processing")]
    pub fn prefill_prepared_input_with_cache(
        &mut self,
        input: &PreparedModelInput,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        input.with_model_input(|input| self.prefill_input_with_cache(input, cache, stream))
    }

    /// Computes initial prompt logits while reporting detailed activations.
    ///
    /// The returned logits have shape `[batch, vocab]` and match
    /// [`LoadedModel::prefill_input_with_cache`] for the same model/cache.
    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        self.model
            .prefill_input_with_observer(input, cache, stream, observer)
    }

    /// Computes initial logits from an owned processor result while observing activations.
    #[cfg(feature = "media-processing")]
    pub fn prefill_prepared_input_with_observer(
        &mut self,
        input: &PreparedModelInput,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        input.with_model_input(|input| {
            self.prefill_input_with_observer(input, cache, stream, observer)
        })
    }

    /// Creates a token iterator from typed input using a cache returned by [`LoadedModel::new_cache`].
    pub fn generate_input_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        self.model
            .generate_input_with_cache(cache, temp, input, prng_key, stream)
    }

    /// Creates a token iterator from typed input with a caller-provided sampler.
    pub fn generate_input_with_cache_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> ModelGenerate<'a, S>
    where
        S: Sampler,
    {
        self.model
            .generate_input_with_cache_sampler(cache, temp, input, prng_key, stream, sampler)
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

struct LoadedGgufModel {
    model: Model,
    eos_token_ids: Vec<u32>,
    chat_template: Option<String>,
    tokenizer: Option<GgufTokenizer>,
}

fn load_gguf_model_data(
    gguf_file: &Path,
    load_tokenizer: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedGgufModel, Error> {
    let (arrays, metadata) = Array::load_gguf_with_metadata(gguf_file, weights_stream)?;
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(architecture)) => architecture.clone(),
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"general.architecture\" has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata is missing required key \"general.architecture\"".into(),
            ));
        }
    };
    let chat_template = match metadata.get("tokenizer.chat_template") {
        Some(GgufMetadataValue::String(template)) => Some(template.clone()),
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"tokenizer.chat_template\" has the wrong type".into(),
            ));
        }
        None => None,
    };
    let tokenizer = load_tokenizer
        .then(|| load_gguf_tokenizer_from_metadata(gguf_file, &metadata))
        .transpose()?;

    let (model, eos_token_ids) = match architecture.as_str() {
        "gemma4" => {
            let loaded = gemma4::load_gemma4_gguf_data(arrays, metadata, stream, weights_stream)?;
            (Model::Gemma4(loaded.model), loaded.eos_token_ids)
        }
        "llama" | "mistral" => {
            let loaded = llama::load_llama_gguf_data(arrays, metadata, stream, weights_stream)?;
            (Model::Llama(loaded.model), loaded.eos_token_ids)
        }
        "nemotron_h" | "nemotron_h_moe" => {
            let loaded = nemotron_h::load_nemotron_h_gguf_data(
                arrays,
                metadata,
                stream,
                weights_stream,
            )?;
            (Model::NemotronH(loaded.model), loaded.eos_token_ids)
        }
        "qwen3" | "qwen3moe" => {
            let loaded = qwen3::load_qwen3_gguf_data(arrays, metadata, stream, weights_stream)?;
            (Model::Qwen3(loaded.model), loaded.eos_token_ids)
        }
        "qwen35" | "qwen35moe" => {
            let loaded = qwen3_5_moe::load_qwen3_5_moe_gguf_data(
                arrays,
                metadata,
                stream,
                weights_stream,
            )?;
            (Model::Qwen35Moe(loaded.model), loaded.eos_token_ids)
        }
        other => return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {other:?}; supported GGUF architectures are gemma4, llama, mistral, nemotron_h, nemotron_h_moe, qwen3, qwen3moe, qwen35, and qwen35moe"
        ))),
    };
    Ok(LoadedGgufModel {
        model,
        eos_token_ids,
        chat_template,
        tokenizer,
    })
}

/// Loads only the model weights and architecture from a model directory.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    load_model_with_options(
        model_dir,
        ModelLoadOptions::default(),
        stream,
        weights_stream,
    )
}

/// Loads only the model weights and architecture using shared load options.
pub fn load_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    if is_gguf_file(model_dir) {
        return Ok(load_gguf_model_data(model_dir, false, stream, weights_stream)?.model);
    }
    let metadata = read_model_metadata(model_dir)?;
    let kind = ModelKind::from_model_type(&effective_model_type(&metadata))?;
    match kind {
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex is a realtime speech-to-speech token model; use models::personaplex::load_model".into(),
        )),
        _ => load_model_for_kind(kind, model_dir, options, stream, weights_stream),
    }
}

fn load_model_for_kind(
    kind: ModelKind,
    model_dir: &Path,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    if let Some(quantization) = options.quantization {
        quantization.validate()?;
        return match kind {
            ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Llama => Ok(Model::Llama(llama::load_llama_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Qwen3 => Ok(Model::Qwen3(qwen3::load_qwen3_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Qwen3Vl => Err(Error::Quantization(
                "Qwen3-VL affine on-load quantization is not implemented; load the dense checkpoint".into(),
            )),
            ModelKind::NemotronH => Err(Error::Quantization(
                "Nemotron-H affine on-load quantization is unavailable because its routed experts use packed rank-3 grouped-matmul weights without an affine grouped-matmul implementation".into(),
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35Moe(
                qwen3_5_moe::load_qwen3_5_moe_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
                "PersonaPlex must be loaded through the realtime API".into(),
            )),
        };
    }

    match kind {
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
        ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(qwen3_vl::load_qwen3_vl_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen35Moe => Ok(Model::Qwen35Moe(qwen3_5_moe::load_qwen3_5_moe_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex must be loaded through the realtime API".into(),
        )),
    }
}

/// Loads only the tokenizer from a supported model directory or GGUF file.
///
/// Loading from GGUF parses embedded tokenizer metadata without creating an
/// MLX stream. A sibling `tokenizer.json` remains a fallback for missing or
/// unsupported embedded tokenizer formats.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let model_dir = model_dir.as_ref();
    if is_gguf_file(model_dir) {
        return Ok(load_gguf_tokenizer(model_dir)?.tokenizer);
    }
    let metadata = read_model_metadata(model_dir)?;
    match ModelKind::from_model_type(&effective_model_type(&metadata))? {
        ModelKind::Gemma4 => gemma4::load_gemma4_tokenizer(model_dir),
        ModelKind::Llama => llama::load_llama_tokenizer(model_dir),
        ModelKind::NemotronH => nemotron_h::load_nemotron_h_tokenizer(model_dir),
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex uses the released SentencePiece tokenizer; load it outside the chat tokenizer API".into(),
        )),
        ModelKind::Qwen3 => qwen3::load_qwen3_tokenizer(model_dir),
        ModelKind::Qwen3Vl => qwen3::load_qwen3_tokenizer(model_dir),
        ModelKind::Qwen35Moe => qwen3_5_moe::load_qwen3_5_moe_tokenizer(model_dir),
    }
}

/// Returns likely user-provided kwargs referenced by a model directory's chat template.
///
/// This reads tokenizer/chat-template metadata only and does not load model weights.
pub fn chat_template_kwargs(model_dir: impl AsRef<Path>) -> Result<Vec<String>, Error> {
    let submitted_path = model_dir.as_ref();
    let (template, model_id, tokenizer_template_kwargs) = if is_gguf_file(submitted_path) {
        let metadata = GgufMetadata::from_file(submitted_path)?;
        let sidecar_dir = gguf_sidecar_dir(submitted_path);
        let template = match metadata.get("tokenizer.chat_template") {
            Some(GgufMetadataValue::String(template)) => Some(template.clone()),
            Some(_) => {
                return Err(Error::GgufTokenizer(
                    "tokenizer.chat_template must be a string".into(),
                ));
            }
            None => load_chat_template(sidecar_dir)?,
        };
        let mut template_kwargs = gguf_tokenizer::template_kwargs(&metadata)?;
        template_kwargs.extend(load_tokenizer_template_kwargs(sidecar_dir)?);
        (
            template,
            submitted_path.display().to_string(),
            template_kwargs,
        )
    } else {
        (
            load_chat_template(submitted_path)?,
            submitted_path.display().to_string(),
            load_tokenizer_template_kwargs(submitted_path)?,
        )
    };
    let Some(template) = template else {
        return Ok(Vec::new());
    };
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

fn is_gguf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn gguf_sidecar_dir(path: &Path) -> &Path {
    path.parent().unwrap_or_else(|| Path::new("."))
}

fn load_gguf_tokenizer(gguf_file: &Path) -> Result<GgufTokenizer, Error> {
    let metadata = GgufMetadata::from_file(gguf_file)?;
    load_gguf_tokenizer_from_metadata(gguf_file, &metadata)
}

fn load_gguf_tokenizer_from_metadata(
    gguf_file: &Path,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
) -> Result<GgufTokenizer, Error> {
    let sidecar_dir = gguf_sidecar_dir(gguf_file);
    if let Some(mut embedded) = gguf_tokenizer::from_metadata(metadata)? {
        embedded
            .template_kwargs
            .extend(load_tokenizer_template_kwargs(sidecar_dir)?);
        return Ok(embedded);
    }
    Ok(GgufTokenizer {
        tokenizer: Tokenizer::from_file(sidecar_dir.join("tokenizer.json"))?,
        template_kwargs: load_tokenizer_template_kwargs(sidecar_dir)?,
    })
}

fn effective_model_type(metadata: &ModelMetadata) -> String {
    if matches!(
        metadata.model_type.as_str(),
        "gemma4" | "gemma4_unified" | "qwen3_vl" | "qwen3_5_moe"
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

    if !model_dir.join("config.json").exists() {
        return Ok(None);
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
        load_chat_template, load_model_with_options, load_tokenizer,
        load_tokenizer_template_kwargs, LoadedModel, ModelLoadOptions,
    };
    use crate::{
        error::Error,
        inspection::ActivationRecorder,
        quantization::{AffineQuantization, CheckpointQuantizationOptions},
    };
    use safemlx::{
        argmax_axis, module::ModuleParameters, Array, Device, DeviceType, ExecutionContext, Stream,
    };
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

    fn append_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn append_gguf_string_value(bytes: &mut Vec<u8>, key: &str, value: &str) {
        append_gguf_string(bytes, key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        append_gguf_string(bytes, value);
    }

    fn append_gguf_strings(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        append_gguf_string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            append_gguf_string(bytes, value);
        }
    }

    #[test]
    fn loads_tokenizer_directly_from_gguf_metadata() {
        let dir = temp_model_dir(r#"{"model_type":"qwen3"}"#);
        fs::remove_file(dir.join("tokenizer.json")).unwrap();
        let file = dir.join("embedded-tokenizer.gguf");
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&6u64.to_le_bytes());
        append_gguf_string_value(&mut bytes, "general.architecture", "qwen3");
        append_gguf_string_value(&mut bytes, "tokenizer.ggml.model", "gpt2");
        append_gguf_strings(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["<eos>", "h", "e", "l", "o", "he", "ll", "hell", "hello"],
        );
        append_gguf_strings(
            &mut bytes,
            "tokenizer.ggml.merges",
            &["h e", "l l", "he ll", "hell o"],
        );
        append_gguf_string(&mut bytes, "tokenizer.ggml.eos_token_id");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        append_gguf_string(&mut bytes, "tokenizer.ggml.add_eos_token");
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.push(1);
        fs::write(&file, bytes).unwrap();

        let tokenizer = load_tokenizer(&file).unwrap();
        assert_eq!(tokenizer.encode("hello", true).unwrap().get_ids(), &[8, 0]);

        fs::remove_dir_all(dir).unwrap();
    }

    fn save_zero_checkpoint<M: ModuleParameters>(
        model: &M,
        dir: &std::path::Path,
        stream: &Stream,
    ) {
        let parameters = model.parameters().flatten();
        let arrays = parameters
            .iter()
            .map(|(name, parameter)| {
                (
                    name.to_string(),
                    Array::zeros::<f32>(parameter.shape(), stream).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, array)| (name.as_str(), array)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
    }

    #[test]
    fn tiny_text_families_quantize_through_high_level_dispatch() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let fixtures = [
            (
                r#"{
                  "model_type":"llama","hidden_size":32,"num_hidden_layers":1,
                  "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                  "max_position_embeddings":128,"tie_word_embeddings":true,
                  "rope_scaling":null
                }"#,
                "llama",
            ),
            (
                r#"{
                  "model_type":"qwen3","hidden_size":32,"num_hidden_layers":1,
                  "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                  "max_position_embeddings":128,"rope_theta":10000.0,
                  "tie_word_embeddings":true,"rope_scaling":null
                }"#,
                "qwen3",
            ),
            (
                r#"{
                  "model_type":"gemma4",
                  "tie_word_embeddings":true,
                  "text_config":{
                    "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":1,
                    "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                    "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                    "max_position_embeddings":128,"tie_word_embeddings":true,
                    "attention_k_eq_v":false,"layer_types":["full_attention"]
                  }
                }"#,
                "gemma4",
            ),
        ];

        for (config, family) in fixtures {
            let dir = temp_model_dir(config);
            match family {
                "llama" => {
                    let args = super::llama::get_llama_model_args(&dir).unwrap();
                    save_zero_checkpoint(
                        &super::llama::Model::new(args, stream).unwrap(),
                        &dir,
                        stream,
                    );
                }
                "qwen3" => {
                    let args = super::qwen3::get_qwen3_model_args(&dir).unwrap();
                    save_zero_checkpoint(
                        &super::qwen3::Model::new(args, stream).unwrap(),
                        &dir,
                        stream,
                    );
                }
                "gemma4" => {
                    let args = super::gemma4::get_gemma4_model_args(&dir).unwrap();
                    save_zero_checkpoint(
                        &super::gemma4::Model::new(args, stream).unwrap(),
                        &dir,
                        stream,
                    );
                }
                _ => unreachable!(),
            }

            let mut dense =
                load_model_with_options(&dir, ModelLoadOptions::default(), stream, weights_stream)
                    .unwrap();
            let quantization = AffineQuantization::new(32, 4).unwrap();
            let mut quantized = load_model_with_options(
                &dir,
                ModelLoadOptions::with_quantization(quantization),
                stream,
                weights_stream,
            )
            .unwrap();
            let saved_dir = dir.with_extension("q4");
            crate::quantization::quantize_checkpoint(
                &dir,
                &saved_dir,
                &CheckpointQuantizationOptions {
                    quantization,
                    ..Default::default()
                },
                stream,
            )
            .unwrap();
            let mut saved_quantized = load_model_with_options(
                &saved_dir,
                ModelLoadOptions::with_quantization(quantization),
                stream,
                weights_stream,
            )
            .unwrap();
            let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
            let parts = [super::input::InputPart::text_token_ids(&tokens)];
            let input = super::input::ModelInput::new(&parts);
            let mut dense_cache = dense.new_cache();
            let dense_logits = dense
                .prefill_input_with_cache(input, &mut dense_cache, stream)
                .unwrap();
            let mut quantized_cache = quantized.new_cache();
            let quantized_logits = quantized
                .prefill_input_with_cache(input, &mut quantized_cache, stream)
                .unwrap();
            assert_eq!(dense_logits.shape(), quantized_logits.shape());
            let dense_token = argmax_axis!(&dense_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream);
            let quantized_token = argmax_axis!(&quantized_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream);
            assert_eq!(dense_token, quantized_token, "{family}");
            let mut saved_cache = saved_quantized.new_cache();
            let saved_logits = saved_quantized
                .prefill_input_with_cache(input, &mut saved_cache, stream)
                .unwrap();
            let saved_token = argmax_axis!(&saved_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream);
            assert_eq!(quantized_token, saved_token, "saved {family}");
            fs::remove_dir_all(dir).unwrap();
            fs::remove_dir_all(saved_dir).unwrap();
        }
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
    fn load_options_reject_unsupported_nemotron_packed_expert_affine_path() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let options = ModelLoadOptions::with_quantization(AffineQuantization::default());
        let dir = temp_model_dir(r#"{"model_type":"nemotron_h"}"#);
        let error = load_model_with_options(&dir, options, context.stream(), context.stream())
            .err()
            .expect("affine loading should be rejected before weight loading");
        assert!(matches!(error, Error::Quantization(_)));
        assert!(error.to_string().contains("packed rank-3"), "{error}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tiny_qwen35_moe_quantizes_packed_experts_through_high_level_dispatch() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let config = r#"{
          "model_type":"qwen3_5_moe",
          "tie_word_embeddings":false,
          "text_config":{
            "model_type":"qwen3_5_moe_text","vocab_size":32,"hidden_size":32,
            "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
            "head_dim":8,"max_position_embeddings":128,"rms_norm_eps":0.000001,
            "tie_word_embeddings":false,"attention_bias":false,"hidden_act":"silu",
            "moe_intermediate_size":32,"shared_expert_intermediate_size":32,
            "num_experts_per_tok":2,"num_experts":4,"norm_topk_prob":true,
            "layer_types":["full_attention"]
          }
        }"#;
        let dir = temp_model_dir(config);
        let (args, image_token_id, video_token_id, vision_config) =
            super::qwen3_5_moe::get_qwen3_5_moe_model_args(&dir).unwrap();
        save_zero_checkpoint(
            &super::qwen3_5_moe::Model::new(
                args,
                image_token_id,
                video_token_id,
                vision_config,
                stream,
            )
            .unwrap(),
            &dir,
            stream,
        );

        let mut dense =
            load_model_with_options(&dir, ModelLoadOptions::default(), stream, weights_stream)
                .unwrap();
        let mut quantized = load_model_with_options(
            &dir,
            ModelLoadOptions::with_quantization(AffineQuantization::new(32, 4).unwrap()),
            stream,
            weights_stream,
        )
        .unwrap();
        let super::Model::Qwen35Moe(quantized_model) = &quantized else {
            panic!("expected Qwen3.5-MoE model");
        };
        let params = quantized_model.parameters().flatten();
        let expert_weight = params
            .get("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap();
        assert_eq!(expert_weight.dtype(), safemlx::Dtype::Uint32);
        assert_eq!(expert_weight.shape(), &[4, 64, 4]);
        assert!(params.contains_key("model.layers.0.mlp.experts.gate_up_proj_scales"));
        assert!(params.contains_key("model.layers.0.mlp.experts.gate_up_proj_biases"));
        drop(params);

        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [super::input::InputPart::text_token_ids(&tokens)];
        let input = super::input::ModelInput::new(&parts);
        let mut dense_cache = dense.new_cache();
        let dense_logits = dense
            .prefill_input_with_cache(input, &mut dense_cache, stream)
            .unwrap();
        let mut quantized_cache = quantized.new_cache();
        let quantized_logits = quantized
            .prefill_input_with_cache(input, &mut quantized_cache, stream)
            .unwrap();
        assert_eq!(dense_logits.shape(), quantized_logits.shape());
        assert_eq!(
            argmax_axis!(&dense_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream),
            argmax_axis!(&quantized_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream)
        );
        fs::remove_dir_all(dir).unwrap();
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
    fn check_model_config_reports_supported_dense_mistral() {
        let support = check_model_config(&json!({
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32032,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": false
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Llama,
                model_type: "mistral".to_string(),
                effective_model_type: "mistral".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_supported_full_attention_mistral_small() {
        let support = check_model_config(&json!({
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "intermediate_size": 32768,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 0.00001,
            "vocab_size": 131072,
            "max_position_embeddings": 32768,
            "rope_theta": 100000000.0,
            "sliding_window": null,
            "tie_word_embeddings": false
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
    fn check_model_config_reports_supported_qwen3_vl() {
        let support = check_model_config(&json!({
            "model_type": "qwen3_vl",
            "image_token_id": 151655,
            "video_token_id": 151656,
            "text_config": {
                "model_type": "qwen3_vl_text",
                "vocab_size": 151936,
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "intermediate_size": 6144,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 0.000001,
                "max_position_embeddings": 262144,
                "rope_theta": 5000000.0,
                "tie_word_embeddings": true,
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20]
                }
            },
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024,
                "hidden_act": "gelu_pytorch_tanh",
                "intermediate_size": 4096,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "in_channels": 3,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 2048,
                "deepstack_visual_indexes": [5, 11, 17]
            }
        }));
        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Qwen3Vl,
                model_type: "qwen3_vl".to_string(),
                effective_model_type: "qwen3_vl_text".to_string(),
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
