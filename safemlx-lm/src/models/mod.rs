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
    ops::{GgufCheckpoint, GgufMetadata, GgufMetadataValue},
    random::RandomState,
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
use crate::models::common::generation::CausalLm;
use crate::parallel::ParallelTopology;
#[cfg(feature = "media-processing")]
use crate::processor::{load_processor, ModelProcessor, PreparedModelInput, ProcessorInput};
use crate::quantization::WeightQuantization;
use crate::sampler::{DefaultSampler, Sampler, SpeculativeSampler};
use crate::{
    cache::{ConcatKeyValueCache, PagedKeyValueCache, SlidingKeyValueCache},
    cache_residency::{
        open_prompt_cache, validate_prompt_cache_model_identity, CacheResidencyManager,
        CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    },
    error::Error,
    layerwise::{LayerExecutionLoadOptions, WeightResidency},
    mtp::{
        LoadedDrafter, MtpBatchOutput, MtpCache, MtpCapability, MtpCheckpointKind, MtpConfig,
        MtpStats,
    },
};

/// Shared building blocks used by multiple decoder-only model families.
pub mod common;
/// DeepSeek-V3 and DeepSeek-R1 decoder support.
pub mod deepseek_v3;
/// Gemma 4 text model support.
pub mod gemma4;
pub(crate) mod gemma4_assistant;
pub(crate) mod gemma4_audio;
pub(crate) mod gemma4_multimodal;
pub(crate) mod gemma4_vision;
/// OpenAI GPT-OSS sparse decoder architecture.
pub mod gpt_oss;
/// Thinking Machines Lab Inkling multimodal decoder support.
pub mod inkling;
/// Typed runtime input support.
pub mod input;
/// Liquid AI LFM2/LFM2.5 dense and MoE text model support.
pub mod lfm2;
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
/// Qwen3-Next hybrid attention/MoE text model support.
pub mod qwen3_next;
/// Qwen3-VL multimodal conditional-generation support.
pub mod qwen3_vl;
/// Qwen3-VL-MoE multimodal conditional-generation support.
pub mod qwen3_vl_moe;
pub(crate) mod qwen_vl;

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
    #[serde(default)]
    model_type: Option<String>,
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
    /// DeepSeek-V3/R1 MLA and MoE architecture.
    DeepSeekV3,
    /// Gemma 4 text architecture.
    Gemma4,
    /// OpenAI GPT-OSS MXFP4 sparse decoder architecture.
    GptOss,
    /// Thinking Machines Lab Inkling multimodal architecture.
    Inkling,
    /// Llama-compatible dense decoder architecture, including Mistral.
    Llama,
    /// Liquid AI LFM2/LFM2.5 dense or MoE architecture.
    Lfm2,
    /// Nemotron-H hybrid Mamba2/attention/MoE architecture.
    NemotronH,
    /// PersonaPlex realtime speech-to-speech architecture.
    PersonaPlex,
    /// Qwen3 decoder architecture.
    Qwen3,
    /// Qwen3-Next hybrid attention/MoE architecture.
    Qwen3Next,
    /// Qwen3-VL multimodal architecture.
    Qwen3Vl,
    /// Qwen3-VL multimodal architecture with a sparse MoE text decoder.
    Qwen3VlMoe,
    /// Qwen3.5 dense or mixture-of-experts architecture.
    Qwen35Moe,
}

/// Architecture-independent options for loading model weights.
///
/// When `quantization` is set for a dense checkpoint, eligible parameters are
/// quantized and materialized one tensor at a time. Checkpoints already
/// carrying matching metadata are loaded directly without requantizing.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ModelLoadOptions {
    /// Optional MLX weight encoding requested during dense checkpoint loading.
    pub quantization: Option<WeightQuantization>,
    /// Optional validated runtime topology and process-local device assignment.
    ///
    /// Singleton topologies preserve normal model loading. Non-replicated
    /// topologies must be loaded through the explicit [`crate::pipeline`],
    /// [`crate::tensor_parallel`], or [`crate::expert_parallel`] APIs.
    pub parallel: Option<ParallelTopology>,
    /// Parameter placement and execution policy for safetensors checkpoints.
    pub weight_residency: WeightResidency,
}

impl ModelLoadOptions {
    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: impl Into<WeightQuantization>) -> Self {
        Self {
            quantization: Some(quantization.into()),
            parallel: None,
            weight_residency: WeightResidency::FullyResident,
        }
    }

    /// Adds a validated runtime parallel topology to these options.
    pub fn with_parallel_topology(mut self, topology: ParallelTopology) -> Self {
        self.parallel = Some(topology);
        self
    }

    /// Creates load options for a validated runtime parallel topology.
    pub fn with_parallel(topology: ParallelTopology) -> Self {
        Self::default().with_parallel_topology(topology)
    }

    /// Selects fully resident or bounded layer execution for safetensors.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.weight_residency = residency;
        self
    }
}

pub(crate) fn ensure_executable_load_options(options: ModelLoadOptions) -> Result<(), Error> {
    if let Some(topology) = options
        .parallel
        .filter(|topology| !topology.is_replicated())
    {
        Err(Error::Parallel(
            if topology.tensor_parallel_size > 1
                && topology.pipeline_parallel_size == 1
                && topology.expert_parallel_size == 1
            {
                "non-replicated pure tensor-parallel loading cannot return the complete Model type; use tensor_parallel::load_tensor_parallel_model_with_options"
                    .into()
            } else if topology.pipeline_parallel_size > 1
                && topology.tensor_parallel_size == 1
                && topology.expert_parallel_size == 1
            {
                "non-replicated pure pipeline loading cannot return the complete Model type; use pipeline::load_pipeline_model_with_options"
                    .into()
            } else if topology.expert_parallel_size > 1
                && topology.tensor_parallel_size == 1
                && topology.pipeline_parallel_size == 1
            {
                "non-replicated pure expert-parallel loading cannot return the complete Model type; use expert_parallel::load_expert_parallel_model_with_options"
                    .into()
            } else {
                "hybrid TP+PP, TP+EP, and PP+EP model loading is unsupported; use a pure tensor-, pipeline-, or expert-parallel topology"
                    .into()
            },
        ))
    } else {
        Ok(())
    }
}

impl ModelKind {
    /// Returns a stable model-family name for diagnostics and capability dispatch.
    pub const fn model_type_name(self) -> &'static str {
        match self {
            Self::DeepSeekV3 => "deepseek_v3",
            Self::Gemma4 => "gemma4",
            Self::GptOss => "gpt_oss",
            Self::Inkling => "inkling_mm_model",
            Self::Llama => "llama/mistral",
            Self::Lfm2 => "lfm2/lfm2_moe",
            Self::NemotronH => "nemotron_h",
            Self::PersonaPlex => "personaplex",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Next => "qwen3_next",
            Self::Qwen3Vl => "qwen3_vl",
            Self::Qwen3VlMoe => "qwen3_vl_moe",
            Self::Qwen35Moe => "qwen3_5",
        }
    }

    fn from_model_type(model_type: &str) -> Result<Self, Error> {
        match model_type {
            "deepseek_v3" => Ok(Self::DeepSeekV3),
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => Ok(Self::Gemma4),
            "gpt_oss" => Ok(Self::GptOss),
            "inkling_mm_model" => Ok(Self::Inkling),
            "llama" | "mistral" => Ok(Self::Llama),
            "lfm2" | "lfm2_moe" => Ok(Self::Lfm2),
            "nemotron_h" => Ok(Self::NemotronH),
            "personaplex" => Ok(Self::PersonaPlex),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_next" => Ok(Self::Qwen3Next),
            "qwen3_vl" | "qwen3_vl_text" => Ok(Self::Qwen3Vl),
            "qwen3_vl_moe" | "qwen3_vl_moe_text" => Ok(Self::Qwen3VlMoe),
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35Moe),
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
        ModelKind::DeepSeekV3 => deepseek_v3::validate_model_config_value(config),
        ModelKind::Gemma4 => gemma4::validate_model_config_value(config),
        ModelKind::GptOss => gpt_oss::validate_model_config_value(config),
        ModelKind::Inkling => inkling::validate_model_config_value(config),
        ModelKind::Llama => llama::validate_model_config_value(config),
        ModelKind::Lfm2 => lfm2::validate_model_config_value(config),
        ModelKind::NemotronH => nemotron_h::validate_model_config_value(config),
        ModelKind::PersonaPlex => personaplex::validate_model_config_value(config),
        ModelKind::Qwen3 => {
            serde_json::from_value::<qwen3::ModelArgs>(config.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid qwen3 config: {error}"))
            })?;
            Ok(())
        }
        ModelKind::Qwen3Next => qwen3_next::validate_model_config_value(config),
        ModelKind::Qwen3Vl => qwen3_vl::validate_model_config_value(config),
        ModelKind::Qwen3VlMoe => qwen3_vl_moe::validate_model_config_value(config),
        ModelKind::Qwen35Moe => qwen3_5_moe::validate_model_config_value(config),
    }
}

/// Loaded model value for any architecture supported by this crate.
pub enum Model {
    /// DeepSeek-V3/R1 model.
    DeepSeekV3(deepseek_v3::Model),
    /// DeepSeek-V3/R1 model using bounded layer execution.
    DeepSeekV3Layerwise(crate::deepseek_v3::DeepSeekV3LayerwiseModel),
    /// Gemma 4 text model.
    Gemma4(gemma4::Model),
    /// Gemma 4 multimodal model using bounded layer execution.
    Gemma4Layerwise(crate::gemma4::Gemma4LayerwiseModel),
    /// OpenAI GPT-OSS model.
    GptOss(gpt_oss::Model),
    /// OpenAI GPT-OSS model using bounded layer execution.
    GptOssLayerwise(crate::gpt_oss::GptOssLayerwiseModel),
    /// Thinking Machines Lab Inkling model.
    Inkling(inkling::Model),
    /// Inkling multimodal model using bounded layer execution.
    InklingLayerwise(crate::inkling::InklingLayerwiseModel),
    /// Llama-compatible dense model.
    Llama(llama::ResidentModel),
    /// Llama-compatible model using the unified bounded layer API.
    LlamaLayerwise(crate::llama::LlamaModel),
    /// Liquid AI LFM2/LFM2.5 model.
    Lfm2(lfm2::Model),
    /// Liquid AI LFM2/LFM2.5 model using bounded layer execution.
    Lfm2Layerwise(crate::lfm2::Lfm2LayerwiseModel),
    /// Nemotron-H hybrid model.
    NemotronH(nemotron_h::Model),
    /// Nemotron-H hybrid model using bounded layer execution.
    NemotronHLayerwise(crate::nemotron_h::NemotronHLayerwiseModel),
    /// Qwen3 model.
    Qwen3(qwen3::Model),
    /// Qwen3 dense or sparse-MoE model using bounded layer execution.
    Qwen3Layerwise(crate::qwen3::Qwen3LayerwiseModel),
    /// Qwen3-Next model.
    Qwen3Next(qwen3_next::Model),
    /// Qwen3-Next model using shared hybrid bounded layer execution.
    Qwen3NextLayerwise(crate::qwen_hybrid::QwenHybridLayerwiseModel),
    /// Qwen3-VL multimodal model.
    Qwen3Vl(qwen3_vl::Model),
    /// Qwen3-VL multimodal model using vision/text bounded layer execution.
    Qwen3VlLayerwise(crate::qwen3_vl::Qwen3VlLayerwiseModel),
    /// Qwen3-VL-MoE multimodal model.
    Qwen3VlMoe(qwen3_vl_moe::Model),
    /// Qwen3-VL-MoE multimodal model using vision/text bounded layer execution.
    Qwen3VlMoeLayerwise(crate::qwen3_vl::Qwen3VlLayerwiseModel),
    /// Qwen3.5 dense or MoE model, optionally multimodal.
    Qwen35Moe(qwen3_5_moe::Model),
    /// Qwen3.5 model using shared vision/hybrid bounded layer execution.
    Qwen35MoeLayerwise(crate::qwen_hybrid::QwenHybridLayerwiseModel),
}

impl Model {
    /// Reports how this model architecture exposes MTP weights.
    pub fn mtp_capability(&self) -> MtpCapability {
        match self {
            Self::Gemma4(_) | Self::Gemma4Layerwise(_) => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Separate,
            },
            Self::DeepSeekV3(model) if model.args.num_nextn_predict_layers > 0 => {
                MtpCapability::Unsupported {
                    checkpoint: MtpCheckpointKind::Embedded,
                    architecture: "deepseek_v3".into(),
                }
            }
            Self::DeepSeekV3Layerwise(model) if model.args().num_nextn_predict_layers > 0 => {
                MtpCapability::Unsupported {
                    checkpoint: MtpCheckpointKind::Embedded,
                    architecture: "deepseek_v3".into(),
                }
            }
            Self::Inkling(_) | Self::InklingLayerwise(_) => MtpCapability::Unsupported {
                checkpoint: MtpCheckpointKind::Embedded,
                architecture: "inkling".into(),
            },
            Self::Qwen3Next(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen3NextLayerwise(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen35Moe(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen35MoeLayerwise(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::NemotronH(_) | Self::NemotronHLayerwise(_) => MtpCapability::Unsupported {
                checkpoint: MtpCheckpointKind::Embedded,
                architecture: "nemotron_h".into(),
            },
            _ => MtpCapability::Unavailable,
        }
    }

    /// Generates with MTP using the default lossless sampling policy.
    pub fn generate_mtp_input(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler(
            drafter,
            cache,
            input,
            config,
            prng_key,
            &DefaultSampler,
            stream,
        )
    }

    /// Generates with MTP using a caller-provided lossless sampling policy.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler<S: SpeculativeSampler>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler_callback(
            drafter,
            cache,
            input,
            config,
            prng_key,
            sampler,
            stream,
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        let assistant = drafter.gemma4_mut();
        match (self, cache) {
            (Self::Gemma4(target), ModelCache::Gemma4(cache)) => {
                validate_gemma4_drafter(&target.args, assistant)?;
                crate::gemma4_mtp::generate_with_callback(
                    target, assistant, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (Self::Gemma4Layerwise(target), ModelCache::Gemma4(cache)) => {
                validate_gemma4_drafter(target.args(), assistant)?;
                crate::gemma4_mtp::generate_with_callback(
                    target, assistant, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Generates with MTP weights embedded in the target checkpoint.
    pub fn generate_embedded_mtp_input(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler(
            cache,
            input,
            config,
            prng_key,
            &DefaultSampler,
            stream,
        )
    }

    /// Generates with embedded MTP weights and a caller-provided sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler<S: SpeculativeSampler>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler_callback(
            cache,
            input,
            config,
            prng_key,
            sampler,
            stream,
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        match (self, cache) {
            (Self::Qwen3Next(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35Moe(target), ModelCache::Qwen35Moe(cache)) => {
                crate::qwen_mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (Self::Qwen3NextLayerwise(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35MoeLayerwise(target), ModelCache::Qwen35Moe(cache)) => {
                crate::qwen_mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Returns residency telemetry when this model uses bounded layer execution.
    pub fn residency_report(&self) -> Result<Option<crate::residency::ResidencyReport>, Error> {
        match self {
            Self::DeepSeekV3Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Gemma4Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::InklingLayerwise(model) => Ok(Some(model.residency_report()?)),
            Self::LlamaLayerwise(model) => model.residency_report(),
            Self::GptOssLayerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Lfm2Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::NemotronHLayerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => {
                Ok(Some(model.residency_report()?))
            }
            Self::Qwen3Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3VlLayerwise(model) | Self::Qwen3VlMoeLayerwise(model) => {
                Ok(Some(model.residency_report()?))
            }
            _ => Ok(None),
        }
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::layerwise::DenseDiskStreamReport>, Error> {
        match self {
            Self::DeepSeekV3Layerwise(model) => model.dense_stream_report(),
            Self::Gemma4Layerwise(model) => model.dense_stream_report(),
            Self::InklingLayerwise(model) => model.dense_stream_report(),
            Self::LlamaLayerwise(model) => model.dense_stream_report(),
            Self::GptOssLayerwise(model) => model.dense_stream_report(),
            Self::Lfm2Layerwise(model) => model.dense_stream_report(),
            Self::NemotronHLayerwise(model) => model.dense_stream_report(),
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => {
                model.dense_stream_report()
            }
            Self::Qwen3Layerwise(model) => model.dense_stream_report(),
            Self::Qwen3VlLayerwise(model) | Self::Qwen3VlMoeLayerwise(model) => {
                model.dense_stream_report()
            }
            _ => Ok(None),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::expert_cache::ExpertCacheReport>, Error> {
        match self {
            Self::DeepSeekV3Layerwise(model) => model.expert_cache_report(),
            Self::GptOssLayerwise(model) => model.expert_cache_report(),
            Self::InklingLayerwise(model) => model.expert_cache_report(),
            Self::Lfm2Layerwise(model) => model.expert_cache_report(),
            Self::NemotronHLayerwise(model) => model.expert_cache_report(),
            Self::Qwen3Layerwise(model) => model.expert_cache_report(),
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => {
                model.expert_cache_report()
            }
            Self::Qwen3VlMoeLayerwise(model) => model.expert_cache_report(),
            _ => Ok(None),
        }
    }

    /// Returns the effective model type used for dispatch.
    pub fn model_type(&self) -> &str {
        match self {
            Self::DeepSeekV3(model) => model.model_type(),
            Self::DeepSeekV3Layerwise(model) => &model.args().model_type,
            Self::Gemma4(model) => model.model_type(),
            Self::Gemma4Layerwise(model) => &model.args().model_type,
            Self::GptOss(model) => model.model_type(),
            Self::GptOssLayerwise(model) => &model.args().model_type,
            Self::Inkling(model) => model.model_type(),
            Self::InklingLayerwise(model) => &model.args().model_type,
            Self::Llama(model) => model.model_type(),
            Self::LlamaLayerwise(model) => &model.args().model_type,
            Self::Lfm2(model) => model.model_type(),
            Self::Lfm2Layerwise(model) => &model.args().model_type,
            Self::NemotronH(model) => model.model_type(),
            Self::NemotronHLayerwise(model) => &model.args().model_type,
            Self::Qwen3(model) => model.model_type(),
            Self::Qwen3Layerwise(model) => &model.args().model_type,
            Self::Qwen3Next(model) => model.model_type(),
            Self::Qwen3NextLayerwise(model) => &model.args().model_type,
            Self::Qwen3Vl(model) => model.model_type(),
            Self::Qwen3VlLayerwise(model) => model.model_type(),
            Self::Qwen3VlMoe(model) => model.model_type(),
            Self::Qwen3VlMoeLayerwise(model) => model.model_type(),
            Self::Qwen35Moe(model) => model.model_type(),
            Self::Qwen35MoeLayerwise(model) => &model.args().model_type,
        }
    }

    /// Returns checkpoint-native quantization storage statistics when available.
    pub fn native_quantization_stats(
        &self,
    ) -> Option<&safemlx::native_quantization::NativeQuantizationStats> {
        match self {
            Self::Gemma4(model) => Some(&model.native_quantization_stats),
            _ => None,
        }
    }

    /// Returns the canonical cache-relevant architecture identity derived from the loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        match self {
            Self::Llama(model) => Ok(llama::prompt_cache_architecture_fingerprint(&model.args)),
            Self::LlamaLayerwise(model) => {
                Ok(llama::prompt_cache_architecture_fingerprint(model.args()))
            }
            Self::DeepSeekV3(model) => Ok(deepseek_v3::prompt_cache_architecture_fingerprint(
                &model.args,
            )),
            Self::DeepSeekV3Layerwise(model) => Ok(
                deepseek_v3::prompt_cache_architecture_fingerprint(model.args()),
            ),
            Self::GptOss(model) => Ok(gpt_oss::prompt_cache_architecture_fingerprint(&model.args)),
            Self::GptOssLayerwise(model) => {
                Ok(gpt_oss::prompt_cache_architecture_fingerprint(model.args()))
            }
            _ => Err(Exception::custom(format!(
                "prompt-cache architecture identity is unsupported for model type {}",
                self.model_type()
            ))),
        }
    }

    /// Runs a detailed instrumented forward pass for supported model families.
    ///
    /// DeepSeek-V3/R1, Llama, Qwen3, Qwen3.5 MoE, and Gemma4 currently report
    /// detailed layer activations. Other families return an error until their
    /// family-specific inspection paths are wired.
    pub fn forward_with_observer(
        &mut self,
        input_tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => model
                .forward_with_observer(
                    deepseek_v3::ModelInput {
                        inputs: input_tokens,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                    observer,
                ),
            (Self::DeepSeekV3Layerwise(_), ModelCache::DeepSeekV3(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer DeepSeek-V3 execution",
            )),
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
            (Self::Llama(_), ModelCache::PagedKeyValue(_)) => Err(Exception::custom(
                "detailed attention inspection is unavailable for paged key/value caches",
            )),
            (Self::LlamaLayerwise(_), ModelCache::LlamaLayerwise(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Llama execution",
            )),
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => model.forward_with_observer(
                qwen3::ModelInput {
                    inputs: input_tokens,
                    mask,
                    cache,
                },
                stream,
                observer,
            ),
            (Self::Qwen3Layerwise(_), ModelCache::KeyValue(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3 execution",
            )),
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
            (Self::Qwen35MoeLayerwise(_), ModelCache::Qwen35Moe(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3.5 execution",
            )),
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => model.forward_with_observer(
                qwen3_next::ModelInput {
                    inputs: input_tokens,
                    inputs_embeds: None,
                    mask,
                    cache: Some(cache),
                },
                stream,
                observer,
            ),
            (Self::Qwen3NextLayerwise(_), ModelCache::Qwen3Next(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3-Next execution",
            )),
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
            (Self::Gemma4Layerwise(_), ModelCache::Gemma4(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Gemma 4 execution",
            )),
            (Self::NemotronH(_) | Self::NemotronHLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            (Self::Lfm2(_) | Self::Lfm2Layerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for lfm2 yet",
            )),
            (Self::Qwen3Vl(_) | Self::Qwen3VlLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl yet",
            )),
            (Self::Qwen3VlMoe(_) | Self::Qwen3VlMoeLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl_moe yet",
            )),
            (Self::GptOss(_) | Self::GptOssLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for gpt_oss yet",
            )),
            (Self::Inkling(_) | Self::InklingLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for Inkling yet",
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
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    deepseek_v3::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache: Some(cache),
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::Gemma4Layerwise(_), ModelCache::Gemma4(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Gemma 4 execution",
            )),
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
            (Self::Llama(_), ModelCache::PagedKeyValue(_)) => Err(Exception::custom(
                "detailed attention inspection is unavailable for paged key/value caches",
            )),
            (Self::LlamaLayerwise(_), ModelCache::LlamaLayerwise(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Llama execution",
            )),
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
            (Self::Qwen3Layerwise(_), ModelCache::KeyValue(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3 execution",
            )),
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::Qwen35MoeLayerwise(_), ModelCache::Qwen35Moe(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3.5 execution",
            )),
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::Qwen3NextLayerwise(_), ModelCache::Qwen3Next(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3-Next execution",
            )),
            (Self::NemotronH(_) | Self::NemotronHLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            (Self::Lfm2(_) | Self::Lfm2Layerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for lfm2 yet",
            )),
            (Self::Inkling(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for Inkling yet",
            )),
            (Self::Qwen3Vl(_) | Self::Qwen3VlLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl yet",
            )),
            (Self::Qwen3VlMoe(_) | Self::Qwen3VlMoeLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl_moe yet",
            )),
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates an empty cache value appropriate for this model.
    pub fn new_cache(&self) -> ModelCache {
        match self {
            Self::DeepSeekV3(model) => ModelCache::DeepSeekV3(model.new_cache()),
            Self::DeepSeekV3Layerwise(model) => ModelCache::DeepSeekV3(model.new_cache()),
            Self::Gemma4(model) => ModelCache::Gemma4(model.new_cache()),
            Self::Gemma4Layerwise(model) => ModelCache::Gemma4(model.new_cache()),
            Self::GptOss(model) => ModelCache::GptOss(model.new_cache()),
            Self::GptOssLayerwise(model) => ModelCache::GptOss(model.new_cache()),
            Self::Inkling(model) => ModelCache::Inkling(model.new_cache()),
            Self::InklingLayerwise(model) => ModelCache::Inkling(model.new_cache()),
            Self::Llama(model) => match model.sliding_window() {
                Some(_) => ModelCache::SlidingKeyValue(model.new_sliding_cache()),
                None => ModelCache::KeyValue(Vec::new()),
            },
            Self::LlamaLayerwise(model) => ModelCache::LlamaLayerwise(model.new_cache()),
            Self::Lfm2(model) => ModelCache::Lfm2(model.new_cache()),
            Self::Lfm2Layerwise(model) => ModelCache::Lfm2(model.new_cache()),
            Self::Qwen3(_) => ModelCache::KeyValue(Vec::new()),
            Self::Qwen3Layerwise(model) => ModelCache::KeyValue(model.new_cache()),
            Self::Qwen3Next(model) => ModelCache::Qwen3Next(model.new_cache()),
            Self::Qwen3NextLayerwise(model) => ModelCache::Qwen3Next(model.new_cache()),
            Self::Qwen3Vl(model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::Qwen3VlLayerwise(model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::Qwen3VlMoe(model) => ModelCache::Qwen3VlMoe(model.new_cache()),
            Self::Qwen3VlMoeLayerwise(model) => ModelCache::Qwen3VlMoe(model.new_cache()),
            Self::NemotronH(model) => ModelCache::NemotronH(model.new_cache()),
            Self::NemotronHLayerwise(model) => ModelCache::NemotronH(model.new_cache()),
            Self::Qwen35Moe(model) => ModelCache::Qwen35Moe(model.new_cache()),
            Self::Qwen35MoeLayerwise(model) => ModelCache::Qwen35Moe(model.new_cache()),
        }
    }

    /// Creates ordinary cache state or an explicitly bounded paged cache.
    ///
    /// Paged construction is currently supported for Llama-compatible text
    /// attention, DeepSeek compressed-latent attention, GPT-OSS, Inkling
    /// relative-position attention, and the corresponding bounded
    /// weight-execution wrappers. Other cache representations return a precise
    /// unsupported error and retain their device-resident defaults.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ModelCache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => match self {
                Self::Llama(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Exception::custom(error.to_string()))?;
                    let layer_count = usize::try_from(model.args.num_hidden_layers)
                        .map_err(|_| Exception::custom("invalid Llama cache layer count"))?;
                    let caches = (0..layer_count)
                        .map(|layer| {
                            PagedKeyValueCache::new(manager.clone(), layer, model.sliding_window())
                                .map(Some)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(ModelCache::PagedKeyValue(caches))
                }
                Self::LlamaLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::LlamaLayerwise)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::DeepSeekV3(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeekV3),
                Self::DeepSeekV3Layerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeekV3)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::GptOss(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::GptOss),
                Self::GptOssLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::GptOss)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Inkling(model) => model.new_paged_cache(options).map(ModelCache::Inkling),
                Self::InklingLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Inkling)
                    .map_err(|error| Exception::custom(error.to_string())),
                _ => Err(Exception::custom(format!(
                    "paged cache residency is unsupported for model type {}",
                    self.model_type()
                ))),
            },
        }
    }

    /// Lazily catalogs a compatible persisted text prefix for a fresh cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(ModelCache, PromptCacheManifest), Exception> {
        match self {
            Self::Llama(model) => {
                let layer_count = usize::try_from(model.args.num_hidden_layers)
                    .map_err(|_| Exception::custom("invalid Llama cache layer count"))?;
                let identity = PromptCacheModelIdentity {
                        model_family: "llama".into(),
                        effective_model_type: model.args.model_type.clone(),
                        architecture_fingerprint:
                            llama::prompt_cache_architecture_fingerprint(&model.args),
                        layer_count,
                        global_layer_start: 0,
                        global_layer_end: layer_count,
                        sliding_window: model.sliding_window(),
                        sink_tokens: 0,
                        topology: Default::default(),
                        layer_layouts: PromptCacheModelIdentity::key_value_layouts(
                            layer_count,
                            model.args.num_key_value_heads,
                            model.args.head_dim,
                        ),
                    };
                validate_prompt_cache_model_identity(expected, &identity)
                .map_err(|error| Exception::custom(error.to_string()))?;
                let (manager, manifest) = open_prompt_cache(
                    directory,
                    expected,
                    &identity,
                    prefix_token_ids,
                    options,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let caches = (0..layer_count)
                    .map(|layer| {
                        PagedKeyValueCache::new(
                            manager.clone(),
                            layer,
                            model.sliding_window(),
                        )
                        .map(Some)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((ModelCache::PagedKeyValue(caches), manifest))
            }
            Self::LlamaLayerwise(model) => model
                .load_prompt_cache(
                    directory,
                    expected,
                    prefix_token_ids,
                    options,
                )
                .map(|(cache, manifest)| (ModelCache::LlamaLayerwise(cache), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV3(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::DeepSeekV3(cache), manifest)),
            Self::DeepSeekV3Layerwise(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::DeepSeekV3(cache), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            Self::GptOss(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::GptOss(cache), manifest)),
            Self::GptOssLayerwise(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::GptOss(cache), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            _ => Err(Exception::custom(
                "prompt-cache loading is unsupported for this model cache representation; multimodal and recurrent prefixes require additional identity state",
            )),
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
            (Self::Gemma4Layerwise(model), ModelCache::Gemma4(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::GptOss(model), ModelCache::GptOss(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::GptOssLayerwise(model), ModelCache::GptOss(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::InklingLayerwise(model), ModelCache::Inkling(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::PagedKeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::LlamaLayerwise(model), ModelCache::LlamaLayerwise(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Lfm2(model), ModelCache::Lfm2(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Lfm2Layerwise(model), ModelCache::Lfm2(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::NemotronHLayerwise(model), ModelCache::NemotronH(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Layerwise(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3VlLayerwise(model), ModelCache::Qwen3Vl(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3VlMoeLayerwise(model), ModelCache::Qwen3VlMoe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3NextLayerwise(model), ModelCache::Qwen3Next(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen35MoeLayerwise(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DeepSeekV3Layerwise(model), ModelCache::DeepSeekV3(cache)) => {
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
            (Self::Gemma4Layerwise(model), ModelCache::Gemma4(cache)) => {
                ModelGenerate::Gemma4Layerwise(crate::gemma4::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Lfm2(model), ModelCache::Lfm2(cache)) => ModelGenerate::Lfm2(
                lfm2::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::Lfm2Layerwise(model), ModelCache::Lfm2(cache)) => {
                ModelGenerate::Lfm2Layerwise(crate::lfm2::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::GptOss(model), ModelCache::GptOss(cache)) => {
                ModelGenerate::GptOss(gpt_oss::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::GptOssLayerwise(model), ModelCache::GptOss(cache)) => {
                ModelGenerate::GptOssLayerwise(crate::gpt_oss::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                ModelGenerate::Inkling(inkling::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::InklingLayerwise(model), ModelCache::Inkling(cache)) => {
                ModelGenerate::InklingLayerwise(crate::inkling::Generate::with_sampler(
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
            (Self::Llama(model), ModelCache::PagedKeyValue(cache)) => ModelGenerate::LlamaPaged(
                llama::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::LlamaLayerwise(model), ModelCache::LlamaLayerwise(cache)) => {
                ModelGenerate::LlamaLayerwise(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => ModelGenerate::Qwen3(
                qwen3::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::Qwen3Layerwise(model), ModelCache::KeyValue(cache)) => {
                ModelGenerate::Qwen3Layerwise(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
                ModelGenerate::Qwen3Vl(qwen3_vl::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3VlLayerwise(model), ModelCache::Qwen3Vl(cache)) => {
                ModelGenerate::Qwen3VlLayerwise(crate::qwen3_vl::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
                ModelGenerate::Qwen3VlMoe(qwen3_vl_moe::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3VlMoeLayerwise(model), ModelCache::Qwen3VlMoe(cache)) => {
                ModelGenerate::Qwen3VlMoeLayerwise(crate::qwen3_vl::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                ModelGenerate::NemotronH(nemotron_h::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::NemotronHLayerwise(model), ModelCache::NemotronH(cache)) => {
                ModelGenerate::NemotronHLayerwise(crate::nemotron_h::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                ModelGenerate::Qwen35Moe(qwen3_5_moe::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen35MoeLayerwise(model), ModelCache::Qwen35Moe(cache)) => {
                ModelGenerate::Qwen35MoeLayerwise(crate::qwen_hybrid::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => {
                ModelGenerate::Qwen3Next(qwen3_next::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3NextLayerwise(model), ModelCache::Qwen3Next(cache)) => {
                ModelGenerate::Qwen3NextLayerwise(crate::qwen_hybrid::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                ModelGenerate::DeepSeekV3(deepseek_v3::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::DeepSeekV3Layerwise(model), ModelCache::DeepSeekV3(cache)) => {
                ModelGenerate::DeepSeekV3Layerwise(crate::deepseek_v3::Generate::with_sampler(
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
    /// Compressed latent MLA cache for DeepSeek-V3/R1.
    DeepSeekV3(deepseek_v3::Cache),
    /// Gemma 4 generation cache.
    Gemma4(gemma4::Cache),
    /// Alternating full/sliding GPT-OSS cache.
    GptOss(gpt_oss::Cache),
    /// Alternating global/local attention and short-convolution Inkling cache.
    Inkling(inkling::Cache),
    /// Homogeneous per-layer key/value cache.
    KeyValue(Vec<Option<ConcatKeyValueCache>>),
    /// Unified Llama cache used by bounded layer execution.
    LlamaLayerwise(crate::llama::LlamaCache),
    /// Qwen3-VL key/value cache and multimodal position state.
    Qwen3Vl(qwen3_vl::Cache),
    /// Qwen3-VL-MoE key/value cache and multimodal position state.
    Qwen3VlMoe(qwen3_vl_moe::Cache),
    /// Homogeneous bounded cache for sliding-window attention.
    SlidingKeyValue(Vec<Option<SlidingKeyValueCache>>),
    /// Homogeneous block-addressable key/value cache under one global budget.
    PagedKeyValue(Vec<Option<PagedKeyValueCache>>),
    /// Heterogeneous Nemotron-H cache.
    NemotronH(nemotron_h::Cache),
    /// Heterogeneous LFM2 attention/convolution cache.
    Lfm2(lfm2::Cache),
    /// Heterogeneous Qwen3.5 MoE cache.
    Qwen35Moe(qwen3_5_moe::Cache),
    /// Heterogeneous Qwen3-Next cache.
    Qwen3Next(qwen3_next::Cache),
}

fn validate_gemma4_drafter(
    target: &gemma4::ModelArgs,
    assistant: &gemma4_assistant::Gemma4AssistantDraftModel,
) -> Result<(), Exception> {
    if assistant.config.model_type != "gemma4_assistant" {
        return Err(Exception::custom(format!(
            "expected a gemma4_assistant checkpoint, got {:?}",
            assistant.config.model_type
        )));
    }
    if assistant.config.backbone_hidden_size != target.hidden_size {
        return Err(Exception::custom(format!(
            "Gemma 4 assistant backbone hidden size {} does not match target hidden size {}",
            assistant.config.backbone_hidden_size, target.hidden_size
        )));
    }
    if assistant.config.text_config.vocab_size != target.vocab_size {
        return Err(Exception::custom(format!(
            "Gemma 4 assistant vocabulary size {} does not match target vocabulary size {}",
            assistant.config.text_config.vocab_size, target.vocab_size
        )));
    }
    if assistant.block_size() <= 1 {
        return Err(Exception::custom(
            "Gemma 4 assistant block_size must permit at least one draft token",
        ));
    }
    Ok(())
}

impl ModelCache {
    /// Returns aggregate cache-residency telemetry when paging is active.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        match self {
            Self::PagedKeyValue(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose(),
            Self::LlamaLayerwise(cache) => cache
                .residency_report()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV3(cache) => cache.residency_report(),
            Self::GptOss(cache) => cache.residency_report(),
            Self::Inkling(cache) => cache.residency_report(),
            _ => Ok(None),
        }
    }

    /// Finalizes and atomically saves a completed immutable text prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Exception> {
        match self {
            Self::PagedKeyValue(caches) => {
                for cache in caches.iter_mut().flatten() {
                    cache.finalize()?;
                }
                caches
                    .iter()
                    .flatten()
                    .next()
                    .ok_or_else(|| Exception::custom("cannot persist an empty paged cache"))?
                    .manager()
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
                    .map_err(|error| Exception::custom(error.to_string()))
            }
            Self::LlamaLayerwise(cache) => cache
                .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV3(cache) => {
                cache.save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            }
            Self::GptOss(cache) => {
                cache.save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            }
            _ => Err(Exception::custom(
                "prompt-cache persistence is unsupported for this model cache representation",
            )),
        }
    }
}

/// Token iterator for any supported model variant.
pub enum ModelGenerate<'a, S = DefaultSampler>
where
    S: Sampler,
{
    /// DeepSeek-V3/R1 generation iterator.
    DeepSeekV3(deepseek_v3::Generate<'a, S>),
    /// DeepSeek-V3/R1 generation using bounded layer execution.
    DeepSeekV3Layerwise(crate::deepseek_v3::Generate<'a, S>),
    /// Gemma 4 generation iterator.
    Gemma4(gemma4::Generate<'a, S>),
    /// Gemma 4 multimodal-prefill generation using bounded layer execution.
    Gemma4Layerwise(crate::gemma4::Generate<'a, S>),
    /// GPT-OSS generation iterator.
    GptOss(gpt_oss::Generate<'a, S>),
    /// GPT-OSS generation using bounded layer execution.
    GptOssLayerwise(crate::gpt_oss::Generate<'a, S>),
    /// Inkling generation iterator.
    Inkling(inkling::Generate<'a, S>),
    /// Inkling multimodal-prefill generation using bounded layer execution.
    InklingLayerwise(crate::inkling::Generate<'a, S>),
    /// Llama generation iterator.
    Llama(llama::Generate<'a, ConcatKeyValueCache, S>),
    /// Llama-compatible generation with bounded sliding-window caches.
    LlamaSliding(llama::Generate<'a, SlidingKeyValueCache, S>),
    /// Llama-compatible generation with block-addressable cache residency.
    LlamaPaged(llama::Generate<'a, PagedKeyValueCache, S>),
    /// Llama-compatible generation using bounded layer execution.
    LlamaLayerwise(
        common::generation::Generate<'a, crate::llama::LlamaModel, crate::llama::LlamaCache, S>,
    ),
    /// Qwen3 generation iterator.
    Qwen3(qwen3::Generate<'a, ConcatKeyValueCache, S>),
    /// Qwen3 generation using bounded layer execution.
    Qwen3Layerwise(
        common::generation::Generate<
            'a,
            crate::qwen3::Qwen3LayerwiseModel,
            Vec<Option<ConcatKeyValueCache>>,
            S,
        >,
    ),
    /// Qwen3-VL generation iterator.
    Qwen3Vl(qwen3_vl::Generate<'a, S>),
    /// Qwen3-VL generation using vision/text bounded layer execution.
    Qwen3VlLayerwise(crate::qwen3_vl::Generate<'a, S>),
    /// Qwen3-VL-MoE generation iterator.
    Qwen3VlMoe(qwen3_vl_moe::Generate<'a, S>),
    /// Qwen3-VL-MoE generation using vision/text bounded layer execution.
    Qwen3VlMoeLayerwise(crate::qwen3_vl::Generate<'a, S>),
    /// Nemotron-H generation iterator.
    NemotronH(nemotron_h::Generate<'a, S>),
    /// Nemotron-H generation using bounded layer execution.
    NemotronHLayerwise(crate::nemotron_h::Generate<'a, S>),
    /// LFM2 generation iterator.
    Lfm2(lfm2::Generate<'a, S>),
    /// LFM2 generation using bounded layer execution.
    Lfm2Layerwise(crate::lfm2::Generate<'a, S>),
    /// Qwen3.5 MoE generation iterator.
    Qwen35Moe(qwen3_5_moe::Generate<'a, S>),
    /// Qwen3.5 multimodal-prefill generation using shared bounded layer execution.
    Qwen35MoeLayerwise(crate::qwen_hybrid::Generate<'a, S>),
    /// Qwen3-Next generation iterator.
    Qwen3Next(qwen3_next::Generate<'a, S>),
    /// Qwen3-Next generation using shared hybrid bounded layer execution.
    Qwen3NextLayerwise(crate::qwen_hybrid::Generate<'a, S>),
}

impl<S> Iterator for ModelGenerate<'_, S>
where
    S: Sampler,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::DeepSeekV3(generate) => generate.next(),
            Self::DeepSeekV3Layerwise(generate) => generate.next(),
            Self::Gemma4(generate) => generate.next(),
            Self::Gemma4Layerwise(generate) => generate.next(),
            Self::GptOss(generate) => generate.next(),
            Self::GptOssLayerwise(generate) => generate.next(),
            Self::Inkling(generate) => generate.next(),
            Self::InklingLayerwise(generate) => generate.next(),
            Self::Llama(generate) => generate.next(),
            Self::LlamaSliding(generate) => generate.next(),
            Self::LlamaPaged(generate) => generate.next(),
            Self::LlamaLayerwise(generate) => generate.next(),
            Self::Lfm2(generate) => generate.next(),
            Self::Lfm2Layerwise(generate) => generate.next(),
            Self::NemotronH(generate) => generate.next(),
            Self::NemotronHLayerwise(generate) => generate.next(),
            Self::Qwen3(generate) => generate.next(),
            Self::Qwen3Layerwise(generate) => generate.next(),
            Self::Qwen3Vl(generate) => generate.next(),
            Self::Qwen3VlLayerwise(generate) => generate.next(),
            Self::Qwen3VlMoe(generate) => generate.next(),
            Self::Qwen3VlMoeLayerwise(generate) => generate.next(),
            Self::Qwen35Moe(generate) => generate.next(),
            Self::Qwen35MoeLayerwise(generate) => generate.next(),
            Self::Qwen3Next(generate) => generate.next(),
            Self::Qwen3NextLayerwise(generate) => generate.next(),
        }
    }
}

/// Stateful tokenizer decoder for incrementally generated token ids.
///
/// Unlike decoding each token independently, this preserves tokenizer context
/// and buffers incomplete byte-fallback sequences until they form valid text.
pub struct TextDecoder {
    tokenizer: Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl TextDecoder {
    /// Decodes one token, returning text only when the token completes a chunk.
    pub fn step(&mut self, id: u32) -> Result<Option<String>, Error> {
        tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer,
            vec![id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(Into::into)
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
    /// Creates an independent stateful decoder for streaming generated tokens.
    pub fn text_decoder(&self, skip_special_tokens: bool) -> TextDecoder {
        TextDecoder {
            tokenizer: (*self.tokenizer).clone(),
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// Reports whether and how this target can perform MTP generation.
    pub fn mtp_capability(&self) -> MtpCapability {
        self.model.mtp_capability()
    }

    /// Creates independent target caches for an MTP text batch.
    pub fn new_mtp_cache(&self, batch_size: usize) -> MtpCache {
        MtpCache::new((0..batch_size).map(|_| self.new_cache()).collect())
    }

    /// Generates through the architecture-independent MTP path.
    pub fn generate_mtp_input(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler(
            drafter,
            cache,
            input,
            config,
            prng_key,
            &DefaultSampler,
            stream,
        )
    }

    /// Generates through MTP with a caller-provided speculative sampling policy.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler<S: SpeculativeSampler>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        self.model.generate_mtp_input_with_sampler(
            drafter, cache, input, &config, prng_key, sampler, stream,
        )
    }

    /// Generates through MTP and reports each committed token as it becomes available.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        self.model.generate_mtp_input_with_sampler_callback(
            drafter, cache, input, &config, prng_key, sampler, stream, on_token,
        )
    }

    /// Generates through MTP weights embedded in the target checkpoint.
    pub fn generate_embedded_mtp_input(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler(
            cache,
            input,
            config,
            prng_key,
            &DefaultSampler,
            stream,
        )
    }

    /// Generates through embedded MTP weights with a caller-provided sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler<S: SpeculativeSampler>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        self.model.generate_embedded_mtp_input_with_sampler(
            cache, input, &config, prng_key, sampler, stream,
        )
    }

    /// Generates through embedded MTP and reports each committed token as it becomes available.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        self.model
            .generate_embedded_mtp_input_with_sampler_callback(
                cache, input, &config, prng_key, sampler, stream, on_token,
            )
    }

    /// Generates an independently accepting and stopping batch of text prompts.
    ///
    /// Each lane owns a separate cache so rejection lengths and EOS positions
    /// may diverge without padding rejected state back into another sequence.
    pub fn generate_mtp_text_batch<S: SpeculativeSampler>(
        &mut self,
        drafter: &mut LoadedDrafter,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        let batch_size = if prompt_tokens.ndim() == 2 {
            prompt_tokens.dim(0) as usize
        } else {
            0
        };
        let mut cache = self.new_mtp_cache(batch_size);
        self.generate_mtp_text_batch_with_cache(
            drafter,
            &mut cache,
            prompt_tokens,
            config,
            prng_key,
            sampler,
            stream,
        )
    }

    /// Generates a text batch using reusable independent per-lane caches.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_text_batch_with_cache<S: SpeculativeSampler>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut MtpCache,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        if prompt_tokens.ndim() != 2 || prompt_tokens.dim(1) == 0 {
            return Err(Exception::custom(format!(
                "MTP text batch must be shaped [batch, nonzero sequence], got {:?}",
                prompt_tokens.shape()
            )));
        }
        if cache.len() != prompt_tokens.dim(0) as usize {
            return Err(Exception::custom(format!(
                "MTP cache has {} lanes but text input has batch size {}",
                cache.len(),
                prompt_tokens.dim(0)
            )));
        }
        if config.temperature != 0.0 && prng_key.is_none() {
            return Err(Exception::custom(
                "random operations require an explicit PRNG key",
            ));
        }
        let mut batch_prng = prng_key.map(RandomState::from_key);
        let mut output = MtpBatchOutput::default();
        for lane in 0..prompt_tokens.dim(0) {
            let row = prompt_tokens.try_index_device((lane, NewAxis, ..), stream)?;
            let lane_key = batch_prng
                .as_mut()
                .map(|state| state.next_key(stream))
                .transpose()?;
            let parts = [input::InputPart::text_token_ids(&row)];
            let input = input::ModelInput::new(&parts);
            let (tokens, stats) = self.generate_mtp_input_with_sampler(
                drafter,
                &mut cache.lanes[lane as usize],
                input,
                config,
                lane_key,
                sampler,
                stream,
            )?;
            output.token_ids.push(tokens);
            output.stats.push(stats);
        }
        Ok(output)
    }

    /// Generates an independently accepting text batch with embedded MTP weights.
    pub fn generate_embedded_mtp_text_batch<S: SpeculativeSampler>(
        &mut self,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        let batch_size = if prompt_tokens.ndim() == 2 {
            prompt_tokens.dim(0) as usize
        } else {
            0
        };
        let mut cache = self.new_mtp_cache(batch_size);
        self.generate_embedded_mtp_text_batch_with_cache(
            &mut cache,
            prompt_tokens,
            config,
            prng_key,
            sampler,
            stream,
        )
    }

    /// Generates a text batch with embedded MTP weights and reusable lane caches.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_text_batch_with_cache<S: SpeculativeSampler>(
        &mut self,
        cache: &mut MtpCache,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        if prompt_tokens.ndim() != 2 || prompt_tokens.dim(1) == 0 {
            return Err(Exception::custom(format!(
                "MTP text batch must be shaped [batch, nonzero sequence], got {:?}",
                prompt_tokens.shape()
            )));
        }
        if cache.len() != prompt_tokens.dim(0) as usize {
            return Err(Exception::custom(format!(
                "MTP cache has {} lanes but text input has batch size {}",
                cache.len(),
                prompt_tokens.dim(0)
            )));
        }
        if config.temperature != 0.0 && prng_key.is_none() {
            return Err(Exception::custom(
                "random operations require an explicit PRNG key",
            ));
        }
        let mut batch_prng = prng_key.map(RandomState::from_key);
        let mut output = MtpBatchOutput::default();
        for lane in 0..prompt_tokens.dim(0) {
            let row = prompt_tokens.try_index_device((lane, NewAxis, ..), stream)?;
            let lane_key = batch_prng
                .as_mut()
                .map(|state| state.next_key(stream))
                .transpose()?;
            let parts = [input::InputPart::text_token_ids(&row)];
            let input = input::ModelInput::new(&parts);
            let (tokens, stats) = self.generate_embedded_mtp_input_with_sampler(
                &mut cache.lanes[lane as usize],
                input,
                config,
                lane_key,
                sampler,
                stream,
            )?;
            output.token_ids.push(tokens);
            output.stats.push(stats);
        }
        Ok(output)
    }

    /// Returns residency telemetry when bounded layer execution was selected.
    pub fn residency_report(&self) -> Result<Option<crate::residency::ResidencyReport>, Error> {
        self.model.residency_report()
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::layerwise::DenseDiskStreamReport>, Error> {
        self.model.dense_stream_report()
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::expert_cache::ExpertCacheReport>, Error> {
        self.model.expert_cache_report()
    }

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
        ensure_executable_load_options(options)?;
        if is_gguf_file(model_dir) {
            if !matches!(options.weight_residency, WeightResidency::FullyResident) {
                return Err(Error::UnsupportedArchitecture(
                    "host-backed weight residency requires safetensors; GGUF is unsupported".into(),
                ));
            }
            let sidecar_dir = gguf_sidecar_dir(model_dir);
            let LoadedGgufModel {
                model,
                eos_token_ids,
                chat_template,
                tokenizer,
            } = load_gguf_model_data(model_dir, true, options, stream, weights_stream)?;
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

    /// Returns checkpoint-native quantization storage statistics when available.
    pub fn native_quantization_stats(
        &self,
    ) -> Option<&safemlx::native_quantization::NativeQuantizationStats> {
        self.model.native_quantization_stats()
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

    /// Creates cache state under an explicit cache-residency policy.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ModelCache, Exception> {
        self.model.new_cache_with_options(policy)
    }

    /// Returns the canonical cache-relevant architecture identity for this loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        self.model.prompt_cache_architecture_fingerprint()
    }

    /// Lazily catalogs a compatible reusable text prefix for this loaded model.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(ModelCache, PromptCacheManifest), Exception> {
        self.model
            .load_prompt_cache(directory, expected, prefix_token_ids, options)
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
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedGgufModel, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = crate::weights::gguf_metadata(&checkpoint);
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
    validate_gguf_quantization_source(&checkpoint, &metadata, options.quantization)?;

    let (model, eos_token_ids) = match architecture.as_str() {
        "deepseek2" => {
            let loaded = deepseek_v3::load_gguf_checkpoint(
                &checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::DeepSeekV3(loaded.model), loaded.eos_token_ids)
        }
        "gemma4" => {
            let loaded = gemma4::load_gemma4_gguf_checkpoint(
                &checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Gemma4(loaded.model), loaded.eos_token_ids)
        }
        "llama" | "mistral" => {
            let loaded = llama::load_llama_gguf_checkpoint(
                &checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Llama(loaded.model), loaded.eos_token_ids)
        }
        "lfm2" | "lfm2moe" => {
            let loaded = lfm2::load_gguf_checkpoint(
                &checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Lfm2(loaded.model), loaded.eos_token_ids)
        }
        "nemotron_h" | "nemotron_h_moe" => {
            if options.quantization.is_some() {
                return Err(Error::Quantization(
                    "Nemotron-H load-time quantization is unavailable for dense safetensors and GGUF inputs"
                        .into(),
                ));
            }
            let loaded = nemotron_h::load_nemotron_h_gguf_checkpoint(
                &checkpoint,
                metadata,
                stream,
                weights_stream,
            )?;
            (Model::NemotronH(loaded.model), loaded.eos_token_ids)
        }
        "qwen3" | "qwen3moe" => {
            let loaded = qwen3::load_qwen3_gguf_checkpoint(
                &checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Qwen3(loaded.model), loaded.eos_token_ids)
        }
        "qwen3vl" => {
            let mmproj_file = qwen3_vl::find_qwen3_vl_mmproj(gguf_file)?;
            let vision_checkpoint = GgufCheckpoint::open(mmproj_file)?;
            let vision_metadata = crate::weights::gguf_metadata(&vision_checkpoint);
            let loaded = qwen3_vl::load_qwen3_vl_gguf_checkpoint(
                &checkpoint,
                metadata,
                &vision_checkpoint,
                vision_metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Qwen3Vl(loaded.model), loaded.eos_token_ids)
        }
        "qwen35" | "qwen35moe" | "qwen3next" => {
            let loaded = qwen3_5_moe::load_qwen3_5_moe_gguf_checkpoint(
                &checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let model = if architecture == "qwen3next" {
                Model::Qwen3Next(loaded.model)
            } else {
                Model::Qwen35Moe(loaded.model)
            };
            (model, loaded.eos_token_ids)
        }
        other => return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {other:?}; supported GGUF architectures are deepseek2, gemma4, llama, mistral, lfm2, lfm2moe, nemotron_h, nemotron_h_moe, qwen3, qwen3moe, qwen3vl, qwen35, qwen35moe, and qwen3next"
        ))),
    };
    Ok(LoadedGgufModel {
        model,
        eos_token_ids,
        chat_template,
        tokenizer,
    })
}

fn validate_gguf_quantization_source<S: crate::weights::GgufTensorNames>(
    source: &S,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
) -> Result<(), Error> {
    let Some(quantization) = quantization else {
        return Ok(());
    };
    quantization.validate()?;

    let has_packed_companions = source.has_affine_gguf_tensor();
    if has_packed_companions {
        return Err(Error::Quantization(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; packed GGUF tensors cannot be implicitly transcoded"
                .into(),
        ));
    }

    let file_type = metadata
        .get("general.file_type")
        .ok_or_else(|| {
            Error::Quantization(
                "GGUF general.file_type metadata is required to verify that load-time quantization is not transcoding packed weights"
                    .into(),
            )
        })?
        .as_i64()
        .ok_or_else(|| {
            Error::Quantization("GGUF general.file_type metadata must be an integer".into())
        })?;
    // llama.cpp's unquantized file types: ALL_F32, MOSTLY_F16, and MOSTLY_BF16.
    if !matches!(file_type, 0 | 1 | 32) {
        return Err(Error::Quantization(format!(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; general.file_type={file_type} is already quantized"
        )));
    }
    Ok(())
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
    ensure_executable_load_options(options)?;
    if is_gguf_file(model_dir) {
        if !matches!(options.weight_residency, WeightResidency::FullyResident) {
            return Err(Error::UnsupportedArchitecture(
                "host-backed weight residency requires safetensors; GGUF is unsupported".into(),
            ));
        }
        return Ok(load_gguf_model_data(model_dir, false, options, stream, weights_stream)?.model);
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
    ensure_executable_load_options(options)?;
    if let WeightResidency::SparseExpertCacheWithDenseLayers(combined) = options.weight_residency {
        if options.quantization.is_some() {
            return Err(Error::Quantization(format!(
                "load-time quantization is unsupported for {} sparse expert caching with dense disk streaming; use a matching checkpoint-native packed format",
                kind.model_type_name()
            )));
        }
        let expert_cache = combined.expert_cache;
        let non_expert = combined.non_expert;
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3Layerwise(
                crate::deepseek_v3::load_deepseek_v3_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::GptOss => Ok(Model::GptOssLayerwise(
                crate::gpt_oss::load_gpt_oss_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::InklingLayerwise(
                crate::inkling::load_inkling_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2Layerwise(
                crate::lfm2::load_lfm2_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronHLayerwise(
                crate::nemotron_h::load_nemotron_h_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3 => Ok(Model::Qwen3Layerwise(
                crate::qwen3::load_qwen3_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3NextLayerwise(
                crate::qwen_hybrid::load_qwen3_next_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoeLayerwise(
                crate::qwen3_vl::load_qwen3_vl_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35MoeLayerwise(
                crate::qwen_hybrid::load_qwen35_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "sparse expert caching with dense disk streaming requires a supported safetensors MoE architecture, not {}",
                kind.model_type_name()
            ))),
        };
    }
    if let WeightResidency::SparseExpertCache(expert_cache) = options.weight_residency {
        if options.quantization.is_some() {
            return Err(Error::Quantization(format!(
                "load-time quantization is unsupported for {} sparse expert caching; use a matching checkpoint-native packed format",
                kind.model_type_name()
            )));
        }
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3Layerwise(
                crate::deepseek_v3::load_deepseek_v3_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::GptOss => Ok(Model::GptOssLayerwise(
                crate::gpt_oss::load_gpt_oss_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::InklingLayerwise(
                crate::inkling::load_inkling_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2Layerwise(
                crate::lfm2::load_lfm2_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronHLayerwise(
                crate::nemotron_h::load_nemotron_h_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3 => Ok(Model::Qwen3Layerwise(
                crate::qwen3::load_qwen3_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3NextLayerwise(
                crate::qwen_hybrid::load_qwen3_next_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoeLayerwise(
                crate::qwen3_vl::load_qwen3_vl_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35MoeLayerwise(
                crate::qwen_hybrid::load_qwen35_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "sparse expert caching requires a supported safetensors MoE architecture, not {}",
                kind.model_type_name()
            ))),
        };
    }
    let layerwise: Option<LayerExecutionLoadOptions> = match options.weight_residency {
        WeightResidency::LayerwiseHost(options) => Some(options.into()),
        WeightResidency::DenseDiskStream(options) => Some(options.into()),
        _ => None,
    };
    if let Some(layerwise) = layerwise {
        if options.quantization.is_some() {
            return Err(Error::Quantization(format!(
                "load-time quantization is unsupported for {} layer streaming; use a matching checkpoint-native packed format",
                kind.model_type_name()
            )));
        }
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3Layerwise(
                crate::deepseek_v3::load_deepseek_v3_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Gemma4 => Ok(Model::Gemma4Layerwise(
                crate::gemma4::load_gemma4_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::InklingLayerwise(
                crate::inkling::load_inkling_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Llama => Ok(Model::LlamaLayerwise(crate::llama::load_llama_model(
                model_dir,
                crate::llama::LlamaLoadOptions {
                    weight_residency: match layerwise {
                        LayerExecutionLoadOptions::LayerwiseHost(options) => {
                            WeightResidency::LayerwiseHost(options)
                        }
                        LayerExecutionLoadOptions::DenseDiskStream(options) => {
                            WeightResidency::DenseDiskStream(options)
                        }
                    },
                },
                stream,
                weights_stream,
            )?)),
            ModelKind::Qwen3 => Ok(Model::Qwen3Layerwise(
                crate::qwen3::load_qwen3_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::GptOss => Ok(Model::GptOssLayerwise(
                crate::gpt_oss::load_gpt_oss_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2Layerwise(
                crate::lfm2::load_lfm2_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronHLayerwise(
                crate::nemotron_h::load_nemotron_h_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3NextLayerwise(
                crate::qwen_hybrid::load_qwen3_next_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Vl => Ok(Model::Qwen3VlLayerwise(
                crate::qwen3_vl::load_qwen3_vl_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoeLayerwise(
                crate::qwen3_vl::load_qwen3_vl_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35MoeLayerwise(
                crate::qwen_hybrid::load_qwen35_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
                "PersonaPlex bounded layer residency is selected through the realtime loader"
                    .into(),
            )),
        };
    }
    if let Some(quantization) = options.quantization {
        quantization.validate()?;
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(
                deepseek_v3::load_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::GptOss => Ok(Model::GptOss(gpt_oss::load_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Inkling => Err(Error::Quantization(
                "Inkling affine/MXFP4 on-load quantization is unavailable because its routed experts use packed rank-3 grouped-matmul weights without a matching quantized grouped-matmul implementation".into(),
            )),
            ModelKind::Llama => Ok(Model::Llama(llama::load_resident_llama_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Lfm2 => Ok(Model::Lfm2(lfm2::load_model_quantized(
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
            ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
                qwen3_next::load_qwen3_next_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(
                qwen3_vl::load_qwen3_vl_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
                qwen3_vl_moe::load_qwen3_vl_moe_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
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
        ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(deepseek_v3::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::GptOss => Ok(Model::GptOss(gpt_oss::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Inkling => Ok(Model::Inkling(inkling::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Llama => Ok(Model::Llama(llama::load_resident_llama_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Lfm2 => Ok(Model::Lfm2(lfm2::load_model(
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
        ModelKind::Qwen3Next => Ok(Model::Qwen3Next(qwen3_next::load_qwen3_next_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(qwen3_vl::load_qwen3_vl_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(qwen3_vl_moe::load_qwen3_vl_moe_model(
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
        ModelKind::DeepSeekV3 => deepseek_v3::load_tokenizer(model_dir),
        ModelKind::Gemma4 => gemma4::load_gemma4_tokenizer(model_dir),
        ModelKind::GptOss => gpt_oss::load_tokenizer(model_dir),
        ModelKind::Inkling => inkling::load_tokenizer(model_dir),
        ModelKind::Llama => llama::load_llama_tokenizer(model_dir),
        ModelKind::Lfm2 => lfm2::load_tokenizer(model_dir),
        ModelKind::NemotronH => nemotron_h::load_nemotron_h_tokenizer(model_dir),
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex uses the released SentencePiece tokenizer; load it outside the chat tokenizer API".into(),
        )),
        ModelKind::Qwen3 => qwen3::load_qwen3_tokenizer(model_dir),
        ModelKind::Qwen3Next => qwen3_next::load_qwen3_next_tokenizer(model_dir),
        ModelKind::Qwen3Vl => qwen3::load_qwen3_tokenizer(model_dir),
        ModelKind::Qwen3VlMoe => qwen3::load_qwen3_tokenizer(model_dir),
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
    if metadata.model_type == "inkling_mm_model" {
        return metadata.model_type.clone();
    }
    if matches!(
        metadata.model_type.as_str(),
        "gemma4" | "gemma4_unified" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe"
    ) {
        metadata
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    } else if ModelKind::from_model_type(&metadata.model_type).is_ok() {
        metadata.model_type.clone()
    } else {
        metadata
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.model_type.clone())
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
                text_config.model_type.as_deref(),
                Some("gemma4_text" | "gemma4_unified_text")
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
        load_tokenizer_template_kwargs, validate_gguf_quantization_source, LoadedModel,
        ModelLoadOptions,
    };
    use crate::{
        error::Error,
        inspection::ActivationRecorder,
        quantization::{AffineQuantization, CheckpointQuantizationOptions, WeightQuantization},
    };
    use safemlx::{
        argmax_axis,
        module::ModuleParameters,
        ops::{zeros_dtype, GgufMetadataValue},
        Array, Device, DeviceType, ExecutionContext, Stream,
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

    #[test]
    fn load_time_quantization_accepts_only_unquantized_gguf_sources() {
        let dense_metadata = std::collections::HashMap::from([(
            "general.file_type".into(),
            GgufMetadataValue::Uint32(1),
        )]);
        validate_gguf_quantization_source(
            &std::collections::HashMap::new(),
            &dense_metadata,
            Some(WeightQuantization::MxFp4),
        )
        .unwrap();

        let error = validate_gguf_quantization_source(
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            Some(WeightQuantization::MxFp4),
        )
        .unwrap_err();
        assert!(error.to_string().contains("general.file_type"));

        let quantized_metadata = std::collections::HashMap::from([(
            "general.file_type".into(),
            GgufMetadataValue::Uint32(7),
        )]);
        let error = validate_gguf_quantization_source(
            &std::collections::HashMap::new(),
            &quantized_metadata,
            Some(WeightQuantization::MxFp4),
        )
        .unwrap_err();
        assert!(error.to_string().contains("already quantized"));

        let packed_arrays = std::collections::HashMap::from([(
            "blk.0.attn_q.scales".into(),
            Array::from_slice(&[1.0f32], &[1]),
        )]);
        let error = validate_gguf_quantization_source(
            &packed_arrays,
            &dense_metadata,
            Some(WeightQuantization::MxFp4),
        )
        .unwrap_err();
        assert!(error.to_string().contains("packed GGUF tensors"));
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
                    zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
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
                  "model_type":"mistral","hidden_size":32,"num_hidden_layers":1,
                  "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                  "max_position_embeddings":128,"sliding_window":16,
                  "tie_word_embeddings":true,"rope_scaling":null
                }"#,
                "mistral",
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
                  "model_type":"qwen3_5_text","vocab_size":32,"hidden_size":32,
                  "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"max_position_embeddings":128,"rms_norm_eps":0.000001,
                  "tie_word_embeddings":true,"attention_bias":false,"hidden_act":"silu",
                  "intermediate_size":64,"layer_types":["full_attention"]
                }"#,
                "qwen3_5",
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
                "llama" | "mistral" => {
                    let args = super::llama::get_llama_model_args(&dir).unwrap();
                    save_zero_checkpoint(
                        &super::llama::ResidentModel::new(args, stream).unwrap(),
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
                "qwen3_5" => {
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

            for quantization in [
                WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
                WeightQuantization::MxFp4,
            ] {
                let mut dense = load_model_with_options(
                    &dir,
                    ModelLoadOptions::default(),
                    stream,
                    weights_stream,
                )
                .unwrap();
                let mut quantized = load_model_with_options(
                    &dir,
                    ModelLoadOptions::with_quantization(quantization),
                    stream,
                    weights_stream,
                )
                .unwrap();
                let suffix = if quantization == WeightQuantization::MxFp4 {
                    "mxfp4"
                } else {
                    "q4"
                };
                let saved_dir = dir.with_extension(suffix);
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
                assert_eq!(dense_token, quantized_token, "{family} {quantization:?}");
                let mut saved_cache = saved_quantized.new_cache();
                let saved_logits = saved_quantized
                    .prefill_input_with_cache(input, &mut saved_cache, stream)
                    .unwrap();
                let saved_token = argmax_axis!(&saved_logits, -1, stream = stream)
                    .unwrap()
                    .item::<u32>(stream);
                assert_eq!(
                    quantized_token, saved_token,
                    "saved {family} {quantization:?}"
                );
                fs::remove_dir_all(saved_dir).unwrap();
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn tiny_gpt_oss_preserves_native_experts_and_quantizes_dense_matrices_to_mxfp4() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let dir = temp_model_dir(
            r#"{
              "model_type":"gpt_oss","hidden_size":32,"intermediate_size":32,
              "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
              "head_dim":8,"vocab_size":32,"num_local_experts":2,
              "num_experts_per_tok":1,"rms_norm_eps":0.00001,"sliding_window":16,
              "max_position_embeddings":128,"rope_scaling":null,
              "quantization_config":{"quant_method":"mxfp4"}
            }"#,
        );
        let args = super::gpt_oss::get_model_args(&dir).unwrap();
        save_zero_checkpoint(
            &super::gpt_oss::Model::new(args, stream).unwrap(),
            &dir,
            stream,
        );

        let model = load_model_with_options(
            &dir,
            ModelLoadOptions::with_quantization(WeightQuantization::MxFp4),
            stream,
            weights_stream,
        )
        .unwrap();
        let super::Model::GptOss(model) = model else {
            panic!("expected GPT-OSS model")
        };
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert!(!params.contains_key("model.layers.0.self_attn.q_proj.biases"));
        assert!(params.contains_key("model.embed_tokens.inner.weight"));
        assert!(params.contains_key("lm_head.inner.weight"));
        assert!(params.contains_key("model.layers.0.mlp.experts.gate_up_proj_blocks"));
        assert!(params.contains_key("model.layers.0.mlp.experts.gate_up_proj_scales"));
        assert!(params.contains_key("model.layers.0.mlp.router.weight"));
        assert!(!params.contains_key("model.layers.0.mlp.router.scales"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tiny_qwen3_vl_mxfp4_on_load_quantizes_only_language_model() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let dir = temp_model_dir(
            r#"{
              "model_type":"qwen3_vl","image_token_id":30,"video_token_id":31,
              "text_config":{
                "model_type":"qwen3_vl_text","hidden_size":32,"num_hidden_layers":1,
                "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":32,
                "max_position_embeddings":128,"rope_theta":10000.0,
                "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1],"mrope_interleaved":true}
              },
              "vision_config":{
                "depth":1,"hidden_size":8,"hidden_act":"gelu_pytorch_tanh",
                "intermediate_size":16,"num_heads":2,"num_position_embeddings":16,
                "in_channels":3,"patch_size":2,"spatial_merge_size":2,
                "temporal_patch_size":2,"window_size":8,"out_hidden_size":32,
                "fullatt_block_indexes":[],"deepstack_visual_indexes":[0]
              }
            }"#,
        );
        let args = super::qwen3_vl::get_qwen3_vl_model_args(&dir).unwrap();
        save_zero_checkpoint(
            &super::qwen3_vl::Model::new(args, stream).unwrap(),
            &dir,
            stream,
        );

        let quantization = WeightQuantization::MxFp4;
        let mut quantized = load_model_with_options(
            &dir,
            ModelLoadOptions::with_quantization(quantization),
            stream,
            weights_stream,
        )
        .unwrap();
        let super::Model::Qwen3Vl(model) = &quantized else {
            panic!("expected Qwen3-VL model");
        };
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.language_model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(params.contains_key("model.language_model.layers.0.self_attn.q_proj.scales"));
        assert!(!params.contains_key("model.language_model.layers.0.self_attn.q_proj.biases"));
        assert!(params.contains_key("model.language_model.embed_tokens.inner.weight"));
        assert!(params.contains_key("model.visual.blocks.0.attn.qkv.weight"));
        assert!(!params.contains_key("model.visual.blocks.0.attn.qkv.scales"));
        drop(params);

        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let pixels = Array::zeros::<f32>(&[4, 24], stream).unwrap();
        let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let parts = [
            super::input::InputPart::text_token_ids(&tokens),
            super::input::InputPart::image_tensor(
                &pixels,
                super::input::InputMetadata::qwen_grid_thw(&grid),
            ),
        ];
        let mut cache = quantized.new_cache();
        let logits = quantized
            .prefill_input_with_cache(super::input::ModelInput::new(&parts), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 32]);

        let saved_dir = dir.with_extension("mxfp4");
        crate::quantization::quantize_checkpoint(
            &dir,
            &saved_dir,
            &CheckpointQuantizationOptions {
                quantization,
                exclude: vec!["model.visual.".into()],
                ..Default::default()
            },
            stream,
        )
        .unwrap();
        let saved_quantized = load_model_with_options(
            &saved_dir,
            ModelLoadOptions::with_quantization(quantization),
            stream,
            weights_stream,
        )
        .unwrap();
        let super::Model::Qwen3Vl(saved_model) = &saved_quantized else {
            panic!("expected saved Qwen3-VL model");
        };
        assert!(saved_model
            .parameters()
            .flatten()
            .contains_key("model.language_model.layers.0.self_attn.q_proj.scales"));

        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(saved_dir).unwrap();
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
    fn tiny_qwen35_moe_mxfp4_quantizes_packed_experts_through_high_level_dispatch() {
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
            ModelLoadOptions::with_quantization(WeightQuantization::MxFp4),
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
        assert!(!params.contains_key("model.layers.0.mlp.experts.gate_up_proj_biases"));
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
    fn check_model_config_reports_supported_lfm2_families() {
        let dense = json!({
            "model_type": "lfm2",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "layer_types": ["conv", "full_attention"]
        });
        assert_eq!(
            check_model_config(&dense),
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Lfm2,
                model_type: "lfm2".into(),
                effective_model_type: "lfm2".into(),
            })
        );
        let mut moe = dense;
        moe["model_type"] = json!("lfm2_moe");
        moe["moe_intermediate_size"] = json!(8);
        moe["num_dense_layers"] = json!(1);
        moe["num_experts"] = json!(4);
        moe["num_experts_per_tok"] = json!(2);
        assert!(check_model_config(&moe).is_supported());
    }

    #[test]
    fn check_model_config_reports_supported_gpt_oss() {
        let support = check_model_config(&json!({
            "model_type": "gpt_oss",
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "rms_norm_eps": 1e-5,
            "sliding_window": 128,
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": false
            },
            "quantization_config": {"quant_method": "mxfp4"}
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::GptOss,
                model_type: "gpt_oss".to_string(),
                effective_model_type: "gpt_oss".to_string(),
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
    fn check_model_config_reports_supported_qwen3_next() {
        let support = check_model_config(&json!({
            "model_type":"qwen3_next","vocab_size":128,"hidden_size":16,
            "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
            "head_dim":8,"max_position_embeddings":128,"intermediate_size":32,
            "moe_intermediate_size":8,"shared_expert_intermediate_size":8,
            "num_experts_per_tok":2,"num_experts":4,"tie_word_embeddings":false,
            "rope_theta":10000000,"partial_rotary_factor":0.25
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Qwen3Next,
                model_type: "qwen3_next".to_string(),
                effective_model_type: "qwen3_next".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_supported_qwen3_vl_moe() {
        let support = check_model_config(&json!({
            "model_type":"qwen3_vl_moe","image_token_id":30,"video_token_id":31,
            "tie_word_embeddings":false,
            "text_config":{
                "model_type":"qwen3_vl_moe_text","hidden_size":16,"num_hidden_layers":2,
                "intermediate_size":32,"num_attention_heads":2,"rms_norm_eps":0.000001,
                "vocab_size":32,"num_key_value_heads":1,"max_position_embeddings":128,
                "rope_theta":10000.0,"head_dim":8,"moe_intermediate_size":8,
                "num_experts":4,"num_experts_per_tok":2,"norm_topk_prob":true,
                "rope_scaling":{"mrope_section":[2,1,1]}
            },
            "vision_config":{
                "depth":1,"hidden_size":8,"hidden_act":"gelu_pytorch_tanh",
                "intermediate_size":16,"num_heads":2,"num_position_embeddings":16,
                "in_channels":3,"patch_size":2,"spatial_merge_size":2,
                "temporal_patch_size":2,"out_hidden_size":16,
                "deepstack_visual_indexes":[0]
            }
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Qwen3VlMoe,
                model_type: "qwen3_vl_moe".to_string(),
                effective_model_type: "qwen3_vl_moe_text".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_supported_dense_qwen3_5() {
        let support = check_model_config(&json!({
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 128,
                "hidden_size": 16,
                "intermediate_size": 32,
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
                model_type: "qwen3_5".to_string(),
                effective_model_type: "qwen3_5_text".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_supported_dense_qwen3_5_text() {
        let support = check_model_config(&json!({
            "model_type": "qwen3_5_text",
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "layer_types": ["full_attention"]
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Qwen35Moe,
                model_type: "qwen3_5_text".to_string(),
                effective_model_type: "qwen3_5_text".to_string(),
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
    fn check_model_config_reports_supported_gemma4_moe() {
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
                "enable_moe_block": true,
                "num_experts": 4,
                "top_k_experts": 2,
                "moe_intermediate_size": 8
            }
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Gemma4,
                model_type: "gemma4".to_string(),
                effective_model_type: "gemma4_text".to_string(),
            })
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
    fn check_model_config_reports_supported_gemma4_unified_moe() {
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
                "enable_moe_block": true,
                "num_experts": 4,
                "top_k_experts": 2,
                "expert_intermediate_size": 8
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
