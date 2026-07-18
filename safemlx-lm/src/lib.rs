//! Language-model loading and generation utilities built on `safemlx`.
//!
//! `safemlx-lm` provides model implementations, tokenizer loading, checkpoint
//! loading, cache management, and simple token generation for MLX-compatible
//! language models. The highest-level entry point is [`models::LoadedModel`],
//! which loads a model directory containing a Hugging Face-style `config.json`,
//! `tokenizer.json`, and safetensors weights.
//!
//! [`offload`] contains architecture-independent residency planning and
//! observability contracts. [`residency`] executes those plans for logical
//! weight units without coupling them to a model family.
//! [`weight_store`] catalogs safetensors checkpoints and safely materializes
//! lazily acquired selections from bounded persistent mappings.
//! [`layerwise`] provides a model-family adapter contract and a reusable
//! host-backed decoder engine. [`llama`] exposes one Llama/Mistral model API
//! across fully resident and layerwise-host residency policies.
//! [`expert_cache`] adds opt-in expert-granular hot-device, warm-host, and
//! cold-checkpoint residency for every supported safetensors MoE family,
//! including rank-owned expert-parallel catalogs and separate prefill/decode
//! telemetry.

#![warn(missing_docs)]

/// Attention key/value cache implementations.
pub mod cache;
/// Layerwise-host execution for DeepSeek-V3 and DeepSeek-R1.
pub mod deepseek_v3;
/// Experimental bounded dense-layer streaming from safetensors checkpoints.
pub mod dense_stream;
/// Error types returned by the language-model runtime.
pub mod error;
/// Architecture-independent sparse routed-expert caching and telemetry.
pub mod expert_cache;
/// Reusable expert-parallel assignment, dispatch, exchange, and model metadata.
pub mod expert_parallel;
/// Multimodal layerwise-host execution for Gemma 4.
pub mod gemma4;
/// Gemma 4 multi-token prediction generation helpers.
pub mod gemma4_mtp;
mod gguf_tokenizer;
/// Unified fully resident and layerwise-host GPT-OSS execution.
pub mod gpt_oss;
/// Multimodal layerwise-host execution for Thinking Machines Lab Inkling.
pub mod inkling;
/// Lightweight activation inspection hooks.
pub mod inspection;
/// Generic model-family adapters and host-backed layerwise execution.
pub mod layerwise;
/// Unified fully resident and layerwise-host LFM2/LFM2.5 execution.
pub mod lfm2;
/// Unified Llama/Mistral loading across weight-residency policies.
pub mod llama;
/// Canonical unloaded-module checkpoint binding and resident assignment helpers.
pub mod module_binding;
/// Layerwise-host execution for Moshi and PersonaPlex realtime token models.
pub mod moshi;
/// Unified fully resident and layerwise-host Nemotron-H execution.
pub mod nemotron_h;
/// Planning contracts and telemetry for weight residency management.
pub mod offload;
// pub mod generate;
/// Supported model implementations and model-directory loading helpers.
pub mod models;
/// Runtime parallel topology, tensor placement plans, and selective checkpoint loading.
pub mod parallel;
/// Executable pure pipeline-parallel model loading and inference.
pub mod pipeline;
/// Model-agnostic media processing and prepared-input helpers.
#[cfg(feature = "media-processing")]
pub mod processor;
/// Generic affine checkpoint quantization and conversion utilities.
pub mod quantization;
/// Unified dense and sparse-MoE Qwen3 layerwise-host execution.
pub mod qwen3;
/// Shared multimodal layerwise-host execution for dense and MoE Qwen3-VL.
pub mod qwen3_vl;
/// Shared layerwise-host execution for Qwen3-Next and multimodal Qwen3.5 models.
pub mod qwen_hybrid;
/// Codec-free realtime speech-to-speech token APIs.
pub mod realtime;
/// Budgeted host and device residency for logical immutable weight units.
pub mod residency;
/// Token sampling strategies.
pub mod sampler;
/// Executable pure tensor-parallel model loading and inference.
pub mod tensor_parallel;
/// Shared tensor, RoPE, attention, and tokenizer utilities.
pub mod utils;
/// Composable metadata-validated checkpoint-derived weight recipes.
pub mod weight_recipe;
/// Persistent checkpoint catalogs, leases, and bounded safetensors mappings.
pub mod weight_store;
/// Strict safetensors loading and validation utilities.
pub mod weights;

pub use dense_stream::{BackgroundPrefetchReport, DenseDiskStreamLoadOptions, DenseStreamError};
pub use expert_cache::SparseExpertDenseStreamLoadOptions;
pub use layerwise::{
    load_general_layerwise_model, DenseCacheMetrics, DenseDiskStreamReport,
    DenseExecutionGroupReport, DensePassReport, DenseTierResidencyReport, GeneralLayerwiseModel,
    GeneralLayerwiseModelAdapter, LayerExecutionLoadOptions, LayerwiseForwardState,
    LayerwiseLoadOptions, LayerwiseModel, LayerwiseModelAdapter, LayerwiseModelMetadata,
    WeightResidency,
};
pub use llama::{load_llama_model, LlamaCache, LlamaLoadOptions, LlamaModel};
pub use models::{
    check_model_config, check_model_config_json, check_model_dir, ModelConfigSupport,
    ModelLoadOptions, SupportedModelConfig,
};
pub use parallel::{
    DeviceAssignment, ParallelTopology, PlacementPlan, RankPartition, TensorPlacement,
};
pub use realtime::{
    load_model as load_realtime_model, load_model_with_options as load_realtime_model_with_options,
    LoadedRealtimeModel, RealtimeModelKind, RealtimeState,
};

use safemlx::Array;

use crate::models::qwen3 as resident_qwen3;

/// Builder passed to [`ModelInput`] implementations during generic generation.
pub struct ModelInputBuilder<'a, C, T> {
    /// Token ids or prompt ids for the current model step.
    pub y: &'a Array,
    /// Mutable per-layer cache used by the model implementation.
    pub cache: &'a mut Vec<Option<C>>,
    /// Caller-owned generation state carried across steps.
    pub state: &'a mut T,
}

/// Converts generic generation state into a model-specific input value.
pub trait ModelInput<'a, C, T> {
    /// Builds the concrete model input expected by a [`safemlx::module::Module`].
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, T>) -> Self;
}

impl<'a, C> ModelInput<'a, C, Option<Array>> for resident_qwen3::ModelInput<'a, C> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, Option<Array>>) -> Self {
        let ModelInputBuilder { y, cache, state } = builder;

        Self {
            inputs: y,
            mask: state.as_ref(),
            cache,
        }
    }
}

/// Output type that exposes logits for token sampling.
pub trait ModelOutput {
    /// Returns the logits tensor for the current generation step.
    fn logits(&self) -> &Array;
}

impl ModelOutput for Array {
    fn logits(&self) -> &Array {
        self
    }
}
