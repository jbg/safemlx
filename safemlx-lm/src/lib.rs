//! Language-model loading and generation utilities built on `safemlx`.
//!
//! `safemlx-lm` provides model implementations, tokenizer loading, checkpoint
//! loading, cache management, and simple token generation for MLX-compatible
//! language models. The highest-level entry point is [`models::LoadedModel`],
//! which loads a model directory containing a Hugging Face-style `config.json`,
//! `tokenizer.json`, and safetensors weights.
//!
//! [`offload`] contains architecture-independent residency planning and
//! observability contracts. It does not yet move or evict model weights.
//! [`weight_store`] catalogs safetensors checkpoints and safely materializes
//! lazily acquired selections from bounded persistent mappings.

#![warn(missing_docs)]

/// Attention key/value cache implementations.
pub mod cache;
/// Error types returned by the language-model runtime.
pub mod error;
/// Reusable expert-parallel assignment, dispatch, exchange, and model metadata.
pub mod expert_parallel;
/// Gemma 4 multi-token prediction generation helpers.
pub mod gemma4_mtp;
mod gguf_tokenizer;
/// Lightweight activation inspection hooks.
pub mod inspection;
/// Planning contracts and telemetry for future weight residency management.
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
/// Codec-free realtime speech-to-speech token APIs.
pub mod realtime;
/// Token sampling strategies.
pub mod sampler;
/// Executable pure tensor-parallel model loading and inference.
pub mod tensor_parallel;
/// Shared tensor, RoPE, attention, and tokenizer utilities.
pub mod utils;
/// Persistent checkpoint catalogs, leases, and bounded safetensors mappings.
pub mod weight_store;
/// Strict safetensors loading and validation utilities.
pub mod weights;

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

use crate::models::qwen3;

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

impl<'a, C> ModelInput<'a, C, Option<Array>> for qwen3::ModelInput<'a, C> {
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
