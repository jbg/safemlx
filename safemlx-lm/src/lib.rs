//! Language-model loading and generation utilities built on `safemlx`.
//!
//! `safemlx-lm` provides model implementations, tokenizer loading, checkpoint
//! loading, cache management, and simple token generation for MLX-compatible
//! language models. The highest-level entry point is [`models::LoadedModel`],
//! which loads a model directory containing a Hugging Face-style `config.json`,
//! `tokenizer.json`, and safetensors weights.

#![warn(missing_docs)]

/// Attention key/value cache implementations.
pub mod cache;
/// Error types returned by the language-model runtime.
pub mod error;
/// Gemma 4 multi-token prediction generation helpers.
pub mod gemma4_mtp;
/// Lightweight activation inspection hooks.
pub mod inspection;
// pub mod generate;
/// Supported model implementations and model-directory loading helpers.
pub mod models;
/// Model-agnostic media processing and prepared-input helpers.
#[cfg(feature = "media-processing")]
pub mod processor;
/// Token sampling strategies.
pub mod sampler;
/// Shared tensor, RoPE, attention, and tokenizer utilities.
pub mod utils;
/// Strict safetensors loading and validation utilities.
pub mod weights;

pub use models::{
    check_model_config, check_model_config_json, check_model_dir, ModelConfigSupport,
    SupportedModelConfig,
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
