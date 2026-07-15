//! Qwen3-VL-MoE multimodal conditional-generation support.
//!
//! The architecture shares Qwen3-VL's vision encoder, DeepStack integration,
//! multimodal RoPE, and runtime input preparation. Its language decoder uses
//! the sparse Qwen3 feed-forward blocks selected by the nested text config.

use std::path::Path;

use safemlx::Stream;

use crate::{error::Error, quantization::WeightQuantization};

pub use super::qwen3_vl::{
    Cache, Generate, Model, ModelArgs, Qwen3VLModel, QwenVisionTransformer, VisionConfig,
};

/// Reads Qwen3-VL-MoE arguments from a Hugging Face model directory.
pub fn get_qwen3_vl_moe_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    super::qwen3_vl::get_qwen3_vl_model_args(model_dir)
}

/// Loads a Qwen3-VL-MoE safetensors checkpoint.
pub fn load_qwen3_vl_moe_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    super::qwen3_vl::load_qwen3_vl_model(model_dir, stream, weights_stream)
}

/// Loads Qwen3-VL-MoE while affine-quantizing eligible language weights.
pub fn load_qwen3_vl_moe_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    super::qwen3_vl::load_qwen3_vl_model_quantized(model_dir, quantization, stream, weights_stream)
}

pub(crate) fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    super::qwen3_vl::validate_model_config_value(config)
}
