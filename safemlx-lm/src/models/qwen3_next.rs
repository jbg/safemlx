//! Qwen3-Next text model support.
//!
//! Qwen3-Next and Qwen3.5 share the same hybrid Gated DeltaNet/full-attention
//! decoder and shared-expert MoE building blocks. This module exposes the
//! architecture-specific loading API while reusing that implementation.

use std::path::Path;

use safemlx::{
    module::ModuleParametersExt,
    ops::{concatenate_axis, indexing::TryIndexOp},
    Array, Stream,
};
use tokenizers::Tokenizer;

use crate::{
    error::Error,
    quantization::WeightQuantization,
    weights::{
        load_safetensors_dir_strict_with_split_swiglu_experts_and_transform, StrictLoadReport,
    },
};

pub use super::qwen3_5_moe::{
    sample, Cache, Generate, LayerCache, LayerType, LinearAttentionCache, Model, ModelArgs,
    ModelInput,
};

/// Reads and normalizes Qwen3-Next model arguments from `config.json`.
pub fn get_qwen3_next_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let (args, image_token_id, video_token_id, vision_config) =
        super::qwen3_5_moe::get_qwen3_5_moe_model_args(model_dir)?;
    if image_token_id.is_some() || video_token_id.is_some() || vision_config.is_some() {
        return Err(Error::UnsupportedArchitecture(
            "qwen3_next is a text-only architecture".into(),
        ));
    }
    Ok(args)
}

/// Loads `tokenizer.json` from a Qwen3-Next model directory.
pub fn load_qwen3_next_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    super::qwen3_5_moe::load_qwen3_5_moe_tokenizer(model_dir)
}

/// Loads a Qwen3-Next safetensors checkpoint.
pub fn load_qwen3_next_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    load_qwen3_next_model_with_quantization(model_dir.as_ref(), None, stream, weights_stream)
}

/// Loads a Qwen3-Next checkpoint while affine-quantizing eligible weights.
pub fn load_qwen3_next_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    quantization.validate()?;
    let model_dir = model_dir.as_ref();
    let args = get_qwen3_next_model_args(model_dir)?;
    if !crate::quantization::should_quantize_on_load("Qwen3-Next", args.quantization, quantization)?
    {
        return load_qwen3_next_model(model_dir, stream, weights_stream);
    }
    load_qwen3_next_model_with_quantization(model_dir, Some(quantization), stream, weights_stream)
}

fn load_qwen3_next_model_with_quantization(
    model_dir: &Path,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let mut args = get_qwen3_next_model_args(model_dir)?;
    if args.quantization_config.is_some() {
        return Err(Error::UnsupportedArchitecture(
            "native FP8 Qwen3-Next checkpoints with fused qkvz/ba projections are not supported"
                .into(),
        ));
    }
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
    }
    let mut model = Model::new(args, None, None, None, stream)?;
    let args = model.args.clone();
    let config = super::qwen3_5_moe::qwen3_5_moe_strict_load_config(false);
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_strict_with_split_swiglu_experts_and_transform(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        quantization,
        &config,
        &mut report,
        args.num_experts,
        |key, value| split_fused_projection(&key, value, &args, stream),
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

fn split_fused_projection(
    key: &str,
    value: Array,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    for suffix in ["weight", "scales", "biases"] {
        let qkvz_suffix = format!("linear_attn.in_proj_qkvz.{suffix}");
        if let Some(prefix) = key.strip_suffix(&qkvz_suffix) {
            let value_dim = args.linear_num_value_heads * args.linear_value_head_dim;
            let parts = split_grouped_rows(
                value,
                args.linear_num_key_heads,
                &[
                    args.linear_key_head_dim,
                    args.linear_key_head_dim,
                    value_dim / args.linear_num_key_heads,
                    value_dim / args.linear_num_key_heads,
                ],
                stream,
            )?;
            let qkv = concatenate_axis(&parts[..3], 0, stream)?;
            return Ok(vec![
                (format!("{prefix}linear_attn.in_proj_qkv.{suffix}"), qkv),
                (
                    format!("{prefix}linear_attn.in_proj_z.{suffix}"),
                    parts[3].clone(),
                ),
            ]);
        }

        let ba_suffix = format!("linear_attn.in_proj_ba.{suffix}");
        if let Some(prefix) = key.strip_suffix(&ba_suffix) {
            let per_key_head = args.linear_num_value_heads / args.linear_num_key_heads;
            let parts = split_grouped_rows(
                value,
                args.linear_num_key_heads,
                &[per_key_head, per_key_head],
                stream,
            )?;
            return Ok(vec![
                (
                    format!("{prefix}linear_attn.in_proj_b.{suffix}"),
                    parts[0].clone(),
                ),
                (
                    format!("{prefix}linear_attn.in_proj_a.{suffix}"),
                    parts[1].clone(),
                ),
            ]);
        }
    }
    Ok(vec![(key.to_string(), value)])
}

fn split_grouped_rows(
    value: Array,
    groups: i32,
    widths: &[i32],
    stream: &Stream,
) -> Result<Vec<Array>, Error> {
    if value.ndim() != 2 || groups <= 0 || widths.iter().any(|width| *width <= 0) {
        return Err(Error::UnsupportedArchitecture(format!(
            "invalid fused Qwen3-Next projection shape {:?}",
            value.shape()
        )));
    }
    let group_width = widths.iter().sum::<i32>();
    if value.dim(0) != groups * group_width {
        return Err(Error::UnsupportedArchitecture(format!(
            "fused Qwen3-Next projection has shape {:?}; expected {} output rows",
            value.shape(),
            groups * group_width
        )));
    }
    let trailing = value.dim(1);
    let grouped = value.reshape(&[groups, group_width, trailing], stream)?;
    let mut start = 0;
    widths
        .iter()
        .map(|width| {
            let part = grouped
                .try_index_device((.., start..start + *width, ..), stream)?
                .reshape(&[-1, trailing], stream)?;
            start += *width;
            Ok(part)
        })
        .collect::<Result<Vec<_>, safemlx::error::Exception>>()
        .map_err(Into::into)
}

pub(crate) fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    if config
        .get("vision_config")
        .is_some_and(|vision| !vision.is_null())
    {
        return Err(Error::UnsupportedArchitecture(
            "qwen3_next is a text-only architecture".into(),
        ));
    }
    super::qwen3_5_moe::validate_model_config_value(config)
}

#[cfg(test)]
mod tests {
    use safemlx::{
        module::ModuleParameters, transforms::eval, Array, Device, DeviceType, ExecutionContext,
    };

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn splits_checkpoint_fused_qkvz_rows_into_runtime_order() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let value = Array::from_slice(&(0..16).collect::<Vec<i32>>(), &[16, 1]);
        let parts = super::split_grouped_rows(value, 2, &[2, 2, 2, 2], stream).unwrap();
        let qkv = safemlx::ops::concatenate_axis(&parts[..3], 0, stream).unwrap();
        eval([&qkv, &parts[3]]).unwrap();
        assert_eq!(
            qkv.evaluated().unwrap().as_slice::<i32>(),
            &[0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13]
        );
        assert_eq!(
            parts[3].evaluated().unwrap().as_slice::<i32>(),
            &[6, 7, 14, 15]
        );
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn qwen3_next_parameter_tree_uses_split_runtime_projections() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let args: super::ModelArgs = serde_json::from_value(serde_json::json!({
            "model_type":"qwen3_next","vocab_size":32,"hidden_size":16,
            "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
            "head_dim":8,"max_position_embeddings":128,"intermediate_size":32,
            "moe_intermediate_size":8,"shared_expert_intermediate_size":8,
            "num_experts_per_tok":2,"num_experts":4,"tie_word_embeddings":false,
            "linear_key_head_dim":4,"linear_value_head_dim":4,
            "linear_num_key_heads":2,"linear_num_value_heads":4,
            "layer_types":["linear_attention"]
        }))
        .unwrap();
        let model = super::Model::new(args, None, None, None, stream).unwrap();
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.linear_attn.in_proj_qkv.weight"));
        assert!(params.contains_key("model.layers.0.linear_attn.in_proj_z.weight"));
        assert!(params.contains_key("model.layers.0.linear_attn.in_proj_b.weight"));
        assert!(params.contains_key("model.layers.0.linear_attn.in_proj_a.weight"));
        assert!(!params.contains_key("model.layers.0.linear_attn.in_proj_qkvz.weight"));
    }
}
