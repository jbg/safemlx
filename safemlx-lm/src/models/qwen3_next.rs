//! Qwen3-Next text model support.
//!
//! Qwen3-Next and Qwen3.5 share the same hybrid Gated DeltaNet/full-attention
//! decoder and shared-expert MoE building blocks. This module exposes the
//! architecture-specific loading API while reusing that implementation.

use std::path::Path;

use safemlx::{
    module::ModuleParametersExt,
    ops::{concatenate_axis, indexing::TryIndexOp},
    transforms::eval,
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
    if args.quantization_config.is_some() {
        return Err(Error::Quantization(
            "Qwen3-Next on-load quantization requires floating-point weights; native FP8 checkpoints cannot be implicitly transcoded".into(),
        ));
    }
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
    if let Some(config) = &args.quantization_config {
        config.validate_supported()?;
    }
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
    }
    let mut model = Model::new(args, None, None, None, stream)?;
    let args = model.args.clone();
    let config = super::qwen3_5_moe::qwen3_5_moe_strict_load_config(false);
    let mut report = StrictLoadReport::default();
    if args.uses_fp8() {
        super::qwen3_5_moe::load_qwen_fp8_safetensors_dir_strict_with_transform(
            &mut model,
            model_dir,
            weights_stream,
            stream,
            &config,
            &mut report,
            args.num_experts,
            |key, value| split_fused_projection(&key, value, &args, stream),
        )?;
    } else {
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
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

pub(crate) fn split_fused_projection(
    key: &str,
    value: Array,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    let (qkvz_widths, ba_width) = fused_projection_widths(args)?;

    let qkvz_scale_suffix = "linear_attn.in_proj_qkvz.weight_scale_inv";
    if let Some(prefix) = key.strip_suffix(qkvz_scale_suffix) {
        let block_widths = fp8_block_row_widths(&qkvz_widths)?;
        let parts = split_grouped_rows(value, args.linear_num_key_heads, &block_widths, stream)?;
        let qkv = concatenate_axis(&parts[..3], 0, stream)?;
        return evaluate_fused_projection_outputs(vec![
            (
                format!("{prefix}linear_attn.in_proj_qkv.weight_scale_inv"),
                qkv,
            ),
            (
                format!("{prefix}linear_attn.in_proj_z.weight_scale_inv"),
                parts[3].clone(),
            ),
        ]);
    }

    if key.ends_with("linear_attn.in_proj_ba.weight_scale_inv") {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3-Next in_proj_ba must remain dense BF16 and cannot carry FP8 inverse scales"
                .into(),
        ));
    }

    for suffix in ["weight", "scales", "biases"] {
        let qkvz_suffix = format!("linear_attn.in_proj_qkvz.{suffix}");
        if let Some(prefix) = key.strip_suffix(&qkvz_suffix) {
            if suffix == "weight" && args.uses_fp8() {
                fp8_block_row_widths(&qkvz_widths)?;
            }
            let parts = split_grouped_rows(value, args.linear_num_key_heads, &qkvz_widths, stream)?;
            let qkv = concatenate_axis(&parts[..3], 0, stream)?;
            return evaluate_fused_projection_outputs(vec![
                (format!("{prefix}linear_attn.in_proj_qkv.{suffix}"), qkv),
                (
                    format!("{prefix}linear_attn.in_proj_z.{suffix}"),
                    parts[3].clone(),
                ),
            ]);
        }

        let ba_suffix = format!("linear_attn.in_proj_ba.{suffix}");
        if let Some(prefix) = key.strip_suffix(&ba_suffix) {
            let parts = split_grouped_rows(
                value,
                args.linear_num_key_heads,
                &[ba_width, ba_width],
                stream,
            )?;
            return evaluate_fused_projection_outputs(vec![
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

fn evaluate_fused_projection_outputs(
    outputs: Vec<(String, Array)>,
) -> Result<Vec<(String, Array)>, Error> {
    // Detach every split from its fused checkpoint source before the loader
    // advances. Otherwise all source arrays remain reachable through lazy MLX
    // graphs until the final model-wide evaluation.
    eval(outputs.iter().map(|(_, value)| value))?;
    Ok(outputs)
}

pub(crate) fn fused_projection_widths(args: &ModelArgs) -> Result<([i32; 4], i32), Error> {
    if args.linear_num_key_heads <= 0
        || args.linear_num_value_heads <= 0
        || args.linear_value_head_dim <= 0
        || args.linear_num_value_heads % args.linear_num_key_heads != 0
    {
        return Err(Error::UnsupportedArchitecture(
            "invalid grouped Qwen3-Next fused projection dimensions".into(),
        ));
    }
    let value_dim = args
        .linear_num_value_heads
        .checked_mul(args.linear_value_head_dim)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen3-Next fused projection dimension overflow".into())
        })?;
    if value_dim % args.linear_num_key_heads != 0 {
        return Err(Error::UnsupportedArchitecture(
            "invalid grouped Qwen3-Next fused projection dimensions".into(),
        ));
    }
    let value_per_key = value_dim / args.linear_num_key_heads;
    Ok((
        [
            args.linear_key_head_dim,
            args.linear_key_head_dim,
            value_per_key,
            value_per_key,
        ],
        args.linear_num_value_heads / args.linear_num_key_heads,
    ))
}

/// Converts grouped FP8 component widths from tensor rows to 128-row scale
/// blocks. Every component boundary must be exactly block-aligned so a fused
/// checkpoint scale tensor can be split without changing quantization groups.
pub(crate) fn fp8_block_row_widths(widths: &[i32]) -> Result<Vec<i32>, Error> {
    widths
        .iter()
        .map(|width| {
            if *width <= 0 || *width % 128 != 0 {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3-Next FP8 fused projection component width {width} is not divisible by 128"
                )));
            }
            Ok(*width / 128)
        })
        .collect()
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
    use std::{collections::HashMap, sync::Arc};

    use safemlx::{
        module::ModuleParameters, transforms::eval, Array, Device, DeviceType, ExecutionContext,
    };

    fn fp8_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"qwen3_next","vocab_size":32,"hidden_size":128,
            "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
            "head_dim":128,"max_position_embeddings":128,"intermediate_size":256,
            "moe_intermediate_size":128,"shared_expert_intermediate_size":128,
            "num_experts_per_tok":1,"num_experts":2,"tie_word_embeddings":false,
            "linear_key_head_dim":128,"linear_value_head_dim":128,
            "linear_num_key_heads":2,"linear_num_value_heads":4,
            "layer_types":["linear_attention"],
            "quantization_config": {
                "quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic",
                "weight_block_size":[128,128]
            }
        })
    }

    fn fp8_args() -> super::ModelArgs {
        serde_json::from_value(fp8_config()).unwrap()
    }

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
    fn splits_fp8_qkvz_inverse_scales_in_block_row_order() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        // Two key-head groups with block widths [1, 1, 2, 2].
        let value = Array::from_slice(&(0..12).collect::<Vec<i32>>(), &[12, 1]);
        let transformed = super::split_fused_projection(
            "model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv",
            value,
            &fp8_args(),
            stream,
        )
        .unwrap();
        assert_eq!(
            transformed[0].0,
            "model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"
        );
        assert_eq!(
            transformed[1].0,
            "model.layers.0.linear_attn.in_proj_z.weight_scale_inv"
        );
        eval([&transformed[0].1, &transformed[1].1]).unwrap();
        assert_eq!(transformed[0].1.shape(), &[8, 1]);
        assert_eq!(transformed[1].1.shape(), &[4, 1]);
        assert_eq!(
            transformed[0].1.evaluated().unwrap().as_slice::<i32>(),
            &[0, 6, 1, 7, 2, 3, 8, 9]
        );
        assert_eq!(
            transformed[1].1.evaluated().unwrap().as_slice::<i32>(),
            &[4, 5, 10, 11]
        );
    }

    #[test]
    fn rejects_non_block_aligned_fp8_components() {
        let error = super::fp8_block_row_widths(&[128, 64, 256]).unwrap_err();
        assert!(error.to_string().contains("not divisible by 128"));
    }

    #[test]
    fn validates_native_fp8_metadata_for_qwen3_next() {
        let mut config = fp8_config();
        super::validate_model_config_value(&config).unwrap();
        config["quantization_config"]["weight_block_size"] = serde_json::json!([64, 128]);
        let error = super::validate_model_config_value(&config).unwrap_err();
        assert!(error.to_string().contains("weight block size"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn packs_fp8_experts_split_across_shards() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model = super::Model::new(fp8_args(), None, None, None, gpu.stream()).unwrap();
        let dir = tempfile::tempdir().unwrap();

        let expert_tensors = |expert: i32, values: [u8; 3]| {
            [
                ("gate_proj", values[0]),
                ("up_proj", values[1]),
                ("down_proj", values[2]),
            ]
            .into_iter()
            .flat_map(|(projection, value)| {
                let prefix = format!("model.layers.0.mlp.experts.{expert}.{projection}");
                [
                    (
                        format!("{prefix}.weight"),
                        Array::from_slice(&vec![value; 128 * 128], &[128, 128]),
                    ),
                    (
                        format!("{prefix}.weight_scale_inv"),
                        Array::from_slice(&[value as f32], &[1, 1]),
                    ),
                ]
            })
            .collect::<Vec<_>>()
        };
        let expert_zero = expert_tensors(0, [10, 11, 12]);
        let expert_one = expert_tensors(1, [20, 21, 22]);
        let mut shard_one = expert_zero;
        shard_one.extend(expert_one[..2].iter().cloned());
        let shard_two = expert_one[2..].to_vec();
        let shard_names = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        for (name, tensors) in shard_names.iter().zip([&shard_one, &shard_two]) {
            Array::save_safetensors(
                tensors.iter().map(|(key, value)| (key.as_str(), value)),
                None,
                dir.path().join(name),
            )
            .unwrap();
        }
        let mut weight_map = HashMap::new();
        for (name, tensors) in shard_names.iter().zip([&shard_one, &shard_two]) {
            for (key, _) in tensors {
                weight_map.insert(key.clone(), (*name).to_string());
            }
        }
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();

        let config = super::super::qwen3_5_moe::qwen3_5_moe_strict_load_config(false);
        let mut report = crate::weights::StrictLoadReport::default();
        super::super::qwen3_5_moe::load_qwen_fp8_safetensors_dir_strict_with_transform(
            &mut model,
            dir.path(),
            cpu.stream(),
            gpu.stream(),
            &config,
            &mut report,
            2,
            |key, value| Ok(vec![(key, value)]),
        )
        .unwrap();

        let params = model.parameters().flatten();
        let gate_up = params
            .get("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap();
        let down = params.get("model.layers.0.mlp.experts.down_proj").unwrap();
        let gate_up_scale = params
            .get("model.layers.0.mlp.experts.gate_up_proj_scale_inv")
            .unwrap();
        let down_scale = params
            .get("model.layers.0.mlp.experts.down_proj_scale_inv")
            .unwrap();
        eval([*gate_up, *down, *gate_up_scale, *down_scale]).unwrap();
        assert_eq!(gate_up.shape(), &[2, 256, 128]);
        assert_eq!(down.shape(), &[2, 128, 128]);
        assert_eq!(gate_up_scale.shape(), &[2, 2, 1]);
        assert_eq!(down_scale.shape(), &[2, 1, 1]);
        let gate_up = gate_up.evaluated().unwrap();
        let gate_up = gate_up.as_slice::<u8>();
        assert_eq!(gate_up[0], 10);
        assert_eq!(gate_up[128 * 128], 11);
        assert_eq!(gate_up[256 * 128], 20);
        assert_eq!(gate_up[384 * 128], 21);
        let down = down.evaluated().unwrap();
        let down = down.as_slice::<u8>();
        assert_eq!(down[0], 12);
        assert_eq!(down[128 * 128], 22);
        assert_eq!(
            gate_up_scale.evaluated().unwrap().as_slice::<f32>(),
            &[10.0, 11.0, 20.0, 21.0]
        );
        assert_eq!(
            down_scale.evaluated().unwrap().as_slice::<f32>(),
            &[12.0, 22.0]
        );

        let store = Arc::new(
            crate::weight_store::SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1)
                .unwrap(),
        );
        let entries =
            crate::qwen_hybrid::qwen_hybrid_expert_catalog(&fp8_args(), store.as_ref()).unwrap();
        let options = crate::expert_cache::ExpertCacheLoadOptions::new(
            crate::layerwise::LayerwiseLoadOptions::new(
                crate::offload::OffloadConfig::new(None, None, 1).unwrap(),
            ),
            crate::offload::OffloadConfig::new(None, None, 1).unwrap(),
            u64::MAX,
        )
        .unwrap();
        let cache = crate::expert_cache::ExpertCache::new(
            Arc::clone(&store),
            entries,
            options,
            cpu.stream().clone(),
            gpu.stream().clone(),
        )
        .unwrap();
        drop(
            cache
                .acquire_route_slice(
                    0,
                    &[1],
                    &[1],
                    crate::expert_cache::ExpertPass::Decode,
                    gpu.stream(),
                )
                .unwrap(),
        );
        let diagnostics = crate::weight_store::WeightStore::diagnostics(store.as_ref()).unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert_eq!(diagnostics.touched_shard_paths.len(), 2);
        assert!(diagnostics.evictions >= 1);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn packs_completed_fp8_layer_before_shard_end() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model = super::Model::new(fp8_args(), None, None, None, gpu.stream()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = Vec::<(String, Array)>::new();
        for expert in 0..2 {
            for (projection, value) in [
                ("gate_proj", 10 + expert),
                ("up_proj", 20 + expert),
                ("down_proj", 30 + expert),
            ] {
                let prefix = format!("model.layers.0.mlp.experts.{expert}.{projection}");
                tensors.push((
                    format!("{prefix}.weight"),
                    Array::from_slice(&vec![value as u8; 128 * 128], &[128, 128]),
                ));
                tensors.push((
                    format!("{prefix}.weight_scale_inv"),
                    Array::from_slice(&[value as f32], &[1, 1]),
                ));
            }
        }
        // Safetensors keys are ordered, so this is visited after the layer tensors.
        tensors.push((
            "zz_stop_after_complete_layer".into(),
            Array::from_slice(&[0u8], &[1]),
        ));
        Array::save_safetensors(
            tensors.iter().map(|(key, value)| (key.as_str(), value)),
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();

        let config = super::super::qwen3_5_moe::qwen3_5_moe_strict_load_config(false);
        let mut report = crate::weights::StrictLoadReport::default();
        let error = super::super::qwen3_5_moe::load_qwen_fp8_safetensors_dir_strict_with_transform(
            &mut model,
            dir.path(),
            cpu.stream(),
            gpu.stream(),
            &config,
            &mut report,
            2,
            |key, value| {
                if key == "zz_stop_after_complete_layer" {
                    Err(crate::error::Error::UnsupportedArchitecture(
                        "intentional stop after complete layer".into(),
                    ))
                } else {
                    Ok(vec![(key, value)])
                }
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("intentional stop"));

        let params = model.parameters().flatten();
        let gate_up = params
            .get("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap();
        eval([*gate_up]).unwrap();
        let gate_up = gate_up.evaluated().unwrap();
        let gate_up = gate_up.as_slice::<u8>();
        assert_eq!(gate_up[0], 10);
        assert_eq!(gate_up[128 * 128], 20);
        assert_eq!(gate_up[256 * 128], 11);
        assert_eq!(gate_up[384 * 128], 21);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn rejects_malformed_fp8_scale_shape_and_scaled_ba() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let malformed = Array::from_slice(&[0f32; 11], &[11, 1]);
        let error = super::split_fused_projection(
            "model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv",
            malformed,
            &fp8_args(),
            stream,
        )
        .unwrap_err();
        assert!(error.to_string().contains("expected 12 output rows"));

        let ba_scale = Array::from_slice(&[1f32], &[1, 1]);
        let error = super::split_fused_projection(
            "model.layers.0.linear_attn.in_proj_ba.weight_scale_inv",
            ba_scale,
            &fp8_args(),
            stream,
        )
        .unwrap_err();
        assert!(error.to_string().contains("must remain dense BF16"));
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
