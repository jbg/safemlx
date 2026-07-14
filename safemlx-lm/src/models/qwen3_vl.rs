//! Qwen3-VL conditional-generation model support.

use std::path::Path;

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::{
        concatenate_axis,
        indexing::{masked_scatter, TryIndexOp},
        zeros_dtype,
    },
    quantization::MaybeQuantized,
    Array, Stream,
};
use serde_json::Value;

pub use super::qwen_vl::{QwenVisionTransformer, VisionConfig};

use crate::{
    cache::ConcatKeyValueCache,
    error::Error,
    models::{
        common::{self, AttentionInput, CausalLm},
        input as runtime_input, qwen3,
        qwen_vl::grid_thw_from_array,
    },
    quantization::AffineQuantization,
    utils::{create_attention_mask, AttentionMask},
    weights::{
        load_safetensors_dir_quantized_strict, load_safetensors_dir_strict, StrictLoadConfig,
        StrictLoadReport,
    },
};

#[derive(Debug, Clone)]
/// Parsed Qwen3-VL configuration.
pub struct ModelArgs {
    /// Text decoder configuration shared with Qwen3.
    pub text_config: qwen3::ModelArgs,
    /// Vision encoder configuration shared across Qwen VL models.
    pub vision_config: VisionConfig,
    /// Placeholder token used for image embeddings.
    pub image_token_id: u32,
    /// Placeholder token used for video embeddings.
    pub video_token_id: u32,
    /// Interleaved temporal/height/width RoPE sections.
    pub mrope_section: Vec<i32>,
}

fn parse_model_args_value(mut value: Value) -> Result<ModelArgs, Error> {
    let object = value.as_object_mut().ok_or_else(|| {
        Error::UnsupportedArchitecture("qwen3_vl config must be a JSON object".into())
    })?;
    if object.get("model_type").and_then(Value::as_str) != Some("qwen3_vl") {
        return Err(Error::UnsupportedModelType(
            object
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("<missing>")
                .to_string(),
        ));
    }
    let image_token_id = object
        .get("image_token_id")
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3_vl config is missing image_token_id".into())
        })?;
    let video_token_id = object
        .get("video_token_id")
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3_vl config is missing video_token_id".into())
        })?;
    let vision_config: VisionConfig =
        serde_json::from_value(object.get("vision_config").cloned().ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3_vl config is missing vision_config".into())
        })?)?;
    let top_level_quantization = object.get("quantization").cloned();
    let top_level_quantization_config = object.get("quantization_config").cloned();
    let mut text_value = object.get("text_config").cloned().ok_or_else(|| {
        Error::UnsupportedArchitecture("qwen3_vl config is missing text_config".into())
    })?;
    let text_object = text_value.as_object_mut().ok_or_else(|| {
        Error::UnsupportedArchitecture("qwen3_vl text_config must be a JSON object".into())
    })?;
    if let Some(quantization) = top_level_quantization {
        text_object.insert("quantization".into(), quantization);
    }
    if let Some(quantization) = top_level_quantization_config {
        text_object.insert("quantization_config".into(), quantization);
    }
    let rope = text_object
        .get_mut("rope_scaling")
        .and_then(Value::as_object_mut);
    let mrope_section = rope
        .as_ref()
        .and_then(|rope| rope.get("mrope_section"))
        .and_then(Value::as_array)
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_i64().and_then(|value| i32::try_from(value).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![24, 20, 20]);
    if let Some(rope) = rope {
        rope.remove("mrope_section");
        rope.remove("mrope_interleaved");
    }
    let mut text_config: qwen3::ModelArgs =
        serde_json::from_value(text_value).map_err(|error| {
            Error::UnsupportedArchitecture(format!("invalid qwen3_vl text_config: {error}"))
        })?;
    text_config.model_type = "qwen3_vl_text".into();
    if !text_config.tie_word_embeddings {
        return Err(Error::UnsupportedArchitecture(
            "qwen3_vl untied language-model heads are not supported".into(),
        ));
    }
    if mrope_section.len() != 3 || mrope_section.iter().any(|&section| section < 0) {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3_vl mrope_section must contain three non-negative values, got {mrope_section:?}"
        )));
    }
    if mrope_section.iter().sum::<i32>() != text_config.head_dim / 2 {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3_vl mrope_section {mrope_section:?} does not cover half of head_dim {}",
            text_config.head_dim
        )));
    }
    Ok(ModelArgs {
        text_config,
        vision_config,
        image_token_id,
        video_token_id,
        mrope_section,
    })
}

/// Reads Qwen3-VL arguments from a Hugging Face model directory.
pub fn get_qwen3_vl_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    parse_model_args_value(serde_json::from_reader(std::fs::File::open(
        model_dir.as_ref().join("config.json"),
    )?)?)
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    parse_model_args_value(config.clone()).map(|_| ())
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3-VL vision encoder and Qwen3 language decoder.
pub struct Qwen3VLModel {
    #[param]
    /// Vision tower.
    pub visual: QwenVisionTransformer,
    #[param]
    /// Qwen3-compatible language model body.
    pub language_model: qwen3::Qwen3Model,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3-VL conditional-generation model.
pub struct Model {
    /// Parsed model configuration.
    pub args: ModelArgs,
    #[param]
    /// Model body matching the public checkpoint parameter tree.
    pub model: Qwen3VLModel,
}

/// Generation state for Qwen3-VL, including multimodal RoPE offset state.
#[derive(Debug, Clone, Default)]
pub struct Cache {
    /// Per-layer key/value caches.
    pub kv: Vec<Option<ConcatKeyValueCache>>,
    rope_delta: i32,
}

struct PreparedPrefill {
    tokens: Array,
    embeddings: Array,
    position_ids: [Vec<i32>; 3],
    rope_delta: i32,
    deepstack_features: Vec<Array>,
}

impl Model {
    /// Creates an unloaded Qwen3-VL model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let visual = QwenVisionTransformer::new_deepstack(args.vision_config.clone(), stream)?;
        let language_model = qwen3::Qwen3Model::new(&args.text_config, stream)?;
        Ok(Self {
            args,
            model: Qwen3VLModel {
                visual,
                language_model,
            },
        })
    }

    /// Returns the effective model type.
    pub fn model_type(&self) -> &str {
        "qwen3_vl"
    }

    /// Creates an empty cache.
    pub fn new_cache(&self) -> Cache {
        Cache::default()
    }

    fn prepare_typed_prefill(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<PreparedPrefill, Exception> {
        let modality_tokens = [
            runtime_input::ModalityToken {
                modality: runtime_input::Modality::Image,
                token_id: self.args.image_token_id,
            },
            runtime_input::ModalityToken {
                modality: runtime_input::Modality::Video,
                token_id: self.args.video_token_id,
            },
        ];
        let deepstack_count = self.args.vision_config.deepstack_visual_indexes.len();
        let mut collected = (0..deepstack_count)
            .map(|_| Vec::new())
            .collect::<Vec<Vec<Array>>>();
        let embed_tokens = &mut self.model.language_model.embed_tokens;
        let visual = &mut self.model.visual;
        let prepared = runtime_input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.text_config.hidden_size,
            "qwen3_vl",
            stream,
            |tokens, stream| embed_tokens.forward(tokens, stream),
            |part, stream| {
                let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                    Exception::custom(format!(
                        "qwen3_vl {} input requires qwen_grid_thw metadata",
                        part.modality.as_str()
                    ))
                })?;
                let tensor = match part.payload {
                    runtime_input::InputPayload::Tensor(tensor) => tensor,
                    runtime_input::InputPayload::Embeddings(_) => {
                        return Err(Exception::custom(
                            "qwen3_vl requires model-native visual tensors because DeepStack features cannot be reconstructed from final embeddings",
                        ));
                    }
                    runtime_input::InputPayload::TokenIds(_) => {
                        return Err(Exception::custom(
                            "qwen3_vl visual input does not accept token-id payloads",
                        ));
                    }
                };
                let output = visual.forward_features(tensor, grid, stream)?;
                if output.deepstack_features.len() != collected.len() {
                    return Err(Exception::custom(format!(
                        "qwen3_vl vision tower returned {} DeepStack features, expected {}",
                        output.deepstack_features.len(),
                        collected.len()
                    )));
                }
                for (layer, feature) in output.deepstack_features.into_iter().enumerate() {
                    collected[layer].push(feature);
                }
                Ok(vec![output.embeddings])
            },
        )?;
        let tokens = prepared.tokens().clone();
        let embeddings = match prepared.embeddings() {
            Some(embeddings) => embeddings.clone(),
            None => self
                .model
                .language_model
                .embed_tokens
                .forward(&tokens, stream)?,
        };
        let (position_ids, rope_delta) = multimodal_position_ids(
            input,
            self.args.vision_config.spatial_merge_size,
            tokens.dim(1),
            stream,
        )?;
        let deepstack_features = if collected
            .first()
            .is_some_and(|features| features.is_empty())
        {
            Vec::new()
        } else {
            collected
                .into_iter()
                .map(|features| concatenate_axis(&features, 1, stream))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(PreparedPrefill {
            tokens,
            embeddings,
            position_ids,
            rope_delta,
            deepstack_features,
        })
    }

    fn forward_prepared(
        &mut self,
        prepared: PreparedPrefill,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        cache.rope_delta = prepared.rope_delta;
        self.forward_embeddings(
            &prepared.tokens,
            prepared.embeddings,
            &prepared.position_ids,
            &prepared.deepstack_features,
            cache,
            stream,
        )
    }

    fn forward_embeddings(
        &mut self,
        tokens: &Array,
        mut hidden: Array,
        position_ids: &[Vec<i32>; 3],
        deepstack_features: &[Array],
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Some(mask),
            Some(AttentionMask::Causal) => {
                return Err(Exception::custom(
                    "qwen3_vl requires an explicit causal mask",
                ));
            }
            None => None,
        };
        if cache.kv.is_empty() {
            cache.kv = (0..self.model.language_model.layers.len())
                .map(|_| Some(ConcatKeyValueCache::default()))
                .collect();
        }
        let (cos, sin) = mrope_embeddings(
            position_ids,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        let visual_mask = if deepstack_features.is_empty() {
            None
        } else {
            Some(
                tokens
                    .eq(Array::from_int(self.args.image_token_id as i32), stream)?
                    .logical_or(
                        &tokens.eq(Array::from_int(self.args.video_token_id as i32), stream)?,
                        stream,
                    )?,
            )
        };
        for (layer_index, (layer, layer_cache)) in self
            .model
            .language_model
            .layers
            .iter_mut()
            .zip(cache.kv.iter_mut())
            .enumerate()
        {
            hidden = layer.forward_with_rotary_embeddings(
                AttentionInput {
                    x: &hidden,
                    mask: mask.as_ref(),
                    cache: layer_cache.as_mut(),
                },
                &cos,
                &sin,
                stream,
            )?;
            if let Some(features) = deepstack_features.get(layer_index) {
                let base = zeros_dtype(hidden.shape(), hidden.dtype(), stream)?;
                let features = features.try_index_device((0, .., ..), stream)?;
                let aligned = masked_scatter(
                    &base,
                    visual_mask.as_ref().expect("DeepStack visual mask"),
                    features,
                    stream,
                )?;
                hidden = hidden.add(aligned, stream)?;
            }
        }
        let hidden = self.model.language_model.norm.forward(&hidden, stream)?;
        let mut lm_head: Option<MaybeQuantized<nn::Linear>> = None;
        common::project_logits_maybe_quantized(
            &mut lm_head,
            &mut self.model.language_model.embed_tokens,
            &hidden,
            stream,
        )
    }
}

fn multimodal_position_ids(
    input: runtime_input::ModelInput<'_>,
    spatial_merge_size: i32,
    expected_len: i32,
    stream: &Stream,
) -> Result<([Vec<i32>; 3], i32), Exception> {
    let mut positions = [Vec::new(), Vec::new(), Vec::new()];
    let mut current = 0;
    for part in input.parts {
        match (part.modality, part.payload) {
            (runtime_input::Modality::Text, runtime_input::InputPayload::TokenIds(tokens)) => {
                for position in current..current + tokens.dim(1) {
                    for axis in &mut positions {
                        axis.push(position);
                    }
                }
                current += tokens.dim(1);
            }
            (runtime_input::Modality::Image | runtime_input::Modality::Video, _) => {
                let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                    Exception::custom("qwen3_vl visual input requires qwen_grid_thw metadata")
                })?;
                for (t, h, w) in grid_thw_from_array(grid, stream)? {
                    let h = h / spatial_merge_size;
                    let w = w / spatial_merge_size;
                    for temporal in 0..t {
                        for height in 0..h {
                            for width in 0..w {
                                positions[0].push(current + temporal);
                                positions[1].push(current + height);
                                positions[2].push(current + width);
                            }
                        }
                    }
                    current += h.max(w);
                }
            }
            _ => {
                return Err(Exception::custom(format!(
                    "qwen3_vl does not support {} input",
                    part.modality.as_str()
                )));
            }
        }
    }
    if positions[0].len() as i32 != expected_len {
        return Err(Exception::custom(format!(
            "qwen3_vl position metadata describes {} tokens, prepared input has {expected_len}",
            positions[0].len()
        )));
    }
    let max_position = positions
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .unwrap_or(0);
    Ok((positions, max_position + 1 - expected_len))
}

fn mrope_embeddings(
    position_ids: &[Vec<i32>; 3],
    head_dim: i32,
    theta: f32,
    sections: &[i32],
) -> (Array, Array) {
    let (cos, sin) = mrope_values(position_ids, head_dim, theta, sections);
    let len = position_ids[0].len() as i32;
    (
        Array::from_slice(&cos, &[1, len, head_dim]),
        Array::from_slice(&sin, &[1, len, head_dim]),
    )
}

fn mrope_values(
    position_ids: &[Vec<i32>; 3],
    head_dim: i32,
    theta: f32,
    sections: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq = (0..half)
        .map(|index| 1.0 / theta.powf(2.0 * index as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    let len = position_ids[0].len();
    let mut cos = Vec::with_capacity(len * head_dim as usize);
    let mut sin = Vec::with_capacity(len * head_dim as usize);
    for ((&temporal, &height), &width) in position_ids[0]
        .iter()
        .zip(&position_ids[1])
        .zip(&position_ids[2])
    {
        let token_positions = [temporal, height, width];
        let mut angles = Vec::with_capacity(half as usize);
        for (index, inv) in inv_freq.iter().enumerate() {
            let axis = if index % 3 == 1 && index < (sections[1] * 3) as usize {
                1
            } else if index % 3 == 2 && index < (sections[2] * 3) as usize {
                2
            } else {
                0
            };
            angles.push(token_positions[axis] as f32 * inv);
        }
        for angle in angles.iter().chain(angles.iter()) {
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

/// Loads Qwen3-VL safetensors from a model directory.
pub fn load_qwen3_vl_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let mut model = Model::new(get_qwen3_vl_model_args(model_dir)?, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_strict(&mut model, model_dir, weights_stream, &config, &mut report)?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads a dense Qwen3-VL checkpoint while affine-quantizing its language model.
///
/// The shared Qwen vision tower remains dense so its parameter and checkpoint
/// layout stays identical for Qwen3-VL and Qwen3.5 multimodal models.
pub fn load_qwen3_vl_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: AffineQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let mut args = get_qwen3_vl_model_args(model_dir)?;
    let existing = args
        .text_config
        .quantization
        .or(args.text_config.quantization_config);
    if !crate::quantization::should_quantize_on_load("Qwen3-VL", existing, quantization)? {
        return load_qwen3_vl_model(model_dir, stream, weights_stream);
    }
    args.text_config.quantization = Some(quantization);
    args.text_config.quantization_config = None;
    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_quantized_strict(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        quantization,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prepared = self.prepare_typed_prefill(input, stream)?;
        self.forward_prepared(prepared, cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embeddings = self
            .model
            .language_model
            .embed_tokens
            .forward(input_tokens, stream)?;
        let start = cache
            .kv
            .first()
            .and_then(Option::as_ref)
            .map(crate::cache::KeyValueCache::offset)
            .unwrap_or(0)
            + cache.rope_delta;
        let positions = [
            (start..start + input_tokens.dim(1)).collect(),
            (start..start + input_tokens.dim(1)).collect(),
            (start..start + input_tokens.dim(1)).collect(),
        ];
        self.forward_embeddings(input_tokens, embeddings, &positions, &[], cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Qwen3-VL generation iterator.
pub type Generate<'a, S = crate::sampler::DefaultSampler> = common::Generate<'a, Model, Cache, S>;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{module::ModuleParameters, Array, Device, DeviceType, ExecutionContext};
    use serde_json::json;

    use crate::models::{common::CausalLm, input as runtime_input};

    fn tiny_model(stream: &safemlx::Stream) -> super::Model {
        let text_config = super::qwen3::ModelArgs {
            model_type: "qwen3_vl_text".into(),
            hidden_size: 12,
            num_hidden_layers: 1,
            intermediate_size: 24,
            num_attention_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 32,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            head_dim: 12,
            tie_word_embeddings: true,
            rope_scaling: Some(HashMap::new()),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            quantized_weight_configs: None,
        };
        let vision_config = super::VisionConfig {
            depth: 1,
            hidden_size: 8,
            hidden_act: "gelu_pytorch_tanh".into(),
            intermediate_size: 16,
            num_heads: 2,
            num_position_embeddings: 16,
            in_channels: 3,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            window_size: 8,
            out_hidden_size: 12,
            fullatt_block_indexes: Vec::new(),
            deepstack_visual_indexes: vec![0],
        };
        super::Model::new(
            super::ModelArgs {
                text_config,
                vision_config,
                image_token_id: 30,
                video_token_id: 31,
                mrope_section: vec![2, 2, 2],
            },
            stream,
        )
        .unwrap()
    }

    #[test]
    fn parses_qwen3_vl_2b_config_shape() {
        let mut config = json!({
            "model_type":"qwen3_vl","image_token_id":151655,"video_token_id":151656,
            "text_config":{
                "model_type":"qwen3_vl_text","hidden_size":2048,"num_hidden_layers":28,
                "intermediate_size":6144,"num_attention_heads":16,"rms_norm_eps":0.000001,
                "vocab_size":151936,"num_key_value_heads":8,"max_position_embeddings":262144,
                "rope_theta":5000000.0,"head_dim":128,"tie_word_embeddings":true,
                "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[24,20,20]}
            },
            "vision_config":{
                "depth":24,"hidden_size":1024,"hidden_act":"gelu_pytorch_tanh",
                "intermediate_size":4096,"num_heads":16,"num_position_embeddings":2304,
                "in_channels":3,"patch_size":16,"spatial_merge_size":2,
                "temporal_patch_size":2,"out_hidden_size":2048,
                "deepstack_visual_indexes":[5,11,17]
            }
        });
        let args = super::parse_model_args_value(config.clone()).unwrap();
        assert_eq!(args.text_config.hidden_size, 2048);
        assert_eq!(args.vision_config.deepstack_visual_indexes, vec![5, 11, 17]);
        assert_eq!(args.mrope_section, vec![24, 20, 20]);
        assert!(args.text_config.quantization.is_none());

        config["quantization"] = json!({"group_size": 64, "bits": 4, "mode": "affine"});
        let args = super::parse_model_args_value(config).unwrap();
        assert_eq!(
            args.text_config.quantization,
            Some(crate::quantization::AffineQuantization::default())
        );
    }

    #[test]
    fn interleaved_mrope_uses_height_and_width_slots() {
        let positions = [vec![1], vec![2], vec![3]];
        let (values, _) = super::mrope_values(&positions, 12, 10_000.0, &[2, 2, 2]);
        assert!((values[0] - 1.0f32.cos()).abs() < 1e-6);
        let height_angle = 2.0 / 10_000.0f32.powf(2.0 / 12.0);
        assert!((values[1] - height_angle.cos()).abs() < 1e-6);
        let width_angle = 3.0 / 10_000.0f32.powf(4.0 / 12.0);
        assert!((values[2] - width_angle.cos()).abs() < 1e-6);

        let (values, _) = super::mrope_values(&positions, 14, 10_000.0, &[3, 2, 2]);
        let temporal_tail_angle = 1.0 / 10_000.0f32.powf(12.0 / 14.0);
        assert!((values[6] - temporal_tail_angle.cos()).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn parameter_tree_matches_public_qwen3_vl_checkpoint_names() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let model = tiny_model(context.stream());
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.language_model.embed_tokens.weight"));
        assert!(params.contains_key("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(params.contains_key("model.visual.patch_embed.proj.weight"));
        assert!(params.contains_key("model.visual.deepstack_merger_list.0.norm.weight"));
        assert!(!params.contains_key("lm_head.weight"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn tiny_image_prefill_runs_deepstack_and_mrope() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut model = tiny_model(stream);
        for (_, parameter) in model.parameters_mut().flatten() {
            *parameter = Array::zeros::<f32>(parameter.shape(), stream).unwrap();
        }
        let text = Array::from_slice(&[1u32, 2], &[1, 2]);
        let pixels = Array::zeros::<f32>(&[4, 24], stream).unwrap();
        let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::qwen_grid_thw(&grid),
            ),
        ];
        let mut cache = model.new_cache();
        let logits = model
            .prefill_input_logits(runtime_input::ModelInput::new(&parts), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 32]);
        assert_eq!(
            cache.kv[0]
                .as_ref()
                .map(crate::cache::KeyValueCache::offset),
            Some(3)
        );
    }
}
