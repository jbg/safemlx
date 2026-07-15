//! Qwen3-VL conditional-generation model support.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::{
        concatenate_axis,
        indexing::{masked_scatter, TryIndexOp},
        stack_axis, zeros_dtype, GgufMetadataArray, GgufMetadataValue,
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
        common::{self, attention::AttentionInput, generation::CausalLm},
        input as runtime_input, qwen3,
        qwen_vl::grid_thw_from_array,
    },
    quantization::WeightQuantization,
    utils::{create_attention_mask, AttentionMask},
    weights::{
        load_arrays_quantized_strict, load_arrays_strict, load_safetensors_dir_quantized_strict,
        load_safetensors_dir_strict, StrictLoadConfig, StrictLoadReport,
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
    let model_type = object
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("<missing>")
        .to_string();
    if !matches!(model_type.as_str(), "qwen3_vl" | "qwen3_vl_moe") {
        return Err(Error::UnsupportedModelType(model_type));
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
        Error::UnsupportedArchitecture(format!("{model_type} text_config must be a JSON object"))
    })?;
    if !text_object.contains_key("tie_word_embeddings") {
        if let Some(tie_word_embeddings) = object.get("tie_word_embeddings").cloned() {
            text_object.insert("tie_word_embeddings".into(), tie_word_embeddings);
        }
    }
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
            Error::UnsupportedArchitecture(format!("invalid {model_type} text_config: {error}"))
        })?;
    text_config.model_type = if model_type == "qwen3_vl_moe" {
        "qwen3_vl_moe_text"
    } else {
        "qwen3_vl_text"
    }
    .into();
    if model_type == "qwen3_vl_moe" && text_config.num_experts <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "qwen3_vl_moe text_config must define routed experts".into(),
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
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
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
        let lm_head = if args.text_config.tie_word_embeddings {
            None
        } else {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.text_config.hidden_size,
                    args.text_config.vocab_size,
                    args.text_config
                        .quantization
                        .or(args.text_config.quantization_config),
                    stream,
                )?,
            )
        };
        Ok(Self {
            args,
            model: Qwen3VLModel {
                visual,
                language_model,
            },
            lm_head,
        })
    }

    /// Returns the effective model type.
    pub fn model_type(&self) -> &str {
        if self.args.text_config.model_type == "qwen3_vl_moe_text" {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
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
        common::linear::project_logits_maybe_quantized(
            &mut self.lm_head,
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
    quantization: WeightQuantization,
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

/// Loads a Qwen3-VL GGUF language model and its llama.cpp-style vision
/// projector. The vision projector must use dense F16/BF16/F32 tensors; the
/// language model may use any GGUF quantization supported by the Qwen3 loader.
pub fn load_qwen3_vl_gguf(
    gguf_file: impl AsRef<Path>,
    mmproj_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    Ok(load_qwen3_vl_gguf_with_metadata(gguf_file, mmproj_file, stream, weights_stream)?.model)
}

pub(crate) struct LoadedQwen3VlGguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn load_qwen3_vl_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    mmproj_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedQwen3VlGguf, Error> {
    let (arrays, metadata) = Array::load_gguf_with_metadata(gguf_file, weights_stream)?;
    let (vision_arrays, vision_metadata) =
        Array::load_gguf_with_metadata(mmproj_file, weights_stream)?;
    load_qwen3_vl_gguf_data(
        arrays,
        metadata,
        vision_arrays,
        vision_metadata,
        stream,
        weights_stream,
    )
}

pub(crate) fn load_qwen3_vl_gguf_data(
    arrays: HashMap<String, Array>,
    metadata: HashMap<String, GgufMetadataValue>,
    vision_arrays: HashMap<String, Array>,
    vision_metadata: HashMap<String, GgufMetadataValue>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedQwen3VlGguf, Error> {
    load_qwen3_vl_gguf_data_with_quantization(
        arrays,
        metadata,
        vision_arrays,
        vision_metadata,
        None,
        stream,
        weights_stream,
    )
}

pub(crate) fn load_qwen3_vl_gguf_data_with_quantization(
    arrays: HashMap<String, Array>,
    metadata: HashMap<String, GgufMetadataValue>,
    mut vision_arrays: HashMap<String, Array>,
    vision_metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedQwen3VlGguf, Error> {
    let architecture = qwen3::gguf_string(&metadata, "general.architecture")?;
    if architecture != "qwen3vl" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports dense qwen3vl"
        )));
    }
    validate_qwen3_vl_mmproj(&vision_metadata)?;

    let qwen3::PreparedQwen3Gguf {
        mut args,
        arrays,
        eos_token_ids,
    } = qwen3::prepare_qwen3_gguf_data(arrays, &metadata, &architecture, false, weights_stream)?;
    args.model_type = "qwen3_vl_text".into();
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
    }
    if !args.tie_word_embeddings {
        return Err(Error::UnsupportedArchitecture(
            "qwen3vl GGUF with an untied output head is not supported".into(),
        ));
    }

    let mrope_section = gguf_integer_array(&metadata, "qwen3vl.rope.dimension_sections", Some(3))?;
    if mrope_section.iter().sum::<i32>() != args.head_dim / 2 {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF RoPE sections {mrope_section:?} do not cover half of head_dim {}",
            args.head_dim
        )));
    }

    let deepstack_visual_indexes = gguf_deepstack_layers(&vision_metadata)?;
    let hidden_size = qwen3::gguf_i32(
        &vision_metadata,
        "clip.vision.embedding_length",
        weights_stream,
    )?;
    let num_position_embeddings = vision_num_position_embeddings(&vision_arrays, hidden_size)?;
    let vision_config = VisionConfig {
        depth: qwen3::gguf_i32(&vision_metadata, "clip.vision.block_count", weights_stream)?,
        hidden_size,
        hidden_act: "gelu_pytorch_tanh".into(),
        intermediate_size: qwen3::gguf_i32(
            &vision_metadata,
            "clip.vision.feed_forward_length",
            weights_stream,
        )?,
        num_heads: qwen3::gguf_i32(
            &vision_metadata,
            "clip.vision.attention.head_count",
            weights_stream,
        )?,
        num_position_embeddings,
        in_channels: 3,
        patch_size: qwen3::gguf_i32(&vision_metadata, "clip.vision.patch_size", weights_stream)?,
        spatial_merge_size: qwen3::gguf_i32(
            &vision_metadata,
            "clip.vision.spatial_merge_size",
            weights_stream,
        )?,
        temporal_patch_size: 2,
        window_size: 112,
        out_hidden_size: qwen3::gguf_i32(
            &vision_metadata,
            "clip.vision.projection_dim",
            weights_stream,
        )?,
        fullatt_block_indexes: Vec::new(),
        deepstack_visual_indexes,
    };
    if vision_config
        .deepstack_visual_indexes
        .iter()
        .any(|&layer| layer < 0 || layer >= vision_config.depth)
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF DeepStack layers {:?} exceed vision depth {}",
            vision_config.deepstack_visual_indexes, vision_config.depth
        )));
    }
    if let Some(value) = metadata.get("qwen3vl.n_deepstack_layers") {
        let expected = value.as_i64().ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "GGUF metadata key \"qwen3vl.n_deepstack_layers\" has the wrong type".into(),
            )
        })?;
        if expected != vision_config.deepstack_visual_indexes.len() as i64 {
            return Err(Error::UnsupportedArchitecture(format!(
                "qwen3vl GGUF expects {expected} DeepStack layers, but its mmproj contains {}",
                vision_config.deepstack_visual_indexes.len()
            )));
        }
    }
    if vision_config.out_hidden_size != args.hidden_size {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF projector output {} does not match language hidden size {}",
            vision_config.out_hidden_size, args.hidden_size
        )));
    }

    let image_token_id = gguf_token_id(&metadata, "<|image_pad|>")?;
    let video_token_id = gguf_token_id(&metadata, "<|video_pad|>")?;
    let mut translated = HashMap::with_capacity(arrays.len() + vision_arrays.len());
    for (name, value) in arrays {
        let name = name
            .strip_prefix("model.")
            .map(|name| format!("model.language_model.{name}"))
            .unwrap_or(name);
        insert_translated(&mut translated, name, value)?;
    }

    if vision_arrays
        .keys()
        .any(|name| name.ends_with(".scales") || name.ends_with(".biases"))
    {
        return Err(Error::UnsupportedArchitecture(
            "quantized qwen3vl mmproj GGUF tensors are not supported; use the F16 projector".into(),
        ));
    }
    reassemble_patch_embedding(&mut vision_arrays, weights_stream)?;
    for (name, value) in vision_arrays {
        let name = translate_qwen3_vl_mmproj_name(&name, &vision_config.deepstack_visual_indexes);
        insert_translated(&mut translated, name, value)?;
    }

    let model_args = ModelArgs {
        text_config: args,
        vision_config,
        image_token_id,
        video_token_id,
        mrope_section,
    };
    let mut model = Model::new(model_args, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    if let Some(quantization) = quantization {
        load_arrays_quantized_strict(
            &mut model,
            translated,
            stream,
            quantization,
            &config,
            &mut report,
        )?;
    } else {
        load_arrays_strict(&mut model, translated, &config, &mut report)?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(LoadedQwen3VlGguf {
        model,
        eos_token_ids,
    })
}

fn validate_qwen3_vl_mmproj(metadata: &HashMap<String, GgufMetadataValue>) -> Result<(), Error> {
    let architecture = qwen3::gguf_string(metadata, "general.architecture")?;
    let projector = qwen3::gguf_string(metadata, "clip.projector_type")?;
    if architecture != "clip" || projector != "qwen3vl_merger" {
        return Err(Error::UnsupportedArchitecture(format!(
            "expected a qwen3vl GGUF vision projector, got architecture {architecture:?} and projector {projector:?}"
        )));
    }
    Ok(())
}

fn gguf_integer_array(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    take: Option<usize>,
) -> Result<Vec<i32>, Error> {
    let values = metadata
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .and_then(GgufMetadataArray::to_i64_vec)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata is missing integer array {key:?}"
            ))
        })?;
    let values = take.map_or(values.as_slice(), |count| {
        &values[..values.len().min(count)]
    });
    values
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF metadata value in {key:?} exceeds i32"
                ))
            })
        })
        .collect()
}

fn gguf_deepstack_layers(metadata: &HashMap<String, GgufMetadataValue>) -> Result<Vec<i32>, Error> {
    let layers = match metadata.get("clip.vision.is_deepstack_layers") {
        Some(GgufMetadataValue::Array(GgufMetadataArray::Bool(layers))) => layers,
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"clip.vision.is_deepstack_layers\" has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "qwen3vl mmproj is missing DeepStack layer metadata".into(),
            ));
        }
    };
    layers
        .iter()
        .enumerate()
        .filter_map(|(index, &enabled)| enabled.then_some(i32::try_from(index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::UnsupportedArchitecture("DeepStack layer index exceeds i32".into()))
}

fn vision_num_position_embeddings(
    arrays: &HashMap<String, Array>,
    hidden_size: i32,
) -> Result<i32, Error> {
    let shape = arrays
        .get("v.position_embd.weight")
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "qwen3vl mmproj is missing v.position_embd.weight".into(),
            )
        })?
        .shape();
    if shape.len() != 2 || shape[1] != hidden_size {
        return Err(Error::UnsupportedArchitecture(format!(
            "unexpected qwen3vl position embedding shape {shape:?}"
        )));
    }
    Ok(shape[0])
}

fn gguf_token_id(metadata: &HashMap<String, GgufMetadataValue>, token: &str) -> Result<u32, Error> {
    let tokens = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3vl GGUF is missing tokenizer.ggml.tokens".into())
        })?;
    let index = tokens
        .iter()
        .position(|candidate| candidate == token)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("qwen3vl GGUF tokenizer is missing {token:?}"))
        })?;
    u32::try_from(index)
        .map_err(|_| Error::UnsupportedArchitecture("qwen3vl token id exceeds u32".into()))
}

fn reassemble_patch_embedding(
    arrays: &mut HashMap<String, Array>,
    stream: &Stream,
) -> Result<(), Error> {
    let first = arrays.remove("v.patch_embd.weight").ok_or_else(|| {
        Error::UnsupportedArchitecture("qwen3vl mmproj is missing v.patch_embd.weight".into())
    })?;
    let second = arrays.remove("v.patch_embd.weight.1").ok_or_else(|| {
        Error::UnsupportedArchitecture("qwen3vl mmproj is missing v.patch_embd.weight.1".into())
    })?;
    arrays.insert(
        "v.patch_embd.weight".into(),
        stack_axis(&[first, second], 2, stream)?,
    );
    Ok(())
}

fn translate_qwen3_vl_mmproj_name(name: &str, deepstack_layers: &[i32]) -> String {
    const ROOTS: [(&str, &str); 6] = [
        ("v.position_embd", "model.visual.pos_embed"),
        ("v.patch_embd", "model.visual.patch_embed.proj"),
        ("v.post_ln", "model.visual.merger.norm"),
        ("mm.0", "model.visual.merger.linear_fc1"),
        ("mm.2", "model.visual.merger.linear_fc2"),
        ("v.blk", "model.visual.blocks"),
    ];
    if let Some(rest) = name.strip_prefix("v.deepstack.") {
        if let Some((layer, suffix)) = rest.split_once('.') {
            if let Ok(layer) = layer.parse::<i32>() {
                if let Some(index) = deepstack_layers.iter().position(|&value| value == layer) {
                    let suffix =
                        suffix
                            .replacen("fc1", "linear_fc1", 1)
                            .replacen("fc2", "linear_fc2", 1);
                    return format!("model.visual.deepstack_merger_list.{index}.{suffix}");
                }
            }
        }
    }
    for (source, target) in ROOTS {
        if name == source || name.starts_with(&format!("{source}.")) {
            let mut translated = name.replacen(source, target, 1);
            if source == "v.blk" {
                translated = translated
                    .replace(".attn_out.", ".attn.proj.")
                    .replace(".attn_qkv.", ".attn.qkv.")
                    .replace(".ffn_up.", ".mlp.linear_fc1.")
                    .replace(".ffn_down.", ".mlp.linear_fc2.")
                    .replace(".ln1.", ".norm1.")
                    .replace(".ln2.", ".norm2.");
            }
            return translated;
        }
    }
    name.to_string()
}

fn insert_translated(
    arrays: &mut HashMap<String, Array>,
    name: String,
    value: Array,
) -> Result<(), Error> {
    if arrays.insert(name.clone(), value).is_some() {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF tensors collide after translating {name:?}"
        )));
    }
    Ok(())
}

/// Finds the dense sibling mmproj used by the single-path model loader.
pub(crate) fn find_qwen3_vl_mmproj(gguf_file: &Path) -> Result<PathBuf, Error> {
    let parent = gguf_file.parent().unwrap_or_else(|| Path::new("."));
    let mut candidates = std::fs::read_dir(parent)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            name.starts_with("mmproj") && name.ends_with(".gguf")
        })
        .collect::<Vec<_>>();
    candidates.sort();
    let dense = candidates
        .iter()
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            name.contains("f16") || name.contains("bf16") || name.contains("f32")
        })
        .cloned()
        .collect::<Vec<_>>();
    match (dense.as_slice(), candidates.as_slice()) {
        ([path], _) => Ok(path.clone()),
        ([], [path]) => Ok(path.clone()),
        _ => Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF requires one dense sibling mmproj file; found {} candidates in {}",
            candidates.len(),
            parent.display()
        ))),
    }
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
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, GgufMetadataArray, GgufMetadataValue},
        Array, Device, DeviceType, ExecutionContext,
    };
    use serde_json::json;

    use crate::models::{common::generation::CausalLm, input as runtime_input};

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
            Some(crate::quantization::AffineQuantization::default().into())
        );
    }

    #[test]
    fn parses_qwen3_vl_moe_config_shape() {
        let config = json!({
            "model_type":"qwen3_vl_moe","image_token_id":151655,"video_token_id":151656,
            "tie_word_embeddings":false,
            "text_config":{
                "model_type":"qwen3_vl_moe_text","hidden_size":2048,"num_hidden_layers":48,
                "intermediate_size":6144,"num_attention_heads":32,"rms_norm_eps":0.000001,
                "vocab_size":151936,"num_key_value_heads":4,"max_position_embeddings":262144,
                "rope_theta":5000000.0,"head_dim":128,
                "moe_intermediate_size":768,"num_experts":128,"num_experts_per_tok":8,
                "norm_topk_prob":true,
                "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[24,20,20]}
            },
            "vision_config":{
                "depth":27,"hidden_size":1152,"hidden_act":"gelu_pytorch_tanh",
                "intermediate_size":4304,"num_heads":16,"num_position_embeddings":2304,
                "in_channels":3,"patch_size":16,"spatial_merge_size":2,
                "temporal_patch_size":2,"out_hidden_size":2048,
                "deepstack_visual_indexes":[8,16,24]
            }
        });
        let args = super::parse_model_args_value(config).unwrap();
        assert_eq!(args.text_config.model_type, "qwen3_vl_moe_text");
        assert_eq!(args.text_config.num_experts, 128);
        assert_eq!(args.text_config.num_experts_per_tok, 8);
        assert!(!args.text_config.tie_word_embeddings);
        assert_eq!(args.vision_config.deepstack_visual_indexes, vec![8, 16, 24]);
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
    fn translates_llama_cpp_qwen3_vl_mmproj_names() {
        let deepstack = [5, 11, 17];
        assert_eq!(
            super::translate_qwen3_vl_mmproj_name("v.blk.7.attn_qkv.weight", &deepstack),
            "model.visual.blocks.7.attn.qkv.weight"
        );
        assert_eq!(
            super::translate_qwen3_vl_mmproj_name("mm.2.bias", &deepstack),
            "model.visual.merger.linear_fc2.bias"
        );
        assert_eq!(
            super::translate_qwen3_vl_mmproj_name("v.deepstack.11.fc1.weight", &deepstack),
            "model.visual.deepstack_merger_list.1.linear_fc1.weight"
        );
    }

    #[test]
    fn parses_qwen3_vl_gguf_deepstack_and_mrope_metadata() {
        let metadata = HashMap::from([
            (
                "qwen3vl.rope.dimension_sections".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 2, 2, 0])),
            ),
            (
                "clip.vision.is_deepstack_layers".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![false, true, false, true])),
            ),
        ]);
        assert_eq!(
            super::gguf_integer_array(&metadata, "qwen3vl.rope.dimension_sections", Some(3))
                .unwrap(),
            vec![2, 2, 2]
        );
        assert_eq!(super::gguf_deepstack_layers(&metadata).unwrap(), vec![1, 3]);
    }

    #[test]
    fn discovers_dense_qwen3_vl_mmproj_sibling() {
        let dir = std::env::temp_dir().join(format!(
            "safemlx-qwen3vl-mmproj-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let model = dir.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let dense = dir.join("mmproj-Qwen3VL-2B-Instruct-F16.gguf");
        let quantized = dir.join("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
        std::fs::File::create(&model).unwrap();
        std::fs::File::create(&dense).unwrap();
        std::fs::File::create(&quantized).unwrap();
        assert_eq!(super::find_qwen3_vl_mmproj(&model).unwrap(), dense);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_loads_dense_qwen3_vl_from_gguf_named_arrays() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let source = tiny_model(stream);
        let mut arrays = HashMap::new();
        let mut vision_arrays = HashMap::new();
        for (name, value) in source.parameters().flatten() {
            if let Some(name) = name.strip_prefix("model.language_model.") {
                let name = name
                    .replace("layers.", "blk.")
                    .replace("self_attn.q_norm", "attn_q_norm")
                    .replace("self_attn.k_norm", "attn_k_norm")
                    .replace("self_attn.q_proj", "attn_q")
                    .replace("self_attn.k_proj", "attn_k")
                    .replace("self_attn.v_proj", "attn_v")
                    .replace("self_attn.o_proj", "attn_output")
                    .replace("input_layernorm", "attn_norm")
                    .replace("post_attention_layernorm", "ffn_norm")
                    .replace("mlp.gate_proj", "ffn_gate")
                    .replace("mlp.down_proj", "ffn_down")
                    .replace("mlp.up_proj", "ffn_up");
                let name = match name.as_str() {
                    "embed_tokens.weight" => "token_embd.weight".into(),
                    "norm.weight" => "output_norm.weight".into(),
                    _ => name,
                };
                arrays.insert(name, value.clone());
                continue;
            }
            let name = name.strip_prefix("model.visual.").unwrap();
            if name == "patch_embed.proj.weight" {
                vision_arrays.insert(
                    "v.patch_embd.weight".into(),
                    value.try_index_device((.., .., 0, .., ..), stream).unwrap(),
                );
                vision_arrays.insert(
                    "v.patch_embd.weight.1".into(),
                    value.try_index_device((.., .., 1, .., ..), stream).unwrap(),
                );
                continue;
            }
            let name = name
                .replace("pos_embed", "v.position_embd")
                .replace("patch_embed.proj", "v.patch_embd")
                .replace("blocks.", "v.blk.")
                .replace(".attn.qkv.", ".attn_qkv.")
                .replace(".attn.proj.", ".attn_out.")
                .replace(".mlp.linear_fc1.", ".ffn_up.")
                .replace(".mlp.linear_fc2.", ".ffn_down.")
                .replace(".norm1.", ".ln1.")
                .replace(".norm2.", ".ln2.")
                .replace("merger.norm", "v.post_ln")
                .replace("merger.linear_fc1", "mm.0")
                .replace("merger.linear_fc2", "mm.2")
                .replace("deepstack_merger_list.0.norm", "v.deepstack.0.norm")
                .replace("deepstack_merger_list.0.linear_fc1", "v.deepstack.0.fc1")
                .replace("deepstack_merger_list.0.linear_fc2", "v.deepstack.0.fc2");
            vision_arrays.insert(name, value.clone());
        }

        let mut tokens = (0..30)
            .map(|index| format!("token-{index}"))
            .collect::<Vec<_>>();
        tokens.extend(["<|image_pad|>".into(), "<|video_pad|>".into()]);
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("qwen3vl".into()),
            ),
            (
                "qwen3vl.embedding_length".into(),
                GgufMetadataValue::Uint32(12),
            ),
            ("qwen3vl.block_count".into(), GgufMetadataValue::Uint32(1)),
            (
                "qwen3vl.feed_forward_length".into(),
                GgufMetadataValue::Uint32(24),
            ),
            (
                "qwen3vl.attention.head_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3vl.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3vl.attention.key_length".into(),
                GgufMetadataValue::Uint32(12),
            ),
            (
                "qwen3vl.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "qwen3vl.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "qwen3vl.rope.freq_base".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            (
                "qwen3vl.rope.dimension_sections".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 2, 2, 0])),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(tokens)),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::Uint32(2),
            ),
        ]);
        let vision_metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("clip".into()),
            ),
            (
                "clip.projector_type".into(),
                GgufMetadataValue::String("qwen3vl_merger".into()),
            ),
            (
                "clip.vision.embedding_length".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "clip.vision.block_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                GgufMetadataValue::Uint32(16),
            ),
            (
                "clip.vision.attention.head_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "clip.vision.patch_size".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "clip.vision.spatial_merge_size".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "clip.vision.projection_dim".into(),
                GgufMetadataValue::Uint32(12),
            ),
            (
                "clip.vision.is_deepstack_layers".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true])),
            ),
        ]);

        let loaded = super::load_qwen3_vl_gguf_data(
            arrays,
            metadata,
            vision_arrays,
            vision_metadata,
            stream,
            stream,
        )
        .unwrap();
        assert_eq!(loaded.model.args.image_token_id, 30);
        assert_eq!(loaded.model.args.video_token_id, 31);
        assert_eq!(loaded.model.args.mrope_section, vec![2, 2, 2]);
        assert_eq!(loaded.eos_token_ids, vec![2]);
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
