//! Gemma 4 assistant draft-model support for multi-token prediction.

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        argpartition_axis, full, gt,
        indexing::{put_along_axis, NewAxis, TryIndexOp},
        lt, matmul, which,
    },
    quantization::MaybeQuantized,
    random::RandomState,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    error::Error,
    models::{
        common,
        gemma4::{sample, Gemma4Embedding, LayerType, Model, ModelArgs, TransformerBlock},
        ModelLoadOptions,
    },
    quantization::WeightQuantization,
    weights::{
        load_safetensors_quantized_strict, load_safetensors_strict, StrictLoadConfig,
        StrictLoadReport,
    },
};

#[derive(Debug, Clone, Deserialize)]
/// Configuration for a Gemma 4 assistant draft model.
pub struct Gemma4AssistantConfig {
    #[serde(default = "default_model_type")]
    /// Assistant model type.
    pub model_type: String,
    /// Hidden size of the target Gemma 4 backbone.
    pub backbone_hidden_size: i32,
    #[serde(default)]
    /// Whether ordered masked embeddings are used for logits.
    pub use_ordered_embeddings: bool,
    #[serde(default = "default_num_centroids")]
    /// Number of token-ordering centroids.
    pub num_centroids: i32,
    #[serde(default = "default_centroid_top_k")]
    /// Number of centroid groups considered for masked logits.
    pub centroid_intermediate_top_k: i32,
    #[serde(default = "default_true")]
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    #[serde(default = "default_block_size")]
    /// Maximum draft block size.
    pub block_size: usize,
    /// Text model configuration for the assistant body.
    pub text_config: ModelArgs,
    #[serde(default)]
    /// Optional MLX affine checkpoint metadata.
    pub quantization: Option<WeightQuantization>,
}

fn default_model_type() -> String {
    "gemma4_assistant".to_string()
}

fn default_num_centroids() -> i32 {
    2048
}

fn default_centroid_top_k() -> i32 {
    32
}

fn default_block_size() -> usize {
    4
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, ModuleParameters)]
/// Inner assistant transformer body.
pub struct DraftInner {
    #[param]
    /// Token embedding table.
    pub embed_tokens: Gemma4Embedding,
    #[param]
    /// Assistant transformer blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final RMSNorm.
    pub norm: nn::RmsNorm,
}

impl DraftInner {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let embed_tokens = Gemma4Embedding::unloaded(
            args.vocab_size,
            args.hidden_size,
            args.quantization_for("embed_tokens.weight"),
            stream,
        )?;
        let layers = (0..args.num_hidden_layers)
            .map(|index| {
                TransformerBlock::new(
                    args,
                    args.layer_type(index as usize),
                    index as usize,
                    stream,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Masked embedding head that evaluates only selected token groups.
pub struct MaskedEmbedder {
    /// Hidden size consumed by the head.
    pub hidden_size: i32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of centroid groups.
    pub num_centroids: i32,
    /// Number of top centroid groups selected.
    pub top_k: i32,
    /// Number of vocabulary entries per centroid.
    pub vocab_size_per_centroid: i32,
    #[param]
    /// Centroid scoring projection.
    pub centroids: nn::Linear,
    #[param]
    /// Token ordering table mapping centroid slots to token ids.
    pub token_ordering: Param<Array>,
}

impl MaskedEmbedder {
    fn new(config: &Gemma4AssistantConfig, stream: &Stream) -> Result<Self, Exception> {
        let hidden_size = config.text_config.hidden_size;
        let vocab_size = config.text_config.vocab_size;
        let num_centroids = config.num_centroids;
        let vocab_size_per_centroid = vocab_size / num_centroids;
        Ok(Self {
            hidden_size,
            vocab_size,
            num_centroids,
            top_k: config.centroid_intermediate_top_k,
            vocab_size_per_centroid,
            centroids: nn::Linear::unloaded(
                hidden_size,
                num_centroids,
                false,
                Dtype::Float32,
                stream,
            )?,
            token_ordering: Param::<Array>::unloaded(&[vocab_size], Dtype::Int32, stream)?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        lm_head_weight: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let b = shape[0];
        let l = shape[1];
        let centroid_logits = self.centroids.forward(hidden_states, stream)?;
        let topk_idx = argpartition_axis(&centroid_logits, -self.top_k, -1, stream)?
            .try_index_device((.., .., -self.top_k..), stream)?;
        let ordering = self
            .token_ordering
            .as_ref()
            .reshape(&[self.num_centroids, self.vocab_size_per_centroid], stream)?;
        let selected_canonical = ordering.try_index_device(&topk_idx, stream)?;
        let flat_idx = selected_canonical.reshape(&[-1], stream)?;
        let selected_emb = lm_head_weight
            .try_index_device(&flat_idx, stream)?
            .reshape(
                &[
                    b,
                    l,
                    self.top_k * self.vocab_size_per_centroid,
                    self.hidden_size,
                ],
                stream,
            )?;
        let selected_logits = matmul(
            &hidden_states.try_index_device((.., .., NewAxis, ..), stream)?,
            selected_emb.transpose_axes(&[0, 1, 3, 2], stream)?,
            stream,
        )?
        .squeeze_axes(&[-2], stream)?;
        let mask_value = selected_logits.min(None, stream)?.item::<f32>(&stream) - 1.0;
        let out = full::<f32>(
            &[b, l, self.vocab_size],
            safemlx::array!(mask_value),
            stream,
        )?;
        let scatter_idx = selected_canonical.reshape(&[b, l, -1], stream)?;
        put_along_axis(&out, &scatter_idx, &selected_logits, -1, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Gemma 4 assistant model used to draft speculative tokens.
pub struct Gemma4AssistantDraftModel {
    /// Assistant configuration.
    pub config: Gemma4AssistantConfig,
    #[param]
    /// Assistant transformer body.
    pub model: DraftInner,
    #[param]
    /// Projection from target hidden state plus token embedding into assistant hidden size.
    pub pre_projection: MaybeQuantized<nn::Linear>,
    #[param]
    /// Projection from assistant hidden size back to target hidden size.
    pub post_projection: MaybeQuantized<nn::Linear>,
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    /// Optional masked embedding head.
    pub masked_embedding: Option<MaskedEmbedder>,
    shared_kv: Option<HashMap<LayerType, (Array, Array)>>,
    kv_offset: i32,
    accept_lens: Vec<usize>,
}

impl Gemma4AssistantDraftModel {
    /// Creates an unloaded Gemma 4 assistant draft model.
    pub fn new(mut config: Gemma4AssistantConfig, stream: &Stream) -> Result<Self, Exception> {
        if config.quantization.is_some() && config.use_ordered_embeddings {
            return Err(Exception::custom(
                "Gemma 4 assistant affine quantization does not support ordered masked embeddings because that head indexes raw dense embedding rows",
            ));
        }
        config.text_config.model_type = "gemma4".to_string();
        config.text_config.quantized = config.quantization.is_some();
        config.text_config.weight_quantization = config.quantization;
        config.text_config.quantization_group_size = config
            .quantization
            .map_or(64, |quantization| quantization.group_size());
        config.text_config.quantization_bits = config
            .quantization
            .map_or(4, |quantization| quantization.bits());
        if config.text_config.num_kv_shared_layers == 0 {
            config.text_config.num_kv_shared_layers = config.text_config.num_hidden_layers;
        }

        let text_config = &config.text_config;
        let model = DraftInner::new(text_config, stream)?;
        let pre_projection = common::linear::unloaded_maybe_quantized_linear(
            2 * config.backbone_hidden_size,
            text_config.hidden_size,
            false,
            config.quantization,
            stream,
        )?;
        let post_projection = common::linear::unloaded_maybe_quantized_linear(
            text_config.hidden_size,
            config.backbone_hidden_size,
            false,
            config.quantization,
            stream,
        )?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear(
                text_config.hidden_size,
                text_config.vocab_size,
                false,
                config.quantization,
                stream,
            )?)
        };
        let masked_embedding = if config.use_ordered_embeddings {
            Some(MaskedEmbedder::new(&config, stream)?)
        } else {
            None
        };
        Ok(Self {
            config,
            model,
            pre_projection,
            post_projection,
            lm_head,
            masked_embedding,
            shared_kv: None,
            kv_offset: 0,
            accept_lens: Vec::new(),
        })
    }

    /// Returns the configured draft block size.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Clears cached shared key/value state and acceptance history.
    pub fn reset(&mut self) {
        self.shared_kv = None;
        self.kv_offset = 0;
        self.accept_lens.clear();
    }

    /// Sets target-model key/value state shared with the assistant.
    pub fn set_shared_kv(&mut self, shared_kv: HashMap<LayerType, (Array, Array)>, kv_offset: i32) {
        self.shared_kv = Some(shared_kv);
        self.kv_offset = kv_offset;
    }

    fn forward(
        &mut self,
        inputs_embeds: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let mut h = self.pre_projection.forward(inputs_embeds, stream)?;
        let query_len = h.shape()[1];
        let query_offset = self.kv_offset.saturating_sub(1);
        let shared_kv = self
            .shared_kv
            .as_mut()
            .ok_or_else(|| Exception::custom("Gemma 4 assistant requires shared K/V states"))?;

        for layer in &mut self.model.layers {
            let kv = shared_kv
                .get(&layer.layer_type)
                .cloned()
                .ok_or_else(|| Exception::custom("missing shared K/V state for assistant layer"))?;
            let mask = drafter_mask(
                layer.layer_type,
                query_len,
                query_offset,
                kv.0.shape()[kv.0.shape().len() - 2],
                self.config.text_config.sliding_window.unwrap_or(0),
                h.dtype(),
                stream,
            )?;
            let mut kv_map = HashMap::new();
            kv_map.insert(layer.layer_type, kv);
            h = layer.forward(
                crate::models::gemma4::AttentionInput {
                    x: &h,
                    mask: mask.as_ref(),
                    cache: None::<&mut crate::cache::ConcatKeyValueCache>,
                    position_offset: query_offset,
                    per_layer_input: None,
                    shared_kv: Some(&mut kv_map),
                    disable_generated_mask: true,
                    generated_sliding_window: None,
                },
                stream,
            )?;
        }

        h = self.model.norm.forward(&h, stream)?;
        let last_hidden = self.post_projection.forward(&h, stream)?;
        let logits = if let Some(masked) = self.masked_embedding.as_mut() {
            masked.forward(&h, self.model.embed_tokens.weight.as_ref(), stream)?
        } else if let Some(lm_head) = self.lm_head.as_mut() {
            lm_head.forward(&h, stream)?
        } else {
            self.model.embed_tokens.as_linear(&h, stream)?
        };
        Ok((last_hidden, logits))
    }

    /// Drafts up to `block_size - 1` speculative tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_block(
        &mut self,
        target_model: &mut Model,
        last_bonus: u32,
        hidden: &Array,
        block_size: usize,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut token = Array::from_slice(&[last_bonus], &[1, 1]);
        let mut h_prev = hidden.clone();
        let mut tokens = Vec::new();
        let mut prng_state = prng_state;

        for _ in 0..block_size.saturating_sub(1) {
            let token_embed = target_model
                .model
                .language_model
                .embed_tokens
                .forward(&token, stream)?
                .multiply(
                    Array::from_f32((target_model.args.hidden_size as f32).sqrt()),
                    stream,
                )?;
            let inputs_embeds = safemlx::ops::concatenate_axis(&[token_embed, h_prev], -1, stream)?;
            let (next_hidden, logits) = self.forward(&inputs_embeds, stream)?;
            token = sample(&logits, temp, prng_state.as_deref_mut(), stream)?;
            tokens.push(token.clone());
            h_prev = next_hidden;
        }

        if tokens.is_empty() {
            Ok(Array::from_slice::<u32>(&[], &[1, 0]))
        } else {
            safemlx::ops::concatenate_axis(&tokens, 1, stream)
        }
    }
}

fn drafter_mask(
    layer_type: LayerType,
    query_len: i32,
    query_offset: i32,
    kv_len: i32,
    sliding_window: i32,
    _dtype: safemlx::Dtype,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    if layer_type == LayerType::FullAttention {
        return Ok(None);
    }
    if sliding_window <= 0
        || (kv_len <= sliding_window && query_offset + query_len <= kv_len + sliding_window)
    {
        return Ok(None);
    }
    let q_idx =
        safemlx::ops::arange::<_, i32>(Some(query_offset), query_offset + query_len, None, stream)?
            .try_index_device((.., NewAxis), stream)?;
    let k_idx = safemlx::ops::arange::<_, i32>(None, kv_len, None, stream)?
        .try_index_device((NewAxis, ..), stream)?;
    let dist = q_idx.subtract(k_idx, stream)?;
    let inside = gt(&dist, Array::from_int(-sliding_window), stream)?
        .logical_and(lt(&dist, Array::from_int(sliding_window), stream)?, stream)?;
    let bias = which(
        &inside,
        Array::from_f32(0.0),
        Array::from_f32(f32::NEG_INFINITY),
        stream,
    )?;
    Ok(Some(
        bias.try_index_device((NewAxis, NewAxis, .., ..), stream)?,
    ))
}

#[derive(Debug, Clone, Deserialize)]
struct WeightMap {
    #[allow(dead_code)]
    metadata: HashMap<String, Value>,
    weight_map: HashMap<String, String>,
}

/// Loads a Gemma 4 assistant draft model from a model directory.
pub fn load_gemma4_assistant_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantDraftModel, Error> {
    load_gemma4_assistant_model_with_options(
        model_dir,
        ModelLoadOptions::default(),
        stream,
        weights_stream,
    )
}

/// Loads a Gemma 4 assistant draft model using shared model-load options.
pub fn load_gemma4_assistant_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantDraftModel, Error> {
    let model_dir = model_dir.as_ref();
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    let mut config: Gemma4AssistantConfig = serde_json::from_reader(file)?;
    let quantize_on_load = if let Some(quantization) = options.quantization {
        if config.use_ordered_embeddings {
            return Err(Error::Quantization(
                "Gemma 4 assistant affine quantization does not support ordered masked embeddings because that head indexes raw dense embedding rows".into(),
            ));
        }
        let quantize = crate::quantization::should_quantize_on_load(
            "Gemma 4 assistant",
            config.quantization,
            quantization,
        )?;
        config.quantization = Some(quantization);
        quantize
    } else {
        false
    };
    let mut model = Gemma4AssistantDraftModel::new(config, stream)?;
    let load_config = StrictLoadConfig::default()
        .allow_missing_suffix(".bias")
        .allow_missing_contains(".self_attn.k_proj.")
        .allow_missing_contains(".self_attn.v_proj.")
        .allow_missing_suffix(".self_attn.k_norm.weight");
    let mut report = StrictLoadReport::default();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: std::collections::HashSet<&String> =
            weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let path = model_dir.join(weight_file);
            if quantize_on_load {
                load_safetensors_quantized_strict(
                    &mut model,
                    path,
                    weights_stream,
                    stream,
                    options.quantization.expect("quantization requested"),
                    &load_config,
                    &mut report,
                )?;
            } else {
                load_safetensors_strict(
                    &mut model,
                    path,
                    weights_stream,
                    &load_config,
                    &mut report,
                )?;
            }
        }
    } else {
        let path = model_dir.join("model.safetensors");
        if quantize_on_load {
            load_safetensors_quantized_strict(
                &mut model,
                path,
                weights_stream,
                stream,
                options.quantization.expect("quantization requested"),
                &load_config,
                &mut report,
            )?;
        } else {
            load_safetensors_strict(&mut model, path, weights_stream, &load_config, &mut report)?;
        }
    }
    report.finish(&model, &load_config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use safemlx::{module::ModuleParameters, Array, Device, DeviceType, ExecutionContext};

    use crate::{
        models::ModelLoadOptions,
        quantization::{AffineQuantization, WeightQuantization},
    };

    const CONFIG: &str = r#"{
      "model_type":"gemma4_assistant",
      "backbone_hidden_size":32,
      "use_ordered_embeddings":false,
      "tie_word_embeddings":false,
      "block_size":4,
      "text_config":{
        "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":1,
        "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
        "max_position_embeddings":128,"tie_word_embeddings":false,
        "attention_k_eq_v":false,"layer_types":["full_attention"]
      }
    }"#;

    #[test]
    fn tiny_assistant_quantizes_through_shared_options() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "safemlx-assistant-quantization-{}-{suffix}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), CONFIG).unwrap();
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let config: super::Gemma4AssistantConfig = serde_json::from_str(CONFIG).unwrap();
        let dense = super::Gemma4AssistantDraftModel::new(config, stream).unwrap();
        let parameters = dense.parameters().flatten();
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

        let quantization = WeightQuantization::MxFp4;
        let model = super::load_gemma4_assistant_model_with_options(
            &dir,
            ModelLoadOptions::with_quantization(quantization),
            stream,
            weights_context.stream(),
        )
        .unwrap();
        assert!(model.pre_projection.is_quantized());
        assert!(model.post_projection.is_quantized());
        assert!(model.lm_head.as_ref().unwrap().is_quantized());
        assert!(model.model.embed_tokens.quantized);
        assert!(!model
            .parameters()
            .flatten()
            .contains_key("pre_projection.biases"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ordered_assistant_head_reports_affine_capability_error() {
        let mut value: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        value["use_ordered_embeddings"] = true.into();
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "safemlx-assistant-capability-{}-{suffix}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&value).unwrap()).unwrap();
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let error = super::load_gemma4_assistant_model_with_options(
            &dir,
            ModelLoadOptions::with_quantization(AffineQuantization::default()),
            context.stream(),
            context.stream(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("ordered masked embeddings"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
