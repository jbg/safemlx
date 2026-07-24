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
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    error::Error,
    models::{
        common,
        gemma4::{self, Gemma4Embedding, LayerType, ModelArgs, TransformerBlock},
        ModelLoadOptions,
    },
    quantization::WeightQuantization,
    utils::rope::FloatOrString,
    weights::{
        gguf_affine_configs, gguf_metadata, load_gguf_strict, load_safetensors_quantized_strict,
        load_safetensors_strict, GgufTensorNames, StrictLoadConfig, StrictLoadReport,
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
}

/// Cloneable per-branch state for one Gemma 4 assistant draft path.
///
/// Keeping this state outside [`Gemma4AssistantDraftModel`] lets a future
/// optimistic scheduler fork a draft path before target verification and
/// discard the losing branch without copying or mutating model parameters.
#[derive(Debug, Clone)]
pub(crate) struct Gemma4AssistantDraftState {
    shared_kv: HashMap<LayerType, (Array, Array)>,
    kv_offset: i32,
    hidden: Array,
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
        })
    }

    /// Returns the configured draft block size.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Begins one generalized speculative round from committed target state.
    pub(crate) fn begin_round(
        &self,
        shared_kv: HashMap<LayerType, (Array, Array)>,
        kv_offset: i32,
        hidden: &Array,
    ) -> Gemma4AssistantDraftState {
        Gemma4AssistantDraftState {
            shared_kv,
            kv_offset,
            hidden: hidden.clone(),
        }
    }

    /// Produces one draft distribution and advances the round's hidden state.
    pub(crate) fn draft_step(
        &mut self,
        token_embedding: &Array,
        state: &mut Gemma4AssistantDraftState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let inputs_embeds =
            safemlx::ops::concatenate_axis(&[token_embedding, &state.hidden], -1, stream)?;
        let (next_hidden, logits) = self.forward(&inputs_embeds, state, stream)?;
        state.hidden = next_hidden;
        state.kv_offset = state.kv_offset.saturating_add(1);
        Ok(logits)
    }

    fn forward(
        &mut self,
        inputs_embeds: &Array,
        state: &mut Gemma4AssistantDraftState,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let mut h = self.pre_projection.forward(inputs_embeds, stream)?;
        let query_len = h.shape()[1];
        let query_offset = state.kv_offset.saturating_sub(1);

        for layer in &mut self.model.layers {
            let kv = state
                .shared_kv
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

/// Loads a Gemma 4 assistant draft model using shared model-load options.
pub(crate) fn load_gemma4_assistant_model_with_options(
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

/// Loads a Gemma 4 assistant from a GGUF file.
///
/// GGUF tensors may use standard llama.cpp Gemma assistant names or the
/// assistant module's canonical parameter names. Packed affine checkpoints
/// must use one common bit width and group size, matching the assistant's
/// existing uniform quantization contract. The model config is read from the
/// `safemlx.mtp.config` JSON metadata string when present, with a sibling
/// `config.json` as the fallback.
pub(crate) fn load_gemma4_assistant_gguf_with_options(
    gguf_file: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantDraftModel, Error> {
    if !matches!(
        options.weight_residency,
        crate::layerwise::WeightResidency::FullyResident
    ) {
        return Err(Error::UnsupportedArchitecture(
            "Gemma 4 assistant GGUF loading supports fully resident weights only".into(),
        ));
    }
    let gguf_file = gguf_file.as_ref();
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(architecture))
            if matches!(
                architecture.as_str(),
                "gemma4_assistant" | "gemma4-assistant"
            ) => {}
        Some(GgufMetadataValue::String(architecture)) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF architecture {architecture:?}; expected gemma4_assistant or gemma4-assistant"
            )))
        }
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF general.architecture must be a string".into(),
            ))
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 assistant GGUF is missing general.architecture".into(),
            ))
        }
    }
    let mut config: Gemma4AssistantConfig = match metadata.get("safemlx.mtp.config") {
        Some(GgufMetadataValue::String(config)) => serde_json::from_str(config)?,
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF safemlx.mtp.config must be a JSON string".into(),
            ))
        }
        None if metadata.contains_key("gemma4-assistant.embedding_length") => {
            gemma4_assistant_config_from_gguf(&checkpoint, &metadata, weights_stream)?
        }
        None => {
            let config_file = gguf_file
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("config.json");
            serde_json::from_reader(std::fs::File::open(&config_file)?)?
        }
    };
    crate::models::validate_gguf_quantization_source(&checkpoint, &metadata, options.quantization)?;
    let packed = gguf_affine_configs(&checkpoint, translate_gguf_weight_name)?;
    if let Some(first) = packed.values().next().copied() {
        if packed.values().any(|config| *config != first) {
            return Err(Error::Quantization(
                "Gemma 4 assistant GGUF requires one affine configuration for all packed tensors"
                    .into(),
            ));
        }
        config.quantization = Some(first.into());
    } else if let Some(requested) = options.quantization {
        if config.use_ordered_embeddings {
            return Err(Error::Quantization(
                "Gemma 4 assistant affine quantization does not support ordered masked embeddings"
                    .into(),
            ));
        }
        config.quantization = Some(requested);
    }
    let mut model = Gemma4AssistantDraftModel::new(config, stream)?;
    let load_config = StrictLoadConfig::default()
        .allow_missing_suffix(".bias")
        .allow_missing_contains(".self_attn.k_proj.")
        .allow_missing_contains(".self_attn.v_proj.")
        .allow_missing_suffix(".self_attn.k_norm.weight")
        .allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    load_gguf_strict(
        &mut model,
        &checkpoint,
        (packed.is_empty())
            .then_some(options.quantization)
            .flatten()
            .map(|quantization| (quantization, stream)),
        &load_config,
        &mut report,
        |name, value| Ok((translate_gguf_weight_name(&name), value)),
    )?;
    report.finish(&model, &load_config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

fn gemma4_assistant_config_from_gguf(
    checkpoint: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    stream: &Stream,
) -> Result<Gemma4AssistantConfig, Error> {
    const PREFIX: &str = "gemma4-assistant";
    let key = |suffix: &str| format!("{PREFIX}.{suffix}");
    let num_hidden_layers = gemma4::gguf_i32(metadata, &key("block_count"), stream)?;
    let layer_pattern = gemma4::gguf_optional_sliding_window_pattern(
        metadata,
        &key("attention.sliding_window_pattern"),
        stream,
    )?
    .unwrap_or_else(|| vec![0; num_hidden_layers as usize]);
    if layer_pattern.len() != num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "Gemma 4 assistant sliding-window pattern has {} entries for {num_hidden_layers} layers",
            layer_pattern.len()
        )));
    }
    let layer_types = layer_pattern
        .into_iter()
        .map(|sliding| {
            if sliding != 0 {
                LayerType::SlidingAttention
            } else {
                LayerType::FullAttention
            }
        })
        .collect::<Vec<_>>();
    let feed_forward_lengths = gemma4::expand_layer_values(
        &key("feed_forward_length"),
        gemma4::gguf_i64_values(metadata, &key("feed_forward_length"), stream)?,
        num_hidden_layers,
    )?;
    let intermediate_size = feed_forward_lengths[0];
    let kv_heads = gemma4::expand_layer_values(
        &key("attention.head_count_kv"),
        gemma4::gguf_i64_values(metadata, &key("attention.head_count_kv"), stream)?,
        num_hidden_layers,
    )?;
    let sliding_kv_heads = layer_types
        .iter()
        .zip(&kv_heads)
        .find_map(|(kind, value)| (*kind == LayerType::SlidingAttention).then_some(*value))
        .unwrap_or(kv_heads[0]);
    let full_kv_heads = layer_types
        .iter()
        .zip(&kv_heads)
        .find_map(|(kind, value)| (*kind == LayerType::FullAttention).then_some(*value))
        .unwrap_or(sliding_kv_heads);
    for (kind, value) in layer_types.iter().zip(&kv_heads) {
        let expected = if *kind == LayerType::FullAttention {
            full_kv_heads
        } else {
            sliding_kv_heads
        };
        if *value != expected {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 assistant uses non-uniform KV-head counts within one attention type"
                    .into(),
            ));
        }
    }

    let global_head_dim = gemma4::gguf_i32(metadata, &key("attention.key_length"), stream)?;
    let head_dim = gemma4::gguf_optional_i64(metadata, &key("attention.key_length_swa"), stream)?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| {
            Error::UnsupportedArchitecture("Gemma 4 assistant SWA head size exceeds i32".into())
        })?
        .unwrap_or(global_head_dim);
    let vocab_size = match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len()).map_err(|_| {
            Error::UnsupportedArchitecture(
                "Gemma 4 assistant tokenizer vocabulary exceeds i32".into(),
            )
        })?,
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 assistant tokenizer.ggml.tokens metadata has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 assistant GGUF is missing tokenizer.ggml.tokens".into(),
            ));
        }
    };
    let full_rope_theta =
        gemma4::gguf_optional_f32(metadata, &key("rope.freq_base"), stream)?.unwrap_or(1_000_000.0);
    let sliding_rope_theta =
        gemma4::gguf_optional_f32(metadata, &key("rope.freq_base_swa"), stream)?
            .unwrap_or(10_000.0);
    let rope_parameters = Some(HashMap::from([
        (
            "full_attention".into(),
            HashMap::from([
                (
                    "rope_type".into(),
                    FloatOrString::String("proportional".into()),
                ),
                ("partial_rotary_factor".into(), FloatOrString::Float(0.25)),
                ("rope_theta".into(), FloatOrString::Float(full_rope_theta)),
            ]),
        ),
        (
            "sliding_attention".into(),
            HashMap::from([
                ("rope_type".into(), FloatOrString::String("default".into())),
                (
                    "rope_theta".into(),
                    FloatOrString::Float(sliding_rope_theta),
                ),
            ]),
        ),
    ]));
    let num_kv_shared_layers =
        gemma4::gguf_optional_i64(metadata, &key("attention.shared_kv_layers"), stream)?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| {
                Error::UnsupportedArchitecture(
                    "Gemma 4 assistant shared-KV count exceeds i32".into(),
                )
            })?
            .unwrap_or(num_hidden_layers);
    let hidden_size_per_layer_input =
        gemma4::gguf_optional_i64(metadata, &key("embedding_length_per_layer_input"), stream)?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| {
                Error::UnsupportedArchitecture(
                    "Gemma 4 assistant per-layer input size exceeds i32".into(),
                )
            })?
            .unwrap_or(0);
    let draft_tokens = gemma4::gguf_optional_i64(metadata, &key("nextn_predict_layers"), stream)?
        .unwrap_or(3)
        .checked_add(1)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("Gemma 4 assistant draft block size is invalid".into())
        })?;

    Ok(Gemma4AssistantConfig {
        model_type: default_model_type(),
        backbone_hidden_size: gemma4::gguf_i32(metadata, &key("embedding_length_out"), stream)?,
        use_ordered_embeddings: checkpoint.contains_gguf_tensor("nextn.centroids.weight")
            || checkpoint.contains_gguf_tensor("mtp.centroids.weight"),
        num_centroids: default_num_centroids(),
        centroid_intermediate_top_k: default_centroid_top_k(),
        tie_word_embeddings: !checkpoint.contains_gguf_tensor("output.weight"),
        block_size: draft_tokens,
        text_config: ModelArgs {
            model_type: "gemma4".into(),
            hidden_size: gemma4::gguf_i32(metadata, &key("embedding_length"), stream)?,
            num_hidden_layers,
            intermediate_size,
            use_double_wide_mlp: false,
            feed_forward_lengths: Some(feed_forward_lengths),
            num_attention_heads: gemma4::gguf_i32(metadata, &key("attention.head_count"), stream)?,
            rms_norm_eps: gemma4::gguf_f32(
                metadata,
                &key("attention.layer_norm_rms_epsilon"),
                stream,
            )?,
            vocab_size,
            pad_token_id: gemma4::gguf_optional_i64(
                metadata,
                "tokenizer.ggml.padding_token_id",
                stream,
            )?
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or(0),
            num_key_value_heads: sliding_kv_heads,
            num_global_key_value_heads: (full_kv_heads != sliding_kv_heads)
                .then_some(full_kv_heads),
            max_position_embeddings: gemma4::gguf_i32(metadata, &key("context_length"), stream)?,
            rope_theta: sliding_rope_theta,
            head_dim,
            global_head_dim: (global_head_dim != head_dim).then_some(global_head_dim),
            tie_word_embeddings: !checkpoint.contains_gguf_tensor("output.weight"),
            attention_bias: false,
            attention_k_eq_v: false,
            quantized: false,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            quantization_group_size: 64,
            quantization_bits: 4,
            hidden_size_per_layer_input,
            vocab_size_per_layer_input: (hidden_size_per_layer_input > 0).then_some(vocab_size),
            num_kv_shared_layers,
            layer_types,
            sliding_window: gemma4::gguf_optional_i64(
                metadata,
                &key("attention.sliding_window"),
                stream,
            )?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| {
                Error::UnsupportedArchitecture(
                    "Gemma 4 assistant sliding window exceeds i32".into(),
                )
            })?,
            final_logit_softcapping: None,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            rope_scaling: None,
            rope_parameters,
        },
        quantization: None,
    })
}

fn translate_gguf_weight_name(name: &str) -> String {
    if matches!(
        name,
        "mtp.token_ordering.weight" | "nextn.token_ordering.weight"
    ) {
        return "masked_embedding.token_ordering".into();
    }
    const ASSISTANT_PARAMETERS: [(&str, &str); 6] = [
        ("mtp.pre_projection", "pre_projection"),
        ("nextn.pre_projection", "pre_projection"),
        ("mtp.post_projection", "post_projection"),
        ("nextn.post_projection", "post_projection"),
        ("mtp.centroids", "masked_embedding.centroids"),
        ("nextn.centroids", "masked_embedding.centroids"),
    ];
    for (source, target) in ASSISTANT_PARAMETERS {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }

    let target = crate::models::gemma4::translate_gguf_weight_name(name);
    target
        .strip_prefix("model.language_model.")
        .map_or(target.clone(), |parameter| format!("model.{parameter}"))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        time::{SystemTime, UNIX_EPOCH},
    };

    use safemlx::{
        module::ModuleParameters,
        ops::{GgufMetadataArray, GgufMetadataValue},
        Array, Device, DeviceType, ExecutionContext,
    };

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
    fn draft_state_forks_without_shared_mutable_progress() {
        let original = super::Gemma4AssistantDraftState {
            shared_kv: HashMap::new(),
            kv_offset: 7,
            hidden: Array::from_slice(&[1.0f32], &[1, 1, 1]),
        };
        let mut fork = original.clone();
        fork.kv_offset += 1;
        fork.hidden = Array::from_slice(&[2.0f32], &[1, 1, 1]);

        assert_eq!(original.kv_offset, 7);
        assert_eq!(fork.kv_offset, 8);
        assert_eq!(
            original.hidden.evaluated().unwrap().as_slice::<f32>(),
            &[1.0]
        );
        assert_eq!(fork.hidden.evaluated().unwrap().as_slice::<f32>(), &[2.0]);
    }

    #[test]
    fn translates_published_gguf_names() {
        let cases = [
            ("token_embd.weight", "model.embed_tokens.weight"),
            (
                "blk.2.attn_output.weight",
                "model.layers.2.self_attn.o_proj.weight",
            ),
            (
                "blk.2.layer_output_scale.weight",
                "model.layers.2.layer_scalar",
            ),
            ("output_norm.weight", "model.norm.weight"),
            ("mtp.pre_projection.weight", "pre_projection.weight"),
            ("nextn.pre_projection.scales", "pre_projection.scales"),
            ("nextn.post_projection.biases", "post_projection.biases"),
            (
                "mtp.token_ordering.weight",
                "masked_embedding.token_ordering",
            ),
        ];
        for (gguf, model) in cases {
            assert_eq!(super::translate_gguf_weight_name(gguf), model);
        }
    }

    #[test]
    fn derives_published_assistant_config_from_gguf_metadata() {
        let metadata = HashMap::from([
            (
                "gemma4-assistant.block_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "gemma4-assistant.attention.sliding_window_pattern".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true, true, true, false])),
            ),
            (
                "gemma4-assistant.feed_forward_length".into(),
                GgufMetadataValue::Uint32(8192),
            ),
            (
                "gemma4-assistant.attention.head_count_kv".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Int32(vec![8, 8, 8, 2])),
            ),
            (
                "gemma4-assistant.attention.key_length".into(),
                GgufMetadataValue::Uint32(512),
            ),
            (
                "gemma4-assistant.attention.key_length_swa".into(),
                GgufMetadataValue::Uint32(256),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec!["token".into(); 32])),
            ),
            (
                "gemma4-assistant.rope.freq_base".into(),
                GgufMetadataValue::Float32(1_000_000.0),
            ),
            (
                "gemma4-assistant.rope.freq_base_swa".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            (
                "gemma4-assistant.attention.shared_kv_layers".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "gemma4-assistant.embedding_length_per_layer_input".into(),
                GgufMetadataValue::Uint32(0),
            ),
            (
                "gemma4-assistant.nextn_predict_layers".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "gemma4-assistant.embedding_length_out".into(),
                GgufMetadataValue::Uint32(2816),
            ),
            (
                "gemma4-assistant.embedding_length".into(),
                GgufMetadataValue::Uint32(1024),
            ),
            (
                "gemma4-assistant.attention.head_count".into(),
                GgufMetadataValue::Uint32(16),
            ),
            (
                "gemma4-assistant.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "gemma4-assistant.context_length".into(),
                GgufMetadataValue::Uint32(262144),
            ),
            (
                "gemma4-assistant.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(1024),
            ),
        ]);
        let arrays = HashMap::<String, Array>::new();
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let config =
            super::gemma4_assistant_config_from_gguf(&arrays, &metadata, context.stream()).unwrap();

        assert_eq!(config.backbone_hidden_size, 2816);
        assert_eq!(config.block_size, 5);
        assert_eq!(config.text_config.hidden_size, 1024);
        assert_eq!(config.text_config.num_hidden_layers, 4);
        assert_eq!(config.text_config.num_key_value_heads, 8);
        assert_eq!(config.text_config.num_global_key_value_heads, Some(2));
        assert_eq!(config.text_config.head_dim, 256);
        assert_eq!(config.text_config.global_head_dim, Some(512));
        assert_eq!(config.text_config.vocab_size, 32);
    }

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
