//! Gemma 4 text model implementation and loader.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::Path,
    time::Instant,
};

use safemlx::{
    builder::Builder,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    native_quantization::{
        native_grouped_linear, native_selected_down_reduce, native_selected_gate_up,
        NativeQuantizationFormat, NativeQuantizationStats, NativeQuantizedTensor,
    },
    nn,
    ops::{
        concatenate_axis, dequantize_with_mode, gather_grouped_rows, grouped_matmul,
        indexing::{NewAxis, TryIndexOp},
        mean_axis, quantized_matmul_with_mode, quantized_packed_dimension, r#where, rsqrt,
        sum_axis, tanh, topk_route_plan, GgufCheckpoint, GgufEndian, GgufMetadataValue, GgufType,
        QuantizationMode,
    },
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

pub use super::common::generation::sample;
use super::{
    gemma4_audio::{Gemma4AudioConfig, Gemma4AudioTower},
    gemma4_multimodal::Gemma4ModalityEmbedder,
    gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower},
};

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    inspection::ActivationObserver,
    models::{
        common::{
            self,
            attention::{
                attention_probabilities, batch_seq, finish_attention, reshape_attention_projection,
            },
            generation::CausalLm,
            moe::{affine_grouped_linear_with_options, top_k_softmax_routing, weighted_route_sum},
        },
        input,
    },
    quantization::{AffineQuantization, WeightQuantization},
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    weights::{
        gguf_metadata, gguf_quantization_configs, load_named_array_strict,
        load_safetensors_quantized_strict, load_safetensors_strict, GgufTensorNames,
        StrictLoadConfig, StrictLoadReport,
    },
};

#[derive(Debug, Clone, Default)]
/// Profiling counters accumulated by Gemma 4 when profiling is enabled.
pub struct PerfStats {
    /// Time spent evaluating token embeddings.
    pub embed_s: f64,
    /// Time spent evaluating per-layer input embeddings/projections.
    pub per_layer_inputs_s: f64,
    /// Time spent evaluating attention outputs.
    pub attention_s: f64,
    /// Time spent evaluating MLP outputs.
    pub mlp_s: f64,
    /// Time spent evaluating per-layer input residuals.
    pub per_layer_residual_s: f64,
    /// Time spent evaluating final normalization.
    pub final_norm_s: f64,
    /// Time spent projecting hidden states to logits.
    pub lm_head_s: f64,
}

impl PerfStats {
    /// Returns the sum of all profiled component durations.
    pub fn component_total_s(&self) -> f64 {
        self.embed_s
            + self.per_layer_inputs_s
            + self.attention_s
            + self.mlp_s
            + self.per_layer_residual_s
            + self.final_norm_s
            + self.lm_head_s
    }

    fn add(&mut self, component: PerfComponent, elapsed_s: f64) {
        match component {
            PerfComponent::Embed => self.embed_s += elapsed_s,
            PerfComponent::PerLayerInputs => self.per_layer_inputs_s += elapsed_s,
            PerfComponent::Attention => self.attention_s += elapsed_s,
            PerfComponent::Mlp => self.mlp_s += elapsed_s,
            PerfComponent::PerLayerResidual => self.per_layer_residual_s += elapsed_s,
            PerfComponent::FinalNorm => self.final_norm_s += elapsed_s,
            PerfComponent::LmHead => self.lm_head_s += elapsed_s,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PerfComponent {
    Embed,
    PerLayerInputs,
    Attention,
    Mlp,
    PerLayerResidual,
    FinalNorm,
    LmHead,
}

thread_local! {
    static PERF_STATS: RefCell<Option<PerfStats>> = const { RefCell::new(None) };
}

/// Enables or disables per-thread Gemma 4 profiling.
pub fn set_perf_profiling(enabled: bool) {
    PERF_STATS.with(|stats| {
        *stats.borrow_mut() = enabled.then(PerfStats::default);
    });
}

/// Resets per-thread Gemma 4 profiling counters.
pub fn reset_perf_stats() {
    PERF_STATS.with(|stats| {
        if let Some(stats) = stats.borrow_mut().as_mut() {
            *stats = PerfStats::default();
        }
    });
}

/// Returns the current per-thread profiling counters, if profiling is enabled.
pub fn perf_stats() -> Option<PerfStats> {
    PERF_STATS.with(|stats| stats.borrow().clone())
}

fn profile_arrays(component: PerfComponent, arrays: &[&Array]) -> Result<(), Exception> {
    let enabled = PERF_STATS.with(|stats| stats.borrow().is_some());
    if !enabled {
        return Ok(());
    }

    let start = Instant::now();
    eval(arrays.iter().copied())?;
    let elapsed_s = start.elapsed().as_secs_f64();
    PERF_STATS.with(|stats| {
        if let Some(stats) = stats.borrow_mut().as_mut() {
            stats.add(component, elapsed_s);
        }
    });
    Ok(())
}

fn profile_array(component: PerfComponent, array: &Array) -> Result<(), Exception> {
    profile_arrays(component, &[array])
}

#[allow(clippy::too_many_arguments)]
fn sliding_window_prefill_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    sliding_window: i32,
    position_offset: i32,
    batch: i32,
    seq_len: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let q_shape = queries.shape();
    let k_shape = keys.shape();
    if q_shape.len() != 4 || k_shape.len() != 4 || values.shape().len() != 4 {
        return Err(Exception::custom(
            "Gemma 4 sliding prefill attention expects rank-4 Q/K/V",
        ));
    }
    let q_len = q_shape[2];
    let kv_len = k_shape[2];
    if q_len != seq_len || kv_len != position_offset + seq_len {
        return Err(Exception::custom(
            "Gemma 4 sliding prefill attention requires full-length KV",
        ));
    }

    if position_offset + seq_len <= sliding_window + 1 {
        return safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            Some(ScaledDotProductAttentionMask::Causal),
            None,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq_len, -1], stream);
    }

    let chunk_size = 256;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < seq_len {
        let end = (start + chunk_size).min(seq_len);
        let query_abs_start = position_offset + start;
        let key_start = (query_abs_start - sliding_window).max(0);
        let key_end = position_offset + end;
        let relative_offset = query_abs_start - key_start;
        let query_chunk = queries.try_index_device((.., .., start..end, ..), stream)?;
        let key_chunk = keys.try_index_device((.., .., key_start..key_end, ..), stream)?;
        let value_chunk = values.try_index_device((.., .., key_start..key_end, ..), stream)?;
        let mask = create_causal_mask(
            end - start,
            Some(relative_offset),
            Some(sliding_window),
            None,
            stream,
        )?;
        let out = safemlx::fast::scaled_dot_product_attention(
            query_chunk,
            key_chunk,
            value_chunk,
            scale,
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
            stream,
        )?;
        chunks.push(out);
        start = end;
    }

    let refs = chunks.iter().collect::<Vec<_>>();
    concatenate_axis(&refs, 2, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq_len, -1], stream)
}

#[derive(Debug, Clone, Deserialize)]
/// Deserialized Gemma 4 text configuration used by this loader.
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    /// Effective text model type.
    pub model_type: String,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Dense MLP intermediate size.
    pub intermediate_size: i32,
    #[serde(default)]
    /// Whether the final shared-KV layers use twice the base MLP width.
    pub use_double_wide_mlp: bool,
    #[serde(skip)]
    /// Optional GGUF-provided per-layer MLP widths.
    pub feed_forward_lengths: Option<Vec<i32>>,
    /// Number of query attention heads.
    pub num_attention_heads: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    #[serde(default)]
    /// Padding token used for media positions in per-layer token embeddings.
    pub pad_token_id: i32,
    /// Number of key/value attention heads.
    pub num_key_value_heads: i32,
    #[serde(default)]
    /// Optional number of key/value heads for global attention layers.
    pub num_global_key_value_heads: Option<i32>,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    /// Default RoPE base frequency.
    pub rope_theta: f32,
    /// Per-head attention dimension.
    pub head_dim: i32,
    #[serde(default)]
    /// Optional per-head dimension for global attention layers.
    pub global_head_dim: Option<i32>,
    #[serde(default = "default_true")]
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    #[serde(default)]
    /// Whether attention projection layers include bias terms.
    pub attention_bias: bool,
    #[serde(default)]
    /// Whether full-attention keys are reused as values.
    pub attention_k_eq_v: bool,
    #[serde(skip)]
    /// Whether Gemma-specific quantized tensors are expected.
    pub quantized: bool,
    #[serde(skip)]
    /// Model-wide safetensors quantization encoding, when present.
    pub weight_quantization: Option<WeightQuantization>,
    #[serde(skip)]
    /// Optional set of parameter weights that are quantized in a mixed checkpoint.
    pub quantized_weights: Option<HashSet<String>>,
    #[serde(skip)]
    /// Exact affine settings for mixed GGUF tensors.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    #[serde(skip)]
    /// Quantization group size for quantized weights.
    pub quantization_group_size: i32,
    #[serde(skip)]
    /// Quantization bit width for quantized weights.
    pub quantization_bits: i32,
    #[serde(default)]
    /// Hidden size for per-layer input embeddings.
    pub hidden_size_per_layer_input: i32,
    #[serde(default)]
    /// Optional vocabulary size for per-layer input embeddings.
    pub vocab_size_per_layer_input: Option<i32>,
    #[serde(default)]
    /// Number of final layers that reuse shared key/value states.
    pub num_kv_shared_layers: i32,
    #[serde(default)]
    /// Layer attention pattern.
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    /// Sliding-window size for sliding-attention layers.
    pub sliding_window: Option<i32>,
    #[serde(default)]
    /// Optional final-logit soft cap.
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    /// Whether the config requests a Gemma MoE block.
    pub enable_moe_block: bool,
    #[serde(default)]
    /// Number of experts when MoE is present.
    pub num_experts: Option<i32>,
    #[serde(default)]
    /// Number of selected experts when MoE is present.
    pub top_k_experts: Option<i32>,
    #[serde(default, alias = "expert_intermediate_size")]
    /// MoE intermediate size when MoE is present.
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    /// Default RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    /// Per-layer-type RoPE parameter overrides.
    pub rope_parameters: Option<HashMap<String, HashMap<String, FloatOrString>>>,
}

fn default_model_type() -> String {
    "gemma4".to_string()
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Gemma 4 attention-layer kind.
pub enum LayerType {
    /// Sliding-window attention layer.
    SlidingAttention,
    /// Full-context attention layer.
    FullAttention,
}

impl<'de> Deserialize<'de> for LayerType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "sliding_attention" => Ok(Self::SlidingAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(serde::de::Error::custom(format!(
                "Unsupported Gemma4 layer type '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4Config {
    text_config: ModelArgs,
    #[serde(default)]
    vision_config: Option<Gemma4VisionConfig>,
    #[serde(default)]
    image_token_id: Option<i32>,
    #[serde(default)]
    video_token_id: Option<i32>,
    #[serde(default)]
    audio_config: Option<Gemma4AudioConfig>,
    #[serde(default)]
    audio_token_id: Option<i32>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    quantization: Option<Value>,
}

impl ModelArgs {
    pub(crate) fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.weight_quantization.or_else(|| {
            self.quantized.then(|| {
                AffineQuantization::new(self.quantization_group_size, self.quantization_bits)
                    .expect("validated affine quantization")
                    .into()
            })
        })
    }

    pub(crate) fn quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        if let Some(config) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name))
        {
            return Some(*config);
        }
        self.is_quantized(weight_name)
            .then(|| self.weight_quantization())
            .flatten()
    }

    fn is_quantized(&self, weight_name: &str) -> bool {
        self.quantized
            && self
                .quantized_weights
                .as_ref()
                .is_none_or(|weights| weights.contains(weight_name))
    }

    fn feed_forward_length_for_layer(&self, layer_index: usize) -> i32 {
        if let Some(lengths) = &self.feed_forward_lengths {
            return lengths
                .get(layer_index)
                .copied()
                .unwrap_or(self.intermediate_size);
        }
        let first_shared_layer = self.num_hidden_layers - self.num_kv_shared_layers;
        if self.use_double_wide_mlp && layer_index as i32 >= first_shared_layer {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }

    fn for_layer(&self, layer_type: LayerType) -> Self {
        let mut args = self.clone();
        if layer_type == LayerType::FullAttention {
            if let Some(global_head_dim) = self.global_head_dim {
                args.head_dim = global_head_dim;
            }
            if let Some(global_kv_heads) = self.num_global_key_value_heads {
                args.num_key_value_heads = global_kv_heads;
            }
        }
        args.rope_theta = self.rope_theta_for_layer(layer_type);
        args.rope_scaling = self.rope_scaling_for_layer(layer_type);
        args
    }

    pub(crate) fn layer_type(&self, index: usize) -> LayerType {
        self.layer_types
            .get(index)
            .copied()
            .unwrap_or(LayerType::FullAttention)
    }

    fn rope_theta_for_layer(&self, layer_type: LayerType) -> f32 {
        let key = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
        };
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.get(key))
            .and_then(|params| params.get("rope_theta"))
            .and_then(|value| match value {
                FloatOrString::Float(v) => Some(*v),
                FloatOrString::String(s) => s.parse().ok(),
                FloatOrString::Bool(_) => None,
            })
            .unwrap_or(self.rope_theta)
    }

    fn rope_scaling_for_layer(
        &self,
        layer_type: LayerType,
    ) -> Option<HashMap<String, FloatOrString>> {
        let key = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
        };
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.get(key).cloned())
    }
}

fn partial_rotary_dims(head_dim: i32, scaling: &Option<HashMap<String, FloatOrString>>) -> i32 {
    if matches!(
        scaling
            .as_ref()
            .and_then(|scaling| scaling.get("rope_type")),
        Some(FloatOrString::String(rope_type)) if rope_type == "proportional"
    ) {
        return head_dim;
    }

    let partial_factor = scaling
        .as_ref()
        .and_then(|scaling| scaling.get("partial_rotary_factor"))
        .and_then(|value| match value {
            FloatOrString::Float(v) => Some(*v),
            FloatOrString::String(s) => s.parse().ok(),
            FloatOrString::Bool(_) => None,
        })
        .unwrap_or(1.0);
    ((head_dim as f32 * partial_factor).round() as i32).clamp(2, head_dim)
}

fn needs_generated_sliding_mask(
    seq_len: i32,
    position_offset: i32,
    sliding_window: Option<i32>,
) -> bool {
    seq_len > 1
        || sliding_window.is_some_and(|window| position_offset.saturating_add(seq_len) > window)
}

fn maybe_quantized_linear_with_config(
    input_dims: i32,
    output_dims: i32,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    match quantization {
        Some(WeightQuantization::GgufIQuant { ggml_type, endian }) => {
            Ok(MaybeQuantized::Quantized(nn::QuantizedLinear::unloaded_iq(
                input_dims,
                output_dims,
                ggml_type,
                endian,
                false,
                stream,
            )?))
        }
        Some(config) => Ok(MaybeQuantized::Quantized(
            nn::QuantizedLinear::unloaded_with_mode(
                input_dims,
                output_dims,
                config.group_size(),
                config.bits(),
                config.mode(),
                false,
                stream,
            )?,
        )),
        None => Ok(MaybeQuantized::Original(nn::Linear::unloaded(
            input_dims,
            output_dims,
            false,
            Dtype::Float32,
            stream,
        )?)),
    }
}

pub(super) fn maybe_quantized_linear_with_bias(
    quantization: Option<WeightQuantization>,
    input_dims: i32,
    output_dims: i32,
    bias: bool,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    if let Some(WeightQuantization::GgufIQuant { ggml_type, endian }) = quantization {
        Ok(MaybeQuantized::Quantized(nn::QuantizedLinear::unloaded_iq(
            input_dims,
            output_dims,
            ggml_type,
            endian,
            bias,
            stream,
        )?))
    } else if let Some(config) = quantization {
        Ok(MaybeQuantized::Quantized(
            nn::QuantizedLinear::unloaded_with_mode(
                input_dims,
                output_dims,
                config.group_size(),
                config.bits(),
                config.mode(),
                bias,
                stream,
            )?,
        ))
    } else {
        Ok(MaybeQuantized::Original(nn::Linear::unloaded(
            input_dims,
            output_dims,
            bias,
            Dtype::Float32,
            stream,
        )?))
    }
}

pub(super) fn rms_norm_without_scale(
    x: &Array,
    eps: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let variance = mean_axis(&x.square(stream)?, -1, true, stream)?;
    x.multiply(
        rsqrt(variance.add(Array::from_f32(eps), stream)?, stream)?,
        stream,
    )
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 attention layer.
pub struct Attention {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads.
    pub n_kv_heads: i32,
    /// Attention scaling factor.
    pub scale: f32,
    /// Whether key projections are reused as value projections.
    pub attention_k_eq_v: bool,
    /// Layer attention pattern.
    pub layer_type: LayerType,
    /// Whether this layer reads shared key/value states from another layer.
    pub is_kv_shared_layer: bool,
    /// Whether this layer stores full-length key/value states for sharing.
    pub store_full_length_kv: bool,

    #[quantizable]
    #[param]
    /// Query projection.
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Optional key projection. Shared-KV layers reuse keys from earlier layers.
    pub k_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    /// Optional value projection.
    pub v_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    /// Output projection.
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Query normalization.
    pub q_norm: nn::RmsNorm,
    #[param]
    /// Optional key normalization. Shared-KV layers reuse normalized keys from earlier layers.
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    /// Rotary position embedding module.
    pub rope: RopeVariant,
}

impl Attention {
    /// Creates an unloaded Gemma 4 attention layer.
    pub fn new(
        args: &ModelArgs,
        layer_type: LayerType,
        layer_idx: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = 1.0;
        let attention_k_eq_v = args.attention_k_eq_v && layer_type == LayerType::FullAttention;
        let first_kv_shared_layer_idx = args.num_hidden_layers - args.num_kv_shared_layers;
        let is_kv_shared_layer =
            args.num_kv_shared_layers > 0 && layer_idx as i32 >= first_kv_shared_layer_idx;
        let store_full_length_kv = if args.num_kv_shared_layers > 0 && !is_kv_shared_layer {
            let first_kv_shared_layer_idx = first_kv_shared_layer_idx.max(0) as usize;
            (0..first_kv_shared_layer_idx)
                .rev()
                .find(|index| args.layer_type(*index) == layer_type)
                .is_some_and(|index| index == layer_idx)
        } else {
            false
        };

        let prefix = format!("model.language_model.layers.{layer_idx}.self_attn");
        let q_proj = maybe_quantized_linear_with_config(
            dim,
            n_heads * head_dim,
            args.quantization_for(&format!("{prefix}.q_proj.weight")),
            stream,
        )?;
        let k_proj = if is_kv_shared_layer {
            None
        } else {
            Some(maybe_quantized_linear_with_config(
                dim,
                n_kv_heads * head_dim,
                args.quantization_for(&format!("{prefix}.k_proj.weight")),
                stream,
            )?)
        };
        let v_proj = if is_kv_shared_layer || attention_k_eq_v {
            None
        } else {
            Some(maybe_quantized_linear_with_config(
                dim,
                n_kv_heads * head_dim,
                args.quantization_for(&format!("{prefix}.v_proj.weight")),
                stream,
            )?)
        };
        let o_proj = maybe_quantized_linear_with_config(
            n_heads * head_dim,
            dim,
            args.quantization_for(&format!("{prefix}.o_proj.weight")),
            stream,
        )?;

        let q_norm = nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?;
        let k_norm = if is_kv_shared_layer {
            None
        } else {
            Some(nn::RmsNorm::unloaded(
                head_dim,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?)
        };

        let rope_dims = partial_rotary_dims(head_dim, &args.rope_scaling);
        let rope = initialize_rope(
            rope_dims,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
            stream,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            attention_k_eq_v,
            layer_type,
            is_kv_shared_layer,
            store_full_length_kv,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
        })
    }
}

/// Input for a Gemma 4 attention or transformer block.
pub struct AttentionInput<'a, C> {
    /// Hidden states with shape `[batch, sequence, hidden]`.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional mutable key/value cache.
    pub cache: Option<&'a mut C>,
    /// RoPE/cache position offset.
    pub position_offset: i32,
    /// Optional per-layer input embedding slice.
    pub per_layer_input: Option<&'a Array>,
    /// Shared key/value states keyed by layer type.
    pub shared_kv: Option<&'a mut HashMap<LayerType, (Array, Array)>>,
    /// Whether generated sliding-window masks should be suppressed.
    pub disable_generated_mask: bool,
    /// Sliding-window size when the mask was generated by this block.
    pub generated_sliding_window: Option<i32>,
}

/// Hidden states and shared KV state returned by the Gemma 4 text body.
pub struct Gemma4TextOutput {
    /// Final normalized hidden states.
    pub hidden: Array,
    /// Hidden states before final normalization.
    pub pre_norm_hidden: Array,
    /// Shared key/value states captured during the pass.
    pub shared_kv_states: HashMap<LayerType, (Array, Array)>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            mut cache,
            position_offset,
            mut shared_kv,
            generated_sliding_window,
            ..
        } = input;

        let (B, L) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        let mut queries = self.q_norm.forward(
            &reshape_attention_projection(queries, B, L, self.n_heads, stream)?,
            stream,
        )?;
        let offset = position_offset;
        queries = self.rope.forward(
            nn::RopeInputBuilder::new(&queries).offset(offset).build()?,
            stream,
        )?;

        let (keys, values) = if self.is_kv_shared_layer {
            shared_kv
                .as_ref()
                .and_then(|shared_kv| shared_kv.get(&self.layer_type))
                .cloned()
                .ok_or_else(|| Exception::custom("missing shared Gemma 4 KV states"))?
        } else {
            let keys = self
                .k_proj
                .as_mut()
                .ok_or_else(|| Exception::custom("missing Gemma 4 key projection"))?
                .forward(x, stream)?;
            let values = if self.attention_k_eq_v {
                keys.clone()
            } else {
                self.v_proj
                    .as_mut()
                    .ok_or_else(|| Exception::custom("missing Gemma 4 value projection"))?
                    .forward(x, stream)?
            };
            let mut keys = self
                .k_norm
                .as_mut()
                .ok_or_else(|| Exception::custom("missing Gemma 4 key normalization"))?
                .forward(
                    &reshape_attention_projection(keys, B, L, self.n_kv_heads, stream)?,
                    stream,
                )?;
            let mut values = rms_norm_without_scale(
                &values.reshape(&[B, L, self.n_kv_heads, -1], stream)?,
                1e-6,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
            keys = self.rope.forward(
                nn::RopeInputBuilder::new(&keys).offset(offset).build()?,
                stream,
            )?;
            if let Some(cache) = cache.as_mut() {
                (keys, values) = cache.update_and_fetch(keys, values, stream)?;
            }
            if let Some(shared_kv) = shared_kv.as_mut() {
                shared_kv.insert(self.layer_type, (keys.clone(), values.clone()));
            }
            (keys, values)
        };

        let attention_cache = if self.is_kv_shared_layer || shared_kv.is_some() {
            None
        } else {
            cache
        };
        let generated_sliding_window = generated_sliding_window.filter(|sliding_window| {
            attention_cache.is_none()
                && mask.is_some()
                && L > 1
                && self.layer_type == LayerType::SlidingAttention
                && keys.shape()[2] == position_offset + L
                && (position_offset + L <= *sliding_window + 1 || L >= 1024)
        });
        let output = if let Some(generated_sliding_window) = generated_sliding_window {
            sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                generated_sliding_window,
                position_offset,
                B,
                L,
                stream,
            )?
        } else {
            finish_attention(
                queries,
                keys,
                values,
                attention_cache,
                self.scale,
                mask,
                B,
                L,
                stream,
            )?
        };

        self.o_proj.forward(&output, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        if let Some(k_proj) = &mut self.k_proj {
            k_proj.training_mode(mode);
        }
        if let Some(v_proj) = &mut self.v_proj {
            v_proj.training_mode(mode);
        }
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        if let Some(k_norm) = &mut self.k_norm {
            k_norm.training_mode(mode);
        }
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

impl Attention {
    /// Forward pass that reports attention activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput {
            x,
            mask,
            mut cache,
            position_offset,
            mut shared_kv,
            generated_sliding_window,
            ..
        } = input;

        let (batch, seq_len) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.q_proj"), &queries)?;
        let mut queries = self.q_norm.forward(
            &reshape_attention_projection(queries, batch, seq_len, self.n_heads, stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.q_norm"), &queries)?;
        queries = self.rope.forward(
            nn::RopeInputBuilder::new(&queries)
                .offset(position_offset)
                .build()?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.queries_rope"), &queries)?;

        let (keys, values) = if self.is_kv_shared_layer {
            let (keys, values) = shared_kv
                .as_ref()
                .and_then(|shared_kv| shared_kv.get(&self.layer_type))
                .cloned()
                .ok_or_else(|| Exception::custom("missing shared Gemma 4 KV states"))?;
            observer.observe(&format!("{prefix}.keys_shared"), &keys)?;
            observer.observe(&format!("{prefix}.values_shared"), &values)?;
            (keys, values)
        } else {
            let keys = self
                .k_proj
                .as_mut()
                .ok_or_else(|| Exception::custom("missing Gemma 4 key projection"))?
                .forward(x, stream)?;
            observer.observe(&format!("{prefix}.k_proj"), &keys)?;
            let values = if self.attention_k_eq_v {
                keys.clone()
            } else {
                let values = self
                    .v_proj
                    .as_mut()
                    .ok_or_else(|| Exception::custom("missing Gemma 4 value projection"))?
                    .forward(x, stream)?;
                observer.observe(&format!("{prefix}.v_proj"), &values)?;
                values
            };
            let mut keys = self
                .k_norm
                .as_mut()
                .ok_or_else(|| Exception::custom("missing Gemma 4 key normalization"))?
                .forward(
                    &reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?,
                    stream,
                )?;
            observer.observe(&format!("{prefix}.k_norm"), &keys)?;
            let mut values = rms_norm_without_scale(
                &values.reshape(&[batch, seq_len, self.n_kv_heads, -1], stream)?,
                1e-6,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
            observer.observe(&format!("{prefix}.values"), &values)?;
            keys = self.rope.forward(
                nn::RopeInputBuilder::new(&keys)
                    .offset(position_offset)
                    .build()?,
                stream,
            )?;
            observer.observe(&format!("{prefix}.keys_rope"), &keys)?;
            if let Some(cache) = cache.as_mut() {
                (keys, values) = cache.update_and_fetch(keys, values, stream)?;
            }
            observer.observe(&format!("{prefix}.keys_cache"), &keys)?;
            observer.observe(&format!("{prefix}.values_cache"), &values)?;
            if let Some(shared_kv) = shared_kv.as_mut() {
                shared_kv.insert(self.layer_type, (keys.clone(), values.clone()));
                observer.observe(&format!("{prefix}.shared_keys_stored"), &keys)?;
                observer.observe(&format!("{prefix}.shared_values_stored"), &values)?;
            }
            (keys, values)
        };

        let attention_probs = attention_probabilities(&queries, &keys, self.scale, mask, stream)?;
        observer.observe(&format!("{prefix}.attention_probs"), &attention_probs)?;

        let attention_cache = if self.is_kv_shared_layer || shared_kv.is_some() {
            None
        } else {
            cache
        };
        let generated_sliding_window = generated_sliding_window.filter(|sliding_window| {
            attention_cache.is_none()
                && mask.is_some()
                && seq_len > 1
                && self.layer_type == LayerType::SlidingAttention
                && keys.shape()[2] == position_offset + seq_len
                && (position_offset + seq_len <= *sliding_window + 1 || seq_len >= 1024)
        });
        let output = if let Some(generated_sliding_window) = generated_sliding_window {
            sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                generated_sliding_window,
                position_offset,
                batch,
                seq_len,
                stream,
            )?
        } else {
            finish_attention(
                queries,
                keys,
                values,
                attention_cache,
                self.scale,
                mask,
                batch,
                seq_len,
                stream,
            )?
        };
        observer.observe(&format!("{prefix}.attention"), &output)?;

        let output = self.o_proj.forward(&output, stream)?;
        observer.observe(&format!("{prefix}.o_proj"), &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 feed-forward layer.
pub struct Mlp {
    /// Dense intermediate size.
    pub hidden_dim: i32,
    #[quantizable]
    #[param]
    /// Gate projection.
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Down projection back to hidden size.
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Up projection.
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    /// Creates an unloaded Gemma 4 MLP.
    pub fn new(
        dim: i32,
        hidden_dim: i32,
        quantized: bool,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let quantization = if quantized {
            Some(
                AffineQuantization::new(group_size, bits)
                    .map_err(|err| Exception::custom(err.to_string()))?
                    .into(),
            )
        } else {
            None
        };
        Self::new_selective(dim, hidden_dim, [quantization; 3], stream)
    }

    fn new_selective(
        dim: i32,
        hidden_dim: i32,
        quantization: [Option<WeightQuantization>; 3],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            hidden_dim,
            gate_proj: maybe_quantized_linear_with_config(
                dim,
                hidden_dim,
                quantization[0],
                stream,
            )?,
            down_proj: maybe_quantized_linear_with_config(
                hidden_dim,
                dim,
                quantization[1],
                stream,
            )?,
            up_proj: maybe_quantized_linear_with_config(dim, hidden_dim, quantization[2], stream)?,
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(input, stream)?;
        let up = self.up_proj.forward(input, stream)?;
        let down_proj_input = nn::gelu_approximate(gate, stream)?.multiply(up, stream)?;
        self.down_proj.forward(&down_proj_input, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

impl Mlp {
    /// Forward pass that reports MLP activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(input, stream)?;
        observer.observe(&format!("{prefix}.gate_proj"), &gate)?;

        let up = self.up_proj.forward(input, stream)?;
        observer.observe(&format!("{prefix}.up_proj"), &up)?;

        let activated_gate = nn::gelu_approximate(gate, stream)?;
        observer.observe(&format!("{prefix}.gate_activation"), &activated_gate)?;

        let down_proj_input = activated_gate.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.down_proj_input"), &down_proj_input)?;

        let output = self.down_proj.forward(&down_proj_input, stream)?;
        observer.observe(&format!("{prefix}.down_proj"), &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 MoE router with learned input and per-expert scales.
pub struct MoeRouter {
    /// Hidden dimension routed by this module.
    pub hidden_size: i32,
    /// Number of experts selected for each token.
    pub top_k: i32,
    /// RMS normalization epsilon.
    pub eps: f32,
    #[quantizable]
    #[param]
    /// Projection from normalized hidden states to expert logits.
    pub proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Learned scale applied to normalized router inputs.
    pub scale: Param<Array>,
    #[param]
    /// Learned multiplicative scale applied after top-k normalization.
    pub per_expert_scale: Param<Array>,
}

impl MoeRouter {
    fn new(
        args: &ModelArgs,
        layer_idx: usize,
        num_experts: i32,
        top_k: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.language_model.layers.{layer_idx}.router");
        Ok(Self {
            hidden_size: args.hidden_size,
            top_k,
            eps: args.rms_norm_eps,
            proj: maybe_quantized_linear_with_config(
                args.hidden_size,
                num_experts,
                args.quantization_for(&format!("{prefix}.proj.weight")),
                stream,
            )?,
            scale: Param::<Array>::unloaded(&[args.hidden_size], Dtype::Float32, stream)?,
            per_expert_scale: Param::<Array>::unloaded(&[num_experts], Dtype::Float32, stream)?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let normalized = rms_norm_without_scale(hidden_states, self.eps, stream)?;
        let scaled = normalized.multiply(self.scale.as_ref(), stream)?.multiply(
            Array::from_f32((self.hidden_size as f32).sqrt().recip()),
            stream,
        )?;
        let logits = self.proj.forward(&scaled, stream)?;
        let (indices, weights) = top_k_softmax_routing(&logits, self.top_k, stream)?;
        let expert_scales = self
            .per_expert_scale
            .as_ref()
            .take_axis(&indices, 0, stream)?;
        Ok((indices, weights.multiply(expert_scales, stream)?))
    }

    fn training_mode(&mut self, mode: bool) {
        self.proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// One expert-major projection with optional affine or MXFP4 storage.
pub struct ExpertProjection {
    /// Quantization encoding, when packed.
    pub quantization: Option<WeightQuantization>,
    /// Checkpoint-native IQ encoding, when packed as GGML blocks.
    pub iquant: Option<WeightQuantization>,
    /// Logical projection input dimension.
    pub input_dim: i32,
    /// Logical projection output dimension.
    pub output_dim: i32,
    /// Optional checkpoint-native expert-major storage.
    pub native: Option<NativeQuantizedTensor>,
    #[param]
    /// Projection weights shaped `[experts, output, input]`.
    pub weight: Param<Array>,
    #[param]
    /// Per-group scales for packed weights.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Per-group biases for affine packed weights.
    pub biases: Param<Option<Array>>,
}

impl ExpertProjection {
    fn new(
        num_experts: i32,
        output_dim: i32,
        input_dim: i32,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if let Some(iquant @ WeightQuantization::GgufIQuant { .. }) = quantization {
            let (ggml_type, _) = iquant.gguf_iquant().expect("IQ expert format");
            let (block_values, block_bytes) = ggml_type
                .block_and_bytes()
                .expect("canonical IQ block geometry");
            Ok(Self {
                quantization: None,
                iquant: Some(iquant),
                input_dim,
                output_dim,
                native: None,
                weight: Param::<Array>::unloaded(
                    &[
                        num_experts,
                        output_dim,
                        input_dim / block_values as i32 * block_bytes as i32,
                    ],
                    Dtype::Uint8,
                    stream,
                )?,
                scales: Param::new(None),
                biases: Param::new(None),
            })
        } else if let Some(quantization) = quantization {
            Ok(Self {
                quantization: Some(quantization),
                iquant: None,
                input_dim,
                output_dim,
                native: None,
                weight: Param::<Array>::unloaded(
                    &[
                        num_experts,
                        output_dim,
                        quantized_packed_dimension(input_dim, quantization.bits()),
                    ],
                    Dtype::Uint32,
                    stream,
                )?,
                scales: Param::<Option<Array>>::unloaded_some(
                    &[
                        num_experts,
                        output_dim,
                        input_dim / quantization.group_size(),
                    ],
                    if quantization == WeightQuantization::MxFp4 {
                        Dtype::Uint8
                    } else {
                        Dtype::Float16
                    },
                    stream,
                )?,
                biases: if quantization.has_biases() {
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            num_experts,
                            output_dim,
                            input_dim / quantization.group_size(),
                        ],
                        Dtype::Float16,
                        stream,
                    )?
                } else {
                    Param::new(None)
                },
            })
        } else {
            Ok(Self {
                quantization: None,
                iquant: None,
                input_dim,
                output_dim,
                native: None,
                weight: Param::<Array>::unloaded(
                    &[num_experts, output_dim, input_dim],
                    Dtype::Float32,
                    stream,
                )?,
                scales: Param::new(None),
                biases: Param::new(None),
            })
        }
    }

    fn forward(
        &self,
        hidden_states: &Array,
        group_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_with_sorted(hidden_states, group_ids, true, stream)
    }

    fn forward_with_sorted(
        &self,
        hidden_states: &Array,
        group_ids: &Array,
        sorted_indices: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if let Some(native) = &self.native {
            return native_grouped_linear(hidden_states, native, group_ids, stream);
        }
        if let Some(iquant) = self.iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.weight.dim(0), self.output_dim, self.input_dim],
                ggml_type,
                endian,
            )?;
            return native_grouped_linear(hidden_states, &native, group_ids, stream);
        }
        if let Some(quantization) = self.quantization {
            affine_grouped_linear_with_options(
                hidden_states,
                self.weight.as_ref(),
                self.scales
                    .as_ref()
                    .as_ref()
                    .expect("quantized Gemma 4 expert scales"),
                self.biases.as_ref().as_ref(),
                group_ids,
                quantization,
                true,
                sorted_indices,
                stream,
            )
        } else {
            grouped_matmul(
                hidden_states,
                &self.weight.as_ref().swap_axes(-1, -2, stream)?,
                group_ids,
                true,
                stream,
            )
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Published Gemma 4 SwitchGLU expert projection tree.
pub struct SwitchGluExperts {
    #[param]
    /// GELU gate projections.
    pub gate_proj: ExpertProjection,
    #[param]
    /// Multiplicative up projections.
    pub up_proj: ExpertProjection,
    #[param]
    /// Down projections back to model width.
    pub down_proj: ExpertProjection,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Routed Gemma 4 gated-GELU experts.
pub struct GemmaExperts {
    /// Number of routed experts.
    pub num_experts: i32,
    /// Model hidden dimension.
    pub hidden_dim: i32,
    /// Optional physical fused gate/up native bank.
    pub native_gate_up: Option<NativeQuantizedTensor>,
    #[param]
    /// SwitchGLU projection bank.
    pub switch_glu: SwitchGluExperts,
}

impl GemmaExperts {
    fn new(
        args: &ModelArgs,
        layer_idx: usize,
        num_experts: i32,
        intermediate_dim: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.language_model.layers.{layer_idx}.experts.switch_glu");
        Ok(Self {
            num_experts,
            hidden_dim: args.hidden_size,
            native_gate_up: None,
            switch_glu: SwitchGluExperts {
                gate_proj: ExpertProjection::new(
                    num_experts,
                    intermediate_dim,
                    args.hidden_size,
                    args.quantization_for(&format!("{prefix}.gate_proj.weight")),
                    stream,
                )?,
                up_proj: ExpertProjection::new(
                    num_experts,
                    intermediate_dim,
                    args.hidden_size,
                    args.quantization_for(&format!("{prefix}.up_proj.weight")),
                    stream,
                )?,
                down_proj: ExpertProjection::new(
                    num_experts,
                    args.hidden_size,
                    intermediate_dim,
                    args.quantization_for(&format!("{prefix}.down_proj.weight")),
                    stream,
                )?,
            },
        })
    }

    fn forward_chunk(
        &self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        if num_tokens == 1 {
            if let Some(fused_gate_up) = &self.native_gate_up {
                let expert_ids = top_k_index.reshape(&[-1], stream)?;
                let route_weights = top_k_weights.reshape(&[-1], stream)?;
                let intermediate = fused_gate_up.rows() / 2;
                let activated = native_selected_gate_up(
                    hidden_states,
                    fused_gate_up,
                    &expert_ids,
                    intermediate,
                    stream,
                )?;
                if let Some(native_down) = &self.switch_glu.down_proj.native {
                    return native_selected_down_reduce(
                        &activated,
                        native_down,
                        &expert_ids,
                        &route_weights,
                        stream,
                    );
                }
                let output = self.switch_glu.down_proj.forward_with_sorted(
                    &activated,
                    &expert_ids,
                    false,
                    stream,
                )?;
                return sum_axis(
                    output.multiply(route_weights.reshape(&[-1, 1], stream)?, stream)?,
                    0,
                    true,
                    stream,
                );
            }
        }
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let gate = self
            .switch_glu
            .gate_proj
            .forward(&hidden, &plan.sorted_group_ids, stream)?;
        let up = self
            .switch_glu
            .up_proj
            .forward(&hidden, &plan.sorted_group_ids, stream)?;
        let activated = nn::gelu_approximate(gate, stream)?.multiply(up, stream)?;
        let output =
            self.switch_glu
                .down_proj
                .forward(&activated, &plan.sorted_group_ids, stream)?;
        weighted_route_sum(output, top_k_weights, &plan, num_tokens, stream)
    }

    fn forward(
        &self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        const CHUNK_THRESHOLD: i32 = 64;
        const CHUNK_TOKENS: i32 = 32;

        let num_tokens = hidden_states.dim(0);
        if num_tokens <= CHUNK_THRESHOLD {
            return self.forward_chunk(hidden_states, top_k_index, top_k_weights, stream);
        }
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < num_tokens {
            let end = (start + CHUNK_TOKENS).min(num_tokens);
            outputs.push(self.forward_chunk(
                &hidden_states.try_index_device((start..end, ..), stream)?,
                &top_k_index.try_index_device((start..end, ..), stream)?,
                &top_k_weights.try_index_device((start..end, ..), stream)?,
                stream,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 0, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

struct Moe {
    router: MoeRouter,
    experts: GemmaExperts,
}

impl Moe {
    fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        let num_experts = args
            .num_experts
            .ok_or_else(|| Exception::custom("Gemma 4 MoE config is missing num_experts"))?;
        let top_k = args
            .top_k_experts
            .ok_or_else(|| Exception::custom("Gemma 4 MoE config is missing top_k_experts"))?;
        let intermediate_dim = args.moe_intermediate_size.ok_or_else(|| {
            Exception::custom("Gemma 4 MoE config is missing moe_intermediate_size")
        })?;
        if num_experts <= 0 || top_k <= 0 || top_k > num_experts || intermediate_dim <= 0 {
            return Err(Exception::custom(
                "Gemma 4 MoE expert count, top-k, and intermediate size must be positive and top-k cannot exceed the expert count",
            ));
        }
        Ok(Self {
            router: MoeRouter::new(args, layer_idx, num_experts, top_k, stream)?,
            experts: GemmaExperts::new(args, layer_idx, num_experts, intermediate_dim, stream)?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Gemma 4 embedding table with optional packed quantized storage.
pub struct Gemma4Embedding {
    #[param]
    /// Embedding weight tensor.
    pub weight: Param<Array>,
    /// Optional checkpoint-native embedding storage.
    pub native: Option<NativeQuantizedTensor>,
    #[param]
    /// Optional quantization scales.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Optional quantization biases.
    pub biases: Param<Option<Array>>,
    /// Whether the embedding is stored in quantized form.
    pub quantized: bool,
    /// Output hidden size.
    pub hidden_size: i32,
    /// Quantization group size.
    pub group_size: i32,
    /// Quantization bit width.
    pub bits: i32,
    /// Quantized weight encoding.
    pub mode: QuantizationMode,
}

impl Gemma4Embedding {
    /// Creates an unloaded embedding table.
    pub fn unloaded(
        vocab_size: i32,
        hidden_size: i32,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let quantized = quantization.is_some();
        let group_size = quantization.map_or(64, WeightQuantization::group_size);
        let bits = quantization.map_or(4, WeightQuantization::bits);
        let mode = quantization.map_or(QuantizationMode::Affine, WeightQuantization::mode);
        let packed_dim = quantized_packed_dimension(hidden_size, bits);
        Ok(Self {
            native: None,
            weight: if quantized {
                Param::<Array>::unloaded(&[vocab_size, packed_dim], Dtype::Uint32, stream)?
            } else {
                Param::<Array>::unloaded(&[vocab_size, hidden_size], Dtype::Float32, stream)?
            },
            scales: if quantized {
                Param::<Option<Array>>::unloaded_some(
                    &[vocab_size, hidden_size / group_size],
                    if mode == QuantizationMode::MxFp4 {
                        Dtype::Uint8
                    } else {
                        Dtype::Float32
                    },
                    stream,
                )?
            } else {
                Param::new(None)
            },
            biases: if quantization.is_some_and(WeightQuantization::has_biases) {
                Param::<Option<Array>>::unloaded_some(
                    &[vocab_size, hidden_size / group_size],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            quantized,
            mode,
            hidden_size,
            group_size,
            bits,
        })
    }

    /// Creates an initialized embedding table.
    pub fn new(
        vocab_size: i32,
        hidden_size: i32,
        quantized: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            native: None,
            weight: Param::new(if quantized {
                Array::from_slice(
                    &vec![
                        0u32;
                        (vocab_size * quantized_packed_dimension(hidden_size, bits)) as usize
                    ],
                    &[vocab_size, quantized_packed_dimension(hidden_size, bits)],
                )
            } else {
                nn::Embedding::new(vocab_size, hidden_size)?.weight.value
            }),
            scales: Param::new(if quantized {
                Some(Array::from_slice(
                    &vec![1.0f32; (vocab_size * (hidden_size / group_size)) as usize],
                    &[vocab_size, hidden_size / group_size],
                ))
            } else {
                None
            }),
            biases: Param::new(if quantized {
                Some(Array::from_slice(
                    &vec![0.0f32; (vocab_size * (hidden_size / group_size)) as usize],
                    &[vocab_size, hidden_size / group_size],
                ))
            } else {
                None
            }),
            quantized,
            mode: QuantizationMode::Affine,
            hidden_size,
            group_size,
            bits,
        })
    }

    /// Embeds token ids.
    pub fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        if let Some(native) = &self.native {
            return native.embedding(input, stream);
        }
        if !self.quantized {
            return self.weight.try_index_device(input, stream);
        }
        let original_shape = input.shape().to_vec();
        let flat = input.flatten(None, None, stream)?;
        let weight = self.weight.try_index_device(&flat, stream)?;
        let scales = self
            .scales
            .as_ref()
            .as_ref()
            .expect("quantized embedding scales")
            .try_index_device(&flat, stream)?;
        let biases = self
            .biases
            .as_ref()
            .as_ref()
            .map(|biases| biases.try_index_device(&flat, stream))
            .transpose()?;
        let out = dequantize_with_mode(
            &weight,
            &scales,
            biases.as_ref(),
            self.group_size,
            self.bits,
            self.mode,
            stream,
        )?;
        let shape = original_shape
            .into_iter()
            .chain(std::iter::once(self.hidden_size))
            .collect::<Vec<_>>();
        out.reshape(&shape, stream)
    }

    /// Applies the embedding table as a tied language-model head.
    pub fn as_linear(&self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        if let Some(native) = &self.native {
            return native.linear(x, true, stream);
        }
        if self.quantized {
            let scales = self
                .scales
                .as_ref()
                .as_ref()
                .expect("quantized embedding scales");
            let biases = self.biases.as_ref().as_ref();
            return quantized_matmul_with_mode(
                x,
                &self.weight,
                scales,
                biases,
                true,
                self.group_size,
                self.bits,
                self.mode,
                stream,
            );
        }
        safemlx::ops::matmul(x, self.weight.as_ref().transpose(stream)?, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 transformer block.
pub struct TransformerBlock {
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Layer attention pattern.
    pub layer_type: LayerType,
    /// Sliding-window size, if any.
    pub sliding_window: Option<i32>,

    #[quantizable]
    #[param]
    /// Self-attention layer.
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    /// Feed-forward layer.
    pub mlp: Mlp,
    #[quantizable]
    #[param]
    /// Optional MoE router.
    pub router: Option<MoeRouter>,
    #[param]
    /// Optional packed routed expert bank.
    pub experts: Option<GemmaExperts>,
    #[quantizable]
    #[param]
    /// Optional gate for per-layer input embeddings.
    pub per_layer_input_gate: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    /// Optional projection for per-layer input embeddings.
    pub per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    /// Optional normalization after per-layer input projection.
    pub post_per_layer_input_norm: Option<nn::RmsNorm>,
    #[param]
    /// Pre-attention RMSNorm.
    pub input_layernorm: nn::RmsNorm,
    #[param]
    /// Post-attention RMSNorm.
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    /// Pre-MLP RMSNorm.
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    /// Post-MLP RMSNorm.
    pub post_feedforward_layernorm: nn::RmsNorm,
    #[param]
    /// Dense-branch norm used before combining dense and routed outputs.
    pub post_feedforward_layernorm_1: Option<nn::RmsNorm>,
    #[param]
    /// Routed-branch input norm.
    pub pre_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    /// Routed-branch output norm.
    pub post_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    /// Learned scalar applied to the block output.
    pub layer_scalar: Param<Array>,
}

impl TransformerBlock {
    /// Creates an unloaded transformer block.
    pub fn new(
        args: &ModelArgs,
        layer_type: LayerType,
        layer_idx: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let layer_args = args.for_layer(layer_type);
        let self_attn = Attention::new(&layer_args, layer_type, layer_idx, stream)?;
        let prefix = format!("model.language_model.layers.{layer_idx}");
        let mlp = Mlp::new_selective(
            args.hidden_size,
            args.feed_forward_length_for_layer(layer_idx),
            [
                args.quantization_for(&format!("{prefix}.mlp.gate_proj.weight")),
                args.quantization_for(&format!("{prefix}.mlp.down_proj.weight")),
                args.quantization_for(&format!("{prefix}.mlp.up_proj.weight")),
            ],
            stream,
        )?;
        let input_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let post_attention_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let pre_feedforward_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let post_feedforward_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let moe = args
            .enable_moe_block
            .then(|| Moe::new(args, layer_idx, stream))
            .transpose()?;
        let (router, experts) = match moe {
            Some(moe) => (Some(moe.router), Some(moe.experts)),
            None => (None, None),
        };
        let post_feedforward_layernorm_1 = args
            .enable_moe_block
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let pre_feedforward_layernorm_2 = args
            .enable_moe_block
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let post_feedforward_layernorm_2 = args
            .enable_moe_block
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let per_layer_input_gate = if args.hidden_size_per_layer_input > 0 {
            Some(maybe_quantized_linear_with_config(
                args.hidden_size,
                args.hidden_size_per_layer_input,
                args.quantization_for(&format!("{prefix}.per_layer_input_gate.weight")),
                stream,
            )?)
        } else {
            None
        };
        let per_layer_projection = if args.hidden_size_per_layer_input > 0 {
            Some(maybe_quantized_linear_with_config(
                args.hidden_size_per_layer_input,
                args.hidden_size,
                args.quantization_for(&format!("{prefix}.per_layer_projection.weight")),
                stream,
            )?)
        } else {
            None
        };
        let post_per_layer_input_norm = if args.hidden_size_per_layer_input > 0 {
            Some(nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?)
        } else {
            None
        };
        Ok(Self {
            num_attention_heads: layer_args.num_attention_heads,
            hidden_size: layer_args.hidden_size,
            layer_type,
            sliding_window: args.sliding_window,
            layer_scalar: Param::new(Array::from_slice(&[1.0f32], &[1])),
            self_attn,
            mlp,
            router,
            experts,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_feedforward_layernorm_1,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_2,
        })
    }
}

impl TransformerBlock {
    fn apply_layer_scalar(&self, x: Array, stream: &Stream) -> Result<Array, Exception> {
        x.multiply(&*self.layer_scalar, stream)
    }

    /// Forward pass that reports block activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput {
            x,
            mask,
            cache,
            position_offset,
            per_layer_input,
            shared_kv,
            disable_generated_mask,
            generated_sliding_window: _,
        } = input;
        let generated_mask = if disable_generated_mask {
            None
        } else if self.layer_type == LayerType::SlidingAttention {
            let seq_len = x.shape()[1];
            if needs_generated_sliding_mask(seq_len, position_offset, self.sliding_window) {
                Some(create_causal_mask(
                    seq_len,
                    Some(position_offset),
                    self.sliding_window,
                    None,
                    stream,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let generated_sliding_window = generated_mask.as_ref().and(self.sliding_window);
        if let Some(mask) = generated_mask.as_ref().or(mask) {
            observer.observe(&format!("{prefix}.attention_mask"), mask)?;
        }

        observer.observe(&format!("{prefix}.input"), x)?;
        observer.observe(&format!("{prefix}.residual_before_attention"), x)?;
        let normed = self.input_layernorm.forward(x, stream)?;
        observer.observe(&format!("{prefix}.input_layernorm"), &normed)?;

        let self_attn_input = AttentionInput {
            x: &normed,
            mask: generated_mask.as_ref().or(mask),
            cache,
            position_offset,
            per_layer_input: None,
            shared_kv,
            disable_generated_mask,
            generated_sliding_window,
        };
        let r = self.self_attn.forward_with_observer(
            self_attn_input,
            stream,
            &format!("{prefix}.self_attn"),
            observer,
        )?;
        profile_array(PerfComponent::Attention, &r)?;
        observer.observe(&format!("{prefix}.self_attn_output"), &r)?;
        let r = self.post_attention_layernorm.forward(&r, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_attention"), &r)?;
        let h = x.add(r, stream)?;
        observer.observe(&format!("{prefix}.post_attention_residual"), &h)?;
        observer.observe(&format!("{prefix}.residual_after_attention"), &h)?;

        observer.observe(&format!("{prefix}.residual_before_mlp"), &h)?;
        let pre_ff = self.pre_feedforward_layernorm.forward(&h, stream)?;
        observer.observe(&format!("{prefix}.pre_feedforward_layernorm"), &pre_ff)?;
        let dense =
            self.mlp
                .forward_with_observer(&pre_ff, stream, &format!("{prefix}.mlp"), observer)?;
        let r = if let (Some(router), Some(experts)) = (self.router.as_mut(), self.experts.as_mut())
        {
            let dense = self
                .post_feedforward_layernorm_1
                .as_mut()
                .expect("MoE dense output norm")
                .forward(&dense, stream)?;
            observer.observe(&format!("{prefix}.post_feedforward_layernorm_1"), &dense)?;
            let shape = h.shape().to_vec();
            let routed_input = self
                .pre_feedforward_layernorm_2
                .as_mut()
                .expect("MoE routed input norm")
                .forward(&h.reshape(&[-1, self.hidden_size], stream)?, stream)?;
            observer.observe(
                &format!("{prefix}.pre_feedforward_layernorm_2"),
                &routed_input,
            )?;
            let (indices, weights) =
                router.forward(&h.reshape(&[-1, self.hidden_size], stream)?, stream)?;
            observer.observe(&format!("{prefix}.router.top_k_experts"), &indices)?;
            observer.observe(&format!("{prefix}.router.top_k_weights"), &weights)?;
            let routed = experts
                .forward(&routed_input, &indices, &weights, stream)?
                .reshape(&shape, stream)?;
            observer.observe(&format!("{prefix}.experts.output"), &routed)?;
            let routed = self
                .post_feedforward_layernorm_2
                .as_mut()
                .expect("MoE routed output norm")
                .forward(&routed, stream)?;
            observer.observe(&format!("{prefix}.post_feedforward_layernorm_2"), &routed)?;
            dense.add(routed, stream)?
        } else {
            dense
        };
        profile_array(PerfComponent::Mlp, &r)?;
        observer.observe(&format!("{prefix}.mlp_output"), &r)?;
        let r = self.post_feedforward_layernorm.forward(&r, stream)?;
        observer.observe(&format!("{prefix}.post_feedforward_layernorm"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_mlp"), &r)?;
        let mut h = h.add(r, stream)?;
        observer.observe(&format!("{prefix}.post_mlp_residual"), &h)?;
        observer.observe(&format!("{prefix}.residual_after_mlp"), &h)?;

        if let (Some(per_layer_input), Some(gate), Some(projection), Some(norm)) = (
            per_layer_input,
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
        ) {
            observer.observe(&format!("{prefix}.per_layer_input"), per_layer_input)?;
            let residual = h.clone();
            let gate_projection = gate.forward(&h, stream)?;
            observer.observe(&format!("{prefix}.per_layer_input_gate"), &gate_projection)?;
            let r =
                nn::gelu_approximate(gate_projection, stream)?.multiply(per_layer_input, stream)?;
            observer.observe(&format!("{prefix}.per_layer_projection_input"), &r)?;
            let r = projection.forward(&r, stream)?;
            observer.observe(&format!("{prefix}.per_layer_projection"), &r)?;
            let r = norm.forward(&r, stream)?;
            observer.observe(&format!("{prefix}.post_per_layer_input_norm"), &r)?;
            profile_array(PerfComponent::PerLayerResidual, &r)?;
            observer.observe(&format!("{prefix}.residual_delta_per_layer_input"), &r)?;
            h = residual.add(r, stream)?;
            observer.observe(&format!("{prefix}.residual_after_per_layer_input"), &h)?;
        }

        let output = self.apply_layer_scalar(h, stream)?;
        let output = observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output);
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(output)
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            cache,
            position_offset,
            per_layer_input,
            shared_kv,
            disable_generated_mask,
            generated_sliding_window: _,
        } = input;
        let generated_mask = if disable_generated_mask {
            None
        } else if self.layer_type == LayerType::SlidingAttention {
            let seq_len = x.shape()[1];
            if needs_generated_sliding_mask(seq_len, position_offset, self.sliding_window) {
                Some(create_causal_mask(
                    seq_len,
                    Some(position_offset),
                    self.sliding_window,
                    None,
                    stream,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let generated_sliding_window = generated_mask.as_ref().and(self.sliding_window);
        let normed = self.input_layernorm.forward(x, stream)?;
        let self_attn_input = AttentionInput {
            x: &normed,
            mask: generated_mask.as_ref().or(mask),
            cache,
            position_offset,
            per_layer_input: None,
            shared_kv,
            disable_generated_mask,
            generated_sliding_window,
        };
        let r = self.self_attn.forward(self_attn_input, stream)?;
        profile_array(PerfComponent::Attention, &r)?;
        let r = self.post_attention_layernorm.forward(&r, stream)?;
        let h = x.add(r, stream)?;
        let pre_ff = self.pre_feedforward_layernorm.forward(&h, stream)?;
        let dense = self.mlp.forward(&pre_ff, stream)?;
        let r = if let (Some(router), Some(experts)) = (self.router.as_mut(), self.experts.as_mut())
        {
            let dense = self
                .post_feedforward_layernorm_1
                .as_mut()
                .expect("MoE dense output norm")
                .forward(&dense, stream)?;
            let shape = h.shape().to_vec();
            let routed_input = self
                .pre_feedforward_layernorm_2
                .as_mut()
                .expect("MoE routed input norm")
                .forward(&h.reshape(&[-1, self.hidden_size], stream)?, stream)?;
            let (indices, weights) =
                router.forward(&h.reshape(&[-1, self.hidden_size], stream)?, stream)?;
            let routed = experts
                .forward(&routed_input, &indices, &weights, stream)?
                .reshape(&shape, stream)?;
            let routed = self
                .post_feedforward_layernorm_2
                .as_mut()
                .expect("MoE routed output norm")
                .forward(&routed, stream)?;
            dense.add(routed, stream)?
        } else {
            dense
        };
        profile_array(PerfComponent::Mlp, &r)?;
        let r = self.post_feedforward_layernorm.forward(&r, stream)?;
        let mut h = h.add(r, stream)?;
        if let (Some(per_layer_input), Some(gate), Some(projection), Some(norm)) = (
            per_layer_input,
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
        ) {
            let residual = h.clone();
            let r = nn::gelu_approximate(gate.forward(&h, stream)?, stream)?
                .multiply(per_layer_input, stream)?;
            let r = projection.forward(&r, stream)?;
            let r = norm.forward(&r, stream)?;
            profile_array(PerfComponent::PerLayerResidual, &r)?;
            h = residual.add(r, stream)?;
        }
        self.apply_layer_scalar(h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        if let Some(router) = &mut self.router {
            router.training_mode(mode);
        }
        if let Some(experts) = &mut self.experts {
            experts.training_mode(mode);
        }
        if let Some(layer) = &mut self.per_layer_input_gate {
            layer.training_mode(mode);
        }
        if let Some(layer) = &mut self.per_layer_projection {
            layer.training_mode(mode);
        }
        if let Some(norm) = &mut self.post_per_layer_input_norm {
            norm.training_mode(mode);
        }
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
        if let Some(norm) = &mut self.post_feedforward_layernorm_1 {
            norm.training_mode(mode);
        }
        if let Some(norm) = &mut self.pre_feedforward_layernorm_2 {
            norm.training_mode(mode);
        }
        if let Some(norm) = &mut self.post_feedforward_layernorm_2 {
            norm.training_mode(mode);
        }
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 text transformer body without the language-model head.
pub struct Gemma4TextModel {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Per-layer input hidden size.
    pub hidden_size_per_layer_input: i32,
    #[param]
    /// Token embedding table.
    pub embed_tokens: Gemma4Embedding,
    #[param]
    /// Optional per-layer token embedding table.
    pub embed_tokens_per_layer: Option<Gemma4Embedding>,
    #[quantizable]
    #[param]
    /// Optional projection used to form per-layer inputs.
    pub per_layer_model_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    /// Optional normalization for per-layer projection outputs.
    pub per_layer_projection_norm: Option<nn::RmsNorm>,
    #[quantizable]
    #[param]
    /// Transformer blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final RMSNorm.
    pub norm: nn::RmsNorm,
}

impl Gemma4TextModel {
    /// Creates an unloaded Gemma 4 text transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let embed_tokens = Gemma4Embedding::unloaded(
            args.vocab_size,
            args.hidden_size,
            args.quantization_for("model.language_model.embed_tokens.weight"),
            stream,
        )?;
        let embed_tokens_per_layer = if args.hidden_size_per_layer_input > 0 {
            Some(Gemma4Embedding::unloaded(
                args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                args.quantization_for("model.language_model.embed_tokens_per_layer.weight"),
                stream,
            )?)
        } else {
            None
        };
        let per_layer_model_projection = if args.hidden_size_per_layer_input > 0 {
            Some(maybe_quantized_linear_with_config(
                args.hidden_size,
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                args.quantization_for("model.language_model.per_layer_model_projection.weight"),
                stream,
            )?)
        } else {
            None
        };
        let per_layer_projection_norm = if args.hidden_size_per_layer_input > 0 {
            Some(nn::RmsNorm::unloaded(
                args.hidden_size_per_layer_input,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?)
        } else {
            None
        };
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
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            hidden_size: args.hidden_size,
            hidden_size_per_layer_input: args.hidden_size_per_layer_input,
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
        })
    }

    fn per_layer_inputs(
        &mut self,
        input_ids: &Array,
        inputs_embeds: &Array,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        let Some(embed_tokens_per_layer) = self.embed_tokens_per_layer.as_mut() else {
            return Ok(None);
        };
        let Some(per_layer_model_projection) = self.per_layer_model_projection.as_mut() else {
            return Ok(None);
        };
        let Some(per_layer_projection_norm) = self.per_layer_projection_norm.as_mut() else {
            return Ok(None);
        };
        let ple_dim = self.hidden_size_per_layer_input;
        let token_identity = embed_tokens_per_layer
            .forward(input_ids, stream)?
            .multiply(Array::from_f32((ple_dim as f32).sqrt()), stream)?
            .reshape(
                &[
                    input_ids.shape()[0],
                    input_ids.shape()[1],
                    self.num_hidden_layers,
                    ple_dim,
                ],
                stream,
            )?;
        let projected = per_layer_model_projection
            .forward(inputs_embeds, stream)?
            .multiply(
                Array::from_f32((self.hidden_size as f32).sqrt().recip()),
                stream,
            )?
            .reshape(
                &[
                    inputs_embeds.shape()[0],
                    inputs_embeds.shape()[1],
                    self.num_hidden_layers,
                    ple_dim,
                ],
                stream,
            )?;
        let projected = per_layer_projection_norm.forward(&projected, stream)?;
        Ok(Some(
            projected
                .add(token_identity, stream)?
                .multiply(Array::from_f32(2.0_f32.powf(-0.5)), stream)?,
        ))
    }

    fn per_layer_inputs_with_observer(
        &mut self,
        input_ids: &Array,
        inputs_embeds: &Array,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Option<Array>, Exception> {
        let Some(embed_tokens_per_layer) = self.embed_tokens_per_layer.as_mut() else {
            return Ok(None);
        };
        let Some(per_layer_model_projection) = self.per_layer_model_projection.as_mut() else {
            return Ok(None);
        };
        let Some(per_layer_projection_norm) = self.per_layer_projection_norm.as_mut() else {
            return Ok(None);
        };
        let ple_dim = self.hidden_size_per_layer_input;
        let token_identity = embed_tokens_per_layer
            .forward(input_ids, stream)?
            .multiply(Array::from_f32((ple_dim as f32).sqrt()), stream)?
            .reshape(
                &[
                    input_ids.shape()[0],
                    input_ids.shape()[1],
                    self.num_hidden_layers,
                    ple_dim,
                ],
                stream,
            )?;
        observer.observe("model.per_layer_token_identity", &token_identity)?;
        let projected = per_layer_model_projection
            .forward(inputs_embeds, stream)?
            .multiply(
                Array::from_f32((self.hidden_size as f32).sqrt().recip()),
                stream,
            )?
            .reshape(
                &[
                    inputs_embeds.shape()[0],
                    inputs_embeds.shape()[1],
                    self.num_hidden_layers,
                    ple_dim,
                ],
                stream,
            )?;
        observer.observe("model.per_layer_model_projection", &projected)?;
        let projected = per_layer_projection_norm.forward(&projected, stream)?;
        observer.observe("model.per_layer_projection_norm", &projected)?;
        let output = projected
            .add(token_identity, stream)?
            .multiply(Array::from_f32(2.0_f32.powf(-0.5)), stream)?;
        observer.observe("model.per_layer_inputs", &output)?;
        Ok(Some(output))
    }
}

/// Input for a Gemma 4 text forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional prepared embeddings replacing the token embedding lookup.
    pub inputs_embeds: Option<&'a Array>,
    /// Optional IDs used only for per-layer token-identity embeddings.
    pub per_layer_input_ids: Option<&'a Array>,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional sliding-layer mask when it differs from the full-attention mask.
    pub sliding_mask: Option<&'a Array>,
    /// Mutable per-layer key/value cache.
    pub cache: &'a mut Vec<Option<C>>,
}

impl Gemma4TextModel {
    /// Runs the text body and returns final hidden states plus state used by assistant drafting.
    pub fn forward_with_state<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Gemma4TextOutput, Exception>
    where
        C: KeyValueCache + Default,
    {
        let ModelInput {
            inputs,
            inputs_embeds,
            per_layer_input_ids,
            mask,
            sliding_mask,
            cache,
        } = input;
        let mut h = match inputs_embeds {
            Some(embeddings) => embeddings.clone(),
            None => self
                .embed_tokens
                .forward(inputs, stream)?
                .multiply(Array::from_f32((self.hidden_size as f32).sqrt()), stream)?,
        };
        profile_array(PerfComponent::Embed, &h)?;
        let per_layer_inputs =
            self.per_layer_inputs(per_layer_input_ids.unwrap_or(inputs), &h, stream)?;
        if let Some(per_layer_inputs) = &per_layer_inputs {
            profile_array(PerfComponent::PerLayerInputs, per_layer_inputs)?;
        }
        let position_offset = cache
            .iter()
            .flatten()
            .map(KeyValueCache::offset)
            .max()
            .unwrap_or(0);
        let mut shared_kv = HashMap::new();
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None if h.shape()[1] > 1 => Some(create_causal_mask(
                h.shape()[1],
                Some(position_offset),
                None,
                None,
                stream,
            )?),
            None => None,
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        for (index, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_ple = per_layer_inputs
                .as_ref()
                .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
                .transpose()?;
            let layer_mask = if layer.layer_type == LayerType::SlidingAttention {
                sliding_mask.or(mask.as_ref())
            } else {
                mask.as_ref()
            };
            let layer_input = AttentionInput {
                x: &h,
                mask: layer_mask,
                cache: c.as_mut(),
                position_offset,
                per_layer_input: layer_ple.as_ref(),
                shared_kv: Some(&mut shared_kv),
                disable_generated_mask: sliding_mask.is_some(),
                generated_sliding_window: None,
            };
            h = layer.forward(layer_input, stream)?;
        }
        let pre_norm_hidden = h.clone();
        let hidden = self.norm.forward(&h, stream)?;
        profile_array(PerfComponent::FinalNorm, &hidden)?;
        Ok(Gemma4TextOutput {
            hidden,
            pre_norm_hidden,
            shared_kv_states: shared_kv,
        })
    }

    /// Forward pass that reports transformer-body activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let ModelInput {
            inputs,
            inputs_embeds,
            per_layer_input_ids,
            mask,
            sliding_mask,
            cache,
        } = input;
        let mut h = match inputs_embeds {
            Some(embeddings) => embeddings.clone(),
            None => self
                .embed_tokens
                .forward(inputs, stream)?
                .multiply(Array::from_f32((self.hidden_size as f32).sqrt()), stream)?,
        };
        profile_array(PerfComponent::Embed, &h)?;
        observer.observe("model.embed_tokens", &h)?;

        let per_layer_inputs = self.per_layer_inputs_with_observer(
            per_layer_input_ids.unwrap_or(inputs),
            &h,
            stream,
            observer,
        )?;
        if let Some(per_layer_inputs) = &per_layer_inputs {
            profile_array(PerfComponent::PerLayerInputs, per_layer_inputs)?;
        }
        let position_offset = cache
            .iter()
            .flatten()
            .map(KeyValueCache::offset)
            .max()
            .unwrap_or(0);
        let mut shared_kv = HashMap::new();
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None if h.shape()[1] > 1 => Some(create_causal_mask(
                h.shape()[1],
                Some(position_offset),
                None,
                None,
                stream,
            )?),
            None => None,
        };
        if let Some(mask) = mask.as_ref() {
            observer.observe("model.attention_mask", mask)?;
        }

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        for (index, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_ple = per_layer_inputs
                .as_ref()
                .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
                .transpose()?;
            let layer_mask = if layer.layer_type == LayerType::SlidingAttention {
                sliding_mask.or(mask.as_ref())
            } else {
                mask.as_ref()
            };
            let layer_input = AttentionInput {
                x: &h,
                mask: layer_mask,
                cache: c.as_mut(),
                position_offset,
                per_layer_input: layer_ple.as_ref(),
                shared_kv: Some(&mut shared_kv),
                disable_generated_mask: sliding_mask.is_some(),
                generated_sliding_window: None,
            };
            h = layer.forward_with_observer(
                layer_input,
                stream,
                &format!("model.layers.{index}"),
                observer,
            )?;
        }
        observer.observe("model.pre_norm_hidden", &h)?;
        let output = self.norm.forward(&h, stream)?;
        profile_array(PerfComponent::FinalNorm, &output)?;
        observer.observe("model.norm", &output)?;
        Ok(output)
    }
}

impl<C> Module<ModelInput<'_, C>> for Gemma4TextModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        Ok(self.forward_with_state(input, stream)?.hidden)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 conditional-generation wrapper.
pub struct Gemma4ForConditionalGeneration {
    #[quantizable]
    #[param]
    /// Text transformer body.
    pub language_model: Gemma4TextModel,
    #[param]
    /// Optional image encoder.
    pub(crate) vision_tower: Option<Gemma4VisionTower>,
    #[quantizable]
    #[param]
    /// Optional projection from vision features into text hidden space.
    pub(crate) embed_vision: Option<Gemma4ModalityEmbedder>,
    #[param]
    /// Optional audio encoder.
    pub(crate) audio_tower: Option<Gemma4AudioTower>,
    #[quantizable]
    #[param]
    /// Optional projection from audio features into text hidden space.
    pub(crate) embed_audio: Option<Gemma4ModalityEmbedder>,
}

impl Gemma4ForConditionalGeneration {
    /// Creates an unloaded conditional-generation wrapper.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            language_model: Gemma4TextModel::new(args, stream)?,
            vision_tower: None,
            embed_vision: None,
            audio_tower: None,
            embed_audio: None,
        })
    }

    pub(crate) fn new_with_modalities(
        args: &ModelArgs,
        vision_config: Option<Gemma4VisionConfig>,
        audio_config: Option<Gemma4AudioConfig>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let vision_tower = vision_config
            .clone()
            .map(|config| Gemma4VisionTower::new(config, stream))
            .transpose()?;
        let embed_vision = vision_config
            .as_ref()
            .map(|config| {
                Gemma4ModalityEmbedder::new(
                    config.hidden_size,
                    args.hidden_size,
                    config.rms_norm_eps,
                    false,
                    args.weight_quantization(),
                    stream,
                )
            })
            .transpose()?;
        let audio_tower = audio_config
            .as_ref()
            .map(|config| Gemma4AudioTower::new(config, stream))
            .transpose()?;
        let embed_audio = audio_config
            .as_ref()
            .map(|config| {
                Gemma4ModalityEmbedder::new(
                    config.output_proj_dims,
                    args.hidden_size,
                    config.rms_norm_eps,
                    false,
                    args.weight_quantization(),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            language_model: Gemma4TextModel::new(args, stream)?,
            vision_tower,
            embed_vision,
            audio_tower,
            embed_audio,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    /// Image media token ID for multimodal checkpoints.
    pub image_token_id: Option<i32>,
    /// Video media token ID for multimodal checkpoints.
    pub video_token_id: Option<i32>,
    /// Audio media token ID for multimodal checkpoints.
    pub audio_token_id: Option<i32>,
    #[quantizable]
    #[param]
    /// Conditional-generation model body.
    pub model: Gemma4ForConditionalGeneration,
    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
    /// Storage accounting for checkpoint-native and generic fallback quantization.
    pub native_quantization_stats: NativeQuantizationStats,
}

impl Model {
    fn project_logits(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(hidden_states, stream)?,
            None => self
                .model
                .language_model
                .embed_tokens
                .as_linear(hidden_states, stream)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(&(logits.divide(Array::from_f32(softcap), stream)?), stream)?
                .multiply(Array::from_f32(softcap), stream)?;
        }
        profile_array(PerfComponent::LmHead, &logits)?;
        Ok(logits)
    }

    pub(crate) fn prefill_typed_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let logits = match self.prepare_typed_prefill(input, stream)? {
            input::PreparedPrefill::Text(tokens) => {
                cache.token_ids = token_ids_from_array(&tokens, stream)?;
                cache.prefix_embeddings = None;
                cache.prefix_len = 0;
                cache.reset_kv(&self.args);
                self.forward_with_observer(
                    ModelInput {
                        inputs: &tokens,
                        inputs_embeds: None,
                        per_layer_input_ids: None,
                        mask: None,
                        sliding_mask: None,
                        cache: &mut cache.kv,
                    },
                    stream,
                    observer,
                )?
            }
            input::PreparedPrefill::Embeddings { tokens, embeddings } => {
                cache.token_ids = token_ids_from_array(&tokens, stream)?;
                cache.prefix_len = cache.token_ids.len();
                cache.prefix_embeddings = Some(embeddings.clone());
                cache.reset_kv(&self.args);
                let per_layer_ids = self.per_layer_ids_for_media(&tokens, stream)?;
                let masks = multimodal_attention_masks(
                    &cache.token_ids,
                    self.image_token_id.map(|id| id as u32),
                    self.video_token_id.map(|id| id as u32),
                    self.args.sliding_window,
                );
                self.forward_with_observer(
                    ModelInput {
                        inputs: &tokens,
                        inputs_embeds: Some(&embeddings),
                        per_layer_input_ids: Some(&per_layer_ids),
                        mask: Some(&masks.full),
                        sliding_mask: Some(&masks.sliding),
                        cache: &mut cache.kv,
                    },
                    stream,
                    observer,
                )?
            }
        };
        logits.try_index_device((.., -1, ..), stream)
    }

    pub(crate) fn forward_logits<C>(
        &mut self,
        input: ModelInput<'_, C>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let hidden_states = self.model.language_model.forward(input, stream)?;
        let hidden_states = if last_token_only {
            hidden_states.try_index_device((.., -1, ..), stream)?
        } else {
            hidden_states
        };
        self.project_logits(&hidden_states, stream)
    }

    /// Creates an unloaded Gemma 4 causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let model = Gemma4ForConditionalGeneration::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(maybe_quantized_linear_with_config(
                args.hidden_size,
                args.vocab_size,
                args.quantization_for("lm_head.weight"),
                stream,
            )?)
        } else {
            None
        };
        Ok(Self {
            args,
            image_token_id: None,
            video_token_id: None,
            audio_token_id: None,
            model,
            lm_head,
            native_quantization_stats: NativeQuantizationStats::default(),
        })
    }

    pub(crate) fn new_with_modalities(
        args: ModelArgs,
        image_token_id: Option<i32>,
        vision_config: Option<Gemma4VisionConfig>,
        video_token_id: Option<i32>,
        audio_token_id: Option<i32>,
        audio_config: Option<Gemma4AudioConfig>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let model = Gemma4ForConditionalGeneration::new_with_modalities(
            &args,
            vision_config,
            audio_config,
            stream,
        )?;
        let lm_head = if !args.tie_word_embeddings {
            Some(maybe_quantized_linear_with_config(
                args.hidden_size,
                args.vocab_size,
                args.quantization_for("lm_head.weight"),
                stream,
            )?)
        } else {
            None
        };
        Ok(Self {
            args,
            image_token_id,
            video_token_id,
            audio_token_id,
            model,
            lm_head,
            native_quantization_stats: NativeQuantizationStats::default(),
        })
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn prepare_typed_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<input::PreparedPrefill, Exception> {
        let modality_tokens = self
            .image_token_id
            .map(|token_id| input::ModalityToken {
                modality: input::Modality::Image,
                token_id: token_id as u32,
            })
            .into_iter()
            .chain(self.video_token_id.map(|token_id| input::ModalityToken {
                modality: input::Modality::Video,
                token_id: token_id as u32,
            }))
            .chain(self.audio_token_id.map(|token_id| input::ModalityToken {
                modality: input::Modality::Audio,
                token_id: token_id as u32,
            }))
            .collect::<Vec<_>>();
        let embed_tokens = &mut self.model.language_model.embed_tokens;
        let vision_tower = &mut self.model.vision_tower;
        let embed_vision = &mut self.model.embed_vision;
        let audio_tower = &mut self.model.audio_tower;
        let embed_audio = &mut self.model.embed_audio;
        let text_scale = (self.args.hidden_size as f32).sqrt();
        let prepared = input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.hidden_size,
            "gemma4",
            stream,
            |tokens, stream| {
                embed_tokens
                    .forward(tokens, stream)?
                    .multiply(Array::from_f32(text_scale), stream)
            },
            |part, stream| match (part.modality, part.payload) {
                (_, input::InputPayload::Embeddings(embeddings)) => Ok(vec![embeddings.clone()]),
                (
                    modality @ (input::Modality::Image | input::Modality::Video),
                    input::InputPayload::Tensor(tensor),
                ) => {
                    let position_ids = part.metadata.patch_position_ids.ok_or_else(|| {
                        Exception::custom(format!(
                            "gemma4 {} tensor input requires patch_position_ids metadata",
                            modality.as_str()
                        ))
                    })?;
                    let features = vision_tower
                        .as_mut()
                        .ok_or_else(|| {
                            Exception::custom(format!(
                                "gemma4 {} tensor input requires vision_config and vision weights",
                                modality.as_str()
                            ))
                        })?
                        .forward(tensor, position_ids, stream)?;
                    let embeddings = embed_vision
                        .as_mut()
                        .ok_or_else(|| {
                            Exception::custom(format!(
                                "gemma4 {} input requires embed_vision weights",
                                modality.as_str()
                            ))
                        })?
                        .forward(&features, stream)?;
                    if modality == input::Modality::Video {
                        let mut frames = Vec::with_capacity(embeddings.dim(0) as usize);
                        for frame in 0..embeddings.dim(0) {
                            frames.push(
                                embeddings.try_index_device((frame..frame + 1, .., ..), stream)?,
                            );
                        }
                        Ok(frames)
                    } else {
                        Ok(vec![embeddings])
                    }
                }
                (input::Modality::Audio, input::InputPayload::Tensor(tensor)) => {
                    let mask = part.metadata.audio_mask.ok_or_else(|| {
                        Exception::custom("gemma4 audio tensor input requires audio_mask metadata")
                    })?;
                    let features = audio_tower
                        .as_mut()
                        .ok_or_else(|| {
                            Exception::custom(
                                "gemma4 audio tensor input requires audio_config and audio weights",
                            )
                        })?
                        .forward(tensor, mask, stream)?;
                    Ok(vec![embed_audio
                        .as_mut()
                        .ok_or_else(|| {
                            Exception::custom("gemma4 audio input requires embed_audio weights")
                        })?
                        .forward(&features, stream)?])
                }
                (modality, input::InputPayload::Tensor(_)) => Err(Exception::custom(format!(
                    "gemma4 does not support {} tensor inputs",
                    modality.as_str()
                ))),
                (modality, input::InputPayload::TokenIds(_)) => Err(Exception::custom(format!(
                    "gemma4 {} input does not accept token-id payloads",
                    modality.as_str()
                ))),
            },
        )?;
        Ok(prepared)
    }

    fn per_layer_ids_for_media(&self, tokens: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut output = tokens.clone();
        for token_id in [
            self.image_token_id,
            self.video_token_id,
            self.audio_token_id,
        ]
        .into_iter()
        .flatten()
        {
            let mask = output.eq(Array::from_int(token_id), stream)?;
            output = r#where(
                &mask,
                Array::from_int(self.args.pad_token_id),
                &output,
                stream,
            )?;
        }
        Ok(output)
    }

    /// Forward pass that reports activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let out = self
            .model
            .language_model
            .forward_with_observer(input, stream, observer)?;
        observer.observe("model.output", &out)?;
        let logits = self.project_logits(&out, stream)?;
        observer.observe("lm_head.logits", &logits)?;
        Ok(logits)
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        self.forward_logits(input, false, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma4TextModel as Module<ModelInput<'_, C>>>::training_mode(
            &mut self.model.language_model,
            mode,
        );
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

/// Loads `tokenizer.json` from a Gemma 4 model directory.
pub fn load_gemma4_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub(crate) struct LoadedGemma4Gguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) struct PreparedGemma4Gguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads the text model from a Gemma 4 GGUF checkpoint.
///
/// Dense tensors and every GGUF quantization supported by the shared backend are
/// accepted for both dense and MoE text checkpoints. Vision, audio, assistant-drafter,
/// and separate multimodal projector GGUF files use their dedicated loaders.
pub fn load_gemma4_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    Ok(load_gemma4_gguf_with_metadata(gguf_file, stream, weights_stream)?.model)
}

pub(crate) fn load_gemma4_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedGemma4Gguf, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    load_gemma4_gguf_checkpoint(&checkpoint, metadata, None, stream, weights_stream)
}

pub(crate) fn load_gemma4_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedGemma4Gguf, Error> {
    let prepared =
        prepare_gemma4_gguf_checkpoint(checkpoint, &metadata, quantization, weights_stream)?;
    let mut model = Model::new(prepared.args, stream)?;
    if quantization.is_none() {
        for tensor in checkpoint
            .catalog()
            .tensors()
            .filter(|tensor| tensor.affine().is_some())
        {
            model
                .native_quantization_stats
                .record_fallback(tensor.descriptor().byte_len);
        }
    }
    let mut config = StrictLoadConfig::default()
        .allow_unused_prefix("rope_freqs.")
        .allow_missing_suffix(".bias");
    let first_shared_layer =
        (model.args.num_hidden_layers - model.args.num_kv_shared_layers).max(0);
    for layer in first_shared_layer..model.args.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        config = config
            .allow_unused_prefix(format!("{prefix}.k_proj."))
            .allow_unused_prefix(format!("{prefix}.v_proj."))
            .allow_unused_prefix(format!("{prefix}.k_norm."));
    }
    let mut report = StrictLoadReport::default();
    load_gemma4_gguf_weights(
        &mut model,
        checkpoint,
        quantization,
        stream,
        weights_stream,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;

    Ok(LoadedGemma4Gguf {
        model,
        eos_token_ids: prepared.eos_token_ids,
    })
}

pub(crate) fn prepare_gemma4_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    weights_stream: &Stream,
) -> Result<PreparedGemma4Gguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "gemma4" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports only gemma4"
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;

    let mut args = gemma4_args_from_gguf(checkpoint, metadata, weights_stream)?;
    let mut quantized_weight_configs =
        gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    if args.enable_moe_block {
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{layer}.experts.switch_glu");
            if let Some(config) =
                quantized_weight_configs.remove(&format!("{prefix}.gate_up_proj.weight"))
            {
                quantized_weight_configs.insert(format!("{prefix}.gate_proj.weight"), config);
                quantized_weight_configs.insert(format!("{prefix}.up_proj.weight"), config);
            }
        }
    }
    let quantized_weights = quantized_weight_configs
        .keys()
        .cloned()
        .collect::<HashSet<_>>();
    let has_quantized_tensors = !quantized_weights.is_empty();
    if let Some(quantization) = quantization {
        args.quantized = true;
        args.weight_quantization = Some(quantization);
        args.quantization_group_size = quantization.group_size();
        args.quantization_bits = quantization.bits();
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
    } else {
        args.quantized_weights = Some(quantized_weights);
        args.quantized_weight_configs = Some(quantized_weight_configs);
        if has_quantized_tensors {
            args.quantized = true;
        }
    }

    let eos_token_ids = super::gguf_eos_token_ids(metadata)?;
    Ok(PreparedGemma4Gguf {
        args,
        eos_token_ids,
    })
}

fn load_gemma4_gguf_weights(
    model: &mut Model,
    checkpoint: &GgufCheckpoint,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    if model.args.enable_moe_block {
        for layer in 0..model.args.num_hidden_layers {
            let prefix = format!("blk.{layer}");
            let fused =
                checkpoint.contains_gguf_tensor(&format!("{prefix}.ffn_gate_up_exps.weight"));
            let gate = checkpoint.contains_gguf_tensor(&format!("{prefix}.ffn_gate_exps.weight"));
            let up = checkpoint.contains_gguf_tensor(&format!("{prefix}.ffn_up_exps.weight"));
            if fused && (gate || up) {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Gemma 4 GGUF layer {layer} mixes fused and separate gate/up expert tensors"
                )));
            }
            if !fused && (!gate || !up) {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Gemma 4 GGUF layer {layer} has incomplete gate/up expert tensors"
                )));
            }
        }
    }

    let mut materializer = checkpoint.materializer();
    for tensor in checkpoint.catalog().tensors() {
        let physical_name = &tensor.descriptor().name;
        if physical_name.contains(".ffn_gate_up_exps.") {
            continue;
        }
        let native_format = if quantization.is_none() {
            match tensor.descriptor().ggml_type {
                GgufType::Q4K => Some(NativeQuantizationFormat::GgufQ4K),
                GgufType::Q8_0 => Some(NativeQuantizationFormat::GgufQ8_0),
                _ => None,
            }
        } else {
            None
        };
        if let Some(native_format) = native_format {
            let raw = materializer.raw_tensor(physical_name)?;
            if raw.endian() == GgufEndian::Little {
                let shape = native_gguf_shape(raw.descriptor().mlx_shape(), physical_name)?;
                let native = match native_format {
                    NativeQuantizationFormat::GgufQ4K => {
                        NativeQuantizedTensor::from_q4k_bytes(raw.data(), &shape, stream)?
                    }
                    NativeQuantizationFormat::GgufQ8_0 => {
                        NativeQuantizedTensor::from_q8_0_bytes(raw.data(), &shape, stream)?
                    }
                    NativeQuantizationFormat::GgufQ5_1 => unreachable!(),
                    _ => unreachable!("IQ tensors are loaded through the general native path"),
                };
                let target = translate_gguf_weight_name(physical_name);
                if attach_native_quantized(model, &target, native, report)? {
                    model
                        .native_quantization_stats
                        .promote_native(native_format, raw.data().len() as u64);
                    continue;
                }
            }
        }
        if quantization.is_none()
            && physical_name.ends_with(".ffn_down_exps.weight")
            && tensor.descriptor().ggml_type == GgufType::Q5_1
        {
            let raw = materializer.raw_tensor(physical_name)?;
            if raw.endian() == GgufEndian::Little {
                let shape = native_gguf_shape(raw.descriptor().mlx_shape(), physical_name)?;
                let native = NativeQuantizedTensor::from_q5_1_bytes(raw.data(), &shape, stream)?;
                let layer = physical_name
                    .strip_prefix("blk.")
                    .and_then(|rest| rest.split_once('.'))
                    .and_then(|(layer, _)| layer.parse::<usize>().ok())
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "cannot identify Gemma 4 layer for {physical_name:?}"
                        ))
                    })?;
                let projection = &mut model.model.language_model.layers[layer]
                    .experts
                    .as_mut()
                    .expect("Gemma 4 MoE layer has experts")
                    .switch_glu
                    .down_proj;
                projection.native = Some(native);
                projection.weight.value = Array::from_slice(&[] as &[u32], &[0]);
                projection.scales.value = None;
                projection.biases.value = None;
                model
                    .native_quantization_stats
                    .promote_native(NativeQuantizationFormat::GgufQ5_1, raw.data().len() as u64);
                report.record_loaded(format!(
                    "model.language_model.layers.{layer}.experts.switch_glu.down_proj.weight"
                ));
                continue;
            }
        }
        for (name, value) in materializer.converted_tensor(physical_name)?.into_arrays() {
            load_named_array_strict(
                model,
                translate_gguf_weight_name(&name),
                value,
                quantization.map(|value| (value, stream)),
                config,
                report,
            )?;
        }
    }

    for layer in 0..model.args.num_hidden_layers {
        let source_prefix = format!("blk.{layer}");
        let fused_name = format!("{source_prefix}.ffn_gate_up_exps.weight");
        if !checkpoint.contains_gguf_tensor(&fused_name) {
            continue;
        }
        let target_prefix = format!("model.language_model.layers.{layer}.experts.switch_glu");
        let catalog_tensor = checkpoint
            .catalog()
            .tensors()
            .find(|tensor| tensor.descriptor().name == fused_name)
            .expect("checked fused GGUF tensor exists");
        if quantization.is_none() && catalog_tensor.descriptor().ggml_type == GgufType::Q4K {
            let raw = materializer.raw_tensor(&fused_name)?;
            if raw.endian() == GgufEndian::Little {
                let shape = native_gguf_shape(raw.descriptor().mlx_shape(), &fused_name)?;
                let native = NativeQuantizedTensor::from_q4k_bytes(raw.data(), &shape, stream)?;
                let intermediate = model.args.moe_intermediate_size.ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Gemma 4 MoE config is missing moe_intermediate_size".into(),
                    )
                })?;
                if native.rows() != 2 * intermediate {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Gemma 4 native fused gate/up layer {layer} has {} rows per expert; expected {}",
                        native.rows(),
                        2 * intermediate
                    )));
                }
                let experts = model.model.language_model.layers[layer as usize]
                    .experts
                    .as_mut()
                    .expect("Gemma 4 MoE layer has experts");
                experts.switch_glu.gate_proj.native = Some(native.row_view(0, intermediate)?);
                experts.switch_glu.up_proj.native =
                    Some(native.row_view(intermediate, intermediate)?);
                experts.native_gate_up = Some(native);
                model
                    .native_quantization_stats
                    .promote_native(NativeQuantizationFormat::GgufQ4K, raw.data().len() as u64);
                for projection in ["gate_proj", "up_proj"] {
                    report.record_loaded(format!("{target_prefix}.{projection}.weight"));
                }
                for projection in [
                    &mut experts.switch_glu.gate_proj,
                    &mut experts.switch_glu.up_proj,
                ] {
                    // Native execution owns the only persistent weight bytes.
                    // Clear unloaded affine placeholders before the model-wide
                    // stream copy, otherwise their declared shapes allocate a
                    // second checkpoint-sized device buffer.
                    projection.weight.value = Array::from_slice(&[] as &[u32], &[0]);
                    projection.scales.value = None;
                    projection.biases.value = None;
                }
                continue;
            }
        }
        let fused = materializer
            .converted_tensor(&fused_name)?
            .into_arrays()
            .into_iter()
            .collect::<HashMap<_, _>>();
        for suffix in ["weight", "scales", "biases"] {
            let source_name = format!("{source_prefix}.ffn_gate_up_exps.{suffix}");
            let Some(value) = fused.get(&source_name) else {
                if suffix == "weight" {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Gemma 4 GGUF is missing fused gate/up expert weights in layer {layer}"
                    )));
                }
                continue;
            };
            let shape = value.shape();
            let intermediate = model.args.moe_intermediate_size.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Gemma 4 MoE config is missing moe_intermediate_size".into(),
                )
            })?;
            if shape.len() != 3 || shape[1] != 2 * intermediate {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Gemma 4 GGUF fused expert {suffix} in layer {layer} has shape {shape:?}; expected output dimension {}",
                    2 * intermediate
                )));
            }
            let gate = value.try_index_device((.., ..intermediate, ..), weights_stream)?;
            let up = value.try_index_device((.., intermediate.., ..), weights_stream)?;
            for (projection, value) in [("gate_proj", gate), ("up_proj", up)] {
                load_named_array_strict(
                    model,
                    format!("{target_prefix}.{projection}.{suffix}"),
                    value,
                    quantization.map(|value| (value, stream)),
                    config,
                    report,
                )?;
            }
        }
    }
    Ok(())
}

fn native_gguf_shape(shape: Vec<u64>, name: &str) -> Result<Vec<i32>, Error> {
    shape
        .into_iter()
        .map(|dimension| {
            i32::try_from(dimension).map_err(|_| {
                Error::UnsupportedArchitecture(format!(
                    "native tensor {name:?} dimension {dimension} exceeds MLX limits"
                ))
            })
        })
        .collect()
}

fn attach_native_linear(
    linear: &mut MaybeQuantized<nn::Linear>,
    target: &str,
    native: NativeQuantizedTensor,
    report: &mut StrictLoadReport,
) -> Result<bool, Error> {
    let MaybeQuantized::Quantized(linear) = linear else {
        return Ok(false);
    };
    let expected_output = linear.inner.weight.dim(0);
    let expected_input = linear.scales.dim(1) * linear.group_size;
    if native.shape() != [expected_output, expected_input] {
        return Err(Error::UnsupportedArchitecture(format!(
            "native tensor {target:?} has shape {:?}; expected [{expected_output}, {expected_input}]",
            native.shape()
        )));
    }
    linear.native = Some(native);
    linear.inner.weight.value = Array::from_slice(&[] as &[u32], &[0]);
    linear.inner.bias.value = None;
    linear.scales.value = Array::from_slice(&[] as &[f32], &[0]);
    linear.biases.value = None;
    let prefix = target
        .strip_suffix(".weight")
        .expect("native linear target ends in .weight");
    report.record_loaded(format!("{prefix}.inner.weight"));
    report.record_loaded(format!("{prefix}.scales"));
    Ok(true)
}

fn attach_native_expert_projection(
    projection: &mut ExpertProjection,
    target: &str,
    native: NativeQuantizedTensor,
    report: &mut StrictLoadReport,
) -> Result<bool, Error> {
    if projection.quantization.is_none() {
        return Ok(false);
    }
    let expected = [
        projection.weight.dim(0),
        projection.weight.dim(1),
        native.columns(),
    ];
    if native.shape() != expected {
        return Err(Error::UnsupportedArchitecture(format!(
            "native expert tensor {target:?} has shape {:?}; expected {expected:?}",
            native.shape()
        )));
    }
    projection.native = Some(native);
    projection.weight.value = Array::from_slice(&[] as &[u32], &[0]);
    projection.scales.value = None;
    projection.biases.value = None;
    report.record_loaded(target.to_string());
    Ok(true)
}

fn attach_native_embedding(
    embedding: &mut Gemma4Embedding,
    target: &str,
    native: NativeQuantizedTensor,
    report: &mut StrictLoadReport,
) -> Result<bool, Error> {
    if !embedding.quantized {
        return Ok(false);
    }
    let expected = [embedding.weight.dim(0), embedding.hidden_size];
    if native.shape() != expected {
        return Err(Error::UnsupportedArchitecture(format!(
            "native embedding {target:?} has shape {:?}; expected {expected:?}",
            native.shape()
        )));
    }
    embedding.native = Some(native);
    embedding.weight.value = Array::from_slice(&[] as &[u32], &[0]);
    embedding.scales.value = None;
    embedding.biases.value = None;
    report.record_loaded(target.to_string());
    Ok(true)
}

/// Installs one native checkpoint tensor into a general Gemma module interface.
///
/// The dispatch is model-layer integration only: physical decoding and
/// device policy remain in `safemlx::native_quantization`.
fn attach_native_quantized(
    model: &mut Model,
    target: &str,
    native: NativeQuantizedTensor,
    report: &mut StrictLoadReport,
) -> Result<bool, Error> {
    match target {
        "model.language_model.embed_tokens.weight" => {
            return attach_native_embedding(
                &mut model.model.language_model.embed_tokens,
                target,
                native,
                report,
            );
        }
        "model.language_model.embed_tokens_per_layer.weight" => {
            let Some(embedding) = model.model.language_model.embed_tokens_per_layer.as_mut() else {
                return Ok(false);
            };
            return attach_native_embedding(embedding, target, native, report);
        }
        "model.language_model.per_layer_model_projection.weight" => {
            let Some(linear) = model
                .model
                .language_model
                .per_layer_model_projection
                .as_mut()
            else {
                return Ok(false);
            };
            return attach_native_linear(linear, target, native, report);
        }
        "lm_head.weight" => {
            let Some(linear) = model.lm_head.as_mut() else {
                return Ok(false);
            };
            return attach_native_linear(linear, target, native, report);
        }
        _ => {}
    }

    let Some(rest) = target.strip_prefix("model.language_model.layers.") else {
        return Ok(false);
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return Ok(false);
    };
    let Ok(layer) = layer.parse::<usize>() else {
        return Ok(false);
    };
    let Some(layer) = model.model.language_model.layers.get_mut(layer) else {
        return Ok(false);
    };

    if let Some(projection) = parameter
        .strip_prefix("experts.switch_glu.")
        .and_then(|name| name.strip_suffix(".weight"))
    {
        let Some(experts) = layer.experts.as_mut() else {
            return Ok(false);
        };
        let projection = match projection {
            "gate_proj" => &mut experts.switch_glu.gate_proj,
            "up_proj" => &mut experts.switch_glu.up_proj,
            "down_proj" => &mut experts.switch_glu.down_proj,
            _ => return Ok(false),
        };
        return attach_native_expert_projection(projection, target, native, report);
    }

    let linear = match parameter {
        "self_attn.q_proj.weight" => Some(&mut layer.self_attn.q_proj),
        "self_attn.k_proj.weight" => layer.self_attn.k_proj.as_mut(),
        "self_attn.v_proj.weight" => layer.self_attn.v_proj.as_mut(),
        "self_attn.o_proj.weight" => Some(&mut layer.self_attn.o_proj),
        "mlp.gate_proj.weight" => Some(&mut layer.mlp.gate_proj),
        "mlp.down_proj.weight" => Some(&mut layer.mlp.down_proj),
        "mlp.up_proj.weight" => Some(&mut layer.mlp.up_proj),
        "router.proj.weight" => layer.router.as_mut().map(|router| &mut router.proj),
        "per_layer_input_gate.weight" => layer.per_layer_input_gate.as_mut(),
        "per_layer_projection.weight" => layer.per_layer_projection.as_mut(),
        _ => None,
    };
    match linear {
        Some(linear) => attach_native_linear(linear, target, native, report),
        None => Ok(false),
    }
}

fn gemma4_args_from_gguf(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    stream: &Stream,
) -> Result<ModelArgs, Error> {
    let expert_count = gguf_optional_i64(metadata, "gemma4.expert_count", stream)?.unwrap_or(0);
    let enable_moe_block = expert_count > 0
        || arrays.any_gguf_tensor(|name| {
            name.contains("ffn_gate_up_exps.")
                || name.contains("ffn_gate_exps.")
                || name.contains("ffn_down_exps.")
        });
    let num_experts = enable_moe_block
        .then(|| {
            i32::try_from(expert_count).map_err(|_| {
                Error::UnsupportedArchitecture("Gemma 4 expert count exceeds i32".into())
            })
        })
        .transpose()?;
    let top_k_experts = enable_moe_block
        .then(|| gguf_i32(metadata, "gemma4.expert_used_count", stream))
        .transpose()?;
    let moe_intermediate_size = enable_moe_block
        .then(|| gguf_i32(metadata, "gemma4.expert_feed_forward_length", stream))
        .transpose()?;
    if enable_moe_block
        && (num_experts.is_none_or(|value| value <= 0)
            || top_k_experts.is_none_or(|value| value <= 0)
            || moe_intermediate_size.is_none_or(|value| value <= 0))
    {
        return Err(Error::UnsupportedArchitecture(
            "Gemma 4 GGUF has incomplete or invalid MoE metadata".into(),
        ));
    }

    let num_hidden_layers = gguf_i32(metadata, "gemma4.block_count", stream)?;
    let layer_pattern = gguf_optional_sliding_window_pattern(
        metadata,
        "gemma4.attention.sliding_window_pattern",
        stream,
    )?
    .unwrap_or_else(|| vec![0; num_hidden_layers as usize]);
    if layer_pattern.len() != num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "Gemma 4 sliding-window pattern has {} entries for {num_hidden_layers} layers",
            layer_pattern.len()
        )));
    }
    let layer_types = layer_pattern
        .into_iter()
        .map(|is_sliding| {
            if is_sliding != 0 {
                LayerType::SlidingAttention
            } else {
                LayerType::FullAttention
            }
        })
        .collect::<Vec<_>>();

    let feed_forward_values = gguf_i64_values(metadata, "gemma4.feed_forward_length", stream)?;
    let feed_forward_lengths = expand_layer_values(
        "gemma4.feed_forward_length",
        feed_forward_values,
        num_hidden_layers,
    )?;
    let intermediate_size = feed_forward_lengths[0];

    let kv_head_values = gguf_i64_values(metadata, "gemma4.attention.head_count_kv", stream)?;
    let kv_head_values = expand_layer_values(
        "gemma4.attention.head_count_kv",
        kv_head_values,
        num_hidden_layers,
    )?;
    let sliding_kv_heads = layer_types
        .iter()
        .zip(&kv_head_values)
        .find_map(|(kind, value)| (*kind == LayerType::SlidingAttention).then_some(*value))
        .unwrap_or(kv_head_values[0]);
    let full_kv_heads = layer_types
        .iter()
        .zip(&kv_head_values)
        .find_map(|(kind, value)| (*kind == LayerType::FullAttention).then_some(*value))
        .unwrap_or(sliding_kv_heads);
    for (kind, value) in layer_types.iter().zip(&kv_head_values) {
        let expected = if *kind == LayerType::FullAttention {
            full_kv_heads
        } else {
            sliding_kv_heads
        };
        if *value != expected {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 GGUF uses non-uniform KV-head counts within one attention type".into(),
            ));
        }
    }

    let hidden_size = gguf_i32(metadata, "gemma4.embedding_length", stream)?;
    let num_attention_heads = gguf_i32(metadata, "gemma4.attention.head_count", stream)?;
    let global_head_dim = gguf_i32(metadata, "gemma4.attention.key_length", stream)?;
    let head_dim = gguf_optional_i64(metadata, "gemma4.attention.key_length_swa", stream)?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("Gemma 4 SWA head size exceeds i32".into()))?
        .unwrap_or(global_head_dim);
    let num_kv_shared_layers =
        gguf_optional_i64(metadata, "gemma4.attention.shared_kv_layers", stream)?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| {
                Error::UnsupportedArchitecture("Gemma 4 shared-KV layer count exceeds i32".into())
            })?
            .unwrap_or(0);
    let hidden_size_per_layer_input =
        gguf_optional_i64(metadata, "gemma4.embedding_length_per_layer_input", stream)?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| {
                Error::UnsupportedArchitecture(
                    "Gemma 4 per-layer embedding size exceeds i32".into(),
                )
            })?
            .unwrap_or(0);
    let vocab_size = match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len()).map_err(|_| {
            Error::UnsupportedArchitecture("GGUF tokenizer vocabulary exceeds i32".into())
        })?,
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type".into(),
            ));
        }
        None => gguf_i32(metadata, "gemma4.vocab_size", stream)?,
    };

    let full_rope_theta =
        gguf_optional_f32(metadata, "gemma4.rope.freq_base", stream)?.unwrap_or(1_000_000.0);
    let sliding_rope_theta =
        gguf_optional_f32(metadata, "gemma4.rope.freq_base_swa", stream)?.unwrap_or(10_000.0);
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

    let first_shared_layer = num_hidden_layers - num_kv_shared_layers;
    let attention_k_eq_v = layer_types
        .iter()
        .enumerate()
        .find(|(index, kind)| {
            **kind == LayerType::FullAttention && *index < first_shared_layer.max(0) as usize
        })
        .is_some_and(|(index, _)| {
            arrays.contains_gguf_tensor(&format!("blk.{index}.attn_k.weight"))
                && !arrays.contains_gguf_tensor(&format!("blk.{index}.attn_v.weight"))
        });

    Ok(ModelArgs {
        model_type: "gemma4".into(),
        hidden_size,
        num_hidden_layers,
        intermediate_size,
        use_double_wide_mlp: false,
        feed_forward_lengths: Some(feed_forward_lengths),
        num_attention_heads,
        rms_norm_eps: gguf_f32(metadata, "gemma4.attention.layer_norm_rms_epsilon", stream)?,
        vocab_size,
        pad_token_id: gguf_optional_i64(metadata, "tokenizer.ggml.padding_token_id", stream)?
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or(0),
        num_key_value_heads: sliding_kv_heads,
        num_global_key_value_heads: (full_kv_heads != sliding_kv_heads).then_some(full_kv_heads),
        max_position_embeddings: gguf_i32(metadata, "gemma4.context_length", stream)?,
        rope_theta: sliding_rope_theta,
        head_dim,
        global_head_dim: (global_head_dim != head_dim).then_some(global_head_dim),
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        attention_bias: arrays.any_gguf_tensor(|name| {
            name.ends_with("attn_q.bias")
                || name.ends_with("attn_k.bias")
                || name.ends_with("attn_v.bias")
                || name.ends_with("attn_output.bias")
        }),
        attention_k_eq_v,
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
        sliding_window: gguf_optional_i64(metadata, "gemma4.attention.sliding_window", stream)?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| {
                Error::UnsupportedArchitecture("Gemma 4 sliding window exceeds i32".into())
            })?,
        final_logit_softcapping: gguf_optional_f32(
            metadata,
            "gemma4.final_logit_softcapping",
            stream,
        )?,
        enable_moe_block,
        num_experts,
        top_k_experts,
        moe_intermediate_size,
        rope_scaling: None,
        rope_parameters,
    })
}

pub(super) fn expand_layer_values(
    key: &str,
    values: Vec<i64>,
    num_hidden_layers: i32,
) -> Result<Vec<i32>, Error> {
    let values = if values.len() == 1 {
        vec![values[0]; num_hidden_layers as usize]
    } else if values.len() == num_hidden_layers as usize {
        values
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has {} values for {num_hidden_layers} layers",
            values.len()
        )));
    };
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
            })
        })
        .collect()
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    const ROOTS: [(&str, &str); 6] = [
        (
            "per_layer_token_embd",
            "model.language_model.embed_tokens_per_layer",
        ),
        (
            "per_layer_model_proj",
            "model.language_model.per_layer_model_projection",
        ),
        (
            "per_layer_proj_norm",
            "model.language_model.per_layer_projection_norm",
        ),
        ("token_embd", "model.language_model.embed_tokens"),
        ("output_norm", "model.language_model.norm"),
        ("output", "lm_head"),
    ];
    for (source, target) in ROOTS {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }

    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };
    if parameter == "ffn_gate_inp.scale" {
        return format!("model.language_model.layers.{layer}.router.scale");
    }
    if parameter == "ffn_down_exps.scale" {
        return format!("model.language_model.layers.{layer}.router.per_expert_scale");
    }
    const EXPERT_PARAMETERS: [(&str, &str); 4] = [
        ("ffn_gate_up_exps", "experts.switch_glu.gate_up_proj"),
        ("ffn_gate_exps", "experts.switch_glu.gate_proj"),
        ("ffn_up_exps", "experts.switch_glu.up_proj"),
        ("ffn_down_exps", "experts.switch_glu.down_proj"),
    ];
    for (source, target) in EXPERT_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = parameter.strip_prefix(source).unwrap_or_default();
            return format!("model.language_model.layers.{layer}.{target}{suffix}");
        }
    }
    if parameter == "layer_output_scale.weight" {
        return format!("model.language_model.layers.{layer}.layer_scalar");
    }
    const BLOCK_PARAMETERS: [(&str, &str); 22] = [
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("ffn_norm", "pre_feedforward_layernorm"),
        ("post_ffw_norm", "post_feedforward_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_gate_inp", "router.proj"),
        ("pre_ffw_norm_2", "pre_feedforward_layernorm_2"),
        ("post_ffw_norm_1", "post_feedforward_layernorm_1"),
        ("post_ffw_norm_2", "post_feedforward_layernorm_2"),
        ("inp_gate", "per_layer_input_gate"),
        ("proj", "per_layer_projection"),
        ("post_norm", "post_per_layer_input_norm"),
        ("layer_output_scale", "layer_scalar"),
        ("layer_output_norm", "layer_output_norm"),
    ];
    for (source, target) in BLOCK_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.language_model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

fn gguf_string(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<String, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata is missing required key {key:?}"
        ))),
    }
}

pub(super) fn gguf_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<i32, Error> {
    i32::try_from(gguf_i64(metadata, key, stream)?).map_err(|_| {
        Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
    })
}

fn gguf_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<i64, Error> {
    gguf_optional_i64(metadata, key, stream)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

pub(super) fn gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<Option<i64>, Error> {
    let Some(values) = gguf_optional_i64_values(metadata, key, stream)? else {
        return Ok(None);
    };
    if values.len() != 1 {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} must be scalar"
        )));
    }
    Ok(values.into_iter().next())
}

pub(super) fn gguf_i64_values(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<Vec<i64>, Error> {
    gguf_optional_i64_values(metadata, key, stream)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_i64_values(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    _stream: &Stream,
) -> Result<Option<Vec<i64>>, Error> {
    match metadata.get(key) {
        Some(value) => value.to_i64_vec().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
        }),
        None => Ok(None),
    }
}

pub(super) fn gguf_optional_sliding_window_pattern(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<Option<Vec<i64>>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(values))) => {
            Ok(Some(values.iter().map(|&value| i64::from(value)).collect()))
        }
        _ => gguf_optional_i64_values(metadata, key, stream),
    }
}

pub(super) fn gguf_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<f32, Error> {
    gguf_optional_f32(metadata, key, stream)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

pub(super) fn gguf_optional_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    _stream: &Stream,
) -> Result<Option<f32>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {key:?} must be a numeric scalar"
            ))
        }),
        None => Ok(None),
    }
}

fn quantization_i32(config: &Option<Value>, key: &str, default: i32) -> i32 {
    config
        .as_ref()
        .and_then(|config| config.get(key))
        .and_then(|value| value.as_i64())
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(default)
}

/// Reads and normalizes Gemma 4 text model arguments from `config.json`.
pub fn get_gemma4_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    Ok(get_gemma4_model_config(model_dir.as_ref())?.0)
}

pub(crate) type Gemma4ModelConfigParts = (
    ModelArgs,
    Option<Gemma4VisionConfig>,
    Option<i32>,
    Option<i32>,
    Option<Gemma4AudioConfig>,
    Option<i32>,
);

pub(crate) fn get_gemma4_model_config(model_dir: &Path) -> Result<Gemma4ModelConfigParts, Error> {
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    let mut config: Gemma4Config = serde_json::from_reader(file)?;
    validate_moe_args(&config.text_config)?;
    config.text_config.model_type = "gemma4".to_string();
    config.text_config.quantized = config.quantization.is_some();
    config.text_config.weight_quantization = config
        .quantization
        .clone()
        .map(serde_json::from_value)
        .transpose()?;
    config.text_config.quantization_group_size =
        quantization_i32(&config.quantization, "group_size", 64);
    config.text_config.quantization_bits = quantization_i32(&config.quantization, "bits", 4);
    config.text_config.tie_word_embeddings = config.tie_word_embeddings;
    Ok((
        config.text_config,
        config.vision_config,
        config.image_token_id,
        config.video_token_id,
        config.audio_config,
        config.audio_token_id,
    ))
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    let config: Gemma4Config = serde_json::from_value(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Gemma 4 config: {error}"))
    })?;
    validate_moe_args(&config.text_config)
}

fn validate_moe_args(args: &ModelArgs) -> Result<(), Error> {
    if !args.enable_moe_block {
        return Ok(());
    }
    let num_experts = args.num_experts.unwrap_or(0);
    let top_k = args.top_k_experts.unwrap_or(0);
    let intermediate = args.moe_intermediate_size.unwrap_or(0);
    if num_experts <= 0 || top_k <= 0 || top_k > num_experts || intermediate <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "Gemma 4 MoE requires positive num_experts, top_k_experts, and moe_intermediate_size, with top_k_experts no greater than num_experts".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
/// Hugging Face safetensors index file.
pub struct WeightMap {
    /// Index metadata.
    pub metadata: HashMap<String, Value>,
    /// Mapping from tensor name to shard file name.
    pub weight_map: HashMap<String, String>,
}

/// Loads a Gemma 4 model and safetensors weights from a model directory.
pub fn load_gemma4_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let (model_args, vision_config, image_token_id, video_token_id, audio_config, audio_token_id) =
        get_gemma4_model_config(model_dir)?;
    let mut model = Model::new_with_modalities(
        model_args,
        image_token_id,
        vision_config,
        video_token_id,
        audio_token_id,
        audio_config,
        stream,
    )?;
    let config = gemma4_strict_load_config();
    let mut report = StrictLoadReport::default();
    load_gemma4_weights(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        None,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads a Gemma 4 checkpoint while affine-quantizing supported weights.
///
/// Transformer weights and modality bridge projections use affine storage.
/// Vision and audio towers remain dense because their convolutional and
/// specialized implementations do not expose MLX affine parameter layouts. A
/// checkpoint already carrying matching affine metadata is loaded directly
/// without requantization.
pub fn load_gemma4_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let (
        mut model_args,
        vision_config,
        image_token_id,
        video_token_id,
        audio_config,
        audio_token_id,
    ) = get_gemma4_model_config(model_dir)?;
    if !crate::quantization::should_quantize_on_load(
        "Gemma 4",
        model_args.weight_quantization(),
        quantization,
    )? {
        return load_gemma4_model(model_dir, stream, weights_stream);
    }
    model_args.quantized = true;
    model_args.weight_quantization = Some(quantization);
    model_args.quantization_group_size = quantization.group_size();
    model_args.quantization_bits = quantization.bits();
    let mut model = Model::new_with_modalities(
        model_args,
        image_token_id,
        vision_config,
        video_token_id,
        audio_token_id,
        audio_config,
        stream,
    )?;
    let config = gemma4_strict_load_config();
    let mut report = StrictLoadReport::default();
    load_gemma4_weights(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        Some(quantization),
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

fn gemma4_strict_load_config() -> StrictLoadConfig {
    StrictLoadConfig::default()
        .rewrite_prefix("language_model.model.", "model.language_model.")
        .rewrite_prefix("model.language_model.", "model.language_model.")
        .rewrite_prefix("vision_tower.", "model.vision_tower.")
        .rewrite_prefix("embed_vision.", "model.embed_vision.")
        .rewrite_prefix("audio_tower.", "model.audio_tower.")
        .rewrite_prefix("embed_audio.", "model.embed_audio.")
        .allow_unused_prefix("multi_modal_projector.")
        .allow_unused_prefix("model.multi_modal_projector.")
        .allow_unused_prefix("model.vision_embedder.")
        .allow_missing_suffix(".bias")
}

fn load_gemma4_weights(
    model: &mut Model,
    model_dir: &Path,
    weights_stream: &Stream,
    quantization_stream: &Stream,
    quantization: Option<WeightQuantization>,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");
    let mut load_file = |path: &Path| match quantization {
        Some(quantization) => load_safetensors_quantized_strict(
            model,
            path,
            weights_stream,
            quantization_stream,
            quantization,
            config,
            report,
        ),
        None => load_safetensors_strict(model, path, weights_stream, config, report),
    };
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            load_file(&weights_filename)?;
        }
    } else {
        load_file(&model_dir.join("model.safetensors"))?;
    }
    Ok(())
}

impl Model {
    /// Runs a Gemma 4 forward pass and returns logits plus assistant-drafting state.
    pub(crate) fn forward_with_state(
        &mut self,
        input: ModelInput<'_, ConcatKeyValueCache>,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        let text_output = self
            .model
            .language_model
            .forward_with_state(input, stream)?;
        let logits = self.project_logits(&text_output.hidden, stream)?;
        Ok(Gemma4StepOutput {
            logits,
            hidden: text_output.pre_norm_hidden,
            shared_kv_states: text_output.shared_kv_states,
        })
    }

    /// Prefills typed input while retaining the state required by an external drafter.
    pub(crate) fn prefill_mtp(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        match self.prepare_typed_prefill(input, stream)? {
            input::PreparedPrefill::Text(prompt_tokens) => {
                cache.token_ids = token_ids_from_array(&prompt_tokens, stream)?;
                cache.prefix_embeddings = None;
                cache.prefix_len = 0;
                cache.reset_kv(&self.args);
                self.forward_with_state(
                    ModelInput {
                        inputs: &prompt_tokens,
                        inputs_embeds: None,
                        per_layer_input_ids: None,
                        mask: None,
                        sliding_mask: None,
                        cache: &mut cache.kv,
                    },
                    stream,
                )
            }
            input::PreparedPrefill::Embeddings { tokens, embeddings } => {
                cache.token_ids = token_ids_from_array(&tokens, stream)?;
                cache.prefix_len = cache.token_ids.len();
                cache.prefix_embeddings = Some(embeddings.clone());
                cache.reset_kv(&self.args);
                let per_layer_ids = self.per_layer_ids_for_media(&tokens, stream)?;
                let masks = multimodal_attention_masks(
                    &cache.token_ids,
                    self.image_token_id.map(|id| id as u32),
                    self.video_token_id.map(|id| id as u32),
                    self.args.sliding_window,
                );
                self.forward_with_state(
                    ModelInput {
                        inputs: &tokens,
                        inputs_embeds: Some(&embeddings),
                        per_layer_input_ids: Some(&per_layer_ids),
                        mask: Some(&masks.full),
                        sliding_mask: Some(&masks.sliding),
                        cache: &mut cache.kv,
                    },
                    stream,
                )
            }
        }
    }

    /// Verifies one speculative block without rebuilding the committed prefix.
    pub(crate) fn verify_mtp(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        cache
            .token_ids
            .extend(token_ids_from_array(input_tokens, stream)?);
        self.forward_with_state(
            ModelInput {
                inputs: input_tokens,
                inputs_embeds: None,
                per_layer_input_ids: None,
                mask: None,
                sliding_mask: None,
                cache: &mut cache.kv,
            },
            stream,
        )
    }

    /// Copies the target token embedding table onto the draft stream.
    pub(crate) fn mtp_embedding_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<Gemma4Embedding, Exception> {
        let mut embedding = self.model.language_model.embed_tokens.clone();
        if copy {
            embedding.copy_to_stream(stream)?;
            embedding.native = embedding
                .native
                .as_ref()
                .map(|native| native.copy_to_stream(stream))
                .transpose()?;
            stream.synchronize()?;
        }
        Ok(embedding)
    }
}

/// Output for a Gemma 4 target-model step used by assistant drafting.
pub(crate) struct Gemma4StepOutput {
    /// Logits for the step.
    pub logits: Array,
    /// Pre-final-normalization hidden states.
    pub hidden: Array,
    /// Shared key/value states for assistant drafting.
    pub shared_kv_states: HashMap<LayerType, (Array, Array)>,
}

impl<C> CausalLm<Vec<Option<C>>> for Model
where
    C: KeyValueCache + Default,
{
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prompt_tokens = input::text_token_ids(input, stream)?;
        self.forward_logits(
            ModelInput {
                inputs: &prompt_tokens,
                inputs_embeds: None,
                per_layer_input_ids: None,
                mask: None,
                sliding_mask: None,
                cache,
            },
            true,
            stream,
        )
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits(
            ModelInput {
                inputs: input_tokens,
                inputs_embeds: None,
                per_layer_input_ids: None,
                mask: None,
                sliding_mask: None,
                cache,
            },
            true,
            stream,
        )
    }
}

/// Gemma 4 generation cache.
#[derive(Clone, Default)]
pub struct Cache {
    pub(crate) kv: Vec<Option<ConcatKeyValueCache>>,
    pub(crate) token_ids: Vec<u32>,
    prefix_embeddings: Option<Array>,
    prefix_len: usize,
}

impl Cache {
    const KV_GROWTH_STEP: i32 = 256;

    pub(crate) fn new(args: &ModelArgs) -> Self {
        let mut cache = Self::default();
        cache.reset_kv(args);
        cache
    }

    pub(crate) fn reset_kv(&mut self, args: &ModelArgs) {
        self.kv = (0..args.num_hidden_layers)
            .map(|_| Some(ConcatKeyValueCache::new_with_step(Self::KV_GROWTH_STEP)))
            .collect();
    }

    /// Returns the committed logical sequence length.
    pub(crate) fn mtp_len(&self) -> usize {
        self.token_ids.len()
    }
}

impl Model {
    pub(crate) fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }
}

pub(crate) fn token_ids_from_array(tokens: &Array, stream: &Stream) -> Result<Vec<u32>, Exception> {
    let shape = tokens.shape();
    if shape.len() != 2 || shape[0] != 1 {
        return Err(Exception::custom(format!(
            "Gemma 4 generation expects batch-1 token ids, got shape {shape:?}"
        )));
    }
    let mut ids = Vec::with_capacity(shape[1] as usize);
    for index in 0..shape[1] {
        ids.push(
            tokens
                .try_index_device((0, index), stream)?
                .item::<u32>(stream),
        );
    }
    Ok(ids)
}

fn array_from_token_ids(token_ids: &[u32], stream: &Stream) -> Result<Array, Exception> {
    Array::from(token_ids).try_index_device(NewAxis, stream)
}

pub(crate) struct Gemma4AttentionMasks {
    pub(crate) full: Array,
    pub(crate) sliding: Array,
}

pub(crate) fn multimodal_attention_masks(
    token_ids: &[u32],
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    sliding_window: Option<i32>,
) -> Gemma4AttentionMasks {
    let sequence = token_ids.len();
    let mut groups = vec![-1i32; sequence];
    let mut group = -1i32;
    let mut previous_visual_token = None;
    for (index, token_id) in token_ids.iter().enumerate() {
        let is_visual = image_token_id == Some(*token_id) || video_token_id == Some(*token_id);
        if is_visual && previous_visual_token != Some(*token_id) {
            group += 1;
        }
        if is_visual {
            groups[index] = group;
        }
        previous_visual_token = is_visual.then_some(*token_id);
    }
    let window = sliding_window.unwrap_or(sequence as i32);
    let mut full = Vec::with_capacity(sequence * sequence);
    let mut sliding = Vec::with_capacity(sequence * sequence);
    for query in 0..sequence {
        for key in 0..sequence {
            let causal = key <= query;
            full.push(if causal { 0.0 } else { -1.0e9 });
            let same_image_group = groups[query] >= 0 && groups[query] == groups[key];
            let in_window = key as i32 > query as i32 - window;
            sliding.push(if in_window && (causal || same_image_group) {
                0.0
            } else {
                -1.0e9
            });
        }
    }
    let shape = [1, 1, sequence as i32, sequence as i32];
    Gemma4AttentionMasks {
        full: Array::from_slice(&full, &shape),
        sliding: Array::from_slice(&sliding, &shape),
    }
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self.prepare_typed_prefill(input, stream)? {
            input::PreparedPrefill::Text(prompt_tokens) => {
                cache.token_ids = token_ids_from_array(&prompt_tokens, stream)?;
                cache.prefix_embeddings = None;
                cache.prefix_len = 0;
                cache.reset_kv(&self.args);
                self.forward_logits(
                    ModelInput {
                        inputs: &prompt_tokens,
                        inputs_embeds: None,
                        per_layer_input_ids: None,
                        mask: None,
                        sliding_mask: None,
                        cache: &mut cache.kv,
                    },
                    true,
                    stream,
                )
            }
            input::PreparedPrefill::Embeddings { tokens, embeddings } => {
                cache.token_ids = token_ids_from_array(&tokens, stream)?;
                cache.prefix_len = cache.token_ids.len();
                cache.prefix_embeddings = Some(embeddings.clone());
                cache.reset_kv(&self.args);
                let per_layer_ids = self.per_layer_ids_for_media(&tokens, stream)?;
                let masks = multimodal_attention_masks(
                    &cache.token_ids,
                    self.image_token_id.map(|id| id as u32),
                    self.video_token_id.map(|id| id as u32),
                    self.args.sliding_window,
                );
                self.forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        inputs_embeds: Some(&embeddings),
                        per_layer_input_ids: Some(&per_layer_ids),
                        mask: Some(&masks.full),
                        sliding_mask: Some(&masks.sliding),
                        cache: &mut cache.kv,
                    },
                    true,
                    stream,
                )
            }
        }
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        cache
            .token_ids
            .extend(token_ids_from_array(input_tokens, stream)?);
        if cache.prefix_embeddings.is_none() {
            return self.forward_logits(
                ModelInput {
                    inputs: input_tokens,
                    inputs_embeds: None,
                    per_layer_input_ids: None,
                    mask: None,
                    sliding_mask: None,
                    cache: &mut cache.kv,
                },
                true,
                stream,
            );
        }

        // Media tokens use non-causal visibility within each media group. The
        // ordinary KV cache cannot preserve that structured mask for appended
        // text, so multimodal generation still replays the prepared prefix.
        cache.reset_kv(&self.args);
        let tokens = array_from_token_ids(&cache.token_ids, stream)?;
        let generated_embeddings = cache
            .prefix_embeddings
            .as_ref()
            .map(|prefix| {
                let generated = array_from_token_ids(&cache.token_ids[cache.prefix_len..], stream)?;
                let generated = self
                    .model
                    .language_model
                    .embed_tokens
                    .forward(&generated, stream)?
                    .multiply(
                        Array::from_f32((self.args.hidden_size as f32).sqrt()),
                        stream,
                    )?;
                concatenate_axis(&[prefix.clone(), generated], 1, stream)
            })
            .transpose()?;
        let per_layer_ids = generated_embeddings
            .as_ref()
            .map(|_| self.per_layer_ids_for_media(&tokens, stream))
            .transpose()?;
        let masks = generated_embeddings.as_ref().map(|_| {
            multimodal_attention_masks(
                &cache.token_ids,
                self.image_token_id.map(|id| id as u32),
                self.video_token_id.map(|id| id as u32),
                self.args.sliding_window,
            )
        });
        self.forward_logits(
            ModelInput {
                inputs: &tokens,
                inputs_embeds: generated_embeddings.as_ref(),
                per_layer_input_ids: per_layer_ids.as_ref(),
                mask: masks.as_ref().map(|masks| &masks.full),
                sliding_mask: masks.as_ref().map(|masks| &masks.sliding),
                cache: &mut cache.kv,
            },
            true,
            stream,
        )
    }
}

/// Gemma 4 token generation iterator.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{
        module::ModuleParameters,
        ops::{zeros_dtype, GgufMetadataValue},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::{
        load_gemma4_model, needs_generated_sliding_mask, partial_rotary_dims, Attention, Cache,
        FloatOrString, LayerType, ModelArgs,
    };
    use crate::models::{
        common::generation::CausalLm,
        input::{InputMetadata, InputPart, ModelInput},
    };
    use crate::weights::{load_arrays_strict, StrictLoadConfig, StrictLoadReport};

    fn test_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn model_args(attention_k_eq_v: bool) -> ModelArgs {
        ModelArgs {
            model_type: "gemma4_unified_text".to_string(),
            hidden_size: 8,
            num_hidden_layers: 1,
            intermediate_size: 16,
            use_double_wide_mlp: false,
            feed_forward_lengths: None,
            num_attention_heads: 2,
            rms_norm_eps: 0.00001,
            vocab_size: 32,
            pad_token_id: 0,
            num_key_value_heads: 1,
            num_global_key_value_heads: None,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            head_dim: 4,
            global_head_dim: None,
            tie_word_embeddings: true,
            attention_bias: false,
            attention_k_eq_v,
            quantized: false,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            quantization_group_size: 64,
            quantization_bits: 4,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            num_kv_shared_layers: 0,
            layer_types: vec![LayerType::FullAttention],
            sliding_window: None,
            final_logit_softcapping: None,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            rope_scaling: None,
            rope_parameters: None,
        }
    }

    #[test]
    #[ignore = "requires a local Gemma 4 E4B checkpoint and Metal"]
    fn local_e4b_image_prefill_and_decode() {
        let home = std::env::var("HOME").unwrap();
        let snapshots = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots");
        let model_dir = std::fs::read_dir(&snapshots)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.join("model.safetensors").exists())
            .unwrap();
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model = load_gemma4_model(&model_dir, gpu.stream(), weights.stream()).unwrap();
        let tokens = Array::from_slice(&[2u32, 258880, 7], &[1, 3]);
        let patches = Array::from_slice(&vec![0.5f32; 9 * 16 * 16 * 3], &[1, 9, 768]);
        let positions = Array::from_slice(
            &[0i32, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2],
            &[1, 9, 2],
        );
        let parts = [
            InputPart::text_token_ids(&tokens),
            InputPart::image_tensor(&patches, InputMetadata::patch_position_ids(&positions)),
        ];
        let mut cache = Cache::default();
        let logits = model
            .prefill_input_logits(ModelInput::new(&parts), &mut cache, gpu.stream())
            .unwrap();
        assert_eq!(logits.shape(), &[1, 262144]);
        let decode = Array::from_slice(&[8u32], &[1, 1]);
        let logits = model
            .decode_logits(&decode, &mut cache, gpu.stream())
            .unwrap();
        assert_eq!(logits.shape(), &[1, 262144]);
    }

    #[test]
    #[ignore = "requires a local Gemma 4 E4B checkpoint and Metal"]
    fn local_e4b_video_prefill_and_decode() {
        let home = std::env::var("HOME").unwrap();
        let snapshots = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots");
        let model_dir = std::fs::read_dir(&snapshots)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.join("model.safetensors").exists())
            .unwrap();
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model = load_gemma4_model(&model_dir, gpu.stream(), weights.stream()).unwrap();
        let tokens = Array::from_slice(&[2u32, 258884, 7, 258884, 8], &[1, 5]);
        let patches = Array::from_slice(&vec![0.5f32; 2 * 9 * 16 * 16 * 3], &[2, 9, 768]);
        let frame_positions = [0i32, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2];
        let positions = Array::from_slice(&[frame_positions, frame_positions].concat(), &[2, 9, 2]);
        let parts = [
            InputPart::text_token_ids(&tokens),
            InputPart::video_tensor(&patches, InputMetadata::patch_position_ids(&positions)),
        ];
        let mut cache = Cache::default();
        let logits = model
            .prefill_input_logits(ModelInput::new(&parts), &mut cache, gpu.stream())
            .unwrap();
        assert_eq!(logits.shape(), &[1, 262144]);
        let decode = Array::from_slice(&[8u32], &[1, 1]);
        let logits = model
            .decode_logits(&decode, &mut cache, gpu.stream())
            .unwrap();
        assert_eq!(logits.shape(), &[1, 262144]);
    }

    #[test]
    #[ignore = "requires a local Gemma 4 E4B checkpoint and Metal"]
    fn local_e4b_audio_prefill_and_decode() {
        let home = std::env::var("HOME").unwrap();
        let snapshots = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots");
        let model_dir = std::fs::read_dir(&snapshots)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.join("model.safetensors").exists())
            .unwrap();
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model = load_gemma4_model(&model_dir, gpu.stream(), weights.stream()).unwrap();
        let tokens = Array::from_slice(&[2u32, 258881, 7], &[1, 3]);
        let features = Array::from_slice(&vec![0.0f32; 16 * 128], &[1, 16, 128]);
        let mask = Array::from_slice(&[true; 16], &[1, 16]);
        let parts = [
            InputPart::text_token_ids(&tokens),
            InputPart::audio_tensor(&features, InputMetadata::audio_mask(&mask)),
        ];
        let mut cache = Cache::default();
        let logits = model
            .prefill_input_logits(ModelInput::new(&parts), &mut cache, gpu.stream())
            .unwrap();
        assert_eq!(logits.shape(), &[1, 262144]);
        let decode = Array::from_slice(&[8u32], &[1, 1]);
        let logits = model
            .decode_logits(&decode, &mut cache, gpu.stream())
            .unwrap();
        assert_eq!(logits.shape(), &[1, 262144]);
    }

    fn parameter_keys(attention: &Attention) -> Vec<String> {
        let mut keys = attention
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        keys.sort();
        keys
    }

    #[test]
    fn proportional_rope_keeps_full_head_dimensions() {
        let scaling = Some(HashMap::from([
            (
                "rope_type".to_string(),
                FloatOrString::String("proportional".to_string()),
            ),
            (
                "partial_rotary_factor".to_string(),
                FloatOrString::Float(0.25),
            ),
        ]));

        assert_eq!(partial_rotary_dims(512, &scaling), 512);
    }

    #[test]
    fn rotary_dims_default_to_full_head() {
        assert_eq!(partial_rotary_dims(256, &None), 256);
    }

    #[test]
    fn single_token_sliding_mask_starts_only_after_window_fills() {
        assert!(!needs_generated_sliding_mask(1, 0, Some(1024)));
        assert!(!needs_generated_sliding_mask(1, 1023, Some(1024)));
        assert!(needs_generated_sliding_mask(1, 1024, Some(1024)));
        assert!(needs_generated_sliding_mask(2, 0, Some(1024)));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn moe_parameter_tree_matches_published_safetensors_layout() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut args = model_args(true);
        args.enable_moe_block = true;
        args.num_experts = Some(4);
        args.top_k_experts = Some(2);
        args.moe_intermediate_size = Some(8);
        let model = super::Model::new(args, context.stream()).unwrap();
        let params = model.parameters().flatten();
        let prefix = "model.language_model.layers.0";

        for key in [
            format!("{prefix}.router.proj.weight"),
            format!("{prefix}.router.scale"),
            format!("{prefix}.router.per_expert_scale"),
            format!("{prefix}.experts.switch_glu.gate_proj.weight"),
            format!("{prefix}.experts.switch_glu.up_proj.weight"),
            format!("{prefix}.experts.switch_glu.down_proj.weight"),
            format!("{prefix}.post_feedforward_layernorm_1.weight"),
            format!("{prefix}.pre_feedforward_layernorm_2.weight"),
            format!("{prefix}.post_feedforward_layernorm_2.weight"),
        ] {
            assert!(params.contains_key(key.as_str()), "missing {key}");
        }
    }

    #[test]
    #[ignore = "requires Metal"]
    fn tiny_moe_prefill_and_decode() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut args = model_args(true);
        args.enable_moe_block = true;
        args.num_experts = Some(4);
        args.top_k_experts = Some(2);
        args.moe_intermediate_size = Some(8);
        let mut model = super::Model::new(args, stream).unwrap();
        let arrays = model
            .parameters()
            .flatten()
            .iter()
            .map(|(name, parameter)| {
                (
                    name.to_string(),
                    zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        let config = StrictLoadConfig::default();
        let mut report = StrictLoadReport::default();
        load_arrays_strict(&mut model, arrays, &config, &mut report).unwrap();
        report.finish(&model, &config).unwrap();

        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [InputPart::text_token_ids(&tokens)];
        let mut cache = Cache::default();
        let logits = model
            .prefill_input_logits(ModelInput::new(&parts), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 32]);
        logits.evaluated().unwrap();

        let decode = Array::from_slice(&[3u32], &[1, 1]);
        let logits = model.decode_logits(&decode, &mut cache, stream).unwrap();
        assert_eq!(logits.shape(), &[1, 32]);
        logits.evaluated().unwrap();

        let mut mtp_cache = Cache::default();
        let state = model
            .prefill_mtp(ModelInput::new(&parts), &mut mtp_cache, stream)
            .unwrap();
        assert!(state
            .shared_kv_states
            .contains_key(&LayerType::FullAttention));
    }

    #[test]
    fn translates_gguf_gemma4_weight_names() {
        let cases = [
            (
                "token_embd.weight",
                "model.language_model.embed_tokens.weight",
            ),
            (
                "per_layer_token_embd.weight",
                "model.language_model.embed_tokens_per_layer.weight",
            ),
            (
                "per_layer_model_proj.weight",
                "model.language_model.per_layer_model_projection.weight",
            ),
            (
                "blk.3.attn_q.weight",
                "model.language_model.layers.3.self_attn.q_proj.weight",
            ),
            (
                "blk.3.post_ffw_norm.weight",
                "model.language_model.layers.3.post_feedforward_layernorm.weight",
            ),
            (
                "blk.20.layer_output_scale.weight",
                "model.language_model.layers.20.layer_scalar",
            ),
            (
                "blk.3.ffn_gate_inp.weight",
                "model.language_model.layers.3.router.proj.weight",
            ),
            (
                "blk.3.ffn_gate_inp.scale",
                "model.language_model.layers.3.router.scale",
            ),
            (
                "blk.3.ffn_gate_up_exps.scales",
                "model.language_model.layers.3.experts.switch_glu.gate_up_proj.scales",
            ),
            (
                "blk.3.ffn_down_exps.scale",
                "model.language_model.layers.3.router.per_expert_scale",
            ),
            (
                "blk.3.ffn_down_exps.biases",
                "model.language_model.layers.3.experts.switch_glu.down_proj.biases",
            ),
            (
                "blk.3.pre_ffw_norm_2.weight",
                "model.language_model.layers.3.pre_feedforward_layernorm_2.weight",
            ),
            (
                "blk.3.post_ffw_norm_1.weight",
                "model.language_model.layers.3.post_feedforward_layernorm_1.weight",
            ),
            (
                "blk.3.post_ffw_norm_2.weight",
                "model.language_model.layers.3.post_feedforward_layernorm_2.weight",
            ),
        ];

        for (gguf, model) in cases {
            assert_eq!(super::translate_gguf_weight_name(gguf), model);
        }
    }

    #[test]
    fn parses_gemma4_gguf_layer_metadata() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let metadata = HashMap::from([
            (
                "gemma4.embedding_length".into(),
                GgufMetadataValue::Uint32(1536),
            ),
            ("gemma4.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "gemma4.feed_forward_length".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![
                    6144, 12288,
                ])),
            ),
            (
                "gemma4.attention.head_count".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![1, 2])),
            ),
            (
                "gemma4.attention.key_length".into(),
                GgufMetadataValue::Uint32(512),
            ),
            (
                "gemma4.attention.key_length_swa".into(),
                GgufMetadataValue::Uint32(256),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(vec![true, false])),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "gemma4.context_length".into(),
                GgufMetadataValue::Uint32(131072),
            ),
            (
                "gemma4.rope.freq_base".into(),
                GgufMetadataValue::Float32(1_000_000.0),
            ),
            (
                "gemma4.rope.freq_base_swa".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::String(vec![
                    "token"
                        .into();
                    32
                ])),
            ),
        ]);

        let args = super::gemma4_args_from_gguf(&HashMap::new(), &metadata, stream).unwrap();
        assert_eq!(args.feed_forward_lengths, Some(vec![6144, 12288]));
        assert_eq!(
            args.layer_types,
            vec![LayerType::SlidingAttention, LayerType::FullAttention]
        );
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.global_head_dim, Some(512));
        assert_eq!(args.num_key_value_heads, 1);
        assert_eq!(args.num_global_key_value_heads, Some(2));
        assert_eq!(args.num_kv_shared_layers, 1);
        assert_eq!(args.vocab_size, 32);
    }

    #[test]
    fn parses_gemma4_moe_gguf_metadata() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let mut metadata = HashMap::from([
            (
                "gemma4.embedding_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            ("gemma4.block_count".into(), GgufMetadataValue::Uint32(1)),
            (
                "gemma4.feed_forward_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "gemma4.attention.head_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "gemma4.attention.key_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "gemma4.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            ("gemma4.vocab_size".into(), GgufMetadataValue::Uint32(32)),
            ("gemma4.expert_count".into(), GgufMetadataValue::Uint32(8)),
            (
                "gemma4.expert_used_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "gemma4.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(16),
            ),
        ]);
        metadata.insert(
            "gemma4.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(vec![false])),
        );
        let arrays = HashMap::<String, Array>::new();

        let args = super::gemma4_args_from_gguf(&arrays, &metadata, stream).unwrap();
        assert!(args.enable_moe_block);
        assert_eq!(args.num_experts, Some(8));
        assert_eq!(args.top_k_experts, Some(2));
        assert_eq!(args.moe_intermediate_size, Some(16));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_attention_with_key_equal_value_does_not_allocate_v_proj() {
        let stream = test_stream();
        let attention =
            Attention::new(&model_args(true), LayerType::FullAttention, 0, &stream).unwrap();
        let keys = parameter_keys(&attention);

        assert!(keys.iter().any(|key| key.starts_with("q_proj.")));
        assert!(keys.iter().any(|key| key.starts_with("k_proj.")));
        assert!(!keys.iter().any(|key| key.starts_with("v_proj.")));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn attention_allocates_v_proj_when_key_equal_value_is_disabled() {
        let stream = test_stream();
        let attention =
            Attention::new(&model_args(false), LayerType::FullAttention, 0, &stream).unwrap();
        let keys = parameter_keys(&attention);

        assert!(keys.iter().any(|key| key.starts_with("v_proj.")));
    }
}
