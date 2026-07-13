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
    fast::{MetalKernel, MetalKernelConfig, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        concatenate_axis,
        indexing::{NewAxis, TryIndexOp},
        mean_axis, quantized_matmul, r#where, rsqrt, tanh,
    },
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

pub use super::common::sample;
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
            self, attention_probabilities, batch_seq, finish_attention,
            reshape_attention_projection, CausalLm,
        },
        input,
    },
    quantization::AffineQuantization,
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    weights::{
        load_safetensors_quantized_strict, load_safetensors_strict, StrictLoadConfig,
        StrictLoadReport,
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
    static GELU_MUL_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
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

fn gelu_mul_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "gemma4_gelu_mul",
        ["fused"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % HIDDEN_DIM;",
            "uint row = elem / HIDDEN_DIM;",
            "uint base = row * HIDDEN_DIM * 2;",
            "float gate = float(fused[base + col]);",
            "float up = float(fused[base + HIDDEN_DIM + col]);",
            "float x3 = gate * gate * gate;",
            "float inner = 0.7978845608028654f * (gate + 0.044715f * x3);",
            "float gelu = 0.5f * gate * (1.0f + metal::tanh(inner));",
            "out[elem] = T(gelu * up);"
        ),
        "",
        true,
        false,
    )
}

fn gelu_mul_fused(fused: &Array, hidden_dim: i32, stream: &Stream) -> Result<Array, Exception> {
    let mut output_shape = fused.shape().to_vec();
    let Some(last) = output_shape.last_mut() else {
        return Err(Exception::custom(
            "fused Gemma 4 MLP output has no dimensions",
        ));
    };
    *last = hidden_dim;

    GELU_MUL_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(gelu_mul_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_dtype("T", fused.dtype())
            .with_template_arg_int("HIDDEN_DIM", hidden_dim)
            .with_grid([output_shape.iter().product(), 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg(output_shape, fused.dtype());
        cell.borrow()
            .as_ref()
            .expect("Gemma 4 GELU/mul kernel initialized")
            .apply_one_device([fused], &config, stream)
    })
}

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
    #[serde(default)]
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
    fn affine_quantization(&self) -> Option<AffineQuantization> {
        self.quantized.then_some(AffineQuantization {
            group_size: self.quantization_group_size,
            bits: self.quantization_bits,
            mode: crate::quantization::AffineQuantizationMode::Affine,
        })
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
        })
        .unwrap_or(1.0);
    ((head_dim as f32 * partial_factor).round() as i32).clamp(2, head_dim)
}

pub(super) fn maybe_quantized_linear(
    quantized: bool,
    input_dims: i32,
    output_dims: i32,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    maybe_quantized_linear_with_bias(
        quantized,
        input_dims,
        output_dims,
        group_size,
        bits,
        false,
        stream,
    )
}

pub(super) fn maybe_quantized_linear_with_bias(
    quantized: bool,
    input_dims: i32,
    output_dims: i32,
    group_size: i32,
    bits: i32,
    bias: bool,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    if quantized {
        Ok(MaybeQuantized::Quantized(nn::QuantizedLinear::unloaded(
            input_dims,
            output_dims,
            group_size,
            bits,
            bias,
            stream,
        )?))
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

        let q_proj = maybe_quantized_linear(
            args.quantized,
            dim,
            n_heads * head_dim,
            args.quantization_group_size,
            args.quantization_bits,
            stream,
        )?;
        let k_proj = if is_kv_shared_layer {
            None
        } else {
            Some(maybe_quantized_linear(
                args.quantized,
                dim,
                n_kv_heads * head_dim,
                args.quantization_group_size,
                args.quantization_bits,
                stream,
            )?)
        };
        let v_proj = if is_kv_shared_layer || attention_k_eq_v {
            None
        } else {
            Some(maybe_quantized_linear(
                args.quantized,
                dim,
                n_kv_heads * head_dim,
                args.quantization_group_size,
                args.quantization_bits,
                stream,
            )?)
        };
        let o_proj = maybe_quantized_linear(
            args.quantized,
            n_heads * head_dim,
            dim,
            args.quantization_group_size,
            args.quantization_bits,
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
            if self.store_full_length_kv {
                if let Some(shared_kv) = shared_kv.as_mut() {
                    shared_kv.insert(self.layer_type, (keys.clone(), values.clone()));
                }
            }
            (keys, values)
        };

        let attention_cache = if self.is_kv_shared_layer || shared_kv.is_some() {
            None
        } else {
            cache
        };
        let output = if attention_cache.is_none()
            && mask.is_some()
            && L > 1
            && self.layer_type == LayerType::SlidingAttention
            && generated_sliding_window.is_some()
            && keys.shape()[2] == position_offset + L
            && (position_offset + L
                <= generated_sliding_window.expect("checked generated sliding window") + 1
                || L >= 1024)
        {
            sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                generated_sliding_window.expect("checked generated sliding window"),
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
            if self.store_full_length_kv {
                if let Some(shared_kv) = shared_kv.as_mut() {
                    shared_kv.insert(self.layer_type, (keys.clone(), values.clone()));
                    observer.observe(&format!("{prefix}.shared_keys_stored"), &keys)?;
                    observer.observe(&format!("{prefix}.shared_values_stored"), &values)?;
                }
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
        let output = if attention_cache.is_none()
            && mask.is_some()
            && seq_len > 1
            && self.layer_type == LayerType::SlidingAttention
            && generated_sliding_window.is_some()
            && keys.shape()[2] == position_offset + seq_len
            && (position_offset + seq_len
                <= generated_sliding_window.expect("checked generated sliding window") + 1
                || seq_len >= 1024)
        {
            sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                generated_sliding_window.expect("checked generated sliding window"),
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
    /// Lazily fused quantized gate/up projection.
    pub fused_gate_up: Option<FusedQuantizedGateUp>,
}

#[derive(Debug, Clone)]
/// Packed gate/up projection used by quantized Gemma 4 MLPs.
pub struct FusedQuantizedGateUp {
    /// Concatenated packed gate/up weights.
    pub weight: Array,
    /// Concatenated quantization scales.
    pub scales: Array,
    /// Concatenated quantization biases.
    pub biases: Array,
    /// Quantization group size.
    pub group_size: i32,
    /// Quantization bit width.
    pub bits: i32,
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
        Ok(Self {
            hidden_dim,
            gate_proj: maybe_quantized_linear(
                quantized, dim, hidden_dim, group_size, bits, stream,
            )?,
            down_proj: maybe_quantized_linear(
                quantized, hidden_dim, dim, group_size, bits, stream,
            )?,
            up_proj: maybe_quantized_linear(quantized, dim, hidden_dim, group_size, bits, stream)?,
            fused_gate_up: None,
        })
    }

    fn ensure_fused_gate_up(
        &mut self,
        stream: &Stream,
    ) -> Result<Option<&FusedQuantizedGateUp>, Exception> {
        if self.fused_gate_up.is_none() {
            let (MaybeQuantized::Quantized(gate), MaybeQuantized::Quantized(up)) =
                (&self.gate_proj, &self.up_proj)
            else {
                return Ok(None);
            };
            if gate.group_size != up.group_size
                || gate.bits != up.bits
                || gate.inner.bias.as_ref().is_some()
                || up.inner.bias.as_ref().is_some()
            {
                return Ok(None);
            }

            self.fused_gate_up = Some(FusedQuantizedGateUp {
                weight: concatenate_axis(
                    &[gate.inner.weight.as_ref(), up.inner.weight.as_ref()],
                    0,
                    stream,
                )?,
                scales: concatenate_axis(&[gate.scales.as_ref(), up.scales.as_ref()], 0, stream)?,
                biases: concatenate_axis(&[gate.biases.as_ref(), up.biases.as_ref()], 0, stream)?,
                group_size: gate.group_size,
                bits: gate.bits,
            });
        }
        Ok(self.fused_gate_up.as_ref())
    }

    fn fused_gate_up_forward(
        &mut self,
        input: &Array,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        let Some(fused) = self.ensure_fused_gate_up(stream)? else {
            return Ok(None);
        };
        let output = quantized_matmul(
            input,
            &fused.weight,
            &fused.scales,
            Some(&fused.biases),
            true,
            fused.group_size,
            fused.bits,
            stream,
        )?;
        Ok(Some(gelu_mul_fused(&output, self.hidden_dim, stream)?))
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let down_proj_input = match self.fused_gate_up_forward(input, stream)? {
            Some(down_proj_input) => down_proj_input,
            None => {
                let gate = self.gate_proj.forward(input, stream)?;
                let up = self.up_proj.forward(input, stream)?;
                nn::gelu_approximate(gate, stream)?.multiply(up, stream)?
            }
        };
        self.down_proj.forward(&down_proj_input, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        if mode {
            self.fused_gate_up = None;
        }
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

#[derive(Debug, Clone, ModuleParameters)]
/// Gemma 4 embedding table with optional packed quantized storage.
pub struct Gemma4Embedding {
    #[param]
    /// Embedding weight tensor.
    pub weight: Param<Array>,
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
}

impl Gemma4Embedding {
    /// Creates an unloaded embedding table.
    pub fn unloaded(
        vocab_size: i32,
        hidden_size: i32,
        quantized: bool,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let packed_per_int = 32 / bits;
        Ok(Self {
            weight: if quantized {
                Param::<Array>::unloaded(
                    &[vocab_size, hidden_size / packed_per_int],
                    Dtype::Uint32,
                    stream,
                )?
            } else {
                Param::<Array>::unloaded(&[vocab_size, hidden_size], Dtype::Float32, stream)?
            },
            scales: if quantized {
                Param::<Option<Array>>::unloaded_some(
                    &[vocab_size, hidden_size / group_size],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            biases: if quantized {
                Param::<Option<Array>>::unloaded_some(
                    &[vocab_size, hidden_size / group_size],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            quantized,
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
            weight: Param::new(if quantized {
                Array::from_slice(
                    &vec![0u32; (vocab_size * (hidden_size / (32 / bits))) as usize],
                    &[vocab_size, hidden_size / (32 / bits)],
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
            hidden_size,
            group_size,
            bits,
        })
    }

    /// Embeds token ids.
    pub fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
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
            .expect("quantized embedding biases")
            .try_index_device(&flat, stream)?;
        let out = safemlx::ops::dequantize(
            &weight,
            &scales,
            &biases,
            self.group_size,
            self.bits,
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
        if self.quantized {
            let scales = self
                .scales
                .as_ref()
                .as_ref()
                .expect("quantized embedding scales");
            let biases = self
                .biases
                .as_ref()
                .as_ref()
                .expect("quantized embedding biases");
            return quantized_matmul(
                x,
                &self.weight,
                scales,
                Some(biases),
                true,
                self.group_size,
                self.bits,
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
        let mlp = Mlp::new(
            args.hidden_size,
            args.intermediate_size,
            args.quantized,
            args.quantization_group_size,
            args.quantization_bits,
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
        let per_layer_input_gate = if args.hidden_size_per_layer_input > 0 {
            Some(maybe_quantized_linear(
                args.quantized,
                args.hidden_size,
                args.hidden_size_per_layer_input,
                args.quantization_group_size,
                args.quantization_bits,
                stream,
            )?)
        } else {
            None
        };
        let per_layer_projection = if args.hidden_size_per_layer_input > 0 {
            Some(maybe_quantized_linear(
                args.quantized,
                args.hidden_size_per_layer_input,
                args.hidden_size,
                args.quantization_group_size,
                args.quantization_bits,
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
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
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
            if seq_len > 1 || self.sliding_window.is_some() {
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
        let r =
            self.mlp
                .forward_with_observer(&pre_ff, stream, &format!("{prefix}.mlp"), observer)?;
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
            if seq_len > 1 || self.sliding_window.is_some() {
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
        let r = self.mlp.forward(&pre_ff, stream)?;
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
            args.quantized,
            args.quantization_group_size,
            args.quantization_bits,
            stream,
        )?;
        let embed_tokens_per_layer = if args.hidden_size_per_layer_input > 0 {
            Some(Gemma4Embedding::unloaded(
                args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                args.quantized,
                args.quantization_group_size,
                args.quantization_bits,
                stream,
            )?)
        } else {
            None
        };
        let per_layer_model_projection = if args.hidden_size_per_layer_input > 0 {
            Some(maybe_quantized_linear(
                args.quantized,
                args.hidden_size,
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                args.quantization_group_size,
                args.quantization_bits,
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

    fn new_with_modalities(
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
                    (
                        args.quantized,
                        args.quantization_group_size,
                        args.quantization_bits,
                    ),
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
                    true,
                    (
                        args.quantized,
                        args.quantization_group_size,
                        args.quantization_bits,
                    ),
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
                cache.kv.clear();
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
                cache.kv.clear();
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

    fn forward_logits<C>(
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
            Some(
                common::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.affine_quantization(),
                    stream,
                )?,
            )
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
        })
    }

    fn new_with_modalities(
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
            Some(
                common::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.affine_quantization(),
                    stream,
                )?,
            )
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

type Gemma4ModelConfigParts = (
    ModelArgs,
    Option<Gemma4VisionConfig>,
    Option<i32>,
    Option<i32>,
    Option<Gemma4AudioConfig>,
    Option<i32>,
);

fn get_gemma4_model_config(model_dir: &Path) -> Result<Gemma4ModelConfigParts, Error> {
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    let mut config: Gemma4Config = serde_json::from_reader(file)?;
    if config.text_config.enable_moe_block {
        return Err(Error::UnsupportedArchitecture(
            "Gemma 4 MoE models are not supported yet".to_string(),
        ));
    }
    config.text_config.model_type = "gemma4".to_string();
    config.text_config.quantized = config.quantization.is_some();
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
    if config.text_config.enable_moe_block {
        return Err(Error::UnsupportedArchitecture(
            "Gemma 4 MoE models are not supported yet".to_string(),
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

/// Loads a dense Gemma 4 checkpoint while affine-quantizing supported weights.
///
/// Transformer weights and modality bridge projections use affine storage.
/// Vision and audio towers remain dense because their convolutional and
/// specialized implementations do not expose MLX affine parameter layouts. A
/// checkpoint already carrying matching affine metadata is loaded directly
/// without requantization.
pub fn load_gemma4_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: AffineQuantization,
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
        model_args.affine_quantization(),
        quantization,
    )? {
        return load_gemma4_model(model_dir, stream, weights_stream);
    }
    model_args.quantized = true;
    model_args.quantization_group_size = quantization.group_size;
    model_args.quantization_bits = quantization.bits;
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
    quantization: Option<AffineQuantization>,
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
    pub fn forward_with_state(
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

    /// Rolls back speculative tokens that were rejected by target-model verification.
    pub fn rollback_speculative_cache(
        &mut self,
        cache: &mut [Option<ConcatKeyValueCache>],
        accepted: usize,
        block_size: usize,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let rejected = block_size.saturating_sub(accepted + 1) as i32;
        if rejected == 0 {
            return Ok(());
        }
        for cache in cache.iter_mut().flatten() {
            let new_len = cache.offset().saturating_sub(rejected);
            cache.truncate(new_len, stream)?;
        }
        Ok(())
    }
}

/// Output for a Gemma 4 target-model step used by assistant drafting.
pub struct Gemma4StepOutput {
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
    Array::from(token_ids)
        .try_index_device(NewAxis, stream)
        .map_err(Into::into)
}

struct Gemma4AttentionMasks {
    full: Array,
    sliding: Array,
}

fn multimodal_attention_masks(
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
                cache.kv.clear();
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
                cache.kv.clear();
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
        cache.kv.clear();
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
pub type Generate<'a, S = crate::sampler::DefaultSampler> = common::Generate<'a, Model, Cache, S>;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{module::ModuleParameters, Array, Device, DeviceType, ExecutionContext, Stream};

    use super::{
        load_gemma4_model, partial_rotary_dims, Attention, Cache, FloatOrString, LayerType,
        ModelArgs,
    };
    use crate::models::{
        common::CausalLm,
        input::{InputMetadata, InputPart, ModelInput},
    };

    fn test_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn model_args(attention_k_eq_v: bool) -> ModelArgs {
        ModelArgs {
            model_type: "gemma4_unified_text".to_string(),
            hidden_size: 8,
            num_hidden_layers: 1,
            intermediate_size: 16,
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
