//! Gemma 4 text model implementation and loader.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{indexing::TryIndexOp, mean_axis, rsqrt, tanh},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

pub use super::common::sample;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    models::common::{self, batch_seq, finish_attention, reshape_attention_projection, CausalLm},
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    weights::{load_safetensors_strict, StrictLoadConfig, StrictLoadReport},
};

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
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    quantization: Option<Value>,
}

impl ModelArgs {
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

fn maybe_quantized_linear(
    quantized: bool,
    input_dims: i32,
    output_dims: i32,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    if quantized {
        Ok(MaybeQuantized::Quantized(nn::QuantizedLinear::unloaded(
            input_dims,
            output_dims,
            group_size,
            bits,
            false,
            stream,
        )?))
    } else {
        Ok(MaybeQuantized::Original(nn::Linear::unloaded(
            input_dims,
            output_dims,
            false,
            Dtype::Float32,
            stream,
        )?))
    }
}

fn rms_norm_without_scale(x: &Array, eps: f32, stream: &Stream) -> Result<Array, Exception> {
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
    /// Key projection.
    pub k_proj: MaybeQuantized<nn::Linear>,
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
    /// Key normalization.
    pub k_norm: nn::RmsNorm,
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
        let k_proj = maybe_quantized_linear(
            args.quantized,
            dim,
            n_kv_heads * head_dim,
            args.quantization_group_size,
            args.quantization_bits,
            stream,
        )?;
        let v_proj = if attention_k_eq_v {
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
        let k_norm = nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?;

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
            let keys = self.k_proj.forward(x, stream)?;
            let values = if self.attention_k_eq_v {
                keys.clone()
            } else {
                self.v_proj
                    .as_mut()
                    .ok_or_else(|| Exception::custom("missing Gemma 4 value projection"))?
                    .forward(x, stream)?
            };
            let mut keys = self.k_norm.forward(
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
        let output = finish_attention(
            queries,
            keys,
            values,
            attention_cache,
            self.scale,
            mask,
            B,
            L,
            stream,
        )?;

        self.o_proj.forward(&output, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        if let Some(v_proj) = &mut self.v_proj {
            v_proj.training_mode(mode);
        }
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 feed-forward layer.
pub struct Mlp {
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
        Ok(Self {
            gate_proj: maybe_quantized_linear(
                quantized, dim, hidden_dim, group_size, bits, stream,
            )?,
            down_proj: maybe_quantized_linear(
                quantized, hidden_dim, dim, group_size, bits, stream,
            )?,
            up_proj: maybe_quantized_linear(quantized, dim, hidden_dim, group_size, bits, stream)?,
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let down_proj_input = nn::gelu_approximate(self.gate_proj.forward(input, stream)?, stream)?
            .multiply(self.up_proj.forward(input, stream)?, stream)?;
        self.down_proj.forward(&down_proj_input, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
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
        let weight = if self.quantized {
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
            safemlx::ops::dequantize(
                &self.weight,
                scales,
                biases,
                self.group_size,
                self.bits,
                stream,
            )?
        } else {
            self.weight.as_ref().clone()
        };
        safemlx::ops::matmul(x, weight.transpose(stream)?, stream)
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
        } = input;
        let generated_mask = if disable_generated_mask {
            None
        } else if self.layer_type == LayerType::SlidingAttention {
            if x.shape()[1] > 1 || self.sliding_window.is_some() {
                Some(create_causal_mask(
                    x.shape()[1],
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
        let normed = self.input_layernorm.forward(x, stream)?;
        let self_attn_input = AttentionInput {
            x: &normed,
            mask: generated_mask.as_ref().or(mask),
            cache,
            position_offset,
            per_layer_input: None,
            shared_kv,
            disable_generated_mask,
        };
        let r = self.self_attn.forward(self_attn_input, stream)?;
        let r = self.post_attention_layernorm.forward(&r, stream)?;
        let h = x.add(r, stream)?;
        let pre_ff = self.pre_feedforward_layernorm.forward(&h, stream)?;
        let r = self.mlp.forward(&pre_ff, stream)?;
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
            Some(MaybeQuantized::Original(nn::Linear::unloaded(
                args.hidden_size,
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                false,
                Dtype::Float32,
                stream,
            )?))
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
}

/// Input for a Gemma 4 text forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
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
            mask,
            cache,
        } = input;
        let mut h = self
            .embed_tokens
            .forward(inputs, stream)?
            .multiply(Array::from_f32((self.hidden_size as f32).sqrt()), stream)?;
        let per_layer_inputs = self.per_layer_inputs(inputs, &h, stream)?;
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
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
                position_offset,
                per_layer_input: layer_ple.as_ref(),
                shared_kv: Some(&mut shared_kv),
                disable_generated_mask: false,
            };
            h = layer.forward(layer_input, stream)?;
        }
        let pre_norm_hidden = h.clone();
        let hidden = self.norm.forward(&h, stream)?;
        Ok(Gemma4TextOutput {
            hidden,
            pre_norm_hidden,
            shared_kv_states: shared_kv,
        })
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
}

impl Gemma4ForConditionalGeneration {
    /// Creates an unloaded conditional-generation wrapper.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            language_model: Gemma4TextModel::new(args, stream)?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Gemma 4 causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
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
    /// Creates an unloaded Gemma 4 causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let model = Gemma4ForConditionalGeneration::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(common::build_unloaded_maybe_quantized_lm_head(
                args.hidden_size,
                args.vocab_size,
                stream,
            )?)
        } else {
            None
        };
        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
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
        let out = self.model.language_model.forward(input, stream)?;
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out, stream)?,
            None => self
                .model
                .language_model
                .embed_tokens
                .as_linear(&out, stream)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(&(logits.divide(Array::from_f32(softcap), stream)?), stream)?
                .multiply(Array::from_f32(softcap), stream)?;
        }
        Ok(logits)
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
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
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
    Ok(config.text_config)
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
    let model_args = get_gemma4_model_args(model_dir)?;
    let mut model = Model::new(model_args, stream)?;
    let weights_index = model_dir.join("model.safetensors.index.json");
    let config = StrictLoadConfig::default()
        .rewrite_prefix("language_model.model.", "model.language_model.")
        .rewrite_prefix("model.language_model.", "model.language_model.")
        .allow_unused_prefix("audio_tower.")
        .allow_unused_prefix("embed_audio.")
        .allow_unused_prefix("embed_vision.")
        .allow_unused_prefix("multi_modal_projector.")
        .allow_unused_prefix("vision_tower.")
        .allow_unused_prefix("model.audio_tower.")
        .allow_unused_prefix("model.embed_audio.")
        .allow_unused_prefix("model.embed_vision.")
        .allow_unused_prefix("model.multi_modal_projector.")
        .allow_unused_prefix("model.vision_embedder.")
        .allow_unused_prefix("model.vision_tower.")
        .allow_missing_suffix(".bias");
    let mut report = StrictLoadReport::default();
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            load_safetensors_strict(
                &mut model,
                weights_filename,
                weights_stream,
                &config,
                &mut report,
            )?;
        }
    } else {
        load_safetensors_strict(
            &mut model,
            model_dir.join("model.safetensors"),
            weights_stream,
            &config,
            &mut report,
        )?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
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
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&text_output.hidden, stream)?,
            None => self
                .model
                .language_model
                .embed_tokens
                .as_linear(&text_output.hidden, stream)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(&(logits.divide(Array::from_f32(softcap), stream)?), stream)?
                .multiply(Array::from_f32(softcap), stream)?;
        }
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
    fn prefill_logits(
        &mut self,
        prompt_tokens: &Array,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prompt_len = prompt_tokens.shape()[1];
        if prompt_len > 1 {
            let prefix = prompt_tokens.try_index_device((.., ..prompt_len - 1), stream)?;
            self.forward(
                ModelInput {
                    inputs: &prefix,
                    mask: None,
                    cache,
                },
                stream,
            )?;
        }
        let last = prompt_tokens.try_index_device((.., prompt_len - 1..), stream)?;
        let logits = self.forward(
            ModelInput {
                inputs: &last,
                mask: None,
                cache,
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.forward(
            ModelInput {
                inputs: input_tokens,
                mask: None,
                cache,
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }
}

/// Gemma 4 token generation iterator.
pub type Generate<'a, C> = common::Generate<'a, Model, Vec<Option<C>>>;

#[cfg(test)]
mod tests {
    use safemlx::{module::ModuleParameters, Device, DeviceType, Stream};

    use super::{Attention, LayerType, ModelArgs};

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
