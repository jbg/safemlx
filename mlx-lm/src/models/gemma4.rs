use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        indexing::{IndexOp, NewAxis},
        mean_axis, rsqrt, tanh,
    },
    quantization::{MaybeQuantized, Quantizable as _},
    Array,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    weights::{load_safetensors_strict, StrictLoadConfig, StrictLoadReport},
};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub head_dim: i32,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(skip)]
    pub quantized: bool,
    #[serde(skip)]
    pub quantization_group_size: i32,
    #[serde(skip)]
    pub quantization_bits: i32,
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub num_kv_shared_layers: i32,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<i32>,
    #[serde(default)]
    pub top_k_experts: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
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
pub enum LayerType {
    SlidingAttention,
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

    fn layer_type(&self, index: usize) -> LayerType {
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
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    let linear = nn::LinearBuilder::new(input_dims, output_dims)
        .bias(false)
        .build()?;
    if quantized {
        Ok(MaybeQuantized::Quantized(
            linear.try_into_quantized(group_size, bits)?,
        ))
    } else {
        Ok(MaybeQuantized::Original(linear))
    }
}

fn rms_norm_without_scale(x: &Array, eps: f32) -> Result<Array, Exception> {
    let variance = mean_axis(&x.square()?, -1, true)?;
    x.multiply(rsqrt(variance.add(Array::from_f32(eps))?)?)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,
    pub attention_k_eq_v: bool,
    pub layer_type: LayerType,
    pub is_kv_shared_layer: bool,
    pub store_full_length_kv: bool,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(
        args: &ModelArgs,
        layer_type: LayerType,
        layer_idx: usize,
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
        )?;
        let k_proj = maybe_quantized_linear(
            args.quantized,
            dim,
            n_kv_heads * head_dim,
            args.quantization_group_size,
            args.quantization_bits,
        )?;
        let v_proj_output_dims = if attention_k_eq_v {
            dim
        } else {
            n_kv_heads * head_dim
        };
        let v_proj = maybe_quantized_linear(
            args.quantized,
            dim,
            v_proj_output_dims,
            args.quantization_group_size,
            args.quantization_bits,
        )?;
        let o_proj = maybe_quantized_linear(
            args.quantized,
            n_heads * head_dim,
            dim,
            args.quantization_group_size,
            args.quantization_bits,
        )?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        let rope_dims = partial_rotary_dims(head_dim, &args.rope_scaling);
        let rope = initialize_rope(
            rope_dims,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
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

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
    pub position_offset: i32,
    pub per_layer_input: Option<&'a Array>,
    pub shared_kv: Option<&'a mut HashMap<LayerType, (Array, Array)>>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            mut cache,
            position_offset,
            mut shared_kv,
            ..
        } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let mut queries = self.q_norm.forward(
            &queries
                .reshape(&[B, L, self.n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let offset = position_offset;
        queries = self
            .rope
            .forward(nn::RopeInputBuilder::new(&queries).offset(offset).build()?)?;

        let (keys, values) = if self.is_kv_shared_layer {
            shared_kv
                .as_ref()
                .and_then(|shared_kv| shared_kv.get(&self.layer_type))
                .cloned()
                .ok_or_else(|| Exception::custom("missing shared Gemma 4 KV states"))?
        } else {
            let keys = self.k_proj.forward(x)?;
            let values = if self.attention_k_eq_v {
                keys.clone()
            } else {
                self.v_proj.forward(x)?
            };
            let mut keys = self.k_norm.forward(
                &keys
                    .reshape(&[B, L, self.n_kv_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
            let mut values =
                rms_norm_without_scale(&values.reshape(&[B, L, self.n_kv_heads, -1])?, 1e-6)?
                    .transpose_axes(&[0, 2, 1, 3])?;
            keys = self
                .rope
                .forward(nn::RopeInputBuilder::new(&keys).offset(offset).build()?)?;
            if let Some(cache) = cache.as_mut() {
                (keys, values) = cache.update_and_fetch(keys, values)?;
            }
            if self.store_full_length_kv {
                if let Some(shared_kv) = shared_kv.as_mut() {
                    shared_kv.insert(self.layer_type, (keys.clone(), values.clone()));
                }
            }
            (keys, values)
        };

        let attention_cache = if self.is_kv_shared_layer { None } else { cache };
        let output = crate::utils::scaled_dot_product_attention(
            queries,
            keys,
            values,
            attention_cache,
            self.scale,
            mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(
        dim: i32,
        hidden_dim: i32,
        quantized: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: maybe_quantized_linear(quantized, dim, hidden_dim, group_size, bits)?,
            down_proj: maybe_quantized_linear(quantized, hidden_dim, dim, group_size, bits)?,
            up_proj: maybe_quantized_linear(quantized, dim, hidden_dim, group_size, bits)?,
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let down_proj_input = nn::gelu_approximate(self.gate_proj.forward(input)?)?
            .multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&down_proj_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Gemma4Embedding {
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Option<Array>>,
    #[param]
    pub biases: Param<Option<Array>>,
    pub quantized: bool,
    pub hidden_size: i32,
    pub group_size: i32,
    pub bits: i32,
}

impl Gemma4Embedding {
    pub fn new(
        vocab_size: i32,
        hidden_size: i32,
        quantized: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(if quantized {
                Array::zeros::<u32>(&[vocab_size, hidden_size / (32 / bits)])?
            } else {
                nn::Embedding::new(vocab_size, hidden_size)?.weight.value
            }),
            scales: Param::new(if quantized {
                Some(Array::ones::<f32>(&[vocab_size, hidden_size / group_size])?)
            } else {
                None
            }),
            biases: Param::new(if quantized {
                Some(Array::zeros::<f32>(&[
                    vocab_size,
                    hidden_size / group_size,
                ])?)
            } else {
                None
            }),
            quantized,
            hidden_size,
            group_size,
            bits,
        })
    }

    pub fn forward(&mut self, input: &Array) -> Result<Array, Exception> {
        if !self.quantized {
            return Ok(self.weight.index(input));
        }
        let original_shape = input.shape().to_vec();
        let flat = input.flatten(None, None)?;
        let weight = self.weight.index(&flat);
        let scales = self
            .scales
            .as_ref()
            .as_ref()
            .expect("quantized embedding scales")
            .index(&flat);
        let biases = self
            .biases
            .as_ref()
            .as_ref()
            .expect("quantized embedding biases")
            .index(&flat);
        let out = mlx_rs::ops::dequantize(&weight, &scales, &biases, self.group_size, self.bits)?;
        let shape = original_shape
            .into_iter()
            .chain(std::iter::once(self.hidden_size))
            .collect::<Vec<_>>();
        out.reshape(&shape)
    }

    pub fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
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
            mlx_rs::ops::dequantize(&self.weight, scales, biases, self.group_size, self.bits)?
        } else {
            self.weight.as_ref().clone()
        };
        mlx_rs::ops::matmul(x, weight.t())
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,
    pub layer_type: LayerType,
    pub sliding_window: Option<i32>,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[quantizable]
    #[param]
    pub per_layer_input_gate: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub post_per_layer_input_norm: Option<nn::RmsNorm>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub layer_scalar: Param<Array>,
}

impl TransformerBlock {
    pub fn new(
        args: &ModelArgs,
        layer_type: LayerType,
        layer_idx: usize,
    ) -> Result<Self, Exception> {
        let layer_args = args.for_layer(layer_type);
        let self_attn = Attention::new(&layer_args, layer_type, layer_idx)?;
        let mlp = Mlp::new(
            args.hidden_size,
            args.intermediate_size,
            args.quantized,
            args.quantization_group_size,
            args.quantization_bits,
        )?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let pre_feedforward_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_feedforward_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let per_layer_input_gate = if args.hidden_size_per_layer_input > 0 {
            Some(maybe_quantized_linear(
                args.quantized,
                args.hidden_size,
                args.hidden_size_per_layer_input,
                args.quantization_group_size,
                args.quantization_bits,
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
            )?)
        } else {
            None
        };
        let post_per_layer_input_norm = if args.hidden_size_per_layer_input > 0 {
            Some(
                nn::RmsNormBuilder::new(args.hidden_size)
                    .eps(args.rms_norm_eps)
                    .build()?,
            )
        } else {
            None
        };
        Ok(Self {
            num_attention_heads: layer_args.num_attention_heads,
            hidden_size: layer_args.hidden_size,
            layer_type,
            sliding_window: args.sliding_window,
            layer_scalar: Param::new(Array::ones::<f32>(&[1])?),
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
    fn apply_layer_scalar(&self, x: Array) -> Result<Array, Exception> {
        x.multiply(&*self.layer_scalar)
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            cache,
            position_offset,
            per_layer_input,
            shared_kv,
        } = input;
        let generated_mask = if self.layer_type == LayerType::SlidingAttention {
            if x.shape()[1] > 1 || self.sliding_window.is_some() {
                Some(create_causal_mask(
                    x.shape()[1],
                    Some(position_offset),
                    self.sliding_window,
                    None,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let self_attn_input = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask: generated_mask.as_ref().or(mask),
            cache,
            position_offset,
            per_layer_input: None,
            shared_kv,
        };
        let r = self.self_attn.forward(self_attn_input)?;
        let r = self.post_attention_layernorm.forward(&r)?;
        let h = x.add(r)?;
        let r = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
        let r = self.post_feedforward_layernorm.forward(&r)?;
        let mut h = h.add(r)?;
        if let (Some(per_layer_input), Some(gate), Some(projection), Some(norm)) = (
            per_layer_input,
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
        ) {
            let residual = h.clone();
            let r = nn::gelu_approximate(gate.forward(&h)?)?.multiply(per_layer_input)?;
            let r = projection.forward(&r)?;
            let r = norm.forward(&r)?;
            h = residual.add(r)?;
        }
        self.apply_layer_scalar(h)
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
pub struct Gemma4TextModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    pub hidden_size: i32,
    pub hidden_size_per_layer_input: i32,
    #[param]
    pub embed_tokens: Gemma4Embedding,
    #[param]
    pub embed_tokens_per_layer: Option<Gemma4Embedding>,
    #[quantizable]
    #[param]
    pub per_layer_model_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub per_layer_projection_norm: Option<nn::RmsNorm>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Gemma4TextModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embed_tokens = Gemma4Embedding::new(
            args.vocab_size,
            args.hidden_size,
            args.quantized,
            args.quantization_group_size,
            args.quantization_bits,
        )?;
        let embed_tokens_per_layer = if args.hidden_size_per_layer_input > 0 {
            Some(Gemma4Embedding::new(
                args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                args.quantized,
                args.quantization_group_size,
                args.quantization_bits,
            )?)
        } else {
            None
        };
        let per_layer_model_projection = if args.hidden_size_per_layer_input > 0 {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(
                    args.hidden_size,
                    args.num_hidden_layers * args.hidden_size_per_layer_input,
                )
                .bias(false)
                .build()?,
            ))
        } else {
            None
        };
        let per_layer_projection_norm = if args.hidden_size_per_layer_input > 0 {
            Some(
                nn::RmsNormBuilder::new(args.hidden_size_per_layer_input)
                    .eps(args.rms_norm_eps)
                    .build()?,
            )
        } else {
            None
        };
        let layers = (0..args.num_hidden_layers)
            .map(|index| {
                TransformerBlock::new(args, args.layer_type(index as usize), index as usize)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
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
            .forward(input_ids)?
            .multiply(Array::from_f32((ple_dim as f32).sqrt()))?
            .reshape(&[
                input_ids.shape()[0],
                input_ids.shape()[1],
                self.num_hidden_layers,
                ple_dim,
            ])?;
        let projected = per_layer_model_projection
            .forward(inputs_embeds)?
            .multiply(Array::from_f32((self.hidden_size as f32).sqrt().recip()))?
            .reshape(&[
                inputs_embeds.shape()[0],
                inputs_embeds.shape()[1],
                self.num_hidden_layers,
                ple_dim,
            ])?;
        let projected = per_layer_projection_norm.forward(&projected)?;
        Ok(Some(
            projected
                .add(token_identity)?
                .multiply(Array::from_f32(2.0_f32.powf(-0.5)))?,
        ))
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Gemma4TextModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;
        let mut h = self
            .embed_tokens
            .forward(inputs)?
            .multiply(Array::from_f32((self.hidden_size as f32).sqrt()))?;
        let per_layer_inputs = self.per_layer_inputs(inputs, &h)?;
        let position_offset = cache
            .iter()
            .flatten()
            .map(KeyValueCache::offset)
            .max()
            .unwrap_or(0);
        let mut shared_kv = HashMap::new();
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None if h.shape()[1] > 1 => Some(create_causal_mask(h.shape()[1], None, None, None)?),
            None => None,
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        for (index, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_ple = per_layer_inputs
                .as_ref()
                .map(|inputs| inputs.index((.., .., index as i32, ..)));
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
                position_offset,
                per_layer_input: layer_ple.as_ref(),
                shared_kv: Some(&mut shared_kv),
            };
            h = layer.forward(layer_input)?;
        }
        self.norm.forward(&h)
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
pub struct Gemma4ForConditionalGeneration {
    #[quantizable]
    #[param]
    pub language_model: Gemma4TextModel,
}

impl Gemma4ForConditionalGeneration {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            language_model: Gemma4TextModel::new(args)?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: Gemma4ForConditionalGeneration,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Gemma4ForConditionalGeneration::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

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

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.language_model.forward(input)?;
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out)?,
            None => self.model.language_model.embed_tokens.as_linear(&out)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(&(logits.divide(Array::from_f32(softcap))?))?
                .multiply(Array::from_f32(softcap))?;
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

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_gemma4_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_gemma4_model_args(model_dir)?;
    let mut model = Model::new(model_args)?;
    let weights_index = model_dir.join("model.safetensors.index.json");
    let config = StrictLoadConfig::default()
        .rewrite_prefix("language_model.model.", "model.language_model.")
        .rewrite_prefix("model.language_model.", "model.language_model.")
        .allow_unused_prefix("audio_tower.")
        .allow_unused_prefix("embed_audio.")
        .allow_unused_prefix("embed_vision.")
        .allow_unused_prefix("multi_modal_projector.")
        .allow_unused_prefix("vision_tower.")
        .allow_missing_suffix(".bias");
    let mut report = StrictLoadReport::default();
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            load_safetensors_strict(&mut model, weights_filename, &config, &mut report)?;
        }
    } else {
        load_safetensors_strict(
            &mut model,
            model_dir.join("model.safetensors"),
            &config,
            &mut report,
        )?;
    }
    report.finish(&model, &config)?;
    Ok(model)
}

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let prompt_len = prompt_token.shape()[1];
                if prompt_len > 1 {
                    let prefix = prompt_token.index((.., ..prompt_len - 1));
                    let input = ModelInput {
                        inputs: &prefix,
                        mask: None,
                        cache: self.cache,
                    };
                    tri!(self.model.forward(input));
                }
                let last = prompt_token.index((.., prompt_len - 1..));
                let input = ModelInput {
                    inputs: &last,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}
