use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::Path,
    time::Instant,
};

use safemlx::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    fast::{MetalKernel, MetalKernelConfig},
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        argpartition_axis, broadcast_to, concatenate_axis, conv1d, exp, gather_grouped_rows,
        gather_route_values, grouped_matmul,
        indexing::{take_along_axis, IndexOp, NewAxis},
        matmul, segment_sum_by_index, sigmoid, softmax_axis, sum_axis, topk_route_plan, zeros,
    },
    transforms::eval,
    Array, Dtype,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
    weights::{load_safetensors_strict, StrictLoadConfig, StrictLoadReport},
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

impl<'de> Deserialize<'de> for LayerType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "linear_attention" => Ok(Self::LinearAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(serde::de::Error::custom(format!(
                "Unsupported Qwen3.5-MoE layer type '{other}'"
            ))),
        }
    }
}

fn silu(x: Array) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x)?)
}

thread_local! {
    static RECURRENT_DECODE_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

const ROUTED_EXPERT_CHUNK_THRESHOLD: i32 = 64;
const ROUTED_EXPERT_CHUNK_TOKENS: i32 = 32;

#[derive(Debug, Clone, Default)]
pub struct PerfStats {
    pub embed_s: f64,
    pub full_attention_s: f64,
    pub linear_attention_s: f64,
    pub moe_router_s: f64,
    pub moe_shared_s: f64,
    pub moe_routed_s: f64,
    pub moe_combine_s: f64,
    pub final_norm_s: f64,
    pub lm_head_s: f64,
    pub prefill_state_dependency_s: f64,
}

impl PerfStats {
    pub fn component_total_s(&self) -> f64 {
        self.embed_s
            + self.full_attention_s
            + self.linear_attention_s
            + self.moe_router_s
            + self.moe_shared_s
            + self.moe_routed_s
            + self.moe_combine_s
            + self.final_norm_s
            + self.lm_head_s
            + self.prefill_state_dependency_s
    }

    fn add(&mut self, component: PerfComponent, elapsed_s: f64) {
        match component {
            PerfComponent::Embed => self.embed_s += elapsed_s,
            PerfComponent::FullAttention => self.full_attention_s += elapsed_s,
            PerfComponent::LinearAttention => self.linear_attention_s += elapsed_s,
            PerfComponent::MoeRouter => self.moe_router_s += elapsed_s,
            PerfComponent::MoeShared => self.moe_shared_s += elapsed_s,
            PerfComponent::MoeRouted => self.moe_routed_s += elapsed_s,
            PerfComponent::MoeCombine => self.moe_combine_s += elapsed_s,
            PerfComponent::FinalNorm => self.final_norm_s += elapsed_s,
            PerfComponent::LmHead => self.lm_head_s += elapsed_s,
            PerfComponent::PrefillStateDependency => {
                self.prefill_state_dependency_s += elapsed_s;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PerfComponent {
    Embed,
    FullAttention,
    LinearAttention,
    MoeRouter,
    MoeShared,
    MoeRouted,
    MoeCombine,
    FinalNorm,
    LmHead,
    PrefillStateDependency,
}

thread_local! {
    static PERF_STATS: RefCell<Option<PerfStats>> = const { RefCell::new(None) };
}

pub fn set_perf_profiling(enabled: bool) {
    PERF_STATS.with(|stats| {
        *stats.borrow_mut() = enabled.then(PerfStats::default);
    });
}

pub fn reset_perf_stats() {
    PERF_STATS.with(|stats| {
        if let Some(stats) = stats.borrow_mut().as_mut() {
            *stats = PerfStats::default();
        }
    });
}

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

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_text_model_type")]
    pub model_type: String,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: i32,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: i32,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: i32,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: i32,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: i32,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: i32,
    #[serde(default = "default_shared_expert_intermediate_size")]
    pub shared_expert_intermediate_size: i32,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    #[serde(default = "default_num_experts")]
    pub num_experts: i32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Deserialize)]
struct TopLevelConfig {
    model_type: String,
    #[serde(default)]
    text_config: Option<ModelArgs>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    image_token_id: Option<i32>,
    #[serde(default)]
    video_token_id: Option<i32>,
}

fn default_true() -> bool {
    true
}

fn default_text_model_type() -> String {
    "qwen3_5_moe_text".to_string()
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_head_dim() -> i32 {
    256
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_linear_conv_kernel_dim() -> i32 {
    4
}

fn default_linear_key_head_dim() -> i32 {
    128
}

fn default_linear_value_head_dim() -> i32 {
    128
}

fn default_linear_num_key_heads() -> i32 {
    16
}

fn default_linear_num_value_heads() -> i32 {
    32
}

fn default_moe_intermediate_size() -> i32 {
    512
}

fn default_shared_expert_intermediate_size() -> i32 {
    512
}

fn default_num_experts_per_tok() -> i32 {
    8
}

fn default_num_experts() -> i32 {
    256
}

fn float_config_value(config: &Option<HashMap<String, Value>>, key: &str) -> Option<f32> {
    config.as_ref().and_then(|config| {
        config.get(key).and_then(|value| match value {
            Value::Number(v) => v.as_f64().map(|v| v as f32),
            Value::String(s) => s.parse().ok(),
            _ => None,
        })
    })
}

fn string_config_value<'a>(
    config: &'a Option<HashMap<String, Value>>,
    key: &str,
) -> Option<&'a str> {
    config.as_ref().and_then(|config| {
        config.get(key).and_then(|value| match value {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        })
    })
}

fn rope_config_value(
    config: Option<HashMap<String, Value>>,
) -> Option<HashMap<String, FloatOrString>> {
    config.map(|config| {
        config
            .into_iter()
            .filter_map(|(key, value)| {
                let value = match value {
                    Value::Number(v) => v.as_f64().map(|v| FloatOrString::Float(v as f32)),
                    Value::String(s) => Some(FloatOrString::String(s)),
                    _ => None,
                }?;
                Some((key, value))
            })
            .collect()
    })
}

impl ModelArgs {
    fn layer_type(&self, index: usize) -> LayerType {
        self.layer_types
            .get(index)
            .copied()
            .unwrap_or_else(|| default_layer_type(index))
    }

    fn rope_theta(&self) -> f32 {
        float_config_value(&self.rope_parameters, "rope_theta")
            .or_else(|| float_config_value(&self.rope_scaling, "rope_theta"))
            .unwrap_or(1_000_000.0)
    }

    fn rope_config(&self) -> Option<HashMap<String, FloatOrString>> {
        rope_config_value(
            self.rope_parameters
                .clone()
                .or_else(|| self.rope_scaling.clone()),
        )
    }

    fn partial_rotary_factor(&self) -> f32 {
        float_config_value(&self.rope_parameters, "partial_rotary_factor")
            .or_else(|| float_config_value(&self.rope_scaling, "partial_rotary_factor"))
            .unwrap_or(0.25)
    }

    fn rope_dims(&self) -> i32 {
        let rope_type = string_config_value(&self.rope_parameters, "rope_type")
            .or_else(|| string_config_value(&self.rope_scaling, "rope_type"))
            .unwrap_or("default");
        if rope_type == "proportional" {
            self.head_dim
        } else {
            ((self.head_dim as f32 * self.partial_rotary_factor()).round() as i32)
                .clamp(2, self.head_dim)
        }
    }
}

fn default_layer_type(index: usize) -> LayerType {
    if (index + 1) % 4 == 0 {
        LayerType::FullAttention
    } else {
        LayerType::LinearAttention
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    pub layers: Vec<LayerCache>,
}

impl Cache {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            layers: (0..args.num_hidden_layers)
                .map(|index| match args.layer_type(index as usize) {
                    LayerType::FullAttention => {
                        LayerCache::FullAttention(ConcatKeyValueCache::new())
                    }
                    LayerType::LinearAttention => {
                        LayerCache::LinearAttention(LinearAttentionCache::default())
                    }
                })
                .collect(),
        }
    }

    fn offset(&self) -> i32 {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                LayerCache::FullAttention(cache) => Some(cache.offset()),
                LayerCache::LinearAttention(cache) => Some(cache.offset),
            })
            .unwrap_or(0)
    }

    fn prefill_state_dependency(&self) -> Result<Option<Array>, Exception> {
        let mut dependency: Option<Array> = None;
        for layer in &self.layers {
            match layer {
                LayerCache::FullAttention(cache) => {
                    for array in cache.arrays() {
                        let term = array.sum(None)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term)?,
                            None => term,
                        });
                    }
                }
                LayerCache::LinearAttention(cache) => {
                    if let Some(conv_state) = &cache.conv_state {
                        let term = conv_state.sum(None)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term)?,
                            None => term,
                        });
                    }
                    if let Some(recurrent_state) = &cache.recurrent_state {
                        let term = recurrent_state.sum(None)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term)?,
                            None => term,
                        });
                    }
                }
            }
        }
        dependency
            .map(|dependency| dependency.multiply(Array::from_f32(0.0)))
            .transpose()
    }
}

#[derive(Debug, Clone)]
pub enum LayerCache {
    FullAttention(ConcatKeyValueCache),
    LinearAttention(LinearAttentionCache),
}

#[derive(Debug, Clone, Default)]
pub struct LinearAttentionCache {
    pub conv_state: Option<Array>,
    pub recurrent_state: Option<Array>,
    pub offset: i32,
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextRmsNorm {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl Qwen3NextRmsNorm {
    pub fn new(dim: i32, eps: f32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::zeros::<f32>(&[dim])?),
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let variance = safemlx::ops::mean_axis(&x.square()?, -1, true)?;
        let normalized = x.multiply(safemlx::ops::rsqrt(
            variance.add(Array::from_f32(self.eps))?,
        )?)?;
        let scale = self.weight.as_ref().add(Array::from_f32(1.0))?;
        normalized.multiply(scale)
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextRmsNormGated {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl Qwen3NextRmsNormGated {
    pub fn new(dim: i32, eps: f32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::ones::<f32>(&[dim])?),
            eps,
        })
    }

    pub fn forward(&self, x: &Array, gate: &Array) -> Result<Array, Exception> {
        let variance = safemlx::ops::mean_axis(&x.square()?, -1, true)?;
        let normalized = x.multiply(safemlx::ops::rsqrt(
            variance.add(Array::from_f32(self.eps))?,
        )?)?;
        normalized
            .multiply(&*self.weight)?
            .multiply(silu(gate.clone())?)
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct FullAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub q_norm: Qwen3NextRmsNorm,
    #[param]
    pub k_norm: Qwen3NextRmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl FullAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let hidden = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let q_proj = nn::LinearBuilder::new(hidden, n_heads * head_dim * 2)
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden)
            .bias(args.attention_bias)
            .build()?;
        let rope_config = args.rope_config();
        let rope = initialize_rope(
            args.rope_dims(),
            args.rope_theta(),
            false,
            &rope_config,
            args.max_position_embeddings,
        )?;
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Qwen3NextRmsNorm::new(head_dim, args.rms_norm_eps)?,
            k_norm: Qwen3NextRmsNorm::new(head_dim, args.rms_norm_eps)?,
            rope,
        })
    }
}

pub struct FullAttentionInput<'a> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut ConcatKeyValueCache>,
}

impl Module<FullAttentionInput<'_>> for FullAttention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: FullAttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let FullAttentionInput { x, mask, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let q_proj = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, 2 * self.head_dim])?;
        let query = q_proj.index((.., .., .., ..self.head_dim));
        let gate = q_proj.index((.., .., .., self.head_dim..)).reshape(&[
            B,
            L,
            self.n_heads * self.head_dim,
        ])?;
        let mut query = self.q_norm.forward(&query)?.transpose_axes(&[0, 2, 1, 3])?;
        let mut key = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x)?
                    .reshape(&[B, L, self.n_kv_heads, self.head_dim])?,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut value = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(cache) = cache.as_mut() {
            let offset = cache.offset();
            query = self
                .rope
                .forward(nn::RopeInputBuilder::new(&query).offset(offset).build()?)?;
            key = self
                .rope
                .forward(nn::RopeInputBuilder::new(&key).offset(offset).build()?)?;
            (key, value) = cache.update_and_fetch(key, value)?;
        } else {
            query = self.rope.forward(nn::RopeInput::new(&query))?;
            key = self.rope.forward(nn::RopeInput::new(&key))?;
        }

        let out =
            crate::utils::scaled_dot_product_attention(query, key, value, cache, self.scale, mask)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?
                .multiply(sigmoid(gate)?)?;
        self.o_proj.forward(&out)
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

#[derive(Debug, Clone, ModuleParameters)]
pub struct DepthwiseConv1d {
    #[param]
    pub weight: Param<Array>,
}

impl DepthwiseConv1d {
    pub fn new(channels: i32, kernel_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::zeros::<f32>(&[channels, 1, kernel_size])?),
        })
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
pub struct LinearAttention {
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_dim: i32,
    pub conv_kernel_size: i32,
    #[param]
    pub conv1d: DepthwiseConv1d,
    #[param]
    pub in_proj_qkv: nn::Linear,
    #[param]
    pub in_proj_z: nn::Linear,
    #[param]
    pub in_proj_b: nn::Linear,
    #[param]
    pub in_proj_a: nn::Linear,
    #[param]
    pub dt_bias: Param<Array>,
    #[param]
    pub A_log: Param<Array>,
    #[param]
    pub norm: Qwen3NextRmsNormGated,
    #[param]
    pub out_proj: nn::Linear,
}

impl LinearAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let num_v_heads = args.linear_num_value_heads;
        let num_k_heads = args.linear_num_key_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let projection_size_qkv = key_dim * 2 + value_dim;
        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: args.linear_conv_kernel_dim,
            conv1d: DepthwiseConv1d::new(conv_dim, args.linear_conv_kernel_dim)?,
            in_proj_qkv: nn::LinearBuilder::new(args.hidden_size, projection_size_qkv)
                .bias(false)
                .build()?,
            in_proj_z: nn::LinearBuilder::new(args.hidden_size, value_dim)
                .bias(false)
                .build()?,
            in_proj_b: nn::LinearBuilder::new(args.hidden_size, num_v_heads)
                .bias(false)
                .build()?,
            in_proj_a: nn::LinearBuilder::new(args.hidden_size, num_v_heads)
                .bias(false)
                .build()?,
            dt_bias: Param::new(Array::ones::<f32>(&[num_v_heads])?),
            A_log: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            norm: Qwen3NextRmsNormGated::new(head_v_dim, args.rms_norm_eps)?,
            out_proj: nn::LinearBuilder::new(value_dim, args.hidden_size)
                .bias(false)
                .build()?,
        })
    }

    #[allow(non_snake_case)]
    fn depthwise_causal_conv(
        &self,
        mixed_qkv: &Array,
        cache: Option<&mut LinearAttentionCache>,
    ) -> Result<Array, Exception> {
        let shape = mixed_qkv.shape();
        let B = shape[0];
        let L = shape[1];
        let C = shape[2];
        let state_len = self.conv_kernel_size - 1;
        let state = cache
            .as_ref()
            .and_then(|cache| cache.conv_state.clone())
            .unwrap_or(zeros::<f32>(&[B, state_len, C])?);
        let padded = concatenate_axis(&[state, mixed_qkv.clone()], 1)?;
        if let Some(cache) = cache {
            cache.conv_state = Some(padded.index((.., L.., ..)));
            cache.offset += L;
        }

        if L > 1 {
            let weight = self.conv1d.weight.swap_axes(1, 2)?;
            let out = conv1d(&padded, &weight, Some(1), Some(0), Some(1), Some(C))?;
            return silu(out);
        }

        let mut out: Option<Array> = None;
        for k in 0..self.conv_kernel_size {
            let window = padded.index((.., k..k + L, ..));
            let weight = self.conv1d.weight.index((.., 0, k)).reshape(&[1, 1, C])?;
            let term = window.multiply(weight)?;
            out = Some(match out {
                Some(acc) => acc.add(term)?,
                None => term,
            });
        }
        silu(out.expect("conv kernel must have at least one tap"))
    }

    fn l2norm(x: Array) -> Result<Array, Exception> {
        let denom = sum_axis(&x.square()?, -1, true)?.add(Array::from_f32(1e-6))?;
        x.multiply(safemlx::ops::rsqrt(denom)?)
    }

    fn recurrent_state_read(state: &Array, vector: &Array) -> Result<Array, Exception> {
        let vector = vector.index((.., .., NewAxis, ..));
        Ok(matmul(&vector, state)?.index((.., .., 0, ..)))
    }

    fn recurrent_state_update(key: &Array, delta: &Array) -> Result<Array, Exception> {
        let key = key.index((.., .., .., NewAxis));
        let delta = delta.index((.., .., NewAxis, ..));
        matmul(&key, &delta)
    }

    fn recurrent_delta_decode_kernel(
        state: &Array,
        query: &Array,
        key: &Array,
        value: &Array,
        g: &Array,
        beta: &Array,
    ) -> Result<(Array, Array), Exception> {
        let shape = state.shape();
        let b = shape[0];
        let h = shape[1];
        let kd = shape[2];
        let vd = shape[3];
        let state = state.as_dtype(Dtype::Float32)?;
        let query = query.as_dtype(Dtype::Float32)?;
        let key = key.as_dtype(Dtype::Float32)?;
        let value = value.as_dtype(Dtype::Float32)?;
        let g = g.as_dtype(Dtype::Float32)?;
        let beta = beta.as_dtype(Dtype::Float32)?;

        let outputs = RECURRENT_DECODE_KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "qwen35_moe_recurrent_decode",
                    ["state", "query", "key", "value", "g", "beta"],
                    ["state_out", "out"],
                    concat!(
                        "uint elem = thread_position_in_grid.x;",
                        "uint vd = elem % VD;",
                        "uint group = elem / VD;",
                        "uint state_base = group * KD * VD;",
                        "uint vec_base = group * KD;",
                        "uint value_base = group * VD;",
                        "float gate = metal::exp(g[group]);",
                        "float kv_mem = 0.0f;",
                        "for (uint kd = 0; kd < KD; ++kd) {",
                        "  uint state_idx = state_base + kd * VD + vd;",
                        "  kv_mem += float(state[state_idx]) * gate * float(key[vec_base + kd]);",
                        "}",
                        "float delta = (float(value[value_base + vd]) - kv_mem) * float(beta[group]);",
                        "float acc = 0.0f;",
                        "for (uint kd = 0; kd < KD; ++kd) {",
                        "  uint state_idx = state_base + kd * VD + vd;",
                        "  float updated = float(state[state_idx]) * gate + float(key[vec_base + kd]) * delta;",
                        "  state_out[state_idx] = updated;",
                        "  acc += updated * float(query[vec_base + kd]);",
                        "}",
                        "out[value_base + vd] = acc;"
                    ),
                    "",
                    true,
                    false,
                )?);
            }
            let config = MetalKernelConfig::new()
                .with_template_arg_int("KD", kd)
                .with_template_arg_int("VD", vd)
                .with_grid([b * h * vd, 1, 1])
                .with_thread_group([256, 1, 1])
                .with_output_arg([b, h, kd, vd], Dtype::Float32)
                .with_output_arg([b, h, vd], Dtype::Float32);
            cell.borrow()
                .as_ref()
                .expect("recurrent decode kernel initialized")
                .apply([&state, &query, &key, &value, &g, &beta], &config)
        })?;

        let mut outputs = outputs;
        let out = outputs.remove(1);
        let state = outputs.remove(0);
        Ok((state, out))
    }

    #[allow(non_snake_case)]
    fn recurrent_delta_rule(
        &self,
        query: Array,
        key: Array,
        value: Array,
        g: Array,
        beta: Array,
        cache: Option<&mut LinearAttentionCache>,
    ) -> Result<Array, Exception> {
        let shape = query.shape();
        let B = shape[0];
        let L = shape[1];
        let H = shape[2];
        let KD = shape[3];
        let VD = value.shape()[3];
        let scale = (KD as f32).sqrt().recip();
        let query = query.multiply(Array::from_f32(scale))?;
        let mut state = cache
            .as_ref()
            .and_then(|cache| cache.recurrent_state.clone())
            .unwrap_or(zeros::<f32>(&[B, H, KD, VD])?);

        if L == 1 {
            let q_t = query.index((.., 0, .., ..));
            let k_t = key.index((.., 0, .., ..));
            let v_t = value.index((.., 0, .., ..));
            let g_t = g.index((.., 0, ..));
            let beta_t = beta.index((.., 0, ..));
            let (new_state, out_t) =
                Self::recurrent_delta_decode_kernel(&state, &q_t, &k_t, &v_t, &g_t, &beta_t)?;
            if let Some(cache) = cache {
                cache.recurrent_state = Some(new_state);
            }
            return Ok(out_t.index((.., NewAxis, .., ..)));
        }

        let mut outs = Vec::with_capacity(L as usize);

        for t in 0..L {
            let q_t = query.index((.., t, .., ..));
            let k_t = key.index((.., t, .., ..));
            let v_t = value.index((.., t, .., ..));
            let g_t = exp(g.index((.., t, ..)).index((.., .., NewAxis, NewAxis)))?;
            let beta_t = beta.index((.., t, ..)).index((.., .., NewAxis));
            state = state.multiply(g_t)?;
            let kv_mem = Self::recurrent_state_read(&state, &k_t)?;
            let delta = v_t.subtract(kv_mem)?.multiply(beta_t)?;
            state = state.add(Self::recurrent_state_update(&k_t, &delta)?)?;
            let out_t = Self::recurrent_state_read(&state, &q_t)?;
            outs.push(out_t.index((.., NewAxis, .., ..)));
        }

        if let Some(cache) = cache {
            cache.recurrent_state = Some(state);
        }
        concatenate_axis(&outs, 1)
    }
}

pub struct LinearAttentionInput<'a> {
    pub x: &'a Array,
    pub cache: Option<&'a mut LinearAttentionCache>,
}

impl Module<LinearAttentionInput<'_>> for LinearAttention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: LinearAttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let LinearAttentionInput { x, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let mixed_qkv = self.in_proj_qkv.forward(x)?;
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape(&[B, L, self.num_v_heads, self.head_v_dim])?;
        let b = self.in_proj_b.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let mixed_qkv = self.depthwise_causal_conv(&mixed_qkv, cache.as_deref_mut())?;
        let query = mixed_qkv.index((.., .., ..self.key_dim)).reshape(&[
            B,
            L,
            self.num_k_heads,
            self.head_k_dim,
        ])?;
        let key = mixed_qkv
            .index((.., .., self.key_dim..2 * self.key_dim))
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim])?;
        let mut value = mixed_qkv.index((.., .., 2 * self.key_dim..)).reshape(&[
            B,
            L,
            self.num_v_heads,
            self.head_v_dim,
        ])?;
        let mut query = Self::l2norm(query)?;
        let mut key = Self::l2norm(key)?;
        let beta = sigmoid(b)?;
        let g = nn::softplus(a.add(self.dt_bias.reshape(&[1, 1, self.num_v_heads])?)?)?
            .multiply(exp(self.A_log.as_ref())?.multiply(Array::from_f32(-1.0))?)?;

        let repeats = self.num_v_heads / self.num_k_heads;
        if repeats > 1 {
            let expanded_query = query.index((.., .., .., NewAxis, ..));
            query = broadcast_to(
                &expanded_query,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim])?;
            let expanded_key = key.index((.., .., .., NewAxis, ..));
            key = broadcast_to(
                &expanded_key,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim])?;
        }

        value = value.as_dtype(x.dtype())?;
        let core = self.recurrent_delta_rule(query, key, value, g, beta, cache)?;
        let z_shape = z.shape().to_vec();
        let core = core.reshape(&[-1, self.head_v_dim])?;
        let z = z.reshape(&[-1, self.head_v_dim])?;
        let out =
            self.norm
                .forward(&core, &z)?
                .reshape(&z_shape)?
                .reshape(&[B, L, self.value_dim])?;
        self.out_proj.forward(&out)
    }

    fn training_mode(&mut self, mode: bool) {
        self.in_proj_qkv.training_mode(mode);
        self.in_proj_z.training_mode(mode);
        self.in_proj_b.training_mode(mode);
        self.in_proj_a.training_mode(mode);
        self.norm.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Mlp {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl Mlp {
    pub fn new(args: &ModelArgs, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(args.hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(args.hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, args.hidden_size)
                .bias(false)
                .build()?,
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let h = silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Experts {
    pub num_experts: i32,
    pub hidden_dim: i32,
    pub intermediate_dim: i32,
    #[param]
    pub gate_up_proj: Param<Array>,
    #[param]
    pub down_proj: Param<Array>,
}

impl Experts {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            num_experts: args.num_experts,
            hidden_dim: args.hidden_size,
            intermediate_dim: args.moe_intermediate_size,
            gate_up_proj: Param::new(Array::zeros::<f32>(&[
                args.num_experts,
                2 * args.moe_intermediate_size,
                args.hidden_size,
            ])?),
            down_proj: Param::new(Array::zeros::<f32>(&[
                args.num_experts,
                args.hidden_size,
                args.moe_intermediate_size,
            ])?),
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        let top_k = top_k_index.shape()[1];
        let selected_gate_up = self.gate_up_proj.as_ref().take_axis(top_k_index, 0)?;
        let hidden = hidden_states.index((.., NewAxis, NewAxis, ..));
        let gate_up = matmul(&hidden, selected_gate_up.swap_axes(-1, -2)?)?.reshape(&[
            num_tokens,
            top_k,
            2 * self.intermediate_dim,
        ])?;
        let gate = gate_up.index((.., .., ..self.intermediate_dim));
        let up = gate_up.index((.., .., self.intermediate_dim..));
        let current = silu(gate)?.multiply(up)?;

        let selected_down = self.down_proj.as_ref().take_axis(top_k_index, 0)?;
        let current = matmul(
            current.index((.., .., NewAxis, ..)),
            selected_down.swap_axes(-1, -2)?,
        )?
        .reshape(&[num_tokens, top_k, self.hidden_dim])?;
        let weighted = current.multiply(top_k_weights.index((.., .., NewAxis)))?;
        sum_axis(&weighted, -2, false)
    }

    pub fn forward_chunked(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        if num_tokens <= ROUTED_EXPERT_CHUNK_THRESHOLD {
            return self.forward(hidden_states, top_k_index, top_k_weights);
        }

        let mut outputs = Vec::with_capacity(
            ((num_tokens + ROUTED_EXPERT_CHUNK_TOKENS - 1) / ROUTED_EXPERT_CHUNK_TOKENS)
                .try_into()
                .expect("number of MoE chunks must fit in usize"),
        );
        let mut start = 0;
        while start < num_tokens {
            let end = (start + ROUTED_EXPERT_CHUNK_TOKENS).min(num_tokens);
            let hidden_chunk = hidden_states.index((start..end, ..));
            let expert_chunk = top_k_index.index((start..end, ..));
            let weight_chunk = top_k_weights.index((start..end, ..));
            outputs.push(self.forward_expert_major_chunk(
                &hidden_chunk,
                &expert_chunk,
                &weight_chunk,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 0)
    }

    fn forward_expert_major_chunk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        let plan = topk_route_plan(top_k_index, self.num_experts)?;
        let hidden = gather_grouped_rows(hidden_states, &plan)?;
        let gate_up_weights = self.gate_up_proj.as_ref().swap_axes(-1, -2)?;
        let gate_up = grouped_matmul(&hidden, &gate_up_weights, &plan.sorted_group_ids, true)?;
        let gate = gate_up.index((.., ..self.intermediate_dim));
        let up = gate_up.index((.., self.intermediate_dim..));
        let current = silu(gate)?.multiply(up)?;

        let down_weights = self.down_proj.as_ref().swap_axes(-1, -2)?;
        let current = grouped_matmul(&current, &down_weights, &plan.sorted_group_ids, true)?;
        let weights = gather_route_values(top_k_weights, &plan)?.index((.., NewAxis));
        let weighted = current.multiply(weights)?;
        segment_sum_by_index(weighted, &plan.token_indices, num_tokens)
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct TopKRouter {
    pub top_k: i32,
    pub num_experts: i32,
    pub norm_topk_prob: bool,
    #[param]
    pub weight: Param<Array>,
}

impl TopKRouter {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            top_k: args.num_experts_per_tok,
            num_experts: args.num_experts,
            norm_topk_prob: args.norm_topk_prob,
            weight: Param::new(Array::zeros::<f32>(&[args.num_experts, args.hidden_size])?),
        })
    }

    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Array), Exception> {
        let router_logits = matmul(hidden_states, self.weight.t())?;
        let router_probs = softmax_axis(&router_logits, -1, true)?;
        let top_k_index =
            argpartition_axis(&router_probs, -self.top_k, -1)?.index((.., -self.top_k..));
        let mut top_k_weights = take_along_axis(&router_probs, &top_k_index, -1)?;
        top_k_weights = top_k_weights.divide(sum_axis(&top_k_weights, -1, true)?)?;
        Ok((top_k_index, top_k_weights.as_dtype(router_logits.dtype())?))
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct SparseMoeBlock {
    #[param]
    pub gate: TopKRouter,
    #[param]
    pub experts: Experts,
    #[param]
    pub shared_expert: Mlp,
    #[param]
    pub shared_expert_gate: nn::Linear,
}

impl SparseMoeBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            gate: TopKRouter::new(args)?,
            experts: Experts::new(args)?,
            shared_expert: Mlp::new(args, args.shared_expert_intermediate_size)?,
            shared_expert_gate: nn::LinearBuilder::new(args.hidden_size, 1)
                .bias(false)
                .build()?,
        })
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, hidden_states: &Array) -> Result<Self::Output, Self::Error> {
        let shape = hidden_states.shape();
        let B = shape[0];
        let L = shape[1];
        let H = shape[2];
        let flat = hidden_states.reshape(&[-1, H])?;
        let shared = self
            .shared_expert
            .forward(&flat)?
            .multiply(sigmoid(self.shared_expert_gate.forward(&flat)?)?)?;
        profile_array(PerfComponent::MoeShared, &shared)?;
        let (selected_experts, routing_weights) = self.gate.forward(&flat)?;
        profile_arrays(
            PerfComponent::MoeRouter,
            &[&selected_experts, &routing_weights],
        )?;
        let routed = self
            .experts
            .forward_chunked(&flat, &selected_experts, &routing_weights)?;
        profile_array(PerfComponent::MoeRouted, &routed)?;
        let output = routed.add(shared)?.reshape(&[B, L, H])?;
        profile_array(PerfComponent::MoeCombine, &output)?;
        Ok(output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
        self.experts.training_mode(mode);
        self.shared_expert.training_mode(mode);
        self.shared_expert_gate.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerBlock {
    pub layer_type: LayerType,
    #[param]
    pub self_attn: Option<FullAttention>,
    #[param]
    pub linear_attn: Option<LinearAttention>,
    #[param]
    pub mlp: SparseMoeBlock,
    #[param]
    pub input_layernorm: Qwen3NextRmsNorm,
    #[param]
    pub post_attention_layernorm: Qwen3NextRmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs, layer_idx: usize) -> Result<Self, Exception> {
        let layer_type = args.layer_type(layer_idx);
        Ok(Self {
            layer_type,
            self_attn: if layer_type == LayerType::FullAttention {
                Some(FullAttention::new(args)?)
            } else {
                None
            },
            linear_attn: if layer_type == LayerType::LinearAttention {
                Some(LinearAttention::new(args)?)
            } else {
                None
            },
            mlp: SparseMoeBlock::new(args)?,
            input_layernorm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            post_attention_layernorm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
        })
    }
}

pub struct BlockInput<'a> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut LayerCache>,
}

impl Module<BlockInput<'_>> for TransformerBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BlockInput<'_>) -> Result<Self::Output, Self::Error> {
        let BlockInput { x, mask, cache } = input;
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = match (self.layer_type, cache) {
            (LayerType::FullAttention, Some(LayerCache::FullAttention(cache))) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(FullAttentionInput {
                    x: &h,
                    mask,
                    cache: Some(cache),
                })?,
            (LayerType::FullAttention, _) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(FullAttentionInput {
                    x: &h,
                    mask,
                    cache: None,
                })?,
            (LayerType::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(LinearAttentionInput {
                    x: &h,
                    cache: Some(cache),
                })?,
            (LayerType::LinearAttention, _) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(LinearAttentionInput { x: &h, cache: None })?,
        };
        match self.layer_type {
            LayerType::FullAttention => profile_array(PerfComponent::FullAttention, &h)?,
            LayerType::LinearAttention => profile_array(PerfComponent::LinearAttention, &h)?,
        }
        let h = residual.add(h)?;
        let residual = h.clone();
        let h = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        residual.add(h)
    }

    fn training_mode(&mut self, mode: bool) {
        if let Some(full_attention) = &mut self.self_attn {
            full_attention.training_mode(mode);
        }
        if let Some(linear_attention) = &mut self.linear_attn {
            linear_attention.training_mode(mode);
        }
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen35MoeTextModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: Qwen3NextRmsNorm,
}

impl Qwen35MoeTextModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new(args, idx as usize))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens,
            layers,
            norm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
        })
    }
}

pub struct ModelInput<'a> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut Cache>,
}

impl Module<ModelInput<'_>> for Qwen35MoeTextModel {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            mut cache,
        } = input;
        let mut h = self.embed_tokens.forward(inputs)?;
        profile_array(PerfComponent::Embed, &h)?;
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => {
                let offset = cache.as_ref().map(|cache| cache.offset()).unwrap_or(0);
                if h.shape()[1] > 1 {
                    match create_attention_mask(&h, &offset_cache(offset), Some(true))? {
                        Some(AttentionMask::Array(a)) => Some(a),
                        Some(AttentionMask::Causal) => {
                            return Err(Exception::custom("Only `Array` mask is supported"));
                        }
                        None => None,
                    }
                } else {
                    None
                }
            }
        };

        if let Some(cache) = cache.as_mut() {
            for (layer, layer_cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
                h = layer.forward(BlockInput {
                    x: &h,
                    mask: mask.as_ref(),
                    cache: Some(layer_cache),
                })?;
            }
        } else {
            for layer in &mut self.layers {
                h = layer.forward(BlockInput {
                    x: &h,
                    mask: mask.as_ref(),
                    cache: None,
                })?;
            }
        }
        let h = self.norm.forward(&h)?;
        profile_array(PerfComponent::FinalNorm, &h)?;
        Ok(h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm.training_mode(mode);
    }
}

fn offset_cache(offset: i32) -> Vec<Option<OffsetOnlyCache>> {
    vec![Some(OffsetOnlyCache { offset })]
}

struct OffsetOnlyCache {
    offset: i32,
}

impl KeyValueCache for OffsetOnlyCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        Ok((keys, values))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub args: ModelArgs,
    pub image_token_id: Option<i32>,
    pub video_token_id: Option<i32>,
    #[param]
    pub model: Qwen35MoeTextModel,
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl Model {
    pub fn new(
        args: ModelArgs,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
    ) -> Result<Self, Exception> {
        let model = Qwen35MoeTextModel::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            )
        } else {
            None
        };
        Ok(Self {
            args,
            image_token_id,
            video_token_id,
            model,
            lm_head,
        })
    }

    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn reject_multimodal_tokens(&self, inputs: &Array) -> Result<(), Exception> {
        for (name, token_id) in [
            ("image", self.image_token_id),
            ("video", self.video_token_id),
        ] {
            if let Some(token_id) = token_id {
                let contains = inputs
                    .eq(Array::from_int(token_id))?
                    .max(None)?
                    .item::<bool>();
                if contains {
                    return Err(Exception::custom(format!(
                        "qwen3_5_moe text-generation support does not accept {name} tokens"
                    )));
                }
            }
        }
        Ok(())
    }

    fn project_logits(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(hidden_states),
            None => self.model.embed_tokens.as_linear(hidden_states),
        }?;
        profile_array(PerfComponent::LmHead, &logits)?;
        Ok(logits)
    }

    fn forward_logits(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
    ) -> Result<Array, Exception> {
        self.reject_multimodal_tokens(input.inputs)?;
        let hidden_states = self.model.forward(input)?;
        let hidden_states = if last_token_only {
            hidden_states.index((.., -1, ..))
        } else {
            hidden_states
        };
        self.project_logits(&hidden_states)
    }
}

impl Module<ModelInput<'_>> for Model {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_>) -> Result<Self::Output, Self::Error> {
        self.forward_logits(input, false)
    }

    fn training_mode(&mut self, mode: bool) {
        self.model.training_mode(mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_qwen3_5_moe_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_qwen3_5_moe_model_args(
    model_dir: impl AsRef<Path>,
) -> Result<(ModelArgs, Option<i32>, Option<i32>), Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let config: TopLevelConfig = serde_json::from_reader(file)?;
    let mut args = match config.model_type.as_str() {
        "qwen3_5_moe" => config.text_config.ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3_5_moe config is missing text_config".to_string())
        })?,
        "qwen3_5_moe_text" => {
            let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
            serde_json::from_reader(file)?
        }
        other => {
            return Err(Error::UnsupportedModelType(other.to_string()));
        }
    };
    args.model_type = "qwen3_5_moe_text".to_string();
    if let Some(tie_word_embeddings) = config.tie_word_embeddings {
        args.tie_word_embeddings = tie_word_embeddings;
    }
    if args.layer_types.is_empty() {
        args.layer_types = (0..args.num_hidden_layers)
            .map(|idx| default_layer_type(idx as usize))
            .collect();
    }
    Ok((args, config.image_token_id, config.video_token_id))
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_qwen3_5_moe_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id) = get_qwen3_5_moe_model_args(model_dir)?;
    let mut model = Model::new(args, image_token_id, video_token_id)?;
    let config = qwen3_5_moe_strict_load_config();
    let mut report = StrictLoadReport::default();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            load_safetensors_strict(
                &mut model,
                model_dir.join(weight_file),
                &config,
                &mut report,
            )?;
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

fn qwen3_5_moe_strict_load_config() -> StrictLoadConfig {
    StrictLoadConfig::default()
        .rewrite_prefix("model.language_model.", "model.")
        .rewrite_prefix("language_model.", "model.")
        .rewrite_prefix("model.model.", "model.")
        .allow_unused_prefix("visual.")
        .allow_unused_prefix("vision_tower.")
        .allow_unused_prefix("model.visual.")
        .allow_unused_prefix("model.vision_tower.")
        .allow_unused_prefix("mtp.")
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

pub struct Generate<'a> {
    model: &'a mut Model,
    cache: &'a mut Cache,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a> Generate<'a> {
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Cache,
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

impl Iterator for Generate<'_> {
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: Some(self.cache),
                };
                let mut logits = tri!(self.model.forward_logits(input, true));
                // Keep the first sampled token dependent on all prefill cache state while
                // avoiding a prompt-length vocabulary projection.
                if let Some(dependency) = tri!(self.cache.prefill_state_dependency()) {
                    tri!(profile_array(
                        PerfComponent::PrefillStateDependency,
                        &dependency
                    ));
                    logits = tri!(logits.add(dependency));
                }
                let y = tri!(sample(&logits, self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: Some(self.cache),
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits, self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        default_layer_type, get_qwen3_5_moe_model_args, load_qwen3_5_moe_model,
        load_qwen3_5_moe_tokenizer, qwen3_5_moe_strict_load_config, FullAttention,
        FullAttentionInput, LayerType, LinearAttention, LinearAttentionInput, Model, ModelArgs,
        SparseMoeBlock,
    };
    use crate::{
        error::Error,
        weights::{load_safetensors_strict, StrictLoadReport},
    };
    use safemlx::{
        module::{Module, ModuleParameters, Param},
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
        Array,
    };
    use std::{
        fs,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Mutex, MutexGuard,
        },
        time::{SystemTime, UNIX_EPOCH},
    };

    static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static MLX_RUNTIME_TEST_MUTEX: Mutex<()> = Mutex::new(());

    fn mlx_runtime_test_guard() -> MutexGuard<'static, ()> {
        MLX_RUNTIME_TEST_MUTEX.lock().unwrap()
    }

    fn cached_test_model_dir() -> std::path::PathBuf {
        let local = std::path::PathBuf::from("../cache/Qwen3.5-35B-A3B");
        if local.exists() {
            return local;
        }

        let home = std::env::var("HOME").expect("HOME must be set");
        let snapshots = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots");
        let mut entries = fs::read_dir(&snapshots)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", snapshots.display()))
            .map(|entry| entry.expect("snapshot entry").path())
            .filter(|path| path.join("config.json").exists())
            .collect::<Vec<_>>();
        entries.sort();
        entries.pop().unwrap_or_else(|| {
            panic!(
                "no Qwen3.5-35B-A3B snapshot found in {}",
                snapshots.display()
            )
        })
    }

    fn tiny_args(layer_types: Vec<LayerType>) -> ModelArgs {
        ModelArgs {
            model_type: "qwen3_5_moe_text".to_string(),
            vocab_size: 128,
            hidden_size: 16,
            num_hidden_layers: layer_types.len() as i32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            attention_bias: false,
            hidden_act: "silu".to_string(),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            moe_intermediate_size: 4,
            shared_expert_intermediate_size: 4,
            num_experts_per_tok: 2,
            num_experts: 4,
            norm_topk_prob: false,
            layer_types,
            rope_parameters: None,
            rope_scaling: None,
        }
    }

    fn temp_model_dir(config: &str) -> std::path::PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "qwen35_moe_test_{}_{}_{}",
            std::process::id(),
            id,
            counter
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), config).unwrap();
        dir
    }

    fn save_model_parameters(
        path: &std::path::Path,
        model: &Model,
        include: impl Fn(&str) -> bool,
        extras: Vec<(String, Array)>,
    ) {
        let params = model.parameters().flatten();
        let mut arrays = params
            .iter()
            .filter(|(key, _)| include(key))
            .map(|(key, value)| (key.to_string(), (*value).clone()))
            .collect::<Vec<_>>();
        arrays.extend(extras);
        Array::save_safetensors(
            arrays.iter().map(|(key, value)| (key.as_str(), value)),
            None,
            path,
        )
        .unwrap();
    }

    #[test]
    fn default_layers_are_three_linear_then_full() {
        assert_eq!(default_layer_type(0), LayerType::LinearAttention);
        assert_eq!(default_layer_type(1), LayerType::LinearAttention);
        assert_eq!(default_layer_type(2), LayerType::LinearAttention);
        assert_eq!(default_layer_type(3), LayerType::FullAttention);
    }

    #[test]
    fn parses_top_level_qwen3_5_moe_config() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe",
              "tie_word_embeddings": false,
              "image_token_id": 248056,
              "video_token_id": 248057,
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 128,
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128
              }
            }"#,
        );
        let (args, image_token_id, video_token_id) = get_qwen3_5_moe_model_args(&dir).unwrap();
        assert_eq!(args.model_type, "qwen3_5_moe_text");
        assert_eq!(args.layer_types.len(), 4);
        assert_eq!(args.layer_types[3], LayerType::FullAttention);
        assert_eq!(image_token_id, Some(248056));
        assert_eq!(video_token_id, Some(248057));
    }

    #[test]
    fn parses_text_only_qwen3_5_moe_config() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe_text",
              "vocab_size": 128,
              "hidden_size": 16,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "max_position_embeddings": 128,
              "layer_types": ["full_attention"]
            }"#,
        );
        let (args, _, _) = get_qwen3_5_moe_model_args(&dir).unwrap();
        assert_eq!(args.model_type, "qwen3_5_moe_text");
        assert_eq!(args.layer_types, vec![LayerType::FullAttention]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_attention_forward_shape_smoke() {
        let _guard = mlx_runtime_test_guard();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut attn = FullAttention::new(&args).unwrap();
        let x = Array::zeros::<f32>(&[1, 2, args.hidden_size]).unwrap();
        let out = attn
            .forward(FullAttentionInput {
                x: &x,
                mask: None,
                cache: None,
            })
            .unwrap();
        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn linear_attention_forward_shape_smoke() {
        let _guard = mlx_runtime_test_guard();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let mut attn = LinearAttention::new(&args).unwrap();
        let x = Array::zeros::<f32>(&[1, 2, args.hidden_size]).unwrap();
        let out = attn
            .forward(LinearAttentionInput { x: &x, cache: None })
            .unwrap();
        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn sparse_moe_forward_shape_smoke() {
        let _guard = mlx_runtime_test_guard();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let mut moe = SparseMoeBlock::new(&args).unwrap();
        let gate_values = (0..args.num_experts)
            .flat_map(|expert| {
                (0..args.hidden_size).map(move |hidden| ((expert + 1) * (hidden + 1)) as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        moe.gate.weight = Param::new(
            Array::from(gate_values.as_slice())
                .reshape(&[args.num_experts, args.hidden_size])
                .unwrap(),
        );
        let input_values = (0..(2 * args.hidden_size))
            .map(|index| index as f32 * 0.01)
            .collect::<Vec<_>>();
        let x = Array::from(input_values.as_slice())
            .reshape(&[1, 2, args.hidden_size])
            .unwrap();
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn parameter_tree_matches_public_checkpoint_key_patterns() {
        let _guard = mlx_runtime_test_guard();
        let args = tiny_args(vec![LayerType::LinearAttention, LayerType::FullAttention]);
        let model = Model::new(args, Some(248056), Some(248057)).unwrap();
        let params = model.parameters().flatten();

        for key in [
            "model.embed_tokens.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_z.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            "model.layers.0.linear_attn.in_proj_a.weight",
            "model.layers.0.linear_attn.conv1d.weight",
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.self_attn.q_norm.weight",
            "model.layers.1.self_attn.k_norm.weight",
            "lm_head.weight",
        ] {
            assert!(params.contains_key(key), "missing parameter key {key}");
        }

        assert!(
            !params.contains_key("model.layers.0.linear_attn.in_proj_qkvz.weight"),
            "combined qkvz projection should not exist"
        );
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_allows_unused_non_text_prefixes() {
        let _guard = mlx_runtime_test_guard();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let source = Model::new(args.clone(), None, None).unwrap();
        let dir = temp_model_dir("{}");
        let weights_path = dir.join("model.safetensors");
        save_model_parameters(
            &weights_path,
            &source,
            |_| true,
            vec![
                (
                    "visual.extra.weight".to_string(),
                    Array::zeros::<f32>(&[1]).unwrap(),
                ),
                (
                    "model.vision_tower.extra.weight".to_string(),
                    Array::zeros::<f32>(&[1]).unwrap(),
                ),
                (
                    "mtp.extra.weight".to_string(),
                    Array::zeros::<f32>(&[1]).unwrap(),
                ),
            ],
        );

        let mut target = Model::new(args, None, None).unwrap();
        let config = qwen3_5_moe_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_safetensors_strict(&mut target, &weights_path, &config, &mut report).unwrap();
        report.finish(&target, &config).unwrap();
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_fails_on_missing_text_weight() {
        let _guard = mlx_runtime_test_guard();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let source = Model::new(args.clone(), None, None).unwrap();
        let dir = temp_model_dir("{}");
        let weights_path = dir.join("model.safetensors");
        save_model_parameters(
            &weights_path,
            &source,
            |key| key != "lm_head.weight",
            Vec::new(),
        );

        let mut target = Model::new(args, None, None).unwrap();
        let config = qwen3_5_moe_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_safetensors_strict(&mut target, &weights_path, &config, &mut report).unwrap();
        let Err(Error::StrictLoadValidation { missing, unused }) = report.finish(&target, &config)
        else {
            panic!("strict load should reject missing text weights");
        };
        assert!(missing.iter().any(|key| key == "lm_head.weight"));
        assert!(unused.is_empty());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_fails_on_unexpected_text_weight() {
        let _guard = mlx_runtime_test_guard();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let source = Model::new(args.clone(), None, None).unwrap();
        let dir = temp_model_dir("{}");
        let weights_path = dir.join("model.safetensors");
        save_model_parameters(
            &weights_path,
            &source,
            |_| true,
            vec![(
                "model.layers.0.linear_attn.unexpected.weight".to_string(),
                Array::zeros::<f32>(&[1]).unwrap(),
            )],
        );

        let mut target = Model::new(args, None, None).unwrap();
        let config = qwen3_5_moe_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_safetensors_strict(&mut target, &weights_path, &config, &mut report).unwrap();
        let Err(Error::StrictLoadValidation { missing, unused }) = report.finish(&target, &config)
        else {
            panic!("strict load should reject unexpected text weights");
        };
        assert!(missing.is_empty());
        assert!(unused
            .iter()
            .any(|key| key == "model.layers.0.linear_attn.unexpected.weight"));
    }

    #[test]
    #[ignore = "requires local Qwen3.5-MoE model files"]
    fn test_load_and_run_qwen3_5_moe_with_model_cache() {
        let _guard = mlx_runtime_test_guard();
        let model_dir = cached_test_model_dir();
        let tokenizer = load_qwen3_5_moe_tokenizer(&model_dir).unwrap();
        let mut model = load_qwen3_5_moe_model(&model_dir).unwrap();
        let cases = [
            (
                "What is 84 * 3 / 2?",
                vec![271, 1206, 11290, 17237, 220, 23, 19, 1088],
            ),
            (
                "Name three primary colors.",
                vec![
                    271, 248068, 198, 90700, 8340, 25, 271, 16, 13, 220, 2972, 2014,
                ],
            ),
            ("Write a haiku about rain.", vec![271, 248068, 271, 248069]),
            (
                "In one sentence, explain why the sky appears blue during the day.",
                vec![271, 248068, 271, 248069],
            ),
        ];

        for (prompt, expected_tokens) in cases {
            let encoding = tokenizer.encode(prompt, false).unwrap();
            let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
            let mut cache = model.new_cache();
            let mut tokens = Vec::new();
            let generate = super::Generate::new(&mut model, &mut cache, 0.0, &prompt_tokens);
            for token in generate.take(expected_tokens.len()) {
                let token = token.unwrap();
                eval([&token]).unwrap();
                tokens.push(token.item::<u32>());
            }
            assert_eq!(tokens, expected_tokens, "prompt: {prompt}");
        }
    }
}
