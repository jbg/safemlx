use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::Path,
    time::Instant,
};

use safemlx::{
    builder::Builder,
    error::Exception,
    fast::{MetalKernel, MetalKernelConfig, RecurrentScanKernel, StatefulMetalKernel},
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        argpartition_axis, broadcast_to, concatenate_axis, conv1d, exp, gather_grouped_rows,
        gather_route_values, grouped_matmul,
        indexing::{take_along_axis, NewAxis, TryIndexOp},
        matmul, segment_sum_by_index, sigmoid, softmax_axis, stack_axis, sum_axis, topk_route_plan,
        zeros,
    },
    transforms::eval,
    Array, Dtype, Stream,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

pub use super::common::sample;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    models::common::{self, project_logits_dense, silu, CausalLm},
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
    weights::{load_arrays_strict, load_safetensors_strict, StrictLoadConfig, StrictLoadReport},
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

thread_local! {
    static RECURRENT_DELTA_KERNELS: RefCell<Option<RecurrentScanKernel>> = const { RefCell::new(None) };
    static FP8_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static FP8_GROUPED_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

const ROUTED_EXPERT_CHUNK_THRESHOLD: i32 = 64;
const ROUTED_EXPERT_CHUNK_TOKENS: i32 = 32;
const RECURRENT_PREFILL_SHORT_SCAN_TOKENS: i32 = 64;
const RECURRENT_PREFILL_MEDIUM_SCAN_TOKENS: i32 = 16;
const RECURRENT_PREFILL_LONG_SCAN_TOKENS: i32 = 32;

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
    #[serde(default)]
    pub quantization_config: Option<QwenFp8QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct QwenFp8QuantizationConfig {
    pub quant_method: String,
    pub fmt: String,
    pub activation_scheme: String,
    #[serde(default)]
    pub weight_block_size: Option<Vec<i32>>,
    #[serde(default)]
    pub modules_to_not_convert: Vec<String>,
}

impl QwenFp8QuantizationConfig {
    fn validate_supported(&self) -> Result<(), Error> {
        if self.quant_method != "fp8" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE quantization method '{}'",
                self.quant_method
            )));
        }
        if self.fmt != "e4m3" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE FP8 format '{}'",
                self.fmt
            )));
        }
        if self.activation_scheme != "dynamic" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE FP8 activation scheme '{}'",
                self.activation_scheme
            )));
        }
        match self.weight_block_size.as_deref() {
            Some([128, 128]) => Ok(()),
            Some(other) => Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE FP8 weight block size {other:?}"
            ))),
            None => Err(Error::UnsupportedArchitecture(
                "Qwen3.5-MoE FP8 config is missing weight_block_size".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TopLevelConfig {
    model_type: String,
    #[serde(default)]
    text_config: Option<ModelArgs>,
    #[serde(default)]
    quantization_config: Option<QwenFp8QuantizationConfig>,
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

fn ceil_div(lhs: i32, rhs: i32) -> i32 {
    (lhs + rhs - 1) / rhs
}

impl ModelArgs {
    fn uses_fp8(&self) -> bool {
        self.quantization_config.is_some()
    }

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

#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenLinear {
    pub input_dims: i32,
    pub output_dims: i32,
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub weight_scale_inv: Param<Option<Array>>,
    #[param]
    pub bias: Param<Option<Array>>,
}

impl QwenLinear {
    fn new(
        input_dims: i32,
        output_dims: i32,
        bias: bool,
        fp8: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let weight_dtype = if fp8 { Dtype::Uint8 } else { Dtype::Float32 };
        Ok(Self {
            input_dims,
            output_dims,
            weight: Param::<Array>::unloaded(&[output_dims, input_dims], weight_dtype, stream)?,
            weight_scale_inv: if fp8 {
                Param::<Option<Array>>::unloaded_some(
                    &[ceil_div(output_dims, 128), ceil_div(input_dims, 128)],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[output_dims], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
        })
    }

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut output = if let Some(scale) = self.weight_scale_inv.as_ref() {
            fp8_linear(input, self.weight.as_ref(), scale, stream)?
        } else {
            matmul(input, self.weight.as_ref().transpose(stream)?, stream)?
        };
        if let Some(bias) = self.bias.as_ref() {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

fn fp8_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let input_shape = input.shape();
    let in_dim = input.dim(-1);
    let out_dim = weight.dim(0);
    let rows = (input.size() as i32) / in_dim;
    let input = input.reshape(&[rows, in_dim], stream)?;
    let scale_cols = scale.dim(-1);

    let out = FP8_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(fp8_linear_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("IN_DIM", in_dim)
            .with_template_arg_int("OUT_DIM", out_dim)
            .with_template_arg_int("SCALE_COLS", scale_cols)
            .with_grid([rows * out_dim, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([rows, out_dim], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("FP8 linear kernel initialized")
            .apply_one_device([&input, weight, scale], &config, stream)
    })?;

    let mut output_shape = input_shape.to_vec();
    if let Some(last) = output_shape.last_mut() {
        *last = out_dim;
    }
    out.reshape(&output_shape, stream)
}

fn fp8_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "qwen35_moe_fp8_linear",
        ["input", "weight", "scale"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint out_col = elem % OUT_DIM;",
            "uint row = elem / OUT_DIM;",
            "float acc = 0.0f;",
            "uint weight_base = out_col * IN_DIM;",
            "uint input_base = row * IN_DIM;",
            "uint scale_row = out_col / 128;",
            "for (uint k = 0; k < IN_DIM; ++k) {",
            "  uint8_t raw = weight[weight_base + k];",
            "  float w = fp8_e4m3_to_float(raw);",
            "  float s = float(scale[scale_row * SCALE_COLS + (k / 128)]);",
            "  acc += float(input[input_base + k]) * w * s;",
            "}",
            "out[elem] = acc;"
        ),
        FP8_METAL_HEADER,
        true,
        false,
    )
}

fn grouped_fp8_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let routes = input.dim(0);
    let in_dim = input.dim(-1);
    let out_dim = weight.dim(1);
    let scale_cols = scale.dim(-1);
    FP8_GROUPED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(grouped_fp8_linear_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("IN_DIM", in_dim)
            .with_template_arg_int("OUT_DIM", out_dim)
            .with_template_arg_int("SCALE_OUT", scale.dim(1))
            .with_template_arg_int("SCALE_COLS", scale_cols)
            .with_grid([routes * out_dim, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([routes, out_dim], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("grouped FP8 linear kernel initialized")
            .apply_one_device([input, weight, scale, group_ids], &config, stream)
    })
}

fn grouped_fp8_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "qwen35_moe_grouped_fp8_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint out_col = elem % OUT_DIM;",
            "uint route = elem / OUT_DIM;",
            "uint expert = uint(group_ids[route]);",
            "float acc = 0.0f;",
            "uint weight_base = (expert * OUT_DIM + out_col) * IN_DIM;",
            "uint input_base = route * IN_DIM;",
            "uint scale_base = (expert * SCALE_OUT + (out_col / 128)) * SCALE_COLS;",
            "for (uint k = 0; k < IN_DIM; ++k) {",
            "  uint8_t raw = weight[weight_base + k];",
            "  float w = fp8_e4m3_to_float(raw);",
            "  float s = float(scale[scale_base + (k / 128)]);",
            "  acc += float(input[input_base + k]) * w * s;",
            "}",
            "out[elem] = acc;"
        ),
        FP8_METAL_HEADER,
        true,
        false,
    )
}

const FP8_METAL_HEADER: &str = concat!(
    "float fp8_e4m3_to_float(uint8_t bits) {",
    "  uint16_t v = uint16_t(bits & 127) << 7;",
    "  half converted = as_type<half>(v);",
    "  converted *= 256.0h;",
    "  return (bits & 128) ? -float(converted) : float(converted);",
    "}\n",
);

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

    fn prefill_state_dependency(&self, stream: &Stream) -> Result<Option<Array>, Exception> {
        let mut dependency: Option<Array> = None;
        for layer in &self.layers {
            match layer {
                LayerCache::FullAttention(cache) => {
                    for array in cache.arrays() {
                        let term = array.sum(None, stream)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term, stream)?,
                            None => term,
                        });
                    }
                }
                LayerCache::LinearAttention(cache) => {
                    if let Some(conv_state) = &cache.conv_state {
                        let term = conv_state.sum(None, stream)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term, stream)?,
                            None => term,
                        });
                    }
                    if let Some(recurrent_state) = &cache.recurrent_state {
                        let term = recurrent_state.sum(None, stream)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term, stream)?,
                            None => term,
                        });
                    }
                }
            }
        }
        dependency
            .map(|dependency| dependency.multiply(Array::from_f32(0.0), stream))
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
    pub fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let variance = safemlx::ops::mean_axis(&x.square(stream)?, -1, true, stream)?;
        let normalized = x.multiply(
            safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
            stream,
        )?;
        let scale = self.weight.as_ref().add(Array::from_f32(1.0), stream)?;
        normalized.multiply(scale, stream)
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
    pub fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array, gate: &Array, stream: &Stream) -> Result<Array, Exception> {
        let variance = safemlx::ops::mean_axis(&x.square(stream)?, -1, true, stream)?;
        let normalized = x.multiply(
            safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
            stream,
        )?;
        normalized
            .multiply(&*self.weight, stream)?
            .multiply(silu(gate.clone(), stream)?, stream)
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
    pub q_proj: QwenLinear,
    #[param]
    pub k_proj: QwenLinear,
    #[param]
    pub v_proj: QwenLinear,
    #[param]
    pub o_proj: QwenLinear,
    #[param]
    pub q_norm: Qwen3NextRmsNorm,
    #[param]
    pub k_norm: Qwen3NextRmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl FullAttention {
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let hidden = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let fp8 = args.uses_fp8();
        let q_proj = QwenLinear::new(
            hidden,
            n_heads * head_dim * 2,
            args.attention_bias,
            fp8,
            stream,
        )?;
        let k_proj = QwenLinear::new(
            hidden,
            n_kv_heads * head_dim,
            args.attention_bias,
            fp8,
            stream,
        )?;
        let v_proj = QwenLinear::new(
            hidden,
            n_kv_heads * head_dim,
            args.attention_bias,
            fp8,
            stream,
        )?;
        let o_proj = QwenLinear::new(n_heads * head_dim, hidden, args.attention_bias, fp8, stream)?;
        let rope_config = args.rope_config();
        let rope = initialize_rope(
            args.rope_dims(),
            args.rope_theta(),
            false,
            &rope_config,
            args.max_position_embeddings,
            stream,
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
            q_norm: Qwen3NextRmsNorm::new(head_dim, args.rms_norm_eps, stream)?,
            k_norm: Qwen3NextRmsNorm::new(head_dim, args.rms_norm_eps, stream)?,
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
    fn forward(
        &mut self,
        input: FullAttentionInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let FullAttentionInput { x, mask, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let q_proj = self
            .q_proj
            .forward(x, stream)?
            .reshape(&[B, L, self.n_heads, 2 * self.head_dim], stream)?;
        let query = q_proj.try_index_device((.., .., .., ..self.head_dim), stream)?;
        let gate = q_proj
            .try_index_device((.., .., .., self.head_dim..), stream)?
            .reshape(&[B, L, self.n_heads * self.head_dim], stream)?;
        let mut query = self
            .q_norm
            .forward(&query, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut key = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x, stream)?
                    .reshape(&[B, L, self.n_kv_heads, self.head_dim], stream)?,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut value = self
            .v_proj
            .forward(x, stream)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;

        if let Some(cache) = cache.as_mut() {
            let offset = cache.offset();
            query = self.rope.forward(
                nn::RopeInputBuilder::new(&query).offset(offset).build()?,
                stream,
            )?;
            key = self.rope.forward(
                nn::RopeInputBuilder::new(&key).offset(offset).build()?,
                stream,
            )?;
            (key, value) = cache.update_and_fetch(key, value, stream)?;
        } else {
            query = self.rope.forward(nn::RopeInput::new(&query), stream)?;
            key = self.rope.forward(nn::RopeInput::new(&key), stream)?;
        }

        let out = crate::utils::scaled_dot_product_attention(
            query, key, value, cache, self.scale, mask, stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[B, L, -1], stream)?
        .multiply(sigmoid(gate, stream)?, stream)?;
        self.o_proj.forward(&out, stream)
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
    pub fn new(channels: i32, kernel_size: i32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[channels, 1, kernel_size], Dtype::Float32, stream)?,
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
    pub in_proj_qkv: QwenLinear,
    #[param]
    pub in_proj_z: QwenLinear,
    #[param]
    pub in_proj_b: QwenLinear,
    #[param]
    pub in_proj_a: QwenLinear,
    #[param]
    pub dt_bias: Param<Array>,
    #[param]
    pub A_log: Param<Array>,
    #[param]
    pub norm: Qwen3NextRmsNormGated,
    #[param]
    pub out_proj: QwenLinear,
}

impl LinearAttention {
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
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
            conv1d: DepthwiseConv1d::new(conv_dim, args.linear_conv_kernel_dim, stream)?,
            in_proj_qkv: QwenLinear::new(
                args.hidden_size,
                projection_size_qkv,
                false,
                args.uses_fp8(),
                stream,
            )?,
            in_proj_z: QwenLinear::new(
                args.hidden_size,
                value_dim,
                false,
                args.uses_fp8(),
                stream,
            )?,
            in_proj_b: QwenLinear::new(args.hidden_size, num_v_heads, false, false, stream)?,
            in_proj_a: QwenLinear::new(args.hidden_size, num_v_heads, false, false, stream)?,
            dt_bias: Param::new(Array::from_slice(
                &vec![1.0f32; num_v_heads as usize],
                &[num_v_heads],
            )),
            A_log: Param::new(Array::from_slice(
                &vec![0.0f32; num_v_heads as usize],
                &[num_v_heads],
            )),
            norm: Qwen3NextRmsNormGated::new(head_v_dim, args.rms_norm_eps, stream)?,
            out_proj: QwenLinear::new(value_dim, args.hidden_size, false, args.uses_fp8(), stream)?,
        })
    }

    #[allow(non_snake_case)]
    fn depthwise_causal_conv(
        &self,
        mixed_qkv: &Array,
        cache: Option<&mut LinearAttentionCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = mixed_qkv.shape();
        let B = shape[0];
        let L = shape[1];
        let C = shape[2];
        let state_len = self.conv_kernel_size - 1;
        let state = cache
            .as_ref()
            .and_then(|cache| cache.conv_state.clone())
            .unwrap_or(zeros::<f32>(&[B, state_len, C], stream)?);
        let padded = concatenate_axis(&[state, mixed_qkv.clone()], 1, stream)?;
        if let Some(cache) = cache {
            cache.conv_state = Some(padded.try_index_device((.., L.., ..), stream)?);
            cache.offset += L;
        }

        if L > 1 {
            let weight = self.conv1d.weight.swap_axes(1, 2, stream)?;
            let out = conv1d(&padded, &weight, Some(1), Some(0), Some(1), Some(C), stream)?;
            return silu(out, stream);
        }

        let mut out: Option<Array> = None;
        for k in 0..self.conv_kernel_size {
            let window = padded.try_index_device((.., k..k + L, ..), stream)?;
            let weight = self
                .conv1d
                .weight
                .try_index_device((.., 0, k), stream)?
                .reshape(&[1, 1, C], stream)?;
            let term = window.multiply(weight, stream)?;
            out = Some(match out {
                Some(acc) => acc.add(term, stream)?,
                None => term,
            });
        }
        silu(out.expect("conv kernel must have at least one tap"), stream)
    }

    fn l2norm(x: Array, stream: &Stream) -> Result<Array, Exception> {
        let denom =
            sum_axis(&x.square(stream)?, -1, true, stream)?.add(Array::from_f32(1e-6), stream)?;
        x.multiply(safemlx::ops::rsqrt(denom, stream)?, stream)
    }

    fn recurrent_delta_kernels() -> Result<RecurrentScanKernel, Exception> {
        Ok(RecurrentScanKernel::new(
            StatefulMetalKernel::new(
                "qwen35_moe_recurrent_decode",
                ["state", "query", "key", "value", "g", "beta"],
                ["out", "state_out"],
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
            )?,
            StatefulMetalKernel::new(
                "qwen35_moe_recurrent_prefill",
                ["state", "query", "key", "value", "g", "beta"],
                ["out", "state_out"],
                concat!(
                    "uint elem = thread_position_in_grid.x;",
                    "uint vd = elem % VD;",
                    "uint group = elem / VD;",
                    "uint h = group % H;",
                    "uint b = group / H;",
                    "uint state_base = group * KD * VD;",
                    "for (uint t = 0; t < L; ++t) {",
                    "  uint gh_idx = (b * L + t) * H + h;",
                    "  uint vec_base = gh_idx * KD;",
                    "  uint value_base = gh_idx * VD;",
                    "  float gate = metal::exp(g[gh_idx]);",
                    "  float kv_mem = 0.0f;",
                    "  for (uint kd = 0; kd < KD; ++kd) {",
                    "    uint state_idx = state_base + kd * VD + vd;",
                    "    float prev = (t == 0) ? float(state[state_idx]) : float(state_out[state_idx]);",
                    "    kv_mem += prev * gate * float(key[vec_base + kd]);",
                    "  }",
                    "  float delta = (float(value[value_base + vd]) - kv_mem) * float(beta[gh_idx]);",
                    "  float acc = 0.0f;",
                    "  for (uint kd = 0; kd < KD; ++kd) {",
                    "    uint state_idx = state_base + kd * VD + vd;",
                    "    float prev = (t == 0) ? float(state[state_idx]) : float(state_out[state_idx]);",
                    "    float updated = prev * gate + float(key[vec_base + kd]) * delta;",
                    "    state_out[state_idx] = updated;",
                    "    acc += updated * float(query[vec_base + kd]);",
                    "  }",
                    "  out[value_base + vd] = acc;",
                    "}"
                ),
                "",
                true,
                false,
            )?,
        ))
    }

    fn recurrent_delta_decode_kernel(
        state: &Array,
        query: &Array,
        key: &Array,
        value: &Array,
        g: &Array,
        beta: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let shape = state.shape();
        let b = shape[0];
        let h = shape[1];
        let kd = shape[2];
        let vd = shape[3];
        let state = state.as_dtype(Dtype::Float32, stream)?;
        let query = query.as_dtype(Dtype::Float32, stream)?;
        let key = key.as_dtype(Dtype::Float32, stream)?;
        let value = value.as_dtype(Dtype::Float32, stream)?;
        let g = g.as_dtype(Dtype::Float32, stream)?;
        let beta = beta.as_dtype(Dtype::Float32, stream)?;

        let output = RECURRENT_DELTA_KERNELS.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(Self::recurrent_delta_kernels()?);
            }
            let config = MetalKernelConfig::new()
                .with_template_arg_int("KD", kd)
                .with_template_arg_int("VD", vd)
                .with_grid([b * h * vd, 1, 1])
                .with_thread_group([256, 1, 1])
                .with_output_arg([b, 1, h, vd], Dtype::Float32)
                .with_output_arg([b, h, kd, vd], Dtype::Float32);
            cell.borrow()
                .as_ref()
                .expect("recurrent delta kernels initialized")
                .decode_device([&state, &query, &key, &value, &g, &beta], &config, stream)
        })?;

        let (out, state) = output.into_tuple();
        Ok((state, out))
    }

    fn recurrent_delta_prefill_kernel(
        state: &Array,
        query: &Array,
        key: &Array,
        value: &Array,
        g: &Array,
        beta: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let shape = query.shape();
        let b = shape[0];
        let l = shape[1];
        let h = shape[2];
        let kd = shape[3];
        let vd = value.shape()[3];
        let state = state.as_dtype(Dtype::Float32, stream)?;
        let query = query.as_dtype(Dtype::Float32, stream)?;
        let key = key.as_dtype(Dtype::Float32, stream)?;
        let value = value.as_dtype(Dtype::Float32, stream)?;
        let g = g.as_dtype(Dtype::Float32, stream)?;
        let beta = beta.as_dtype(Dtype::Float32, stream)?;

        let output = RECURRENT_DELTA_KERNELS.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(Self::recurrent_delta_kernels()?);
            }
            let config = MetalKernelConfig::new()
                .with_template_arg_int("L", l)
                .with_template_arg_int("H", h)
                .with_template_arg_int("KD", kd)
                .with_template_arg_int("VD", vd)
                .with_grid([b * h * vd, 1, 1])
                .with_thread_group([256, 1, 1])
                .with_output_arg([b, l, h, vd], Dtype::Float32)
                .with_output_arg([b, h, kd, vd], Dtype::Float32);
            cell.borrow()
                .as_ref()
                .expect("recurrent delta kernels initialized")
                .prefill_device([&state, &query, &key, &value, &g, &beta], &config, stream)
        })?;

        let (out, state) = output.into_tuple();
        Ok((state, out))
    }

    fn recurrent_delta_prefill_scan_chunked(
        mut state: Array,
        query: &Array,
        key: &Array,
        value: &Array,
        g: &Array,
        beta: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let l = query.shape()[1];
        let chunk_tokens = if l <= RECURRENT_PREFILL_SHORT_SCAN_TOKENS {
            RECURRENT_PREFILL_SHORT_SCAN_TOKENS
        } else if l <= 256 {
            RECURRENT_PREFILL_MEDIUM_SCAN_TOKENS
        } else {
            RECURRENT_PREFILL_LONG_SCAN_TOKENS
        };
        let mut outs = Vec::with_capacity(((l + chunk_tokens - 1) / chunk_tokens) as usize);
        let mut start = 0;
        while start < l {
            let end = (start + chunk_tokens).min(l);
            let query_chunk = query.try_index_device((.., start..end, .., ..), stream)?;
            let key_chunk = key.try_index_device((.., start..end, .., ..), stream)?;
            let value_chunk = value.try_index_device((.., start..end, .., ..), stream)?;
            let g_chunk = g.try_index_device((.., start..end, ..), stream)?;
            let beta_chunk = beta.try_index_device((.., start..end, ..), stream)?;
            let (new_state, out) = Self::recurrent_delta_prefill_kernel(
                &state,
                &query_chunk,
                &key_chunk,
                &value_chunk,
                &g_chunk,
                &beta_chunk,
                stream,
            )?;
            state = new_state;
            outs.push(out);
            start = end;
        }

        Ok((state, concatenate_axis(&outs, 1, stream)?))
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
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = query.shape();
        let B = shape[0];
        let L = shape[1];
        let H = shape[2];
        let KD = shape[3];
        let VD = value.shape()[3];
        let scale = (KD as f32).sqrt().recip();
        let query = query.multiply(Array::from_f32(scale), stream)?;
        let state = cache
            .as_ref()
            .and_then(|cache| cache.recurrent_state.clone())
            .unwrap_or(zeros::<f32>(&[B, H, KD, VD], stream)?);

        if L == 1 {
            let q_t = query.try_index_device((.., 0, .., ..), stream)?;
            let k_t = key.try_index_device((.., 0, .., ..), stream)?;
            let v_t = value.try_index_device((.., 0, .., ..), stream)?;
            let g_t = g.try_index_device((.., 0, ..), stream)?;
            let beta_t = beta.try_index_device((.., 0, ..), stream)?;
            let (new_state, out_t) = Self::recurrent_delta_decode_kernel(
                &state, &q_t, &k_t, &v_t, &g_t, &beta_t, stream,
            )?;
            if let Some(cache) = cache {
                cache.recurrent_state = Some(new_state);
            }
            return Ok(out_t);
        }

        let (new_state, out) = Self::recurrent_delta_prefill_scan_chunked(
            state, &query, &key, &value, &g, &beta, stream,
        )?;
        if let Some(cache) = cache {
            cache.recurrent_state = Some(new_state);
        }
        Ok(out)
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
    fn forward(
        &mut self,
        input: LinearAttentionInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let LinearAttentionInput { x, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let mixed_qkv = self.in_proj_qkv.forward(x, stream)?;
        let z = self
            .in_proj_z
            .forward(x, stream)?
            .reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        let b = self.in_proj_b.forward(x, stream)?;
        let a = self.in_proj_a.forward(x, stream)?;
        let mixed_qkv = self.depthwise_causal_conv(&mixed_qkv, cache.as_deref_mut(), stream)?;
        let query = mixed_qkv
            .try_index_device((.., .., ..self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        let key = mixed_qkv
            .try_index_device((.., .., self.key_dim..2 * self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        let mut value = mixed_qkv
            .try_index_device((.., .., 2 * self.key_dim..), stream)?
            .reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        let mut query = Self::l2norm(query, stream)?;
        let mut key = Self::l2norm(key, stream)?;
        let beta = sigmoid(b, stream)?;
        let dt_bias = self.dt_bias.reshape(&[1, 1, self.num_v_heads], stream)?;
        let g = nn::softplus(a.add(dt_bias, stream)?, stream)?.multiply(
            exp(self.A_log.as_ref(), stream)?.multiply(Array::from_f32(-1.0), stream)?,
            stream,
        )?;

        let repeats = self.num_v_heads / self.num_k_heads;
        if repeats > 1 {
            let expanded_query = query.try_index_device((.., .., .., NewAxis, ..), stream)?;
            query = broadcast_to(
                &expanded_query,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
            let expanded_key = key.try_index_device((.., .., .., NewAxis, ..), stream)?;
            key = broadcast_to(
                &expanded_key,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
        }

        value = value.as_dtype(x.dtype(), stream)?;
        let core = self.recurrent_delta_rule(query, key, value, g, beta, cache, stream)?;
        let z_shape = z.shape().to_vec();
        let core = core.reshape(&[-1, self.head_v_dim], stream)?;
        let z = z.reshape(&[-1, self.head_v_dim], stream)?;
        let out = self
            .norm
            .forward(&core, &z, stream)?
            .reshape(&z_shape, stream)?
            .reshape(&[B, L, self.value_dim], stream)?;
        self.out_proj.forward(&out, stream)
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
    pub gate_proj: QwenLinear,
    #[param]
    pub up_proj: QwenLinear,
    #[param]
    pub down_proj: QwenLinear,
}

impl Mlp {
    fn new(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        fp8: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QwenLinear::new(dim, hidden_dim, bias, fp8, stream)?,
            up_proj: QwenLinear::new(dim, hidden_dim, bias, fp8, stream)?,
            down_proj: QwenLinear::new(hidden_dim, dim, bias, fp8, stream)?,
        })
    }

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let down_proj_input = silu(self.gate_proj.forward(input, stream)?, stream)?
            .multiply(self.up_proj.forward(input, stream)?, stream)?;
        self.down_proj.forward(&down_proj_input, stream)
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
    pub use_fp8: bool,
    #[param]
    pub gate_up_proj: Param<Array>,
    #[param]
    pub gate_up_proj_scale_inv: Param<Option<Array>>,
    #[param]
    pub down_proj: Param<Array>,
    #[param]
    pub down_proj_scale_inv: Param<Option<Array>>,
}

impl Experts {
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let expert_weight_dtype = if args.uses_fp8() {
            Dtype::Uint8
        } else {
            Dtype::Float32
        };
        Ok(Self {
            num_experts: args.num_experts,
            hidden_dim: args.hidden_size,
            intermediate_dim: args.moe_intermediate_size,
            use_fp8: args.uses_fp8(),
            gate_up_proj: Param::<Array>::unloaded(
                &[
                    args.num_experts,
                    2 * args.moe_intermediate_size,
                    args.hidden_size,
                ],
                expert_weight_dtype,
                stream,
            )?,
            gate_up_proj_scale_inv: if args.uses_fp8() {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        args.num_experts,
                        ceil_div(2 * args.moe_intermediate_size, 128),
                        ceil_div(args.hidden_size, 128),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            down_proj: Param::<Array>::unloaded(
                &[
                    args.num_experts,
                    args.hidden_size,
                    args.moe_intermediate_size,
                ],
                expert_weight_dtype,
                stream,
            )?,
            down_proj_scale_inv: if args.uses_fp8() {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        args.num_experts,
                        ceil_div(args.hidden_size, 128),
                        ceil_div(args.moe_intermediate_size, 128),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if self.use_fp8 {
            return self.forward_expert_major_chunk(
                hidden_states,
                top_k_index,
                top_k_weights,
                stream,
            );
        }

        let num_tokens = hidden_states.shape()[0];
        let top_k = top_k_index.shape()[1];
        let selected_gate_up = self
            .gate_up_proj
            .as_ref()
            .take_axis(top_k_index, 0, stream)?;
        let hidden = hidden_states.try_index_device((.., NewAxis, NewAxis, ..), stream)?;
        let gate_up = matmul(&hidden, selected_gate_up.swap_axes(-1, -2, stream)?, stream)?
            .reshape(&[num_tokens, top_k, 2 * self.intermediate_dim], stream)?;
        let gate = gate_up.try_index_device((.., .., ..self.intermediate_dim), stream)?;
        let up = gate_up.try_index_device((.., .., self.intermediate_dim..), stream)?;
        let current = silu(gate, stream)?.multiply(up, stream)?;

        let selected_down = self.down_proj.as_ref().take_axis(top_k_index, 0, stream)?;
        let current = matmul(
            current.try_index_device((.., .., NewAxis, ..), stream)?,
            selected_down.swap_axes(-1, -2, stream)?,
            stream,
        )?
        .reshape(&[num_tokens, top_k, self.hidden_dim], stream)?;
        let weighted = current.multiply(
            top_k_weights.try_index_device((.., .., NewAxis), stream)?,
            stream,
        )?;
        sum_axis(&weighted, -2, false, stream)
    }

    pub fn forward_chunked(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        if num_tokens <= ROUTED_EXPERT_CHUNK_THRESHOLD {
            return self.forward(hidden_states, top_k_index, top_k_weights, stream);
        }

        let mut outputs = Vec::with_capacity(
            ((num_tokens + ROUTED_EXPERT_CHUNK_TOKENS - 1) / ROUTED_EXPERT_CHUNK_TOKENS)
                .try_into()
                .expect("number of MoE chunks must fit in usize"),
        );
        let mut start = 0;
        while start < num_tokens {
            let end = (start + ROUTED_EXPERT_CHUNK_TOKENS).min(num_tokens);
            let hidden_chunk = hidden_states.try_index_device((start..end, ..), stream)?;
            let expert_chunk = top_k_index.try_index_device((start..end, ..), stream)?;
            let weight_chunk = top_k_weights.try_index_device((start..end, ..), stream)?;
            outputs.push(self.forward_expert_major_chunk(
                &hidden_chunk,
                &expert_chunk,
                &weight_chunk,
                stream,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 0, stream)
    }

    fn forward_expert_major_chunk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let gate_up = if let Some(scale) = self.gate_up_proj_scale_inv.as_ref() {
            grouped_fp8_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                scale,
                &plan.sorted_group_ids,
                stream,
            )?
        } else {
            let gate_up_weights = self.gate_up_proj.as_ref().swap_axes(-1, -2, stream)?;
            grouped_matmul(
                &hidden,
                &gate_up_weights,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        let gate = gate_up.try_index_device((.., ..self.intermediate_dim), stream)?;
        let up = gate_up.try_index_device((.., self.intermediate_dim..), stream)?;
        let current = silu(gate, stream)?.multiply(up, stream)?;

        let current = if let Some(scale) = self.down_proj_scale_inv.as_ref() {
            grouped_fp8_linear(
                &current,
                self.down_proj.as_ref(),
                scale,
                &plan.sorted_group_ids,
                stream,
            )?
        } else {
            let down_weights = self.down_proj.as_ref().swap_axes(-1, -2, stream)?;
            grouped_matmul(
                &current,
                &down_weights,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        let weights = gather_route_values(top_k_weights, &plan, stream)?
            .try_index_device((.., NewAxis), stream)?;
        let weighted = current.multiply(weights, stream)?;
        segment_sum_by_index(weighted, &plan.token_indices, num_tokens, stream)
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
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            top_k: args.num_experts_per_tok,
            num_experts: args.num_experts,
            norm_topk_prob: args.norm_topk_prob,
            weight: Param::<Array>::unloaded(
                &[args.num_experts, args.hidden_size],
                Dtype::Float32,
                stream,
            )?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let router_logits = matmul(hidden_states, self.weight.transpose(stream)?, stream)?;
        let router_probs = softmax_axis(&router_logits, -1, true, stream)?;
        let top_k_index = argpartition_axis(&router_probs, -self.top_k, -1, stream)?
            .try_index_device((.., -self.top_k..), stream)?;
        let mut top_k_weights = take_along_axis(&router_probs, &top_k_index, -1, stream)?;
        top_k_weights =
            top_k_weights.divide(sum_axis(&top_k_weights, -1, true, stream)?, stream)?;
        Ok((
            top_k_index,
            top_k_weights.as_dtype(router_logits.dtype(), stream)?,
        ))
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
    pub shared_expert_gate: QwenLinear,
}

impl SparseMoeBlock {
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            gate: TopKRouter::new(args, stream)?,
            experts: Experts::new(args, stream)?,
            shared_expert: Mlp::new(
                args.hidden_size,
                args.shared_expert_intermediate_size,
                false,
                args.uses_fp8(),
                stream,
            )?,
            shared_expert_gate: QwenLinear::new(args.hidden_size, 1, false, false, stream)?,
        })
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let shape = hidden_states.shape();
        let B = shape[0];
        let L = shape[1];
        let H = shape[2];
        let flat = hidden_states.reshape(&[-1, H], stream)?;
        let shared = self.shared_expert.forward(&flat, stream)?.multiply(
            sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?,
            stream,
        )?;
        profile_array(PerfComponent::MoeShared, &shared)?;
        let (selected_experts, routing_weights) = self.gate.forward(&flat, stream)?;
        profile_arrays(
            PerfComponent::MoeRouter,
            &[&selected_experts, &routing_weights],
        )?;
        let routed =
            self.experts
                .forward_chunked(&flat, &selected_experts, &routing_weights, stream)?;
        profile_array(PerfComponent::MoeRouted, &routed)?;
        let output = routed.add(shared, stream)?.reshape(&[B, L, H], stream)?;
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
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        let layer_type = args.layer_type(layer_idx);
        Ok(Self {
            layer_type,
            self_attn: if layer_type == LayerType::FullAttention {
                Some(FullAttention::new(args, stream)?)
            } else {
                None
            },
            linear_attn: if layer_type == LayerType::LinearAttention {
                Some(LinearAttention::new(args, stream)?)
            } else {
                None
            },
            mlp: SparseMoeBlock::new(args, stream)?,
            input_layernorm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
            post_attention_layernorm: Qwen3NextRmsNorm::new(
                args.hidden_size,
                args.rms_norm_eps,
                stream,
            )?,
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

    fn forward(
        &mut self,
        input: BlockInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let BlockInput { x, mask, cache } = input;
        let residual = x;
        let h = self.input_layernorm.forward(x, stream)?;
        let h = match (self.layer_type, cache) {
            (LayerType::FullAttention, Some(LayerCache::FullAttention(cache))) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerType::FullAttention, _) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerType::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerType::LinearAttention, _) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(LinearAttentionInput { x: &h, cache: None }, stream)?,
        };
        match self.layer_type {
            LayerType::FullAttention => profile_array(PerfComponent::FullAttention, &h)?,
            LayerType::LinearAttention => profile_array(PerfComponent::LinearAttention, &h)?,
        }
        let h = residual.add(h, stream)?;
        let residual = h.clone();
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        let h = self.mlp.forward(&post_normed, stream)?;
        residual.add(h, stream)
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
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let embed_tokens =
            nn::Embedding::unloaded(args.vocab_size, args.hidden_size, Dtype::Float32, stream)?;
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new(args, idx as usize, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens,
            layers,
            norm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
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

    fn forward(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            mut cache,
        } = input;
        let mut h = self.embed_tokens.forward(inputs, stream)?;
        profile_array(PerfComponent::Embed, &h)?;
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => {
                let offset = cache.as_ref().map(|cache| cache.offset()).unwrap_or(0);
                if h.shape()[1] > 1 {
                    match create_attention_mask(&h, &offset_cache(offset), Some(true), stream)? {
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
                h = layer.forward(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: Some(layer_cache),
                    },
                    stream,
                )?;
            }
        } else {
            for layer in &mut self.layers {
                h = layer.forward(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: None,
                    },
                    stream,
                )?;
            }
        }
        let h = self.norm.forward(&h, stream)?;
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
        _stream: &Stream,
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
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let model = Qwen35MoeTextModel::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(nn::Linear::unloaded(
                args.hidden_size,
                args.vocab_size,
                false,
                Dtype::Float32,
                stream,
            )?)
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

    fn reject_multimodal_tokens(&self, inputs: &Array, stream: &Stream) -> Result<(), Exception> {
        for (name, token_id) in [
            ("image", self.image_token_id),
            ("video", self.video_token_id),
        ] {
            if let Some(token_id) = token_id {
                let contains = inputs
                    .eq(Array::from_int(token_id), stream)?
                    .max(None, stream)?
                    .item::<bool>(&stream);
                if contains {
                    return Err(Exception::custom(format!(
                        "qwen3_5_moe text-generation support does not accept {name} tokens"
                    )));
                }
            }
        }
        Ok(())
    }

    fn project_logits(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = project_logits_dense(
            &mut self.lm_head,
            &self.model.embed_tokens,
            hidden_states,
            stream,
        )?;
        profile_array(PerfComponent::LmHead, &logits)?;
        Ok(logits)
    }

    fn forward_logits(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.reject_multimodal_tokens(input.inputs, stream)?;
        let hidden_states = self.model.forward(input, stream)?;
        let hidden_states = if last_token_only {
            hidden_states.try_index_device((.., -1, ..), stream)?
        } else {
            hidden_states
        };
        self.project_logits(&hidden_states, stream)
    }
}

impl Module<ModelInput<'_>> for Model {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        self.forward_logits(input, false, stream)
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
    args.quantization_config = config
        .quantization_config
        .or_else(|| args.quantization_config.clone());
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

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    let value = config.clone();
    let config: TopLevelConfig = serde_json::from_value(value.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid qwen3_5_moe config: {error}"))
    })?;
    match config.model_type.as_str() {
        "qwen3_5_moe" => {
            let text_config = config.text_config.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "qwen3_5_moe config is missing text_config".to_string(),
                )
            })?;
            if let Some(quantization_config) = &text_config.quantization_config {
                quantization_config.validate_supported()?;
            }
        }
        "qwen3_5_moe_text" => {
            let args = serde_json::from_value::<ModelArgs>(value).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid qwen3_5_moe_text config: {error}"))
            })?;
            if let Some(quantization_config) = &args.quantization_config {
                quantization_config.validate_supported()?;
            }
        }
        other => {
            return Err(Error::UnsupportedModelType(other.to_string()));
        }
    }
    if let Some(quantization_config) = &config.quantization_config {
        quantization_config.validate_supported()?;
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_qwen3_5_moe_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id) = get_qwen3_5_moe_model_args(model_dir)?;
    if let Some(quantization_config) = &args.quantization_config {
        quantization_config.validate_supported()?;
    }
    let uses_fp8 = args.quantization_config.is_some();
    let mut model = Model::new(args, image_token_id, video_token_id, stream)?;
    let config = qwen3_5_moe_strict_load_config();
    let mut report = StrictLoadReport::default();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            load_qwen3_5_moe_safetensors_strict(
                &mut model,
                model_dir.join(weight_file),
                weights_stream,
                stream,
                &config,
                &mut report,
                uses_fp8,
            )?;
        }
    } else {
        load_qwen3_5_moe_safetensors_strict(
            &mut model,
            model_dir.join("model.safetensors"),
            weights_stream,
            stream,
            &config,
            &mut report,
            uses_fp8,
        )?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

fn load_qwen3_5_moe_safetensors_strict(
    model: &mut Model,
    path: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    uses_fp8: bool,
) -> Result<(), Error> {
    let path = path.as_ref();
    if !uses_fp8 {
        return load_safetensors_strict(model, path, weights_stream, config, report);
    }

    let loaded = Array::load_safetensors(path, weights_stream)?;
    let loaded = transform_qwen3_5_moe_fp8_weights(loaded, &model.args, transform_stream)?;
    load_arrays_strict(model, loaded, config, report)
}

#[derive(Default)]
struct Fp8ExpertParts {
    gate: Option<Array>,
    gate_scale: Option<Array>,
    up: Option<Array>,
    up_scale: Option<Array>,
    down: Option<Array>,
    down_scale: Option<Array>,
}

fn transform_qwen3_5_moe_fp8_weights(
    loaded: HashMap<String, Array>,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut transformed = HashMap::with_capacity(loaded.len());
    let mut expert_parts: HashMap<(String, i32), Fp8ExpertParts> = HashMap::new();

    for (key, value) in &loaded {
        if let Some((prefix, expert, projection)) = parse_fp8_expert_projection_key(key) {
            let scale_key = fp8_scale_key(key);
            let scale = loaded.get(&scale_key).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 tensor '{key}' is missing sibling scale '{scale_key}'"
                ))
            })?;
            let parts = expert_parts.entry((prefix, expert)).or_default();
            match projection {
                Fp8ExpertProjection::Gate => {
                    parts.gate = Some(value.clone());
                    parts.gate_scale = Some(scale.clone());
                }
                Fp8ExpertProjection::Up => {
                    parts.up = Some(value.clone());
                    parts.up_scale = Some(scale.clone());
                }
                Fp8ExpertProjection::Down => {
                    parts.down = Some(value.clone());
                    parts.down_scale = Some(scale.clone());
                }
            }
            continue;
        }

        if parse_fp8_expert_scale_key(key).is_none() {
            transformed.insert(key.clone(), value.clone());
        }
    }

    let mut layer_prefixes = expert_parts
        .keys()
        .map(|(prefix, _)| prefix.clone())
        .collect::<Vec<_>>();
    layer_prefixes.sort();
    layer_prefixes.dedup();

    for prefix in layer_prefixes {
        let mut gate_up = Vec::with_capacity(args.num_experts as usize);
        let mut gate_up_scale = Vec::with_capacity(args.num_experts as usize);
        let mut down = Vec::with_capacity(args.num_experts as usize);
        let mut down_scale = Vec::with_capacity(args.num_experts as usize);
        for expert in 0..args.num_experts {
            let parts = expert_parts
                .remove(&(prefix.clone(), expert))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Qwen3.5-MoE FP8 checkpoint is missing expert {expert} for '{prefix}'"
                    ))
                })?;
            let gate = parts.gate.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.gate_proj.weight"
                ))
            })?;
            let gate_scale = parts.gate_scale.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.gate_proj.weight_scale_inv"
                ))
            })?;
            let up = parts.up.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.up_proj.weight"
                ))
            })?;
            let up_scale = parts.up_scale.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.up_proj.weight_scale_inv"
                ))
            })?;
            let down_proj = parts.down.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.down_proj.weight"
                ))
            })?;
            let down_proj_scale = parts.down_scale.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.down_proj.weight_scale_inv"
                ))
            })?;
            let gate_up_proj = concatenate_axis(&[gate, up], 0, stream)?;
            let gate_up_proj_scale = concatenate_axis(&[gate_scale, up_scale], 0, stream)?;
            gate_up.push(gate_up_proj);
            gate_up_scale.push(gate_up_proj_scale);
            down.push(down_proj);
            down_scale.push(down_proj_scale);
        }

        let gate_up_proj = stack_axis(&gate_up, 0, stream)?;
        let gate_up_proj_scale = stack_axis(&gate_up_scale, 0, stream)?;
        let down_proj = stack_axis(&down, 0, stream)?;
        let down_proj_scale = stack_axis(&down_scale, 0, stream)?;
        eval([
            &gate_up_proj,
            &gate_up_proj_scale,
            &down_proj,
            &down_proj_scale,
        ])?;
        transformed.insert(format!("{prefix}.gate_up_proj"), gate_up_proj);
        transformed.insert(
            format!("{prefix}.gate_up_proj_scale_inv"),
            gate_up_proj_scale,
        );
        transformed.insert(format!("{prefix}.down_proj"), down_proj);
        transformed.insert(format!("{prefix}.down_proj_scale_inv"), down_proj_scale);
    }

    Ok(transformed)
}

#[derive(Debug, Clone, Copy)]
enum Fp8ExpertProjection {
    Gate,
    Up,
    Down,
}

fn parse_fp8_expert_projection_key(key: &str) -> Option<(String, i32, Fp8ExpertProjection)> {
    let (prefix, rest) = key.split_once(".mlp.experts.")?;
    let mut parts = rest.split('.');
    let expert = parts.next()?.parse().ok()?;
    let projection = match parts.next()? {
        "gate_proj" => Fp8ExpertProjection::Gate,
        "up_proj" => Fp8ExpertProjection::Up,
        "down_proj" => Fp8ExpertProjection::Down,
        _ => return None,
    };
    if parts.next()? != "weight" || parts.next().is_some() {
        return None;
    }
    Some((format!("{prefix}.mlp.experts"), expert, projection))
}

fn parse_fp8_expert_scale_key(key: &str) -> Option<(String, i32, Fp8ExpertProjection)> {
    let weight_key = key
        .strip_suffix(".weight_scale_inv")
        .map(|prefix| format!("{prefix}.weight"))?;
    parse_fp8_expert_projection_key(&weight_key)
}

fn fp8_scale_key(key: &str) -> String {
    key.strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.weight_scale_inv"))
        .unwrap_or_else(|| format!("{key}_scale_inv"))
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

impl CausalLm<Cache> for Model {
    fn prefill_logits(
        &mut self,
        prompt_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits(
            ModelInput {
                inputs: prompt_tokens,
                mask: None,
                cache: Some(cache),
            },
            true,
            stream,
        )
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.forward(
            ModelInput {
                inputs: input_tokens,
                mask: None,
                cache: Some(cache),
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }

    fn adjust_prefill_logits(
        &mut self,
        mut logits: Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // Keep the first sampled token dependent on all prefill cache state while
        // avoiding a prompt-length vocabulary projection.
        if let Some(dependency) = cache.prefill_state_dependency(stream)? {
            profile_array(PerfComponent::PrefillStateDependency, &dependency)?;
            logits = logits.add(dependency, stream)?;
        }
        Ok(logits)
    }
}

pub type Generate<'a> = common::Generate<'a, Model, Cache>;

#[cfg(test)]
mod tests {
    use super::{
        default_layer_type, get_qwen3_5_moe_model_args, load_qwen3_5_moe_model,
        load_qwen3_5_moe_tokenizer, parse_fp8_expert_projection_key,
        qwen3_5_moe_strict_load_config, Fp8ExpertProjection, FullAttention, FullAttentionInput,
        LayerType, LinearAttention, LinearAttentionInput, Model, ModelArgs, SparseMoeBlock,
    };
    use crate::{
        error::Error,
        weights::{load_safetensors_strict, StrictLoadReport},
    };
    use safemlx::{
        module::{Module, ModuleParameters, Param},
        ops::indexing::{NewAxis, TryIndexOp},
        transforms::eval,
        Array, ExecutionContext,
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
            quantization_config: None,
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
    fn parses_top_level_fp8_quantization_config() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe",
              "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128]
              },
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 128,
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128
              }
            }"#,
        );
        let (args, _, _) = get_qwen3_5_moe_model_args(&dir).unwrap();
        let quantization_config = args.quantization_config.unwrap();
        assert_eq!(quantization_config.quant_method, "fp8");
        assert_eq!(quantization_config.fmt, "e4m3");
        assert_eq!(quantization_config.activation_scheme, "dynamic");
        assert_eq!(
            quantization_config.weight_block_size.as_deref(),
            Some(&[128, 128][..])
        );
        quantization_config.validate_supported().unwrap();
    }

    #[test]
    fn parses_fp8_split_expert_projection_keys() {
        let parsed = parse_fp8_expert_projection_key(
            "model.language_model.layers.3.mlp.experts.17.gate_proj.weight",
        )
        .unwrap();
        assert_eq!(parsed.0, "model.language_model.layers.3.mlp.experts");
        assert_eq!(parsed.1, 17);
        assert!(matches!(parsed.2, Fp8ExpertProjection::Gate));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_attention_forward_shape_smoke() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut attn = FullAttention::new(&args, stream).unwrap();
        let x = Array::zeros::<f32>(&[1, 2, args.hidden_size], stream).unwrap();
        let out = attn
            .forward(
                FullAttentionInput {
                    x: &x,
                    mask: None,
                    cache: None,
                },
                stream,
            )
            .unwrap();
        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn linear_attention_forward_shape_smoke() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let mut attn = LinearAttention::new(&args, stream).unwrap();
        let x = Array::zeros::<f32>(&[1, 2, args.hidden_size], stream).unwrap();
        let out = attn
            .forward(LinearAttentionInput { x: &x, cache: None }, stream)
            .unwrap();
        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn sparse_moe_forward_shape_smoke() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let mut moe = SparseMoeBlock::new(&args, stream).unwrap();
        let gate_values = (0..args.num_experts)
            .flat_map(|expert| {
                (0..args.hidden_size).map(move |hidden| ((expert + 1) * (hidden + 1)) as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        moe.gate.weight = Param::new(
            Array::from(gate_values.as_slice())
                .reshape(&[args.num_experts, args.hidden_size], stream)
                .unwrap(),
        );
        let input_values = (0..(2 * args.hidden_size))
            .map(|index| index as f32 * 0.01)
            .collect::<Vec<_>>();
        let x = Array::from(input_values.as_slice())
            .reshape(&[1, 2, args.hidden_size], stream)
            .unwrap();
        let out = moe.forward(&x, stream).unwrap();
        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn parameter_tree_matches_public_checkpoint_key_patterns() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention, LayerType::FullAttention]);
        let model = Model::new(args, Some(248056), Some(248057), stream).unwrap();
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
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let source = Model::new(args.clone(), None, None, stream).unwrap();
        let dir = temp_model_dir("{}");
        let weights_path = dir.join("model.safetensors");
        save_model_parameters(
            &weights_path,
            &source,
            |_| true,
            vec![
                (
                    "visual.extra.weight".to_string(),
                    Array::zeros::<f32>(&[1], stream).unwrap(),
                ),
                (
                    "model.vision_tower.extra.weight".to_string(),
                    Array::zeros::<f32>(&[1], stream).unwrap(),
                ),
                (
                    "mtp.extra.weight".to_string(),
                    Array::zeros::<f32>(&[1], stream).unwrap(),
                ),
            ],
        );

        let mut target = Model::new(args, None, None, stream).unwrap();
        let config = qwen3_5_moe_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_safetensors_strict(&mut target, &weights_path, stream, &config, &mut report).unwrap();
        report.finish(&target, &config).unwrap();
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_fails_on_missing_text_weight() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let source = Model::new(args.clone(), None, None, stream).unwrap();
        let dir = temp_model_dir("{}");
        let weights_path = dir.join("model.safetensors");
        save_model_parameters(
            &weights_path,
            &source,
            |key| key != "lm_head.weight",
            Vec::new(),
        );

        let mut target = Model::new(args, None, None, stream).unwrap();
        let config = qwen3_5_moe_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_safetensors_strict(&mut target, &weights_path, stream, &config, &mut report).unwrap();
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
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let source = Model::new(args.clone(), None, None, stream).unwrap();
        let dir = temp_model_dir("{}");
        let weights_path = dir.join("model.safetensors");
        save_model_parameters(
            &weights_path,
            &source,
            |_| true,
            vec![(
                "model.layers.0.linear_attn.unexpected.weight".to_string(),
                Array::zeros::<f32>(&[1], stream).unwrap(),
            )],
        );

        let mut target = Model::new(args, None, None, stream).unwrap();
        let config = qwen3_5_moe_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_safetensors_strict(&mut target, &weights_path, stream, &config, &mut report).unwrap();
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
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();
        let model_dir = cached_test_model_dir();
        let tokenizer = load_qwen3_5_moe_tokenizer(&model_dir).unwrap();
        let mut model = load_qwen3_5_moe_model(&model_dir, stream, weights_stream).unwrap();
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
            let prompt_tokens = Array::from(encoding.get_ids())
                .try_index_device(NewAxis, stream)
                .unwrap();
            let mut cache = model.new_cache();
            let mut tokens = Vec::new();
            let generate =
                super::Generate::new(&mut model, &mut cache, 0.0, &prompt_tokens, None, stream);
            for token in generate.take(expected_tokens.len()) {
                let token = token.unwrap();
                eval([&token]).unwrap();
                tokens.push(token.item::<u32>(&stream));
            }
            assert_eq!(tokens, expected_tokens, "prompt: {prompt}");
        }
    }
}
