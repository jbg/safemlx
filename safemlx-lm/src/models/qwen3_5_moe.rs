//! Qwen3.5 MoE text model implementation and loader.

use std::{cell::RefCell, collections::HashMap, path::Path, time::Instant};

use safemlx::{
    builder::Builder,
    error::Exception,
    fast::{MetalKernel, MetalKernelConfig, RecurrentScanKernel, StatefulMetalKernel},
    macros::ModuleParameters,
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{
        arange, broadcast_to, concatenate_axis, conv1d, dequantize, exp, gather_grouped_rows,
        gather_qmm, grouped_matmul,
        indexing::{NewAxis, TryIndexOp},
        matmul, quantized_matmul, quantized_packed_dimension, sigmoid, stack_axis, sum_axis,
        topk_route_plan, zeros, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
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
    inspection::{ActivationObserver, MoeRoutingObservation},
    models::{
        common::{
            self, attention_probabilities, project_logits_maybe_quantized, silu, CausalLm,
            TopKRouterScoreFunction,
        },
        input as runtime_input,
    },
    quantization::{quantize_tensor, AffineQuantization, QuantizedTensor},
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
    weights::{
        for_each_safetensor_array, load_array_quantized_strict, load_array_strict,
        load_arrays_strict, load_safetensors_strict, safetensors_files, StrictLoadConfig,
        StrictLoadReport,
    },
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Qwen3.5 MoE layer kind.
pub enum LayerType {
    /// Recurrent linear-attention layer.
    LinearAttention,
    /// Full self-attention layer.
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
    static FP8_LINEAR_SCALAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static FP8_GROUPED_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static FP8_GROUPED_LINEAR_SCALAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

const FP8_LINEAR_OUT_TILE: i32 = 16;
const FP8_TILED_ROW_THRESHOLD: i32 = 8;
const ROUTED_EXPERT_CHUNK_THRESHOLD: i32 = 64;
const ROUTED_EXPERT_CHUNK_TOKENS: i32 = 32;
const RECURRENT_PREFILL_SHORT_SCAN_TOKENS: i32 = 64;
const RECURRENT_PREFILL_MEDIUM_SCAN_TOKENS: i32 = 16;
const RECURRENT_PREFILL_LONG_SCAN_TOKENS: i32 = 32;

#[derive(Debug, Clone, Default)]
/// Profiling counters accumulated by Qwen3.5 MoE when profiling is enabled.
pub struct PerfStats {
    /// Time spent evaluating token embeddings.
    pub embed_s: f64,
    /// Time spent evaluating full-attention layers.
    pub full_attention_s: f64,
    /// Time spent evaluating linear-attention layers.
    pub linear_attention_s: f64,
    /// Time spent evaluating MoE routing.
    pub moe_router_s: f64,
    /// Time spent evaluating the shared expert.
    pub moe_shared_s: f64,
    /// Time spent evaluating routed experts.
    pub moe_routed_s: f64,
    /// Time spent combining MoE outputs.
    pub moe_combine_s: f64,
    /// Time spent evaluating final normalization.
    pub final_norm_s: f64,
    /// Time spent projecting hidden states to logits.
    pub lm_head_s: f64,
    /// Time spent materializing the prefill state dependency.
    pub prefill_state_dependency_s: f64,
}

impl PerfStats {
    /// Returns the sum of all profiled component durations.
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

/// Enables or disables per-thread Qwen3.5 MoE profiling.
pub fn set_perf_profiling(enabled: bool) {
    PERF_STATS.with(|stats| {
        *stats.borrow_mut() = enabled.then(PerfStats::default);
    });
}

/// Resets per-thread Qwen3.5 MoE profiling counters.
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

#[derive(Debug, Clone, Deserialize)]
/// Deserialized Qwen3.5 MoE text configuration used by this loader.
pub struct ModelArgs {
    #[serde(default = "default_text_model_type")]
    /// Effective text model type.
    pub model_type: String,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Number of full-attention query heads.
    pub num_attention_heads: i32,
    /// Number of full-attention key/value heads.
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    /// Full-attention head dimension.
    pub head_dim: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    #[serde(default = "default_rms_norm_eps")]
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    #[serde(default)]
    /// Whether full-attention projections include bias terms.
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    /// Activation function name from the config.
    pub hidden_act: String,
    #[serde(default = "default_linear_conv_kernel_dim")]
    /// Causal convolution kernel width in linear-attention layers.
    pub linear_conv_kernel_dim: i32,
    #[serde(default = "default_linear_key_head_dim")]
    /// Key head dimension in linear-attention layers.
    pub linear_key_head_dim: i32,
    #[serde(default = "default_linear_value_head_dim")]
    /// Value head dimension in linear-attention layers.
    pub linear_value_head_dim: i32,
    #[serde(default = "default_linear_num_key_heads")]
    /// Number of key heads in linear-attention layers.
    pub linear_num_key_heads: i32,
    #[serde(default = "default_linear_num_value_heads")]
    /// Number of value heads in linear-attention layers.
    pub linear_num_value_heads: i32,
    #[serde(default)]
    /// Dense SwiGLU intermediate size. Zero for MoE checkpoints.
    pub intermediate_size: i32,
    #[serde(default = "default_moe_intermediate_size")]
    /// Routed-expert intermediate size.
    pub moe_intermediate_size: i32,
    #[serde(default = "default_shared_expert_intermediate_size")]
    /// Shared-expert intermediate size.
    pub shared_expert_intermediate_size: i32,
    #[serde(default = "default_num_experts_per_tok")]
    /// Number of experts selected per token.
    pub num_experts_per_tok: i32,
    #[serde(default = "default_num_experts")]
    /// Total number of routed experts.
    pub num_experts: i32,
    #[serde(default)]
    /// Whether top-k routing probabilities are normalized.
    pub norm_topk_prob: bool,
    #[serde(default)]
    /// Layer-kind pattern.
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    /// RoPE parameter overrides.
    pub rope_parameters: Option<HashMap<String, Value>>,
    #[serde(default)]
    /// RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, Value>>,
    #[serde(default)]
    /// Optional FP8 quantization configuration.
    pub quantization_config: Option<QwenFp8QuantizationConfig>,
    #[serde(skip)]
    /// Exact GGUF affine settings keyed by runtime weight name.
    pub quantized_weight_configs: Option<HashMap<String, AffineQuantization>>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
/// FP8 quantization settings supported by the Qwen3.5 MoE loader.
pub struct QwenFp8QuantizationConfig {
    /// Quantization method, expected to be `fp8`.
    pub quant_method: String,
    /// FP8 format, expected to be `e4m3`.
    pub fmt: String,
    /// Activation quantization scheme, expected to be `dynamic`.
    pub activation_scheme: String,
    #[serde(default)]
    /// FP8 weight block size.
    pub weight_block_size: Option<Vec<i32>>,
    #[serde(default)]
    /// Module names excluded from quantization.
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
    vision_config: Option<VisionConfig>,
    #[serde(default)]
    quantization_config: Option<QwenFp8QuantizationConfig>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    image_token_id: Option<i32>,
    #[serde(default)]
    video_token_id: Option<i32>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
/// Qwen VL vision encoder configuration.
pub struct VisionConfig {
    #[serde(default = "default_vision_depth")]
    /// Number of vision transformer blocks.
    pub depth: i32,
    #[serde(default = "default_vision_hidden_size")]
    /// Vision transformer hidden size.
    pub hidden_size: i32,
    #[serde(default = "default_vision_hidden_act")]
    /// Vision MLP activation function.
    pub hidden_act: String,
    #[serde(default = "default_vision_intermediate_size")]
    /// Vision MLP intermediate size.
    pub intermediate_size: i32,
    #[serde(default = "default_vision_num_heads")]
    /// Number of vision attention heads.
    pub num_heads: i32,
    #[serde(default = "default_vision_num_position_embeddings")]
    /// Number of learned spatial position embeddings.
    pub num_position_embeddings: i32,
    #[serde(default = "default_vision_in_channels")]
    /// Number of input pixel channels.
    pub in_channels: i32,
    #[serde(default = "default_vision_patch_size")]
    /// Spatial patch size.
    pub patch_size: i32,
    #[serde(default = "default_vision_spatial_merge_size")]
    /// Spatial merge factor used before language-model insertion.
    pub spatial_merge_size: i32,
    #[serde(default = "default_vision_temporal_patch_size")]
    /// Temporal patch size.
    pub temporal_patch_size: i32,
    #[serde(default = "default_vision_window_size")]
    /// Window attention size from the public config.
    pub window_size: i32,
    #[serde(default = "default_vision_out_hidden_size")]
    /// Output hidden size projected into the language model space.
    pub out_hidden_size: i32,
    #[serde(default = "default_vision_fullatt_block_indexes")]
    /// Blocks configured for full attention in the reference implementation.
    pub fullatt_block_indexes: Vec<i32>,
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

fn default_vision_depth() -> i32 {
    32
}

fn default_vision_hidden_size() -> i32 {
    3584
}

fn default_vision_hidden_act() -> String {
    "silu".to_string()
}

fn default_vision_intermediate_size() -> i32 {
    3420
}

fn default_vision_num_heads() -> i32 {
    16
}

fn default_vision_num_position_embeddings() -> i32 {
    2304
}

fn default_vision_in_channels() -> i32 {
    3
}

fn default_vision_patch_size() -> i32 {
    14
}

fn default_vision_spatial_merge_size() -> i32 {
    2
}

fn default_vision_temporal_patch_size() -> i32 {
    2
}

fn default_vision_window_size() -> i32 {
    112
}

fn default_vision_out_hidden_size() -> i32 {
    3584
}

fn default_vision_fullatt_block_indexes() -> Vec<i32> {
    vec![7, 15, 23, 31]
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

    fn affine_quantization_for(&self, weight_name: &str) -> Option<AffineQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name).copied())
    }

    fn is_moe(&self) -> bool {
        self.num_experts > 0
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum QwenWeightFormat {
    Dense,
    Fp8,
    Affine(AffineQuantization),
}

impl QwenWeightFormat {
    fn for_text(args: &ModelArgs, affine: Option<AffineQuantization>) -> Self {
        match affine {
            Some(affine) => Self::Affine(affine),
            None if args.uses_fp8() => Self::Fp8,
            None => Self::Dense,
        }
    }

    fn affine(self) -> Option<AffineQuantization> {
        match self {
            Self::Affine(affine) => Some(affine),
            Self::Dense | Self::Fp8 => None,
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Linear layer that can hold dense, Qwen FP8, or MLX affine weights.
pub struct QwenLinear {
    /// Input feature dimension.
    pub input_dims: i32,
    /// Output feature dimension.
    pub output_dims: i32,
    #[param]
    /// Weight tensor.
    pub weight: Param<Array>,
    #[param]
    /// Optional FP8 inverse scale tensor.
    pub weight_scale_inv: Param<Option<Array>>,
    #[param]
    /// Optional affine quantization scales.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Optional affine quantization biases.
    pub biases: Param<Option<Array>>,
    #[param]
    /// Optional bias tensor.
    pub bias: Param<Option<Array>>,
    /// Affine quantization group size, or zero for dense/FP8 storage.
    pub group_size: i32,
    /// Affine quantization bit width, or zero for dense/FP8 storage.
    pub bits: i32,
}

impl QwenLinear {
    fn new(
        input_dims: i32,
        output_dims: i32,
        bias: bool,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (weight_shape, weight_dtype) = match format {
            QwenWeightFormat::Dense => (vec![output_dims, input_dims], Dtype::Float32),
            QwenWeightFormat::Fp8 => (vec![output_dims, input_dims], Dtype::Uint8),
            QwenWeightFormat::Affine(quantization) => (
                vec![output_dims, input_dims / (32 / quantization.bits)],
                Dtype::Uint32,
            ),
        };
        Ok(Self {
            input_dims,
            output_dims,
            weight: Param::<Array>::unloaded(&weight_shape, weight_dtype, stream)?,
            weight_scale_inv: if format == QwenWeightFormat::Fp8 {
                Param::<Option<Array>>::unloaded_some(
                    &[ceil_div(output_dims, 128), ceil_div(input_dims, 128)],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            scales: if let Some(quantization) = format.affine() {
                Param::<Option<Array>>::unloaded_some(
                    &[output_dims, input_dims / quantization.group_size],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            biases: if let Some(quantization) = format.affine() {
                Param::<Option<Array>>::unloaded_some(
                    &[output_dims, input_dims / quantization.group_size],
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
            group_size: format.affine().map_or(0, |value| value.group_size),
            bits: format.affine().map_or(0, |value| value.bits),
        })
    }

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut output =
            if let (Some(scales), Some(biases)) = (self.scales.as_ref(), self.biases.as_ref()) {
                quantized_matmul(
                    input,
                    self.weight.as_ref(),
                    scales,
                    biases,
                    true,
                    self.group_size,
                    self.bits,
                    stream,
                )?
            } else if let Some(scale) = self.weight_scale_inv.as_ref() {
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

    let out = if rows <= FP8_TILED_ROW_THRESHOLD {
        fp8_linear_tiled(
            &input, weight, scale, rows, in_dim, out_dim, scale_cols, stream,
        )?
    } else {
        fp8_linear_scalar(
            &input, weight, scale, rows, in_dim, out_dim, scale_cols, stream,
        )?
    };

    let mut output_shape = input_shape.to_vec();
    if let Some(last) = output_shape.last_mut() {
        *last = out_dim;
    }
    out.reshape(&output_shape, stream)
}

#[allow(clippy::too_many_arguments)]
fn fp8_linear_tiled(
    input: &Array,
    weight: &Array,
    scale: &Array,
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let out_grid = ceil_div(out_dim, FP8_LINEAR_OUT_TILE) * FP8_LINEAR_OUT_TILE;

    FP8_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(fp8_linear_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("IN_DIM", in_dim)
            .with_template_arg_int("OUT_DIM", out_dim)
            .with_template_arg_int("SCALE_COLS", scale_cols)
            .with_grid([out_grid, rows * 16, 1])
            .with_thread_group([16, 16, 1])
            .with_output_arg([rows, out_dim], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("FP8 linear kernel initialized")
            .apply_one_device([input, weight, scale], &config, stream)
    })
}

#[allow(clippy::too_many_arguments)]
fn fp8_linear_scalar(
    input: &Array,
    weight: &Array,
    scale: &Array,
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    FP8_LINEAR_SCALAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(fp8_linear_scalar_kernel()?);
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
            .expect("scalar FP8 linear kernel initialized")
            .apply_one_device([input, weight, scale], &config, stream)
    })
}

fn fp8_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "qwen35_moe_fp8_linear_k16",
        ["input", "weight", "scale"],
        ["out"],
        concat!(
            "uint out_col = thread_position_in_grid.x;",
            "uint row = thread_position_in_grid.y / 16;",
            "uint lane_k = thread_position_in_grid.y % 16;",
            "uint local_col = thread_position_in_grid.x % 16;",
            "uint input_base = row * IN_DIM;",
            "threadgroup float partial[16][16];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint k = lane_k; k < IN_DIM; k += 16) {",
            "  uint8_t raw = weight[out_col * IN_DIM + k];",
            "  float x = float(input[input_base + k]);",
            "  uint scale_col = k / 128;",
            "  float s = float(scale[(out_col / 128) * SCALE_COLS + scale_col]);",
            "  acc += x * fp8_e4m3_to_float(raw) * s;",
            "}",
            "}",
            "partial[lane_k][local_col] = acc;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            "  float sum = 0.0f;",
            "  for (uint lane = 0; lane < 16; ++lane) {",
            "    sum += partial[lane][local_col];",
            "  }",
            "  out[row * OUT_DIM + out_col] = sum;",
            "}"
        ),
        FP8_METAL_HEADER,
        true,
        false,
    )
}

fn fp8_linear_scalar_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "qwen35_moe_fp8_linear_scalar",
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
    if routes <= FP8_TILED_ROW_THRESHOLD {
        return grouped_fp8_linear_tiled(
            input, weight, scale, group_ids, routes, in_dim, out_dim, scale_cols, stream,
        );
    }

    grouped_fp8_linear_scalar(
        input, weight, scale, group_ids, routes, in_dim, out_dim, scale_cols, stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn grouped_fp8_linear_tiled(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let out_grid = ceil_div(out_dim, FP8_LINEAR_OUT_TILE) * FP8_LINEAR_OUT_TILE;
    FP8_GROUPED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(grouped_fp8_linear_kernel()?);
        }
        let config = MetalKernelConfig::new()
            .with_template_arg_int("IN_DIM", in_dim)
            .with_template_arg_int("OUT_DIM", out_dim)
            .with_template_arg_int("SCALE_OUT", scale.dim(1))
            .with_template_arg_int("SCALE_COLS", scale_cols)
            .with_grid([out_grid, routes * 16, 1])
            .with_thread_group([16, 16, 1])
            .with_output_arg([routes, out_dim], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("grouped FP8 linear kernel initialized")
            .apply_one_device([input, weight, scale, group_ids], &config, stream)
    })
}

#[allow(clippy::too_many_arguments)]
fn grouped_fp8_linear_scalar(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    FP8_GROUPED_LINEAR_SCALAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(grouped_fp8_linear_scalar_kernel()?);
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
            .expect("scalar grouped FP8 linear kernel initialized")
            .apply_one_device([input, weight, scale, group_ids], &config, stream)
    })
}

fn grouped_fp8_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "qwen35_moe_grouped_fp8_linear_k16",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint out_col = thread_position_in_grid.x;",
            "uint route = thread_position_in_grid.y / 16;",
            "uint lane_k = thread_position_in_grid.y % 16;",
            "uint local_col = thread_position_in_grid.x % 16;",
            "uint expert = uint(group_ids[route]);",
            "uint input_base = route * IN_DIM;",
            "threadgroup float partial[16][16];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint k = lane_k; k < IN_DIM; k += 16) {",
            "  uint weight_idx = (expert * OUT_DIM + out_col) * IN_DIM + k;",
            "  uint scale_idx = (expert * SCALE_OUT + (out_col / 128)) * SCALE_COLS + (k / 128);",
            "  float x = float(input[input_base + k]);",
            "  acc += x * fp8_e4m3_to_float(weight[weight_idx]) * float(scale[scale_idx]);",
            " }",
            "}",
            "partial[lane_k][local_col] = acc;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            "  float sum = 0.0f;",
            "  for (uint lane = 0; lane < 16; ++lane) {",
            "    sum += partial[lane][local_col];",
            "  }",
            "  out[route * OUT_DIM + out_col] = sum;",
            "}"
        ),
        FP8_METAL_HEADER,
        true,
        false,
    )
}

fn grouped_fp8_linear_scalar_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "qwen35_moe_grouped_fp8_linear_scalar",
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
/// Heterogeneous cache for Qwen3.5 MoE layers.
pub struct Cache {
    /// One cache entry per transformer layer.
    pub layers: Vec<LayerCache>,
}

impl Cache {
    /// Creates an empty cache matching the layer pattern in `args`.
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
            .map(|layer| match layer {
                LayerCache::FullAttention(cache) => cache.offset(),
                LayerCache::LinearAttention(cache) => cache.offset,
            })
            .next()
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
/// Per-layer cache for a Qwen3.5 MoE layer.
pub enum LayerCache {
    /// Full-attention key/value cache.
    FullAttention(ConcatKeyValueCache),
    /// Linear-attention convolution and recurrent cache.
    LinearAttention(LinearAttentionCache),
}

#[derive(Debug, Clone, Default)]
/// Cache state for recurrent linear-attention layers.
pub struct LinearAttentionCache {
    /// Cached causal-convolution state.
    pub conv_state: Option<Array>,
    /// Cached recurrent attention state.
    pub recurrent_state: Option<Array>,
    /// Number of tokens consumed by the layer.
    pub offset: i32,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3Next RMSNorm variant with learned offset scale.
pub struct Qwen3NextRmsNorm {
    #[param]
    /// Learned scale offset.
    pub weight: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
}

impl Qwen3NextRmsNorm {
    /// Creates an unloaded RMSNorm layer.
    pub fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    /// Applies normalization.
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let variance = safemlx::ops::mean_axis(&x.square(stream)?, -1, true, stream)?;
        let normalized = x.multiply(
            safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
            stream,
        )?;
        let scale = self.weight.as_ref().add(Array::from_f32(1.0), stream)?;
        normalized.multiply(scale, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Gated Qwen3Next RMSNorm used by linear attention.
pub struct Qwen3NextRmsNormGated {
    #[param]
    /// Learned scale.
    pub weight: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
}

impl Qwen3NextRmsNormGated {
    /// Creates an unloaded gated RMSNorm layer.
    pub fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    /// Applies normalization and SiLU gate modulation.
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

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Standard RMSNorm used by the Qwen vision encoder.
pub struct QwenVisionRmsNorm {
    #[param]
    /// Learned scale.
    pub weight: Param<Array>,
    #[param]
    /// Learned bias.
    pub bias: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
}

impl QwenVisionRmsNorm {
    fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            bias: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    fn forward(&self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mean = safemlx::ops::mean_axis(x, -1, true, stream)?;
        let centered = x.subtract(mean, stream)?;
        let variance = safemlx::ops::mean_axis(&centered.square(stream)?, -1, true, stream)?;
        let normalized = centered.multiply(
            safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
            stream,
        )?;
        normalized
            .multiply(&*self.weight, stream)?
            .add(&*self.bias, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Conv3d patch-projection weight for Qwen vision inputs.
pub struct QwenVisionPatchProjection {
    #[param]
    /// Projection weight shaped `[hidden, channels, temporal, height, width]`.
    pub weight: Param<Array>,
    #[param]
    /// Projection bias.
    pub bias: Param<Array>,
}

impl QwenVisionPatchProjection {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(
                &[
                    config.hidden_size,
                    config.in_channels,
                    config.temporal_patch_size,
                    config.patch_size,
                    config.patch_size,
                ],
                Dtype::Float32,
                stream,
            )?,
            bias: Param::<Array>::unloaded(&[config.hidden_size], Dtype::Float32, stream)?,
        })
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Patch embedding layer for preprocessed Qwen vision tensors.
pub struct QwenVisionPatchEmbed {
    /// Input channels.
    pub in_channels: i32,
    /// Temporal patch size.
    pub temporal_patch_size: i32,
    /// Spatial patch size.
    pub patch_size: i32,
    /// Output embedding dimension.
    pub embed_dim: i32,
    #[param]
    /// Conv3d projection represented as a flattened matrix multiply.
    pub proj: QwenVisionPatchProjection,
}

impl QwenVisionPatchEmbed {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            in_channels: config.in_channels,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
            embed_dim: config.hidden_size,
            proj: QwenVisionPatchProjection::new(config, stream)?,
        })
    }

    fn input_dim(&self) -> i32 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    fn forward(&self, pixel_values: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = pixel_values.shape();
        if shape.len() != 2 || shape[1] != self.input_dim() {
            return Err(Exception::custom(format!(
                "qwen3_5_moe image tensor must be shaped [patches, {}], got {shape:?}",
                self.input_dim()
            )));
        }
        let weight = self
            .proj
            .weight
            .as_ref()
            .reshape(&[self.embed_dim, self.input_dim()], stream)?;
        let output = matmul(pixel_values, weight.transpose(stream)?, stream)?;
        output.add(&*self.proj.bias, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Feed-forward block used by Qwen vision transformer layers.
pub struct QwenVisionMlp {
    /// Activation function name.
    pub hidden_act: String,
    #[param]
    /// First projection.
    pub linear_fc1: nn::Linear,
    #[param]
    /// Second projection.
    pub linear_fc2: nn::Linear,
}

impl QwenVisionMlp {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            hidden_act: config.hidden_act.clone(),
            linear_fc1: nn::Linear::unloaded(
                config.hidden_size,
                config.intermediate_size,
                true,
                Dtype::Float32,
                stream,
            )?,
            linear_fc2: nn::Linear::unloaded(
                config.intermediate_size,
                config.hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn activate(hidden_act: &str, x: Array, stream: &Stream) -> Result<Array, Exception> {
        match hidden_act {
            "silu" => silu(x, stream),
            "gelu" => nn::gelu(x, stream),
            "gelu_pytorch_tanh" => nn::gelu_approximate(x, stream),
            other => Err(Exception::custom(format!(
                "qwen3_5_moe vision MLP activation '{other}' is not supported"
            ))),
        }
    }

    fn forward(&mut self, hidden_states: &Array, stream: &Stream) -> Result<Array, Exception> {
        let hidden_act = self.hidden_act.clone();
        let hidden = Self::activate(
            hidden_act.as_str(),
            self.linear_fc1.forward(hidden_states, stream)?,
            stream,
        )?;
        self.linear_fc2.forward(&hidden, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear_fc1.training_mode(mode);
        self.linear_fc2.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Full attention used inside the Qwen vision encoder.
pub struct QwenVisionAttention {
    /// Number of attention heads.
    pub num_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Attention scale.
    pub scale: f32,
    #[param]
    /// Packed query/key/value projection.
    pub qkv: nn::Linear,
    #[param]
    /// Output projection.
    pub proj: nn::Linear,
}

impl QwenVisionAttention {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        if config.hidden_size % config.num_heads != 0 {
            return Err(Exception::custom(format!(
                "qwen3_5_moe vision hidden_size {} is not divisible by num_heads {}",
                config.hidden_size, config.num_heads
            )));
        }
        let head_dim = config.hidden_size / config.num_heads;
        Ok(Self {
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            qkv: nn::Linear::unloaded(
                config.hidden_size,
                config.hidden_size * 3,
                true,
                Dtype::Float32,
                stream,
            )?,
            proj: nn::Linear::unloaded(
                config.hidden_size,
                config.hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        chunk_lengths: &[i32],
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);
        let qkv = self
            .qkv
            .forward(hidden_states, stream)?
            .reshape(&[seq_len, 3, self.num_heads, self.head_dim], stream)?;
        let mut query = qkv.try_index_device((.., 0, .., ..), stream)?;
        let mut key = qkv.try_index_device((.., 1, .., ..), stream)?;
        let value = qkv.try_index_device((.., 2, .., ..), stream)?;
        (query, key) = apply_vision_rotary_pos_emb(query, key, cos, sin, stream)?;

        let mut outputs = Vec::with_capacity(chunk_lengths.len());
        let mut start = 0;
        for &len in chunk_lengths {
            let end = start + len;
            let q = query
                .try_index_device((start..end, .., ..), stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .try_index_device((NewAxis, .., .., ..), stream)?;
            let k = key
                .try_index_device((start..end, .., ..), stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .try_index_device((NewAxis, .., .., ..), stream)?;
            let v = value
                .try_index_device((start..end, .., ..), stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .try_index_device((NewAxis, .., .., ..), stream)?;
            let out = crate::utils::scaled_dot_product_attention(
                q,
                k,
                v,
                Option::<ConcatKeyValueCache>::None,
                self.scale,
                None,
                stream,
            )?
            .try_index_device((0, .., .., ..), stream)?
            .transpose_axes(&[1, 0, 2], stream)?
            .reshape(&[len, self.num_heads * self.head_dim], stream)?;
            outputs.push(out);
            start = end;
        }
        let out = concatenate_axis(&outputs, 0, stream)?;
        self.proj.forward(&out, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.qkv.training_mode(mode);
        self.proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Transformer block used by the Qwen vision encoder.
pub struct QwenVisionBlock {
    #[param]
    /// First RMSNorm.
    pub norm1: QwenVisionRmsNorm,
    #[param]
    /// Attention module.
    pub attn: QwenVisionAttention,
    #[param]
    /// Second RMSNorm.
    pub norm2: QwenVisionRmsNorm,
    #[param]
    /// Feed-forward module.
    pub mlp: QwenVisionMlp,
}

impl QwenVisionBlock {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            norm1: QwenVisionRmsNorm::new(config.hidden_size, 1e-6, stream)?,
            attn: QwenVisionAttention::new(config, stream)?,
            norm2: QwenVisionRmsNorm::new(config.hidden_size, 1e-6, stream)?,
            mlp: QwenVisionMlp::new(config, stream)?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: Array,
        chunk_lengths: &[i32],
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normed = self.norm1.forward(&hidden_states, stream)?;
        let attn = self
            .attn
            .forward(&normed, chunk_lengths, cos, sin, stream)?;
        let hidden_states = hidden_states.add(attn, stream)?;
        let normed = self.norm2.forward(&hidden_states, stream)?;
        let mlp = self.mlp.forward(&normed, stream)?;
        hidden_states.add(mlp, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm1.training_mode(mode);
        self.attn.training_mode(mode);
        self.norm2.training_mode(mode);
        self.mlp.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Patch merger that maps vision features to language-model embeddings.
pub struct QwenVisionPatchMerger {
    /// Number of patch tokens merged per output token.
    pub spatial_merge_unit: i32,
    /// Vision hidden size.
    pub context_dim: i32,
    /// Flattened merger hidden size.
    pub hidden_size: i32,
    #[param]
    /// Pre-merge RMSNorm.
    pub norm: QwenVisionRmsNorm,
    #[param]
    /// First merger projection.
    pub linear_fc1: nn::Linear,
    #[param]
    /// Final projection into language hidden size.
    pub linear_fc2: nn::Linear,
}

impl QwenVisionPatchMerger {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        let spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size;
        let hidden_size = config.hidden_size * spatial_merge_unit;
        Ok(Self {
            spatial_merge_unit,
            context_dim: config.hidden_size,
            hidden_size,
            norm: QwenVisionRmsNorm::new(config.hidden_size, 1e-6, stream)?,
            linear_fc1: nn::Linear::unloaded(
                hidden_size,
                hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
            linear_fc2: nn::Linear::unloaded(
                hidden_size,
                config.out_hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(&mut self, hidden_states: &Array, stream: &Stream) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);
        if seq_len % self.spatial_merge_unit != 0 {
            return Err(Exception::custom(format!(
                "qwen3_5_moe vision sequence length {seq_len} is not divisible by spatial merge unit {}",
                self.spatial_merge_unit
            )));
        }
        let hidden_states = self.norm.forward(hidden_states, stream)?;
        let hidden_states =
            hidden_states.reshape(&[-1, self.spatial_merge_unit, self.context_dim], stream)?;
        let hidden_states = hidden_states.reshape(&[-1, self.hidden_size], stream)?;
        let hidden_states = self.linear_fc1.forward(&hidden_states, stream)?;
        let hidden_states = nn::gelu_approximate(hidden_states, stream)?;
        self.linear_fc2.forward(&hidden_states, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        self.linear_fc1.training_mode(mode);
        self.linear_fc2.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen VL vision transformer used to encode image tensors.
pub struct QwenVisionTransformer {
    /// Vision configuration.
    pub config: VisionConfig,
    #[param]
    /// Learned spatial position embedding table.
    pub pos_embed: nn::Embedding,
    #[param]
    /// Patch embedding.
    pub patch_embed: QwenVisionPatchEmbed,
    #[param]
    /// Vision transformer blocks.
    pub blocks: Vec<QwenVisionBlock>,
    #[param]
    /// Patch merger.
    pub merger: QwenVisionPatchMerger,
}

impl QwenVisionTransformer {
    fn new(config: VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        if config.spatial_merge_size <= 0 {
            return Err(Exception::custom(
                "qwen3_5_moe vision spatial_merge_size must be positive",
            ));
        }
        let pos_embed = nn::Embedding::unloaded(
            config.num_position_embeddings,
            config.hidden_size,
            Dtype::Float32,
            stream,
        )?;
        let patch_embed = QwenVisionPatchEmbed::new(&config, stream)?;
        let mut blocks = Vec::with_capacity(config.depth as usize);
        for _ in 0..config.depth {
            blocks.push(QwenVisionBlock::new(&config, stream)?);
        }
        let merger = QwenVisionPatchMerger::new(&config, stream)?;
        Ok(Self {
            config,
            pos_embed,
            patch_embed,
            blocks,
            merger,
        })
    }

    fn forward(
        &mut self,
        pixel_values: &Array,
        grid_thw: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let grid = grid_thw_from_array(grid_thw, stream)?;
        validate_vision_grid(&grid, self.config.spatial_merge_size, pixel_values)?;
        let mut hidden_states = self.patch_embed.forward(pixel_values, stream)?;
        let seq_len = hidden_states.dim(0);
        let position_indices = vision_position_indices(&grid, self.config.num_position_embeddings)?;
        let position_indices =
            Array::from_slice(&position_indices, &[position_indices.len() as i32]);
        let position_embeddings = self.pos_embed.forward(&position_indices, stream)?;
        hidden_states = hidden_states.add(position_embeddings, stream)?;
        let full_chunk_lengths = vision_attention_chunk_lengths(&grid);
        let total: i32 = full_chunk_lengths.iter().sum();
        if total != seq_len {
            return Err(Exception::custom(format!(
                "qwen3_5_moe vision grid describes {total} patches but image tensor has {seq_len}"
            )));
        }
        let (window_index, window_chunk_lengths) = vision_window_index(
            &grid,
            self.config.spatial_merge_size,
            self.config.window_size,
            self.config.patch_size,
        )?;
        let window_index_array = Array::from_slice(&window_index, &[window_index.len() as i32]);
        hidden_states = hidden_states.reshape(
            &[
                seq_len / (self.config.spatial_merge_size * self.config.spatial_merge_size),
                self.config.spatial_merge_size * self.config.spatial_merge_size,
                -1,
            ],
            stream,
        )?;
        hidden_states = hidden_states.try_index_device((&window_index_array, .., ..), stream)?;
        hidden_states = hidden_states.reshape(&[seq_len, -1], stream)?;

        let (cos, sin) = vision_rotary_embeddings(
            &grid,
            self.config.spatial_merge_size,
            self.config.hidden_size / self.config.num_heads,
        );
        let cos = cos.reshape(
            &[
                seq_len / (self.config.spatial_merge_size * self.config.spatial_merge_size),
                self.config.spatial_merge_size * self.config.spatial_merge_size,
                -1,
            ],
            stream,
        )?;
        let cos = cos
            .try_index_device((&window_index_array, .., ..), stream)?
            .reshape(&[seq_len, -1], stream)?;
        let sin = sin.reshape(
            &[
                seq_len / (self.config.spatial_merge_size * self.config.spatial_merge_size),
                self.config.spatial_merge_size * self.config.spatial_merge_size,
                -1,
            ],
            stream,
        )?;
        let sin = sin
            .try_index_device((&window_index_array, .., ..), stream)?
            .reshape(&[seq_len, -1], stream)?;

        for (layer_num, block) in self.blocks.iter_mut().enumerate() {
            let chunk_lengths = if self
                .config
                .fullatt_block_indexes
                .contains(&(layer_num as i32))
            {
                &full_chunk_lengths
            } else {
                &window_chunk_lengths
            };
            hidden_states = block.forward(hidden_states, chunk_lengths, &cos, &sin, stream)?;
        }
        let hidden_states = self.merger.forward(&hidden_states, stream)?;
        let reverse_index = reverse_permutation(&window_index);
        let reverse_index_array = Array::from_slice(&reverse_index, &[reverse_index.len() as i32]);
        hidden_states
            .try_index_device((&reverse_index_array, ..), stream)?
            .try_index_device((NewAxis, .., ..), stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.patch_embed.training_mode(mode);
        for block in &mut self.blocks {
            block.training_mode(mode);
        }
        self.merger.training_mode(mode);
    }
}

fn apply_vision_rotary_pos_emb(
    query: Array,
    key: Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let cos = cos.try_index_device((.., NewAxis, ..), stream)?;
    let sin = sin.try_index_device((.., NewAxis, ..), stream)?;
    let query_embed = query.multiply(&cos, stream)?.add(
        rotate_half_vision(&query, stream)?.multiply(&sin, stream)?,
        stream,
    )?;
    let key_embed = key.multiply(cos, stream)?.add(
        rotate_half_vision(&key, stream)?.multiply(sin, stream)?,
        stream,
    )?;
    Ok((query_embed, key_embed))
}

fn rotate_half_vision(x: &Array, stream: &Stream) -> Result<Array, Exception> {
    let half = x.dim(-1) / 2;
    let x1 = x.try_index_device((.., .., ..half), stream)?;
    let x2 = x.try_index_device((.., .., half..), stream)?;
    concatenate_axis(
        &[x2.multiply(Array::from_f32(-1.0), stream)?, x1],
        -1,
        stream,
    )
}

fn grid_thw_from_array(
    grid_thw: &Array,
    stream: &Stream,
) -> Result<Vec<(i32, i32, i32)>, Exception> {
    let shape = grid_thw.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(Exception::custom(format!(
            "qwen3_5_moe qwen_grid_thw must be shaped [items, 3], got {shape:?}"
        )));
    }
    let mut grid = Vec::with_capacity(shape[0] as usize);
    for index in 0..shape[0] {
        let t = grid_thw
            .try_index_device((index, 0), stream)?
            .item::<i32>(stream);
        let h = grid_thw
            .try_index_device((index, 1), stream)?
            .item::<i32>(stream);
        let w = grid_thw
            .try_index_device((index, 2), stream)?
            .item::<i32>(stream);
        grid.push((t, h, w));
    }
    Ok(grid)
}

fn validate_vision_grid(
    grid: &[(i32, i32, i32)],
    spatial_merge_size: i32,
    pixel_values: &Array,
) -> Result<(), Exception> {
    let patches: i32 = grid.iter().map(|(t, h, w)| t * h * w).sum();
    if patches != pixel_values.dim(0) {
        return Err(Exception::custom(format!(
            "qwen3_5_moe qwen_grid_thw describes {patches} patches but image tensor has {}",
            pixel_values.dim(0)
        )));
    }
    for &(t, h, w) in grid {
        if t <= 0 || h <= 0 || w <= 0 {
            return Err(Exception::custom(format!(
                "qwen3_5_moe qwen_grid_thw entries must be positive, got {:?}",
                (t, h, w)
            )));
        }
        if h % spatial_merge_size != 0 || w % spatial_merge_size != 0 {
            return Err(Exception::custom(format!(
                "qwen3_5_moe qwen_grid_thw spatial dimensions must be divisible by spatial_merge_size {spatial_merge_size}, got {:?}",
                (t, h, w)
            )));
        }
    }
    Ok(())
}

fn vision_position_indices(
    grid: &[(i32, i32, i32)],
    num_position_embeddings: i32,
) -> Result<Vec<u32>, Exception> {
    let side = (num_position_embeddings as f64).sqrt() as i32;
    if side * side != num_position_embeddings {
        return Err(Exception::custom(format!(
            "qwen3_5_moe vision num_position_embeddings must be a square, got {num_position_embeddings}"
        )));
    }
    let mut indices = Vec::new();
    for &(t, h, w) in grid {
        if h > side || w > side {
            return Err(Exception::custom(format!(
                "qwen3_5_moe qwen_grid_thw spatial dimensions {:?} exceed learned position table side {side}",
                (h, w)
            )));
        }
        for _ in 0..t {
            for h_pos in 0..h {
                for w_pos in 0..w {
                    indices.push((h_pos * side + w_pos) as u32);
                }
            }
        }
    }
    Ok(indices)
}

fn vision_attention_chunk_lengths(grid: &[(i32, i32, i32)]) -> Vec<i32> {
    let mut lengths = Vec::new();
    for &(t, h, w) in grid {
        for _ in 0..t {
            lengths.push(h * w);
        }
    }
    lengths
}

fn vision_window_index(
    grid: &[(i32, i32, i32)],
    spatial_merge_size: i32,
    window_size: i32,
    patch_size: i32,
) -> Result<(Vec<i32>, Vec<i32>), Exception> {
    let vit_merger_window_size = window_size / spatial_merge_size / patch_size;
    if vit_merger_window_size <= 0 {
        return Err(Exception::custom(format!(
            "qwen3_5_moe vision window_size {window_size} is too small for spatial_merge_size {spatial_merge_size} and patch_size {patch_size}"
        )));
    }
    let spatial_merge_unit = spatial_merge_size * spatial_merge_size;
    let mut window_index = Vec::new();
    let mut cumulative_seqlens = vec![0];
    let mut window_index_id = 0;
    for &(grid_t, grid_h, grid_w) in grid {
        let llm_grid_h = grid_h / spatial_merge_size;
        let llm_grid_w = grid_w / spatial_merge_size;
        let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
        let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
        let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
        for t in 0..grid_t {
            for window_h in 0..num_windows_h {
                for window_w in 0..num_windows_w {
                    let mut window_groups = 0;
                    for inner_h in 0..vit_merger_window_size {
                        for inner_w in 0..vit_merger_window_size {
                            let h = window_h * vit_merger_window_size + inner_h;
                            let w = window_w * vit_merger_window_size + inner_w;
                            if h < llm_grid_h && w < llm_grid_w {
                                let index = t * llm_grid_h * llm_grid_w + h * llm_grid_w + w;
                                window_index.push(window_index_id + index);
                                window_groups += 1;
                            }
                        }
                    }
                    let next = cumulative_seqlens.last().copied().unwrap_or(0)
                        + window_groups * spatial_merge_unit;
                    if cumulative_seqlens.last().copied() != Some(next) {
                        cumulative_seqlens.push(next);
                    }
                }
            }
        }
        window_index_id += grid_t * llm_grid_h * llm_grid_w;
    }
    let chunk_lengths = cumulative_seqlens
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect::<Vec<_>>();
    Ok((window_index, chunk_lengths))
}

fn reverse_permutation(indices: &[i32]) -> Vec<i32> {
    let mut reverse = vec![0; indices.len()];
    for (position, &index) in indices.iter().enumerate() {
        reverse[index as usize] = position as i32;
    }
    reverse
}

fn vision_rotary_embeddings(
    grid: &[(i32, i32, i32)],
    spatial_merge_size: i32,
    head_dim: i32,
) -> (Array, Array) {
    let rotary_dim = head_dim / 2;
    let inv_freq = (0..rotary_dim)
        .step_by(2)
        .map(|idx| 1.0f32 / 10000.0f32.powf(idx as f32 / rotary_dim as f32))
        .collect::<Vec<_>>();
    let mut cos_values = Vec::new();
    let mut sin_values = Vec::new();
    for &(t, h, w) in grid {
        for _ in 0..t {
            for h_block in 0..(h / spatial_merge_size) {
                for w_block in 0..(w / spatial_merge_size) {
                    for h_inner in 0..spatial_merge_size {
                        for w_inner in 0..spatial_merge_size {
                            let h_pos = h_block * spatial_merge_size + h_inner;
                            let w_pos = w_block * spatial_merge_size + w_inner;
                            let mut angles = Vec::with_capacity(rotary_dim as usize);
                            for position in [h_pos, w_pos] {
                                for inv in &inv_freq {
                                    angles.push(position as f32 * inv);
                                }
                            }
                            let full_angles = angles
                                .iter()
                                .chain(angles.iter())
                                .copied()
                                .collect::<Vec<_>>();
                            for angle in full_angles {
                                cos_values.push(angle.cos());
                                sin_values.push(angle.sin());
                            }
                        }
                    }
                }
            }
        }
    }
    let seq_len = (cos_values.len() as i32) / head_dim;
    (
        Array::from_slice(&cos_values, &[seq_len, head_dim]),
        Array::from_slice(&sin_values, &[seq_len, head_dim]),
    )
}

#[derive(Debug, Clone, ModuleParameters)]
/// Full self-attention layer in Qwen3.5 MoE.
pub struct FullAttention {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads.
    pub n_kv_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Attention scaling factor.
    pub scale: f32,
    #[param]
    /// Query projection.
    pub q_proj: QwenLinear,
    #[param]
    /// Key projection.
    pub k_proj: QwenLinear,
    #[param]
    /// Value projection.
    pub v_proj: QwenLinear,
    #[param]
    /// Output projection.
    pub o_proj: QwenLinear,
    #[param]
    /// Query normalization.
    pub q_norm: Qwen3NextRmsNorm,
    #[param]
    /// Key normalization.
    pub k_norm: Qwen3NextRmsNorm,
    #[param]
    /// Rotary position embedding module.
    pub rope: RopeVariant,
}

impl FullAttention {
    /// Creates an unloaded full-attention layer.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(args, QwenWeightFormat::for_text(args, None), stream)
    }

    fn new_with_format(
        args: &ModelArgs,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let hidden = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let q_proj = QwenLinear::new(
            hidden,
            n_heads * head_dim * 2,
            args.attention_bias,
            format,
            stream,
        )?;
        let k_proj = QwenLinear::new(
            hidden,
            n_kv_heads * head_dim,
            args.attention_bias,
            format,
            stream,
        )?;
        let v_proj = QwenLinear::new(
            hidden,
            n_kv_heads * head_dim,
            args.attention_bias,
            format,
            stream,
        )?;
        let o_proj = QwenLinear::new(
            n_heads * head_dim,
            hidden,
            args.attention_bias,
            format,
            stream,
        )?;
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

/// Input for a Qwen3.5 full-attention layer.
pub struct FullAttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional key/value cache.
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

impl FullAttention {
    /// Forward pass that reports full-attention activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: FullAttentionInput<'_>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let FullAttentionInput { x, mask, mut cache } = input;
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let q_proj = self
            .q_proj
            .forward(x, stream)?
            .reshape(&[b, l, self.n_heads, 2 * self.head_dim], stream)?;
        observer.observe(&format!("{prefix}.q_proj"), &q_proj)?;
        let query = q_proj.try_index_device((.., .., .., ..self.head_dim), stream)?;
        let gate = q_proj
            .try_index_device((.., .., .., self.head_dim..), stream)?
            .reshape(&[b, l, self.n_heads * self.head_dim], stream)?;
        observer.observe(&format!("{prefix}.gate"), &gate)?;
        let mut query = self
            .q_norm
            .forward(&query, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observer.observe(&format!("{prefix}.q_norm"), &query)?;
        let mut key = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x, stream)?
                    .reshape(&[b, l, self.n_kv_heads, self.head_dim], stream)?,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observer.observe(&format!("{prefix}.k_norm"), &key)?;
        let mut value = self
            .v_proj
            .forward(x, stream)?
            .reshape(&[b, l, self.n_kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observer.observe(&format!("{prefix}.values"), &value)?;

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
        observer.observe(&format!("{prefix}.queries_rope"), &query)?;
        observer.observe(&format!("{prefix}.keys_rope"), &key)?;
        observer.observe(&format!("{prefix}.values_cache"), &value)?;
        let attention_probs = attention_probabilities(&query, &key, self.scale, mask, stream)?;
        observer.observe(&format!("{prefix}.attention_probs"), &attention_probs)?;

        let out = crate::utils::scaled_dot_product_attention(
            query, key, value, cache, self.scale, mask, stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[b, l, -1], stream)?;
        observer.observe(&format!("{prefix}.attention"), &out)?;
        let gated = out.multiply(sigmoid(gate, stream)?, stream)?;
        observer.observe(&format!("{prefix}.attention_gated"), &gated)?;
        let output = self.o_proj.forward(&gated, stream)?;
        observer.observe(&format!("{prefix}.o_proj"), &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Depthwise one-dimensional convolution parameters.
pub struct DepthwiseConv1d {
    #[param]
    /// Convolution weights.
    pub weight: Param<Array>,
}

impl DepthwiseConv1d {
    /// Creates an unloaded depthwise convolution.
    pub fn new(channels: i32, kernel_size: i32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[channels, 1, kernel_size], Dtype::Float32, stream)?,
        })
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
/// Recurrent linear-attention layer used by Qwen3.5 MoE.
pub struct LinearAttention {
    /// Number of value heads.
    pub num_v_heads: i32,
    /// Number of key heads.
    pub num_k_heads: i32,
    /// Key head dimension.
    pub head_k_dim: i32,
    /// Value head dimension.
    pub head_v_dim: i32,
    /// Total key dimension.
    pub key_dim: i32,
    /// Total value dimension.
    pub value_dim: i32,
    /// Convolution input dimension.
    pub conv_dim: i32,
    /// Causal convolution kernel size.
    pub conv_kernel_size: i32,
    #[param]
    /// Depthwise causal convolution.
    pub conv1d: DepthwiseConv1d,
    #[param]
    /// Joint query/key/value projection.
    pub in_proj_qkv: QwenLinear,
    #[param]
    /// Output gate projection.
    pub in_proj_z: QwenLinear,
    #[param]
    /// Beta projection.
    pub in_proj_b: QwenLinear,
    #[param]
    /// Delta projection.
    pub in_proj_a: QwenLinear,
    #[param]
    /// Delta bias.
    pub dt_bias: Param<Array>,
    #[param]
    /// Log transition parameter.
    pub A_log: Param<Array>,
    #[param]
    /// Gated normalization.
    pub norm: Qwen3NextRmsNormGated,
    #[param]
    /// Output projection.
    pub out_proj: QwenLinear,
}

impl LinearAttention {
    /// Creates an unloaded linear-attention layer.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(args, QwenWeightFormat::for_text(args, None), stream)
    }

    fn new_with_format(
        args: &ModelArgs,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
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
                format,
                stream,
            )?,
            in_proj_z: QwenLinear::new(args.hidden_size, value_dim, false, format, stream)?,
            in_proj_b: QwenLinear::new(
                args.hidden_size,
                num_v_heads,
                false,
                QwenWeightFormat::Dense,
                stream,
            )?,
            in_proj_a: QwenLinear::new(
                args.hidden_size,
                num_v_heads,
                false,
                QwenWeightFormat::Dense,
                stream,
            )?,
            dt_bias: Param::new(Array::from_slice(
                &vec![1.0f32; num_v_heads as usize],
                &[num_v_heads],
            )),
            A_log: Param::new(Array::from_slice(
                &vec![0.0f32; num_v_heads as usize],
                &[num_v_heads],
            )),
            norm: Qwen3NextRmsNormGated::new(head_v_dim, args.rms_norm_eps, stream)?,
            out_proj: QwenLinear::new(value_dim, args.hidden_size, false, format, stream)?,
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

    #[allow(non_snake_case, clippy::too_many_arguments)]
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

/// Input for a Qwen3.5 linear-attention layer.
pub struct LinearAttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional linear-attention cache.
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

impl LinearAttention {
    /// Forward pass that reports recurrent linear-attention internals.
    #[allow(non_snake_case)]
    pub fn forward_with_observer(
        &mut self,
        input: LinearAttentionInput<'_>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let LinearAttentionInput { x, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let mixed_qkv = self.in_proj_qkv.forward(x, stream)?;
        observer.observe(&format!("{prefix}.in_proj_qkv"), &mixed_qkv)?;
        let z = self
            .in_proj_z
            .forward(x, stream)?
            .reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.z_proj"), &z)?;
        let b = self.in_proj_b.forward(x, stream)?;
        observer.observe(&format!("{prefix}.beta_proj"), &b)?;
        let a = self.in_proj_a.forward(x, stream)?;
        observer.observe(&format!("{prefix}.a_proj"), &a)?;
        let mixed_qkv = self.depthwise_causal_conv(&mixed_qkv, cache.as_deref_mut(), stream)?;
        observer.observe(&format!("{prefix}.causal_conv"), &mixed_qkv)?;

        let query = mixed_qkv
            .try_index_device((.., .., ..self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        observer.observe(&format!("{prefix}.query_raw"), &query)?;
        let key = mixed_qkv
            .try_index_device((.., .., self.key_dim..2 * self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        observer.observe(&format!("{prefix}.key_raw"), &key)?;
        let mut value = mixed_qkv
            .try_index_device((.., .., 2 * self.key_dim..), stream)?
            .reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.value"), &value)?;
        let mut query = Self::l2norm(query, stream)?;
        observer.observe(&format!("{prefix}.query_l2norm"), &query)?;
        let mut key = Self::l2norm(key, stream)?;
        observer.observe(&format!("{prefix}.key_l2norm"), &key)?;
        let beta = sigmoid(b, stream)?;
        observer.observe(&format!("{prefix}.beta"), &beta)?;
        let dt_bias = self.dt_bias.reshape(&[1, 1, self.num_v_heads], stream)?;
        let g = nn::softplus(a.add(dt_bias, stream)?, stream)?.multiply(
            exp(self.A_log.as_ref(), stream)?.multiply(Array::from_f32(-1.0), stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.decay"), &g)?;

        let repeats = self.num_v_heads / self.num_k_heads;
        if repeats > 1 {
            let expanded_query = query.try_index_device((.., .., .., NewAxis, ..), stream)?;
            query = broadcast_to(
                &expanded_query,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
            observer.observe(&format!("{prefix}.query_repeated"), &query)?;
            let expanded_key = key.try_index_device((.., .., .., NewAxis, ..), stream)?;
            key = broadcast_to(
                &expanded_key,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
            observer.observe(&format!("{prefix}.key_repeated"), &key)?;
        }

        value = value.as_dtype(x.dtype(), stream)?;
        let core = self.recurrent_delta_rule(query, key, value, g, beta, cache, stream)?;
        observer.observe(&format!("{prefix}.recurrent_core"), &core)?;
        let z_shape = z.shape().to_vec();
        let core = core.reshape(&[-1, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.recurrent_core_flat"), &core)?;
        let z = z.reshape(&[-1, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.z_flat"), &z)?;
        let normalized = self.norm.forward(&core, &z, stream)?;
        observer.observe(&format!("{prefix}.gated_norm"), &normalized)?;
        let out = normalized
            .reshape(&z_shape, stream)?
            .reshape(&[B, L, self.value_dim], stream)?;
        observer.observe(&format!("{prefix}.pre_out_proj"), &out)?;
        let output = self.out_proj.forward(&out, stream)?;
        observer.observe(&format!("{prefix}.out_proj"), &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Dense SwiGLU MLP used by the shared expert.
pub struct Mlp {
    #[param]
    /// Gate projection.
    pub gate_proj: QwenLinear,
    #[param]
    /// Up projection.
    pub up_proj: QwenLinear,
    #[param]
    /// Down projection.
    pub down_proj: QwenLinear,
}

impl Mlp {
    fn new(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QwenLinear::new(dim, hidden_dim, bias, format, stream)?,
            up_proj: QwenLinear::new(dim, hidden_dim, bias, format, stream)?,
            down_proj: QwenLinear::new(hidden_dim, dim, bias, format, stream)?,
        })
    }

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let down_proj_input = silu(self.gate_proj.forward(input, stream)?, stream)?
            .multiply(self.up_proj.forward(input, stream)?, stream)?;
        self.down_proj.forward(&down_proj_input, stream)
    }

    fn forward_with_observer(
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
        let hidden = silu(gate, stream)?.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.down_proj_input"), &hidden)?;
        let output = self.down_proj.forward(&hidden, stream)?;
        observer.observe(&format!("{prefix}.down_proj"), &output)?;
        Ok(output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Routed expert bank for Qwen3.5 MoE.
pub struct Experts {
    /// Number of experts.
    pub num_experts: i32,
    /// Model hidden dimension.
    pub hidden_dim: i32,
    /// Expert intermediate dimension.
    pub intermediate_dim: i32,
    /// Whether expert weights are stored as FP8.
    pub use_fp8: bool,
    /// Optional affine quantization for the packed gate/up bank.
    pub gate_up_affine: Option<AffineQuantization>,
    /// Optional affine quantization for the packed down bank.
    pub down_affine: Option<AffineQuantization>,
    #[param]
    /// Packed gate and up projection weights for all experts.
    pub gate_up_proj: Param<Array>,
    #[param]
    /// Optional FP8 inverse scales for gate/up projection weights.
    pub gate_up_proj_scale_inv: Param<Option<Array>>,
    #[param]
    /// Optional affine scales for gate/up projection weights.
    pub gate_up_proj_scales: Param<Option<Array>>,
    #[param]
    /// Optional affine biases for gate/up projection weights.
    pub gate_up_proj_biases: Param<Option<Array>>,
    #[param]
    /// Down projection weights for all experts.
    pub down_proj: Param<Array>,
    #[param]
    /// Optional FP8 inverse scales for down projection weights.
    pub down_proj_scale_inv: Param<Option<Array>>,
    #[param]
    /// Optional affine scales for down projection weights.
    pub down_proj_scales: Param<Option<Array>>,
    #[param]
    /// Optional affine biases for down projection weights.
    pub down_proj_biases: Param<Option<Array>>,
}

type AffineExpertProjectionParams = (Param<Array>, Param<Option<Array>>, Param<Option<Array>>);

impl Experts {
    /// Creates an unloaded routed expert bank.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(
            args,
            layer_idx,
            QwenWeightFormat::for_text(args, None),
            stream,
        )
    }

    fn new_with_format(
        args: &ModelArgs,
        layer_idx: usize,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer_idx}.mlp.experts");
        let gate_up_affine = args
            .affine_quantization_for(&format!("{prefix}.gate_up_proj"))
            .or_else(|| format.affine());
        let down_affine = args
            .affine_quantization_for(&format!("{prefix}.down_proj"))
            .or_else(|| format.affine());
        let use_fp8 = format == QwenWeightFormat::Fp8;
        let expert_weight_dtype = if use_fp8 {
            Dtype::Uint8
        } else {
            Dtype::Float32
        };
        let projection = |out_features: i32,
                          in_features: i32,
                          affine: Option<AffineQuantization>|
         -> Result<AffineExpertProjectionParams, Exception> {
            if let Some(affine) = affine {
                Ok((
                    Param::<Array>::unloaded(
                        &[
                            args.num_experts,
                            out_features,
                            quantized_packed_dimension(in_features, affine.bits),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            args.num_experts,
                            out_features,
                            in_features / affine.group_size,
                        ],
                        Dtype::Float16,
                        stream,
                    )?,
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            args.num_experts,
                            out_features,
                            in_features / affine.group_size,
                        ],
                        Dtype::Float16,
                        stream,
                    )?,
                ))
            } else {
                Ok((
                    Param::<Array>::unloaded(
                        &[args.num_experts, out_features, in_features],
                        expert_weight_dtype,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ))
            }
        };
        let (gate_up_proj, gate_up_proj_scales, gate_up_proj_biases) = projection(
            2 * args.moe_intermediate_size,
            args.hidden_size,
            gate_up_affine,
        )?;
        let (down_proj, down_proj_scales, down_proj_biases) =
            projection(args.hidden_size, args.moe_intermediate_size, down_affine)?;
        Ok(Self {
            num_experts: args.num_experts,
            hidden_dim: args.hidden_size,
            intermediate_dim: args.moe_intermediate_size,
            use_fp8,
            gate_up_affine,
            down_affine,
            gate_up_proj,
            gate_up_proj_scale_inv: if use_fp8 {
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
            gate_up_proj_scales,
            gate_up_proj_biases,
            down_proj,
            down_proj_scale_inv: if use_fp8 {
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
            down_proj_scales,
            down_proj_biases,
        })
    }

    fn affine_grouped_linear(
        input: &Array,
        weight: &Array,
        scales: &Array,
        biases: &Array,
        group_ids: &Array,
        affine: AffineQuantization,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let routes = input.dim(0);
        let out_features = weight.dim(-2);
        let lhs_indices = arange::<i32, u32>(0, routes, 1, stream)?;
        gather_qmm(
            input.reshape(&[routes, 1, input.dim(-1)], stream)?,
            weight,
            scales,
            biases,
            &lhs_indices,
            group_ids,
            true,
            affine.group_size,
            affine.bits,
            true,
            stream,
        )?
        .reshape(&[routes, out_features], stream)
    }

    /// Evaluates routed experts for flattened token hidden states.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if self.use_fp8 || self.gate_up_affine.is_some() || self.down_affine.is_some() {
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

    /// Evaluates routed experts in chunks for long prefill inputs.
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

    /// Evaluates routed experts while reporting per-route expert internals.
    pub fn forward_chunked_with_observer(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        if num_tokens <= ROUTED_EXPERT_CHUNK_THRESHOLD {
            return self.forward_with_observer(
                hidden_states,
                top_k_index,
                top_k_weights,
                stream,
                prefix,
                observer,
            );
        }

        let mut outputs = Vec::with_capacity(
            ((num_tokens + ROUTED_EXPERT_CHUNK_TOKENS - 1) / ROUTED_EXPERT_CHUNK_TOKENS)
                .try_into()
                .expect("number of MoE chunks must fit in usize"),
        );
        let mut start = 0;
        let mut chunk = 0;
        while start < num_tokens {
            let end = (start + ROUTED_EXPERT_CHUNK_TOKENS).min(num_tokens);
            let hidden_chunk = hidden_states.try_index_device((start..end, ..), stream)?;
            observer.observe(&format!("{prefix}.chunks.{chunk}.input"), &hidden_chunk)?;
            let expert_chunk = top_k_index.try_index_device((start..end, ..), stream)?;
            let weight_chunk = top_k_weights.try_index_device((start..end, ..), stream)?;
            outputs.push(self.forward_expert_major_chunk_with_observer(
                &hidden_chunk,
                &expert_chunk,
                &weight_chunk,
                stream,
                &format!("{prefix}.chunks.{chunk}"),
                observer,
            )?);
            start = end;
            chunk += 1;
        }
        let output = concatenate_axis(&outputs, 0, stream)?;
        observer.observe(&format!("{prefix}.chunked_output"), &output)?;
        Ok(output)
    }

    /// Evaluates routed experts for flattened token hidden states with observer hooks.
    pub fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        observer.observe(&format!("{prefix}.input"), hidden_states)?;
        observer.observe(&format!("{prefix}.top_k_experts"), top_k_index)?;
        observer.observe(&format!("{prefix}.top_k_weights"), top_k_weights)?;
        if self.use_fp8 || self.gate_up_affine.is_some() || self.down_affine.is_some() {
            return self.forward_expert_major_chunk_with_observer(
                hidden_states,
                top_k_index,
                top_k_weights,
                stream,
                prefix,
                observer,
            );
        }

        let num_tokens = hidden_states.shape()[0];
        let top_k = top_k_index.shape()[1];
        let selected_gate_up = self
            .gate_up_proj
            .as_ref()
            .take_axis(top_k_index, 0, stream)?;
        observer.observe(
            &format!("{prefix}.selected_gate_up_weight"),
            &selected_gate_up,
        )?;
        let hidden = hidden_states.try_index_device((.., NewAxis, NewAxis, ..), stream)?;
        let gate_up = matmul(&hidden, selected_gate_up.swap_axes(-1, -2, stream)?, stream)?
            .reshape(&[num_tokens, top_k, 2 * self.intermediate_dim], stream)?;
        observer.observe(&format!("{prefix}.gate_up_proj"), &gate_up)?;
        let gate = gate_up.try_index_device((.., .., ..self.intermediate_dim), stream)?;
        observer.observe(&format!("{prefix}.gate_proj"), &gate)?;
        let up = gate_up.try_index_device((.., .., self.intermediate_dim..), stream)?;
        observer.observe(&format!("{prefix}.up_proj"), &up)?;
        let gate_activation = silu(gate, stream)?;
        observer.observe(&format!("{prefix}.gate_activation"), &gate_activation)?;
        let current = gate_activation.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.down_proj_input"), &current)?;

        let selected_down = self.down_proj.as_ref().take_axis(top_k_index, 0, stream)?;
        observer.observe(&format!("{prefix}.selected_down_weight"), &selected_down)?;
        let route_output = matmul(
            current.try_index_device((.., .., NewAxis, ..), stream)?,
            selected_down.swap_axes(-1, -2, stream)?,
            stream,
        )?
        .reshape(&[num_tokens, top_k, self.hidden_dim], stream)?;
        observer.observe(&format!("{prefix}.route_output"), &route_output)?;
        let weighted = route_output.multiply(
            top_k_weights.try_index_device((.., .., NewAxis), stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.weighted_route_output"), &weighted)?;
        let output = sum_axis(&weighted, -2, false, stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(output)
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
        let gate_up = if let Some(affine) = self.gate_up_affine {
            Self::affine_grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                self.gate_up_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine gate/up scales"),
                self.gate_up_proj_biases
                    .as_ref()
                    .as_ref()
                    .expect("affine gate/up biases"),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.gate_up_proj_scale_inv.as_ref() {
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

        let current = if let Some(affine) = self.down_affine {
            Self::affine_grouped_linear(
                &current,
                self.down_proj.as_ref(),
                self.down_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine down scales"),
                self.down_proj_biases
                    .as_ref()
                    .as_ref()
                    .expect("affine down biases"),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.down_proj_scale_inv.as_ref() {
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
        common::weighted_route_sum(current, top_k_weights, &plan, num_tokens, stream)
    }

    fn forward_expert_major_chunk_with_observer(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        observer.observe(&format!("{prefix}.route_indices"), &plan.route_indices)?;
        observer.observe(&format!("{prefix}.token_indices"), &plan.token_indices)?;
        observer.observe(&format!("{prefix}.slot_indices"), &plan.slot_indices)?;
        observer.observe(
            &format!("{prefix}.sorted_group_ids"),
            &plan.sorted_group_ids,
        )?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        observer.observe(&format!("{prefix}.expert_major_input"), &hidden)?;
        let gate_up = if let Some(affine) = self.gate_up_affine {
            Self::affine_grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                self.gate_up_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine gate/up scales"),
                self.gate_up_proj_biases
                    .as_ref()
                    .as_ref()
                    .expect("affine gate/up biases"),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.gate_up_proj_scale_inv.as_ref() {
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
        observer.observe(&format!("{prefix}.expert_major_gate_up_proj"), &gate_up)?;
        let gate = gate_up.try_index_device((.., ..self.intermediate_dim), stream)?;
        observer.observe(&format!("{prefix}.expert_major_gate_proj"), &gate)?;
        let up = gate_up.try_index_device((.., self.intermediate_dim..), stream)?;
        observer.observe(&format!("{prefix}.expert_major_up_proj"), &up)?;
        let gate_activation = silu(gate, stream)?;
        observer.observe(
            &format!("{prefix}.expert_major_gate_activation"),
            &gate_activation,
        )?;
        let current = gate_activation.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.expert_major_down_proj_input"), &current)?;

        let route_output = if let Some(affine) = self.down_affine {
            Self::affine_grouped_linear(
                &current,
                self.down_proj.as_ref(),
                self.down_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine down scales"),
                self.down_proj_biases
                    .as_ref()
                    .as_ref()
                    .expect("affine down biases"),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.down_proj_scale_inv.as_ref() {
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
        observer.observe(
            &format!("{prefix}.expert_major_route_output"),
            &route_output,
        )?;
        let output =
            common::weighted_route_sum(route_output, top_k_weights, &plan, num_tokens, stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(output)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

/// Top-k router for Qwen3.5 MoE experts.
pub type TopKRouter = common::TopKRouter;

#[derive(Debug, Clone, ModuleParameters)]
/// Sparse MoE block with routed experts plus a shared expert.
pub struct SparseMoeBlock {
    #[param]
    /// Top-k router.
    pub gate: TopKRouter,
    #[param]
    /// Routed expert bank.
    pub experts: Experts,
    #[param]
    /// Shared expert MLP.
    pub shared_expert: Mlp,
    #[param]
    /// Gate applied to the shared expert output.
    pub shared_expert_gate: QwenLinear,
}

impl SparseMoeBlock {
    /// Creates an unloaded sparse MoE block.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(
            args,
            layer_idx,
            QwenWeightFormat::for_text(args, None),
            stream,
        )
    }

    fn new_with_format(
        args: &ModelArgs,
        layer_idx: usize,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate: TopKRouter::new(
                common::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.num_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Softmax,
                    norm_topk_prob: true,
                    normalization_epsilon: 0.0,
                    routed_scaling_factor: 1.0,
                    n_group: 1,
                    topk_group: 1,
                    score_correction_bias: false,
                },
                stream,
            )?,
            experts: Experts::new_with_format(args, layer_idx, format, stream)?,
            shared_expert: Mlp::new(
                args.hidden_size,
                args.shared_expert_intermediate_size,
                false,
                format,
                stream,
            )?,
            shared_expert_gate: QwenLinear::new(
                args.hidden_size,
                1,
                false,
                QwenWeightFormat::Dense,
                stream,
            )?,
        })
    }

    /// Forward pass that reports router and expert activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let b = shape[0];
        let l = shape[1];
        let h = shape[2];
        let flat = hidden_states.reshape(&[-1, h], stream)?;
        observer.observe(&format!("{prefix}.input_flat"), &flat)?;

        let shared_gate = sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?;
        observer.observe(&format!("{prefix}.shared_expert_gate"), &shared_gate)?;
        let shared = self
            .shared_expert
            .forward(&flat, stream)?
            .multiply(shared_gate, stream)?;
        observer.observe(&format!("{prefix}.shared_expert_output"), &shared)?;
        profile_array(PerfComponent::MoeShared, &shared)?;

        let routing =
            self.gate
                .forward_with_observer(&flat, stream, &format!("{prefix}.gate"), observer)?;
        let selected_experts = routing.indices;
        let selected_scores = routing.scores;
        let routing_weights = routing.weights;
        profile_arrays(
            PerfComponent::MoeRouter,
            &[&selected_experts, &routing_weights],
        )?;

        let routed = self.experts.forward_chunked_with_observer(
            &flat,
            &selected_experts,
            &routing_weights,
            stream,
            &format!("{prefix}.experts"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.routed_expert_output"), &routed)?;
        profile_array(PerfComponent::MoeRouted, &routed)?;

        let combined = routed.add(&shared, stream)?;
        observer.observe(&format!("{prefix}.combined_flat"), &combined)?;
        observer.observe_moe_routing(MoeRoutingObservation {
            prefix,
            selected_experts: &selected_experts,
            selected_scores: &selected_scores,
            routing_weights: &routing_weights,
            routed_output: &routed,
            shared_output: Some(&shared),
            combined_output: Some(&combined),
            num_experts: self.gate.num_experts,
        })?;
        let output = combined.reshape(&[b, l, h], stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        profile_array(PerfComponent::MoeCombine, &output)?;
        Ok(output)
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
/// Qwen3.5 MoE transformer block.
pub struct TransformerBlock {
    /// Layer kind.
    pub layer_type: LayerType,
    #[param]
    /// Full-attention layer when `layer_type` is [`LayerType::FullAttention`].
    pub self_attn: Option<FullAttention>,
    #[param]
    /// Linear-attention layer when `layer_type` is [`LayerType::LinearAttention`].
    pub linear_attn: Option<LinearAttention>,
    #[param]
    /// Sparse MoE feed-forward block, for `qwen35moe` checkpoints.
    pub mlp: Option<SparseMoeBlock>,
    #[param]
    /// Dense SwiGLU block, for `qwen35` checkpoints.
    pub dense_mlp: Option<Mlp>,
    #[param]
    /// Pre-attention normalization.
    pub input_layernorm: Qwen3NextRmsNorm,
    #[param]
    /// Pre-MoE normalization.
    pub post_attention_layernorm: Qwen3NextRmsNorm,
}

impl TransformerBlock {
    /// Creates an unloaded transformer block.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(
            args,
            layer_idx,
            QwenWeightFormat::for_text(args, None),
            stream,
        )
    }

    fn new_with_format(
        args: &ModelArgs,
        layer_idx: usize,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let layer_type = args.layer_type(layer_idx);
        Ok(Self {
            layer_type,
            self_attn: if layer_type == LayerType::FullAttention {
                Some(FullAttention::new_with_format(args, format, stream)?)
            } else {
                None
            },
            linear_attn: if layer_type == LayerType::LinearAttention {
                Some(LinearAttention::new_with_format(args, format, stream)?)
            } else {
                None
            },
            mlp: args
                .is_moe()
                .then(|| SparseMoeBlock::new_with_format(args, layer_idx, format, stream))
                .transpose()?,
            dense_mlp: (!args.is_moe())
                .then(|| {
                    Mlp::new(
                        args.hidden_size,
                        args.intermediate_size,
                        false,
                        format,
                        stream,
                    )
                })
                .transpose()?,
            input_layernorm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
            post_attention_layernorm: Qwen3NextRmsNorm::new(
                args.hidden_size,
                args.rms_norm_eps,
                stream,
            )?,
        })
    }
}

/// Input for a Qwen3.5 transformer block.
pub struct BlockInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional layer cache.
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
        let h = if let Some(mlp) = &mut self.mlp {
            mlp.forward(&post_normed, stream)?
        } else {
            self.dense_mlp
                .as_mut()
                .expect("dense Qwen3.5 MLP")
                .forward(&post_normed, stream)?
        };
        residual.add(h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        if let Some(full_attention) = &mut self.self_attn {
            full_attention.training_mode(mode);
        }
        if let Some(linear_attention) = &mut self.linear_attn {
            linear_attention.training_mode(mode);
        }
        if let Some(mlp) = &mut self.mlp {
            mlp.training_mode(mode);
        }
        if let Some(mlp) = &mut self.dense_mlp {
            mlp.training_mode(mode);
        }
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

impl TransformerBlock {
    /// Forward pass that reports Qwen3.5 MoE block activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: BlockInput<'_>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let BlockInput { x, mask, cache } = input;
        observer.observe(&format!("{prefix}.input"), x)?;
        observer.observe(&format!("{prefix}.residual_before_attention"), x)?;
        let residual = x;
        let h = self.input_layernorm.forward(x, stream)?;
        observer.observe(&format!("{prefix}.input_layernorm"), &h)?;
        let h = match (self.layer_type, cache) {
            (LayerType::FullAttention, Some(LayerCache::FullAttention(cache))) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_with_observer(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                    &format!("{prefix}.self_attn"),
                    observer,
                )?,
            (LayerType::FullAttention, _) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_with_observer(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                    &format!("{prefix}.self_attn"),
                    observer,
                )?,
            (LayerType::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_with_observer(
                    LinearAttentionInput {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                    &format!("{prefix}.linear_attn"),
                    observer,
                )?,
            (LayerType::LinearAttention, _) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_with_observer(
                    LinearAttentionInput { x: &h, cache: None },
                    stream,
                    &format!("{prefix}.linear_attn"),
                    observer,
                )?,
        };
        observer.observe(&format!("{prefix}.attention_output"), &h)?;
        observer.observe(&format!("{prefix}.residual_delta_attention"), &h)?;
        match self.layer_type {
            LayerType::FullAttention => profile_array(PerfComponent::FullAttention, &h)?,
            LayerType::LinearAttention => profile_array(PerfComponent::LinearAttention, &h)?,
        }
        let h = residual.add(h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_residual"), &h)?;
        observer.observe(&format!("{prefix}.residual_after_attention"), &h)?;

        let feed_forward_name = if self.mlp.is_some() { "moe" } else { "mlp" };
        observer.observe(&format!("{prefix}.residual_before_{feed_forward_name}"), &h)?;
        let residual = h.clone();
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &post_normed)?;
        let h = if let Some(mlp) = &mut self.mlp {
            mlp.forward_with_observer(&post_normed, stream, &format!("{prefix}.moe"), observer)?
        } else {
            self.dense_mlp
                .as_mut()
                .expect("dense Qwen3.5 MLP")
                .forward_with_observer(&post_normed, stream, &format!("{prefix}.mlp"), observer)?
        };
        observer.observe(&format!("{prefix}.{feed_forward_name}_output"), &h)?;
        observer.observe(&format!("{prefix}.residual_delta_{feed_forward_name}"), &h)?;
        let output = residual.add(h, stream)?;
        let output = observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output);
        observer.observe(&format!("{prefix}.output"), &output)?;
        observer.observe(
            &format!("{prefix}.residual_after_{feed_forward_name}"),
            &output,
        )?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3.5 MoE text transformer body without the language-model head.
pub struct Qwen35MoeTextModel {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    #[param]
    /// Token embedding table.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    /// Transformer blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final normalization.
    pub norm: Qwen3NextRmsNorm,
}

impl Qwen35MoeTextModel {
    /// Creates an unloaded Qwen3.5 MoE text transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(args, QwenWeightFormat::for_text(args, None), stream)
    }

    fn new_with_format(
        args: &ModelArgs,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let embed_tokens = common::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            format.affine(),
            stream,
        )?;
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new_with_format(args, idx as usize, format, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens,
            layers,
            norm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
        })
    }

    /// Forward pass that reports activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let ModelInput {
            inputs,
            inputs_embeds,
            mask,
            mut cache,
        } = input;
        let mut h = match inputs_embeds {
            Some(inputs_embeds) => inputs_embeds.clone(),
            None => self.embed_tokens.forward(inputs, stream)?,
        };
        observer.observe("model.embed_tokens", &h)?;
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
        if let Some(mask) = mask.as_ref() {
            observer.observe("model.attention_mask", mask)?;
        }

        if let Some(cache) = cache.as_mut() {
            for (i, (layer, layer_cache)) in self
                .layers
                .iter_mut()
                .zip(cache.layers.iter_mut())
                .enumerate()
            {
                h = layer.forward_with_observer(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: Some(layer_cache),
                    },
                    stream,
                    &format!("model.layers.{i}"),
                    observer,
                )?;
            }
        } else {
            for (i, layer) in self.layers.iter_mut().enumerate() {
                h = layer.forward_with_observer(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: None,
                    },
                    stream,
                    &format!("model.layers.{i}"),
                    observer,
                )?;
            }
        }
        let h = self.norm.forward(&h, stream)?;
        observer.observe("model.norm", &h)?;
        profile_array(PerfComponent::FinalNorm, &h)?;
        Ok(h)
    }
}

/// Input for a Qwen3.5 MoE text forward pass.
pub struct ModelInput<'a> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional prepared embeddings with shape `[batch, sequence, hidden]`.
    pub inputs_embeds: Option<&'a Array>,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional heterogeneous cache.
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
            inputs_embeds,
            mask,
            mut cache,
        } = input;
        let mut h = match inputs_embeds {
            Some(inputs_embeds) => inputs_embeds.clone(),
            None => self.embed_tokens.forward(inputs, stream)?,
        };
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
/// Qwen3.5 MoE causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    /// Optional vision configuration.
    pub vision_args: Option<VisionConfig>,
    /// Optional image token id rejected by text-only generation.
    pub image_token_id: Option<i32>,
    /// Optional video media token id.
    pub video_token_id: Option<i32>,
    #[param]
    /// Optional Qwen vision encoder.
    pub visual: Option<QwenVisionTransformer>,
    #[param]
    /// Text transformer body.
    pub model: Qwen35MoeTextModel,
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded Qwen3.5 MoE causal language model.
    pub fn new(
        args: ModelArgs,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_args: Option<VisionConfig>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_affine(
            args,
            image_token_id,
            video_token_id,
            vision_args,
            None,
            stream,
        )
    }

    fn new_with_affine(
        args: ModelArgs,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_args: Option<VisionConfig>,
        affine: Option<AffineQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let format = QwenWeightFormat::for_text(&args, affine);
        let model = Qwen35MoeTextModel::new_with_format(&args, format, stream)?;
        let visual = vision_args
            .clone()
            .map(|vision_args| QwenVisionTransformer::new(vision_args, stream))
            .transpose()?;
        let lm_head = if !args.tie_word_embeddings {
            Some(
                common::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    format.affine(),
                    stream,
                )?,
            )
        } else {
            None
        };
        Ok(Self {
            args,
            vision_args,
            image_token_id,
            video_token_id,
            visual,
            model,
            lm_head,
        })
    }

    /// Creates an empty heterogeneous cache for this model.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn reject_multimodal_tokens(
        &self,
        inputs: &Array,
        allow_visual: bool,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for (name, token_id) in [
            (
                "image",
                (!allow_visual).then_some(self.image_token_id).flatten(),
            ),
            (
                "video",
                (!allow_visual).then_some(self.video_token_id).flatten(),
            ),
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
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
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
        self.reject_multimodal_tokens(input.inputs, input.inputs_embeds.is_some(), stream)?;
        let hidden_states = self.model.forward(input, stream)?;
        let hidden_states = if last_token_only {
            hidden_states.try_index_device((.., -1, ..), stream)?
        } else {
            hidden_states
        };
        self.project_logits(&hidden_states, stream)
    }

    /// Forward pass that reports activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        self.reject_multimodal_tokens(input.inputs, input.inputs_embeds.is_some(), stream)?;
        let hidden_states = self.model.forward_with_observer(input, stream, observer)?;
        observer.observe("model.output", &hidden_states)?;
        let logits = self.project_logits(&hidden_states, stream)?;
        observer.observe("lm_head.logits", &logits)?;
        Ok(logits)
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
        if let Some(visual) = &mut self.visual {
            visual.training_mode(mode);
        }
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub(crate) struct LoadedQwen35Gguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads a text-only dense or MoE Qwen3.5 model from a GGUF checkpoint.
pub fn load_qwen3_5_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    Ok(load_qwen3_5_moe_gguf_with_metadata(gguf_file, stream, weights_stream)?.model)
}

/// Backward-compatible name for loading a Qwen3.5 GGUF checkpoint.
pub fn load_qwen3_5_moe_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    load_qwen3_5_gguf(gguf_file, stream, weights_stream)
}

pub(crate) fn load_qwen3_5_moe_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedQwen35Gguf, Error> {
    let (arrays, metadata) = Array::load_gguf_with_metadata(gguf_file, weights_stream)?;
    load_qwen3_5_moe_gguf_data(arrays, metadata, stream, weights_stream)
}

pub(crate) fn load_qwen3_5_moe_gguf_data(
    arrays: HashMap<String, Array>,
    metadata: HashMap<String, GgufMetadataValue>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedQwen35Gguf, Error> {
    let architecture = qwen35_gguf_string(&metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "qwen35" | "qwen35moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports qwen35 and qwen35moe"
        )));
    }
    let is_moe = architecture == "qwen35moe";
    if arrays
        .keys()
        .any(|name| name.starts_with("v.") || name.starts_with("mm."))
    {
        return Err(Error::UnsupportedArchitecture(
            "multimodal Qwen3.5 GGUF checkpoints are not supported; load a text-only qwen35 or qwen35moe checkpoint"
                .into(),
        ));
    }

    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let block_count = qwen35_gguf_i32(&metadata, &key("block_count"), weights_stream)?;
    let nextn_layers =
        qwen35_gguf_optional_i64(&metadata, &key("nextn_predict_layers"), weights_stream)?
            .unwrap_or(0);
    let nextn_layers = i32::try_from(nextn_layers).map_err(|_| {
        Error::UnsupportedArchitecture(
            "Qwen3.5 next-token prediction layer count exceeds i32".into(),
        )
    })?;
    if nextn_layers < 0 || nextn_layers >= block_count {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF has invalid block_count {block_count} and nextn_predict_layers {nextn_layers}"
        )));
    }
    let num_hidden_layers = block_count - nextn_layers;

    // Released Qwen3.5 GGUF files append MTP tensors as additional blk.N entries.
    // The text runtime, like the safetensors loader, intentionally ignores them.
    let arrays = arrays
        .into_iter()
        .filter(|(name, _)| {
            qwen35_gguf_block_index(name).is_none_or(|index| index < num_hidden_layers)
        })
        .collect::<HashMap<_, _>>();
    let mut args = qwen35_args_from_gguf(
        &arrays,
        &metadata,
        &architecture,
        num_hidden_layers,
        weights_stream,
    )?;

    let arrays = qwen35_dequantize_non_experts(arrays, weights_stream)?;
    let mut translated = HashMap::with_capacity(arrays.len());
    for (name, value) in arrays {
        let (name, value) = qwen35_translate_gguf_weight(name, value, &args, weights_stream)?;
        if translated.insert(name.clone(), value).is_some() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3.5 GGUF tensors collide after translating {name:?}"
            )));
        }
    }
    if is_moe {
        qwen35_pack_expert_banks(&mut translated, &args, weights_stream)?;
    }
    let quantized_weight_configs = qwen35_gguf_quantized_weight_configs(&translated)?;
    args.quantized_weight_configs = Some(quantized_weight_configs);

    let mut model = Model::new(args, None, None, None, stream)?;
    let config = qwen3_5_moe_strict_load_config(false).allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    load_arrays_strict(&mut model, translated, &config, &mut report)?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;

    let eos_token_ids =
        qwen35_gguf_optional_i64(&metadata, "tokenizer.ggml.eos_token_id", weights_stream)?
            .and_then(|value| u32::try_from(value).ok())
            .into_iter()
            .collect();
    Ok(LoadedQwen35Gguf {
        model,
        eos_token_ids,
    })
}

fn qwen35_args_from_gguf(
    arrays: &HashMap<String, Array>,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    num_hidden_layers: i32,
    stream: &Stream,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let is_moe = architecture == "qwen35moe";
    let hidden_size = qwen35_gguf_i32(metadata, &key("embedding_length"), stream)?;
    let num_attention_heads = qwen35_gguf_i32(metadata, &key("attention.head_count"), stream)?;
    let num_key_value_heads = qwen35_gguf_i32(metadata, &key("attention.head_count_kv"), stream)?;
    let head_dim = qwen35_gguf_optional_i64(metadata, &key("attention.key_length"), stream)?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("Qwen3.5 head size exceeds i32".into()))?
        .unwrap_or(hidden_size / num_attention_heads);
    let rope_dims = qwen35_gguf_optional_i64(metadata, &key("rope.dimension_count"), stream)?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("Qwen3.5 RoPE dimension exceeds i32".into()))?
        .unwrap_or(head_dim / 4);
    let full_attention_interval =
        qwen35_gguf_optional_i64(metadata, &key("full_attention_interval"), stream)?.unwrap_or(4);
    let full_attention_interval = usize::try_from(full_attention_interval).map_err(|_| {
        Error::UnsupportedArchitecture("Qwen3.5 full attention interval must be positive".into())
    })?;
    if full_attention_interval == 0 {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3.5 full attention interval must be positive".into(),
        ));
    }
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
        None => qwen35_gguf_i32(metadata, &key("vocab_size"), stream)?,
    };
    let rope_theta =
        qwen35_gguf_optional_f32(metadata, &key("rope.freq_base"), stream)?.unwrap_or(10_000_000.0);
    let mut rope_parameters = HashMap::new();
    rope_parameters.insert("rope_theta".into(), serde_json::json!(rope_theta));
    rope_parameters.insert(
        "partial_rotary_factor".into(),
        serde_json::json!(rope_dims as f32 / head_dim as f32),
    );

    Ok(ModelArgs {
        model_type: if is_moe {
            "qwen3_5_moe_text".into()
        } else {
            "qwen3_5_text".into()
        },
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings: qwen35_gguf_i32(metadata, &key("context_length"), stream)?,
        rms_norm_eps: qwen35_gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"), stream)?,
        tie_word_embeddings: !arrays.contains_key("output.weight"),
        attention_bias: arrays.keys().any(|name| {
            name.ends_with("attn_q.bias")
                || name.ends_with("attn_k.bias")
                || name.ends_with("attn_v.bias")
                || name.ends_with("attn_output.bias")
        }),
        hidden_act: "silu".into(),
        linear_conv_kernel_dim: qwen35_gguf_i32(metadata, &key("ssm.conv_kernel"), stream)?,
        linear_key_head_dim: qwen35_gguf_i32(metadata, &key("ssm.state_size"), stream)?,
        linear_value_head_dim: qwen35_gguf_i32(metadata, &key("ssm.state_size"), stream)?,
        linear_num_key_heads: qwen35_gguf_i32(metadata, &key("ssm.group_count"), stream)?,
        linear_num_value_heads: qwen35_gguf_i32(metadata, &key("ssm.time_step_rank"), stream)?,
        intermediate_size: if is_moe {
            0
        } else {
            qwen35_gguf_i32(metadata, &key("feed_forward_length"), stream)?
        },
        moe_intermediate_size: if is_moe {
            qwen35_gguf_i32(metadata, &key("expert_feed_forward_length"), stream)?
        } else {
            0
        },
        shared_expert_intermediate_size: if is_moe {
            qwen35_gguf_i32(metadata, &key("expert_shared_feed_forward_length"), stream)?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            qwen35_gguf_i32(metadata, &key("expert_used_count"), stream)?
        } else {
            0
        },
        num_experts: if is_moe {
            qwen35_gguf_i32(metadata, &key("expert_count"), stream)?
        } else {
            0
        },
        norm_topk_prob: true,
        layer_types: (0..num_hidden_layers as usize)
            .map(|index| {
                if (index + 1) % full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect(),
        rope_parameters: Some(rope_parameters),
        rope_scaling: None,
        quantization_config: None,
        quantized_weight_configs: None,
    })
}

fn qwen35_gguf_block_index(name: &str) -> Option<i32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

fn qwen35_is_routed_expert_weight(name: &str) -> bool {
    name.contains(".ffn_gate_exps.")
        || name.contains(".ffn_up_exps.")
        || name.contains(".ffn_down_exps.")
}

fn qwen35_dequantize_non_experts(
    mut arrays: HashMap<String, Array>,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let scale_names = arrays
        .keys()
        .filter(|name| name.ends_with(".scales"))
        .cloned()
        .collect::<Vec<_>>();
    for scales_name in scale_names {
        if qwen35_is_routed_expert_weight(&scales_name) {
            continue;
        }
        let prefix = scales_name
            .strip_suffix(".scales")
            .expect("filtered GGUF scale suffix");
        let weight_name = format!("{prefix}.weight");
        let biases_name = format!("{prefix}.biases");
        let scales = arrays.remove(&scales_name).ok_or_else(|| {
            Error::Quantization(format!("missing GGUF scales tensor {scales_name:?}"))
        })?;
        let weight = arrays.remove(&weight_name).ok_or_else(|| {
            Error::Quantization(format!(
                "GGUF quantization scales {scales_name:?} have no matching weight"
            ))
        })?;
        let biases = arrays.remove(&biases_name).ok_or_else(|| {
            Error::Quantization(format!(
                "GGUF quantized tensor {weight_name:?} is missing affine biases"
            ))
        })?;
        let config = qwen35_gguf_affine_quantization(weight.shape(), scales.shape(), &weight_name)?;
        let value = dequantize(
            &weight,
            &scales,
            &biases,
            config.group_size,
            config.bits,
            stream,
        )?;
        arrays.insert(weight_name, value);
    }
    Ok(arrays)
}

fn qwen35_translate_gguf_weight(
    name: String,
    mut value: Array,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(String, Array), Error> {
    if name.ends_with(".attn_qkv.weight") {
        value = qwen35_restore_v_tail(
            value,
            2 * args.linear_num_key_heads * args.linear_key_head_dim,
            0,
            args,
            stream,
        )?;
    } else if name.ends_with(".ssm_conv1d.weight") {
        value = qwen35_restore_v_tail(
            value,
            2 * args.linear_num_key_heads * args.linear_key_head_dim,
            0,
            args,
            stream,
        )?;
        value = value.reshape(&[value.dim(0), 1, value.dim(1)], stream)?;
    } else if name.ends_with(".attn_gate.weight") {
        value = qwen35_restore_v_head_order(value, 0, args.linear_value_head_dim, args, stream)?;
    } else if name.ends_with(".ssm_alpha.weight")
        || name.ends_with(".ssm_beta.weight")
        || name.ends_with(".ssm_dt.bias")
    {
        value = qwen35_restore_v_head_order(value, 0, 1, args, stream)?;
    } else if name.ends_with(".ssm_a") {
        value = qwen35_restore_v_head_order(value, 0, 1, args, stream)?
            .multiply(Array::from_f32(-1.0), stream)?
            .log(stream)?;
    } else if name.ends_with(".ssm_out.weight") {
        value = qwen35_restore_v_head_order(value, 1, args.linear_value_head_dim, args, stream)?;
    } else if name.ends_with(".ffn_gate_inp_shexp.weight") && value.ndim() == 1 {
        value = value.reshape(&[1, value.dim(0)], stream)?;
    }

    if qwen35_is_offset_norm(&name) {
        value = value.subtract(Array::from_f32(1.0), stream)?;
    }
    Ok((qwen35_translate_gguf_weight_name(&name), value))
}

fn qwen35_restore_v_tail(
    value: Array,
    prefix: i32,
    axis: usize,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Array, Error> {
    if axis != 0 || value.ndim() != 2 {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3.5 GGUF value-tail restoration expects a rank-2 row projection".into(),
        ));
    }
    if prefix <= 0 || prefix >= value.dim(0) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF value-tail split {prefix} is invalid for shape {:?}",
            value.shape()
        )));
    }
    let leading = value.try_index_device((..prefix, ..), stream)?;
    let tail = value.try_index_device((prefix.., ..), stream)?;
    let tail = qwen35_restore_v_head_order(tail, 0, args.linear_value_head_dim, args, stream)?;
    Ok(concatenate_axis(&[leading, tail], 0, stream)?)
}

fn qwen35_restore_v_head_order(
    value: Array,
    axis: usize,
    head_dim: i32,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Array, Error> {
    let num_k = args.linear_num_key_heads;
    let num_v = args.linear_num_value_heads;
    if num_k <= 0 || num_v % num_k != 0 || axis >= value.ndim() {
        return Err(Error::UnsupportedArchitecture(format!(
            "invalid Qwen3.5 GGUF value-head layout: {num_v} value heads, {num_k} key heads"
        )));
    }
    let original_shape = value.shape().to_vec();
    if original_shape[axis] != num_v * head_dim {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF value-head axis has length {}, expected {} in shape {:?}",
            original_shape[axis],
            num_v * head_dim,
            original_shape
        )));
    }
    let repeats = num_v / num_k;
    let mut expanded_shape = original_shape.clone();
    expanded_shape.splice(axis..=axis, [repeats, num_k, head_dim]);
    let expanded = value.reshape(&expanded_shape, stream)?;
    let mut axes = (0..expanded_shape.len() as i32).collect::<Vec<_>>();
    axes.swap(axis, axis + 1);
    Ok(expanded
        .transpose_axes(&axes, stream)?
        .reshape(&original_shape, stream)?)
}

fn qwen35_is_offset_norm(name: &str) -> bool {
    name == "output_norm.weight"
        || (name.starts_with("blk.")
            && name.ends_with("_norm.weight")
            && !name.ends_with("ssm_norm.weight"))
}

fn qwen35_translate_gguf_weight_name(name: &str) -> String {
    const ROOTS: [(&str, &str); 3] = [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
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
    const PARAMETERS: [(&str, &str); 29] = [
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_qkv", "linear_attn.in_proj_qkv"),
        ("attn_gate", "linear_attn.in_proj_z"),
        ("ssm_beta", "linear_attn.in_proj_b"),
        ("ssm_alpha", "linear_attn.in_proj_a"),
        ("ssm_conv1d", "linear_attn.conv1d"),
        ("ssm_dt.bias", "linear_attn.dt_bias"),
        ("ssm_a", "linear_attn.A_log"),
        ("ssm_norm", "linear_attn.norm"),
        ("ssm_out", "linear_attn.out_proj"),
        ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
        ("ffn_gate_inp", "mlp.gate"),
        ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
        ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
        ("ffn_gate", "dense_mlp.gate_proj"),
        ("ffn_up", "dense_mlp.up_proj"),
        ("ffn_down", "dense_mlp.down_proj"),
        ("rope_freqs", "rope_freqs"),
    ];
    for (source, target) in PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut suffix = parameter.strip_prefix(source).unwrap_or_default();
            if target.starts_with("mlp.experts.") {
                suffix = match suffix {
                    ".weight" => "",
                    ".scales" => "_scales",
                    ".biases" => "_biases",
                    other => other,
                };
            }
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    name.to_string()
}

fn qwen35_pack_expert_banks(
    arrays: &mut HashMap<String, Array>,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(), Error> {
    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        for suffix in ["", "_scales", "_biases"] {
            let gate_name = format!("{prefix}.gate_proj{suffix}");
            let up_name = format!("{prefix}.up_proj{suffix}");
            let gate = arrays.remove(&gate_name).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5 GGUF is missing routed expert tensor {gate_name:?}"
                ))
            })?;
            let up = arrays.remove(&up_name).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5 GGUF is missing routed expert tensor {up_name:?}"
                ))
            })?;
            if gate.shape().len() != 3 || gate.shape()[0] != args.num_experts {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3.5 GGUF expert tensor {gate_name:?} has invalid shape {:?}",
                    gate.shape()
                )));
            }
            if gate.shape()[0] != up.shape()[0] || gate.shape()[2] != up.shape()[2] {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3.5 GGUF gate/up expert tensor shapes are incompatible: {:?} and {:?}",
                    gate.shape(),
                    up.shape()
                )));
            }
            arrays.insert(
                format!("{prefix}.gate_up_proj{suffix}"),
                concatenate_axis(&[gate, up], 1, stream)?,
            );
        }
        for suffix in ["", "_scales", "_biases"] {
            let source = format!("{prefix}.down_proj{suffix}");
            let value = arrays.remove(&source).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5 GGUF is missing routed expert tensor {source:?}"
                ))
            })?;
            arrays.insert(source, value);
        }
    }
    Ok(())
}

fn qwen35_gguf_quantized_weight_configs(
    arrays: &HashMap<String, Array>,
) -> Result<HashMap<String, AffineQuantization>, Error> {
    let mut configs = HashMap::new();
    for (scales_name, scales) in arrays {
        let Some(weight_name) = scales_name.strip_suffix("_scales") else {
            continue;
        };
        let Some(weight) = arrays.get(weight_name) else {
            continue;
        };
        let config = qwen35_gguf_affine_quantization(weight.shape(), scales.shape(), weight_name)?;
        configs.insert(weight_name.to_string(), config);
    }
    Ok(configs)
}

fn qwen35_gguf_affine_quantization(
    weight_shape: &[i32],
    scales_shape: &[i32],
    weight_name: &str,
) -> Result<AffineQuantization, Error> {
    crate::quantization::gguf_affine_quantization(weight_shape, scales_shape, weight_name)
}

fn qwen35_gguf_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<String, Error> {
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

fn qwen35_gguf_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<i32, Error> {
    i32::try_from(qwen35_gguf_i64(metadata, key, stream)?).map_err(|_| {
        Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
    })
}

fn qwen35_gguf_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<i64, Error> {
    qwen35_gguf_optional_i64(metadata, key, stream)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn qwen35_gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    _stream: &Stream,
) -> Result<Option<i64>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {key:?} must be an integer scalar"
            ))
        }),
        None => Ok(None),
    }
}

fn qwen35_gguf_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<f32, Error> {
    qwen35_gguf_optional_f32(metadata, key, stream)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn qwen35_gguf_optional_f32(
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

/// Loads `tokenizer.json` from a Qwen3.5 MoE model directory.
pub fn load_qwen3_5_moe_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads and normalizes Qwen3.5 MoE model arguments from `config.json`.
pub fn get_qwen3_5_moe_model_args(
    model_dir: impl AsRef<Path>,
) -> Result<(ModelArgs, Option<i32>, Option<i32>, Option<VisionConfig>), Error> {
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
    Ok((
        args,
        config.image_token_id,
        config.video_token_id,
        config.vision_config,
    ))
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

/// Loads a Qwen3.5 MoE model and safetensors weights from a model directory.
pub fn load_qwen3_5_moe_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id, vision_config) =
        get_qwen3_5_moe_model_args(model_dir)?;
    if let Some(quantization_config) = &args.quantization_config {
        quantization_config.validate_supported()?;
    }
    let uses_fp8 = args.quantization_config.is_some();
    let load_visual = vision_config.is_some();
    let mut model = Model::new(args, image_token_id, video_token_id, vision_config, stream)?;
    let config = qwen3_5_moe_strict_load_config(load_visual);
    let mut report = StrictLoadReport::default();
    for weight_file in safetensors_files(model_dir)? {
        load_qwen3_5_moe_safetensors_strict(
            &mut model,
            weight_file,
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

/// Loads a dense Qwen3.5/3.6 MoE checkpoint while affine-quantizing eligible
/// text weights, including the packed rank-3 routed expert banks.
pub fn load_qwen3_5_moe_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: AffineQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    quantization.validate()?;
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id, vision_config) =
        get_qwen3_5_moe_model_args(model_dir)?;
    if args.quantization_config.is_some() {
        return Err(Error::Quantization(
            "Qwen3.5/3.6 MoE affine on-load quantization requires a dense checkpoint; native FP8 checkpoints cannot be requantized to affine Q4".into(),
        ));
    }
    let load_visual = vision_config.is_some();
    let mut model = Model::new_with_affine(
        args,
        image_token_id,
        video_token_id,
        vision_config,
        Some(quantization),
        stream,
    )?;
    let config = qwen3_5_moe_strict_load_config(load_visual);
    let mut report = StrictLoadReport::default();
    for weight_file in safetensors_files(model_dir)? {
        load_qwen3_5_moe_affine_safetensors_strict(
            &mut model,
            weight_file,
            weights_stream,
            stream,
            quantization,
            &config,
            &mut report,
        )?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

fn is_packed_expert_weight(key: &str) -> bool {
    key.ends_with(".mlp.experts.gate_up_proj") || key.ends_with(".mlp.experts.down_proj")
}

fn quantize_packed_expert_tensor(
    value: &Array,
    quantization: AffineQuantization,
    stream: &Stream,
) -> Result<QuantizedTensor, Error> {
    if value.ndim() != 3 || !value.dtype().is_float() {
        return Err(Error::Quantization(format!(
            "expected a floating-point rank-3 expert bank, got shape {:?} and dtype {:?}",
            value.shape(),
            value.dtype()
        )));
    }
    let shape = value.shape();
    let experts = shape[0];
    let output_dims = shape[1];
    let input_dims = shape[2];
    let matrix = value.reshape(&[experts * output_dims, input_dims], stream)?;
    let quantized = quantize_tensor(&matrix, quantization, stream)?;
    let packed_per_int = 32 / quantization.bits;
    Ok(QuantizedTensor {
        weight: quantized
            .weight
            .reshape(&[experts, output_dims, input_dims / packed_per_int], stream)?,
        scales: quantized.scales.reshape(
            &[experts, output_dims, input_dims / quantization.group_size],
            stream,
        )?,
        biases: quantized.biases.reshape(
            &[experts, output_dims, input_dims / quantization.group_size],
            stream,
        )?,
    })
}

fn load_qwen3_5_moe_affine_safetensors_strict(
    model: &mut Model,
    path: impl AsRef<Path>,
    weights_stream: &Stream,
    quantization_stream: &Stream,
    quantization: AffineQuantization,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let mut params = model.parameters_mut().flatten();
    for_each_safetensor_array(path, weights_stream, |key, value| {
        if is_packed_expert_weight(&key) {
            let quantized =
                quantize_packed_expert_tensor(&value, quantization, quantization_stream)?;
            eval([&quantized.weight, &quantized.scales, &quantized.biases])?;
            quantization_stream.synchronize()?;
            load_array_strict(&mut params, key.clone(), quantized.weight, config, report);
            load_array_strict(
                &mut params,
                format!("{key}_scales"),
                quantized.scales,
                config,
                report,
            );
            load_array_strict(
                &mut params,
                format!("{key}_biases"),
                quantized.biases,
                config,
                report,
            );
            Ok(())
        } else {
            load_array_quantized_strict(
                &mut params,
                key,
                value,
                quantization_stream,
                quantization,
                config,
                report,
            )
        }
    })
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

    load_qwen3_5_moe_fp8_safetensors_strict(
        model,
        path,
        weights_stream,
        transform_stream,
        config,
        report,
    )
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

fn load_qwen3_5_moe_fp8_safetensors_strict(
    model: &mut Model,
    path: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    let num_experts = model.args.num_experts;
    let mut expert_parts: HashMap<(String, i32), Fp8ExpertParts> = HashMap::new();
    let mut params = model.parameters_mut().flatten();

    for_each_safetensor_array(path, weights_stream, |key, value| {
        if let Some((prefix, expert, projection)) = parse_fp8_expert_projection_key(&key) {
            let parts = expert_parts.entry((prefix, expert)).or_default();
            set_fp8_expert_part(parts, projection, value, false);
        } else if let Some((prefix, expert, projection)) = parse_fp8_expert_scale_key(&key) {
            let parts = expert_parts.entry((prefix, expert)).or_default();
            set_fp8_expert_part(parts, projection, value, true);
        } else {
            load_array_strict(&mut params, key, value, config, report);
        }
        Ok(())
    })?;

    let mut layer_prefixes = expert_parts
        .keys()
        .map(|(prefix, _)| prefix.clone())
        .collect::<Vec<_>>();
    layer_prefixes.sort();
    layer_prefixes.dedup();

    for prefix in layer_prefixes {
        for (key, value) in
            pack_fp8_expert_prefix(&mut expert_parts, &prefix, num_experts, transform_stream)?
        {
            load_array_strict(&mut params, key, value, config, report);
        }
    }

    Ok(())
}

fn set_fp8_expert_part(
    parts: &mut Fp8ExpertParts,
    projection: Fp8ExpertProjection,
    value: Array,
    is_scale: bool,
) {
    match (projection, is_scale) {
        (Fp8ExpertProjection::Gate, false) => parts.gate = Some(value),
        (Fp8ExpertProjection::Gate, true) => parts.gate_scale = Some(value),
        (Fp8ExpertProjection::Up, false) => parts.up = Some(value),
        (Fp8ExpertProjection::Up, true) => parts.up_scale = Some(value),
        (Fp8ExpertProjection::Down, false) => parts.down = Some(value),
        (Fp8ExpertProjection::Down, true) => parts.down_scale = Some(value),
    }
}

fn pack_fp8_expert_prefix(
    expert_parts: &mut HashMap<(String, i32), Fp8ExpertParts>,
    prefix: &str,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut gate_up = Vec::with_capacity(num_experts as usize);
    let mut gate_up_scale = Vec::with_capacity(num_experts as usize);
    let mut down = Vec::with_capacity(num_experts as usize);
    let mut down_scale = Vec::with_capacity(num_experts as usize);
    for expert in 0..num_experts {
        let parts = expert_parts
            .remove(&(prefix.to_string(), expert))
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
    Ok(HashMap::from([
        (format!("{prefix}.gate_up_proj"), gate_up_proj),
        (
            format!("{prefix}.gate_up_proj_scale_inv"),
            gate_up_proj_scale,
        ),
        (format!("{prefix}.down_proj"), down_proj),
        (format!("{prefix}.down_proj_scale_inv"), down_proj_scale),
    ]))
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

fn qwen3_5_moe_strict_load_config(load_visual: bool) -> StrictLoadConfig {
    let config = StrictLoadConfig::default()
        .rewrite_prefix("model.language_model.", "model.")
        .rewrite_prefix("language_model.", "model.")
        .rewrite_prefix("model.model.", "model.")
        .rewrite_prefix("vision_tower.", "visual.")
        .rewrite_prefix("model.visual.", "visual.")
        .rewrite_prefix("model.vision_tower.", "visual.")
        .rewrite_prefix("visual.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("visual.merger.mlp.2.", "visual.merger.mlp.fc2.")
        .rewrite_prefix("vision_tower.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("vision_tower.merger.mlp.2.", "visual.merger.mlp.fc2.")
        .rewrite_prefix("model.visual.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("model.visual.merger.mlp.2.", "visual.merger.mlp.fc2.")
        .rewrite_prefix("model.vision_tower.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("model.vision_tower.merger.mlp.2.", "visual.merger.mlp.fc2.")
        .allow_unused_prefix("mtp.");
    if load_visual {
        config
    } else {
        config
            .allow_unused_prefix("visual.")
            .allow_unused_prefix("vision_tower.")
            .allow_unused_prefix("model.visual.")
            .allow_unused_prefix("model.vision_tower.")
    }
}

impl Model {
    fn prepare_typed_prefill(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<runtime_input::PreparedPrefill, Exception> {
        let modality_tokens = [
            self.image_token_id
                .map(|token_id| runtime_input::ModalityToken {
                    modality: runtime_input::Modality::Image,
                    token_id: token_id as u32,
                }),
            self.video_token_id
                .map(|token_id| runtime_input::ModalityToken {
                    modality: runtime_input::Modality::Video,
                    token_id: token_id as u32,
                }),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let embed_tokens = &mut self.model.embed_tokens;
        let visual = &mut self.visual;
        let vision_args = self.vision_args.as_ref();
        let prepared = runtime_input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.hidden_size,
            "qwen3_5_moe",
            stream,
            |tokens, stream| embed_tokens.forward(tokens, stream),
            |part, stream| {
                let embeddings = visual_embeddings_from_payload(visual, part, stream)?;
                if part.modality == runtime_input::Modality::Video {
                    video_embedding_chunks(vision_args, part, &embeddings, stream)
                } else {
                    Ok(vec![embeddings])
                }
            },
        )?;
        Ok(prepared)
    }

    fn forward_prepared_prefill(
        &mut self,
        prepared: runtime_input::PreparedPrefill,
        cache: &mut Cache,
        stream: &Stream,
        forward: impl FnOnce(&mut Self, ModelInput<'_>, &Stream) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let inputs = prepared.tokens();
        let inputs_embeds = prepared.embeddings();
        forward(
            self,
            ModelInput {
                inputs,
                inputs_embeds,
                mask: None,
                cache: Some(cache),
            },
            stream,
        )
    }

    pub(crate) fn prefill_typed_with_observer(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let prepared = self.prepare_typed_prefill(input, stream)?;
        let logits =
            self.forward_prepared_prefill(prepared, cache, stream, |model, input, stream| {
                model.forward_with_observer(input, stream, observer)
            })?;
        let logits = logits.try_index_device((.., -1, ..), stream)?;
        self.adjust_prefill_logits(logits, cache, stream)
    }
}

fn visual_embeddings_from_payload(
    visual: &mut Option<QwenVisionTransformer>,
    part: &runtime_input::InputPart<'_>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let grid_thw = part.metadata.qwen_grid_thw.ok_or_else(|| {
        Exception::custom(format!(
            "qwen3_5_moe {} input requires qwen_grid_thw metadata",
            part.modality.as_str()
        ))
    })?;
    match part.payload {
        runtime_input::InputPayload::Embeddings(embeddings) => Ok(embeddings.clone()),
        runtime_input::InputPayload::Tensor(tensor) => visual
            .as_mut()
            .ok_or_else(|| {
                Exception::custom(
                    "qwen3_5_moe visual tensor input requires vision_config and visual weights",
                )
            })?
            .forward(tensor, grid_thw, stream),
        runtime_input::InputPayload::TokenIds(_) => Err(Exception::custom(
            "qwen3_5_moe visual input does not accept token-id payloads",
        )),
    }
}

fn video_embedding_chunks(
    vision_args: Option<&VisionConfig>,
    part: &runtime_input::InputPart<'_>,
    embeddings: &Array,
    stream: &Stream,
) -> Result<Vec<Array>, Exception> {
    let grid_thw = part.metadata.qwen_grid_thw.ok_or_else(|| {
        Exception::custom("qwen3_5_moe video input requires qwen_grid_thw metadata")
    })?;
    let grid = grid_thw_from_array(grid_thw, stream)?;
    if grid.len() != 1 {
        return Err(Exception::custom(format!(
            "qwen3_5_moe each video input part requires one grid entry, got {}",
            grid.len()
        )));
    }
    let (grid_t, grid_h, grid_w) = grid[0];
    let merge = vision_args
        .map(|config| config.spatial_merge_size)
        .ok_or_else(|| Exception::custom("qwen3_5_moe video input requires vision_config"))?;
    let chunk_len = grid_h * grid_w / (merge * merge);
    let expected = grid_t * chunk_len;
    if embeddings.dim(1) != expected {
        return Err(Exception::custom(format!(
            "qwen3_5_moe video grid expects {expected} merged embeddings, got {}",
            embeddings.dim(1)
        )));
    }
    let mut chunks = Vec::with_capacity(grid_t as usize);
    for index in 0..grid_t {
        let start = index * chunk_len;
        chunks.push(embeddings.try_index_device((.., start..start + chunk_len, ..), stream)?);
    }
    Ok(chunks)
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prepared = self.prepare_typed_prefill(input, stream)?;
        self.forward_prepared_prefill(prepared, cache, stream, |model, input, stream| {
            model.forward_logits(input, true, stream)
        })
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
                inputs_embeds: None,
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

/// Qwen3.5 MoE token generation iterator.
pub type Generate<'a, S = crate::sampler::DefaultSampler> = common::Generate<'a, Model, Cache, S>;

#[cfg(test)]
mod tests {
    use super::{
        default_layer_type, get_qwen3_5_moe_model_args, load_qwen3_5_gguf, load_qwen3_5_moe_model,
        load_qwen3_5_moe_tokenizer, parse_fp8_expert_projection_key,
        qwen35_gguf_affine_quantization, qwen35_gguf_block_index, qwen35_is_offset_norm,
        qwen35_restore_v_head_order, qwen35_translate_gguf_weight_name,
        qwen3_5_moe_strict_load_config, reverse_permutation, vision_window_index,
        Fp8ExpertProjection, FullAttention, FullAttentionInput, LayerType, LinearAttention,
        LinearAttentionInput, Model, ModelArgs, SparseMoeBlock, VisionConfig,
    };
    #[cfg(feature = "image-processing")]
    use crate::processor::{load_processor, MediaInput, ProcessorInput, RgbImageView};
    use crate::{
        error::Error,
        inspection::ActivationRecorder,
        models::{common::CausalLm, input as runtime_input, Model as AnyModel, ModelCache},
        quantization::AffineQuantization,
        weights::{load_safetensors_strict, StrictLoadReport},
    };
    use safemlx::{
        module::{Module, ModuleParameters, Param},
        ops::indexing::{NewAxis, TryIndexOp},
        transforms::eval,
        Array, ExecutionContext,
    };
    use std::{
        collections::HashSet,
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
            intermediate_size: 0,
            moe_intermediate_size: 4,
            shared_expert_intermediate_size: 4,
            num_experts_per_tok: 2,
            num_experts: 4,
            norm_topk_prob: false,
            layer_types,
            rope_parameters: None,
            rope_scaling: None,
            quantization_config: None,
            quantized_weight_configs: None,
        }
    }

    fn tiny_vision_config(out_hidden_size: i32) -> VisionConfig {
        VisionConfig {
            depth: 1,
            hidden_size: 8,
            hidden_act: "silu".to_string(),
            intermediate_size: 4,
            num_heads: 2,
            num_position_embeddings: 16,
            in_channels: 3,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            window_size: 8,
            out_hidden_size,
            fullatt_block_indexes: vec![0],
        }
    }

    fn zero_model_parameters(model: &mut Model, stream: &safemlx::Stream) {
        for (_, param) in model.parameters_mut().flatten() {
            let shape = param.shape().to_vec();
            *param = Array::zeros::<f32>(&shape, stream).unwrap();
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
    fn affine_packed_experts_match_explicitly_dequantized_routing() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut args = tiny_args(vec![LayerType::FullAttention]);
        args.hidden_size = 32;
        args.num_attention_heads = 4;
        args.num_key_value_heads = 2;
        args.moe_intermediate_size = 32;
        args.shared_expert_intermediate_size = 32;
        let quantization = crate::quantization::AffineQuantization::new(32, 4).unwrap();

        let gate_up_values = (0..(4 * 64 * 32))
            .map(|index| ((index % 97) as f32 - 48.0) / 64.0)
            .collect::<Vec<_>>();
        let down_values = (0..(4 * 32 * 32))
            .map(|index| ((index % 71) as f32 - 35.0) / 48.0)
            .collect::<Vec<_>>();
        let gate_up = Array::from_slice(&gate_up_values, &[4, 64, 32]);
        let down = Array::from_slice(&down_values, &[4, 32, 32]);
        let gate_up_q =
            super::quantize_packed_expert_tensor(&gate_up, quantization, stream).unwrap();
        let down_q = super::quantize_packed_expert_tensor(&down, quantization, stream).unwrap();
        let gate_up_dequantized = safemlx::ops::dequantize(
            &gate_up_q.weight,
            &gate_up_q.scales,
            &gate_up_q.biases,
            32,
            4,
            stream,
        )
        .unwrap();
        let down_dequantized = safemlx::ops::dequantize(
            &down_q.weight,
            &down_q.scales,
            &down_q.biases,
            32,
            4,
            stream,
        )
        .unwrap();

        let mut dense = super::Experts::new(&args, 0, stream).unwrap();
        dense.gate_up_proj = Param::new(gate_up_dequantized);
        dense.down_proj = Param::new(down_dequantized);
        let mut affine = super::Experts::new_with_format(
            &args,
            0,
            super::QwenWeightFormat::Affine(quantization),
            stream,
        )
        .unwrap();
        affine.gate_up_proj = Param::new(gate_up_q.weight);
        affine.gate_up_proj_scales = Param::new(Some(gate_up_q.scales));
        affine.gate_up_proj_biases = Param::new(Some(gate_up_q.biases));
        affine.down_proj = Param::new(down_q.weight);
        affine.down_proj_scales = Param::new(Some(down_q.scales));
        affine.down_proj_biases = Param::new(Some(down_q.biases));

        let hidden_values = (0..(3 * 32))
            .map(|index| ((index % 29) as f32 - 14.0) / 32.0)
            .collect::<Vec<_>>();
        let hidden = Array::from_slice(&hidden_values, &[3, 32]);
        let expert_ids = Array::from_slice(&[0u32, 2, 1, 3, 2, 0], &[3, 2]);
        let route_weights = Array::from_slice(&[0.7f32, 0.3, 0.6, 0.4, 0.8, 0.2], &[3, 2]);
        let dense_output = dense
            .forward(&hidden, &expert_ids, &route_weights, stream)
            .unwrap();
        let affine_output = affine
            .forward(&hidden, &expert_ids, &route_weights, stream)
            .unwrap();

        assert!(dense_output
            .all_close(&affine_output, 1e-4, 1e-4, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    fn qwen35_gguf_names_and_affine_shapes_translate() {
        assert_eq!(
            qwen35_gguf_block_index("blk.40.nextn_norm.weight"),
            Some(40)
        );
        assert_eq!(qwen35_gguf_block_index("output.weight"), None);
        assert_eq!(
            qwen35_translate_gguf_weight_name("blk.7.attn_qkv.weight"),
            "model.layers.7.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(
            qwen35_translate_gguf_weight_name("blk.3.ffn_gate_exps.scales"),
            "model.layers.3.mlp.experts.gate_proj_scales"
        );
        assert_eq!(
            qwen35_translate_gguf_weight_name("blk.3.ffn_gate_inp_shexp.weight"),
            "model.layers.3.mlp.shared_expert_gate.weight"
        );
        assert_eq!(
            qwen35_translate_gguf_weight_name("blk.3.ffn_gate.weight"),
            "model.layers.3.dense_mlp.gate_proj.weight"
        );
        assert!(qwen35_is_offset_norm("blk.3.attn_q_norm.weight"));
        assert!(!qwen35_is_offset_norm("blk.3.ssm_norm.weight"));
        assert_eq!(
            qwen35_gguf_affine_quantization(&[256, 1024, 256], &[256, 1024, 64], "expert").unwrap(),
            AffineQuantization::new(32, 4).unwrap()
        );
        assert_eq!(
            qwen35_gguf_affine_quantization(&[256, 2048, 128], &[256, 2048, 16], "expert").unwrap(),
            AffineQuantization::new(32, 8).unwrap()
        );
    }

    #[test]
    fn qwen35_gguf_restores_grouped_value_head_order() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut args = tiny_args(vec![LayerType::LinearAttention]);
        args.linear_num_key_heads = 2;
        args.linear_num_value_heads = 4;
        args.linear_value_head_dim = 1;
        let llama_order = Array::from_slice(&[0i32, 2, 1, 3], &[4]);
        let restored = qwen35_restore_v_head_order(llama_order, 0, 1, &args, stream).unwrap();
        eval([&restored]).unwrap();
        assert_eq!(
            restored.evaluated().unwrap().as_slice::<i32>(),
            &[0, 1, 2, 3]
        );
    }

    #[test]
    #[ignore = "requires QWEN35_GGUF and Metal"]
    fn strict_loads_and_runs_real_dense_qwen35_gguf() {
        let _guard = mlx_runtime_test_guard();
        let gguf_file = std::path::PathBuf::from(
            std::env::var("QWEN35_GGUF").expect("set QWEN35_GGUF to a local checkpoint"),
        );
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut model = load_qwen3_5_gguf(&gguf_file, stream, weights_ctx.stream()).unwrap();

        assert_eq!(model.model_type(), "qwen3_5_text");
        assert_eq!(model.args.num_hidden_layers, 32);
        assert_eq!(model.args.intermediate_size, 12288);
        assert_eq!(model.args.num_experts, 0);

        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [runtime_input::InputPart::text_token_ids(&tokens)];
        let mut cache = model.new_cache();
        let logits = CausalLm::prefill_input_logits(
            &mut model,
            runtime_input::ModelInput::new(&parts),
            &mut cache,
            stream,
        )
        .unwrap();
        assert_eq!(logits.shape(), &[1, 248320]);
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
        let (args, image_token_id, video_token_id, vision_config) =
            get_qwen3_5_moe_model_args(&dir).unwrap();
        assert_eq!(args.model_type, "qwen3_5_moe_text");
        assert_eq!(args.layer_types.len(), 4);
        assert_eq!(args.layer_types[3], LayerType::FullAttention);
        assert_eq!(image_token_id, Some(248056));
        assert_eq!(video_token_id, Some(248057));
        assert!(vision_config.is_none());
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
        let (args, _, _, _) = get_qwen3_5_moe_model_args(&dir).unwrap();
        assert_eq!(args.model_type, "qwen3_5_moe_text");
        assert_eq!(args.layer_types, vec![LayerType::FullAttention]);
    }

    #[test]
    fn parses_top_level_vision_config() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe",
              "image_token_id": 248056,
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 128,
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128
              },
              "vision_config": {
                "depth": 1,
                "hidden_size": 8,
                "hidden_act": "silu",
                "intermediate_size": 4,
                "num_heads": 2,
                "num_position_embeddings": 16,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "window_size": 8,
                "out_hidden_size": 16,
                "fullatt_block_indexes": [0]
              }
            }"#,
        );
        let (_, image_token_id, _, vision_config) = get_qwen3_5_moe_model_args(&dir).unwrap();

        assert_eq!(image_token_id, Some(248056));
        assert_eq!(vision_config, Some(tiny_vision_config(16)));
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
        let (args, _, _, _) = get_qwen3_5_moe_model_args(&dir).unwrap();
        let quantization_config = args.quantization_config.unwrap();
        assert_eq!(quantization_config.quant_method, "fp8");
        assert_eq!(quantization_config.fmt, "e4m3");
        assert_eq!(quantization_config.activation_scheme, "dynamic");
        assert_eq!(
            quantization_config.weight_block_size.as_deref(),
            Some(&[128, 128][..])
        );
        quantization_config.validate_supported().unwrap();

        let cpu = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let error = super::load_qwen3_5_moe_model_quantized(
            &dir,
            crate::quantization::AffineQuantization::default(),
            cpu.stream(),
            cpu.stream(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Quantization(_)));
        assert!(error.to_string().contains("requires a dense checkpoint"));
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
    fn strict_load_rewrites_public_vision_keys() {
        let config = qwen3_5_moe_strict_load_config(true);

        assert!(config
            .candidates("model.visual.merger.linear_fc1.weight")
            .contains(&"visual.merger.linear_fc1.weight".to_string()));
        assert!(config
            .candidates("model.visual.blocks.0.mlp.linear_fc2.bias")
            .contains(&"visual.blocks.0.mlp.linear_fc2.bias".to_string()));
    }

    #[test]
    fn vision_window_index_preserves_patch_group_permutation() {
        let (window_index, chunk_lengths) = vision_window_index(&[(1, 6, 6)], 2, 8, 2).unwrap();

        assert_eq!(window_index, vec![0, 1, 3, 4, 2, 5, 6, 7, 8]);
        assert_eq!(chunk_lengths, vec![16, 8, 8, 4]);
        assert_eq!(
            reverse_permutation(&window_index),
            vec![0, 1, 4, 2, 3, 5, 6, 7, 8]
        );
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
    fn linear_attention_observer_reports_internal_hooks() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let mut attn = LinearAttention::new(&args, stream).unwrap();
        let x = Array::zeros::<f32>(&[1, 2, args.hidden_size], stream).unwrap();
        let mut recorder = ActivationRecorder::new();

        let out = attn
            .forward_with_observer(
                LinearAttentionInput { x: &x, cache: None },
                stream,
                "model.layers.0.linear_attn",
                &mut recorder,
            )
            .unwrap();

        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
        let names = recorder
            .activations()
            .iter()
            .map(|activation| activation.name.as_str())
            .collect::<Vec<_>>();
        for expected in [
            "model.layers.0.linear_attn.in_proj_qkv",
            "model.layers.0.linear_attn.causal_conv",
            "model.layers.0.linear_attn.query_l2norm",
            "model.layers.0.linear_attn.recurrent_core",
            "model.layers.0.linear_attn.gated_norm",
            "model.layers.0.linear_attn.out_proj",
        ] {
            assert!(names.contains(&expected), "{names:?}");
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn sparse_moe_forward_shape_smoke() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let mut moe = SparseMoeBlock::new(&args, 0, stream).unwrap();
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
    fn sparse_moe_observer_reports_routed_expert_internals() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let mut moe = SparseMoeBlock::new(&args, 0, stream).unwrap();
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
        let mut recorder = ActivationRecorder::new();

        let out = moe
            .forward_with_observer(&x, stream, "model.layers.0.moe", &mut recorder)
            .unwrap();

        assert_eq!(out.shape(), &[1, 2, args.hidden_size]);
        let names = recorder
            .activations()
            .iter()
            .map(|activation| activation.name.as_str())
            .collect::<Vec<_>>();
        for expected in [
            "model.layers.0.moe.gate.router_logits",
            "model.layers.0.moe.gate.top_k_experts",
            "model.layers.0.moe.experts.gate_proj",
            "model.layers.0.moe.experts.up_proj",
            "model.layers.0.moe.experts.down_proj_input",
            "model.layers.0.moe.experts.route_output",
            "model.layers.0.moe.experts.weighted_route_output",
            "model.layers.0.moe.combined_flat",
        ] {
            assert!(names.contains(&expected), "{names:?}");
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn parameter_tree_matches_public_checkpoint_key_patterns() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention, LayerType::FullAttention]);
        let model = Model::new(args, Some(248056), Some(248057), None, stream).unwrap();
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
    fn parameter_tree_includes_visual_checkpoint_key_patterns() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let model = Model::new(
            args,
            Some(248056),
            Some(248057),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        let params = model.parameters().flatten();

        for key in [
            "visual.pos_embed.weight",
            "visual.patch_embed.proj.weight",
            "visual.patch_embed.proj.bias",
            "visual.blocks.0.norm1.weight",
            "visual.blocks.0.norm1.bias",
            "visual.blocks.0.attn.qkv.weight",
            "visual.blocks.0.attn.qkv.bias",
            "visual.blocks.0.attn.proj.weight",
            "visual.blocks.0.mlp.linear_fc1.weight",
            "visual.blocks.0.mlp.linear_fc2.weight",
            "visual.merger.norm.weight",
            "visual.merger.norm.bias",
            "visual.merger.linear_fc1.weight",
            "visual.merger.linear_fc2.weight",
        ] {
            assert!(params.contains_key(key), "missing parameter key {key}");
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn typed_image_tensor_prefill_runs_through_visual_encoder() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut model = Model::new(
            args,
            Some(42),
            Some(43),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        zero_model_parameters(&mut model, stream);

        let text = runtime_input::token_ids_array(&[7, 8], stream).unwrap();
        let grid_thw = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        let pixel_values = Array::zeros::<f32>(&[8, 12], stream).unwrap();
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixel_values,
                runtime_input::InputMetadata::qwen_grid_thw(&grid_thw),
            ),
        ];
        let input = runtime_input::ModelInput::new(&parts);
        let mut cache = model.new_cache();

        let logits = model
            .prefill_input_logits(input, &mut cache, stream)
            .unwrap();

        assert_eq!(logits.shape(), &[1, 128]);
        assert_eq!(cache.offset(), 5);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn observer_image_tensor_prefill_uses_typed_preparation() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut qwen = Model::new(
            args,
            Some(42),
            Some(43),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        zero_model_parameters(&mut qwen, stream);
        let mut model = AnyModel::Qwen35Moe(qwen);

        let before = runtime_input::token_ids_array(&[7], stream).unwrap();
        let after = runtime_input::token_ids_array(&[8], stream).unwrap();
        let grid_thw = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        let pixel_values = Array::zeros::<f32>(&[8, 12], stream).unwrap();
        let parts = [
            runtime_input::InputPart::text_token_ids(&before),
            runtime_input::InputPart::image_tensor(
                &pixel_values,
                runtime_input::InputMetadata::qwen_grid_thw(&grid_thw),
            ),
            runtime_input::InputPart::text_token_ids(&after),
        ];
        let mut cache = model.new_cache();
        let mut recorder = ActivationRecorder::new();

        let logits = model
            .prefill_input_with_observer(
                runtime_input::ModelInput::new(&parts),
                &mut cache,
                stream,
                &mut recorder,
            )
            .unwrap();

        assert_eq!(logits.shape(), &[1, 128]);
        let ModelCache::Qwen35Moe(cache) = cache else {
            panic!("expected qwen cache");
        };
        assert_eq!(cache.offset(), 5);
        assert!(recorder
            .activations()
            .iter()
            .any(|activation| activation.name == "lm_head.logits"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn normal_and_observer_multimodal_prefill_share_cache_semantics() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut normal_qwen = Model::new(
            args.clone(),
            Some(42),
            Some(43),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        let mut observed_qwen = Model::new(
            args,
            Some(42),
            Some(43),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        zero_model_parameters(&mut normal_qwen, stream);
        zero_model_parameters(&mut observed_qwen, stream);
        let mut normal = AnyModel::Qwen35Moe(normal_qwen);
        let mut observed = AnyModel::Qwen35Moe(observed_qwen);

        let before = runtime_input::token_ids_array(&[7], stream).unwrap();
        let after = runtime_input::token_ids_array(&[8], stream).unwrap();
        let image_embeddings = Array::zeros::<f32>(&[1, 3, 16], stream).unwrap();
        let grid_thw = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&before),
            runtime_input::InputPart {
                modality: runtime_input::Modality::Image,
                payload: runtime_input::InputPayload::Embeddings(&image_embeddings),
                metadata: runtime_input::InputMetadata::qwen_grid_thw(&grid_thw),
            },
            runtime_input::InputPart::text_token_ids(&after),
        ];
        let input = runtime_input::ModelInput::new(&parts);
        let mut normal_cache = normal.new_cache();
        let mut observed_cache = observed.new_cache();
        let mut recorder = ActivationRecorder::new();

        let normal_logits = normal
            .prefill_input_with_cache(input, &mut normal_cache, stream)
            .unwrap();
        let observed_logits = observed
            .prefill_input_with_observer(input, &mut observed_cache, stream, &mut recorder)
            .unwrap();

        assert_eq!(normal_logits.shape(), observed_logits.shape());
        let ModelCache::Qwen35Moe(normal_cache) = normal_cache else {
            panic!("expected qwen cache");
        };
        let ModelCache::Qwen35Moe(observed_cache) = observed_cache else {
            panic!("expected qwen cache");
        };
        assert_eq!(normal_cache.offset(), observed_cache.offset());
        assert_eq!(observed_cache.offset(), 5);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn text_only_observer_prefill_still_works() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut qwen = Model::new(args, None, None, None, stream).unwrap();
        zero_model_parameters(&mut qwen, stream);
        let mut model = AnyModel::Qwen35Moe(qwen);

        let text = runtime_input::token_ids_array(&[7, 8, 9], stream).unwrap();
        let parts = [runtime_input::InputPart::text_token_ids(&text)];
        let mut cache = model.new_cache();
        let mut recorder = ActivationRecorder::new();

        let logits = model
            .prefill_input_with_observer(
                runtime_input::ModelInput::new(&parts),
                &mut cache,
                stream,
                &mut recorder,
            )
            .unwrap();

        assert_eq!(logits.shape(), &[1, 128]);
        let ModelCache::Qwen35Moe(cache) = cache else {
            panic!("expected qwen cache");
        };
        assert_eq!(cache.offset(), 3);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn text_only_qwen_rejects_observer_image_input_clearly() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut qwen = Model::new(args, None, None, None, stream).unwrap();
        zero_model_parameters(&mut qwen, stream);
        let mut model = AnyModel::Qwen35Moe(qwen);

        let image_embeddings = Array::zeros::<f32>(&[1, 3, 16], stream).unwrap();
        let grid_thw = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        let parts = [runtime_input::InputPart {
            modality: runtime_input::Modality::Image,
            payload: runtime_input::InputPayload::Embeddings(&image_embeddings),
            metadata: runtime_input::InputMetadata::qwen_grid_thw(&grid_thw),
        }];
        let mut cache = model.new_cache();
        let mut recorder = ActivationRecorder::new();

        let error = model
            .prefill_input_with_observer(
                runtime_input::ModelInput::new(&parts),
                &mut cache,
                stream,
                &mut recorder,
            )
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("qwen3_5_moe typed input does not support image input yet"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn typed_video_tensor_prefill_splits_temporal_embeddings_and_decodes() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut model = Model::new(
            args,
            Some(42),
            Some(43),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        zero_model_parameters(&mut model, stream);

        let text = runtime_input::token_ids_array(&[7, 8, 9], stream).unwrap();
        let grid_thw = Array::from_slice(&[2i32, 2, 4], &[1, 3]);
        let pixel_values = Array::zeros::<f32>(&[16, 12], stream).unwrap();
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart {
                modality: runtime_input::Modality::Video,
                payload: runtime_input::InputPayload::Tensor(&pixel_values),
                metadata: runtime_input::InputMetadata::qwen_grid_thw(&grid_thw),
            },
        ];
        let mut cache = model.new_cache();
        let logits = model
            .prefill_input_logits(runtime_input::ModelInput::new(&parts), &mut cache, stream)
            .unwrap();

        assert_eq!(logits.shape(), &[1, 128]);
        assert_eq!(cache.offset(), 7);
        let next = runtime_input::token_ids_array(&[10], stream).unwrap();
        let decode = model.decode_logits(&next, &mut cache, stream).unwrap();
        assert_eq!(decode.shape(), &[1, 128]);
        assert_eq!(cache.offset(), 8);
    }

    #[test]
    #[cfg(feature = "image-processing")]
    #[ignore = "requires MLX runtime execution"]
    fn raw_rgb_processor_output_prefills_through_visual_encoder() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut model = Model::new(
            args,
            Some(42),
            Some(43),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        zero_model_parameters(&mut model, stream);

        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe",
              "image_token_id": 42,
              "vision_start_token_id": 44,
              "vision_end_token_id": 45,
              "text_config": { "model_type": "qwen3_5_moe_text" }
            }"#,
        );
        fs::write(
            dir.join("preprocessor_config.json"),
            r#"{
              "size": { "shortest_edge": 32, "longest_edge": 32 },
              "patch_size": 2,
              "temporal_patch_size": 1,
              "merge_size": 2,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5]
            }"#,
        )
        .unwrap();
        let processor = load_processor(&dir).unwrap().unwrap();
        let pixels = vec![128u8; 8 * 4 * 3];
        let image = RgbImageView::packed(&pixels, 8, 4).unwrap();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::image_rgb8(image)),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_text| Ok(Vec::new()),
            )
            .unwrap();
        let mut cache = model.new_cache();

        let logits = prepared
            .with_model_input(|input| model.prefill_input_logits(input, &mut cache, stream))
            .unwrap();

        assert_eq!(logits.shape(), &[1, 128]);
        assert_eq!(cache.offset(), 4);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[cfg(feature = "image-processing")]
    #[ignore = "requires MLX runtime execution"]
    fn raw_rgb_video_processor_output_prefills_and_decodes() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::FullAttention]);
        let mut model = Model::new(
            args,
            Some(42),
            Some(43),
            Some(tiny_vision_config(16)),
            stream,
        )
        .unwrap();
        zero_model_parameters(&mut model, stream);

        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe",
              "image_token_id": 42,
              "video_token_id": 43,
              "vision_start_token_id": 44,
              "vision_end_token_id": 45,
              "text_config": { "model_type": "qwen3_5_moe_text" }
            }"#,
        );
        fs::write(
            dir.join("video_preprocessor_config.json"),
            r#"{
              "size": { "shortest_edge": 64, "longest_edge": 64 },
              "patch_size": 2,
              "temporal_patch_size": 1,
              "merge_size": 2,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "fps": 2.0,
              "min_frames": 2,
              "max_frames": 2
            }"#,
        )
        .unwrap();
        let processor = load_processor(&dir).unwrap().unwrap();
        let frame_pixels = [vec![64u8; 8 * 4 * 3], vec![192u8; 8 * 4 * 3]];
        let frames = frame_pixels
            .iter()
            .map(|pixels| RgbImageView::packed(pixels, 8, 4).unwrap())
            .collect::<Vec<_>>();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::video_rgb8(&frames, Some(2.0))),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_timestamp| Ok(vec![50]),
            )
            .unwrap();
        let mut cache = model.new_cache();

        let logits = prepared
            .with_model_input(|input| model.prefill_input_logits(input, &mut cache, stream))
            .unwrap();

        assert_eq!(logits.shape(), &[1, 128]);
        assert_eq!(cache.offset(), 12);
        let next = runtime_input::token_ids_array(&[10], stream).unwrap();
        model.decode_logits(&next, &mut cache, stream).unwrap();
        assert_eq!(cache.offset(), 13);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[cfg(feature = "image-processing")]
    #[ignore = "requires local Qwen3.5-MoE model files and MLX runtime execution"]
    fn local_qwen35_processor_config_produces_checkpoint_native_tensors() {
        let _guard = mlx_runtime_test_guard();
        let model_dir = cached_test_model_dir();
        let processor = load_processor(model_dir).unwrap().unwrap();
        let pixels = vec![128u8; 480 * 320 * 3];
        let image = RgbImageView::packed(&pixels, 480, 320).unwrap();
        let prepared = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::image_rgb8(image)),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_text| Ok(Vec::new()),
            )
            .unwrap();
        let parts = prepared.input_parts();
        let image_part = parts
            .iter()
            .find(|part| part.modality == runtime_input::Modality::Image)
            .unwrap();
        let runtime_input::InputPayload::Tensor(patches) = image_part.payload else {
            panic!("expected image tensor payload");
        };

        assert_eq!(patches.shape(), &[600, 1536]);
        assert_eq!(
            image_part
                .metadata
                .qwen_grid_thw
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 20, 30]
        );

        let video_pixels = vec![96u8; 480 * 320 * 3];
        let frame = RgbImageView::packed(&video_pixels, 480, 320).unwrap();
        let frames = [frame; 4];
        let prepared_video = processor
            .prepare_input(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::video_rgb8(&frames, Some(2.0))),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_timestamp| Ok(vec![50]),
            )
            .unwrap();
        let video_parts = prepared_video.input_parts();
        let video_part = video_parts
            .iter()
            .find(|part| part.modality == runtime_input::Modality::Video)
            .unwrap();
        let runtime_input::InputPayload::Tensor(video_patches) = video_part.payload else {
            panic!("expected video tensor payload");
        };
        assert_eq!(video_patches.shape(), &[600, 1536]);
        assert_eq!(
            video_part
                .metadata
                .qwen_grid_thw
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 20, 30]
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.5-MoE model files"]
    fn local_qwen35_visual_index_keys_match_parameter_tree() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let model_dir = cached_test_model_dir();
        let (args, image_token_id, video_token_id, vision_config) =
            get_qwen3_5_moe_model_args(&model_dir).unwrap();
        let model =
            Model::new(args, image_token_id, video_token_id, vision_config, stream).unwrap();
        let params = model.parameters().flatten();
        let strict_config = qwen3_5_moe_strict_load_config(true);
        let index_file =
            std::fs::File::open(model_dir.join("model.safetensors.index.json")).unwrap();
        let index: serde_json::Value = serde_json::from_reader(index_file).unwrap();
        let weight_map = index
            .get("weight_map")
            .and_then(|value| value.as_object())
            .expect("safetensors index weight_map");

        let mut loaded_visual_params = HashSet::new();
        let unmatched = weight_map
            .keys()
            .filter(|key| key.contains(".visual."))
            .filter_map(|key| {
                let matched = strict_config
                    .candidates(key)
                    .into_iter()
                    .find(|candidate| params.contains_key(candidate.as_str()));
                if let Some(matched) = matched {
                    loaded_visual_params.insert(matched);
                    None
                } else {
                    Some(key.to_string())
                }
            })
            .collect::<Vec<_>>();
        assert!(unmatched.is_empty(), "unmatched visual keys: {unmatched:?}");

        let missing = params
            .keys()
            .filter(|key| key.starts_with("visual."))
            .filter(|key| !loaded_visual_params.contains(key.as_ref()))
            .map(|key| key.to_string())
            .collect::<Vec<_>>();
        assert!(missing.is_empty(), "missing visual keys: {missing:?}");
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_allows_unused_non_text_prefixes() {
        let _guard = mlx_runtime_test_guard();
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_args(vec![LayerType::LinearAttention]);
        let source = Model::new(args.clone(), None, None, None, stream).unwrap();
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

        let mut target = Model::new(args, None, None, None, stream).unwrap();
        let config = qwen3_5_moe_strict_load_config(false);
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
        let source = Model::new(args.clone(), None, None, None, stream).unwrap();
        let dir = temp_model_dir("{}");
        let weights_path = dir.join("model.safetensors");
        save_model_parameters(
            &weights_path,
            &source,
            |key| key != "lm_head.weight",
            Vec::new(),
        );

        let mut target = Model::new(args, None, None, None, stream).unwrap();
        let config = qwen3_5_moe_strict_load_config(false);
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
        let source = Model::new(args.clone(), None, None, None, stream).unwrap();
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

        let mut target = Model::new(args, None, None, None, stream).unwrap();
        let config = qwen3_5_moe_strict_load_config(false);
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
            let input_parts = [crate::models::input::InputPart::text_token_ids(
                &prompt_tokens,
            )];
            let input = crate::models::input::ModelInput::new(&input_parts);
            let generate = super::Generate::new(&mut model, &mut cache, 0.0, input, None, stream);
            for token in generate.take(expected_tokens.len()) {
                let token = token.unwrap();
                eval([&token]).unwrap();
                tokens.push(token.item::<u32>(&stream));
            }
            assert_eq!(tokens, expected_tokens, "prompt: {prompt}");
        }
    }
}
