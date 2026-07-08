//! Nemotron-H configuration parsing, runtime blocks, and strict checkpoint loading.

use std::path::Path;

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        broadcast_to, clip, concatenate_axis, conv1d, exp,
        indexing::{NewAxis, TryIndexOp},
        sigmoid, sum_axis, zeros,
    },
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    models::common::{
        self, apply_rope_and_update_cache, batch_seq, finish_attention, project_logits_dense,
        relu2, reshape_attention_projection, CausalLm, PackedRelu2Experts, TopKRouterScoreFunction,
    },
    utils::{create_attention_mask, rope::initialize_rope, AttentionMask},
    weights::{
        load_safetensors_dir_strict_with_split_relu2_experts, transform_split_relu2_experts,
        StrictLoadConfig, StrictLoadReport,
    },
};

/// Layer block kind encoded by `hybrid_override_pattern`.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayerBlockType {
    /// Mamba2 state-space layer.
    Mamba,
    /// Grouped-query self-attention layer.
    Attention,
    /// Dense feed-forward MLP layer.
    Mlp,
    /// Sparse mixture-of-experts feed-forward layer.
    Moe,
}

impl LayerBlockType {
    fn from_pattern_char(ch: char) -> Result<Self, Error> {
        match ch {
            'M' => Ok(Self::Mamba),
            '*' => Ok(Self::Attention),
            '-' => Ok(Self::Mlp),
            'E' => Ok(Self::Moe),
            other => Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H hybrid_override_pattern contains unsupported layer marker '{other}'"
            ))),
        }
    }
}

/// Deserialized Nemotron-H configuration fields used by this loader.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    /// Model type from the configuration.
    pub model_type: String,
    /// Token vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Whether logits use tied input embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Transformer hidden size.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    /// Dense MLP intermediate size.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    /// Number of hybrid layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    /// Per-layer block pattern: `M` Mamba2, `*` attention, `-` MLP, `E` MoE.
    #[serde(default = "default_hybrid_override_pattern")]
    pub hybrid_override_pattern: String,
    /// Number of query attention heads for GQA layers.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    /// Per-head attention dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Number of key/value heads for GQA layers.
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    /// RoPE base frequency for attention layers.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Maximum configured sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// Whether attention projections include bias terms.
    #[serde(default)]
    pub attention_bias: bool,
    /// Whether MLP projections include bias terms.
    #[serde(default)]
    pub mlp_bias: bool,
    /// Global bias flag from Nemotron-H configs.
    #[serde(default)]
    pub use_bias: bool,
    /// LayerNorm epsilon used by the reference implementation.
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,
    /// RMSNorm epsilon alias used by released Nemotron-H checkpoints.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Whether residual paths are accumulated in float32.
    #[serde(default)]
    pub residual_in_fp32: bool,
    /// Number of prompt logits kept during generation.
    #[serde(default = "default_num_logits_to_keep")]
    pub num_logits_to_keep: i32,
    /// Optional sliding-window attention size.
    #[serde(default)]
    pub sliding_window: Option<i32>,
    /// Mamba2 state dimension.
    #[serde(default = "default_ssm_state_size")]
    pub ssm_state_size: i32,
    /// Number of Mamba heads.
    #[serde(default = "default_mamba_num_heads")]
    pub mamba_num_heads: i32,
    /// Number of Mamba groups.
    #[serde(default = "default_n_groups")]
    pub n_groups: i32,
    /// Mamba head dimension.
    #[serde(default = "default_mamba_head_dim")]
    pub mamba_head_dim: i32,
    /// Mamba causal convolution kernel width.
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: i32,
    /// Mamba expansion factor.
    #[serde(default = "default_expand")]
    pub expand: i32,
    /// Mamba activation name.
    #[serde(default = "default_mamba_hidden_act")]
    pub mamba_hidden_act: String,
    /// Minimum Mamba time step.
    #[serde(default = "default_time_step_min")]
    pub time_step_min: f32,
    /// Maximum Mamba time step.
    #[serde(default = "default_time_step_max")]
    pub time_step_max: f32,
    /// Floor for Mamba time-step initialization.
    #[serde(default = "default_time_step_floor")]
    pub time_step_floor: f32,
    /// Whether Mamba convolution includes a bias.
    #[serde(default = "default_true")]
    pub use_conv_bias: bool,
    /// Whether Mamba projections include bias terms.
    #[serde(default)]
    pub mamba_proj_bias: bool,
    /// Mamba scan chunk size.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: i32,
    /// Whether prenorm residuals are rescaled.
    #[serde(default = "default_true")]
    pub rescale_prenorm_residual: bool,
    /// Dense MLP activation name.
    #[serde(default = "default_mlp_hidden_act")]
    pub mlp_hidden_act: String,
    /// Number of routed MoE experts.
    #[serde(default = "default_n_routed_experts")]
    pub n_routed_experts: i32,
    /// Number of shared MoE experts.
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: i32,
    /// Routed-expert intermediate size.
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: i32,
    /// Shared-expert intermediate size.
    #[serde(default = "default_moe_shared_expert_intermediate_size")]
    pub moe_shared_expert_intermediate_size: i32,
    /// Number of experts selected per token.
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    /// Routed expert scaling factor.
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    /// Router group count.
    #[serde(default = "default_n_group")]
    pub n_group: i32,
    /// Number of router groups considered by top-k routing.
    #[serde(default = "default_topk_group")]
    pub topk_group: i32,
    /// Whether selected top-k probabilities are normalized.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    /// Torch dtype string from the Hugging Face config.
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

impl ModelArgs {
    /// Returns the parsed layer kinds from `hybrid_override_pattern`.
    pub fn layer_block_types(&self) -> Result<Vec<LayerBlockType>, Error> {
        self.hybrid_override_pattern
            .chars()
            .map(LayerBlockType::from_pattern_char)
            .collect()
    }

    fn layer_block_type(&self, index: usize) -> Result<LayerBlockType, Error> {
        self.hybrid_override_pattern
            .chars()
            .nth(index)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Nemotron-H layer index {index} is outside hybrid_override_pattern"
                ))
            })
            .and_then(LayerBlockType::from_pattern_char)
    }
}

fn default_true() -> bool {
    true
}

fn default_vocab_size() -> i32 {
    131_072
}

fn default_hidden_size() -> i32 {
    4096
}

fn default_intermediate_size() -> i32 {
    21_504
}

fn default_num_hidden_layers() -> i32 {
    52
}

fn default_hybrid_override_pattern() -> String {
    "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-".to_string()
}

fn default_num_attention_heads() -> i32 {
    32
}

fn default_head_dim() -> i32 {
    128
}

fn default_num_key_value_heads() -> i32 {
    8
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_max_position_embeddings() -> i32 {
    4096
}

fn default_layer_norm_epsilon() -> f32 {
    1e-5
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_num_logits_to_keep() -> i32 {
    1
}

fn default_ssm_state_size() -> i32 {
    128
}

fn default_mamba_num_heads() -> i32 {
    128
}

fn default_n_groups() -> i32 {
    8
}

fn default_mamba_head_dim() -> i32 {
    64
}

fn default_conv_kernel() -> i32 {
    4
}

fn default_expand() -> i32 {
    2
}

fn default_mamba_hidden_act() -> String {
    "silu".to_string()
}

fn default_time_step_min() -> f32 {
    0.001
}

fn default_time_step_max() -> f32 {
    0.1
}

fn default_time_step_floor() -> f32 {
    0.0001
}

fn default_chunk_size() -> i32 {
    128
}

fn default_mlp_hidden_act() -> String {
    "relu2".to_string()
}

fn default_n_routed_experts() -> i32 {
    8
}

fn default_n_shared_experts() -> i32 {
    1
}

fn default_moe_intermediate_size() -> i32 {
    7688
}

fn default_moe_shared_expert_intermediate_size() -> i32 {
    7688
}

fn default_num_experts_per_tok() -> i32 {
    2
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

fn default_n_group() -> i32 {
    1
}

fn default_topk_group() -> i32 {
    1
}

fn ensure_positive(name: &str, value: i32) -> Result<(), Error> {
    if value > 0 {
        Ok(())
    } else {
        Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H {name} must be positive, got {value}"
        )))
    }
}

fn silu(x: Array, stream: &Stream) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x, stream)?, stream)
}

#[derive(Debug, Clone, ModuleParameters)]
/// Gated RMSNorm used by Nemotron-H Mamba2 layers.
pub struct MambaRmsNormGated {
    #[param]
    /// Learned RMSNorm scale.
    pub weight: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
    /// Number of feature groups.
    pub n_groups: i32,
    /// Features per group.
    pub group_size: i32,
}

impl MambaRmsNormGated {
    /// Creates an unloaded gated RMSNorm.
    pub fn new(
        intermediate_size: i32,
        n_groups: i32,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[intermediate_size], Dtype::Float32, stream)?,
            eps,
            n_groups,
            group_size: intermediate_size / n_groups,
        })
    }

    /// Applies grouped RMS normalization followed by SiLU gate modulation.
    pub fn forward(&self, x: &Array, gate: &Array, stream: &Stream) -> Result<Array, Exception> {
        let original_shape = x.shape().to_vec();
        let grouped = x.reshape(&[-1, self.n_groups, self.group_size], stream)?;
        let variance = safemlx::ops::mean_axis(&grouped.square(stream)?, -1, true, stream)?;
        let normalized = grouped
            .multiply(
                safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
                stream,
            )?
            .reshape(&original_shape, stream)?;
        normalized
            .multiply(&*self.weight, stream)?
            .multiply(silu(gate.clone(), stream)?, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Dense Nemotron-H feed-forward block using `relu2(up_proj(x))`.
pub struct Mlp {
    #[param]
    /// Up projection.
    pub up_proj: nn::Linear,
    #[param]
    /// Down projection.
    pub down_proj: nn::Linear,
}

impl Mlp {
    /// Creates an unloaded MLP.
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            up_proj: nn::Linear::unloaded(
                hidden_size,
                intermediate_size,
                bias,
                Dtype::Float32,
                stream,
            )?,
            down_proj: nn::Linear::unloaded(
                intermediate_size,
                hidden_size,
                bias,
                Dtype::Float32,
                stream,
            )?,
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let hidden = relu2(self.up_proj.forward(x, stream)?, stream)?;
        self.down_proj.forward(&hidden, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

/// Sigmoid top-k router used by Nemotron-H MoE layers.
pub type TopKRouter = common::TopKRouter;

/// Packed routed expert bank for Nemotron-H MoE layers.
pub type Experts = PackedRelu2Experts;

#[derive(Debug, Clone, ModuleParameters)]
/// Sparse MoE block with routed experts plus one shared dense expert.
pub struct SparseMoeBlock {
    #[param]
    /// Top-k router.
    pub gate: TopKRouter,
    #[param]
    /// Routed expert bank.
    pub experts: Experts,
    #[param]
    /// Shared dense expert.
    pub shared_experts: Mlp,
}

impl SparseMoeBlock {
    /// Creates an unloaded sparse MoE block.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            gate: TopKRouter::new(
                common::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.n_routed_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Sigmoid,
                    norm_topk_prob: args.norm_topk_prob,
                    normalization_epsilon: 1e-20,
                    routed_scaling_factor: args.routed_scaling_factor,
                    n_group: args.n_group,
                    topk_group: args.topk_group,
                    score_correction_bias: true,
                },
                stream,
            )?,
            experts: Experts::new(
                args.n_routed_experts,
                args.hidden_size,
                args.moe_intermediate_size,
                stream,
            )?,
            shared_experts: Mlp::new(
                args.hidden_size,
                args.moe_shared_expert_intermediate_size,
                args.mlp_bias,
                stream,
            )?,
        })
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let (top_k_index, top_k_weights) = self.gate.forward(&flat, stream)?;
        let routed = self
            .experts
            .forward(&flat, &top_k_index, &top_k_weights, stream)?;
        let shared = self
            .shared_experts
            .forward(hidden_states, stream)?
            .reshape(&[-1, shape[2]], stream)?;
        routed.add(shared, stream)?.reshape(shape, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
        self.experts.training_mode(mode);
        self.shared_experts.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Grouped-query self-attention layer used by `*` Nemotron-H blocks.
pub struct Attention {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads.
    pub n_kv_heads: i32,
    /// Attention scale.
    pub scale: f32,
    #[param]
    /// Query projection.
    pub q_proj: nn::Linear,
    #[param]
    /// Key projection.
    pub k_proj: nn::Linear,
    #[param]
    /// Value projection.
    pub v_proj: nn::Linear,
    #[param]
    /// Output projection.
    pub o_proj: nn::Linear,
    #[param]
    /// Rotary position embedding module.
    pub rope: crate::utils::rope::RopeVariant,
}

impl Attention {
    /// Creates an unloaded attention layer.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            scale: (args.head_dim as f32).sqrt().recip(),
            q_proj: nn::Linear::unloaded(
                args.hidden_size,
                args.num_attention_heads * args.head_dim,
                args.attention_bias,
                Dtype::Float32,
                stream,
            )?,
            k_proj: nn::Linear::unloaded(
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
                args.attention_bias,
                Dtype::Float32,
                stream,
            )?,
            v_proj: nn::Linear::unloaded(
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
                args.attention_bias,
                Dtype::Float32,
                stream,
            )?,
            o_proj: nn::Linear::unloaded(
                args.num_attention_heads * args.head_dim,
                args.hidden_size,
                args.attention_bias,
                Dtype::Float32,
                stream,
            )?,
            rope: initialize_rope(
                args.head_dim,
                args.rope_theta,
                false,
                &None,
                args.max_position_embeddings,
                stream,
            )?,
        })
    }
}

/// Input for a Nemotron-H attention layer.
pub struct AttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional key/value cache.
    pub cache: Option<&'a mut ConcatKeyValueCache>,
}

impl Module<AttentionInput<'_>> for Attention {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: AttentionInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;
        let (batch, seq_len) = batch_seq(x);
        let queries = reshape_attention_projection(
            self.q_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_heads,
            stream,
        )?;
        let keys = reshape_attention_projection(
            self.k_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let values = reshape_attention_projection(
            self.v_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        let out = finish_attention(
            queries,
            keys,
            values,
            None::<&mut ConcatKeyValueCache>,
            self.scale,
            mask,
            batch,
            seq_len,
            stream,
        )?;
        self.o_proj.forward(&out, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <crate::utils::rope::RopeVariant as Module<nn::RopeInput>>::training_mode(
            &mut self.rope,
            mode,
        );
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Depthwise convolution parameters for Nemotron-H Mamba2.
pub struct DepthwiseConv1d {
    #[param]
    /// Convolution weights shaped `[channels, 1, kernel]`.
    pub weight: Param<Array>,
    #[param]
    /// Optional convolution bias.
    pub bias: Param<Option<Array>>,
}

impl DepthwiseConv1d {
    /// Creates unloaded depthwise convolution parameters.
    pub fn new(
        channels: i32,
        kernel_size: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[channels, 1, kernel_size], Dtype::Float32, stream)?,
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[channels], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
        })
    }
}

#[derive(Debug, Clone, Default)]
/// Cache state for a Nemotron-H Mamba2 block.
pub struct Mamba2Cache {
    /// Cached causal-convolution state shaped `[batch, kernel - 1, conv_dim]`.
    pub conv_state: Option<Array>,
    /// Cached SSM state shaped `[batch, heads, head_dim, state]`.
    pub ssm_state: Option<Array>,
    /// Number of consumed tokens.
    pub offset: i32,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
/// Mamba2 mixer used by `M` Nemotron-H blocks.
pub struct Mamba2Mixer {
    /// Number of Mamba heads.
    pub num_heads: i32,
    /// Mamba head dimension.
    pub head_dim: i32,
    /// SSM state size.
    pub ssm_state_size: i32,
    /// Number of B/C groups.
    pub n_groups: i32,
    /// Intermediate Mamba width.
    pub intermediate_size: i32,
    /// Convolution input dimension.
    pub conv_dim: i32,
    /// Causal convolution kernel size.
    pub conv_kernel_size: i32,
    /// Unused MLP branch size from the reference split.
    pub d_mlp: i32,
    /// Minimum timestep clamp.
    pub time_step_min: f32,
    /// Maximum timestep clamp.
    pub time_step_max: f32,
    /// Number of tokens per prefill scan chunk.
    pub chunk_size: i32,
    #[param]
    /// Joint input projection.
    pub in_proj: nn::Linear,
    #[param]
    /// Depthwise causal convolution.
    pub conv1d: DepthwiseConv1d,
    #[param]
    /// Timestep bias.
    pub dt_bias: Param<Array>,
    #[param]
    /// Log transition parameter.
    pub A_log: Param<Array>,
    #[param]
    /// Skip parameter.
    pub D: Param<Array>,
    #[param]
    /// Gated RMSNorm.
    pub norm: MambaRmsNormGated,
    #[param]
    /// Output projection.
    pub out_proj: nn::Linear,
}

impl Mamba2Mixer {
    /// Creates an unloaded Mamba2 mixer.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let intermediate_size = args.mamba_num_heads * args.mamba_head_dim;
        let conv_dim = intermediate_size + 2 * args.n_groups * args.ssm_state_size;
        let projection_size = intermediate_size + conv_dim + args.mamba_num_heads;
        let d_mlp = (projection_size
            - 2 * intermediate_size
            - 2 * args.n_groups * args.ssm_state_size
            - args.mamba_num_heads)
            / 2;
        Ok(Self {
            num_heads: args.mamba_num_heads,
            head_dim: args.mamba_head_dim,
            ssm_state_size: args.ssm_state_size,
            n_groups: args.n_groups,
            intermediate_size,
            conv_dim,
            conv_kernel_size: args.conv_kernel,
            d_mlp,
            time_step_min: args.time_step_min,
            time_step_max: args.time_step_max,
            chunk_size: args.chunk_size,
            in_proj: nn::Linear::unloaded(
                args.hidden_size,
                projection_size,
                args.use_bias,
                Dtype::Float32,
                stream,
            )?,
            conv1d: DepthwiseConv1d::new(conv_dim, args.conv_kernel, args.use_conv_bias, stream)?,
            dt_bias: Param::<Array>::unloaded(&[args.mamba_num_heads], Dtype::Float32, stream)?,
            A_log: Param::<Array>::unloaded(&[args.mamba_num_heads], Dtype::Float32, stream)?,
            D: Param::<Array>::unloaded(&[args.mamba_num_heads], Dtype::Float32, stream)?,
            norm: MambaRmsNormGated::new(
                intermediate_size,
                args.n_groups,
                args.layer_norm_epsilon,
                stream,
            )?,
            out_proj: nn::Linear::unloaded(
                intermediate_size,
                args.hidden_size,
                args.use_bias,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn depthwise_causal_conv(
        &self,
        hidden_states_b_c: &Array,
        cache: Option<&mut Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states_b_c.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        let state_len = self.conv_kernel_size - 1;
        let state = cache
            .as_ref()
            .and_then(|cache| cache.conv_state.clone())
            .unwrap_or(zeros::<f32>(&[batch, state_len, channels], stream)?);
        let padded = concatenate_axis(&[state, hidden_states_b_c.clone()], 1, stream)?;
        if let Some(cache) = cache {
            cache.conv_state = Some(padded.try_index_device((.., seq_len.., ..), stream)?);
            cache.offset += seq_len;
        }

        let weight = self.conv1d.weight.as_ref().swap_axes(1, 2, stream)?;
        let mut out = conv1d(
            &padded,
            &weight,
            Some(1),
            Some(0),
            Some(1),
            Some(channels),
            stream,
        )?;
        if let Some(bias) = self.conv1d.bias.as_ref() {
            out = out.add(bias, stream)?;
        }
        silu(out, stream)
    }

    fn expand_group_states(
        &self,
        x: Array,
        batch: i32,
        seq_len: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let repeats = self.num_heads / self.n_groups;
        let grouped = x.reshape(
            &[batch, seq_len, self.n_groups, self.ssm_state_size],
            stream,
        )?;
        if repeats == 1 {
            return Ok(grouped);
        }
        broadcast_to(
            &grouped.try_index_device((.., .., .., NewAxis, ..), stream)?,
            &[batch, seq_len, self.n_groups, repeats, self.ssm_state_size],
            stream,
        )?
        .reshape(
            &[batch, seq_len, self.num_heads, self.ssm_state_size],
            stream,
        )
    }

    fn initial_ssm_state(
        &self,
        batch: i32,
        cache: Option<&Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        cache
            .and_then(|cache| cache.ssm_state.clone())
            .map(Ok)
            .unwrap_or_else(|| {
                zeros::<f32>(
                    &[batch, self.num_heads, self.head_dim, self.ssm_state_size],
                    stream,
                )
            })
    }

    fn scan_token(
        &self,
        state: Array,
        x_t: Array,
        b_t: Array,
        c_t: Array,
        dt_t: Array,
        a: &Array,
        d: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let batch = x_t.dim(0);
        let dt_t = clip(&dt_t, (self.time_step_min, self.time_step_max), stream)?;
        let d_a = exp(
            dt_t.reshape(&[batch, self.num_heads, 1, 1], stream)?
                .multiply(a, stream)?,
            stream,
        )?;
        let d_b = dt_t
            .reshape(&[batch, self.num_heads, 1], stream)?
            .multiply(&b_t, stream)?;
        let d_b_x = x_t
            .try_index_device((.., .., .., NewAxis), stream)?
            .multiply(d_b.try_index_device((.., .., NewAxis, ..), stream)?, stream)?;
        let state = state.multiply(d_a, stream)?.add(d_b_x, stream)?;
        let y = sum_axis(
            state.multiply(c_t.try_index_device((.., .., NewAxis, ..), stream)?, stream)?,
            -1,
            false,
            stream,
        )?
        .add(x_t.multiply(d, stream)?, stream)?;
        Ok((state, y))
    }

    fn scan_chunk(
        &self,
        hidden_states: &Array,
        b_states: &Array,
        c_states: &Array,
        dt: &Array,
        start: i32,
        end: i32,
        mut state: Array,
        a: &Array,
        d: &Array,
        dt_bias: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let batch = hidden_states.dim(0);
        let mut outputs = Vec::with_capacity((end - start) as usize);
        for index in start..end {
            let x_t = hidden_states.try_index_device((.., index, .., ..), stream)?;
            let b_t = b_states.try_index_device((.., index, .., ..), stream)?;
            let c_t = c_states.try_index_device((.., index, .., ..), stream)?;
            let dt_t = nn::softplus(
                dt.try_index_device((.., index..index + 1, ..), stream)?
                    .add(dt_bias, stream)?,
                stream,
            )?
            .reshape(&[batch, self.num_heads], stream)?;
            let (next_state, y) = self.scan_token(state, x_t, b_t, c_t, dt_t, a, d, stream)?;
            state = next_state;
            outputs.push(y.try_index_device((.., NewAxis, .., ..), stream)?);
        }
        Ok((state, concatenate_axis(&outputs, 1, stream)?))
    }

    fn selective_scan_prefill_chunked(
        &self,
        hidden_states: Array,
        b_states: Array,
        c_states: Array,
        dt: Array,
        cache: Option<&mut Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut state = self.initial_ssm_state(batch, cache.as_deref(), stream)?;
        let a = exp(self.A_log.as_ref(), stream)?
            .multiply(Array::from_f32(-1.0), stream)?
            .reshape(&[1, self.num_heads, 1, 1], stream)?;
        let d = self.D.as_ref().reshape(&[1, self.num_heads, 1], stream)?;
        let dt_bias = self
            .dt_bias
            .as_ref()
            .reshape(&[1, 1, self.num_heads], stream)?;
        let chunk_size = self.chunk_size.max(1);
        let mut outputs = Vec::with_capacity(((seq_len + chunk_size - 1) / chunk_size) as usize);
        let mut start = 0;
        while start < seq_len {
            let end = (start + chunk_size).min(seq_len);
            let (next_state, chunk) = self.scan_chunk(
                &hidden_states,
                &b_states,
                &c_states,
                &dt,
                start,
                end,
                state,
                &a,
                &d,
                &dt_bias,
                stream,
            )?;
            state = next_state;
            outputs.push(chunk);
            start = end;
        }
        if let Some(cache) = cache {
            cache.ssm_state = Some(state);
        }
        concatenate_axis(&outputs, 1, stream)
    }

    fn selective_scan_decode(
        &self,
        hidden_states: Array,
        b_states: Array,
        c_states: Array,
        dt: Array,
        cache: Option<&mut Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if hidden_states.dim(1) != 1 {
            return Err(Exception::custom(
                "Nemotron-H Mamba2 decode scan expects exactly one token",
            ));
        }
        let batch = hidden_states.dim(0);
        let state = self.initial_ssm_state(batch, cache.as_deref(), stream)?;
        let a = exp(self.A_log.as_ref(), stream)?
            .multiply(Array::from_f32(-1.0), stream)?
            .reshape(&[1, self.num_heads, 1, 1], stream)?;
        let d = self.D.as_ref().reshape(&[1, self.num_heads, 1], stream)?;
        let dt_bias = self
            .dt_bias
            .as_ref()
            .reshape(&[1, 1, self.num_heads], stream)?;
        let x_t = hidden_states.try_index_device((.., 0, .., ..), stream)?;
        let b_t = b_states.try_index_device((.., 0, .., ..), stream)?;
        let c_t = c_states.try_index_device((.., 0, .., ..), stream)?;
        let dt_t = nn::softplus(
            dt.try_index_device((.., 0..1, ..), stream)?
                .add(&dt_bias, stream)?,
            stream,
        )?
        .reshape(&[batch, self.num_heads], stream)?;
        let (state, y) = self.scan_token(state, x_t, b_t, c_t, dt_t, &a, &d, stream)?;
        if let Some(cache) = cache {
            cache.ssm_state = Some(state);
        }
        y.try_index_device((.., NewAxis, .., ..), stream)
    }
}

/// Input for a Mamba2 mixer.
pub struct Mamba2Input<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional Mamba2 cache.
    pub cache: Option<&'a mut Mamba2Cache>,
}

impl Module<Mamba2Input<'_>> for Mamba2Mixer {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: Mamba2Input<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let Mamba2Input { x, mut cache } = input;
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let is_decode = seq_len == 1 && cache.as_ref().is_some_and(|cache| cache.offset > 0);
        let projected = self.in_proj.forward(x, stream)?;
        let gate_start = 2 * self.d_mlp;
        let conv_start = gate_start + self.intermediate_size;
        let dt_start = conv_start + self.conv_dim;
        let gate = projected.try_index_device((.., .., gate_start..conv_start), stream)?;
        let hidden_states_b_c =
            projected.try_index_device((.., .., conv_start..dt_start), stream)?;
        let dt = projected.try_index_device((.., .., dt_start..), stream)?;
        let hidden_states_b_c =
            self.depthwise_causal_conv(&hidden_states_b_c, cache.as_deref_mut(), stream)?;
        let hidden_states = hidden_states_b_c
            .try_index_device((.., .., ..self.intermediate_size), stream)?
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim], stream)?;
        let b_start = self.intermediate_size;
        let c_start = b_start + self.n_groups * self.ssm_state_size;
        let b_states = self.expand_group_states(
            hidden_states_b_c.try_index_device((.., .., b_start..c_start), stream)?,
            batch,
            seq_len,
            stream,
        )?;
        let c_states = self.expand_group_states(
            hidden_states_b_c.try_index_device((.., .., c_start..), stream)?,
            batch,
            seq_len,
            stream,
        )?;
        let scan = if is_decode {
            self.selective_scan_decode(hidden_states, b_states, c_states, dt, cache, stream)?
        } else {
            self.selective_scan_prefill_chunked(
                hidden_states,
                b_states,
                c_states,
                dt,
                cache,
                stream,
            )?
        }
        .reshape(&[batch, seq_len, self.intermediate_size], stream)?;
        let scan = self.norm.forward(&scan, &gate, stream)?;
        self.out_proj.forward(&scan, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.in_proj.training_mode(mode);
        self.norm.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone)]
/// Per-layer cache for a Nemotron-H block.
pub enum LayerCache {
    /// Mamba2 convolution and SSM cache.
    Mamba(Mamba2Cache),
    /// Attention key/value cache.
    Attention(ConcatKeyValueCache),
    /// MLP block cache placeholder.
    Mlp,
    /// MoE block cache placeholder.
    Moe,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Pattern-selected Nemotron-H block.
pub struct TransformerBlock {
    /// Layer block type.
    pub block_type: LayerBlockType,
    #[param]
    /// Pre-mixer RMSNorm.
    pub norm: nn::RmsNorm,
    #[param]
    /// Mamba2 mixer for `M` layers.
    pub mamba: Option<Mamba2Mixer>,
    #[param]
    /// GQA attention mixer for `*` layers.
    pub attention: Option<Attention>,
    #[param]
    /// Dense MLP mixer for `-` layers.
    pub mlp: Option<Mlp>,
    #[param]
    /// Sparse MoE mixer for `E` layers.
    pub moe: Option<SparseMoeBlock>,
}

impl TransformerBlock {
    /// Creates an unloaded block for `layer_idx`.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Error> {
        let block_type = args.layer_block_type(layer_idx)?;
        Ok(Self {
            block_type,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.layer_norm_epsilon,
                Dtype::Float32,
                stream,
            )?,
            mamba: if block_type == LayerBlockType::Mamba {
                Some(Mamba2Mixer::new(args, stream)?)
            } else {
                None
            },
            attention: if block_type == LayerBlockType::Attention {
                Some(Attention::new(args, stream)?)
            } else {
                None
            },
            mlp: if block_type == LayerBlockType::Mlp {
                Some(Mlp::new(
                    args.hidden_size,
                    args.intermediate_size,
                    args.mlp_bias,
                    stream,
                )?)
            } else {
                None
            },
            moe: if block_type == LayerBlockType::Moe {
                Some(SparseMoeBlock::new(args, stream)?)
            } else {
                None
            },
        })
    }
}

/// Input for a Nemotron-H block.
pub struct BlockInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask for attention layers.
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
        let h = self.norm.forward(x, stream)?;
        let h = match (self.block_type, cache) {
            (LayerBlockType::Mamba, Some(LayerCache::Mamba(cache))) => {
                self.mamba.as_mut().expect("mamba block").forward(
                    Mamba2Input {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerBlockType::Mamba, _) => self
                .mamba
                .as_mut()
                .expect("mamba block")
                .forward(Mamba2Input { x: &h, cache: None }, stream)?,
            (LayerBlockType::Attention, Some(LayerCache::Attention(cache))) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerBlockType::Attention, _) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (LayerBlockType::Mlp, _) => {
                self.mlp.as_mut().expect("mlp block").forward(&h, stream)?
            }
            (LayerBlockType::Moe, _) => {
                self.moe.as_mut().expect("moe block").forward(&h, stream)?
            }
        };
        residual.add(h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        if let Some(mamba) = &mut self.mamba {
            mamba.training_mode(mode);
        }
        if let Some(attention) = &mut self.attention {
            attention.training_mode(mode);
        }
        if let Some(mlp) = &mut self.mlp {
            mlp.training_mode(mode);
        }
        if let Some(moe) = &mut self.moe {
            moe.training_mode(mode);
        }
    }
}

impl LayerCache {
    fn new(block_type: LayerBlockType) -> Self {
        match block_type {
            LayerBlockType::Mamba => Self::Mamba(Mamba2Cache::default()),
            LayerBlockType::Attention => Self::Attention(ConcatKeyValueCache::new()),
            LayerBlockType::Mlp => Self::Mlp,
            LayerBlockType::Moe => Self::Moe,
        }
    }

    fn offset(&self) -> Option<i32> {
        match self {
            Self::Mamba(cache) => Some(cache.offset),
            Self::Attention(cache) => Some(cache.offset()),
            Self::Mlp | Self::Moe => None,
        }
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous cache for Nemotron-H hybrid layers.
pub struct Cache {
    /// Per-layer cache state.
    pub layers: Vec<LayerCache>,
}

impl Cache {
    /// Creates an empty cache matching the hybrid layer pattern.
    pub fn new(args: &ModelArgs) -> Result<Self, Error> {
        Ok(Self {
            layers: args
                .layer_block_types()?
                .into_iter()
                .map(LayerCache::new)
                .collect(),
        })
    }

    /// Returns the current sequence offset represented by the first stateful layer.
    pub fn offset(&self) -> i32 {
        self.layers.iter().find_map(LayerCache::offset).unwrap_or(0)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Nemotron-H transformer body without the language-model head.
pub struct NemotronHModel {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of hybrid layers.
    pub num_hidden_layers: i32,
    #[param]
    /// Token embedding table.
    pub embeddings: nn::Embedding,
    #[param]
    /// Hybrid transformer blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final normalization.
    pub norm_f: nn::RmsNorm,
}

impl NemotronHModel {
    /// Creates an unloaded Nemotron-H transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let embeddings =
            nn::Embedding::unloaded(args.vocab_size, args.hidden_size, Dtype::Float32, stream)?;
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new(args, idx as usize, stream))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let norm_f =
            nn::RmsNorm::unloaded(args.hidden_size, args.norm_eps, Dtype::Float32, stream)?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embeddings,
            layers,
            norm_f,
        })
    }
}

/// Input for a Nemotron-H forward pass.
pub struct ModelInput<'a> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional heterogeneous cache.
    pub cache: Option<&'a mut Cache>,
}

impl Module<ModelInput<'_>> for NemotronHModel {
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
        let mut h = self.embeddings.forward(inputs, stream)?;
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
        self.norm_f.forward(&h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embeddings.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm_f.training_mode(mode);
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
/// Nemotron-H causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    #[param]
    /// Transformer body.
    pub model: NemotronHModel,
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<nn::Linear>,
}

impl Model {
    /// Creates an unloaded Nemotron-H causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let model = NemotronHModel::new(&args, stream)?;
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
            model,
            lm_head,
        })
    }

    /// Creates an empty heterogeneous cache for this model.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args).expect("validated Nemotron-H layer pattern")
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn project_logits(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        project_logits_dense(
            &mut self.lm_head,
            &self.model.embeddings,
            hidden_states,
            stream,
        )
    }

    fn forward_logits(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
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
}

/// Nemotron-H token generation iterator.
pub type Generate<'a, S = crate::sampler::DefaultSampler> = common::Generate<'a, Model, Cache, S>;

/// Loads `tokenizer.json` from a Nemotron-H model directory.
pub fn load_nemotron_h_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads Nemotron-H model arguments from `config.json`.
pub fn get_nemotron_h_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(file)?;
    validate_model_args(&args)?;
    Ok(args)
}

/// Loads a Nemotron-H model and safetensors weights from a model directory.
pub fn load_nemotron_h_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_nemotron_h_model_args(model_dir)?;
    let mut model = Model::new(args.clone(), stream)?;
    let config = nemotron_h_strict_load_config();
    let mut report = StrictLoadReport::default();
    load_nemotron_h_safetensors_strict(
        &mut model,
        model_dir,
        &args,
        weights_stream,
        stream,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Strict-loading key rules for Nemotron-H checkpoints.
pub fn nemotron_h_strict_load_config() -> StrictLoadConfig {
    StrictLoadConfig::default()
        .rewrite_prefix("backbone.", "model.")
        .rewrite_prefix("model.backbone.", "model.")
}

/// Remaps public Nemotron-H checkpoint tensors to the runtime parameter tree.
pub fn transform_nemotron_h_weights(
    loaded: std::collections::HashMap<String, Array>,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<std::collections::HashMap<String, Array>, Error> {
    let mut rewritten = std::collections::HashMap::with_capacity(loaded.len());
    for (key, value) in loaded {
        rewritten.insert(rewrite_nemotron_h_weight_key(&key, args)?, value);
    }
    transform_split_relu2_experts(rewritten, args.n_routed_experts, stream)
}

fn rewrite_nemotron_h_weight_key(key: &str, args: &ModelArgs) -> Result<String, Error> {
    let Some(rest) = key.strip_prefix("backbone.layers.") else {
        return Ok(key.to_string());
    };
    let Some((layer_idx, suffix)) = rest.split_once(".mixer.") else {
        return Ok(key.to_string());
    };
    let layer_idx = layer_idx.parse::<usize>().map_err(|error| {
        Error::UnsupportedArchitecture(format!(
            "invalid Nemotron-H layer index in checkpoint key '{key}': {error}"
        ))
    })?;
    let field = match args.layer_block_type(layer_idx)? {
        LayerBlockType::Mamba => "mamba",
        LayerBlockType::Attention => "attention",
        LayerBlockType::Mlp => "mlp",
        LayerBlockType::Moe => "moe",
    };
    Ok(format!("backbone.layers.{layer_idx}.{field}.{suffix}"))
}

/// Strict-loads Nemotron-H safetensors into a module after applying checkpoint remaps.
pub fn load_nemotron_h_safetensors_strict<M: safemlx::module::ModuleParameters>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    args: &ModelArgs,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    load_safetensors_dir_strict_with_split_relu2_experts(
        model,
        model_dir,
        weights_stream,
        transform_stream,
        config,
        report,
        args.n_routed_experts,
        |key| rewrite_nemotron_h_weight_key(key, args),
    )
}

/// Validates a parsed Nemotron-H config value.
pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    let args: ModelArgs = serde_json::from_value(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid nemotron_h config: {error}"))
    })?;
    validate_model_args(&args)
}

fn validate_model_args(args: &ModelArgs) -> Result<(), Error> {
    if args.model_type != "nemotron_h" {
        return Err(Error::UnsupportedModelType(args.model_type.clone()));
    }

    let layers = args.layer_block_types()?;
    if layers.len() != args.num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H hybrid_override_pattern has {} layers, expected {}",
            layers.len(),
            args.num_hidden_layers
        )));
    }

    ensure_positive("vocab_size", args.vocab_size)?;
    ensure_positive("hidden_size", args.hidden_size)?;
    ensure_positive("num_hidden_layers", args.num_hidden_layers)?;
    ensure_positive("num_attention_heads", args.num_attention_heads)?;
    ensure_positive("num_key_value_heads", args.num_key_value_heads)?;
    ensure_positive("head_dim", args.head_dim)?;
    ensure_positive("ssm_state_size", args.ssm_state_size)?;
    ensure_positive("mamba_num_heads", args.mamba_num_heads)?;
    ensure_positive("n_groups", args.n_groups)?;
    ensure_positive("mamba_head_dim", args.mamba_head_dim)?;
    ensure_positive("conv_kernel", args.conv_kernel)?;
    ensure_positive("chunk_size", args.chunk_size)?;
    ensure_positive("n_routed_experts", args.n_routed_experts)?;
    ensure_positive("n_shared_experts", args.n_shared_experts)?;
    ensure_positive("num_experts_per_tok", args.num_experts_per_tok)?;

    if args.num_experts_per_tok > args.n_routed_experts {
        return Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H num_experts_per_tok ({}) exceeds n_routed_experts ({})",
            args.num_experts_per_tok, args.n_routed_experts
        )));
    }
    if args.mlp_hidden_act != "relu2" {
        return Err(Error::UnsupportedArchitecture(format!(
            "unsupported Nemotron-H MLP activation '{}'",
            args.mlp_hidden_act
        )));
    }
    if args.mamba_hidden_act != "silu" {
        return Err(Error::UnsupportedArchitecture(format!(
            "unsupported Nemotron-H Mamba activation '{}'",
            args.mamba_hidden_act
        )));
    }
    if let Some(torch_dtype) = &args.torch_dtype {
        if !matches!(
            torch_dtype.as_str(),
            "bfloat16" | "bf16" | "float16" | "float32"
        ) {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Nemotron-H torch_dtype '{torch_dtype}'"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        load_nemotron_h_model, load_nemotron_h_safetensors_strict, rewrite_nemotron_h_weight_key,
        validate_model_config_value, LayerBlockType, Model, ModelArgs, SparseMoeBlock,
    };
    use crate::models::common::{CausalLm, TopKRouterScoreFunction};
    use crate::weights::{StrictLoadConfig, StrictLoadReport};
    use safemlx::{module::ModuleParameters, ops::indexing::TryIndexOp, Array, ExecutionContext};
    use serde_json::json;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn nemotron_nano_config() -> serde_json::Value {
        serde_json::from_str(
            r#"{
              "attention_bias": false,
              "chunk_size": 128,
              "conv_kernel": 4,
              "eos_token_id": 2,
              "expand": 2,
              "head_dim": 128,
              "hidden_size": 2688,
              "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
              "intermediate_size": 1856,
              "layer_norm_epsilon": 1e-05,
              "mamba_head_dim": 64,
              "mamba_hidden_act": "silu",
              "mamba_num_heads": 64,
              "mamba_proj_bias": false,
              "max_position_embeddings": 262144,
              "mlp_bias": false,
              "mlp_hidden_act": "relu2",
              "model_type": "nemotron_h",
              "moe_intermediate_size": 1856,
              "moe_shared_expert_intermediate_size": 3712,
              "n_group": 1,
              "n_groups": 8,
              "n_routed_experts": 128,
              "n_shared_experts": 1,
              "norm_eps": 1e-05,
              "norm_topk_prob": true,
              "num_attention_heads": 32,
              "num_experts_per_tok": 6,
              "num_hidden_layers": 52,
              "num_key_value_heads": 2,
              "rescale_prenorm_residual": true,
              "rope_theta": 10000,
              "routed_scaling_factor": 2.5,
              "ssm_state_size": 128,
              "tie_word_embeddings": false,
              "time_step_floor": 0.0001,
              "time_step_max": 0.1,
              "time_step_min": 0.001,
              "topk_group": 1,
              "torch_dtype": "bfloat16",
              "use_bias": false,
              "use_conv_bias": true,
              "vocab_size": 131072
            }"#,
        )
        .unwrap()
    }

    fn temp_model_dir() -> std::path::PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nemotron_h_test_{}_{}_{}",
            std::process::id(),
            id,
            counter
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_moe_args() -> ModelArgs {
        let mut config = nemotron_nano_config();
        config["hidden_size"] = json!(4);
        config["intermediate_size"] = json!(3);
        config["moe_intermediate_size"] = json!(3);
        config["moe_shared_expert_intermediate_size"] = json!(5);
        config["n_routed_experts"] = json!(2);
        config["num_experts_per_tok"] = json!(2);
        serde_json::from_value(config).unwrap()
    }

    fn tiny_full_args() -> ModelArgs {
        let mut config = nemotron_nano_config();
        config["vocab_size"] = json!(16);
        config["hidden_size"] = json!(8);
        config["intermediate_size"] = json!(12);
        config["num_hidden_layers"] = json!(4);
        config["hybrid_override_pattern"] = json!("M-E*");
        config["num_attention_heads"] = json!(2);
        config["num_key_value_heads"] = json!(1);
        config["head_dim"] = json!(4);
        config["mamba_num_heads"] = json!(2);
        config["mamba_head_dim"] = json!(4);
        config["n_groups"] = json!(1);
        config["ssm_state_size"] = json!(4);
        config["conv_kernel"] = json!(3);
        config["chunk_size"] = json!(2);
        config["moe_intermediate_size"] = json!(6);
        config["moe_shared_expert_intermediate_size"] = json!(10);
        config["n_routed_experts"] = json!(2);
        config["num_experts_per_tok"] = json!(2);
        serde_json::from_value(config).unwrap()
    }

    #[test]
    fn parses_nemotron_nano_config_fields() {
        let args: ModelArgs = serde_json::from_value(nemotron_nano_config()).unwrap();
        assert_eq!(args.model_type, "nemotron_h");
        assert_eq!(args.hidden_size, 2688);
        assert_eq!(args.num_hidden_layers, 52);
        assert_eq!(args.n_routed_experts, 128);
        assert_eq!(args.n_shared_experts, 1);
        assert_eq!(args.num_experts_per_tok, 6);
        assert_eq!(args.mlp_hidden_act, "relu2");
        assert_eq!(args.mamba_hidden_act, "silu");
        assert_eq!(args.torch_dtype.as_deref(), Some("bfloat16"));

        let blocks = args.layer_block_types().unwrap();
        assert_eq!(blocks.len(), 52);
        assert_eq!(
            blocks
                .iter()
                .filter(|&&block| block == LayerBlockType::Mamba)
                .count(),
            23
        );
        assert_eq!(
            blocks
                .iter()
                .filter(|&&block| block == LayerBlockType::Moe)
                .count(),
            23
        );
        assert_eq!(
            blocks
                .iter()
                .filter(|&&block| block == LayerBlockType::Attention)
                .count(),
            6
        );
    }

    #[test]
    fn validates_nemotron_nano_config() {
        validate_model_config_value(&nemotron_nano_config()).unwrap();
    }

    #[test]
    fn rewrites_public_mixer_keys_by_layer_type() {
        let args = tiny_full_args();
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.0.mixer.in_proj.weight", &args).unwrap(),
            "backbone.layers.0.mamba.in_proj.weight"
        );
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.1.mixer.up_proj.weight", &args).unwrap(),
            "backbone.layers.1.mlp.up_proj.weight"
        );
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.2.mixer.gate.weight", &args).unwrap(),
            "backbone.layers.2.moe.gate.weight"
        );
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.3.mixer.q_proj.weight", &args).unwrap(),
            "backbone.layers.3.attention.q_proj.weight"
        );
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_packs_public_split_moe_expert_weights() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let args = tiny_moe_args();
        let dir = temp_model_dir();
        let weights_path = dir.join("model.safetensors");
        let arrays = vec![
            (
                "backbone.layers.1.mixer.gate.weight",
                Array::zeros::<f32>(&[2, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.gate.e_score_correction_bias",
                Array::zeros::<f32>(&[2], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.0.up_proj.weight",
                Array::zeros::<f32>(&[3, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.0.down_proj.weight",
                Array::zeros::<f32>(&[4, 3], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.1.up_proj.weight",
                Array::zeros::<f32>(&[3, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.1.down_proj.weight",
                Array::zeros::<f32>(&[4, 3], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.shared_experts.up_proj.weight",
                Array::zeros::<f32>(&[5, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.shared_experts.down_proj.weight",
                Array::zeros::<f32>(&[4, 5], stream).unwrap(),
            ),
        ];
        Array::save_safetensors(
            arrays.iter().map(|(key, value)| (*key, value)),
            None,
            &weights_path,
        )
        .unwrap();

        let mut moe = SparseMoeBlock::new(&args, stream).unwrap();
        let config = StrictLoadConfig::default().rewrite_prefix("backbone.layers.1.moe.", "");
        let mut report = StrictLoadReport::default();
        load_nemotron_h_safetensors_strict(
            &mut moe,
            &dir,
            &args,
            stream,
            stream,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&moe, &config).unwrap();
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_model_parameter_tree_matches_public_checkpoint_roots() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let model = Model::new(tiny_full_args(), stream).unwrap();
        let params = model.parameters().flatten();

        for key in [
            "model.embeddings.weight",
            "model.layers.0.norm.weight",
            "model.layers.0.mamba.in_proj.weight",
            "model.layers.0.mamba.conv1d.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.2.moe.gate.weight",
            "model.layers.2.moe.gate.e_score_correction_bias",
            "model.layers.2.moe.experts.up_proj",
            "model.layers.2.moe.shared_experts.up_proj.weight",
            "model.layers.3.attention.q_proj.weight",
            "model.norm_f.weight",
            "lm_head.weight",
        ] {
            assert!(params.contains_key(key), "missing parameter key {key}");
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_model_prefill_and_decode_shape_smoke() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut model = Model::new(tiny_full_args(), stream).unwrap();
        let mut cache = model.new_cache();
        let prompt = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
        let logits = CausalLm::prefill_logits(&mut model, &prompt, &mut cache, stream).unwrap();
        assert_eq!(logits.shape(), &[1, 16]);
        assert!(cache.offset() >= 3);

        let next = Array::from_slice(&[4_u32], &[1, 1]);
        let logits = CausalLm::decode_logits(&mut model, &next, &mut cache, stream).unwrap();
        assert_eq!(logits.shape(), &[1, 16]);
        assert!(cache.offset() >= 4);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_full_model_from_public_checkpoint_key_shapes() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();
        let args = tiny_full_args();
        let source = Model::new(args.clone(), stream).unwrap();
        let dir = temp_model_dir();
        let weights_path = dir.join("model.safetensors");
        let mut arrays = Vec::new();
        for (key, value) in source.parameters().flatten() {
            if key.as_ref() == "model.layers.2.moe.experts.up_proj" {
                for expert in 0..args.n_routed_experts {
                    arrays.push((
                        format!("backbone.layers.2.mixer.experts.{expert}.up_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
                continue;
            }
            if key.as_ref() == "model.layers.2.moe.experts.down_proj" {
                for expert in 0..args.n_routed_experts {
                    arrays.push((
                        format!("backbone.layers.2.mixer.experts.{expert}.down_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
                continue;
            }

            let public_key = key
                .strip_prefix("model.embeddings.")
                .map(|rest| format!("backbone.embeddings.{rest}"))
                .or_else(|| {
                    key.strip_prefix("model.norm_f.")
                        .map(|rest| format!("backbone.norm_f.{rest}"))
                })
                .or_else(|| {
                    key.strip_prefix("model.layers.").map(|rest| {
                        let (layer, suffix) = rest.split_once('.').unwrap();
                        if suffix.starts_with("norm.") {
                            return format!("backbone.layers.{layer}.{suffix}");
                        }
                        let suffix = suffix
                            .strip_prefix("mamba.")
                            .or_else(|| suffix.strip_prefix("attention."))
                            .or_else(|| suffix.strip_prefix("mlp."))
                            .or_else(|| suffix.strip_prefix("moe."))
                            .unwrap_or(suffix);
                        format!("backbone.layers.{layer}.mixer.{suffix}")
                    })
                })
                .unwrap_or_else(|| key.to_string());
            arrays.push((public_key, value.clone()));
        }
        Array::save_safetensors(
            arrays.iter().map(|(key, value)| (key.as_str(), value)),
            None,
            &weights_path,
        )
        .unwrap();

        let mut target = Model::new(args.clone(), stream).unwrap();
        let config = super::nemotron_h_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_nemotron_h_safetensors_strict(
            &mut target,
            &dir,
            &args,
            weights_stream,
            stream,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&target, &config).unwrap();
    }

    #[test]
    #[ignore = "requires a local Nemotron-H checkpoint and MLX runtime execution"]
    fn strict_loads_real_public_checkpoint() {
        let model_dir = PathBuf::from(
            std::env::var("NEMOTRON_H_MODEL_DIR")
                .expect("set NEMOTRON_H_MODEL_DIR to a local Nemotron-H snapshot"),
        );
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();

        let model = load_nemotron_h_model(&model_dir, stream, weights_stream).unwrap();
        assert_eq!(model.model_type(), "nemotron_h");
        assert_eq!(model.args.num_hidden_layers, 52);
        assert_eq!(model.args.n_routed_experts, 128);
        assert_eq!(model.args.num_experts_per_tok, 6);
        assert_eq!(model.new_cache().layers.len(), 52);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn moe_parameter_tree_uses_nemotron_weight_names_and_policy() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args: ModelArgs = serde_json::from_value(nemotron_nano_config()).unwrap();
        let moe = SparseMoeBlock::new(&args, stream).unwrap();
        let params = moe.parameters().flatten();

        assert_eq!(moe.gate.top_k, 6);
        assert_eq!(moe.gate.num_experts, 128);
        assert_eq!(moe.gate.score_function, TopKRouterScoreFunction::Sigmoid);
        assert_eq!(moe.gate.routed_scaling_factor, 2.5);
        assert!(moe.gate.norm_topk_prob);
        assert_eq!(moe.experts.num_experts, 128);

        for key in [
            "gate.weight",
            "gate.e_score_correction_bias",
            "experts.up_proj",
            "experts.down_proj",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        ] {
            assert!(params.contains_key(key), "missing parameter key {key}");
        }
        assert_eq!(params["gate.weight"].shape(), &[128, args.hidden_size]);
        assert_eq!(params["gate.e_score_correction_bias"].shape(), &[128]);
        assert_eq!(
            params["experts.up_proj"].shape(),
            &[128, args.moe_intermediate_size, args.hidden_size]
        );
        assert_eq!(
            params["experts.down_proj"].shape(),
            &[128, args.hidden_size, args.moe_intermediate_size]
        );
    }

    #[test]
    fn rejects_mismatched_hybrid_pattern_length() {
        let mut config = nemotron_nano_config();
        config["hybrid_override_pattern"] = json!("ME");

        let error = validate_model_config_value(&config).unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported model architecture: Nemotron-H hybrid_override_pattern has 2 layers, expected 52"
        );
    }
}
