//! Shared layers and generation machinery for decoder-only causal LMs.

use std::marker::PhantomData;

use safemlx::{
    argmax_axis, array,
    builder::Builder,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        argpartition_axis, broadcast_to, concatenate_axis, gather_grouped_rows,
        gather_route_values, grouped_matmul,
        indexing::{take_along_axis, topk_axis, NewAxis, TryIndexOp},
        matmul, maximum, r#where, segment_sum_by_index, sigmoid, softmax_axis, sum_axis,
        topk_route_plan, GroupedRoutePlan,
    },
    quantization::MaybeQuantized,
    random::{self, RandomState},
    Array, Dtype, Stream,
};

use crate::{
    cache::KeyValueCache,
    inspection::ActivationObserver,
    models::input,
    quantization::AffineQuantization,
    sampler::{DefaultSampler, Sampler},
    utils::{create_causal_mask, rope::RopeVariant, scaled_dot_product_attention},
};

/// Applies the SiLU activation function.
pub fn silu(x: Array, stream: &Stream) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x, stream)?, stream)
}

/// Applies the squared ReLU activation used by Nemotron-H dense and MoE MLPs.
pub fn relu2(x: Array, stream: &Stream) -> Result<Array, Exception> {
    maximum(&x, Array::from_f32(0.0), stream)?.square(stream)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// SwiGLU MLP with optionally quantized projections.
pub struct SwiGluMlp {
    #[quantizable]
    #[param]
    /// Gate projection.
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Down projection back to the model hidden size.
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Up projection.
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl SwiGluMlp {
    /// Creates an initialized SwiGLU MLP.
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            ),
            down_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden_dim, dim).bias(bias).build()?,
            ),
            up_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            ),
        })
    }

    /// Creates an unloaded SwiGLU MLP whose parameters can be populated from weights.
    pub fn unloaded(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::unloaded_with_quantization(dim, hidden_dim, bias, None, stream)
    }

    /// Creates an unloaded SwiGLU MLP with optional MLX affine projections.
    pub fn unloaded_with_quantization(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        quantization: Option<AffineQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: unloaded_maybe_quantized_linear(
                dim,
                hidden_dim,
                bias,
                quantization,
                stream,
            )?,
            down_proj: unloaded_maybe_quantized_linear(
                hidden_dim,
                dim,
                bias,
                quantization,
                stream,
            )?,
            up_proj: unloaded_maybe_quantized_linear(dim, hidden_dim, bias, quantization, stream)?,
        })
    }

    /// Forward pass that reports intermediate activations to an observer.
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

        let activated_gate = silu(gate, stream)?;
        observer.observe(&format!("{prefix}.gate_activation"), &activated_gate)?;

        let down_proj_input = activated_gate.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.down_proj_input"), &down_proj_input)?;

        let output = self.down_proj.forward(&down_proj_input, stream)?;
        observer.observe(&format!("{prefix}.down_proj"), &output)?;
        Ok(output)
    }
}

impl Module<&Array> for SwiGluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let down_proj_input = silu(self.gate_proj.forward(input, stream)?, stream)?
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
/// Dense SwiGLU MLP without quantized projection wrappers.
pub struct DenseSwiGluMlp {
    #[param]
    /// Gate projection.
    pub gate_proj: nn::Linear,
    #[param]
    /// Up projection.
    pub up_proj: nn::Linear,
    #[param]
    /// Down projection back to the model hidden size.
    pub down_proj: nn::Linear,
}

impl DenseSwiGluMlp {
    /// Creates an initialized dense SwiGLU MLP.
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            up_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            down_proj: nn::LinearBuilder::new(hidden_dim, dim).bias(bias).build()?,
        })
    }

    /// Creates an unloaded dense SwiGLU MLP whose parameters can be populated from weights.
    pub fn unloaded(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::Linear::unloaded(dim, hidden_dim, bias, Dtype::Float32, stream)?,
            up_proj: nn::Linear::unloaded(dim, hidden_dim, bias, Dtype::Float32, stream)?,
            down_proj: nn::Linear::unloaded(hidden_dim, dim, bias, Dtype::Float32, stream)?,
        })
    }
}

impl Module<&Array> for DenseSwiGluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let h = silu(self.gate_proj.forward(input, stream)?, stream)?
            .multiply(self.up_proj.forward(input, stream)?, stream)?;
        self.down_proj.forward(&h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

/// Router score transform used before top-k expert selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TopKRouterScoreFunction {
    /// Softmax scores, as used by Qwen MoE routers.
    Softmax,
    /// Sigmoid scores, as used by Nemotron/DeepSeek-style routers.
    Sigmoid,
}

/// Configuration for a reusable top-k MoE router.
#[derive(Debug, Clone, Copy)]
pub struct TopKRouterConfig {
    /// Number of selected experts per token.
    pub top_k: i32,
    /// Total number of routed experts.
    pub num_experts: i32,
    /// Hidden dimension consumed by the router projection.
    pub hidden_size: i32,
    /// Score transform to apply to router logits.
    pub score_function: TopKRouterScoreFunction,
    /// Whether selected top-k weights are normalized after gathering.
    pub norm_topk_prob: bool,
    /// Optional epsilon added to the normalization denominator.
    pub normalization_epsilon: f32,
    /// Final multiplier applied to gathered routing weights.
    pub routed_scaling_factor: f32,
    /// Number of routing groups.
    pub n_group: i32,
    /// Number of routing groups selected before expert top-k.
    pub topk_group: i32,
    /// Whether to allocate Nemotron-style expert score correction bias.
    pub score_correction_bias: bool,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Reusable top-k router for sparse MoE layers.
pub struct TopKRouter {
    /// Number of selected experts per token.
    pub top_k: i32,
    /// Total number of routed experts.
    pub num_experts: i32,
    /// Router score transform.
    pub score_function: TopKRouterScoreFunction,
    /// Whether selected probabilities are normalized.
    pub norm_topk_prob: bool,
    /// Optional epsilon added to the normalization denominator.
    pub normalization_epsilon: f32,
    /// Final multiplier applied to routing weights.
    pub routed_scaling_factor: f32,
    /// Number of routing groups.
    pub n_group: i32,
    /// Number of selected routing groups.
    pub topk_group: i32,
    #[param]
    /// Router projection weight.
    pub weight: Param<Array>,
    #[param]
    /// Optional score correction bias used only when choosing experts.
    pub e_score_correction_bias: Param<Option<Array>>,
}

/// Selected expert ids plus the score and weight arrays produced by a top-k router.
pub struct TopKRouterOutput {
    /// Selected expert ids with shape `[tokens, top_k]`.
    pub indices: Array,
    /// Router probabilities or scores gathered at the selected ids.
    pub scores: Array,
    /// Final routing weights after optional normalization/scaling.
    pub weights: Array,
}

impl TopKRouter {
    /// Creates an unloaded router.
    pub fn new(config: TopKRouterConfig, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            top_k: config.top_k,
            num_experts: config.num_experts,
            score_function: config.score_function,
            norm_topk_prob: config.norm_topk_prob,
            normalization_epsilon: config.normalization_epsilon,
            routed_scaling_factor: config.routed_scaling_factor,
            n_group: config.n_group,
            topk_group: config.topk_group,
            weight: Param::<Array>::unloaded(
                &[config.num_experts, config.hidden_size],
                Dtype::Float32,
                stream,
            )?,
            e_score_correction_bias: if config.score_correction_bias {
                Param::<Option<Array>>::unloaded_some(
                    &[config.num_experts],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
        })
    }

    /// Returns selected expert ids and per-route weights.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let flat = hidden_states.reshape(&[-1, hidden_states.dim(-1)], stream)?;
        let logits = matmul(&flat, self.weight.as_ref().transpose(stream)?, stream)?;
        let scores = match self.score_function {
            TopKRouterScoreFunction::Softmax => softmax_axis(&logits, -1, true, stream)?,
            TopKRouterScoreFunction::Sigmoid => sigmoid(logits, stream)?,
        };
        let mut scores_for_choice = scores.clone();
        if let Some(bias) = self.e_score_correction_bias.as_ref() {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
        }

        let top_k_index = self.topk_indices(&scores_for_choice, stream)?;
        let mut top_k_weights = take_along_axis(&scores, &top_k_index, -1, stream)?;
        if self.norm_topk_prob {
            let mut denominator = sum_axis(&top_k_weights, -1, true, stream)?;
            if self.normalization_epsilon != 0.0 {
                denominator =
                    denominator.add(Array::from_f32(self.normalization_epsilon), stream)?;
            }
            top_k_weights = top_k_weights.divide(denominator, stream)?;
        }
        if self.routed_scaling_factor != 1.0 {
            top_k_weights =
                top_k_weights.multiply(Array::from_f32(self.routed_scaling_factor), stream)?;
        }
        Ok((top_k_index, top_k_weights))
    }

    /// Returns selected expert ids and weights while reporting router internals.
    pub fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<TopKRouterOutput, Exception> {
        let flat = hidden_states.reshape(&[-1, hidden_states.dim(-1)], stream)?;
        let logits = matmul(&flat, self.weight.as_ref().transpose(stream)?, stream)?;
        observer.observe(&format!("{prefix}.router_logits"), &logits)?;
        let scores = match self.score_function {
            TopKRouterScoreFunction::Softmax => softmax_axis(&logits, -1, true, stream)?,
            TopKRouterScoreFunction::Sigmoid => sigmoid(logits, stream)?,
        };
        observer.observe(&format!("{prefix}.router_scores"), &scores)?;

        let mut scores_for_choice = scores.clone();
        if let Some(bias) = self.e_score_correction_bias.as_ref() {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
            observer.observe(
                &format!("{prefix}.router_scores_for_choice"),
                &scores_for_choice,
            )?;
        }

        let top_k_index = self.topk_indices(&scores_for_choice, stream)?;
        observer.observe(&format!("{prefix}.top_k_experts"), &top_k_index)?;
        let mut top_k_weights = take_along_axis(&scores, &top_k_index, -1, stream)?;
        let top_k_scores = top_k_weights.clone();
        observer.observe(&format!("{prefix}.top_k_scores"), &top_k_weights)?;
        if self.norm_topk_prob {
            let mut denominator = sum_axis(&top_k_weights, -1, true, stream)?;
            if self.normalization_epsilon != 0.0 {
                denominator =
                    denominator.add(Array::from_f32(self.normalization_epsilon), stream)?;
            }
            top_k_weights = top_k_weights.divide(denominator, stream)?;
            observer.observe(
                &format!("{prefix}.top_k_weights_normalized"),
                &top_k_weights,
            )?;
        }
        if self.routed_scaling_factor != 1.0 {
            top_k_weights =
                top_k_weights.multiply(Array::from_f32(self.routed_scaling_factor), stream)?;
            observer.observe(&format!("{prefix}.top_k_weights_scaled"), &top_k_weights)?;
        }
        Ok(TopKRouterOutput {
            indices: top_k_index,
            scores: top_k_scores,
            weights: top_k_weights,
        })
    }

    fn topk_indices(&self, scores_for_choice: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.n_group == 1 && self.topk_group == 1 {
            return argpartition_axis(scores_for_choice, -self.top_k, -1, stream)?
                .try_index_device((.., -self.top_k..), stream);
        }
        if self.n_group <= 0
            || self.topk_group <= 0
            || self.topk_group > self.n_group
            || self.num_experts % self.n_group != 0
        {
            return Err(Exception::custom(
                "invalid grouped MoE router configuration",
            ));
        }

        let tokens = scores_for_choice.dim(0);
        let experts_per_group = self.num_experts / self.n_group;
        let grouped =
            scores_for_choice.reshape(&[tokens, self.n_group, experts_per_group], stream)?;
        let group_top = 2.min(experts_per_group);
        let group_scores = sum_axis(
            &topk_axis(grouped, group_top, -1, stream)?,
            -1,
            false,
            stream,
        )?;
        let group_idx = argpartition_axis(&group_scores, -self.topk_group, -1, stream)?
            .try_index_device((.., -self.topk_group..), stream)?;

        let expert_group_ids: Vec<i32> = (0..self.num_experts)
            .map(|expert| expert / experts_per_group)
            .collect();
        let expert_group_ids = Array::from_slice(&expert_group_ids, &[1, 1, self.num_experts]);
        let selected_groups = group_idx.try_index_device((.., .., NewAxis), stream)?;
        let group_mask = selected_groups.eq(expert_group_ids, stream)?;
        let group_mask = sum_axis(
            &group_mask.as_dtype(Dtype::Int32, stream)?,
            1,
            false,
            stream,
        )?
        .gt(Array::from_int(0), stream)?;
        let masked_scores = r#where(&group_mask, scores_for_choice, Array::from_f32(0.0), stream)?;
        argpartition_axis(masked_scores, -self.top_k, -1, stream)?
            .try_index_device((.., -self.top_k..), stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

/// Applies route weights and reduces expert-major route outputs back to source tokens.
pub fn weighted_route_sum(
    current: Array,
    top_k_weights: &Array,
    plan: &GroupedRoutePlan,
    num_tokens: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let weights = gather_route_values(top_k_weights, plan, stream)?
        .try_index_device((.., NewAxis), stream)?;
    let weighted = current.multiply(weights, stream)?;
    segment_sum_by_index(weighted, &plan.token_indices, num_tokens, stream)
}

#[derive(Debug, Clone, ModuleParameters)]
/// Packed routed expert bank for ReLU2 experts with `up_proj` and `down_proj` weights.
pub struct PackedRelu2Experts {
    /// Number of routed experts.
    pub num_experts: i32,
    /// Model hidden size.
    pub hidden_size: i32,
    /// Expert intermediate size.
    pub intermediate_size: i32,
    #[param]
    /// Packed expert up-projection weights, shaped `[experts, intermediate, hidden]`.
    pub up_proj: Param<Array>,
    #[param]
    /// Packed expert down-projection weights, shaped `[experts, hidden, intermediate]`.
    pub down_proj: Param<Array>,
}

impl PackedRelu2Experts {
    /// Creates an unloaded packed expert bank.
    pub fn new(
        num_experts: i32,
        hidden_size: i32,
        intermediate_size: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            num_experts,
            hidden_size,
            intermediate_size,
            up_proj: Param::<Array>::unloaded(
                &[num_experts, intermediate_size, hidden_size],
                Dtype::Float32,
                stream,
            )?,
            down_proj: Param::<Array>::unloaded(
                &[num_experts, hidden_size, intermediate_size],
                Dtype::Float32,
                stream,
            )?,
        })
    }

    /// Evaluates routed experts and reduces route outputs back to tokens.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let up_weights = self.up_proj.as_ref().swap_axes(-1, -2, stream)?;
        let hidden = grouped_matmul(&hidden, &up_weights, &plan.sorted_group_ids, true, stream)?;
        let hidden = relu2(hidden, stream)?;
        let down_weights = self.down_proj.as_ref().swap_axes(-1, -2, stream)?;
        let current = grouped_matmul(&hidden, &down_weights, &plan.sorted_group_ids, true, stream)?;
        weighted_route_sum(current, top_k_weights, &plan, num_tokens, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

/// Builds an initialized untied language-model head.
pub fn build_lm_head(hidden_size: i32, vocab_size: i32) -> Result<nn::Linear, Exception> {
    nn::LinearBuilder::new(hidden_size, vocab_size)
        .bias(false)
        .build()
}

/// Builds an unloaded untied language-model head.
pub fn build_unloaded_lm_head(
    hidden_size: i32,
    vocab_size: i32,
    stream: &Stream,
) -> Result<nn::Linear, Exception> {
    nn::Linear::unloaded(hidden_size, vocab_size, false, Dtype::Float32, stream)
}

/// Builds an initialized language-model head wrapped for optional quantization.
pub fn build_maybe_quantized_lm_head(
    hidden_size: i32,
    vocab_size: i32,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    Ok(MaybeQuantized::Original(build_lm_head(
        hidden_size,
        vocab_size,
    )?))
}

/// Builds an unloaded language-model head wrapped for optional quantization.
pub fn build_unloaded_maybe_quantized_lm_head(
    hidden_size: i32,
    vocab_size: i32,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    unloaded_maybe_quantized_linear(hidden_size, vocab_size, false, None, stream)
}

/// Creates an unloaded linear using the standard dense or affine parameter tree.
pub fn unloaded_maybe_quantized_linear(
    input_dims: i32,
    output_dims: i32,
    bias: bool,
    quantization: Option<AffineQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    match quantization {
        Some(config) => Ok(MaybeQuantized::Quantized(nn::QuantizedLinear::unloaded(
            input_dims,
            output_dims,
            config.group_size,
            config.bits,
            bias,
            stream,
        )?)),
        None => Ok(MaybeQuantized::Original(nn::Linear::unloaded(
            input_dims,
            output_dims,
            bias,
            Dtype::Float32,
            stream,
        )?)),
    }
}

/// Creates an unloaded embedding using the standard dense or affine parameter tree.
pub fn unloaded_maybe_quantized_embedding(
    embedding_count: i32,
    dimensions: i32,
    quantization: Option<AffineQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Embedding>, Exception> {
    match quantization {
        Some(config) => Ok(MaybeQuantized::Quantized(nn::QuantizedEmbedding::unloaded(
            embedding_count,
            dimensions,
            config.group_size,
            config.bits,
            stream,
        )?)),
        None => Ok(MaybeQuantized::Original(nn::Embedding::unloaded(
            embedding_count,
            dimensions,
            Dtype::Float32,
            stream,
        )?)),
    }
}

/// Builds an unloaded language-model head with optional affine quantization.
pub fn build_unloaded_maybe_quantized_lm_head_with_quantization(
    hidden_size: i32,
    vocab_size: i32,
    quantization: Option<AffineQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    unloaded_maybe_quantized_linear(hidden_size, vocab_size, false, quantization, stream)
}

/// Projects hidden states to logits, using tied embeddings when `lm_head` is absent.
pub fn project_logits_maybe_quantized(
    lm_head: &mut Option<MaybeQuantized<nn::Linear>>,
    embed_tokens: &mut MaybeQuantized<nn::Embedding>,
    hidden_states: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states, stream),
        None => match embed_tokens {
            MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(hidden_states, stream),
            MaybeQuantized::Quantized(q_embed_tokens) => {
                q_embed_tokens.as_linear(hidden_states, stream)
            }
        },
    }
}

/// Projects hidden states to logits for dense, non-quantized heads.
pub fn project_logits_dense(
    lm_head: &mut Option<nn::Linear>,
    embed_tokens: &nn::Embedding,
    hidden_states: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states, stream),
        None => embed_tokens.as_linear(hidden_states, stream),
    }
}

/// Common attention-layer input.
pub struct AttentionInput<'a, C> {
    /// Hidden states with shape `[batch, sequence, hidden]`.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional mutable key/value cache.
    pub cache: Option<&'a mut C>,
}

/// Returns the batch size and sequence length from a hidden-state tensor.
pub fn batch_seq(x: &Array) -> (i32, i32) {
    let shape = x.shape();
    (shape[0], shape[1])
}

/// Reshapes a projected Q/K/V tensor to `[batch, heads, sequence, head_dim]`.
pub fn reshape_attention_projection(
    projection: Array,
    batch: i32,
    seq_len: i32,
    heads: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    projection
        .reshape(&[batch, seq_len, heads, -1], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)
}

/// Computes explicit attention probabilities for inspection views.
pub fn attention_probabilities(
    queries: &Array,
    keys: &Array,
    scale: f32,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let queries_shape = queries.shape();
    let keys_shape = keys.shape();
    let batch = queries_shape[0];
    let query_heads = queries_shape[1];
    let key_heads = keys_shape[1];
    let key_len = keys_shape[2];
    let head_dim = keys_shape[3];
    let keys = if query_heads == key_heads {
        keys.clone()
    } else if query_heads % key_heads == 0 {
        let repeats = query_heads / key_heads;
        broadcast_to(
            &keys.reshape(&[batch, key_heads, 1, key_len, head_dim], stream)?,
            &[batch, key_heads, repeats, key_len, head_dim],
            stream,
        )?
        .reshape(&[batch, query_heads, key_len, head_dim], stream)?
    } else {
        return Err(Exception::custom(
            "query attention heads are not divisible by key/value heads",
        ));
    };

    let mut scores = matmul(
        &queries.multiply(Array::from_f32(scale), stream)?,
        &keys.swap_axes(-1, -2, stream)?,
        stream,
    )?;
    if let Some(mask) = mask {
        if mask.dtype() == Dtype::Bool {
            let finfo_min = scores.dtype().finfo_min()?;
            scores = r#where(mask, scores, Array::from_f32(finfo_min as f32), stream)?;
        } else {
            scores = scores.add(mask, stream)?;
        }
    }
    softmax_axis(&scores, -1, true, stream)
}

/// Applies RoPE to queries and keys, then updates a cache when provided.
pub fn apply_rope_and_update_cache<C>(
    rope: &mut RopeVariant,
    mut queries: Array,
    mut keys: Array,
    mut values: Array,
    cache: &mut Option<&mut C>,
    stream: &Stream,
) -> Result<(Array, Array, Array), Exception>
where
    C: KeyValueCache,
{
    if let Some(cache) = cache.as_mut() {
        let offset = cache.offset();
        queries = rope.forward(
            nn::RopeInputBuilder::new(&queries).offset(offset).build()?,
            stream,
        )?;
        keys = rope.forward(
            nn::RopeInputBuilder::new(&keys).offset(offset).build()?,
            stream,
        )?;
        (keys, values) = cache.update_and_fetch(keys, values, stream)?;
    } else {
        queries = rope.forward(nn::RopeInput::new(&queries), stream)?;
        keys = rope.forward(nn::RopeInput::new(&keys), stream)?;
    }

    Ok((queries, keys, values))
}

/// Applies caller-provided rotary embeddings and updates a key/value cache.
///
/// This is shared by multimodal decoders whose positions are not representable
/// by a single monotonically increasing RoPE offset.
pub(crate) fn apply_rotary_embeddings_and_update_cache<C>(
    queries: Array,
    keys: Array,
    mut values: Array,
    cos: &Array,
    sin: &Array,
    cache: &mut Option<&mut C>,
    stream: &Stream,
) -> Result<(Array, Array, Array), Exception>
where
    C: KeyValueCache,
{
    let cos = cos
        .as_dtype(queries.dtype(), stream)?
        .try_index_device((.., NewAxis, .., ..), stream)?;
    let sin = sin
        .as_dtype(queries.dtype(), stream)?
        .try_index_device((.., NewAxis, .., ..), stream)?;
    let rotate_half = |x: &Array| -> Result<Array, Exception> {
        let half = x.dim(-1) / 2;
        let first = x.try_index_device((.., .., .., ..half), stream)?;
        let second = x.try_index_device((.., .., .., half..), stream)?;
        concatenate_axis(
            &[second.multiply(Array::from_f32(-1.0), stream)?, first],
            -1,
            stream,
        )
    };
    let queries = queries
        .multiply(&cos, stream)?
        .add(rotate_half(&queries)?.multiply(&sin, stream)?, stream)?;
    let mut keys = keys
        .multiply(&cos, stream)?
        .add(rotate_half(&keys)?.multiply(&sin, stream)?, stream)?;
    if let Some(cache) = cache.as_mut() {
        (keys, values) = cache.update_and_fetch(keys, values, stream)?;
    }
    Ok((queries, keys, values))
}

#[allow(clippy::too_many_arguments)]
/// Runs scaled dot-product attention and reshapes the output back to hidden states.
pub fn finish_attention<C>(
    queries: Array,
    keys: Array,
    values: Array,
    cache: Option<&mut C>,
    scale: f32,
    mask: Option<&Array>,
    batch: i32,
    seq_len: i32,
    stream: &Stream,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    scaled_dot_product_attention(queries, keys, values, cache, scale, mask, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq_len, -1], stream)
}

#[allow(clippy::too_many_arguments)]
/// Computes causal sliding-window attention without a prompt-sized square mask.
pub(crate) fn sliding_window_prefill_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    window_size: i32,
    query_position_offset: i32,
    batch: i32,
    seq_len: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if window_size <= 0 {
        return Err(Exception::custom(
            "sliding attention window must be positive",
        ));
    }
    let q_shape = queries.shape();
    let k_shape = keys.shape();
    if q_shape.len() != 4 || k_shape.len() != 4 || values.shape().len() != 4 {
        return Err(Exception::custom(
            "sliding prefill attention expects rank-4 Q/K/V",
        ));
    }
    let key_len = k_shape[2];
    if q_shape[2] != seq_len || values.shape()[2] != key_len {
        return Err(Exception::custom(
            "sliding prefill attention received inconsistent sequence lengths",
        ));
    }
    let key_position_offset = query_position_offset + seq_len - key_len;
    if key_position_offset < 0 {
        return Err(Exception::custom(
            "sliding prefill attention key origin precedes position zero",
        ));
    }

    if query_position_offset == 0 && seq_len <= window_size {
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

    let max_past = window_size - 1;
    let chunk_size = 256;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < seq_len {
        let end = (start + chunk_size).min(seq_len);
        let query_abs_start = query_position_offset + start;
        let wanted_key_start = (query_abs_start - max_past).max(key_position_offset);
        let key_start = wanted_key_start - key_position_offset;
        let key_end = query_position_offset + end - key_position_offset;
        let relative_offset = query_abs_start - wanted_key_start;
        let query_chunk = queries.try_index_device((.., .., start..end, ..), stream)?;
        let key_chunk = keys.try_index_device((.., .., key_start..key_end, ..), stream)?;
        let value_chunk = values.try_index_device((.., .., key_start..key_end, ..), stream)?;
        let mask = create_causal_mask(
            end - start,
            Some(relative_offset),
            Some(max_past),
            None,
            stream,
        )?;
        chunks.push(safemlx::fast::scaled_dot_product_attention(
            query_chunk,
            key_chunk,
            value_chunk,
            scale,
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
            stream,
        )?);
        start = end;
    }

    let refs = chunks.iter().collect::<Vec<_>>();
    concatenate_axis(&refs, 2, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq_len, -1], stream)
}

/// Samples a token id from logits.
///
/// A temperature of `0.0` uses greedy argmax; non-zero temperatures use
/// categorical sampling and require `prng_state`.
pub fn sample(
    logits: &Array,
    temp: f32,
    prng_state: Option<&mut RandomState>,
    stream: &Stream,
) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1, stream = stream),
        _ => {
            let prng_state = prng_state.ok_or_else(|| {
                Exception::custom("random operations require an explicit PRNG key")
            })?;
            let key = prng_state.next_key(stream)?;
            let logits = logits.multiply(array!(1.0 / temp), stream)?;
            random::categorical(&logits, None, None, &key, stream)
        }
    }
}

/// Minimal interface required by the generic token generator.
pub trait CausalLm<C> {
    /// Computes logits for an initial typed input and fills `cache`.
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut C,
        stream: &Stream,
    ) -> Result<Array, Exception>;

    /// Computes logits for one or more decode tokens using an existing cache.
    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut C,
        stream: &Stream,
    ) -> Result<Array, Exception>;

    /// Gives implementations a chance to adjust prefill logits before sampling.
    fn adjust_prefill_logits(
        &mut self,
        logits: Array,
        _cache: &mut C,
        _stream: &Stream,
    ) -> Result<Array, Exception> {
        Ok(logits)
    }
}

/// Current state of a generic generation iterator.
pub enum GenerateState<'a> {
    /// The iterator has not consumed the prompt yet.
    Prefill {
        /// Typed input used for the initial prefill pass.
        input: input::ModelInput<'a>,
    },
    /// The iterator is decoding from the previous sampled token.
    Decode {
        /// Previously sampled token id array.
        y: Array,
    },
}

/// Generic token iterator for a causal LM.
pub struct Generate<'a, M, C, S = DefaultSampler>
where
    M: CausalLm<C>,
    S: Sampler,
{
    model: &'a mut M,
    cache: &'a mut C,
    temp: f32,
    prng_state: Option<RandomState>,
    sampler: S,
    stream: &'a Stream,
    state: GenerateState<'a>,
    _cache: PhantomData<C>,
}

impl<'a, M, C> Generate<'a, M, C, DefaultSampler>
where
    M: CausalLm<C>,
{
    /// Creates a generation iterator over token-id arrays using the default sampler.
    pub fn new(
        model: &'a mut M,
        cache: &'a mut C,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Self {
        Self::with_sampler(model, cache, temp, input, prng_key, stream, DefaultSampler)
    }
}

impl<'a, M, C, S> Generate<'a, M, C, S>
where
    M: CausalLm<C>,
    S: Sampler,
{
    /// Creates a generation iterator over token-id arrays using a caller-provided sampler.
    pub fn with_sampler(
        model: &'a mut M,
        cache: &'a mut C,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            prng_state: prng_key.map(RandomState::from_key),
            sampler,
            stream,
            state: GenerateState::Prefill { input },
            _cache: PhantomData,
        }
    }
}

impl<M, C, S> Iterator for Generate<'_, M, C, S>
where
    M: CausalLm<C>,
    S: Sampler,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { input } => {
                let logits = match self
                    .model
                    .prefill_input_logits(*input, self.cache, self.stream)
                {
                    Ok(logits) => logits,
                    Err(err) => return Some(Err(err)),
                };
                let logits = match self
                    .model
                    .adjust_prefill_logits(logits, self.cache, self.stream)
                {
                    Ok(logits) => logits,
                    Err(err) => return Some(Err(err)),
                };
                let y = match self.sampler.sample(
                    &logits,
                    self.temp,
                    self.prng_state.as_mut(),
                    self.stream,
                ) {
                    Ok(y) => y,
                    Err(err) => return Some(Err(err)),
                };
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = match y.try_index_device((.., NewAxis), self.stream) {
                    Ok(inputs) => inputs,
                    Err(err) => return Some(Err(err)),
                };
                let logits = match self.model.decode_logits(&inputs, self.cache, self.stream) {
                    Ok(logits) => logits,
                    Err(err) => return Some(Err(err)),
                };
                let y = match self.sampler.sample(
                    &logits,
                    self.temp,
                    self.prng_state.as_mut(),
                    self.stream,
                ) {
                    Ok(y) => y,
                    Err(err) => return Some(Err(err)),
                };
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{attention_probabilities, sample, sliding_window_prefill_attention};
    use safemlx::{
        fast::ScaledDotProductAttentionMask, Array, Device, DeviceType, Dtype, ExecutionContext,
    };

    use crate::utils::create_causal_mask;

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn non_greedy_sample_requires_prng_key() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let logits = Array::from_slice(&[0.0f32, 1.0], &[1, 2]);

        let error = sample(&logits, 1.0, None, ctx.stream()).unwrap_err();

        assert!(error
            .to_string()
            .contains("random operations require an explicit PRNG key"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn bool_attention_mask_keeps_attention_probabilities_float32() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let queries = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]);
        let keys = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]);
        let mask = Array::from_slice(&[false, true, false, false], &[1, 1, 2, 2]);

        let probs = attention_probabilities(&queries, &keys, 1.0, Some(&mask), stream).unwrap();

        assert_eq!(probs.dtype(), Dtype::Float32);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn chunked_sliding_prefill_matches_full_masked_gqa_attention() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let queries = Array::from_slice(
            &[
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                0.4, 0.3, 0.2, 0.1,
            ],
            &[1, 2, 5, 2],
        );
        let keys = Array::from_slice(
            &[0.2f32, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
            &[1, 1, 5, 2],
        );
        let values = Array::from_slice(
            &[1.0f32, 0.0, 0.8, 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8],
            &[1, 1, 5, 2],
        );
        let mask = create_causal_mask(5, None, Some(2), None, stream).unwrap();
        let reference = safemlx::fast::scaled_dot_product_attention(
            queries.clone(),
            keys.clone(),
            values.clone(),
            2.0f32.sqrt().recip(),
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
            stream,
        )
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3], stream)
        .unwrap()
        .reshape(&[1, 5, 4], stream)
        .unwrap();

        let chunked = sliding_window_prefill_attention(
            queries,
            keys,
            values,
            2.0f32.sqrt().recip(),
            3,
            0,
            1,
            5,
            stream,
        )
        .unwrap();

        assert!(chunked
            .all_close(&reference, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }
}
