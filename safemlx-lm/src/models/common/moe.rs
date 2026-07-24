//! Mixture-of-experts routing and packed expert implementations.

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::Param,
    native_quantization::{native_grouped_linear, NativeQuantizedTensor},
    ops::{
        arange, argpartition_axis, concatenate_axis, gather_grouped_rows, gather_qmm_with_mode,
        gather_route_values, grouped_matmul,
        indexing::{scatter_single, take_along_axis, topk_axis, NewAxis, TryIndexOp},
        matmul, quantized_matmul_with_mode, quantized_packed_dimension, r#where, sigmoid,
        softmax_axis, sum_axis, topk_route_plan, zeros_dtype, GroupedRoutePlan, QuantizationMode,
    },
    Array, Dtype, Stream,
};

use crate::{
    error::Error,
    inspection::ActivationObserver,
    quantization::{quantize_tensor, QuantizedTensor, WeightQuantization},
};

use super::layers::{relu2, silu};

/// Applies one affine-quantized packed expert projection to expert-major rows.
///
/// The packed weight and its metadata keep the expert dimension leading, so
/// this is usable by any checkpoint layout once split experts have been
/// assembled into `[experts, output, input]` banks.
pub fn affine_grouped_linear(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<Array, Exception> {
    affine_grouped_linear_with_transpose(
        input,
        weight,
        scales,
        biases,
        group_ids,
        quantization,
        true,
        stream,
    )
}

/// Applies an affine packed grouped projection in either matrix direction.
pub fn affine_grouped_linear_with_transpose(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    transpose: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    affine_grouped_linear_with_options(
        input,
        weight,
        scales,
        biases,
        group_ids,
        quantization,
        transpose,
        true,
        stream,
    )
}

/// Applies an affine packed grouped projection with explicit route-order metadata.
#[allow(clippy::too_many_arguments)]
pub fn affine_grouped_linear_with_options(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    transpose: bool,
    sorted_indices: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    let routes = input.dim(0);
    let out_features = if transpose {
        weight.dim(-2)
    } else {
        weight.dim(-1) * 32 / quantization.bits()
    };
    if quantization.group_size() == 16 {
        if !transpose {
            return Err(Exception::custom(
                "group-16 affine expert projections require transposed packed weights",
            ));
        }
        let selected_weight = weight.take_axis(group_ids, 0, stream)?;
        let selected_scales = scales.take_axis(group_ids, 0, stream)?;
        let selected_biases = biases
            .map(|biases| biases.take_axis(group_ids, 0, stream))
            .transpose()?;
        return quantized_matmul_with_mode(
            input.reshape(&[routes, 1, input.dim(-1)], stream)?,
            &selected_weight,
            &selected_scales,
            selected_biases.as_ref(),
            true,
            quantization.group_size(),
            quantization.bits(),
            quantization.mode(),
            stream,
        )?
        .reshape(&[routes, out_features], stream);
    }

    let lhs_indices = arange::<i32, u32>(0, routes, 1, stream)?;
    gather_qmm_with_mode(
        input.reshape(&[routes, 1, input.dim(-1)], stream)?,
        weight,
        scales,
        biases,
        Some(&lhs_indices),
        Some(group_ids),
        transpose,
        quantization.group_size(),
        quantization.bits(),
        sorted_indices,
        quantization.mode(),
        stream,
    )?
    .reshape(&[routes, out_features], stream)
}

/// Quantizes a floating-point rank-3 packed expert bank while preserving its
/// leading expert dimension in the emitted weight, scale, and bias tensors.
pub fn quantize_expert_bank(
    value: &Array,
    quantization: WeightQuantization,
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
    Ok(QuantizedTensor {
        weight: quantized.weight.reshape(
            &[
                experts,
                output_dims,
                quantized_packed_dimension(input_dims, quantization.bits()),
            ],
            stream,
        )?,
        scales: quantized.scales.reshape(
            &[experts, output_dims, input_dims / quantization.group_size()],
            stream,
        )?,
        biases: quantized
            .biases
            .map(|biases| {
                biases.reshape(
                    &[experts, output_dims, input_dims / quantization.group_size()],
                    stream,
                )
            })
            .transpose()?,
    })
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
    /// Optional affine scales for a packed router projection.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Optional affine biases for a packed router projection.
    pub biases: Param<Option<Array>>,
    #[param]
    /// Optional score correction bias used only when choosing experts.
    pub e_score_correction_bias: Param<Option<Array>>,
    /// Affine group size, or zero for a dense router.
    pub group_size: i32,
    /// Affine bit width, or zero for a dense router.
    pub bits: i32,
    /// Packed quantization encoding.
    pub mode: QuantizationMode,
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

/// Selects the largest router logits and normalizes only the selected values.
/// This matches routers such as GPT-OSS where the softmax is applied after
/// top-k selection rather than across every expert.
pub fn top_k_softmax_routing(
    logits: &Array,
    top_k: i32,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let indices =
        argpartition_axis(logits, -top_k, -1, stream)?.try_index_device((.., -top_k..), stream)?;
    let selected = take_along_axis(logits, &indices, -1, stream)?;
    Ok((indices, softmax_axis(&selected, -1, true, stream)?))
}

impl TopKRouter {
    /// Creates an unloaded router.
    pub fn new(config: TopKRouterConfig, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_quantization(config, None, stream)
    }

    /// Creates an unloaded dense or affine-packed router.
    pub fn new_with_quantization(
        config: TopKRouterConfig,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if let Some(quantization) = quantization {
            if config.hidden_size <= 0 || config.hidden_size % quantization.group_size() != 0 {
                return Err(Exception::custom(format!(
                    "affine router hidden dimension {} is not divisible by group size {}",
                    config.hidden_size,
                    quantization.group_size()
                )));
            }
        }
        Ok(Self {
            top_k: config.top_k,
            num_experts: config.num_experts,
            score_function: config.score_function,
            norm_topk_prob: config.norm_topk_prob,
            normalization_epsilon: config.normalization_epsilon,
            routed_scaling_factor: config.routed_scaling_factor,
            n_group: config.n_group,
            topk_group: config.topk_group,
            weight: if let Some(quantization) = quantization {
                Param::<Array>::unloaded(
                    &[
                        config.num_experts,
                        quantized_packed_dimension(config.hidden_size, quantization.bits()),
                    ],
                    Dtype::Uint32,
                    stream,
                )?
            } else {
                Param::<Array>::unloaded(
                    &[config.num_experts, config.hidden_size],
                    Dtype::Float32,
                    stream,
                )?
            },
            scales: if let Some(quantization) = quantization {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        config.num_experts,
                        config.hidden_size / quantization.group_size(),
                    ],
                    Dtype::Float16,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            biases: if let Some(quantization) = quantization.filter(|q| q.has_biases()) {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        config.num_experts,
                        config.hidden_size / quantization.group_size(),
                    ],
                    Dtype::Float16,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            e_score_correction_bias: if config.score_correction_bias {
                Param::<Option<Array>>::unloaded_some(
                    &[config.num_experts],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            group_size: quantization.map_or(0, WeightQuantization::group_size),
            bits: quantization.map_or(0, WeightQuantization::bits),
            mode: quantization.map_or(QuantizationMode::Affine, WeightQuantization::mode),
        })
    }

    /// Returns selected expert ids and per-route weights.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        self.forward_with_selection_bias(hidden_states, None, stream)
    }

    /// Returns selected expert ids and weights using an optional bias only for selection.
    ///
    /// The gathered route weights always come from the unbiased transformed scores.
    pub fn forward_with_selection_bias(
        &mut self,
        hidden_states: &Array,
        selection_bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let flat = hidden_states.reshape(&[-1, hidden_states.dim(-1)], stream)?;
        let logits = if let Some(scales) = self.scales.as_ref() {
            let input = if self.score_function == TopKRouterScoreFunction::Sigmoid {
                flat.as_dtype(Dtype::Float32, stream)?
            } else {
                flat
            };
            quantized_matmul_with_mode(
                &input,
                self.weight.as_ref(),
                scales,
                self.biases.as_ref().as_ref(),
                true,
                self.group_size,
                self.bits,
                self.mode,
                stream,
            )?
        } else if self.score_function == TopKRouterScoreFunction::Sigmoid {
            matmul(
                &flat.as_dtype(Dtype::Float32, stream)?,
                &self
                    .weight
                    .as_ref()
                    .as_dtype(Dtype::Float32, stream)?
                    .transpose(stream)?,
                stream,
            )?
        } else {
            matmul(&flat, self.weight.as_ref().transpose(stream)?, stream)?
        };
        let scores = match self.score_function {
            TopKRouterScoreFunction::Softmax => softmax_axis(&logits, -1, true, stream)?,
            TopKRouterScoreFunction::Sigmoid => sigmoid(logits, stream)?,
        };
        let mut scores_for_choice = scores.clone();
        if let Some(bias) = self.e_score_correction_bias.as_ref() {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
        }
        if let Some(bias) = selection_bias {
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
        observer: &mut dyn ActivationObserver,
    ) -> Result<TopKRouterOutput, Exception> {
        let flat = hidden_states.reshape(&[-1, hidden_states.dim(-1)], stream)?;
        let logits = if self.score_function == TopKRouterScoreFunction::Sigmoid {
            matmul(
                &flat.as_dtype(Dtype::Float32, stream)?,
                &self
                    .weight
                    .as_ref()
                    .as_dtype(Dtype::Float32, stream)?
                    .transpose(stream)?,
                stream,
            )?
        } else {
            matmul(&flat, self.weight.as_ref().transpose(stream)?, stream)?
        };
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
        let masked_scores = r#where(
            &group_mask,
            scores_for_choice,
            Array::from_f32(f32::NEG_INFINITY),
            stream,
        )?;
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

    // Each route index is unique, so restore the expert-major rows with a
    // collision-free scatter and reduce the original top-k slots in their
    // stable order. A segment sum can use unordered GPU atomics here; the
    // resulting roundoff was sufficient to change near-tied downstream routing
    // decisions between identical passes.
    let routes = weighted.dim(0);
    let width = weighted.dim(-1);
    let ordered = scatter_single(
        zeros_dtype(&[routes, width], weighted.dtype(), stream)?,
        &plan.route_indices,
        weighted.reshape(&[routes, 1, width], stream)?,
        0,
        stream,
    )?;
    let top_k = top_k_weights.dim(-1);
    let ordered = ordered.reshape(&[num_tokens, top_k, width], stream)?;
    sum_axis(ordered, 1, false, stream)
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

#[cfg(test)]
mod tests {
    use super::weighted_route_sum;
    use safemlx::{ops::topk_route_plan, Array, Device, DeviceType, ExecutionContext};

    #[test]
    fn weighted_route_sum_restores_original_topk_order() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let expert_ids = Array::from_slice(&[2i32, 0, 1, 2, 0, 1], &[3, 2]);
        let plan = topk_route_plan(&expert_ids, 3, stream).unwrap();

        // The plan orders original routes [1, 4, 2, 5, 0, 3] by expert id.
        let expert_major = Array::from_slice(&[2.0f32, 5.0, 3.0, 6.0, 1.0, 4.0], &[6, 1]);
        let weights = Array::ones::<f32>(&[3, 2], stream).unwrap();
        let reduced = weighted_route_sum(expert_major, &weights, &plan, 3, stream).unwrap();

        assert_eq!(
            reduced.evaluated().unwrap().as_slice::<f32>(),
            &[3.0, 7.0, 11.0]
        );
    }
}

const ROUTED_EXPERT_CHUNK_THRESHOLD: i32 = 64;
const ROUTED_EXPERT_CHUNK_TOKENS: i32 = 32;

#[derive(Debug, Clone, ModuleParameters)]
/// Packed SwiGLU expert bank with optional MLX affine or MXFP4 projections.
pub struct PackedSwiGluExperts {
    /// Number of experts.
    pub num_experts: i32,
    /// Model hidden dimension.
    pub hidden_dim: i32,
    /// Per-expert intermediate dimension.
    pub intermediate_dim: i32,
    /// Optional encoding for the concatenated gate/up projection.
    pub gate_up_affine: Option<WeightQuantization>,
    /// Optional encoding for the down projection.
    pub down_affine: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the gate/up projection.
    pub gate_up_iquant: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the down projection.
    pub down_iquant: Option<WeightQuantization>,
    #[param]
    /// Concatenated gate/up weights shaped `[experts, 2 * intermediate, hidden]`.
    pub gate_up_proj: Param<Array>,
    #[param]
    /// Gate/up quantization scales.
    pub gate_up_proj_scales: Param<Option<Array>>,
    #[param]
    /// Gate/up quantization biases.
    pub gate_up_proj_biases: Param<Option<Array>>,
    #[param]
    /// Down weights shaped `[experts, hidden, intermediate]`.
    pub down_proj: Param<Array>,
    #[param]
    /// Down quantization scales.
    pub down_proj_scales: Param<Option<Array>>,
    #[param]
    /// Down quantization biases.
    pub down_proj_biases: Param<Option<Array>>,
}

type ExpertProjectionParams = (Param<Array>, Param<Option<Array>>, Param<Option<Array>>);

impl PackedSwiGluExperts {
    /// Creates an unloaded packed expert bank.
    pub fn new(
        num_experts: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        gate_up_affine: Option<WeightQuantization>,
        down_affine: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (gate_up_affine, gate_up_iquant) = match gate_up_affine {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
            affine => (affine, None),
        };
        let (down_affine, down_iquant) = match down_affine {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
            affine => (affine, None),
        };
        let projection = |out_features: i32,
                          in_features: i32,
                          quantization: Option<WeightQuantization>,
                          iquant: Option<WeightQuantization>|
         -> Result<ExpertProjectionParams, Exception> {
            if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ expert format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            in_features / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ))
            } else if let Some(quantization) = quantization {
                Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            quantized_packed_dimension(in_features, quantization.bits()),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            num_experts,
                            out_features,
                            in_features / quantization.group_size(),
                        ],
                        if quantization == WeightQuantization::MxFp4 {
                            Dtype::Uint8
                        } else {
                            Dtype::Float16
                        },
                        stream,
                    )?,
                    if quantization.has_biases() {
                        Param::<Option<Array>>::unloaded_some(
                            &[
                                num_experts,
                                out_features,
                                in_features / quantization.group_size(),
                            ],
                            Dtype::Float16,
                            stream,
                        )?
                    } else {
                        Param::new(None)
                    },
                ))
            } else {
                Ok((
                    Param::<Array>::unloaded(
                        &[num_experts, out_features, in_features],
                        Dtype::Float32,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ))
            }
        };
        let (gate_up_proj, gate_up_proj_scales, gate_up_proj_biases) = projection(
            2 * intermediate_dim,
            hidden_dim,
            gate_up_affine,
            gate_up_iquant,
        )?;
        let (down_proj, down_proj_scales, down_proj_biases) =
            projection(hidden_dim, intermediate_dim, down_affine, down_iquant)?;
        Ok(Self {
            num_experts,
            hidden_dim,
            intermediate_dim,
            gate_up_affine,
            down_affine,
            gate_up_iquant,
            down_iquant,
            gate_up_proj,
            gate_up_proj_scales,
            gate_up_proj_biases,
            down_proj,
            down_proj_scales,
            down_proj_biases,
        })
    }

    fn forward_chunk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let gate_up = if let Some(iquant) = self.gate_up_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.gate_up_proj.value.clone(),
                &[self.num_experts, 2 * self.intermediate_dim, self.hidden_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(quantization) = self.gate_up_affine {
            affine_grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                self.gate_up_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("quantized gate/up scales"),
                self.gate_up_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                quantization,
                stream,
            )?
        } else {
            grouped_matmul(
                &hidden,
                &self.gate_up_proj.as_ref().swap_axes(-1, -2, stream)?,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        let gate = gate_up.try_index_device((.., ..self.intermediate_dim), stream)?;
        let up = gate_up.try_index_device((.., self.intermediate_dim..), stream)?;
        let activated = silu(gate, stream)?.multiply(up, stream)?;
        let output = if let Some(iquant) = self.down_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.num_experts, self.hidden_dim, self.intermediate_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&activated, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(quantization) = self.down_affine {
            affine_grouped_linear(
                &activated,
                self.down_proj.as_ref(),
                self.down_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("quantized down scales"),
                self.down_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                quantization,
                stream,
            )?
        } else {
            grouped_matmul(
                &activated,
                &self.down_proj.as_ref().swap_axes(-1, -2, stream)?,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        weighted_route_sum(output, top_k_weights, &plan, num_tokens, stream)
    }

    /// Evaluates selected experts and reduces route outputs back to source tokens.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        if num_tokens <= ROUTED_EXPERT_CHUNK_THRESHOLD {
            return self.forward_chunk(hidden_states, top_k_index, top_k_weights, stream);
        }
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < num_tokens {
            let end = (start + ROUTED_EXPERT_CHUNK_TOKENS).min(num_tokens);
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

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}
