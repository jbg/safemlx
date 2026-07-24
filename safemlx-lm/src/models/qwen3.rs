//! Qwen3 decoder-only model implementation.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt},
    nn,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use super::common::generation::sample;

use crate::{
    cache::KeyValueCache,
    error::Error,
    inspection::{ActivationObserver, MoeRoutingObservation},
    models::{
        common::{
            self,
            attention::{
                apply_rope_and_update_cache, apply_rotary_embeddings_and_update_cache,
                attention_probabilities, batch_seq, finish_attention, reshape_attention_projection,
                AttentionInput,
            },
            generation::CausalLm,
            layers::SwiGluMlp,
            linear::project_logits_maybe_quantized,
            moe::TopKRouterScoreFunction,
        },
        input,
    },
    quantization::{AffineQuantization, WeightQuantization},
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
    weights::{
        gguf_affine_configs, gguf_metadata, load_gguf_strict, load_named_array_strict,
        load_safetensors_dir_lenient, load_safetensors_dir_quantized_strict, GgufTensorNames,
        StrictLoadConfig, StrictLoadReport,
    },
};

#[derive(Debug, Clone, Deserialize)]
/// Deserialized Qwen3 `config.json` fields used by this loader.
pub struct ModelArgs {
    /// Model type from the configuration.
    pub model_type: String,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Intermediate size for the SwiGLU MLP.
    pub intermediate_size: i32,
    /// Number of query attention heads.
    pub num_attention_heads: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Per-head attention dimension.
    pub head_dim: i32,
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    /// Optional RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Preferred MLX-LM affine quantization metadata.
    #[serde(default)]
    pub quantization: Option<WeightQuantization>,
    /// Hugging Face-compatible alias emitted by MLX-LM converters.
    #[serde(default)]
    pub quantization_config: Option<WeightQuantization>,
    /// Optional exact weight names that use affine quantization.
    ///
    /// `None` preserves MLX-LM's model-wide quantization behavior. GGUF
    /// loading uses `Some` for checkpoints mixing packed and dense matrices.
    #[serde(skip)]
    pub quantized_weights: Option<HashSet<String>>,
    /// Routed-expert intermediate size for Qwen3 MoE checkpoints.
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Number of routed experts. Zero for dense Qwen3 checkpoints.
    #[serde(default)]
    pub num_experts: i32,
    /// Number of experts selected per token.
    #[serde(default)]
    pub num_experts_per_tok: i32,
    /// Whether selected routing probabilities are normalized.
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Per-weight affine settings for mixed GGUF Q2/Q3/Q4/Q5/Q6/Q8 tensors.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, AffineQuantization>>,
}

impl ModelArgs {
    fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization.or(self.quantization_config)
    }

    pub(crate) fn weight_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        if let Some(config) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name))
        {
            return Some((*config).into());
        }
        let quantization = self.weight_quantization()?;
        match &self.quantized_weights {
            Some(names) if !names.contains(weight_name) => None,
            _ => Some(quantization),
        }
    }

    pub(crate) fn is_moe(&self) -> bool {
        self.num_experts > 0
    }
}

fn quantization_for(
    args: &ModelArgs,
    prefix: Option<&str>,
    parameter: &str,
) -> Option<WeightQuantization> {
    match prefix {
        Some(prefix) => args.weight_quantization_for(&format!("{prefix}.{parameter}.weight")),
        None => args.weight_quantization(),
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Qwen3 attention layer.
pub struct Attention {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads.
    pub n_kv_heads: i32,
    /// Attention scaling factor.
    pub scale: f32,

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
    /// Value projection.
    pub v_proj: MaybeQuantized<nn::Linear>,
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
    /// Creates an unloaded attention layer from model arguments.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_prefix(args, None, stream)
    }

    pub(crate) fn new_for_layer(
        args: &ModelArgs,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_prefix(
            args,
            Some(format!("model.layers.{layer_index}.self_attn")),
            stream,
        )
    }

    fn new_with_prefix(
        args: &ModelArgs,
        prefix: Option<String>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_heads * head_dim,
            false,
            quantization_for(args, prefix.as_deref(), "q_proj"),
            stream,
        )?;
        let k_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            false,
            quantization_for(args, prefix.as_deref(), "k_proj"),
            stream,
        )?;
        let v_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            false,
            quantization_for(args, prefix.as_deref(), "v_proj"),
            stream,
        )?;
        let o_proj = common::linear::unloaded_maybe_quantized_linear(
            n_heads * head_dim,
            dim,
            false,
            quantization_for(args, prefix.as_deref(), "o_proj"),
            stream,
        )?;

        let q_norm = nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?;
        let k_norm = nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?;

        let rope = initialize_rope(
            head_dim,
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
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
        })
    }

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
        let AttentionInput { x, mask, mut cache } = input;

        let (batch, seq_len) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.q_proj"), &queries)?;
        let keys = self.k_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.k_proj"), &keys)?;
        let values = self.v_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.v_proj"), &values)?;

        let queries = self.q_norm.forward(
            &reshape_attention_projection(queries, batch, seq_len, self.n_heads, stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.q_norm"), &queries)?;
        let keys = self.k_norm.forward(
            &reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.k_norm"), &keys)?;
        let values = reshape_attention_projection(values, batch, seq_len, self.n_kv_heads, stream)?;
        observer.observe(&format!("{prefix}.values"), &values)?;

        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        observer.observe(&format!("{prefix}.queries_rope"), &queries)?;
        observer.observe(&format!("{prefix}.keys_rope"), &keys)?;
        observer.observe(&format!("{prefix}.values_cache"), &values)?;
        let attention_probs = attention_probabilities(&queries, &keys, self.scale, mask, stream)?;
        observer.observe(&format!("{prefix}.attention_probs"), &attention_probs)?;

        let output = finish_attention(
            queries, keys, values, cache, self.scale, mask, batch, seq_len, stream,
        )?;
        observer.observe(&format!("{prefix}.attention"), &output)?;

        let output = self.o_proj.forward(&output, stream)?;
        observer.observe(&format!("{prefix}.o_proj"), &output)?;
        Ok(output)
    }

    pub(crate) fn forward_with_rotary_embeddings<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, mut cache } = input;
        let (batch, seq_len) = batch_seq(x);
        let queries = self.q_norm.forward(
            &reshape_attention_projection(
                self.q_proj.forward(x, stream)?,
                batch,
                seq_len,
                self.n_heads,
                stream,
            )?,
            stream,
        )?;
        let keys = self.k_norm.forward(
            &reshape_attention_projection(
                self.k_proj.forward(x, stream)?,
                batch,
                seq_len,
                self.n_kv_heads,
                stream,
            )?,
            stream,
        )?;
        let values = reshape_attention_projection(
            self.v_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let (queries, keys, values) = apply_rotary_embeddings_and_update_cache(
            queries, keys, values, cos, sin, &mut cache, stream,
        )?;
        let output = finish_attention(
            queries, keys, values, cache, self.scale, mask, batch, seq_len, stream,
        )?;
        self.o_proj.forward(&output, stream)
    }
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
        let AttentionInput { x, mask, mut cache } = input;

        let (B, L) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        let keys = self.k_proj.forward(x, stream)?;
        let values = self.v_proj.forward(x, stream)?;

        let queries = self.q_norm.forward(
            &reshape_attention_projection(queries, B, L, self.n_heads, stream)?,
            stream,
        )?;
        let keys = self.k_norm.forward(
            &reshape_attention_projection(keys, B, L, self.n_kv_heads, stream)?,
            stream,
        )?;
        let values = reshape_attention_projection(values, B, L, self.n_kv_heads, stream)?;
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        let output =
            finish_attention(queries, keys, values, cache, self.scale, mask, B, L, stream)?;

        self.o_proj.forward(&output, stream)
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

/// Qwen3 feed-forward block.
pub type Mlp = SwiGluMlp;

/// Packed routed-expert bank shared with other SwiGLU MoE architectures.
pub type Experts = common::moe::PackedSwiGluExperts;

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3 sparse MoE feed-forward block.
pub struct SparseMoeBlock {
    #[param]
    /// Top-k router.
    pub gate: common::moe::TopKRouter,
    #[param]
    /// Routed expert bank.
    pub experts: Experts,
}

impl SparseMoeBlock {
    fn new(args: &ModelArgs, layer_index: i32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            gate: common::moe::TopKRouter::new(
                common::moe::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.num_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Softmax,
                    norm_topk_prob: args.norm_topk_prob,
                    normalization_epsilon: 0.0,
                    routed_scaling_factor: 1.0,
                    n_group: 1,
                    topk_group: 1,
                    score_correction_bias: false,
                },
                stream,
            )?,
            experts: Experts::new(
                args.num_experts,
                args.hidden_size,
                args.moe_intermediate_size,
                args.weight_quantization_for(&format!(
                    "model.layers.{layer_index}.mlp.experts.gate_up_proj"
                )),
                args.weight_quantization_for(&format!(
                    "model.layers.{layer_index}.mlp.experts.down_proj"
                )),
                stream,
            )?,
        })
    }

    fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let routing =
            self.gate
                .forward_with_observer(&flat, stream, &format!("{prefix}.gate"), observer)?;
        let output = self
            .experts
            .forward(&flat, &routing.indices, &routing.weights, stream)?;
        observer.observe(&format!("{prefix}.experts.output"), &output)?;
        observer.observe_moe_routing(MoeRoutingObservation {
            prefix,
            selected_experts: &routing.indices,
            selected_scores: &routing.scores,
            routing_weights: &routing.weights,
            routed_output: &output,
            local_routed_output: None,
            reduced_routed_output: Some(&output),
            shared_output: None,
            combined_output: Some(&output),
            num_experts: self.gate.num_experts,
        })?;
        output.reshape(shape, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        hidden_states: &Array,
        assignment: &crate::expert_parallel::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::expert_parallel::RoutingStatistics,
        prefix: &str,
        mut observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        crate::expert_parallel::materialize_timing_phase([&flat])?;
        let moe_started = std::time::Instant::now();
        let previous_moe_time = statistics.total_time;
        let router_started = std::time::Instant::now();
        let (indices, selected_scores, weights) = if let Some(observer) = observer.as_deref_mut() {
            let routing = self.gate.forward_with_observer(
                &flat,
                stream,
                &format!("{prefix}.gate"),
                observer,
            )?;
            (routing.indices, Some(routing.scores), routing.weights)
        } else {
            let (indices, weights) = self.gate.forward(&flat, stream)?;
            (indices, None, weights)
        };
        let mut router_outputs = vec![&indices, &weights];
        if let Some(scores) = selected_scores.as_ref() {
            router_outputs.push(scores);
        }
        crate::expert_parallel::materialize_timing_phase(router_outputs)?;
        statistics.router_time += router_started.elapsed();
        let returned = crate::expert_parallel::dispatch_replicated(
            &flat,
            &indices,
            &weights,
            assignment,
            &mut self.experts,
            group,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        if let Some(observer) = observer {
            observer.observe(
                &format!("{prefix}.experts.local_output"),
                &returned.local_output,
            )?;
            observer.observe(
                &format!("{prefix}.experts.reduced_output"),
                &returned.reduced_output,
            )?;
            observer.observe_moe_routing(MoeRoutingObservation {
                prefix,
                selected_experts: &indices,
                selected_scores: selected_scores
                    .as_ref()
                    .expect("observed EP routing scores initialized"),
                routing_weights: &weights,
                routed_output: &returned.reduced_output,
                local_routed_output: Some(&returned.local_output),
                reduced_routed_output: Some(&returned.reduced_output),
                shared_output: None,
                combined_output: Some(&returned.reduced_output),
                num_experts: self.gate.num_experts,
            })?;
        }
        let output = returned.reduced_output.reshape(shape, stream)?;
        crate::expert_parallel::materialize_timing_phase([&output])?;
        statistics.total_time = previous_moe_time + moe_started.elapsed();
        Ok(output)
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = input.shape();
        let flat = input.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        self.experts
            .forward(&flat, &indices, &weights, stream)?
            .reshape(shape, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
        self.experts.training_mode(mode);
    }
}

#[derive(Debug, Clone)]
/// Dense or sparse Qwen3 feed-forward layer stored under the checkpoint-native `mlp` namespace.
pub enum FeedForward {
    /// Dense SwiGLU MLP.
    Dense(Mlp),
    /// Sparse mixture-of-experts block.
    Moe(SparseMoeBlock),
}

impl FeedForward {
    fn new(args: &ModelArgs, layer_index: i32, stream: &Stream) -> Result<Self, Exception> {
        if args.is_moe() {
            Ok(Self::Moe(SparseMoeBlock::new(args, layer_index, stream)?))
        } else {
            let prefix = format!("model.layers.{layer_index}.mlp");
            Ok(Self::Dense(SwiGluMlp {
                gate_proj: common::linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.intermediate_size,
                    false,
                    args.weight_quantization_for(&format!("{prefix}.gate_proj.weight")),
                    stream,
                )?,
                down_proj: common::linear::unloaded_maybe_quantized_linear(
                    args.intermediate_size,
                    args.hidden_size,
                    false,
                    args.weight_quantization_for(&format!("{prefix}.down_proj.weight")),
                    stream,
                )?,
                up_proj: common::linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.intermediate_size,
                    false,
                    args.weight_quantization_for(&format!("{prefix}.up_proj.weight")),
                    stream,
                )?,
            }))
        }
    }

    fn is_moe(&self) -> bool {
        matches!(self, Self::Moe(_))
    }

    fn forward_with_observer(
        &mut self,
        input: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward_with_observer(input, stream, prefix, observer),
            Self::Moe(moe) => moe.forward_with_observer(input, stream, prefix, observer),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        hidden_states: &Array,
        assignment: &crate::expert_parallel::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::expert_parallel::RoutingStatistics,
        prefix: &str,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden_states, stream),
            Self::Moe(moe) => moe.forward_expert_parallel(
                hidden_states,
                assignment,
                group,
                statistics,
                prefix,
                observer,
                stream,
            ),
        }
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        match self {
            Self::Dense(mlp) => mlp.forward(input, stream),
            Self::Moe(moe) => moe.forward(input, stream),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            Self::Dense(mlp) => mlp.training_mode(mode),
            Self::Moe(moe) => moe.training_mode(mode),
        }
    }
}

impl ModuleParametersTrait for FeedForward {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.num_parameters(),
            Self::Moe(moe) => moe.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters(),
            Self::Moe(moe) => moe.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters_mut(),
            Self::Moe(moe) => moe.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.trainable_parameters(),
            Self::Moe(moe) => moe.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.freeze_parameters(recursive),
            Self::Moe(moe) => moe.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.unfreeze_parameters(recursive),
            Self::Moe(moe) => moe.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.all_frozen(),
            Self::Moe(moe) => moe.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.any_frozen(),
            Self::Moe(moe) => moe.any_frozen(),
        }
    }
}

impl safemlx::quantization::Quantizable for FeedForward {
    type Quantized = Self;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        match self {
            Self::Dense(mlp) => Ok(Self::Dense(
                safemlx::quantization::Quantizable::try_into_quantized(
                    mlp, group_size, bits, stream,
                )?,
            )),
            Self::Moe(moe) => Ok(Self::Moe(moe)),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Qwen3 decoder block.
pub struct TransformerBlock {
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Transformer hidden size.
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    /// Self-attention layer.
    pub self_attn: Attention,

    #[quantizable]
    #[param]
    /// Dense or sparse feed-forward layer.
    pub mlp: FeedForward,

    #[param]
    /// Pre-attention RMSNorm.
    pub input_layernorm: nn::RmsNorm,

    #[param]
    /// Pre-MLP RMSNorm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    /// Creates an unloaded decoder block from model arguments.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_for_layer(args, 0, stream)
    }

    pub(crate) fn new_for_layer(
        args: &ModelArgs,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Attention::new_for_layer(args, layer_index, stream)?;
        let mlp = FeedForward::new(args, layer_index, stream)?;
        let input_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let post_attention_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;

        Ok(Self {
            num_attention_heads,
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
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
        let AttentionInput { x, mask, cache } = input;

        observer.observe(&format!("{prefix}.input"), x)?;
        observer.observe(&format!("{prefix}.residual_before_attention"), x)?;
        let normed = self.input_layernorm.forward(x, stream)?;
        observer.observe(&format!("{prefix}.input_layernorm"), &normed)?;

        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
        };
        let r = self.self_attn.forward_with_observer(
            self_attn_input,
            stream,
            &format!("{prefix}.self_attn"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.self_attn_output"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_attention"), &r)?;
        let h = x.add(r, stream)?;
        observer.observe(&format!("{prefix}.post_attention_residual"), &h)?;
        observer.observe(&format!("{prefix}.residual_after_attention"), &h)?;

        let feed_forward_name = if self.mlp.is_moe() { "moe" } else { "mlp" };
        observer.observe(&format!("{prefix}.residual_before_{feed_forward_name}"), &h)?;
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &post_normed)?;
        let r = self.mlp.forward_with_observer(
            &post_normed,
            stream,
            &format!("{prefix}.mlp"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.{feed_forward_name}_output"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_{feed_forward_name}"), &r)?;
        let output = h.add(r, stream)?;
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

    pub(crate) fn forward_with_rotary_embeddings<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward_with_rotary_embeddings(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            cos,
            sin,
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = self.mlp.forward(&normed, stream)?;
        hidden.add(mlp, stream)
    }

    /// Executes a block while delegating routed-expert evaluation to a compact bank.
    pub(crate) fn forward_sparse_experts<C, F>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normed, stream)?,
            FeedForward::Moe(moe) => {
                let shape = normed.shape();
                let flat = normed.reshape(&[-1, normed.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    pub(crate) fn forward_sparse_experts_with_rotary<C, F>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward_with_rotary_embeddings(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            cos,
            sin,
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normed, stream)?,
            FeedForward::Moe(moe) => {
                let shape = normed.shape();
                let flat = normed.reshape(&[-1, normed.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        assignment: &crate::expert_parallel::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::expert_parallel::RoutingStatistics,
        prefix: &str,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = self.mlp.forward_expert_parallel(
            &normed,
            assignment,
            group,
            statistics,
            &format!("{prefix}.mlp"),
            observer,
            stream,
        )?;
        hidden.add(mlp, stream)
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
        let AttentionInput { x, mask, cache } = input;

        let normed = self.input_layernorm.forward(x, stream)?;
        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
        };
        let r = self.self_attn.forward(self_attn_input, stream)?;
        let h = x.add(r, stream)?;

        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        let r = self.mlp.forward(&post_normed, stream)?;
        h.add(r, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Qwen3 transformer body without the language-model head.
pub struct Qwen3Model {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    /// Token embedding table.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    /// Decoder blocks.
    pub layers: Vec<TransformerBlock>,

    #[param]
    /// Final RMSNorm.
    pub norm: nn::RmsNorm,
}

impl Qwen3Model {
    /// Creates an unloaded Qwen3 transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let layers = (0..num_hidden_layers)
            .map(|layer_index| TransformerBlock::new_for_layer(args, layer_index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens,
            layers,
            norm,
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
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs, stream)?;
        observer.observe("model.embed_tokens", &h)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true), stream)? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"));
                }
                None => None,
            },
        };
        if let Some(mask) = mask.as_ref() {
            observer.observe("model.attention_mask", mask)?;
        }

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (i, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward_with_observer(
                layer_input,
                stream,
                &format!("model.layers.{i}"),
                observer,
            )?;
        }

        let output = self.norm.forward(&h, stream)?;
        observer.observe("model.norm", &output)?;
        Ok(output)
    }

    pub(crate) fn forward_expert_parallel<C>(
        &mut self,
        input: ModelInput<'_, C>,
        assignment: &crate::expert_parallel::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::expert_parallel::RoutingStatistics,
        mut observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;
        let mut hidden = self.embed_tokens.forward(inputs, stream)?;
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&hidden, cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => unreachable!("array mask requested"),
                None => None,
            },
        };
        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        for (index, (layer, cache)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_observer = observer
                .as_mut()
                .map(|observer| &mut **observer as &mut dyn ActivationObserver);
            hidden = layer.forward_expert_parallel(
                AttentionInput {
                    x: &hidden,
                    mask: mask.as_ref(),
                    cache: cache.as_mut(),
                },
                assignment,
                group,
                statistics,
                &format!("model.layers.{index}"),
                layer_observer,
                stream,
            )?;
        }
        self.norm.forward(&hidden, stream)
    }
}

/// Input for a Qwen3 forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Mutable per-layer key/value cache.
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Qwen3Model
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
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs, stream)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true), stream)? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"));
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input, stream)?;
        }

        self.norm.forward(&h, stream)
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
/// Qwen3 causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: Qwen3Model,

    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded Qwen3 causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let model = Qwen3Model::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )?,
            )
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
        let out = self.model.forward_with_observer(input, stream, observer)?;
        observer.observe("model.output", &out)?;
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &out,
            stream,
        )?;
        observer.observe("lm_head.logits", &logits)?;
        Ok(logits)
    }

    pub(crate) fn forward_expert_parallel<C>(
        &mut self,
        input: ModelInput<'_, C>,
        assignment: &crate::expert_parallel::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::expert_parallel::RoutingStatistics,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let hidden = self
            .model
            .forward_expert_parallel(input, assignment, group, statistics, observer, stream)?;
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )
    }

    /// Runs pure expert parallelism with externally supplied cache-backed experts.
    pub(crate) fn forward_cached_expert_parallel<C, F>(
        &mut self,
        input: ModelInput<'_, C>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;
        let mut hidden = self.model.embed_tokens.forward(inputs, stream)?;
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&hidden, cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => unreachable!("array mask requested"),
                None => None,
            },
        };
        if cache.is_empty() {
            *cache = (0..self.model.layers.len())
                .map(|_| Some(C::default()))
                .collect();
        }
        for (index, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(cache.iter_mut())
            .enumerate()
        {
            hidden = layer.forward_sparse_experts(
                AttentionInput {
                    x: &hidden,
                    mask: mask.as_ref(),
                    cache: layer_cache.as_mut(),
                },
                stream,
                |flat, indices, weights, stream| execute(index, flat, indices, weights, stream),
            )?;
        }
        let hidden = self.model.norm.forward(&hidden, stream)?;
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )
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
        let out = self.model.forward(input, stream)?;
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &out,
            stream,
        )
    }

    fn training_mode(&mut self, mode: bool) {
        <Qwen3Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

/// Loads `tokenizer.json` from a Qwen3 model directory.
pub fn load_qwen3_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads Qwen3 model arguments from `config.json`.
pub fn get_qwen3_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

pub(crate) struct LoadedQwen3Gguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads a Qwen3 GGUF checkpoint.
///
/// Dense tensors and GGUF Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0 tensors are
/// supported. The quantized formats are consumed in the packed affine
/// representation emitted by MLX's GGUF loader.
pub fn load_qwen3_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    Ok(load_qwen3_gguf_with_metadata(gguf_file, stream, weights_stream)?.model)
}

pub(crate) fn load_qwen3_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedQwen3Gguf, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    load_qwen3_gguf_checkpoint(&checkpoint, metadata, None, stream, weights_stream)
}

pub(crate) fn load_qwen3_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedQwen3Gguf, Error> {
    let architecture = gguf_string(&metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "qwen3" | "qwen3moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports qwen3 and qwen3moe"
        )));
    }
    let is_moe = architecture == "qwen3moe";
    let translate = |name: &str| translate_qwen3_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut args =
        qwen3_args_from_gguf(checkpoint, &metadata, &architecture, is_moe, weights_stream)?;
    let mut configs = gguf_affine_configs(checkpoint, translate)?;
    if is_moe {
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.mlp.experts");
            if let Some(config) = configs.remove(&format!("{prefix}.gate_proj")) {
                configs.remove(&format!("{prefix}.up_proj"));
                configs.insert(format!("{prefix}.gate_up_proj"), config);
            }
        }
    }
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    args.quantization = None;
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
    }

    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    if !is_moe {
        load_gguf_strict(
            &mut model,
            checkpoint,
            quantization.map(|value| (value, stream)),
            &config,
            &mut report,
            |name, value| Ok((translate_gguf_weight_name(&name), value)),
        )?;
    } else {
        let mut materializer = checkpoint.materializer();
        for tensor in checkpoint.catalog().tensors() {
            let physical_name = &tensor.descriptor().name;
            if physical_name.contains("ffn_gate_exps") || physical_name.contains("ffn_up_exps") {
                continue;
            }
            for (name, value) in materializer.converted_tensor(physical_name)?.into_arrays() {
                load_named_array_strict(
                    &mut model,
                    translate_qwen3_gguf_weight_name(&name, true),
                    value,
                    quantization.map(|value| (value, stream)),
                    &config,
                    &mut report,
                )?;
            }
        }
        for layer in 0..model.args.num_hidden_layers {
            let source_prefix = format!("blk.{layer}");
            let target_prefix = format!("model.layers.{layer}.mlp.experts");
            let gate = materializer
                .converted_tensor(&format!("{source_prefix}.ffn_gate_exps.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let up = materializer
                .converted_tensor(&format!("{source_prefix}.ffn_up_exps.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            for (source_suffix, target_suffix) in
                [("weight", ""), ("scales", "_scales"), ("biases", "_biases")]
            {
                let gate_name = format!("{source_prefix}.ffn_gate_exps.{source_suffix}");
                let up_name = format!("{source_prefix}.ffn_up_exps.{source_suffix}");
                match (gate.get(&gate_name), up.get(&up_name)) {
                    (Some(gate), Some(up)) => {
                        let value = concatenate_axis(&[gate.clone(), up.clone()], 1, weights_stream)?;
                        load_named_array_strict(
                            &mut model,
                            format!("{target_prefix}.gate_up_proj{target_suffix}"),
                            value,
                            quantization.map(|value| (value, stream)),
                            &config,
                            &mut report,
                        )?;
                    }
                    (None, None) if source_suffix != "weight" => {}
                    _ => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Qwen3 MoE GGUF has incomplete gate/up expert tensors under {source_prefix}"
                        )))
                    }
                }
            }
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    let eos_token_ids =
        gguf_optional_i64(&metadata, "tokenizer.ggml.eos_token_id", weights_stream)?
            .and_then(|value| u32::try_from(value).ok())
            .into_iter()
            .collect();
    Ok(LoadedQwen3Gguf {
        model,
        eos_token_ids,
    })
}

pub(crate) fn prepare_qwen3_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    is_moe: bool,
    weights_stream: &Stream,
) -> Result<(ModelArgs, Vec<u32>), Error> {
    let translate = |name: &str| translate_qwen3_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut args =
        qwen3_args_from_gguf(checkpoint, metadata, architecture, is_moe, weights_stream)?;
    let mut configs = gguf_affine_configs(checkpoint, translate)?;
    if is_moe {
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.mlp.experts");
            if let Some(config) = configs.remove(&format!("{prefix}.gate_proj")) {
                configs.remove(&format!("{prefix}.up_proj"));
                configs.insert(format!("{prefix}.gate_up_proj"), config);
            }
        }
    }
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    args.quantization = None;
    let eos_token_ids = gguf_optional_i64(metadata, "tokenizer.ggml.eos_token_id", weights_stream)?
        .and_then(|value| u32::try_from(value).ok())
        .into_iter()
        .collect();
    Ok((args, eos_token_ids))
}

fn qwen3_args_from_gguf(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    is_moe: bool,
    stream: &Stream,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let hidden_size = gguf_i32(metadata, &key("embedding_length"), stream)?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"), stream)?;
    let num_key_value_heads = gguf_optional_i64(metadata, &key("attention.head_count_kv"), stream)?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("GGUF KV-head count exceeds i32".into()))?
        .unwrap_or(num_attention_heads);
    let head_dim = gguf_optional_i64(metadata, &key("attention.key_length"), stream)?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| {
            Error::UnsupportedArchitecture("GGUF attention key length exceeds i32".into())
        })?
        .unwrap_or(hidden_size / num_attention_heads);
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
        None => gguf_i32(metadata, &key("vocab_size"), stream)?,
    };

    Ok(ModelArgs {
        model_type: if is_moe { "qwen3_moe" } else { "qwen3" }.to_string(),
        hidden_size,
        num_hidden_layers: gguf_i32(metadata, &key("block_count"), stream)?,
        intermediate_size: if is_moe {
            gguf_optional_i64(metadata, &key("feed_forward_length"), stream)?
                .map(i32::try_from)
                .transpose()
                .map_err(|_| {
                    Error::UnsupportedArchitecture("GGUF feed-forward length exceeds i32".into())
                })?
                .unwrap_or(0)
        } else {
            gguf_i32(metadata, &key("feed_forward_length"), stream)?
        },
        num_attention_heads,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"), stream)?,
        vocab_size,
        num_key_value_heads,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"), stream)?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"), stream)?
            .unwrap_or(1_000_000.0),
        head_dim,
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        rope_scaling: gguf_rope_scaling(metadata, architecture, stream)?,
        quantization: None,
        quantization_config: None,
        quantized_weights: None,
        moe_intermediate_size: if is_moe {
            gguf_i32(metadata, &key("expert_feed_forward_length"), stream)?
        } else {
            0
        },
        num_experts: if is_moe {
            gguf_i32(metadata, &key("expert_count"), stream)?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            gguf_i32(metadata, &key("expert_used_count"), stream)?
        } else {
            0
        },
        norm_topk_prob: is_moe,
        quantized_weight_configs: None,
    })
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    stream: &Stream,
) -> Result<Option<HashMap<String, FloatOrString>>, Error> {
    let scaling_type_key = format!("{architecture}.rope.scaling.type");
    let Some(scaling_type) = gguf_optional_string(metadata, &scaling_type_key)? else {
        return Ok(None);
    };
    match scaling_type.as_str() {
        "none" | "default" => Ok(None),
        "linear" => {
            let factor_key = format!("{architecture}.rope.scaling.factor");
            let factor = gguf_optional_f32(metadata, &factor_key, stream)?.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "linear GGUF RoPE scaling is missing qwen3.rope.scaling.factor".into(),
                )
            })?;
            Ok(Some(HashMap::from([
                (
                    "rope_type".to_string(),
                    FloatOrString::String("linear".to_string()),
                ),
                ("factor".to_string(), FloatOrString::Float(factor)),
            ])))
        }
        other => Err(Error::UnsupportedArchitecture(format!(
            "GGUF RoPE scaling type {other:?} is not supported by the Qwen3 GGUF loader"
        ))),
    }
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    translate_qwen3_gguf_weight_name(name, false)
}

pub(crate) fn translate_qwen3_gguf_weight_name(name: &str, is_moe: bool) -> String {
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
    if is_moe {
        const MOE_PARAMETERS: [(&str, &str); 4] = [
            ("ffn_gate_inp", "mlp.gate"),
            ("ffn_gate_exps", "mlp.experts.gate_proj"),
            ("ffn_up_exps", "mlp.experts.up_proj"),
            ("ffn_down_exps", "mlp.experts.down_proj"),
        ];
        for (source, target) in MOE_PARAMETERS {
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
    }

    const PARAMETERS: [(&str, &str); 12] = [
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("rope_freqs", "rope_freqs"),
    ];
    for (source, target) in PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

pub(crate) fn gguf_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<String, Error> {
    gguf_optional_string(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<String>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Ok(None),
    }
}

pub(crate) fn gguf_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<i32, Error> {
    let value = gguf_i64(metadata, key, stream)?;
    i32::try_from(value).map_err(|_| {
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

fn gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    _stream: &Stream,
) -> Result<Option<i64>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
        }),
        None => Ok(None),
    }
}

fn gguf_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    stream: &Stream,
) -> Result<f32, Error> {
    gguf_optional_f32(metadata, key, stream)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    _stream: &Stream,
) -> Result<Option<f32>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
        }),
        None => Ok(None),
    }
}

#[derive(Debug, Clone, Deserialize)]
/// Hugging Face safetensors index file.
pub struct WeightMap {
    /// Index metadata.
    pub metadata: HashMap<String, Value>,
    /// Mapping from tensor name to shard file name.
    pub weight_map: HashMap<String, String>,
}

/// Loads a Qwen3 model and safetensors weights from a model directory.
pub fn load_qwen3_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_qwen3_model_args(model_dir)?;
    let mut model = Model::new(model_args, stream)?;

    load_safetensors_dir_lenient(&mut model, model_dir, weights_stream)?;
    model.copy_to_stream(stream)?;

    Ok(model)
}

/// Loads a dense Qwen3 checkpoint while quantizing matrices tensor-by-tensor.
pub fn load_qwen3_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let mut model_args = get_qwen3_model_args(model_dir)?;
    if !crate::quantization::should_quantize_on_load(
        "Qwen3",
        model_args.weight_quantization(),
        quantization,
    )? {
        return load_qwen3_model(model_dir, stream, weights_stream);
    }
    model_args.quantization = Some(quantization);
    let mut model = Model::new(model_args, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_quantized_strict(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        quantization,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
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
        let logits = self.forward(
            ModelInput {
                inputs: &prompt_tokens,
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

/// Qwen3 token generation iterator.
pub type Generate<'a, C, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Vec<Option<C>>, S>;

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use safemlx::{
        module::ModuleParameters,
        ops::indexing::{NewAxis, TryIndexOp},
        ops::GgufMetadataValue,
        transforms::eval,
        Array,
    };

    use crate::{
        cache::{ConcatKeyValueCache, KeyValueCache},
        models::common::generation::CausalLm,
        models::qwen3::{load_qwen3_model, load_qwen3_tokenizer},
        quantization::AffineQuantization,
    };

    const CACHED_TEST_MODEL_DIR: &str = "../cache/Qwen3-4B-bf16";

    fn tiny_args() -> super::ModelArgs {
        super::ModelArgs {
            model_type: "qwen3".into(),
            hidden_size: 32,
            num_hidden_layers: 1,
            intermediate_size: 64,
            num_attention_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 32,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            rope_theta: 1_000_000.0,
            head_dim: 32,
            tie_word_embeddings: true,
            rope_scaling: None,
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            quantized_weight_configs: None,
        }
    }

    #[test]
    fn translates_gguf_qwen3_weight_names() {
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.attn_q.weight"),
            "model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.attn_q_norm.weight"),
            "model.layers.3.self_attn.q_norm.weight"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.attn_k_norm.weight"),
            "model.layers.3.self_attn.k_norm.weight"
        );
        assert_eq!(
            super::translate_gguf_weight_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn translates_qwen3_moe_experts_and_mixed_affine_shapes() {
        assert_eq!(
            super::translate_qwen3_gguf_weight_name("blk.3.ffn_gate_inp.weight", true),
            "model.layers.3.mlp.gate.weight"
        );
        assert_eq!(
            super::translate_qwen3_gguf_weight_name("blk.3.ffn_gate_exps.scales", true),
            "model.layers.3.mlp.experts.gate_proj_scales"
        );
        assert_eq!(
            super::translate_qwen3_gguf_weight_name("blk.3.ffn_down_exps.weight", true),
            "model.layers.3.mlp.experts.down_proj"
        );
        assert_eq!(
            crate::quantization::gguf_affine_quantization(&[4096, 256], &[4096, 64], "q_proj",)
                .unwrap(),
            AffineQuantization::new(32, 4).unwrap()
        );
        assert_eq!(
            crate::quantization::gguf_affine_quantization(&[512, 512], &[512, 64], "k_proj",)
                .unwrap(),
            AffineQuantization::new(32, 8).unwrap()
        );
        assert_eq!(
            crate::quantization::gguf_affine_quantization(&[4096, 320], &[4096, 64], "v_proj",)
                .unwrap(),
            AffineQuantization::new(32, 5).unwrap()
        );
        assert_eq!(
            crate::quantization::gguf_affine_quantization(&[1024, 192], &[1024, 64], "down_proj",)
                .unwrap(),
            AffineQuantization::new(16, 6).unwrap()
        );
        assert_eq!(
            crate::quantization::gguf_affine_quantization(&[1024, 64], &[1024, 64], "q2_proj",)
                .unwrap(),
            AffineQuantization::new(16, 2).unwrap()
        );
        assert_eq!(
            crate::quantization::gguf_affine_quantization(&[1024, 96], &[1024, 64], "q3_proj",)
                .unwrap(),
            AffineQuantization::new(16, 3).unwrap()
        );
    }

    #[test]
    fn qwen3_moe_builds_packed_expert_parameter_tree() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut args = tiny_args();
        args.model_type = "qwen3_moe".into();
        args.intermediate_size = 0;
        args.moe_intermediate_size = 8;
        args.num_experts = 4;
        args.num_experts_per_tok = 2;
        args.norm_topk_prob = true;
        args.quantized_weight_configs = Some(HashMap::from([
            (
                "model.layers.0.mlp.experts.gate_up_proj".into(),
                AffineQuantization::new(32, 4).unwrap(),
            ),
            (
                "model.layers.0.mlp.experts.down_proj".into(),
                AffineQuantization::new(32, 4).unwrap(),
            ),
        ]));
        let model = super::Model::new(args, ctx.stream()).unwrap();
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.mlp.gate.weight"));
        assert_eq!(
            params["model.layers.0.mlp.experts.gate_up_proj"].shape(),
            &[4, 16, 4]
        );
        assert_eq!(
            params["model.layers.0.mlp.experts.down_proj"].shape(),
            &[4, 32, 1]
        );
        assert!(!params.contains_key("model.layers.0.mlp.gate_proj.weight"));
    }

    #[test]
    fn mixed_quantization_builds_only_selected_qwen3_parameters() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut args = tiny_args();
        args.quantization = Some(AffineQuantization::new(32, 4).unwrap().into());
        args.quantized_weights = Some(HashSet::from([
            "model.layers.0.self_attn.q_proj.weight".to_string()
        ]));

        let model = super::Model::new(args, ctx.stream()).unwrap();
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert!(params.contains_key("model.layers.0.self_attn.k_proj.weight"));
        assert!(!params.contains_key("model.layers.0.self_attn.k_proj.scales"));
    }

    #[test]
    fn parses_qwen3_gguf_metadata_with_explicit_head_dim() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let metadata = HashMap::from([
            (
                "qwen3.embedding_length".into(),
                GgufMetadataValue::Uint32(1024),
            ),
            ("qwen3.block_count".into(), GgufMetadataValue::Uint32(28)),
            (
                "qwen3.feed_forward_length".into(),
                GgufMetadataValue::Uint32(3072),
            ),
            (
                "qwen3.attention.head_count".into(),
                GgufMetadataValue::Uint32(16),
            ),
            (
                "qwen3.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "qwen3.attention.key_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "qwen3.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "qwen3.context_length".into(),
                GgufMetadataValue::Uint32(40960),
            ),
            (
                "qwen3.rope.freq_base".into(),
                GgufMetadataValue::Float32(1_000_000.0),
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
        let args = super::qwen3_args_from_gguf(&HashMap::new(), &metadata, "qwen3", false, stream)
            .unwrap();

        assert_eq!(args.head_dim, 128);
        assert_eq!(args.num_key_value_heads, 8);
        assert_eq!(args.vocab_size, 32);
        assert!(args.tie_word_embeddings);
    }

    #[test]
    fn loads_dense_qwen3_from_synthetic_gguf_checkpoint() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let source = super::Model::new(tiny_args(), stream).unwrap();
        let arrays = source
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| {
                let name = name
                    .replace("model.layers.", "blk.")
                    .replace("self_attn.q_norm", "attn_q_norm")
                    .replace("self_attn.k_norm", "attn_k_norm")
                    .replace("self_attn.q_proj", "attn_q")
                    .replace("self_attn.k_proj", "attn_k")
                    .replace("self_attn.v_proj", "attn_v")
                    .replace("self_attn.o_proj", "attn_output")
                    .replace("input_layernorm", "attn_norm")
                    .replace("post_attention_layernorm", "ffn_norm")
                    .replace("mlp.gate_proj", "ffn_gate")
                    .replace("mlp.down_proj", "ffn_down")
                    .replace("mlp.up_proj", "ffn_up")
                    .replace("model.embed_tokens", "token_embd")
                    .replace("model.norm", "output_norm");
                (name, value.clone())
            })
            .collect();
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("qwen3".into()),
            ),
            (
                "qwen3.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            ("qwen3.block_count".into(), GgufMetadataValue::Uint32(1)),
            (
                "qwen3.feed_forward_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            (
                "qwen3.attention.head_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3.attention.key_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "qwen3.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "qwen3.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "qwen3.rope.freq_base".into(),
                GgufMetadataValue::Float32(1_000_000.0),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::String(vec![
                    "token"
                        .into();
                    32
                ])),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::Uint32(1),
            ),
        ]);

        let fixture = crate::test_utils::SyntheticGguf::dense(&arrays, &metadata);
        let loaded = super::load_qwen3_gguf_with_metadata(fixture.path(), stream, stream).unwrap();
        assert_eq!(loaded.model.args.head_dim, 32);
        assert_eq!(loaded.eos_token_ids, vec![1]);
    }

    #[test]
    fn pairs_moe_gate_and_up_banks_across_synthetic_gguf_shards() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let mut args = tiny_args();
        args.model_type = "qwen3_moe".into();
        args.intermediate_size = 0;
        args.moe_intermediate_size = 8;
        args.num_experts = 4;
        args.num_experts_per_tok = 2;
        args.norm_topk_prob = true;
        let mut source = super::Model::new(args, stream).unwrap();
        let gate = Array::full::<f32>(&[4, 8, 32], Array::from_f32(3.0), stream).unwrap();
        let up = Array::full::<f32>(&[4, 8, 32], Array::from_f32(7.0), stream).unwrap();
        let gate_up =
            safemlx::ops::concatenate_axis(&[gate.clone(), up.clone()], 1, stream).unwrap();
        **source
            .parameters_mut()
            .flatten()
            .get_mut("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap() = gate_up.clone();

        let mut arrays = HashMap::new();
        for (name, value) in source.parameters().flatten() {
            if name.as_ref() == "model.layers.0.mlp.experts.gate_up_proj" {
                arrays.insert("blk.0.ffn_gate_exps.weight".into(), gate.clone());
                arrays.insert("blk.0.ffn_up_exps.weight".into(), up.clone());
                continue;
            }
            let name = if name.as_ref() == "model.layers.0.mlp.experts.down_proj" {
                "blk.0.ffn_down_exps.weight".into()
            } else {
                name.replace("model.layers.", "blk.")
                    .replace("self_attn.q_norm", "attn_q_norm")
                    .replace("self_attn.k_norm", "attn_k_norm")
                    .replace("self_attn.q_proj", "attn_q")
                    .replace("self_attn.k_proj", "attn_k")
                    .replace("self_attn.v_proj", "attn_v")
                    .replace("self_attn.o_proj", "attn_output")
                    .replace("input_layernorm", "attn_norm")
                    .replace("post_attention_layernorm", "ffn_norm")
                    .replace("mlp.gate.weight", "ffn_gate_inp.weight")
                    .replace("model.embed_tokens", "token_embd")
                    .replace("model.norm", "output_norm")
            };
            arrays.insert(name, value.clone());
        }
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("qwen3moe".into()),
            ),
            (
                "qwen3moe.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            ("qwen3moe.block_count".into(), GgufMetadataValue::Uint32(1)),
            (
                "qwen3moe.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(8),
            ),
            ("qwen3moe.expert_count".into(), GgufMetadataValue::Uint32(4)),
            (
                "qwen3moe.expert_used_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "qwen3moe.attention.head_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3moe.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3moe.attention.key_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "qwen3moe.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "qwen3moe.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "qwen3moe.rope.freq_base".into(),
                GgufMetadataValue::Float32(1_000_000.0),
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
        let fixture =
            crate::test_utils::SyntheticGguf::sharded_dense(&arrays, &metadata, 2, |name| {
                usize::from(name == "blk.0.ffn_up_exps.weight")
            });
        let checkpoint = safemlx::ops::GgufCheckpoint::open(fixture.path()).unwrap();
        assert_eq!(checkpoint.catalog().shards().len(), 2);
        assert_eq!(checkpoint.catalog().physical_tensor_count(), arrays.len());

        let loaded = super::load_qwen3_gguf_with_metadata(fixture.path(), stream, stream).unwrap();
        assert_eq!(loaded.model.model_type(), "qwen3_moe");
        let parameters = loaded.model.parameters().flatten();
        let paired = &parameters["model.layers.0.mlp.experts.gate_up_proj"];
        assert!(paired
            .all_close(&gate_up, None, None, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    #[ignore = "requires QWEN3_MOE_GGUF and Metal"]
    fn strict_loads_and_runs_real_qwen3_moe_gguf() {
        let gguf_file = std::path::PathBuf::from(
            std::env::var("QWEN3_MOE_GGUF")
                .expect("set QWEN3_MOE_GGUF to a local Qwen3 MoE checkpoint"),
        );
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut model = super::load_qwen3_gguf(&gguf_file, stream, weights_ctx.stream()).unwrap();
        assert_eq!(model.model_type(), "qwen3_moe");
        assert_eq!(model.args.num_hidden_layers, 48);
        assert_eq!(model.args.num_experts, 128);
        assert_eq!(model.args.num_experts_per_tok, 8);

        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [crate::models::input::InputPart::text_token_ids(&tokens)];
        let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let logits = CausalLm::prefill_input_logits(
            &mut model,
            crate::models::input::ModelInput::new(&parts),
            &mut cache,
            stream,
        )
        .unwrap();
        assert_eq!(logits.shape(), &[1, 151936]);
        assert_eq!(cache.len(), 48);
        assert!(cache
            .iter()
            .all(|layer| layer.as_ref().is_some_and(|layer| layer.offset() == 2)));

        let next = Array::from_slice(&[151667_u32], &[1, 1]);
        let logits = CausalLm::decode_logits(&mut model, &next, &mut cache, stream).unwrap();
        assert_eq!(logits.shape(), &[1, 151936]);
        assert!(cache
            .iter()
            .all(|layer| layer.as_ref().is_some_and(|layer| layer.offset() == 3)));
    }

    #[test]
    #[ignore = "requires QWEN3_Q4_K_M_GGUF and Metal"]
    fn strict_loads_and_runs_real_qwen3_q4_k_m_gguf() {
        let gguf_file = std::path::PathBuf::from(
            std::env::var("QWEN3_Q4_K_M_GGUF")
                .expect("set QWEN3_Q4_K_M_GGUF to a local Qwen3 checkpoint"),
        );
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut model = super::load_qwen3_gguf(&gguf_file, stream, weights_ctx.stream()).unwrap();
        assert!(model
            .args
            .quantized_weight_configs
            .as_ref()
            .is_some_and(|configs| configs.values().any(|config| config.bits == 4)));

        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [crate::models::input::InputPart::text_token_ids(&tokens)];
        let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let logits = CausalLm::prefill_input_logits(
            &mut model,
            crate::models::input::ModelInput::new(&parts),
            &mut cache,
            stream,
        )
        .unwrap();
        assert_eq!(logits.shape(), &[1, model.args.vocab_size]);
        assert_eq!(cache.len(), model.args.num_hidden_layers as usize);
    }

    fn strict_loads_and_runs_real_qwen3_group16_gguf(env_var: &str, bits: i32) {
        let gguf_file = std::path::PathBuf::from(std::env::var(env_var).unwrap_or_else(|_| {
            panic!("set {env_var} to a local Qwen3 group-16 K-quant checkpoint")
        }));
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut model = super::load_qwen3_gguf(&gguf_file, stream, weights_ctx.stream()).unwrap();
        assert!(model
            .args
            .quantized_weight_configs
            .as_ref()
            .is_some_and(|configs| configs
                .values()
                .any(|config| config.group_size == 16 && config.bits == bits)));

        // Keep this above every QMV/QMM crossover so the real-checkpoint test
        // exercises the tiled group-16 prefill kernels in every projection.
        let token_ids = vec![1_u32; 64];
        let tokens = Array::from_slice(&token_ids, &[1, 64]);
        let parts = [crate::models::input::InputPart::text_token_ids(&tokens)];
        let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let logits = CausalLm::prefill_input_logits(
            &mut model,
            crate::models::input::ModelInput::new(&parts),
            &mut cache,
            stream,
        )
        .unwrap();
        assert_eq!(logits.shape(), &[1, model.args.vocab_size]);
        assert_eq!(cache.len(), model.args.num_hidden_layers as usize);
        assert!(cache
            .iter()
            .all(|layer| layer.as_ref().is_some_and(|layer| layer.offset() == 64)));
    }

    #[test]
    #[ignore = "requires QWEN3_Q2_K_GGUF and Metal"]
    fn strict_loads_and_runs_real_qwen3_q2_k_gguf() {
        strict_loads_and_runs_real_qwen3_group16_gguf("QWEN3_Q2_K_GGUF", 2);
    }

    #[test]
    #[ignore = "requires QWEN3_Q3_K_GGUF and Metal"]
    fn strict_loads_and_runs_real_qwen3_q3_k_gguf() {
        strict_loads_and_runs_real_qwen3_group16_gguf("QWEN3_Q3_K_GGUF", 3);
    }

    #[test]
    #[ignore = "requires QWEN3_Q6_K_GGUF and Metal"]
    fn strict_loads_and_runs_real_qwen3_q6_k_gguf() {
        strict_loads_and_runs_real_qwen3_group16_gguf("QWEN3_Q6_K_GGUF", 6);
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_qwen3_model() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let _model =
            super::load_qwen3_model(CACHED_TEST_MODEL_DIR, ctx.stream(), weights_ctx.stream())
                .unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_qwen3_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_qwen3_with_concat_cache() {
        let tokenizer = load_qwen3_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();
        let mut model = load_qwen3_model(CACHED_TEST_MODEL_DIR, stream, weights_stream).unwrap();

        let encoding = tokenizer.encode("hello", true).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids())
            .try_index_device(NewAxis, stream)
            .unwrap();
        let mut cache = Vec::new();

        let mut tokens = Vec::new();
        let input_parts = [crate::models::input::InputPart::text_token_ids(
            &prompt_tokens,
        )];
        let input = crate::models::input::ModelInput::new(&input_parts);
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model, &mut cache, 0.0, input, None, stream,
        );
        for (token, ntoks) in generate.zip(0..10) {
            let token = token.unwrap();
            tokens.push(token.clone());

            if ntoks == 0 {
                eval(&tokens).unwrap();
            }

            if tokens.len() % 20 == 0 {
                eval(&tokens).unwrap();
                let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>(&stream)).collect();
                let s = tokenizer.decode(&slice, true).unwrap();
                print!("{s}");
            }
        }

        eval(&tokens).unwrap();
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>(&stream)).collect();
        let s = tokenizer.decode(&slice, true).unwrap();
        println!("{s}");

        println!("------");
    }
}
