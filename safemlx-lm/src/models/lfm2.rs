//! Liquid AI LFM2/LFM2.5 dense and mixture-of-experts text models.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    models::{
        common::{
            self,
            attention::{
                apply_rope_and_update_cache, batch_seq, finish_attention,
                reshape_attention_projection,
            },
            convolution::{causal_depthwise_conv1d, CausalConv1dCache, DepthwiseConv1d},
            generation::CausalLm,
            linear::project_logits_maybe_quantized,
            moe::{PackedSwiGluExperts, TopKRouterScoreFunction},
        },
        input,
        qwen3::{gguf_i32, gguf_string},
    },
    quantization::{AffineQuantization, WeightQuantization},
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
    weights::{
        load_arrays_quantized_strict, load_arrays_strict, load_safetensors_dir_quantized_strict,
        load_safetensors_dir_strict, load_safetensors_dir_strict_with_split_swiglu_experts,
        StrictLoadConfig, StrictLoadReport,
    },
};

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_conv_l_cache() -> i32 {
    3
}

fn default_block_multiple_of() -> i32 {
    256
}

fn default_block_ffn_dim_multiplier() -> f32 {
    1.0
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Stateful operator used by an LFM2 decoder layer.
pub enum LayerType {
    /// Gated causal depthwise convolution.
    Conv,
    /// Full grouped-query self-attention.
    FullAttention,
}

impl LayerType {
    fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "conv" => Ok(Self::Conv),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(Error::UnsupportedArchitecture(format!(
                "LFM2 layer type {other:?} is unsupported"
            ))),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
/// Deserialized LFM2/LFM2.5 configuration.
pub struct ModelArgs {
    /// Hugging Face model type (`lfm2` or `lfm2_moe`).
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Configured dense intermediate size before optional LFM adjustment.
    pub intermediate_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Number of query heads.
    pub num_attention_heads: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Maximum configured context length.
    pub max_position_embeddings: i32,
    /// RMSNorm epsilon.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Per-layer operator schedule.
    pub layer_types: Vec<String>,
    /// Causal convolution kernel width.
    #[serde(default = "default_conv_l_cache", rename = "conv_L_cache")]
    pub conv_l_cache: i32,
    /// Whether convolution projections and kernels include biases.
    #[serde(default)]
    pub conv_bias: bool,
    /// Dense FFN rounding multiple.
    #[serde(default = "default_block_multiple_of")]
    pub block_multiple_of: i32,
    /// Dense FFN multiplier after the LFM two-thirds adjustment.
    #[serde(default = "default_block_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: f32,
    /// Whether the configured dense FFN size receives LFM's two-thirds adjustment.
    #[serde(default = "default_true")]
    pub block_auto_adjust_ff_dim: bool,
    /// Legacy dense hidden-size alias.
    #[serde(default)]
    pub block_dim: Option<i32>,
    /// Legacy dense FFN-size alias.
    #[serde(default)]
    pub block_ff_dim: Option<i32>,
    /// Whether logits use tied input embeddings.
    #[serde(default = "default_true", alias = "tie_embedding")]
    pub tie_word_embeddings: bool,
    /// Legacy top-level RoPE base.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Hugging Face v5 RoPE configuration.
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, FloatOrString>>,
    /// Per-expert intermediate size for MoE checkpoints.
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Number of leading dense feed-forward layers in MoE checkpoints.
    #[serde(default)]
    pub num_dense_layers: i32,
    /// Number of routed experts.
    #[serde(default)]
    pub num_experts: i32,
    /// Experts selected per token.
    #[serde(default)]
    pub num_experts_per_tok: i32,
    /// Whether selected route weights are normalized.
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Router output scale.
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    /// Whether MoE selection uses the checkpoint expert bias.
    #[serde(default)]
    pub use_expert_bias: bool,
    /// Preferred MLX quantization metadata.
    #[serde(default)]
    pub quantization: Option<WeightQuantization>,
    /// Hugging Face-compatible quantization metadata alias.
    #[serde(default)]
    pub quantization_config: Option<WeightQuantization>,
    /// Exact mixed-quantization weight names populated by GGUF loading.
    #[serde(skip)]
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-weight affine layouts populated by GGUF loading.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, AffineQuantization>>,
}

impl ModelArgs {
    /// Returns whether this is an MoE checkpoint.
    pub fn is_moe(&self) -> bool {
        self.model_type == "lfm2_moe"
    }

    pub(crate) fn layer_type(&self, index: usize) -> Result<LayerType, Error> {
        self.layer_types
            .get(index)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "LFM2 layer index {index} is outside layer_types"
                ))
            })
            .and_then(|value| LayerType::parse(value))
    }

    fn dense_intermediate_size(&self) -> i32 {
        let mut size = self.block_ff_dim.unwrap_or(self.intermediate_size);
        if self.block_auto_adjust_ff_dim {
            size = 2 * size / 3;
            size = (self.block_ffn_dim_multiplier * size as f32) as i32;
            size = self.block_multiple_of
                * ((size + self.block_multiple_of - 1) / self.block_multiple_of);
        }
        size
    }

    fn rope_theta(&self) -> f32 {
        match self
            .rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.get("rope_theta"))
        {
            Some(FloatOrString::Float(value)) => *value,
            Some(FloatOrString::String(value)) => value.parse().unwrap_or(self.rope_theta),
            _ => self.rope_theta,
        }
    }

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
}

/// Validates a parsed LFM2 configuration.
pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    let args: ModelArgs = serde_json::from_value(config.clone())
        .map_err(|error| Error::UnsupportedArchitecture(format!("invalid LFM2 config: {error}")))?;
    validate_args(&args)
}

fn validate_args(args: &ModelArgs) -> Result<(), Error> {
    if !matches!(args.model_type.as_str(), "lfm2" | "lfm2_moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "LFM2 loader received model_type {:?}",
            args.model_type
        )));
    }
    for (name, value) in [
        ("vocab_size", args.vocab_size),
        ("hidden_size", args.hidden_size),
        ("intermediate_size", args.intermediate_size),
        ("num_hidden_layers", args.num_hidden_layers),
        ("num_attention_heads", args.num_attention_heads),
        ("num_key_value_heads", args.num_key_value_heads),
        ("conv_L_cache", args.conv_l_cache),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "LFM2 {name} must be positive, got {value}"
            )));
        }
    }
    if args.layer_types.len() != args.num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "LFM2 layer_types has {} entries, expected {}",
            args.layer_types.len(),
            args.num_hidden_layers
        )));
    }
    for layer_type in &args.layer_types {
        LayerType::parse(layer_type)?;
    }
    if args.hidden_size % args.num_attention_heads != 0
        || args.num_attention_heads % args.num_key_value_heads != 0
    {
        return Err(Error::UnsupportedArchitecture(
            "LFM2 attention head counts do not divide hidden/query dimensions".into(),
        ));
    }
    if args.dense_intermediate_size() <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "LFM2 adjusted dense intermediate size must be positive".into(),
        ));
    }
    if args.is_moe()
        && (args.moe_intermediate_size <= 0
            || args.num_experts <= 0
            || args.num_experts_per_tok <= 0
            || args.num_experts_per_tok > args.num_experts
            || args.num_dense_layers < 0
            || args.num_dense_layers > args.num_hidden_layers)
    {
        return Err(Error::UnsupportedArchitecture(
            "LFM2 MoE expert or dense-prefix configuration is invalid".into(),
        ));
    }
    Ok(())
}

/// LFM2 attention input.
pub struct AttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional causal mask.
    pub mask: Option<&'a Array>,
    /// Optional KV cache.
    pub cache: Option<&'a mut ConcatKeyValueCache>,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// QK-normalized grouped-query attention used by LFM2 full-attention layers.
pub struct Attention {
    /// Query head count.
    pub n_heads: i32,
    /// Key/value head count.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale.
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
    pub out_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Query head RMSNorm.
    pub q_layernorm: nn::RmsNorm,
    #[param]
    /// Key head RMSNorm.
    pub k_layernorm: nn::RmsNorm,
    #[param]
    /// Rotary embedding.
    pub rope: RopeVariant,
}

impl Attention {
    fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let head_dim = args.hidden_size / args.num_attention_heads;
        let prefix = format!("model.layers.{layer}.self_attn");
        Ok(Self {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.num_attention_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.q_proj.weight")),
                stream,
            )?,
            k_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.num_key_value_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.k_proj.weight")),
                stream,
            )?,
            v_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.num_key_value_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.v_proj.weight")),
                stream,
            )?,
            out_proj: common::linear::unloaded_maybe_quantized_linear(
                args.num_attention_heads * head_dim,
                args.hidden_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.out_proj.weight")),
                stream,
            )?,
            q_layernorm: nn::RmsNorm::unloaded(head_dim, args.norm_eps, Dtype::Float32, stream)?,
            k_layernorm: nn::RmsNorm::unloaded(head_dim, args.norm_eps, Dtype::Float32, stream)?,
            rope: initialize_rope(
                head_dim,
                args.rope_theta(),
                false,
                &None,
                args.max_position_embeddings,
                stream,
            )?,
        })
    }
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
        let queries = self.q_layernorm.forward(
            &reshape_attention_projection(
                self.q_proj.forward(x, stream)?,
                batch,
                seq_len,
                self.n_heads,
                stream,
            )?,
            stream,
        )?;
        let keys = self.k_layernorm.forward(
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
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        let output = finish_attention(
            queries, keys, values, cache, self.scale, mask, batch, seq_len, stream,
        )?;
        self.out_proj.forward(&output, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.out_proj.training_mode(mode);
        self.q_layernorm.training_mode(mode);
        self.k_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// LFM2 gated causal short convolution.
pub struct ShortConv {
    #[param]
    /// Depthwise convolution kernel.
    pub conv: DepthwiseConv1d,
    #[quantizable]
    #[param]
    /// Joint B/C/x projection.
    pub in_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Output projection.
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl ShortConv {
    fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.conv");
        Ok(Self {
            conv: DepthwiseConv1d::new(
                args.hidden_size,
                args.conv_l_cache,
                args.conv_bias,
                stream,
            )?,
            in_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                3 * args.hidden_size,
                args.conv_bias,
                args.weight_quantization_for(&format!("{prefix}.in_proj.weight")),
                stream,
            )?,
            out_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.hidden_size,
                args.conv_bias,
                args.weight_quantization_for(&format!("{prefix}.out_proj.weight")),
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        cache: Option<&mut CausalConv1dCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let projected = self.in_proj.forward(x, stream)?;
        let hidden = x.dim(-1);
        let b = projected.try_index_device((.., .., ..hidden), stream)?;
        let c = projected.try_index_device((.., .., hidden..2 * hidden), stream)?;
        let x = projected.try_index_device((.., .., 2 * hidden..), stream)?;
        let bx = b.multiply(x, stream)?;
        let convolution = causal_depthwise_conv1d(&self.conv, &bx, cache, stream)?;
        self.out_proj
            .forward(&c.multiply(convolution, stream)?, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Dense or sparse LFM2 feed-forward block with checkpoint-compatible names.
pub struct FeedForward {
    /// Whether this block is sparse MoE.
    pub is_moe: bool,
    #[quantizable]
    #[param]
    /// Dense gate projection.
    pub w1: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    /// Dense down projection.
    pub w2: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    /// Dense up projection.
    pub w3: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    /// Sparse router.
    pub gate: Option<common::moe::TopKRouter>,
    #[param]
    /// Packed routed experts.
    pub experts: Option<PackedSwiGluExperts>,
    #[param]
    /// Optional selection-only expert bias.
    pub expert_bias: Param<Option<Array>>,
}

impl FeedForward {
    fn dense(
        args: &ModelArgs,
        layer: i32,
        intermediate_size: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.feed_forward");
        Ok(Self {
            is_moe: false,
            w1: Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                intermediate_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.w1.weight")),
                stream,
            )?),
            w2: Some(common::linear::unloaded_maybe_quantized_linear(
                intermediate_size,
                args.hidden_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.w2.weight")),
                stream,
            )?),
            w3: Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                intermediate_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.w3.weight")),
                stream,
            )?),
            gate: None,
            experts: None,
            expert_bias: Param::new(None),
        })
    }

    fn moe(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.feed_forward.experts");
        Ok(Self {
            is_moe: true,
            w1: None,
            w2: None,
            w3: None,
            gate: Some(common::moe::TopKRouter::new(
                common::moe::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.num_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Sigmoid,
                    norm_topk_prob: args.norm_topk_prob,
                    normalization_epsilon: 1e-6,
                    routed_scaling_factor: args.routed_scaling_factor,
                    n_group: 1,
                    topk_group: 1,
                    score_correction_bias: false,
                },
                stream,
            )?),
            experts: Some(PackedSwiGluExperts::new(
                args.num_experts,
                args.hidden_size,
                args.moe_intermediate_size,
                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?),
            expert_bias: if args.use_expert_bias {
                Param::<Option<Array>>::unloaded_some(&[args.num_experts], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
        })
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        if !self.is_moe {
            let gate = self.w1.as_mut().expect("dense w1").forward(x, stream)?;
            let up = self.w3.as_mut().expect("dense w3").forward(x, stream)?;
            let hidden = common::layers::silu(gate, stream)?.multiply(up, stream)?;
            return self.w2.as_mut().expect("dense w2").forward(&hidden, stream);
        }
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self
            .gate
            .as_mut()
            .expect("MoE gate")
            .forward_with_selection_bias(&flat, self.expert_bias.as_ref().as_ref(), stream)?;
        self.experts
            .as_mut()
            .expect("MoE experts")
            .forward(&flat, &indices, &weights, stream)?
            .reshape(shape, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        for projection in [&mut self.w1, &mut self.w2, &mut self.w3]
            .into_iter()
            .flatten()
        {
            projection.training_mode(mode);
        }
    }
}

#[derive(Debug, Clone)]
/// Cache for one LFM2 operator layer.
pub enum LayerCache {
    /// Full-attention KV cache.
    Attention(ConcatKeyValueCache),
    /// Short-convolution state.
    Conv(CausalConv1dCache),
}

impl LayerCache {
    pub(crate) fn new(layer_type: LayerType) -> Self {
        match layer_type {
            LayerType::Conv => Self::Conv(CausalConv1dCache::default()),
            // Match mlx-lm's KVCache growth policy. Chunked backing arrays
            // avoid concatenating the complete cache for every decode token.
            LayerType::FullAttention => Self::Attention(ConcatKeyValueCache::new_with_step(256)),
        }
    }

    pub(crate) fn offset(&self) -> i32 {
        match self {
            Self::Attention(cache) => cache.offset(),
            Self::Conv(cache) => cache.offset,
        }
    }

    pub(crate) fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Attention(cache) => cache.retained_arrays(),
            Self::Conv(cache) => cache.state.iter().collect(),
        }
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous LFM2 generation cache.
pub struct Cache {
    /// Per-layer operator caches.
    pub layers: Vec<LayerCache>,
}

impl Cache {
    pub(crate) fn new(args: &ModelArgs) -> Result<Self, Error> {
        Ok(Self {
            layers: (0..args.num_hidden_layers)
                .map(|index| args.layer_type(index as usize).map(LayerCache::new))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Returns the consumed-token offset.
    pub fn offset(&self) -> i32 {
        self.layers.first().map(LayerCache::offset).unwrap_or(0)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// One LFM2 decoder layer.
pub struct DecoderLayer {
    /// Operator kind.
    pub layer_type: LayerType,
    #[quantizable]
    #[param]
    /// Full attention operator.
    pub self_attn: Option<Attention>,
    #[quantizable]
    #[param]
    /// Short convolution operator.
    pub conv: Option<ShortConv>,
    #[quantizable]
    #[param]
    /// Dense or sparse feed-forward block.
    pub feed_forward: FeedForward,
    #[param]
    /// Operator pre-norm.
    pub operator_norm: nn::RmsNorm,
    #[param]
    /// Feed-forward pre-norm.
    pub ffn_norm: nn::RmsNorm,
}

impl DecoderLayer {
    pub(crate) fn new(args: &ModelArgs, index: i32, stream: &Stream) -> Result<Self, Error> {
        let layer_type = args.layer_type(index as usize)?;
        Ok(Self {
            layer_type,
            self_attn: if layer_type == LayerType::FullAttention {
                Some(Attention::new(args, index, stream)?)
            } else {
                None
            },
            conv: if layer_type == LayerType::Conv {
                Some(ShortConv::new(args, index, stream)?)
            } else {
                None
            },
            feed_forward: if args.is_moe() && index >= args.num_dense_layers {
                FeedForward::moe(args, index, stream)?
            } else {
                FeedForward::dense(
                    args,
                    index,
                    if args.is_moe() {
                        args.intermediate_size
                    } else {
                        args.dense_intermediate_size()
                    },
                    stream,
                )?
            },
            operator_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.norm_eps,
                Dtype::Float32,
                stream,
            )?,
            ffn_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_type, cache) {
            (LayerType::FullAttention, Some(LayerCache::Attention(cache))) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerType::FullAttention, _) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (LayerType::Conv, Some(LayerCache::Conv(cache))) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward(&normalized, Some(cache), stream)?,
            (LayerType::Conv, _) => {
                self.conv
                    .as_mut()
                    .expect("conv layer")
                    .forward(&normalized, None, stream)?
            }
        };
        let h = x.add(operator, stream)?;
        let feed_forward = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&h, stream)?, stream)?;
        h.add(feed_forward, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// LFM2 transformer body.
pub struct Lfm2Model {
    #[quantizable]
    #[param]
    /// Token embeddings.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    /// Decoder layers.
    pub layers: Vec<DecoderLayer>,
    #[param]
    /// Final embedding normalization.
    pub embedding_norm: nn::RmsNorm,
}

impl Lfm2Model {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            embed_tokens: common::linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|index| DecoderLayer::new(args, index, stream))
                .collect::<Result<Vec<_>, _>>()?,
            embedding_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        inputs: &Array,
        mut cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs, stream)?;
        let offset = cache.as_ref().map(|cache| cache.offset()).unwrap_or(0);
        let mask = if h.dim(1) > 1 {
            match create_attention_mask(&h, &offset_cache(offset), Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("LFM2 requires an array causal mask"));
                }
                None => None,
            }
        } else {
            None
        };
        if let Some(cache) = cache.as_mut() {
            for (layer, layer_cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
                h = layer.forward(&h, mask.as_ref(), Some(layer_cache), stream)?;
            }
        } else {
            for layer in &mut self.layers {
                h = layer.forward(&h, mask.as_ref(), None, stream)?;
            }
        }
        self.embedding_norm.forward(&h, stream)
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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// LFM2 causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: Lfm2Model,
    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded LFM2 model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let model = Lfm2Model::new(&args, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
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

    /// Creates an empty heterogeneous cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args).expect("validated LFM2 layer schedule")
    }

    pub(crate) fn forward_logits(
        &mut self,
        inputs: &Array,
        cache: Option<&mut Cache>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward(inputs, cache, stream)?;
        let hidden = if last_token_only {
            hidden.try_index_device((.., -1, ..), stream)?
        } else {
            hidden
        };
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )
    }
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward_logits(&tokens, Some(cache), true, stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits(input_tokens, Some(cache), true, stream)
    }
}

/// LFM2 token generation iterator.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

/// Reads and validates LFM2 model arguments.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    validate_model_config_value(&value)?;
    Ok(serde_json::from_value(value)?)
}

/// Loads an LFM2 safetensors checkpoint.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;
    let mut model = Model::new(args.clone(), stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    if args.is_moe() {
        load_safetensors_dir_strict_with_split_swiglu_experts(
            &mut model,
            model_dir,
            weights_stream,
            stream,
            None,
            &config,
            &mut report,
            args.num_experts,
        )?;
    } else {
        load_safetensors_dir_strict(&mut model, model_dir, weights_stream, &config, &mut report)?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads an LFM2 checkpoint while quantizing eligible projections.
pub fn load_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let mut args = get_model_args(model_dir)?;
    if !crate::quantization::should_quantize_on_load(
        "LFM2",
        args.weight_quantization(),
        quantization,
    )? {
        return load_model(model_dir, stream, weights_stream);
    }
    args.quantization = Some(quantization);
    let mut model = Model::new(args.clone(), stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    if args.is_moe() {
        load_safetensors_dir_strict_with_split_swiglu_experts(
            &mut model,
            model_dir,
            weights_stream,
            stream,
            Some(quantization),
            &config,
            &mut report,
            args.num_experts,
        )?;
    } else {
        load_safetensors_dir_quantized_strict(
            &mut model,
            model_dir,
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

/// Loads the tokenizer stored next to an LFM2 checkpoint.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Ok(Tokenizer::from_file(
        model_dir.as_ref().join("tokenizer.json"),
    )?)
}

pub(crate) struct LoadedLfm2Gguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads an LFM2 or LFM2-MoE GGUF checkpoint.
pub fn load_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let (arrays, metadata) = Array::load_gguf_with_metadata(gguf_file, weights_stream)?;
    Ok(load_gguf_data(arrays, metadata, None, stream, weights_stream)?.model)
}

pub(crate) fn load_gguf_data(
    arrays: HashMap<String, Array>,
    metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedLfm2Gguf, Error> {
    let architecture = gguf_string(&metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "lfm2" | "lfm2moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports lfm2 and lfm2moe"
        )));
    }
    let is_moe = architecture == "lfm2moe";
    let mut args = args_from_gguf(&arrays, &metadata, &architecture, is_moe, weights_stream)?;
    let mut translated = HashMap::with_capacity(arrays.len());
    for (name, mut value) in arrays {
        if name.ends_with(".shortconv.conv.weight") && value.ndim() == 2 {
            value = value.reshape(&[value.dim(0), 1, value.dim(1)], weights_stream)?;
        }
        let translated_name = translate_gguf_weight_name(&name, is_moe);
        if translated.insert(translated_name.clone(), value).is_some() {
            return Err(Error::UnsupportedArchitecture(format!(
                "LFM2 GGUF tensors collide after translating {translated_name:?}"
            )));
        }
    }
    if is_moe {
        pack_moe_experts(&mut translated, &args, weights_stream)?;
    }
    let configs = gguf_quantized_weight_configs(&translated)?;
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
    }
    validate_args(&args)?;

    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    if let Some(quantization) = quantization {
        load_arrays_quantized_strict(
            &mut model,
            translated,
            stream,
            quantization,
            &config,
            &mut report,
        )?;
    } else {
        load_arrays_strict(&mut model, translated, &config, &mut report)?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    let eos_token_ids = gguf_optional_i64(&metadata, "tokenizer.ggml.eos_token_id")?
        .and_then(|value| u32::try_from(value).ok())
        .into_iter()
        .collect();
    Ok(LoadedLfm2Gguf {
        model,
        eos_token_ids,
    })
}

fn args_from_gguf(
    arrays: &HashMap<String, Array>,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    is_moe: bool,
    stream: &Stream,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"), stream)?;
    let kv_heads = expand_layer_values(
        &key("attention.head_count_kv"),
        gguf_i64_values(metadata, &key("attention.head_count_kv"))?,
        num_hidden_layers,
    )?;
    let num_key_value_heads = unique_nonzero(&key("attention.head_count_kv"), &kv_heads)?;
    let layer_types = kv_heads
        .iter()
        .map(|heads| {
            if *heads == 0 {
                "conv".to_string()
            } else {
                "full_attention".to_string()
            }
        })
        .collect();
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
    let expert_bias_name =
        |name: &str| name.contains("ffn_exp_probs_b") || name.contains("exp_probs_b");
    Ok(ModelArgs {
        model_type: if is_moe { "lfm2_moe" } else { "lfm2" }.into(),
        vocab_size,
        hidden_size: gguf_i32(metadata, &key("embedding_length"), stream)?,
        intermediate_size: gguf_i32(metadata, &key("feed_forward_length"), stream)?,
        num_hidden_layers,
        num_attention_heads: gguf_i32(metadata, &key("attention.head_count"), stream)?,
        num_key_value_heads,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"), stream)?,
        norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        layer_types,
        conv_l_cache: gguf_i32(metadata, &key("shortconv.l_cache"), stream)?,
        conv_bias: arrays
            .keys()
            .any(|name| name.contains("shortconv") && name.ends_with(".bias")),
        block_multiple_of: 1,
        block_ffn_dim_multiplier: 1.0,
        block_auto_adjust_ff_dim: false,
        block_dim: None,
        block_ff_dim: None,
        tie_word_embeddings: !arrays.contains_key("output.weight"),
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        rope_parameters: None,
        moe_intermediate_size: if is_moe {
            gguf_i32(metadata, &key("expert_feed_forward_length"), stream)?
        } else {
            0
        },
        num_dense_layers: if is_moe {
            gguf_i32(metadata, &key("leading_dense_block_count"), stream)?
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
        norm_topk_prob: if is_moe {
            gguf_optional_i64(metadata, &key("expert_weights_norm"))?.unwrap_or(0) != 0
        } else {
            false
        },
        routed_scaling_factor: gguf_optional_f32(metadata, &key("expert_weights_scale"))?
            .unwrap_or_else(default_routed_scaling_factor),
        use_expert_bias: arrays.keys().any(|name| expert_bias_name(name)),
        quantization: None,
        quantization_config: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    })
}

fn translate_gguf_weight_name(name: &str, is_moe: bool) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("token_embd_norm", "model.embedding_norm"),
        ("output", "lm_head"),
    ] {
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
        for (source, target) in [
            ("ffn_gate_inp", "feed_forward.gate"),
            ("ffn_gate_exps", "feed_forward.experts.gate_proj"),
            ("ffn_up_exps", "feed_forward.experts.up_proj"),
            ("ffn_down_exps", "feed_forward.experts.down_proj"),
            ("ffn_exp_probs_b", "feed_forward.expert_bias"),
            ("exp_probs_b", "feed_forward.expert_bias"),
        ] {
            if parameter == source || parameter.starts_with(&format!("{source}.")) {
                let suffix = parameter.strip_prefix(source).unwrap_or_default();
                let suffix = if target.ends_with("expert_bias") && suffix == ".bias" {
                    ""
                } else if target.contains("experts.") {
                    match suffix {
                        ".weight" => "",
                        ".scales" => "_scales",
                        ".biases" => "_biases",
                        other => other,
                    }
                } else {
                    suffix
                };
                return format!("model.layers.{layer}.{target}{suffix}");
            }
        }
    }
    for (source, target) in [
        ("shortconv.conv", "conv.conv"),
        ("shortconv.in_proj", "conv.in_proj"),
        ("shortconv.out_proj", "conv.out_proj"),
        ("attn_q_norm", "self_attn.q_layernorm"),
        ("attn_k_norm", "self_attn.k_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.out_proj"),
        ("attn_norm", "operator_norm"),
        ("ffn_norm", "ffn_norm"),
        ("ffn_gate", "feed_forward.w1"),
        ("ffn_down", "feed_forward.w2"),
        ("ffn_up", "feed_forward.w3"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

fn pack_moe_experts(
    arrays: &mut HashMap<String, Array>,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(), Error> {
    for layer in args.num_dense_layers..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}.feed_forward.experts");
        let affine = arrays.contains_key(&format!("{prefix}.gate_proj_scales"))
            || arrays.contains_key(&format!("{prefix}.up_proj_scales"));
        let suffixes: &[&str] = if affine {
            &["", "_scales", "_biases"]
        } else {
            &[""]
        };
        for suffix in suffixes {
            let gate_name = format!("{prefix}.gate_proj{suffix}");
            let up_name = format!("{prefix}.up_proj{suffix}");
            match (arrays.remove(&gate_name), arrays.remove(&up_name)) {
                (Some(gate), Some(up)) => {
                    arrays.insert(
                        format!("{prefix}.gate_up_proj{suffix}"),
                        concatenate_axis(&[gate, up], 1, stream)?,
                    );
                }
                (None, None) if *suffix == "_biases" => {}
                _ => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "LFM2 MoE GGUF has incomplete gate/up expert tensors under {prefix}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn gguf_quantized_weight_configs(
    arrays: &HashMap<String, Array>,
) -> Result<HashMap<String, AffineQuantization>, Error> {
    let mut configs = HashMap::new();
    for (scales_name, scales) in arrays {
        let weight_name = if let Some(prefix) = scales_name.strip_suffix(".scales") {
            format!("{prefix}.weight")
        } else if let Some(prefix) = scales_name.strip_suffix("_scales") {
            prefix.to_string()
        } else {
            continue;
        };
        if let Some(weight) = arrays.get(&weight_name) {
            configs.insert(
                weight_name.clone(),
                crate::quantization::gguf_affine_quantization(
                    weight.shape(),
                    scales.shape(),
                    &weight_name,
                )?,
            );
        }
    }
    Ok(configs)
}

fn gguf_i64_values(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Vec<i64>, Error> {
    metadata
        .get(key)
        .and_then(GgufMetadataValue::to_i64_vec)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata is missing numeric key {key:?}"))
        })
}

fn gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i64>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {key:?} must be a numeric scalar"
            ))
        }),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<f32, Error> {
    gguf_optional_f32(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
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

fn expand_layer_values(key: &str, values: Vec<i64>, layers: i32) -> Result<Vec<i32>, Error> {
    let values = if values.len() == 1 {
        vec![values[0]; layers as usize]
    } else if values.len() == layers as usize {
        values
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has {} values for {layers} layers",
            values.len()
        )));
    };
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
            })
        })
        .collect()
}

fn unique_nonzero(key: &str, values: &[i32]) -> Result<i32, Error> {
    let Some(value) = values.iter().copied().find(|value| *value > 0) else {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has no attention-layer value"
        )));
    };
    if values.iter().any(|other| *other > 0 && *other != value) {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has non-uniform attention-layer values"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use safemlx::{module::ModuleParameters, Device, DeviceType, ExecutionContext};

    use super::{translate_gguf_weight_name, validate_model_config_value, LayerType, ModelArgs};
    use serde_json::json;

    fn dense_config() -> serde_json::Value {
        json!({
            "model_type": "lfm2",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "block_multiple_of": 4,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true,
            "layer_types": ["conv", "full_attention", "conv"],
            "tie_embedding": true
        })
    }

    #[test]
    fn parses_dense_schedule_and_adjusts_ffn() {
        let args: ModelArgs = serde_json::from_value(dense_config()).unwrap();
        assert_eq!(args.layer_type(0).unwrap(), LayerType::Conv);
        assert_eq!(args.layer_type(1).unwrap(), LayerType::FullAttention);
        assert_eq!(args.dense_intermediate_size(), 16);
        validate_model_config_value(&dense_config()).unwrap();
    }

    #[test]
    fn rejects_bad_schedule_length() {
        let mut config = dense_config();
        config["layer_types"] = json!(["conv"]);
        assert!(validate_model_config_value(&config).is_err());
    }

    #[test]
    fn accepts_published_dense_aliases_and_rope_parameters() {
        let mut config = dense_config();
        config["block_norm_eps"] = json!(0.00001);
        config["rope_parameters"] = json!({
            "rope_theta": 1_000_000.0,
            "rope_type": "default"
        });
        validate_model_config_value(&config).unwrap();
        let args: ModelArgs = serde_json::from_value(config).unwrap();
        assert!(args.tie_word_embeddings);
        assert_eq!(args.rope_theta(), 1_000_000.0);
    }

    #[test]
    fn accepts_published_moe_shape() {
        let config = json!({
            "model_type": "lfm2_moe",
            "vocab_size": 65536,
            "hidden_size": 2048,
            "intermediate_size": 7168,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 128000,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "layer_types": ["conv", "conv", "full_attention", "conv"],
            "moe_intermediate_size": 1792,
            "num_dense_layers": 2,
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "norm_topk_prob": true,
            "use_expert_bias": true
        });
        validate_model_config_value(&config).unwrap();
    }

    #[test]
    fn translates_dense_and_moe_gguf_tensor_names() {
        assert_eq!(
            translate_gguf_weight_name("token_embd.weight", false),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.shortconv.conv.weight", false),
            "model.layers.2.conv.conv.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.attn_q_norm.weight", false),
            "model.layers.2.self_attn.q_layernorm.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.3.ffn_gate_exps.scales", true),
            "model.layers.3.feed_forward.experts.gate_proj_scales"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.3.ffn_exp_probs_b.bias", true),
            "model.layers.3.feed_forward.expert_bias"
        );
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn dense_parameter_tree_and_cache_match_public_checkpoint_layout() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let args: ModelArgs = serde_json::from_value(dense_config()).unwrap();
        let model = super::Model::new(args, context.stream()).unwrap();
        let params = model.parameters().flatten();
        assert_eq!(
            params["model.layers.0.conv.conv.weight"].shape(),
            &[16, 1, 3]
        );
        assert!(params.contains_key("model.layers.0.conv.in_proj.weight"));
        assert!(params.contains_key("model.layers.0.feed_forward.w1.weight"));
        assert!(params.contains_key("model.layers.1.self_attn.q_proj.weight"));
        assert!(params.contains_key("model.layers.1.self_attn.q_layernorm.weight"));
        assert!(params.contains_key("model.embedding_norm.weight"));
        assert!(!params.contains_key("lm_head.weight"));
        let cache = model.new_cache();
        assert!(matches!(cache.layers[0], super::LayerCache::Conv(_)));
        assert!(matches!(cache.layers[1], super::LayerCache::Attention(_)));
        assert_eq!(cache.offset(), 0);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn moe_parameter_tree_packs_experts_after_dense_prefix() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let config = json!({
            "model_type": "lfm2_moe",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "layer_types": ["conv", "full_attention"],
            "moe_intermediate_size": 8,
            "num_dense_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "use_expert_bias": true
        });
        let args: ModelArgs = serde_json::from_value(config).unwrap();
        let model = super::Model::new(args, context.stream()).unwrap();
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.feed_forward.w1.weight"));
        assert!(params.contains_key("model.layers.1.feed_forward.gate.weight"));
        assert_eq!(
            params["model.layers.1.feed_forward.experts.gate_up_proj"].shape(),
            &[4, 16, 16]
        );
        assert_eq!(
            params["model.layers.1.feed_forward.experts.down_proj"].shape(),
            &[4, 16, 8]
        );
        assert_eq!(
            params["model.layers.1.feed_forward.expert_bias"].shape(),
            &[4]
        );
        assert!(!params.contains_key("model.layers.1.feed_forward.w1.weight"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn dense_and_moe_prefill_decode_smoke() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let dense: ModelArgs = serde_json::from_value(dense_config()).unwrap();
        let moe: ModelArgs = serde_json::from_value(json!({
            "model_type": "lfm2_moe",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "layer_types": ["conv", "full_attention", "conv"],
            "moe_intermediate_size": 8,
            "num_dense_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "use_expert_bias": true
        }))
        .unwrap();

        for args in [dense, moe] {
            let mut model = super::Model::new(args, stream).unwrap();
            for (_, parameter) in model.parameters_mut().flatten() {
                *parameter = safemlx::ops::zeros_dtype(
                    &parameter.shape().to_vec(),
                    parameter.dtype(),
                    stream,
                )
                .unwrap();
            }
            let mut cache = model.new_cache();
            let prompt = safemlx::Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
            let parts = [crate::models::input::InputPart::text_token_ids(&prompt)];
            let logits = crate::models::common::generation::CausalLm::prefill_input_logits(
                &mut model,
                crate::models::input::ModelInput::new(&parts),
                &mut cache,
                stream,
            )
            .unwrap();
            assert_eq!(logits.shape(), &[1, 32]);
            assert_eq!(cache.offset(), 3);
            assert_eq!(logits.max(None, stream).unwrap().item::<f32>(stream), 0.0);

            let next = safemlx::Array::from_slice(&[4_u32], &[1, 1]);
            let logits = crate::models::common::generation::CausalLm::decode_logits(
                &mut model, &next, &mut cache, stream,
            )
            .unwrap();
            assert_eq!(logits.shape(), &[1, 32]);
            assert_eq!(cache.offset(), 4);
        }
    }
}
