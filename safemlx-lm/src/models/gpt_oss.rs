//! OpenAI GPT-OSS decoder-only mixture-of-experts implementation.

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        arange, clip, gather_grouped_rows, gather_qmm_with_mode,
        indexing::{IntoStrideBy, TryIndexOp},
        sigmoid, topk_route_plan, QuantizationMode,
    },
    Array, Dtype, Stream,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache, SlidingKeyValueCache},
    error::Error,
    models::{common, common::CausalLm, input},
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    weights::{load_safetensors_dir_strict, StrictLoadConfig, StrictLoadReport},
};

fn default_head_dim() -> i32 {
    64
}

fn default_sliding_window() -> i32 {
    128
}

fn default_rope_theta() -> f32 {
    150_000.0
}

fn default_swiglu_limit() -> f32 {
    7.0
}

/// GPT-OSS checkpoint quantization metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct MxFp4Config {
    /// Must be `mxfp4` for the published GPT-OSS checkpoints.
    pub quant_method: String,
}

/// Deserialized GPT-OSS `config.json` fields used by this loader.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    /// Architecture identifier.
    pub model_type: String,
    /// Transformer width.
    pub hidden_size: i32,
    /// Expert hidden width.
    pub intermediate_size: i32,
    /// Number of transformer blocks.
    pub num_hidden_layers: i32,
    /// Query attention heads.
    pub num_attention_heads: i32,
    /// Key/value attention heads.
    pub num_key_value_heads: i32,
    /// Attention head width.
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Number of local routed experts.
    pub num_local_experts: i32,
    /// Experts selected for each token.
    pub num_experts_per_tok: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Sliding attention cache width.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Maximum configured context length.
    pub max_position_embeddings: i32,
    /// RoPE base.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// YaRN scaling configuration.
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Per-layer `sliding_attention` or `full_attention` selection.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Published checkpoint MXFP4 metadata.
    pub quantization_config: MxFp4Config,
    /// GPT-OSS clipped SwiGLU limit.
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
}

impl ModelArgs {
    fn validate(&self) -> Result<(), Error> {
        if self.model_type != "gpt_oss" {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS loader requires model_type gpt_oss, got {:?}",
                self.model_type
            )));
        }
        if self.quantization_config.quant_method != "mxfp4" {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS expert weights require quant_method mxfp4, got {:?}",
                self.quantization_config.quant_method
            )));
        }
        if self.hidden_size % 32 != 0 || self.intermediate_size % 32 != 0 {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS MXFP4 projection dimensions must be divisible by 32".into(),
            ));
        }
        if self.num_attention_heads * self.head_dim <= 0
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS attention head configuration is invalid".into(),
            ));
        }
        if self.num_experts_per_tok <= 0 || self.num_experts_per_tok > self.num_local_experts {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS expert routing configuration is invalid".into(),
            ));
        }
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers as usize
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if self
            .effective_layer_types()
            .iter()
            .any(|kind| !matches!(kind.as_str(), "sliding_attention" | "full_attention"))
        {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS layer_types entries must be sliding_attention or full_attention".into(),
            ));
        }
        Ok(())
    }

    fn effective_layer_types(&self) -> Vec<String> {
        if !self.layer_types.is_empty() {
            return self.layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|index| {
                if index % 2 == 0 {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
                .to_string()
            })
            .collect()
    }
}

/// Validates a parsed GPT-OSS configuration.
pub fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    let args: ModelArgs = serde_json::from_value(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid gpt_oss config: {error}"))
    })?;
    args.validate()
}

/// One attention layer with learned sink logits.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    #[param]
    /// Learned per-query-head attention sink.
    pub sinks: Param<Array>,
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
    /// Attention output projection.
    pub o_proj: nn::Linear,
    #[param]
    rope: RopeVariant,
}

impl Attention {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            head_dim: args.head_dim,
            scale: 1.0 / (args.head_dim as f32).sqrt(),
            sinks: Param::<Array>::unloaded(&[args.num_attention_heads], Dtype::Float32, stream)?,
            q_proj: nn::Linear::unloaded(
                args.hidden_size,
                args.num_attention_heads * args.head_dim,
                true,
                Dtype::Float32,
                stream,
            )?,
            k_proj: nn::Linear::unloaded(
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
                true,
                Dtype::Float32,
                stream,
            )?,
            v_proj: nn::Linear::unloaded(
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
                true,
                Dtype::Float32,
                stream,
            )?,
            o_proj: nn::Linear::unloaded(
                args.num_attention_heads * args.head_dim,
                args.hidden_size,
                true,
                Dtype::Float32,
                stream,
            )?,
            rope: initialize_rope(
                args.head_dim,
                args.rope_theta,
                false,
                &args.rope_scaling,
                args.max_position_embeddings,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, length) = (shape[0], shape[1]);
        let project = |projection: Array, heads: i32| {
            projection
                .reshape(&[batch, length, heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)
        };
        let mut q = project(self.q_proj.forward(x, stream)?, self.n_heads)?;
        let mut k = project(self.k_proj.forward(x, stream)?, self.n_kv_heads)?;
        let v = project(self.v_proj.forward(x, stream)?, self.n_kv_heads)?;
        let offset = cache.offset();
        q = self.rope.forward(nn::RopeInput { x: &q, offset }, stream)?;
        k = self.rope.forward(nn::RopeInput { x: &k, offset }, stream)?;
        let (k, v) = cache.update_and_fetch(k, v, stream)?;
        let attended = safemlx::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            self.scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            self.sinks.as_ref(),
            stream,
        )?;
        self.o_proj.forward(
            &attended
                .transpose_axes(&[0, 2, 1, 3], stream)?
                .reshape(&[batch, length, -1], stream)?,
            stream,
        )
    }
}

/// Checkpoint-native combined MXFP4 expert tensors.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Experts {
    num_experts: i32,
    hidden_size: i32,
    intermediate_size: i32,
    limit: f32,
    #[param]
    /// Combined alternating gate/up packed FP4 blocks.
    pub gate_up_proj_blocks: Param<Array>,
    #[param]
    /// Combined alternating gate/up E8M0 scales.
    pub gate_up_proj_scales: Param<Array>,
    #[param]
    /// Combined alternating gate/up projection bias.
    pub gate_up_proj_bias: Param<Array>,
    #[param]
    /// Packed down-projection FP4 blocks.
    pub down_proj_blocks: Param<Array>,
    #[param]
    /// Down-projection E8M0 scales.
    pub down_proj_scales: Param<Array>,
    #[param]
    /// Down-projection bias.
    pub down_proj_bias: Param<Array>,
}

impl Experts {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_experts: args.num_local_experts,
            hidden_size: args.hidden_size,
            intermediate_size: args.intermediate_size,
            limit: args.swiglu_limit,
            gate_up_proj_blocks: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    2 * args.intermediate_size,
                    args.hidden_size / 32,
                    16,
                ],
                Dtype::Uint8,
                stream,
            )?,
            gate_up_proj_scales: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    2 * args.intermediate_size,
                    args.hidden_size / 32,
                ],
                Dtype::Uint8,
                stream,
            )?,
            gate_up_proj_bias: Param::<Array>::unloaded(
                &[args.num_local_experts, 2 * args.intermediate_size],
                Dtype::Float32,
                stream,
            )?,
            down_proj_blocks: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    args.hidden_size,
                    args.intermediate_size / 32,
                    16,
                ],
                Dtype::Uint8,
                stream,
            )?,
            down_proj_scales: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    args.hidden_size,
                    args.intermediate_size / 32,
                ],
                Dtype::Uint8,
                stream,
            )?,
            down_proj_bias: Param::<Array>::unloaded(
                &[args.num_local_experts, args.hidden_size],
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn mxfp4_linear(
        input: &Array,
        blocks: &Array,
        scales: &Array,
        projection_bias: &Array,
        expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let routes = input.dim(0);
        let output_size = blocks.dim(1);
        let packed = blocks
            .view::<u32>(stream)?
            .reshape(&[blocks.dim(0), output_size, -1], stream)?;
        let lhs_indices = arange::<i32, u32>(0, routes, 1, stream)?;
        let output = gather_qmm_with_mode(
            input.reshape(&[routes, 1, input.dim(-1)], stream)?,
            packed,
            scales,
            None,
            Some(&lhs_indices),
            Some(expert_ids),
            true,
            32,
            4,
            true,
            QuantizationMode::MxFp4,
            stream,
        )?
        .reshape(&[routes, output_size], stream)?;
        output.add(projection_bias.take_axis(expert_ids, 0, stream)?, stream)
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = hidden_states.dim(0);
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let routed = gather_grouped_rows(hidden_states, &plan, stream)?;

        let gate_up = Self::mxfp4_linear(
            &routed,
            self.gate_up_proj_blocks.as_ref(),
            self.gate_up_proj_scales.as_ref(),
            self.gate_up_proj_bias.as_ref(),
            &plan.sorted_group_ids,
            stream,
        )?;
        let gate = gate_up.try_index_device((.., (0..).stride_by(2)), stream)?;
        let linear = gate_up.try_index_device((.., (1..).stride_by(2)), stream)?;
        let gate = clip(gate, ((), self.limit), stream)?;
        let linear = clip(linear, (-self.limit, self.limit), stream)?;
        let activated = gate
            .multiply(
                sigmoid(gate.multiply(Array::from_f32(1.702), stream)?, stream)?,
                stream,
            )?
            .multiply(linear.add(Array::from_f32(1.0), stream)?, stream)?;
        debug_assert_eq!(activated.dim(-1), self.intermediate_size);

        let output = Self::mxfp4_linear(
            &activated,
            self.down_proj_blocks.as_ref(),
            self.down_proj_scales.as_ref(),
            self.down_proj_bias.as_ref(),
            &plan.sorted_group_ids,
            stream,
        )?;
        debug_assert_eq!(output.dim(-1), self.hidden_size);
        common::weighted_route_sum(output, top_k_weights, &plan, tokens, stream)
    }
}

/// GPT-OSS sparse MoE block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Mlp {
    top_k: i32,
    #[param]
    /// Biased expert router, matching the checkpoint's `mlp.router` tree.
    pub router: nn::Linear,
    #[param]
    /// Checkpoint-native routed expert bank.
    pub experts: Experts,
}

impl Mlp {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            top_k: args.num_experts_per_tok,
            router: nn::Linear::unloaded(
                args.hidden_size,
                args.num_local_experts,
                true,
                Dtype::Float32,
                stream,
            )?,
            experts: Experts::new(args, stream)?,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let logits = self.router.forward(&flat, stream)?;
        let (indices, weights) = common::top_k_softmax_routing(&logits, self.top_k, stream)?;
        self.experts
            .forward(&flat, &indices, &weights, stream)?
            .reshape(shape, stream)
    }
}

/// One GPT-OSS decoder block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerBlock {
    #[param]
    /// Self-attention.
    pub self_attn: Attention,
    #[param]
    /// Sparse MoE feed-forward block.
    pub mlp: Mlp,
    #[param]
    /// Pre-attention RMSNorm.
    pub input_layernorm: nn::RmsNorm,
    #[param]
    /// Pre-MoE RMSNorm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: Attention::new(args, stream)?,
            mlp: Mlp::new(args, stream)?,
            input_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            post_attention_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x, stream)?;
        let hidden = x.add(
            self.self_attn.forward(&normed, mask, cache, stream)?,
            stream,
        )?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        hidden.add(self.mlp.forward(&normed, stream)?, stream)
    }
}

/// Per-layer cache matching alternating full and sliding attention.
#[derive(Debug, Clone)]
pub enum LayerCache {
    /// Unbounded full-attention cache.
    Full(ConcatKeyValueCache),
    /// Bounded sliding-attention cache.
    Sliding(SlidingKeyValueCache),
}

impl KeyValueCache for LayerCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Full(cache) => cache.offset(),
            Self::Sliding(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Full(cache) => cache.max_size(),
            Self::Sliding(cache) => cache.max_size(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Full(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Sliding(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }
}

/// Heterogeneous generation cache for GPT-OSS.
#[derive(Debug, Clone, Default)]
pub struct Cache {
    /// One cache per decoder block.
    pub layers: Vec<LayerCache>,
}

/// GPT-OSS transformer body.
#[derive(Debug, Clone, ModuleParameters)]
pub struct GptOssModel {
    layer_types: Vec<String>,
    sliding_window: i32,
    #[param]
    /// Token embedding table.
    pub embed_tokens: nn::Embedding,
    #[param]
    /// Decoder blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final RMSNorm.
    pub norm: nn::RmsNorm,
}

impl GptOssModel {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            layer_types: args.effective_layer_types(),
            sliding_window: args.sliding_window,
            embed_tokens: nn::Embedding::unloaded(
                args.vocab_size,
                args.hidden_size,
                Dtype::Float32,
                stream,
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|_| TransformerBlock::new(args, stream))
                .collect::<Result<_, _>>()?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn new_cache(&self) -> Cache {
        Cache {
            layers: self
                .layer_types
                .iter()
                .map(|kind| {
                    if kind == "sliding_attention" {
                        LayerCache::Sliding(SlidingKeyValueCache::new(self.sliding_window))
                    } else {
                        LayerCache::Full(ConcatKeyValueCache::new())
                    }
                })
                .collect(),
        }
    }

    fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
        }
        let mut hidden = self.embed_tokens.forward(inputs, stream)?;
        let length = hidden.dim(1);
        for (layer, layer_cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
            let offset = layer_cache.offset();
            let window = layer_cache.max_size();
            let needs_mask = length > 1 || window.is_some_and(|size| offset >= size);
            let mask = needs_mask
                .then(|| {
                    let max_past = window.map(|size| size - 1);
                    create_causal_mask(
                        length,
                        Some(offset.min(window.unwrap_or(offset))),
                        max_past,
                        None,
                        stream,
                    )
                })
                .transpose()?;
            hidden = layer.forward(&hidden, mask.as_ref(), layer_cache, stream)?;
        }
        self.norm.forward(&hidden, stream)
    }
}

/// GPT-OSS causal language model.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    #[param]
    /// Transformer body.
    pub model: GptOssModel,
    #[param]
    /// Untied output projection.
    pub lm_head: nn::Linear,
}

impl Model {
    /// Creates an unloaded GPT-OSS model with the native checkpoint parameter tree.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            model: GptOssModel::new(&args, stream)?,
            lm_head: nn::Linear::unloaded(
                args.hidden_size,
                args.vocab_size,
                false,
                Dtype::Float32,
                stream,
            )?,
            args,
        })
    }

    /// Returns the model architecture identifier.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Creates alternating full/sliding caches.
    pub fn new_cache(&self) -> Cache {
        self.model.new_cache()
    }

    fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward(inputs, cache, stream)?;
        self.lm_head.forward(&hidden, stream)
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
        self.forward(&tokens, cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }
}

/// GPT-OSS token generation iterator.
pub type Generate<'a, S = crate::sampler::DefaultSampler> = common::Generate<'a, Model, Cache, S>;

/// Reads GPT-OSS model arguments from `config.json`.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(file)?;
    args.validate()?;
    Ok(args)
}

/// Loads a GPT-OSS safetensors checkpoint strictly, without rewriting keys.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let mut model = Model::new(get_model_args(model_dir)?, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_strict(&mut model, model_dir, weights_stream, &config, &mut report)?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads `tokenizer.json` from a GPT-OSS model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{
        module::ModuleParameters,
        ops::{ones_dtype, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext,
    };

    use super::{Cache, Model, ModelArgs, MxFp4Config};
    use crate::utils::rope::FloatOrString;

    fn tiny_args() -> ModelArgs {
        ModelArgs {
            model_type: "gpt_oss".into(),
            hidden_size: 32,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 32,
            vocab_size: 32,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            rms_norm_eps: 1e-5,
            sliding_window: 8,
            max_position_embeddings: 128,
            rope_theta: 150_000.0,
            rope_scaling: Some(HashMap::from([
                ("rope_type".into(), FloatOrString::String("yarn".into())),
                ("factor".into(), FloatOrString::Float(2.0)),
                (
                    "original_max_position_embeddings".into(),
                    FloatOrString::Float(64.0),
                ),
                ("beta_fast".into(), FloatOrString::Float(32.0)),
                ("beta_slow".into(), FloatOrString::Float(1.0)),
                ("truncate".into(), FloatOrString::Bool(false)),
            ])),
            layer_types: vec!["sliding_attention".into(), "full_attention".into()],
            quantization_config: MxFp4Config {
                quant_method: "mxfp4".into(),
            },
            swiglu_limit: 7.0,
        }
    }

    #[test]
    fn published_config_shape_is_accepted() {
        let value = serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "rms_norm_eps": 1e-5,
            "sliding_window": 128,
            "max_position_embeddings": 131072,
            "rope_theta": 150000,
            "rope_scaling": {
                "rope_type": "yarn", "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0, "beta_slow": 1.0
            },
            "layer_types": std::iter::repeat(["sliding_attention", "full_attention"])
                .take(12).flatten().collect::<Vec<_>>(),
            "quantization_config": {"quant_method": "mxfp4"}
        });
        let args: ModelArgs = serde_json::from_value(value).unwrap();
        args.validate().unwrap();
        assert_eq!(args.effective_layer_types().len(), 24);
    }

    #[test]
    fn parameter_tree_matches_native_checkpoint_names() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let model = Model::new(tiny_args(), ctx.stream()).unwrap();
        let parameters = model.parameters().flatten();
        for key in [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.sinks",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.mlp.router.weight",
            "model.layers.0.mlp.router.bias",
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.0.mlp.experts.gate_up_proj_scales",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj_blocks",
            "model.layers.0.mlp.experts.down_proj_scales",
            "model.layers.0.mlp.experts.down_proj_bias",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            assert!(parameters.contains_key(key), "missing parameter key {key}");
        }
    }

    #[test]
    fn zero_weight_forward_exercises_mxfp4_and_mixed_cache() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut model = Model::new(tiny_args(), stream).unwrap();
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if name.ends_with("_scales") {
                Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
            } else if name.ends_with("layernorm.weight") || name.as_ref() == "model.norm.weight" {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
        let tokens = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let logits = model
            .forward(&tokens, &mut Cache::default(), stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);
        assert_eq!(logits.max(None, stream).unwrap().item::<f32>(stream), 0.0);
    }
}
