//! Llama decoder-only model implementation.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use super::common::generation::sample;

use crate::{
    cache::{KeyValueCache, SlidingKeyValueCache},
    cache_residency::derive_prompt_cache_architecture_fingerprint,
    error::Error,
    inspection::ActivationObserver,
    models::{
        common::{
            self,
            attention::{
                apply_rope_and_update_cache, attention_probabilities, batch_seq, finish_attention,
                reshape_attention_projection,
            },
            generation::CausalLm,
            layers::SwiGluMlp,
            linear::project_logits_maybe_quantized,
        },
        input,
    },
    quantization::{AffineQuantization, WeightQuantization},
    utils::{
        create_attention_mask, create_sliding_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
    weights::{
        gguf_affine_configs, gguf_metadata, load_gguf_strict, load_safetensors_dir_lenient,
        load_safetensors_dir_quantized_strict, GgufTensorNames, StrictLoadConfig, StrictLoadReport,
    },
};

#[derive(Debug, Clone, Deserialize)]
/// Deserialized Llama `config.json` fields used by this loader.
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
    #[serde(default)]
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    #[serde(default)]
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    /// RoPE base frequency.
    pub rope_theta: f32,
    #[serde(default)]
    /// Whether RoPE uses adjacent-pair ordering instead of split-half ordering.
    pub rope_traditional: bool,
    #[serde(default)]
    /// Per-head attention dimension.
    pub head_dim: i32,
    #[serde(default = "default_true")]
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    #[serde(default)]
    /// Whether attention projection layers include bias terms.
    pub attention_bias: bool,
    #[serde(default)]
    /// Whether MLP projection layers include bias terms.
    pub mlp_bias: bool,
    /// Optional RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Optional total causal attention window, including the current token.
    #[serde(default)]
    pub sliding_window: Option<i32>,
    /// Preferred MLX-LM affine quantization metadata.
    #[serde(default)]
    pub quantization: Option<WeightQuantization>,
    /// Hugging Face-compatible alias emitted by MLX-LM converters.
    #[serde(default)]
    pub quantization_config: Option<WeightQuantization>,
    /// Optional exact weight names that use affine quantization.
    ///
    /// `None` preserves MLX-LM's model-wide quantization behavior. GGUF
    /// loading uses `Some` to represent files containing a mixture of packed
    /// and dense matrices.
    #[serde(skip)]
    pub quantized_weights: Option<HashSet<String>>,
    /// Exact affine settings for mixed GGUF tensors.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, AffineQuantization>>,
}

impl ModelArgs {
    pub(crate) fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization.or(self.quantization_config)
    }

    pub(crate) fn affine_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
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

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let rope_scaling = args.rope_scaling.as_ref().map_or_else(
        || "none".to_string(),
        |config| {
            let mut entries = config.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            entries
                .into_iter()
                .map(|(key, value)| {
                    let value = match value {
                        FloatOrString::Float(value) => format!("f32:{:08x}", value.to_bits()),
                        FloatOrString::String(value) => format!("string:{value}"),
                        FloatOrString::Bool(value) => format!("bool:{value}"),
                    };
                    format!("{key}={value}")
                })
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    let mut quantized_weights = args
        .quantized_weights
        .as_ref()
        .map(|weights| weights.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    quantized_weights.sort_unstable();
    let mut quantized_weight_configs = args
        .quantized_weight_configs
        .as_ref()
        .map(|configs| {
            configs
                .iter()
                .map(|(name, config)| format!("{name}={config:?}"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    quantized_weight_configs.sort_unstable();
    derive_prompt_cache_architecture_fingerprint(
        "llama",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("intermediate_size", args.intermediate_size.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("num_key_value_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            (
                "rms_norm_eps",
                format!("{:08x}", args.rms_norm_eps.to_bits()),
            ),
            ("vocab_size", args.vocab_size.to_string()),
            (
                "max_position_embeddings",
                args.max_position_embeddings.to_string(),
            ),
            ("rope_theta", format!("{:08x}", args.rope_theta.to_bits())),
            ("rope_traditional", args.rope_traditional.to_string()),
            ("rope_scaling", rope_scaling),
            ("sliding_window", format!("{:?}", args.sliding_window)),
            ("tie_word_embeddings", args.tie_word_embeddings.to_string()),
            ("attention_bias", args.attention_bias.to_string()),
            ("mlp_bias", args.mlp_bias.to_string()),
            ("quantization", format!("{:?}", args.weight_quantization())),
            ("quantized_weights", quantized_weights.join(";")),
            (
                "quantized_weight_configs",
                quantized_weight_configs.join(";"),
            ),
        ],
    )
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000.0
}

/// Internal input shared by Llama-compatible attention and decoder blocks.
pub struct AttentionInput<'a, C> {
    /// Hidden states with shape `[batch, sequence, hidden]`.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional mutable key/value cache.
    pub cache: Option<&'a mut C>,
    /// Generated total sliding-window size eligible for chunked prefill.
    pub generated_sliding_window: Option<i32>,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Llama attention layer.
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
    /// Rotary position embedding module.
    pub rope: RopeVariant,
}

impl Attention {
    /// Creates an unloaded attention layer from model arguments.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_prefix(args, None, stream)
    }

    fn new_for_layer(
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
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.q_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;
        let k_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.k_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;
        let v_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.v_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;
        let o_proj = common::linear::unloaded_maybe_quantized_linear(
            n_heads * head_dim,
            dim,
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.o_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
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
        let AttentionInput {
            x, mask, mut cache, ..
        } = input;

        let (batch, seq_len) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.q_proj"), &queries)?;
        let keys = self.k_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.k_proj"), &keys)?;
        let values = self.v_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.v_proj"), &values)?;

        let queries = reshape_attention_projection(queries, batch, seq_len, self.n_heads, stream)?;
        observer.observe(&format!("{prefix}.queries"), &queries)?;
        let keys = reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?;
        observer.observe(&format!("{prefix}.keys"), &keys)?;
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
        let AttentionInput {
            x,
            mask,
            mut cache,
            generated_sliding_window,
        } = input;

        let (B, L) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        let keys = self.k_proj.forward(x, stream)?;
        let values = self.v_proj.forward(x, stream)?;

        let queries = reshape_attention_projection(queries, B, L, self.n_heads, stream)?;
        let keys = reshape_attention_projection(keys, B, L, self.n_kv_heads, stream)?;
        let values = reshape_attention_projection(values, B, L, self.n_kv_heads, stream)?;
        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        let output = if let Some(window_size) = generated_sliding_window.filter(|_| L > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                window_size,
                position_offset,
                B,
                L,
                stream,
            )?
        } else {
            finish_attention(queries, keys, values, cache, self.scale, mask, B, L, stream)?
        };

        self.o_proj.forward(&output, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// Llama feed-forward block.
pub type Mlp = SwiGluMlp;

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Llama decoder block.
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
    /// Feed-forward layer.
    pub mlp: Mlp,

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
        let mlp_prefix = format!("model.layers.{layer_index}.mlp");
        let mlp = SwiGluMlp {
            gate_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.intermediate_size,
                args.mlp_bias,
                args.affine_quantization_for(&format!("{mlp_prefix}.gate_proj.weight")),
                stream,
            )?,
            down_proj: common::linear::unloaded_maybe_quantized_linear(
                args.intermediate_size,
                args.hidden_size,
                args.mlp_bias,
                args.affine_quantization_for(&format!("{mlp_prefix}.down_proj.weight")),
                stream,
            )?,
            up_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.intermediate_size,
                args.mlp_bias,
                args.affine_quantization_for(&format!("{mlp_prefix}.up_proj.weight")),
                stream,
            )?,
        };
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
        let AttentionInput {
            x,
            mask,
            cache,
            generated_sliding_window,
        } = input;

        observer.observe(&format!("{prefix}.input"), x)?;
        observer.observe(&format!("{prefix}.residual_before_attention"), x)?;
        let normed = self.input_layernorm.forward(x, stream)?;
        observer.observe(&format!("{prefix}.input_layernorm"), &normed)?;

        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
            generated_sliding_window,
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

        observer.observe(&format!("{prefix}.residual_before_mlp"), &h)?;
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &post_normed)?;
        let r = self.mlp.forward_with_observer(
            &post_normed,
            stream,
            &format!("{prefix}.mlp"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.mlp_output"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_mlp"), &r)?;
        let output = h.add(r, stream)?;
        let output = observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output);
        observer.observe(&format!("{prefix}.output"), &output)?;
        observer.observe(&format!("{prefix}.residual_after_mlp"), &output)?;
        Ok(output)
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
        let AttentionInput {
            x,
            mask,
            cache,
            generated_sliding_window,
        } = input;

        let normed = self.input_layernorm.forward(x, stream)?;
        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
            generated_sliding_window,
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
/// Llama transformer body without the language-model head.
pub struct ResidentDecoder {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Optional total causal attention window.
    pub sliding_window: Option<i32>,

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

impl ResidentDecoder {
    /// Creates an unloaded Llama transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.affine_quantization_for("model.embed_tokens.weight"),
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
            sliding_window: args.sliding_window,
            embed_tokens,
            layers,
            norm,
        })
    }

    fn attention_mask<C>(
        &self,
        h: &Array,
        cache: &[Option<C>],
        stream: &Stream,
    ) -> Result<Option<Array>, Exception>
    where
        C: KeyValueCache,
    {
        if let Some(window_size) = self.sliding_window {
            return create_sliding_attention_mask(h, cache, window_size, stream);
        }

        match create_attention_mask(h, cache, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Ok(Some(mask)),
            Some(AttentionMask::Causal) => Err(Exception::custom(
                "Llama-compatible decoders require an explicit attention mask",
            )),
            None => Ok(None),
        }
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
            None => self.attention_mask(&h, cache, stream)?,
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
                generated_sliding_window: None,
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
}

/// Input for a Llama forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Mutable per-layer key/value cache.
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for ResidentDecoder
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

        let (mask, generated_sliding_window) = match mask {
            Some(mask) => (Some(mask.clone()), None),
            None if self.sliding_window.is_some() && h.shape()[1] > 1 => {
                (None, self.sliding_window)
            }
            None => (self.attention_mask(&h, cache, stream)?, None),
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
                generated_sliding_window,
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
/// Llama causal language model.
pub struct ResidentModel {
    /// Model configuration.
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: ResidentDecoder,

    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl ResidentModel {
    /// Creates an unloaded Llama causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let model = ResidentDecoder::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.affine_quantization_for("lm_head.weight"),
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

    /// Returns the configured total sliding-window size, if any.
    pub fn sliding_window(&self) -> Option<i32> {
        self.args.sliding_window
    }

    /// Creates bounded per-layer caches for a sliding-window configuration.
    pub fn new_sliding_cache(&self) -> Vec<Option<SlidingKeyValueCache>> {
        let window_size = self
            .args
            .sliding_window
            .expect("new_sliding_cache requires a sliding-window model");
        (0..self.args.num_hidden_layers)
            .map(|_| Some(SlidingKeyValueCache::new(window_size)))
            .collect()
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
}

impl<C> Module<ModelInput<'_, C>> for ResidentModel
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
        <ResidentDecoder as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

/// Loads `tokenizer.json` from a Llama model directory.
pub fn load_llama_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads and normalizes Llama model arguments from `config.json`.
pub fn get_llama_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;
    normalize_model_args(model_args)
}

fn normalize_model_args(mut model_args: ModelArgs) -> Result<ModelArgs, Error> {
    if model_args.num_key_value_heads == 0 {
        model_args.num_key_value_heads = model_args.num_attention_heads;
    }
    if model_args.head_dim == 0 {
        if model_args.num_attention_heads <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "num_attention_heads must be positive, got {}",
                model_args.num_attention_heads
            )));
        }
        model_args.head_dim = model_args.hidden_size / model_args.num_attention_heads;
    }
    if model_args.max_position_embeddings == 0 {
        model_args.max_position_embeddings = 2048;
    }

    validate_model_args(&model_args)?;
    Ok(model_args)
}

fn validate_model_args(model_args: &ModelArgs) -> Result<(), Error> {
    if !matches!(model_args.model_type.as_str(), "llama" | "mistral") {
        return Err(Error::UnsupportedModelType(model_args.model_type.clone()));
    }
    for (name, value) in [
        ("hidden_size", model_args.hidden_size),
        ("num_hidden_layers", model_args.num_hidden_layers),
        ("intermediate_size", model_args.intermediate_size),
        ("num_attention_heads", model_args.num_attention_heads),
        ("num_key_value_heads", model_args.num_key_value_heads),
        ("vocab_size", model_args.vocab_size),
        (
            "max_position_embeddings",
            model_args.max_position_embeddings,
        ),
        ("head_dim", model_args.head_dim),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "{name} must be positive, got {value}"
            )));
        }
    }
    if model_args.num_attention_heads % model_args.num_key_value_heads != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            model_args.num_attention_heads, model_args.num_key_value_heads
        )));
    }
    if let Some(window_size) = model_args.sliding_window {
        if window_size <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "sliding_window must be positive, got {window_size}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    let args = serde_json::from_value::<ModelArgs>(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Llama-compatible config: {error}"))
    })?;
    normalize_model_args(args).map(|_| ())
}

pub(crate) struct LoadedLlamaGguf {
    pub(crate) model: ResidentModel,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) struct PreparedLlamaGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads a Llama-compatible GGUF checkpoint, including Mistral.
///
/// Dense tensors and GGUF Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0 tensors are
/// supported. Quantized formats are consumed in the packed affine
/// representation emitted by MLX's GGUF loader.
pub fn load_llama_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ResidentModel, Error> {
    Ok(load_llama_gguf_with_metadata(gguf_file, stream, weights_stream)?.model)
}

pub(crate) fn load_llama_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedLlamaGguf, Error> {
    let gguf_file = gguf_file.as_ref();
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    load_llama_gguf_checkpoint(&checkpoint, metadata, None, stream, weights_stream)
}

pub(crate) fn load_llama_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedLlamaGguf, Error> {
    let prepared =
        prepare_llama_gguf_checkpoint(checkpoint, &metadata, quantization, weights_stream)?;
    let mut model = ResidentModel::new(prepared.args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    load_gguf_strict(
        &mut model,
        checkpoint,
        quantization.map(|value| (value, stream)),
        &config,
        &mut report,
        |name, value| Ok((translate_gguf_weight_name(&name), value)),
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;

    Ok(LoadedLlamaGguf {
        model,
        eos_token_ids: prepared.eos_token_ids,
    })
}

pub(crate) fn prepare_llama_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    weights_stream: &Stream,
) -> Result<PreparedLlamaGguf, Error> {
    let architecture = gguf_string(&metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "llama" | "mistral") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports llama and mistral"
        )));
    }

    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let mut args = llama_args_from_gguf(checkpoint, metadata, &architecture, weights_stream)?;
    let quantized_weight_configs = gguf_affine_configs(checkpoint, translate_gguf_weight_name)?;
    if let Some(quantization) = quantization {
        args.quantized_weights = None;
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
    } else {
        args.quantized_weights = Some(quantized_weight_configs.keys().cloned().collect());
        args.quantization = None;
        args.quantized_weight_configs = Some(quantized_weight_configs);
    }

    let eos_token_ids = gguf_optional_i64(metadata, "tokenizer.ggml.eos_token_id", weights_stream)?
        .and_then(|value| u32::try_from(value).ok())
        .into_iter()
        .collect();
    Ok(PreparedLlamaGguf {
        args,
        eos_token_ids,
    })
}

fn llama_args_from_gguf(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
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
    let rope_theta = gguf_optional_f32(metadata, &key("rope.freq_base"), stream)?
        .unwrap_or_else(default_rope_theta);
    let rope_scaling = gguf_rope_scaling(metadata, architecture, stream)?;
    let sliding_window = gguf_optional_i64(metadata, &key("attention.sliding_window"), stream)?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("GGUF sliding-window size exceeds i32".into()))?
        .filter(|window| *window != 0);
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
        model_type: architecture.to_string(),
        hidden_size,
        num_hidden_layers: gguf_i32(metadata, &key("block_count"), stream)?,
        intermediate_size: gguf_i32(metadata, &key("feed_forward_length"), stream)?,
        num_attention_heads,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"), stream)?,
        vocab_size,
        num_key_value_heads,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"), stream)?,
        rope_theta,
        rope_traditional: true,
        head_dim,
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        attention_bias: arrays.any_gguf_tensor(|name| {
            name.starts_with("blk.")
                && matches!(
                    name.rsplit_once('.'),
                    Some((prefix, "bias")) if prefix.ends_with("attn_q")
                        || prefix.ends_with("attn_k")
                        || prefix.ends_with("attn_v")
                        || prefix.ends_with("attn_output")
                )
        }),
        mlp_bias: arrays.any_gguf_tensor(|name| {
            name.starts_with("blk.")
                && matches!(
                    name.rsplit_once('.'),
                    Some((prefix, "bias")) if prefix.ends_with("ffn_gate")
                        || prefix.ends_with("ffn_down")
                        || prefix.ends_with("ffn_up")
                )
        }),
        rope_scaling,
        sliding_window,
        quantization: None,
        quantization_config: None,
        quantized_weights: None,
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
            let scaling_factor_key = format!("{architecture}.rope.scaling.factor");
            let factor =
                gguf_optional_f32(metadata, &scaling_factor_key, stream)?.ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "linear GGUF RoPE scaling is missing {scaling_factor_key}"
                    ))
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
            "GGUF RoPE scaling type {other:?} is not supported by the initial GGUF loader"
        ))),
    }
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    name.replace("blk.", "model.layers.")
        .replace("ffn_gate", "mlp.gate_proj")
        .replace("ffn_down", "mlp.down_proj")
        .replace("ffn_up", "mlp.up_proj")
        .replace("attn_q", "self_attn.q_proj")
        .replace("attn_k", "self_attn.k_proj")
        .replace("attn_v", "self_attn.v_proj")
        .replace("attn_output", "self_attn.o_proj")
        .replace("attn_norm", "input_layernorm")
        .replace("ffn_norm", "post_attention_layernorm")
        .replace("token_embd", "model.embed_tokens")
        .replace("output_norm", "model.norm")
        .replace("output", "lm_head")
}

fn gguf_string(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<String, Error> {
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

fn gguf_i32(
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

/// Loads a Llama model and safetensors weights from a model directory.
pub fn load_resident_llama_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ResidentModel, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_llama_model_args(model_dir)?;
    let mut model = ResidentModel::new(model_args, stream)?;

    load_safetensors_dir_lenient(&mut model, model_dir, weights_stream)?;
    model.copy_to_stream(stream)?;

    Ok(model)
}

/// Loads a dense Llama checkpoint while quantizing matrices tensor-by-tensor.
pub fn load_resident_llama_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ResidentModel, Error> {
    let model_dir = model_dir.as_ref();
    let mut model_args = get_llama_model_args(model_dir)?;
    if !crate::quantization::should_quantize_on_load(
        "Llama",
        model_args.weight_quantization(),
        quantization,
    )? {
        return load_resident_llama_model(model_dir, stream, weights_stream);
    }
    model_args.quantization = Some(quantization);
    let mut model = ResidentModel::new(model_args, stream)?;
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

impl<C> CausalLm<Vec<Option<C>>> for ResidentModel
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

/// Llama token generation iterator.
pub type Generate<'a, C, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, ResidentModel, Vec<Option<C>>, S>;

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        env::home_dir,
        fs,
    };

    use lazy_static::lazy_static;
    use safemlx::{
        ops::indexing::{NewAxis, TryIndexOp},
        ops::{GgufMetadataArray, GgufMetadataValue},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::ConcatKeyValueCache,
        models::llama::{load_llama_tokenizer, load_resident_llama_model},
        quantization::AffineQuantization,
    };

    #[test]
    fn normalizes_hermes_mistral_config() {
        let args: super::ModelArgs = serde_json::from_value(serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32032,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let args = super::normalize_model_args(args).unwrap();

        assert_eq!(args.model_type, "mistral");
        assert_eq!(args.head_dim, 128);
        assert_eq!(args.num_key_value_heads, 8);
        assert_eq!(args.sliding_window, Some(4096));
    }

    #[test]
    fn prompt_cache_architecture_fingerprint_is_derived_from_rope_configuration() {
        let mut args: super::ModelArgs = serde_json::from_value(serde_json::json!({
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 0.00001,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rope_scaling": {"factor": 2.0, "rope_type": "linear"}
        }))
        .unwrap();
        let first = super::prompt_cache_architecture_fingerprint(&args);
        assert_eq!(first, super::prompt_cache_architecture_fingerprint(&args));
        args.rope_theta = 500_000.0;
        let changed = super::prompt_cache_architecture_fingerprint(&args);
        assert_ne!(first, changed);
    }

    #[test]
    fn preserves_mistral_small_explicit_head_dimension() {
        let args: super::ModelArgs = serde_json::from_value(serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "intermediate_size": 32768,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 0.00001,
            "vocab_size": 131072,
            "max_position_embeddings": 32768,
            "rope_theta": 100000000.0,
            "sliding_window": null,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let args = super::normalize_model_args(args).unwrap();

        assert_eq!(args.head_dim, 128);
        assert_eq!(args.hidden_size, 5120);
        assert_eq!(args.sliding_window, None);
    }

    #[test]
    fn translates_gguf_llama_weight_names() {
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.attn_q.weight"),
            "model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.1.ffn_down.scales"),
            "model.layers.1.mlp.down_proj.scales"
        );
        assert_eq!(
            super::translate_gguf_weight_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            super::translate_gguf_weight_name("output_norm.weight"),
            "model.norm.weight"
        );
    }

    #[test]
    fn loads_dense_mistral_from_synthetic_gguf_checkpoint() {
        use safemlx::module::ModuleParameters;

        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let source = super::ResidentModel::new(
            super::ModelArgs {
                model_type: "mistral".into(),
                hidden_size: 32,
                num_hidden_layers: 1,
                intermediate_size: 64,
                num_attention_heads: 1,
                rms_norm_eps: 1e-5,
                vocab_size: 32,
                num_key_value_heads: 1,
                max_position_embeddings: 128,
                rope_theta: 10_000.0,
                rope_traditional: true,
                head_dim: 32,
                tie_word_embeddings: true,
                attention_bias: false,
                mlp_bias: false,
                rope_scaling: None,
                sliding_window: Some(16),
                quantization: None,
                quantization_config: None,
                quantized_weights: None,
                quantized_weight_configs: None,
            },
            stream,
        )
        .unwrap();
        let arrays: HashMap<String, Array> = source
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| {
                let name = name
                    .replace("model.layers.", "blk.")
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
                GgufMetadataValue::String("mistral".into()),
            ),
            (
                "mistral.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            ("mistral.block_count".into(), GgufMetadataValue::Uint32(1)),
            (
                "mistral.feed_forward_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            (
                "mistral.attention.head_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "mistral.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "mistral.attention.key_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "mistral.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-5),
            ),
            (
                "mistral.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(16),
            ),
            (
                "mistral.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "mistral.rope.freq_base".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec!["token".into(); 32])),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::Uint32(2),
            ),
        ]);

        let fixture = crate::test_utils::SyntheticGguf::dense(&arrays, &metadata);
        let loaded = super::load_llama_gguf_with_metadata(fixture.path(), stream, stream).unwrap();

        assert_eq!(loaded.model.model_type(), "mistral");
        assert_eq!(loaded.model.sliding_window(), Some(16));
        assert_eq!(loaded.eos_token_ids, vec![2]);

        let gpu = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let checkpoint = safemlx::ops::GgufCheckpoint::open(fixture.path()).unwrap();
        let metadata = crate::weights::gguf_metadata(&checkpoint);
        let quantized = super::load_llama_gguf_checkpoint(
            &checkpoint,
            metadata,
            Some(crate::quantization::WeightQuantization::MxFp4),
            gpu.stream(),
            stream,
        )
        .unwrap();
        let params = quantized.model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert!(!params.contains_key("model.layers.0.self_attn.q_proj.biases"));
        assert!(params.contains_key("model.embed_tokens.scales"));
        assert!(!params.contains_key("model.embed_tokens.biases"));
    }

    #[test]
    fn mixed_quantization_builds_only_selected_llama_parameters() {
        use safemlx::module::ModuleParameters;

        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let selected = HashSet::from(["model.layers.0.self_attn.q_proj.weight".to_string()]);
        let args = super::ModelArgs {
            model_type: "llama".into(),
            hidden_size: 32,
            num_hidden_layers: 1,
            intermediate_size: 32,
            num_attention_heads: 1,
            rms_norm_eps: 1e-5,
            vocab_size: 32,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            rope_traditional: true,
            head_dim: 32,
            tie_word_embeddings: true,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            sliding_window: None,
            quantization: Some(AffineQuantization::new(32, 4).unwrap().into()),
            quantization_config: None,
            quantized_weights: Some(selected),
            quantized_weight_configs: None,
        };
        let model = super::ResidentModel::new(args, ctx.stream()).unwrap();
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert!(params.contains_key("model.layers.0.self_attn.k_proj.weight"));
        assert!(!params.contains_key("model.layers.0.self_attn.k_proj.scales"));
    }

    /// Resolve the HuggingFace cache directory to the actual snapshot path.
    /// The structure is:
    ///   models--<org>--<name>/
    ///     refs/
    ///       main  (contains the commit hash)
    ///     snapshots/
    ///       <commit_hash>/  (actual model files)
    fn resolve_hf_cache_dir(model_cache_dir: &str) -> String {
        let refs_main = std::path::Path::new(model_cache_dir)
            .join("refs")
            .join("main");
        let commit_hash = fs::read_to_string(&refs_main)
            .unwrap_or_default()
            .trim()
            .to_string();
        std::path::Path::new(model_cache_dir)
            .join("snapshots")
            .join(commit_hash)
            .to_string_lossy()
            .into_owned()
    }

    lazy_static! {
        static ref CACHED_TEST_MODEL_DIR: String = {
            let cache_dir = home_dir()
                .map(|p| {
                    p.join(".cache")
                        .join("huggingface")
                        .join("hub")
                        .join("models--meta-llama--Llama-3.2-1B-Instruct")
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default();

            resolve_hf_cache_dir(&cache_dir)
        };
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_llama_model() {
        use safemlx::module::ModuleParameters;

        let model_dir = CACHED_TEST_MODEL_DIR.as_str();
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let model_args = super::get_llama_model_args(model_dir).unwrap();
        let model = super::ResidentModel::new(model_args, stream).unwrap();

        // Print some model parameter keys
        let params = model.parameters().flatten();
        let mut param_keys: Vec<_> = params.keys().map(|k| k.to_string()).collect();
        param_keys.sort();
        println!("=== Model parameter keys (first 20) ===");
        for key in param_keys.iter().take(20) {
            println!("  {key}");
        }

        // Print some safetensor keys
        let weights_path = std::path::Path::new(model_dir).join("model.safetensors");
        let loaded = safemlx::Array::load_safetensors(&weights_path, stream).unwrap();
        let mut weight_keys: Vec<_> = loaded.keys().map(|k| k.to_string()).collect();
        weight_keys.sort();
        println!("=== Safetensor weight keys (first 20) ===");
        for key in weight_keys.iter().take(20) {
            println!("  {key}");
        }

        // Find unmatched keys
        let param_set: std::collections::HashSet<_> = param_keys.iter().collect();
        let weight_set: std::collections::HashSet<_> = weight_keys.iter().collect();
        let unloaded: Vec<_> = weight_set.difference(&param_set).collect();
        let missing: Vec<_> = param_set.difference(&weight_set).collect();
        println!(
            "=== Weight keys NOT in model params ({}) ===",
            unloaded.len()
        );
        for key in unloaded.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "=== Model param keys NOT in weights ({}) ===",
            missing.len()
        );
        for key in missing.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "Total model params: {}, Total weight keys: {}",
            param_keys.len(),
            weight_keys.len()
        );
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();

        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_llama_with_concat_cache() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();
        let mut model =
            load_resident_llama_model(CACHED_TEST_MODEL_DIR.as_str(), stream, weights_stream)
                .unwrap();

        let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids())
            .try_index_device(NewAxis, stream)
            .unwrap();
        let mut cache = Vec::new();

        let eos_token_id = 128001u32;
        let eot_token_id = 128009u32;

        let mut token_ids = Vec::new();
        let input_parts = [crate::models::input::InputPart::text_token_ids(
            &prompt_tokens,
        )];
        let input = crate::models::input::ModelInput::new(&input_parts);
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model, &mut cache, 0.0, input, None, stream,
        );
        for (token, _ntoks) in generate.zip(0..50) {
            let token = token.unwrap();
            eval([&token]).unwrap();
            let token_id = token.item::<u32>(&stream);
            print!("[{}]", token_id);
            if token_id == eos_token_id || token_id == eot_token_id {
                break;
            }
            token_ids.push(token_id);
        }
        println!();

        let output = tokenizer.decode(&token_ids, true).unwrap();
        println!("Response: {output}");
        println!("------");
    }
}
