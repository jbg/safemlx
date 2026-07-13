//! Llama decoder-only model implementation.

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::TryIndexOp,
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use super::common::sample;

use crate::{
    cache::KeyValueCache,
    error::Error,
    inspection::ActivationObserver,
    models::{
        common::{
            self, apply_rope_and_update_cache, attention_probabilities, batch_seq,
            finish_attention, project_logits_maybe_quantized, reshape_attention_projection,
            AttentionInput, CausalLm, SwiGluMlp,
        },
        input,
    },
    utils::rope::{initialize_rope, FloatOrString, RopeVariant},
    weights::load_safetensors_dir_lenient,
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
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000.0
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
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::Linear::unloaded(
            dim,
            n_heads * head_dim,
            args.attention_bias,
            Dtype::Float32,
            stream,
        )?;
        let k_proj = nn::Linear::unloaded(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            Dtype::Float32,
            stream,
        )?;
        let v_proj = nn::Linear::unloaded(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            Dtype::Float32,
            stream,
        )?;
        let o_proj = nn::Linear::unloaded(
            n_heads * head_dim,
            dim,
            args.attention_bias,
            Dtype::Float32,
            stream,
        )?;

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
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
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
        let AttentionInput { x, mask, mut cache } = input;

        let (B, L) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        let keys = self.k_proj.forward(x, stream)?;
        let values = self.v_proj.forward(x, stream)?;

        let queries = reshape_attention_projection(queries, B, L, self.n_heads, stream)?;
        let keys = reshape_attention_projection(keys, B, L, self.n_kv_heads, stream)?;
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
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Attention::new(args, stream)?;
        let mlp = SwiGluMlp::unloaded(
            args.hidden_size,
            args.intermediate_size,
            args.mlp_bias,
            stream,
        )?;
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
/// Llama transformer body without the language-model head.
pub struct LlamaModel {
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

impl LlamaModel {
    /// Creates an unloaded Llama transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens =
            nn::Embedding::unloaded(args.vocab_size, args.hidden_size, Dtype::Float32, stream)?;
        let layers = (0..num_hidden_layers)
            .map(|_| TransformerBlock::new(args, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
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
            None => {
                if h.shape()[1] > 1 {
                    let m = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(
                        h.shape()[1],
                        stream,
                    )?;
                    Some(m.as_dtype(h.dtype(), stream)?)
                } else {
                    None
                }
            }
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

impl<C> Module<ModelInput<'_, C>> for LlamaModel
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
            None => {
                if h.shape()[1] > 1 {
                    let m = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(
                        h.shape()[1],
                        stream,
                    )?;
                    Some(m.as_dtype(h.dtype(), stream)?)
                } else {
                    None
                }
            }
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
/// Llama causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: LlamaModel,

    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded Llama causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let model = LlamaModel::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(common::build_unloaded_maybe_quantized_lm_head(
                args.hidden_size,
                args.vocab_size,
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
        <LlamaModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
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
    let mut model_args: ModelArgs = serde_json::from_reader(file)?;
    if model_args.num_key_value_heads == 0 {
        model_args.num_key_value_heads = model_args.num_attention_heads;
    }
    if model_args.head_dim == 0 {
        model_args.head_dim = model_args.hidden_size / model_args.num_attention_heads;
    }
    if model_args.max_position_embeddings == 0 {
        model_args.max_position_embeddings = 2048;
    }

    Ok(model_args)
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
pub fn load_llama_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_llama_model_args(model_dir)?;
    let mut model = Model::new(model_args, stream)?;

    load_safetensors_dir_lenient(&mut model, model_dir, weights_stream)?;
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

/// Llama token generation iterator.
pub type Generate<'a, C, S = crate::sampler::DefaultSampler> =
    common::Generate<'a, Model, Vec<Option<C>>, S>;

#[cfg(test)]
mod tests {
    use std::{env::home_dir, fs};

    use lazy_static::lazy_static;
    use safemlx::{
        ops::indexing::{NewAxis, TryIndexOp},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::ConcatKeyValueCache,
        models::llama::{load_llama_model, load_llama_tokenizer},
    };

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
        let model = super::Model::new(model_args, stream).unwrap();

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
            load_llama_model(CACHED_TEST_MODEL_DIR.as_str(), stream, weights_stream).unwrap();

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
