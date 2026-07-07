//! Qwen3 decoder-only model implementation.

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
    models::common::{
        self, apply_rope_and_update_cache, batch_seq, finish_attention,
        project_logits_maybe_quantized, reshape_attention_projection, AttentionInput, CausalLm,
        SwiGluMlp,
    },
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
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
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::Linear::unloaded(dim, n_heads * head_dim, false, Dtype::Float32, stream)?;
        let k_proj =
            nn::Linear::unloaded(dim, n_kv_heads * head_dim, false, Dtype::Float32, stream)?;
        let v_proj =
            nn::Linear::unloaded(dim, n_kv_heads * head_dim, false, Dtype::Float32, stream)?;
        let o_proj = nn::Linear::unloaded(n_heads * head_dim, dim, false, Dtype::Float32, stream)?;

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
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
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
        let mlp = SwiGluMlp::unloaded(args.hidden_size, args.intermediate_size, false, stream)?;
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
    C: KeyValueCache,
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
            *cache = (0..self.layers.len()).map(|_| None).collect();
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
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache,
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

    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;

        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            model.load_safetensors(weights_filename, weights_stream)?;
        }
    } else {
        model.load_safetensors(model_dir.join("model.safetensors"), weights_stream)?;
    }
    model.copy_to_stream(stream)?;

    Ok(model)
}

impl<C> CausalLm<Vec<Option<C>>> for Model
where
    C: KeyValueCache,
{
    fn prefill_logits(
        &mut self,
        prompt_tokens: &Array,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.forward(
            ModelInput {
                inputs: prompt_tokens,
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
pub type Generate<'a, C> = common::Generate<'a, Model, Vec<Option<C>>>;

#[cfg(test)]
mod tests {
    use safemlx::{
        ops::indexing::{NewAxis, TryIndexOp},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::ConcatKeyValueCache,
        models::qwen3::{load_qwen3_model, load_qwen3_tokenizer},
    };

    const CACHED_TEST_MODEL_DIR: &str = "../cache/Qwen3-4B-bf16";

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
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt_tokens,
            None,
            stream,
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
