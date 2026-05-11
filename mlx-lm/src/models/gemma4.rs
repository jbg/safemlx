use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        indexing::{IndexOp, NewAxis},
        tanh,
    },
    quantization::MaybeQuantized,
    Array,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub head_dim: i32,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, HashMap<String, FloatOrString>>>,
}

fn default_model_type() -> String {
    "gemma4".to_string()
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl<'de> Deserialize<'de> for LayerType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "sliding_attention" => Ok(Self::SlidingAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(serde::de::Error::custom(format!(
                "Unsupported Gemma4 layer type '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4Config {
    text_config: ModelArgs,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

impl ModelArgs {
    fn for_layer(&self, layer_type: LayerType) -> Self {
        let mut args = self.clone();
        if layer_type == LayerType::FullAttention {
            if let Some(global_head_dim) = self.global_head_dim {
                args.head_dim = global_head_dim;
            }
            if let Some(global_kv_heads) = self.num_global_key_value_heads {
                args.num_key_value_heads = global_kv_heads;
            }
        }
        args.rope_theta = self.rope_theta_for_layer(layer_type);
        args.rope_scaling = self.rope_scaling_for_layer(layer_type);
        args
    }

    fn layer_type(&self, index: usize) -> LayerType {
        self.layer_types
            .get(index)
            .copied()
            .unwrap_or(LayerType::FullAttention)
    }

    fn rope_theta_for_layer(&self, layer_type: LayerType) -> f32 {
        let key = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
        };
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.get(key))
            .and_then(|params| params.get("rope_theta"))
            .and_then(|value| match value {
                FloatOrString::Float(v) => Some(*v),
                FloatOrString::String(s) => s.parse().ok(),
            })
            .unwrap_or(self.rope_theta)
    }

    fn rope_scaling_for_layer(
        &self,
        layer_type: LayerType,
    ) -> Option<HashMap<String, FloatOrString>> {
        let key = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
        };
        self.rope_parameters.as_ref().and_then(|params| {
            params.get(key).map(|params| {
                let mut scaling = params.clone();
                if matches!(scaling.get("rope_type"), Some(FloatOrString::String(s)) if s == "proportional") {
                    scaling.insert("rope_type".to_string(), FloatOrString::String("default".to_string()));
                }
                scaling
            })
        })
    }
}

fn partial_rotary_dims(head_dim: i32, scaling: &Option<HashMap<String, FloatOrString>>) -> i32 {
    let partial_factor = scaling
        .as_ref()
        .and_then(|scaling| scaling.get("partial_rotary_factor"))
        .and_then(|value| match value {
            FloatOrString::Float(v) => Some(*v),
            FloatOrString::String(s) => s.parse().ok(),
        })
        .unwrap_or(1.0);
    ((head_dim as f32 * partial_factor).round() as i32).clamp(2, head_dim)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(args.attention_bias)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        let rope_dims = partial_rotary_dims(head_dim, &args.rope_scaling);
        let rope = initialize_rope(
            rope_dims,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
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

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = self.q_norm.forward(
            &queries
                .reshape(&[B, L, self.n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut keys = self.k_norm.forward(
            &keys
                .reshape(&[B, L, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(cache) = cache.as_mut() {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = self.rope.forward(k_input)?;
            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        let output = crate::utils::scaled_dot_product_attention(
            queries, keys, values, cache, self.scale, mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let down_proj_input = nn::gelu_approximate(self.gate_proj.forward(input)?)?
            .multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&down_proj_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,
    pub layer_type: LayerType,
    pub sliding_window: Option<i32>,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub layer_scalar: Param<Array>,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs, layer_type: LayerType) -> Result<Self, Exception> {
        let layer_args = args.for_layer(layer_type);
        let self_attn = Attention::new(&layer_args)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size)?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let pre_feedforward_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_feedforward_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            num_attention_heads: layer_args.num_attention_heads,
            hidden_size: layer_args.hidden_size,
            layer_type,
            sliding_window: args.sliding_window,
            layer_scalar: Param::new(Array::ones::<f32>(&[args.hidden_size])?),
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }
}

impl TransformerBlock {
    fn apply_layer_scalar(&self, x: Array) -> Result<Array, Exception> {
        x.multiply(&*self.layer_scalar)
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let generated_mask = if self.layer_type == LayerType::SlidingAttention {
            let offset = cache.as_ref().map(|cache| cache.offset()).unwrap_or(0);
            if x.shape()[1] > 1 || self.sliding_window.is_some() {
                Some(create_causal_mask(
                    x.shape()[1],
                    Some(offset),
                    self.sliding_window,
                    None,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let self_attn_input = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask: generated_mask.as_ref().or(mask),
            cache,
        };
        let r = self.self_attn.forward(self_attn_input)?;
        let r = self.post_attention_layernorm.forward(&r)?;
        let r = self.apply_layer_scalar(r)?;
        let h = x.add(r)?;
        let r = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
        let r = self.post_feedforward_layernorm.forward(&r)?;
        let r = self.apply_layer_scalar(r)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma4TextModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Gemma4TextModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|index| TransformerBlock::new(args, args.layer_type(index as usize)))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Gemma4TextModel
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;
        let mut h = self.embed_tokens.forward(inputs)?;
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None if h.shape()[1] > 1 => Some(create_causal_mask(h.shape()[1], None, None, None)?),
            None => None,
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
            h = layer.forward(layer_input)?;
        }
        self.norm.forward(&h)
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
pub struct Gemma4ForConditionalGeneration {
    #[quantizable]
    #[param]
    pub language_model: Gemma4TextModel,
}

impl Gemma4ForConditionalGeneration {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            language_model: Gemma4TextModel::new(args)?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: Gemma4ForConditionalGeneration,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Gemma4ForConditionalGeneration::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

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

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.language_model.forward(input)?;
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out)?,
            None => match &mut self.model.language_model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out)?,
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out)?,
            },
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(&(logits.divide(Array::from_f32(softcap))?))?
                .multiply(Array::from_f32(softcap))?;
        }
        Ok(logits)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma4TextModel as Module<ModelInput<'_, C>>>::training_mode(
            &mut self.model.language_model,
            mode,
        );
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_gemma4_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_gemma4_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let mut config: Gemma4Config = serde_json::from_reader(file)?;
    config.text_config.model_type = "gemma4".to_string();
    config.text_config.tie_word_embeddings = config.tie_word_embeddings;
    Ok(config.text_config)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_gemma4_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_gemma4_model_args(model_dir)?;
    let mut model = Model::new(model_args)?;
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            model.load_safetensors(weights_filename)?;
        }
    } else {
        model.load_safetensors(model_dir.join("model.safetensors"))?;
    }
    Ok(model)
}

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}
