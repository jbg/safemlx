use std::marker::PhantomData;

use safemlx::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{
        indexing::{IndexOp, NewAxis},
        sigmoid,
    },
    quantization::MaybeQuantized,
    Array,
};

use crate::{
    cache::KeyValueCache,
    utils::{rope::RopeVariant, scaled_dot_product_attention},
};

pub fn silu(x: Array) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x)?)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct SwiGluMlp {
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

impl SwiGluMlp {
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
}

impl Module<&Array> for SwiGluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let down_proj_input =
            silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&down_proj_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct DenseSwiGluMlp {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl DenseSwiGluMlp {
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            up_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?,
            down_proj: nn::LinearBuilder::new(hidden_dim, dim).bias(bias).build()?,
        })
    }
}

impl Module<&Array> for DenseSwiGluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let h = silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

pub fn build_lm_head(hidden_size: i32, vocab_size: i32) -> Result<nn::Linear, Exception> {
    nn::LinearBuilder::new(hidden_size, vocab_size)
        .bias(false)
        .build()
}

pub fn build_maybe_quantized_lm_head(
    hidden_size: i32,
    vocab_size: i32,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    Ok(MaybeQuantized::Original(build_lm_head(
        hidden_size,
        vocab_size,
    )?))
}

pub fn project_logits_maybe_quantized(
    lm_head: &mut Option<MaybeQuantized<nn::Linear>>,
    embed_tokens: &mut MaybeQuantized<nn::Embedding>,
    hidden_states: &Array,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states),
        None => match embed_tokens {
            MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(hidden_states),
            MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(hidden_states),
        },
    }
}

pub fn project_logits_dense(
    lm_head: &mut Option<nn::Linear>,
    embed_tokens: &nn::Embedding,
    hidden_states: &Array,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states),
        None => embed_tokens.as_linear(hidden_states),
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

pub fn batch_seq(x: &Array) -> (i32, i32) {
    let shape = x.shape();
    (shape[0], shape[1])
}

pub fn reshape_attention_projection(
    projection: Array,
    batch: i32,
    seq_len: i32,
    heads: i32,
) -> Result<Array, Exception> {
    projection
        .reshape(&[batch, seq_len, heads, -1])?
        .transpose_axes(&[0, 2, 1, 3])
}

pub fn apply_rope_and_update_cache<C>(
    rope: &mut RopeVariant,
    mut queries: Array,
    mut keys: Array,
    mut values: Array,
    cache: &mut Option<&mut C>,
) -> Result<(Array, Array, Array), Exception>
where
    C: KeyValueCache,
{
    if let Some(cache) = cache.as_mut() {
        let offset = cache.offset();
        queries = rope.forward(nn::RopeInputBuilder::new(&queries).offset(offset).build()?)?;
        keys = rope.forward(nn::RopeInputBuilder::new(&keys).offset(offset).build()?)?;
        (keys, values) = cache.update_and_fetch(keys, values)?;
    } else {
        queries = rope.forward(nn::RopeInput::new(&queries))?;
        keys = rope.forward(nn::RopeInput::new(&keys))?;
    }

    Ok((queries, keys, values))
}

pub fn finish_attention<C>(
    queries: Array,
    keys: Array,
    values: Array,
    cache: Option<&mut C>,
    scale: f32,
    mask: Option<&Array>,
    batch: i32,
    seq_len: i32,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    scaled_dot_product_attention(queries, keys, values, cache, scale, mask)?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, seq_len, -1])
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

pub trait CausalLm<C> {
    fn prefill_logits(&mut self, prompt_tokens: &Array, cache: &mut C) -> Result<Array, Exception>;

    fn decode_logits(&mut self, input_tokens: &Array, cache: &mut C) -> Result<Array, Exception>;

    fn adjust_prefill_logits(&mut self, logits: Array, _cache: &mut C) -> Result<Array, Exception> {
        Ok(logits)
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_tokens: &'a Array },
    Decode { y: Array },
}

pub struct Generate<'a, M, C>
where
    M: CausalLm<C>,
{
    model: &'a mut M,
    cache: &'a mut C,
    temp: f32,
    state: GenerateState<'a>,
    _cache: PhantomData<C>,
}

impl<'a, M, C> Generate<'a, M, C>
where
    M: CausalLm<C>,
{
    pub fn new(model: &'a mut M, cache: &'a mut C, temp: f32, prompt_tokens: &'a Array) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_tokens },
            _cache: PhantomData,
        }
    }
}

impl<M, C> Iterator for Generate<'_, M, C>
where
    M: CausalLm<C>,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_tokens } => {
                let logits = match self.model.prefill_logits(prompt_tokens, self.cache) {
                    Ok(logits) => logits,
                    Err(err) => return Some(Err(err)),
                };
                let logits = match self.model.adjust_prefill_logits(logits, self.cache) {
                    Ok(logits) => logits,
                    Err(err) => return Some(Err(err)),
                };
                let y = match sample(&logits, self.temp) {
                    Ok(y) => y,
                    Err(err) => return Some(Err(err)),
                };
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let logits = match self.model.decode_logits(&inputs, self.cache) {
                    Ok(logits) => logits,
                    Err(err) => return Some(Err(err)),
                };
                let y = match sample(&logits, self.temp) {
                    Ok(y) => y,
                    Err(err) => return Some(Err(err)),
                };
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}
