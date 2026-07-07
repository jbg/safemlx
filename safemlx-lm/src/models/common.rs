//! Shared layers and generation machinery for decoder-only causal LMs.

use std::marker::PhantomData;

use safemlx::{
    argmax_axis, array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{
        indexing::{NewAxis, TryIndexOp},
        sigmoid,
    },
    quantization::MaybeQuantized,
    random::{self, RandomState},
    Array, Dtype, Stream,
};

use crate::{
    cache::KeyValueCache,
    utils::{rope::RopeVariant, scaled_dot_product_attention},
};

/// Applies the SiLU activation function.
pub fn silu(x: Array, stream: &Stream) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x, stream)?, stream)
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
        Ok(Self {
            gate_proj: MaybeQuantized::Original(nn::Linear::unloaded(
                dim,
                hidden_dim,
                bias,
                Dtype::Float32,
                stream,
            )?),
            down_proj: MaybeQuantized::Original(nn::Linear::unloaded(
                hidden_dim,
                dim,
                bias,
                Dtype::Float32,
                stream,
            )?),
            up_proj: MaybeQuantized::Original(nn::Linear::unloaded(
                dim,
                hidden_dim,
                bias,
                Dtype::Float32,
                stream,
            )?),
        })
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
    Ok(MaybeQuantized::Original(build_unloaded_lm_head(
        hidden_size,
        vocab_size,
        stream,
    )?))
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
    /// Computes logits for an initial prompt and fills `cache`.
    fn prefill_logits(
        &mut self,
        prompt_tokens: &Array,
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
        /// Prompt token ids used for the initial prefill pass.
        prompt_tokens: &'a Array,
    },
    /// The iterator is decoding from the previous sampled token.
    Decode {
        /// Previously sampled token id array.
        y: Array,
    },
}

/// Generic token iterator for a causal LM.
pub struct Generate<'a, M, C>
where
    M: CausalLm<C>,
{
    model: &'a mut M,
    cache: &'a mut C,
    temp: f32,
    prng_state: Option<RandomState>,
    stream: &'a Stream,
    state: GenerateState<'a>,
    _cache: PhantomData<C>,
}

impl<'a, M, C> Generate<'a, M, C>
where
    M: CausalLm<C>,
{
    /// Creates a generation iterator over token-id arrays.
    pub fn new(
        model: &'a mut M,
        cache: &'a mut C,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            prng_state: prng_key.map(RandomState::from_key),
            stream,
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
                let logits = match self
                    .model
                    .prefill_logits(prompt_tokens, self.cache, self.stream)
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
                let y = match sample(&logits, self.temp, self.prng_state.as_mut(), self.stream) {
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
                let y = match sample(&logits, self.temp, self.prng_state.as_mut(), self.stream) {
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
    use super::sample;
    use safemlx::{Array, Device, DeviceType, ExecutionContext};

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
}
