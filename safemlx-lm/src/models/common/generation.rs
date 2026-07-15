//! Causal language-model generation traits, sampling, and iterators.

use std::marker::PhantomData;

use safemlx::{
    argmax_axis, array,
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    random::{self, RandomState},
    Array, Stream,
};

use crate::{
    models::input,
    sampler::{DefaultSampler, Sampler},
};

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
    /// Computes logits for an initial typed input and fills `cache`.
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
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
        /// Typed input used for the initial prefill pass.
        input: input::ModelInput<'a>,
    },
    /// The iterator is decoding from the previous sampled token.
    Decode {
        /// Previously sampled token id array.
        y: Array,
    },
}

/// Generic token iterator for a causal LM.
pub struct Generate<'a, M, C, S = DefaultSampler>
where
    M: CausalLm<C>,
    S: Sampler,
{
    model: &'a mut M,
    cache: &'a mut C,
    temp: f32,
    prng_state: Option<RandomState>,
    sampler: S,
    stream: &'a Stream,
    state: GenerateState<'a>,
    _cache: PhantomData<C>,
}

impl<'a, M, C> Generate<'a, M, C, DefaultSampler>
where
    M: CausalLm<C>,
{
    /// Creates a generation iterator over token-id arrays using the default sampler.
    pub fn new(
        model: &'a mut M,
        cache: &'a mut C,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Self {
        Self::with_sampler(model, cache, temp, input, prng_key, stream, DefaultSampler)
    }
}

impl<'a, M, C, S> Generate<'a, M, C, S>
where
    M: CausalLm<C>,
    S: Sampler,
{
    /// Creates a generation iterator over token-id arrays using a caller-provided sampler.
    pub fn with_sampler(
        model: &'a mut M,
        cache: &'a mut C,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            prng_state: prng_key.map(RandomState::from_key),
            sampler,
            stream,
            state: GenerateState::Prefill { input },
            _cache: PhantomData,
        }
    }
}

impl<M, C, S> Iterator for Generate<'_, M, C, S>
where
    M: CausalLm<C>,
    S: Sampler,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { input } => {
                let logits = match self
                    .model
                    .prefill_input_logits(*input, self.cache, self.stream)
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
                let y = match self.sampler.sample(
                    &logits,
                    self.temp,
                    self.prng_state.as_mut(),
                    self.stream,
                ) {
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
                let y = match self.sampler.sample(
                    &logits,
                    self.temp,
                    self.prng_state.as_mut(),
                    self.stream,
                ) {
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
