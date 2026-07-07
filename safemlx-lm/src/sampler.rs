use safemlx::{
    argmax_axis, array,
    error::Exception,
    random::{self, RandomState},
    Array, Stream,
};

/// Strategy for choosing a token from model logits.
pub trait Sampler {
    /// Samples one token id from `logits`.
    ///
    /// Implementations may use `temp` and `prng_state`; stochastic samplers
    /// should return an error when randomness is required but no PRNG state is
    /// supplied.
    fn sample(
        &mut self,
        logits: &Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception>;
}

/// Default sampler used by generation helpers.
///
/// A temperature of `0.0` uses greedy argmax sampling. Non-zero temperatures
/// sample from a categorical distribution and require a PRNG key.
pub struct DefaultSampler;

impl Sampler for DefaultSampler {
    fn sample(
        &mut self,
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
}
