use safemlx::{
    argmax_axis, array,
    error::Exception,
    random::{self, RandomState},
    Array, Stream,
};

pub trait Sampler {
    fn sample(
        &mut self,
        logits: &Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception>;
}

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
