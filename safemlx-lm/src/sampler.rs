use safemlx::{
    argmax_axis, array,
    error::Exception,
    ops::indexing::TryIndexOp,
    random::{self, RandomState},
    Array, Stream,
};

/// Sampling policy suitable for lossless speculative decoding.
///
/// Unlike [`Sampler`], this interface separates logits processing, sampling,
/// and history commitment.  A speculative decoder can therefore inspect the
/// exact target and draft distributions without recording rejected tokens.
pub trait SpeculativeSampler {
    /// Applies penalties, filters, and temperature using the supplied logical
    /// token history, returning canonical-vocabulary logits.
    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception>;

    /// Samples from logits returned by [`SpeculativeSampler::process_logits`].
    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match temperature {
            0.0 => argmax_axis!(logits, -1, stream = stream),
            _ => {
                let prng_state = prng_state.ok_or_else(|| {
                    Exception::custom("random operations require an explicit PRNG key")
                })?;
                let key = prng_state.next_key(stream)?;
                random::categorical(logits, None, None, &key, stream)
            }
        }
    }

    /// Commits an emitted token from a processed target distribution.
    ///
    /// Stateless policies may use the default no-op. Adaptive policies update
    /// state here only after speculative verification accepts a proposal or
    /// chooses its replacement.
    fn commit_token(
        &mut self,
        _processed_logits: &Array,
        _token: u32,
        _stream: &Stream,
    ) -> Result<(), Exception> {
        Ok(())
    }
}

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
#[derive(Debug, Clone, Copy)]
pub struct DefaultSampler;

impl SpeculativeSampler for DefaultSampler {
    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        _history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if temperature == 0.0 {
            Ok(logits.clone())
        } else {
            logits.multiply(array!(1.0 / temperature), stream)
        }
    }
}

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

/// Adaptive Mirostat V2 sampler for single-sequence text generation.
///
/// Mirostat V2 targets a configurable surprise value instead of applying a
/// fixed top-k or top-p cutoff. On each step it:
///
/// 1. applies repetition, frequency, and presence penalties;
/// 2. computes the temperature-scaled token probabilities;
/// 3. retains tokens with surprise no greater than the adaptive value `mu`;
/// 4. samples from the retained tokens; and
/// 5. adjusts `mu` toward the target surprise `tau` using learning rate `eta`.
///
/// The initial `mu` is `2 * tau`, as in the reference algorithm. This sampler
/// is stateful and currently supports one sequence at a time. Under
/// [`SpeculativeSampler`], `mu` advances only from committed target
/// distributions; rejected draft tokens never update adaptive state.
#[derive(Debug, Clone)]
pub struct MirostatV2Sampler {
    tau: f32,
    eta: f32,
    mu: f32,
    penalties: GenerationSampler,
}

impl Default for MirostatV2Sampler {
    fn default() -> Self {
        Self {
            tau: 5.0,
            eta: 0.1,
            mu: 10.0,
            penalties: GenerationSampler::new().top_k(0).top_p(1.0).min_p(0.0),
        }
    }
}

impl MirostatV2Sampler {
    /// Creates a Mirostat V2 sampler with initial adaptive value `2 * tau`.
    ///
    /// `tau` is the target surprise in bits and `eta` is the adaptation rate.
    /// Both values must be finite and greater than zero.
    pub fn new(tau: f32, eta: f32) -> Result<Self, Exception> {
        validate_positive_finite("Mirostat V2 tau", tau)?;
        validate_positive_finite("Mirostat V2 eta", eta)?;
        Ok(Self {
            tau,
            eta,
            mu: 2.0 * tau,
            penalties: GenerationSampler::new().top_k(0).top_p(1.0).min_p(0.0),
        })
    }

    /// Sets repetition, frequency, and presence penalties applied before
    /// Mirostat truncation.
    pub fn penalties(
        mut self,
        repeat_penalty: f32,
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        self.penalties = self.penalties.penalties(
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
        );
        self
    }

    /// Returns the target surprise in bits.
    pub const fn tau(&self) -> f32 {
        self.tau
    }

    /// Returns the adaptation rate.
    pub const fn eta(&self) -> f32 {
        self.eta
    }

    /// Returns the current adaptive surprise limit.
    pub const fn mu(&self) -> f32 {
        self.mu
    }

    /// Returns generated token ids already accepted by this sampler.
    pub fn generated_tokens(&self) -> &[u32] {
        self.penalties.generated_tokens()
    }

    /// Records a token accepted outside this sampler and updates adaptive state.
    ///
    /// `probability` must be the token's probability after Mirostat truncation
    /// and renormalization.
    pub fn accept_token(&mut self, token_id: u32, probability: f32) -> Result<(), Exception> {
        if !probability.is_finite() || probability <= 0.0 || probability > 1.0 {
            return Err(Exception::custom(
                "accepted Mirostat V2 token probability must be finite and in (0, 1]",
            ));
        }
        self.update_mu(-probability.log2());
        self.penalties.accept_token(token_id);
        Ok(())
    }

    /// Resets adaptive state and accepted-token history.
    pub fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
        self.penalties.clear_generated_tokens();
    }

    fn update_mu(&mut self, observed_surprise: f32) {
        self.mu -= self.eta * (observed_surprise - self.tau);
    }

    fn process_logits_for(
        &self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Exception::custom(
                "Mirostat V2 requires a finite temperature greater than zero",
            ));
        }
        let vocab_size = logits.dim(-1) as usize;
        if vocab_size == 0 || logits.size() / vocab_size != 1 {
            return Err(Exception::custom(
                "Mirostat V2 currently requires logits for exactly one sequence",
            ));
        }

        let logits = self
            .penalties
            .apply_penalties_for(logits, history, stream)?;
        let scaled_logits = logits.multiply(array!(1.0 / temperature), stream)?;
        let probabilities = safemlx::ops::softmax_axis(&scaled_logits, -1, true, stream)?;

        // A token's surprise is -log2(p), so surprise <= mu is equivalent to
        // p >= 2^-mu. If mu is temporarily below every surprise, retain only
        // the first argmax token, matching the reference algorithm's fallback.
        let cutoff_probability = Array::from_f32((-self.mu).exp2());
        let maximum_probability = probabilities.max_axis(-1, true, stream)?;
        let cutoff_mask = probabilities.lt(&cutoff_probability, stream)?;
        let best_token =
            argmax_axis!(&probabilities, -1, stream = stream)?.expand_dims_axes(&[-1], stream)?;
        let fallback_mask = Array::full::<bool>(logits.shape(), Array::from_bool(true), stream)?;
        let keep_best = Array::full::<bool>(best_token.shape(), Array::from_bool(false), stream)?;
        let fallback_mask = safemlx::ops::indexing::put_along_axis(
            &fallback_mask,
            &best_token,
            &keep_best,
            -1,
            stream,
        )?;
        let needs_fallback = cutoff_probability.gt(maximum_probability, stream)?;
        let mask = safemlx::ops::r#where(needs_fallback, fallback_mask, cutoff_mask, stream)?;
        mask_logits(mask, scaled_logits, stream)
    }

    fn commit_processed_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let vocab_size = processed_logits.dim(-1) as usize;
        if token as usize >= vocab_size {
            return Err(Exception::custom(format!(
                "sampled token {token} exceeds vocabulary size {vocab_size}"
            )));
        }
        let probabilities = safemlx::ops::softmax_axis(processed_logits, -1, true, stream)?;
        let selected = match probabilities.ndim() {
            1 => probabilities.try_index_device(token as i32, stream)?,
            2 => probabilities.try_index_device((0, token as i32), stream)?,
            3 => probabilities.try_index_device((0, 0, token as i32), stream)?,
            rank => {
                return Err(Exception::custom(format!(
                    "Mirostat V2 processed logits must have rank 1, 2, or 3, got rank {rank}"
                )))
            }
        };
        self.accept_token(token, selected.item::<f32>(stream))
    }
}

impl Sampler for MirostatV2Sampler {
    fn sample(
        &mut self,
        logits: &Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let processed_logits =
            self.process_logits_for(logits, temp, self.penalties.generated_tokens(), stream)?;
        let prng_state = prng_state
            .ok_or_else(|| Exception::custom("random operations require an explicit PRNG key"))?;
        let key = prng_state.next_key(stream)?;
        let token = random::categorical(&processed_logits, None, None, &key, stream)?;
        self.commit_processed_token(&processed_logits, token.clone().item::<u32>(stream), stream)?;
        Ok(token)
    }
}

impl SpeculativeSampler for MirostatV2Sampler {
    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.process_logits_for(logits, temperature, history, stream)
    }

    fn commit_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.commit_processed_token(processed_logits, token, stream)
    }
}

/// Configurable sampler for text generation.
///
/// The sampler mirrors the common llama.cpp sampling chain used by Goose:
/// repetition/frequency/presence penalties, then top-k, top-p, min-p,
/// temperature, and finally greedy or categorical token selection.
#[derive(Debug, Clone)]
pub struct GenerationSampler {
    /// Keep only the `top_k` highest-logit tokens when positive.
    pub top_k: i32,
    /// Keep the smallest prefix of tokens whose probability mass reaches `top_p`.
    pub top_p: f32,
    /// Keep tokens whose probability is at least `min_p * max_probability`.
    pub min_p: f32,
    /// Repetition penalty applied to recently generated tokens. `1.0` disables it.
    pub repeat_penalty: f32,
    /// Number of generated tokens considered by repetition penalties. Negative means all.
    pub repeat_last_n: i32,
    /// Frequency penalty subtracted once per generated occurrence.
    pub frequency_penalty: f32,
    /// Presence penalty subtracted once for any generated occurrence.
    pub presence_penalty: f32,
    generated_tokens: Vec<u32>,
}

impl Default for GenerationSampler {
    fn default() -> Self {
        Self {
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            generated_tokens: Vec::new(),
        }
    }
}

impl GenerationSampler {
    /// Creates a sampler with default generation settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a sampler with an initial accepted-token history.
    ///
    /// The history is used by repetition, frequency, and presence penalties.
    /// This is useful when resuming generation or when tokens were accepted by
    /// a caller outside of [`Sampler::sample`].
    pub fn with_generated_tokens(mut self, token_ids: impl IntoIterator<Item = u32>) -> Self {
        self.generated_tokens = token_ids.into_iter().collect();
        self
    }

    /// Sets top-k filtering.
    pub fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets top-p filtering.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Sets min-p filtering.
    pub fn min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Sets repetition, frequency, and presence penalties.
    pub fn penalties(
        mut self,
        repeat_penalty: f32,
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.frequency_penalty = frequency_penalty;
        self.presence_penalty = presence_penalty;
        self
    }

    /// Returns generated token ids already accepted by this sampler.
    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated_tokens
    }

    /// Replaces the accepted-token history used by repetition penalties.
    pub fn set_generated_tokens(&mut self, token_ids: impl IntoIterator<Item = u32>) {
        self.generated_tokens = token_ids.into_iter().collect();
    }

    /// Records a token accepted by the caller.
    ///
    /// [`Sampler::sample`] records sampled tokens automatically. Call this only
    /// for tokens chosen outside the sampler, for example a constrained token
    /// or an externally selected branch token.
    pub fn accept_token(&mut self, token_id: u32) {
        self.generated_tokens.push(token_id);
    }

    /// Clears accepted-token history.
    pub fn clear_generated_tokens(&mut self) {
        self.generated_tokens.clear();
    }

    fn apply_penalties_for(
        &self,
        logits: &Array,
        generated_tokens: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if generated_tokens.is_empty()
            || (self.repeat_penalty == 1.0
                && self.frequency_penalty == 0.0
                && self.presence_penalty == 0.0)
        {
            return Ok(logits.clone());
        }

        let vocab_size = logits.dim(-1) as usize;
        if vocab_size == 0 {
            return Ok(logits.clone());
        }
        let row_count = logits.size() / vocab_size;
        let mut repeat_mask = vec![false; logits.size()];
        let mut penalties = vec![0.0f32; logits.size()];

        let start = if self.repeat_last_n < 0 {
            0
        } else {
            generated_tokens
                .len()
                .saturating_sub(self.repeat_last_n as usize)
        };
        let mut counts = std::collections::HashMap::<u32, usize>::new();
        for &token_id in &generated_tokens[start..] {
            *counts.entry(token_id).or_default() += 1;
        }

        for (token_id, count) in counts {
            let token_index = token_id as usize;
            if token_index >= vocab_size {
                continue;
            }
            for row in 0..row_count {
                let index = row * vocab_size + token_index;
                repeat_mask[index] = true;
                penalties[index] = self.frequency_penalty * count as f32 + self.presence_penalty;
            }
        }

        let mut adjusted = logits.clone();
        if self.repeat_penalty != 1.0 {
            let mask = Array::from_slice(&repeat_mask, logits.shape());
            let positive = adjusted.divide(array!(self.repeat_penalty), stream)?;
            let negative = adjusted.multiply(array!(self.repeat_penalty), stream)?;
            let penalized = safemlx::ops::r#where(
                adjusted.gt(Array::from_f32(0.0), stream)?,
                positive,
                negative,
                stream,
            )?;
            adjusted = safemlx::ops::r#where(mask, penalized, adjusted, stream)?;
        }

        if self.frequency_penalty != 0.0 || self.presence_penalty != 0.0 {
            adjusted = adjusted.subtract(Array::from_slice(&penalties, logits.shape()), stream)?;
        }

        Ok(adjusted)
    }

    fn apply_penalties(&self, logits: &Array, stream: &Stream) -> Result<Array, Exception> {
        self.apply_penalties_for(logits, &self.generated_tokens, stream)
    }

    fn apply_top_k(&self, logits: Array, stream: &Stream) -> Result<Array, Exception> {
        let vocab_size = logits.dim(-1);
        if self.top_k <= 0 || self.top_k >= vocab_size {
            return Ok(logits);
        }

        let top_values = safemlx::ops::indexing::topk_axis(&logits, self.top_k, -1, stream)?;
        let threshold = top_values.min_axis(-1, true, stream)?;
        mask_logits(logits.lt(threshold, stream)?, logits, stream)
    }

    fn apply_min_p(&self, logits: Array, stream: &Stream) -> Result<Array, Exception> {
        if self.min_p <= 0.0 {
            return Ok(logits);
        }

        let probabilities = safemlx::ops::softmax_axis(&logits, -1, true, stream)?;
        let max_probability = probabilities.max_axis(-1, true, stream)?;
        let threshold = max_probability.multiply(Array::from_f32(self.min_p), stream)?;
        mask_logits(probabilities.lt(threshold, stream)?, logits, stream)
    }

    fn sample_filtered(
        &mut self,
        logits: &Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let token = match temp {
            0.0 => argmax_axis!(logits, -1, stream = stream)?,
            _ => {
                let prng_state = prng_state.ok_or_else(|| {
                    Exception::custom("random operations require an explicit PRNG key")
                })?;
                let key = prng_state.next_key(stream)?;
                let logits = logits.multiply(array!(1.0 / temp), stream)?;
                random::categorical(&logits, None, None, &key, stream)?
            }
        };
        self.generated_tokens
            .push(token.clone().item::<u32>(stream));
        Ok(token)
    }

    fn sample_top_p(
        &mut self,
        logits: Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if self.top_p >= 1.0 {
            let logits = self.apply_min_p(logits, stream)?;
            return self.sample_filtered(&logits, temp, prng_state, stream);
        }

        let descending_indices = safemlx::ops::argsort_axis(logits.negative(stream)?, -1, stream)?;
        let sorted_logits =
            safemlx::ops::indexing::take_along_axis(&logits, &descending_indices, -1, stream)?;
        let probabilities = safemlx::ops::softmax_axis(&sorted_logits, -1, true, stream)?;
        let cumulative_probabilities = probabilities.cumsum(-1, None, None, stream)?;
        let cumulative_before_token = cumulative_probabilities.subtract(probabilities, stream)?;
        let mask = cumulative_before_token.gt(Array::from_f32(self.top_p.max(0.0)), stream)?;
        let sorted_logits = mask_logits(mask, sorted_logits, stream)?;
        let sorted_logits = self.apply_min_p(sorted_logits, stream)?;
        let sorted_token = self.sample_filtered(&sorted_logits, temp, prng_state, stream)?;
        let token = safemlx::ops::indexing::take_along_axis(
            descending_indices,
            &sorted_token.expand_dims_axes(&[-1], stream)?,
            -1,
            stream,
        )?
        .squeeze_axes(&[-1], stream)?;
        if let Some(last) = self.generated_tokens.last_mut() {
            *last = token.clone().item::<u32>(stream);
        }
        Ok(token)
    }
}

impl SpeculativeSampler for GenerationSampler {
    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.apply_penalties_for(logits, history, stream)?;
        let logits = self.apply_top_k(logits, stream)?;
        let logits = if self.top_p >= 1.0 {
            self.apply_min_p(logits, stream)?
        } else {
            let descending_indices =
                safemlx::ops::argsort_axis(logits.negative(stream)?, -1, stream)?;
            let sorted_logits =
                safemlx::ops::indexing::take_along_axis(&logits, &descending_indices, -1, stream)?;
            let probabilities = safemlx::ops::softmax_axis(&sorted_logits, -1, true, stream)?;
            let cumulative = probabilities.cumsum(-1, None, None, stream)?;
            let before = cumulative.subtract(probabilities, stream)?;
            let mask = before.gt(Array::from_f32(self.top_p.max(0.0)), stream)?;
            let sorted_logits = mask_logits(mask, sorted_logits, stream)?;
            let sorted_logits = self.apply_min_p(sorted_logits, stream)?;
            let fill = Array::full::<f32>(
                logits.shape(),
                Array::from_f32(logits.dtype().finfo_min()? as f32),
                stream,
            )?
            .as_dtype(logits.dtype(), stream)?;
            safemlx::ops::indexing::put_along_axis(
                &fill,
                &descending_indices,
                &sorted_logits,
                -1,
                stream,
            )?
        };
        if temperature == 0.0 {
            Ok(logits)
        } else {
            logits.multiply(array!(1.0 / temperature), stream)
        }
    }
}

impl Sampler for GenerationSampler {
    fn sample(
        &mut self,
        logits: &Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.apply_penalties(logits, stream)?;
        let logits = self.apply_top_k(logits, stream)?;
        self.sample_top_p(logits, temp, prng_state, stream)
    }
}

fn mask_logits(mask: Array, logits: Array, stream: &Stream) -> Result<Array, Exception> {
    let min_value = Array::from_f32(logits.dtype().finfo_min()? as f32);
    safemlx::ops::r#where(mask, min_value, logits, stream)
}

fn validate_positive_finite(name: &str, value: f32) -> Result<(), Exception> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Exception::custom(format!(
            "{name} must be finite and greater than zero"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationSampler, MirostatV2Sampler};

    #[test]
    fn generation_sampler_accepts_external_token_history() {
        let mut sampler = GenerationSampler::new().with_generated_tokens([1, 2]);
        assert_eq!(sampler.generated_tokens(), &[1, 2]);

        sampler.accept_token(3);
        assert_eq!(sampler.generated_tokens(), &[1, 2, 3]);

        sampler.set_generated_tokens([5, 8]);
        assert_eq!(sampler.generated_tokens(), &[5, 8]);

        sampler.clear_generated_tokens();
        assert!(sampler.generated_tokens().is_empty());
    }

    #[test]
    fn mirostat_v2_defaults_and_reset_restore_adaptive_state() {
        let mut sampler = MirostatV2Sampler::default();
        assert_eq!(sampler.tau(), 5.0);
        assert_eq!(sampler.eta(), 0.1);
        assert_eq!(sampler.mu(), 10.0);

        sampler.accept_token(42, 2.0f32.powi(-7)).unwrap();
        assert!((sampler.mu() - 9.8).abs() < 1e-6);
        assert_eq!(sampler.generated_tokens(), &[42]);

        sampler.reset();
        assert_eq!(sampler.mu(), 10.0);
        assert!(sampler.generated_tokens().is_empty());
    }

    #[test]
    fn mirostat_v2_validates_configuration() {
        assert!(MirostatV2Sampler::new(3.0, 0.2).is_ok());
        assert!(MirostatV2Sampler::new(0.0, 0.2).is_err());
        assert!(MirostatV2Sampler::new(3.0, f32::NAN).is_err());

        let mut sampler = MirostatV2Sampler::default();
        assert!(sampler.accept_token(0, 0.0).is_err());
        assert!(sampler.accept_token(0, 1.1).is_err());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mirostat_v2_samples_and_updates_mu() {
        use super::Sampler;
        use safemlx::{
            random::{self, RandomState},
            Array, Device, DeviceType, ExecutionContext,
        };

        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let logits = Array::from_slice(&[0.0f32, -100.0, -100.0], &[1, 3]);
        let mut state = RandomState::from_key(random::key(0).unwrap());
        let mut sampler = MirostatV2Sampler::new(5.0, 0.1).unwrap();

        let token = sampler
            .sample(&logits, 1.0, Some(&mut state), stream)
            .unwrap();

        assert_eq!(token.item::<u32>(stream), 0);
        assert!(sampler.mu() > 10.0);
        assert_eq!(sampler.generated_tokens(), &[0]);
    }
}
