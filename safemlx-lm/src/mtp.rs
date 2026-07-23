//! Architecture-independent multi-token prediction and speculative decoding.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, maximum, softmax_axis},
    random::{self, RandomState},
    transforms::eval,
    Array, Stream,
};

use crate::{
    error::Error,
    models::{
        gemma4_assistant::{
            load_gemma4_assistant_gguf_with_options, load_gemma4_assistant_model_with_options,
            Gemma4AssistantDraftModel,
        },
        input::{InputPayload, Modality, ModelInput},
        ModelCache, ModelLoadOptions,
    },
    sampler::SpeculativeSampler,
};

/// Architecture-dispatched draft model loaded independently of a target.
pub struct LoadedDrafter {
    model: DrafterModel,
}

enum DrafterModel {
    Gemma4(Gemma4AssistantDraftModel),
}

/// Per-lane target caches for independently progressing MTP text batches.
pub struct MtpCache {
    pub(crate) lanes: Vec<ModelCache>,
}

impl MtpCache {
    pub(crate) fn new(lanes: Vec<ModelCache>) -> Self {
        Self { lanes }
    }

    /// Returns the number of independent sequence lanes.
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Returns whether this cache contains no sequence lanes.
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }
}

impl LoadedDrafter {
    /// Loads a drafter from an explicit checkpoint path.
    pub fn load(
        source: impl AsRef<Path>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::load_with_options(source, ModelLoadOptions::default(), stream, weights_stream)
    }

    /// Loads a drafter using architecture-independent weight options.
    pub fn load_with_options(
        source: impl AsRef<Path>,
        options: ModelLoadOptions,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let source = source.as_ref();
        if source
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            return Ok(Self {
                model: DrafterModel::Gemma4(load_gemma4_assistant_gguf_with_options(
                    source,
                    options,
                    stream,
                    weights_stream,
                )?),
            });
        }
        Ok(Self {
            model: DrafterModel::Gemma4(load_gemma4_assistant_model_with_options(
                source,
                options,
                stream,
                weights_stream,
            )?),
        })
    }

    pub(crate) fn gemma4_mut(&mut self) -> &mut Gemma4AssistantDraftModel {
        match &mut self.model {
            DrafterModel::Gemma4(model) => model,
        }
    }
}

/// How an architecture exposes draft-token weights.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MtpCheckpointKind {
    /// Drafting weights live in a separately loaded checkpoint.
    Separate,
    /// Drafting weights are embedded in the target checkpoint.
    Embedded,
}

/// Runtime MTP status reported by a loaded model.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MtpCapability {
    /// The model does not advertise MTP weights.
    Unavailable,
    /// MTP is executable when the stated checkpoint form is provided.
    Ready {
        /// Location of the drafting weights.
        checkpoint: MtpCheckpointKind,
    },
    /// The architecture can carry MTP weights, but its runtime adapter is pending.
    Unsupported {
        /// Location of the drafting weights.
        checkpoint: MtpCheckpointKind,
        /// Stable architecture name.
        architecture: String,
    },
}

/// Options shared by architecture-specific MTP backends.
#[derive(Debug, Clone)]
pub struct MtpConfig {
    /// Maximum number of output tokens, including a terminal EOS token.
    pub max_tokens: usize,
    /// Maximum number of speculative tokens proposed per verification round.
    pub max_draft_tokens: usize,
    /// Sampling temperature. Zero selects greedy verification.
    pub temperature: f32,
    /// Token ids that terminate a sequence.
    pub eos_token_ids: Vec<u32>,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            max_draft_tokens: 4,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        }
    }
}

/// Statistics collected from one speculative sequence.
#[derive(Debug, Clone, Default)]
pub struct MtpStats {
    /// Target tokens evaluated during prefill and verification.
    pub target_tokens: usize,
    /// Assistant tokens proposed.
    pub draft_tokens: usize,
    /// Assistant tokens accepted by target verification.
    pub accepted_tokens: usize,
    /// Number of target verification rounds.
    pub rounds: usize,
    /// Accepted proposal count for each round.
    pub accept_lens: Vec<usize>,
    /// Tokens emitted, including a terminal EOS token when one is produced.
    pub emitted_tokens: usize,
    /// Wall-clock generation duration.
    pub elapsed: Duration,
}

/// Completed independently progressing text batch.
#[derive(Debug, Clone, Default)]
pub struct MtpBatchOutput {
    /// Generated token ids in original batch-lane order.
    pub token_ids: Vec<Vec<u32>>,
    /// Per-lane speculative statistics in the same order.
    pub stats: Vec<MtpStats>,
}

impl MtpStats {
    /// Fraction of proposed tokens accepted by the target.
    pub fn accept_rate(&self) -> f64 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.draft_tokens as f64
        }
    }
}

/// Target output after prefill.
pub struct MtpPrefill<S> {
    /// Logits predicting the first generated token.
    pub logits: Array,
    /// Architecture-owned state needed to begin drafting.
    pub state: S,
    /// Number of target input tokens evaluated.
    pub evaluated_tokens: usize,
}

/// Result of committing one target verification transaction.
pub struct MtpCommit<S> {
    /// Draft context matching the committed target cache.
    pub state: S,
    /// Target tokens replayed to restore an exact cache after rejection.
    pub replayed_tokens: usize,
}

/// One architecture-independent MTP backend.
///
/// Implementations may combine separate target and assistant models or operate
/// on a single checkpoint containing embedded MTP layers.
pub trait MtpBackend {
    /// Complete target cache type.
    type Cache;
    /// Target state consumed when starting a draft round.
    type TargetState;
    /// Mutable state used while producing one proposal block.
    type DraftState;
    /// Architecture-specific cache transaction marker.
    type CacheCheckpoint;
    /// Architecture-specific target verification result.
    type Verification;

    /// Maximum proposals supported by this backend in one round.
    fn max_draft_tokens(&self) -> usize {
        usize::MAX
    }

    /// Prefills a typed input and returns first-token logits plus draft context.
    fn prefill(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<MtpPrefill<Self::TargetState>, Exception>;

    /// Starts one draft round from committed target state.
    fn begin_draft(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Self::DraftState, Exception>;

    /// Returns raw next-token logits and advances private draft state.
    fn draft_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception>;

    /// Captures cache state before speculative verification.
    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint;

    /// Evaluates `[last_committed_token, proposed_tokens...]` in one target pass.
    fn verify(
        &mut self,
        input_tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<Self::Verification, Exception>;

    /// Returns verification logits shaped `[1, input_length, vocabulary]`.
    fn verification_logits(output: &Self::Verification) -> &Array;

    /// Retains exactly `verified_inputs` tokens after the checkpoint and
    /// returns draft context matching the retained prefix.
    fn commit_verification(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        stream: &Stream,
    ) -> Result<MtpCommit<Self::TargetState>, Exception>;
}

/// Runs one batch-one speculative sequence using an architecture backend.
pub fn generate<B, S>(
    backend: &mut B,
    cache: &mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &Stream,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MtpBackend,
    S: SpeculativeSampler,
{
    generate_with_callback(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        stream,
        |_| Ok(()),
    )
}

/// Runs one batch-one speculative sequence and reports each committed token.
///
/// The callback is invoked as soon as a token is committed to the target
/// sequence, including a terminal EOS token. Tokens are reported in generation
/// order and before this function returns, allowing callers to stream decoded
/// text while speculative generation is still in progress.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_callback<B, S, F>(
    backend: &mut B,
    cache: &mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &Stream,
    mut on_token: F,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MtpBackend,
    S: SpeculativeSampler,
    F: FnMut(u32) -> Result<(), Exception>,
{
    let started = Instant::now();
    let mut stats = MtpStats::default();
    if config.max_tokens == 0 {
        return Ok((Vec::new(), stats));
    }
    validate_input(input)?;
    if config.max_draft_tokens == 0 {
        return Err(Exception::custom("MTP max_draft_tokens must be positive"));
    }
    if backend.max_draft_tokens() == 0 {
        return Err(Exception::custom(
            "MTP backend does not permit any draft tokens",
        ));
    }
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(Exception::custom(
            "MTP temperature must be finite and non-negative",
        ));
    }
    if config.temperature != 0.0 && prng_key.is_none() {
        return Err(Exception::custom(
            "random operations require an explicit PRNG key",
        ));
    }

    let mut prng_state = prng_key.map(RandomState::from_key);
    let prefill = backend.prefill(input, cache, stream)?;
    stats.target_tokens += prefill.evaluated_tokens;
    let first_logits = sampler.process_logits(&prefill.logits, config.temperature, &[], stream)?;
    let first = sampler.sample_processed(
        &first_logits,
        config.temperature,
        prng_state.as_mut(),
        stream,
    )?;
    eval([&first])?;
    let first = first.item::<u32>(stream);
    sampler.commit_token(&first_logits, first, stream)?;
    let mut output = vec![first];
    stats.emitted_tokens = 1;
    on_token(first)?;
    let mut target_state = prefill.state;
    if config.eos_token_ids.contains(&first) {
        stats.elapsed = started.elapsed();
        return Ok((output, stats));
    }

    while output.len() < config.max_tokens {
        let last = *output.last().expect("first token emitted");
        let proposal_count = config
            .max_draft_tokens
            .min(backend.max_draft_tokens())
            .min(config.max_tokens.saturating_sub(output.len()));
        if proposal_count == 0 {
            break;
        }
        let mut draft_state = backend.begin_draft(&target_state, last, stream)?;
        let mut proposed = Vec::with_capacity(proposal_count);
        let mut draft_logits = Vec::with_capacity(proposal_count);
        for _ in 0..proposal_count {
            let previous = proposed.last().copied().unwrap_or(last);
            let raw = backend.draft_logits(&mut draft_state, previous, stream)?;
            let mut history = output.clone();
            history.extend(proposed.iter().copied());
            let processed = sampler.process_logits(&raw, config.temperature, &history, stream)?;
            let token = sampler.sample_processed(
                &processed,
                config.temperature,
                prng_state.as_mut(),
                stream,
            )?;
            eval([&token])?;
            let token = token.item::<u32>(stream);
            proposed.push(token);
            draft_logits.push(processed);
            if config.eos_token_ids.contains(&token) {
                break;
            }
        }

        let mut verify_ids = Vec::with_capacity(proposed.len() + 1);
        verify_ids.push(last);
        verify_ids.extend(proposed.iter().copied());
        let verify_input = Array::from_slice(&verify_ids, &[1, verify_ids.len() as i32]);
        let checkpoint = B::checkpoint(cache);
        let verification = backend.verify(&verify_input, cache, stream)?;
        let target_raw = B::verification_logits(&verification);
        stats.target_tokens += verify_ids.len();
        stats.draft_tokens += proposed.len();

        let mut accepted = 0usize;
        let mut replacement = None;
        for (index, (&token, draft)) in proposed.iter().zip(&draft_logits).enumerate() {
            let raw = target_raw.try_index_device((.., index as i32, ..), stream)?;
            let mut history = output.clone();
            history.extend(proposed[..index].iter().copied());
            let target = sampler.process_logits(&raw, config.temperature, &history, stream)?;
            if config.temperature == 0.0 {
                let chosen = sampler
                    .sample_processed(&target, 0.0, None, stream)?
                    .item::<u32>(stream);
                if chosen == token {
                    sampler.commit_token(&target, token, stream)?;
                    accepted += 1;
                    continue;
                }
                sampler.commit_token(&target, chosen, stream)?;
                replacement = Some(chosen);
                break;
            }

            let p = probabilities(&target, stream)?;
            let q = probabilities(draft, stream)?;
            let p_token = probability_at(&p, token, stream)?;
            let q_token = probability_at(&q, token, stream)?;
            let acceptance = if q_token <= 0.0 {
                1.0
            } else {
                (p_token / q_token).min(1.0)
            };
            if uniform(prng_state.as_mut(), stream)? <= acceptance {
                sampler.commit_token(&target, token, stream)?;
                accepted += 1;
                continue;
            }
            let chosen = sample_residual(
                &p,
                &q,
                &target,
                sampler,
                config.temperature,
                prng_state.as_mut(),
                stream,
            )?;
            sampler.commit_token(&target, chosen, stream)?;
            replacement = Some(chosen);
            break;
        }

        if accepted == proposed.len()
            && output.len() + accepted < config.max_tokens
            && !proposed
                .last()
                .is_some_and(|token| config.eos_token_ids.contains(token))
        {
            let raw = target_raw.try_index_device((.., accepted as i32, ..), stream)?;
            let mut history = output.clone();
            history.extend(proposed.iter().copied());
            let target = sampler.process_logits(&raw, config.temperature, &history, stream)?;
            let chosen = sampler
                .sample_processed(&target, config.temperature, prng_state.as_mut(), stream)?
                .item::<u32>(stream);
            sampler.commit_token(&target, chosen, stream)?;
            replacement = Some(chosen);
        }

        stats.accepted_tokens += accepted;
        stats.accept_lens.push(accepted);
        stats.rounds += 1;
        let verified_inputs = 1 + accepted;
        let commit = backend.commit_verification(
            verification,
            draft_state,
            cache,
            checkpoint,
            verified_inputs,
            stream,
        )?;
        target_state = commit.state;
        stats.target_tokens += commit.replayed_tokens;

        let mut emitted = proposed[..accepted].to_vec();
        if output.len() + emitted.len() < config.max_tokens {
            if let Some(token) = replacement {
                emitted.push(token);
            }
        }
        let mut stopped = false;
        for token in emitted {
            output.push(token);
            stats.emitted_tokens += 1;
            on_token(token)?;
            if config.eos_token_ids.contains(&token) || output.len() == config.max_tokens {
                stopped = true;
                break;
            }
        }
        if stopped {
            break;
        }
    }

    stats.elapsed = started.elapsed();
    Ok((output, stats))
}

fn validate_input(input: ModelInput<'_>) -> Result<(), Exception> {
    if input.parts.is_empty() {
        return Err(Exception::custom(
            "MTP input must contain at least one part",
        ));
    }
    if input
        .parts
        .iter()
        .all(|part| part.modality == Modality::Text)
    {
        let mut tokens = 0i32;
        for part in input.parts {
            let InputPayload::TokenIds(ids) = part.payload else {
                return Err(Exception::custom(
                    "MTP text input must contain token-id payloads",
                ));
            };
            if ids.ndim() != 2 {
                return Err(Exception::custom(format!(
                    "MTP text token ids must have rank 2, got {:?}",
                    ids.shape()
                )));
            }
            tokens = tokens.saturating_add(ids.dim(1));
        }
        if tokens == 0 {
            return Err(Exception::custom("MTP text input contains no tokens"));
        }
    }
    Ok(())
}

fn probabilities(logits: &Array, stream: &Stream) -> Result<Array, Exception> {
    softmax_axis(&logits.as_type::<f32>(stream)?, -1, true, stream)
}

fn probability_at(probabilities: &Array, token: u32, stream: &Stream) -> Result<f32, Exception> {
    if token as i32 >= probabilities.dim(-1) {
        return Err(Exception::custom(format!(
            "sampled token {token} exceeds vocabulary size {}",
            probabilities.dim(-1)
        )));
    }
    let value = match probabilities.ndim() {
        2 => probabilities.try_index_device((0, token as i32), stream)?,
        3 => probabilities.try_index_device((0, 0, token as i32), stream)?,
        ndim => {
            return Err(Exception::custom(format!(
                "speculative distribution must be rank 2 or 3, got rank {ndim}"
            )))
        }
    };
    Ok(value.item::<f32>(stream))
}

fn uniform(state: Option<&mut RandomState>, stream: &Stream) -> Result<f32, Exception> {
    let state = state.ok_or_else(|| Exception::custom("stochastic MTP requires a PRNG key"))?;
    let key = state.next_key(stream)?;
    Ok(random::uniform::<_, f32>(0.0, 1.0, &[1], &key, stream)?.item::<f32>(stream))
}

fn sample_residual<S: SpeculativeSampler>(
    target_probabilities: &Array,
    draft_probabilities: &Array,
    target_logits: &Array,
    sampler: &mut S,
    temperature: f32,
    prng_state: Option<&mut RandomState>,
    stream: &Stream,
) -> Result<u32, Exception> {
    let residual = maximum(
        target_probabilities.subtract(draft_probabilities, stream)?,
        Array::from_f32(0.0),
        stream,
    )?;
    let mass = residual.sum(None, stream)?.item::<f32>(stream);
    let logits = if mass <= f32::EPSILON {
        target_logits.clone()
    } else {
        residual.log(stream)?
    };
    Ok(sampler
        .sample_processed(&logits, temperature, prng_state, stream)?
        .item::<u32>(stream))
}

#[cfg(test)]
mod tests {
    use safemlx::{Device, DeviceType, ExecutionContext};

    use super::*;
    use crate::{
        models::input::InputPart,
        sampler::{DefaultSampler, MirostatV2Sampler},
    };

    struct ScriptedBackend {
        reject_first: bool,
        accept_second: bool,
    }

    impl MtpBackend for ScriptedBackend {
        type Cache = usize;
        type TargetState = ();
        type DraftState = usize;
        type CacheCheckpoint = usize;
        type Verification = Array;

        fn max_draft_tokens(&self) -> usize {
            2
        }

        fn prefill(
            &mut self,
            _input: ModelInput<'_>,
            cache: &mut Self::Cache,
            _stream: &Stream,
        ) -> Result<MtpPrefill<Self::TargetState>, Exception> {
            *cache = 1;
            Ok(MtpPrefill {
                logits: Array::from_slice(&[0.0f32, 10.0, 0.0], &[1, 3]),
                state: (),
                evaluated_tokens: 1,
            })
        }

        fn begin_draft(
            &mut self,
            _state: &Self::TargetState,
            _last_token: u32,
            _stream: &Stream,
        ) -> Result<Self::DraftState, Exception> {
            Ok(0)
        }

        fn draft_logits(
            &mut self,
            state: &mut Self::DraftState,
            _last_token: u32,
            _stream: &Stream,
        ) -> Result<Array, Exception> {
            let logits = if *state == 0 {
                Array::from_slice(&[0.0f32, 0.0, 10.0], &[1, 1, 3])
            } else {
                Array::from_slice(&[10.0f32, 0.0, 0.0], &[1, 1, 3])
            };
            *state += 1;
            Ok(logits)
        }

        fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
            *cache
        }

        fn verify(
            &mut self,
            input_tokens: &Array,
            cache: &mut Self::Cache,
            _stream: &Stream,
        ) -> Result<Self::Verification, Exception> {
            *cache += input_tokens.dim(1) as usize;
            let first = if self.reject_first {
                [0.0f32, 10.0, 0.0]
            } else {
                [0.0f32, 0.0, 10.0]
            };
            let second = if self.accept_second {
                [10.0f32, 0.0, 0.0]
            } else {
                [0.0f32, 10.0, 0.0]
            };
            Ok(Array::from_slice(
                &[
                    first[0], first[1], first[2], second[0], second[1], second[2], 10.0, 0.0, 0.0,
                ],
                &[1, 3, 3],
            ))
        }

        fn verification_logits(output: &Self::Verification) -> &Array {
            output
        }

        fn commit_verification(
            &mut self,
            _output: Self::Verification,
            _draft_state: Self::DraftState,
            cache: &mut Self::Cache,
            checkpoint: Self::CacheCheckpoint,
            verified_inputs: usize,
            _stream: &Stream,
        ) -> Result<MtpCommit<Self::TargetState>, Exception> {
            *cache = checkpoint + verified_inputs;
            Ok(MtpCommit {
                state: (),
                replayed_tokens: 0,
            })
        }
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn greedy_engine_commits_only_the_accepted_prefix() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        };
        let mut cache = 0;
        let mut emitted = Vec::new();
        let (tokens, stats) = generate_with_callback(
            &mut ScriptedBackend {
                reject_first: false,
                accept_second: false,
            },
            &mut cache,
            input,
            &config,
            None,
            &mut DefaultSampler,
            stream,
            |token| {
                emitted.push(token);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 1]);
        assert_eq!(emitted, tokens);
        assert_eq!(stats.accept_lens, vec![1]);
        assert_eq!(stats.accepted_tokens, 1);
        assert_eq!(cache, 3);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn mirostat_v2_mtp_commits_accepted_target_distributions() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 4,
            max_draft_tokens: 4,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        let mut cache = 0;
        let mut sampler = MirostatV2Sampler::default();
        let key = safemlx::random::key(7).unwrap();

        let (tokens, stats) = generate(
            &mut ScriptedBackend {
                reject_first: false,
                accept_second: true,
            },
            &mut cache,
            input,
            &config,
            Some(key),
            &mut sampler,
            stream,
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 0, 0]);
        assert_eq!(stats.draft_tokens, 2);
        assert_eq!(stats.accepted_tokens, 2);
        assert_eq!(sampler.generated_tokens(), tokens);
        assert!((sampler.mu() - 12.0).abs() < 1e-4);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn mirostat_v2_mtp_commits_replacement_not_rejected_draft() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 2,
            max_draft_tokens: 4,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        let mut cache = 0;
        let mut sampler = MirostatV2Sampler::default();
        let key = safemlx::random::key(11).unwrap();

        let (tokens, stats) = generate(
            &mut ScriptedBackend {
                reject_first: true,
                accept_second: false,
            },
            &mut cache,
            input,
            &config,
            Some(key),
            &mut sampler,
            stream,
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 1]);
        assert_eq!(stats.draft_tokens, 1);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(sampler.generated_tokens(), tokens);
        assert!((sampler.mu() - 11.0).abs() < 1e-4);
    }

    #[test]
    fn empty_stats_have_zero_acceptance_rate() {
        assert_eq!(MtpStats::default().accept_rate(), 0.0);
    }
}
