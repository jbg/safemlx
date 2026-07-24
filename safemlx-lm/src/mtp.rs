//! Architecture-independent multi-token prediction and speculative decoding.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, maximum, softmax_axis},
    random::{self, RandomState},
    transforms::{async_eval, eval},
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

/// Target and draft streams used by one MTP generation sequence.
///
/// A single-stream execution preserves the original behavior. Supplying
/// distinct streams places target prefill/verification and accepted-token
/// sampling on `target`, while proposal generation runs on `draft`. Submitted
/// target verification remains unresolved while eligible draft work runs;
/// explicit synchronization is retained at state/data dependency boundaries.
#[derive(Debug, Clone, Copy)]
pub struct MtpExecutionStreams<'a> {
    target: &'a Stream,
    draft: &'a Stream,
}

impl<'a> MtpExecutionStreams<'a> {
    /// Creates an execution assignment with explicit target and draft streams.
    pub const fn new(target: &'a Stream, draft: &'a Stream) -> Self {
        Self { target, draft }
    }

    /// Creates the legacy assignment in which all MTP work uses one stream.
    pub const fn single(stream: &'a Stream) -> Self {
        Self {
            target: stream,
            draft: stream,
        }
    }

    /// Returns the stream used for target prefill and verification.
    pub const fn target(self) -> &'a Stream {
        self.target
    }

    /// Returns the stream used for proposal generation.
    pub const fn draft(self) -> &'a Stream {
        self.draft
    }

    /// Returns whether target and draft work use different streams.
    pub fn is_split(self) -> bool {
        self.target != self.draft
    }
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
    /// Tokens drafted on an optimistic continuation while target verification was in flight.
    pub optimistic_draft_tokens: usize,
    /// Optimistic continuation blocks drafted.
    pub optimistic_draft_blocks: usize,
    /// Optimistically drafted tokens promoted after full acceptance.
    pub reused_optimistic_tokens: usize,
    /// Optimistic continuation blocks promoted after full acceptance.
    pub reused_optimistic_blocks: usize,
    /// Optimistically drafted tokens discarded after rejection, EOS, or completion.
    pub discarded_optimistic_tokens: usize,
    /// Optimistic continuation blocks discarded after rejection, EOS, or completion.
    pub discarded_optimistic_blocks: usize,
    /// Scheduler operations performed for this request.
    pub scheduler_turns: usize,
    /// Times this request was drafted while another request had target work in flight.
    pub cross_request_draft_opportunities: usize,
    /// Wall-clock generation duration.
    pub elapsed: Duration,
}

/// Aggregate bounded-scheduler telemetry.
#[derive(Debug, Clone, Default)]
pub struct MtpSchedulerStats {
    /// Total scheduler operations.
    pub turns: usize,
    /// Draft turns performed for a request while another request was being verified.
    pub cross_request_draft_opportunities: usize,
    /// Maximum simultaneously retained target verification transactions.
    pub peak_in_flight_verifications: usize,
    /// Maximum simultaneously retained optimistic draft branches.
    pub peak_optimistic_branches: usize,
}

/// Completed independently progressing text batch.
#[derive(Debug, Clone, Default)]
pub struct MtpBatchOutput {
    /// Generated token ids in original batch-lane order.
    pub token_ids: Vec<Vec<u32>>,
    /// Per-lane speculative statistics in the same order.
    pub stats: Vec<MtpStats>,
    /// Aggregate fair-scheduler telemetry.
    pub scheduler: MtpSchedulerStats,
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
    type DraftState: Clone;
    /// Architecture-specific cache transaction marker.
    type CacheCheckpoint;
    /// Architecture-specific target verification result.
    type Verification;

    /// Maximum proposals supported by this backend in one round.
    fn max_draft_tokens(&self) -> usize {
        usize::MAX
    }

    /// Returns whether a draft state can continue across a fully accepted
    /// verification block without fresh target-to-draft state handoff.
    ///
    /// External Gemma assistants support this. Embedded Qwen paths deliberately
    /// return `false`: their target-owned MTP cache is advanced during commit.
    fn supports_optimistic_lookahead(&self) -> bool {
        false
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

    /// Starts one draft round with explicit target and draft streams.
    ///
    /// Existing backends may rely on the default draft-stream-only behavior.
    /// Backends that must transfer committed target state should override this
    /// method.
    fn begin_draft_with_streams(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        self.begin_draft(state, last_token, streams.draft())
    }

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

    /// Commits verification with explicit target and draft streams.
    ///
    /// The default preserves the original target-stream commit behavior.
    fn commit_verification_with_streams(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<MtpCommit<Self::TargetState>, Exception> {
        self.commit_verification(
            output,
            draft_state,
            cache,
            checkpoint,
            verified_inputs,
            streams.target(),
        )
    }
}

/// Bounded fair-scheduler configuration.
#[derive(Debug, Clone, Copy)]
pub struct MtpSchedulerOptions {
    /// Maximum submitted target verification transactions retained at once.
    pub max_in_flight_verifications: usize,
    /// Maximum retained optimistic continuation branches.
    pub max_optimistic_branches: usize,
    /// Number of proposal blocks drafted ahead for one request.
    ///
    /// The current scheduler supports zero or one. One is the default because
    /// it overlaps useful CPU work without multiplying speculative memory.
    pub lookahead_blocks: usize,
}

impl Default for MtpSchedulerOptions {
    fn default() -> Self {
        Self {
            max_in_flight_verifications: 1,
            max_optimistic_branches: 1,
            lookahead_blocks: 1,
        }
    }
}

/// Stable scheduler-local request identifier.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct MtpRequestId(usize);

impl MtpRequestId {
    /// Returns the scheduler insertion index.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Explicit request/round state.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MtpRequestPhase {
    /// Target prompt prefill and first-token sampling.
    Prefill,
    /// Committed target state is ready to seed a proposal block.
    ReadyToDraft,
    /// A proposal block is ready for target submission.
    ReadyToSubmitVerification,
    /// Target verification has been submitted and remains unresolved.
    TargetVerificationInFlight,
    /// A same-request continuation is being drafted under full acceptance.
    OptimisticDraftInProgress,
    /// Target verification remains in flight and its optimistic branch is ready.
    OptimisticDraftReady,
    /// Target results are being accepted/rejected and committed.
    VerificationResolution,
    /// The request reached EOS or its token limit.
    Completed,
    /// The request was cancelled independently.
    Cancelled,
}

struct DraftBlock<D> {
    state: D,
    tokens: Vec<u32>,
    distributions: Vec<Array>,
}

struct OptimisticBranch<D> {
    block: DraftBlock<D>,
    draft_prng: Option<RandomState>,
}

struct InFlight<B: MtpBackend> {
    verification: B::Verification,
    checkpoint: B::CacheCheckpoint,
    block: DraftBlock<B::DraftState>,
    optimistic: Option<OptimisticBranch<B::DraftState>>,
}

struct ScheduledRequest<'a, B: MtpBackend, S> {
    id: MtpRequestId,
    cache: &'a mut B::Cache,
    config: MtpConfig,
    sampler: S,
    target_prng: Option<RandomState>,
    draft_prng: Option<RandomState>,
    output: Vec<u32>,
    stats: MtpStats,
    started: Instant,
    target_state: Option<B::TargetState>,
    block: Option<DraftBlock<B::DraftState>>,
    in_flight: Option<InFlight<B>>,
    phase: MtpRequestPhase,
    cancel_requested: bool,
    on_token: Box<dyn FnMut(u32) -> Result<(), Exception> + 'a>,
}

/// One completed scheduled request.
pub struct MtpRequestOutput<S> {
    /// Stable request identifier.
    pub id: MtpRequestId,
    /// Generated tokens in request order.
    pub token_ids: Vec<u32>,
    /// Per-request statistics.
    pub stats: MtpStats,
    /// Final canonical sampler state.
    pub sampler: S,
    /// Whether the request was cancelled.
    pub cancelled: bool,
}

/// Completed scheduler output.
pub struct MtpScheduleOutput<S> {
    /// Requests in submission order.
    pub requests: Vec<MtpRequestOutput<S>>,
    /// Aggregate scheduler telemetry.
    pub scheduler: MtpSchedulerStats,
}

/// Single-threaded fair MTP request scheduler.
///
/// MLX streams already provide asynchronous device queues. The scheduler stays
/// deliberately single-threaded: it submits lazy target graphs, performs CPU
/// draft work, and synchronizes only when a verification result is resolved.
/// Model parameters are shared through the backend; every request owns its
/// cache, target state, sampler, PRNG substreams, output, and statistics.
pub struct MtpScheduler<'a, B: MtpBackend, S> {
    backend: &'a mut B,
    streams: MtpExecutionStreams<'a>,
    options: MtpSchedulerOptions,
    requests: Vec<ScheduledRequest<'a, B, S>>,
    cursor: usize,
    stats: MtpSchedulerStats,
}

impl<'a, B, S> MtpScheduler<'a, B, S>
where
    B: MtpBackend,
    S: SpeculativeSampler + Clone,
{
    /// Creates a scheduler over shared model parameters and explicit streams.
    pub fn new(
        backend: &'a mut B,
        streams: MtpExecutionStreams<'a>,
        options: MtpSchedulerOptions,
    ) -> Result<Self, Exception> {
        if options.max_in_flight_verifications == 0 {
            return Err(Exception::custom(
                "MTP max_in_flight_verifications must be positive",
            ));
        }
        if options.lookahead_blocks > 1 {
            return Err(Exception::custom(
                "MTP scheduler currently supports at most one lookahead block",
            ));
        }
        if options.lookahead_blocks > 0 && options.max_optimistic_branches == 0 {
            return Err(Exception::custom(
                "MTP lookahead requires at least one optimistic branch slot",
            ));
        }
        Ok(Self {
            backend,
            streams,
            options,
            requests: Vec::new(),
            cursor: 0,
            stats: MtpSchedulerStats::default(),
        })
    }

    /// Submits one independent request.
    #[allow(clippy::too_many_arguments)]
    pub fn submit<F>(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: MtpConfig,
        prng_key: Option<Array>,
        sampler: S,
        on_token: F,
    ) -> Result<MtpRequestId, Exception>
    where
        F: FnMut(u32) -> Result<(), Exception> + 'a,
    {
        validate_config(self.backend, &config, prng_key.as_ref())?;
        let id = MtpRequestId(self.requests.len());
        let started = Instant::now();
        if config.max_tokens == 0 {
            self.requests.push(ScheduledRequest {
                id,
                cache,
                config,
                sampler,
                target_prng: None,
                draft_prng: None,
                output: Vec::new(),
                stats: MtpStats::default(),
                started,
                target_state: None,
                block: None,
                in_flight: None,
                phase: MtpRequestPhase::Completed,
                cancel_requested: false,
                on_token: Box::new(on_token),
            });
            return Ok(id);
        }

        validate_input(input)?;
        let (target_prng, draft_prng) =
            split_random_states(prng_key, config.temperature, self.streams)?;
        self.requests.push(ScheduledRequest {
            id,
            cache,
            config,
            sampler,
            target_prng,
            draft_prng,
            output: Vec::new(),
            stats: MtpStats::default(),
            started,
            target_state: None,
            block: None,
            in_flight: None,
            phase: MtpRequestPhase::Prefill,
            cancel_requested: false,
            on_token: Box::new(on_token),
        });

        let prefill_result = {
            let request = &mut self.requests[id.0];
            (|| {
                let prefill = self
                    .backend
                    .prefill(input, request.cache, self.streams.target())?;
                request.stats.target_tokens = prefill.evaluated_tokens;
                request.stats.scheduler_turns = 1;
                let first_logits = request.sampler.process_logits(
                    &prefill.logits,
                    request.config.temperature,
                    &[],
                    self.streams.target(),
                )?;
                let first = request.sampler.sample_processed(
                    &first_logits,
                    request.config.temperature,
                    request.target_prng.as_mut(),
                    self.streams.target(),
                )?;
                eval([&first])?;
                let first = first.item::<u32>(self.streams.target());
                request
                    .sampler
                    .commit_token(&first_logits, first, self.streams.target())?;
                (request.on_token)(first)?;
                request.output.push(first);
                request.stats.emitted_tokens = 1;
                request.target_state = Some(prefill.state);
                let completed =
                    request.config.eos_token_ids.contains(&first) || request.config.max_tokens == 1;
                request.phase = if completed {
                    request.stats.elapsed = request.started.elapsed();
                    MtpRequestPhase::Completed
                } else {
                    MtpRequestPhase::ReadyToDraft
                };
                Ok::<(), Exception>(())
            })()
        };
        if let Err(error) = prefill_result {
            self.requests.pop();
            return Err(error);
        }
        self.stats.turns += 1;
        Ok(id)
    }

    /// Returns the current phase for a submitted request.
    pub fn phase(&self, id: MtpRequestId) -> Option<MtpRequestPhase> {
        self.requests.get(id.0).map(|request| request.phase)
    }

    /// Requests independent cancellation.
    ///
    /// An in-flight target transaction is resolved to a safe cache boundary
    /// before the request enters `Cancelled`; other requests continue.
    pub fn cancel(&mut self, id: MtpRequestId) -> Result<(), Exception> {
        let request = self
            .requests
            .get_mut(id.0)
            .ok_or_else(|| Exception::custom("unknown MTP request id"))?;
        if matches!(
            request.phase,
            MtpRequestPhase::Completed | MtpRequestPhase::Cancelled
        ) {
            return Ok(());
        }
        if request.in_flight.is_some() {
            request.cancel_requested = true;
        } else {
            request.block = None;
            request.phase = MtpRequestPhase::Cancelled;
            request.stats.elapsed = request.started.elapsed();
        }
        Ok(())
    }

    /// Returns whether every request is completed or cancelled.
    pub fn is_finished(&self) -> bool {
        self.requests.iter().all(|request| {
            matches!(
                request.phase,
                MtpRequestPhase::Completed | MtpRequestPhase::Cancelled
            )
        })
    }

    /// Performs one fair scheduler operation.
    ///
    /// Returns `false` when every request is terminal.
    pub fn step(&mut self) -> Result<bool, Exception> {
        if self.is_finished() {
            return Ok(false);
        }

        let in_flight = self.in_flight_count();
        if in_flight < self.options.max_in_flight_verifications {
            if let Some(index) =
                self.select(|request| request.phase == MtpRequestPhase::ReadyToSubmitVerification)
            {
                self.submit_verification(index)?;
                return Ok(true);
            }
        }

        if in_flight > 0 {
            if self.optimistic_count() < self.options.max_optimistic_branches
                && self.options.lookahead_blocks > 0
                && self.streams.is_split()
            {
                if let Some(index) = self.select(|request| {
                    matches!(request.phase, MtpRequestPhase::TargetVerificationInFlight)
                        && request
                            .in_flight
                            .as_ref()
                            .is_some_and(|flight| flight.optimistic.is_none())
                }) {
                    if self.can_optimistically_draft(index) {
                        self.draft_optimistic(index)?;
                        return Ok(true);
                    }
                }
            }

            if let Some(index) =
                self.select(|request| request.phase == MtpRequestPhase::ReadyToDraft)
            {
                self.draft_committed(index, true)?;
                return Ok(true);
            }

            if let Some(index) = self.select(|request| {
                matches!(
                    request.phase,
                    MtpRequestPhase::TargetVerificationInFlight
                        | MtpRequestPhase::OptimisticDraftReady
                )
            }) {
                self.resolve_verification(index)?;
                return Ok(true);
            }
        } else if let Some(index) =
            self.select(|request| request.phase == MtpRequestPhase::ReadyToDraft)
        {
            self.draft_committed(index, false)?;
            return Ok(true);
        }

        Err(Exception::custom(
            "MTP scheduler reached a non-terminal state with no eligible operation",
        ))
    }

    /// Drives all submitted requests to completion.
    pub fn run(&mut self) -> Result<(), Exception> {
        while self.step()? {}
        Ok(())
    }

    /// Consumes a finished scheduler and returns results in submission order.
    pub fn finish(self) -> Result<MtpScheduleOutput<S>, Exception> {
        if !self.is_finished() {
            return Err(Exception::custom(
                "cannot finish an MTP scheduler with active requests",
            ));
        }
        Ok(MtpScheduleOutput {
            requests: self
                .requests
                .into_iter()
                .map(|request| MtpRequestOutput {
                    id: request.id,
                    token_ids: request.output,
                    stats: request.stats,
                    sampler: request.sampler,
                    cancelled: request.phase == MtpRequestPhase::Cancelled,
                })
                .collect(),
            scheduler: self.stats,
        })
    }

    fn select(&mut self, predicate: impl Fn(&ScheduledRequest<'a, B, S>) -> bool) -> Option<usize> {
        let len = self.requests.len();
        for offset in 0..len {
            let index = (self.cursor + offset) % len;
            if predicate(&self.requests[index]) {
                self.cursor = (index + 1) % len;
                return Some(index);
            }
        }
        None
    }

    fn record_turn(&mut self, index: usize) {
        self.stats.turns += 1;
        self.requests[index].stats.scheduler_turns += 1;
    }

    fn in_flight_count(&self) -> usize {
        self.requests
            .iter()
            .filter(|request| request.in_flight.is_some())
            .count()
    }

    fn optimistic_count(&self) -> usize {
        self.requests
            .iter()
            .filter(|request| {
                request
                    .in_flight
                    .as_ref()
                    .is_some_and(|flight| flight.optimistic.is_some())
            })
            .count()
    }

    fn draft_committed(&mut self, index: usize, cross_request: bool) -> Result<(), Exception> {
        self.record_turn(index);
        let backend_limit = self.backend.max_draft_tokens();
        let request = &mut self.requests[index];
        let count = request.config.max_draft_tokens.min(backend_limit).min(
            request
                .config
                .max_tokens
                .saturating_sub(request.output.len()),
        );
        if count == 0 {
            request.phase = MtpRequestPhase::Completed;
            request.stats.elapsed = request.started.elapsed();
            return Ok(());
        }
        let last = *request.output.last().expect("prefill emitted a token");
        let target_state = request
            .target_state
            .as_ref()
            .expect("ready request has target state");
        let mut state = self
            .backend
            .begin_draft_with_streams(target_state, last, self.streams)?;
        let (tokens, distributions) = draft_block(
            self.backend,
            &mut state,
            last,
            count,
            &request.output,
            &request.config,
            &request.sampler,
            &mut request.draft_prng,
            self.streams.draft(),
        )?;
        request.stats.draft_tokens += tokens.len();
        if cross_request {
            request.stats.cross_request_draft_opportunities += 1;
            self.stats.cross_request_draft_opportunities += 1;
        }
        request.block = Some(DraftBlock {
            state,
            tokens,
            distributions,
        });
        request.phase = MtpRequestPhase::ReadyToSubmitVerification;
        Ok(())
    }

    fn submit_verification(&mut self, index: usize) -> Result<(), Exception> {
        self.record_turn(index);
        let request = &mut self.requests[index];
        let block = request
            .block
            .take()
            .expect("verification-ready request has a draft block");
        let mut verify_ids = Vec::with_capacity(block.tokens.len() + 1);
        verify_ids.push(*request.output.last().expect("prefill emitted a token"));
        verify_ids.extend(block.tokens.iter().copied());
        let verify_input = Array::from_slice(&verify_ids, &[1, verify_ids.len() as i32]);
        let verify_input = if self.streams.is_split() {
            verify_input.copy(self.streams.target())?
        } else {
            verify_input
        };
        let checkpoint = B::checkpoint(request.cache);
        let verification =
            self.backend
                .verify(&verify_input, request.cache, self.streams.target())?;
        async_eval([B::verification_logits(&verification)])?;
        request.stats.target_tokens += verify_ids.len();
        request.in_flight = Some(InFlight {
            verification,
            checkpoint,
            block,
            optimistic: None,
        });
        request.phase = MtpRequestPhase::TargetVerificationInFlight;
        self.stats.peak_in_flight_verifications = self
            .stats
            .peak_in_flight_verifications
            .max(self.in_flight_count());
        Ok(())
    }

    fn can_optimistically_draft(&self, index: usize) -> bool {
        let request = &self.requests[index];
        let Some(flight) = request.in_flight.as_ref() else {
            return false;
        };
        self.backend.supports_optimistic_lookahead()
            && request.sampler.supports_optimistic_lookahead()
            && !flight.block.tokens.is_empty()
            && !flight
                .block
                .tokens
                .last()
                .is_some_and(|token| request.config.eos_token_ids.contains(token))
            && request.output.len() + flight.block.tokens.len() < request.config.max_tokens
    }

    fn draft_optimistic(&mut self, index: usize) -> Result<(), Exception> {
        self.record_turn(index);
        let backend_limit = self.backend.max_draft_tokens();
        let request = &mut self.requests[index];
        request.phase = MtpRequestPhase::OptimisticDraftInProgress;
        let flight = request
            .in_flight
            .as_mut()
            .expect("optimistic request has an in-flight verification");
        let assumed_len = request.output.len() + flight.block.tokens.len();
        let count = request
            .config
            .max_draft_tokens
            .min(backend_limit)
            .min(request.config.max_tokens.saturating_sub(assumed_len));
        let mut state = flight.block.state.clone();
        let last = *flight
            .block
            .tokens
            .last()
            .expect("optimistic block has an assumed token");
        let mut history = Vec::with_capacity(assumed_len);
        history.extend_from_slice(&request.output);
        history.extend_from_slice(&flight.block.tokens);
        let mut branch_prng = request.draft_prng.clone();
        let (tokens, distributions) = draft_block(
            self.backend,
            &mut state,
            last,
            count,
            &history,
            &request.config,
            &request.sampler,
            &mut branch_prng,
            self.streams.draft(),
        )?;
        request.stats.optimistic_draft_tokens += tokens.len();
        request.stats.optimistic_draft_blocks += 1;
        flight.optimistic = Some(OptimisticBranch {
            block: DraftBlock {
                state,
                tokens,
                distributions,
            },
            draft_prng: branch_prng,
        });
        request.phase = MtpRequestPhase::OptimisticDraftReady;
        self.stats.peak_optimistic_branches = self
            .stats
            .peak_optimistic_branches
            .max(self.optimistic_count());
        Ok(())
    }

    fn resolve_verification(&mut self, index: usize) -> Result<(), Exception> {
        self.record_turn(index);
        let request = &mut self.requests[index];
        request.phase = MtpRequestPhase::VerificationResolution;
        let mut flight = request
            .in_flight
            .take()
            .expect("resolving request has an in-flight verification");
        let DraftBlock {
            state,
            tokens: proposed,
            distributions,
        } = flight.block;

        if request.cancel_requested {
            discard_optimistic(&mut request.stats, flight.optimistic.take());
            let commit = self.backend.commit_verification_with_streams(
                flight.verification,
                state,
                request.cache,
                flight.checkpoint,
                1,
                self.streams,
            )?;
            request.stats.target_tokens += commit.replayed_tokens;
            request.phase = MtpRequestPhase::Cancelled;
            request.stats.elapsed = request.started.elapsed();
            return Ok(());
        }

        let distributions = if request.config.temperature != 0.0 && self.streams.is_split() {
            eval(distributions.iter())?;
            self.streams.draft().synchronize()?;
            distributions
                .iter()
                .map(|distribution| distribution.copy(self.streams.target()))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            distributions
        };
        let target_raw = B::verification_logits(&flight.verification);
        let mut history = request.output.clone();
        let mut accepted = 0usize;
        let mut replacement = None;
        for (index, (&token, draft)) in proposed.iter().zip(&distributions).enumerate() {
            let raw = target_raw.try_index_device((.., index as i32, ..), self.streams.target())?;
            let target = request.sampler.process_logits(
                &raw,
                request.config.temperature,
                &history,
                self.streams.target(),
            )?;
            if request.config.temperature == 0.0 {
                let chosen = request
                    .sampler
                    .sample_processed(&target, 0.0, None, self.streams.target())?
                    .item::<u32>(self.streams.target());
                if chosen == token {
                    request
                        .sampler
                        .commit_token(&target, token, self.streams.target())?;
                    accepted += 1;
                    history.push(token);
                    continue;
                }
                request
                    .sampler
                    .commit_token(&target, chosen, self.streams.target())?;
                replacement = Some(chosen);
                break;
            }

            let p = probabilities(&target, self.streams.target())?;
            let q = probabilities(draft, self.streams.target())?;
            let p_token = probability_at(&p, token, self.streams.target())?;
            let q_token = probability_at(&q, token, self.streams.target())?;
            let acceptance = if q_token <= 0.0 {
                1.0
            } else {
                (p_token / q_token).min(1.0)
            };
            if uniform(request.target_prng.as_mut(), self.streams.target())? <= acceptance {
                request
                    .sampler
                    .commit_token(&target, token, self.streams.target())?;
                accepted += 1;
                history.push(token);
                continue;
            }
            let chosen = sample_residual(
                &p,
                &q,
                &target,
                &mut request.sampler,
                request.config.temperature,
                request.target_prng.as_mut(),
                self.streams.target(),
            )?;
            request
                .sampler
                .commit_token(&target, chosen, self.streams.target())?;
            replacement = Some(chosen);
            break;
        }

        // A target bonus is safe only when no optimistic continuation was
        // rooted at the last proposal. Pipelined rounds omit it so a promoted
        // branch always matches the exact committed prefix.
        if accepted == proposed.len()
            && flight.optimistic.is_none()
            && request.output.len() + accepted < request.config.max_tokens
            && !proposed
                .last()
                .is_some_and(|token| request.config.eos_token_ids.contains(token))
        {
            let raw =
                target_raw.try_index_device((.., accepted as i32, ..), self.streams.target())?;
            let target = request.sampler.process_logits(
                &raw,
                request.config.temperature,
                &history,
                self.streams.target(),
            )?;
            let chosen = request
                .sampler
                .sample_processed(
                    &target,
                    request.config.temperature,
                    request.target_prng.as_mut(),
                    self.streams.target(),
                )?
                .item::<u32>(self.streams.target());
            request
                .sampler
                .commit_token(&target, chosen, self.streams.target())?;
            replacement = Some(chosen);
        }

        request.stats.accepted_tokens += accepted;
        request.stats.accept_lens.push(accepted);
        request.stats.rounds += 1;
        // The target cache intentionally trails the emitted output by one
        // token: the next verification processes that token as its leading
        // input. A rejection replacement or target bonus is not part of the
        // verification inputs, so retaining `1 + accepted` leaves that emitted
        // token uncached. When full acceptance emits no bonus, however, the
        // last accepted proposal itself is the trailing emitted token and must
        // be excluded from the retained inputs.
        let verified_inputs = if replacement.is_some() {
            1 + accepted
        } else {
            debug_assert_eq!(accepted, proposed.len());
            accepted
        };
        let commit = self.backend.commit_verification_with_streams(
            flight.verification,
            state,
            request.cache,
            flight.checkpoint,
            verified_inputs,
            self.streams,
        )?;
        request.stats.target_tokens += commit.replayed_tokens;
        request.target_state = Some(commit.state);

        let mut stopped = false;
        for token in proposed[..accepted].iter().copied().chain(replacement) {
            if request.output.len() == request.config.max_tokens {
                break;
            }
            request.output.push(token);
            request.stats.emitted_tokens += 1;
            (request.on_token)(token)?;
            if request.config.eos_token_ids.contains(&token)
                || request.output.len() == request.config.max_tokens
            {
                stopped = true;
                break;
            }
        }

        if stopped {
            discard_optimistic(&mut request.stats, flight.optimistic.take());
            request.phase = MtpRequestPhase::Completed;
            request.stats.elapsed = request.started.elapsed();
        } else if accepted == proposed.len() {
            if let Some(branch) = flight.optimistic.take() {
                request.stats.draft_tokens += branch.block.tokens.len();
                request.stats.reused_optimistic_tokens += branch.block.tokens.len();
                request.stats.reused_optimistic_blocks += 1;
                request.draft_prng = branch.draft_prng;
                request.block = Some(branch.block);
                request.phase = MtpRequestPhase::ReadyToSubmitVerification;
            } else {
                request.phase = MtpRequestPhase::ReadyToDraft;
            }
        } else {
            discard_optimistic(&mut request.stats, flight.optimistic.take());
            request.phase = MtpRequestPhase::ReadyToDraft;
        }
        Ok(())
    }
}

fn discard_optimistic<D>(stats: &mut MtpStats, branch: Option<OptimisticBranch<D>>) {
    if let Some(branch) = branch {
        stats.discarded_optimistic_tokens += branch.block.tokens.len();
        stats.discarded_optimistic_blocks += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn draft_block<B, S>(
    backend: &mut B,
    state: &mut B::DraftState,
    first_previous: u32,
    count: usize,
    base_history: &[u32],
    config: &MtpConfig,
    sampler: &S,
    prng: &mut Option<RandomState>,
    stream: &Stream,
) -> Result<(Vec<u32>, Vec<Array>), Exception>
where
    B: MtpBackend,
    S: SpeculativeSampler + Clone,
{
    let mut branch_sampler = sampler.clone();
    let mut history = Vec::with_capacity(base_history.len() + count);
    history.extend_from_slice(base_history);
    let mut tokens = Vec::with_capacity(count);
    let mut distributions = Vec::with_capacity(count);
    for _ in 0..count {
        let previous = tokens.last().copied().unwrap_or(first_previous);
        let raw = backend.draft_logits(state, previous, stream)?;
        let processed =
            branch_sampler.process_logits(&raw, config.temperature, &history, stream)?;
        let token = branch_sampler.sample_processed(
            &processed,
            config.temperature,
            prng.as_mut(),
            stream,
        )?;
        eval([&token])?;
        let token = token.item::<u32>(stream);
        tokens.push(token);
        distributions.push(processed);
        history.push(token);
        if config.eos_token_ids.contains(&token) {
            break;
        }
    }
    Ok((tokens, distributions))
}

fn validate_config<B: MtpBackend>(
    backend: &B,
    config: &MtpConfig,
    prng_key: Option<&Array>,
) -> Result<(), Exception> {
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
    Ok(())
}

fn split_random_states(
    prng_key: Option<Array>,
    temperature: f32,
    streams: MtpExecutionStreams<'_>,
) -> Result<(Option<RandomState>, Option<RandomState>), Exception> {
    if temperature == 0.0 {
        return Ok((None, None));
    }
    let mut root = RandomState::from_key(prng_key.expect("validated stochastic PRNG key"));
    let target_key = root.next_key(streams.target())?;
    let draft_key = root.next_key(streams.target())?;
    let draft_key = if streams.is_split() {
        eval([&draft_key])?;
        streams.target().synchronize()?;
        let copied = draft_key.copy(streams.draft())?;
        eval([&copied])?;
        streams.draft().synchronize()?;
        copied
    } else {
        draft_key
    };
    Ok((
        Some(RandomState::from_key(target_key)),
        Some(RandomState::from_key(draft_key)),
    ))
}

/// Runs one scheduled speculative request.
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
    S: SpeculativeSampler + Clone,
{
    generate_with_streams(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        MtpExecutionStreams::single(stream),
    )
}

/// Runs one scheduled speculative request with explicit target/draft streams.
pub fn generate_with_streams<B, S>(
    backend: &mut B,
    cache: &mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    streams: MtpExecutionStreams<'_>,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MtpBackend,
    S: SpeculativeSampler + Clone,
{
    generate_with_streams_and_callback(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        streams,
        |_| Ok(()),
    )
}

/// Runs one scheduled request and reports committed tokens.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_callback<B, S, F>(
    backend: &mut B,
    cache: &mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &Stream,
    on_token: F,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MtpBackend,
    S: SpeculativeSampler + Clone,
    F: FnMut(u32) -> Result<(), Exception>,
{
    generate_with_streams_and_callback(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        MtpExecutionStreams::single(stream),
        on_token,
    )
}

/// Runs one scheduled request with explicit streams and a commit callback.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_streams_and_callback<B, S, F>(
    backend: &mut B,
    cache: &mut B::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    streams: MtpExecutionStreams<'_>,
    on_token: F,
) -> Result<(Vec<u32>, MtpStats), Exception>
where
    B: MtpBackend,
    S: SpeculativeSampler + Clone,
    F: FnMut(u32) -> Result<(), Exception>,
{
    let final_sampler;
    let token_ids;
    let stats;
    {
        let mut scheduler = MtpScheduler::new(backend, streams, MtpSchedulerOptions::default())?;
        scheduler.submit(
            cache,
            input,
            config.clone(),
            prng_key,
            sampler.clone(),
            on_token,
        )?;
        scheduler.run()?;
        let mut output = scheduler.finish()?.requests;
        let request = output.pop().expect("one request was submitted");
        final_sampler = request.sampler;
        token_ids = request.token_ids;
        stats = request.stats;
    }
    *sampler = final_sampler;
    Ok((token_ids, stats))
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
    use std::sync::Arc;

    use safemlx::{Device, DeviceType, ExecutionContext};

    use super::*;
    use crate::{
        models::input::InputPart,
        sampler::{DefaultSampler, MirostatV2Sampler},
    };

    #[derive(Clone, Default)]
    struct CountingSampler {
        process_calls: usize,
    }

    impl SpeculativeSampler for CountingSampler {
        fn supports_optimistic_lookahead(&self) -> bool {
            true
        }

        fn process_logits(
            &mut self,
            logits: &Array,
            _temperature: f32,
            _history: &[u32],
            _stream: &Stream,
        ) -> Result<Array, Exception> {
            self.process_calls += 1;
            Ok(logits.clone())
        }
    }

    struct ScriptedBackend {
        first_token: u32,
        rejection_token: u32,
        reject_first: bool,
        accept_second: bool,
        routes: Vec<(&'static str, DeviceType)>,
        draft_storage: Vec<usize>,
    }

    #[derive(Clone)]
    struct ScriptedDraftState {
        step: usize,
        storage: Arc<()>,
    }

    impl ScriptedBackend {
        fn record(&mut self, operation: &'static str, stream: &Stream) -> Result<(), Exception> {
            self.routes
                .push((operation, stream.get_device()?.get_type()?));
            Ok(())
        }
    }

    impl MtpBackend for ScriptedBackend {
        type Cache = usize;
        type TargetState = ();
        type DraftState = ScriptedDraftState;
        type CacheCheckpoint = usize;
        type Verification = Array;

        fn max_draft_tokens(&self) -> usize {
            2
        }

        fn supports_optimistic_lookahead(&self) -> bool {
            true
        }

        fn prefill(
            &mut self,
            _input: ModelInput<'_>,
            cache: &mut Self::Cache,
            stream: &Stream,
        ) -> Result<MtpPrefill<Self::TargetState>, Exception> {
            self.record("prefill", stream)?;
            *cache = 1;
            let mut first = [0.0f32; 3];
            first[self.first_token as usize] = 10.0;
            Ok(MtpPrefill {
                logits: Array::from_slice(&first, &[1, 3]),
                state: (),
                evaluated_tokens: 1,
            })
        }

        fn begin_draft(
            &mut self,
            _state: &Self::TargetState,
            _last_token: u32,
            stream: &Stream,
        ) -> Result<Self::DraftState, Exception> {
            self.record("begin_draft", stream)?;
            Ok(ScriptedDraftState {
                step: 0,
                storage: Arc::new(()),
            })
        }

        fn begin_draft_with_streams(
            &mut self,
            _state: &Self::TargetState,
            _last_token: u32,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<Self::DraftState, Exception> {
            self.record("begin_target", streams.target())?;
            self.record("begin_draft", streams.draft())?;
            Ok(ScriptedDraftState {
                step: 0,
                storage: Arc::new(()),
            })
        }

        fn draft_logits(
            &mut self,
            state: &mut Self::DraftState,
            _last_token: u32,
            stream: &Stream,
        ) -> Result<Array, Exception> {
            self.record("draft", stream)?;
            self.draft_storage
                .push(Arc::as_ptr(&state.storage) as usize);
            let logits = if state.step == 0 {
                Array::from_slice(&[0.0f32, 0.0, 10.0], &[1, 1, 3])
            } else {
                Array::from_slice(&[10.0f32, 0.0, 0.0], &[1, 1, 3])
            };
            state.step += 1;
            Ok(logits)
        }

        fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
            *cache
        }

        fn verify(
            &mut self,
            input_tokens: &Array,
            cache: &mut Self::Cache,
            stream: &Stream,
        ) -> Result<Self::Verification, Exception> {
            self.record("verify", stream)?;
            *cache += input_tokens.dim(1) as usize;
            let first = if self.reject_first {
                let mut logits = [0.0f32; 3];
                logits[self.rejection_token as usize] = 10.0;
                logits
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
            stream: &Stream,
        ) -> Result<MtpCommit<Self::TargetState>, Exception> {
            self.record("commit_target", stream)?;
            *cache = checkpoint + verified_inputs;
            Ok(MtpCommit {
                state: (),
                replayed_tokens: 0,
            })
        }

        fn commit_verification_with_streams(
            &mut self,
            _output: Self::Verification,
            _draft_state: Self::DraftState,
            cache: &mut Self::Cache,
            checkpoint: Self::CacheCheckpoint,
            verified_inputs: usize,
            streams: MtpExecutionStreams<'_>,
        ) -> Result<MtpCommit<Self::TargetState>, Exception> {
            self.record("commit_target", streams.target())?;
            self.record("commit_draft", streams.draft())?;
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
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: false,
                routes: Vec::new(),
                draft_storage: Vec::new(),
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
    fn split_stream_engine_routes_draft_and_target_work() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        };
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: false,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache = 0;

        let (tokens, _) = generate_with_streams(
            &mut backend,
            &mut cache,
            input,
            &config,
            None,
            &mut DefaultSampler,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 1]);
        for (operation, device) in backend.routes {
            let expected = if operation == "begin_draft"
                || operation == "draft"
                || operation == "commit_draft"
            {
                DeviceType::Cpu
            } else {
                DeviceType::Gpu
            };
            assert_eq!(device, expected, "{operation}");
        }
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn split_stream_engine_preserves_stochastic_acceptance() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: 4,
            max_draft_tokens: 4,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache = 0;
        let mut sampler = MirostatV2Sampler::default();
        let key = safemlx::random::key(7).unwrap();

        let (tokens, stats) = generate_with_streams(
            &mut backend,
            &mut cache,
            input,
            &config,
            Some(key),
            &mut sampler,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
        )
        .unwrap();

        assert_eq!(tokens, vec![1, 2, 0, 0]);
        assert_eq!(stats.accepted_tokens, 2);
        assert_eq!(sampler.generated_tokens(), tokens);
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
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                routes: Vec::new(),
                draft_storage: Vec::new(),
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
                first_token: 1,
                rejection_token: 1,
                reject_first: true,
                accept_second: false,
                routes: Vec::new(),
                draft_storage: Vec::new(),
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
    #[ignore = "requires an MLX Metal device"]
    fn full_acceptance_promotes_shared_optimistic_branch() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();
        let stats = &output.requests[0].stats;

        assert_eq!(stats.optimistic_draft_blocks, 1);
        assert_eq!(stats.reused_optimistic_blocks, 1);
        assert_eq!(stats.reused_optimistic_tokens, 2);
        assert_eq!(output.scheduler.peak_optimistic_branches, 1);
        assert_eq!(backend.draft_storage[0], backend.draft_storage[2]);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn promoted_round_leaves_last_emitted_token_out_of_target_cache() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        let id = scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();

        scheduler.step().unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        assert_eq!(
            scheduler.phase(id),
            Some(MtpRequestPhase::ReadyToSubmitVerification)
        );
        scheduler.cancel(id).unwrap();
        let output = scheduler.finish().unwrap();

        assert_eq!(output.requests[0].token_ids, vec![1, 2, 0]);
        // Prefill retained one token. The fully accepted verification evaluated
        // `[first, proposal_1, proposal_2]`, but proposal_2 is the last emitted
        // token and must remain outside the cache for the next round.
        assert_eq!(cache, 3);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn rejection_discards_branch_sampler_prng_history_and_cache_state() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: true,
            accept_second: false,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        let id = scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                CountingSampler::default(),
                |_| Ok(()),
            )
            .unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();
        assert_eq!(
            scheduler.phase(id),
            Some(MtpRequestPhase::OptimisticDraftReady)
        );
        scheduler.step().unwrap();
        scheduler.cancel(id).unwrap();
        let output = scheduler.finish().unwrap();
        let request = &output.requests[0];

        assert_eq!(request.token_ids, vec![1, 1]);
        assert_eq!(request.sampler.process_calls, 2);
        assert_eq!(request.stats.discarded_optimistic_blocks, 1);
        assert_eq!(request.stats.discarded_optimistic_tokens, 2);
        assert_eq!(cache, 2);
        assert_eq!(backend.draft_storage[0], backend.draft_storage[2]);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn rejection_matches_execution_with_lookahead_disabled() {
        fn run(options: MtpSchedulerOptions) -> (Vec<u32>, usize, MtpStats) {
            let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt = Array::from_slice(&[7u32], &[1, 1]);
            let parts = [InputPart::text_token_ids(&prompt)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: true,
                accept_second: false,
                routes: Vec::new(),
                draft_storage: Vec::new(),
            };
            let mut cache = 0;
            let mut scheduler = MtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()),
                options,
            )
            .unwrap();
            scheduler
                .submit(
                    &mut cache,
                    ModelInput::new(&parts),
                    MtpConfig {
                        max_tokens: 5,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    DefaultSampler,
                    |_| Ok(()),
                )
                .unwrap();
            scheduler.run().unwrap();
            let mut requests = scheduler.finish().unwrap().requests;
            let output = requests.pop().unwrap();
            (output.token_ids, cache, output.stats)
        }

        let without = run(MtpSchedulerOptions {
            max_in_flight_verifications: 1,
            max_optimistic_branches: 0,
            lookahead_blocks: 0,
        });
        let with = run(MtpSchedulerOptions::default());
        assert_eq!(with.0, without.0);
        assert_eq!(with.1, without.1);
        assert_eq!(with.2.accept_lens, without.2.accept_lens);
        assert!(with.2.discarded_optimistic_tokens > 0);
        assert_eq!(without.2.optimistic_draft_tokens, 0);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn target_eos_discards_in_flight_lookahead() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 0,
            reject_first: true,
            accept_second: false,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                MtpConfig {
                    max_tokens: 5,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: vec![0],
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();
        assert_eq!(request.token_ids, vec![1, 0]);
        assert_eq!(request.stats.optimistic_draft_tokens, 1);
        assert_eq!(request.stats.discarded_optimistic_tokens, 1);
        assert_eq!(request.stats.reused_optimistic_tokens, 0);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn independent_requests_progress_fairly_and_preserve_output_order() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
        let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
        let parts_a = [InputPart::text_token_ids(&prompt_a)];
        let parts_b = [InputPart::text_token_ids(&prompt_b)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache_a = 0;
        let mut cache_b = 0;
        let mut scheduler = MtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
            MtpSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache_a,
                ModelInput::new(&parts_a),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: vec![0],
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler
            .submit(
                &mut cache_b,
                ModelInput::new(&parts_b),
                MtpConfig {
                    max_tokens: 6,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: vec![2],
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();

        assert_eq!(output.requests[0].token_ids, vec![1, 2, 0]);
        assert_eq!(output.requests[1].token_ids, vec![1, 2]);
        assert!(output.scheduler.cross_request_draft_opportunities > 0);
        let verify = backend
            .routes
            .iter()
            .position(|(operation, _)| *operation == "verify")
            .unwrap();
        let cross_draft = backend.routes[verify + 1..]
            .iter()
            .position(|(operation, _)| *operation == "draft")
            .map(|offset| verify + 1 + offset)
            .unwrap();
        let resolve = backend
            .routes
            .iter()
            .position(|(operation, _)| *operation == "commit_target")
            .unwrap();
        assert!(verify < cross_draft && cross_draft < resolve);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn scheduler_limits_bound_retained_transactions_and_branches() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
        let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
        let parts_a = [InputPart::text_token_ids(&prompt_a)];
        let parts_b = [InputPart::text_token_ids(&prompt_b)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            routes: Vec::new(),
            draft_storage: Vec::new(),
        };
        let mut cache_a = 0;
        let mut cache_b = 0;
        let mut scheduler = MtpScheduler::new(
            &mut backend,
            MtpExecutionStreams::new(target.stream(), draft.stream()),
            MtpSchedulerOptions {
                max_in_flight_verifications: 2,
                max_optimistic_branches: 1,
                lookahead_blocks: 1,
            },
        )
        .unwrap();
        for (cache, parts) in [(&mut cache_a, &parts_a), (&mut cache_b, &parts_b)] {
            scheduler
                .submit(
                    cache,
                    ModelInput::new(parts),
                    MtpConfig {
                        max_tokens: 5,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    DefaultSampler,
                    |_| Ok(()),
                )
                .unwrap();
        }
        scheduler.run().unwrap();
        let stats = scheduler.finish().unwrap().scheduler;
        assert!(stats.peak_in_flight_verifications <= 2);
        assert!(stats.peak_optimistic_branches <= 1);
        assert_eq!(stats.peak_optimistic_branches, 1);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn stochastic_request_is_reproducible_across_scheduler_interleavings() {
        fn run(with_peer: bool) -> (Vec<u32>, Vec<usize>) {
            let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
            let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
            let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
            let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
            let parts_a = [InputPart::text_token_ids(&prompt_a)];
            let parts_b = [InputPart::text_token_ids(&prompt_b)];
            let mut backend = ScriptedBackend {
                first_token: 1,
                rejection_token: 1,
                reject_first: false,
                accept_second: true,
                routes: Vec::new(),
                draft_storage: Vec::new(),
            };
            let mut cache_a = 0;
            let mut cache_b = 0;
            let mut scheduler = MtpScheduler::new(
                &mut backend,
                MtpExecutionStreams::new(target.stream(), draft.stream()),
                MtpSchedulerOptions::default(),
            )
            .unwrap();
            let config = MtpConfig {
                max_tokens: 5,
                max_draft_tokens: 2,
                temperature: 1.0,
                eos_token_ids: Vec::new(),
            };
            scheduler
                .submit(
                    &mut cache_a,
                    ModelInput::new(&parts_a),
                    config.clone(),
                    Some(safemlx::random::key(7).unwrap()),
                    MirostatV2Sampler::default(),
                    |_| Ok(()),
                )
                .unwrap();
            if with_peer {
                scheduler
                    .submit(
                        &mut cache_b,
                        ModelInput::new(&parts_b),
                        config,
                        Some(safemlx::random::key(99).unwrap()),
                        MirostatV2Sampler::default(),
                        |_| Ok(()),
                    )
                    .unwrap();
            }
            scheduler.run().unwrap();
            let output = scheduler.finish().unwrap();
            (
                output.requests[0].token_ids.clone(),
                output.requests[0].stats.accept_lens.clone(),
            )
        }

        assert_eq!(run(false), run(true));
    }

    #[test]
    fn empty_stats_have_zero_acceptance_rate() {
        assert_eq!(MtpStats::default().accept_rate(), 0.0);
    }
}
