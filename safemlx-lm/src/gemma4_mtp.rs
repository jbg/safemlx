//! Gemma 4 adapter for the architecture-independent MTP engine.

use std::collections::HashMap;

use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::{
    gemma4::Gemma4LayerwiseModel,
    models::{
        gemma4::{Cache, Gemma4StepOutput, LayerType, Model as Gemma4Model},
        gemma4_assistant::Gemma4AssistantDraftModel,
        input::ModelInput as RuntimeInput,
    },
    mtp::{self, MtpBackend, MtpCommit, MtpConfig, MtpPrefill},
    sampler::SpeculativeSampler,
};

pub(crate) struct Gemma4TargetState {
    hidden: Array,
    shared_kv: HashMap<LayerType, (Array, Array)>,
    cache_len: usize,
}

pub(crate) struct Gemma4DraftState {
    hidden: Array,
}

pub(crate) struct Gemma4Verification {
    output: Gemma4StepOutput,
    inputs: Array,
}

pub(crate) trait Gemma4MtpTarget {
    fn prefill_mtp_target(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception>;
    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception>;
    fn mtp_embedding(&mut self, token: u32, stream: &Stream) -> Result<Array, Exception>;
}

impl Gemma4MtpTarget for Gemma4Model {
    fn prefill_mtp_target(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn mtp_embedding(&mut self, token: u32, stream: &Stream) -> Result<Array, Exception> {
        self.mtp_token_embedding(token, stream)
    }
}

impl Gemma4MtpTarget for Gemma4LayerwiseModel {
    fn prefill_mtp_target(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn mtp_embedding(&mut self, token: u32, stream: &Stream) -> Result<Array, Exception> {
        self.mtp_token_embedding(token, stream)
    }
}

/// Gemma 4 target plus an external Gemma assistant.
pub(crate) struct Gemma4MtpBackend<'a, T> {
    target: &'a mut T,
    assistant: &'a mut Gemma4AssistantDraftModel,
}

impl<'a, T> Gemma4MtpBackend<'a, T> {
    pub(crate) fn new(target: &'a mut T, assistant: &'a mut Gemma4AssistantDraftModel) -> Self {
        Self { target, assistant }
    }

    fn state_at(
        output: &Gemma4StepOutput,
        row: i32,
        cache_len: usize,
        stream: &Stream,
    ) -> Result<Gemma4TargetState, Exception> {
        let hidden = output
            .hidden
            .try_index_device((.., row..row + 1, ..), stream)?;
        let retained = i32::try_from(cache_len)
            .map_err(|_| Exception::custom("Gemma 4 MTP state length exceeds i32"))?;
        let mut shared_kv = HashMap::with_capacity(output.shared_kv_states.len());
        for (kind, (keys, values)) in &output.shared_kv_states {
            let key_len = keys.dim(-2).min(retained);
            let value_len = values.dim(-2).min(retained);
            shared_kv.insert(
                *kind,
                (
                    keys.try_index_device((.., .., ..key_len, ..), stream)?,
                    values.try_index_device((.., .., ..value_len, ..), stream)?,
                ),
            );
        }
        Ok(Gemma4TargetState {
            hidden,
            shared_kv,
            cache_len,
        })
    }
}

impl<T: Gemma4MtpTarget> MtpBackend for Gemma4MtpBackend<'_, T> {
    type Cache = Cache;
    type TargetState = Gemma4TargetState;
    type DraftState = Gemma4DraftState;
    type CacheCheckpoint = Cache;
    type Verification = Gemma4Verification;

    fn max_draft_tokens(&self) -> usize {
        self.assistant.block_size().saturating_sub(1)
    }

    fn prefill(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<MtpPrefill<Self::TargetState>, Exception> {
        let output = self.target.prefill_mtp_target(input, cache, stream)?;
        let sequence = output.logits.dim(-2);
        if sequence == 0 {
            return Err(Exception::custom(
                "MTP input must contain at least one token",
            ));
        }
        let logits = output
            .logits
            .try_index_device((.., sequence - 1, ..), stream)?;
        let state = Self::state_at(&output, sequence - 1, cache.mtp_len(), stream)?;
        Ok(MtpPrefill {
            logits,
            state,
            evaluated_tokens: sequence as usize,
        })
    }

    fn begin_draft(
        &mut self,
        state: &Self::TargetState,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Self::DraftState, Exception> {
        let offset = i32::try_from(state.cache_len)
            .map_err(|_| Exception::custom("Gemma 4 MTP cache offset exceeds i32"))?;
        let hidden = self
            .assistant
            .begin_round(state.shared_kv.clone(), offset, &state.hidden);
        Ok(Gemma4DraftState { hidden })
    }

    fn draft_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embedding = self.target.mtp_embedding(last_token, stream)?;
        self.assistant
            .draft_step(&embedding, &mut state.hidden, stream)
    }

    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
        cache.clone()
    }

    fn verify(
        &mut self,
        input_tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<Self::Verification, Exception> {
        Ok(Gemma4Verification {
            output: self.target.verify_mtp_target(input_tokens, cache, stream)?,
            inputs: input_tokens.clone(),
        })
    }

    fn verification_logits(output: &Self::Verification) -> &Array {
        &output.output.logits
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        _draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        stream: &Stream,
    ) -> Result<MtpCommit<Self::TargetState>, Exception> {
        let input_len = output.inputs.dim(1) as usize;
        if verified_inputs > input_len {
            return Err(Exception::custom(format!(
                "cannot commit {verified_inputs} verified inputs from a block of {input_len}"
            )));
        }
        if verified_inputs == input_len {
            let state = Self::state_at(
                &output.output,
                verified_inputs.saturating_sub(1) as i32,
                cache.mtp_len(),
                stream,
            )?;
            return Ok(MtpCommit {
                state,
                replayed_tokens: 0,
            });
        }

        *cache = checkpoint;
        let retained_inputs = output
            .inputs
            .try_index_device((.., ..verified_inputs as i32), stream)?;
        let replayed = self
            .target
            .verify_mtp_target(&retained_inputs, cache, stream)?;
        let state = Self::state_at(
            &replayed,
            verified_inputs.saturating_sub(1) as i32,
            cache.mtp_len(),
            stream,
        )?;
        Ok(MtpCommit {
            state,
            replayed_tokens: verified_inputs,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate<T, S>(
    target: &mut T,
    assistant: &mut Gemma4AssistantDraftModel,
    cache: &mut Cache,
    input: RuntimeInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &S,
    stream: &Stream,
) -> Result<(Vec<u32>, mtp::MtpStats), Exception>
where
    T: Gemma4MtpTarget,
    S: SpeculativeSampler,
{
    let mut backend = Gemma4MtpBackend::new(target, assistant);
    mtp::generate(
        &mut backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        stream,
    )
}
