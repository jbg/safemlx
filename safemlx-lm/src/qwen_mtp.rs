//! Qwen3-Next and Qwen3.5/3.6 adapters for embedded MTP layers.

use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::{
    models::{
        input::{self, ModelInput},
        qwen3_5_moe::{Cache, LayerCache, Model, QwenMtpStepOutput},
    },
    mtp::{self, MtpBackend, MtpCommit, MtpConfig, MtpPrefill},
    sampler::SpeculativeSampler,
};

pub(crate) trait QwenMtpTarget {
    fn prefill_mtp_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception>;
    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception>;
    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception>;
    fn mtp_layer_count(&self) -> usize;
}

impl QwenMtpTarget for Model {
    fn prefill_mtp_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_mtp_head(hidden, tokens, cache, stream)
    }

    fn mtp_layer_count(&self) -> usize {
        self.mtp_len()
    }
}

impl QwenMtpTarget for crate::qwen_hybrid::QwenHybridLayerwiseModel {
    fn prefill_mtp_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_mtp_head(hidden, tokens, cache, stream)
    }

    fn mtp_layer_count(&self) -> usize {
        self.mtp_len()
    }
}

pub(crate) struct QwenTargetState {
    hidden: Array,
    mtp_cache: Vec<LayerCache>,
}

pub(crate) struct QwenDraftState {
    hidden: Array,
    mtp_cache: Vec<LayerCache>,
}

pub(crate) struct QwenVerification {
    output: QwenMtpStepOutput,
    inputs: Array,
}

pub(crate) struct QwenMtpBackend<'a, T> {
    target: &'a mut T,
}

impl<'a, T: QwenMtpTarget> QwenMtpBackend<'a, T> {
    pub(crate) fn new(target: &'a mut T) -> Self {
        Self { target }
    }

    fn state_at(
        output: &QwenMtpStepOutput,
        row: i32,
        mtp_cache: &[LayerCache],
        stream: &Stream,
    ) -> Result<QwenTargetState, Exception> {
        Ok(QwenTargetState {
            hidden: output
                .hidden
                .try_index_device((.., row..row + 1, ..), stream)?,
            mtp_cache: mtp_cache.to_vec(),
        })
    }

    fn prefill_draft_cache(
        &mut self,
        output: &QwenMtpStepOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next_tokens = tokens.try_index_device((.., 1..), stream)?;
        let _ = self.target.forward_mtp_drafter(
            &hidden,
            &next_tokens,
            &mut cache.mtp_layers,
            stream,
        )?;
        Ok(())
    }
}

impl<T: QwenMtpTarget> MtpBackend for QwenMtpBackend<'_, T> {
    type Cache = Cache;
    type TargetState = QwenTargetState;
    type DraftState = QwenDraftState;
    type CacheCheckpoint = Cache;
    type Verification = QwenVerification;

    fn max_draft_tokens(&self) -> usize {
        usize::from(self.target.mtp_layer_count() > 0)
    }

    fn prefill(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<MtpPrefill<Self::TargetState>, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        let output = self.target.prefill_mtp_target(input, cache, stream)?;
        let sequence = output.logits.dim(-2);
        if sequence == 0 {
            return Err(Exception::custom(
                "Qwen MTP input must contain at least one token",
            ));
        }
        self.prefill_draft_cache(&output, &tokens, cache, stream)?;
        let logits = output
            .logits
            .try_index_device((.., sequence - 1, ..), stream)?;
        let state = Self::state_at(&output, sequence - 1, &cache.mtp_layers, stream)?;
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
        Ok(QwenDraftState {
            hidden: state.hidden.clone(),
            mtp_cache: state.mtp_cache.clone(),
        })
    }

    fn draft_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
        self.target
            .forward_mtp_drafter(&state.hidden, &token, &mut state.mtp_cache, stream)
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
        Ok(QwenVerification {
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
        mut draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        stream: &Stream,
    ) -> Result<MtpCommit<Self::TargetState>, Exception> {
        let input_len = output.inputs.dim(1) as usize;
        if verified_inputs == 0 || verified_inputs > input_len {
            return Err(Exception::custom(format!(
                "cannot commit {verified_inputs} verified Qwen inputs from a block of {input_len}"
            )));
        }

        if verified_inputs > 1 {
            let accepted = verified_inputs as i32 - 1;
            let hidden = output
                .output
                .hidden
                .try_index_device((.., ..accepted, ..), stream)?;
            let tokens = output
                .inputs
                .try_index_device((.., 1..verified_inputs as i32), stream)?;
            let _ = self.target.forward_mtp_drafter(
                &hidden,
                &tokens,
                &mut draft_state.mtp_cache,
                stream,
            )?;
        }

        let (committed, replayed_tokens) = if verified_inputs == input_len {
            (output.output, 0)
        } else {
            *cache = checkpoint;
            let retained = output
                .inputs
                .try_index_device((.., ..verified_inputs as i32), stream)?;
            (
                self.target.verify_mtp_target(&retained, cache, stream)?,
                verified_inputs,
            )
        };
        cache.mtp_layers.clone_from(&draft_state.mtp_cache);
        let state = Self::state_at(
            &committed,
            verified_inputs as i32 - 1,
            &cache.mtp_layers,
            stream,
        )?;
        Ok(MtpCommit {
            state,
            replayed_tokens,
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // Used directly by device-gated backend parity tests.
pub(crate) fn generate<T: QwenMtpTarget, S: SpeculativeSampler>(
    target: &mut T,
    cache: &mut Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &S,
    stream: &Stream,
) -> Result<(Vec<u32>, mtp::MtpStats), Exception> {
    generate_with_callback(
        target,
        cache,
        input,
        config,
        prng_key,
        sampler,
        stream,
        |_| Ok(()),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_callback<T, S, F>(
    target: &mut T,
    cache: &mut Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &S,
    stream: &Stream,
    on_token: F,
) -> Result<(Vec<u32>, mtp::MtpStats), Exception>
where
    T: QwenMtpTarget,
    S: SpeculativeSampler,
    F: FnMut(u32) -> Result<(), Exception>,
{
    let mut backend = QwenMtpBackend::new(target);
    mtp::generate_with_callback(
        &mut backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        stream,
        on_token,
    )
}
