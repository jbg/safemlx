//! Codec-free realtime speech-to-speech token APIs.
//!
//! Realtime speech models in this crate operate on discrete codec tokens rather
//! than PCM. Callers are expected to encode live audio into model-native
//! codebook frames before calling these APIs, and decode emitted codebook frames
//! with a codec implementation outside `safemlx-lm`.

use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, stack_axis},
    random::RandomState,
    Array, Stream,
};
use serde::Deserialize;
use std::path::Path;

use crate::{
    error::Error,
    models::{moshi, personaplex},
    sampler::{DefaultSampler, Sampler},
};

/// Static token-stream metadata needed to pair a realtime model with a codec.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RealtimeSpeechConfig<'a> {
    /// Total number of audio codebooks consumed by the temporal model.
    pub total_audio_codebooks: i32,
    /// Number of live input-side codebooks expected per realtime step.
    pub input_audio_codebooks: i32,
    /// Number of generated-side codebooks emitted per realtime step.
    pub generated_audio_codebooks: i32,
    /// Number of depth-transformer codebooks sampled or teacher-forced per step.
    pub depth_audio_codebooks: i32,
    /// Text token used before any sampled text is available.
    pub text_padding_token: i32,
    /// Audio token used while delayed streams warm up.
    pub audio_padding_token: i32,
    /// Per-audio-codebook delays, excluding the leading text delay.
    pub audio_delays: &'a [i32],
}

impl RealtimeSpeechConfig<'_> {
    /// Largest audio delay in frames.
    pub fn max_audio_delay(&self) -> i32 {
        self.audio_delays.iter().copied().max().unwrap_or(0)
    }
}

/// One encoded input-side audio frame for a realtime model step.
#[derive(Debug, Clone, Copy)]
pub struct RealtimeStepInput<'a> {
    /// Encoded input-side audio tokens shaped `[batch, input_audio_codebooks]`.
    pub input_audio_tokens: &'a Array,
}

impl<'a> RealtimeStepInput<'a> {
    /// Creates a realtime step input from encoded audio-codebook tokens.
    pub fn encoded_audio(input_audio_tokens: &'a Array) -> Self {
        Self { input_audio_tokens }
    }
}

/// Caller-provided sampling controls for one realtime model step.
pub struct RealtimeSampling<'a, TS, AS> {
    /// Sampler used for text logits.
    pub text_sampler: &'a mut TS,
    /// One sampler per depth codebook sampled by the model.
    pub audio_samplers: &'a mut [AS],
    /// Text sampling temperature.
    pub text_temperature: f32,
    /// Audio sampling temperature.
    pub audio_temperature: f32,
    /// Optional PRNG state for stochastic samplers.
    pub prng_state: Option<&'a mut RandomState>,
}

impl<'a, TS, AS> RealtimeSampling<'a, TS, AS> {
    /// Creates realtime sampling controls.
    pub fn new(
        text_sampler: &'a mut TS,
        audio_samplers: &'a mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&'a mut RandomState>,
    ) -> Self {
        Self {
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            prng_state,
        }
    }
}

/// Output from one encoded-audio realtime generation step.
pub struct RealtimeStepOutput {
    /// Text token sampled at this model step, shaped `[batch, 1]`.
    pub text_token: Array,
    /// Newly sampled generated-codebook tokens before delay alignment.
    pub sampled_audio_tokens: Array,
    /// Delay-aligned codec frame ready for decoding, shaped `[batch, generated_audio_codebooks]`.
    ///
    /// This is `None` while delayed generated streams are warming up.
    pub output_audio_tokens: Option<Array>,
}

/// Text tokens and delay-aligned codec tokens from offline generation.
pub struct EncodedAudioOutput {
    /// Sampled text tokens, shaped `[batch, input_frames]`.
    pub text_tokens: Array,
    /// Generated codec tokens, shaped `[batch, generated_audio_codebooks, output_frames]`.
    ///
    /// The output may have fewer frames than the input because delayed streams
    /// need future encoded input frames before a coherent output frame exists.
    pub audio_tokens: Array,
}

/// Common codec-token API for realtime speech-to-speech models.
pub trait RealtimeSpeechModel {
    /// Stateful cache/session type used across realtime steps.
    type State;

    /// Returns the codec-token stream configuration for this model.
    fn realtime_config(&self) -> RealtimeSpeechConfig<'_>;

    /// Creates a fresh realtime state.
    fn new_realtime_state(&self) -> Self::State;

    /// Consumes one encoded input-side frame and advances realtime generation.
    fn step_realtime<TS, AS>(
        &mut self,
        state: &mut Self::State,
        input: RealtimeStepInput<'_>,
        sampling: RealtimeSampling<'_, TS, AS>,
        stream: &Stream,
    ) -> Result<RealtimeStepOutput, Exception>
    where
        TS: Sampler,
        AS: Sampler;
}

/// Supported realtime speech-to-speech model-family dispatch target.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RealtimeModelKind {
    /// Moshi-family realtime token model with a native Moshi/MLX checkpoint layout.
    Moshi,
    /// NVIDIA PersonaPlex realtime token model with its released PyTorch safetensors layout.
    PersonaPlex,
}

impl RealtimeModelKind {
    /// Returns the model type string used for user-facing dispatch messages.
    pub fn model_type(self) -> &'static str {
        match self {
            Self::Moshi => "moshi",
            Self::PersonaPlex => "personaplex",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RealtimeModelMetadata {
    #[serde(default)]
    model_type: Option<String>,
}

fn realtime_model_kind(model_dir: impl AsRef<Path>) -> Result<RealtimeModelKind, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    if !config_path.exists() {
        return Ok(RealtimeModelKind::Moshi);
    }

    let metadata: RealtimeModelMetadata =
        serde_json::from_reader(std::fs::File::open(config_path)?)?;
    match metadata.model_type.as_deref() {
        None | Some("moshi") => Ok(RealtimeModelKind::Moshi),
        Some("personaplex") => Ok(RealtimeModelKind::PersonaPlex),
        Some(other) => Err(Error::UnsupportedArchitecture(format!(
            "{other} is not a realtime speech-to-speech token model"
        ))),
    }
}

/// Loads a supported realtime speech-to-speech token model from a model directory.
///
/// This is the high-level realtime counterpart to [`crate::models::LoadedModel`].
/// It does not load a text tokenizer or audio codec: callers bring tokenization,
/// codec encode/decode, transport, and device I/O.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedRealtimeModel, Error> {
    match realtime_model_kind(&model_dir)? {
        RealtimeModelKind::Moshi => Ok(LoadedRealtimeModel::Moshi(moshi::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        RealtimeModelKind::PersonaPlex => Ok(LoadedRealtimeModel::PersonaPlex(
            personaplex::load_model(model_dir, stream, weights_stream)?,
        )),
    }
}

/// Loaded realtime speech-to-speech token model.
///
/// The enum gives consumers a single public entry point for codec-token models
/// while still allowing family-specific helpers, such as PersonaPlex prompt
/// prefill, to live in their model modules.
pub enum LoadedRealtimeModel {
    /// Moshi-family model.
    Moshi(moshi::Model),
    /// PersonaPlex model.
    PersonaPlex(personaplex::Model),
}

impl LoadedRealtimeModel {
    /// Returns the loaded realtime model family.
    pub fn kind(&self) -> RealtimeModelKind {
        match self {
            Self::Moshi(_) => RealtimeModelKind::Moshi,
            Self::PersonaPlex(_) => RealtimeModelKind::PersonaPlex,
        }
    }

    /// Returns the loaded realtime model family as a model type string.
    pub fn model_type(&self) -> &'static str {
        self.kind().model_type()
    }

    /// Returns the underlying Moshi-family token model.
    pub fn as_moshi_model(&self) -> &moshi::Model {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model,
        }
    }

    /// Returns the underlying Moshi-family token model mutably.
    pub fn as_moshi_model_mut(&mut self) -> &mut moshi::Model {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model,
        }
    }

    /// Consumes this wrapper and returns the underlying Moshi-family token model.
    pub fn into_moshi_model(self) -> moshi::Model {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model,
        }
    }
}

impl RealtimeSpeechModel for LoadedRealtimeModel {
    type State = RealtimeState;

    fn realtime_config(&self) -> RealtimeSpeechConfig<'_> {
        self.as_moshi_model().realtime_config()
    }

    fn new_realtime_state(&self) -> Self::State {
        match self {
            Self::Moshi(model) => RealtimeState::Moshi(model.new_realtime_state()),
            Self::PersonaPlex(model) => RealtimeState::PersonaPlex(model.new_realtime_state()),
        }
    }

    fn step_realtime<TS, AS>(
        &mut self,
        state: &mut Self::State,
        input: RealtimeStepInput<'_>,
        sampling: RealtimeSampling<'_, TS, AS>,
        stream: &Stream,
    ) -> Result<RealtimeStepOutput, Exception>
    where
        TS: Sampler,
        AS: Sampler,
    {
        match (self, state) {
            (Self::Moshi(model), RealtimeState::Moshi(state)) => {
                model.step_realtime(state, input, sampling, stream)
            }
            (Self::PersonaPlex(model), RealtimeState::PersonaPlex(state)) => {
                model.step_realtime(state, input, sampling, stream)
            }
            _ => Err(Exception::custom(
                "realtime state type does not match loaded realtime model",
            )),
        }
    }
}

/// Stateful realtime generation session matching a [`LoadedRealtimeModel`].
pub enum RealtimeState {
    /// Moshi-family generation state.
    Moshi(moshi::GenerationState),
    /// PersonaPlex generation state.
    PersonaPlex(personaplex::GenerationState),
}

impl RealtimeState {
    /// Returns the model family this state belongs to.
    pub fn kind(&self) -> RealtimeModelKind {
        match self {
            Self::Moshi(_) => RealtimeModelKind::Moshi,
            Self::PersonaPlex(_) => RealtimeModelKind::PersonaPlex,
        }
    }
}

/// Greedily generates delay-aligned codec tokens from an encoded input sequence.
///
/// Input and output use `[batch, codebooks, frames]` layout. This helper does
/// not append encoded silence, so delayed tail frames are not flushed after the
/// supplied input ends.
pub fn generate_encoded_greedy<M>(
    model: &mut M,
    input_audio_tokens: &Array,
    stream: &Stream,
) -> Result<EncodedAudioOutput, Exception>
where
    M: RealtimeSpeechModel,
{
    let config = model.realtime_config();
    let input_audio_codebooks = config.input_audio_codebooks;
    let generated_audio_codebooks = config.generated_audio_codebooks;
    let depth_audio_codebooks = config.depth_audio_codebooks;
    if input_audio_tokens.shape().len() != 3 || input_audio_tokens.dim(1) != input_audio_codebooks {
        return Err(Exception::custom(format!(
            "encoded input sequence must have shape [batch, {}, frames], got {:?}",
            input_audio_codebooks,
            input_audio_tokens.shape()
        )));
    }

    let batch = input_audio_tokens.dim(0);
    let mut state = model.new_realtime_state();
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..depth_audio_codebooks)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    let mut text = Vec::with_capacity(input_audio_tokens.dim(2) as usize);
    let mut audio = Vec::new();
    for frame in 0..input_audio_tokens.dim(2) {
        let input = input_audio_tokens.try_index_device((.., .., frame), stream)?;
        let output = model.step_realtime(
            &mut state,
            RealtimeStepInput::encoded_audio(&input),
            RealtimeSampling::new(&mut text_sampler, &mut audio_samplers, 0.0, 0.0, None),
            stream,
        )?;
        text.push(output.text_token.squeeze_axes(&[-1], stream)?);
        if let Some(tokens) = output.output_audio_tokens {
            audio.push(tokens);
        }
    }
    let text_tokens = if text.is_empty() {
        Array::zeros::<i32>(&[batch, 0], stream)?
    } else {
        stack_axis(&text, 1, stream)?
    };
    let audio_tokens = if audio.is_empty() {
        Array::zeros::<i32>(&[batch, generated_audio_codebooks, 0], stream)?
    } else {
        stack_axis(&audio, 2, stream)?
    };
    Ok(EncodedAudioOutput {
        text_tokens,
        audio_tokens,
    })
}
