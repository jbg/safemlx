//! PersonaPlex realtime speech-to-speech token model support.
//!
//! PersonaPlex is a Moshi-family full-duplex model with hybrid system prompts:
//! a voice segment forced on the generated audio stream and a text segment
//! forced on the generated text stream. This module intentionally remains
//! codec-free; callers provide Mimi/codec tokens and decode emitted tokens with
//! a codec outside `safemlx-lm`.

use std::path::Path;

use safemlx::{
    error::Exception, module::ModuleParametersExt, ops::broadcast_to, ops::indexing::TryIndexOp,
    random::RandomState, Array, Stream,
};
use serde::Deserialize;

use crate::{
    error::Error,
    models::moshi,
    quantization::WeightQuantization,
    sampler::{DefaultSampler, Sampler},
};

/// Hugging Face repository for the released PersonaPlex checkpoint.
pub const DEFAULT_HF_REPO: &str = "nvidia/personaplex-7b-v1";
/// Released PersonaPlex language-model checkpoint filename.
pub const MODEL_SAFETENSORS: &str = "model.safetensors";
/// Released Mimi codec checkpoint filename.
pub const MIMI_SAFETENSORS: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
/// Released text tokenizer filename used by NVIDIA's runtime.
pub const TEXT_TOKENIZER: &str = "tokenizer_spm_32k_3.model";

/// Number of Mimi codebooks per side in PersonaPlex's dual-stream layout.
pub const AUDIO_TOKENS_PER_STREAM: i32 = 8;
/// PersonaPlex uses the tokenizer's existing pad id during prompt forcing.
pub const TEXT_PADDING_TOKEN: i32 = 3;
/// PersonaPlex audio tokens used for an agent-side silence frame.
pub const SILENCE_TOKENS: [i32; 8] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008];
/// PersonaPlex audio tokens used as a user-side 440 Hz conditioning frame.
pub const SINE_TOKENS: [i32; 8] = [430, 1268, 381, 1611, 1095, 1495, 56, 472];

/// PersonaPlex `config.json` metadata from the released HF repository.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelMetadata {
    /// Expected to be `personaplex`.
    pub model_type: String,
    /// Released model version, for example `7b-v1`.
    #[serde(default)]
    pub version: Option<String>,
    /// Optional MLX affine checkpoint quantization settings.
    #[serde(default)]
    pub quantization: Option<WeightQuantization>,
}

/// PersonaPlex uses the Moshi-family token model implementation.
pub type Model = moshi::Model;
/// Stateful realtime generation session.
pub type GenerationState = moshi::GenerationState;
/// Output from one realtime generation step.
pub type GenerationStepOutput = moshi::GenerationStepOutput;
/// Offline encoded-audio output.
pub type EncodedAudioOutput = moshi::EncodedAudioOutput;

/// PersonaPlex model operations needed by forced system-prompt prefill.
pub trait PromptModel {
    /// Number of within-frame depth codebooks.
    fn depth_codebooks(&self) -> i32;

    /// Advances one prompt frame with forced agent audio and text.
    #[allow(clippy::too_many_arguments)]
    fn generate_prompt_step<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<GenerationStepOutput, Exception>;
}

impl PromptModel for Model {
    fn depth_codebooks(&self) -> i32 {
        self.args.dep_q
    }

    fn generate_prompt_step<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<GenerationStepOutput, Exception> {
        self.generate_step_forced(
            state,
            input_audio_tokens,
            forced_generated_audio_tokens,
            forced_text_token,
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            prng_state,
            stream,
        )
    }
}

impl PromptModel for crate::moshi::MoshiLayerwiseModel {
    fn depth_codebooks(&self) -> i32 {
        self.args().dep_q
    }

    fn generate_prompt_step<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<GenerationStepOutput, Exception> {
        self.generate_step_forced(
            state,
            input_audio_tokens,
            forced_generated_audio_tokens,
            forced_text_token,
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            prng_state,
            stream,
        )
    }
}

/// Returns the published PersonaPlex 7B v1 language-model defaults.
pub fn model_args_7b_v1() -> moshi::ModelArgs {
    moshi::ModelArgs {
        model_type: Some("personaplex".to_string()),
        dim: 4_096,
        text_card: 32_000,
        existing_text_padding_id: Some(TEXT_PADDING_TOKEN),
        n_q: 16,
        dep_q: 16,
        generated_audio_codebooks: Some(AUDIO_TOKENS_PER_STREAM),
        card: 2_048,
        num_heads: 32,
        num_layers: 32,
        dim_feedforward: Some((4.125 * 4_096.0) as i32),
        causal: true,
        context: 3_000,
        max_period: 10_000.0,
        positional_embedding: "rope".to_string(),
        depformer_dim: 1_024,
        depformer_dim_feedforward: Some((4.125 * 1_024.0) as i32),
        depformer_num_heads: 16,
        depformer_num_layers: 6,
        depformer_context: Some(8),
        depformer_max_period: Some(10_000.0),
        depformer_pos_emb: "none".to_string(),
        delays: vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        moshi_name: Some(MODEL_SAFETENSORS.to_string()),
        conditioners: Default::default(),
        cross_attention: false,
        demux_second_stream: false,
        depformer_low_rank_embeddings: None,
        extra_heads_num_heads: 0,
        quantization: None,
    }
}

/// Reads and validates PersonaPlex metadata.
pub fn get_model_metadata(model_dir: impl AsRef<Path>) -> Result<ModelMetadata, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    let metadata: ModelMetadata = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    validate_metadata(&metadata)?;
    Ok(metadata)
}

/// Validates parsed PersonaPlex metadata.
pub fn validate_metadata(metadata: &ModelMetadata) -> Result<(), Error> {
    if metadata.model_type != "personaplex" {
        return Err(Error::UnsupportedModelType(metadata.model_type.clone()));
    }
    match metadata.version.as_deref() {
        None | Some("7b-v1") => Ok(()),
        Some(version) => Err(Error::UnsupportedArchitecture(format!(
            "unsupported PersonaPlex version {version}; only 7b-v1 defaults are known"
        ))),
    }
}

/// Validates a `personaplex` config value.
pub fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    let metadata: ModelMetadata = serde_json::from_value(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid PersonaPlex config: {error}"))
    })?;
    validate_metadata(&metadata)
}

/// Creates an unloaded PersonaPlex language model from the published defaults.
pub fn new(stream: &Stream) -> Result<Model, Error> {
    moshi::Model::new(model_args_7b_v1(), stream)
}

/// Loads a PersonaPlex checkpoint in this crate's native Moshi layout.
pub fn load_native_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    load_native_model_with_options(
        model_dir,
        crate::models::ModelLoadOptions::default(),
        stream,
        weights_stream,
    )
}

/// Loads a native-layout PersonaPlex checkpoint using shared model-load options.
pub fn load_native_model_with_options(
    model_dir: impl AsRef<Path>,
    options: crate::models::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let metadata = get_model_metadata(&model_dir)?;
    let quantize_on_load = if let Some(quantization) = options.quantization {
        crate::quantization::should_quantize_on_load(
            "PersonaPlex",
            metadata.quantization,
            quantization,
        )?
    } else {
        false
    };
    let mut args = model_args_7b_v1();
    args.moshi_name = Some(MODEL_SAFETENSORS.to_string());
    args.quantization = options.quantization.or(metadata.quantization);
    let mut model = moshi::Model::new(args, stream)?;
    let config = crate::weights::StrictLoadConfig::default();
    let mut report = crate::weights::StrictLoadReport::default();
    if model_dir
        .as_ref()
        .join("model.safetensors.index.json")
        .exists()
    {
        if quantize_on_load {
            crate::weights::load_safetensors_dir_quantized_strict(
                &mut model,
                &model_dir,
                weights_stream,
                stream,
                options.quantization.expect("quantization requested"),
                &config,
                &mut report,
            )?;
        } else {
            crate::weights::load_safetensors_dir_strict(
                &mut model,
                &model_dir,
                weights_stream,
                &config,
                &mut report,
            )?;
        }
    } else {
        let path = model_dir.as_ref().join(MODEL_SAFETENSORS);
        if quantize_on_load {
            crate::weights::load_safetensors_quantized_strict(
                &mut model,
                path,
                weights_stream,
                stream,
                options.quantization.expect("quantization requested"),
                &config,
                &mut report,
            )?;
        } else {
            crate::weights::load_safetensors_strict(
                &mut model,
                path,
                weights_stream,
                &config,
                &mut report,
            )?;
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads the released PersonaPlex PyTorch-layout safetensors checkpoint.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let metadata = get_model_metadata(&model_dir)?;
    let mut args = model_args_7b_v1();
    args.quantization = metadata.quantization;
    if metadata.quantization.is_some()
        || model_dir
            .as_ref()
            .join("model.safetensors.index.json")
            .exists()
    {
        let mut model = moshi::Model::new(args, stream)?;
        let files = crate::weights::safetensors_files(&model_dir)?;
        moshi::load_pytorch_safetensors_files_strict(&mut model, files, weights_stream)?;
        model.copy_to_stream(stream)?;
        Ok(model)
    } else {
        moshi::load_pytorch_safetensors_model(
            args,
            model_dir.as_ref().join(MODEL_SAFETENSORS),
            stream,
            weights_stream,
        )
    }
}

/// Loads the dense released PersonaPlex checkpoint with on-the-fly affine quantization.
pub fn load_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let metadata = get_model_metadata(&model_dir)?;
    if !crate::quantization::should_quantize_on_load(
        "PersonaPlex",
        metadata.quantization,
        quantization,
    )? {
        return load_model(model_dir, stream, weights_stream);
    }
    let mut args = model_args_7b_v1();
    args.quantization = Some(quantization);
    let mut model = moshi::Model::new(args, stream)?;
    let files = crate::weights::safetensors_files(&model_dir)?;
    moshi::load_pytorch_safetensors_files_quantized_strict(
        &mut model,
        files,
        weights_stream,
        stream,
        quantization,
    )?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// One forced system-prompt frame expressed entirely in codec/text tokens.
pub struct PromptFrame<'a> {
    /// Generated-side agent audio tokens shaped `[batch, 8]`.
    pub agent_audio_tokens: &'a Array,
    /// User-side conditioning audio tokens shaped `[batch, 8]`.
    pub user_audio_tokens: &'a Array,
    /// Agent text token shaped `[batch, 1]`.
    pub text_token: &'a Array,
}

/// Wraps text prompt content with PersonaPlex system tags if absent.
pub fn wrap_system_prompt(text: &str) -> String {
    let text = text.trim();
    if text.starts_with("<system>") && text.ends_with("<system>") {
        text.to_string()
    } else {
        format!("<system> {text} <system>")
    }
}

/// Creates a repeated silence frame shaped `[batch, 8]`.
pub fn silence_frame(batch: i32, stream: &Stream) -> Result<Array, Exception> {
    repeated_frame(&SILENCE_TOKENS, batch, stream)
}

/// Creates a repeated sine-conditioning frame shaped `[batch, 8]`.
pub fn sine_frame(batch: i32, stream: &Stream) -> Result<Array, Exception> {
    repeated_frame(&SINE_TOKENS, batch, stream)
}

/// Creates a repeated text-padding token shaped `[batch, 1]`.
pub fn text_padding_frame(batch: i32, stream: &Stream) -> Result<Array, Exception> {
    Array::full::<i32>(&[batch, 1], Array::from_int(TEXT_PADDING_TOKEN), stream)
}

fn repeated_frame(tokens: &[i32; 8], batch: i32, stream: &Stream) -> Result<Array, Exception> {
    broadcast_to(
        Array::from_slice(tokens, &[1, AUDIO_TOKENS_PER_STREAM]),
        &[batch, AUDIO_TOKENS_PER_STREAM],
        stream,
    )
}

/// Runs one forced PersonaPlex prompt frame.
#[allow(clippy::too_many_arguments)]
pub fn step_prompt_frame<M: PromptModel, TS: Sampler, AS: Sampler>(
    model: &mut M,
    state: &mut GenerationState,
    frame: PromptFrame<'_>,
    text_sampler: &mut TS,
    audio_samplers: &mut [AS],
    text_temperature: f32,
    audio_temperature: f32,
    prng_state: Option<&mut RandomState>,
    stream: &Stream,
) -> Result<GenerationStepOutput, Exception> {
    model.generate_prompt_step(
        state,
        frame.user_audio_tokens,
        Some(frame.agent_audio_tokens),
        Some(frame.text_token),
        text_sampler,
        audio_samplers,
        text_temperature,
        audio_temperature,
        prng_state,
        stream,
    )
}

/// Greedily runs one forced PersonaPlex prompt frame.
pub fn step_prompt_frame_greedy<M: PromptModel>(
    model: &mut M,
    state: &mut GenerationState,
    frame: PromptFrame<'_>,
    stream: &Stream,
) -> Result<GenerationStepOutput, Exception> {
    let mut text_sampler = DefaultSampler;
    let mut audio_samplers = (0..model.depth_codebooks())
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    step_prompt_frame(
        model,
        state,
        frame,
        &mut text_sampler,
        &mut audio_samplers,
        0.0,
        0.0,
        None,
        stream,
    )
}

/// Runs a sequence of forced voice-prompt frames.
///
/// `voice_prompt_tokens` uses codec layout `[batch, 8, frames]`; the user side
/// is filled with PersonaPlex's sine-conditioning token frame and text is
/// forced to the existing text pad id.
pub fn prefill_voice_prompt_greedy<M: PromptModel>(
    model: &mut M,
    state: &mut GenerationState,
    voice_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<(), Exception> {
    if voice_prompt_tokens.shape().len() != 3
        || voice_prompt_tokens.dim(1) != AUDIO_TOKENS_PER_STREAM
    {
        return Err(Exception::custom(format!(
            "PersonaPlex voice prompt tokens must have shape [batch, 8, frames], got {:?}",
            voice_prompt_tokens.shape()
        )));
    }
    let batch = voice_prompt_tokens.dim(0);
    let sine = sine_frame(batch, stream)?;
    let text = text_padding_frame(batch, stream)?;
    for frame in 0..voice_prompt_tokens.dim(2) {
        let agent = voice_prompt_tokens.try_index_device((.., .., frame), stream)?;
        step_prompt_frame_greedy(
            model,
            state,
            PromptFrame {
                agent_audio_tokens: &agent,
                user_audio_tokens: &sine,
                text_token: &text,
            },
            stream,
        )?;
    }
    Ok(())
}

/// Runs a sequence of forced text-prompt tokens.
///
/// `text_prompt_tokens` is shaped `[batch, frames]` and should contain token ids
/// from the caller's PersonaPlex-compatible text tokenizer. The generated audio
/// side is forced to PersonaPlex silence while the user side is filled with the
/// sine-conditioning frame.
pub fn prefill_text_prompt_greedy<M: PromptModel>(
    model: &mut M,
    state: &mut GenerationState,
    text_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<(), Exception> {
    if text_prompt_tokens.shape().len() != 2 {
        return Err(Exception::custom(format!(
            "PersonaPlex text prompt tokens must have shape [batch, frames], got {:?}",
            text_prompt_tokens.shape()
        )));
    }
    let batch = text_prompt_tokens.dim(0);
    let silence = silence_frame(batch, stream)?;
    let sine = sine_frame(batch, stream)?;
    for frame in 0..text_prompt_tokens.dim(1) {
        let text = text_prompt_tokens
            .try_index_device((.., frame), stream)?
            .expand_dims(1, stream)?;
        step_prompt_frame_greedy(
            model,
            state,
            PromptFrame {
                agent_audio_tokens: &silence,
                user_audio_tokens: &sine,
                text_token: &text,
            },
            stream,
        )?;
    }
    Ok(())
}

/// Runs PersonaPlex's hybrid system prompt prefill from codec and text tokens.
///
/// `voice_prompt_tokens`, when present, uses codec layout `[batch, 8, frames]`.
/// `text_prompt_tokens` uses text-token layout `[batch, frames]`. Text wrapping
/// and tokenization stay outside this crate; callers can use
/// [`wrap_system_prompt`] before tokenizing with a compatible SentencePiece
/// tokenizer.
pub fn prefill_system_prompt_greedy<M: PromptModel>(
    model: &mut M,
    state: &mut GenerationState,
    voice_prompt_tokens: Option<&Array>,
    text_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<(), Exception> {
    if let Some(tokens) = voice_prompt_tokens {
        prefill_voice_prompt_greedy(model, state, tokens, stream)?;
    }
    prefill_text_prompt_greedy(model, state, text_prompt_tokens, stream)
}

#[cfg(test)]
mod tests {
    use super::{
        model_args_7b_v1, validate_model_config_value, AUDIO_TOKENS_PER_STREAM, TEXT_PADDING_TOKEN,
    };
    use crate::{
        realtime::{
            LoadedRealtimeModel, RealtimeModelKind, RealtimeSampling, RealtimeSpeechModel,
            RealtimeState, RealtimeStepInput,
        },
        sampler::DefaultSampler,
    };
    use safemlx::{Array, Device, DeviceType, ExecutionContext};

    #[test]
    fn validates_released_config_metadata() {
        let config = serde_json::json!({
            "model_type": "personaplex",
            "version": "7b-v1"
        });
        validate_model_config_value(&config).unwrap();
    }

    #[test]
    fn defaults_match_dual_stream_layout() {
        let args = model_args_7b_v1();
        assert_eq!(args.n_q, 16);
        assert_eq!(args.dep_q, 16);
        assert_eq!(args.generated_audio_codebooks(), 8);
        assert_eq!(args.input_audio_codebooks(), 8);
        assert_eq!(args.text_padding_token(), TEXT_PADDING_TOKEN);
        assert_eq!(
            args.audio_delays(),
            &[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        );
    }

    #[test]
    #[ignore = "requires SAFEMLX_PERSONAPLEX_DIR with the released checkpoint"]
    fn local_released_checkpoint_loads() {
        let model_dir = std::env::var("SAFEMLX_PERSONAPLEX_DIR")
            .expect("SAFEMLX_PERSONAPLEX_DIR must point to a PersonaPlex model directory");
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let model = super::load_model(&model_dir, ctx.stream(), ctx.stream()).unwrap();
        assert_eq!(model.args.dep_q, 16);
        assert_eq!(model.args.generated_audio_codebooks(), 8);
    }

    #[test]
    #[ignore = "requires SAFEMLX_PERSONAPLEX_DIR with the dense released checkpoint and Metal"]
    fn local_released_checkpoint_on_the_fly_q4_loads() {
        let model_dir = std::env::var("SAFEMLX_PERSONAPLEX_DIR")
            .expect("SAFEMLX_PERSONAPLEX_DIR must point to a PersonaPlex model directory");
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let model = crate::realtime::load_model_with_options(
            &model_dir,
            crate::models::ModelLoadOptions::with_quantization(
                crate::quantization::AffineQuantization::default(),
            ),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap()
        .into_moshi_model();
        assert_eq!(model.args.quantization.unwrap().bits(), 4);
        assert_eq!(model.args.dep_q, 16);
    }

    #[test]
    #[ignore = "requires SAFEMLX_PERSONAPLEX_DIR with the released checkpoint and Metal"]
    fn local_released_checkpoint_realtime_smoke() {
        let model_dir = std::env::var("SAFEMLX_PERSONAPLEX_DIR")
            .expect("SAFEMLX_PERSONAPLEX_DIR must point to a PersonaPlex model directory");
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let mut model =
            crate::realtime::load_model(&model_dir, ctx.stream(), ctx.stream()).unwrap();
        assert_eq!(model.kind(), RealtimeModelKind::PersonaPlex);
        let mut state = model.new_realtime_state();
        let stream = ctx.stream();

        let text_prompt =
            Array::full::<i32>(&[1, 2], Array::from_int(TEXT_PADDING_TOKEN), stream).unwrap();
        match (&mut model, &mut state) {
            (LoadedRealtimeModel::PersonaPlex(model), RealtimeState::PersonaPlex(state)) => {
                super::prefill_text_prompt_greedy(model, state, &text_prompt, stream).unwrap()
            }
            _ => panic!("expected PersonaPlex realtime model and state"),
        }

        let input = super::sine_frame(1, stream).unwrap();
        let mut text_sampler = DefaultSampler;
        let mut audio_samplers = (0..model.realtime_config().depth_audio_codebooks)
            .map(|_| DefaultSampler)
            .collect::<Vec<_>>();
        let mut emitted = None;
        for _ in 0..3 {
            let output = model
                .step_realtime(
                    &mut state,
                    RealtimeStepInput::encoded_audio(&input),
                    RealtimeSampling::new(&mut text_sampler, &mut audio_samplers, 0.0, 0.0, None),
                    stream,
                )
                .unwrap();
            emitted = output.output_audio_tokens;
        }
        let emitted = emitted.expect("PersonaPlex should emit a delay-aligned frame");
        assert_eq!(emitted.shape(), &[1, AUDIO_TOKENS_PER_STREAM]);
    }
}
