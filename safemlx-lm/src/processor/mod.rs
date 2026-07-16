//! Media preprocessing before typed model prefill.

use std::{fs, path::Path};

use safemlx::Array;

use crate::{
    error::Error,
    models::input::{InputMetadata, InputPart, InputPayload, Modality, ModelInput},
};

/// Shared PCM waveform validation and spectral operations.
#[cfg(feature = "audio-processing")]
pub mod audio;
mod gemma4;
/// Shared decoded-image operations.
#[cfg(feature = "image-processing")]
pub mod image;
mod inkling;
#[cfg(feature = "image-processing")]
mod qwen;
/// Shared decoded-video validation, sampling, and timing operations.
#[cfg(feature = "image-processing")]
pub mod video;

#[cfg(feature = "audio-processing")]
pub use audio::AudioWaveform;
#[cfg(feature = "image-processing")]
pub use image::RgbImageView;

/// One decoded media item supplied to a model processor.
#[derive(Debug, Clone, Copy)]
pub struct MediaInput<'a> {
    /// Declared modality of the item.
    pub modality: Modality,
    /// Decoded media payload.
    pub payload: MediaPayload<'a>,
}

impl<'a> MediaInput<'a> {
    /// Creates an RGB8 image input.
    #[cfg(feature = "image-processing")]
    pub fn image_rgb8(image: RgbImageView<'a>) -> Self {
        Self {
            modality: Modality::Image,
            payload: MediaPayload::Rgb8(image),
        }
    }

    /// Creates a decoded RGB8 video input using processor-default sampling.
    #[cfg(feature = "image-processing")]
    pub fn video_rgb8(frames: &'a [RgbImageView<'a>], source_fps: Option<f64>) -> Self {
        Self::video_rgb8_with_sampling(frames, source_fps, VideoSampling::ProcessorDefault)
    }

    /// Creates a decoded RGB8 video input with an explicit sampling policy.
    #[cfg(feature = "image-processing")]
    pub fn video_rgb8_with_sampling(
        frames: &'a [RgbImageView<'a>],
        source_fps: Option<f64>,
        sampling: VideoSampling,
    ) -> Self {
        Self {
            modality: Modality::Video,
            payload: MediaPayload::VideoFrames(VideoFrames {
                frames,
                source_fps,
                sampling,
            }),
        }
    }

    /// Creates a mono floating-point PCM audio input.
    #[cfg(feature = "audio-processing")]
    pub fn audio_f32(samples: &'a [f32], sample_rate: u32) -> Result<Self, Error> {
        Ok(Self {
            modality: Modality::Audio,
            payload: MediaPayload::AudioF32(AudioWaveform::new(samples, sample_rate)?),
        })
    }
}

/// One ordered input segment supplied to a model processor.
#[derive(Debug, Clone, Copy)]
pub enum ProcessorInput<'a> {
    /// Text that should be tokenized by the caller-provided encoder.
    Text(&'a str),
    /// Already-tokenized text IDs.
    TokenIds(&'a [u32]),
    /// Decoded media to preprocess and insert at this exact position.
    Media(MediaInput<'a>),
}

/// Frame-selection policy for decoded video input.
#[cfg(feature = "image-processing")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum VideoSampling {
    /// Uses the model processor's default frame rate and limits.
    #[default]
    ProcessorDefault,
    /// Uniformly samples approximately this many frames per second.
    Fps(f64),
    /// Uniformly samples exactly this many frames, capped by the source length.
    FrameCount(usize),
    /// Uses every decoded source frame.
    All,
}

/// Borrowed sequence of decoded RGB8 video frames.
#[cfg(feature = "image-processing")]
#[derive(Debug, Clone, Copy)]
pub struct VideoFrames<'a> {
    /// Frames in source order.
    pub frames: &'a [RgbImageView<'a>],
    /// Source frame rate used for sampling and timestamp generation.
    pub source_fps: Option<f64>,
    /// Frame-selection policy.
    pub sampling: VideoSampling,
}

/// Decoded data accepted by media processors.
#[derive(Debug, Clone, Copy)]
pub enum MediaPayload<'a> {
    /// Decoded RGB8 image pixels.
    #[cfg(feature = "image-processing")]
    Rgb8(RgbImageView<'a>),
    /// Decoded RGB8 video frames and timing metadata.
    #[cfg(feature = "image-processing")]
    VideoFrames(VideoFrames<'a>),
    /// Mono floating-point PCM samples and their sampling rate.
    #[cfg(feature = "audio-processing")]
    AudioF32(AudioWaveform<'a>),
    #[cfg(not(any(feature = "image-processing", feature = "audio-processing")))]
    #[doc(hidden)]
    _Lifetime(std::marker::PhantomData<&'a ()>),
}

#[derive(Debug)]
enum OwnedInputPayload {
    TokenIds(Array),
    #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
    Tensor(Array),
}

#[derive(Debug, Default)]
enum OwnedInputMetadata {
    #[default]
    None,
    #[cfg(feature = "image-processing")]
    GridThw(Array),
    #[cfg(feature = "image-processing")]
    PatchPositionIds(Array),
    #[cfg(feature = "audio-processing")]
    AudioMask(Array),
}

/// One owned part of a prepared model input.
#[derive(Debug)]
pub struct PreparedInputPart {
    modality: Modality,
    payload: OwnedInputPayload,
    metadata: OwnedInputMetadata,
}

impl PreparedInputPart {
    fn text_token_ids(ids: &[u32]) -> Self {
        Self {
            modality: Modality::Text,
            payload: OwnedInputPayload::TokenIds(Array::from_slice(ids, &[1, ids.len() as i32])),
            metadata: OwnedInputMetadata::default(),
        }
    }

    #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
    fn media_tensor(modality: Modality, tensor: Array, metadata: OwnedInputMetadata) -> Self {
        Self {
            modality,
            payload: OwnedInputPayload::Tensor(tensor),
            metadata,
        }
    }

    /// Borrows this owned part as a runtime input part.
    pub fn as_input_part(&self) -> InputPart<'_> {
        let payload = match &self.payload {
            OwnedInputPayload::TokenIds(value) => InputPayload::TokenIds(value),
            #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
            OwnedInputPayload::Tensor(value) => InputPayload::Tensor(value),
        };
        InputPart {
            modality: self.modality,
            payload,
            metadata: InputMetadata {
                qwen_grid_thw: match &self.metadata {
                    #[cfg(feature = "image-processing")]
                    OwnedInputMetadata::GridThw(value) => Some(value),
                    _ => None,
                },
                patch_position_ids: match &self.metadata {
                    #[cfg(feature = "image-processing")]
                    OwnedInputMetadata::PatchPositionIds(value) => Some(value),
                    _ => None,
                },
                audio_mask: match &self.metadata {
                    #[cfg(feature = "audio-processing")]
                    OwnedInputMetadata::AudioMask(value) => Some(value),
                    _ => None,
                },
            },
        }
    }
}

/// Owned runtime input produced by a media processor.
#[derive(Debug)]
pub struct PreparedModelInput {
    parts: Vec<PreparedInputPart>,
}

impl PreparedModelInput {
    fn new(parts: Vec<PreparedInputPart>) -> Self {
        Self { parts }
    }

    /// Returns the number of ordered runtime parts.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Returns true when no parts are present.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Borrows the prepared data as ordinary typed runtime input parts.
    ///
    /// Keep the returned vector alive for as long as the resulting
    /// [`ModelInput`] is used.
    pub fn input_parts(&self) -> Vec<InputPart<'_>> {
        self.parts
            .iter()
            .map(PreparedInputPart::as_input_part)
            .collect()
    }

    /// Calls `function` with a borrowed typed runtime input.
    pub fn with_model_input<T>(&self, function: impl FnOnce(ModelInput<'_>) -> T) -> T {
        let parts = self.input_parts();
        function(ModelInput::new(&parts))
    }
}

/// Architecture-dispatched media processor loaded from a model directory.
#[derive(Debug, Clone)]
pub struct ModelProcessor {
    kind: ProcessorKind,
}

#[derive(Debug, Clone)]
enum ProcessorKind {
    Gemma4(gemma4::Gemma4Processor),
    Inkling(inkling::InklingProcessor),
    #[cfg(feature = "image-processing")]
    Qwen(qwen::QwenProcessor),
}

impl ModelProcessor {
    pub(crate) fn load_gemma4(model_dir: &Path) -> Result<Option<Self>, Error> {
        gemma4::Gemma4Processor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Gemma4(processor),
            })
        })
    }

    pub(crate) fn load_inkling(model_dir: &Path) -> Result<Option<Self>, Error> {
        inkling::InklingProcessor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Inkling(processor),
            })
        })
    }

    #[cfg(feature = "image-processing")]
    pub(crate) fn load_qwen(model_dir: &Path) -> Result<Option<Self>, Error> {
        qwen::QwenProcessor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Qwen(processor),
            })
        })
    }

    /// Converts ordered text and decoded media segments into owned runtime input.
    pub fn prepare_input(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<PreparedModelInput, Error> {
        #[cfg(not(feature = "image-processing"))]
        let _ = &encode_text;
        match &self.kind {
            ProcessorKind::Gemma4(processor) => processor.prepare_input(input, encode_text),
            ProcessorKind::Inkling(processor) => processor.prepare_input(input, encode_text),
            #[cfg(feature = "image-processing")]
            ProcessorKind::Qwen(processor) => processor.prepare_input(input, encode_text),
        }
    }
}

/// Loads a supported media processor without loading model weights.
pub fn load_processor(model_dir: impl AsRef<Path>) -> Result<Option<ModelProcessor>, Error> {
    #[derive(serde::Deserialize)]
    struct Metadata {
        model_type: String,
        #[serde(default)]
        text_config: Option<TextMetadata>,
    }

    #[derive(serde::Deserialize)]
    struct TextMetadata {
        #[serde(default)]
        model_type: Option<String>,
    }

    let model_dir = model_dir.as_ref();
    let metadata: Metadata = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    let effective_type = metadata
        .text_config
        .as_ref()
        .and_then(|text| text.model_type.as_deref())
        .unwrap_or(&metadata.model_type);
    match effective_type {
        "inkling_mm_model" => ModelProcessor::load_inkling(model_dir),
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
            ModelProcessor::load_gemma4(model_dir)
        }
        "qwen3_vl" | "qwen3_vl_text" | "qwen3_vl_moe" | "qwen3_vl_moe_text" | "qwen3_5_moe"
        | "qwen3_5_moe_text" => {
            #[cfg(feature = "image-processing")]
            {
                ModelProcessor::load_qwen(model_dir)
            }
            #[cfg(not(feature = "image-processing"))]
            {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

fn prepared_model_input(parts: Vec<PreparedInputPart>) -> Result<PreparedModelInput, Error> {
    if parts.is_empty() {
        return Err(Error::Processor(
            "prepared model input must not be empty".to_string(),
        ));
    }
    Ok(PreparedModelInput::new(parts))
}

fn push_text_token_ids(parts: &mut Vec<PreparedInputPart>, token_ids: &[u32]) {
    if !token_ids.is_empty() {
        parts.push(PreparedInputPart::text_token_ids(token_ids));
    }
}
