//! Media preprocessing before typed model prefill.

use std::{fs, path::Path};

use safemlx::Array;

use crate::{
    error::Error,
    models::input::{InputMetadata, InputPart, InputPayload, Modality, ModelInput},
};

/// Shared decoded-image operations.
pub mod image;
mod qwen;
/// Shared decoded-video validation, sampling, and timing operations.
pub mod video;

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
    pub fn image_rgb8(image: RgbImageView<'a>) -> Self {
        Self {
            modality: Modality::Image,
            payload: MediaPayload::Rgb8(image),
        }
    }

    /// Creates a decoded RGB8 video input using processor-default sampling.
    pub fn video_rgb8(frames: &'a [RgbImageView<'a>], source_fps: Option<f64>) -> Self {
        Self::video_rgb8_with_sampling(frames, source_fps, VideoSampling::ProcessorDefault)
    }

    /// Creates a decoded RGB8 video input with an explicit sampling policy.
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
}

/// Frame-selection policy for decoded video input.
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
    Rgb8(RgbImageView<'a>),
    /// Decoded RGB8 video frames and timing metadata.
    VideoFrames(VideoFrames<'a>),
}

#[derive(Debug)]
enum OwnedInputPayload {
    TokenIds(Array),
    Tensor(Array),
}

#[derive(Debug, Default)]
enum OwnedInputMetadata {
    #[default]
    None,
    GridThw(Array),
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
            OwnedInputPayload::Tensor(value) => InputPayload::Tensor(value),
        };
        InputPart {
            modality: self.modality,
            payload,
            metadata: InputMetadata {
                qwen_grid_thw: match &self.metadata {
                    OwnedInputMetadata::None => None,
                    OwnedInputMetadata::GridThw(value) => Some(value),
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
    Qwen(qwen::QwenProcessor),
}

impl ModelProcessor {
    pub(crate) fn load_qwen(model_dir: &Path) -> Result<Option<Self>, Error> {
        qwen::QwenProcessor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Qwen(processor),
            })
        })
    }

    /// Converts tokenized prompt text and decoded media into owned runtime input.
    ///
    /// Prompt placeholder tokens are replaced by ordered media parts. The model
    /// runtime expands each media part to its encoded sequence length.
    pub fn prepare_token_ids(
        &self,
        token_ids: &[u32],
        media: &[MediaInput<'_>],
    ) -> Result<PreparedModelInput, Error> {
        let mut no_text_encoder = |_text: &str| {
            Err(Error::Processor(
                "video preparation requires a text encoder for timestamps".to_string(),
            ))
        };
        self.prepare_token_ids_with_text_encoder(token_ids, media, &mut no_text_encoder)
    }

    /// Converts tokenized prompt text and decoded media using an encoder for
    /// processor-generated text such as video timestamps.
    pub fn prepare_token_ids_with_text_encoder(
        &self,
        token_ids: &[u32],
        media: &[MediaInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<PreparedModelInput, Error> {
        match &self.kind {
            ProcessorKind::Qwen(processor) => {
                processor.prepare_token_ids(token_ids, media, encode_text)
            }
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
        model_type: String,
    }

    let model_dir = model_dir.as_ref();
    let metadata: Metadata = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    let effective_type = metadata
        .text_config
        .as_ref()
        .map(|text| text.model_type.as_str())
        .unwrap_or(&metadata.model_type);
    match effective_type {
        "qwen3_5_moe" | "qwen3_5_moe_text" => ModelProcessor::load_qwen(model_dir),
        _ => Ok(None),
    }
}

struct PreparedMediaBinding {
    placeholder_token_id: u32,
    replacement_token_ids: Vec<u32>,
    part: PreparedInputPart,
}

fn bind_media_parts(
    token_ids: &[u32],
    placeholder_token_ids: &[u32],
    bindings: Vec<PreparedMediaBinding>,
) -> Result<PreparedModelInput, Error> {
    let placeholder_count = token_ids
        .iter()
        .filter(|token| placeholder_token_ids.contains(token))
        .count();
    if placeholder_count != bindings.len() {
        return Err(Error::Processor(format!(
            "prompt contains {placeholder_count} media placeholders but {} media items were supplied",
            bindings.len()
        )));
    }

    let mut parts = Vec::with_capacity(bindings.len() * 3 + 1);
    let mut media = bindings.into_iter();
    let mut start = 0;
    for (index, token) in token_ids.iter().enumerate() {
        if !placeholder_token_ids.contains(token) {
            continue;
        }
        let binding = media.next().expect("placeholder count was validated");
        if *token != binding.placeholder_token_id {
            return Err(Error::Processor(format!(
                "prompt media placeholder {} does not match supplied {} media item",
                token, binding.placeholder_token_id
            )));
        }
        if start < index {
            parts.push(PreparedInputPart::text_token_ids(&token_ids[start..index]));
        }
        if !binding.replacement_token_ids.is_empty() {
            parts.push(PreparedInputPart::text_token_ids(
                &binding.replacement_token_ids,
            ));
        }
        parts.push(binding.part);
        start = index + 1;
    }
    if start < token_ids.len() {
        parts.push(PreparedInputPart::text_token_ids(&token_ids[start..]));
    }
    if parts.is_empty() {
        return Err(Error::Processor(
            "prepared model input must not be empty".to_string(),
        ));
    }
    Ok(PreparedModelInput::new(parts))
}

#[cfg(test)]
mod tests {
    use super::{bind_media_parts, OwnedInputMetadata, PreparedInputPart, PreparedMediaBinding};
    use crate::models::input::{InputPayload, Modality};
    use safemlx::Array;

    #[test]
    fn placeholder_binding_preserves_part_order() {
        let image = PreparedInputPart::media_tensor(
            Modality::Image,
            Array::from_slice(&[0.0f32], &[1, 1]),
            OwnedInputMetadata::GridThw(Array::from_slice(&[1i32, 1, 1], &[1, 3])),
        );
        let prepared = bind_media_parts(
            &[10, 42, 11],
            &[42],
            vec![PreparedMediaBinding {
                placeholder_token_id: 42,
                replacement_token_ids: Vec::new(),
                part: image,
            }],
        )
        .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].modality, Modality::Text);
        assert_eq!(parts[1].modality, Modality::Image);
        assert_eq!(parts[2].modality, Modality::Text);
        assert!(matches!(parts[1].payload, InputPayload::Tensor(_)));
    }

    #[test]
    fn placeholder_binding_rejects_count_mismatch() {
        let error = bind_media_parts(&[10, 42, 11], &[42], Vec::new()).unwrap_err();
        assert!(error.to_string().contains("1 media placeholders"));
    }
}
