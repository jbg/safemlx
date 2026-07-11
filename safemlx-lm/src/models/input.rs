//! Typed runtime inputs for model prefill.

use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::NewAxis, indexing::TryIndexOp},
    Array, Stream,
};

/// Ordered runtime input for model prefill.
#[derive(Debug, Clone, Copy)]
pub struct ModelInput<'a> {
    /// Ordered input parts consumed by the model.
    pub parts: &'a [InputPart<'a>],
}

impl<'a> ModelInput<'a> {
    /// Creates a typed input from ordered parts.
    pub fn new(parts: &'a [InputPart<'a>]) -> Self {
        Self { parts }
    }
}

/// One ordered input part with an explicit modality.
#[derive(Debug, Clone, Copy)]
pub struct InputPart<'a> {
    /// The modality of this part.
    pub modality: Modality,
    /// The payload for this part.
    pub payload: InputPayload<'a>,
    /// Optional typed metadata needed by some model families.
    pub metadata: InputMetadata<'a>,
}

impl<'a> InputPart<'a> {
    /// Creates a text token-id part.
    pub fn text_token_ids(token_ids: &'a Array) -> Self {
        Self {
            modality: Modality::Text,
            payload: InputPayload::TokenIds(token_ids),
            metadata: InputMetadata::default(),
        }
    }

    /// Creates an image tensor part.
    pub fn image_tensor(tensor: &'a Array, metadata: InputMetadata<'a>) -> Self {
        Self {
            modality: Modality::Image,
            payload: InputPayload::Tensor(tensor),
            metadata,
        }
    }

    /// Creates a video tensor part.
    pub fn video_tensor(tensor: &'a Array, metadata: InputMetadata<'a>) -> Self {
        Self {
            modality: Modality::Video,
            payload: InputPayload::Tensor(tensor),
            metadata,
        }
    }

    /// Creates an audio feature tensor part.
    pub fn audio_tensor(tensor: &'a Array, metadata: InputMetadata<'a>) -> Self {
        Self {
            modality: Modality::Audio,
            payload: InputPayload::Tensor(tensor),
            metadata,
        }
    }
}

/// Runtime modality.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Modality {
    /// Text token input.
    Text,
    /// Image tensor input.
    Image,
    /// Audio tensor input.
    Audio,
    /// Video tensor input.
    Video,
}

impl Modality {
    /// Returns a stable lowercase name for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Payload for one input part.
#[derive(Debug, Clone, Copy)]
pub enum InputPayload<'a> {
    /// Token ids shaped `[batch, sequence]`.
    TokenIds(&'a Array),
    /// Model-native tensor input for non-text modalities.
    Tensor(&'a Array),
    /// Already-projected embeddings shaped `[batch, sequence, hidden]`.
    Embeddings(&'a Array),
}

/// Optional metadata carried by an input part.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputMetadata<'a> {
    /// Qwen image/video grid metadata shaped as expected by the checkpoint.
    pub qwen_grid_thw: Option<&'a Array>,
    /// Image or video-frame patch positions shaped `[batch, patches, 2]`, with negative coordinates for padding.
    pub patch_position_ids: Option<&'a Array>,
    /// Valid-frame mask for model-native audio features.
    pub audio_mask: Option<&'a Array>,
}

impl<'a> InputMetadata<'a> {
    /// Creates empty metadata.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Creates metadata carrying Qwen `grid_thw`.
    pub fn qwen_grid_thw(grid_thw: &'a Array) -> Self {
        Self {
            qwen_grid_thw: Some(grid_thw),
            patch_position_ids: None,
            audio_mask: None,
        }
    }

    /// Creates metadata carrying generic 2-D patch positions.
    pub fn patch_position_ids(position_ids: &'a Array) -> Self {
        Self {
            qwen_grid_thw: None,
            patch_position_ids: Some(position_ids),
            audio_mask: None,
        }
    }

    /// Creates metadata carrying a valid-frame mask for audio features.
    pub fn audio_mask(mask: &'a Array) -> Self {
        Self {
            qwen_grid_thw: None,
            patch_position_ids: None,
            audio_mask: Some(mask),
        }
    }
}

/// Result of preparing typed input for a decoder model.
#[derive(Debug)]
pub(crate) enum PreparedPrefill {
    /// Text-only token IDs, which can use the architecture's ordinary fast path.
    Text(Array),
    /// Token IDs paired with embeddings after encoded media has been inserted.
    Embeddings { tokens: Array, embeddings: Array },
}

/// Placeholder token associated with one non-text modality.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModalityToken {
    pub modality: Modality,
    pub token_id: u32,
}

enum PreparedPart {
    Text(Vec<u32>),
    Media(Vec<usize>),
}

struct MediaEmbedding {
    modality: Modality,
    embeddings: Array,
    consumed: bool,
}

/// Assembles ordered typed input into decoder tokens and embeddings.
///
/// Architecture implementations supply only text embedding and media encoding;
/// placeholder matching and ordered insertion remain shared across model families.
pub(crate) fn prepare_decoder_prefill(
    input: ModelInput<'_>,
    modality_tokens: &[ModalityToken],
    hidden_size: i32,
    model_name: &str,
    stream: &Stream,
    mut embed_text: impl FnMut(&Array, &Stream) -> Result<Array, Exception>,
    mut encode_media: impl FnMut(&InputPart<'_>, &Stream) -> Result<Vec<Array>, Exception>,
) -> Result<PreparedPrefill, Exception> {
    validate(input)?;
    if input
        .parts
        .iter()
        .all(|part| part.modality == Modality::Text)
    {
        return Ok(PreparedPrefill::Text(text_token_ids(input, stream)?));
    }

    let mut prepared_parts = Vec::new();
    let mut media_embeddings = Vec::new();
    for part in input.parts {
        match (part.modality, part.payload) {
            (Modality::Text, InputPayload::TokenIds(tokens)) => {
                ensure_batch_one(tokens, &format!("{model_name} text tokens"))?;
                prepared_parts.push(PreparedPart::Text(token_ids_from_array(tokens, stream)?));
            }
            (Modality::Text, InputPayload::Embeddings(_)) => {
                return Err(Exception::custom(format!(
                    "{model_name} typed input does not support text embeddings yet"
                )));
            }
            (Modality::Text, InputPayload::Tensor(_)) => {
                return Err(Exception::custom(format!(
                    "{model_name} text input does not accept tensor payloads"
                )));
            }
            (modality, _) => {
                if !modality_tokens
                    .iter()
                    .any(|entry| entry.modality == modality)
                {
                    return Err(Exception::custom(format!(
                        "{model_name} typed input does not support {} input yet",
                        modality.as_str()
                    )));
                }
                let chunks = encode_media(part, stream)?;
                if chunks.is_empty() {
                    return Err(Exception::custom(format!(
                        "{model_name} {} input produced no embeddings",
                        modality.as_str()
                    )));
                }
                let mut indices = Vec::with_capacity(chunks.len());
                for embeddings in chunks {
                    ensure_batch_one(
                        &embeddings,
                        &format!("{model_name} {} embeddings", modality.as_str()),
                    )?;
                    ensure_hidden_size(
                        &embeddings,
                        hidden_size,
                        &format!("{model_name} {} embeddings", modality.as_str()),
                    )?;
                    indices.push(media_embeddings.len());
                    media_embeddings.push(MediaEmbedding {
                        modality,
                        embeddings,
                        consumed: false,
                    });
                }
                prepared_parts.push(PreparedPart::Media(indices));
            }
        }
    }

    for entry in modality_tokens {
        let placeholders = prepared_parts
            .iter()
            .filter_map(|part| match part {
                PreparedPart::Text(ids) => Some(ids),
                PreparedPart::Media(_) => None,
            })
            .flatten()
            .filter(|id| **id == entry.token_id)
            .count();
        let chunks = media_embeddings
            .iter()
            .filter(|media| media.modality == entry.modality)
            .count();
        if placeholders != 0 && placeholders != chunks {
            return Err(Exception::custom(format!(
                "{model_name} {} input produced {chunks} embedding groups but prompt contains {placeholders} placeholders",
                entry.modality.as_str()
            )));
        }
    }

    let mut token_parts = Vec::new();
    let mut embedding_parts = Vec::new();
    for part in prepared_parts {
        match part {
            PreparedPart::Text(ids) => {
                for id in ids {
                    let modality = modality_tokens
                        .iter()
                        .find(|entry| entry.token_id == id)
                        .map(|entry| entry.modality);
                    if let Some(modality) = modality {
                        let media = media_embeddings
                            .iter_mut()
                            .find(|media| media.modality == modality && !media.consumed)
                            .ok_or_else(|| {
                                Exception::custom(format!(
                                    "{model_name} {} placeholder has no matching input",
                                    modality.as_str()
                                ))
                            })?;
                        media.consumed = true;
                        let embeddings = media.embeddings.clone();
                        token_parts.push(placeholder_tokens(
                            id,
                            embeddings.dim(1) as usize,
                            stream,
                        )?);
                        embedding_parts.push(embeddings);
                    } else {
                        let tokens = token_ids_array(&[id], stream)?;
                        embedding_parts.push(embed_text(&tokens, stream)?);
                        token_parts.push(tokens);
                    }
                }
            }
            PreparedPart::Media(indices) => {
                for index in indices {
                    let media = &mut media_embeddings[index];
                    if media.consumed {
                        continue;
                    }
                    media.consumed = true;
                    let token_id = modality_tokens
                        .iter()
                        .find(|entry| entry.modality == media.modality)
                        .expect("media modality was validated")
                        .token_id;
                    token_parts.push(placeholder_tokens(
                        token_id,
                        media.embeddings.dim(1) as usize,
                        stream,
                    )?);
                    embedding_parts.push(media.embeddings.clone());
                }
            }
        }
    }

    Ok(PreparedPrefill::Embeddings {
        tokens: concatenate_axis(&token_parts, 1, stream)?,
        embeddings: concatenate_axis(&embedding_parts, 1, stream)?,
    })
}

pub(crate) fn ensure_batch_one(array: &Array, name: &str) -> Result<(), Exception> {
    let shape = array.shape();
    if shape.first() != Some(&1) {
        return Err(Exception::custom(format!(
            "{name} currently supports batch size 1, got shape {shape:?}"
        )));
    }
    Ok(())
}

pub(crate) fn ensure_hidden_size(
    array: &Array,
    hidden_size: i32,
    name: &str,
) -> Result<(), Exception> {
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != hidden_size {
        return Err(Exception::custom(format!(
            "{name} must be shaped [batch, sequence, {hidden_size}], got {shape:?}"
        )));
    }
    Ok(())
}

fn placeholder_tokens(token_id: u32, len: usize, stream: &Stream) -> Result<Array, Exception> {
    token_ids_array(&vec![token_id; len], stream)
}

pub(crate) fn token_ids_from_array(tokens: &Array, stream: &Stream) -> Result<Vec<u32>, Exception> {
    ensure_batch_one(tokens, "typed input token ids")?;
    if tokens.ndim() != 2 {
        return Err(Exception::custom(format!(
            "typed input token ids must have rank 2, got {:?}",
            tokens.shape()
        )));
    }
    let mut ids = Vec::with_capacity(tokens.dim(1) as usize);
    for index in 0..tokens.dim(1) {
        ids.push(
            tokens
                .try_index_device((0, index), stream)?
                .item::<u32>(stream),
        );
    }
    Ok(ids)
}

/// Validates basic modality/payload compatibility.
pub fn validate(input: ModelInput<'_>) -> Result<(), Exception> {
    if input.parts.is_empty() {
        return Err(Exception::custom(
            "model input must contain at least one part",
        ));
    }
    for part in input.parts {
        match (part.modality, part.payload) {
            (Modality::Text, InputPayload::TokenIds(tokens)) => validate_token_ids(tokens)?,
            (Modality::Text, InputPayload::Embeddings(embeddings)) => {
                validate_embeddings(embeddings, "text embeddings")?
            }
            (Modality::Text, InputPayload::Tensor(_)) => {
                return Err(Exception::custom(
                    "text input does not accept tensor payloads",
                ));
            }
            (Modality::Image | Modality::Audio | Modality::Video, InputPayload::Tensor(tensor)) => {
                validate_rank_at_least(tensor, 2, part.modality.as_str())?;
            }
            (
                Modality::Image | Modality::Audio | Modality::Video,
                InputPayload::Embeddings(embeddings),
            ) => validate_embeddings(embeddings, part.modality.as_str())?,
            (Modality::Image | Modality::Audio | Modality::Video, InputPayload::TokenIds(_)) => {
                return Err(Exception::custom(format!(
                    "{} input does not accept token-id payloads",
                    part.modality.as_str()
                )));
            }
        }
    }
    Ok(())
}

/// Builds a `[batch, sequence]` token array from text-only typed input.
pub fn text_token_ids(input: ModelInput<'_>, stream: &Stream) -> Result<Array, Exception> {
    validate(input)?;
    let mut parts = Vec::new();
    for part in input.parts {
        match (part.modality, part.payload) {
            (Modality::Text, InputPayload::TokenIds(tokens)) => parts.push(tokens.clone()),
            (Modality::Text, InputPayload::Embeddings(_)) => {
                return Err(Exception::custom(
                    "text embeddings are not supported by this model",
                ));
            }
            _ => {
                return Err(Exception::custom(format!(
                    "{} input is not supported by this model",
                    part.modality.as_str()
                )));
            }
        }
    }
    concatenate_token_parts(&parts, stream)
}

/// Converts a slice of token IDs to a batch-1 text input array.
pub fn token_ids_array(token_ids: &[u32], stream: &Stream) -> Result<Array, Exception> {
    Array::from(token_ids)
        .try_index_device(NewAxis, stream)
        .map_err(Into::into)
}

fn concatenate_token_parts(parts: &[Array], stream: &Stream) -> Result<Array, Exception> {
    if parts.is_empty() {
        return Err(Exception::custom("text input must contain token ids"));
    }
    if parts.len() == 1 {
        return Ok(parts[0].clone());
    }
    let refs = parts.iter().collect::<Vec<_>>();
    concatenate_axis(&refs, 1, stream).map_err(Into::into)
}

fn validate_token_ids(tokens: &Array) -> Result<(), Exception> {
    let shape = tokens.shape();
    if shape.len() != 2 {
        return Err(Exception::custom(format!(
            "token ids must be shaped [batch, sequence], got {shape:?}"
        )));
    }
    if shape[0] <= 0 || shape[1] <= 0 {
        return Err(Exception::custom(format!(
            "token ids must have non-empty batch and sequence dimensions, got {shape:?}"
        )));
    }
    Ok(())
}

fn validate_embeddings(embeddings: &Array, name: &str) -> Result<(), Exception> {
    let shape = embeddings.shape();
    if shape.len() != 3 {
        return Err(Exception::custom(format!(
            "{name} must be shaped [batch, sequence, hidden], got {shape:?}"
        )));
    }
    if shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 {
        return Err(Exception::custom(format!(
            "{name} must have non-empty dimensions, got {shape:?}"
        )));
    }
    Ok(())
}

fn validate_rank_at_least(tensor: &Array, min_rank: usize, name: &str) -> Result<(), Exception> {
    if tensor.shape().len() < min_rank {
        return Err(Exception::custom(format!(
            "{name} tensor must have rank at least {min_rank}, got shape {:?}",
            tensor.shape()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        prepare_decoder_prefill, validate, InputMetadata, InputPart, InputPayload, Modality,
        ModalityToken, ModelInput, PreparedPrefill,
    };
    use safemlx::{Array, Device, DeviceType, Stream};

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    #[test]
    fn validates_text_token_part() {
        let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
        let parts = [InputPart::text_token_ids(&tokens)];

        validate(ModelInput::new(&parts)).unwrap();
    }

    #[test]
    fn rejects_empty_input() {
        let err = validate(ModelInput::new(&[])).unwrap_err();

        assert!(err.to_string().contains("at least one part"));
    }

    #[test]
    fn rejects_text_tensor_payload() {
        let tensor = Array::from_slice(&[0.0_f32, 1.0], &[1, 2]);
        let parts = [InputPart {
            modality: Modality::Text,
            payload: InputPayload::Tensor(&tensor),
            metadata: InputMetadata::empty(),
        }];

        let err = validate(ModelInput::new(&parts)).unwrap_err();

        assert!(err
            .to_string()
            .contains("text input does not accept tensor"));
    }

    #[test]
    fn accepts_future_modality_tensor_payloads() {
        let tensor = Array::from_slice(&[0.0_f32, 1.0], &[1, 2]);
        let parts = [
            InputPart {
                modality: Modality::Audio,
                payload: InputPayload::Tensor(&tensor),
                metadata: InputMetadata::empty(),
            },
            InputPart {
                modality: Modality::Video,
                payload: InputPayload::Tensor(&tensor),
                metadata: InputMetadata::empty(),
            },
        ];

        validate(ModelInput::new(&parts)).unwrap();
    }

    #[test]
    fn shared_prefill_replaces_placeholder_and_preserves_order() {
        let stream = stream();
        let tokens = Array::from_slice(&[10u32, 42, 11], &[1, 3]);
        let image = Array::from_slice(&[1.0f32; 8], &[1, 2, 4]);
        let parts = [
            InputPart::text_token_ids(&tokens),
            InputPart {
                modality: Modality::Image,
                payload: InputPayload::Embeddings(&image),
                metadata: InputMetadata::empty(),
            },
        ];
        let prepared = prepare_decoder_prefill(
            ModelInput::new(&parts),
            &[ModalityToken {
                modality: Modality::Image,
                token_id: 42,
            }],
            4,
            "test",
            &stream,
            |_tokens, _stream| Ok(Array::from_slice(&[0.0f32; 4], &[1, 1, 4])),
            |part, _stream| match part.payload {
                InputPayload::Embeddings(value) => Ok(vec![value.clone()]),
                _ => unreachable!(),
            },
        )
        .unwrap();
        let PreparedPrefill::Embeddings { tokens, embeddings } = prepared else {
            panic!("expected prepared embeddings")
        };
        assert_eq!(tokens.shape(), &[1, 4]);
        assert_eq!(embeddings.shape(), &[1, 4, 4]);
        let ids = super::token_ids_from_array(&tokens, &stream).unwrap();
        assert_eq!(ids, vec![10, 42, 42, 11]);
    }

    #[test]
    fn shared_prefill_rejects_placeholder_group_mismatch() {
        let stream = stream();
        let tokens = Array::from_slice(&[42u32, 42], &[1, 2]);
        let image = Array::from_slice(&[1.0f32; 4], &[1, 1, 4]);
        let parts = [
            InputPart::text_token_ids(&tokens),
            InputPart {
                modality: Modality::Image,
                payload: InputPayload::Embeddings(&image),
                metadata: InputMetadata::empty(),
            },
        ];
        let error = prepare_decoder_prefill(
            ModelInput::new(&parts),
            &[ModalityToken {
                modality: Modality::Image,
                token_id: 42,
            }],
            4,
            "test",
            &stream,
            |_tokens, _stream| Ok(Array::from_slice(&[0.0f32; 4], &[1, 1, 4])),
            |part, _stream| match part.payload {
                InputPayload::Embeddings(value) => Ok(vec![value.clone()]),
                _ => unreachable!(),
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("2 placeholders"));
    }
}
