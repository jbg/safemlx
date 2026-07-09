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
        }
    }
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
    use super::{validate, InputMetadata, InputPart, InputPayload, Modality, ModelInput};
    use safemlx::Array;

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
}
