//! Neural audio codec components built on `safemlx`.
//!
//! This crate keeps codec implementations optional and separate from
//! `safemlx-lm`. Realtime language models can operate on discrete codec tokens,
//! while applications that need audio encode/decode can depend on this crate.

#![warn(missing_docs)]

/// Mimi neural audio tokenizer support.
pub mod mimi;

use safemlx::{Array, Stream};

/// Common interface for neural audio tokenizers.
pub trait AudioTokenizer {
    /// Codec configuration.
    fn config(&self) -> AudioTokenizerConfig;

    /// Encodes mono PCM shaped `[batch, channels, samples]` into codec tokens.
    fn encode(&mut self, pcm: &Array, stream: &Stream) -> Result<Array, Error>;

    /// Decodes codec tokens shaped `[batch, codebooks, frames]` into PCM.
    fn decode(&mut self, codes: &Array, stream: &Stream) -> Result<Array, Error>;
}

/// Static metadata for pairing an audio tokenizer with a realtime model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioTokenizerConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: f64,
    /// Codec frame rate in Hz.
    pub frame_rate: f64,
    /// Number of audio channels supported by the codec.
    pub channels: i32,
    /// Number of active codebooks used for encode/decode.
    pub codebooks: i32,
    /// Codebook cardinality.
    pub cardinality: i32,
}

/// Errors returned by codec loaders and tokenization operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The requested operation is not implemented yet.
    #[error("{0}")]
    Unsupported(String),

    /// Invalid input or checkpoint shape.
    #[error("{0}")]
    InvalidShape(String),

    /// Underlying MLX error.
    #[error(transparent)]
    Exception(#[from] safemlx::error::Exception),

    /// Safetensors loading error.
    #[error(transparent)]
    LoadWeights(#[from] safemlx::error::IoError),

    /// Filesystem I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Boxed third-party codec loader error.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
