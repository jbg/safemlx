//! Pure-Rust GGUF container I/O and GGML tensor conversion.
//!
//! The crate has no tensor-framework or native-code dependency. [`Reader`]
//! parses descriptors with configurable resource limits and reads one tensor at
//! a time. [`Checkpoint`] validates complete single-file or sharded checkpoints
//! without reading tensor payloads and then streams their conversion. [`Writer`]
//! emits deterministic GGUF v3 files to seekable outputs. Canonical nonlinear
//! IQ encodings remain in their native block representation through conversion
//! and can be executed directly by IQ-aware runtimes.

mod catalog;
mod convert;
mod error;
mod format;
mod iquant;
#[doc(hidden)]
pub mod iquant_tables;
mod reader;
mod writer;

pub use catalog::{
    CatalogShard, CatalogTensor, Checkpoint, ConvertedCheckpointTensor, ConvertedTensorIter,
    LogicalDtype, LogicalTensorLayout, RawCheckpointTensor, TensorMaterializer,
    TranslatedTensorLayout,
};
pub use convert::{AffineTensor, ConvertedTensor, DenseDtype, DenseTensor, IQuantTensor};
pub use error::{Error, Result};
pub use format::{
    Endian, GgmlType, MetadataArray, MetadataValue, TensorDescriptor, DEFAULT_ALIGNMENT,
};
pub use reader::{Limits, OuterSelection, Reader};
pub use writer::{TensorInput, Writer, WriterOptions};
