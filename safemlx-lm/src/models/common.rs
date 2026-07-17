//! Components shared by decoder-only causal language models.
//!
//! The implementation is organized by domain so model architectures can import
//! only the component groups they use.

pub mod attention;
/// Block-scaled E4M3 projections shared by native FP8 model families.
pub mod block_fp8;
pub mod convolution;
pub mod generation;
pub mod layers;
pub mod linear;
pub mod moe;
