# safemlx

Rust bindings for the MLX machine learning framework.

`safemlx` provides a safe, idiomatic Rust interface over the low-level
`safemlx-sys` bindings. It includes array operations, neural-network building
blocks, transforms, optimizers, quantization helpers, optional SafeTensors
support, and typed GGUF tensor/metadata loading.

This crate targets macOS 14+, iOS/iPadOS 17+, tvOS 17+, and visionOS 1+ on
Apple silicon, as well as CPU-only and NVIDIA CUDA Linux systems and native
Windows x86-64 MSVC. The default feature set enables Accelerate and Metal on
Apple targets; those features are ignored on Linux and Windows, where `cuda`
can be selected explicitly. Cross-compilation, Xcode Metal-resource integration,
and native backend prerequisites are documented in the
[`safemlx-sys` README](../safemlx-sys/README.md).

## Features

- `accelerate`: enables Accelerate-backed MLX operations.
- `cuda`: builds MLX's CUDA backend on Linux or Windows x86-64 MSVC.
- `metal`: enables Metal-backed MLX operations.
- `nccl`: enables CUDA plus MLX's optional Linux-only NCCL distributed backend.
- `safetensors`: enables conversion between `Array` and
  `safetensors::TensorView`.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
safemlx = "0.1"
```

## Versioning

The `safemlx` crates use normal Rust semantic versioning. The initial
crates.io release is `0.1.0`.

## Status

`safemlx` is in active development.

## MSRV

The minimum supported Rust version is 1.85.0.

Each published crate declares its MSRV in `Cargo.toml`.

## License

Licensed under either MIT or Apache-2.0.
