# safemlx

Rust bindings for Apple's MLX machine learning framework.

`safemlx` provides a safe, idiomatic Rust interface over the low-level
`safemlx-sys` bindings. It includes array operations, neural-network building
blocks, transforms, optimizers, quantization helpers, and optional
SafeTensors support.

This crate targets Apple platforms supported by MLX. The default feature set
enables both Accelerate and Metal support.

## Features

- `accelerate`: enables Accelerate-backed MLX operations.
- `metal`: enables Metal-backed MLX operations.
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
