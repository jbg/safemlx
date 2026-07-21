# safemlx

Rust bindings for the MLX machine learning framework.

`safemlx` provides a safe, idiomatic Rust interface over the low-level
`safemlx-sys` bindings. It includes array operations, neural-network building
blocks, transforms, optimizers, quantization helpers, optional SafeTensors
support, and typed streaming GGUF tensor/metadata loading.

This crate targets macOS 14+, iOS/iPadOS 17+, tvOS 17+, and visionOS 1+ on
Apple silicon, as well as CPU-only and NVIDIA CUDA Linux systems and native
Windows x86-64 MSVC. The default feature set enables Accelerate and Metal on
Apple targets; those features are ignored on Linux and Windows, where `cuda`
can be selected explicitly. Cross-compilation, Xcode Metal-resource integration,
and native backend prerequisites are documented in the
[`safemlx-sys` README](../safemlx-sys/README.md).

GGUF checkpoints are opened with `ops::GgufCheckpoint`. Opening validates all
canonical shard headers without reading payloads; `converted_tensors` and
`for_each_converted_tensor` then materialize one physical tensor as either a
dense array or one atomic affine weight/scales/biases group.
`GgufCheckpoint::materializer` provides indexed named access while reusing one
open shard reader, which is useful for bounded multi-tensor model transforms.

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

## Distributed MLX

The `distributed` module wraps MLX groups, collectives, and point-to-point
operations with owned handles and explicit streams. Non-strict initialization
keeps MLX's useful singleton fallback:

```rust
use safemlx::{distributed::{self, Backend}, Array, Device, DeviceType, Stream};

let group = distributed::init(false, Backend::Any)?;
let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
let input = Array::ones::<f32>(&[2], &stream)?;
let sum = distributed::all_sum(&input, &group, &stream)?;
# Ok::<(), safemlx::error::Exception>(())
```

Choose process-local devices with `distributed::device_for_local_rank`. A
global distributed rank is not a local GPU index because ranks may span
machines. In a one-process-per-visible-GPU launch, the local device index is
often zero: `CUDA_VISIBLE_DEVICES` has already restricted each process to one
GPU.

The real two-process Ring integration test is opt-in because it launches child
processes and opens loopback sockets. Run it on Unix with:

```console
cargo test -p safemlx --test distributed_ring ring_two_process_loopback -- --ignored --exact --nocapture
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
