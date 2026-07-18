# SafeMLX LM

This repository contains Rust crates for running language models with the MLX
framework on Apple silicon and Linux or Windows x86-64 systems with NVIDIA GPUs:

- `safemlx`
- `safemlx-sys`
- `safemlx-tests`
- `safemlx-lm`
- `safemlx-lm-utils`
- `examples/safemlx-lm-cli` (example `safemlx-lm` generation binary)

This fork carries additional model/runtime support, including Gemma 4 and
Thinking Machines Lab Inkling support, Gemma 4 assistant drafting, expanded model loading, and chat-template handling
for structured messages and tools.

`safemlx-lm` also exposes persistent, lazy safetensors checkpoint storage.
Checkpoint keys can be cataloged without creating MLX arrays, selected tensors
are safely materialized through mmap-pinning leases, and mapped payload shards
are reused under a deterministic bound. Rank-aware loading resolves placement
before materialization so remote-only indexed shards stay unopened.

The crate also provides a budgeted, architecture-independent residency manager
for caller-defined multi-tensor units. It materializes evaluated host or device
copies on explicit streams, applies pinned, windowed, and cacheable policies,
evicts eligible units deterministically, and protects in-use arrays with RAII
leases. Prefetch and execution windows are synchronous and feed existing
offload telemetry. Apple CPU and GPU tiers share physical unified memory, so
their logical budgets do not increase physical capacity. Llama and Mistral
safetensors have an explicit layerwise host-offload loader that keeps decoder
weights on a CPU stream and moves a bounded layer window to the execution
device. Static weights and KV caches stay on the device, and existing packed
affine or MXFP4 tensors remain packed. Transfers are synchronous because MLX
does not expose stream events. This expands capacity on discrete CUDA systems;
on Metal it validates scheduling and logical residency within unified memory.
The standard fully resident loader is unchanged. GGUF, load-time conversion,
other model families, pinned host buffers, and asynchronous overlap are not
supported by this path.

## Crates

The crates use SafeMLX package names on crates.io to avoid confusion with the
upstream `mlx-lm` packages:

```toml
safemlx = "0.1"
safemlx-sys = "0.1"
safemlx-lm = "0.4"
safemlx-lm-utils = "0.1"
```

## Provenance

The `safemlx`, `safemlx-sys`, `safemlx-macros`,
`safemlx-internal-macros`, and `safemlx-tests` crates were imported from
[`oxiglade/mlx-rs`](https://github.com/oxiglade/mlx-rs) at commit
`f4aa309c79b6be35255ca7d34157dfc10d9ed4c9`. Their upstream package authors
were Minghua Wu `<michael.wu1107@gmail.com>` and David Chavez
`<david@dcvz.io>`.

The vendored `safemlx-sys/src/mlx-c` source was imported from the upstream
[`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c) project.

The `safemlx-lm` and `safemlx-lm-utils` crates are derived from the `mlx-lm`
and `mlx-lm-utils` crates in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281) and merged
as commit `7c667cb7`. The original implementation and authorship belong to the
`oxideai/mlx-rs` contributors.

## Linux and CUDA

The core `safemlx` crates support CPU-only Linux builds and opt-in CUDA builds.
CUDA support requires the CUDA toolkit, cuDNN, BLAS, and LAPACK development
packages. Build the core crate with:

```sh
cargo build --release -p safemlx --features cuda
```

See the [safemlx-sys Linux and CUDA instructions](safemlx-sys/README.md#linux-and-cuda)
for prerequisites, architecture selection, and NCCL support.

## Windows x86-64 and CUDA

Native Windows uses the MSVC toolchain and a CMake-managed DLL boundary for
MLX's CUDA dependencies. CUDA 12.9 and 13.0 with cuDNN 9 are compile/link tested;
runtime GPU validation is optional and is not yet a required hosted check. See
the [native Windows CUDA instructions](safemlx-sys/README.md#native-windows-x86-64-cuda)
for Visual Studio requirements, environment variables, PowerShell commands,
DLL discovery, and troubleshooting. Windows ARM CUDA and NCCL are not supported.

## Apple mobile and spatial platforms

`safemlx` and `safemlx-sys` can be cross-compiled on macOS for iOS/iPadOS,
tvOS, and visionOS device and Apple Silicon simulator targets. The build also
exports the target-specific `mlx.metallib` needed by the application bundle.
See the [Apple target and Xcode integration instructions](safemlx-sys/README.md#apple-platform-targets).

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
