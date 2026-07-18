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
their logical budgets do not increase physical capacity. The normal model-load
options select fully resident or layerwise host execution for DeepSeek-V3/R1,
Gemma 4, Inkling, Llama, Mistral, GPT-OSS, LFM2/LFM2.5,
Nemotron-H, Qwen3, Qwen3-Next, Qwen3-VL, Qwen3-VL-MoE, and Qwen3.5
language-model safetensors, plus Moshi and PersonaPlex realtime checkpoints,
including dense and MoE variants. Layerwise execution keeps complete decoder
blocks on a CPU stream and moves a bounded window to the execution device;
embeddings, final normalization, output heads, and architecture-owned cache or
recurrent state stay pinned on the device. Existing checkpoint-native packed
tensors remain packed, while split expert banks are packed one layer at a time
on the host. Transfers are
synchronous because MLX does not expose stream events. On Metal this validates
scheduling and logical residency within unified memory, not additional model
capacity. GGUF layerwise residency, KV-cache offload, pinned host buffers, and
asynchronous overlap are not supported.

Supported safetensors MoE models can instead use opt-in sparse expert caching
through `WeightResidency::SparseExpertCache`. This includes DeepSeek-V3/R1,
GPT-OSS, Inkling, LFM2, Nemotron-H, Qwen3, Qwen3-Next, Qwen3-VL-MoE, and
Qwen3.5-MoE. The cache accepts the checkpoint-native routed-expert layouts used
by each loader:

| Family | Sparse-cache expert layouts |
| --- | --- |
| DeepSeek-V3/R1 | packed or official split experts; dense, affine, and block-FP8 companions |
| GPT-OSS | native packed MXFP4 blocks, scales, and biases |
| Inkling | released interleaved `w13` plus `w2`, or runtime-packed banks |
| LFM2 | packed banks, including affine companions, or split `w1`/`w3`/`w2` experts |
| Nemotron-H | public split experts or packed banks with affine companions |
| Qwen3 and Qwen3-VL-MoE | packed dense/affine banks or supported split SwiGLU experts |
| Qwen3-Next and Qwen3.5-MoE | packed dense/affine/FP8 banks or supported split dense/FP8 experts |

Attention, routers, normalization, dense MLPs, and shared experts continue
through the layerwise host engine. Every routed expert is a separate cacheable
unit: hot copies remain on the execution device, warm copies may remain on the host
stream, and cold experts remain in the persistent checkpoint store. Packed
expert-major checkpoints are sliced on axis zero before materialization;
supported split experts use per-expert recipes. Checkpoint-native dense,
affine/MXFP4, and DeepSeek block-FP8 representations remain packed. Load-time
expert quantization is rejected because it cannot be performed lazily without
changing the existing loading contract.

Routes are synchronously inspected once per routed block because the vendored
MLX C API has no event or fence primitive. Duplicate requests share one cache
acquisition, then a temporary deterministic compact bank is evaluated before
its source leases are released. Compact-bank bytes are governed by a separate
scratch limit and are not claimed by the logical device-residency budget.
Apple unified memory does not provide extra physical capacity for the logical
host and device tiers. Disk-backed inference therefore depends heavily on
routing locality and filesystem page-cache behavior. Storage diagnostics report
mapped-shard activity and logical transfers, not exact physical disk reads.
Pure expert parallelism can combine the same sparse cache with DeepSeek-V3/R1,
GPT-OSS, Inkling, LFM2, Nemotron-H, Qwen3, Qwen3-Next, Qwen3-VL-MoE, and
Qwen3.5-MoE. Each rank catalogs and can acquire only its owned global experts;
remote expert tensors are omitted before checkpoint payload materialization.
GGUF remains fully resident and is rejected for sparse caching rather than
silently falling back to eager expert banks.

The route readback, batched pending-residency protocol, remaining synchronous
evaluation boundaries, and the event-backed completion API needed for genuine
cross-stream overlap are described in
[`safemlx-lm/EXPERT_CACHE_SYNCHRONIZATION.md`](safemlx-lm/EXPERT_CACHE_SYNCHRONIZATION.md).

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
