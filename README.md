# SafeMLX

SafeMLX is an unofficial Rust workspace for
[MLX](https://github.com/ml-explore/mlx). It provides Rust bindings and
higher-level libraries for array computation, neural networks, local language
model inference, GGUF and SafeTensors checkpoints, and neural audio codecs.

The project is intended for Rust applications that want to use MLX directly,
or run supported local models without a Python runtime. Apple silicon with
Metal is the primary platform; the core bindings also maintain CPU and NVIDIA
CUDA builds for x86-64 Linux and Windows.

SafeMLX is independent of Apple and is not an official MLX project.

## What is included

- An idiomatic Rust API for MLX arrays, operations, transforms, neural-network
  modules, optimizers, quantization, streams, and distributed execution.
- High-level loading and generation for supported text, multimodal, and
  realtime speech model families.
- SafeTensors and bounded, streaming GGUF checkpoint readers.
- Fully resident, layerwise, dense disk-streaming, and sparse expert-cache
  weight-loading policies for supported SafeTensors models.
- A native Mimi audio codec implementation for codec-token speech models.
- An example command-line text generator.

See [model and checkpoint support](doc/model-support.md) for the current model
families, formats, feature gates, and residency limitations.

## Workspace

| Path | Purpose |
| --- | --- |
| [`safemlx`](safemlx/) | Safe array, operation, neural-network, transform, optimizer, I/O, and distributed APIs |
| [`safemlx-sys`](safemlx-sys/) | Low-level bindings and native build integration for the vendored MLX C API |
| [`safemlx-gguf`](safemlx-gguf/) | Framework-independent, bounded pure-Rust GGUF reader, writer, and quantization converter |
| [`safemlx-lm`](safemlx-lm/) | Model loading, tokenization, generation, multimodal processing, parallelism, and weight residency |
| [`safemlx-lm-utils`](safemlx-lm-utils/) | Tokenizer and chat-template utilities |
| [`safemlx-codec`](safemlx-codec/) | Neural audio codec components, currently including Mimi |
| `safemlx-macros`, `safemlx-internal-macros` | Procedural macros used by the public crates |
| `safemlx-tests` | Workspace integration and compile-time tests; not published |
| [`examples/safemlx-lm-cli`](examples/safemlx-lm-cli/) | Example `safemlx-lm` text-generation binary; not published |

The `safemlx` package names distinguish these crates from the projects from
which parts of the workspace were derived.

## Getting started

The minimum supported Rust version is 1.88. Add the core crate to a project:

```toml
[dependencies]
safemlx = "0.1.3"
```

MLX records operations lazily. Create a stream for the target device, build the
array graph, and evaluate it before reading values on the host:

```rust
use safemlx::{array, Device, DeviceType, Stream};

let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
let left = array!([1.0, 2.0, 3.0]);
let right = array!([4.0, 5.0, 6.0]);
let sum = left.add(&right, &stream)?.into_evaluated()?;

assert_eq!(sum.as_slice::<f32>(), &[5.0, 7.0, 9.0]);
# Ok::<(), safemlx::error::Exception>(())
```

More examples are in the [`safemlx` crate README](safemlx/README.md). To try
local text generation with a supported model directory or GGUF file, use the
[`safemlx-lm` CLI example](examples/safemlx-lm-cli/README.md).

## Platforms

| Platform | Current support |
| --- | --- |
| macOS on Apple silicon | Default Accelerate and Metal backend; the full workspace is tested in CI |
| Linux x86-64 | CPU builds are checked; CUDA 12/13 with cuDNN 9 is optional, with compile coverage and an opt-in GPU smoke workflow |
| Windows x86-64 | Native MSVC CPU builds and file-format tests; optional CUDA 12.9/13.0 compile and link coverage, with runtime GPU validation opt-in and experimental |
| iOS/iPadOS, tvOS, visionOS | `safemlx` and `safemlx-sys` cross-build for Apple silicon devices and simulators; applications must bundle the generated `mlx.metallib` |

Native prerequisites, CUDA architecture selection, NCCL, Windows DLL handling,
Apple deployment targets, and Xcode integration are documented in the
[`safemlx-sys` README](safemlx-sys/README.md).

## Development status

SafeMLX is under active development and its pre-1.0 APIs may still change.
Model support is architecture-specific rather than a promise that every
checkpoint using a related name will load; `safemlx-lm` exposes config-checking
APIs and returns explicit errors for unsupported configurations.

Normal development uses the latest stable Rust release. Update it before
working on the workspace:

```sh
rustup update stable
```

The committed lockfile makes CI and local compatibility checks reproducible.
To verify the language-model crates and their default features against the
minimum supported Rust version, install Rust 1.88.0 and run:

```sh
rustup toolchain install 1.88.0
cargo +1.88.0 check --locked -p safemlx-lm-utils -p safemlx-lm
```

The macOS CI suite runs the workspace tests as follows, keeping tests that
exercise concurrent MLX use on a single test thread:

```sh
cargo test --workspace -- \
  --skip cpu_stream_creation_is_concurrent_safe \
  --skip async_eval_cpu_streams_are_concurrent_safe
cargo test --workspace concurrent_safe -- --test-threads=1
```

The platform workflows in [`.github/workflows`](.github/workflows/) are the
reference for the build and test commands exercised on Linux, Windows, and
Apple cross-compilation targets.

## Provenance

The core bindings originated in `mlx-rs`; the language-model crates were
derived from the `mlx-lm` work in a later `mlx-rs` fork; and the vendored C API
originated in `mlx-c`. All have since been modified in this repository. See
[the provenance notes](doc/provenance.md) for source repositories, import
commits, and the GGUF conversion lineage.

## License

The SafeMLX crates are available under MIT or Apache-2.0 unless a crate or
vendored component states otherwise. See [`LICENSE-MIT`](LICENSE-MIT),
[`LICENSE-APACHE`](LICENSE-APACHE), and the metadata and notices shipped with
individual components.
