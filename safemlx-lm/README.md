# safemlx-lm

`safemlx-lm` is a Rust runtime for MLX language models.

The crate is derived from the `mlx-lm` crate in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281), merged as
commit `7c667cb7`.

The original implementation and authorship belong to the `oxideai/mlx-rs`
contributors.

This fork adds model/runtime support including Gemma 4 loading, Gemma 4
assistant drafting, expanded model dispatch, and related generation utilities.

## GGUF models

The standard `models::load_model` and `models::LoadedModel::load` entry points
accept Hugging Face-style model directories for Gemma 4, Llama, dense Mistral,
dense and sparse-MoE Nemotron-H, Qwen3, and Qwen3.5 MoE. They also accept the
GGUF architectures listed below. Canonically named sharded GGUF checkpoints
are supported by passing the first
`-00001-of-NNNNN.gguf` shard; the remaining shards are discovered and
validated automatically. Put `tokenizer.json` next to a GGUF file when using
`LoadedModel` or
`load_tokenizer`; adjacent
`tokenizer_config.json` and `chat_template.jinja` files are used when present.

```rust,ignore
use safemlx_lm::models::LoadedModel;

let model = LoadedModel::load(
    "/path/to/model-00001-of-00004.gguf",
    execution_stream,
    cpu_weights_stream,
)?;
```

Dense GGUF tensors are loaded directly. MLX-native packed loading is enabled
for Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0, including checkpoints
that mix packed and dense matrices. Q4_K and Q5_K are losslessly repacked to
MLX's 32-value affine groups, while Q2_K, Q3_K, and Q6_K map exactly to
16-value affine groups. Group-16 K-quants use tiled quantized matrix kernels for
prefill and the corresponding vector kernels for decode. These formats execute
without expanding matrix weights to float16.
Q5_0 and Q5_1 tensors are converted to float16 while loading; other GGUF
quantization types use MLX's bundled converter when
supported, and unsupported tensor types return an error. Model dispatch uses
`general.architecture`; the current GGUF adapters support text-only `gemma4`,
`llama`, `mistral`, `nemotron_h`, `nemotron_h_moe`, `qwen3`, `qwen3moe`, dense
`qwen35`, and `qwen35moe` architectures.
Nemotron-H routed expert banks retain Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 packed weights
and execute through selected-expert quantized matrix multiplication. Qwen3 MoE
uses the same packed expert-major execution with per-tensor mixed Q2/Q3/Q4/Q5/Q6/Q8
settings. Dense Qwen3.5 uses the hybrid linear/full-attention runtime with
conventional SwiGLU layers; Qwen3.5 MoE keeps its
Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 routed expert banks packed while loading mixed
quantization types. Gemma 4
multimodal projectors, MoE, and assistant-drafter files are separate formats
and are not handled by the initial Gemma 4 adapter. Nemotron-H latent-space MoE and
Omni/multimodal checkpoints remain separate formats. Qwen3-VL and Qwen3.5-VL
GGUF files are not handled by these text-only adapters.

## Usage

```toml
[dependencies]
safemlx-lm = { version = "0.4", features = ["image-processing"] }
```

Dense checkpoints can be affine-quantized while loading through the same
architecture-dispatched API used for ordinary loading:

```rust,ignore
use safemlx_lm::{
    models::{LoadedModel, ModelLoadOptions},
    quantization::AffineQuantization,
};

let options = ModelLoadOptions::with_quantization(AffineQuantization::new(64, 4)?);
let model = LoadedModel::load_with_options(model_dir, options, stream, weights_stream)?;
```

The realtime counterpart is `load_realtime_model_with_options`. Both APIs
recognize matching pre-quantized checkpoints and load them directly rather
than quantizing them again. A requested format that differs from existing
checkpoint metadata is an error.

### Quantized loading coverage

| Architecture | Dense | Existing quantized | Affine on load | High-level dispatch | Special policy |
|---|---:|---:|---:|---:|---|
| Llama | yes | MLX affine | yes | `LoadedModel` | Linear, embedding, tied/untied head targets |
| Mistral | yes | MLX affine | yes | `LoadedModel` | Reuses the Llama-compatible dense decoder; configured sliding attention uses bounded KV caches |
| Qwen3 | yes | MLX affine | yes | `LoadedModel` | Linear, embedding, tied/untied head targets |
| Gemma 4 | yes | MLX affine | yes | `LoadedModel` | Transformer and modality-bridge projections use affine storage; vision/audio towers remain deliberately dense |
| Gemma 4 assistant | yes | MLX affine | yes | assistant loader with `ModelLoadOptions` | Transformer/projection/head targets; ordered masked-embedding heads return a capability error |
| Nemotron-H | yes | no | capability error | `LoadedModel` (dense) | Packed rank-3 routed experts require an affine grouped-matmul kernel |
| Qwen3.5/3.6-MoE | yes | block FP8 | yes, from dense checkpoints | `LoadedModel` (dense/FP8/affine) | Rank-3 expert banks are quantized row-wise and executed with routed `gather_qmm`; native FP8 checkpoints cannot be requantized to affine on load |
| Moshi | yes | MLX affine | yes | realtime loader | Temporal/depth projections and embeddings; no codec dependency |
| PersonaPlex | yes, transformed PyTorch layout | MLX affine | yes | realtime loader | Preserves per-depth checkpoint transformation; no codec dependency |

On-load selection is driven by the target module parameter tree, not by
blindly quantizing every rank-2 checkpoint tensor. Therefore specialized
convolutions, modality towers, routers, and packed expert banks stay dense only
when the architecture explicitly supports that policy, or the request is
rejected before weights are loaded.

For Gemma 4 or Qwen image prompts, pass text and media as ordered processor
segments. Media is inserted where the segment appears; callers do not put
image/video/audio media tokens in rendered prompt text:

```rust,ignore
use safemlx_lm::processor::{MediaInput, ProcessorInput, RgbImageView};

let image = RgbImageView::packed(rgb_pixels, width, height)?;
let prepared = model.prepare_input(
    &[
        ProcessorInput::Text(prompt_before_image),
        ProcessorInput::Media(MediaInput::image_rgb8(image)),
        ProcessorInput::Text(prompt_after_image),
    ],
)?;
let logits = model.prefill_prepared_input_with_cache(
    &prepared,
    &mut cache,
    stream,
)?;
```

Decoded videos use the same processor with an ordered frame sequence and source
frame rate. Container decoding remains with the caller:

```rust,ignore
let frames = decoded_rgb_frames
    .iter()
    .map(|frame| RgbImageView::packed(frame, width, height))
    .collect::<Result<Vec<_>, _>>()?;
let prepared = model.prepare_input(
    &[
        ProcessorInput::Text(prompt_before_video),
        ProcessorInput::Media(MediaInput::video_rgb8(&frames, Some(source_fps))),
        ProcessorInput::Text(prompt_after_video),
    ],
)?;
```

The optional `image-processing` feature enables architecture-dispatched Gemma 4
and Qwen processors. Shared code owns decoded-image validation, frame sampling,
and timestamp operations; each processor adds its model-native patch packing,
prompt format, metadata, and ordered media insertion. Gemma samples up to 32 frames
by default and encodes each timestamped frame through its vision tower. Qwen
uses its temporal patch packing and timestamp format. Without the feature,
callers can still supply Gemma 4 or Qwen `Image/Tensor` and `Video/Tensor`
inputs directly without depending on the `image` crate.

Gemma 4 audio accepts model-native log-mel tensors through the typed input API
without optional dependencies. Enable `audio-processing` to prepare mono `f32`
PCM in the shared processor instead:

```toml
[dependencies]
safemlx-lm = { version = "0.4", features = ["audio-processing"] }
```

```rust,ignore
use safemlx_lm::processor::{MediaInput, ProcessorInput};

let audio = MediaInput::audio_f32(mono_pcm, sample_rate)?;
let prepared = model.prepare_input(&[
    ProcessorInput::Text(prompt_before_audio),
    ProcessorInput::Media(audio),
    ProcessorInput::Text(prompt_after_audio),
])?;
let logits = model.prefill_prepared_input_with_cache(&prepared, &mut cache, stream)?;
```

The common audio processor validates and resamples neither channels nor sample
rate: Gemma 4 currently requires mono 16 kHz PCM. It computes the model's log-mel
features and valid-frame mask. The optional FFT dependency is only enabled by
`audio-processing`; callers that provide `Audio/Tensor` and `audio_mask` directly
do not pay that dependency cost.

## Realtime encoded audio

The `realtime` module defines a codec-free API for realtime speech-to-speech
models. Models consume discrete codec-token frames and emit delay-aligned
generated codec-token frames; callers keep audio encoding, decoding, transport,
and device I/O outside `safemlx-lm`.

Use `load_realtime_model` when the model directory contains a realtime
codec-token model. It dispatches PersonaPlex, Moshi, and future realtime model
families separately from the chat/text `LoadedModel` path:

```rust,ignore
use safemlx_lm::{
    load_realtime_model,
    realtime::{RealtimeSampling, RealtimeSpeechModel, RealtimeStepInput},
    sampler::DefaultSampler,
};

let mut model = load_realtime_model(model_dir, stream, weights_stream)?;
let config = model.realtime_config();
let mut state = model.new_realtime_state();
let mut text_sampler = DefaultSampler;
let mut audio_samplers = (0..config.depth_audio_codebooks)
    .map(|_| DefaultSampler)
    .collect::<Vec<_>>();

// Your codec supplies one user/input-side frame shaped
// [batch, config.input_audio_codebooks].
let output = model.step_realtime(
    &mut state,
    RealtimeStepInput::encoded_audio(&encoded_input_frame),
    RealtimeSampling::new(&mut text_sampler, &mut audio_samplers, 0.0, 0.0, None),
    stream,
)?;

if let Some(codec_tokens) = output.output_audio_tokens {
    // Decode [batch, config.generated_audio_codebooks] with your codec.
}
```

The `models::moshi` module implements Moshi's temporal and depth language
models over pre-tokenized Mimi streams. `GenerationState` accepts one
input-side Mimi frame at a time and returns delay-aligned generated-side Mimi
frames; `generate_encoded_greedy` is the offline sequence convenience API.
Sequence tensors use Mimi's `[batch, codebooks, frames]` layout.

`models::personaplex` exposes PersonaPlex's Moshi-family realtime token API,
published 7B v1 defaults, dual-stream codebook layout, and hybrid system-prompt
helpers. It can load the released Hugging Face PyTorch-layout
`model.safetensors` directly via the shared Moshi-family PyTorch importer.

PersonaPlex consumes 8 user-side codec codebooks per realtime frame and emits 8
agent-side codec codebooks per output frame. Its depth transformer still samples
or teacher-forces 16 codebooks, so realtime sampling requires 16 audio samplers.
Prompt helpers remain token-only: use `wrap_system_prompt` before external text
tokenization, pass text ids shaped `[batch, frames]` to
`prefill_text_prompt_greedy`, and optionally pass agent voice codec tokens
shaped `[batch, 8, frames]` to `prefill_system_prompt_greedy`.

Mimi audio encoding/decoding and audio device I/O deliberately remain outside
`safemlx-lm`. The sibling `safemlx-codec` crate provides safemlx-native codec
building blocks, including Mimi checkpoint loading, PCM encode/decode,
residual-vector quantization, and stateful tokens-to-PCM decode. Audio device
I/O remains optional codec surface rather than an `safemlx-lm` dependency.

Moshi loads dense and MLX affine-quantized checkpoints. For the original
released Moshika/Moshiko repositories, the loader uses Moshi's built-in v0.1
config when the model directory has no `config.json`.

## Checkpoint quantization

The generic checkpoint converter quantizes eligible two-dimensional
`*.weight` tensors one at a time, writes bounded-size safetensors shards, and
copies tokenizer and other model assets. Its output follows the MLX-LM affine
convention: packed `weight` tensors have sibling `scales` and `biases`, while
`config.json` contains identical `quantization` and `quantization_config`
objects.

```sh
cargo run --release -p safemlx-lm --example quantize_checkpoint -- \
  /path/to/dense-model /path/to/model-4bit \
  --group-size 64 --bits 4
```

Use repeatable `--include` and `--exclude` substring filters to experiment on
part of any safetensors checkpoint, `--minimum-elements` to leave small
matrices dense, and `--shard-size-mib` to control peak buffered output and
shard size. The output directory must not already exist.

Library callers can use `quantization::quantize_checkpoint` for conversion,
the shared `ModelLoadOptions` APIs for architecture dispatch, or
`weights::load_safetensors_dir_quantized_strict` to populate a model that
exposes the standard packed parameter tree. Model-specific
`load_*_model_quantized` helpers remain available. All modes call
`quantization::quantize_tensor` with a caller-owned explicit stream, so saving
and direct loading use the same numerical transform.
Direct loading materializes each packed weight/scale/bias triple before reading
the next dense tensor. This prevents MLX's lazy graphs from retaining the whole
dense checkpoint during conversion while preserving exact parity with a saved
quantized checkpoint.

To include direct Q4 conversion in a PersonaPlex load/step benchmark, use the
dense checkpoint with `--quantize-on-load`:

```sh
cargo run --release -p safemlx-lm --example personaplex_step_bench -- \
  /path/to/personaplex-dense 64 --quantize-on-load
```

Generate a deterministic fixture with the upstream `moshi_mlx` package, then
replay it through Rust:

```sh
python safemlx-lm/scripts/moshi_mlx_token_fixture.py \
  /path/to/moshika-mlx-bf16 /tmp/moshi-token-parity.safetensors \
  --require-mlx-version 0.32.0

cargo run -p safemlx-lm --release --example moshi_token_parity -- \
  /path/to/moshika-mlx-bf16 /tmp/moshi-token-parity.safetensors
```

Use the MLX version pinned by `safemlx-sys/src/mlx-c/CMakeLists.txt` when
generating a reference fixture. The version guard prevents comparisons across
different MLX kernel implementations.

The comparator uses standard relative and absolute closeness checks and
defaults to `rtol=0.02` and `atol=0.02`, suitable for BF16 cached inference.
It reports the largest absolute difference observed. Pass explicit tolerances
as the third and fourth arguments.

The fixture contains delayed temporal inputs, teacher-forced depth inputs, the
normalized temporal states, text logits, logits from every depth slice, and an
end-to-end greedy encoded-audio generation sequence. By default the exporter
creates deterministic synthetic tokens; pass `--inputs` with a safetensors file
containing `input.text`, `input.audio`, and `input.depth` to replay a prerecorded
Mimi-token sequence for the teacher-forced portion.

For a lightweight end-to-end check without downloading released weights, add
`--create-tiny --steps 6`. This creates a deterministic miniature BF16
checkpoint in the supplied model directory before exporting its reference
fixture.

Moshi projections preserve their checkpoint dtype. MLX 0.32.0 fixes the
locally built NAX metallib behavior that previously required FP32 promotion
with MLX 0.31.2.

## License

Licensed under either Apache-2.0 or MIT.
