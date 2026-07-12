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

## Usage

```toml
[dependencies]
safemlx-lm = { version = "0.4", features = ["image-processing"] }
```

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

## Moshi token parity

The `models::moshi` module implements Moshi's temporal and depth language
models over pre-tokenized Mimi streams. This first milestone does not include
Mimi audio encoding/decoding or realtime audio I/O, and currently loads
unquantized MLX checkpoints (`model.safetensors`). For the original released
Moshika/Moshiko repositories, the loader uses Moshi's built-in v0.1 config when
the model directory has no `config.json`.

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
normalized temporal states, text logits, and logits from every depth slice. By
default the exporter creates deterministic synthetic tokens; pass `--inputs`
with a safetensors file containing `input.text`, `input.audio`, and `input.depth`
to replay a prerecorded Mimi-token sequence.

For a lightweight end-to-end check without downloading released weights, add
`--create-tiny --steps 6`. This creates a deterministic miniature BF16
checkpoint in the supplied model directory before exporting its reference
fixture.

Moshi projections preserve their checkpoint dtype. MLX 0.32.0 fixes the
locally built NAX metallib behavior that previously required FP32 promotion
with MLX 0.31.2.

## License

Licensed under either Apache-2.0 or MIT.
