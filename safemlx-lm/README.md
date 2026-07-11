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
safemlx-lm = { version = "0.3", features = ["image-processing"] }
```

For Gemma 4 or Qwen image prompts, render the chat template with one image
placeholder per image, then prepare decoded RGB8 pixels before prefill:

```rust,ignore
use safemlx_lm::processor::{MediaInput, RgbImageView};

let image = RgbImageView::packed(rgb_pixels, width, height)?;
let prepared = model.prepare_input(
    &rendered_prompt,
    &[MediaInput::image_rgb8(image)],
    false,
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
    &rendered_video_prompt,
    &[MediaInput::video_rgb8(&frames, Some(source_fps))],
    false,
)?;
```

The optional `image-processing` feature enables architecture-dispatched Gemma 4
and Qwen processors. Shared code owns decoded-image validation and resizing;
each processor adds its model-native patch packing, metadata, and placeholder
binding. Qwen also supports decoded video sampling and timestamps. Without the
feature, callers can still supply Gemma 4 or Qwen `Image/Tensor` inputs, and
Qwen `Video/Tensor` inputs, directly without depending on the `image` crate.

## License

Licensed under either Apache-2.0 or MIT.
