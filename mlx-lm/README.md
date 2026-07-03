# safemlx-lm

`safemlx-lm` is a Rust runtime for MLX language models used by Goose.

The crate is derived from the `mlx-lm` crate in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281), merged as
commit `7c667cb7`.

The original implementation and authorship belong to the `oxideai/mlx-rs`
contributors. This fork carries additional changes used by Goose.

This fork adds model/runtime support needed by Goose, including Gemma 4 loading,
Gemma 4 assistant drafting, expanded model dispatch, and related generation
utilities.

## Usage

```toml
[dependencies]
safemlx-lm = "0.1"
```

## License

Licensed under either Apache-2.0 or MIT.
