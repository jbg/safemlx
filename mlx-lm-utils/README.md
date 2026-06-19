# goose-mlx-lm-utils

`goose-mlx-lm-utils` contains Goose-maintained utility code for MLX language
model runtimes.

The crate is derived from the `mlx-lm-utils` crate in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281), merged as
commit `7c667cb7`.

The original implementation and authorship belong to the `oxideai/mlx-rs`
contributors. Goose contributors maintain this fork and its additional changes.

This fork adds chat-template support needed by Goose, including structured JSON
messages, system roles, and tool metadata passed into Jinja templates.

## Usage

```toml
[dependencies]
goose-mlx-lm-utils = "0.1"
```

## License

Licensed under either Apache-2.0 or MIT.
