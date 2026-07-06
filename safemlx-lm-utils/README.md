# safemlx-lm-utils

`safemlx-lm-utils` contains utility code for MLX language model runtimes.

The crate is derived from the `mlx-lm-utils` crate in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281), merged as
commit `7c667cb7`.

The original implementation and authorship belong to the `oxideai/mlx-rs`
contributors.

This fork adds chat-template support including structured JSON messages, system
roles, and tool metadata passed into Jinja templates.

## Usage

```toml
[dependencies]
safemlx-lm-utils = "0.1"
```

## License

Licensed under either Apache-2.0 or MIT.
