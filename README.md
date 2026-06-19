# Goose MLX LM

This repository contains Goose-maintained Rust crates for running language models
with Apple's MLX framework:

- `goose-mlx-lm`
- `goose-mlx-lm-utils`

These crates are derived from the `mlx-lm` and `mlx-lm-utils` crates in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs). The original crates were
introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281), merged as
commit `7c667cb7`.

The original implementation and authorship belong to the `oxideai/mlx-rs`
contributors. Goose contributors maintain this fork and its additional changes.

This fork carries additional model/runtime support used by Goose, including
Gemma 4 support, Gemma 4 assistant drafting, expanded model loading, and
chat-template handling for structured messages and tools.

## Crates

The crates use Goose-specific package names on crates.io to avoid confusion with
the upstream `mlx-lm` packages:

```toml
goose-mlx-lm = "0.1"
goose-mlx-lm-utils = "0.1"
```

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
