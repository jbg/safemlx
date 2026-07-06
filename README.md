# SafeMLX LM

This repository contains Rust crates for running language models with Apple's
MLX framework:

- `safemlx`
- `safemlx-sys`
- `safemlx-tests`
- `safemlx-lm`
- `safemlx-lm-utils`

This fork carries additional model/runtime support, including Gemma 4 support,
Gemma 4 assistant drafting, expanded model loading, and chat-template handling
for structured messages and tools.

## Crates

The crates use SafeMLX package names on crates.io to avoid confusion with the
upstream `mlx-lm` packages:

```toml
safemlx = "0.1"
safemlx-sys = "0.1"
safemlx-lm = "0.1"
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
[`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c) submodule at commit
`fba4470b89073180056c9ea46c443051375f7399`.

The `safemlx-lm` and `safemlx-lm-utils` crates are derived from the `mlx-lm`
and `mlx-lm-utils` crates in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281) and merged
as commit `7c667cb7`. The original implementation and authorship belong to the
`oxideai/mlx-rs` contributors.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
