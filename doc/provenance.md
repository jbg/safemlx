# Provenance

SafeMLX combines code from several upstream projects with continued development
in this repository. This page records the main lineage; Git history and the
notices inside vendored source trees remain authoritative for individual
changes and files.

## Core Rust bindings

The `safemlx`, `safemlx-sys`, `safemlx-macros`,
`safemlx-internal-macros`, and `safemlx-tests` crates were imported from
[`oxiglade/mlx-rs`](https://github.com/oxiglade/mlx-rs) at commit
`f4aa309c79b6be35255ca7d34157dfc10d9ed4c9`. The upstream package authors were
Minghua Wu and David Chavez.

The crates were renamed to the `safemlx` package namespace and have since
received substantial API, platform, I/O, concurrency, and runtime changes in
this repository.

## Language-model crates

`safemlx-lm` and `safemlx-lm-utils` are derived from the `mlx-lm` and
`mlx-lm-utils` crates in [`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs).
That work was introduced in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281) and merged
as commit `7c667cb7`. The original implementation and authorship belong to the
contributors to that project; subsequent model and runtime work is recorded in
this repository's history.

## MLX C API

The source under `safemlx-sys/src/mlx-c` originated in Apple's
[`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c) project and is
vendored so the Rust crate can build a compatible native library. The current
native build pins MLX v0.32.0. Local changes to the C surface and build system
are visible in this repository's history.

## GGUF conversion

The pure-Rust conversion code in `safemlx-gguf` is a translation of Apple MLX
v0.32.0's `mlx/io/gguf_quants.cpp` together with earlier SafeMLX K-quant
patches. Differential fixtures and their sources are documented in
[`safemlx-gguf/tests/fixtures/README.md`](../safemlx-gguf/tests/fixtures/README.md).

## Licensing

The workspace-level license is `MIT OR Apache-2.0`. Individual crate manifests
and vendored trees can be more specific: notably, `safemlx-sys` declares MIT,
and the vendored MLX C source carries its upstream notices. Consult the license
files and component metadata when redistributing a subset of the workspace.
