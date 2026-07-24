# safemlx-gguf

`safemlx-gguf` is safemlx's framework-independent, pure-Rust GGUF backend. It
parses GGUF v1-v3 in either byte order, exposes typed metadata and validated
tensor descriptors, reads one payload at a time, converts affine-compatible
GGML blocks, retains nonlinear IQ blocks explicitly, and writes
deterministic seekable GGUF files.
`Checkpoint` discovers canonical shards, validates the complete descriptor
set, exposes converted logical tensor layouts without reading payload bytes, and
materializes one dense tensor, atomic affine group, or packed IQ tensor at a time.
For bounded out-of-order access, `Checkpoint::materializer` indexes physical
names once and reuses the current shard reader across named requests.

```rust,no_run
use safemlx_gguf::Checkpoint;

let checkpoint = Checkpoint::open("model-00001-of-00004.gguf")?;
checkpoint.for_each_converted_tensor(|tensor| {
    println!("{}", tensor.descriptor().name);
    Ok(())
})?;
# Ok::<(), safemlx_gguf::Error>(())
```

Dense support: F32, F16, BF16, F64, I8, I16, I32, and I64. Affine support:
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K. Q5_0 and
Q5_1 are repacked directly into MLX's 5-bit, group-size-32 affine layout. The
writer accepts the same dense types and preserves canonical raw bytes for every
listed quantized type; it does not invent a lossy inverse from MLX affine
triples.

All nine canonical GGML IQ tensor encodings at upstream llama.cpp commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9` are supported:

| Tensor encoding | Code | Values/block | Bytes/block |
| --- | ---: | ---: | ---: |
| IQ2_XXS | 16 | 256 | 66 |
| IQ2_XS | 17 | 256 | 74 |
| IQ3_XXS | 18 | 256 | 98 |
| IQ1_S | 19 | 256 | 50 |
| IQ4_NL | 20 | 32 | 18 |
| IQ3_S | 21 | 256 | 110 |
| IQ2_S | 22 | 256 | 82 |
| IQ4_XS | 23 | 256 | 136 |
| IQ1_M | 29 | 256 | 56 |

IQ codebooks and scale layouts are nonlinear, so these tensors are not exposed
as affine weights/scales/biases. Conversion returns an explicit packed IQ value
containing the original blocks, byte order, logical shape, and tensor type.
IQ-aware runtimes can execute those bytes directly; the canonical scalar
decoder remains available for differential testing and non-accelerated
fallbacks. Raw reads and writes retain the original payload bytes exactly.

`IQ2_M`, `IQ3_M`, Unsloth `UD-*`, and similar names are mixed-precision file
recipes, not additional GGML tensor encodings. Such files are compatible when
their individual tensors use the encodings listed above (and the existing dense
or K-quant encodings). In particular, dynamic Q2_K/Q3_K recipes that select
IQ4_NL for some tensors are supported.

Codes 36, 37, and 38 (`IQ4_NL_4_4`, `IQ4_NL_4_8`, and `IQ4_NL_8_8`) are known
by `GgmlType` for diagnostics, but current upstream explicitly marks them as
removed runtime-repacking layouts with zero block/type sizes. They were never
canonical GGUF tensor encodings and are therefore rejected rather than guessed.

`Limits` bounds metadata counts, string/array sizes, tensor counts and ranks,
nesting depth, and per-tensor allocation. Parsing uses checked arithmetic and
rejects duplicate names, invalid alignment, impossible block shapes, truncated
or out-of-range data, and overlapping tensor ranges.

The affine conversion code is a Rust translation of Apple MLX v0.32.0's
`mlx/io/gguf_quants.cpp` and the former safemlx K-quant patches (MIT licensed).
The IQ decoder and codebooks are safe Rust translations of the pinned
llama.cpp scalar implementation. See `tests/fixtures/README.md` for both
differential oracle provenances.
