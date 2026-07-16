# safemlx-gguf

`safemlx-gguf` is safemlx's framework-independent, pure-Rust GGUF backend. It
parses GGUF v1-v3 in either byte order, exposes typed metadata and validated
tensor descriptors, reads one payload at a time, converts supported GGML blocks
to MLX's affine representation, and writes deterministic seekable GGUF files.

Dense support: F32, F16, BF16, F64, I8, I16, I32, and I64. Affine support:
Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K. Legacy Q5_0 and Q5_1
are decoded to F16 to preserve the previous patched-MLX behavior. The writer
accepts the same dense types and preserves canonical raw bytes for every listed
quantized type; it does not invent a lossy inverse from MLX affine triples.

`Limits` bounds metadata counts, string/array sizes, tensor counts and ranks,
nesting depth, and per-tensor allocation. Parsing uses checked arithmetic and
rejects duplicate names, invalid alignment, impossible block shapes, truncated
or out-of-range data, and overlapping tensor ranges.

The conversion code is a Rust translation of Apple MLX v0.32.0's
`mlx/io/gguf_quants.cpp` and the former safemlx K-quant patches (MIT licensed).
See `tests/fixtures/README.md` for the differential oracle provenance.
