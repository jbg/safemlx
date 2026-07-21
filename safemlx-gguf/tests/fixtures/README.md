# Patched MLX oracle

`mlx-v0.32.0.oracle` was captured before removing native GGUF support by running:

```sh
cargo run -q -p safemlx --example capture_gguf_oracle --no-default-features
```

at the pre-removal revision (parent of this implementation). The capture tool
writes deterministic two-block GGUF tensors through the Rust container writer,
then loads them through MLX v0.32.0 with safemlx's five GGUF patches still
applied. Each line records the GGML type code, raw block bytes, output names,
shapes, MLX dtypes, packed u32 weights, f16 scales and biases, and native MLX
dequantized f16 values. Inputs mix signs, nibble/bit boundaries, irregular scale
bytes, positive and negative half scales, and consecutive blocks. Separate
tests cover all-zero and maximum-code blocks.

The example remains as regeneration documentation; it must be run at a revision
where GGUF loading still used patched MLX, such as commit `4e53c5ec`.
