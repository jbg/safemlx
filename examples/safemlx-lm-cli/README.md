# safemlx-lm CLI example

This workspace crate provides a small, script-friendly text-generation binary
using `safemlx-lm`. A model can be a local Hugging Face-style directory, a GGUF
file, or a Hugging Face identifier already present in the local cache.

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model mlx-community/Qwen3-0.6B-4bit \
  "Write a Rust function that adds two integers."
```

The Hugging Face form never downloads files. It scans the cache selected by
`HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, or `HF_HOME` and uses the cached
`main` revision. Use `--revision` to select another cached ref or commit.

Local model paths use the same interface:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  --temperature 0.7 --top-p 0.9 --max-tokens 512 \
  "Tell me a short story."
```

When the positional prompt is omitted, the binary reads it from stdin. Only
the generated text is written to stdout, making it convenient to pipe or
capture; `--verbose` writes model details, separate load and generation times,
time to first token, overall generated-token rate, total execution time, and
MLX peak/current/cache unified-memory statistics to stderr. Generation time
includes prompt prefill, and token rate is generated tokens divided by that
generation time. The memory values cover allocations managed by MLX, not total
process resident memory or memory-mapped files.

```sh
printf 'Summarize the purpose of MLX.' | \
  cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model > response.txt
```

Chat templates are applied automatically when supplied by the model. Pass
`--raw` to tokenize the prompt directly. Run with `--help` for all sampling
and repetition-penalty options.
