# safemlx-lm CLI example

This workspace crate provides a small, script-friendly text-generation binary
using `safemlx-lm`. A model can be a local Hugging Face-style directory, a GGUF
file, or a Hugging Face identifier already present in the local cache.

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model mlx-community/Qwen3-0.6B-4bit \
  "Write a Rust function that adds two integers."
```

On a Linux system with the CUDA prerequisites installed, add the workspace
feature to build and run the same CLI on MLX's CUDA backend:

```sh
cargo run --release -p safemlx-lm-cli --features cuda -- \
  --model /path/to/model "Write a Rust function that adds two integers."
```

The Hugging Face form never downloads files. It scans the cache selected by
`HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, or `HF_HOME`. Model directories use
the cached `main` revision by default. GGUF selectors prefer `main`, then search
other cached snapshots when `main` does not contain the requested
quantization. Use `--revision` to limit selection to a cached ref or commit.

For a cached repository containing multiple GGUF files, append a
case-insensitive quantization selector to the model identifier. The full
quantization name and the llama.cpp-style alias are both accepted; for example,
`UD-Q4_K_M` can also be selected with `Q4_K_M` when no exact `Q4_K_M` file is
cached:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
  "Explain imaginary numbers."
```

Selection is limited to files already present in the cache.
For sharded GGUF checkpoints, the CLI resolves the first canonical shard and
the loader discovers the remaining shards.

Local model paths use the same interface:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  --temperature 0.7 --top-p 0.9 --max-tokens 512 \
  "Tell me a short story."
```

Mirostat V2 adaptively targets a desired surprise instead of using fixed
top-k, top-p, and min-p cutoffs. Set a nonzero temperature and optionally tune
the target (`tau`, in bits) and adaptation rate (`eta`):

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  --mirostat-v2 --temperature 1.0 \
  --mirostat-tau 5.0 --mirostat-eta 0.1 \
  "Tell me a strange but coherent short story."
```

Repetition, frequency, and presence penalties are applied before Mirostat
truncation. Mirostat V2 supports embedded and external MTP; its adaptive state
advances only for tokens committed from the target distribution.

Gemma 4 can use an explicit external assistant through the generalized MTP
engine. The target may be fully resident or use `--layerwise-host`; the
assistant is loaded independently and remains fully resident:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/gemma4 \
  --draft-model /path/to/gemma4-assistant \
  --mtp-draft-tokens 3 --temperature 0.7 \
  "Explain speculative decoding."
```

The assistant may be a safetensors directory or a GGUF file with
`general.architecture = "gemma4_assistant"` or the published
`"gemma4-assistant"` spelling. GGUF config is read from a
`safemlx.mtp.config` JSON metadata string or a sibling `config.json`.
Cached target and assistant GGUFs can use independent quantization selectors,
even when individual downloads placed them in separate commit snapshots. The
CLI distinguishes standardized `mtp-` sidecar filenames from target-model
GGUFs, so target and assistant files may use the same quantization:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M \
  --draft-model unsloth/gemma-4-26B-A4B-it-GGUF:Q8_0 \
  --mtp-draft-tokens 3 \
  "Explain speculative decoding."
```

Stochastic MTP uses lossless probability-ratio acceptance and supports the
same top-k, top-p, min-p, and repetition/frequency/presence policies as normal
generation. Under `--verbose`, the CLI reports proposal and acceptance counts.

Qwen3-Next and Qwen3.5/3.6 safetensors checkpoints with native MTP weights use
those embedded weights automatically; no `--draft-model` is needed. Their
native head proposes one token per verification round, so larger
`--mtp-draft-tokens` values are safely capped by the model.

Dense checkpoints can be quantized while loading. For example, 4-bit affine
weights substantially reduce decode-time weight traffic and memory use:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model LiquidAI/LFM2.5-1.2B-Instruct \
  --quantize 4 \
  "Explain MLX in one paragraph."
```

The default quantization group size is 64 weights; change it with
`--quantization-group-size`. Load-time quantization is performed on every run,
so use a checkpoint already carrying matching quantization metadata when
startup time is important.

For a SafeTensors or registered GGUF family, select a bounded device window
through the same architecture-detecting loader:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model --layerwise-host \
  --device-layer-window 1 --mapped-shards 4 \
  --host-budget-bytes 24000000000 --device-budget-bytes 8000000000 \
  "Summarize bounded weight residency."
```

`--verbose` also prints logical current/peak host and device parameter bytes,
synchronous transfer counts, and backend-tagged shard/reader diagnostics. Apple
CPU and GPU tiers share unified physical memory, so these logical tiers do not
increase total capacity. Load-time quantization and KV cache offload are not
supported by this path; checkpoint-native GGUF quantization is.

Stream dense layers from either backend with finite tier controls:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model.gguf --dense-disk-stream \
  --device-budget-bytes 8000000000 --host-budget-bytes 16000000000 \
  --dense-host-lookahead 2 --dense-device-lookahead 1 \
  --dense-background-queue 2 "Explain bounded GGUF loading."
```

Supported MoE models can cache routed experts separately. This
includes DeepSeek-V3/R1, GPT-OSS, Inkling, LFM2, Nemotron-H, Qwen3,
Qwen3-Next, Qwen3-VL-MoE, and Qwen3.5-MoE:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/sparse-model --expert-cache \
  --device-layer-window 1 --mapped-shards 4 \
  --expert-cache-device-budget-bytes 8000000000 \
  --expert-cache-host-budget-bytes 16000000000 \
  --expert-cache-scratch-bytes 2000000000 \
  --expert-cache-eviction lfu \
  "Explain sparse expert residency."
```

For GGUF, sparse expert caching is available for DeepSeek2, LFM2-MoE,
Nemotron-H-MoE, Qwen3-MoE, Qwen3.5-MoE, and MoE Qwen3-Next. Other GGUF
families still support the host-layerwise and dense-stream modes listed above.

The ordinary device and host budgets govern non-expert layerwise weights; the
`--expert-cache-*` budgets govern hot and warm expert copies. A zero expert host
budget promotes misses directly from checkpoint storage. The scratch limit is
checked against each temporary compact bank and is separate from the device
cache budget. `--verbose` reports prefill and decode requests, hits, misses,
evictions, compact-bank bytes, and current expert occupancy separately.
Combine `--expert-cache` with `--dense-disk-stream` to stream non-expert units
while keeping expert-granular reuse.

Add `--expert-cache-benchmark` to run the real prompt through a cold prefill,
a repeated prefill with fresh attention state, and one decode using the repeated
prefill's state before normal generation begins. Each measurement reports its
own latency, route and coalescing counts, host/device hits, misses and evictions,
compact-bank bytes, and ending cache occupancy. The measurements are diagnostic
samples rather than performance guarantees; filesystem caching and routing
locality can substantially change later runs.

Route inspection and transfers are synchronous. Unified memory does not create
additional physical capacity, and useful disk-backed performance depends on
expert-routing locality. Mapped-shard and logical-transfer counters do not
measure exact physical disk I/O. Checkpoint-native packed formats are preserved;
load-time conversion and unsupported model families fail explicitly.

When the positional prompt is omitted, the binary reads it from stdin. Generated
text is decoded and flushed to stdout incrementally, including when MTP is in
use. When generation reaches `--max-tokens`, `stop_reason: max_tokens` is written
to stderr even without `--verbose`; normal EOS termination stays silent.
This keeps stdout convenient to pipe or capture. `--verbose` reports every stop
reason (`eos`, `max_tokens`, or `generator_exhausted`) and writes model details,
separate load and generation times, time to first token, decode-only and overall
generated-token rates, total execution time, and MLX peak/current/cache
unified-memory statistics to stderr. It also prints explicit
diagnostics/content section markers so the two streams remain visually distinct
in a terminal. Generation time includes prompt prefill, and `token_rate` is
generated tokens divided by that generation time. `decode_token_rate` excludes
time to first token and the first generated token. The memory values cover
allocations managed by MLX, not total process resident memory or memory-mapped
files.

```sh
printf 'Summarize the purpose of MLX.' | \
  cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model > response.txt
```

Chat templates are applied automatically when supplied by the model. Pass
`--thinking on` or `--thinking off` to override thinking/reasoning when the
template exposes the standard `enable_thinking` switch:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model mlx-community/Qwen3-0.6B-4bit \
  --thinking off \
  "Answer in one sentence."
```

The default, `--thinking auto`, leaves the template's model-specific behavior
unchanged. An explicit mode fails with a clear error when the template does not
support the switch. Pass `--raw` to tokenize the prompt directly; raw prompts
cannot use an explicit thinking mode. Run with `--help` for all sampling and
repetition-penalty options.
