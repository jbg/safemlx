# Model and checkpoint support

This page summarizes the high-level support implemented by `safemlx-lm`. For
API details and the full layerwise-residency matrix, see the
[`safemlx-lm` README](../safemlx-lm/README.md).

Support is determined from checkpoint metadata and validated configuration,
not from a model's display name. Applications can inspect a SafeTensors model
directory before loading it with `check_model_dir`, `check_model_config`, or
`check_model_config_json`.

## SafeTensors model directories

The standard loader accepts Hugging Face-style directories containing
`config.json`, tokenizer files, and either a single or sharded SafeTensors
checkpoint. The current architecture dispatch covers:

- DeepSeek-V3 and DeepSeek-R1
- Gemma 4 dense and MoE text and multimodal models
- GPT-OSS
- Thinking Machines Lab Inkling
- Llama and Mistral
- LFM2 and LFM2.5, including MoE variants
- Nemotron-H
- Qwen3, including MoE variants
- Qwen3-Next
- Qwen3-VL and Qwen3-VL-MoE
- Qwen3.5 dense and MoE models

Moshi and PersonaPlex are exposed through the separate realtime
speech-to-speech token API. That API operates on encoded audio tokens; codec
encoding/decoding is provided separately by `safemlx-codec`, and audio device
I/O remains the application's responsibility.

Image preprocessing requires the `safemlx-lm/image-processing` feature. Audio
preprocessing requires `safemlx-lm/audio-processing`. These features are not
enabled by default.

## GGUF

The high-level loader accepts a GGUF file for these `general.architecture`
values:

- `deepseek2`
- `gemma4`
- `llama` and `mistral`
- `lfm2` and `lfm2moe`
- `nemotron_h` and `nemotron_h_moe`
- `qwen3` and `qwen3moe`
- `qwen3next`
- `qwen3vl` (with its companion vision projection checkpoint)
- `qwen35` and `qwen35moe`

The tokenizer and chat template are reconstructed from GGUF metadata when
possible. A sibling `tokenizer.json` can supply a tokenizer that is absent from
the file or uses an unsupported embedded tokenizer model.

`safemlx-gguf` parses GGUF v1-v3 in either byte order and validates all shard
headers before payload materialization. Its supported dense and quantized
tensor encodings are listed in the
[`safemlx-gguf` README](../safemlx-gguf/README.md).

## Weight loading and residency

Fully resident loading is the default. SafeTensors and registered GGUF families
can also use host-backed layer windows or experimental dense disk streaming.
GGUF bounded loading covers DeepSeek2, Gemma 4, Llama/Mistral, LFM2,
Nemotron-H, Qwen3, dense Qwen3-VL with its mmproj, Qwen3.5, and Qwen3-Next.
Supported MoE families can cache routed experts independently; for GGUF these
are DeepSeek2, LFM2-MoE, Nemotron-H-MoE, Qwen3-MoE, Qwen3.5-MoE, and MoE
Qwen3-Next.

Qwen3-Next supports the official native fine-grained E4M3 checkpoint format
(`fp8`, dynamic activations, 128 x 128 weight blocks) with fully resident,
layerwise, sparse expert-cache, and pure expert-parallel loading. Fused QKVZ
weights and inverse scales are split without dequantization, dense BF16 BA is
preserved, and routed expert weights remain checkpoint-backed at expert
granularity for sparse-cache and expert-parallel execution.

Important boundaries:

- GGUF remains fully resident by default. `LayerwiseHost`, `DenseDiskStream`,
  and supported sparse-expert policies use header-only logical catalogs and
  bounded payload materialization.
- Load-time quantization is incompatible with streamed or sparse-cache loading;
  use a checkpoint-native packed format for those policies.
- Transfers and route inspection are synchronous because the pinned MLX C API
  does not expose the events or fences required for safe cross-stream overlap.
- On Apple silicon, reported host and device residency are logical tiers over
  the same physical unified memory. They do not create additional capacity.
- Parameter budgets do not include activations, KV or recurrent state, kernels,
  allocator caches, checkpoint mappings, or every temporary buffer.
- SafeTensors mapping and logical-transfer counters cannot report exact
  physical disk I/O. GGUF additionally reports physical payload read requests
  and bytes issued by its selected-read backend;
  operating-system page caching materially affects disk-backed performance.

The example CLI exposes the common loading policies and their diagnostics. See
its [usage guide](../examples/safemlx-lm-cli/README.md) for concrete commands.

## Parallel execution

The language-model crate contains explicit APIs for pure tensor, pipeline, and
expert parallelism. A non-replicated topology must be loaded through the
matching API; the ordinary complete-model loader rejects it. Hybrid tensor +
pipeline, tensor + expert, and pipeline + expert topologies are not currently
supported.
