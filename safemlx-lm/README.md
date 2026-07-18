# safemlx-lm

`safemlx-lm` is a Rust runtime for MLX language models.

The crate is derived from the `mlx-lm` crate in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281), merged as
commit `7c667cb7`.

The original implementation and authorship belong to the `oxideai/mlx-rs`
contributors.

This fork adds model/runtime support including Gemma 4 loading, Gemma 4
assistant drafting, expanded model dispatch, and related generation utilities.

## Offload planning and observability

Distributed placement decides which tensors a rank owns. Residency is a
separate concern that decides where an owned logical unit lives and for how
long. The public `safemlx_lm::offload` module provides architecture-independent
configuration, explicit deterministic plans, tier byte totals, and reusable
telemetry for a future residency manager.

These APIs do **not** move, load, prefetch, or evict model weights and do not
add CPU or disk offload support to model loading. Existing checkpoint and
model-loading behavior is unchanged.

On Apple silicon, CPU and GPU execution share the same physical unified-memory
pool. Choosing CPU execution does not increase total model capacity, although
it can change execution behavior, wired memory, and residency pressure. On a
discrete CUDA system, a future implementation may use host-resident weights to
extend effective capacity beyond VRAM, with transfer costs. Disk streaming is a
not implemented; dense autoregressive decode would repeatedly wait on storage
and is therefore expected to be slow rather than a general high-performance
path.

The `safemlx::memory` controls affect process-global MLX-managed allocations.
They do not directly constrain process RSS, checkpoint mappings, or unrelated
native allocations. The pinned MLX 0.32.0 C surface has whole-stream
synchronization but no event/fence primitive, so initial residency execution
will require conservative stream synchronization until an event-backed API is
available.

## Linux and CUDA

Enable the `cuda` feature to propagate MLX CUDA support through this crate:

```toml
safemlx-lm = { version = "0.4", features = ["cuda"] }
```

Most model code uses backend-neutral MLX operations. Qwen3.5 MoE's custom
Metal FP8 and recurrent kernels use portable MLX operation fallbacks on CUDA;
these prioritize correctness and can be slower or use more temporary memory
than the Metal-specialized paths.

## GGUF models

The standard `models::load_model` and `models::LoadedModel::load` entry points
accept Hugging Face-style model directories for Gemma 4, GPT-OSS, Inkling, Llama, dense Mistral,
dense LFM2/LFM2.5 and LFM2-MoE, dense and sparse-MoE Nemotron-H, Qwen3,
Qwen3-Next, Qwen3-VL, Qwen3-VL-MoE, and dense or MoE Qwen3.5. They also accept the
GGUF architectures listed below. Canonically named sharded GGUF checkpoints
are supported by passing the first
`-00001-of-NNNNN.gguf` shard; the remaining shards are discovered and
validated automatically. Put `tokenizer.json` next to a GGUF file when using
`LoadedModel` or
`load_tokenizer`; adjacent
`tokenizer_config.json` and `chat_template.jinja` files are used when present.

```rust,ignore
use safemlx_lm::models::LoadedModel;

let model = LoadedModel::load(
    "/path/to/model-00001-of-00004.gguf",
    execution_stream,
    cpu_weights_stream,
)?;
```

Dense GGUF tensors are loaded directly. MLX-native packed loading is enabled
for Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0, including checkpoints
that mix packed and dense matrices. Q4_K and Q5_K are losslessly repacked to
MLX's 32-value affine groups, while Q2_K, Q3_K, and Q6_K map exactly to
16-value affine groups. Group-16 K-quants use tiled quantized matrix kernels for
prefill and the corresponding vector kernels for decode. These formats execute
without expanding matrix weights to float16.
Q5_0 and Q5_1 tensors are converted to float16 while loading; other GGUF
quantization types use MLX's bundled converter when
supported, and unsupported tensor types return an error. Model dispatch uses
`general.architecture`; the current GGUF adapters support text-only `gemma4`,
`llama`, `mistral`, `lfm2`, `lfm2moe`, `nemotron_h`, `nemotron_h_moe`, `qwen3`, `qwen3moe`, dense
`qwen35`, and `qwen35moe` architectures, plus dense `qwen3vl` with its separate
vision projector. For Qwen3-VL, put the llama.cpp-style dense F16/BF16/F32
`mmproj-*.gguf` next to the language-model GGUF. The single-path loaders prefer
the unique dense projector automatically; callers that need an explicit pair
can use `models::qwen3_vl::load_qwen3_vl_gguf`.
Nemotron-H routed expert banks retain Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 packed weights
and execute through selected-expert quantized matrix multiplication. Qwen3 MoE
uses the same packed expert-major execution with per-tensor mixed Q2/Q3/Q4/Q5/Q6/Q8
settings. Dense Qwen3.5 uses the hybrid linear/full-attention runtime with
conventional SwiGLU layers; Qwen3.5 MoE keeps its
Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 routed expert banks packed while loading mixed
quantization types. Gemma 4
multimodal projectors, MoE, and assistant-drafter files are separate formats
and are not handled by the initial Gemma 4 adapter. Nemotron-H latent-space MoE and
Omni/multimodal checkpoints remain separate formats. Quantized Qwen3-VL language
GGUFs retain their supported packed affine weights while the vision projector
remains dense; quantized Qwen3-VL projectors and Qwen3.5-VL GGUF files are not
currently handled.

## Usage

```toml
[dependencies]
safemlx-lm = { version = "0.4", features = ["image-processing"] }
```

### Executable pipeline parallelism

Runtime parallel topology is configured independently of a model's
`config.json`. `ParallelTopology` uses pipeline-major, tensor, then expert rank
ordering (expert is the fastest-changing coordinate). The process-local device
index is always explicit: a global rank identifies a process in the distributed
group and must not be reused as a local GPU index.

```rust,ignore
use safemlx::{distributed::{self, Backend}, DeviceType, Stream};
use safemlx_lm::{
    pipeline::{load_pipeline_model_with_options, PipelineStep},
    DeviceAssignment, ModelLoadOptions, ParallelTopology,
};

let group = distributed::init(true, Backend::Ring)?;
let topology = ParallelTopology::from_group(
    &group,
    1, // tensor-parallel size
    2, // pipeline-parallel size
    1, // expert-parallel size
    DeviceAssignment::new(DeviceType::Gpu, local_device_index),
)?;
let stream = Stream::new_with_device(&topology.device.device()?);

let options = ModelLoadOptions::default().with_parallel_topology(topology);
let mut model = load_pipeline_model_with_options(
    model_dir,
    options,
    &stream,
    cpu_weights_stream,
)?;
let mut cache = model.new_cache();
let step = PipelineStep::new(1, prompt_length)?;
let logits = model.forward_pipeline(
    (group.rank() == 0).then_some(&prompt_tokens),
    step,
    None,
    &mut cache,
    &group,
    &stream,
)?;
```

Pure pipeline parallelism currently requires `PP > 1`, `TP = 1`, and `EP = 1`.
Hybrid TP+PP and PP+EP jobs fail before checkpoint payloads are loaded. The
ordinary `Model` loader remains a complete single-device API and directs
non-replicated requests to the explicit pipeline loader.

Decoder layers use balanced contiguous placement from
`ParallelTopology::layer_range`. Stage zero owns token embedding and its local
layers. Intermediate stages own only their local layers and constants. The last
stage owns its local layers, final normalization, and the language-model head.
For tied Llama weights, the embedding table is present only on stage zero and
the last stage. DeepSeek routed and shared experts stay with their decoder
layer; expert banks for remote layers are filtered before packing.

Indexed safetensors placement is resolved before payload files are opened, so
remote-only shards are skipped and remote tensors never become MLX arrays.
Quantized companions remain colocated. Dense and supported prequantized
safetensors are supported for Llama-compatible models. DeepSeek supports its
official split-expert safetensors, native block-FP8 and affine layouts, and
local expert-bank packing. Requested on-load quantization is applied only to
selected local tensors. Pipeline GGUF is rejected early because the current
GGUF reader cannot guarantee local-layer bounded memory.

`PipelineCache` contains only the local global-layer range: standard or
sliding-window KV entries for Llama and compressed-latent entries for DeepSeek.
Cache reuse and reset are explicit. Every stage recreates causal mask state from
the shared `PipelineStep` and its local cache offset; explicit masks must be
supplied consistently by every rank.

Execution is correctness-first and serial: receive from the predecessor,
execute local layers, then send to the successor. Lazy point-to-point arrays are
evaluated and their stream synchronized at each boundary. Logits stay on the
last stage. `sample_and_synchronize` samples only there, then all ranks enter
the same two collectives for the small token id and EOS/stop flag. Other ranks
never mutate sampler or PRNG state and only the last rank should print text.

There is currently no microbatch overlap, so prefill and decode latency include
all stages in series. Pipeline training/backward, multimodal models, expert
token dispatch, hybrid pipeline/tensor execution, and pipeline GGUF are not
supported.

A checkpoint's DeepSeek `ep_size` remains checkpoint layout/compatibility
metadata and retains its existing validation. Runtime
`expert_parallel_size` only describes this inference job; it does not override
or reinterpret the checkpoint field.

The phase-two partition proof and executable pipeline Ring proof are opt-in:

```sh
cargo test -p safemlx-lm --test distributed_partition_ring \
  ring_two_process_partition_load -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_pipeline -- --ignored --exact --nocapture
```

See `cargo run -p safemlx-lm --example pipeline_generate -- MODEL_DIR` for the
minimal rank-aware prefill/decode probe. Launch one process per stage with the
Ring environment (`MLX_RANK` and `MLX_HOSTFILE`) configured for all processes.

### Executable tensor parallelism

Pure tensor parallelism uses `TP > 1`, `PP = 1`, and `EP = 1`. Hybrid TP+PP
and TP+EP configurations fail before checkpoint payloads are opened because
Ring and JACCL cannot reliably form the required subgroups in the vendored MLX
version. Use `tensor_parallel::load_tensor_parallel_model` with a topology
whose tensor size equals the world size.

All-to-sharded (column-parallel) projections take a complete replicated input,
shard weight/output rows, and keep the output local without communication.
Sharded-to-all (row-parallel) projections take a final-feature input shard,
compute a full-width partial result, all-sum it, and add ordinary linear bias
once after reduction. Affine quantization `biases` are per-group metadata and
stay with their packed weight shards; they are distinct from ordinary linear
bias. Row partitions must align with the affine group, the MXFP4 32-value
group, or DeepSeek's 128-by-128 block-FP8 grid.

Llama Q/K/V and gate/up outputs are sharded. RoPE, attention, SiLU, and the
gated product remain local; only attention output and MLP down projections
all-sum. Query heads, KV heads, and intermediate width must divide TP, and the
local query/KV ratio must preserve GQA. Each cache contains only local K/V
heads, including bounded sliding-window caches.

DeepSeek keeps Q-LoRA input, compressed KV latent projection/normalization,
and routing replicated. Head-expanded MLA projections use contiguous head
shards, while output projection all-sums once. Compressed-latent caches remain
rank-local and work for both prefill and absorbed decode. Routed and shared
experts retain all expert identities on each rank but shard their intermediate
dimension; their combined residual delta is all-summed once.

Embedding and output rows use balanced contiguous vocabulary ranges, including
uneven vocabulary sizes. Embedding masks out non-local ids then all-sums hidden
states. `forward_local_logits` returns the local range; `forward`, `prefill`,
and `decode` pad and gather shards along the final logits axis. Sampling occurs
only on the designated rank, which alone mutates sampler/PRNG state, and only
the selected token and stop flag are synchronized.

Rank-aware safetensors selection happens before execution-device
materialization. Indexed payload shards with no local tensors are not opened.
Dense, MLX affine/MXFP4, DeepSeek native block-FP8, official split-expert, and
local on-load quantization paths are supported subject to alignment. TP GGUF is
rejected early because its reader cannot guarantee bounded local-range loads.

Ring is useful for correctness testing. Practical low-latency TP should use
JACCL or NCCL where available. Run the collective proof with:

```sh
cargo test -p safemlx --test distributed_ring \
  ring_two_process_loopback -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_tensor_parallel -- --ignored --exact --nocapture
```

The model-level probe is:

```sh
cargo run -p safemlx-lm --example tensor_parallel_generate -- MODEL_DIR
```

Launch every rank with the same token input and configured `MLX_RANK` and
`MLX_HOSTFILE`; only rank zero prints generated tokens in the example.

Dense safetensors checkpoints and unquantized F32/F16/BF16 GGUF checkpoints can be affine- or
MXFP4-quantized while loading through the same architecture-dispatched API used for ordinary
loading:

```rust,ignore
use safemlx_lm::{
    models::{LoadedModel, ModelLoadOptions},
    quantization::{AffineQuantization, WeightQuantization},
};

let affine = ModelLoadOptions::with_quantization(WeightQuantization::Affine(
    AffineQuantization::new(64, 4)?,
));
let mxfp4 = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4);
let model = LoadedModel::load_with_options(model_dir, mxfp4, stream, weights_stream)?;
```

The realtime counterpart is `load_realtime_model_with_options`. Both APIs
recognize matching pre-quantized checkpoints and load them directly rather
than quantizing them again. A requested format that differs from existing
checkpoint metadata is an error.

### Quantized loading coverage

| Architecture | Dense | Existing quantized | Affine / MXFP4 on load | High-level dispatch | Special policy |
|---|---:|---:|---:|---:|---|
| Llama | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Linear, embedding, tied/untied head targets |
| Mistral | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Reuses the Llama-compatible dense decoder; configured sliding attention uses bounded KV caches |
| LFM2/LFM2.5 and LFM2-MoE | yes | MLX affine/MXFP4 and packed GGUF affine | yes / yes | `LoadedModel` | Alternating short-convolution/attention cache; MoE uses sigmoid top-k routing and packed expert-major SwiGLU execution |
| Qwen3 | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Linear, embedding, tied/untied head targets |
| Qwen3-VL | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Language-model targets are quantized; the vision tower remains dense |
| Qwen3-VL-MoE | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Reuses Qwen3-VL DeepStack/MRoPE and Qwen3 packed expert-major SwiGLU execution; the vision tower remains dense |
| Gemma 4 | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Currently eligible language and modality-bridge projections are quantized; specialized vision/audio components remain dense |
| Gemma 4 assistant | yes | MLX affine/MXFP4 | yes / yes | assistant loader with `ModelLoadOptions` | Transformer/projection/head targets; ordered masked-embedding heads return a capability error |
| GPT-OSS | dense attention, MXFP4 experts | checkpoint-native MXFP4 experts | no / yes | `LoadedModel` | Native experts stay unchanged; attention projections, embeddings, and LM head can be MXFP4, while the router stays dense |
| Inkling | yes | no | capability error | `LoadedModel` | Alternating local/global relative-bias attention, four short-convolution states per layer, routed plus shared experts, and native hMLP/dMel towers; MTP draft layers are skipped |
| Nemotron-H | yes | no | capability error | `LoadedModel` (dense) | Packed rank-3 routed experts require an affine grouped-matmul kernel |
| Qwen3.5/3.6-MoE | yes | block FP8, MLX affine/MXFP4 | yes / yes, from dense checkpoints | `LoadedModel` | Rank-3 expert banks are quantized row-wise and executed with routed `gather_qmm`; native FP8 checkpoints are never implicitly transcoded |
| Qwen3-Next | yes | MLX affine/MXFP4 | yes / yes, from dense checkpoints | `LoadedModel` | Reuses the hybrid Gated DeltaNet/full-attention runtime and shared-expert MoE implementation; fused checkpoint projections are split while streaming |
| Moshi | yes | MLX affine/MXFP4 | yes / yes | realtime loader | Temporal/depth projections and embeddings; no codec dependency |
| PersonaPlex | yes, transformed PyTorch layout | MLX affine/MXFP4 | yes / yes | realtime loader | Preserves per-depth checkpoint transformation; no codec dependency |

On-load selection is driven by the target module parameter tree, not by
blindly quantizing every rank-2 checkpoint tensor. Therefore specialized
convolutions, modality towers, routers, and packed expert banks stay dense only
when the architecture explicitly supports that policy, or the request is
rejected before weights are loaded.

For Gemma 4, Inkling, or Qwen image prompts, pass text and media as ordered processor
segments. Media is inserted where the segment appears; callers do not put
image/video/audio media tokens in rendered prompt text:

```rust,ignore
use safemlx_lm::processor::{MediaInput, ProcessorInput, RgbImageView};

let image = RgbImageView::packed(rgb_pixels, width, height)?;
let prepared = model.prepare_input(
    &[
        ProcessorInput::Text(prompt_before_image),
        ProcessorInput::Media(MediaInput::image_rgb8(image)),
        ProcessorInput::Text(prompt_after_image),
    ],
)?;
let logits = model.prefill_prepared_input_with_cache(
    &prepared,
    &mut cache,
    stream,
)?;
```

Decoded videos use the same processor with an ordered frame sequence and source
frame rate. Container decoding remains with the caller:

```rust,ignore
let frames = decoded_rgb_frames
    .iter()
    .map(|frame| RgbImageView::packed(frame, width, height))
    .collect::<Result<Vec<_>, _>>()?;
let prepared = model.prepare_input(
    &[
        ProcessorInput::Text(prompt_before_video),
        ProcessorInput::Media(MediaInput::video_rgb8(&frames, Some(source_fps))),
        ProcessorInput::Text(prompt_after_video),
    ],
)?;
```

The optional `image-processing` feature enables architecture-dispatched Gemma 4,
Inkling, and Qwen processors. Shared code owns decoded-image validation, frame sampling,
and timestamp operations; each processor adds its model-native patch packing,
prompt format, metadata, and ordered media insertion. Inkling divides images into
40-pixel patches and feeds its released four-layer hMLP tower. Gemma samples up to
32 frames by default and encodes each timestamped frame through its vision tower.
Qwen uses its temporal patch packing and timestamp format. Without the feature,
callers can still supply Gemma 4, Inkling, or Qwen `Image/Tensor` and `Video/Tensor`
inputs directly without depending on the `image` crate.

Gemma 4 audio accepts model-native log-mel tensors and Inkling accepts discrete
dMel IDs through the typed input API
without optional dependencies. Enable `audio-processing` to prepare mono `f32`
PCM in the shared processor instead:

```toml
[dependencies]
safemlx-lm = { version = "0.4", features = ["audio-processing"] }
```

```rust,ignore
use safemlx_lm::processor::{MediaInput, ProcessorInput};

let audio = MediaInput::audio_f32(mono_pcm, sample_rate)?;
let prepared = model.prepare_input(&[
    ProcessorInput::Text(prompt_before_audio),
    ProcessorInput::Media(audio),
    ProcessorInput::Text(prompt_after_audio),
])?;
let logits = model.prefill_prepared_input_with_cache(&prepared, &mut cache, stream)?;
```

The common audio processor validates and resamples neither channels nor sample
rate: Gemma 4 and Inkling currently require mono 16 kHz PCM. It computes each
model's log-mel features and valid-frame mask; Inkling then quantizes them to its
16-bin dMel representation. The optional FFT dependency is only enabled by
`audio-processing`; callers that provide `Audio/Tensor` and `audio_mask` directly
do not pay that dependency cost.

## Realtime encoded audio

The `realtime` module defines a codec-free API for realtime speech-to-speech
models. Models consume discrete codec-token frames and emit delay-aligned
generated codec-token frames; callers keep audio encoding, decoding, transport,
and device I/O outside `safemlx-lm`.

Use `load_realtime_model` when the model directory contains a realtime
codec-token model. It dispatches PersonaPlex, Moshi, and future realtime model
families separately from the chat/text `LoadedModel` path:

```rust,ignore
use safemlx_lm::{
    load_realtime_model,
    realtime::{RealtimeSampling, RealtimeSpeechModel, RealtimeStepInput},
    sampler::DefaultSampler,
};

let mut model = load_realtime_model(model_dir, stream, weights_stream)?;
let config = model.realtime_config();
let mut state = model.new_realtime_state();
let mut text_sampler = DefaultSampler;
let mut audio_samplers = (0..config.depth_audio_codebooks)
    .map(|_| DefaultSampler)
    .collect::<Vec<_>>();

// Your codec supplies one user/input-side frame shaped
// [batch, config.input_audio_codebooks].
let output = model.step_realtime(
    &mut state,
    RealtimeStepInput::encoded_audio(&encoded_input_frame),
    RealtimeSampling::new(&mut text_sampler, &mut audio_samplers, 0.0, 0.0, None),
    stream,
)?;

if let Some(codec_tokens) = output.output_audio_tokens {
    // Decode [batch, config.generated_audio_codebooks] with your codec.
}
```

The `models::moshi` module implements Moshi's temporal and depth language
models over pre-tokenized Mimi streams. `GenerationState` accepts one
input-side Mimi frame at a time and returns delay-aligned generated-side Mimi
frames; `generate_encoded_greedy` is the offline sequence convenience API.
Sequence tensors use Mimi's `[batch, codebooks, frames]` layout.

`models::personaplex` exposes PersonaPlex's Moshi-family realtime token API,
published 7B v1 defaults, dual-stream codebook layout, and hybrid system-prompt
helpers. It can load the released Hugging Face PyTorch-layout
`model.safetensors` directly via the shared Moshi-family PyTorch importer.

PersonaPlex consumes 8 user-side codec codebooks per realtime frame and emits 8
agent-side codec codebooks per output frame. Its depth transformer still samples
or teacher-forces 16 codebooks, so realtime sampling requires 16 audio samplers.
Prompt helpers remain token-only: use `wrap_system_prompt` before external text
tokenization, pass text ids shaped `[batch, frames]` to
`prefill_text_prompt_greedy`, and optionally pass agent voice codec tokens
shaped `[batch, 8, frames]` to `prefill_system_prompt_greedy`.

Mimi audio encoding/decoding and audio device I/O deliberately remain outside
`safemlx-lm`. The sibling `safemlx-codec` crate provides safemlx-native codec
building blocks, including Mimi checkpoint loading, PCM encode/decode,
residual-vector quantization, and stateful tokens-to-PCM decode. Audio device
I/O remains optional codec surface rather than an `safemlx-lm` dependency.

Moshi loads dense and MLX affine- or MXFP4-quantized checkpoints. For the original
released Moshika/Moshiko repositories, the loader uses Moshi's built-in v0.1
config when the model directory has no `config.json`.

## Checkpoint quantization

The generic checkpoint converter quantizes eligible two-dimensional
`*.weight` tensors one at a time, writes bounded-size safetensors shards, and
copies tokenizer and other model assets. Affine output has packed `weight`, `scales`, and
`biases`; MXFP4 output has only packed E2M1 `weight` and E8M0 `scales`. In both cases,
`config.json` contains identical `quantization` and `quantization_config`
objects.

```sh
cargo run --release -p safemlx-lm --example quantize_checkpoint -- \
  /path/to/dense-model /path/to/model-4bit \
  --group-size 64 --bits 4

cargo run --release -p safemlx-lm --example quantize_checkpoint -- \
  /path/to/dense-model /path/to/model-mxfp4 --mode mxfp4
```

Use repeatable `--include` and `--exclude` substring filters to experiment on
part of any safetensors checkpoint, `--minimum-elements` to leave small
matrices dense, and `--shard-size-mib` to control peak buffered output and
shard size. The output directory must not already exist.

The checkpoint converter accepts dense safetensors inputs. Load-time conversion also accepts
unquantized F32, F16, and BF16 GGUF inputs through `ModelLoadOptions`. GGUF files containing
packed quantized tensors are rejected rather than being implicitly dequantized and transcoded to
affine or MXFP4 storage.

Library callers can use `quantization::quantize_checkpoint` for conversion,
the shared `ModelLoadOptions` APIs for architecture dispatch, or
`weights::load_safetensors_dir_quantized_strict` to populate a model that
exposes the standard packed parameter tree. Model-specific
`load_*_model_quantized` helpers remain available. All modes call
`quantization::quantize_tensor` with a caller-owned explicit stream, so saving
and direct loading use the same numerical transform.
Direct loading materializes each packed weight/scale/bias triple before reading
the next dense tensor. This prevents MLX's lazy graphs from retaining the whole
dense checkpoint during conversion while preserving exact parity with a saved
quantized checkpoint.

To include direct Q4 conversion in a PersonaPlex load/step benchmark, use the
dense checkpoint with `--quantize-on-load`:

```sh
cargo run --release -p safemlx-lm --example personaplex_step_bench -- \
  /path/to/personaplex-dense 64 --quantize-on-load
```

Generate a deterministic fixture with the upstream `moshi_mlx` package, then
replay it through Rust:

```sh
python safemlx-lm/scripts/moshi_mlx_token_fixture.py \
  /path/to/moshika-mlx-bf16 /tmp/moshi-token-parity.safetensors \
  --require-mlx-version 0.32.0

cargo run -p safemlx-lm --release --example moshi_token_parity -- \
  /path/to/moshika-mlx-bf16 /tmp/moshi-token-parity.safetensors
```

Use the MLX version pinned by `safemlx-sys/src/mlx-c/CMakeLists.txt` when
generating a reference fixture. The version guard prevents comparisons across
different MLX kernel implementations.

The comparator uses standard relative and absolute closeness checks and
defaults to `rtol=0.02` and `atol=0.02`, suitable for BF16 cached inference.
It reports the largest absolute difference observed. Pass explicit tolerances
as the third and fourth arguments.

The fixture contains delayed temporal inputs, teacher-forced depth inputs, the
normalized temporal states, text logits, logits from every depth slice, and an
end-to-end greedy encoded-audio generation sequence. By default the exporter
creates deterministic synthetic tokens; pass `--inputs` with a safetensors file
containing `input.text`, `input.audio`, and `input.depth` to replay a prerecorded
Mimi-token sequence for the teacher-forced portion.

For a lightweight end-to-end check without downloading released weights, add
`--create-tiny --steps 6`. This creates a deterministic miniature BF16
checkpoint in the supplied model directory before exporting its reference
fixture.

Moshi projections preserve their checkpoint dtype. MLX 0.32.0 fixes the
locally built NAX metallib behavior that previously required FP32 promotion
with MLX 0.31.2.

## Expert-parallel sparse MoE inference

`expert_parallel` provides executable pure expert parallelism for official
DeepSeek-V3/R1 safetensors and Qwen3 sparse-MoE safetensors. The initial model
API requires `EP > 1`, `TP = 1`, and `PP = 1`; hybrid EP+TP and EP+PP are
rejected before checkpoint payloads are opened. Dense models and GGUF are also
rejected. Checkpoint `ep_size` describes a stored layout and is not the runtime
EP degree.

`ExpertAssignment` supports balanced-contiguous (the model default),
round-robin, and explicit owner maps. Pass a non-default assignment to
`load_expert_parallel_model_with_assignment`, or combine it with quantization
through `load_expert_parallel_model_with_options_and_assignment`. Packed
checkpoints select the exact ordered expert rows for non-contiguous policies;
they do not materialize the enclosing range. Routers and observations always
use checkpoint-global expert ids. Only immediately before a grouped expert
kernel does the dispatcher translate them to dense owner-local ids, so
non-contiguous policies do not depend on `global_id - range.start`.

The pure-EP model path uses replicated-input dispatch. Attention, norms,
routers, embeddings, dense MLPs, heads, and DeepSeek shared experts are
replicated, so every rank already has the same hidden rows and router result.
Each rank compacts only locally owned routes, executes only its local expert
bank, reduces those routes into a full zero-initialized token buffer, and uses
one all-sum for the routed contribution. DeepSeek's replicated shared expert is
computed once per rank and added *after* that all-sum; it is never multiplied
by EP size. Exact compaction performs one scalar route-count synchronization
per sparse layer, with no per-expert synchronization and no capacity dropping.

For future token-sharded execution, `all_to_all_v` accepts destination-major
activation or metadata blocks and returns received rows in source-rank order.
Because MLX 0.32 has no native all-to-all C API, it gathers counts, pads all
blocks to the global maximum, all-gathers the destination matrix, extracts the
current destination, and removes padding. This is a real Ring-compatible
fallback, but its transfer and temporary storage replicate `O(world_size)`
data. `RoutingStatistics` exposes route counts, padding, synchronization,
logical exchanged bytes, and router/compaction/exchange/expert/reduction/shared
expert/total MoE wall times so probes can report imbalance and phase overhead;
`model_time` separately records the complete model forward. Normal inference
keeps MLX's lazy scheduling, so those fields primarily measure host submission
apart from explicit waits. The opt-in `profile_expert_parallel_timings` guard
materializes each phase and the final logits before its timer stops. It is meant
for measurement only: the inserted synchronization changes scheduling and can
reduce production throughput. Use MLX device profiling when kernel-only timing
is required.

`ExpertParallelModel::forward_with_observer` preserves global router ids and
weights while exposing the rank-local routed contribution, globally reduced
routed contribution, replicated shared-expert contribution, and final combined
MoE output as distinct fields. Qwen3 callers can select the standard growing
cache with `new_cache()` or a bounded cache with
`new_qwen3_sliding_cache(window)`; both retain the same EP routing semantics.

Packed Qwen3 expert axes and official split DeepSeek experts are selected by
placement before payload materialization; remote-only indexed shards are not
opened. Dense, affine/MXFP4, and DeepSeek block-FP8 banks retain their existing
physical kernels behind `LocalExpertBank`. Routed-expert bytes should therefore
scale approximately with `1 / EP`, while `replicated_parameter_bytes` remains
constant. On-load affine/MXFP4 conversion also follows the placement plan:
only this rank's local expert banks are materialized and quantized, while
ordinary replicated matrices retain the existing tensor-at-a-time conversion.

Run a two-process Ring generation probe with the usual MLX Ring host file and
rank environment:

```sh
cargo run --release -p safemlx-lm --example expert_parallel_generate -- /path/to/model
```

The example prints assignment metadata, performs prefill and multiple decode
steps, samples only on rank zero, synchronizes token/stop state, and reports
routing counters. Ring is intended for correctness and functional testing.
JACCL or NCCL is expected for practical low-latency EP. Small-batch decode is
often dominated by expert imbalance, and the all-gather all-to-all fallback has
substantial memory/bandwidth overhead. Replicated-input EP avoids token exchange
entirely and is usually preferable until a native all-to-all is available. No
speedup is implied without measurements on the target checkpoint and backend.

For a device-complete performance probe comparing a complete model on rank
zero, replicated-input EP on every rank, and a variable-count synthetic
sharded-input exchange, run:

```sh
cargo run --release -p safemlx-lm --example expert_parallel_benchmark -- \
  /path/to/model --backend jaccl --device gpu --warmup 1 --iterations 3
```

The CSV reports prefill and fixed-token decode latency/throughput, every MoE
phase, summed MoE and whole-model time, routes-per-rank imbalance, logical
bytes, padding, synchronization, peak MLX memory, and complete-versus-EP logit
error. The synthetic case deliberately uses uneven destination counts to expose
fallback padding. Ring can be used with `--backend ring --device cpu` for a
functional comparison; MLX Ring collectives do not currently execute on GPU.
Results include the synchronization introduced by phase profiling and should
not be presented as unprofiled production throughput or as an automatic EP
speedup.

Useful verification and opt-in probe commands are:

```sh
cargo test -p safemlx-lm expert_parallel --lib
cargo test -p safemlx-lm --test distributed_expert_exchange_ring -- --ignored --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_two_process_model_parity -- --ignored --exact --nocapture
cargo run --release -p safemlx-lm --example expert_parallel_generate -- /path/to/model
cargo run --release -p safemlx-lm --example expert_parallel_benchmark -- /path/to/model --backend jaccl
```

The exchange Ring test covers variable-count transport plus complete two-rank
forward and reverse sharded dispatch. Its non-monotonic route metadata runs
through packed ReLU2 and native block-FP8 local banks, and a separate case
keeps one rank completely route-empty while every collective is still entered.

The model-parity Ring test uses tiny deterministic complete-model references
and checks prefill, two cached decode steps, and three synchronized tokens for
dense and affine-packed Qwen3/DeepSeek banks plus native DeepSeek block-FP8.
It also runs packed Qwen with round-robin placement and split DeepSeek with an
explicit non-contiguous owner map.
Its DeepSeek fixture crosses a dense-to-MoE layer boundary, uses two router
groups, and deliberately gives one rank zero routes to exercise imbalance and
empty-local-work behavior. GPU FP8 keeps the packed Metal kernels; CPU Ring
uses the slower dequantized FP8 reference path.

## License

Licensed under either Apache-2.0 or MIT.
