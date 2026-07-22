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

## Persistent safetensors storage

The public `safemlx_lm::weight_store` module catalogs safetensors checkpoints
without materializing tensor arrays. `SafetensorsWeightStore` accepts a direct
payload file, a directory containing `model.safetensors`, or a Hugging
Face-style sharded index. Indexed construction reads only the index; payload
shards and their tensor metadata are mapped lazily when a tensor is acquired.

An acquired `WeightLease` pins its mapped bytes. Full tensors, contiguous axis
ranges, and ordered axis indices are selected before the result is copied to a
caller-provided execution stream. Materialization evaluates and conservatively
synchronizes its source and execution streams before returning, so the result
cannot retain a lazy dependency on mmap storage after the lease is dropped.
The deterministic mapped-shard cache has a configurable nonzero per-store bound. A live
lease pins its cache entry; if every entry at the bound is leased, acquisition
returns a structured capacity error instead of exceeding the limit.

Rank-aware loading uses the same store and selection implementation. Placement
is resolved before execution-stream materialization, and indexed shards that
contain only remote tensors remain untouched. Cache hits and memory-mapped page
faults are not reported as known physical disk transfers because logical
materialization and storage I/O are different measurements.

`ModelLoadOptions` selects either existing eager execution or the generic
layerwise engine. DeepSeek-V3/R1, Gemma 4, Inkling, Llama,
Mistral, GPT-OSS, LFM2/LFM2.5, Nemotron-H, Qwen3, Qwen3-Next, Qwen3-VL,
Qwen3-VL-MoE, and Qwen3.5 safetensors have registered adapters,
including dense and MoE variants. Moshi and PersonaPlex use the same engine with
independent temporal-layer and depth-codebook-slice windows. A requested family without a registered adapter returns
a specific error and never silently falls back to fully resident execution.

## Offload planning and observability

Distributed placement decides which tensors a rank owns. Residency is a
separate concern that decides where an owned logical unit lives and for how
long. The public `safemlx_lm::offload` module provides architecture-independent
configuration, explicit deterministic plans, tier byte totals, and reusable
telemetry. The public `safemlx_lm::residency` module executes those plans for
caller-defined logical units. Each `OffloadUnit` groups one or more named
checkpoint selections, including companion tensors that must become visible
atomically.

`ResidencyManager` starts disk-planned units without arrays, materializes
host-planned units on an explicit CPU stream, and materializes device-planned
units on an explicit execution stream. A unit may hold both host and device
copies. Dropping a host copy leaves the checkpoint as its canonical disk-backed
source; dropping a device copy falls back to an existing host copy or the
checkpoint. Host and device copies consume their finite logical budgets
independently.

Pinned units cannot be evicted. Windowed units are protected in the active
execution window and preferred for eviction after departure. Cacheable units
remain opportunistically and are evicted by deterministic LRU order, with unit
identifiers breaking ties. RAII `ResidentUnitLease` values explicitly pin the
requested tier while in use. Callers should not retain cloned MLX arrays beyond
a lease when authoritative residency accounting is required.

Ordinary `LayerwiseHost` prefetch and execution-window lookahead honor
`OffloadConfig::prefetch_depth`, but run synchronously. Experimental dense disk
streaming adds a bounded, joined CPU worker for disk-to-host layer warming.
Device promotion remains on the ordered execution path because MLX does not
expose the cross-stream events or fences needed for arbitrary transfer/compute
overlap. Transfer, stall, eviction, current, and peak residency observations
feed the offload telemetry, while mapped-shard and process page-fault
diagnostics remain separate.

`ResidentLayerGroup` adds named, deterministic ordered-unit preparation and
explicit trimming even under an unlimited device budget. Independent groups
can represent text, vision, audio, temporal, or depth-transformer stacks and
can be cleared without disturbing each other. `LayerwiseModel<A>` preserves the
compatible homogeneous-KV path. `GeneralLayerwiseModel<A>` adds associated
input and cache types, heterogeneous runtime units, full-cache access, and
central retained-state evaluation for recurrent and multimodal adapters.

## Experimental dense disk streaming

`WeightResidency::DenseDiskStream(DenseDiskStreamLoadOptions)` keeps ordinary
decoder and execution-stack layers disk planned and array-free at model load.
Its device and host parameter budgets are always finite. Protected host and
device lookahead depths are independent; older cacheable copies remain resident
until deterministic LRU or LFU eviction is needed. A zero host budget is an
explicit direct disk-to-device mode and requires zero host lookahead and queue
capacity. The persistent `SafetensorsWeightStore` remains the canonical cold
source and its mapped-shard cache stays independently bounded.

This mode is experimental and capacity-oriented. A dense decoder touches
essentially every layer for every token. If neither the logical layer caches nor
the operating system page cache retains the working set, small-token
autoregressive decode can approach reading the checkpoint once per generated
token and may be extremely slow. Background host prefetch can hide some page
fault and materialization latency, but it cannot remove required bytes. The
library promises correctness and bounded logical layer residency, not a
throughput improvement.

Dense disk streaming is mainly appropriate for capacity-first experiments,
large prefill or offline batches, quantized checkpoints, fast local storage,
and workloads that benefit from OS page-cache retention. On Apple silicon, CPU
and GPU arrays share physical unified memory, so logical host/device tiers do
not create additional physical capacity. Pinned embeddings, final norms and
output projections, activations, KV or recurrent state, kernels, allocator
caches, and temporary compact expert banks are outside streamed-layer totals.
Exact physical disk I/O is not observable from mmap telemetry: logical misses,
mapping hits and misses, known logical transfer bytes, and minor/major page
faults are distinct observations.

Direct dense streaming of an MoE checkpoint streams the complete layer,
including its expert bank. Use
`WeightResidency::SparseExpertCacheWithDenseLayers` when expert-granular reuse
is desired while non-expert layer weights remain disk streamed. Existing
`ExpertCacheLoadOptions::non_expert` continues to select eager host-backed
non-expert semantics for `SparseExpertCache`.

Pure Llama/Mistral and DeepSeek-V3/R1 pipeline stages stream locally: each rank
catalogs its contiguous global layer range and owns a separate worker and
residency report. The first stage alone owns the input embedding, and the final
stage alone owns final normalization and output projection weights.

The `llama_residency` example accepts `--dense-disk-stream` together with
`--stream-host-budget`, `--stream-device-budget`,
`--stream-host-lookahead`, `--stream-device-lookahead`, and
`--stream-queue-capacity`. It reports load-time residency, first-process and
repeated-process prefill latency, decode latency, cache occupancy, logical
transfers, queue waits, mappings, and available process page-fault samples.
Those labels do not imply physically cold or warm storage, and the example
never drops operating-system caches.

`DerivedWeightRecipe` composes checkpoint selection and renaming,
concatenation, stacking, reshape, axis permutation, and dtype cast. Recipes are
validated from metadata, keep all source leases alive through evaluation, and
materialize the transformed runtime representation on the host before device
promotion. Direct `WeightBinding` construction remains compatible.

CUDA behavior is verified by CI; local verification in this workspace does not
make discrete-memory capacity or performance claims.

The `safemlx::memory` controls affect process-global MLX-managed allocations.
They do not directly constrain process RSS, checkpoint mappings, or unrelated
native allocations. The pinned MLX 0.32.0 C surface has whole-stream
synchronization but no event/fence primitive, so residency execution uses
conservative stream synchronization until an event-backed API is available.

## Llama-compatible weight residency

`llama::load_llama_model` returns one `LlamaModel` inference facade. Choose
`LlamaLoadOptions::fully_resident()` for the eager execution-device model or
`LlamaLoadOptions::layerwise_host(...)` for the generic host-backed decoder
engine. Both policies use the same `LlamaCache`, `forward`, `prefill`, `decode`,
and `CausalLm` surface. Dense and already-packed MLX affine/MXFP4 Llama and
Mistral safetensors are supported.

```rust
use safemlx_lm::{
    load_llama_model, LayerwiseLoadOptions, LlamaLoadOptions,
};

let eager = LlamaLoadOptions::fully_resident();
let bounded = LlamaLoadOptions::layerwise_host(
    LayerwiseLoadOptions::new(offload_config),
);
let mut model = load_llama_model(model_dir, bounded, stream, cpu_stream)?;
let mut cache = model.new_cache();
let logits = model.prefill(&token_ids, &mut cache, stream)?;
```

The embedding, final normalization, untied output projection when present,
activations, and KV cache remain on the execution device. Every decoder layer
is initialized on the caller's CPU stream and remains host-resident. Decoder
copies move through a bounded device window whose size is
`OffloadConfig::prefetch_depth()` and includes the current layer. A temporary
unloaded block receives handle clones from its resident lease, executes, and is
dropped after the output and updated cache have been evaluated and the stream
has synchronized. Packed weights, scales, and biases move unchanged; they are
not dequantized or repacked.

Host budgets must contain all decoder weights. Device budgets must contain
pinned static weights plus the largest permitted consecutive layer window.
Residency reports account for parameter copies only; activations, KV state,
kernels, and allocator cache can make MLX peak memory larger.

Transfers are synchronous because the pinned MLX API exposes whole-stream
synchronization but no events. GGUF, load-time quantization, pinned host
buffers, KV-cache offload, and asynchronous transfer or
compute overlap are not supported by this policy. The opt-in
`llama_residency` example accepts a real checkpoint directory and reports
latency, throughput, logical residency, transfer telemetry, allocator samples,
and mapped-shard diagnostics.

## Qwen3 weight residency

Dense and sparse-MoE Qwen3 use one adapter. Token embeddings, final norm, and
the tied or untied output projection stay pinned. Each complete transformer
block, including its routed expert bank, is one `text_decoder` execution unit.
Standard KV caches remain device resident. Matching checkpoint-native affine
and MXFP4 parameter trees load directly; load-time conversion in the layerwise
path is rejected.

## GPT-OSS weight residency

GPT-OSS keeps embeddings, final norm, and the output head pinned while complete
sparse decoder blocks move through the `text_decoder` window. The adapter owns
the alternating full/sliding cache schedule, sink-token mask behavior, and RoPE
state. Checkpoint-native MXFP4 expert blocks and scales remain packed.

## LFM2/LFM2.5 weight residency

Dense and MoE LFM2 variants share one hybrid adapter for full-attention and
short-convolution layers. KV arrays and bounded convolution state are evaluated
before a block lease is released. Public per-expert `w1`/`w2`/`w3` tensors are
concatenated and stacked into runtime expert banks one layer at a time on the
host; already-packed checkpoint representations load directly.

## DeepSeek-V3/R1 weight residency

DeepSeek keeps embeddings, final normalization, and the output head pinned while
complete MLA blocks move through the `text_decoder` window. Compressed latent and
rotary-key cache arrays are evaluated before each block lease is released. The
dense prefix and routed-plus-shared MoE suffix use the same adapter; official
per-expert tensors are stacked per layer for dense, affine, and native 128-by-128
block-FP8 checkpoints. Appended multi-token-prediction weights remain explicitly
ignored just as they are by the eager text-model loader.

## Inkling weight residency

Inkling local and global attention blocks share one text-decoder window. Global
and sliding KV arrays plus all four short-convolution states per block are
evaluated before lease release. Released `model.llm` names are rewritten, short
convolution weights are cast to the runtime dtype, and interleaved dense, routed,
and shared `w13` tensors are selected into runtime gate/up order on the host.
The dMel encoder is a pinned static unit, while the four released hMLP
projection/fold layers use an independent vision window. Typed prompts may
interleave text, discrete audio, precomputed media embeddings, and image patches.

## Nemotron-H weight residency

One hybrid adapter handles Mamba2, attention, dense MLP, and sparse MoE blocks.
Mamba convolution and SSM arrays plus attention KV arrays are evaluated before
lease release. Public `backbone`/`mixer` names are resolved through the same key
rewrite used by eager loading, and split ReLU2 experts are stacked per layer.

## Qwen hybrid weight residency

Qwen3-Next and Qwen3.5 share one adapter for recurrent linear attention
and full attention. Qwen3-Next fused QKVZ/BA tensors are selected into runtime
projections without materializing the complete checkpoint, including 128-row
block-space selection for native FP8 QKVZ inverse scales, and public split
SwiGLU experts are packed per layer. The official dynamic E4M3 128 x 128 format
is supported by resident, layerwise, sparse expert-cache, and pure
expert-parallel Qwen3-Next loading. Qwen3.5 dense and packed-MoE checkpoints use
the same block loop. Multimodal checkpoints add an independent Qwen vision-block
group and reuse the resident patch, position, and merger math around that group.

## Layerwise safetensors coverage

The table records the architecture inventory used by the normal and realtime
dispatch surfaces. “Precise error” means a layerwise request is rejected and is
never replaced by eager loading.

| Family | Eager loader | Layerwise loader | Cache/state | Pinned static modules | Windowed unit | Checkpoint transform / native packing | Parity coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama / Mistral | yes | yes | growing or sliding KV | embedding, norm, head | decoder block | direct affine/MXFP4 | prefill and multi-step decode |
| Qwen3 dense / MoE | yes | yes | growing KV | embedding, norm, head | decoder block with local experts | direct affine/MXFP4 | dense and MoE prefill/decode |
| GPT-OSS | yes | yes | alternating full/sliding KV | embedding, norm, head | sparse decoder block | native MXFP4 experts | both attention modes and multi-step decode |
| LFM2/LFM2.5 dense / MoE | yes | yes | growing KV or convolution state | embedding, norm, tied/untied head | hybrid decoder block | split SwiGLU experts packed per layer; packed form accepted | dense and split-MoE hybrid prefill/decode |
| DeepSeek-V3/R1 | yes | yes | compressed MLA latent and rotary-key state | embedding, norm, head | MLA decoder block with dense or routed/shared experts | official split experts stacked per layer; direct dense/affine and native block-FP8 banks | dense-to-MoE prefill/decode at two depths; native block-FP8 prefill/decode |
| Gemma 4 multimodal | yes | yes | alternating KV plus transient shared-KV and media state | patch embedding/pooling, audio subsampling/output, modality projections, token/per-layer embeddings, norm, head | independent vision, audio, and sliding/full text groups | public prefix rewrite; direct affine/MXFP4 text and modality projections | vision/audio/text typed prefill parity; per-layer inputs, shared KV, prefill/decode at two depths |
| Inkling multimodal | yes | yes | global/local KV, four convolution states per layer, transient hMLP activations | dMel embedding/norm, hMLP final norm, text embedding/norm/head | independent hMLP and local/global dense-or-MoE text groups | released-name rewrite, convolution cast, dense/routed/shared w13 deinterleave | audio/text typed prefill parity; local/global and dense/MoE prefill/decode at two depths |
| Nemotron-H | yes | yes | attention KV and Mamba convolution/SSM state | embedding, norm, tied/untied head | hybrid block | public key rewrite and split ReLU2 expert packing | all four block kinds, split MoE, prefill/decode |
| Qwen3-Next / Qwen3.5 | yes | yes | full-attention KV, recurrent linear-attention state, transient vision state | Qwen vision patch/position/merger modules, embedding, norm, tied/untied head | optional vision group plus shared hybrid text group | fused QKVZ/BA selection; split SwiGLU and FP8 expert recipes | Qwen3.5 image/text prefill parity; Qwen3-Next dense/split-MoE and Qwen3.5 dense/MoE prefill/decode |
| Qwen3-VL / Qwen3-VL-MoE | yes | yes | text KV plus multimodal RoPE delta and transient DeepStack state | patch/position embeddings, vision mergers, text embedding/norm/head | independent vision block and dense/MoE text-block groups | direct public DeepStack vision and packed Qwen3 expert trees | image prefill plus multi-step decode for dense/MoE; two depths for dense |
| PersonaPlex / Moshi realtime | yes, realtime API | yes, realtime API | temporal KV plus reset-per-frame depth KV and delayed-stream state | text/audio embeddings, temporal norm and heads | independent temporal layers and per-codebook depth slices | native Moshi layout; released PersonaPlex PyTorch norms, packed attention, embeddings, and projections derived lazily | teacher-forced logits, consecutive realtime frames, offline encoded sequence, forced prompt/cache continuity |

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
Q5_0 and Q5_1 tensors are converted to float16 while loading; unsupported GGUF
tensor types return an error. Model dispatch uses
`general.architecture`; the current GGUF adapters support text-only `deepseek2`,
`gemma4`, `llama`, `mistral`, `lfm2`, `lfm2moe`, `nemotron_h`,
`nemotron_h_moe`, `qwen3`, `qwen3moe`, dense `qwen35`, and `qwen35moe`
architectures, plus `qwen3next` and dense `qwen3vl` with its separate vision projector. For
Qwen3-VL, put the llama.cpp-style dense F16/BF16/F32
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
selected local tensors. Pipeline GGUF remains unsupported because its
placement-aware loader has not yet been migrated to the GGUF tensor stream.

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

The partition and executable pipeline Ring proofs are opt-in:

```sh
cargo test -p safemlx-lm --test distributed_partition_ring \
  ring_two_process_partition_load -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_deepseek_pipeline_persistence -- --ignored --exact --nocapture
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
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_deepseek_tensor_parallel_persistence -- --ignored --exact --nocapture
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
| Qwen3-Next | yes | native block FP8, MLX affine/MXFP4 | yes / yes, from dense checkpoints | `LoadedModel` | Official dynamic E4M3 128 x 128 checkpoints work with resident, layerwise, sparse expert-cache, and expert-parallel policies; fused weights/scales are split while streaming and native FP8 is never implicitly transcoded |
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

Pass `ModelLoadOptions::default().with_weight_residency(WeightResidency::LayerwiseHost(...))`
to `load_realtime_model_with_options` to keep temporal layers and Moshi-family
depth-codebook slices on the host. Text/audio embeddings and temporal output
modules remain pinned, and `residency_report()` exposes the two execution groups.
PersonaPlex system-prompt helpers accept either the fully resident model or
`MoshiLayerwiseModel`, so forced voice/text prefill continues into ordinary
realtime generation with the same delayed-stream and transformer caches.

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

`expert_parallel` provides executable pure expert parallelism for the
safetensors MoE families supported by sparse expert caching: DeepSeek-V3/R1,
GPT-OSS, Inkling, LFM2, Nemotron-H, Qwen3, Qwen3-Next, Qwen3-VL-MoE, and
Qwen3.5-MoE. GPT-OSS and the hybrid or multimodal families require
`WeightResidency::SparseExpertCache`; DeepSeek and Qwen3 additionally retain
their fully resident EP loaders. The model API requires `EP > 1`, `TP = 1`,
and `PP = 1`; hybrid EP+TP and EP+PP are rejected before checkpoint payloads
are opened. Dense models and GGUF are also rejected. Checkpoint `ep_size`
describes a stored layout and is not the runtime EP degree.

`ExpertAssignment` supports balanced-contiguous (the model default),
round-robin, and explicit owner maps. Pass a non-default assignment to
`load_expert_parallel_model_with_assignment`, or use
`load_expert_parallel_model_with_options_and_assignment` for sparse caching or
the fully resident DeepSeek/Qwen3 quantization path. Packed
checkpoints select the exact ordered expert rows for non-contiguous policies;
they do not materialize the enclosing range. Routers and observations always
use checkpoint-global expert ids. Only immediately before a grouped expert
kernel does the dispatcher translate them to dense owner-local ids, so
non-contiguous policies do not depend on `global_id - range.start`.

The pure-EP model path uses replicated-input dispatch. Attention, norms,
routers, embeddings, dense MLPs, heads, and architecture-specific shared
experts are replicated, so every rank already has the same hidden rows and
router result.
Each rank compacts only locally owned routes, executes only its local expert
bank, reduces those routes into a full zero-initialized token buffer, and uses
one all-sum for the routed contribution. Replicated shared experts are computed
once per rank and added *after* that all-sum; they are never multiplied by EP
size. Exact compaction performs one scalar route-count synchronization
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

For fully resident DeepSeek and Qwen3,
`ExpertParallelModel::forward_with_observer` preserves global router ids and
weights while exposing the rank-local routed contribution, globally reduced
routed contribution, replicated shared-expert contribution, and final combined
MoE output as distinct fields. Detailed activation observation is currently
unavailable on the sparse expert-cache EP path. Qwen3 callers can select the
standard growing cache with `new_cache()` or a bounded cache with
`new_qwen3_sliding_cache(window, paging_options)`; the bounded form uses the
shared paged residency manager and both retain the same EP routing semantics.

Every supported packed or split expert layout is selected by placement before
payload materialization; remote-only indexed shards are not opened. Dense,
affine/MXFP4, FP8, and ReLU2/SwiGLU banks retain their architecture-specific
physical kernels behind the common replicated dispatch. With sparse caching,
`routed_expert_bytes` is zero and `owned_expert_bytes` describes the rank's
cold, warm, or hot catalog; with fully resident DeepSeek/Qwen3 it scales
approximately with `1 / EP`. `replicated_parameter_bytes` remains constant.
Load-time conversion is rejected for sparse-cache EP because it would require
eager expert materialization.

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
dense and affine-packed Qwen3/DeepSeek banks, native DeepSeek block-FP8, and
sparse expert-cache EP for every supported MoE family.
It also runs packed Qwen with round-robin placement and split DeepSeek with an
explicit non-contiguous owner map.
Its DeepSeek fixture crosses a dense-to-MoE layer boundary, uses two router
groups, and deliberately gives one rank zero routes to exercise imbalance and
empty-local-work behavior. GPU FP8 keeps the packed Metal kernels; CPU Ring
uses the slower dequantized FP8 reference path.

## License

Licensed under either Apache-2.0 or MIT.
Paged attention-cache residency and reusable prompt-cache persistence are
opt-in. Device-resident caches remain the default. See
[`CACHE_RESIDENCY.md`](CACHE_RESIDENCY.md) for configuration, compatibility,
cost, and safety details, and run `paged_prompt_cache` for a deterministic
save/drop/reopen parity check.
