# Paged attention caches

Device-resident caching remains the default. Cache residency is configured
separately from `WeightResidency`; static weights, streamed weights,
activations, recurrent state, and KV or compressed-latent state have separate
budgets and lifetimes.

`CacheResidencyPolicy::Paged` divides attention state into immutable sealed
blocks and a mutable device tail. One `CacheResidencyManager` is shared by all
attention layers in a model cache, so finite device and host byte limits apply
globally instead of once per layer. Constructors reject zero device budgets,
zero recent-block protection, non-positive block sizes, and unbounded implicit
limits.

```rust
use safemlx_lm::{CacheResidencyPolicy, PagedCacheOptions};

let options = PagedCacheOptions::new(
    128,          // tokens per block
    512 << 20,    // finite logical device-cache bytes
    2 << 30,      // finite logical host-cache bytes
    1,            // recent device blocks protected per layer
)?
.with_full_attention(true);

let cache = model.new_cache_with_options(CacheResidencyPolicy::Paged(options))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The ordinary `new_cache()` methods are unchanged. Supported paged construction
currently covers:

- Llama/Mistral full and sliding attention through resident, layerwise-host,
  dense-streamed, pipeline, and tensor-parallel execution.
- DeepSeek compressed latent/rotary state through resident, layerwise-host,
  dense-streamed, sparse-expert, pipeline, tensor-parallel, and replicated
  expert-parallel execution.
- GPT-OSS alternating full/sliding attention, including learned softmax sinks,
  through resident, layerwise, sparse-expert, and replicated expert-parallel
  execution.

Inkling relative-position attention, Qwen recurrent linear-attention state,
Mamba state, convolution state, and multimodal transient state are not mapped
onto this block representation. High-level paged constructors reject those
representations instead of changing their semantics.

## Sliding and full attention

Sliding-window paging is naturally bounded. Sealed blocks older than the
semantic attention window are discarded unless persistence retention was
explicitly enabled. Absolute token offsets continue to advance after discard.
A multi-token update keeps every new token plus the preceding visible window,
so each query can attend to the correct earlier tokens in the same update.

Full attention is explicit and experimental. It uses an FP32 online-softmax
reduction over ordered blocks: each block updates a running row maximum,
normalization sum, and weighted-value accumulator. Causal positions, rank-2
boolean or additive masks, grouped-query heads, multi-token queries, sliding
windows, pinned prefixes, and GPT-OSS learned sinks are preserved. Running
state is evaluated before the current block lease is released, and no
whole-history key/value array is assembled.

Every full-attention decode token still reads all historical blocks. Host
paging reduces logical device residency but can add large per-token transfers.
Live disk-backed full attention can approach rereading the complete context for
every token and is intended only for experimentation. DeepSeek compressed
latents reduce stored bytes, but attention still consumes every historical
latent block and reconstructs head-specific contributions one block at a time.
No throughput improvement is promised.

Apple unified memory means logical host/device placement does not create
additional physical capacity. Residency reports describe the runtime's logical
tiers and requested transfer bytes. They do not claim physical disk bytes:
mapped pages may be served by the operating-system cache. The MLX C API does
not expose cross-stream event or fence primitives, so cache evaluation and
device-to-host completion use conservative synchronization. Disk reads and
writes use a bounded manager-owned worker queue that is stopped and joined on
drop.

Live disk backing is disabled by default and requires a dedicated directory,
finite logical disk budget, and finite queue capacity:

```rust
let options = PagedCacheOptions::new(128, 512 << 20, 2 << 30, 1)?
    .with_full_attention(true)
    .with_live_disk("/private/tmp/my-live-cache", 8 << 30, 2)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Mutable tails are never written. Sealed blocks are evaluated before a worker
receives them. Required blocks are protected by leases; queue or I/O failures
are returned to inference. Ephemeral live-cache files are removed when the last
manager is dropped. Persistent prompt-cache files are never treated as
ephemeral state.

## Reusable prompt caches

A prompt cache is a completed immutable text prefix, not an arbitrary snapshot
of lazy in-flight arrays. `ModelCache::save_prompt_cache` seals partial tails,
writes one bounded safetensors shard per cache block into a temporary sibling
directory, flushes data and the manifest, and publishes the directory with an
atomic rename. Existing destinations are replaced only when
`PromptCacheOptions::replace_existing` is true.

`inspect_prompt_cache` validates the manifest and safetensors metadata without
creating MLX arrays. `Model::load_prompt_cache` and
`LoadedModel::load_prompt_cache` catalog compatible shards as read-only disk
blocks. Arrays are loaded on demand, and suffix tokens append new mutable and
sealed blocks without modifying imported files.

The manifest records:

- schema version, model family, effective model type, and caller-supplied
  checkpoint and architecture fingerprints;
- global layer range, representation, block ranges, array names, shapes,
  dtypes, batch size, and total prefix length;
- SHA-256 of the exact little-endian prefix token IDs;
- sliding window, sink count, and distributed topology;
- optional application namespace, which is never accepted as compatibility
  evidence.

Loading requires the exact prefix token IDs and rejects wrong fingerprints,
architecture settings, layer ranges, topology, block size, or prefix identity.
Inspection rejects missing or truncated shards, duplicate or overlapping
blocks, gaps, invalid ranges, inconsistent shapes or dtypes, extra arrays, and
relative paths or symlinks that escape the cache directory. A tensor-parallel
cache is rank-local and cannot be reused under a different topology. Pipeline
stages persist only their global layer range. Expert-parallel attention state
is recorded as replicated, not expert-sharded.

Checkpoint fingerprints are supplied by the application because hashing every
checkpoint byte can be expensive. They must be based on stable checkpoint
content or a trusted immutable model identifier, not solely on an absolute
path. Applications that require a stronger identity must compute and supply a
content hash. The architecture fingerprint must cover RoPE scaling and every
cache-relevant normalized setting.

Token IDs are sufficient only for text prefixes. Multimodal and realtime
prefixes need image, audio, video, timing, and processor identity that this
format does not yet encode, so high-level loading rejects those cache
representations. A loaded prefix contains attention state, not logits for an
empty suffix. Run at least one suffix token before sampling, or persist logits
separately in the application.

Run the deterministic text-prefix example with explicit compatibility
identities:

```sh
cargo run -p safemlx-lm --example paged_prompt_cache -- \
  /path/to/model /tmp/reusable-prefix \
  --prompt "Explain unified memory briefly." \
  --suffix-token 42 \
  --layer-count 32 \
  --checkpoint-fingerprint sha256:CHECKPOINT_DIGEST \
  --architecture-fingerprint sha256:ARCHITECTURE_DIGEST
```

The example runs uninterrupted and restored suffix paths, prints logit parity,
and reports block, tier, transfer, attention-scan, mapping, queue, and
persistence counters. Use `--live-disk-dir` to demonstrate explicit live
backing. Compare that output with `--device-cache` for the ordinary cache and
with a model's configured sliding window for bounded sliding residency. Do not
flush privileged operating-system caches when measuring.
