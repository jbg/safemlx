//! Executable pure tensor-parallel inference for Llama-compatible and
//! DeepSeek-V3/R1 text models.
//!
//! Every rank executes every decoder layer. Q/K/V and gate/up projections are
//! column sharded, their intermediate activations stay local, and output/down
//! projections are row sharded followed by exactly one all-sum. Embeddings and
//! output logits use balanced contiguous vocabulary ranges.

use std::{
    collections::HashMap,
    ops::Range,
    path::{Path, PathBuf},
};

use safemlx::{
    distributed::{self, Group},
    module::{Module, ModuleParameters},
    nn,
    ops::{indexing::TryIndexOp, ones, quantized_matmul_with_mode, zeros, zeros_like},
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    cache::{CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, SlidingKeyValueCache},
    error::Error,
    models::{
        common::{
            attention::{
                apply_rope_and_update_cache, batch_seq, finish_attention,
                reshape_attention_projection, sliding_window_prefill_attention,
            },
            layers::silu,
            linear,
        },
        deepseek_v3, llama, ModelKind, ModelLoadOptions,
    },
    parallel::{
        balanced_contiguous_range, load_safetensors_partition_on_streams, ParallelTopology,
        PlacementPlan, RankPartition, TensorPlacement,
    },
    pipeline::{assign_module, load_deepseek_experts, SynchronizedToken},
    quantization::{should_quantize_on_load, WeightQuantization},
    sampler::Sampler,
    utils::create_causal_mask,
    weights::StrictLoadConfig,
};

/// Immutable description of one rank's tensor-parallel model state.
#[derive(Debug, Clone)]
pub struct TensorParallelInfo {
    /// Rank in the communication group.
    pub global_rank: usize,
    /// Tensor-parallel coordinate.
    pub tensor_parallel_rank: usize,
    /// Tensor-parallel process count.
    pub tensor_parallel_size: usize,
    /// Loaded architecture.
    pub model_kind: ModelKind,
    /// Global query-head count.
    pub global_attention_heads: i32,
    /// Local query-head count.
    pub local_attention_heads: i32,
    /// Global key/value-head count. DeepSeek MLA uses its query-head count.
    pub global_kv_heads: i32,
    /// Local key/value-head count.
    pub local_kv_heads: i32,
    /// Balanced local vocabulary range.
    pub local_vocabulary_range: Range<usize>,
    /// Names of checkpoint slices materialized on this rank.
    pub owned_tensors: Vec<String>,
    /// Bytes in locally selected checkpoint tensors before runtime packing.
    pub local_parameter_bytes: usize,
    /// Checkpoint payload shards actually opened by this rank.
    pub opened_checkpoint_shards: Vec<PathBuf>,
}

/// Llama cache storage with only this rank's local K/V heads.
#[derive(Debug, Clone)]
pub enum TensorParallelLlamaLayerCache {
    /// Unbounded concatenating cache.
    Standard(ConcatKeyValueCache),
    /// Bounded sliding-window cache.
    Sliding(SlidingKeyValueCache),
}

/// Architecture-checked rank-local tensor-parallel cache.
#[derive(Debug, Clone)]
pub enum TensorParallelCache {
    /// Llama-compatible local-head caches.
    Llama(Vec<TensorParallelLlamaLayerCache>),
    /// DeepSeek compressed-latent caches.
    DeepSeek(Vec<CompressedLatentCache>),
}

impl TensorParallelCache {
    /// Clears all retained sequence state.
    pub fn reset(&mut self) {
        match self {
            Self::Llama(caches) => caches.iter_mut().for_each(|cache| match cache {
                TensorParallelLlamaLayerCache::Standard(cache) => cache.clear(),
                TensorParallelLlamaLayerCache::Sliding(cache) => cache.clear(),
            }),
            Self::DeepSeek(caches) => caches.iter_mut().for_each(CompressedLatentCache::clear),
        }
    }

    /// Returns the common cache sequence offset.
    pub fn offset(&self) -> i32 {
        match self {
            Self::Llama(caches) => caches.first().map_or(0, |cache| match cache {
                TensorParallelLlamaLayerCache::Standard(cache) => cache.offset(),
                TensorParallelLlamaLayerCache::Sliding(cache) => cache.offset(),
            }),
            Self::DeepSeek(caches) => caches.first().map_or(0, CompressedLatentCache::offset),
        }
    }
}

struct LlamaTensorModel {
    global_args: llama::ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    layers: Vec<llama::TransformerBlock>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

struct DeepSeekTensorModel {
    global_args: deepseek_v3::ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    layers: Vec<deepseek_v3::DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
}

enum TensorArchitecture {
    Llama(LlamaTensorModel),
    DeepSeek(DeepSeekTensorModel),
}

/// Executable rank-local pure tensor-parallel model.
pub struct TensorParallelModel {
    topology: ParallelTopology,
    info: TensorParallelInfo,
    architecture: TensorArchitecture,
}

impl std::fmt::Debug for TensorParallelModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorParallelModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl TensorParallelModel {
    /// Returns immutable rank-local placement and memory information.
    pub fn info(&self) -> &TensorParallelInfo {
        &self.info
    }

    /// Allocates one rank-local cache entry per decoder layer.
    pub fn new_cache(&self) -> TensorParallelCache {
        match &self.architecture {
            TensorArchitecture::Llama(model) => TensorParallelCache::Llama(
                (0..model.layers.len())
                    .map(|_| match model.global_args.sliding_window {
                        Some(window) => TensorParallelLlamaLayerCache::Sliding(
                            SlidingKeyValueCache::new(window),
                        ),
                        None => TensorParallelLlamaLayerCache::Standard(ConcatKeyValueCache::new()),
                    })
                    .collect(),
            ),
            TensorArchitecture::DeepSeek(model) => TensorParallelCache::DeepSeek(
                (0..model.layers.len())
                    .map(|_| CompressedLatentCache::new())
                    .collect(),
            ),
        }
    }

    /// Runs prefill or decode and returns this rank's vocabulary-logit shard.
    pub fn forward_local_logits(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_group(group)?;
        self.topology.validate_execution_stream(stream)?;
        if tokens.ndim() != 2 {
            return Err(Error::Parallel(format!(
                "tensor-parallel token input must have rank 2 [batch, sequence], got {:?}",
                tokens.shape()
            )));
        }
        match (&mut self.architecture, cache) {
            (TensorArchitecture::Llama(model), TensorParallelCache::Llama(caches)) => {
                forward_llama(model, tokens, mask, caches, &self.info, group, stream)
            }
            (TensorArchitecture::DeepSeek(model), TensorParallelCache::DeepSeek(caches)) => {
                forward_deepseek(model, tokens, mask, caches, &self.info, group, stream)
            }
            (TensorArchitecture::Llama(_), _) => Err(Error::Parallel(
                "tensor-parallel cache is DeepSeek but model is Llama".into(),
            )),
            (TensorArchitecture::DeepSeek(_), _) => Err(Error::Parallel(
                "tensor-parallel cache is Llama but model is DeepSeek".into(),
            )),
        }
    }

    /// Runs prefill or decode and gathers complete vocabulary logits on every rank.
    pub fn forward(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let local = self.forward_local_logits(tokens, mask, cache, group, stream)?;
        let widths = vocabulary_widths(
            self.global_vocabulary_size(),
            self.topology.tensor_parallel_size,
        )?;
        Ok(distributed::all_gather_uneven_axis(
            &local, -1, &widths, group, stream,
        )?)
    }

    /// Alias for a prompt forward pass returning complete logits.
    pub fn prefill(
        &mut self,
        tokens: &Array,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, None, cache, group, stream)
    }

    /// Alias for one or more autoregressive decode tokens returning complete logits.
    pub fn decode(
        &mut self,
        tokens: &Array,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, None, cache, group, stream)
    }

    /// Samples only on `sampling_rank` and synchronizes the token and stop flag.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        complete_logits: &Array,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
    ) -> Result<SynchronizedToken, Error> {
        self.validate_group(group)?;
        if sampling_rank >= group.size() {
            return Err(Error::Parallel(format!(
                "sampling rank {sampling_rank} is outside TP size {}",
                group.size()
            )));
        }
        let batch = complete_logits.dim(0);
        let local_token = if group.rank() == sampling_rank {
            let logits = if complete_logits.ndim() == 3 {
                complete_logits.try_index_device((.., -1, ..), stream)?
            } else {
                complete_logits.clone()
            };
            sampler
                .sample(&logits, temperature, prng_state, stream)?
                .reshape(&[batch, 1], stream)?
        } else {
            zeros::<u32>(&[batch, 1], stream)?
        };
        let token = distributed::all_sum(&local_token, group, stream)?;
        let local_finished = if group.rank() == sampling_rank && finished {
            ones::<i32>(&[], stream)?
        } else {
            zeros::<i32>(&[], stream)?
        };
        let finished = distributed::all_sum(&local_finished, group, stream)?;
        eval([&token, &finished])?;
        stream.synchronize()?;
        Ok(SynchronizedToken {
            token,
            finished: finished.try_item::<i32>(stream)? != 0,
        })
    }

    fn global_vocabulary_size(&self) -> usize {
        match &self.architecture {
            TensorArchitecture::Llama(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::DeepSeek(model) => model.global_args.vocab_size as usize,
        }
    }

    fn validate_group(&self, group: &Group) -> Result<(), Error> {
        if group.rank() != self.topology.global_rank || group.size() != self.topology.world_size {
            return Err(Error::Parallel(format!(
                "tensor-parallel topology expects group rank {}/{} but received {}/{}",
                self.topology.global_rank,
                self.topology.world_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }
}

fn validate_pure_tensor(topology: ParallelTopology) -> Result<(), Error> {
    if topology.tensor_parallel_size <= 1 {
        return Err(Error::Parallel(
            "tensor-parallel loading requires tensor_parallel_size > 1".into(),
        ));
    }
    if topology.pipeline_parallel_size != 1 || topology.expert_parallel_size != 1 {
        return Err(Error::Parallel(format!(
            "pure tensor-parallel execution requires PP=1 and EP=1, got TP={} PP={} EP={}; hybrid TP+PP and TP+EP are unsupported",
            topology.tensor_parallel_size,
            topology.pipeline_parallel_size,
            topology.expert_parallel_size
        )));
    }
    if topology.world_size != topology.tensor_parallel_size {
        return Err(Error::Parallel(
            "pure tensor-parallel world size must equal tensor-parallel size".into(),
        ));
    }
    Ok(())
}

fn exact_division(name: &str, value: i32, parts: usize) -> Result<i32, Error> {
    let parts_i32 = i32::try_from(parts)
        .map_err(|_| Error::Parallel("tensor-parallel size does not fit in i32".into()))?;
    if value <= 0 || value % parts_i32 != 0 {
        return Err(Error::Parallel(format!(
            "{name} {value} is not divisible by tensor-parallel size {parts}"
        )));
    }
    Ok(value / parts_i32)
}

fn require_alignment(
    tensor: &str,
    dimension: i32,
    alignment: i32,
    topology: ParallelTopology,
) -> Result<(), Error> {
    if alignment <= 0 || dimension % alignment != 0 {
        return Err(Error::Parallel(format!(
            "tensor {tensor} local dimension {dimension} is not aligned to block/group size {alignment} for TP size {}",
            topology.tensor_parallel_size
        )));
    }
    Ok(())
}

fn vocabulary_widths(vocabulary: usize, parts: usize) -> Result<Vec<usize>, Error> {
    (0..parts)
        .map(|rank| {
            balanced_contiguous_range(vocabulary, parts, rank, false).map(|range| range.len())
        })
        .collect()
}

fn checkpoint_name(parameter_name: &str) -> String {
    parameter_name
        .replace(".inner.weight", ".weight")
        .replace(".inner.bias", ".bias")
}

fn parameter_names(module: &impl ModuleParameters, prefix: &str) -> Vec<String> {
    module
        .parameters()
        .flatten()
        .keys()
        .map(|name| format!("{prefix}.{name}"))
        .collect()
}

#[derive(Clone, Copy)]
enum ProjectionPlacement {
    Replicated,
    Column,
    Row,
}

fn projection_tensor_placement(
    name: &str,
    projection: ProjectionPlacement,
    topology: ParallelTopology,
) -> TensorPlacement {
    match projection {
        ProjectionPlacement::Replicated => TensorPlacement::Replicated,
        ProjectionPlacement::Column => TensorPlacement::Shard {
            axis: 0,
            index: topology.tensor_parallel_rank,
            parts: topology.tensor_parallel_size,
        },
        ProjectionPlacement::Row if name.ends_with(".bias") => TensorPlacement::Replicated,
        ProjectionPlacement::Row => TensorPlacement::Shard {
            axis: 1,
            index: topology.tensor_parallel_rank,
            parts: topology.tensor_parallel_size,
        },
    }
}

fn insert_llama_layer_plan(
    plan: &mut PlacementPlan,
    layer: &llama::TransformerBlock,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    for destination in parameter_names(layer, &prefix) {
        let source = checkpoint_name(&destination);
        let projection = if source.contains(".self_attn.q_proj.")
            || source.contains(".self_attn.k_proj.")
            || source.contains(".self_attn.v_proj.")
            || source.contains(".mlp.gate_proj.")
            || source.contains(".mlp.up_proj.")
        {
            ProjectionPlacement::Column
        } else if source.contains(".self_attn.o_proj.") || source.contains(".mlp.down_proj.") {
            ProjectionPlacement::Row
        } else if source.contains("layernorm") || source.contains(".rope.") {
            ProjectionPlacement::Replicated
        } else {
            return Err(Error::Parallel(format!(
                "unknown Llama tensor-parallel tensor {source}"
            )));
        };
        plan.insert(
            destination.clone(),
            projection_tensor_placement(&source, projection, plan.topology()),
        );
    }
    Ok(())
}

fn deepseek_projection(name: &str) -> Option<ProjectionPlacement> {
    if name.contains("layernorm")
        || name.contains(".rope.")
        || name.contains(".mlp.gate.")
        || name.contains(".q_a_proj.")
        || name.contains(".q_a_layernorm.")
        || name.contains(".kv_a_proj_with_mqa.")
        || name.contains(".kv_a_layernorm.")
    {
        Some(ProjectionPlacement::Replicated)
    } else if name.contains(".q_proj.")
        || name.contains(".q_b_proj.")
        || name.contains(".kv_b_proj.")
        || name.contains(".k_b_proj.")
        || name.contains(".v_b_proj.")
        || name.contains(".gate_proj.")
        || name.contains(".up_proj.")
    {
        Some(ProjectionPlacement::Column)
    } else if name.contains(".o_proj.") || name.contains(".down_proj.") {
        Some(ProjectionPlacement::Row)
    } else {
        None
    }
}

fn insert_deepseek_layer_plan(
    plan: &mut PlacementPlan,
    args: &deepseek_v3::ModelArgs,
    layer: &deepseek_v3::DecoderLayer,
    index: usize,
    dense_source: bool,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    for destination in parameter_names(layer, &prefix) {
        let source = checkpoint_name(&destination);
        let projection = deepseek_projection(&source).ok_or_else(|| {
            Error::Parallel(format!("unknown DeepSeek tensor-parallel tensor {source}"))
        })?;
        plan.insert(
            destination,
            projection_tensor_placement(&source, projection, plan.topology()),
        );
    }
    if args.is_moe_layer(index as i32) {
        insert_deepseek_expert_plan(plan, args, index, dense_source)?;
    }
    Ok(())
}

fn insert_deepseek_expert_plan(
    plan: &mut PlacementPlan,
    args: &deepseek_v3::ModelArgs,
    layer: usize,
    dense_source: bool,
) -> Result<(), Error> {
    let components: &[&str] = if dense_source {
        &["weight"]
    } else if args.native_fp8_config().is_some() {
        &["weight", "weight_scale_inv"]
    } else if let Some(quantization) = args.affine_quantization()? {
        if quantization.has_biases() {
            &["weight", "scales", "biases"]
        } else {
            &["weight", "scales"]
        }
    } else {
        &["weight"]
    };
    for expert in 0..args.n_routed_experts {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for component in components {
                let row = projection == "down_proj";
                let axis = usize::from(row);
                let logical = args.moe_intermediate_size;
                let alignment = if args.native_fp8_config().is_some() {
                    128
                } else if row {
                    args.affine_quantization()?.map_or(1, |q| q.group_size())
                } else {
                    1
                };
                if logical % i32::try_from(plan.topology().tensor_parallel_size).unwrap() != 0
                    || logical / i32::try_from(plan.topology().tensor_parallel_size).unwrap()
                        % alignment
                        != 0
                {
                    return Err(Error::Parallel(format!(
                        "tensor model.layers.{layer}.mlp.experts.{expert}.{projection}.{component} dimension {logical} with block/group size {alignment} cannot be sharded across TP size {}",
                        plan.topology().tensor_parallel_size
                    )));
                }
                plan.insert(
                    format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.{component}"),
                    TensorPlacement::Shard {
                        axis,
                        index: plan.topology().tensor_parallel_rank,
                        parts: plan.topology().tensor_parallel_size,
                    },
                );
            }
        }
    }
    Ok(())
}

fn insert_vocabulary_plan(
    plan: &mut PlacementPlan,
    module: &impl ModuleParameters,
    prefix: &str,
    range: &Range<usize>,
) {
    for destination in parameter_names(module, prefix) {
        plan.insert(
            destination,
            TensorPlacement::Range {
                axis: 0,
                start: range.start,
                end: range.end,
            },
        );
    }
}

fn partition_info(
    partition: &RankPartition,
    topology: ParallelTopology,
    kind: ModelKind,
    heads: (i32, i32),
    local_heads: (i32, i32),
    vocabulary: Range<usize>,
) -> TensorParallelInfo {
    let mut owned_tensors = partition
        .tensors()
        .map(|(name, _)| checkpoint_name(name))
        .collect::<Vec<_>>();
    owned_tensors.sort();
    TensorParallelInfo {
        global_rank: topology.global_rank,
        tensor_parallel_rank: topology.tensor_parallel_rank,
        tensor_parallel_size: topology.tensor_parallel_size,
        model_kind: kind,
        global_attention_heads: heads.0,
        local_attention_heads: local_heads.0,
        global_kv_heads: heads.1,
        local_kv_heads: local_heads.1,
        local_vocabulary_range: vocabulary,
        owned_tensors,
        local_parameter_bytes: partition.tensors().map(|(_, value)| value.nbytes()).sum(),
        opened_checkpoint_shards: partition.opened_shards().to_vec(),
    }
}

/// Loads an executable rank-local pure tensor-parallel model.
pub fn load_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    load_tensor_parallel_model_with_options(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        stream,
        weights_stream,
    )
}

/// Loads an executable rank-local pure tensor-parallel model with options.
pub fn load_tensor_parallel_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let model_dir = model_dir.as_ref();
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("tensor-parallel loading requires ModelLoadOptions::parallel".into())
    })?;
    validate_pure_tensor(topology)?;
    topology.validate_execution_stream(stream)?;
    if model_dir
        .extension()
        .is_some_and(|extension| extension == "gguf")
    {
        return Err(Error::Parallel(
            "tensor-parallel GGUF loading is unsupported because bounded local-range selection is unavailable; use safetensors"
                .into(),
        ));
    }
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    match config.get("model_type").and_then(serde_json::Value::as_str) {
        Some("llama" | "mistral") => {
            load_llama(model_dir, topology, options.quantization, stream, weights_stream)
        }
        Some("deepseek_v3") => {
            load_deepseek(model_dir, topology, options.quantization, stream, weights_stream)
        }
        Some(model_type) => Err(Error::UnsupportedArchitecture(format!(
            "tensor-parallel execution supports Llama-compatible and DeepSeek-V3/R1 text models, not {model_type}"
        ))),
        None => Err(Error::UnsupportedArchitecture(
            "tensor-parallel model config is missing model_type".into(),
        )),
    }
}

fn load_partition(
    model_dir: &Path,
    plan: &PlacementPlan,
    weights_stream: &Stream,
    stream: &Stream,
    strict: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    load_safetensors_partition_on_streams(model_dir, plan, weights_stream, stream, strict)
}

fn load_llama(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = llama::get_llama_model_args(model_dir)?;
    if source_args.hidden_size <= 0 || source_args.head_dim <= 0 {
        return Err(Error::Parallel(
            "Llama hidden_size and head_dim must be positive".into(),
        ));
    }
    if source_args.hidden_size != source_args.num_attention_heads * source_args.head_dim {
        return Err(Error::Parallel(format!(
            "Llama hidden size {} does not match {} attention heads * head dimension {}",
            source_args.hidden_size, source_args.num_attention_heads, source_args.head_dim
        )));
    }
    let local_heads = exact_division(
        "Llama attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "Llama KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    let local_intermediate = exact_division(
        "Llama intermediate size",
        source_args.intermediate_size,
        topology.tensor_parallel_size,
    )?;
    if local_heads % local_kv_heads != 0 {
        return Err(Error::Parallel(
            "Llama local query heads must be divisible by local KV heads for GQA".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Llama tensor parallel",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if let Some(quantization) = quantize_on_load.or(source_args.weight_quantization()) {
        require_alignment(
            "model.layers.*.self_attn.o_proj.weight",
            local_heads * source_args.head_dim,
            quantization.group_size(),
            topology,
        )?;
        require_alignment(
            "model.layers.*.mlp.down_proj.weight",
            local_intermediate,
            quantization.group_size(),
            topology,
        )?;
    }
    let vocabulary = balanced_contiguous_range(
        source_args.vocab_size as usize,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let mut plan = PlacementPlan::new(topology);
    let source_embedding = linear::unloaded_maybe_quantized_embedding(
        source_args.vocab_size,
        source_args.hidden_size,
        source_args.affine_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    insert_vocabulary_plan(
        &mut plan,
        &source_embedding,
        "model.embed_tokens",
        &vocabulary,
    );
    for index in 0..source_args.num_hidden_layers as usize {
        let layer = llama::TransformerBlock::new_for_layer(&source_args, index as i32, stream)?;
        insert_llama_layer_plan(&mut plan, &layer, index)?;
    }
    let source_norm = nn::RmsNorm::unloaded(
        source_args.hidden_size,
        source_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    for name in parameter_names(&source_norm, "model.norm") {
        plan.insert(name, TensorPlacement::Replicated);
    }
    if !source_args.tie_word_embeddings {
        let source_head = linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
            source_args.hidden_size,
            source_args.vocab_size,
            source_args.affine_quantization_for("lm_head.weight"),
            stream,
        )?;
        insert_vocabulary_plan(&mut plan, &source_head, "lm_head", &vocabulary);
    }
    let partition = load_partition(
        model_dir,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::Llama,
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.num_key_value_heads = local_kv_heads;
    target_args.intermediate_size = local_intermediate;
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
    }
    let local_vocab = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local vocabulary does not fit in i32".into()))?;
    let mut embedding = linear::unloaded_maybe_quantized_embedding(
        local_vocab,
        target_args.hidden_size,
        target_args.affine_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    let mut layers = (0..target_args.num_hidden_layers)
        .map(|index| llama::TransformerBlock::new_for_layer(&target_args, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let mut norm = nn::RmsNorm::unloaded(
        target_args.hidden_size,
        target_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    let mut lm_head = if target_args.tie_word_embeddings {
        None
    } else {
        Some(
            linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                target_args.hidden_size,
                local_vocab,
                target_args.affine_quantization_for("lm_head.weight"),
                stream,
            )?,
        )
    };
    let mut tensors = partition.into_tensors();
    assign_module(
        &mut embedding,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
    }
    assign_module(&mut norm, "model.norm", &mut tensors, None, stream)?;
    if let Some(head) = lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::Llama(LlamaTensorModel {
            global_args: source_args,
            embedding,
            layers,
            norm,
            lm_head,
        }),
    })
}

fn load_deepseek(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = deepseek_v3::get_model_args(model_dir)?;
    if requested_quantization.is_some() && source_args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 tensor-parallel weights cannot be implicitly requantized"
                .into(),
        ));
    }
    let local_heads = exact_division(
        "DeepSeek attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_intermediate = exact_division(
        "DeepSeek dense intermediate size",
        source_args.intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let local_moe_intermediate = exact_division(
        "DeepSeek expert intermediate size",
        source_args.moe_intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "DeepSeek tensor parallel",
                source_args.affine_quantization()?,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if source_args.native_fp8_config().is_some() {
        for (tensor, dimension) in [
            (
                "model.layers.*.self_attn.q_proj.weight",
                local_heads * (source_args.qk_nope_head_dim + source_args.qk_rope_head_dim),
            ),
            (
                "model.layers.*.self_attn.kv_b_proj.weight",
                local_heads * (source_args.qk_nope_head_dim + source_args.v_head_dim),
            ),
            (
                "model.layers.*.self_attn.o_proj.weight",
                local_heads * source_args.v_head_dim,
            ),
            ("model.layers.*.mlp.down_proj.weight", local_intermediate),
            (
                "model.layers.*.mlp.experts.*.down_proj.weight",
                local_moe_intermediate,
            ),
        ] {
            require_alignment(tensor, dimension, 128, topology)?;
        }
    }
    if let Some(quantization) = quantize_on_load.or(source_args.affine_quantization()?) {
        for (tensor, dimension) in [
            (
                "model.layers.*.self_attn.o_proj.weight",
                local_heads * source_args.v_head_dim,
            ),
            ("model.layers.*.mlp.down_proj.weight", local_intermediate),
            (
                "model.layers.*.mlp.experts.*.down_proj.weight",
                local_moe_intermediate,
            ),
        ] {
            require_alignment(tensor, dimension, quantization.group_size(), topology)?;
        }
    }
    let vocabulary = balanced_contiguous_range(
        source_args.vocab_size as usize,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let mut plan = PlacementPlan::new(topology);
    let source_embedding = linear::unloaded_maybe_quantized_embedding(
        source_args.vocab_size,
        source_args.hidden_size,
        source_args.weight_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    insert_vocabulary_plan(
        &mut plan,
        &source_embedding,
        "model.embed_tokens",
        &vocabulary,
    );
    for index in 0..source_args.num_hidden_layers as usize {
        let layer = deepseek_v3::DecoderLayer::new(&source_args, index as i32, stream)?;
        insert_deepseek_layer_plan(
            &mut plan,
            &source_args,
            &layer,
            index,
            quantize_on_load.is_some(),
        )?;
    }
    let source_norm = nn::RmsNorm::unloaded(
        source_args.hidden_size,
        source_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    for name in parameter_names(&source_norm, "model.norm") {
        plan.insert(name, TensorPlacement::Replicated);
    }
    let source_head = linear::unloaded_maybe_quantized_linear(
        source_args.hidden_size,
        source_args.vocab_size,
        false,
        source_args.weight_quantization_for("lm_head.weight"),
        stream,
    )?;
    insert_vocabulary_plan(&mut plan, &source_head, "lm_head", &vocabulary);
    let mut strict = StrictLoadConfig::default();
    for index in 0..source_args.num_nextn_predict_layers {
        strict = strict.allow_unused_prefix(format!(
            "model.layers.{}.",
            source_args.num_hidden_layers + index
        ));
    }
    let partition = load_partition(model_dir, &plan, weights_stream, stream, &strict)?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::DeepSeekV3,
        (
            source_args.num_attention_heads,
            source_args.num_attention_heads,
        ),
        (local_heads, local_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.intermediate_size = local_intermediate;
    target_args.moe_intermediate_size = local_moe_intermediate;
    if let Some(quantization) = quantize_on_load {
        target_args.quantization_config = None;
        target_args.quantization = Some(quantization);
    }
    let local_vocab = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local vocabulary does not fit in i32".into()))?;
    let mut embedding = linear::unloaded_maybe_quantized_embedding(
        local_vocab,
        target_args.hidden_size,
        target_args.weight_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    let mut layers = (0..target_args.num_hidden_layers)
        .map(|index| deepseek_v3::DecoderLayer::new(&target_args, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let mut norm = nn::RmsNorm::unloaded(
        target_args.hidden_size,
        target_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    let mut lm_head = linear::unloaded_maybe_quantized_linear(
        target_args.hidden_size,
        local_vocab,
        false,
        target_args.weight_quantization_for("lm_head.weight"),
        stream,
    )?;
    let mut tensors = partition.into_tensors();
    assign_module(
        &mut embedding,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
        if let Some(moe) = layer.mlp.moe_mut() {
            load_deepseek_experts(
                moe,
                index,
                (
                    target_args.n_routed_experts,
                    target_args.hidden_size,
                    target_args.moe_intermediate_size,
                ),
                &mut tensors,
                quantize_on_load,
                stream,
            )?;
        }
    }
    assign_module(&mut norm, "model.norm", &mut tensors, None, stream)?;
    assign_module(
        &mut lm_head,
        "lm_head",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::DeepSeek(DeepSeekTensorModel {
            global_args: source_args,
            embedding,
            layers,
            norm,
            lm_head,
        }),
    })
}

fn ensure_no_unused(tensors: HashMap<String, Array>) -> Result<(), Error> {
    if tensors.is_empty() {
        Ok(())
    } else {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        })
    }
}

fn vocabulary_embedding(
    embedding: &mut MaybeQuantized<nn::Embedding>,
    tokens: &Array,
    range: &Range<usize>,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    let start = Array::from_int(
        i32::try_from(range.start)
            .map_err(|_| Error::Parallel("vocabulary range start does not fit in i32".into()))?,
    );
    let end = Array::from_int(
        i32::try_from(range.end)
            .map_err(|_| Error::Parallel("vocabulary range end does not fit in i32".into()))?,
    );
    let valid = tokens
        .ge(&start, stream)?
        .logical_and(tokens.lt(&end, stream)?, stream)?;
    let local_ids = tokens.subtract(&start, stream)?;
    let safe_ids = safemlx::ops::r#where(&valid, &local_ids, Array::from_int(0), stream)?;
    let local = embedding.forward(&safe_ids, stream)?;
    let valid = valid.expand_dims(-1, stream)?;
    let local = safemlx::ops::r#where(&valid, &local, zeros_like(&local, stream)?, stream)?;
    Ok(distributed::all_sum(&local, group, stream)?)
}

fn partial_projection(
    projection: &mut MaybeQuantized<nn::Linear>,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(match projection {
        MaybeQuantized::Original(linear) => {
            safemlx::ops::matmul(input, linear.weight.value.transpose(stream)?, stream)?
        }
        MaybeQuantized::Quantized(linear) => quantized_matmul_with_mode(
            input,
            &linear.inner.weight,
            &linear.scales,
            linear.biases.value.as_ref(),
            true,
            linear.group_size,
            linear.bits,
            linear.mode,
            stream,
        )?,
    })
}

fn add_projection_bias(
    projection: &MaybeQuantized<nn::Linear>,
    output: Array,
    stream: &Stream,
) -> Result<Array, Error> {
    let bias = match projection {
        MaybeQuantized::Original(linear) => linear.bias.value.as_ref(),
        MaybeQuantized::Quantized(linear) => linear.inner.bias.value.as_ref(),
    };
    Ok(match bias {
        Some(bias) => output.add(bias, stream)?,
        None => output,
    })
}

fn llama_attention<C: KeyValueCache>(
    attention: &mut llama::Attention,
    x: &Array,
    mask: Option<&Array>,
    cache: Option<&mut C>,
    generated_sliding_window: Option<i32>,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    let (batch, sequence) = batch_seq(x);
    let queries = attention.q_proj.forward(x, stream)?;
    let keys = attention.k_proj.forward(x, stream)?;
    let values = attention.v_proj.forward(x, stream)?;
    let queries =
        reshape_attention_projection(queries, batch, sequence, attention.n_heads, stream)?;
    let keys = reshape_attention_projection(keys, batch, sequence, attention.n_kv_heads, stream)?;
    let values =
        reshape_attention_projection(values, batch, sequence, attention.n_kv_heads, stream)?;
    let offset = cache.as_ref().map_or(0, |cache| cache.offset());
    let mut cache = cache;
    let (queries, keys, values) = apply_rope_and_update_cache(
        &mut attention.rope,
        queries,
        keys,
        values,
        &mut cache,
        stream,
    )?;
    let attended = if let Some(window) = generated_sliding_window.filter(|_| sequence > 1) {
        sliding_window_prefill_attention(
            queries,
            keys,
            values,
            attention.scale,
            window,
            offset,
            batch,
            sequence,
            stream,
        )?
    } else {
        finish_attention(
            queries,
            keys,
            values,
            cache,
            attention.scale,
            mask,
            batch,
            sequence,
            stream,
        )?
    };
    let partial = partial_projection(&mut attention.o_proj, &attended, stream)?;
    let output = distributed::all_sum(&partial, group, stream)?;
    add_projection_bias(&attention.o_proj, output, stream)
}

fn llama_mlp(
    mlp: &mut llama::Mlp,
    x: &Array,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    let gate = silu(mlp.gate_proj.forward(x, stream)?, stream)?;
    let up = mlp.up_proj.forward(x, stream)?;
    let sharded = gate.multiply(up, stream)?;
    let partial = partial_projection(&mut mlp.down_proj, &sharded, stream)?;
    let output = distributed::all_sum(&partial, group, stream)?;
    add_projection_bias(&mlp.down_proj, output, stream)
}

fn forward_llama(
    model: &mut LlamaTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    caches: &mut [TensorParallelLlamaLayerCache],
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if caches.len() != model.layers.len() {
        return Err(Error::Parallel(format!(
            "Llama TP cache has {} layers, expected {}",
            caches.len(),
            model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.embedding,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let offset = caches.first().map_or(0, |cache| match cache {
        TensorParallelLlamaLayerCache::Standard(cache) => cache.offset(),
        TensorParallelLlamaLayerCache::Sliding(cache) => cache.offset(),
    });
    let generated_sliding_window = (explicit_mask.is_none() && sequence > 1)
        .then_some(model.global_args.sliding_window)
        .flatten();
    let generated_mask = if explicit_mask.is_some() || generated_sliding_window.is_some() {
        None
    } else if let Some(window) = model.global_args.sliding_window {
        let retained = offset.min(window);
        let keys = retained + sequence;
        ((sequence > 1 || keys > window) && keys > 1)
            .then(|| create_causal_mask(sequence, Some(retained), Some(window - 1), None, stream))
            .transpose()?
    } else {
        (sequence > 1)
            .then(|| create_causal_mask(sequence, Some(offset), None, None, stream))
            .transpose()?
    };
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model.layers.iter_mut().zip(caches) {
        let normalized = layer.input_layernorm.forward(&hidden, stream)?;
        let attention = match cache {
            TensorParallelLlamaLayerCache::Standard(cache) => llama_attention(
                &mut layer.self_attn,
                &normalized,
                mask,
                Some(cache),
                generated_sliding_window,
                group,
                stream,
            )?,
            TensorParallelLlamaLayerCache::Sliding(cache) => llama_attention(
                &mut layer.self_attn,
                &normalized,
                mask,
                Some(cache),
                generated_sliding_window,
                group,
                stream,
            )?,
        };
        hidden = hidden.add(attention, stream)?;
        let normalized = layer.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = llama_mlp(&mut layer.mlp, &normalized, group, stream)?;
        hidden = hidden.add(mlp, stream)?;
    }
    let hidden = model.norm.forward(&hidden, stream)?;
    match model.lm_head.as_mut() {
        Some(head) => Ok(head.forward(&hidden, stream)?),
        None => match &mut model.embedding {
            MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
            MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
        },
    }
}

fn forward_deepseek(
    model: &mut DeepSeekTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    caches: &mut [CompressedLatentCache],
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if caches.len() != model.layers.len() {
        return Err(Error::Parallel(format!(
            "DeepSeek TP cache has {} layers, expected {}",
            caches.len(),
            model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.embedding,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let offset = caches.first().map_or(0, CompressedLatentCache::offset);
    let generated_mask = (explicit_mask.is_none() && sequence > 1 && offset > 0)
        .then(|| create_causal_mask(sequence, Some(offset), None, None, stream))
        .transpose()?;
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model.layers.iter_mut().zip(caches) {
        hidden = layer.forward_tensor_parallel(&hidden, mask, Some(cache), group, stream)?;
    }
    let hidden = model.norm.forward(&hidden, stream)?;
    Ok(model.lm_head.forward(&hidden, stream)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::DeviceAssignment;
    use safemlx::DeviceType;

    fn topology(world: usize, rank: usize, tp: usize) -> ParallelTopology {
        ParallelTopology::from_rank(
            world,
            rank,
            tp,
            1,
            1,
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
        .unwrap()
    }

    #[test]
    fn pure_tensor_topology_rejects_hybrids_before_loading() {
        assert!(validate_pure_tensor(topology(2, 0, 2)).is_ok());
        let hybrid =
            ParallelTopology::from_rank(4, 0, 2, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(validate_pure_tensor(hybrid)
            .unwrap_err()
            .to_string()
            .contains("hybrid"));
    }

    #[test]
    fn uneven_vocabulary_widths_cover_global_vocabulary() {
        assert_eq!(vocabulary_widths(11, 3).unwrap(), vec![4, 4, 3]);
    }
}
