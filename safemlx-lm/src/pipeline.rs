//! Executable serial pipeline parallelism for decoder-only language models.
//!
//! A [`PipelineModel`] owns only one balanced contiguous decoder-layer range
//! and the boundary modules required by its explicit stage role. Communication
//! groups are borrowed for each operation and are never retained by model
//! state.

use std::{
    collections::HashMap,
    ops::Range,
    path::{Path, PathBuf},
};

use safemlx::{
    distributed::{self, Group},
    module::{Module, ModuleParameters},
    nn,
    ops::indexing::TryIndexOp,
    ops::{ones, quantized_packed_dimension, stack_axis, zeros},
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    cache::{CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, SlidingKeyValueCache},
    error::Error,
    inspection::ActivationObserver,
    models::{
        common::{linear, linear::project_logits_maybe_quantized},
        deepseek_v3, llama, ModelKind, ModelLoadOptions,
    },
    parallel::{
        load_safetensors_partition_on_streams, ParallelTopology, PlacementPlan, RankPartition,
        TensorPlacement,
    },
    quantization::{quantize_tensor, WeightQuantization},
    sampler::Sampler,
    utils::create_causal_mask,
    weights::StrictLoadConfig,
};

/// Immutable, inspectable description of the local pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStageInfo {
    /// Rank in the distributed group.
    pub global_rank: usize,
    /// Zero-based pipeline coordinate.
    pub pipeline_stage: usize,
    /// Number of pipeline stages.
    pub pipeline_stages: usize,
    /// Whether this stage performs token embedding.
    pub is_first: bool,
    /// Whether this stage performs final normalization and projection.
    pub is_last: bool,
    /// Global decoder-layer indices owned by this stage.
    pub global_layer_range: Range<usize>,
    /// Previous stage's global rank, if any.
    pub predecessor_rank: Option<usize>,
    /// Next stage's global rank, if any.
    pub successor_rank: Option<usize>,
    /// Architecture adapter used by the stage.
    pub model_kind: ModelKind,
    /// Decoder hidden width.
    pub hidden_size: i32,
    /// Dtype used for transferred hidden activations.
    pub activation_dtype: Dtype,
    /// Checkpoint tensors selected for this rank.
    pub owned_tensors: Vec<String>,
    /// Total uncompressed bytes of locally selected checkpoint tensors.
    pub local_parameter_bytes: usize,
    /// Payload shards actually opened for this rank.
    pub opened_checkpoint_shards: Vec<PathBuf>,
}

/// Shape metadata shared by every rank for one pipeline operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PipelineStep {
    /// Batch dimension.
    batch_size: i32,
    /// Sequence dimension (prompt length or one for decode).
    sequence_length: i32,
}

impl PipelineStep {
    /// Creates validated positive step dimensions.
    pub fn new(batch_size: i32, sequence_length: i32) -> Result<Self, Error> {
        if batch_size <= 0 || sequence_length <= 0 {
            return Err(Error::Parallel(format!(
                "pipeline batch and sequence dimensions must be positive, got [{batch_size}, {sequence_length}]"
            )));
        }
        Ok(Self {
            batch_size,
            sequence_length,
        })
    }

    /// Returns the validated batch dimension.
    pub const fn batch_size(self) -> i32 {
        self.batch_size
    }

    /// Returns the validated sequence dimension.
    pub const fn sequence_length(self) -> i32 {
        self.sequence_length
    }

    fn activation_shape(self, hidden_size: i32) -> [i32; 3] {
        [self.batch_size, self.sequence_length, hidden_size]
    }
}

/// Explicit input to stage-local execution.
pub enum PipelineStageInput<'a> {
    /// Integer token ids for stage zero.
    Tokens(&'a Array),
    /// Hidden activations for every later stage.
    Hidden(&'a Array),
}

/// Result of one stage-local forward operation.
#[derive(Debug)]
pub enum PipelineStageOutput {
    /// Hidden activations to transfer to the next stage.
    Hidden(Array),
    /// Vocabulary logits produced only by the final stage.
    Logits(Array),
}

/// One globally identified Llama cache entry.
#[derive(Debug, Clone)]
pub enum PipelineLlamaLayerCache {
    /// Unbounded concatenating KV cache.
    Standard {
        /// Global decoder-layer index.
        global_layer: usize,
        /// Layer-local cache state.
        cache: ConcatKeyValueCache,
    },
    /// Bounded sliding-window KV cache.
    Sliding {
        /// Global decoder-layer index.
        global_layer: usize,
        /// Layer-local cache state.
        cache: SlidingKeyValueCache,
    },
}

/// One globally identified DeepSeek compressed-latent cache entry.
#[derive(Debug, Clone)]
pub struct PipelineDeepSeekLayerCache {
    /// Global decoder-layer index.
    pub global_layer: usize,
    /// Layer-local MLA cache state.
    pub cache: CompressedLatentCache,
}

/// Architecture-checked stage-local inference caches.
#[derive(Debug, Clone)]
pub enum PipelineCache {
    /// Llama standard or sliding-window cache entries.
    Llama(Vec<PipelineLlamaLayerCache>),
    /// DeepSeek compressed-latent cache entries.
    DeepSeek(Vec<PipelineDeepSeekLayerCache>),
}

impl PipelineCache {
    /// Returns the global decoder-layer ids represented locally.
    pub fn global_layers(&self) -> Vec<usize> {
        match self {
            Self::Llama(layers) => layers
                .iter()
                .map(|layer| match layer {
                    PipelineLlamaLayerCache::Standard { global_layer, .. }
                    | PipelineLlamaLayerCache::Sliding { global_layer, .. } => *global_layer,
                })
                .collect(),
            Self::DeepSeek(layers) => layers.iter().map(|layer| layer.global_layer).collect(),
        }
    }

    /// Clears retained state without changing local layer ownership.
    pub fn reset(&mut self) {
        match self {
            Self::Llama(layers) => {
                for layer in layers {
                    match layer {
                        PipelineLlamaLayerCache::Standard { cache, .. } => cache.clear(),
                        PipelineLlamaLayerCache::Sliding { cache, .. } => cache.clear(),
                    }
                }
            }
            Self::DeepSeek(layers) => {
                for layer in layers {
                    layer.cache.clear();
                }
            }
        }
    }
}

struct LlamaStage {
    args: llama::ModelArgs,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<llama::TransformerBlock>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

struct DeepSeekStage {
    args: deepseek_v3::ModelArgs,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<deepseek_v3::DecoderLayer>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

enum ArchitectureStage {
    Llama(LlamaStage),
    DeepSeek(DeepSeekStage),
}

/// An executable, rank-local piece of a pipeline-parallel model.
pub struct PipelineModel {
    topology: ParallelTopology,
    info: PipelineStageInfo,
    stage: ArchitectureStage,
}

impl std::fmt::Debug for PipelineModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PipelineModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl PipelineModel {
    /// Returns the immutable stage description.
    pub fn stage_info(&self) -> &PipelineStageInfo {
        &self.info
    }

    /// Allocates cache entries only for locally owned decoder layers.
    pub fn new_cache(&self) -> PipelineCache {
        match &self.stage {
            ArchitectureStage::Llama(stage) => PipelineCache::Llama(
                stage
                    .range
                    .clone()
                    .map(|global_layer| match stage.args.sliding_window {
                        Some(window) => PipelineLlamaLayerCache::Sliding {
                            global_layer,
                            cache: SlidingKeyValueCache::new(window),
                        },
                        None => PipelineLlamaLayerCache::Standard {
                            global_layer,
                            cache: ConcatKeyValueCache::new(),
                        },
                    })
                    .collect(),
            ),
            ArchitectureStage::DeepSeek(stage) => PipelineCache::DeepSeek(
                stage
                    .range
                    .clone()
                    .map(|global_layer| PipelineDeepSeekLayerCache {
                        global_layer,
                        cache: CompressedLatentCache::new(),
                    })
                    .collect(),
            ),
        }
    }

    /// Executes only this stage, without communication.
    ///
    /// This operation is useful for deterministic single-process composition
    /// tests and custom schedulers. Distributed callers normally use
    /// [`Self::forward_pipeline`].
    pub fn forward_stage(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.topology.validate_execution_stream(stream)?;
        validate_stage_input(&self.info, &input, step)?;
        match (&mut self.stage, cache) {
            (ArchitectureStage::Llama(stage), PipelineCache::Llama(cache)) => {
                stage.forward(input, step, mask, cache, stream)
            }
            (ArchitectureStage::DeepSeek(stage), PipelineCache::DeepSeek(cache)) => {
                stage.forward(input, step, mask, cache, stream)
            }
            (ArchitectureStage::Llama(_), _) => Err(Error::Parallel(
                "pipeline cache architecture is DeepSeek, but model stage is Llama".into(),
            )),
            (ArchitectureStage::DeepSeek(_), _) => Err(Error::Parallel(
                "pipeline cache architecture is Llama, but model stage is DeepSeek".into(),
            )),
        }
    }

    /// Runs one serial distributed pipeline operation.
    ///
    /// Stage zero embeds and sends, intermediate stages receive/execute/send,
    /// and the final stage receives and returns logits. Every lazy point-to-
    /// point operation is evaluated and synchronized before the operation
    /// returns.
    pub fn forward_pipeline(
        &mut self,
        tokens: Option<&Array>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        self.validate_group(group)?;
        let received;
        let input =
            if self.info.is_first {
                PipelineStageInput::Tokens(tokens.ok_or_else(|| {
                    Error::Parallel("pipeline stage zero requires token ids".into())
                })?)
            } else {
                if tokens.is_some() {
                    return Err(Error::Parallel(format!(
                    "pipeline stage {} receives hidden activations and must not receive token ids",
                    self.info.pipeline_stage
                )));
                }
                let peer = self.info.predecessor_rank.expect("non-first predecessor");
                received = distributed::recv(
                    &step.activation_shape(self.info.hidden_size),
                    self.info.activation_dtype,
                    peer,
                    group,
                    stream,
                )
                .map_err(|error| {
                    Error::Parallel(format!(
                    "stage {} failed to receive {:?} {:?} activations from rank {peer}: {error}",
                    self.info.pipeline_stage,
                    step.activation_shape(self.info.hidden_size),
                    self.info.activation_dtype
                ))
                })?;
                eval([&received])?;
                stream.synchronize()?;
                PipelineStageInput::Hidden(&received)
            };

        match self.forward_stage(input, step, mask, cache, stream)? {
            PipelineStageOutput::Hidden(hidden) => {
                let expected = step.activation_shape(self.info.hidden_size);
                if hidden.shape() != expected || hidden.dtype() != self.info.activation_dtype {
                    return Err(Error::Parallel(format!(
                        "stage {} produced activations shaped {:?} with {:?}, expected {expected:?} with {:?}",
                        self.info.pipeline_stage,
                        hidden.shape(),
                        hidden.dtype(),
                        self.info.activation_dtype
                    )));
                }
                let peer = self.info.successor_rank.expect("non-final successor");
                let sent = distributed::send(&hidden, peer, group, stream).map_err(|error| {
                    Error::Parallel(format!(
                        "stage {} failed to send {:?} {:?} activations to rank {peer}: {error}",
                        self.info.pipeline_stage,
                        hidden.shape(),
                        hidden.dtype()
                    ))
                })?;
                eval([&sent])?;
                stream.synchronize()?;
                Ok(None)
            }
            PipelineStageOutput::Logits(logits) => Ok(Some(logits)),
        }
    }

    /// Samples on the last stage and broadcasts only the selected token and
    /// EOS/stop state via identically ordered all-sums.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        logits: Option<&Array>,
        step: PipelineStep,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        group: &Group,
        stream: &Stream,
    ) -> Result<SynchronizedToken, Error> {
        self.validate_group(group)?;
        let local_token = if self.info.is_last {
            let logits = logits.ok_or_else(|| {
                Error::Parallel("the last pipeline stage requires logits for sampling".into())
            })?;
            if logits.dim(0) != step.batch_size {
                return Err(Error::Parallel(format!(
                    "last-stage logits batch {} does not match pipeline batch {}",
                    logits.dim(0),
                    step.batch_size
                )));
            }
            let sampling_logits = if logits.ndim() == 3 {
                logits.try_index_device((.., -1, ..), stream)?
            } else {
                logits.clone()
            };
            sampler
                .sample(&sampling_logits, temperature, prng_state, stream)?
                .reshape(&[step.batch_size, 1], stream)?
        } else {
            if logits.is_some() {
                return Err(Error::Parallel(
                    "only the last pipeline stage may supply logits".into(),
                ));
            }
            zeros::<u32>(&[step.batch_size, 1], stream)?
        };
        let token = distributed::all_sum(&local_token, group, stream)?;
        let local_finished = if self.info.is_last && finished {
            ones::<i32>(&[], stream)?
        } else {
            zeros::<i32>(&[], stream)?
        };
        let finished = distributed::all_sum(&local_finished, group, stream)?;
        eval([&token, &finished])?;
        stream.synchronize()?;
        let finished = finished.try_item::<i32>(stream)? != 0;
        Ok(SynchronizedToken { token, finished })
    }

    fn validate_group(&self, group: &Group) -> Result<(), Error> {
        if group.rank() != self.topology.global_rank || group.size() != self.topology.world_size {
            return Err(Error::Parallel(format!(
                "pipeline topology expects group rank {}/{} but received rank {}/{}",
                self.topology.global_rank,
                self.topology.world_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }
}

/// Sampled token and globally synchronized termination state.
#[derive(Debug)]
pub struct SynchronizedToken {
    /// Selected token id array; full logits are never broadcast.
    pub token: Array,
    /// Whether every rank should terminate generation.
    pub finished: bool,
}

fn validate_pure_pipeline(topology: ParallelTopology) -> Result<(), Error> {
    if topology.pipeline_parallel_size <= 1 {
        return Err(Error::Parallel(
            "pipeline loading requires pipeline_parallel_size > 1".into(),
        ));
    }
    if topology.tensor_parallel_size != 1 || topology.expert_parallel_size != 1 {
        return Err(Error::Parallel(format!(
            "pipeline execution currently supports pure PP only (TP=1, EP=1), got TP={}, PP={}, EP={}",
            topology.tensor_parallel_size,
            topology.pipeline_parallel_size,
            topology.expert_parallel_size
        )));
    }
    Ok(())
}

fn rank_for_stage(topology: ParallelTopology, stage: usize) -> usize {
    (stage * topology.tensor_parallel_size) * topology.expert_parallel_size
}

fn base_info(
    topology: ParallelTopology,
    range: Range<usize>,
    model_kind: ModelKind,
    hidden_size: i32,
) -> PipelineStageInfo {
    let stage = topology.pipeline_parallel_rank;
    let last = topology.pipeline_parallel_size - 1;
    PipelineStageInfo {
        global_rank: topology.global_rank,
        pipeline_stage: stage,
        pipeline_stages: topology.pipeline_parallel_size,
        is_first: stage == 0,
        is_last: stage == last,
        global_layer_range: range,
        predecessor_rank: (stage > 0).then(|| rank_for_stage(topology, stage - 1)),
        successor_rank: (stage < last).then(|| rank_for_stage(topology, stage + 1)),
        model_kind,
        hidden_size,
        activation_dtype: Dtype::Float32,
        owned_tensors: Vec::new(),
        local_parameter_bytes: 0,
        opened_checkpoint_shards: Vec::new(),
    }
}

fn owns_embedding_weight(info: &PipelineStageInfo, tied: bool) -> bool {
    info.is_first || (tied && info.is_last)
}

fn validate_stage_input(
    info: &PipelineStageInfo,
    input: &PipelineStageInput<'_>,
    step: PipelineStep,
) -> Result<(), Error> {
    match (info.is_first, input) {
        (true, PipelineStageInput::Tokens(tokens)) => {
            if tokens.ndim() != 2 || tokens.shape() != [step.batch_size, step.sequence_length] {
                return Err(Error::Parallel(format!(
                    "first stage expected token ids shaped [{}, {}], got {:?}",
                    step.batch_size,
                    step.sequence_length,
                    tokens.shape()
                )));
            }
        }
        (false, PipelineStageInput::Hidden(hidden)) => {
            validate_hidden_metadata(info, hidden.shape(), hidden.dtype(), step)?;
        }
        (true, PipelineStageInput::Hidden(_)) => {
            return Err(Error::Parallel(
                "first stage requires token ids, not hidden states".into(),
            ))
        }
        (false, PipelineStageInput::Tokens(_)) => {
            return Err(Error::Parallel(format!(
                "pipeline stage {} requires hidden states, not token ids",
                info.pipeline_stage
            )))
        }
    }
    Ok(())
}

fn validate_hidden_metadata(
    info: &PipelineStageInfo,
    shape: &[i32],
    dtype: Dtype,
    step: PipelineStep,
) -> Result<(), Error> {
    let expected = step.activation_shape(info.hidden_size);
    if shape != expected {
        return Err(Error::Parallel(format!(
            "stage {} expected hidden activations shaped {expected:?}, got {shape:?}",
            info.pipeline_stage
        )));
    }
    if dtype != info.activation_dtype {
        return Err(Error::Parallel(format!(
            "stage {} expected {:?} activations, got {:?}",
            info.pipeline_stage, info.activation_dtype, dtype
        )));
    }
    Ok(())
}

fn full_parameter_names(module: &impl ModuleParameters, prefix: &str) -> Vec<String> {
    module
        .parameters()
        .flatten()
        .keys()
        .map(|name| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        })
        .collect()
}

fn checkpoint_name(parameter_name: &str) -> String {
    crate::module_binding::canonical_checkpoint_name(parameter_name)
}

fn infer_activation_dtype(partition: &RankPartition) -> Dtype {
    partition
        .tensors()
        .map(|(_, value)| value)
        .find(|value| value.ndim() >= 2 && value.dtype().is_float())
        .map_or(Dtype::Float32, Array::dtype)
}

fn insert_module_plan(
    plan: &mut PlacementPlan,
    module: &impl ModuleParameters,
    prefix: &str,
    local: bool,
) {
    let placement = if local {
        TensorPlacement::Local
    } else {
        TensorPlacement::Omit
    };
    for parameter in full_parameter_names(module, prefix) {
        plan.insert(parameter, placement.clone());
    }
}

pub(crate) fn assign_module(
    module: &mut impl ModuleParameters,
    prefix: &str,
    tensors: &mut HashMap<String, Array>,
    quantize_on_load: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<(), Error> {
    assign_module_excluding(module, prefix, tensors, quantize_on_load, stream, |_| false)
}

pub(crate) fn assign_module_excluding<F>(
    module: &mut impl ModuleParameters,
    prefix: &str,
    tensors: &mut HashMap<String, Array>,
    quantize_on_load: Option<WeightQuantization>,
    stream: &Stream,
    excluded: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    let mut params = module.parameters_mut().flatten();
    let destinations = params
        .iter()
        .map(|(name, value)| {
            let name = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            };
            (name, value.shape().to_vec())
        })
        .filter(|(name, _)| !excluded(name))
        .collect::<HashMap<_, _>>();
    let mut loaded = HashMap::new();

    for destination in destinations.keys() {
        let source = checkpoint_name(destination);
        if loaded.contains_key(destination) {
            continue;
        }
        let tensor_key = if tensors.contains_key(destination) {
            destination.as_str()
        } else {
            source.as_str()
        };
        let Some(value) = tensors.remove(tensor_key) else {
            continue;
        };
        if destinations[destination] == value.shape() {
            loaded.insert(destination.clone(), value);
            continue;
        }
        let Some(quantization) = quantize_on_load.filter(|_| source.ends_with(".weight")) else {
            return Err(Error::Parallel(format!(
                "pipeline tensor {source} has shape {:?}, expected {:?}",
                value.shape(),
                destinations[destination]
            )));
        };
        let quantized = quantize_tensor(&value, quantization, stream)?;
        eval(
            [&quantized.weight, &quantized.scales]
                .into_iter()
                .chain(quantized.biases.as_ref()),
        )?;
        stream.synchronize()?;
        loaded.insert(destination.clone(), quantized.weight);
        let base = destination
            .strip_suffix(".inner.weight")
            .or_else(|| destination.strip_suffix(".weight"))
            .expect("quantized destination weight");
        loaded.insert(format!("{base}.scales"), quantized.scales);
        if let Some(biases) = quantized.biases {
            loaded.insert(format!("{base}.biases"), biases);
        }
    }

    let mut missing = Vec::new();
    for (local_name, parameter) in &mut params {
        let destination = if prefix.is_empty() {
            local_name.to_string()
        } else {
            format!("{prefix}.{local_name}")
        };
        if excluded(&destination) {
            continue;
        } else if let Some(value) = loaded.remove(&destination) {
            if parameter.shape() != value.shape() {
                return Err(Error::Parallel(format!(
                    "pipeline tensor {destination} has shape {:?}, expected {:?}",
                    value.shape(),
                    parameter.shape()
                )));
            }
            **parameter = value;
        } else {
            missing.push(destination);
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        missing.sort();
        Err(Error::StrictLoadValidation {
            missing,
            unused: Vec::new(),
        })
    }
}

fn load_partition(
    model_dir: &Path,
    plan: &PlacementPlan,
    weights_stream: &Stream,
    stream: &Stream,
    config: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    load_safetensors_partition_on_streams(model_dir, plan, weights_stream, stream, config)
}

/// Loads a pure-PP model using default non-quantizing options.
pub fn load_pipeline_model(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    load_pipeline_model_with_options(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        stream,
        weights_stream,
    )
}

/// Loads an executable rank-local pure pipeline stage.
pub fn load_pipeline_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let model_dir = model_dir.as_ref();
    if model_dir
        .extension()
        .is_some_and(|extension| extension == "gguf")
    {
        return Err(Error::Parallel(
            "pipeline GGUF loading is unsupported because bounded local-layer selection is not available; use safetensors"
                .into(),
        ));
    }
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("pipeline loading requires ModelLoadOptions::parallel".into())
    })?;
    validate_pure_pipeline(topology)?;
    topology.validate_execution_stream(stream)?;

    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    match config.get("model_type").and_then(serde_json::Value::as_str) {
        Some("llama" | "mistral") => load_llama_pipeline(
            model_dir,
            topology,
            options.quantization,
            stream,
            weights_stream,
        ),
        Some("deepseek_v3") => load_deepseek_pipeline(
            model_dir,
            topology,
            options.quantization,
            stream,
            weights_stream,
        ),
        Some(model_type) => Err(Error::UnsupportedArchitecture(format!(
            "pipeline execution supports Llama-compatible and DeepSeek-V3/R1 text models, not {model_type}"
        ))),
        None => Err(Error::UnsupportedArchitecture(
            "pipeline model config is missing model_type".into(),
        )),
    }
}

fn load_llama_pipeline(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let source_args = llama::get_llama_model_args(model_dir)?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::quantization::should_quantize_on_load(
                "Llama pipeline",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
    }
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        ModelKind::Llama,
        source_args.hidden_size,
    );
    let mut plan = PlacementPlan::new(topology);
    let last_stage = topology.pipeline_parallel_size - 1;

    let embedding = linear::unloaded_maybe_quantized_embedding(
        source_args.vocab_size,
        source_args.hidden_size,
        source_args.affine_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    insert_module_plan(
        &mut plan,
        &embedding,
        "model.embed_tokens",
        owns_embedding_weight(&info, source_args.tie_word_embeddings),
    );
    for global_layer in 0..source_args.num_hidden_layers as usize {
        let layer =
            llama::TransformerBlock::new_for_layer(&source_args, global_layer as i32, stream)?;
        insert_module_plan(
            &mut plan,
            &layer,
            &format!("model.layers.{global_layer}"),
            range.contains(&global_layer),
        );
    }
    let norm = nn::RmsNorm::unloaded(
        source_args.hidden_size,
        source_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    insert_module_plan(&mut plan, &norm, "model.norm", info.is_last);
    if !source_args.tie_word_embeddings {
        let head = linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
            source_args.hidden_size,
            source_args.vocab_size,
            source_args.affine_quantization_for("lm_head.weight"),
            stream,
        )?;
        insert_module_plan(&mut plan, &head, "lm_head", info.is_last);
    }
    let partition = load_partition(
        model_dir,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )?;
    info.activation_dtype = infer_activation_dtype(&partition);
    info.local_parameter_bytes = partition.tensors().map(|(_, value)| value.nbytes()).sum();
    info.opened_checkpoint_shards = partition.opened_shards().to_vec();
    info.owned_tensors = partition
        .tensors()
        .map(|(name, _)| checkpoint_name(name))
        .collect();
    info.owned_tensors.sort();
    let mut tensors = partition.into_tensors();

    let mut stage = LlamaStage::new(target_args, range, &info, stream)?;
    stage.load(&mut tensors, quantize_on_load, stream)?;
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    debug_assert_eq!(last_stage, info.pipeline_stages - 1);
    Ok(PipelineModel {
        topology,
        info,
        stage: ArchitectureStage::Llama(stage),
    })
}

impl LlamaStage {
    fn new(
        args: llama::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let make_embedding = || {
            linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.affine_quantization_for("model.embed_tokens.weight"),
                stream,
            )
        };
        let embedding = info.is_first.then(make_embedding).transpose()?;
        let output_embedding = (info.is_last && args.tie_word_embeddings)
            .then(make_embedding)
            .transpose()?;
        let layers = range
            .clone()
            .map(|layer| llama::TransformerBlock::new_for_layer(&args, layer as i32, stream))
            .collect::<Result<_, _>>()?;
        let norm = info
            .is_last
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let lm_head = (info.is_last && !args.tie_word_embeddings)
            .then(|| {
                linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.affine_quantization_for("lm_head.weight"),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            range,
            embedding,
            output_embedding,
            layers,
            norm,
            lm_head,
        })
    }

    fn load(
        &mut self,
        tensors: &mut HashMap<String, Array>,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(embedding) = &mut self.embedding {
            assign_module(
                embedding,
                "model.embed_tokens",
                tensors,
                quantization,
                stream,
            )?;
        }
        if let Some(embedding) = &mut self.output_embedding {
            assign_module(
                embedding,
                "model.embed_tokens",
                tensors,
                quantization,
                stream,
            )?;
        }
        for (global_layer, layer) in self.range.clone().zip(&mut self.layers) {
            assign_module(
                layer,
                &format!("model.layers.{global_layer}"),
                tensors,
                quantization,
                stream,
            )?;
        }
        if let Some(norm) = &mut self.norm {
            assign_module(norm, "model.norm", tensors, None, stream)?;
        }
        if let Some(head) = &mut self.lm_head {
            assign_module(head, "lm_head", tensors, quantization, stream)?;
        }
        Ok(())
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLlamaLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Llama stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let mut hidden = match input {
            PipelineStageInput::Tokens(tokens) => self
                .embedding
                .as_mut()
                .expect("first stage embedding")
                .forward(tokens, stream)?,
            PipelineStageInput::Hidden(hidden) => hidden.clone(),
        };
        let offset = caches.first().map_or(0, |cache| match cache {
            PipelineLlamaLayerCache::Standard { cache, .. } => cache.offset(),
            PipelineLlamaLayerCache::Sliding { cache, .. } => cache.offset(),
        });
        let generated_sliding_window = (explicit_mask.is_none() && step.sequence_length > 1)
            .then_some(self.args.sliding_window)
            .flatten();
        let generated_mask = if explicit_mask.is_some() || generated_sliding_window.is_some() {
            None
        } else if let Some(window) = self.args.sliding_window {
            let retained = offset.min(window);
            let keys = retained + step.sequence_length;
            ((step.sequence_length > 1 || keys > window) && keys > 1)
                .then(|| {
                    create_causal_mask(
                        step.sequence_length,
                        Some(retained),
                        Some(window - 1),
                        None,
                        stream,
                    )
                })
                .transpose()?
        } else {
            (step.sequence_length > 1)
                .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
                .transpose()?
        };
        let mask = explicit_mask.or(generated_mask.as_ref());
        for ((global_layer, layer), cache) in self
            .range
            .clone()
            .zip(&mut self.layers)
            .zip(caches.iter_mut())
        {
            match cache {
                PipelineLlamaLayerCache::Standard {
                    global_layer: cached_layer,
                    cache,
                } if *cached_layer == global_layer => {
                    hidden = layer.forward(
                        llama::AttentionInput {
                            x: &hidden,
                            mask,
                            cache: Some(cache),
                            generated_sliding_window,
                        },
                        stream,
                    )?;
                }
                PipelineLlamaLayerCache::Sliding {
                    global_layer: cached_layer,
                    cache,
                } if *cached_layer == global_layer => {
                    hidden = layer.forward(
                        llama::AttentionInput {
                            x: &hidden,
                            mask,
                            cache: Some(cache),
                            generated_sliding_window,
                        },
                        stream,
                    )?;
                }
                _ => {
                    return Err(Error::Parallel(format!(
                        "Llama stage cache does not match global layer {global_layer}"
                    )))
                }
            }
        }
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .expect("last tied stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            Ok(PipelineStageOutput::Logits(logits))
        } else {
            Ok(PipelineStageOutput::Hidden(hidden))
        }
    }
}

fn load_deepseek_pipeline(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let source_args = deepseek_v3::get_model_args(model_dir)?;
    if requested_quantization.is_some() && source_args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 pipeline weights cannot be implicitly requantized".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::quantization::should_quantize_on_load(
                "DeepSeek pipeline",
                source_args.affine_quantization()?,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization_config = None;
        target_args.quantization = Some(quantization);
    }
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        ModelKind::DeepSeekV3,
        source_args.hidden_size,
    );
    let mut plan = PlacementPlan::new(topology);
    let embedding = linear::unloaded_maybe_quantized_embedding(
        source_args.vocab_size,
        source_args.hidden_size,
        source_args.weight_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    insert_module_plan(&mut plan, &embedding, "model.embed_tokens", info.is_first);
    for global_layer in 0..source_args.num_hidden_layers as usize {
        let layer = deepseek_v3::DecoderLayer::new(&source_args, global_layer as i32, stream)?;
        let local = range.contains(&global_layer);
        insert_module_plan(
            &mut plan,
            &layer,
            &format!("model.layers.{global_layer}"),
            local,
        );
        if source_args.is_moe_layer(global_layer as i32) {
            insert_deepseek_expert_plan(
                &mut plan,
                &source_args,
                global_layer,
                local,
                quantize_on_load.is_some(),
            );
        }
    }
    let norm = nn::RmsNorm::unloaded(
        source_args.hidden_size,
        source_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    insert_module_plan(&mut plan, &norm, "model.norm", info.is_last);
    let head = linear::unloaded_maybe_quantized_linear(
        source_args.hidden_size,
        source_args.vocab_size,
        false,
        source_args.weight_quantization_for("lm_head.weight"),
        stream,
    )?;
    insert_module_plan(&mut plan, &head, "lm_head", info.is_last);
    let mut strict = StrictLoadConfig::default();
    for index in 0..source_args.num_nextn_predict_layers {
        strict = strict.allow_unused_prefix(format!(
            "model.layers.{}.",
            source_args.num_hidden_layers + index
        ));
    }
    let partition = load_partition(model_dir, &plan, weights_stream, stream, &strict)?;
    info.activation_dtype = infer_activation_dtype(&partition);
    info.local_parameter_bytes = partition.tensors().map(|(_, value)| value.nbytes()).sum();
    info.opened_checkpoint_shards = partition.opened_shards().to_vec();
    info.owned_tensors = partition
        .tensors()
        .map(|(name, _)| checkpoint_name(name))
        .collect();
    info.owned_tensors.sort();
    let mut tensors = partition.into_tensors();
    let mut stage = DeepSeekStage::new(target_args, range, &info, stream)?;
    stage.load(&mut tensors, quantize_on_load, stream)?;
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    Ok(PipelineModel {
        topology,
        info,
        stage: ArchitectureStage::DeepSeek(stage),
    })
}

fn insert_deepseek_expert_plan(
    plan: &mut PlacementPlan,
    args: &deepseek_v3::ModelArgs,
    layer: usize,
    local: bool,
    dense_source: bool,
) {
    let placement = if local {
        TensorPlacement::Local
    } else {
        TensorPlacement::Omit
    };
    let components: &[&str] = if dense_source {
        &["weight"]
    } else if args.native_fp8_config().is_some() {
        &["weight", "weight_scale_inv"]
    } else if let Some(quantization) = args
        .affine_quantization()
        .expect("validated DeepSeek quantization")
    {
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
                plan.insert(
                    format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.{component}"),
                    placement.clone(),
                );
            }
        }
    }
}

impl DeepSeekStage {
    fn new(
        args: deepseek_v3::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let embedding = info
            .is_first
            .then(|| {
                linear::unloaded_maybe_quantized_embedding(
                    args.vocab_size,
                    args.hidden_size,
                    args.weight_quantization_for("model.embed_tokens.weight"),
                    stream,
                )
            })
            .transpose()?;
        let layers = range
            .clone()
            .map(|layer| deepseek_v3::DecoderLayer::new(&args, layer as i32, stream))
            .collect::<Result<_, _>>()?;
        let norm = info
            .is_last
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let lm_head = info
            .is_last
            .then(|| {
                linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.vocab_size,
                    false,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            range,
            embedding,
            layers,
            norm,
            lm_head,
        })
    }

    fn load(
        &mut self,
        tensors: &mut HashMap<String, Array>,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(embedding) = &mut self.embedding {
            assign_module(
                embedding,
                "model.embed_tokens",
                tensors,
                quantization,
                stream,
            )?;
        }
        for (global_layer, layer) in self.range.clone().zip(&mut self.layers) {
            assign_module(
                layer,
                &format!("model.layers.{global_layer}"),
                tensors,
                quantization,
                stream,
            )?;
            if let Some(moe) = layer.mlp.moe_mut() {
                load_deepseek_experts(
                    moe,
                    global_layer,
                    (
                        self.args.n_routed_experts,
                        self.args.hidden_size,
                        self.args.moe_intermediate_size,
                    ),
                    tensors,
                    quantization,
                    stream,
                )?;
            }
        }
        if let Some(norm) = &mut self.norm {
            assign_module(norm, "model.norm", tensors, None, stream)?;
        }
        if let Some(head) = &mut self.lm_head {
            assign_module(head, "lm_head", tensors, quantization, stream)?;
        }
        Ok(())
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineDeepSeekLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "DeepSeek stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let mut hidden = match input {
            PipelineStageInput::Tokens(tokens) => self
                .embedding
                .as_mut()
                .expect("first stage embedding")
                .forward(tokens, stream)?,
            PipelineStageInput::Hidden(hidden) => hidden.clone(),
        };
        let offset = caches.first().map_or(0, |cache| cache.cache.offset());
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1 && offset > 0)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        for ((global_layer, layer), cache) in self
            .range
            .clone()
            .zip(&mut self.layers)
            .zip(caches.iter_mut())
        {
            if cache.global_layer != global_layer {
                return Err(Error::Parallel(format!(
                    "DeepSeek stage cache does not match global layer {global_layer}"
                )));
            }
            hidden = layer.forward_stage(&hidden, mask, Some(&mut cache.cache), stream)?;
        }
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = self
                .lm_head
                .as_mut()
                .expect("last stage head")
                .forward(&hidden, stream)?;
            Ok(PipelineStageOutput::Logits(logits))
        } else {
            Ok(PipelineStageOutput::Hidden(hidden))
        }
    }
}

pub(crate) fn load_deepseek_experts(
    moe: &mut deepseek_v3::Moe,
    layer: usize,
    dimensions: (i32, i32, i32),
    tensors: &mut HashMap<String, Array>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<(), Error> {
    let (num_experts, hidden_size, intermediate_size) = dimensions;
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let take_component = |component: &str,
                              tensors: &mut HashMap<String, Array>|
         -> Result<Option<Array>, Error> {
            let mut values = Vec::with_capacity(num_experts as usize);
            for expert in 0..num_experts {
                let name =
                    format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.{component}");
                match tensors.remove(&name) {
                    Some(value) => values.push(value),
                    None if expert == 0 => return Ok(None),
                    None => {
                        return Err(Error::StrictLoadValidation {
                            missing: vec![name],
                            unused: Vec::new(),
                        })
                    }
                }
            }
            let refs = values.iter().collect::<Vec<_>>();
            Ok(Some(stack_axis(&refs, 0, stream)?))
        };
        let weight =
            take_component("weight", tensors)?.ok_or_else(|| Error::StrictLoadValidation {
                missing: vec![format!(
                    "model.layers.{layer}.mlp.experts.0.{projection}.weight"
                )],
                unused: Vec::new(),
            })?;
        let mut fp8_scale = take_component("weight_scale_inv", tensors)?;
        let mut scales = take_component("scales", tensors)?;
        let mut biases = take_component("biases", tensors)?;
        let experts = &mut moe.experts;
        let (output_dims, input_dims, affine) = match projection {
            "gate_proj" => (intermediate_size, hidden_size, experts.gate_affine),
            "up_proj" => (intermediate_size, hidden_size, experts.up_affine),
            "down_proj" => (hidden_size, intermediate_size, experts.down_affine),
            _ => unreachable!(),
        };
        let source_affine = quantization.is_none().then_some(affine).flatten();
        let stored_input_dims = source_affine.map_or(input_dims, |quantization| {
            quantized_packed_dimension(input_dims, quantization.bits())
        });
        validate_expert_bank_shape(
            layer,
            projection,
            "weight",
            &weight,
            &[num_experts, output_dims, stored_input_dims],
        )?;
        if let Some(scale) = &fp8_scale {
            validate_expert_bank_shape(
                layer,
                projection,
                "weight_scale_inv",
                scale,
                &[
                    num_experts,
                    (output_dims + 127) / 128,
                    (input_dims + 127) / 128,
                ],
            )?;
        }
        if let Some(affine) = source_affine {
            let expected = [num_experts, output_dims, input_dims / affine.group_size()];
            if let Some(value) = &scales {
                validate_expert_bank_shape(layer, projection, "scales", value, &expected)?;
            }
            if let Some(value) = &biases {
                validate_expert_bank_shape(layer, projection, "biases", value, &expected)?;
            }
        }
        let weight = if let Some(quantization) = quantization {
            let quantized =
                crate::models::common::moe::quantize_expert_bank(&weight, quantization, stream)?;
            scales = Some(quantized.scales);
            biases = quantized.biases;
            quantized.weight
        } else {
            weight
        };
        eval(
            [&weight]
                .into_iter()
                .chain(fp8_scale.as_ref())
                .chain(scales.as_ref())
                .chain(biases.as_ref()),
        )?;
        stream.synchronize()?;
        match projection {
            "gate_proj" => {
                experts.gate_proj = safemlx::module::Param::new(Some(weight));
                experts.gate_proj_scale_inv = safemlx::module::Param::new(fp8_scale.take());
                experts.gate_proj_scales = safemlx::module::Param::new(scales.take());
                experts.gate_proj_biases = safemlx::module::Param::new(biases.take());
            }
            "up_proj" => {
                experts.up_proj = safemlx::module::Param::new(Some(weight));
                experts.up_proj_scale_inv = safemlx::module::Param::new(fp8_scale.take());
                experts.up_proj_scales = safemlx::module::Param::new(scales.take());
                experts.up_proj_biases = safemlx::module::Param::new(biases.take());
            }
            "down_proj" => {
                experts.down_proj = safemlx::module::Param::new(Some(weight));
                experts.down_proj_scale_inv = safemlx::module::Param::new(fp8_scale.take());
                experts.down_proj_scales = safemlx::module::Param::new(scales.take());
                experts.down_proj_biases = safemlx::module::Param::new(biases.take());
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn validate_expert_bank_shape(
    layer: usize,
    projection: &str,
    component: &str,
    value: &Array,
    expected: &[i32],
) -> Result<(), Error> {
    if value.shape() == expected {
        Ok(())
    } else {
        Err(Error::Parallel(format!(
            "DeepSeek pipeline layer {layer} expert {projection}.{component} bank has shape {:?}, expected {expected:?}",
            value.shape()
        )))
    }
}

/// Runs stage-local execution while retaining global observer layer names.
pub fn forward_stage_with_observer(
    model: &mut PipelineModel,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    mask: Option<&Array>,
    cache: &mut PipelineCache,
    stream: &Stream,
    observer: &mut impl ActivationObserver,
) -> Result<PipelineStageOutput, Error> {
    // Common boundary observations are stable; architecture layers already
    // retain global identity in `stage_info` and the normal detailed adapters
    // can be extended without changing orchestration.
    let output = model.forward_stage(input, step, mask, cache, stream)?;
    match &output {
        PipelineStageOutput::Hidden(hidden) => observer.observe(
            &format!(
                "model.layers.{}.pipeline_stage_output",
                model.info.global_layer_range.end - 1
            ),
            hidden,
        )?,
        PipelineStageOutput::Logits(logits) => observer.observe("lm_head.logits", logits)?,
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::DeviceAssignment;
    use safemlx::{module::Param, ops::ones_dtype, Device, DeviceType, ExecutionContext};

    fn topology(world: usize, rank: usize, pp: usize) -> ParallelTopology {
        ParallelTopology::from_rank(
            world,
            rank,
            1,
            pp,
            1,
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
        .unwrap()
    }

    #[test]
    fn stage_roles_and_neighbor_ranks_are_explicit() {
        let first = base_info(topology(3, 0, 3), 0..2, ModelKind::Llama, 8);
        assert!(first.is_first);
        assert!(!first.is_last);
        assert_eq!(first.predecessor_rank, None);
        assert_eq!(first.successor_rank, Some(1));

        let middle = base_info(topology(3, 1, 3), 2..4, ModelKind::Llama, 8);
        assert!(!middle.is_first);
        assert!(!middle.is_last);
        assert_eq!(middle.predecessor_rank, Some(0));
        assert_eq!(middle.successor_rank, Some(2));

        let last = base_info(topology(3, 2, 3), 4..5, ModelKind::Llama, 8);
        assert!(!last.is_first);
        assert!(last.is_last);
        assert_eq!(last.predecessor_rank, Some(1));
        assert_eq!(last.successor_rank, None);
    }

    #[test]
    fn boundary_and_tied_embedding_ownership_is_not_replicated() {
        let first = base_info(topology(3, 0, 3), 0..1, ModelKind::Llama, 8);
        let middle = base_info(topology(3, 1, 3), 1..2, ModelKind::Llama, 8);
        let last = base_info(topology(3, 2, 3), 2..3, ModelKind::Llama, 8);
        assert!(owns_embedding_weight(&first, false));
        assert!(!owns_embedding_weight(&middle, false));
        assert!(!owns_embedding_weight(&last, false));
        assert!(owns_embedding_weight(&first, true));
        assert!(!owns_embedding_weight(&middle, true));
        assert!(owns_embedding_weight(&last, true));
    }

    #[test]
    fn pure_pipeline_validation_rejects_singleton_and_hybrids() {
        assert!(validate_pure_pipeline(topology(2, 0, 2)).is_ok());
        assert!(validate_pure_pipeline(topology(1, 0, 1)).is_err());
        let hybrid =
            ParallelTopology::from_rank(4, 0, 2, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(validate_pure_pipeline(hybrid).is_err());
    }

    #[test]
    fn activation_shape_validation_is_role_aware() {
        let later = base_info(topology(2, 1, 2), 1..2, ModelKind::Llama, 8);
        let step = PipelineStep::new(1, 3).unwrap();
        assert!(validate_hidden_metadata(&later, &[1, 3, 8], Dtype::Float32, step).is_ok());
        assert!(validate_hidden_metadata(&later, &[1, 3, 7], Dtype::Float32, step).is_err());
        assert!(validate_hidden_metadata(&later, &[1, 3, 8], Dtype::Float16, step).is_err());
    }

    #[test]
    fn cache_reports_only_local_global_layers() {
        let cache = PipelineCache::Llama(vec![
            PipelineLlamaLayerCache::Standard {
                global_layer: 3,
                cache: ConcatKeyValueCache::new(),
            },
            PipelineLlamaLayerCache::Standard {
                global_layer: 4,
                cache: ConcatKeyValueCache::new(),
            },
        ]);
        assert_eq!(cache.global_layers(), vec![3, 4]);
    }

    fn gpu_topology(rank: usize) -> ParallelTopology {
        ParallelTopology::from_rank(2, rank, 1, 2, 1, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    fn initialize_parameters(module: &mut impl ModuleParameters, stream: &Stream) {
        for (name, parameter) in module.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter =
                if name.ends_with("layernorm.weight") || name.as_ref() == "model.norm.weight" {
                    ones_dtype(&shape, dtype, stream).unwrap()
                } else {
                    Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
                };
        }
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 1e-5, "{left} != {right}");
        }
    }

    fn llama_args(tied: bool) -> llama::ModelArgs {
        llama::ModelArgs {
            model_type: "llama".into(),
            hidden_size: 8,
            num_hidden_layers: 2,
            intermediate_size: 16,
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rope_traditional: false,
            head_dim: 4,
            tie_word_embeddings: tied,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            sliding_window: None,
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    fn llama_pipeline_stages(
        source: &llama::ResidentModel,
        stream: &Stream,
    ) -> (PipelineModel, PipelineModel) {
        let first_topology = gpu_topology(0);
        let last_topology = gpu_topology(1);
        let first_info = base_info(
            first_topology,
            0..1,
            ModelKind::Llama,
            source.args.hidden_size,
        );
        let last_info = base_info(
            last_topology,
            1..2,
            ModelKind::Llama,
            source.args.hidden_size,
        );
        let first = LlamaStage {
            args: source.args.clone(),
            range: 0..1,
            embedding: Some(source.model.embed_tokens.clone()),
            output_embedding: None,
            layers: vec![source.model.layers[0].clone()],
            norm: None,
            lm_head: None,
        };
        let last = LlamaStage {
            args: source.args.clone(),
            range: 1..2,
            embedding: None,
            output_embedding: source
                .args
                .tie_word_embeddings
                .then(|| source.model.embed_tokens.clone()),
            layers: vec![source.model.layers[1].clone()],
            norm: Some(source.model.norm.clone()),
            lm_head: source.lm_head.clone(),
        };
        let _ = stream;
        (
            PipelineModel {
                topology: first_topology,
                info: first_info,
                stage: ArchitectureStage::Llama(first),
            },
            PipelineModel {
                topology: last_topology,
                info: last_info,
                stage: ArchitectureStage::Llama(last),
            },
        )
    }

    #[test]
    fn sequential_llama_pipeline_matches_tied_and_untied_prefill_decode() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        for tied in [false, true] {
            let mut reference = llama::ResidentModel::new(llama_args(tied), stream).unwrap();
            initialize_parameters(&mut reference, stream);
            let (mut first, mut last) = llama_pipeline_stages(&reference, stream);
            let mut reference_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
            let mut first_cache = first.new_cache();
            let mut last_cache = last.new_cache();
            let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);

            for (tokens, sequence) in [
                (&prompt, 2),
                (&Array::from_slice(&[3u32], &[1, 1]), 1),
                (&Array::from_slice(&[4u32], &[1, 1]), 1),
            ] {
                let reference_logits = reference
                    .forward(
                        llama::ModelInput {
                            inputs: tokens,
                            mask: None,
                            cache: &mut reference_cache,
                        },
                        stream,
                    )
                    .unwrap();
                let hidden = match first
                    .forward_stage(
                        PipelineStageInput::Tokens(tokens),
                        PipelineStep::new(1, sequence).unwrap(),
                        None,
                        &mut first_cache,
                        stream,
                    )
                    .unwrap()
                {
                    PipelineStageOutput::Hidden(hidden) => hidden,
                    PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
                };
                let pipeline_logits = match last
                    .forward_stage(
                        PipelineStageInput::Hidden(&hidden),
                        PipelineStep::new(1, sequence).unwrap(),
                        None,
                        &mut last_cache,
                        stream,
                    )
                    .unwrap()
                {
                    PipelineStageOutput::Logits(logits) => logits,
                    PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
                };
                assert_close(&pipeline_logits, &reference_logits);
            }
        }
    }

    fn deepseek_args() -> deepseek_v3::ModelArgs {
        deepseek_v3::ModelArgs {
            model_type: "deepseek_v3".into(),
            hidden_size: 8,
            intermediate_size: 16,
            moe_intermediate_size: 4,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rope_scaling: None,
            q_lora_rank: Some(4),
            kv_lora_rank: 4,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            v_head_dim: 2,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
            n_routed_experts: 4,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            topk_method: "noaux_tc".into(),
            scoring_func: "sigmoid".into(),
            norm_topk_prob: true,
            routed_scaling_factor: 1.5,
            num_nextn_predict_layers: 0,
            quantization_config: None,
            quantization: None,
            quantized_weight_configs: None,
            split_kv_b: false,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn sequential_deepseek_pipeline_matches_local_moe_prefill_decode() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut reference = deepseek_v3::Model::new(deepseek_args(), stream).unwrap();
        if let deepseek_v3::FeedForward::Moe(moe) = &mut reference.model.layers[1].mlp {
            let experts = reference.args.n_routed_experts;
            let hidden = reference.args.hidden_size;
            let intermediate = reference.args.moe_intermediate_size;
            moe.experts.gate_proj = Param::new(Some(
                Array::full::<f32>(
                    &[experts, intermediate, hidden],
                    Array::from_f32(0.01),
                    stream,
                )
                .unwrap(),
            ));
            moe.experts.up_proj = moe.experts.gate_proj.clone();
            moe.experts.down_proj = Param::new(Some(
                Array::full::<f32>(
                    &[experts, hidden, intermediate],
                    Array::from_f32(0.01),
                    stream,
                )
                .unwrap(),
            ));
        } else {
            panic!("second tiny DeepSeek layer must be MoE");
        }
        initialize_parameters(&mut reference, stream);

        let first_topology = gpu_topology(0);
        let last_topology = gpu_topology(1);
        let mut first = PipelineModel {
            topology: first_topology,
            info: base_info(first_topology, 0..1, ModelKind::DeepSeekV3, 8),
            stage: ArchitectureStage::DeepSeek(DeepSeekStage {
                args: reference.args.clone(),
                range: 0..1,
                embedding: Some(reference.model.embed_tokens.clone()),
                layers: vec![reference.model.layers[0].clone()],
                norm: None,
                lm_head: None,
            }),
        };
        let mut last = PipelineModel {
            topology: last_topology,
            info: base_info(last_topology, 1..2, ModelKind::DeepSeekV3, 8),
            stage: ArchitectureStage::DeepSeek(DeepSeekStage {
                args: reference.args.clone(),
                range: 1..2,
                embedding: None,
                layers: vec![reference.model.layers[1].clone()],
                norm: Some(reference.model.norm.clone()),
                lm_head: Some(reference.lm_head.clone()),
            }),
        };
        let mut reference_cache = reference.new_cache();
        let mut first_cache = first.new_cache();
        let mut last_cache = last.new_cache();
        let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);

        for (tokens, sequence) in [
            (&prompt, 2),
            (&Array::from_slice(&[3u32], &[1, 1]), 1),
            (&Array::from_slice(&[4u32], &[1, 1]), 1),
        ] {
            let reference_logits = reference
                .forward(
                    deepseek_v3::ModelInput {
                        inputs: tokens,
                        mask: None,
                        cache: Some(&mut reference_cache),
                    },
                    stream,
                )
                .unwrap();
            let hidden = match first
                .forward_stage(
                    PipelineStageInput::Tokens(tokens),
                    PipelineStep::new(1, sequence).unwrap(),
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Hidden(hidden) => hidden,
                PipelineStageOutput::Logits(_) => panic!("first stage produced logits"),
            };
            let pipeline_logits = match last
                .forward_stage(
                    PipelineStageInput::Hidden(&hidden),
                    PipelineStep::new(1, sequence).unwrap(),
                    None,
                    &mut last_cache,
                    stream,
                )
                .unwrap()
            {
                PipelineStageOutput::Logits(logits) => logits,
                PipelineStageOutput::Hidden(_) => panic!("last stage produced hidden state"),
            };
            assert_close(&pipeline_logits, &reference_logits);
        }
        assert_eq!(last_cache.global_layers(), vec![1]);
    }
}
