//! Rank-aware checkpoint topology, placement planning, and selective loading.
//!
//! Runtime parallelism is deliberately independent of architecture metadata in
//! `config.json`. In particular, a checkpoint's `ep_size` describes how that
//! checkpoint was written, while [`ParallelTopology::expert_parallel_size`]
//! describes how the current inference job is arranged.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    ops::Range,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;
use safemlx::{
    distributed::{self, Group},
    transforms::eval,
    Array, Device, DeviceType, Stream,
};
use safetensors::SafeTensors;

use crate::{
    error::Error,
    weights::{StrictLoadConfig, WeightMap},
};

/// Explicit process-local execution-device assignment.
///
/// This value is never inferred from a global distributed rank. On a
/// one-process-per-visible-GPU launcher it is commonly GPU index zero on every
/// process, even though each process has a different global rank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DeviceAssignment {
    /// Device family used by this process.
    pub device_type: DeviceType,
    /// Index within this process's visible devices.
    pub local_index: usize,
}

impl DeviceAssignment {
    /// Creates an explicit process-local assignment.
    pub const fn new(device_type: DeviceType, local_index: usize) -> Self {
        Self {
            device_type,
            local_index,
        }
    }

    /// Resolves this assignment to an MLX device.
    pub fn device(self) -> Result<Device, Error> {
        Ok(distributed::device_for_local_rank(
            self.device_type,
            self.local_index,
        )?)
    }
}

/// Validated, architecture-independent runtime parallel coordinates.
///
/// Rank ordering is pipeline-major, then tensor, then expert, with expert as
/// the fastest-changing coordinate:
/// `global_rank = ((pipeline_rank * tensor_size) + tensor_rank) * expert_size + expert_rank`.
/// The ordering is stable and should be used by later execution phases.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ParallelTopology {
    /// Number of processes in the communication group.
    pub world_size: usize,
    /// Rank in the communication group.
    pub global_rank: usize,
    /// Tensor-parallel process count.
    pub tensor_parallel_size: usize,
    /// Tensor-parallel coordinate of this process.
    pub tensor_parallel_rank: usize,
    /// Pipeline-parallel process count.
    pub pipeline_parallel_size: usize,
    /// Pipeline-parallel coordinate of this process.
    pub pipeline_parallel_rank: usize,
    /// Expert-parallel process count.
    pub expert_parallel_size: usize,
    /// Expert-parallel coordinate of this process.
    pub expert_parallel_rank: usize,
    /// Explicit process-local device assignment.
    pub device: DeviceAssignment,
}

impl ParallelTopology {
    /// Snapshots and validates runtime coordinates from a distributed group.
    pub fn from_group(
        group: &Group,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        Self::from_rank(
            group.size(),
            group.rank(),
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            device,
        )
    }

    /// Validates a topology snapshot with an explicit world size and rank.
    ///
    /// Most applications should use [`Self::from_group`]. This constructor is
    /// useful to validate launcher metadata before initializing model state.
    pub fn from_rank(
        world_size: usize,
        global_rank: usize,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        if world_size == 0
            || tensor_parallel_size == 0
            || pipeline_parallel_size == 0
            || expert_parallel_size == 0
        {
            return Err(Error::Parallel(
                "world, tensor, pipeline, and expert parallel sizes must all be nonzero".into(),
            ));
        }
        if global_rank >= world_size {
            return Err(Error::Parallel(format!(
                "global rank {global_rank} is outside world size {world_size}"
            )));
        }
        if i32::try_from(device.local_index).is_err() {
            return Err(Error::Parallel(format!(
                "local device index {} does not fit in MLX's i32 device index",
                device.local_index
            )));
        }
        let topology_size = pipeline_parallel_size
            .checked_mul(tensor_parallel_size)
            .and_then(|value| value.checked_mul(expert_parallel_size))
            .ok_or_else(|| Error::Parallel("parallel topology size overflowed usize".into()))?;
        if topology_size != world_size {
            return Err(Error::Parallel(format!(
                "TP({tensor_parallel_size}) * PP({pipeline_parallel_size}) * EP({expert_parallel_size}) = {topology_size}, not world size {world_size}"
            )));
        }

        let expert_parallel_rank = global_rank % expert_parallel_size;
        let outer = global_rank / expert_parallel_size;
        let tensor_parallel_rank = outer % tensor_parallel_size;
        let pipeline_parallel_rank = outer / tensor_parallel_size;

        Ok(Self {
            world_size,
            global_rank,
            tensor_parallel_size,
            tensor_parallel_rank,
            pipeline_parallel_size,
            pipeline_parallel_rank,
            expert_parallel_size,
            expert_parallel_rank,
            device,
        })
    }

    /// Returns whether every runtime parallel dimension is a singleton.
    pub const fn is_replicated(self) -> bool {
        self.world_size == 1
            && self.tensor_parallel_size == 1
            && self.pipeline_parallel_size == 1
            && self.expert_parallel_size == 1
    }

    /// Returns this pipeline stage's balanced contiguous decoder-layer range.
    ///
    /// Empty stages are rejected. Use [`balanced_contiguous_range`] directly
    /// with `allow_empty = true` when an architecture explicitly supports them.
    pub fn layer_range(self, decoder_layers: usize) -> Result<Range<usize>, Error> {
        balanced_contiguous_range(
            decoder_layers,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
            false,
        )
    }

    /// Returns this expert rank's balanced contiguous routed-expert range.
    ///
    /// Empty expert partitions are rejected.
    pub fn expert_range(self, routed_experts: usize) -> Result<Range<usize>, Error> {
        balanced_contiguous_range(
            routed_experts,
            self.expert_parallel_size,
            self.expert_parallel_rank,
            false,
        )
    }

    /// Verifies that an execution stream uses this process's assigned device.
    pub fn validate_execution_stream(self, stream: &Stream) -> Result<(), Error> {
        let actual = stream.get_device()?;
        let actual_type = actual.get_type()?;
        let actual_index = actual.get_index()?;
        let expected_index = i32::try_from(self.device.local_index)
            .expect("topology construction validated the local device index");
        if actual_type == self.device.device_type && actual_index == expected_index {
            Ok(())
        } else {
            Err(Error::Parallel(format!(
                "execution stream uses {actual_type:?} device {actual_index}, but this rank is assigned {:?} device {expected_index}",
                self.device.device_type
            )))
        }
    }
}

/// Computes a deterministic balanced contiguous range.
///
/// The first `total % parts` partitions receive one extra item. Therefore the
/// ranges cover `0..total` without gaps or overlap, including uneven splits.
pub fn balanced_contiguous_range(
    total: usize,
    parts: usize,
    index: usize,
    allow_empty: bool,
) -> Result<Range<usize>, Error> {
    if parts == 0 {
        return Err(Error::Parallel("partition count must be nonzero".into()));
    }
    if index >= parts {
        return Err(Error::Parallel(format!(
            "partition index {index} is outside {parts} parts"
        )));
    }
    if !allow_empty && total < parts {
        return Err(Error::Parallel(format!(
            "cannot divide {total} items among {parts} non-empty partitions"
        )));
    }
    let base = total / parts;
    let extra = total % parts;
    let start = index
        .checked_mul(base)
        .and_then(|value| value.checked_add(index.min(extra)))
        .ok_or_else(|| Error::Parallel("balanced range calculation overflowed usize".into()))?;
    let len = base + usize::from(index < extra);
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::Parallel("balanced range calculation overflowed usize".into()))?;
    Ok(start..end)
}

/// A validated contiguous slice of a source tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorSlice {
    /// Source tensor axis being divided.
    pub axis: usize,
    /// Inclusive element offset on `axis`.
    pub start: usize,
    /// Exclusive element offset on `axis`.
    pub end: usize,
    /// Shard index.
    pub index: usize,
    /// Total number of equal shards.
    pub parts: usize,
}

impl TensorSlice {
    /// Validates and calculates an equal contiguous tensor slice.
    pub fn for_shape(
        shape: &[usize],
        axis: usize,
        index: usize,
        parts: usize,
    ) -> Result<Self, Error> {
        if axis >= shape.len() {
            return Err(Error::Parallel(format!(
                "tensor axis {axis} is outside rank {} shape {shape:?}",
                shape.len()
            )));
        }
        if parts == 0 {
            return Err(Error::Parallel("tensor shard count must be nonzero".into()));
        }
        if index >= parts {
            return Err(Error::Parallel(format!(
                "tensor shard index {index} is outside {parts} parts"
            )));
        }
        let dimension = shape[axis];
        if dimension == 0 || dimension % parts != 0 {
            return Err(Error::Parallel(format!(
                "tensor dimension {dimension} on axis {axis} is not nonzero and divisible by {parts}"
            )));
        }
        let width = dimension / parts;
        let start = index
            .checked_mul(width)
            .ok_or_else(|| Error::Parallel("tensor slice offset overflowed usize".into()))?;
        Ok(Self {
            axis,
            start,
            end: start + width,
            index,
            parts,
        })
    }

    /// Returns the local tensor shape produced by this slice.
    pub fn local_shape(&self, source_shape: &[usize]) -> Vec<usize> {
        let mut shape = source_shape.to_vec();
        shape[self.axis] = self.end - self.start;
        shape
    }
}

/// Typed placement decision for one target tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TensorPlacement {
    /// Materialize the complete tensor on every rank.
    Replicated,
    /// Materialize the complete tensor on this rank.
    Local,
    /// Intentionally omit this tensor on this rank.
    Omit,
    /// Materialize the complete tensor only on one global rank.
    Rank {
        /// Owning global rank.
        rank: usize,
    },
    /// Materialize the complete tensor only on one pipeline stage.
    PipelineStage {
        /// Owning pipeline-stage coordinate.
        stage: usize,
    },
    /// Materialize an equal contiguous source-tensor slice.
    Shard {
        /// Source tensor axis being sharded.
        axis: usize,
        /// Shard index.
        index: usize,
        /// Total shard count.
        parts: usize,
    },
}

#[derive(Debug, Clone)]
struct TensorPlan {
    placement: TensorPlacement,
    expected_source_shape: Option<Vec<usize>>,
}

/// Inspectable mapping from rewritten target names to typed placement decisions.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    topology: ParallelTopology,
    tensors: HashMap<String, TensorPlan>,
    default: Option<TensorPlacement>,
}

impl PlacementPlan {
    /// Creates a strict plan in which every checkpoint tensor must be named.
    pub fn new(topology: ParallelTopology) -> Self {
        Self {
            topology,
            tensors: HashMap::new(),
            default: None,
        }
    }

    /// Creates a plan that replicates every checkpoint tensor.
    pub fn replicated(topology: ParallelTopology) -> Self {
        Self::new(topology).with_default(TensorPlacement::Replicated)
    }

    /// Sets the placement used for checkpoint keys without an explicit entry.
    pub fn with_default(mut self, placement: TensorPlacement) -> Self {
        self.default = Some(placement);
        self
    }

    /// Returns the topology captured by this plan.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }

    /// Adds or replaces a target-tensor placement.
    pub fn insert(&mut self, target: impl Into<String>, placement: TensorPlacement) {
        self.tensors.insert(
            target.into(),
            TensorPlan {
                placement,
                expected_source_shape: None,
            },
        );
    }

    /// Adds a placement with a required pre-slice checkpoint shape.
    pub fn insert_expected(
        &mut self,
        target: impl Into<String>,
        expected_source_shape: impl Into<Vec<usize>>,
        placement: TensorPlacement,
    ) -> Result<(), Error> {
        let expected_source_shape = expected_source_shape.into();
        validate_placement(&placement, &expected_source_shape, self.topology)?;
        self.tensors.insert(
            target.into(),
            TensorPlan {
                placement,
                expected_source_shape: Some(expected_source_shape),
            },
        );
        Ok(())
    }

    /// Adds weight, scales, and optional biases using one logical placement.
    ///
    /// Keeping companions in one call prevents a quantized module's metadata
    /// from being accidentally placed differently from its packed weight.
    pub fn insert_quantized_companions(
        &mut self,
        prefix: &str,
        placement: TensorPlacement,
        has_biases: bool,
    ) {
        self.insert(format!("{prefix}.weight"), placement.clone());
        self.insert(format!("{prefix}.scales"), placement.clone());
        if has_biases {
            self.insert(format!("{prefix}.biases"), placement);
        }
    }

    /// Adds a tensor-parallel shard using this rank's TP coordinate.
    pub fn insert_tensor_parallel(&mut self, target: impl Into<String>, axis: usize) {
        self.insert(
            target,
            TensorPlacement::Shard {
                axis,
                index: self.topology.tensor_parallel_rank,
                parts: self.topology.tensor_parallel_size,
            },
        );
    }

    /// Returns an explicit tensor placement by rewritten target name.
    pub fn placement(&self, target: &str) -> Option<&TensorPlacement> {
        self.tensors.get(target).map(|plan| &plan.placement)
    }

    /// Validates every placement whose constraints are known before loading.
    ///
    /// Axis bounds and divisibility require `insert_expected`; ownership and
    /// shard-coordinate bounds are validated for all entries.
    pub fn validate(&self) -> Result<(), Error> {
        for (target, tensor) in &self.tensors {
            validate_plan_entry(tensor, self.topology).map_err(|error| {
                Error::Parallel(format!("placement for tensor {target}: {error}"))
            })?;
        }
        if let Some(default) = &self.default {
            validate_plan_entry(
                &TensorPlan {
                    placement: default.clone(),
                    expected_source_shape: None,
                },
                self.topology,
            )?;
        }
        Ok(())
    }

    fn source_plan(&self, source: &str, config: &StrictLoadConfig) -> SourcePlan {
        for candidate in config.candidates(source) {
            if let Some(plan) = self.tensors.get(&candidate) {
                return SourcePlan::Known {
                    target: candidate,
                    tensor: plan.clone(),
                };
            }
        }
        if let Some(placement) = &self.default {
            let target = config
                .candidates(source)
                .into_iter()
                .next()
                .unwrap_or_else(|| source.to_string());
            SourcePlan::Known {
                target,
                tensor: TensorPlan {
                    placement: placement.clone(),
                    expected_source_shape: None,
                },
            }
        } else {
            SourcePlan::Unexpected
        }
    }
}

fn validate_plan_entry(plan: &TensorPlan, topology: ParallelTopology) -> Result<(), Error> {
    match &plan.placement {
        TensorPlacement::Rank { rank } if *rank >= topology.world_size => {
            Err(Error::Parallel(format!(
                "owner rank {rank} is outside world size {}",
                topology.world_size
            )))
        }
        TensorPlacement::PipelineStage { stage } if *stage >= topology.pipeline_parallel_size => {
            Err(Error::Parallel(format!(
                "pipeline owner stage {stage} is outside {} stages",
                topology.pipeline_parallel_size
            )))
        }
        TensorPlacement::Shard { index, parts, .. } if *parts == 0 || *index >= *parts => {
            Err(Error::Parallel(format!(
                "tensor shard index {index} is invalid for {parts} parts"
            )))
        }
        placement => {
            if let Some(shape) = &plan.expected_source_shape {
                validate_placement(placement, shape, topology)?;
            }
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
enum SourcePlan {
    Known { target: String, tensor: TensorPlan },
    Unexpected,
}

#[derive(Debug)]
enum ResolvedPlacement {
    Materialize,
    Omit,
    Shard(TensorSlice),
}

fn validate_placement(
    placement: &TensorPlacement,
    shape: &[usize],
    topology: ParallelTopology,
) -> Result<(), Error> {
    match placement {
        TensorPlacement::Rank { rank } if *rank >= topology.world_size => {
            Err(Error::Parallel(format!(
                "owner rank {rank} is outside world size {}",
                topology.world_size
            )))
        }
        TensorPlacement::PipelineStage { stage } if *stage >= topology.pipeline_parallel_size => {
            Err(Error::Parallel(format!(
                "pipeline owner stage {stage} is outside {} stages",
                topology.pipeline_parallel_size
            )))
        }
        TensorPlacement::Shard { axis, index, parts } => {
            TensorSlice::for_shape(shape, *axis, *index, *parts).map(|_| ())
        }
        _ => Ok(()),
    }
}

fn resolve_placement(
    plan: &TensorPlan,
    shape: &[usize],
    topology: ParallelTopology,
) -> Result<ResolvedPlacement, Error> {
    if let Some(expected) = &plan.expected_source_shape {
        if expected != shape {
            return Err(Error::Parallel(format!(
                "expected checkpoint shape {expected:?}, got {shape:?}"
            )));
        }
    }
    validate_placement(&plan.placement, shape, topology)?;
    Ok(match &plan.placement {
        TensorPlacement::Replicated | TensorPlacement::Local => ResolvedPlacement::Materialize,
        TensorPlacement::Omit => ResolvedPlacement::Omit,
        TensorPlacement::Rank { rank } => {
            if *rank == topology.global_rank {
                ResolvedPlacement::Materialize
            } else {
                ResolvedPlacement::Omit
            }
        }
        TensorPlacement::PipelineStage { stage } => {
            if *stage == topology.pipeline_parallel_rank {
                ResolvedPlacement::Materialize
            } else {
                ResolvedPlacement::Omit
            }
        }
        TensorPlacement::Shard { axis, index, parts } => {
            ResolvedPlacement::Shard(TensorSlice::for_shape(shape, *axis, *index, *parts)?)
        }
    })
}

/// Locally materialized checkpoint partition.
///
/// This is intentionally not an executable model. Later distributed execution
/// phases can consume it together with a communication group without storing a
/// borrowed group inside long-lived model state.
#[derive(Debug)]
pub struct RankPartition {
    topology: ParallelTopology,
    tensors: HashMap<String, Array>,
    opened_shards: Vec<PathBuf>,
}

impl RankPartition {
    /// Returns the validated topology used for this partition.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }

    /// Returns a locally materialized tensor by rewritten target name.
    pub fn get(&self, target: &str) -> Option<&Array> {
        self.tensors.get(target)
    }

    /// Iterates over locally materialized tensors.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Array)> {
        self.tensors
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// Returns the number of locally materialized tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether this partition contains no local tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns checkpoint payload shards that were actually opened.
    pub fn opened_shards(&self) -> &[PathBuf] {
        &self.opened_shards
    }
}

#[derive(Default)]
struct PartitionReport {
    loaded: HashSet<String>,
    unexpected: Vec<String>,
}

impl PartitionReport {
    fn finish(self, plan: &PlacementPlan, config: &StrictLoadConfig) -> Result<(), Error> {
        let mut missing = Vec::new();
        for (target, tensor) in &plan.tensors {
            let locally_required = match tensor.placement {
                TensorPlacement::Replicated
                | TensorPlacement::Local
                | TensorPlacement::Shard { .. } => true,
                TensorPlacement::Omit => false,
                TensorPlacement::Rank { rank } => rank == plan.topology.global_rank,
                TensorPlacement::PipelineStage { stage } => {
                    stage == plan.topology.pipeline_parallel_rank
                }
            };
            if locally_required && !self.loaded.contains(target) {
                missing.push(target.clone());
            }
        }
        missing.sort();
        let mut unexpected = self
            .unexpected
            .into_iter()
            .filter(|source| !config.is_unused_allowed(source))
            .collect::<Vec<_>>();
        unexpected.sort();
        unexpected.dedup();
        if missing.is_empty() && unexpected.is_empty() {
            Ok(())
        } else {
            Err(Error::StrictLoadValidation {
                missing,
                unused: unexpected,
            })
        }
    }
}

/// Selectively loads a safetensors checkpoint directory according to `plan`.
///
/// For indexed checkpoints, key rewrites and placement are resolved from the
/// index before any payload shard is opened. A shard containing no local
/// tensors is therefore skipped completely. Within an opened shard, omitted
/// tensors never become MLX arrays. Selected source views are sliced before
/// their final stream copy, then explicitly evaluated while the mmap is alive.
/// Peak temporary memory is bounded by the accumulated local partition plus at
/// most the selected source tensor currently being transformed.
pub fn load_safetensors_partition(
    model_dir: impl AsRef<Path>,
    plan: &PlacementPlan,
    stream: &Stream,
    config: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    load_safetensors_partition_on_streams(model_dir, plan, stream, stream, config)
}

/// Selectively loads on a source/weights stream, then places only local results
/// on `execution_stream`.
///
/// Use a CPU `source_stream` with a GPU `execution_stream` to ensure a full
/// source tensor is never copied to the GPU merely to discard other ranks'
/// slices. The source device holds at most the tensor currently being
/// transformed in addition to the accumulated local partition.
pub fn load_safetensors_partition_on_streams(
    model_dir: impl AsRef<Path>,
    plan: &PlacementPlan,
    source_stream: &Stream,
    execution_stream: &Stream,
    config: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    let model_dir = model_dir.as_ref();
    plan.validate()?;
    plan.topology.validate_execution_stream(execution_stream)?;
    let index_path = model_dir.join("model.safetensors.index.json");
    let mut report = PartitionReport::default();
    let mut tensors = HashMap::new();
    let mut opened_shards = Vec::new();

    if index_path.exists() {
        let index: WeightMap = serde_json::from_str(&std::fs::read_to_string(index_path)?)?;
        let mut selected_by_file: HashMap<String, HashSet<String>> = HashMap::new();
        for (source, file) in &index.weight_map {
            match plan.source_plan(source, config) {
                SourcePlan::Unexpected => report.unexpected.push(source.clone()),
                SourcePlan::Known { tensor, .. } => {
                    let potentially_local = !matches!(tensor.placement, TensorPlacement::Omit)
                        && !matches!(tensor.placement, TensorPlacement::Rank { rank } if rank != plan.topology.global_rank)
                        && !matches!(tensor.placement, TensorPlacement::PipelineStage { stage } if stage != plan.topology.pipeline_parallel_rank);
                    if potentially_local {
                        selected_by_file
                            .entry(file.clone())
                            .or_default()
                            .insert(source.clone());
                    }
                }
            }
        }
        let mut files = selected_by_file.into_iter().collect::<Vec<_>>();
        files.sort_by(|left, right| left.0.cmp(&right.0));
        for (file, selected_sources) in files {
            let path = model_dir.join(file);
            load_selected_shard(
                &path,
                Some(&selected_sources),
                plan,
                source_stream,
                execution_stream,
                config,
                &mut tensors,
                &mut report,
            )?;
            opened_shards.push(path);
        }
    } else {
        let path = if model_dir
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            model_dir.to_path_buf()
        } else {
            model_dir.join("model.safetensors")
        };
        load_selected_shard(
            &path,
            None,
            plan,
            source_stream,
            execution_stream,
            config,
            &mut tensors,
            &mut report,
        )?;
        opened_shards.push(path);
    }

    report.finish(plan, config)?;
    Ok(RankPartition {
        topology: plan.topology,
        tensors,
        opened_shards,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_selected_shard(
    path: &Path,
    _selected_sources: Option<&HashSet<String>>,
    plan: &PlacementPlan,
    source_stream: &Stream,
    execution_stream: &Stream,
    config: &StrictLoadConfig,
    output: &mut HashMap<String, Array>,
    report: &mut PartitionReport,
) -> Result<(), Error> {
    let file = File::open(path)?;
    // SAFETY: the mapping remains alive through explicit evaluation and stream
    // synchronization of every Array/view derived from it below.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let checkpoint =
        SafeTensors::deserialize(&mmap).map_err(|error| Error::Other(Box::new(error)))?;
    for (source, view) in checkpoint.iter() {
        let SourcePlan::Known { target, tensor } = plan.source_plan(source, config) else {
            report.unexpected.push(source.to_string());
            continue;
        };
        let shape = view.shape().to_vec();
        let resolved = resolve_placement(&tensor, &shape, plan.topology).map_err(|error| {
            Error::Parallel(format!("checkpoint tensor {source} -> {target}: {error}"))
        })?;
        let value = match resolved {
            ResolvedPlacement::Omit => continue,
            ResolvedPlacement::Materialize => {
                let source_value =
                    Array::try_from(view).map_err(|error| Error::Other(Box::new(error)))?;
                source_value.copy(execution_stream)?
            }
            ResolvedPlacement::Shard(slice) => {
                let source_value =
                    Array::try_from(view).map_err(|error| Error::Other(Box::new(error)))?;
                let parts = i32::try_from(slice.parts).map_err(|_| {
                    Error::Parallel("tensor shard count does not fit in i32".into())
                })?;
                let axis = i32::try_from(slice.axis)
                    .map_err(|_| Error::Parallel("tensor axis does not fit in i32".into()))?;
                // Move the sharded axis to the front, where equal `split`
                // produces lazy contiguous range views. Only the selected
                // view is moved back and evaluated/copied, so other ranks'
                // pieces do not become execution-device allocations.
                let front = if slice.axis == 0 {
                    source_value
                } else {
                    source_value.move_axis(axis, 0, source_stream)?
                };
                let selected = front
                    .split(parts, Some(0), source_stream)?
                    .into_iter()
                    .nth(slice.index)
                    .expect("validated shard index");
                let selected = if slice.axis == 0 {
                    selected
                } else {
                    selected.move_axis(0, axis, source_stream)?
                };
                let selected = if slice.axis == 0 {
                    selected
                } else {
                    // An inner-axis range is a non-contiguous view. Compact
                    // only this rank's selected view before final placement;
                    // the temporary is local-slice-sized, never source-sized.
                    let local_shape = slice
                        .local_shape(&shape)
                        .into_iter()
                        .map(|dimension| {
                            i32::try_from(dimension).map_err(|_| {
                                Error::Parallel("local tensor dimension does not fit in i32".into())
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    selected
                        .flatten(None, None, source_stream)?
                        .reshape(&local_shape, source_stream)?
                };
                selected.copy(execution_stream)?
            }
        };
        eval([&value])?;
        source_stream.synchronize()?;
        execution_stream.synchronize()?;
        report.loaded.insert(target.clone());
        if output.insert(target.clone(), value).is_some() {
            return Err(Error::Parallel(format!(
                "multiple checkpoint tensors resolved to local target {target}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn write_index(dir: &Path, mappings: &[(&str, &str)]) {
        let weight_map = mappings
            .iter()
            .map(|(key, file)| ((*key).to_string(), serde_json::json!(file)))
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {},
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_i32_tensor(path: &Path, name: &str, values: &[i32], shape: Vec<usize>) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::I32, shape, &bytes).unwrap();
        serialize_to_file([(name, view)], None, path).unwrap();
    }

    fn topology(world: usize, rank: usize, tp: usize, pp: usize, ep: usize) -> ParallelTopology {
        ParallelTopology::from_rank(
            world,
            rank,
            tp,
            pp,
            ep,
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
        .unwrap()
    }

    #[test]
    fn validates_topology_and_exhaustive_coordinate_ordering() {
        assert!(ParallelTopology::from_rank(
            8,
            0,
            0,
            2,
            2,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            1,
            0,
            1,
            1,
            1,
            DeviceAssignment::new(DeviceType::Cpu, usize::MAX)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            7,
            0,
            2,
            2,
            2,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            8,
            8,
            2,
            2,
            2,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            usize::MAX,
            0,
            usize::MAX,
            2,
            1,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());

        for pp_rank in 0..2 {
            for tp_rank in 0..3 {
                for ep_rank in 0..2 {
                    let rank = ((pp_rank * 3) + tp_rank) * 2 + ep_rank;
                    let value = ParallelTopology::from_rank(
                        12,
                        rank,
                        3,
                        2,
                        2,
                        DeviceAssignment::new(DeviceType::Cpu, 99),
                    )
                    .unwrap();
                    assert_eq!(value.pipeline_parallel_rank, pp_rank);
                    assert_eq!(value.tensor_parallel_rank, tp_rank);
                    assert_eq!(value.expert_parallel_rank, ep_rank);
                    assert_eq!(value.device.local_index, 99);
                    assert_ne!(value.device.local_index, value.global_rank);
                }
            }
        }
    }

    #[test]
    fn balanced_ranges_cover_uneven_layers_and_experts() {
        let ranges = (0..3)
            .map(|index| balanced_contiguous_range(8, 3, index, false).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ranges, [0..3, 3..6, 6..8]);
        assert!(balanced_contiguous_range(2, 3, 0, false).is_err());
        assert_eq!(balanced_contiguous_range(2, 3, 2, true).unwrap(), 2..2);
        assert_eq!(topology(6, 5, 1, 3, 2).layer_range(8).unwrap(), 6..8);
        assert_eq!(topology(6, 5, 1, 3, 2).expert_range(5).unwrap(), 3..5);
    }

    #[test]
    fn validates_tensor_slices() {
        let slice = TensorSlice::for_shape(&[4, 12], 1, 2, 3).unwrap();
        assert_eq!(slice.start, 8);
        assert_eq!(slice.end, 12);
        assert_eq!(slice.local_shape(&[4, 12]), [4, 4]);
        assert!(TensorSlice::for_shape(&[4, 11], 1, 0, 3).is_err());
        assert!(TensorSlice::for_shape(&[4, 12], 2, 0, 3).is_err());
        assert!(TensorSlice::for_shape(&[4, 12], 1, 3, 3).is_err());
    }

    #[test]
    fn validates_explicit_execution_stream_device() {
        let stream = stream();
        topology(1, 0, 1, 1, 1)
            .validate_execution_stream(&stream)
            .unwrap();
        let other_assignment =
            ParallelTopology::from_rank(1, 0, 1, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 1))
                .unwrap();
        assert!(other_assignment.validate_execution_stream(&stream).is_err());
    }

    #[test]
    fn plan_exposes_replicated_omitted_and_quantized_companions() {
        let mut plan = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        plan.insert("replicated", TensorPlacement::Replicated);
        plan.insert("remote", TensorPlacement::Omit);
        plan.insert_quantized_companions("projection", TensorPlacement::Local, true);
        assert_eq!(
            plan.placement("replicated"),
            Some(&TensorPlacement::Replicated)
        );
        assert_eq!(plan.placement("remote"), Some(&TensorPlacement::Omit));
        assert_eq!(
            plan.placement("projection.weight"),
            Some(&TensorPlacement::Local)
        );
        assert_eq!(
            plan.placement("projection.scales"),
            Some(&TensorPlacement::Local)
        );
        assert_eq!(
            plan.placement("projection.biases"),
            Some(&TensorPlacement::Local)
        );

        let mut invalid = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        invalid.insert("bad_owner", TensorPlacement::Rank { rank: 1 });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn model_load_options_preserve_singleton_behavior_and_reject_partial_models() {
        let default = crate::models::ModelLoadOptions::default();
        assert_eq!(default.quantization, None);
        assert_eq!(default.parallel, None);
        crate::models::ensure_executable_load_options(default).unwrap();

        let singleton = crate::models::ModelLoadOptions::with_parallel(topology(1, 0, 1, 1, 1));
        crate::models::ensure_executable_load_options(singleton).unwrap();
        let combined = crate::models::ModelLoadOptions::with_quantization(
            crate::quantization::WeightQuantization::MxFp4,
        )
        .with_parallel_topology(topology(1, 0, 1, 1, 1));
        assert_eq!(
            combined.quantization,
            Some(crate::quantization::WeightQuantization::MxFp4)
        );
        assert!(combined.parallel.unwrap().is_replicated());

        let partitioned = crate::models::ModelLoadOptions::with_parallel(topology(2, 0, 2, 1, 1));
        assert!(matches!(
            crate::models::ensure_executable_load_options(partitioned),
            Err(Error::Parallel(_))
        ));
    }

    #[test]
    fn typed_rank_and_pipeline_ownership_resolve_locally() {
        let rank_zero = topology(4, 0, 2, 2, 1);
        let rank_three = topology(4, 3, 2, 2, 1);
        let rank_owned = TensorPlan {
            placement: TensorPlacement::Rank { rank: 3 },
            expected_source_shape: None,
        };
        assert!(matches!(
            resolve_placement(&rank_owned, &[2], rank_zero).unwrap(),
            ResolvedPlacement::Omit
        ));
        assert!(matches!(
            resolve_placement(&rank_owned, &[2], rank_three).unwrap(),
            ResolvedPlacement::Materialize
        ));

        let stage_owned = TensorPlan {
            placement: TensorPlacement::PipelineStage { stage: 1 },
            expected_source_shape: None,
        };
        assert!(matches!(
            resolve_placement(&stage_owned, &[2], rank_zero).unwrap(),
            ResolvedPlacement::Omit
        ));
        assert!(matches!(
            resolve_placement(&stage_owned, &[2], rank_three).unwrap(),
            ResolvedPlacement::Materialize
        ));
    }

    #[test]
    fn selective_loader_skips_remote_shards_and_reconstructs_tp_slices() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("local.safetensors"),
            "model.projection.weight",
            &[0, 1, 2, 3, 10, 11, 12, 13],
            vec![2, 4],
        );
        // This is deliberately not a safetensors file. Correct index-level
        // selection must never open it for either rank.
        std::fs::write(dir.path().join("remote.safetensors"), b"must not be opened").unwrap();
        write_index(
            dir.path(),
            &[
                ("model.projection.weight", "local.safetensors"),
                ("model.remote.weight", "remote.safetensors"),
            ],
        );

        let mut reconstructed = Vec::new();
        for rank in 0..2 {
            let topology = topology(2, rank, 2, 1, 1);
            let mut plan = PlacementPlan::new(topology);
            plan.insert_expected(
                "projection.weight",
                vec![2, 4],
                TensorPlacement::Shard {
                    axis: 1,
                    index: rank,
                    parts: 2,
                },
            )
            .unwrap();
            plan.insert("remote.weight", TensorPlacement::Omit);
            let config = StrictLoadConfig::default().strip_prefix("model.");
            let partition =
                load_safetensors_partition(dir.path(), &plan, &stream, &config).unwrap();
            assert_eq!(partition.len(), 1);
            assert_eq!(
                partition.opened_shards(),
                &[dir.path().join("local.safetensors")]
            );
            assert!(partition.get("remote.weight").is_none());
            let local = partition
                .get("projection.weight")
                .unwrap()
                .evaluated()
                .unwrap();
            assert_eq!(local.as_array().shape(), &[2, 2]);
            reconstructed.push(local.as_slice::<i32>().to_vec());
        }
        // Slices are axis-1 contiguous views, so reconstruct each row from
        // the corresponding rows of both rank-local tensors.
        assert_eq!(reconstructed[0], [0, 1, 10, 11]);
        assert_eq!(reconstructed[1], [2, 3, 12, 13]);
        let union = [
            reconstructed[0][0],
            reconstructed[0][1],
            reconstructed[1][0],
            reconstructed[1][1],
            reconstructed[0][2],
            reconstructed[0][3],
            reconstructed[1][2],
            reconstructed[1][3],
        ];
        assert_eq!(union, [0, 1, 2, 3, 10, 11, 12, 13]);
    }

    #[test]
    fn replicated_default_loads_the_original_full_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "weight",
            &[3, 5, 7, 9],
            vec![2, 2],
        );
        let plan = PlacementPlan::replicated(topology(1, 0, 1, 1, 1));
        let partition =
            load_safetensors_partition(dir.path(), &plan, &stream, &StrictLoadConfig::default())
                .unwrap();
        let loaded = partition.get("weight").unwrap().evaluated().unwrap();
        assert_eq!(loaded.as_slice::<i32>(), &[3, 5, 7, 9]);
    }

    #[test]
    fn omitted_unsupported_tensor_is_never_materialized() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = [0u8; 4];
        let unsupported = TensorView::new(Dtype::F8_E5M2, vec![4], &bytes).unwrap();
        serialize_to_file(
            [("remote", unsupported)],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let mut plan = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        plan.insert("remote", TensorPlacement::Omit);
        let partition =
            load_safetensors_partition(dir.path(), &plan, &stream(), &StrictLoadConfig::default())
                .unwrap();
        assert!(partition.is_empty());
    }

    #[test]
    fn remote_only_index_shard_is_never_opened() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("remote.safetensors"), b"not safetensors").unwrap();
        write_index(dir.path(), &[("remote.weight", "remote.safetensors")]);
        let mut plan = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        plan.insert("remote.weight", TensorPlacement::Omit);
        let partition =
            load_safetensors_partition(dir.path(), &plan, &stream(), &StrictLoadConfig::default())
                .unwrap();
        assert!(partition.is_empty());
        assert!(partition.opened_shards().is_empty());
    }

    #[test]
    fn strict_partition_rejects_missing_malformed_and_unexpected_local_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "present",
            &[1, 2, 3, 4],
            vec![2, 2],
        );
        let topology = topology(1, 0, 1, 1, 1);

        let mut malformed = PlacementPlan::new(topology);
        malformed
            .insert_expected("present", vec![4, 2], TensorPlacement::Local)
            .unwrap();
        assert!(matches!(
            load_safetensors_partition(
                dir.path(),
                &malformed,
                &stream,
                &StrictLoadConfig::default()
            ),
            Err(Error::Parallel(_))
        ));

        let mut missing = PlacementPlan::new(topology);
        missing.insert("present", TensorPlacement::Omit);
        missing.insert("required", TensorPlacement::Local);
        let error =
            load_safetensors_partition(dir.path(), &missing, &stream, &StrictLoadConfig::default())
                .unwrap_err();
        match error {
            Error::StrictLoadValidation { missing, unused } => {
                assert_eq!(missing, ["required"]);
                assert!(unused.is_empty());
            }
            other => panic!("unexpected error: {other}"),
        }

        let strict_empty = PlacementPlan::new(topology);
        let error = load_safetensors_partition(
            dir.path(),
            &strict_empty,
            &stream,
            &StrictLoadConfig::default(),
        )
        .unwrap_err();
        match error {
            Error::StrictLoadValidation { missing, unused } => {
                assert!(missing.is_empty());
                assert_eq!(unused, ["present"]);
            }
            other => panic!("unexpected error: {other}"),
        }

        let allowed = load_safetensors_partition(
            dir.path(),
            &strict_empty,
            &stream,
            &StrictLoadConfig::default().allow_unused_prefix("present"),
        )
        .unwrap();
        assert!(allowed.is_empty());
    }
}
