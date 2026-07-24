//! Reusable expert-parallel assignment, routing, and exchange infrastructure.
//!
//! Pure expert parallelism keeps ordinary model state replicated and partitions
//! only routed expert banks.  [`dispatch_replicated`] exploits the replicated
//! token layout: ranks compact only routes owned by their experts and all-sum
//! the resulting token buffer.  [`all_to_all_v`] is the general sharded-token
//! transport.  It is intentionally an all-gather fallback and therefore uses
//! `O(group_size)` temporary replication until MLX exposes native all-to-all.

use std::{
    cell::Cell,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Group},
    error::Exception,
    module::{ModuleParameters, Param},
    ops::{concatenate_axis, indexing::TryIndexOp, r#where, segment_sum_by_index, zeros_dtype},
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache},
    cache_residency::{
        open_prompt_cache, validate_prompt_cache_model_identity, CacheRankIdentity,
        CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions,
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
        PromptCacheTopology,
    },
    error::Error,
    expert_cache::{
        AcquiredExperts, ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertPass,
    },
    inspection::ActivationObserver,
    models::{
        deepseek_v3, gpt_oss, inkling, input as runtime_input, lfm2, nemotron_h, qwen3,
        qwen3_5_moe, qwen3_next, qwen3_vl, ModelKind, ModelLoadOptions,
    },
    mtp::{MtpCapability, MtpCheckpointKind, MtpConfig, MtpStats},
    parallel::{
        load_safetensors_partition_from_store_on_streams, ParallelTopology, PlacementPlan,
        TensorPlacement,
    },
    pipeline::{assign_module, assign_module_excluding, load_deepseek_experts, SynchronizedToken},
    quantization::{should_quantize_on_load, WeightQuantization},
    sampler::{DefaultSampler, Sampler, SpeculativeSampler},
    weight_store::{SafetensorsWeightStore, WeightStore},
    weights::{transform_split_swiglu_experts, StrictLoadConfig},
};

use crate::layerwise::WeightResidency;

use crate::models::{
    common::moe::{quantize_expert_bank, PackedRelu2Experts, PackedSwiGluExperts},
    deepseek_v3::RoutedExperts,
};

thread_local! {
    static EAGER_TIMING_PROFILING: Cell<bool> = const { Cell::new(false) };
}

/// Scoped opt-in profiling mode for expert-parallel phase timings.
///
/// MLX executes lazily, so ordinary phase timings primarily describe graph
/// submission. While this guard is alive, expert-parallel code materializes
/// phase outputs before stopping each timer. This makes the measurements useful
/// for benchmarks, at the cost of extra synchronization and changed scheduling.
#[must_use]
pub struct ExpertParallelTimingGuard {
    previous: bool,
}

impl Drop for ExpertParallelTimingGuard {
    fn drop(&mut self) {
        EAGER_TIMING_PROFILING.with(|enabled| enabled.set(self.previous));
    }
}

/// Enables device-complete expert-parallel phase timings for the current thread.
pub fn profile_expert_parallel_timings() -> ExpertParallelTimingGuard {
    let previous = EAGER_TIMING_PROFILING.with(|enabled| {
        let previous = enabled.get();
        enabled.set(true);
        previous
    });
    ExpertParallelTimingGuard { previous }
}

pub(crate) fn timing_profiling_enabled() -> bool {
    EAGER_TIMING_PROFILING.with(Cell::get)
}

pub(crate) fn materialize_timing_phase<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    if timing_profiling_enabled() {
        eval(outputs)?;
    }
    Ok(())
}

/// Policy used to assign global routed experts to ranks.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ExpertAssignmentPolicy {
    /// Balanced contiguous ranges, with lower ranks receiving any remainder.
    BalancedContiguous,
    /// Expert `e` is owned by rank `e % group_size`.
    RoundRobin,
    /// Explicit global-expert-to-owner-rank table.
    Explicit(Vec<usize>),
}

/// Validated bidirectional mapping between checkpoint-global and owner-local ids.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertAssignment {
    global_expert_count: usize,
    group_size: usize,
    rank: usize,
    policy: ExpertAssignmentPolicy,
    owners: Vec<usize>,
    owner_local: Vec<usize>,
    local_global: Vec<usize>,
}

impl ExpertAssignment {
    /// Creates the default balanced contiguous assignment.
    pub fn balanced(global_experts: usize, group_size: usize, rank: usize) -> Result<Self, Error> {
        Self::balanced_with_empty(global_experts, group_size, rank, false)
    }

    /// Creates a balanced assignment and optionally permits empty ranks.
    pub fn balanced_with_empty(
        global_experts: usize,
        group_size: usize,
        rank: usize,
        allow_empty: bool,
    ) -> Result<Self, Error> {
        validate_dimensions(global_experts, group_size, rank, allow_empty)?;
        let base = global_experts / group_size;
        let extra = global_experts % group_size;
        let mut owners = Vec::with_capacity(global_experts);
        for owner in 0..group_size {
            owners.extend(std::iter::repeat_n(
                owner,
                base + usize::from(owner < extra),
            ));
        }
        Self::from_owners_impl(
            owners,
            group_size,
            rank,
            ExpertAssignmentPolicy::BalancedContiguous,
            allow_empty,
        )
    }

    /// Creates a deterministic round-robin assignment.
    pub fn round_robin(
        global_experts: usize,
        group_size: usize,
        rank: usize,
    ) -> Result<Self, Error> {
        validate_dimensions(global_experts, group_size, rank, false)?;
        let owners = (0..global_experts)
            .map(|expert| expert % group_size)
            .collect();
        Self::from_owners_impl(
            owners,
            group_size,
            rank,
            ExpertAssignmentPolicy::RoundRobin,
            false,
        )
    }

    /// Creates an assignment from one owner rank per global expert.
    pub fn explicit(owners: Vec<usize>, group_size: usize, rank: usize) -> Result<Self, Error> {
        let policy = ExpertAssignmentPolicy::Explicit(owners.clone());
        Self::from_owners_impl(owners, group_size, rank, policy, false)
    }

    /// Creates an explicit assignment and optionally permits empty ranks.
    pub fn explicit_with_empty(
        owners: Vec<usize>,
        group_size: usize,
        rank: usize,
        allow_empty: bool,
    ) -> Result<Self, Error> {
        let policy = ExpertAssignmentPolicy::Explicit(owners.clone());
        Self::from_owners_impl(owners, group_size, rank, policy, allow_empty)
    }

    fn from_owners_impl(
        owners: Vec<usize>,
        group_size: usize,
        rank: usize,
        policy: ExpertAssignmentPolicy,
        allow_empty: bool,
    ) -> Result<Self, Error> {
        validate_dimensions(owners.len(), group_size, rank, allow_empty)?;
        if let Some((expert, owner)) = owners
            .iter()
            .copied()
            .enumerate()
            .find(|(_, owner)| *owner >= group_size)
        {
            return Err(Error::Parallel(format!(
                "global expert {expert} has invalid owner rank {owner} for EP size {group_size}"
            )));
        }
        let mut next_local = vec![0usize; group_size];
        let mut owner_local = Vec::with_capacity(owners.len());
        let mut local_global = Vec::new();
        for (global, owner) in owners.iter().copied().enumerate() {
            owner_local.push(next_local[owner]);
            next_local[owner] = next_local[owner].checked_add(1).ok_or_else(|| {
                Error::Parallel("owner-local expert index overflowed usize".into())
            })?;
            if owner == rank {
                local_global.push(global);
            }
        }
        if !allow_empty && next_local.contains(&0) {
            return Err(Error::Parallel(format!(
                "expert assignment creates an empty rank: counts {next_local:?}"
            )));
        }
        if owners.len() > i32::MAX as usize
            || next_local.iter().any(|count| *count > i32::MAX as usize)
        {
            return Err(Error::Parallel(
                "expert assignment exceeds MLX i32 indexing limits".into(),
            ));
        }
        Ok(Self {
            global_expert_count: owners.len(),
            group_size,
            rank,
            policy,
            owners,
            owner_local,
            local_global,
        })
    }

    /// Total checkpoint-global routed expert count.
    pub const fn global_expert_count(&self) -> usize {
        self.global_expert_count
    }
    /// EP group size.
    pub const fn group_size(&self) -> usize {
        self.group_size
    }
    /// Current rank within the EP group.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Assignment policy.
    pub fn policy(&self) -> &ExpertAssignmentPolicy {
        &self.policy
    }
    /// Global expert ids owned by this rank, in owner-local order.
    pub fn local_global_expert_ids(&self) -> &[usize] {
        &self.local_global
    }
    /// Number of experts owned by this rank.
    pub fn local_expert_count(&self) -> usize {
        self.local_global.len()
    }
    /// Returns the owner rank of a global expert.
    pub fn owner(&self, global: usize) -> Option<usize> {
        self.owners.get(global).copied()
    }
    /// Returns the owner-local id of a global expert.
    pub fn owner_local_id(&self, global: usize) -> Option<usize> {
        self.owner_local.get(global).copied()
    }
    /// Returns the global id corresponding to a local id on this rank.
    pub fn global_id(&self, local: usize) -> Option<usize> {
        self.local_global.get(local).copied()
    }
    /// Complete global-to-owner mapping.
    pub fn owners(&self) -> &[usize] {
        &self.owners
    }
    /// Complete global-to-owner-local mapping.
    pub fn owner_local_ids(&self) -> &[usize] {
        &self.owner_local
    }
}

fn validate_dimensions(
    global_experts: usize,
    group_size: usize,
    rank: usize,
    allow_empty: bool,
) -> Result<(), Error> {
    if global_experts == 0 || group_size == 0 {
        return Err(Error::Parallel(
            "expert count and EP size must be nonzero".into(),
        ));
    }
    if rank >= group_size {
        return Err(Error::Parallel(format!(
            "EP rank {rank} is outside size {group_size}"
        )));
    }
    if !allow_empty && global_experts < group_size {
        return Err(Error::Parallel(format!(
            "cannot assign {global_experts} experts to {group_size} non-empty ranks"
        )));
    }
    Ok(())
}

/// Token ownership layout used by expert dispatch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TokenLayout {
    /// Every EP rank has identical hidden rows and router results.
    Replicated,
    /// Each source rank owns disjoint hidden rows and exchanges routes.
    Sharded,
}

/// Transport selected for expert routes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertExchangeStrategy {
    /// Compact local routes, execute local experts, and all-sum token outputs.
    ReplicatedInputAllSum,
    /// Variable-count all-to-all emulated with padded all-gather.
    AllGatherAllToAllV,
}

/// Per-dispatch counters used by diagnostics and benchmark probes.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct RoutingStatistics {
    /// Total selected routes visible to the source rank.
    pub total_routes: usize,
    /// Selected routes owned and executed by this rank.
    pub local_routes: usize,
    /// Routes sent by a sharded-input exchange.
    pub sent_routes: usize,
    /// Routes received by a sharded-input exchange.
    pub received_routes: usize,
    /// Padding rows introduced by the fallback transport.
    pub padding_routes: usize,
    /// Explicit host-visible synchronization points.
    pub synchronization_count: usize,
    /// Payload bytes transferred logically, excluding backend internals.
    pub exchanged_bytes: usize,
    /// Time spent waiting for explicit route metadata synchronization.
    pub synchronization_time: Duration,
    /// Wall time spent computing router decisions.
    pub router_time: Duration,
    /// Wall time spent validating and compacting owner-local routes.
    pub compaction_time: Duration,
    /// Wall time spent in route transport collectives.
    pub exchange_time: Duration,
    /// Wall time spent in local expert computation.
    pub expert_time: Duration,
    /// Wall time spent reducing or recombining routed outputs.
    pub reduction_time: Duration,
    /// Wall time spent computing replicated shared experts.
    pub shared_expert_time: Duration,
    /// End-to-end wall time summed across represented MoE blocks or one dispatch.
    pub total_time: Duration,
    /// End-to-end wall time for the complete model forward containing those blocks.
    pub model_time: Duration,
}

impl RoutingStatistics {
    /// Adds counters and measured synchronization time from another dispatch.
    pub fn accumulate(&mut self, other: &Self) {
        self.total_routes += other.total_routes;
        self.local_routes += other.local_routes;
        self.sent_routes += other.sent_routes;
        self.received_routes += other.received_routes;
        self.padding_routes += other.padding_routes;
        self.synchronization_count += other.synchronization_count;
        self.exchanged_bytes += other.exchanged_bytes;
        self.synchronization_time += other.synchronization_time;
        self.router_time += other.router_time;
        self.compaction_time += other.compaction_time;
        self.exchange_time += other.exchange_time;
        self.expert_time += other.expert_time;
        self.reduction_time += other.reduction_time;
        self.shared_expert_time += other.shared_expert_time;
        self.total_time += other.total_time;
        self.model_time += other.model_time;
    }
}

/// Compact device-side routes owned by the current rank.
pub struct DispatchedRoutes {
    /// Hidden rows in stable original route order.
    pub hidden: Array,
    /// Checkpoint-global expert ids.
    pub global_expert_ids: Array,
    /// Dense owner-local ids passed to grouped kernels.
    pub local_expert_ids: Array,
    /// Original flattened route positions.
    pub original_route_indices: Array,
    /// Source token indices.
    pub token_indices: Array,
    /// Top-k slot indices.
    pub slot_indices: Array,
    /// Route weights, not yet applied.
    pub weights: Array,
}

/// Result of a replicated-input expert dispatch.
pub struct ReturnedRoutes {
    /// Rank-local weighted token buffer before the collective.
    pub local_output: Array,
    /// Exact routed output after all-sum.
    pub reduced_output: Array,
    /// Dispatch counters.
    pub statistics: RoutingStatistics,
}

/// Architecture-specific execution behind the common route dispatcher.
pub trait LocalExpertBank {
    /// Executes compact hidden rows using dense owner-local expert ids.
    /// Returned route rows must be unweighted and retain input order.
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error>;
}

fn unit_route_weights(routes: i32, dtype: Dtype, stream: &Stream) -> Result<Array, Error> {
    Ok(safemlx::ops::ones_dtype(&[routes, 1], dtype, stream)?)
}

impl LocalExpertBank for PackedSwiGluExperts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

impl LocalExpertBank for PackedRelu2Experts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

impl LocalExpertBank for RoutedExperts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward_local(hidden, &ids, &weights, stream)?)
    }
}

/// Compacts routes owned by this rank with exactly one scalar synchronization.
pub fn compact_local_routes(
    hidden_states: &Array,
    expert_ids: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<(DispatchedRoutes, RoutingStatistics), Error> {
    if expert_ids.ndim() != 2 || weights.shape() != expert_ids.shape() {
        return Err(Error::Parallel(format!(
            "expert ids and weights must have matching [tokens, top_k] shapes, got {:?} and {:?}",
            expert_ids.shape(),
            weights.shape()
        )));
    }
    if hidden_states.ndim() != 2 || hidden_states.dim(0) != expert_ids.dim(0) {
        return Err(Error::Parallel(format!(
            "hidden states must be [tokens, hidden] matching route tokens, got {:?}",
            hidden_states.shape()
        )));
    }
    if !matches!(
        expert_ids.dtype(),
        Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
    ) {
        return Err(Error::Parallel(format!(
            "expert ids must use an integer dtype, got {:?}",
            expert_ids.dtype()
        )));
    }
    if !weights.dtype().is_float() || !hidden_states.dtype().is_float() {
        return Err(Error::Parallel(
            "route weights and hidden states must be floating point".into(),
        ));
    }
    let flat_ids = expert_ids
        .reshape(&[-1], stream)?
        .as_dtype(Dtype::Int32, stream)?;
    let valid = flat_ids.ge(Array::from_int(0), stream)?.logical_and(
        flat_ids.lt(
            Array::from_int(assignment.global_expert_count as i32),
            stream,
        )?,
        stream,
    )?;
    let invalid = valid.logical_not(stream)?.count_nonzero(stream)?;
    // Use a safe placeholder for invalid ids so validation and the compact
    // count can share the same single host synchronization below.
    let safe_ids = r#where(
        &valid,
        flat_ids.clone(),
        Array::zeros::<i32>(&[flat_ids.size() as i32], stream)?,
        stream,
    )?;
    let owners = Array::from_slice(
        &assignment
            .owners
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>(),
        &[assignment.global_expert_count as i32],
    );
    let owner_local = Array::from_slice(
        &assignment
            .owner_local
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>(),
        &[assignment.global_expert_count as i32],
    );
    let route_owners = owners.take(&safe_ids, stream)?;
    let mask = route_owners
        .eq(Array::from_int(assignment.rank as i32), stream)?
        .logical_and(valid, stream)?;
    let compact = mask.compact_indices(stream)?;
    let started = std::time::Instant::now();
    eval([&invalid, &compact.count])?;
    let synchronization_time = started.elapsed();
    if invalid.clone().try_item::<i32>(stream)? != 0 {
        return Err(Error::Parallel(
            "route contains a globally invalid expert id".into(),
        ));
    }
    let local_routes = compact.count.clone().try_item::<i32>(stream)? as usize;
    let positions = compact
        .indices
        .try_index_device(..local_routes as i32, stream)?;
    let global_expert_ids = flat_ids.take(&positions, stream)?;
    let local_expert_ids = owner_local.take(&global_expert_ids, stream)?;
    let top_k = expert_ids.dim(1);
    let token_indices = positions.floor_divide(Array::from_int(top_k), stream)?;
    let slot_indices = positions.remainder(Array::from_int(top_k), stream)?;
    let hidden = hidden_states.take_axis(&token_indices, 0, stream)?;
    let route_weights = weights.reshape(&[-1], stream)?.take(&positions, stream)?;
    Ok((
        DispatchedRoutes {
            hidden,
            global_expert_ids,
            local_expert_ids,
            original_route_indices: positions,
            token_indices,
            slot_indices,
            weights: route_weights,
        },
        RoutingStatistics {
            total_routes: expert_ids.size(),
            local_routes,
            synchronization_count: 1,
            synchronization_time,
            ..RoutingStatistics::default()
        },
    ))
}

/// Executes compact local routes and exactly recombines them across EP ranks.
pub fn dispatch_replicated(
    hidden_states: &Array,
    expert_ids: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    bank: &mut impl LocalExpertBank,
    group: &Group,
    stream: &Stream,
) -> Result<ReturnedRoutes, Error> {
    dispatch_replicated_with(
        hidden_states,
        expert_ids,
        weights,
        assignment,
        group,
        stream,
        |routes, stream| {
            bank.execute_local_routes(&routes.hidden, &routes.local_expert_ids, stream)
        },
    )
}

/// Dispatches replicated routes while delegating exact local route execution.
///
/// The callback receives both global and owner-local ids after the existing
/// validated route compaction, so cache-backed banks can retain global identity
/// without duplicating transport or recombination.
pub fn dispatch_replicated_with<F>(
    hidden_states: &Array,
    expert_ids: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    group: &Group,
    stream: &Stream,
    execute: F,
) -> Result<ReturnedRoutes, Error>
where
    F: FnOnce(&DispatchedRoutes, &Stream) -> Result<Array, Error>,
{
    let total_started = Instant::now();
    if group.rank() != assignment.rank || group.size() != assignment.group_size {
        return Err(Error::Parallel(
            "expert assignment does not match the supplied group".into(),
        ));
    }
    let compaction_started = Instant::now();
    let (routes, mut statistics) =
        compact_local_routes(hidden_states, expert_ids, weights, assignment, stream)?;
    materialize_timing_phase([
        &routes.hidden,
        &routes.global_expert_ids,
        &routes.local_expert_ids,
        &routes.original_route_indices,
        &routes.token_indices,
        &routes.slot_indices,
        &routes.weights,
    ])?;
    statistics.compaction_time += compaction_started.elapsed();
    let expert_started = Instant::now();
    let local_output = if statistics.local_routes == 0 {
        zeros_dtype(hidden_states.shape(), hidden_states.dtype(), stream)?
    } else {
        let output = execute(&routes, stream)?;
        if output.ndim() != 2 || output.dim(0) != statistics.local_routes as i32 {
            return Err(Error::Parallel(format!(
                "local expert bank returned invalid shape {:?}",
                output.shape()
            )));
        }
        let weighted = output.multiply(routes.weights.expand_dims(1, stream)?, stream)?;
        segment_sum_by_index(
            weighted,
            &routes.token_indices,
            hidden_states.dim(0),
            stream,
        )?
    };
    materialize_timing_phase([&local_output])?;
    statistics.expert_time += expert_started.elapsed();
    let reduction_started = Instant::now();
    let reduced_output = distributed::all_sum(&local_output, group, stream)?;
    materialize_timing_phase([&reduced_output])?;
    statistics.reduction_time += reduction_started.elapsed();
    statistics.total_time = total_started.elapsed();
    Ok(ReturnedRoutes {
        local_output,
        reduced_output,
        statistics,
    })
}

/// Result of one variable-count all-to-all fallback.
pub struct ExchangeResult {
    /// Received rows concatenated in source-rank order.
    pub received: Array,
    /// Number of logical rows received from every source rank.
    pub source_counts: Vec<usize>,
    /// Transport counters.
    pub statistics: RoutingStatistics,
}

/// Destination-major route blocks for sharded-token expert dispatch.
///
/// Every vector has exactly one block per destination EP rank and matching
/// leading row counts. Global expert ids and original flattened route indices
/// remain visible at this transport boundary.
pub struct ShardedRouteBlocks {
    /// Hidden activation rows addressed to each expert owner.
    pub hidden: Vec<Array>,
    /// Checkpoint-global expert ids for each row.
    pub global_expert_ids: Vec<Array>,
    /// Original source-rank flattened route indices for each row.
    pub original_route_indices: Vec<Array>,
    /// Route weights for each row, applied exactly once by the owner.
    pub weights: Vec<Array>,
    /// Number of top-k slots per source token.
    pub top_k: i32,
    /// Number of tokens owned by this source rank.
    pub source_tokens: i32,
}

/// Returned source-local output from sharded-input dispatch.
pub struct ShardedReturnedRoutes {
    /// Weighted route output reduced to source token order.
    pub output: Array,
    /// Transport and execution counters.
    pub statistics: RoutingStatistics,
}

fn validate_sharded_blocks(blocks: &ShardedRouteBlocks, world: usize) -> Result<(), Error> {
    if blocks.top_k <= 0 || blocks.source_tokens < 0 {
        return Err(Error::Parallel(
            "sharded dispatch requires positive top_k and nonnegative source token count".into(),
        ));
    }
    if blocks.hidden.len() != world
        || blocks.global_expert_ids.len() != world
        || blocks.original_route_indices.len() != world
        || blocks.weights.len() != world
    {
        return Err(Error::Parallel(format!(
            "sharded dispatch requires {world} blocks for every payload and metadata field"
        )));
    }
    for destination in 0..world {
        let rows = blocks.hidden[destination].dim(0);
        if blocks.hidden[destination].ndim() != 2
            || blocks.global_expert_ids[destination].shape() != [rows]
            || blocks.original_route_indices[destination].shape() != [rows]
            || blocks.weights[destination].shape() != [rows]
        {
            return Err(Error::Parallel(format!(
                "destination {destination} sharded route fields have inconsistent row counts"
            )));
        }
    }
    Ok(())
}

/// Exchanges sharded-token routes, executes owner-local experts, and returns
/// exact weighted results to their source ranks.
///
/// All payload and metadata exchange uses [`all_to_all_v`]. Collectives are
/// entered in a fixed order on every rank, including ranks with zero routes.
pub fn dispatch_sharded(
    blocks: ShardedRouteBlocks,
    assignment: &ExpertAssignment,
    bank: &mut impl LocalExpertBank,
    group: &Group,
    stream: &Stream,
) -> Result<ShardedReturnedRoutes, Error> {
    let total_started = Instant::now();
    if group.rank() != assignment.rank() || group.size() != assignment.group_size() {
        return Err(Error::Parallel(
            "expert assignment does not match the supplied group".into(),
        ));
    }
    validate_sharded_blocks(&blocks, group.size())?;
    let total_routes = blocks
        .hidden
        .iter()
        .map(|block| block.dim(0) as usize)
        .sum();
    let hidden = all_to_all_v(&blocks.hidden, group, stream)?;
    let global_ids = all_to_all_v(&blocks.global_expert_ids, group, stream)?;
    let route_indices = all_to_all_v(&blocks.original_route_indices, group, stream)?;
    let weights = all_to_all_v(&blocks.weights, group, stream)?;
    if hidden.source_counts != global_ids.source_counts
        || hidden.source_counts != route_indices.source_counts
        || hidden.source_counts != weights.source_counts
    {
        return Err(Error::Parallel(
            "sharded route payload and metadata receive counts diverged".into(),
        ));
    }
    let received_routes = hidden.received.dim(0);
    let owner_local = Array::from_slice(
        &assignment
            .owner_local_ids()
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>(),
        &[assignment.global_expert_count() as i32],
    );
    let local_ids =
        owner_local.take(&global_ids.received.as_dtype(Dtype::Int32, stream)?, stream)?;
    let expert_started = Instant::now();
    let weighted = if received_routes == 0 {
        let mut shape = hidden.received.shape().to_vec();
        shape[0] = 0;
        zeros_dtype(&shape, hidden.received.dtype(), stream)?
    } else {
        bank.execute_local_routes(&hidden.received, &local_ids, stream)?
            .multiply(weights.received.expand_dims(1, stream)?, stream)?
    };
    materialize_timing_phase([&weighted])?;
    let expert_time = expert_started.elapsed();
    let mut output_to_source = Vec::with_capacity(group.size());
    let mut indices_to_source = Vec::with_capacity(group.size());
    let mut offset = 0i32;
    for count in &hidden.source_counts {
        let end = offset + *count as i32;
        output_to_source.push(weighted.try_index_device(offset..end, stream)?);
        indices_to_source.push(
            route_indices
                .received
                .try_index_device(offset..end, stream)?,
        );
        offset = end;
    }
    let returned_output = all_to_all_v(&output_to_source, group, stream)?;
    let returned_indices = all_to_all_v(&indices_to_source, group, stream)?;
    if returned_output.source_counts != returned_indices.source_counts {
        return Err(Error::Parallel(
            "returned sharded outputs and route indices diverged".into(),
        ));
    }
    let token_indices = returned_indices
        .received
        .as_dtype(Dtype::Int32, stream)?
        .floor_divide(Array::from_int(blocks.top_k), stream)?;
    let reduction_started = Instant::now();
    let output = segment_sum_by_index(
        returned_output.received.clone(),
        token_indices,
        blocks.source_tokens,
        stream,
    )?;
    materialize_timing_phase([&output])?;
    let reduction_time = reduction_started.elapsed();
    let mut statistics = RoutingStatistics {
        total_routes,
        local_routes: received_routes as usize,
        sent_routes: total_routes,
        received_routes: received_routes as usize,
        expert_time,
        reduction_time,
        ..Default::default()
    };
    for exchange in [
        hidden,
        global_ids,
        route_indices,
        weights,
        returned_output,
        returned_indices,
    ] {
        statistics.padding_routes += exchange.statistics.padding_routes;
        statistics.synchronization_count += exchange.statistics.synchronization_count;
        statistics.exchanged_bytes += exchange.statistics.exchanged_bytes;
        statistics.synchronization_time += exchange.statistics.synchronization_time;
        statistics.exchange_time += exchange.statistics.exchange_time;
    }
    statistics.total_time = total_started.elapsed();
    Ok(ShardedReturnedRoutes { output, statistics })
}

/// Exchanges destination-major variable-sized blocks using padded all-gather.
///
/// `send_blocks[d]` contains rows addressed to destination rank `d`; all
/// blocks must have the same trailing shape and dtype.  The fallback gathers
/// `group_size` destination blocks from every source, so peak transfer storage
/// and bandwidth are `O(group_size)` larger than a native all-to-all.
pub fn all_to_all_v(
    send_blocks: &[Array],
    group: &Group,
    stream: &Stream,
) -> Result<ExchangeResult, Error> {
    let total_started = Instant::now();
    let world = group.size();
    if send_blocks.len() != world || send_blocks.is_empty() {
        return Err(Error::Parallel(format!(
            "all_to_all_v requires {world} destination blocks"
        )));
    }
    if send_blocks.iter().any(|block| block.ndim() == 0) {
        return Err(Error::Parallel(
            "all_to_all_v blocks must have a leading row dimension".into(),
        ));
    }
    let dtype = send_blocks[0].dtype();
    let first_shape = send_blocks[0].shape();
    let tail = &first_shape[1..];
    if send_blocks
        .iter()
        .any(|block| block.dtype() != dtype || &block.shape()[1..] != tail)
    {
        return Err(Error::Parallel(
            "all_to_all_v blocks must share dtype and trailing shape".into(),
        ));
    }
    let local_counts = send_blocks
        .iter()
        .map(|block| block.dim(0))
        .collect::<Vec<_>>();
    // Materialize the tiny host count vector onto the explicit execution
    // stream before entering the collective.
    let counts = Array::from_slice(&local_counts, &[world as i32]).copy(stream)?;
    let gathered_counts = distributed::all_gather(&counts, group, stream)?;
    let started = std::time::Instant::now();
    let evaluated_counts = gathered_counts.evaluated()?;
    let synchronization_time = started.elapsed();
    let all_counts = evaluated_counts.as_slice::<i32>();
    let max_rows = all_counts.iter().copied().max().unwrap_or(0) as usize;
    if max_rows == 0 {
        let mut shape = send_blocks[0].shape().to_vec();
        shape[0] = 0;
        let exchange_time = total_started.elapsed();
        return Ok(ExchangeResult {
            received: zeros_dtype(&shape, dtype, stream)?,
            source_counts: vec![0; world],
            statistics: RoutingStatistics {
                synchronization_count: 1,
                synchronization_time,
                exchange_time,
                total_time: exchange_time,
                ..Default::default()
            },
        });
    }
    let mut padded = Vec::with_capacity(world);
    for block in send_blocks {
        let rows = block.dim(0) as usize;
        if rows == max_rows {
            padded.push(block.clone());
        } else {
            let mut shape = block.shape().to_vec();
            shape[0] = (max_rows - rows) as i32;
            let padding = zeros_dtype(&shape, dtype, stream)?;
            padded.push(concatenate_axis(&[block, &padding], 0, stream)?);
        }
    }
    let refs = padded.iter().collect::<Vec<_>>();
    let packed = concatenate_axis(&refs, 0, stream)?;
    let gathered = distributed::all_gather(&packed, group, stream)?;
    let mut received = Vec::with_capacity(world);
    let mut source_counts = Vec::with_capacity(world);
    for source in 0..world {
        let count = all_counts[source * world + group.rank()] as usize;
        source_counts.push(count);
        let start = (source * world * max_rows + group.rank() * max_rows) as i32;
        received.push(gathered.try_index_device(start..start + count as i32, stream)?);
    }
    let refs = received.iter().collect::<Vec<_>>();
    let received = concatenate_axis(&refs, 0, stream)?;
    materialize_timing_phase([&received])?;
    let sent_routes = local_counts
        .iter()
        .map(|value| *value as usize)
        .sum::<usize>();
    let received_routes = source_counts.iter().sum::<usize>();
    let padding_routes = world * world * max_rows
        - all_counts
            .iter()
            .map(|value| *value as usize)
            .sum::<usize>();
    let row_bytes = tail
        .iter()
        .map(|dimension| *dimension as usize)
        .product::<usize>()
        * send_blocks[0].item_size();
    let exchange_time = total_started.elapsed();
    Ok(ExchangeResult {
        received,
        source_counts,
        statistics: RoutingStatistics {
            sent_routes,
            received_routes,
            padding_routes,
            synchronization_count: 1,
            synchronization_time,
            exchange_time,
            total_time: exchange_time,
            exchanged_bytes: world * world * max_rows * row_bytes,
            ..Default::default()
        },
    })
}

/// Immutable description of a rank-local expert-parallel model.
#[derive(Debug, Clone)]
pub struct ExpertParallelInfo {
    /// Global rank.
    pub global_rank: usize,
    /// Rank in the EP group.
    pub expert_parallel_rank: usize,
    /// EP group size.
    pub expert_parallel_size: usize,
    /// Loaded architecture.
    pub model_kind: ModelKind,
    /// Assignment metadata.
    pub assignment: ExpertAssignment,
    /// Bytes in all locally materialized parameters.
    pub local_parameter_bytes: usize,
    /// Bytes in local routed-expert tensors.
    pub routed_expert_bytes: usize,
    /// Bytes in all cold, warm, or hot routed experts owned by this rank.
    pub owned_expert_bytes: usize,
    /// Bytes in replicated tensors.
    pub replicated_parameter_bytes: usize,
    /// Checkpoint shards opened by this rank.
    pub opened_checkpoint_shards: Vec<PathBuf>,
    /// Active route transport.
    pub exchange_strategy: ExpertExchangeStrategy,
}

/// Architecture-checked replicated attention cache used by an EP model.
#[derive(Debug, Clone)]
pub enum ExpertParallelCache {
    /// DeepSeek compressed-latent attention cache.
    DeepSeek(deepseek_v3::Cache),
    /// Qwen3 standard key/value cache.
    Qwen3(Vec<Option<ConcatKeyValueCache>>),
    /// Qwen3 bounded sliding-window key/value cache.
    Qwen3Sliding(Vec<Option<SlidingKeyValueCache>>),
    /// Qwen3 globally budgeted paged key/value cache.
    Qwen3Paged(Vec<Option<PagedKeyValueCache>>),
    /// GPT-OSS alternating full/sliding attention cache.
    GptOss(gpt_oss::Cache),
    /// Inkling attention and convolution cache.
    Inkling(inkling::Cache),
    /// LFM2 heterogeneous attention/convolution cache.
    Lfm2(lfm2::Cache),
    /// Nemotron-H heterogeneous recurrent/attention cache.
    NemotronH(nemotron_h::Cache),
    /// Qwen3-Next/Qwen3.5 heterogeneous attention cache.
    QwenHybrid(qwen3_5_moe::Cache),
    /// Qwen3-VL-MoE multimodal-RoPE text cache.
    Qwen3Vl(qwen3_vl::Cache),
}

impl ExpertParallelCache {
    /// Clears all cached attention state.
    pub fn reset(&mut self) -> Result<(), Error> {
        match self {
            Self::DeepSeek(cache) => {
                for cache in &mut cache.layers {
                    cache.clear()?;
                }
            }
            Self::Qwen3(cache) => cache
                .iter_mut()
                .flatten()
                .for_each(ConcatKeyValueCache::clear),
            Self::Qwen3Sliding(cache) => cache
                .iter_mut()
                .flatten()
                .for_each(SlidingKeyValueCache::clear),
            Self::Qwen3Paged(caches) => {
                if let Some(first) = caches.iter().flatten().next() {
                    first
                        .manager()
                        .clear()
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                }
                for cache in caches.iter_mut().flatten() {
                    cache.reset_local_after_manager_clear();
                }
            }
            Self::GptOss(cache) => cache.reset()?,
            Self::Inkling(cache) => cache.reset()?,
            Self::Lfm2(cache) => cache.reset(),
            Self::NemotronH(cache) => cache.reset(),
            Self::QwenHybrid(cache) => cache.reset(),
            Self::Qwen3Vl(cache) => *cache = qwen3_vl::Cache::default(),
        }
        Ok(())
    }

    /// Returns the common replicated cache offset.
    pub fn offset(&self) -> i32 {
        match self {
            Self::DeepSeek(cache) => cache.offset(),
            Self::Qwen3(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Qwen3Sliding(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Qwen3Paged(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::GptOss(cache) => cache.offset(),
            Self::Inkling(cache) => cache.offset(),
            Self::Lfm2(cache) => cache.offset(),
            Self::NemotronH(cache) => cache.offset(),
            Self::QwenHybrid(cache) => cache.offset(),
            Self::Qwen3Vl(cache) => cache
                .kv
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
        }
    }
}

enum ExpertArchitecture {
    DeepSeek(Box<deepseek_v3::Model>),
    Qwen3(Box<qwen3::Model>),
    GptOss(Box<gpt_oss::Model>),
    Inkling(Box<inkling::Model>),
    Lfm2(Box<lfm2::Model>),
    NemotronH(Box<nemotron_h::Model>),
    QwenHybrid(Box<qwen3_5_moe::Model>),
    Qwen3Vl(Box<qwen3_vl::Model>),
}

/// Executable rank-local pure expert-parallel model.
pub struct ExpertParallelModel {
    topology: ParallelTopology,
    info: ExpertParallelInfo,
    architecture: ExpertArchitecture,
    expert_cache: Option<ExpertCache>,
    latest_statistics: RoutingStatistics,
    cumulative_statistics: RoutingStatistics,
}

struct ExpertParallelQwenMtpTarget<'a> {
    model: &'a mut ExpertParallelModel,
    group: &'a Group,
}

struct ExpertParallelSpeculativeSampler<'a, S> {
    sampler: &'a mut S,
    sampling_rank: usize,
    group: &'a Group,
}

impl<S: SpeculativeSampler> SpeculativeSampler for ExpertParallelSpeculativeSampler<'_, S> {
    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.sampler
            .process_logits(logits, temperature, history, stream)
    }

    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // Sample on every rank so identical PRNG states remain aligned, then
        // retain only the designated rank's choice before synchronizing it.
        let sampled = self
            .sampler
            .sample_processed(logits, temperature, prng_state, stream)?;
        let selected = if self.group.rank() == self.sampling_rank {
            sampled
        } else {
            zeros_dtype(sampled.shape(), sampled.dtype(), stream)?
        };
        distributed::all_sum(&selected, self.group, stream)
    }

    fn commit_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.sampler.commit_token(processed_logits, token, stream)
    }
}

impl std::fmt::Debug for ExpertParallelModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpertParallelModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl crate::qwen_mtp::QwenMtpTarget for ExpertParallelQwenMtpTarget<'_> {
    fn prefill_mtp_target(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut qwen3_5_moe::Cache,
        stream: &Stream,
    ) -> Result<qwen3_5_moe::QwenMtpStepOutput, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        cache.reset();
        self.model
            .forward_qwen_mtp_target(&tokens, cache, self.group, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut qwen3_5_moe::Cache,
        stream: &Stream,
    ) -> Result<qwen3_5_moe::QwenMtpStepOutput, Exception> {
        self.model
            .forward_qwen_mtp_target(tokens, cache, self.group, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [qwen3_5_moe::LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match &mut self.model.architecture {
            ExpertArchitecture::QwenHybrid(model) => {
                model.forward_mtp_head(hidden, tokens, cache, stream)
            }
            _ => Err(Exception::custom(
                "embedded Qwen MTP requires a Qwen3-Next or Qwen3.5 EP model",
            )),
        }
    }

    fn mtp_layer_count(&self) -> usize {
        match &self.model.architecture {
            ExpertArchitecture::QwenHybrid(model) => model.mtp_len(),
            _ => 0,
        }
    }
}

impl ExpertParallelModel {
    /// Returns placement, assignment, and memory diagnostics.
    pub fn info(&self) -> &ExpertParallelInfo {
        &self.info
    }

    /// Reports whether this EP target can perform embedded MTP generation.
    pub fn mtp_capability(&self) -> MtpCapability {
        match &self.architecture {
            ExpertArchitecture::QwenHybrid(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            _ => MtpCapability::Unavailable,
        }
    }

    /// Returns dynamic rank-owned expert residency when sparse caching is active.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Allocates an empty architecture-appropriate replicated cache.
    pub fn new_cache(&self) -> ExpertParallelCache {
        match &self.architecture {
            ExpertArchitecture::DeepSeek(model) => ExpertParallelCache::DeepSeek(model.new_cache()),
            ExpertArchitecture::Qwen3(_) => ExpertParallelCache::Qwen3(Vec::new()),
            ExpertArchitecture::GptOss(model) => ExpertParallelCache::GptOss(model.new_cache()),
            ExpertArchitecture::Inkling(model) => ExpertParallelCache::Inkling(model.new_cache()),
            ExpertArchitecture::Lfm2(model) => ExpertParallelCache::Lfm2(model.new_cache()),
            ExpertArchitecture::NemotronH(model) => {
                ExpertParallelCache::NemotronH(model.new_cache())
            }
            ExpertArchitecture::QwenHybrid(model) => {
                ExpertParallelCache::QwenHybrid(model.new_cache())
            }
            ExpertArchitecture::Qwen3Vl(model) => ExpertParallelCache::Qwen3Vl(model.new_cache()),
        }
    }

    /// Allocates replicated attention state under an explicit cache policy.
    ///
    /// DeepSeek compressed attention, GPT-OSS alternating attention, Qwen3 KV,
    /// and Inkling relative-position attention are supported. Inkling's
    /// convolution state remains resident; recurrent and multimodal state is
    /// rejected because it is not represented by paged KV blocks.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ExpertParallelCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => match &self.architecture {
                ExpertArchitecture::DeepSeek(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: None,
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    };
                    model
                        .new_cache_with_manager(manager, Some(rank))
                        .map(ExpertParallelCache::DeepSeek)
                        .map_err(Into::into)
                }
                ExpertArchitecture::GptOss(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: None,
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    };
                    model
                        .new_cache_with_manager(manager, Some(rank))
                        .map(ExpertParallelCache::GptOss)
                        .map_err(Into::into)
                }
                ExpertArchitecture::Qwen3(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = Some(CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: None,
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    });
                    let caches = (0..model.model.layers.len())
                        .map(|layer| {
                            PagedKeyValueCache::new_with_layout(
                                manager.clone(),
                                layer,
                                None,
                                0,
                                rank,
                            )
                            .map(Some)
                            .map_err(Error::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(ExpertParallelCache::Qwen3Paged(caches))
                }
                ExpertArchitecture::Inkling(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = Some(CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: None,
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    });
                    model
                        .new_paged_cache_with_manager(manager, rank)
                        .map(ExpertParallelCache::Inkling)
                        .map_err(Into::into)
                }
                _ => Err(Error::Parallel(
                    "paged cache residency is unsupported for this expert-parallel cache representation"
                        .into(),
                )),
            },
        }
    }

    /// Returns aggregate cache-residency telemetry for replicated paged attention state.
    pub fn cache_residency_report(
        &self,
        cache: &ExpertParallelCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        match cache {
            ExpertParallelCache::DeepSeek(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::GptOss(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::Qwen3Paged(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose()
                .map_err(Into::into),
            ExpertParallelCache::Inkling(cache) => cache.residency_report().map_err(Into::into),
            _ => Ok(None),
        }
    }

    /// Persists this rank's replicated paged attention prefix below a shared root.
    pub fn save_prompt_cache(
        &self,
        cache: &mut ExpertParallelCache,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let directory = self.prompt_cache_rank_directory(root.as_ref());
        match cache {
            ExpertParallelCache::DeepSeek(cache) => cache
                .save_prompt_cache(directory, descriptor, prefix_token_ids, options)
                .map_err(Into::into),
            ExpertParallelCache::GptOss(cache) => cache
                .save_prompt_cache(directory, descriptor, prefix_token_ids, options)
                .map_err(Into::into),
            _ => Err(Error::Parallel(
                "expert-parallel prompt persistence requires a supported paged attention cache"
                    .into(),
            )),
        }
    }

    /// Opens this rank's compatible replicated prefix without eager array loading.
    pub fn load_prompt_cache(
        &self,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(ExpertParallelCache, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (manager, manifest) = open_prompt_cache(
            self.prompt_cache_rank_directory(root.as_ref()),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let rank = Some(CacheRankIdentity {
            pipeline_rank: None,
            tensor_parallel_rank: None,
            expert_parallel_rank: Some(self.topology.expert_parallel_rank),
        });
        let cache =
            match &self.architecture {
                ExpertArchitecture::DeepSeek(model) => model
                    .new_cache_with_manager(manager, rank)
                    .map(ExpertParallelCache::DeepSeek)?,
                ExpertArchitecture::GptOss(model) => model
                    .new_cache_with_manager(manager, rank)
                    .map(ExpertParallelCache::GptOss)?,
                _ => return Err(Error::Parallel(
                    "expert-parallel prompt loading is unsupported for this cache representation"
                        .into(),
                )),
            };
        Ok((cache, manifest))
    }

    fn prompt_cache_rank_directory(&self, root: &Path) -> PathBuf {
        root.join(format!("rank-{:05}", self.topology.global_rank))
    }

    /// Returns the canonical cache-relevant architecture identity for this rank.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let (
            model_family,
            effective_model_type,
            architecture_fingerprint,
            layer_count,
            sliding_window,
        ) =
            match &self.architecture {
                ExpertArchitecture::DeepSeek(model) => (
                    "deepseek_v3".to_string(),
                    model.args.model_type.clone(),
                    crate::models::deepseek_v3::prompt_cache_architecture_fingerprint(&model.args),
                    usize::try_from(model.args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid DeepSeek layer count".into()))?,
                    None,
                ),
                ExpertArchitecture::GptOss(model) => (
                    "gpt_oss".to_string(),
                    model.args.model_type.clone(),
                    crate::models::gpt_oss::prompt_cache_architecture_fingerprint(&model.args),
                    usize::try_from(model.args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid GPT-OSS layer count".into()))?,
                    Some(model.args.sliding_window),
                ),
                _ => return Err(Error::Parallel(
                    "prompt-cache persistence is unsupported for this expert-parallel architecture"
                        .into(),
                )),
            };
        Ok(PromptCacheModelIdentity {
            model_family,
            effective_model_type,
            architecture_fingerprint,
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sliding_window,
            sink_tokens: 0,
            topology: PromptCacheTopology {
                pipeline: None,
                tensor_parallel: None,
                expert_parallel: Some((
                    self.topology.expert_parallel_size,
                    self.topology.expert_parallel_rank,
                )),
                expert_parallel_cache_replicated: true,
            },
            layer_layouts: match &self.architecture {
                ExpertArchitecture::DeepSeek(model) => {
                    PromptCacheModelIdentity::compressed_layouts(
                        layer_count,
                        model.args.kv_lora_rank,
                        model.args.qk_rope_head_dim,
                    )
                }
                ExpertArchitecture::GptOss(model) => PromptCacheModelIdentity::key_value_layouts(
                    layer_count,
                    model.args.num_key_value_heads,
                    model.args.head_dim,
                ),
                _ => unreachable!("identity rejects unsupported expert architectures"),
            },
        })
    }

    /// Allocates a bounded Qwen3 sliding-window cache.
    pub fn new_qwen3_sliding_cache(
        &self,
        max_size: i32,
        options: PagedCacheOptions,
    ) -> Result<ExpertParallelCache, Error> {
        if max_size <= 0 {
            return Err(Error::Parallel(
                "Qwen3 sliding cache size must be positive".into(),
            ));
        }
        match &self.architecture {
            ExpertArchitecture::Qwen3(model) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                let rank = Some(CacheRankIdentity {
                    pipeline_rank: None,
                    tensor_parallel_rank: None,
                    expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                });
                Ok(ExpertParallelCache::Qwen3Paged(
                    (0..model.model.layers.len())
                        .map(|layer| {
                            PagedKeyValueCache::new_with_layout(
                                manager.clone(),
                                layer,
                                Some(max_size),
                                0,
                                rank,
                            )
                            .map(Some)
                            .map_err(Error::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ))
            }
            _ => Err(Error::Parallel(
                "sliding key/value caches are only available for Qwen3 expert parallelism".into(),
            )),
        }
    }

    /// Counters from the most recent complete model forward.
    pub fn latest_routing_statistics(&self) -> &RoutingStatistics {
        &self.latest_statistics
    }

    /// Counters accumulated across all forwards.
    pub fn cumulative_routing_statistics(&self) -> &RoutingStatistics {
        &self.cumulative_statistics
    }

    /// Runs prefill or decode with identical input tokens on every EP rank.
    pub fn forward(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_impl(tokens, mask, cache, group, None, stream)
    }

    /// Runs expert-parallel inference while exposing global router decisions,
    /// rank-local expert contributions, reduced routed outputs, and shared experts.
    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        group: &Group,
        observer: &mut impl ActivationObserver,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_impl(tokens, mask, cache, group, Some(observer), stream)
    }

    fn forward_impl(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        group: &Group,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let total_started = Instant::now();
        self.validate_group(group)?;
        self.topology.validate_execution_stream(stream)?;
        if tokens.ndim() != 2 {
            return Err(Error::Parallel(format!(
                "expert-parallel token input must be [batch, sequence], got {:?}",
                tokens.shape()
            )));
        }
        let mut statistics = RoutingStatistics::default();
        let logits = if let Some(expert_cache) = self.expert_cache.as_ref() {
            if observer.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "detailed activation observation is unavailable for sparse expert-cached expert parallelism"
                        .into(),
                ));
            }
            let pass = if tokens.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            let assignment = &self.info.assignment;
            match (&mut self.architecture, cache) {
                (ExpertArchitecture::DeepSeek(model), ExpertParallelCache::DeepSeek(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_deepseek(
                                        &args,
                                        layer,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::Qwen3(model), ExpertParallelCache::Qwen3(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        qwen3::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::Qwen3(model), ExpertParallelCache::Qwen3Sliding(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        qwen3::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::Qwen3(model), ExpertParallelCache::Qwen3Paged(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        qwen3::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::GptOss(model), ExpertParallelCache::GptOss(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_gpt_oss(
                                        &args,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::Inkling(model), ExpertParallelCache::Inkling(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_inkling(
                                        &args,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::Lfm2(model), ExpertParallelCache::Lfm2(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_lfm2(
                                        &args,
                                        layer,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::NemotronH(model), ExpertParallelCache::NemotronH(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_nemotron_h(
                                        &args,
                                        layer,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::QwenHybrid(model), ExpertParallelCache::QwenHybrid(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_qwen_hybrid(
                                        &args,
                                        layer,
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::Qwen3Vl(model), ExpertParallelCache::Qwen3Vl(cache)) => {
                    let args = model.args.text_config.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    let acquired = expert_cache.acquire_routes(
                                        layer,
                                        &routes.global_expert_ids,
                                        pass,
                                        stream,
                                    )?;
                                    execute_cached_qwen3_at(
                                        &args,
                                        layer,
                                        "model.language_model.layers",
                                        &routes.hidden,
                                        &acquired,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                _ => {
                    return Err(Error::Parallel(
                        "expert-parallel cache architecture mismatch".into(),
                    ))
                }
            }
        } else {
            match (&mut self.architecture, cache) {
                (ExpertArchitecture::DeepSeek(model), ExpertParallelCache::DeepSeek(cache)) => {
                    model.forward_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        &self.info.assignment,
                        group,
                        &mut statistics,
                        observer,
                        stream,
                    )?
                }
                (ExpertArchitecture::Qwen3(model), ExpertParallelCache::Qwen3(cache)) => model
                    .forward_expert_parallel(
                        qwen3::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        &self.info.assignment,
                        group,
                        &mut statistics,
                        observer,
                        stream,
                    )?,
                (ExpertArchitecture::Qwen3(model), ExpertParallelCache::Qwen3Sliding(cache)) => {
                    model.forward_expert_parallel(
                        qwen3::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        &self.info.assignment,
                        group,
                        &mut statistics,
                        observer,
                        stream,
                    )?
                }
                (ExpertArchitecture::Qwen3(model), ExpertParallelCache::Qwen3Paged(cache)) => model
                    .forward_expert_parallel(
                        qwen3::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        &self.info.assignment,
                        group,
                        &mut statistics,
                        observer,
                        stream,
                    )?,
                _ => {
                    return Err(Error::Parallel(
                        "expert-parallel cache architecture mismatch".into(),
                    ))
                }
            }
        };
        materialize_timing_phase([&logits])?;
        statistics.model_time = total_started.elapsed();
        self.latest_statistics = statistics;
        self.cumulative_statistics
            .accumulate(&self.latest_statistics);
        Ok(logits)
    }

    fn forward_qwen_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut qwen3_5_moe::Cache,
        group: &Group,
        stream: &Stream,
    ) -> Result<qwen3_5_moe::QwenMtpStepOutput, Error> {
        let total_started = Instant::now();
        self.validate_group(group)?;
        self.topology.validate_execution_stream(stream)?;
        if tokens.ndim() != 2 {
            return Err(Error::Parallel(format!(
                "expert-parallel token input must be [batch, sequence], got {:?}",
                tokens.shape()
            )));
        }
        let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "Qwen embedded MTP currently requires sparse expert-cached EP loading".into(),
            )
        })?;
        let pass = if tokens.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let assignment = &self.info.assignment;
        let mut statistics = RoutingStatistics::default();
        let output = match &mut self.architecture {
            ExpertArchitecture::QwenHybrid(model) => {
                let args = model.args.clone();
                model.forward_cached_expert_parallel_mtp(
                    tokens,
                    cache,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            group,
                            stream,
                            |routes, stream| {
                                let acquired = expert_cache.acquire_routes(
                                    layer,
                                    &routes.global_expert_ids,
                                    pass,
                                    stream,
                                )?;
                                execute_cached_qwen_hybrid(
                                    &args,
                                    layer,
                                    &routes.hidden,
                                    &acquired,
                                    expert_cache,
                                    stream,
                                )
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))?;
                        statistics.accumulate(&returned.statistics);
                        Ok(returned.reduced_output)
                    },
                    stream,
                )?
            }
            _ => {
                return Err(Error::UnsupportedArchitecture(
                    "embedded Qwen MTP requires a Qwen3-Next or Qwen3.5 EP model".into(),
                ))
            }
        };
        materialize_timing_phase([&output.logits])?;
        statistics.model_time = total_started.elapsed();
        self.latest_statistics = statistics;
        self.cumulative_statistics
            .accumulate(&self.latest_statistics);
        Ok(output)
    }

    /// Prompt forward alias.
    pub fn prefill(
        &mut self,
        tokens: &Array,
        cache: &mut ExpertParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, None, cache, group, stream)
    }

    /// Autoregressive decode alias.
    pub fn decode(
        &mut self,
        tokens: &Array,
        cache: &mut ExpertParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, None, cache, group, stream)
    }

    /// Generates with replicated Qwen MTP weights and EP target verification.
    ///
    /// Every rank must call this method with identical inputs and PRNG keys.
    /// Sampling decisions are selected on `sampling_rank` and synchronized
    /// across the EP group.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input(
        &mut self,
        cache: &mut ExpertParallelCache,
        input: runtime_input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler(
            cache,
            input,
            config,
            prng_key,
            &mut DefaultSampler,
            sampling_rank,
            group,
            stream,
        )
    }

    /// Generates through embedded Qwen MTP with a caller-provided sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler<S: SpeculativeSampler>(
        &mut self,
        cache: &mut ExpertParallelCache,
        input: runtime_input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler_callback(
            cache,
            input,
            config,
            prng_key,
            sampler,
            sampling_rank,
            group,
            stream,
            |_| Ok(()),
        )
    }

    /// Generates through embedded Qwen MTP and reports committed tokens on the
    /// designated sampling rank.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        cache: &mut ExpertParallelCache,
        input: runtime_input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
        mut on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        self.validate_group(group)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if sampling_rank >= group.size() {
            return Err(Exception::custom(format!(
                "sampling rank {sampling_rank} is outside EP size {}",
                group.size()
            )));
        }
        if !matches!(
            self.mtp_capability(),
            MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded
            }
        ) {
            return Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for EP model type {} ({:?})",
                self.info.model_kind.model_type_name(),
                self.mtp_capability()
            )));
        }
        let ExpertParallelCache::QwenHybrid(cache) = cache else {
            return Err(Exception::custom(
                "embedded Qwen MTP requires a Qwen hybrid EP cache",
            ));
        };
        let mut synchronized_sampler = ExpertParallelSpeculativeSampler {
            sampler,
            sampling_rank,
            group,
        };
        let emit_callbacks = group.rank() == sampling_rank;
        let mut target = ExpertParallelQwenMtpTarget { model: self, group };
        crate::qwen_mtp::generate_with_callback(
            &mut target,
            cache,
            input,
            config,
            prng_key,
            &mut synchronized_sampler,
            stream,
            |token| {
                if emit_callbacks {
                    on_token(token)
                } else {
                    Ok(())
                }
            },
        )
    }

    /// Samples on one rank and synchronizes only token ids and stop state.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        logits: &Array,
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
                "sampling rank {sampling_rank} is outside EP size {}",
                group.size()
            )));
        }
        let batch = logits.dim(0);
        let local_token = if group.rank() == sampling_rank {
            let last = if logits.ndim() == 3 {
                logits.try_index_device((.., -1, ..), stream)?
            } else {
                logits.clone()
            };
            sampler
                .sample(&last, temperature, prng_state, stream)?
                .reshape(&[batch, 1], stream)?
        } else {
            Array::zeros::<u32>(&[batch, 1], stream)?
        };
        let token = distributed::all_sum(&local_token, group, stream)?;
        let local_finished = if group.rank() == sampling_rank && finished {
            Array::ones::<i32>(&[], stream)?
        } else {
            Array::zeros::<i32>(&[], stream)?
        };
        let finished = distributed::all_sum(&local_finished, group, stream)?;
        eval([&token, &finished])?;
        Ok(SynchronizedToken {
            token,
            finished: finished.try_item::<i32>(stream)? != 0,
        })
    }

    fn validate_group(&self, group: &Group) -> Result<(), Error> {
        if group.rank() != self.topology.global_rank || group.size() != self.topology.world_size {
            return Err(Error::Parallel(format!(
                "expert-parallel topology expects group rank {}/{} but received {}/{}",
                self.topology.global_rank,
                self.topology.world_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }
}

fn validate_pure_expert(topology: ParallelTopology) -> Result<(), Error> {
    if topology.expert_parallel_size <= 1 {
        return Err(Error::Parallel(
            "expert-parallel loading requires expert_parallel_size > 1".into(),
        ));
    }
    if topology.tensor_parallel_size != 1 || topology.pipeline_parallel_size != 1 {
        return Err(Error::Parallel(format!(
            "pure expert-parallel execution requires TP=1 and PP=1, got TP={} PP={} EP={}; hybrid TP+EP and PP+EP are unsupported",
            topology.tensor_parallel_size, topology.pipeline_parallel_size, topology.expert_parallel_size
        )));
    }
    if topology.world_size != topology.expert_parallel_size {
        return Err(Error::Parallel(
            "pure expert-parallel world size must equal expert-parallel size".into(),
        ));
    }
    Ok(())
}

/// Loads an executable pure expert-parallel safetensors MoE model.
pub fn load_expert_parallel_model(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        stream,
        weights_stream,
    )
}

/// Loads an executable pure-EP model with a caller-supplied expert assignment.
///
/// The assignment must describe the checkpoint's complete routed-expert set
/// and match this process's EP rank and group size.
pub fn load_expert_parallel_model_with_assignment(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    assignment: ExpertAssignment,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_with_options_and_assignment(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        assignment,
        stream,
        weights_stream,
    )
}

/// Loads an executable pure-EP model with explicit load options.
pub fn load_expert_parallel_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_impl(model_dir, options, None, stream, weights_stream)
}

/// Loads an executable pure-EP model with explicit model options and expert
/// assignment.
pub fn load_expert_parallel_model_with_options_and_assignment(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    assignment: ExpertAssignment,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_impl(model_dir, options, Some(assignment), stream, weights_stream)
}

fn load_expert_parallel_model_impl(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let model_dir = model_dir.as_ref();
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("expert-parallel loading requires ModelLoadOptions::parallel".into())
    })?;
    validate_pure_expert(topology)?;
    topology.validate_execution_stream(stream)?;
    if model_dir
        .extension()
        .is_some_and(|extension| extension == "gguf")
    {
        return Err(Error::Parallel("expert-parallel GGUF loading is unsupported because bounded local-expert selection is unavailable; use safetensors".into()));
    }
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    match config.get("model_type").and_then(serde_json::Value::as_str) {
        Some("deepseek_v3") => load_deepseek_ep(
            model_dir,
            topology,
            options,
            assignment,
            stream,
            weights_stream,
        ),
        Some("qwen3" | "qwen3_moe") => {
            load_qwen3_ep(
                model_dir,
                topology,
                options,
                assignment,
                stream,
                weights_stream,
            )
        }
        Some("gpt_oss") => load_additional_cached_ep(
            model_dir, topology, options, assignment, ModelKind::GptOss, stream, weights_stream,
        ),
        Some("inkling_mm_model") => load_additional_cached_ep(
            model_dir, topology, options, assignment, ModelKind::Inkling, stream, weights_stream,
        ),
        Some("lfm2" | "lfm2_moe") => load_additional_cached_ep(
            model_dir, topology, options, assignment, ModelKind::Lfm2, stream, weights_stream,
        ),
        Some("nemotron_h") => load_additional_cached_ep(
            model_dir, topology, options, assignment, ModelKind::NemotronH, stream, weights_stream,
        ),
        Some("qwen3_next") => load_additional_cached_ep(
            model_dir, topology, options, assignment, ModelKind::Qwen3Next, stream, weights_stream,
        ),
        Some("qwen3_vl_moe" | "qwen3_vl_moe_text") => load_additional_cached_ep(
            model_dir, topology, options, assignment, ModelKind::Qwen3VlMoe, stream, weights_stream,
        ),
        Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => load_additional_cached_ep(
            model_dir, topology, options, assignment, ModelKind::Qwen35Moe, stream, weights_stream,
        ),
        Some(model_type) => Err(Error::UnsupportedArchitecture(format!(
            "expert-parallel execution requires a supported safetensors MoE architecture, not {model_type}"
        ))),
        None => Err(Error::UnsupportedArchitecture("expert-parallel model config is missing model_type".into())),
    }
}

fn resolve_model_assignment(
    assignment: Option<ExpertAssignment>,
    global_experts: usize,
    topology: ParallelTopology,
) -> Result<ExpertAssignment, Error> {
    let assignment = assignment.map_or_else(
        || {
            ExpertAssignment::balanced(
                global_experts,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )
        },
        Ok,
    )?;
    if assignment.global_expert_count() != global_experts
        || assignment.group_size() != topology.expert_parallel_size
        || assignment.rank() != topology.expert_parallel_rank
    {
        return Err(Error::Parallel(format!(
            "expert assignment describes {} experts at rank {}/{}, but the model and topology require {global_experts} experts at rank {}/{}",
            assignment.global_expert_count(),
            assignment.rank(),
            assignment.group_size(),
            topology.expert_parallel_rank,
            topology.expert_parallel_size,
        )));
    }
    if assignment.local_expert_count() == 0 {
        return Err(Error::Parallel(format!(
            "expert-parallel model loading does not support an empty local expert bank on rank {}",
            assignment.rank()
        )));
    }
    Ok(assignment)
}

fn slice_axis_zero(
    value: &Array,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<Array, Error> {
    let ids = assignment.local_global_expert_ids();
    let contiguous = ids.windows(2).all(|pair| pair[1] == pair[0] + 1);
    if contiguous {
        Ok(value.try_index_device(ids[0] as i32..(ids[ids.len() - 1] + 1) as i32, stream)?)
    } else {
        let ids = Array::from_slice(
            &ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            &[ids.len() as i32],
        );
        Ok(value.take_axis(&ids, 0, stream)?)
    }
}

fn slice_optional(
    param: &mut Param<Option<Array>>,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    if let Some(value) = param.as_ref() {
        let local = slice_axis_zero(value, assignment, stream)?;
        let bytes = local.nbytes();
        *param = Param::new(Some(local));
        Ok(bytes)
    } else {
        Ok(0)
    }
}

fn slice_required(
    param: &mut Param<Array>,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    let local = slice_axis_zero(param.as_ref(), assignment, stream)?;
    let bytes = local.nbytes();
    *param = Param::new(local);
    Ok(bytes)
}

fn parameter_bytes(module: &impl ModuleParameters) -> usize {
    module
        .parameters()
        .flatten()
        .into_values()
        .map(|value| value.nbytes())
        .sum()
}

fn parameter_bytes_excluding(module: &impl ModuleParameters, marker: &str) -> usize {
    module
        .parameters()
        .flatten()
        .into_iter()
        .filter(|(name, _)| !name.contains(marker))
        .map(|(_, value)| value.nbytes())
        .sum()
}

fn qwen_hybrid_replicated_parameter_bytes(module: &impl ModuleParameters) -> usize {
    module
        .parameters()
        .flatten()
        .into_iter()
        .filter(|(name, _)| !is_qwen_hybrid_decoder_expert_key(name))
        .map(|(_, value)| value.nbytes())
        .sum()
}

fn execute_cached_deepseek(
    args: &deepseek_v3::ModelArgs,
    layer: usize,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let started = Instant::now();
    let mut bank = RoutedExperts::new_compact(
        args,
        layer as i32,
        acquired.identities().len() as i32,
        stream,
    )?;
    macro_rules! required {
        ($field:ident, $name:literal) => {
            bank.$field = Param::new(Some(acquired.compact_binding($name, stream)?));
        };
    }
    macro_rules! optional {
        ($field:ident, $name:literal) => {
            bank.$field = Param::new(acquired.optional_compact_binding($name, stream)?);
        };
    }
    required!(gate_proj, "gate_proj");
    optional!(gate_proj_scale_inv, "gate_proj_scale_inv");
    optional!(gate_proj_scales, "gate_proj_scales");
    optional!(gate_proj_biases, "gate_proj_biases");
    required!(up_proj, "up_proj");
    optional!(up_proj_scale_inv, "up_proj_scale_inv");
    optional!(up_proj_scales, "up_proj_scales");
    optional!(up_proj_biases, "up_proj_biases");
    required!(down_proj, "down_proj");
    optional!(down_proj_scale_inv, "down_proj_scale_inv");
    optional!(down_proj_scales, "down_proj_scales");
    optional!(down_proj_biases, "down_proj_biases");
    cache.record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())?;
    let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
    let output = bank.forward_local(hidden, acquired.compact_routes(), &weights, stream)?;
    eval([&output])?;
    acquired.complete_pending()?;
    Ok(output)
}

fn execute_cached_qwen3(
    args: &qwen3::ModelArgs,
    layer: usize,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_qwen3_at(args, layer, "model.layers", hidden, acquired, cache, stream)
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_qwen3_at(
    args: &qwen3::ModelArgs,
    layer: usize,
    layer_root: &str,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let started = Instant::now();
    let prefix = format!("{layer_root}.{layer}.mlp.experts");
    let mut bank = PackedSwiGluExperts::new(
        acquired.identities().len() as i32,
        args.hidden_size,
        args.moe_intermediate_size,
        args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
        args.weight_quantization_for(&format!("{prefix}.down_proj")),
        stream,
    )?;
    bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
    bank.gate_up_proj_scales =
        Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
    bank.gate_up_proj_biases =
        Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
    bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
    bank.down_proj_scales =
        Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
    bank.down_proj_biases =
        Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
    cache.record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())?;
    let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
    let output = bank.forward(hidden, acquired.compact_routes(), &weights, stream)?;
    eval([&output])?;
    acquired.complete_pending()?;
    Ok(output)
}

fn execute_cached_gpt_oss(
    args: &gpt_oss::ModelArgs,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let started = Instant::now();
    let mut compact_args = args.clone();
    compact_args.num_local_experts = acquired.identities().len() as i32;
    let mut bank = gpt_oss::Experts::new(&compact_args, stream)?;
    bank.gate_up_proj_blocks = Param::new(acquired.compact_binding("gate_up_proj_blocks", stream)?);
    bank.gate_up_proj_scales = Param::new(acquired.compact_binding("gate_up_proj_scales", stream)?);
    bank.gate_up_proj_bias = Param::new(acquired.compact_binding("gate_up_proj_bias", stream)?);
    bank.down_proj_blocks = Param::new(acquired.compact_binding("down_proj_blocks", stream)?);
    bank.down_proj_scales = Param::new(acquired.compact_binding("down_proj_scales", stream)?);
    bank.down_proj_bias = Param::new(acquired.compact_binding("down_proj_bias", stream)?);
    cache.record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())?;
    let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
    let output = bank.forward(hidden, acquired.compact_routes(), &weights, stream)?;
    eval([&output])?;
    acquired.complete_pending()?;
    Ok(output)
}

fn execute_cached_inkling(
    args: &inkling::ModelArgs,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let started = Instant::now();
    let text = &args.text_config;
    let mut bank = PackedSwiGluExperts::new(
        acquired.identities().len() as i32,
        text.hidden_size,
        text.moe_intermediate_size(),
        None,
        None,
        stream,
    )?;
    bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
    bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
    cache.record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())?;
    let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
    let output = bank.forward(hidden, acquired.compact_routes(), &weights, stream)?;
    eval([&output])?;
    acquired.complete_pending()?;
    Ok(output)
}

fn execute_cached_lfm2(
    args: &lfm2::ModelArgs,
    layer: usize,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let started = Instant::now();
    let prefix = format!("model.layers.{layer}.feed_forward.experts");
    let mut bank = PackedSwiGluExperts::new(
        acquired.identities().len() as i32,
        args.hidden_size,
        args.moe_intermediate_size,
        args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
        args.weight_quantization_for(&format!("{prefix}.down_proj")),
        stream,
    )?;
    populate_swiglu_bank(&mut bank, acquired, stream)?;
    cache.record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())?;
    let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
    let output = bank.forward(hidden, acquired.compact_routes(), &weights, stream)?;
    eval([&output])?;
    acquired.complete_pending()?;
    Ok(output)
}

fn execute_cached_nemotron_h(
    args: &nemotron_h::ModelArgs,
    layer: usize,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let started = Instant::now();
    let prefix = format!("model.layers.{layer}.moe.experts");
    let mut bank = nemotron_h::Experts::new(
        acquired.identities().len() as i32,
        args.hidden_size,
        args.moe_intermediate_size,
        [
            args.weight_quantization_for(&format!("{prefix}.up_proj")),
            args.weight_quantization_for(&format!("{prefix}.down_proj")),
        ],
        stream,
    )?;
    bank.up_proj = Param::new(acquired.compact_binding("up_proj", stream)?);
    bank.up_proj_scales = Param::new(acquired.optional_compact_binding("up_proj_scales", stream)?);
    bank.up_proj_biases = Param::new(acquired.optional_compact_binding("up_proj_biases", stream)?);
    bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
    bank.down_proj_scales =
        Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
    bank.down_proj_biases =
        Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
    cache.record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())?;
    let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
    let output = bank.forward(hidden, acquired.compact_routes(), &weights, stream)?;
    eval([&output])?;
    acquired.complete_pending()?;
    Ok(output)
}

fn execute_cached_qwen_hybrid(
    args: &qwen3_5_moe::ModelArgs,
    layer: usize,
    hidden: &Array,
    acquired: &AcquiredExperts,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let started = Instant::now();
    let mut compact_args = args.clone();
    compact_args.num_experts = acquired.identities().len() as i32;
    let mut bank = qwen3_5_moe::Experts::new(&compact_args, layer, stream)?;
    bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
    bank.gate_up_proj_scale_inv =
        Param::new(acquired.optional_compact_binding("gate_up_proj_scale_inv", stream)?);
    bank.gate_up_proj_scales =
        Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
    bank.gate_up_proj_biases =
        Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
    bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
    bank.down_proj_scale_inv =
        Param::new(acquired.optional_compact_binding("down_proj_scale_inv", stream)?);
    bank.down_proj_scales =
        Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
    bank.down_proj_biases =
        Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
    cache.record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())?;
    let routes = acquired.compact_routes().reshape(&[-1, 1], stream)?;
    let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
    let output = bank.forward_chunked(hidden, &routes, &weights, stream)?;
    eval([&output])?;
    acquired.complete_pending()?;
    Ok(output)
}

fn populate_swiglu_bank(
    bank: &mut PackedSwiGluExperts,
    acquired: &AcquiredExperts,
    stream: &Stream,
) -> Result<(), Error> {
    bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
    bank.gate_up_proj_scales =
        Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
    bank.gate_up_proj_biases =
        Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
    bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
    bank.down_proj_scales =
        Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
    bank.down_proj_biases =
        Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
    Ok(())
}

fn expert_bank_needs_slicing(
    bank_experts: i32,
    assignment: &ExpertAssignment,
) -> Result<bool, Error> {
    let bank_experts = usize::try_from(bank_experts).map_err(|_| {
        Error::Parallel(format!(
            "expert bank has invalid negative expert count {bank_experts}"
        ))
    })?;
    if bank_experts == assignment.global_expert_count() {
        Ok(true)
    } else if bank_experts == assignment.local_expert_count() {
        Ok(false)
    } else {
        Err(Error::Parallel(format!(
            "expert bank contains {bank_experts} experts, expected either {} global experts or {} experts local to EP rank {}",
            assignment.global_expert_count(),
            assignment.local_expert_count(),
            assignment.rank(),
        )))
    }
}

fn finalize_deepseek_expert_bank(
    bank: &mut RoutedExperts,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    if !expert_bank_needs_slicing(bank.num_experts, assignment)? {
        return Ok(parameter_bytes(bank));
    }
    let mut bytes = 0;
    bytes += slice_optional(&mut bank.gate_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_proj_scale_inv, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_proj_biases, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj_scale_inv, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj_biases, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_scale_inv, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_biases, assignment, stream)?;
    bank.num_experts = assignment.local_expert_count() as i32;
    Ok(bytes)
}

fn finalize_qwen3_expert_bank(
    bank: &mut PackedSwiGluExperts,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    if !expert_bank_needs_slicing(bank.num_experts, assignment)? {
        return Ok(parameter_bytes(bank));
    }
    let mut bytes = 0;
    bytes += slice_required(&mut bank.gate_up_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_up_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_up_proj_biases, assignment, stream)?;
    bytes += slice_required(&mut bank.down_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_biases, assignment, stream)?;
    bank.num_experts = assignment.local_expert_count() as i32;
    Ok(bytes)
}

fn split_expert_id(name: &str) -> Option<usize> {
    let (_, rest) = name.split_once(".mlp.experts.")?;
    rest.split('.').next()?.parse().ok()
}

fn localize_split_expert_name(name: &str, assignment: &ExpertAssignment) -> Option<String> {
    let global = split_expert_id(name)?;
    if assignment.owner(global)? != assignment.rank() {
        return None;
    }
    let local = assignment.owner_local_id(global)?;
    let marker = format!(".mlp.experts.{global}.");
    Some(name.replacen(&marker, &format!(".mlp.experts.{local}."), 1))
}

fn expert_placement_plan(
    store: &(impl WeightStore + ?Sized),
    topology: ParallelTopology,
    assignment: &ExpertAssignment,
) -> Result<(PlacementPlan, bool), Error> {
    let mut plan = PlacementPlan::replicated(topology);
    let mut has_split = false;
    for key in store.keys() {
        if let Some(global) = split_expert_id(&key) {
            has_split = true;
            let placement = if assignment.owner(global) == Some(assignment.rank()) {
                TensorPlacement::Local
            } else {
                TensorPlacement::Omit
            };
            plan.insert(key, placement);
        } else if key.contains(".mlp.experts.")
            && matches!(
                key.rsplit('.').next(),
                Some(
                    "gate_up_proj"
                        | "gate_proj"
                        | "up_proj"
                        | "down_proj"
                        | "gate_proj_scale_inv"
                        | "up_proj_scale_inv"
                        | "down_proj_scale_inv"
                        | "gate_up_proj_scales"
                        | "gate_up_proj_biases"
                        | "gate_proj_scales"
                        | "gate_proj_biases"
                        | "up_proj_scales"
                        | "up_proj_biases"
                        | "down_proj_scales"
                        | "down_proj_biases"
                )
            )
        {
            let ids = assignment.local_global_expert_ids();
            let placement = if ids.windows(2).all(|pair| pair[1] == pair[0] + 1) {
                TensorPlacement::Range {
                    axis: 0,
                    start: ids[0],
                    end: ids[ids.len() - 1] + 1,
                }
            } else {
                TensorPlacement::Indices {
                    axis: 0,
                    indices: ids.to_vec(),
                }
            };
            plan.insert(key, placement);
        }
    }
    Ok((plan, has_split))
}

fn quantize_qwen3_local_experts(
    tensors: &mut std::collections::HashMap<String, Array>,
    num_hidden_layers: i32,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(), Error> {
    for layer in 0..num_hidden_layers {
        for projection in ["gate_up_proj", "down_proj"] {
            let key = format!("model.layers.{layer}.mlp.experts.{projection}");
            let value = tensors
                .remove(&key)
                .ok_or_else(|| Error::StrictLoadValidation {
                    missing: vec![key.clone()],
                    unused: Vec::new(),
                })?;
            let quantized = quantize_expert_bank(&value, quantization, stream)?;
            eval(
                [&quantized.weight, &quantized.scales]
                    .into_iter()
                    .chain(quantized.biases.as_ref()),
            )?;
            tensors.insert(key.clone(), quantized.weight);
            tensors.insert(format!("{key}_scales"), quantized.scales);
            if let Some(biases) = quantized.biases {
                tensors.insert(format!("{key}_biases"), biases);
            }
        }
    }
    Ok(())
}

fn load_deepseek_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if let WeightResidency::SparseExpertCache(expert_options) = options.weight_residency {
        return load_deepseek_cached_ep(
            model_dir,
            topology,
            options,
            expert_options,
            assignment,
            stream,
            weights_stream,
        );
    }
    if matches!(
        options.weight_residency,
        WeightResidency::LayerwiseHost(_)
            | WeightResidency::DenseDiskStream(_)
            | WeightResidency::SparseExpertCacheWithDenseLayers(_)
    ) {
        return Err(Error::Parallel(
            "expert-parallel loading accepts fully resident weights or SparseExpertCache; dense non-expert layer streaming is not supported"
                .into(),
        ));
    }
    let source_args = deepseek_v3::get_model_args(model_dir)?;
    if source_args.n_routed_experts <= 0 {
        return Err(Error::Parallel(
            "DeepSeek config has no routed experts".into(),
        ));
    }
    if options.quantization.is_some() && source_args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 expert-parallel weights cannot be implicitly dequantized and requantized".into(),
        ));
    }
    let quantize_on_load = options
        .quantization
        .map(|requested| {
            should_quantize_on_load(
                "DeepSeek-V3 expert-parallel",
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
    let assignment =
        resolve_model_assignment(assignment, source_args.n_routed_experts as usize, topology)?;
    let store = SafetensorsWeightStore::open(model_dir)?;
    let (plan, _) = expert_placement_plan(&store, topology, &assignment)?;
    let mut strict = StrictLoadConfig::default();
    for index in 0..source_args.num_nextn_predict_layers {
        strict = strict.allow_unused_prefix(format!(
            "model.layers.{}.",
            source_args.num_hidden_layers + index
        ));
    }
    let partition = load_safetensors_partition_from_store_on_streams(
        &store,
        &plan,
        weights_stream,
        stream,
        &strict,
    )?;
    let opened_checkpoint_shards = partition.opened_shards().to_vec();
    let mut tensors = partition.into_tensors();
    let mut model = deepseek_v3::Model::new(target_args, stream)?;
    assign_module(&mut model, "", &mut tensors, quantize_on_load, stream)?;
    for layer_index in 0..source_args.num_hidden_layers as usize {
        let Some(moe) = model.model.layers[layer_index].mlp.moe_mut() else {
            continue;
        };
        let mut localized = Vec::new();
        for name in tensors.keys() {
            if name.starts_with(&format!("model.layers.{layer_index}.mlp.experts.")) {
                if let Some(local) = localize_split_expert_name(name, &assignment) {
                    localized.push((name.clone(), local));
                }
            }
        }
        let localized = localized
            .into_iter()
            .map(|(global, local)| {
                let value = tensors.remove(&global).expect("listed local expert tensor");
                (local, value)
            })
            .collect::<Vec<_>>();
        for (local, value) in localized {
            tensors.insert(local, value);
        }
        load_deepseek_experts(
            moe,
            layer_index,
            (
                assignment.local_expert_count() as i32,
                source_args.hidden_size,
                source_args.moe_intermediate_size,
            ),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
        moe.experts.num_experts = assignment.local_expert_count() as i32;
    }
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    let mut routed_expert_bytes = 0;
    for layer in &mut model.model.layers {
        if let Some(moe) = layer.mlp.moe_mut() {
            routed_expert_bytes +=
                finalize_deepseek_expert_bank(&mut moe.experts, &assignment, stream)?;
        }
    }
    let local_parameter_bytes = parameter_bytes(&model);
    Ok(ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: ModelKind::DeepSeekV3,
            assignment,
            local_parameter_bytes,
            routed_expert_bytes,
            owned_expert_bytes: routed_expert_bytes,
            replicated_parameter_bytes: local_parameter_bytes - routed_expert_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture: ExpertArchitecture::DeepSeek(Box::new(model)),
        expert_cache: None,
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    })
}

fn load_qwen3_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if let WeightResidency::SparseExpertCache(expert_options) = options.weight_residency {
        return load_qwen3_cached_ep(
            model_dir,
            topology,
            options,
            expert_options,
            assignment,
            stream,
            weights_stream,
        );
    }
    if matches!(
        options.weight_residency,
        WeightResidency::LayerwiseHost(_)
            | WeightResidency::DenseDiskStream(_)
            | WeightResidency::SparseExpertCacheWithDenseLayers(_)
    ) {
        return Err(Error::Parallel(
            "expert-parallel loading accepts fully resident weights or SparseExpertCache; dense non-expert layer streaming is not supported"
                .into(),
        ));
    }
    let source_args = qwen3::get_qwen3_model_args(model_dir)?;
    if source_args.num_experts <= 0 {
        return Err(Error::Parallel(
            "Qwen3 config is dense and has no routed experts".into(),
        ));
    }
    let source_quantization = source_args.quantization.or(source_args.quantization_config);
    let quantize_on_load = options
        .quantization
        .map(|requested| {
            should_quantize_on_load("Qwen3 expert-parallel", source_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
    }
    let assignment =
        resolve_model_assignment(assignment, source_args.num_experts as usize, topology)?;
    let store = SafetensorsWeightStore::open(model_dir)?;
    let (plan, has_split) = expert_placement_plan(&store, topology, &assignment)?;
    let partition = load_safetensors_partition_from_store_on_streams(
        &store,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )?;
    let opened_checkpoint_shards = partition.opened_shards().to_vec();
    let mut tensors = partition.into_tensors();
    if has_split {
        let mut localized = std::collections::HashMap::new();
        for (name, value) in tensors {
            if split_expert_id(&name).is_some() {
                if let Some(local) = localize_split_expert_name(&name, &assignment) {
                    localized.insert(local, value);
                }
            } else {
                localized.insert(name, value);
            }
        }
        tensors = transform_split_swiglu_experts(
            localized,
            assignment.local_expert_count() as i32,
            stream,
        )?;
    }
    if let Some(quantization) = quantize_on_load {
        quantize_qwen3_local_experts(
            &mut tensors,
            source_args.num_hidden_layers,
            quantization,
            stream,
        )?;
    }
    let mut model = qwen3::Model::new(target_args.clone(), stream)?;
    for (layer_index, layer) in model.model.layers.iter_mut().enumerate() {
        if let qwen3::FeedForward::Moe(moe) = &mut layer.mlp {
            let prefix = format!("model.layers.{layer_index}.mlp.experts");
            moe.experts = PackedSwiGluExperts::new(
                assignment.local_expert_count() as i32,
                source_args.hidden_size,
                source_args.moe_intermediate_size,
                target_args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                target_args.weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?;
        }
    }
    assign_module(&mut model, "", &mut tensors, quantize_on_load, stream)?;
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    let mut routed_expert_bytes = 0;
    for layer in &mut model.model.layers {
        if let qwen3::FeedForward::Moe(moe) = &mut layer.mlp {
            routed_expert_bytes +=
                finalize_qwen3_expert_bank(&mut moe.experts, &assignment, stream)?;
        }
    }
    let local_parameter_bytes = parameter_bytes(&model);
    Ok(ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: ModelKind::Qwen3,
            assignment,
            local_parameter_bytes,
            routed_expert_bytes,
            owned_expert_bytes: routed_expert_bytes,
            replicated_parameter_bytes: local_parameter_bytes - routed_expert_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture: ExpertArchitecture::Qwen3(Box::new(model)),
        expert_cache: None,
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    })
}

fn expert_cache_base_plan(
    store: &dyn WeightStore,
    topology: ParallelTopology,
    kind: ModelKind,
) -> PlacementPlan {
    let mut plan = PlacementPlan::replicated(topology);
    for key in store.keys() {
        if is_routed_expert_key(kind, &key) || is_auxiliary_checkpoint_key(kind, &key) {
            plan.insert(key, TensorPlacement::Omit);
        }
    }
    plan
}

fn is_routed_expert_key(kind: ModelKind, key: &str) -> bool {
    match kind {
        ModelKind::Lfm2 => key.contains(".feed_forward.experts."),
        ModelKind::NemotronH => key.contains(".experts.") && !key.contains(".shared_experts."),
        ModelKind::Inkling => key.contains(".mlp.experts.") || key.contains(".moe.experts."),
        ModelKind::Qwen3Next | ModelKind::Qwen35Moe => is_qwen_hybrid_decoder_expert_key(key),
        _ => key.contains(".mlp.experts."),
    }
}

fn is_qwen_hybrid_decoder_expert_key(key: &str) -> bool {
    key.starts_with("model.layers.") && key.contains(".mlp.experts.")
}

fn is_auxiliary_checkpoint_key(kind: ModelKind, key: &str) -> bool {
    match kind {
        ModelKind::Inkling => key.starts_with("model.mtp."),
        ModelKind::Qwen3Next => key.starts_with("model.mtp."),
        _ => false,
    }
}

fn rank_owned_expert_cache(
    store: &std::sync::Arc<SafetensorsWeightStore>,
    entries: Vec<ExpertCatalogEntry>,
    assignment: &ExpertAssignment,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(ExpertCache, usize), Error> {
    let entries = entries
        .into_iter()
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
        .collect::<Vec<_>>();
    let owned_expert_bytes =
        usize::try_from(entries.iter().map(ExpertCatalogEntry::bytes).sum::<u64>())
            .map_err(|_| Error::Parallel("owned expert bytes exceed usize".into()))?;
    let cache = ExpertCache::new(
        std::sync::Arc::clone(store),
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?;
    Ok((cache, owned_expert_bytes))
}

fn transform_partition_tensors<F>(
    tensors: std::collections::HashMap<String, Array>,
    mut transform: F,
) -> Result<std::collections::HashMap<String, Array>, Error>
where
    F: FnMut(String, Array) -> Result<Vec<(String, Array)>, Error>,
{
    let mut transformed = std::collections::HashMap::with_capacity(tensors.len());
    for (key, value) in tensors {
        for (key, value) in transform(key, value)? {
            if transformed.insert(key.clone(), value).is_some() {
                return Err(Error::StrictLoadValidation {
                    missing: Vec::new(),
                    unused: vec![format!("duplicate transformed checkpoint key {key}")],
                });
            }
        }
    }
    Ok(transformed)
}

#[allow(clippy::too_many_arguments)]
fn finish_additional_cached_ep(
    topology: ParallelTopology,
    kind: ModelKind,
    assignment: ExpertAssignment,
    architecture: ExpertArchitecture,
    expert_cache: ExpertCache,
    owned_expert_bytes: usize,
    replicated_parameter_bytes: usize,
    opened_checkpoint_shards: Vec<PathBuf>,
) -> ExpertParallelModel {
    ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: kind,
            assignment,
            local_parameter_bytes: replicated_parameter_bytes,
            routed_expert_bytes: 0,
            owned_expert_bytes,
            replicated_parameter_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture,
        expert_cache: Some(expert_cache),
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    }
}

#[allow(clippy::too_many_arguments)]
fn load_additional_cached_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    kind: ModelKind,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if options.quantization.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported with sparse expert-cached expert parallelism; use checkpoint-native weights"
                .into(),
        ));
    }
    let expert_options = match options.weight_residency {
        WeightResidency::SparseExpertCache(options) => options,
        _ => {
            return Err(Error::Parallel(format!(
                "{} expert-parallel execution requires WeightResidency::SparseExpertCache so remote experts are never materialized",
                kind.model_type_name()
            )))
        }
    };
    let store = std::sync::Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(
        model_dir,
        expert_options.non_expert.max_mapped_shards,
    )?);
    let plan = expert_cache_base_plan(store.as_ref(), topology, kind);
    let partition = load_safetensors_partition_from_store_on_streams(
        store.as_ref(),
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )?;
    let opened_checkpoint_shards = partition.opened_shards().to_vec();
    let tensors = partition.into_tensors();

    match kind {
        ModelKind::GptOss => {
            let args = gpt_oss::get_model_args(model_dir)?;
            let assignment =
                resolve_model_assignment(assignment, args.num_local_experts as usize, topology)?;
            let mut model = gpt_oss::Model::new(args.clone(), stream)?;
            let mut tensors = tensors;
            assign_module_excluding(&mut model, "", &mut tensors, None, stream, |name| {
                name.contains(".mlp.experts.")
            })?;
            ensure_no_unused_tensors(tensors)?;
            let entries = crate::gpt_oss::gpt_oss_expert_catalog(&args, store.as_ref())?;
            let (cache, owned) = rank_owned_expert_cache(
                &store,
                entries,
                &assignment,
                expert_options,
                stream,
                weights_stream,
            )?;
            let replicated = parameter_bytes_excluding(&model, ".mlp.experts.");
            Ok(finish_additional_cached_ep(
                topology,
                kind,
                assignment,
                ExpertArchitecture::GptOss(Box::new(model)),
                cache,
                owned,
                replicated,
                opened_checkpoint_shards,
            ))
        }
        ModelKind::Inkling => {
            let args = inkling::get_model_args(model_dir)?;
            let global_experts = usize::try_from(args.text_config.n_routed_experts)
                .map_err(|_| Error::Parallel("Inkling routed expert count is negative".into()))?;
            if global_experts == 0
                || !(0..args.text_config.num_hidden_layers)
                    .any(|layer| !args.text_config.is_dense(layer))
            {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires an Inkling checkpoint with routed MoE layers"
                        .into(),
                ));
            }
            let assignment = resolve_model_assignment(assignment, global_experts, topology)?;
            let mut tensors = transform_partition_tensors(tensors, |key, value| {
                inkling::transform_weight(key, value, stream)
            })?;
            let mut model = inkling::Model::new(args.clone(), stream)?;
            assign_module_excluding(&mut model, "", &mut tensors, None, stream, |name| {
                name.contains(".moe.experts.")
            })?;
            ensure_no_unused_tensors(tensors)?;
            let entries = crate::inkling::inkling_expert_catalog(&args, store.as_ref())?;
            let (cache, owned) = rank_owned_expert_cache(
                &store,
                entries,
                &assignment,
                expert_options,
                stream,
                weights_stream,
            )?;
            let replicated = parameter_bytes_excluding(&model, ".moe.experts.");
            Ok(finish_additional_cached_ep(
                topology,
                kind,
                assignment,
                ExpertArchitecture::Inkling(Box::new(model)),
                cache,
                owned,
                replicated,
                opened_checkpoint_shards,
            ))
        }
        ModelKind::Lfm2 => {
            let args = lfm2::get_model_args(model_dir)?;
            if !args.is_moe() {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires an LFM2 MoE checkpoint".into(),
                ));
            }
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let mut model = lfm2::Model::new(args.clone(), stream)?;
            let mut tensors = tensors;
            assign_module_excluding(&mut model, "", &mut tensors, None, stream, |name| {
                name.contains(".feed_forward.experts.")
            })?;
            ensure_no_unused_tensors(tensors)?;
            let entries = crate::lfm2::lfm2_expert_catalog(&args, store.as_ref())?;
            let (cache, owned) = rank_owned_expert_cache(
                &store,
                entries,
                &assignment,
                expert_options,
                stream,
                weights_stream,
            )?;
            let replicated = parameter_bytes_excluding(&model, ".feed_forward.experts.");
            Ok(finish_additional_cached_ep(
                topology,
                kind,
                assignment,
                ExpertArchitecture::Lfm2(Box::new(model)),
                cache,
                owned,
                replicated,
                opened_checkpoint_shards,
            ))
        }
        ModelKind::NemotronH => {
            let args = nemotron_h::get_nemotron_h_model_args(model_dir)?;
            if !args
                .layer_block_types()?
                .contains(&nemotron_h::LayerBlockType::Moe)
            {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires a Nemotron-H MoE checkpoint".into(),
                ));
            }
            let assignment =
                resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
            let tensors = nemotron_h::transform_nemotron_h_weights(tensors, &args, stream)?;
            let mut tensors = transform_partition_tensors(tensors, |key, value| {
                let key = key
                    .strip_prefix("backbone.")
                    .map_or(key.clone(), |suffix| format!("model.{suffix}"));
                Ok(vec![(key, value)])
            })?;
            let mut model = nemotron_h::Model::new(args.clone(), stream)?;
            assign_module_excluding(&mut model, "", &mut tensors, None, stream, |name| {
                name.contains(".moe.experts.")
            })?;
            ensure_no_unused_tensors(tensors)?;
            let entries = crate::nemotron_h::nemotron_h_expert_catalog(&args, store.as_ref())?;
            let (cache, owned) = rank_owned_expert_cache(
                &store,
                entries,
                &assignment,
                expert_options,
                stream,
                weights_stream,
            )?;
            let replicated = parameter_bytes_excluding(&model, ".moe.experts.");
            Ok(finish_additional_cached_ep(
                topology,
                kind,
                assignment,
                ExpertArchitecture::NemotronH(Box::new(model)),
                cache,
                owned,
                replicated,
                opened_checkpoint_shards,
            ))
        }
        ModelKind::Qwen3Next | ModelKind::Qwen35Moe => {
            let (args, image_token_id, video_token_id, vision_config) =
                if kind == ModelKind::Qwen3Next {
                    (
                        qwen3_next::get_qwen3_next_model_args(model_dir)?,
                        None,
                        None,
                        None,
                    )
                } else {
                    qwen3_5_moe::get_qwen3_5_moe_model_args(model_dir)?
                };
            if !args.is_moe() {
                return Err(Error::UnsupportedArchitecture(format!(
                    "expert parallelism requires a {} MoE checkpoint",
                    kind.model_type_name()
                )));
            }
            if let Some(config) = &args.quantization_config {
                config.validate_supported()?;
            }
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let tensors = if kind == ModelKind::Qwen3Next {
                transform_partition_tensors(tensors, |key, value| {
                    qwen3_next::split_fused_projection(&key, value, &args, stream)
                })?
            } else {
                tensors
            };
            // Decoder experts remain rank-owned and cache-backed, while the
            // embedded MTP head is ordinary replicated state. Public
            // Qwen3-Next checkpoints store both banks as split expert tensors,
            // so pack the retained MTP bank before strict assignment.
            let mut tensors = if args.uses_fp8() {
                qwen3_5_moe::transform_split_qwen_fp8_experts(tensors, args.num_experts, stream)?
            } else {
                transform_split_swiglu_experts(tensors, args.num_experts, stream)?
            };
            let mut model = qwen3_5_moe::Model::new(
                args.clone(),
                image_token_id,
                video_token_id,
                vision_config,
                stream,
            )?;
            assign_module_excluding(&mut model, "", &mut tensors, None, stream, |name| {
                is_qwen_hybrid_decoder_expert_key(name)
            })?;
            ensure_no_unused_tensors(tensors)?;
            let entries = crate::qwen_hybrid::qwen_hybrid_expert_catalog(&args, store.as_ref())?;
            let (cache, owned) = rank_owned_expert_cache(
                &store,
                entries,
                &assignment,
                expert_options,
                stream,
                weights_stream,
            )?;
            let replicated = qwen_hybrid_replicated_parameter_bytes(&model);
            Ok(finish_additional_cached_ep(
                topology,
                kind,
                assignment,
                ExpertArchitecture::QwenHybrid(Box::new(model)),
                cache,
                owned,
                replicated,
                opened_checkpoint_shards,
            ))
        }
        ModelKind::Qwen3VlMoe => {
            let args = qwen3_vl::get_qwen3_vl_model_args(model_dir)?;
            if !args.text_config.is_moe() {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires a Qwen3-VL-MoE checkpoint".into(),
                ));
            }
            let assignment = resolve_model_assignment(
                assignment,
                args.text_config.num_experts as usize,
                topology,
            )?;
            let mut model = qwen3_vl::Model::new(args.clone(), stream)?;
            let mut tensors = tensors;
            assign_module_excluding(&mut model, "", &mut tensors, None, stream, |name| {
                name.contains(".mlp.experts.")
            })?;
            ensure_no_unused_tensors(tensors)?;
            let entries = crate::qwen3::qwen3_expert_catalog_at(
                &args.text_config,
                store.as_ref(),
                "model.language_model.layers",
            )?;
            let (cache, owned) = rank_owned_expert_cache(
                &store,
                entries,
                &assignment,
                expert_options,
                stream,
                weights_stream,
            )?;
            let replicated = parameter_bytes_excluding(&model, ".mlp.experts.");
            Ok(finish_additional_cached_ep(
                topology,
                kind,
                assignment,
                ExpertArchitecture::Qwen3Vl(Box::new(model)),
                cache,
                owned,
                replicated,
                opened_checkpoint_shards,
            ))
        }
        _ => Err(Error::UnsupportedArchitecture(format!(
            "{} is not an additional sparse expert-parallel architecture",
            kind.model_type_name()
        ))),
    }
}

fn ensure_no_unused_tensors(
    tensors: std::collections::HashMap<String, Array>,
) -> Result<(), Error> {
    if tensors.is_empty() {
        return Ok(());
    }
    let mut unused = tensors.into_keys().collect::<Vec<_>>();
    unused.sort();
    Err(Error::StrictLoadValidation {
        missing: Vec::new(),
        unused,
    })
}

#[cfg(test)]
pub(crate) fn assert_rank_owned_sparse_ep_load(
    model_dir: &Path,
    expert_options: ExpertCacheLoadOptions,
    expected_kind: ModelKind,
    expected_owned_experts: usize,
    stream: &Stream,
    weights_stream: &Stream,
) {
    use crate::parallel::DeviceAssignment;
    use safemlx::DeviceType;

    let topology =
        ParallelTopology::from_rank(2, 1, 1, 1, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap();
    let model = load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions {
            quantization: None,
            parallel: Some(topology),
            weight_residency: WeightResidency::SparseExpertCache(expert_options),
        },
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(model.info().model_kind, expected_kind);
    assert_eq!(model.info().expert_parallel_rank, 1);
    assert_eq!(model.info().expert_parallel_size, 2);
    assert_eq!(model.info().routed_expert_bytes, 0);
    assert!(model.info().owned_expert_bytes > 0);
    assert_eq!(
        model.expert_cache_report().unwrap().unwrap().owned_experts,
        expected_owned_experts
    );
}

#[allow(clippy::too_many_arguments)]
fn load_deepseek_cached_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    expert_options: ExpertCacheLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if options.quantization.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported with sparse expert-cached expert parallelism; use checkpoint-native weights"
                .into(),
        ));
    }
    let args = deepseek_v3::get_model_args(model_dir)?;
    args.validate()?;
    let assignment =
        resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
    let store = std::sync::Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(
        model_dir,
        expert_options.non_expert.max_mapped_shards,
    )?);
    let plan = expert_cache_base_plan(store.as_ref(), topology, ModelKind::DeepSeekV3);
    let mut strict = StrictLoadConfig::default();
    for index in 0..args.num_nextn_predict_layers {
        strict =
            strict.allow_unused_prefix(format!("model.layers.{}.", args.num_hidden_layers + index));
    }
    let partition = load_safetensors_partition_from_store_on_streams(
        store.as_ref(),
        &plan,
        weights_stream,
        stream,
        &strict,
    )?;
    let opened_checkpoint_shards = partition.opened_shards().to_vec();
    let mut tensors = partition.into_tensors();
    let mut model = deepseek_v3::Model::new(args.clone(), stream)?;
    assign_module(&mut model, "", &mut tensors, None, stream)?;
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    let entries = crate::deepseek_v3::deepseek_expert_catalog(&args, store.as_ref())?
        .into_iter()
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
        .collect::<Vec<_>>();
    let owned_expert_bytes_u64 = entries.iter().map(ExpertCatalogEntry::bytes).sum::<u64>();
    let owned_expert_bytes = usize::try_from(owned_expert_bytes_u64)
        .map_err(|_| Error::Parallel("owned expert bytes exceed usize".into()))?;
    let expert_cache = ExpertCache::new(
        std::sync::Arc::clone(&store),
        entries,
        expert_options,
        weights_stream.clone(),
        stream.clone(),
    )?;
    let replicated_parameter_bytes = parameter_bytes_excluding(&model, ".mlp.experts.");
    Ok(ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: ModelKind::DeepSeekV3,
            assignment,
            local_parameter_bytes: replicated_parameter_bytes,
            routed_expert_bytes: 0,
            owned_expert_bytes,
            replicated_parameter_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture: ExpertArchitecture::DeepSeek(Box::new(model)),
        expert_cache: Some(expert_cache),
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    })
}

#[allow(clippy::too_many_arguments)]
fn load_qwen3_cached_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    expert_options: ExpertCacheLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if options.quantization.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported with sparse expert-cached expert parallelism; use checkpoint-native weights"
                .into(),
        ));
    }
    let args = qwen3::get_qwen3_model_args(model_dir)?;
    if !args.is_moe() {
        return Err(Error::Parallel(
            "Qwen3 config is dense and has no routed experts".into(),
        ));
    }
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let store = std::sync::Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(
        model_dir,
        expert_options.non_expert.max_mapped_shards,
    )?);
    let plan = expert_cache_base_plan(store.as_ref(), topology, ModelKind::Qwen3);
    let partition = load_safetensors_partition_from_store_on_streams(
        store.as_ref(),
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )?;
    let opened_checkpoint_shards = partition.opened_shards().to_vec();
    let mut tensors = partition.into_tensors();
    let mut model = qwen3::Model::new(args.clone(), stream)?;
    assign_module_excluding(&mut model, "", &mut tensors, None, stream, |name| {
        name.contains(".mlp.experts.")
    })?;
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    let entries = crate::qwen3::qwen3_expert_catalog(&args, store.as_ref())?
        .into_iter()
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
        .collect::<Vec<_>>();
    let owned_expert_bytes_u64 = entries.iter().map(ExpertCatalogEntry::bytes).sum::<u64>();
    let owned_expert_bytes = usize::try_from(owned_expert_bytes_u64)
        .map_err(|_| Error::Parallel("owned expert bytes exceed usize".into()))?;
    let expert_cache = ExpertCache::new(
        std::sync::Arc::clone(&store),
        entries,
        expert_options,
        weights_stream.clone(),
        stream.clone(),
    )?;
    let replicated_parameter_bytes = parameter_bytes_excluding(&model, ".mlp.experts.");
    Ok(ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: ModelKind::Qwen3,
            assignment,
            local_parameter_bytes: replicated_parameter_bytes,
            routed_expert_bytes: 0,
            owned_expert_bytes,
            replicated_parameter_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture: ExpertArchitecture::Qwen3(Box::new(model)),
        expert_cache: Some(expert_cache),
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::DeviceAssignment;
    use safemlx::{
        distributed::Backend, module::ModuleParameters, ops::zeros_dtype, Device, DeviceType,
        ExecutionContext,
    };

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    #[test]
    fn timing_profiling_guard_restores_previous_state() {
        assert!(!timing_profiling_enabled());
        {
            let _outer = profile_expert_parallel_timings();
            assert!(timing_profiling_enabled());
            {
                let _inner = profile_expert_parallel_timings();
                assert!(timing_profiling_enabled());
            }
            assert!(timing_profiling_enabled());
        }
        assert!(!timing_profiling_enabled());
    }

    #[test]
    fn qwen_mtp_weights_are_replicated_while_decoder_experts_are_partitioned() {
        assert!(!is_auxiliary_checkpoint_key(
            ModelKind::Qwen3Next,
            "mtp.fc.weight"
        ));
        assert!(is_auxiliary_checkpoint_key(
            ModelKind::Qwen3Next,
            "model.mtp.fc.weight"
        ));
        assert!(!is_auxiliary_checkpoint_key(
            ModelKind::Qwen35Moe,
            "mtp.fc.weight"
        ));
        assert!(is_auxiliary_checkpoint_key(
            ModelKind::Inkling,
            "model.mtp.fc.weight"
        ));
        assert!(!is_auxiliary_checkpoint_key(
            ModelKind::Inkling,
            "mtp.fc.weight"
        ));
        for kind in [ModelKind::Qwen3Next, ModelKind::Qwen35Moe] {
            assert!(is_routed_expert_key(
                kind,
                "model.layers.0.mlp.experts.1.down_proj.weight"
            ));
            assert!(!is_routed_expert_key(
                kind,
                "mtp.layers.0.mlp.experts.1.down_proj.weight"
            ));
        }
    }

    fn save_zero_checkpoint(model: &impl ModuleParameters, directory: &Path, stream: &Stream) {
        let parameters = model.parameters().flatten();
        let arrays = parameters
            .iter()
            .map(|(name, parameter)| {
                (
                    name.to_string(),
                    zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, array)| (name.as_str(), array)),
            None,
            directory.join("model.safetensors"),
        )
        .unwrap();
    }

    fn rank_one_topology() -> ParallelTopology {
        ParallelTopology::from_rank(2, 1, 1, 1, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    struct IdentityBank;

    impl LocalExpertBank for IdentityBank {
        fn execute_local_routes(
            &mut self,
            hidden: &Array,
            _local_expert_ids: &Array,
            _stream: &Stream,
        ) -> Result<Array, Error> {
            Ok(hidden.clone())
        }
    }

    #[test]
    fn assignment_policies_and_round_trips() {
        let balanced = ExpertAssignment::balanced(7, 3, 1).unwrap();
        assert_eq!(balanced.owners(), &[0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(balanced.local_global_expert_ids(), &[3, 4]);
        assert_eq!(balanced.owner_local_id(4), Some(1));
        assert_eq!(balanced.global_id(1), Some(4));

        let rr = ExpertAssignment::round_robin(7, 3, 1).unwrap();
        assert_eq!(rr.local_global_expert_ids(), &[1, 4]);
        assert_eq!(rr.owner_local_id(4), Some(1));

        let explicit = ExpertAssignment::explicit(vec![1, 0, 1, 0], 2, 0).unwrap();
        assert_eq!(explicit.local_global_expert_ids(), &[1, 3]);
        assert_eq!(explicit.global_id(1), Some(3));
    }

    #[test]
    fn assignment_rejects_invalid_or_empty_ownership() {
        assert!(ExpertAssignment::balanced(0, 2, 0).is_err());
        assert!(ExpertAssignment::balanced(1, 2, 0).is_err());
        assert!(ExpertAssignment::explicit(vec![0, 2], 2, 0).is_err());
        assert!(ExpertAssignment::explicit(vec![0, 0], 2, 0).is_err());
        assert!(ExpertAssignment::explicit_with_empty(vec![0, 0], 2, 1, true).is_ok());
        assert!(resolve_model_assignment(
            Some(ExpertAssignment::balanced(6, 2, 1).unwrap()),
            4,
            rank_one_topology(),
        )
        .is_err());
    }

    #[test]
    fn already_local_expert_bank_is_not_sliced_again_on_nonzero_rank() {
        let stream = stream();
        let assignment = ExpertAssignment::balanced(4, 2, 1).unwrap();
        assert!(!expert_bank_needs_slicing(2, &assignment).unwrap());
        assert!(expert_bank_needs_slicing(4, &assignment).unwrap());
        assert!(expert_bank_needs_slicing(3, &assignment).is_err());

        let mut bank = PackedSwiGluExperts::new(2, 4, 3, None, None, &stream).unwrap();
        let gate_up_shape = bank.gate_up_proj.shape().to_vec();
        let down_shape = bank.down_proj.shape().to_vec();
        let expected_bytes = parameter_bytes(&bank);
        let bytes = finalize_qwen3_expert_bank(&mut bank, &assignment, &stream).unwrap();

        assert_eq!(bytes, expected_bytes);
        assert_eq!(bank.num_experts, 2);
        assert_eq!(bank.gate_up_proj.shape(), gate_up_shape);
        assert_eq!(bank.down_proj.shape(), down_shape);
    }

    #[test]
    fn qwen3_round_robin_loader_materializes_only_rank_one_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let fixture = tempfile::tempdir().unwrap();
        std::fs::write(
            fixture.path().join("config.json"),
            r#"{
              "model_type":"qwen3_moe","hidden_size":32,"num_hidden_layers":1,
              "intermediate_size":64,"num_attention_heads":1,"num_key_value_heads":1,
              "head_dim":32,"rms_norm_eps":0.000001,"vocab_size":32,
              "max_position_embeddings":128,"rope_theta":1000000.0,
              "tie_word_embeddings":false,"rope_scaling":null,
              "moe_intermediate_size":32,"num_experts":4,
              "num_experts_per_tok":2,"norm_topk_prob":true
            }"#,
        )
        .unwrap();
        let args = qwen3::get_qwen3_model_args(fixture.path()).unwrap();
        let source = qwen3::Model::new(args, stream).unwrap();
        save_zero_checkpoint(&source, fixture.path(), stream);

        let options = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
            .with_parallel_topology(rank_one_topology());
        let assignment = ExpertAssignment::round_robin(4, 2, 1).unwrap();
        let loaded = load_expert_parallel_model_with_options_and_assignment(
            fixture.path(),
            options,
            assignment,
            stream,
            weights_stream,
        )
        .unwrap();

        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[1, 3]);
        assert_eq!(
            loaded.info.assignment.policy(),
            &ExpertAssignmentPolicy::RoundRobin
        );
        let ExpertArchitecture::Qwen3(model) = &loaded.architecture else {
            panic!("expected Qwen3");
        };
        let qwen3::FeedForward::Moe(moe) = &model.model.layers[0].mlp else {
            panic!("expected sparse MoE layer");
        };
        assert_eq!(moe.experts.num_experts, 2);
        assert_eq!(moe.experts.gate_up_proj.shape(), &[2, 64, 4]);
        assert_eq!(moe.experts.down_proj.shape(), &[2, 32, 4]);
        assert_eq!(moe.experts.gate_up_proj.dtype(), Dtype::Uint32);
        assert_eq!(
            moe.experts
                .gate_up_proj_scales
                .as_ref()
                .as_ref()
                .unwrap()
                .shape(),
            &[2, 64, 1]
        );

        let expert_options = ExpertCacheLoadOptions::new(
            crate::layerwise::LayerwiseLoadOptions::new(
                crate::offload::OffloadConfig::new(None, None, 1).unwrap(),
            ),
            crate::offload::OffloadConfig::new(Some(1 << 20), Some(0), 1).unwrap(),
            1 << 20,
        )
        .unwrap();
        let options = ModelLoadOptions::with_parallel(rank_one_topology())
            .with_weight_residency(WeightResidency::SparseExpertCache(expert_options));
        let assignment = ExpertAssignment::round_robin(4, 2, 1).unwrap();
        let cached = load_expert_parallel_model_with_options_and_assignment(
            fixture.path(),
            options,
            assignment,
            stream,
            weights_stream,
        )
        .unwrap();
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 2);
        assert_eq!(report.host_resident_experts, 0);
        assert_eq!(report.device_resident_experts, 0);
        assert_eq!(cached.info.routed_expert_bytes, 0);
        assert!(cached.info.owned_expert_bytes > 0);
        assert_eq!(
            report
                .residency
                .units()
                .iter()
                .map(|unit| unit.id().as_str())
                .collect::<Vec<_>>(),
            vec![
                "expert.layer.00000.global.00001",
                "expert.layer.00000.global.00003"
            ]
        );
    }

    #[test]
    fn deepseek_explicit_loader_materializes_only_rank_one_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let fixture = tempfile::tempdir().unwrap();
        std::fs::write(
            fixture.path().join("config.json"),
            r#"{
              "model_type":"deepseek_v3","hidden_size":32,"intermediate_size":64,
              "moe_intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":1,
              "vocab_size":32,"rms_norm_eps":0.000001,"max_position_embeddings":128,
              "rope_theta":10000,"q_lora_rank":null,"kv_lora_rank":32,
              "qk_nope_head_dim":32,"qk_rope_head_dim":8,"v_head_dim":32,
              "first_k_dense_replace":0,"moe_layer_freq":1,"n_routed_experts":4,
              "n_shared_experts":1,"num_experts_per_tok":2,"n_group":2,
              "topk_group":1,"topk_method":"noaux_tc","scoring_func":"sigmoid",
              "norm_topk_prob":true,"routed_scaling_factor":1.0,
              "num_nextn_predict_layers":0,"tie_word_embeddings":false
            }"#,
        )
        .unwrap();
        let args = deepseek_v3::get_model_args(fixture.path()).unwrap();
        let source = deepseek_v3::Model::new(args.clone(), stream).unwrap();
        let parameters = source.parameters().flatten();
        let mut arrays = parameters
            .iter()
            .map(|(name, parameter)| {
                (
                    name.to_string(),
                    zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        for expert in 0..args.n_routed_experts {
            for (projection, shape) in [
                ("gate_proj", [args.moe_intermediate_size, args.hidden_size]),
                ("up_proj", [args.moe_intermediate_size, args.hidden_size]),
                ("down_proj", [args.hidden_size, args.moe_intermediate_size]),
            ] {
                arrays.push((
                    format!("model.layers.0.mlp.experts.{expert}.{projection}.weight"),
                    Array::zeros::<f32>(&shape, stream).unwrap(),
                ));
            }
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, array)| (name.as_str(), array)),
            None,
            fixture.path().join("model.safetensors"),
        )
        .unwrap();

        let options = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
            .with_parallel_topology(rank_one_topology());
        let assignment = ExpertAssignment::explicit(vec![1, 0, 0, 1], 2, 1).unwrap();
        let loaded = load_expert_parallel_model_with_options_and_assignment(
            fixture.path(),
            options,
            assignment,
            stream,
            weights_stream,
        )
        .unwrap();

        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[0, 3]);
        assert_eq!(
            loaded.info.assignment.policy(),
            &ExpertAssignmentPolicy::Explicit(vec![1, 0, 0, 1])
        );
        let ExpertArchitecture::DeepSeek(model) = &loaded.architecture else {
            panic!("expected DeepSeek");
        };
        let deepseek_v3::FeedForward::Moe(moe) = &model.model.layers[0].mlp else {
            panic!("expected sparse MoE layer");
        };
        assert_eq!(moe.experts.num_experts, 2);
        assert_eq!(
            moe.experts.gate_proj.as_ref().as_ref().unwrap().shape(),
            &[2, 32, 4]
        );
        assert_eq!(
            moe.experts
                .gate_proj_scales
                .as_ref()
                .as_ref()
                .unwrap()
                .shape(),
            &[2, 32, 1]
        );
    }

    #[test]
    fn compact_routes_preserves_tokens_slots_and_global_ids() {
        let stream = stream();
        let assignment = ExpertAssignment::balanced(4, 2, 1).unwrap();
        let hidden = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let ids = Array::from_slice(&[0i32, 2, 1, 3], &[2, 2]);
        let weights = Array::from_slice(&[0.1f32, 0.9, 0.25, 0.75], &[2, 2]);
        let (routes, stats) =
            compact_local_routes(&hidden, &ids, &weights, &assignment, &stream).unwrap();
        eval([
            &routes.global_expert_ids,
            &routes.local_expert_ids,
            &routes.token_indices,
            &routes.slot_indices,
            &routes.weights,
        ])
        .unwrap();
        assert_eq!(stats.total_routes, 4);
        assert_eq!(stats.local_routes, 2);
        assert_eq!(stats.synchronization_count, 1);
        assert_eq!(
            routes
                .global_expert_ids
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[2, 3]
        );
        assert_eq!(
            routes
                .local_expert_ids
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[0, 1]
        );
        assert_eq!(
            routes.token_indices.evaluated().unwrap().as_slice::<i32>(),
            &[0, 1]
        );
        assert_eq!(
            routes.slot_indices.evaluated().unwrap().as_slice::<i32>(),
            &[1, 1]
        );
    }

    #[test]
    fn replicated_dispatch_recombines_weights_exactly() {
        let stream = stream();
        let group = Group::init(false, Backend::Any).unwrap();
        assert_eq!(group.size(), 1);
        let assignment = ExpertAssignment::balanced(3, 1, 0).unwrap();
        let hidden = Array::from_slice(&[2.0f32, 4.0, 10.0, 20.0], &[2, 2]);
        let ids = Array::from_slice(&[0i32, 2, 1, 1], &[2, 2]);
        let weights = Array::from_slice(&[0.25f32, 0.75, 0.4, 0.6], &[2, 2]);
        let returned = dispatch_replicated(
            &hidden,
            &ids,
            &weights,
            &assignment,
            &mut IdentityBank,
            &group,
            &stream,
        )
        .unwrap();
        eval([&returned.reduced_output]).unwrap();
        assert_eq!(
            returned
                .reduced_output
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[2.0, 4.0, 10.0, 20.0]
        );
    }

    #[test]
    fn all_to_all_v_singleton_preserves_payload_and_zero_counts() {
        let stream = stream();
        let group = Group::init(false, Backend::Any).unwrap();
        let payload = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        let received = all_to_all_v(&[payload], &group, &stream).unwrap();
        eval([&received.received]).unwrap();
        assert_eq!(received.source_counts, vec![2]);
        assert_eq!(
            received.received.evaluated().unwrap().as_slice::<i32>(),
            &[1, 2, 3, 4]
        );

        let empty = Array::from_slice::<i32>(&[], &[0, 2]);
        let received = all_to_all_v(&[empty], &group, &stream).unwrap();
        assert_eq!(received.source_counts, vec![0]);
        assert_eq!(received.received.shape(), &[0, 2]);
    }

    #[test]
    fn sharded_and_replicated_dispatch_match_on_singleton() {
        let stream = stream();
        let group = Group::init(false, Backend::Any).unwrap();
        let assignment = ExpertAssignment::balanced(2, 1, 0).unwrap();
        let hidden = Array::from_slice(&[2.0f32, 4.0, 10.0, 20.0], &[2, 2]);
        let ids = Array::from_slice(&[0i32, 1, 1, 0], &[2, 2]);
        let weights = Array::from_slice(&[0.25f32, 0.75, 0.4, 0.6], &[2, 2]);
        let replicated = dispatch_replicated(
            &hidden,
            &ids,
            &weights,
            &assignment,
            &mut IdentityBank,
            &group,
            &stream,
        )
        .unwrap();
        let route_tokens = Array::from_slice(&[0i32, 0, 1, 1], &[4]);
        let routed_hidden = hidden.take_axis(&route_tokens, 0, &stream).unwrap();
        let sharded = dispatch_sharded(
            ShardedRouteBlocks {
                hidden: vec![routed_hidden],
                global_expert_ids: vec![ids.reshape(&[4], &stream).unwrap()],
                original_route_indices: vec![
                    Array::arange::<i32, i32>(Some(0), 4, None, &stream).unwrap()
                ],
                weights: vec![weights.reshape(&[4], &stream).unwrap()],
                top_k: 2,
                source_tokens: 2,
            },
            &assignment,
            &mut IdentityBank,
            &group,
            &stream,
        )
        .unwrap();
        eval([&replicated.reduced_output, &sharded.output]).unwrap();
        assert_eq!(
            replicated
                .reduced_output
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            sharded.output.evaluated().unwrap().as_slice::<f32>()
        );
    }
}
