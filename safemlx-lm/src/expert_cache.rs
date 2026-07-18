//! Architecture-independent sparse routed-expert caching.
//!
//! Each logical expert is an atomic disk-planned residency unit. Route ids are
//! inspected once per routed block, validated before acquisition, coalesced in
//! deterministic global-id order, and rewritten to a temporary compact bank.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use safemlx::{ops::concatenate_axis, transforms::eval, Array, Dtype, Stream};

use crate::{
    layerwise::LayerwiseLoadOptions,
    offload::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    },
    residency::{
        OffloadUnit, ResidencyError, ResidencyManager, ResidencyReport, ResidentUnitLease,
    },
    weight_store::WeightStore,
};

/// Stable architecture-neutral identity for one layer-local global expert.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExpertIdentity {
    /// Zero-based decoder layer identity.
    pub layer: usize,
    /// Global expert identity from the model router.
    pub global_expert: usize,
}

impl ExpertIdentity {
    /// Creates one logical expert identity.
    pub const fn new(layer: usize, global_expert: usize) -> Self {
        Self {
            layer,
            global_expert,
        }
    }

    /// Returns the deterministic residency unit identifier.
    pub fn unit_id(self) -> OffloadUnitId {
        OffloadUnitId::new(format!(
            "expert.layer.{:05}.global.{:05}",
            self.layer, self.global_expert
        ))
        .expect("expert unit identifier is non-empty")
    }
}

/// Public controls for sparse routed-expert residency.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ExpertCacheLoadOptions {
    /// Residency controls for router, attention, normalization, and dense weights.
    pub non_expert: LayerwiseLoadOptions,
    /// Independent host/device budgets and eviction policy for expert units.
    pub experts: OffloadConfig,
    /// Maximum materialized temporary compact-bank bytes for one routed block.
    pub compact_bank_scratch_bytes: u64,
}

impl ExpertCacheLoadOptions {
    /// Creates strict sparse expert caching options.
    pub fn new(
        non_expert: LayerwiseLoadOptions,
        experts: OffloadConfig,
        compact_bank_scratch_bytes: u64,
    ) -> Result<Self, ExpertCacheError> {
        if compact_bank_scratch_bytes == 0 {
            return Err(ExpertCacheError::ZeroScratchLimit);
        }
        Ok(Self {
            non_expert,
            experts,
            compact_bank_scratch_bytes,
        })
    }
}

impl Default for ExpertCacheLoadOptions {
    fn default() -> Self {
        Self {
            non_expert: LayerwiseLoadOptions::default(),
            experts: OffloadConfig::default(),
            compact_bank_scratch_bytes: u64::MAX,
        }
    }
}

/// Public execution-path classification for independent cache statistics.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExpertPass {
    /// Prompt processing with more than one input token.
    Prefill,
    /// Autoregressive processing of one input token.
    Decode,
}

/// One atomic expert definition supplied by an architecture adapter.
pub struct ExpertCatalogEntry {
    identity: ExpertIdentity,
    unit: OffloadUnit,
    bytes: u64,
}

impl ExpertCatalogEntry {
    /// Creates one catalog entry and verifies its stable unit identity.
    pub fn new(
        identity: ExpertIdentity,
        unit: OffloadUnit,
        bytes: u64,
    ) -> Result<Self, ExpertCacheError> {
        if bytes == 0 {
            return Err(ExpertCacheError::ZeroSizedExpert { identity });
        }
        let expected = identity.unit_id();
        if unit.id() != &expected {
            return Err(ExpertCacheError::UnitIdentityMismatch {
                identity,
                expected,
                actual: unit.id().clone(),
            });
        }
        Ok(Self {
            identity,
            unit,
            bytes,
        })
    }

    /// Returns the logical identity.
    pub const fn identity(&self) -> ExpertIdentity {
        self.identity
    }

    /// Returns the atomic materialized byte length.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Tier-local cache request counters.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ExpertTierStatistics {
    /// Logical expert acquisition requests after duplicate coalescing.
    pub requests: u64,
    /// Requests served by an already resident copy.
    pub hits: u64,
    /// Requests that materialized or promoted a copy.
    pub misses: u64,
    /// Copies evicted while satisfying cache requests.
    pub evictions: u64,
    /// Bytes evicted while satisfying cache requests.
    pub eviction_bytes: u64,
}

/// Cumulative statistics for one public execution-path class.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ExpertPassStatistics {
    /// Route rows requested by the router, including duplicates.
    pub requested_routes: u64,
    /// Distinct logical experts requested after coalescing.
    pub distinct_experts: u64,
    /// Duplicate requests eliminated before materialization.
    pub coalesced_duplicates: u64,
    /// Temporary compact banks built.
    pub compact_banks: u64,
    /// Cumulative compact-bank bytes.
    pub compact_bank_bytes: u64,
    /// Peak temporary compact-bank bytes.
    pub peak_compact_bank_bytes: u64,
    /// Cumulative compact-bank construction time.
    pub compact_bank_time: Duration,
    /// Time waiting on synchronous expert materialization or promotion.
    pub materialization_wait: Duration,
    /// Host-tier cache activity.
    pub host: ExpertTierStatistics,
    /// Device-tier cache activity.
    pub device: ExpertTierStatistics,
}

/// Point-in-time sparse expert residency and execution report.
pub struct ExpertCacheReport {
    /// Owned logical expert count.
    pub owned_experts: usize,
    /// Owned logical expert bytes, including cold checkpoint-only experts.
    pub owned_bytes: u64,
    /// Current host-resident expert count.
    pub host_resident_experts: usize,
    /// Current device-resident expert count.
    pub device_resident_experts: usize,
    /// Current host-resident expert bytes.
    pub host_resident_bytes: u64,
    /// Current device-resident expert bytes.
    pub device_resident_bytes: u64,
    /// Peak host-resident expert bytes.
    pub peak_host_resident_bytes: u64,
    /// Peak device-resident expert bytes.
    pub peak_device_resident_bytes: u64,
    /// Prompt-processing statistics.
    pub prefill: ExpertPassStatistics,
    /// Autoregressive decode statistics.
    pub decode: ExpertPassStatistics,
    /// Underlying logical transfer and checkpoint diagnostics.
    pub residency: ResidencyReport,
}

#[derive(Default)]
struct ExpertStatistics {
    prefill: ExpertPassStatistics,
    decode: ExpertPassStatistics,
}

impl ExpertStatistics {
    fn pass_mut(&mut self, pass: ExpertPass) -> &mut ExpertPassStatistics {
        match pass {
            ExpertPass::Prefill => &mut self.prefill,
            ExpertPass::Decode => &mut self.decode,
        }
    }
}

/// Shared sparse expert catalog, scheduler, residency manager, and telemetry.
pub struct ExpertCache {
    manager: ResidencyManager,
    catalog: BTreeMap<ExpertIdentity, u64>,
    layer_expert_counts: BTreeMap<usize, usize>,
    host_budget: Option<u64>,
    scratch_limit: u64,
    statistics: Mutex<ExpertStatistics>,
}

impl ExpertCache {
    /// Creates a disk-planned cache over exactly the supplied owned experts.
    pub fn new<S>(
        store: Arc<S>,
        entries: impl IntoIterator<Item = ExpertCatalogEntry>,
        options: ExpertCacheLoadOptions,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ExpertCacheError>
    where
        S: WeightStore + Send + Sync + 'static,
    {
        if options.compact_bank_scratch_bytes == 0 {
            return Err(ExpertCacheError::ZeroScratchLimit);
        }
        let mut catalog = BTreeMap::new();
        let mut definitions = Vec::new();
        let mut specs = Vec::new();
        let mut layer_expert_counts = BTreeMap::new();
        for entry in entries {
            if catalog.insert(entry.identity, entry.bytes).is_some() {
                return Err(ExpertCacheError::DuplicateExpert {
                    identity: entry.identity,
                });
            }
            *layer_expert_counts.entry(entry.identity.layer).or_insert(0) += 1;
            specs.push(OffloadUnitSpec::new(
                entry.identity.unit_id(),
                entry.bytes,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )?);
            definitions.push(entry.unit);
        }
        if catalog.is_empty() {
            return Err(ExpertCacheError::EmptyCatalog);
        }
        let plan = OffloadPlan::new(options.experts, specs)?;
        let manager =
            ResidencyManager::new(store, plan, definitions, source_stream, device_stream)?;
        manager.initialize()?;
        Ok(Self {
            manager,
            catalog,
            layer_expert_counts,
            host_budget: options.experts.host_budget_bytes(),
            scratch_limit: options.compact_bank_scratch_bytes,
            statistics: Mutex::new(ExpertStatistics::default()),
        })
    }

    /// Returns the underlying reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.manager
    }

    /// Discovers, validates, coalesces, and acquires routed experts.
    ///
    /// Router ids are evaluated together and synchronized once. The returned
    /// leases must remain live until compact-bank output evaluation completes.
    pub fn acquire_routes(
        &self,
        layer: usize,
        routed_ids: &Array,
        pass: ExpertPass,
        stream: &Stream,
    ) -> Result<AcquiredExperts, ExpertCacheError> {
        if !matches!(
            routed_ids.dtype(),
            Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
        ) {
            return Err(ExpertCacheError::InvalidRouteDtype {
                actual: routed_ids.dtype(),
            });
        }
        let normalized = if routed_ids.dtype() == Dtype::Int32 {
            routed_ids.clone()
        } else {
            routed_ids.as_dtype(Dtype::Int32, stream)?
        };
        eval([&normalized])?;
        stream.synchronize()?;
        let evaluated = normalized.evaluated()?;
        self.acquire_route_slice(
            layer,
            evaluated.as_slice::<i32>(),
            routed_ids.shape(),
            pass,
            stream,
        )
    }

    /// Acquires a caller-provided route table while preserving its exact shape and order.
    pub fn acquire_route_slice(
        &self,
        layer: usize,
        routed_ids: &[i32],
        route_shape: &[i32],
        pass: ExpertPass,
        stream: &Stream,
    ) -> Result<AcquiredExperts, ExpertCacheError> {
        let expected_elements = route_shape.iter().try_fold(1usize, |count, dimension| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| ExpertCacheError::InvalidRouteShape(route_shape.to_vec()))?;
            count
                .checked_mul(dimension)
                .ok_or_else(|| ExpertCacheError::InvalidRouteShape(route_shape.to_vec()))
        })?;
        if expected_elements != routed_ids.len() {
            return Err(ExpertCacheError::RouteShapeMismatch {
                shape: route_shape.to_vec(),
                elements: routed_ids.len(),
            });
        }
        let layer_count = self
            .layer_expert_counts
            .get(&layer)
            .copied()
            .ok_or(ExpertCacheError::UnknownLayer { layer })?;
        let mut demand = BTreeMap::<ExpertIdentity, u64>::new();
        for id in routed_ids {
            let global_expert =
                usize::try_from(*id).map_err(|_| ExpertCacheError::InvalidExpertId {
                    layer,
                    expert: i64::from(*id),
                    known_owned_experts: layer_count,
                })?;
            let identity = ExpertIdentity::new(layer, global_expert);
            if !self.catalog.contains_key(&identity) {
                return Err(ExpertCacheError::MissingOwnedExpert { identity });
            }
            let count = demand.entry(identity).or_insert(0);
            *count = count.saturating_add(1);
        }

        let scratch_bytes = demand.keys().try_fold(0u64, |total, identity| {
            total
                .checked_add(self.catalog[identity])
                .ok_or(ExpertCacheError::ByteOverflow)
        })?;
        if scratch_bytes > self.scratch_limit {
            return Err(ExpertCacheError::ScratchLimitExceeded {
                required_bytes: scratch_bytes,
                limit_bytes: self.scratch_limit,
                distinct_experts: demand.len(),
            });
        }

        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        let translations = compact_ids
            .iter()
            .enumerate()
            .map(|(compact, identity)| (*identity, compact as i32))
            .collect::<BTreeMap<_, _>>();
        let compact_values = routed_ids
            .iter()
            .map(|id| translations[&ExpertIdentity::new(layer, *id as usize)])
            .collect::<Vec<_>>();
        let compact_routes = Array::from_slice(&compact_values, route_shape).copy(stream)?;

        let before = self.resident_snapshot()?;
        let started = Instant::now();
        let mut leases = Vec::with_capacity(compact_ids.len());
        let mut host_hits = 0u64;
        let mut host_misses = 0u64;
        let mut device_hits = 0u64;
        let mut device_misses = 0u64;
        for identity in &compact_ids {
            let unit = identity.unit_id();
            let route_demand = demand[identity];
            let host_hit = self.manager.is_resident(&unit, MemoryTier::Host)?;
            let device_hit = self.manager.is_resident(&unit, MemoryTier::Device)?;
            if host_hit {
                host_hits = host_hits.saturating_add(1);
            } else {
                host_misses = host_misses.saturating_add(1);
            }
            if device_hit {
                device_hits = device_hits.saturating_add(1);
            } else {
                device_misses = device_misses.saturating_add(1);
            }

            if !device_hit && self.host_budget != Some(0) {
                match self
                    .manager
                    .acquire_with_demand(&unit, MemoryTier::Host, route_demand)
                {
                    Ok(host) => drop(host),
                    Err(ResidencyError::BudgetExhausted {
                        tier: MemoryTier::Host,
                        ..
                    }) => {}
                    Err(error) => return Err(error.into()),
                }
            } else if host_hit {
                let host =
                    self.manager
                        .acquire_with_demand(&unit, MemoryTier::Host, route_demand)?;
                drop(host);
            }
            leases.push(self.manager.acquire_with_demand(
                &unit,
                MemoryTier::Device,
                route_demand,
            )?);
        }
        let wait = started.elapsed();
        let after = self.resident_snapshot()?;
        let (host_evictions, host_eviction_bytes) = before.evicted(&after, MemoryTier::Host);
        let (device_evictions, device_eviction_bytes) = before.evicted(&after, MemoryTier::Device);

        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| ExpertCacheError::StatisticsPoisoned)?;
        let stats = statistics.pass_mut(pass);
        let routes = routed_ids.len() as u64;
        let distinct = compact_ids.len() as u64;
        stats.requested_routes = stats.requested_routes.saturating_add(routes);
        stats.distinct_experts = stats.distinct_experts.saturating_add(distinct);
        stats.coalesced_duplicates = stats
            .coalesced_duplicates
            .saturating_add(routes.saturating_sub(distinct));
        stats.materialization_wait = stats.materialization_wait.saturating_add(wait);
        stats.host.requests = stats.host.requests.saturating_add(distinct);
        stats.host.hits = stats.host.hits.saturating_add(host_hits);
        stats.host.misses = stats.host.misses.saturating_add(host_misses);
        stats.host.evictions = stats.host.evictions.saturating_add(host_evictions);
        stats.host.eviction_bytes = stats
            .host
            .eviction_bytes
            .saturating_add(host_eviction_bytes);
        stats.device.requests = stats.device.requests.saturating_add(distinct);
        stats.device.hits = stats.device.hits.saturating_add(device_hits);
        stats.device.misses = stats.device.misses.saturating_add(device_misses);
        stats.device.evictions = stats.device.evictions.saturating_add(device_evictions);
        stats.device.eviction_bytes = stats
            .device
            .eviction_bytes
            .saturating_add(device_eviction_bytes);
        drop(statistics);

        Ok(AcquiredExperts {
            identities: compact_ids,
            demand: demand.into_values().collect(),
            compact_routes,
            scratch_bytes,
            pass,
            leases,
        })
    }

    /// Records a completed compact-bank construction.
    pub fn record_compact_bank(
        &self,
        pass: ExpertPass,
        bytes: u64,
        duration: Duration,
    ) -> Result<(), ExpertCacheError> {
        if bytes > self.scratch_limit {
            return Err(ExpertCacheError::ScratchLimitExceeded {
                required_bytes: bytes,
                limit_bytes: self.scratch_limit,
                distinct_experts: 0,
            });
        }
        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| ExpertCacheError::StatisticsPoisoned)?;
        let stats = statistics.pass_mut(pass);
        stats.compact_banks = stats.compact_banks.saturating_add(1);
        stats.compact_bank_bytes = stats.compact_bank_bytes.saturating_add(bytes);
        stats.peak_compact_bank_bytes = stats.peak_compact_bank_bytes.max(bytes);
        stats.compact_bank_time = stats.compact_bank_time.saturating_add(duration);
        Ok(())
    }

    /// Returns current expert residency, transfer, storage, and pass statistics.
    pub fn report(&self) -> Result<ExpertCacheReport, ExpertCacheError> {
        let residency = self.manager.report()?;
        let mut host_resident_experts = 0;
        let mut device_resident_experts = 0;
        let mut host_resident_bytes = 0u64;
        let mut device_resident_bytes = 0u64;
        for unit in residency.units() {
            if unit.host_resident() {
                host_resident_experts += 1;
                host_resident_bytes = host_resident_bytes.saturating_add(unit.expected_bytes());
            }
            if unit.device_resident() {
                device_resident_experts += 1;
                device_resident_bytes = device_resident_bytes.saturating_add(unit.expected_bytes());
            }
        }
        let statistics = self
            .statistics
            .lock()
            .map_err(|_| ExpertCacheError::StatisticsPoisoned)?;
        Ok(ExpertCacheReport {
            owned_experts: self.catalog.len(),
            owned_bytes: self.catalog.values().copied().sum(),
            host_resident_experts,
            device_resident_experts,
            host_resident_bytes,
            device_resident_bytes,
            peak_host_resident_bytes: residency
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Host),
            peak_device_resident_bytes: residency
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device),
            prefill: statistics.prefill,
            decode: statistics.decode,
            residency,
        })
    }

    fn resident_snapshot(&self) -> Result<ResidentSnapshot, ExpertCacheError> {
        let report = self.manager.report()?;
        Ok(ResidentSnapshot {
            host: report
                .units()
                .iter()
                .filter(|unit| unit.host_resident())
                .map(|unit| (unit.id().clone(), unit.expected_bytes()))
                .collect(),
            device: report
                .units()
                .iter()
                .filter(|unit| unit.device_resident())
                .map(|unit| (unit.id().clone(), unit.expected_bytes()))
                .collect(),
        })
    }
}

struct ResidentSnapshot {
    host: BTreeMap<OffloadUnitId, u64>,
    device: BTreeMap<OffloadUnitId, u64>,
}

impl ResidentSnapshot {
    fn evicted(&self, after: &Self, tier: MemoryTier) -> (u64, u64) {
        let (before, after) = match tier {
            MemoryTier::Host => (&self.host, &after.host),
            MemoryTier::Device => (&self.device, &after.device),
            MemoryTier::Disk => return (0, 0),
        };
        before
            .iter()
            .filter(|(id, _)| !after.contains_key(*id))
            .fold((0u64, 0u64), |(count, bytes), (_, size)| {
                (count.saturating_add(1), bytes.saturating_add(*size))
            })
    }
}

/// A deterministic compact route table and the leases protecting its sources.
pub struct AcquiredExperts {
    identities: Vec<ExpertIdentity>,
    demand: Vec<u64>,
    compact_routes: Array,
    scratch_bytes: u64,
    pass: ExpertPass,
    leases: Vec<ResidentUnitLease>,
}

impl AcquiredExperts {
    /// Returns selected experts in compact-bank order.
    pub fn identities(&self) -> &[ExpertIdentity] {
        &self.identities
    }

    /// Returns duplicate-preserving demand counts in compact-bank order.
    pub fn demand(&self) -> &[u64] {
        &self.demand
    }

    /// Returns routes rewritten bijectively to compact-bank ids.
    pub const fn compact_routes(&self) -> &Array {
        &self.compact_routes
    }

    /// Returns the conservatively reserved compact-bank byte count.
    pub const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Returns the execution-path classification used for telemetry.
    pub const fn pass(&self) -> ExpertPass {
        self.pass
    }

    /// Returns source leases in the same order as [`Self::identities`].
    pub fn leases(&self) -> &[ResidentUnitLease] {
        &self.leases
    }

    /// Concatenates one required per-expert binding along its leading axis.
    pub fn compact_binding(&self, name: &str, stream: &Stream) -> Result<Array, ExpertCacheError> {
        let values = self
            .leases
            .iter()
            .map(|lease| lease.array(name).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        if values.is_empty() {
            return Err(ExpertCacheError::EmptyCompactBinding {
                name: name.to_string(),
            });
        }
        Ok(concatenate_axis(&values, 0, stream)?)
    }

    /// Concatenates an optional companion binding when every expert provides it.
    pub fn optional_compact_binding(
        &self,
        name: &str,
        stream: &Stream,
    ) -> Result<Option<Array>, ExpertCacheError> {
        let present = self
            .leases
            .iter()
            .map(|lease| lease.binding_names().any(|binding| binding == name))
            .collect::<Vec<_>>();
        if present.iter().all(|value| !value) {
            return Ok(None);
        }
        if present.iter().any(|value| !value) {
            return Err(ExpertCacheError::InconsistentCompanion {
                name: name.to_string(),
            });
        }
        self.compact_binding(name, stream).map(Some)
    }

    /// Returns whether no routed experts were selected.
    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }
}

/// Structured sparse expert cache failures.
#[derive(Debug, thiserror::Error)]
pub enum ExpertCacheError {
    /// No expert definitions were supplied.
    #[error("sparse expert cache requires at least one owned expert")]
    EmptyCatalog,
    /// Compact-bank scratch accounting was disabled with a zero limit.
    #[error("sparse expert compact-bank scratch limit must be nonzero")]
    ZeroScratchLimit,
    /// One logical expert declared no materialized bytes.
    #[error("expert {identity:?} must contain at least one byte")]
    ZeroSizedExpert {
        /// Invalid logical identity.
        identity: ExpertIdentity,
    },
    /// Two catalog entries used the same layer/global identity.
    #[error("duplicate sparse expert catalog entry {identity:?}")]
    DuplicateExpert {
        /// Duplicated logical identity.
        identity: ExpertIdentity,
    },
    /// The architecture adapter used a noncanonical residency unit id.
    #[error("expert {identity:?} requires unit id {expected}, got {actual}")]
    UnitIdentityMismatch {
        /// Logical catalog identity.
        identity: ExpertIdentity,
        /// Required stable unit id.
        expected: OffloadUnitId,
        /// Adapter-supplied unit id.
        actual: OffloadUnitId,
    },
    /// No owned expert catalog exists for this decoder layer.
    #[error("sparse expert cache has no catalog for layer {layer}")]
    UnknownLayer {
        /// Missing decoder layer identity.
        layer: usize,
    },
    /// A route id was negative or otherwise invalid.
    #[error("invalid routed expert id {expert} for layer {layer}; this rank catalogs {known_owned_experts} owned experts")]
    InvalidExpertId {
        /// Decoder layer containing the route.
        layer: usize,
        /// Invalid signed route value.
        expert: i64,
        /// Owned experts cataloged for diagnostics.
        known_owned_experts: usize,
    },
    /// A valid global route referred to an expert this cache does not own.
    #[error("routed expert {identity:?} is not owned by this cache")]
    MissingOwnedExpert {
        /// Requested non-owned global identity.
        identity: ExpertIdentity,
    },
    /// Router ids used an unsupported scalar type.
    #[error("routed expert ids must use an integer dtype, got {actual:?}")]
    InvalidRouteDtype {
        /// Unsupported router-id scalar type.
        actual: Dtype,
    },
    /// A supplied route shape had a negative dimension or overflowed.
    #[error("invalid routed expert shape {0:?}")]
    InvalidRouteShape(Vec<i32>),
    /// Route shape and host values disagreed.
    #[error("routed expert shape {shape:?} does not describe {elements} values")]
    RouteShapeMismatch {
        /// Declared route shape.
        shape: Vec<i32>,
        /// Supplied host value count.
        elements: usize,
    },
    /// Selected experts exceed the configured temporary compact-bank allowance.
    #[error("compact expert bank for {distinct_experts} experts requires {required_bytes} bytes, exceeding the {limit_bytes}-byte scratch limit")]
    ScratchLimitExceeded {
        /// Required compact-bank bytes.
        required_bytes: u64,
        /// Configured compact-bank byte limit.
        limit_bytes: u64,
        /// Selected unique expert count.
        distinct_experts: usize,
    },
    /// Expert byte arithmetic overflowed.
    #[error("sparse expert byte accounting overflowed")]
    ByteOverflow,
    /// Cache statistics mutex was poisoned by a panic.
    #[error("sparse expert cache statistics are unavailable after a panic")]
    StatisticsPoisoned,
    /// A required compact binding had no selected source experts.
    #[error("compact expert binding {name:?} has no source arrays")]
    EmptyCompactBinding {
        /// Required binding name.
        name: String,
    },
    /// Optional companion presence differed across selected experts.
    #[error("compact expert companion {name:?} is missing from only part of the selected bank")]
    InconsistentCompanion {
        /// Inconsistent optional binding name.
        name: String,
    },
    /// Invalid offload plan configuration.
    #[error(transparent)]
    Offload(#[from] crate::offload::OffloadError),
    /// Residency validation or materialization failed.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
    /// MLX evaluation, synchronization, or transfer failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use safemlx::{Device, DeviceType};
    use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};

    use super::*;
    use crate::{
        offload::CacheEvictionPolicy,
        residency::WeightBinding,
        weight_store::{SafetensorsWeightStore, TensorSelection},
    };

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        let values = [[1i32, 2], [3, 4], [5, 6]]
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            values.iter().enumerate().map(|(expert, bytes)| {
                (
                    format!("expert.{expert}"),
                    TensorView::new(StoredDtype::I32, vec![1, 2], bytes).unwrap(),
                )
            }),
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        (dir, store)
    }

    fn entries() -> Vec<ExpertCatalogEntry> {
        (0..3)
            .map(|expert| {
                let identity = ExpertIdentity::new(2, expert);
                let bindings = [
                    WeightBinding::new(
                        "weight",
                        format!("expert.{expert}"),
                        TensorSelection::Full,
                        8,
                    )
                    .unwrap(),
                    WeightBinding::new(
                        "scale",
                        format!("expert.{expert}"),
                        TensorSelection::Full,
                        8,
                    )
                    .unwrap(),
                ];
                let unit = OffloadUnit::new(identity.unit_id(), bindings).unwrap();
                ExpertCatalogEntry::new(identity, unit, 16).unwrap()
            })
            .collect()
    }

    fn cache(
        store: Arc<SafetensorsWeightStore>,
        device: u64,
        host: u64,
        scratch: u64,
        eviction: CacheEvictionPolicy,
    ) -> ExpertCache {
        let experts = OffloadConfig::new(Some(device), Some(host), 1)
            .unwrap()
            .with_eviction_policy(eviction);
        ExpertCache::new(
            store,
            entries(),
            ExpertCacheLoadOptions::new(LayerwiseLoadOptions::default(), experts, scratch).unwrap(),
            stream(),
            stream(),
        )
        .unwrap()
    }

    #[test]
    fn coalesces_routes_in_global_order_and_separates_pass_counters() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 32, 32, CacheEvictionPolicy::LeastRecentlyUsed);
        let first = cache
            .acquire_route_slice(2, &[2, 0, 2, 0], &[2, 2], ExpertPass::Prefill, &stream())
            .unwrap();
        assert_eq!(
            first.identities(),
            &[ExpertIdentity::new(2, 0), ExpertIdentity::new(2, 2)]
        );
        assert_eq!(first.demand(), &[2, 2]);
        assert_eq!(
            first
                .compact_routes()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 0, 1, 0]
        );
        drop(first);

        let second = cache
            .acquire_route_slice(2, &[0, 2], &[1, 2], ExpertPass::Decode, &stream())
            .unwrap();
        drop(second);
        let report = cache.report().unwrap();
        assert_eq!(report.prefill.requested_routes, 4);
        assert_eq!(report.prefill.distinct_experts, 2);
        assert_eq!(report.prefill.coalesced_duplicates, 2);
        assert_eq!(report.prefill.device.misses, 2);
        assert_eq!(report.decode.requested_routes, 2);
        assert_eq!(report.decode.device.hits, 2);
        assert_eq!(report.owned_experts, 3);
        assert_eq!(report.owned_bytes, 48);
    }

    #[test]
    fn rejects_invalid_missing_and_over_scratch_routes_before_loading() {
        let (_dir, store) = fixture();
        let cache = cache(store, 48, 0, 16, CacheEvictionPolicy::LeastRecentlyUsed);
        assert!(matches!(
            cache.acquire_route_slice(2, &[-1], &[1], ExpertPass::Decode, &stream()),
            Err(ExpertCacheError::InvalidExpertId { .. })
        ));
        assert!(matches!(
            cache.acquire_route_slice(2, &[3], &[1], ExpertPass::Decode, &stream()),
            Err(ExpertCacheError::MissingOwnedExpert { .. })
        ));
        assert!(matches!(
            cache.acquire_route_slice(2, &[0, 1], &[2], ExpertPass::Prefill, &stream()),
            Err(ExpertCacheError::ScratchLimitExceeded { .. })
        ));
        let report = cache.report().unwrap();
        assert_eq!(report.device_resident_experts, 0);
        assert_eq!(report.prefill.requested_routes, 0);
        assert_eq!(report.decode.requested_routes, 0);
    }

    #[test]
    fn empty_routes_do_not_materialize_or_build_a_bank() {
        let (_dir, store) = fixture();
        let cache = cache(store, 16, 16, 16, CacheEvictionPolicy::LeastRecentlyUsed);
        let acquired = cache
            .acquire_route_slice(2, &[], &[0, 2], ExpertPass::Decode, &stream())
            .unwrap();
        assert!(acquired.is_empty());
        assert_eq!(acquired.scratch_bytes(), 0);
        assert_eq!(acquired.compact_routes().shape(), &[0, 2]);
        drop(acquired);
        let report = cache.report().unwrap();
        assert_eq!(report.host_resident_experts, 0);
        assert_eq!(report.device_resident_experts, 0);
        assert_eq!(report.decode.compact_banks, 0);
    }

    #[test]
    fn lfu_uses_duplicate_route_demand_and_deterministic_recency_ties() {
        let (_dir, store) = fixture();
        let cache = cache(store, 32, 0, 32, CacheEvictionPolicy::LeastFrequentlyUsed);
        drop(
            cache
                .acquire_route_slice(2, &[0, 0, 0], &[3], ExpertPass::Decode, &stream())
                .unwrap(),
        );
        drop(
            cache
                .acquire_route_slice(2, &[1], &[1], ExpertPass::Decode, &stream())
                .unwrap(),
        );
        drop(
            cache
                .acquire_route_slice(2, &[2], &[1], ExpertPass::Decode, &stream())
                .unwrap(),
        );
        let report = cache.report().unwrap();
        let resident = report
            .residency
            .units()
            .iter()
            .filter(|unit| unit.device_resident())
            .map(|unit| unit.id().as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            resident,
            vec![
                ExpertIdentity::new(2, 0).unit_id().as_str(),
                ExpertIdentity::new(2, 2).unit_id().as_str()
            ]
        );
        assert_eq!(report.decode.device.evictions, 1);
    }
}
