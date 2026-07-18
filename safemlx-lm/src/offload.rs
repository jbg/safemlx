//! Architecture-independent contracts and telemetry for weight offload.
//!
//! Placement determines which tensors a rank owns. The types in this module
//! describe a separate residency decision: the tier in which an owned logical
//! unit is intended to reside and its lifetime policy. This module validates
//! explicit plans and records observations. The architecture-independent
//! executor lives in [`crate::residency`].
//!
//! The vendored MLX C API does not expose stream events or fences. Residency
//! execution therefore uses conservative
//! [`safemlx::Stream::synchronize`] boundaries. The transfer telemetry is
//! independent of that implementation so event-backed coordination can be
//! introduced later without changing this public API.

use std::{fmt, time::Duration};

/// A storage or execution-memory tier used by an offload plan.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryTier {
    /// Memory directly used for device execution.
    Device,
    /// Host-accessible memory.
    Host,
    /// Disk-backed storage.
    Disk,
}

impl MemoryTier {
    const fn index(self) -> usize {
        match self {
            Self::Device => 0,
            Self::Host => 1,
            Self::Disk => 2,
        }
    }
}

/// The intended lifetime behavior of an offload unit within a tier.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResidencyPolicy {
    /// Keep the unit resident for the lifetime of the residency manager.
    Pinned,
    /// Keep the unit resident only within a bounded execution window.
    Windowed,
    /// Allow the residency manager to retain or evict the unit as cache policy permits.
    Cacheable,
}

/// Deterministic eviction ordering for cacheable residency units.
#[derive(Debug, Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CacheEvictionPolicy {
    /// Evict the least recently used cacheable copy first.
    #[default]
    LeastRecentlyUsed,
    /// Evict the least frequently used copy, using recency and unit id as ties.
    LeastFrequentlyUsed,
}

/// A stable logical identifier for one independently managed offload unit.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OffloadUnitId(String);

impl OffloadUnitId {
    /// Creates an identifier from a non-empty string.
    pub fn new(id: impl Into<String>) -> Result<Self, OffloadError> {
        let id = id.into();
        if id.trim().is_empty() {
            Err(OffloadError::EmptyUnitId)
        } else {
            Ok(Self(id))
        }
    }

    /// Returns the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for OffloadUnitId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for OffloadUnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Global limits and lookahead used when validating an explicit offload plan.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct OffloadConfig {
    device_budget_bytes: Option<u64>,
    host_budget_bytes: Option<u64>,
    prefetch_depth: usize,
    eviction_policy: CacheEvictionPolicy,
}

impl OffloadConfig {
    /// Creates a configuration with optional finite device and host budgets.
    ///
    /// A zero-byte budget is meaningful and forbids assigning non-empty units
    /// to that tier. `prefetch_depth` must be nonzero.
    pub fn new(
        device_budget_bytes: Option<u64>,
        host_budget_bytes: Option<u64>,
        prefetch_depth: usize,
    ) -> Result<Self, OffloadError> {
        if prefetch_depth == 0 {
            return Err(OffloadError::ZeroPrefetchDepth);
        }
        Ok(Self {
            device_budget_bytes,
            host_budget_bytes,
            prefetch_depth,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
        })
    }

    /// Returns the finite device-tier budget, if configured.
    pub const fn device_budget_bytes(self) -> Option<u64> {
        self.device_budget_bytes
    }

    /// Returns the finite host-tier budget, if configured.
    pub const fn host_budget_bytes(self) -> Option<u64> {
        self.host_budget_bytes
    }

    /// Returns the number of logical units the executor may prefetch ahead.
    pub const fn prefetch_depth(self) -> usize {
        self.prefetch_depth
    }

    /// Selects deterministic cache eviction without changing tier budgets.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Returns the configured cache eviction ordering.
    pub const fn eviction_policy(self) -> CacheEvictionPolicy {
        self.eviction_policy
    }
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            device_budget_bytes: None,
            host_budget_bytes: None,
            prefetch_depth: 1,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
        }
    }
}

/// One explicit logical unit assignment in an offload plan.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadUnitSpec {
    id: OffloadUnitId,
    bytes: u64,
    policy: ResidencyPolicy,
    tier: MemoryTier,
}

impl OffloadUnitSpec {
    /// Creates and validates one explicit assignment.
    pub fn new(
        id: OffloadUnitId,
        bytes: u64,
        policy: ResidencyPolicy,
        tier: MemoryTier,
    ) -> Result<Self, OffloadError> {
        if bytes == 0 {
            return Err(OffloadError::ZeroSizedUnit { id });
        }
        if policy == ResidencyPolicy::Pinned && tier == MemoryTier::Disk {
            return Err(OffloadError::ContradictoryAssignment {
                id,
                policy,
                tier,
                reason: "pinned units must be assigned to a resident memory tier",
            });
        }
        Ok(Self {
            id,
            bytes,
            policy,
            tier,
        })
    }

    /// Returns the logical unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the planned unit size in bytes.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the planned residency policy.
    pub const fn policy(&self) -> ResidencyPolicy {
        self.policy
    }

    /// Returns the explicitly assigned tier.
    pub const fn tier(&self) -> MemoryTier {
        self.tier
    }
}

/// Byte totals indexed by memory tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct TierByteTotals {
    device: u64,
    host: u64,
    disk: u64,
}

/// Current or peak logical resident-unit counts by tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct TierUnitTotals {
    device: usize,
    host: usize,
    disk: usize,
}

impl TierUnitTotals {
    /// Creates explicit device, host, and disk unit totals.
    pub const fn new(device: usize, host: usize, disk: usize) -> Self {
        Self { device, host, disk }
    }

    /// Returns the unit total for `tier`.
    pub const fn get(self, tier: MemoryTier) -> usize {
        match tier {
            MemoryTier::Device => self.device,
            MemoryTier::Host => self.host,
            MemoryTier::Disk => self.disk,
        }
    }

    fn set(&mut self, tier: MemoryTier, units: usize) {
        match tier {
            MemoryTier::Device => self.device = units,
            MemoryTier::Host => self.host = units,
            MemoryTier::Disk => self.disk = units,
        }
    }
}

impl TierByteTotals {
    /// Creates explicit device, host, and disk byte totals.
    pub const fn new(device: u64, host: u64, disk: u64) -> Self {
        Self { device, host, disk }
    }

    /// Returns the byte total for `tier`.
    pub const fn get(self, tier: MemoryTier) -> u64 {
        match tier {
            MemoryTier::Device => self.device,
            MemoryTier::Host => self.host,
            MemoryTier::Disk => self.disk,
        }
    }

    fn set(&mut self, tier: MemoryTier, bytes: u64) {
        match tier {
            MemoryTier::Device => self.device = bytes,
            MemoryTier::Host => self.host = bytes,
            MemoryTier::Disk => self.disk = bytes,
        }
    }

    fn checked_add(&mut self, tier: MemoryTier, bytes: u64) -> Result<(), OffloadError> {
        let total = self
            .get(tier)
            .checked_add(bytes)
            .ok_or(OffloadError::ByteTotalOverflow { tier })?;
        self.set(tier, total);
        Ok(())
    }
}

/// A deterministic, validated explicit offload plan.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadPlan {
    config: OffloadConfig,
    units: Vec<OffloadUnitSpec>,
    planned_bytes: TierByteTotals,
}

impl OffloadPlan {
    /// Validates explicit assignments and sorts them by logical identifier.
    ///
    /// This constructor does not materialize tensors or choose assignments.
    pub fn new(
        config: OffloadConfig,
        units: impl IntoIterator<Item = OffloadUnitSpec>,
    ) -> Result<Self, OffloadError> {
        let mut units = units.into_iter().collect::<Vec<_>>();
        units.sort_by(|left, right| left.id.cmp(&right.id));

        if let Some(pair) = units.windows(2).find(|pair| pair[0].id == pair[1].id) {
            return Err(OffloadError::DuplicateUnitId {
                id: pair[0].id.clone(),
            });
        }

        let mut planned_bytes = TierByteTotals::default();
        for unit in &units {
            planned_bytes.checked_add(unit.tier, unit.bytes)?;
        }

        validate_budget(
            MemoryTier::Device,
            planned_bytes.device,
            config.device_budget_bytes,
        )?;
        validate_budget(
            MemoryTier::Host,
            planned_bytes.host,
            config.host_budget_bytes,
        )?;

        Ok(Self {
            config,
            units,
            planned_bytes,
        })
    }

    /// Returns the configuration used to validate this plan.
    pub const fn config(&self) -> OffloadConfig {
        self.config
    }

    /// Returns assignments in stable logical-identifier order.
    pub fn units(&self) -> &[OffloadUnitSpec] {
        &self.units
    }

    /// Looks up a unit by its logical identifier.
    pub fn unit(&self, id: &OffloadUnitId) -> Option<&OffloadUnitSpec> {
        self.units
            .binary_search_by(|unit| unit.id.cmp(id))
            .ok()
            .map(|index| &self.units[index])
    }

    /// Returns checked planned byte totals for every tier.
    pub const fn planned_bytes(&self) -> TierByteTotals {
        self.planned_bytes
    }
}

fn validate_budget(
    tier: MemoryTier,
    planned_bytes: u64,
    budget_bytes: Option<u64>,
) -> Result<(), OffloadError> {
    if let Some(budget_bytes) = budget_bytes {
        if planned_bytes > budget_bytes {
            return Err(OffloadError::BudgetExceeded {
                tier,
                planned_bytes,
                budget_bytes,
            });
        }
    }
    Ok(())
}

/// Structured validation failures for offload contracts.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum OffloadError {
    /// A logical identifier was empty or whitespace-only.
    #[error("offload unit identifiers must not be empty")]
    EmptyUnitId,
    /// A unit had no bytes to manage.
    #[error("offload unit {id} must contain at least one byte")]
    ZeroSizedUnit {
        /// The invalid unit identifier.
        id: OffloadUnitId,
    },
    /// More than one unit used the same stable identifier.
    #[error("duplicate offload unit identifier: {id}")]
    DuplicateUnitId {
        /// The duplicated identifier.
        id: OffloadUnitId,
    },
    /// Summing unit sizes overflowed the stable byte counter.
    #[error("planned byte total overflowed for the {tier:?} tier")]
    ByteTotalOverflow {
        /// The tier whose total overflowed.
        tier: MemoryTier,
    },
    /// Explicit assignments exceeded a configured finite budget.
    #[error(
        "planned {planned_bytes} bytes for the {tier:?} tier exceed its {budget_bytes}-byte budget"
    )]
    BudgetExceeded {
        /// The over-budget tier.
        tier: MemoryTier,
        /// The checked planned total.
        planned_bytes: u64,
        /// The configured finite budget.
        budget_bytes: u64,
    },
    /// A policy and tier assignment had incompatible meanings.
    #[error("offload unit {id} has contradictory {policy:?}/{tier:?} assignment: {reason}")]
    ContradictoryAssignment {
        /// The invalid unit identifier.
        id: OffloadUnitId,
        /// The requested policy.
        policy: ResidencyPolicy,
        /// The requested tier.
        tier: MemoryTier,
        /// A stable explanation of the contradiction.
        reason: &'static str,
    },
    /// Prefetching was configured with a meaningless zero-unit lookahead.
    #[error("offload prefetch depth must be nonzero")]
    ZeroPrefetchDepth,
}

/// A strongly typed transfer direction between two distinct tiers.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TransferDirection {
    /// Device memory to host memory.
    DeviceToHost,
    /// Device memory to disk.
    DeviceToDisk,
    /// Host memory to device memory.
    HostToDevice,
    /// Host memory to disk.
    HostToDisk,
    /// Disk to device memory.
    DiskToDevice,
    /// Disk to host memory.
    DiskToHost,
}

impl TransferDirection {
    /// All directions in stable reporting order.
    pub const ALL: [Self; 6] = [
        Self::DeviceToHost,
        Self::DeviceToDisk,
        Self::HostToDevice,
        Self::HostToDisk,
        Self::DiskToDevice,
        Self::DiskToHost,
    ];

    /// Returns the source tier.
    pub const fn source(self) -> MemoryTier {
        match self {
            Self::DeviceToHost | Self::DeviceToDisk => MemoryTier::Device,
            Self::HostToDevice | Self::HostToDisk => MemoryTier::Host,
            Self::DiskToDevice | Self::DiskToHost => MemoryTier::Disk,
        }
    }

    /// Returns the destination tier.
    pub const fn destination(self) -> MemoryTier {
        match self {
            Self::DeviceToHost | Self::DiskToHost => MemoryTier::Host,
            Self::DeviceToDisk | Self::HostToDisk => MemoryTier::Disk,
            Self::HostToDevice | Self::DiskToDevice => MemoryTier::Device,
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::DeviceToHost => 0,
            Self::DeviceToDisk => 1,
            Self::HostToDevice => 2,
            Self::HostToDisk => 3,
            Self::DiskToDevice => 4,
            Self::DiskToHost => 5,
        }
    }
}

/// Accumulated transfer observations for one direction.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct TransferMetrics {
    count: u64,
    bytes: u64,
    duration: Duration,
}

impl TransferMetrics {
    /// Returns the number of recorded transfers.
    pub const fn count(self) -> u64 {
        self.count
    }

    /// Returns the number of recorded bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Returns the accumulated transfer duration.
    pub const fn duration(self) -> Duration {
        self.duration
    }

    fn record(&mut self, bytes: u64, duration: Duration) {
        self.count = self.count.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
        self.duration = self.duration.saturating_add(duration);
    }
}

/// The result of one completed prefetch request.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefetchOutcome {
    /// The requested unit was already available at the required tier.
    Hit,
    /// The request required a transfer or load.
    Miss,
}

/// Accumulated prefetch observations.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct PrefetchMetrics {
    requests: u64,
    hits: u64,
    misses: u64,
    stalls: u64,
    stall_duration: Duration,
}

impl PrefetchMetrics {
    /// Returns the number of completed prefetch requests.
    pub const fn requests(self) -> u64 {
        self.requests
    }

    /// Returns the number of prefetch hits.
    pub const fn hits(self) -> u64 {
        self.hits
    }

    /// Returns the number of prefetch misses.
    pub const fn misses(self) -> u64 {
        self.misses
    }

    /// Returns the number of demand waits attributed to prefetching.
    pub const fn stalls(self) -> u64 {
        self.stalls
    }

    /// Returns the accumulated prefetch stall duration.
    pub const fn stall_duration(self) -> Duration {
        self.stall_duration
    }
}

/// Accumulated eviction observations.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct EvictionMetrics {
    count: u64,
    bytes: u64,
}

impl EvictionMetrics {
    /// Returns the number of recorded evictions.
    pub const fn count(self) -> u64 {
        self.count
    }

    /// Returns the number of recorded evicted bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// A point-in-time sample of MLX-managed allocator memory.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MlxMemoryMetrics {
    active_bytes: u64,
    cached_bytes: u64,
    peak_bytes: u64,
}

impl MlxMemoryMetrics {
    /// Creates an explicit MLX allocator sample.
    pub const fn new(active_bytes: u64, cached_bytes: u64, peak_bytes: u64) -> Self {
        Self {
            active_bytes,
            cached_bytes,
            peak_bytes,
        }
    }

    /// Returns active MLX-managed bytes.
    pub const fn active_bytes(self) -> u64 {
        self.active_bytes
    }

    /// Returns bytes retained by the MLX allocator cache.
    pub const fn cached_bytes(self) -> u64 {
        self.cached_bytes
    }

    /// Returns peak active MLX-managed bytes.
    pub const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }
}

/// Optional process-level memory and page-fault observations.
///
/// Individual values are absent when they cannot be obtained safely on the
/// current platform. The built-in sampler currently reads Linux `/proc`; it
/// makes no availability guarantee on Apple or Windows targets.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ProcessMetrics {
    rss_bytes: Option<u64>,
    minor_page_faults: Option<u64>,
    major_page_faults: Option<u64>,
}

impl ProcessMetrics {
    /// Creates an explicit process sample.
    pub const fn new(
        rss_bytes: Option<u64>,
        minor_page_faults: Option<u64>,
        major_page_faults: Option<u64>,
    ) -> Self {
        Self {
            rss_bytes,
            minor_page_faults,
            major_page_faults,
        }
    }

    /// Returns resident-set bytes when available.
    pub const fn rss_bytes(self) -> Option<u64> {
        self.rss_bytes
    }

    /// Returns minor page faults when available.
    pub const fn minor_page_faults(self) -> Option<u64> {
        self.minor_page_faults
    }

    /// Returns major page faults when available.
    pub const fn major_page_faults(self) -> Option<u64> {
        self.major_page_faults
    }
}

/// Samples optional process metrics without adding a platform runtime dependency.
pub fn sample_process_metrics() -> ProcessMetrics {
    platform_process_metrics()
}

#[cfg(target_os = "linux")]
fn platform_process_metrics() -> ProcessMetrics {
    let rss_bytes = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let value = line.strip_prefix("VmRSS:")?.trim();
                let kibibytes = value.strip_suffix("kB")?.trim().parse::<u64>().ok()?;
                kibibytes.checked_mul(1024)
            })
        });

    let faults = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            // The parenthesized command name may contain spaces. Fields after
            // its final ')' begin at process-stat field 3 (state).
            let fields = stat.get(stat.rfind(')')? + 1..)?.split_whitespace();
            let fields = fields.collect::<Vec<_>>();
            Some((fields.get(7)?.parse().ok()?, fields.get(9)?.parse().ok()?))
        });

    ProcessMetrics::new(
        rss_bytes,
        faults.map(|value| value.0),
        faults.map(|value| value.1),
    )
}

#[cfg(not(target_os = "linux"))]
fn platform_process_metrics() -> ProcessMetrics {
    ProcessMetrics::default()
}

/// Mutable, single-threaded offload telemetry collector.
///
/// Updates use saturating arithmetic for monotonic counters and durations.
/// Resident bytes are set explicitly, and setting a new value updates the
/// corresponding peak. Wrap this value in a mutex if multiple threads need to
/// record into one collector.
#[derive(Debug, Default, Clone)]
pub struct OffloadTelemetry {
    planned_bytes: TierByteTotals,
    resident_bytes: TierByteTotals,
    peak_resident_bytes: TierByteTotals,
    resident_units: TierUnitTotals,
    peak_resident_units: TierUnitTotals,
    transfers: [TransferMetrics; 6],
    prefetch: PrefetchMetrics,
    tier_prefetch: [PrefetchMetrics; 3],
    evictions: EvictionMetrics,
    tier_evictions: [EvictionMetrics; 3],
    mlx_memory: Option<MlxMemoryMetrics>,
    process: ProcessMetrics,
    process_sampled: bool,
}

impl OffloadTelemetry {
    /// Creates a collector initialized with a validated plan's byte totals.
    pub fn from_plan(plan: &OffloadPlan) -> Self {
        Self {
            planned_bytes: plan.planned_bytes,
            ..Self::default()
        }
    }

    /// Replaces the planned byte totals recorded by this collector.
    pub fn set_planned_bytes(&mut self, planned_bytes: TierByteTotals) {
        self.planned_bytes = planned_bytes;
    }

    /// Sets current resident bytes and updates the peak for `tier`.
    pub fn set_resident_bytes(&mut self, tier: MemoryTier, bytes: u64) {
        self.resident_bytes.set(tier, bytes);
        if bytes > self.peak_resident_bytes.get(tier) {
            self.peak_resident_bytes.set(tier, bytes);
        }
    }

    /// Sets current resident units and updates the peak for `tier`.
    pub fn set_resident_units(&mut self, tier: MemoryTier, units: usize) {
        self.resident_units.set(tier, units);
        if units > self.peak_resident_units.get(tier) {
            self.peak_resident_units.set(tier, units);
        }
    }

    /// Records one completed transfer using saturating counter updates.
    pub fn record_transfer(
        &mut self,
        direction: TransferDirection,
        bytes: u64,
        duration: Duration,
    ) {
        self.transfers[direction.index()].record(bytes, duration);
    }

    /// Records one completed prefetch request and its outcome.
    pub fn record_prefetch(&mut self, outcome: PrefetchOutcome) {
        self.prefetch.requests = self.prefetch.requests.saturating_add(1);
        match outcome {
            PrefetchOutcome::Hit => {
                self.prefetch.hits = self.prefetch.hits.saturating_add(1);
            }
            PrefetchOutcome::Miss => {
                self.prefetch.misses = self.prefetch.misses.saturating_add(1);
            }
        }
    }

    /// Records a cache request both globally and for its target tier.
    pub fn record_tier_prefetch(&mut self, tier: MemoryTier, outcome: PrefetchOutcome) {
        self.record_prefetch(outcome);
        let metrics = &mut self.tier_prefetch[tier.index()];
        metrics.requests = metrics.requests.saturating_add(1);
        match outcome {
            PrefetchOutcome::Hit => metrics.hits = metrics.hits.saturating_add(1),
            PrefetchOutcome::Miss => metrics.misses = metrics.misses.saturating_add(1),
        }
    }

    /// Records a demand stall while waiting for a prefetched unit.
    pub fn record_prefetch_stall(&mut self, duration: Duration) {
        self.prefetch.stalls = self.prefetch.stalls.saturating_add(1);
        self.prefetch.stall_duration = self.prefetch.stall_duration.saturating_add(duration);
    }

    /// Records one eviction using saturating counter updates.
    pub fn record_eviction(&mut self, bytes: u64) {
        self.evictions.count = self.evictions.count.saturating_add(1);
        self.evictions.bytes = self.evictions.bytes.saturating_add(bytes);
    }

    /// Records an eviction both globally and for its source tier.
    pub fn record_tier_eviction(&mut self, tier: MemoryTier, bytes: u64) {
        self.record_eviction(bytes);
        let metrics = &mut self.tier_evictions[tier.index()];
        metrics.count = metrics.count.saturating_add(1);
        metrics.bytes = metrics.bytes.saturating_add(bytes);
    }

    /// Records an externally obtained MLX allocator sample.
    pub fn record_mlx_memory(&mut self, metrics: MlxMemoryMetrics) {
        self.mlx_memory = Some(metrics);
    }

    /// Samples active, cached, and peak MLX-managed allocator bytes.
    pub fn sample_mlx_memory(&mut self) -> Result<(), safemlx::error::Exception> {
        let active = safemlx::memory::active_memory()?;
        let cached = safemlx::memory::cache_memory()?;
        let peak = safemlx::memory::peak_memory()?;
        self.mlx_memory = Some(MlxMemoryMetrics::new(
            u64::try_from(active).unwrap_or(u64::MAX),
            u64::try_from(cached).unwrap_or(u64::MAX),
            u64::try_from(peak).unwrap_or(u64::MAX),
        ));
        Ok(())
    }

    /// Records an externally obtained process sample.
    pub fn record_process_metrics(&mut self, metrics: ProcessMetrics) {
        self.process = metrics;
        self.process_sampled = true;
    }

    /// Updates process observations using the built-in optional sampler.
    pub fn sample_process_metrics(&mut self) {
        self.process = sample_process_metrics();
        self.process_sampled = true;
    }

    /// Returns an immutable point-in-time report.
    pub fn snapshot(&self) -> OffloadReport {
        OffloadReport {
            planned_bytes: self.planned_bytes,
            resident_bytes: self.resident_bytes,
            peak_resident_bytes: self.peak_resident_bytes,
            resident_units: self.resident_units,
            peak_resident_units: self.peak_resident_units,
            transfers: self.transfers,
            prefetch: self.prefetch,
            tier_prefetch: self.tier_prefetch,
            evictions: self.evictions,
            tier_evictions: self.tier_evictions,
            mlx_memory: self.mlx_memory,
            process: self.process,
            process_sampled: self.process_sampled,
        }
    }

    /// Clears all configuration and observations, including resident peaks.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Immutable point-in-time offload telemetry report.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadReport {
    planned_bytes: TierByteTotals,
    resident_bytes: TierByteTotals,
    peak_resident_bytes: TierByteTotals,
    resident_units: TierUnitTotals,
    peak_resident_units: TierUnitTotals,
    transfers: [TransferMetrics; 6],
    prefetch: PrefetchMetrics,
    tier_prefetch: [PrefetchMetrics; 3],
    evictions: EvictionMetrics,
    tier_evictions: [EvictionMetrics; 3],
    mlx_memory: Option<MlxMemoryMetrics>,
    process: ProcessMetrics,
    process_sampled: bool,
}

impl OffloadReport {
    /// Returns planned bytes per tier.
    pub const fn planned_bytes(&self) -> TierByteTotals {
        self.planned_bytes
    }

    /// Returns current resident bytes per tier.
    pub const fn resident_bytes(&self) -> TierByteTotals {
        self.resident_bytes
    }

    /// Returns peak resident bytes per tier.
    pub const fn peak_resident_bytes(&self) -> TierByteTotals {
        self.peak_resident_bytes
    }

    /// Returns current resident-unit counts per tier.
    pub const fn resident_units(&self) -> TierUnitTotals {
        self.resident_units
    }

    /// Returns peak resident-unit counts per tier.
    pub const fn peak_resident_units(&self) -> TierUnitTotals {
        self.peak_resident_units
    }

    /// Returns accumulated metrics for one transfer direction.
    pub const fn transfer(&self, direction: TransferDirection) -> TransferMetrics {
        self.transfers[direction.index()]
    }

    /// Returns accumulated prefetch metrics.
    pub const fn prefetch(&self) -> PrefetchMetrics {
        self.prefetch
    }

    /// Returns cache request metrics for one target tier.
    pub const fn tier_prefetch(&self, tier: MemoryTier) -> PrefetchMetrics {
        self.tier_prefetch[tier.index()]
    }

    /// Returns accumulated eviction metrics.
    pub const fn evictions(&self) -> EvictionMetrics {
        self.evictions
    }

    /// Returns eviction metrics for one source tier.
    pub const fn tier_evictions(&self, tier: MemoryTier) -> EvictionMetrics {
        self.tier_evictions[tier.index()]
    }

    /// Returns the latest MLX allocator sample, if one was recorded.
    pub const fn mlx_memory(&self) -> Option<MlxMemoryMetrics> {
        self.mlx_memory
    }

    /// Returns the latest optional process sample.
    pub const fn process_metrics(&self) -> ProcessMetrics {
        self.process
    }

    /// Returns whether process sampling was requested, including unsupported platforms.
    pub const fn process_sampled(&self) -> bool {
        self.process_sampled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(id: &str, bytes: u64, policy: ResidencyPolicy, tier: MemoryTier) -> OffloadUnitSpec {
        OffloadUnitSpec::new(OffloadUnitId::new(id).unwrap(), bytes, policy, tier).unwrap()
    }

    #[test]
    fn explicit_plan_is_validated_sorted_and_inspectable() {
        let config = OffloadConfig::new(Some(80), Some(40), 2).unwrap();
        let plan = OffloadPlan::new(
            config,
            [
                unit("layer.2", 20, ResidencyPolicy::Windowed, MemoryTier::Disk),
                unit("layer.0", 40, ResidencyPolicy::Pinned, MemoryTier::Device),
                unit("layer.1", 30, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
        )
        .unwrap();

        assert_eq!(
            plan.units()
                .iter()
                .map(|unit| unit.id().as_str())
                .collect::<Vec<_>>(),
            ["layer.0", "layer.1", "layer.2"]
        );
        assert_eq!(plan.config(), config);
        assert_eq!(plan.planned_bytes(), TierByteTotals::new(40, 30, 20));
        assert_eq!(
            plan.unit(&OffloadUnitId::new("layer.1").unwrap())
                .unwrap()
                .bytes(),
            30
        );
    }

    #[test]
    fn duplicate_identifiers_are_rejected_deterministically() {
        let duplicate = OffloadPlan::new(
            OffloadConfig::default(),
            [
                unit("b", 1, ResidencyPolicy::Cacheable, MemoryTier::Host),
                unit("a", 1, ResidencyPolicy::Pinned, MemoryTier::Device),
                unit("a", 2, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
        )
        .unwrap_err();
        assert_eq!(
            duplicate,
            OffloadError::DuplicateUnitId {
                id: OffloadUnitId::new("a").unwrap()
            }
        );
    }

    #[test]
    fn finite_tier_budgets_are_enforced() {
        let error = OffloadPlan::new(
            OffloadConfig::new(Some(9), None, 1).unwrap(),
            [unit(
                "weights",
                10,
                ResidencyPolicy::Pinned,
                MemoryTier::Device,
            )],
        )
        .unwrap_err();
        assert_eq!(
            error,
            OffloadError::BudgetExceeded {
                tier: MemoryTier::Device,
                planned_bytes: 10,
                budget_bytes: 9,
            }
        );
    }

    #[test]
    fn byte_total_overflow_is_reported() {
        let error = OffloadPlan::new(
            OffloadConfig::default(),
            [
                unit("a", u64::MAX, ResidencyPolicy::Cacheable, MemoryTier::Host),
                unit("b", 1, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            OffloadError::ByteTotalOverflow {
                tier: MemoryTier::Host
            }
        );
    }

    #[test]
    fn meaningless_and_contradictory_inputs_are_rejected() {
        assert_eq!(
            OffloadConfig::new(None, None, 0),
            Err(OffloadError::ZeroPrefetchDepth)
        );
        let id = OffloadUnitId::new("empty").unwrap();
        assert_eq!(
            OffloadUnitSpec::new(id.clone(), 0, ResidencyPolicy::Cacheable, MemoryTier::Host),
            Err(OffloadError::ZeroSizedUnit { id })
        );
        assert!(matches!(
            OffloadUnitSpec::new(
                OffloadUnitId::new("pinned-disk").unwrap(),
                1,
                ResidencyPolicy::Pinned,
                MemoryTier::Disk
            ),
            Err(OffloadError::ContradictoryAssignment { .. })
        ));
    }

    #[test]
    fn telemetry_accounts_for_residency_activity_and_runtime_samples() {
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [unit("a", 10, ResidencyPolicy::Pinned, MemoryTier::Device)],
        )
        .unwrap();
        let mut telemetry = OffloadTelemetry::from_plan(&plan);
        telemetry.set_resident_bytes(MemoryTier::Device, 8);
        telemetry.set_resident_bytes(MemoryTier::Device, 12);
        telemetry.set_resident_bytes(MemoryTier::Device, 6);
        telemetry.set_resident_units(MemoryTier::Device, 1);
        telemetry.set_resident_units(MemoryTier::Device, 2);
        telemetry.set_resident_units(MemoryTier::Device, 1);
        telemetry.record_transfer(TransferDirection::HostToDevice, 5, Duration::from_millis(2));
        telemetry.record_transfer(TransferDirection::HostToDevice, 7, Duration::from_millis(3));
        telemetry.record_tier_prefetch(MemoryTier::Device, PrefetchOutcome::Hit);
        telemetry.record_tier_prefetch(MemoryTier::Host, PrefetchOutcome::Miss);
        telemetry.record_prefetch_stall(Duration::from_millis(4));
        telemetry.record_tier_eviction(MemoryTier::Device, 3);
        telemetry.record_tier_eviction(MemoryTier::Host, 4);
        telemetry.record_mlx_memory(MlxMemoryMetrics::new(11, 12, 13));
        telemetry.record_process_metrics(ProcessMetrics::new(Some(14), Some(15), Some(16)));

        let report = telemetry.snapshot();
        assert_eq!(report.planned_bytes().get(MemoryTier::Device), 10);
        assert_eq!(report.resident_bytes().get(MemoryTier::Device), 6);
        assert_eq!(report.peak_resident_bytes().get(MemoryTier::Device), 12);
        assert_eq!(report.resident_units().get(MemoryTier::Device), 1);
        assert_eq!(report.peak_resident_units().get(MemoryTier::Device), 2);
        assert_eq!(report.transfer(TransferDirection::HostToDevice).count(), 2);
        assert_eq!(report.transfer(TransferDirection::HostToDevice).bytes(), 12);
        assert_eq!(
            report.transfer(TransferDirection::HostToDevice).duration(),
            Duration::from_millis(5)
        );
        assert_eq!(report.prefetch().requests(), 2);
        assert_eq!(report.prefetch().hits(), 1);
        assert_eq!(report.prefetch().misses(), 1);
        assert_eq!(report.prefetch().stalls(), 1);
        assert_eq!(report.prefetch().stall_duration(), Duration::from_millis(4));
        assert_eq!(report.evictions().count(), 2);
        assert_eq!(report.evictions().bytes(), 7);
        assert_eq!(report.tier_prefetch(MemoryTier::Device).hits(), 1);
        assert_eq!(report.tier_prefetch(MemoryTier::Host).misses(), 1);
        assert_eq!(report.tier_evictions(MemoryTier::Device).bytes(), 3);
        assert_eq!(report.tier_evictions(MemoryTier::Host).bytes(), 4);
        assert_eq!(report.mlx_memory().unwrap().peak_bytes(), 13);
        assert_eq!(report.process_metrics().rss_bytes(), Some(14));
        assert!(report.process_sampled());
    }

    #[test]
    fn snapshot_is_immutable_and_reset_clears_everything() {
        let mut telemetry = OffloadTelemetry::default();
        telemetry.set_planned_bytes(TierByteTotals::new(1, 2, 3));
        telemetry.set_resident_bytes(MemoryTier::Host, 4);
        telemetry.set_resident_units(MemoryTier::Host, 1);
        let snapshot = telemetry.snapshot();

        telemetry.set_resident_bytes(MemoryTier::Host, 9);
        telemetry.record_eviction(5);
        assert_eq!(snapshot.resident_bytes().get(MemoryTier::Host), 4);
        assert_eq!(snapshot.resident_units().get(MemoryTier::Host), 1);
        assert_eq!(snapshot.evictions(), EvictionMetrics::default());
        assert!(!snapshot.process_sampled());

        telemetry.reset();
        assert_eq!(telemetry.snapshot(), OffloadTelemetry::default().snapshot());
    }

    #[test]
    fn telemetry_counters_saturate() {
        let mut telemetry = OffloadTelemetry::default();
        telemetry.transfers[TransferDirection::DiskToHost.index()] = TransferMetrics {
            count: u64::MAX,
            bytes: u64::MAX,
            duration: Duration::MAX,
        };
        telemetry.evictions = EvictionMetrics {
            count: u64::MAX,
            bytes: u64::MAX,
        };
        telemetry.record_transfer(TransferDirection::DiskToHost, 1, Duration::from_nanos(1));
        telemetry.record_eviction(1);
        let report = telemetry.snapshot();
        assert_eq!(
            report.transfer(TransferDirection::DiskToHost).count(),
            u64::MAX
        );
        assert_eq!(
            report.transfer(TransferDirection::DiskToHost).bytes(),
            u64::MAX
        );
        assert_eq!(
            report.transfer(TransferDirection::DiskToHost).duration(),
            Duration::MAX
        );
        assert_eq!(report.evictions().count(), u64::MAX);
        assert_eq!(report.evictions().bytes(), u64::MAX);
    }

    #[test]
    fn optional_process_sampler_never_requires_platform_support() {
        let metrics = sample_process_metrics();
        if let Some(rss_bytes) = metrics.rss_bytes() {
            assert!(rss_bytes > 0);
        }
        let mut telemetry = OffloadTelemetry::default();
        telemetry.sample_process_metrics();
        let report = telemetry.snapshot();
        let _ = report.process_metrics();
        assert!(report.process_sampled());
    }
}
