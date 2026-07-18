//! Block-addressable residency for mutable attention state.
//!
//! This module is deliberately independent from weight residency. Attention
//! blocks are mutable activation state until sealed, while checkpoint weights
//! are immutable inputs with a different ownership and persistence model.

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, Mutex, MutexGuard,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use safemlx::{transforms::eval, Array, Device, DeviceType, Dtype, Stream};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::offload::CacheEvictionPolicy;

const PROMPT_CACHE_SCHEMA_VERSION: u32 = 1;
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/// Selects the existing device-resident cache or bounded paged residency.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum CacheResidencyPolicy {
    /// Keep the existing cache representation entirely device resident.
    #[default]
    Device,
    /// Store sealed state in token-addressable blocks under finite budgets.
    Paged(PagedCacheOptions),
}

/// Controls optional disk backing for a live inference cache.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum LiveCacheDiskPolicy {
    /// Do not write live attention state to disk.
    #[default]
    Disabled,
    /// Retain demoted sealed blocks in an explicit ephemeral directory.
    Enabled {
        /// Directory dedicated to this live cache.
        directory: PathBuf,
        /// Finite logical byte limit for live cache files.
        budget_bytes: u64,
        /// Bound on pending reader or writer requests.
        queue_capacity: usize,
    },
}

/// Validated finite limits for a paged attention cache.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PagedCacheOptions {
    block_size_tokens: i32,
    device_budget_bytes: u64,
    host_budget_bytes: u64,
    recent_device_blocks: usize,
    eviction_policy: CacheEvictionPolicy,
    full_attention: bool,
    retain_discarded_for_persistence: bool,
    live_disk: LiveCacheDiskPolicy,
    sample_process: bool,
}

impl PagedCacheOptions {
    /// Creates paged-cache limits. Every memory limit is finite and explicit.
    pub fn new(
        block_size_tokens: i32,
        device_budget_bytes: u64,
        host_budget_bytes: u64,
        recent_device_blocks: usize,
    ) -> Result<Self, CacheResidencyError> {
        if block_size_tokens <= 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "cache block size must be positive".into(),
            ));
        }
        if device_budget_bytes == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "paged cache device budget must be nonzero".into(),
            ));
        }
        if recent_device_blocks == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "paged cache must protect at least one recent device block".into(),
            ));
        }
        Ok(Self {
            block_size_tokens,
            device_budget_bytes,
            host_budget_bytes,
            recent_device_blocks,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
            full_attention: false,
            retain_discarded_for_persistence: false,
            live_disk: LiveCacheDiskPolicy::Disabled,
            sample_process: false,
        })
    }

    /// Enables exact blockwise full-context attention.
    pub const fn with_full_attention(mut self, enabled: bool) -> Self {
        self.full_attention = enabled;
        self
    }

    /// Retains blocks older than a sliding window solely for later persistence.
    pub const fn with_persistence_retention(mut self, enabled: bool) -> Self {
        self.retain_discarded_for_persistence = enabled;
        self
    }

    /// Selects deterministic block eviction ordering.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Configures explicit live disk backing.
    pub fn with_live_disk(
        mut self,
        directory: impl Into<PathBuf>,
        budget_bytes: u64,
        queue_capacity: usize,
    ) -> Result<Self, CacheResidencyError> {
        if budget_bytes == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "live cache disk budget must be nonzero".into(),
            ));
        }
        if queue_capacity == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "live cache disk queue capacity must be nonzero".into(),
            ));
        }
        let directory = directory.into();
        if directory.as_os_str().is_empty() {
            return Err(CacheResidencyError::InvalidOptions(
                "live cache disk directory must not be empty".into(),
            ));
        }
        self.live_disk = LiveCacheDiskPolicy::Enabled {
            directory,
            budget_bytes,
            queue_capacity,
        };
        Ok(self)
    }

    /// Enables optional process-memory sampling in reports.
    pub const fn with_process_sampling(mut self, enabled: bool) -> Self {
        self.sample_process = enabled;
        self
    }

    /// Returns the block size in tokens.
    pub const fn block_size_tokens(&self) -> i32 {
        self.block_size_tokens
    }

    /// Returns the finite logical device-cache budget.
    pub const fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }

    /// Returns the finite logical host-cache budget.
    pub const fn host_budget_bytes(&self) -> u64 {
        self.host_budget_bytes
    }

    /// Returns the recent block count protected on the execution device per layer.
    pub const fn recent_device_blocks(&self) -> usize {
        self.recent_device_blocks
    }

    /// Returns whether exact blockwise full attention is enabled.
    pub const fn full_attention_enabled(&self) -> bool {
        self.full_attention
    }

    /// Returns whether discarded sliding state is retained for persistence.
    pub const fn retains_discarded_for_persistence(&self) -> bool {
        self.retain_discarded_for_persistence
    }

    /// Returns the live disk policy.
    pub const fn live_disk_policy(&self) -> &LiveCacheDiskPolicy {
        &self.live_disk
    }
}

/// Representation stored atomically in one cache block.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheRepresentation {
    /// Standard attention keys and values.
    KeyValue,
    /// DeepSeek compressed latent state and rotary keys.
    CompressedLatentRotary,
}

/// Optional rank identity included in a stable cache block identifier.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CacheRankIdentity {
    /// Pipeline rank, when pipeline partitioning is active.
    pub pipeline_rank: Option<usize>,
    /// Tensor-parallel rank, when cache heads are sharded.
    pub tensor_parallel_rank: Option<usize>,
    /// Expert-parallel rank for replicated attention state.
    pub expert_parallel_rank: Option<usize>,
}

/// Stable identity for one immutable sealed cache block.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CacheBlockId {
    /// Identity shared by every block in one live cache.
    pub session_id: u64,
    /// Architecture-global decoder layer index.
    pub global_layer: usize,
    /// Stored attention representation.
    pub representation: CacheRepresentation,
    /// Inclusive absolute token position.
    pub start: i64,
    /// Exclusive absolute token position.
    pub end: i64,
    /// Rank-local ownership identity.
    pub rank: Option<CacheRankIdentity>,
}

/// Logical location of a sealed cache block.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTier {
    /// Available to execution without a catalog load.
    Device,
    /// Evaluated host-accessible state.
    Host,
    /// Stored in a live or persistent safetensors shard.
    Disk,
}

/// Lifecycle state visible through cache diagnostics.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CacheBlockLifecycle {
    /// A layer-owned append target that has not been sealed.
    MutableDeviceTail,
    /// Immutable state kept on the execution device.
    SealedDevice,
    /// Immutable evaluated host state.
    SealedHost,
    /// Immutable disk-backed state.
    DiskBacked,
    /// A transfer is being completed synchronously.
    InFlight,
    /// State removed from live attention and persistence retention.
    Discarded,
    /// Read-only state cataloged from a prompt cache.
    ImportedReadOnly,
}

#[derive(Debug, Clone)]
pub(crate) enum CacheBlockArrays {
    KeyValue { keys: Array, values: Array },
    CompressedLatentRotary { latent: Array, rotary_key: Array },
}

impl CacheBlockArrays {
    pub(crate) fn representation(&self) -> CacheRepresentation {
        match self {
            Self::KeyValue { .. } => CacheRepresentation::KeyValue,
            Self::CompressedLatentRotary { .. } => CacheRepresentation::CompressedLatentRotary,
        }
    }

    fn arrays(&self) -> [&Array; 2] {
        match self {
            Self::KeyValue { keys, values } => [keys, values],
            Self::CompressedLatentRotary { latent, rotary_key } => [latent, rotary_key],
        }
    }

    fn bytes(&self) -> u64 {
        self.arrays()
            .iter()
            .map(|array| array.nbytes() as u64)
            .sum()
    }

    fn shapes(&self) -> [Vec<i32>; 2] {
        let arrays = self.arrays();
        [arrays[0].shape().to_vec(), arrays[1].shape().to_vec()]
    }

    fn dtypes(&self) -> [String; 2] {
        let arrays = self.arrays();
        [dtype_name(arrays[0].dtype()), dtype_name(arrays[1].dtype())]
    }
}

#[derive(Debug, Clone)]
struct DiskLocation {
    path: PathBuf,
    first_name: String,
    second_name: String,
    persistent: bool,
}

enum DiskRequest {
    Write {
        directory: PathBuf,
        id: CacheBlockId,
        arrays: CacheBlockArrays,
        response: mpsc::Sender<Result<DiskLocation, CacheResidencyError>>,
    },
    Read {
        location: DiskLocation,
        representation: CacheRepresentation,
        response: mpsc::Sender<Result<CacheBlockArrays, CacheResidencyError>>,
    },
    Stop,
}

#[derive(Debug)]
struct DiskWorker {
    sender: SyncSender<DiskRequest>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl DiskWorker {
    fn new(capacity: usize) -> Result<Self, CacheResidencyError> {
        let (sender, receiver) = mpsc::sync_channel::<DiskRequest>(capacity);
        let handle = thread::Builder::new()
            .name("safemlx-cache-disk".into())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    match request {
                        DiskRequest::Write {
                            directory,
                            id,
                            arrays,
                            response,
                        } => {
                            let _ = response.send(write_live_block(&directory, &id, &arrays));
                        }
                        DiskRequest::Read {
                            location,
                            representation,
                            response,
                        } => {
                            let _ =
                                response.send(load_block_arrays_direct(&location, representation));
                        }
                        DiskRequest::Stop => break,
                    }
                }
            })
            .map_err(|source| CacheResidencyError::Io {
                action: "start live cache disk worker",
                path: PathBuf::from("safemlx-cache-disk"),
                source,
            })?;
        Ok(Self {
            sender,
            handle: Mutex::new(Some(handle)),
        })
    }

    fn request<T>(
        &self,
        request: DiskRequest,
        response: mpsc::Receiver<Result<T, CacheResidencyError>>,
    ) -> Result<(T, bool), CacheResidencyError> {
        let backpressure = match self.sender.try_send(request) {
            Ok(()) => false,
            Err(TrySendError::Full(request)) => {
                self.sender.send(request).map_err(|_| {
                    CacheResidencyError::Runtime("live cache disk worker stopped".into())
                })?;
                true
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(CacheResidencyError::Runtime(
                    "live cache disk worker stopped".into(),
                ))
            }
        };
        let value = response.recv().map_err(|_| {
            CacheResidencyError::Runtime("live cache disk worker dropped a response".into())
        })??;
        Ok((value, backpressure))
    }

    fn write(
        &self,
        directory: &Path,
        id: &CacheBlockId,
        arrays: &CacheBlockArrays,
    ) -> Result<(DiskLocation, bool), CacheResidencyError> {
        let (sender, receiver) = mpsc::channel();
        self.request(
            DiskRequest::Write {
                directory: directory.to_path_buf(),
                id: id.clone(),
                arrays: arrays.clone(),
                response: sender,
            },
            receiver,
        )
    }

    fn read(
        &self,
        location: &DiskLocation,
        representation: CacheRepresentation,
    ) -> Result<(CacheBlockArrays, bool), CacheResidencyError> {
        let (sender, receiver) = mpsc::channel();
        self.request(
            DiskRequest::Read {
                location: location.clone(),
                representation,
                response: sender,
            },
            receiver,
        )
    }
}

impl Drop for DiskWorker {
    fn drop(&mut self) {
        let _ = self.sender.send(DiskRequest::Stop);
        if let Ok(handle) = self.handle.get_mut() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CacheBlockRecord {
    id: CacheBlockId,
    tier: CacheTier,
    arrays: Option<CacheBlockArrays>,
    disk: Option<DiskLocation>,
    bytes: u64,
    shapes: [Vec<i32>; 2],
    dtypes: [String; 2],
    imported: bool,
    leases: usize,
    access_count: u64,
    last_access: u64,
    protected_prefix: bool,
}

/// Aggregated logical residency and transfer observations.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheResidencyReport {
    /// Absolute token count represented by the longest layer.
    pub logical_cached_tokens: u64,
    /// Sealed key/value blocks.
    pub key_value_blocks: u64,
    /// Sealed compressed-latent/rotary blocks.
    pub compressed_latent_blocks: u64,
    /// Blocks cataloged on the execution device.
    pub device_blocks: u64,
    /// Blocks cataloged in host memory.
    pub host_blocks: u64,
    /// Blocks cataloged on disk.
    pub disk_blocks: u64,
    /// Current logical device bytes, including mutable tails.
    pub current_device_bytes: u64,
    /// Peak logical device bytes.
    pub peak_device_bytes: u64,
    /// Current logical host bytes.
    pub current_host_bytes: u64,
    /// Peak logical host bytes.
    pub peak_host_bytes: u64,
    /// Current logical disk bytes.
    pub current_disk_bytes: u64,
    /// Peak logical disk bytes.
    pub peak_disk_bytes: u64,
    /// Current bytes in mutable tails.
    pub mutable_tail_bytes: u64,
    /// Recent blocks protected from device demotion.
    pub protected_recent_blocks: u64,
    /// Prefix or sink blocks protected for attention.
    pub protected_prefix_blocks: u64,
    /// Host-to-device logical promotions.
    pub host_promotions: u64,
    /// Disk-to-device logical promotions.
    pub disk_promotions: u64,
    /// Device-to-host logical demotions.
    pub host_demotions: u64,
    /// Host-to-disk logical demotions.
    pub disk_demotions: u64,
    /// Logical bytes named by promotion and demotion operations.
    pub transfer_bytes: u64,
    /// Time inference waited for cache transfers.
    pub transfer_wait: Duration,
    /// Blocks evicted because all configured tiers were exhausted.
    pub evictions: u64,
    /// Sliding-window blocks discarded as semantically invisible.
    pub discarded_sliding_blocks: u64,
    /// Completed block seals.
    pub block_seals: u64,
    /// Mutable tail allocations.
    pub tail_allocations: u64,
    /// Requests served by an already device-cataloged block.
    pub demand_hits: u64,
    /// Requests requiring host or disk promotion.
    pub demand_misses: u64,
    /// Requests that joined an existing transfer.
    pub in_flight_waits: u64,
    /// Configured disk request queue capacity.
    pub queue_capacity: usize,
    /// Peak observed queue occupancy.
    pub queue_peak_occupancy: usize,
    /// Requests delayed by queue capacity.
    pub queue_backpressure: u64,
    /// Requests canceled by reset or truncation.
    pub cancellations: u64,
    /// Cache transfer or persistence failures.
    pub failures: u64,
    /// Blocks scanned by full attention during prefill.
    pub prefill_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during prefill.
    pub prefill_full_attention_bytes: u64,
    /// Blocks scanned by full attention during decode.
    pub decode_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during decode.
    pub decode_full_attention_bytes: u64,
    /// Peak logical scratch bytes used by attention.
    pub attention_scratch_peak_bytes: u64,
    /// Successful prompt-cache saves.
    pub prompt_cache_saves: u64,
    /// Successful prompt-cache loads.
    pub prompt_cache_loads: u64,
    /// Logical bytes written or cataloged for prompt caches.
    pub prompt_cache_bytes: u64,
    /// Imported persistent shard count.
    pub imported_mapped_shards: u64,
    /// Optional peak process resident-set size sampled from the operating system.
    pub process_rss_bytes: Option<u64>,
    /// Optional cumulative minor page faults.
    pub process_minor_page_faults: Option<u64>,
    /// Optional cumulative major page faults.
    pub process_major_page_faults: Option<u64>,
}

#[derive(Debug, Default)]
struct CacheCounters {
    report: CacheResidencyReport,
}

#[derive(Debug)]
struct CacheManagerState {
    generation: u64,
    access_clock: u64,
    blocks: BTreeMap<CacheBlockId, CacheBlockRecord>,
    tails: HashMap<usize, (u64, i64)>,
    counters: CacheCounters,
    recent_device_blocks: usize,
}

/// Shared architecture-independent manager enforcing budgets across all layers.
#[derive(Debug, Clone)]
pub struct CacheResidencyManager {
    session_id: u64,
    options: PagedCacheOptions,
    state: Arc<Mutex<CacheManagerState>>,
    disk_worker: Option<Arc<DiskWorker>>,
}

impl CacheResidencyManager {
    /// Creates an empty manager with globally shared finite limits.
    pub fn new(options: PagedCacheOptions) -> Result<Self, CacheResidencyError> {
        if let LiveCacheDiskPolicy::Enabled { directory, .. } = &options.live_disk {
            fs::create_dir_all(directory).map_err(|source| CacheResidencyError::Io {
                action: "create live cache directory",
                path: directory.clone(),
                source,
            })?;
        }
        let queue_capacity = match &options.live_disk {
            LiveCacheDiskPolicy::Disabled => 0,
            LiveCacheDiskPolicy::Enabled { queue_capacity, .. } => *queue_capacity,
        };
        let mut counters = CacheCounters::default();
        counters.report.queue_capacity = queue_capacity;
        let disk_worker = match &options.live_disk {
            LiveCacheDiskPolicy::Disabled => None,
            LiveCacheDiskPolicy::Enabled { queue_capacity, .. } => {
                Some(Arc::new(DiskWorker::new(*queue_capacity)?))
            }
        };
        let recent_device_blocks = options.recent_device_blocks;
        Ok(Self {
            session_id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
            options,
            state: Arc::new(Mutex::new(CacheManagerState {
                generation: 0,
                access_clock: 0,
                blocks: BTreeMap::new(),
                tails: HashMap::new(),
                counters,
                recent_device_blocks,
            })),
            disk_worker,
        })
    }

    /// Returns the live cache identity included in every block id.
    pub const fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Returns validated paged-cache options.
    pub const fn options(&self) -> &PagedCacheOptions {
        &self.options
    }

    fn lock(&self) -> Result<MutexGuard<'_, CacheManagerState>, CacheResidencyError> {
        self.state
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)
    }

    pub(crate) fn set_tail_state(
        &self,
        layer: usize,
        bytes: u64,
        end: i64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        let previous = state.tails.insert(layer, (bytes, end));
        let allocated = previous.is_none_or(|tail| tail.0 == 0) && bytes > 0;
        if allocated {
            state.counters.report.tail_allocations += 1;
        }
        if let Err(error) = self.rebalance_locked(&mut state, None) {
            match previous {
                Some(previous) => {
                    state.tails.insert(layer, previous);
                }
                None => {
                    state.tails.remove(&layer);
                }
            }
            if allocated {
                state.counters.report.tail_allocations =
                    state.counters.report.tail_allocations.saturating_sub(1);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn seal_block(
        &self,
        global_layer: usize,
        start: i64,
        end: i64,
        rank: Option<CacheRankIdentity>,
        arrays: CacheBlockArrays,
        protected_prefix: bool,
    ) -> Result<CacheBlockId, CacheResidencyError> {
        if start < 0 || end <= start {
            return Err(CacheResidencyError::InvalidTokenRange { start, end });
        }
        let representation = arrays.representation();
        validate_block_arrays(&arrays, end - start)?;
        eval(arrays.arrays()).map_err(|source| CacheResidencyError::Runtime(source.to_string()))?;
        let id = CacheBlockId {
            session_id: self.session_id,
            global_layer,
            representation,
            start,
            end,
            rank,
        };
        let mut state = self.lock()?;
        if state.blocks.contains_key(&id) {
            return Err(CacheResidencyError::DuplicateBlock(id));
        }
        state.access_clock += 1;
        let bytes = arrays.bytes();
        let record = CacheBlockRecord {
            id: id.clone(),
            tier: CacheTier::Device,
            shapes: arrays.shapes(),
            dtypes: arrays.dtypes(),
            arrays: Some(arrays),
            disk: None,
            bytes,
            imported: false,
            leases: 0,
            access_count: 0,
            last_access: state.access_clock,
            protected_prefix,
        };
        state.blocks.insert(id.clone(), record);
        if let Err(error) = self.rebalance_locked(&mut state, Some(&id)) {
            if let Some(record) = state.blocks.remove(&id) {
                remove_ephemeral_file(&record);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        state.counters.report.block_seals += 1;
        Ok(id)
    }

    pub(crate) fn layer_block_ids(
        &self,
        layer: usize,
        representation: CacheRepresentation,
        visible_start: i64,
        visible_end: i64,
        prefix_tokens: i64,
    ) -> Result<Vec<CacheBlockId>, CacheResidencyError> {
        let state = self.lock()?;
        Ok(state
            .blocks
            .keys()
            .filter(|id| {
                id.global_layer == layer
                    && id.representation == representation
                    && id.start < visible_end
                    && (id.end > visible_start || id.start < prefix_tokens)
            })
            .cloned()
            .collect())
    }

    pub(crate) fn layer_end(
        &self,
        layer: usize,
        representation: CacheRepresentation,
    ) -> Result<i64, CacheResidencyError> {
        let state = self.lock()?;
        Ok(state
            .blocks
            .keys()
            .filter(|id| id.global_layer == layer && id.representation == representation)
            .map(|id| id.end)
            .max()
            .unwrap_or(0))
    }

    pub(crate) fn remove_block(&self, id: &CacheBlockId) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        let record = state
            .blocks
            .get(id)
            .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
        if record.leases != 0 {
            return Err(CacheResidencyError::BlockLeased(id.clone()));
        }
        let record = state
            .blocks
            .remove(id)
            .expect("validated cache block still present");
        remove_ephemeral_file(&record);
        update_report_totals(&mut state);
        Ok(())
    }

    pub(crate) fn lease_block(
        &self,
        id: &CacheBlockId,
        _stream: &Stream,
    ) -> Result<CacheBlockLease, CacheResidencyError> {
        let started = Instant::now();
        let mut state = self.lock()?;
        state.access_clock += 1;
        let access_clock = state.access_clock;
        let (tier, disk) = {
            let record = state
                .blocks
                .get(id)
                .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
            (record.tier, record.disk.clone())
        };
        let loaded = if tier == CacheTier::Disk {
            let location = disk
                .as_ref()
                .ok_or_else(|| CacheResidencyError::MissingDiskLocation(id.clone()))?;
            let (arrays, backpressure) = match &self.disk_worker {
                Some(worker) => worker.read(location, id.representation)?,
                None => (
                    load_block_arrays_direct(location, id.representation)?,
                    false,
                ),
            };
            state.counters.report.queue_peak_occupancy =
                state.counters.report.queue_peak_occupancy.max(1);
            state.counters.report.queue_backpressure += u64::from(backpressure);
            let record = state
                .blocks
                .get(id)
                .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
            if arrays.shapes() != record.shapes || arrays.dtypes() != record.dtypes {
                return Err(CacheResidencyError::MalformedShard {
                    path: disk.expect("disk-backed record has a location").path,
                    reason: "array shape or dtype does not match the manifest".into(),
                });
            }
            Some(arrays)
        } else {
            None
        };
        let (arrays, bytes) = {
            let record = state
                .blocks
                .get_mut(id)
                .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
            if let Some(arrays) = loaded {
                record.arrays = Some(arrays);
            }
            let arrays = record
                .arrays
                .as_ref()
                .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?
                .clone();
            record.tier = CacheTier::Device;
            record.leases += 1;
            record.access_count += 1;
            record.last_access = access_clock;
            (arrays, record.bytes)
        };
        match tier {
            CacheTier::Device => state.counters.report.demand_hits += 1,
            CacheTier::Host => {
                state.counters.report.demand_misses += 1;
                state.counters.report.host_promotions += 1;
                state.counters.report.transfer_bytes += bytes;
            }
            CacheTier::Disk => {
                state.counters.report.demand_misses += 1;
                state.counters.report.disk_promotions += 1;
                state.counters.report.transfer_bytes += bytes;
            }
        }
        state.counters.report.transfer_wait += started.elapsed();
        self.rebalance_locked(&mut state, Some(id))?;
        Ok(CacheBlockLease {
            id: id.clone(),
            arrays,
            manager: self.clone(),
            released: false,
        })
    }

    pub(crate) fn discard_before(
        &self,
        layer: usize,
        representation: CacheRepresentation,
        visible_start: i64,
        prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if self.options.retain_discarded_for_persistence {
            return Ok(());
        }
        let mut state = self.lock()?;
        let ids = state
            .blocks
            .iter()
            .filter(|(id, record)| {
                id.global_layer == layer
                    && id.representation == representation
                    && id.end <= visible_start
                    && id.end > prefix_tokens
                    && record.leases == 0
                    && !record.imported
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in ids {
            if let Some(record) = state.blocks.remove(&id) {
                remove_ephemeral_file(&record);
                state.counters.report.discarded_sliding_blocks += 1;
            }
        }
        update_report_totals(&mut state);
        Ok(())
    }

    /// Clears every live block and advances the manager generation.
    pub fn clear(&self) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if let Some(id) = state
            .blocks
            .values()
            .find(|record| record.leases != 0)
            .map(|record| record.id.clone())
        {
            return Err(CacheResidencyError::BlockLeased(id));
        }
        for record in state.blocks.values() {
            remove_ephemeral_file(record);
        }
        state.generation += 1;
        state.counters.report.cancellations += state.blocks.len() as u64;
        state.blocks.clear();
        state.tails.clear();
        update_report_totals(&mut state);
        Ok(())
    }

    /// Returns a bounded aggregate snapshot without retaining per-block history.
    pub fn report(&self) -> Result<CacheResidencyReport, CacheResidencyError> {
        let mut state = self.lock()?;
        update_report_totals(&mut state);
        if self.options.sample_process {
            sample_process(&mut state.counters.report);
        }
        Ok(state.counters.report.clone())
    }

    pub(crate) fn record_attention_scan(
        &self,
        prefill: bool,
        blocks: u64,
        bytes: u64,
        scratch_bytes: u64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if prefill {
            state.counters.report.prefill_full_attention_blocks += blocks;
            state.counters.report.prefill_full_attention_bytes += bytes;
        } else {
            state.counters.report.decode_full_attention_blocks += blocks;
            state.counters.report.decode_full_attention_bytes += bytes;
        }
        state.counters.report.attention_scratch_peak_bytes = state
            .counters
            .report
            .attention_scratch_peak_bytes
            .max(scratch_bytes);
        Ok(())
    }

    fn release_lease(&self, id: &CacheBlockId) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(record) = state.blocks.get_mut(id) {
                record.leases = record.leases.saturating_sub(1);
            }
            let _ = self.rebalance_locked(&mut state, None);
        }
    }

    fn rebalance_locked(
        &self,
        state: &mut CacheManagerState,
        required: Option<&CacheBlockId>,
    ) -> Result<(), CacheResidencyError> {
        let prior_peaks = (
            state.counters.report.peak_device_bytes,
            state.counters.report.peak_host_bytes,
            state.counters.report.peak_disk_bytes,
        );
        let result = self.rebalance_inner_locked(state, required);
        update_report_totals(state);
        let report = &mut state.counters.report;
        if result.is_ok() {
            report.peak_device_bytes = prior_peaks.0.max(report.current_device_bytes);
            report.peak_host_bytes = prior_peaks.1.max(report.current_host_bytes);
            report.peak_disk_bytes = prior_peaks.2.max(report.current_disk_bytes);
        } else {
            report.peak_device_bytes = prior_peaks.0;
            report.peak_host_bytes = prior_peaks.1;
            report.peak_disk_bytes = prior_peaks.2;
        }
        result
    }

    fn rebalance_inner_locked(
        &self,
        state: &mut CacheManagerState,
        required: Option<&CacheBlockId>,
    ) -> Result<(), CacheResidencyError> {
        loop {
            update_report_totals(state);
            if state.counters.report.current_device_bytes <= self.options.device_budget_bytes {
                break;
            }
            let candidate = eviction_candidate(
                state,
                CacheTier::Device,
                required,
                self.options.recent_device_blocks,
                self.options.eviction_policy,
            );
            let Some(id) = candidate else {
                state.counters.report.failures += 1;
                return Err(CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Device,
                    required: state.counters.report.current_device_bytes,
                    budget: self.options.device_budget_bytes,
                });
            };
            let record = state.blocks.get_mut(&id).expect("candidate exists");
            record.tier = CacheTier::Host;
            state.counters.report.host_demotions += 1;
            state.counters.report.transfer_bytes += record.bytes;
        }

        loop {
            update_report_totals(state);
            if state.counters.report.current_host_bytes <= self.options.host_budget_bytes {
                break;
            }
            let candidate = eviction_candidate(
                state,
                CacheTier::Host,
                required,
                0,
                self.options.eviction_policy,
            );
            let Some(id) = candidate else {
                state.counters.report.failures += 1;
                return Err(CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Host,
                    required: state.counters.report.current_host_bytes,
                    budget: self.options.host_budget_bytes,
                });
            };
            match &self.options.live_disk {
                LiveCacheDiskPolicy::Disabled => {
                    state.counters.report.failures += 1;
                    return Err(CacheResidencyError::LiveDiskRequired {
                        required: state.counters.report.current_host_bytes,
                        budget: self.options.host_budget_bytes,
                    });
                }
                LiveCacheDiskPolicy::Enabled {
                    directory,
                    budget_bytes,
                    ..
                } => {
                    let record = state.blocks.get(&id).expect("candidate exists");
                    let live_disk_bytes = state
                        .blocks
                        .values()
                        .filter(|record| {
                            record.tier == CacheTier::Disk
                                && record
                                    .disk
                                    .as_ref()
                                    .is_some_and(|location| !location.persistent)
                        })
                        .map(|record| record.bytes)
                        .sum::<u64>();
                    let projected = live_disk_bytes.saturating_add(record.bytes);
                    if projected > *budget_bytes {
                        state.counters.report.failures += 1;
                        return Err(CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Disk,
                            required: projected,
                            budget: *budget_bytes,
                        });
                    }
                    let arrays = record
                        .arrays
                        .as_ref()
                        .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?
                        .clone();
                    let (location, backpressure) = self
                        .disk_worker
                        .as_ref()
                        .ok_or_else(|| {
                            CacheResidencyError::Runtime(
                                "live cache disk worker is unavailable".into(),
                            )
                        })?
                        .write(directory, &id, &arrays)?;
                    state.counters.report.queue_peak_occupancy =
                        state.counters.report.queue_peak_occupancy.max(1);
                    state.counters.report.queue_backpressure += u64::from(backpressure);
                    let record = state.blocks.get_mut(&id).expect("candidate exists");
                    record.disk = Some(location);
                    record.arrays = None;
                    record.tier = CacheTier::Disk;
                    state.counters.report.disk_demotions += 1;
                    state.counters.report.transfer_bytes += record.bytes;
                }
            }
        }
        update_report_totals(state);
        Ok(())
    }

    /// Writes a completed immutable prefix atomically to a persistent directory.
    pub fn save_prompt_cache(
        &self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, CacheResidencyError> {
        let destination = destination.as_ref();
        if descriptor.layer_count == 0
            || descriptor.global_layer_start >= descriptor.global_layer_end
            || descriptor.global_layer_end > descriptor.layer_count
            || descriptor.batch_size == 0
            || descriptor.batch_size > i32::MAX as usize
        {
            return Err(CacheResidencyError::MalformedManifest(
                "invalid prompt-cache descriptor dimensions".into(),
            ));
        }
        let parent = destination.parent().ok_or_else(|| {
            CacheResidencyError::InvalidPromptCachePath(destination.to_path_buf())
        })?;
        fs::create_dir_all(parent).map_err(|source| CacheResidencyError::Io {
            action: "create prompt cache parent",
            path: parent.to_path_buf(),
            source,
        })?;
        if destination.exists() && !options.replace_existing {
            return Err(CacheResidencyError::PromptCacheExists(
                destination.to_path_buf(),
            ));
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let file_name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| CacheResidencyError::InvalidPromptCachePath(destination.into()))?;
        let temporary = parent.join(format!(".{file_name}.tmp-{nonce}"));
        fs::create_dir(&temporary).map_err(|source| CacheResidencyError::Io {
            action: "create temporary prompt cache",
            path: temporary.clone(),
            source,
        })?;

        let result = (|| {
            let records = {
                let state = self.lock()?;
                if let Some(id) = state
                    .blocks
                    .values()
                    .find(|record| record.leases != 0)
                    .map(|record| record.id.clone())
                {
                    return Err(CacheResidencyError::BlockLeased(id));
                }
                state.blocks.values().cloned().collect::<Vec<_>>()
            };
            validate_complete_prefix(
                &records,
                descriptor.global_layer_start,
                descriptor.global_layer_end,
                prefix_token_ids.len(),
            )?;
            let mut manifest_blocks = Vec::with_capacity(records.len());
            let mut logical_bytes = 0u64;
            for (index, record) in records.iter().enumerate() {
                let arrays = match &record.arrays {
                    Some(arrays) => arrays.clone(),
                    None => load_block_arrays_direct(
                        record.disk.as_ref().ok_or_else(|| {
                            CacheResidencyError::MissingDiskLocation(record.id.clone())
                        })?,
                        record.id.representation,
                    )?,
                };
                let shard = format!("block-{index:08}.safetensors");
                let shard_path = temporary.join(&shard);
                save_block_arrays(&shard_path, &arrays)?;
                sync_file(&shard_path)?;
                logical_bytes += arrays.bytes();
                let names = array_names(record.id.representation);
                manifest_blocks.push(PromptCacheBlock {
                    global_layer: record.id.global_layer,
                    representation: record.id.representation,
                    start: record.id.start,
                    end: record.id.end,
                    rank: record.id.rank,
                    shard,
                    first_array: names.0.into(),
                    second_array: names.1.into(),
                    first_shape: record.shapes[0].clone(),
                    second_shape: record.shapes[1].clone(),
                    first_dtype: record.dtypes[0].clone(),
                    second_dtype: record.dtypes[1].clone(),
                    logical_bytes: record.bytes,
                });
            }
            let manifest = PromptCacheManifest {
                schema_version: PROMPT_CACHE_SCHEMA_VERSION,
                model_family: descriptor.model_family,
                effective_model_type: descriptor.effective_model_type,
                checkpoint_fingerprint: descriptor.checkpoint_fingerprint,
                architecture_fingerprint: descriptor.architecture_fingerprint,
                layer_count: descriptor.layer_count,
                global_layer_start: descriptor.global_layer_start,
                global_layer_end: descriptor.global_layer_end,
                block_size_tokens: self.options.block_size_tokens,
                batch_size: descriptor.batch_size,
                total_prefix_tokens: prefix_token_ids.len(),
                prefix_sha256: hash_token_ids(prefix_token_ids),
                sliding_window: descriptor.sliding_window,
                sink_tokens: descriptor.sink_tokens,
                topology: descriptor.topology,
                application_namespace: options.application_namespace.clone(),
                blocks: manifest_blocks,
            };
            let manifest_path = temporary.join("manifest.json");
            let file = File::create(&manifest_path).map_err(|source| CacheResidencyError::Io {
                action: "create prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &manifest)
                .map_err(CacheResidencyError::ManifestJson)?;
            writer
                .write_all(b"\n")
                .map_err(|source| CacheResidencyError::Io {
                    action: "write prompt cache manifest",
                    path: manifest_path.clone(),
                    source,
                })?;
            writer.flush().map_err(|source| CacheResidencyError::Io {
                action: "flush prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?;
            sync_file(&manifest_path)?;
            validate_manifest(&temporary, &manifest)?;
            sync_directory(&temporary)?;

            let backup = parent.join(format!(".{file_name}.old-{nonce}"));
            if destination.exists() {
                fs::rename(destination, &backup).map_err(|source| CacheResidencyError::Io {
                    action: "move replaced prompt cache",
                    path: destination.to_path_buf(),
                    source,
                })?;
            }
            if let Err(source) = fs::rename(&temporary, destination) {
                if backup.exists() {
                    let _ = fs::rename(&backup, destination);
                }
                return Err(CacheResidencyError::Io {
                    action: "publish prompt cache",
                    path: destination.to_path_buf(),
                    source,
                });
            }
            if backup.exists() {
                fs::remove_dir_all(&backup).map_err(|source| CacheResidencyError::Io {
                    action: "remove replaced prompt cache",
                    path: backup,
                    source,
                })?;
            }
            sync_directory(parent)?;
            let mut state = self.lock()?;
            state.counters.report.prompt_cache_saves += 1;
            state.counters.report.prompt_cache_bytes += logical_bytes;
            Ok(manifest)
        })();

        if result.is_err() && temporary.exists() {
            let _ = fs::remove_dir_all(&temporary);
        }
        result
    }
}

impl Drop for CacheResidencyManager {
    fn drop(&mut self) {
        if Arc::strong_count(&self.state) != 1 {
            return;
        }
        if let Ok(state) = self.state.lock() {
            for record in state.blocks.values() {
                remove_ephemeral_file(record);
            }
        }
    }
}

pub(crate) struct CacheBlockLease {
    id: CacheBlockId,
    arrays: CacheBlockArrays,
    manager: CacheResidencyManager,
    released: bool,
}

impl CacheBlockLease {
    pub(crate) fn arrays(&self) -> &CacheBlockArrays {
        &self.arrays
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.arrays.bytes()
    }
}

impl Drop for CacheBlockLease {
    fn drop(&mut self) {
        if !self.released {
            self.manager.release_lease(&self.id);
            self.released = true;
        }
    }
}

/// Compatibility identity supplied by a model when persisting a prefix cache.
#[derive(Debug, Clone)]
pub struct PromptCacheDescriptor {
    /// Stable architecture family, such as `llama` or `deepseek_v3`.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Caller-verified checkpoint identity that is not based only on a path.
    pub checkpoint_fingerprint: String,
    /// Hash or canonical serialization of RoPE and cache-relevant architecture settings.
    pub architecture_fingerprint: String,
    /// Total model layer count.
    pub layer_count: usize,
    /// Inclusive global layer range stored by this rank.
    pub global_layer_start: usize,
    /// Exclusive global layer range stored by this rank.
    pub global_layer_end: usize,
    /// Prefix batch size.
    pub batch_size: usize,
    /// Sliding attention window, when present.
    pub sliding_window: Option<i32>,
    /// Attention sink or pinned-prefix token count.
    pub sink_tokens: usize,
    /// Distributed rank-local layout.
    pub topology: PromptCacheTopology,
}

/// Rank-local topology recorded in a prompt-cache manifest.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheTopology {
    /// Pipeline world size and rank.
    pub pipeline: Option<(usize, usize)>,
    /// Tensor-parallel world size and rank.
    pub tensor_parallel: Option<(usize, usize)>,
    /// Expert-parallel world size and rank.
    pub expert_parallel: Option<(usize, usize)>,
    /// Whether attention cache state is replicated on the expert-parallel axis.
    pub expert_parallel_cache_replicated: bool,
}

impl Default for PromptCacheTopology {
    fn default() -> Self {
        Self {
            pipeline: None,
            tensor_parallel: None,
            expert_parallel: None,
            expert_parallel_cache_replicated: true,
        }
    }
}

/// Explicit persistence behavior for a reusable prefix cache.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheOptions {
    /// Optional application grouping label; never used for compatibility checks.
    pub application_namespace: Option<String>,
    /// Allows atomically replacing an existing destination.
    pub replace_existing: bool,
}

/// Versioned metadata that can be inspected without loading block arrays.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheManifest {
    /// Persistence schema version.
    pub schema_version: u32,
    /// Model architecture family.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Checkpoint identity contract selected by the caller.
    pub checkpoint_fingerprint: String,
    /// RoPE and cache-relevant architecture identity.
    pub architecture_fingerprint: String,
    /// Total model layer count.
    pub layer_count: usize,
    /// Inclusive first global layer represented locally.
    pub global_layer_start: usize,
    /// Exclusive global layer boundary represented locally.
    pub global_layer_end: usize,
    /// Block size used by the producer.
    pub block_size_tokens: i32,
    /// Prefix batch size.
    pub batch_size: usize,
    /// Exact prefix token count.
    pub total_prefix_tokens: usize,
    /// SHA-256 over little-endian prefix token ids.
    pub prefix_sha256: String,
    /// Sliding attention window, when applicable.
    pub sliding_window: Option<i32>,
    /// Pinned prefix or sink token count.
    pub sink_tokens: usize,
    /// Distributed rank-local representation.
    pub topology: PromptCacheTopology,
    /// Optional non-authoritative application grouping label.
    pub application_namespace: Option<String>,
    /// Ordered immutable cache blocks.
    pub blocks: Vec<PromptCacheBlock>,
}

/// One cache block catalog entry in a prompt-cache manifest.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheBlock {
    /// Architecture-global layer identity.
    pub global_layer: usize,
    /// Stored attention representation.
    pub representation: CacheRepresentation,
    /// Inclusive absolute token position.
    pub start: i64,
    /// Exclusive absolute token position.
    pub end: i64,
    /// Optional rank identity.
    pub rank: Option<CacheRankIdentity>,
    /// Safe relative safetensors shard path.
    pub shard: String,
    /// First array name.
    pub first_array: String,
    /// Second array name.
    pub second_array: String,
    /// First array shape.
    pub first_shape: Vec<i32>,
    /// Second array shape.
    pub second_shape: Vec<i32>,
    /// First array dtype.
    pub first_dtype: String,
    /// Second array dtype.
    pub second_dtype: String,
    /// Logical bytes in both arrays.
    pub logical_bytes: u64,
}

/// Reads and validates a prompt-cache manifest without loading its arrays.
pub fn inspect_prompt_cache(
    directory: impl AsRef<Path>,
) -> Result<PromptCacheManifest, CacheResidencyError> {
    let directory = directory.as_ref();
    let manifest_path = directory.join("manifest.json");
    let reader =
        BufReader::new(
            File::open(&manifest_path).map_err(|source| CacheResidencyError::Io {
                action: "open prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?,
        );
    let manifest: PromptCacheManifest =
        serde_json::from_reader(reader).map_err(CacheResidencyError::ManifestJson)?;
    validate_manifest(directory, &manifest)?;
    Ok(manifest)
}

/// Catalogs a compatible prompt prefix lazily as read-only disk-backed blocks.
pub fn open_prompt_cache(
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    options: PagedCacheOptions,
) -> Result<(CacheResidencyManager, PromptCacheManifest), CacheResidencyError> {
    let directory = directory.as_ref();
    let manifest = inspect_prompt_cache(directory)?;
    validate_compatibility(&manifest, expected, prefix_token_ids)?;
    if manifest.block_size_tokens != options.block_size_tokens {
        return Err(CacheResidencyError::IncompatiblePromptCache(format!(
            "block size {} does not match requested {}",
            manifest.block_size_tokens, options.block_size_tokens
        )));
    }
    let manager = CacheResidencyManager::new(options)?;
    {
        let mut state = manager.lock()?;
        for block in &manifest.blocks {
            let id = CacheBlockId {
                session_id: manager.session_id,
                global_layer: block.global_layer,
                representation: block.representation,
                start: block.start,
                end: block.end,
                rank: block.rank,
            };
            let shard = safe_shard_path(directory, &block.shard)?;
            let record = CacheBlockRecord {
                id: id.clone(),
                tier: CacheTier::Disk,
                arrays: None,
                disk: Some(DiskLocation {
                    path: shard,
                    first_name: block.first_array.clone(),
                    second_name: block.second_array.clone(),
                    persistent: true,
                }),
                bytes: block.logical_bytes,
                shapes: [block.first_shape.clone(), block.second_shape.clone()],
                dtypes: [block.first_dtype.clone(), block.second_dtype.clone()],
                imported: true,
                leases: 0,
                access_count: 0,
                last_access: 0,
                protected_prefix: block.end <= manifest.sink_tokens as i64,
            };
            if state.blocks.insert(id.clone(), record).is_some() {
                return Err(CacheResidencyError::DuplicateBlock(id));
            }
        }
        state.counters.report.prompt_cache_loads += 1;
        state.counters.report.prompt_cache_bytes += manifest
            .blocks
            .iter()
            .map(|block| block.logical_bytes)
            .sum::<u64>();
        state.counters.report.imported_mapped_shards += manifest.blocks.len() as u64;
        update_report_totals(&mut state);
    }
    Ok((manager, manifest))
}

fn validate_compatibility(
    manifest: &PromptCacheManifest,
    expected: &PromptCacheDescriptor,
    prefix_token_ids: &[u32],
) -> Result<(), CacheResidencyError> {
    macro_rules! require_equal {
        ($field:ident) => {
            if manifest.$field != expected.$field {
                return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                    "{} mismatch",
                    stringify!($field)
                )));
            }
        };
    }
    require_equal!(model_family);
    require_equal!(effective_model_type);
    require_equal!(checkpoint_fingerprint);
    require_equal!(architecture_fingerprint);
    require_equal!(layer_count);
    require_equal!(global_layer_start);
    require_equal!(global_layer_end);
    require_equal!(batch_size);
    require_equal!(sliding_window);
    require_equal!(sink_tokens);
    require_equal!(topology);
    if manifest.total_prefix_tokens != prefix_token_ids.len()
        || manifest.prefix_sha256 != hash_token_ids(prefix_token_ids)
    {
        return Err(CacheResidencyError::PrefixIdentityMismatch);
    }
    Ok(())
}

fn validate_manifest(
    directory: &Path,
    manifest: &PromptCacheManifest,
) -> Result<(), CacheResidencyError> {
    if manifest.schema_version != PROMPT_CACHE_SCHEMA_VERSION {
        return Err(CacheResidencyError::UnsupportedSchema(
            manifest.schema_version,
        ));
    }
    if manifest.block_size_tokens <= 0
        || manifest.layer_count == 0
        || manifest.global_layer_start >= manifest.global_layer_end
        || manifest.global_layer_end > manifest.layer_count
        || manifest.batch_size == 0
        || manifest.batch_size > i32::MAX as usize
        || manifest.total_prefix_tokens == 0
    {
        return Err(CacheResidencyError::MalformedManifest(
            "invalid global cache dimensions".into(),
        ));
    }
    for (name, topology) in [
        ("pipeline", manifest.topology.pipeline),
        ("tensor parallel", manifest.topology.tensor_parallel),
        ("expert parallel", manifest.topology.expert_parallel),
    ] {
        if topology.is_some_and(|(world_size, rank)| world_size == 0 || rank >= world_size) {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "invalid {name} topology"
            )));
        }
    }
    let mut by_layer: BTreeMap<(usize, CacheRepresentation), Vec<&PromptCacheBlock>> =
        BTreeMap::new();
    for block in &manifest.blocks {
        if block.global_layer < manifest.global_layer_start
            || block.global_layer >= manifest.global_layer_end
            || block.start < 0
            || block.end <= block.start
            || block.end > manifest.total_prefix_tokens as i64
            || block.logical_bytes == 0
            || block.first_shape.is_empty()
            || block.second_shape.is_empty()
        {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "invalid block at layer {} range {}..{}",
                block.global_layer, block.start, block.end
            )));
        }
        let expected_rank = CacheRankIdentity {
            pipeline_rank: manifest.topology.pipeline.map(|(_, rank)| rank),
            tensor_parallel_rank: manifest.topology.tensor_parallel.map(|(_, rank)| rank),
            expert_parallel_rank: manifest.topology.expert_parallel.map(|(_, rank)| rank),
        };
        let has_rank = expected_rank.pipeline_rank.is_some()
            || expected_rank.tensor_parallel_rank.is_some()
            || expected_rank.expert_parallel_rank.is_some();
        if block.rank != has_rank.then_some(expected_rank) {
            return Err(CacheResidencyError::MalformedManifest(
                "block rank identity does not match the recorded topology".into(),
            ));
        }
        if block.first_dtype != block.second_dtype
            || block.first_shape.first() != Some(&(manifest.batch_size as i32))
            || block.second_shape.first() != Some(&(manifest.batch_size as i32))
        {
            return Err(CacheResidencyError::MalformedManifest(
                "block batch dimension or dtype is inconsistent".into(),
            ));
        }
        let names = array_names(block.representation);
        if block.first_array != names.0 || block.second_array != names.1 {
            return Err(CacheResidencyError::MalformedManifest(
                "block array names do not match its representation".into(),
            ));
        }
        match block.representation {
            CacheRepresentation::KeyValue
                if block.first_shape.len() != 4 || block.first_shape != block.second_shape =>
            {
                return Err(CacheResidencyError::MalformedManifest(
                    "key/value blocks must use identical rank-4 shapes".into(),
                ));
            }
            CacheRepresentation::CompressedLatentRotary
                if block.first_shape.len() != 3 || block.second_shape.len() != 3 =>
            {
                return Err(CacheResidencyError::MalformedManifest(
                    "compressed latent/rotary blocks must use rank-3 shapes".into(),
                ));
            }
            _ => {}
        }
        let sequence_axis = match block.representation {
            CacheRepresentation::KeyValue => block.first_shape.len().checked_sub(2),
            CacheRepresentation::CompressedLatentRotary => Some(1),
        }
        .ok_or_else(|| CacheResidencyError::MalformedManifest("invalid block rank".into()))?;
        if block.first_shape.get(sequence_axis) != Some(&((block.end - block.start) as i32))
            || block.second_shape.get(sequence_axis) != Some(&((block.end - block.start) as i32))
        {
            return Err(CacheResidencyError::MalformedManifest(
                "block token range does not match array shapes".into(),
            ));
        }
        let shard = safe_shard_path(directory, &block.shard)?;
        if !shard.is_file() {
            return Err(CacheResidencyError::MissingShard(shard));
        }
        validate_shard_file(&shard, block)?;
        by_layer
            .entry((block.global_layer, block.representation))
            .or_default()
            .push(block);
    }
    for layer in manifest.global_layer_start..manifest.global_layer_end {
        let entries = by_layer
            .iter()
            .filter(|((entry_layer, _), _)| *entry_layer == layer)
            .flat_map(|(_, blocks)| blocks.iter().copied())
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "missing blocks for global layer {layer}"
            )));
        }
        let mut entries = entries;
        entries.sort_by_key(|block| block.start);
        let mut expected_start = 0i64;
        for block in entries {
            if block.start != expected_start {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "gap or overlap at global layer {layer}: expected {expected_start}, found {}",
                    block.start
                )));
            }
            expected_start = block.end;
        }
        if expected_start != manifest.total_prefix_tokens as i64 {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {layer} ends at {expected_start}, expected {}",
                manifest.total_prefix_tokens
            )));
        }
    }
    Ok(())
}

fn validate_complete_prefix(
    records: &[CacheBlockRecord],
    global_layer_start: usize,
    global_layer_end: usize,
    prefix_tokens: usize,
) -> Result<(), CacheResidencyError> {
    if prefix_tokens == 0 {
        return Err(CacheResidencyError::MalformedManifest(
            "cannot persist an empty prefix".into(),
        ));
    }
    if records.iter().any(|record| {
        record.id.global_layer < global_layer_start || record.id.global_layer >= global_layer_end
    }) {
        return Err(CacheResidencyError::MalformedManifest(
            "cache contains blocks outside the persisted global layer range".into(),
        ));
    }
    for layer in global_layer_start..global_layer_end {
        let mut blocks = records
            .iter()
            .filter(|record| record.id.global_layer == layer)
            .collect::<Vec<_>>();
        blocks.sort_by_key(|record| record.id.start);
        let mut end = 0i64;
        for block in blocks {
            if block.id.start != end {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "global layer {layer} has a gap or overlap at {end}"
                )));
            }
            end = block.id.end;
        }
        if end != prefix_tokens as i64 {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {layer} contains {end} tokens, expected {prefix_tokens}"
            )));
        }
    }
    Ok(())
}

fn validate_block_arrays(
    arrays: &CacheBlockArrays,
    token_count: i64,
) -> Result<(), CacheResidencyError> {
    let [first, second] = arrays.arrays();
    if first.dtype() != second.dtype() {
        return Err(CacheResidencyError::ArrayMismatch(
            "both arrays in a cache block must share a dtype".into(),
        ));
    }
    let sequence_axis = match arrays {
        CacheBlockArrays::KeyValue { .. } => {
            if first.ndim() < 2 || second.ndim() < 2 {
                return Err(CacheResidencyError::ArrayMismatch(
                    "key/value blocks must have a sequence axis".into(),
                ));
            }
            first.ndim() - 2
        }
        CacheBlockArrays::CompressedLatentRotary { .. } => {
            if first.ndim() != 3 || second.ndim() != 3 {
                return Err(CacheResidencyError::ArrayMismatch(
                    "compressed latent blocks must be rank-3".into(),
                ));
            }
            1
        }
    };
    if first.dim(sequence_axis as i32) as i64 != token_count
        || second.dim(sequence_axis as i32) as i64 != token_count
    {
        return Err(CacheResidencyError::ArrayMismatch(
            "cache block range does not match its sequence dimensions".into(),
        ));
    }
    match arrays {
        CacheBlockArrays::KeyValue { .. } => {
            if first.shape() != second.shape() {
                return Err(CacheResidencyError::ArrayMismatch(
                    "key and value block shapes must match".into(),
                ));
            }
        }
        CacheBlockArrays::CompressedLatentRotary { .. } => {
            if first.dim(0) != second.dim(0) || first.dim(1) != second.dim(1) {
                return Err(CacheResidencyError::ArrayMismatch(
                    "compressed latent and rotary blocks must share batch and sequence dimensions"
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

fn update_report_totals(state: &mut CacheManagerState) {
    let report = &mut state.counters.report;
    report.key_value_blocks = 0;
    report.compressed_latent_blocks = 0;
    report.device_blocks = 0;
    report.host_blocks = 0;
    report.disk_blocks = 0;
    report.current_device_bytes = state.tails.values().map(|tail| tail.0).sum();
    report.mutable_tail_bytes = report.current_device_bytes;
    report.current_host_bytes = 0;
    report.current_disk_bytes = 0;
    report.protected_prefix_blocks = 0;
    report.protected_recent_blocks = 0;
    report.logical_cached_tokens = 0;
    let mut layer_ends: HashMap<usize, i64> = HashMap::new();
    for (layer, (_, end)) in &state.tails {
        layer_ends.insert(*layer, *end);
    }
    for record in state.blocks.values() {
        match record.id.representation {
            CacheRepresentation::KeyValue => report.key_value_blocks += 1,
            CacheRepresentation::CompressedLatentRotary => report.compressed_latent_blocks += 1,
        }
        match record.tier {
            CacheTier::Device => {
                report.device_blocks += 1;
                report.current_device_bytes += record.bytes;
            }
            CacheTier::Host => {
                report.host_blocks += 1;
                report.current_host_bytes += record.bytes;
            }
            CacheTier::Disk => {
                report.disk_blocks += 1;
                report.current_disk_bytes += record.bytes;
            }
        }
        if record.protected_prefix {
            report.protected_prefix_blocks += 1;
        }
        layer_ends
            .entry(record.id.global_layer)
            .and_modify(|end| *end = (*end).max(record.id.end))
            .or_insert(record.id.end);
    }
    let mut device_starts = HashMap::<usize, Vec<i64>>::new();
    for record in state
        .blocks
        .values()
        .filter(|record| record.tier == CacheTier::Device && !record.protected_prefix)
    {
        device_starts
            .entry(record.id.global_layer)
            .or_default()
            .push(record.id.start);
    }
    report.protected_recent_blocks = device_starts
        .values()
        .map(|starts| starts.len().min(state.recent_device_blocks) as u64)
        .sum();
    report.logical_cached_tokens = layer_ends.values().copied().max().unwrap_or(0).max(0) as u64;
    report.peak_device_bytes = report.peak_device_bytes.max(report.current_device_bytes);
    report.peak_host_bytes = report.peak_host_bytes.max(report.current_host_bytes);
    report.peak_disk_bytes = report.peak_disk_bytes.max(report.current_disk_bytes);
}

fn eviction_candidate(
    state: &CacheManagerState,
    tier: CacheTier,
    required: Option<&CacheBlockId>,
    recent_per_layer: usize,
    policy: CacheEvictionPolicy,
) -> Option<CacheBlockId> {
    let mut recent = HashMap::<usize, Vec<i64>>::new();
    if tier == CacheTier::Device && recent_per_layer > 0 {
        for record in state.blocks.values().filter(|record| record.tier == tier) {
            recent
                .entry(record.id.global_layer)
                .or_default()
                .push(record.id.start);
        }
        for starts in recent.values_mut() {
            starts.sort_unstable_by(|a, b| b.cmp(a));
            starts.truncate(recent_per_layer);
        }
    }
    state
        .blocks
        .values()
        .filter(|record| {
            record.tier == tier
                && record.leases == 0
                && required != Some(&record.id)
                && !record.protected_prefix
                && !recent
                    .get(&record.id.global_layer)
                    .is_some_and(|starts| starts.contains(&record.id.start))
        })
        .min_by_key(|record| match policy {
            CacheEvictionPolicy::LeastRecentlyUsed => {
                (record.last_access, record.access_count, record.id.clone())
            }
            CacheEvictionPolicy::LeastFrequentlyUsed => {
                (record.access_count, record.last_access, record.id.clone())
            }
        })
        .map(|record| record.id.clone())
}

fn write_live_block(
    directory: &Path,
    id: &CacheBlockId,
    arrays: &CacheBlockArrays,
) -> Result<DiskLocation, CacheResidencyError> {
    let name = format!(
        "session-{}-layer-{:05}-{}-{}.safetensors",
        id.session_id, id.global_layer, id.start, id.end
    );
    let path = directory.join(name);
    let temporary = directory.join(format!(
        ".session-{}-layer-{:05}-{}-{}.tmp.safetensors",
        id.session_id, id.global_layer, id.start, id.end
    ));
    save_block_arrays(&temporary, arrays)?;
    sync_file(&temporary)?;
    fs::rename(&temporary, &path).map_err(|source| CacheResidencyError::Io {
        action: "publish live cache block",
        path: path.clone(),
        source,
    })?;
    let names = array_names(id.representation);
    Ok(DiskLocation {
        path,
        first_name: names.0.into(),
        second_name: names.1.into(),
        persistent: false,
    })
}

fn save_block_arrays(path: &Path, arrays: &CacheBlockArrays) -> Result<(), CacheResidencyError> {
    let names = array_names(arrays.representation());
    let values = arrays.arrays();
    Array::save_safetensors([(names.0, values[0]), (names.1, values[1])], None, path).map_err(
        |source| CacheResidencyError::Runtime(format!("save {}: {source}", path.display())),
    )
}

fn load_block_arrays_direct(
    location: &DiskLocation,
    representation: CacheRepresentation,
) -> Result<CacheBlockArrays, CacheResidencyError> {
    let stream = cpu_stream();
    let mut arrays = Array::load_safetensors(&location.path, &stream).map_err(|source| {
        CacheResidencyError::Runtime(format!("load {}: {source}", location.path.display()))
    })?;
    let first =
        arrays
            .remove(&location.first_name)
            .ok_or_else(|| CacheResidencyError::MalformedShard {
                path: location.path.clone(),
                reason: format!("missing array {}", location.first_name),
            })?;
    let second = arrays.remove(&location.second_name).ok_or_else(|| {
        CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: format!("missing array {}", location.second_name),
        }
    })?;
    if !arrays.is_empty() {
        return Err(CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: "unexpected extra arrays".into(),
        });
    }
    Ok(match representation {
        CacheRepresentation::KeyValue => CacheBlockArrays::KeyValue {
            keys: first,
            values: second,
        },
        CacheRepresentation::CompressedLatentRotary => CacheBlockArrays::CompressedLatentRotary {
            latent: first,
            rotary_key: second,
        },
    })
}

fn remove_ephemeral_file(record: &CacheBlockRecord) {
    if let Some(location) = &record.disk {
        if !location.persistent {
            let _ = fs::remove_file(&location.path);
        }
    }
}

fn array_names(representation: CacheRepresentation) -> (&'static str, &'static str) {
    match representation {
        CacheRepresentation::KeyValue => ("keys", "values"),
        CacheRepresentation::CompressedLatentRotary => ("latent", "rotary_key"),
    }
}

fn dtype_name(dtype: Dtype) -> String {
    format!("{dtype:?}")
}

fn hash_token_ids(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn safe_shard_path(directory: &Path, relative: &str) -> Result<PathBuf, CacheResidencyError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(CacheResidencyError::UnsafeShardPath(relative.into()));
    }
    let joined = directory.join(path);
    if joined.exists() {
        let root = fs::canonicalize(directory).map_err(|source| CacheResidencyError::Io {
            action: "canonicalize prompt cache directory",
            path: directory.to_path_buf(),
            source,
        })?;
        let canonical = fs::canonicalize(&joined).map_err(|source| CacheResidencyError::Io {
            action: "canonicalize prompt cache shard",
            path: joined.clone(),
            source,
        })?;
        if !canonical.starts_with(&root) {
            return Err(CacheResidencyError::UnsafeShardPath(relative.into()));
        }
    }
    Ok(joined)
}

fn validate_shard_file(path: &Path, block: &PromptCacheBlock) -> Result<(), CacheResidencyError> {
    let bytes = fs::read(path).map_err(|source| CacheResidencyError::Io {
        action: "read prompt cache shard metadata",
        path: path.to_path_buf(),
        source,
    })?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;
    let entries = tensors.tensors();
    if entries.len() != 2 {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!("expected two arrays, found {}", entries.len()),
        });
    }
    let mut logical_bytes = 0u64;
    for (name, expected_shape, expected_dtype) in [
        (&block.first_array, &block.first_shape, &block.first_dtype),
        (
            &block.second_array,
            &block.second_shape,
            &block.second_dtype,
        ),
    ] {
        let tensor = tensors
            .tensor(name)
            .map_err(|error| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: error.to_string(),
            })?;
        let shape = tensor
            .shape()
            .iter()
            .map(|dimension| i32::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: "array dimension exceeds runtime range".into(),
            })?;
        if &shape != expected_shape || stored_dtype_name(tensor.dtype()) != *expected_dtype {
            return Err(CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: format!("array {name} shape or dtype does not match the manifest"),
            });
        }
        logical_bytes = logical_bytes.saturating_add(tensor.data().len() as u64);
    }
    if logical_bytes != block.logical_bytes {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!(
                "logical byte count {logical_bytes} does not match manifest value {}",
                block.logical_bytes
            ),
        });
    }
    Ok(())
}

fn stored_dtype_name(dtype: safetensors::Dtype) -> String {
    use safetensors::Dtype as Stored;
    match dtype {
        Stored::BOOL => "Bool",
        Stored::U8 => "Uint8",
        Stored::U16 => "Uint16",
        Stored::U32 => "Uint32",
        Stored::U64 => "Uint64",
        Stored::I8 => "Int8",
        Stored::I16 => "Int16",
        Stored::I32 => "Int32",
        Stored::I64 => "Int64",
        Stored::F16 => "Float16",
        Stored::BF16 => "Bfloat16",
        Stored::F32 => "Float32",
        Stored::F64 => "Float64",
        dtype => return format!("{dtype:?}"),
    }
    .into()
}

fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn sync_file(path: &Path) -> Result<(), CacheResidencyError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| CacheResidencyError::Io {
            action: "synchronize cache file",
            path: path.to_path_buf(),
            source,
        })
}

fn sync_directory(path: &Path) -> Result<(), CacheResidencyError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| CacheResidencyError::Io {
            action: "synchronize cache directory",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(unix)]
fn sample_process(report: &mut CacheResidencyReport) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to writable storage of the exact type required by
    // `getrusage`; the value is read only after a successful return.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return;
    }
    // SAFETY: a successful `getrusage` call initialized the structure.
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    let rss_bytes = usage.ru_maxrss.max(0) as u64;
    #[cfg(not(target_os = "macos"))]
    let rss_bytes = (usage.ru_maxrss.max(0) as u64).saturating_mul(1024);
    report.process_rss_bytes = Some(rss_bytes);
    report.process_minor_page_faults = Some(usage.ru_minflt.max(0) as u64);
    report.process_major_page_faults = Some(usage.ru_majflt.max(0) as u64);
}

#[cfg(not(unix))]
fn sample_process(_report: &mut CacheResidencyReport) {}

/// Structured cache residency and persistence failures.
#[derive(Debug, thiserror::Error)]
pub enum CacheResidencyError {
    /// Paged options were contradictory or unbounded.
    #[error("invalid paged cache options: {0}")]
    InvalidOptions(String),
    /// A sealed block used an invalid absolute token range.
    #[error("invalid cache block token range {start}..{end}")]
    InvalidTokenRange {
        /// Inclusive absolute token position.
        start: i64,
        /// Exclusive absolute token position.
        end: i64,
    },
    /// Both arrays in a block did not describe the same token range.
    #[error("invalid cache block arrays: {0}")]
    ArrayMismatch(String),
    /// A stable block identity was published more than once.
    #[error("duplicate cache block {0:?}")]
    DuplicateBlock(CacheBlockId),
    /// A requested block was not cataloged.
    #[error("missing cache block {0:?}")]
    MissingBlock(CacheBlockId),
    /// A disk-backed block had no safe location.
    #[error("cache block has no disk location: {0:?}")]
    MissingDiskLocation(CacheBlockId),
    /// A host or device block had no evaluated arrays.
    #[error("cache block has no resident arrays: {0:?}")]
    MissingResidentArrays(CacheBlockId),
    /// Active attention prevented mutation or eviction.
    #[error("cache block is leased by active attention: {0:?}")]
    BlockLeased(CacheBlockId),
    /// A finite tier budget could not admit required state.
    #[error("{tier:?} cache budget exceeded: requires {required} bytes, budget is {budget}")]
    BudgetExceeded {
        /// Tier that could not admit required state.
        tier: CacheTier,
        /// Logical bytes required by the operation.
        required: u64,
        /// Configured finite tier budget.
        budget: u64,
    },
    /// Full-context history exceeded host memory without explicit disk backing.
    #[error(
        "host cache requires {required} bytes but budget is {budget}; enable live disk backing or use a larger finite budget"
    )]
    LiveDiskRequired {
        /// Logical host bytes required by retained history.
        required: u64,
        /// Configured finite host budget.
        budget: u64,
    },
    /// The manager lock was poisoned by a panic.
    #[error("cache residency manager lock was poisoned")]
    ManagerPoisoned,
    /// MLX evaluation or array I/O failed.
    #[error("cache runtime failure: {0}")]
    Runtime(String),
    /// A filesystem operation failed.
    #[error("failed to {action} at {path}: {source}")]
    Io {
        /// Filesystem action that failed.
        action: &'static str,
        /// Path involved in the failed action.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// A manifest could not be encoded or decoded.
    #[error("invalid prompt cache manifest JSON: {0}")]
    ManifestJson(#[source] serde_json::Error),
    /// A manifest did not satisfy structural invariants.
    #[error("malformed prompt cache manifest: {0}")]
    MalformedManifest(String),
    /// The manifest schema is not supported by this runtime.
    #[error("unsupported prompt cache schema version {0}")]
    UnsupportedSchema(u32),
    /// A shard path could escape the prompt-cache directory.
    #[error("unsafe prompt cache shard path {0:?}")]
    UnsafeShardPath(String),
    /// A manifest referenced a missing shard.
    #[error("missing prompt cache shard {0}")]
    MissingShard(PathBuf),
    /// A safetensors block had missing, extra, or corrupt arrays.
    #[error("malformed prompt cache shard {path}: {reason}")]
    MalformedShard {
        /// Invalid shard path.
        path: PathBuf,
        /// Structural or data validation failure.
        reason: String,
    },
    /// The supplied model or topology differs from the producer.
    #[error("incompatible prompt cache: {0}")]
    IncompatiblePromptCache(String),
    /// Caller-provided prefix ids did not match the persisted prefix.
    #[error("prompt cache prefix token identity does not match")]
    PrefixIdentityMismatch,
    /// The target path cannot be published atomically.
    #[error("invalid prompt cache path {0}")]
    InvalidPromptCachePath(PathBuf),
    /// The destination exists and explicit replacement was not requested.
    #[error("prompt cache destination already exists: {0}")]
    PromptCacheExists(PathBuf),
}

#[cfg(test)]
mod tests {
    use super::{
        hash_token_ids, inspect_prompt_cache, safe_shard_path, CacheResidencyError,
        PagedCacheOptions,
    };
    use std::{fs, path::Path};

    #[test]
    fn paged_options_require_finite_nonzero_limits() {
        assert!(PagedCacheOptions::new(0, 1, 1, 1).is_err());
        assert!(PagedCacheOptions::new(16, 0, 1, 1).is_err());
        assert!(PagedCacheOptions::new(16, 1, 1, 0).is_err());
        assert!(PagedCacheOptions::new(16, 1, 0, 1).is_ok());
    }

    #[test]
    fn prefix_hash_is_stable_and_order_sensitive() {
        assert_eq!(hash_token_ids(&[1, 2, 3]), hash_token_ids(&[1, 2, 3]));
        assert_ne!(hash_token_ids(&[1, 2, 3]), hash_token_ids(&[3, 2, 1]));
    }

    #[test]
    fn shard_paths_cannot_escape_cache_directory() {
        let root = Path::new("/tmp/cache");
        assert_eq!(
            safe_shard_path(root, "block-0001.safetensors").unwrap(),
            root.join("block-0001.safetensors")
        );
        assert!(matches!(
            safe_shard_path(root, "../outside.safetensors"),
            Err(CacheResidencyError::UnsafeShardPath(_))
        ));
        assert!(safe_shard_path(root, "/outside.safetensors").is_err());
    }

    #[test]
    fn malformed_manifest_is_rejected_without_loading_arrays() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("manifest.json"), b"{not-json").unwrap();
        assert!(matches!(
            inspect_prompt_cache(directory.path()),
            Err(CacheResidencyError::ManifestJson(_))
        ));
    }
}
