//! Block-addressable residency for mutable attention state.
//!
//! This module is deliberately independent from weight residency. Attention
//! blocks are mutable activation state until sealed, while checkpoint weights
//! are immutable inputs with a different ownership and persistence model.

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use memmap2::{Mmap, MmapOptions};
use safemlx::{transforms::eval, Array, Device, DeviceType, Dtype, Stream};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::offload::CacheEvictionPolicy;

const PROMPT_CACHE_SCHEMA_VERSION: u32 = 2;
const MAX_PROMPT_CACHE_SHARD_HEADER_BYTES: u64 = 1024 * 1024;
const PROMPT_CACHE_GENERATIONS_DIRECTORY: &str = ".generations";
const PROMPT_CACHE_CURRENT_FILE: &str = "CURRENT";
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_LIVE_SHARD_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_WRITE_RESERVATION_ID: AtomicU64 = AtomicU64::new(1);
static LIVE_PROCESS_NAMESPACE: OnceLock<String> = OnceLock::new();

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
    /// Evaluated CPU-resident state with no execution-device copy retained by the manager.
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
    /// A shared background disk transfer is queued or running.
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

    fn copy_to_stream(
        &self,
        stream: &Stream,
        operation: &'static str,
    ) -> Result<Self, CacheResidencyError> {
        let copy = match self {
            Self::KeyValue { keys, values } => Self::KeyValue {
                keys: keys
                    .copy(stream)
                    .map_err(|source| transfer_error(operation, source))?,
                values: values
                    .copy(stream)
                    .map_err(|source| transfer_error(operation, source))?,
            },
            Self::CompressedLatentRotary { latent, rotary_key } => Self::CompressedLatentRotary {
                latent: latent
                    .copy(stream)
                    .map_err(|source| transfer_error(operation, source))?,
                rotary_key: rotary_key
                    .copy(stream)
                    .map_err(|source| transfer_error(operation, source))?,
            },
        };
        eval(copy.arrays()).map_err(|source| transfer_error(operation, source))?;
        stream
            .synchronize()
            .map_err(|source| transfer_error(operation, source))?;
        // MLX may implement `copy` as an alias when memory is accessible from
        // both streams (notably with Metal's unified memory). Residency tier
        // transitions must own independent storage so replacing the record
        // actually releases the allocation held by the previous tier.
        match copy {
            Self::KeyValue { keys, values } => Ok(Self::KeyValue {
                keys: keys
                    .deep_clone()
                    .map_err(|source| transfer_error(operation, source))?,
                values: values
                    .deep_clone()
                    .map_err(|source| transfer_error(operation, source))?,
            }),
            Self::CompressedLatentRotary { latent, rotary_key } => {
                Ok(Self::CompressedLatentRotary {
                    latent: latent
                        .deep_clone()
                        .map_err(|source| transfer_error(operation, source))?,
                    rotary_key: rotary_key
                        .deep_clone()
                        .map_err(|source| transfer_error(operation, source))?,
                })
            }
        }
    }
}

fn transfer_error(
    operation: &'static str,
    source: safemlx::error::Exception,
) -> CacheResidencyError {
    CacheResidencyError::Runtime(format!("{operation}: {source}"))
}

#[derive(Debug, Clone)]
struct DiskLocation {
    path: PathBuf,
    first_name: String,
    second_name: String,
    persistent: bool,
    mapped: Option<Arc<Mmap>>,
    payload_sha256: Option<String>,
    payload_verification: Arc<OnceLock<Result<(), String>>>,
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
enum DiskOperationKind {
    Write,
    Read,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
struct DiskOperationKey {
    generation: u64,
    id: CacheBlockId,
    kind: DiskOperationKind,
}

enum DiskTask {
    Write {
        directory: PathBuf,
        id: CacheBlockId,
        arrays: CacheBlockArrays,
        commit: Option<DiskWriteCommit>,
    },
    Read {
        location: DiskLocation,
        representation: CacheRepresentation,
    },
    #[cfg(test)]
    Pause {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    },
    #[cfg(test)]
    PauseWrite {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        commit: Option<DiskWriteCommit>,
    },
    #[cfg(test)]
    Panic,
}

enum DiskRequest {
    Operation {
        key: DiskOperationKey,
        task: DiskTask,
        completion: Arc<DiskCompletion>,
    },
    Stop,
}

struct DiskWriteCommit {
    state: Weak<Mutex<CacheManagerState>>,
    key: DiskOperationKey,
    reservation_id: u64,
    armed: bool,
}

#[derive(Debug, Clone)]
struct HostWriteReservation {
    reservation_id: u64,
    global_layer: usize,
    bytes: u64,
    ticket: DiskTicket,
}

#[derive(Debug, Clone)]
enum DiskResult {
    Write(DiskLocation),
    Read(CacheBlockArrays),
    #[cfg(test)]
    Test,
}

impl DiskWriteCommit {
    fn reconcile(&self, result: &Result<DiskResult, CacheResidencyError>) {
        let Some(state) = self.state.upgrade() else {
            if let Ok(DiskResult::Write(location)) = result {
                if !location.persistent {
                    let _ = fs::remove_file(&location.path);
                }
            }
            return;
        };
        let Ok(mut state) = state.lock() else {
            return;
        };
        let stale = state.generation != self.key.generation;
        let mut cleanup = None;
        match result {
            Ok(DiskResult::Write(location)) if !stale => {
                let mut transitioned_to_disk = false;
                let mut bytes = 0;
                if let Some(record) = state.blocks.get_mut(&self.key.id) {
                    if record
                        .pending_disk
                        .as_ref()
                        .is_some_and(|pending| pending.ticket.key == self.key)
                    {
                        record.disk = Some(location.clone());
                        record.pending_disk = None;
                        bytes = record.bytes;
                        if record.tier == CacheTier::Host && record.leases == 0 {
                            record.arrays = None;
                            record.tier = CacheTier::Disk;
                            transitioned_to_disk = true;
                        }
                    }
                }
                if bytes != 0 {
                    state.counters.report.transfer_bytes += bytes;
                    state
                        .layer_activity_mut(self.key.id.global_layer)
                        .transfer_bytes += bytes;
                }
                if transitioned_to_disk {
                    state.counters.report.disk_demotions += 1;
                    state
                        .layer_activity_mut(self.key.id.global_layer)
                        .disk_demotions += 1;
                }
            }
            Ok(DiskResult::Write(location)) => {
                if !location.persistent {
                    cleanup = Some(location.path.clone());
                }
            }
            Ok(_) => {
                state.counters.report.failures += 1;
                state.layer_activity_mut(self.key.id.global_layer).failures += 1;
                state.background_disk_error =
                    Some("cache disk worker returned an unexpected write result".into());
            }
            Err(_) if stale => {}
            Err(error) => {
                if let Some(record) = state.blocks.get_mut(&self.key.id) {
                    if record
                        .pending_disk
                        .as_ref()
                        .is_some_and(|pending| pending.ticket.key == self.key)
                    {
                        record.pending_disk = None;
                    }
                }
                state.counters.report.failures += 1;
                state.layer_activity_mut(self.key.id.global_layer).failures += 1;
                state.background_disk_error = Some(error.to_string());
            }
        }
        update_report_totals(&mut state);
        drop(state);
        if let Some(path) = cleanup {
            let _ = fs::remove_file(path);
        }
    }
}

impl Drop for DiskWriteCommit {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let Ok(mut state) = state.lock() else {
            return;
        };
        if !state
            .host_write_reservations
            .get(&self.key)
            .is_some_and(|reservation| reservation.reservation_id == self.reservation_id)
        {
            return;
        }
        if let Some(record) = state.blocks.get_mut(&self.key.id) {
            if record
                .pending_disk
                .as_ref()
                .is_some_and(|pending| pending.ticket.key == self.key)
            {
                record.pending_disk = None;
            }
        }
        if state.host_write_reservations.remove(&self.key).is_some() {
            update_report_totals(&mut state);
        }
    }
}

#[derive(Debug, Clone)]
enum DiskCompletionState {
    Finished(Result<DiskResult, String>),
    Cancelled,
}

#[derive(Debug, Default)]
struct DiskCompletion {
    state: Mutex<Option<DiskCompletionState>>,
    ready: Condvar,
    released: Mutex<bool>,
    released_ready: Condvar,
}

impl DiskCompletion {
    fn finish(&self, result: Result<DiskResult, CacheResidencyError>) {
        if let Ok(mut state) = self.state.lock() {
            if state.is_none() {
                *state = Some(DiskCompletionState::Finished(
                    result.map_err(|error| error.to_string()),
                ));
                self.ready.notify_all();
            }
        }
    }

    fn cancel(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.is_some() {
            return false;
        }
        *state = Some(DiskCompletionState::Cancelled);
        self.ready.notify_all();
        true
    }

    fn is_ready(&self) -> bool {
        self.state.lock().map_or(true, |state| state.is_some())
    }

    fn wait(&self, generation: u64) -> Result<DiskResult, CacheResidencyError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        while state.is_none() {
            state = self
                .ready
                .wait(state)
                .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        }
        match state.as_ref().expect("completion state was awaited") {
            DiskCompletionState::Finished(Ok(result)) => Ok(result.clone()),
            DiskCompletionState::Finished(Err(error)) => {
                Err(CacheResidencyError::Runtime(error.clone()))
            }
            DiskCompletionState::Cancelled => {
                Err(CacheResidencyError::DiskOperationCancelled { generation })
            }
        }
    }

    fn release_task_resources(&self) {
        if let Ok(mut released) = self.released.lock() {
            *released = true;
            self.released_ready.notify_all();
        }
    }

    fn wait_for_task_resources(&self) -> Result<(), CacheResidencyError> {
        let mut released = self
            .released
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        while !*released {
            released = self
                .released_ready
                .wait(released)
                .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct DiskTicket {
    key: DiskOperationKey,
    completion: Arc<DiskCompletion>,
    shared: Arc<DiskWorkerShared>,
}

impl DiskTicket {
    fn wait(&self) -> Result<DiskResult, CacheResidencyError> {
        self.completion.wait(self.key.generation)
    }

    fn cancel(&self) -> bool {
        let Ok(_space) = self.shared.space.lock() else {
            return false;
        };
        let cancelled = self.completion.cancel();
        self.shared.space_available.notify_all();
        cancelled
    }

    fn wait_for_task_resources(&self) -> Result<(), CacheResidencyError> {
        self.completion.wait_for_task_resources()
    }
}

struct DiskSubmission {
    ticket: DiskTicket,
    sender: SyncSender<DiskRequest>,
    shared: Arc<DiskWorkerShared>,
    unsent: Option<DiskRequest>,
    joined: bool,
    write_reservation_id: Option<u64>,
}

#[derive(Debug)]
struct DiskSubmissionOutcome {
    joined: bool,
    backpressure: bool,
    peak_occupancy: usize,
}

impl DiskSubmission {
    fn enqueue(mut self) -> Result<DiskSubmissionOutcome, CacheResidencyError> {
        let mut backpressure = false;
        if let Some(mut request) = self.unsent.take() {
            let mut space = match self.shared.space.lock() {
                Ok(space) => space,
                Err(_) => {
                    drop(request);
                    self.ticket.completion.release_task_resources();
                    return Err(CacheResidencyError::ManagerPoisoned);
                }
            };
            loop {
                if self.ticket.completion.is_ready() {
                    drop(request);
                    self.ticket.completion.release_task_resources();
                    break;
                }
                match self.sender.try_send(request) {
                    Ok(()) => {
                        let occupancy = self.shared.queued.fetch_add(1, Ordering::AcqRel) + 1;
                        update_atomic_max(
                            &self.shared.peak_occupancy,
                            occupancy.min(self.shared.capacity),
                        );
                        break;
                    }
                    Err(TrySendError::Full(returned)) => {
                        request = returned;
                        backpressure = true;
                        space = match self.shared.space_available.wait(space) {
                            Ok(space) => space,
                            Err(_) => {
                                drop(request);
                                self.ticket.completion.release_task_resources();
                                return Err(CacheResidencyError::ManagerPoisoned);
                            }
                        };
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        self.ticket
                            .completion
                            .finish(Err(CacheResidencyError::Runtime(
                                "live cache disk worker stopped".into(),
                            )));
                        self.ticket.completion.release_task_resources();
                        break;
                    }
                }
            }
        }
        Ok(DiskSubmissionOutcome {
            joined: self.joined,
            backpressure,
            peak_occupancy: self.shared.peak_occupancy.load(Ordering::Acquire),
        })
    }
}

impl Drop for DiskSubmission {
    fn drop(&mut self) {
        let Some(request) = self.unsent.take() else {
            return;
        };
        self.ticket.cancel();
        drop(request);
        self.ticket.completion.release_task_resources();
        retire_disk_completion(&self.shared, &self.ticket.key, &self.ticket.completion);
    }
}

#[derive(Debug)]
struct DiskWorkerShared {
    in_flight: Mutex<HashMap<DiskOperationKey, Arc<DiskCompletion>>>,
    space: Mutex<()>,
    space_available: Condvar,
    queued: AtomicUsize,
    peak_occupancy: AtomicUsize,
    capacity: usize,
}

impl DiskWorkerShared {
    fn new(capacity: usize) -> Self {
        Self {
            in_flight: Mutex::new(HashMap::new()),
            space: Mutex::new(()),
            space_available: Condvar::new(),
            queued: AtomicUsize::new(0),
            peak_occupancy: AtomicUsize::new(0),
            capacity,
        }
    }
}

fn retire_disk_completion(
    shared: &DiskWorkerShared,
    key: &DiskOperationKey,
    completion: &Arc<DiskCompletion>,
) {
    if let Ok(mut in_flight) = shared.in_flight.lock() {
        if in_flight
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, completion))
        {
            in_flight.remove(key);
        }
    }
}

#[derive(Debug)]
struct DiskWorker {
    sender: SyncSender<DiskRequest>,
    handle: Mutex<Option<JoinHandle<()>>>,
    shared: Arc<DiskWorkerShared>,
}

impl DiskWorker {
    fn new(capacity: usize) -> Result<Self, CacheResidencyError> {
        let (sender, receiver) = mpsc::sync_channel::<DiskRequest>(capacity);
        let shared = Arc::new(DiskWorkerShared::new(capacity));
        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("safemlx-cache-disk".into())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    match request {
                        DiskRequest::Operation {
                            key,
                            task,
                            completion,
                        } => {
                            if let Ok(_space) = worker_shared.space.lock() {
                                worker_shared.queued.fetch_sub(1, Ordering::AcqRel);
                                worker_shared.space_available.notify_all();
                            }
                            if completion.is_ready() {
                                drop(task);
                                completion.release_task_resources();
                                retire_disk_completion(&worker_shared, &key, &completion);
                                continue;
                            }
                            let mut write_commit = None;
                            let result = catch_unwind(AssertUnwindSafe(|| match task {
                                DiskTask::Write {
                                    directory,
                                    id,
                                    arrays,
                                    commit,
                                } => {
                                    write_commit = commit;
                                    write_live_block(&directory, &id, &arrays)
                                        .map(DiskResult::Write)
                                }
                                DiskTask::Read {
                                    location,
                                    representation,
                                } => load_block_arrays_direct(&location, representation)
                                    .map(DiskResult::Read),
                                #[cfg(test)]
                                DiskTask::Pause { started, release } => {
                                    let _ = started.send(());
                                    let _ = release.recv();
                                    Ok(DiskResult::Test)
                                }
                                #[cfg(test)]
                                DiskTask::PauseWrite {
                                    started,
                                    release,
                                    commit,
                                } => {
                                    write_commit = commit;
                                    let _ = started.send(());
                                    let _ = release.recv();
                                    Err(CacheResidencyError::Runtime(
                                        "injected canceled cache write".into(),
                                    ))
                                }
                                #[cfg(test)]
                                DiskTask::Panic => panic!("injected cache disk worker panic"),
                            }))
                            .unwrap_or_else(|_| {
                                Err(CacheResidencyError::Runtime(
                                    "live cache disk worker operation panicked".into(),
                                ))
                            });
                            if let Some(commit) = &write_commit {
                                commit.reconcile(&result);
                            }
                            // The task-local arrays have been dropped by this
                            // point. Release their reservation before waking
                            // logical completion waiters.
                            drop(write_commit);
                            if completion.is_ready() {
                                if let Ok(DiskResult::Write(location)) = result {
                                    if !location.persistent {
                                        let _ = fs::remove_file(location.path);
                                    }
                                }
                            } else {
                                completion.finish(result);
                            }
                            completion.release_task_resources();
                            retire_disk_completion(&worker_shared, &key, &completion);
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
            shared,
        })
    }

    fn prepare(
        &self,
        key: DiskOperationKey,
        task: DiskTask,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        self.prepare_with_write_reservation(key, task, None)
    }

    fn prepare_with_write_reservation(
        &self,
        key: DiskOperationKey,
        mut task: DiskTask,
        write_reservation_id: Option<u64>,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        let mut in_flight = self
            .shared
            .in_flight
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        if let Some(completion) = in_flight.get(&key) {
            if let DiskTask::Write { commit, .. } = &mut task {
                if let Some(commit) = commit {
                    commit.armed = false;
                }
            }
            return Ok(DiskSubmission {
                ticket: DiskTicket {
                    key,
                    completion: Arc::clone(completion),
                    shared: Arc::clone(&self.shared),
                },
                sender: self.sender.clone(),
                shared: Arc::clone(&self.shared),
                unsent: None,
                joined: true,
                write_reservation_id: None,
            });
        }
        let completion = Arc::new(DiskCompletion::default());
        in_flight.insert(key.clone(), Arc::clone(&completion));
        drop(in_flight);
        let request = DiskRequest::Operation {
            key: key.clone(),
            task,
            completion: Arc::clone(&completion),
        };
        Ok(DiskSubmission {
            ticket: DiskTicket {
                key,
                completion,
                shared: Arc::clone(&self.shared),
            },
            sender: self.sender.clone(),
            shared: Arc::clone(&self.shared),
            unsent: Some(request),
            joined: false,
            write_reservation_id,
        })
    }

    fn prepare_write(
        &self,
        generation: u64,
        directory: &Path,
        id: &CacheBlockId,
        arrays: &CacheBlockArrays,
        state: Weak<Mutex<CacheManagerState>>,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        let reservation_id = NEXT_HOST_WRITE_RESERVATION_ID.fetch_add(1, Ordering::Relaxed);
        self.prepare_with_write_reservation(
            DiskOperationKey {
                generation,
                id: id.clone(),
                kind: DiskOperationKind::Write,
            },
            DiskTask::Write {
                directory: directory.to_path_buf(),
                id: id.clone(),
                arrays: arrays.clone(),
                commit: Some(DiskWriteCommit {
                    state,
                    key: DiskOperationKey {
                        generation,
                        id: id.clone(),
                        kind: DiskOperationKind::Write,
                    },
                    reservation_id,
                    armed: true,
                }),
            },
            Some(reservation_id),
        )
    }

    fn prepare_read(
        &self,
        generation: u64,
        id: &CacheBlockId,
        location: &DiskLocation,
        representation: CacheRepresentation,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        self.prepare(
            DiskOperationKey {
                generation,
                id: id.clone(),
                kind: DiskOperationKind::Read,
            },
            DiskTask::Read {
                location: location.clone(),
                representation,
            },
        )
    }

    fn retire(&self, ticket: &DiskTicket) {
        retire_disk_completion(&self.shared, &ticket.key, &ticket.completion);
    }
}

fn update_atomic_max(target: &AtomicUsize, value: usize) {
    let mut current = target.load(Ordering::Acquire);
    while value > current {
        match target.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
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
struct PendingDiskOperation {
    ticket: DiskTicket,
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
    pending_disk: Option<PendingDiskOperation>,
}

/// Maximum number of individually identified layers in a residency report.
///
/// Additional active layers are folded into
/// [`CacheResidencyReport::per_layer_overflow`], so report size is independent
/// of caller-provided layer identifiers and remains bounded.
pub const CACHE_RESIDENCY_LAYER_REPORT_LIMIT: usize = 128;

/// Current residency and cumulative activity attributable to one layer or a
/// bounded overflow group of layers.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheLayerResidencyStats {
    /// Logical cached tokens. For an overflow aggregate this is the sum of the
    /// per-layer logical token counts rather than a shared sequence length.
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
    /// Current logical host bytes, including arrays retained by in-flight writes.
    pub current_host_bytes: u64,
    /// Current logical disk bytes.
    pub current_disk_bytes: u64,
    /// Current bytes in mutable tails.
    pub mutable_tail_bytes: u64,
    /// Blocks whose host staging arrays are owned by background disk writes.
    pub in_flight_write_blocks: u64,
    /// Logical bytes owned by background disk writes.
    pub in_flight_write_bytes: u64,
    /// Recent device blocks protected from demotion.
    pub protected_recent_blocks: u64,
    /// Prefix or sink blocks protected for attention.
    pub protected_prefix_blocks: u64,
    /// Cumulative promotions from host memory to the execution device.
    pub host_promotions: u64,
    /// Cumulative promotions from disk to the execution device.
    pub disk_promotions: u64,
    /// Cumulative demotions from the execution device to host memory.
    pub host_demotions: u64,
    /// Cumulative demotions to disk.
    pub disk_demotions: u64,
    /// Cumulative logical bytes transferred between tiers.
    pub transfer_bytes: u64,
    /// Cumulative time spent waiting for layer-attributable transfers.
    pub transfer_wait: Duration,
    /// Cumulative demand accesses already resident on the execution device.
    pub demand_hits: u64,
    /// Cumulative demand accesses that required promotion.
    pub demand_misses: u64,
    /// Cumulative waits that joined or awaited an in-flight disk operation.
    pub in_flight_waits: u64,
    /// Cumulative layer-attributable residency or transfer failures.
    pub failures: u64,
    /// Sealed and mutable blocks scanned by full attention during prefill.
    pub prefill_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during prefill.
    pub prefill_full_attention_bytes: u64,
    /// Sealed and mutable blocks scanned by full attention during decode.
    pub decode_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during decode.
    pub decode_full_attention_bytes: u64,
    /// Peak logical scratch bytes used by this layer's attention.
    pub attention_scratch_peak_bytes: u64,
}

impl CacheLayerResidencyStats {
    fn accumulate(&mut self, other: &Self) {
        self.logical_cached_tokens += other.logical_cached_tokens;
        self.key_value_blocks += other.key_value_blocks;
        self.compressed_latent_blocks += other.compressed_latent_blocks;
        self.device_blocks += other.device_blocks;
        self.host_blocks += other.host_blocks;
        self.disk_blocks += other.disk_blocks;
        self.current_device_bytes += other.current_device_bytes;
        self.current_host_bytes += other.current_host_bytes;
        self.current_disk_bytes += other.current_disk_bytes;
        self.mutable_tail_bytes += other.mutable_tail_bytes;
        self.in_flight_write_blocks += other.in_flight_write_blocks;
        self.in_flight_write_bytes += other.in_flight_write_bytes;
        self.protected_recent_blocks += other.protected_recent_blocks;
        self.protected_prefix_blocks += other.protected_prefix_blocks;
        self.host_promotions += other.host_promotions;
        self.disk_promotions += other.disk_promotions;
        self.host_demotions += other.host_demotions;
        self.disk_demotions += other.disk_demotions;
        self.transfer_bytes += other.transfer_bytes;
        self.transfer_wait += other.transfer_wait;
        self.demand_hits += other.demand_hits;
        self.demand_misses += other.demand_misses;
        self.in_flight_waits += other.in_flight_waits;
        self.failures += other.failures;
        self.prefill_full_attention_blocks += other.prefill_full_attention_blocks;
        self.prefill_full_attention_bytes += other.prefill_full_attention_bytes;
        self.decode_full_attention_blocks += other.decode_full_attention_blocks;
        self.decode_full_attention_bytes += other.decode_full_attention_bytes;
        self.attention_scratch_peak_bytes = self
            .attention_scratch_peak_bytes
            .max(other.attention_scratch_peak_bytes);
    }
}

#[derive(Debug, Clone, Default)]
struct CacheLayerActivityCounters {
    stats: CacheLayerResidencyStats,
}

impl CacheLayerActivityCounters {
    fn apply_to(&self, stats: &mut CacheLayerResidencyStats) {
        stats.host_promotions += self.stats.host_promotions;
        stats.disk_promotions += self.stats.disk_promotions;
        stats.host_demotions += self.stats.host_demotions;
        stats.disk_demotions += self.stats.disk_demotions;
        stats.transfer_bytes += self.stats.transfer_bytes;
        stats.transfer_wait += self.stats.transfer_wait;
        stats.demand_hits += self.stats.demand_hits;
        stats.demand_misses += self.stats.demand_misses;
        stats.in_flight_waits += self.stats.in_flight_waits;
        stats.failures += self.stats.failures;
        stats.prefill_full_attention_blocks += self.stats.prefill_full_attention_blocks;
        stats.prefill_full_attention_bytes += self.stats.prefill_full_attention_bytes;
        stats.decode_full_attention_blocks += self.stats.decode_full_attention_blocks;
        stats.decode_full_attention_bytes += self.stats.decode_full_attention_bytes;
        stats.attention_scratch_peak_bytes = stats
            .attention_scratch_peak_bytes
            .max(self.stats.attention_scratch_peak_bytes);
    }
}

/// Bounded, individually identified per-layer residency observations.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheLayerResidencyReport {
    /// Global model layer identifier.
    pub global_layer: usize,
    /// Current residency and cumulative activity for this layer.
    pub stats: CacheLayerResidencyStats,
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
    /// Peak successfully admitted logical device bytes.
    pub peak_device_bytes: u64,
    /// Current logical host bytes, including arrays retained by in-flight writes.
    pub current_host_bytes: u64,
    /// Peak successfully admitted logical host bytes.
    pub peak_host_bytes: u64,
    /// Current logical disk bytes.
    pub current_disk_bytes: u64,
    /// Peak successfully admitted logical disk bytes.
    pub peak_disk_bytes: u64,
    /// Blocks whose host staging arrays are owned by background disk writes.
    pub in_flight_write_blocks: u64,
    /// Logical bytes owned by background disk writes (included in current host bytes).
    pub in_flight_write_bytes: u64,
    /// Peak logical bytes owned by background disk writes.
    pub peak_in_flight_write_bytes: u64,
    /// Current bytes in mutable tails.
    pub mutable_tail_bytes: u64,
    /// Recent blocks protected from device demotion.
    pub protected_recent_blocks: u64,
    /// Prefix or sink blocks protected for attention.
    pub protected_prefix_blocks: u64,
    /// Current residency and cumulative per-layer activity, sorted by global
    /// layer and capped by [`CACHE_RESIDENCY_LAYER_REPORT_LIMIT`].
    pub per_layer: Vec<CacheLayerResidencyReport>,
    /// Number of active layers folded into `per_layer_overflow`.
    pub per_layer_overflow_layers: u64,
    /// Exact current aggregate of active omitted layers plus cumulative
    /// activity for every layer without an identified row.
    pub per_layer_overflow: CacheLayerResidencyStats,
    /// Completed synchronized host-to-device promotions.
    pub host_promotions: u64,
    /// Completed synchronized disk-to-device promotions.
    pub disk_promotions: u64,
    /// Completed synchronized device-to-host demotions.
    pub host_demotions: u64,
    /// Completed resident-to-disk demotions using existing or newly written backing.
    pub disk_demotions: u64,
    /// Logical bytes copied by promotion and demotion operations.
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
    /// Effective bounded disk request queue capacity.
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
    background_disk_error: Option<String>,
    access_clock: u64,
    blocks: BTreeMap<CacheBlockId, CacheBlockRecord>,
    tails: HashMap<usize, (u64, i64)>,
    host_write_reservations: HashMap<DiskOperationKey, HostWriteReservation>,
    layer_activity: BTreeMap<usize, CacheLayerActivityCounters>,
    layer_activity_overflow: CacheLayerActivityCounters,
    counters: CacheCounters,
    recent_device_blocks: usize,
    device_budget_bytes: u64,
    host_budget_bytes: u64,
    disk_budget_bytes: Option<u64>,
    host_stream: Stream,
}

// SAFETY: all access to the CPU stream and cache arrays is serialized by the
// manager mutex. MLX operations additionally use safemlx's runtime guard.
unsafe impl Send for CacheManagerState {}

impl CacheManagerState {
    fn layer_activity_mut(&mut self, global_layer: usize) -> &mut CacheLayerResidencyStats {
        if self.layer_activity.contains_key(&global_layer)
            || self.layer_activity.len() < CACHE_RESIDENCY_LAYER_REPORT_LIMIT
        {
            &mut self.layer_activity.entry(global_layer).or_default().stats
        } else {
            &mut self.layer_activity_overflow.stats
        }
    }
}

/// Shared architecture-independent manager enforcing budgets across all layers.
#[derive(Debug, Clone)]
pub struct CacheResidencyManager {
    session_id: u64,
    options: PagedCacheOptions,
    state: Arc<Mutex<CacheManagerState>>,
    disk_worker: Option<Arc<DiskWorker>>,
}

enum HostDemotionProgress {
    Retry,
    Freed,
    Pending(DiskTicket),
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
        let effective_queue_capacity = queue_capacity.max(1);
        let mut counters = CacheCounters::default();
        counters.report.queue_capacity = effective_queue_capacity;
        let disk_worker = Some(Arc::new(DiskWorker::new(effective_queue_capacity)?));
        let recent_device_blocks = options.recent_device_blocks;
        let device_budget_bytes = options.device_budget_bytes;
        let host_budget_bytes = options.host_budget_bytes;
        let disk_budget_bytes = match &options.live_disk {
            LiveCacheDiskPolicy::Disabled => None,
            LiveCacheDiskPolicy::Enabled { budget_bytes, .. } => Some(*budget_bytes),
        };
        Ok(Self {
            session_id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
            options,
            state: Arc::new(Mutex::new(CacheManagerState {
                generation: 0,
                background_disk_error: None,
                access_clock: 0,
                blocks: BTreeMap::new(),
                tails: HashMap::new(),
                host_write_reservations: HashMap::new(),
                layer_activity: BTreeMap::new(),
                layer_activity_overflow: CacheLayerActivityCounters::default(),
                counters,
                recent_device_blocks,
                device_budget_bytes,
                host_budget_bytes,
                disk_budget_bytes,
                host_stream: cpu_stream(),
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
        drop(state);
        if let Err(error) = self.rebalance(None) {
            let mut state = self.lock()?;
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
            pending_disk: None,
        };
        state.blocks.insert(id.clone(), record);
        drop(state);
        if let Err(error) = self.rebalance(Some(&id)) {
            let mut state = self.lock()?;
            if let Some(record) = state.blocks.remove(&id) {
                cancel_record_operation(&record, &mut state.counters.report);
                remove_ephemeral_file(&record);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        let mut state = self.lock()?;
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
        let tickets = advance_generation_locked(&mut state);
        let record = state
            .blocks
            .remove(id)
            .expect("validated cache block still present");
        remove_ephemeral_file(&record);
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    pub(crate) fn truncate_layer_transaction(
        &self,
        global_layer: usize,
        representation: CacheRepresentation,
        end: i64,
        replacement: Option<(CacheBlockId, CacheBlockArrays)>,
        protected_prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if end < 0 {
            return Err(CacheResidencyError::InvalidTokenRange { start: 0, end });
        }
        if let Some((old_id, arrays)) = &replacement {
            if old_id.global_layer != global_layer
                || old_id.representation != representation
                || old_id.start >= end
                || old_id.end <= end
                || arrays.representation() != representation
            {
                return Err(CacheResidencyError::ArrayMismatch(
                    "trailing cache replacement does not match the truncated layer".into(),
                ));
            }
            validate_block_arrays(arrays, end - old_id.start)?;
            eval(arrays.arrays())
                .map_err(|source| CacheResidencyError::Runtime(source.to_string()))?;
        }

        let mut state = self.lock()?;
        if let Some(error) = state.background_disk_error.take() {
            return Err(CacheResidencyError::Runtime(format!(
                "background cache disk write failed: {error}"
            )));
        }
        let affected = state
            .blocks
            .keys()
            .filter(|id| {
                id.global_layer == global_layer
                    && id.representation == representation
                    && id.end > end
            })
            .cloned()
            .collect::<Vec<_>>();
        let crossing = affected.iter().find(|id| id.start < end);
        match (crossing, replacement.as_ref()) {
            (Some(crossing), Some((old_id, _))) if crossing == old_id => {}
            (None, None) => {}
            _ => {
                return Err(CacheResidencyError::ArrayMismatch(
                    "trailing cache replacement does not match the block crossing the truncation boundary"
                        .into(),
                ))
            }
        }
        for id in &affected {
            let record = state
                .blocks
                .get(id)
                .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
            let owned_replacement_lease =
                replacement.as_ref().is_some_and(|(old_id, _)| old_id == id);
            let expected_leases = usize::from(owned_replacement_lease);
            if record.leases != expected_leases {
                return Err(CacheResidencyError::BlockLeased(id.clone()));
            }
            if owned_replacement_lease && record.tier != CacheTier::Device {
                return Err(CacheResidencyError::Runtime(
                    "truncated cache replacement lease is not device resident".into(),
                ));
            }
        }

        let replacement_id = replacement.as_ref().map(|(old_id, _)| CacheBlockId {
            session_id: self.session_id,
            global_layer,
            representation,
            start: old_id.start,
            end,
            rank: old_id.rank,
        });
        if let Some(id) = &replacement_id {
            if state.blocks.contains_key(id) && !affected.contains(id) {
                return Err(CacheResidencyError::DuplicateBlock(id.clone()));
            }
        }

        let tickets = advance_generation_locked(&mut state);
        let mut removed = Vec::with_capacity(affected.len());
        for id in &affected {
            if let Some(record) = state.blocks.remove(id) {
                removed.push(record);
            }
        }
        state.tails.insert(global_layer, (0, end));
        if let Some((old_id, arrays)) = replacement {
            state.access_clock += 1;
            let id = replacement_id.expect("validated replacement id is available");
            debug_assert_eq!(id.start, old_id.start);
            let record = CacheBlockRecord {
                id: id.clone(),
                tier: CacheTier::Device,
                shapes: arrays.shapes(),
                dtypes: arrays.dtypes(),
                bytes: arrays.bytes(),
                arrays: Some(arrays),
                disk: None,
                imported: false,
                leases: 0,
                access_count: 0,
                last_access: state.access_clock,
                protected_prefix: end <= protected_prefix_tokens,
                pending_disk: None,
            };
            let previous = state.blocks.insert(id, record);
            debug_assert!(previous.is_none());
            state.counters.report.block_seals += 1;
        }
        update_report_totals(&mut state);
        drop(state);

        for record in &removed {
            remove_ephemeral_file(record);
        }
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    pub(crate) fn lease_block(
        &self,
        id: &CacheBlockId,
        stream: &Stream,
    ) -> Result<CacheBlockLease, CacheResidencyError> {
        let started = Instant::now();
        let mut loaded_from_disk = false;
        loop {
            let mut state = self.lock()?;
            let generation = state.generation;
            let (tier, source_arrays, disk, pending) = {
                let record = state
                    .blocks
                    .get(id)
                    .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                (
                    record.tier,
                    record.arrays.clone(),
                    record.disk.clone(),
                    record.pending_disk.clone(),
                )
            };

            if tier == CacheTier::Disk {
                let worker = self.disk_worker.as_ref().ok_or_else(|| {
                    CacheResidencyError::Runtime("cache disk worker is unavailable".into())
                })?;
                let (ticket, submission, joined) = if let Some(pending) = pending {
                    (pending.ticket, None, true)
                } else {
                    let location = disk
                        .as_ref()
                        .ok_or_else(|| CacheResidencyError::MissingDiskLocation(id.clone()))?;
                    let submission =
                        worker.prepare_read(generation, id, location, id.representation)?;
                    let ticket = submission.ticket.clone();
                    state
                        .blocks
                        .get_mut(id)
                        .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?
                        .pending_disk = Some(PendingDiskOperation {
                        ticket: ticket.clone(),
                    });
                    (ticket, Some(submission), false)
                };
                drop(state);

                let outcome = match submission {
                    Some(submission) => Some(submission.enqueue()?),
                    None => None,
                };
                let result = ticket.wait();
                let mut state = self.lock()?;
                if joined || outcome.as_ref().is_some_and(|outcome| outcome.joined) {
                    state.counters.report.in_flight_waits += 1;
                    state.layer_activity_mut(id.global_layer).in_flight_waits += 1;
                }
                if let Some(outcome) = &outcome {
                    state.counters.report.queue_peak_occupancy = state
                        .counters
                        .report
                        .queue_peak_occupancy
                        .max(outcome.peak_occupancy);
                    state.counters.report.queue_backpressure += u64::from(outcome.backpressure);
                }
                let stale = state.generation != ticket.key.generation;
                match result {
                    Ok(DiskResult::Read(arrays)) if !stale => {
                        let record = state
                            .blocks
                            .get_mut(id)
                            .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                        if arrays.shapes() != record.shapes || arrays.dtypes() != record.dtypes {
                            let path = disk.expect("disk-backed record has a location").path;
                            record.pending_disk = None;
                            drop(state);
                            worker.retire(&ticket);
                            return Err(CacheResidencyError::MalformedShard {
                                path,
                                reason: "array shape or dtype does not match the manifest".into(),
                            });
                        }
                        if record
                            .pending_disk
                            .as_ref()
                            .is_some_and(|pending| pending.ticket.key == ticket.key)
                        {
                            record.arrays = Some(arrays);
                            record.tier = CacheTier::Host;
                            record.pending_disk = None;
                        }
                        loaded_from_disk = true;
                    }
                    Ok(DiskResult::Read(_))
                    | Err(CacheResidencyError::DiskOperationCancelled { .. })
                        if stale =>
                    {
                        drop(state);
                        worker.retire(&ticket);
                        return Err(CacheResidencyError::DiskOperationCancelled {
                            generation: ticket.key.generation,
                        });
                    }
                    Ok(_) => {
                        drop(state);
                        worker.retire(&ticket);
                        return Err(CacheResidencyError::Runtime(
                            "cache disk worker returned an unexpected operation result".into(),
                        ));
                    }
                    Err(error) => {
                        if let Some(record) = state.blocks.get_mut(id) {
                            if record
                                .pending_disk
                                .as_ref()
                                .is_some_and(|pending| pending.ticket.key == ticket.key)
                            {
                                record.pending_disk = None;
                            }
                        }
                        state.counters.report.failures += 1;
                        state.layer_activity_mut(id.global_layer).failures += 1;
                        drop(state);
                        worker.retire(&ticket);
                        return Err(error);
                    }
                }
                drop(state);
                worker.retire(&ticket);
                continue;
            }

            let source_arrays = source_arrays
                .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?;
            if tier == CacheTier::Device {
                state.access_clock += 1;
                let access_clock = state.access_clock;
                let record = state
                    .blocks
                    .get_mut(id)
                    .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                record.leases += 1;
                record.access_count += 1;
                record.last_access = access_clock;
                state.counters.report.demand_hits += 1;
                let transfer_wait = started.elapsed();
                state.counters.report.transfer_wait += transfer_wait;
                let activity = state.layer_activity_mut(id.global_layer);
                activity.demand_hits += 1;
                activity.transfer_wait += transfer_wait;
                drop(state);
                if let Err(error) = self.rebalance(Some(id)) {
                    if let Ok(mut state) = self.lock() {
                        if let Some(record) = state.blocks.get_mut(id) {
                            record.leases = record.leases.saturating_sub(1);
                        }
                        update_report_totals(&mut state);
                    }
                    return Err(error);
                }
                return Ok(CacheBlockLease {
                    id: id.clone(),
                    arrays: source_arrays,
                    manager: self.clone(),
                    released: false,
                });
            }

            drop(state);
            let device_arrays =
                source_arrays.copy_to_stream(stream, "copy cache block from host to device")?;
            let mut state = self.lock()?;
            if state.generation != generation {
                return Err(CacheResidencyError::DiskOperationCancelled { generation });
            }
            state.access_clock += 1;
            let access_clock = state.access_clock;
            let bytes = {
                let record = state
                    .blocks
                    .get_mut(id)
                    .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                record.arrays = Some(device_arrays.clone());
                record.tier = CacheTier::Device;
                record.leases += 1;
                record.access_count += 1;
                record.last_access = access_clock;
                record.bytes
            };
            state.counters.report.demand_misses += 1;
            if loaded_from_disk {
                state.counters.report.disk_promotions += 1;
            } else {
                state.counters.report.host_promotions += 1;
            }
            state.counters.report.transfer_bytes += bytes;
            let transfer_wait = started.elapsed();
            state.counters.report.transfer_wait += transfer_wait;
            let activity = state.layer_activity_mut(id.global_layer);
            activity.demand_misses += 1;
            if loaded_from_disk {
                activity.disk_promotions += 1;
            } else {
                activity.host_promotions += 1;
            }
            activity.transfer_bytes += bytes;
            activity.transfer_wait += transfer_wait;
            drop(state);
            if let Err(error) = self.rebalance(Some(id)) {
                let mut state = self.lock()?;
                if state.generation == generation {
                    if let Some(record) = state.blocks.get_mut(id) {
                        record.leases = record.leases.saturating_sub(1);
                        record.tier = CacheTier::Host;
                        record.arrays = Some(source_arrays);
                    }
                    update_report_totals(&mut state);
                }
                return Err(error);
            }
            return Ok(CacheBlockLease {
                id: id.clone(),
                arrays: device_arrays,
                manager: self.clone(),
                released: false,
            });
        }
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
        let tickets = if ids.is_empty() {
            Vec::new()
        } else {
            advance_generation_locked(&mut state)
        };
        for id in ids {
            if let Some(record) = state.blocks.remove(&id) {
                remove_ephemeral_file(&record);
                state.counters.report.discarded_sliding_blocks += 1;
            }
        }
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
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
        let tickets = advance_generation_locked(&mut state);
        for record in state.blocks.values() {
            remove_ephemeral_file(record);
        }
        state.blocks.clear();
        state.tails.clear();
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
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

    fn retire_tickets(&self, tickets: &[DiskTicket]) -> Result<(), CacheResidencyError> {
        if let Some(worker) = &self.disk_worker {
            for ticket in tickets {
                ticket.wait_for_task_resources()?;
                worker.retire(ticket);
            }
        }
        Ok(())
    }

    pub(crate) fn record_attention_scan(
        &self,
        global_layer: usize,
        prefill: bool,
        blocks: u64,
        bytes: u64,
        scratch_bytes: u64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if prefill {
            state.counters.report.prefill_full_attention_blocks += blocks;
            state.counters.report.prefill_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.prefill_full_attention_blocks += blocks;
            activity.prefill_full_attention_bytes += bytes;
        } else {
            state.counters.report.decode_full_attention_blocks += blocks;
            state.counters.report.decode_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.decode_full_attention_blocks += blocks;
            activity.decode_full_attention_bytes += bytes;
        }
        state.counters.report.attention_scratch_peak_bytes = state
            .counters
            .report
            .attention_scratch_peak_bytes
            .max(scratch_bytes);
        let activity = state.layer_activity_mut(global_layer);
        activity.attention_scratch_peak_bytes =
            activity.attention_scratch_peak_bytes.max(scratch_bytes);
        Ok(())
    }

    fn release_lease(&self, id: &CacheBlockId) {
        let mut background_failed = false;
        if let Ok(mut state) = self.state.lock() {
            if let Some(record) = state.blocks.get_mut(id) {
                record.leases = record.leases.saturating_sub(1);
            }
            background_failed = state.background_disk_error.is_some();
        }
        if !background_failed {
            let _ = self.rebalance(None);
        }
    }

    fn begin_host_demotion(
        &self,
        id: &CacheBlockId,
    ) -> Result<HostDemotionProgress, CacheResidencyError> {
        let mut state = self.lock()?;
        let Some(record) = state.blocks.get(id) else {
            return Ok(HostDemotionProgress::Retry);
        };
        if record.tier != CacheTier::Host || record.leases != 0 || record.pending_disk.is_some() {
            return Ok(HostDemotionProgress::Retry);
        }

        // Persistent prompt-cache blocks and completed live-cache writes can be
        // released immediately; they do not require live writeback to be enabled.
        if record.disk.is_some() {
            let record = state.blocks.get_mut(id).expect("host block exists");
            record.arrays = None;
            record.tier = CacheTier::Disk;
            state.counters.report.disk_demotions += 1;
            state.layer_activity_mut(id.global_layer).disk_demotions += 1;
            update_report_totals(&mut state);
            return Ok(HostDemotionProgress::Freed);
        }

        let (directory, budget_bytes) = match &self.options.live_disk {
            LiveCacheDiskPolicy::Disabled => {
                state.counters.report.failures += 1;
                state.layer_activity_mut(id.global_layer).failures += 1;
                return Err(CacheResidencyError::LiveDiskRequired {
                    required: state.counters.report.current_host_bytes,
                    budget: self.options.host_budget_bytes,
                });
            }
            LiveCacheDiskPolicy::Enabled {
                directory,
                budget_bytes,
                ..
            } => (directory.clone(), *budget_bytes),
        };
        let worker = self.disk_worker.as_ref().ok_or_else(|| {
            CacheResidencyError::Runtime("live cache disk worker is unavailable".into())
        })?;
        let record = state.blocks.get(id).expect("host block exists");
        let live_disk_bytes = state
            .blocks
            .values()
            .filter(|record| {
                record
                    .disk
                    .as_ref()
                    .is_some_and(|location| !location.persistent)
            })
            .map(|record| record.bytes)
            .sum::<u64>()
            .saturating_add(
                state
                    .host_write_reservations
                    .values()
                    .map(|reservation| reservation.bytes)
                    .sum(),
            );
        let projected = live_disk_bytes.saturating_add(record.bytes);
        if projected > budget_bytes {
            state.counters.report.failures += 1;
            state.layer_activity_mut(id.global_layer).failures += 1;
            return Err(CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Disk,
                required: projected,
                budget: budget_bytes,
            });
        }
        let arrays = record
            .arrays
            .as_ref()
            .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?
            .clone();
        let submission = worker.prepare_write(
            state.generation,
            &directory,
            id,
            &arrays,
            Arc::downgrade(&self.state),
        )?;
        let ticket = submission.ticket.clone();
        let record_bytes = record.bytes;
        if let Some(reservation_id) = submission.write_reservation_id {
            state.host_write_reservations.insert(
                ticket.key.clone(),
                HostWriteReservation {
                    reservation_id,
                    global_layer: id.global_layer,
                    bytes: record_bytes,
                    ticket: ticket.clone(),
                },
            );
        }
        state
            .blocks
            .get_mut(id)
            .expect("host block exists")
            .pending_disk = Some(PendingDiskOperation {
            ticket: ticket.clone(),
        });
        update_report_totals(&mut state);
        drop(state);

        let enqueue_started = Instant::now();
        let outcome = match submission.enqueue() {
            Ok(outcome) => outcome,
            Err(error) => {
                let mut state = self.lock()?;
                if let Some(record) = state.blocks.get_mut(&ticket.key.id) {
                    if record
                        .pending_disk
                        .as_ref()
                        .is_some_and(|pending| pending.ticket.key == ticket.key)
                    {
                        record.pending_disk = None;
                    }
                }
                state.counters.report.failures += 1;
                state
                    .layer_activity_mut(ticket.key.id.global_layer)
                    .failures += 1;
                update_report_totals(&mut state);
                drop(state);
                worker.retire(&ticket);
                return Err(error);
            }
        };
        let enqueue_wait = enqueue_started.elapsed();
        let mut state = self.lock()?;
        if outcome.joined {
            state.counters.report.in_flight_waits += 1;
            state
                .layer_activity_mut(ticket.key.id.global_layer)
                .in_flight_waits += 1;
        }
        state.counters.report.queue_peak_occupancy = state
            .counters
            .report
            .queue_peak_occupancy
            .max(outcome.peak_occupancy);
        state.counters.report.queue_backpressure += u64::from(outcome.backpressure);
        if outcome.backpressure {
            state.counters.report.transfer_wait += enqueue_wait;
            state
                .layer_activity_mut(ticket.key.id.global_layer)
                .transfer_wait += enqueue_wait;
        }
        update_report_totals(&mut state);
        Ok(HostDemotionProgress::Pending(ticket))
    }

    fn wait_for_host_release(&self, ticket: &DiskTicket) -> Result<(), CacheResidencyError> {
        let started = Instant::now();
        let result = ticket.wait();
        ticket.wait_for_task_resources()?;
        let elapsed = started.elapsed();
        let mut state = self.lock()?;
        state.counters.report.in_flight_waits += 1;
        state.counters.report.transfer_wait += elapsed;
        let activity = state.layer_activity_mut(ticket.key.id.global_layer);
        activity.in_flight_waits += 1;
        activity.transfer_wait += elapsed;
        if result.is_err() {
            // The write commit records its error for asynchronous callers. This
            // caller observed it directly, so do not surface the same failure twice.
            state.background_disk_error = None;
        }
        update_report_totals(&mut state);
        drop(state);
        match result {
            Ok(DiskResult::Write(_)) => Ok(()),
            Ok(_) => Err(CacheResidencyError::Runtime(
                "cache disk worker returned an unexpected write result".into(),
            )),
            Err(error) => Err(error),
        }
    }

    fn rebalance(&self, required: Option<&CacheBlockId>) -> Result<(), CacheResidencyError> {
        loop {
            let mut state = self.lock()?;
            if let Some(error) = state.background_disk_error.take() {
                return Err(CacheResidencyError::Runtime(format!(
                    "background cache disk write failed: {error}"
                )));
            }
            update_report_totals(&mut state);
            if state.counters.report.current_device_bytes > self.options.device_budget_bytes {
                let candidate = eviction_candidate(
                    &state,
                    CacheTier::Device,
                    required,
                    self.options.recent_device_blocks,
                    self.options.eviction_policy,
                );
                let Some(id) = candidate else {
                    state.counters.report.failures += 1;
                    if let Some(required) = required {
                        state.layer_activity_mut(required.global_layer).failures += 1;
                    } else {
                        state.layer_activity_overflow.stats.failures += 1;
                    }
                    return Err(CacheResidencyError::BudgetExceeded {
                        tier: CacheTier::Device,
                        required: state.counters.report.current_device_bytes,
                        budget: self.options.device_budget_bytes,
                    });
                };
                if state
                    .blocks
                    .get(&id)
                    .is_some_and(|record| record.disk.is_some())
                {
                    let record = state.blocks.get_mut(&id).expect("candidate exists");
                    record.arrays = None;
                    record.tier = CacheTier::Disk;
                    state.counters.report.disk_demotions += 1;
                    state.layer_activity_mut(id.global_layer).disk_demotions += 1;
                    continue;
                }
                let candidate_bytes = state.blocks.get(&id).expect("candidate exists").bytes;
                let required_host_bytes = state
                    .counters
                    .report
                    .current_host_bytes
                    .saturating_add(candidate_bytes);
                if required_host_bytes > self.options.host_budget_bytes {
                    if candidate_bytes > self.options.host_budget_bytes {
                        state.counters.report.failures += 1;
                        state.layer_activity_mut(id.global_layer).failures += 1;
                        return Err(CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Host,
                            required: candidate_bytes,
                            budget: self.options.host_budget_bytes,
                        });
                    }
                    let host_candidate = eviction_candidate(
                        &state,
                        CacheTier::Host,
                        required,
                        0,
                        self.options.eviction_policy,
                    );
                    let pending = state
                        .host_write_reservations
                        .values()
                        .next()
                        .map(|reservation| reservation.ticket.clone())
                        .or_else(|| {
                            state.blocks.values().find_map(|record| {
                                record
                                    .pending_disk
                                    .as_ref()
                                    .filter(|pending| {
                                        pending.ticket.key.kind == DiskOperationKind::Write
                                    })
                                    .map(|pending| pending.ticket.clone())
                            })
                        });
                    drop(state);
                    if let Some(id) = host_candidate {
                        match self.begin_host_demotion(&id)? {
                            HostDemotionProgress::Retry | HostDemotionProgress::Freed => continue,
                            HostDemotionProgress::Pending(ticket) => {
                                self.wait_for_host_release(&ticket)?;
                                continue;
                            }
                        }
                    }
                    if let Some(ticket) = pending {
                        self.wait_for_host_release(&ticket)?;
                        continue;
                    }
                    let mut state = self.lock()?;
                    state.counters.report.failures += 1;
                    state.layer_activity_mut(id.global_layer).failures += 1;
                    return Err(match &self.options.live_disk {
                        LiveCacheDiskPolicy::Disabled => CacheResidencyError::LiveDiskRequired {
                            required: required_host_bytes,
                            budget: self.options.host_budget_bytes,
                        },
                        LiveCacheDiskPolicy::Enabled { .. } => {
                            CacheResidencyError::BudgetExceeded {
                                tier: CacheTier::Host,
                                required: required_host_bytes,
                                budget: self.options.host_budget_bytes,
                            }
                        }
                    });
                }
                let device_arrays = state
                    .blocks
                    .get(&id)
                    .and_then(|record| record.arrays.clone())
                    .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?;
                let host_arrays = device_arrays
                    .copy_to_stream(&state.host_stream, "copy cache block from device to host")?;
                let record = state.blocks.get_mut(&id).expect("candidate exists");
                record.arrays = Some(host_arrays);
                record.tier = CacheTier::Host;
                let bytes = record.bytes;
                state.counters.report.host_demotions += 1;
                state.counters.report.transfer_bytes += bytes;
                let activity = state.layer_activity_mut(id.global_layer);
                activity.host_demotions += 1;
                activity.transfer_bytes += bytes;
                continue;
            }

            if state.counters.report.current_host_bytes > self.options.host_budget_bytes {
                let candidate = eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options.eviction_policy,
                );
                let pending = state
                    .host_write_reservations
                    .values()
                    .next()
                    .map(|reservation| reservation.ticket.clone())
                    .or_else(|| {
                        state.blocks.values().find_map(|record| {
                            record
                                .pending_disk
                                .as_ref()
                                .filter(|pending| {
                                    pending.ticket.key.kind == DiskOperationKind::Write
                                })
                                .map(|pending| pending.ticket.clone())
                        })
                    });
                let required_host_bytes = state.counters.report.current_host_bytes;
                drop(state);
                if let Some(id) = candidate {
                    match self.begin_host_demotion(&id)? {
                        HostDemotionProgress::Retry | HostDemotionProgress::Freed => continue,
                        HostDemotionProgress::Pending(ticket) => {
                            self.wait_for_host_release(&ticket)?;
                            continue;
                        }
                    }
                }
                if let Some(ticket) = pending {
                    self.wait_for_host_release(&ticket)?;
                    continue;
                }
                let mut state = self.lock()?;
                state.counters.report.failures += 1;
                if let Some(required) = required {
                    state.layer_activity_mut(required.global_layer).failures += 1;
                } else {
                    state.layer_activity_overflow.stats.failures += 1;
                }
                return Err(CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Host,
                    required: required_host_bytes,
                    budget: self.options.host_budget_bytes,
                });
            }

            // Start one background write as soon as the finite host tier fills.
            // It remains charged to host memory until the worker commits and
            // releases its arrays; a later demotion waits only if it needs space.
            let proactive = matches!(&self.options.live_disk, LiveCacheDiskPolicy::Enabled { .. })
                && state.counters.report.current_host_bytes != 0
                && state.counters.report.current_host_bytes >= self.options.host_budget_bytes;
            let candidate = if proactive {
                eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options.eviction_policy,
                )
            } else {
                None
            };
            drop(state);
            if let Some(id) = candidate {
                let _ = self.begin_host_demotion(&id)?;
            }
            return Ok(());
        }
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
        let replacing = destination.exists();
        if replacing && !options.replace_existing {
            return Err(CacheResidencyError::PromptCacheExists(
                destination.to_path_buf(),
            ));
        }
        if replacing && !destination.is_dir() {
            return Err(CacheResidencyError::InvalidPromptCachePath(
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
        let generation_name = format!("generation-{nonce}");
        let generations = destination.join(PROMPT_CACHE_GENERATIONS_DIRECTORY);
        if replacing {
            fs::create_dir_all(&generations).map_err(|source| CacheResidencyError::Io {
                action: "create prompt cache generation directory",
                path: generations.clone(),
                source,
            })?;
        }
        let temporary = if replacing {
            generations.join(format!(".tmp-{nonce}"))
        } else {
            parent.join(format!(".{file_name}.tmp-{nonce}"))
        };
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
                let payload_sha256 = hash_shard_payload(&shard_path)?;
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
                    payload_sha256,
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

            if replacing {
                let generation = generations.join(&generation_name);
                fs::rename(&temporary, &generation).map_err(|source| CacheResidencyError::Io {
                    action: "publish prompt cache generation",
                    path: generation.clone(),
                    source,
                })?;
                sync_directory(&generations)?;
                publish_prompt_cache_generation(destination, &generation_name, nonce)?;
            } else {
                fs::rename(&temporary, destination).map_err(|source| CacheResidencyError::Io {
                    action: "publish prompt cache",
                    path: destination.to_path_buf(),
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

/// Cache-relevant structure derived from a loaded model instance.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct PromptCacheModelIdentity {
    pub(crate) model_family: String,
    pub(crate) effective_model_type: String,
    pub(crate) architecture_fingerprint: String,
    pub(crate) layer_count: usize,
    pub(crate) global_layer_start: usize,
    pub(crate) global_layer_end: usize,
    pub(crate) sliding_window: Option<i32>,
    pub(crate) sink_tokens: usize,
    pub(crate) topology: PromptCacheTopology,
    pub(crate) layer_layouts: Vec<PromptCacheLayerLayout>,
}

/// Exact cache tensor layout expected by one model-owned decoder layer.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum PromptCacheLayerLayout {
    KeyValue {
        num_key_value_heads: i32,
        head_dim: i32,
    },
    CompressedLatentRotary {
        latent_dim: i32,
        rotary_dim: i32,
    },
}

impl PromptCacheModelIdentity {
    pub(crate) fn key_value_layouts(
        layer_count: usize,
        num_key_value_heads: i32,
        head_dim: i32,
    ) -> Vec<PromptCacheLayerLayout> {
        vec![
            PromptCacheLayerLayout::KeyValue {
                num_key_value_heads,
                head_dim,
            };
            layer_count
        ]
    }

    pub(crate) fn compressed_layouts(
        layer_count: usize,
        latent_dim: i32,
        rotary_dim: i32,
    ) -> Vec<PromptCacheLayerLayout> {
        vec![
            PromptCacheLayerLayout::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
            };
            layer_count
        ]
    }
}

pub(crate) fn validate_prompt_cache_model_identity(
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
) -> Result<(), CacheResidencyError> {
    macro_rules! require_model_equal {
        ($field:ident) => {
            if expected.$field != model.$field {
                return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                    "caller descriptor {} does not match the loaded model",
                    stringify!($field)
                )));
            }
        };
    }
    require_model_equal!(model_family);
    require_model_equal!(effective_model_type);
    require_model_equal!(architecture_fingerprint);
    require_model_equal!(layer_count);
    require_model_equal!(global_layer_start);
    require_model_equal!(global_layer_end);
    require_model_equal!(sliding_window);
    require_model_equal!(sink_tokens);
    require_model_equal!(topology);
    let owned_layers = model
        .global_layer_end
        .checked_sub(model.global_layer_start)
        .ok_or_else(|| {
            CacheResidencyError::IncompatiblePromptCache(
                "loaded model has an invalid prompt-cache layer range".into(),
            )
        })?;
    if model.layer_layouts.len() != owned_layers {
        return Err(CacheResidencyError::IncompatiblePromptCache(format!(
            "loaded model supplied {} cache layouts for {owned_layers} owned layers",
            model.layer_layouts.len()
        )));
    }
    Ok(())
}

pub(crate) fn derive_prompt_cache_architecture_fingerprint<I, K, V>(
    model_family: &str,
    fields: I,
) -> String
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let mut fields = fields
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect::<Vec<_>>();
    fields.sort_unstable();
    let mut hasher = Sha256::new();
    hash_fingerprint_component(&mut hasher, b"safemlx-prompt-cache-architecture-v1");
    hash_fingerprint_component(&mut hasher, model_family.as_bytes());
    for (key, value) in fields {
        hash_fingerprint_component(&mut hasher, key.as_bytes());
        hash_fingerprint_component(&mut hasher, value.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn hash_fingerprint_component(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
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
    /// SHA-256 of the exact safetensors payload bytes.
    #[serde(default)]
    pub payload_sha256: String,
}

/// Reads and validates a prompt-cache manifest without loading its arrays.
pub fn inspect_prompt_cache(
    directory: impl AsRef<Path>,
) -> Result<PromptCacheManifest, CacheResidencyError> {
    let directory = resolve_prompt_cache_root(directory.as_ref())?;
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
    validate_manifest(&directory, &manifest)?;
    Ok(manifest)
}

/// Catalogs a compatible prompt prefix lazily as read-only disk-backed blocks.
pub(crate) fn open_prompt_cache(
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
    prefix_token_ids: &[u32],
    options: PagedCacheOptions,
) -> Result<(CacheResidencyManager, PromptCacheManifest), CacheResidencyError> {
    validate_prompt_cache_model_identity(expected, model)?;
    let directory = directory.as_ref();
    let cache_root = resolve_prompt_cache_root(directory)?;
    let manifest = inspect_prompt_cache(directory)?;
    validate_compatibility(&manifest, expected, prefix_token_ids)?;
    validate_prompt_cache_layer_layouts(&manifest, model)?;
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
            let shard = safe_shard_path(&cache_root, &block.shard)?;
            let mapped = map_prompt_cache_shard(&shard)?;
            let record = CacheBlockRecord {
                id: id.clone(),
                tier: CacheTier::Disk,
                arrays: None,
                disk: Some(DiskLocation {
                    path: shard,
                    first_name: block.first_array.clone(),
                    second_name: block.second_array.clone(),
                    persistent: true,
                    mapped: Some(mapped),
                    payload_sha256: Some(block.payload_sha256.clone()),
                    payload_verification: Arc::new(OnceLock::new()),
                }),
                bytes: block.logical_bytes,
                shapes: [block.first_shape.clone(), block.second_shape.clone()],
                dtypes: [block.first_dtype.clone(), block.second_dtype.clone()],
                imported: true,
                leases: 0,
                access_count: 0,
                last_access: 0,
                protected_prefix: block.end <= manifest.sink_tokens as i64,
                pending_disk: None,
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

fn validate_prompt_cache_layer_layouts(
    manifest: &PromptCacheManifest,
    model: &PromptCacheModelIdentity,
) -> Result<(), CacheResidencyError> {
    let owned_layers = model
        .global_layer_end
        .checked_sub(model.global_layer_start)
        .ok_or_else(|| {
            CacheResidencyError::IncompatiblePromptCache(
                "loaded model has an invalid prompt-cache layer range".into(),
            )
        })?;
    if model.layer_layouts.len() != owned_layers {
        return Err(CacheResidencyError::IncompatiblePromptCache(format!(
            "loaded model supplied {} cache layouts for {owned_layers} owned layers",
            model.layer_layouts.len()
        )));
    }
    for block in &manifest.blocks {
        let layout_index = block
            .global_layer
            .checked_sub(model.global_layer_start)
            .filter(|index| *index < model.layer_layouts.len())
            .ok_or_else(|| {
                CacheResidencyError::IncompatiblePromptCache(format!(
                    "cache block layer {} is not owned by the loaded model",
                    block.global_layer
                ))
            })?;
        let token_count = i32::try_from(block.end - block.start).map_err(|_| {
            CacheResidencyError::IncompatiblePromptCache(format!(
                "cache block layer {} token range exceeds runtime dimensions",
                block.global_layer
            ))
        })?;
        let batch = i32::try_from(manifest.batch_size).map_err(|_| {
            CacheResidencyError::IncompatiblePromptCache(
                "prompt-cache batch size exceeds runtime dimensions".into(),
            )
        })?;
        let (representation, first_shape, second_shape) = match model.layer_layouts[layout_index] {
            PromptCacheLayerLayout::KeyValue {
                num_key_value_heads,
                head_dim,
            } => (
                CacheRepresentation::KeyValue,
                vec![batch, num_key_value_heads, token_count, head_dim],
                vec![batch, num_key_value_heads, token_count, head_dim],
            ),
            PromptCacheLayerLayout::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
            } => (
                CacheRepresentation::CompressedLatentRotary,
                vec![batch, token_count, latent_dim],
                vec![batch, token_count, rotary_dim],
            ),
        };
        if block.representation != representation {
            return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                "cache block layer {} uses {:?}, but the loaded model expects {:?}",
                block.global_layer, block.representation, representation
            )));
        }
        if block.first_shape != first_shape || block.second_shape != second_shape {
            return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                "cache block layer {} dimensions {:?}/{:?} do not match the loaded model's expected {:?}/{:?}",
                block.global_layer,
                block.first_shape,
                block.second_shape,
                first_shape,
                second_shape
            )));
        }
    }
    Ok(())
}

fn resolve_prompt_cache_root(directory: &Path) -> Result<PathBuf, CacheResidencyError> {
    let current_path = directory.join(PROMPT_CACHE_CURRENT_FILE);
    if !current_path.exists() {
        return Ok(directory.to_path_buf());
    }
    let length = current_path
        .metadata()
        .map_err(|source| CacheResidencyError::Io {
            action: "stat prompt cache generation pointer",
            path: current_path.clone(),
            source,
        })?
        .len();
    if length == 0 || length > 256 {
        return Err(CacheResidencyError::MalformedManifest(
            "prompt-cache generation pointer has an invalid length".into(),
        ));
    }
    let generation =
        fs::read_to_string(&current_path).map_err(|source| CacheResidencyError::Io {
            action: "read prompt cache generation pointer",
            path: current_path.clone(),
            source,
        })?;
    let generation = generation.trim();
    let generation_path = Path::new(generation);
    if generation.is_empty()
        || generation_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || generation_path.components().count() != 1
    {
        return Err(CacheResidencyError::MalformedManifest(
            "prompt-cache generation pointer is unsafe".into(),
        ));
    }
    let root = directory
        .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
        .join(generation_path);
    if !root.is_dir() {
        return Err(CacheResidencyError::MalformedManifest(format!(
            "prompt-cache generation {generation:?} is missing"
        )));
    }
    Ok(root)
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
            || !is_sha256_hex(&block.payload_sha256)
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

fn cancel_record_operation(record: &CacheBlockRecord, report: &mut CacheResidencyReport) {
    if let Some(pending) = &record.pending_disk {
        if pending.ticket.cancel() {
            report.cancellations += 1;
        }
    }
}

fn advance_generation_locked(state: &mut CacheManagerState) -> Vec<DiskTicket> {
    state.generation = state.generation.wrapping_add(1);
    state.background_disk_error = None;
    let mut tickets = Vec::new();
    for record in state.blocks.values_mut() {
        if let Some(pending) = record.pending_disk.as_ref() {
            if pending.ticket.cancel() {
                state.counters.report.cancellations += 1;
            }
            tickets.push(pending.ticket.clone());
            if pending.ticket.key.kind != DiskOperationKind::Write {
                record.pending_disk = None;
            }
        }
    }
    tickets
}

fn update_report_totals(state: &mut CacheManagerState) {
    let device_budget_bytes = state.device_budget_bytes;
    let host_budget_bytes = state.host_budget_bytes;
    let disk_budget_bytes = state.disk_budget_bytes;
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
    report.in_flight_write_blocks = 0;
    report.in_flight_write_bytes = 0;
    report.protected_prefix_blocks = 0;
    report.protected_recent_blocks = 0;
    report.logical_cached_tokens = 0;
    report.per_layer.clear();
    report.per_layer_overflow_layers = 0;
    report.per_layer_overflow = CacheLayerResidencyStats::default();
    let mut per_layer = BTreeMap::<usize, CacheLayerResidencyStats>::new();
    let mut layer_ends: HashMap<usize, i64> = HashMap::new();
    for (layer, (bytes, end)) in &state.tails {
        layer_ends.insert(*layer, *end);
        let layer_report = per_layer.entry(*layer).or_default();
        layer_report.current_device_bytes += *bytes;
        layer_report.mutable_tail_bytes += *bytes;
        layer_report.logical_cached_tokens = (*end).max(0) as u64;
    }
    for record in state.blocks.values() {
        let layer_report = per_layer.entry(record.id.global_layer).or_default();
        match record.id.representation {
            CacheRepresentation::KeyValue => {
                report.key_value_blocks += 1;
                layer_report.key_value_blocks += 1;
            }
            CacheRepresentation::CompressedLatentRotary => {
                report.compressed_latent_blocks += 1;
                layer_report.compressed_latent_blocks += 1;
            }
        }
        let pending_write = record
            .pending_disk
            .as_ref()
            .is_some_and(|pending| pending.ticket.key.kind == DiskOperationKind::Write);
        let pending_write_is_reserved = record.pending_disk.as_ref().is_some_and(|pending| {
            state
                .host_write_reservations
                .contains_key(&pending.ticket.key)
        });
        if pending_write && !pending_write_is_reserved {
            report.in_flight_write_blocks += 1;
            report.in_flight_write_bytes += record.bytes;
            layer_report.in_flight_write_blocks += 1;
            layer_report.in_flight_write_bytes += record.bytes;
            if record.tier != CacheTier::Host {
                report.current_host_bytes += record.bytes;
                layer_report.current_host_bytes += record.bytes;
            }
        }
        match record.tier {
            CacheTier::Device => {
                report.device_blocks += 1;
                report.current_device_bytes += record.bytes;
                layer_report.device_blocks += 1;
                layer_report.current_device_bytes += record.bytes;
            }
            CacheTier::Host => {
                report.host_blocks += 1;
                report.current_host_bytes += record.bytes;
                layer_report.host_blocks += 1;
                layer_report.current_host_bytes += record.bytes;
            }
            CacheTier::Disk => {
                report.disk_blocks += 1;
                report.current_disk_bytes += record.bytes;
                layer_report.disk_blocks += 1;
                layer_report.current_disk_bytes += record.bytes;
            }
        }
        if record.protected_prefix {
            report.protected_prefix_blocks += 1;
            layer_report.protected_prefix_blocks += 1;
        }
        layer_report.logical_cached_tokens = layer_report
            .logical_cached_tokens
            .max(record.id.end.max(0) as u64);
        layer_ends
            .entry(record.id.global_layer)
            .and_modify(|end| *end = (*end).max(record.id.end))
            .or_insert(record.id.end);
    }
    for (key, reservation) in &state.host_write_reservations {
        report.in_flight_write_blocks += 1;
        report.in_flight_write_bytes += reservation.bytes;
        let layer_report = per_layer.entry(reservation.global_layer).or_default();
        layer_report.in_flight_write_blocks += 1;
        layer_report.in_flight_write_bytes += reservation.bytes;
        let covered_by_host_record = state.blocks.get(&key.id).is_some_and(|record| {
            record.tier == CacheTier::Host
                && record
                    .pending_disk
                    .as_ref()
                    .is_some_and(|pending| pending.ticket.key == *key)
        });
        if !covered_by_host_record {
            report.current_host_bytes += reservation.bytes;
            layer_report.current_host_bytes += reservation.bytes;
        }
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
    for (layer, starts) in &device_starts {
        per_layer.entry(*layer).or_default().protected_recent_blocks =
            starts.len().min(state.recent_device_blocks) as u64;
    }
    report.logical_cached_tokens = layer_ends.values().copied().max().unwrap_or(0).max(0) as u64;
    // Historical counters keep the first observed layer identities stable.
    // Fill any remaining bounded slots with currently active layers, and fold
    // both current and historical activity for all other layers into overflow.
    let mut selected_layers = state.layer_activity.keys().copied().collect::<Vec<_>>();
    for global_layer in per_layer.keys().copied() {
        if selected_layers.len() == CACHE_RESIDENCY_LAYER_REPORT_LIMIT {
            break;
        }
        if !state.layer_activity.contains_key(&global_layer) {
            selected_layers.push(global_layer);
        }
    }
    selected_layers.sort_unstable();
    for global_layer in selected_layers {
        let mut stats = per_layer.remove(&global_layer).unwrap_or_default();
        if let Some(activity) = state.layer_activity.get(&global_layer) {
            activity.apply_to(&mut stats);
        }
        report.per_layer.push(CacheLayerResidencyReport {
            global_layer,
            stats,
        });
    }
    for (_, stats) in per_layer {
        report.per_layer_overflow_layers += 1;
        report.per_layer_overflow.accumulate(&stats);
    }
    state
        .layer_activity_overflow
        .apply_to(&mut report.per_layer_overflow);
    if report.current_device_bytes <= device_budget_bytes {
        report.peak_device_bytes = report.peak_device_bytes.max(report.current_device_bytes);
    }
    if report.current_host_bytes <= host_budget_bytes {
        report.peak_host_bytes = report.peak_host_bytes.max(report.current_host_bytes);
    }
    if disk_budget_bytes.is_none_or(|budget| report.current_disk_bytes <= budget) {
        report.peak_disk_bytes = report.peak_disk_bytes.max(report.current_disk_bytes);
    }
    report.peak_in_flight_write_bytes = report
        .peak_in_flight_write_bytes
        .max(report.in_flight_write_bytes);
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
                && record.pending_disk.is_none()
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
    let (path, temporary) = live_block_paths(directory, id);
    let mut temporary_guard = TemporaryFileGuard::new(temporary);
    save_block_arrays(temporary_guard.path(), arrays)?;
    sync_file(temporary_guard.path())?;
    publish_live_block_file(temporary_guard.path(), &path)?;
    temporary_guard.disarm();
    let names = array_names(id.representation);
    Ok(DiskLocation {
        path,
        first_name: names.0.into(),
        second_name: names.1.into(),
        persistent: false,
        mapped: None,
        payload_sha256: None,
        payload_verification: Arc::new(OnceLock::new()),
    })
}

fn publish_live_block_file(
    temporary: &Path,
    destination: &Path,
) -> Result<(), CacheResidencyError> {
    // A hard-link publication is atomic and fails if a destination somehow
    // collides, whereas rename would silently replace another process's shard.
    fs::hard_link(temporary, destination).map_err(|source| CacheResidencyError::Io {
        action: "publish uniquely named live cache block",
        path: destination.to_path_buf(),
        source,
    })?;
    if let Err(source) = fs::remove_file(temporary) {
        let _ = fs::remove_file(destination);
        return Err(CacheResidencyError::Io {
            action: "remove published live cache temporary file",
            path: temporary.to_path_buf(),
            source,
        });
    }
    Ok(())
}

fn live_block_paths(directory: &Path, id: &CacheBlockId) -> (PathBuf, PathBuf) {
    let process_namespace = LIVE_PROCESS_NAMESPACE.get_or_init(|| {
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("p{:08x}-t{started:032x}", std::process::id())
    });
    let write_id = NEXT_LIVE_SHARD_ID.fetch_add(1, Ordering::Relaxed);
    let representation = match id.representation {
        CacheRepresentation::KeyValue => "kv",
        CacheRepresentation::CompressedLatentRotary => "mla",
    };
    let rank_component =
        |rank: Option<usize>| rank.map_or_else(|| "x".to_string(), |rank| rank.to_string());
    let rank = id.rank.map_or_else(
        || "rank-px-tx-ex".to_string(),
        |rank| {
            format!(
                "rank-p{}-t{}-e{}",
                rank_component(rank.pipeline_rank),
                rank_component(rank.tensor_parallel_rank),
                rank_component(rank.expert_parallel_rank)
            )
        },
    );
    let base = format!(
        "live-{process_namespace}-w{write_id:016x}-s{:016x}-layer-{:05}-{representation}-{rank}-{}-{}",
        id.session_id, id.global_layer, id.start, id.end
    );
    (
        directory.join(format!("{base}.safetensors")),
        directory.join(format!(".{base}.tmp.safetensors")),
    )
}

struct TemporaryFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TemporaryFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryFileGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn save_block_arrays(path: &Path, arrays: &CacheBlockArrays) -> Result<(), CacheResidencyError> {
    let names = array_names(arrays.representation());
    let values = arrays.arrays();
    Array::save_safetensors([(names.0, values[0]), (names.1, values[1])], None, path).map_err(
        |source| CacheResidencyError::Runtime(format!("save {}: {source}", path.display())),
    )
}

fn hash_shard_payload(path: &Path) -> Result<String, CacheResidencyError> {
    let (_, _, data_start) = read_shard_metadata(path)?;
    let mut file = File::open(path).map_err(|source| CacheResidencyError::Io {
        action: "open prompt cache shard payload",
        path: path.to_path_buf(),
        source,
    })?;
    file.seek(SeekFrom::Start(data_start))
        .map_err(|source| CacheResidencyError::Io {
            action: "seek prompt cache shard payload",
            path: path.to_path_buf(),
            source,
        })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| CacheResidencyError::Io {
                action: "hash prompt cache shard payload",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_disk_payload(location: &DiskLocation) -> Result<(), CacheResidencyError> {
    let Some(expected) = &location.payload_sha256 else {
        return Ok(());
    };
    let verification = location.payload_verification.get_or_init(|| {
        let actual = if let Some(mapped) = &location.mapped {
            if mapped.len() < 8 {
                return Err("file is too short for a safetensors header".into());
            }
            let mut length_bytes = [0u8; 8];
            length_bytes.copy_from_slice(&mapped[..8]);
            let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
                .map_err(|_| "safetensors header length exceeds addressable memory".to_string())?;
            let data_start = 8usize
                .checked_add(header_len)
                .filter(|start| *start <= mapped.len())
                .ok_or_else(|| "safetensors header extends beyond the mapped shard".to_string())?;
            format!("{:x}", Sha256::digest(&mapped[data_start..]))
        } else {
            hash_shard_payload(&location.path).map_err(|error| error.to_string())?
        };
        if &actual == expected {
            Ok(())
        } else {
            Err(format!(
                "payload SHA-256 mismatch: expected {expected}, computed {actual}"
            ))
        }
    });
    verification
        .as_ref()
        .map_err(|reason| CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: reason.clone(),
        })
        .copied()
}

fn load_block_arrays_direct(
    location: &DiskLocation,
    representation: CacheRepresentation,
) -> Result<CacheBlockArrays, CacheResidencyError> {
    verify_disk_payload(location)?;
    let stream = cpu_stream();
    let (mut arrays, _mapped_sources) = if let Some(mapped) = &location.mapped {
        let tensors = safetensors::SafeTensors::deserialize(mapped.as_ref()).map_err(|error| {
            CacheResidencyError::MalformedShard {
                path: location.path.clone(),
                reason: error.to_string(),
            }
        })?;
        let mut arrays = HashMap::with_capacity(2);
        let mut sources = Vec::with_capacity(2);
        for name in [&location.first_name, &location.second_name] {
            let view =
                tensors
                    .tensor(name)
                    .map_err(|error| CacheResidencyError::MalformedShard {
                        path: location.path.clone(),
                        reason: error.to_string(),
                    })?;
            // The source array borrows bytes from `mapped`. Copy and evaluate it
            // before this scope can release the mapping.
            let source = Array::try_from(view).map_err(|error| {
                CacheResidencyError::Runtime(format!(
                    "map {} array {name}: {error}",
                    location.path.display()
                ))
            })?;
            let copy = source
                .copy(&stream)
                .map_err(|error| transfer_error("copy mapped prompt-cache block to host", error))?;
            arrays.insert(name.clone(), copy);
            sources.push(source);
        }
        (arrays, Some(sources))
    } else {
        (
            Array::load_safetensors(&location.path, &stream).map_err(|source| {
                CacheResidencyError::Runtime(format!("load {}: {source}", location.path.display()))
            })?,
            None,
        )
    };
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
    let arrays = match representation {
        CacheRepresentation::KeyValue => CacheBlockArrays::KeyValue {
            keys: first,
            values: second,
        },
        CacheRepresentation::CompressedLatentRotary => CacheBlockArrays::CompressedLatentRotary {
            latent: first,
            rotary_key: second,
        },
    };
    eval(arrays.arrays())
        .map_err(|source| transfer_error("materialize disk cache block on host", source))?;
    stream
        .synchronize()
        .map_err(|source| transfer_error("materialize disk cache block on host", source))?;
    Ok(arrays)
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

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
    let (metadata, file_len, data_start) = read_shard_metadata(path)?;
    let entries = metadata.tensors();
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
        let tensor = metadata
            .info(name)
            .ok_or_else(|| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: format!("missing array {name}"),
            })?;
        let shape = tensor
            .shape
            .iter()
            .map(|dimension| i32::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: "array dimension exceeds runtime range".into(),
            })?;
        if &shape != expected_shape || stored_dtype_name(tensor.dtype) != *expected_dtype {
            return Err(CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: format!("array {name} shape or dtype does not match the manifest"),
            });
        }
        logical_bytes = logical_bytes.saturating_add(
            u64::try_from(tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0))
                .unwrap_or(u64::MAX),
        );
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
    let expected_file_len = data_start
        .checked_add(metadata.data_len() as u64)
        .ok_or_else(|| CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "safetensors file length overflow".into(),
        })?;
    if expected_file_len != file_len {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!(
                "safetensors payload boundary {expected_file_len} does not match file length {file_len}"
            ),
        });
    }
    Ok(())
}

fn read_shard_metadata(
    path: &Path,
) -> Result<(safetensors::tensor::Metadata, u64, u64), CacheResidencyError> {
    let mut file = File::open(path).map_err(|source| CacheResidencyError::Io {
        action: "open prompt cache shard metadata",
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| CacheResidencyError::Io {
            action: "stat prompt cache shard",
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| CacheResidencyError::Io {
            action: "read prompt cache shard header length",
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_PROMPT_CACHE_SHARD_HEADER_BYTES {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!(
                "safetensors header length {header_len} exceeds the prompt-cache bound"
            ),
        });
    }
    let data_start =
        8u64.checked_add(header_len)
            .ok_or_else(|| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: "safetensors header length overflow".into(),
            })?;
    if data_start > file_len {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "safetensors header extends beyond the file".into(),
        });
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|source| CacheResidencyError::Io {
            action: "read prompt cache shard header",
            path: path.to_path_buf(),
            source,
        })?;
    let metadata =
        serde_json::from_slice::<safetensors::tensor::Metadata>(&header).map_err(|error| {
            CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
    Ok((metadata, file_len, data_start))
}

fn map_prompt_cache_shard(path: &Path) -> Result<Arc<Mmap>, CacheResidencyError> {
    let file = File::open(path).map_err(|source| CacheResidencyError::Io {
        action: "open prompt cache shard for mapping",
        path: path.to_path_buf(),
        source,
    })?;
    // SAFETY: prompt-cache shards are immutable after publication and the Mmap
    // is retained by every DiskLocation that can create an MLX view from it.
    let mapped =
        unsafe { MmapOptions::new().map(&file) }.map_err(|source| CacheResidencyError::Io {
            action: "map prompt cache shard",
            path: path.to_path_buf(),
            source,
        })?;
    safetensors::SafeTensors::deserialize(&mapped).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;
    Ok(Arc::new(mapped))
}

fn publish_prompt_cache_generation(
    destination: &Path,
    generation_name: &str,
    nonce: u128,
) -> Result<(), CacheResidencyError> {
    let temporary = destination.join(format!(".{PROMPT_CACHE_CURRENT_FILE}.tmp-{nonce}"));
    let current = destination.join(PROMPT_CACHE_CURRENT_FILE);
    let mut file = File::create(&temporary).map_err(|source| CacheResidencyError::Io {
        action: "create prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    writeln!(file, "{generation_name}").map_err(|source| CacheResidencyError::Io {
        action: "write prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    file.sync_all().map_err(|source| CacheResidencyError::Io {
        action: "sync prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, &current).map_err(|source| CacheResidencyError::Io {
        action: "switch prompt cache generation",
        path: current,
        source,
    })?;
    sync_directory(destination)
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
    /// A queued or in-flight disk operation belonged to an invalidated generation.
    #[error("cache disk operation from generation {generation} was cancelled")]
    DiskOperationCancelled {
        /// Generation invalidated by reset or truncation.
        generation: u64,
    },
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
        hash_shard_payload, hash_token_ids, inspect_prompt_cache, live_block_paths,
        map_prompt_cache_shard, open_prompt_cache, publish_live_block_file,
        publish_prompt_cache_generation, safe_shard_path, validate_prompt_cache_model_identity,
        verify_disk_payload, CacheBlockArrays, CacheBlockId, CacheBlockRecord,
        CacheLayerResidencyStats, CacheRankIdentity, CacheRepresentation, CacheResidencyError,
        CacheResidencyManager, CacheTier, DiskLocation, DiskOperationKey, DiskOperationKind,
        DiskResult, DiskTask, DiskWorker, DiskWriteCommit, HostWriteReservation, PagedCacheOptions,
        PendingDiskOperation, PromptCacheBlock, PromptCacheDescriptor, PromptCacheManifest,
        PromptCacheModelIdentity, PromptCacheTopology, TemporaryFileGuard,
        CACHE_RESIDENCY_LAYER_REPORT_LIMIT, MAX_PROMPT_CACHE_SHARD_HEADER_BYTES,
        PROMPT_CACHE_GENERATIONS_DIRECTORY, PROMPT_CACHE_SCHEMA_VERSION,
    };
    use safemlx::{transforms::eval, Array, Device, DeviceType, Stream};
    use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};
    use std::{
        fs,
        fs::OpenOptions,
        io::Write as _,
        path::Path,
        sync::{mpsc, Arc, OnceLock},
        thread,
        time::Duration,
    };

    fn disk_test_id(start: i64) -> CacheBlockId {
        CacheBlockId {
            session_id: 7,
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start,
            end: start + 1,
            rank: None,
        }
    }

    fn missing_location(root: &Path, name: &str) -> DiskLocation {
        DiskLocation {
            path: root.join(name),
            first_name: "keys".into(),
            second_name: "values".into(),
            persistent: false,
            mapped: None,
            payload_sha256: None,
            payload_verification: Arc::new(OnceLock::new()),
        }
    }

    fn manager_with_leased_block() -> CacheResidencyManager {
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(1, 64, 64, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let id = CacheBlockId {
            session_id: manager.session_id,
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 1,
            rank: None,
        };
        manager.lock().unwrap().blocks.insert(
            id.clone(),
            CacheBlockRecord {
                id,
                tier: CacheTier::Host,
                arrays: None,
                disk: None,
                bytes: 0,
                shapes: [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
                leases: 1,
                access_count: 0,
                last_access: 0,
                protected_prefix: false,
                pending_disk: None,
            },
        );
        manager
    }

    fn prompt_descriptor() -> PromptCacheDescriptor {
        PromptCacheDescriptor {
            model_family: "llama".into(),
            effective_model_type: "llama".into(),
            checkpoint_fingerprint: "checkpoint".into(),
            architecture_fingerprint: "architecture".into(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            batch_size: 1,
            sliding_window: None,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        }
    }

    fn prompt_model_identity() -> PromptCacheModelIdentity {
        let descriptor = prompt_descriptor();
        PromptCacheModelIdentity {
            model_family: descriptor.model_family,
            effective_model_type: descriptor.effective_model_type,
            architecture_fingerprint: descriptor.architecture_fingerprint,
            layer_count: descriptor.layer_count,
            global_layer_start: descriptor.global_layer_start,
            global_layer_end: descriptor.global_layer_end,
            sliding_window: descriptor.sliding_window,
            sink_tokens: descriptor.sink_tokens,
            topology: descriptor.topology,
            layer_layouts: PromptCacheModelIdentity::key_value_layouts(1, 1, 1),
        }
    }

    fn write_prompt_fixture(root: &Path, namespace: &str) -> PromptCacheManifest {
        fs::create_dir_all(root).unwrap();
        let keys = 1.0f32.to_le_bytes();
        let values = 2.0f32.to_le_bytes();
        let key_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &keys).unwrap();
        let value_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &values).unwrap();
        serialize_to_file(
            [("keys", key_view), ("values", value_view)],
            None,
            &root.join("block.safetensors"),
        )
        .unwrap();
        let descriptor = prompt_descriptor();
        let manifest = PromptCacheManifest {
            schema_version: PROMPT_CACHE_SCHEMA_VERSION,
            model_family: descriptor.model_family,
            effective_model_type: descriptor.effective_model_type,
            checkpoint_fingerprint: descriptor.checkpoint_fingerprint,
            architecture_fingerprint: descriptor.architecture_fingerprint,
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            block_size_tokens: 1,
            batch_size: 1,
            total_prefix_tokens: 1,
            prefix_sha256: hash_token_ids(&[7]),
            sliding_window: None,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
            application_namespace: Some(namespace.into()),
            blocks: vec![PromptCacheBlock {
                global_layer: 0,
                representation: CacheRepresentation::KeyValue,
                start: 0,
                end: 1,
                rank: None,
                shard: "block.safetensors".into(),
                first_array: "keys".into(),
                second_array: "values".into(),
                first_shape: vec![1, 1, 1, 1],
                second_shape: vec![1, 1, 1, 1],
                first_dtype: "Float32".into(),
                second_dtype: "Float32".into(),
                logical_bytes: 8,
                payload_sha256: hash_shard_payload(&root.join("block.safetensors")).unwrap(),
            }],
        };
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

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
    fn live_shard_paths_include_process_representation_rank_and_unique_write_identity() {
        let directory = tempfile::tempdir().unwrap();
        let id = disk_test_id(0);
        let (first, first_temporary) = live_block_paths(directory.path(), &id);
        let (second, _) = live_block_paths(directory.path(), &id);
        assert_ne!(first, second);
        let first_name = first.file_name().unwrap().to_string_lossy();
        assert!(first_name.contains("live-p"));
        assert!(first_name.contains("-kv-rank-px-tx-ex-"));
        assert!(first_temporary
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with(".tmp.safetensors"));

        let mut ranked = id;
        ranked.representation = CacheRepresentation::CompressedLatentRotary;
        ranked.rank = Some(CacheRankIdentity {
            pipeline_rank: Some(1),
            tensor_parallel_rank: Some(2),
            expert_parallel_rank: Some(3),
        });
        let (ranked_path, _) = live_block_paths(directory.path(), &ranked);
        assert!(ranked_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("-mla-rank-p1-t2-e3-"));
    }

    #[test]
    fn temporary_file_guard_removes_failed_and_panicking_writes() {
        let directory = tempfile::tempdir().unwrap();
        let failed = directory.path().join("failed.tmp.safetensors");
        {
            let _guard = TemporaryFileGuard::new(failed.clone());
            fs::write(&failed, b"partial").unwrap();
        }
        assert!(!failed.exists());

        let panicking = directory.path().join("panicking.tmp.safetensors");
        let _ = std::panic::catch_unwind(|| {
            let _guard = TemporaryFileGuard::new(panicking.clone());
            fs::write(&panicking, b"partial").unwrap();
            panic!("injected write panic");
        });
        assert!(!panicking.exists());
    }

    #[test]
    fn live_shard_publication_never_clobbers_an_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("live.safetensors");
        let temporary = directory.path().join(".live.tmp.safetensors");
        fs::write(&destination, b"first process").unwrap();
        {
            let _guard = TemporaryFileGuard::new(temporary.clone());
            fs::write(&temporary, b"second process").unwrap();
            assert!(publish_live_block_file(&temporary, &destination).is_err());
        }
        assert_eq!(fs::read(&destination).unwrap(), b"first process");
        assert!(!temporary.exists());
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

    #[test]
    fn same_length_prompt_payload_corruption_is_rejected_before_array_conversion() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = write_prompt_fixture(directory.path(), "payload-checksum");
        let shard = directory.path().join(&manifest.blocks[0].shard);
        let mut bytes = fs::read(&shard).unwrap();
        let final_byte = bytes.last_mut().expect("fixture shard has a payload");
        *final_byte ^= 0x01;
        fs::write(&shard, &bytes).unwrap();

        // Header-only inspection remains valid because metadata and length did
        // not change. The mapped payload gate must still reject the shard.
        inspect_prompt_cache(directory.path()).unwrap();
        let location = DiskLocation {
            path: shard.clone(),
            first_name: "keys".into(),
            second_name: "values".into(),
            persistent: true,
            mapped: Some(map_prompt_cache_shard(&shard).unwrap()),
            payload_sha256: Some(manifest.blocks[0].payload_sha256.clone()),
            payload_verification: Arc::new(OnceLock::new()),
        };
        let error = verify_disk_payload(&location).unwrap_err();
        assert!(error.to_string().contains("payload SHA-256 mismatch"));
    }

    #[test]
    fn shard_inspection_enforces_a_bounded_header_read() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.safetensors");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&(MAX_PROMPT_CACHE_SHARD_HEADER_BYTES + 1).to_le_bytes())
            .unwrap();
        let block = PromptCacheBlock {
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 1,
            rank: None,
            shard: "oversized.safetensors".into(),
            first_array: "keys".into(),
            second_array: "values".into(),
            first_shape: vec![1, 1, 1, 1],
            second_shape: vec![1, 1, 1, 1],
            first_dtype: "Float32".into(),
            second_dtype: "Float32".into(),
            logical_bytes: 8,
            payload_sha256: "0".repeat(64),
        };
        assert!(matches!(
            super::validate_shard_file(&path, &block),
            Err(CacheResidencyError::MalformedShard { .. })
        ));
    }

    #[test]
    fn imported_prompt_shards_are_actually_mapped_and_retained() {
        let directory = tempfile::tempdir().unwrap();
        write_prompt_fixture(directory.path(), "mapped");
        let options = PagedCacheOptions::new(1, 64, 64, 1).unwrap();
        let (manager, _) = open_prompt_cache(
            directory.path(),
            &prompt_descriptor(),
            &prompt_model_identity(),
            &[7],
            options,
        )
        .unwrap();
        let state = manager.lock().unwrap();
        assert_eq!(state.counters.report.imported_mapped_shards, 1);
        assert!(state.blocks.values().all(|record| record
            .disk
            .as_ref()
            .and_then(|location| location.mapped.as_ref())
            .is_some()));
        for record in state.blocks.values() {
            verify_disk_payload(record.disk.as_ref().unwrap()).unwrap();
        }
    }

    #[test]
    fn generation_switch_keeps_the_previous_cache_canonical_until_commit() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("prompt-cache");
        write_prompt_fixture(&destination, "old");
        let generation_name = "generation-test";
        let generation = destination
            .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
            .join(generation_name);
        write_prompt_fixture(&generation, "new");

        assert_eq!(
            inspect_prompt_cache(&destination)
                .unwrap()
                .application_namespace
                .as_deref(),
            Some("old")
        );
        publish_prompt_cache_generation(&destination, generation_name, 1).unwrap();
        assert_eq!(
            inspect_prompt_cache(&destination)
                .unwrap()
                .application_namespace
                .as_deref(),
            Some("new")
        );
    }

    #[test]
    fn loaded_model_identity_rejects_a_forged_caller_descriptor() {
        let mut descriptor = prompt_descriptor();
        descriptor.layer_count = 2;
        descriptor.global_layer_end = 2;
        let loaded_model = PromptCacheModelIdentity {
            model_family: "llama".into(),
            effective_model_type: "llama".into(),
            architecture_fingerprint: descriptor.architecture_fingerprint.clone(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            sliding_window: None,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
            layer_layouts: PromptCacheModelIdentity::key_value_layouts(1, 1, 1),
        };
        assert!(matches!(
            validate_prompt_cache_model_identity(&descriptor, &loaded_model),
            Err(CacheResidencyError::IncompatiblePromptCache(_))
        ));
    }

    #[test]
    fn loaded_model_identity_rejects_a_forged_architecture_fingerprint() {
        let mut descriptor = prompt_descriptor();
        let loaded_model = PromptCacheModelIdentity {
            model_family: descriptor.model_family.clone(),
            effective_model_type: descriptor.effective_model_type.clone(),
            architecture_fingerprint: "sha256:derived-from-loaded-model".into(),
            layer_count: descriptor.layer_count,
            global_layer_start: descriptor.global_layer_start,
            global_layer_end: descriptor.global_layer_end,
            sliding_window: descriptor.sliding_window,
            sink_tokens: descriptor.sink_tokens,
            topology: descriptor.topology.clone(),
            layer_layouts: PromptCacheModelIdentity::key_value_layouts(1, 1, 1),
        };
        descriptor.architecture_fingerprint = "sha256:caller-repeated-stale-value".into();
        let error = validate_prompt_cache_model_identity(&descriptor, &loaded_model).unwrap_err();
        assert!(error.to_string().contains("architecture_fingerprint"));
    }

    #[test]
    fn prompt_load_rejects_model_incompatible_key_value_dimensions() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = write_prompt_fixture(directory.path(), "wrong-kv-dimensions");
        let keys = vec![0u8; 32];
        let values = vec![0u8; 32];
        let key_view = TensorView::new(StoredDtype::F32, vec![1, 2, 1, 4], &keys).unwrap();
        let value_view = TensorView::new(StoredDtype::F32, vec![1, 2, 1, 4], &values).unwrap();
        serialize_to_file(
            [("keys", key_view), ("values", value_view)],
            None,
            &directory.path().join("block.safetensors"),
        )
        .unwrap();
        manifest.blocks[0].first_shape = vec![1, 2, 1, 4];
        manifest.blocks[0].second_shape = vec![1, 2, 1, 4];
        manifest.blocks[0].logical_bytes = 64;
        fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let error = open_prompt_cache(
            directory.path(),
            &prompt_descriptor(),
            &prompt_model_identity(),
            &[7],
            PagedCacheOptions::new(1, 64, 64, 1).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("dimensions"));
    }

    #[test]
    fn prompt_load_rejects_model_incompatible_layer_representation() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = write_prompt_fixture(directory.path(), "wrong-representation");
        let latent = vec![0u8; 16];
        let rotary = vec![0u8; 8];
        let latent_view = TensorView::new(StoredDtype::F32, vec![1, 1, 4], &latent).unwrap();
        let rotary_view = TensorView::new(StoredDtype::F32, vec![1, 1, 2], &rotary).unwrap();
        serialize_to_file(
            [("latent", latent_view), ("rotary_key", rotary_view)],
            None,
            &directory.path().join("block.safetensors"),
        )
        .unwrap();
        manifest.blocks[0].representation = CacheRepresentation::CompressedLatentRotary;
        manifest.blocks[0].first_array = "latent".into();
        manifest.blocks[0].second_array = "rotary_key".into();
        manifest.blocks[0].first_shape = vec![1, 1, 4];
        manifest.blocks[0].second_shape = vec![1, 1, 2];
        manifest.blocks[0].logical_bytes = 24;
        fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let error = open_prompt_cache(
            directory.path(),
            &prompt_descriptor(),
            &prompt_model_identity(),
            &[7],
            PagedCacheOptions::new(1, 64, 64, 1).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("expects KeyValue"));
    }

    #[test]
    fn model_reset_surfaces_propagate_paged_clear_failures() {
        use crate::{
            cache::PagedKeyValueCache,
            expert_parallel::ExpertParallelCache,
            llama::LlamaCache,
            models::gpt_oss::{Cache as GptOssCache, LayerCache as GptOssLayerCache},
            pipeline::{PipelineCache, PipelineLlamaLayerCache},
            tensor_parallel::{TensorParallelCache, TensorParallelLlamaLayerCache},
        };

        let manager = manager_with_leased_block();
        let mut llama = LlamaCache::Paged(vec![Some(
            PagedKeyValueCache::new(manager.clone(), 0, None).unwrap(),
        )]);
        assert!(llama.clear().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);

        let manager = manager_with_leased_block();
        let mut tensor_parallel =
            TensorParallelCache::Llama(vec![TensorParallelLlamaLayerCache::Paged(
                PagedKeyValueCache::new(manager.clone(), 0, None).unwrap(),
            )]);
        assert!(tensor_parallel.reset().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);

        let manager = manager_with_leased_block();
        let mut pipeline = PipelineCache::Llama(vec![PipelineLlamaLayerCache::Paged {
            global_layer: 0,
            cache: PagedKeyValueCache::new(manager.clone(), 0, None).unwrap(),
        }]);
        assert!(pipeline.reset().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);

        let manager = manager_with_leased_block();
        let gpt_cache = || GptOssCache {
            layers: vec![GptOssLayerCache::Paged(
                PagedKeyValueCache::new(manager.clone(), 0, None).unwrap(),
            )],
        };
        let mut gpt_oss = gpt_cache();
        assert!(gpt_oss.reset().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);

        let mut expert_parallel = ExpertParallelCache::GptOss(gpt_cache());
        assert!(expert_parallel.reset().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);
    }

    #[test]
    fn disk_worker_coalesces_duplicate_in_flight_reads() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let id = disk_test_id(0);
        let location = missing_location(directory.path(), "missing.safetensors");
        let first = worker
            .prepare_read(3, &id, &location, CacheRepresentation::KeyValue)
            .unwrap();
        let ticket = first.ticket.clone();
        let second = worker
            .prepare_read(3, &id, &location, CacheRepresentation::KeyValue)
            .unwrap();
        assert!(second.joined);
        let second_ticket = second.ticket.clone();
        first.enqueue().unwrap();
        second.enqueue().unwrap();
        assert!(ticket.wait().is_err());
        assert!(second_ticket.wait().is_err());
        assert!(std::sync::Arc::ptr_eq(
            &ticket.completion,
            &second_ticket.completion
        ));
        worker.retire(&ticket);
    }

    #[test]
    fn disk_worker_applies_backpressure_only_outside_submission() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let (first_started_tx, first_started_rx) = mpsc::channel();
        let (first_release_tx, first_release_rx) = mpsc::channel();
        let first = worker
            .prepare(
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
                },
                DiskTask::Pause {
                    started: first_started_tx,
                    release: first_release_rx,
                },
            )
            .unwrap();
        let first_ticket = first.ticket.clone();
        first.enqueue().unwrap();
        first_started_rx.recv().unwrap();

        let (second_started_tx, second_started_rx) = mpsc::channel();
        let (second_release_tx, second_release_rx) = mpsc::channel();
        let second = worker
            .prepare(
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(1),
                    kind: DiskOperationKind::Read,
                },
                DiskTask::Pause {
                    started: second_started_tx,
                    release: second_release_rx,
                },
            )
            .unwrap();
        let second_ticket = second.ticket.clone();
        second.enqueue().unwrap();

        let third = worker
            .prepare_read(
                0,
                &disk_test_id(2),
                &missing_location(directory.path(), "third.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let third_ticket = third.ticket.clone();
        let (outcome_tx, outcome_rx) = mpsc::channel();
        let enqueue_thread = thread::spawn(move || outcome_tx.send(third.enqueue()).unwrap());
        assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());

        first_release_tx.send(()).unwrap();
        second_started_rx.recv().unwrap();
        let outcome = outcome_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert!(outcome.backpressure);
        assert_eq!(outcome.peak_occupancy, 1);
        second_release_tx.send(()).unwrap();
        enqueue_thread.join().unwrap();
        assert!(matches!(first_ticket.wait().unwrap(), DiskResult::Test));
        assert!(matches!(second_ticket.wait().unwrap(), DiskResult::Test));
        assert!(third_ticket.wait().is_err());
        worker.retire(&first_ticket);
        worker.retire(&second_ticket);
        worker.retire(&third_ticket);
    }

    #[test]
    fn disk_worker_cancels_queued_generation_work() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                DiskOperationKey {
                    generation: 8,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let cancelled = worker
            .prepare_read(
                8,
                &disk_test_id(1),
                &missing_location(directory.path(), "cancelled.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let cancelled_ticket = cancelled.ticket.clone();
        cancelled.enqueue().unwrap();
        assert!(cancelled_ticket.cancel());
        assert!(matches!(
            cancelled_ticket.wait(),
            Err(CacheResidencyError::DiskOperationCancelled { generation: 8 })
        ));
        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        worker.retire(&blocker_ticket);
        worker.retire(&cancelled_ticket);
    }

    #[test]
    fn cancellation_wakes_a_backpressured_submitter() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                DiskOperationKey {
                    generation: 4,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let queued = worker
            .prepare_read(
                4,
                &disk_test_id(1),
                &missing_location(directory.path(), "queued.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let queued_ticket = queued.ticket.clone();
        queued.enqueue().unwrap();
        let blocked = worker
            .prepare_read(
                4,
                &disk_test_id(2),
                &missing_location(directory.path(), "blocked.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let blocked_ticket = blocked.ticket.clone();
        let (outcome_tx, outcome_rx) = mpsc::channel();
        let enqueue_thread = thread::spawn(move || outcome_tx.send(blocked.enqueue()).unwrap());
        assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());

        assert!(blocked_ticket.cancel());
        let outcome = outcome_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert!(outcome.backpressure);
        enqueue_thread.join().unwrap();
        assert!(matches!(
            blocked_ticket.wait(),
            Err(CacheResidencyError::DiskOperationCancelled { generation: 4 })
        ));
        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        assert!(queued_ticket.wait().is_err());
        worker.retire(&blocker_ticket);
        worker.retire(&queued_ticket);
        worker.retire(&blocked_ticket);
    }

    #[test]
    fn disk_worker_reports_operation_panics_and_keeps_running() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let panicking = worker
            .prepare(
                DiskOperationKey {
                    generation: 2,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
                },
                DiskTask::Panic,
            )
            .unwrap();
        let panicking_ticket = panicking.ticket.clone();
        panicking.enqueue().unwrap();
        assert!(matches!(
            panicking_ticket.wait(),
            Err(CacheResidencyError::Runtime(message))
                if message.contains("operation panicked")
        ));
        worker.retire(&panicking_ticket);

        let following = worker
            .prepare_read(
                2,
                &disk_test_id(1),
                &missing_location(directory.path(), "following.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let following_ticket = following.ticket.clone();
        following.enqueue().unwrap();
        assert!(following_ticket.wait().is_err());
        worker.retire(&following_ticket);
    }

    #[test]
    fn background_write_failures_surface_on_the_next_foreground_operation() {
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(1, 64, 64, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let worker = manager.disk_worker.as_ref().unwrap();
        let id = disk_test_id(0);
        let key = DiskOperationKey {
            generation: 0,
            id: id.clone(),
            kind: DiskOperationKind::Write,
        };
        let submission = worker.prepare(key.clone(), DiskTask::Panic).unwrap();
        let ticket = submission.ticket.clone();
        manager.lock().unwrap().blocks.insert(
            id.clone(),
            CacheBlockRecord {
                id,
                tier: CacheTier::Host,
                arrays: None,
                disk: None,
                bytes: 0,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
                leases: 0,
                access_count: 0,
                last_access: 0,
                protected_prefix: false,
                pending_disk: Some(PendingDiskOperation {
                    ticket: ticket.clone(),
                }),
            },
        );
        DiskWriteCommit {
            state: Arc::downgrade(&manager.state),
            key,
            reservation_id: 0,
            armed: true,
        }
        .reconcile(&Err(CacheResidencyError::Runtime(
            "injected asynchronous write failure".into(),
        )));

        let error = manager.set_tail_state(0, 0, 0).unwrap_err();
        assert!(error
            .to_string()
            .contains("injected asynchronous write failure"));
        let report = manager.report().unwrap();
        assert_eq!(report.failures, 1);
        assert_eq!(report.per_layer.len(), 1);
        assert_eq!(report.per_layer[0].global_layer, 0);
        assert_eq!(report.per_layer[0].stats.failures, report.failures);
        worker.retire(&ticket);
    }

    #[test]
    fn promoted_and_cancelled_writes_retain_host_reservations_until_release() {
        let directory = tempfile::tempdir().unwrap();
        let options = PagedCacheOptions::new(1, 16, 16, 1)
            .unwrap()
            .with_live_disk(directory.path(), 1024, 1)
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let worker = manager.disk_worker.as_ref().unwrap();
        let id = disk_test_id(0);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let key = DiskOperationKey {
            generation: 0,
            id: id.clone(),
            kind: DiskOperationKind::Write,
        };
        let submission = worker
            .prepare(
                key.clone(),
                DiskTask::PauseWrite {
                    started: started_tx,
                    release: release_rx,
                    commit: Some(DiskWriteCommit {
                        state: Arc::downgrade(&manager.state),
                        key: key.clone(),
                        reservation_id: 7,
                        armed: true,
                    }),
                },
            )
            .unwrap();
        let ticket = submission.ticket.clone();
        submission.enqueue().unwrap();
        started_rx.recv().unwrap();
        {
            let mut state = manager.lock().unwrap();
            state.host_write_reservations.insert(
                key,
                HostWriteReservation {
                    reservation_id: 7,
                    global_layer: id.global_layer,
                    bytes: 16,
                    ticket: ticket.clone(),
                },
            );
            state.blocks.insert(
                id.clone(),
                CacheBlockRecord {
                    id: id.clone(),
                    tier: CacheTier::Host,
                    arrays: None,
                    disk: None,
                    bytes: 16,
                    shapes: [vec![1], vec![1]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                    leases: 0,
                    access_count: 0,
                    last_access: 0,
                    protected_prefix: false,
                    pending_disk: Some(PendingDiskOperation {
                        ticket: ticket.clone(),
                    }),
                },
            );
        }

        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.in_flight_write_bytes, 16);
        assert_eq!(report.host_blocks, 1);

        // Demand promotion replaces the record's arrays with device storage,
        // but the disk task still owns the original host staging allocation.
        manager.lock().unwrap().blocks.get_mut(&id).unwrap().tier = CacheTier::Device;
        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 1);
        assert_eq!(report.host_blocks, 0);
        assert_eq!(report.current_device_bytes, 16);
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.in_flight_write_bytes, 16);

        let clear_manager = manager.clone();
        let (cleared_tx, cleared_rx) = mpsc::channel();
        let clear_thread = thread::spawn(move || {
            cleared_tx.send(clear_manager.clear()).unwrap();
        });
        assert!(matches!(
            ticket.wait(),
            Err(CacheResidencyError::DiskOperationCancelled { generation: 0 })
        ));
        assert!(matches!(
            cleared_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let report = manager.report().unwrap();
        assert_eq!(report.cancellations, 1);
        assert_eq!(report.host_blocks, 0);
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.in_flight_write_bytes, 16);
        release_tx.send(()).unwrap();
        cleared_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("clear did not finish after the write released its arrays")
            .unwrap();
        clear_thread.join().unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, 0);
        assert_eq!(report.in_flight_write_bytes, 0);
    }

    #[test]
    fn disk_backed_device_blocks_bypass_a_zero_host_budget() {
        let directory = tempfile::tempdir().unwrap();
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, 16, 0, 1).unwrap()).unwrap();
        let older = disk_test_id(0);
        let recent = disk_test_id(2);
        {
            let mut state = manager.lock().unwrap();
            for (id, disk) in [
                (
                    older.clone(),
                    Some(missing_location(directory.path(), "older.safetensors")),
                ),
                (recent.clone(), None),
            ] {
                state.blocks.insert(
                    id.clone(),
                    CacheBlockRecord {
                        id,
                        tier: CacheTier::Device,
                        arrays: None,
                        disk,
                        bytes: 16,
                        shapes: [vec![1], vec![1]],
                        dtypes: ["Float32".into(), "Float32".into()],
                        imported: false,
                        leases: 0,
                        access_count: 0,
                        last_access: 0,
                        protected_prefix: false,
                        pending_disk: None,
                    },
                );
            }
        }

        manager.rebalance(None).unwrap();
        let state = manager.lock().unwrap();
        assert_eq!(state.blocks.get(&older).unwrap().tier, CacheTier::Disk);
        assert_eq!(state.blocks.get(&recent).unwrap().tier, CacheTier::Device);
        assert_eq!(state.counters.report.current_host_bytes, 0);
        assert_eq!(state.counters.report.current_device_bytes, 16);
        assert_eq!(state.counters.report.current_disk_bytes, 16);
    }

    #[test]
    fn per_layer_residency_report_is_bounded_and_losslessly_aggregated() {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, u64::MAX, u64::MAX, 1).unwrap())
                .unwrap();
        let layer_count = CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 3;
        {
            let mut state = manager.lock().unwrap();
            for global_layer in 0..layer_count {
                let representation = if global_layer % 2 == 0 {
                    CacheRepresentation::KeyValue
                } else {
                    CacheRepresentation::CompressedLatentRotary
                };
                let tier = match global_layer % 3 {
                    0 => CacheTier::Device,
                    1 => CacheTier::Host,
                    _ => CacheTier::Disk,
                };
                let id = CacheBlockId {
                    session_id: manager.session_id,
                    global_layer,
                    representation,
                    start: 0,
                    end: global_layer as i64 + 1,
                    rank: None,
                };
                state.blocks.insert(
                    id.clone(),
                    CacheBlockRecord {
                        id,
                        tier,
                        arrays: None,
                        disk: None,
                        bytes: global_layer as u64 + 1,
                        shapes: [vec![1], vec![1]],
                        dtypes: ["Float32".into(), "Float32".into()],
                        imported: false,
                        leases: 0,
                        access_count: 0,
                        last_access: 0,
                        protected_prefix: global_layer % 5 == 0,
                        pending_disk: None,
                    },
                );
                state
                    .tails
                    .insert(global_layer, (2, global_layer as i64 + 1));
            }
        }

        let report = manager.report().unwrap();
        assert_eq!(report.per_layer.len(), CACHE_RESIDENCY_LAYER_REPORT_LIMIT);
        assert_eq!(report.per_layer_overflow_layers, 3);
        assert_eq!(
            report
                .per_layer
                .iter()
                .map(|layer| layer.global_layer)
                .collect::<Vec<_>>(),
            (0..CACHE_RESIDENCY_LAYER_REPORT_LIMIT).collect::<Vec<_>>()
        );

        let mut aggregate = CacheLayerResidencyStats::default();
        for layer in &report.per_layer {
            aggregate.accumulate(&layer.stats);
        }
        aggregate.accumulate(&report.per_layer_overflow);
        assert_eq!(aggregate.key_value_blocks, report.key_value_blocks);
        assert_eq!(
            aggregate.compressed_latent_blocks,
            report.compressed_latent_blocks
        );
        assert_eq!(aggregate.device_blocks, report.device_blocks);
        assert_eq!(aggregate.host_blocks, report.host_blocks);
        assert_eq!(aggregate.disk_blocks, report.disk_blocks);
        assert_eq!(aggregate.current_device_bytes, report.current_device_bytes);
        assert_eq!(aggregate.current_host_bytes, report.current_host_bytes);
        assert_eq!(aggregate.current_disk_bytes, report.current_disk_bytes);
        assert_eq!(aggregate.mutable_tail_bytes, report.mutable_tail_bytes);
        assert_eq!(
            aggregate.protected_recent_blocks,
            report.protected_recent_blocks
        );
        assert_eq!(
            aggregate.protected_prefix_blocks,
            report.protected_prefix_blocks
        );
        assert_eq!(report.logical_cached_tokens, layer_count as u64);
        assert_eq!(
            report.per_layer_overflow.logical_cached_tokens,
            ((CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 1)..=layer_count)
                .map(|tokens| tokens as u64)
                .sum::<u64>()
        );
    }

    #[test]
    fn per_layer_cumulative_attention_is_bounded_and_survives_clear() {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, u64::MAX, u64::MAX, 1).unwrap())
                .unwrap();
        let layer_count = CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 3;
        for global_layer in 0..layer_count {
            manager
                .record_attention_scan(
                    global_layer,
                    global_layer % 2 == 0,
                    1,
                    global_layer as u64 + 1,
                    global_layer as u64 + 7,
                )
                .unwrap();
        }

        let report = manager.report().unwrap();
        assert_eq!(report.per_layer.len(), CACHE_RESIDENCY_LAYER_REPORT_LIMIT);
        assert_eq!(report.per_layer_overflow_layers, 0);
        assert_eq!(
            report
                .per_layer
                .iter()
                .map(|layer| layer.global_layer)
                .collect::<Vec<_>>(),
            (0..CACHE_RESIDENCY_LAYER_REPORT_LIMIT).collect::<Vec<_>>()
        );
        let mut aggregate = CacheLayerResidencyStats::default();
        for layer in &report.per_layer {
            aggregate.accumulate(&layer.stats);
        }
        aggregate.accumulate(&report.per_layer_overflow);
        assert_eq!(
            aggregate.prefill_full_attention_blocks,
            report.prefill_full_attention_blocks
        );
        assert_eq!(
            aggregate.prefill_full_attention_bytes,
            report.prefill_full_attention_bytes
        );
        assert_eq!(
            aggregate.decode_full_attention_blocks,
            report.decode_full_attention_blocks
        );
        assert_eq!(
            aggregate.decode_full_attention_bytes,
            report.decode_full_attention_bytes
        );
        assert_eq!(
            aggregate.attention_scratch_peak_bytes,
            report.attention_scratch_peak_bytes
        );
        assert_eq!(report.per_layer_overflow.prefill_full_attention_blocks, 2);
        assert_eq!(report.per_layer_overflow.decode_full_attention_blocks, 1);

        manager.clear().unwrap();
        let after_clear = manager.report().unwrap();
        assert_eq!(
            after_clear.per_layer.len(),
            CACHE_RESIDENCY_LAYER_REPORT_LIMIT
        );
        assert_eq!(
            after_clear.prefill_full_attention_blocks,
            report.prefill_full_attention_blocks
        );
        assert_eq!(
            after_clear.per_layer_overflow.decode_full_attention_bytes,
            report.per_layer_overflow.decode_full_attention_bytes
        );
        assert!(after_clear
            .per_layer
            .iter()
            .all(|layer| layer.stats.current_device_bytes == 0
                && layer.stats.current_host_bytes == 0
                && layer.stats.current_disk_bytes == 0));
    }

    fn execution_key_value_block(stream: &Stream) -> CacheBlockArrays {
        let keys = Array::zeros::<f32>(&[1, 1, 2, 1], stream).unwrap();
        let values = Array::ones::<f32>(&[1, 1, 2, 1], stream).unwrap();
        eval([&keys, &values]).unwrap();
        stream.synchronize().unwrap();
        CacheBlockArrays::KeyValue { keys, values }
    }

    fn f32_storage_pointers(arrays: &CacheBlockArrays) -> [usize; 2] {
        arrays
            .arrays()
            .map(|array| array.evaluated().unwrap().as_slice::<f32>().as_ptr() as usize)
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn promotion_waits_for_pending_write_without_overcommitting_host_storage() {
        let directory = tempfile::tempdir().unwrap();
        let options = PagedCacheOptions::new(2, 32, 16, 1)
            .unwrap()
            .with_live_disk(directory.path(), 1024, 1)
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let worker = manager.disk_worker.as_ref().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(99),
                    kind: DiskOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let first = manager
            .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
            .unwrap();
        manager
            .seal_block(0, 2, 4, None, execution_key_value_block(&stream), false)
            .unwrap();
        manager
            .seal_block(0, 4, 6, None, execution_key_value_block(&stream), false)
            .unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.in_flight_write_bytes, 16);

        let promotion_manager = manager.clone();
        let (promoted_tx, promoted_rx) = mpsc::channel();
        let promotion_thread = thread::spawn(move || {
            let promotion_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let result = promotion_manager.lease_block(&first, &promotion_stream);
            promoted_tx.send(result.map(drop)).unwrap();
        });
        match promoted_rx.recv_timeout(Duration::from_millis(20)) {
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            result => panic!("promotion completed before host capacity was released: {result:?}"),
        }
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.in_flight_write_bytes, 16);
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());

        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        promoted_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("promotion did not finish after writeback released host capacity")
            .unwrap();
        promotion_thread.join().unwrap();
        let report = manager.report().unwrap();
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
        assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn host_to_disk_demotion_returns_before_background_write_completes() {
        let directory = tempfile::tempdir().unwrap();
        let options = PagedCacheOptions::new(2, 16, 16, 1)
            .unwrap()
            .with_live_disk(directory.path(), 1024, 1)
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let worker = manager.disk_worker.as_ref().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(99),
                    kind: DiskOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        manager
            .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
            .unwrap();
        let second = execution_key_value_block(&stream);
        let background_manager = manager.clone();
        let (sealed_tx, sealed_rx) = mpsc::channel();
        let seal_thread = thread::spawn(move || {
            sealed_tx
                .send(background_manager.seal_block(0, 2, 4, None, second, false))
                .unwrap();
        });

        sealed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("demotion waited for the blocked disk worker")
            .unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.in_flight_write_blocks, 1);
        assert_eq!(report.in_flight_write_bytes, 16);
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.host_blocks, 1);
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
        assert_eq!(report.disk_demotions, 0);

        // A third block needs the same host slot. It must wait for the pending
        // write to commit instead of retaining another host allocation beyond
        // the byte budget.
        let third = execution_key_value_block(&stream);
        let waiting_manager = manager.clone();
        let (third_tx, third_rx) = mpsc::channel();
        let third_thread = thread::spawn(move || {
            third_tx
                .send(waiting_manager.seal_block(0, 4, 6, None, third, false))
                .unwrap();
        });
        assert!(matches!(
            third_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.in_flight_write_bytes, 16);

        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        seal_thread.join().unwrap();
        third_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("host-capacity wait did not finish after writeback")
            .unwrap();
        third_thread.join().unwrap();
        for _ in 0..100 {
            let report = manager.report().unwrap();
            if report.disk_demotions >= 2 {
                assert_eq!(report.in_flight_write_blocks, 0);
                assert_eq!(report.disk_blocks, 2);
                assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
                assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
                assert!(report.in_flight_waits >= 1);
                let layer = report
                    .per_layer
                    .iter()
                    .find(|layer| layer.global_layer == 0)
                    .unwrap();
                assert_eq!(layer.stats.disk_demotions, report.disk_demotions);
                assert_eq!(layer.stats.in_flight_waits, report.in_flight_waits);
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("background disk write did not commit");
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn host_demotion_and_promotion_replace_physical_array_copies() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        // Two 16-byte blocks fit on the device. A third block forces the oldest
        // one to a distinct CPU allocation while retaining one recent block.
        let options = PagedCacheOptions::new(2, 32, 16, 1).unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let first_arrays = execution_key_value_block(&stream);
        let first_device_pointers = f32_storage_pointers(&first_arrays);
        let first = manager
            .seal_block(0, 0, 2, None, first_arrays, false)
            .unwrap();
        manager
            .seal_block(0, 2, 4, None, execution_key_value_block(&stream), false)
            .unwrap();
        manager
            .seal_block(0, 4, 6, None, execution_key_value_block(&stream), false)
            .unwrap();

        let first_host_pointers = {
            let state = manager.lock().unwrap();
            let record = state.blocks.get(&first).unwrap();
            assert_eq!(record.tier, CacheTier::Host);
            let arrays = record.arrays.as_ref().unwrap();
            f32_storage_pointers(arrays)
        };
        assert_ne!(first_host_pointers, first_device_pointers);
        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 2);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 32);
        assert_eq!(report.current_host_bytes, 16);

        let lease = manager.lease_block(&first, &stream).unwrap();
        let promoted_pointers = f32_storage_pointers(lease.arrays());
        assert_ne!(promoted_pointers, first_host_pointers);
        match lease.arrays() {
            CacheBlockArrays::KeyValue { keys, values } => {
                assert_eq!(keys.evaluated().unwrap().as_slice::<f32>(), &[0.0, 0.0]);
                assert_eq!(values.evaluated().unwrap().as_slice::<f32>(), &[1.0, 1.0]);
            }
            CacheBlockArrays::CompressedLatentRotary { .. } => unreachable!(),
        }
        {
            let state = manager.lock().unwrap();
            let record = state.blocks.get(&first).unwrap();
            assert_eq!(record.tier, CacheTier::Device);
            assert_eq!(
                f32_storage_pointers(record.arrays.as_ref().unwrap()),
                promoted_pointers
            );
        }
        drop(lease);

        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 2);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 32);
        assert_eq!(report.current_host_bytes, 16);
        assert_eq!(report.host_promotions, 1);
        assert_eq!(report.host_demotions, 2);
        let layer = report
            .per_layer
            .iter()
            .find(|layer| layer.global_layer == 0)
            .unwrap();
        assert_eq!(layer.stats.host_promotions, report.host_promotions);
        assert_eq!(layer.stats.host_demotions, report.host_demotions);
        assert_eq!(layer.stats.transfer_bytes, report.transfer_bytes);
        assert_eq!(layer.stats.demand_misses, report.demand_misses);
        assert_eq!(first.representation, CacheRepresentation::KeyValue);
    }
}
