//! Experimental bounded streaming of dense execution units from safetensors.
//!
//! The worker in this module performs only disk-to-host materialization. Device
//! promotion remains on the caller's ordered execution path because MLX does
//! not currently expose the cross-stream fences needed for arbitrary transfer
//! overlap.

use std::{
    collections::{BTreeMap, BTreeSet},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{mpsc, Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{
    offload::{CacheEvictionPolicy, MemoryTier, OffloadUnitId},
    residency::{ResidencyManager, ResidentUnitLease},
};

/// Public controls for experimental dense disk streaming.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DenseDiskStreamLoadOptions {
    /// Finite logical device parameter budget, including pinned static weights.
    pub device_budget_bytes: u64,
    /// Finite logical host layer budget. Zero selects direct disk-to-device loading.
    pub host_budget_bytes: u64,
    /// Number of current and imminent layer host copies protected from eviction.
    pub host_lookahead: usize,
    /// Number of current and imminent layer device copies protected from eviction.
    pub device_lookahead: usize,
    /// Maximum number of pending background host materializations.
    pub background_queue_capacity: usize,
    /// Deterministic ordering used when unprotected cached copies must be evicted.
    pub eviction_policy: CacheEvictionPolicy,
    /// Maximum number of checkpoint payload shards retained as mappings.
    pub max_mapped_shards: usize,
    /// Reject checkpoint tensors unrelated to the adapter's parameter tree.
    pub strict_loading: bool,
    /// Sample MLX allocator memory after a forward pass.
    pub sample_mlx_memory: bool,
    /// Sample process memory and page-fault counters after a forward pass.
    pub sample_process_memory: bool,
}

impl DenseDiskStreamLoadOptions {
    /// Creates strict streaming options with finite tier budgets.
    pub fn new(
        device_budget_bytes: u64,
        host_budget_bytes: u64,
        host_lookahead: usize,
        device_lookahead: usize,
        background_queue_capacity: usize,
    ) -> Result<Self, DenseStreamError> {
        let options = Self {
            device_budget_bytes,
            host_budget_bytes,
            host_lookahead,
            device_lookahead,
            background_queue_capacity,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
            max_mapped_shards: crate::weight_store::DEFAULT_MAX_MAPPED_SHARDS,
            strict_loading: true,
            sample_mlx_memory: false,
            sample_process_memory: false,
        };
        options.validate()?;
        Ok(options)
    }

    /// Revalidates public fields after caller customization.
    pub fn validate(self) -> Result<(), DenseStreamError> {
        if self.device_lookahead == 0 {
            return Err(DenseStreamError::ZeroDeviceLookahead);
        }
        if self.host_budget_bytes == 0 {
            if self.host_lookahead != 0 || self.background_queue_capacity != 0 {
                return Err(DenseStreamError::HostDisabledControls);
            }
        } else {
            if self.host_lookahead == 0 {
                return Err(DenseStreamError::ZeroHostLookahead);
            }
            if self.background_queue_capacity == 0 {
                return Err(DenseStreamError::ZeroQueueCapacity);
            }
        }
        Ok(())
    }

    /// Selects deterministic cache eviction.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }
}

impl Default for DenseDiskStreamLoadOptions {
    fn default() -> Self {
        Self::new(4 << 30, 16 << 30, 2, 1, 2)
            .expect("default dense disk streaming controls are valid")
    }
}

/// Immutable background-prefetch observations.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct BackgroundPrefetchReport {
    /// Requests accepted for background execution.
    pub submitted: u64,
    /// Duplicate queued or active requests folded into existing work.
    pub coalesced: u64,
    /// Requests begun by the worker.
    pub started: u64,
    /// Requests completed successfully.
    pub completed: u64,
    /// Stale requests skipped before publication.
    pub cancelled: u64,
    /// Requests that returned an error.
    pub failed: u64,
    /// Configured bounded queue capacity.
    pub queue_capacity: usize,
    /// Largest observed number of queued requests.
    pub peak_queue_occupancy: usize,
    /// Submissions that waited for bounded queue capacity.
    pub backpressure_count: u64,
    /// Time spent waiting for bounded queue capacity.
    pub backpressure_duration: Duration,
    /// Demand acquisitions that waited for queued or active work.
    pub demand_waits: u64,
    /// Time spent waiting for demanded background work.
    pub demand_wait_duration: Duration,
    /// Prefetches found resident before demand.
    pub ready_before_demand: u64,
    /// Prefetches still active when demanded.
    pub in_flight_at_demand: u64,
    /// Completed prefetched copies evicted before their first demand.
    pub evicted_before_use: u64,
}

#[derive(Debug)]
enum WorkerMessage {
    Prefetch { generation: u64, id: OffloadUnitId },
    Shutdown,
}

#[derive(Default)]
struct SharedState {
    generation: u64,
    queued: BTreeSet<OffloadUnitId>,
    in_flight: BTreeSet<OffloadUnitId>,
    completed: BTreeSet<OffloadUnitId>,
    failures: BTreeMap<OffloadUnitId, String>,
    report: BackgroundPrefetchReport,
}

/// One bounded, deterministically joined disk-to-host worker.
pub(crate) struct BackgroundLayerPrefetch {
    manager: ResidencyManager,
    sender: mpsc::SyncSender<WorkerMessage>,
    shared: Arc<(Mutex<SharedState>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}

impl BackgroundLayerPrefetch {
    pub(crate) fn new(
        manager: ResidencyManager,
        capacity: usize,
    ) -> Result<Self, DenseStreamError> {
        if capacity == 0 {
            return Err(DenseStreamError::ZeroQueueCapacity);
        }
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let shared = Arc::new((Mutex::new(SharedState::default()), Condvar::new()));
        shared
            .0
            .lock()
            .map_err(|_| DenseStreamError::StatePoisoned)?
            .report
            .queue_capacity = capacity;
        let worker_shared = Arc::clone(&shared);
        let worker_manager = manager.clone();
        let worker = thread::Builder::new()
            .name("safemlx-dense-layer-prefetch".into())
            .spawn(move || worker_loop(worker_manager, receiver, worker_shared))?;
        Ok(Self {
            manager,
            sender,
            shared,
            worker: Some(worker),
        })
    }

    pub(crate) fn submit(&self, id: &OffloadUnitId) -> Result<(), DenseStreamError> {
        let resident = self.manager.is_resident(id, MemoryTier::Host)?;
        let generation = {
            let mut state = self
                .shared
                .0
                .lock()
                .map_err(|_| DenseStreamError::StatePoisoned)?;
            if state.queued.contains(id) || state.in_flight.contains(id) {
                state.report.coalesced = state.report.coalesced.saturating_add(1);
                return Ok(());
            }
            if state.completed.contains(id) && !resident {
                state.completed.remove(id);
                state.report.evicted_before_use = state.report.evicted_before_use.saturating_add(1);
            }
            if resident {
                state.completed.insert(id.clone());
                state.report.coalesced = state.report.coalesced.saturating_add(1);
                return Ok(());
            }
            state.queued.insert(id.clone());
            state.report.submitted = state.report.submitted.saturating_add(1);
            state.report.peak_queue_occupancy = state
                .report
                .peak_queue_occupancy
                .max(state.queued.len().min(state.report.queue_capacity));
            state.generation
        };
        let message = WorkerMessage::Prefetch {
            generation,
            id: id.clone(),
        };
        match self.sender.try_send(message) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(message)) => {
                let started = Instant::now();
                self.sender
                    .send(message)
                    .map_err(|_| DenseStreamError::WorkerDisconnected)?;
                let mut state = self
                    .shared
                    .0
                    .lock()
                    .map_err(|_| DenseStreamError::StatePoisoned)?;
                state.report.backpressure_count = state.report.backpressure_count.saturating_add(1);
                state.report.backpressure_duration = state
                    .report
                    .backpressure_duration
                    .saturating_add(started.elapsed());
                Ok(())
            }
            Err(mpsc::TrySendError::Disconnected(_)) => Err(DenseStreamError::WorkerDisconnected),
        }
    }

    pub(crate) fn acquire(
        &self,
        id: &OffloadUnitId,
    ) -> Result<ResidentUnitLease, DenseStreamError> {
        let started = Instant::now();
        let mut waited = false;
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| DenseStreamError::StatePoisoned)?;
        if state.in_flight.contains(id) {
            state.report.in_flight_at_demand = state.report.in_flight_at_demand.saturating_add(1);
        }
        while state.queued.contains(id) || state.in_flight.contains(id) {
            waited = true;
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| DenseStreamError::StatePoisoned)?;
        }
        if let Some(message) = state.failures.remove(id) {
            return Err(DenseStreamError::PrefetchFailed {
                id: id.clone(),
                message,
            });
        }
        if state.completed.remove(id) {
            state.report.ready_before_demand = state.report.ready_before_demand.saturating_add(1);
        }
        if waited {
            state.report.demand_waits = state.report.demand_waits.saturating_add(1);
            state.report.demand_wait_duration = state
                .report
                .demand_wait_duration
                .saturating_add(started.elapsed());
        }
        drop(state);
        Ok(self.manager.acquire(id, MemoryTier::Host)?)
    }

    pub(crate) fn cancel(&self) -> Result<(), DenseStreamError> {
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| DenseStreamError::StatePoisoned)?;
        state.generation = state.generation.wrapping_add(1);
        let cancelled = state.queued.len() as u64;
        state.report.cancelled = state.report.cancelled.saturating_add(cancelled);
        while !state.queued.is_empty() || !state.in_flight.is_empty() {
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| DenseStreamError::StatePoisoned)?;
        }
        let failure = state
            .failures
            .iter()
            .next()
            .map(|(id, message)| (id.clone(), message.clone()));
        state.completed.clear();
        state.failures.clear();
        self.shared.1.notify_all();
        match failure {
            Some((id, message)) => Err(DenseStreamError::PrefetchFailed { id, message }),
            None => Ok(()),
        }
    }

    pub(crate) fn report(&self) -> Result<BackgroundPrefetchReport, DenseStreamError> {
        Ok(self
            .shared
            .0
            .lock()
            .map_err(|_| DenseStreamError::StatePoisoned)?
            .report)
    }
}

impl Drop for BackgroundLayerPrefetch {
    fn drop(&mut self) {
        let _ = self.cancel();
        let _ = self.sender.send(WorkerMessage::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn worker_loop(
    manager: ResidencyManager,
    receiver: mpsc::Receiver<WorkerMessage>,
    shared: Arc<(Mutex<SharedState>, Condvar)>,
) {
    while let Ok(message) = receiver.recv() {
        let WorkerMessage::Prefetch { generation, id } = message else {
            break;
        };
        {
            let Ok(mut state) = shared.0.lock() else {
                break;
            };
            state.queued.remove(&id);
            if generation != state.generation {
                shared.1.notify_all();
                continue;
            }
            state.in_flight.insert(id.clone());
            state.report.started = state.report.started.saturating_add(1);
            shared.1.notify_all();
        }
        let result = catch_unwind(AssertUnwindSafe(|| manager.prefetch(&id, MemoryTier::Host)))
            .map_err(|payload| {
                payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "background materialization panicked".to_string())
            })
            .and_then(|result| result.map_err(|error| error.to_string()));
        let Ok(mut state) = shared.0.lock() else {
            break;
        };
        state.in_flight.remove(&id);
        if generation != state.generation {
            state.report.cancelled = state.report.cancelled.saturating_add(1);
        } else {
            match result {
                Ok(_) => {
                    state.completed.insert(id);
                    state.report.completed = state.report.completed.saturating_add(1);
                }
                Err(error) => {
                    state.failures.insert(id, error);
                    state.report.failed = state.report.failed.saturating_add(1);
                }
            }
        }
        shared.1.notify_all();
    }
}

/// Structured validation and worker failures for dense disk streaming.
#[derive(Debug, thiserror::Error)]
pub enum DenseStreamError {
    /// Device lookahead must include the current execution unit.
    #[error("dense disk streaming device lookahead must be nonzero")]
    ZeroDeviceLookahead,
    /// Enabled host caching needs a protected current unit.
    #[error("dense disk streaming host lookahead must be nonzero when the host budget is enabled")]
    ZeroHostLookahead,
    /// Enabled background work requires bounded capacity.
    #[error("dense disk streaming background queue capacity must be nonzero when host caching is enabled")]
    ZeroQueueCapacity,
    /// Direct-to-device mode cannot configure host-only controls.
    #[error("dense disk streaming with a zero host budget requires zero host lookahead and queue capacity")]
    HostDisabledControls,
    /// Shared worker state was poisoned.
    #[error("dense disk streaming worker state is poisoned")]
    StatePoisoned,
    /// The worker ended before accepting or completing required work.
    #[error("dense disk streaming worker disconnected")]
    WorkerDisconnected,
    /// A worker-side materialization failed and was observed by demand.
    #[error("background materialization of {id} failed: {message}")]
    PrefetchFailed {
        /// Failed unit.
        id: OffloadUnitId,
        /// Original residency error.
        message: String,
    },
    /// A residency transition failed.
    #[error(transparent)]
    Residency(#[from] crate::residency::ResidencyError),
    /// Worker creation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        offload::{OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy},
        residency::{OffloadUnit, WeightBinding},
        weight_store::{SafetensorsWeightStore, TensorSelection},
    };
    use safemlx::{Device, DeviceType, Stream};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    #[test]
    fn configuration_requires_meaningful_finite_controls() {
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 1, 1, 0, 1),
            Err(DenseStreamError::ZeroDeviceLookahead)
        ));
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 1, 0, 1, 1),
            Err(DenseStreamError::ZeroHostLookahead)
        ));
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 1, 1, 1, 0),
            Err(DenseStreamError::ZeroQueueCapacity)
        ));
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 0, 1, 1, 0),
            Err(DenseStreamError::HostDisabledControls)
        ));
        assert!(DenseDiskStreamLoadOptions::new(1, 0, 0, 1, 0).is_ok());
    }

    #[test]
    fn mutated_public_controls_are_revalidated() {
        let mut options = DenseDiskStreamLoadOptions::default();
        options.background_queue_capacity = 0;
        assert!(matches!(
            options.validate(),
            Err(DenseStreamError::ZeroQueueCapacity)
        ));
    }

    #[test]
    fn duplicate_background_requests_coalesce_and_join_demand() {
        let directory = tempfile::tempdir().unwrap();
        let bytes = [1i32, 2]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::I32, vec![2], &bytes).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let id = OffloadUnitId::new("layer.0").unwrap();
        let binding = WeightBinding::new("weight", "weight", TensorSelection::Full, 8).unwrap();
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [
                OffloadUnitSpec::new(id.clone(), 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                    .unwrap(),
            ],
        )
        .unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let manager = ResidencyManager::new(
            store,
            plan,
            [OffloadUnit::new(id.clone(), [binding]).unwrap()],
            stream.clone(),
            stream,
        )
        .unwrap();
        manager.initialize().unwrap();

        let prefetch = BackgroundLayerPrefetch::new(manager, 1).unwrap();
        prefetch.submit(&id).unwrap();
        prefetch.submit(&id).unwrap();
        let lease = prefetch.acquire(&id).unwrap();
        assert_eq!(lease.array("weight").unwrap().shape(), &[2]);
        let report = prefetch.report().unwrap();
        assert_eq!(report.submitted, 1);
        assert!(report.coalesced >= 1);
        assert_eq!(report.completed, 1);
    }
}
