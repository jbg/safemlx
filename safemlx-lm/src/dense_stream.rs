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

type HostPrefetchOperation =
    Arc<dyn Fn(&OffloadUnitId) -> Result<(), String> + Send + Sync + 'static>;

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
        let worker_manager = manager.clone();
        let operation = Arc::new(move |id: &OffloadUnitId| {
            worker_manager
                .prefetch(id, MemoryTier::Host)
                .map(|_| ())
                .map_err(|error| error.to_string())
        });
        Self::new_with_operation(manager, capacity, operation)
    }

    fn new_with_operation(
        manager: ResidencyManager,
        capacity: usize,
        operation: HostPrefetchOperation,
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
        let worker = thread::Builder::new()
            .name("safemlx-dense-layer-prefetch".into())
            .spawn(move || worker_loop(operation, receiver, worker_shared))?;
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
                let disconnected = self.sender.send(message).is_err();
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
                if disconnected {
                    rollback_submission(&mut state, id);
                    self.shared.1.notify_all();
                    Err(DenseStreamError::WorkerDisconnected)
                } else {
                    Ok(())
                }
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                let mut state = self
                    .shared
                    .0
                    .lock()
                    .map_err(|_| DenseStreamError::StatePoisoned)?;
                rollback_submission(&mut state, id);
                self.shared.1.notify_all();
                Err(DenseStreamError::WorkerDisconnected)
            }
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
        self.shared.1.notify_all();
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

    #[cfg(test)]
    fn wait_idle(&self) -> Result<(), DenseStreamError> {
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| DenseStreamError::StatePoisoned)?;
        while !state.queued.is_empty() || !state.in_flight.is_empty() {
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| DenseStreamError::StatePoisoned)?;
        }
        Ok(())
    }
}

fn rollback_submission(state: &mut SharedState, id: &OffloadUnitId) {
    if state.queued.remove(id) {
        state.report.submitted = state.report.submitted.saturating_sub(1);
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
    operation: HostPrefetchOperation,
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
        let result = catch_unwind(AssertUnwindSafe(|| operation(&id)))
            .map_err(|payload| {
                payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "background materialization panicked".to_string())
            })
            .and_then(|result| result);
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
    /// Forward telemetry lifecycle calls were inconsistent.
    #[error("invalid dense streaming forward telemetry state: {0}")]
    InvalidForwardTelemetry(&'static str),
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

    #[derive(Default)]
    struct OperationGate {
        state: Mutex<(usize, bool)>,
        changed: Condvar,
    }

    impl OperationGate {
        fn enter_and_wait(&self) {
            let mut state = self.state.lock().unwrap();
            state.0 += 1;
            self.changed.notify_all();
            while !state.1 {
                state = self.changed.wait(state).unwrap();
            }
        }

        fn wait_for_starts(&self, expected: usize) {
            let mut state = self.state.lock().unwrap();
            while state.0 < expected {
                state = self.changed.wait(state).unwrap();
            }
        }

        fn release(&self) {
            let mut state = self.state.lock().unwrap();
            state.1 = true;
            self.changed.notify_all();
        }
    }

    fn test_manager(
        tensors: Vec<(String, Dtype, Vec<u8>)>,
        host_budget: u64,
    ) -> (tempfile::TempDir, ResidencyManager, Vec<OffloadUnitId>) {
        let directory = tempfile::tempdir().unwrap();
        serialize_to_file(
            tensors.iter().map(|(name, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, vec![2], bytes).unwrap(),
                )
            }),
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let mut specs = Vec::new();
        let mut definitions = Vec::new();
        let mut ids = Vec::new();
        for (index, (key, _, bytes)) in tensors.iter().enumerate() {
            let id = OffloadUnitId::new(format!("layer.{index}")).unwrap();
            let expected_bytes = bytes.len() as u64;
            let binding =
                WeightBinding::new("weight", key, TensorSelection::Full, expected_bytes).unwrap();
            specs.push(
                OffloadUnitSpec::new(
                    id.clone(),
                    expected_bytes,
                    ResidencyPolicy::Cacheable,
                    MemoryTier::Disk,
                )
                .unwrap(),
            );
            definitions.push(OffloadUnit::new(id.clone(), [binding]).unwrap());
            ids.push(id);
        }
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(u64::MAX), Some(host_budget), 1).unwrap(),
            specs,
        )
        .unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let manager =
            ResidencyManager::new(store, plan, definitions, stream.clone(), stream).unwrap();
        manager.initialize().unwrap();
        (directory, manager, ids)
    }

    fn i32_manager(
        count: usize,
        host_budget: u64,
    ) -> (tempfile::TempDir, ResidencyManager, Vec<OffloadUnitId>) {
        test_manager(
            (0..count)
                .map(|index| {
                    (
                        format!("weight.{index}"),
                        Dtype::I32,
                        [index as i32, index as i32 + 1]
                            .into_iter()
                            .flat_map(i32::to_le_bytes)
                            .collect(),
                    )
                })
                .collect(),
            host_budget,
        )
    }

    fn blocking_operation(
        manager: ResidencyManager,
        gate: Arc<OperationGate>,
    ) -> HostPrefetchOperation {
        Arc::new(move |id| {
            gate.enter_and_wait();
            manager
                .prefetch(id, MemoryTier::Host)
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
    }

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
        let options = DenseDiskStreamLoadOptions {
            background_queue_capacity: 0,
            ..DenseDiskStreamLoadOptions::default()
        };
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

    #[test]
    fn bounded_queue_backpressures_without_dropping_required_work() {
        let (_directory, manager, ids) = i32_manager(3, 24);
        let gate = Arc::new(OperationGate::default());
        let prefetch = Arc::new(
            BackgroundLayerPrefetch::new_with_operation(
                manager.clone(),
                1,
                blocking_operation(manager, Arc::clone(&gate)),
            )
            .unwrap(),
        );
        prefetch.submit(&ids[0]).unwrap();
        gate.wait_for_starts(1);
        prefetch.submit(&ids[1]).unwrap();

        let (started_tx, started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let submitting = Arc::clone(&prefetch);
        let required = ids[2].clone();
        let submitter = thread::spawn(move || {
            started_tx.send(()).unwrap();
            let outcome = submitting.submit(&required);
            result_tx.send(outcome).unwrap();
        });
        started_rx.recv().unwrap();
        gate.release();
        result_rx.recv().unwrap().unwrap();
        submitter.join().unwrap();
        prefetch.wait_idle().unwrap();

        let lease = prefetch.acquire(&ids[2]).unwrap();
        assert_eq!(lease.array("weight").unwrap().shape(), &[2]);
        let report = prefetch.report().unwrap();
        assert_eq!(report.queue_capacity, 1);
        assert_eq!(report.peak_queue_occupancy, 1);
        assert_eq!(report.submitted, 3);
        assert_eq!(report.started, 3);
        assert_eq!(report.completed, 3);
        assert_eq!(report.failed, 0);
        assert!(report.backpressure_count >= 1);
    }

    #[test]
    fn cancellation_skips_queued_and_active_generations() {
        let (_directory, manager, ids) = i32_manager(2, 16);
        let gate = Arc::new(OperationGate::default());
        let prefetch = Arc::new(
            BackgroundLayerPrefetch::new_with_operation(
                manager.clone(),
                1,
                blocking_operation(manager, Arc::clone(&gate)),
            )
            .unwrap(),
        );
        prefetch.submit(&ids[0]).unwrap();
        gate.wait_for_starts(1);
        prefetch.submit(&ids[1]).unwrap();

        let shared = Arc::clone(&prefetch.shared);
        let cancelling = Arc::clone(&prefetch);
        let (cancelled_tx, cancelled_rx) = mpsc::channel();
        thread::spawn(move || cancelled_tx.send(cancelling.cancel()).unwrap());
        let mut state = shared.0.lock().unwrap();
        while state.generation == 0 {
            state = shared.1.wait(state).unwrap();
        }
        drop(state);
        gate.release();
        cancelled_rx.recv().unwrap().unwrap();

        let report = prefetch.report().unwrap();
        assert_eq!(report.started, 1);
        assert_eq!(report.completed, 0);
        assert_eq!(report.cancelled, 2);
        assert_eq!(report.failed, 0);
    }

    #[test]
    fn worker_errors_and_panics_reach_demand_and_release_reservations() {
        let unsupported = vec![("unsupported".to_string(), Dtype::F8_E5M2, vec![0x3c, 0x40])];
        let (_directory, manager, ids) = test_manager(unsupported, 2);
        let prefetch = BackgroundLayerPrefetch::new(manager.clone(), 1).unwrap();
        prefetch.submit(&ids[0]).unwrap();
        let error = match prefetch.acquire(&ids[0]) {
            Ok(_) => panic!("unsupported stored dtype unexpectedly prefetched"),
            Err(error) => error,
        };
        assert!(matches!(error, DenseStreamError::PrefetchFailed { .. }));
        let report = manager.report().unwrap();
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Host), 0);
        assert!(!report.units()[0].host_resident());
        assert_eq!(prefetch.report().unwrap().failed, 1);

        let (_directory, manager, ids) = i32_manager(1, 8);
        let operation: HostPrefetchOperation = Arc::new(|_| -> Result<(), String> {
            panic!("controlled worker panic");
        });
        let prefetch =
            BackgroundLayerPrefetch::new_with_operation(manager.clone(), 1, operation).unwrap();
        prefetch.submit(&ids[0]).unwrap();
        let error = match prefetch.acquire(&ids[0]) {
            Ok(_) => panic!("panicking operation unexpectedly prefetched"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("controlled worker panic"));
        assert_eq!(prefetch.report().unwrap().failed, 1);
        assert!(!manager.is_resident(&ids[0], MemoryTier::Host).unwrap());
    }

    #[test]
    fn completed_prefetch_evicted_before_demand_is_counted() {
        let (_directory, manager, ids) = i32_manager(2, 8);
        let prefetch = BackgroundLayerPrefetch::new(manager.clone(), 1).unwrap();
        prefetch.submit(&ids[0]).unwrap();
        prefetch.wait_idle().unwrap();
        prefetch.submit(&ids[1]).unwrap();
        prefetch.wait_idle().unwrap();
        assert!(!manager.is_resident(&ids[0], MemoryTier::Host).unwrap());

        prefetch.submit(&ids[0]).unwrap();
        prefetch.wait_idle().unwrap();
        assert_eq!(prefetch.report().unwrap().evicted_before_use, 1);
    }

    #[test]
    fn disconnected_submission_rolls_back_and_drop_joins_worker() {
        let (_directory, manager, ids) = i32_manager(1, 8);
        let mut disconnected = BackgroundLayerPrefetch::new(manager, 1).unwrap();
        disconnected.sender.send(WorkerMessage::Shutdown).unwrap();
        disconnected.worker.take().unwrap().join().unwrap();
        assert!(matches!(
            disconnected.submit(&ids[0]),
            Err(DenseStreamError::WorkerDisconnected)
        ));
        assert_eq!(disconnected.report().unwrap().submitted, 0);
        disconnected.cancel().unwrap();

        let (_directory, manager, ids) = i32_manager(1, 8);
        let gate = Arc::new(OperationGate::default());
        let prefetch = BackgroundLayerPrefetch::new_with_operation(
            manager.clone(),
            1,
            blocking_operation(manager, Arc::clone(&gate)),
        )
        .unwrap();
        prefetch.submit(&ids[0]).unwrap();
        gate.wait_for_starts(1);
        let shared = Arc::clone(&prefetch.shared);
        let (finished_tx, finished_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(prefetch);
            finished_tx.send(()).unwrap();
        });
        let mut state = shared.0.lock().unwrap();
        while state.generation == 0 {
            state = shared.1.wait(state).unwrap();
        }
        drop(state);
        gate.release();
        finished_rx.recv().unwrap();
    }
}
