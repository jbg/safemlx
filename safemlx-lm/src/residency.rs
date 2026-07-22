//! Budgeted, architecture-independent residency for immutable weight units.
//!
//! A [`crate::residency::ResidencyManager`] moves caller-defined groups of
//! checkpoint selections from a [`crate::weight_store::WeightStore`] into
//! evaluated host or execution-stream arrays. The
//! manager accounts for logical host and device copies independently, even on
//! unified-memory systems. Missing units can be reserved and evaluated as one
//! batch. Completion remains host-synchronous because the pinned MLX C API has
//! no event or fence primitive.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, MutexGuard, Weak},
    time::Instant,
};

use safemlx::{transforms::eval, Array, DeviceType, Stream};

use crate::{
    offload::{
        CacheEvictionPolicy, MemoryTier, OffloadPlan, OffloadReport, OffloadTelemetry,
        OffloadUnitId, OffloadUnitSpec, PrefetchOutcome, ResidencyPolicy, TransferDirection,
    },
    weight_recipe::{DerivedWeightRecipe, WeightRecipeError},
    weight_store::{
        PendingWeightMaterialization, TensorSelection, WeightStore, WeightStoreDiagnostics,
        WeightStoreError,
    },
};

/// One named checkpoint selection within an atomic resident unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightBinding {
    name: String,
    checkpoint_key: String,
    selection: TensorSelection,
    recipe: Option<DerivedWeightRecipe>,
    expected_bytes: u64,
}

impl WeightBinding {
    /// Creates a binding with a stable local name and expected selected size.
    pub fn new(
        name: impl Into<String>,
        checkpoint_key: impl Into<String>,
        selection: TensorSelection,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ResidencyError::InvalidBindingName);
        }
        let checkpoint_key = checkpoint_key.into();
        if checkpoint_key.trim().is_empty() {
            return Err(ResidencyError::InvalidCheckpointKey { name });
        }
        if expected_bytes == 0 {
            return Err(ResidencyError::ZeroSizedBinding { name });
        }
        Ok(Self {
            name,
            checkpoint_key,
            selection,
            recipe: None,
            expected_bytes,
        })
    }

    /// Creates a binding backed by a composable derived-weight recipe.
    ///
    /// The recipe is validated against checkpoint metadata when the residency
    /// manager is constructed and materialized once on the host during
    /// initialization. Device promotion copies that transformed representation.
    pub fn from_recipe(
        name: impl Into<String>,
        recipe: DerivedWeightRecipe,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ResidencyError::InvalidBindingName);
        }
        let checkpoint_key = recipe
            .source_keys()
            .first()
            .map(|key| (*key).to_string())
            .ok_or_else(|| ResidencyError::Recipe {
                binding: name.clone(),
                source: WeightRecipeError::EmptyInputs,
            })?;
        if expected_bytes == 0 {
            return Err(ResidencyError::ZeroSizedBinding { name });
        }
        Ok(Self {
            name,
            checkpoint_key,
            selection: TensorSelection::Full,
            recipe: Some(recipe),
            expected_bytes,
        })
    }

    /// Returns the stable name used to look up a resident array.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the source checkpoint key.
    pub fn checkpoint_key(&self) -> &str {
        &self.checkpoint_key
    }

    /// Returns the checkpoint selection delegated to the weight store.
    pub fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Returns the derived recipe when this is not a direct binding.
    pub const fn recipe(&self) -> Option<&DerivedWeightRecipe> {
        self.recipe.as_ref()
    }

    /// Returns every checkpoint key consumed by this binding.
    pub fn checkpoint_keys(&self) -> Vec<&str> {
        match &self.recipe {
            Some(recipe) => recipe.source_keys(),
            None => vec![self.checkpoint_key.as_str()],
        }
    }

    /// Returns the expected logical and materialized byte length.
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }
}

/// A deterministic group of weight bindings managed as one atomic unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadUnit {
    id: OffloadUnitId,
    bindings: Vec<WeightBinding>,
}

impl OffloadUnit {
    /// Creates a non-empty unit and sorts its bindings by local name.
    pub fn new(
        id: OffloadUnitId,
        bindings: impl IntoIterator<Item = WeightBinding>,
    ) -> Result<Self, ResidencyError> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        if bindings.is_empty() {
            return Err(ResidencyError::EmptyUnit { id });
        }
        bindings.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = bindings
            .windows(2)
            .find(|pair| pair[0].name == pair[1].name)
        {
            return Err(ResidencyError::DuplicateBindingName {
                id,
                name: pair[0].name.clone(),
            });
        }
        Ok(Self { id, bindings })
    }

    /// Returns the plan identifier for this unit.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns bindings in stable local-name order.
    pub fn bindings(&self) -> &[WeightBinding] {
        &self.bindings
    }
}

/// A resident unit that prevents eviction of one tier until it is dropped.
pub struct ResidentUnitLease {
    id: OffloadUnitId,
    tier: MemoryTier,
    arrays: Arc<ResidentArrays>,
    pending: Option<Arc<PendingResidentSources>>,
    manager: Weak<ManagerInner>,
}

impl ResidentUnitLease {
    /// Returns the acquired unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the protected resident tier.
    pub const fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Looks up an immutable resident array by stable binding name.
    ///
    /// Consumers should not retain cloned `Array` handles beyond this lease if
    /// residency accounting is expected to remain authoritative. Arbitrary
    /// external array clones cannot be tracked by the manager.
    pub fn array(&self, name: &str) -> Result<&Array, ResidencyError> {
        self.arrays
            .arrays
            .get(name)
            .ok_or_else(|| ResidencyError::UnknownBinding {
                id: self.id.clone(),
                name: name.to_string(),
            })
    }

    /// Returns binding names in stable order.
    pub fn binding_names(&self) -> impl Iterator<Item = &str> {
        self.arrays.arrays.keys().map(String::as_str)
    }

    /// Releases deferred source dependencies after a dependent output was evaluated.
    pub fn complete_pending(&self) -> Result<(), ResidencyError> {
        if let Some(pending) = &self.pending {
            pending.complete_after_evaluation()?;
            self.clear_completed_pending(pending)?;
        }
        Ok(())
    }

    fn clear_completed_pending(
        &self,
        pending: &Arc<PendingResidentSources>,
    ) -> Result<(), ResidencyError> {
        if pending.is_pending() {
            return Ok(());
        }
        let Some(manager) = self.manager.upgrade() else {
            return Ok(());
        };
        let mut state = manager
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        let Some(copy) = state
            .units
            .get_mut(&self.id)
            .and_then(|unit| unit.copy_mut(self.tier))
        else {
            return Ok(());
        };
        if copy
            .pending
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, pending))
        {
            copy.pending = None;
        }
        Ok(())
    }
}

impl Drop for ResidentUnitLease {
    fn drop(&mut self) {
        if let Some(pending) = &self.pending {
            pending.abort();
        }
        let Some(manager) = self.manager.upgrade() else {
            return;
        };
        let Ok(mut state) = manager.state.lock() else {
            return;
        };
        let Some(unit) = state.units.get_mut(&self.id) else {
            return;
        };
        if let Some(copy) = unit.copy_mut(self.tier) {
            if self.pending.as_ref().is_some_and(|pending| {
                !pending.is_pending()
                    && copy
                        .pending
                        .as_ref()
                        .is_some_and(|current| Arc::ptr_eq(current, pending))
            }) {
                copy.pending = None;
            }
            copy.pins = copy.pins.saturating_sub(1);
        }
    }
}

/// A resident unit that can prevent progress under a finite budget.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyBlocker {
    /// Stable unit identifier.
    pub id: OffloadUnitId,
    /// Whether policy prevents eviction.
    pub pinned: bool,
    /// Number of active resident leases at the requested tier.
    pub in_use: u64,
    /// Whether the current execution window protects the unit.
    pub active_window: bool,
}

/// Structured failures from residency validation and state transitions.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    /// A named execution group had an empty identifier.
    #[error("resident execution group id must not be empty")]
    InvalidGroupId,
    /// An ordered layer window had no units.
    #[error("device layer window requires at least one ordered unit")]
    EmptyLayerWindow,
    /// The configured device layer window exceeded the ordered unit count.
    #[error("device layer window depth {depth} exceeds {layer_count} ordered units")]
    OversizedLayerWindow {
        /// Requested resident-layer bound.
        depth: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A layer index was outside the ordered sequence.
    #[error("device layer index {index} is outside {layer_count} ordered units")]
    InvalidLayerIndex {
        /// Requested index.
        index: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A binding name was empty or whitespace-only.
    #[error("weight binding names must not be empty")]
    InvalidBindingName,
    /// A binding checkpoint key was empty.
    #[error("weight binding {name:?} has an empty checkpoint key")]
    InvalidCheckpointKey {
        /// Invalid local binding name.
        name: String,
    },
    /// A binding declared no bytes.
    #[error("weight binding {name:?} must contain at least one byte")]
    ZeroSizedBinding {
        /// Invalid local binding name.
        name: String,
    },
    /// A unit had no bindings.
    #[error("residency unit {id} must contain at least one binding")]
    EmptyUnit {
        /// Invalid unit identifier.
        id: OffloadUnitId,
    },
    /// Two bindings in one unit had the same local name.
    #[error("residency unit {id} has duplicate binding name {name:?}")]
    DuplicateBindingName {
        /// Invalid unit identifier.
        id: OffloadUnitId,
        /// Duplicated local name.
        name: String,
    },
    /// More than one definition used the same plan identifier.
    #[error("duplicate residency unit definition: {id}")]
    DuplicateUnitDefinition {
        /// Duplicated identifier.
        id: OffloadUnitId,
    },
    /// A batched acquisition requested the same unit more than once.
    #[error("batched residency acquisition contains a duplicate unit")]
    DuplicateBatchUnit,
    /// The plan had no matching unit definition.
    #[error("offload plan unit {id} has no residency unit definition")]
    MissingUnitDefinition {
        /// Missing identifier.
        id: OffloadUnitId,
    },
    /// A definition had no matching plan entry.
    #[error("residency unit {id} is absent from the offload plan")]
    UnexpectedUnitDefinition {
        /// Unexpected identifier.
        id: OffloadUnitId,
    },
    /// Binding sizes did not sum to the plan's unit size.
    #[error(
        "residency unit {id} defines {actual_bytes} bytes but its plan reserves {planned_bytes}"
    )]
    UnitByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Bytes reserved by the plan.
        planned_bytes: u64,
        /// Sum of binding sizes.
        actual_bytes: u64,
    },
    /// A binding's selected checkpoint size contradicted its definition.
    #[error("binding {binding:?} in unit {id} selects {actual_bytes} bytes but declares {expected_bytes}")]
    BindingByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Binding name.
        binding: String,
        /// Declared size.
        expected_bytes: u64,
        /// Store-validated size.
        actual_bytes: u64,
    },
    /// A derived-weight recipe was invalid or could not be materialized.
    #[error("derived-weight recipe for binding {binding:?} failed: {source}")]
    Recipe {
        /// Local binding name.
        binding: String,
        /// Recipe failure.
        #[source]
        source: WeightRecipeError,
    },
    /// A caller requested disk as an MLX array target.
    #[error("{operation} requires Host or Device residency, not Disk")]
    InvalidTargetTier {
        /// Operation that rejected disk.
        operation: &'static str,
    },
    /// The configured source stream was not a CPU stream.
    #[error("the residency source stream must target the CPU")]
    InvalidSourceStream,
    /// A transition was requested before explicit initialization completed.
    #[error("the residency manager has not completed initialization")]
    NotInitialized,
    /// A requested unit does not exist.
    #[error("unknown residency unit {id}")]
    UnknownUnit {
        /// Unknown identifier.
        id: OffloadUnitId,
    },
    /// A binding lookup failed on a valid resident unit.
    #[error("residency unit {id} has no binding named {name:?}")]
    UnknownBinding {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Unknown local name.
        name: String,
    },
    /// A finite tier budget could not accommodate a unit.
    #[error("cannot make {required_bytes} bytes resident for {requested} in {tier:?}: budget {budget_bytes}, currently resident {resident_bytes}, blockers {blocking_units:?}")]
    BudgetExhausted {
        /// Requested unit.
        requested: OffloadUnitId,
        /// Requested resident tier.
        tier: MemoryTier,
        /// Full unit size.
        required_bytes: u64,
        /// Configured finite budget.
        budget_bytes: u64,
        /// Bytes still resident after eligible evictions.
        resident_bytes: u64,
        /// Stable list of protected residents.
        blocking_units: Vec<ResidencyBlocker>,
    },
    /// Explicit eviction targeted a pinned unit.
    #[error("pinned residency unit {id} cannot be evicted from {tier:?}")]
    PinnedEviction {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Protected tier.
        tier: MemoryTier,
    },
    /// Explicit eviction targeted an in-use unit.
    #[error("residency unit {id} has {pin_count} live leases in {tier:?}")]
    InUseEviction {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Protected tier.
        tier: MemoryTier,
        /// Active lease count.
        pin_count: u64,
    },
    /// Checked byte or recency arithmetic overflowed.
    #[error("residency arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: &'static str,
    },
    /// Persistent store validation or materialization failed.
    #[error(transparent)]
    WeightStore(#[from] WeightStoreError),
    /// An MLX copy or evaluation failed.
    #[error("MLX {operation} failed for residency unit {id}: {source}")]
    Mlx {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Failed operation.
        operation: &'static str,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// Explicit stream synchronization failed.
    #[error("stream synchronization failed for residency unit {id}: {source}")]
    Synchronization {
        /// Unit identifier.
        id: OffloadUnitId,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// Serialized manager state was poisoned by a prior panic.
    #[error("residency manager state is poisoned")]
    StatePoisoned,
}

/// Point-in-time state for one logical unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UnitResidencyReport {
    id: OffloadUnitId,
    planned_tier: MemoryTier,
    policy: ResidencyPolicy,
    expected_bytes: u64,
    host_resident: bool,
    device_resident: bool,
    host_pins: u64,
    device_pins: u64,
    active_window: bool,
}

impl UnitResidencyReport {
    /// Returns the stable unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }
    /// Returns the plan's initial tier.
    pub const fn planned_tier(&self) -> MemoryTier {
        self.planned_tier
    }
    /// Returns the operational residency policy.
    pub const fn policy(&self) -> ResidencyPolicy {
        self.policy
    }
    /// Returns the validated unit size.
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }
    /// Returns whether evaluated host arrays are resident.
    pub const fn host_resident(&self) -> bool {
        self.host_resident
    }
    /// Returns whether evaluated execution-stream arrays are resident.
    pub const fn device_resident(&self) -> bool {
        self.device_resident
    }
    /// Returns active host leases.
    pub const fn host_pins(&self) -> u64 {
        self.host_pins
    }
    /// Returns active device leases.
    pub const fn device_pins(&self) -> u64 {
        self.device_pins
    }
    /// Returns whether the unit is in the current protected window.
    pub const fn active_window(&self) -> bool {
        self.active_window
    }
}

/// Immutable manager, telemetry, and store diagnostic snapshot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyReport {
    initialized: bool,
    offload: OffloadReport,
    units: Vec<UnitResidencyReport>,
    active_window: Vec<OffloadUnitId>,
    weight_store: WeightStoreDiagnostics,
}

impl ResidencyReport {
    /// Returns whether explicit initialization completed successfully.
    pub const fn initialized(&self) -> bool {
        self.initialized
    }
    /// Returns the immutable offload telemetry snapshot.
    pub const fn offload(&self) -> &OffloadReport {
        &self.offload
    }
    /// Returns unit states in identifier order.
    pub fn units(&self) -> &[UnitResidencyReport] {
        &self.units
    }
    /// Returns the protected execution window in identifier order.
    pub fn active_window(&self) -> &[OffloadUnitId] {
        &self.active_window
    }
    /// Returns storage diagnostics, distinct from logical residency telemetry.
    pub const fn weight_store(&self) -> &WeightStoreDiagnostics {
        &self.weight_store
    }
}

/// Serialized, shareable manager for immutable checkpoint weight residency.
#[derive(Clone)]
pub struct ResidencyManager {
    inner: Arc<ManagerInner>,
}

/// Deterministic controller for a bounded ordered device-layer window.
///
/// The current layer counts toward `depth`. Preparation is synchronous and
/// explicit trimming is performed even when the manager has an unlimited
/// device budget, so stale decoder copies cannot accumulate.
#[derive(Debug, Clone)]
pub struct DeviceLayerWindow {
    units: Vec<OffloadUnitId>,
    depth: usize,
}

/// A named sequential execution stack with an independent device window.
///
/// Models with text, vision, audio, temporal, or depth-transformer stacks can
/// use one group per ordered stack without imposing a checkpoint naming scheme
/// on the residency core.
#[derive(Debug, Clone)]
pub struct ResidentLayerGroup {
    id: String,
    window: DeviceLayerWindow,
}

impl ResidentLayerGroup {
    /// Creates a named group over ordered residency units.
    pub fn new(
        id: impl Into<String>,
        units: impl IntoIterator<Item = OffloadUnitId>,
        depth: usize,
    ) -> Result<Self, ResidencyError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ResidencyError::InvalidGroupId);
        }
        Ok(Self {
            id,
            window: DeviceLayerWindow::new(units, depth)?,
        })
    }

    /// Returns the stable group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns ordered units in this group.
    pub fn units(&self) -> &[OffloadUnitId] {
        self.window.units()
    }

    /// Returns the configured device-unit bound.
    pub const fn depth(&self) -> usize {
        self.window.depth()
    }

    /// Synchronously prepares this group's window without replacing another group's window.
    pub fn prepare(
        &self,
        manager: &ResidencyManager,
        current: usize,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        let desired = self.window.desired(current)?;
        let outcomes =
            manager.prepare_group_window(&self.id, desired, desired, MemoryTier::Device)?;
        self.window.trim_to(manager, desired)?;
        Ok(outcomes)
    }

    /// Trims this group to the desired window.
    pub fn trim_to(
        &self,
        manager: &ResidencyManager,
        desired: &[OffloadUnitId],
    ) -> Result<(), ResidencyError> {
        self.window.trim_to(manager, desired)
    }

    /// Clears only this group's protection and device copies.
    pub fn clear(&self, manager: &ResidencyManager) -> Result<(), ResidencyError> {
        manager.prepare_group_window(&self.id, &[], &[], MemoryTier::Device)?;
        self.window.trim_to(manager, &[])
    }

    /// Returns current logical residency attributed to this group's units.
    pub fn report(
        &self,
        manager: &ResidencyManager,
    ) -> Result<ResidentLayerGroupReport, ResidencyError> {
        let report = manager.report()?;
        let ids = self.units().iter().collect::<BTreeSet<_>>();
        let mut host_bytes = 0u64;
        let mut device_bytes = 0u64;
        let mut device_units = 0usize;
        for unit in report.units().iter().filter(|unit| ids.contains(unit.id())) {
            if unit.host_resident() {
                host_bytes = host_bytes.checked_add(unit.expected_bytes()).ok_or(
                    ResidencyError::ArithmeticOverflow {
                        context: "execution group host bytes",
                    },
                )?;
            }
            if unit.device_resident() {
                device_bytes = device_bytes.checked_add(unit.expected_bytes()).ok_or(
                    ResidencyError::ArithmeticOverflow {
                        context: "execution group device bytes",
                    },
                )?;
                device_units += 1;
            }
        }
        Ok(ResidentLayerGroupReport {
            id: self.id.clone(),
            ordered_units: self.units().len(),
            window_depth: self.depth(),
            host_bytes,
            device_bytes,
            device_units,
        })
    }
}

/// Logical residency attributed to one named execution group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidentLayerGroupReport {
    id: String,
    ordered_units: usize,
    window_depth: usize,
    host_bytes: u64,
    device_bytes: u64,
    device_units: usize,
}

impl ResidentLayerGroupReport {
    /// Returns the group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns the number of ordered units.
    pub const fn ordered_units(&self) -> usize {
        self.ordered_units
    }
    /// Returns the configured maximum device-unit count.
    pub const fn window_depth(&self) -> usize {
        self.window_depth
    }
    /// Returns current host-resident bytes for group units.
    pub const fn host_bytes(&self) -> u64 {
        self.host_bytes
    }
    /// Returns current device-resident bytes for group units.
    pub const fn device_bytes(&self) -> u64 {
        self.device_bytes
    }
    /// Returns current device-resident group units.
    pub const fn device_units(&self) -> usize {
        self.device_units
    }
}

impl DeviceLayerWindow {
    /// Creates a controller for a non-empty ordered unit sequence.
    pub fn new(
        units: impl IntoIterator<Item = OffloadUnitId>,
        depth: usize,
    ) -> Result<Self, ResidencyError> {
        let units = units.into_iter().collect::<Vec<_>>();
        if units.is_empty() {
            return Err(ResidencyError::EmptyLayerWindow);
        }
        if depth == 0 || depth > units.len() {
            return Err(ResidencyError::OversizedLayerWindow {
                depth,
                layer_count: units.len(),
            });
        }
        let unique = units.iter().collect::<BTreeSet<_>>();
        if unique.len() != units.len() {
            return Err(ResidencyError::DuplicateUnitDefinition {
                id: units
                    .iter()
                    .find(|id| units.iter().filter(|candidate| *candidate == *id).count() > 1)
                    .expect("duplicate exists")
                    .clone(),
            });
        }
        Ok(Self { units, depth })
    }

    /// Returns the maximum number of decoder units kept on the device.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns decoder units in execution order.
    pub fn units(&self) -> &[OffloadUnitId] {
        &self.units
    }

    /// Returns the desired window beginning at `current`.
    pub fn desired(&self, current: usize) -> Result<&[OffloadUnitId], ResidencyError> {
        if current >= self.units.len() {
            return Err(ResidencyError::InvalidLayerIndex {
                index: current,
                layer_count: self.units.len(),
            });
        }
        let end = current.saturating_add(self.depth).min(self.units.len());
        Ok(&self.units[current..end])
    }

    /// Synchronously prepares and trims the window beginning at `current`.
    pub fn prepare(
        &self,
        manager: &ResidencyManager,
        current: usize,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        let desired = self.desired(current)?;
        let outcomes = manager.prepare_window(desired, desired, MemoryTier::Device)?;
        self.trim_to(manager, desired)?;
        Ok(outcomes)
    }

    /// Explicitly evicts every managed device copy outside `desired`.
    pub fn trim_to(
        &self,
        manager: &ResidencyManager,
        desired: &[OffloadUnitId],
    ) -> Result<(), ResidencyError> {
        let desired = desired.iter().collect::<BTreeSet<_>>();
        for id in &self.units {
            if !desired.contains(id) {
                manager.evict(id, MemoryTier::Device)?;
            }
        }
        Ok(())
    }

    /// Clears protection and removes every managed device-layer copy.
    pub fn clear(&self, manager: &ResidencyManager) -> Result<(), ResidencyError> {
        manager.prepare_window(&[], &[], MemoryTier::Device)?;
        self.trim_to(manager, &[])
    }
}

impl ResidencyManager {
    /// Validates plan/unit identity, binding sizes, selections, and streams.
    ///
    /// Construction does not create MLX arrays. Call [`Self::initialize`] to
    /// materialize units assigned to host or device by the plan.
    pub fn new<S>(
        store: Arc<S>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError>
    where
        S: WeightStore + Send + Sync + 'static,
    {
        let source_device = source_stream
            .get_device()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source stream inspection",
                source,
            })?;
        if source_device
            .get_type()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source device inspection",
                source,
            })?
            != DeviceType::Cpu
        {
            return Err(ResidencyError::InvalidSourceStream);
        }

        let mut definitions = BTreeMap::new();
        for unit in units {
            let id = unit.id.clone();
            if definitions.insert(id.clone(), unit).is_some() {
                return Err(ResidencyError::DuplicateUnitDefinition { id });
            }
        }
        for spec in plan.units() {
            if !definitions.contains_key(spec.id()) {
                return Err(ResidencyError::MissingUnitDefinition {
                    id: spec.id().clone(),
                });
            }
        }
        if let Some(id) = definitions
            .keys()
            .find(|id| plan.unit(id).is_none())
            .cloned()
        {
            return Err(ResidencyError::UnexpectedUnitDefinition { id });
        }

        let store: Arc<dyn WeightStore + Send + Sync> = store;
        let mut records = BTreeMap::new();
        for spec in plan.units() {
            let definition = definitions.remove(spec.id()).expect("validated above");
            validate_unit_bytes(store.as_ref(), spec, &definition)?;
            records.insert(
                spec.id().clone(),
                UnitRecord {
                    definition,
                    spec: spec.clone(),
                    host: None,
                    device: None,
                },
            );
        }

        let telemetry = OffloadTelemetry::from_plan(&plan);
        Ok(Self {
            inner: Arc::new(ManagerInner {
                store,
                state: Mutex::new(ManagerState {
                    plan,
                    units: records,
                    source_stream,
                    device_stream,
                    active_windows: BTreeMap::new(),
                    group_windows: BTreeMap::new(),
                    telemetry,
                    host_bytes: 0,
                    device_bytes: 0,
                    tick: 0,
                    initialized: false,
                }),
            }),
        })
    }

    /// Materializes all planned host and device units in identifier order.
    ///
    /// Disk units remain array-free. A failure never publishes a partial unit;
    /// units completed earlier remain resident and fully accounted, allowing a
    /// caller to inspect the report and retry initialization.
    pub fn initialize(&self) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if state.initialized {
            return Ok(());
        }
        let assignments = state
            .units
            .values()
            .map(|unit| (unit.spec.id().clone(), unit.spec.tier()))
            .collect::<Vec<_>>();
        for (id, tier) in assignments {
            if tier != MemoryTier::Disk {
                ensure_resident(&mut state, self.inner.store.as_ref(), &id, tier)?;
            }
        }
        state.initialized = true;
        Ok(())
    }

    /// Synchronously prepares one host or device copy and records hit/miss telemetry.
    ///
    /// This provides caller-directed lookahead but does not overlap transfer
    /// with computation.
    pub fn prefetch(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<PrefetchOutcome, ResidencyError> {
        validate_target(tier, "prefetch")?;
        let mut state = self.lock()?;
        require_initialized(&state)?;
        prefetch_locked(&mut state, self.inner.store.as_ref(), id, tier)
    }

    /// Ensures residency and returns an RAII lease protecting the requested copy.
    pub fn acquire(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_with_demand(id, tier, 1)
    }

    /// Ensures residency and records route-weighted demand for eviction policy.
    ///
    /// `demand` may be larger than one when duplicate routed-expert requests
    /// share a single acquisition. Frequency counters saturate on overflow.
    pub fn acquire_with_demand(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        demand: u64,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_many_with_demand(&[(id.clone(), demand)], tier)?
            .pop()
            .ok_or(ResidencyError::StatePoisoned)
    }

    /// Acquires a deterministic expert set with one batched residency transition.
    ///
    /// Missing copies reserve capacity before any materialization starts. All
    /// requested units are protected from eviction, all lazy outputs are
    /// evaluated together, and leases are published only after the batch is
    /// complete.
    pub fn acquire_many_with_demand(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Vec<ResidentUnitLease>, ResidencyError> {
        self.acquire_many_with_mode(requests, tier, false)
    }

    /// Acquires device copies without evaluating their lazy transfer graphs.
    ///
    /// Callers must evaluate a dependent output and then invoke
    /// [`ResidentUnitLease::complete_pending`] on every returned lease. Early
    /// lease drop takes the conservative synchronized cleanup path.
    pub(crate) fn acquire_many_deferred_with_demand(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Vec<ResidentUnitLease>, ResidencyError> {
        self.acquire_many_with_mode(requests, tier, true)
    }

    fn acquire_many_with_mode(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        defer_evaluation: bool,
    ) -> Result<Vec<ResidentUnitLease>, ResidencyError> {
        validate_target(tier, "acquire")?;
        let mut seen = BTreeSet::new();
        if requests.iter().any(|(id, _)| !seen.insert(id.clone())) {
            return Err(ResidencyError::DuplicateBatchUnit);
        }
        let mut state = self.lock()?;
        require_initialized(&state)?;
        for (id, _) in requests {
            if !state.units.contains_key(id) {
                return Err(ResidencyError::UnknownUnit { id: id.clone() });
            }
        }
        let missing = requests
            .iter()
            .filter(|(id, _)| !state.units[id].is_resident(tier))
            .count();
        let started = Instant::now();
        let ids = requests
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let residency = ensure_many_resident(
            &mut state,
            self.inner.store.as_ref(),
            &ids,
            tier,
            defer_evaluation,
        );
        if missing > 0 {
            state.telemetry.record_prefetch_stall(started.elapsed());
        }
        residency?;
        requests
            .iter()
            .map(|(id, demand)| {
                let tick = next_tick(&mut state)?;
                let unit = state
                    .units
                    .get_mut(id)
                    .ok_or_else(|| ResidencyError::UnknownUnit { id: id.clone() })?;
                let copy = unit.copy_mut(tier).ok_or(ResidencyError::StatePoisoned)?;
                copy.pins = copy
                    .pins
                    .checked_add(1)
                    .ok_or(ResidencyError::ArithmeticOverflow {
                        context: "resident lease count",
                    })?;
                copy.last_used = tick;
                copy.frequency = copy.frequency.saturating_add(*demand);
                Ok(ResidentUnitLease {
                    id: id.clone(),
                    tier,
                    arrays: Arc::clone(&copy.arrays),
                    pending: copy.pending.as_ref().map(Arc::clone),
                    manager: Arc::downgrade(&self.inner),
                })
            })
            .collect()
    }

    /// Returns whether a logical copy currently resides in a memory tier.
    pub fn is_resident(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<bool, ResidencyError> {
        validate_target(tier, "is_resident")?;
        let state = self.lock()?;
        Ok(state
            .units
            .get(id)
            .ok_or_else(|| ResidencyError::UnknownUnit { id: id.clone() })?
            .is_resident(tier))
    }

    /// Replaces the protected window and synchronously prepares bounded lookahead.
    ///
    /// `active` units are protected from automatic eviction. At most the first
    /// configured number of distinct `upcoming` units are prefetched, in caller
    /// order. Repeated and overlapping windows are deterministic.
    pub fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        self.prepare_group_window("default", active, upcoming, tier)
    }

    /// Replaces one named group's protected window and prepares bounded lookahead.
    ///
    /// Protection owned by other groups remains active. This permits independent
    /// text, vision, audio, temporal, and depth stack scheduling.
    pub fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        if group.trim().is_empty() {
            return Err(ResidencyError::InvalidGroupId);
        }
        validate_target(tier, "prepare_group_window")?;
        let mut state = self.lock()?;
        require_initialized(&state)?;
        for id in active.iter().chain(upcoming) {
            if !state.units.contains_key(id) {
                return Err(ResidencyError::UnknownUnit { id: id.clone() });
            }
        }
        let key = (group.to_string(), tier);
        if active.is_empty() {
            state.group_windows.remove(&key);
        } else {
            state
                .group_windows
                .insert(key, active.iter().cloned().collect());
        }
        let active_window = state
            .group_windows
            .iter()
            .filter(|((_, window_tier), _)| *window_tier == tier)
            .flat_map(|(_, window)| window.iter().cloned())
            .collect();
        state.active_windows.insert(tier, active_window);
        let depth = state.plan.config().prefetch_depth();
        let mut seen = BTreeSet::new();
        let selected = upcoming
            .iter()
            .filter(|id| seen.insert((*id).clone()))
            .take(depth)
            .cloned()
            .collect::<Vec<_>>();
        selected
            .into_iter()
            .map(|id| {
                prefetch_locked(&mut state, self.inner.store.as_ref(), &id, tier)
                    .map(|outcome| (id, outcome))
            })
            .collect()
    }

    /// Replaces one named protected window without materializing its units.
    ///
    /// This is used by schedulers that submit materialization through a
    /// separate bounded service. Protection owned by other named windows is
    /// preserved.
    pub fn protect_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), ResidencyError> {
        if group.trim().is_empty() {
            return Err(ResidencyError::InvalidGroupId);
        }
        let mut state = self.lock()?;
        require_initialized(&state)?;
        for id in active {
            if !state.units.contains_key(id) {
                return Err(ResidencyError::UnknownUnit { id: id.clone() });
            }
        }
        validate_target(tier, "protect_group_window")?;
        let key = (group.to_string(), tier);
        if active.is_empty() {
            state.group_windows.remove(&key);
        } else {
            state
                .group_windows
                .insert(key, active.iter().cloned().collect());
        }
        let active_window = state
            .group_windows
            .iter()
            .filter(|((_, window_tier), _)| *window_tier == tier)
            .flat_map(|(_, window)| window.iter().cloned())
            .collect();
        state.active_windows.insert(tier, active_window);
        Ok(())
    }

    /// Explicitly evicts one host or device copy.
    ///
    /// Evicting an absent copy is an idempotent success returning `false`.
    pub fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, ResidencyError> {
        validate_target(tier, "evict")?;
        let mut state = self.lock()?;
        let unit = state
            .units
            .get(id)
            .ok_or_else(|| ResidencyError::UnknownUnit { id: id.clone() })?;
        let Some(copy) = unit.copy(tier) else {
            return Ok(false);
        };
        if unit.spec.policy() == ResidencyPolicy::Pinned {
            return Err(ResidencyError::PinnedEviction {
                id: id.clone(),
                tier,
            });
        }
        if copy.pins != 0 {
            return Err(ResidencyError::InUseEviction {
                id: id.clone(),
                tier,
                pin_count: copy.pins,
            });
        }
        remove_copy(&mut state, id, tier)?;
        Ok(true)
    }

    /// Samples optional MLX allocator and process metrics on explicit request.
    pub fn sample_memory(
        &self,
        include_mlx: bool,
        include_process: bool,
    ) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if include_mlx {
            state
                .telemetry
                .sample_mlx_memory()
                .map_err(|source| ResidencyError::Mlx {
                    id: internal_id(),
                    operation: "allocator memory sampling",
                    source,
                })?;
        }
        if include_process {
            state.telemetry.sample_process_metrics();
        }
        Ok(())
    }

    /// Returns an immutable point-in-time residency and storage report.
    pub fn report(&self) -> Result<ResidencyReport, ResidencyError> {
        let (initialized, offload, units, active_window) = self.telemetry_snapshot()?;
        Ok(ResidencyReport {
            initialized,
            offload,
            units,
            active_window,
            weight_store: self.inner.store.diagnostics()?,
        })
    }

    pub(crate) fn telemetry_snapshot(
        &self,
    ) -> Result<
        (
            bool,
            OffloadReport,
            Vec<UnitResidencyReport>,
            Vec<OffloadUnitId>,
        ),
        ResidencyError,
    > {
        let state = self.lock()?;
        let active = state
            .active_windows
            .values()
            .flat_map(|window| window.iter().cloned())
            .collect::<BTreeSet<_>>();
        let units = state
            .units
            .values()
            .map(|unit| UnitResidencyReport {
                id: unit.spec.id().clone(),
                planned_tier: unit.spec.tier(),
                policy: unit.spec.policy(),
                expected_bytes: unit.spec.bytes(),
                host_resident: unit.host.is_some(),
                device_resident: unit.device.is_some(),
                host_pins: unit.host.as_ref().map_or(0, |copy| copy.pins),
                device_pins: unit.device.as_ref().map_or(0, |copy| copy.pins),
                active_window: active.contains(unit.spec.id()),
            })
            .collect();
        Ok((
            state.initialized,
            state.telemetry.snapshot(),
            units,
            active.into_iter().collect(),
        ))
    }

    fn lock(&self) -> Result<MutexGuard<'_, ManagerState>, ResidencyError> {
        self.inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)
    }
}

struct ManagerInner {
    store: Arc<dyn WeightStore + Send + Sync>,
    state: Mutex<ManagerState>,
}

struct ManagerState {
    plan: OffloadPlan,
    units: BTreeMap<OffloadUnitId, UnitRecord>,
    source_stream: Stream,
    device_stream: Stream,
    active_windows: BTreeMap<MemoryTier, BTreeSet<OffloadUnitId>>,
    group_windows: BTreeMap<(String, MemoryTier), BTreeSet<OffloadUnitId>>,
    telemetry: OffloadTelemetry,
    host_bytes: u64,
    device_bytes: u64,
    tick: u64,
    initialized: bool,
}

// SAFETY: every access to the MLX stream handles and resident arrays in this
// state is serialized by `ManagerInner::state`. No stream reference escapes
// the lock, and MLX operations use safemlx's runtime guard internally.
unsafe impl Send for ManagerState {}

struct UnitRecord {
    definition: OffloadUnit,
    spec: OffloadUnitSpec,
    host: Option<ResidentCopy>,
    device: Option<ResidentCopy>,
}

impl UnitRecord {
    fn copy(&self, tier: MemoryTier) -> Option<&ResidentCopy> {
        match tier {
            MemoryTier::Host => self.host.as_ref(),
            MemoryTier::Device => self.device.as_ref(),
            MemoryTier::Disk => None,
        }
    }

    fn copy_mut(&mut self, tier: MemoryTier) -> Option<&mut ResidentCopy> {
        match tier {
            MemoryTier::Host => self.host.as_mut(),
            MemoryTier::Device => self.device.as_mut(),
            MemoryTier::Disk => None,
        }
    }

    fn is_resident(&self, tier: MemoryTier) -> bool {
        self.copy(tier).is_some()
    }
}

struct ResidentCopy {
    arrays: Arc<ResidentArrays>,
    pending: Option<Arc<PendingResidentSources>>,
    bytes: u64,
    pins: u64,
    last_used: u64,
    frequency: u64,
}

struct ResidentArrays {
    arrays: BTreeMap<String, Array>,
}

struct PendingResidentDependencies {
    sources: Vec<PendingWeightMaterialization>,
    retained_arrays: Vec<Array>,
    retained_host: Option<Arc<ResidentArrays>>,
}

struct PendingResidentSources {
    dependencies: Mutex<Option<PendingResidentDependencies>>,
}

// SAFETY: the mutex gives exactly one thread ownership of the retained MLX
// handles and mmap leases for completion or cleanup. MLX array handles are
// Send + Sync, and stream operations use safemlx's runtime guard internally.
unsafe impl Send for PendingResidentSources {}
unsafe impl Sync for PendingResidentSources {}

impl PendingResidentSources {
    fn complete_after_evaluation(&self) -> Result<(), ResidencyError> {
        let mut dependencies = self
            .dependencies
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        if let Some(dependencies) = dependencies.take() {
            for source in dependencies.sources {
                source.complete();
            }
            drop(dependencies.retained_arrays);
            drop(dependencies.retained_host);
        }
        Ok(())
    }

    fn abort(&self) {
        if let Ok(mut dependencies) = self.dependencies.lock() {
            drop(dependencies.take());
        }
    }

    fn is_pending(&self) -> bool {
        self.dependencies
            .lock()
            .map_or(true, |dependencies| dependencies.is_some())
    }
}

fn validate_unit_bytes(
    store: &dyn WeightStore,
    spec: &OffloadUnitSpec,
    unit: &OffloadUnit,
) -> Result<(), ResidencyError> {
    let mut total = 0u64;
    for binding in &unit.bindings {
        total = total.checked_add(binding.expected_bytes).ok_or(
            ResidencyError::ArithmeticOverflow {
                context: "unit binding byte total",
            },
        )?;
        let actual = match &binding.recipe {
            Some(recipe) => recipe
                .infer(store)
                .map_err(|source| ResidencyError::Recipe {
                    binding: binding.name.clone(),
                    source,
                })?
                .byte_len(),
            None => {
                let lease = store.acquire(&binding.checkpoint_key, binding.selection.clone())?;
                u64::try_from(lease.selected_byte_len()).map_err(|_| {
                    ResidencyError::ArithmeticOverflow {
                        context: "selected binding byte conversion",
                    }
                })?
            }
        };
        if actual != binding.expected_bytes {
            return Err(ResidencyError::BindingByteMismatch {
                id: unit.id.clone(),
                binding: binding.name.clone(),
                expected_bytes: binding.expected_bytes,
                actual_bytes: actual,
            });
        }
    }
    if total != spec.bytes() {
        return Err(ResidencyError::UnitByteMismatch {
            id: unit.id.clone(),
            planned_bytes: spec.bytes(),
            actual_bytes: total,
        });
    }
    Ok(())
}

fn require_initialized(state: &ManagerState) -> Result<(), ResidencyError> {
    if state.initialized {
        Ok(())
    } else {
        Err(ResidencyError::NotInitialized)
    }
}

fn validate_target(tier: MemoryTier, operation: &'static str) -> Result<(), ResidencyError> {
    if tier == MemoryTier::Disk {
        Err(ResidencyError::InvalidTargetTier { operation })
    } else {
        Ok(())
    }
}

fn internal_id() -> OffloadUnitId {
    OffloadUnitId::new("residency-manager").expect("static identifier is valid")
}

fn prefetch_locked(
    state: &mut ManagerState,
    store: &dyn WeightStore,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<PrefetchOutcome, ResidencyError> {
    let hit = state
        .units
        .get(id)
        .ok_or_else(|| ResidencyError::UnknownUnit { id: id.clone() })?
        .is_resident(tier);
    let outcome = if hit {
        PrefetchOutcome::Hit
    } else {
        PrefetchOutcome::Miss
    };
    state.telemetry.record_tier_prefetch(tier, outcome);
    ensure_resident(state, store, id, tier)?;
    Ok(outcome)
}

fn ensure_resident(
    state: &mut ManagerState,
    store: &dyn WeightStore,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<bool, ResidencyError> {
    ensure_many_resident(state, store, std::slice::from_ref(id), tier, false)
        .map(|created| created[0])
}

fn ensure_many_resident(
    state: &mut ManagerState,
    store: &dyn WeightStore,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    defer_evaluation: bool,
) -> Result<Vec<bool>, ResidencyError> {
    validate_target(tier, "residency transition")?;
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut unique = BTreeSet::new();
    for id in ids {
        if !unique.insert(id.clone()) {
            return Err(ResidencyError::DuplicateBatchUnit);
        }
        if !state.units.contains_key(id) {
            return Err(ResidencyError::UnknownUnit { id: id.clone() });
        }
    }
    let created = ids
        .iter()
        .map(|id| !state.units[id].is_resident(tier))
        .collect::<Vec<_>>();
    if created.iter().all(|value| !value) {
        complete_existing_pending(state, ids, tier, defer_evaluation, false)?;
        for id in ids {
            let tick = next_tick(state)?;
            state
                .units
                .get_mut(id)
                .and_then(|unit| unit.copy_mut(tier))
                .ok_or(ResidencyError::StatePoisoned)?
                .last_used = tick;
        }
        return Ok(created);
    }

    let temporary_protection = ids
        .iter()
        .filter(|id| {
            state
                .active_windows
                .entry(tier)
                .or_default()
                .insert((*id).clone())
        })
        .cloned()
        .collect::<Vec<_>>();
    let started = Instant::now();
    let mut reserved = 0u64;
    let result = (|| {
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                continue;
            }
            let required = state.units[id].spec.bytes();
            reserve_capacity(state, id, tier, required)?;
            let total = tier_bytes(state, tier).checked_add(required).ok_or(
                ResidencyError::ArithmeticOverflow {
                    context: "resident byte reservation",
                },
            )?;
            let next_reserved =
                reserved
                    .checked_add(required)
                    .ok_or(ResidencyError::ArithmeticOverflow {
                        context: "resident batch reservation",
                    })?;
            set_tier_bytes(state, tier, total);
            reserved = next_reserved;
        }

        let mut prepared = Vec::new();
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                continue;
            }
            let bindings = state.units[id].definition.bindings.clone();
            let item = loop {
                let item = match tier {
                    MemoryTier::Host => prepare_from_disk(
                        store,
                        &bindings,
                        &state.source_stream,
                        &state.source_stream,
                        TransferDirection::DiskToHost,
                    ),
                    MemoryTier::Device => {
                        if let Some(host) = state.units[id]
                            .host
                            .as_ref()
                            .map(|copy| Arc::clone(&copy.arrays))
                        {
                            prepare_copy_to_device(id, host, &state.device_stream)
                        } else {
                            prepare_from_disk(
                                store,
                                &bindings,
                                &state.source_stream,
                                &state.device_stream,
                                TransferDirection::DiskToDevice,
                            )
                        }
                    }
                    MemoryTier::Disk => unreachable!("validated above"),
                };
                match item {
                    Ok(item) => break item,
                    Err(error)
                        if is_mapping_capacity_error(&error)
                            && prepared.iter().any(
                                |(_, item): &(OffloadUnitId, PreparedResidentArrays)| {
                                    !item.pending_sources.is_empty()
                                },
                            ) =>
                    {
                        // Earlier units in this batch can pin the only mapped
                        // shard while a later cross-shard expert is prepared.
                        // Their output arrays are complete evaluation roots, so
                        // detach those mappings and retry the current unit.
                        eval(prepared.iter().flat_map(|(_, item)| item.arrays.values())).map_err(
                            |source| ResidencyError::Mlx {
                                id: internal_id(),
                                operation: "mapping-capacity batch evaluation",
                                source,
                            },
                        )?;
                        for (_, item) in &mut prepared {
                            for source in item.pending_sources.drain(..) {
                                source.complete();
                            }
                            item.retained_arrays.clear();
                            item.retained_host = None;
                        }
                    }
                    Err(error) => return Err(error),
                }
            };
            prepared.push((id.clone(), item));
        }

        if !defer_evaluation {
            let existing = ids
                .iter()
                .zip(&created)
                .filter(|(_, is_missing)| !**is_missing)
                .filter_map(|(id, _)| state.units[id].copy(tier))
                .filter(|copy| {
                    copy.pending
                        .as_ref()
                        .is_some_and(|pending| pending.is_pending())
                });
            eval(
                prepared
                    .iter()
                    .flat_map(|(_, item)| item.arrays.values())
                    .chain(existing.flat_map(|copy| copy.arrays.arrays.values())),
            )
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "batched residency evaluation",
                source,
            })?;
            complete_existing_pending(state, ids, tier, false, true)?;
        }

        for (id, item) in &prepared {
            let actual = arrays_nbytes(&item.arrays)?;
            let required = state.units[id].spec.bytes();
            if actual != required {
                return Err(ResidencyError::UnitByteMismatch {
                    id: id.clone(),
                    planned_bytes: required,
                    actual_bytes: actual,
                });
            }
        }

        for (id, mut item) in prepared {
            let pending = if defer_evaluation {
                Some(Arc::new(PendingResidentSources {
                    dependencies: Mutex::new(Some(PendingResidentDependencies {
                        sources: std::mem::take(&mut item.pending_sources),
                        retained_arrays: std::mem::take(&mut item.retained_arrays),
                        retained_host: item.retained_host.take(),
                    })),
                }))
            } else {
                for source in item.pending_sources.drain(..) {
                    source.complete();
                }
                None
            };
            let actual = arrays_nbytes(&item.arrays)?;
            let tick = next_tick(state)?;
            let copy = ResidentCopy {
                arrays: Arc::new(ResidentArrays {
                    arrays: item.arrays,
                }),
                pending,
                bytes: actual,
                pins: 0,
                last_used: tick,
                frequency: 0,
            };
            let unit = state
                .units
                .get_mut(&id)
                .ok_or_else(|| ResidencyError::UnknownUnit { id: id.clone() })?;
            match tier {
                MemoryTier::Host => unit.host = Some(copy),
                MemoryTier::Device => unit.device = Some(copy),
                MemoryTier::Disk => unreachable!("validated above"),
            }
            state
                .telemetry
                .record_transfer(item.direction, actual, started.elapsed());
        }
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                let tick = next_tick(state)?;
                state
                    .units
                    .get_mut(id)
                    .and_then(|unit| unit.copy_mut(tier))
                    .ok_or(ResidencyError::StatePoisoned)?
                    .last_used = tick;
            }
        }
        update_resident_telemetry(state, tier);
        Ok(created.clone())
    })();

    if result.is_err() && reserved > 0 {
        let current = tier_bytes(state, tier);
        set_tier_bytes(state, tier, current.saturating_sub(reserved));
        update_resident_telemetry(state, tier);
    }
    for id in temporary_protection {
        if let Some(window) = state.active_windows.get_mut(&tier) {
            window.remove(&id);
        }
    }
    result
}

fn complete_existing_pending(
    state: &mut ManagerState,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    defer_evaluation: bool,
    already_evaluated: bool,
) -> Result<(), ResidencyError> {
    if defer_evaluation {
        return Ok(());
    }
    let pending = ids
        .iter()
        .filter_map(|id| state.units[id].copy(tier))
        .filter_map(|copy| copy.pending.as_ref().map(Arc::clone))
        .filter(|pending| pending.is_pending())
        .collect::<Vec<_>>();
    if pending.is_empty() {
        return Ok(());
    }
    if !already_evaluated {
        eval(ids.iter().flat_map(|id| {
            state.units[id]
                .copy(tier)
                .into_iter()
                .filter(|copy| {
                    copy.pending
                        .as_ref()
                        .is_some_and(|pending| pending.is_pending())
                })
                .flat_map(|copy| copy.arrays.arrays.values())
        }))
        .map_err(|source| ResidencyError::Mlx {
            id: internal_id(),
            operation: "pending residency evaluation",
            source,
        })?;
    }
    for completion in pending {
        completion.complete_after_evaluation()?;
    }
    for id in ids {
        if let Some(copy) = state.units.get_mut(id).and_then(|unit| unit.copy_mut(tier)) {
            if copy
                .pending
                .as_ref()
                .is_some_and(|pending| !pending.is_pending())
            {
                copy.pending = None;
            }
        }
    }
    Ok(())
}

struct PreparedResidentArrays {
    arrays: BTreeMap<String, Array>,
    pending_sources: Vec<PendingWeightMaterialization>,
    retained_arrays: Vec<Array>,
    retained_host: Option<Arc<ResidentArrays>>,
    direction: TransferDirection,
}

fn prepare_from_disk(
    store: &dyn WeightStore,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
    direction: TransferDirection,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = BTreeMap::new();
    let mut pending_sources = Vec::new();
    let mut retained_arrays = Vec::new();
    for binding in bindings {
        let mut retried_after_capacity = false;
        loop {
            let prepared = (|| match &binding.recipe {
                Some(recipe) => {
                    let pending = recipe
                        .prepare_materialization(store, source_stream)
                        .map_err(|source| ResidencyError::Recipe {
                            binding: binding.name.clone(),
                            source,
                        })?;
                    let (host, sources) = pending.into_parts();
                    if execution_stream == source_stream {
                        Ok((host, sources, None))
                    } else {
                        let output = host.copy(execution_stream).map_err(|source| {
                            ResidencyError::Recipe {
                                binding: binding.name.clone(),
                                source: WeightRecipeError::Mlx(source),
                            }
                        })?;
                        Ok((output, sources, Some(host)))
                    }
                }
                None => {
                    let lease =
                        store.acquire(&binding.checkpoint_key, binding.selection.clone())?;
                    let pending = lease.prepare_materialization(source_stream, execution_stream)?;
                    let output = pending.output().clone();
                    Ok((output, vec![pending], None))
                }
            })();
            match prepared {
                Ok((output, sources, retained)) => {
                    pending_sources.extend(sources);
                    retained_arrays.extend(retained);
                    arrays.insert(binding.name.clone(), output);
                    break;
                }
                Err(error)
                    if !retried_after_capacity
                        && !pending_sources.is_empty()
                        && is_mapping_capacity_error(&error) =>
                {
                    eval(arrays.values()).map_err(|source| ResidencyError::Mlx {
                        id: internal_id(),
                        operation: "mapping-capacity residency evaluation",
                        source,
                    })?;
                    for source in pending_sources.drain(..) {
                        source.complete();
                    }
                    retained_arrays.clear();
                    retried_after_capacity = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources,
        retained_arrays,
        retained_host: None,
        direction,
    })
}

fn is_mapping_capacity_error(error: &ResidencyError) -> bool {
    matches!(
        error,
        ResidencyError::WeightStore(WeightStoreError::CapacityExhausted { .. })
            | ResidencyError::Recipe {
                source: WeightRecipeError::WeightStore(WeightStoreError::CapacityExhausted { .. }),
                ..
            }
    )
}

fn prepare_copy_to_device(
    id: &OffloadUnitId,
    host: Arc<ResidentArrays>,
    device_stream: &Stream,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let arrays = host
        .arrays
        .iter()
        .map(|(name, array)| {
            array
                .copy(device_stream)
                .map(|copy| (name.clone(), copy))
                .map_err(|source| ResidencyError::Mlx {
                    id: id.clone(),
                    operation: "host-to-device copy",
                    source,
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources: Vec::new(),
        retained_arrays: Vec::new(),
        retained_host: Some(host),
        direction: TransferDirection::HostToDevice,
    })
}

fn arrays_nbytes(arrays: &BTreeMap<String, Array>) -> Result<u64, ResidencyError> {
    arrays.values().try_fold(0u64, |total, array| {
        let bytes =
            u64::try_from(array.nbytes()).map_err(|_| ResidencyError::ArithmeticOverflow {
                context: "array byte conversion",
            })?;
        total
            .checked_add(bytes)
            .ok_or(ResidencyError::ArithmeticOverflow {
                context: "resident array byte total",
            })
    })
}

fn reserve_capacity(
    state: &mut ManagerState,
    requested: &OffloadUnitId,
    tier: MemoryTier,
    required: u64,
) -> Result<(), ResidencyError> {
    let budget = match tier {
        MemoryTier::Host => state.plan.config().host_budget_bytes(),
        MemoryTier::Device => state.plan.config().device_budget_bytes(),
        MemoryTier::Disk => None,
    };
    let Some(budget) = budget else {
        return Ok(());
    };
    loop {
        let resident = tier_bytes(state, tier);
        let needed = resident
            .checked_add(required)
            .ok_or(ResidencyError::ArithmeticOverflow {
                context: "budget reservation",
            })?;
        if needed <= budget {
            return Ok(());
        }
        let victim = eviction_candidate(state, tier);
        if let Some(victim) = victim {
            remove_copy(state, &victim, tier)?;
            continue;
        }
        return Err(ResidencyError::BudgetExhausted {
            requested: requested.clone(),
            tier,
            required_bytes: required,
            budget_bytes: budget,
            resident_bytes: resident,
            blocking_units: blockers(state, tier),
        });
    }
}

fn eviction_candidate(state: &ManagerState, tier: MemoryTier) -> Option<OffloadUnitId> {
    state
        .units
        .values()
        .filter_map(|unit| {
            let copy = unit.copy(tier)?;
            if unit.spec.policy() == ResidencyPolicy::Pinned
                || copy.pins != 0
                || state
                    .active_windows
                    .get(&tier)
                    .is_some_and(|window| window.contains(unit.spec.id()))
            {
                return None;
            }
            let priority = match unit.spec.policy() {
                ResidencyPolicy::Windowed => 0u8,
                ResidencyPolicy::Cacheable => 1u8,
                ResidencyPolicy::Pinned => return None,
            };
            let frequency = match state.plan.config().eviction_policy() {
                CacheEvictionPolicy::LeastRecentlyUsed => 0,
                CacheEvictionPolicy::LeastFrequentlyUsed => copy.frequency,
            };
            Some((priority, frequency, copy.last_used, unit.spec.id().clone()))
        })
        .min()
        .map(|(_, _, _, id)| id)
}

fn blockers(state: &ManagerState, tier: MemoryTier) -> Vec<ResidencyBlocker> {
    state
        .units
        .values()
        .filter_map(|unit| {
            let copy = unit.copy(tier)?;
            let pinned = unit.spec.policy() == ResidencyPolicy::Pinned;
            let active_window = state
                .active_windows
                .get(&tier)
                .is_some_and(|window| window.contains(unit.spec.id()));
            (pinned || copy.pins != 0 || active_window).then(|| ResidencyBlocker {
                id: unit.spec.id().clone(),
                pinned,
                in_use: copy.pins,
                active_window,
            })
        })
        .collect()
}

fn remove_copy(
    state: &mut ManagerState,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<(), ResidencyError> {
    let bytes = state
        .units
        .get(id)
        .and_then(|unit| unit.copy(tier))
        .ok_or(ResidencyError::StatePoisoned)?
        .bytes;
    let total = tier_bytes(state, tier)
        .checked_sub(bytes)
        .ok_or(ResidencyError::StatePoisoned)?;
    let copy = match tier {
        MemoryTier::Host => state.units.get_mut(id).and_then(|unit| unit.host.take()),
        MemoryTier::Device => state.units.get_mut(id).and_then(|unit| unit.device.take()),
        MemoryTier::Disk => None,
    }
    .ok_or(ResidencyError::StatePoisoned)?;
    debug_assert_eq!(copy.bytes, bytes);
    set_tier_bytes(state, tier, total);
    update_resident_telemetry(state, tier);
    state.telemetry.record_tier_eviction(tier, copy.bytes);
    Ok(())
}

fn update_resident_telemetry(state: &mut ManagerState, tier: MemoryTier) {
    let bytes = tier_bytes(state, tier);
    let units = state
        .units
        .values()
        .filter(|unit| unit.is_resident(tier))
        .count();
    state.telemetry.set_resident_bytes(tier, bytes);
    state.telemetry.set_resident_units(tier, units);
}

fn tier_bytes(state: &ManagerState, tier: MemoryTier) -> u64 {
    match tier {
        MemoryTier::Host => state.host_bytes,
        MemoryTier::Device => state.device_bytes,
        MemoryTier::Disk => 0,
    }
}

fn set_tier_bytes(state: &mut ManagerState, tier: MemoryTier, bytes: u64) {
    match tier {
        MemoryTier::Host => state.host_bytes = bytes,
        MemoryTier::Device => state.device_bytes = bytes,
        MemoryTier::Disk => {}
    }
}

fn next_tick(state: &mut ManagerState) -> Result<u64, ResidencyError> {
    if state.tick == u64::MAX {
        for unit in state.units.values_mut() {
            if let Some(copy) = unit.host.as_mut() {
                copy.last_used /= 2;
            }
            if let Some(copy) = unit.device.as_mut() {
                copy.last_used /= 2;
            }
        }
        state.tick /= 2;
    }
    state.tick += 1;
    Ok(state.tick)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use safemlx::{Device, DeviceType};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    use super::*;
    use crate::{
        offload::{OffloadConfig, OffloadUnitSpec},
        weight_store::SafetensorsWeightStore,
    };

    fn cpu_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn write_fixture(path: &std::path::Path) {
        let a = [1i32, 2]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let b = [3i32, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let c = [5i32, 6]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let matrix = [10i32, 11, 12, 13, 14, 15]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                ("a", TensorView::new(Dtype::I32, vec![2], &a).unwrap()),
                ("b", TensorView::new(Dtype::I32, vec![2], &b).unwrap()),
                ("c", TensorView::new(Dtype::I32, vec![2], &c).unwrap()),
                (
                    "matrix",
                    TensorView::new(Dtype::I32, vec![3, 2], &matrix).unwrap(),
                ),
            ],
            None,
            path,
        )
        .unwrap();
    }

    fn fixture_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(&dir.path().join("model.safetensors"));
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        (dir, store)
    }

    fn cross_shard_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        for (file, key, values) in [
            ("model-00001-of-00002.safetensors", "left", [1i32, 2]),
            ("model-00002-of-00002.safetensors", "right", [3i32, 4]),
        ] {
            let bytes = values
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            serialize_to_file(
                [(key, TensorView::new(Dtype::I32, vec![2], &bytes).unwrap())],
                None,
                &dir.path().join(file),
            )
            .unwrap();
        }
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "left": "model-00001-of-00002.safetensors",
                    "right": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let store =
            Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1).unwrap());
        (dir, store)
    }

    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }

    fn binding(name: &str, key: &str, selection: TensorSelection, bytes: u64) -> WeightBinding {
        WeightBinding::new(name, key, selection, bytes).unwrap()
    }

    fn unit(name: &str, bindings: impl IntoIterator<Item = WeightBinding>) -> OffloadUnit {
        OffloadUnit::new(id(name), bindings).unwrap()
    }

    fn spec(name: &str, bytes: u64, policy: ResidencyPolicy, tier: MemoryTier) -> OffloadUnitSpec {
        OffloadUnitSpec::new(id(name), bytes, policy, tier).unwrap()
    }

    fn manager(
        store: Arc<SafetensorsWeightStore>,
        config: OffloadConfig,
        specs: impl IntoIterator<Item = OffloadUnitSpec>,
        units: impl IntoIterator<Item = OffloadUnit>,
    ) -> ResidencyManager {
        ResidencyManager::new(
            store,
            OffloadPlan::new(config, specs).unwrap(),
            units,
            cpu_stream(),
            cpu_stream(),
        )
        .unwrap()
    }

    fn single(name: &str, key: &str) -> OffloadUnit {
        unit(name, [binding("weight", key, TensorSelection::Full, 8)])
    }

    #[test]
    fn named_execution_groups_keep_independent_windows_and_clear_in_isolation() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, None, 1).unwrap(),
            [
                spec("text.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("text.1", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("vision.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            ],
            [
                single("text.0", "a"),
                single("text.1", "b"),
                single("vision.0", "c"),
            ],
        );
        manager.initialize().unwrap();
        let text = ResidentLayerGroup::new("text", [id("text.0"), id("text.1")], 1).unwrap();
        let vision = ResidentLayerGroup::new("vision", [id("vision.0")], 1).unwrap();

        text.prepare(&manager, 0).unwrap();
        vision.prepare(&manager, 0).unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.active_window(), &[id("text.0"), id("vision.0")]);
        assert!(state(&report, "text.0").device_resident());
        assert!(state(&report, "vision.0").device_resident());

        text.clear(&manager).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "text.0").device_resident());
        assert!(state(&report, "vision.0").device_resident());
        assert_eq!(report.active_window(), &[id("vision.0")]);
        let vision_report = vision.report(&manager).unwrap();
        assert_eq!(vision_report.device_units(), 1);
        assert_eq!(vision_report.device_bytes(), 8);
    }

    fn state<'a>(report: &'a ResidencyReport, name: &str) -> &'a UnitResidencyReport {
        report
            .units()
            .iter()
            .find(|unit| unit.id() == &id(name))
            .unwrap()
    }

    #[test]
    fn failed_batch_reservation_rolls_back_and_cache_remains_usable() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b")],
        );
        manager.initialize().unwrap();
        assert!(matches!(
            manager.acquire_many_with_demand(&[(id("a"), 1), (id("b"), 1)], MemoryTier::Device),
            Err(ResidencyError::BudgetExhausted { .. })
        ));
        let report = manager.report().unwrap();
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 0);
        assert!(!state(&report, "a").device_resident());
        assert!(!state(&report, "b").device_resident());

        let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(lease.array("weight").unwrap().shape(), &[2]);
    }

    #[test]
    fn batched_units_detach_prior_shards_at_mapping_capacity() {
        let (_dir, store) = cross_shard_store();
        let manager = manager(
            Arc::clone(&store),
            OffloadConfig::new(None, None, 1).unwrap(),
            [
                spec("left", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("right", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("left", "left"), single("right", "right")],
        );
        manager.initialize().unwrap();

        let leases = manager
            .acquire_many_with_demand(&[(id("left"), 1), (id("right"), 1)], MemoryTier::Host)
            .unwrap();

        assert_eq!(
            leases[0]
                .array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            leases[1]
                .array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[3, 4]
        );
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert!(diagnostics.evictions >= 1);
    }

    #[test]
    fn deferred_acquisition_stays_pending_until_dependent_output_completes() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let mut leases = manager
            .acquire_many_deferred_with_demand(&[(id("a"), 2)], MemoryTier::Device)
            .unwrap();
        let lease = leases.pop().unwrap();
        {
            let state = manager.lock().unwrap();
            let copy = state.units[&id("a")].device.as_ref().unwrap();
            assert_eq!(copy.pins, 1);
            assert!(copy.pending.as_ref().unwrap().is_pending());
        }

        eval([lease.array("weight").unwrap()]).unwrap();
        lease.complete_pending().unwrap();
        {
            let state = manager.lock().unwrap();
            let copy = state.units[&id("a")].device.as_ref().unwrap();
            assert!(copy.pending.is_none());
        }
        drop(lease);
        assert_eq!(state(&manager.report().unwrap(), "a").device_pins(), 0);
    }

    #[test]
    fn synchronous_acquisition_completes_an_existing_pending_copy() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let deferred = manager
            .acquire_many_deferred_with_demand(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        let synchronous = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(
            synchronous
                .array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2]
        );
        assert!(manager.lock().unwrap().units[&id("a")]
            .device
            .as_ref()
            .unwrap()
            .pending
            .is_none());
        drop(deferred);
    }

    #[test]
    fn validates_unit_identity_bindings_sizes_and_targets() {
        let (_dir, store) = fixture_store();
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        )
        .unwrap();
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::MissingUnitDefinition { .. })
        ));
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [single("a", "a"), single("a", "a")],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::DuplicateUnitDefinition { .. })
        ));
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [single("a", "a"), single("b", "b")],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::UnexpectedUnitDefinition { .. })
        ));
        assert!(matches!(
            OffloadUnit::new(id("empty"), []),
            Err(ResidencyError::EmptyUnit { .. })
        ));
        let duplicate = binding("same", "a", TensorSelection::Full, 8);
        assert!(matches!(
            OffloadUnit::new(id("duplicate"), [duplicate.clone(), duplicate]),
            Err(ResidencyError::DuplicateBindingName { .. })
        ));
        let wrong = unit("a", [binding("weight", "a", TensorSelection::Full, 4)]);
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [wrong],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::BindingByteMismatch { .. })
        ));

        let valid =
            ResidencyManager::new(store, plan, [single("a", "a")], cpu_stream(), cpu_stream())
                .unwrap();
        assert!(matches!(
            valid.prefetch(&id("a"), MemoryTier::Disk),
            Err(ResidencyError::InvalidTargetTier { .. })
        ));
    }

    #[test]
    fn detects_unit_total_overflow_before_checkpoint_access() {
        let (_dir, store) = fixture_store();
        let overflowing = unit(
            "a",
            [
                binding("a", "a", TensorSelection::Full, 8),
                binding("z", "missing", TensorSelection::Full, u64::MAX),
            ],
        );
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        )
        .unwrap();
        assert!(matches!(
            ResidencyManager::new(store, plan, [overflowing], cpu_stream(), cpu_stream()),
            Err(ResidencyError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn initialization_honors_planned_tiers_and_pinning() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [
                spec("disk", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("host", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
                spec("device", 8, ResidencyPolicy::Cacheable, MemoryTier::Device),
            ],
            [
                single("disk", "a"),
                single("host", "b"),
                single("device", "c"),
            ],
        );
        assert!(matches!(
            manager.acquire(&id("disk"), MemoryTier::Host),
            Err(ResidencyError::NotInitialized)
        ));
        manager.initialize().unwrap();
        let report = manager.report().unwrap();
        assert!(report.initialized());
        assert!(!state(&report, "disk").host_resident());
        assert!(state(&report, "host").host_resident());
        assert!(state(&report, "device").device_resident());
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Host), 8);
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
        assert!(matches!(
            manager.evict(&id("host"), MemoryTier::Host),
            Err(ResidencyError::PinnedEviction { .. })
        ));
    }

    #[test]
    fn partial_initialization_failure_remains_consistent_and_inspectable() {
        let dir = tempfile::tempdir().unwrap();
        let good = [7u8, 8];
        let bad = [9u8, 10];
        serialize_to_file(
            [
                ("good", TensorView::new(Dtype::U8, vec![2], &good).unwrap()),
                (
                    "unsupported",
                    TensorView::new(Dtype::F8_E5M2, vec![2], &bad).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(6), 1).unwrap(),
            [
                spec("a-good", 2, ResidencyPolicy::Cacheable, MemoryTier::Host),
                spec("z-bad", 4, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
            [
                unit(
                    "a-good",
                    [binding("weight", "good", TensorSelection::Full, 2)],
                ),
                unit(
                    "z-bad",
                    [
                        binding("a-good-copy", "good", TensorSelection::Full, 2),
                        binding("z-unsupported", "unsupported", TensorSelection::Full, 2),
                    ],
                ),
            ],
        );
        assert!(matches!(
            manager.initialize(),
            Err(ResidencyError::WeightStore(
                WeightStoreError::UnsupportedStoredDtype { .. }
            ))
        ));
        let report = manager.report().unwrap();
        assert!(!report.initialized());
        assert!(state(&report, "a-good").host_resident());
        assert!(!state(&report, "z-bad").host_resident());
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Host), 2);
    }

    #[test]
    fn materializes_promotes_and_publishes_multi_tensor_units_atomically() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(16), Some(16), 1).unwrap(),
            [spec(
                "quantized",
                16,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )],
            [unit(
                "quantized",
                [
                    binding("scales", "b", TensorSelection::Full, 8),
                    binding("weight", "a", TensorSelection::Full, 8),
                ],
            )],
        );
        manager.initialize().unwrap();
        assert_eq!(
            manager
                .prefetch(&id("quantized"), MemoryTier::Host)
                .unwrap(),
            PrefetchOutcome::Miss
        );
        let host = manager.acquire(&id("quantized"), MemoryTier::Host).unwrap();
        assert_eq!(
            host.binding_names().collect::<Vec<_>>(),
            ["scales", "weight"]
        );
        assert_eq!(
            host.array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2]
        );
        assert!(matches!(
            host.array("unknown"),
            Err(ResidencyError::UnknownBinding { .. })
        ));
        drop(host);
        assert_eq!(
            manager
                .prefetch(&id("quantized"), MemoryTier::Device)
                .unwrap(),
            PrefetchOutcome::Miss
        );
        let device = manager
            .acquire(&id("quantized"), MemoryTier::Device)
            .unwrap();
        assert_eq!(device.array("scales").unwrap().shape(), &[2]);
        let report = manager.report().unwrap();
        assert!(state(&report, "quantized").host_resident());
        assert!(state(&report, "quantized").device_resident());
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToHost)
                .bytes(),
            16
        );
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .bytes(),
            16
        );
    }

    #[test]
    fn direct_disk_to_device_does_not_create_a_host_copy() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "a").device_resident());
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToDevice)
                .bytes(),
            8
        );
    }

    #[test]
    fn budgets_use_deterministic_policy_then_lru_eviction() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("cache-a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("window-b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("cache-c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                single("cache-a", "a"),
                single("window-b", "b"),
                single("cache-c", "c"),
            ],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(state(&report, "cache-a").host_resident());
        assert!(!state(&report, "window-b").host_resident());
        assert!(state(&report, "cache-c").host_resident());
        assert_eq!(report.offload().evictions().count(), 1);
        assert_eq!(report.offload().evictions().bytes(), 8);

        manager.evict(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.evict(&id("cache-c"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "cache-a").host_resident());
        assert!(state(&report, "cache-c").host_resident());
        assert!(state(&report, "window-b").host_resident());
        assert!(report.offload().resident_bytes().get(MemoryTier::Host) <= 16);
    }

    #[test]
    fn equal_recency_uses_identifier_as_the_eviction_tie_breaker() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
        {
            let mut state = manager.lock().unwrap();
            state
                .units
                .get_mut(&id("a"))
                .unwrap()
                .host
                .as_mut()
                .unwrap()
                .last_used = 10;
            state
                .units
                .get_mut(&id("b"))
                .unwrap()
                .host
                .as_mut()
                .unwrap()
                .last_used = 10;
        }
        manager.prefetch(&id("c"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "b").host_resident());
        assert!(state(&report, "c").host_resident());
    }

    #[test]
    fn host_and_device_budgets_are_independent() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
        manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "a").device_resident());
        assert!(state(&report, "b").host_resident());
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Host), 8);
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
    }

    #[test]
    fn leases_block_eviction_and_drop_or_unwind_releases_pins() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
        assert!(matches!(
            manager.evict(&id("a"), MemoryTier::Host),
            Err(ResidencyError::InUseEviction { pin_count: 1, .. })
        ));
        drop(lease);
        assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
        assert!(!manager.evict(&id("a"), MemoryTier::Host).unwrap());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let manager = manager.clone();
            move || {
                let _lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
                panic!("exercise lease unwinding");
            }
        }));
        assert!(result.is_err());
        assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    }

    #[test]
    fn concurrent_acquisition_materializes_once_and_counts_pins() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let manager = manager.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
                    barrier.wait();
                    barrier.wait();
                    drop(lease);
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let report = manager.report().unwrap();
        assert_eq!(state(&report, "a").host_pins(), 2);
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToHost)
                .count(),
            1
        );
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Host), 8);
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(state(&manager.report().unwrap(), "a").host_pins(), 0);
    }

    #[test]
    fn windows_bound_lookahead_protect_active_units_and_record_hits() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 2).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        let first = manager
            .prepare_window(&[id("a")], &[id("a"), id("b"), id("c")], MemoryTier::Host)
            .unwrap();
        assert_eq!(first.len(), 2);
        assert!(first
            .iter()
            .all(|(_, value)| *value == PrefetchOutcome::Miss));
        let report_before = manager.report().unwrap();
        assert!(state(&report_before, "a").host_resident());
        assert!(state(&report_before, "b").host_resident());
        assert!(!state(&report_before, "c").host_resident());

        let second = manager
            .prepare_window(&[id("b")], &[id("b"), id("c")], MemoryTier::Host)
            .unwrap();
        assert_eq!(second[0].1, PrefetchOutcome::Hit);
        assert_eq!(second[1].1, PrefetchOutcome::Miss);
        let report_after = manager.report().unwrap();
        assert!(!state(&report_after, "a").host_resident());
        assert!(state(&report_after, "b").host_resident());
        assert!(state(&report_after, "c").host_resident());
        assert_eq!(report_after.active_window(), &[id("b")]);
        assert_eq!(report_after.offload().prefetch().requests(), 4);
        assert_eq!(report_after.offload().prefetch().hits(), 1);
        assert_eq!(report_after.offload().prefetch().misses(), 3);
        assert_eq!(report_before.offload().prefetch().requests(), 2);
    }

    #[test]
    fn exhaustion_reports_pinned_in_use_and_active_blockers() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("pinned", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
                spec("leased", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("wanted", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                single("pinned", "a"),
                single("leased", "b"),
                single("wanted", "c"),
            ],
        );
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("leased"), MemoryTier::Host).unwrap();
        let error = manager
            .prefetch(&id("wanted"), MemoryTier::Host)
            .unwrap_err();
        match error {
            ResidencyError::BudgetExhausted {
                required_bytes,
                budget_bytes,
                blocking_units,
                ..
            } => {
                assert_eq!(required_bytes, 8);
                assert_eq!(budget_bytes, 16);
                assert_eq!(blocking_units.len(), 2);
                assert!(blocking_units.iter().any(|unit| unit.pinned));
                assert!(blocking_units.iter().any(|unit| unit.in_use == 1));
            }
            other => panic!("unexpected error: {other}"),
        }
        drop(lease);
    }

    #[test]
    fn demand_stalls_and_rank_local_selections_are_reported() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("range", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("indices", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                unit(
                    "range",
                    [binding(
                        "weight",
                        "matrix",
                        TensorSelection::Range {
                            axis: 0,
                            start: 1,
                            end: 2,
                        },
                        8,
                    )],
                ),
                unit(
                    "indices",
                    [binding(
                        "weight",
                        "matrix",
                        TensorSelection::Indices {
                            axis: 0,
                            indices: vec![2],
                        },
                        8,
                    )],
                ),
            ],
        );
        manager.initialize().unwrap();
        let range = manager.acquire(&id("range"), MemoryTier::Host).unwrap();
        assert_eq!(
            range
                .array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[12, 13]
        );
        drop(range);
        let indices = manager.acquire(&id("indices"), MemoryTier::Host).unwrap();
        assert_eq!(
            indices
                .array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[14, 15]
        );
        let report = manager.report().unwrap();
        assert_eq!(report.offload().prefetch().stalls(), 2);
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Host), 16);
        assert_eq!(
            report.offload().peak_resident_bytes().get(MemoryTier::Host),
            16
        );
        assert!(report.weight_store().mapping_hits > 0);
    }

    #[test]
    fn ordered_device_window_trims_stale_units_with_unlimited_budget() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(24), 2).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        let window = DeviceLayerWindow::new([id("a"), id("b"), id("c")], 2).unwrap();

        window.prepare(&manager, 0).unwrap();
        let first = manager.report().unwrap();
        assert!(state(&first, "a").device_resident());
        assert!(state(&first, "b").device_resident());
        assert!(!state(&first, "c").device_resident());

        let lease = manager.acquire(&id("b"), MemoryTier::Device).unwrap();
        window.prepare(&manager, 1).unwrap();
        let second = manager.report().unwrap();
        assert!(!state(&second, "a").device_resident());
        assert!(state(&second, "b").device_resident());
        assert!(state(&second, "c").device_resident());
        assert_eq!(state(&second, "b").device_pins(), 1);
        drop(lease);

        window.clear(&manager).unwrap();
        assert!(manager
            .report()
            .unwrap()
            .units()
            .iter()
            .all(|unit| !unit.device_resident()));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn promotes_host_arrays_to_a_real_metal_stream() {
        let (_dir, store) = fixture_store();
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Host)],
        )
        .unwrap();
        let manager = ResidencyManager::new(
            store,
            plan,
            [single("a", "a")],
            cpu_stream(),
            Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)),
        )
        .unwrap();
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(
            lease
                .array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            manager
                .report()
                .unwrap()
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .count(),
            1
        );
    }
}
