//! Architecture-independent execution of decoder models from resident layers.
//!
//! [`LayerwiseModel`] owns checkpoint storage, residency, bounded device
//! windows, and synchronization. Model-family behavior is supplied by a
//! [`LayerwiseModelAdapter`].

use std::{collections::BTreeSet, path::Path, sync::Arc};

use safemlx::{module::ModuleParameters, transforms::eval, Array, Stream};

use crate::{
    cache::KeyValueCache,
    error::Error,
    module_binding::{
        binding_bytes, build_module_bindings, populate_module_from_lease, ModuleBindingError,
    },
    offload::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    },
    residency::{
        OffloadUnit, ResidencyError, ResidencyManager, ResidencyReport, ResidentLayerGroup,
        ResidentUnitLease,
    },
    weight_store::{SafetensorsWeightStore, WeightStore},
};

/// Loader controls for a host-backed layerwise execution engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LayerwiseLoadOptions {
    /// Residency budgets and maximum device-layer window.
    pub offload: OffloadConfig,
    /// Maximum number of checkpoint payload shards retained as mappings.
    pub max_mapped_shards: usize,
    /// Reject checkpoint tensors unrelated to the adapter's parameter tree.
    pub strict_loading: bool,
    /// Sample MLX allocator memory when a forward pass completes.
    pub sample_mlx_memory: bool,
    /// Sample process memory metrics when a forward pass completes.
    pub sample_process_memory: bool,
}

impl LayerwiseLoadOptions {
    /// Creates strict options with the default mapped-shard bound.
    pub fn new(offload: OffloadConfig) -> Self {
        Self {
            offload,
            ..Self::default()
        }
    }
}

impl Default for LayerwiseLoadOptions {
    fn default() -> Self {
        Self {
            offload: OffloadConfig::default(),
            max_mapped_shards: crate::weight_store::DEFAULT_MAX_MAPPED_SHARDS,
            strict_loading: true,
            sample_mlx_memory: false,
            sample_process_memory: false,
        }
    }
}

/// Weight placement choices exposed by architecture-level model loaders.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightResidency {
    /// Construct every module once and keep all parameters on the execution device.
    FullyResident,
    /// Keep decoder weights on a host stream and execute through a bounded device window.
    LayerwiseHost(LayerwiseLoadOptions),
    /// Keep non-expert decoder weights layerwise while caching routed experts independently.
    SparseExpertCache(crate::expert_cache::ExpertCacheLoadOptions),
}

impl Default for WeightResidency {
    fn default() -> Self {
        Self::FullyResident
    }
}

/// Inspectable parameter-residency metadata for a layerwise model.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LayerwiseModelMetadata {
    model_type: String,
    quantization: Option<crate::quantization::WeightQuantization>,
    layer_count: usize,
    static_device_bytes: u64,
    host_layer_bytes: u64,
    maximum_window_bytes: u64,
    device_layer_window: usize,
}

impl LayerwiseModelMetadata {
    /// Returns the checkpoint model type supplied by the adapter.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }
    /// Returns checkpoint-native packed quantization metadata, if present.
    pub const fn quantization(&self) -> Option<crate::quantization::WeightQuantization> {
        self.quantization
    }
    /// Returns the decoder layer count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }
    /// Returns pinned static parameter bytes on the execution device.
    pub const fn static_device_bytes(&self) -> u64 {
        self.static_device_bytes
    }
    /// Returns the complete decoder-weight byte total retained on the host.
    pub const fn host_layer_bytes(&self) -> u64 {
        self.host_layer_bytes
    }
    /// Returns the largest permitted consecutive device-window byte total.
    pub const fn maximum_window_bytes(&self) -> u64 {
        self.maximum_window_bytes
    }
    /// Returns the configured device decoder-layer bound.
    pub const fn device_layer_window(&self) -> usize {
        self.device_layer_window
    }
}

/// One pinned static module and its checkpoint bindings.
pub struct StaticUnitBindings {
    id: OffloadUnitId,
    bindings: Vec<crate::residency::WeightBinding>,
}

impl StaticUnitBindings {
    /// Creates a pinned static unit definition.
    pub fn new(
        id: impl Into<String>,
        bindings: Vec<crate::residency::WeightBinding>,
    ) -> Result<Self, Error> {
        Ok(Self {
            id: OffloadUnitId::new(id.into())?,
            bindings,
        })
    }
}

/// Architecture behavior required by the generic layerwise execution engine.
pub trait LayerwiseModelAdapter: Sized {
    /// Temporary unloaded decoder-block type.
    type Layer: ModuleParameters;
    /// Per-forward state shared by every decoder block.
    type ForwardContext;

    /// Returns the checkpoint model type.
    fn model_type(&self) -> &str;
    /// Returns checkpoint-native packed quantization metadata, if present.
    fn quantization(&self) -> Option<crate::quantization::WeightQuantization>;
    /// Returns the number of decoder blocks.
    fn layer_count(&self) -> Result<usize, Error>;
    /// Builds bindings for modules that remain pinned on the execution device.
    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error>;
    /// Assigns pinned leases to the adapter's static modules.
    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error>;
    /// Creates one metadata-only decoder block.
    fn new_layer(&self, index: usize, stream: &Stream) -> Result<Self::Layer, Error>;
    /// Returns the checkpoint prefix for one decoder block.
    fn layer_checkpoint_prefix(&self, index: usize) -> String;
    /// Returns the stable residency unit name for one decoder block.
    fn layer_unit_name(&self, index: usize) -> String;
    /// Populates one temporary decoder block from its protected lease.
    fn populate_layer(
        &self,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        Ok(populate_module_from_lease(layer, lease)?)
    }
    /// Builds direct or derived bindings for one decoder block.
    fn layer_bindings(
        &self,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<crate::residency::WeightBinding>, Error> {
        Ok(build_module_bindings(
            layer,
            &self.layer_checkpoint_prefix(index),
            store,
        )?)
    }
    /// Returns checkpoint keys consumed by dependent units outside the block unit.
    fn additional_consumed_checkpoint_keys(&self, _store: &dyn WeightStore) -> Vec<String> {
        Vec::new()
    }
    /// Executes the architecture's input embedding.
    fn embed(&mut self, inputs: &Array, stream: &Stream) -> Result<Array, Error>;
    /// Prepares masks or other state shared by the decoder-block loop.
    fn prepare_forward<C: KeyValueCache>(
        &self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &[Option<C>],
        stream: &Stream,
    ) -> Result<Self::ForwardContext, Error>;
    /// Executes one populated decoder block.
    fn forward_layer<C: KeyValueCache>(
        &self,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut C,
        context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error>;
    /// Applies final normalization and logits projection.
    fn finish(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Error>;
    /// Returns whether a checkpoint key is intentionally ignored by strict loading.
    fn ignores_checkpoint_key(&self, _key: &str) -> bool {
        false
    }
}

/// Forward state returned by a generalized architecture adapter.
pub struct LayerwiseForwardState<C> {
    /// Activation consumed by the first sequential execution group.
    pub hidden: Array,
    /// Architecture-owned masks, positions, and auxiliary per-forward state.
    pub context: C,
}

/// General adapter contract for heterogeneous caches and architecture-specific input.
///
/// The original [`LayerwiseModelAdapter`] remains available for Llama-compatible
/// callers. New hybrid, multimodal, and realtime adapters can use this companion
/// contract without pretending recurrent or convolution state is a KV cache.
pub trait GeneralLayerwiseModelAdapter: Sized {
    /// Borrowed family-specific forward input.
    type Input<'a>;
    /// Complete architecture-owned cache and recurrent state.
    type Cache;
    /// Runtime execution unit. Families with heterogeneous blocks may use an enum.
    type Layer: ModuleParameters;
    /// Masks, positions, prepared media, or other per-forward state.
    type ForwardContext;

    /// Builds bindings for modules that remain pinned on the execution device.
    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error>;

    /// Assigns pinned leases to the adapter's static modules.
    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error>;

    /// Validates or initializes the complete cache before any weight lease is acquired.
    fn validate_cache(&self, cache: &mut Self::Cache) -> Result<(), Error>;

    /// Embeds or prepares the input and creates family-owned forward context.
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error>;

    /// Returns the number of named sequential groups used by this adapter.
    fn execution_group_count(&self) -> usize;

    /// Returns the stable name of one sequential execution group.
    fn execution_group_id(&self, group: usize) -> Result<String, Error>;

    /// Returns whether a group is needed for this particular forward pass.
    ///
    /// This lets multimodal adapters skip vision groups during text-only decode.
    fn should_execute_group(&self, _group: usize, _context: &Self::ForwardContext) -> bool {
        true
    }

    /// Returns the number of ordered units in one group.
    fn layer_count(&self, group: usize) -> Result<usize, Error>;

    /// Creates a metadata-only runtime unit for one group position.
    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error>;

    /// Returns the checkpoint prefix for one runtime unit.
    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String;

    /// Returns the stable residency unit name for one runtime unit.
    fn layer_unit_name(&self, group: usize, index: usize) -> String;
    /// Populates one temporary execution unit from its protected lease.
    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        Ok(populate_module_from_lease(layer, lease)?)
    }

    /// Builds direct or derived bindings for one runtime unit.
    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<crate::residency::WeightBinding>, Error> {
        Ok(build_module_bindings(
            layer,
            &self.layer_checkpoint_prefix(group, index),
            store,
        )?)
    }

    /// Returns checkpoint keys consumed by dependent units outside execution groups.
    fn additional_consumed_checkpoint_keys(&self, _store: &dyn WeightStore) -> Vec<String> {
        Vec::new()
    }

    /// Executes one populated unit while inspecting and mutating the complete cache.
    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Returns every cache/state array that must be evaluated before lease release.
    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array>;

    /// Returns transient forward-context arrays that must be evaluated before lease release.
    fn retained_context_arrays<'a>(
        &self,
        _context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Vec<&'a Array> {
        Vec::new()
    }

    /// Converts one group's output into the activation consumed by the next group.
    ///
    /// Multimodal adapters use this hook to merge encoded media before entering
    /// a text decoder. Homogeneous adapters keep the activation unchanged.
    fn finish_execution_group(
        &mut self,
        _group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(hidden.clone())
    }

    /// Applies final normalization, projections, or family-specific output assembly.
    fn finish(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Returns whether a checkpoint key is intentionally ignored by strict loading.
    fn ignores_checkpoint_key(&self, _key: &str) -> bool {
        false
    }
}

/// Residency-owned execution engine for generalized adapters.
///
/// Group windows, lease lifetime, retained-state evaluation, stream
/// synchronization, and telemetry stay centralized here. Adapter code owns only
/// architecture math, cache validation, and runtime-unit construction.
pub struct GeneralLayerwiseModel<A: GeneralLayerwiseModelAdapter> {
    adapter: A,
    store: Arc<SafetensorsWeightStore>,
    residency: ResidencyManager,
    groups: Vec<ResidentLayerGroup>,
    static_leases: Vec<ResidentUnitLease>,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

impl<A: GeneralLayerwiseModelAdapter> GeneralLayerwiseModel<A> {
    /// Creates an engine from a validated residency manager and execution groups.
    pub fn new(
        adapter: A,
        store: Arc<SafetensorsWeightStore>,
        residency: ResidencyManager,
        groups: Vec<ResidentLayerGroup>,
        static_leases: Vec<ResidentUnitLease>,
    ) -> Result<Self, Error> {
        if groups.len() != adapter.execution_group_count() {
            return Err(LayerwiseModelError::ExecutionGroupCount {
                adapter: adapter.execution_group_count(),
                configured: groups.len(),
            }
            .into());
        }
        for (group_index, group) in groups.iter().enumerate() {
            let expected = adapter.layer_count(group_index)?;
            if expected != group.units().len() {
                return Err(LayerwiseModelError::ExecutionGroupLength {
                    group: group.id().to_string(),
                    adapter: expected,
                    configured: group.units().len(),
                }
                .into());
            }
        }
        Ok(Self {
            adapter,
            store,
            residency,
            groups,
            static_leases,
            sample_mlx_memory: false,
            sample_process_memory: false,
        })
    }

    /// Enables optional allocator and process-memory samples after forward.
    pub fn with_memory_sampling(mut self, mlx: bool, process: bool) -> Self {
        self.sample_mlx_memory = mlx;
        self.sample_process_memory = process;
        self
    }

    /// Returns the architecture adapter.
    pub const fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Returns the mutable adapter for loader-time dependent-unit setup.
    pub(crate) fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }

    /// Returns a shared handle to the persistent checkpoint store.
    pub(crate) fn weight_store_arc(&self) -> Arc<SafetensorsWeightStore> {
        Arc::clone(&self.store)
    }

    /// Returns the persistent checkpoint store.
    pub fn weight_store(&self) -> &SafetensorsWeightStore {
        &self.store
    }

    /// Returns named execution groups in deterministic order.
    pub fn execution_groups(&self) -> &[ResidentLayerGroup] {
        &self.groups
    }

    /// Returns the reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.residency
    }

    /// Returns a current residency and transfer report.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        Ok(self.residency.report()?)
    }

    /// Runs every sequential group while centrally enforcing lease safety.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            .map(|(output, _)| output)
    }

    /// Runs a generalized forward pass and invokes `hook` after each execution unit.
    ///
    /// Realtime autoregressive subgroups use this to turn one unit's logits into
    /// the token consumed by the next unit without moving lease ownership out of
    /// the shared residency engine.
    pub(crate) fn forward_with_context_hook<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        mut hook: F,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.adapter.validate_cache(cache)?;
        let LayerwiseForwardState {
            mut hidden,
            mut context,
        } = self.adapter.begin_forward(input, cache, stream)?;

        for (group_index, group) in self.groups.iter().enumerate() {
            if self.adapter.should_execute_group(group_index, &context) {
                for index in 0..group.units().len() {
                    group.prepare(&self.residency, index)?;
                    let id = &group.units()[index];
                    {
                        let lease = self.residency.acquire(id, MemoryTier::Device)?;
                        let mut layer = self.adapter.new_layer(group_index, index, stream)?;
                        self.adapter
                            .populate_layer(group_index, index, &mut layer, &lease)?;
                        hidden = self.adapter.forward_layer(
                            group_index,
                            index,
                            &mut layer,
                            &hidden,
                            cache,
                            &mut context,
                            stream,
                        )?;
                        let hook_result = hook(group_index, index, &mut context);
                        let retained = self.adapter.retained_arrays(cache, group_index, index);
                        let retained_context =
                            self.adapter
                                .retained_context_arrays(&context, group_index, index);
                        eval(
                            std::iter::once(&hidden)
                                .chain(retained.into_iter())
                                .chain(retained_context.into_iter()),
                        )?;
                        stream.synchronize()?;
                        hook_result?;
                    }
                    let end = index.saturating_add(group.depth()).min(group.units().len());
                    group.trim_to(&self.residency, &group.units()[index..end])?;
                }
            }
            hidden = self.adapter.finish_execution_group(
                group_index,
                &hidden,
                cache,
                &mut context,
                stream,
            )?;
            let retained_context =
                self.adapter
                    .retained_context_arrays(&context, group_index, group.units().len());
            eval(std::iter::once(&hidden).chain(retained_context))?;
            stream.synchronize()?;
        }

        let output = self.adapter.finish(&hidden, cache, &context, stream)?;
        eval([&output])?;
        stream.synchronize()?;
        if self.sample_mlx_memory || self.sample_process_memory {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        Ok((output, context))
    }

    /// Clears one named execution group without affecting other groups.
    pub fn clear_device_group(&self, id: &str) -> Result<(), Error> {
        let group = self
            .groups
            .iter()
            .find(|group| group.id() == id)
            .ok_or_else(|| LayerwiseModelError::UnknownExecutionGroup(id.to_string()))?;
        Ok(group.clear(&self.residency)?)
    }

    /// Clears every temporary device execution group.
    pub fn clear_all_device_groups(&self) -> Result<(), Error> {
        for group in &self.groups {
            group.clear(&self.residency)?;
        }
        Ok(())
    }

    /// Returns the number of pinned static leases held by the engine.
    pub fn static_lease_count(&self) -> usize {
        self.static_leases.len()
    }
}

/// Builds a generalized host-backed model with independently bounded groups.
pub fn load_general_layerwise_model<A: GeneralLayerwiseModelAdapter>(
    model_dir: impl AsRef<Path>,
    mut adapter: A,
    options: LayerwiseLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GeneralLayerwiseModel<A>, Error> {
    let model_dir = model_dir.as_ref();
    if model_dir.extension().and_then(|value| value.to_str()) == Some("gguf") {
        return Err(LayerwiseModelError::GgufUnsupported.into());
    }
    let depth = options.offload.prefetch_depth();
    let store = Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(
        model_dir,
        options.max_mapped_shards,
    )?);

    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut static_device_bytes = 0u64;
    let mut static_ids = Vec::new();
    for unit in adapter.static_units(store.as_ref())? {
        static_ids.push(unit.id.clone());
        add_unit(
            &mut definitions,
            &mut specs,
            &mut consumed,
            unit.id,
            unit.bindings,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            &mut static_device_bytes,
        )?;
    }

    let mut groups = Vec::with_capacity(adapter.execution_group_count());
    let mut host_layer_bytes = 0u64;
    let mut combined_window_bytes = 0u64;
    for group_index in 0..adapter.execution_group_count() {
        let layer_count = adapter.layer_count(group_index)?;
        if depth > layer_count {
            return Err(LayerwiseModelError::InvalidLayerWindow { depth, layer_count }.into());
        }
        let mut layer_ids = Vec::with_capacity(layer_count);
        let mut layer_bytes = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let layer = adapter.new_layer(group_index, index, stream)?;
            let bindings = adapter.layer_bindings(group_index, index, &layer, store.as_ref())?;
            let bytes = binding_bytes(&bindings)?;
            host_layer_bytes = host_layer_bytes.checked_add(bytes).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "host execution-unit byte total",
                },
            )?;
            let id = OffloadUnitId::new(adapter.layer_unit_name(group_index, index))?;
            consumed.extend(
                bindings
                    .iter()
                    .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
            );
            definitions.push(OffloadUnit::new(id.clone(), bindings)?);
            specs.push(OffloadUnitSpec::new(
                id.clone(),
                bytes,
                ResidencyPolicy::Windowed,
                MemoryTier::Host,
            )?);
            layer_ids.push(id);
            layer_bytes.push(bytes);
        }
        combined_window_bytes = combined_window_bytes
            .checked_add(largest_window_bytes(&layer_bytes, depth)?)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "combined device execution-window byte total",
            })?;
        groups.push(ResidentLayerGroup::new(
            adapter.execution_group_id(group_index)?,
            layer_ids,
            depth,
        )?);
    }

    consumed.extend(adapter.additional_consumed_checkpoint_keys(store.as_ref()));

    validate_unused(store.as_ref(), &consumed, options.strict_loading, |key| {
        adapter.ignores_checkpoint_key(key)
    })?;
    validate_host_budget(options.offload, host_layer_bytes)?;
    validate_device_budget(
        options.offload,
        static_device_bytes,
        combined_window_bytes,
        depth,
    )?;

    let plan = OffloadPlan::new(options.offload, specs)?;
    let residency = ResidencyManager::new(
        Arc::clone(&store),
        plan,
        definitions,
        weights_stream.clone(),
        stream.clone(),
    )?;
    residency.initialize()?;
    let static_leases = static_ids
        .iter()
        .map(|id| residency.acquire(id, MemoryTier::Device))
        .collect::<Result<Vec<_>, _>>()?;
    adapter.populate_static(&static_leases)?;

    GeneralLayerwiseModel::new(adapter, store, residency, groups, static_leases).map(|model| {
        model.with_memory_sampling(options.sample_mlx_memory, options.sample_process_memory)
    })
}

/// Generic host-backed layerwise decoder execution engine.
pub struct LayerwiseModel<A: LayerwiseModelAdapter> {
    adapter: A,
    store: Arc<SafetensorsWeightStore>,
    residency: ResidencyManager,
    layer_group: ResidentLayerGroup,
    static_leases: Vec<ResidentUnitLease>,
    metadata: LayerwiseModelMetadata,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

impl<A: LayerwiseModelAdapter> LayerwiseModel<A> {
    /// Returns the architecture adapter.
    pub const fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Returns the mutable architecture adapter for loader-time dependent-unit setup.
    pub(crate) fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }

    /// Returns a shared handle to the persistent checkpoint store.
    pub(crate) fn weight_store_arc(&self) -> Arc<SafetensorsWeightStore> {
        Arc::clone(&self.store)
    }

    /// Returns parameter-residency metadata.
    pub const fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Returns the persistent checkpoint store.
    pub fn weight_store(&self) -> &SafetensorsWeightStore {
        &self.store
    }

    /// Returns the reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.residency
    }

    /// Returns a current logical residency, transfer, and store report.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        Ok(self.residency.report()?)
    }

    /// Runs the model with a caller-selected compatible cache implementation.
    pub fn forward_with_cache<C>(
        &mut self,
        input: LayerwiseInput<'_, C>,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        C: KeyValueCache + Default,
    {
        let LayerwiseInput {
            inputs,
            mask,
            cache,
        } = input;
        validate_cache(cache, self.metadata.layer_count)?;

        let mut h = self.adapter.embed(inputs, stream)?;
        let context = self.adapter.prepare_forward(&h, mask, cache, stream)?;

        for index in 0..self.metadata.layer_count {
            self.layer_group.prepare(&self.residency, index)?;
            let id = &self.layer_group.units()[index];
            {
                let lease = self.residency.acquire(id, MemoryTier::Device)?;
                let mut layer = self.adapter.new_layer(index, stream)?;
                self.adapter.populate_layer(&mut layer, &lease)?;
                let layer_cache = cache[index]
                    .as_mut()
                    .ok_or(LayerwiseModelError::MissingLayerCache { index })?;
                h = self.adapter.forward_layer(
                    index,
                    &mut layer,
                    &h,
                    layer_cache,
                    &context,
                    stream,
                )?;

                // MLX is lazy. Materialize both the activation and every cache
                // handle updated by this block before its lease can be dropped.
                eval(std::iter::once(&h).chain(layer_cache.retained_arrays()))?;
                stream.synchronize()?;
            }
            let end = index
                .saturating_add(self.layer_group.depth())
                .min(self.layer_group.units().len());
            let desired = &self.layer_group.units()[index..end];
            self.layer_group.trim_to(&self.residency, desired)?;
        }

        let logits = self.adapter.finish(&h, stream)?;
        if self.sample_mlx_memory || self.sample_process_memory {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        Ok(logits)
    }

    /// Explicitly evicts all decoder copies from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        Ok(self.layer_group.clear(&self.residency)?)
    }

    /// Returns the number of long-lived pinned static leases.
    pub fn static_lease_count(&self) -> usize {
        self.static_leases.len()
    }
}

/// Input shared by architecture adapters using the layerwise engine.
pub struct LayerwiseInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional caller-provided attention mask.
    pub mask: Option<&'a Array>,
    /// Mutable per-layer caches.
    pub cache: &'a mut Vec<Option<C>>,
}

/// Builds a generic layerwise model from an architecture adapter and safetensors.
pub fn load_layerwise_model<A: LayerwiseModelAdapter>(
    model_dir: impl AsRef<Path>,
    mut adapter: A,
    options: LayerwiseLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error> {
    let model_dir = model_dir.as_ref();
    if model_dir.extension().and_then(|value| value.to_str()) == Some("gguf") {
        return Err(LayerwiseModelError::GgufUnsupported.into());
    }
    let layer_count = adapter.layer_count()?;
    let depth = options.offload.prefetch_depth();
    if depth > layer_count {
        return Err(LayerwiseModelError::InvalidLayerWindow { depth, layer_count }.into());
    }
    let store = Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(
        model_dir,
        options.max_mapped_shards,
    )?);

    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut static_device_bytes = 0u64;
    let mut static_ids = Vec::new();

    for unit in adapter.static_units(store.as_ref())? {
        static_ids.push(unit.id.clone());
        add_unit(
            &mut definitions,
            &mut specs,
            &mut consumed,
            unit.id,
            unit.bindings,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            &mut static_device_bytes,
        )?;
    }

    let mut layer_ids = Vec::with_capacity(layer_count);
    let mut layer_bytes = Vec::with_capacity(layer_count);
    let mut host_layer_bytes = 0u64;
    for index in 0..layer_count {
        let layer = adapter.new_layer(index, stream)?;
        let bindings = adapter.layer_bindings(index, &layer, store.as_ref())?;
        let bytes = binding_bytes(&bindings)?;
        host_layer_bytes =
            host_layer_bytes
                .checked_add(bytes)
                .ok_or(LayerwiseModelError::ArithmeticOverflow {
                    context: "host decoder byte total",
                })?;
        let id = OffloadUnitId::new(adapter.layer_unit_name(index))?;
        consumed.extend(
            bindings
                .iter()
                .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
        );
        definitions.push(OffloadUnit::new(id.clone(), bindings)?);
        specs.push(OffloadUnitSpec::new(
            id.clone(),
            bytes,
            ResidencyPolicy::Windowed,
            MemoryTier::Host,
        )?);
        layer_ids.push(id);
        layer_bytes.push(bytes);
    }

    consumed.extend(adapter.additional_consumed_checkpoint_keys(store.as_ref()));

    validate_unused(store.as_ref(), &consumed, options.strict_loading, |key| {
        adapter.ignores_checkpoint_key(key)
    })?;
    validate_host_budget(options.offload, host_layer_bytes)?;
    let maximum_window_bytes = largest_window_bytes(&layer_bytes, depth)?;
    validate_device_budget(
        options.offload,
        static_device_bytes,
        maximum_window_bytes,
        depth,
    )?;

    let plan = OffloadPlan::new(options.offload, specs)?;
    let residency = ResidencyManager::new(
        Arc::clone(&store),
        plan,
        definitions,
        weights_stream.clone(),
        stream.clone(),
    )?;
    residency.initialize()?;

    let static_leases = static_ids
        .iter()
        .map(|id| residency.acquire(id, MemoryTier::Device))
        .collect::<Result<Vec<_>, _>>()?;
    adapter.populate_static(&static_leases)?;

    let layer_group = ResidentLayerGroup::new("text_decoder", layer_ids, depth)?;
    let metadata = LayerwiseModelMetadata {
        model_type: adapter.model_type().to_string(),
        quantization: adapter.quantization(),
        layer_count,
        static_device_bytes,
        host_layer_bytes,
        maximum_window_bytes,
        device_layer_window: depth,
    };
    Ok(LayerwiseModel {
        adapter,
        store,
        residency,
        layer_group,
        static_leases,
        metadata,
        sample_mlx_memory: options.sample_mlx_memory,
        sample_process_memory: options.sample_process_memory,
    })
}

fn add_unit(
    definitions: &mut Vec<OffloadUnit>,
    specs: &mut Vec<OffloadUnitSpec>,
    consumed: &mut BTreeSet<String>,
    id: OffloadUnitId,
    bindings: Vec<crate::residency::WeightBinding>,
    policy: ResidencyPolicy,
    tier: MemoryTier,
    byte_total: &mut u64,
) -> Result<(), Error> {
    let bytes = binding_bytes(&bindings)?;
    *byte_total = byte_total
        .checked_add(bytes)
        .ok_or(LayerwiseModelError::ArithmeticOverflow {
            context: "static device byte total",
        })?;
    consumed.extend(
        bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
    );
    definitions.push(OffloadUnit::new(id.clone(), bindings)?);
    specs.push(OffloadUnitSpec::new(id, bytes, policy, tier)?);
    Ok(())
}

fn validate_unused<F>(
    store: &dyn WeightStore,
    consumed: &BTreeSet<String>,
    strict: bool,
    ignored: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    if !strict {
        return Ok(());
    }
    let unused = store
        .keys()
        .into_iter()
        .filter(|key| !consumed.contains(key))
        .filter(|key| !ignored(key))
        .collect::<Vec<_>>();
    if unused.is_empty() {
        Ok(())
    } else {
        Err(LayerwiseModelError::UnexpectedCheckpointParameters { unused }.into())
    }
}

fn validate_cache<C>(cache: &mut Vec<Option<C>>, layer_count: usize) -> Result<(), Error>
where
    C: KeyValueCache + Default,
{
    if cache.is_empty() {
        *cache = (0..layer_count).map(|_| Some(C::default())).collect();
        return Ok(());
    }
    if cache.len() != layer_count {
        return Err(LayerwiseModelError::CacheLengthMismatch {
            expected: layer_count,
            actual: cache.len(),
        }
        .into());
    }
    if let Some(index) = cache.iter().position(Option::is_none) {
        return Err(LayerwiseModelError::MissingLayerCache { index }.into());
    }
    Ok(())
}

fn largest_window_bytes(layer_bytes: &[u64], depth: usize) -> Result<u64, Error> {
    let mut largest = 0u64;
    for start in 0..layer_bytes.len() {
        let mut current = 0u64;
        for bytes in layer_bytes.iter().skip(start).take(depth) {
            current =
                current
                    .checked_add(*bytes)
                    .ok_or(LayerwiseModelError::ArithmeticOverflow {
                        context: "device layer window byte total",
                    })?;
        }
        largest = largest.max(current);
    }
    Ok(largest)
}

fn validate_host_budget(config: OffloadConfig, required: u64) -> Result<(), Error> {
    if let Some(budget) = config.host_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::HostBudgetTooSmall { required, budget }.into());
        }
    }
    Ok(())
}

fn validate_device_budget(
    config: OffloadConfig,
    static_bytes: u64,
    window_bytes: u64,
    depth: usize,
) -> Result<(), Error> {
    let required =
        static_bytes
            .checked_add(window_bytes)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "static plus device-window byte total",
            })?;
    if let Some(budget) = config.device_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::DeviceBudgetTooSmall {
                static_bytes,
                window_bytes,
                depth,
                required,
                budget,
            }
            .into());
        }
    }
    Ok(())
}

/// Structured failures produced by the generic layerwise execution engine.
#[derive(Debug, thiserror::Error)]
pub enum LayerwiseModelError {
    /// Adapter and configured execution-group counts differ.
    #[error("adapter declares {adapter} execution groups but {configured} were configured")]
    ExecutionGroupCount {
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// Adapter and configured unit counts differ for one execution group.
    #[error("execution group {group:?} has {configured} configured units but adapter declares {adapter}")]
    ExecutionGroupLength {
        /// Group id.
        group: String,
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// A requested execution group does not exist.
    #[error("unknown resident execution group {0:?}")]
    UnknownExecutionGroup(String),
    /// GGUF is intentionally outside this loader's safetensors contract.
    #[error("layerwise host residency requires safetensors; GGUF is unsupported")]
    GgufUnsupported,
    /// The configured ordered layer window was invalid.
    #[error("device layer window depth {depth} must be between 1 and layer count {layer_count}")]
    InvalidLayerWindow {
        /// Requested depth.
        depth: usize,
        /// Decoder layer count.
        layer_count: usize,
    },
    /// Strict loading found unrelated checkpoint tensors.
    #[error("strict layerwise loading found unexpected checkpoint parameters: {unused:?}")]
    UnexpectedCheckpointParameters {
        /// Unexpected keys in stable order.
        unused: Vec<String>,
    },
    /// The host cannot retain every decoder layer.
    #[error("host budget {budget} bytes cannot contain all {required} decoder-weight bytes")]
    HostBudgetTooSmall {
        /// Required decoder bytes.
        required: u64,
        /// Configured host budget.
        budget: u64,
    },
    /// The device cannot contain static weights plus the configured window.
    #[error("device budget {budget} bytes cannot contain {static_bytes} static bytes plus the depth-{depth} layer window ({window_bytes} bytes, {required} total)")]
    DeviceBudgetTooSmall {
        /// Pinned static device bytes.
        static_bytes: u64,
        /// Largest consecutive window bytes.
        window_bytes: u64,
        /// Configured layer count.
        depth: usize,
        /// Total required parameter bytes.
        required: u64,
        /// Configured device budget.
        budget: u64,
    },
    /// A cache vector had the wrong number of layers.
    #[error("layerwise cache has {actual} layers, expected {expected}")]
    CacheLengthMismatch {
        /// Model decoder count.
        expected: usize,
        /// Supplied cache count.
        actual: usize,
    },
    /// A cache entry was absent.
    #[error("layerwise cache entry {index} is missing")]
    MissingLayerCache {
        /// Missing decoder index.
        index: usize,
    },
    /// Checked byte or index arithmetic overflowed.
    #[error("layerwise model arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Failed calculation.
        context: &'static str,
    },
    /// Module checkpoint binding failed.
    #[error(transparent)]
    ModuleBinding(#[from] ModuleBindingError),
    /// Residency execution failed.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
}

#[cfg(test)]
mod tests {
    use std::fs;

    use safemlx::{
        module::ModuleParameters, ops::ones_dtype, Device, DeviceType, ExecutionContext,
    };

    use super::*;
    use crate::{
        llama::{load_llama_model, LlamaCache, LlamaLoadOptions, LlamaModel},
        models::llama::{self, ModelArgs},
        offload::TransferDirection,
        residency::UnitResidencyReport,
    };

    fn load_layerwise_llama(
        model_dir: impl AsRef<Path>,
        offload: OffloadConfig,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<LlamaModel, Error> {
        load_llama_model(
            model_dir,
            LlamaLoadOptions::layerwise_host(LayerwiseLoadOptions::new(offload)),
            stream,
            weights_stream,
        )
    }

    fn args(model_type: &str, tied: bool, sliding_window: Option<i32>) -> ModelArgs {
        ModelArgs {
            model_type: model_type.into(),
            hidden_size: 8,
            num_hidden_layers: 3,
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
            attention_bias: true,
            mlp_bias: true,
            rope_scaling: None,
            sliding_window,
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    fn initialize(module: &mut impl ModuleParameters, stream: &Stream) {
        let mut names = module
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        names.sort();
        let mut params = module.parameters_mut().flatten();
        for (index, name) in names.iter().enumerate() {
            let parameter = params.get_mut(name.as_str()).unwrap();
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter = if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.0025 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &llama::ResidentModel) {
        let params = model.parameters().flatten();
        let arrays = params
            .iter()
            .map(|(name, value)| {
                (
                    crate::module_binding::canonical_checkpoint_name(name),
                    *value,
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), *value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        let mut config = serde_json::json!({
            "model_type": model.args.model_type,
            "hidden_size": model.args.hidden_size,
            "num_hidden_layers": model.args.num_hidden_layers,
            "intermediate_size": model.args.intermediate_size,
            "num_attention_heads": model.args.num_attention_heads,
            "num_key_value_heads": model.args.num_key_value_heads,
            "rms_norm_eps": model.args.rms_norm_eps,
            "vocab_size": model.args.vocab_size,
            "max_position_embeddings": model.args.max_position_embeddings,
            "rope_theta": model.args.rope_theta,
            "rope_traditional": model.args.rope_traditional,
            "head_dim": model.args.head_dim,
            "tie_word_embeddings": model.args.tie_word_embeddings,
            "attention_bias": model.args.attention_bias,
            "mlp_bias": model.args.mlp_bias
        });
        if let Some(window) = model.args.sliding_window {
            config["sliding_window"] = window.into();
        }
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 2e-5, "{left} != {right}");
        }
    }

    fn layer_reports(report: &ResidencyReport) -> Vec<&UnitResidencyReport> {
        report
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("llama.layer."))
            .collect()
    }

    fn run_parity(model_type: &str, tied: bool, sliding_window: Option<i32>, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = gpu.stream();
        let mut reference =
            llama::ResidentModel::new(args(model_type, tied, sliding_window), stream).unwrap();
        initialize(&mut reference, stream);
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let mut fully_resident = load_llama_model(
            dir.path(),
            LlamaLoadOptions::fully_resident(),
            stream,
            cpu.stream(),
        )
        .unwrap();
        assert!(fully_resident.is_fully_resident());
        assert!(fully_resident.residency_report().unwrap().is_none());
        let config = OffloadConfig::new(None, None, depth).unwrap();
        let mut offloaded = load_layerwise_llama(dir.path(), config, stream, cpu.stream()).unwrap();
        assert!(!offloaded.is_fully_resident());
        let initial = offloaded.residency_report().unwrap().unwrap();
        assert!(layer_reports(&initial)
            .iter()
            .all(|unit| unit.host_resident()));
        assert!(layer_reports(&initial)
            .iter()
            .all(|unit| !unit.device_resident()));

        let mut resident_cache = fully_resident.new_cache();
        let mut cache = offloaded.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = fully_resident
                .forward(&tokens, &mut resident_cache, stream)
                .unwrap();
            let actual = offloaded.forward(&tokens, &mut cache, stream).unwrap();
            assert_close(&actual, &expected);
            let report = offloaded.residency_report().unwrap().unwrap();
            assert!(layer_reports(&report)
                .iter()
                .all(|unit| unit.host_resident()));
            assert!(
                layer_reports(&report)
                    .iter()
                    .filter(|unit| unit.device_resident())
                    .count()
                    <= depth
            );
        }

        let report = offloaded.residency_report().unwrap().unwrap();
        assert!(
            report
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .count()
                >= 3
        );
        assert_eq!(
            offloaded.layerwise_static_lease_count().unwrap(),
            if tied { 2 } else { 3 }
        );
        offloaded.clear_device_layer_window().unwrap();
        let cleared = offloaded.residency_report().unwrap().unwrap();
        assert!(layer_reports(&cleared)
            .iter()
            .all(|unit| !unit.device_resident()));
        assert!(cleared
            .units()
            .iter()
            .filter(|unit| unit.device_resident())
            .all(|unit| unit.policy() == ResidencyPolicy::Pinned));
    }

    #[test]
    fn llama_residency_dense_prefill_decode_parity() {
        run_parity("llama", true, None, 1);
        run_parity("llama", false, None, 2);
        run_parity("mistral", false, Some(4), 2);
    }

    #[test]
    fn budget_and_cache_validation_are_structured() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut reference =
            llama::ResidentModel::new(args("llama", true, None), gpu.stream()).unwrap();
        initialize(&mut reference, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let host_error = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(None, Some(1), 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .err()
        .unwrap();
        assert!(host_error.to_string().contains("host budget"));

        let device_error = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(Some(1), None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .err()
        .unwrap();
        assert!(device_error.to_string().contains("device budget"));

        let mut model = load_layerwise_llama(
            dir.path(),
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut bad_cache = LlamaCache::Standard(vec![None]);
        let error = model
            .forward(
                &Array::from_slice(&[1u32], &[1, 1]),
                &mut bad_cache,
                gpu.stream(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("cache has 1 layers"));
    }

    #[test]
    fn llama_residency_packed_affine_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut quant_args = args("llama", false, None);
        quant_args.hidden_size = 32;
        quant_args.intermediate_size = 64;
        quant_args.num_attention_heads = 4;
        quant_args.num_key_value_heads = 2;
        quant_args.head_dim = 8;
        quant_args.vocab_size = 32;
        quant_args.num_hidden_layers = 2;
        let mut dense = llama::ResidentModel::new(quant_args, gpu.stream()).unwrap();
        initialize(&mut dense, gpu.stream());
        let source = tempfile::tempdir().unwrap();
        write_fixture(source.path(), &dense);

        let converted_root = tempfile::tempdir().unwrap();
        let converted = converted_root.path().join("affine");
        let options = crate::quantization::CheckpointQuantizationOptions {
            quantization: crate::quantization::AffineQuantization::new(32, 4)
                .unwrap()
                .into(),
            ..Default::default()
        };
        crate::quantization::quantize_checkpoint(source.path(), &converted, &options, gpu.stream())
            .unwrap();

        let mut resident = load_llama_model(
            &converted,
            LlamaLoadOptions::fully_resident(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut offloaded = load_layerwise_llama(
            &converted,
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        assert!(offloaded
            .layerwise_metadata()
            .unwrap()
            .quantization()
            .is_some());
        let mut resident_cache = resident.new_cache();
        let mut offloaded_cache = offloaded.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut resident_cache, gpu.stream())
                .unwrap();
            let actual = offloaded
                .forward(&tokens, &mut offloaded_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
    }
}
