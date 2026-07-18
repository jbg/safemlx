//! Explicit layerwise host offload for Llama-compatible safetensors models.
//!
//! Decoder weights remain evaluated on the configured CPU source stream and
//! move synchronously through a bounded execution-device window. Embeddings,
//! final normalization, an untied output head, activations, and KV caches stay
//! on the execution device.

use std::{collections::BTreeSet, path::Path, sync::Arc};

use safemlx::{
    error::Exception, module::Module, nn, ops::indexing::TryIndexOp, quantization::MaybeQuantized,
    transforms::eval, Array, Dtype, Stream,
};

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache, SlidingKeyValueCache},
    error::Error,
    models::{
        common::{
            generation::CausalLm,
            linear::{
                build_unloaded_maybe_quantized_lm_head_with_quantization,
                project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            },
        },
        input,
        llama::{self, AttentionInput, ModelArgs, ModelInput, TransformerBlock},
    },
    module_binding::{
        binding_bytes, build_module_bindings, populate_module_from_lease, ModuleBindingError,
    },
    offload::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    },
    residency::{
        DeviceLayerWindow, OffloadUnit, ResidencyError, ResidencyManager, ResidencyReport,
        ResidentUnitLease,
    },
    utils::{create_attention_mask, create_sliding_attention_mask, AttentionMask},
    weight_store::{SafetensorsWeightStore, WeightStore},
};

const EMBEDDING_UNIT: &str = "llama.static.embedding";
const NORM_UNIT: &str = "llama.static.norm";
const HEAD_UNIT: &str = "llama.static.output";

/// Loader controls consumed by the explicit Llama host-offload path.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LlamaHostOffloadOptions {
    /// Residency budgets and maximum device-layer window.
    pub offload: OffloadConfig,
    /// Maximum number of checkpoint payload shards retained as mappings.
    pub max_mapped_shards: usize,
    /// Reject checkpoint tensors unrelated to the Llama parameter tree.
    pub strict_loading: bool,
    /// Sample MLX allocator memory when a forward pass completes.
    pub sample_mlx_memory: bool,
    /// Sample process memory metrics when a forward pass completes.
    pub sample_process_memory: bool,
}

impl LlamaHostOffloadOptions {
    /// Creates strict options with the default mapped-shard bound.
    pub fn new(offload: OffloadConfig) -> Self {
        Self {
            offload,
            ..Self::default()
        }
    }
}

impl Default for LlamaHostOffloadOptions {
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

/// Inspectable parameter-residency metadata for an offloaded Llama model.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadedLlamaMetadata {
    model_type: String,
    quantization: Option<crate::quantization::WeightQuantization>,
    layer_count: usize,
    static_device_bytes: u64,
    host_layer_bytes: u64,
    maximum_window_bytes: u64,
    device_layer_window: usize,
}

impl OffloadedLlamaMetadata {
    /// Returns `llama` or `mistral`.
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

/// Model-aware standard or sliding per-layer KV cache.
#[derive(Debug, Clone)]
pub enum OffloadedLlamaCache {
    /// Unbounded concatenating caches for ordinary causal attention.
    Standard(Vec<Option<ConcatKeyValueCache>>),
    /// Bounded caches for Mistral-style sliding-window attention.
    Sliding(Vec<Option<SlidingKeyValueCache>>),
}

impl OffloadedLlamaCache {
    /// Returns the common absolute token offset, or zero for an empty cache.
    pub fn offset(&self) -> i32 {
        match self {
            Self::Standard(caches) => caches
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Sliding(caches) => caches
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
        }
    }

    /// Clears retained arrays without changing cache type or window size.
    pub fn clear(&mut self) {
        match self {
            Self::Standard(caches) => caches.iter_mut().flatten().for_each(|cache| cache.clear()),
            Self::Sliding(caches) => caches.iter_mut().flatten().for_each(|cache| cache.clear()),
        }
    }
}

/// Executable Llama/Mistral model whose decoder weights are host-offloaded.
pub struct OffloadedLlamaModel {
    args: ModelArgs,
    store: Arc<SafetensorsWeightStore>,
    residency: ResidencyManager,
    layer_window: DeviceLayerWindow,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    static_leases: Vec<ResidentUnitLease>,
    metadata: OffloadedLlamaMetadata,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

impl OffloadedLlamaModel {
    /// Returns normalized model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns parameter-residency metadata.
    pub const fn metadata(&self) -> &OffloadedLlamaMetadata {
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

    /// Creates the cache type required by this model's attention configuration.
    pub fn new_cache(&self) -> OffloadedLlamaCache {
        match self.args.sliding_window {
            Some(window) => OffloadedLlamaCache::Sliding(
                (0..self.args.num_hidden_layers)
                    .map(|_| Some(SlidingKeyValueCache::new(window)))
                    .collect(),
            ),
            None => OffloadedLlamaCache::Standard(
                (0..self.args.num_hidden_layers)
                    .map(|_| Some(ConcatKeyValueCache::new()))
                    .collect(),
            ),
        }
    }

    /// Creates bounded sliding caches, rejecting full-attention configurations.
    pub fn new_sliding_cache(&self) -> Result<OffloadedLlamaCache, Error> {
        let Some(window) = self.args.sliding_window else {
            return Err(LlamaHostOffloadError::CacheTypeMismatch {
                expected: "standard",
                actual: "sliding",
            }
            .into());
        };
        Ok(OffloadedLlamaCache::Sliding(
            (0..self.args.num_hidden_layers)
                .map(|_| Some(SlidingKeyValueCache::new(window)))
                .collect(),
        ))
    }

    /// Runs embedding, decoder, normalization, and logits projection.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut OffloadedLlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (&mut *cache, self.args.sliding_window) {
            (OffloadedLlamaCache::Standard(caches), None) => self.forward_with_cache(
                ModelInput {
                    inputs,
                    mask: None,
                    cache: caches,
                },
                stream,
            ),
            (OffloadedLlamaCache::Sliding(caches), Some(_)) => self.forward_with_cache(
                ModelInput {
                    inputs,
                    mask: None,
                    cache: caches,
                },
                stream,
            ),
            (OffloadedLlamaCache::Standard(_), Some(_)) => {
                Err(LlamaHostOffloadError::CacheTypeMismatch {
                    expected: "sliding",
                    actual: "standard",
                }
                .into())
            }
            (OffloadedLlamaCache::Sliding(_), None) => {
                Err(LlamaHostOffloadError::CacheTypeMismatch {
                    expected: "standard",
                    actual: "sliding",
                }
                .into())
            }
        }
    }

    /// Runs the model with a caller-selected compatible cache implementation.
    pub fn forward_with_cache<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        C: KeyValueCache + Default,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;
        validate_cache(cache, self.metadata.layer_count)?;

        let mut h = self.embedding.forward(inputs, stream)?;
        let (mask, generated_sliding_window) = match mask {
            Some(mask) => (Some(mask.clone()), None),
            None if self.args.sliding_window.is_some() && h.shape()[1] > 1 => {
                (None, self.args.sliding_window)
            }
            None => (
                attention_mask(&h, cache, self.args.sliding_window, stream)?,
                None,
            ),
        };

        for index in 0..self.metadata.layer_count {
            self.layer_window.prepare(&self.residency, index)?;
            let id = &self.layer_window.units()[index];
            {
                let lease = self.residency.acquire(id, MemoryTier::Device)?;
                let mut layer = TransformerBlock::new_for_layer(&self.args, index as i32, stream)?;
                populate_module_from_lease(&mut layer, &lease)?;
                let layer_cache = cache[index]
                    .as_mut()
                    .ok_or(LlamaHostOffloadError::MissingLayerCache { index })?;
                h = layer.forward(
                    AttentionInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: Some(layer_cache),
                        generated_sliding_window,
                    },
                    stream,
                )?;

                // MLX is lazy. Materialize both the activation and every cache
                // handle updated by this block before its lease can be dropped.
                eval(std::iter::once(&h).chain(layer_cache.retained_arrays()))?;
                stream.synchronize()?;
            }
            let desired = self.layer_window.desired(index)?;
            self.layer_window.trim_to(&self.residency, desired)?;
        }

        let hidden = self.norm.forward(&h, stream)?;
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embedding,
            &hidden,
            stream,
        )?;
        if self.sample_mlx_memory || self.sample_process_memory {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        Ok(logits)
    }

    /// Runs prompt prefill and returns last-token logits.
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut OffloadedLlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(inputs, cache, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    /// Runs cached autoregressive decode and returns last-token logits.
    pub fn decode(
        &mut self,
        input_tokens: &Array,
        cache: &mut OffloadedLlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prefill(input_tokens, cache, stream)
    }

    /// Explicitly evicts all decoder copies from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        Ok(self.layer_window.clear(&self.residency)?)
    }

    /// Returns the number of long-lived pinned static leases.
    pub fn static_lease_count(&self) -> usize {
        self.static_leases.len()
    }
}

impl CausalLm<OffloadedLlamaCache> for OffloadedLlamaModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut OffloadedLlamaCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.prefill(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut OffloadedLlamaCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

/// Loads an explicit Llama/Mistral layerwise host-offloaded safetensors model.
pub fn load_llama_host_offloaded_model(
    model_dir: impl AsRef<Path>,
    offload: OffloadConfig,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<OffloadedLlamaModel, Error> {
    load_llama_host_offloaded_model_with_options(
        model_dir,
        LlamaHostOffloadOptions::new(offload),
        stream,
        weights_stream,
    )
}

/// Loads with explicit mapping, strictness, and diagnostics controls.
pub fn load_llama_host_offloaded_model_with_options(
    model_dir: impl AsRef<Path>,
    options: LlamaHostOffloadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<OffloadedLlamaModel, Error> {
    let model_dir = model_dir.as_ref();
    if model_dir.extension().and_then(|value| value.to_str()) == Some("gguf") {
        return Err(LlamaHostOffloadError::GgufUnsupported.into());
    }
    let args = llama::get_llama_model_args(model_dir)?;
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(|_| {
        LlamaHostOffloadError::ArithmeticOverflow {
            context: "decoder layer count",
        }
    })?;
    let depth = options.offload.prefetch_depth();
    if depth > layer_count {
        return Err(LlamaHostOffloadError::InvalidLayerWindow { depth, layer_count }.into());
    }
    let store = Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(
        model_dir,
        options.max_mapped_shards,
    )?);

    let mut embedding = unloaded_maybe_quantized_embedding(
        args.vocab_size,
        args.hidden_size,
        args.affine_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    let mut norm =
        nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
    let mut lm_head = if args.tie_word_embeddings {
        None
    } else {
        Some(build_unloaded_maybe_quantized_lm_head_with_quantization(
            args.hidden_size,
            args.vocab_size,
            args.affine_quantization_for("lm_head.weight"),
            stream,
        )?)
    };

    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut static_device_bytes = 0u64;

    add_unit(
        &mut definitions,
        &mut specs,
        &mut consumed,
        EMBEDDING_UNIT,
        build_module_bindings(&embedding, "model.embed_tokens", store.as_ref())?,
        ResidencyPolicy::Pinned,
        MemoryTier::Device,
        &mut static_device_bytes,
    )?;
    add_unit(
        &mut definitions,
        &mut specs,
        &mut consumed,
        NORM_UNIT,
        build_module_bindings(&norm, "model.norm", store.as_ref())?,
        ResidencyPolicy::Pinned,
        MemoryTier::Device,
        &mut static_device_bytes,
    )?;
    if let Some(head) = &lm_head {
        add_unit(
            &mut definitions,
            &mut specs,
            &mut consumed,
            HEAD_UNIT,
            build_module_bindings(head, "lm_head", store.as_ref())?,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            &mut static_device_bytes,
        )?;
    }

    let mut layer_ids = Vec::with_capacity(layer_count);
    let mut layer_bytes = Vec::with_capacity(layer_count);
    let mut host_layer_bytes = 0u64;
    for index in 0..layer_count {
        let layer = TransformerBlock::new_for_layer(&args, index as i32, stream)?;
        let bindings =
            build_module_bindings(&layer, &format!("model.layers.{index}"), store.as_ref())?;
        let bytes = binding_bytes(&bindings)?;
        host_layer_bytes = host_layer_bytes.checked_add(bytes).ok_or(
            LlamaHostOffloadError::ArithmeticOverflow {
                context: "host decoder byte total",
            },
        )?;
        let id = OffloadUnitId::new(format!("llama.layer.{index:05}"))?;
        consumed.extend(
            bindings
                .iter()
                .map(|binding| binding.checkpoint_key().to_string()),
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

    validate_unused(store.as_ref(), &consumed, options.strict_loading)?;
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

    let mut static_leases = Vec::with_capacity(if lm_head.is_some() { 3 } else { 2 });
    let embedding_lease = acquire_static(&residency, EMBEDDING_UNIT)?;
    populate_module_from_lease(&mut embedding, &embedding_lease)?;
    static_leases.push(embedding_lease);
    let norm_lease = acquire_static(&residency, NORM_UNIT)?;
    populate_module_from_lease(&mut norm, &norm_lease)?;
    static_leases.push(norm_lease);
    if let Some(head) = &mut lm_head {
        let head_lease = acquire_static(&residency, HEAD_UNIT)?;
        populate_module_from_lease(head, &head_lease)?;
        static_leases.push(head_lease);
    }

    let layer_window = DeviceLayerWindow::new(layer_ids, depth)?;
    let metadata = OffloadedLlamaMetadata {
        model_type: args.model_type.clone(),
        quantization: args.weight_quantization(),
        layer_count,
        static_device_bytes,
        host_layer_bytes,
        maximum_window_bytes,
        device_layer_window: depth,
    };
    Ok(OffloadedLlamaModel {
        args,
        store,
        residency,
        layer_window,
        embedding,
        norm,
        lm_head,
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
    name: &str,
    bindings: Vec<crate::residency::WeightBinding>,
    policy: ResidencyPolicy,
    tier: MemoryTier,
    byte_total: &mut u64,
) -> Result<(), Error> {
    let bytes = binding_bytes(&bindings)?;
    *byte_total =
        byte_total
            .checked_add(bytes)
            .ok_or(LlamaHostOffloadError::ArithmeticOverflow {
                context: "static device byte total",
            })?;
    consumed.extend(
        bindings
            .iter()
            .map(|binding| binding.checkpoint_key().to_string()),
    );
    let id = OffloadUnitId::new(name)?;
    definitions.push(OffloadUnit::new(id.clone(), bindings)?);
    specs.push(OffloadUnitSpec::new(id, bytes, policy, tier)?);
    Ok(())
}

fn acquire_static(residency: &ResidencyManager, name: &str) -> Result<ResidentUnitLease, Error> {
    Ok(residency.acquire(&OffloadUnitId::new(name)?, MemoryTier::Device)?)
}

fn validate_unused(
    store: &dyn WeightStore,
    consumed: &BTreeSet<String>,
    strict: bool,
) -> Result<(), Error> {
    if !strict {
        return Ok(());
    }
    let unused = store
        .keys()
        .into_iter()
        .filter(|key| !consumed.contains(key))
        .filter(|key| !key.starts_with("rope_freqs.") && !key.ends_with(".rotary_emb.inv_freq"))
        .collect::<Vec<_>>();
    if unused.is_empty() {
        Ok(())
    } else {
        Err(LlamaHostOffloadError::UnexpectedCheckpointParameters { unused }.into())
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
        return Err(LlamaHostOffloadError::CacheLengthMismatch {
            expected: layer_count,
            actual: cache.len(),
        }
        .into());
    }
    if let Some(index) = cache.iter().position(Option::is_none) {
        return Err(LlamaHostOffloadError::MissingLayerCache { index }.into());
    }
    Ok(())
}

fn attention_mask<C: KeyValueCache>(
    h: &Array,
    cache: &[Option<C>],
    sliding_window: Option<i32>,
    stream: &Stream,
) -> Result<Option<Array>, Error> {
    if let Some(window) = sliding_window {
        return Ok(create_sliding_attention_mask(h, cache, window, stream)?);
    }
    match create_attention_mask(h, cache, Some(true), stream)? {
        Some(AttentionMask::Array(mask)) => Ok(Some(mask)),
        Some(AttentionMask::Causal) => Err(Exception::custom(
            "Llama-compatible decoders require an explicit attention mask",
        )
        .into()),
        None => Ok(None),
    }
}

fn largest_window_bytes(layer_bytes: &[u64], depth: usize) -> Result<u64, Error> {
    let mut largest = 0u64;
    for start in 0..layer_bytes.len() {
        let mut current = 0u64;
        for bytes in layer_bytes.iter().skip(start).take(depth) {
            current =
                current
                    .checked_add(*bytes)
                    .ok_or(LlamaHostOffloadError::ArithmeticOverflow {
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
            return Err(LlamaHostOffloadError::HostBudgetTooSmall { required, budget }.into());
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
    let required = static_bytes.checked_add(window_bytes).ok_or(
        LlamaHostOffloadError::ArithmeticOverflow {
            context: "static plus device-window byte total",
        },
    )?;
    if let Some(budget) = config.device_budget_bytes() {
        if required > budget {
            return Err(LlamaHostOffloadError::DeviceBudgetTooSmall {
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

/// Structured failures specific to the explicit Llama host-offload adapter.
#[derive(Debug, thiserror::Error)]
pub enum LlamaHostOffloadError {
    /// GGUF is intentionally outside this loader's safetensors contract.
    #[error("Llama host offload requires safetensors; GGUF offload is unsupported")]
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
    #[error(
        "strict Llama host-offload loading found unexpected checkpoint parameters: {unused:?}"
    )]
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
    #[error("Llama cache has {actual} layers, expected {expected}")]
    CacheLengthMismatch {
        /// Model decoder count.
        expected: usize,
        /// Supplied cache count.
        actual: usize,
    },
    /// A cache entry was absent.
    #[error("Llama cache entry {index} is missing")]
    MissingLayerCache {
        /// Missing decoder index.
        index: usize,
    },
    /// The cache implementation did not match the model attention mode.
    #[error("cache type mismatch: model requires {expected}, supplied {actual}")]
    CacheTypeMismatch {
        /// Required cache kind.
        expected: &'static str,
        /// Supplied cache kind.
        actual: &'static str,
    },
    /// Checked byte or index arithmetic overflowed.
    #[error("Llama host-offload arithmetic overflow: {context}")]
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
        module::{Module, ModuleParameters},
        ops::ones_dtype,
        Device, DeviceType, ExecutionContext,
    };

    use super::*;
    use crate::{models::llama, offload::TransferDirection, residency::UnitResidencyReport};

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

    fn write_fixture(dir: &Path, model: &llama::Model) {
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
            llama::Model::new(args(model_type, tied, sliding_window), stream).unwrap();
        initialize(&mut reference, stream);
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let config = OffloadConfig::new(None, None, depth).unwrap();
        let mut offloaded =
            load_llama_host_offloaded_model(dir.path(), config, stream, cpu.stream()).unwrap();
        let initial = offloaded.residency_report().unwrap();
        assert!(layer_reports(&initial)
            .iter()
            .all(|unit| unit.host_resident()));
        assert!(layer_reports(&initial)
            .iter()
            .all(|unit| !unit.device_resident()));

        let mut reference_standard = Vec::<Option<ConcatKeyValueCache>>::new();
        let mut reference_sliding = sliding_window.map(|window| {
            (0..3)
                .map(|_| Some(SlidingKeyValueCache::new(window)))
                .collect::<Vec<_>>()
        });
        let mut cache = offloaded.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = if let Some(caches) = &mut reference_sliding {
                reference
                    .forward(
                        llama::ModelInput {
                            inputs: &tokens,
                            mask: None,
                            cache: caches,
                        },
                        stream,
                    )
                    .unwrap()
            } else {
                reference
                    .forward(
                        llama::ModelInput {
                            inputs: &tokens,
                            mask: None,
                            cache: &mut reference_standard,
                        },
                        stream,
                    )
                    .unwrap()
            };
            let actual = offloaded.forward(&tokens, &mut cache, stream).unwrap();
            assert_close(&actual, &expected);
            let report = offloaded.residency_report().unwrap();
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

        let report = offloaded.residency_report().unwrap();
        assert!(
            report
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .count()
                >= 3
        );
        assert_eq!(offloaded.static_lease_count(), if tied { 2 } else { 3 });
        offloaded.clear_device_layer_window().unwrap();
        let cleared = offloaded.residency_report().unwrap();
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
    fn host_offload_dense_prefill_decode_parity() {
        run_parity("llama", true, None, 1);
        run_parity("llama", false, None, 2);
        run_parity("mistral", false, Some(4), 2);
    }

    #[test]
    fn budget_and_cache_validation_are_structured() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut reference = llama::Model::new(args("llama", true, None), gpu.stream()).unwrap();
        initialize(&mut reference, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &reference);

        let host_error = load_llama_host_offloaded_model(
            dir.path(),
            OffloadConfig::new(None, Some(1), 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .err()
        .unwrap();
        assert!(host_error.to_string().contains("host budget"));

        let device_error = load_llama_host_offloaded_model(
            dir.path(),
            OffloadConfig::new(Some(1), None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .err()
        .unwrap();
        assert!(device_error.to_string().contains("device budget"));

        let mut model = load_llama_host_offloaded_model(
            dir.path(),
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut bad_cache = OffloadedLlamaCache::Standard(vec![None]);
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
    fn host_offload_packed_affine_parity() {
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
        let mut dense = llama::Model::new(quant_args, gpu.stream()).unwrap();
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

        let mut resident = llama::load_llama_model(&converted, gpu.stream(), cpu.stream()).unwrap();
        let mut offloaded = load_llama_host_offloaded_model(
            &converted,
            OffloadConfig::new(None, None, 1).unwrap(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        assert!(offloaded.metadata().quantization().is_some());
        let mut resident_cache = Vec::<Option<ConcatKeyValueCache>>::new();
        let mut offloaded_cache = offloaded.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    llama::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut resident_cache,
                    },
                    gpu.stream(),
                )
                .unwrap();
            let actual = offloaded
                .forward(&tokens, &mut offloaded_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
    }
}
