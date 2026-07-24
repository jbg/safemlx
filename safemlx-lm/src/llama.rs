//! Unified Llama/Mistral loading across weight-residency policies.

use std::{collections::HashMap, path::Path, sync::Arc};

use safemlx::{
    error::Exception,
    module::Module,
    nn,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache},
    cache_residency::{
        open_prompt_cache, validate_prompt_cache_model_identity, CacheResidencyManager,
        CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    },
    error::Error,
    layerwise::{
        load_layerwise_model, load_layerwise_model_with_store, DenseDiskStreamReport,
        LayerwiseInput, LayerwiseLoadOptions, LayerwiseModel, LayerwiseModelAdapter,
        LayerwiseModelMetadata, StaticUnitBindings, WeightResidency,
    },
    models::{
        common::{
            generation::CausalLm,
            linear::{
                build_unloaded_maybe_quantized_lm_head_with_quantization,
                project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            },
        },
        input,
        llama::{self as resident, AttentionInput, ModelArgs, TransformerBlock},
    },
    module_binding::{build_module_bindings, populate_module_from_lease},
    residency::{ResidencyReport, ResidentUnitLease},
    utils::{create_attention_mask, create_sliding_attention_mask, AttentionMask},
    weight_store::{GgufWeightStore, WeightStore},
};

const EMBEDDING_UNIT: &str = "llama.static.embedding";
const NORM_UNIT: &str = "llama.static.norm";
const HEAD_UNIT: &str = "llama.static.output";

/// Options for the unified Llama/Mistral loader.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub struct LlamaLoadOptions {
    /// Determines where decoder weights live and how they execute.
    pub weight_residency: WeightResidency,
}

impl LlamaLoadOptions {
    /// Selects the eager engine with every parameter on the execution device.
    pub const fn fully_resident() -> Self {
        Self {
            weight_residency: WeightResidency::FullyResident,
        }
    }

    /// Selects host-backed decoder layers with a bounded device window.
    pub const fn layerwise_host(options: LayerwiseLoadOptions) -> Self {
        Self {
            weight_residency: WeightResidency::LayerwiseHost(options),
        }
    }

    /// Selects experimental dense disk streaming with finite tier budgets.
    pub const fn dense_disk_stream(
        options: crate::dense_stream::DenseDiskStreamLoadOptions,
    ) -> Self {
        Self {
            weight_residency: WeightResidency::DenseDiskStream(options),
        }
    }
}

/// Standard or sliding per-layer KV cache selected from Llama configuration.
#[derive(Debug, Clone)]
pub enum LlamaCache {
    /// Unbounded concatenating caches for ordinary causal attention.
    Standard(Vec<Option<ConcatKeyValueCache>>),
    /// Bounded caches for Mistral-style sliding-window attention.
    Sliding(Vec<Option<SlidingKeyValueCache>>),
    /// Block-addressable caches sharing one model-wide residency manager.
    Paged(Vec<Option<PagedKeyValueCache>>),
}

impl LlamaCache {
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
            Self::Paged(caches) => caches
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
        }
    }

    /// Clears retained arrays without changing cache type or window size.
    pub fn clear(&mut self) -> Result<(), Error> {
        match self {
            Self::Standard(caches) => caches.iter_mut().flatten().for_each(|cache| cache.clear()),
            Self::Sliding(caches) => caches.iter_mut().flatten().for_each(|cache| cache.clear()),
            Self::Paged(caches) => {
                for cache in caches.iter_mut().flatten() {
                    cache.clear()?;
                }
            }
        }
        Ok(())
    }

    /// Returns aggregate cache-residency telemetry for a paged cache.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Error> {
        match self {
            Self::Paged(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(|cache| cache.report().map_err(Into::into))
                .transpose(),
            Self::Standard(_) | Self::Sliding(_) => Ok(None),
        }
    }

    /// Finalizes every mutable tail and atomically persists a completed text prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let Self::Paged(caches) = self else {
            return Err(Exception::custom(
                "prompt-cache persistence requires an explicitly configured paged cache",
            )
            .into());
        };
        for cache in caches.iter_mut().flatten() {
            cache.finalize()?;
        }
        let manager = caches
            .iter()
            .flatten()
            .next()
            .ok_or_else(|| Exception::custom("cannot persist an empty paged cache"))?
            .manager()
            .clone();
        manager
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(|error| Exception::custom(error.to_string()).into())
    }
}

enum LlamaExecution {
    FullyResident(Box<resident::ResidentModel>),
    LayerwiseHost(Box<LayerwiseModel<LlamaLayerwiseAdapter>>),
}

/// Llama/Mistral causal LM whose execution engine follows its residency policy.
pub struct LlamaModel {
    execution: LlamaExecution,
}

impl LlamaModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        match &self.execution {
            LlamaExecution::FullyResident(model) => &model.args,
            LlamaExecution::LayerwiseHost(model) => model.adapter().args(),
        }
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        crate::models::llama::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns whether all parameters use the eager execution-device engine.
    pub const fn is_fully_resident(&self) -> bool {
        matches!(&self.execution, LlamaExecution::FullyResident(_))
    }

    /// Returns layerwise parameter metadata when that engine is selected.
    pub fn layerwise_metadata(&self) -> Option<&LayerwiseModelMetadata> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => None,
            LlamaExecution::LayerwiseHost(model) => Some(model.metadata()),
        }
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => Ok(None),
            LlamaExecution::LayerwiseHost(model) => Ok(Some(model.residency_report()?)),
        }
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => Ok(None),
            LlamaExecution::LayerwiseHost(model) => model.dense_stream_report(),
        }
    }

    /// Returns the persistent checkpoint store used by a layerwise model.
    pub fn checkpoint_store(&self) -> Option<&(dyn WeightStore + Send + Sync)> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => None,
            LlamaExecution::LayerwiseHost(model) => Some(model.checkpoint_store()),
        }
    }

    /// Backward-compatible alias for [`Self::checkpoint_store`].
    pub fn weight_store(&self) -> Option<&(dyn WeightStore + Send + Sync)> {
        self.checkpoint_store()
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn layerwise_static_lease_count(&self) -> Option<usize> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => None,
            LlamaExecution::LayerwiseHost(model) => Some(model.static_lease_count()),
        }
    }

    /// Creates the cache representation required by the model configuration.
    pub fn new_cache(&self) -> LlamaCache {
        let args = self.args();
        match args.sliding_window {
            Some(window) => LlamaCache::Sliding(
                (0..args.num_hidden_layers)
                    .map(|_| Some(SlidingKeyValueCache::new(window)))
                    .collect(),
            ),
            None => LlamaCache::Standard(
                (0..args.num_hidden_layers)
                    .map(|_| Some(ConcatKeyValueCache::new()))
                    .collect(),
            ),
        }
    }

    /// Creates a device-resident or explicitly bounded paged model cache.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<LlamaCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => self.new_paged_cache(options, None),
        }
    }

    /// Catalogs a compatible reusable prefix without loading all cache blocks.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(LlamaCache, PromptCacheManifest), Error> {
        let args = self.args();
        let layer_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Llama cache layer count"))?;
        let identity = PromptCacheModelIdentity {
            model_family: "llama".into(),
            effective_model_type: args.model_type.clone(),
            architecture_fingerprint: crate::models::llama::prompt_cache_architecture_fingerprint(
                args,
            ),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sliding_window: args.sliding_window,
            sink_tokens: 0,
            topology: Default::default(),
            layer_layouts: PromptCacheModelIdentity::key_value_layouts(
                layer_count,
                args.num_key_value_heads,
                args.head_dim,
            ),
        };
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let (manager, manifest) =
            open_prompt_cache(directory, expected, &identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let cache = self.new_paged_cache_from_manager(manager)?;
        Ok((cache, manifest))
    }

    fn new_paged_cache(
        &self,
        options: PagedCacheOptions,
        manager: Option<CacheResidencyManager>,
    ) -> Result<LlamaCache, Error> {
        let manager = match manager {
            Some(manager) => manager,
            None => CacheResidencyManager::new(options)
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        self.new_paged_cache_from_manager(manager)
    }

    fn new_paged_cache_from_manager(
        &self,
        manager: CacheResidencyManager,
    ) -> Result<LlamaCache, Error> {
        let args = self.args();
        let layer_count = usize::try_from(args.num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: args.num_hidden_layers,
            }
        })?;
        let caches = (0..layer_count)
            .map(|layer| {
                PagedKeyValueCache::new(manager.clone(), layer, args.sliding_window).map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LlamaCache::Paged(caches))
    }

    /// Runs embedding, decoder layers, final normalization, and projection.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        match (&mut self.execution, cache) {
            (LlamaExecution::FullyResident(model), LlamaCache::Standard(caches)) => Ok(model
                .forward(
                    resident::ModelInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                )?),
            (LlamaExecution::FullyResident(model), LlamaCache::Sliding(caches)) => Ok(model
                .forward(
                    resident::ModelInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                )?),
            (LlamaExecution::LayerwiseHost(model), LlamaCache::Standard(caches)) => model
                .forward_with_cache(
                    LayerwiseInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                ),
            (LlamaExecution::LayerwiseHost(model), LlamaCache::Sliding(caches)) => model
                .forward_with_cache(
                    LayerwiseInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                ),
            (LlamaExecution::FullyResident(model), LlamaCache::Paged(caches)) => Ok(model
                .forward(
                    resident::ModelInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                )?),
            (LlamaExecution::LayerwiseHost(model), LlamaCache::Paged(caches)) => model
                .forward_with_cache(
                    LayerwiseInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                ),
        }
    }

    /// Runs prompt prefill and returns last-token logits.
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(inputs, cache, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    /// Runs cached decode and returns last-token logits.
    pub fn decode(
        &mut self,
        input_tokens: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prefill(input_tokens, cache, stream)
    }

    /// Clears temporary execution-device decoder copies when layerwise residency is active.
    ///
    /// Returns `true` when a layerwise window was cleared and `false` for the
    /// fully resident engine.
    pub fn clear_device_layer_window(&self) -> Result<bool, Error> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => Ok(false),
            LlamaExecution::LayerwiseHost(model) => {
                model.clear_device_layer_window()?;
                Ok(true)
            }
        }
    }

    fn validate_cache(&self, cache: &LlamaCache) -> Result<(), Error> {
        let expected_layers = usize::try_from(self.args().num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: self.args().num_hidden_layers,
            }
        })?;
        let (kind, actual_layers) = match cache {
            LlamaCache::Standard(caches) => ("standard", caches.len()),
            LlamaCache::Sliding(caches) => ("sliding", caches.len()),
            LlamaCache::Paged(caches) => ("paged", caches.len()),
        };
        let expected_kind = if self.args().sliding_window.is_some() {
            "sliding"
        } else {
            "standard"
        };
        if kind != "paged" && kind != expected_kind {
            return Err(LlamaModelError::CacheTypeMismatch {
                expected: expected_kind,
                actual: kind,
            }
            .into());
        }
        if actual_layers != expected_layers {
            return Err(LlamaModelError::CacheLengthMismatch {
                expected: expected_layers,
                actual: actual_layers,
            }
            .into());
        }
        Ok(())
    }
}

impl CausalLm<LlamaCache> for LlamaModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.prefill(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

/// Loads a Llama/Mistral safetensors model using the selected residency policy.
pub fn load_llama_model(
    model_dir: impl AsRef<Path>,
    options: LlamaLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
    let execution = match options.weight_residency {
        WeightResidency::FullyResident => LlamaExecution::FullyResident(Box::new(
            resident::load_resident_llama_model(model_dir, stream, weights_stream)?,
        )),
        WeightResidency::LayerwiseHost(options) => {
            let model_dir = model_dir.as_ref();
            let args = resident::get_llama_model_args(model_dir)?;
            let adapter = LlamaLayerwiseAdapter::new(args, stream)?;
            LlamaExecution::LayerwiseHost(Box::new(load_layerwise_model(
                model_dir,
                adapter,
                options,
                stream,
                weights_stream,
            )?))
        }
        WeightResidency::DenseDiskStream(options) => {
            let model_dir = model_dir.as_ref();
            let args = resident::get_llama_model_args(model_dir)?;
            let adapter = LlamaLayerwiseAdapter::new(args, stream)?;
            LlamaExecution::LayerwiseHost(Box::new(load_layerwise_model(
                model_dir,
                adapter,
                options,
                stream,
                weights_stream,
            )?))
        }
        WeightResidency::SparseExpertCache(_)
        | WeightResidency::SparseExpertCacheWithDenseLayers(_) => {
            return Err(Error::UnsupportedArchitecture(
                "sparse expert caching is not supported for Llama checkpoints".into(),
            ));
        }
    };
    Ok(LlamaModel { execution })
}

/// Loads a Llama/Mistral GGUF checkpoint using the selected residency policy.
pub(crate) fn load_llama_gguf_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LlamaModel, Vec<u32>), Error> {
    let prepared =
        resident::prepare_llama_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let adapter = LlamaLayerwiseAdapter::new(prepared.args, stream)?;
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => LlamaExecution::LayerwiseHost(Box::new(
            load_layerwise_model_with_store(store, adapter, options, stream, weights_stream)?,
        )),
        WeightResidency::DenseDiskStream(options) => LlamaExecution::LayerwiseHost(Box::new(
            load_layerwise_model_with_store(store, adapter, options, stream, weights_stream)?,
        )),
        WeightResidency::SparseExpertCache(_)
        | WeightResidency::SparseExpertCacheWithDenseLayers(_) => {
            return Err(Error::UnsupportedArchitecture(
                "sparse expert caching is not supported for Llama GGUF checkpoints".into(),
            ));
        }
        WeightResidency::FullyResident => {
            return Err(Error::UnsupportedArchitecture(
                "the bounded GGUF Llama loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((LlamaModel { execution }, prepared.eos_token_ids))
}

/// Llama implementation of the generic layerwise model-family contract.
pub struct LlamaLayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl LlamaLayerwiseAdapter {
    /// Creates metadata-only static modules for a normalized Llama configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let embedding = unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.affine_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(build_unloaded_maybe_quantized_lm_head_with_quantization(
                args.hidden_size,
                args.vocab_size,
                args.affine_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
            args,
            embedding,
            norm,
            lm_head,
        })
    }

    /// Returns normalized Llama arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }
}

/// Llama mask state shared by every temporary decoder block.
pub struct LlamaForwardContext {
    mask: Option<Array>,
    generated_sliding_window: Option<i32>,
}

impl LayerwiseModelAdapter for LlamaLayerwiseAdapter {
    type Layer = TransformerBlock;
    type ForwardContext = LlamaForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn quantization(&self) -> Option<crate::quantization::WeightQuantization> {
        self.args.weight_quantization()
    }

    fn layer_count(&self) -> Result<usize, Error> {
        usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: self.args.num_hidden_layers,
            }
            .into()
        })
    }

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings(&self.embedding, "model.embed_tokens", store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings(&self.norm, "model.norm", store)?,
            )?,
        ];
        if let Some(head) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings(head, "lm_head", store)?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Exception::custom(format!(
                "Llama adapter received {} static leases, expected {expected}",
                leases.len()
            ))
            .into());
        }
        populate_module_from_lease(&mut self.embedding, &leases[0])?;
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[2])?;
        }
        Ok(())
    }

    fn new_layer(&self, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        let index =
            i32::try_from(index).map_err(|_| LlamaModelError::LayerIndexOverflow { index })?;
        Ok(TransformerBlock::new_for_layer(&self.args, index, stream)?)
    }

    fn layer_checkpoint_prefix(&self, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, index: usize) -> String {
        format!("llama.layer.{index:05}")
    }

    fn embed(&mut self, inputs: &Array, stream: &Stream) -> Result<Array, Error> {
        Ok(self.embedding.forward(inputs, stream)?)
    }

    fn prepare_forward<C: KeyValueCache>(
        &self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &[Option<C>],
        stream: &Stream,
    ) -> Result<Self::ForwardContext, Error> {
        let (mask, generated_sliding_window) = match mask {
            Some(mask) => (Some(mask.clone()), None),
            None if self.args.sliding_window.is_some() && hidden.shape()[1] > 1 => {
                (None, self.args.sliding_window)
            }
            None => (
                llama_attention_mask(hidden, cache, self.args.sliding_window, stream)?,
                None,
            ),
        };
        Ok(LlamaForwardContext {
            mask,
            generated_sliding_window,
        })
    }

    fn forward_layer<C: KeyValueCache>(
        &self,
        _index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut C,
        context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(layer.forward(
            AttentionInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(cache),
                generated_sliding_window: context.generated_sliding_window,
            },
            stream,
        )?)
    }

    fn finish(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embedding,
            &hidden,
            stream,
        )?)
    }

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        key.starts_with("rope_freqs.") || key.ends_with(".rotary_emb.inv_freq")
    }
}

fn llama_attention_mask<C: KeyValueCache>(
    hidden: &Array,
    cache: &[Option<C>],
    sliding_window: Option<i32>,
    stream: &Stream,
) -> Result<Option<Array>, Error> {
    if let Some(window) = sliding_window {
        return Ok(create_sliding_attention_mask(
            hidden, cache, window, stream,
        )?);
    }
    match create_attention_mask(hidden, cache, Some(true), stream)? {
        Some(AttentionMask::Array(mask)) => Ok(Some(mask)),
        Some(AttentionMask::Causal) => Err(Exception::custom(
            "Llama-compatible decoders require an explicit attention mask",
        )
        .into()),
        None => Ok(None),
    }
}

/// Structured failures at the unified Llama model boundary.
#[derive(Debug, thiserror::Error)]
pub enum LlamaModelError {
    /// The normalized decoder count cannot be represented by this runtime.
    #[error("invalid Llama decoder layer count {count}")]
    InvalidLayerCount {
        /// Invalid configured count.
        count: i32,
    },
    /// A decoder index cannot be represented by the model implementation.
    #[error("Llama decoder index {index} exceeds the supported range")]
    LayerIndexOverflow {
        /// Invalid decoder index.
        index: usize,
    },
    /// A cache vector had the wrong number of layers.
    #[error("Llama cache has {actual} layers, expected {expected}")]
    CacheLengthMismatch {
        /// Model decoder count.
        expected: usize,
        /// Supplied cache count.
        actual: usize,
    },
    /// The cache implementation did not match the model attention mode.
    #[error("cache type mismatch: model requires {expected}, supplied {actual}")]
    CacheTypeMismatch {
        /// Required cache kind.
        expected: &'static str,
        /// Supplied cache kind.
        actual: &'static str,
    },
}
