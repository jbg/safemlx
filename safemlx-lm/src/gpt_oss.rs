//! Unified fully resident and layerwise-host GPT-OSS execution.

use std::{path::Path, time::Instant};

use safemlx::{
    error::Exception,
    module::{Module, Param},
    nn,
    ops::indexing::TryIndexOp,
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    cache::KeyValueCache,
    error::Error,
    expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry, ExpertIdentity,
        ExpertPass,
    },
    layerwise::{
        load_general_layerwise_model, GeneralLayerwiseModel, GeneralLayerwiseModelAdapter,
        LayerwiseForwardState, LayerwiseLoadOptions, StaticUnitBindings,
    },
    models::{
        common::{self, generation::CausalLm},
        gpt_oss::{self as resident, Cache, Experts, LayerCache, ModelArgs, TransformerBlock},
        input,
    },
    module_binding::{
        build_module_bindings, populate_module_from_lease, populate_module_from_lease_excluding,
    },
    residency::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
    utils::create_causal_mask,
    weight_recipe::DerivedWeightRecipe,
    weight_store::{SafetensorsWeightStore, TensorSelection, WeightStore},
};

const EMBEDDING_UNIT: &str = "gpt_oss.static.embedding";
const NORM_UNIT: &str = "gpt_oss.static.norm";
const HEAD_UNIT: &str = "gpt_oss.static.output";

/// GPT-OSS causal LM using bounded host residency for complete decoder blocks.
pub struct GptOssLayerwiseModel {
    execution: GeneralLayerwiseModel<GptOssLayerwiseAdapter>,
}

impl GptOssLayerwiseModel {
    /// Returns the validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates the architecture's alternating sliding/full attention cache.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Returns sparse expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the persistent checkpoint store.
    pub fn weight_store(&self) -> &SafetensorsWeightStore {
        self.execution.weight_store()
    }

    /// Runs GPT-OSS while preserving its heterogeneous cache schedule.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }

    /// Clears temporary decoder copies from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalLm<Cache> for GptOssLayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads GPT-OSS through the generalized bounded host-residency engine.
pub fn load_gpt_oss_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: LayerwiseLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    let adapter = GptOssLayerwiseAdapter::new(args, stream)?;
    Ok(GptOssLayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads GPT-OSS with expert-granular sparse caching.
pub fn load_gpt_oss_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    let mut adapter = GptOssLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution = load_general_layerwise_model(
        model_dir,
        adapter,
        options.non_expert,
        stream,
        weights_stream,
    )?;
    let store = execution.weight_store_arc();
    let entries = gpt_oss_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(GptOssLayerwiseModel { execution })
}

/// Generalized adapter for GPT-OSS native MXFP4 sparse decoder blocks.
pub struct GptOssLayerwiseAdapter {
    args: ModelArgs,
    layer_types: Vec<String>,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl GptOssLayerwiseAdapter {
    /// Creates metadata-only pinned modules for a validated configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        args.validate()?;
        let layer_types = args.effective_layer_types();
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.quantization,
            stream,
        )?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let lm_head = common::linear::unloaded_maybe_quantized_linear(
            args.hidden_size,
            args.vocab_size,
            false,
            args.quantization,
            stream,
        )?;
        Ok(Self {
            args,
            layer_types,
            embedding,
            norm,
            lm_head,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    /// Returns the validated model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache {
            layers: self
                .layer_types
                .iter()
                .map(|kind| {
                    if kind == "sliding_attention" {
                        LayerCache::Sliding(crate::cache::SlidingKeyValueCache::new(
                            self.args.sliding_window,
                        ))
                    } else {
                        LayerCache::Full(crate::cache::ConcatKeyValueCache::new())
                    }
                })
                .collect(),
        }
    }
}

/// GPT-OSS state shared across temporary decoder blocks.
pub struct GptOssForwardContext {
    sequence_length: i32,
}

impl GeneralLayerwiseModelAdapter for GptOssLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = TransformerBlock;
    type ForwardContext = GptOssForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        Ok(vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings(&self.embedding, "model.embed_tokens", store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings(&self.norm, "model.norm", store)?,
            )?,
            StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings(&self.lm_head, "lm_head", store)?,
            )?,
        ])
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        if leases.len() != 3 {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS adapter received {} static leases, expected 3",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.embedding, &leases[0])?;
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        populate_module_from_lease(&mut self.lm_head, &leases[2])?;
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.layer_types.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS cache has {} layers, expected {}",
                cache.layers.len(),
                self.layer_types.len()
            )));
        }
        for (index, (cache, kind)) in cache.layers.iter().zip(&self.layer_types).enumerate() {
            let matches = matches!(
                (cache, kind.as_str()),
                (LayerCache::Sliding(_), "sliding_attention")
                    | (LayerCache::Full(_), "full_attention")
            );
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GPT-OSS cache kind does not match layer_types at layer {index}"
                )));
            }
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input, stream)?;
        Ok(LayerwiseForwardState {
            context: GptOssForwardContext {
                sequence_length: hidden.dim(1),
            },
            hidden,
        })
    }

    fn execution_group_count(&self) -> usize {
        1
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        if group == 0 {
            Ok("text_decoder".into())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS has no execution group {group}"
            )))
        }
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.layer_types.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS has no execution group {group}"
            )))
        }
    }

    fn new_layer(
        &self,
        group: usize,
        _index: usize,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        Ok(TransformerBlock::new(&self.args, stream)?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if self.sparse_expert_cache {
            Ok(populate_module_from_lease_excluding(
                layer,
                lease,
                |name| name.starts_with("mlp.experts."),
            )?)
        } else {
            Ok(populate_module_from_lease(layer, lease)?)
        }
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings = build_module_bindings(layer, &format!("model.layers.{index}"), store)?;
        Ok(if self.sparse_expert_cache {
            bindings
                .into_iter()
                .filter(|binding| !binding.name().starts_with("mlp.experts."))
                .collect()
        } else {
            bindings
        })
    }

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
                .into_iter()
                .filter(|key| key.contains(".mlp.experts."))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("gpt_oss.layer.{index:05}")
    }

    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.layer_count(group)?;
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let window = layer_cache.max_size();
        let needs_mask = context.sequence_length > 1 || window.is_some_and(|size| offset >= size);
        let mask = needs_mask
            .then(|| {
                create_causal_mask(
                    context.sequence_length,
                    Some(offset.min(window.unwrap_or(offset))),
                    window.map(|size| size - 1),
                    None,
                    stream,
                )
            })
            .transpose()?;
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "GPT-OSS sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_with_expert_executor(
                hidden,
                mask.as_ref(),
                layer_cache,
                stream,
                |flat, indices, weights, stream| {
                    let acquired = expert_cache
                        .acquire_routes(index, indices, pass, stream)
                        .map_err(|error| Exception::custom(error.to_string()))?;
                    let started = Instant::now();
                    let mut compact_args = self.args.clone();
                    compact_args.num_local_experts = acquired.identities().len() as i32;
                    let mut bank = Experts::new(&compact_args, stream)?;
                    bank.gate_up_proj_blocks = Param::new(
                        acquired
                            .compact_binding("gate_up_proj_blocks", stream)
                            .map_err(|error| Exception::custom(error.to_string()))?,
                    );
                    bank.gate_up_proj_scales = Param::new(
                        acquired
                            .compact_binding("gate_up_proj_scales", stream)
                            .map_err(|error| Exception::custom(error.to_string()))?,
                    );
                    bank.gate_up_proj_bias = Param::new(
                        acquired
                            .compact_binding("gate_up_proj_bias", stream)
                            .map_err(|error| Exception::custom(error.to_string()))?,
                    );
                    bank.down_proj_blocks = Param::new(
                        acquired
                            .compact_binding("down_proj_blocks", stream)
                            .map_err(|error| Exception::custom(error.to_string()))?,
                    );
                    bank.down_proj_scales = Param::new(
                        acquired
                            .compact_binding("down_proj_scales", stream)
                            .map_err(|error| Exception::custom(error.to_string()))?,
                    );
                    bank.down_proj_bias = Param::new(
                        acquired
                            .compact_binding("down_proj_bias", stream)
                            .map_err(|error| Exception::custom(error.to_string()))?,
                    );
                    expert_cache
                        .record_compact_bank(pass, acquired.scratch_bytes(), started.elapsed())
                        .map_err(|error| Exception::custom(error.to_string()))?;
                    let output = bank.forward(flat, acquired.compact_routes(), weights, stream)?;
                    eval([&output])?;
                    acquired
                        .complete_pending()
                        .map_err(|error| Exception::custom(error.to_string()))?;
                    Ok(output)
                },
            )?);
        }
        Ok(layer.forward(hidden, mask.as_ref(), layer_cache, stream)?)
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        _group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        cache.layers[index].retained_arrays()
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(self.lm_head.forward(&hidden, stream)?)
    }
}

pub(crate) fn gpt_oss_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        for expert in 0..args.num_local_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            for name in [
                "gate_up_proj_blocks",
                "gate_up_proj_scales",
                "gate_up_proj_bias",
                "down_proj_blocks",
                "down_proj_scales",
                "down_proj_bias",
            ] {
                let recipe = DerivedWeightRecipe::source(
                    format!("{prefix}.{name}"),
                    TensorSelection::Range {
                        axis: 0,
                        start: expert,
                        end: expert + 1,
                    },
                );
                let bytes = recipe.infer(store)?.byte_len();
                bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("GPT-OSS expert byte total overflowed".into())
                })
            })?;
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

/// GPT-OSS token generation iterator using layerwise-host execution.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, GptOssLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{ones_dtype, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::{load_gpt_oss_layerwise_model, load_gpt_oss_sparse_expert_cache_model};
    use crate::{
        cache::KeyValueCache,
        expert_cache::ExpertCacheLoadOptions,
        layerwise::LayerwiseLoadOptions,
        models::gpt_oss::{self as resident, Cache, Model, ModelArgs, MxFp4Config},
        offload::{OffloadConfig, ResidencyPolicy},
    };

    fn tiny_args() -> ModelArgs {
        ModelArgs {
            model_type: "gpt_oss".into(),
            hidden_size: 32,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 32,
            vocab_size: 32,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            rms_norm_eps: 1e-5,
            sliding_window: 4,
            max_position_embeddings: 64,
            rope_theta: 150_000.0,
            rope_scaling: None,
            layer_types: vec!["sliding_attention".into(), "full_attention".into()],
            quantization_config: MxFp4Config {
                quant_method: "mxfp4".into(),
            },
            quantization: None,
            swiglu_limit: 7.0,
        }
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if name.ends_with("_scales") {
                Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
            } else if name.ends_with("layernorm.weight") || name.as_ref() == "model.norm.weight" {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &Model) {
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
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": model.args.model_type,
                "hidden_size": model.args.hidden_size,
                "intermediate_size": model.args.intermediate_size,
                "num_hidden_layers": model.args.num_hidden_layers,
                "num_attention_heads": model.args.num_attention_heads,
                "num_key_value_heads": model.args.num_key_value_heads,
                "head_dim": model.args.head_dim,
                "vocab_size": model.args.vocab_size,
                "num_local_experts": model.args.num_local_experts,
                "num_experts_per_tok": model.args.num_experts_per_tok,
                "rms_norm_eps": model.args.rms_norm_eps,
                "sliding_window": model.args.sliding_window,
                "max_position_embeddings": model.args.max_position_embeddings,
                "rope_theta": model.args.rope_theta,
                "layer_types": model.args.layer_types,
                "quantization_config": {"quant_method": "mxfp4"},
                "swiglu_limit": model.args.swiglu_limit
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 3e-5, "{left} != {right}");
        }
    }

    fn parity(depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(tiny_args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_gpt_oss_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = Cache::default();
        let mut layerwise_cache = Cache::default();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
            Array::from_slice(&[6u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut resident_cache, gpu.stream())
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                assert_eq!(expected.offset(), actual.offset());
                assert_eq!(expected.max_size(), actual.max_size());
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("gpt_oss.layer."))
                .collect::<Vec<_>>();
            assert!(layers.iter().all(|unit| unit.host_resident()));
            assert!(layers.iter().filter(|unit| unit.device_resident()).count() <= depth);
            assert!(report
                .units()
                .iter()
                .filter(|unit| unit.device_resident() && !layers.contains(unit))
                .all(|unit| unit.policy() == ResidencyPolicy::Pinned));
        }
    }

    #[test]
    fn gpt_oss_native_mxfp4_layerwise_prefill_and_cached_decode_parity() {
        parity(1);
        parity(2);
    }

    #[test]
    fn gpt_oss_sparse_expert_cache_prefill_and_decode_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(tiny_args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);
        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
        )
        .unwrap();
        let mut cached =
            load_gpt_oss_sparse_expert_cache_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap();
        let mut resident_cache = Cache::default();
        let mut cached_cache = Cache::default();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut resident_cache, gpu.stream())
                .unwrap();
            let actual = cached
                .forward(&tokens, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 4);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        crate::expert_parallel::assert_rank_owned_sparse_ep_load(
            dir.path(),
            options,
            crate::models::ModelKind::GptOss,
            report.owned_experts / 2,
            gpu.stream(),
            cpu.stream(),
        );
    }
}
