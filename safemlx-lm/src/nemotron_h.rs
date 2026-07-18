//! Unified fully resident and layerwise-host Nemotron-H execution.

use std::{collections::BTreeMap, path::Path};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::indexing::TryIndexOp,
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    cache::KeyValueCache,
    error::Error,
    layerwise::{
        load_general_layerwise_model, GeneralLayerwiseModel, GeneralLayerwiseModelAdapter,
        LayerwiseForwardState, LayerwiseLoadOptions, StaticUnitBindings,
    },
    models::{
        common::{self, generation::CausalLm, linear::project_logits_maybe_quantized},
        input,
        nemotron_h::{
            self as resident, BlockInput, Cache, LayerBlockType, LayerCache, ModelArgs,
            TransformerBlock,
        },
    },
    module_binding::{
        build_module_bindings_with_recipes, canonical_checkpoint_name, populate_module_from_lease,
    },
    residency::{ResidencyReport, ResidentUnitLease, WeightBinding},
    utils::{create_attention_mask, AttentionMask},
    weight_recipe::DerivedWeightRecipe,
    weight_store::{SafetensorsWeightStore, TensorSelection, WeightStore},
};

const EMBEDDING_UNIT: &str = "nemotron_h.static.embedding";
const NORM_UNIT: &str = "nemotron_h.static.norm";
const HEAD_UNIT: &str = "nemotron_h.static.output";

/// Nemotron-H causal LM using bounded host residency for hybrid blocks.
pub struct NemotronHLayerwiseModel {
    execution: GeneralLayerwiseModel<NemotronHLayerwiseAdapter>,
}

impl NemotronHLayerwiseModel {
    /// Returns validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates cache/state matching the hybrid block pattern.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Returns the persistent checkpoint store.
    pub fn weight_store(&self) -> &SafetensorsWeightStore {
        self.execution.weight_store()
    }

    /// Runs the hybrid decoder while preserving KV and Mamba state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }

    /// Clears temporary hybrid blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalLm<Cache> for NemotronHLayerwiseModel {
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

/// Loads Nemotron-H through the generalized bounded host-residency engine.
pub fn load_nemotron_h_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: LayerwiseLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_nemotron_h_model_args(model_dir)?;
    let adapter = NemotronHLayerwiseAdapter::new(args, stream)?;
    Ok(NemotronHLayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Adapter shared by Nemotron-H Mamba, attention, dense, and MoE blocks.
pub struct NemotronHLayerwiseAdapter {
    args: ModelArgs,
    embeddings: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl NemotronHLayerwiseAdapter {
    /// Creates metadata-only pinned modules.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let embeddings = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.affine_quantization_for("model.embeddings.weight")
                .map(Into::into),
            stream,
        )?;
        let norm = nn::RmsNorm::unloaded(args.hidden_size, args.norm_eps, Dtype::Float32, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.affine_quantization_for("lm_head.weight")
                    .map(Into::into),
                stream,
            )?)
        };
        Ok(Self {
            args,
            embeddings,
            norm,
            lm_head,
        })
    }

    /// Returns validated model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args).expect("validated Nemotron-H hybrid pattern")
    }

    fn recipes_for_module(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
        layer_index: Option<usize>,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let normalized = normalized_checkpoint_keys(store, &self.args)?;
        let keys = store.keys();
        let mut recipes = BTreeMap::new();

        if let Some(index) = layer_index
            .filter(|index| self.args.layer_block_type(*index).ok() == Some(LayerBlockType::Moe))
        {
            let packed_prefix = format!("model.layers.{index}.moe.experts");
            if !keys.contains(&format!("{packed_prefix}.up_proj"))
                && !normalized.contains_key(&format!("{packed_prefix}.up_proj"))
            {
                let mut up = Vec::with_capacity(self.args.n_routed_experts as usize);
                let mut down = Vec::with_capacity(self.args.n_routed_experts as usize);
                for expert in 0..self.args.n_routed_experts {
                    up.push(source_for_normalized(
                        &normalized,
                        &format!("{packed_prefix}.{expert}.up_proj.weight"),
                    )?);
                    down.push(source_for_normalized(
                        &normalized,
                        &format!("{packed_prefix}.{expert}.down_proj.weight"),
                    )?);
                }
                recipes.insert(
                    "moe.experts.up_proj".into(),
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: up,
                    },
                );
                recipes.insert(
                    "moe.experts.down_proj".into(),
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: down,
                    },
                );
            }
        }

        for local_name in module.parameters().flatten().keys() {
            if recipes.contains_key(local_name.as_ref()) {
                continue;
            }
            let destination = if prefix.is_empty() {
                local_name.to_string()
            } else {
                format!("{prefix}.{local_name}")
            };
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Nemotron-H checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(recipes)
    }
}

fn normalized_checkpoint_keys(
    store: &dyn WeightStore,
    args: &ModelArgs,
) -> Result<BTreeMap<String, String>, Error> {
    let mut normalized = BTreeMap::new();
    for raw in store.keys() {
        let rewritten = resident::rewrite_nemotron_h_weight_key(&raw, args)?;
        let runtime = if let Some(rest) = rewritten.strip_prefix("model.backbone.") {
            format!("model.{rest}")
        } else if let Some(rest) = rewritten.strip_prefix("backbone.") {
            format!("model.{rest}")
        } else {
            rewritten
        };
        normalized.insert(runtime, raw);
    }
    Ok(normalized)
}

fn source_for_normalized(
    normalized: &BTreeMap<String, String>,
    runtime: &str,
) -> Result<DerivedWeightRecipe, Error> {
    normalized
        .get(runtime)
        .cloned()
        .map(|key| DerivedWeightRecipe::source(key, TensorSelection::Full))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Nemotron-H checkpoint is missing split expert tensor {runtime}"
            ))
        })
}

/// Per-forward causal mask shared by attention blocks.
pub struct NemotronHForwardContext {
    mask: Option<Array>,
}

struct OffsetOnlyCache(i32);

impl KeyValueCache for OffsetOnlyCache {
    fn offset(&self) -> i32 {
        self.0
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        _stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        Ok((keys, values))
    }
}

impl GeneralLayerwiseModelAdapter for NemotronHLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = TransformerBlock;
    type ForwardContext = NemotronHForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.embeddings,
                    "model.embeddings",
                    store,
                    self.recipes_for_module(&self.embeddings, "model.embeddings", store, None)?,
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm_f",
                    store,
                    self.recipes_for_module(&self.norm, "model.norm_f", store, None)?,
                )?,
            )?,
        ];
        if let Some(head) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(
                    head,
                    "lm_head",
                    store,
                    self.recipes_for_module(head, "lm_head", store, None)?,
                )?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.embeddings, &leases[0])?;
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        let types = self.args.layer_block_types()?;
        if cache.layers.len() != types.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H cache has {} layers, expected {}",
                cache.layers.len(),
                types.len()
            )));
        }
        for (index, (block_type, cache)) in types.iter().zip(&cache.layers).enumerate() {
            let matches = matches!(
                (block_type, cache),
                (LayerBlockType::Mamba, LayerCache::Mamba(_))
                    | (LayerBlockType::Attention, LayerCache::Attention(_))
                    | (LayerBlockType::Mlp, LayerCache::Mlp)
                    | (LayerBlockType::Moe, LayerCache::Moe)
            );
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Nemotron-H cache kind does not match layer pattern at layer {index}"
                )));
            }
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embeddings.forward(input, stream)?;
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Nemotron-H requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: NemotronHForwardContext { mask },
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
                "Nemotron-H has no execution group {group}"
            )))
        }
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        TransformerBlock::new(&self.args, index, stream)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("nemotron_h.layer.{index:05}")
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = format!("model.layers.{index}");
        Ok(build_module_bindings_with_recipes(
            layer,
            &prefix,
            store,
            self.recipes_for_module(layer, &prefix, store, Some(index))?,
        )?)
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
        Ok(layer.forward(
            BlockInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(&mut cache.layers[index]),
            },
            stream,
        )?)
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
        Ok(project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embeddings,
            &hidden,
            stream,
        )?)
    }
}

/// Nemotron-H token generation iterator using layerwise-host execution.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, NemotronHLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, ones_dtype, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::load_nemotron_h_layerwise_model;
    use crate::{
        layerwise::LayerwiseLoadOptions,
        models::nemotron_h::{self as resident, Cache, LayerCache, Model, ModelArgs, ModelInput},
        offload::{OffloadConfig, ResidencyPolicy},
    };

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "nemotron_h",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 4,
            "hybrid_override_pattern": "M-E*",
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "mamba_num_heads": 2,
            "mamba_head_dim": 4,
            "n_groups": 1,
            "ssm_state_size": 4,
            "conv_kernel": 3,
            "chunk_size": 2,
            "moe_intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 10,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": false,
            "torch_dtype": "float32"
        })
    }

    fn args() -> ModelArgs {
        serde_json::from_value(config()).unwrap()
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if name.ends_with("norm.weight") || name.as_ref() == "model.norm_f.weight"
            {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
    }

    fn public_name(runtime: &str, args: &ModelArgs) -> String {
        if let Some(rest) = runtime.strip_prefix("model.embeddings.") {
            return format!("backbone.embeddings.{rest}");
        }
        if let Some(rest) = runtime.strip_prefix("model.norm_f.") {
            return format!("backbone.norm_f.{rest}");
        }
        for index in 0..args.num_hidden_layers as usize {
            let prefix = format!("model.layers.{index}.");
            let Some(rest) = runtime.strip_prefix(&prefix) else {
                continue;
            };
            let field = match args.layer_block_type(index).unwrap() {
                crate::models::nemotron_h::LayerBlockType::Mamba => "mamba",
                crate::models::nemotron_h::LayerBlockType::Attention => "attention",
                crate::models::nemotron_h::LayerBlockType::Mlp => "mlp",
                crate::models::nemotron_h::LayerBlockType::Moe => "moe",
            };
            if let Some(mixer_rest) = rest.strip_prefix(&format!("{field}.")) {
                return format!("backbone.layers.{index}.mixer.{mixer_rest}");
            }
            return format!("backbone.layers.{index}.{rest}");
        }
        runtime.to_string()
    }

    fn write_fixture(dir: &Path, model: &Model, stream: &Stream) {
        let params = model.parameters().flatten();
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in params {
            let runtime = crate::module_binding::canonical_checkpoint_name(&name);
            if runtime.ends_with("moe.experts.up_proj") {
                let prefix = public_name(runtime.trim_end_matches(".up_proj"), &model.args);
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.up_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
            } else if runtime.ends_with("moe.experts.down_proj") {
                let prefix = public_name(runtime.trim_end_matches(".down_proj"), &model.args);
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.down_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
            } else {
                arrays.push((public_name(&runtime, &model.args), value.clone()));
            }
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config()).unwrap(),
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
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());

        let mut resident =
            resident::load_nemotron_h_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_nemotron_h_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                let expected_offset = match expected {
                    LayerCache::Mamba(cache) => Some(cache.offset),
                    LayerCache::Attention(cache) => {
                        Some(crate::cache::KeyValueCache::offset(cache))
                    }
                    LayerCache::Mlp | LayerCache::Moe => None,
                };
                let actual_offset = match actual {
                    LayerCache::Mamba(cache) => Some(cache.offset),
                    LayerCache::Attention(cache) => {
                        Some(crate::cache::KeyValueCache::offset(cache))
                    }
                    LayerCache::Mlp | LayerCache::Moe => None,
                };
                assert_eq!(expected_offset, actual_offset);
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("nemotron_h.layer."))
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
    fn nemotron_h_public_split_moe_hybrid_layerwise_parity() {
        parity(1);
        parity(2);
    }
}
