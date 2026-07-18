//! Layerwise-host execution for DeepSeek-V3 and DeepSeek-R1 checkpoints.

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
    error::Error,
    layerwise::{
        load_general_layerwise_model, GeneralLayerwiseModel, GeneralLayerwiseModelAdapter,
        LayerwiseForwardState, LayerwiseLoadOptions, StaticUnitBindings,
    },
    models::{
        common::{self, generation::CausalLm},
        deepseek_v3::{self as resident, Cache, DecoderLayer, ModelArgs},
        input,
    },
    module_binding::{
        build_module_bindings_with_recipes, canonical_checkpoint_name, populate_module_from_lease,
    },
    residency::{ResidencyReport, ResidentUnitLease, WeightBinding},
    utils::create_causal_mask,
    weight_recipe::DerivedWeightRecipe,
    weight_store::{SafetensorsWeightStore, TensorSelection, WeightStore},
};

const EMBEDDING_UNIT: &str = "deepseek_v3.static.embedding";
const NORM_UNIT: &str = "deepseek_v3.static.norm";
const HEAD_UNIT: &str = "deepseek_v3.static.output";

/// DeepSeek-V3/R1 causal LM using bounded host residency for decoder blocks.
pub struct DeepSeekV3LayerwiseModel {
    execution: GeneralLayerwiseModel<DeepSeekV3LayerwiseAdapter>,
}

impl DeepSeekV3LayerwiseModel {
    /// Returns the validated architecture arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates one compressed MLA cache per decoder block.
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

    /// Runs MLA and dense/MoE decoder blocks while preserving compressed state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }

    /// Clears temporary decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalLm<Cache> for DeepSeekV3LayerwiseModel {
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

/// Loads DeepSeek-V3/R1 through the generalized host-residency engine.
pub fn load_deepseek_v3_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: LayerwiseLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let adapter = DeepSeekV3LayerwiseAdapter::new(args, stream)?;
    Ok(DeepSeekV3LayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Adapter for compressed MLA and mixed dense/MoE DeepSeek decoder blocks.
pub struct DeepSeekV3LayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
}

impl DeepSeekV3LayerwiseAdapter {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            embedding: common::linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            lm_head: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?,
            args,
        })
    }

    /// Returns the validated architecture arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(self.args.num_hidden_layers)
    }

    fn recipes_for_layer(
        &self,
        layer: &DecoderLayer,
        index: usize,
        store: &dyn WeightStore,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let prefix = format!("model.layers.{index}");
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.keys();
        let mut recipes = BTreeMap::new();

        for local_name in layer.parameters().flatten().keys() {
            let destination = format!("{prefix}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            if let Some((projection, component)) = expert_destination(local_name.as_ref()) {
                let mut inputs = Vec::with_capacity(self.args.n_routed_experts as usize);
                for expert in 0..self.args.n_routed_experts {
                    let runtime = format!("{prefix}.mlp.experts.{expert}.{projection}.{component}");
                    let raw = normalized.get(&runtime).ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "DeepSeek-V3 checkpoint is missing split expert tensor {runtime}"
                        ))
                    })?;
                    inputs.push(DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Full,
                    ));
                }
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::Stack { axis: 0, inputs },
                );
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DeepSeek-V3 checkpoint is missing runtime parameter {canonical}"
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

fn normalized_checkpoint_keys(store: &dyn WeightStore) -> BTreeMap<String, String> {
    store
        .keys()
        .into_iter()
        .map(|raw| (canonical_checkpoint_name(&raw), raw))
        .collect()
}

fn expert_destination(local_name: &str) -> Option<(&'static str, &'static str)> {
    ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .find_map(|projection| {
            [
                ("", "weight"),
                ("_scale_inv", "weight_scale_inv"),
                ("_scales", "scales"),
                ("_biases", "biases"),
            ]
            .into_iter()
            .find_map(|(runtime_suffix, checkpoint_component)| {
                (local_name == format!("mlp.experts.{projection}{runtime_suffix}"))
                    .then_some((projection, checkpoint_component))
            })
        })
}

/// Per-forward causal mask shared by all MLA blocks.
pub struct DeepSeekV3ForwardContext {
    mask: Option<Array>,
}

impl GeneralLayerwiseModelAdapter for DeepSeekV3LayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = DeepSeekV3ForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        Ok(vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    BTreeMap::new(),
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    BTreeMap::new(),
                )?,
            )?,
            StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(
                    &self.lm_head,
                    "lm_head",
                    store,
                    BTreeMap::new(),
                )?,
            )?,
        ])
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        if leases.len() != 3 {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 adapter received {} static leases, expected 3",
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
        }
        if cache.layers.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.num_hidden_layers
            )));
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input, stream)?;
        let offset = cache.offset();
        let mask = if hidden.dim(1) > 1 && offset > 0 {
            Some(create_causal_mask(
                hidden.dim(1),
                Some(offset),
                None,
                None,
                stream,
            )?)
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: DeepSeekV3ForwardContext { mask },
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
                "DeepSeek-V3 decoder has no execution group {group}"
            )))
        }
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 decoder has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        Ok(DecoderLayer::new_layerwise(
            &self.args,
            index as i32,
            stream,
        )?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("deepseek_v3.layer.{index:05}")
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
            self.recipes_for_layer(layer, index, store)?,
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
        Ok(layer.forward_stage(
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
            stream,
        )?)
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        _group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        cache.layers[index]
            .arrays()
            .map(|(latent, rotary)| vec![latent, rotary])
            .unwrap_or_default()
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

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        (0..self.args.num_nextn_predict_layers).any(|index| {
            key.starts_with(&format!(
                "model.layers.{}.",
                self.args.num_hidden_layers + index
            ))
        })
    }
}

/// DeepSeek token generation using layerwise-host execution.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, DeepSeekV3LayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::{ModuleParameters, Param},
        ops::{indexing::TryIndexOp, ones_dtype, zeros_dtype},
        Array, Device, DeviceType, Dtype, ExecutionContext, Stream,
    };

    use super::load_deepseek_v3_layerwise_model;
    use crate::{
        layerwise::LayerwiseLoadOptions,
        models::deepseek_v3::{self as resident, FeedForward, Model, ModelArgs, ModelInput},
        module_binding::canonical_checkpoint_name,
        offload::{OffloadConfig, ResidencyPolicy},
    };

    fn config(fp8: bool) -> serde_json::Value {
        let mut value = serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.5,
            "num_nextn_predict_layers": 1,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "eos_token_id": 1
        });
        if fp8 {
            value.as_object_mut().unwrap().insert(
                "quantization_config".into(),
                serde_json::json!({
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128]
                }),
            );
        }
        value
    }

    fn args(fp8: bool) -> ModelArgs {
        serde_json::from_value(config(fp8)).unwrap()
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for layer in &mut model.model.layers {
            if let FeedForward::Moe(moe) = &mut layer.mlp {
                let experts = model.args.n_routed_experts;
                let hidden = model.args.hidden_size;
                let intermediate = model.args.moe_intermediate_size;
                let weight = |shape: &[i32]| {
                    if model.args.native_fp8_config().is_some() {
                        Array::full::<u8>(shape, Array::from_slice(&[0x38u8], &[]), stream).unwrap()
                    } else {
                        Array::full::<f32>(shape, Array::from_f32(0.01), stream).unwrap()
                    }
                };
                moe.experts.gate_proj = Param::new(Some(weight(&[experts, intermediate, hidden])));
                moe.experts.up_proj = Param::new(Some(weight(&[experts, intermediate, hidden])));
                moe.experts.down_proj = Param::new(Some(weight(&[experts, hidden, intermediate])));
                if model.args.native_fp8_config().is_some() {
                    moe.experts.gate_proj_scale_inv =
                        Param::new(Some(Array::ones::<f32>(&[experts, 1, 1], stream).unwrap()));
                    moe.experts.up_proj_scale_inv =
                        Param::new(Some(Array::ones::<f32>(&[experts, 1, 1], stream).unwrap()));
                    moe.experts.down_proj_scale_inv =
                        Param::new(Some(Array::ones::<f32>(&[experts, 1, 1], stream).unwrap()));
                }
            }
        }
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if dtype == Dtype::Uint8 {
                Array::full::<u8>(&shape, Array::from_slice(&[0x38u8], &[]), stream).unwrap()
            } else if name.ends_with("layernorm.weight")
                || name.as_ref() == "model.norm.weight"
                || name.ends_with("weight_scale_inv")
                || name.ends_with("_scale_inv")
            {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else if dtype == Dtype::Float32 {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &Model, fp8: bool, stream: &Stream) {
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in model.parameters().flatten() {
            let name = canonical_checkpoint_name(&name);
            let packed = ["gate_proj", "up_proj", "down_proj"]
                .into_iter()
                .find_map(|projection| {
                    [
                        ("", "weight"),
                        ("_scale_inv", "weight_scale_inv"),
                        ("_scales", "scales"),
                        ("_biases", "biases"),
                    ]
                    .into_iter()
                    .find_map(|(runtime_suffix, checkpoint_component)| {
                        name.ends_with(&format!(".mlp.experts.{projection}{runtime_suffix}"))
                            .then_some((projection, runtime_suffix, checkpoint_component))
                    })
                });
            if let Some((projection, runtime_suffix, checkpoint_component)) = packed {
                let suffix = format!(".experts.{projection}{runtime_suffix}");
                let prefix = name.strip_suffix(&suffix).unwrap();
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.experts.{expert}.{projection}.{checkpoint_component}"),
                        value.try_index_device(expert, stream).unwrap(),
                    ));
                }
            } else {
                arrays.push((name, value.clone()));
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
            serde_json::to_vec(&config(fp8)).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array, tolerance: f32) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= tolerance, "{left} != {right}");
        }
    }

    fn parity(fp8: bool, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(fp8), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, fp8, gpu.stream());

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap());
        let mut layerwise =
            load_deepseek_v3_layerwise_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = resident::Cache { layers: Vec::new() };
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
            assert_close(&actual, &expected, if fp8 { 2e-4 } else { 3e-5 });
            assert_eq!(resident_cache.offset(), layerwise_cache.offset());
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                assert_eq!(expected.offset(), actual.offset());
                let (expected_latent, expected_rotary) = expected.arrays().unwrap();
                let (actual_latent, actual_rotary) = actual.arrays().unwrap();
                assert_eq!(expected_latent.shape(), actual_latent.shape());
                assert_eq!(expected_rotary.shape(), actual_rotary.shape());
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("deepseek_v3.layer."))
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
    fn deepseek_v3_split_moe_layerwise_parity() {
        parity(false, 1);
        parity(false, 2);
    }

    #[test]
    fn deepseek_v3_native_fp8_split_moe_layerwise_parity() {
        parity(true, 1);
    }
}
