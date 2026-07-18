//! Unified fully resident and layerwise-host Qwen3 execution.

use std::path::Path;

use safemlx::{
    error::Exception, module::Module, nn, ops::indexing::TryIndexOp, quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    layerwise::{
        load_layerwise_model, LayerwiseInput, LayerwiseLoadOptions, LayerwiseModel,
        LayerwiseModelAdapter, StaticUnitBindings,
    },
    models::{
        common::{
            attention::AttentionInput,
            generation::CausalLm,
            linear::{
                build_unloaded_maybe_quantized_lm_head_with_quantization,
                project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            },
        },
        input,
        qwen3::{self as resident, ModelArgs, TransformerBlock},
    },
    module_binding::{build_module_bindings, populate_module_from_lease},
    residency::{ResidencyReport, ResidentUnitLease},
    utils::{create_attention_mask, AttentionMask},
    weight_store::{SafetensorsWeightStore, WeightStore},
};

const EMBEDDING_UNIT: &str = "qwen3.static.embedding";
const NORM_UNIT: &str = "qwen3.static.norm";
const HEAD_UNIT: &str = "qwen3.static.output";

/// Host-backed Qwen3 dense or sparse-MoE causal LM.
pub struct Qwen3LayerwiseModel {
    execution: LayerwiseModel<Qwen3LayerwiseAdapter>,
}

impl Qwen3LayerwiseModel {
    /// Returns normalized Qwen3 arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates one standard device-resident KV cache per decoder block.
    pub fn new_cache(&self) -> Vec<Option<ConcatKeyValueCache>> {
        (0..self.args().num_hidden_layers)
            .map(|_| Some(ConcatKeyValueCache::new()))
            .collect()
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Returns the persistent checkpoint store.
    pub fn weight_store(&self) -> &SafetensorsWeightStore {
        self.execution.weight_store()
    }

    /// Runs dense or sparse-MoE Qwen3 with a standard KV cache.
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_with_cache(
            LayerwiseInput {
                inputs,
                mask,
                cache,
            },
            stream,
        )
    }

    /// Clears temporary device decoder copies.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_layer_window()
    }
}

impl CausalLm<Vec<Option<ConcatKeyValueCache>>> for Qwen3LayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads dense or sparse-MoE Qwen3 through the bounded host-residency engine.
pub fn load_qwen3_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: LayerwiseLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_qwen3_model_args(model_dir)?;
    let adapter = Qwen3LayerwiseAdapter::new(args, stream)?;
    Ok(Qwen3LayerwiseModel {
        execution: load_layerwise_model(model_dir, adapter, options, stream, weights_stream)?,
    })
}

/// Dense and sparse-MoE Qwen3 adapter sharing one complete-block execution path.
pub struct Qwen3LayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Qwen3LayerwiseAdapter {
    /// Creates metadata-only static Qwen3 modules.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let embedding = unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
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
                args.weight_quantization_for("lm_head.weight"),
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

    /// Returns normalized model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }
}

/// Attention mask shared by every temporary Qwen3 decoder block.
pub struct Qwen3ForwardContext {
    mask: Option<Array>,
}

impl LayerwiseModelAdapter for Qwen3LayerwiseAdapter {
    type Layer = TransformerBlock;
    type ForwardContext = Qwen3ForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn quantization(&self) -> Option<crate::quantization::WeightQuantization> {
        self.args.quantization.or(self.args.quantization_config)
    }

    fn layer_count(&self) -> Result<usize, Error> {
        usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3 layer count {} is invalid",
                self.args.num_hidden_layers
            ))
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
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3 adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.embedding, &leases[0])?;
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[2])?;
        }
        Ok(())
    }

    fn new_layer(&self, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        let index = i32::try_from(index)
            .map_err(|_| Error::UnsupportedArchitecture("Qwen3 layer index exceeds i32".into()))?;
        Ok(TransformerBlock::new_for_layer(&self.args, index, stream)?)
    }

    fn layer_checkpoint_prefix(&self, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, index: usize) -> String {
        format!("qwen3.layer.{index:05}")
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
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(hidden, cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Qwen3 layerwise execution requires an array attention mask".into(),
                    ));
                }
                None => None,
            },
        };
        Ok(Qwen3ForwardContext { mask })
    }

    fn forward_layer<C: KeyValueCache>(
        &self,
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
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters, ops::ones_dtype, Array, Device, DeviceType, ExecutionContext,
    };

    use super::*;
    use crate::{
        models::qwen3,
        offload::{OffloadConfig, ResidencyPolicy},
    };

    fn args(moe: bool) -> ModelArgs {
        ModelArgs {
            model_type: "qwen3".into(),
            hidden_size: 8,
            num_hidden_layers: 3,
            intermediate_size: if moe { 0 } else { 16 },
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            head_dim: 4,
            tie_word_embeddings: false,
            rope_scaling: None,
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: if moe { 8 } else { 0 },
            num_experts: if moe { 4 } else { 0 },
            num_experts_per_tok: if moe { 2 } else { 0 },
            norm_topk_prob: moe,
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
                Array::full::<f32>(&shape, Array::from_f32(0.001 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &qwen3::Model) {
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
                "num_hidden_layers": model.args.num_hidden_layers,
                "intermediate_size": model.args.intermediate_size,
                "num_attention_heads": model.args.num_attention_heads,
                "num_key_value_heads": model.args.num_key_value_heads,
                "rms_norm_eps": model.args.rms_norm_eps,
                "vocab_size": model.args.vocab_size,
                "max_position_embeddings": model.args.max_position_embeddings,
                "rope_theta": model.args.rope_theta,
                "head_dim": model.args.head_dim,
                "tie_word_embeddings": model.args.tie_word_embeddings,
                "moe_intermediate_size": model.args.moe_intermediate_size,
                "num_experts": model.args.num_experts,
                "num_experts_per_tok": model.args.num_experts_per_tok,
                "norm_topk_prob": model.args.norm_topk_prob
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

    fn parity(moe: bool, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = qwen3::Model::new(args(moe), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);

        let mut resident = qwen3::load_qwen3_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_qwen3_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let mut layerwise_cache = layerwise.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    qwen3::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut resident_cache,
                    },
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, None, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("qwen3.layer."))
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
    fn qwen3_dense_layerwise_prefill_and_cached_decode_parity() {
        parity(false, 1);
        parity(false, 2);
    }

    #[test]
    fn qwen3_sparse_moe_layerwise_prefill_and_cached_decode_parity() {
        parity(true, 1);
    }
}
