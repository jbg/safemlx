//! Shared layerwise-host execution for dense and MoE Qwen3-VL models.

use std::{path::Path, time::Instant};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{
        concatenate_axis,
        indexing::{masked_scatter, TryIndexOp},
        zeros_dtype,
    },
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Stream,
};

use crate::{
    cache::KeyValueCache,
    error::Error,
    expert_cache::{ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertPass},
    layerwise::{
        load_general_layerwise_model, GeneralLayerwiseModel, GeneralLayerwiseModelAdapter,
        LayerExecutionLoadOptions, LayerwiseForwardState, StaticUnitBindings,
    },
    models::{
        common::{self, attention::AttentionInput, generation::CausalLm},
        input,
        qwen3::{Experts as QwenExperts, Qwen3Model, TransformerBlock},
        qwen3_vl::{self as resident, Cache, ModelArgs},
        qwen_vl::{
            grid_thw_from_array, QwenVisionBlock, QwenVisionLayerwiseState,
            QwenVisionLayerwiseStatic, QwenVisionTransformer,
        },
    },
    module_binding::{
        build_module_bindings, populate_module_from_lease, populate_module_from_lease_excluding,
    },
    residency::{ResidencyReport, ResidentUnitLease, WeightBinding},
    utils::{create_attention_mask, AttentionMask},
    weight_store::{SafetensorsWeightStore, WeightStore},
};

const VISION_STATIC_UNIT: &str = "qwen3_vl.static.vision";
const EMBEDDING_UNIT: &str = "qwen3_vl.static.embedding";
const NORM_UNIT: &str = "qwen3_vl.static.norm";
const HEAD_UNIT: &str = "qwen3_vl.static.output";

/// Dense or MoE Qwen3-VL with independent vision and text residency windows.
pub struct Qwen3VlLayerwiseModel {
    execution: GeneralLayerwiseModel<Qwen3VlLayerwiseAdapter>,
}

impl Qwen3VlLayerwiseModel {
    /// Returns the parsed multimodal model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
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

    /// Returns the public architecture type.
    pub fn model_type(&self) -> &'static str {
        if self.args().text_config.model_type == "qwen3_vl_moe_text" {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
    }

    /// Creates empty KV and multimodal position state.
    pub fn new_cache(&self) -> Cache {
        Cache::default()
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::layerwise::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns the persistent checkpoint store.
    pub fn weight_store(&self) -> &SafetensorsWeightStore {
        self.execution.weight_store()
    }

    /// Runs typed multimodal prefill through vision and text execution groups.
    pub fn prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Qwen3VlInput::Prefill(input), cache, stream)
    }

    /// Runs a text decode step using cached multimodal RoPE state.
    pub fn decode(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Qwen3VlInput::Decode(tokens), cache, stream)
    }

    /// Clears temporary copies for one execution group.
    pub fn clear_device_group(&self, group: &str) -> Result<(), Error> {
        self.execution.clear_device_group(group)
    }
}

impl CausalLm<Cache> for Qwen3VlLayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.prefill(input, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads either Qwen3-VL architecture through shared bounded residency.
pub fn load_qwen3_vl_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    let adapter = Qwen3VlLayerwiseAdapter::new(args, stream)?;
    Ok(Qwen3VlLayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Qwen3-VL-MoE with expert-granular sparse caching.
pub fn load_qwen3_vl_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    load_qwen3_vl_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Qwen3-VL-MoE with expert caching and disk-streamed non-expert units.
pub fn load_qwen3_vl_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    load_qwen3_vl_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

fn load_qwen3_vl_sparse_expert_cache_model_with_non_expert(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3-VL-MoE checkpoint".into(),
        ));
    }
    let mut adapter = Qwen3VlLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution =
        load_general_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.weight_store_arc();
    let entries = crate::qwen3::qwen3_expert_catalog_at(
        &args.text_config,
        store.as_ref(),
        "model.language_model.layers",
    )?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(Qwen3VlLayerwiseModel { execution })
}

/// Family-specific input distinguishing typed prefill from cached decode.
pub enum Qwen3VlInput<'a> {
    /// Ordered text and visual prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text token IDs for a cached decode step.
    Decode(&'a Array),
}

enum PreparedPart {
    Text(Array),
    Visual(i32),
}

/// Per-forward vision preparation and multimodal text state.
pub struct Qwen3VlForwardContext {
    tokens: Array,
    parts: Vec<PreparedPart>,
    vision: Option<QwenVisionLayerwiseState>,
    mask: Option<Array>,
    cos: Array,
    sin: Array,
    visual_mask: Option<Array>,
    deepstack_features: Vec<Array>,
}

/// One temporary unit from either the vision or text group.
pub enum Qwen3VlLayer {
    /// Vision transformer block.
    Vision(QwenVisionBlock),
    /// Dense or sparse-MoE Qwen3 decoder block.
    Text(TransformerBlock),
}

impl ModuleParameters for Qwen3VlLayer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Vision(layer) => layer.num_parameters(),
            Self::Text(layer) => layer.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.parameters(),
            Self::Text(layer) => layer.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Vision(layer) => layer.parameters_mut(),
            Self::Text(layer) => layer.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.trainable_parameters(),
            Self::Text(layer) => layer.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.freeze_parameters(recursive),
            Self::Text(layer) => layer.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.unfreeze_parameters(recursive),
            Self::Text(layer) => layer.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.all_frozen(),
            Self::Text(layer) => layer.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.any_frozen(),
            Self::Text(layer) => layer.any_frozen(),
        }
    }
}

/// Shared dense/MoE multimodal adapter.
pub struct Qwen3VlLayerwiseAdapter {
    args: ModelArgs,
    vision: QwenVisionLayerwiseStatic,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl Qwen3VlLayerwiseAdapter {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let visual = QwenVisionTransformer::new_deepstack(args.vision_config.clone(), stream)?;
        let text = Qwen3Model::new(&args.text_config, stream)?;
        let lm_head = if args.text_config.tie_word_embeddings {
            None
        } else {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.text_config.hidden_size,
                    args.text_config.vocab_size,
                    args.text_config
                        .quantization
                        .or(args.text_config.quantization_config),
                    stream,
                )?,
            )
        };
        Ok(Self {
            args,
            vision: QwenVisionLayerwiseStatic::from_transformer(visual),
            embedding: text.embed_tokens,
            norm: text.norm,
            lm_head,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    /// Returns parsed multimodal arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn prepare_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Qwen3VlForwardContext>, Error> {
        input::validate(typed)?;
        let mut token_parts = Vec::new();
        let mut prepared_parts = Vec::new();
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        let merge = self.args.vision_config.spatial_merge_size;
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    token_parts.push(tokens.clone());
                    prepared_parts
                        .push(PreparedPart::Text(self.embedding.forward(tokens, stream)?));
                }
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Tensor(tensor),
                ) => {
                    let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Qwen3-VL {} input requires qwen_grid_thw metadata",
                            part.modality.as_str()
                        ))
                    })?;
                    let merged = grid_thw_from_array(grid, stream)?
                        .into_iter()
                        .map(|(t, h, w)| t * (h / merge) * (w / merge))
                        .sum::<i32>();
                    let token_id = if part.modality == input::Modality::Image {
                        self.args.image_token_id
                    } else {
                        self.args.video_token_id
                    };
                    token_parts.push(input::token_ids_array(
                        &vec![token_id; merged as usize],
                        stream,
                    )?);
                    prepared_parts.push(PreparedPart::Visual(merged));
                    pixels.push(tensor.clone());
                    grids.push(grid.clone());
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Qwen3-VL layerwise input does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        let token_refs = token_parts.iter().collect::<Vec<_>>();
        let tokens = concatenate_axis(&token_refs, 1, stream)?;
        let (position_ids, rope_delta) =
            resident::multimodal_position_ids(typed, merge, tokens.dim(1), stream)?;
        cache.rope_delta = rope_delta;
        let (cos, sin) = resident::mrope_embeddings(
            &position_ids,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        let (hidden, vision) = if pixels.is_empty() {
            let hidden = prepared_parts
                .iter()
                .filter_map(|part| match part {
                    PreparedPart::Text(value) => Some(value),
                    PreparedPart::Visual(_) => None,
                })
                .collect::<Vec<_>>();
            (concatenate_axis(&hidden, 1, stream)?, None)
        } else {
            let pixel_refs = pixels.iter().collect::<Vec<_>>();
            let grid_refs = grids.iter().collect::<Vec<_>>();
            let pixels = concatenate_axis(&pixel_refs, 0, stream)?;
            let grids = concatenate_axis(&grid_refs, 0, stream)?;
            let (hidden, state) = self.vision.begin(&pixels, &grids, stream)?;
            (hidden, Some(state))
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: Qwen3VlForwardContext {
                tokens,
                parts: prepared_parts,
                vision,
                mask: None,
                cos,
                sin,
                visual_mask: None,
                deepstack_features: Vec::new(),
            },
        })
    }

    fn prepare_decode(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Qwen3VlForwardContext>, Error> {
        let hidden = self.embedding.forward(tokens, stream)?;
        let start = cache
            .kv
            .first()
            .and_then(Option::as_ref)
            .map(KeyValueCache::offset)
            .unwrap_or(0)
            + cache.rope_delta;
        let positions = [
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
        ];
        let (cos, sin) = resident::mrope_embeddings(
            &positions,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        Ok(LayerwiseForwardState {
            hidden,
            context: Qwen3VlForwardContext {
                tokens: tokens.clone(),
                parts: Vec::new(),
                vision: None,
                mask: None,
                cos,
                sin,
                visual_mask: None,
                deepstack_features: Vec::new(),
            },
        })
    }
}

impl GeneralLayerwiseModelAdapter for Qwen3VlLayerwiseAdapter {
    type Input<'a> = Qwen3VlInput<'a>;
    type Cache = Cache;
    type Layer = Qwen3VlLayer;
    type ForwardContext = Qwen3VlForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                VISION_STATIC_UNIT,
                build_module_bindings(&self.vision, "model.visual", store)?,
            )?,
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings(&self.embedding, "model.language_model.embed_tokens", store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings(&self.norm, "model.language_model.norm", store)?,
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
        let expected = if self.lm_head.is_some() { 4 } else { 3 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.vision, &leases[0])?;
        populate_module_from_lease(&mut self.embedding, &leases[1])?;
        populate_module_from_lease(&mut self.norm, &leases[2])?;
        if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[3])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.kv.is_empty() {
            cache.kv = (0..self.args.text_config.num_hidden_layers)
                .map(|_| Some(Default::default()))
                .collect();
        }
        if cache.kv.len() != self.args.text_config.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL cache has {} layers, expected {}",
                cache.kv.len(),
                self.args.text_config.num_hidden_layers
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
        match input {
            Qwen3VlInput::Prefill(input) => self.prepare_prefill(input, cache, stream),
            Qwen3VlInput::Decode(tokens) => self.prepare_decode(tokens, cache, stream),
        }
    }

    fn execution_group_count(&self) -> usize {
        2
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        match group {
            0 => Ok("vision_encoder".into()),
            1 => Ok("text_decoder".into()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        group != 0 || context.vision.is_some()
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.args.vision_config.depth as usize),
            1 => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        match group {
            0 => Ok(Qwen3VlLayer::Vision(QwenVisionBlock::new(
                &self.args.vision_config,
                stream,
            )?)),
            1 => Ok(Qwen3VlLayer::Text(TransformerBlock::new_for_layer(
                &self.args.text_config,
                index as i32,
                stream,
            )?)),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("model.visual.blocks.{index}")
        } else {
            format!("model.language_model.layers.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("qwen3_vl.vision.{index:05}")
        } else {
            format!("qwen3_vl.text.{index:05}")
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings =
            build_module_bindings(layer, &self.layer_checkpoint_prefix(group, index), store)?;
        Ok(if self.sparse_expert_cache && group == 1 {
            bindings
                .into_iter()
                .filter(|binding| !binding.name().starts_with("mlp.experts."))
                .collect()
        } else {
            bindings
        })
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
        match (group, layer) {
            (0, Qwen3VlLayer::Vision(block)) => {
                let Some(state) = context.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let output =
                    self.vision
                        .forward_block(block, index, hidden.clone(), state, stream)?;
                self.vision
                    .capture_deepstack(index, &output, state, stream)?;
                Ok(output)
            }
            (1, Qwen3VlLayer::Text(block)) => {
                let mut output = if self.sparse_expert_cache {
                    let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Qwen3-VL sparse expert cache was not initialized".into(),
                        )
                    })?;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    block.forward_sparse_experts_with_rotary(
                        AttentionInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: cache.kv[index].as_mut(),
                        },
                        &context.cos,
                        &context.sin,
                        stream,
                        |flat, indices, weights, stream| {
                            let acquired = expert_cache
                                .acquire_routes(index, indices, pass, stream)
                                .map_err(|error| Exception::custom(error.to_string()))?;
                            let started = Instant::now();
                            let prefix = format!("model.language_model.layers.{index}.mlp.experts");
                            let args = &self.args.text_config;
                            let mut bank = QwenExperts::new(
                                acquired.identities().len() as i32,
                                args.hidden_size,
                                args.moe_intermediate_size,
                                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                                stream,
                            )?;
                            bank.gate_up_proj = Param::new(
                                acquired
                                    .compact_binding("gate_up_proj", stream)
                                    .map_err(|error| Exception::custom(error.to_string()))?,
                            );
                            bank.gate_up_proj_scales = Param::new(
                                acquired
                                    .optional_compact_binding("gate_up_proj_scales", stream)
                                    .map_err(|error| Exception::custom(error.to_string()))?,
                            );
                            bank.gate_up_proj_biases = Param::new(
                                acquired
                                    .optional_compact_binding("gate_up_proj_biases", stream)
                                    .map_err(|error| Exception::custom(error.to_string()))?,
                            );
                            bank.down_proj = Param::new(
                                acquired
                                    .compact_binding("down_proj", stream)
                                    .map_err(|error| Exception::custom(error.to_string()))?,
                            );
                            bank.down_proj_scales = Param::new(
                                acquired
                                    .optional_compact_binding("down_proj_scales", stream)
                                    .map_err(|error| Exception::custom(error.to_string()))?,
                            );
                            bank.down_proj_biases = Param::new(
                                acquired
                                    .optional_compact_binding("down_proj_biases", stream)
                                    .map_err(|error| Exception::custom(error.to_string()))?,
                            );
                            expert_cache
                                .record_compact_bank(
                                    pass,
                                    acquired.scratch_bytes(),
                                    started.elapsed(),
                                )
                                .map_err(|error| Exception::custom(error.to_string()))?;
                            let output =
                                bank.forward(flat, acquired.compact_routes(), weights, stream)?;
                            eval([&output])?;
                            acquired
                                .complete_pending()
                                .map_err(|error| Exception::custom(error.to_string()))?;
                            Ok(output)
                        },
                    )?
                } else {
                    block.forward_with_rotary_embeddings(
                        AttentionInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: cache.kv[index].as_mut(),
                        },
                        &context.cos,
                        &context.sin,
                        stream,
                    )?
                };
                if let Some(features) = context.deepstack_features.get(index) {
                    let base = zeros_dtype(output.shape(), output.dtype(), stream)?;
                    let features = features.try_index_device((0, .., ..), stream)?;
                    let aligned = masked_scatter(
                        &base,
                        context.visual_mask.as_ref().expect("DeepStack visual mask"),
                        features,
                        stream,
                    )?;
                    output = output.add(aligned, stream)?;
                }
                Ok(output)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL execution unit does not match group {group}"
            ))),
        }
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        if group == 1 {
            cache.kv[index]
                .as_ref()
                .map(KeyValueCache::retained_arrays)
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    fn retained_context_arrays<'a>(
        &self,
        context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Vec<&'a Array> {
        context
            .vision
            .as_ref()
            .map(QwenVisionLayerwiseState::retained_arrays)
            .unwrap_or_default()
            .into_iter()
            .chain(context.deepstack_features.iter())
            .collect()
    }

    fn finish_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if group != 0 {
            return Ok(hidden.clone());
        }
        let hidden = if let Some(mut state) = context.vision.take() {
            let output = self.vision.finish(hidden, &mut state, stream)?;
            context.deepstack_features = output.deepstack_features;
            let mut visual_offset = 0;
            let mut assembled = Vec::with_capacity(context.parts.len());
            for part in &context.parts {
                match part {
                    PreparedPart::Text(embedding) => assembled.push(embedding.clone()),
                    PreparedPart::Visual(len) => {
                        assembled.push(output.embeddings.try_index_device(
                            (.., visual_offset..visual_offset + *len, ..),
                            stream,
                        )?);
                        visual_offset += *len;
                    }
                }
            }
            let refs = assembled.iter().collect::<Vec<_>>();
            concatenate_axis(&refs, 1, stream)?
        } else {
            hidden.clone()
        };
        context.mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Some(mask),
            Some(AttentionMask::Causal) => {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen3-VL layerwise execution requires an explicit causal mask".into(),
                ));
            }
            None => None,
        };
        context.visual_mask = if context.deepstack_features.is_empty() {
            None
        } else {
            Some(
                context
                    .tokens
                    .eq(Array::from_int(self.args.image_token_id as i32), stream)?
                    .logical_or(
                        &context
                            .tokens
                            .eq(Array::from_int(self.args.video_token_id as i32), stream)?,
                        stream,
                    )?,
            )
        };
        Ok(hidden)
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(common::linear::project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embedding,
            &hidden,
            stream,
        )?)
    }
}

/// Qwen3-VL generation using shared vision/text layerwise-host execution.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, Qwen3VlLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{ones_dtype, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::*;
    use crate::{
        expert_cache::ExpertCacheLoadOptions, layerwise::LayerwiseLoadOptions,
        models::qwen3_vl as eager, offload::OffloadConfig,
    };

    fn config(moe: bool) -> serde_json::Value {
        let model_type = if moe { "qwen3_vl_moe" } else { "qwen3_vl" };
        serde_json::json!({
            "model_type": model_type,
            "image_token_id": 30,
            "video_token_id": 31,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": format!("{model_type}_text"),
                "hidden_size": 12,
                "num_hidden_layers": 2,
                "intermediate_size": 24,
                "num_attention_heads": 1,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "head_dim": 12,
                "tie_word_embeddings": true,
                "moe_intermediate_size": if moe { 8 } else { 0 },
                "num_experts": if moe { 4 } else { 0 },
                "num_experts_per_tok": if moe { 2 } else { 0 },
                "norm_topk_prob": moe,
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_interleaved": true,
                    "mrope_section": [2, 2, 2]
                }
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 8,
                "hidden_act": "gelu_pytorch_tanh",
                "intermediate_size": 16,
                "num_heads": 2,
                "num_position_embeddings": 16,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "window_size": 8,
                "out_hidden_size": 12,
                "fullatt_block_indexes": [1],
                "deepstack_visual_indexes": [0]
            }
        })
    }

    fn initialize(model: &mut eager::Model, stream: &Stream) {
        let mut names = model
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        names.sort();
        let mut params = model.parameters_mut().flatten();
        for (index, name) in names.iter().enumerate() {
            let parameter = params.get_mut(name.as_str()).unwrap();
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else if name.ends_with(".bias") {
                zeros_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.0002 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &eager::Model, moe: bool) {
        let arrays = model
            .parameters()
            .flatten()
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
            serde_json::to_vec(&config(moe)).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 5e-5, "{left} != {right}");
        }
    }

    fn parity(moe: bool, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let config_dir = tempfile::tempdir().unwrap();
        fs::write(
            config_dir.path().join("config.json"),
            serde_json::to_vec(&config(moe)).unwrap(),
        )
        .unwrap();
        let args = eager::get_qwen3_vl_model_args(config_dir.path()).unwrap();
        let mut fixture = eager::Model::new(args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, moe);

        let mut resident =
            eager::load_qwen3_vl_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_qwen3_vl_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let before = Array::from_slice(&[1u32], &[1, 1]);
        let after = Array::from_slice(&[2u32], &[1, 1]);
        let pixels = Array::from_slice(&[0.01f32; 96], &[4, 24]);
        let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let parts = [
            input::InputPart::text_token_ids(&before),
            input::InputPart::image_tensor(&pixels, input::InputMetadata::qwen_grid_thw(&grid)),
            input::InputPart::text_token_ids(&after),
        ];
        let prompt = input::ModelInput::new(&parts);
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        let expected = resident
            .prefill_input_logits(prompt, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = layerwise
            .prefill_input_logits(prompt, &mut layerwise_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected);
        for token in [3u32, 4u32] {
            let token = Array::from_slice(&[token], &[1, 1]);
            let expected = resident
                .decode_logits(&token, &mut resident_cache, gpu.stream())
                .unwrap();
            let actual = layerwise
                .decode_logits(&token, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        let report = layerwise.residency_report().unwrap();
        for prefix in ["qwen3_vl.vision.", "qwen3_vl.text."] {
            let resident = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with(prefix))
                .filter(|unit| unit.device_resident())
                .count();
            assert!(resident <= depth);
        }
    }

    #[test]
    fn qwen3_vl_dense_multimodal_and_decode_parity() {
        parity(false, 1);
        parity(false, 2);
    }

    #[test]
    fn qwen3_vl_moe_multimodal_and_decode_parity() {
        parity(true, 1);
    }

    #[test]
    fn qwen3_vl_moe_sparse_expert_cache_multimodal_and_decode_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let config_dir = tempfile::tempdir().unwrap();
        fs::write(
            config_dir.path().join("config.json"),
            serde_json::to_vec(&config(true)).unwrap(),
        )
        .unwrap();
        let args = eager::get_qwen3_vl_model_args(config_dir.path()).unwrap();
        let mut fixture = eager::Model::new(args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, true);
        let mut resident =
            eager::load_qwen3_vl_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
        )
        .unwrap();
        let mut cached = load_qwen3_vl_sparse_expert_cache_model(
            dir.path(),
            options,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let before = Array::from_slice(&[1u32], &[1, 1]);
        let after = Array::from_slice(&[2u32], &[1, 1]);
        let pixels = Array::from_slice(&[0.01f32; 96], &[4, 24]);
        let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let parts = [
            input::InputPart::text_token_ids(&before),
            input::InputPart::image_tensor(&pixels, input::InputMetadata::qwen_grid_thw(&grid)),
            input::InputPart::text_token_ids(&after),
        ];
        let prompt = input::ModelInput::new(&parts);
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = cached.new_cache();
        let expected = resident
            .prefill_input_logits(prompt, &mut resident_cache, gpu.stream())
            .unwrap();
        let prompt = input::ModelInput::new(&parts);
        let actual = cached
            .prefill_input_logits(prompt, &mut cached_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected);
        let token = Array::from_slice(&[3u32], &[1, 1]);
        let expected = resident
            .decode_logits(&token, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = cached
            .decode_logits(&token, &mut cached_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected);
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 8);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        crate::expert_parallel::assert_rank_owned_sparse_ep_load(
            dir.path(),
            options,
            crate::models::ModelKind::Qwen3VlMoe,
            report.owned_experts / 2,
            gpu.stream(),
            cpu.stream(),
        );
    }
}
