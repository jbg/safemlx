//! Shared bounded layer execution for Qwen3-Next and Qwen3.5 text models.

use std::{collections::BTreeMap, path::Path, time::Instant};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    ops::{concatenate_axis, indexing::TryIndexOp},
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Stream,
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
        LayerExecutionLoadOptions, LayerwiseForwardState, StaticUnitBindings,
    },
    models::{
        common::{self, generation::CausalLm, linear::project_logits_maybe_quantized},
        input,
        qwen3_5_moe::{
            self as resident, BlockInput, Cache, Experts, LayerCache, LayerType, ModelArgs,
            MtpModule, Qwen3NextRmsNorm, QwenMtpStepOutput, QwenWeightFormat, TransformerBlock,
        },
        qwen3_next,
        qwen_vl::{
            grid_thw_from_array, QwenVisionBlock, QwenVisionLayerwiseState,
            QwenVisionLayerwiseStatic, QwenVisionTransformer, VisionConfig,
        },
    },
    module_binding::{
        build_module_bindings_with_recipes, canonical_checkpoint_name, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    residency::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
    utils::{create_attention_mask, AttentionMask},
    weight_recipe::DerivedWeightRecipe,
    weight_store::{SafetensorsWeightStore, TensorSelection, WeightStore},
};

const EMBEDDING_UNIT: &str = "qwen_hybrid.static.embedding";
const NORM_UNIT: &str = "qwen_hybrid.static.norm";
const HEAD_UNIT: &str = "qwen_hybrid.static.output";
const VISION_STATIC_UNIT: &str = "qwen_hybrid.static.vision";
const MTP_STATIC_UNIT: &str = "qwen_hybrid.static.mtp";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum QwenHybridFamily {
    Qwen3Next,
    Qwen35,
}

/// Qwen3-Next or Qwen3.5 text model using host-backed hybrid blocks.
pub struct QwenHybridLayerwiseModel {
    execution: GeneralLayerwiseModel<QwenHybridLayerwiseAdapter>,
}

impl QwenHybridLayerwiseModel {
    /// Returns normalized text-model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates heterogeneous recurrent/full-attention cache state.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
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

    /// Runs the shared hybrid decoder while preserving recurrent and KV state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(QwenHybridInput::Decode(inputs), cache, stream)
    }

    pub(crate) fn prefill_mtp(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        cache.reset();
        self.forward_mtp(QwenHybridInput::Prefill(input), cache, stream)
    }

    pub(crate) fn verify_mtp(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.forward_mtp(QwenHybridInput::Decode(tokens), cache, stream)
    }

    fn forward_mtp(
        &mut self,
        input: QwenHybridInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("Qwen layerwise pass did not retain MTP hidden state")
        })?;
        Ok(QwenMtpStepOutput { logits, hidden })
    }

    pub(crate) fn forward_mtp_head(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .adapter_mut()
            .forward_mtp_head(hidden, tokens, cache, stream)
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.execution
            .adapter()
            .mtp
            .as_ref()
            .map_or(0, MtpModule::len)
    }

    /// Clears temporary vision and decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_all_device_groups()
    }
}

impl CausalLm<Cache> for QwenHybridLayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .forward(QwenHybridInput::Prefill(input), cache, stream)
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

/// Loads a text-only Qwen3-Next model through bounded layer residency.
pub fn load_qwen3_next_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = qwen3_next::get_qwen3_next_model_args(model_dir)?;
    if let Some(config) = &args.quantization_config {
        config.validate_supported()?;
    }
    load_qwen_hybrid_layerwise_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen3Next,
        options,
        stream,
        weights_stream,
    )
}

/// Loads a text-only or multimodal dense/MoE Qwen3.5 model through bounded residency.
pub fn load_qwen35_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id, vision) =
        resident::get_qwen3_5_moe_model_args(model_dir)?;
    load_qwen_hybrid_layerwise_model_with_vision(
        model_dir,
        args,
        QwenHybridFamily::Qwen35,
        image_token_id,
        video_token_id,
        vision,
        options,
        stream,
        weights_stream,
    )
}

/// Loads Qwen3-Next with expert-granular sparse caching.
pub fn load_qwen3_next_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = qwen3_next::get_qwen3_next_model_args(model_dir)?;
    if let Some(config) = &args.quantization_config {
        config.validate_supported()?;
    }
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3-Next MoE checkpoint".into(),
        ));
    }
    load_qwen_hybrid_sparse_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen3Next,
        None,
        None,
        None,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Qwen3-Next with expert caching and disk-streamed non-expert units.
pub fn load_qwen3_next_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = qwen3_next::get_qwen3_next_model_args(model_dir)?;
    if let Some(config) = &args.quantization_config {
        config.validate_supported()?;
    }
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3-Next MoE checkpoint".into(),
        ));
    }
    load_qwen_hybrid_sparse_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen3Next,
        None,
        None,
        None,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Qwen3.5 MoE with expert-granular sparse caching.
pub fn load_qwen35_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id, vision) =
        resident::get_qwen3_5_moe_model_args(model_dir)?;
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3.5 MoE checkpoint".into(),
        ));
    }
    load_qwen_hybrid_sparse_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen35,
        image_token_id,
        video_token_id,
        vision,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Qwen3.5 MoE with expert caching and disk-streamed non-expert units.
pub fn load_qwen35_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id, vision) =
        resident::get_qwen3_5_moe_model_args(model_dir)?;
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3.5 MoE checkpoint".into(),
        ));
    }
    load_qwen_hybrid_sparse_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen35,
        image_token_id,
        video_token_id,
        vision,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_hybrid_sparse_model(
    model_dir: &Path,
    args: ModelArgs,
    family: QwenHybridFamily,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    vision_config: Option<VisionConfig>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let mut adapter = QwenHybridLayerwiseAdapter::new(
        args.clone(),
        family,
        image_token_id,
        video_token_id,
        vision_config,
        stream,
    )?;
    adapter.sparse_expert_cache = true;
    let mut execution =
        load_general_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.weight_store_arc();
    let entries = qwen_hybrid_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(QwenHybridLayerwiseModel { execution })
}

fn load_qwen_hybrid_layerwise_model(
    model_dir: &Path,
    args: ModelArgs,
    family: QwenHybridFamily,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    load_qwen_hybrid_layerwise_model_with_vision(
        model_dir,
        args,
        family,
        None,
        None,
        None,
        options,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_hybrid_layerwise_model_with_vision(
    model_dir: &Path,
    args: ModelArgs,
    family: QwenHybridFamily,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    vision_config: Option<VisionConfig>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let adapter = QwenHybridLayerwiseAdapter::new(
        args,
        family,
        image_token_id,
        video_token_id,
        vision_config,
        stream,
    )?;
    Ok(QwenHybridLayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Shared adapter for recurrent linear-attention and full-attention Qwen blocks.
pub struct QwenHybridLayerwiseAdapter {
    args: ModelArgs,
    family: QwenHybridFamily,
    embedding: MaybeQuantized<safemlx::nn::Embedding>,
    norm: Qwen3NextRmsNorm,
    lm_head: Option<MaybeQuantized<safemlx::nn::Linear>>,
    mtp: Option<MtpModule>,
    vision: Option<QwenVisionLayerwiseStatic>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl QwenHybridLayerwiseAdapter {
    fn new(
        args: ModelArgs,
        family: QwenHybridFamily,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_config: Option<VisionConfig>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.quantization,
            stream,
        )?;
        let norm = Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.quantization,
                    stream,
                )?,
            )
        };
        let mtp = (args.mtp_num_hidden_layers > 0)
            .then(|| {
                MtpModule::new_with_format(&args, QwenWeightFormat::for_text(&args, None), stream)
            })
            .transpose()?;
        let vision = vision_config
            .map(|config| QwenVisionTransformer::new(config, stream))
            .transpose()?
            .map(QwenVisionLayerwiseStatic::from_transformer);
        Ok(Self {
            args,
            family,
            embedding,
            norm,
            lm_head,
            mtp,
            vision,
            image_token_id,
            video_token_id,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    /// Returns normalized text-model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }

    fn forward_mtp_head(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embeddings = self.embedding.forward(tokens, stream)?;
        let hidden = self
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?
            .forward(hidden, &embeddings, cache, stream)?;
        project_logits_maybe_quantized(&mut self.lm_head, &mut self.embedding, &hidden, stream)
    }

    fn recipes_for_module(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
        layer_index: Option<usize>,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.keys();
        let mut recipes = BTreeMap::new();

        if let Some(index) = layer_index {
            if self.family == QwenHybridFamily::Qwen3Next
                && self.args.layer_type(index) == LayerType::LinearAttention
            {
                add_fused_projection_recipes(&mut recipes, &normalized, index, &self.args)?;
            }
            if self.args.is_moe() {
                add_expert_recipes(
                    &mut recipes,
                    &normalized,
                    index,
                    &self.args,
                    self.args.uses_fp8(),
                )?;
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
                        "Qwen hybrid checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(recipes)
    }

    fn mtp_recipes(
        &self,
        store: &dyn WeightStore,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen hybrid model has no MTP module".into())
        })?;
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.keys();
        let mut recipes = BTreeMap::new();
        if self.args.is_moe() {
            for index in 0..self.args.mtp_num_hidden_layers as usize {
                add_expert_recipes_for_prefix(
                    &mut recipes,
                    &normalized,
                    &format!("mtp.layers.{index}.mlp.experts"),
                    &format!("layers.{index}.mlp"),
                    &self.args,
                    self.args.uses_fp8(),
                )?;
            }
        }
        for local_name in mtp.parameters().flatten().keys() {
            if recipes.contains_key(local_name.as_ref()) {
                continue;
            }
            let destination = format!("mtp.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Qwen hybrid checkpoint is missing runtime parameter {canonical}"
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
        .map(|raw| {
            let runtime = ["model.language_model.", "language_model.", "model.model."]
                .into_iter()
                .find_map(|prefix| raw.strip_prefix(prefix).map(|rest| format!("model.{rest}")))
                .or_else(|| {
                    ["model.vision_tower.", "model.visual.", "vision_tower."]
                        .into_iter()
                        .find_map(|prefix| {
                            raw.strip_prefix(prefix)
                                .map(|rest| format!("visual.{rest}"))
                        })
                })
                .unwrap_or_else(|| raw.clone())
                .replace("visual.merger.mlp.0.", "visual.merger.mlp.fc1.")
                .replace("visual.merger.mlp.2.", "visual.merger.mlp.fc2.");
            (runtime, raw)
        })
        .collect()
}

fn add_fused_projection_recipes(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    normalized: &BTreeMap<String, String>,
    index: usize,
    args: &ModelArgs,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}.linear_attn");
    let (qkvz_widths, ba_width) = qwen3_next::fused_projection_widths(args)?;
    for suffix in ["weight", "scales", "biases"] {
        let qkvz_runtime = format!("{prefix}.in_proj_qkvz.{suffix}");
        if let Some(raw) = normalized.get(&qkvz_runtime) {
            for (local, components) in [
                (format!("linear_attn.in_proj_qkv.{suffix}"), vec![0, 1, 2]),
                (format!("linear_attn.in_proj_z.{suffix}"), vec![3]),
            ] {
                recipes.insert(
                    local,
                    DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Indices {
                            axis: 0,
                            indices: grouped_component_indices(
                                self::usize_from_i32(args.linear_num_key_heads)?,
                                &qkvz_widths,
                                &components,
                            )?,
                        },
                    ),
                );
            }
        }
        let ba_runtime = format!("{prefix}.in_proj_ba.{suffix}");
        if let Some(raw) = normalized.get(&ba_runtime) {
            for (local, component) in [
                (format!("linear_attn.in_proj_b.{suffix}"), 0),
                (format!("linear_attn.in_proj_a.{suffix}"), 1),
            ] {
                recipes.insert(
                    local,
                    DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Indices {
                            axis: 0,
                            indices: grouped_component_indices(
                                usize_from_i32(args.linear_num_key_heads)?,
                                &[ba_width, ba_width],
                                &[component],
                            )?,
                        },
                    ),
                );
            }
        }
    }
    if args.uses_fp8() {
        let block_widths = qwen3_next::fp8_block_row_widths(&qkvz_widths)?;
        let qkvz_runtime = format!("{prefix}.in_proj_qkvz.weight_scale_inv");
        if let Some(raw) = normalized.get(&qkvz_runtime) {
            for (local, components) in [
                (
                    "linear_attn.in_proj_qkv.weight_scale_inv".to_string(),
                    vec![0, 1, 2],
                ),
                (
                    "linear_attn.in_proj_z.weight_scale_inv".to_string(),
                    vec![3],
                ),
            ] {
                recipes.insert(
                    local,
                    DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Indices {
                            axis: 0,
                            indices: grouped_component_indices(
                                usize_from_i32(args.linear_num_key_heads)?,
                                &block_widths,
                                &components,
                            )?,
                        },
                    ),
                );
            }
        }
        if normalized.contains_key(&format!("{prefix}.in_proj_ba.weight_scale_inv")) {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3-Next in_proj_ba must remain dense BF16 and cannot carry FP8 inverse scales"
                    .into(),
            ));
        }
    }
    Ok(())
}

fn usize_from_i32(value: i32) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| {
        Error::UnsupportedArchitecture("Qwen hybrid projection dimension is invalid".into())
    })
}

fn grouped_component_indices(
    groups: usize,
    widths: &[i32],
    components: &[usize],
) -> Result<Vec<usize>, Error> {
    let widths = widths
        .iter()
        .map(|width| usize_from_i32(*width))
        .collect::<Result<Vec<_>, _>>()?;
    let group_width = widths.iter().sum::<usize>();
    let mut starts = Vec::with_capacity(widths.len());
    let mut start = 0usize;
    for width in &widths {
        starts.push(start);
        start = start.checked_add(*width).ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen hybrid projection index overflow".into())
        })?;
    }
    let mut indices = Vec::new();
    for component in components {
        let width = *widths.get(*component).ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen hybrid projection component is invalid".into())
        })?;
        for group in 0..groups {
            let base = group
                .checked_mul(group_width)
                .and_then(|base| base.checked_add(starts[*component]))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture("Qwen hybrid projection index overflow".into())
                })?;
            indices.extend(base..base + width);
        }
    }
    Ok(indices)
}

fn add_expert_recipes(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    normalized: &BTreeMap<String, String>,
    index: usize,
    args: &ModelArgs,
    fp8: bool,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}.mlp.experts");
    add_expert_recipes_for_prefix(recipes, normalized, &prefix, "mlp", args, fp8)
}

fn add_expert_recipes_for_prefix(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    local_prefix: &str,
    args: &ModelArgs,
    fp8: bool,
) -> Result<(), Error> {
    if normalized.contains_key(&format!("{prefix}.gate_up_proj")) {
        return Ok(());
    }
    let mut gate_up = Vec::with_capacity(args.num_experts as usize);
    let mut down = Vec::with_capacity(args.num_experts as usize);
    let mut gate_up_scale = Vec::new();
    let mut down_scale = Vec::new();
    for expert in 0..args.num_experts {
        let gate = expert_source(normalized, prefix, expert, &["gate_proj", "w1"], "weight")?;
        let up = expert_source(normalized, prefix, expert, &["up_proj", "w3"], "weight")?;
        down.push(expert_source(
            normalized,
            prefix,
            expert,
            &["down_proj", "w2"],
            "weight",
        )?);
        gate_up.push(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![gate, up],
        });
        if fp8 {
            let gate_scale = expert_source(
                normalized,
                prefix,
                expert,
                &["gate_proj"],
                "weight_scale_inv",
            )?;
            let up_scale =
                expert_source(normalized, prefix, expert, &["up_proj"], "weight_scale_inv")?;
            gate_up_scale.push(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![gate_scale, up_scale],
            });
            down_scale.push(expert_source(
                normalized,
                prefix,
                expert,
                &["down_proj"],
                "weight_scale_inv",
            )?);
        }
    }
    recipes.insert(
        format!("{local_prefix}.experts.gate_up_proj"),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: gate_up,
        },
    );
    recipes.insert(
        format!("{local_prefix}.experts.down_proj"),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: down,
        },
    );
    if fp8 {
        recipes.insert(
            format!("{local_prefix}.experts.gate_up_proj_scale_inv"),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: gate_up_scale,
            },
        );
        recipes.insert(
            format!("{local_prefix}.experts.down_proj_scale_inv"),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: down_scale,
            },
        );
    }
    Ok(())
}

fn expert_source(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: i32,
    projections: &[&str],
    suffix: &str,
) -> Result<DerivedWeightRecipe, Error> {
    projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.{suffix}"))
        .find_map(|runtime| normalized.get(&runtime).cloned())
        .map(|raw| DerivedWeightRecipe::source(raw, TensorSelection::Full))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen hybrid checkpoint is missing expert {expert} tensor under {prefix}"
            ))
        })
}

pub(crate) fn qwen_hybrid_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let packed = normalized.contains_key(&format!("{prefix}.gate_up_proj"));
        for expert in 0..args.num_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            if packed {
                for (name, required) in [
                    ("gate_up_proj", true),
                    ("gate_up_proj_scale_inv", false),
                    ("gate_up_proj_scales", false),
                    ("gate_up_proj_biases", false),
                    ("down_proj", true),
                    ("down_proj_scale_inv", false),
                    ("down_proj_scales", false),
                    ("down_proj_biases", false),
                ] {
                    let runtime = format!("{prefix}.{name}");
                    let Some(raw) = normalized.get(&runtime) else {
                        if required {
                            return Err(Error::UnsupportedArchitecture(format!(
                                "Qwen hybrid checkpoint is missing packed expert tensor {runtime}"
                            )));
                        }
                        continue;
                    };
                    bindings.push(qwen_hybrid_recipe_binding(
                        name,
                        DerivedWeightRecipe::source(
                            raw.clone(),
                            TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            },
                        ),
                        store,
                    )?);
                }
            } else {
                let gate = expert_source(
                    &normalized,
                    &prefix,
                    expert as i32,
                    &["gate_proj", "w1"],
                    "weight",
                )?;
                let up = expert_source(
                    &normalized,
                    &prefix,
                    expert as i32,
                    &["up_proj", "w3"],
                    "weight",
                )?;
                let down = expert_source(
                    &normalized,
                    &prefix,
                    expert as i32,
                    &["down_proj", "w2"],
                    "weight",
                )?;
                bindings.push(qwen_hybrid_recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![gate, up],
                        }],
                    },
                    store,
                )?);
                bindings.push(qwen_hybrid_recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![down],
                    },
                    store,
                )?);
                if args.uses_fp8() {
                    let gate = expert_source(
                        &normalized,
                        &prefix,
                        expert as i32,
                        &["gate_proj"],
                        "weight_scale_inv",
                    )?;
                    let up = expert_source(
                        &normalized,
                        &prefix,
                        expert as i32,
                        &["up_proj"],
                        "weight_scale_inv",
                    )?;
                    let down = expert_source(
                        &normalized,
                        &prefix,
                        expert as i32,
                        &["down_proj"],
                        "weight_scale_inv",
                    )?;
                    bindings.push(qwen_hybrid_recipe_binding(
                        "gate_up_proj_scale_inv",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![gate, up],
                            }],
                        },
                        store,
                    )?);
                    bindings.push(qwen_hybrid_recipe_binding(
                        "down_proj_scale_inv",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![down],
                        },
                        store,
                    )?);
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Qwen hybrid expert byte total overflowed".into(),
                    )
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

fn qwen_hybrid_recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn WeightStore,
) -> Result<WeightBinding, Error> {
    let bytes = recipe.infer(store)?.byte_len();
    Ok(WeightBinding::from_recipe(name, recipe, bytes)?)
}

/// Input mode for typed prefill and cached text decode.
pub enum QwenHybridInput<'a> {
    /// Ordered text and visual prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text tokens for cached decode.
    Decode(&'a Array),
}

enum QwenHybridPreparedPart {
    Ready(Array),
    Vision(usize),
}

struct QwenHybridVisionJob {
    hidden: Array,
    state: QwenVisionLayerwiseState,
}

/// Per-forward vision assembly and causal mask state.
pub struct QwenHybridForwardContext {
    mask: Option<Array>,
    parts: Vec<QwenHybridPreparedPart>,
    vision_jobs: Vec<QwenHybridVisionJob>,
    needs_assembly: bool,
    draft_hidden: Option<Array>,
}

/// One leased vision or hybrid text block.
pub enum QwenHybridLayer {
    /// Qwen vision transformer block.
    Vision(Box<QwenVisionBlock>),
    /// Qwen recurrent or full-attention text block.
    Text(Box<TransformerBlock>),
}

impl ModuleParameters for QwenHybridLayer {
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

impl GeneralLayerwiseModelAdapter for QwenHybridLayerwiseAdapter {
    type Input<'a> = QwenHybridInput<'a>;
    type Cache = Cache;
    type Layer = QwenHybridLayer;
    type ForwardContext = QwenHybridForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    self.recipes_for_module(&self.embedding, "model.embed_tokens", store, None)?,
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    self.recipes_for_module(&self.norm, "model.norm", store, None)?,
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
        if let Some(mtp) = &self.mtp {
            units.push(StaticUnitBindings::new(
                MTP_STATIC_UNIT,
                build_module_bindings_with_recipes(mtp, "mtp", store, self.mtp_recipes(store)?)?,
            )?);
        }
        if let Some(vision) = &self.vision {
            units.push(StaticUnitBindings::new(
                VISION_STATIC_UNIT,
                build_module_bindings_with_recipes(
                    vision,
                    "visual",
                    store,
                    self.recipes_for_module(vision, "visual", store, None)?,
                )?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = 2
            + usize::from(self.lm_head.is_some())
            + usize::from(self.mtp.is_some())
            + usize::from(self.vision.is_some());
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.embedding, &leases[0])?;
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        let mut index = 2;
        if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[index])?;
            index += 1;
        }
        if let Some(mtp) = &mut self.mtp {
            populate_module_from_lease(mtp, &leases[index])?;
            index += 1;
        }
        if let Some(vision) = &mut self.vision {
            populate_module_from_lease(vision, &leases[index])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.num_hidden_layers
            )));
        }
        for (index, cache) in cache.layers.iter().enumerate() {
            let matches = matches!(
                (self.args.layer_type(index), cache),
                (LayerType::FullAttention, LayerCache::FullAttention(_))
                    | (LayerType::LinearAttention, LayerCache::LinearAttention(_))
            );
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen hybrid cache kind does not match layer_types at layer {index}"
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
        let (hidden, parts, vision_jobs, needs_assembly) = match input {
            QwenHybridInput::Decode(tokens) => (
                self.embedding.forward(tokens, stream)?,
                Vec::new(),
                Vec::new(),
                false,
            ),
            QwenHybridInput::Prefill(typed) => {
                input::validate(typed)?;
                let mut parts = Vec::with_capacity(typed.parts.len());
                let mut jobs = Vec::new();
                for part in typed.parts {
                    match (part.modality, part.payload) {
                        (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                            parts.push(QwenHybridPreparedPart::Ready(
                                self.embedding.forward(tokens, stream)?,
                            ));
                        }
                        (
                            input::Modality::Image | input::Modality::Video,
                            input::InputPayload::Tensor(pixels),
                        ) => {
                            let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                                Error::UnsupportedArchitecture(format!(
                                    "Qwen3.5 {} input requires qwen_grid_thw metadata",
                                    part.modality.as_str()
                                ))
                            })?;
                            let vision = self.vision.as_mut().ok_or_else(|| {
                                Error::UnsupportedArchitecture(
                                    "Qwen3.5 visual tensor input requires vision_config and visual weights".into(),
                                )
                            })?;
                            let token_id = if part.modality == input::Modality::Image {
                                self.image_token_id
                            } else {
                                self.video_token_id
                            };
                            if token_id.is_none() {
                                return Err(Error::UnsupportedArchitecture(format!(
                                    "Qwen3.5 config does not define a {} token ID",
                                    part.modality.as_str()
                                )));
                            }
                            let merge = vision.config.spatial_merge_size;
                            let merged = grid_thw_from_array(grid, stream)?
                                .into_iter()
                                .map(|(t, h, w)| t * (h / merge) * (w / merge))
                                .sum::<i32>();
                            if merged <= 0 {
                                return Err(Error::UnsupportedArchitecture(
                                    "Qwen3.5 visual grid produced no merged tokens".into(),
                                ));
                            }
                            let (hidden, state) = vision.begin(pixels, grid, stream)?;
                            let job = jobs.len();
                            jobs.push(QwenHybridVisionJob { hidden, state });
                            parts.push(QwenHybridPreparedPart::Vision(job));
                        }
                        (
                            input::Modality::Image | input::Modality::Video,
                            input::InputPayload::Embeddings(embeddings),
                        ) => {
                            input::ensure_hidden_size(
                                embeddings,
                                self.args.hidden_size,
                                "Qwen3.5 visual embeddings",
                            )?;
                            parts.push(QwenHybridPreparedPart::Ready(embeddings.clone()));
                        }
                        (modality, _) => {
                            return Err(Error::UnsupportedArchitecture(format!(
                                "Qwen3.5 layerwise input does not support {} payloads of this kind",
                                modality.as_str()
                            )));
                        }
                    }
                }
                if jobs.is_empty() {
                    let ready = parts
                        .iter()
                        .map(|part| match part {
                            QwenHybridPreparedPart::Ready(value) => value,
                            QwenHybridPreparedPart::Vision(_) => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    (concatenate_axis(&ready, 1, stream)?, parts, jobs, false)
                } else {
                    (jobs[0].hidden.clone(), parts, jobs, true)
                }
            }
        };
        let mask = if !needs_assembly && hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Qwen hybrid decoder requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: QwenHybridForwardContext {
                mask,
                parts,
                vision_jobs,
                needs_assembly,
                draft_hidden: None,
            },
        })
    }

    fn execution_group_count(&self) -> usize {
        1 + usize::from(self.vision.is_some())
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        match (self.vision.is_some(), group) {
            (true, 0) => Ok("vision_encoder".into()),
            (true, 1) | (false, 0) => Ok("text_decoder".into()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid model has no execution group {group}"
            ))),
        }
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        self.execution_group_id(group)
            .is_ok_and(|id| id != "vision_encoder" || !context.vision_jobs.is_empty())
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match self.execution_group_id(group)?.as_str() {
            "vision_encoder" => {
                Ok(self.vision.as_ref().expect("vision group").config.depth as usize)
            }
            "text_decoder" => Ok(self.args.num_hidden_layers as usize),
            _ => unreachable!(),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        if self.execution_group_id(group)? == "vision_encoder" {
            Ok(QwenHybridLayer::Vision(Box::new(QwenVisionBlock::new(
                &self.vision.as_ref().expect("vision group").config,
                stream,
            )?)))
        } else {
            Ok(QwenHybridLayer::Text(Box::new(TransformerBlock::new(
                &self.args, index, stream,
            )?)))
        }
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if self.execution_group_id(group).ok().as_deref() == Some("vision_encoder") {
            format!("visual.blocks.{index}")
        } else {
            format!("model.layers.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if self.execution_group_id(group).ok().as_deref() == Some("vision_encoder") {
            format!("qwen_hybrid.vision.{index:05}")
        } else {
            format!("qwen_hybrid.layer.{index:05}")
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = self.layer_checkpoint_prefix(group, index);
        let bindings = build_module_bindings_with_recipes(
            layer,
            &prefix,
            store,
            self.recipes_for_module(
                layer,
                &prefix,
                store,
                (self.execution_group_id(group)? == "text_decoder").then_some(index),
            )?,
        )?;
        Ok(
            if self.sparse_expert_cache && self.execution_group_id(group)? == "text_decoder" {
                bindings
                    .into_iter()
                    .filter(|binding| !binding.name().starts_with("mlp.experts."))
                    .collect()
            } else {
                bindings
            },
        )
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
        match (self.execution_group_id(group)?.as_str(), layer) {
            ("vision_encoder", QwenHybridLayer::Vision(block)) => {
                let vision = self.vision.as_mut().expect("vision group");
                for job in &mut context.vision_jobs {
                    job.hidden = vision.forward_block(
                        block,
                        index,
                        job.hidden.clone(),
                        &job.state,
                        stream,
                    )?;
                    vision.capture_deepstack(index, &job.hidden, &mut job.state, stream)?;
                }
                Ok(context.vision_jobs[0].hidden.clone())
            }
            ("text_decoder", QwenHybridLayer::Text(block)) if self.sparse_expert_cache => {
                let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Qwen hybrid sparse expert cache was not initialized".into(),
                    )
                })?;
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                Ok(block.forward_sparse_experts(
                    BlockInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(&mut cache.layers[index]),
                    },
                    stream,
                    |flat, indices, weights, stream| {
                        let acquired = expert_cache
                            .acquire_routes(index, indices, pass, stream)
                            .map_err(|error| Exception::custom(error.to_string()))?;
                        let started = Instant::now();
                        let mut compact_args = self.args.clone();
                        compact_args.num_experts = acquired.identities().len() as i32;
                        let mut bank = Experts::new(&compact_args, index, stream)?;
                        bank.gate_up_proj = Param::new(
                            acquired
                                .compact_binding("gate_up_proj", stream)
                                .map_err(|error| Exception::custom(error.to_string()))?,
                        );
                        bank.gate_up_proj_scale_inv = Param::new(
                            acquired
                                .optional_compact_binding("gate_up_proj_scale_inv", stream)
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
                        bank.down_proj_scale_inv = Param::new(
                            acquired
                                .optional_compact_binding("down_proj_scale_inv", stream)
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
                            .record_compact_bank(pass, acquired.scratch_bytes(), started.elapsed())
                            .map_err(|error| Exception::custom(error.to_string()))?;
                        let output =
                            bank.forward_chunked(flat, acquired.compact_routes(), weights, stream)?;
                        eval([&output])?;
                        acquired
                            .complete_pending()
                            .map_err(|error| Exception::custom(error.to_string()))?;
                        Ok(output)
                    },
                )?)
            }
            ("text_decoder", QwenHybridLayer::Text(block)) => Ok(block.forward(
                BlockInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(&mut cache.layers[index]),
                },
                stream,
            )?),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid execution unit does not match group {group}"
            ))),
        }
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        if self.execution_group_id(group).ok().as_deref() == Some("text_decoder") {
            cache.layers[index].retained_arrays()
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
            .vision_jobs
            .iter()
            .flat_map(|job| std::iter::once(&job.hidden).chain(job.state.retained_arrays()))
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
        let group_id = self.execution_group_id(group)?;
        if group + 1 == self.execution_group_count() {
            context.draft_hidden = Some(hidden.clone());
        }
        let should_assemble =
            context.needs_assembly && (group_id == "vision_encoder" || self.vision.is_none());
        if !should_assemble {
            return Ok(hidden.clone());
        }
        if let Some(vision) = &mut self.vision {
            for job in &mut context.vision_jobs {
                let output = vision.finish(&job.hidden, &mut job.state, stream)?;
                job.hidden = output.embeddings;
            }
        }
        let assembled = context
            .parts
            .iter()
            .map(|part| match part {
                QwenHybridPreparedPart::Ready(value) => value,
                QwenHybridPreparedPart::Vision(job) => &context.vision_jobs[*job].hidden,
            })
            .collect::<Vec<_>>();
        let hidden = concatenate_axis(&assembled, 1, stream)?;
        context.mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Qwen hybrid decoder requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        context.needs_assembly = false;
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
        Ok(project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embedding,
            &hidden,
            stream,
        )?)
    }

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        self.vision.is_none()
            && (key.starts_with("visual.")
                || key.starts_with("vision_tower.")
                || key.starts_with("model.visual.")
                || key.starts_with("model.vision_tower."))
    }
}

/// Shared Qwen hybrid token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, QwenHybridLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::{
        load_qwen35_layerwise_model, load_qwen35_sparse_expert_cache_model,
        load_qwen3_next_layerwise_model, load_qwen3_next_sparse_expert_cache_model,
    };
    use crate::{
        expert_cache::ExpertCacheLoadOptions,
        layerwise::LayerwiseLoadOptions,
        models::{
            common::generation::CausalLm,
            input as runtime_input,
            qwen3_5_moe::{self as resident, Cache, LayerCache, Model, ModelArgs, ModelInput},
            qwen3_next,
            qwen_vl::VisionConfig,
        },
        offload::{OffloadConfig, ResidencyPolicy},
    };

    fn config(next: bool, moe: bool) -> serde_json::Value {
        serde_json::json!({
            "model_type": if next { "qwen3_next" } else if moe { "qwen3_5_moe_text" } else { "qwen3_5_text" },
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": if moe { 0 } else { 32 },
            "moe_intermediate_size": if moe { 8 } else { 0 },
            "shared_expert_intermediate_size": if moe { 8 } else { 0 },
            "num_experts_per_tok": if moe { 1 } else { 0 },
            "num_experts": if moe { 2 } else { 0 },
            "norm_topk_prob": moe,
            "layer_types": ["linear_attention", "full_attention"]
        })
    }

    fn args(next: bool, moe: bool) -> ModelArgs {
        serde_json::from_value(config(next, moe)).unwrap()
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for (_, parameter) in model.parameters_mut().flatten() {
            *parameter = zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap();
        }
    }

    fn write_fixture(dir: &Path, model: &Model, next: bool, stream: &Stream) {
        let params = model.parameters().flatten();
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in params {
            let name = crate::module_binding::canonical_checkpoint_name(&name);
            if next && name.ends_with("mlp.experts.gate_up_proj") {
                let prefix = name.trim_end_matches(".gate_up_proj");
                for expert in 0..model.args.num_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.gate_proj.weight"),
                        value
                            .try_index_device(
                                (expert, ..model.args.moe_intermediate_size, ..),
                                stream,
                            )
                            .unwrap(),
                    ));
                    arrays.push((
                        format!("{prefix}.{expert}.up_proj.weight"),
                        value
                            .try_index_device(
                                (expert, model.args.moe_intermediate_size.., ..),
                                stream,
                            )
                            .unwrap(),
                    ));
                }
                continue;
            }
            if next && name.ends_with("mlp.experts.down_proj") {
                let prefix = name.trim_end_matches(".down_proj");
                for expert in 0..model.args.num_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.down_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
                continue;
            }
            let fused_part = next
                && [
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.in_proj_a.weight",
                ]
                .iter()
                .any(|suffix| name.ends_with(suffix));
            if !fused_part {
                arrays.push((name, value.clone()));
            }
        }
        if next {
            let qkvz_rows = model.args.linear_num_key_heads
                * (2 * model.args.linear_key_head_dim
                    + 2 * model.args.linear_num_value_heads * model.args.linear_value_head_dim
                        / model.args.linear_num_key_heads);
            let ba_rows = 2 * model.args.linear_num_value_heads;
            arrays.push((
                "model.layers.0.linear_attn.in_proj_qkvz.weight".into(),
                Array::zeros::<f32>(&[qkvz_rows, model.args.hidden_size], stream).unwrap(),
            ));
            arrays.push((
                "model.layers.0.linear_attn.in_proj_ba.weight".into(),
                Array::zeros::<f32>(&[ba_rows, model.args.hidden_size], stream).unwrap(),
            ));
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config(next, model.args.is_moe())).unwrap(),
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

    fn parity(next: bool, moe: bool, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(next, moe), None, None, None, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, next, gpu.stream());

        let mut resident = if next {
            qwen3_next::load_qwen3_next_model(dir.path(), gpu.stream(), cpu.stream()).unwrap()
        } else {
            resident::load_qwen3_5_moe_model(dir.path(), gpu.stream(), cpu.stream()).unwrap()
        };
        let options = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap());
        let mut layerwise = if next {
            load_qwen3_next_layerwise_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap()
        } else {
            load_qwen35_layerwise_model(dir.path(), options, gpu.stream(), cpu.stream()).unwrap()
        };
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = Cache {
            layers: Vec::new(),
            mtp_layers: Vec::new(),
        };
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
                        inputs_embeds: None,
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
                    LayerCache::FullAttention(cache) => crate::cache::KeyValueCache::offset(cache),
                    LayerCache::LinearAttention(cache) => cache.offset,
                };
                let actual_offset = match actual {
                    LayerCache::FullAttention(cache) => crate::cache::KeyValueCache::offset(cache),
                    LayerCache::LinearAttention(cache) => cache.offset,
                };
                assert_eq!(expected_offset, actual_offset);
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("qwen_hybrid.layer."))
                .collect::<Vec<_>>();
            assert!(layers.iter().all(|unit| unit.host_resident()));
            assert!(layers.iter().filter(|unit| unit.device_resident()).count() <= depth);
            assert!(report
                .units()
                .iter()
                .filter(|unit| unit.device_resident() && !layers.contains(unit))
                .all(|unit| unit.policy() == ResidencyPolicy::Pinned));
        }

        let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [runtime_input::InputPart::text_token_ids(&prompt)];
        let mtp_config = crate::mtp::MtpConfig {
            max_tokens: 3,
            max_draft_tokens: 1,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        };
        let mut resident_cache = resident.new_cache();
        let (expected, expected_stats) = crate::qwen_mtp::generate(
            &mut resident,
            &mut resident_cache,
            runtime_input::ModelInput::new(&parts),
            &mtp_config,
            None,
            &mut crate::sampler::DefaultSampler,
            gpu.stream(),
        )
        .unwrap();
        let mut layerwise_cache = layerwise.new_cache();
        let (actual, actual_stats) = crate::qwen_mtp::generate(
            &mut layerwise,
            &mut layerwise_cache,
            runtime_input::ModelInput::new(&parts),
            &mtp_config,
            None,
            &mut crate::sampler::DefaultSampler,
            gpu.stream(),
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual_stats.rounds, expected_stats.rounds);
        assert_eq!(actual_stats.accepted_tokens, expected_stats.accepted_tokens);
    }

    #[test]
    fn qwen3_next_fused_hybrid_layerwise_parity() {
        parity(true, false, 1);
        parity(true, false, 2);
        parity(true, true, 1);
    }

    #[test]
    fn qwen35_dense_and_moe_hybrid_layerwise_parity() {
        parity(false, false, 1);
        parity(false, true, 1);
    }

    fn sparse_expert_cache_parity(next: bool) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(next, true), None, None, None, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, next, gpu.stream());
        let mut resident = if next {
            qwen3_next::load_qwen3_next_model(dir.path(), gpu.stream(), cpu.stream()).unwrap()
        } else {
            resident::load_qwen3_5_moe_model(dir.path(), gpu.stream(), cpu.stream()).unwrap()
        };
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
        )
        .unwrap();
        let mut cached = if next {
            load_qwen3_next_sparse_expert_cache_model(
                dir.path(),
                options,
                gpu.stream(),
                cpu.stream(),
            )
            .unwrap()
        } else {
            load_qwen35_sparse_expert_cache_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap()
        };
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = Cache {
            layers: Vec::new(),
            mtp_layers: Vec::new(),
        };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        inputs_embeds: None,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    false,
                    gpu.stream(),
                )
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
            if next {
                crate::models::ModelKind::Qwen3Next
            } else {
                crate::models::ModelKind::Qwen35Moe
            },
            report.owned_experts / 2,
            gpu.stream(),
            cpu.stream(),
        );
    }

    #[test]
    fn qwen_hybrid_sparse_expert_cache_prefill_and_decode_parity() {
        sparse_expert_cache_parity(true);
        sparse_expert_cache_parity(false);
    }

    #[test]
    fn qwen35_multimodal_vision_and_text_group_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let vision = VisionConfig {
            depth: 1,
            hidden_size: 8,
            hidden_act: "silu".into(),
            intermediate_size: 4,
            num_heads: 2,
            num_position_embeddings: 16,
            in_channels: 3,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            window_size: 8,
            out_hidden_size: 16,
            fullatt_block_indexes: vec![0],
            deepstack_visual_indexes: Vec::new(),
        };
        let mut fixture = Model::new(
            args(false, false),
            Some(42),
            Some(43),
            Some(vision.clone()),
            gpu.stream(),
        )
        .unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        let arrays = fixture
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| {
                (
                    crate::module_binding::canonical_checkpoint_name(name),
                    (*value).clone(),
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "qwen3_5",
                "image_token_id": 42,
                "video_token_id": 43,
                "text_config": config(false, false),
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 8,
                    "hidden_act": "silu",
                    "intermediate_size": 4,
                    "num_heads": 2,
                    "num_position_embeddings": 16,
                    "in_channels": 3,
                    "patch_size": 2,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "window_size": 8,
                    "out_hidden_size": 16,
                    "fullatt_block_indexes": [0],
                    "deepstack_visual_indexes": [],
                },
            }))
            .unwrap(),
        )
        .unwrap();

        let mut resident =
            resident::load_qwen3_5_moe_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_qwen35_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let text = runtime_input::token_ids_array(&[1, 2], gpu.stream()).unwrap();
        let grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        let pixels = Array::zeros::<f32>(&[8, 12], gpu.stream()).unwrap();
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::qwen_grid_thw(&grid),
            ),
        ];
        let typed = runtime_input::ModelInput::new(&parts);
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        let expected = resident
            .prefill_input_logits(typed, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = layerwise
            .prefill_input_logits(typed, &mut layerwise_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected);
        assert_eq!(resident_cache.offset(), layerwise_cache.offset());
        let report = layerwise.residency_report().unwrap();
        assert!(report
            .units()
            .iter()
            .any(|unit| unit.id().as_str().starts_with("qwen_hybrid.vision.")));
    }
}
