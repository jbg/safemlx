//! Text-decoder layerwise-host execution for Thinking Machines Lab Inkling.

use std::{collections::BTreeMap, path::Path, time::Instant};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{concatenate_axis, indexing::NewAxis, indexing::TryIndexOp},
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
        common::{self, generation::CausalLm, moe::PackedSwiGluExperts},
        inkling::{
            self as resident, AudioModel, Cache, DecoderLayer, ModelArgs, VisionLayer, VisionModel,
        },
        input,
    },
    module_binding::{
        build_module_bindings_with_recipes, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    residency::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
    weight_recipe::DerivedWeightRecipe,
    weight_store::{SafetensorsWeightStore, TensorSelection, WeightStore},
};

const EMBEDDING_UNIT: &str = "inkling.static.embedding";
const EMBED_NORM_UNIT: &str = "inkling.static.embed_norm";
const NORM_UNIT: &str = "inkling.static.norm";
const HEAD_UNIT: &str = "inkling.static.output";
const AUDIO_UNIT: &str = "inkling.static.audio";
const VISION_NORM_UNIT: &str = "inkling.static.vision_norm";

/// Inkling multimodal model using bounded host residency for hMLP and decoder blocks.
pub struct InklingLayerwiseModel {
    execution: GeneralLayerwiseModel<InklingLayerwiseAdapter>,
}

impl InklingLayerwiseModel {
    /// Returns the parsed Inkling configuration.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates global/sliding KV and short-convolution state for every layer.
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

    /// Runs the text decoder while preserving KV and convolution state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(InklingInput::Decode(inputs), cache, stream)
    }

    /// Clears temporary vision and decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_all_device_groups()
    }
}

impl CausalLm<Cache> for InklingLayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .forward(InklingInput::Prefill(input), cache, stream)
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

/// Loads Inkling's multimodal model through the generalized host-residency engine.
pub fn load_inkling_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: LayerwiseLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    let adapter = InklingLayerwiseAdapter::new(args, stream)?;
    Ok(InklingLayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Inkling with expert-granular sparse caching for routed text experts.
pub fn load_inkling_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    if args.text_config.n_routed_experts <= 0
        || !(0..args.text_config.num_hidden_layers).any(|layer| !args.text_config.is_dense(layer))
    {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires an Inkling checkpoint with routed MoE layers".into(),
        ));
    }
    let mut adapter = InklingLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution = load_general_layerwise_model(
        model_dir,
        adapter,
        options.non_expert,
        stream,
        weights_stream,
    )?;
    let store = execution.weight_store_arc();
    let entries = inkling_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(InklingLayerwiseModel { execution })
}

/// Adapter for Inkling local/global attention and dense/MoE text blocks.
struct InklingLayerwiseAdapter {
    args: ModelArgs,
    embedding: nn::Embedding,
    embed_norm: nn::RmsNorm,
    norm: nn::RmsNorm,
    lm_head: nn::Linear,
    audio: Option<AudioModel>,
    vision_norm: Option<nn::RmsNorm>,
    vision_depth: usize,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl InklingLayerwiseAdapter {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let text = &args.text_config;
        let audio = args
            .audio_config
            .as_ref()
            .map(|config| AudioModel::new(config, stream))
            .transpose()?;
        let vision = args
            .vision_config
            .as_ref()
            .map(|config| VisionModel::new(config, stream))
            .transpose()?;
        let (vision_norm, vision_depth) = match vision {
            Some(vision) => (Some(vision.final_norm), vision.layers.len()),
            None => (None, 0),
        };
        Ok(Self {
            embedding: nn::Embedding::unloaded(
                text.vocab_size,
                text.hidden_size,
                Dtype::Float32,
                stream,
            )?,
            embed_norm: nn::RmsNorm::unloaded(
                text.hidden_size,
                text.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            norm: nn::RmsNorm::unloaded(
                text.hidden_size,
                text.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            lm_head: nn::Linear::unloaded(
                text.hidden_size,
                text.vocab_size,
                false,
                Dtype::Float32,
                stream,
            )?,
            audio,
            vision_norm,
            vision_depth,
            sparse_expert_cache: false,
            expert_cache: None,
            args,
        })
    }

    /// Returns the parsed Inkling configuration.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args.text_config)
    }

    fn recipes_for_module(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let normalized = normalized_checkpoint_keys(store);
        let direct = store.keys();
        let mut recipes = BTreeMap::new();
        for local_name in module.parameters().flatten().keys() {
            let destination = format!("{prefix}.{local_name}");
            if direct.contains(&destination) {
                continue;
            }
            if let Some(recipe) = inkling_w13_recipe(&destination, &normalized, store)? {
                recipes.insert(local_name.to_string(), recipe);
                continue;
            }
            let raw = normalized.get(&destination).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Inkling checkpoint is missing runtime parameter {destination}"
                ))
            })?;
            let source = DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full);
            recipes.insert(
                local_name.to_string(),
                if raw.ends_with("_sconv.weight") {
                    DerivedWeightRecipe::Cast {
                        input: Box::new(source),
                        dtype: Dtype::Float32,
                    }
                } else {
                    source
                },
            );
        }
        Ok(recipes)
    }
}

fn normalized_checkpoint_keys(store: &dyn WeightStore) -> BTreeMap<String, String> {
    store
        .keys()
        .into_iter()
        .filter_map(|raw| normalize_checkpoint_key(&raw).map(|runtime| (runtime, raw)))
        .collect()
}

fn normalize_checkpoint_key(raw: &str) -> Option<String> {
    if raw.starts_with("model.mtp.") {
        return None;
    }
    if let Some(suffix) = raw.strip_prefix("model.audio.") {
        return Some(format!("audio.{suffix}"));
    }
    if let Some(suffix) = raw.strip_prefix("model.visual.") {
        let mut suffix = suffix.to_string();
        for layer in 0..4 {
            suffix = suffix
                .replace(
                    &format!("layers.linear_{layer}.weight"),
                    &format!("layers.{layer}.projection.weight"),
                )
                .replace(
                    &format!("layers.norm_{layer}.weight"),
                    &format!("layers.{layer}.layer_norm.weight"),
                );
        }
        return Some(format!("visual.{suffix}"));
    }
    if !raw.starts_with("model.llm.") {
        return Some(raw.to_string());
    }
    let mut key = raw.replacen("model.llm.", "model.", 1);
    key = key
        .replace("model.embed.weight", "model.embed_tokens.weight")
        .replace("model.unembed.weight", "lm_head.weight")
        .replace(".attn_norm.weight", ".input_layernorm.weight")
        .replace(".mlp_norm.weight", ".post_attention_layernorm.weight")
        .replace(".attn.wq_du.weight", ".self_attn.q_proj.weight")
        .replace(".attn.wk_dv.weight", ".self_attn.k_proj.weight")
        .replace(".attn.wv_dv.weight", ".self_attn.v_proj.weight")
        .replace(".attn.wr_du.weight", ".self_attn.r_proj.weight")
        .replace(".attn.wo_ud.weight", ".self_attn.o_proj.weight")
        .replace(".attn.q_norm.weight", ".self_attn.q_norm.weight")
        .replace(".attn.k_norm.weight", ".self_attn.k_norm.weight")
        .replace(".attn.rel_logits_proj.proj", ".self_attn.rel_proj")
        .replace(".attn.k_sconv.weight", ".self_attn.k_sconv.weight")
        .replace(".attn.v_sconv.weight", ".self_attn.v_sconv.weight")
        .replace(".mlp.w2_md.weight", ".dense.down_proj.weight")
        .replace(".mlp.global_scale", ".dense_global_scale")
        .replace(".mlp.gate.weight", ".moe.router.weight")
        .replace(".mlp.gate.bias", ".moe.router.bias")
        .replace(".mlp.gate.global_scale", ".moe.router.global_scale")
        .replace(".mlp.experts.w2_weight", ".moe.experts.down_proj")
        .replace(
            ".mlp.shared_experts.shared_w2_weight",
            ".moe.shared_experts.down_proj",
        );
    Some(key)
}

/// Input mode for typed prefill and cached text decode.
pub enum InklingInput<'a> {
    /// Ordered multimodal prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text tokens for a cached decode step.
    Decode(&'a Array),
}

enum PreparedPart {
    Ready { tokens: Array, embeddings: Array },
    Vision { tokens: Array, job: usize },
}

struct VisionJob {
    hidden: Array,
}

/// Transient media and ordered prompt assembly state.
struct InklingForwardContext {
    parts: Vec<PreparedPart>,
    vision_jobs: Vec<VisionJob>,
    needs_assembly: bool,
}

/// One leased Inkling hMLP or decoder unit.
enum InklingLayer {
    /// One hMLP projection/fold layer.
    Vision(VisionLayer),
    /// One text decoder block.
    Text(Box<DecoderLayer>),
}

impl ModuleParameters for InklingLayer {
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

fn inkling_w13_recipe(
    destination: &str,
    normalized: &BTreeMap<String, String>,
    store: &dyn WeightStore,
) -> Result<Option<DerivedWeightRecipe>, Error> {
    let (source_runtime, axis, parity, concatenate) =
        if let Some(prefix) = destination.strip_suffix(".dense.gate_proj.weight") {
            (format!("{prefix}.mlp.w13_dn.weight"), 0, 0, false)
        } else if let Some(prefix) = destination.strip_suffix(".dense.up_proj.weight") {
            (format!("{prefix}.mlp.w13_dn.weight"), 0, 1, false)
        } else if let Some(prefix) = destination.strip_suffix(".moe.experts.gate_up_proj") {
            (format!("{prefix}.mlp.experts.w13_weight"), 1, 0, true)
        } else if let Some(prefix) = destination.strip_suffix(".moe.shared_experts.gate_up_proj") {
            (
                format!("{prefix}.mlp.shared_experts.shared_w13_weight"),
                1,
                0,
                true,
            )
        } else {
            return Ok(None);
        };
    let Some(raw) = normalized.get(&source_runtime) else {
        return Ok(None);
    };
    let metadata = store.metadata(raw)?;
    let rows = metadata
        .shape
        .get(axis)
        .copied()
        .ok_or_else(|| Error::UnsupportedArchitecture("Inkling w13 rank is invalid".into()))?;
    if rows % 2 != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor {raw} has odd interleaved width {rows}"
        )));
    }
    let selected = |parity: usize| {
        DerivedWeightRecipe::source(
            raw.clone(),
            TensorSelection::Indices {
                axis,
                indices: (parity..rows).step_by(2).collect(),
            },
        )
    };
    Ok(Some(if concatenate {
        DerivedWeightRecipe::Concatenate {
            axis,
            inputs: vec![selected(0), selected(1)],
        }
    } else {
        selected(parity)
    }))
}

impl GeneralLayerwiseModelAdapter for InklingLayerwiseAdapter {
    type Input<'a> = InklingInput<'a>;
    type Cache = Cache;
    type Layer = InklingLayer;
    type ForwardContext = InklingForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    self.recipes_for_module(&self.embedding, "model.embed_tokens", store)?,
                )?,
            )?,
            StaticUnitBindings::new(
                EMBED_NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.embed_norm,
                    "model.embed_norm",
                    store,
                    self.recipes_for_module(&self.embed_norm, "model.embed_norm", store)?,
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    self.recipes_for_module(&self.norm, "model.norm", store)?,
                )?,
            )?,
            StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(
                    &self.lm_head,
                    "lm_head",
                    store,
                    self.recipes_for_module(&self.lm_head, "lm_head", store)?,
                )?,
            )?,
        ];
        if let Some(audio) = &self.audio {
            units.push(StaticUnitBindings::new(
                AUDIO_UNIT,
                build_module_bindings_with_recipes(
                    audio,
                    "audio",
                    store,
                    self.recipes_for_module(audio, "audio", store)?,
                )?,
            )?);
        }
        if let Some(norm) = &self.vision_norm {
            units.push(StaticUnitBindings::new(
                VISION_NORM_UNIT,
                build_module_bindings_with_recipes(
                    norm,
                    "visual.final_norm",
                    store,
                    self.recipes_for_module(norm, "visual.final_norm", store)?,
                )?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected =
            4 + usize::from(self.audio.is_some()) + usize::from(self.vision_norm.is_some());
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.embedding, &leases[0])?;
        populate_module_from_lease(&mut self.embed_norm, &leases[1])?;
        populate_module_from_lease(&mut self.norm, &leases[2])?;
        populate_module_from_lease(&mut self.lm_head, &leases[3])?;
        let mut index = 4;
        if let Some(audio) = &mut self.audio {
            populate_module_from_lease(audio, &leases[index])?;
            index += 1;
        }
        if let Some(norm) = &mut self.vision_norm {
            populate_module_from_lease(norm, &leases[index])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
        }
        if cache.layers.len() != self.args.text_config.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.text_config.num_hidden_layers
            )));
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        if let InklingInput::Decode(tokens) = input {
            let hidden = self
                .embed_norm
                .forward(&self.embedding.forward(tokens, stream)?, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts: Vec::new(),
                    vision_jobs: Vec::new(),
                    needs_assembly: false,
                },
            });
        }
        let InklingInput::Prefill(typed) = input else {
            unreachable!()
        };
        input::validate(typed)?;
        let mut parts = Vec::with_capacity(typed.parts.len());
        let mut vision_jobs = Vec::new();
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    let embeddings = self
                        .embed_norm
                        .forward(&self.embedding.forward(tokens, stream)?, stream)?;
                    parts.push(PreparedPart::Ready {
                        tokens: tokens.clone(),
                        embeddings,
                    });
                }
                (input::Modality::Image, input::InputPayload::Tensor(pixels)) => {
                    if self.vision_norm.is_none() {
                        return Err(Error::UnsupportedArchitecture(
                            "Inkling image input requires vision_config and vision weights".into(),
                        ));
                    }
                    let job = vision_jobs.len();
                    vision_jobs.push(VisionJob {
                        hidden: pixels.clone(),
                    });
                    let count = pixels.dim(0) as usize;
                    parts.push(PreparedPart::Vision {
                        tokens: input::token_ids_array(
                            &vec![self.args.image_token_id; count],
                            stream,
                        )?,
                        job,
                    });
                }
                (input::Modality::Audio, input::InputPayload::Tensor(ids)) => {
                    let embeddings = self
                        .audio
                        .as_mut()
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "Inkling audio input requires audio_config and audio weights"
                                    .into(),
                            )
                        })?
                        .forward(ids, part.metadata.audio_mask, stream)?;
                    parts.push(PreparedPart::Ready {
                        tokens: input::token_ids_array(
                            &vec![self.args.audio_token_id; embeddings.dim(1) as usize],
                            stream,
                        )?,
                        embeddings,
                    });
                }
                (
                    input::Modality::Image | input::Modality::Audio,
                    input::InputPayload::Embeddings(embeddings),
                ) => {
                    input::ensure_hidden_size(
                        embeddings,
                        self.args.text_config.hidden_size,
                        "Inkling media embeddings",
                    )?;
                    let token = if part.modality == input::Modality::Image {
                        self.args.image_token_id
                    } else {
                        self.args.audio_token_id
                    };
                    parts.push(PreparedPart::Ready {
                        tokens: input::token_ids_array(
                            &vec![token; embeddings.dim(1) as usize],
                            stream,
                        )?,
                        embeddings: embeddings.clone(),
                    });
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling layerwise input does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        if vision_jobs.is_empty() {
            let token_parts = parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Ready { tokens, .. } => tokens,
                    PreparedPart::Vision { .. } => unreachable!(),
                })
                .collect::<Vec<_>>();
            let embedding_parts = parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Ready { embeddings, .. } => embeddings,
                    PreparedPart::Vision { .. } => unreachable!(),
                })
                .collect::<Vec<_>>();
            let _tokens = concatenate_axis(&token_parts, 1, stream)?;
            let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts,
                    vision_jobs,
                    needs_assembly: false,
                },
            });
        }
        let hidden = vision_jobs
            .first()
            .map(|job| job.hidden.clone())
            .unwrap_or_else(|| {
                parts
                    .first()
                    .map(|part| match part {
                        PreparedPart::Ready { embeddings, .. } => embeddings.clone(),
                        PreparedPart::Vision { .. } => unreachable!(),
                    })
                    .expect("validated non-empty Inkling input")
            });
        Ok(LayerwiseForwardState {
            hidden,
            context: InklingForwardContext {
                parts,
                vision_jobs,
                needs_assembly: true,
            },
        })
    }

    fn execution_group_count(&self) -> usize {
        1 + usize::from(self.vision_depth > 0)
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        match (self.vision_depth > 0, group) {
            (true, 0) => Ok("vision_encoder".into()),
            (true, 1) | (false, 0) => Ok("text_decoder".into()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Inkling has no execution group {group}"
            ))),
        }
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        self.execution_group_id(group)
            .is_ok_and(|id| id != "vision_encoder" || !context.vision_jobs.is_empty())
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match self.execution_group_id(group)?.as_str() {
            "vision_encoder" => Ok(self.vision_depth),
            "text_decoder" => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => unreachable!(),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        if self.execution_group_id(group)? == "vision_encoder" {
            let args = self
                .args
                .vision_config
                .as_ref()
                .expect("vision group config");
            let specs = [
                (75, 128, 1, 5),
                (512, 512, 1, 2),
                (8192, 4800, 1, 4),
                (9600, args.text_hidden_size, 2, 1),
            ];
            let (input_dim, output_dim, t_fold, hw_fold) = specs[index];
            Ok(InklingLayer::Vision(VisionLayer::new(
                input_dim,
                output_dim,
                t_fold,
                hw_fold,
                index + 1 != specs.len(),
                args.rms_norm_eps,
                stream,
            )?))
        } else {
            Ok(InklingLayer::Text(Box::new(DecoderLayer::new(
                &self.args.text_config,
                index as i32,
                stream,
            )?)))
        }
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if self.execution_group_id(group).ok().as_deref() == Some("vision_encoder") {
            format!("visual.layers.{index}")
        } else {
            format!("model.layers.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if self.execution_group_id(group).ok().as_deref() == Some("vision_encoder") {
            format!("inkling.vision.{index:05}")
        } else {
            format!("inkling.layer.{index:05}")
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
            self.recipes_for_module(layer, &prefix, store)?,
        )?;
        Ok(
            if self.sparse_expert_cache && self.execution_group_id(group)? == "text_decoder" {
                bindings
                    .into_iter()
                    .filter(|binding| !binding.name().starts_with("moe.experts."))
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
                |name| name.starts_with("moe.experts."),
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
                .filter(|key| key.contains(".mlp.experts.") || key.contains(".moe.experts."))
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
            ("vision_encoder", InklingLayer::Vision(layer)) => {
                for job in &mut context.vision_jobs {
                    job.hidden = layer.forward(&job.hidden, stream)?;
                }
                Ok(context.vision_jobs[0].hidden.clone())
            }
            ("text_decoder", InklingLayer::Text(layer)) => {
                if self.sparse_expert_cache && !self.args.text_config.is_dense(index as i32) {
                    let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Inkling sparse expert cache was not initialized".into(),
                        )
                    })?;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    return Ok(layer.forward_with_expert_executor(
                        hidden,
                        Some(&mut cache.layers[index]),
                        stream,
                        |flat, indices, weights, stream| {
                            let acquired = expert_cache
                                .acquire_routes(index, indices, pass, stream)
                                .map_err(|error| Exception::custom(error.to_string()))?;
                            let started = Instant::now();
                            let text = &self.args.text_config;
                            let mut bank = PackedSwiGluExperts::new(
                                acquired.identities().len() as i32,
                                text.hidden_size,
                                text.moe_intermediate_size(),
                                None,
                                None,
                                stream,
                            )?;
                            bank.gate_up_proj = Param::new(
                                acquired
                                    .compact_binding("gate_up_proj", stream)
                                    .map_err(|error| Exception::custom(error.to_string()))?,
                            );
                            bank.down_proj = Param::new(
                                acquired
                                    .compact_binding("down_proj", stream)
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
                    )?);
                }
                Ok(layer.forward(hidden, Some(&mut cache.layers[index]), stream)?)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Inkling execution unit does not match group {group}"
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
            let layer = &cache.layers[index];
            let mut arrays = layer.kv.retained_arrays();
            arrays.extend(
                layer
                    .convolutions
                    .iter()
                    .filter_map(|cache| cache.state.as_ref()),
            );
            arrays
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
        context.vision_jobs.iter().map(|job| &job.hidden).collect()
    }

    fn finish_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let group_id = self.execution_group_id(group)?;
        let should_assemble = context.needs_assembly && group_id == "vision_encoder";
        if !should_assemble {
            return Ok(hidden.clone());
        }
        if let Some(norm) = &mut self.vision_norm {
            for job in &mut context.vision_jobs {
                job.hidden = norm
                    .forward(&job.hidden, stream)?
                    .reshape(&[-1, self.args.text_config.hidden_size], stream)?
                    .try_index_device(NewAxis, stream)?;
            }
        }
        let mut tokens = Vec::with_capacity(context.parts.len());
        let mut embeddings = Vec::with_capacity(context.parts.len());
        for part in &context.parts {
            match part {
                PreparedPart::Ready {
                    tokens: ids,
                    embeddings: value,
                } => {
                    tokens.push(ids);
                    embeddings.push(value);
                }
                PreparedPart::Vision { tokens: ids, job } => {
                    tokens.push(ids);
                    embeddings.push(&context.vision_jobs[*job].hidden);
                }
            }
        }
        let _tokens = concatenate_axis(&tokens, 1, stream)?;
        context.needs_assembly = false;
        Ok(concatenate_axis(&embeddings, 1, stream)?)
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?.divide(
            Array::from_f32(self.args.text_config.logits_mup_width_multiplier),
            stream,
        )?;
        let logits = self.lm_head.forward(&hidden, stream)?;
        if let Some(size) = self.args.text_config.unpadded_vocab_size {
            if size < logits.dim(-1) {
                return Ok(logits.try_index_device((.., .., ..size), stream)?);
            }
        }
        Ok(logits)
    }

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        key.starts_with("model.mtp.")
    }
}

pub(crate) fn inkling_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let text = &args.text_config;
    let mut entries = Vec::new();
    for layer in 0..text.num_hidden_layers as usize {
        if text.is_dense(layer as i32) {
            continue;
        }
        let runtime_prefix = format!("model.layers.{layer}");
        let gate_up_runtime = format!("{runtime_prefix}.moe.experts.gate_up_proj");
        let down_runtime = format!("{runtime_prefix}.moe.experts.down_proj");
        let gate_up_raw = normalized
            .get(&gate_up_runtime)
            .cloned()
            .or_else(|| {
                normalized
                    .get(&format!("{runtime_prefix}.mlp.experts.w13_weight"))
                    .cloned()
            })
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Inkling checkpoint is missing routed gate/up bank for layer {layer}"
                ))
            })?;
        let down_raw = normalized.get(&down_runtime).cloned().ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Inkling checkpoint is missing routed down bank for layer {layer}"
            ))
        })?;
        let gate_metadata = store.metadata(&gate_up_raw)?;
        let interleaved = gate_metadata.shape.get(1).copied().ok_or_else(|| {
            Error::UnsupportedArchitecture("Inkling routed gate/up rank is invalid".into())
        })?;
        if !normalized.contains_key(&gate_up_runtime) && interleaved % 2 != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling routed w13 bank for layer {layer} has odd interleaved width {interleaved}"
            )));
        }
        for expert in 0..text.n_routed_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let selected_expert = DerivedWeightRecipe::source(
                gate_up_raw.clone(),
                TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                },
            );
            let gate_up = if normalized.contains_key(&gate_up_runtime) {
                selected_expert
            } else {
                let select = |parity| DerivedWeightRecipe::Select {
                    input: Box::new(selected_expert.clone()),
                    selection: TensorSelection::Indices {
                        axis: 1,
                        indices: (parity..interleaved).step_by(2).collect(),
                    },
                };
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![select(0), select(1)],
                }
            };
            let down = DerivedWeightRecipe::source(
                down_raw.clone(),
                TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                },
            );
            let mut bindings = Vec::new();
            for (name, recipe) in [("gate_up_proj", gate_up), ("down_proj", down)] {
                let bytes = recipe.infer(store)?.byte_len();
                bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Inkling expert byte total overflowed".into())
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

/// Inkling text token generation using layerwise-host execution.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    common::generation::Generate<'a, InklingLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, ones_dtype, stack_axis},
        Array, Device, DeviceType, Dtype, ExecutionContext, Stream,
    };

    use super::{load_inkling_layerwise_model, load_inkling_sparse_expert_cache_model};
    use crate::{
        cache::KeyValueCache,
        expert_cache::ExpertCacheLoadOptions,
        layerwise::LayerwiseLoadOptions,
        models::{
            common::generation::CausalLm,
            inkling::{self as resident, Model, ModelArgs},
            input as runtime_input,
        },
        offload::{OffloadConfig, ResidencyPolicy},
    };

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "inkling_mm_model",
            "eos_token_id": 1,
            "text_config": {
                "hidden_size": 16,
                "num_hidden_layers": 3,
                "vocab_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "swa_num_attention_heads": 2,
                "swa_num_key_value_heads": 1,
                "swa_head_dim": 8,
                "sliding_window_size": 4,
                "local_layer_ids": [0, 1],
                "dense_mlp_idx": 1,
                "sconv_kernel_size": 3,
                "d_rel": 4,
                "rel_extent": 8,
                "intermediate_size": 8,
                "dense_intermediate_size": 16,
                "moe_intermediate_size": 8,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "n_shared_experts": 1,
                "route_scale": 1.0,
                "use_sconv": true,
                "use_embed_norm": true,
                "shared_expert_sink": true,
                "use_gate_bias": true,
                "norm_after_topk": true,
                "use_global_scale": true,
                "gate_activation": "sigmoid",
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "q_bias": false,
                "o_bias": false,
                "logits_mup_width_multiplier": 2.0,
                "unpadded_vocab_size": 30
            }
        })
    }

    fn args() -> ModelArgs {
        serde_json::from_value(config()).unwrap()
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            *parameter = if name.ends_with("norm.weight")
                || name.ends_with("layernorm.weight")
                || name.ends_with("global_scale")
            {
                ones_dtype(&shape, parameter.dtype(), stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
        }
    }

    fn released_name(runtime: &str) -> String {
        if runtime == "lm_head.weight" {
            return "model.llm.unembed.weight".into();
        }
        if let Some(rest) = runtime.strip_prefix("audio.") {
            return format!("model.audio.{rest}");
        }
        if let Some(rest) = runtime.strip_prefix("visual.") {
            return format!("model.visual.{rest}");
        }
        let rest = runtime.strip_prefix("model.").unwrap();
        let mut raw = format!("model.llm.{rest}");
        raw = raw
            .replace("model.llm.embed_tokens.weight", "model.llm.embed.weight")
            .replace(".input_layernorm.weight", ".attn_norm.weight")
            .replace(".post_attention_layernorm.weight", ".mlp_norm.weight")
            .replace(".self_attn.q_proj.weight", ".attn.wq_du.weight")
            .replace(".self_attn.k_proj.weight", ".attn.wk_dv.weight")
            .replace(".self_attn.v_proj.weight", ".attn.wv_dv.weight")
            .replace(".self_attn.r_proj.weight", ".attn.wr_du.weight")
            .replace(".self_attn.o_proj.weight", ".attn.wo_ud.weight")
            .replace(".self_attn.q_norm.weight", ".attn.q_norm.weight")
            .replace(".self_attn.k_norm.weight", ".attn.k_norm.weight")
            .replace(".self_attn.rel_proj", ".attn.rel_logits_proj.proj")
            .replace(".self_attn.k_sconv.weight", ".attn.k_sconv.weight")
            .replace(".self_attn.v_sconv.weight", ".attn.v_sconv.weight")
            .replace(".dense.down_proj.weight", ".mlp.w2_md.weight")
            .replace(".dense_global_scale", ".mlp.global_scale")
            .replace(".moe.router.weight", ".mlp.gate.weight")
            .replace(".moe.router.bias", ".mlp.gate.bias")
            .replace(".moe.router.global_scale", ".mlp.gate.global_scale")
            .replace(".moe.experts.down_proj", ".mlp.experts.w2_weight")
            .replace(
                ".moe.shared_experts.down_proj",
                ".mlp.shared_experts.shared_w2_weight",
            );
        raw
    }

    fn interleave(gate: &Array, up: &Array, axis: i32, stream: &Stream) -> Array {
        let stacked = stack_axis(&[gate.clone(), up.clone()], axis, stream).unwrap();
        let mut shape = gate.shape().to_vec();
        let row_axis = shape.len() - 2;
        shape[row_axis] *= 2;
        stacked.reshape(&shape, stream).unwrap()
    }

    fn write_fixture(dir: &Path, model: &Model, stream: &Stream) {
        let parameters = model.parameters().flatten();
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in &parameters {
            let name = name.as_ref();
            if name.ends_with(".dense.up_proj.weight") {
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".dense.gate_proj.weight") {
                let up_name = format!("{prefix}.dense.up_proj.weight");
                let up = parameters.get(up_name.as_str()).unwrap();
                arrays.push((
                    format!("model.llm.{}.mlp.w13_dn.weight", &prefix["model.".len()..]),
                    interleave(value, up, 1, stream),
                ));
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".moe.experts.gate_up_proj") {
                let intermediate = model.args.text_config.moe_intermediate_size.unwrap();
                let gate = value
                    .try_index_device((.., ..intermediate, ..), stream)
                    .unwrap();
                let up = value
                    .try_index_device((.., intermediate.., ..), stream)
                    .unwrap();
                arrays.push((
                    format!(
                        "model.llm.{}.mlp.experts.w13_weight",
                        &prefix["model.".len()..]
                    ),
                    interleave(&gate, &up, 2, stream),
                ));
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".moe.shared_experts.gate_up_proj") {
                let intermediate = model.args.text_config.moe_intermediate_size.unwrap();
                let gate = value
                    .try_index_device((.., ..intermediate, ..), stream)
                    .unwrap();
                let up = value
                    .try_index_device((.., intermediate.., ..), stream)
                    .unwrap();
                arrays.push((
                    format!(
                        "model.llm.{}.mlp.shared_experts.shared_w13_weight",
                        &prefix["model.".len()..]
                    ),
                    interleave(&gate, &up, 2, stream),
                ));
                continue;
            }
            let raw = released_name(name);
            let value = if raw.ends_with("_sconv.weight") {
                value.as_dtype(Dtype::Bfloat16, stream).unwrap()
            } else {
                (*value).clone()
            };
            arrays.push((raw, value));
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
            assert!((left - right).abs() <= 5e-5, "{left} != {right}");
        }
    }

    fn parity(depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap());
        let mut layerwise =
            load_inkling_layerwise_model(dir.path(), options, gpu.stream(), cpu.stream()).unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = resident::Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
            Array::from_slice(&[6u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    &tokens,
                    None,
                    Some(&mut resident_cache),
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            assert_eq!(resident_cache.offset(), layerwise_cache.offset());
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                assert_eq!(expected.kv.offset(), actual.kv.offset());
                for (expected, actual) in expected.convolutions.iter().zip(&actual.convolutions) {
                    assert_eq!(expected.offset, actual.offset);
                    assert_eq!(
                        expected.state.as_ref().map(Array::shape),
                        actual.state.as_ref().map(Array::shape)
                    );
                }
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("inkling.layer."))
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
    fn inkling_released_layout_layerwise_parity() {
        parity(1);
        parity(2);
    }

    #[test]
    fn inkling_sparse_expert_cache_prefill_and_decode_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());
        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
        )
        .unwrap();
        let mut cached =
            load_inkling_sparse_expert_cache_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = resident::Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    &tokens,
                    None,
                    Some(&mut resident_cache),
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
            crate::models::ModelKind::Inkling,
            report.owned_experts / 2,
            gpu.stream(),
            cpu.stream(),
        );
    }

    #[test]
    fn inkling_audio_and_text_layerwise_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["audio_config"] = serde_json::json!({
            "text_hidden_size": 16,
            "num_codebooks": 2,
            "codebook_size": 8,
            "bias": false,
            "use_audio_norm": true,
            "audio_mode": "dmel",
            "rms_norm_eps": 1e-6,
        });
        value["audio_token_id"] = serde_json::json!(20);
        let mut fixture =
            Model::new(serde_json::from_value(value.clone()).unwrap(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_inkling_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let text = runtime_input::token_ids_array(&[1, 2], gpu.stream()).unwrap();
        let audio_ids = Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[3, 2]);
        let mask = Array::from_slice(&[true, true, false], &[1, 3]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::audio_tensor(
                &audio_ids,
                runtime_input::InputMetadata::audio_mask(&mask),
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
    }
}
