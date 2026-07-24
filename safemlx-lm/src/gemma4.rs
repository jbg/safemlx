//! Text-decoder bounded layer execution for Gemma 4 checkpoints.

use std::{collections::BTreeMap, collections::HashMap, path::Path, sync::Arc};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::{
        concatenate_axis, indexing::TryIndexOp, r#where, tanh, GgufCheckpoint, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Stream,
};

use crate::{
    cache::KeyValueCache,
    error::Error,
    layerwise::{
        load_general_layerwise_model, load_general_layerwise_model_with_store,
        GeneralLayerwiseModel, GeneralLayerwiseModelAdapter, LayerExecutionLoadOptions,
        LayerwiseForwardState, StaticUnitBindings, WeightResidency,
    },
    models::{
        common::generation::CausalLm,
        gemma4::{
            self as resident, AttentionInput, Cache, Gemma4Embedding, Gemma4TextModel, LayerType,
            ModelArgs, TransformerBlock,
        },
        gemma4_audio::{
            AudioLayer, Gemma4AudioConfig, Gemma4AudioLayerwiseStatic, Gemma4AudioTower,
        },
        gemma4_multimodal::Gemma4ModalityEmbedder,
        gemma4_vision::{
            Gemma4VisionConfig, Gemma4VisionLayerwiseState, Gemma4VisionLayerwiseStatic,
            Gemma4VisionTower, VisionLayer,
        },
        input,
    },
    module_binding::{
        build_module_bindings_with_recipes, canonical_checkpoint_name, populate_module_from_lease,
    },
    residency::{ResidencyReport, ResidentUnitLease, WeightBinding},
    utils::create_causal_mask,
    weight_recipe::DerivedWeightRecipe,
    weight_store::{GgufWeightStore, TensorSelection, WeightStore},
};

const EMBEDDING_UNIT: &str = "gemma4.static.embedding";
const PER_LAYER_EMBEDDING_UNIT: &str = "gemma4.static.per_layer_embedding";
const PER_LAYER_PROJECTION_UNIT: &str = "gemma4.static.per_layer_projection";
const PER_LAYER_NORM_UNIT: &str = "gemma4.static.per_layer_norm";
const NORM_UNIT: &str = "gemma4.static.norm";
const HEAD_UNIT: &str = "gemma4.static.output";
const VISION_STATIC_UNIT: &str = "gemma4.static.vision";
const VISION_EMBED_UNIT: &str = "gemma4.static.vision_embed";
const AUDIO_STATIC_UNIT: &str = "gemma4.static.audio";
const AUDIO_EMBED_UNIT: &str = "gemma4.static.audio_embed";

/// Gemma 4 multimodal model using bounded residency for media and text blocks.
pub struct Gemma4LayerwiseModel {
    execution: GeneralLayerwiseModel<Gemma4LayerwiseAdapter>,
}

impl Gemma4LayerwiseModel {
    /// Returns normalized Gemma 4 text arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates an empty Gemma 4 generation cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(self.args())
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
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
    }

    /// Backward-compatible alias for [`Self::checkpoint_store`].
    pub fn weight_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.checkpoint_store()
    }

    /// Runs the text decoder while preserving alternating and shared KV state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Gemma4Input::Decode(inputs), cache, stream)
    }

    pub(crate) fn prefill_mtp(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<resident::Gemma4StepOutput, Exception> {
        self.forward_mtp(Gemma4Input::Prefill(input), cache, stream)
    }

    pub(crate) fn verify_mtp(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<resident::Gemma4StepOutput, Exception> {
        self.forward_mtp(Gemma4Input::Decode(tokens), cache, stream)
    }

    fn forward_mtp(
        &mut self,
        input: Gemma4Input<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<resident::Gemma4StepOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("Gemma 4 layerwise pass did not retain target draft state")
        })?;
        Ok(resident::Gemma4StepOutput {
            logits,
            hidden,
            shared_kv_states: context.shared_kv,
        })
    }

    pub(crate) fn mtp_token_embedding(
        &mut self,
        token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .adapter_mut()
            .mtp_token_embedding(token, stream)
    }

    /// Clears temporary media and decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_all_device_groups()
    }
}

impl CausalLm<Cache> for Gemma4LayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .forward(Gemma4Input::Prefill(input), cache, stream)
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

/// Loads Gemma 4 text and configured media towers through bounded residency.
pub fn load_gemma4_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let (args, vision, image_token_id, video_token_id, audio, audio_token_id) =
        resident::get_gemma4_model_config(model_dir)?;
    let adapter = Gemma4LayerwiseAdapter::new(
        args,
        vision,
        image_token_id,
        video_token_id,
        audio,
        audio_token_id,
        stream,
    )?;
    Ok(Gemma4LayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_gemma4_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Gemma4LayerwiseModel, Vec<u32>), Error> {
    let prepared =
        resident::prepare_gemma4_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let adapter = Gemma4LayerwiseAdapter::new(prepared.args, None, None, None, None, None, stream)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => load_general_layerwise_model_with_store(
            store,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::DenseDiskStream(options) => load_general_layerwise_model_with_store(
            store,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::SparseExpertCache(_)
        | WeightResidency::SparseExpertCacheWithDenseLayers(_) => {
            return Err(Error::UnsupportedArchitecture(
                "sparse expert caching is not supported for Gemma 4 GGUF checkpoints".into(),
            ));
        }
        WeightResidency::FullyResident => {
            return Err(Error::UnsupportedArchitecture(
                "the bounded GGUF Gemma 4 loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((Gemma4LayerwiseModel { execution }, prepared.eos_token_ids))
}

/// Adapter for Gemma 4 per-layer inputs and shared-KV attention blocks.
pub struct Gemma4LayerwiseAdapter {
    args: ModelArgs,
    embedding: Gemma4Embedding,
    per_layer_embedding: Option<Gemma4Embedding>,
    per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    per_layer_norm: Option<nn::RmsNorm>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    vision: Option<Gemma4VisionLayerwiseStatic>,
    embed_vision: Option<Gemma4ModalityEmbedder>,
    audio: Option<Gemma4AudioLayerwiseStatic>,
    embed_audio: Option<Gemma4ModalityEmbedder>,
    vision_depth: usize,
    audio_depth: usize,
    audio_config: Option<Gemma4AudioConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    audio_token_id: Option<i32>,
}

impl Gemma4LayerwiseAdapter {
    fn new(
        args: ModelArgs,
        vision_config: Option<Gemma4VisionConfig>,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        audio_config: Option<Gemma4AudioConfig>,
        audio_token_id: Option<i32>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let text = Gemma4TextModel::new(&args, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(crate::models::common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                args.hidden_size,
                args.vocab_size,
                args.quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        let vision_tower = vision_config
            .clone()
            .map(|config| Gemma4VisionTower::new(config, stream))
            .transpose()?;
        let vision_depth = vision_config
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        let vision = vision_tower.map(Gemma4VisionLayerwiseStatic::from_tower);
        let embed_vision = vision_config
            .as_ref()
            .map(|config| {
                Gemma4ModalityEmbedder::new(
                    config.hidden_size,
                    args.hidden_size,
                    config.rms_norm_eps,
                    false,
                    args.weight_quantization(),
                    stream,
                )
            })
            .transpose()?;
        let audio_tower = audio_config
            .as_ref()
            .map(|config| Gemma4AudioTower::new(config, stream))
            .transpose()?;
        let audio_depth = audio_config
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        let audio = audio_tower.map(Gemma4AudioLayerwiseStatic::from_tower);
        let embed_audio = audio_config
            .as_ref()
            .map(|config| {
                Gemma4ModalityEmbedder::new(
                    config.output_proj_dims,
                    args.hidden_size,
                    config.rms_norm_eps,
                    false,
                    args.weight_quantization(),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            embedding: text.embed_tokens,
            per_layer_embedding: text.embed_tokens_per_layer,
            per_layer_projection: text.per_layer_model_projection,
            per_layer_norm: text.per_layer_projection_norm,
            norm: text.norm,
            lm_head,
            vision,
            embed_vision,
            audio,
            embed_audio,
            vision_depth,
            audio_depth,
            audio_config,
            image_token_id,
            video_token_id,
            audio_token_id,
        })
    }

    /// Returns normalized Gemma 4 text arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn mtp_token_embedding(&mut self, token: u32, stream: &Stream) -> Result<Array, Exception> {
        self.embedding
            .forward(&Array::from_slice(&[token], &[1, 1]), stream)?
            .multiply(
                Array::from_f32((self.args.hidden_size as f32).sqrt()),
                stream,
            )
    }

    fn recipes_for(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
    ) -> BTreeMap<String, DerivedWeightRecipe> {
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.keys();
        let parameters = module.parameters().flatten();
        let mut recipes = BTreeMap::new();
        if let Some(intermediate) = self.args.moe_intermediate_size {
            let fused = format!("{prefix}.experts.switch_glu.gate_up_proj");
            for suffix in ["weight", "scales", "biases"] {
                let source = format!("{fused}.{suffix}");
                if !keys.contains(&source) {
                    continue;
                }
                for (projection, start, end) in [
                    ("gate_proj", 0usize, intermediate as usize),
                    (
                        "up_proj",
                        intermediate as usize,
                        (2 * intermediate) as usize,
                    ),
                ] {
                    recipes.insert(
                        format!("experts.switch_glu.{projection}.{suffix}"),
                        DerivedWeightRecipe::Select {
                            input: Box::new(DerivedWeightRecipe::source(
                                source.clone(),
                                TensorSelection::Full,
                            )),
                            selection: TensorSelection::Range {
                                axis: 1,
                                start,
                                end,
                            },
                        },
                    );
                }
            }
        }
        for local_name in parameters.keys() {
            if recipes.contains_key(local_name.as_ref()) {
                continue;
            }
            let destination = format!("{prefix}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            if let Some(raw) = normalized.get(&canonical) {
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::Cast {
                        input: Box::new(DerivedWeightRecipe::source(
                            raw.clone(),
                            TensorSelection::Full,
                        )),
                        dtype: parameters
                            .get(local_name)
                            .expect("parameter came from the same flattened tree")
                            .dtype(),
                    },
                );
            }
        }
        recipes
    }

    fn bindings(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        Ok(build_module_bindings_with_recipes(
            module,
            prefix,
            store,
            self.recipes_for(module, prefix, store),
        )?)
    }

    fn prepare_per_layer_inputs(
        &mut self,
        input_ids: &Array,
        hidden: &Array,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        match (
            self.per_layer_embedding.as_mut(),
            self.per_layer_projection.as_mut(),
            self.per_layer_norm.as_mut(),
        ) {
            (Some(token_embedding), Some(projection), Some(norm)) => {
                let ple = self.args.hidden_size_per_layer_input;
                let token_identity = token_embedding
                    .forward(input_ids, stream)?
                    .multiply(Array::from_f32((ple as f32).sqrt()), stream)?
                    .reshape(
                        &[
                            input_ids.dim(0),
                            input_ids.dim(1),
                            self.args.num_hidden_layers,
                            ple,
                        ],
                        stream,
                    )?;
                let projected = projection
                    .forward(hidden, stream)?
                    .multiply(
                        Array::from_f32((self.args.hidden_size as f32).sqrt().recip()),
                        stream,
                    )?
                    .reshape(
                        &[
                            hidden.dim(0),
                            hidden.dim(1),
                            self.args.num_hidden_layers,
                            ple,
                        ],
                        stream,
                    )?;
                Ok(Some(
                    norm.forward(&projected, stream)?
                        .add(token_identity, stream)?
                        .multiply(Array::from_f32(2.0_f32.powf(-0.5)), stream)?,
                ))
            }
            (None, None, None) => Ok(None),
            _ => Err(Error::UnsupportedArchitecture(
                "Gemma 4 per-layer input modules are incomplete".into(),
            )),
        }
    }

    fn media_safe_per_layer_ids(&self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut output = tokens.clone();
        for token_id in [
            self.image_token_id,
            self.video_token_id,
            self.audio_token_id,
        ]
        .into_iter()
        .flatten()
        {
            let mask = output.eq(Array::from_int(token_id), stream)?;
            output = r#where(
                &mask,
                Array::from_int(self.args.pad_token_id),
                &output,
                stream,
            )?;
        }
        Ok(output)
    }
}

fn normalized_checkpoint_keys(store: &dyn WeightStore) -> BTreeMap<String, String> {
    store
        .keys()
        .into_iter()
        .map(|raw| {
            let canonical = canonical_checkpoint_name(&raw);
            let runtime = canonical
                .strip_prefix("language_model.model.")
                .map(|rest| format!("model.language_model.{rest}"))
                .or_else(|| {
                    [
                        ("vision_tower.", "model.vision_tower."),
                        ("embed_vision.", "model.embed_vision."),
                        ("audio_tower.", "model.audio_tower."),
                        ("embed_audio.", "model.embed_audio."),
                    ]
                    .into_iter()
                    .find_map(|(source, destination)| {
                        canonical
                            .strip_prefix(source)
                            .map(|rest| format!("{destination}{rest}"))
                    })
                })
                .unwrap_or(canonical);
            (runtime, raw)
        })
        .collect()
}

/// Input mode for typed prefill and cached text decode.
pub enum Gemma4Input<'a> {
    /// Ordered multimodal prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text tokens for cached decode.
    Decode(&'a Array),
}

enum Gemma4PreparedPart {
    Ready { tokens: Array, embeddings: Array },
    Vision { token_id: u32, job: usize },
    Audio { token_id: u32, job: usize },
}

struct Gemma4VisionJob {
    hidden: Array,
    state: Gemma4VisionLayerwiseState,
}

struct Gemma4AudioJob {
    hidden: Array,
    valid: i32,
}

/// One leased Gemma 4 media or text unit.
pub enum Gemma4Layer {
    /// Vision transformer block.
    Vision(Box<VisionLayer>),
    /// Audio conformer-style block.
    Audio(Box<AudioLayer>),
    /// Text transformer block.
    Text(Box<TransformerBlock>),
}

impl ModuleParameters for Gemma4Layer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Vision(x) => x.num_parameters(),
            Self::Audio(x) => x.num_parameters(),
            Self::Text(x) => x.num_parameters(),
        }
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(x) => x.parameters(),
            Self::Audio(x) => x.parameters(),
            Self::Text(x) => x.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Vision(x) => x.parameters_mut(),
            Self::Audio(x) => x.parameters_mut(),
            Self::Text(x) => x.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(x) => x.trainable_parameters(),
            Self::Audio(x) => x.trainable_parameters(),
            Self::Text(x) => x.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(x) => x.freeze_parameters(recursive),
            Self::Audio(x) => x.freeze_parameters(recursive),
            Self::Text(x) => x.freeze_parameters(recursive),
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(x) => x.unfreeze_parameters(recursive),
            Self::Audio(x) => x.unfreeze_parameters(recursive),
            Self::Text(x) => x.unfreeze_parameters(recursive),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(x) => x.all_frozen(),
            Self::Audio(x) => x.all_frozen(),
            Self::Text(x) => x.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(x) => x.any_frozen(),
            Self::Audio(x) => x.any_frozen(),
            Self::Text(x) => x.any_frozen(),
        }
    }
}

/// Transient Gemma 4 values shared across one multimodal decoder pass.
pub struct Gemma4ForwardContext {
    per_layer_inputs: Option<Array>,
    mask: Option<Array>,
    sliding_mask: Option<Array>,
    position_offset: i32,
    shared_kv: HashMap<LayerType, (Array, Array)>,
    parts: Vec<Gemma4PreparedPart>,
    vision_jobs: Vec<Gemma4VisionJob>,
    audio_jobs: Vec<Gemma4AudioJob>,
    tokens: Option<Array>,
    needs_assembly: bool,
    draft_hidden: Option<Array>,
}

impl GeneralLayerwiseModelAdapter for Gemma4LayerwiseAdapter {
    type Input<'a> = Gemma4Input<'a>;
    type Cache = Cache;
    type Layer = Gemma4Layer;
    type ForwardContext = Gemma4ForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![StaticUnitBindings::new(
            EMBEDDING_UNIT,
            self.bindings(&self.embedding, "model.language_model.embed_tokens", store)?,
        )?];
        if let Some(module) = &self.per_layer_embedding {
            units.push(StaticUnitBindings::new(
                PER_LAYER_EMBEDDING_UNIT,
                self.bindings(module, "model.language_model.embed_tokens_per_layer", store)?,
            )?);
        }
        if let Some(module) = &self.per_layer_projection {
            units.push(StaticUnitBindings::new(
                PER_LAYER_PROJECTION_UNIT,
                self.bindings(
                    module,
                    "model.language_model.per_layer_model_projection",
                    store,
                )?,
            )?);
        }
        if let Some(module) = &self.per_layer_norm {
            units.push(StaticUnitBindings::new(
                PER_LAYER_NORM_UNIT,
                self.bindings(
                    module,
                    "model.language_model.per_layer_projection_norm",
                    store,
                )?,
            )?);
        }
        units.push(StaticUnitBindings::new(
            NORM_UNIT,
            self.bindings(&self.norm, "model.language_model.norm", store)?,
        )?);
        if let Some(module) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                self.bindings(module, "lm_head", store)?,
            )?);
        }
        if let Some(module) = &self.vision {
            units.push(StaticUnitBindings::new(
                VISION_STATIC_UNIT,
                self.bindings(module, "model.vision_tower", store)?,
            )?);
        }
        if let Some(module) = &self.embed_vision {
            units.push(StaticUnitBindings::new(
                VISION_EMBED_UNIT,
                self.bindings(module, "model.embed_vision", store)?,
            )?);
        }
        if let Some(module) = &self.audio {
            units.push(StaticUnitBindings::new(
                AUDIO_STATIC_UNIT,
                self.bindings(module, "model.audio_tower", store)?,
            )?);
        }
        if let Some(module) = &self.embed_audio {
            units.push(StaticUnitBindings::new(
                AUDIO_EMBED_UNIT,
                self.bindings(module, "model.embed_audio", store)?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let mut index = 0;
        populate_module_from_lease(&mut self.embedding, &leases[index])?;
        index += 1;
        if let Some(module) = &mut self.per_layer_embedding {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.per_layer_projection {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.per_layer_norm {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        populate_module_from_lease(&mut self.norm, &leases[index])?;
        index += 1;
        if let Some(module) = &mut self.lm_head {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.vision {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.embed_vision {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.audio {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.embed_audio {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if index != leases.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 adapter received {} static leases, consumed {index}",
                leases.len()
            )));
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.kv.is_empty() {
            cache.reset_kv(&self.args);
        }
        if cache.kv.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 cache has {} layers, expected {}",
                cache.kv.len(),
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
        if let Gemma4Input::Prefill(typed) = input {
            input::validate(typed)?;
            cache.token_ids.clear();
            cache.reset_kv(&self.args);
            let mut parts = Vec::with_capacity(typed.parts.len());
            let mut vision_jobs = Vec::new();
            let mut audio_jobs = Vec::new();
            let scale = Array::from_f32((self.args.hidden_size as f32).sqrt());
            for part in typed.parts {
                match (part.modality, part.payload) {
                    (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                        parts.push(Gemma4PreparedPart::Ready {
                            tokens: tokens.clone(),
                            embeddings: self
                                .embedding
                                .forward(tokens, stream)?
                                .multiply(&scale, stream)?,
                        });
                    }
                    (
                        modality @ (input::Modality::Image | input::Modality::Video),
                        input::InputPayload::Tensor(pixels),
                    ) => {
                        let positions = part.metadata.patch_position_ids.ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 {} tensor input requires patch_position_ids metadata",
                                modality.as_str()
                            ))
                        })?;
                        let token_id = if modality == input::Modality::Image {
                            self.image_token_id
                        } else {
                            self.video_token_id
                        }
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 config does not define a {} token ID",
                                modality.as_str()
                            ))
                        })? as u32;
                        let (hidden, state) = self
                            .vision
                            .as_mut()
                            .ok_or_else(|| {
                                Error::UnsupportedArchitecture(format!(
                                    "gemma4 {} tensor input requires vision_config and vision weights",
                                    modality.as_str()
                                ))
                            })?
                            .begin(pixels, positions, stream)?;
                        let job = vision_jobs.len();
                        vision_jobs.push(Gemma4VisionJob { hidden, state });
                        parts.push(Gemma4PreparedPart::Vision { token_id, job });
                    }
                    (input::Modality::Audio, input::InputPayload::Tensor(features)) => {
                        let mask = part.metadata.audio_mask.ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "gemma4 audio tensor input requires audio_mask metadata".into(),
                            )
                        })?;
                        let token_id = self.audio_token_id.ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "gemma4 config does not define an audio token ID".into(),
                            )
                        })? as u32;
                        let (hidden, valid) = self
                            .audio
                            .as_mut()
                            .ok_or_else(|| {
                                Error::UnsupportedArchitecture(
                                    "gemma4 audio tensor input requires audio_config and audio weights".into(),
                                )
                            })?
                            .begin(features, mask, stream)?;
                        let job = audio_jobs.len();
                        audio_jobs.push(Gemma4AudioJob { hidden, valid });
                        parts.push(Gemma4PreparedPart::Audio { token_id, job });
                    }
                    (
                        modality @ (input::Modality::Image
                        | input::Modality::Video
                        | input::Modality::Audio),
                        input::InputPayload::Embeddings(embeddings),
                    ) => {
                        input::ensure_hidden_size(
                            embeddings,
                            self.args.hidden_size,
                            "Gemma 4 media embeddings",
                        )?;
                        let token_id = match modality {
                            input::Modality::Image => self.image_token_id,
                            input::Modality::Video => self.video_token_id,
                            input::Modality::Audio => self.audio_token_id,
                            input::Modality::Text => unreachable!(),
                        }
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 config does not define a {} token ID",
                                modality.as_str()
                            ))
                        })? as u32;
                        parts.push(Gemma4PreparedPart::Ready {
                            tokens: input::token_ids_array(
                                &vec![token_id; embeddings.dim(1) as usize],
                                stream,
                            )?,
                            embeddings: embeddings.clone(),
                        });
                    }
                    (modality, _) => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "gemma4 layerwise input does not support {} payloads of this kind",
                            modality.as_str()
                        )));
                    }
                }
            }
            if self.vision_depth == 0 && self.audio_depth == 0 {
                let token_parts = parts
                    .iter()
                    .filter_map(|part| match part {
                        Gemma4PreparedPart::Ready { tokens, .. } => Some(tokens),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let embedding_parts = parts
                    .iter()
                    .filter_map(|part| match part {
                        Gemma4PreparedPart::Ready { embeddings, .. } => Some(embeddings),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let tokens = concatenate_axis(&token_parts, 1, stream)?;
                let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
                cache.token_ids = resident::token_ids_from_array(&tokens, stream)?;
                let per_layer_inputs = self.prepare_per_layer_inputs(&tokens, &hidden, stream)?;
                let mask = (hidden.dim(1) > 1)
                    .then(|| create_causal_mask(hidden.dim(1), Some(0), None, None, stream))
                    .transpose()?;
                return Ok(LayerwiseForwardState {
                    hidden,
                    context: Gemma4ForwardContext {
                        per_layer_inputs,
                        mask,
                        sliding_mask: None,
                        position_offset: 0,
                        shared_kv: HashMap::new(),
                        parts,
                        vision_jobs,
                        audio_jobs,
                        tokens: Some(tokens),
                        needs_assembly: false,
                        draft_hidden: None,
                    },
                });
            }
            let hidden = vision_jobs
                .first()
                .map(|job| job.hidden.clone())
                .or_else(|| audio_jobs.first().map(|job| job.hidden.clone()))
                .or_else(|| {
                    parts.iter().find_map(|part| match part {
                        Gemma4PreparedPart::Ready { embeddings, .. } => Some(embeddings.clone()),
                        _ => None,
                    })
                })
                .expect("validated non-empty Gemma 4 input");
            return Ok(LayerwiseForwardState {
                hidden,
                context: Gemma4ForwardContext {
                    per_layer_inputs: None,
                    mask: None,
                    sliding_mask: None,
                    position_offset: 0,
                    shared_kv: HashMap::new(),
                    parts,
                    vision_jobs,
                    audio_jobs,
                    tokens: None,
                    needs_assembly: true,
                    draft_hidden: None,
                },
            });
        }
        let Gemma4Input::Decode(tokens) = input else {
            unreachable!()
        };
        cache
            .token_ids
            .extend(resident::token_ids_from_array(tokens, stream)?);
        let hidden = self.embedding.forward(tokens, stream)?.multiply(
            Array::from_f32((self.args.hidden_size as f32).sqrt()),
            stream,
        )?;
        let position_offset = cache
            .kv
            .iter()
            .flatten()
            .map(KeyValueCache::offset)
            .max()
            .unwrap_or(0);
        let mask = (hidden.dim(1) > 1)
            .then(|| create_causal_mask(hidden.dim(1), Some(position_offset), None, None, stream))
            .transpose()?;
        let per_layer_inputs = self.prepare_per_layer_inputs(tokens, &hidden, stream)?;
        Ok(LayerwiseForwardState {
            hidden,
            context: Gemma4ForwardContext {
                per_layer_inputs,
                mask,
                sliding_mask: None,
                position_offset,
                shared_kv: HashMap::new(),
                parts: Vec::new(),
                vision_jobs: Vec::new(),
                audio_jobs: Vec::new(),
                tokens: Some(tokens.clone()),
                needs_assembly: false,
                draft_hidden: None,
            },
        })
    }

    fn execution_group_count(&self) -> usize {
        1 + usize::from(self.vision_depth > 0) + usize::from(self.audio_depth > 0)
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        let mut index = 0;
        if self.vision_depth > 0 {
            if group == index {
                return Ok("vision_encoder".into());
            }
            index += 1;
        }
        if self.audio_depth > 0 {
            if group == index {
                return Ok("audio_encoder".into());
            }
            index += 1;
        }
        if group == index {
            Ok("text_decoder".into())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 has no execution group {group}"
            )))
        }
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        self.execution_group_id(group)
            .is_ok_and(|id| match id.as_str() {
                "vision_encoder" => !context.vision_jobs.is_empty(),
                "audio_encoder" => !context.audio_jobs.is_empty(),
                _ => true,
            })
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match self.execution_group_id(group)?.as_str() {
            "vision_encoder" => Ok(self.vision_depth),
            "audio_encoder" => Ok(self.audio_depth),
            "text_decoder" => Ok(self.args.num_hidden_layers as usize),
            _ => unreachable!(),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        match self.execution_group_id(group)?.as_str() {
            "vision_encoder" => Ok(Gemma4Layer::Vision(Box::new(VisionLayer::new(
                &self.vision.as_ref().expect("vision group").config,
                stream,
            )?))),
            "audio_encoder" => Ok(Gemma4Layer::Audio(Box::new(AudioLayer::new(
                self.audio_config.as_ref().expect("audio group"),
                stream,
            )?))),
            "text_decoder" => Ok(Gemma4Layer::Text(Box::new(TransformerBlock::new(
                &self.args,
                self.args.layer_type(index),
                index,
                stream,
            )?))),
            _ => unreachable!(),
        }
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        match self.execution_group_id(group).ok().as_deref() {
            Some("vision_encoder") => format!("model.vision_tower.encoder.layers.{index}"),
            Some("audio_encoder") => format!("model.audio_tower.layers.{index}"),
            _ => format!("model.language_model.layers.{index}"),
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        match self.execution_group_id(group).ok().as_deref() {
            Some("vision_encoder") => format!("gemma4.vision.{index:05}"),
            Some("audio_encoder") => format!("gemma4.audio.{index:05}"),
            _ => format!("gemma4.layer.{index:05}"),
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.bindings(layer, &self.layer_checkpoint_prefix(group, index), store)
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
            ("vision_encoder", Gemma4Layer::Vision(layer)) => {
                for job in &mut context.vision_jobs {
                    job.hidden = layer.forward(
                        &job.hidden,
                        &job.state.padding,
                        &job.state.cos,
                        &job.state.sin,
                        stream,
                    )?;
                }
                Ok(context.vision_jobs[0].hidden.clone())
            }
            ("audio_encoder", Gemma4Layer::Audio(layer)) => {
                for job in &mut context.audio_jobs {
                    job.hidden = layer.forward(&job.hidden, job.valid, stream)?;
                }
                Ok(context.audio_jobs[0].hidden.clone())
            }
            ("text_decoder", Gemma4Layer::Text(layer)) => {
                let per_layer_input = context
                    .per_layer_inputs
                    .as_ref()
                    .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
                    .transpose()?;
                let mask = if layer.layer_type == LayerType::SlidingAttention {
                    context.sliding_mask.as_ref().or(context.mask.as_ref())
                } else {
                    context.mask.as_ref()
                };
                Ok(layer.forward(
                    AttentionInput {
                        x: hidden,
                        mask,
                        cache: cache.kv[index].as_mut(),
                        position_offset: context.position_offset,
                        per_layer_input: per_layer_input.as_ref(),
                        shared_kv: Some(&mut context.shared_kv),
                        disable_generated_mask: false,
                        generated_sliding_window: None,
                    },
                    stream,
                )?)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 execution unit does not match group {group}"
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
        let mut arrays = context
            .shared_kv
            .values()
            .flat_map(|(keys, values)| [keys, values])
            .collect::<Vec<_>>();
        for job in &context.vision_jobs {
            arrays.push(&job.hidden);
            arrays.extend(job.state.retained_arrays());
        }
        arrays.extend(context.audio_jobs.iter().map(|job| &job.hidden));
        arrays
    }

    fn finish_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if !context.needs_assembly || group + 1 != self.execution_group_count() - 1 {
            if group + 1 == self.execution_group_count() {
                context.draft_hidden = Some(hidden.clone());
            }
            return Ok(hidden.clone());
        }
        if let (Some(vision), Some(embedder)) = (&self.vision, &mut self.embed_vision) {
            for job in &mut context.vision_jobs {
                job.hidden =
                    embedder.forward(&vision.finish(&job.hidden, &job.state, stream)?, stream)?;
            }
        }
        if let (Some(audio), Some(embedder)) = (&mut self.audio, &mut self.embed_audio) {
            for job in &mut context.audio_jobs {
                job.hidden =
                    embedder.forward(&audio.finish(&job.hidden, job.valid, stream)?, stream)?;
            }
        }
        let mut token_parts = Vec::with_capacity(context.parts.len());
        let mut embedding_parts = Vec::with_capacity(context.parts.len());
        for part in &context.parts {
            match part {
                Gemma4PreparedPart::Ready { tokens, embeddings } => {
                    token_parts.push(tokens.clone());
                    embedding_parts.push(embeddings.clone());
                }
                Gemma4PreparedPart::Vision { token_id, job } => {
                    let embeddings = context.vision_jobs[*job].hidden.clone();
                    token_parts.push(input::token_ids_array(
                        &vec![*token_id; embeddings.dim(0) as usize * embeddings.dim(1) as usize],
                        stream,
                    )?);
                    embedding_parts.push(if embeddings.dim(0) == 1 {
                        embeddings
                    } else {
                        embeddings.reshape(&[1, -1, embeddings.dim(2)], stream)?
                    });
                }
                Gemma4PreparedPart::Audio { token_id, job } => {
                    let embeddings = context.audio_jobs[*job].hidden.clone();
                    token_parts.push(input::token_ids_array(
                        &vec![*token_id; embeddings.dim(1) as usize],
                        stream,
                    )?);
                    embedding_parts.push(embeddings);
                }
            }
        }
        let token_refs = token_parts.iter().collect::<Vec<_>>();
        let embedding_refs = embedding_parts.iter().collect::<Vec<_>>();
        let tokens = concatenate_axis(&token_refs, 1, stream)?;
        let hidden = concatenate_axis(&embedding_refs, 1, stream)?;
        cache.token_ids = resident::token_ids_from_array(&tokens, stream)?;
        let per_layer_ids = self.media_safe_per_layer_ids(&tokens, stream)?;
        context.per_layer_inputs =
            self.prepare_per_layer_inputs(&per_layer_ids, &hidden, stream)?;
        let masks = resident::multimodal_attention_masks(
            &cache.token_ids,
            self.image_token_id.map(|id| id as u32),
            self.video_token_id.map(|id| id as u32),
            self.args.sliding_window,
        );
        context.mask = Some(masks.full);
        context.sliding_mask = Some(masks.sliding);
        context.tokens = Some(tokens);
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
        let mut logits = match self.lm_head.as_mut() {
            Some(head) => head.forward(&hidden, stream)?,
            None => self.embedding.as_linear(&hidden, stream)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(&logits.divide(Array::from_f32(softcap), stream)?, stream)?
                .multiply(Array::from_f32(softcap), stream)?;
        }
        Ok(logits)
    }

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        [
            "multi_modal_projector.",
            "model.multi_modal_projector.",
            "model.vision_embedder.",
        ]
        .iter()
        .any(|prefix| key.starts_with(prefix))
    }
}

/// Gemma 4 token generation using bounded text-layer execution.
pub type Generate<'a, S = crate::sampler::DefaultSampler> =
    crate::models::common::generation::Generate<'a, Gemma4LayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters, ops::ones_dtype, Array, Device, DeviceType, ExecutionContext,
        Stream,
    };

    use super::*;
    use crate::{
        cache::ConcatKeyValueCache,
        layerwise::LayerwiseLoadOptions,
        models::{
            common::generation::CausalLm,
            gemma4::{self as resident, Model, ModelInput},
            gemma4_audio::Gemma4AudioConfig,
            gemma4_vision::Gemma4VisionConfig,
            input as runtime_input,
        },
        offload::OffloadConfig,
    };

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "gemma4",
                "hidden_size": 8,
                "num_hidden_layers": 3,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "pad_token_id": 0,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "head_dim": 4,
                "attention_bias": false,
                "hidden_size_per_layer_input": 4,
                "vocab_size_per_layer_input": 32,
                "num_kv_shared_layers": 1,
                "layer_types": ["sliding_attention", "full_attention", "full_attention"],
                "sliding_window": 8,
                "final_logit_softcapping": 4.0
            }
        })
    }

    fn initialize(model: &mut Model, stream: &Stream) {
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
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.0005 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &Model) {
        let arrays = model
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| {
                let name = crate::module_binding::canonical_checkpoint_name(name).replacen(
                    "model.language_model.",
                    "language_model.model.",
                    1,
                );
                (name, *value)
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
        let mut args: ModelArgs = serde_json::from_value(config()["text_config"].clone()).unwrap();
        args.tie_word_embeddings = false;
        let mut fixture = Model::new(args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);

        let mut eager =
            resident::load_gemma4_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_gemma4_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut eager_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let mut layerwise_cache = layerwise.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = eager
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        inputs_embeds: None,
                        per_layer_input_ids: None,
                        mask: None,
                        sliding_mask: None,
                        cache: &mut eager_cache,
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        let report = layerwise.residency_report().unwrap();
        let resident_layers = report
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("gemma4.layer."))
            .filter(|unit| unit.device_resident())
            .count();
        assert!(resident_layers <= depth);
    }

    #[test]
    fn gemma4_per_layer_inputs_and_shared_kv_parity() {
        parity(1);
        parity(2);
    }

    #[test]
    fn gemma4_multimodal_vision_audio_and_text_group_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut args: ModelArgs = serde_json::from_value(config()["text_config"].clone()).unwrap();
        args.tie_word_embeddings = false;
        let vision = Gemma4VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            patch_size: 2,
            pooling_kernel_size: 2,
            position_embedding_size: 4,
            rms_norm_eps: 1e-6,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
        };
        let audio = Gemma4AudioConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            output_proj_dims: 8,
            conv_kernel_size: 3,
            attention_chunk_size: 4,
            attention_context_left: 4,
            attention_context_right: 0,
            attention_invalid_logits_value: -1.0e9,
            attention_logit_cap: 10.0,
            residual_weight: 0.5,
            rms_norm_eps: 1e-6,
            subsampling_conv_channels: vec![2, 2],
        };
        let mut fixture = Model::new_with_modalities(
            args,
            Some(20),
            Some(vision),
            Some(21),
            Some(22),
            Some(audio),
            gpu.stream(),
        )
        .unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);
        let mut value = config();
        value["image_token_id"] = serde_json::json!(20);
        value["video_token_id"] = serde_json::json!(21);
        value["audio_token_id"] = serde_json::json!(22);
        value["vision_config"] = serde_json::json!({
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "position_embedding_size": 4,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "standardize": false,
        });
        value["audio_config"] = serde_json::json!({
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "output_proj_dims": 8,
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 4,
            "attention_context_right": 0,
            "attention_invalid_logits_value": -1.0e9,
            "attention_logit_cap": 10.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 1e-6,
            "subsampling_conv_channels": [2, 2],
        });
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();

        let mut resident =
            resident::load_gemma4_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_gemma4_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let text = runtime_input::token_ids_array(&[1, 2], gpu.stream()).unwrap();
        let pixels = Array::zeros::<f32>(&[1, 4, 12], gpu.stream()).unwrap();
        let positions = Array::from_slice(&[0i32, 0, 0, 1, 1, 0, 1, 1], &[1, 4, 2]);
        let audio_features = Array::zeros::<f32>(&[1, 8, 128], gpu.stream()).unwrap();
        let audio_mask = Array::from_slice(&[true; 8], &[1, 8]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::patch_position_ids(&positions),
            ),
            runtime_input::InputPart::audio_tensor(
                &audio_features,
                runtime_input::InputMetadata::audio_mask(&audio_mask),
            ),
        ];
        let typed = runtime_input::ModelInput::new(&parts);
        let mut resident_cache = Cache::default();
        let mut layerwise_cache = layerwise.new_cache();
        let expected = resident
            .prefill_input_logits(typed, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = layerwise
            .prefill_input_logits(typed, &mut layerwise_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected);
        let report = layerwise.residency_report().unwrap();
        assert!(report
            .units()
            .iter()
            .any(|unit| unit.id().as_str().starts_with("gemma4.vision.")));
        assert!(report
            .units()
            .iter()
            .any(|unit| unit.id().as_str().starts_with("gemma4.audio.")));
    }
}
