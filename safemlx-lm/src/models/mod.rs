use std::path::Path;

use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    Array, Stream,
};
use safemlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Chat, Tokenizer as ChatTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{cache::ConcatKeyValueCache, error::Error};

pub mod common;
pub mod gemma4;
pub mod gemma4_assistant;
pub mod llama;
pub mod qwen3;
pub mod qwen3_5_moe;

#[derive(Debug, Clone, Deserialize)]
struct ModelMetadata {
    model_type: String,
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
    #[serde(default)]
    text_config: Option<TextModelMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
struct TextModelMetadata {
    model_type: String,
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenIdOrIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl TokenIdOrIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(id) => vec![id],
            Self::Multiple(ids) => ids,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ModelKind {
    Gemma4,
    Llama,
    Qwen3,
    Qwen35Moe,
}

impl ModelKind {
    fn from_model_type(model_type: &str) -> Result<Self, Error> {
        match model_type {
            "gemma4" | "gemma4_text" => Ok(Self::Gemma4),
            "llama" => Ok(Self::Llama),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35Moe),
            other => Err(Error::UnsupportedModelType(other.to_string())),
        }
    }
}

/// Details for a model config that this crate can load.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SupportedModelConfig {
    /// The runtime model implementation that will be used.
    pub kind: ModelKind,
    /// The top-level `model_type` from the submitted config.
    pub model_type: String,
    /// The resolved text model type used for dispatch.
    pub effective_model_type: String,
}

/// Result of checking whether a submitted model config is supported.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModelConfigSupport {
    /// The config is supported by this crate's loader.
    Supported(SupportedModelConfig),
    /// The config is not supported, with a human-readable reason.
    Unsupported { reason: String },
}

impl ModelConfigSupport {
    /// Returns true when this config is supported.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// Returns the unsupported reason, if this result is unsupported.
    pub fn unsupported_reason(&self) -> Option<&str> {
        match self {
            Self::Supported(_) => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// Checks a `config.json` string and reports whether it is supported.
pub fn check_model_config_json(config_json: &str) -> ModelConfigSupport {
    match serde_json::from_str::<Value>(config_json) {
        Ok(config) => check_model_config(&config),
        Err(error) => ModelConfigSupport::Unsupported {
            reason: format!("invalid model config JSON: {error}"),
        },
    }
}

/// Checks a parsed model config value and reports whether it is supported.
pub fn check_model_config(config: &Value) -> ModelConfigSupport {
    let metadata = match serde_json::from_value::<ModelMetadata>(config.clone()) {
        Ok(metadata) => metadata,
        Err(error) => {
            return ModelConfigSupport::Unsupported {
                reason: format!("invalid model config metadata: {error}"),
            };
        }
    };

    let effective_model_type = effective_model_type(&metadata);
    let kind = match ModelKind::from_model_type(&effective_model_type) {
        Ok(kind) => kind,
        Err(error) => {
            return ModelConfigSupport::Unsupported {
                reason: error.to_string(),
            };
        }
    };

    if let Err(error) = validate_model_config(kind, config) {
        return ModelConfigSupport::Unsupported {
            reason: error.to_string(),
        };
    }

    ModelConfigSupport::Supported(SupportedModelConfig {
        kind,
        model_type: metadata.model_type,
        effective_model_type,
    })
}

/// Reads `config.json` from a model directory and reports whether it is supported.
pub fn check_model_dir(model_dir: impl AsRef<Path>) -> ModelConfigSupport {
    let config_path = model_dir.as_ref().join("config.json");
    match std::fs::read_to_string(&config_path) {
        Ok(config_json) => check_model_config_json(&config_json),
        Err(error) => ModelConfigSupport::Unsupported {
            reason: format!("could not read {}: {error}", config_path.display()),
        },
    }
}

fn validate_model_config(kind: ModelKind, config: &Value) -> Result<(), Error> {
    match kind {
        ModelKind::Gemma4 => gemma4::validate_model_config_value(config),
        ModelKind::Llama => {
            serde_json::from_value::<llama::ModelArgs>(config.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid llama config: {error}"))
            })?;
            Ok(())
        }
        ModelKind::Qwen3 => {
            serde_json::from_value::<qwen3::ModelArgs>(config.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid qwen3 config: {error}"))
            })?;
            Ok(())
        }
        ModelKind::Qwen35Moe => qwen3_5_moe::validate_model_config_value(config),
    }
}

pub enum Model {
    Gemma4(gemma4::Model),
    Llama(llama::Model),
    Qwen3(qwen3::Model),
    Qwen35Moe(qwen3_5_moe::Model),
}

impl Model {
    pub fn model_type(&self) -> &str {
        match self {
            Self::Gemma4(model) => model.model_type(),
            Self::Llama(model) => model.model_type(),
            Self::Qwen3(model) => model.model_type(),
            Self::Qwen35Moe(model) => model.model_type(),
        }
    }

    pub fn generate<'a>(
        &'a mut self,
        cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Generate<'a> {
        match self {
            Self::Gemma4(model) => Generate::Gemma4(gemma4::Generate::new(
                model,
                cache,
                temp,
                prompt_tokens,
                prng_key,
                stream,
            )),
            Self::Llama(model) => Generate::Llama(llama::Generate::new(
                model,
                cache,
                temp,
                prompt_tokens,
                prng_key,
                stream,
            )),
            Self::Qwen3(model) => Generate::Qwen3(qwen3::Generate::new(
                model,
                cache,
                temp,
                prompt_tokens,
                prng_key,
                stream,
            )),
            Self::Qwen35Moe(_) => {
                panic!("qwen3_5_moe requires ModelCache; use generate_with_cache")
            }
        }
    }

    pub fn new_cache(&self) -> ModelCache {
        match self {
            Self::Gemma4(_) | Self::Llama(_) | Self::Qwen3(_) => ModelCache::KeyValue(Vec::new()),
            Self::Qwen35Moe(model) => ModelCache::Qwen35Moe(model.new_cache()),
        }
    }

    pub fn generate_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::KeyValue(cache)) => ModelGenerate::Gemma4(
                gemma4::Generate::new(model, cache, temp, prompt_tokens, prng_key, stream),
            ),
            (Self::Llama(model), ModelCache::KeyValue(cache)) => ModelGenerate::Llama(
                llama::Generate::new(model, cache, temp, prompt_tokens, prng_key, stream),
            ),
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => ModelGenerate::Qwen3(
                qwen3::Generate::new(model, cache, temp, prompt_tokens, prng_key, stream),
            ),
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => ModelGenerate::Qwen35Moe(
                qwen3_5_moe::Generate::new(model, cache, temp, prompt_tokens, prng_key, stream),
            ),
            _ => panic!("model cache type does not match model kind"),
        }
    }
}

pub enum Generate<'a> {
    Gemma4(gemma4::Generate<'a, ConcatKeyValueCache>),
    Llama(llama::Generate<'a, ConcatKeyValueCache>),
    Qwen3(qwen3::Generate<'a, ConcatKeyValueCache>),
}

impl Iterator for Generate<'_> {
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Gemma4(generate) => generate.next(),
            Self::Llama(generate) => generate.next(),
            Self::Qwen3(generate) => generate.next(),
        }
    }
}

pub enum ModelCache {
    KeyValue(Vec<Option<ConcatKeyValueCache>>),
    Qwen35Moe(qwen3_5_moe::Cache),
}

pub enum ModelGenerate<'a> {
    Gemma4(gemma4::Generate<'a, ConcatKeyValueCache>),
    Llama(llama::Generate<'a, ConcatKeyValueCache>),
    Qwen3(qwen3::Generate<'a, ConcatKeyValueCache>),
    Qwen35Moe(qwen3_5_moe::Generate<'a>),
}

impl Iterator for ModelGenerate<'_> {
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Gemma4(generate) => generate.next(),
            Self::Llama(generate) => generate.next(),
            Self::Qwen3(generate) => generate.next(),
            Self::Qwen35Moe(generate) => generate.next(),
        }
    }
}

pub struct LoadedModel {
    model: Model,
    tokenizer: ChatTokenizer,
    chat_template: Option<String>,
    model_id: String,
    eos_token_ids: Vec<u32>,
}

impl LoadedModel {
    pub fn load(model_dir: impl AsRef<Path>, stream: &Stream) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        let metadata = read_model_metadata(model_dir)?;
        let model_type = effective_model_type(&metadata);
        let kind = ModelKind::from_model_type(&model_type)?;
        let tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(model_dir)?);
        let chat_template = load_chat_template(model_dir)?;
        let model = match kind {
            ModelKind::Gemma4 => Model::Gemma4(gemma4::load_gemma4_model(model_dir, stream)?),
            ModelKind::Llama => Model::Llama(llama::load_llama_model(model_dir, stream)?),
            ModelKind::Qwen3 => Model::Qwen3(qwen3::load_qwen3_model(model_dir, stream)?),
            ModelKind::Qwen35Moe => {
                Model::Qwen35Moe(qwen3_5_moe::load_qwen3_5_moe_model(model_dir, stream)?)
            }
        };
        let eos_token_ids = metadata
            .eos_token_id
            .or_else(|| {
                metadata
                    .text_config
                    .and_then(|text_config| text_config.eos_token_id)
            })
            .map(TokenIdOrIds::into_vec)
            .unwrap_or_default();

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            model_id: model_type,
            eos_token_ids,
        })
    }

    pub fn model_type(&self) -> &str {
        self.model.model_type()
    }

    pub fn model_id_for_template(&self) -> &str {
        &self.model_id
    }

    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    pub fn apply_chat_template<'a, I, R, T>(
        &'a mut self,
        conversations: I,
        tools: Option<&'a [serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + 'a,
    {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };

        let rendered = self.tokenizer.apply_chat_template(
            template,
            ApplyChatTemplateArgs {
                conversations,
                tools,
                documents: None,
                model_id: &self.model_id,
                chat_template_id: None,
                add_generation_prompt: Some(add_generation_prompt),
                continue_final_message: None,
            },
        )?;
        Ok(rendered.into_iter().next())
    }

    pub fn apply_chat_template_json(
        &mut self,
        conversations: impl IntoIterator<Item = Vec<serde_json::Value>>,
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, Error> {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };

        let rendered = self.tokenizer.apply_chat_template_json(
            template,
            conversations,
            tools,
            &self.model_id,
            add_generation_prompt,
        )?;
        Ok(rendered.into_iter().next())
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Error> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)?
            .get_ids()
            .to_vec())
    }

    pub fn encode_to_array(
        &self,
        text: &str,
        add_special_tokens: bool,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = self.encode(text, add_special_tokens)?;
        Ok(Array::from(ids.as_slice()).try_index_device(NewAxis, stream)?)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(Into::into)
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn is_eos_token(&self, id: u32) -> bool {
        self.eos_token_ids.contains(&id)
    }

    pub fn generate<'a>(
        &'a mut self,
        cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Generate<'a> {
        self.model
            .generate(cache, temp, prompt_tokens, prng_key, stream)
    }

    pub fn new_cache(&self) -> ModelCache {
        self.model.new_cache()
    }

    pub fn generate_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        prompt_tokens: &'a Array,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        self.model
            .generate_with_cache(cache, temp, prompt_tokens, prng_key, stream)
    }

    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
}

pub fn load_model(model_dir: impl AsRef<Path>, stream: &Stream) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let metadata = read_model_metadata(model_dir)?;
    match ModelKind::from_model_type(&effective_model_type(&metadata))? {
        ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model(model_dir, stream)?)),
        ModelKind::Llama => Ok(Model::Llama(llama::load_llama_model(model_dir, stream)?)),
        ModelKind::Qwen3 => Ok(Model::Qwen3(qwen3::load_qwen3_model(model_dir, stream)?)),
        ModelKind::Qwen35Moe => Ok(Model::Qwen35Moe(qwen3_5_moe::load_qwen3_5_moe_model(
            model_dir, stream,
        )?)),
    }
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let model_dir = model_dir.as_ref();
    let metadata = read_model_metadata(model_dir)?;
    match ModelKind::from_model_type(&effective_model_type(&metadata))? {
        ModelKind::Gemma4 => gemma4::load_gemma4_tokenizer(model_dir),
        ModelKind::Llama => llama::load_llama_tokenizer(model_dir),
        ModelKind::Qwen3 => qwen3::load_qwen3_tokenizer(model_dir),
        ModelKind::Qwen35Moe => qwen3_5_moe::load_qwen3_5_moe_tokenizer(model_dir),
    }
}

fn read_model_metadata(model_dir: &Path) -> Result<ModelMetadata, Error> {
    let config_path = model_dir.join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

fn effective_model_type(metadata: &ModelMetadata) -> String {
    if matches!(metadata.model_type.as_str(), "gemma4" | "qwen3_5_moe") {
        metadata
            .text_config
            .as_ref()
            .map(|text_config| text_config.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    } else if ModelKind::from_model_type(&metadata.model_type).is_ok() {
        metadata.model_type.clone()
    } else {
        metadata
            .text_config
            .as_ref()
            .map(|text_config| text_config.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    }
}

fn load_chat_template(model_dir: &Path) -> Result<Option<String>, Error> {
    let config_path = model_dir.join("tokenizer_config.json");
    if config_path.exists() {
        if let Some(template) = load_model_chat_template_from_file(config_path)? {
            return Ok(Some(template));
        }
    }

    let metadata = read_model_metadata(model_dir)?;
    if metadata.model_type == "gemma4"
        || metadata
            .text_config
            .as_ref()
            .is_some_and(|text_config| text_config.model_type == "gemma4_text")
    {
        return Ok(Some(GEMMA4_TEXT_TEMPLATE.to_string()));
    }

    Ok(None)
}

const GEMMA4_TEXT_TEMPLATE: &str = r#"<bos>{% for message in messages %}{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}<|turn>{{ role }}
{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] }}{% elif content['type'] == 'image' %}<|image>{% elif content['type'] == 'audio' %}<|audio>{% endif %}{% endfor %}{% endif %}<turn|>
{% endfor %}{% if add_generation_prompt %}<|turn>model
{% endif %}"#;

#[cfg(test)]
mod tests {
    use super::{check_model_config, check_model_config_json, check_model_dir, load_tokenizer};
    use serde_json::json;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };
    use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

    static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_model_dir(config: &str) -> std::path::PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "model_metadata_test_{}_{}_{}",
            std::process::id(),
            id,
            counter
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), config).unwrap();
        Tokenizer::new(WordLevel::default())
            .save(dir.join("tokenizer.json"), false)
            .unwrap();
        dir
    }

    #[test]
    fn load_tokenizer_accepts_top_level_qwen3_5_moe_metadata() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text"
              }
            }"#,
        );
        let tokenizer = load_tokenizer(&dir).unwrap();
        assert_eq!(tokenizer.get_vocab_size(false), 0);
    }

    #[test]
    fn check_model_config_reports_supported_llama() {
        let support = check_model_config(&json!({
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "head_dim": 4
        }));

        assert!(support.is_supported(), "{support:?}");
    }

    #[test]
    fn check_model_config_reports_unsupported_model_type() {
        let support = check_model_config(&json!({
            "model_type": "not_a_model"
        }));

        assert!(!support.is_supported());
        assert_eq!(
            support.unsupported_reason(),
            Some("unsupported model type: not_a_model")
        );
    }

    #[test]
    fn check_model_config_json_reports_invalid_json() {
        let support = check_model_config_json("{not json");

        assert!(!support.is_supported());
        assert!(support
            .unsupported_reason()
            .unwrap()
            .starts_with("invalid model config JSON:"));
    }

    #[test]
    fn check_model_config_reports_qwen3_5_moe_missing_text_config() {
        let support = check_model_config(&json!({
            "model_type": "qwen3_5_moe"
        }));

        assert!(!support.is_supported());
        assert_eq!(
            support.unsupported_reason(),
            Some("unsupported model architecture: qwen3_5_moe config is missing text_config")
        );
    }

    #[test]
    fn check_model_config_reports_supported_qwen3_5_moe() {
        let support = check_model_config(&json!({
            "model_type": "qwen3_5_moe",
            "image_token_id": 248056,
            "video_token_id": 248057,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 128,
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128
            }
        }));

        assert_eq!(
            support,
            super::ModelConfigSupport::Supported(super::SupportedModelConfig {
                kind: super::ModelKind::Qwen35Moe,
                model_type: "qwen3_5_moe".to_string(),
                effective_model_type: "qwen3_5_moe_text".to_string(),
            })
        );
    }

    #[test]
    fn check_model_config_reports_gemma4_moe_unsupported() {
        let support = check_model_config(&json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "rms_norm_eps": 0.00001,
                "vocab_size": 32,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "head_dim": 4,
                "enable_moe_block": true
            }
        }));

        assert!(!support.is_supported());
        assert_eq!(
            support.unsupported_reason(),
            Some("unsupported model architecture: Gemma 4 MoE models are not supported yet")
        );
    }

    #[test]
    fn check_model_dir_reads_config_json() {
        let dir = temp_model_dir(
            r#"{
              "model_type": "llama",
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "rms_norm_eps": 0.00001,
              "vocab_size": 32,
              "num_key_value_heads": 2,
              "max_position_embeddings": 128,
              "head_dim": 4
            }"#,
        );

        let support = check_model_dir(&dir);
        assert!(support.is_supported(), "{support:?}");
    }
}
