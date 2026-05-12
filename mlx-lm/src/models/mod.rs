use std::path::Path;

use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Chat, Tokenizer as ChatTokenizer,
};
use mlx_rs::{
    error::Exception,
    ops::indexing::{IndexOp, NewAxis},
    Array,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{cache::ConcatKeyValueCache, error::Error};

pub mod gemma4;
pub mod llama;
pub mod qwen3;

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
}

impl ModelKind {
    fn from_model_type(model_type: &str) -> Result<Self, Error> {
        match model_type {
            "gemma4" | "gemma4_text" => Ok(Self::Gemma4),
            "llama" => Ok(Self::Llama),
            "qwen3" => Ok(Self::Qwen3),
            other => Err(Error::UnsupportedModelType(other.to_string())),
        }
    }
}

pub enum Model {
    Gemma4(gemma4::Model),
    Llama(llama::Model),
    Qwen3(qwen3::Model),
}

impl Model {
    pub fn model_type(&self) -> &str {
        match self {
            Self::Gemma4(model) => model.model_type(),
            Self::Llama(model) => model.model_type(),
            Self::Qwen3(model) => model.model_type(),
        }
    }

    pub fn generate<'a>(
        &'a mut self,
        cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
        temp: f32,
        prompt_tokens: &'a Array,
    ) -> Generate<'a> {
        match self {
            Self::Gemma4(model) => {
                Generate::Gemma4(gemma4::Generate::new(model, cache, temp, prompt_tokens))
            }
            Self::Llama(model) => {
                Generate::Llama(llama::Generate::new(model, cache, temp, prompt_tokens))
            }
            Self::Qwen3(model) => {
                Generate::Qwen3(qwen3::Generate::new(model, cache, temp, prompt_tokens))
            }
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

pub struct LoadedModel {
    model: Model,
    tokenizer: ChatTokenizer,
    chat_template: Option<String>,
    model_id: String,
    eos_token_ids: Vec<u32>,
}

impl LoadedModel {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        let metadata = read_model_metadata(model_dir)?;
        let model_type = if ModelKind::from_model_type(&metadata.model_type).is_ok() {
            metadata.model_type.clone()
        } else {
            metadata
                .text_config
                .as_ref()
                .map(|text_config| text_config.model_type.clone())
                .unwrap_or_else(|| metadata.model_type.clone())
        };
        let kind = ModelKind::from_model_type(&model_type)?;
        let tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(model_dir)?);
        let chat_template = load_chat_template(model_dir)?;
        let model = match kind {
            ModelKind::Gemma4 => Model::Gemma4(gemma4::load_gemma4_model(model_dir)?),
            ModelKind::Llama => Model::Llama(llama::load_llama_model(model_dir)?),
            ModelKind::Qwen3 => Model::Qwen3(qwen3::load_qwen3_model(model_dir)?),
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

    pub fn encode_to_array(&self, text: &str, add_special_tokens: bool) -> Result<Array, Error> {
        let ids = self.encode(text, add_special_tokens)?;
        Ok(Array::from(ids.as_slice()).index(NewAxis))
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
    ) -> Generate<'a> {
        self.model.generate(cache, temp, prompt_tokens)
    }
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    match ModelKind::from_model_type(&read_model_metadata(model_dir)?.model_type)? {
        ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model(model_dir)?)),
        ModelKind::Llama => Ok(Model::Llama(llama::load_llama_model(model_dir)?)),
        ModelKind::Qwen3 => Ok(Model::Qwen3(qwen3::load_qwen3_model(model_dir)?)),
    }
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let model_dir = model_dir.as_ref();
    match ModelKind::from_model_type(&read_model_metadata(model_dir)?.model_type)? {
        ModelKind::Gemma4 => gemma4::load_gemma4_tokenizer(model_dir),
        ModelKind::Llama => llama::load_llama_tokenizer(model_dir),
        ModelKind::Qwen3 => qwen3::load_qwen3_tokenizer(model_dir),
    }
}

fn read_model_metadata(model_dir: &Path) -> Result<ModelMetadata, Error> {
    let config_path = model_dir.join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
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
