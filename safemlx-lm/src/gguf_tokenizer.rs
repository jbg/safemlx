use std::collections::{HashMap, HashSet};

use safemlx::ops::GgufMetadataValue;
use serde_json::{Map, Value};
use tokenizers::{
    decoders::{
        byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence,
        strip::Strip,
    },
    models::{
        bpe::{Vocab, BPE},
        unigram::Unigram,
    },
    normalizers::{Replace, NFC},
    pre_tokenizers::{
        byte_level::ByteLevel,
        metaspace::{Metaspace, PrependScheme},
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    processors::template::TemplateProcessing,
    AddedToken, DecoderWrapper, SplitDelimiterBehavior, Tokenizer,
};

use crate::error::Error;

pub(crate) struct GgufTokenizer {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) template_kwargs: Map<String, Value>,
}

pub(crate) fn from_metadata(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Option<GgufTokenizer>, Error> {
    if let Some(json) = metadata
        .get("tokenizer.huggingface.json")
        .and_then(GgufMetadataValue::as_str)
    {
        let tokenizer = Tokenizer::from_bytes(json.as_bytes())?;
        return Ok(Some(GgufTokenizer {
            tokenizer,
            template_kwargs: special_token_kwargs(metadata)?,
        }));
    }

    let Some(tokens) = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    else {
        return Ok(None);
    };
    let Some(model_type) = metadata
        .get("tokenizer.ggml.model")
        .and_then(GgufMetadataValue::as_str)
    else {
        return Ok(None);
    };
    let architecture = metadata
        .get("general.architecture")
        .and_then(GgufMetadataValue::as_str)
        .unwrap_or_default();

    let mut tokenizer = match (architecture, model_type) {
        ("gemma4", _) => build_gemma(tokens, metadata)?,
        ("llama", _) => build_llama(tokens, metadata)?,
        (_, "llama") => build_llama(tokens, metadata)?,
        (_, "gpt2") => build_gpt(tokens, metadata)?,
        _ => return Ok(None),
    };
    register_special_tokens(&mut tokenizer, tokens, metadata)?;
    configure_post_processor(&mut tokenizer, tokens, metadata)?;

    Ok(Some(GgufTokenizer {
        tokenizer,
        template_kwargs: special_token_kwargs(metadata)?,
    }))
}

pub(crate) fn template_kwargs(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Map<String, Value>, Error> {
    special_token_kwargs(metadata)
}

fn build_gpt(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Tokenizer, Error> {
    let merges = required_merges(metadata)?;
    let mut builder = BPE::builder()
        .vocab_and_merges(vocab(tokens)?, merges)
        .fuse_unk(true);
    if let Some(unknown) = special_token(metadata, tokens, "unknown_token_id")? {
        builder = builder.unk_token(unknown);
    }
    let model = builder.build()?;
    let mut tokenizer = Tokenizer::new(model);
    let pre_tokenizer = metadata
        .get("tokenizer.ggml.pre")
        .and_then(GgufMetadataValue::as_str)
        .unwrap_or_default();
    let architecture = metadata
        .get("general.architecture")
        .and_then(GgufMetadataValue::as_str)
        .unwrap_or_default();
    if pre_tokenizer == "lfm2" || matches!(architecture, "lfm2" | "lfm2moe") {
        const LFM2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Split(Split::new(
                SplitPattern::Regex(LFM2_PATTERN.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));
    } else if matches!(pre_tokenizer, "qwen2" | "qwen35")
        || matches!(architecture, "qwen3" | "qwen3moe" | "qwen35" | "qwen35moe")
    {
        const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        tokenizer.with_normalizer(Some(NFC))?;
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Split(Split::new(
                SplitPattern::Regex(QWEN_PATTERN.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));
    } else {
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
    }
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, true)));
    Ok(tokenizer)
}

fn build_llama(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Tokenizer, Error> {
    let merges = match metadata
        .get("tokenizer.ggml.merges")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(values) => parse_merges(values)?,
        None => derive_merges(tokens, metadata)?,
    };
    let mut builder = BPE::builder()
        .vocab_and_merges(vocab(tokens)?, merges)
        .fuse_unk(true)
        .byte_fallback(true);
    if let Some(unknown) = special_token(metadata, tokens, "unknown_token_id")? {
        builder = builder.unk_token(unknown);
    }
    let mut tokenizer = Tokenizer::new(builder.build()?);
    let is_llama3 = metadata
        .get("tokenizer.ggml.model")
        .and_then(GgufMetadataValue::as_str)
        != Some("llama");
    let add_prefix_space =
        metadata_bool(metadata, "tokenizer.ggml.add_space_prefix")?.unwrap_or(!is_llama3);
    if is_llama3 {
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
    } else {
        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            '▁',
            if add_prefix_space {
                PrependScheme::Always
            } else {
                PrependScheme::Never
            },
            true,
        )));
    }
    tokenizer.with_decoder(Some(sentencepiece_decoder(add_prefix_space, is_llama3)?));
    Ok(tokenizer)
}

fn build_gemma(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Tokenizer, Error> {
    let scores = required_scores(metadata, tokens.len())?;
    let vocabulary = tokens
        .iter()
        .zip(scores)
        .map(|(token, score)| {
            let token = if token == "<0x09>" {
                "\t".to_string()
            // Gemma GGUF vocabularies may contain literal newline tokens. Only
            // ordinary spaces use SentencePiece's metaspace representation;
            // converting every whitespace character would flatten line breaks.
            } else if !token.is_empty() && token.chars().all(|character| character == ' ') {
                "▁".repeat(token.chars().count())
            } else {
                token.clone()
            };
            (token, f64::from(score))
        })
        .collect();
    let unknown = special_id(metadata, "unknown_token_id")?;
    let model = Unigram::from(vocabulary, unknown, true)?;
    let mut tokenizer = Tokenizer::new(model);
    let add_prefix_space =
        metadata_bool(metadata, "tokenizer.ggml.add_space_prefix")?.unwrap_or(true);
    tokenizer.with_normalizer(Some(Replace::new(" ", "▁")?))?;
    tokenizer.with_pre_tokenizer(Some(Metaspace::new(
        '▁',
        if add_prefix_space {
            PrependScheme::Always
        } else {
            PrependScheme::Never
        },
        true,
    )));
    tokenizer.with_decoder(Some(gemma_decoder(add_prefix_space)?));
    Ok(tokenizer)
}

fn gemma_decoder(add_prefix_space: bool) -> Result<DecoderSequence, Error> {
    let mut decoders = vec![
        DecoderWrapper::Replace(Replace::new("▁", " ")?),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
    ];
    if add_prefix_space {
        decoders.push(DecoderWrapper::Strip(Strip::new(' ', 1, 0)));
    }
    Ok(DecoderSequence::new(decoders))
}

fn sentencepiece_decoder(
    add_prefix_space: bool,
    append_byte_level: bool,
) -> Result<DecoderSequence, Error> {
    let mut decoders = vec![
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
        DecoderWrapper::Replace(Replace::new("▁", " ")?),
    ];
    if append_byte_level {
        decoders.push(DecoderWrapper::ByteLevel(ByteLevel::new(
            false, false, true,
        )));
    }
    if add_prefix_space {
        decoders.push(DecoderWrapper::Strip(Strip::new(' ', 1, 0)));
    }
    Ok(DecoderSequence::new(decoders))
}

fn vocab(tokens: &[String]) -> Result<Vocab, Error> {
    tokens
        .iter()
        .enumerate()
        .map(|(id, token)| {
            u32::try_from(id)
                .map(|id| (token.clone(), id))
                .map_err(|_| tokenizer_error("vocabulary has more than u32::MAX entries"))
        })
        .collect()
}

fn required_merges(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Vec<(String, String)>, Error> {
    let values = metadata
        .get("tokenizer.ggml.merges")
        .and_then(GgufMetadataValue::as_strings)
        .ok_or_else(|| tokenizer_error("BPE metadata is missing tokenizer.ggml.merges"))?;
    parse_merges(values)
}

fn parse_merges(values: &[String]) -> Result<Vec<(String, String)>, Error> {
    values
        .iter()
        .map(|merge| {
            merge
                .split_once(' ')
                .map(|(left, right)| (left.to_string(), right.to_string()))
                .ok_or_else(|| tokenizer_error(format!("invalid BPE merge {merge:?}")))
        })
        .collect()
}

fn derive_merges(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Vec<(String, String)>, Error> {
    let scores = required_scores(metadata, tokens.len())?;
    let token_scores = tokens
        .iter()
        .cloned()
        .zip(scores.iter().copied())
        .collect::<HashMap<_, _>>();
    let mut merges = Vec::new();
    for (token, score) in tokens.iter().zip(scores) {
        let mut local = token
            .char_indices()
            .skip(1)
            .filter_map(|(index, _)| {
                let (left, right) = token.split_at(index);
                Some((
                    left.to_string(),
                    right.to_string(),
                    *token_scores.get(left)?,
                    *token_scores.get(right)?,
                    score,
                ))
            })
            .collect::<Vec<_>>();
        local.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.3.total_cmp(&a.3)));
        merges.extend(local);
    }
    merges.sort_by(|a, b| b.4.total_cmp(&a.4));
    Ok(merges
        .into_iter()
        .map(|(left, right, _, _, _)| (left, right))
        .collect())
}

fn required_scores(
    metadata: &HashMap<String, GgufMetadataValue>,
    expected: usize,
) -> Result<Vec<f32>, Error> {
    let scores = metadata
        .get("tokenizer.ggml.scores")
        .and_then(GgufMetadataValue::as_array)
        .and_then(|values| values.to_f32_vec())
        .ok_or_else(|| tokenizer_error("metadata is missing tokenizer.ggml.scores"))?;
    if scores.len() != expected {
        return Err(tokenizer_error(format!(
            "tokenizer.ggml.scores has {} entries for a {expected}-token vocabulary",
            scores.len()
        )));
    }
    Ok(scores)
}

fn register_special_tokens(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<(), Error> {
    let mut ids = HashSet::new();
    let mut user_defined = Vec::new();
    if let Some(types) = metadata
        .get("tokenizer.ggml.token_type")
        .and_then(GgufMetadataValue::as_array)
        .and_then(|values| values.to_i64_vec())
    {
        if types.len() != tokens.len() {
            return Err(tokenizer_error(
                "tokenizer.ggml.token_type length does not match the vocabulary",
            ));
        }
        ids.extend(
            types
                .iter()
                .enumerate()
                .filter_map(|(id, kind)| (*kind == 3).then_some(id)),
        );
        user_defined.extend(
            types
                .iter()
                .enumerate()
                .filter_map(|(id, kind)| (*kind == 4).then_some(id)),
        );
    }
    for name in [
        "bos_token_id",
        "eos_token_id",
        "unknown_token_id",
        "padding_token_id",
        "separator_token_id",
    ] {
        if name == "eos_token_id" {
            ids.extend(special_ids(metadata, name)?);
        } else if let Some(id) = special_id(metadata, name)? {
            ids.insert(id);
        }
    }
    for token in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] {
        if let Some(id) = tokens.iter().position(|candidate| candidate == token) {
            ids.insert(id);
        }
    }
    let added = ids
        .into_iter()
        .map(|id| {
            tokens
                .get(id)
                .cloned()
                .map(|token| AddedToken::from(token, true).normalized(false))
                .ok_or_else(|| {
                    tokenizer_error(format!("special token id {id} is outside the vocabulary"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    tokenizer.add_special_tokens(added)?;
    tokenizer.add_tokens(
        user_defined
            .into_iter()
            .map(|id| AddedToken::from(tokens[id].clone(), false).normalized(false)),
    )?;
    Ok(())
}

fn configure_post_processor(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<(), Error> {
    let bos = metadata_bool(metadata, "tokenizer.ggml.add_bos_token")?
        .unwrap_or(false)
        .then(|| special_token(metadata, tokens, "bos_token_id"))
        .transpose()?
        .flatten();
    let eos = metadata_bool(metadata, "tokenizer.ggml.add_eos_token")?
        .unwrap_or(false)
        .then(|| special_token(metadata, tokens, "eos_token_id"))
        .transpose()?
        .flatten();
    if bos.is_none() && eos.is_none() {
        return Ok(());
    }
    let mut single = Vec::new();
    let mut pair = Vec::new();
    let mut specials = Vec::new();
    if let Some(token) = &bos {
        single.push(token.clone());
        pair.push(token.clone());
        specials.push((token.clone(), tokenizer.token_to_id(token).unwrap()));
    }
    single.push("$A".into());
    pair.push("$A".into());
    if let Some(token) = &eos {
        single.push(token.clone());
        pair.push(token.clone());
        specials.push((token.clone(), tokenizer.token_to_id(token).unwrap()));
    }
    pair.push("$B:1".into());
    if let Some(token) = &eos {
        pair.push(format!("{token}:1"));
    }
    specials.sort_by_key(|(_, id)| *id);
    specials.dedup_by_key(|(_, id)| *id);
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single(single.join(" "))
            .map_err(|error| tokenizer_error(error.to_string()))?
            .try_pair(pair.join(" "))
            .map_err(|error| tokenizer_error(error.to_string()))?
            .special_tokens(specials)
            .build()
            .map_err(|error| tokenizer_error(error.to_string()))?,
    ));
    Ok(())
}

fn special_token_kwargs(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Map<String, Value>, Error> {
    let Some(tokens) = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    else {
        return Ok(Map::new());
    };
    let mut kwargs = Map::new();
    for (metadata_name, kwarg_name) in [
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("unknown_token_id", "unk_token"),
        ("padding_token_id", "pad_token"),
        ("separator_token_id", "sep_token"),
    ] {
        if let Some(token) = special_token(metadata, tokens, metadata_name)? {
            kwargs.insert(kwarg_name.into(), Value::String(token));
        }
    }
    Ok(kwargs)
}

fn special_token(
    metadata: &HashMap<String, GgufMetadataValue>,
    tokens: &[String],
    name: &str,
) -> Result<Option<String>, Error> {
    let id = if name == "eos_token_id" {
        special_ids(metadata, name)?.into_iter().next()
    } else {
        special_id(metadata, name)?
    };
    id.map(|id| {
        tokens.get(id).cloned().ok_or_else(|| {
            tokenizer_error(format!(
                "tokenizer.ggml.{name}={id} is outside the vocabulary"
            ))
        })
    })
    .transpose()
}

fn special_ids(
    metadata: &HashMap<String, GgufMetadataValue>,
    name: &str,
) -> Result<Vec<usize>, Error> {
    let key = format!("tokenizer.ggml.{name}");
    let Some(value) = metadata.get(&key) else {
        return Ok(Vec::new());
    };
    let values = value
        .to_i64_vec()
        .ok_or_else(|| tokenizer_error(format!("{key} must be an integer or integer array")))?;
    values
        .into_iter()
        .map(|id| {
            u32::try_from(id).map(|id| id as usize).map_err(|_| {
                tokenizer_error(format!(
                    "{key} must contain integers from 0 through {}",
                    u32::MAX
                ))
            })
        })
        .collect()
}

fn special_id(
    metadata: &HashMap<String, GgufMetadataValue>,
    name: &str,
) -> Result<Option<usize>, Error> {
    let key = format!("tokenizer.ggml.{name}");
    metadata
        .get(&key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|id| usize::try_from(id).ok())
                .ok_or_else(|| tokenizer_error(format!("{key} must be a non-negative integer")))
        })
        .transpose()
}

fn metadata_bool(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<bool>, Error> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| tokenizer_error(format!("{key} must be a boolean")))
        })
        .transpose()
}

fn tokenizer_error(message: impl Into<String>) -> Error {
    Error::GgufTokenizer(message.into())
}

#[cfg(test)]
mod tests {
    use safemlx::ops::{GgufMetadataArray, GgufMetadataValue};

    use super::*;

    #[test]
    fn builds_embedded_gpt_tokenizer() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("qwen3".into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                GgufMetadataValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "<eos>".into(),
                    "h".into(),
                    "e".into(),
                    "l".into(),
                    "o".into(),
                    "he".into(),
                    "ll".into(),
                    "hell".into(),
                    "hello".into(),
                    "<eos2>".into(),
                ])),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "h e".into(),
                    "l l".into(),
                    "he ll".into(),
                    "hell o".into(),
                ])),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![0, 9])),
            ),
            (
                "tokenizer.ggml.add_eos_token".into(),
                GgufMetadataValue::Bool(true),
            ),
        ]);

        let loaded = from_metadata(&metadata).unwrap().unwrap();
        let encoding = loaded.tokenizer.encode("hello", true).unwrap();
        assert_eq!(encoding.get_ids(), &[8, 0]);
        assert_eq!(loaded.tokenizer.decode(&[0, 9], true).unwrap(), "");
        assert_eq!(loaded.template_kwargs["eos_token"], "<eos>");
    }

    #[test]
    fn uses_lfm2_pretokenizer_without_unicode_normalization() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("lfm2".into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                GgufMetadataValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                GgufMetadataValue::String("lfm2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "<eos>".into(),
                    "a".into(),
                ])),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![])),
            ),
        ]);
        let tokenizer = from_metadata(&metadata).unwrap().unwrap().tokenizer;
        let serialized = tokenizer.to_string(false).unwrap();
        assert!(serialized.contains(r"\\p{N}{1,3}"));
        assert!(serialized.contains("ByteLevel"));
        assert!(tokenizer.get_normalizer().is_none());
    }

    #[test]
    fn derives_llama_merges_from_scores() {
        let metadata = sentencepiece_metadata("llama");
        let loaded = from_metadata(&metadata).unwrap().unwrap();
        let encoding = loaded.tokenizer.encode("hi", false).unwrap();
        assert_eq!(encoding.get_ids(), &[5]);
        assert_eq!(
            loaded.tokenizer.decode(encoding.get_ids(), false).unwrap(),
            "hi"
        );
    }

    #[test]
    fn builds_gemma_unigram_tokenizer() {
        let metadata = sentencepiece_metadata("gemma4");
        let loaded = from_metadata(&metadata).unwrap().unwrap();
        let encoding = loaded.tokenizer.encode("hi", false).unwrap();
        assert_eq!(encoding.get_ids(), &[5]);
        assert_eq!(
            loaded.tokenizer.decode(encoding.get_ids(), false).unwrap(),
            "hi"
        );
    }

    #[test]
    fn gemma_decoder_preserves_literal_newlines() {
        let metadata = sentencepiece_metadata("gemma4");
        let tokenizer = from_metadata(&metadata).unwrap().unwrap().tokenizer;

        assert_eq!(
            tokenizer.decode(&[5, 6, 4, 7, 4], false).unwrap(),
            "hi\nhi\n\nhi"
        );
        assert_eq!(tokenizer.decode(&[5, 6, 5], false).unwrap(), "hi\n hi");
        assert_eq!(tokenizer.decode(&[5, 6, 1, 5], false).unwrap(), "hi\n  hi");
        let encoding = tokenizer.encode("hi\nhi", false).unwrap();
        assert_eq!(
            tokenizer.decode(encoding.get_ids(), false).unwrap(),
            "hi\nhi"
        );
    }

    fn sentencepiece_metadata(architecture: &str) -> HashMap<String, GgufMetadataValue> {
        HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String(architecture.into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                GgufMetadataValue::String("llama".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "<unk>".into(),
                    "▁".into(),
                    "h".into(),
                    "i".into(),
                    "hi".into(),
                    "▁hi".into(),
                    "\n".into(),
                    "\n\n".into(),
                ])),
            ),
            (
                "tokenizer.ggml.scores".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Float32(vec![
                    0.0, 1.0, 2.0, 2.0, 3.0, 10.0, 1.0, 1.0,
                ])),
            ),
            (
                "tokenizer.ggml.unknown_token_id".into(),
                GgufMetadataValue::Uint32(0),
            ),
        ])
    }
}
