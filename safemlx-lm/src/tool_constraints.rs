//! Private constrained-decoding implementation for native tool plans.

#![allow(dead_code)]

use std::{
    collections::{BTreeSet, HashSet},
    num::NonZeroUsize,
    sync::Arc,
};

use llguidance::{
    toktrie::{SimpleVob, TokEnv, TokenId},
    Matcher, ParserFactory,
};
use safemlx_lm_utils::tokenizer::Tokenizer as ChatTokenizer;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use toktrie_hf_tokenizers::ByteTokenizer;

use crate::{
    chat::{GenerationConstraint, ParallelToolCallPolicy, ToolChoice, ToolRuntimePlan},
    format_dialect::{DialectParameters, FormatDialect},
};

const MAX_SCHEMA_DEPTH: usize = 64;

/// Tokenizer-wide llguidance data. A `LoadedModel` constructs exactly one and
/// every request grammar shares it.
pub(crate) struct ConstraintCompiler {
    factory: Arc<ParserFactory>,
}

#[allow(dead_code)]
pub(crate) struct ConstraintBlueprint {
    matcher: Matcher,
}

#[allow(dead_code)]
pub(crate) struct GrammarState {
    matcher: Matcher,
}

impl ConstraintCompiler {
    pub(crate) fn from_tokenizer(
        tokenizer: &ChatTokenizer,
        eos_token_ids: &[u32],
    ) -> Result<Self, String> {
        let serialized = tokenizer
            .to_string(false)
            .map_err(|error| format!("failed to serialize tokenizer: {error}"))?;
        let mut byte_tokenizer = ByteTokenizer::from_json_bytes(serialized.as_bytes())
            .map_err(|error| format!("failed to analyze tokenizer vocabulary: {error}"))?;
        if !eos_token_ids.is_empty() {
            let vocab_size = byte_tokenizer.tokrx_info().vocab_size;
            if let Some(id) = eos_token_ids.iter().find(|&&id| id >= vocab_size) {
                return Err(format!(
                    "EOS token ID {id} is outside tokenizer vocabulary {vocab_size}"
                ));
            }
            byte_tokenizer.set_eos_tokens(eos_token_ids);
        }
        let token_env = byte_tokenizer
            .into_tok_env(Some(tokenizer.get_vocab_size(true)))
            .map_err(|error| format!("failed to build tokenizer trie: {error}"))?;
        Self::from_tok_env(token_env)
    }

    fn from_tok_env(token_env: TokEnv) -> Result<Self, String> {
        let mut factory = ParserFactory::new_simple(&token_env)
            .map_err(|error| format!("failed to analyze tokenizer trie: {error}"))?;
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
        })
    }

    #[cfg(test)]
    pub(crate) fn synthetic_for_tests() -> Self {
        Self::from_tok_env(llguidance::toktrie::ApproximateTokEnv::single_byte_env())
            .expect("single-byte tokenizer must support llguidance")
    }

    pub(crate) fn compile_tool_plan(
        &self,
        dialect: &'static dyn FormatDialect,
        parameters: DialectParameters,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
    ) -> Result<ToolRuntimePlan, String> {
        let configuration = dialect.constraint_configuration(
            parameters,
            tools,
            tool_choice,
            parallel_tool_calls,
        )?;
        let fingerprint: [u8; 32] = Sha256::digest(
            serde_json::to_vec(&configuration.grammar).expect("grammar configuration serializes"),
        )
        .into();
        let parser = self
            .factory
            .create_parser(configuration.grammar)
            .map_err(|error| format!("failed to compile tool grammar: {error}"))?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(error) = matcher.get_error() {
            return Err(format!("failed to compile tool grammar: {error}"));
        }
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(format!(
                "tool grammar produced unsupported warnings: {}",
                warnings.join("; ")
            ));
        }
        Ok(ToolRuntimePlan::from_constraint(
            GenerationConstraint::new(fingerprint, ConstraintBlueprint { matcher }),
            if tool_choice == ToolChoice::Auto {
                dialect.auto_activation_trigger(parameters)?
            } else {
                None
            },
        ))
    }
}

#[allow(dead_code)]
impl ConstraintBlueprint {
    fn state(&self) -> GrammarState {
        GrammarState {
            matcher: self.matcher.deep_clone(),
        }
    }
}

#[allow(dead_code)]
impl GrammarState {
    pub(crate) fn fork(&self) -> Self {
        Self {
            matcher: self.matcher.deep_clone(),
        }
    }

    pub(crate) fn allowed_tokens(&mut self) -> Result<SimpleVob, String> {
        self.matcher
            .compute_mask_or_eos()
            .map_err(|error| format!("failed to compute grammar token mask: {error}"))
    }

    pub(crate) fn commit(&mut self, token: TokenId) -> Result<(), String> {
        let consumed = self
            .matcher
            .try_consume_tokens(&[token])
            .map_err(|error| format!("failed to commit grammar token: {error}"))?;
        if consumed != 1 {
            return Err(format!("token {token} is not allowed by the tool grammar"));
        }
        Ok(())
    }

    pub(crate) fn is_complete(&mut self) -> Result<bool, String> {
        self.matcher
            .is_accepting()
            .map_err(|error| format!("failed to inspect grammar completion: {error}"))
    }

    pub(crate) fn rollback(&mut self, token_count: usize) -> Result<(), String> {
        self.matcher
            .rollback(token_count)
            .map_err(|error| format!("failed to roll back grammar state: {error}"))
    }
}

impl GenerationConstraint {
    pub(crate) fn new(fingerprint: [u8; 32], inner: ConstraintBlueprint) -> Self {
        Self {
            fingerprint,
            inner: Arc::new(inner),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn grammar_state(&self) -> GrammarState {
        self.inner.state()
    }
}

#[derive(Debug)]
struct ToolDefinition {
    name: String,
    parameters: Value,
}

pub(crate) fn tool_call_bounds(
    tool_choice: ToolChoice,
    parallel_tool_calls: ParallelToolCallPolicy,
    tools: &[Value],
) -> Result<(usize, Option<usize>), String> {
    if tool_choice == ToolChoice::Required && tools.is_empty() {
        return Err("tool_choice is required but no tools were supplied".into());
    }

    let (min_calls, max_calls) = match tool_choice {
        ToolChoice::None => (0, Some(0)),
        ToolChoice::Auto => (
            0,
            match parallel_tool_calls {
                ParallelToolCallPolicy::Disabled => Some(1),
                ParallelToolCallPolicy::Enabled { max_calls } => max_calls.map(NonZeroUsize::get),
            },
        ),
        ToolChoice::Required => (
            1,
            match parallel_tool_calls {
                ParallelToolCallPolicy::Disabled => Some(1),
                ParallelToolCallPolicy::Enabled { max_calls } => max_calls.map(NonZeroUsize::get),
            },
        ),
    };
    if max_calls.is_some_and(|maximum| maximum < min_calls) {
        return Err("parallel tool-call limit cannot satisfy tool_choice".into());
    }
    Ok((min_calls, max_calls))
}

pub(crate) fn tool_call_schema(
    tools: &[Value],
    name_field: &str,
    arguments_field: &str,
) -> Result<Value, String> {
    let tools = parse_tools(tools)?;
    let item_schema = if tools.is_empty() {
        json!({"type": "null"})
    } else {
        let alternatives = tools
            .into_iter()
            .map(|tool| {
                json!({
                    "type": "object",
                    "properties": {
                        name_field: {"type": "string", "enum": [tool.name]},
                        arguments_field: tool.parameters,
                    },
                    "required": [name_field, arguments_field],
                    "additionalProperties": false,
                })
            })
            .collect::<Vec<_>>();
        if alternatives.len() == 1 {
            alternatives.into_iter().next().expect("one alternative")
        } else {
            json!({"oneOf": alternatives})
        }
    };

    Ok(item_schema)
}

fn parse_tools(tools: &[Value]) -> Result<Vec<ToolDefinition>, String> {
    let mut names = HashSet::new();
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let path = format!("tools[{index}]");
            let object = tool
                .as_object()
                .ok_or_else(|| format!("{path} must be an object"))?;
            reject_unknown_keys(object, &["type", "function"], &path)?;
            if object.get("type").and_then(Value::as_str) != Some("function") {
                return Err(format!("{path}.type must be \"function\""));
            }
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("{path}.function must be an object"))?;
            reject_unknown_keys(
                function,
                &["name", "description", "parameters"],
                &format!("{path}.function"),
            )?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| format!("{path}.function.name must be a non-empty string"))?
                .to_owned();
            if !names.insert(name.clone()) {
                return Err(format!("duplicate tool function name {name:?}"));
            }
            if function
                .get("description")
                .is_some_and(|description| !description.is_string())
            {
                return Err(format!("{path}.function.description must be a string"));
            }
            let parameters = function
                .get("parameters")
                .ok_or_else(|| format!("{path}.function.parameters is required"))?;
            let parameters =
                resolve_and_validate_schema(parameters, &format!("{path}.function.parameters"))?;
            if parameters.get("type").and_then(Value::as_str) != Some("object") {
                return Err(format!(
                    "{path}.function.parameters must resolve to an object schema"
                ));
            }
            Ok(ToolDefinition { name, parameters })
        })
        .collect()
}

fn reject_unknown_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    path: &str,
) -> Result<(), String> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("{path} contains unsupported field {key:?}"));
    }
    Ok(())
}

fn resolve_and_validate_schema(schema: &Value, path: &str) -> Result<Value, String> {
    if !schema.is_object() {
        return Err(format!("{path} must be a JSON Schema object"));
    }
    let mut reference_stack = Vec::new();
    resolve_schema(schema, schema, path, &mut reference_stack, 0)
}

fn resolve_schema(
    schema: &Value,
    root: &Value,
    path: &str,
    reference_stack: &mut Vec<String>,
    depth: usize,
) -> Result<Value, String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(format!("{path} exceeds the supported schema nesting depth"));
    }
    let object = schema
        .as_object()
        .ok_or_else(|| format!("{path} must be a JSON Schema object"))?;
    if let Some(reference) = object.get("$ref") {
        if object.len() != 1 {
            return Err(format!("{path} cannot combine $ref with sibling keywords"));
        }
        let reference = reference
            .as_str()
            .ok_or_else(|| format!("{path}.$ref must be a string"))?;
        if reference_stack.iter().any(|active| active == reference) {
            return Err(format!(
                "{path} contains unsupported recursive reference {reference:?}"
            ));
        }
        let target = resolve_local_reference(root, reference, path)?;
        reference_stack.push(reference.to_owned());
        let resolved = resolve_schema(target, root, path, reference_stack, depth + 1);
        reference_stack.pop();
        return resolved;
    }

    validate_schema_keywords(object, path)?;
    let mut resolved = Map::new();
    for (keyword, value) in object {
        match keyword.as_str() {
            "$defs" | "definitions" => {
                validate_definition_map(value, root, path, reference_stack, depth + 1)?;
            }
            "properties" => {
                let properties = value
                    .as_object()
                    .ok_or_else(|| format!("{path}.properties must be an object"))?;
                let mut output = Map::new();
                for (name, property) in properties {
                    output.insert(
                        name.clone(),
                        resolve_schema(
                            property,
                            root,
                            &format!("{path}.properties[{name:?}]"),
                            reference_stack,
                            depth + 1,
                        )?,
                    );
                }
                resolved.insert(keyword.clone(), Value::Object(output));
            }
            "items" => {
                resolved.insert(
                    keyword.clone(),
                    resolve_schema(
                        value,
                        root,
                        &format!("{path}.items"),
                        reference_stack,
                        depth + 1,
                    )?,
                );
            }
            "additionalProperties" if value.is_object() => {
                resolved.insert(
                    keyword.clone(),
                    resolve_schema(
                        value,
                        root,
                        &format!("{path}.additionalProperties"),
                        reference_stack,
                        depth + 1,
                    )?,
                );
            }
            _ => {
                resolved.insert(keyword.clone(), value.clone());
            }
        }
    }
    validate_schema_shape(&resolved, path)?;
    Ok(Value::Object(resolved))
}

fn validate_definition_map(
    definitions: &Value,
    root: &Value,
    path: &str,
    reference_stack: &mut Vec<String>,
    depth: usize,
) -> Result<(), String> {
    let definitions = definitions
        .as_object()
        .ok_or_else(|| format!("{path} definitions must be an object"))?;
    for (name, definition) in definitions {
        resolve_schema(
            definition,
            root,
            &format!("{path} definition {name:?}"),
            reference_stack,
            depth,
        )?;
    }
    Ok(())
}

fn validate_schema_keywords(schema: &Map<String, Value>, path: &str) -> Result<(), String> {
    const SUPPORTED: &[&str] = &[
        "$ref",
        "$defs",
        "definitions",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "enum",
        "description",
        "title",
        "default",
    ];
    if let Some(keyword) = schema
        .keys()
        .find(|keyword| !SUPPORTED.contains(&keyword.as_str()))
    {
        let kind = if matches!(
            keyword.as_str(),
            "allOf" | "anyOf" | "oneOf" | "not" | "if" | "then" | "else"
        ) {
            "composition"
        } else {
            "keyword"
        };
        return Err(format!(
            "{path} contains unsupported schema {kind} {keyword:?}"
        ));
    }
    Ok(())
}

fn validate_schema_shape(schema: &Map<String, Value>, path: &str) -> Result<(), String> {
    let schema_type = schema
        .get("type")
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("{path}.type must be one string, not a union"))
        })
        .transpose()?;
    if let Some(schema_type) = schema_type {
        if !matches!(
            schema_type,
            "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
        ) {
            return Err(format!("{path}.type {schema_type:?} is unsupported"));
        }
    }

    let has_object_keywords = schema.contains_key("properties")
        || schema.contains_key("required")
        || schema.contains_key("additionalProperties");
    let has_array_keywords = schema.contains_key("items")
        || schema.contains_key("minItems")
        || schema.contains_key("maxItems");
    if has_object_keywords && has_array_keywords {
        return Err(format!("{path} mixes object and array keywords"));
    }
    let structural_type = if has_object_keywords {
        Some("object")
    } else if has_array_keywords {
        Some("array")
    } else {
        None
    };
    if let Some(structural_type) = structural_type {
        if schema_type != Some(structural_type) {
            return Err(format!(
                "{path} uses {structural_type} keywords without type {structural_type:?}"
            ));
        }
    }

    if schema_type == Some("object") {
        let properties = match schema.get("properties") {
            Some(properties) => properties
                .as_object()
                .ok_or_else(|| format!("{path}.properties must be an object"))?,
            None => {
                static EMPTY_PROPERTIES: std::sync::LazyLock<Map<String, Value>> =
                    std::sync::LazyLock::new(Map::new);
                &EMPTY_PROPERTIES
            }
        };
        if let Some(required) = schema.get("required") {
            let required = required
                .as_array()
                .ok_or_else(|| format!("{path}.required must be an array"))?;
            let mut seen = BTreeSet::new();
            for name in required {
                let name = name
                    .as_str()
                    .ok_or_else(|| format!("{path}.required entries must be strings"))?;
                if !seen.insert(name) {
                    return Err(format!("{path}.required contains duplicate {name:?}"));
                }
                if !properties.contains_key(name) {
                    return Err(format!("{path}.required names unknown property {name:?}"));
                }
            }
        }
        if let Some(additional) = schema.get("additionalProperties") {
            if !additional.is_boolean() && !additional.is_object() {
                return Err(format!(
                    "{path}.additionalProperties must be a boolean or schema object"
                ));
            }
        }
    }

    if schema_type == Some("array") {
        if !schema.get("items").is_some_and(Value::is_object) {
            return Err(format!("{path}.items must be a schema object"));
        }
        let min = schema_usize(schema, "minItems", path)?;
        let max = schema_usize(schema, "maxItems", path)?;
        if min.zip(max).is_some_and(|(min, max)| min > max) {
            return Err(format!("{path}.minItems exceeds maxItems"));
        }
    }

    if let Some(values) = schema.get("enum") {
        let values = values
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| format!("{path}.enum must be a non-empty array"))?;
        if let Some(schema_type) = schema_type {
            if let Some(value) = values
                .iter()
                .find(|value| !value_matches_type(value, schema_type))
            {
                return Err(format!(
                    "{path}.enum value {value} does not match type {schema_type:?}"
                ));
            }
        }
    }
    for annotation in ["description", "title"] {
        if schema
            .get(annotation)
            .is_some_and(|value| !value.is_string())
        {
            return Err(format!("{path}.{annotation} must be a string"));
        }
    }
    if schema_type.is_none() && !schema.contains_key("enum") {
        return Err(format!(
            "{path} must declare a supported type, enum, or local $ref"
        ));
    }
    Ok(())
}

fn schema_usize(
    schema: &Map<String, Value>,
    keyword: &str,
    path: &str,
) -> Result<Option<usize>, String> {
    schema
        .get(keyword)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("{path}.{keyword} must be a non-negative integer"))
        })
        .transpose()
}

fn value_matches_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn resolve_local_reference<'a>(
    root: &'a Value,
    reference: &str,
    path: &str,
) -> Result<&'a Value, String> {
    if !reference.starts_with('#') {
        return Err(format!(
            "{path} uses unsupported non-local reference {reference:?}"
        ));
    }
    if reference.contains('%') {
        return Err(format!(
            "{path} uses unsupported percent-encoded reference {reference:?}"
        ));
    }
    let pointer = &reference[1..];
    if pointer.is_empty() {
        return Ok(root);
    }
    if !pointer.starts_with('/') {
        return Err(format!("{path} has invalid local reference {reference:?}"));
    }
    for segment in pointer[1..].split('/') {
        let bytes = segment.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'~' {
                if bytes
                    .get(index + 1)
                    .is_none_or(|escaped| !matches!(escaped, b'0' | b'1'))
                {
                    return Err(format!("{path} has invalid local reference {reference:?}"));
                }
                index += 1;
            }
            index += 1;
        }
    }
    root.pointer(pointer)
        .ok_or_else(|| format!("{path} references missing schema {reference:?}"))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use llguidance::toktrie::TokenId;
    use serde_json::json;

    use super::{ConstraintCompiler, ParallelToolCallPolicy, ToolChoice};
    use crate::format_dialect::{
        DeclarativeDialectSpec, DeclarativePayloadShape, DialectParameters, ExactEnvelope,
        GenerationPromptBehavior, ParallelCallLayout, DECLARATIVE_DIALECT,
    };

    const SYNTHETIC_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        output: ExactEnvelope {
            prefix: r#"{"calls":"#,
            suffix: "}",
        },
        call: ExactEnvelope {
            prefix: "",
            suffix: "",
        },
        payload_shape: DeclarativePayloadShape::JsonList,
        name_field: "name",
        arguments_field: "arguments",
        reasoning_channel: None,
        text_channel: None,
        call_separator: ",",
        parallel_layout: ParallelCallLayout::SingleEnvelope,
        auto_activation_trigger: Some(r#"{"calls":"#),
        required_structural_token_ids: &[],
        stop_sequences: &[],
    };

    const SYNTHETIC_PARAMETERS: DialectParameters = DialectParameters::Declarative(&SYNTHETIC_SPEC);

    fn compiler() -> ConstraintCompiler {
        ConstraintCompiler::synthetic_for_tests()
    }

    fn tool(name: &str, parameters: serde_json::Value) -> serde_json::Value {
        json!({
            "type": "function",
            "function": {"name": name, "parameters": parameters}
        })
    }

    fn accepts(plan: &crate::chat::ToolRuntimePlan, value: serde_json::Value) -> bool {
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut state = plan.generation_constraint().grammar_state();
        for byte in bytes {
            if state.commit(byte as TokenId).is_err() {
                return false;
            }
        }
        state.is_complete().unwrap()
    }

    #[test]
    fn restricts_function_names_and_supports_required_optional_and_nested_values() {
        let compiler = compiler();
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[
                    tool(
                        "lookup",
                        json!({
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "limit": {"type": "integer"},
                                        "exact": {"type": "boolean"}
                                    },
                                    "required": ["limit"],
                                    "additionalProperties": false
                                }
                            },
                            "required": ["query"],
                            "additionalProperties": false
                        }),
                    ),
                    tool(
                        "ping",
                        json!({
                            "type": "object",
                            "properties": {},
                            "additionalProperties": false
                        }),
                    ),
                ],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
            )
            .unwrap();

        assert!(accepts(
            &plan,
            json!({"calls": [{
                "name": "lookup",
                "arguments": {
                    "query": "snowman ☃ and \"quotes\"",
                    "options": {"limit": 3, "exact": true}
                }
            }]})
        ));
        assert!(accepts(
            &plan,
            json!({"calls": [{"name": "lookup", "arguments": {"query": "optional omitted"}}]})
        ));
        assert!(!accepts(
            &plan,
            json!({"calls": [{"name": "unknown", "arguments": {}}]})
        ));
    }

    #[test]
    fn resolves_local_references_and_supports_arrays_enums_and_scalars() {
        let compiler = compiler();
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[tool(
                    "batch",
                    json!({
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/item~1type~0v1"},
                                "minItems": 1,
                                "maxItems": 2
                            },
                            "mode": {"type": "string", "enum": ["fast", "安全"]},
                            "ratio": {"type": "number"},
                            "enabled": {"type": "boolean"},
                            "nothing": {"type": "null"}
                        },
                        "required": ["items", "mode", "ratio", "enabled", "nothing"],
                        "additionalProperties": false,
                        "$defs": {
                            "item/type~v1": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                                "additionalProperties": false
                            }
                        }
                    }),
                )],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
            )
            .unwrap();

        assert!(accepts(
            &plan,
            json!({"calls": [{
                "name": "batch",
                "arguments": {
                    "items": [{"value": "α"}, {"value": "β"}],
                    "mode": "安全",
                    "ratio": 1.5,
                    "enabled": false,
                    "nothing": null
                }
            }]})
        ));
        assert!(!accepts(
            &plan,
            json!({"calls": [{"name": "batch", "arguments": {
                "items": [], "mode": "slow", "ratio": 1, "enabled": true, "nothing": null
            }}]})
        ));
    }

    #[test]
    fn rejects_invalid_references_schemas_keywords_and_compositions() {
        let compiler = compiler();
        let invalid = [
            tool(
                "missing",
                json!({"type": "object", "properties": {"x": {"$ref": "#/$defs/nope"}}}),
            ),
            tool(
                "external",
                json!({"type": "object", "properties": {"x": {"$ref": "https://example.test/schema"}}}),
            ),
            tool("malformed", json!({"type": "object", "required": "x"})),
            tool(
                "unsupported",
                json!({"type": "object", "patternProperties": {".*": {"type": "string"}}}),
            ),
            tool(
                "composition",
                json!({"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "number"}]}}}),
            ),
        ];
        for tool in invalid {
            let error = compiler
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    SYNTHETIC_PARAMETERS,
                    &[tool],
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                )
                .unwrap_err();
            assert!(
                error.contains("reference")
                    || error.contains("required")
                    || error.contains("unsupported"),
                "{error}"
            );
        }
    }

    #[test]
    fn enforces_single_and_parallel_call_limits() {
        let compiler = compiler();
        let tools = [tool(
            "ping",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        )];
        let single = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
            )
            .unwrap();
        let parallel = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Enabled {
                    max_calls: NonZeroUsize::new(2),
                },
            )
            .unwrap();
        let two = json!({"calls": [
            {"name": "ping", "arguments": {}},
            {"name": "ping", "arguments": {}}
        ]});
        let three = json!({"calls": [
            {"name": "ping", "arguments": {}},
            {"name": "ping", "arguments": {}},
            {"name": "ping", "arguments": {}}
        ]});
        assert!(!accepts(&single, two.clone()));
        assert!(accepts(&parallel, two));
        assert!(!accepts(&parallel, three));
    }

    #[test]
    fn grammar_state_forks_commits_completes_and_rolls_back() {
        let compiler = compiler();
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[tool(
                    "ping",
                    json!({"type": "object", "properties": {}, "additionalProperties": false}),
                )],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
            )
            .unwrap();
        let bytes =
            serde_json::to_vec(&json!({"calls": [{"name": "ping", "arguments": {}}]})).unwrap();
        let split = bytes.len() / 2;
        let mut state = plan.generation_constraint().grammar_state();
        for byte in &bytes[..split] {
            state.commit(*byte as TokenId).unwrap();
        }
        let mut fork = state.fork();
        assert!(!state.allowed_tokens().unwrap().is_empty());
        for byte in &bytes[split..] {
            state.commit(*byte as TokenId).unwrap();
        }
        assert!(state.is_complete().unwrap());
        fork.commit(bytes[split] as TokenId).unwrap();
        fork.rollback(1).unwrap();
        assert!(!fork.is_complete().unwrap());
        for byte in &bytes[split..] {
            fork.commit(*byte as TokenId).unwrap();
        }
        assert!(fork.is_complete().unwrap());
    }
}
