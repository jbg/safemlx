//! Public chat preparation and native tool-runtime contracts.
//!
//! Chat templates remain checkpoint-owned Jinja programs. Format profiles are
//! selected only from registered signatures of the selected template body;
//! model architecture metadata is deliberately not a fallback.

use std::{fmt, num::NonZeroUsize, sync::Arc};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::format_dialect::{
    DeclarativeDialectSpec, DeclarativePayloadShape, ExactEnvelope, ParallelCallLayout,
    DECLARATIVE_DIALECT,
};
use crate::{
    format_dialect::{
        DialectParameters, FormatDialect, FormatRegistryEntry, GenerationPromptBehavior,
    },
    tool_constraints::ConstraintBlueprint,
};

pub use safemlx_lm_utils::tokenizer::ChatTemplateIdentity;

/// Controls whether the model may emit a native tool call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// Native tool calls are forbidden.
    None,
    /// The model may answer normally or call a tool.
    #[default]
    Auto,
    /// The model must call a tool.
    Required,
}

/// Controls whether one assistant turn may contain parallel native tool calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParallelToolCallPolicy {
    /// At most one tool call may be emitted in an assistant turn.
    #[default]
    Disabled,
    /// Parallel calls are allowed, optionally up to a caller-supplied limit.
    Enabled {
        /// Maximum calls in one assistant turn, or `None` for no caller-supplied limit.
        max_calls: Option<NonZeroUsize>,
    },
}

/// Inputs used to render and prepare one checkpoint-native chat prompt.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatTemplateRequest {
    /// JSON-valued chat messages in checkpoint template order.
    pub messages: Vec<Value>,
    /// JSON Schema tool definitions made available to the chat template.
    pub tools: Vec<Value>,
    /// Whether native tool calls are forbidden, optional, or required.
    pub tool_choice: ToolChoice,
    /// Parallel native tool-call policy and optional per-turn limit.
    pub parallel_tool_calls: ParallelToolCallPolicy,
    /// Explicit thinking/reasoning toggle, or `None` to preserve the template default.
    pub enable_thinking: Option<bool>,
    /// Whether the returned prompt includes the template's generation prompt.
    pub add_generation_prompt: bool,
    /// Additional variables exposed to the checkpoint chat template.
    ///
    /// `enable_thinking`, when explicitly set above, overrides a same-named
    /// entry. Existing renderer precedence for all other keys is preserved.
    pub extra_template_kwargs: Map<String, Value>,
}

/// An opaque generation constraint owned by a native tool runtime plan.
///
/// The representation is intentionally private so future constraint engines
/// can evolve without exposing a dialect-specific implementation as public API.
#[derive(Clone)]
pub struct GenerationConstraint {
    pub(crate) fingerprint: [u8; 32],
    #[allow(dead_code)]
    pub(crate) inner: Arc<ConstraintBlueprint>,
}

impl fmt::Debug for GenerationConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationConstraint")
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

impl PartialEq for GenerationConstraint {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
    }
}

impl Eq for GenerationConstraint {}

/// An opaque, format-profile-specific plan for native tool generation.
///
/// Plans can be inspected only by this crate's generation runtime. Callers
/// should treat a value as a capability token carried by
/// [`NativeToolSupport::Supported`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRuntimePlan {
    generation_constraint: GenerationConstraint,
    auto_activation_trigger: Option<String>,
}

impl ToolRuntimePlan {
    pub(crate) fn from_constraint(
        generation_constraint: GenerationConstraint,
        auto_activation_trigger: Option<&str>,
    ) -> Self {
        Self {
            generation_constraint,
            auto_activation_trigger: auto_activation_trigger.map(str::to_owned),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn generation_constraint(&self) -> &GenerationConstraint {
        &self.generation_constraint
    }

    #[allow(dead_code)]
    pub(crate) fn auto_activation_trigger(&self) -> Option<&str> {
        self.auto_activation_trigger.as_deref()
    }
}

/// Whether the selected checkpoint template has registered native tool support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeToolSupport {
    /// The selected format profile produced a native tool runtime plan.
    Supported(ToolRuntimePlan),
    /// No safe native tool runtime plan could be selected.
    Unsupported {
        /// Human-readable explanation suitable for diagnostics.
        reason: String,
    },
}

/// A rendered chat prompt together with generation and parsing metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChat {
    /// The rendered prompt, honoring the request's generation-prompt toggle.
    pub rendered_prompt: String,
    /// The suffix contributed when `add_generation_prompt` is enabled.
    ///
    /// This is empty when the template adds no suffix or when its two render
    /// modes cannot be represented as a simple appended contribution.
    pub generation_prompt: String,
    /// Stable identity of the selected checkpoint chat template.
    pub template_identity: ChatTemplateIdentity,
    /// Registered format-profile identity, when one exact profile matched.
    pub format_profile_identity: Option<String>,
    /// Native tool capability for the selected template and profile.
    pub native_tool_support: NativeToolSupport,
    /// Checkpoint EOS token IDs used to stop generation.
    pub eos_token_ids: Vec<u32>,
    /// Profile-owned structural token IDs that decoding must preserve.
    pub preserved_structural_token_ids: Vec<u32>,
    /// Profile-owned text sequences that stop generation.
    pub profile_stop_sequences: Vec<String>,
}

impl PreparedChat {
    /// Returns the rendered prompt.
    pub fn rendered_prompt(&self) -> &str {
        &self.rendered_prompt
    }

    /// Returns the separately computed generation-prompt contribution.
    pub fn generation_prompt(&self) -> &str {
        &self.generation_prompt
    }

    /// Returns the selected checkpoint template identity.
    pub fn template_identity(&self) -> &ChatTemplateIdentity {
        &self.template_identity
    }

    /// Returns the registered format-profile identity, if one matched.
    pub fn format_profile_identity(&self) -> Option<&str> {
        self.format_profile_identity.as_deref()
    }

    /// Returns native tool capability for the selected template.
    pub fn native_tool_support(&self) -> &NativeToolSupport {
        &self.native_tool_support
    }

    /// Returns checkpoint EOS token IDs.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Returns structural token IDs that must survive decoding.
    pub fn preserved_structural_token_ids(&self) -> &[u32] {
        &self.preserved_structural_token_ids
    }

    /// Returns format-profile stop sequences.
    pub fn profile_stop_sequences(&self) -> &[String] {
        &self.profile_stop_sequences
    }
}

#[derive(Debug)]
pub(crate) struct PreparedFormatProfile {
    pub(crate) identity: Option<String>,
    pub(crate) dialect: Option<&'static dyn FormatDialect>,
    pub(crate) dialect_parameters: Option<DialectParameters>,
    pub(crate) generation_prompt_behavior: GenerationPromptBehavior,
    pub(crate) native_tool_unavailable_reason: Option<String>,
    pub(crate) preserved_structural_token_ids: Vec<u32>,
    pub(crate) stop_sequences: Vec<String>,
}

/// Test-only protocol surface used to exercise constrained tool generation
/// without claiming compatibility with a production checkpoint dialect.
#[allow(dead_code)]
pub(crate) const SYNTHETIC_TOOL_TEMPLATE: &str = concat!(
    "{% if fail_render %}{{ raise_exception('rendered before constraint compilation') }}",
    "{% endif %}safemlx synthetic tool template",
);

#[cfg(test)]
const SYNTHETIC_TOOL_TEMPLATE_SIGNATURE: [u8; 32] = [
    0x5e, 0xc6, 0xe8, 0xcc, 0x55, 0x35, 0x8f, 0x00, 0x81, 0xdf, 0x23, 0xf7, 0x16, 0x52, 0x95, 0xc0,
    0x2a, 0x4b, 0xf7, 0x9c, 0x15, 0x33, 0xd6, 0x8d, 0x04, 0x77, 0x90, 0x30, 0x3d, 0xd8, 0x59, 0xf4,
];

#[cfg(test)]
const SYNTHETIC_DECLARATIVE_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
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

#[cfg(test)]
const FORMAT_REGISTRY: &[FormatRegistryEntry] = &[FormatRegistryEntry {
    identity: "safemlx.synthetic-tools.v1",
    template_signature: SYNTHETIC_TOOL_TEMPLATE_SIGNATURE,
    dialect: &DECLARATIVE_DIALECT,
    parameters: DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC),
}];

#[cfg(not(test))]
const FORMAT_REGISTRY: &[FormatRegistryEntry] = &[];

pub(crate) fn template_signature(template: &str) -> [u8; 32] {
    Sha256::digest(template.as_bytes()).into()
}

fn matching_registry_entries<'a>(
    template: &str,
    registry: &'a [FormatRegistryEntry],
) -> Vec<&'a FormatRegistryEntry> {
    let signature = template_signature(template);
    registry
        .iter()
        .filter(|entry| entry.template_signature == signature)
        .collect()
}

pub(crate) fn prepare_format_profile_with_registry(
    template: &str,
    registry: &[FormatRegistryEntry],
) -> PreparedFormatProfile {
    match matching_registry_entries(template, registry).as_slice() {
        [] => PreparedFormatProfile {
            identity: None,
            dialect: None,
            dialect_parameters: None,
            generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
            native_tool_unavailable_reason: Some(
                "no registered format profile matches the selected chat template".into(),
            ),
            preserved_structural_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        },
        [entry] => {
            let generation_prompt_behavior =
                entry.dialect.generation_prompt_behavior(entry.parameters);
            let preserved = entry
                .dialect
                .preserved_structural_token_ids(entry.parameters);
            let stops = entry.dialect.stop_sequences(entry.parameters);
            match (generation_prompt_behavior, preserved, stops) {
                (Ok(generation_prompt_behavior), Ok(preserved), Ok(stops)) => {
                    PreparedFormatProfile {
                        identity: Some(entry.identity.to_owned()),
                        dialect: Some(entry.dialect),
                        dialect_parameters: Some(entry.parameters),
                        generation_prompt_behavior,
                        native_tool_unavailable_reason: None,
                        preserved_structural_token_ids: preserved.to_vec(),
                        stop_sequences: stops
                            .iter()
                            .map(|sequence| (*sequence).to_owned())
                            .collect(),
                    }
                }
                (generation, preserved, stops) => {
                    let reason = generation
                        .err()
                        .or_else(|| preserved.err())
                        .or_else(|| stops.err())
                        .expect("one dialect property failed");
                    PreparedFormatProfile {
                        identity: Some(entry.identity.to_owned()),
                        dialect: None,
                        dialect_parameters: None,
                        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
                        native_tool_unavailable_reason: Some(format!(
                            "format profile {:?} is invalid: {reason}",
                            entry.identity
                        )),
                        preserved_structural_token_ids: Vec::new(),
                        stop_sequences: Vec::new(),
                    }
                }
            }
        }
        _ => PreparedFormatProfile {
            identity: None,
            dialect: None,
            dialect_parameters: None,
            generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
            native_tool_unavailable_reason: Some(
                "multiple registered format profiles match the selected chat template".into(),
            ),
            preserved_structural_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        },
    }
}

pub(crate) fn prepare_format_profile(template: &str) -> PreparedFormatProfile {
    prepare_format_profile_with_registry(template, FORMAT_REGISTRY)
}

#[cfg(test)]
mod tests {
    use super::{
        prepare_format_profile, prepare_format_profile_with_registry, template_signature,
        DialectParameters, FormatRegistryEntry, DECLARATIVE_DIALECT, SYNTHETIC_DECLARATIVE_SPEC,
        SYNTHETIC_TOOL_TEMPLATE, SYNTHETIC_TOOL_TEMPLATE_SIGNATURE,
    };

    #[test]
    fn registry_does_not_guess_unknown_templates() {
        let prepared = prepare_format_profile("unknown template");

        assert_eq!(prepared.identity, None);
        assert!(prepared.dialect.is_none());
        assert!(prepared
            .native_tool_unavailable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("no registered format profile")));
        assert!(prepared.preserved_structural_token_ids.is_empty());
        assert!(prepared.stop_sequences.is_empty());
    }

    #[test]
    fn registry_treats_duplicate_signatures_as_ambiguous() {
        let signature = template_signature("same template");
        let registry = [
            FormatRegistryEntry {
                identity: "first",
                template_signature: signature,
                dialect: &DECLARATIVE_DIALECT,
                parameters: DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC),
            },
            FormatRegistryEntry {
                identity: "second",
                template_signature: signature,
                dialect: &DECLARATIVE_DIALECT,
                parameters: DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC),
            },
        ];

        let prepared = prepare_format_profile_with_registry("same template", &registry);
        assert_eq!(prepared.identity, None);
        assert!(prepared.dialect.is_none());
        assert!(prepared
            .native_tool_unavailable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("multiple registered format profiles")));
    }

    #[test]
    fn synthetic_profile_uses_an_exact_auditable_signature() {
        assert_eq!(
            template_signature(SYNTHETIC_TOOL_TEMPLATE),
            SYNTHETIC_TOOL_TEMPLATE_SIGNATURE
        );
        let prepared = prepare_format_profile(SYNTHETIC_TOOL_TEMPLATE);
        assert_eq!(
            prepared.identity.as_deref(),
            Some("safemlx.synthetic-tools.v1")
        );
        assert!(prepared.dialect.is_some());
        assert!(prepared.dialect_parameters.is_some());
        assert_eq!(prepared.native_tool_unavailable_reason, None);
    }
}
