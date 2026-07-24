//! Public chat preparation and native tool-runtime contracts.
//!
//! Chat templates remain checkpoint-owned Jinja programs. Format profiles are
//! selected only from registered signatures of the selected template body;
//! model architecture metadata is deliberately not a fallback.

use std::num::NonZeroUsize;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationConstraint {
    _private: (),
}

/// An opaque, format-profile-specific plan for native tool generation.
///
/// Plans can be inspected only by this crate's generation runtime. Callers
/// should treat a value as a capability token carried by
/// [`NativeToolSupport::Supported`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRuntimePlan {
    generation_constraint: GenerationConstraint,
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
    pub(crate) native_tool_support: NativeToolSupport,
    pub(crate) preserved_structural_token_ids: Vec<u32>,
    pub(crate) stop_sequences: Vec<String>,
}

#[derive(Debug)]
struct FormatProfile {
    identity: &'static str,
    template_signatures: &'static [[u8; 32]],
    preserved_structural_token_ids: &'static [u32],
    stop_sequences: &'static [&'static str],
    tool_runtime_plan: Option<&'static ToolRuntimePlan>,
}

// Intentionally empty: production dialects and constraint engines are added
// only alongside audited, exact template signatures.
const FORMAT_PROFILES: &[FormatProfile] = &[];

fn template_signature(template: &str) -> [u8; 32] {
    Sha256::digest(template.as_bytes()).into()
}

fn matching_profiles<'a>(template: &str, profiles: &'a [FormatProfile]) -> Vec<&'a FormatProfile> {
    let signature = template_signature(template);
    profiles
        .iter()
        .filter(|profile| profile.template_signatures.contains(&signature))
        .collect()
}

fn prepare_format_profile_with_registry(
    template: &str,
    profiles: &[FormatProfile],
) -> PreparedFormatProfile {
    match matching_profiles(template, profiles).as_slice() {
        [] => PreparedFormatProfile {
            identity: None,
            native_tool_support: NativeToolSupport::Unsupported {
                reason: "no registered format profile matches the selected chat template".into(),
            },
            preserved_structural_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        },
        [profile] => PreparedFormatProfile {
            identity: Some(profile.identity.to_owned()),
            native_tool_support: match profile.tool_runtime_plan {
                Some(plan) => NativeToolSupport::Supported(plan.clone()),
                None => NativeToolSupport::Unsupported {
                    reason: format!(
                        "format profile {:?} does not provide a native tool runtime plan",
                        profile.identity
                    ),
                },
            },
            preserved_structural_token_ids: profile.preserved_structural_token_ids.to_vec(),
            stop_sequences: profile
                .stop_sequences
                .iter()
                .map(|sequence| (*sequence).to_owned())
                .collect(),
        },
        _ => PreparedFormatProfile {
            identity: None,
            native_tool_support: NativeToolSupport::Unsupported {
                reason: "multiple registered format profiles match the selected chat template"
                    .into(),
            },
            preserved_structural_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        },
    }
}

pub(crate) fn prepare_format_profile(template: &str) -> PreparedFormatProfile {
    prepare_format_profile_with_registry(template, FORMAT_PROFILES)
}

#[cfg(test)]
mod tests {
    use super::{
        prepare_format_profile, prepare_format_profile_with_registry, template_signature,
        FormatProfile, NativeToolSupport,
    };

    #[test]
    fn registry_does_not_guess_unknown_templates() {
        let prepared = prepare_format_profile("unknown template");

        assert_eq!(prepared.identity, None);
        assert!(prepared.preserved_structural_token_ids.is_empty());
        assert!(prepared.stop_sequences.is_empty());
        assert!(matches!(
            prepared.native_tool_support,
            NativeToolSupport::Unsupported { ref reason }
                if reason.contains("no registered format profile")
        ));
    }

    #[test]
    fn registry_treats_duplicate_signatures_as_ambiguous() {
        let signature = template_signature("same template");
        let signatures: &'static [[u8; 32]] = Box::leak(vec![signature].into_boxed_slice());
        let profiles = [
            FormatProfile {
                identity: "first",
                template_signatures: signatures,
                preserved_structural_token_ids: &[],
                stop_sequences: &[],
                tool_runtime_plan: None,
            },
            FormatProfile {
                identity: "second",
                template_signatures: signatures,
                preserved_structural_token_ids: &[],
                stop_sequences: &[],
                tool_runtime_plan: None,
            },
        ];

        let prepared = prepare_format_profile_with_registry("same template", &profiles);
        assert_eq!(prepared.identity, None);
        assert!(matches!(
            prepared.native_tool_support,
            NativeToolSupport::Unsupported { ref reason }
                if reason.contains("multiple registered format profiles")
        ));
    }
}
