use safemlx::error::Exception;

fn format_keys(keys: &[String]) -> String {
    const LIMIT: usize = 50;
    if keys.is_empty() {
        return "  <none>".to_string();
    }
    let mut lines = keys
        .iter()
        .take(LIMIT)
        .map(|key| format!("  {key}"))
        .collect::<Vec<_>>();
    if keys.len() > LIMIT {
        lines.push(format!("  ... and {} more", keys.len() - LIMIT));
    }
    lines.join("\n")
}

#[derive(Debug, thiserror::Error)]
/// Error type used by `safemlx-lm` loaders and tokenizer helpers.
pub enum Error {
    /// Invalid or failed Llama-compatible layerwise host offload.
    #[error(transparent)]
    LlamaHostOffload(#[from] crate::llama_host_offload::LlamaHostOffloadError),

    /// Invalid module-to-checkpoint or resident-lease binding.
    #[error(transparent)]
    ModuleBinding(#[from] crate::module_binding::ModuleBindingError),

    /// Persistent checkpoint catalog, mapping, or materialization failure.
    #[error(transparent)]
    WeightStore(#[from] crate::weight_store::WeightStoreError),

    /// Invalid architecture-independent offload planning request.
    #[error(transparent)]
    Offload(#[from] crate::offload::OffloadError),

    /// Invalid or failed weight residency operation.
    #[error(transparent)]
    Residency(#[from] crate::residency::ResidencyError),

    /// Invalid runtime parallel topology, tensor placement, or partition request.
    #[error("parallel placement error: {0}")]
    Parallel(String),

    /// Invalid or unsupported checkpoint quantization request.
    #[error("checkpoint quantization error: {0}")]
    Quantization(String),

    /// The `model_type` in `config.json` is not supported by this crate.
    #[error("unsupported model type: {0}")]
    UnsupportedModelType(String),

    /// The model family is recognized but this specific architecture is unsupported.
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Media processor configuration or input error.
    #[error("media processor error: {0}")]
    Processor(String),

    /// Embedded GGUF tokenizer metadata is invalid or cannot be reconstructed.
    #[error("GGUF tokenizer error: {0}")]
    GgufTokenizer(String),

    /// Strict weight loading found missing parameters or unused checkpoint tensors.
    #[error("strict weight-load validation failed: {missing_count} missing parameters, {unused_count} unused weights\nmissing:\n{missing}\nunused:\n{unused}", missing_count = .missing.len(), unused_count = .unused.len(), missing = format_keys(.missing), unused = format_keys(.unused))]
    StrictLoadValidation {
        /// Model parameters that were not populated from the checkpoint.
        missing: Vec<String>,
        /// Checkpoint tensors that were not consumed by the model.
        unused: Vec<String>,
    },

    /// Error reported by the underlying MLX bindings.
    #[error(transparent)]
    Exception(#[from] Exception),

    /// Filesystem I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON configuration deserialization error.
    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    /// Safetensors loading error from `safemlx`.
    #[error(transparent)]
    LoadWeights(#[from] safemlx::error::IoError),

    /// Chat-template or tokenizer utility error.
    #[error(transparent)]
    Template(#[from] safemlx_lm_utils::error::Error),

    /// Boxed error used for third-party loader failures.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
