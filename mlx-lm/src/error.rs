use mlx_rs::error::Exception;

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
pub enum Error {
    #[error("unsupported model type: {0}")]
    UnsupportedModelType(String),

    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("strict weight-load validation failed: {missing_count} missing parameters, {unused_count} unused weights\nmissing:\n{missing}\nunused:\n{unused}", missing_count = .missing.len(), unused_count = .unused.len(), missing = format_keys(.missing), unused = format_keys(.unused))]
    StrictLoadValidation {
        missing: Vec<String>,
        unused: Vec<String>,
    },

    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError),

    #[error(transparent)]
    Template(#[from] safemlx_lm_utils::error::Error),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
