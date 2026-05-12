use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported model type: {0}")]
    UnsupportedModelType(String),

    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError),

    #[error(transparent)]
    Template(#[from] mlx_lm_utils::error::Error),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
