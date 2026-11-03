//! Crate-wide error type.

/// Every fallible operation in `scaleup_core` returns this error.
#[derive(Debug, thiserror::Error)]
pub enum ScaleUpError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to decode image: {0}")]
    Decode(String),
    #[error("failed to encode image: {0}")]
    Encode(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("cancelled")]
    Cancelled,
    #[error("invalid option: {0}")]
    InvalidOption(String),
    #[error("download failed: {0}")]
    Download(String),
}

pub type Result<T> = std::result::Result<T, ScaleUpError>;
