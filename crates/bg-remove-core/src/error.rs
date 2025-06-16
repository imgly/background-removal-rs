//! Error types for background removal operations

use thiserror::Error;

/// Result type alias for background removal operations
pub type Result<T> = std::result::Result<T, BgRemovalError>;

/// Comprehensive error types for background removal operations
#[derive(Error, Debug)]
pub enum BgRemovalError {
    /// Input/output errors (file not found, permission denied, etc.)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Image format or processing errors
    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    /// Backend inference errors
    #[error("Inference error: {0}")]
    Inference(String),

    /// Invalid configuration or parameters
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Unsupported file format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Model loading or initialization errors
    #[error("Model error: {0}")]
    Model(String),

    /// Memory allocation or processing errors
    #[error("Processing error: {0}")]
    Processing(String),

    /// Generic error for unexpected conditions
    #[error("Internal error: {0}")]
    Internal(String),
}

impl BgRemovalError {
    /// Create a new invalid configuration error
    pub fn invalid_config<S: Into<String>>(msg: S) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a new unsupported format error
    pub fn unsupported_format<S: Into<String>>(format: S) -> Self {
        Self::UnsupportedFormat(format.into())
    }

    /// Create a new model error
    pub fn model<S: Into<String>>(msg: S) -> Self {
        Self::Model(msg.into())
    }

    /// Create a new processing error
    pub fn processing<S: Into<String>>(msg: S) -> Self {
        Self::Processing(msg.into())
    }

    /// Create a new inference error
    pub fn inference<S: Into<String>>(msg: S) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a new internal error
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        Self::Internal(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = BgRemovalError::invalid_config("test config error");
        assert!(matches!(err, BgRemovalError::InvalidConfig(_)));

        let err = BgRemovalError::unsupported_format("TIFF");
        assert!(matches!(err, BgRemovalError::UnsupportedFormat(_)));
    }

    #[test]
    fn test_error_display() {
        let err = BgRemovalError::invalid_config("Invalid model path");
        assert_eq!(err.to_string(), "Invalid configuration: Invalid model path");
    }
}
