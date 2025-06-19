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

    // Enhanced contextual error creators

    /// Create file I/O error with operation context
    pub fn file_io_error<P: AsRef<std::path::Path>>(
        operation: &str,
        path: P,
        error: std::io::Error,
    ) -> Self {
        let path_display = path.as_ref().display();
        Self::Io(std::io::Error::new(
            error.kind(),
            format!("Failed to {} '{}': {}", operation, path_display, error),
        ))
    }

    /// Create image loading error with format context
    pub fn image_load_error<P: AsRef<std::path::Path>>(
        path: P,
        error: image::ImageError,
    ) -> Self {
        let path_display = path.as_ref().display();
        let extension = path.as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        
        Self::Image(image::ImageError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Failed to load image '{}' (format: {}): {}. Supported formats: PNG, JPEG, WebP, TIFF, BMP",
                path_display, extension, error
            ),
        )))
    }

    /// Create model error with troubleshooting context
    pub fn model_error_with_context<P: AsRef<std::path::Path>>(
        operation: &str,
        model_path: P,
        error: &str,
        suggestions: &[&str],
    ) -> Self {
        let path_display = model_path.as_ref().display();
        let suggestion_text = if suggestions.is_empty() {
            String::new()
        } else {
            format!(" Suggestions: {}", suggestions.join(", "))
        };
        
        Self::Model(format!(
            "Failed to {} model '{}': {}.{}",
            operation, path_display, error, suggestion_text
        ))
    }

    /// Create configuration error with valid ranges
    pub fn config_value_error<T: std::fmt::Display>(
        parameter: &str,
        value: T,
        valid_range: &str,
        recommended: Option<T>,
    ) -> Self {
        let recommendation = match recommended {
            Some(rec) => format!(" Recommended: {}", rec),
            None => String::new(),
        };
        
        Self::InvalidConfig(format!(
            "Invalid {}: {} (valid range: {}).{}",
            parameter, value, valid_range, recommendation
        ))
    }

    /// Create inference error with provider context
    pub fn inference_error_with_provider(
        provider: &str,
        operation: &str,
        error: &str,
        fallback_suggestions: &[&str],
    ) -> Self {
        let suggestions = if fallback_suggestions.is_empty() {
            String::new()
        } else {
            format!(" Try: {}", fallback_suggestions.join(" or "))
        };
        
        Self::Inference(format!(
            "{} failed using '{}' provider: {}.{}",
            operation, provider, error, suggestions
        ))
    }

    /// Create processing error with stage context
    pub fn processing_stage_error(
        stage: &str,
        details: &str,
        input_info: Option<&str>,
    ) -> Self {
        let input_context = match input_info {
            Some(info) => format!(" (input: {})", info),
            None => String::new(),
        };
        
        Self::Processing(format!(
            "Processing failed at stage '{}'{}: {}",
            stage, input_context, details
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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

    #[test]
    fn test_enhanced_error_context() {
        // Test file I/O error with context
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err = BgRemovalError::file_io_error("read config file", Path::new("/etc/config.json"), io_error);
        let error_string = err.to_string();
        assert!(error_string.contains("read config file"));
        assert!(error_string.contains("/etc/config.json"));

        // Test model error with suggestions
        let err = BgRemovalError::model_error_with_context(
            "initialize",
            Path::new("/models/invalid.onnx"),
            "file not found",
            &["check file path", "verify permissions"]
        );
        let error_string = err.to_string();
        assert!(error_string.contains("initialize"));
        assert!(error_string.contains("/models/invalid.onnx"));
        assert!(error_string.contains("Suggestions"));

        // Test config value error with recommendations
        let err = BgRemovalError::config_value_error("quality", 150, "0-100", Some(85));
        let error_string = err.to_string();
        assert!(error_string.contains("quality"));
        assert!(error_string.contains("150"));
        assert!(error_string.contains("0-100"));
        assert!(error_string.contains("Recommended: 85"));

        // Test inference error with provider context
        let err = BgRemovalError::inference_error_with_provider(
            "CUDA",
            "Model inference",
            "out of memory",
            &["try CPU provider", "reduce batch size"]
        );
        let error_string = err.to_string();
        assert!(error_string.contains("CUDA"));
        assert!(error_string.contains("Model inference"));
        assert!(error_string.contains("Try: try CPU provider or reduce batch size"));

        // Test processing stage error
        let err = BgRemovalError::processing_stage_error(
            "preprocessing",
            "invalid tensor shape",
            Some("1920x1080 RGB")
        );
        let error_string = err.to_string();
        assert!(error_string.contains("preprocessing"));
        assert!(error_string.contains("1920x1080 RGB"));
    }
}
