//! Model validation utilities
//!
//! Provides centralized validation for model specifications, configurations,
//! and related parameters.

use crate::{
    error::{BgRemovalError, Result},
    models::{ModelSource, ModelSpec},
};
use std::path::Path;

/// Validator for model-related configurations and specifications
pub struct ModelValidator;

impl ModelValidator {
    /// Validate a model specification
    ///
    /// Checks if embedded model name is valid or external path exists
    pub fn validate_model_spec(model_spec: &ModelSpec) -> Result<()> {
        match &model_spec.source {
            ModelSource::External(path) => Self::validate_external_model_path(path),
            ModelSource::Embedded(name) => Self::validate_embedded_model_name(name),
        }?;

        // Validate variant if specified
        if let Some(variant) = &model_spec.variant {
            Self::validate_variant_name(variant)?;
        }

        Ok(())
    }

    /// Validate an external model path
    fn validate_external_model_path(path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(BgRemovalError::invalid_config(&format!(
                "External model path does not exist: {}",
                path.display()
            )));
        }
        if !path.is_dir() {
            return Err(BgRemovalError::invalid_config(&format!(
                "External model path must be a directory: {}",
                path.display()
            )));
        }
        Ok(())
    }

    /// Validate an embedded model name
    fn validate_embedded_model_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(BgRemovalError::invalid_config(
                "Embedded model name cannot be empty",
            ));
        }

        // Check for reasonable model name format
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(BgRemovalError::invalid_config(&format!(
                "Invalid characters in embedded model name: {}",
                name
            )));
        }

        Ok(())
    }

    /// Validate a model variant name
    pub fn validate_variant_name(variant: &str) -> Result<()> {
        if variant.is_empty() {
            return Err(BgRemovalError::invalid_config(
                "Model variant cannot be empty",
            ));
        }

        // Check for valid variant format
        if !variant
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(BgRemovalError::invalid_config(&format!(
                "Invalid characters in model variant: {}",
                variant
            )));
        }

        Ok(())
    }

    /// Validate model configuration JSON structure
    pub fn validate_model_config(config: &serde_json::Value) -> Result<()> {
        // Check required top-level fields
        if !config.is_object() {
            return Err(BgRemovalError::invalid_config(
                "Model configuration must be a JSON object",
            ));
        }

        let obj = config.as_object().unwrap();

        // Required fields
        const REQUIRED_FIELDS: &[&str] = &["name", "description", "variants", "preprocessing"];

        for field in REQUIRED_FIELDS {
            if !obj.contains_key(*field) {
                return Err(BgRemovalError::invalid_config(&format!(
                    "Model configuration missing required field: {}",
                    field
                )));
            }
        }

        // Validate variants
        let variants = obj.get("variants").unwrap();
        if !variants.is_object() || variants.as_object().unwrap().is_empty() {
            return Err(BgRemovalError::invalid_config(
                "Model configuration must have at least one variant",
            ));
        }

        // Validate preprocessing
        let preprocessing = obj.get("preprocessing").unwrap();
        Self::validate_preprocessing_config(preprocessing)?;

        Ok(())
    }

    /// Validate preprocessing configuration
    fn validate_preprocessing_config(config: &serde_json::Value) -> Result<()> {
        if !config.is_object() {
            return Err(BgRemovalError::invalid_config(
                "Preprocessing configuration must be a JSON object",
            ));
        }

        let obj = config.as_object().unwrap();

        // Check required preprocessing fields
        if !obj.contains_key("target_size") {
            return Err(BgRemovalError::invalid_config(
                "Preprocessing configuration missing 'target_size'",
            ));
        }

        if !obj.contains_key("normalization") {
            return Err(BgRemovalError::invalid_config(
                "Preprocessing configuration missing 'normalization'",
            ));
        }

        // Validate target size
        let target_size = obj.get("target_size").unwrap();
        if !target_size.is_array() || target_size.as_array().unwrap().len() != 2 {
            return Err(BgRemovalError::invalid_config(
                "target_size must be an array of 2 numbers",
            ));
        }

        Ok(())
    }

    /// Validate a model file path
    pub fn validate_model_file_path(path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(BgRemovalError::invalid_config(&format!(
                "Model file does not exist: {}",
                path.display()
            )));
        }

        // Check for ONNX extension
        if path.extension().and_then(|s| s.to_str()) != Some("onnx") {
            return Err(BgRemovalError::invalid_config(&format!(
                "Model file must have .onnx extension: {}",
                path.display()
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelSource;
    use std::path::PathBuf;

    #[test]
    fn test_validate_embedded_model_name() {
        // Valid names
        assert!(ModelValidator::validate_embedded_model_name("isnet-fp16").is_ok());
        assert!(ModelValidator::validate_embedded_model_name("birefnet_portrait").is_ok());
        assert!(ModelValidator::validate_embedded_model_name("model123").is_ok());

        // Invalid names
        assert!(ModelValidator::validate_embedded_model_name("").is_err());
        assert!(ModelValidator::validate_embedded_model_name("model with spaces").is_err());
        assert!(ModelValidator::validate_embedded_model_name("model@special").is_err());
    }

    #[test]
    fn test_validate_variant_name() {
        // Valid variants
        assert!(ModelValidator::validate_variant_name("fp16").is_ok());
        assert!(ModelValidator::validate_variant_name("fp32").is_ok());
        assert!(ModelValidator::validate_variant_name("optimized-v2").is_ok());

        // Invalid variants
        assert!(ModelValidator::validate_variant_name("").is_err());
        assert!(ModelValidator::validate_variant_name("fp 16").is_err());
        assert!(ModelValidator::validate_variant_name("fp16!").is_err());
    }

    #[test]
    fn test_validate_model_spec() {
        // Valid embedded model spec
        let spec = ModelSpec {
            source: ModelSource::Embedded("isnet-fp16".to_string()),
            variant: Some("fp32".to_string()),
        };
        assert!(ModelValidator::validate_model_spec(&spec).is_ok());

        // Invalid embedded model spec
        let invalid_spec = ModelSpec {
            source: ModelSource::Embedded("".to_string()),
            variant: None,
        };
        assert!(ModelValidator::validate_model_spec(&invalid_spec).is_err());

        // External model spec (will fail because path doesn't exist)
        let external_spec = ModelSpec {
            source: ModelSource::External(PathBuf::from("/nonexistent/path")),
            variant: None,
        };
        assert!(ModelValidator::validate_model_spec(&external_spec).is_err());
    }

    #[test]
    fn test_validate_model_config() {
        // Valid config
        let valid_config = serde_json::json!({
            "name": "isnet",
            "description": "ISNet model",
            "variants": {
                "fp16": {},
                "fp32": {}
            },
            "preprocessing": {
                "target_size": [1024, 1024],
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            }
        });
        assert!(ModelValidator::validate_model_config(&valid_config).is_ok());

        // Missing required field
        let invalid_config = serde_json::json!({
            "name": "isnet",
            "variants": {
                "fp16": {}
            }
        });
        assert!(ModelValidator::validate_model_config(&invalid_config).is_err());

        // Empty variants
        let empty_variants = serde_json::json!({
            "name": "isnet",
            "description": "ISNet model",
            "variants": {},
            "preprocessing": {
                "target_size": [1024, 1024],
                "normalization": {}
            }
        });
        assert!(ModelValidator::validate_model_config(&empty_variants).is_err());
    }
}
