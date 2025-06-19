//! Model specification parsing and management utilities
//!
//! Consolidates model handling logic that was previously in the CLI crate.

use crate::{
    error::{BgRemovalError, Result},
    models::{ModelSource, ModelSpec},
};
use std::path::{Path, PathBuf};

/// Utility for parsing and managing model specifications
pub struct ModelSpecParser;

impl ModelSpecParser {
    /// Parse model argument into ModelSpec with optional variant suffix
    ///
    /// Supports syntax: "model" or "model:variant"
    /// If path exists on filesystem, treats as external model.
    /// Otherwise treats as embedded model name.
    ///
    /// # Arguments
    /// * `model_arg` - Model argument string
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::utils::ModelSpecParser;
    ///
    /// // Embedded model without variant
    /// let spec = ModelSpecParser::parse("isnet-fp16");
    ///
    /// // Embedded model with variant
    /// let spec = ModelSpecParser::parse("birefnet:fp32");
    ///
    /// // External model path
    /// let spec = ModelSpecParser::parse("/path/to/model");
    /// ```
    pub fn parse(model_arg: &str) -> ModelSpec {
        // Check for suffix syntax: "model:variant"
        if let Some((path_part, variant_part)) = model_arg.split_once(':') {
            let source = if Path::new(path_part).exists() {
                ModelSource::External(PathBuf::from(path_part))
            } else {
                ModelSource::Embedded(path_part.to_string())
            };

            return ModelSpec {
                source,
                variant: Some(variant_part.to_string()),
            };
        }

        // No suffix - determine source type based on path existence
        let source = if Path::new(model_arg).exists() {
            ModelSource::External(PathBuf::from(model_arg))
        } else {
            ModelSource::Embedded(model_arg.to_string())
        };

        ModelSpec {
            source,
            variant: None,
        }
    }

    /// Resolve the final variant to use based on precedence rules
    ///
    /// Precedence order:
    /// 1. CLI parameter (highest)
    /// 2. Suffix in model specification
    /// 3. Auto-detection (prefers fp16)
    /// 4. First available variant
    ///
    /// # Arguments
    /// * `model_spec` - Model specification
    /// * `cli_variant` - Variant specified via CLI parameter
    /// * `available_variants` - List of available variants for the model
    pub fn resolve_variant(
        model_spec: &ModelSpec,
        cli_variant: Option<&str>,
        available_variants: &[String],
    ) -> Result<String> {
        // 1. CLI parameter has highest precedence
        if let Some(variant) = cli_variant {
            if available_variants.contains(&variant.to_string()) {
                return Ok(variant.to_string());
            }
            return Err(BgRemovalError::invalid_config(&format!(
                "Variant '{}' not available. Available variants: {:?}",
                variant, available_variants
            )));
        }

        // 2. Suffix syntax has medium precedence
        if let Some(variant) = &model_spec.variant {
            if available_variants.contains(variant) {
                return Ok(variant.clone());
            } else {
                return Err(BgRemovalError::invalid_config(&format!(
                    "Variant '{}' not available. Available variants: {:?}",
                    variant, available_variants
                )));
            }
        }

        // 3. Auto-detection: prefer fp16, fallback to fp32
        if available_variants.contains(&"fp16".to_string()) {
            return Ok("fp16".to_string());
        }

        if available_variants.contains(&"fp32".to_string()) {
            return Ok("fp32".to_string());
        }

        // 4. Use first available variant
        if let Some(first) = available_variants.first() {
            return Ok(first.clone());
        }

        Err(BgRemovalError::invalid_config(
            "No variants available for model",
        ))
    }

    /// Validate model specification
    ///
    /// Checks if embedded model name is valid or external path exists
    pub fn validate(model_spec: &ModelSpec) -> Result<()> {
        match &model_spec.source {
            ModelSource::External(path) => {
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
            },
            ModelSource::Embedded(name) => {
                // Basic validation - check for valid characters
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
            },
        }

        // Validate variant if specified
        if let Some(variant) = &model_spec.variant {
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
        }

        Ok(())
    }

    /// Convert ModelSpec back to string representation
    ///
    /// Reconstructs the original string format used to create the ModelSpec
    pub fn to_string(model_spec: &ModelSpec) -> String {
        let base = match &model_spec.source {
            ModelSource::External(path) => path.to_string_lossy().to_string(),
            ModelSource::Embedded(name) => name.clone(),
        };

        if let Some(variant) = &model_spec.variant {
            format!("{}:{}", base, variant)
        } else {
            base
        }
    }

    /// Check if a model specification represents an embedded model
    pub fn is_embedded(model_spec: &ModelSpec) -> bool {
        matches!(model_spec.source, ModelSource::Embedded(_))
    }

    /// Check if a model specification represents an external model
    pub fn is_external(model_spec: &ModelSpec) -> bool {
        matches!(model_spec.source, ModelSource::External(_))
    }

    /// Get the model name (for embedded) or directory name (for external)
    pub fn get_model_name(model_spec: &ModelSpec) -> String {
        match &model_spec.source {
            ModelSource::External(path) => path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            ModelSource::Embedded(name) => name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_embedded_without_variant() {
        let spec = ModelSpecParser::parse("isnet");
        match spec.source {
            ModelSource::Embedded(name) => assert_eq!(name, "isnet"),
            _ => panic!("Expected embedded model"),
        }
        assert_eq!(spec.variant, None);
    }

    #[test]
    fn test_parse_embedded_with_variant() {
        let spec = ModelSpecParser::parse("birefnet:fp32");
        match spec.source {
            ModelSource::Embedded(name) => assert_eq!(name, "birefnet"),
            _ => panic!("Expected embedded model"),
        }
        assert_eq!(spec.variant, Some("fp32".to_string()));
    }

    #[test]
    fn test_parse_nonexistent_external() {
        // Non-existent path should be treated as embedded
        let spec = ModelSpecParser::parse("/non/existent/path");
        match spec.source {
            ModelSource::Embedded(name) => assert_eq!(name, "/non/existent/path"),
            _ => panic!("Expected embedded model for non-existent path"),
        }
    }

    #[test]
    fn test_resolve_variant_cli_precedence() {
        let available = vec!["fp16".to_string(), "fp32".to_string()];

        let spec = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: Some("fp32".to_string()),
        };

        // CLI parameter should win over suffix
        let result = ModelSpecParser::resolve_variant(&spec, Some("fp16"), &available).unwrap();
        assert_eq!(result, "fp16");
    }

    #[test]
    fn test_resolve_variant_suffix_precedence() {
        let available = vec!["fp16".to_string(), "fp32".to_string()];

        let spec = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: Some("fp32".to_string()),
        };

        // Suffix should be used when no CLI param
        let result = ModelSpecParser::resolve_variant(&spec, None, &available).unwrap();
        assert_eq!(result, "fp32");
    }

    #[test]
    fn test_resolve_variant_auto_detection() {
        let available = vec!["fp16".to_string(), "fp32".to_string()];

        let spec = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: None,
        };

        // Should prefer fp16 in auto-detection
        let result = ModelSpecParser::resolve_variant(&spec, None, &available).unwrap();
        assert_eq!(result, "fp16");
    }

    #[test]
    fn test_resolve_variant_fallback_fp32() {
        let available = vec!["fp32".to_string()];

        let spec = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: None,
        };

        // Should fallback to fp32 if fp16 not available
        let result = ModelSpecParser::resolve_variant(&spec, None, &available).unwrap();
        assert_eq!(result, "fp32");
    }

    #[test]
    fn test_resolve_variant_invalid() {
        let available = vec!["fp16".to_string(), "fp32".to_string()];

        let spec = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: None,
        };

        // Should error on invalid variant
        let result = ModelSpecParser::resolve_variant(&spec, Some("invalid"), &available);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_embedded() {
        let spec = ModelSpec {
            source: ModelSource::Embedded("isnet-fp16".to_string()),
            variant: None,
        };
        assert!(ModelSpecParser::validate(&spec).is_ok());

        let invalid_spec = ModelSpec {
            source: ModelSource::Embedded("".to_string()),
            variant: None,
        };
        assert!(ModelSpecParser::validate(&invalid_spec).is_err());
    }

    #[test]
    fn test_to_string() {
        let spec1 = ModelSpec {
            source: ModelSource::Embedded("isnet".to_string()),
            variant: None,
        };
        assert_eq!(ModelSpecParser::to_string(&spec1), "isnet");

        let spec2 = ModelSpec {
            source: ModelSource::Embedded("birefnet".to_string()),
            variant: Some("fp32".to_string()),
        };
        assert_eq!(ModelSpecParser::to_string(&spec2), "birefnet:fp32");
    }

    #[test]
    fn test_is_embedded_external() {
        let embedded_spec = ModelSpec {
            source: ModelSource::Embedded("isnet".to_string()),
            variant: None,
        };
        assert!(ModelSpecParser::is_embedded(&embedded_spec));
        assert!(!ModelSpecParser::is_external(&embedded_spec));

        let external_spec = ModelSpec {
            source: ModelSource::External(PathBuf::from("/path/to/model")),
            variant: None,
        };
        assert!(!ModelSpecParser::is_embedded(&external_spec));
        assert!(ModelSpecParser::is_external(&external_spec));
    }

    #[test]
    fn test_get_model_name() {
        let embedded_spec = ModelSpec {
            source: ModelSource::Embedded("isnet-fp16".to_string()),
            variant: None,
        };
        assert_eq!(
            ModelSpecParser::get_model_name(&embedded_spec),
            "isnet-fp16"
        );

        let external_spec = ModelSpec {
            source: ModelSource::External(PathBuf::from("/path/to/my_model")),
            variant: None,
        };
        assert_eq!(ModelSpecParser::get_model_name(&external_spec), "my_model");
    }
}
