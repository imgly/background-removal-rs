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
    /// Parse model argument into `ModelSpec` with optional variant suffix
    ///
    /// Supports syntax: "model" or "model:variant"
    /// If path exists on filesystem, treats as external model.
    /// If argument is a URL, treats as download target and converts to model ID.
    /// Otherwise treats as downloaded model ID.
    ///
    /// # Arguments
    /// * `model_arg` - Model argument string
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::utils::ModelSpecParser;
    ///
    /// // Downloaded model ID without variant
    /// let spec = ModelSpecParser::parse("imgly--isnet-general-onnx");
    ///
    /// // Downloaded model ID with variant
    /// let spec = ModelSpecParser::parse("imgly--birefnet-portrait:fp32");
    ///
    /// // External model path
    /// let spec = ModelSpecParser::parse("/path/to/model");
    ///
    /// // URL (converts to model ID)
    /// let spec = ModelSpecParser::parse("https://huggingface.co/imgly/isnet-general-onnx");
    /// ```
    #[must_use]
    pub fn parse(model_arg: &str) -> ModelSpec {
        // Check for suffix syntax: "model:variant", but exclude URLs
        if !model_arg.starts_with("http") && model_arg.contains(':') {
            if let Some((path_part, variant_part)) = model_arg.split_once(':') {
                let source = if Path::new(path_part).exists() {
                    ModelSource::External(PathBuf::from(path_part))
                } else {
                    // Assume it's a downloaded model ID
                    ModelSource::Downloaded(path_part.to_string())
                };

                return ModelSpec {
                    source,
                    variant: Some(variant_part.to_string()),
                };
            }
        }

        // No suffix - determine source type
        let source = if Path::new(model_arg).exists() {
            ModelSource::External(PathBuf::from(model_arg))
        } else if model_arg.starts_with("http") {
            // URL - convert to model ID
            #[cfg(feature = "cli")]
            {
                ModelSource::Downloaded(crate::cache::ModelCache::url_to_model_id(model_arg))
            }
            #[cfg(not(feature = "cli"))]
            {
                // Fallback for non-CLI builds
                ModelSource::Downloaded(model_arg.to_string())
            }
        } else {
            // Assume it's a downloaded model ID
            ModelSource::Downloaded(model_arg.to_string())
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
            return Err(BgRemovalError::invalid_config(format!(
                "Variant '{}' not available. Available variants: {:?}",
                variant, available_variants
            )));
        }

        // 2. Suffix syntax has medium precedence
        if let Some(variant) = &model_spec.variant {
            if available_variants.contains(variant) {
                return Ok(variant.clone());
            }
            return Err(BgRemovalError::invalid_config(format!(
                "Variant '{}' not available. Available variants: {:?}",
                variant, available_variants
            )));
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
    /// Checks if downloaded model ID is valid or external path exists
    pub fn validate(model_spec: &ModelSpec) -> Result<()> {
        match &model_spec.source {
            ModelSource::External(path) => {
                if !path.exists() {
                    return Err(BgRemovalError::invalid_config(format!(
                        "External model path does not exist: {}",
                        path.display()
                    )));
                }
                if !path.is_dir() {
                    return Err(BgRemovalError::invalid_config(format!(
                        "External model path must be a directory: {}",
                        path.display()
                    )));
                }
            },
            ModelSource::Downloaded(model_id) => {
                // Basic validation - check for valid characters
                if model_id.is_empty() {
                    return Err(BgRemovalError::invalid_config(
                        "Downloaded model ID cannot be empty",
                    ));
                }

                // Check for reasonable model ID format
                if !model_id
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
                {
                    return Err(BgRemovalError::invalid_config(format!(
                        "Invalid characters in downloaded model ID: {}",
                        model_id
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
                return Err(BgRemovalError::invalid_config(format!(
                    "Invalid characters in model variant: {}",
                    variant
                )));
            }
        }

        Ok(())
    }

    /// Convert `ModelSpec` back to string representation
    ///
    /// Reconstructs the original string format used to create the `ModelSpec`
    #[must_use]
    pub fn to_string(model_spec: &ModelSpec) -> String {
        let base = match &model_spec.source {
            ModelSource::External(path) => path.to_string_lossy().to_string(),
            ModelSource::Downloaded(model_id) => model_id.clone(),
        };

        if let Some(variant) = &model_spec.variant {
            format!("{}:{}", base, variant)
        } else {
            base
        }
    }

    /// Check if a model specification represents an external model
    #[must_use]
    pub fn is_external(model_spec: &ModelSpec) -> bool {
        matches!(model_spec.source, ModelSource::External(_))
    }

    /// Check if a model specification represents a downloaded model
    #[must_use]
    pub fn is_downloaded(model_spec: &ModelSpec) -> bool {
        matches!(model_spec.source, ModelSource::Downloaded(_))
    }

    /// Get the model name (directory name for external, model ID for downloaded)
    #[must_use]
    pub fn get_model_name(model_spec: &ModelSpec) -> String {
        match &model_spec.source {
            ModelSource::External(path) => path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            ModelSource::Downloaded(model_id) => model_id.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_downloaded_without_variant() {
        let spec = ModelSpecParser::parse("imgly--isnet-general-onnx");
        match spec.source {
            ModelSource::Downloaded(model_id) => assert_eq!(model_id, "imgly--isnet-general-onnx"),
            ModelSource::External(_) => panic!("Expected downloaded model"),
        }
        assert_eq!(spec.variant, None);
    }

    #[test]
    fn test_parse_downloaded_with_variant() {
        let spec = ModelSpecParser::parse("imgly--birefnet-portrait:fp32");
        match spec.source {
            ModelSource::Downloaded(model_id) => assert_eq!(model_id, "imgly--birefnet-portrait"),
            ModelSource::External(_) => panic!("Expected downloaded model"),
        }
        assert_eq!(spec.variant, Some("fp32".to_string()));
    }

    #[test]
    fn test_parse_url() {
        let url = "https://huggingface.co/imgly/isnet-general-onnx";
        let spec = ModelSpecParser::parse(url);
        match spec.source {
            ModelSource::Downloaded(model_id) => {
                #[cfg(feature = "cli")]
                assert_eq!(model_id, "imgly--isnet-general-onnx");
                #[cfg(not(feature = "cli"))]
                assert_eq!(model_id, "https://huggingface.co/imgly/isnet-general-onnx");
            },
            ModelSource::External(_) => panic!("Expected downloaded model for URL"),
        }
    }

    #[test]
    #[cfg(feature = "cli")]
    fn test_url_conversion_consistency() {
        let url = "https://huggingface.co/imgly/isnet-general-onnx";

        // Test direct cache function call
        let cache_result = crate::cache::ModelCache::url_to_model_id(url);
        assert_eq!(cache_result, "imgly--isnet-general-onnx");

        // Test through ModelSpecParser
        let spec = ModelSpecParser::parse(url);
        match spec.source {
            ModelSource::Downloaded(model_id) => {
                assert_eq!(
                    model_id, cache_result,
                    "ModelSpecParser should produce same result as direct cache call"
                );
            },
            ModelSource::External(_) => panic!("Expected downloaded model for URL"),
        }
    }

    #[test]
    fn test_parse_nonexistent_external() {
        // Non-existent path should be treated as downloaded model ID
        let spec = ModelSpecParser::parse("/non/existent/path");
        match spec.source {
            ModelSource::Downloaded(model_id) => assert_eq!(model_id, "/non/existent/path"),
            _ => panic!("Expected downloaded model for non-existent path"),
        }
    }

    #[test]
    fn test_resolve_variant_cli_precedence() {
        let available = vec!["fp16".to_string(), "fp32".to_string()];

        let spec = ModelSpec {
            source: ModelSource::Downloaded("test-model".to_string()),
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
            source: ModelSource::Downloaded("test-model".to_string()),
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
            source: ModelSource::Downloaded("test-model".to_string()),
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
            source: ModelSource::Downloaded("test-model".to_string()),
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
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: None,
        };

        // Should error on invalid variant
        let result = ModelSpecParser::resolve_variant(&spec, Some("invalid"), &available);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_downloaded() {
        let spec = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: None,
        };
        assert!(ModelSpecParser::validate(&spec).is_ok());

        let invalid_spec = ModelSpec {
            source: ModelSource::Downloaded(String::new()),
            variant: None,
        };
        assert!(ModelSpecParser::validate(&invalid_spec).is_err());
    }

    #[test]
    fn test_to_string() {
        let spec1 = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: None,
        };
        assert_eq!(
            ModelSpecParser::to_string(&spec1),
            "imgly--isnet-general-onnx"
        );

        let spec2 = ModelSpec {
            source: ModelSource::Downloaded("imgly--birefnet-portrait".to_string()),
            variant: Some("fp32".to_string()),
        };
        assert_eq!(
            ModelSpecParser::to_string(&spec2),
            "imgly--birefnet-portrait:fp32"
        );
    }

    #[test]
    fn test_is_downloaded_external() {
        let downloaded_spec = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: None,
        };
        assert!(ModelSpecParser::is_downloaded(&downloaded_spec));
        assert!(!ModelSpecParser::is_external(&downloaded_spec));

        let external_spec = ModelSpec {
            source: ModelSource::External(PathBuf::from("/path/to/model")),
            variant: None,
        };
        assert!(!ModelSpecParser::is_downloaded(&external_spec));
        assert!(ModelSpecParser::is_external(&external_spec));
    }

    #[test]
    fn test_get_model_name() {
        let downloaded_spec = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: None,
        };
        assert_eq!(
            ModelSpecParser::get_model_name(&downloaded_spec),
            "imgly--isnet-general-onnx"
        );

        let external_spec = ModelSpec {
            source: ModelSource::External(PathBuf::from("/path/to/my_model")),
            variant: None,
        };
        assert_eq!(ModelSpecParser::get_model_name(&external_spec), "my_model");
    }
}
