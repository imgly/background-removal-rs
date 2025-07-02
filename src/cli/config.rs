//! Configuration conversion utilities for CLI arguments

#![allow(dead_code)]

use crate::cli::main_impl::{Cli, CliOutputFormat};
use crate::{
    config::OutputFormat,
    models::{ModelSource, ModelSpec},
    processor::{ProcessorConfig, ProcessorConfigBuilder},
    utils::{ConfigValidator, ExecutionProviderManager, ModelSpecParser},
};
use anyhow::{Context, Result};

/// Convert CLI arguments to unified `ProcessorConfig`
pub(crate) struct CliConfigBuilder;

impl CliConfigBuilder {
    /// Build `ProcessorConfig` from CLI arguments
    pub(crate) fn from_cli(cli: &Cli) -> Result<ProcessorConfig> {
        // Parse model specification
        let (model_spec, _model_arg) = if let Some(model_arg) = &cli.model {
            let model_spec = ModelSpecParser::parse(model_arg);
            (model_spec, model_arg.clone())
        } else {
            // Show available downloaded models when none specified
            Self::show_available_models_info();

            // Check for downloaded models first
            {
                use crate::cache::ModelCache;
                if let Ok(cache) = ModelCache::new() {
                    if let Ok(downloaded_models) = cache.scan_cached_models() {
                        if downloaded_models.is_empty() {
                            // No downloaded models available - use default model URL for auto-download
                            let default_url = ModelCache::get_default_model_url();
                            let model_spec = ModelSpec {
                                source: ModelSource::Downloaded(ModelCache::url_to_model_id(
                                    default_url,
                                )),
                                variant: None,
                            };
                            (model_spec, ModelCache::url_to_model_id(default_url))
                        } else {
                            // Use first available downloaded model - safe indexing with bounds check
                            if let Some(first_model) = downloaded_models.first() {
                                let default_model = &first_model.model_id;
                                let model_spec = ModelSpec {
                                    source: ModelSource::Downloaded(default_model.clone()),
                                    variant: None,
                                };
                                (model_spec, default_model.clone())
                            } else {
                                // Fallback to default model if somehow the vec became empty between checks
                                let default_url = ModelCache::get_default_model_url();
                                let model_spec = ModelSpec {
                                    source: ModelSource::Downloaded(ModelCache::url_to_model_id(
                                        default_url,
                                    )),
                                    variant: None,
                                };
                                (model_spec, ModelCache::url_to_model_id(default_url))
                            }
                        }
                    } else {
                        // Cache scan failed - use default model URL for auto-download
                        let default_url = ModelCache::get_default_model_url();
                        let model_spec = ModelSpec {
                            source: ModelSource::Downloaded(ModelCache::url_to_model_id(
                                default_url,
                            )),
                            variant: None,
                        };
                        (model_spec, ModelCache::url_to_model_id(default_url))
                    }
                } else {
                    // Cache creation failed - use default model URL for auto-download
                    let default_url = ModelCache::get_default_model_url();
                    let model_spec = ModelSpec {
                        source: ModelSource::Downloaded(ModelCache::url_to_model_id(default_url)),
                        variant: None,
                    };
                    (model_spec, ModelCache::url_to_model_id(default_url))
                }
            }
        };

        // Parse execution provider and backend type
        let (backend_type, execution_provider) =
            ExecutionProviderManager::parse_provider_string(&cli.execution_provider)
                .context("Invalid execution provider format")?;

        // Parse output format
        let output_format = match cli.format {
            CliOutputFormat::Png => OutputFormat::Png,
            CliOutputFormat::Jpeg => OutputFormat::Jpeg,
            CliOutputFormat::Webp => OutputFormat::WebP,
            CliOutputFormat::Tiff => OutputFormat::Tiff,
            CliOutputFormat::Rgba8 => OutputFormat::Rgba8,
        };

        // Create final model spec with resolved variant
        let final_model_spec = ModelSpec {
            source: model_spec.source.clone(),
            variant: cli.variant.clone().or_else(|| model_spec.variant.clone()),
        };

        // Build configuration
        let config = ProcessorConfigBuilder::new()
            .model_spec(final_model_spec)
            .backend_type(backend_type)
            .execution_provider(execution_provider)
            .output_format(output_format)
            .jpeg_quality(cli.jpeg_quality)
            .webp_quality(cli.webp_quality)
            .debug(cli.verbose >= 2)
            // Use the same thread count for both intra and inter operations
            // This provides optimal performance in most cases
            .intra_threads(cli.threads)
            .inter_threads(cli.threads)
            .preserve_color_profiles(cli.preserve_color_profiles)
            .disable_cache(cli.no_cache)
            .build()
            .context("Invalid configuration")?;

        Ok(config)
    }

    /// Show available downloaded models when no model is specified
    fn show_available_models_info() {
        use tracing::info;
        {
            use crate::cache::ModelCache;
            if let Ok(cache) = ModelCache::new() {
                if let Ok(downloaded_models) = cache.scan_cached_models() {
                    if !downloaded_models.is_empty() {
                        info!("📦 Available downloaded models:");
                        for model in &downloaded_models {
                            info!("  • {} ({})", model.model_id, model.variants.join(", "));
                        }
                        info!(
                            "💡 To use a specific model: imgly-bgremove --model MODEL_ID input.jpg"
                        );
                        if let Some(first_model) = downloaded_models.first() {
                            info!("Using first available model: {}", first_model.model_id);
                        }
                    }
                }
            }
        }
    }

    /// Validate CLI arguments for consistency
    pub(crate) fn validate_cli(cli: &Cli) -> Result<()> {
        // Validate execution provider format
        ExecutionProviderManager::parse_provider_string(&cli.execution_provider)
            .context("Invalid execution provider format")?;

        // Validate quality settings using shared validator
        ConfigValidator::validate_quality_settings(cli.jpeg_quality, cli.webp_quality)
            .context("Invalid quality settings")?;

        // Validate model specification if provided
        if let Some(model_arg) = &cli.model {
            let model_spec = ModelSpecParser::parse(model_arg);
            ModelSpecParser::validate(&model_spec).context("Invalid model specification")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, CliOutputFormat};
    use crate::{config::ExecutionProvider, processor::BackendType};

    fn create_test_cli() -> Cli {
        Cli {
            input: vec!["test.jpg".to_string()],
            output: None,
            format: CliOutputFormat::Png,
            execution_provider: "onnx:auto".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            threads: 0,
            verbose: 0,
            recursive: false,
            pattern: None,
            show_providers: false,
            model: None,
            variant: None,
            preserve_color_profiles: true,
            only_download: false,
            list_models: false,
            clear_cache: false,
            show_cache_dir: false,
            cache_dir: None,
            no_cache: false,
        }
    }

    #[test]
    fn test_cli_config_conversion() {
        let mut cli = create_test_cli();
        // Provide a model since there are no embedded models
        cli.model = Some("test-model".to_string());

        let config = CliConfigBuilder::from_cli(&cli).unwrap();

        assert_eq!(config.backend_type, BackendType::Onnx);
        assert_eq!(config.execution_provider, ExecutionProvider::Auto);
        assert_eq!(config.output_format, OutputFormat::Png);
        assert_eq!(config.jpeg_quality, 90);
        assert_eq!(config.webp_quality, 85);
        assert!(!config.debug);
    }

    #[test]
    fn test_cli_validation() {
        let mut cli = create_test_cli();
        assert!(CliConfigBuilder::validate_cli(&cli).is_ok());

        // Test invalid execution provider
        cli.execution_provider = "invalid:provider".to_string();
        assert!(CliConfigBuilder::validate_cli(&cli).is_err());

        // Reset and test invalid quality
        cli.execution_provider = "onnx:auto".to_string();
        cli.jpeg_quality = 150;
        assert!(CliConfigBuilder::validate_cli(&cli).is_err());
    }

    #[test]
    fn test_no_cache_flag_config() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());
        cli.no_cache = true;

        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(config.disable_cache);

        // Test default (cache enabled)
        cli.no_cache = false;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(!config.disable_cache);
    }

    #[test]
    fn test_cli_output_format_conversion() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test all CLI output formats convert correctly
        cli.format = CliOutputFormat::Png;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.output_format, OutputFormat::Png);

        cli.format = CliOutputFormat::Jpeg;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.output_format, OutputFormat::Jpeg);

        cli.format = CliOutputFormat::Webp;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.output_format, OutputFormat::WebP);

        cli.format = CliOutputFormat::Tiff;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.output_format, OutputFormat::Tiff);

        cli.format = CliOutputFormat::Rgba8;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.output_format, OutputFormat::Rgba8);
    }

    #[test]
    fn test_execution_provider_parsing() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test various execution provider formats
        let test_cases = vec![
            ("onnx:auto", BackendType::Onnx, ExecutionProvider::Auto),
            ("onnx:cpu", BackendType::Onnx, ExecutionProvider::Cpu),
            ("onnx:cuda", BackendType::Onnx, ExecutionProvider::Cuda),
            ("onnx:coreml", BackendType::Onnx, ExecutionProvider::CoreMl),
            ("tract:cpu", BackendType::Tract, ExecutionProvider::Cpu),
            ("onnx", BackendType::Onnx, ExecutionProvider::Auto), // Default provider
            ("tract", BackendType::Tract, ExecutionProvider::Cpu), // Default provider
        ];

        for (provider_str, expected_backend, expected_provider) in test_cases {
            cli.execution_provider = provider_str.to_string();
            let config = CliConfigBuilder::from_cli(&cli).unwrap();
            assert_eq!(
                config.backend_type, expected_backend,
                "Failed for provider string: {}",
                provider_str
            );
            assert_eq!(
                config.execution_provider, expected_provider,
                "Failed for provider string: {}",
                provider_str
            );
        }
    }

    #[test]
    fn test_thread_configuration() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test thread count configuration
        cli.threads = 0; // Auto-detect
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.intra_threads, 0);
        assert_eq!(config.inter_threads, 0);

        cli.threads = 4;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.intra_threads, 4);
        assert_eq!(config.inter_threads, 4); // CLI uses same value for both

        cli.threads = 8;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.intra_threads, 8);
        assert_eq!(config.inter_threads, 8);
    }

    #[test]
    fn test_quality_settings() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test JPEG quality
        cli.jpeg_quality = 75;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.jpeg_quality, 75);

        // Test WebP quality
        cli.webp_quality = 95;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.webp_quality, 95);

        // Test default values
        cli.jpeg_quality = 90;
        cli.webp_quality = 85;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.jpeg_quality, 90);
        assert_eq!(config.webp_quality, 85);
    }

    #[test]
    fn test_debug_mode_configuration() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test verbose levels and debug mode
        cli.verbose = 0;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(!config.debug);

        cli.verbose = 1;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(!config.debug);

        cli.verbose = 2; // Debug mode threshold
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(config.debug);

        cli.verbose = 3;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(config.debug);
    }

    #[test]
    fn test_color_profile_preservation() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test color profile preservation flag
        cli.preserve_color_profiles = true;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(config.preserve_color_profiles);

        cli.preserve_color_profiles = false;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert!(!config.preserve_color_profiles);
    }

    #[test]
    fn test_model_variant_handling() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test without variant
        cli.variant = None;
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.model_spec.variant, None);

        // Test with variant
        cli.variant = Some("fp16".to_string());
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.model_spec.variant, Some("fp16".to_string()));

        cli.variant = Some("fp32".to_string());
        let config = CliConfigBuilder::from_cli(&cli).unwrap();
        assert_eq!(config.model_spec.variant, Some("fp32".to_string()));
    }

    #[test]
    fn test_validation_edge_cases() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test valid quality ranges
        cli.jpeg_quality = 0;
        cli.webp_quality = 0;
        assert!(CliConfigBuilder::validate_cli(&cli).is_ok());

        cli.jpeg_quality = 100;
        cli.webp_quality = 100;
        assert!(CliConfigBuilder::validate_cli(&cli).is_ok());

        // Test invalid quality values
        cli.jpeg_quality = 101;
        assert!(CliConfigBuilder::validate_cli(&cli).is_err());

        cli.jpeg_quality = 90; // Reset to valid
        cli.webp_quality = 150;
        assert!(CliConfigBuilder::validate_cli(&cli).is_err());
    }

    #[test]
    fn test_invalid_execution_provider_validation() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test invalid provider formats
        let invalid_providers = vec![
            "invalid:provider",
            "unsupported:backend",
            "onnx:invalid_provider",
            "tract:invalid_provider",
            "malformed",
            "",
            ":",
            "onnx:",
            ":auto",
        ];

        for invalid_provider in invalid_providers {
            cli.execution_provider = invalid_provider.to_string();
            let result = CliConfigBuilder::validate_cli(&cli);
            assert!(
                result.is_err(),
                "Should fail validation for provider: {}",
                invalid_provider
            );
        }
    }

    #[test]
    fn test_model_spec_validation() {
        let mut cli = create_test_cli();

        // Test with invalid model specifications
        cli.model = Some("".to_string()); // Empty model
        let result = CliConfigBuilder::validate_cli(&cli);
        assert!(result.is_err());

        // Test with valid model specifications
        cli.model = Some("test-model".to_string());
        assert!(CliConfigBuilder::validate_cli(&cli).is_ok());

        cli.model = Some("downloaded:model-name".to_string());
        assert!(CliConfigBuilder::validate_cli(&cli).is_ok());

        cli.model = Some("https://huggingface.co/user/model".to_string());
        assert!(CliConfigBuilder::validate_cli(&cli).is_ok());
    }

    #[test]
    fn test_config_builder_comprehensive() {
        let cli = Cli {
            input: vec!["test1.jpg".to_string(), "test2.png".to_string()],
            output: Some("output.png".to_string()),
            format: CliOutputFormat::Webp,
            execution_provider: "onnx:coreml".to_string(),
            jpeg_quality: 85,
            webp_quality: 95,
            threads: 6,
            verbose: 3,
            recursive: true,
            pattern: Some("*.jpg".to_string()),
            show_providers: false,
            model: Some("custom-model".to_string()),
            variant: Some("fp16".to_string()),
            preserve_color_profiles: false,
            only_download: false,
            list_models: false,
            clear_cache: false,
            show_cache_dir: false,
            cache_dir: Some("/custom/cache".to_string()),
            no_cache: true,
        };

        let config = CliConfigBuilder::from_cli(&cli).unwrap();

        // Verify all settings are correctly converted
        assert_eq!(config.backend_type, BackendType::Onnx);
        assert_eq!(config.execution_provider, ExecutionProvider::CoreMl);
        assert_eq!(config.output_format, OutputFormat::WebP);
        assert_eq!(config.jpeg_quality, 85);
        assert_eq!(config.webp_quality, 95);
        assert_eq!(config.intra_threads, 6);
        assert_eq!(config.inter_threads, 6);
        assert!(config.debug); // verbose >= 2
        assert!(!config.preserve_color_profiles);
        assert!(config.disable_cache);
        assert_eq!(config.model_spec.variant, Some("fp16".to_string()));
    }

    #[test]
    fn test_default_model_selection_logic() {
        let mut cli = create_test_cli();
        cli.model = None; // No model specified

        // This should use the default model selection logic
        // The exact behavior depends on cache state, but should not panic
        let result = CliConfigBuilder::from_cli(&cli);

        // Should either succeed with a default model or potentially fail gracefully
        // depending on cache state and available models
        match result {
            Ok(config) => {
                // If successful, should have a valid model spec
                assert!(!config.model_spec.source.display_name().is_empty());
            },
            Err(_) => {
                // May fail if no models are available, which is acceptable for testing
            },
        }
    }

    #[test]
    fn test_show_available_models_info_function() {
        // Test that the function doesn't panic when called
        // Note: This function logs to tracing, so we can't easily test output
        // but we can verify it doesn't crash
        CliConfigBuilder::show_available_models_info();

        // If we reach here, the function completed without panicking
        assert!(true);
    }

    #[test]
    fn test_cli_config_builder_error_propagation() {
        let mut cli = create_test_cli();
        cli.model = Some("test-model".to_string());

        // Test that validation errors are properly propagated
        cli.execution_provider = "invalid:format".to_string();
        let result = CliConfigBuilder::from_cli(&cli);
        assert!(result.is_err());

        // Test that the error contains useful information
        let error = result.unwrap_err();
        let error_msg = error.to_string();
        assert!(error_msg.contains("execution provider") || error_msg.contains("Invalid"));
    }

    #[test]
    fn test_cli_output_format_enum_properties() {
        // Test that all enum variants are unique (using Vec instead of HashSet)
        let formats = vec![
            CliOutputFormat::Png,
            CliOutputFormat::Jpeg,
            CliOutputFormat::Webp,
            CliOutputFormat::Tiff,
            CliOutputFormat::Rgba8,
        ];

        // Check uniqueness by ensuring no duplicates in vec
        for (i, format1) in formats.iter().enumerate() {
            for (j, format2) in formats.iter().enumerate() {
                if i != j {
                    assert_ne!(format1, format2);
                }
            }
        }

        // Test ordering consistency
        assert!(CliOutputFormat::Png < CliOutputFormat::Jpeg);
        assert!(CliOutputFormat::Jpeg < CliOutputFormat::Webp);
        assert!(CliOutputFormat::Webp < CliOutputFormat::Tiff);
        assert!(CliOutputFormat::Tiff < CliOutputFormat::Rgba8);

        // Test copy/clone behavior
        let format1 = CliOutputFormat::Png;
        let format2 = format1; // Copy
        assert_eq!(format1, format2);

        let format3 = format1.clone(); // Clone (even though Copy is implemented)
        assert_eq!(format1, format3);
    }
}
