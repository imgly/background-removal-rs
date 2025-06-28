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

/// Convert CLI arguments to unified ProcessorConfig
pub(crate) struct CliConfigBuilder;

impl CliConfigBuilder {
    /// Build ProcessorConfig from CLI arguments
    pub(crate) fn from_cli(cli: &Cli) -> Result<ProcessorConfig> {
        // Parse model specification
        let (model_spec, _model_arg) = if let Some(model_arg) = &cli.model {
            let model_spec = ModelSpecParser::parse(model_arg);
            (model_spec, model_arg.clone())
        } else {
            // Show available downloaded models when none specified
            Self::show_available_models_info();

            // Check for downloaded models first
            #[cfg(feature = "cli")]
            {
                use crate::cache::ModelCache;
                if let Ok(cache) = ModelCache::new() {
                    if let Ok(downloaded_models) = cache.scan_cached_models() {
                        if !downloaded_models.is_empty() {
                            // Use first available downloaded model
                            let default_model = &downloaded_models[0].model_id;
                            let model_spec = ModelSpec {
                                source: ModelSource::Downloaded(default_model.clone()),
                                variant: None,
                            };
                            (model_spec, default_model.clone())
                        } else {
                            // No downloaded models available - use default model URL for auto-download
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
            #[cfg(not(feature = "cli"))]
            {
                return Err(anyhow::anyhow!(
                    "No model specified. Use --model to specify a model URL or path."
                ));
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
            .build()
            .context("Invalid configuration")?;

        Ok(config)
    }

    /// Show available downloaded models when no model is specified
    fn show_available_models_info() {
        #[cfg(feature = "cli")]
        {
            use crate::cache::ModelCache;
            if let Ok(cache) = ModelCache::new() {
                if let Ok(downloaded_models) = cache.scan_cached_models() {
                    if !downloaded_models.is_empty() {
                        println!("📦 Available downloaded models:");
                        for model in &downloaded_models {
                            println!("  • {} ({})", model.model_id, model.variants.join(", "));
                        }
                        println!(
                            "💡 To use a specific model: imgly-bgremove --model MODEL_ID input.jpg"
                        );
                        println!(
                            "Using first available model: {}",
                            downloaded_models[0].model_id
                        );
                        println!();
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
}
