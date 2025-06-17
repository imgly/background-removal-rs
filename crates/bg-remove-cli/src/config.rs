//! Configuration conversion utilities for CLI arguments

use anyhow::{Context, Result};
use bg_remove_core::{
    processor::{ProcessorConfig, ProcessorConfigBuilder},
    utils::{ColorParser, ExecutionProviderManager, ModelSpecParser},
    config::{ColorManagementConfig, OutputFormat},
    models::{get_available_embedded_models, ModelSource, ModelSpec},
};
use crate::Cli;

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
            // Use first available embedded model
            let available_embedded = get_available_embedded_models();
            if available_embedded.is_empty() {
                anyhow::bail!(
                    "No model specified and no embedded models available. Use --model to specify a model name or path, or build with embed-* features."
                );
            }
            let default_model = &available_embedded[0];
            let model_spec = ModelSpec {
                source: ModelSource::Embedded(default_model.clone()),
                variant: None,
            };
            (model_spec, default_model.clone())
        };

        // Parse execution provider and backend type
        let (backend_type, execution_provider) = ExecutionProviderManager::parse_provider_string(&cli.execution_provider)
            .context("Invalid execution provider format")?;

        // Parse background color
        let background_color = ColorParser::parse_hex(&cli.background_color)
            .context("Invalid background color format")?;

        // Parse output format
        let output_format = match cli.format {
            crate::CliOutputFormat::Png => OutputFormat::Png,
            crate::CliOutputFormat::Jpeg => OutputFormat::Jpeg,
            crate::CliOutputFormat::Webp => OutputFormat::WebP,
            crate::CliOutputFormat::Tiff => OutputFormat::Tiff,
            crate::CliOutputFormat::Rgba8 => OutputFormat::Rgba8,
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
            .background_color(background_color)
            .jpeg_quality(cli.jpeg_quality)
            .webp_quality(cli.webp_quality)
            .debug(cli.debug)
            .intra_threads(if cli.intra_threads > 0 {
                cli.intra_threads
            } else {
                0
            })
            .inter_threads(if cli.inter_threads > 0 {
                cli.inter_threads
            } else {
                0
            })
            .color_management(ColorManagementConfig {
                preserve_color_profile: cli.preserve_color_profile && !cli.no_preserve_color_profile,
                force_srgb_output: cli.force_srgb,
                fallback_to_srgb: true,
                embed_profile_in_output: cli.embed_profile && !cli.no_embed_profile,
            })
            .build()
            .context("Invalid configuration")?;

        Ok(config)
    }

    /// Validate CLI arguments for consistency
    pub(crate) fn validate_cli(cli: &Cli) -> Result<()> {
        // Validate execution provider format
        ExecutionProviderManager::parse_provider_string(&cli.execution_provider)
            .context("Invalid execution provider format")?;

        // Validate background color format
        ColorParser::parse_hex(&cli.background_color)
            .context("Invalid background color format")?;

        // Validate quality settings
        if cli.jpeg_quality > 100 {
            anyhow::bail!("JPEG quality must be between 0 and 100");
        }
        if cli.webp_quality > 100 {
            anyhow::bail!("WebP quality must be between 0 and 100");
        }

        // Validate model specification if provided
        if let Some(model_arg) = &cli.model {
            let model_spec = ModelSpecParser::parse(model_arg);
            ModelSpecParser::validate(&model_spec)
                .context("Invalid model specification")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, CliOutputFormat};
    use bg_remove_core::{
        processor::BackendType,
        config::ExecutionProvider,
    };

    fn create_test_cli() -> Cli {
        Cli {
            input: vec!["test.jpg".to_string()],
            output: None,
            format: CliOutputFormat::Png,
            execution_provider: "onnx:auto".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            background_color: "#ffffff".to_string(),
            intra_threads: 0,
            inter_threads: 0,
            threads: 0,
            debug: false,
            verbose: 0,
            recursive: false,
            pattern: None,
            show_providers: false,
            model: None,
            variant: None,
            preserve_color_profile: true,
            no_preserve_color_profile: false,
            force_srgb: false,
            embed_profile: true,
            no_embed_profile: false,
        }
    }

    #[test]
    fn test_cli_config_conversion() {
        let cli = create_test_cli();
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

        // Reset and test invalid color
        cli.execution_provider = "onnx:auto".to_string();
        cli.background_color = "invalid".to_string();
        assert!(CliConfigBuilder::validate_cli(&cli).is_err());

        // Reset and test invalid quality
        cli.background_color = "#ffffff".to_string();
        cli.jpeg_quality = 150;
        assert!(CliConfigBuilder::validate_cli(&cli).is_err());
    }
}