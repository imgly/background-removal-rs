//! Configuration conversion utilities for Web/WASM environments

use bg_remove_core::{
    processor::{BackendType, ProcessorConfig, ProcessorConfigBuilder},
    utils::ColorParser,
    config::{ColorManagementConfig, ExecutionProvider, OutputFormat},
    models::{get_available_embedded_models, ModelSource, ModelSpec},
};
use crate::WebRemovalConfig;

/// Convert Web configuration to unified ProcessorConfig
pub(crate) struct WebConfigBuilder;

impl WebConfigBuilder {
    /// Build ProcessorConfig from WebRemovalConfig
    pub(crate) fn from_web_config(web_config: &WebRemovalConfig, model_name: Option<String>) -> Result<ProcessorConfig, bg_remove_core::error::BgRemovalError> {
        // Determine model specification
        let model_spec = if let Some(model_name) = model_name {
            ModelSpec {
                source: ModelSource::Embedded(model_name),
                variant: None,
            }
        } else {
            // Use first available embedded model
            let available_embedded = get_available_embedded_models();
            if available_embedded.is_empty() {
                return Err(bg_remove_core::error::BgRemovalError::invalid_config(
                    "No model specified and no embedded models available. Build with embed-* features."
                ));
            }
            let default_model = &available_embedded[0];
            ModelSpec {
                source: ModelSource::Embedded(default_model.clone()),
                variant: None,
            }
        };

        // Parse background color using core utilities
        let background_color = ColorParser::parse_hex(&web_config.background_color)?;

        // Parse output format
        let output_format = match web_config.output_format.as_str() {
            "png" => OutputFormat::Png,
            "jpeg" | "jpg" => OutputFormat::Jpeg,
            "webp" => OutputFormat::WebP,
            "tiff" => OutputFormat::Tiff,
            "rgba8" => OutputFormat::Rgba8,
            _ => OutputFormat::Png, // Default fallback
        };

        // Build configuration using unified builder
        let config = ProcessorConfigBuilder::new()
            .model_spec(model_spec)
            .backend_type(BackendType::Tract) // WASM only supports Tract
            .execution_provider(ExecutionProvider::Cpu) // WASM only supports CPU
            .output_format(output_format)
            .background_color(background_color)
            .jpeg_quality(web_config.jpeg_quality)
            .webp_quality(web_config.webp_quality)
            .debug(web_config.debug)
            .intra_threads(web_config.intra_threads as usize)
            .inter_threads(web_config.inter_threads as usize)
            .color_management(ColorManagementConfig {
                preserve_color_profile: web_config.preserve_color_profile,
                force_srgb_output: web_config.force_srgb_output,
                fallback_to_srgb: web_config.fallback_to_srgb,
                embed_profile_in_output: web_config.embed_profile_in_output,
            })
            .build()?;

        Ok(config)
    }

    /// Validate Web configuration
    pub(crate) fn validate_web_config(web_config: &WebRemovalConfig) -> Result<(), bg_remove_core::error::BgRemovalError> {
        // Validate background color format using core utilities
        ColorParser::parse_hex(&web_config.background_color)?;

        // Validate quality settings
        if web_config.jpeg_quality > 100 {
            return Err(bg_remove_core::error::BgRemovalError::invalid_config("JPEG quality must be between 0 and 100"));
        }
        if web_config.webp_quality > 100 {
            return Err(bg_remove_core::error::BgRemovalError::invalid_config("WebP quality must be between 0 and 100"));
        }

        // Validate output format
        let valid_formats = ["png", "jpeg", "jpg", "webp", "tiff", "rgba8"];
        if !valid_formats.contains(&web_config.output_format.to_lowercase().as_str()) {
            return Err(bg_remove_core::error::BgRemovalError::invalid_config(
                &format!("Invalid output format: {}", web_config.output_format)
            ));
        }

        Ok(())
    }

    /// Convert ProcessorConfig back to WebRemovalConfig
    pub(crate) fn to_web_config(config: &ProcessorConfig) -> WebRemovalConfig {
        let background_hex = format!("#{:02x}{:02x}{:02x}", 
            config.background_color.r, 
            config.background_color.g, 
            config.background_color.b);
            
        let output_format = match config.output_format {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpeg",
            OutputFormat::WebP => "webp",
            OutputFormat::Tiff => "tiff",
            OutputFormat::Rgba8 => "rgba8",
        }.to_string();

        WebRemovalConfig {
            output_format,
            jpeg_quality: config.jpeg_quality,
            webp_quality: config.webp_quality,
            background_color: background_hex,
            debug: config.debug,
            intra_threads: config.intra_threads as u32,
            inter_threads: config.inter_threads as u32,
            preserve_color_profile: config.color_management.preserve_color_profile,
            force_srgb_output: config.color_management.force_srgb_output,
            fallback_to_srgb: config.color_management.fallback_to_srgb,
            embed_profile_in_output: config.color_management.embed_profile_in_output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_web_config() -> WebRemovalConfig {
        WebRemovalConfig {
            output_format: "png".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            background_color: "#ffffff".to_string(),
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profile: true,
            force_srgb_output: false,
            fallback_to_srgb: true,
            embed_profile_in_output: true,
        }
    }

    #[test]
    fn test_web_config_conversion() {
        let web_config = create_test_web_config();
        let config = WebConfigBuilder::from_web_config(&web_config, Some("test-model".to_string())).unwrap();
        
        assert_eq!(config.backend_type, BackendType::Tract);
        assert_eq!(config.execution_provider, ExecutionProvider::Cpu);
        assert_eq!(config.output_format, OutputFormat::Png);
        assert_eq!(config.jpeg_quality, 90);
        assert_eq!(config.webp_quality, 85);
        assert!(!config.debug);
    }

    #[test]
    fn test_web_config_validation() {
        let mut web_config = create_test_web_config();
        assert!(WebConfigBuilder::validate_web_config(&web_config).is_ok());

        // Test invalid color
        web_config.background_color = "invalid".to_string();
        assert!(WebConfigBuilder::validate_web_config(&web_config).is_err());

        // Reset and test invalid quality
        web_config.background_color = "#ffffff".to_string();
        web_config.jpeg_quality = 150;
        assert!(WebConfigBuilder::validate_web_config(&web_config).is_err());

        // Reset and test invalid format
        web_config.jpeg_quality = 90;
        web_config.output_format = "invalid".to_string();
        assert!(WebConfigBuilder::validate_web_config(&web_config).is_err());
    }
}