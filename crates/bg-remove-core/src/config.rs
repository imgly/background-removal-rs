//! Configuration types for background removal operations

use serde::{Deserialize, Serialize};

/// Execution provider options for ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionProvider {
    /// Auto-detect best available provider (CUDA > CoreML > CPU)
    Auto,
    /// CPU execution (always available)
    Cpu,
    /// NVIDIA CUDA GPU acceleration
    Cuda,
    /// Apple Silicon GPU acceleration (Metal Performance Shaders)
    CoreMl,
}

impl Default for ExecutionProvider {
    fn default() -> Self {
        // Default to auto-detection for best performance
        Self::Auto
    }
}

/// Output image format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// PNG with alpha channel transparency
    Png,
    /// JPEG with configurable background color (no transparency)
    Jpeg,
    /// WebP with alpha channel transparency
    WebP,
    /// Raw RGBA8 pixel data (4 bytes per pixel)
    Rgba8,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Png
    }
}

/// Background color for formats that don't support transparency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Default for BackgroundColor {
    fn default() -> Self {
        // Default to white background
        Self {
            r: 255,
            g: 255,
            b: 255,
        }
    }
}

impl BackgroundColor {
    /// Create a new background color
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// White background
    pub fn white() -> Self {
        Self::new(255, 255, 255)
    }

    /// Black background
    pub fn black() -> Self {
        Self::new(0, 0, 0)
    }

    /// Transparent (for formats supporting alpha)
    pub fn transparent() -> Self {
        Self::new(0, 0, 0) // Will be ignored for transparent formats
    }
}

/// Configuration for background removal operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemovalConfig {
    /// Execution provider for ONNX Runtime
    pub execution_provider: ExecutionProvider,

    /// Output format
    pub output_format: OutputFormat,

    /// Background color for non-transparent formats
    pub background_color: BackgroundColor,

    /// JPEG quality (0-100, only used for JPEG output)
    pub jpeg_quality: u8,

    /// WebP quality (0-100, only used for WebP output)
    pub webp_quality: u8,

    /// Enable debug mode (additional logging and validation)
    pub debug: bool,

    /// Number of intra-op threads for inference (0 = auto)
    pub intra_threads: usize,

    /// Number of inter-op threads for inference (0 = auto)
    pub inter_threads: usize,
}

impl Default for RemovalConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            output_format: OutputFormat::default(),
            background_color: BackgroundColor::default(),
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0, // Auto-detect optimal intra-op threads
            inter_threads: 0, // Auto-detect optimal inter-op threads
        }
    }
}

impl RemovalConfig {
    /// Create a new configuration builder
    pub fn builder() -> RemovalConfigBuilder {
        RemovalConfigBuilder::default()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> crate::Result<()> {
        if self.jpeg_quality > 100 {
            return Err(crate::error::BgRemovalError::invalid_config(
                "JPEG quality must be between 0-100",
            ));
        }

        if self.webp_quality > 100 {
            return Err(crate::error::BgRemovalError::invalid_config(
                "WebP quality must be between 0-100",
            ));
        }

        Ok(())
    }
}

/// Builder for RemovalConfig
#[derive(Debug, Default)]
pub struct RemovalConfigBuilder {
    config: RemovalConfig,
}

impl RemovalConfigBuilder {
    /// Set execution provider
    pub fn execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.config.execution_provider = provider;
        self
    }

    /// Set output format
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Set background color
    pub fn background_color(mut self, color: BackgroundColor) -> Self {
        self.config.background_color = color;
        self
    }

    /// Set JPEG quality
    pub fn jpeg_quality(mut self, quality: u8) -> Self {
        self.config.jpeg_quality = quality.min(100);
        self
    }

    /// Set WebP quality
    pub fn webp_quality(mut self, quality: u8) -> Self {
        self.config.webp_quality = quality.min(100);
        self
    }

    /// Enable debug mode
    pub fn debug(mut self, debug: bool) -> Self {
        self.config.debug = debug;
        self
    }

    /// Set number of intra-op threads
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self
    }

    /// Set number of inter-op threads
    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.config.inter_threads = threads;
        self
    }

    /// Set both intra and inter threads (convenience method)
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self.config.inter_threads = if threads > 0 { (threads / 2).max(1) } else { 0 };
        self
    }

    /// Build the configuration
    pub fn build(self) -> crate::Result<RemovalConfig> {
        let config = self.config;
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RemovalConfig::default();
        assert_eq!(config.output_format, OutputFormat::Png);
        assert_eq!(config.jpeg_quality, 90);
        assert_eq!(config.webp_quality, 85);
        assert!(!config.debug);
    }

    #[test]
    fn test_config_builder() {
        let config = RemovalConfig::builder()
            .output_format(OutputFormat::Jpeg)
            .background_color(BackgroundColor::black())
            .jpeg_quality(95)
            .debug(true)
            .build()
            .unwrap();

        assert_eq!(config.output_format, OutputFormat::Jpeg);
        assert_eq!(config.background_color, BackgroundColor::black());
        assert_eq!(config.jpeg_quality, 95);
        assert!(config.debug);
    }

    #[test]
    fn test_config_validation() {
        let mut config = RemovalConfig::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid JPEG quality should fail
        config.jpeg_quality = 150;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_background_colors() {
        assert_eq!(
            BackgroundColor::white(),
            BackgroundColor::new(255, 255, 255)
        );
        assert_eq!(BackgroundColor::black(), BackgroundColor::new(0, 0, 0));
    }
}
