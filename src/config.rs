//! Configuration types for background removal operations

use serde::{Deserialize, Serialize};

/// Execution provider options for ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionProvider {
    /// Auto-detect best available provider (CUDA > `CoreML` > CPU)
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
    /// JPEG (no transparency, premultiplied RGB output)
    Jpeg,
    /// WebP with alpha channel transparency
    WebP,
    /// TIFF with alpha channel transparency and lossless compression
    Tiff,
    /// Raw RGBA8 pixel data (4 bytes per pixel)
    Rgba8,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Png
    }
}

/// Configuration for background removal operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemovalConfig {
    /// Execution provider for ONNX Runtime
    pub execution_provider: ExecutionProvider,

    /// Output format
    pub output_format: OutputFormat,

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

    /// Preserve ICC color profiles from input images (default: true)
    pub preserve_color_profiles: bool,
}

impl Default for RemovalConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            output_format: OutputFormat::default(),
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,              // Auto-detect optimal intra-op threads
            inter_threads: 0,              // Auto-detect optimal inter-op threads
            preserve_color_profiles: true, // Default: preserve color profiles
        }
    }
}

impl RemovalConfig {
    /// Create a new configuration builder for fluent API construction
    ///
    /// The builder pattern allows for easy and readable configuration setup
    /// with method chaining and validation at build time.
    ///
    /// # Examples
    ///
    /// ## Basic configuration
    /// ```rust
    /// use imgly_bgremove::RemovalConfig;
    ///
    /// let config = RemovalConfig::builder()
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// ## Advanced configuration
    /// ```rust
    /// use imgly_bgremove::{RemovalConfig, ExecutionProvider, OutputFormat};
    ///
    /// let config = RemovalConfig::builder()
    ///     .execution_provider(ExecutionProvider::CoreMl)
    ///     .output_format(OutputFormat::WebP)
    ///     .webp_quality(95)
    ///     .debug(true)
    ///     .num_threads(8)
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn builder() -> RemovalConfigBuilder {
        RemovalConfigBuilder::default()
    }

    /// Validate all configuration parameters
    ///
    /// Ensures that quality values are within valid ranges and other
    /// configuration parameters are logically consistent.
    ///
    /// # Validation Rules
    ///
    /// - JPEG quality: 0-100 (inclusive)
    /// - WebP quality: 0-100 (inclusive)
    /// - Thread counts: Any non-negative value (0 = auto-detect)
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid, `Err(BgRemovalError)` with
    /// descriptive message if validation fails.
    ///
    /// # Errors
    /// - Invalid JPEG quality value (must be 0-100)
    /// - Invalid WebP quality value (must be 0-100)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use imgly_bgremove::RemovalConfig;
    ///
    /// let mut config = RemovalConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// config.jpeg_quality = 150; // Invalid
    /// assert!(config.validate().is_err());
    /// ```
    pub fn validate(&self) -> crate::Result<()> {
        if self.jpeg_quality > 100 {
            return Err(crate::error::BgRemovalError::config_value_error(
                "JPEG quality",
                self.jpeg_quality,
                "0-100",
                Some(90),
            ));
        }

        if self.webp_quality > 100 {
            return Err(crate::error::BgRemovalError::config_value_error(
                "WebP quality",
                self.webp_quality,
                "0-100",
                Some(85),
            ));
        }

        Ok(())
    }
}

/// Builder for `RemovalConfig`
#[derive(Debug, Default)]
pub struct RemovalConfigBuilder {
    config: RemovalConfig,
}

impl RemovalConfigBuilder {
    /// Set execution provider
    #[must_use]
    pub fn execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.config.execution_provider = provider;
        self
    }

    /// Set output format
    #[must_use]
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Set JPEG quality
    #[must_use]
    pub fn jpeg_quality(mut self, quality: u8) -> Self {
        self.config.jpeg_quality = quality.min(100);
        self
    }

    /// Set WebP quality
    #[must_use]
    pub fn webp_quality(mut self, quality: u8) -> Self {
        self.config.webp_quality = quality.min(100);
        self
    }

    /// Enable debug mode
    #[must_use]
    pub fn debug(mut self, debug: bool) -> Self {
        self.config.debug = debug;
        self
    }

    /// Set number of intra-op threads
    #[must_use]
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self
    }

    /// Set number of inter-op threads
    #[must_use]
    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.config.inter_threads = threads;
        self
    }

    /// Set both intra and inter threads with optimal defaults (convenience method)
    ///
    /// This method automatically sets both intra-op and inter-op thread counts
    /// with optimal ratios for background removal workloads.
    ///
    /// # Arguments
    /// * `threads` - Total thread count (0 = auto-detect optimal values)
    ///
    /// # Thread Allocation
    /// - **Intra-op threads**: Set to `threads` (within operations like matrix multiplication)
    /// - **Inter-op threads**: Set to `threads/2` (between operations, minimum 1)
    /// - **threads = 0**: Both values set to 0 for auto-detection
    ///
    /// # Performance Guidelines
    /// - **CPU cores**: Generally use physical core count (not hyperthreads)
    /// - **`ISNet` models**: Benefits from 4-8 threads
    /// - **`BiRefNet` models**: Benefits from 8-16 threads
    /// - **Memory bound**: More threads may not help beyond 8-12
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::RemovalConfig;
    ///
    /// // Auto-detect optimal thread count
    /// let config = RemovalConfig::builder()
    ///     .num_threads(0)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Use 8 threads total (8 intra, 4 inter)
    /// let config = RemovalConfig::builder()
    ///     .num_threads(8)
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self.config.inter_threads = if threads > 0 { (threads / 2).max(1) } else { 0 };
        self
    }

    /// Enable or disable ICC color profile preservation
    #[must_use]
    pub fn preserve_color_profiles(mut self, preserve: bool) -> Self {
        self.config.preserve_color_profiles = preserve;
        self
    }

    /// Build and validate the configuration
    ///
    /// Constructs the final `RemovalConfig` and runs validation to ensure
    /// all parameters are within acceptable ranges.
    ///
    /// # Returns
    /// `Ok(RemovalConfig)` if all parameters are valid, otherwise returns
    /// `Err(BgRemovalError)` with a detailed validation error message.
    ///
    /// # Validation Performed
    /// - JPEG and WebP quality values are 0-100
    /// - Thread counts are non-negative
    /// - All configuration combinations are logically consistent
    ///
    /// # Errors
    /// - Invalid JPEG quality value (must be 0-100)
    /// - Invalid WebP quality value (must be 0-100)
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::{RemovalConfig, ExecutionProvider};
    ///
    /// // Valid configuration
    /// let config = RemovalConfig::builder()
    ///     .execution_provider(ExecutionProvider::Auto)
    ///     .jpeg_quality(90)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Values > 100 are automatically clamped to 100
    /// let result = RemovalConfig::builder()
    ///     .jpeg_quality(150)  // Clamped to 100
    ///     .build();
    /// assert!(result.is_ok());
    /// assert_eq!(result.unwrap().jpeg_quality, 100);
    /// ```
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
            .jpeg_quality(95)
            .debug(true)
            .build()
            .unwrap();

        assert_eq!(config.output_format, OutputFormat::Jpeg);
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
}
