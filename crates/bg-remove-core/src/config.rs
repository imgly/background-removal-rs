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
    /// Create a new background color with RGB values
    ///
    /// # Arguments
    /// * `r` - Red component (0-255)
    /// * `g` - Green component (0-255) 
    /// * `b` - Blue component (0-255)
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::BackgroundColor;
    /// let purple = BackgroundColor::new(128, 0, 128);
    /// let orange = BackgroundColor::new(255, 165, 0);
    /// ```
    #[must_use] pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Create a white background color (255, 255, 255)
    ///
    /// Commonly used for product photography and clean presentations.
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::BackgroundColor;
    /// let white = BackgroundColor::white();
    /// assert_eq!(white.r, 255);
    /// assert_eq!(white.g, 255);
    /// assert_eq!(white.b, 255);
    /// ```
    #[must_use] pub fn white() -> Self {
        Self::new(255, 255, 255)
    }

    /// Create a black background color (0, 0, 0)
    ///
    /// Useful for dramatic effects or when the foreground is light-colored.
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::BackgroundColor;
    /// let black = BackgroundColor::black();
    /// assert_eq!(black.r, 0);
    /// assert_eq!(black.g, 0);
    /// assert_eq!(black.b, 0);
    /// ```
    #[must_use] pub fn black() -> Self {
        Self::new(0, 0, 0)
    }

    /// Create a transparent background placeholder
    ///
    /// Note: This returns black (0,0,0) but the color is ignored for formats
    /// that support transparency (PNG, WebP). Only used for JPEG output.
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::{BackgroundColor, OutputFormat};
    /// let transparent = BackgroundColor::transparent();
    /// // Used with PNG - color ignored, true transparency
    /// // Used with JPEG - renders as black background
    /// ```
    #[must_use] pub fn transparent() -> Self {
        Self::new(0, 0, 0) // Will be ignored for transparent formats
    }
}

/// Color management configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorManagementConfig {
    /// Preserve ICC color profiles from input images
    pub preserve_color_profile: bool,
    
    /// Force sRGB output regardless of input profile
    pub force_srgb_output: bool,
    
    /// Fallback to sRGB when color space detection fails
    pub fallback_to_srgb: bool,
    
    /// Embed color profile in output (when supported by format)
    pub embed_profile_in_output: bool,
}

impl Default for ColorManagementConfig {
    fn default() -> Self {
        Self {
            preserve_color_profile: true,
            force_srgb_output: false,
            fallback_to_srgb: true,
            embed_profile_in_output: true,
        }
    }
}

impl ColorManagementConfig {
    /// Create a configuration that preserves color profiles
    pub fn preserve() -> Self {
        Self {
            preserve_color_profile: true,
            force_srgb_output: false,
            fallback_to_srgb: true,
            embed_profile_in_output: true,
        }
    }
    
    /// Create a configuration that ignores color profiles (legacy behavior)
    pub fn ignore() -> Self {
        Self {
            preserve_color_profile: false,
            force_srgb_output: false,
            fallback_to_srgb: true,
            embed_profile_in_output: false,
        }
    }
    
    /// Create a configuration that forces sRGB output
    pub fn force_srgb() -> Self {
        Self {
            preserve_color_profile: true,
            force_srgb_output: true,
            fallback_to_srgb: true,
            embed_profile_in_output: true,
        }
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

    /// Color management configuration
    pub color_management: ColorManagementConfig,
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
            color_management: ColorManagementConfig::default(),
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
    /// use bg_remove_core::RemovalConfig;
    ///
    /// let config = RemovalConfig::builder()
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// ## Advanced configuration
    /// ```rust
    /// use bg_remove_core::{RemovalConfig, ExecutionProvider, OutputFormat, BackgroundColor};
    ///
    /// let config = RemovalConfig::builder()
    ///     .execution_provider(ExecutionProvider::CoreMl)
    ///     .output_format(OutputFormat::WebP)
    ///     .webp_quality(95)
    ///     .background_color(BackgroundColor::new(240, 248, 255))
    ///     .debug(true)
    ///     .num_threads(8)
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use] pub fn builder() -> RemovalConfigBuilder {
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
    /// # Examples
    ///
    /// ```rust
    /// use bg_remove_core::RemovalConfig;
    ///
    /// let mut config = RemovalConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// config.jpeg_quality = 150; // Invalid
    /// assert!(config.validate().is_err());
    /// ```
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

/// Builder for `RemovalConfig`
#[derive(Debug, Default)]
pub struct RemovalConfigBuilder {
    config: RemovalConfig,
}

impl RemovalConfigBuilder {
    /// Set execution provider
    #[must_use] pub fn execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.config.execution_provider = provider;
        self
    }

    /// Set output format
    #[must_use] pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Set background color
    #[must_use] pub fn background_color(mut self, color: BackgroundColor) -> Self {
        self.config.background_color = color;
        self
    }

    /// Set JPEG quality
    #[must_use] pub fn jpeg_quality(mut self, quality: u8) -> Self {
        self.config.jpeg_quality = quality.min(100);
        self
    }

    /// Set WebP quality
    #[must_use] pub fn webp_quality(mut self, quality: u8) -> Self {
        self.config.webp_quality = quality.min(100);
        self
    }

    /// Enable debug mode
    #[must_use] pub fn debug(mut self, debug: bool) -> Self {
        self.config.debug = debug;
        self
    }

    /// Set number of intra-op threads
    #[must_use] pub fn intra_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self
    }

    /// Set number of inter-op threads
    #[must_use] pub fn inter_threads(mut self, threads: usize) -> Self {
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
    /// - **ISNet models**: Benefits from 4-8 threads
    /// - **BiRefNet models**: Benefits from 8-16 threads
    /// - **Memory bound**: More threads may not help beyond 8-12
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::RemovalConfig;
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
    #[must_use] pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self.config.inter_threads = if threads > 0 { (threads / 2).max(1) } else { 0 };
        self
    }

    /// Set color management configuration
    #[must_use] pub fn color_management(mut self, color_management: ColorManagementConfig) -> Self {
        self.config.color_management = color_management;
        self
    }

    /// Enable or disable ICC color profile preservation
    #[must_use] pub fn preserve_color_profile(mut self, preserve: bool) -> Self {
        self.config.color_management.preserve_color_profile = preserve;
        self
    }

    /// Force sRGB output regardless of input color profile
    #[must_use] pub fn force_srgb_output(mut self, force: bool) -> Self {
        self.config.color_management.force_srgb_output = force;
        self
    }

    /// Enable or disable embedding ICC profiles in output images
    #[must_use] pub fn embed_profile_in_output(mut self, embed: bool) -> Self {
        self.config.color_management.embed_profile_in_output = embed;
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
    /// # Examples
    /// ```rust
    /// use bg_remove_core::{RemovalConfig, ExecutionProvider};
    ///
    /// // Valid configuration
    /// let config = RemovalConfig::builder()
    ///     .execution_provider(ExecutionProvider::Auto)
    ///     .jpeg_quality(90)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Invalid configuration (quality > 100)
    /// let result = RemovalConfig::builder()
    ///     .jpeg_quality(150)
    ///     .build();
    /// assert!(result.is_err());
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
