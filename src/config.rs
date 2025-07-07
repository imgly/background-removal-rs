//! Configuration types for background removal operations

use crate::models::ModelSpec;
use image::ImageFormat;
use serde::{Deserialize, Serialize};

#[cfg(feature = "video-support")]
use crate::backends::video::{VideoCodec, VideoEncodingConfig, QualityPreset};

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

impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::CoreMl => write!(f, "coreml"),
        }
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

/// Video processing configuration
#[cfg(feature = "video-support")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoProcessingConfig {
    /// Video encoding configuration
    pub encoding: VideoEncodingConfig,
    
    /// Number of frames to process in each batch
    pub batch_size: usize,
    
    /// Whether to preserve the original audio track
    pub preserve_audio: bool,
    
    /// Frame rate override (None = use original frame rate)
    pub fps_override: Option<f64>,
    
    /// Whether to enable parallel frame processing
    pub parallel_processing: bool,
    
    /// Maximum number of parallel workers (0 = auto-detect)
    pub max_workers: usize,
    
    /// Temporary directory for frame processing (None = system temp)
    pub temp_dir: Option<std::path::PathBuf>,
}

#[cfg(feature = "video-support")]
impl Default for VideoProcessingConfig {
    fn default() -> Self {
        Self {
            encoding: VideoEncodingConfig::default(),
            batch_size: 8, // Process 8 frames at a time
            preserve_audio: true,
            fps_override: None,
            parallel_processing: true,
            max_workers: 0, // Auto-detect
            temp_dir: None, // Use system temp
        }
    }
}

#[cfg(feature = "video-support")]
impl VideoProcessingConfig {
    /// Create new video config with specific codec
    pub fn new(codec: VideoCodec) -> Self {
        Self {
            encoding: VideoEncodingConfig::new(codec),
            ..Default::default()
        }
    }
    
    /// Set video codec
    pub fn with_codec(mut self, codec: VideoCodec) -> Self {
        self.encoding.codec = codec;
        self
    }
    
    /// Set video quality
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.encoding.quality = quality;
        self
    }
    
    /// Set quality preset
    pub fn with_preset(mut self, preset: QualityPreset) -> Self {
        self.encoding.preset = preset;
        self
    }
    
    /// Set batch size for frame processing
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1); // Ensure at least 1
        self
    }
    
    /// Enable or disable audio preservation
    pub fn with_audio_preservation(mut self, preserve: bool) -> Self {
        self.preserve_audio = preserve;
        self
    }
    
    /// Override frame rate
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps_override = Some(fps);
        self
    }
    
    /// Enable or disable parallel processing
    pub fn with_parallel_processing(mut self, enabled: bool) -> Self {
        self.parallel_processing = enabled;
        self
    }
    
    /// Set maximum number of workers
    pub fn with_max_workers(mut self, workers: usize) -> Self {
        self.max_workers = workers;
        self
    }
    
    /// Set temporary directory
    pub fn with_temp_dir<P: Into<std::path::PathBuf>>(mut self, dir: P) -> Self {
        self.temp_dir = Some(dir.into());
        self
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

    /// Disable all caches during processing (default: false)
    pub disable_cache: bool,

    /// Model specification including source and variant
    pub model_spec: ModelSpec,

    /// Optional format hint for reader-based processing
    #[serde(skip)]
    pub format_hint: Option<ImageFormat>,

    /// Video processing configuration (only used when processing videos)
    #[cfg(feature = "video-support")]
    pub video_config: Option<VideoProcessingConfig>,
}

impl Default for RemovalConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            output_format: OutputFormat::default(),
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,                 // Auto-detect optimal intra-op threads
            inter_threads: 0,                 // Auto-detect optimal inter-op threads
            preserve_color_profiles: true,    // Default: preserve color profiles
            disable_cache: false,             // Default: enable caches
            model_spec: ModelSpec::default(), // Default: use first available cached model
            format_hint: None,                // Default: auto-detect format
            #[cfg(feature = "video-support")]
            video_config: None,               // Default: no video processing config
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

    /// Enable or disable all caches during processing
    #[must_use]
    pub fn disable_cache(mut self, disable: bool) -> Self {
        self.config.disable_cache = disable;
        self
    }

    /// Set the model specification
    #[must_use]
    pub fn model_spec(mut self, model_spec: ModelSpec) -> Self {
        self.config.model_spec = model_spec;
        self
    }

    /// Set the format hint for reader-based processing
    #[must_use]
    pub fn format_hint(mut self, format: Option<ImageFormat>) -> Self {
        self.config.format_hint = format;
        self
    }

    /// Set video processing configuration (when video support is enabled)
    #[cfg(feature = "video-support")]
    #[must_use]
    pub fn video_config(mut self, config: VideoProcessingConfig) -> Self {
        self.config.video_config = Some(config);
        self
    }

    /// Set video codec (convenience method when video support is enabled)
    #[cfg(feature = "video-support")]
    #[must_use]
    pub fn video_codec(mut self, codec: VideoCodec) -> Self {
        let video_config = self.config.video_config
            .unwrap_or_default()
            .with_codec(codec);
        self.config.video_config = Some(video_config);
        self
    }

    /// Set video quality (convenience method when video support is enabled)
    #[cfg(feature = "video-support")]
    #[must_use]
    pub fn video_quality(mut self, quality: u8) -> Self {
        let video_config = self.config.video_config
            .unwrap_or_default()
            .with_quality(quality);
        self.config.video_config = Some(video_config);
        self
    }

    /// Set video batch size (convenience method when video support is enabled)
    #[cfg(feature = "video-support")]
    #[must_use]
    pub fn video_batch_size(mut self, batch_size: usize) -> Self {
        let video_config = self.config.video_config
            .unwrap_or_default()
            .with_batch_size(batch_size);
        self.config.video_config = Some(video_config);
        self
    }

    /// Enable or disable audio preservation (convenience method when video support is enabled)
    #[cfg(feature = "video-support")]
    #[must_use]
    pub fn preserve_audio(mut self, preserve: bool) -> Self {
        let video_config = self.config.video_config
            .unwrap_or_default()
            .with_audio_preservation(preserve);
        self.config.video_config = Some(video_config);
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

    #[test]
    fn test_disable_cache_config() {
        // Test default (cache enabled)
        let config = RemovalConfig::default();
        assert!(!config.disable_cache);

        // Test disable cache via builder
        let config = RemovalConfig::builder()
            .disable_cache(true)
            .build()
            .unwrap();
        assert!(config.disable_cache);

        // Test enable cache explicitly via builder
        let config = RemovalConfig::builder()
            .disable_cache(false)
            .build()
            .unwrap();
        assert!(!config.disable_cache);
    }

    #[test]
    fn test_execution_provider_enum() {
        // Test default implementation
        assert_eq!(ExecutionProvider::default(), ExecutionProvider::Auto);

        // Test display formatting
        assert_eq!(format!("{}", ExecutionProvider::Auto), "auto");
        assert_eq!(format!("{}", ExecutionProvider::Cpu), "cpu");
        assert_eq!(format!("{}", ExecutionProvider::Cuda), "cuda");
        assert_eq!(format!("{}", ExecutionProvider::CoreMl), "coreml");

        // Test debug formatting
        let debug_str = format!("{:?}", ExecutionProvider::Auto);
        assert!(debug_str.contains("Auto"));

        // Test equality and ordering
        assert_eq!(ExecutionProvider::Auto, ExecutionProvider::Auto);
        assert_ne!(ExecutionProvider::Auto, ExecutionProvider::Cpu);

        // Test copy and clone
        let provider1 = ExecutionProvider::Cuda;
        let provider2 = provider1; // Copy
        assert_eq!(provider1, provider2);

        let provider3 = provider1.clone(); // Clone
        assert_eq!(provider1, provider3);
    }

    #[test]
    fn test_output_format_enum() {
        // Test default implementation
        assert_eq!(OutputFormat::default(), OutputFormat::Png);

        // Test all variants
        let formats = vec![
            OutputFormat::Png,
            OutputFormat::Jpeg,
            OutputFormat::WebP,
            OutputFormat::Tiff,
            OutputFormat::Rgba8,
        ];

        for format in formats {
            // Test debug formatting
            let debug_str = format!("{:?}", format);
            assert!(!debug_str.is_empty());

            // Test equality
            assert_eq!(format, format);

            // Test copy/clone
            let format_copy = format;
            assert_eq!(format, format_copy);

            let format_clone = format.clone();
            assert_eq!(format, format_clone);
        }
    }

    #[test]
    fn test_removal_config_builder_thread_methods() {
        // Test intra_threads method
        let config = RemovalConfig::builder().intra_threads(4).build().unwrap();
        assert_eq!(config.intra_threads, 4);

        // Test inter_threads method
        let config = RemovalConfig::builder().inter_threads(2).build().unwrap();
        assert_eq!(config.inter_threads, 2);

        // Test num_threads method with optimal ratios
        let config = RemovalConfig::builder().num_threads(8).build().unwrap();
        assert_eq!(config.intra_threads, 8);
        assert_eq!(config.inter_threads, 4); // 8/2 = 4

        // Test num_threads with odd number
        let config = RemovalConfig::builder().num_threads(7).build().unwrap();
        assert_eq!(config.intra_threads, 7);
        assert_eq!(config.inter_threads, 3); // 7/2 = 3 (rounded down)

        // Test num_threads with 1 (minimum inter_threads)
        let config = RemovalConfig::builder().num_threads(1).build().unwrap();
        assert_eq!(config.intra_threads, 1);
        assert_eq!(config.inter_threads, 1); // max(1/2, 1) = 1

        // Test num_threads with 0 (auto-detect)
        let config = RemovalConfig::builder().num_threads(0).build().unwrap();
        assert_eq!(config.intra_threads, 0);
        assert_eq!(config.inter_threads, 0);
    }

    #[test]
    fn test_removal_config_builder_quality_clamping() {
        // Test JPEG quality clamping
        let config = RemovalConfig::builder()
            .jpeg_quality(150) // Should be clamped to 100
            .build()
            .unwrap();
        assert_eq!(config.jpeg_quality, 100);

        let config = RemovalConfig::builder()
            .jpeg_quality(50) // Valid value
            .build()
            .unwrap();
        assert_eq!(config.jpeg_quality, 50);

        // Test WebP quality clamping
        let config = RemovalConfig::builder()
            .webp_quality(200) // Should be clamped to 100
            .build()
            .unwrap();
        assert_eq!(config.webp_quality, 100);

        let config = RemovalConfig::builder()
            .webp_quality(75) // Valid value
            .build()
            .unwrap();
        assert_eq!(config.webp_quality, 75);
    }

    #[test]
    fn test_removal_config_format_hint() {
        use image::ImageFormat;

        // Test format hint setting
        let config = RemovalConfig::builder()
            .format_hint(Some(ImageFormat::Png))
            .build()
            .unwrap();
        assert_eq!(config.format_hint, Some(ImageFormat::Png));

        let config = RemovalConfig::builder()
            .format_hint(Some(ImageFormat::Jpeg))
            .build()
            .unwrap();
        assert_eq!(config.format_hint, Some(ImageFormat::Jpeg));

        let config = RemovalConfig::builder().format_hint(None).build().unwrap();
        assert_eq!(config.format_hint, None);

        // Test default is None
        let config = RemovalConfig::builder().build().unwrap();
        assert_eq!(config.format_hint, None);
    }

    #[test]
    fn test_removal_config_model_spec() {
        use crate::models::{ModelSource, ModelSpec};

        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: Some("fp16".to_string()),
        };

        let config = RemovalConfig::builder()
            .model_spec(model_spec.clone())
            .build()
            .unwrap();

        assert_eq!(config.model_spec, model_spec);
    }

    #[test]
    fn test_removal_config_builder_chaining() {
        use crate::models::{ModelSource, ModelSpec};

        // Test method chaining
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: Some("fp32".to_string()),
        };

        let config = RemovalConfig::builder()
            .execution_provider(ExecutionProvider::CoreMl)
            .output_format(OutputFormat::WebP)
            .jpeg_quality(85)
            .webp_quality(95)
            .debug(true)
            .intra_threads(6)
            .inter_threads(3)
            .preserve_color_profiles(false)
            .disable_cache(true)
            .model_spec(model_spec.clone())
            .build()
            .unwrap();

        assert_eq!(config.execution_provider, ExecutionProvider::CoreMl);
        assert_eq!(config.output_format, OutputFormat::WebP);
        assert_eq!(config.jpeg_quality, 85);
        assert_eq!(config.webp_quality, 95);
        assert!(config.debug);
        assert_eq!(config.intra_threads, 6);
        assert_eq!(config.inter_threads, 3);
        assert!(!config.preserve_color_profiles);
        assert!(config.disable_cache);
        assert_eq!(config.model_spec, model_spec);
    }

    #[test]
    fn test_removal_config_validation_comprehensive() {
        // Test valid config passes validation
        let config = RemovalConfig::default();
        assert!(config.validate().is_ok());

        // Test JPEG quality validation
        let mut config = RemovalConfig::default();
        config.jpeg_quality = 101;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("JPEG quality"));

        // Test WebP quality validation
        config.jpeg_quality = 90; // Reset to valid
        config.webp_quality = 101;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("WebP quality"));

        // Test edge cases (0 and 100)
        config.webp_quality = 85; // Reset to valid
        config.jpeg_quality = 0;
        config.webp_quality = 0;
        assert!(config.validate().is_ok());

        config.jpeg_quality = 100;
        config.webp_quality = 100;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_removal_config_builder_validation_error() {
        // Test that builder validation catches invalid values
        // Note: Quality values are clamped in the builder, so they won't cause build() to fail
        // But we can test other validation scenarios

        let config = RemovalConfig::builder().build();
        assert!(config.is_ok()); // Default config should be valid

        // Test that the builder properly validates the final config
        let builder = RemovalConfig::builder();
        let config = builder.build().unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_removal_config_serde_attributes() {
        use crate::models::{ModelSource, ModelSpec};

        // Test that format_hint is properly skipped in serialization
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: None,
        };

        let config = RemovalConfig {
            execution_provider: ExecutionProvider::Auto,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: true,
            disable_cache: false,
            model_spec,
            format_hint: Some(ImageFormat::Png),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();

        // Verify format_hint is not in the JSON (due to #[serde(skip)])
        assert!(!json.contains("format_hint"));

        // Verify other fields are present
        assert!(json.contains("execution_provider"));
        assert!(json.contains("output_format"));
        assert!(json.contains("jpeg_quality"));

        // Test deserialization
        let deserialized: RemovalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.execution_provider, config.execution_provider);
        assert_eq!(deserialized.output_format, config.output_format);
        assert_eq!(deserialized.jpeg_quality, config.jpeg_quality);
        assert_eq!(deserialized.format_hint, None); // Should be None after deserialization
    }

    #[test]
    fn test_removal_config_debug_formatting() {
        let config = RemovalConfig::default();
        let debug_str = format!("{:?}", config);

        // Verify debug string contains key fields
        assert!(debug_str.contains("RemovalConfig"));
        assert!(debug_str.contains("execution_provider"));
        assert!(debug_str.contains("output_format"));
        assert!(debug_str.contains("jpeg_quality"));
        assert!(debug_str.contains("webp_quality"));
    }

    #[test]
    fn test_removal_config_clone() {
        let config1 = RemovalConfig::default();
        let config2 = config1.clone();

        assert_eq!(config1, config2);
        assert_eq!(config1.execution_provider, config2.execution_provider);
        assert_eq!(config1.output_format, config2.output_format);
        assert_eq!(config1.jpeg_quality, config2.jpeg_quality);
        assert_eq!(config1.webp_quality, config2.webp_quality);
    }

    #[test]
    fn test_removal_config_partial_eq() {
        let config1 = RemovalConfig::default();
        let mut config2 = RemovalConfig::default();

        assert_eq!(config1, config2);

        config2.jpeg_quality = 95;
        assert_ne!(config1, config2);

        config2.jpeg_quality = config1.jpeg_quality;
        assert_eq!(config1, config2);

        config2.debug = !config1.debug;
        assert_ne!(config1, config2);
    }

    #[test]
    fn test_removal_config_builder_default() {
        let builder = RemovalConfigBuilder::default();
        let config = builder.build().unwrap();

        // Should match RemovalConfig::default()
        let default_config = RemovalConfig::default();
        assert_eq!(config, default_config);
    }

    #[test]
    fn test_execution_provider_serde() {
        // Test serialization/deserialization of ExecutionProvider
        let providers = vec![
            ExecutionProvider::Auto,
            ExecutionProvider::Cpu,
            ExecutionProvider::Cuda,
            ExecutionProvider::CoreMl,
        ];

        for provider in providers {
            let json = serde_json::to_string(&provider).unwrap();
            let deserialized: ExecutionProvider = serde_json::from_str(&json).unwrap();
            assert_eq!(provider, deserialized);
        }
    }

    #[test]
    fn test_output_format_serde() {
        // Test serialization/deserialization of OutputFormat
        let formats = vec![
            OutputFormat::Png,
            OutputFormat::Jpeg,
            OutputFormat::WebP,
            OutputFormat::Tiff,
            OutputFormat::Rgba8,
        ];

        for format in formats {
            let json = serde_json::to_string(&format).unwrap();
            let deserialized: OutputFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(format, deserialized);
        }
    }
}
