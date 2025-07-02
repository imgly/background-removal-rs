// All clippy issues have been addressed
#![doc = include_str!("../README.md")]

pub mod backends;
pub mod cache;
#[cfg(feature = "cli")]
pub mod cli;
pub mod color_profile;
pub mod config;
pub mod download;
pub mod error;
pub mod inference;
pub mod models;
pub mod processor;
pub mod services;
pub mod session_cache;
#[cfg(feature = "cli")]
pub mod tracing_config;
pub mod types;
pub mod utils;

// Internal imports for lib functions
use tokio::io::AsyncRead;

// Public API exports
pub use backends::*;
pub use cache::{format_size, CachedModelInfo, ModelCache};
pub use color_profile::{ProfileEmbedder, ProfileExtractor};
pub use config::{ExecutionProvider, OutputFormat, RemovalConfig};
pub use download::{parse_huggingface_url, validate_model_url, DownloadProgress, ModelDownloader};
pub use error::{BgRemovalError, Result};
pub use inference::InferenceBackend;
pub use models::{ModelManager, ModelSource, ModelSpec};
pub use processor::{
    BackendType, BackgroundRemovalProcessor, ProcessorConfig, ProcessorConfigBuilder,
};
pub use services::{
    ConsoleProgressReporter, ImageIOService, NoOpProgressReporter, OutputFormatHandler,
    ProcessingStage, ProgressReporter, ProgressTracker, ProgressUpdate,
};
pub use session_cache::{format_cache_size, SessionCache, SessionCacheEntry, SessionCacheStats};
pub use types::{ColorProfile, ColorSpace, RemovalResult, SegmentationMask};
pub use utils::{
    ConfigValidator, ExecutionProviderManager, ImagePreprocessor, ModelSpecParser, ModelValidator,
    NumericValidator, PathValidator, PreprocessingOptions, ProviderInfo, TensorValidator,
};

#[cfg(feature = "cli")]
pub use tracing_config::{
    events, init_cli_tracing, init_library_tracing, spans, TracingConfig, TracingFormat,
    TracingOutput,
};

/// Remove background from an image provided as bytes
///
/// This is a stream-based API that accepts image data as bytes, making it suitable
/// for web servers, memory-based processing, and scenarios where files aren't available.
///
/// # Arguments
///
/// * `image_bytes` - Raw image data as bytes (JPEG, PNG, WebP, BMP, TIFF)
/// * `config` - Configuration for the removal operation
/// * `model_spec` - Specification of which model to use
///
/// # Returns
///
/// A `RemovalResult` containing the processed image, mask, and metadata
///
/// # Examples
///
/// ## Web server usage
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_from_bytes, ModelSpec, ModelSource};
///
/// # fn example(upload_bytes: Vec<u8>) -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_bytes(&upload_bytes, &config)?;
/// let output_bytes = result.to_bytes(config.output_format, 90)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Memory-based processing
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_from_bytes, ModelSpec, ModelSource};
///
/// # fn example() -> anyhow::Result<()> {
/// let image_data = download_image_from_api()?;
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_bytes(&image_data, &config)?;
/// let png_bytes = result.to_bytes(imgly_bgremove::OutputFormat::Png, 100)?;
/// # Ok(())
/// # }
/// # fn download_image_from_api() -> anyhow::Result<Vec<u8>> { Ok(vec![]) }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - Image decoding failures
/// - Model loading errors
/// - Inference execution errors
pub fn remove_background_from_bytes(
    image_bytes: &[u8],
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    // Load image from bytes using the image crate
    let image = image::load_from_memory(image_bytes).map_err(|e| {
        BgRemovalError::processing(format!("Failed to decode image from bytes: {e}"))
    })?;

    // Use the existing image processing function with model_spec from config
    remove_background_from_image(&image, config)
}

/// Remove background from a `DynamicImage` directly
///
/// This is the most flexible API for in-memory image processing. It accepts
/// a pre-loaded `DynamicImage` and processes it without any file I/O.
///
/// # Arguments
///
/// * `image` - A `DynamicImage` to process (from image crate)
/// * `config` - Configuration for the removal operation (includes model specification)
///
/// # Returns
///
/// A `RemovalResult` containing the processed image, mask, and metadata
///
/// # Examples
///
/// ## Process existing image
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_from_image, ModelSpec, ModelSource};
/// use image::DynamicImage;
///
/// # fn example(img: DynamicImage) -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_image(&img, &config)?;
/// result.save_png("output.png")?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - Model loading errors
/// - Inference execution errors
/// - Image processing failures
pub fn remove_background_from_image(
    image: &image::DynamicImage,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    // Convert RemovalConfig to ProcessorConfig for unified processor
    let processor_config = ProcessorConfigBuilder::new()
        .model_spec(config.model_spec.clone())
        .backend_type(BackendType::Onnx) // Use ONNX for realistic processing
        .execution_provider(config.execution_provider)
        .output_format(config.output_format)
        .jpeg_quality(config.jpeg_quality)
        .webp_quality(config.webp_quality)
        .debug(config.debug)
        .intra_threads(config.intra_threads)
        .inter_threads(config.inter_threads)
        .preserve_color_profiles(config.preserve_color_profiles)
        .build()?;

    let mut unified_processor = BackgroundRemovalProcessor::new(processor_config)?;
    unified_processor.process_image(image)
}

/// Remove background from an async reader stream
///
/// This is the primary API for background removal from streams. It accepts any async
/// readable stream, making it suitable for processing images from network streams,
/// large files, or any other async data source.
///
/// # Arguments
///
/// * `reader` - Any type implementing `AsyncRead + Unpin`
/// * `config` - Complete configuration including model specification and processing options
///
/// # Returns
///
/// A `RemovalResult` containing the processed image, mask, and metadata
///
/// # Examples
///
/// ## Process from file stream
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_from_reader, ModelSpec, ModelSource};
/// use tokio::fs::File;
///
/// # async fn example() -> anyhow::Result<()> {
/// let file = File::open("large_image.jpg").await?;
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_reader(file, &config).await?;
/// result.save_png("output.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## Process from network stream
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_from_reader, ModelSpec, ModelSource};
/// use std::io::Cursor;
///
/// # async fn example() -> anyhow::Result<()> {
/// // Download image bytes from network
/// let image_bytes = download_image_bytes().await?;
/// let reader = Cursor::new(image_bytes);
///
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_reader(reader, &config).await?;
/// # Ok(())
/// # }
/// # async fn download_image_bytes() -> anyhow::Result<Vec<u8>> { Ok(vec![]) }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - Stream reading failures
/// - Image decoding failures
/// - Model loading errors
/// - Inference execution errors
pub async fn remove_background_from_reader<R: AsyncRead + Unpin>(
    mut reader: R,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    // Read all data from the stream into memory
    // TODO: For very large images, consider streaming decode if image crate supports it
    let mut buffer = Vec::new();
    tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut buffer)
        .await
        .map_err(|e| BgRemovalError::processing(format!("Failed to read from stream: {e}")))?;

    // Use the bytes-based API with model_spec from config
    remove_background_from_bytes(&buffer, config)
}

/// Session-based API for efficient model reuse across multiple images
///
/// This struct maintains a loaded model in memory and reuses it for multiple
/// background removal operations. This is more efficient than the high-level
/// convenience functions when processing multiple images, as it avoids the
/// overhead of loading the model on each call.
///
/// # Examples
///
/// ## Processing multiple images efficiently
/// ```rust,no_run
/// use imgly_bgremove::{RemovalSession, RemovalConfig, ModelSpec, ModelSource};
/// use tokio::fs::File;
///
/// # async fn example() -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder().model_spec(model_spec).build()?;
///
/// // Create session (loads model once)
/// let mut session = RemovalSession::new(config)?;
///
/// // Process multiple images with same loaded model
/// for image_path in ["photo1.jpg", "photo2.jpg", "photo3.jpg"] {
///     let file = File::open(image_path).await?;
///     let result = session.remove_background_from_reader(file).await?;
///     let output_name = format!("output_{}", image_path.replace(".jpg", ".png"));
///     result.save_png(&output_name)?;
/// }
/// # Ok(())
/// # }
/// ```
pub struct RemovalSession {
    processor: BackgroundRemovalProcessor,
}

impl RemovalSession {
    /// Create a new removal session with the specified configuration
    ///
    /// This loads the model into memory and prepares it for processing.
    /// The model will stay loaded for the lifetime of this session.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration including model specification and processing options
    ///
    /// # Returns
    ///
    /// A `RemovalSession` ready for processing images
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Invalid configuration parameters
    /// - Model loading failures
    /// - Backend initialization errors
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalSession, RemovalConfig, ModelSpec, ModelSource};
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let model_spec = ModelSpec {
    ///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///     variant: None,
    /// };
    /// let config = RemovalConfig::builder().model_spec(model_spec).build()?;
    /// let session = RemovalSession::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: RemovalConfig) -> Result<Self> {
        // Convert RemovalConfig to ProcessorConfig
        let processor_config = ProcessorConfigBuilder::new()
            .model_spec(config.model_spec)
            .backend_type(BackendType::Onnx) // Use ONNX for realistic processing
            .execution_provider(config.execution_provider)
            .output_format(config.output_format)
            .jpeg_quality(config.jpeg_quality)
            .webp_quality(config.webp_quality)
            .debug(config.debug)
            .intra_threads(config.intra_threads)
            .inter_threads(config.inter_threads)
            .preserve_color_profiles(config.preserve_color_profiles)
            .build()?;

        let processor = BackgroundRemovalProcessor::new(processor_config)?;

        Ok(Self { processor })
    }

    /// Remove background from image bytes
    ///
    /// Processes image data provided as bytes and returns the result.
    /// The model stays loaded in memory for subsequent calls.
    ///
    /// # Arguments
    ///
    /// * `image_bytes` - Raw image data as bytes (JPEG, PNG, WebP, BMP, TIFF)
    ///
    /// # Returns
    ///
    /// A `RemovalResult` containing the processed image, mask, and metadata
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use imgly_bgremove::{RemovalSession, RemovalConfig, ModelSpec, ModelSource};
    /// # fn example(session: &mut RemovalSession, image_data: Vec<u8>) -> anyhow::Result<()> {
    /// let result = session.remove_background_from_bytes(&image_data)?;
    /// result.save_png("output.png")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Image decoding failures
    /// - Processing errors
    /// - Inference execution errors
    pub fn remove_background_from_bytes(&mut self, image_bytes: &[u8]) -> Result<RemovalResult> {
        // Load image from bytes using the image crate
        let image = image::load_from_memory(image_bytes).map_err(|e| {
            BgRemovalError::processing(format!("Failed to decode image from bytes: {e}"))
        })?;

        // Process using the session's processor
        self.processor.process_image(&image)
    }

    /// Remove background from a pre-loaded `DynamicImage`
    ///
    /// Processes a `DynamicImage` directly without any file I/O.
    /// The model stays loaded in memory for subsequent calls.
    ///
    /// # Arguments
    ///
    /// * `image` - A `DynamicImage` to process (from image crate)
    ///
    /// # Returns
    ///
    /// A `RemovalResult` containing the processed image, mask, and metadata
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use imgly_bgremove::{RemovalSession, RemovalConfig, ModelSpec, ModelSource};
    /// # use image::DynamicImage;
    /// # fn example(session: &mut RemovalSession, img: DynamicImage) -> anyhow::Result<()> {
    /// let result = session.remove_background_from_image(&img)?;
    /// result.save_png("output.png")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Processing errors
    /// - Inference execution errors
    pub fn remove_background_from_image(
        &mut self,
        image: &image::DynamicImage,
    ) -> Result<RemovalResult> {
        self.processor.process_image(image)
    }

    /// Remove background from an async reader stream
    ///
    /// Reads image data from any async readable stream and processes it.
    /// The model stays loaded in memory for subsequent calls.
    ///
    /// # Arguments
    ///
    /// * `reader` - Any type implementing `AsyncRead + Unpin`
    ///
    /// # Returns
    ///
    /// A `RemovalResult` containing the processed image, mask, and metadata
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use imgly_bgremove::{RemovalSession, RemovalConfig, ModelSpec, ModelSource};
    /// # use tokio::fs::File;
    /// # async fn example(session: &mut RemovalSession) -> anyhow::Result<()> {
    /// let file = File::open("input.jpg").await?;
    /// let result = session.remove_background_from_reader(file).await?;
    /// result.save_png("output.png")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Stream reading failures
    /// - Image decoding failures
    /// - Processing errors
    /// - Inference execution errors
    pub async fn remove_background_from_reader<R: AsyncRead + Unpin>(
        &mut self,
        mut reader: R,
    ) -> Result<RemovalResult> {
        // Read all data from the stream into memory
        let mut buffer = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut buffer)
            .await
            .map_err(|e| BgRemovalError::processing(format!("Failed to read from stream: {e}")))?;

        // Use the bytes-based API
        self.remove_background_from_bytes(&buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_compiles() {
        // Basic compilation test to ensure API is well-formed
        let _config = RemovalConfig::default();
        // API compiles successfully if we reach this point
    }
}
