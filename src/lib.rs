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

#[cfg(feature = "video-support")]
pub use backends::video::{FrameProcessingStats, VideoFrame};
#[cfg(feature = "video-support")]
pub use config::VideoProcessingConfig;
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

#[cfg(feature = "video-support")]
pub use types::VideoRemovalResult;
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
/// # let image_data = vec![0u8; 1024]; // Mock image data
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
/// # let image_bytes = vec![0u8; 1024]; // Mock image data
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

/// Remove background from a video file (when video support is enabled)
///
/// This function processes a video file frame by frame, removing the background
/// from each frame and reassembling the result into a new video file.
///
/// # Arguments
///
/// * `input_path` - Path to the input video file
/// * `config` - Configuration for the removal operation
///
/// # Returns
///
/// A `VideoRemovalResult` containing the processed video data and metadata
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "video-support")]
/// # {
/// use imgly_bgremove::{RemovalConfig, remove_background_from_video_file, ModelSpec, ModelSource, VideoProcessingConfig};
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .video_config(VideoProcessingConfig::default())
///     .build()?;
///
/// let result = remove_background_from_video_file("input.mp4", &config).await?;
/// result.save("output.mp4")?;
/// # Ok(())
/// # }
/// # }
/// ```
#[cfg(feature = "video-support")]
pub async fn remove_background_from_video_file<P: AsRef<std::path::Path>>(
    input_path: P,
    config: &RemovalConfig,
) -> Result<VideoRemovalResult> {
    use crate::backends::video::{FfmpegBackend, VideoBackend};
    use crate::processor::BackgroundRemovalProcessor;
    use crate::types::ProcessingMetadata;
    use futures::StreamExt;

    let backend = FfmpegBackend::new()?;
    let metadata = backend.get_metadata(input_path.as_ref()).await?;

    // Extract frames
    let mut frame_stream = backend.extract_frames(input_path.as_ref()).await?;
    let mut processed_frames = Vec::new();
    let mut frame_stats = FrameProcessingStats::new();

    // Create processor for background removal
    let default_video_config = VideoProcessingConfig::default();
    let video_config = config
        .video_config
        .as_ref()
        .unwrap_or(&default_video_config);
    let processor_config = ProcessorConfigBuilder::new()
        .model_spec(config.model_spec.clone())
        .execution_provider(config.execution_provider)
        .build()?;
    let mut processor = BackgroundRemovalProcessor::new(processor_config)?;

    // Process frames
    let _start_time = instant::Instant::now();
    let total_frames = metadata.frame_count.unwrap_or(0);
    let mut current_frame = 0u64;
    
    while let Some(frame_result) = frame_stream.next().await {
        let frame = frame_result?;
        let frame_start = instant::Instant::now();

        // Call progress callback if provided
        if let Some(ref callback) = config.video_progress_callback {
            callback(current_frame, total_frames);
        }

        match processor.process_image(&frame.to_dynamic_image()) {
            Ok(result) => {
                // Convert result back to video frame
                let processed_frame = VideoFrame::from_dynamic_image(
                    result.image,
                    frame.frame_number,
                    frame.timestamp,
                );
                processed_frames.push(processed_frame);
                frame_stats.add_frame_time(frame_start.elapsed());
            },
            Err(_) => {
                frame_stats.mark_frame_failed();
            },
        }
        
        current_frame += 1;
    }

    // Reassemble video
    let temp_output = tempfile::NamedTempFile::new().map_err(|e| {
        BgRemovalError::processing(format!("Failed to create temporary file: {}", e))
    })?;

    let frame_stream = futures::stream::iter(processed_frames.into_iter().map(Ok));
    backend
        .reassemble_video(
            Box::pin(frame_stream),
            temp_output.path(),
            &metadata,
            video_config.preserve_audio,
        )
        .await?;

    // Read the processed video data
    let video_data = std::fs::read(temp_output.path()).map_err(|e| {
        BgRemovalError::processing(format!("Failed to read processed video: {}", e))
    })?;

    let processing_metadata = ProcessingMetadata::new("video".to_string());

    Ok(VideoRemovalResult::new(
        video_data,
        metadata,
        frame_stats,
        processing_metadata,
        None, // Color profile extraction from video not yet implemented
    ))
}

/// Remove background from video data provided as bytes (when video support is enabled)
///
/// This function processes video data from memory, making it suitable for
/// web servers, streaming applications, and scenarios where files aren't available.
///
/// # Arguments
///
/// * `video_bytes` - Raw video data as bytes
/// * `config` - Configuration for the removal operation
///
/// # Returns
///
/// A `VideoRemovalResult` containing the processed video data and metadata
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "video-support")]
/// # {
/// use imgly_bgremove::{RemovalConfig, remove_background_from_video_bytes, ModelSpec, ModelSource, VideoProcessingConfig};
///
/// # async fn example(video_data: Vec<u8>) -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .video_config(VideoProcessingConfig::default())
///     .build()?;
///
/// let result = remove_background_from_video_bytes(&video_data, &config).await?;
/// let output_data = result.to_bytes();
/// # Ok(())
/// # }
/// # }
/// ```
#[cfg(feature = "video-support")]
pub async fn remove_background_from_video_bytes(
    video_bytes: &[u8],
    config: &RemovalConfig,
) -> Result<VideoRemovalResult> {
    // Write bytes to temporary file and process
    let temp_input = tempfile::NamedTempFile::new().map_err(|e| {
        BgRemovalError::processing(format!("Failed to create temporary input file: {}", e))
    })?;

    std::fs::write(temp_input.path(), video_bytes).map_err(|e| {
        BgRemovalError::processing(format!(
            "Failed to write video data to temporary file: {}",
            e
        ))
    })?;

    remove_background_from_video_file(temp_input.path(), config).await
}

/// Remove background from a video provided via an async reader (when video support is enabled)
///
/// This function reads video data from any async reader and processes it,
/// making it suitable for streaming from files, network connections, or
/// any other async source.
///
/// # Arguments
///
/// * `reader` - Any type implementing `AsyncRead + Unpin`
/// * `config` - Configuration for the removal operation
///
/// # Returns
///
/// A `VideoRemovalResult` containing the processed video data and metadata
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "video-support")]
/// # {
/// use imgly_bgremove::{RemovalConfig, remove_background_from_video_reader, ModelSpec, ModelSource, VideoProcessingConfig};
/// use tokio::fs::File;
///
/// # async fn example() -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .video_config(VideoProcessingConfig::default())
///     .build()?;
///
/// let file = File::open("input.mp4").await?;
/// let result = remove_background_from_video_reader(file, &config).await?;
/// result.save("output.mp4")?;
/// # Ok(())
/// # }
/// # }
/// ```
#[cfg(feature = "video-support")]
pub async fn remove_background_from_video_reader<R: AsyncRead + Unpin>(
    mut reader: R,
    config: &RemovalConfig,
) -> Result<VideoRemovalResult> {
    use tokio::io::AsyncReadExt;

    // Read all video data into memory
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .await
        .map_err(|e| BgRemovalError::processing(format!("Failed to read video data: {}", e)))?;

    // Use the bytes-based API
    remove_background_from_video_bytes(&buffer, config).await
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
    
    #[cfg(feature = "video-support")]
    #[tokio::test]
    async fn test_video_progress_callback() {
        use std::sync::{Arc, Mutex};
        
        // Create a shared state to track progress calls
        let progress_calls = Arc::new(Mutex::new(Vec::<(u64, u64)>::new()));
        let progress_calls_clone = progress_calls.clone();
        
        // Create a RemovalConfig with a progress callback
        let mut config = RemovalConfig::default();
        config.video_progress_callback = Some(Box::new(move |current, total| {
            progress_calls_clone.lock().unwrap().push((current, total));
        }));
        
        // The callback should be set
        assert!(config.video_progress_callback.is_some());
        
        // Test the callback directly
        if let Some(ref callback) = config.video_progress_callback {
            callback(0, 100);
            callback(25, 100);
            callback(50, 100);
            callback(75, 100);
            callback(100, 100);
        }
        
        // Verify the progress was tracked correctly
        let calls = progress_calls.lock().unwrap();
        assert_eq!(calls.len(), 5);
        assert_eq!(calls[0], (0, 100));
        assert_eq!(calls[1], (25, 100));
        assert_eq!(calls[2], (50, 100));
        assert_eq!(calls[3], (75, 100));
        assert_eq!(calls[4], (100, 100));
    }
    
    #[cfg(feature = "video-support")]
    #[test]
    fn test_video_progress_callback_with_config_builder() {
        use std::sync::{Arc, Mutex};
        
        // Create a shared state to track progress
        let progress_state = Arc::new(Mutex::new(Vec::<(u64, u64)>::new()));
        let progress_state_clone = progress_state.clone();
        
        // Build config with video progress callback
        let config = RemovalConfig::builder()
            .video_progress_callback(Box::new(move |current, total| {
                progress_state_clone.lock().unwrap().push((current, total));
            }))
            .build()
            .unwrap();
        
        // Test that callback works
        if let Some(ref callback) = config.video_progress_callback {
            callback(10, 50);
            callback(20, 50);
            callback(30, 50);
        }
        
        let calls = progress_state.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], (10, 50));
        assert_eq!(calls[1], (20, 50));
        assert_eq!(calls[2], (30, 50));
    }
    
    #[cfg(feature = "video-support")]
    #[test]
    fn test_video_progress_callback_clone_behavior() {
        // Test that RemovalConfig can be cloned even with video_progress_callback
        let mut config1 = RemovalConfig::default();
        config1.video_progress_callback = Some(Box::new(|_current, _total| {
            // Empty callback
        }));
        
        // Clone should work and callback should be None (as per our Clone implementation)
        let config2 = config1.clone();
        assert!(config1.video_progress_callback.is_some());
        assert!(config2.video_progress_callback.is_none());
    }
    
    #[cfg(feature = "video-support")]
    #[test]
    fn test_video_progress_callback_with_unknown_total() {
        use std::sync::{Arc, Mutex};
        
        // Test callback behavior when total frames is unknown (0)
        let progress_calls = Arc::new(Mutex::new(Vec::<(u64, u64)>::new()));
        let progress_calls_clone = progress_calls.clone();
        
        let mut config = RemovalConfig::default();
        config.video_progress_callback = Some(Box::new(move |current, total| {
            progress_calls_clone.lock().unwrap().push((current, total));
        }));
        
        // Simulate progress with unknown total frames
        if let Some(ref callback) = config.video_progress_callback {
            callback(0, 0);
            callback(10, 0);
            callback(20, 0);
            callback(30, 0);
        }
        
        let calls = progress_calls.lock().unwrap();
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0], (0, 0));
        assert_eq!(calls[1], (10, 0));
        assert_eq!(calls[2], (20, 0));
        assert_eq!(calls[3], (30, 0));
    }
    
    #[cfg(feature = "video-support")]
    #[test]
    fn test_removal_config_partial_eq_with_callback() {
        // Test PartialEq implementation with video_progress_callback
        let mut config1 = RemovalConfig::default();
        let config2 = RemovalConfig::default();
        
        // Initially equal
        assert_eq!(config1, config2);
        
        // Add callback to config1
        config1.video_progress_callback = Some(Box::new(|_, _| {}));
        
        // Still equal because PartialEq ignores video_progress_callback
        assert_eq!(config1, config2);
        
        // Change a different field
        config1.output_format = OutputFormat::WebP;
        assert_ne!(config1, config2);
    }
    
    #[cfg(feature = "video-support")]
    #[test]
    fn test_removal_config_debug_with_callback() {
        // Test Debug implementation with video_progress_callback
        let mut config = RemovalConfig::default();
        config.video_progress_callback = Some(Box::new(|current, total| {
            println!("Progress: {}/{}", current, total);
        }));
        
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("video_progress_callback: true"));
        
        // Test without callback
        let config_no_callback = RemovalConfig::default();
        let debug_str_no_callback = format!("{:?}", config_no_callback);
        assert!(debug_str_no_callback.contains("video_progress_callback: false"));
    }
    
    #[cfg(feature = "video-support")]
    #[tokio::test]
    async fn test_video_progress_callback_thread_safety() {
        use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
        
        // Test that callback can be used across threads
        let current_frame = Arc::new(AtomicU64::new(0));
        let total_frames = Arc::new(AtomicU64::new(0));
        
        let current_clone = current_frame.clone();
        let total_clone = total_frames.clone();
        
        let mut config = RemovalConfig::default();
        config.video_progress_callback = Some(Box::new(move |current, total| {
            current_clone.store(current, Ordering::SeqCst);
            total_clone.store(total, Ordering::SeqCst);
        }));
        
        // Spawn a task to call the callback
        let config_clone = Arc::new(config);
        let handle = tokio::spawn(async move {
            if let Some(ref callback) = config_clone.video_progress_callback {
                callback(42, 100);
            }
        });
        
        handle.await.unwrap();
        
        // Verify the values were updated
        assert_eq!(current_frame.load(Ordering::SeqCst), 42);
        assert_eq!(total_frames.load(Ordering::SeqCst), 100);
    }
}
