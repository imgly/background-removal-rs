#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::unused_async)]

//! # IMG.LY Background Removal Library
//!
//! A high-performance Rust library for background removal using ONNX Runtime and Tract backends
//! with support for multiple neural network models including `ISNet`, `BiRefNet`, and `BiRefNet` Lite.
//!
//! This consolidated library provides efficient background removal capabilities with support for
//! multiple image formats, execution providers, and model configurations. It's designed to be
//! 2-5x faster than JavaScript implementations while maintaining API compatibility.
//!
//! ## Features
//!
//! - **Multiple Models**: `ISNet`, `BiRefNet`, `BiRefNet` Lite with FP16/FP32 variants
//! - **Multiple Backends**: ONNX Runtime (GPU acceleration) and Tract (Pure Rust)
//! - **Format Support**: JPEG, PNG, WebP, BMP, TIFF with ICC color profile preservation
//! - **Hardware Acceleration**: CUDA, `CoreML`, and CPU execution providers
//! - **Model Management**: Automatic downloading and caching of models from `HuggingFace`
//! - **Session Caching**: Performance optimization through cached ONNX Runtime sessions
//! - **CLI Integration**: Optional command-line interface (enable with `cli` feature)
//! - **WebAssembly Compatible**: Pure Rust Tract backend works in WASM
//! - **Async and Sync APIs**: Flexible processing options
//!
//! ## Quick Start
//!
//! ### Ultra-Simple Usage (One Function Call)
//!
//! If you have a model cached, background removal is just one function call:
//!
//! ```rust,no_run
//! use imgly_bgremove::remove_background_simple;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // ONE LINE: Remove background with default settings
//! remove_background_simple("input.jpg", "output.png").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Full Control Usage
//!
//! For custom configuration and model management:
//!
//! ```rust,no_run
//! use imgly_bgremove::{
//!     ModelDownloader, ModelSpec, ModelSource,
//!     RemovalConfig, ExecutionProvider, remove_background_with_model
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Download and cache a model (one-time setup)
//! let downloader = ModelDownloader::new()?;
//! let model_url = "https://huggingface.co/imgly/isnet-general-onnx";
//! let model_id = downloader.download_model(model_url, true).await?;
//!
//! // Configure processing
//! let config = RemovalConfig::builder()
//!     .execution_provider(ExecutionProvider::Auto) // Auto-detect best provider
//!     .build()?;
//!
//! let model_spec = ModelSpec {
//!     source: ModelSource::Downloaded(model_id),
//!     variant: None, // Auto-select variant
//! };
//!
//! // Process image
//! let result = remove_background_with_model("input.jpg", &config, &model_spec).await?;
//! result.save_png("output.png")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Library vs CLI Usage
//!
//! This crate is designed to work seamlessly both as a library and as a CLI application:
//!
//! - **Library Usage**: All core functionality (downloading, caching, processing) is available by default
//! - **CLI Usage**: Enable the `cli` feature for command-line interface and progress reporting
//!
//! ### Feature Flags
//!
//! - `onnx` (default): ONNX Runtime backend with GPU acceleration support
//! - `tract` (default): Pure Rust backend (WASM compatible)
//! - `cli` (default): Command-line interface and progress reporting (optional for library usage)
//! - `webp-support` (default): WebP image format support
//!
//! ### Library-Only Usage
//!
//! To use only as a library without CLI dependencies:
//!
//! ```toml
//! [dependencies]
//! imgly-bgremove = { version = "0.2", default-features = false, features = ["onnx", "tract"] }
//! ```
//!
//! ## Backend Selection
//!
//! ```rust,no_run
//! use imgly_bgremove::backends::{OnnxBackend, TractBackend};
//!
//! // ONNX Runtime backend with GPU acceleration
//! #[cfg(feature = "onnx")]
//! let onnx_backend = OnnxBackend::new();
//!
//! // Pure Rust backend (WASM compatible)
//! #[cfg(feature = "tract")]
//! let tract_backend = TractBackend::new();
//! ```

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
pub mod types;
pub mod utils;

// Internal imports for lib functions
use crate::types::ProcessingMetadata;
use image::{GenericImageView, ImageFormat};
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
    BackendFactory, BackendType, BackgroundRemovalProcessor, DefaultBackendFactory,
    ProcessorConfig, ProcessorConfigBuilder,
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

/// Remove background with minimal setup (convenience function)
///
/// This is the simplest way to remove backgrounds when you have a cached model.
/// Uses default configuration and the first available cached model.
///
/// # Arguments
/// * `input_path` - Path to input image
/// * `output_path` - Path for output image (PNG format)
///
/// # Returns
/// `Ok(())` if successful, error if no cached models or processing fails
///
/// # Examples
/// ```rust,no_run
/// use imgly_bgremove::remove_background_simple;
///
/// # async fn example() -> anyhow::Result<()> {
/// // Assumes you have already downloaded a model
/// remove_background_simple("photo.jpg", "result.png").await?;
/// # Ok(())
/// # }
/// ```
pub async fn remove_background_simple<P: AsRef<std::path::Path>>(
    input_path: P,
    output_path: P,
) -> Result<()> {
    // Read image bytes
    let image_bytes = tokio::fs::read(input_path)
        .await
        .map_err(|e| BgRemovalError::processing(format!("Failed to read input file: {}", e)))?;

    // Process using stream-based API
    let png_bytes = remove_background_simple_bytes(&image_bytes).await?;

    // Write output bytes
    tokio::fs::write(output_path, png_bytes)
        .await
        .map_err(|e| BgRemovalError::processing(format!("Failed to write output file: {}", e)))?;

    Ok(())
}

/// Remove background from an image file with specific model selection
///
/// This is the primary entry point for background removal operations with full control
/// over model selection and configuration. Supports both embedded and external models
/// with automatic provider-aware variant selection.
///
/// # Arguments
///
/// * `input_path` - Path to the input image file (supports JPEG, PNG, WebP, BMP, TIFF)
/// * `config` - Configuration for the removal operation including execution provider and output format
/// * `model_spec` - Specification of which model to use (embedded or external path)
///
/// # Returns
///
/// A `RemovalResult` containing:
/// - The processed image with background removed
/// - The segmentation mask used for removal
/// - Detailed processing metadata and timing information
/// - Original image dimensions
///
/// # Performance
///
/// - **CPU**: 2-5 seconds typical processing time
/// - **CUDA**: 200-500ms with compatible NVIDIA GPU
/// - **`CoreML`**: 100-400ms on Apple Silicon (model-dependent)
///
/// # Supported Models
///
/// - **`ISNet`**: Fast, general-purpose background removal
/// - **`BiRefNet`**: High-quality portrait segmentation
/// - **`BiRefNet` Lite**: Balanced speed/quality option
///
/// # Examples
///
/// ## Basic usage with downloaded model
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_with_model, ModelSpec, ModelSource};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let result = remove_background_with_model("photo.jpg", &config, &model_spec).await?;
/// result.save_png("result.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## High-performance with `CoreML`
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_with_model, ModelSpec, ModelSource, ExecutionProvider};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::builder()
///     .execution_provider(ExecutionProvider::CoreMl)
///     .build()?;
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: Some("fp32".to_string()),
/// };
/// let result = remove_background_with_model("portrait.jpg", &config, &model_spec).await?;
/// result.save_png("portrait_no_bg.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## External model with custom configuration
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_with_model, ModelSpec, ModelSource, OutputFormat};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::builder()
///     .output_format(OutputFormat::WebP)
///     .webp_quality(90)
///     .build()?;
/// let model_spec = ModelSpec {
///     source: ModelSource::External(PathBuf::from("./models/custom_model")),
///     variant: Some("fp32".to_string()),
/// };
/// let result = remove_background_with_model("image.jpg", &config, &model_spec).await?;
/// result.save("output.webp", config.output_format, config.webp_quality)?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - Invalid or unsupported image formats
/// - Model loading failures (missing files, invalid ONNX models)
/// - Execution provider unavailability (e.g., CUDA on non-NVIDIA systems)
/// - Memory allocation failures during processing
/// - File I/O errors when reading input or writing output
pub async fn remove_background_with_model<P: AsRef<std::path::Path>>(
    input_path: P,
    config: &RemovalConfig,
    model_spec: &ModelSpec,
) -> Result<RemovalResult> {
    // Read image bytes
    let image_bytes = tokio::fs::read(input_path)
        .await
        .map_err(|e| BgRemovalError::processing(format!("Failed to read input file: {}", e)))?;

    // Use the bytes-based API
    remove_background_from_bytes(&image_bytes, config, model_spec).await
}

/// Remove background from an image file with a custom backend
///
/// This function allows injecting a specific inference backend, useful when the core
/// crate doesn't have direct access to backend implementations.
///
/// # Arguments
///
/// * `input_path` - Path to the input image file
/// * `config` - Configuration for the removal operation
/// * `backend` - Pre-initialized inference backend
///
/// # Returns
///
/// A `RemovalResult` containing the processed image, mask, and metadata
pub async fn remove_background_with_backend<P: AsRef<std::path::Path>>(
    input_path: P,
    config: &RemovalConfig,
    mut backend: Box<dyn InferenceBackend>,
) -> Result<RemovalResult> {
    // Initialize the backend first
    backend.initialize(config)?;

    // Load image
    let image = image::open(&input_path)
        .map_err(|e| BgRemovalError::processing(format!("Failed to load image: {}", e)))?;

    // Get preprocessing config and preprocess
    let preprocessing_config = backend.get_preprocessing_config()?;
    let input_tensor = ImagePreprocessor::preprocess_for_inference(&image, &preprocessing_config)?;

    // Run inference
    let output_tensor = backend.infer(&input_tensor)?;

    // Convert output to mask - we need to extract this logic from processor
    let (_, _, height, width) = output_tensor.dim();
    let original_dimensions = image.dimensions();

    // Create mask data
    let mut mask_data =
        Vec::with_capacity((original_dimensions.0 * original_dimensions.1) as usize);
    for y in 0..original_dimensions.1 {
        for x in 0..original_dimensions.0 {
            // Map original coordinates to mask coordinates
            let mask_x = (x as f32 * width as f32 / original_dimensions.0 as f32) as usize;
            let mask_y = (y as f32 * height as f32 / original_dimensions.1 as f32) as usize;

            let mask_value = if mask_x < width && mask_y < height {
                #[allow(clippy::indexing_slicing)]
                output_tensor[[0, 0, mask_y.min(height - 1), mask_x.min(width - 1)]]
            } else {
                0.0
            };

            mask_data.push((mask_value.clamp(0.0, 1.0) * 255.0) as u8);
        }
    }

    let mask = SegmentationMask::new(mask_data, original_dimensions);

    // Apply mask to create result image
    let mut rgba_image = image.to_rgba8();
    mask.apply_to_image(&mut rgba_image)?;

    // Handle output format using the service
    let result_image = OutputFormatHandler::convert_format(rgba_image, config.output_format)?;

    let metadata = ProcessingMetadata::new("custom_backend".to_string());
    Ok(RemovalResult::new(
        result_image,
        mask,
        original_dimensions,
        metadata,
    ))
}

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
/// # async fn example(upload_bytes: Vec<u8>) -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let result = remove_background_from_bytes(&upload_bytes, &config, &model_spec).await?;
/// let output_bytes = result.to_bytes(config.output_format, 90)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Memory-based processing
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_from_bytes, ModelSpec, ModelSource};
///
/// # async fn example() -> anyhow::Result<()> {
/// let image_data = download_image_from_api().await?;
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let result = remove_background_from_bytes(&image_data, &config, &model_spec).await?;
/// let png_bytes = result.to_bytes(imgly_bgremove::OutputFormat::Png, 100)?;
/// # Ok(())
/// # }
/// # async fn download_image_from_api() -> anyhow::Result<Vec<u8>> { Ok(vec![]) }
/// ```
pub async fn remove_background_from_bytes(
    image_bytes: &[u8],
    config: &RemovalConfig,
    model_spec: &ModelSpec,
) -> Result<RemovalResult> {
    // Load image from bytes using the image crate
    let image = image::load_from_memory(image_bytes).map_err(|e| {
        BgRemovalError::processing(format!("Failed to decode image from bytes: {}", e))
    })?;

    // Use the existing image processing function
    remove_background_from_image(image, config, model_spec).await
}

/// Remove background from a `DynamicImage` directly
///
/// This is the most flexible API for in-memory image processing. It accepts
/// a pre-loaded `DynamicImage` and processes it without any file I/O.
///
/// # Arguments
///
/// * `image` - A `DynamicImage` to process (from image crate)
/// * `config` - Configuration for the removal operation
/// * `model_spec` - Specification of which model to use
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
/// # async fn example(img: DynamicImage) -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let result = remove_background_from_image(img, &config, &model_spec).await?;
/// result.save_png("output.png")?;
/// # Ok(())
/// # }
/// ```
pub async fn remove_background_from_image(
    image: image::DynamicImage,
    config: &RemovalConfig,
    model_spec: &ModelSpec,
) -> Result<RemovalResult> {
    // Convert RemovalConfig to ProcessorConfig for unified processor
    let processor_config = ProcessorConfigBuilder::new()
        .model_spec(model_spec.clone())
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

    let backend_factory = Box::new(DefaultBackendFactory);
    let mut unified_processor =
        BackgroundRemovalProcessor::with_factory(processor_config, backend_factory)?;
    unified_processor.process_image(&image)
}

/// Remove background from an async reader stream
///
/// This API accepts any async readable stream, making it suitable for processing
/// images from network streams, large files, or any other async data source.
///
/// # Arguments
///
/// * `reader` - Any type implementing `AsyncRead + Unpin`
/// * `format_hint` - Optional hint about the image format for better performance
/// * `config` - Configuration for the removal operation
/// * `model_spec` - Specification of which model to use
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
/// use image::ImageFormat;
///
/// # async fn example() -> anyhow::Result<()> {
/// let file = File::open("large_image.jpg").await?;
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let result = remove_background_from_reader(
///     file,
///     Some(ImageFormat::Jpeg),
///     &config,
///     &model_spec
/// ).await?;
/// result.save_png("output.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## Process from network (simplified example)
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_from_bytes, ModelSpec, ModelSource};
///
/// # async fn example() -> anyhow::Result<()> {
/// // Download image bytes from network
/// let image_bytes = download_image_bytes().await?;
///
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let result = remove_background_from_bytes(&image_bytes, &config, &model_spec).await?;
/// # Ok(())
/// # }
/// # async fn download_image_bytes() -> anyhow::Result<Vec<u8>> { Ok(vec![]) }
/// ```
pub async fn remove_background_from_reader<R: AsyncRead + Unpin>(
    mut reader: R,
    _format_hint: Option<ImageFormat>,
    config: &RemovalConfig,
    model_spec: &ModelSpec,
) -> Result<RemovalResult> {
    // Read all data from the stream into memory
    // TODO: For very large images, consider streaming decode if image crate supports it
    let mut buffer = Vec::new();
    tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut buffer)
        .await
        .map_err(|e| BgRemovalError::processing(format!("Failed to read from stream: {}", e)))?;

    // Use the bytes-based API
    remove_background_from_bytes(&buffer, config, model_spec).await
}

/// Remove background with minimal setup and return PNG bytes
///
/// This is the simplest stream-based API - provide image bytes, get PNG bytes back.
/// Uses default configuration and the first available cached model.
///
/// # Arguments
/// * `image_bytes` - Raw image data as bytes
///
/// # Returns
/// PNG-encoded bytes with transparent background
///
/// # Examples
/// ```rust,no_run
/// use imgly_bgremove::remove_background_simple_bytes;
///
/// # async fn example(upload_data: Vec<u8>) -> anyhow::Result<()> {
/// // Ultra-simple: bytes in, PNG bytes out
/// let png_bytes = remove_background_simple_bytes(&upload_data).await?;
/// // Send PNG bytes back to client, save to file, etc.
/// tokio::fs::write("result.png", png_bytes).await?;
/// # Ok(())
/// # }
/// ```
pub async fn remove_background_simple_bytes(image_bytes: &[u8]) -> Result<Vec<u8>> {
    // Find first available cached model
    let cache = ModelCache::new()?;
    let cached_models = cache.scan_cached_models()?;

    if cached_models.is_empty() {
        return Err(BgRemovalError::invalid_config(
            "No cached models found. Download a model first using ModelDownloader or the CLI."
                .to_string(),
        ));
    }

    let model_spec = ModelSpec {
        source: ModelSource::Downloaded(
            cached_models
                .first()
                .ok_or_else(|| {
                    BgRemovalError::invalid_config("No cached models available".to_string())
                })?
                .model_id
                .clone(),
        ),
        variant: None,
    };

    let config = RemovalConfig::default();
    let result = remove_background_from_bytes(image_bytes, &config, &model_spec).await?;

    // Return PNG bytes
    result.to_bytes(OutputFormat::Png, 100)
}

/// Remove background with model selection and return bytes
///
/// This stream-based API provides full control over configuration and model selection
/// while returning the result as bytes in the specified format.
///
/// # Arguments
/// * `image_bytes` - Raw image data as bytes
/// * `config` - Configuration including output format and quality
/// * `model_spec` - Specification of which model to use
///
/// # Returns
/// Image bytes in the format specified by `config.output_format`
///
/// # Examples
/// ```rust,no_run
/// use imgly_bgremove::{
///     remove_background_with_model_bytes, RemovalConfig, ModelSpec, ModelSource, OutputFormat
/// };
///
/// # async fn example(image_data: Vec<u8>) -> anyhow::Result<()> {
/// let config = RemovalConfig::builder()
///     .output_format(OutputFormat::WebP)
///     .webp_quality(90)
///     .build()?;
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let webp_bytes = remove_background_with_model_bytes(&image_data, &config, &model_spec).await?;
/// # Ok(())
/// # }
/// ```
pub async fn remove_background_with_model_bytes(
    image_bytes: &[u8],
    config: &RemovalConfig,
    model_spec: &ModelSpec,
) -> Result<Vec<u8>> {
    let result = remove_background_from_bytes(image_bytes, config, model_spec).await?;
    result.to_bytes(
        config.output_format,
        config.webp_quality.max(config.jpeg_quality),
    )
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
