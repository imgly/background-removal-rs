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
//! ### Primary API Usage
//!
//! For custom configuration and model management:
//!
//! ```rust,no_run
//! use imgly_bgremove::{
//!     ModelDownloader, ModelSpec, ModelSource,
//!     RemovalConfig, ExecutionProvider, remove_background_from_reader
//! };
//! use tokio::fs::File;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Download and cache a model (one-time setup)
//! let downloader = ModelDownloader::new()?;
//! let model_url = "https://huggingface.co/imgly/isnet-general-onnx";
//! let model_id = downloader.download_model(model_url, true).await?;
//!
//! // Configure processing with model
//! let model_spec = ModelSpec {
//!     source: ModelSource::Downloaded(model_id),
//!     variant: None, // Auto-select variant
//! };
//! let config = RemovalConfig::builder()
//!     .execution_provider(ExecutionProvider::Auto) // Auto-detect best provider
//!     .model_spec(model_spec)
//!     .build()?;
//!
//! // Process image using reader-based API
//! let file = File::open("input.jpg").await?;
//! let result = remove_background_from_reader(file, &config).await?;
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
/// # async fn example(upload_bytes: Vec<u8>) -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_bytes(&upload_bytes, &config).await?;
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
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_bytes(&image_data, &config).await?;
/// let png_bytes = result.to_bytes(imgly_bgremove::OutputFormat::Png, 100)?;
/// # Ok(())
/// # }
/// # async fn download_image_from_api() -> anyhow::Result<Vec<u8>> { Ok(vec![]) }
/// ```
pub async fn remove_background_from_bytes(
    image_bytes: &[u8],
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    // Load image from bytes using the image crate
    let image = image::load_from_memory(image_bytes).map_err(|e| {
        BgRemovalError::processing(format!("Failed to decode image from bytes: {}", e))
    })?;

    // Use the existing image processing function with model_spec from config
    remove_background_from_image(image, config).await
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
/// # async fn example(img: DynamicImage) -> anyhow::Result<()> {
/// let model_spec = ModelSpec {
///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
///     variant: None,
/// };
/// let config = RemovalConfig::builder()
///     .model_spec(model_spec)
///     .build()?;
/// let result = remove_background_from_image(img, &config).await?;
/// result.save_png("output.png")?;
/// # Ok(())
/// # }
/// ```
pub async fn remove_background_from_image(
    image: image::DynamicImage,
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

    let backend_factory = Box::new(DefaultBackendFactory);
    let mut unified_processor =
        BackgroundRemovalProcessor::with_factory(processor_config, backend_factory)?;
    unified_processor.process_image(&image)
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
pub async fn remove_background_from_reader<R: AsyncRead + Unpin>(
    mut reader: R,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    // Read all data from the stream into memory
    // TODO: For very large images, consider streaming decode if image crate supports it
    let mut buffer = Vec::new();
    tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut buffer)
        .await
        .map_err(|e| BgRemovalError::processing(format!("Failed to read from stream: {}", e)))?;

    // Use the bytes-based API with model_spec from config
    remove_background_from_bytes(&buffer, config).await
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
