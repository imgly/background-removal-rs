#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::unused_async)]

//! # IMG.LY Background Removal Library
//!
//! A high-performance Rust library for background removal using ONNX Runtime and Tract backends
//! with support for multiple neural network models including ISNet, BiRefNet, and BiRefNet Lite.
//!
//! This consolidated library provides efficient background removal capabilities with support for
//! multiple image formats, execution providers, and model configurations. It's designed to be
//! 2-5x faster than JavaScript implementations while maintaining API compatibility.
//!
//! ## Features
//!
//! - **Multiple Models**: ISNet, BiRefNet, BiRefNet Lite with FP16/FP32 variants
//! - **Multiple Backends**: ONNX Runtime (GPU acceleration) and Tract (Pure Rust)
//! - **Format Support**: JPEG, PNG, WebP, BMP, TIFF with ICC color profile preservation
//! - **Hardware Acceleration**: CUDA, CoreML, and CPU execution providers
//! - **CLI Integration**: Command-line interface included by default
//! - **WebAssembly Compatible**: Pure Rust Tract backend works in WASM
//! - **Async and Sync APIs**: Flexible processing options
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use imgly_bgremove::{RemovalConfig, remove_background, ExecutionProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = RemovalConfig::builder()
//!     .execution_provider(ExecutionProvider::Auto) // Auto-detect best provider
//!     .build()?;
//! let result = remove_background("input.jpg", &config).await?;
//! result.save_png("output.png")?;
//! # Ok(())
//! # }
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
#[cfg(feature = "cli")]
pub mod cache;
#[cfg(feature = "cli")]
pub mod cli;
pub mod color_profile;
pub mod config;
#[cfg(feature = "cli")]
pub mod download;
pub mod error;
pub mod inference;
pub mod models;
pub mod processor;
pub mod services;
pub mod types;
pub mod utils;

// Internal imports for lib functions
use crate::types::ProcessingMetadata;
use image::GenericImageView;

// Public API exports
pub use backends::*;
#[cfg(feature = "cli")]
pub use cache::{format_size, CachedModelInfo, ModelCache};
pub use color_profile::{ProfileEmbedder, ProfileExtractor};
pub use config::{ExecutionProvider, OutputFormat, RemovalConfig};
#[cfg(feature = "cli")]
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
pub use types::{ColorProfile, ColorSpace, RemovalResult, SegmentationMask};
pub use utils::{
    ConfigValidator, ExecutionProviderManager, ImagePreprocessor, ModelSpecParser, ModelValidator,
    NumericValidator, PathValidator, PreprocessingOptions, ProviderInfo, TensorValidator,
};

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
    unified_processor.process_file(input_path).await
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

/// Remove background from an image file (deprecated - use remove_background_with_model)
///
/// This function has been deprecated as embedded models are no longer supported.
/// Use `remove_background_with_model()` with downloaded or external models instead.
///
/// # Migration Guide
///
/// Instead of using this function, use one of these approaches:
///
/// ## Option 1: Download and use models
/// ```bash
/// # Download a model using the CLI
/// imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx
/// ```
///
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
/// ## Option 2: Use external model directory
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, remove_background_with_model, ModelSpec, ModelSource};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::External(PathBuf::from("/path/to/model")),
///     variant: Some("fp16".to_string()),
/// };
/// let result = remove_background_with_model("photo.jpg", &config, &model_spec).await?;
/// result.save_png("result.png")?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// This function always returns an error directing users to the new download workflow.
///
/// # Note
///
/// See the [migration guide](https://docs.rs/imgly-bgremove/latest/imgly_bgremove/index.html)
/// for detailed instructions on migrating from embedded models to downloaded models.
#[deprecated(
    since = "0.3.0",
    note = "Use remove_background_with_model() with downloaded or external models. See migration guide for details."
)]
pub async fn remove_background<P: AsRef<std::path::Path>>(
    _input_path: P,
    _config: &RemovalConfig,
) -> Result<RemovalResult> {
    Err(BgRemovalError::invalid_config(
        "Embedded models are no longer supported. Please use remove_background_with_model() with downloaded models.\n\
         \n\
         To download models:\n\
         1. CLI: imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx\n\
         2. List downloaded models: imgly-bgremove --list-models\n\
         3. Use with ModelSource::Downloaded(\"model-id\") in remove_background_with_model()\n\
         \n\
         See documentation for migration guide."
    ))
}

/// Process a `DynamicImage` directly for background removal (deprecated)
///
/// This function has been deprecated as embedded models are no longer supported.
/// Use the unified `BackgroundRemovalProcessor` with downloaded or external models instead.
///
/// # Arguments
///
/// * `image` - The input `DynamicImage` to process (from `image` crate)
/// * `config` - Configuration for execution provider and output settings
///
/// # Returns
///
/// A `RemovalResult` containing:
/// - Processed image with background removed as alpha channel
/// - Binary segmentation mask used for removal
/// - Processing timing and metadata
///
/// # Performance
///
/// Since there's no file I/O overhead, this is typically 10-50ms faster than file-based processing:
/// - **CPU**: 1.8-4.5 seconds typical
/// - **GPU accelerated**: 80-450ms (CoreML/CUDA)
///
/// # Memory Usage
///
/// Requires approximately 3x the input image size in memory during processing:
/// - Input image (original format)
/// - Preprocessed tensor data (float32)
/// - Output image with alpha channel
///
/// # Examples
///
/// ## Basic in-memory processing
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, process_image};
/// use image::DynamicImage;
///
/// # fn example(img: DynamicImage) -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let result = process_image(img, &config)?;
/// result.save_png("result.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## Processing image from bytes with custom config
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, process_image, ExecutionProvider};
/// use image::ImageFormat;
/// use std::io::Cursor;
///
/// # fn example(image_bytes: Vec<u8>) -> anyhow::Result<()> {
/// let img = image::load(Cursor::new(image_bytes), ImageFormat::Jpeg)?;
/// let config = RemovalConfig::builder()
///     .execution_provider(ExecutionProvider::CoreMl)
///     .debug(true)
///     .build()?;
/// let result = process_image(img, &config)?;
///
/// // Get result as bytes for web response
/// let output_bytes = result.to_bytes(config.output_format, 90)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Batch processing with timing analysis
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, process_image};
/// use image::DynamicImage;
///
/// # fn example(images: Vec<DynamicImage>) -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
///
/// for (i, img) in images.into_iter().enumerate() {
///     let result = process_image(img, &config)?;
///     println!("Image {}: {}", i, result.timing_summary());
///     result.save_png(format!("output_{}.png", i))?;
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - No embedded models available
/// - Image format incompatibilities or corrupted data
/// - Insufficient memory for processing
/// - ONNX Runtime execution failures
///
/// # Note
///
/// This function is synchronous and uses the first available embedded model.
/// For async processing or specific model selection, load the image and use
/// `remove_background_with_model()` instead.
#[deprecated(
    since = "0.3.0",
    note = "Use remove_background_with_model() with downloaded or external models. See migration guide for details."
)]
#[allow(clippy::needless_pass_by_value)]
pub fn process_image(
    _image: image::DynamicImage,
    _config: &RemovalConfig,
) -> Result<RemovalResult> {
    Err(BgRemovalError::invalid_config(
        "Embedded models are no longer supported. Please use the new download workflow.\n\
         \n\
         To download models:\n\
         1. CLI: imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx\n\
         2. Use ModelSource::Downloaded(\"model-id\") with unified processor\n\
         \n\
         See documentation for migration guide."
    ))
}

/// Extract foreground segmentation mask from an image without background removal
///
/// Returns only the binary segmentation mask without applying it to create a transparent image.
/// Useful for advanced workflows where you need the mask separately or want to apply
/// custom post-processing before background removal.
///
/// # Arguments
///
/// * `input_path` - Path to the input image file (JPEG, PNG, WebP, BMP, TIFF)
/// * `config` - Configuration for execution provider and model selection
///
/// # Returns
///
/// A `SegmentationMask` containing:
/// - Binary mask data as grayscale values (0-255)
/// - Mask dimensions matching the model input size (typically 1024x1024)
/// - Statistics about foreground/background pixel ratios
///
/// # Use Cases
///
/// - **Mask analysis**: Examine segmentation quality before applying
/// - **Custom post-processing**: Apply morphological operations, smoothing, or edge refinement
/// - **Batch processing**: Generate masks separately from background removal
/// - **Quality control**: Validate mask quality programmatically
/// - **Alternative backgrounds**: Apply different backgrounds or effects
///
/// # Performance
///
/// Slightly faster than full background removal since it skips the final compositing step:
/// - **CPU**: 1.5-4 seconds typical
/// - **GPU accelerated**: 80-400ms (CoreML/CUDA)
///
/// # Examples
///
/// ## Basic mask extraction
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, segment_foreground};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let mask = segment_foreground("photo.jpg", &config).await?;
///
/// // Analyze mask quality
/// let stats = mask.statistics();
/// println!("Foreground: {:.1}% of image", stats.foreground_ratio * 100.0);
///
/// // Save mask for inspection
/// mask.save_png("mask.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## High-quality mask with analysis
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, segment_foreground, ExecutionProvider};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::builder()
///     .execution_provider(ExecutionProvider::CoreMl)
///     .build()?;
/// let mask = segment_foreground("portrait.jpg", &config).await?;
///
/// let stats = mask.statistics();
/// if stats.foreground_ratio < 0.1 {
///     println!("Warning: Very small foreground detected ({:.1}%)", stats.foreground_ratio * 100.0);
/// }
///
/// // Resize mask to original image dimensions if needed
/// let resized_mask = mask.resize(1920, 1080)?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - Invalid or unsupported image formats
/// - Model loading or execution failures
/// - Memory allocation errors during inference
/// - File I/O errors when reading input
#[deprecated(
    since = "0.3.0",
    note = "Use the unified processor with downloaded or external models. See migration guide for details."
)]
pub async fn segment_foreground<P: AsRef<std::path::Path>>(
    _input_path: P,
    _config: &RemovalConfig,
) -> Result<SegmentationMask> {
    Err(BgRemovalError::invalid_config(
        "Embedded models are no longer supported. Please use the new download workflow.\n\
         \n\
         To download models:\n\
         1. CLI: imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx\n\
         2. Use ModelSource::Downloaded(\"model-id\") with unified processor\n\
         \n\
         See documentation for migration guide."
    ))
}

/// Apply a pre-computed segmentation mask to an image for background removal
///
/// Takes an existing segmentation mask and applies it to an image to create a transparent
/// background. Useful for applying masks generated separately, reusing masks across
/// similar images, or applying custom-processed masks.
///
/// # Arguments
///
/// * `input_path` - Path to the input image file to apply the mask to
/// * `mask` - Pre-computed segmentation mask (must match image dimensions)
/// * `config` - Configuration for output format and quality settings
///
/// # Returns
///
/// A `RemovalResult` containing:
/// - Image with mask applied as alpha channel (transparent background)
/// - Copy of the applied segmentation mask
/// - Processing metadata (timing will show minimal inference time)
///
/// # Mask Requirements
///
/// - Mask dimensions must match the target image dimensions, or
/// - Use `mask.resize()` to scale the mask to match the image
/// - Mask values: 0-255 grayscale (0=background, 255=foreground)
///
/// # Performance
///
/// Very fast since no AI inference is required - only image compositing:
/// - **Typical**: 10-50ms regardless of execution provider
/// - Performance scales with image resolution, not model complexity
///
/// # Use Cases
///
/// - **Batch processing**: Generate one high-quality mask, apply to multiple similar images
/// - **Custom workflows**: Post-process masks before applying (smoothing, dilation, etc.)
/// - **Mask reuse**: Save and reapply masks for consistent results
/// - **A/B testing**: Compare different mask generation approaches
/// - **Performance optimization**: Pre-compute masks for real-time applications
///
/// # Examples
///
/// ## Apply existing mask to new image
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, segment_foreground, apply_segmentation_mask};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
///
/// // Generate mask from one image
/// let mask = segment_foreground("reference.jpg", &config).await?;
///
/// // Apply to another similar image
/// let result = apply_segmentation_mask("target.jpg", &mask, &config).await?;
/// result.save_png("target_no_bg.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## Batch processing with mask reuse
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, segment_foreground, apply_segmentation_mask};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
///
/// // Generate high-quality mask once
/// let mask = segment_foreground("template.jpg", &config).await?;
///
/// // Apply to multiple images quickly
/// for i in 1..=10 {
///     let input = format!("photo_{}.jpg", i);
///     let output = format!("result_{}.png", i);
///     let result = apply_segmentation_mask(&input, &mask, &config).await?;
///     result.save_png(output)?;
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Apply resized mask to different image size
/// ```rust,no_run
/// use imgly_bgremove::{RemovalConfig, apply_segmentation_mask, SegmentationMask};
///
/// # async fn example(mask: SegmentationMask) -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
///
/// // Resize mask to match target image (e.g., 4K image)
/// let resized_mask = mask.resize(3840, 2160)?;
///
/// let result = apply_segmentation_mask("high_res.jpg", &resized_mask, &config).await?;
/// result.save_png("high_res_no_bg.png")?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - Image and mask dimension mismatches
/// - Invalid input image format or corrupted files
/// - File I/O errors when reading input or writing output
/// - Memory allocation failures during compositing
#[deprecated(
    since = "0.3.0",
    note = "Use the unified processor with downloaded or external models. See migration guide for details."
)]
pub async fn apply_segmentation_mask<P: AsRef<std::path::Path>>(
    _input_path: P,
    _mask: &SegmentationMask,
    _config: &RemovalConfig,
) -> Result<RemovalResult> {
    Err(BgRemovalError::invalid_config(
        "Embedded models are no longer supported. Please use the new download workflow.\n\
         \n\
         To download models:\n\
         1. CLI: imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx\n\
         2. Use ModelSource::Downloaded(\"model-id\") with unified processor\n\
         \n\
         See documentation for migration guide."
    ))
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
