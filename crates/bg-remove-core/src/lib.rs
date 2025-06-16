//! # Background Removal Core Library
//!
//! A high-performance Rust library for background removal using ONNX Runtime and `ISNet` models.
//!
//! This library provides efficient background removal capabilities with support for multiple
//! image formats and model configurations. It's designed to be 2-5x faster than JavaScript
//! implementations while maintaining API compatibility.
//!
//! ## Features
//!
//! - High-performance background removal using `ISNet` models
//! - Support for multiple image formats (JPEG, PNG, WebP)
//! - ICC color profile preservation (enabled by default)
//! - Configurable model precision (FP32, FP16)
//! - Async and sync API support
//! - Memory-efficient processing
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use bg_remove_core::{RemovalConfig, remove_background, ExecutionProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = RemovalConfig::builder()
//!     .execution_provider(ExecutionProvider::Auto) // or Cpu, Cuda, CoreMl
//!     .build()?;
//! let result = remove_background("input.jpg", &config).await?;
//! result.save_png("output.png")?;
//! # Ok(())
//! # }
//! ```

pub mod backends;
pub mod color_profile;
pub mod config;
pub mod error;
pub mod image_processing;
pub mod inference;
pub mod models;
pub mod types;

// Public API exports
pub use backends::{MockBackend, OnnxBackend};
pub use color_profile::{ProfileEmbedder, ProfileExtractor};
pub use config::{
    BackgroundColor, ColorManagementConfig, ExecutionProvider, OutputFormat, RemovalConfig,
};
pub use error::{BgRemovalError, Result};
pub use image_processing::{ImageProcessor, ProcessingOptions};
pub use inference::InferenceBackend;
pub use models::{get_available_embedded_models, ModelManager, ModelSource, ModelSpec};
pub use types::{ColorProfile, ColorSpace, RemovalResult, SegmentationMask};

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
/// ## Basic usage with embedded model
/// ```rust,no_run
/// use bg_remove_core::{RemovalConfig, remove_background_with_model, ModelSpec, ModelSource};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let model_spec = ModelSpec {
///     source: ModelSource::Embedded("isnet-fp16".to_string()),
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
/// use bg_remove_core::{RemovalConfig, remove_background_with_model, ModelSpec, ModelSource, ExecutionProvider};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::builder()
///     .execution_provider(ExecutionProvider::CoreMl)
///     .build()?;
/// let model_spec = ModelSpec {
///     source: ModelSource::Embedded("isnet-fp32".to_string()),
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
/// use bg_remove_core::{RemovalConfig, remove_background_with_model, ModelSpec, ModelSource, OutputFormat};
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
    let model_manager =
        ModelManager::from_spec_with_provider(model_spec, Some(&config.execution_provider))?;
    let mut processor = ImageProcessor::with_model_manager(config, model_manager)?;
    processor.remove_background(input_path).await
}

/// Remove background from an image file using first available embedded model
///
/// This is a convenience function that automatically selects the first available embedded model.
/// Use `remove_background_with_model()` for explicit model control and better performance.
///
/// # Automatic Model Selection
///
/// Selects the first available embedded model in this priority order:
/// 1. `isnet-fp16` - Fast general-purpose model
/// 2. `birefnet-fp16` - High-quality portrait model  
/// 3. `birefnet-lite-fp32` - Balanced performance model
/// 4. Any other available embedded models
///
/// # Arguments
///
/// * `input_path` - Path to the input image file (JPEG, PNG, WebP, BMP, TIFF)
/// * `config` - Configuration for execution provider, output format, and quality settings
///
/// # Returns
///
/// A `RemovalResult` containing the processed image, segmentation mask, and timing metadata
///
/// # Performance
///
/// Performance depends on the automatically selected model and execution provider:
/// - **CPU**: 2-5 seconds typical
/// - **GPU accelerated**: 100-500ms (CoreML/CUDA)
///
/// # Examples
///
/// ## Basic usage with default settings
/// ```rust,no_run
/// use bg_remove_core::{RemovalConfig, remove_background};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let result = remove_background("photo.jpg", &config).await?;
/// result.save_png("result.png")?;
/// # Ok(())
/// # }
/// ```
///
/// ## Custom execution provider and output format
/// ```rust,no_run
/// use bg_remove_core::{RemovalConfig, remove_background, ExecutionProvider, OutputFormat};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RemovalConfig::builder()
///     .execution_provider(ExecutionProvider::CoreMl)
///     .output_format(OutputFormat::Jpeg)
///     .jpeg_quality(95)
///     .build()?;
/// let result = remove_background("portrait.jpg", &config).await?;
/// result.save_jpeg("portrait_no_bg.jpg", config.jpeg_quality)?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `BgRemovalError` for:
/// - No embedded models available (requires building with embed-* features)
/// - Invalid input image format or corrupted files
/// - Execution provider setup failures
/// - Memory allocation or processing errors
///
/// # Note
///
/// For production use, prefer `remove_background_with_model()` with explicit model
/// selection for predictable performance and behavior.
pub async fn remove_background<P: AsRef<std::path::Path>>(
    input_path: P,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    let mut processor = ImageProcessor::new(config)?;
    processor.remove_background(input_path).await
}

/// Process a `DynamicImage` directly for background removal
///
/// This allows processing images already loaded into memory without file I/O overhead.
/// Useful for processing images from web requests, camera feeds, or other in-memory sources.
/// Uses the first available embedded model.
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
/// use bg_remove_core::{RemovalConfig, process_image};
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
/// use bg_remove_core::{RemovalConfig, process_image, ExecutionProvider};
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
/// use bg_remove_core::{RemovalConfig, process_image};
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
pub fn process_image(image: image::DynamicImage, config: &RemovalConfig) -> Result<RemovalResult> {
    let mut processor = ImageProcessor::new(config)?;
    processor.process_image(image)
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
/// use bg_remove_core::{RemovalConfig, segment_foreground};
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
/// use bg_remove_core::{RemovalConfig, segment_foreground, ExecutionProvider};
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
pub async fn segment_foreground<P: AsRef<std::path::Path>>(
    input_path: P,
    config: &RemovalConfig,
) -> Result<SegmentationMask> {
    let mut processor = ImageProcessor::new(config)?;
    processor.segment_foreground(input_path).await
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
/// use bg_remove_core::{RemovalConfig, segment_foreground, apply_segmentation_mask};
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
/// use bg_remove_core::{RemovalConfig, segment_foreground, apply_segmentation_mask};
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
/// use bg_remove_core::{RemovalConfig, apply_segmentation_mask, SegmentationMask};
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
pub async fn apply_segmentation_mask<P: AsRef<std::path::Path>>(
    input_path: P,
    mask: &SegmentationMask,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    let processor = ImageProcessor::new(config)?;
    processor.apply_mask(input_path, mask).await
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
