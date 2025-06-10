//! # Background Removal Core Library
//!
//! A high-performance Rust library for background removal using ONNX Runtime and ISNet models.
//! 
//! This library provides efficient background removal capabilities with support for multiple
//! image formats and model configurations. It's designed to be 2-5x faster than JavaScript
//! implementations while maintaining API compatibility.
//!
//! ## Features
//!
//! - High-performance background removal using ISNet models
//! - Support for multiple image formats (JPEG, PNG, WebP)
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

pub mod config;
pub mod error;
pub mod image_processing;
pub mod inference;
pub mod models;
pub mod types;

// Public API exports
pub use config::{RemovalConfig, OutputFormat, ModelPrecision, ExecutionProvider};
pub use error::{BgRemovalError, Result};
pub use image_processing::{ImageProcessor, ProcessingOptions};
pub use inference::{InferenceBackend, OnnxBackend};
pub use types::{RemovalResult, SegmentationMask};

/// Remove background from an image file
///
/// This is the main entry point for background removal operations.
/// 
/// # Arguments
/// 
/// * `input_path` - Path to the input image file
/// * `config` - Configuration for the removal operation
///
/// # Returns
/// 
/// A `RemovalResult` containing the processed image and metadata
///
/// # Examples
/// 
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
pub async fn remove_background<P: AsRef<std::path::Path>>(
    input_path: P,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    let mut processor = ImageProcessor::new(config)?;
    processor.remove_background(input_path).await
}

/// Process a DynamicImage directly for background removal
///
/// This allows processing images loaded from memory or other sources.
/// 
/// # Arguments
/// 
/// * `image` - The input image to process
/// * `config` - Configuration for the removal operation
///
/// # Returns
/// 
/// A `RemovalResult` containing the processed image and metadata
///
/// # Examples
/// 
/// ```rust,no_run
/// use bg_remove_core::{RemovalConfig, process_image};
/// use image::DynamicImage;
/// 
/// # async fn example(img: DynamicImage) -> anyhow::Result<()> {
/// let config = RemovalConfig::default();
/// let result = process_image(img, &config)?;
/// result.save_png("result.png")?;
/// # Ok(())
/// # }
/// ```
pub fn process_image(
    image: image::DynamicImage,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    let mut processor = ImageProcessor::new(config)?;
    processor.process_image(image)
}

/// Extract foreground segmentation mask from an image
///
/// Returns only the segmentation mask without applying it to the image.
/// 
/// # Arguments
/// 
/// * `input_path` - Path to the input image file
/// * `config` - Configuration for the segmentation operation
///
/// # Returns
/// 
/// A `SegmentationMask` containing the binary mask data
pub async fn segment_foreground<P: AsRef<std::path::Path>>(
    input_path: P,
    config: &RemovalConfig,
) -> Result<SegmentationMask> {
    let mut processor = ImageProcessor::new(config)?;
    processor.segment_foreground(input_path).await
}

/// Apply a segmentation mask to an image
///
/// Applies an existing segmentation mask to an image to remove the background.
/// 
/// # Arguments
/// 
/// * `input_path` - Path to the input image file
/// * `mask` - The segmentation mask to apply
/// * `config` - Configuration for the operation
///
/// # Returns
/// 
/// A `RemovalResult` with the mask applied
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