//! Image I/O operations service
//!
//! This module separates file I/O operations from business logic,
//! making the system more testable and maintainable.

use crate::{
    config::OutputFormat,
    error::{BgRemovalError, Result},
    types::ColorProfile,
};
use image::DynamicImage;
use std::path::Path;

/// Service for handling image file input/output operations
pub struct ImageIOService;

impl ImageIOService {
    /// Load an image from a file path
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// * `Ok(DynamicImage)` - Successfully loaded image
    /// * `Err(BgRemovalError)` - Failed to load image
    ///
    /// # Examples
    /// ```rust,no_run
    /// use bg_remove_core::services::ImageIOService;
    ///
    /// let image = ImageIOService::load_image("input.jpg")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        let path_ref = path.as_ref();

        // Check if file exists
        if !path_ref.exists() {
            return Err(BgRemovalError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Image file does not exist: {}", path_ref.display()),
            )));
        }

        // Load the image
        image::open(path_ref).map_err(|e| {
            BgRemovalError::processing(format!(
                "Failed to load image from {}: {}",
                path_ref.display(),
                e
            ))
        })
    }

    /// Save an image to a file with the specified format and color profile preservation
    ///
    /// # Arguments
    /// * `image` - The image to save
    /// * `path` - Output file path
    /// * `format` - Output format specification
    /// * `preserve_color_profiles` - Whether to preserve color profiles
    ///
    /// # Returns
    /// * `Ok(())` - Successfully saved image
    /// * `Err(BgRemovalError)` - Failed to save image
    ///
    /// # Examples
    /// ```rust,no_run
    /// use bg_remove_core::{services::ImageIOService, config::OutputFormat};
    /// use image::DynamicImage;
    ///
    /// # let image = DynamicImage::new_rgb8(100, 100);
    /// ImageIOService::save_image(
    ///     &image,
    ///     "output.png",
    ///     OutputFormat::Png,
    ///     true
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn save_image<P: AsRef<Path>>(
        image: &DynamicImage,
        path: P,
        format: OutputFormat,
        preserve_color_profiles: bool,
    ) -> Result<()> {
        let path_ref = path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path_ref.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                BgRemovalError::processing(format!(
                    "Failed to create output directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        // Save based on format
        let result = match format {
            OutputFormat::Png => image.save_with_format(path_ref, image::ImageFormat::Png),
            OutputFormat::Jpeg => image.save_with_format(path_ref, image::ImageFormat::Jpeg),
            OutputFormat::WebP => image.save_with_format(path_ref, image::ImageFormat::WebP),
            OutputFormat::Tiff => image.save_with_format(path_ref, image::ImageFormat::Tiff),
            OutputFormat::Rgba8 => {
                // For raw RGBA8, we need to handle this specially
                let rgba8 = image.to_rgba8();
                std::fs::write(path_ref, rgba8.as_raw()).map_err(|e| {
                    BgRemovalError::processing(format!(
                        "Failed to write RGBA8 data to {}: {}",
                        path_ref.display(),
                        e
                    ))
                })?;
                return Ok(());
            },
        };

        result.map_err(|e| {
            BgRemovalError::processing(format!(
                "Failed to save image to {}: {}",
                path_ref.display(),
                e
            ))
        })?;

        // TODO: Implement color profile preservation if requested
        if preserve_color_profiles {
            // This would involve extracting and re-embedding color profiles
            // For now, we'll just log that this feature is requested
            log::debug!(
                "Color profile preservation requested for {}",
                path_ref.display()
            );
        }

        Ok(())
    }

    /// Extract color profile from an image file
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// * `Ok(Option<ColorProfile>)` - Successfully extracted profile (if present)
    /// * `Err(BgRemovalError)` - Failed to read file or extract profile
    pub fn extract_color_profile<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        let _path_ref = path.as_ref();

        // TODO: Implement actual color profile extraction
        // This would involve reading ICC profiles from image metadata
        log::debug!("Color profile extraction not yet implemented");
        Ok(None)
    }

    /// Check if a file path has a supported image extension
    ///
    /// # Arguments
    /// * `path` - Path to check
    ///
    /// # Returns
    /// * `true` - If the file extension is supported
    /// * `false` - If the file extension is not supported or missing
    pub fn is_supported_format<P: AsRef<Path>>(path: P) -> bool {
        let path_ref = path.as_ref();

        if let Some(extension) = path_ref.extension() {
            if let Some(ext_str) = extension.to_str() {
                let ext_lower = ext_str.to_lowercase();
                matches!(
                    ext_lower.as_str(),
                    "jpg" | "jpeg" | "png" | "webp" | "tiff" | "tif" | "bmp"
                )
            } else {
                false
            }
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_is_supported_format() {
        assert!(ImageIOService::is_supported_format("test.jpg"));
        assert!(ImageIOService::is_supported_format("test.jpeg"));
        assert!(ImageIOService::is_supported_format("test.png"));
        assert!(ImageIOService::is_supported_format("test.webp"));
        assert!(ImageIOService::is_supported_format("test.tiff"));
        assert!(ImageIOService::is_supported_format("test.tif"));
        assert!(ImageIOService::is_supported_format("test.bmp"));

        assert!(!ImageIOService::is_supported_format("test.txt"));
        assert!(!ImageIOService::is_supported_format("test.pdf"));
        assert!(!ImageIOService::is_supported_format("test"));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = ImageIOService::load_image("nonexistent.jpg");
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(e.to_string().contains("does not exist"));
        }
    }

    #[test]
    fn test_save_image_creates_directory() {
        let temp_dir = tempdir().unwrap();
        let nested_path = temp_dir.path().join("nested").join("dir").join("test.png");

        // Create a simple 1x1 image
        let image = DynamicImage::new_rgb8(1, 1);

        let result = ImageIOService::save_image(&image, &nested_path, OutputFormat::Png, false);

        assert!(result.is_ok());
        assert!(nested_path.exists());
    }
}
