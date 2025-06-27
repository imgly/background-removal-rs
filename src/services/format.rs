//! Output format handling service
//!
//! This module separates output format conversion logic from business logic,
//! making the system more testable and maintainable.

use crate::{config::OutputFormat, error::Result};
use image::{DynamicImage, ImageBuffer, RgbaImage};

/// Service for handling output format conversions
pub struct OutputFormatHandler;

impl OutputFormatHandler {
    /// Convert an RGBA image to the specified output format
    ///
    /// # Arguments
    /// * `rgba_image` - Source RGBA image to convert
    /// * `format` - Target output format
    ///
    /// # Returns
    /// * `Ok(DynamicImage)` - Successfully converted image
    /// * `Err(BgRemovalError)` - Failed to convert image
    ///
    /// # Examples
    /// ```rust,no_run
    /// use bg_remove_core::{services::OutputFormatHandler, config::OutputFormat};
    /// use image::RgbaImage;
    ///
    /// let rgba_image = RgbaImage::new(100, 100);
    /// let result = OutputFormatHandler::convert_format(rgba_image, OutputFormat::Png)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn convert_format(rgba_image: RgbaImage, format: OutputFormat) -> Result<DynamicImage> {
        match format {
            OutputFormat::Png | OutputFormat::Rgba8 | OutputFormat::Tiff => {
                Ok(DynamicImage::ImageRgba8(rgba_image))
            },
            OutputFormat::Jpeg => {
                // Convert RGBA to RGB by dropping alpha channel
                let (width, height) = rgba_image.dimensions();
                let mut rgb_image = ImageBuffer::new(width, height);

                for (x, y, pixel) in rgba_image.enumerate_pixels() {
                    rgb_image.put_pixel(x, y, image::Rgb([pixel[0], pixel[1], pixel[2]]));
                }

                Ok(DynamicImage::ImageRgb8(rgb_image))
            },
            OutputFormat::WebP => Ok(DynamicImage::ImageRgba8(rgba_image)),
        }
    }

    /// Get the appropriate file extension for a given output format
    ///
    /// # Arguments
    /// * `format` - Output format
    ///
    /// # Returns
    /// String containing the file extension (without the dot)
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::{services::OutputFormatHandler, config::OutputFormat};
    ///
    /// assert_eq!(OutputFormatHandler::get_extension(OutputFormat::Png), "png");
    /// assert_eq!(OutputFormatHandler::get_extension(OutputFormat::Jpeg), "jpg");
    /// ```
    pub fn get_extension(format: OutputFormat) -> &'static str {
        match format {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpg",
            OutputFormat::WebP => "webp",
            OutputFormat::Tiff => "tiff",
            OutputFormat::Rgba8 => "raw",
        }
    }

    /// Check if a format supports transparency (alpha channel)
    ///
    /// # Arguments
    /// * `format` - Output format to check
    ///
    /// # Returns
    /// * `true` - Format supports transparency
    /// * `false` - Format does not support transparency
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::{services::OutputFormatHandler, config::OutputFormat};
    ///
    /// assert!(OutputFormatHandler::supports_transparency(OutputFormat::Png));
    /// assert!(!OutputFormatHandler::supports_transparency(OutputFormat::Jpeg));
    /// ```
    pub fn supports_transparency(format: OutputFormat) -> bool {
        match format {
            OutputFormat::Png | OutputFormat::WebP | OutputFormat::Tiff | OutputFormat::Rgba8 => {
                true
            },
            OutputFormat::Jpeg => false,
        }
    }

    /// Validate that a format is appropriate for background removal results
    ///
    /// # Arguments
    /// * `format` - Output format to validate
    ///
    /// # Note
    /// This function provides warnings for formats that don't support transparency,
    /// as they may not be ideal for background removal results.
    pub fn validate_for_background_removal(format: OutputFormat) {
        if !Self::supports_transparency(format) {
            log::warn!(
                "Output format {:?} does not support transparency. Background removal results may appear with a solid background.",
                format
            );
        }
    }

    /// Get the recommended quality settings for a format
    ///
    /// # Arguments
    /// * `format` - Output format
    ///
    /// # Returns
    /// Tuple of (default_quality, min_quality, max_quality) where applicable
    /// Returns None for formats that don't use quality settings
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::{services::OutputFormatHandler, config::OutputFormat};
    ///
    /// let (default, min, max) = OutputFormatHandler::get_quality_range(OutputFormat::Jpeg).unwrap();
    /// assert_eq!(default, 90);
    /// assert_eq!(min, 0);
    /// assert_eq!(max, 100);
    /// ```
    pub fn get_quality_range(format: OutputFormat) -> Option<(u8, u8, u8)> {
        match format {
            OutputFormat::Jpeg => Some((90, 0, 100)), // (default, min, max)
            OutputFormat::WebP => Some((85, 0, 100)),
            OutputFormat::Png | OutputFormat::Tiff | OutputFormat::Rgba8 => None, // Lossless
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn test_convert_format_png() {
        let rgba_image = RgbaImage::from_pixel(2, 2, Rgba([255, 0, 0, 255]));
        let result = OutputFormatHandler::convert_format(rgba_image, OutputFormat::Png);

        assert!(result.is_ok());
        let converted = result.unwrap();
        assert_eq!(converted.width(), 2);
        assert_eq!(converted.height(), 2);
    }

    #[test]
    fn test_convert_format_jpeg() {
        let rgba_image = RgbaImage::from_pixel(2, 2, Rgba([255, 0, 0, 128]));
        let result = OutputFormatHandler::convert_format(rgba_image, OutputFormat::Jpeg);

        assert!(result.is_ok());
        let converted = result.unwrap();
        assert_eq!(converted.width(), 2);
        assert_eq!(converted.height(), 2);

        // JPEG should be RGB, not RGBA
        match converted {
            DynamicImage::ImageRgb8(_) => {}, // Expected
            _ => panic!("Expected RGB8 image for JPEG format"),
        }
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(OutputFormatHandler::get_extension(OutputFormat::Png), "png");
        assert_eq!(
            OutputFormatHandler::get_extension(OutputFormat::Jpeg),
            "jpg"
        );
        assert_eq!(
            OutputFormatHandler::get_extension(OutputFormat::WebP),
            "webp"
        );
        assert_eq!(
            OutputFormatHandler::get_extension(OutputFormat::Tiff),
            "tiff"
        );
        assert_eq!(
            OutputFormatHandler::get_extension(OutputFormat::Rgba8),
            "raw"
        );
    }

    #[test]
    fn test_supports_transparency() {
        assert!(OutputFormatHandler::supports_transparency(
            OutputFormat::Png
        ));
        assert!(OutputFormatHandler::supports_transparency(
            OutputFormat::WebP
        ));
        assert!(OutputFormatHandler::supports_transparency(
            OutputFormat::Tiff
        ));
        assert!(OutputFormatHandler::supports_transparency(
            OutputFormat::Rgba8
        ));
        assert!(!OutputFormatHandler::supports_transparency(
            OutputFormat::Jpeg
        ));
    }

    #[test]
    fn test_validate_for_background_removal() {
        // Should complete for all formats but warn for JPEG
        OutputFormatHandler::validate_for_background_removal(OutputFormat::Png);
        OutputFormatHandler::validate_for_background_removal(OutputFormat::Jpeg);
        OutputFormatHandler::validate_for_background_removal(OutputFormat::WebP);
    }

    #[test]
    fn test_get_quality_range() {
        let jpeg_range = OutputFormatHandler::get_quality_range(OutputFormat::Jpeg);
        assert_eq!(jpeg_range, Some((90, 0, 100)));

        let webp_range = OutputFormatHandler::get_quality_range(OutputFormat::WebP);
        assert_eq!(webp_range, Some((85, 0, 100)));

        let png_range = OutputFormatHandler::get_quality_range(OutputFormat::Png);
        assert_eq!(png_range, None);
    }
}
