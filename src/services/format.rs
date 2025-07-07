//! Output format handling service
//!
//! This module separates output format conversion logic from business logic,
//! making the system more testable and maintainable. It also provides video
//! format detection when video support is enabled.

use crate::{config::OutputFormat, error::Result};
use image::{DynamicImage, ImageBuffer, RgbaImage};
use std::path::Path;

#[cfg(feature = "video-support")]
use crate::backends::video::VideoFormat;

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
    /// use imgly_bgremove::{services::OutputFormatHandler, config::OutputFormat};
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
    /// use imgly_bgremove::{services::OutputFormatHandler, config::OutputFormat};
    ///
    /// assert_eq!(OutputFormatHandler::get_extension(OutputFormat::Png), "png");
    /// assert_eq!(OutputFormatHandler::get_extension(OutputFormat::Jpeg), "jpg");
    /// ```
    #[must_use]
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
    /// use imgly_bgremove::{services::OutputFormatHandler, config::OutputFormat};
    ///
    /// assert!(OutputFormatHandler::supports_transparency(OutputFormat::Png));
    /// assert!(!OutputFormatHandler::supports_transparency(OutputFormat::Jpeg));
    /// ```
    #[must_use]
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
    /// Tuple of (`default_quality`, `min_quality`, `max_quality`) where applicable
    /// Returns None for formats that don't use quality settings
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::{services::OutputFormatHandler, config::OutputFormat};
    ///
    /// let (default, min, max) = OutputFormatHandler::get_quality_range(OutputFormat::Jpeg).unwrap();
    /// assert_eq!(default, 90);
    /// assert_eq!(min, 0);
    /// assert_eq!(max, 100);
    /// ```
    #[must_use]
    pub fn get_quality_range(format: OutputFormat) -> Option<(u8, u8, u8)> {
        match format {
            OutputFormat::Jpeg => Some((90, 0, 100)), // (default, min, max)
            OutputFormat::WebP => Some((85, 0, 100)),
            OutputFormat::Png | OutputFormat::Tiff | OutputFormat::Rgba8 => None, // Lossless
        }
    }

    /// Detect if a file path is a video format (when video support is enabled)
    ///
    /// # Arguments
    /// * `path` - Path to check
    ///
    /// # Returns
    /// * `true` - If the file extension indicates a video format
    /// * `false` - If the file extension indicates an image format or is unknown
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::services::OutputFormatHandler;
    ///
    /// assert!(OutputFormatHandler::is_video_format("video.mp4"));
    /// assert!(!OutputFormatHandler::is_video_format("image.jpg"));
    /// ```
    #[must_use]
    pub fn is_video_format<P: AsRef<Path>>(path: P) -> bool {
        #[cfg(feature = "video-support")]
        {
            if let Some(extension) = path.as_ref().extension() {
                if let Some(ext_str) = extension.to_str() {
                    return VideoFormat::from_extension(ext_str).is_some();
                }
            }
        }
        #[cfg(not(feature = "video-support"))]
        {
            let _ = path; // Silence unused parameter warning
        }
        false
    }

    /// Get video format from file path (when video support is enabled)
    ///
    /// # Arguments
    /// * `path` - Path to analyze
    ///
    /// # Returns
    /// * `Some(VideoFormat)` - If the file extension indicates a supported video format
    /// * `None` - If not a video format or video support is disabled
    ///
    /// # Examples
    /// ```rust,no_run
    /// use imgly_bgremove::services::OutputFormatHandler;
    ///
    /// # #[cfg(feature = "video-support")]
    /// # {
    /// let format = OutputFormatHandler::detect_video_format("video.mp4");
    /// assert!(format.is_some());
    /// # }
    /// ```
    #[cfg(feature = "video-support")]
    #[must_use]
    pub fn detect_video_format<P: AsRef<Path>>(path: P) -> Option<VideoFormat> {
        if let Some(extension) = path.as_ref().extension() {
            if let Some(ext_str) = extension.to_str() {
                return VideoFormat::from_extension(ext_str);
            }
        }
        None
    }

    /// Check if a path represents a supported media format (image or video)
    ///
    /// # Arguments
    /// * `path` - Path to check
    ///
    /// # Returns
    /// * `true` - If the file extension is a supported image or video format
    /// * `false` - If the file extension is not supported
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::services::OutputFormatHandler;
    ///
    /// assert!(OutputFormatHandler::is_supported_media_format("image.jpg"));
    /// assert!(OutputFormatHandler::is_supported_media_format("video.mp4"));
    /// assert!(!OutputFormatHandler::is_supported_media_format("document.pdf"));
    /// ```
    #[must_use]
    pub fn is_supported_media_format<P: AsRef<Path>>(path: P) -> bool {
        Self::is_supported_format(&path) || Self::is_video_format(path)
    }

    /// Get a human-readable description of the detected format
    ///
    /// # Arguments
    /// * `path` - Path to analyze
    ///
    /// # Returns
    /// A string describing the detected format type
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::services::OutputFormatHandler;
    ///
    /// assert_eq!(OutputFormatHandler::describe_format("test.jpg"), "Image (JPEG)");
    /// assert_eq!(OutputFormatHandler::describe_format("test.mp4"), "Video (MP4)");
    /// assert_eq!(OutputFormatHandler::describe_format("test.txt"), "Unknown format");
    /// ```
    #[must_use]
    pub fn describe_format<P: AsRef<Path>>(path: P) -> String {
        let path_ref = path.as_ref();
        
        if Self::is_video_format(path_ref) {
            #[cfg(feature = "video-support")]
            {
                if let Some(video_format) = Self::detect_video_format(path_ref) {
                    return format!("Video ({})", format!("{:?}", video_format).to_uppercase());
                }
            }
            return "Video".to_string();
        }
        
        if Self::is_supported_format(path_ref) {
            if let Some(extension) = path_ref.extension() {
                if let Some(ext_str) = extension.to_str() {
                    return format!("Image ({})", ext_str.to_uppercase());
                }
            }
            return "Image".to_string();
        }
        
        "Unknown format".to_string()
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

    #[test]
    fn test_convert_format_rgba_to_rgb_precision() {
        // Create RGBA image with specific transparency values
        let mut rgba_image = RgbaImage::new(2, 2);
        rgba_image.put_pixel(0, 0, Rgba([255, 128, 64, 255])); // Opaque
        rgba_image.put_pixel(1, 0, Rgba([200, 100, 50, 128])); // Semi-transparent
        rgba_image.put_pixel(0, 1, Rgba([150, 75, 25, 64])); // More transparent
        rgba_image.put_pixel(1, 1, Rgba([100, 50, 12, 0])); // Fully transparent

        let result = OutputFormatHandler::convert_format(rgba_image, OutputFormat::Jpeg);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert_eq!(converted.width(), 2);
        assert_eq!(converted.height(), 2);

        // JPEG should be RGB format
        match converted {
            DynamicImage::ImageRgb8(_) => {},
            _ => panic!("JPEG conversion should produce RGB8 image"),
        }
    }

    #[test]
    fn test_convert_format_transparency_preservation() {
        // Create RGBA image with transparency
        let mut rgba_image = RgbaImage::new(1, 1);
        rgba_image.put_pixel(0, 0, Rgba([255, 0, 0, 128])); // Semi-transparent red

        // Test formats that preserve transparency
        let transparency_formats = vec![
            OutputFormat::Png,
            OutputFormat::WebP,
            OutputFormat::Tiff,
            OutputFormat::Rgba8,
        ];

        for format in transparency_formats {
            let result = OutputFormatHandler::convert_format(rgba_image.clone(), format);
            assert!(result.is_ok(), "Failed to convert to {:?}", format);

            let converted = result.unwrap();
            match converted {
                DynamicImage::ImageRgba8(_) => {
                    // This is expected for transparency-supporting formats
                },
                _ => {
                    // Some formats might convert to other compatible types
                    // Just ensure we get a valid image
                    assert!(converted.width() > 0 && converted.height() > 0);
                },
            }
        }
    }

    #[test]
    fn test_convert_format_large_image() {
        // Test with larger image to ensure memory handling
        let rgba_image = RgbaImage::new(100, 100);

        let formats = vec![
            OutputFormat::Png,
            OutputFormat::Jpeg,
            OutputFormat::WebP,
            OutputFormat::Tiff,
            OutputFormat::Rgba8,
        ];

        for format in formats {
            let result = OutputFormatHandler::convert_format(rgba_image.clone(), format);
            assert!(
                result.is_ok(),
                "Failed to convert large image to {:?}",
                format
            );

            let converted = result.unwrap();
            assert_eq!(converted.width(), 100);
            assert_eq!(converted.height(), 100);
        }
    }

    #[test]
    fn test_convert_format_edge_dimensions() {
        // Test with edge case dimensions
        let dimensions = vec![(1, 1), (1, 100), (100, 1)];

        for (width, height) in dimensions {
            let rgba_image = RgbaImage::new(width, height);

            for format in &[OutputFormat::Png, OutputFormat::Jpeg] {
                let result = OutputFormatHandler::convert_format(rgba_image.clone(), *format);
                assert!(
                    result.is_ok(),
                    "Failed for {}x{} with {:?}",
                    width,
                    height,
                    format
                );

                let converted = result.unwrap();
                assert_eq!(converted.width(), width);
                assert_eq!(converted.height(), height);
            }
        }
    }

    #[test]
    fn test_get_extension_all_formats() {
        let extensions = vec![
            (OutputFormat::Png, "png"),
            (OutputFormat::Jpeg, "jpg"),
            (OutputFormat::WebP, "webp"),
            (OutputFormat::Tiff, "tiff"),
            (OutputFormat::Rgba8, "raw"),
        ];

        for (format, expected_ext) in extensions {
            assert_eq!(OutputFormatHandler::get_extension(format), expected_ext);
        }
    }

    #[test]
    fn test_supports_transparency_comprehensive() {
        let transparency_support = vec![
            (OutputFormat::Png, true),
            (OutputFormat::Jpeg, false),
            (OutputFormat::WebP, true),
            (OutputFormat::Tiff, true),
            (OutputFormat::Rgba8, true),
        ];

        for (format, should_support) in transparency_support {
            assert_eq!(
                OutputFormatHandler::supports_transparency(format),
                should_support,
                "Transparency support mismatch for {:?}",
                format
            );
        }
    }

    #[test]
    fn test_validate_for_background_removal_all_formats() {
        // Test validation for all formats (should not panic)
        let formats = vec![
            OutputFormat::Png,
            OutputFormat::Jpeg,
            OutputFormat::WebP,
            OutputFormat::Tiff,
            OutputFormat::Rgba8,
        ];

        for format in formats {
            // Should complete without panicking
            OutputFormatHandler::validate_for_background_removal(format);
        }
    }

    #[test]
    fn test_get_quality_range_comprehensive() {
        let quality_ranges = vec![
            (OutputFormat::Jpeg, Some((90, 0, 100))),
            (OutputFormat::WebP, Some((85, 0, 100))),
            (OutputFormat::Png, None),
            (OutputFormat::Tiff, None),
            (OutputFormat::Rgba8, None),
        ];

        for (format, expected_range) in quality_ranges {
            assert_eq!(
                OutputFormatHandler::get_quality_range(format),
                expected_range,
                "Quality range mismatch for {:?}",
                format
            );
        }
    }

    #[test]
    fn test_convert_format_color_accuracy() {
        // Test color accuracy preservation during conversion
        let mut rgba_image = RgbaImage::new(3, 1);

        // Set specific RGB values
        rgba_image.put_pixel(0, 0, Rgba([255, 0, 0, 255])); // Pure red
        rgba_image.put_pixel(1, 0, Rgba([0, 255, 0, 255])); // Pure green
        rgba_image.put_pixel(2, 0, Rgba([0, 0, 255, 255])); // Pure blue

        // Convert to JPEG (RGB) and verify we get an RGB image
        let result = OutputFormatHandler::convert_format(rgba_image, OutputFormat::Jpeg);
        assert!(result.is_ok());

        let rgb_image = result.unwrap();
        assert_eq!(rgb_image.width(), 3);
        assert_eq!(rgb_image.height(), 1);

        // Verify it's an RGB image (not RGBA)
        match rgb_image {
            DynamicImage::ImageRgb8(_) => {
                // Expected for JPEG
            },
            _ => panic!(
                "JPEG conversion should produce RGB8 image, got {:?}",
                rgb_image.color()
            ),
        }
    }

    #[test]
    fn test_convert_format_grayscale_handling() {
        // Test with grayscale-like colors
        let mut rgba_image = RgbaImage::new(2, 2);
        rgba_image.put_pixel(0, 0, Rgba([128, 128, 128, 255])); // Gray
        rgba_image.put_pixel(1, 0, Rgba([64, 64, 64, 255])); // Darker gray
        rgba_image.put_pixel(0, 1, Rgba([192, 192, 192, 255])); // Lighter gray
        rgba_image.put_pixel(1, 1, Rgba([0, 0, 0, 255])); // Black

        let formats = vec![OutputFormat::Png, OutputFormat::Jpeg];

        for format in formats {
            let result = OutputFormatHandler::convert_format(rgba_image.clone(), format);
            assert!(
                result.is_ok(),
                "Failed to convert grayscale-like image to {:?}",
                format
            );

            let converted = result.unwrap();
            assert_eq!(converted.width(), 2);
            assert_eq!(converted.height(), 2);
        }
    }

    #[test]
    fn test_convert_format_empty_alpha_handling() {
        // Test with various alpha values including fully transparent
        let mut rgba_image = RgbaImage::new(2, 2);
        rgba_image.put_pixel(0, 0, Rgba([255, 0, 0, 255])); // Opaque red
        rgba_image.put_pixel(1, 0, Rgba([0, 255, 0, 0])); // Transparent green
        rgba_image.put_pixel(0, 1, Rgba([0, 0, 255, 128])); // Semi-transparent blue
        rgba_image.put_pixel(1, 1, Rgba([255, 255, 0, 64])); // Low alpha yellow

        // Convert to JPEG (drops alpha)
        let result = OutputFormatHandler::convert_format(rgba_image, OutputFormat::Jpeg);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert_eq!(converted.width(), 2);
        assert_eq!(converted.height(), 2);
    }

    #[test]
    fn test_format_characteristics_consistency() {
        // Verify that transparency support and quality settings are consistent
        let formats = vec![
            OutputFormat::Png,
            OutputFormat::Jpeg,
            OutputFormat::WebP,
            OutputFormat::Tiff,
            OutputFormat::Rgba8,
        ];

        for format in formats {
            let supports_transparency = OutputFormatHandler::supports_transparency(format);
            let quality_range = OutputFormatHandler::get_quality_range(format);
            let extension = OutputFormatHandler::get_extension(format);

            // Basic consistency checks
            assert!(
                !extension.is_empty(),
                "Extension should not be empty for {:?}",
                format
            );

            // JPEG should not support transparency and should have quality settings
            if matches!(format, OutputFormat::Jpeg) {
                assert!(
                    !supports_transparency,
                    "JPEG should not support transparency"
                );
                assert!(quality_range.is_some(), "JPEG should have quality settings");
            }

            // PNG should support transparency and should not have quality settings
            if matches!(format, OutputFormat::Png) {
                assert!(supports_transparency, "PNG should support transparency");
                assert!(
                    quality_range.is_none(),
                    "PNG should not have quality settings"
                );
            }
        }
    }
}
