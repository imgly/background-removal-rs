//! ICC color profile extraction and handling

use crate::{
    error::{BgRemovalError, Result},
    types::ColorProfile,
};
use image::codecs::{jpeg::JpegDecoder, png::PngDecoder};
use image::ImageDecoder;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// ICC profile extractor for various image formats
pub struct ProfileExtractor;

impl ProfileExtractor {
    /// Extract ICC profile from an image file
    ///
    /// Attempts to extract ICC color profile from supported image formats.
    /// Uses format-specific decoders to access embedded ICC profile data.
    ///
    /// # Supported Formats
    /// - **JPEG/JPG**: Full ICC profile support via JpegDecoder
    /// - **PNG**: Full ICC profile support via PngDecoder
    /// - **Other formats**: Returns None (not supported in current implementation)
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// - `Ok(Some(ColorProfile))` - ICC profile found and extracted
    /// - `Ok(None)` - No ICC profile found or unsupported format
    /// - `Err(BgRemovalError)` - File I/O error or decoder error
    ///
    /// # Examples
    /// ```rust,no_run
    /// use bg_remove_core::color_profile::ProfileExtractor;
    ///
    /// # fn example() -> bg_remove_core::Result<()> {
    /// if let Some(profile) = ProfileExtractor::extract_from_image("photo.jpg")? {
    ///     println!("Found ICC profile: {} ({} bytes)", 
    ///         profile.color_space, profile.data_size());
    /// } else {
    ///     println!("No ICC profile found");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_from_image<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        let path = path.as_ref();
        
        // Determine format from file extension
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        match extension.as_deref() {
            Some("jpg") | Some("jpeg") => Self::extract_from_jpeg(path),
            Some("png") => Self::extract_from_png(path),
            Some("tiff") | Some("tif") => Self::extract_from_tiff(path),
            Some("webp") => Self::extract_from_webp(path),
            _ => Ok(None), // Unsupported format
        }
    }

    /// Extract ICC profile from JPEG image
    fn extract_from_jpeg<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut decoder = JpegDecoder::new(&mut reader).map_err(|e| {
            BgRemovalError::processing(format!("Failed to create JPEG decoder: {}", e))
        })?;

        if let Some(icc_data) = decoder.icc_profile() {
            Ok(Some(ColorProfile::from_icc_data(icc_data)))
        } else {
            Ok(None)
        }
    }

    /// Extract ICC profile from PNG image
    fn extract_from_png<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut decoder = PngDecoder::new(&mut reader).map_err(|e| {
            BgRemovalError::processing(format!("Failed to create PNG decoder: {}", e))
        })?;

        if let Some(icc_data) = decoder.icc_profile() {
            Ok(Some(ColorProfile::from_icc_data(icc_data)))
        } else {
            Ok(None)
        }
    }

    /// Extract ICC profile from TIFF image (placeholder implementation)
    fn extract_from_tiff<P: AsRef<Path>>(_path: P) -> Result<Option<ColorProfile>> {
        // TIFF ICC profile extraction not implemented yet
        // Would require TIFF-specific decoder with ICC support
        Ok(None)
    }

    /// Extract ICC profile from WebP image (placeholder implementation)
    fn extract_from_webp<P: AsRef<Path>>(_path: P) -> Result<Option<ColorProfile>> {
        // WebP ICC profile extraction not implemented yet
        // Would require WebP-specific decoder with ICC support
        Ok(None)
    }
}

/// ICC profile embedder for output images
pub struct ProfileEmbedder;

impl ProfileEmbedder {
    /// Embed ICC profile in output image (placeholder implementation)
    ///
    /// This is a placeholder for Phase 4 implementation.
    /// Current image crate 0.24.9 has limited ICC profile embedding support.
    ///
    /// # Arguments
    /// * `image` - The image to embed profile in
    /// * `profile` - The ICC profile to embed
    /// * `output_path` - Output file path
    /// * `format` - Output image format
    ///
    /// # Returns
    /// Currently returns an error indicating the feature is not implemented.
    /// Will be implemented in Phase 4 with proper encoder support.
    pub fn embed_in_output<P: AsRef<Path>>(
        _image: &image::DynamicImage,
        _profile: &ColorProfile,
        _output_path: P,
        _format: image::ImageFormat,
    ) -> Result<()> {
        Err(BgRemovalError::processing(
            "ICC profile embedding not yet implemented - Phase 4 feature"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ColorSpace;

    #[test]
    fn test_unsupported_format_returns_none() {
        // Test with a non-existent file with unsupported extension
        let result = ProfileExtractor::extract_from_image("test.bmp");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_nonexistent_file_returns_error() {
        // Test with non-existent JPEG file
        let result = ProfileExtractor::extract_from_image("nonexistent.jpg");
        assert!(result.is_err());
    }

    #[test]
    fn test_color_profile_creation() {
        let profile = ColorProfile::new(None, ColorSpace::Srgb);
        assert_eq!(profile.color_space, ColorSpace::Srgb);
        assert_eq!(profile.data_size(), 0);
    }

    #[test]
    fn test_color_profile_from_icc_data() {
        let icc_data = b"fake sRGB profile data".to_vec();
        let profile = ColorProfile::from_icc_data(icc_data.clone());
        assert_eq!(profile.data_size(), icc_data.len());
        // Should detect as sRGB due to "sRGB" in the fake data
        assert_eq!(profile.color_space, ColorSpace::Srgb);
    }

    #[test]
    fn test_embedding_not_implemented() {
        use image::DynamicImage;
        
        let image = DynamicImage::new_rgb8(1, 1);
        let profile = ColorProfile::new(None, ColorSpace::Srgb);
        
        let result = ProfileEmbedder::embed_in_output(
            &image,
            &profile, 
            "test.png",
            image::ImageFormat::Png
        );
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
    }
}