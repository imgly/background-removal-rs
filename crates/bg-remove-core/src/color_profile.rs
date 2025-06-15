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
    /// - **JPEG/JPG**: Full ICC profile support via `JpegDecoder`
    /// - **PNG**: Full ICC profile support via `PngDecoder`
    /// - **Other formats**: Returns None (not supported in current implementation)
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// - `Ok(Some(ColorProfile))` - ICC profile found and extracted
    /// - `Ok(None)` - No ICC profile found or unsupported format
    ///
    /// # Errors
    /// - File I/O error when reading the image file
    /// - Decoder error when parsing the image format
    /// - Invalid ICC profile data
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
            .map(str::to_lowercase);

        match extension.as_deref() {
            Some("jpg" | "jpeg") => Self::extract_from_jpeg(path),
            Some("png") => Self::extract_from_png(path),
            Some("tiff" | "tif") => Ok(Self::extract_from_tiff(path)),
            Some("webp") => Self::extract_from_webp(path),
            _ => Ok(None), // Unsupported format
        }
    }

    /// Extract ICC profile from JPEG image
    fn extract_from_jpeg<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut decoder = JpegDecoder::new(&mut reader).map_err(|e| {
            BgRemovalError::processing(format!("Failed to create JPEG decoder: {e}"))
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
            BgRemovalError::processing(format!("Failed to create PNG decoder: {e}"))
        })?;

        if let Some(icc_data) = decoder.icc_profile() {
            Ok(Some(ColorProfile::from_icc_data(icc_data)))
        } else {
            Ok(None)
        }
    }

    /// Extract ICC profile from TIFF image (placeholder implementation)
    fn extract_from_tiff<P: AsRef<Path>>(_path: P) -> Option<ColorProfile> {
        // TIFF ICC profile extraction not implemented yet
        // Would require TIFF-specific decoder with ICC support
        None
    }

    /// Extract ICC profile from WebP image
    fn extract_from_webp<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        use crate::encoders::webp_encoder::WebPIccEncoder;

        let path = path.as_ref();
        let webp_data = std::fs::read(path).map_err(|e| {
            BgRemovalError::processing(format!("Failed to read WebP file {}: {e}", path.display()))
        })?;

        if let Some(icc_data) = WebPIccEncoder::extract_icc_profile(&webp_data)? {
            log::debug!(
                "Extracted ICC profile from WebP: {len} bytes",
                len = icc_data.len()
            );
            Ok(Some(ColorProfile::from_icc_data(icc_data)))
        } else {
            log::debug!("No ICC profile found in WebP file");
            Ok(None)
        }
    }
}

/// ICC profile embedder for output images
pub struct ProfileEmbedder;

impl ProfileEmbedder {
    /// Embed ICC profile in output image
    ///
    /// Embeds ICC color profiles in output images using format-specific encoders.
    /// Supports PNG (via iCCP chunks) and JPEG (via APP2 markers) formats.
    ///
    /// # Supported Formats
    /// - **PNG**: Embeds using custom iCCP chunks
    /// - **JPEG**: Embeds using APP2 markers with custom encoder
    /// - **WebP**: Embeds using ICCP chunks in RIFF container
    /// - **Other formats**: Returns error (not supported)
    ///
    /// # Arguments
    /// * `image` - The image to embed profile in
    /// * `profile` - The ICC profile to embed
    /// * `output_path` - Output file path
    /// * `format` - Output image format
    /// * `quality` - Quality setting (used for JPEG, 0-100)
    ///
    /// # Returns
    /// Result indicating success or failure of the embedding operation
    ///
    /// # Errors
    /// - Unsupported output format (only PNG, JPEG, WebP are supported)
    /// - File I/O errors when writing the output image
    /// - Image encoding errors from the underlying format encoders
    /// - Color profile has no data to embed (empty ICC profile)
    ///
    /// # Examples
    /// ```rust,no_run
    /// use bg_remove_core::{
    ///     color_profile::ProfileEmbedder,
    ///     types::ColorProfile,
    /// };
    /// use image::{DynamicImage, ImageFormat};
    ///
    /// # fn example(image: DynamicImage, profile: ColorProfile) -> bg_remove_core::Result<()> {
    /// ProfileEmbedder::embed_in_output(&image, &profile, "output.png", ImageFormat::Png, 0)?;
    /// ProfileEmbedder::embed_in_output(&image, &profile, "output.jpg", ImageFormat::Jpeg, 90)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_in_output<P: AsRef<Path>>(
        image: &image::DynamicImage,
        profile: &ColorProfile,
        output_path: P,
        format: image::ImageFormat,
        quality: u8,
    ) -> Result<()> {
        use crate::encoders::{
            jpeg_encoder::JpegIccEncoder, png_encoder::PngIccEncoder, webp_encoder::WebPIccEncoder,
        };

        match format {
            image::ImageFormat::Png => {
                PngIccEncoder::encode_with_profile(image, profile, output_path)
            },
            image::ImageFormat::Jpeg => {
                JpegIccEncoder::encode_with_profile(image, profile, output_path, quality)
            },
            image::ImageFormat::WebP => {
                WebPIccEncoder::encode_with_profile(image, profile, output_path, quality)
            },
            _ => {
                Err(BgRemovalError::processing(format!(
                    "ICC profile embedding not supported for format: {format:?}. Supported formats: PNG, JPEG, WebP"
                )))
            }
        }
    }

    /// Embed ICC profile in PNG output
    ///
    /// Convenience method for PNG-specific ICC embedding.
    ///
    /// # Arguments
    /// * `image` - The image to embed profile in
    /// * `profile` - The ICC profile to embed
    /// * `output_path` - Output PNG file path
    ///
    /// # Errors
    /// - File I/O errors when writing the PNG file
    /// - PNG encoding errors from the underlying encoder
    /// - Color profile has no data to embed (empty ICC profile)
    pub fn embed_in_png<P: AsRef<Path>>(
        image: &image::DynamicImage,
        profile: &ColorProfile,
        output_path: P,
    ) -> Result<()> {
        use crate::encoders::png_encoder::PngIccEncoder;
        PngIccEncoder::encode_with_profile(image, profile, output_path)
    }

    /// Embed ICC profile in JPEG output
    ///
    /// Convenience method for JPEG-specific ICC embedding.
    ///
    /// # Arguments
    /// * `image` - The image to embed profile in
    /// * `profile` - The ICC profile to embed
    /// * `output_path` - Output JPEG file path
    /// * `quality` - JPEG quality (0-100)
    ///
    /// # Errors
    /// - File I/O errors when writing the JPEG file
    /// - JPEG encoding errors from the underlying encoder
    /// - Color profile has no data to embed (empty ICC profile)
    pub fn embed_in_jpeg<P: AsRef<Path>>(
        image: &image::DynamicImage,
        profile: &ColorProfile,
        output_path: P,
        quality: u8,
    ) -> Result<()> {
        use crate::encoders::jpeg_encoder::JpegIccEncoder;
        JpegIccEncoder::encode_with_profile(image, profile, output_path, quality)
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
            image::ImageFormat::Png,
            80,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no data to embed"));
    }
}
