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

        // Try to get ICC profile - handle Result<Option<Vec<u8>>, ImageError>
        match decoder.icc_profile() {
            Ok(Some(icc_data)) => Ok(Some(ColorProfile::from_icc_data(icc_data))),
            Ok(None) => Ok(None),
            Err(e) => {
                log::debug!("Failed to extract ICC profile from JPEG: {e}");
                Ok(None)
            },
        }
    }

    /// Extract ICC profile from PNG image
    fn extract_from_png<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut decoder = PngDecoder::new(&mut reader).map_err(|e| {
            BgRemovalError::processing(format!("Failed to create PNG decoder: {e}"))
        })?;

        // Try to get ICC profile - handle Result<Option<Vec<u8>>, ImageError>
        match decoder.icc_profile() {
            Ok(Some(icc_data)) => Ok(Some(ColorProfile::from_icc_data(icc_data))),
            Ok(None) => Ok(None),
            Err(e) => {
                log::debug!("Failed to extract ICC profile from PNG: {e}");
                Ok(None)
            },
        }
    }

    /// Extract ICC profile from TIFF image (placeholder implementation)
    fn extract_from_tiff<P: AsRef<Path>>(_path: P) -> Option<ColorProfile> {
        // TIFF ICC profile extraction not implemented yet
        // Would require TIFF-specific decoder with ICC support
        None
    }

    /// Extract ICC profile from WebP image
    #[cfg(feature = "webp-support")]
    fn extract_from_webp<P: AsRef<Path>>(path: P) -> Result<Option<ColorProfile>> {
        // Try to use image-rs WebP decoder to extract ICC profile
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Try to create a WebP decoder using image-rs
        match image::codecs::webp::WebPDecoder::new(&mut reader) {
            Ok(mut decoder) => {
                // Try to get ICC profile - handle Result<Option<Vec<u8>>, ImageError>
                match decoder.icc_profile() {
                    Ok(Some(icc_data)) => {
                        log::debug!(
                            "Extracted ICC profile from WebP: {len} bytes",
                            len = icc_data.len()
                        );
                        Ok(Some(ColorProfile::from_icc_data(icc_data)))
                    },
                    Ok(None) => {
                        log::debug!("No ICC profile found in WebP file");
                        Ok(None)
                    },
                    Err(e) => {
                        log::debug!("Failed to extract ICC profile from WebP: {e}");
                        Ok(None)
                    },
                }
            },
            Err(e) => {
                log::debug!("Failed to create WebP decoder: {e}");
                Ok(None)
            },
        }
    }

    /// Fallback WebP profile extraction when webp feature is disabled
    #[cfg(not(feature = "webp-support"))]
    fn extract_from_webp<P: AsRef<Path>>(_path: P) -> Result<Option<ColorProfile>> {
        log::debug!("WebP support disabled - cannot extract ICC profile from WebP files");
        Ok(None)
    }
}

/// ICC profile embedder for output images using image-rs unified API
pub struct ProfileEmbedder;

impl ProfileEmbedder {
    /// Embed ICC profile in output image using image-rs 0.25.6 unified API
    ///
    /// Uses the standardized ImageEncoder::set_icc_profile() method for reliable ICC embedding
    pub fn embed_in_output<P: AsRef<Path>>(
        image: &image::DynamicImage,
        profile: &ColorProfile,
        output_path: P,
        format: image::ImageFormat,
        quality: u8,
    ) -> Result<()> {
        use image::{ExtendedColorType, ImageEncoder};
        use std::io::BufWriter;

        let output_path = output_path.as_ref();
        let file = File::create(output_path)?;
        let writer = BufWriter::new(file);

        match format {
            image::ImageFormat::Jpeg => {
                let rgb_image = image.to_rgb8();
                let mut encoder =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(writer, quality);

                // Set ICC profile if available
                if let Some(icc_data) = &profile.icc_data {
                    if let Err(e) = encoder.set_icc_profile(icc_data.clone()) {
                        log::debug!("Failed to embed ICC profile in JPEG: {e}");
                        log::debug!("Continuing without ICC profile");
                    } else {
                        log::debug!(
                            "Successfully set ICC profile for JPEG output ({} bytes)",
                            icc_data.len()
                        );
                    }
                }

                encoder.write_image(
                    rgb_image.as_raw(),
                    rgb_image.width(),
                    rgb_image.height(),
                    ExtendedColorType::Rgb8,
                )?;
            },
            image::ImageFormat::Png => {
                let rgba_image = image.to_rgba8();
                let mut encoder = image::codecs::png::PngEncoder::new(writer);

                // Set ICC profile if available
                if let Some(icc_data) = &profile.icc_data {
                    if let Err(e) = encoder.set_icc_profile(icc_data.clone()) {
                        log::debug!("Failed to embed ICC profile in PNG: {e}");
                        log::debug!("Continuing without ICC profile");
                    } else {
                        log::debug!(
                            "Successfully set ICC profile for PNG output ({} bytes)",
                            icc_data.len()
                        );
                    }
                }

                encoder.write_image(
                    rgba_image.as_raw(),
                    rgba_image.width(),
                    rgba_image.height(),
                    ExtendedColorType::Rgba8,
                )?;
            },
            #[cfg(feature = "webp-support")]
            image::ImageFormat::WebP => {
                let rgba_image = image.to_rgba8();

                // Always use image-rs WebP encoder (no external C dependencies)
                let mut encoder = if quality >= 100 {
                    image::codecs::webp::WebPEncoder::new_lossless(writer)
                } else {
                    // Note: image-rs WebP lossy encoder doesn't have quality parameter in constructor
                    // Using lossless for now to maintain ICC profile support
                    log::info!("Using lossless WebP encoding to preserve ICC profile");
                    image::codecs::webp::WebPEncoder::new_lossless(writer)
                };

                // Set ICC profile if available
                if let Some(icc_data) = &profile.icc_data {
                    if let Err(e) = encoder.set_icc_profile(icc_data.clone()) {
                        log::warn!("Failed to embed ICC profile in WebP: {e}");
                        log::warn!("WebP saved without ICC profile");
                    } else {
                        log::debug!(
                            "Successfully embedded ICC profile in lossless WebP ({} bytes)",
                            icc_data.len()
                        );
                    }
                }

                encoder.write_image(
                    rgba_image.as_raw(),
                    rgba_image.width(),
                    rgba_image.height(),
                    ExtendedColorType::Rgba8,
                )?;
            },
            #[cfg(not(feature = "webp-support"))]
            image::ImageFormat::WebP => {
                log::warn!("WebP support disabled - falling back to PNG format");
                return Self::embed_in_output(
                    image,
                    profile,
                    output_path,
                    image::ImageFormat::Png,
                    quality,
                );
            },
            image::ImageFormat::Tiff => {
                let rgba_image = image.to_rgba8();
                let mut encoder = image::codecs::tiff::TiffEncoder::new(writer);

                // Set ICC profile if available
                if let Some(icc_data) = &profile.icc_data {
                    if let Err(e) = encoder.set_icc_profile(icc_data.clone()) {
                        log::warn!("Failed to embed ICC profile in TIFF: {e}");
                        log::warn!("TIFF saved without ICC profile");
                    } else {
                        log::debug!(
                            "Successfully embedded ICC profile in TIFF ({} bytes)",
                            icc_data.len()
                        );
                    }
                }

                encoder.write_image(
                    rgba_image.as_raw(),
                    rgba_image.width(),
                    rgba_image.height(),
                    ExtendedColorType::Rgba8,
                )?;
            },
            _ => {
                // Fallback to image-rs save for other formats
                image.save_with_format(output_path, format)?;

                if profile.icc_data.is_some() {
                    log::debug!(
                        "ICC profile embedding not implemented for format: {:?}",
                        format
                    );
                    log::debug!("Image saved without ICC profile");
                }
            },
        }

        Ok(())
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
    fn test_embedding_requires_icc_data() {
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

        // Should succeed but without ICC profile
        assert!(result.is_ok());
    }
}
