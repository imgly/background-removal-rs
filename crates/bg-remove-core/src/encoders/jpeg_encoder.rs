//! JPEG encoder with ICC profile embedding support
//!
//! This module provides JPEG encoding with ICC color profile embedding using APP2 markers.
//! JPEG files embed ICC profiles using the APP2 application marker with the "ICC_PROFILE" identifier.

use crate::{
    error::{BgRemovalError, Result},
    types::ColorProfile,
};
use image::DynamicImage;
use std::path::Path;
// Removed unused imports

/// JPEG encoder with ICC profile embedding capability
pub struct JpegIccEncoder;

impl JpegIccEncoder {
    /// Encode image to JPEG with embedded ICC profile
    ///
    /// Creates a JPEG file with an embedded ICC color profile using APP2 markers.
    /// Large ICC profiles are split across multiple APP2 segments to comply with
    /// JPEG's 64KB segment size limit.
    ///
    /// # JPEG ICC Profile Format
    /// - APP2 marker (0xFFE2)
    /// - Segment length (2 bytes)
    /// - ICC_PROFILE identifier (12 bytes: "ICC_PROFILE\0")
    /// - Sequence number (1 byte: 1-based index)
    /// - Total sequences (1 byte: total number of segments)
    /// - Profile data (remaining bytes in segment)
    ///
    /// # Arguments
    /// * `image` - The image to encode
    /// * `profile` - The ICC profile to embed
    /// * `output_path` - Output file path
    /// * `quality` - JPEG quality (0-100)
    ///
    /// # Returns
    /// Result indicating success or failure of the encoding operation
    ///
    /// # Examples
    /// ```rust,no_run
    /// use bg_remove_core::{
    ///     encoders::jpeg_encoder::JpegIccEncoder,
    ///     types::ColorProfile,
    /// };
    /// use image::DynamicImage;
    ///
    /// # fn example(image: DynamicImage, profile: ColorProfile) -> bg_remove_core::Result<()> {
    /// JpegIccEncoder::encode_with_profile(&image, &profile, "output.jpg", 90)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_with_profile<P: AsRef<Path>>(
        image: &DynamicImage,
        profile: &ColorProfile,
        output_path: P,
        quality: u8,
    ) -> Result<()> {
        let output_path = output_path.as_ref();
        
        // Validate ICC profile data
        let icc_data = profile.icc_data.as_ref().ok_or_else(|| {
            BgRemovalError::processing("ICC profile has no data to embed")
        })?;

        // First, save JPEG without ICC profile to a temporary buffer
        let mut temp_jpeg = Vec::new();
        {
            let mut cursor = std::io::Cursor::new(&mut temp_jpeg);
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
            
            // Convert to RGB for JPEG (no alpha channel)
            let rgb_image = image.to_rgb8();
            encoder.encode(
                rgb_image.as_raw(),
                rgb_image.width(),
                rgb_image.height(),
                image::ColorType::Rgb8,
            ).map_err(|e| {
                BgRemovalError::processing(format!("Failed to encode JPEG: {}", e))
            })?;
        }

        // Now embed ICC profile into the JPEG data
        let jpeg_with_icc = Self::embed_icc_in_jpeg(&temp_jpeg, icc_data)?;

        // Write final JPEG to file
        std::fs::write(output_path, jpeg_with_icc).map_err(|e| {
            BgRemovalError::processing(format!("Failed to write JPEG file {}: {}", output_path.display(), e))
        })?;

        Ok(())
    }

    /// Embed ICC profile into JPEG data using APP2 markers
    fn embed_icc_in_jpeg(jpeg_data: &[u8], icc_data: &[u8]) -> Result<Vec<u8>> {
        // Find SOI (Start of Image) marker
        if jpeg_data.len() < 2 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8 {
            return Err(BgRemovalError::processing("Invalid JPEG: missing SOI marker"));
        }

        let mut result = Vec::new();
        
        // Copy SOI marker
        result.extend_from_slice(&jpeg_data[0..2]);

        // Generate ICC APP2 segments
        let icc_segments = Self::create_icc_app2_segments(icc_data)?;
        
        // Add ICC segments after SOI
        for segment in icc_segments {
            result.extend_from_slice(&segment);
        }

        // Copy rest of JPEG data (skip SOI)
        result.extend_from_slice(&jpeg_data[2..]);

        Ok(result)
    }

    /// Create APP2 segments containing ICC profile data
    fn create_icc_app2_segments(icc_data: &[u8]) -> Result<Vec<Vec<u8>>> {
        const APP2_MARKER: u16 = 0xFFE2;
        const ICC_IDENTIFIER: &[u8] = b"ICC_PROFILE\0";
        const MAX_SEGMENT_SIZE: usize = 65535; // Maximum JPEG segment size
        const HEADER_SIZE: usize = 2 + 2 + 12 + 1 + 1; // Marker + Length + Identifier + Seq + Total
        const MAX_ICC_DATA_PER_SEGMENT: usize = MAX_SEGMENT_SIZE - HEADER_SIZE;

        let mut segments = Vec::new();
        let total_segments = (icc_data.len() + MAX_ICC_DATA_PER_SEGMENT - 1) / MAX_ICC_DATA_PER_SEGMENT;
        
        if total_segments > 255 {
            return Err(BgRemovalError::processing(
                "ICC profile too large: requires more than 255 segments"
            ));
        }
        
        let total_segments = total_segments as u8;

        for (chunk_index, icc_chunk) in icc_data.chunks(MAX_ICC_DATA_PER_SEGMENT).enumerate() {
            let sequence_number = (chunk_index + 1) as u8; // 1-based indexing
            let segment_length = (ICC_IDENTIFIER.len() + 1 + 1 + icc_chunk.len() + 2) as u16;

            let mut segment = Vec::new();
            
            // APP2 marker
            segment.extend_from_slice(&APP2_MARKER.to_be_bytes());
            
            // Segment length (including length field itself)
            segment.extend_from_slice(&segment_length.to_be_bytes());
            
            // ICC_PROFILE identifier
            segment.extend_from_slice(ICC_IDENTIFIER);
            
            // Sequence number (1-based)
            segment.push(sequence_number);
            
            // Total number of segments
            segment.push(total_segments);
            
            // ICC profile data chunk
            segment.extend_from_slice(icc_chunk);

            segments.push(segment);
        }

        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ColorProfile, ColorSpace};

    #[test]
    fn test_jpeg_icc_encoder_creation() {
        // Test that we can create the encoder
        let _encoder = JpegIccEncoder;
    }

    #[test]
    fn test_create_icc_app2_segments() {
        let test_icc_data = vec![0u8; 1000]; // 1KB test ICC data
        let segments = JpegIccEncoder::create_icc_app2_segments(&test_icc_data);
        
        assert!(segments.is_ok());
        let segments = segments.unwrap();
        assert_eq!(segments.len(), 1); // Should fit in one segment
        
        let segment = &segments[0];
        // Check APP2 marker
        assert_eq!(segment[0], 0xFF);
        assert_eq!(segment[1], 0xE2);
        
        // Check ICC_PROFILE identifier
        assert_eq!(&segment[4..16], b"ICC_PROFILE\0");
        
        // Check sequence number and total
        assert_eq!(segment[16], 1); // First segment
        assert_eq!(segment[17], 1); // Total segments
    }

    #[test]
    fn test_create_large_icc_segments() {
        let test_icc_data = vec![0u8; 100_000]; // 100KB test ICC data (requires multiple segments)
        let segments = JpegIccEncoder::create_icc_app2_segments(&test_icc_data);
        
        assert!(segments.is_ok());
        let segments = segments.unwrap();
        assert!(segments.len() > 1); // Should require multiple segments
        
        // Check that sequence numbers are correct
        for (i, segment) in segments.iter().enumerate() {
            assert_eq!(segment[16], (i + 1) as u8); // Sequence number
            assert_eq!(segment[17], segments.len() as u8); // Total segments
        }
    }

    #[test]
    fn test_encode_with_profile_validates_input() {
        use image::{RgbImage, DynamicImage};
        
        // Create a test image
        let img = RgbImage::new(100, 100);
        let dynamic_img = DynamicImage::ImageRgb8(img);
        
        // Create profile without ICC data
        let profile = ColorProfile::new(None, ColorSpace::Srgb);
        
        // Should fail due to no ICC data
        let result = JpegIccEncoder::encode_with_profile(&dynamic_img, &profile, "test.jpg", 90);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no data to embed"));
    }

    #[test]
    fn test_embed_icc_validates_jpeg() {
        let invalid_jpeg = vec![0x00, 0x00]; // Not a valid JPEG
        let icc_data = vec![0u8; 100];
        
        let result = JpegIccEncoder::embed_icc_in_jpeg(&invalid_jpeg, &icc_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing SOI marker"));
    }
}