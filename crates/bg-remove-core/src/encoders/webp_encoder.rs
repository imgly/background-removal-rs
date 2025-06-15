//! WebP encoder with ICC profile embedding support
//!
//! This module provides WebP encoding with ICC color profile embedding using the ICCP chunk.
//! Implements manual ICCP chunk creation and insertion according to the WebP/RIFF specification
//! to ensure proper ICC profile embedding in WebP files.

use crate::{
    error::{BgRemovalError, Result},
    types::ColorProfile,
};
use image::DynamicImage;
use std::path::Path;
use std::io::{Cursor, Write};

/// WebP encoder with ICC profile embedding capability
pub struct WebPIccEncoder;

impl WebPIccEncoder {
    /// Encode image to WebP with embedded ICC profile
    ///
    /// Creates a WebP file with an embedded ICC color profile using the ICCP chunk.
    /// Implements manual ICCP chunk creation and insertion according to WebP/RIFF specification.
    ///
    /// # Arguments
    /// * `image` - The image to encode
    /// * `profile` - The ICC profile to embed
    /// * `output_path` - Output file path
    /// * `quality` - WebP compression quality (0-100)
    ///
    /// # Returns
    /// Result indicating success or failure of the encoding operation
    ///
    /// # Errors
    /// - Color profile has no ICC data to embed
    /// - WebP encoding errors from the underlying library
    /// - File I/O errors when writing the output file
    /// - Invalid WebP data format when embedding ICC profile
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

        log::info!(
            "Embedding ICC color profile in WebP: {} ({} bytes)",
            profile.color_space,
            profile.data_size()
        );

        // Step 1: Create standard WebP without ICC profile
        let mut webp_buffer = Vec::new();
        {
            let rgb_image = image.to_rgb8();
            let mut cursor = Cursor::new(&mut webp_buffer);
            
            // Use webp crate for basic encoding
            let encoder = webp::Encoder::from_rgb(&rgb_image, rgb_image.width(), rgb_image.height());
            let webp_data = encoder.encode(f32::from(quality));
            cursor.write_all(&webp_data).map_err(|e| {
                BgRemovalError::processing(format!("Failed to create WebP buffer: {e}"))
            })?;
        }

        // Step 2: Insert ICCP chunk into WebP
        let webp_with_icc = Self::insert_iccp_chunk(&webp_buffer, icc_data)?;

        // Step 3: Write final WebP to file
        std::fs::write(output_path, webp_with_icc).map_err(|e| {
            BgRemovalError::processing(format!("Failed to write WebP file: {e}"))
        })?;

        log::info!("Successfully created WebP with embedded ICC profile: {}", output_path.display());
        Ok(())
    }

    /// Insert ICCP chunk into WebP data
    ///
    /// Creates an ICCP chunk with ICC profile data and inserts it into the WebP/RIFF
    /// file structure before the image data, following WebP container specification.
    ///
    /// # Arguments
    /// * `webp_data` - Original WebP file data
    /// * `icc_data` - Raw ICC profile data to embed
    ///
    /// # Returns
    /// WebP data with embedded ICCP chunk
    fn insert_iccp_chunk(webp_data: &[u8], icc_data: &[u8]) -> Result<Vec<u8>> {
        // Validate RIFF/WebP signature
        if webp_data.len() < 12 || webp_data.get(0..4) != Some(b"RIFF") || webp_data.get(8..12) != Some(b"WEBP") {
            return Err(BgRemovalError::processing("Invalid WebP/RIFF signature"));
        }

        let mut result = Vec::new();
        let mut pos = 0;

        // Copy RIFF header (12 bytes: "RIFF" + size + "WEBP")
        result.extend_from_slice(&webp_data[0..12]);
        pos += 12;

        // Create ICCP chunk
        let iccp_chunk = Self::create_iccp_chunk(icc_data);
        let mut iccp_inserted = false;

        // Parse and copy chunks, inserting ICCP early in the structure
        while pos < webp_data.len() {
            if pos + 8 > webp_data.len() {
                break;
            }

            // Read chunk header (4 bytes fourcc + 4 bytes size)
            let chunk_fourcc = webp_data.get(pos..pos + 4)
                .ok_or_else(|| BgRemovalError::processing("Truncated WebP: incomplete chunk fourCC"))?;
            let chunk_size_bytes = webp_data.get(pos + 4..pos + 8)
                .and_then(|bytes| bytes.try_into().ok())
                .ok_or_else(|| BgRemovalError::processing("Truncated WebP: incomplete chunk size"))?;
            let chunk_size = u32::from_le_bytes(chunk_size_bytes) as usize;

            // Insert ICCP chunk before VP8 or VP8L data
            if (chunk_fourcc == b"VP8 " || chunk_fourcc == b"VP8L") && !iccp_inserted {
                result.extend_from_slice(&iccp_chunk);
                iccp_inserted = true;
                log::debug!("Inserted ICCP chunk before {} chunk", String::from_utf8_lossy(chunk_fourcc));
            }

            // Copy current chunk (header + data + optional padding)
            let chunk_total_size = 8 + chunk_size + (chunk_size % 2); // RIFF chunks must be word-aligned
            if pos + chunk_total_size > webp_data.len() {
                // Handle case where chunk extends beyond file
                let remaining = webp_data.len() - pos;
                result.extend_from_slice(&webp_data[pos..pos + remaining]);
                break;
            }
            result.extend_from_slice(&webp_data[pos..pos + chunk_total_size]);
            pos += chunk_total_size;
        }

        if !iccp_inserted {
            return Err(BgRemovalError::processing("Could not find VP8/VP8L chunk to insert ICCP"));
        }

        // Update the total file size in RIFF header
        let new_file_size = (result.len() - 8) as u32; // Exclude RIFF header itself
        result[4..8].copy_from_slice(&new_file_size.to_le_bytes());

        Ok(result)
    }

    /// Create ICCP chunk data according to RIFF/WebP specification
    ///
    /// Format: `FourCC` `"ICCP"` + size + ICC profile data + optional padding
    ///
    /// # Arguments
    /// * `icc_data` - Raw ICC profile data
    ///
    /// # Returns
    /// Complete `ICCP` chunk including `FourCC`, size, data, and padding
    fn create_iccp_chunk(icc_data: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::new();
        
        // FourCC "ICCP" (4 bytes)
        chunk.extend_from_slice(b"ICCP");
        
        // Chunk size (4 bytes, little-endian) - size of ICC data only
        chunk.extend_from_slice(&(icc_data.len() as u32).to_le_bytes());
        
        // ICC profile data
        chunk.extend_from_slice(icc_data);
        
        // Add padding byte if chunk size is odd (RIFF chunks must be word-aligned)
        if icc_data.len() % 2 != 0 {
            chunk.push(0);
        }

        log::debug!(
            "Created ICCP chunk: size={}, total_chunk_size={}, padded={}",
            icc_data.len(),
            chunk.len(),
            icc_data.len() % 2 != 0
        );

        chunk
    }

    /// Extract ICC profile from WebP file
    ///
    /// Parses WebP/RIFF structure to find and extract ICC profile data from ICCP chunk.
    ///
    /// # Arguments
    /// * `webp_data` - WebP file data
    ///
    /// # Returns
    /// ICC profile data if found, None if no ICCP chunk present
    ///
    /// # Errors
    /// - Invalid WebP/RIFF file signature
    /// - Corrupted WebP file structure
    /// - Truncated WebP data or malformed chunks
    pub fn extract_icc_profile(webp_data: &[u8]) -> Result<Option<Vec<u8>>> {
        // Validate RIFF/WebP signature
        if webp_data.len() < 12 || webp_data.get(0..4) != Some(b"RIFF") || webp_data.get(8..12) != Some(b"WEBP") {
            return Err(BgRemovalError::processing("Invalid WebP/RIFF signature"));
        }

        let mut pos = 12; // Skip RIFF header

        // Parse chunks looking for ICCP
        while pos < webp_data.len() {
            if pos + 8 > webp_data.len() {
                break;
            }

            // Read chunk header
            let chunk_fourcc = webp_data.get(pos..pos + 4)
                .ok_or_else(|| BgRemovalError::processing("Truncated WebP: incomplete chunk fourCC"))?;
            let chunk_size_bytes = webp_data.get(pos + 4..pos + 8)
                .and_then(|bytes| bytes.try_into().ok())
                .ok_or_else(|| BgRemovalError::processing("Truncated WebP: incomplete chunk size"))?;
            let chunk_size = u32::from_le_bytes(chunk_size_bytes) as usize;

            if chunk_fourcc == b"ICCP" {
                // Found ICCP chunk, extract ICC data
                let data_start = pos + 8;
                let data_end = data_start + chunk_size;
                
                if data_end > webp_data.len() {
                    return Err(BgRemovalError::processing("Invalid ICCP chunk size"));
                }

                let icc_data = webp_data[data_start..data_end].to_vec();
                log::debug!("Extracted ICC profile from WebP ICCP chunk: {} bytes", icc_data.len());
                return Ok(Some(icc_data));
            }

            // Move to next chunk (header + data + optional padding)
            let chunk_total_size = 8 + chunk_size + (chunk_size % 2);
            pos += chunk_total_size;
        }

        // No ICCP chunk found
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ColorProfile, ColorSpace};

    #[test]
    fn test_webp_icc_encoder_creation() {
        // Test that we can create the encoder
        let _encoder = WebPIccEncoder;
    }

    #[test]
    fn test_create_iccp_chunk() {
        let test_icc_data = vec![0x01, 0x02, 0x03, 0x04]; // 4 bytes
        let chunk = WebPIccEncoder::create_iccp_chunk(&test_icc_data);
        
        // Should be: "ICCP" + size(4) + data(4) = 12 bytes (no padding needed)
        assert_eq!(chunk.len(), 12);
        if let Some(identifier) = chunk.get(0..4) {
            assert_eq!(identifier, b"ICCP");
        } else {
            panic!("ICCP identifier not found");
        }
        if let (Some(&b4), Some(&b5), Some(&b6), Some(&b7)) = (chunk.get(4), chunk.get(5), chunk.get(6), chunk.get(7)) {
            assert_eq!(u32::from_le_bytes([b4, b5, b6, b7]), 4);
        } else {
            panic!("Size bytes not found");
        }
        if let Some(data) = chunk.get(8..12) {
            assert_eq!(data, &test_icc_data);
        } else {
            panic!("Data section not found");
        }
    }

    #[test]
    fn test_create_iccp_chunk_with_padding() {
        let test_icc_data = vec![0x01, 0x02, 0x03]; // 3 bytes (odd)
        let chunk = WebPIccEncoder::create_iccp_chunk(&test_icc_data);
        
        // Should be: "ICCP" + size(4) + data(3) + padding(1) = 12 bytes
        assert_eq!(chunk.len(), 12);
        if let Some(identifier) = chunk.get(0..4) {
            assert_eq!(identifier, b"ICCP");
        } else {
            panic!("ICCP identifier not found");
        }
        if let (Some(&b4), Some(&b5), Some(&b6), Some(&b7)) = (chunk.get(4), chunk.get(5), chunk.get(6), chunk.get(7)) {
            assert_eq!(u32::from_le_bytes([b4, b5, b6, b7]), 3);
        } else {
            panic!("Size bytes not found");
        }
        if let Some(data) = chunk.get(8..11) {
            assert_eq!(data, &test_icc_data);
        } else {
            panic!("Data section not found");
        }
        assert_eq!(chunk.get(11), Some(&0)); // Padding byte
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
        let result = WebPIccEncoder::encode_with_profile(&dynamic_img, &profile, "test.webp", 80);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no data to embed"));
    }
}