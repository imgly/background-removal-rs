//! PNG encoder with ICC profile embedding support
//!
//! This module provides PNG encoding with ICC color profile embedding using the iCCP chunk.
//! Implements manual iCCP chunk creation and insertion according to the PNG specification
//! to ensure proper ICC profile embedding in PNG files.

use crate::{
    error::{BgRemovalError, Result},
    types::ColorProfile,
};
use image::DynamicImage;
use std::path::Path;
use std::io::{Cursor, Write};
use flate2::{Compression, write::ZlibEncoder};

/// PNG encoder with ICC profile embedding capability
pub struct PngIccEncoder;

impl PngIccEncoder {
    /// Encode image to PNG with embedded ICC profile
    ///
    /// Creates a PNG file with an embedded ICC color profile using the iCCP chunk.
    /// Implements manual iCCP chunk creation and insertion according to PNG specification.
    ///
    /// # Arguments
    /// * `image` - The image to encode
    /// * `profile` - The ICC profile to embed
    /// * `output_path` - Output file path
    ///
    /// # Returns
    /// Result indicating success or failure of the encoding operation
    ///
    /// # Errors
    /// - Color profile has no ICC data to embed
    /// - PNG encoding errors from the underlying image library
    /// - File I/O errors when writing the output file
    /// - Invalid PNG data format when embedding ICC profile
    ///
    /// # Examples
    /// ```rust,no_run
    /// use bg_remove_core::{
    ///     encoders::png_encoder::PngIccEncoder,
    ///     types::ColorProfile,
    /// };
    /// use image::DynamicImage;
    ///
    /// # fn example(image: DynamicImage, profile: ColorProfile) -> bg_remove_core::Result<()> {
    /// PngIccEncoder::encode_with_profile(&image, &profile, "output.png")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_with_profile<P: AsRef<Path>>(
        image: &DynamicImage,
        profile: &ColorProfile,
        output_path: P,
    ) -> Result<()> {
        let output_path = output_path.as_ref();
        
        // Validate ICC profile data
        let icc_data = profile.icc_data.as_ref().ok_or_else(|| {
            BgRemovalError::processing("ICC profile has no data to embed")
        })?;

        log::info!(
            "Embedding ICC color profile in PNG: {} ({} bytes)",
            profile.color_space,
            profile.data_size()
        );

        // Step 1: Create standard PNG without ICC profile
        let mut png_buffer = Vec::new();
        {
            let rgba_image = image.to_rgba8();
            let mut cursor = Cursor::new(&mut png_buffer);
            rgba_image.write_to(&mut cursor, image::ImageFormat::Png).map_err(|e| {
                BgRemovalError::processing(format!("Failed to create PNG buffer: {e}"))
            })?;
        }

        // Step 2: Insert iCCP chunk into PNG
        let profile_name = profile.color_space.to_string();
        let png_with_icc = Self::insert_iccp_chunk(&png_buffer, icc_data, &profile_name)?;

        // Step 3: Write final PNG to file
        std::fs::write(output_path, png_with_icc).map_err(|e| {
            BgRemovalError::processing(format!("Failed to write PNG file: {e}"))
        })?;

        log::info!("Successfully created PNG with embedded ICC profile: {}", output_path.display());
        Ok(())
    }

    /// Insert iCCP chunk into PNG data
    ///
    /// Creates an iCCP chunk with compressed ICC profile data and inserts it
    /// into the PNG file structure before the first IDAT chunk.
    ///
    /// # Arguments
    /// * `png_data` - Original PNG file data
    /// * `icc_data` - Raw ICC profile data to embed
    /// * `profile_name` - Name for the ICC profile
    ///
    /// # Returns
    /// PNG data with embedded iCCP chunk
    fn insert_iccp_chunk(png_data: &[u8], icc_data: &[u8], profile_name: &str) -> Result<Vec<u8>> {
        // Validate PNG signature
        if png_data.len() < 8 || png_data.get(0..8) != Some(b"\x89PNG\r\n\x1a\n") {
            return Err(BgRemovalError::processing("Invalid PNG signature"));
        }

        let mut result = Vec::new();
        let mut pos = 0;

        // Copy PNG signature (validated above)
        #[allow(clippy::indexing_slicing)] // Safe: validated above to have 8 bytes
        result.extend_from_slice(&png_data[0..8]);
        pos += 8;

        // Create iCCP chunk
        let iccp_chunk = Self::create_iccp_chunk(icc_data, profile_name)?;
        let mut iccp_inserted = false;

        // Parse and copy chunks, inserting iCCP before first IDAT
        while pos < png_data.len() {
            if pos + 8 > png_data.len() {
                break;
            }

            // Read chunk length and type  
            let chunk_length_bytes = png_data.get(pos..pos + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .ok_or_else(|| BgRemovalError::processing("Truncated PNG: incomplete chunk length"))?;
            let chunk_length = u32::from_be_bytes(chunk_length_bytes);
            let chunk_type = png_data.get(pos + 4..pos + 8)
                .ok_or_else(|| BgRemovalError::processing("Truncated PNG: incomplete chunk type"))?;

            // Insert iCCP chunk before first IDAT chunk
            if chunk_type == b"IDAT" && !iccp_inserted {
                result.extend_from_slice(&iccp_chunk);
                iccp_inserted = true;
                log::debug!("Inserted iCCP chunk before IDAT chunk");
            }

            // Copy current chunk (length + type + data + crc)
            let chunk_length_usize: usize = chunk_length.try_into()
                .map_err(|_| BgRemovalError::processing("PNG chunk length too large for usize"))?;
            let chunk_total_size = 12 + chunk_length_usize; // 4 + 4 + data + 4
            if pos + chunk_total_size > png_data.len() {
                break;
            }
            #[allow(clippy::indexing_slicing)] // Safe: bounds checked above
            result.extend_from_slice(&png_data[pos..pos + chunk_total_size]);
            pos += chunk_total_size;

            // If this is IEND, we're done
            if chunk_type == b"IEND" {
                break;
            }
        }

        if !iccp_inserted {
            return Err(BgRemovalError::processing("Could not find IDAT chunk to insert iCCP"));
        }

        Ok(result)
    }

    /// Create iCCP chunk data according to PNG specification
    ///
    /// Format: Profile name + null separator + compression method + compressed profile
    ///
    /// # Arguments
    /// * `icc_data` - Raw ICC profile data
    /// * `profile_name` - Profile name (max 79 chars)
    ///
    /// # Returns
    /// Complete iCCP chunk including length, type, data, and CRC
    fn create_iccp_chunk(icc_data: &[u8], profile_name: &str) -> Result<Vec<u8>> {
        // Validate profile name
        if profile_name.len() > 79 {
            return Err(BgRemovalError::processing("ICC profile name too long (max 79 chars)"));
        }

        // Compress ICC profile data using zlib
        let mut compressed_data = Vec::new();
        {
            let mut encoder = ZlibEncoder::new(&mut compressed_data, Compression::default());
            encoder.write_all(icc_data).map_err(|e| {
                BgRemovalError::processing(format!("Failed to compress ICC data: {e}"))
            })?;
            encoder.finish().map_err(|e| {
                BgRemovalError::processing(format!("Failed to finish ICC compression: {e}"))
            })?;
        }

        // Build chunk data: profile name + null + compression method + compressed data
        let mut chunk_data = Vec::new();
        chunk_data.extend_from_slice(profile_name.as_bytes());
        chunk_data.push(0); // Null separator
        chunk_data.push(0); // Compression method (0 = zlib)
        chunk_data.extend_from_slice(&compressed_data);

        // Create complete chunk: length + type + data + CRC
        let mut chunk = Vec::new();
        
        // Chunk length (4 bytes, big-endian)
        let chunk_data_len: u32 = chunk_data.len()
            .try_into()
            .map_err(|_| BgRemovalError::processing("ICC profile data too large for PNG chunk (>4GB)"))?;
        chunk.extend_from_slice(&chunk_data_len.to_be_bytes());
        
        // Chunk type (4 bytes)
        chunk.extend_from_slice(b"iCCP");
        
        // Chunk data
        chunk.extend_from_slice(&chunk_data);
        
        // CRC32 (4 bytes, big-endian) - calculated over type + data
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"iCCP");
        crc_data.extend_from_slice(&chunk_data);
        let crc = crc32fast::hash(&crc_data);
        chunk.extend_from_slice(&crc.to_be_bytes());

        log::debug!(
            "Created iCCP chunk: profile='{}', original_size={}, compressed_size={}, total_chunk_size={}",
            profile_name,
            icc_data.len(),
            compressed_data.len(),
            chunk.len()
        );

        Ok(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ColorProfile, ColorSpace};

    #[test]
    fn test_png_icc_encoder_creation() {
        // Test that we can create the encoder - the struct exists
        let _ = PngIccEncoder;
    }

    #[test]
    fn test_encode_with_profile_validates_input() {
        use image::{RgbaImage, DynamicImage};
        
        // Create a test image
        let img = RgbaImage::new(100, 100);
        let dynamic_img = DynamicImage::ImageRgba8(img);
        
        // Create profile without ICC data
        let profile = ColorProfile::new(None, ColorSpace::Srgb);
        
        // Should fail due to no ICC data
        let result = PngIccEncoder::encode_with_profile(&dynamic_img, &profile, "test.png");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no data to embed"));
    }
}