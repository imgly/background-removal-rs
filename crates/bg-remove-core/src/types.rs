//! Core types for background removal operations

use crate::{error::Result, config::OutputFormat};
use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Result of a background removal operation
#[derive(Debug, Clone)]
pub struct RemovalResult {
    /// The processed image with background removed
    pub image: DynamicImage,
    
    /// The segmentation mask used for removal
    pub mask: SegmentationMask,
    
    /// Original image dimensions
    pub original_dimensions: (u32, u32),
    
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

impl RemovalResult {
    /// Create a new removal result
    pub fn new(
        image: DynamicImage,
        mask: SegmentationMask,
        original_dimensions: (u32, u32),
        metadata: ProcessingMetadata,
    ) -> Self {
        Self {
            image,
            mask,
            original_dimensions,
            metadata,
        }
    }

    /// Save the result as PNG with alpha channel
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.image.save_with_format(path, image::ImageFormat::Png)?;
        Ok(())
    }

    /// Save the result as JPEG with background color
    pub fn save_jpeg<P: AsRef<Path>>(&self, path: P, quality: u8) -> Result<()> {
        // Convert to RGB and apply background color for JPEG
        let rgb_image = self.image.to_rgb8();
        let mut jpeg_encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
            std::fs::File::create(path)?,
            quality,
        );
        jpeg_encoder.encode_image(&rgb_image)?;
        Ok(())
    }

    /// Save the result as WebP
    pub fn save_webp<P: AsRef<Path>>(&self, path: P, quality: u8) -> Result<()> {
        // Note: WebP support depends on image crate features
        let webp_data = self.encode_webp(quality)?;
        std::fs::write(path, webp_data)?;
        Ok(())
    }

    /// Save in the specified format
    pub fn save<P: AsRef<Path>>(&self, path: P, format: OutputFormat, quality: u8) -> Result<()> {
        match format {
            OutputFormat::Png => self.save_png(path),
            OutputFormat::Jpeg => self.save_jpeg(path, quality),
            OutputFormat::WebP => self.save_webp(path, quality),
            OutputFormat::Rgba8 => {
                // For RGBA8 format, save the raw RGBA bytes
                let rgba_image = self.image.to_rgba8();
                std::fs::write(path, rgba_image.as_raw())?;
                Ok(())
            }
        }
    }

    /// Get the image as raw RGBA bytes
    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        self.image.to_rgba8().into_raw()
    }

    /// Get the image as encoded bytes in the specified format
    pub fn to_bytes(&self, format: OutputFormat, quality: u8) -> Result<Vec<u8>> {
        match format {
            OutputFormat::Png => {
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                self.image.write_to(&mut cursor, image::ImageFormat::Png)?;
                Ok(buffer)
            }
            OutputFormat::Jpeg => {
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                let rgb_image = self.image.to_rgb8();
                let mut jpeg_encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
                jpeg_encoder.encode_image(&rgb_image)?;
                Ok(buffer)
            }
            OutputFormat::WebP => {
                // For now, fall back to PNG for WebP until proper WebP encoding is implemented
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                self.image.write_to(&mut cursor, image::ImageFormat::Png)?;
                Ok(buffer)
            }
            OutputFormat::Rgba8 => {
                Ok(self.to_rgba_bytes())
            }
        }
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    /// Encode as WebP (placeholder implementation)
    fn encode_webp(&self, _quality: u8) -> Result<Vec<u8>> {
        // This would need proper WebP encoding implementation
        // For now, return an error indicating it's not implemented
        Err(crate::error::BgRemovalError::processing(
            "WebP encoding not yet implemented"
        ))
    }
}

/// Binary segmentation mask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationMask {
    /// Mask data as grayscale values (0-255)
    pub data: Vec<u8>,
    
    /// Mask dimensions (width, height)
    pub dimensions: (u32, u32),
}

impl SegmentationMask {
    /// Create a new segmentation mask
    pub fn new(data: Vec<u8>, dimensions: (u32, u32)) -> Self {
        Self {
            data,
            dimensions,
        }
    }

    /// Create mask from a grayscale image
    pub fn from_image(image: &ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Self {
        let (width, height) = image.dimensions();
        let data = image.as_raw().clone();
        
        Self::new(data, (width, height))
    }

    /// Convert mask to a grayscale image
    pub fn to_image(&self) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        let (width, height) = self.dimensions;
        ImageBuffer::from_raw(width, height, self.data.clone())
            .ok_or_else(|| crate::error::BgRemovalError::processing(
                "Failed to create image from mask data"
            ))
    }

    /// Apply the mask to an RGBA image
    pub fn apply_to_image(&self, image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<()> {
        let (img_width, img_height) = image.dimensions();
        let (mask_width, mask_height) = self.dimensions;

        if img_width != mask_width || img_height != mask_height {
            return Err(crate::error::BgRemovalError::processing(
                "Image and mask dimensions do not match"
            ));
        }

        for (i, pixel) in image.pixels_mut().enumerate() {
            if i < self.data.len() {
                let alpha = self.data[i];
                pixel[3] = alpha; // Set alpha channel
            }
        }

        Ok(())
    }

    /// Resize the mask to new dimensions
    pub fn resize(&self, new_width: u32, new_height: u32) -> Result<SegmentationMask> {
        let current_image = self.to_image()?;
        let resized = image::imageops::resize(
            &current_image,
            new_width,
            new_height,
            image::imageops::FilterType::Lanczos3,
        );
        
        Ok(SegmentationMask::from_image(&resized))
    }

    /// Get mask statistics
    pub fn statistics(&self) -> MaskStatistics {
        let total_pixels = self.data.len() as f32;
        let foreground_pixels = self.data.iter().filter(|&&x| x > 127).count() as f32;
        let background_pixels = total_pixels - foreground_pixels;
        
        MaskStatistics {
            total_pixels: total_pixels as usize,
            foreground_pixels: foreground_pixels as usize,
            background_pixels: background_pixels as usize,
            foreground_ratio: foreground_pixels / total_pixels,
            background_ratio: background_pixels / total_pixels,
        }
    }

    /// Save mask as PNG
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let image = self.to_image()?;
        image.save_with_format(path, image::ImageFormat::Png)?;
        Ok(())
    }
}

/// Statistics about a segmentation mask
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskStatistics {
    pub total_pixels: usize,
    pub foreground_pixels: usize,
    pub background_pixels: usize,
    pub foreground_ratio: f32,
    pub background_ratio: f32,
}

/// Metadata about the processing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Time taken for inference (milliseconds)
    pub inference_time_ms: u64,
    
    /// Time taken for preprocessing (milliseconds)
    pub preprocessing_time_ms: u64,
    
    /// Time taken for postprocessing (milliseconds)
    pub postprocessing_time_ms: u64,
    
    /// Total processing time (milliseconds)
    pub total_time_ms: u64,
    
    /// Model used for inference
    pub model_name: String,
    
    /// Model precision used
    pub model_precision: String,
    
    /// Input image format
    pub input_format: String,
    
    /// Output image format
    pub output_format: String,
    
    /// Memory usage peak (bytes)
    pub peak_memory_bytes: u64,
}

impl ProcessingMetadata {
    /// Create new processing metadata
    pub fn new(model_name: String) -> Self {
        Self {
            inference_time_ms: 0,
            preprocessing_time_ms: 0,
            postprocessing_time_ms: 0,
            total_time_ms: 0,
            model_name,
            model_precision: "fp16".to_string(),
            input_format: "unknown".to_string(),
            output_format: "png".to_string(),
            peak_memory_bytes: 0,
        }
    }

    /// Set timing information
    pub fn set_timings(&mut self, inference: u64, preprocessing: u64, postprocessing: u64) {
        self.inference_time_ms = inference;
        self.preprocessing_time_ms = preprocessing;
        self.postprocessing_time_ms = postprocessing;
        self.total_time_ms = inference + preprocessing + postprocessing;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmentation_mask_creation() {
        let data = vec![255, 128, 0, 255];
        let mask = SegmentationMask::new(data, (2, 2));
        
        assert_eq!(mask.dimensions, (2, 2));
        assert_eq!(mask.data.len(), 4);
    }

    #[test]
    fn test_mask_statistics() {
        let data = vec![255, 255, 0, 0]; // 2 foreground, 2 background
        let mask = SegmentationMask::new(data, (2, 2));
        
        let stats = mask.statistics();
        assert_eq!(stats.total_pixels, 4);
        assert_eq!(stats.foreground_pixels, 2);
        assert_eq!(stats.background_pixels, 2);
        assert_eq!(stats.foreground_ratio, 0.5);
        assert_eq!(stats.background_ratio, 0.5);
    }

    #[test]
    fn test_processing_metadata() {
        let mut metadata = ProcessingMetadata::new("isnet".to_string());
        metadata.set_timings(100, 50, 25);
        
        assert_eq!(metadata.inference_time_ms, 100);
        assert_eq!(metadata.preprocessing_time_ms, 50);
        assert_eq!(metadata.postprocessing_time_ms, 25);
        assert_eq!(metadata.total_time_ms, 175);
    }
}