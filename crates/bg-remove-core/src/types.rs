//! Core types for background removal operations

use crate::{config::OutputFormat, error::Result};
use chrono::Utc;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use log::info;
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

    /// Original input path (for logging purposes)
    pub input_path: Option<String>,
}

impl RemovalResult {
    /// Create a new removal result
    #[must_use] pub fn new(
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
            input_path: None,
        }
    }

    /// Create a new removal result with input path
    #[must_use] pub fn with_input_path(
        image: DynamicImage,
        mask: SegmentationMask,
        original_dimensions: (u32, u32),
        metadata: ProcessingMetadata,
        input_path: String,
    ) -> Self {
        Self {
            image,
            mask,
            original_dimensions,
            metadata,
            input_path: Some(input_path),
        }
    }

    /// Save the result as PNG with alpha channel
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.image.save_with_format(path, image::ImageFormat::Png)?;
        Ok(())
    }

    /// Save the result as PNG with alpha channel and return encoding time
    pub fn save_png_with_timing<P: AsRef<Path>>(&self, path: P) -> Result<u64> {
        let encode_start = std::time::Instant::now();
        self.image.save_with_format(path, image::ImageFormat::Png)?;
        Ok(encode_start.elapsed().as_millis() as u64)
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
            },
        }
    }

    /// Get the image as raw RGBA bytes
    #[must_use] pub fn to_rgba_bytes(&self) -> Vec<u8> {
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
            },
            OutputFormat::Jpeg => {
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                let rgb_image = self.image.to_rgb8();
                let mut jpeg_encoder =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
                jpeg_encoder.encode_image(&rgb_image)?;
                Ok(buffer)
            },
            OutputFormat::WebP => {
                // For now, fall back to PNG for WebP until proper WebP encoding is implemented
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                self.image.write_to(&mut cursor, image::ImageFormat::Png)?;
                Ok(buffer)
            },
            OutputFormat::Rgba8 => Ok(self.to_rgba_bytes()),
        }
    }

    /// Get image dimensions
    #[must_use] pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    /// Get detailed timing breakdown
    #[must_use] pub fn timings(&self) -> &ProcessingTimings {
        &self.metadata.timings
    }

    /// Save and measure encoding time (updates internal timing)
    pub fn save_with_timing<P: AsRef<Path>>(
        &mut self,
        path: P,
        format: image::ImageFormat,
    ) -> Result<()> {
        let path_str = path.as_ref().display().to_string();
        let encode_start = std::time::Instant::now();
        self.image.save_with_format(&path, format)?;
        let encode_ms = encode_start.elapsed().as_millis() as u64;

        // Update the timings
        self.metadata.timings.image_encode_ms = Some(encode_ms);

        // Log encoding completion first
        info!(
            "[{}Z INFO bg_remove] Image Encoding completed in {}ms",
            Utc::now().format("%Y-%m-%dT%H:%M:%S"),
            encode_ms
        );

        // Then log final completion with total processing time
        let total_time_s = self.metadata.timings.total_ms as f64 / 1000.0;
        let input_path = self.input_path.as_deref().unwrap_or("input");
        info!(
            "[{}Z INFO bg_remove] Processed: {} -> {} in {:.2}s",
            Utc::now().format("%Y-%m-%dT%H:%M:%S"),
            input_path,
            path_str,
            total_time_s
        );

        Ok(())
    }

    /// Save as PNG and measure encoding time
    pub fn save_png_timed<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.save_with_timing(path, image::ImageFormat::Png)
    }

    /// Get timing summary for display
    #[must_use] pub fn timing_summary(&self) -> String {
        let t = &self.metadata.timings;
        let breakdown = t.breakdown_percentages();

        let mut summary = format!(
            "Total: {}ms | Decode: {}ms ({:.1}%) | Preprocess: {}ms ({:.1}%) | Inference: {}ms ({:.1}%) | Postprocess: {}ms ({:.1}%)",
            t.total_ms,
            t.image_decode_ms, breakdown.decode_pct,
            t.preprocessing_ms, breakdown.preprocessing_pct,
            t.inference_ms, breakdown.inference_pct,
            t.postprocessing_ms, breakdown.postprocessing_pct
        );

        // Add encode timing if present
        if let Some(encode_ms) = t.image_encode_ms {
            summary.push_str(&format!(
                " | Encode: {}ms ({:.1}%)",
                encode_ms, breakdown.encode_pct
            ));
        }

        // Add other/overhead if significant (>1% or >5ms)
        let other_ms = t.other_overhead_ms();
        if other_ms > 5 || breakdown.other_pct > 1.0 {
            summary.push_str(&format!(
                " | Other: {}ms ({:.1}%)",
                other_ms, breakdown.other_pct
            ));
        }

        summary
    }

    /// Encode as WebP (placeholder implementation)
    fn encode_webp(&self, _quality: u8) -> Result<Vec<u8>> {
        // This would need proper WebP encoding implementation
        // For now, return an error indicating it's not implemented
        Err(crate::error::BgRemovalError::processing(
            "WebP encoding not yet implemented",
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
    #[must_use] pub fn new(data: Vec<u8>, dimensions: (u32, u32)) -> Self {
        Self { data, dimensions }
    }

    /// Create mask from a grayscale image
    #[must_use] pub fn from_image(image: &ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Self {
        let (width, height) = image.dimensions();
        let data = image.as_raw().clone();

        Self::new(data, (width, height))
    }

    /// Convert mask to a grayscale image
    pub fn to_image(&self) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        let (width, height) = self.dimensions;
        ImageBuffer::from_raw(width, height, self.data.clone()).ok_or_else(|| {
            crate::error::BgRemovalError::processing("Failed to create image from mask data")
        })
    }

    /// Apply the mask to an RGBA image
    pub fn apply_to_image(&self, image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<()> {
        let (img_width, img_height) = image.dimensions();
        let (mask_width, mask_height) = self.dimensions;

        if img_width != mask_width || img_height != mask_height {
            return Err(crate::error::BgRemovalError::processing(
                "Image and mask dimensions do not match",
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
    #[must_use] pub fn statistics(&self) -> MaskStatistics {
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

/// Detailed timing breakdown for background removal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimings {
    /// Model loading time (first call only)
    pub model_load_ms: u64,

    /// Image loading and decoding from file
    pub image_decode_ms: u64,

    /// Image preprocessing (resize, normalize, tensor conversion)
    pub preprocessing_ms: u64,

    /// ONNX Runtime inference execution
    pub inference_ms: u64,

    /// Postprocessing (mask generation, alpha application)
    pub postprocessing_ms: u64,

    /// Final image encoding (if saving to file)
    pub image_encode_ms: Option<u64>,

    /// Total end-to-end processing time
    pub total_ms: u64,
}

impl ProcessingTimings {
    #[must_use] pub fn new() -> Self {
        Self {
            model_load_ms: 0,
            image_decode_ms: 0,
            preprocessing_ms: 0,
            inference_ms: 0,
            postprocessing_ms: 0,
            image_encode_ms: None,
            total_ms: 0,
        }
    }

    /// Calculate efficiency metrics
    #[must_use] pub fn inference_ratio(&self) -> f64 {
        if self.total_ms == 0 {
            0.0
        } else {
            self.inference_ms as f64 / self.total_ms as f64
        }
    }

    /// Get breakdown percentages
    #[must_use] pub fn breakdown_percentages(&self) -> TimingBreakdown {
        if self.total_ms == 0 {
            return TimingBreakdown::default();
        }

        let total = self.total_ms as f64;
        let measured_time = self.model_load_ms
            + self.image_decode_ms
            + self.preprocessing_ms
            + self.inference_ms
            + self.postprocessing_ms
            + self.image_encode_ms.unwrap_or(0);

        let other_ms = if self.total_ms > measured_time {
            self.total_ms - measured_time
        } else {
            0
        };

        TimingBreakdown {
            model_load_pct: (self.model_load_ms as f64 / total) * 100.0,
            decode_pct: (self.image_decode_ms as f64 / total) * 100.0,
            preprocessing_pct: (self.preprocessing_ms as f64 / total) * 100.0,
            inference_pct: (self.inference_ms as f64 / total) * 100.0,
            postprocessing_pct: (self.postprocessing_ms as f64 / total) * 100.0,
            encode_pct: (self.image_encode_ms.unwrap_or(0) as f64 / total) * 100.0,
            other_pct: (other_ms as f64 / total) * 100.0,
        }
    }

    /// Get the "other" overhead time (unaccounted time)
    #[must_use] pub fn other_overhead_ms(&self) -> u64 {
        let measured_time = self.model_load_ms
            + self.image_decode_ms
            + self.preprocessing_ms
            + self.inference_ms
            + self.postprocessing_ms
            + self.image_encode_ms.unwrap_or(0);

        if self.total_ms > measured_time {
            self.total_ms - measured_time
        } else {
            0
        }
    }
}

impl Default for ProcessingTimings {
    fn default() -> Self {
        Self::new()
    }
}

/// Percentage breakdown of timing phases
#[derive(Debug, Clone)]
pub struct TimingBreakdown {
    pub model_load_pct: f64,
    pub decode_pct: f64,
    pub preprocessing_pct: f64,
    pub inference_pct: f64,
    pub postprocessing_pct: f64,
    pub encode_pct: f64,
    pub other_pct: f64,
}

impl Default for TimingBreakdown {
    fn default() -> Self {
        Self {
            model_load_pct: 0.0,
            decode_pct: 0.0,
            preprocessing_pct: 0.0,
            inference_pct: 0.0,
            postprocessing_pct: 0.0,
            encode_pct: 0.0,
            other_pct: 0.0,
        }
    }
}

/// Metadata about the processing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Detailed timing breakdown
    pub timings: ProcessingTimings,

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

    // Legacy timing fields for backward compatibility
    /// Time taken for inference (milliseconds) - DEPRECATED: use `timings.inference_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_time_ms: Option<u64>,

    /// Time taken for preprocessing (milliseconds) - DEPRECATED: use `timings.preprocessing_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocessing_time_ms: Option<u64>,

    /// Time taken for postprocessing (milliseconds) - DEPRECATED: use `timings.postprocessing_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocessing_time_ms: Option<u64>,

    /// Total processing time (milliseconds) - DEPRECATED: use `timings.total_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<u64>,
}

impl ProcessingMetadata {
    /// Create new processing metadata
    #[must_use] pub fn new(model_name: String) -> Self {
        Self {
            timings: ProcessingTimings::new(),
            model_name,
            model_precision: "fp16".to_string(),
            input_format: "unknown".to_string(),
            output_format: "png".to_string(),
            peak_memory_bytes: 0,
            // Legacy fields set to None by default
            inference_time_ms: None,
            preprocessing_time_ms: None,
            postprocessing_time_ms: None,
            total_time_ms: None,
        }
    }

    /// Set timing information (new detailed version)
    pub fn set_detailed_timings(&mut self, timings: ProcessingTimings) {
        // Also set legacy fields for backward compatibility
        self.inference_time_ms = Some(timings.inference_ms);
        self.preprocessing_time_ms = Some(timings.preprocessing_ms);
        self.postprocessing_time_ms = Some(timings.postprocessing_ms);
        self.total_time_ms = Some(timings.total_ms);

        // Update detailed timings (move after using fields)
        self.timings = timings;
    }

    /// Set timing information (legacy version for backward compatibility)
    #[deprecated(note = "Use set_detailed_timings instead")]
    pub fn set_timings(&mut self, inference: u64, preprocessing: u64, postprocessing: u64) {
        let total = inference + preprocessing + postprocessing;
        let timings = ProcessingTimings {
            model_load_ms: 0,
            image_decode_ms: 0,
            preprocessing_ms: preprocessing,
            inference_ms: inference,
            postprocessing_ms: postprocessing,
            image_encode_ms: None,
            total_ms: total,
        };
        self.set_detailed_timings(timings);
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

        // Use new detailed timing method instead of deprecated set_timings
        let timings = ProcessingTimings {
            model_load_ms: 0,
            image_decode_ms: 0,
            preprocessing_ms: 50,
            inference_ms: 100,
            postprocessing_ms: 25,
            image_encode_ms: None,
            total_ms: 175,
        };
        metadata.set_detailed_timings(timings);

        assert_eq!(metadata.inference_time_ms, Some(100));
        assert_eq!(metadata.preprocessing_time_ms, Some(50));
        assert_eq!(metadata.postprocessing_time_ms, Some(25));
        assert_eq!(metadata.total_time_ms, Some(175));
    }
}
