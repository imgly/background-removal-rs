//! Image processing pipeline for background removal

use crate::{
    config::{RemovalConfig, OutputFormat, BackgroundColor},
    error::Result,
    inference::{InferenceBackend, OnnxBackend, MockBackend},
    types::{RemovalResult, SegmentationMask, ProcessingMetadata},
};
use image::{DynamicImage, ImageBuffer, RgbaImage, GenericImageView};
use ndarray::Array4;
use std::path::Path;
use std::time::Instant;

/// Processing options for fine-tuning behavior
#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    /// Confidence threshold for mask generation (0.0-1.0)
    pub confidence_threshold: f32,
    
    /// Enable mask smoothing
    pub smooth_mask: bool,
    
    /// Feather edges (blur radius in pixels)
    pub feather_radius: u32,
    
    /// Apply post-processing filters
    pub apply_filters: bool,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            smooth_mask: true,
            feather_radius: 2,
            apply_filters: true,
        }
    }
}

/// Main image processor for background removal operations
pub struct ImageProcessor {
    backend: Box<dyn InferenceBackend>,
    config: RemovalConfig,
    options: ProcessingOptions,
}

impl ImageProcessor {
    /// Create a new image processor with the given configuration
    pub fn new(config: &RemovalConfig) -> Result<Self> {
        let mut backend: Box<dyn InferenceBackend> = if config.debug {
            // Use mock backend for debugging
            Box::new(MockBackend::new())
        } else {
            // Use ONNX backend for production
            Box::new(OnnxBackend::new())
        };

        // Initialize the backend
        backend.initialize(config)?;

        Ok(Self {
            backend,
            config: config.clone(),
            options: ProcessingOptions::default(),
        })
    }

    /// Create processor with custom processing options
    pub fn with_options(config: &RemovalConfig, options: ProcessingOptions) -> Result<Self> {
        let mut processor = Self::new(config)?;
        processor.options = options;
        Ok(processor)
    }

    /// Remove background from an image file
    pub async fn remove_background<P: AsRef<Path>>(&mut self, input_path: P) -> Result<RemovalResult> {
        let _start_time = Instant::now();
        let mut metadata = ProcessingMetadata::new("ISNet".to_string());

        // Load and preprocess image
        let preprocess_start = Instant::now();
        let image = self.load_image(input_path)?;
        let original_dimensions = image.dimensions();
        let (processed_image, input_tensor) = self.preprocess_image(&image)?;
        let preprocessing_time = preprocess_start.elapsed().as_millis() as u64;

        // Run inference
        let inference_start = Instant::now();
        let output_tensor = self.backend.infer(&input_tensor)?;
        let inference_time = inference_start.elapsed().as_millis() as u64;

        // Postprocess results
        let postprocess_start = Instant::now();
        let mask = self.tensor_to_mask(&output_tensor, processed_image.dimensions())?;
        let result_image = self.apply_background_removal(&processed_image, &mask)?;
        let postprocessing_time = postprocess_start.elapsed().as_millis() as u64;

        // Set metadata
        metadata.set_timings(inference_time, preprocessing_time, postprocessing_time);
        metadata.input_format = self.detect_image_format(&image);
        metadata.output_format = format!("{:?}", self.config.output_format).to_lowercase();

        Ok(RemovalResult::new(
            result_image,
            mask,
            original_dimensions,
            metadata,
        ))
    }

    /// Extract foreground segmentation mask only
    pub async fn segment_foreground<P: AsRef<Path>>(&mut self, input_path: P) -> Result<SegmentationMask> {
        let image = self.load_image(input_path)?;
        let (processed_image, input_tensor) = self.preprocess_image(&image)?;
        
        let output_tensor = self.backend.infer(&input_tensor)?;
        self.tensor_to_mask(&output_tensor, processed_image.dimensions())
    }

    /// Apply an existing mask to an image
    pub async fn apply_mask<P: AsRef<Path>>(
        &self,
        input_path: P,
        mask: &SegmentationMask,
    ) -> Result<RemovalResult> {
        let _start_time = Instant::now();
        let mut metadata = ProcessingMetadata::new("MaskApplication".to_string());

        let image = self.load_image(input_path)?;
        let original_dimensions = image.dimensions();
        
        // Resize mask if needed
        let resized_mask = if mask.dimensions != image.dimensions() {
            mask.resize(image.dimensions().0, image.dimensions().1)?
        } else {
            mask.clone()
        };

        let result_image = self.apply_background_removal(&image, &resized_mask)?;
        
        metadata.set_timings(0, 0, _start_time.elapsed().as_millis() as u64);
        metadata.input_format = self.detect_image_format(&image);
        metadata.output_format = format!("{:?}", self.config.output_format).to_lowercase();

        Ok(RemovalResult::new(
            result_image,
            resized_mask,
            original_dimensions,
            metadata,
        ))
    }

    /// Load image from file path
    fn load_image<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        let image = image::open(path)?;
        
        // Apply max dimension constraint if configured
        if let Some(max_dim) = self.config.max_dimension {
            let (width, height) = image.dimensions();
            if width > max_dim || height > max_dim {
                let scale = (max_dim as f32) / width.max(height) as f32;
                let new_width = (width as f32 * scale) as u32;
                let new_height = (height as f32 * scale) as u32;
                
                return Ok(image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3));
            }
        }
        
        Ok(image)
    }

    /// Preprocess image for model inference
    fn preprocess_image(&self, image: &DynamicImage) -> Result<(DynamicImage, Array4<f32>)> {
        // Convert to RGB and resize to model input size (1024x1024 like JavaScript)
        let rgb_image = image.to_rgb8();
        
        // ISNet expects 1024x1024 input (no aspect ratio preservation, like JavaScript)
        let target_size = 1024u32;
        let resized = image::imageops::resize(
            &rgb_image,
            target_size,
            target_size,
            image::imageops::FilterType::Triangle, // Bilinear-like
        );

        // Convert to tensor format (NCHW) with JavaScript-compatible normalization
        let mut tensor = Array4::<f32>::zeros((1, 3, target_size as usize, target_size as usize));
        
        for (y, row) in resized.rows().enumerate() {
            for (x, pixel) in row.enumerate() {
                // JavaScript normalization: (pixel - mean) / std
                // Mean: [128, 128, 128], Std: [256, 256, 256]
                tensor[[0, 0, y, x]] = (pixel[0] as f32 - 128.0) / 256.0; // R
                tensor[[0, 1, y, x]] = (pixel[1] as f32 - 128.0) / 256.0; // G
                tensor[[0, 2, y, x]] = (pixel[2] as f32 - 128.0) / 256.0; // B
            }
        }

        Ok((image.clone(), tensor))
    }

    /// Convert model output tensor to segmentation mask
    fn tensor_to_mask(&self, tensor: &Array4<f32>, target_size: (u32, u32)) -> Result<SegmentationMask> {
        let (batch, channels, height, width) = tensor.dim();
        
        if batch != 1 || channels != 1 {
            return Err(crate::error::BgRemovalError::processing(
                "Expected single-channel output tensor"
            ));
        }

        // Extract mask data - use tensor values directly as alpha channel
        let mut mask_data = Vec::with_capacity(height * width);
        for y in 0..height {
            for x in 0..width {
                let raw_value = tensor[[0, 0, y, x]];
                // Convert float tensor value directly to u8 alpha
                let pixel_value = (raw_value * 255.0).clamp(0.0, 255.0) as u8;
                mask_data.push(pixel_value);
            }
        }

        // Create initial mask
        let mut mask = SegmentationMask::new(
            mask_data,
            (width as u32, height as u32),
            self.options.confidence_threshold,
        );

        // Resize to target dimensions if needed
        if mask.dimensions != target_size {
            mask = mask.resize(target_size.0, target_size.1)?;
        }

        // Apply post-processing
        if self.options.smooth_mask {
            mask = self.smooth_mask(mask)?;
        }

        Ok(mask)
    }

    /// Apply background removal using the mask
    fn apply_background_removal(&self, image: &DynamicImage, mask: &SegmentationMask) -> Result<DynamicImage> {
        let mut rgba_image = image.to_rgba8();
        
        // Apply mask to alpha channel
        mask.apply_to_image(&mut rgba_image)?;

        // Handle different output formats
        match self.config.output_format {
            OutputFormat::Png | OutputFormat::Raw => {
                // Keep alpha channel
                Ok(DynamicImage::ImageRgba8(rgba_image))
            }
            OutputFormat::Jpeg => {
                // Apply background color
                let bg_color = self.config.background_color;
                let rgb_image = self.apply_background_color(&rgba_image, bg_color)?;
                Ok(DynamicImage::ImageRgb8(rgb_image))
            }
            OutputFormat::WebP => {
                // WebP supports alpha, so keep it
                Ok(DynamicImage::ImageRgba8(rgba_image))
            }
        }
    }

    /// Apply background color for formats without alpha support
    fn apply_background_color(
        &self,
        rgba_image: &RgbaImage,
        bg_color: BackgroundColor,
    ) -> Result<ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
        let (width, height) = rgba_image.dimensions();
        let mut rgb_image = ImageBuffer::new(width, height);

        for (x, y, pixel) in rgba_image.enumerate_pixels() {
            let alpha = pixel[3] as f32 / 255.0;
            let inv_alpha = 1.0 - alpha;

            let r = (pixel[0] as f32 * alpha + bg_color.r as f32 * inv_alpha) as u8;
            let g = (pixel[1] as f32 * alpha + bg_color.g as f32 * inv_alpha) as u8;
            let b = (pixel[2] as f32 * alpha + bg_color.b as f32 * inv_alpha) as u8;

            rgb_image.put_pixel(x, y, image::Rgb([r, g, b]));
        }

        Ok(rgb_image)
    }

    /// Smooth mask edges using simple blur
    fn smooth_mask(&self, mask: SegmentationMask) -> Result<SegmentationMask> {
        if self.options.feather_radius == 0 {
            return Ok(mask);
        }

        let mask_image = mask.to_image()?;
        let blurred = image::imageops::blur(&mask_image, self.options.feather_radius as f32);
        
        Ok(SegmentationMask::from_image(&blurred, mask.threshold))
    }

    /// Detect image format from dynamic image
    fn detect_image_format(&self, image: &DynamicImage) -> String {
        match image {
            DynamicImage::ImageRgb8(_) => "rgb8".to_string(),
            DynamicImage::ImageRgba8(_) => "rgba8".to_string(),
            DynamicImage::ImageLuma8(_) => "luma8".to_string(),
            DynamicImage::ImageLumaA8(_) => "luma_a8".to_string(),
            _ => "unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RemovalConfig;
    use tempfile::NamedTempFile;

    #[test]
    fn test_processing_options() {
        let options = ProcessingOptions::default();
        assert_eq!(options.confidence_threshold, 0.5);
        assert!(options.smooth_mask);
        assert_eq!(options.feather_radius, 2);
    }

    #[tokio::test]
    async fn test_image_processor_creation() {
        let mut config = RemovalConfig::default();
        config.debug = true; // Use mock backend
        
        let processor = ImageProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_preprocess_image() {
        let mut config = RemovalConfig::default();
        config.debug = true;
        
        let processor = ImageProcessor::new(&config).unwrap();
        
        // Create a test image
        let test_image = DynamicImage::new_rgb8(100, 100);
        let (processed, tensor) = processor.preprocess_image(&test_image).unwrap();
        
        // Check tensor shape matches backend requirements
        let expected_shape = processor.backend.input_shape();
        assert_eq!(tensor.dim(), expected_shape);
    }
}