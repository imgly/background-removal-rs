//! Shared image preprocessing utilities
//!
//! This module consolidates image preprocessing logic that was previously
//! duplicated between `ImageProcessor` and `BackgroundRemovalProcessor`.

use crate::{
    error::{BgRemovalError, Result},
    models::PreprocessingConfig,
};
use image::{DynamicImage, ImageBuffer, RgbImage};
use ndarray::Array4;

/// Configuration for preprocessing behavior
#[derive(Debug, Clone)]
pub struct PreprocessingOptions {
    /// Padding color for aspect ratio preservation (RGB)
    pub padding_color: [u8; 3],
    /// Whether to return the preprocessed image for debugging
    pub return_preprocessed_image: bool,
}

impl Default for PreprocessingOptions {
    fn default() -> Self {
        Self {
            padding_color: [255, 255, 255], // White padding
            return_preprocessed_image: false,
        }
    }
}

/// Shared image preprocessing utilities
pub struct ImagePreprocessor;

impl ImagePreprocessor {
    /// Preprocess image for model inference with comprehensive error handling
    ///
    /// This function handles:
    /// - RGB conversion
    /// - Aspect ratio preserving resize  
    /// - Center padding to target size
    /// - Normalization to tensor format (NCHW)
    ///
    /// # Arguments
    /// * `image` - Input image to preprocess
    /// * `preprocessing_config` - Model preprocessing configuration
    /// * `options` - Preprocessing behavior options
    ///
    /// # Returns
    /// * `Ok((preprocessed_image_opt, tensor))` - Preprocessed image (if requested) and tensor
    /// * `Err(BgRemovalError)` - On preprocessing errors
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // Casting is acceptable for image processing math - precision loss is expected
    pub fn preprocess_image(
        image: &DynamicImage,
        preprocessing_config: &PreprocessingConfig,
        options: &PreprocessingOptions,
    ) -> Result<(Option<DynamicImage>, Array4<f32>)> {
        let target_size = preprocessing_config.target_size[0];

        // Convert to RGB
        let rgb_image = image.to_rgb8();
        let (orig_width, orig_height) = rgb_image.dimensions();

        // Calculate aspect ratio preserving dimensions with bounds checking
        let target_size_f32 = target_size as f32;
        let orig_width_f32 = orig_width as f32;
        let orig_height_f32 = orig_height as f32;

        // Use the more conservative scaling approach for better quality
        let scale = target_size_f32
            .min((target_size_f32 / orig_width_f32).min(target_size_f32 / orig_height_f32));

        let new_width_f32 = (orig_width_f32 * scale).round();
        let new_height_f32 = (orig_height_f32 * scale).round();

        // Validate calculated dimensions
        if new_width_f32 < 0.0 || new_width_f32 > u32::MAX as f32 {
            return Err(BgRemovalError::processing(
                "Calculated new width out of valid range",
            ));
        }
        if new_height_f32 < 0.0 || new_height_f32 > u32::MAX as f32 {
            return Err(BgRemovalError::processing(
                "Calculated new height out of valid range",
            ));
        }

        let new_width = new_width_f32 as u32;
        let new_height = new_height_f32 as u32;

        // Resize image maintaining aspect ratio
        let resized = image::imageops::resize(
            &rgb_image,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle,
        );

        // Create target_size x target_size canvas with configurable padding
        let padding = options.padding_color;
        let mut canvas = ImageBuffer::from_pixel(
            target_size,
            target_size,
            image::Rgb([padding[0], padding[1], padding[2]]),
        );

        // Calculate centering offset
        let offset_x = (target_size - new_width) / 2;
        let offset_y = (target_size - new_height) / 2;

        // Copy resized image to center of canvas
        for (x, y, pixel) in resized.enumerate_pixels() {
            let canvas_x = x + offset_x;
            let canvas_y = y + offset_y;
            if canvas_x < target_size && canvas_y < target_size {
                canvas.put_pixel(canvas_x, canvas_y, *pixel);
            }
        }

        // Convert to tensor format (NCHW) with normalization
        let target_size_usize = target_size.try_into().map_err(|_| {
            BgRemovalError::processing(
                "Target size too large for usize conversion in tensor allocation",
            )
        })?;

        let tensor = Self::canvas_to_tensor(&canvas, preprocessing_config, target_size_usize);

        // Return preprocessed image if requested
        let preprocessed_image = if options.return_preprocessed_image {
            Some(DynamicImage::ImageRgb8(canvas))
        } else {
            None
        };

        Ok((preprocessed_image, tensor))
    }

    /// Convert canvas to normalized tensor
    fn canvas_to_tensor(
        canvas: &RgbImage,
        preprocessing_config: &PreprocessingConfig,
        target_size: usize,
    ) -> Array4<f32> {
        let mut tensor = Array4::<f32>::zeros((1, 3, target_size, target_size));

        #[allow(clippy::indexing_slicing)]
        // Safe: tensor dimensions pre-allocated to match canvas size
        for (y, row) in canvas.rows().enumerate() {
            for (x, pixel) in row.enumerate() {
                // Convert to 0-1 range and apply normalization
                let normalized_r = (f32::from(pixel[0]) / 255.0
                    - preprocessing_config.normalization_mean[0])
                    / preprocessing_config.normalization_std[0];
                let normalized_g = (f32::from(pixel[1]) / 255.0
                    - preprocessing_config.normalization_mean[1])
                    / preprocessing_config.normalization_std[1];
                let normalized_b = (f32::from(pixel[2]) / 255.0
                    - preprocessing_config.normalization_mean[2])
                    / preprocessing_config.normalization_std[2];

                tensor[[0, 0, y, x]] = normalized_r; // R channel
                tensor[[0, 1, y, x]] = normalized_g; // G channel
                tensor[[0, 2, y, x]] = normalized_b; // B channel
            }
        }

        tensor
    }

    /// Simple preprocessing for unified processor (tensor only)
    ///
    /// This is a convenience method that doesn't return the preprocessed image,
    /// making it suitable for use in the unified processor.
    ///
    /// # Arguments
    /// * `image` - Input image to preprocess
    /// * `preprocessing_config` - Model preprocessing configuration
    ///
    /// # Returns
    /// * `Ok(tensor)` - Preprocessed tensor ready for inference
    /// * `Err(BgRemovalError)` - On preprocessing errors
    pub fn preprocess_for_inference(
        image: &DynamicImage,
        preprocessing_config: &PreprocessingConfig,
    ) -> Result<Array4<f32>> {
        let options = PreprocessingOptions::default();
        let (_, tensor) = Self::preprocess_image(image, preprocessing_config, &options)?;
        Ok(tensor)
    }

    /// Preprocess with custom padding color (for legacy `ImageProcessor` compatibility)
    ///
    /// # Arguments
    /// * `image` - Input image to preprocess
    /// * `preprocessing_config` - Model preprocessing configuration
    /// * `padding_color` - RGB padding color
    ///
    /// # Returns
    /// * `Ok((preprocessed_image, tensor))` - Preprocessed image and tensor
    /// * `Err(BgRemovalError)` - On preprocessing errors
    pub fn preprocess_with_padding(
        image: &DynamicImage,
        preprocessing_config: &PreprocessingConfig,
        padding_color: [u8; 3],
    ) -> Result<(DynamicImage, Array4<f32>)> {
        let options = PreprocessingOptions {
            padding_color,
            return_preprocessed_image: true,
        };
        let (preprocessed_opt, tensor) =
            Self::preprocess_image(image, preprocessing_config, &options)?;
        let preprocessed = preprocessed_opt.ok_or_else(|| {
            BgRemovalError::processing("Expected preprocessed image but none returned")
        })?;
        Ok((preprocessed, tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_test_preprocessing_config() -> PreprocessingConfig {
        PreprocessingConfig {
            target_size: [1024, 1024],
            normalization_mean: [0.485, 0.456, 0.406],
            normalization_std: [0.229, 0.224, 0.225],
        }
    }

    fn create_test_image() -> DynamicImage {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(100, 100, Rgb([255, 0, 0]));
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_preprocess_for_inference() {
        let image = create_test_image();
        let config = create_test_preprocessing_config();

        let tensor = ImagePreprocessor::preprocess_for_inference(&image, &config).unwrap();

        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn test_preprocess_with_padding() {
        let image = create_test_image();
        let config = create_test_preprocessing_config();
        let padding = [128, 128, 128]; // Gray padding

        let (preprocessed, tensor) =
            ImagePreprocessor::preprocess_with_padding(&image, &config, padding).unwrap();

        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
        assert_eq!(preprocessed.width(), 1024);
        assert_eq!(preprocessed.height(), 1024);
    }

    #[test]
    fn test_preprocess_with_options() {
        let image = create_test_image();
        let config = create_test_preprocessing_config();
        let options = PreprocessingOptions {
            padding_color: [0, 255, 0], // Green padding
            return_preprocessed_image: true,
        };

        let (preprocessed_opt, tensor) =
            ImagePreprocessor::preprocess_image(&image, &config, &options).unwrap();

        assert!(preprocessed_opt.is_some());
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn test_preprocess_no_image_return() {
        let image = create_test_image();
        let config = create_test_preprocessing_config();
        let options = PreprocessingOptions {
            padding_color: [255, 255, 255],
            return_preprocessed_image: false,
        };

        let (preprocessed_opt, tensor) =
            ImagePreprocessor::preprocess_image(&image, &config, &options).unwrap();

        assert!(preprocessed_opt.is_none());
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }
}
