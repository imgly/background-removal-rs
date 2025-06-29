//! Tensor validation utilities
//!
//! Provides centralized validation for tensor shapes, dimensions, and
//! array-related operations.

use crate::error::{BgRemovalError, Result};
use ndarray::Array4;

/// Validator for tensor operations and shape validation
pub struct TensorValidator;

impl TensorValidator {
    /// Validate tensor shape matches expected dimensions
    pub fn validate_tensor_shape(
        tensor: &Array4<f32>,
        expected_shape: (usize, usize, usize, usize),
    ) -> Result<()> {
        let actual_shape = tensor.shape();
        let (batch, channels, height, width) = expected_shape;

        if actual_shape.len() != 4 {
            return Err(BgRemovalError::processing(format!(
                "Tensor must have 4 dimensions, got {}",
                actual_shape.len()
            )));
        }

        let actual = (
            actual_shape.get(0).copied().unwrap_or(0),
            actual_shape.get(1).copied().unwrap_or(0),
            actual_shape.get(2).copied().unwrap_or(0),
            actual_shape.get(3).copied().unwrap_or(0),
        );
        if actual != expected_shape {
            return Err(BgRemovalError::processing(format!(
                "Tensor shape mismatch. Expected [{}, {}, {}, {}], got [{}, {}, {}, {}]",
                batch, channels, height, width, actual.0, actual.1, actual.2, actual.3
            )));
        }

        Ok(())
    }

    /// Validate that tensor has batch size of 1 and single channel
    pub fn validate_single_batch_single_channel(tensor: &Array4<f32>) -> Result<()> {
        let shape = tensor.shape();
        if shape.get(0).copied().unwrap_or(0) != 1 || shape.get(1).copied().unwrap_or(0) != 1 {
            return Err(BgRemovalError::processing(
                "Tensor must have batch size 1 and single channel for mask generation",
            ));
        }
        Ok(())
    }

    /// Validate image dimensions are within reasonable bounds
    pub fn validate_image_dimensions(width: u32, height: u32) -> Result<()> {
        const MAX_DIMENSION: u32 = 16384; // 16K pixels
        const MIN_DIMENSION: u32 = 1;

        if width < MIN_DIMENSION || height < MIN_DIMENSION {
            return Err(BgRemovalError::invalid_config(format!(
                "Image dimensions too small: {}x{}. Minimum: {}x{}",
                width, height, MIN_DIMENSION, MIN_DIMENSION
            )));
        }

        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(BgRemovalError::invalid_config(format!(
                "Image dimensions too large: {}x{}. Maximum: {}x{}",
                width, height, MAX_DIMENSION, MAX_DIMENSION
            )));
        }

        Ok(())
    }

    /// Validate canvas dimensions for preprocessing
    pub fn validate_canvas_dimensions(width: u32, height: u32, max_size: u32) -> Result<()> {
        if width > max_size || height > max_size {
            return Err(BgRemovalError::processing(format!(
                "Canvas dimensions {}x{} exceed maximum size {}",
                width, height, max_size
            )));
        }
        Ok(())
    }

    /// Validate tensor coordinate bounds
    pub fn validate_tensor_coordinates(tensor: &Array4<f32>, x: usize, y: usize) -> Result<()> {
        let shape = tensor.shape();
        let height = shape.get(2).copied().unwrap_or(0);
        let width = shape.get(3).copied().unwrap_or(0);

        if x >= width {
            return Err(BgRemovalError::processing(format!(
                "X coordinate {} out of bounds (width: {})",
                x, width
            )));
        }

        if y >= height {
            return Err(BgRemovalError::processing(format!(
                "Y coordinate {} out of bounds (height: {})",
                y, height
            )));
        }

        Ok(())
    }

    /// Validate tensor values are within expected range
    pub fn validate_tensor_value_range(tensor: &Array4<f32>, min: f32, max: f32) -> Result<()> {
        for value in tensor {
            if *value < min || *value > max {
                return Err(BgRemovalError::processing(format!(
                    "Tensor value {} out of range [{}, {}]",
                    value, min, max
                )));
            }
            if !value.is_finite() {
                return Err(BgRemovalError::processing(
                    "Tensor contains non-finite values (NaN or infinity)",
                ));
            }
        }
        Ok(())
    }

    /// Validate mask dimensions match original image dimensions
    pub fn validate_mask_dimensions(mask_data_len: usize, width: u32, height: u32) -> Result<()> {
        let expected_len = (width * height) as usize;
        if mask_data_len != expected_len {
            return Err(BgRemovalError::processing(format!(
                "Mask data length {} doesn't match image dimensions {}x{} (expected {} pixels)",
                mask_data_len, width, height, expected_len
            )));
        }
        Ok(())
    }

    /// Validate preprocessing target size
    pub fn validate_target_size(size: &[u64]) -> Result<(u32, u32)> {
        if size.len() != 2 {
            return Err(BgRemovalError::invalid_config(format!(
                "Target size must have exactly 2 dimensions, got {}",
                size.len()
            )));
        }

        let width = size.first().copied().unwrap_or(0) as u32;
        let height = size.get(1).copied().unwrap_or(0) as u32;

        Self::validate_image_dimensions(width, height)?;

        // Additional validation for preprocessing target size
        if width != height {
            log::warn!(
                "Non-square target size {}x{} may affect model performance",
                width,
                height
            );
        }

        Ok((width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_validate_tensor_shape() {
        // Valid 4D tensor
        let tensor = Array::zeros((1, 1, 256, 256));
        assert!(TensorValidator::validate_tensor_shape(&tensor, (1, 1, 256, 256)).is_ok());

        // Wrong shape
        assert!(TensorValidator::validate_tensor_shape(&tensor, (1, 3, 256, 256)).is_err());
        assert!(TensorValidator::validate_tensor_shape(&tensor, (2, 1, 256, 256)).is_err());
    }

    #[test]
    fn test_validate_single_batch_single_channel() {
        // Valid single batch, single channel
        let valid_tensor = Array::zeros((1, 1, 256, 256));
        assert!(TensorValidator::validate_single_batch_single_channel(&valid_tensor).is_ok());

        // Invalid batch size
        let invalid_batch = Array::zeros((2, 1, 256, 256));
        assert!(TensorValidator::validate_single_batch_single_channel(&invalid_batch).is_err());

        // Invalid channel count
        let invalid_channels = Array::zeros((1, 3, 256, 256));
        assert!(TensorValidator::validate_single_batch_single_channel(&invalid_channels).is_err());
    }

    #[test]
    fn test_validate_image_dimensions() {
        // Valid dimensions
        assert!(TensorValidator::validate_image_dimensions(1920, 1080).is_ok());
        assert!(TensorValidator::validate_image_dimensions(100, 100).is_ok());

        // Too small
        assert!(TensorValidator::validate_image_dimensions(0, 100).is_err());
        assert!(TensorValidator::validate_image_dimensions(100, 0).is_err());

        // Too large
        assert!(TensorValidator::validate_image_dimensions(20000, 1080).is_err());
        assert!(TensorValidator::validate_image_dimensions(1920, 20000).is_err());
    }

    #[test]
    fn test_validate_tensor_coordinates() {
        let tensor = Array::zeros((1, 1, 100, 200));

        // Valid coordinates
        assert!(TensorValidator::validate_tensor_coordinates(&tensor, 0, 0).is_ok());
        assert!(TensorValidator::validate_tensor_coordinates(&tensor, 199, 99).is_ok());

        // Out of bounds
        assert!(TensorValidator::validate_tensor_coordinates(&tensor, 200, 50).is_err());
        assert!(TensorValidator::validate_tensor_coordinates(&tensor, 50, 100).is_err());
    }

    #[test]
    fn test_validate_tensor_value_range() {
        // Valid range
        let valid_tensor = Array::from_elem((1, 1, 2, 2), 0.5);
        assert!(TensorValidator::validate_tensor_value_range(&valid_tensor, 0.0, 1.0).is_ok());

        // Out of range
        let invalid_tensor = Array::from_elem((1, 1, 2, 2), 1.5);
        assert!(TensorValidator::validate_tensor_value_range(&invalid_tensor, 0.0, 1.0).is_err());

        // NaN values
        let nan_tensor = Array::from_elem((1, 1, 2, 2), f32::NAN);
        assert!(TensorValidator::validate_tensor_value_range(&nan_tensor, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_validate_mask_dimensions() {
        // Valid mask
        assert!(TensorValidator::validate_mask_dimensions(1000, 25, 40).is_ok()); // 25 * 40 = 1000

        // Invalid mask length
        assert!(TensorValidator::validate_mask_dimensions(999, 25, 40).is_err());
        assert!(TensorValidator::validate_mask_dimensions(1001, 25, 40).is_err());
    }

    #[test]
    fn test_validate_target_size() {
        // Valid target sizes
        assert!(TensorValidator::validate_target_size(&[1024, 1024]).is_ok());
        assert!(TensorValidator::validate_target_size(&[512, 768]).is_ok());

        // Invalid number of dimensions
        assert!(TensorValidator::validate_target_size(&[1024]).is_err());
        assert!(TensorValidator::validate_target_size(&[1024, 1024, 3]).is_err());

        // Invalid dimension values
        assert!(TensorValidator::validate_target_size(&[0, 1024]).is_err());
        assert!(TensorValidator::validate_target_size(&[20000, 1024]).is_err());
    }
}
