//! Mock backend implementation for testing and debugging

use crate::config::RemovalConfig;
use crate::error::Result;
use crate::inference::InferenceBackend;
use ndarray::Array4;

/// Mock backend for testing and debugging purposes
///
/// This backend provides a simple edge detection algorithm as a mock
/// segmentation mask, useful for testing without requiring actual model files.
#[derive(Debug)]
pub struct MockBackend {
    input_shape: (usize, usize, usize, usize),
    output_shape: (usize, usize, usize, usize),
}

impl MockBackend {
    /// Create a new mock backend
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_shape: (1, 3, 1024, 1024),
            output_shape: (1, 1, 1024, 1024),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockBackend {
    fn initialize(&mut self, _config: &RemovalConfig) -> Result<Option<std::time::Duration>> {
        // Mock backend doesn't need initialization and has no model loading time
        Ok(None)
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        let (n, _c, h, w) = input.dim();

        // Create a mock segmentation mask (simple edge detection)
        let mut output = Array4::<f32>::zeros((n, 1, h, w));

        for batch in 0..n {
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    // Simple edge detection as mock segmentation
                    let center = input[[batch, 0, y, x]];
                    let left = input[[batch, 0, y, x - 1]];
                    let right = input[[batch, 0, y, x + 1]];
                    let top = input[[batch, 0, y - 1, x]];
                    let bottom = input[[batch, 0, y + 1, x]];

                    let edge_strength = ((center - left).abs()
                        + (center - right).abs()
                        + (center - top).abs()
                        + (center - bottom).abs())
                        / 4.0;

                    // Create a reasonable mock mask - use safe indexing
                    if let Some(elem) = output.get_mut([batch, 0, y, x]) {
                        *elem = if edge_strength > 0.1 { 1.0 } else { 0.0 };
                    }
                }
            }
        }

        Ok(output)
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        self.input_shape
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        self.output_shape
    }

    fn get_preprocessing_config(&self) -> Result<crate::models::PreprocessingConfig> {
        // Return a default preprocessing config for mock backend
        Ok(crate::models::PreprocessingConfig {
            target_size: [1024, 1024],
            normalization_mean: [0.485, 0.456, 0.406],
            normalization_std: [0.229, 0.224, 0.225],
        })
    }

    fn get_model_info(&self) -> Result<crate::models::ModelInfo> {
        // Return mock model info
        Ok(crate::models::ModelInfo {
            name: "Mock Backend".to_string(),
            precision: "mock".to_string(),
            size_bytes: 0,
            input_shape: self.input_shape,
            output_shape: self.output_shape,
        })
    }

    fn is_initialized(&self) -> bool {
        true // Mock backend is always "initialized"
    }
}
