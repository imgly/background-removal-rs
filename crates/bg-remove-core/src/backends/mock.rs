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
    #[must_use] pub fn new() -> Self {
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
    fn initialize(&mut self, _config: &RemovalConfig) -> Result<()> {
        // Mock backend doesn't need initialization
        Ok(())
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

                    // Create a reasonable mock mask
                    output[[batch, 0, y, x]] = if edge_strength > 0.1 { 1.0 } else { 0.0 };
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
}
