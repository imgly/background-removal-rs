//! Mock backend implementation for testing and debugging

use crate::config::RemovalConfig;
use crate::error::Result;
use crate::inference::InferenceBackend;
use ndarray::Array4;

// Use instant crate for cross-platform time compatibility
use instant::Duration;

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
    fn initialize(&mut self, _config: &RemovalConfig) -> Result<Option<Duration>> {
        // Mock backend doesn't need initialization and has no model loading time
        Ok(None)
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        let (n, _c, h, w) = input.dim();

        // Create a mock segmentation mask (simple center circle/square detection)
        let mut output = Array4::<f32>::zeros((n, 1, h, w));

        for batch in 0..n {
            // Create a simple mask with a circle/square in the center
            let center_y = h / 2;
            let center_x = w / 2;
            let radius = (h.min(w) / 3) as f32;
            
            for y in 0..h {
                for x in 0..w {
                    // Calculate distance from center
                    let dy = (y as f32 - center_y as f32).abs();
                    let dx = (x as f32 - center_x as f32).abs();
                    let distance = (dy * dy + dx * dx).sqrt();
                    
                    // Create a circular mask with soft edges
                    let mask_value = if distance < radius {
                        1.0
                    } else if distance < radius + 50.0 {
                        // Soft edge falloff
                        1.0 - (distance - radius) / 50.0
                    } else {
                        0.0
                    };
                    
                    // Set the mask value
                    if let Some(elem) = output.get_mut([batch, 0, y, x]) {
                        *elem = mask_value.clamp(0.0, 1.0);
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
