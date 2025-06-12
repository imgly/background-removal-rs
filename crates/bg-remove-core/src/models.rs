//! Model management and embedding system

use crate::error::Result;

// Include generated model configuration constants
include!(concat!(env!("OUT_DIR"), "/model_config.rs"));

/// Model information and metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub precision: String,
    pub size_bytes: usize,
    pub input_shape: (usize, usize, usize, usize), // NCHW format
    pub output_shape: (usize, usize, usize, usize),
}

/// Model provider trait for loading models
pub trait ModelProvider: std::fmt::Debug {
    /// Load model data as bytes (uses embedded model based on compile-time feature)
    fn load_model_data(&self) -> Result<Vec<u8>>;

    /// Get model information (uses embedded model based on compile-time feature)
    fn get_model_info(&self) -> Result<ModelInfo>;
}

/// Embedded model provider (models compiled into binary)
#[derive(Debug)]
pub struct EmbeddedModelProvider;

impl ModelProvider for EmbeddedModelProvider {
    fn load_model_data(&self) -> Result<Vec<u8>> {
        // Load model using generated function
        Ok(load_embedded_model())
    }

    fn get_model_info(&self) -> Result<ModelInfo> {
        // Use generated constants from model.json
        Ok(ModelInfo {
            name: format!("{}-{}", EMBEDDED_MODEL_NAME, EMBEDDED_MODEL_VARIANT.to_uppercase()),
            precision: EMBEDDED_MODEL_VARIANT.to_string(),
            size_bytes: 0, // TODO: Could be calculated from file size at build time
            input_shape: (
                EMBEDDED_INPUT_SHAPE[0],
                EMBEDDED_INPUT_SHAPE[1], 
                EMBEDDED_INPUT_SHAPE[2],
                EMBEDDED_INPUT_SHAPE[3]
            ),
            output_shape: (
                EMBEDDED_OUTPUT_SHAPE[0],
                EMBEDDED_OUTPUT_SHAPE[1],
                EMBEDDED_OUTPUT_SHAPE[2], 
                EMBEDDED_OUTPUT_SHAPE[3]
            ),
        })
    }
}

/// Model manager for handling different model sources
#[derive(Debug)]
pub struct ModelManager {
    provider: Box<dyn ModelProvider>,
}

impl ModelManager {
    /// Create a new model manager with embedded models
    #[must_use] pub fn with_embedded() -> Self {
        Self {
            provider: Box::new(EmbeddedModelProvider),
        }
    }

    /// Load model data (uses embedded model based on compile-time feature)
    pub fn load_model(&self) -> Result<Vec<u8>> {
        self.provider.load_model_data()
    }

    /// Get model information (uses embedded model based on compile-time feature)
    pub fn get_info(&self) -> Result<ModelInfo> {
        self.provider.get_model_info()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_model_provider() {
        let provider = EmbeddedModelProvider;

        // Test model info retrieval (uses compile-time embedded model)
        let info = provider.get_model_info().unwrap();

        // FP32 takes precedence when both features are enabled
        #[cfg(feature = "fp32-model")]
        {
            assert_eq!(info.name, "ISNet-FP32");
            assert_eq!(info.precision, "fp32");
        }

        // FP16 only when FP32 is not enabled
        #[cfg(all(feature = "fp16-model", not(feature = "fp32-model")))]
        {
            assert_eq!(info.name, "ISNet-FP16");
            assert_eq!(info.precision, "fp16");
        }
    }

    #[test]
    fn test_model_manager() {
        let manager = ModelManager::with_embedded();
        let info = manager.get_info().unwrap();

        // Verify the model is correctly loaded based on feature precedence
        #[cfg(feature = "fp32-model")]
        assert_eq!(info.precision, "fp32");

        #[cfg(all(feature = "fp16-model", not(feature = "fp32-model")))]
        assert_eq!(info.precision, "fp16");
    }
}
