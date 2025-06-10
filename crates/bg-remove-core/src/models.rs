//! Model management and embedding system

use crate::{config::ModelPrecision, error::Result};

/// Model information and metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub precision: ModelPrecision,
    pub size_bytes: usize,
    pub input_shape: (usize, usize, usize, usize), // NCHW format
    pub output_shape: (usize, usize, usize, usize),
}

/// Model provider trait for loading models
pub trait ModelProvider {
    /// Load model data as bytes
    fn load_model_data(&self, precision: ModelPrecision) -> Result<Vec<u8>>;
    
    /// Get model information
    fn get_model_info(&self, precision: ModelPrecision) -> Result<ModelInfo>;
}

/// Embedded model provider (models compiled into binary)
pub struct EmbeddedModelProvider;

impl ModelProvider for EmbeddedModelProvider {
    fn load_model_data(&self, precision: ModelPrecision) -> Result<Vec<u8>> {
        match precision {
            ModelPrecision::Fp32 => {
                // Load embedded FP32 ISNet model
                Ok(include_bytes!("../../../models/isnet_fp32.onnx").to_vec())
            }
            ModelPrecision::Fp16 => {
                // Load embedded FP16 ISNet model  
                Ok(include_bytes!("../../../models/isnet_fp16.onnx").to_vec())
            }
        }
    }

    fn get_model_info(&self, precision: ModelPrecision) -> Result<ModelInfo> {
        match precision {
            ModelPrecision::Fp32 => Ok(ModelInfo {
                name: "ISNet-FP32".to_string(),
                precision,
                size_bytes: 168 * 1024 * 1024, // ~168MB (actual size)
                input_shape: (1, 3, 1024, 1024),
                output_shape: (1, 1, 1024, 1024),
            }),
            ModelPrecision::Fp16 => Ok(ModelInfo {
                name: "ISNet-FP16".to_string(),
                precision,
                size_bytes: 84 * 1024 * 1024, // ~84MB (actual size)
                input_shape: (1, 3, 1024, 1024),
                output_shape: (1, 1, 1024, 1024),
            }),
        }
    }
}


/// Model manager for handling different model sources
pub struct ModelManager {
    provider: Box<dyn ModelProvider>,
}

impl ModelManager {
    /// Create a new model manager with embedded models
    pub fn with_embedded() -> Self {
        Self {
            provider: Box::new(EmbeddedModelProvider),
        }
    }


    /// Load model data for the specified precision
    pub fn load_model(&self, precision: ModelPrecision) -> Result<Vec<u8>> {
        self.provider.load_model_data(precision)
    }

    /// Get model information
    pub fn get_info(&self, precision: ModelPrecision) -> Result<ModelInfo> {
        self.provider.get_model_info(precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_model_provider() {
        let provider = EmbeddedModelProvider;
        
        // Test model info retrieval
        let info = provider.get_model_info(ModelPrecision::Fp16).unwrap();
        assert_eq!(info.name, "ISNet-FP16");
        assert_eq!(info.precision, ModelPrecision::Fp16);
        
        let info = provider.get_model_info(ModelPrecision::Fp32).unwrap();
        assert_eq!(info.name, "ISNet-FP32");
        assert_eq!(info.precision, ModelPrecision::Fp32);
    }

    #[test]
    fn test_model_manager() {
        let manager = ModelManager::with_embedded();
        let info = manager.get_info(ModelPrecision::Fp16).unwrap();
        assert_eq!(info.precision, ModelPrecision::Fp16);
    }
}