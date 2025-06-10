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
    /// Load model data as bytes (uses embedded model based on compile-time feature)
    fn load_model_data(&self) -> Result<Vec<u8>>;
    
    /// Get model information (uses embedded model based on compile-time feature)
    fn get_model_info(&self) -> Result<ModelInfo>;
}

/// Embedded model provider (models compiled into binary)
pub struct EmbeddedModelProvider;

impl ModelProvider for EmbeddedModelProvider {
    fn load_model_data(&self) -> Result<Vec<u8>> {
        #[cfg(feature = "fp32-model")]
        {
            // Load embedded FP32 ISNet model
            Ok(include_bytes!("../../../models/isnet_fp32.onnx").to_vec())
        }
        #[cfg(feature = "fp16-model")]
        {
            // Load embedded FP16 ISNet model  
            Ok(include_bytes!("../../../models/isnet_fp16.onnx").to_vec())
        }
        #[cfg(not(any(feature = "fp32-model", feature = "fp16-model")))]
        {
            compile_error!("Either fp32-model or fp16-model feature must be enabled")
        }
    }

    fn get_model_info(&self) -> Result<ModelInfo> {
        #[cfg(feature = "fp32-model")]
        {
            Ok(ModelInfo {
                name: "ISNet-FP32".to_string(),
                precision: ModelPrecision::Fp32,
                size_bytes: 168 * 1024 * 1024, // ~168MB (actual size)
                input_shape: (1, 3, 1024, 1024),
                output_shape: (1, 1, 1024, 1024),
            })
        }
        #[cfg(feature = "fp16-model")]
        {
            Ok(ModelInfo {
                name: "ISNet-FP16".to_string(),
                precision: ModelPrecision::Fp16,
                size_bytes: 84 * 1024 * 1024, // ~84MB (actual size)
                input_shape: (1, 3, 1024, 1024),
                output_shape: (1, 1, 1024, 1024),
            })
        }
        #[cfg(not(any(feature = "fp32-model", feature = "fp16-model")))]
        {
            compile_error!("Either fp32-model or fp16-model feature must be enabled")
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
        
        // The name and precision depend on which feature is enabled
        #[cfg(feature = "fp16-model")]
        {
            assert_eq!(info.name, "ISNet-FP16");
            assert_eq!(info.precision, ModelPrecision::Fp16);
        }
        
        #[cfg(feature = "fp32-model")]
        {
            assert_eq!(info.name, "ISNet-FP32");
            assert_eq!(info.precision, ModelPrecision::Fp32);
        }
    }

    #[test]
    fn test_model_manager() {
        let manager = ModelManager::with_embedded();
        let info = manager.get_info().unwrap();
        
        // Verify the model is correctly loaded based on feature flags
        #[cfg(feature = "fp16-model")]
        assert_eq!(info.precision, ModelPrecision::Fp16);
        
        #[cfg(feature = "fp32-model")]
        assert_eq!(info.precision, ModelPrecision::Fp32);
    }
}