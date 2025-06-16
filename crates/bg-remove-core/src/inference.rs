//! Inference backend abstraction and registry

use crate::{
    backends::MockBackend,
    config::RemovalConfig,
    error::Result,
};
use ndarray::Array4;

/// Trait for inference backends
pub trait InferenceBackend {
    /// Initialize the backend with the given configuration
    ///
    /// # Errors
    /// - Backend initialization failures
    /// - Model loading or validation errors
    /// - Invalid configuration parameters
    fn initialize(&mut self, config: &RemovalConfig) -> Result<Option<std::time::Duration>>;

    /// Run inference on the input tensor
    ///
    /// # Errors
    /// - Backend not initialized
    /// - Model inference failures
    /// - Tensor conversion or processing errors
    /// - Invalid input tensor dimensions
    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>>;

    /// Get the expected input shape for this backend
    fn input_shape(&self) -> (usize, usize, usize, usize);

    /// Get the expected output shape for this backend
    fn output_shape(&self) -> (usize, usize, usize, usize);

    /// Get preprocessing configuration for this backend
    ///
    /// # Errors
    /// - Model manager not initialized
    /// - Invalid or missing preprocessing configuration
    fn get_preprocessing_config(&self) -> Result<crate::models::PreprocessingConfig>;

    /// Get model information for this backend
    ///
    /// # Errors
    /// - Model manager not initialized
    /// - Model metadata unavailable or invalid
    fn get_model_info(&self) -> Result<crate::models::ModelInfo>;

    /// Check if backend is initialized
    fn is_initialized(&self) -> bool;
}

/// Backend registry for managing different inference backends
pub struct BackendRegistry {
    backends: std::collections::HashMap<String, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self {
            backends: std::collections::HashMap::new(),
        };

        // Register default backends
        // TODO: ONNX backend moved to separate crate - need backend injection mechanism
        // let onnx_backend = OnnxBackend::new();
        // registry.register("onnx", Box::new(onnx_backend));
        registry.register("mock", Box::new(MockBackend::new()));

        registry
    }

    pub fn register(&mut self, name: &str, backend: Box<dyn InferenceBackend>) {
        self.backends.insert(name.to_string(), backend);
    }

    pub fn get(&mut self, name: &str) -> Option<&mut Box<dyn InferenceBackend>> {
        self.backends.get_mut(name)
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RemovalConfig;

    #[test]
    fn test_mock_backend() {
        let mut backend = MockBackend::new();
        let config = RemovalConfig::default();

        backend.initialize(&config).unwrap();

        // Test shapes
        assert_eq!(backend.input_shape(), (1, 3, 1024, 1024));
        assert_eq!(backend.output_shape(), (1, 1, 1024, 1024));

        // Test inference with small tensor
        let input = Array4::<f32>::zeros((1, 3, 4, 4));
        let output = backend.infer(&input).unwrap();
        assert_eq!(output.dim(), (1, 1, 4, 4));
    }

    #[test]
    fn test_backend_registry() {
        let mut registry = BackendRegistry::new();

        // Test that default backends are registered
        assert!(registry.get("mock").is_some());
        assert!(registry.get("onnx").is_some());
        assert!(registry.get("nonexistent").is_none());
    }
}
