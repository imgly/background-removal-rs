//! Inference backend abstraction and registry

use crate::{config::RemovalConfig, error::Result};
use ndarray::Array4;

// Use instant crate for cross-platform time compatibility
use instant::Duration;

/// Trait for inference backends
pub trait InferenceBackend {
    /// Initialize the backend with the given configuration
    ///
    /// # Errors
    /// - Backend initialization failures
    /// - Model loading or validation errors
    /// - Invalid configuration parameters
    fn initialize(&mut self, config: &RemovalConfig) -> Result<Option<Duration>>;

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
        // Register default backends
        // TODO: ONNX backend moved to separate crate - need backend injection mechanism
        // Backends must now be injected via BackendFactory pattern

        Self {
            backends: std::collections::HashMap::new(),
        }
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
    use crate::backends::test_utils::{MockOnnxBackend, MockTractBackend};

    #[test]
    fn test_backend_registry() {
        let registry = BackendRegistry::new();

        // Registry should be empty by default - backends injected via factory pattern
        // This test validates the registry can be created without mock backends
        assert_eq!(registry.backends.len(), 0);
    }

    #[test]
    fn test_backend_registry_registration() {
        let mut registry = BackendRegistry::new();

        // Test registering mock backends
        let mock_onnx = Box::new(MockOnnxBackend::new()) as Box<dyn InferenceBackend>;
        let mock_tract = Box::new(MockTractBackend::new()) as Box<dyn InferenceBackend>;

        registry.register("onnx", mock_onnx);
        registry.register("tract", mock_tract);

        assert_eq!(registry.backends.len(), 2);

        // Test retrieving backends
        let onnx_backend = registry.get("onnx");
        assert!(onnx_backend.is_some());

        let tract_backend = registry.get("tract");
        assert!(tract_backend.is_some());

        // Test non-existent backend
        let invalid_backend = registry.get("invalid");
        assert!(invalid_backend.is_none());
    }

    #[test]
    fn test_backend_registry_backend_operations() {
        let mut registry = BackendRegistry::new();

        // Register a mock backend
        let mock_backend = Box::new(MockOnnxBackend::new()) as Box<dyn InferenceBackend>;
        registry.register("test_backend", mock_backend);

        // Test backend operations
        if let Some(backend) = registry.get("test_backend") {
            // Test initial state
            assert!(!backend.is_initialized());

            // Test shape queries
            let input_shape = backend.input_shape();
            let output_shape = backend.output_shape();

            assert_eq!(input_shape.0, 1); // Batch size
            assert_eq!(input_shape.1, 3); // RGB channels
            assert_eq!(output_shape.0, 1); // Batch size
            assert_eq!(output_shape.1, 1); // Single channel mask

            // Test preprocessing config
            let preprocessing_result = backend.get_preprocessing_config();
            if let Ok(config) = preprocessing_result {
                assert!(config.target_size[0] > 0);
                assert!(config.target_size[1] > 0);
                assert_eq!(config.normalization_mean.len(), 3);
                assert_eq!(config.normalization_std.len(), 3);
            }

            // Test model info
            let model_info_result = backend.get_model_info();
            if let Ok(info) = model_info_result {
                assert!(!info.name.is_empty());
                assert!(info.size_bytes > 0);
            }
        } else {
            panic!("Backend should be registered");
        }
    }

    #[test]
    fn test_backend_registry_multiple_backends() {
        let mut registry = BackendRegistry::new();

        // Register multiple different backends
        let backends = vec![
            (
                "onnx1",
                Box::new(MockOnnxBackend::new()) as Box<dyn InferenceBackend>,
            ),
            (
                "onnx2",
                Box::new(MockOnnxBackend::new()) as Box<dyn InferenceBackend>,
            ),
            (
                "tract1",
                Box::new(MockTractBackend::new()) as Box<dyn InferenceBackend>,
            ),
            (
                "tract2",
                Box::new(MockTractBackend::new()) as Box<dyn InferenceBackend>,
            ),
        ];

        for (name, backend) in backends {
            registry.register(name, backend);
        }

        assert_eq!(registry.backends.len(), 4);

        // Verify all backends are accessible
        for name in &["onnx1", "onnx2", "tract1", "tract2"] {
            assert!(
                registry.get(name).is_some(),
                "Backend {} should be available",
                name
            );
        }
    }

    #[test]
    fn test_backend_registry_backend_replacement() {
        let mut registry = BackendRegistry::new();

        // Register initial backend
        let backend1 = Box::new(MockOnnxBackend::new()) as Box<dyn InferenceBackend>;
        registry.register("test", backend1);
        assert_eq!(registry.backends.len(), 1);

        // Replace with different backend
        let backend2 = Box::new(MockTractBackend::new()) as Box<dyn InferenceBackend>;
        registry.register("test", backend2);
        assert_eq!(registry.backends.len(), 1); // Should still be 1, not 2

        // Verify the backend is accessible
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_backend_trait_consistency() {
        // Test that both mock backends implement the trait consistently
        let onnx_backend = MockOnnxBackend::new();
        let tract_backend = MockTractBackend::new();

        // Both should start uninitialized
        assert!(!onnx_backend.is_initialized());
        assert!(!tract_backend.is_initialized());

        // Both should provide valid shapes
        let onnx_input = onnx_backend.input_shape();
        let onnx_output = onnx_backend.output_shape();
        let tract_input = tract_backend.input_shape();
        let tract_output = tract_backend.output_shape();

        // All shapes should be valid (positive dimensions)
        assert!(onnx_input.0 > 0 && onnx_input.1 > 0 && onnx_input.2 > 0 && onnx_input.3 > 0);
        assert!(onnx_output.0 > 0 && onnx_output.1 > 0 && onnx_output.2 > 0 && onnx_output.3 > 0);
        assert!(tract_input.0 > 0 && tract_input.1 > 0 && tract_input.2 > 0 && tract_input.3 > 0);
        assert!(
            tract_output.0 > 0 && tract_output.1 > 0 && tract_output.2 > 0 && tract_output.3 > 0
        );

        // Input should be RGB (3 channels), output should be single channel
        assert_eq!(onnx_input.1, 3);
        assert_eq!(onnx_output.1, 1);
        assert_eq!(tract_input.1, 3);
        assert_eq!(tract_output.1, 1);
    }
}
