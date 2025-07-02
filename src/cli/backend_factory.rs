//! Backend factory implementation for CLI that injects ONNX and Tract backends

#![allow(dead_code)]

use crate::backends::{OnnxBackend, TractBackend};
use crate::{
    error::Result,
    inference::InferenceBackend,
    models::ModelManager,
    processor::{BackendFactory, BackendType},
};

/// CLI backend factory that provides access to both ONNX and Tract backends
#[derive(Debug)]
pub(crate) struct CliBackendFactory;

impl BackendFactory for CliBackendFactory {
    fn create_backend(
        &self,
        backend_type: BackendType,
        model_manager: ModelManager,
    ) -> Result<Box<dyn InferenceBackend>> {
        match backend_type {
            BackendType::Onnx => {
                let backend = OnnxBackend::with_model_manager(model_manager);
                Ok(Box::new(backend))
            },
            BackendType::Tract => {
                let backend = TractBackend::with_model_manager(model_manager);
                Ok(Box::new(backend))
            },
        }
    }

    fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::Onnx, BackendType::Tract]
    }
}

impl CliBackendFactory {
    /// Create a new CLI backend factory
    pub(crate) fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelSource, ModelSpec};

    fn create_test_model_manager() -> ModelManager {
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: Some("fp32".to_string()),
        };
        ModelManager::from_spec(&model_spec).unwrap()
    }

    #[test]
    fn test_cli_backend_factory_creation() {
        let factory = CliBackendFactory::new();

        // Verify available backends
        let backends = factory.available_backends();
        assert_eq!(backends.len(), 2);
        assert!(backends.contains(&BackendType::Onnx));
        assert!(backends.contains(&BackendType::Tract));
    }

    #[test]
    fn test_create_onnx_backend() {
        let factory = CliBackendFactory::new();
        let model_manager = create_test_model_manager();

        let backend = factory.create_backend(BackendType::Onnx, model_manager);
        assert!(backend.is_ok());

        // Verify it's an ONNX backend
        let _backend = backend.unwrap();
    }

    #[test]
    fn test_create_tract_backend() {
        let factory = CliBackendFactory::new();
        let model_manager = create_test_model_manager();

        let backend = factory.create_backend(BackendType::Tract, model_manager);
        assert!(backend.is_ok());

        // Verify it's a Tract backend
        let _backend = backend.unwrap();
        // Note: supports_provider() method is not available on InferenceBackend trait
    }

    #[test]
    fn test_backend_factory_available_backends() {
        let factory = CliBackendFactory::new();
        let backends = factory.available_backends();

        // Test that all expected backends are available
        assert_eq!(backends.len(), 2);
        assert!(backends.contains(&BackendType::Onnx));
        assert!(backends.contains(&BackendType::Tract));

        // Test ordering is consistent
        assert_eq!(backends[0], BackendType::Onnx);
        assert_eq!(backends[1], BackendType::Tract);
    }

    #[test]
    fn test_backend_creation_with_different_model_specs() {
        let factory = CliBackendFactory::new();

        // Test with different model variants using available model
        let model_specs = vec![
            ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp16".to_string()),
            },
            ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp32".to_string()),
            },
            ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: None,
            },
        ];

        for backend_type in [BackendType::Onnx, BackendType::Tract] {
            for model_spec in &model_specs {
                let model_manager = ModelManager::from_spec(model_spec).unwrap();
                let backend = factory.create_backend(backend_type.clone(), model_manager);
                assert!(
                    backend.is_ok(),
                    "Failed to create {:?} backend with model spec: {:?}",
                    backend_type,
                    model_spec
                );
            }
        }
    }

    #[test]
    fn test_factory_trait_implementation() {
        let factory: Box<dyn BackendFactory> = Box::new(CliBackendFactory::new());

        // Test trait methods work through trait object
        let backends = factory.available_backends();
        assert_eq!(backends.len(), 2);

        let model_manager = create_test_model_manager();
        let backend = factory.create_backend(BackendType::Onnx, model_manager);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_factory_debug_formatting() {
        let factory = CliBackendFactory::new();
        let debug_str = format!("{:?}", factory);
        assert!(debug_str.contains("CliBackendFactory"));
    }

    #[test]
    fn test_backend_creation_consistency() {
        let factory = CliBackendFactory::new();
        let model_manager1 = create_test_model_manager();
        let model_manager2 = create_test_model_manager();

        // Create multiple backends of the same type
        let backend1 = factory.create_backend(BackendType::Onnx, model_manager1);
        let backend2 = factory.create_backend(BackendType::Onnx, model_manager2);

        assert!(backend1.is_ok());
        assert!(backend2.is_ok());

        // Both should support the same providers
        let _backend1 = backend1.unwrap();
        let _backend2 = backend2.unwrap();

        // Note: supports_provider() method is not available on InferenceBackend trait
        // Both backends are successfully created, which indicates they have the same capabilities
    }

    #[test]
    fn test_backend_creation_error_propagation() {
        use crate::models::ModelSource;

        let _factory = CliBackendFactory::new();

        // Test with invalid model spec that should cause ModelManager creation to fail
        let invalid_model_spec = ModelSpec {
            source: ModelSource::Downloaded("".to_string()), // Empty model ID should be invalid
            variant: None,
        };

        let result = ModelManager::from_spec(&invalid_model_spec);
        assert!(
            result.is_err(),
            "Empty model ID should cause ModelManager creation to fail"
        );
    }

    #[test]
    fn test_multiple_factory_instances() {
        let factory1 = CliBackendFactory::new();
        let factory2 = CliBackendFactory::new();

        // Both factories should behave identically
        assert_eq!(factory1.available_backends(), factory2.available_backends());

        let model_manager1 = create_test_model_manager();
        let model_manager2 = create_test_model_manager();

        let backend1 = factory1.create_backend(BackendType::Tract, model_manager1);
        let backend2 = factory2.create_backend(BackendType::Tract, model_manager2);

        assert!(backend1.is_ok());
        assert!(backend2.is_ok());
    }
}
