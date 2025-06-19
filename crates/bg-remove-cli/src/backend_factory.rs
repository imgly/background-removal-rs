//! Backend factory implementation for CLI that injects ONNX and Tract backends

use bg_remove_core::{
    error::Result,
    inference::InferenceBackend,
    models::ModelManager,
    processor::{BackendFactory, BackendType},
};
use bg_remove_onnx::OnnxBackend;
use bg_remove_tract::TractBackend;

/// CLI backend factory that provides access to both ONNX and Tract backends
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
            BackendType::Mock => {
                let backend = bg_remove_core::MockBackend::new();
                Ok(Box::new(backend))
            },
        }
    }

    fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::Onnx, BackendType::Tract, BackendType::Mock]
    }
}

impl CliBackendFactory {
    /// Create a new CLI backend factory
    pub(crate) fn new() -> Self {
        Self
    }
}
