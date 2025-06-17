//! Tract backend factory implementation for Web/WASM environments

use bg_remove_core::{
    processor::{BackendFactory, BackendType},
    inference::InferenceBackend,
    models::ModelManager,
    error::Result,
};
use bg_remove_tract::TractBackend;

/// Web backend factory that provides access to Tract backend for WASM compatibility
pub(crate) struct WebBackendFactory;

impl BackendFactory for WebBackendFactory {
    fn create_backend(
        &self,
        backend_type: BackendType,
        model_manager: ModelManager,
    ) -> Result<Box<dyn InferenceBackend>> {
        match backend_type {
            BackendType::Tract => {
                let backend = TractBackend::with_model_manager(model_manager);
                Ok(Box::new(backend))
            }
            BackendType::Mock => {
                let backend = bg_remove_core::MockBackend::new();
                Ok(Box::new(backend))
            }
            BackendType::Onnx => {
                // ONNX Runtime is not supported in WASM
                Err(bg_remove_core::error::BgRemovalError::invalid_config(
                    "ONNX Runtime backend is not supported in WebAssembly environments. Use Tract backend instead."
                ))
            }
        }
    }
    
    fn available_backends(&self) -> Vec<BackendType> {
        vec![
            BackendType::Tract,
            BackendType::Mock,
        ]
    }
}

impl WebBackendFactory {
    /// Create a new Web backend factory
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Default for WebBackendFactory {
    fn default() -> Self {
        Self::new()
    }
}