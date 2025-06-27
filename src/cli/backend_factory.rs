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
