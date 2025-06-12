//! ONNX Runtime backend implementation for `ISNet` model inference

use crate::config::{ExecutionProvider, RemovalConfig};
use crate::error::Result;
use crate::inference::InferenceBackend;
use crate::models::{ModelManager, EMBEDDED_INPUT_NAME, EMBEDDED_OUTPUT_NAME, EMBEDDED_INPUT_SHAPE, EMBEDDED_OUTPUT_SHAPE};
use log;
use ndarray::Array4;
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider as OrtExecutionProvider,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::{self, value::Value};

/// ONNX Runtime backend for running `ISNet` models
#[derive(Debug)]
pub struct OnnxBackend {
    session: Option<Session>,
    model_manager: ModelManager,
    initialized: bool,
}

impl OnnxBackend {
    /// Create a new ONNX backend
    #[must_use] pub fn new() -> Self {
        Self {
            session: None,
            model_manager: ModelManager::with_embedded(),
            initialized: false,
        }
    }

    /// Load and initialize the ONNX model
    fn load_model(&mut self, config: &RemovalConfig) -> Result<()> {
        // Load the model data (uses embedded model based on compile-time feature)
        let model_data = self.model_manager.load_model()?;

        // Create ONNX Runtime session with specified execution provider
        let mut session_builder =
            Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

        // Configure execution providers with availability checking
        session_builder = match config.execution_provider {
            ExecutionProvider::Auto => {
                // Auto-detect: try CUDA > CoreML > CPU with availability checking
                let mut providers = Vec::new();

                // Check CUDA availability
                let cuda_provider = CUDAExecutionProvider::default();
                if OrtExecutionProvider::is_available(&cuda_provider).unwrap_or(false) {
                    log::info!("CUDA execution provider is available");
                    providers.push(cuda_provider.build());
                } else {
                    log::debug!("CUDA execution provider is not available");
                }

                // Check CoreML availability
                let coreml_provider = CoreMLExecutionProvider::default();
                if OrtExecutionProvider::is_available(&coreml_provider).unwrap_or(false) {
                    log::info!("CoreML execution provider is available");
                    providers.push(coreml_provider.build());
                } else {
                    log::debug!("CoreML execution provider is not available");
                }

                if providers.is_empty() {
                    log::info!("No hardware acceleration available, using CPU");
                    session_builder
                } else {
                    session_builder.with_execution_providers(providers)?
                }
            },
            ExecutionProvider::Cpu => {
                // CPU only
                log::info!("Using CPU execution provider");
                session_builder
            },
            ExecutionProvider::Cuda => {
                // CUDA only with availability check
                let cuda_provider = CUDAExecutionProvider::default();
                if OrtExecutionProvider::is_available(&cuda_provider).unwrap_or(false) {
                    log::info!("Using CUDA execution provider");
                    session_builder.with_execution_providers([cuda_provider.build()])?
                } else {
                    log::warn!(
                        "CUDA execution provider requested but not available, falling back to CPU"
                    );
                    session_builder
                }
            },
            ExecutionProvider::CoreMl => {
                // CoreML only with availability check
                let coreml_provider = CoreMLExecutionProvider::default();
                if OrtExecutionProvider::is_available(&coreml_provider).unwrap_or(false) {
                    log::info!("Using CoreML execution provider");
                    session_builder.with_execution_providers([coreml_provider.build()])?
                } else {
                    log::warn!("CoreML execution provider requested but not available, falling back to CPU");
                    session_builder
                }
            },
        };

        // Calculate optimal threading if auto-detect (0)
        let intra_threads = if config.intra_threads > 0 {
            config.intra_threads
        } else {
            // Optimal intra-op threads: Use all physical cores for compute-intensive operations
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(8)
        };

        let inter_threads = if config.inter_threads > 0 {
            config.inter_threads
        } else {
            // Optimal inter-op threads: Use fewer threads for coordination (typically 1-4)
            (std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(8)
                / 4)
            .max(1)
        };

        let session = session_builder
            .with_parallel_execution(true)?           // Enable parallel execution
            .with_intra_threads(intra_threads)?       // Threads within operations
            .with_inter_threads(inter_threads)?       // Threads between operations
            .commit_from_memory(&model_data)?;

        // Log comprehensive configuration details
        let model_info = self.model_manager.get_info()?;
        log::info!("ONNX Runtime session created successfully");
        log::info!("Execution provider: {:?}", config.execution_provider);
        log::info!(
            "Threading: {intra_threads} intra-op threads, {inter_threads} inter-op threads, parallel execution enabled"
        );
        log::info!(
            "Optimization level: Level3, Model: {} ({:?})",
            model_info.name,
            model_info.precision
        );

        self.session = Some(session);
        self.initialized = true;

        Ok(())
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for OnnxBackend {
    fn initialize(&mut self, config: &RemovalConfig) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        self.load_model(config)?;
        Ok(())
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        if !self.initialized {
            return Err(crate::error::BgRemovalError::internal(
                "Backend not initialized",
            ));
        }

        let session = self.session.as_mut().ok_or_else(|| {
            crate::error::BgRemovalError::internal("ONNX session not initialized")
        })?;

        // Convert ndarray to ort Value
        let input_value = Value::from_array(input.clone()).map_err(|e| {
            crate::error::BgRemovalError::processing(format!(
                "Failed to convert input tensor: {e}"
            ))
        })?;

        // Run inference using generated input tensor name
        let outputs = session.run(ort::inputs![EMBEDDED_INPUT_NAME => input_value])
            .map_err(|e| crate::error::BgRemovalError::processing(format!("ONNX inference failed. This might be due to incorrect input name '{}'. Original error: {e}", EMBEDDED_INPUT_NAME)))?;

        // Extract output tensor using generated output tensor name
        let output_tensor = if let Ok(output) = outputs[EMBEDDED_OUTPUT_NAME].try_extract_array::<f32>() {
            output
        } else if let Ok(output) = outputs[0].try_extract_array::<f32>() {
            // Try first output if named access fails
            output
        } else {
            return Err(crate::error::BgRemovalError::processing(
                "Failed to extract output tensor from ONNX model",
            ));
        };

        // Convert output to Array4<f32> - reshape if needed
        let output_shape = output_tensor.shape();
        let output_data = output_tensor.view().to_owned();

        if output_shape.len() == 4 {
            let output_array = Array4::from_shape_vec(
                (
                    output_shape[0],
                    output_shape[1],
                    output_shape[2],
                    output_shape[3],
                ),
                output_data.into_raw_vec_and_offset().0,
            )
            .map_err(|e| {
                crate::error::BgRemovalError::processing(format!(
                    "Failed to reshape output tensor: {e}"
                ))
            })?;

            Ok(output_array)
        } else {
            Err(crate::error::BgRemovalError::processing(format!(
                "Expected 4D output tensor, got {}D",
                output_shape.len()
            )))
        }
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        // Use generated input shape from model.json
        (EMBEDDED_INPUT_SHAPE[0], EMBEDDED_INPUT_SHAPE[1], EMBEDDED_INPUT_SHAPE[2], EMBEDDED_INPUT_SHAPE[3])
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        // Use generated output shape from model.json
        (EMBEDDED_OUTPUT_SHAPE[0], EMBEDDED_OUTPUT_SHAPE[1], EMBEDDED_OUTPUT_SHAPE[2], EMBEDDED_OUTPUT_SHAPE[3])
    }
}
