//! ONNX Runtime backend implementation for `ISNet` model inference

use crate::config::{ExecutionProvider, RemovalConfig};
use crate::error::Result;
use crate::inference::InferenceBackend;
use crate::models::ModelManager;
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
    model_manager: Option<ModelManager>,
    initialized: bool,
}

impl OnnxBackend {
    /// Create a new ONNX backend with specific model manager
    pub fn with_model_manager(model_manager: ModelManager) -> Self {
        Self {
            session: None,
            model_manager: Some(model_manager),
            initialized: false,
        }
    }
    
    /// Create a new ONNX backend (legacy - uses first available embedded model)
    pub fn new() -> Result<Self> {
        Ok(Self {
            session: None,
            model_manager: None,
            initialized: false,
        })
    }
    
    /// Set the model manager for this backend
    pub fn set_model_manager(&mut self, model_manager: ModelManager) {
        self.model_manager = Some(model_manager);
    }

    /// Load and initialize the ONNX model
    fn load_model(&mut self, config: &RemovalConfig) -> Result<()> {
        // Get or create model manager
        let model_manager = if let Some(ref manager) = self.model_manager {
            manager
        } else {
            // Fall back to embedded model if no model manager was set
            let embedded_manager = ModelManager::with_embedded()?;
            self.model_manager = Some(embedded_manager);
            self.model_manager.as_ref().unwrap()
        };
        
        // Load the model data
        let model_data = model_manager.load_model()?;

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
        let model_info = self.model_manager.as_ref().unwrap().get_info()?;
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
        Self::new().expect("Failed to create default OnnxBackend")
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

        // Run inference using model-specific input tensor name
        let model_manager = self.model_manager.as_ref().ok_or_else(|| {
            crate::error::BgRemovalError::internal("Model manager not initialized")
        })?;
        let input_name = model_manager.get_input_name()?;
        let output_name = model_manager.get_output_name()?;
        
        // Convert to SessionInputs format expected by ORT
        let inputs = vec![(input_name.as_str(), input_value)];
        let outputs = session.run(inputs)
            .map_err(|e| crate::error::BgRemovalError::processing(format!("ONNX inference failed. This might be due to incorrect input name '{}'. Original error: {e}", input_name)))?;

        // Extract output tensor using model-specific output tensor name
        let output_tensor = if let Ok(output) = outputs[output_name.as_str()].try_extract_array::<f32>() {
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
        // Use model-specific input shape from model info
        self.model_manager.as_ref()
            .and_then(|manager| manager.get_info().ok())
            .map(|info| info.input_shape)
            .unwrap_or((1, 3, 1024, 1024)) // Default fallback
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        // Use model-specific output shape from model info
        self.model_manager.as_ref()
            .and_then(|manager| manager.get_info().ok())
            .map(|info| info.output_shape)
            .unwrap_or((1, 1, 1024, 1024)) // Default fallback
    }
    
    fn get_preprocessing_config(&self) -> Result<crate::models::PreprocessingConfig> {
        let model_manager = self.model_manager.as_ref().ok_or_else(|| {
            crate::error::BgRemovalError::internal("Model manager not initialized")
        })?;
        model_manager.get_preprocessing_config()
    }
    
    fn get_model_info(&self) -> Result<crate::models::ModelInfo> {
        let model_manager = self.model_manager.as_ref().ok_or_else(|| {
            crate::error::BgRemovalError::internal("Model manager not initialized")
        })?;
        model_manager.get_info()
    }
}
