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
                    log::info!("🚀 CUDA execution provider is available and will be used");
                    log::debug!("CUDA provider configuration: {:?}", cuda_provider);
                    providers.push(cuda_provider.build());
                } else {
                    log::debug!("CUDA execution provider is not available");
                }

                // Check CoreML availability with detailed diagnostics
                let coreml_provider = CoreMLExecutionProvider::default();
                let coreml_available = OrtExecutionProvider::is_available(&coreml_provider).unwrap_or(false);
                
                if coreml_available {
                    log::info!("🍎 CoreML execution provider is available and will be used");
                    log::info!("CoreML provider details:");
                    log::info!("  - This will use Apple Neural Engine and GPU acceleration");
                    log::info!("  - Expected significant performance improvement on Apple Silicon");
                    log::info!("  - Provider configuration: {:?}", coreml_provider);
                    
                    // Add CoreML-specific configuration for better performance
                    let coreml_provider = CoreMLExecutionProvider::default()
                        .with_subgraphs(true); // Enable subgraphs for better performance
                    
                    log::debug!("Enhanced CoreML provider config: {:?}", coreml_provider);
                    providers.push(coreml_provider.build());
                } else {
                    log::warn!("🚫 CoreML execution provider is not available");
                    log::warn!("  - This means no GPU acceleration on Apple Silicon");
                    log::warn!("  - Performance will be significantly slower");
                    log::warn!("  - Check if running on Apple Silicon Mac");
                }

                if providers.is_empty() {
                    log::warn!("⚠️ No hardware acceleration available, falling back to CPU");
                    log::warn!("  - This will result in significantly slower performance");
                    session_builder
                } else {
                    log::info!("✅ Hardware acceleration enabled with {} provider(s)", providers.len());
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
                // CoreML only with availability check and detailed diagnostics
                let coreml_provider = CoreMLExecutionProvider::default();
                let coreml_available = OrtExecutionProvider::is_available(&coreml_provider).unwrap_or(false);
                
                if coreml_available {
                    log::info!("🍎 Using CoreML execution provider (explicitly requested)");
                    log::info!("CoreML provider details:");
                    log::info!("  - Will use Apple Neural Engine and GPU acceleration");
                    log::info!("  - This should provide significant speedup on Apple Silicon");
                    log::info!("  - Base provider config: {:?}", coreml_provider);
                    
                    // Enhanced CoreML configuration for better performance
                    let enhanced_coreml_provider = CoreMLExecutionProvider::default()
                        .with_subgraphs(true); // Enable subgraphs for better performance
                    
                    log::info!("Enhanced CoreML provider config: {:?}", enhanced_coreml_provider);
                    session_builder.with_execution_providers([enhanced_coreml_provider.build()])?
                } else {
                    log::error!("🚫 CoreML execution provider requested but not available!");
                    log::error!("  - This is unexpected on Apple Silicon Mac");
                    log::error!("  - Possible causes:");
                    log::error!("    * Not running on Apple Silicon Mac");
                    log::error!("    * ONNX Runtime not built with CoreML support");
                    log::error!("    * macOS version too old");
                    log::error!("  - Falling back to CPU (will be much slower)");
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

        // Log comprehensive configuration details and verify active providers
        let model_info = self.model_manager.as_ref().unwrap().get_info()?;
        log::info!("✅ ONNX Runtime session created successfully");
        log::info!("Session configuration:");
        log::info!("  - Requested provider: {:?}", config.execution_provider);
        log::info!("  - Threading: {intra_threads} intra-op threads, {inter_threads} inter-op threads");
        log::info!("  - Parallel execution: enabled");
        log::info!("  - Optimization level: Level3");
        log::info!("  - Model: {} ({})", model_info.name, model_info.precision);
        log::info!("  - Model size: {:.2} MB", model_info.size_bytes as f64 / (1024.0 * 1024.0));
        
        // Try to get active execution providers (this is diagnostic info)
        // Note: ONNX Runtime doesn't expose this directly, but we can infer from our configuration
        match config.execution_provider {
            ExecutionProvider::Auto => {
                log::info!("🔍 Active execution providers (in priority order):");
                if OrtExecutionProvider::is_available(&CUDAExecutionProvider::default()).unwrap_or(false) {
                    log::info!("  1. CUDA (GPU acceleration)");
                }
                if OrtExecutionProvider::is_available(&CoreMLExecutionProvider::default()).unwrap_or(false) {
                    log::info!("  2. CoreML (Apple Silicon acceleration)");
                    log::info!("     📊 Expected performance: 3-10x faster than CPU");
                    log::info!("     🎯 This should provide significant speedup");
                }
                log::info!("  3. CPU (fallback)");
            },
            ExecutionProvider::CoreMl => {
                if OrtExecutionProvider::is_available(&CoreMLExecutionProvider::default()).unwrap_or(false) {
                    log::info!("🎯 Active execution provider: CoreML");
                    log::info!("  📊 Expected performance: 3-10x faster than CPU");
                    log::info!("  🚀 Using Apple Neural Engine and GPU acceleration");
                } else {
                    log::warn!("⚠️ Active execution provider: CPU (CoreML not available)");
                }
            },
            ExecutionProvider::Cuda => {
                if OrtExecutionProvider::is_available(&CUDAExecutionProvider::default()).unwrap_or(false) {
                    log::info!("🎯 Active execution provider: CUDA");
                } else {
                    log::warn!("⚠️ Active execution provider: CPU (CUDA not available)");
                }
            },
            ExecutionProvider::Cpu => {
                log::info!("🎯 Active execution provider: CPU (explicitly requested)");
            },
        }

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
        use std::time::Instant;
        
        if !self.initialized {
            return Err(crate::error::BgRemovalError::internal(
                "Backend not initialized",
            ));
        }

        let session = self.session.as_mut().ok_or_else(|| {
            crate::error::BgRemovalError::internal("ONNX session not initialized")
        })?;

        // Start detailed timing for inference diagnostics
        let inference_start = Instant::now();
        log::debug!("🚀 Starting inference with input shape: {:?}", input.dim());

        // Convert ndarray to ort Value
        let tensor_conversion_start = Instant::now();
        let input_value = Value::from_array(input.clone()).map_err(|e| {
            crate::error::BgRemovalError::processing(format!(
                "Failed to convert input tensor: {e}"
            ))
        })?;
        let tensor_conversion_time = tensor_conversion_start.elapsed();
        log::debug!("  ⏱️ Tensor conversion: {:.2}ms", tensor_conversion_time.as_secs_f64() * 1000.0);

        // Run inference using model-specific input tensor name
        let model_manager = self.model_manager.as_ref().ok_or_else(|| {
            crate::error::BgRemovalError::internal("Model manager not initialized")
        })?;
        let input_name = model_manager.get_input_name()?;
        let output_name = model_manager.get_output_name()?;
        
        log::debug!("  📋 Using tensor names: input='{}', output='{}'", input_name, output_name);
        
        // Convert to SessionInputs format expected by ORT
        let inputs = vec![(input_name.as_str(), input_value)];
        
        // This is the critical CoreML inference step - measure it precisely
        let core_inference_start = Instant::now();
        log::debug!("  🧠 Starting core ONNX inference...");
        
        let outputs = session.run(inputs)
            .map_err(|e| crate::error::BgRemovalError::processing(format!("ONNX inference failed. This might be due to incorrect input name '{}'. Original error: {e}", input_name)))?;
        
        let core_inference_time = core_inference_start.elapsed();
        log::info!("  ⚡ Core inference: {:.2}ms", core_inference_time.as_secs_f64() * 1000.0);
        
        // Performance analysis
        if core_inference_time.as_millis() > 1000 {
            log::warn!("  ⚠️ Inference took longer than 1 second ({:.2}ms)", core_inference_time.as_secs_f64() * 1000.0);
            log::warn!("    - This suggests GPU acceleration may not be working");
            log::warn!("    - Expected CoreML performance: 100-300ms for typical models");
            log::warn!("    - Current performance suggests CPU execution");
        } else if core_inference_time.as_millis() < 500 {
            log::info!("  🎯 Good inference performance ({:.2}ms) - likely using hardware acceleration", core_inference_time.as_secs_f64() * 1000.0);
        }

        // Extract output tensor using model-specific output tensor name
        let output_extraction_start = Instant::now();
        let output_tensor = if let Ok(output) = outputs[output_name.as_str()].try_extract_array::<f32>() {
            output
        } else if let Ok(output) = outputs[0].try_extract_array::<f32>() {
            // Try first output if named access fails
            log::debug!("  📋 Used fallback output tensor access (index 0)");
            output
        } else {
            return Err(crate::error::BgRemovalError::processing(
                "Failed to extract output tensor from ONNX model",
            ));
        };
        let output_extraction_time = output_extraction_start.elapsed();
        log::debug!("  ⏱️ Output extraction: {:.2}ms", output_extraction_time.as_secs_f64() * 1000.0);

        // Convert output to Array4<f32> - reshape if needed
        let reshape_start = Instant::now();
        let output_shape = output_tensor.shape();
        let output_data = output_tensor.view().to_owned();

        let result = if output_shape.len() == 4 {
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
        };
        
        let reshape_time = reshape_start.elapsed();
        log::debug!("  ⏱️ Output reshape: {:.2}ms", reshape_time.as_secs_f64() * 1000.0);
        
        // Total inference timing summary
        let total_inference_time = inference_start.elapsed();
        log::info!("📊 Inference complete: {:.2}ms total", total_inference_time.as_secs_f64() * 1000.0);
        log::debug!("  └─ Breakdown:");
        log::debug!("     ├─ Tensor conversion: {:.2}ms ({:.1}%)", 
            tensor_conversion_time.as_secs_f64() * 1000.0,
            (tensor_conversion_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0);
        log::debug!("     ├─ Core inference: {:.2}ms ({:.1}%)", 
            core_inference_time.as_secs_f64() * 1000.0,
            (core_inference_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0);
        log::debug!("     ├─ Output extraction: {:.2}ms ({:.1}%)", 
            output_extraction_time.as_secs_f64() * 1000.0,
            (output_extraction_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0);
        log::debug!("     └─ Output reshape: {:.2}ms ({:.1}%)", 
            reshape_time.as_secs_f64() * 1000.0,
            (reshape_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0);

        result
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
