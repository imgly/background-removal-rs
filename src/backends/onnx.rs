//! ONNX Runtime backend implementation for background removal models
//!
//! This module provides ONNX Runtime-based inference backend for the bg-remove-core library.
//! It implements the `InferenceBackend` trait from bg-remove-core to provide model inference
//! using ONNX Runtime with support for multiple execution providers (CPU, CUDA, CoreML).

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

/// ONNX Runtime backend for running background removal models
#[derive(Debug)]
pub struct OnnxBackend {
    session: Option<Session>,
    model_manager: Option<ModelManager>,
    initialized: bool,
}

impl OnnxBackend {
    /// List all ONNX Runtime execution providers with availability status and descriptions
    ///
    /// Returns a vector of tuples containing:
    /// - Provider name (String)
    /// - Availability status (bool)
    /// - Description (String)
    ///
    /// # Examples
    /// ```rust
    /// use imgly_bgremove::backends::OnnxBackend;
    ///
    /// let providers = OnnxBackend::list_providers();
    /// for (name, available, description) in providers {
    ///     println!("{}: {} - {}", name, if available { "✅" } else { "❌" }, description);
    /// }
    /// ```
    pub fn list_providers() -> Vec<(String, bool, String)> {
        let mut providers = Vec::new();

        // System information for diagnostics
        log::debug!("🔍 System Hardware Analysis:");
        log::debug!("  - Platform: {os}", os = std::env::consts::OS);
        log::debug!("  - Architecture: {arch}", arch = std::env::consts::ARCH);
        log::debug!(
            "  - CPU cores: {cores}",
            cores = std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
        );

        // macOS-specific checks for Apple Silicon
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;

            // Check if running on Apple Silicon
            if let Ok(output) = Command::new("sysctl")
                .arg("-n")
                .arg("machdep.cpu.brand_string")
                .output()
            {
                let cpu_brand = String::from_utf8_lossy(&output.stdout);
                log::debug!("  - CPU: {cpu}", cpu = cpu_brand.trim());

                if cpu_brand.contains("Apple") {
                    log::debug!("  - ✅ Apple Silicon detected - CoreML should be available");
                } else {
                    log::debug!("  - ⚠️ Intel Mac detected - CoreML may have limited support");
                }
            }

            // Check macOS version
            if let Ok(output) = Command::new("sw_vers").arg("-productVersion").output() {
                let version = String::from_utf8_lossy(&output.stdout);
                log::debug!("  - macOS version: {version}", version = version.trim());
            }
        }

        // CPU is always available
        providers.push((
            "CPU".to_string(),
            true,
            "Always available, uses CPU for inference".to_string(),
        ));

        // Check CUDA availability with diagnostics
        log::debug!("🔍 Checking CUDA availability...");
        let cuda_available =
            OrtExecutionProvider::is_available(&CUDAExecutionProvider::default()).unwrap_or(false);
        if cuda_available {
            log::info!("✅ CUDA execution provider is available");
        } else {
            log::debug!("❌ CUDA execution provider is not available");
        }
        providers.push((
            "CUDA".to_string(),
            cuda_available,
            "NVIDIA GPU acceleration (requires CUDA toolkit and compatible GPU)".to_string(),
        ));

        // Check CoreML availability with detailed diagnostics
        log::debug!("🔍 Checking CoreML availability...");
        let coreml_available =
            OrtExecutionProvider::is_available(&CoreMLExecutionProvider::default())
                .unwrap_or(false);
        if coreml_available {
            log::info!("✅ CoreML execution provider is available");
            log::info!("  - This enables Apple Neural Engine and GPU acceleration");
            log::info!("  - Expected performance improvement: 3-10x over CPU");
        } else {
            log::warn!("❌ CoreML execution provider is not available");
            #[cfg(target_os = "macos")]
            {
                log::warn!("  - This is unexpected on macOS");
                log::warn!("  - Possible causes:");
                log::warn!("    * ONNX Runtime not built with CoreML support");
                log::warn!("    * macOS version too old (requires macOS 11+)");
                log::warn!("    * Missing CoreML framework");
            }
            #[cfg(not(target_os = "macos"))]
            {
                log::debug!("  - Expected: CoreML is only available on macOS");
            }
        }
        providers.push((
            "CoreML".to_string(),
            coreml_available,
            "Apple Silicon GPU acceleration (macOS only)".to_string(),
        ));

        providers
    }

    /// Create a new ONNX backend with specific model manager
    #[must_use]
    pub fn with_model_manager(model_manager: ModelManager) -> Self {
        Self {
            session: None,
            model_manager: Some(model_manager),
            initialized: false,
        }
    }

    /// Create a new ONNX backend (legacy - uses first available embedded model)
    #[must_use]
    pub fn new() -> Self {
        Self {
            session: None,
            model_manager: None,
            initialized: false,
        }
    }

    /// Set the model manager for this backend
    pub fn set_model_manager(&mut self, model_manager: ModelManager) {
        self.model_manager = Some(model_manager);
    }

    /// Load and initialize the ONNX model
    #[allow(clippy::too_many_lines)] // Complex provider configuration logic
    fn load_model(&mut self, config: &RemovalConfig) -> Result<std::time::Duration> {
        let model_load_start = std::time::Instant::now();
        // Get or create model manager
        let model_manager = if let Some(ref manager) = self.model_manager {
            manager
        } else {
            // Fall back to embedded model if no model manager was set
            let embedded_manager = ModelManager::with_embedded()?;
            self.model_manager = Some(embedded_manager);
            self.model_manager.as_ref().ok_or_else(|| {
                crate::error::BgRemovalError::internal(
                    "Model manager unexpectedly missing after insertion",
                )
            })?
        };

        // Load the model data
        let model_data = model_manager.load_model()?;

        // Create ONNX Runtime session with specified execution provider
        let mut session_builder = Session::builder()
            .map_err(|e| {
                crate::error::BgRemovalError::inference(format!(
                    "Failed to create session builder: {e}"
                ))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                crate::error::BgRemovalError::inference(format!(
                    "Failed to set optimization level: {e}"
                ))
            })?;

        // Configure execution providers with availability checking
        session_builder = match config.execution_provider {
            ExecutionProvider::Auto => {
                // Auto-detect: try CUDA > CoreML > CPU with availability checking
                let mut providers = Vec::new();

                // Check CUDA availability
                let cuda_provider = CUDAExecutionProvider::default();
                if OrtExecutionProvider::is_available(&cuda_provider).unwrap_or(false) {
                    log::info!("🚀 CUDA execution provider is available and will be used");
                    log::debug!("CUDA provider configuration: {cuda_provider:?}");
                    providers.push(cuda_provider.build());
                } else {
                    log::debug!("CUDA execution provider is not available");
                }

                // Check CoreML availability with detailed diagnostics
                let coreml_provider = CoreMLExecutionProvider::default();
                let coreml_available =
                    OrtExecutionProvider::is_available(&coreml_provider).unwrap_or(false);

                if coreml_available {
                    log::info!("🍎 CoreML execution provider is available and will be used");
                    log::debug!("CoreML provider details:");
                    log::debug!("  - This will use Apple Neural Engine and GPU acceleration");
                    log::debug!(
                        "  - Expected significant performance improvement on Apple Silicon"
                    );
                    log::debug!("  - Provider configuration: {coreml_provider:?}");

                    // Add CoreML-specific configuration for better performance
                    let coreml_provider = CoreMLExecutionProvider::default().with_subgraphs(true); // Enable subgraphs for better performance

                    log::debug!("Enhanced CoreML provider config: {coreml_provider:?}");
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
                    log::info!(
                        "✅ Hardware acceleration enabled with {count} provider(s)",
                        count = providers.len()
                    );
                    session_builder
                        .with_execution_providers(providers)
                        .map_err(|e| {
                            crate::error::BgRemovalError::inference(format!(
                                "Failed to set auto execution providers: {e}"
                            ))
                        })?
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
                    session_builder
                        .with_execution_providers([cuda_provider.build()])
                        .map_err(|e| {
                            crate::error::BgRemovalError::inference(format!(
                                "Failed to set CUDA execution provider: {e}"
                            ))
                        })?
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
                let coreml_available =
                    OrtExecutionProvider::is_available(&coreml_provider).unwrap_or(false);

                if coreml_available {
                    log::info!("🍎 Using CoreML execution provider (explicitly requested)");
                    log::debug!("CoreML provider details:");
                    log::debug!("  - Will use Apple Neural Engine and GPU acceleration");
                    log::debug!("  - This should provide significant speedup on Apple Silicon");
                    log::debug!("  - Base provider config: {coreml_provider:?}");

                    // Enhanced CoreML configuration for better performance
                    let enhanced_coreml_provider =
                        CoreMLExecutionProvider::default().with_subgraphs(true); // Enable subgraphs for better performance

                    log::debug!("Enhanced CoreML provider config: {enhanced_coreml_provider:?}");
                    session_builder
                        .with_execution_providers([enhanced_coreml_provider.build()])
                        .map_err(|e| {
                            crate::error::BgRemovalError::inference(format!(
                                "Failed to set CoreML execution provider: {e}"
                            ))
                        })?
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
            .with_parallel_execution(true)
            .map_err(|e| crate::error::BgRemovalError::inference(format!("Failed to enable parallel execution: {e}")))?           // Enable parallel execution
            .with_intra_threads(intra_threads)
            .map_err(|e| crate::error::BgRemovalError::inference(format!("Failed to set intra threads: {e}")))?       // Threads within operations
            .with_inter_threads(inter_threads)
            .map_err(|e| crate::error::BgRemovalError::inference(format!("Failed to set inter threads: {e}")))?       // Threads between operations
            .commit_from_memory(&model_data)
            .map_err(|e| crate::error::BgRemovalError::inference(format!("Failed to create session from model data: {e}")))?;

        // Log comprehensive configuration details and verify active providers
        let model_info = self
            .model_manager
            .as_ref()
            .ok_or_else(|| crate::error::BgRemovalError::internal("Model manager not initialized"))?
            .get_info()?;
        log::debug!("✅ ONNX Runtime session created successfully");
        log::debug!("Session configuration:");
        log::debug!("  - Requested provider: {:?}", config.execution_provider);
        log::debug!(
            "  - Threading: {intra_threads} intra-op threads, {inter_threads} inter-op threads"
        );
        log::debug!("  - Parallel execution: enabled");
        log::debug!("  - Optimization level: Level3");
        log::debug!("  - Model: {} ({})", model_info.name, model_info.precision);
        // Model size in MB (precision loss acceptable for display)
        #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for logging display
        let size_mb = model_info.size_bytes as f64 / (1024.0 * 1024.0);
        log::debug!("  - Model size: {size_mb:.2} MB");

        // Try to get active execution providers (this is diagnostic info)
        // Note: ONNX Runtime doesn't expose this directly, but we can infer from our configuration
        match config.execution_provider {
            ExecutionProvider::Auto => {
                log::debug!("🔍 Active execution providers (in priority order):");
                if OrtExecutionProvider::is_available(&CUDAExecutionProvider::default())
                    .unwrap_or(false)
                {
                    log::debug!("  1. CUDA (GPU acceleration)");
                }
                if OrtExecutionProvider::is_available(&CoreMLExecutionProvider::default())
                    .unwrap_or(false)
                {
                    log::debug!("  2. CoreML (Apple Silicon acceleration)");
                    log::debug!("     📊 Expected performance: 3-10x faster than CPU");
                    log::debug!("     🎯 This should provide significant speedup");
                }
                log::debug!("  3. CPU (fallback)");
            },
            ExecutionProvider::CoreMl => {
                if OrtExecutionProvider::is_available(&CoreMLExecutionProvider::default())
                    .unwrap_or(false)
                {
                    log::info!("🎯 Active execution provider: CoreML");
                    log::debug!("  📊 Expected performance: 3-10x faster than CPU");
                    log::debug!("  🚀 Using Apple Neural Engine and GPU acceleration");
                } else {
                    log::warn!("⚠️ Active execution provider: CPU (CoreML not available)");
                }
            },
            ExecutionProvider::Cuda => {
                if OrtExecutionProvider::is_available(&CUDAExecutionProvider::default())
                    .unwrap_or(false)
                {
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

        let model_load_time = model_load_start.elapsed();
        log::info!(
            "📊 Model loading complete: {:.0}ms",
            model_load_time.as_secs_f64() * 1000.0
        );

        Ok(model_load_time)
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for OnnxBackend {
    fn initialize(&mut self, config: &RemovalConfig) -> Result<Option<std::time::Duration>> {
        if self.initialized {
            return Ok(None); // No model loading time for already initialized backend
        }

        let model_load_time = self.load_model(config)?;
        Ok(Some(model_load_time))
    }

    #[allow(clippy::too_many_lines)] // Complex inference with detailed diagnostics
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
            crate::error::BgRemovalError::processing(format!("Failed to convert input tensor: {e}"))
        })?;
        let tensor_conversion_time = tensor_conversion_start.elapsed();
        log::debug!(
            "  ⏱️ Tensor conversion: {:.2}ms",
            tensor_conversion_time.as_secs_f64() * 1000.0
        );

        // Use positional inputs instead of named inputs for better compatibility
        log::debug!("  📋 Using positional inputs (eliminates tensor name dependencies)");

        // This is the critical CoreML inference step - measure it precisely
        let core_inference_start = Instant::now();
        log::debug!("  🧠 Starting core ONNX inference...");

        let outputs = session.run(ort::inputs![input_value]).map_err(|e| {
            crate::error::BgRemovalError::processing(format!("ONNX inference failed: {e}"))
        })?;

        let core_inference_time = core_inference_start.elapsed();
        log::debug!(
            "  ⚡ Core inference: {:.2}ms",
            core_inference_time.as_secs_f64() * 1000.0
        );

        // Performance logging (debug only)
        if core_inference_time.as_millis() < 500 {
            log::debug!(
                "  🎯 Good inference performance ({:.2}ms) - likely using hardware acceleration",
                core_inference_time.as_secs_f64() * 1000.0
            );
        }

        // Extract output tensor using positional access (first output)
        let output_extraction_start = Instant::now();
        let output_tensor = {
            let keys: Vec<_> = outputs.keys().collect();
            if let Some(first_key) = keys.first() {
                log::debug!(
                    "  📋 Using positional output access (first output: {})",
                    first_key
                );
                outputs
                    .get(first_key)
                    .ok_or_else(|| {
                        crate::error::BgRemovalError::processing("First output tensor not found")
                    })?
                    .try_extract_array::<f32>()
                    .map_err(|e| {
                        crate::error::BgRemovalError::processing(format!(
                            "Failed to extract output tensor: {e}"
                        ))
                    })?
            } else {
                return Err(crate::error::BgRemovalError::processing(
                    "No output tensors found",
                ));
            }
        };
        let output_extraction_time = output_extraction_start.elapsed();
        log::debug!(
            "  ⏱️ Output extraction: {:.2}ms",
            output_extraction_time.as_secs_f64() * 1000.0
        );

        // Convert output to Array4<f32> - reshape if needed
        let reshape_start = Instant::now();
        let output_shape = output_tensor.shape();
        let output_data = output_tensor.view().to_owned();

        let result = if output_shape.len() == 4 {
            let output_array = Array4::from_shape_vec(
                (
                    output_shape.first().copied().unwrap_or(1),
                    output_shape.get(1).copied().unwrap_or(1),
                    output_shape.get(2).copied().unwrap_or(1),
                    output_shape.get(3).copied().unwrap_or(1),
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
        log::debug!(
            "  ⏱️ Output reshape: {:.2}ms",
            reshape_time.as_secs_f64() * 1000.0
        );

        // Total inference timing summary
        let total_inference_time = inference_start.elapsed();
        log::info!(
            "📊 Inference complete: {:.2}ms total",
            total_inference_time.as_secs_f64() * 1000.0
        );
        log::debug!("  └─ Breakdown:");
        log::debug!(
            "     ├─ Tensor conversion: {:.2}ms ({:.1}%)",
            tensor_conversion_time.as_secs_f64() * 1000.0,
            (tensor_conversion_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0
        );
        log::debug!(
            "     ├─ Core inference: {:.2}ms ({:.1}%)",
            core_inference_time.as_secs_f64() * 1000.0,
            (core_inference_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0
        );
        log::debug!(
            "     ├─ Output extraction: {:.2}ms ({:.1}%)",
            output_extraction_time.as_secs_f64() * 1000.0,
            (output_extraction_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0
        );
        log::debug!(
            "     └─ Output reshape: {:.2}ms ({:.1}%)",
            reshape_time.as_secs_f64() * 1000.0,
            (reshape_time.as_secs_f64() / total_inference_time.as_secs_f64()) * 100.0
        );

        result
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        // Use model-specific input shape from model info
        self.model_manager
            .as_ref()
            .and_then(|manager| manager.get_info().ok())
            .map_or((1, 3, 1024, 1024), |info| info.input_shape) // Default fallback
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        // Use model-specific output shape from model info
        self.model_manager
            .as_ref()
            .and_then(|manager| manager.get_info().ok())
            .map_or((1, 1, 1024, 1024), |info| info.output_shape) // Default fallback
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

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}
