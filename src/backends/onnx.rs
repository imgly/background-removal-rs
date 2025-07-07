//! ONNX Runtime backend implementation for background removal models
//!
//! This module provides ONNX Runtime-based inference backend for the bg-remove-core library.
//! It implements the `InferenceBackend` trait from bg-remove-core to provide model inference
//! using ONNX Runtime with support for multiple execution providers (CPU, CUDA, `CoreML`).

use crate::config::{ExecutionProvider, RemovalConfig};
use crate::error::Result;
use crate::inference::InferenceBackend;
use crate::models::ModelManager;
use crate::session_cache::SessionCache;
use log;
use ndarray::Array4;
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider as OrtExecutionProvider,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::{self, value::Value};
use tracing::{debug, instrument, span, Level};

/// ONNX Runtime backend for running background removal models
#[derive(Debug)]
pub struct OnnxBackend {
    session: Option<Session>,
    model_manager: Option<ModelManager>,
    initialized: bool,
    session_cache: Option<SessionCache>,
    /// Time taken to load and initialize the model
    model_load_time: Option<std::time::Duration>,
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
            session_cache: SessionCache::new().ok(),
            model_load_time: None,
        }
    }

    /// Create a new ONNX backend (legacy - uses first available embedded model)
    #[must_use]
    pub fn new() -> Self {
        Self {
            session: None,
            model_manager: None,
            initialized: false,
            session_cache: SessionCache::new().ok(),
            model_load_time: None,
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
        // Get model manager (must be provided)
        let model_manager = self.model_manager.as_ref().ok_or_else(|| {
            crate::error::BgRemovalError::invalid_config(
                "No model manager provided. Use ModelManager::from_spec() to create one with downloaded or external models."
            )
        })?;

        // Load the model data
        let model_data = model_manager.load_model()?;

        // Generate cache key for session caching (skip if caching is disabled)
        let (cache_key, model_hash, provider_config) = if self.session_cache.is_some() {
            if config.disable_cache {
                log::debug!("Session caching disabled by --no-cache flag");
                (None, None, None)
            } else {
                let model_hash = SessionCache::calculate_model_hash(&model_data);
                let provider_config = Self::serialize_provider_config(config);
                let cache_key = SessionCache::generate_cache_key(
                    &model_hash,
                    config.execution_provider,
                    &GraphOptimizationLevel::Level3,
                    &provider_config,
                );
                (Some(cache_key), Some(model_hash), Some(provider_config))
            }
        } else {
            (None, None, None)
        };

        // Try to load cached session first
        if let (Some(cache), Some(key)) = (&mut self.session_cache, &cache_key) {
            if let Ok(Some(cached_session)) =
                cache.load_cached_session(key, config.execution_provider)
            {
                log::info!("🎯 Loaded cached ONNX session: {}", key);
                self.session = Some(cached_session);
                self.initialized = true;

                let model_load_time = model_load_start.elapsed();
                self.model_load_time = Some(model_load_time);
                log::info!(
                    "📊 Model loading complete (cached): {:.0}ms",
                    model_load_time.as_secs_f64() * 1000.0
                );
                return Ok(model_load_time);
            }
        }

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

                    // Add CoreML-specific configuration for better performance and caching
                    let mut coreml_provider =
                        CoreMLExecutionProvider::default().with_subgraphs(true); // Enable subgraphs for better performance

                    // Configure CoreML model cache directory for persistent optimized models
                    if let Some(session_cache) = &self.session_cache {
                        let coreml_cache_dir = session_cache.get_coreml_cache_dir();
                        if let Err(e) = std::fs::create_dir_all(&coreml_cache_dir) {
                            log::warn!("Failed to create CoreML cache directory: {}", e);
                        } else {
                            coreml_provider = coreml_provider
                                .with_model_cache_dir(coreml_cache_dir.to_string_lossy());
                            log::debug!(
                                "CoreML model cache directory: {}",
                                coreml_cache_dir.display()
                            );
                        }
                    }

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

                    // Enhanced CoreML configuration for better performance and caching
                    let mut enhanced_coreml_provider =
                        CoreMLExecutionProvider::default().with_subgraphs(true); // Enable subgraphs for better performance

                    // Configure CoreML model cache directory for persistent optimized models
                    if let Some(session_cache) = &self.session_cache {
                        let coreml_cache_dir = session_cache.get_coreml_cache_dir();
                        if let Err(e) = std::fs::create_dir_all(&coreml_cache_dir) {
                            log::warn!("Failed to create CoreML cache directory: {}", e);
                        } else {
                            enhanced_coreml_provider = enhanced_coreml_provider
                                .with_model_cache_dir(coreml_cache_dir.to_string_lossy());
                            log::debug!(
                                "CoreML model cache directory: {}",
                                coreml_cache_dir.display()
                            );
                        }
                    }

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

        // Cache the session for future use (skip if caching is disabled)
        if let (Some(cache), Some(key), Some(hash), Some(config_str)) = (
            &mut self.session_cache,
            &cache_key,
            &model_hash,
            &provider_config,
        ) {
            if config.disable_cache {
                log::debug!("Skipping session cache save due to --no-cache flag");
            } else {
                // Create session configuration for rebuilding
                let session_config = crate::session_cache::SessionConfig {
                    model_path: model_manager.get_model_path()?,
                    execution_provider: config.execution_provider,
                    optimization_level: "Level3".to_string(),
                    inter_op_num_threads: Some(
                        i16::try_from(inter_threads.min(i16::MAX as usize)).unwrap_or(i16::MAX),
                    ),
                    intra_op_num_threads: Some(
                        i16::try_from(intra_threads.min(i16::MAX as usize)).unwrap_or(i16::MAX),
                    ),
                    parallel_execution: true,
                    provider_options: config_str.clone(),
                };

                if let Err(e) = cache.cache_session(
                    &session,
                    key,
                    hash,
                    config.execution_provider,
                    &GraphOptimizationLevel::Level3,
                    config_str,
                    session_config,
                ) {
                    log::warn!("Failed to cache session: {}", e);
                    // Continue execution - caching failure is not critical
                } else {
                    log::debug!("✅ Cached new ONNX session: {}", key);
                }
            }
        }

        self.session = Some(session);
        self.initialized = true;

        let model_load_time = model_load_start.elapsed();
        self.model_load_time = Some(model_load_time);
        log::info!(
            "📊 Model loading complete: {:.0}ms",
            model_load_time.as_secs_f64() * 1000.0
        );

        Ok(model_load_time)
    }

    /// Serialize provider configuration for cache key generation
    fn serialize_provider_config(config: &RemovalConfig) -> String {
        // Include all configuration parameters that affect session compilation
        // Use a simple format string since serde_json might not be available
        format!(
            "{{\"execution_provider\":\"{:?}\",\"intra_threads\":{},\"inter_threads\":{},\"parallel_execution\":true,\"optimization_level\":\"Level3\"}}",
            config.execution_provider,
            config.intra_threads,
            config.inter_threads
        )
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for OnnxBackend {
    #[instrument(skip(self, config), fields(execution_provider = %config.execution_provider))]
    fn initialize(&mut self, config: &RemovalConfig) -> Result<Option<std::time::Duration>> {
        if self.initialized {
            return Ok(None); // No model loading time for already initialized backend
        }

        let model_load_time = {
            let _span = span!(Level::INFO, "model_loading", provider = %config.execution_provider)
                .entered();
            self.load_model(config)?
        };
        Ok(Some(model_load_time))
    }

    #[allow(clippy::too_many_lines)] // Complex inference with detailed diagnostics
    #[allow(clippy::get_first)]
    #[instrument(skip(self, input), fields(input_shape = ?input.dim()))]
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
        debug!(
            input_shape = ?input.dim(),
            "🚀 Starting ONNX inference"
        );

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
                    output_shape.get(0).copied().unwrap_or(1),
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

        // Include model load time if available
        if let Some(load_time) = self.model_load_time {
            log::debug!(
                "     ├─ Model load/init: {:.2}ms (one-time)",
                load_time.as_secs_f64() * 1000.0
            );
        }

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

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;
    use crate::config::{ExecutionProvider, RemovalConfig};
    use crate::models::{ModelManager, ModelSource, ModelSpec};

    #[test]
    fn test_onnx_backend_creation() {
        // Test backend creation without model manager (fallback behavior)
        let backend = OnnxBackend::new();

        assert!(!backend.is_initialized());
        assert_eq!(backend.input_shape(), (1, 3, 1024, 1024)); // Default fallback
        assert_eq!(backend.output_shape(), (1, 1, 1024, 1024)); // Default fallback
    }

    #[test]
    fn test_onnx_backend_list_providers() {
        let providers = OnnxBackend::list_providers();

        // Should contain at least CPU provider
        assert!(!providers.is_empty());

        // Check CPU provider is always available
        let cpu_provider = providers.iter().find(|(name, _, _)| name == "CPU");
        assert!(cpu_provider.is_some());
        let (_, available, description) = cpu_provider.unwrap();
        assert!(*available);
        assert!(!description.is_empty());

        // Verify all provider names are valid
        for (name, _, desc) in &providers {
            assert!(!name.is_empty());
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_onnx_backend_default_shapes() {
        // Test default shapes when backend has no model manager (fallback behavior)
        let backend = OnnxBackend::new();

        // Test default shapes when model is not initialized
        let input_shape = backend.input_shape();
        let output_shape = backend.output_shape();

        assert_eq!(input_shape.0, 1); // Batch size
        assert_eq!(input_shape.1, 3); // RGB channels
        assert!(input_shape.2 > 0); // Height
        assert!(input_shape.3 > 0); // Width

        assert_eq!(output_shape.0, 1); // Batch size
        assert_eq!(output_shape.1, 1); // Single channel mask
        assert!(output_shape.2 > 0); // Height
        assert!(output_shape.3 > 0); // Width
    }

    #[test]
    fn test_onnx_backend_uninitialized_operations() {
        // Test operations on uninitialized backend without model manager
        let backend = OnnxBackend::new();

        // Test operations on uninitialized backend
        assert!(!backend.is_initialized());

        // These should work without initialization (fallback behavior)
        let input_shape = backend.input_shape();
        let output_shape = backend.output_shape();
        assert_eq!(input_shape, (1, 3, 1024, 1024)); // Default fallback
        assert_eq!(output_shape, (1, 1, 1024, 1024)); // Default fallback

        // Test that model info and preprocessing config fail gracefully when no model manager
        let model_info_result = backend.get_model_info();
        let preprocessing_result = backend.get_preprocessing_config();

        // These should fail gracefully when no model manager is present
        assert!(model_info_result.is_err());
        assert!(preprocessing_result.is_err());
    }

    #[test]
    fn test_onnx_backend_provider_descriptions() {
        let providers = OnnxBackend::list_providers();

        for (name, available, description) in providers {
            // Verify provider names are known types
            assert!(
                name == "CPU" || name == "CUDA" || name == "CoreML",
                "Unknown provider: {}",
                name
            );

            // CPU should always be available
            if name == "CPU" {
                assert!(available, "CPU provider should always be available");
            }

            // Descriptions should contain useful information
            assert!(
                description.len() > 10,
                "Provider description too short: {}",
                description
            );

            // Check description contains relevant keywords
            match name.as_str() {
                "CPU" => assert!(
                    description.to_lowercase().contains("cpu"),
                    "CPU description should mention CPU: {}",
                    description
                ),
                "CUDA" => assert!(
                    description.to_lowercase().contains("gpu")
                        || description.to_lowercase().contains("cuda")
                        || description.to_lowercase().contains("nvidia"),
                    "CUDA description should mention GPU/CUDA/NVIDIA: {}",
                    description
                ),
                "CoreML" => assert!(
                    description.to_lowercase().contains("apple")
                        || description.to_lowercase().contains("coreml")
                        || description.to_lowercase().contains("macos"),
                    "CoreML description should mention Apple/CoreML/macOS: {}",
                    description
                ),
                _ => {}, // Other providers
            }
        }
    }

    #[test]
    fn test_onnx_backend_initialization_requirements() {
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("nonexistent-model".to_string()),
            variant: Some("fp32".to_string()),
        };

        // Test with invalid model manager
        let model_manager_result = ModelManager::from_spec(&model_spec);

        if let Ok(model_manager) = model_manager_result {
            let mut backend = OnnxBackend::with_model_manager(model_manager);
            assert!(!backend.is_initialized());

            // Try to initialize with a basic config
            let config = RemovalConfig {
                execution_provider: ExecutionProvider::Cpu,
                output_format: crate::config::OutputFormat::Png,
                jpeg_quality: 90,
                webp_quality: 85,
                debug: false,
                intra_threads: 0,
                inter_threads: 0,
                preserve_color_profiles: false,
                disable_cache: false,
                model_spec: model_spec.clone(),
                format_hint: None,
                #[cfg(feature = "video-support")]
                video_config: None,
            };

            // This will likely fail due to missing model, but should not panic
            let init_result = backend.initialize(&config);

            // Just verify the backend handles the failure gracefully
            if init_result.is_err() {
                assert!(!backend.is_initialized());
            }
        }
    }

    #[test]
    fn test_onnx_backend_thread_configuration() {
        // Test thread configuration without requiring cached model
        let mut backend = OnnxBackend::new();

        // Create a model spec for configuration validation
        let model_spec = ModelSpec {
            source: ModelSource::External("test-model".into()),
            variant: Some("fp32".to_string()),
        };

        // Test different thread configurations
        let configs = vec![
            (0, 0), // Auto threads
            (1, 1), // Single threaded
            (2, 4), // Mixed configuration
            (4, 2), // Reversed configuration
        ];

        for (intra, inter) in configs {
            let config = RemovalConfig {
                execution_provider: ExecutionProvider::Cpu,
                output_format: crate::config::OutputFormat::Png,
                jpeg_quality: 90,
                webp_quality: 85,
                debug: false,
                intra_threads: intra,
                inter_threads: inter,
                preserve_color_profiles: false,
                disable_cache: false,
                model_spec: model_spec.clone(),
                format_hint: None,
                #[cfg(feature = "video-support")]
                video_config: None,
            };

            // Attempt initialization (may fail due to missing model, but shouldn't panic)
            let _ = backend.initialize(&config);
        }
    }

    #[test]
    fn test_onnx_backend_debug_mode() {
        let mut backend = OnnxBackend::new();

        // Test debug mode configuration - create a valid model spec for config validation
        let model_spec = ModelSpec {
            source: ModelSource::External("test-model".into()),
            variant: Some("fp32".to_string()),
        };

        // Test debug mode configuration
        let config = crate::config::RemovalConfig {
            execution_provider: crate::config::ExecutionProvider::Cpu,
            output_format: crate::config::OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: true, // Enable debug mode
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: false,
            disable_cache: false,
            model_spec: model_spec.clone(),
            format_hint: None,
            #[cfg(feature = "video-support")]
            video_config: None,
        };

        // Attempt initialization with debug mode - should fail gracefully without model manager
        let init_result = backend.initialize(&config);

        // Should fail because no model manager, but shouldn't panic
        assert!(init_result.is_err());
        assert!(!backend.is_initialized());
    }
}
