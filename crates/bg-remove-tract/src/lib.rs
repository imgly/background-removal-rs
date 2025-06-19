//! Tract backend implementation for background removal models
//!
//! This crate provides Tract-based inference backend for the bg-remove-core library.
//! It implements the `InferenceBackend` trait from bg-remove-core to provide model inference
//! using Tract, a pure Rust neural network inference library with no external dependencies.
//!
//! Tract offers several advantages:
//! - Pure Rust implementation (no C++ dependencies)
//! - WebAssembly support
//! - Lightweight and portable
//! - Memory safe without FFI boundaries
//! - Faster compilation times

use bg_remove_core::config::RemovalConfig;
use bg_remove_core::error::Result;
use bg_remove_core::inference::InferenceBackend;
use bg_remove_core::models::ModelManager;
use log;
use ndarray::Array4;
use tract_onnx::prelude::*;

// Use instant crate for cross-platform time compatibility
use instant::{Duration, Instant};

/// Tract backend for running background removal models using pure Rust inference
#[derive(Debug)]
pub struct TractBackend {
    model: Option<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>,
    model_manager: Option<ModelManager>,
    initialized: bool,
}

impl TractBackend {
    /// List all Tract execution providers with availability status and descriptions
    ///
    /// Returns a vector of tuples containing:
    /// - Provider name (String)
    /// - Availability status (bool)
    /// - Description (String)
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_tract::TractBackend;
    ///
    /// let providers = TractBackend::list_providers();
    /// for (name, available, description) in providers {
    ///     println!("{}: {} - {}", name, if available { "✅" } else { "❌" }, description);
    /// }
    /// ```
    pub fn list_providers() -> Vec<(String, bool, String)> {
        let mut providers = Vec::new();

        // System information for diagnostics
        log::debug!("🔍 Tract Backend System Analysis:");
        log::debug!("  - Platform: {os}", os = std::env::consts::OS);
        log::debug!("  - Architecture: {arch}", arch = std::env::consts::ARCH);
        log::debug!("  - Pure Rust: No external dependencies required");

        // Get CPU information
        let cpu_count = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        log::debug!("  - CPU cores: {cpu_count}");

        // CPU is the only execution provider for Tract (pure Rust implementation)
        providers.push((
            "CPU".to_string(),
            true,
            "Pure Rust CPU inference with no external dependencies".to_string(),
        ));

        providers
    }

    /// Create a new uninitialized Tract backend
    pub fn new() -> Self {
        Self {
            model: None,
            model_manager: None,
            initialized: false,
        }
    }

    /// Create a Tract backend with a pre-configured model manager
    pub fn with_model_manager(model_manager: ModelManager) -> Self {
        Self {
            model: None,
            model_manager: Some(model_manager),
            initialized: false,
        }
    }

    /// Set the model manager for this backend
    pub fn set_model_manager(&mut self, model_manager: ModelManager) {
        self.model_manager = Some(model_manager);
    }

    /// Load and initialize the model using Tract
    fn load_model(&mut self, _config: &RemovalConfig) -> Result<Duration> {
        let model_load_start = Instant::now();

        // Get or create model manager
        let model_manager = if let Some(ref manager) = self.model_manager {
            manager
        } else {
            return Err(bg_remove_core::error::BgRemovalError::model(
                "No model manager available for Tract backend",
            ));
        };

        // Load model data
        let model_data = model_manager.load_model()?;
        let model_info = model_manager.get_info()?;

        log::info!("🚀 Initializing Tract Backend");
        log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        log::info!("🧠 Model: {} ({})", model_info.name, model_info.precision);
        log::info!("📦 Backend: Tract (Pure Rust)");
        log::info!("⚡ Execution Provider: CPU (Pure Rust)");

        // Model size in MB (precision loss acceptable for display)
        #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for logging display
        let size_mb = model_info.size_bytes as f64 / (1024.0 * 1024.0);
        log::info!("📏 Model size: {size_mb:.2} MB");

        // Create Tract model from ONNX data
        log::debug!("Creating Tract model from ONNX data...");

        // Use different optimization strategies for WASM vs native due to different optimization behavior
        #[cfg(target_arch = "wasm32")]
        let model = {
            log::debug!("WASM build: using typed model without aggressive optimization to avoid dimension conflicts");
            onnx()
                .model_for_read(&mut std::io::Cursor::new(model_data))
                .map_err(|e| {
                    bg_remove_core::error::BgRemovalError::model(format!(
                        "Failed to load ONNX model: {e}"
                    ))
                })?
                .into_typed()
                .map_err(|e| {
                    bg_remove_core::error::BgRemovalError::model(format!(
                        "Failed to create typed model: {e}"
                    ))
                })?
                .into_runnable()
                .map_err(|e| {
                    bg_remove_core::error::BgRemovalError::model(format!(
                        "Failed to create runnable model: {e}"
                    ))
                })?
        };

        #[cfg(not(target_arch = "wasm32"))]
        let model = {
            log::debug!("Native build: using full optimization");
            onnx()
                .model_for_read(&mut std::io::Cursor::new(model_data))
                .map_err(|e| {
                    bg_remove_core::error::BgRemovalError::model(format!(
                        "Failed to load ONNX model: {e}"
                    ))
                })?
                .into_optimized()
                .map_err(|e| {
                    bg_remove_core::error::BgRemovalError::model(format!(
                        "Failed to optimize model: {e}"
                    ))
                })?
                .into_runnable()
                .map_err(|e| {
                    bg_remove_core::error::BgRemovalError::model(format!(
                        "Failed to create runnable model: {e}"
                    ))
                })?
        };

        self.model = Some(model);
        self.initialized = true;

        let model_load_time = model_load_start.elapsed();
        log::info!(
            "✅ Tract backend initialized in {:.2}ms",
            model_load_time.as_millis()
        );
        log::info!("🎯 Ready for inference");

        Ok(model_load_time)
    }

    /// Get the input name for the model
    fn get_input_name(&self) -> Result<String> {
        if let Some(ref manager) = self.model_manager {
            manager.get_input_name()
        } else {
            Err(bg_remove_core::error::BgRemovalError::model(
                "Model manager not available",
            ))
        }
    }

    /// Get the output name for the model
    fn get_output_name(&self) -> Result<String> {
        if let Some(ref manager) = self.model_manager {
            manager.get_output_name()
        } else {
            Err(bg_remove_core::error::BgRemovalError::model(
                "Model manager not available",
            ))
        }
    }
}

impl Default for TractBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for TractBackend {
    fn initialize(&mut self, config: &RemovalConfig) -> Result<Option<Duration>> {
        if self.initialized {
            return Ok(None); // No model loading time for already initialized backend
        }

        let model_load_time = self.load_model(config)?;
        Ok(Some(model_load_time))
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        let _input_name = self.get_input_name()?;
        let _output_name = self.get_output_name()?;

        let model = self.model.as_ref().ok_or_else(|| {
            bg_remove_core::error::BgRemovalError::inference("Tract model not initialized")
        })?;

        log::debug!("🔮 Running Tract inference");
        log::debug!("  - Input tensor: {:?}", input.shape());
        log::debug!("  - Backend: Pure Rust (Tract)");

        let inference_start = Instant::now();

        // Convert ndarray to Tract tensor
        let input_tensor = Tensor::from(input.clone());

        // Run inference
        let outputs = model.run(tvec![input_tensor.into()]).map_err(|e| {
            bg_remove_core::error::BgRemovalError::inference(format!("Tract inference failed: {e}"))
        })?;

        // Extract output tensor
        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or_else(|| {
                bg_remove_core::error::BgRemovalError::inference("No output tensor found")
            })?
            .into_arc_tensor();

        // Convert back to ndarray
        let output_data = output_tensor.to_array_view::<f32>().map_err(|e| {
            bg_remove_core::error::BgRemovalError::inference(format!(
                "Failed to convert output tensor: {e}"
            ))
        })?;

        let output_shape = output_data.shape();
        if output_shape.len() != 4 {
            return Err(bg_remove_core::error::BgRemovalError::inference(format!(
                "Expected 4D output tensor, got {}D",
                output_shape.len()
            )));
        }

        let output_array = Array4::from_shape_vec(
            (
                output_shape[0],
                output_shape[1],
                output_shape[2],
                output_shape[3],
            ),
            output_data.to_owned().into_raw_vec_and_offset().0,
        )
        .map_err(|e| {
            bg_remove_core::error::BgRemovalError::inference(format!(
                "Failed to reshape output tensor: {e}"
            ))
        })?;

        let inference_time = inference_start.elapsed();
        log::debug!(
            "✅ Tract inference completed in {:.2}ms",
            inference_time.as_millis()
        );
        log::debug!("  - Output tensor: {:?}", output_array.shape());

        Ok(output_array)
    }

    fn is_initialized(&self) -> bool {
        self.initialized
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

    fn get_preprocessing_config(&self) -> Result<bg_remove_core::models::PreprocessingConfig> {
        let model_manager = self.model_manager.as_ref().ok_or_else(|| {
            bg_remove_core::error::BgRemovalError::internal("Model manager not initialized")
        })?;
        model_manager.get_preprocessing_config()
    }

    fn get_model_info(&self) -> Result<bg_remove_core::models::ModelInfo> {
        let model_manager = self.model_manager.as_ref().ok_or_else(|| {
            bg_remove_core::error::BgRemovalError::internal("Model manager not initialized")
        })?;
        model_manager.get_info()
    }
}
