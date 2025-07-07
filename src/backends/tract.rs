//! Tract backend implementation for background removal models
//!
//! This module provides Tract-based inference backend for the bg-remove-core library.
//! It implements the `InferenceBackend` trait from bg-remove-core to provide model inference
//! using Tract, a pure Rust neural network inference library with no external dependencies.
//!
//! Tract offers several advantages:
//! - Pure Rust implementation (no C++ dependencies)
//! - Lightweight and portable
//! - Memory safe without FFI boundaries
//! - Faster compilation times

use crate::config::RemovalConfig;
use crate::error::Result;
use crate::inference::InferenceBackend;
use crate::models::ModelManager;
use log;
use ndarray::Array4;
use tract_onnx::prelude::*;

/// Type alias for the complex Tract model type to reduce complexity warnings
type TractModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// Use instant crate for cross-platform time compatibility
use instant::{Duration, Instant};

/// Tract backend for running background removal models using pure Rust inference
#[derive(Debug)]
pub struct TractBackend {
    model: Option<TractModel>,
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
    /// use imgly_bgremove::backends::TractBackend;
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: None,
            model_manager: None,
            initialized: false,
        }
    }

    /// Create a Tract backend with a pre-configured model manager
    #[must_use]
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
        let Some(ref model_manager) = self.model_manager else {
            return Err(crate::error::BgRemovalError::model(
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

        let model = onnx()
            .model_for_read(&mut std::io::Cursor::new(model_data))
            .map_err(|e| {
                crate::error::BgRemovalError::model(format!("Failed to load ONNX model: {e}"))
            })?
            .into_optimized()
            .map_err(|e| {
                crate::error::BgRemovalError::model(format!("Failed to optimize model: {e}"))
            })?
            .into_runnable()
            .map_err(|e| {
                crate::error::BgRemovalError::model(format!("Failed to create runnable model: {e}"))
            })?;

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
    #[allow(dead_code)]
    fn get_input_name(&self) -> Result<String> {
        if let Some(ref manager) = self.model_manager {
            manager.get_input_name()
        } else {
            Err(crate::error::BgRemovalError::model(
                "Model manager not available",
            ))
        }
    }

    /// Get the output name for the model
    #[allow(dead_code)]
    fn get_output_name(&self) -> Result<String> {
        if let Some(ref manager) = self.model_manager {
            manager.get_output_name()
        } else {
            Err(crate::error::BgRemovalError::model(
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

    #[allow(clippy::get_first)]
    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        let model = self.model.as_ref().ok_or_else(|| {
            crate::error::BgRemovalError::inference("Tract model not initialized")
        })?;

        log::debug!("🔮 Running Tract inference");
        log::debug!("  - Input tensor: {:?}", input.shape());
        log::debug!("  - Backend: Pure Rust (Tract)");

        let inference_start = Instant::now();

        // Convert ndarray to Tract tensor
        let input_tensor = Tensor::from(input.clone());

        // Run inference
        let outputs = model.run(tvec![input_tensor.into()]).map_err(|e| {
            crate::error::BgRemovalError::inference(format!("Tract inference failed: {e}"))
        })?;

        // Extract output tensor
        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or_else(|| crate::error::BgRemovalError::inference("No output tensor found"))?
            .into_arc_tensor();

        // Convert back to ndarray
        let output_data = output_tensor.to_array_view::<f32>().map_err(|e| {
            crate::error::BgRemovalError::inference(format!("Failed to convert output tensor: {e}"))
        })?;

        let output_shape = output_data.shape();
        if output_shape.len() != 4 {
            return Err(crate::error::BgRemovalError::inference(format!(
                "Expected 4D output tensor, got {}D",
                output_shape.len()
            )));
        }

        let output_array = Array4::from_shape_vec(
            (
                output_shape.get(0).copied().unwrap_or(1),
                output_shape.get(1).copied().unwrap_or(1),
                output_shape.get(2).copied().unwrap_or(1024),
                output_shape.get(3).copied().unwrap_or(1024),
            ),
            output_data.to_owned().into_raw_vec_and_offset().0,
        )
        .map_err(|e| {
            crate::error::BgRemovalError::inference(format!("Failed to reshape output tensor: {e}"))
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

#[cfg(all(test, feature = "tract"))]
mod tests {
    use super::*;
    use crate::config::{ExecutionProvider, RemovalConfig};
    use crate::models::{ModelManager, ModelSource, ModelSpec};

    #[test]
    fn test_tract_backend_creation() {
        // Test backend creation without model manager (fallback behavior)
        let backend = TractBackend::new();

        assert!(!backend.is_initialized());
        assert_eq!(backend.input_shape(), (1, 3, 1024, 1024)); // Default fallback
        assert_eq!(backend.output_shape(), (1, 1, 1024, 1024)); // Default fallback
    }

    #[test]
    fn test_tract_backend_default_shapes() {
        // Test default shapes when backend has no model manager (fallback behavior)
        let backend = TractBackend::new();

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
    fn test_tract_backend_uninitialized_operations() {
        // Test operations on uninitialized backend without model manager
        let backend = TractBackend::new();

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
    fn test_tract_backend_initialization_requirements() {
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("nonexistent-model".to_string()),
            variant: Some("fp32".to_string()),
        };

        // Test with invalid model manager
        let model_manager_result = ModelManager::from_spec(&model_spec);

        if let Ok(model_manager) = model_manager_result {
            let mut backend = TractBackend::with_model_manager(model_manager);
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
    fn test_tract_backend_thread_configuration() {
        // Test thread configuration without requiring cached model
        let mut backend = TractBackend::new();

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
    fn test_tract_backend_pure_rust_characteristics() {
        // Test pure Rust characteristics without model manager
        let backend = TractBackend::new();

        // Tract backend should be pure Rust - test basic properties
        assert!(!backend.is_initialized());

        // Should provide sensible defaults
        let input_shape = backend.input_shape();
        let output_shape = backend.output_shape();

        // Input should be RGB (3 channels)
        assert_eq!(input_shape.1, 3);
        // Output should be single channel mask
        assert_eq!(output_shape.1, 1);

        // Dimensions should be reasonable
        assert!(input_shape.2 >= 256 && input_shape.2 <= 2048);
        assert!(input_shape.3 >= 256 && input_shape.3 <= 2048);
        assert!(output_shape.2 >= 256 && output_shape.2 <= 2048);
        assert!(output_shape.3 >= 256 && output_shape.3 <= 2048);
    }

    #[test]
    fn test_tract_backend_model_management() {
        // Test model management behavior without cached model
        let backend = TractBackend::new();

        // Test model information access
        let model_info_result = backend.get_model_info();
        let preprocessing_result = backend.get_preprocessing_config();

        // Even if these fail due to missing models, they should fail gracefully
        match model_info_result {
            Ok(info) => {
                assert!(!info.name.is_empty());
                assert!(!info.precision.is_empty());
                assert!(info.size_bytes > 0);
                assert!(info.input_shape.0 > 0);
                assert!(info.output_shape.0 > 0);
            },
            Err(_) => {
                // Expected for test models that don't exist
            },
        }

        match preprocessing_result {
            Ok(config) => {
                assert!(config.target_size[0] > 0);
                assert!(config.target_size[1] > 0);
                assert!(config.normalization_mean.len() == 3);
                assert!(config.normalization_std.len() == 3);

                // Check normalization values are reasonable
                for &mean in &config.normalization_mean {
                    assert!(mean >= 0.0 && mean <= 255.0); // Allow both [0,1] and [0,255] ranges
                }
                for &std in &config.normalization_std {
                    assert!(std > 0.0 && std <= 255.0); // Allow both [0,1] and [0,255] ranges
                }
            },
            Err(_) => {
                // Expected for test models that don't exist
            },
        }
    }
}
