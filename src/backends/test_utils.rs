//! Test utilities and mock backends for testing inference functionality
//!
//! This module provides mock implementations of the `InferenceBackend` trait
//! to enable comprehensive testing without requiring actual model files or
//! external dependencies like ONNX Runtime or Tract.

use crate::{
    config::RemovalConfig,
    error::{BgRemovalError, Result},
    inference::InferenceBackend,
    models::{ModelInfo, PreprocessingConfig},
};
use instant::Duration;
use ndarray::Array4;
use std::sync::{Arc, Mutex};

/// Mock ONNX backend for testing
#[derive(Debug, Clone)]
pub struct MockOnnxBackend {
    /// Whether the backend has been initialized
    initialized: bool,
    /// Simulated model information
    model_info: ModelInfo,
    /// Preprocessing configuration
    preprocessing_config: PreprocessingConfig,
    /// Call history for verification in tests
    call_history: Arc<Mutex<Vec<String>>>,
    /// Whether to simulate initialization failure
    should_fail_init: bool,
    /// Whether to simulate inference failure
    should_fail_inference: bool,
}

impl MockOnnxBackend {
    /// Create a new mock ONNX backend with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            model_info: ModelInfo {
                name: "mock-onnx-model".to_string(),
                precision: "fp32".to_string(),
                size_bytes: 1024 * 1024, // 1MB
                input_shape: (1, 3, 320, 320),
                output_shape: (1, 1, 320, 320),
            },
            preprocessing_config: PreprocessingConfig {
                target_size: [320, 320],
                normalization_mean: [0.485, 0.456, 0.406],
                normalization_std: [0.229, 0.224, 0.225],
            },
            call_history: Arc::new(Mutex::new(Vec::new())),
            should_fail_init: false,
            should_fail_inference: false,
        }
    }

    /// Create a mock backend that will fail during initialization
    #[must_use]
    pub fn new_failing_init() -> Self {
        let mut backend = Self::new();
        backend.should_fail_init = true;
        backend
    }

    /// Create a mock backend that will fail during inference
    #[must_use]
    pub fn new_failing_inference() -> Self {
        let mut backend = Self::new();
        backend.should_fail_inference = true;
        backend
    }

    /// Get the call history for verification in tests
    pub fn get_call_history(&self) -> Vec<String> {
        self.call_history.lock().unwrap().clone()
    }

    /// Clear the call history
    pub fn clear_call_history(&self) {
        self.call_history.lock().unwrap().clear();
    }

    /// Record a method call for testing verification
    fn record_call(&self, method: &str) {
        if let Ok(mut history) = self.call_history.lock() {
            history.push(method.to_string());
        }
    }

    /// Generate a mock output tensor based on input dimensions
    fn generate_mock_output(&self, input: &Array4<f32>) -> Array4<f32> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let output_height = self.model_info.output_shape.2;
        let output_width = self.model_info.output_shape.3;

        // Create a simple mock mask with a circular pattern
        let mut output = Array4::<f32>::zeros((batch_size, 1, output_height, output_width));

        let center_x = output_width as f32 / 2.0;
        let center_y = output_height as f32 / 2.0;
        let radius = (output_width.min(output_height) as f32 / 3.0).max(10.0);

        for b in 0..batch_size {
            for y in 0..output_height {
                for x in 0..output_width {
                    let dx = x as f32 - center_x;
                    let dy = y as f32 - center_y;
                    let distance = (dx * dx + dy * dy).sqrt();

                    // Create a circular mask with soft edges
                    let mask_value = if distance < radius {
                        ((radius - distance) / radius).max(0.0).min(1.0)
                    } else {
                        0.0
                    };

                    output[[b, 0, y, x]] = mask_value;
                }
            }
        }

        output
    }
}

impl Default for MockOnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockOnnxBackend {
    fn initialize(&mut self, _config: &RemovalConfig) -> Result<Option<Duration>> {
        self.record_call("initialize");

        if self.should_fail_init {
            return Err(BgRemovalError::processing(
                "Mock ONNX backend initialization failed",
            ));
        }

        self.initialized = true;
        // Simulate a reasonable initialization time
        Ok(Some(Duration::from_millis(150)))
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        self.record_call("infer");

        if !self.initialized {
            return Err(BgRemovalError::processing(
                "Mock ONNX backend not initialized",
            ));
        }

        if self.should_fail_inference {
            return Err(BgRemovalError::processing(
                "Mock ONNX backend inference failed",
            ));
        }

        // Validate input shape
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BgRemovalError::processing(
                "Input tensor must be 4-dimensional (NCHW)",
            ));
        }

        Ok(self.generate_mock_output(input))
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        self.record_call("input_shape");
        self.model_info.input_shape
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        self.record_call("output_shape");
        self.model_info.output_shape
    }

    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        self.record_call("get_preprocessing_config");
        Ok(self.preprocessing_config.clone())
    }

    fn get_model_info(&self) -> Result<ModelInfo> {
        self.record_call("get_model_info");
        Ok(self.model_info.clone())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Mock Tract backend for testing
#[derive(Debug, Clone)]
pub struct MockTractBackend {
    /// Whether the backend has been initialized
    initialized: bool,
    /// Simulated model information
    model_info: ModelInfo,
    /// Preprocessing configuration
    preprocessing_config: PreprocessingConfig,
    /// Call history for verification in tests
    call_history: Arc<Mutex<Vec<String>>>,
    /// Whether to simulate initialization failure
    should_fail_init: bool,
    /// Whether to simulate inference failure
    should_fail_inference: bool,
}

impl MockTractBackend {
    /// Create a new mock Tract backend with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            model_info: ModelInfo {
                name: "mock-tract-model".to_string(),
                precision: "fp32".to_string(),
                size_bytes: 512 * 1024, // 512KB (smaller than ONNX for realism)
                input_shape: (1, 3, 256, 256),
                output_shape: (1, 1, 256, 256),
            },
            preprocessing_config: PreprocessingConfig {
                target_size: [256, 256],
                normalization_mean: [0.5, 0.5, 0.5],
                normalization_std: [0.5, 0.5, 0.5],
            },
            call_history: Arc::new(Mutex::new(Vec::new())),
            should_fail_init: false,
            should_fail_inference: false,
        }
    }

    /// Create a mock backend that will fail during initialization
    #[must_use]
    pub fn new_failing_init() -> Self {
        let mut backend = Self::new();
        backend.should_fail_init = true;
        backend
    }

    /// Create a mock backend that will fail during inference
    #[must_use]
    pub fn new_failing_inference() -> Self {
        let mut backend = Self::new();
        backend.should_fail_inference = true;
        backend
    }

    /// Get the call history for verification in tests
    pub fn get_call_history(&self) -> Vec<String> {
        self.call_history.lock().unwrap().clone()
    }

    /// Clear the call history
    pub fn clear_call_history(&self) {
        self.call_history.lock().unwrap().clear();
    }

    /// Record a method call for testing verification
    fn record_call(&self, method: &str) {
        if let Ok(mut history) = self.call_history.lock() {
            history.push(method.to_string());
        }
    }

    /// Generate a mock output tensor with a different pattern than ONNX
    fn generate_mock_output(&self, input: &Array4<f32>) -> Array4<f32> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let output_height = self.model_info.output_shape.2;
        let output_width = self.model_info.output_shape.3;

        // Create a simple mock mask with a rectangular pattern
        let mut output = Array4::<f32>::zeros((batch_size, 1, output_height, output_width));

        let margin_x = output_width / 8;
        let margin_y = output_height / 8;

        for b in 0..batch_size {
            for y in 0..output_height {
                for x in 0..output_width {
                    // Create a rectangular mask with soft edges
                    let mask_value = if x >= margin_x
                        && x < (output_width - margin_x)
                        && y >= margin_y
                        && y < (output_height - margin_y)
                    {
                        // Create gradient from edges
                        let edge_dist_x = (x - margin_x).min(output_width - margin_x - x);
                        let edge_dist_y = (y - margin_y).min(output_height - margin_y - y);
                        let edge_dist = edge_dist_x.min(edge_dist_y) as f32;
                        let fade_width = 20.0;

                        (edge_dist / fade_width).min(1.0)
                    } else {
                        0.0
                    };

                    output[[b, 0, y, x]] = mask_value;
                }
            }
        }

        output
    }
}

impl Default for MockTractBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockTractBackend {
    fn initialize(&mut self, _config: &RemovalConfig) -> Result<Option<Duration>> {
        self.record_call("initialize");

        if self.should_fail_init {
            return Err(BgRemovalError::processing(
                "Mock Tract backend initialization failed",
            ));
        }

        self.initialized = true;
        // Simulate faster initialization than ONNX (pure Rust advantage)
        Ok(Some(Duration::from_millis(50)))
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        self.record_call("infer");

        if !self.initialized {
            return Err(BgRemovalError::processing(
                "Mock Tract backend not initialized",
            ));
        }

        if self.should_fail_inference {
            return Err(BgRemovalError::processing(
                "Mock Tract backend inference failed",
            ));
        }

        // Validate input shape
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BgRemovalError::processing(
                "Input tensor must be 4-dimensional (NCHW)",
            ));
        }

        Ok(self.generate_mock_output(input))
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        self.record_call("input_shape");
        self.model_info.input_shape
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        self.record_call("output_shape");
        self.model_info.output_shape
    }

    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        self.record_call("get_preprocessing_config");
        Ok(self.preprocessing_config.clone())
    }

    fn get_model_info(&self) -> Result<ModelInfo> {
        self.record_call("get_model_info");
        Ok(self.model_info.clone())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Test factory for creating mock backends
#[derive(Debug)]
pub struct MockBackendFactory {
    /// Whether to create failing backends
    pub create_failing_backends: bool,
    /// Whether to fail backend creation entirely
    pub fail_backend_creation: bool,
}

impl MockBackendFactory {
    /// Create a new mock factory with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            create_failing_backends: false,
            fail_backend_creation: false,
        }
    }

    /// Create a factory that produces failing backends
    #[must_use]
    pub fn new_failing() -> Self {
        Self {
            create_failing_backends: true,
            fail_backend_creation: false,
        }
    }

    /// Create a factory that fails to create backends
    #[must_use]
    pub fn new_creation_failing() -> Self {
        Self {
            create_failing_backends: false,
            fail_backend_creation: true,
        }
    }
}

impl Default for MockBackendFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::processor::BackendFactory for MockBackendFactory {
    fn create_backend(
        &self,
        backend_type: crate::processor::BackendType,
        _model_manager: crate::models::ModelManager,
    ) -> Result<Box<dyn InferenceBackend>> {
        if self.fail_backend_creation {
            return Err(BgRemovalError::processing(
                "Mock factory configured to fail backend creation",
            ));
        }

        match backend_type {
            crate::processor::BackendType::Onnx => {
                if self.create_failing_backends {
                    Ok(Box::new(MockOnnxBackend::new_failing_init()))
                } else {
                    Ok(Box::new(MockOnnxBackend::new()))
                }
            },
            crate::processor::BackendType::Tract => {
                if self.create_failing_backends {
                    Ok(Box::new(MockTractBackend::new_failing_init()))
                } else {
                    Ok(Box::new(MockTractBackend::new()))
                }
            },
        }
    }

    fn available_backends(&self) -> Vec<crate::processor::BackendType> {
        vec![
            crate::processor::BackendType::Onnx,
            crate::processor::BackendType::Tract,
        ]
    }
}

/// Helper functions for creating test images and tensors
pub mod test_helpers {
    use super::*;
    use image::DynamicImage;
    use ndarray::Array4;

    /// Create a test image with specified dimensions
    pub fn create_test_image(width: u32, height: u32) -> DynamicImage {
        use image::{ImageBuffer, Rgb};

        let img = ImageBuffer::from_fn(width, height, |x, y| {
            // Create a simple gradient pattern
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = 128;
            Rgb([r, g, b])
        });

        DynamicImage::ImageRgb8(img)
    }

    /// Create a test tensor with specified shape
    pub fn create_test_tensor(
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Array4<f32> {
        Array4::<f32>::from_shape_fn((batch, channels, height, width), |(b, c, h, w)| {
            // Create a simple pattern for testing
            (b + c + h + w) as f32 / (batch + channels + height + width) as f32
        })
    }

    /// Create a test preprocessing config
    pub fn create_test_preprocessing_config() -> PreprocessingConfig {
        PreprocessingConfig {
            target_size: [320, 320],
            normalization_mean: [0.485, 0.456, 0.406],
            normalization_std: [0.229, 0.224, 0.225],
        }
    }

    /// Create a test model info
    pub fn create_test_model_info() -> ModelInfo {
        ModelInfo {
            name: "test-model".to_string(),
            precision: "fp32".to_string(),
            size_bytes: 1024 * 1024,
            input_shape: (1, 3, 320, 320),
            output_shape: (1, 1, 320, 320),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ExecutionProvider, OutputFormat};
    use crate::models::{ModelSource, ModelSpec};
    use crate::processor::BackendFactory;

    #[test]
    fn test_mock_onnx_backend_creation() {
        let backend = MockOnnxBackend::new();
        assert!(!backend.is_initialized());
        assert_eq!(backend.input_shape(), (1, 3, 320, 320));
        assert_eq!(backend.output_shape(), (1, 1, 320, 320));
    }

    #[test]
    fn test_mock_tract_backend_creation() {
        let backend = MockTractBackend::new();
        assert!(!backend.is_initialized());
        assert_eq!(backend.input_shape(), (1, 3, 256, 256));
        assert_eq!(backend.output_shape(), (1, 1, 256, 256));
    }

    #[test]
    fn test_mock_backend_initialization() {
        let mut backend = MockOnnxBackend::new();
        let config = RemovalConfig {
            execution_provider: ExecutionProvider::Cpu,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: false,
            disable_cache: false,
            model_spec: ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp32".to_string()),
            },
            format_hint: None,
        };

        assert!(!backend.is_initialized());
        let result = backend.initialize(&config);
        assert!(result.is_ok());
        assert!(backend.is_initialized());

        // Check initialization time is reasonable
        let init_time = result.unwrap();
        assert!(init_time.is_some());
        assert!(init_time.unwrap() > Duration::from_millis(0));
    }

    #[test]
    fn test_mock_backend_inference() {
        let mut backend = MockOnnxBackend::new();
        let config = RemovalConfig {
            execution_provider: ExecutionProvider::Cpu,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: false,
            disable_cache: false,
            model_spec: ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp32".to_string()),
            },
            format_hint: None,
        };

        // Initialize backend
        backend.initialize(&config).unwrap();

        // Create test input
        let input = test_helpers::create_test_tensor(1, 3, 320, 320);

        // Run inference
        let output = backend.infer(&input).unwrap();

        // Verify output shape
        assert_eq!(output.shape(), &[1, 1, 320, 320]);

        // Verify output values are in valid range (0-1 for mask)
        for value in output.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }

    #[test]
    fn test_mock_backend_failure_scenarios() {
        // Test initialization failure
        let mut failing_backend = MockOnnxBackend::new_failing_init();
        let config = RemovalConfig {
            execution_provider: ExecutionProvider::Cpu,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: false,
            disable_cache: false,
            model_spec: ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp32".to_string()),
            },
            format_hint: None,
        };

        let result = failing_backend.initialize(&config);
        assert!(result.is_err());
        assert!(!failing_backend.is_initialized());

        // Test inference failure
        let mut inference_failing_backend = MockOnnxBackend::new_failing_inference();
        inference_failing_backend.initialize(&config).unwrap();

        let input = test_helpers::create_test_tensor(1, 3, 320, 320);
        let result = inference_failing_backend.infer(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_backend_call_history() {
        let backend = MockOnnxBackend::new();

        // Check initial state
        assert!(backend.get_call_history().is_empty());

        // Make some calls
        let _ = backend.input_shape();
        let _ = backend.output_shape();
        let _ = backend.get_preprocessing_config();

        let history = backend.get_call_history();
        assert_eq!(history.len(), 3);
        assert!(history.contains(&"input_shape".to_string()));
        assert!(history.contains(&"output_shape".to_string()));
        assert!(history.contains(&"get_preprocessing_config".to_string()));

        // Clear history
        backend.clear_call_history();
        assert!(backend.get_call_history().is_empty());
    }

    #[test]
    fn test_mock_backend_factory() {
        let factory = MockBackendFactory::new();

        // Test available backends
        let backends = factory.available_backends();
        assert_eq!(backends.len(), 2);
        assert!(backends.contains(&crate::processor::BackendType::Onnx));
        assert!(backends.contains(&crate::processor::BackendType::Tract));

        // Test backend creation
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: Some("fp32".to_string()),
        };
        let model_manager = crate::models::ModelManager::from_spec(&model_spec).unwrap();

        let onnx_backend =
            factory.create_backend(crate::processor::BackendType::Onnx, model_manager);
        assert!(onnx_backend.is_ok());

        // Create a second model manager for tract backend
        let model_manager2 = crate::models::ModelManager::from_spec(&model_spec).unwrap();
        let tract_backend =
            factory.create_backend(crate::processor::BackendType::Tract, model_manager2);
        assert!(tract_backend.is_ok());
    }

    #[test]
    fn test_mock_backend_factory_failures() {
        // Test failing factory
        let failing_factory = MockBackendFactory::new_creation_failing();
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: Some("fp32".to_string()),
        };
        let model_manager = crate::models::ModelManager::from_spec(&model_spec).unwrap();

        let result =
            failing_factory.create_backend(crate::processor::BackendType::Onnx, model_manager);
        assert!(result.is_err());

        // Test factory that creates failing backends
        let factory_with_failing_backends = MockBackendFactory::new_failing();
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: Some("fp32".to_string()),
        };
        let model_manager = crate::models::ModelManager::from_spec(&model_spec).unwrap();

        let backend = factory_with_failing_backends
            .create_backend(crate::processor::BackendType::Onnx, model_manager)
            .unwrap();

        // Backend should be created but will fail on initialization
        let config = RemovalConfig {
            execution_provider: ExecutionProvider::Cpu,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: false,
            disable_cache: false,
            model_spec: ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp32".to_string()),
            },
            format_hint: None,
        };

        let mut backend_mut = backend;
        let result = backend_mut.initialize(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_different_backend_patterns() {
        let mut onnx_backend = MockOnnxBackend::new();
        let mut tract_backend = MockTractBackend::new();

        let config = RemovalConfig {
            execution_provider: ExecutionProvider::Cpu,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: false,
            disable_cache: false,
            model_spec: ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp32".to_string()),
            },
            format_hint: None,
        };

        // Initialize both backends
        onnx_backend.initialize(&config).unwrap();
        tract_backend.initialize(&config).unwrap();

        // Test different output patterns
        let input = test_helpers::create_test_tensor(1, 3, 256, 256);

        // Note: ONNX backend expects 320x320 input but should handle 256x256 gracefully
        let onnx_input = test_helpers::create_test_tensor(1, 3, 320, 320);
        let onnx_output = onnx_backend.infer(&onnx_input).unwrap();
        let tract_output = tract_backend.infer(&input).unwrap();

        // Outputs should be different patterns (circular vs rectangular)
        assert_eq!(onnx_output.shape(), &[1, 1, 320, 320]);
        assert_eq!(tract_output.shape(), &[1, 1, 256, 256]);

        // Both should have valid mask values
        for value in onnx_output.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
        for value in tract_output.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }
}
