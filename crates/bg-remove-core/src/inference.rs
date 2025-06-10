//! Inference backend abstraction and ONNX Runtime implementation

use crate::{
    config::{ModelPrecision, RemovalConfig, ExecutionProvider},
    error::Result,
    models::ModelManager,
};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::{self, value::Value};
use ort::execution_providers::{CoreMLExecutionProvider, CUDAExecutionProvider};

/// Trait for inference backends
pub trait InferenceBackend {
    /// Initialize the backend with the given configuration
    fn initialize(&mut self, config: &RemovalConfig) -> Result<()>;
    
    /// Run inference on the input tensor
    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>>;
    
    /// Get the expected input shape for this backend
    fn input_shape(&self) -> (usize, usize, usize, usize);
    
    /// Get the expected output shape for this backend
    fn output_shape(&self) -> (usize, usize, usize, usize);
}

/// ONNX Runtime inference backend
pub struct OnnxBackend {
    session: Option<Session>,
    model_manager: ModelManager,
    precision: ModelPrecision,
    initialized: bool,
}

impl OnnxBackend {
    /// Create a new ONNX backend
    pub fn new() -> Self {
        Self {
            session: None,
            model_manager: ModelManager::with_embedded(),
            precision: ModelPrecision::Fp16,
            initialized: false,
        }
    }

    /// Create ONNX backend with external model
    pub fn with_external_model<P: Into<std::path::PathBuf>>(path: P) -> Self {
        Self {
            session: None,
            model_manager: ModelManager::with_external(path),
            precision: ModelPrecision::Fp16,
            initialized: false,
        }
    }

    /// Load and initialize the ONNX model
    fn load_model(&mut self, config: &RemovalConfig) -> Result<()> {
        // Load the model data
        let model_data = self.model_manager.load_model(config.model_precision)?;
        
        // Create ONNX Runtime session with specified execution provider
        let mut session_builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;

        // Configure execution providers based on user choice
        session_builder = match config.execution_provider {
            ExecutionProvider::Auto => {
                // Auto-detect: try CUDA > CoreML > CPU
                session_builder.with_execution_providers([
                    CUDAExecutionProvider::default().build(),
                    CoreMLExecutionProvider::default().build(),
                ])?
            },
            ExecutionProvider::Cpu => {
                // CPU only
                session_builder
            },
            ExecutionProvider::Cuda => {
                // CUDA only (will fall back to CPU if not available)
                session_builder.with_execution_providers([
                    CUDAExecutionProvider::default().build(),
                ])?
            },
            ExecutionProvider::CoreMl => {
                // CoreML only (will fall back to CPU if not available)
                session_builder.with_execution_providers([
                    CoreMLExecutionProvider::default().build(),
                ])?
            },
        };

        let session = session_builder
            .with_intra_threads(if config.num_threads > 0 { config.num_threads } else { 4 })?
            .commit_from_memory(&model_data)?;
        
        // Log which execution provider is being used
        log::info!("ONNX Runtime session created with execution provider: {:?}", config.execution_provider);
        
        self.session = Some(session);
        self.precision = config.model_precision;
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

        // If external model is specified, use that
        if let Some(ref model_path) = config.model_path {
            self.model_manager = ModelManager::with_external(model_path.clone());
        }

        self.load_model(config)?;
        Ok(())
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        if !self.initialized {
            return Err(crate::error::BgRemovalError::internal(
                "Backend not initialized"
            ));
        }

        let session = self.session.as_mut().ok_or_else(|| {
            crate::error::BgRemovalError::internal("ONNX session not initialized")
        })?;

        // Convert ndarray to ort Value
        let input_value = Value::from_array(input.clone())
            .map_err(|e| crate::error::BgRemovalError::processing(&format!("Failed to convert input tensor: {}", e)))?;
        
        // Run inference - ISNet models typically use "input" as the input name
        let outputs = session.run(ort::inputs!["input" => input_value])
            .map_err(|e| crate::error::BgRemovalError::processing(&format!("ONNX inference failed. This might be due to incorrect input name. Original error: {}", e)))?;
        
        // Extract output tensor - try common output names
        let output_tensor = if let Ok(output) = outputs["output"].try_extract_array::<f32>() {
            output
        } else if let Ok(output) = outputs[0].try_extract_array::<f32>() {
            // Try first output if named access fails
            output
        } else {
            return Err(crate::error::BgRemovalError::processing(
                "Failed to extract output tensor from ONNX model"
            ));
        };
        
        // Convert output to Array4<f32> - reshape if needed
        let output_shape = output_tensor.shape();
        let output_data = output_tensor.view().to_owned();
        
        if output_shape.len() == 4 {
            let output_array = Array4::from_shape_vec(
                (output_shape[0], output_shape[1], output_shape[2], output_shape[3]),
                output_data.into_raw_vec_and_offset().0,
            ).map_err(|e| crate::error::BgRemovalError::processing(&format!("Failed to reshape output tensor: {}", e)))?;
            
            Ok(output_array)
        } else {
            Err(crate::error::BgRemovalError::processing(
                &format!("Expected 4D output tensor, got {}D", output_shape.len())
            ))
        }
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        // Standard ISNet input shape: NCHW (1, 3, 1024, 1024)
        (1, 3, 1024, 1024)
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        // Standard ISNet output shape: NCHW (1, 1, 1024, 1024)
        (1, 1, 1024, 1024)
    }
}

/// Mock backend for testing
pub struct MockBackend {
    input_shape: (usize, usize, usize, usize),
    output_shape: (usize, usize, usize, usize),
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            input_shape: (1, 3, 1024, 1024),
            output_shape: (1, 1, 1024, 1024),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockBackend {
    fn initialize(&mut self, _config: &RemovalConfig) -> Result<()> {
        // Mock backend doesn't need initialization
        Ok(())
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        let (n, _c, h, w) = input.dim();
        
        // Create a mock segmentation mask (simple edge detection)
        let mut output = Array4::<f32>::zeros((n, 1, h, w));
        
        for batch in 0..n {
            for y in 1..h-1 {
                for x in 1..w-1 {
                    // Simple edge detection as mock segmentation
                    let center = input[[batch, 0, y, x]];
                    let left = input[[batch, 0, y, x-1]];
                    let right = input[[batch, 0, y, x+1]];
                    let top = input[[batch, 0, y-1, x]];
                    let bottom = input[[batch, 0, y+1, x]];
                    
                    let edge_strength = ((center - left).abs() + 
                                       (center - right).abs() + 
                                       (center - top).abs() + 
                                       (center - bottom).abs()) / 4.0;
                    
                    // Create a reasonable mock mask
                    output[[batch, 0, y, x]] = if edge_strength > 0.1 { 1.0 } else { 0.0 };
                }
            }
        }
        
        Ok(output)
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        self.input_shape
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        self.output_shape
    }
}

/// Backend registry for managing different inference backends
pub struct BackendRegistry {
    backends: std::collections::HashMap<String, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            backends: std::collections::HashMap::new(),
        };
        
        // Register default backends
        registry.register("onnx", Box::new(OnnxBackend::new()));
        registry.register("mock", Box::new(MockBackend::new()));
        
        registry
    }

    pub fn register(&mut self, name: &str, backend: Box<dyn InferenceBackend>) {
        self.backends.insert(name.to_string(), backend);
    }

    pub fn get(&mut self, name: &str) -> Option<&mut Box<dyn InferenceBackend>> {
        self.backends.get_mut(name)
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RemovalConfig;

    #[test]
    fn test_mock_backend() {
        let mut backend = MockBackend::new();
        let config = RemovalConfig::default();
        
        backend.initialize(&config).unwrap();
        
        // Test shapes
        assert_eq!(backend.input_shape(), (1, 3, 1024, 1024));
        assert_eq!(backend.output_shape(), (1, 1, 1024, 1024));
        
        // Test inference with small tensor
        let input = Array4::<f32>::zeros((1, 3, 4, 4));
        let output = backend.infer(&input).unwrap();
        assert_eq!(output.dim(), (1, 1, 4, 4));
    }

    #[test]
    fn test_backend_registry() {
        let mut registry = BackendRegistry::new();
        
        // Test that default backends are registered
        assert!(registry.get("mock").is_some());
        assert!(registry.get("onnx").is_some());
        assert!(registry.get("nonexistent").is_none());
    }
}