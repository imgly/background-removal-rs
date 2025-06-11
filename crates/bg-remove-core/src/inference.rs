//! Inference backend abstraction and ONNX Runtime implementation

use crate::{
    config::{ExecutionProvider, RemovalConfig},
    error::Result,
    models::ModelManager,
};
use ndarray::Array4;
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider as OrtExecutionProvider,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::{self, value::Value};

/// Check availability of all execution providers
pub fn check_provider_availability() -> Vec<(String, bool, String)> {
    let mut providers = Vec::new();

    // CPU is always available
    providers.push((
        "CPU".to_string(),
        true,
        "Always available, uses CPU for inference".to_string(),
    ));

    // Check CUDA availability
    let cuda_available =
        OrtExecutionProvider::is_available(&CUDAExecutionProvider::default()).unwrap_or(false);
    providers.push((
        "CUDA".to_string(),
        cuda_available,
        "NVIDIA GPU acceleration (requires CUDA toolkit and compatible GPU)".to_string(),
    ));

    // Check CoreML availability
    let coreml_available =
        OrtExecutionProvider::is_available(&CoreMLExecutionProvider::default()).unwrap_or(false);
    providers.push((
        "CoreML".to_string(),
        coreml_available,
        "Apple Silicon GPU acceleration (macOS only)".to_string(),
    ));

    providers
}

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
    initialized: bool,
}

impl OnnxBackend {
    /// Create a new ONNX backend
    pub fn new() -> Self {
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
                .map(|n| n.get())
                .unwrap_or(8)
        };

        let inter_threads = if config.inter_threads > 0 {
            config.inter_threads
        } else {
            // Optimal inter-op threads: Use fewer threads for coordination (typically 1-4)
            (std::thread::available_parallelism()
                .map(|n| n.get())
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
            "Threading: {} intra-op threads, {} inter-op threads, parallel execution enabled",
            intra_threads,
            inter_threads
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
            crate::error::BgRemovalError::processing(&format!(
                "Failed to convert input tensor: {}",
                e
            ))
        })?;

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
                crate::error::BgRemovalError::processing(&format!(
                    "Failed to reshape output tensor: {}",
                    e
                ))
            })?;

            Ok(output_array)
        } else {
            Err(crate::error::BgRemovalError::processing(&format!(
                "Expected 4D output tensor, got {}D",
                output_shape.len()
            )))
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
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    // Simple edge detection as mock segmentation
                    let center = input[[batch, 0, y, x]];
                    let left = input[[batch, 0, y, x - 1]];
                    let right = input[[batch, 0, y, x + 1]];
                    let top = input[[batch, 0, y - 1, x]];
                    let bottom = input[[batch, 0, y + 1, x]];

                    let edge_strength = ((center - left).abs()
                        + (center - right).abs()
                        + (center - top).abs()
                        + (center - bottom).abs())
                        / 4.0;

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
