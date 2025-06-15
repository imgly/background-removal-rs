//! Model management and embedding system

use crate::error::Result;
use std::path::{Path, PathBuf};
use std::fs;

/// Model source specification
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Embedded model by name
    Embedded(String),
    /// External model from filesystem path
    External(PathBuf),
}

/// Complete model specification including source and optional variant
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub source: ModelSource,
    pub variant: Option<String>,
}

// Include generated model configuration and registry
include!(concat!(env!("OUT_DIR"), "/model_config.rs"));

/// Model information and metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub precision: String,
    pub size_bytes: usize,
    pub input_shape: (usize, usize, usize, usize), // NCHW format
    pub output_shape: (usize, usize, usize, usize),
}

/// Model provider trait for loading models
pub trait ModelProvider: std::fmt::Debug {
    /// Load model data as bytes
    ///
    /// # Errors
    /// - Model file not found or inaccessible
    /// - File I/O errors when reading model data
    /// - Invalid model file format
    fn load_model_data(&self) -> Result<Vec<u8>>;
    
    /// Get model information
    ///
    /// # Errors
    /// - Model configuration parsing errors
    /// - Missing required metadata fields
    /// - Invalid shape or precision information
    fn get_model_info(&self) -> Result<ModelInfo>;
    
    /// Get preprocessing configuration
    ///
    /// # Errors
    /// - Missing preprocessing configuration in model metadata
    /// - Invalid normalization or target size values
    /// - JSON parsing errors for external models
    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig>;
    
    /// Get input tensor name
    ///
    /// # Errors
    /// - Missing input tensor name in model configuration
    /// - Invalid or empty tensor name
    fn get_input_name(&self) -> Result<String>;
    
    /// Get output tensor name
    ///
    /// # Errors
    /// - Missing output tensor name in model configuration
    /// - Invalid or empty tensor name
    fn get_output_name(&self) -> Result<String>;
}

/// Embedded model provider for specific model by name
#[derive(Debug)]
pub struct EmbeddedModelProvider {
    model_name: String,
}

impl EmbeddedModelProvider {
    /// Create provider for specific embedded model
    ///
    /// # Errors
    /// - Requested model not found in embedded registry
    /// - Model name validation fails
    pub fn new(model_name: String) -> Result<Self> {
        // Validate model exists in registry
        if EmbeddedModelRegistry::get_model(&model_name).is_none() {
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Embedded model '{model_name}' not found. Available models: {:?}", 
                    EmbeddedModelRegistry::list_available())
            ));
        }
        
        Ok(Self { model_name })
    }
    
    /// List all available embedded models
    #[must_use] pub fn list_available() -> &'static [&'static str] {
        EmbeddedModelRegistry::list_available()
    }
    
    fn get_model_data(&self) -> Result<EmbeddedModelData> {
        EmbeddedModelRegistry::get_model(&self.model_name)
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config(
                format!("Embedded model '{}' not found", self.model_name)
            ))
    }
}

impl ModelProvider for EmbeddedModelProvider {
    fn load_model_data(&self) -> Result<Vec<u8>> {
        let model_data = self.get_model_data()?;
        Ok(model_data.model_data)
    }
    
    fn get_model_info(&self) -> Result<ModelInfo> {
        let model_data = self.get_model_data()?;
        Ok(ModelInfo {
            name: model_data.name.clone(),
            precision: extract_precision_from_name(&model_data.name),
            size_bytes: model_data.model_data.len(),
            input_shape: (
                model_data.input_shape[0],
                model_data.input_shape[1],
                model_data.input_shape[2], 
                model_data.input_shape[3]
            ),
            output_shape: (
                model_data.output_shape[0],
                model_data.output_shape[1],
                model_data.output_shape[2],
                model_data.output_shape[3]
            ),
        })
    }
    
    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        let model_data = self.get_model_data()?;
        Ok(model_data.preprocessing)
    }
    
    fn get_input_name(&self) -> Result<String> {
        let model_data = self.get_model_data()?;
        Ok(model_data.input_name)
    }
    
    fn get_output_name(&self) -> Result<String> {
        let model_data = self.get_model_data()?;
        Ok(model_data.output_name)
    }
}

/// External model provider for loading models from filesystem paths
#[derive(Debug)]
pub struct ExternalModelProvider {
    model_path: PathBuf,
    model_config: serde_json::Value,
    variant: String,
}

impl ExternalModelProvider {
    /// Create provider for external model from folder path
    ///
    /// # Errors
    /// - Model path does not exist or is not a directory
    /// - Missing or invalid model.json configuration file
    /// - JSON parsing errors in model configuration
    /// - Requested variant not found in model
    pub fn new<P: AsRef<Path>>(model_path: P, variant: Option<String>) -> Result<Self> {
        Self::new_with_provider(model_path, variant, None)
    }
    
    /// Create provider for external model from folder path with execution provider optimization
    ///
    /// # Errors
    /// - Model path does not exist or is not a directory
    /// - Missing or invalid model.json configuration file
    /// - JSON parsing errors in model configuration
    /// - Requested variant not found in model
    /// - Invalid execution provider configuration
    pub fn new_with_provider<P: AsRef<Path>>(
        model_path: P, 
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        
        // Validate path exists and is directory
        if !model_path.exists() {
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Model path does not exist: {}", model_path.display())
            ));
        }
        
        if !model_path.is_dir() {
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Model path must be a directory: {}", model_path.display())
            ));
        }
        
        // Load and validate model.json
        let model_json_path = model_path.join("model.json");
        if !model_json_path.exists() {
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("model.json not found in: {}", model_path.display())
            ));
        }
        
        let json_content = fs::read_to_string(&model_json_path)
            .map_err(|e| crate::error::BgRemovalError::invalid_config(
                format!("Failed to read model.json: {e}")
            ))?;
            
        let model_config: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| crate::error::BgRemovalError::invalid_config(
                format!("Failed to parse model.json: {e}")
            ))?;
        
        // Validate required fields
        Self::validate_model_config(&model_config)?;
        
        // Determine variant to use with execution provider optimization
        let resolved_variant = Self::resolve_variant_for_provider(&model_config, variant, execution_provider)?;
        
        // Validate variant exists
        let variants_obj = model_config.get("variants")
            .and_then(|v| v.as_object())
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config("variants section not found or not an object"))?;
        if !variants_obj.contains_key(&resolved_variant) {
            let available: Vec<String> = variants_obj.keys().cloned().collect();
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Variant '{resolved_variant}' not found. Available variants: {available:?}")
            ));
        }
        
        Ok(Self {
            model_path,
            model_config,
            variant: resolved_variant,
        })
    }
    
    fn validate_model_config(config: &serde_json::Value) -> Result<()> {
        // Check required top-level fields (description is optional for backward compatibility)
        let required_fields = ["name", "variants", "preprocessing"];
        for field in required_fields {
            if config.get(field).is_none() {
                return Err(crate::error::BgRemovalError::invalid_config(
                    format!("Missing required field '{field}' in model.json")
                ));
            }
        }
        
        // Check variants is an object
        if config.get("variants").map_or(true, |v| !v.is_object()) {
            return Err(crate::error::BgRemovalError::invalid_config(
                "Field 'variants' must be an object"
            ));
        }
        
        // Validate each variant has required fields
        let variants_obj = config.get("variants")
            .and_then(|v| v.as_object())
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config("variants section not found"))?;
        for (variant_name, variant_config) in variants_obj {
            let required_variant_fields = ["input_shape", "output_shape", "input_name", "output_name"];
            for field in required_variant_fields {
                if variant_config.get(field).is_none() {
                    return Err(crate::error::BgRemovalError::invalid_config(
                        format!("Missing required field '{field}' in variant '{variant_name}'")
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    
    /// Resolve variant with execution provider compatibility from model.json
    fn resolve_variant_for_provider(
        config: &serde_json::Value, 
        requested_variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>
    ) -> Result<String> {
        let available_variants: Vec<String> = config["variants"].as_object()
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config("variants section not found"))?
            .keys().cloned().collect();
            
        // If variant explicitly requested, use it (but warn if not optimal)
        if let Some(variant) = requested_variant {
            if available_variants.contains(&variant) {
                // Check if this variant is compatible with the execution provider
                if let Some(provider) = execution_provider {
                    Self::warn_if_incompatible(config, &variant, *provider);
                }
                return Ok(variant);
            }
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Requested variant '{variant}' not available. Available variants: {available_variants:?}")
            ));
        }
        
        // Auto-detection using provider_recommendations from model.json
        if let Some(provider) = execution_provider {
            if let Some(recommendations) = config.get("provider_recommendations") {
                let provider_name = Self::execution_provider_to_string(*provider);
                
                // First try the specific provider recommendation
                if let Some(recommended_variant) = recommendations.get(&provider_name) {
                    if let Some(variant_str) = recommended_variant.as_str() {
                        if available_variants.contains(&variant_str.to_string()) {
                            log::info!("🎯 Using model-recommended variant '{variant_str}' for {provider_name} provider");
                            return Ok(variant_str.to_string());
                        }
                    }
                }
                
                // For Auto provider, use the most compatible variant
                if matches!(provider, crate::config::ExecutionProvider::Auto) {
                    // Check for CoreML availability and use its recommendation if available
                    #[cfg(target_os = "macos")]
                    {
                        use ort::execution_providers::{CoreMLExecutionProvider, ExecutionProvider as OrtExecutionProvider};
                        if OrtExecutionProvider::is_available(&CoreMLExecutionProvider::default()).unwrap_or(false) {
                            if let Some(coreml_variant) = recommendations.get("coreml") {
                                if let Some(variant_str) = coreml_variant.as_str() {
                                    if available_variants.contains(&variant_str.to_string()) {
                                        log::info!("🍎 Auto provider: Using CoreML-optimized variant '{variant_str}' (Apple Silicon detected)");
                                        return Ok(variant_str.to_string());
                                    }
                                }
                            }
                        }
                    }
                    
                    // Fall back to CPU recommendation
                    if let Some(cpu_variant) = recommendations.get("cpu") {
                        if let Some(variant_str) = cpu_variant.as_str() {
                            if available_variants.contains(&variant_str.to_string()) {
                                log::info!("🖥️ Auto provider: Using CPU-optimized variant '{variant_str}'");
                                return Ok(variant_str.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback to old behavior: prefer fp16, then fp32
        if available_variants.contains(&"fp16".to_string()) {
            return Ok("fp16".to_string());
        }
        
        if available_variants.contains(&"fp32".to_string()) {
            return Ok("fp32".to_string());
        }
        
        // Use first available variant
        if let Some(first) = available_variants.first() {
            return Ok(first.clone());
        }
        
        Err(crate::error::BgRemovalError::invalid_config(
            "No variants available in model.json"
        ))
    }
    
    fn execution_provider_to_string(provider: crate::config::ExecutionProvider) -> String {
        match provider {
            crate::config::ExecutionProvider::Cpu => "cpu".to_string(),
            crate::config::ExecutionProvider::Cuda => "cuda".to_string(),
            crate::config::ExecutionProvider::CoreMl => "coreml".to_string(),
            crate::config::ExecutionProvider::Auto => "auto".to_string(),
        }
    }
    
    fn warn_if_incompatible(config: &serde_json::Value, variant: &str, provider: crate::config::ExecutionProvider) {
        if let Some(variants) = config.get("variants") {
            if let Some(variant_config) = variants.get(variant) {
                if let Some(compatible_providers) = variant_config.get("compatible_providers") {
                    if let Some(providers_array) = compatible_providers.as_array() {
                        let provider_name = Self::execution_provider_to_string(provider);
                        let is_compatible = providers_array.iter()
                            .any(|p| p.as_str() == Some(&provider_name));
                            
                        if !is_compatible {
                            log::warn!("⚠️ Variant '{variant}' may not be optimal for {provider_name} provider");
                            if let Some(notes) = variant_config.get("performance_notes") {
                                if let Some(notes_str) = notes.as_str() {
                                    log::warn!("   Note: {notes_str}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    fn get_variant_config(&self) -> &serde_json::Value {
        self.model_config
            .get("variants")
            .and_then(|variants| variants.get(&self.variant))
            .unwrap_or(&serde_json::Value::Null)
    }
    
    fn get_model_file_path(&self) -> PathBuf {
        self.model_path.join(format!("model_{}.onnx", self.variant))
    }
}

impl ModelProvider for ExternalModelProvider {
    fn load_model_data(&self) -> Result<Vec<u8>> {
        let model_file_path = self.get_model_file_path();
        
        if !model_file_path.exists() {
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Model file not found: {}", model_file_path.display())
            ));
        }
        
        fs::read(&model_file_path)
            .map_err(|e| crate::error::BgRemovalError::model(
                format!("Failed to read model file: {e}")
            ))
    }
    
    fn get_model_info(&self) -> Result<ModelInfo> {
        let variant_config = self.get_variant_config();
        let model_data = self.load_model_data()?;
        
        Ok(ModelInfo {
            name: format!("{}-{}", 
                self.model_config.get("name").and_then(|v| v.as_str()).unwrap_or("unknown"), 
                self.variant),
            precision: self.variant.clone(),
            size_bytes: model_data.len(),
            input_shape: (
                variant_config.get("input_shape").and_then(|arr| arr.get(0)).and_then(|v| v.as_u64()).unwrap_or(1) as usize,
                variant_config.get("input_shape").and_then(|arr| arr.get(1)).and_then(|v| v.as_u64()).unwrap_or(3) as usize,
                variant_config.get("input_shape").and_then(|arr| arr.get(2)).and_then(|v| v.as_u64()).unwrap_or(1024) as usize,
                variant_config.get("input_shape").and_then(|arr| arr.get(3)).and_then(|v| v.as_u64()).unwrap_or(1024) as usize,
            ),
            output_shape: (
                variant_config.get("output_shape").and_then(|arr| arr.get(0)).and_then(|v| v.as_u64()).unwrap_or(1) as usize,
                variant_config.get("output_shape").and_then(|arr| arr.get(1)).and_then(|v| v.as_u64()).unwrap_or(1) as usize,
                variant_config.get("output_shape").and_then(|arr| arr.get(2)).and_then(|v| v.as_u64()).unwrap_or(1024) as usize,
                variant_config.get("output_shape").and_then(|arr| arr.get(3)).and_then(|v| v.as_u64()).unwrap_or(1024) as usize,
            ),
        })
    }
    
    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        let preprocessing = self.model_config.get("preprocessing")
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing preprocessing config"))?;
        
        Ok(PreprocessingConfig {
            target_size: [
                preprocessing.get("target_size").and_then(|arr| arr.get(0))
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing target_size[0] in preprocessing config"))? as u32,
                preprocessing.get("target_size").and_then(|arr| arr.get(1))
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing target_size[1] in preprocessing config"))? as u32,
            ],
            normalization_mean: [
                preprocessing.get("normalization").and_then(|norm| norm.get("mean")).and_then(|arr| arr.get(0))
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing normalization mean[0] in preprocessing config"))? as f32,
                preprocessing.get("normalization").and_then(|norm| norm.get("mean")).and_then(|arr| arr.get(1))
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing normalization mean[1] in preprocessing config"))? as f32,
                preprocessing.get("normalization").and_then(|norm| norm.get("mean")).and_then(|arr| arr.get(2))
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing normalization mean[2] in preprocessing config"))? as f32,
            ],
            normalization_std: [
                preprocessing.get("normalization").and_then(|norm| norm.get("std")).and_then(|arr| arr.get(0))
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing normalization std[0] in preprocessing config"))? as f32,
                preprocessing.get("normalization").and_then(|norm| norm.get("std")).and_then(|arr| arr.get(1))
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing normalization std[1] in preprocessing config"))? as f32,
                preprocessing.get("normalization").and_then(|norm| norm.get("std")).and_then(|arr| arr.get(2))
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| crate::error::BgRemovalError::invalid_config("Missing normalization std[2] in preprocessing config"))? as f32,
            ],
        })
    }
    
    fn get_input_name(&self) -> Result<String> {
        let variant_config = self.get_variant_config();
        let input_name = variant_config["input_name"].as_str()
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config(
                format!("Missing or invalid input_name in variant '{}'", self.variant)
            ))?;
        Ok(input_name.to_string())
    }
    
    fn get_output_name(&self) -> Result<String> {
        let variant_config = self.get_variant_config();
        let output_name = variant_config["output_name"].as_str()
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config(
                format!("Missing or invalid output_name in variant '{}'", self.variant)
            ))?;
        Ok(output_name.to_string())
    }
}

/// Get list of all available embedded models
///
/// Returns a list of model names that are embedded in the current binary.
/// These models can be used without external files and are immediately available.
///
/// # Returns
///
/// A `Vec<String>` containing embedded model names like:
/// - `"isnet-fp16"` - Fast general-purpose model (FP16 precision)
/// - `"isnet-fp32"` - General-purpose model (FP32 precision, better `CoreML` performance)
/// - `"birefnet-fp16"` - High-quality portrait model (FP16 precision)
/// - `"birefnet-fp32"` - High-quality portrait model (FP32 precision)
/// - `"birefnet-lite-fp32"` - Balanced performance model (FP32 precision)
///
/// # Build Dependencies
///
/// The returned list depends on which embedding features were enabled at build time:
/// - `embed-isnet-fp16` - Includes `ISNet` FP16 model
/// - `embed-isnet-fp32` - Includes `ISNet` FP32 model  
/// - `embed-birefnet-fp16` - Includes `BiRefNet` FP16 model
/// - `embed-birefnet-fp32` - Includes `BiRefNet` FP32 model
/// - `embed-birefnet-lite-fp32` - Includes `BiRefNet` Lite FP32 model
/// - `embed-all` - Includes all available models
///
/// # Performance Characteristics
///
/// - **`ISNet` models**: Fastest inference, good general-purpose quality
/// - **`BiRefNet` models**: Highest quality, slower inference, excellent for portraits
/// - **`BiRefNet` Lite**: Balanced speed/quality, good compromise option
/// - **FP16 models**: Faster on CPU/CUDA, poor `CoreML` performance
/// - **FP32 models**: Better `CoreML` performance, slightly larger size
///
/// # Examples
///
/// ## Check available models
/// ```rust
/// use bg_remove_core::get_available_embedded_models;
///
/// let models = get_available_embedded_models();
/// if models.is_empty() {
///     println!("No embedded models available. Build with --features embed-all");
/// } else {
///     println!("Available models: {:?}", models);
/// }
/// ```
///
/// ## Select best model for execution provider
/// ```rust
/// use bg_remove_core::{get_available_embedded_models, ExecutionProvider};
///
/// fn select_optimal_model(provider: ExecutionProvider) -> Option<String> {
///     let models = get_available_embedded_models();
///     
///     match provider {
///         ExecutionProvider::CoreMl => {
///             // Prefer FP32 models for CoreML
///             models.iter().find(|m| m.contains("fp32")).cloned()
///         },
///         ExecutionProvider::Cuda | ExecutionProvider::Cpu => {
///             // Prefer FP16 models for CPU/CUDA
///             models.iter().find(|m| m.contains("fp16")).cloned()
///         },
///         ExecutionProvider::Auto => {
///             // Use first available model
///             models.first().cloned()
///         }
///     }
/// }
/// ```
///
/// ## Model selection priority
/// ```rust
/// use bg_remove_core::get_available_embedded_models;
///
/// fn get_recommended_model() -> Option<String> {
///     let models = get_available_embedded_models();
///     
///     // Priority: ISNet (fast) > BiRefNet Lite (balanced) > BiRefNet (quality)
///     ["isnet-fp16", "isnet-fp32", "birefnet-lite-fp32", "birefnet-fp16", "birefnet-fp32"]
///         .iter()
///         .find(|&name| models.contains(&name.to_string()))
///         .map(|&name| name.to_string())
/// }
/// ```
///
/// # Note
///
/// If no models are embedded (empty list), you must either:
/// 1. Rebuild with embedding features: `cargo build --features embed-all`
/// 2. Use external models with `ModelSource::External(path)`
#[must_use] pub fn get_available_embedded_models() -> Vec<String> {
    EmbeddedModelProvider::list_available().iter().map(|s| s.to_string()).collect()
}

/// Extract precision from model name (e.g., "isnet-fp16" -> "fp16")
#[must_use] fn extract_precision_from_name(name: &str) -> String {
    if name.contains("fp32") {
        "fp32".to_string()
    } else if name.contains("fp16") {
        "fp16".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Model manager for handling different model sources
#[derive(Debug)]
pub struct ModelManager {
    provider: Box<dyn ModelProvider>,
}

impl ModelManager {
    /// Create a new model manager from a model specification
    pub fn from_spec(spec: &ModelSpec) -> Result<Self> {
        Self::from_spec_with_provider(spec, None)
    }
    
    /// Create a new model manager from a model specification with execution provider optimization
    pub fn from_spec_with_provider(
        spec: &ModelSpec, 
        execution_provider: Option<&crate::config::ExecutionProvider>
    ) -> Result<Self> {
        match &spec.source {
            ModelSource::Embedded(model_name) => {
                Self::with_embedded_model(model_name.clone())
            },
            ModelSource::External(model_path) => {
                Self::with_external_model_and_provider(model_path, spec.variant.clone(), execution_provider)
            },
        }
    }
    
    /// Create a new model manager with specific embedded model
    pub fn with_embedded_model(model_name: String) -> Result<Self> {
        let provider = EmbeddedModelProvider::new(model_name)?;
        Ok(Self {
            provider: Box::new(provider),
        })
    }
    
    /// Create model manager with external model from folder path
    pub fn with_external_model<P: AsRef<Path>>(model_path: P, variant: Option<String>) -> Result<Self> {
        let provider = ExternalModelProvider::new(model_path, variant)?;
        Ok(Self {
            provider: Box::new(provider),
        })
    }
    
    /// Create model manager with external model from folder path and execution provider optimization
    pub fn with_external_model_and_provider<P: AsRef<Path>>(
        model_path: P, 
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>
    ) -> Result<Self> {
        let provider = ExternalModelProvider::new_with_provider(model_path, variant, execution_provider)?;
        Ok(Self {
            provider: Box::new(provider),
        })
    }
    
    /// Create model manager with first available embedded model (legacy compatibility)
    pub fn with_embedded() -> Result<Self> {
        let available = EmbeddedModelProvider::list_available();
        if available.is_empty() {
            return Err(crate::error::BgRemovalError::invalid_config(
                "No embedded models available. Build with embed-* features or use external model."
            ));
        }
        
        Self::with_embedded_model(available.first()
            .ok_or_else(|| crate::error::BgRemovalError::invalid_config("No embedded models available"))?
            .to_string())
    }
    
    /// Load model data
    pub fn load_model(&self) -> Result<Vec<u8>> {
        self.provider.load_model_data()
    }
    
    /// Get model information
    pub fn get_info(&self) -> Result<ModelInfo> {
        self.provider.get_model_info()
    }
    
    /// Get preprocessing configuration
    pub fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        self.provider.get_preprocessing_config()
    }
    
    /// Get input tensor name
    pub fn get_input_name(&self) -> Result<String> {
        self.provider.get_input_name()
    }
    
    /// Get output tensor name
    pub fn get_output_name(&self) -> Result<String> {
        self.provider.get_output_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_embedded_model_registry() {
        let available = EmbeddedModelProvider::list_available();
        
        // Registry should work even with no models (empty list)
        // When models are embedded, they should be accessible
        for model_name in available {
            let provider = EmbeddedModelProvider::new(model_name.to_string()).unwrap();
            let info = provider.get_model_info().unwrap();
            assert!(!info.name.is_empty());
            assert!(!info.precision.is_empty());
        }
    }

    #[test]
    fn test_model_manager_creation() {
        let available = EmbeddedModelProvider::list_available();
        
        if available.is_empty() {
            // No embedded models - should error gracefully
            let result = ModelManager::with_embedded();
            assert!(result.is_err());
        } else {
            // With embedded models - should work
            let manager = ModelManager::with_embedded().unwrap();
            let info = manager.get_info().unwrap();
            assert!(!info.name.is_empty());
        }
    }
    
    #[test]
    fn test_external_model_validation() {
        // Test nonexistent path
        let result = ExternalModelProvider::new("nonexistent", None);
        assert!(result.is_err());
        
        // Test path without model.json
        let temp_dir = std::env::temp_dir().join("test_empty_model");
        let _ = fs::create_dir_all(&temp_dir);
        let result = ExternalModelProvider::new(&temp_dir, None);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&temp_dir);
    }
    
    #[test] 
    fn test_model_spec_creation() {
        let spec = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: Some("fp16".to_string()),
        };
        
        assert!(matches!(spec.source, ModelSource::Embedded(_)));
        assert_eq!(spec.variant, Some("fp16".to_string()));
    }
    
    #[test]
    fn test_extract_precision_from_name() {
        assert_eq!(extract_precision_from_name("isnet-fp16"), "fp16");
        assert_eq!(extract_precision_from_name("birefnet-fp32"), "fp32");
        assert_eq!(extract_precision_from_name("unknown-model"), "unknown");
    }
}