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
    fn load_model_data(&self) -> Result<Vec<u8>>;
    
    /// Get model information
    fn get_model_info(&self) -> Result<ModelInfo>;
    
    /// Get preprocessing configuration
    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig>;
    
    /// Get input tensor name
    fn get_input_name(&self) -> Result<String>;
    
    /// Get output tensor name  
    fn get_output_name(&self) -> Result<String>;
}

/// Embedded model provider for specific model by name
#[derive(Debug)]
pub struct EmbeddedModelProvider {
    model_name: String,
}

impl EmbeddedModelProvider {
    /// Create provider for specific embedded model
    pub fn new(model_name: String) -> Result<Self> {
        // Validate model exists in registry
        if EmbeddedModelRegistry::get_model(&model_name).is_none() {
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Embedded model '{}' not found. Available models: {:?}", 
                    model_name, EmbeddedModelRegistry::list_available())
            ));
        }
        
        Ok(Self { model_name })
    }
    
    /// List all available embedded models
    pub fn list_available() -> &'static [&'static str] {
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
    pub fn new<P: AsRef<Path>>(model_path: P, variant: Option<String>) -> Result<Self> {
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
                format!("Failed to read model.json: {}", e)
            ))?;
            
        let model_config: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| crate::error::BgRemovalError::invalid_config(
                format!("Failed to parse model.json: {}", e)
            ))?;
        
        // Validate required fields
        Self::validate_model_config(&model_config)?;
        
        // Determine variant to use
        let resolved_variant = Self::resolve_variant(&model_config, variant)?;
        
        // Validate variant exists
        if !model_config["variants"].as_object().unwrap().contains_key(&resolved_variant) {
            let available: Vec<String> = model_config["variants"].as_object().unwrap()
                .keys().map(|k| k.clone()).collect();
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Variant '{}' not found. Available variants: {:?}", resolved_variant, available)
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
            if !config.get(field).is_some() {
                return Err(crate::error::BgRemovalError::invalid_config(
                    format!("Missing required field '{}' in model.json", field)
                ));
            }
        }
        
        // Check variants is an object
        if !config["variants"].is_object() {
            return Err(crate::error::BgRemovalError::invalid_config(
                "Field 'variants' must be an object"
            ));
        }
        
        // Validate each variant has required fields
        for (variant_name, variant_config) in config["variants"].as_object().unwrap() {
            let required_variant_fields = ["input_shape", "output_shape", "input_name", "output_name"];
            for field in required_variant_fields {
                if !variant_config.get(field).is_some() {
                    return Err(crate::error::BgRemovalError::invalid_config(
                        format!("Missing required field '{}' in variant '{}'", field, variant_name)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    fn resolve_variant(config: &serde_json::Value, requested_variant: Option<String>) -> Result<String> {
        let available_variants: Vec<String> = config["variants"].as_object().unwrap()
            .keys().map(|k| k.clone()).collect();
            
        // If variant explicitly requested, use it
        if let Some(variant) = requested_variant {
            if available_variants.contains(&variant) {
                return Ok(variant);
            } else {
                return Err(crate::error::BgRemovalError::invalid_config(
                    format!("Requested variant '{}' not available. Available variants: {:?}", 
                        variant, available_variants)
                ));
            }
        }
        
        // Auto-detection: prefer fp16, fallback to available
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
    
    fn get_variant_config(&self) -> &serde_json::Value {
        &self.model_config["variants"][&self.variant]
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
                format!("Failed to read model file: {}", e)
            ))
    }
    
    fn get_model_info(&self) -> Result<ModelInfo> {
        let variant_config = self.get_variant_config();
        let model_data = self.load_model_data()?;
        
        Ok(ModelInfo {
            name: format!("{}-{}", self.model_config["name"].as_str().unwrap(), self.variant),
            precision: self.variant.clone(),
            size_bytes: model_data.len(),
            input_shape: (
                variant_config["input_shape"][0].as_u64().unwrap() as usize,
                variant_config["input_shape"][1].as_u64().unwrap() as usize,
                variant_config["input_shape"][2].as_u64().unwrap() as usize,
                variant_config["input_shape"][3].as_u64().unwrap() as usize,
            ),
            output_shape: (
                variant_config["output_shape"][0].as_u64().unwrap() as usize,
                variant_config["output_shape"][1].as_u64().unwrap() as usize,
                variant_config["output_shape"][2].as_u64().unwrap() as usize,
                variant_config["output_shape"][3].as_u64().unwrap() as usize,
            ),
        })
    }
    
    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        let preprocessing = &self.model_config["preprocessing"];
        
        Ok(PreprocessingConfig {
            target_size: [
                preprocessing["target_size"][0].as_u64().unwrap() as u32,
                preprocessing["target_size"][1].as_u64().unwrap() as u32,
            ],
            normalization_mean: [
                preprocessing["normalization"]["mean"][0].as_f64().unwrap() as f32,
                preprocessing["normalization"]["mean"][1].as_f64().unwrap() as f32,
                preprocessing["normalization"]["mean"][2].as_f64().unwrap() as f32,
            ],
            normalization_std: [
                preprocessing["normalization"]["std"][0].as_f64().unwrap() as f32,
                preprocessing["normalization"]["std"][1].as_f64().unwrap() as f32,
                preprocessing["normalization"]["std"][2].as_f64().unwrap() as f32,
            ],
        })
    }
    
    fn get_input_name(&self) -> Result<String> {
        let variant_config = self.get_variant_config();
        Ok(variant_config["input_name"].as_str().unwrap().to_string())
    }
    
    fn get_output_name(&self) -> Result<String> {
        let variant_config = self.get_variant_config();
        Ok(variant_config["output_name"].as_str().unwrap().to_string())
    }
}

/// Extract precision from model name (e.g., "isnet-fp16" -> "fp16")
fn extract_precision_from_name(name: &str) -> String {
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
        match &spec.source {
            ModelSource::Embedded(model_name) => {
                Self::with_embedded_model(model_name.clone())
            },
            ModelSource::External(model_path) => {
                Self::with_external_model(model_path, spec.variant.clone())
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
    
    /// Create model manager with first available embedded model (legacy compatibility)
    pub fn with_embedded() -> Result<Self> {
        let available = EmbeddedModelProvider::list_available();
        if available.is_empty() {
            return Err(crate::error::BgRemovalError::invalid_config(
                "No embedded models available. Build with embed-* features or use external model."
            ));
        }
        
        Self::with_embedded_model(available[0].to_string())
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