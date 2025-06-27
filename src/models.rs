//! Model management and embedding system

use crate::error::Result;
use std::fs;
use std::path::{Path, PathBuf};

/// Model source specification
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// External model from filesystem path
    External(PathBuf),
    /// Downloaded model from cache by model ID
    Downloaded(String),
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

    /// Get input tensor name (deprecated - using positional inputs)
    ///
    /// # Deprecated
    /// This method is kept for compatibility only. Inference now uses positional inputs.
    ///
    /// # Errors
    /// - Missing input tensor name in model configuration
    /// - Invalid or empty tensor name
    fn get_input_name(&self) -> Result<String>;

    /// Get output tensor name (deprecated - using positional inputs)
    ///
    /// # Deprecated
    /// This method is kept for compatibility only. Inference now uses positional inputs.
    ///
    /// # Errors
    /// - Missing output tensor name in model configuration
    /// - Invalid or empty tensor name
    fn get_output_name(&self) -> Result<String>;
}

/// Model format detection
#[derive(Debug, Clone, PartialEq)]
enum ModelFormat {
    /// Legacy format with model.json
    Legacy,
    /// HuggingFace format with `config.json` + `preprocessor_config.json`
    HuggingFace,
}

/// External model provider for loading models from filesystem paths
#[derive(Debug)]
pub struct ExternalModelProvider {
    model_path: PathBuf,
    format: ModelFormat,
    model_config: serde_json::Value,
    preprocessor_config: Option<serde_json::Value>,
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
    /// - Missing or invalid configuration files
    /// - JSON parsing errors in model configuration
    /// - Requested variant not found in model
    /// - Invalid execution provider configuration
    pub fn new_with_provider<P: AsRef<Path>>(
        model_path: P,
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        // Validate path exists and is directory
        if !model_path.exists() {
            return Err(crate::error::BgRemovalError::invalid_config(format!(
                "Model path does not exist: {}",
                model_path.display()
            )));
        }

        if !model_path.is_dir() {
            return Err(crate::error::BgRemovalError::invalid_config(format!(
                "Model path must be a directory: {}",
                model_path.display()
            )));
        }

        // Detect model format
        let format = Self::detect_model_format(&model_path)?;

        match format {
            ModelFormat::Legacy => Self::new_legacy_format(model_path, variant, execution_provider),
            ModelFormat::HuggingFace => {
                Self::new_huggingface_format(model_path, variant, execution_provider)
            },
        }
    }

    /// Detect model format based on available configuration files
    fn detect_model_format(model_path: &Path) -> Result<ModelFormat> {
        let legacy_config = model_path.join("model.json");
        let hf_config = model_path.join("config.json");
        let hf_preprocessor = model_path.join("preprocessor_config.json");

        if hf_config.exists() && hf_preprocessor.exists() {
            Ok(ModelFormat::HuggingFace)
        } else if legacy_config.exists() {
            Ok(ModelFormat::Legacy)
        } else {
            Err(crate::error::BgRemovalError::invalid_config(format!(
                "No valid model configuration found in: {}. Expected either model.json (legacy) or config.json + preprocessor_config.json (HuggingFace)",
                model_path.display()
            )))
        }
    }

    /// Create provider for legacy format model
    fn new_legacy_format(
        model_path: PathBuf,
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        // Load and validate model.json
        let model_json_path = model_path.join("model.json");
        let json_content = fs::read_to_string(&model_json_path).map_err(|e| {
            crate::error::BgRemovalError::invalid_config(format!("Failed to read model.json: {e}"))
        })?;

        let model_config: serde_json::Value = serde_json::from_str(&json_content).map_err(|e| {
            crate::error::BgRemovalError::invalid_config(format!("Failed to parse model.json: {e}"))
        })?;

        // Validate required fields
        Self::validate_model_config(&model_config)?;

        // Determine variant to use with execution provider optimization
        let resolved_variant =
            Self::resolve_variant_for_provider(&model_config, variant, execution_provider)?;

        // Validate variant exists
        let variants_obj = model_config
            .get("variants")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config(
                    "variants section not found or not an object",
                )
            })?;
        if !variants_obj.contains_key(&resolved_variant) {
            let available: Vec<String> = variants_obj.keys().cloned().collect();
            let suggestions: Vec<&str> = if available.is_empty() {
                vec![]
            } else {
                vec![
                    "check model.json configuration",
                    "verify model files are complete",
                ]
            };
            return Err(crate::error::BgRemovalError::model_error_with_context(
                "load variant",
                &model_path,
                &format!("variant '{resolved_variant}' not found. Available: {available:?}"),
                &suggestions,
            ));
        }

        Ok(Self {
            model_path,
            format: ModelFormat::Legacy,
            model_config,
            preprocessor_config: None,
            variant: resolved_variant,
        })
    }

    /// Create provider for `HuggingFace` format model
    fn new_huggingface_format(
        model_path: PathBuf,
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        // Load config.json
        let config_path = model_path.join("config.json");
        let config_content = fs::read_to_string(&config_path).map_err(|e| {
            crate::error::BgRemovalError::invalid_config(format!("Failed to read config.json: {e}"))
        })?;

        let model_config: serde_json::Value =
            serde_json::from_str(&config_content).map_err(|e| {
                crate::error::BgRemovalError::invalid_config(format!(
                    "Failed to parse config.json: {e}"
                ))
            })?;

        // Load preprocessor_config.json
        let preprocessor_path = model_path.join("preprocessor_config.json");
        let preprocessor_content = fs::read_to_string(&preprocessor_path).map_err(|e| {
            crate::error::BgRemovalError::invalid_config(format!(
                "Failed to read preprocessor_config.json: {e}"
            ))
        })?;

        let preprocessor_config: serde_json::Value = serde_json::from_str(&preprocessor_content)
            .map_err(|e| {
                crate::error::BgRemovalError::invalid_config(format!(
                    "Failed to parse preprocessor_config.json: {e}"
                ))
            })?;

        // Determine variant from available ONNX files
        let resolved_variant =
            Self::resolve_huggingface_variant(&model_path, variant, execution_provider)?;

        Ok(Self {
            model_path,
            format: ModelFormat::HuggingFace,
            model_config,
            preprocessor_config: Some(preprocessor_config),
            variant: resolved_variant,
        })
    }

    /// Resolve variant for `HuggingFace` format by scanning onnx directory
    fn resolve_huggingface_variant(
        model_path: &Path,
        requested_variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<String> {
        let onnx_dir = model_path.join("onnx");
        if !onnx_dir.exists() || !onnx_dir.is_dir() {
            return Err(crate::error::BgRemovalError::invalid_config(format!(
                "onnx directory not found in HuggingFace model: {}",
                model_path.display()
            )));
        }

        // Scan for available ONNX files
        let mut available_variants = Vec::new();
        if let Ok(entries) = fs::read_dir(&onnx_dir) {
            for entry in entries.flatten() {
                if let Some(file_name) = entry.file_name().to_str() {
                    if file_name.ends_with(".onnx") {
                        if file_name == "model.onnx" {
                            available_variants.push("fp32".to_string());
                        } else if file_name == "model_fp16.onnx" {
                            available_variants.push("fp16".to_string());
                        }
                    }
                }
            }
        }

        if available_variants.is_empty() {
            return Err(crate::error::BgRemovalError::invalid_config(format!(
                "No ONNX model files found in: {}",
                onnx_dir.display()
            )));
        }

        // If variant explicitly requested, use it
        if let Some(variant) = requested_variant {
            if available_variants.contains(&variant) {
                return Ok(variant);
            }
            return Err(crate::error::BgRemovalError::invalid_config(
                format!("Requested variant '{variant}' not available. Available variants: {available_variants:?}")
            ));
        }

        // Auto-select based on execution provider
        if let Some(provider) = execution_provider {
            match provider {
                crate::config::ExecutionProvider::Cuda | crate::config::ExecutionProvider::Cpu => {
                    // Prefer FP16 for CPU/CUDA
                    if available_variants.contains(&"fp16".to_string()) {
                        return Ok("fp16".to_string());
                    }
                },
                crate::config::ExecutionProvider::CoreMl => {
                    // Prefer FP32 for CoreML
                    if available_variants.contains(&"fp32".to_string()) {
                        return Ok("fp32".to_string());
                    }
                },
                crate::config::ExecutionProvider::Auto => {
                    // On macOS, prefer FP32, otherwise FP16
                    #[cfg(target_os = "macos")]
                    {
                        if available_variants.contains(&"fp32".to_string()) {
                            return Ok("fp32".to_string());
                        }
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        if available_variants.contains(&"fp16".to_string()) {
                            return Ok("fp16".to_string());
                        }
                    }
                },
            }
        }

        // Fallback: prefer fp16, then fp32
        if available_variants.contains(&"fp16".to_string()) {
            return Ok("fp16".to_string());
        }

        if available_variants.contains(&"fp32".to_string()) {
            return Ok("fp32".to_string());
        }

        // Use first available variant
        Ok(available_variants.into_iter().next().unwrap())
    }

    fn validate_model_config(config: &serde_json::Value) -> Result<()> {
        // Check required top-level fields (description is optional for backward compatibility)
        let required_fields = ["name", "variants", "preprocessing"];
        for field in required_fields {
            if config.get(field).is_none() {
                return Err(crate::error::BgRemovalError::invalid_config(format!(
                    "Missing required field '{field}' in model.json"
                )));
            }
        }

        // Check variants is an object
        if config.get("variants").map_or(true, |v| !v.is_object()) {
            return Err(crate::error::BgRemovalError::invalid_config(
                "Field 'variants' must be an object",
            ));
        }

        // Validate each variant has required fields
        let variants_obj = config
            .get("variants")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config("variants section not found")
            })?;
        for (variant_name, variant_config) in variants_obj {
            let required_variant_fields =
                ["input_shape", "output_shape", "input_name", "output_name"];
            for field in required_variant_fields {
                if variant_config.get(field).is_none() {
                    return Err(crate::error::BgRemovalError::invalid_config(format!(
                        "Missing required field '{field}' in variant '{variant_name}'"
                    )));
                }
            }
        }

        Ok(())
    }

    /// Resolve variant with execution provider compatibility from model.json
    fn resolve_variant_for_provider(
        config: &serde_json::Value,
        requested_variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<String> {
        let available_variants: Vec<String> = config["variants"]
            .as_object()
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config("variants section not found")
            })?
            .keys()
            .cloned()
            .collect();

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
                            log::debug!("🎯 Using model-recommended variant '{variant_str}' for {provider_name} provider");
                            return Ok(variant_str.to_string());
                        }
                    }
                }

                // For Auto provider, use the most compatible variant
                if matches!(provider, crate::config::ExecutionProvider::Auto) {
                    // On macOS, prefer CoreML-optimized variants for auto provider
                    #[cfg(target_os = "macos")]
                    {
                        if let Some(coreml_variant) = recommendations.get("coreml") {
                            if let Some(variant_str) = coreml_variant.as_str() {
                                if available_variants.contains(&variant_str.to_string()) {
                                    log::debug!("🍎 Auto provider: Using CoreML-optimized variant '{variant_str}' (macOS detected)");
                                    return Ok(variant_str.to_string());
                                }
                            }
                        }
                    }

                    // Fall back to CPU recommendation
                    if let Some(cpu_variant) = recommendations.get("cpu") {
                        if let Some(variant_str) = cpu_variant.as_str() {
                            if available_variants.contains(&variant_str.to_string()) {
                                log::debug!(
                                    "🖥️ Auto provider: Using CPU-optimized variant '{variant_str}'"
                                );
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
            "No variants available in model.json",
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

    fn warn_if_incompatible(
        config: &serde_json::Value,
        variant: &str,
        provider: crate::config::ExecutionProvider,
    ) {
        if let Some(variants) = config.get("variants") {
            if let Some(variant_config) = variants.get(variant) {
                if let Some(compatible_providers) = variant_config.get("compatible_providers") {
                    if let Some(providers_array) = compatible_providers.as_array() {
                        let provider_name = Self::execution_provider_to_string(provider);
                        let is_compatible = providers_array
                            .iter()
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
        match self.format {
            ModelFormat::Legacy => self
                .model_path
                .join(format!("model_{variant}.onnx", variant = self.variant)),
            ModelFormat::HuggingFace => {
                let onnx_dir = self.model_path.join("onnx");
                match self.variant.as_str() {
                    "fp16" => onnx_dir.join("model_fp16.onnx"),
                    "fp32" => onnx_dir.join("model.onnx"),
                    _ => onnx_dir.join("model.onnx"), // Fallback to default
                }
            },
        }
    }

    /// Parse shape from legacy format variant config
    fn parse_shape_from_legacy(
        variant_config: &serde_json::Value,
        shape_key: &str,
        default: (usize, usize, usize, usize),
    ) -> Result<(usize, usize, usize, usize)> {
        let shape_array = variant_config.get(shape_key).and_then(|v| v.as_array());

        if let Some(arr) = shape_array {
            if arr.len() >= 4 {
                let dim0 = arr[0].as_u64().unwrap_or(default.0 as u64) as usize;
                let dim1 = arr[1].as_u64().unwrap_or(default.1 as u64) as usize;
                let dim2 = arr[2].as_u64().unwrap_or(default.2 as u64) as usize;
                let dim3 = arr[3].as_u64().unwrap_or(default.3 as u64) as usize;
                return Ok((dim0, dim1, dim2, dim3));
            }
        }

        Ok(default)
    }

    /// Parse target size from legacy format
    fn parse_target_size_legacy(preprocessing: &serde_json::Value) -> Result<[u32; 2]> {
        let size0_u64 = preprocessing
            .get("target_size")
            .and_then(|arr| arr.get(0))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config(
                    "Missing target_size[0] in preprocessing config",
                )
            })?;
        let size1_u64 = preprocessing
            .get("target_size")
            .and_then(|arr| arr.get(1))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config(
                    "Missing target_size[1] in preprocessing config",
                )
            })?;

        let size0 = size0_u64.try_into().map_err(|_| {
            crate::error::BgRemovalError::invalid_config("Target size[0] too large for u32")
        })?;
        let size1 = size1_u64.try_into().map_err(|_| {
            crate::error::BgRemovalError::invalid_config("Target size[1] too large for u32")
        })?;

        Ok([size0, size1])
    }

    /// Parse target size from `HuggingFace` format
    fn parse_target_size_huggingface(preprocessor: &serde_json::Value) -> Result<[u32; 2]> {
        let size = preprocessor.get("size").ok_or_else(|| {
            crate::error::BgRemovalError::invalid_config("Missing size in preprocessor config")
        })?;

        let height = size
            .get("height")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config("Missing height in size config")
            })?;

        let width = size
            .get("width")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config("Missing width in size config")
            })?;

        let height_u32 = height.try_into().map_err(|_| {
            crate::error::BgRemovalError::invalid_config("Height too large for u32")
        })?;
        let width_u32 = width
            .try_into()
            .map_err(|_| crate::error::BgRemovalError::invalid_config("Width too large for u32"))?;

        Ok([height_u32, width_u32])
    }

    /// Parse normalization values from legacy format
    fn parse_normalization_legacy(
        preprocessing: &serde_json::Value,
        key: &str,
    ) -> Result<[f32; 3]> {
        let values = preprocessing
            .get("normalization")
            .and_then(|norm| norm.get(key))
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config(format!(
                    "Missing normalization {key} in preprocessing config"
                ))
            })?;

        if values.len() < 3 {
            return Err(crate::error::BgRemovalError::invalid_config(format!(
                "Normalization {key} must have at least 3 values"
            )));
        }

        let v0 = values[0].as_f64().ok_or_else(|| {
            crate::error::BgRemovalError::invalid_config(format!("Invalid {key}[0] value"))
        })? as f32;
        let v1 = values[1].as_f64().ok_or_else(|| {
            crate::error::BgRemovalError::invalid_config(format!("Invalid {key}[1] value"))
        })? as f32;
        let v2 = values[2].as_f64().ok_or_else(|| {
            crate::error::BgRemovalError::invalid_config(format!("Invalid {key}[2] value"))
        })? as f32;

        Ok([v0, v1, v2])
    }

    /// Parse image_mean from `HuggingFace` format (convert from 0-255 to 0-1 range)
    fn parse_image_mean_huggingface(preprocessor: &serde_json::Value) -> Result<[f32; 3]> {
        let image_mean = preprocessor
            .get("image_mean")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config(
                    "Missing image_mean in preprocessor config",
                )
            })?;

        if image_mean.len() < 3 {
            return Err(crate::error::BgRemovalError::invalid_config(
                "image_mean must have at least 3 values",
            ));
        }

        // Convert from 0-255 range to 0-1 range
        let mean0 = (image_mean[0].as_f64().unwrap_or(128.0) / 255.0) as f32;
        let mean1 = (image_mean[1].as_f64().unwrap_or(128.0) / 255.0) as f32;
        let mean2 = (image_mean[2].as_f64().unwrap_or(128.0) / 255.0) as f32;

        Ok([mean0, mean1, mean2])
    }

    /// Parse image_std from `HuggingFace` format (convert from 0-255 to 0-1 range)
    fn parse_image_std_huggingface(preprocessor: &serde_json::Value) -> Result<[f32; 3]> {
        let image_std = preprocessor
            .get("image_std")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config(
                    "Missing image_std in preprocessor config",
                )
            })?;

        if image_std.len() < 3 {
            return Err(crate::error::BgRemovalError::invalid_config(
                "image_std must have at least 3 values",
            ));
        }

        // Convert from 0-255 range to 0-1 range
        let std0 = (image_std[0].as_f64().unwrap_or(256.0) / 255.0) as f32;
        let std1 = (image_std[1].as_f64().unwrap_or(256.0) / 255.0) as f32;
        let std2 = (image_std[2].as_f64().unwrap_or(256.0) / 255.0) as f32;

        Ok([std0, std1, std2])
    }
}

impl ModelProvider for ExternalModelProvider {
    fn load_model_data(&self) -> Result<Vec<u8>> {
        let model_file_path = self.get_model_file_path();

        if !model_file_path.exists() {
            return Err(crate::error::BgRemovalError::invalid_config(format!(
                "Model file not found: {}",
                model_file_path.display()
            )));
        }

        fs::read(&model_file_path).map_err(|e| {
            crate::error::BgRemovalError::model(format!("Failed to read model file: {e}"))
        })
    }

    #[allow(clippy::too_many_lines)] // Complex model configuration parsing with extensive JSON extraction
    fn get_model_info(&self) -> Result<ModelInfo> {
        let model_data = self.load_model_data()?;

        match self.format {
            ModelFormat::Legacy => {
                let variant_config = self.get_variant_config();
                Ok(ModelInfo {
                    name: format!(
                        "{}-{}",
                        self.model_config
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown"),
                        self.variant
                    ),
                    precision: self.variant.clone(),
                    size_bytes: model_data.len(),
                    input_shape: Self::parse_shape_from_legacy(
                        variant_config,
                        "input_shape",
                        (1, 3, 1024, 1024),
                    )?,
                    output_shape: Self::parse_shape_from_legacy(
                        variant_config,
                        "output_shape",
                        (1, 1, 1024, 1024),
                    )?,
                })
            },
            ModelFormat::HuggingFace => {
                let model_type = self
                    .model_config
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                // For HuggingFace models, infer shapes from preprocessor config
                let preprocessor = self.preprocessor_config.as_ref().ok_or_else(|| {
                    crate::error::BgRemovalError::invalid_config(
                        "Missing preprocessor config for HuggingFace model",
                    )
                })?;

                let size = preprocessor.get("size").ok_or_else(|| {
                    crate::error::BgRemovalError::invalid_config(
                        "Missing size in preprocessor config",
                    )
                })?;

                let height = size.get("height").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;
                let width = size.get("width").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;

                Ok(ModelInfo {
                    name: format!("{}-{}", model_type, self.variant),
                    precision: self.variant.clone(),
                    size_bytes: model_data.len(),
                    input_shape: (1, 3, height, width), // Standard format: NCHW
                    output_shape: (1, 1, height, width), // Single channel mask output
                })
            },
        }
    }

    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        match self.format {
            ModelFormat::Legacy => {
                let preprocessing = self.model_config.get("preprocessing").ok_or_else(|| {
                    crate::error::BgRemovalError::invalid_config("Missing preprocessing config")
                })?;

                Ok(PreprocessingConfig {
                    target_size: Self::parse_target_size_legacy(preprocessing)?,
                    normalization_mean: Self::parse_normalization_legacy(preprocessing, "mean")?,
                    normalization_std: Self::parse_normalization_legacy(preprocessing, "std")?,
                })
            },
            ModelFormat::HuggingFace => {
                let preprocessor = self.preprocessor_config.as_ref().ok_or_else(|| {
                    crate::error::BgRemovalError::invalid_config(
                        "Missing preprocessor config for HuggingFace model",
                    )
                })?;

                Ok(PreprocessingConfig {
                    target_size: Self::parse_target_size_huggingface(preprocessor)?,
                    normalization_mean: Self::parse_image_mean_huggingface(preprocessor)?,
                    normalization_std: Self::parse_image_std_huggingface(preprocessor)?,
                })
            },
        }
    }

    fn get_input_name(&self) -> Result<String> {
        // NOTE: With positional inputs, tensor names are no longer used for inference
        // This method is kept for compatibility and debugging purposes only
        match self.format {
            ModelFormat::Legacy => {
                let variant_config = self.get_variant_config();
                let input_name = variant_config["input_name"].as_str().ok_or_else(|| {
                    crate::error::BgRemovalError::invalid_config(format!(
                        "Missing or invalid input_name in variant '{variant}'",
                        variant = self.variant
                    ))
                })?;
                Ok(input_name.to_string())
            },
            ModelFormat::HuggingFace => {
                // Generic name for HuggingFace models since we use positional inputs
                Ok("input".to_string())
            },
        }
    }

    fn get_output_name(&self) -> Result<String> {
        // NOTE: With positional inputs, tensor names are no longer used for inference
        // This method is kept for compatibility and debugging purposes only
        match self.format {
            ModelFormat::Legacy => {
                let variant_config = self.get_variant_config();
                let output_name = variant_config["output_name"].as_str().ok_or_else(|| {
                    crate::error::BgRemovalError::invalid_config(format!(
                        "Missing or invalid output_name in variant '{variant}'",
                        variant = self.variant
                    ))
                })?;
                Ok(output_name.to_string())
            },
            ModelFormat::HuggingFace => {
                // Generic name for HuggingFace models since we use positional inputs
                Ok("output".to_string())
            },
        }
    }
}

/// Model manager for handling different model sources
#[derive(Debug)]
pub struct ModelManager {
    provider: Box<dyn ModelProvider>,
}

impl ModelManager {
    /// Create a new model manager from a model specification
    ///
    /// # Errors
    /// - Embedded model not found in registry if using `ModelSource::Embedded`
    /// - Model validation failures for embedded models
    /// - External model path does not exist or is not a directory if using `ModelSource::External`
    /// - Missing or invalid `model.json` configuration file for external models
    /// - JSON parsing errors in external model configuration
    /// - Requested variant not found in external model
    /// - File system I/O errors when accessing external model files
    pub fn from_spec(spec: &ModelSpec) -> Result<Self> {
        Self::from_spec_with_provider(spec, None)
    }

    /// Create a new model manager from a model specification with execution provider optimization
    ///
    /// # Errors
    /// - Model path does not exist or is not a directory
    /// - Missing or invalid model configuration files
    /// - JSON parsing errors when reading configuration
    /// - Invalid variant configuration or execution provider settings
    pub fn from_spec_with_provider(
        spec: &ModelSpec,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        match &spec.source {
            ModelSource::External(model_path) => Self::with_external_model_and_provider(
                model_path,
                spec.variant.clone(),
                execution_provider,
            ),
            ModelSource::Downloaded(model_id) => Self::with_downloaded_model(
                model_id.clone(),
                spec.variant.clone(),
                execution_provider,
            ),
        }
    }

    /// Create model manager with external model from folder path
    ///
    /// # Errors
    /// - Model path does not exist or is not a directory
    /// - Missing or invalid `model.json` configuration file in model directory
    /// - JSON parsing errors when reading `model.json`
    /// - Missing required fields in model configuration (name, variants, preprocessing)
    /// - Requested variant not found in available variants
    /// - Invalid variant configuration (missing `input_shape`, `output_shape`, tensor names)
    /// - File system I/O errors when accessing model directory or files
    pub fn with_external_model<P: AsRef<Path>>(
        model_path: P,
        variant: Option<String>,
    ) -> Result<Self> {
        let provider = ExternalModelProvider::new(model_path, variant)?;
        Ok(Self {
            provider: Box::new(provider),
        })
    }

    /// Create model manager with external model from folder path and execution provider optimization
    ///
    /// # Errors
    /// - Model path does not exist or is not a directory
    /// - Missing or invalid `model.json` configuration file in model directory
    /// - JSON parsing errors when reading `model.json`
    /// - Missing required fields in model configuration (name, variants, preprocessing)
    /// - Requested variant not found in available variants
    /// - Invalid variant configuration (missing `input_shape`, `output_shape`, tensor names)
    /// - File system I/O errors when accessing model directory or files
    /// - Execution provider configuration or optimization errors
    pub fn with_external_model_and_provider<P: AsRef<Path>>(
        model_path: P,
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        let provider =
            ExternalModelProvider::new_with_provider(model_path, variant, execution_provider)?;
        Ok(Self {
            provider: Box::new(provider),
        })
    }

    /// Create model manager with downloaded model from cache
    ///
    /// # Errors
    /// - Model not found in cache directory
    /// - Missing or invalid configuration files in cached model
    /// - JSON parsing errors when reading cached model configuration
    /// - Requested variant not available in cached model
    /// - Cache directory access errors
    /// - Execution provider configuration errors
    pub fn with_downloaded_model(
        model_id: String,
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        let provider = DownloadedModelProvider::new(model_id, variant, execution_provider)?;
        Ok(Self {
            provider: Box::new(provider),
        })
    }

    /// Load model data
    ///
    /// # Errors
    /// - Model file not found or inaccessible (for external models)
    /// - File I/O errors when reading model data
    /// - Invalid model file format or corrupted data
    /// - Insufficient permissions to read model file
    /// - Memory allocation failures for large models
    pub fn load_model(&self) -> Result<Vec<u8>> {
        self.provider.load_model_data()
    }

    /// Get model information
    ///
    /// # Errors
    /// - Model configuration parsing errors
    /// - Missing required metadata fields (name, shapes, precision)
    /// - Invalid shape or precision information in model config
    /// - Numeric conversion errors when parsing shape dimensions
    /// - Model registry access failures for embedded models
    pub fn get_info(&self) -> Result<ModelInfo> {
        self.provider.get_model_info()
    }

    /// Get preprocessing configuration
    ///
    /// # Errors
    /// - Missing preprocessing configuration in model metadata
    /// - Invalid normalization values (mean, std arrays)
    /// - Invalid or missing target size values
    /// - JSON parsing errors for external model preprocessing config
    /// - Numeric conversion errors when parsing float/integer values
    pub fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        self.provider.get_preprocessing_config()
    }

    /// Get input tensor name
    ///
    /// # Errors
    /// - Missing input tensor name in model configuration
    /// - Invalid or empty tensor name
    /// - Model variant configuration access errors
    /// - Registry lookup failures for embedded models
    pub fn get_input_name(&self) -> Result<String> {
        self.provider.get_input_name()
    }

    /// Get output tensor name
    ///
    /// # Errors
    /// - Missing output tensor name in model configuration
    /// - Invalid or empty tensor name
    /// - Model variant configuration access errors
    /// - Registry lookup failures for embedded models
    pub fn get_output_name(&self) -> Result<String> {
        self.provider.get_output_name()
    }
}

/// Downloaded model provider for cached models
///
/// This provider loads models from the cache directory that were previously
/// downloaded from URLs. It supports the HuggingFace model format with
/// automatic variant detection and preprocessing configuration.
#[cfg(feature = "cli")]
#[derive(Debug)]
pub struct DownloadedModelProvider {
    /// Model identifier in cache
    model_id: String,
    /// Path to cached model directory
    model_path: PathBuf,
    /// Parsed model configuration
    model_config: serde_json::Value,
    /// Parsed preprocessor configuration
    preprocessor_config: serde_json::Value,
    /// Selected model variant (fp16, fp32)
    variant: String,
    /// Model cache reference
    cache: crate::cache::ModelCache,
}

#[cfg(feature = "cli")]
impl DownloadedModelProvider {
    /// Create a new provider for a cached model
    ///
    /// # Arguments
    /// * `model_id` - Model identifier in cache (e.g., "imgly--isnet-general-onnx")
    /// * `variant` - Optional variant preference (fp16, fp32)
    /// * `execution_provider` - Optional execution provider for variant optimization
    ///
    /// # Errors
    /// - Model not found in cache
    /// - Invalid or missing configuration files
    /// - JSON parsing errors
    /// - Requested variant not available
    pub fn new(
        model_id: String,
        variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        let cache = crate::cache::ModelCache::new()?;

        // Check if model is cached
        if !cache.is_model_cached(&model_id) {
            return Err(crate::error::BgRemovalError::model(format!(
                "Model '{}' not found in cache. Available models: {:?}",
                model_id,
                cache
                    .scan_cached_models()?
                    .iter()
                    .map(|m| &m.model_id)
                    .collect::<Vec<_>>()
            )));
        }

        let model_path = cache.get_model_path(&model_id);

        // Load configuration files
        let (model_config, preprocessor_config) = Self::load_configurations(&model_path)?;

        // Determine the best variant to use
        let resolved_variant = Self::resolve_variant(&model_path, variant, execution_provider)?;

        Ok(Self {
            model_id,
            model_path,
            model_config,
            preprocessor_config,
            variant: resolved_variant,
            cache,
        })
    }

    /// Load model and preprocessor configuration files
    fn load_configurations(model_path: &Path) -> Result<(serde_json::Value, serde_json::Value)> {
        // Load config.json
        let config_path = model_path.join("config.json");
        let config_content = fs::read_to_string(&config_path).map_err(|e| {
            crate::error::BgRemovalError::file_io_error("read model config.json", &config_path, &e)
        })?;

        let model_config: serde_json::Value =
            serde_json::from_str(&config_content).map_err(|e| {
                crate::error::BgRemovalError::model(format!("Failed to parse config.json: {}", e))
            })?;

        // Load preprocessor_config.json
        let preprocessor_path = model_path.join("preprocessor_config.json");
        let preprocessor_content = fs::read_to_string(&preprocessor_path).map_err(|e| {
            crate::error::BgRemovalError::file_io_error(
                "read preprocessor_config.json",
                &preprocessor_path,
                &e,
            )
        })?;

        let preprocessor_config: serde_json::Value = serde_json::from_str(&preprocessor_content)
            .map_err(|e| {
                crate::error::BgRemovalError::model(format!(
                    "Failed to parse preprocessor_config.json: {}",
                    e
                ))
            })?;

        Ok((model_config, preprocessor_config))
    }

    /// Resolve the best variant to use based on availability and execution provider
    fn resolve_variant(
        model_path: &Path,
        requested_variant: Option<String>,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<String> {
        let onnx_dir = model_path.join("onnx");
        if !onnx_dir.exists() {
            return Err(crate::error::BgRemovalError::model(format!(
                "ONNX directory not found in cached model: {}",
                model_path.display()
            )));
        }

        // Scan for available variants
        let mut available_variants = Vec::new();
        if let Ok(entries) = fs::read_dir(&onnx_dir) {
            for entry in entries.flatten() {
                if let Some(file_name) = entry.file_name().to_str() {
                    if file_name.ends_with(".onnx") {
                        match file_name {
                            "model.onnx" => available_variants.push("fp32".to_string()),
                            "model_fp16.onnx" => available_variants.push("fp16".to_string()),
                            _ => {
                                // Other ONNX files
                                if let Some(variant) = file_name
                                    .strip_prefix("model_")
                                    .and_then(|s| s.strip_suffix(".onnx"))
                                {
                                    available_variants.push(variant.to_string());
                                }
                            },
                        }
                    }
                }
            }
        }

        if available_variants.is_empty() {
            return Err(crate::error::BgRemovalError::model(format!(
                "No ONNX model files found in cached model: {}",
                onnx_dir.display()
            )));
        }

        // If variant explicitly requested, use it if available
        if let Some(variant) = requested_variant {
            if available_variants.contains(&variant) {
                return Ok(variant);
            }
            return Err(crate::error::BgRemovalError::model(format!(
                "Requested variant '{}' not available in cached model. Available: {:?}",
                variant, available_variants
            )));
        }

        // Auto-select based on execution provider preferences
        if let Some(provider) = execution_provider {
            match provider {
                crate::config::ExecutionProvider::CoreMl => {
                    // Prefer FP32 for CoreML
                    if available_variants.contains(&"fp32".to_string()) {
                        return Ok("fp32".to_string());
                    }
                },
                crate::config::ExecutionProvider::Cuda | crate::config::ExecutionProvider::Cpu => {
                    // Prefer FP16 for CPU/CUDA
                    if available_variants.contains(&"fp16".to_string()) {
                        return Ok("fp16".to_string());
                    }
                },
                crate::config::ExecutionProvider::Auto => {
                    // Platform-specific preferences
                    #[cfg(target_os = "macos")]
                    {
                        if available_variants.contains(&"fp32".to_string()) {
                            return Ok("fp32".to_string());
                        }
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        if available_variants.contains(&"fp16".to_string()) {
                            return Ok("fp16".to_string());
                        }
                    }
                },
            }
        }

        // Fallback: prefer fp16, then fp32, then first available
        for preferred in ["fp16", "fp32"] {
            if available_variants.contains(&preferred.to_string()) {
                return Ok(preferred.to_string());
            }
        }

        Ok(available_variants.into_iter().next().unwrap())
    }

    /// Get the path to the ONNX model file for the selected variant
    fn get_model_file_path(&self) -> PathBuf {
        let onnx_dir = self.model_path.join("onnx");
        match self.variant.as_str() {
            "fp16" => onnx_dir.join("model_fp16.onnx"),
            "fp32" => onnx_dir.join("model.onnx"),
            variant => onnx_dir.join(format!("model_{}.onnx", variant)),
        }
    }

    /// Parse image size from preprocessor config
    fn parse_image_size(preprocessor: &serde_json::Value) -> Result<[u32; 2]> {
        let size = preprocessor.get("size").ok_or_else(|| {
            crate::error::BgRemovalError::model("Missing size in preprocessor config")
        })?;

        let height = size
            .get("height")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                crate::error::BgRemovalError::model("Missing or invalid height in size config")
            })?;

        let width = size
            .get("width")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                crate::error::BgRemovalError::model("Missing or invalid width in size config")
            })?;

        let height_u32 = height
            .try_into()
            .map_err(|_| crate::error::BgRemovalError::model("Height value too large for u32"))?;
        let width_u32 = width
            .try_into()
            .map_err(|_| crate::error::BgRemovalError::model("Width value too large for u32"))?;

        Ok([height_u32, width_u32])
    }

    /// Parse image mean from preprocessor config (convert from 0-255 to 0-1 range)
    fn parse_image_mean(preprocessor: &serde_json::Value) -> Result<[f32; 3]> {
        let image_mean = preprocessor
            .get("image_mean")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                crate::error::BgRemovalError::model(
                    "Missing or invalid image_mean in preprocessor config",
                )
            })?;

        if image_mean.len() < 3 {
            return Err(crate::error::BgRemovalError::model(
                "image_mean must have at least 3 values",
            ));
        }

        // Convert from 0-255 range to 0-1 range
        let mean0 = (image_mean[0].as_f64().unwrap_or(128.0) / 255.0) as f32;
        let mean1 = (image_mean[1].as_f64().unwrap_or(128.0) / 255.0) as f32;
        let mean2 = (image_mean[2].as_f64().unwrap_or(128.0) / 255.0) as f32;

        Ok([mean0, mean1, mean2])
    }

    /// Parse image std from preprocessor config (convert from 0-255 to 0-1 range)
    fn parse_image_std(preprocessor: &serde_json::Value) -> Result<[f32; 3]> {
        let image_std = preprocessor
            .get("image_std")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                crate::error::BgRemovalError::model(
                    "Missing or invalid image_std in preprocessor config",
                )
            })?;

        if image_std.len() < 3 {
            return Err(crate::error::BgRemovalError::model(
                "image_std must have at least 3 values",
            ));
        }

        // Convert from 0-255 range to 0-1 range
        let std0 = (image_std[0].as_f64().unwrap_or(255.0) / 255.0) as f32;
        let std1 = (image_std[1].as_f64().unwrap_or(255.0) / 255.0) as f32;
        let std2 = (image_std[2].as_f64().unwrap_or(255.0) / 255.0) as f32;

        Ok([std0, std1, std2])
    }

    /// Get the model cache reference
    #[must_use]
    pub fn cache(&self) -> &crate::cache::ModelCache {
        &self.cache
    }

    /// Get the model ID
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the selected variant
    #[must_use]
    pub fn variant(&self) -> &str {
        &self.variant
    }
}

#[cfg(feature = "cli")]
impl ModelProvider for DownloadedModelProvider {
    fn load_model_data(&self) -> Result<Vec<u8>> {
        let model_file_path = self.get_model_file_path();

        if !model_file_path.exists() {
            return Err(crate::error::BgRemovalError::model(format!(
                "Model file not found: {}. Expected variant: {}",
                model_file_path.display(),
                self.variant
            )));
        }

        fs::read(&model_file_path).map_err(|e| {
            crate::error::BgRemovalError::file_io_error(
                "read cached model file",
                &model_file_path,
                &e,
            )
        })
    }

    fn get_model_info(&self) -> Result<ModelInfo> {
        let model_data = self.load_model_data()?;

        // Extract model type and name from config
        let model_type = self
            .model_config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Get image dimensions from preprocessor config
        let target_size = Self::parse_image_size(&self.preprocessor_config)?;
        let height = target_size[0] as usize;
        let width = target_size[1] as usize;

        Ok(ModelInfo {
            name: format!("{}-{}", model_type, self.variant),
            precision: self.variant.clone(),
            size_bytes: model_data.len(),
            input_shape: (1, 3, height, width),  // NCHW format
            output_shape: (1, 1, height, width), // Single channel mask output
        })
    }

    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        let target_size = Self::parse_image_size(&self.preprocessor_config)?;
        let normalization_mean = Self::parse_image_mean(&self.preprocessor_config)?;
        let normalization_std = Self::parse_image_std(&self.preprocessor_config)?;

        Ok(PreprocessingConfig {
            target_size,
            normalization_mean,
            normalization_std,
        })
    }

    fn get_input_name(&self) -> Result<String> {
        // For downloaded models, use generic input name since we use positional inputs
        Ok("input".to_string())
    }

    fn get_output_name(&self) -> Result<String> {
        // For downloaded models, use generic output name since we use positional outputs
        Ok("output".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_embedded_model_registry() {
        // Test that empty registry works properly
        let available = EmbeddedModelRegistry::list_available();
        assert!(
            available.is_empty(),
            "No embedded models should be available"
        );

        // Test that get_model returns None for any query
        assert!(EmbeddedModelRegistry::get_model("any-model").is_none());
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
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: Some("fp16".to_string()),
        };

        assert!(matches!(spec.source, ModelSource::Downloaded(_)));
        assert_eq!(spec.variant, Some("fp16".to_string()));
    }
}
