//! Model management and embedding system

use crate::error::Result;
use std::fs;
use std::path::{Path, PathBuf};

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
            let available = EmbeddedModelRegistry::list_available();
            let suggestions: Vec<&str> = if available.is_empty() {
                vec!["rebuild with --features embed-isnet or embed-birefnet"]
            } else {
                vec![
                    "check available models with --list-models",
                    "use external model with --model /path/to/model",
                ]
            };
            return Err(crate::error::BgRemovalError::model_error_with_context(
                "load embedded model",
                Path::new(&model_name),
                &format!("model '{model_name}' not found. Available: {available:?}"),
                &suggestions,
            ));
        }

        Ok(Self { model_name })
    }

    /// List all available embedded models
    #[must_use]
    pub fn list_available() -> &'static [&'static str] {
        EmbeddedModelRegistry::list_available()
    }

    fn get_model_data(&self) -> Result<EmbeddedModelData> {
        EmbeddedModelRegistry::get_model(&self.model_name).ok_or_else(|| {
            crate::error::BgRemovalError::invalid_config(format!(
                "Embedded model '{model_name}' not found",
                model_name = self.model_name
            ))
        })
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
                model_data.input_shape[3],
            ),
            output_shape: (
                model_data.output_shape[0],
                model_data.output_shape[1],
                model_data.output_shape[2],
                model_data.output_shape[3],
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
#[must_use]
pub fn get_available_embedded_models() -> Vec<String> {
    EmbeddedModelProvider::list_available()
        .iter()
        .map(ToString::to_string)
        .collect()
}

/// Extract precision from model name (e.g., "isnet-fp16" -> "fp16")
#[must_use]
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
    /// - Requested embedded model not found in registry
    /// - Model path does not exist or is not a directory
    /// - Missing or invalid model configuration files
    /// - JSON parsing errors when reading configuration
    /// - Invalid variant configuration or execution provider settings
    pub fn from_spec_with_provider(
        spec: &ModelSpec,
        execution_provider: Option<&crate::config::ExecutionProvider>,
    ) -> Result<Self> {
        match &spec.source {
            ModelSource::Embedded(model_name) => Self::with_embedded_model(model_name.clone()),
            ModelSource::External(model_path) => Self::with_external_model_and_provider(
                model_path,
                spec.variant.clone(),
                execution_provider,
            ),
        }
    }

    /// Create a new model manager with specific embedded model
    ///
    /// # Errors
    /// - Requested embedded model not found in registry
    /// - Model name validation fails
    /// - Internal model provider creation errors
    pub fn with_embedded_model(model_name: String) -> Result<Self> {
        let provider = EmbeddedModelProvider::new(model_name)?;
        Ok(Self {
            provider: Box::new(provider),
        })
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

    /// Create model manager with first available embedded model (legacy compatibility)
    ///
    /// # Errors
    /// - No embedded models available (build without embedding features)
    /// - Model provider creation fails for the first available model
    /// - Internal registry access errors
    pub fn with_embedded() -> Result<Self> {
        let available = EmbeddedModelProvider::list_available();
        if available.is_empty() {
            return Err(crate::error::BgRemovalError::invalid_config(
                "No embedded models available. Build with embed-* features or use external model.",
            ));
        }

        Self::with_embedded_model(
            (*available.first().ok_or_else(|| {
                crate::error::BgRemovalError::invalid_config("No embedded models available")
            })?)
            .to_string(),
        )
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
            let provider = EmbeddedModelProvider::new((*model_name).to_string()).unwrap();
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
