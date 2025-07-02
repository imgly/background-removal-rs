//! Model cache management for downloaded models
//!
//! This module provides functionality for managing cached models in an XDG-compliant
//! directory structure. It handles cache directory creation, model scanning for the
//! --list-models functionality, and model ID generation from URLs.

use crate::error::{BgRemovalError, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Information about a cached model
#[derive(Debug, Clone)]
pub struct CachedModelInfo {
    /// Model identifier (derived from URL)
    pub model_id: String,
    /// Path to the cached model directory
    pub path: PathBuf,
    /// Whether the model has `HuggingFace` format files
    pub has_config: bool,
    /// Whether the model has preprocessor config
    pub has_preprocessor: bool,
    /// Available ONNX model variants (fp16, fp32)
    pub variants: Vec<String>,
    /// Estimated size of the model directory in bytes
    pub size_bytes: u64,
}

/// Model cache manager
#[derive(Debug)]
pub struct ModelCache {
    cache_dir: PathBuf,
}

impl ModelCache {
    /// Create a new model cache manager
    ///
    /// Uses XDG Base Directory specification for cache location:
    /// - Linux/macOS: `~/.cache/imgly-bgremove/models/`
    /// - Windows: `%LOCALAPPDATA%/imgly-bgremove/models/`
    ///
    /// # Errors
    /// - Failed to determine cache directory
    /// - Failed to create cache directory
    /// - Insufficient permissions to access cache directory
    pub fn new() -> Result<Self> {
        let cache_dir = Self::get_cache_dir()?;

        // Ensure cache directory exists
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).map_err(|e| {
                BgRemovalError::file_io_error("create cache directory", &cache_dir, &e)
            })?;
        }

        Ok(Self { cache_dir })
    }

    /// Get the XDG-compliant cache directory path
    ///
    /// # Errors
    /// - Failed to determine user cache directory
    /// - Invalid cache directory path
    fn get_cache_dir() -> Result<PathBuf> {
        // Try environment variable override first
        if let Ok(cache_override) = std::env::var("IMGLY_BGREMOVE_CACHE_DIR") {
            return Ok(PathBuf::from(cache_override).join("models"));
        }

        // Use XDG-compliant cache directory
        Ok(dirs::cache_dir()
            .ok_or_else(|| {
                BgRemovalError::invalid_config(
                    "Failed to determine cache directory. Set IMGLY_BGREMOVE_CACHE_DIR environment variable.".to_string()
                )
            })?
            .join("imgly-bgremove")
            .join("models"))
    }

    /// Generate a model ID from a URL
    ///
    /// Converts URLs like "<https://huggingface.co/imgly/isnet-general-onnx>"
    /// to cache-safe identifiers like "imgly--isnet-general-onnx"
    ///
    /// # Examples
    /// ```
    /// use imgly_bgremove::cache::ModelCache;
    ///
    /// let id = ModelCache::url_to_model_id("https://huggingface.co/imgly/isnet-general-onnx");
    /// assert_eq!(id, "imgly--isnet-general-onnx");
    /// ```
    #[must_use]
    pub fn url_to_model_id(url: &str) -> String {
        // Extract the model path from HuggingFace URLs
        let prefix = "https://huggingface.co/";
        if url.starts_with(prefix) {
            // Replace '/' with '--' to create filesystem-safe identifier
            // Safe string slicing - we already verified the prefix exists with starts_with
            url.get(prefix.len()..).unwrap_or(url).replace('/', "--")
        } else {
            // For non-HuggingFace URLs, use a hash-based approach
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(url.as_bytes());
            let hash_string = format!("url-{:x}", hasher.finalize());
            // Safe string slicing with bounds check
            if hash_string.len() >= 16 {
                hash_string.get(..16).unwrap_or(&hash_string).to_string()
            } else {
                hash_string
            }
        }
    }

    /// Check if a model is cached
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (from URL or direct name)
    ///
    /// # Returns
    /// `true` if the model directory exists and contains valid model files
    #[must_use]
    pub fn is_model_cached(&self, model_id: &str) -> bool {
        let model_path = self.cache_dir.join(model_id);
        model_path.exists() && Self::validate_model_directory(&model_path)
    }

    /// Get the path to a cached model directory
    ///
    /// # Arguments
    /// * `model_id` - Model identifier
    ///
    /// # Returns
    /// Path to the model directory (may not exist)
    #[must_use]
    pub fn get_model_path(&self, model_id: &str) -> PathBuf {
        self.cache_dir.join(model_id)
    }

    /// Scan cache directory and return all available models
    ///
    /// This is used by the --list-models functionality to show cached models.
    ///
    /// # Errors
    /// - Failed to read cache directory
    /// - Failed to analyze model directories
    /// - I/O errors when accessing model files
    pub fn scan_cached_models(&self) -> Result<Vec<CachedModelInfo>> {
        let mut models = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(models); // Empty cache
        }

        let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
            BgRemovalError::file_io_error("read cache directory", &self.cache_dir, &e)
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                BgRemovalError::network_error("Failed to read cache directory entry", e)
            })?;

            let path = entry.path();
            if path.is_dir() {
                if let Some(model_info) = Self::analyze_model_directory(&path)? {
                    models.push(model_info);
                }
            }
        }

        // Sort by model ID for consistent output
        models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        Ok(models)
    }

    /// Validate that a model directory contains required files
    fn validate_model_directory(model_path: &Path) -> bool {
        let config_path = model_path.join("config.json");
        let preprocessor_path = model_path.join("preprocessor_config.json");
        let onnx_dir = model_path.join("onnx");

        // Must have HuggingFace format files
        config_path.exists() && preprocessor_path.exists() && onnx_dir.exists()
    }

    /// Analyze a model directory and extract information
    fn analyze_model_directory(model_path: &Path) -> Result<Option<CachedModelInfo>> {
        let model_id = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                BgRemovalError::invalid_config(format!(
                    "Invalid model directory name: {}",
                    model_path.display()
                ))
            })?
            .to_string();

        // Check if this is a valid model directory
        if !Self::validate_model_directory(model_path) {
            log::debug!("Skipping invalid model directory: {}", model_path.display());
            return Ok(None);
        }

        let config_path = model_path.join("config.json");
        let preprocessor_path = model_path.join("preprocessor_config.json");
        let onnx_dir = model_path.join("onnx");

        // Scan for available ONNX variants
        let mut variants = Vec::new();
        if let Ok(entries) = fs::read_dir(&onnx_dir) {
            for entry in entries.flatten() {
                if let Some(file_name) = entry.file_name().to_str() {
                    if Path::new(file_name)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
                    {
                        match file_name {
                            "model.onnx" => variants.push("fp32".to_string()),
                            "model_fp16.onnx" => variants.push("fp16".to_string()),
                            _ => {
                                // Other ONNX files - extract variant from filename
                                if let Some(variant) = file_name
                                    .strip_prefix("model_")
                                    .and_then(|s| s.strip_suffix(".onnx"))
                                {
                                    variants.push(variant.to_string());
                                }
                            },
                        }
                    }
                }
            }
        }

        // Calculate directory size
        let size_bytes = Self::calculate_directory_size(model_path).unwrap_or(0);

        Ok(Some(CachedModelInfo {
            model_id,
            path: model_path.to_path_buf(),
            has_config: config_path.exists(),
            has_preprocessor: preprocessor_path.exists(),
            variants,
            size_bytes,
        }))
    }

    /// Calculate the total size of a directory
    fn calculate_directory_size(dir_path: &Path) -> Result<u64> {
        let mut total_size = 0;

        Self::visit_dir(dir_path, &mut total_size)
            .map_err(|e| BgRemovalError::file_io_error("calculate directory size", dir_path, &e))?;

        Ok(total_size)
    }

    /// Recursively visit directory and accumulate file sizes
    fn visit_dir(dir: &Path, total: &mut u64) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::visit_dir(&path, total)?;
            } else {
                *total += entry.metadata()?.len();
            }
        }
        Ok(())
    }

    /// Get the default model ID (`ISNet` General)
    #[must_use]
    pub fn get_default_model_id() -> String {
        Self::url_to_model_id("https://huggingface.co/imgly/isnet-general-onnx")
    }

    /// Get the default model URL
    #[must_use]
    pub fn get_default_model_url() -> &'static str {
        "https://huggingface.co/imgly/isnet-general-onnx"
    }

    /// Clean up cache directory by removing invalid model directories
    ///
    /// # Errors
    /// - Failed to access cache directory
    /// - Failed to remove invalid directories
    /// - I/O errors during cleanup operations
    pub fn cleanup_invalid_models(&self) -> Result<Vec<String>> {
        let mut removed_models = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(removed_models);
        }

        let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
            BgRemovalError::file_io_error("read cache directory", &self.cache_dir, &e)
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                BgRemovalError::network_error("Failed to read cache directory entry", e)
            })?;

            let path = entry.path();
            if path.is_dir() {
                let model_id = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown");

                if !Self::validate_model_directory(&path) {
                    log::warn!("Removing invalid model directory: {}", path.display());
                    fs::remove_dir_all(&path).map_err(|e| {
                        BgRemovalError::file_io_error("remove invalid model directory", &path, &e)
                    })?;
                    removed_models.push(model_id.to_string());
                }
            }
        }

        Ok(removed_models)
    }

    /// Clear all cached models
    ///
    /// Removes all model directories from the cache, effectively clearing the entire cache.
    ///
    /// # Returns
    /// Vector of removed model IDs for user feedback
    ///
    /// # Errors
    /// - Failed to access cache directory
    /// - Failed to remove model directories
    /// - I/O errors during removal operations
    pub fn clear_all_models(&self) -> Result<Vec<String>> {
        let mut removed_models = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(removed_models);
        }

        let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
            BgRemovalError::file_io_error("read cache directory", &self.cache_dir, &e)
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                BgRemovalError::network_error("Failed to read cache directory entry", e)
            })?;

            let path = entry.path();
            if path.is_dir() {
                let model_id = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown");

                log::info!("Removing cached model: {}", model_id);
                fs::remove_dir_all(&path).map_err(|e| {
                    BgRemovalError::file_io_error("remove cached model directory", &path, &e)
                })?;
                removed_models.push(model_id.to_string());
            }
        }

        Ok(removed_models)
    }

    /// Clear a specific cached model
    ///
    /// Removes the specified model directory from the cache.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier to remove from cache
    ///
    /// # Returns
    /// `true` if the model was found and removed, `false` if the model was not cached
    ///
    /// # Errors
    /// - Failed to remove model directory
    /// - I/O errors during removal operations
    pub fn clear_specific_model(&self, model_id: &str) -> Result<bool> {
        let model_path = self.get_model_path(model_id);

        if !model_path.exists() {
            return Ok(false);
        }

        log::info!("Removing cached model: {}", model_id);
        fs::remove_dir_all(&model_path).map_err(|e| {
            BgRemovalError::file_io_error("remove specific cached model", &model_path, &e)
        })?;

        Ok(true)
    }

    /// Create a new model cache with a custom cache directory
    ///
    /// # Arguments
    /// * `cache_dir` - Custom cache directory path
    ///
    /// # Errors
    /// - Failed to create cache directory
    /// - Insufficient permissions to access cache directory
    pub fn with_custom_cache_dir(cache_dir: &Path) -> Result<Self> {
        let models_dir = cache_dir.join("models");

        // Ensure cache directory exists
        if !models_dir.exists() {
            fs::create_dir_all(&models_dir).map_err(|e| {
                BgRemovalError::file_io_error("create custom cache directory", &models_dir, &e)
            })?;
        }

        Ok(Self {
            cache_dir: models_dir,
        })
    }

    /// Get the current cache directory path
    #[must_use]
    pub fn get_current_cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new().expect("Failed to create default model cache")
    }
}

/// Format file size in human-readable format
#[must_use]
pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS.get(unit_index).unwrap_or(&"B"))
    } else {
        format!("{:.1} {}", size, UNITS.get(unit_index).unwrap_or(&"B"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_url_to_model_id() {
        // HuggingFace URL
        assert_eq!(
            ModelCache::url_to_model_id("https://huggingface.co/imgly/isnet-general-onnx"),
            "imgly--isnet-general-onnx"
        );

        // Another HuggingFace URL
        assert_eq!(
            ModelCache::url_to_model_id("https://huggingface.co/ZhengPeng7/BiRefNet"),
            "ZhengPeng7--BiRefNet"
        );

        // Non-HuggingFace URL should create hash-based ID
        let id = ModelCache::url_to_model_id("https://example.com/model.onnx");
        assert!(id.starts_with("url-"));
        assert_eq!(id.len(), 16); // "url-" + 12 hex chars
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_default_model_constants() {
        assert_eq!(
            ModelCache::get_default_model_id(),
            "imgly--isnet-general-onnx"
        );
        assert_eq!(
            ModelCache::get_default_model_url(),
            "https://huggingface.co/imgly/isnet-general-onnx"
        );
    }

    #[test]
    fn test_cache_creation() {
        let _cache = ModelCache::new().expect("Should create cache successfully");
    }

    #[test]
    fn test_model_cache_with_temp_dir() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("models");

        // Test cache directory creation logic
        assert!(!cache_dir.exists());
        fs::create_dir_all(&cache_dir).unwrap();
        assert!(cache_dir.exists());
    }

    #[test]
    fn test_custom_cache_dir() {
        let temp_dir = TempDir::new().unwrap();
        let custom_cache = temp_dir.path().join("custom_cache");

        // Create cache with custom directory
        let cache = ModelCache::with_custom_cache_dir(custom_cache.as_path()).unwrap();

        // Verify the cache directory was created
        assert!(custom_cache.join("models").exists());
        assert_eq!(cache.get_current_cache_dir(), &custom_cache.join("models"));
    }

    #[test]
    fn test_clear_specific_model() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        let model_id = "test-model";
        let model_path = cache.get_model_path(model_id);

        // Create a fake model directory
        fs::create_dir_all(&model_path).unwrap();
        assert!(model_path.exists());

        // Clear the specific model
        let result = cache.clear_specific_model(model_id).unwrap();
        assert!(result); // Should return true for successful removal
        assert!(!model_path.exists()); // Directory should be removed

        // Try to clear non-existent model
        let result = cache.clear_specific_model("non-existent-model").unwrap();
        assert!(!result); // Should return false for non-existent model
    }

    #[test]
    fn test_clear_all_models() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        // Create some fake model directories
        let model_ids = vec!["model1", "model2", "model3"];
        for model_id in &model_ids {
            let model_path = cache.get_model_path(model_id);
            fs::create_dir_all(&model_path).unwrap();
            assert!(model_path.exists());
        }

        // Clear all models
        let removed = cache.clear_all_models().unwrap();
        assert_eq!(removed.len(), 3);
        assert!(removed.iter().all(|id| model_ids.contains(&id.as_str())));

        // Verify all model directories are removed
        for model_id in &model_ids {
            let model_path = cache.get_model_path(model_id);
            assert!(!model_path.exists());
        }

        // Clear empty cache (should succeed with no removed models)
        let removed = cache.clear_all_models().unwrap();
        assert!(removed.is_empty());
    }

    #[test]
    fn test_url_to_model_id_comprehensive() {
        // Test various HuggingFace URL formats
        assert_eq!(
            ModelCache::url_to_model_id("https://huggingface.co/user/simple-model"),
            "user--simple-model"
        );
        assert_eq!(
            ModelCache::url_to_model_id("https://huggingface.co/org/model_with_underscores"),
            "org--model_with_underscores"
        );
        assert_eq!(
            ModelCache::url_to_model_id("https://huggingface.co/complex/model-name.v2"),
            "complex--model-name.v2"
        );

        // Test URL with trailing slash
        assert_eq!(
            ModelCache::url_to_model_id("https://huggingface.co/user/model/"),
            "user--model--"
        );

        // Test URL with additional path components
        assert_eq!(
            ModelCache::url_to_model_id("https://huggingface.co/user/model/tree/main"),
            "user--model--tree--main"
        );

        // Test non-HuggingFace URLs (should create hash-based IDs)
        let github_id = ModelCache::url_to_model_id("https://github.com/user/repo");
        assert!(github_id.starts_with("url-"));
        assert_eq!(github_id.len(), 16);

        let custom_id = ModelCache::url_to_model_id("https://example.com/model.onnx");
        assert!(custom_id.starts_with("url-"));
        assert_eq!(custom_id.len(), 16);

        // Test edge case: very short non-HuggingFace URL
        let short_id = ModelCache::url_to_model_id("a");
        assert!(short_id.starts_with("url-"));
        assert!(short_id.len() <= 16); // Should be truncated if necessary
    }

    #[test]
    fn test_is_model_cached() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        let model_id = "test-model";

        // Model not cached initially
        assert!(!cache.is_model_cached(model_id));

        // Create incomplete model directory (missing required files)
        let model_path = cache.get_model_path(model_id);
        fs::create_dir_all(&model_path).unwrap();
        assert!(!cache.is_model_cached(model_id)); // Still not valid

        // Create required structure for valid model
        fs::write(model_path.join("config.json"), "{}").unwrap();
        fs::write(model_path.join("preprocessor_config.json"), "{}").unwrap();
        fs::create_dir_all(model_path.join("onnx")).unwrap();

        // Now it should be considered cached
        assert!(cache.is_model_cached(model_id));
    }

    #[test]
    fn test_validate_model_directory() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test-model");
        fs::create_dir_all(&model_path).unwrap();

        // Empty directory should not be valid
        assert!(!ModelCache::validate_model_directory(&model_path));

        // Create config.json only
        fs::write(model_path.join("config.json"), "{}").unwrap();
        assert!(!ModelCache::validate_model_directory(&model_path));

        // Add preprocessor_config.json
        fs::write(model_path.join("preprocessor_config.json"), "{}").unwrap();
        assert!(!ModelCache::validate_model_directory(&model_path));

        // Add onnx directory - now it should be valid
        fs::create_dir_all(model_path.join("onnx")).unwrap();
        assert!(ModelCache::validate_model_directory(&model_path));
    }

    #[test]
    fn test_scan_cached_models_empty_cache() {
        let temp_dir = TempDir::new().unwrap();
        let non_existent_cache = temp_dir.path().join("non-existent");
        let cache = ModelCache::with_custom_cache_dir(&non_existent_cache).unwrap();

        // Remove the directory to test non-existent cache
        fs::remove_dir_all(cache.get_current_cache_dir()).unwrap();

        let models = cache.scan_cached_models().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_scan_cached_models_with_valid_models() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        // Create valid model structure
        let model_id = "test-model";
        let model_path = cache.get_model_path(model_id);
        fs::create_dir_all(&model_path).unwrap();
        fs::write(model_path.join("config.json"), "{}").unwrap();
        fs::write(model_path.join("preprocessor_config.json"), "{}").unwrap();

        let onnx_dir = model_path.join("onnx");
        fs::create_dir_all(&onnx_dir).unwrap();
        fs::write(onnx_dir.join("model.onnx"), "fake onnx data").unwrap();
        fs::write(onnx_dir.join("model_fp16.onnx"), "fake fp16 data").unwrap();

        let models = cache.scan_cached_models().unwrap();
        assert_eq!(models.len(), 1);

        let model_info = &models[0];
        assert_eq!(model_info.model_id, model_id);
        assert!(model_info.has_config);
        assert!(model_info.has_preprocessor);
        assert!(model_info.variants.contains(&"fp32".to_string()));
        assert!(model_info.variants.contains(&"fp16".to_string()));
        assert!(model_info.size_bytes > 0);
    }

    #[test]
    fn test_scan_cached_models_with_invalid_models() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        // Create invalid model directory (missing required files)
        let invalid_model_path = cache.get_model_path("invalid-model");
        fs::create_dir_all(&invalid_model_path).unwrap();
        fs::write(invalid_model_path.join("some-file.txt"), "not a model").unwrap();

        // Create valid model
        let valid_model_path = cache.get_model_path("valid-model");
        fs::create_dir_all(&valid_model_path).unwrap();
        fs::write(valid_model_path.join("config.json"), "{}").unwrap();
        fs::write(valid_model_path.join("preprocessor_config.json"), "{}").unwrap();
        fs::create_dir_all(valid_model_path.join("onnx")).unwrap();

        let models = cache.scan_cached_models().unwrap();
        assert_eq!(models.len(), 1); // Only valid model should be returned
        assert_eq!(models[0].model_id, "valid-model");
    }

    #[test]
    fn test_analyze_model_directory_variants() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test-model");
        fs::create_dir_all(&model_path).unwrap();

        // Create required files
        fs::write(model_path.join("config.json"), "{}").unwrap();
        fs::write(model_path.join("preprocessor_config.json"), "{}").unwrap();

        let onnx_dir = model_path.join("onnx");
        fs::create_dir_all(&onnx_dir).unwrap();

        // Create different ONNX variants
        fs::write(onnx_dir.join("model.onnx"), "fp32 data").unwrap();
        fs::write(onnx_dir.join("model_fp16.onnx"), "fp16 data").unwrap();
        fs::write(onnx_dir.join("model_quantized.onnx"), "quantized data").unwrap();
        fs::write(onnx_dir.join("model_optimized.onnx"), "optimized data").unwrap();
        fs::write(onnx_dir.join("not_onnx.txt"), "not an onnx file").unwrap();

        let model_info = ModelCache::analyze_model_directory(&model_path)
            .unwrap()
            .unwrap();

        // Should detect standard variants
        assert!(model_info.variants.contains(&"fp32".to_string()));
        assert!(model_info.variants.contains(&"fp16".to_string()));
        assert!(model_info.variants.contains(&"quantized".to_string()));
        assert!(model_info.variants.contains(&"optimized".to_string()));

        // Should not include non-ONNX files
        assert_eq!(model_info.variants.len(), 4);
    }

    #[test]
    fn test_calculate_directory_size() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("size-test");
        fs::create_dir_all(&test_dir).unwrap();

        // Create files with known sizes
        fs::write(test_dir.join("file1.txt"), "12345").unwrap(); // 5 bytes
        fs::write(test_dir.join("file2.txt"), "abcdefghij").unwrap(); // 10 bytes

        // Create subdirectory with more files
        let sub_dir = test_dir.join("subdir");
        fs::create_dir_all(&sub_dir).unwrap();
        fs::write(sub_dir.join("file3.txt"), "xyz").unwrap(); // 3 bytes

        let total_size = ModelCache::calculate_directory_size(&test_dir).unwrap();
        assert_eq!(total_size, 18); // 5 + 10 + 3 bytes
    }

    #[test]
    fn test_cleanup_invalid_models() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        // Create valid model
        let valid_model_path = cache.get_model_path("valid-model");
        fs::create_dir_all(&valid_model_path).unwrap();
        fs::write(valid_model_path.join("config.json"), "{}").unwrap();
        fs::write(valid_model_path.join("preprocessor_config.json"), "{}").unwrap();
        fs::create_dir_all(valid_model_path.join("onnx")).unwrap();

        // Create invalid model
        let invalid_model_path = cache.get_model_path("invalid-model");
        fs::create_dir_all(&invalid_model_path).unwrap();
        fs::write(invalid_model_path.join("some-file.txt"), "not a model").unwrap();

        // Create another invalid model
        let invalid_model2_path = cache.get_model_path("invalid-model2");
        fs::create_dir_all(&invalid_model2_path).unwrap();

        // Cleanup invalid models
        let removed = cache.cleanup_invalid_models().unwrap();

        // Should remove both invalid models
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&"invalid-model".to_string()));
        assert!(removed.contains(&"invalid-model2".to_string()));

        // Valid model should still exist
        assert!(valid_model_path.exists());
        assert!(!invalid_model_path.exists());
        assert!(!invalid_model2_path.exists());
    }

    #[test]
    fn test_cleanup_invalid_models_empty_cache() {
        let temp_dir = TempDir::new().unwrap();
        let non_existent_cache = temp_dir.path().join("non-existent");
        let cache = ModelCache::with_custom_cache_dir(&non_existent_cache).unwrap();

        // Remove the directory to test non-existent cache
        fs::remove_dir_all(cache.get_current_cache_dir()).unwrap();

        let removed = cache.cleanup_invalid_models().unwrap();
        assert!(removed.is_empty());
    }

    #[test]
    fn test_cache_with_environment_variable() {
        use std::env;

        let temp_dir = TempDir::new().unwrap();
        let custom_cache_path = temp_dir.path().join("custom-env-cache");

        // Set environment variable
        env::set_var("IMGLY_BGREMOVE_CACHE_DIR", &custom_cache_path);

        // Create cache (should use environment variable)
        let cache = ModelCache::new().unwrap();

        // Should use custom path from environment
        let expected_path = custom_cache_path.join("models");
        assert_eq!(cache.get_current_cache_dir(), &expected_path);
        assert!(expected_path.exists());

        // Clean up environment variable
        env::remove_var("IMGLY_BGREMOVE_CACHE_DIR");
    }

    #[test]
    fn test_get_model_path() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        let model_id = "test-model-123";
        let model_path = cache.get_model_path(model_id);

        assert_eq!(model_path, cache.get_current_cache_dir().join(model_id));

        // Path should not exist initially
        assert!(!model_path.exists());
    }

    #[test]
    fn test_cached_model_info_structure() {
        let temp_dir = TempDir::new().unwrap();

        let info = CachedModelInfo {
            model_id: "test-model".to_string(),
            path: temp_dir.path().join("test-model"),
            has_config: true,
            has_preprocessor: false,
            variants: vec!["fp32".to_string(), "fp16".to_string()],
            size_bytes: 1024,
        };

        assert_eq!(info.model_id, "test-model");
        assert!(info.has_config);
        assert!(!info.has_preprocessor);
        assert_eq!(info.variants.len(), 2);
        assert_eq!(info.size_bytes, 1024);

        // Test clone and debug
        let cloned = info.clone();
        assert_eq!(info.model_id, cloned.model_id);

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("test-model"));
    }

    #[test]
    fn test_model_cache_debug() {
        let cache = ModelCache::new().unwrap();
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("ModelCache"));
    }

    #[test]
    fn test_model_cache_default() {
        let cache = ModelCache::default();
        assert!(cache.get_current_cache_dir().exists());
    }

    #[test]
    fn test_format_size_comprehensive() {
        // Test zero bytes
        assert_eq!(format_size(0), "0 B");

        // Test bytes
        assert_eq!(format_size(1), "1 B");
        assert_eq!(format_size(999), "999 B");

        // Test kilobytes
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(2048), "2.0 KB");

        // Test megabytes
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1536), "1.5 MB");

        // Test gigabytes
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(1024_u64 * 1024 * 1536), "1.5 GB");

        // Test terabytes
        assert_eq!(format_size(1024_u64 * 1024 * 1024 * 1024), "1.0 TB");
        assert_eq!(format_size(1024_u64 * 1024 * 1024 * 1536), "1.5 TB");

        // Test very large values (beyond TB should stay in TB)
        assert_eq!(
            format_size(1024_u64 * 1024 * 1024 * 1024 * 1024),
            "1024.0 TB"
        );
    }

    #[test]
    fn test_scan_models_sorting() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        // Create models in non-alphabetical order
        let model_ids = vec!["zebra-model", "alpha-model", "beta-model"];
        for model_id in &model_ids {
            let model_path = cache.get_model_path(model_id);
            fs::create_dir_all(&model_path).unwrap();
            fs::write(model_path.join("config.json"), "{}").unwrap();
            fs::write(model_path.join("preprocessor_config.json"), "{}").unwrap();
            fs::create_dir_all(model_path.join("onnx")).unwrap();
        }

        let models = cache.scan_cached_models().unwrap();
        assert_eq!(models.len(), 3);

        // Should be sorted alphabetically
        assert_eq!(models[0].model_id, "alpha-model");
        assert_eq!(models[1].model_id, "beta-model");
        assert_eq!(models[2].model_id, "zebra-model");
    }

    #[test]
    fn test_scan_models_with_files_and_directories() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_custom_cache_dir(temp_dir.path()).unwrap();

        // Create a valid model directory
        let model_path = cache.get_model_path("valid-model");
        fs::create_dir_all(&model_path).unwrap();
        fs::write(model_path.join("config.json"), "{}").unwrap();
        fs::write(model_path.join("preprocessor_config.json"), "{}").unwrap();
        fs::create_dir_all(model_path.join("onnx")).unwrap();

        // Create a regular file in the cache directory (should be ignored)
        fs::write(cache.get_current_cache_dir().join("readme.txt"), "info").unwrap();

        let models = cache.scan_cached_models().unwrap();
        assert_eq!(models.len(), 1); // Only the valid model directory
        assert_eq!(models[0].model_id, "valid-model");
    }

    #[test]
    fn test_visit_dir_nested_structure() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("nested-test");
        fs::create_dir_all(&test_dir).unwrap();

        // Create nested directory structure
        fs::write(test_dir.join("root.txt"), "12345").unwrap(); // 5 bytes

        let level1 = test_dir.join("level1");
        fs::create_dir_all(&level1).unwrap();
        fs::write(level1.join("file1.txt"), "abcde").unwrap(); // 5 bytes

        let level2 = level1.join("level2");
        fs::create_dir_all(&level2).unwrap();
        fs::write(level2.join("file2.txt"), "xyz").unwrap(); // 3 bytes

        let mut total_size = 0;
        ModelCache::visit_dir(&test_dir, &mut total_size).unwrap();
        assert_eq!(total_size, 13); // 5 + 5 + 3 bytes
    }
}
