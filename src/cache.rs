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
    /// Whether the model has HuggingFace format files
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
        #[cfg(feature = "cli")]
        {
            Ok(dirs::cache_dir()
                .ok_or_else(|| {
                    BgRemovalError::invalid_config(
                        "Failed to determine cache directory. Set IMGLY_BGREMOVE_CACHE_DIR environment variable.".to_string()
                    )
                })?
                .join("imgly-bgremove")
                .join("models"))
        }

        #[cfg(not(feature = "cli"))]
        {
            Err(BgRemovalError::invalid_config(
                "Cache functionality requires CLI features to be enabled".to_string(),
            ))
        }
    }

    /// Generate a model ID from a URL
    ///
    /// Converts URLs like "https://huggingface.co/imgly/isnet-general-onnx"
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
            url[prefix.len()..].replace('/', "--")
        } else {
            // For non-HuggingFace URLs, use a hash-based approach
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(url.as_bytes());
            format!("url-{:x}", hasher.finalize())[..16].to_string()
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
        model_path.exists() && self.validate_model_directory(&model_path)
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
                if let Some(model_info) = self.analyze_model_directory(&path)? {
                    models.push(model_info);
                }
            }
        }

        // Sort by model ID for consistent output
        models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        Ok(models)
    }

    /// Validate that a model directory contains required files
    fn validate_model_directory(&self, model_path: &Path) -> bool {
        let config_path = model_path.join("config.json");
        let preprocessor_path = model_path.join("preprocessor_config.json");
        let onnx_dir = model_path.join("onnx");

        // Must have HuggingFace format files
        config_path.exists() && preprocessor_path.exists() && onnx_dir.exists()
    }

    /// Analyze a model directory and extract information
    fn analyze_model_directory(&self, model_path: &Path) -> Result<Option<CachedModelInfo>> {
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
        if !self.validate_model_directory(model_path) {
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
                    if file_name.ends_with(".onnx") {
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
        let size_bytes = self.calculate_directory_size(model_path).unwrap_or(0);

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
    fn calculate_directory_size(&self, dir_path: &Path) -> Result<u64> {
        let mut total_size = 0;

        fn visit_dir(dir: &Path, total: &mut u64) -> std::io::Result<()> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dir(&path, total)?;
                } else {
                    *total += entry.metadata()?.len();
                }
            }
            Ok(())
        }

        visit_dir(dir_path, &mut total_size)
            .map_err(|e| BgRemovalError::file_io_error("calculate directory size", dir_path, &e))?;

        Ok(total_size)
    }

    /// Get the default model ID (ISNet General)
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

                if !self.validate_model_directory(&path) {
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
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
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

    #[cfg(feature = "cli")]
    #[test]
    fn test_cache_creation() {
        // This test only runs when CLI features are enabled
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
}
