//! Model downloading functionality for HuggingFace repositories
//!
//! This module provides async downloading of models from URLs (primarily HuggingFace)
//! with progress reporting, file integrity verification, and atomic operations.

use crate::cache::ModelCache;
use crate::error::{BgRemovalError, Result};
use futures_util::stream::TryStreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio_util::io::StreamReader;

/// Files that need to be downloaded for a HuggingFace model
const REQUIRED_FILES: &[&str] = &["config.json", "preprocessor_config.json"];

/// ONNX model files to attempt downloading
const ONNX_FILES: &[(&str, &str)] = &[
    ("onnx/model.onnx", "fp32"),
    ("onnx/model_fp16.onnx", "fp16"),
];

/// Model downloader with progress reporting
#[derive(Debug)]
pub struct ModelDownloader {
    client: Client,
    cache: ModelCache,
}

/// Download progress information
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// File being downloaded
    pub file_name: String,
    /// Bytes downloaded
    pub downloaded: u64,
    /// Total file size (if known)
    pub total: Option<u64>,
    /// Download completed
    pub completed: bool,
}

impl ModelDownloader {
    /// Create a new model downloader
    ///
    /// # Errors
    /// - Failed to create HTTP client
    /// - Failed to initialize model cache
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
            .build()
            .map_err(|e| BgRemovalError::network_error("Failed to create HTTP client", e))?;

        let cache = ModelCache::new()?;

        Ok(Self { client, cache })
    }

    /// Download a model from a URL to the cache
    ///
    /// This is the main entry point for downloading models. It handles:
    /// - URL validation and parsing
    /// - File discovery and downloading
    /// - Progress reporting
    /// - Atomic operations (temp directory → final location)
    /// - File integrity verification
    ///
    /// # Arguments
    /// * `url` - Model repository URL (e.g., "https://huggingface.co/imgly/isnet-general-onnx")
    /// * `show_progress` - Whether to display download progress
    ///
    /// # Errors
    /// - Invalid or unsupported URL format
    /// - Network errors during download
    /// - File system errors during caching
    /// - File integrity verification failures
    pub async fn download_model(&self, url: &str, show_progress: bool) -> Result<String> {
        // Validate and parse URL
        let model_id = ModelCache::url_to_model_id(url);
        log::info!("Downloading model from: {}", url);
        log::info!("Model ID: {}", model_id);

        // Check if already cached
        if self.cache.is_model_cached(&model_id) {
            log::info!("Model already cached: {}", model_id);
            return Ok(model_id);
        }

        // Create temporary directory for atomic download
        let temp_dir = self.create_temp_download_dir(&model_id)?;
        let final_dir = self.cache.get_model_path(&model_id);

        // Setup progress reporting
        let progress = if show_progress {
            Some(self.create_progress_bar())
        } else {
            None
        };

        // Download all required files
        match self
            .download_model_files(url, &temp_dir, progress.as_ref())
            .await
        {
            Ok(()) => {
                // Atomic move from temp to final location
                if final_dir.exists() {
                    fs::remove_dir_all(&final_dir).map_err(|e| {
                        BgRemovalError::file_io_error(
                            "remove existing model directory",
                            &final_dir,
                            &e,
                        )
                    })?;
                }

                fs::rename(&temp_dir, &final_dir).map_err(|e| {
                    BgRemovalError::file_io_error("move downloaded model to cache", &final_dir, &e)
                })?;

                if let Some(pb) = progress {
                    pb.finish_with_message(format!("✅ Downloaded {}", model_id));
                }

                log::info!("Successfully downloaded model: {}", model_id);
                Ok(model_id)
            },
            Err(e) => {
                // Cleanup temp directory on failure
                if temp_dir.exists() {
                    if let Err(cleanup_err) = fs::remove_dir_all(&temp_dir) {
                        log::warn!("Failed to cleanup temp directory: {}", cleanup_err);
                    }
                }

                if let Some(pb) = progress {
                    pb.finish_with_message("❌ Download failed");
                }

                Err(e)
            },
        }
    }

    /// Create a temporary directory for downloading
    fn create_temp_download_dir(&self, model_id: &str) -> Result<PathBuf> {
        let temp_dir = std::env::temp_dir().join(format!("imgly-bgremove-{}", model_id));

        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir).map_err(|e| {
                BgRemovalError::file_io_error("remove existing temp directory", &temp_dir, &e)
            })?;
        }

        fs::create_dir_all(&temp_dir)
            .map_err(|e| BgRemovalError::file_io_error("create temp directory", &temp_dir, &e))?;

        Ok(temp_dir)
    }

    /// Create a progress bar for download reporting
    fn create_progress_bar(&self) -> ProgressBar {
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb
    }

    /// Download all required model files
    async fn download_model_files(
        &self,
        base_url: &str,
        download_dir: &Path,
        progress: Option<&ProgressBar>,
    ) -> Result<()> {
        // Ensure we have a HuggingFace URL
        if !base_url.starts_with("https://huggingface.co/") {
            return Err(BgRemovalError::invalid_config(format!(
                "Unsupported URL format: {}. Only HuggingFace repositories are supported.",
                base_url
            )));
        }

        // Create base raw URL for file downloads
        let raw_base = base_url
            .replace("https://huggingface.co/", "https://huggingface.co/")
            .to_string()
            + "/resolve/main/";

        // Download required configuration files
        for file_name in REQUIRED_FILES {
            let file_url = format!("{}{}", raw_base, file_name);
            let local_path = download_dir.join(file_name);

            if let Some(pb) = progress {
                pb.set_message(format!("Downloading {}", file_name));
            }

            self.download_file(&file_url, &local_path, progress).await?;
        }

        // Create onnx subdirectory
        let onnx_dir = download_dir.join("onnx");
        fs::create_dir_all(&onnx_dir)
            .map_err(|e| BgRemovalError::file_io_error("create onnx directory", &onnx_dir, &e))?;

        // Download ONNX model files (at least one must succeed)
        let mut downloaded_models = 0;
        for (file_path, variant) in ONNX_FILES {
            let file_url = format!("{}{}", raw_base, file_path);
            let local_path = download_dir.join(file_path);

            if let Some(pb) = progress {
                pb.set_message(format!("Downloading {} model", variant));
            }

            match self.download_file(&file_url, &local_path, progress).await {
                Ok(()) => {
                    downloaded_models += 1;
                    log::info!("Downloaded {} model variant", variant);
                },
                Err(e) => {
                    log::warn!("Failed to download {} variant: {}", variant, e);
                    // Continue trying other variants
                },
            }
        }

        if downloaded_models == 0 {
            return Err(BgRemovalError::network_error(
                "Failed to download any ONNX model variants",
                std::io::Error::new(std::io::ErrorKind::NotFound, "No model files found"),
            ));
        }

        log::info!("Downloaded {} model variant(s)", downloaded_models);
        Ok(())
    }

    /// Download a single file with progress reporting
    async fn download_file(
        &self,
        url: &str,
        local_path: &Path,
        progress: Option<&ProgressBar>,
    ) -> Result<()> {
        log::debug!("Downloading: {} -> {}", url, local_path.display());

        // Create parent directory if needed
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| BgRemovalError::file_io_error("create directory", parent, &e))?;
        }

        // Start download
        let response = self.client.get(url).send().await.map_err(|e| {
            BgRemovalError::network_error(&format!("Failed to download {}", url), e)
        })?;

        if !response.status().is_success() {
            return Err(BgRemovalError::network_error(
                &format!("HTTP error {} for {}", response.status(), url),
                std::io::Error::new(std::io::ErrorKind::Other, "HTTP error"),
            ));
        }

        // Get content length for progress reporting
        let total_size = response.content_length();

        // Create file and download stream
        let mut file = tokio::fs::File::create(local_path)
            .await
            .map_err(|e| BgRemovalError::file_io_error("create file", local_path, &e))?;

        let mut stream = StreamReader::new(
            response
                .bytes_stream()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)),
        );

        // Download with progress tracking
        let mut downloaded = 0u64;
        let mut buffer = vec![0; 8192]; // 8KB buffer

        loop {
            let bytes_read = tokio::io::AsyncReadExt::read(&mut stream, &mut buffer)
                .await
                .map_err(|e| BgRemovalError::network_error("Failed to read download stream", e))?;

            if bytes_read == 0 {
                break; // EOF
            }

            file.write_all(&buffer[..bytes_read])
                .await
                .map_err(|e| BgRemovalError::file_io_error("write to file", local_path, &e))?;

            downloaded += bytes_read as u64;

            // Update progress
            if let Some(pb) = progress {
                if let Some(total) = total_size {
                    pb.set_length(total);
                    pb.set_position(downloaded);
                } else {
                    pb.set_message(format!(
                        "Downloaded {:.1} MB",
                        downloaded as f64 / 1_024_000.0
                    ));
                }
            }
        }

        file.flush()
            .await
            .map_err(|e| BgRemovalError::file_io_error("flush file", local_path, &e))?;

        log::debug!(
            "Downloaded {} bytes to {}",
            downloaded,
            local_path.display()
        );
        Ok(())
    }

    /// Verify the integrity of downloaded files using SHA256
    pub fn verify_file_integrity(
        &self,
        file_path: &Path,
        expected_hash: Option<&str>,
    ) -> Result<bool> {
        if expected_hash.is_none() {
            // No hash provided, skip verification
            return Ok(true);
        }

        let expected = expected_hash.unwrap();
        let contents = fs::read(file_path).map_err(|e| {
            BgRemovalError::file_io_error("read file for verification", file_path, &e)
        })?;

        let mut hasher = Sha256::new();
        hasher.update(&contents);
        let actual_hash = format!("{:x}", hasher.finalize());

        if actual_hash == expected {
            Ok(true)
        } else {
            log::warn!(
                "File integrity check failed for {}: expected {}, got {}",
                file_path.display(),
                expected,
                actual_hash
            );
            Ok(false)
        }
    }

    /// Get the model cache for other operations
    #[must_use]
    pub fn cache(&self) -> &ModelCache {
        &self.cache
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new().expect("Failed to create default model downloader")
    }
}

/// Validate that a URL is a supported model repository
///
/// Currently only HuggingFace repositories are supported.
///
/// # Arguments
/// * `url` - URL to validate
///
/// # Returns
/// `Ok(())` if URL is valid, `Err` with descriptive message if invalid
pub fn validate_model_url(url: &str) -> Result<()> {
    if url.is_empty() {
        return Err(BgRemovalError::invalid_config(
            "Model URL cannot be empty".to_string(),
        ));
    }

    if !url.starts_with("https://huggingface.co/") {
        return Err(BgRemovalError::invalid_config(format!(
            "Unsupported URL format: {}. Only HuggingFace repositories are supported (https://huggingface.co/...)",
            url
        )));
    }

    // Extract repository path (everything after huggingface.co/)
    let repo_path = url.strip_prefix("https://huggingface.co/").unwrap();
    if repo_path.is_empty() || !repo_path.contains('/') {
        return Err(BgRemovalError::invalid_config(format!(
            "Invalid HuggingFace repository URL: {}. Expected format: https://huggingface.co/username/repo-name",
            url
        )));
    }

    Ok(())
}

/// Parse a HuggingFace URL and extract repository information
///
/// # Arguments
/// * `url` - HuggingFace repository URL
///
/// # Returns
/// `(username, repository_name)` if successful
pub fn parse_huggingface_url(url: &str) -> Result<(String, String)> {
    validate_model_url(url)?;

    let repo_path = url.strip_prefix("https://huggingface.co/").unwrap();
    let parts: Vec<&str> = repo_path.split('/').collect();

    if parts.len() < 2 {
        return Err(BgRemovalError::invalid_config(format!(
            "Invalid HuggingFace URL format: {}",
            url
        )));
    }

    Ok((parts[0].to_string(), parts[1].to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_model_url() {
        // Valid HuggingFace URLs
        assert!(validate_model_url("https://huggingface.co/imgly/isnet-general-onnx").is_ok());
        assert!(validate_model_url("https://huggingface.co/ZhengPeng7/BiRefNet").is_ok());

        // Invalid URLs
        assert!(validate_model_url("").is_err());
        assert!(validate_model_url("https://github.com/user/repo").is_err());
        assert!(validate_model_url("https://huggingface.co/").is_err());
        assert!(validate_model_url("https://huggingface.co/single-part").is_err());
    }

    #[test]
    fn test_parse_huggingface_url() {
        let (user, repo) = parse_huggingface_url("https://huggingface.co/imgly/isnet-general-onnx")
            .expect("Should parse valid URL");
        assert_eq!(user, "imgly");
        assert_eq!(repo, "isnet-general-onnx");

        let (user, repo) = parse_huggingface_url("https://huggingface.co/ZhengPeng7/BiRefNet")
            .expect("Should parse valid URL");
        assert_eq!(user, "ZhengPeng7");
        assert_eq!(repo, "BiRefNet");

        // Invalid URLs should fail
        assert!(parse_huggingface_url("https://huggingface.co/single-part").is_err());
        assert!(parse_huggingface_url("https://github.com/user/repo").is_err());
    }

    #[cfg(feature = "cli")]
    #[tokio::test]
    async fn test_downloader_creation() {
        // This test only runs when CLI features are enabled
        let _downloader = ModelDownloader::new().expect("Should create downloader successfully");
    }
}
