//! Path validation utilities
//!
//! Provides centralized validation for file paths, extensions, and directories.

use crate::error::{BgRemovalError, Result};
use std::path::Path;

/// Validator for file system paths and extensions
pub struct PathValidator;

impl PathValidator {
    /// Validate that a file exists
    pub fn validate_file_exists<P: AsRef<Path>>(path: P) -> Result<()> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(BgRemovalError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File does not exist: {}", path_ref.display()),
            )));
        }
        Ok(())
    }

    /// Validate that a path is a directory
    pub fn validate_is_directory<P: AsRef<Path>>(path: P) -> Result<()> {
        let path_ref = path.as_ref();
        Self::validate_file_exists(&path_ref)?;

        if !path_ref.is_dir() {
            return Err(BgRemovalError::invalid_config(&format!(
                "Path is not a directory: {}",
                path_ref.display()
            )));
        }
        Ok(())
    }

    /// Validate that a path is a file (not a directory)
    pub fn validate_is_file<P: AsRef<Path>>(path: P) -> Result<()> {
        let path_ref = path.as_ref();
        Self::validate_file_exists(&path_ref)?;

        if !path_ref.is_file() {
            return Err(BgRemovalError::invalid_config(&format!(
                "Path is not a file: {}",
                path_ref.display()
            )));
        }
        Ok(())
    }

    /// Validate that a path has a supported image extension
    pub fn validate_image_extension<P: AsRef<Path>>(path: P) -> Result<()> {
        let path_ref = path.as_ref();

        if !Self::is_supported_image_format(&path_ref) {
            let extension = path_ref
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("(no extension)");

            return Err(BgRemovalError::invalid_config(&format!(
                "Unsupported image format '{}'. Supported formats: jpg, jpeg, png, webp, tiff, tif, bmp",
                extension
            )));
        }
        Ok(())
    }

    /// Check if a file path has a supported image extension
    pub fn is_supported_image_format<P: AsRef<Path>>(path: P) -> bool {
        let path_ref = path.as_ref();

        if let Some(extension) = path_ref.extension() {
            if let Some(ext_str) = extension.to_str() {
                let ext_lower = ext_str.to_lowercase();
                return matches!(
                    ext_lower.as_str(),
                    "jpg" | "jpeg" | "png" | "webp" | "tiff" | "tif" | "bmp"
                );
            }
        }
        false
    }

    /// Get the list of supported image extensions
    pub fn supported_image_extensions() -> &'static [&'static str] {
        &["jpg", "jpeg", "png", "webp", "tiff", "tif", "bmp"]
    }

    /// Validate that a path has a specific extension
    pub fn validate_extension<P: AsRef<Path>>(path: P, expected_ext: &str) -> Result<()> {
        let path_ref = path.as_ref();
        let actual_ext = path_ref.extension().and_then(|s| s.to_str()).unwrap_or("");

        if actual_ext.to_lowercase() != expected_ext.to_lowercase() {
            return Err(BgRemovalError::invalid_config(&format!(
                "Expected {} file, but got: {}",
                expected_ext,
                path_ref.display()
            )));
        }
        Ok(())
    }

    /// Validate that a parent directory exists (for output paths)
    pub fn validate_parent_exists<P: AsRef<Path>>(path: P) -> Result<()> {
        let path_ref = path.as_ref();

        if let Some(parent) = path_ref.parent() {
            if !parent.exists() {
                return Err(BgRemovalError::invalid_config(&format!(
                    "Parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }
        Ok(())
    }

    /// Create parent directories if they don't exist
    pub fn ensure_parent_dirs<P: AsRef<Path>>(path: P) -> Result<()> {
        let path_ref = path.as_ref();

        if let Some(parent) = path_ref.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                BgRemovalError::processing(format!(
                    "Failed to create parent directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_is_supported_image_format() {
        // Supported formats
        assert!(PathValidator::is_supported_image_format("test.jpg"));
        assert!(PathValidator::is_supported_image_format("test.jpeg"));
        assert!(PathValidator::is_supported_image_format("test.png"));
        assert!(PathValidator::is_supported_image_format("test.webp"));
        assert!(PathValidator::is_supported_image_format("test.tiff"));
        assert!(PathValidator::is_supported_image_format("test.tif"));
        assert!(PathValidator::is_supported_image_format("test.bmp"));

        // Case insensitive
        assert!(PathValidator::is_supported_image_format("test.JPG"));
        assert!(PathValidator::is_supported_image_format("test.PNG"));

        // Unsupported formats
        assert!(!PathValidator::is_supported_image_format("test.txt"));
        assert!(!PathValidator::is_supported_image_format("test.pdf"));
        assert!(!PathValidator::is_supported_image_format("test"));
        assert!(!PathValidator::is_supported_image_format(""));
    }

    #[test]
    fn test_validate_file_exists() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // File doesn't exist
        assert!(PathValidator::validate_file_exists(&file_path).is_err());

        // Create file
        fs::write(&file_path, "test").unwrap();
        assert!(PathValidator::validate_file_exists(&file_path).is_ok());
    }

    #[test]
    fn test_validate_is_directory() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().join("subdir");
        let file_path = temp_dir.path().join("file.txt");

        // Directory doesn't exist
        assert!(PathValidator::validate_is_directory(&dir_path).is_err());

        // Create directory
        fs::create_dir(&dir_path).unwrap();
        assert!(PathValidator::validate_is_directory(&dir_path).is_ok());

        // Create file and check it's not a directory
        fs::write(&file_path, "test").unwrap();
        assert!(PathValidator::validate_is_directory(&file_path).is_err());
    }

    #[test]
    fn test_validate_image_extension() {
        assert!(PathValidator::validate_image_extension("photo.jpg").is_ok());
        assert!(PathValidator::validate_image_extension("photo.png").is_ok());
        assert!(PathValidator::validate_image_extension("photo.txt").is_err());
        assert!(PathValidator::validate_image_extension("photo").is_err());
    }

    #[test]
    fn test_validate_extension() {
        assert!(PathValidator::validate_extension("model.onnx", "onnx").is_ok());
        assert!(PathValidator::validate_extension("model.ONNX", "onnx").is_ok());
        assert!(PathValidator::validate_extension("model.pb", "onnx").is_err());
        assert!(PathValidator::validate_extension("model", "onnx").is_err());
    }

    #[test]
    fn test_ensure_parent_dirs() {
        let temp_dir = TempDir::new().unwrap();
        let nested_path = temp_dir
            .path()
            .join("a")
            .join("b")
            .join("c")
            .join("file.txt");

        // Parent doesn't exist
        assert!(!nested_path.parent().unwrap().exists());

        // Create parent directories
        assert!(PathValidator::ensure_parent_dirs(&nested_path).is_ok());
        assert!(nested_path.parent().unwrap().exists());
    }
}
