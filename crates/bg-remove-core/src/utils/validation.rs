//! Configuration validation utilities
//!
//! Shared validation logic for configuration parameters used across
//! CLI and Web crates to ensure consistency and reduce duplication.

use crate::{config::OutputFormat, error::BgRemovalError, Result};

/// Utility for validating configuration parameters
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate JPEG quality parameter
    ///
    /// # Arguments
    /// * `quality` - JPEG quality value to validate (should be 0-100)
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(BgRemovalError)` if invalid
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::utils::ConfigValidator;
    /// 
    /// assert!(ConfigValidator::validate_jpeg_quality(90).is_ok());
    /// assert!(ConfigValidator::validate_jpeg_quality(150).is_err());
    /// ```
    pub fn validate_jpeg_quality(quality: u8) -> Result<()> {
        if quality > 100 {
            return Err(BgRemovalError::invalid_config(
                "JPEG quality must be between 0 and 100"
            ));
        }
        Ok(())
    }

    /// Validate WebP quality parameter
    ///
    /// # Arguments
    /// * `quality` - WebP quality value to validate (should be 0-100)
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(BgRemovalError)` if invalid
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::utils::ConfigValidator;
    /// 
    /// assert!(ConfigValidator::validate_webp_quality(85).is_ok());
    /// assert!(ConfigValidator::validate_webp_quality(150).is_err());
    /// ```
    pub fn validate_webp_quality(quality: u8) -> Result<()> {
        if quality > 100 {
            return Err(BgRemovalError::invalid_config(
                "WebP quality must be between 0 and 100"
            ));
        }
        Ok(())
    }

    /// Validate both JPEG and WebP quality parameters
    ///
    /// # Arguments
    /// * `jpeg_quality` - JPEG quality value (0-100)
    /// * `webp_quality` - WebP quality value (0-100)
    ///
    /// # Returns
    /// `Ok(())` if both are valid, `Err(BgRemovalError)` if either is invalid
    pub fn validate_quality_settings(jpeg_quality: u8, webp_quality: u8) -> Result<()> {
        Self::validate_jpeg_quality(jpeg_quality)?;
        Self::validate_webp_quality(webp_quality)?;
        Ok(())
    }

    /// Validate output format string
    ///
    /// # Arguments
    /// * `format` - Output format string to validate
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(BgRemovalError)` if invalid
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::utils::ConfigValidator;
    /// 
    /// assert!(ConfigValidator::validate_output_format("png").is_ok());
    /// assert!(ConfigValidator::validate_output_format("invalid").is_err());
    /// ```
    pub fn validate_output_format(format: &str) -> Result<()> {
        let valid_formats = ["png", "jpeg", "jpg", "webp", "tiff", "rgba8"];
        if !valid_formats.contains(&format.to_lowercase().as_str()) {
            return Err(BgRemovalError::invalid_config(
                &format!("Invalid output format: {}. Valid formats: {}", 
                        format, 
                        valid_formats.join(", "))
            ));
        }
        Ok(())
    }

    /// Parse and validate output format string to OutputFormat enum
    ///
    /// # Arguments
    /// * `format` - Output format string to parse and validate
    ///
    /// # Returns
    /// `Ok(OutputFormat)` if valid, `Err(BgRemovalError)` if invalid
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::{utils::ConfigValidator, config::OutputFormat};
    /// 
    /// assert_eq!(ConfigValidator::parse_output_format("png").unwrap(), OutputFormat::Png);
    /// assert_eq!(ConfigValidator::parse_output_format("jpeg").unwrap(), OutputFormat::Jpeg);
    /// ```
    pub fn parse_output_format(format: &str) -> Result<OutputFormat> {
        Self::validate_output_format(format)?;
        
        match format.to_lowercase().as_str() {
            "png" => Ok(OutputFormat::Png),
            "jpeg" | "jpg" => Ok(OutputFormat::Jpeg),
            "webp" => Ok(OutputFormat::WebP),
            "tiff" => Ok(OutputFormat::Tiff),
            "rgba8" => Ok(OutputFormat::Rgba8),
            _ => unreachable!("Format already validated"), // Should never reach here
        }
    }

    /// Validate thread count parameter
    ///
    /// # Arguments
    /// * `threads` - Thread count to validate
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(BgRemovalError)` if invalid
    ///
    /// # Note
    /// Currently accepts any non-negative value (0 means auto-detect)
    pub fn validate_thread_count(_threads: usize) -> Result<()> {
        // Thread counts are valid as long as they're non-negative
        // 0 means auto-detect, which is valid
        Ok(())
    }

    /// List valid output formats
    ///
    /// # Returns
    /// Vector of valid output format strings
    pub fn valid_output_formats() -> Vec<&'static str> {
        vec!["png", "jpeg", "jpg", "webp", "tiff", "rgba8"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jpeg_quality_validation() {
        assert!(ConfigValidator::validate_jpeg_quality(0).is_ok());
        assert!(ConfigValidator::validate_jpeg_quality(50).is_ok());
        assert!(ConfigValidator::validate_jpeg_quality(100).is_ok());
        assert!(ConfigValidator::validate_jpeg_quality(101).is_err());
        assert!(ConfigValidator::validate_jpeg_quality(255).is_err());
    }

    #[test]
    fn test_webp_quality_validation() {
        assert!(ConfigValidator::validate_webp_quality(0).is_ok());
        assert!(ConfigValidator::validate_webp_quality(85).is_ok());
        assert!(ConfigValidator::validate_webp_quality(100).is_ok());
        assert!(ConfigValidator::validate_webp_quality(101).is_err());
        assert!(ConfigValidator::validate_webp_quality(255).is_err());
    }

    #[test]
    fn test_quality_settings_validation() {
        assert!(ConfigValidator::validate_quality_settings(90, 85).is_ok());
        assert!(ConfigValidator::validate_quality_settings(101, 85).is_err());
        assert!(ConfigValidator::validate_quality_settings(90, 101).is_err());
        assert!(ConfigValidator::validate_quality_settings(101, 101).is_err());
    }

    #[test]
    fn test_output_format_validation() {
        assert!(ConfigValidator::validate_output_format("png").is_ok());
        assert!(ConfigValidator::validate_output_format("PNG").is_ok());
        assert!(ConfigValidator::validate_output_format("jpeg").is_ok());
        assert!(ConfigValidator::validate_output_format("jpg").is_ok());
        assert!(ConfigValidator::validate_output_format("webp").is_ok());
        assert!(ConfigValidator::validate_output_format("tiff").is_ok());
        assert!(ConfigValidator::validate_output_format("rgba8").is_ok());
        assert!(ConfigValidator::validate_output_format("invalid").is_err());
        assert!(ConfigValidator::validate_output_format("bmp").is_err());
    }

    #[test]
    fn test_parse_output_format() {
        assert_eq!(ConfigValidator::parse_output_format("png").unwrap(), OutputFormat::Png);
        assert_eq!(ConfigValidator::parse_output_format("PNG").unwrap(), OutputFormat::Png);
        assert_eq!(ConfigValidator::parse_output_format("jpeg").unwrap(), OutputFormat::Jpeg);
        assert_eq!(ConfigValidator::parse_output_format("jpg").unwrap(), OutputFormat::Jpeg);
        assert_eq!(ConfigValidator::parse_output_format("webp").unwrap(), OutputFormat::WebP);
        assert_eq!(ConfigValidator::parse_output_format("tiff").unwrap(), OutputFormat::Tiff);
        assert_eq!(ConfigValidator::parse_output_format("rgba8").unwrap(), OutputFormat::Rgba8);
        assert!(ConfigValidator::parse_output_format("invalid").is_err());
    }

    #[test]
    fn test_thread_count_validation() {
        assert!(ConfigValidator::validate_thread_count(0).is_ok());
        assert!(ConfigValidator::validate_thread_count(1).is_ok());
        assert!(ConfigValidator::validate_thread_count(8).is_ok());
        assert!(ConfigValidator::validate_thread_count(1000).is_ok());
    }

    #[test]
    fn test_valid_output_formats() {
        let formats = ConfigValidator::valid_output_formats();
        assert!(formats.contains(&"png"));
        assert!(formats.contains(&"jpeg"));
        assert!(formats.contains(&"jpg"));
        assert!(formats.contains(&"webp"));
        assert!(formats.contains(&"tiff"));
        assert!(formats.contains(&"rgba8"));
        assert_eq!(formats.len(), 6);
    }
}