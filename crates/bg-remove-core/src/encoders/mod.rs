//! ICC color profile encoders for various image formats
//!
//! This module provides format-specific ICC profile embedding capabilities
//! for PNG, JPEG, and WebP image formats. Each encoder implements the
//! appropriate standards-compliant method for embedding ICC profiles:
//!
//! - **PNG**: Custom iCCP chunk creation with zlib compression
//! - **JPEG**: APP2 marker implementation with multi-segment support
//! - **WebP**: RIFF ICCP chunk embedding in container structure
//!
//! # Standards Compliance
//!
//! All encoders follow their respective format specifications:
//! - PNG: PNG Specification 1.2 (iCCP chunk format)
//! - JPEG: JPEG ICC Profile Specification (APP2 markers)
//! - WebP: WebP Container Specification (RIFF ICCP chunks)
//!
//! # Usage
//!
//! ```rust,no_run
//! use bg_remove_core::encoders::{JpegIccEncoder, PngIccEncoder, WebPIccEncoder};
//! use bg_remove_core::types::ColorProfile;
//! use image::DynamicImage;
//!
//! # fn example(image: DynamicImage, profile: ColorProfile) -> bg_remove_core::Result<()> {
//! // PNG with ICC profile
//! PngIccEncoder::encode_with_profile(&image, &profile, "output.png")?;
//!
//! // JPEG with ICC profile and quality setting
//! JpegIccEncoder::encode_with_profile(&image, &profile, "output.jpg", 90)?;
//!
//! // WebP with ICC profile and quality setting
//! WebPIccEncoder::encode_with_profile(&image, &profile, "output.webp", 85)?;
//! # Ok(())
//! # }
//! ```

pub mod jpeg_encoder;
pub mod png_encoder;
pub mod webp_encoder;

// Re-export the main encoder types for convenience
pub use jpeg_encoder::JpegIccEncoder;
pub use png_encoder::PngIccEncoder;
pub use webp_encoder::WebPIccEncoder;