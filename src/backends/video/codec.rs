//! Video codec handling and validation
//!
//! This module provides utilities for working with video codecs,
//! quality settings, and encoding parameters.

use crate::error::{BgRemovalError, Result};
use std::fmt;

/// Video codec enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 codec (most compatible)
    H264,
    /// H.265/HEVC codec (better compression)
    H265,
    /// VP8 codec (WebM)
    VP8,
    /// VP9 codec (WebM)
    VP9,
    /// AV1 codec (latest standard)
    AV1,
}

impl VideoCodec {
    /// Get FFmpeg codec name
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            Self::H264 => "libx264",
            Self::H265 => "libx265",
            Self::VP8 => "libvpx",
            Self::VP9 => "libvpx-vp9",
            Self::AV1 => "libaom-av1",
        }
    }

    /// Get codec description
    pub fn description(&self) -> &'static str {
        match self {
            Self::H264 => "H.264/AVC - Most compatible, good quality",
            Self::H265 => "H.265/HEVC - Better compression than H.264",
            Self::VP8 => "VP8 - Open source, WebM compatible",
            Self::VP9 => "VP9 - Open source, better than VP8",
            Self::AV1 => "AV1 - Latest standard, best compression",
        }
    }

    /// Check if codec supports hardware acceleration
    pub fn supports_hardware_acceleration(&self) -> bool {
        matches!(self, Self::H264 | Self::H265)
    }

    /// Get recommended quality range (0-100)
    pub fn quality_range(&self) -> (u8, u8) {
        match self {
            Self::H264 => (18, 28), // CRF values
            Self::H265 => (20, 30), // CRF values
            Self::VP8 => (4, 63),   // Quality values
            Self::VP9 => (15, 35),  // CRF values
            Self::AV1 => (20, 40),  // CRF values
        }
    }

    /// Get default quality setting
    pub fn default_quality(&self) -> u8 {
        let (min, max) = self.quality_range();
        (min + max) / 2
    }

    /// Parse codec from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "h264" | "libx264" | "avc" => Ok(Self::H264),
            "h265" | "libx265" | "hevc" => Ok(Self::H265),
            "vp8" | "libvpx" => Ok(Self::VP8),
            "vp9" | "libvpx-vp9" => Ok(Self::VP9),
            "av1" | "libaom-av1" => Ok(Self::AV1),
            _ => Err(BgRemovalError::processing(format!(
                "Unsupported video codec: {}",
                s
            ))),
        }
    }

    /// Get all supported codecs
    pub fn all() -> &'static [Self] {
        &[Self::H264, Self::H265, Self::VP8, Self::VP9, Self::AV1]
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ffmpeg_name())
    }
}

/// Video quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    /// Fastest encoding, largest file size
    Fast,
    /// Balanced encoding speed and quality
    Medium,
    /// Slow encoding, best quality/size ratio
    Slow,
    /// Custom quality settings
    Custom,
}

impl QualityPreset {
    /// Get FFmpeg preset name
    pub fn ffmpeg_preset(&self) -> Option<&'static str> {
        match self {
            Self::Fast => Some("fast"),
            Self::Medium => Some("medium"),
            Self::Slow => Some("slow"),
            Self::Custom => None,
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Fast => "Fast encoding, larger files",
            Self::Medium => "Balanced speed and quality",
            Self::Slow => "Best quality, slower encoding",
            Self::Custom => "Custom encoding parameters",
        }
    }
}

/// Video encoding configuration
#[derive(Debug, Clone)]
pub struct VideoEncodingConfig {
    /// Video codec to use
    pub codec: VideoCodec,
    /// Quality preset
    pub preset: QualityPreset,
    /// Quality value (codec-specific)
    pub quality: u8,
    /// Bitrate in kbps (optional, for bitrate-based encoding)
    pub bitrate: Option<u32>,
    /// Enable hardware acceleration if available
    pub hardware_acceleration: bool,
    /// Number of encoding threads
    pub threads: Option<u8>,
    /// Pixel format
    pub pixel_format: PixelFormat,
}

impl VideoEncodingConfig {
    /// Create new encoding config with defaults
    pub fn new(codec: VideoCodec) -> Self {
        Self {
            codec,
            preset: QualityPreset::Medium,
            quality: codec.default_quality(),
            bitrate: None,
            hardware_acceleration: codec.supports_hardware_acceleration(),
            threads: None,
            pixel_format: PixelFormat::Yuv420p,
        }
    }

    /// Set quality value
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Set quality preset
    pub fn with_preset(mut self, preset: QualityPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Set bitrate for bitrate-based encoding
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    /// Enable or disable hardware acceleration
    pub fn with_hardware_acceleration(mut self, enabled: bool) -> Self {
        self.hardware_acceleration = enabled && self.codec.supports_hardware_acceleration();
        self
    }

    /// Set number of encoding threads
    pub fn with_threads(mut self, threads: u8) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Set pixel format
    pub fn with_pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        let (min_quality, max_quality) = self.codec.quality_range();
        if self.quality < min_quality || self.quality > max_quality {
            return Err(BgRemovalError::processing(format!(
                "Quality {} is out of range for codec {} (valid range: {}-{})",
                self.quality,
                self.codec,
                min_quality,
                max_quality
            )));
        }

        if let Some(bitrate) = self.bitrate {
            if bitrate == 0 {
                return Err(BgRemovalError::processing(
                    "Bitrate cannot be zero".to_string(),
                ));
            }
        }

        if let Some(threads) = self.threads {
            if threads == 0 {
                return Err(BgRemovalError::processing(
                    "Thread count cannot be zero".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Default for VideoEncodingConfig {
    fn default() -> Self {
        Self::new(VideoCodec::H264)
    }
}

/// Pixel format for video encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// YUV 4:2:0 (most common)
    Yuv420p,
    /// YUV 4:2:2
    Yuv422p,
    /// YUV 4:4:4
    Yuv444p,
    /// RGB 24-bit
    Rgb24,
    /// RGBA 32-bit
    Rgba,
}

impl PixelFormat {
    /// Get FFmpeg pixel format name
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            Self::Yuv420p => "yuv420p",
            Self::Yuv422p => "yuv422p",
            Self::Yuv444p => "yuv444p",
            Self::Rgb24 => "rgb24",
            Self::Rgba => "rgba",
        }
    }

    /// Check if format supports transparency
    pub fn supports_transparency(&self) -> bool {
        matches!(self, Self::Rgba)
    }

    /// Get bits per pixel
    pub fn bits_per_pixel(&self) -> u8 {
        match self {
            Self::Yuv420p => 12,
            Self::Yuv422p => 16,
            Self::Yuv444p => 24,
            Self::Rgb24 => 24,
            Self::Rgba => 32,
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ffmpeg_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_properties() {
        let h264 = VideoCodec::H264;
        assert_eq!(h264.ffmpeg_name(), "libx264");
        assert!(h264.supports_hardware_acceleration());
        assert_eq!(h264.quality_range(), (18, 28));
        assert_eq!(h264.default_quality(), 23);
    }

    #[test]
    fn test_codec_from_string() {
        assert_eq!(VideoCodec::from_str("h264").unwrap(), VideoCodec::H264);
        assert_eq!(VideoCodec::from_str("H265").unwrap(), VideoCodec::H265);
        assert_eq!(VideoCodec::from_str("vp9").unwrap(), VideoCodec::VP9);
        assert!(VideoCodec::from_str("invalid").is_err());
    }

    #[test]
    fn test_encoding_config() {
        let config = VideoEncodingConfig::new(VideoCodec::H264)
            .with_quality(20)
            .with_preset(QualityPreset::Fast)
            .with_bitrate(2000);

        assert_eq!(config.codec, VideoCodec::H264);
        assert_eq!(config.quality, 20);
        assert_eq!(config.preset, QualityPreset::Fast);
        assert_eq!(config.bitrate, Some(2000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = VideoEncodingConfig::new(VideoCodec::H264);
        config.quality = 100; // Out of range for H264
        assert!(config.validate().is_err());

        config.quality = 23; // Valid
        config.bitrate = Some(0); // Invalid
        assert!(config.validate().is_err());

        config.bitrate = Some(1000); // Valid
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pixel_format_properties() {
        let yuv420p = PixelFormat::Yuv420p;
        assert_eq!(yuv420p.ffmpeg_name(), "yuv420p");
        assert!(!yuv420p.supports_transparency());
        assert_eq!(yuv420p.bits_per_pixel(), 12);

        let rgba = PixelFormat::Rgba;
        assert!(rgba.supports_transparency());
        assert_eq!(rgba.bits_per_pixel(), 32);
    }

    #[test]
    fn test_quality_preset() {
        let fast = QualityPreset::Fast;
        assert_eq!(fast.ffmpeg_preset(), Some("fast"));
        
        let custom = QualityPreset::Custom;
        assert_eq!(custom.ffmpeg_preset(), None);
    }

    #[test]
    fn test_all_codecs() {
        let codecs = VideoCodec::all();
        assert!(codecs.len() >= 5);
        assert!(codecs.contains(&VideoCodec::H264));
        assert!(codecs.contains(&VideoCodec::H265));
    }
}