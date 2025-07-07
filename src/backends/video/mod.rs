//! Video processing backend module
//!
//! This module provides video processing capabilities using FFmpeg for frame extraction,
//! processing, and video reassembly. It integrates with the existing background removal
//! pipeline to enable frame-by-frame processing of video files.

#[cfg(feature = "video-support")]
pub mod ffmpeg;

#[cfg(feature = "video-support")]
pub mod frame;

#[cfg(feature = "video-support")]
pub mod codec;

#[cfg(feature = "video-support")]
pub use ffmpeg::*;

#[cfg(feature = "video-support")]
pub use frame::*;

#[cfg(feature = "video-support")]
pub use codec::*;

use crate::error::Result;
use async_trait::async_trait;
use std::path::Path;

/// Video format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFormat {
    /// MP4 format (H.264/H.265)
    Mp4,
    /// AVI format
    Avi,
    /// MOV format (QuickTime)
    Mov,
    /// MKV format (Matroska)
    Mkv,
    /// WebM format
    WebM,
}

impl VideoFormat {
    /// Get file extension for the video format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Avi => "avi",
            Self::Mov => "mov",
            Self::Mkv => "mkv",
            Self::WebM => "webm",
        }
    }

    /// Get MIME type for the video format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Mp4 => "video/mp4",
            Self::Avi => "video/x-msvideo",
            Self::Mov => "video/quicktime",
            Self::Mkv => "video/x-matroska",
            Self::WebM => "video/webm",
        }
    }

    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp4" => Some(Self::Mp4),
            "avi" => Some(Self::Avi),
            "mov" => Some(Self::Mov),
            "mkv" => Some(Self::Mkv),
            "webm" => Some(Self::WebM),
            _ => None,
        }
    }
}

/// Video metadata information
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    /// Video duration in seconds
    pub duration: f64,
    /// Video width in pixels
    pub width: u32,
    /// Video height in pixels
    pub height: u32,
    /// Frames per second
    pub fps: f64,
    /// Video format
    pub format: VideoFormat,
    /// Video codec name
    pub codec: String,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
    /// Whether the video has an audio track
    pub has_audio: bool,
}

/// Trait for video backend implementations
#[async_trait]
pub trait VideoBackend {
    /// Extract frames from a video file
    ///
    /// # Arguments
    /// * `input_path` - Path to the input video file
    ///
    /// # Returns
    /// A stream of frames with their metadata
    async fn extract_frames(&self, input_path: &Path) -> Result<FrameStream>;

    /// Reassemble frames into a video file
    ///
    /// # Arguments
    /// * `frames` - Stream of processed frames
    /// * `output_path` - Path for the output video file
    /// * `metadata` - Original video metadata for encoding parameters
    /// * `preserve_audio` - Whether to preserve the original audio track
    async fn reassemble_video(
        &self,
        frames: FrameStream,
        output_path: &Path,
        metadata: &VideoMetadata,
        preserve_audio: bool,
    ) -> Result<()>;

    /// Get video metadata without extracting frames
    ///
    /// # Arguments
    /// * `input_path` - Path to the video file
    ///
    /// # Returns
    /// Video metadata information
    async fn get_metadata(&self, input_path: &Path) -> Result<VideoMetadata>;

    /// Get supported video formats
    fn supported_formats(&self) -> &[VideoFormat];
}

/// Type alias for frame stream
pub type FrameStream = std::pin::Pin<Box<dyn futures::Stream<Item = Result<VideoFrame>> + Send>>;
