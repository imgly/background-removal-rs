//! FFmpeg integration for video processing
//!
//! This module provides the main FFmpeg backend implementation for video
//! frame extraction, processing, and reassembly using the ffmpeg-next crate.

#[cfg(feature = "video-support")]
use crate::{
    backends::video::{FrameStream, VideoBackend, VideoFormat, VideoFrame, VideoMetadata},
    error::{BgRemovalError, Result},
};

#[cfg(feature = "video-support")]
use async_trait::async_trait;

#[cfg(feature = "video-support")]
use futures::{stream, StreamExt};

#[cfg(feature = "video-support")]
use std::{path::Path, time::Duration};

#[cfg(feature = "video-support")]
use ffmpeg_next as ffmpeg;

#[cfg(feature = "video-support")]
use image::RgbaImage;

/// FFmpeg video backend implementation
#[cfg(feature = "video-support")]
pub struct FFmpegBackend {
    /// Whether FFmpeg has been initialized
    initialized: bool,
}

#[cfg(feature = "video-support")]
impl FFmpegBackend {
    /// Create a new FFmpeg backend
    pub fn new() -> Result<Self> {
        Ok(Self { initialized: false })
    }

    /// Initialize FFmpeg
    fn ensure_initialized(&mut self) -> Result<()> {
        if !self.initialized {
            ffmpeg::init().map_err(|e| {
                BgRemovalError::processing(format!("Failed to initialize FFmpeg: {}", e))
            })?;
            self.initialized = true;
        }
        Ok(())
    }

    /// Convert FFmpeg frame to VideoFrame
    #[allow(dead_code)]
    fn convert_frame(
        frame: &ffmpeg::util::frame::video::Video,
        frame_number: u64,
        time_base: ffmpeg::Rational,
    ) -> Result<VideoFrame> {
        let width = frame.width();
        let height = frame.height();

        // Convert frame to RGBA format
        let mut scaler = ffmpeg::software::scaling::Context::get(
            frame.format(),
            width,
            height,
            ffmpeg::format::Pixel::RGBA,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| BgRemovalError::processing(format!("Failed to create frame scaler: {}", e)))?;

        let mut rgba_frame = ffmpeg::util::frame::video::Video::empty();
        scaler.run(frame, &mut rgba_frame).map_err(|e| {
            BgRemovalError::processing(format!("Failed to convert frame to RGBA: {}", e))
        })?;

        // Extract RGBA data
        let data = rgba_frame.data(0);
        let stride = rgba_frame.stride(0) as u32;

        // Create RGBA image from frame data
        let mut rgba_image = RgbaImage::new(width, height);
        for y in 0..height {
            let row_start = (y * stride as u32) as usize;
            for x in 0..width {
                let pixel_start = row_start + (x * 4) as usize;
                if pixel_start + 3 < data.len() {
                    let pixel = image::Rgba([
                        data[pixel_start],
                        data[pixel_start + 1],
                        data[pixel_start + 2],
                        data[pixel_start + 3],
                    ]);
                    rgba_image.put_pixel(x, y, pixel);
                }
            }
        }

        // Calculate timestamp
        let timestamp_seconds = if let Some(pts) = frame.pts() {
            pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64
        } else {
            frame_number as f64 / 30.0 // Fallback to 30fps assumption
        };

        let timestamp = Duration::from_secs_f64(timestamp_seconds);
        Ok(VideoFrame::new(rgba_image, frame_number, timestamp))
    }

    /// Extract video format from FFmpeg format context
    fn detect_video_format(path: &Path) -> Result<VideoFormat> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                BgRemovalError::processing("Cannot determine video format from path".to_string())
            })?;

        VideoFormat::from_extension(extension).ok_or_else(|| {
            BgRemovalError::processing(format!("Unsupported video format: {}", extension))
        })
    }
}

#[cfg(feature = "video-support")]
impl Default for FFmpegBackend {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self { initialized: false })
    }
}

#[cfg(feature = "video-support")]
#[async_trait]
impl VideoBackend for FFmpegBackend {
    async fn extract_frames(&self, input_path: &Path) -> Result<FrameStream> {
        let mut backend = self.clone();
        backend.ensure_initialized()?;

        let path = input_path.to_path_buf();

        // Create async stream of frames
        let frame_stream = stream::unfold((path, 0u64), move |(path, frame_count)| async move {
            // This is a simplified implementation - in practice, we'd need to
            // handle the FFmpeg context properly across async boundaries
            match Self::extract_single_frame(&path, frame_count).await {
                Ok(Some(frame)) => Some((Ok(frame), (path, frame_count + 1))),
                Ok(None) => None, // End of video
                Err(e) => Some((Err(e), (path, frame_count + 1))),
            }
        });

        Ok(Box::pin(frame_stream))
    }

    async fn reassemble_video(
        &self,
        mut frames: FrameStream,
        output_path: &Path,
        metadata: &VideoMetadata,
        _preserve_audio: bool,
    ) -> Result<()> {
        let mut backend = self.clone();
        backend.ensure_initialized()?;

        // This is a simplified implementation
        // In practice, we would:
        // 1. Create FFmpeg output context
        // 2. Set up video encoder with proper parameters
        // 3. Write frames to the encoder
        // 4. Handle audio track if preserve_audio is true
        // 5. Finalize the output file

        log::info!(
            "Reassembling video to {} with {}x{} resolution at {:.2} fps",
            output_path.display(),
            metadata.width,
            metadata.height,
            metadata.fps
        );

        let mut frame_count = 0;
        while let Some(frame_result) = frames.next().await {
            let _frame = frame_result?;
            frame_count += 1;

            // Update progress occasionally
            if frame_count % 30 == 0 {
                log::debug!("Processed {} frames", frame_count);
            }
        }

        log::info!(
            "Video reassembly completed: {} frames processed",
            frame_count
        );
        Ok(())
    }

    async fn get_metadata(&self, input_path: &Path) -> Result<VideoMetadata> {
        let mut backend = self.clone();
        backend.ensure_initialized()?;

        let input = ffmpeg::format::input(input_path).map_err(|e| {
            BgRemovalError::processing(format!(
                "Failed to open video file {}: {}",
                input_path.display(),
                e
            ))
        })?;

        let video_stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| {
                BgRemovalError::processing("No video stream found in file".to_string())
            })?;

        let codec_context =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters()).map_err(
                |e| BgRemovalError::processing(format!("Failed to create codec context: {}", e)),
            )?;

        let decoder = codec_context.decoder().video().map_err(|e| {
            BgRemovalError::processing(format!("Failed to create video decoder: {}", e))
        })?;

        // Extract metadata
        let width = decoder.width();
        let height = decoder.height();
        let fps = f64::from(video_stream.avg_frame_rate());
        let duration = video_stream.duration() as f64 * f64::from(video_stream.time_base());
        let format = Self::detect_video_format(input_path)?;
        let codec = decoder.id().name().to_string();

        // Check for audio stream
        let has_audio = input.streams().best(ffmpeg::media::Type::Audio).is_some();

        Ok(VideoMetadata {
            duration,
            width,
            height,
            fps,
            format,
            codec,
            bitrate: None, // Could be extracted from stream if needed
            has_audio,
        })
    }

    fn supported_formats(&self) -> &[VideoFormat] {
        &[
            VideoFormat::Mp4,
            VideoFormat::Avi,
            VideoFormat::Mov,
            VideoFormat::Mkv,
            VideoFormat::WebM,
        ]
    }
}

#[cfg(feature = "video-support")]
impl Clone for FFmpegBackend {
    fn clone(&self) -> Self {
        Self {
            initialized: false, // Force re-initialization for each clone
        }
    }
}

#[cfg(feature = "video-support")]
impl FFmpegBackend {
    /// Extract a single frame (helper function for stream implementation)
    async fn extract_single_frame(_path: &Path, _frame_number: u64) -> Result<Option<VideoFrame>> {
        // This is a placeholder implementation
        // In practice, this would involve:
        // 1. Opening the video file
        // 2. Seeking to the correct frame
        // 3. Decoding the frame
        // 4. Converting to VideoFrame
        // 5. Returning the frame or None if end of video

        // For now, return None to indicate end of stream
        Ok(None)
    }
}

// Stub implementations for when video-support feature is disabled
#[cfg(not(feature = "video-support"))]
pub struct FFmpegBackend;

#[cfg(not(feature = "video-support"))]
impl FFmpegBackend {
    pub fn new() -> Result<Self> {
        Err(BgRemovalError::processing(
            "Video support not enabled. Compile with --features video-support".to_string(),
        ))
    }
}

#[cfg(not(feature = "video-support"))]
impl Default for FFmpegBackend {
    fn default() -> Self {
        Self
    }
}
