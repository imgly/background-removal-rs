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
use futures::StreamExt;

#[cfg(feature = "video-support")]
use std::{path::Path, time::Duration};

#[cfg(feature = "video-support")]
use ffmpeg_next as ffmpeg;

#[cfg(feature = "video-support")]
use image::RgbaImage;

#[cfg(feature = "video-support")]
use tokio_stream::wrappers::ReceiverStream;

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
        // Initialize FFmpeg globally if needed
        if !self.initialized {
            ffmpeg::init().map_err(|e| {
                BgRemovalError::processing(format!("Failed to initialize FFmpeg: {}", e))
            })?;
        }

        // Open input file
        let mut input = ffmpeg::format::input(input_path).map_err(|e| {
            BgRemovalError::processing(format!(
                "Failed to open video file {}: {}",
                input_path.display(),
                e
            ))
        })?;

        // Find video stream
        let video_stream_index = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| BgRemovalError::processing("No video stream found in file".to_string()))?
            .index();

        // Get video stream info
        let video_stream = input.stream(video_stream_index).unwrap();
        let time_base = video_stream.time_base();

        // Create decoder
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters()).map_err(
                |e| BgRemovalError::processing(format!("Failed to create codec context: {}", e)),
            )?;

        let mut decoder = context_decoder.decoder().video().map_err(|e| {
            BgRemovalError::processing(format!("Failed to create video decoder: {}", e))
        })?;

        // Create channel for frame transfer
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // Spawn blocking task for FFmpeg operations
        tokio::task::spawn_blocking(move || {
            let mut frame_number = 0u64;
            let mut packet_count = 0;

            // Process packets
            for (stream, packet) in input.packets() {
                if stream.index() == video_stream_index {
                    packet_count += 1;

                    // Send packet to decoder
                    if let Err(e) = decoder.send_packet(&packet) {
                        log::error!("Failed to send packet to decoder: {}", e);
                        continue;
                    }

                    // Receive decoded frames
                    let mut decoded = ffmpeg::util::frame::video::Video::empty();
                    while decoder.receive_frame(&mut decoded).is_ok() {
                        // Convert frame to VideoFrame
                        match FFmpegBackend::convert_frame(&decoded, frame_number, time_base) {
                            Ok(video_frame) => {
                                if tx.blocking_send(Ok(video_frame)).is_err() {
                                    // Receiver dropped, stop processing
                                    return;
                                }
                                frame_number += 1;
                            },
                            Err(e) => {
                                log::error!("Failed to convert frame {}: {}", frame_number, e);
                                if tx.blocking_send(Err(e)).is_err() {
                                    return;
                                }
                            },
                        }
                    }
                }
            }

            // Send remaining frames
            decoder.send_eof().ok();
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                match FFmpegBackend::convert_frame(&decoded, frame_number, time_base) {
                    Ok(video_frame) => {
                        if tx.blocking_send(Ok(video_frame)).is_err() {
                            return;
                        }
                        frame_number += 1;
                    },
                    Err(e) => {
                        log::error!("Failed to convert frame {}: {}", frame_number, e);
                        if tx.blocking_send(Err(e)).is_err() {
                            return;
                        }
                    },
                }
            }

            log::info!(
                "Extracted {} frames from {} packets",
                frame_number,
                packet_count
            );
        });

        // Convert receiver to stream
        let frame_stream = ReceiverStream::new(rx);
        Ok(Box::pin(frame_stream))
    }

    async fn reassemble_video(
        &self,
        mut frames: FrameStream,
        output_path: &Path,
        metadata: &VideoMetadata,
        preserve_audio: bool,
    ) -> Result<()> {
        log::info!(
            "Reassembling video to {} with {}x{} resolution at {:.2} fps",
            output_path.display(),
            metadata.width,
            metadata.height,
            metadata.fps
        );

        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                BgRemovalError::processing(format!("Failed to create output directory: {}", e))
            })?;
        }

        // For now, create a simple text file explaining the current state
        // and save frames to a directory for user access
        let output_dir = output_path.with_extension("frames");
        std::fs::create_dir_all(&output_dir).map_err(|e| {
            BgRemovalError::processing(format!("Failed to create frames directory: {}", e))
        })?;

        let mut frame_count = 0;
        while let Some(frame_result) = frames.next().await {
            let frame = frame_result?;
            
            // Save frame as PNG
            let frame_path = output_dir.join(format!("frame_{:06}.png", frame_count));
            frame.image.save(&frame_path).map_err(|e| {
                BgRemovalError::processing(format!("Failed to save frame {}: {}", frame_count, e))
            })?;

            frame_count += 1;

            // Update progress occasionally
            if frame_count % 30 == 0 {
                log::debug!("Processed {} frames", frame_count);
            }
        }

        // Create a simple text file with metadata and instructions
        let readme_path = output_dir.join("README.txt");
        let readme_content = format!(
            "Background Removal Video Processing Results\n\
             ==========================================\n\
             \n\
             Original video: {}\n\
             Duration: {:.2}s\n\
             Resolution: {}x{}\n\
             FPS: {:.2}\n\
             Codec: {}\n\
             Has Audio: {}\n\
             Frames Processed: {}\n\
             \n\
             FRAME EXPORT COMPLETE\n\
             ---------------------\n\
             This directory contains individual processed frames with background removal applied.\n\
             Each frame has been processed through the AI model to remove the background.\n\
             \n\
             TO CREATE A VIDEO:\n\
             Use FFmpeg to reassemble these frames into a video:\n\
             \n\
             # Basic video creation (no audio):\n\
             ffmpeg -framerate {:.2} -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4\n\
             \n\
             # With higher quality settings:\n\
             ffmpeg -framerate {:.2} -i frame_%06d.png -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p output.mp4\n\
             \n\
             # To preserve transparency (WebM format):\n\
             ffmpeg -framerate {:.2} -i frame_%06d.png -c:v libvpx-vp9 -pix_fmt yuva420p output.webm\n\
             \n\
             For more advanced options, refer to FFmpeg documentation.\n\
             Full video encoding will be implemented in a future update of this tool.\n",
            output_path.display(),
            metadata.duration,
            metadata.width,
            metadata.height,
            metadata.fps,
            metadata.codec,
            metadata.has_audio,
            frame_count,
            metadata.fps,
            metadata.fps,
            metadata.fps
        );

        std::fs::write(&readme_path, readme_content).map_err(|e| {
            BgRemovalError::processing(format!("Failed to write README file: {}", e))
        })?;

        // Create a placeholder message in the output file
        let info_message = format!(
            "Background removal video processing completed.\n\
             {} frames were processed and saved to: {}\n\
             \n\
             See the README.txt file in that directory for instructions on creating a video.\n\
             \n\
             Full video encoding will be implemented in a future update.\n",
            frame_count,
            output_dir.display()
        );

        std::fs::write(output_path, info_message).map_err(|e| {
            BgRemovalError::processing(format!("Failed to write output file: {}", e))
        })?;

        log::info!(
            "Video processing completed: {} frames saved to {}",
            frame_count,
            output_dir.display()
        );
        log::info!(
            "Check {} for instructions on creating a video from the processed frames",
            readme_path.display()
        );

        if preserve_audio && metadata.has_audio {
            log::warn!("Audio preservation not yet implemented - audio track will be lost");
        }

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
        let format = FFmpegBackend::detect_video_format(input_path)?;
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
