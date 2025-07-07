//! FFmpeg-based video processing backend
//!
//! This module provides video processing capabilities using FFmpeg for frame extraction,
//! processing, and video reassembly.

use crate::backends::video::{FrameStream, VideoBackend, VideoFormat, VideoFrame, VideoMetadata};
use crate::error::{BgRemovalError, Result};
use async_trait::async_trait;
use futures::StreamExt;
use image::RgbaImage;
use log::{info, warn};
use std::path::Path;
use std::time::Duration;

// Re-export ffmpeg-next as ffmpeg for simpler usage
use ffmpeg_next as ffmpeg;

/// FFmpeg-based video processing backend
pub struct FfmpegBackend;

impl FfmpegBackend {
    /// Create a new FFmpeg backend
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl Default for FfmpegBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create FFmpeg backend")
    }
}

#[async_trait]
impl VideoBackend for FfmpegBackend {
    async fn extract_frames(&self, input_path: &Path) -> Result<FrameStream> {
        info!("Extracting frames from {}", input_path.display());

        // Initialize FFmpeg
        ffmpeg::init().map_err(|e| {
            BgRemovalError::processing(format!("Failed to initialize FFmpeg: {}", e))
        })?;

        // Open input file
        let mut input = ffmpeg::format::input(&input_path).map_err(|e| {
            BgRemovalError::processing(format!("Failed to open input video: {}", e))
        })?;

        // Find the best video stream
        let video_stream_index = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| BgRemovalError::processing("No video stream found".to_string()))?
            .index();

        // Get video decoder
        let video_stream = input.stream(video_stream_index).unwrap();
        let time_base = video_stream.time_base(); // Get time_base before mutable borrow
        let decoder_context =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters()).map_err(
                |e| BgRemovalError::processing(format!("Failed to create decoder context: {}", e)),
            )?;
        let mut decoder = decoder_context.decoder().video().map_err(|e| {
            BgRemovalError::processing(format!("Failed to create video decoder: {}", e))
        })?;

        // Extract all frames synchronously first
        let mut frames = Vec::new();

        // Use better scaling flags for higher quality
        let scaling_flags = ffmpeg::software::scaling::Flags::LANCZOS
            | ffmpeg::software::scaling::Flags::ACCURATE_RND
            | ffmpeg::software::scaling::Flags::FULL_CHR_H_INT;

        let mut scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::RGBA,
            decoder.width(),
            decoder.height(),
            scaling_flags,
        )
        .map_err(|e| BgRemovalError::processing(format!("Failed to create scaler: {}", e)))?;

        for (stream, packet) in input.packets() {
            if stream.index() == video_stream_index {
                info!("Processing packet for stream {}", stream.index());
                decoder.send_packet(&packet).map_err(|e| {
                    BgRemovalError::processing(format!("Failed to send packet to decoder: {}", e))
                })?;

                let mut decoded_frame = ffmpeg::frame::Video::empty();
                while decoder.receive_frame(&mut decoded_frame).is_ok() {
                    info!("Received frame from decoder");
                    let mut rgba_frame = ffmpeg::frame::Video::empty();
                    scaler.run(&decoded_frame, &mut rgba_frame).map_err(|e| {
                        BgRemovalError::processing(format!("Failed to scale frame: {}", e))
                    })?;

                    // Convert FFmpeg frame to our VideoFrame
                    let width = rgba_frame.width();
                    let height = rgba_frame.height();
                    let data = rgba_frame.data(0);
                    let stride = rgba_frame.stride(0) as usize;

                    // Validate frame dimensions
                    if width == 0 || height == 0 {
                        warn!(
                            "Skipping frame with invalid dimensions: {}x{}",
                            width, height
                        );
                        continue;
                    }

                    let row_bytes = (width * 4) as usize; // 4 bytes per RGBA pixel
                    let mut image_data = vec![0u8; (width * height * 4) as usize];

                    // Copy row by row, handling stride properly
                    for y in 0..height {
                        let src_row_start = (y as usize) * stride;
                        let src_row_end = src_row_start + row_bytes;
                        let dst_row_start = (y as usize) * row_bytes;
                        let dst_row_end = dst_row_start + row_bytes;

                        // Validate bounds before copying
                        if src_row_end <= data.len() && dst_row_end <= image_data.len() {
                            image_data[dst_row_start..dst_row_end]
                                .copy_from_slice(&data[src_row_start..src_row_end]);
                        } else {
                            warn!("Frame data bounds check failed at row {}: src_end={}, data_len={}, dst_end={}, image_len={}", 
                                  y, src_row_end, data.len(), dst_row_end, image_data.len());
                            // Fill with transparent pixels for safety
                            if dst_row_end <= image_data.len() {
                                for i in (dst_row_start..dst_row_end).step_by(4) {
                                    if i + 3 < image_data.len() {
                                        image_data[i] = 0; // R
                                        image_data[i + 1] = 0; // G
                                        image_data[i + 2] = 0; // B
                                        image_data[i + 3] = 0; // A (transparent)
                                    }
                                }
                            }
                        }
                    }

                    let rgba_image =
                        RgbaImage::from_raw(width, height, image_data).ok_or_else(|| {
                            BgRemovalError::processing("Failed to create RGBA image".to_string())
                        })?;

                    let frame_number = frames.len() as u64;

                    // Calculate proper timestamp from PTS and time base
                    let pts = decoded_frame.pts().unwrap_or(0);
                    let timestamp_seconds =
                        pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64;
                    let timestamp = Duration::from_secs_f64(timestamp_seconds.max(0.0));

                    let video_frame = VideoFrame {
                        image: rgba_image,
                        frame_number,
                        timestamp,
                        width,
                        height,
                    };

                    // Validate frame before adding to collection
                    if let Err(validation_error) = video_frame.validate() {
                        warn!(
                            "Frame {} validation failed: {}",
                            frame_number, validation_error
                        );
                        continue;
                    }

                    if video_frame.is_likely_corrupted() {
                        warn!("Frame {} appears to be corrupted, skipping", frame_number);
                        continue;
                    }

                    frames.push(Ok(video_frame));
                }
            }
        }

        // Flush decoder
        decoder.send_eof().ok();
        let mut decoded_frame = ffmpeg::frame::Video::empty();
        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            let mut rgba_frame = ffmpeg::frame::Video::empty();
            scaler
                .run(&decoded_frame, &mut rgba_frame)
                .map_err(|e| BgRemovalError::processing(format!("Failed to scale frame: {}", e)))?;

            // Convert final frames with proper stride handling
            let width = rgba_frame.width();
            let height = rgba_frame.height();
            let data = rgba_frame.data(0);
            let stride = rgba_frame.stride(0) as usize;

            // Validate frame dimensions
            if width == 0 || height == 0 {
                warn!(
                    "Skipping final frame with invalid dimensions: {}x{}",
                    width, height
                );
                continue;
            }

            let row_bytes = (width * 4) as usize; // 4 bytes per RGBA pixel
            let mut image_data = vec![0u8; (width * height * 4) as usize];

            // Copy row by row, handling stride properly
            for y in 0..height {
                let src_row_start = (y as usize) * stride;
                let src_row_end = src_row_start + row_bytes;
                let dst_row_start = (y as usize) * row_bytes;
                let dst_row_end = dst_row_start + row_bytes;

                // Validate bounds before copying
                if src_row_end <= data.len() && dst_row_end <= image_data.len() {
                    image_data[dst_row_start..dst_row_end]
                        .copy_from_slice(&data[src_row_start..src_row_end]);
                } else {
                    warn!("Final frame data bounds check failed at row {}: src_end={}, data_len={}, dst_end={}, image_len={}", 
                          y, src_row_end, data.len(), dst_row_end, image_data.len());
                    // Fill with transparent pixels for safety
                    if dst_row_end <= image_data.len() {
                        for i in (dst_row_start..dst_row_end).step_by(4) {
                            if i + 3 < image_data.len() {
                                image_data[i] = 0; // R
                                image_data[i + 1] = 0; // G
                                image_data[i + 2] = 0; // B
                                image_data[i + 3] = 0; // A (transparent)
                            }
                        }
                    }
                }
            }

            let rgba_image = RgbaImage::from_raw(width, height, image_data).ok_or_else(|| {
                BgRemovalError::processing("Failed to create RGBA image".to_string())
            })?;

            let frame_number = frames.len() as u64;

            // Calculate proper timestamp from PTS and time base
            let pts = decoded_frame.pts().unwrap_or(0);
            let timestamp_seconds =
                pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64;
            let timestamp = Duration::from_secs_f64(timestamp_seconds.max(0.0));

            let video_frame = VideoFrame {
                image: rgba_image,
                frame_number,
                timestamp,
                width,
                height,
            };

            // Validate frame before adding to collection
            if let Err(validation_error) = video_frame.validate() {
                warn!(
                    "Final frame {} validation failed: {}",
                    frame_number, validation_error
                );
                continue;
            }

            if video_frame.is_likely_corrupted() {
                warn!(
                    "Final frame {} appears to be corrupted, skipping",
                    frame_number
                );
                continue;
            }

            frames.push(Ok(video_frame));
        }

        info!("Extracted {} frames", frames.len());

        // Convert to async stream
        let stream = futures::stream::iter(frames);
        Ok(Box::pin(stream))
    }

    async fn reassemble_video(
        &self,
        frames: FrameStream,
        output_path: &Path,
        metadata: &VideoMetadata,
        preserve_audio: bool,
    ) -> Result<()> {
        // Collect frames first to avoid borrow checker issues with async
        let mut collected_frames = Vec::new();
        let mut frames_pin = std::pin::pin!(frames);

        while let Some(frame_result) = StreamExt::next(&mut frames_pin).await {
            collected_frames.push(frame_result?);
        }

        // Run encoding in blocking task to avoid Send issues with FFmpeg
        let output_path = output_path.to_owned();
        let metadata = metadata.clone();

        tokio::task::spawn_blocking(move || {
            Self::encode_frames_sync(collected_frames, &output_path, &metadata, preserve_audio)
        })
        .await
        .map_err(|e| BgRemovalError::processing(format!("Task join error: {}", e)))??;

        Ok(())
    }

    async fn get_metadata(&self, input_path: &Path) -> Result<VideoMetadata> {
        info!("Getting metadata for {}", input_path.display());

        // Initialize FFmpeg
        ffmpeg::init().map_err(|e| {
            BgRemovalError::processing(format!("Failed to initialize FFmpeg: {}", e))
        })?;

        // Open input file
        let input = ffmpeg::format::input(&input_path).map_err(|e| {
            BgRemovalError::processing(format!("Failed to open input video: {}", e))
        })?;

        // Find the best video stream
        let video_stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| BgRemovalError::processing("No video stream found".to_string()))?;

        // Extract metadata
        let duration = input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);

        // Get actual video dimensions from decoder parameters
        let decoder_context =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters()).map_err(
                |e| BgRemovalError::processing(format!("Failed to create decoder context: {}", e)),
            )?;
        let decoder = decoder_context.decoder().video().map_err(|e| {
            BgRemovalError::processing(format!("Failed to create video decoder: {}", e))
        })?;

        // Extract actual dimensions from decoder
        let width = decoder.width();
        let height = decoder.height();

        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(BgRemovalError::processing(format!(
                "Invalid video dimensions: {}x{}",
                width, height
            )));
        }

        let fps = video_stream.avg_frame_rate().numerator() as f64
            / video_stream.avg_frame_rate().denominator() as f64;

        // Detect format from file extension
        let format = input_path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(VideoFormat::from_extension)
            .unwrap_or(VideoFormat::Mp4);

        // Check for audio stream
        let has_audio = input
            .streams()
            .any(|s| s.parameters().medium() == ffmpeg::media::Type::Audio);

        // Get codec name
        let codec = video_stream.parameters().id().name().to_string();

        // Estimate bitrate
        let bitrate = input.bit_rate() as u64;

        info!(
            "Video metadata extracted: {}x{} at {:.2} fps, duration: {:.2}s, codec: {}",
            width, height, fps, duration, codec
        );

        Ok(VideoMetadata {
            duration,
            width,
            height,
            fps,
            format,
            codec,
            bitrate: if bitrate > 0 { Some(bitrate) } else { None },
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

impl FfmpegBackend {
    fn encode_frames_sync(
        frames: Vec<VideoFrame>,
        output_path: &Path,
        metadata: &VideoMetadata,
        preserve_audio: bool,
    ) -> Result<()> {
        info!(
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

        // Initialize FFmpeg globally
        ffmpeg::init().map_err(|e| {
            BgRemovalError::processing(format!("Failed to initialize FFmpeg: {}", e))
        })?;

        // Create output format and context
        let output_format_name = match metadata.format {
            VideoFormat::Mp4 => "mp4",
            VideoFormat::Avi => "avi",
            VideoFormat::Mov => "mov",
            VideoFormat::Mkv => "matroska",
            VideoFormat::WebM => "webm",
        };

        let mut output =
            ffmpeg::format::output_as(&output_path, output_format_name).map_err(|e| {
                BgRemovalError::processing(format!("Failed to create output format: {}", e))
            })?;

        // Set up video encoder
        let codec_id = match metadata.format {
            VideoFormat::Mp4 | VideoFormat::Mov | VideoFormat::Avi => ffmpeg::codec::Id::H264,
            VideoFormat::Mkv => ffmpeg::codec::Id::H264,
            VideoFormat::WebM => ffmpeg::codec::Id::VP9,
        };

        let codec = ffmpeg::encoder::find(codec_id)
            .ok_or_else(|| BgRemovalError::processing(format!("Codec {:?} not found", codec_id)))?;

        let pixel_format = match codec_id {
            ffmpeg::codec::Id::H264 => ffmpeg::format::Pixel::YUV420P,
            ffmpeg::codec::Id::VP9 => ffmpeg::format::Pixel::YUVA420P,
            _ => ffmpeg::format::Pixel::YUV420P,
        };

        let mut video_encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|e| {
                BgRemovalError::processing(format!("Failed to create video encoder context: {}", e))
            })?;

        // Configure encoder
        video_encoder.set_width(metadata.width);
        video_encoder.set_height(metadata.height);
        // Use a high-resolution time base for precise timing (90kHz is common for video)
        video_encoder.set_time_base(ffmpeg::Rational::new(1, 90000));
        video_encoder.set_format(pixel_format);
        video_encoder.set_bit_rate(metadata.bitrate.unwrap_or(1_000_000) as usize);

        let mut video_encoder = video_encoder.open_as(codec).map_err(|e| {
            BgRemovalError::processing(format!("Failed to open video encoder: {}", e))
        })?;

        // Add video stream to output
        {
            let mut video_stream = output.add_stream(codec).map_err(|e| {
                BgRemovalError::processing(format!("Failed to add video stream: {}", e))
            })?;
            video_stream.set_parameters(&video_encoder);
            // Set stream time base to match encoder time base
            video_stream.set_time_base(ffmpeg::Rational::new(1, 90000));
        }

        // Write output header
        output.write_header().map_err(|e| {
            BgRemovalError::processing(format!("Failed to write output header: {}", e))
        })?;

        // Create scaling context for converting RGBA to encoder format
        // Use better scaling flags for higher quality during encoding
        let encoding_scaling_flags = ffmpeg::software::scaling::Flags::LANCZOS
            | ffmpeg::software::scaling::Flags::ACCURATE_RND
            | ffmpeg::software::scaling::Flags::FULL_CHR_H_INT;

        let mut scaler = ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGBA,
            metadata.width,
            metadata.height,
            pixel_format,
            metadata.width,
            metadata.height,
            encoding_scaling_flags,
        )
        .map_err(|e| BgRemovalError::processing(format!("Failed to create scaler: {}", e)))?;

        // Process frames and encode video
        for (frame_count, video_frame) in frames.iter().enumerate() {
            // Create FFmpeg frame from our VideoFrame
            let mut rgba_frame = ffmpeg::util::frame::video::Video::new(
                ffmpeg::format::Pixel::RGBA,
                metadata.width,
                metadata.height,
            );

            // Copy image data to FFmpeg frame with proper stride handling
            let image_data = video_frame.image.as_raw();
            let stride = rgba_frame.stride(0) as usize;
            let row_bytes = (metadata.width * 4) as usize; // 4 bytes per RGBA pixel

            {
                let frame_data = rgba_frame.data_mut(0);
                for y in 0..metadata.height {
                    let src_row_start = (y as usize) * row_bytes;
                    let src_row_end = src_row_start + row_bytes;
                    let dst_row_start = (y as usize) * stride;
                    let dst_row_end = dst_row_start + row_bytes;

                    // Validate bounds and copy row data
                    if src_row_end <= image_data.len() && dst_row_end <= frame_data.len() {
                        frame_data[dst_row_start..dst_row_end]
                            .copy_from_slice(&image_data[src_row_start..src_row_end]);
                    } else {
                        warn!("Encoding frame data bounds check failed at row {}: src_end={}, image_len={}, dst_end={}, frame_len={}", 
                              y, src_row_end, image_data.len(), dst_row_end, frame_data.len());
                        // Fill with black transparent pixels for safety
                        if dst_row_end <= frame_data.len() {
                            for i in (dst_row_start..dst_row_end).step_by(4) {
                                if i + 3 < frame_data.len() {
                                    frame_data[i] = 0; // R
                                    frame_data[i + 1] = 0; // G
                                    frame_data[i + 2] = 0; // B
                                    frame_data[i + 3] = 0; // A (transparent)
                                }
                            }
                        }
                    }
                }
            }

            // Calculate PTS based on frame count and frame rate
            // Time base is 1/90000, so convert frame timing to 90kHz units
            // Frame duration in seconds = 1/fps, in 90kHz units = 90000/fps
            let frame_duration_in_timebase = (90000.0 / metadata.fps) as i64;
            let pts = frame_count as i64 * frame_duration_in_timebase;

            // Set frame timestamp
            rgba_frame.set_pts(Some(pts));

            // Create output frame for encoder
            let mut output_frame = ffmpeg::util::frame::video::Video::empty();
            output_frame.set_format(pixel_format);
            output_frame.set_width(metadata.width);
            output_frame.set_height(metadata.height);

            // Scale RGBA to encoder format
            scaler
                .run(&rgba_frame, &mut output_frame)
                .map_err(|e| BgRemovalError::processing(format!("Failed to scale frame: {}", e)))?;

            // Use the frame-based PTS that matches our time base
            output_frame.set_pts(Some(pts));

            // Encode frame
            video_encoder.send_frame(&output_frame).map_err(|e| {
                BgRemovalError::processing(format!("Failed to send frame to encoder: {}", e))
            })?;

            // Receive encoded packets
            let mut encoded_packet = ffmpeg::Packet::empty();
            while video_encoder.receive_packet(&mut encoded_packet).is_ok() {
                encoded_packet.set_stream(0); // First stream
                encoded_packet.write_interleaved(&mut output).map_err(|e| {
                    BgRemovalError::processing(format!("Failed to write packet: {}", e))
                })?;
            }
        }

        // Flush encoder
        video_encoder.send_eof().map_err(|e| {
            BgRemovalError::processing(format!("Failed to send EOF to encoder: {}", e))
        })?;

        // Receive any remaining packets
        let mut encoded_packet = ffmpeg::Packet::empty();
        while video_encoder.receive_packet(&mut encoded_packet).is_ok() {
            encoded_packet.set_stream(0);
            encoded_packet.write_interleaved(&mut output).map_err(|e| {
                BgRemovalError::processing(format!("Failed to write final packet: {}", e))
            })?;
        }

        // Write trailer
        output
            .write_trailer()
            .map_err(|e| BgRemovalError::processing(format!("Failed to write trailer: {}", e)))?;

        info!(
            "Successfully encoded {} frames to {}",
            frames.len(),
            output_path.display()
        );

        // Handle audio preservation if requested
        if preserve_audio && metadata.has_audio {
            warn!("Audio preservation not yet implemented - output video will have no audio");
        }

        Ok(())
    }
}
