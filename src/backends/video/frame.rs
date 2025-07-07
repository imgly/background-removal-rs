//! Video frame data structures and utilities
//!
//! This module defines the core data structures for representing video frames
//! and their metadata, as well as utilities for frame processing.

use image::{DynamicImage, RgbaImage};
use std::time::Duration;

/// Represents a single video frame with its metadata
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame image data
    pub image: RgbaImage,
    /// Frame number in the video sequence
    pub frame_number: u64,
    /// Timestamp of this frame in the video
    pub timestamp: Duration,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
}

impl VideoFrame {
    /// Create a new video frame
    ///
    /// # Arguments
    /// * `image` - The frame image data as RGBA
    /// * `frame_number` - Sequential frame number
    /// * `timestamp` - Frame timestamp in the video
    pub fn new(image: RgbaImage, frame_number: u64, timestamp: Duration) -> Self {
        let (width, height) = image.dimensions();
        Self {
            image,
            frame_number,
            timestamp,
            width,
            height,
        }
    }

    /// Validate frame data integrity
    ///
    /// Checks for common issues that could cause frame distortion:
    /// - Non-zero dimensions
    /// - Valid image data size
    /// - Reasonable timestamp values
    pub fn validate(&self) -> Result<(), String> {
        // Check dimensions
        if self.width == 0 || self.height == 0 {
            return Err(format!("Invalid frame dimensions: {}x{}", self.width, self.height));
        }

        // Check if dimensions are too large (could cause memory issues)
        if self.width > 8192 || self.height > 8192 {
            return Err(format!("Frame dimensions too large: {}x{} (max 8192x8192)", self.width, self.height));
        }

        // Check if image data size matches expected size
        let expected_size = (self.width * self.height * 4) as usize; // 4 bytes per RGBA pixel
        let actual_size = self.image.as_raw().len();
        if actual_size != expected_size {
            return Err(format!(
                "Frame data size mismatch: expected {} bytes, got {} bytes", 
                expected_size, actual_size
            ));
        }

        // Check timestamp is reasonable (not negative, not extremely large)
        if self.timestamp.as_secs() > 86400 { // More than 24 hours
            return Err(format!("Unreasonable timestamp: {:?}", self.timestamp));
        }

        Ok(())
    }

    /// Check if frame appears to be corrupted
    ///
    /// Performs basic corruption detection by checking for common patterns
    /// that indicate frame processing issues. This is conservative to avoid
    /// false positives with background-removed images.
    pub fn is_likely_corrupted(&self) -> bool {
        let data = self.image.as_raw();
        let total_pixels = (self.width * self.height) as usize;
        
        if data.is_empty() || total_pixels == 0 {
            return true;
        }

        // Only consider completely zero pixels as corruption if ALL pixels are zero
        // This is more conservative since background removal creates many transparent pixels
        let zero_pixels = data.chunks_exact(4)
            .filter(|pixel| pixel == &[0, 0, 0, 0])
            .count();
        
        // Only flag as corrupted if 100% of pixels are zero (empty frame)
        if zero_pixels == total_pixels {
            return true;
        }

        // Check for data corruption patterns (invalid color values)
        // Look for patterns that indicate memory corruption, not valid transparent content
        let mut invalid_count = 0;
        for chunk in data.chunks_exact(4) {
            // Check for specific corruption patterns that are unlikely in valid frames
            // - All channels at maximum with alpha at 0 (impossible combination)  
            // - Repeating byte patterns that indicate memory corruption
            if chunk[0] == 255 && chunk[1] == 255 && chunk[2] == 255 && chunk[3] == 0 {
                invalid_count += 1;
            }
        }

        // If more than 50% of pixels show corruption patterns, likely corrupted
        invalid_count > (total_pixels / 2)
    }

    /// Convert frame to DynamicImage for processing
    pub fn to_dynamic_image(&self) -> DynamicImage {
        DynamicImage::ImageRgba8(self.image.clone())
    }

    /// Create frame from DynamicImage
    ///
    /// # Arguments
    /// * `image` - Source image
    /// * `frame_number` - Sequential frame number  
    /// * `timestamp` - Frame timestamp in the video
    pub fn from_dynamic_image(image: DynamicImage, frame_number: u64, timestamp: Duration) -> Self {
        let rgba_image = image.to_rgba8();
        Self::new(rgba_image, frame_number, timestamp)
    }

    /// Get frame dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get frame timestamp in seconds
    pub fn timestamp_seconds(&self) -> f64 {
        self.timestamp.as_secs_f64()
    }

    /// Calculate frame rate from frame number and timestamp
    pub fn calculate_fps(&self) -> f64 {
        if self.timestamp.as_secs_f64() > 0.0 {
            self.frame_number as f64 / self.timestamp_seconds()
        } else {
            0.0
        }
    }
}

/// Frame processing statistics
#[derive(Debug, Clone, Default)]
pub struct FrameProcessingStats {
    /// Total number of frames processed
    pub frames_processed: u64,
    /// Total processing time for all frames
    pub total_processing_time: Duration,
    /// Average processing time per frame
    pub average_frame_time: Duration,
    /// Number of frames that failed processing
    pub failed_frames: u64,
    /// Number of frames skipped due to errors
    pub skipped_frames: u64,
}

impl FrameProcessingStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Add processing time for a frame
    pub fn add_frame_time(&mut self, processing_time: Duration) {
        self.frames_processed += 1;
        self.total_processing_time += processing_time;

        if self.frames_processed > 0 {
            self.average_frame_time = self.total_processing_time / self.frames_processed as u32;
        }
    }

    /// Mark a frame as failed
    pub fn mark_frame_failed(&mut self) {
        self.failed_frames += 1;
    }

    /// Mark a frame as skipped
    pub fn mark_frame_skipped(&mut self) {
        self.skipped_frames += 1;
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        let total = self.frames_processed + self.failed_frames + self.skipped_frames;
        if total > 0 {
            (self.frames_processed as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get processing speed in frames per second
    pub fn processing_fps(&self) -> f64 {
        if self.total_processing_time.as_secs_f64() > 0.0 {
            self.frames_processed as f64 / self.total_processing_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Frame batch for efficient processing
#[derive(Debug)]
pub struct FrameBatch {
    /// Frames in this batch
    pub frames: Vec<VideoFrame>,
    /// Batch size
    pub size: usize,
    /// Batch number in the processing sequence
    pub batch_number: u64,
}

impl FrameBatch {
    /// Create a new frame batch
    pub fn new(frames: Vec<VideoFrame>, batch_number: u64) -> Self {
        let size = frames.len();
        Self {
            frames,
            size,
            batch_number,
        }
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Get frame count in batch
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Get frame by index
    pub fn get_frame(&self, index: usize) -> Option<&VideoFrame> {
        self.frames.get(index)
    }

    /// Extract images for batch processing
    pub fn extract_images(&self) -> Vec<DynamicImage> {
        self.frames.iter().map(|f| f.to_dynamic_image()).collect()
    }

    /// Create batch from frame iterator
    pub fn from_frames<I>(frames: I, batch_number: u64) -> Self
    where
        I: IntoIterator<Item = VideoFrame>,
    {
        let frames: Vec<_> = frames.into_iter().collect();
        Self::new(frames, batch_number)
    }
}

/// Iterator for processing frames in batches
pub struct FrameBatchIterator {
    frames: Vec<VideoFrame>,
    batch_size: usize,
    current_batch: u64,
    current_index: usize,
}

impl FrameBatchIterator {
    /// Create new batch iterator
    ///
    /// # Arguments
    /// * `frames` - All frames to process
    /// * `batch_size` - Number of frames per batch
    pub fn new(frames: Vec<VideoFrame>, batch_size: usize) -> Self {
        Self {
            frames,
            batch_size,
            current_batch: 0,
            current_index: 0,
        }
    }
}

impl Iterator for FrameBatchIterator {
    type Item = FrameBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.frames.len() {
            return None;
        }

        let end_index = std::cmp::min(self.current_index + self.batch_size, self.frames.len());

        let batch_frames = self.frames[self.current_index..end_index].to_vec();
        let batch = FrameBatch::new(batch_frames, self.current_batch);

        self.current_index = end_index;
        self.current_batch += 1;

        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, Rgba};

    #[test]
    fn test_video_frame_creation() {
        let image = RgbaImage::from_pixel(100, 100, Rgba([255, 0, 0, 255]));
        let frame = VideoFrame::new(image, 42, Duration::from_secs(5));

        assert_eq!(frame.frame_number, 42);
        assert_eq!(frame.timestamp, Duration::from_secs(5));
        assert_eq!(frame.dimensions(), (100, 100));
        assert_eq!(frame.timestamp_seconds(), 5.0);
    }

    #[test]
    fn test_frame_processing_stats() {
        let mut stats = FrameProcessingStats::new();

        stats.add_frame_time(Duration::from_millis(100));
        stats.add_frame_time(Duration::from_millis(200));
        stats.mark_frame_failed();

        assert_eq!(stats.frames_processed, 2);
        assert_eq!(stats.failed_frames, 1);
        assert_eq!(stats.total_processing_time, Duration::from_millis(300));
        assert_eq!(stats.average_frame_time, Duration::from_millis(150));

        let expected_success_rate = (2.0 / 3.0) * 100.0;
        assert!((stats.success_rate() - expected_success_rate).abs() < 0.01);
    }

    #[test]
    fn test_frame_batch() {
        let frames = vec![
            VideoFrame::new(RgbaImage::new(10, 10), 1, Duration::from_millis(33)),
            VideoFrame::new(RgbaImage::new(10, 10), 2, Duration::from_millis(66)),
        ];

        let batch = FrameBatch::new(frames, 0);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch.batch_number, 0);
    }

    #[test]
    fn test_frame_batch_iterator() {
        let frames = (0..5)
            .map(|i| VideoFrame::new(RgbaImage::new(10, 10), i, Duration::from_millis(i * 33)))
            .collect();

        let mut iter = FrameBatchIterator::new(frames, 2);

        let batch1 = iter.next().unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1.batch_number, 0);

        let batch2 = iter.next().unwrap();
        assert_eq!(batch2.len(), 2);
        assert_eq!(batch2.batch_number, 1);

        let batch3 = iter.next().unwrap();
        assert_eq!(batch3.len(), 1);
        assert_eq!(batch3.batch_number, 2);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dynamic_image_conversion() {
        let rgba_image = RgbaImage::from_pixel(50, 50, Rgba([0, 255, 0, 255]));
        let frame = VideoFrame::new(rgba_image, 1, Duration::from_secs(1));

        let dynamic = frame.to_dynamic_image();
        assert_eq!(dynamic.dimensions(), (50, 50));

        let frame_from_dynamic = VideoFrame::from_dynamic_image(dynamic, 2, Duration::from_secs(2));
        assert_eq!(frame_from_dynamic.frame_number, 2);
        assert_eq!(frame_from_dynamic.dimensions(), (50, 50));
    }
}
