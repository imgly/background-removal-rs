//! Core types for background removal operations

use crate::{
    config::OutputFormat,
    error::{BgRemovalError, Result},
};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use log;
use serde::{Deserialize, Serialize};
use std::fmt::Write;
use std::path::Path;

// Use instant crate for cross-platform time compatibility
use instant::Instant;

#[cfg(feature = "video-support")]
use crate::backends::video::{FrameProcessingStats, VideoMetadata};

/// ICC color profile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorProfile {
    /// Raw ICC profile data
    pub icc_data: Option<Vec<u8>>,
    /// Detected color space
    pub color_space: ColorSpace,
}

impl ColorProfile {
    /// Create a new color profile
    #[must_use]
    pub fn new(icc_data: Option<Vec<u8>>, color_space: ColorSpace) -> Self {
        Self {
            icc_data,
            color_space,
        }
    }

    /// Create a color profile from raw ICC data
    #[must_use]
    pub fn from_icc_data(icc_data: Vec<u8>) -> Self {
        let color_space = Self::detect_color_space_from_data(&icc_data);
        Self {
            icc_data: Some(icc_data),
            color_space,
        }
    }

    /// Detect color space from ICC profile data using basic heuristics
    fn detect_color_space_from_data(icc_data: &[u8]) -> ColorSpace {
        let data_str = String::from_utf8_lossy(icc_data);

        if data_str.contains("sRGB") || data_str.contains("srgb") {
            ColorSpace::Srgb
        } else if data_str.contains("Adobe RGB") || data_str.contains("ADBE") {
            ColorSpace::AdobeRgb
        } else if data_str.contains("Display P3") || data_str.contains("APPL") {
            ColorSpace::DisplayP3
        } else if data_str.contains("ProPhoto") || data_str.contains("ROMM") {
            ColorSpace::ProPhotoRgb
        } else {
            ColorSpace::Unknown("ICC Present".to_string())
        }
    }

    /// Get the size of ICC profile data in bytes
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.icc_data.as_ref().map_or(0, Vec::len)
    }

    /// Check if this color profile has ICC data
    #[must_use]
    pub fn has_color_profile(&self) -> bool {
        self.icc_data.is_some()
    }
}

/// Color space enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// Standard RGB color space (sRGB)
    Srgb,
    /// Adobe RGB color space (wider gamut)
    AdobeRgb,
    /// Apple Display P3 color space
    DisplayP3,
    /// `ProPhoto` RGB color space (very wide gamut)
    ProPhotoRgb,
    /// Unknown or unsupported color space
    Unknown(String),
}

impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorSpace::Srgb => write!(f, "sRGB"),
            ColorSpace::AdobeRgb => write!(f, "Adobe RGB"),
            ColorSpace::DisplayP3 => write!(f, "Display P3"),
            ColorSpace::ProPhotoRgb => write!(f, "ProPhoto RGB"),
            ColorSpace::Unknown(desc) => write!(f, "Unknown ({desc})"),
        }
    }
}

/// Result of a background removal operation
#[derive(Debug, Clone)]
pub struct RemovalResult {
    /// The processed image with background removed
    pub image: DynamicImage,

    /// The segmentation mask used for removal
    pub mask: SegmentationMask,

    /// Original image dimensions
    pub original_dimensions: (u32, u32),

    /// Processing metadata
    pub metadata: ProcessingMetadata,

    /// Original input path (for logging purposes)
    pub input_path: Option<String>,

    /// ICC color profile from the original image
    pub color_profile: Option<ColorProfile>,
}

impl RemovalResult {
    /// Create a new removal result
    #[must_use]
    pub fn new(
        image: DynamicImage,
        mask: SegmentationMask,
        original_dimensions: (u32, u32),
        metadata: ProcessingMetadata,
    ) -> Self {
        Self {
            image,
            mask,
            original_dimensions,
            metadata,
            input_path: None,
            color_profile: None,
        }
    }

    /// Create a new removal result with input path
    #[must_use]
    pub fn with_input_path(
        image: DynamicImage,
        mask: SegmentationMask,
        original_dimensions: (u32, u32),
        metadata: ProcessingMetadata,
        input_path: String,
    ) -> Self {
        Self {
            image,
            mask,
            original_dimensions,
            metadata,
            input_path: Some(input_path),
            color_profile: None,
        }
    }

    /// Create a new removal result with color profile
    #[must_use]
    pub fn with_color_profile(
        image: DynamicImage,
        mask: SegmentationMask,
        original_dimensions: (u32, u32),
        metadata: ProcessingMetadata,
        color_profile: Option<ColorProfile>,
    ) -> Self {
        Self {
            image,
            mask,
            original_dimensions,
            metadata,
            input_path: None,
            color_profile,
        }
    }

    /// Create a new removal result with input path and color profile
    #[must_use]
    pub fn with_input_path_and_profile(
        image: DynamicImage,
        mask: SegmentationMask,
        original_dimensions: (u32, u32),
        metadata: ProcessingMetadata,
        input_path: String,
        color_profile: Option<ColorProfile>,
    ) -> Self {
        Self {
            image,
            mask,
            original_dimensions,
            metadata,
            input_path: Some(input_path),
            color_profile,
        }
    }

    /// Save the result as PNG with full alpha channel transparency
    ///
    /// PNG format preserves the alpha channel created by background removal,
    /// resulting in a truly transparent background that works in web browsers,
    /// image editors, and other applications supporting transparency.
    ///
    /// # Arguments
    /// * `path` - Output file path (will be created or overwritten)
    ///
    /// # File Format
    /// - **Format**: PNG with RGBA channels
    /// - **Compression**: Lossless PNG compression
    /// - **Transparency**: Full alpha channel support
    /// - **Quality**: No quality loss (lossless format)
    ///
    /// # Use Cases
    /// - Web development (transparent images for overlays)
    /// - Graphic design (compositing in image editors)
    /// - Print materials (transparent logos and graphics)
    /// - Mobile apps (icons and UI elements)
    ///
    /// # Examples
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalConfig, remove_background_from_reader, ModelSpec, ModelSource};
    /// use tokio::fs::File;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let model_spec = ModelSpec {
    ///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///     variant: None,
    /// };
    /// let config = RemovalConfig::builder()
    ///     .model_spec(model_spec)
    ///     .build()?;
    /// let file = File::open("input.jpg").await?;
    /// let result = remove_background_from_reader(file, &config).await?;
    /// result.save_png("output.png")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `BgRemovalError` for file I/O errors, permission issues,
    /// or disk space problems.
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let png_bytes = self.to_bytes(OutputFormat::Png, 100)?;
        std::fs::write(path, png_bytes)
            .map_err(|e| BgRemovalError::processing(format!("Failed to write PNG file: {}", e)))?;
        Ok(())
    }

    /// Save the result as PNG with alpha channel and return encoding time
    ///
    /// # Errors
    /// - File I/O errors when writing output file
    /// - Permission errors or disk space issues
    /// - PNG encoding errors from underlying image library
    /// - Timing conversion errors if encoding time exceeds u64 range
    pub fn save_png_with_timing<P: AsRef<Path>>(&self, path: P) -> Result<u64> {
        let encode_start = Instant::now();
        self.image.save_with_format(path, image::ImageFormat::Png)?;
        encode_start
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| BgRemovalError::processing("PNG encoding time too large for u64"))
    }

    /// Save the result as JPEG with ICC profile support
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - JPEG encoding errors from underlying image library
    /// - Invalid quality parameter (though already validated in builder)
    pub fn save_jpeg<P: AsRef<Path>>(&self, path: P, quality: u8) -> Result<()> {
        let jpeg_bytes = self.to_bytes(OutputFormat::Jpeg, quality)?;
        std::fs::write(path, jpeg_bytes)
            .map_err(|e| BgRemovalError::processing(format!("Failed to write JPEG file: {}", e)))?;
        Ok(())
    }

    /// Save the result as WebP with RGBA transparency and ICC profile support
    ///
    /// # Errors
    /// - WebP encoding errors during compression
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or disk space issues
    /// - Invalid quality parameter (outside 0-100 range)
    pub fn save_webp<P: AsRef<Path>>(&self, path: P, quality: u8) -> Result<()> {
        let webp_bytes = self.to_bytes(OutputFormat::WebP, quality)?;
        std::fs::write(path, webp_bytes)
            .map_err(|e| BgRemovalError::processing(format!("Failed to write WebP file: {}", e)))?;
        Ok(())
    }

    /// Save the result as TIFF with RGBA transparency and lossless compression
    ///
    /// TIFF format preserves the alpha channel created by background removal,
    /// resulting in a truly transparent background with lossless compression.
    /// TIFF is ideal for high-quality archival and professional workflows.
    ///
    /// # Arguments
    /// * `path` - Output file path (will be created or overwritten)
    ///
    /// # File Format
    /// - **Format**: TIFF with RGBA channels
    /// - **Compression**: Lossless LZW or ZIP compression
    /// - **Transparency**: Full alpha channel support
    /// - **Quality**: No quality loss (lossless format)
    ///
    /// # Use Cases
    /// - Professional photography workflows
    /// - High-quality archival storage
    /// - Print production pipelines
    /// - Graphic design applications requiring lossless editing
    ///
    /// # Errors
    /// Returns `BgRemovalError` for file I/O errors, permission issues,
    /// or disk space problems.
    pub fn save_tiff<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let tiff_bytes = self.to_bytes(OutputFormat::Tiff, 100)?;
        std::fs::write(path, tiff_bytes)
            .map_err(|e| BgRemovalError::processing(format!("Failed to write TIFF file: {}", e)))?;
        Ok(())
    }

    /// Save in the specified format
    ///
    /// Automatically uses color profile-aware saving when color profiles are available.
    /// For legacy compatibility, this method preserves existing behavior while enabling
    /// color profile embedding when possible.
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - Image encoding errors specific to the chosen format:
    ///   - PNG: Compression or metadata errors
    ///   - JPEG: Invalid quality parameter or RGB conversion errors
    ///   - WebP: Encoding or compression failures
    ///   - RGBA8: Raw data writing errors
    /// - Color profile embedding errors when profiles are present
    pub fn save<P: AsRef<Path>>(&self, path: P, format: OutputFormat, quality: u8) -> Result<()> {
        // Use color profile-aware saving if available, otherwise fallback to standard saving
        if self.has_color_profile() {
            self.save_with_color_profile(path, format, quality)
        } else {
            match format {
                OutputFormat::Png => self.save_png(path),
                OutputFormat::Jpeg => self.save_jpeg(path, quality),
                OutputFormat::WebP => self.save_webp(path, quality),
                OutputFormat::Tiff => self.save_tiff(path),
                OutputFormat::Rgba8 => {
                    // For RGBA8 format, save the raw RGBA bytes
                    let rgba_image = self.image.to_rgba8();
                    std::fs::write(path, rgba_image.as_raw())?;
                    Ok(())
                },
            }
        }
    }

    /// Get the processed image as raw RGBA bytes
    ///
    /// Returns the image data as a flat vector of RGBA bytes suitable for
    /// direct use with graphics APIs, web frameworks, or custom image processing.
    ///
    /// # Format
    /// - **Layout**: Flat array of RGBA pixels
    /// - **Order**: Row-major, left-to-right, top-to-bottom
    /// - **Channels**: Red, Green, Blue, Alpha (4 bytes per pixel)
    /// - **Alpha**: 0 = transparent background, 255 = opaque foreground
    ///
    /// # Size
    /// The returned vector size is `width × height × 4` bytes.
    ///
    /// # Use Cases
    /// - **Web APIs**: Canvas `ImageData`, `WebGL` textures
    /// - **Game engines**: Texture loading, sprite processing
    /// - **Custom processing**: Direct pixel manipulation
    /// - **Memory-efficient**: No intermediate file encoding
    ///
    /// # Examples
    ///
    /// ## Web API integration
    /// ```rust,no_run
    /// use imgly_bgremove::{BackgroundRemovalProcessor, ProcessorConfigBuilder, ModelSpec, ModelSource, BackendType};
    /// use image::DynamicImage;
    ///
    /// # fn example(img: DynamicImage) -> anyhow::Result<()> {
    /// let config = ProcessorConfigBuilder::new()
    ///     .model_spec(ModelSpec {
    ///         source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///         variant: None,
    ///     })
    ///     .backend_type(BackendType::Onnx)
    ///     .build()?;
    /// let mut processor = BackgroundRemovalProcessor::new(config)?;
    /// let result = processor.process_image(&img)?;
    ///
    /// let rgba_bytes = result.to_rgba_bytes();
    /// let (width, height) = result.dimensions();
    ///
    /// // Use with web APIs
    /// // canvas.putImageData(rgba_bytes, width, height);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Custom pixel processing
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalConfig, remove_background_from_reader, ModelSpec, ModelSource};
    /// use tokio::fs::File;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let model_spec = ModelSpec {
    ///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///     variant: None,
    /// };
    /// let config = RemovalConfig::builder()
    ///     .model_spec(model_spec)
    ///     .build()?;
    /// let file = File::open("input.jpg").await?;
    /// let result = remove_background_from_reader(file, &config).await?;
    ///
    /// let rgba_bytes = result.to_rgba_bytes();
    /// let (width, height) = result.dimensions();
    ///
    /// // Process each pixel
    /// for chunk in rgba_bytes.chunks(4) {
    ///     let [r, g, b, a] = [chunk[0], chunk[1], chunk[2], chunk[3]];
    ///     // Custom processing logic
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        self.image.to_rgba8().into_raw()
    }

    /// Get the image as encoded bytes in the specified format
    ///
    /// # Errors
    /// - Image encoding errors specific to the chosen format:
    ///   - PNG: Compression or metadata encoding errors
    ///   - JPEG: Invalid quality parameter or RGB conversion errors
    ///   - WebP: Encoding or compression failures (currently falls back to PNG)
    ///   - RGBA8: Memory allocation errors for raw bytes
    /// - Memory allocation errors when creating output buffer
    /// - Image format conversion errors (e.g., RGBA to RGB for JPEG)
    pub fn to_bytes(&self, format: OutputFormat, quality: u8) -> Result<Vec<u8>> {
        match format {
            OutputFormat::Png => {
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                self.image.write_to(&mut cursor, image::ImageFormat::Png)?;
                Ok(buffer)
            },
            OutputFormat::Jpeg => {
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                let rgb_image = self.image.to_rgb8();
                let mut jpeg_encoder =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
                jpeg_encoder.encode_image(&rgb_image)?;
                Ok(buffer)
            },
            OutputFormat::WebP => Ok(self.encode_webp(quality)),
            OutputFormat::Tiff => {
                let mut buffer = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buffer);
                self.image.write_to(&mut cursor, image::ImageFormat::Tiff)?;
                Ok(buffer)
            },
            OutputFormat::Rgba8 => Ok(self.to_rgba_bytes()),
        }
    }

    /// Write result to an async writer stream
    ///
    /// This method writes the processed image to any async writer, making it suitable
    /// for streaming to network connections, files, or any other async destination.
    ///
    /// # Arguments
    /// * `writer` - Any type implementing `AsyncWrite + Unpin`
    /// * `format` - Output image format (PNG, JPEG, WebP, etc.)
    /// * `quality` - Quality setting for lossy formats (0-100)
    ///
    /// # Returns
    /// Number of bytes written to the stream
    ///
    /// # Examples
    ///
    /// ## Stream to file
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalResult, OutputFormat};
    /// use tokio::fs::File;
    ///
    /// # async fn example(result: RemovalResult) -> anyhow::Result<()> {
    /// let mut output_file = File::create("result.png").await?;
    /// let bytes_written = result.write_to(output_file, OutputFormat::Png, 100).await?;
    /// println!("Wrote {} bytes", bytes_written);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Stream to network
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalResult, OutputFormat};
    /// use tokio::net::TcpStream;
    ///
    /// # async fn example(result: RemovalResult, mut stream: TcpStream) -> anyhow::Result<()> {
    /// let bytes_written = result.write_to(&mut stream, OutputFormat::Jpeg, 90).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `BgRemovalError` for:
    /// - I/O errors when writing to the stream
    /// - Image encoding errors
    /// - Network errors when streaming over network connections
    pub async fn write_to<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        mut writer: W,
        format: OutputFormat,
        quality: u8,
    ) -> Result<u64> {
        use tokio::io::AsyncWriteExt;

        // Encode to bytes first (reuse existing logic)
        let bytes = self.to_bytes(format, quality)?;

        // Write to stream
        AsyncWriteExt::write_all(&mut writer, &bytes)
            .await
            .map_err(|e| BgRemovalError::processing(format!("Failed to write to stream: {}", e)))?;

        // Flush to ensure data is sent
        AsyncWriteExt::flush(&mut writer)
            .await
            .map_err(|e| BgRemovalError::processing(format!("Failed to flush stream: {}", e)))?;

        Ok(bytes.len() as u64)
    }

    /// Get image dimensions
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    /// Get detailed timing breakdown
    #[must_use]
    pub fn timings(&self) -> &ProcessingTimings {
        &self.metadata.timings
    }

    /// Save and measure encoding time (updates internal timing)
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - Image encoding errors specific to the chosen format
    /// - Timing conversion errors if encoding time exceeds u64 range
    /// - Path conversion errors for logging purposes
    #[allow(clippy::cast_precision_loss)] // Acceptable for timing display
    pub fn save_with_timing<P: AsRef<Path>>(
        &mut self,
        path: P,
        format: image::ImageFormat,
    ) -> Result<()> {
        let path_str = path.as_ref().display().to_string();
        let encode_start = Instant::now();
        self.image.save_with_format(&path, format)?;
        let encode_ms = encode_start
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| BgRemovalError::processing("Image encoding time too large for u64"))?;

        // Update the timings
        self.metadata.timings.image_encode_ms = Some(encode_ms);

        // Log encoding completion at debug level
        log::debug!("Image Encoding completed in {}ms", encode_ms);

        // Log final completion with total processing time
        let total_time_s = self.metadata.timings.total_ms as f64 / 1000.0;
        let input_path = self.input_path.as_deref().unwrap_or("input");
        log::debug!(
            "Processed: {} -> {} in {:.2}s",
            input_path,
            path_str,
            total_time_s
        );

        Ok(())
    }

    /// Save as PNG and measure encoding time
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - PNG encoding errors from underlying image library
    /// - Timing conversion errors if encoding time exceeds u64 range
    /// - Path conversion errors for logging purposes
    pub fn save_png_timed<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.save_with_timing(path, image::ImageFormat::Png)
    }

    /// Save as JPEG and measure encoding time
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - JPEG encoding errors from underlying image library
    /// - Timing conversion errors if encoding time exceeds u64 range
    /// - Path conversion errors for logging purposes
    pub fn save_jpeg_timed<P: AsRef<Path>>(&mut self, path: P, quality: u8) -> Result<()> {
        let path_str = path.as_ref().display().to_string();
        let encode_start = Instant::now();

        // Convert to RGB and apply background color for JPEG
        let rgb_image = self.image.to_rgb8();
        let mut jpeg_encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
            std::fs::File::create(&path)?,
            quality,
        );
        jpeg_encoder.encode_image(&rgb_image)?;

        let encode_ms = encode_start
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| BgRemovalError::processing("JPEG encoding time too large for u64"))?;

        // Update the timings
        self.metadata.timings.image_encode_ms = Some(encode_ms);

        // Log encoding completion at debug level
        log::debug!("Image Encoding completed in {}ms", encode_ms);

        // Log final completion with total processing time
        let total_time_s = self.metadata.timings.total_ms as f64 / 1000.0;
        let input_path = self.input_path.as_deref().unwrap_or("input");
        log::debug!(
            "Processed: {} -> {} in {:.2}s",
            input_path,
            path_str,
            total_time_s
        );

        Ok(())
    }

    /// Save as WebP and measure encoding time
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - WebP encoding errors from underlying implementation
    /// - Timing conversion errors if encoding time exceeds u64 range
    /// - Path conversion errors for logging purposes
    pub fn save_webp_timed<P: AsRef<Path>>(&mut self, path: P, quality: u8) -> Result<()> {
        let path_str = path.as_ref().display().to_string();
        let encode_start = Instant::now();

        // Use the existing WebP encoding method
        let webp_data = self.encode_webp(quality);
        std::fs::write(&path, webp_data)?;

        let encode_ms = encode_start
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| BgRemovalError::processing("WebP encoding time too large for u64"))?;

        // Update the timings
        self.metadata.timings.image_encode_ms = Some(encode_ms);

        // Log encoding completion at debug level
        log::debug!("Image Encoding completed in {}ms", encode_ms);

        // Log final completion with total processing time
        let total_time_s = self.metadata.timings.total_ms as f64 / 1000.0;
        let input_path = self.input_path.as_deref().unwrap_or("input");
        log::debug!(
            "Processed: {} -> {} in {:.2}s",
            input_path,
            path_str,
            total_time_s
        );

        Ok(())
    }

    /// Save as TIFF and measure encoding time
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - TIFF encoding errors from underlying image library
    /// - Timing conversion errors if encoding time exceeds u64 range
    /// - Path conversion errors for logging purposes
    pub fn save_tiff_timed<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_str = path.as_ref().display().to_string();
        let encode_start = Instant::now();

        // Save as TIFF with RGBA transparency
        self.image
            .save_with_format(&path, image::ImageFormat::Tiff)?;

        let encode_ms = encode_start
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| BgRemovalError::processing("TIFF encoding time too large for u64"))?;

        // Update the timings
        self.metadata.timings.image_encode_ms = Some(encode_ms);

        // Log encoding completion at debug level
        log::debug!("Image Encoding completed in {}ms", encode_ms);

        // Log final completion with total processing time
        let total_time_s = self.metadata.timings.total_ms as f64 / 1000.0;
        let input_path = self.input_path.as_deref().unwrap_or("input");
        log::debug!(
            "Processed: {} -> {} in {:.2}s",
            input_path,
            path_str,
            total_time_s
        );

        Ok(())
    }

    /// Save in the specified format with timing measurement
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - Image encoding errors specific to the chosen format
    /// - Timing conversion errors if encoding time exceeds u64 range
    /// - Path conversion errors for logging purposes
    pub fn save_timed<P: AsRef<Path>>(
        &mut self,
        path: P,
        format: OutputFormat,
        quality: u8,
    ) -> Result<()> {
        match format {
            OutputFormat::Png => self.save_png_timed(path),
            OutputFormat::Jpeg => self.save_jpeg_timed(path, quality),
            OutputFormat::WebP => self.save_webp_timed(path, quality),
            OutputFormat::Tiff => self.save_tiff_timed(path),
            OutputFormat::Rgba8 => {
                let path_str = path.as_ref().display().to_string();
                let encode_start = Instant::now();

                // For RGBA8 format, save the raw RGBA bytes
                let rgba_image = self.image.to_rgba8();
                std::fs::write(&path, rgba_image.as_raw())?;

                let encode_ms = encode_start.elapsed().as_millis().try_into().map_err(|_| {
                    BgRemovalError::processing("RGBA8 encoding time too large for u64")
                })?;

                // Update the timings
                self.metadata.timings.image_encode_ms = Some(encode_ms);

                // Log encoding completion at debug level
                log::debug!("Image Encoding completed in {}ms", encode_ms);

                // Log final completion with total processing time
                let total_time_s = self.metadata.timings.total_ms as f64 / 1000.0;
                let input_path = self.input_path.as_deref().unwrap_or("input");
                log::debug!(
                    "Processed: {} -> {} in {:.2}s",
                    input_path,
                    path_str,
                    total_time_s
                );

                Ok(())
            },
        }
    }

    /// Get timing summary for display
    #[must_use]
    pub fn timing_summary(&self) -> String {
        let t = &self.metadata.timings;
        let breakdown = t.breakdown_percentages();

        let mut summary = format!("Total: {}ms", t.total_ms);

        // Add model load timing if present (only for first time)
        if t.model_load_ms > 0 {
            write!(
                summary,
                " | Model Load: {}ms ({:.1}%)",
                t.model_load_ms, breakdown.model_load_pct
            )
            .expect("Writing to String should never fail");
        }

        write!(
            summary,
            " | Decode: {}ms ({:.1}%) | Preprocess: {}ms ({:.1}%) | Inference: {}ms ({:.1}%) | Postprocess: {}ms ({:.1}%)",
            t.image_decode_ms, breakdown.decode_pct,
            t.preprocessing_ms, breakdown.preprocessing_pct,
            t.inference_ms, breakdown.inference_pct,
            t.postprocessing_ms, breakdown.postprocessing_pct
        )
        .expect("Writing to String should never fail");

        // Add encode timing if present
        if let Some(encode_ms) = t.image_encode_ms {
            write!(
                summary,
                " | Encode: {}ms ({:.1}%)",
                encode_ms, breakdown.encode_pct
            )
            .expect("Writing to String should never fail");
        }

        // Add other/overhead if significant (>1% or >5ms)
        let other_ms = t.other_overhead_ms();
        if other_ms > 5 || breakdown.other_pct > 1.0 {
            write!(
                summary,
                " | Other: {}ms ({:.1}%)",
                other_ms, breakdown.other_pct
            )
            .expect("Writing to String should never fail");
        }

        summary
    }

    /// Save the result with ICC color profile preservation when supported
    ///
    /// Attempts to preserve the original ICC color profile in the output image
    /// when supported by the target format. Falls back to standard saving
    /// if color profile embedding is not supported or disabled.
    ///
    /// # Supported Formats for ICC Profiles
    /// - **PNG**: Planned support (requires implementation)
    /// - **JPEG**: Planned support (requires implementation)  
    /// - **WebP**: Not supported in current implementation
    /// - **RGBA8**: Not applicable (raw bytes)
    ///
    /// # Arguments
    /// * `path` - Output file path
    /// * `format` - Output image format
    /// * `quality` - Quality setting for lossy formats (0-100)
    ///
    /// # Examples
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalConfig, remove_background_from_reader, ModelSpec, ModelSource, OutputFormat};
    /// use tokio::fs::File;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let model_spec = ModelSpec {
    ///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///     variant: None,
    /// };
    /// let config = RemovalConfig::builder()
    ///     .model_spec(model_spec)
    ///     .preserve_color_profiles(true)
    ///     .build()?;
    /// let file = File::open("photo.jpg").await?;
    /// let result = remove_background_from_reader(file, &config).await?;
    ///
    /// // Save with color profile preservation
    /// result.save_with_color_profile("output.png", OutputFormat::Png, 0)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Supported Formats
    /// - **PNG**: Embeds ICC profiles using iCCP chunks (✅ implemented)
    /// - **JPEG**: Embeds ICC profiles using APP2 markers (✅ implemented)
    /// - **WebP**: Not yet supported (falls back to standard save)
    /// - **RGBA8**: Not applicable (raw format, falls back to standard save)
    ///
    /// # Implementation Details
    /// - PNG embedding uses the `png` crate's built-in iCCP support
    /// - JPEG embedding uses custom APP2 marker implementation
    /// - Large profiles are automatically split across multiple JPEG segments
    /// - Profile validation and error handling ensure data integrity
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - Image encoding errors specific to the chosen format
    /// - Color profile embedding errors:
    ///   - Invalid ICC profile data
    ///   - Profile too large for format constraints
    ///   - Format-specific embedding failures
    /// - Falls back to standard save on profile embedding failures
    pub fn save_with_color_profile<P: AsRef<Path>>(
        &self,
        path: P,
        format: OutputFormat,
        quality: u8,
    ) -> Result<()> {
        use crate::color_profile::ProfileEmbedder;

        // Check if we have a color profile to embed
        if let Some(ref profile) = self.color_profile {
            log::debug!(
                "Embedding ICC color profile ({}, {} bytes) in output image",
                profile.color_space,
                profile.data_size()
            );

            // Convert OutputFormat to ImageFormat for ProfileEmbedder
            let image_format = match format {
                OutputFormat::Png => image::ImageFormat::Png,
                OutputFormat::Jpeg => image::ImageFormat::Jpeg,
                OutputFormat::WebP => image::ImageFormat::WebP,
                OutputFormat::Tiff => image::ImageFormat::Tiff,
                OutputFormat::Rgba8 => {
                    log::warn!("RGBA8 format does not support ICC profiles, saving raw data");
                    return self.save(path, format, quality);
                },
            };

            // Use ProfileEmbedder to save with ICC profile
            ProfileEmbedder::embed_in_output(&self.image, profile, path, image_format, quality)
        } else {
            // No color profile to embed, use standard saving
            log::debug!("No ICC color profile available, using standard save");
            self.save(path, format, quality)
        }
    }

    /// Save with color profile preservation and measure encoding time
    ///
    /// # Errors
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - Image encoding errors specific to the chosen format
    /// - Color profile embedding errors
    /// - Timing conversion errors if encoding time exceeds u64 range
    /// - Path conversion errors for logging purposes
    pub fn save_with_color_profile_timed<P: AsRef<Path>>(
        &mut self,
        path: P,
        format: OutputFormat,
        quality: u8,
    ) -> Result<()> {
        use crate::color_profile::ProfileEmbedder;

        let path_str = path.as_ref().display().to_string();
        let encode_start = Instant::now();

        // Check if we have a color profile to embed
        if let Some(ref profile) = self.color_profile {
            log::debug!(
                "Embedding ICC color profile ({}, {} bytes) in output image",
                profile.color_space,
                profile.data_size()
            );

            // Convert OutputFormat to ImageFormat for ProfileEmbedder
            let image_format = match format {
                OutputFormat::Png => image::ImageFormat::Png,
                OutputFormat::Jpeg => image::ImageFormat::Jpeg,
                OutputFormat::WebP => image::ImageFormat::WebP,
                OutputFormat::Tiff => image::ImageFormat::Tiff,
                OutputFormat::Rgba8 => {
                    log::warn!("RGBA8 format does not support ICC profiles, saving raw data");
                    return self.save_timed(path, format, quality);
                },
            };

            // Use ProfileEmbedder to save with ICC profile
            ProfileEmbedder::embed_in_output(&self.image, profile, &path, image_format, quality)?;
        } else {
            // No color profile to embed, use standard saving
            log::debug!("No ICC color profile available, using standard save");
            return self.save_timed(path, format, quality);
        }

        let encode_ms = encode_start.elapsed().as_millis().try_into().map_err(|_| {
            BgRemovalError::processing("Color profile encoding time too large for u64")
        })?;

        // Update the timings
        self.metadata.timings.image_encode_ms = Some(encode_ms);

        // Log encoding completion at debug level
        log::debug!("Image Encoding completed in {}ms", encode_ms);

        // Log final completion with total processing time
        let total_time_s = self.metadata.timings.total_ms as f64 / 1000.0;
        let input_path = self.input_path.as_deref().unwrap_or("input");
        log::debug!(
            "Processed: {} -> {} in {:.2}s",
            input_path,
            path_str,
            total_time_s
        );

        Ok(())
    }

    /// Get the ICC color profile if available
    ///
    /// Returns the ICC color profile that was extracted from the original input image,
    /// if color profile preservation was enabled during processing.
    ///
    /// # Returns
    /// - `Some(ColorProfile)` - ICC profile extracted from input image
    /// - `None` - No color profile available (not preserved or not present in input)
    ///
    /// # Examples
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalConfig, remove_background_from_reader, ModelSpec, ModelSource};
    /// use tokio::fs::File;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = RemovalConfig::builder()
    ///     .preserve_color_profiles(true)
    ///     .build()?;
    /// let model_spec = ModelSpec {
    ///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///     variant: None,
    /// };
    /// let file = File::open("photo.jpg").await?;
    /// let result = remove_background_from_reader(file, &config).await?;
    ///
    /// if let Some(profile) = result.get_color_profile() {
    ///     println!("Original color space: {}", profile.color_space);
    ///     println!("Profile size: {} bytes", profile.data_size());
    /// } else {
    ///     println!("No color profile available");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn get_color_profile(&self) -> Option<&ColorProfile> {
        self.color_profile.as_ref()
    }

    /// Check if the result has an ICC color profile
    ///
    /// Returns `true` if an ICC color profile was preserved from the input image.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use imgly_bgremove::{RemovalConfig, remove_background_from_reader, ModelSpec, ModelSource};
    /// use tokio::fs::File;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let model_spec = ModelSpec {
    ///     source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///     variant: None,
    /// };
    /// let config = RemovalConfig::builder()
    ///     .model_spec(model_spec)
    ///     .build()?;
    /// let file = File::open("photo.jpg").await?;
    /// let result = remove_background_from_reader(file, &config).await?;
    ///
    /// if result.has_color_profile() {
    ///     // Use color-profile-aware saving
    ///     result.save_with_color_profile("output.png", config.output_format, 0)?;
    /// } else {
    ///     // Standard saving
    ///     result.save_png("output.png")?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn has_color_profile(&self) -> bool {
        self.color_profile.is_some()
    }

    /// Encode as WebP with RGBA transparency support
    #[cfg(feature = "webp-support")]
    fn encode_webp(&self, _quality: u8) -> Vec<u8> {
        use image::ImageEncoder;

        let rgba_image = self.image.to_rgba8();
        let mut buffer = Vec::new();

        // Note: image 0.25.6 only supports lossless WebP encoding
        // Using lossless for all qualities to avoid external dependencies
        let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut buffer);
        encoder
            .write_image(
                rgba_image.as_raw(),
                rgba_image.width(),
                rgba_image.height(),
                image::ExtendedColorType::Rgba8,
            )
            .expect("WebP encoding failed");

        buffer
    }

    /// Fallback WebP encoding when webp feature is disabled (returns PNG instead)
    #[cfg(not(feature = "webp-support"))]
    fn encode_webp(&self, _quality: u8) -> Vec<u8> {
        log::warn!("WebP support disabled - falling back to PNG encoding");
        self.to_bytes(OutputFormat::Png, 100).unwrap_or_default()
    }
}

/// Binary segmentation mask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationMask {
    /// Mask data as grayscale values (0-255)
    pub data: Vec<u8>,

    /// Mask dimensions (width, height)
    pub dimensions: (u32, u32),
}

impl SegmentationMask {
    /// Create a new segmentation mask
    #[must_use]
    pub fn new(data: Vec<u8>, dimensions: (u32, u32)) -> Self {
        Self { data, dimensions }
    }

    /// Create mask from a grayscale image
    #[must_use]
    pub fn from_image(image: &ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Self {
        let (width, height) = image.dimensions();
        let data = image.as_raw().clone();

        Self::new(data, (width, height))
    }

    /// Convert mask to a grayscale image
    ///
    /// # Errors
    /// - Processing errors when mask data dimensions don't match expected size
    /// - Memory allocation errors when creating image buffer
    /// - Invalid mask data that cannot be converted to a valid image buffer
    pub fn to_image(&self) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        let (width, height) = self.dimensions;
        ImageBuffer::from_raw(width, height, self.data.clone())
            .ok_or_else(|| BgRemovalError::processing("Failed to create image from mask data"))
    }

    /// Apply the mask to an RGBA image
    ///
    /// # Errors
    /// - Processing errors when image and mask dimensions don't match
    /// - Invalid mask data that would cause out-of-bounds access
    /// - Memory access errors during pixel manipulation
    pub fn apply_to_image(&self, image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<()> {
        let (img_width, img_height) = image.dimensions();
        let (mask_width, mask_height) = self.dimensions;

        if img_width != mask_width || img_height != mask_height {
            return Err(BgRemovalError::processing(
                "Image and mask dimensions do not match",
            ));
        }

        for (i, pixel) in image.pixels_mut().enumerate() {
            if let Some(alpha) = self.data.get(i) {
                pixel[3] = *alpha; // Set alpha channel
            }
        }

        Ok(())
    }

    /// Resize the mask to new dimensions
    ///
    /// # Errors
    /// - Processing errors when converting mask to image format
    /// - Image processing errors during resize operation
    /// - Memory allocation errors for new image buffer
    /// - Invalid dimensions (zero width or height)
    pub fn resize(&self, new_width: u32, new_height: u32) -> Result<SegmentationMask> {
        let current_image = self.to_image()?;
        let resized = image::imageops::resize(
            &current_image,
            new_width,
            new_height,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(SegmentationMask::from_image(&resized))
    }

    /// Calculate comprehensive statistics about the segmentation mask
    ///
    /// Analyzes the mask to provide insights about the segmentation quality
    /// and composition. Useful for quality control and automated validation.
    ///
    /// # Returns
    /// `MaskStatistics` containing:
    /// - **Total pixels**: Complete pixel count in the mask
    /// - **Foreground pixels**: Pixels classified as subject (value > 127)
    /// - **Background pixels**: Pixels classified as background (value ≤ 127)
    /// - **Foreground ratio**: Percentage of image that is foreground (0.0-1.0)
    /// - **Background ratio**: Percentage of image that is background (0.0-1.0)
    ///
    /// # Threshold
    /// Uses 127 as the threshold (mid-point of 0-255 range):
    /// - Values 0-127: Background
    /// - Values 128-255: Foreground
    ///
    /// # Use Cases
    /// - **Quality validation**: Detect masks with too small/large foreground
    /// - **Automated filtering**: Skip images with poor segmentation
    /// - **Performance monitoring**: Track segmentation accuracy over time
    /// - **Batch processing**: Generate reports on processing results
    ///
    /// # Examples
    ///
    /// ## Basic statistics analysis
    /// ```rust,no_run
    /// use imgly_bgremove::{BackgroundRemovalProcessor, ProcessorConfigBuilder, ModelSpec, ModelSource, BackendType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = ProcessorConfigBuilder::new()
    ///     .model_spec(ModelSpec {
    ///         source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///         variant: None,
    ///     })
    ///     .backend_type(BackendType::Onnx)
    ///     .build()?;
    /// let mut processor = BackgroundRemovalProcessor::new(config)?;
    /// let result = processor.process_file("photo.jpg")?;
    /// let mask = &result.mask;
    ///
    /// let stats = mask.statistics();
    /// println!("Foreground: {:.1}% ({} pixels)",
    ///     stats.foreground_ratio * 100.0,
    ///     stats.foreground_pixels);
    /// println!("Background: {:.1}% ({} pixels)",
    ///     stats.background_ratio * 100.0,
    ///     stats.background_pixels);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Quality control workflow
    /// ```rust,no_run
    /// use imgly_bgremove::{BackgroundRemovalProcessor, ProcessorConfigBuilder, ModelSpec, ModelSource, BackendType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = ProcessorConfigBuilder::new()
    ///     .model_spec(ModelSpec {
    ///         source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///         variant: None,
    ///     })
    ///     .backend_type(BackendType::Onnx)
    ///     .build()?;
    /// let mut processor = BackgroundRemovalProcessor::new(config)?;
    /// let result = processor.process_file("portrait.jpg")?;
    /// let mask = &result.mask;
    ///
    /// let stats = mask.statistics();
    ///
    /// if stats.foreground_ratio < 0.05 {
    ///     println!("Warning: Very small subject detected");
    /// } else if stats.foreground_ratio > 0.8 {
    ///     println!("Warning: Most of image classified as foreground");
    /// } else {
    ///     println!("Good segmentation ratio: {:.1}%", stats.foreground_ratio * 100.0);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    #[must_use]
    pub fn statistics(&self) -> MaskStatistics {
        let total_pixels = self.data.len() as f32;
        let foreground_pixels = self.data.iter().filter(|&&x| x > 127).count() as f32;
        let background_pixels = total_pixels - foreground_pixels;

        MaskStatistics {
            total_pixels: total_pixels as usize,
            foreground_pixels: foreground_pixels as usize,
            background_pixels: background_pixels as usize,
            foreground_ratio: foreground_pixels / total_pixels,
            background_ratio: background_pixels / total_pixels,
        }
    }

    /// Save mask as PNG
    ///
    /// # Errors
    /// - Processing errors when converting mask to image format
    /// - File I/O errors when creating or writing output file
    /// - Permission errors or insufficient disk space
    /// - PNG encoding errors from underlying image library
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let image = self.to_image()?;
        image.save_with_format(path, image::ImageFormat::Png)?;
        Ok(())
    }
}

/// Statistics about a segmentation mask
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskStatistics {
    pub total_pixels: usize,
    pub foreground_pixels: usize,
    pub background_pixels: usize,
    pub foreground_ratio: f32,
    pub background_ratio: f32,
}

/// Detailed timing breakdown for background removal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimings {
    /// Model loading time (first call only)
    pub model_load_ms: u64,

    /// Image loading and decoding from file
    pub image_decode_ms: u64,

    /// Image preprocessing (resize, normalize, tensor conversion)
    pub preprocessing_ms: u64,

    /// ONNX Runtime inference execution
    pub inference_ms: u64,

    /// Postprocessing (mask generation, alpha application)
    pub postprocessing_ms: u64,

    /// Final image encoding (if saving to file)
    pub image_encode_ms: Option<u64>,

    /// Total end-to-end processing time
    pub total_ms: u64,
}

impl ProcessingTimings {
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_load_ms: 0,
            image_decode_ms: 0,
            preprocessing_ms: 0,
            inference_ms: 0,
            postprocessing_ms: 0,
            image_encode_ms: None,
            total_ms: 0,
        }
    }

    /// Calculate efficiency metrics
    #[must_use]
    pub fn inference_ratio(&self) -> f64 {
        if self.total_ms == 0 {
            0.0
        } else {
            let inference_f64 = self.inference_ms as f64;
            let total_f64 = self.total_ms as f64;
            inference_f64 / total_f64
        }
    }

    /// Get breakdown percentages
    #[must_use]
    pub fn breakdown_percentages(&self) -> TimingBreakdown {
        if self.total_ms == 0 {
            return TimingBreakdown::default();
        }

        let total = self.total_ms as f64;
        let measured_time = self.model_load_ms
            + self.image_decode_ms
            + self.preprocessing_ms
            + self.inference_ms
            + self.postprocessing_ms
            + self.image_encode_ms.unwrap_or(0);

        let other_ms = self.total_ms.saturating_sub(measured_time);

        let model_load_f64 = self.model_load_ms as f64;
        let decode_f64 = self.image_decode_ms as f64;
        let preprocessing_f64 = self.preprocessing_ms as f64;
        let inference_f64 = self.inference_ms as f64;
        let postprocessing_f64 = self.postprocessing_ms as f64;
        let encode_f64 = self.image_encode_ms.unwrap_or(0) as f64;
        let other_f64 = other_ms as f64;

        TimingBreakdown {
            model_load_pct: (model_load_f64 / total) * 100.0,
            decode_pct: (decode_f64 / total) * 100.0,
            preprocessing_pct: (preprocessing_f64 / total) * 100.0,
            inference_pct: (inference_f64 / total) * 100.0,
            postprocessing_pct: (postprocessing_f64 / total) * 100.0,
            encode_pct: (encode_f64 / total) * 100.0,
            other_pct: (other_f64 / total) * 100.0,
        }
    }

    /// Get the "other" overhead time (unaccounted time)
    #[must_use]
    pub fn other_overhead_ms(&self) -> u64 {
        let measured_time = self.model_load_ms
            + self.image_decode_ms
            + self.preprocessing_ms
            + self.inference_ms
            + self.postprocessing_ms
            + self.image_encode_ms.unwrap_or(0);

        self.total_ms.saturating_sub(measured_time)
    }
}

impl Default for ProcessingTimings {
    fn default() -> Self {
        Self::new()
    }
}

/// Percentage breakdown of timing phases
#[derive(Debug, Clone)]
pub struct TimingBreakdown {
    pub model_load_pct: f64,
    pub decode_pct: f64,
    pub preprocessing_pct: f64,
    pub inference_pct: f64,
    pub postprocessing_pct: f64,
    pub encode_pct: f64,
    pub other_pct: f64,
}

impl Default for TimingBreakdown {
    fn default() -> Self {
        Self {
            model_load_pct: 0.0,
            decode_pct: 0.0,
            preprocessing_pct: 0.0,
            inference_pct: 0.0,
            postprocessing_pct: 0.0,
            encode_pct: 0.0,
            other_pct: 0.0,
        }
    }
}

/// Metadata about the processing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Detailed timing breakdown
    pub timings: ProcessingTimings,

    /// Model used for inference
    pub model_name: String,

    /// Model precision used
    pub model_precision: String,

    /// Input image format
    pub input_format: String,

    /// Output image format
    pub output_format: String,

    /// Memory usage peak (bytes)
    pub peak_memory_bytes: u64,

    /// ICC color profile from input image  
    pub color_profile: Option<ColorProfile>,

    // Legacy timing fields for backward compatibility
    /// Time taken for inference (milliseconds) - DEPRECATED: use `timings.inference_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_time_ms: Option<u64>,

    /// Time taken for preprocessing (milliseconds) - DEPRECATED: use `timings.preprocessing_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocessing_time_ms: Option<u64>,

    /// Time taken for postprocessing (milliseconds) - DEPRECATED: use `timings.postprocessing_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocessing_time_ms: Option<u64>,

    /// Total processing time (milliseconds) - DEPRECATED: use `timings.total_ms`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<u64>,
}

impl ProcessingMetadata {
    /// Create new processing metadata
    #[must_use]
    pub fn new(model_name: String) -> Self {
        Self {
            timings: ProcessingTimings::new(),
            model_name,
            model_precision: "fp16".to_string(),
            input_format: "unknown".to_string(),
            output_format: "png".to_string(),
            peak_memory_bytes: 0,
            color_profile: None,
            // Legacy fields set to None by default
            inference_time_ms: None,
            preprocessing_time_ms: None,
            postprocessing_time_ms: None,
            total_time_ms: None,
        }
    }

    /// Set timing information (new detailed version)
    pub fn set_detailed_timings(&mut self, timings: ProcessingTimings) {
        // Also set legacy fields for backward compatibility
        self.inference_time_ms = Some(timings.inference_ms);
        self.preprocessing_time_ms = Some(timings.preprocessing_ms);
        self.postprocessing_time_ms = Some(timings.postprocessing_ms);
        self.total_time_ms = Some(timings.total_ms);

        // Update detailed timings (move after using fields)
        self.timings = timings;
    }
}

/// Video processing result containing processed video data and metadata
#[cfg(feature = "video-support")]
#[derive(Debug)]
pub struct VideoRemovalResult {
    /// Video data as bytes
    pub video_data: Vec<u8>,
    /// Original video metadata
    pub original_metadata: VideoMetadata,
    /// Frame processing statistics
    pub frame_stats: FrameProcessingStats,
    /// Processing metadata for the video operation
    pub processing_metadata: ProcessingMetadata,
    /// Color profile from the original video (if available)
    pub color_profile: Option<ColorProfile>,
}

#[cfg(feature = "video-support")]
impl VideoRemovalResult {
    /// Create a new video removal result
    pub fn new(
        video_data: Vec<u8>,
        original_metadata: VideoMetadata,
        frame_stats: FrameProcessingStats,
        processing_metadata: ProcessingMetadata,
        color_profile: Option<ColorProfile>,
    ) -> Self {
        Self {
            video_data,
            original_metadata,
            frame_stats,
            processing_metadata,
            color_profile,
        }
    }

    /// Save the processed video to a file
    ///
    /// # Arguments
    /// * `path` - Output file path
    ///
    /// # Returns
    /// * `Ok(())` - Successfully saved video
    /// * `Err(BgRemovalError)` - Failed to save video
    ///
    /// # Examples
    /// ```rust,no_run
    /// # #[cfg(feature = "video-support")]
    /// # {
    /// use imgly_bgremove::VideoRemovalResult;
    ///
    /// # async fn example(result: VideoRemovalResult) -> anyhow::Result<()> {
    /// result.save("output.mp4")?;
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        std::fs::write(path.as_ref(), &self.video_data)
            .map_err(|e| BgRemovalError::file_io_error("write video file", path.as_ref(), &e))?;
        Ok(())
    }

    /// Save the processed video to a file with specific codec settings
    ///
    /// # Arguments
    /// * `path` - Output file path
    /// * `codec` - Video codec to use for encoding
    /// * `quality` - Encoding quality (codec-specific)
    ///
    /// # Returns
    /// * `Ok(())` - Successfully saved video
    /// * `Err(BgRemovalError)` - Failed to save video
    pub fn save_with_codec<P: AsRef<Path>>(
        &self,
        path: P,
        _codec: crate::backends::video::VideoCodec,
        _quality: u8,
    ) -> Result<()> {
        // For now, just save the existing data
        // In a full implementation, this would re-encode with the specified codec
        self.save(path)
    }

    /// Write video data to an async writer
    ///
    /// # Arguments
    /// * `writer` - Async writer destination
    ///
    /// # Returns
    /// Number of bytes written
    pub async fn write_to<W>(&self, mut writer: W) -> Result<u64>
    where
        W: tokio::io::AsyncWrite + Unpin,
    {
        use tokio::io::AsyncWriteExt;

        writer.write_all(&self.video_data).await.map_err(|e| {
            BgRemovalError::processing(format!("Failed to write video data: {}", e))
        })?;

        writer.flush().await.map_err(|e| {
            BgRemovalError::processing(format!("Failed to flush video data: {}", e))
        })?;

        Ok(self.video_data.len() as u64)
    }

    /// Get video data as bytes
    pub fn to_bytes(&self) -> &[u8] {
        &self.video_data
    }

    /// Get video data size in bytes
    pub fn size(&self) -> usize {
        self.video_data.len()
    }

    /// Get video duration in seconds
    pub fn duration(&self) -> f64 {
        self.original_metadata.duration
    }

    /// Get video dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.original_metadata.width, self.original_metadata.height)
    }

    /// Get video frame rate
    pub fn fps(&self) -> f64 {
        self.original_metadata.fps
    }

    /// Get processing success rate
    pub fn success_rate(&self) -> f64 {
        self.frame_stats.success_rate()
    }

    /// Get average processing time per frame
    pub fn average_frame_time(&self) -> std::time::Duration {
        self.frame_stats.average_frame_time
    }

    /// Get total frames processed
    pub fn frames_processed(&self) -> u64 {
        self.frame_stats.frames_processed
    }

    /// Generate a summary report of the video processing
    pub fn summary(&self) -> String {
        format!(
            "Video Processing Summary:\n\
             - Duration: {:.2}s\n\
             - Dimensions: {}x{}\n\
             - Frame Rate: {:.2} fps\n\
             - Frames Processed: {}\n\
             - Success Rate: {:.1}%\n\
             - Average Frame Time: {:.2}ms\n\
             - Total Processing Time: {:.2}s\n\
             - Output Size: {:.2} MB",
            self.duration(),
            self.original_metadata.width,
            self.original_metadata.height,
            self.fps(),
            self.frames_processed(),
            self.success_rate(),
            self.average_frame_time().as_secs_f64() * 1000.0,
            self.frame_stats.total_processing_time.as_secs_f64(),
            self.size() as f64 / (1024.0 * 1024.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmentation_mask_creation() {
        let data = vec![255, 128, 0, 255];
        let mask = SegmentationMask::new(data, (2, 2));

        assert_eq!(mask.dimensions, (2, 2));
        assert_eq!(mask.data.len(), 4);
    }

    #[test]
    fn test_mask_statistics() {
        let data = vec![255, 255, 0, 0]; // 2 foreground, 2 background
        let mask = SegmentationMask::new(data, (2, 2));

        let stats = mask.statistics();
        assert_eq!(stats.total_pixels, 4);
        assert_eq!(stats.foreground_pixels, 2);
        assert_eq!(stats.background_pixels, 2);
        assert!((stats.foreground_ratio - 0.5).abs() < f32::EPSILON);
        assert!((stats.background_ratio - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_processing_metadata() {
        let mut metadata = ProcessingMetadata::new("isnet".to_string());

        // Use new detailed timing method instead of deprecated set_timings
        let timings = ProcessingTimings {
            model_load_ms: 0,
            image_decode_ms: 0,
            preprocessing_ms: 50,
            inference_ms: 100,
            postprocessing_ms: 25,
            image_encode_ms: None,
            total_ms: 175,
        };
        metadata.set_detailed_timings(timings);

        assert_eq!(metadata.inference_time_ms, Some(100));
        assert_eq!(metadata.preprocessing_time_ms, Some(50));
        assert_eq!(metadata.postprocessing_time_ms, Some(25));
        assert_eq!(metadata.total_time_ms, Some(175));
    }
}
