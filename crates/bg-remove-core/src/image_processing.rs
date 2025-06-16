//! Image processing pipeline for background removal

use crate::{
    backends::{MockBackend, OnnxBackend},
    color_profile::ProfileExtractor,
    config::{BackgroundColor, OutputFormat, RemovalConfig},
    error::Result,
    inference::InferenceBackend,
    models::ModelManager,
    types::{ColorProfile, ProcessingMetadata, RemovalResult, SegmentationMask},
};
use image::{DynamicImage, GenericImageView, ImageBuffer, RgbaImage};
use log;
use ndarray::Array4;
use std::path::Path;
use std::time::Instant;

/// Processing options for fine-tuning behavior
#[derive(Debug, Clone, Default)]
pub struct ProcessingOptions {
    /// Padding color for aspect ratio preservation (RGB)
    pub padding_color: [u8; 3],
}

/// Main image processor for background removal operations
pub struct ImageProcessor {
    backend: Box<dyn InferenceBackend>,
    config: RemovalConfig,
    options: ProcessingOptions,
}

impl ImageProcessor {
    /// Create a new image processor with the given configuration and model manager
    ///
    /// # Arguments
    /// * `config` - Processing configuration including execution provider and quality settings
    /// * `model_manager` - Pre-configured model manager with loaded model
    ///
    /// # Errors
    /// - Backend initialization failures
    /// - Model loading or validation errors
    /// - Invalid configuration parameters
    pub fn with_model_manager(config: &RemovalConfig, model_manager: ModelManager) -> Result<Self> {
        let mut backend: Box<dyn InferenceBackend> = if config.debug {
            // Use mock backend for debugging - it doesn't need the model manager
            Box::new(MockBackend::new())
        } else {
            // Use ONNX backend with the specific model manager
            Box::new(OnnxBackend::with_model_manager(model_manager))
        };

        // Initialize the backend and capture model load timing
        let _model_load_time = backend.initialize(config)?;

        Ok(Self {
            backend,
            config: config.clone(),
            options: ProcessingOptions::default(),
        })
    }

    /// Create a new image processor with the given configuration (legacy - uses first available embedded model)
    ///
    /// # Errors
    /// - No embedded models available in the build
    /// - Backend initialization failures
    /// - Model loading or validation errors
    pub fn new(config: &RemovalConfig) -> Result<Self> {
        let model_manager = ModelManager::with_embedded()?;
        Self::with_model_manager(config, model_manager)
    }

    /// Create processor with custom processing options and model manager
    ///
    /// # Errors
    /// - Backend initialization failures
    /// - Model loading or validation errors
    /// - Invalid configuration parameters
    pub fn with_options_and_model(
        config: &RemovalConfig,
        options: ProcessingOptions,
        model_manager: ModelManager,
    ) -> Result<Self> {
        let mut processor = Self::with_model_manager(config, model_manager)?;
        processor.options = options;
        Ok(processor)
    }

    /// Create processor with custom processing options (legacy - uses first available embedded model)
    ///
    /// # Errors
    /// - No embedded models available in the build
    /// - Backend initialization failures
    /// - Model loading or validation errors
    pub fn with_options(config: &RemovalConfig, options: ProcessingOptions) -> Result<Self> {
        let mut processor = Self::new(config)?;
        processor.options = options;
        Ok(processor)
    }

    /// Remove background from an image file
    ///
    /// # Errors
    /// - Image loading failures (file not found, invalid format, etc.)
    /// - Model inference errors
    /// - ICC color profile extraction failures (if enabled)
    /// - Image processing and mask application errors
    #[allow(clippy::unused_async)] // Kept async for API consistency
    #[allow(clippy::too_many_lines)] // Complex pipeline with detailed logging and timing
    pub async fn remove_background<P: AsRef<Path>>(
        &mut self,
        input_path: P,
    ) -> Result<RemovalResult> {
        let input_path_str = input_path.as_ref().display().to_string();
        let total_start = Instant::now();
        let mut metadata = ProcessingMetadata::new("ISNet".to_string());
        let mut timings = crate::types::ProcessingTimings::new();

        // Log start of processing with actual model information from backend
        let model_info = match self.backend.get_model_info() {
            Ok(info) => format!("{} ({})", info.name, info.precision),
            Err(_) => "Unknown Model".to_string(),
        };

        log::info!(
            "Starting processing: {} - Model: {}",
            input_path_str,
            model_info
        );

        // 1. Image decode timing with ICC profile extraction
        let decode_start = Instant::now();
        let (image, color_profile) = self.load_image_with_profile(input_path)?;
        let original_dimensions = image.dimensions();
        timings.image_decode_ms = decode_start.elapsed().as_millis().try_into().map_err(|_| {
            crate::error::BgRemovalError::processing("Image decode time too large for u64")
        })?;

        // Log color profile information if found
        if let Some(ref profile) = color_profile {
            log::debug!(
                "Image decoded: {}x{} in {}ms - Color Profile: {} ({} bytes)",
                original_dimensions.0,
                original_dimensions.1,
                timings.image_decode_ms,
                profile.color_space,
                profile.data_size()
            );
        } else {
            log::debug!(
                "Image decoded: {}x{} in {}ms - No color profile",
                original_dimensions.0,
                original_dimensions.1,
                timings.image_decode_ms
            );
        }

        // 2. Preprocessing timing
        let preprocess_start = Instant::now();
        let (_processed_image, input_tensor) = self.preprocess_image(&image)?;
        timings.preprocessing_ms =
            preprocess_start
                .elapsed()
                .as_millis()
                .try_into()
                .map_err(|_| {
                    crate::error::BgRemovalError::processing("Preprocessing time too large for u64")
                })?;

        log::debug!(
            "Preprocessing completed in {}ms",
            timings.preprocessing_ms
        );

        // 3. Model loading and inference timing
        let was_initialized = self.backend.is_initialized();
        let inference_start = Instant::now();
        
        // Initialize backend if needed and capture model load time
        if !was_initialized {
            let model_load_time = self.backend.initialize(&self.config)?;
            if let Some(load_duration) = model_load_time {
                timings.model_load_ms = load_duration.as_millis().try_into().map_err(|_| {
                    crate::error::BgRemovalError::processing("Model load time too large for u64")
                })?;
                log::debug!(
                    "Model loaded in {}ms",
                    timings.model_load_ms
                );
            }
        }
        
        let output_tensor = self.backend.infer(&input_tensor)?;
        timings.inference_ms = inference_start
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| {
                crate::error::BgRemovalError::processing("Inference time too large for u64")
            })?;

        log::debug!(
            "Inference completed in {}ms",
            timings.inference_ms
        );

        // 4. Postprocessing timing
        let postprocess_start = Instant::now();
        let mask = self.tensor_to_mask(&output_tensor, original_dimensions)?;
        let result_image = self.apply_background_removal(&image, &mask)?;
        timings.postprocessing_ms =
            postprocess_start
                .elapsed()
                .as_millis()
                .try_into()
                .map_err(|_| {
                    crate::error::BgRemovalError::processing(
                        "Postprocessing time too large for u64",
                    )
                })?;

        log::debug!(
            "Postprocessing completed in {}ms",
            timings.postprocessing_ms
        );

        // Calculate total time
        timings.total_ms = total_start.elapsed().as_millis().try_into().map_err(|_| {
            crate::error::BgRemovalError::processing("Total processing time too large for u64")
        })?;

        // Set metadata with detailed timings and color profile
        metadata.set_detailed_timings(timings);
        metadata.input_format = Self::detect_image_format(&image);
        metadata.output_format = format!("{:?}", self.config.output_format).to_lowercase();
        metadata.color_profile.clone_from(&color_profile);

        Ok(RemovalResult::with_input_path_and_profile(
            result_image,
            mask,
            original_dimensions,
            metadata,
            input_path_str,
            color_profile,
        ))
    }

    /// Extract foreground segmentation mask only
    ///
    /// # Errors
    /// - Image loading failures (file not found, invalid format, etc.)
    /// - Model inference errors
    /// - Image preprocessing errors
    /// - Tensor conversion errors
    #[allow(clippy::unused_async)] // Kept async for API consistency
    pub async fn segment_foreground<P: AsRef<Path>>(
        &mut self,
        input_path: P,
    ) -> Result<SegmentationMask> {
        let image = Self::load_image(input_path)?;
        let (processed_image, input_tensor) = self.preprocess_image(&image)?;

        let output_tensor = self.backend.infer(&input_tensor)?;
        self.tensor_to_mask(&output_tensor, processed_image.dimensions())
    }

    /// Apply an existing mask to an image
    ///
    /// # Errors
    /// - Image loading failures (file not found, invalid format, etc.)
    /// - Mask resizing errors if dimensions don't match
    /// - Image processing and mask application errors
    /// - ICC color profile extraction failures (if enabled)
    #[allow(clippy::unused_async)] // Kept async for API consistency
    pub async fn apply_mask<P: AsRef<Path>>(
        &self,
        input_path: P,
        mask: &SegmentationMask,
    ) -> Result<RemovalResult> {
        let total_start = Instant::now();
        let mut metadata = ProcessingMetadata::new("MaskApplication".to_string());
        let mut timings = crate::types::ProcessingTimings::new();

        // Image decode timing with ICC profile extraction
        let decode_start = Instant::now();
        let (image, color_profile) = self.load_image_with_profile(input_path)?;
        let original_dimensions = image.dimensions();
        timings.image_decode_ms = decode_start.elapsed().as_millis().try_into().map_err(|_| {
            crate::error::BgRemovalError::processing("Image decode time too large for u64")
        })?;

        // Mask preprocessing timing (resize if needed)
        let preprocess_start = Instant::now();
        let resized_mask = if mask.dimensions == image.dimensions() {
            mask.clone()
        } else {
            mask.resize(image.dimensions().0, image.dimensions().1)?
        };
        timings.preprocessing_ms =
            preprocess_start
                .elapsed()
                .as_millis()
                .try_into()
                .map_err(|_| {
                    crate::error::BgRemovalError::processing(
                        "Mask preprocessing time too large for u64",
                    )
                })?;

        // No inference for mask application
        timings.inference_ms = 0;

        // Apply mask timing
        let postprocess_start = Instant::now();
        let result_image = self.apply_background_removal(&image, &resized_mask)?;
        timings.postprocessing_ms =
            postprocess_start
                .elapsed()
                .as_millis()
                .try_into()
                .map_err(|_| {
                    crate::error::BgRemovalError::processing(
                        "Mask application time too large for u64",
                    )
                })?;

        // Calculate total time
        timings.total_ms = total_start.elapsed().as_millis().try_into().map_err(|_| {
            crate::error::BgRemovalError::processing(
                "Total mask application time too large for u64",
            )
        })?;

        metadata.set_detailed_timings(timings);
        metadata.input_format = Self::detect_image_format(&image);
        metadata.output_format = format!("{:?}", self.config.output_format).to_lowercase();
        metadata.color_profile.clone_from(&color_profile);

        Ok(RemovalResult::with_color_profile(
            result_image,
            resized_mask,
            original_dimensions,
            metadata,
            color_profile,
        ))
    }

    /// Process a `DynamicImage` directly for background removal
    ///
    /// # Errors
    /// - Model inference errors
    /// - Image preprocessing errors
    /// - Tensor conversion errors
    /// - Mask processing and application errors
    #[allow(clippy::needless_pass_by_value)] // DynamicImage is consumed by preprocessing
    pub fn process_image(&mut self, image: DynamicImage) -> Result<RemovalResult> {
        let total_start = Instant::now();
        let mut metadata = ProcessingMetadata::new("ISNet".to_string());
        let mut timings = crate::types::ProcessingTimings::new();

        let original_dimensions = image.dimensions();

        // Note: No image decode timing since image is already loaded
        timings.image_decode_ms = 0;

        // Preprocess image
        let preprocess_start = Instant::now();
        let (_preprocessed_image, input_tensor) = self.preprocess_image(&image)?;
        timings.preprocessing_ms =
            preprocess_start
                .elapsed()
                .as_millis()
                .try_into()
                .map_err(|_| {
                    crate::error::BgRemovalError::processing(
                        "Image preprocessing time too large for u64",
                    )
                })?;

        // Run inference
        let inference_start = Instant::now();
        let output_tensor = self.backend.infer(&input_tensor)?;
        timings.inference_ms = inference_start
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| {
                crate::error::BgRemovalError::processing(
                    "Direct image inference time too large for u64",
                )
            })?;

        // Postprocess results
        let postprocess_start = Instant::now();
        let mask = self.tensor_to_mask(&output_tensor, original_dimensions)?;
        let result_image = self.apply_background_removal(&image, &mask)?;
        timings.postprocessing_ms =
            postprocess_start
                .elapsed()
                .as_millis()
                .try_into()
                .map_err(|_| {
                    crate::error::BgRemovalError::processing(
                        "Direct image postprocessing time too large for u64",
                    )
                })?;

        // Calculate total time
        timings.total_ms = total_start.elapsed().as_millis().try_into().map_err(|_| {
            crate::error::BgRemovalError::processing(
                "Total direct image processing time too large for u64",
            )
        })?;

        // Set metadata with detailed timings
        metadata.set_detailed_timings(timings);
        metadata.input_format = Self::detect_image_format(&image);
        metadata.output_format = format!("{:?}", self.config.output_format).to_lowercase();
        metadata.color_profile = None; // No color profile available for in-memory images

        Ok(RemovalResult::new(
            result_image,
            mask,
            original_dimensions,
            metadata,
        ))
    }

    /// Load image from file path
    fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        let image = image::open(path)?;
        Ok(image)
    }

    /// Load image with ICC color profile extraction
    fn load_image_with_profile<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(DynamicImage, Option<ColorProfile>)> {
        // Extract ICC profile first if color management is enabled
        let profile = if self.config.color_management.preserve_color_profile {
            ProfileExtractor::extract_from_image(&path)?
        } else {
            None
        };

        // Load image using high-level function
        let image = image::open(&path)?;

        Ok((image, profile))
    }

    /// Preprocess image for model inference with aspect ratio preservation
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    // Allow casting for image processing math - precision loss is acceptable for resizing
    fn preprocess_image(&self, image: &DynamicImage) -> Result<(DynamicImage, Array4<f32>)> {
        let preprocessing_config = self.backend.get_preprocessing_config()?;
        let target_size = preprocessing_config.target_size[0];

        // Convert to RGB
        let rgb_image = image.to_rgb8();
        let (orig_width, orig_height) = rgb_image.dimensions();

        // Calculate aspect ratio preserving dimensions
        // Note: u32 to f32 conversion is safe for reasonable image dimensions
        let target_size_f32 = target_size as f32;
        let orig_width_f32 = orig_width as f32;
        let orig_height_f32 = orig_height as f32;

        let scale = target_size_f32
            .min((target_size_f32 / orig_width_f32).min(target_size_f32 / orig_height_f32));

        let new_width_f32 = (orig_width_f32 * scale).round();
        let new_height_f32 = (orig_height_f32 * scale).round();

        if new_width_f32 < 0.0 || new_width_f32 > u32::MAX as f32 {
            return Err(crate::error::BgRemovalError::processing(
                "Calculated new width out of valid range",
            ));
        }
        if new_height_f32 < 0.0 || new_height_f32 > u32::MAX as f32 {
            return Err(crate::error::BgRemovalError::processing(
                "Calculated new height out of valid range",
            ));
        }

        let new_width = new_width_f32 as u32;
        let new_height = new_height_f32 as u32;

        // Resize image maintaining aspect ratio
        let resized = image::imageops::resize(
            &rgb_image,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle,
        );

        // Create a target_size x target_size canvas with configurable padding color
        let padding = self.options.padding_color;
        let mut canvas = ImageBuffer::from_pixel(
            target_size,
            target_size,
            image::Rgb([padding[0], padding[1], padding[2]]),
        );

        // Calculate centering offset
        let offset_x = (target_size - new_width) / 2;
        let offset_y = (target_size - new_height) / 2;

        // Copy resized image to center of canvas
        for (x, y, pixel) in resized.enumerate_pixels() {
            let canvas_x = x + offset_x;
            let canvas_y = y + offset_y;
            if canvas_x < target_size && canvas_y < target_size {
                canvas.put_pixel(canvas_x, canvas_y, *pixel);
            }
        }

        // Convert to tensor format (NCHW) with normalization
        let target_size_usize = target_size.try_into().map_err(|_| {
            crate::error::BgRemovalError::processing(
                "Target size too large for usize conversion in tensor allocation",
            )
        })?;
        let mut tensor = Array4::<f32>::zeros((1, 3, target_size_usize, target_size_usize));

        #[allow(clippy::indexing_slicing)]
        // Safe: tensor dimensions pre-allocated to match canvas size
        for (y, row) in canvas.rows().enumerate() {
            for (x, pixel) in row.enumerate() {
                // Convert to 0-1 range and apply normalization using generated constants from model.json
                let normalized_r = (f32::from(pixel[0]) / 255.0
                    - preprocessing_config.normalization_mean[0])
                    / preprocessing_config.normalization_std[0];
                let normalized_g = (f32::from(pixel[1]) / 255.0
                    - preprocessing_config.normalization_mean[1])
                    / preprocessing_config.normalization_std[1];
                let normalized_b = (f32::from(pixel[2]) / 255.0
                    - preprocessing_config.normalization_mean[2])
                    / preprocessing_config.normalization_std[2];

                tensor[[0, 0, y, x]] = normalized_r; // R
                tensor[[0, 1, y, x]] = normalized_g; // G
                tensor[[0, 2, y, x]] = normalized_b; // B
            }
        }

        // Return the preprocessed image for debugging/metadata
        let preprocessed_image = DynamicImage::ImageRgb8(canvas);
        Ok((preprocessed_image, tensor))
    }

    /// Convert model output tensor to segmentation mask with aspect ratio handling
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    // Allow casting for tensor processing math - precision loss is acceptable
    fn tensor_to_mask(
        &self,
        tensor: &Array4<f32>,
        original_size: (u32, u32),
    ) -> Result<SegmentationMask> {
        let (batch, channels, height, width) = tensor.dim();

        if batch != 1 || channels != 1 {
            return Err(crate::error::BgRemovalError::processing(
                "Expected single-channel output tensor",
            ));
        }

        let preprocessing_config = self.backend.get_preprocessing_config()?;
        let model_size = preprocessing_config.target_size[0];
        let (orig_width, orig_height) = original_size;

        // Calculate the scale and padding used during preprocessing
        // Note: u32 to f32 conversion is safe for reasonable image dimensions
        let model_size_f32 = model_size as f32;
        let orig_width_f32 = orig_width as f32;
        let orig_height_f32 = orig_height as f32;

        let scale = model_size_f32
            .min((model_size_f32 / orig_width_f32).min(model_size_f32 / orig_height_f32));

        let scaled_width_f32 = (orig_width_f32 * scale).round();
        let scaled_height_f32 = (orig_height_f32 * scale).round();

        if scaled_width_f32 < 0.0 || scaled_width_f32 > u32::MAX as f32 {
            return Err(crate::error::BgRemovalError::processing(
                "Calculated scaled width out of valid range",
            ));
        }
        if scaled_height_f32 < 0.0 || scaled_height_f32 > u32::MAX as f32 {
            return Err(crate::error::BgRemovalError::processing(
                "Calculated scaled height out of valid range",
            ));
        }

        let scaled_width = scaled_width_f32 as u32;
        let scaled_height = scaled_height_f32 as u32;
        let offset_x = (model_size - scaled_width) / 2;
        let offset_y = (model_size - scaled_height) / 2;

        // Extract the relevant portion of the mask (crop out black padding)
        let mask_capacity = scaled_width
            .checked_mul(scaled_height)
            .and_then(|size| size.try_into().ok())
            .ok_or_else(|| {
                crate::error::BgRemovalError::processing(
                    "Mask data size too large for Vec allocation",
                )
            })?;
        let mut cropped_mask_data = Vec::with_capacity(mask_capacity);

        for y in offset_y..(offset_y + scaled_height) {
            for x in offset_x..(offset_x + scaled_width) {
                let height_u32 = height.try_into().map_err(|_| {
                    crate::error::BgRemovalError::processing(
                        "Tensor height too large for u32 comparison",
                    )
                })?;
                let width_u32 = width.try_into().map_err(|_| {
                    crate::error::BgRemovalError::processing(
                        "Tensor width too large for u32 comparison",
                    )
                })?;

                if y < height_u32 && x < width_u32 {
                    let y_usize = y.try_into().map_err(|_| {
                        crate::error::BgRemovalError::processing(
                            "Y coordinate too large for tensor indexing",
                        )
                    })?;
                    let x_usize = x.try_into().map_err(|_| {
                        crate::error::BgRemovalError::processing(
                            "X coordinate too large for tensor indexing",
                        )
                    })?;

                    let raw_value = tensor[[0, 0, y_usize, x_usize]];
                    let clamped_value = (raw_value * 255.0).clamp(0.0, 255.0).round();
                    let pixel_value =
                        if clamped_value.is_finite() && (0.0..=255.0).contains(&clamped_value) {
                            clamped_value as u8
                        } else {
                            return Err(crate::error::BgRemovalError::processing(
                                "Mask pixel value out of u8 range",
                            ));
                        };
                    cropped_mask_data.push(pixel_value);
                } else {
                    cropped_mask_data.push(0u8); // Black padding areas
                }
            }
        }

        // Create mask from cropped data
        let mut mask = SegmentationMask::new(cropped_mask_data, (scaled_width, scaled_height));

        // Resize mask back to original dimensions
        if (scaled_width, scaled_height) != original_size {
            mask = mask.resize(orig_width, orig_height)?;
        }

        Ok(mask)
    }

    /// Apply background removal using the mask
    fn apply_background_removal(
        &self,
        image: &DynamicImage,
        mask: &SegmentationMask,
    ) -> Result<DynamicImage> {
        let mut rgba_image = image.to_rgba8();

        // Apply mask to alpha channel
        mask.apply_to_image(&mut rgba_image)?;

        // Handle different output formats
        match self.config.output_format {
            OutputFormat::Png | OutputFormat::Rgba8 => {
                // Keep alpha channel
                Ok(DynamicImage::ImageRgba8(rgba_image))
            },
            OutputFormat::Jpeg => {
                // Apply background color
                let bg_color = self.config.background_color;
                let rgb_image = Self::apply_background_color(&rgba_image, bg_color)?;
                Ok(DynamicImage::ImageRgb8(rgb_image))
            },
            OutputFormat::WebP => {
                // WebP supports alpha, so keep it
                Ok(DynamicImage::ImageRgba8(rgba_image))
            },
        }
    }

    /// Apply background color for formats without alpha support
    fn apply_background_color(
        rgba_image: &RgbaImage,
        bg_color: BackgroundColor,
    ) -> Result<ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
        let (width, height) = rgba_image.dimensions();
        let mut rgb_image = ImageBuffer::new(width, height);

        for (x, y, pixel) in rgba_image.enumerate_pixels() {
            let alpha = f32::from(pixel[3]) / 255.0;
            let inv_alpha = 1.0 - alpha;

            let r_blend = (f32::from(pixel[0]) * alpha + f32::from(bg_color.r) * inv_alpha)
                .round()
                .clamp(0.0, 255.0);
            let g_blend = (f32::from(pixel[1]) * alpha + f32::from(bg_color.g) * inv_alpha)
                .round()
                .clamp(0.0, 255.0);
            let b_blend = (f32::from(pixel[2]) * alpha + f32::from(bg_color.b) * inv_alpha)
                .round()
                .clamp(0.0, 255.0);

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            // Validated via clamp and finite check
            let r = if r_blend.is_finite() {
                r_blend as u8
            } else {
                return Err(crate::error::BgRemovalError::processing(
                    "Red color blend result out of u8 range",
                ));
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            // Validated via clamp and finite check
            let g = if g_blend.is_finite() {
                g_blend as u8
            } else {
                return Err(crate::error::BgRemovalError::processing(
                    "Green color blend result out of u8 range",
                ));
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            // Validated via clamp and finite check
            let b = if b_blend.is_finite() {
                b_blend as u8
            } else {
                return Err(crate::error::BgRemovalError::processing(
                    "Blue color blend result out of u8 range",
                ));
            };

            rgb_image.put_pixel(x, y, image::Rgb([r, g, b]));
        }

        Ok(rgb_image)
    }

    /// Detect image format from dynamic image
    fn detect_image_format(image: &DynamicImage) -> String {
        match image {
            DynamicImage::ImageRgb8(_) => "rgb8".to_string(),
            DynamicImage::ImageRgba8(_) => "rgba8".to_string(),
            DynamicImage::ImageLuma8(_) => "luma8".to_string(),
            DynamicImage::ImageLumaA8(_) => "luma_a8".to_string(),
            _ => "unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RemovalConfig;

    #[test]
    fn test_processing_options() {
        let options = ProcessingOptions::default();
        assert_eq!(options.padding_color, [0, 0, 0]);
    }

    #[tokio::test]
    async fn test_image_processor_creation() {
        let config = RemovalConfig {
            debug: true, // Use mock backend
            ..Default::default()
        };

        let processor = ImageProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_preprocess_image() {
        let config = RemovalConfig {
            debug: true,
            ..Default::default()
        };

        let processor = ImageProcessor::new(&config).unwrap();

        // Create a test image
        let test_image = DynamicImage::new_rgb8(100, 100);
        let (_processed, tensor) = processor.preprocess_image(&test_image).unwrap();

        // Check tensor shape matches backend requirements
        let expected_shape = processor.backend.input_shape();
        assert_eq!(tensor.dim(), expected_shape);
    }
}
