//! Unified background removal processor
//!
//! This module provides the main `BackgroundRemovalProcessor` that consolidates
//! all business logic for background removal operations. This processor is used
//! by both CLI and Web frontends to ensure consistent behavior.

use crate::{
    config::{ExecutionProvider, OutputFormat, RemovalConfig},
    error::{BgRemovalError, Result},
    inference::InferenceBackend,
    models::{ModelManager, ModelSource, ModelSpec},
    services::{OutputFormatHandler, ProcessingStage, ProgressTracker},
    types::{ProcessingMetadata, ProcessingTimings, RemovalResult, SegmentationMask},
    utils::ImagePreprocessor,
};
use image::{DynamicImage, GenericImageView, ImageBuffer, RgbaImage};
use instant::Instant;
use log::{debug, info};
use ndarray::Array4;
use std::path::Path;
use tracing::{debug as trace_debug, info as trace_info, instrument, span, Level};

/// Coordinate transformation parameters for tensor-to-mask conversion
#[derive(Debug, Clone)]
struct CoordinateTransformation {
    /// Scale factor used during preprocessing
    scale: f32,
    /// X offset for centering
    offset_x: u32,
    /// Y offset for centering
    offset_y: u32,
    /// Mask width in tensor coordinates
    mask_width: u32,
    /// Mask height in tensor coordinates
    mask_height: u32,
}

/// Backend type enumeration for runtime selection
#[derive(Clone, Debug, PartialEq)]
pub enum BackendType {
    /// ONNX Runtime backend (supports GPU acceleration)
    Onnx,
    /// Tract backend (pure Rust, no external dependencies)
    Tract,
}

/// Factory trait for creating inference backends
pub trait BackendFactory: Send + Sync {
    /// Create a backend instance of the specified type with the given model manager
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Unsupported backend types
    /// - Backend initialization failures
    /// - Model loading errors
    fn create_backend(
        &self,
        backend_type: BackendType,
        model_manager: ModelManager,
    ) -> Result<Box<dyn InferenceBackend>>;

    /// List available backend types
    fn available_backends(&self) -> Vec<BackendType>;
}

/// Default backend factory implementation
pub struct DefaultBackendFactory;

impl BackendFactory for DefaultBackendFactory {
    fn create_backend(
        &self,
        backend_type: BackendType,
        _model_manager: ModelManager,
    ) -> Result<Box<dyn InferenceBackend>> {
        match backend_type {
            BackendType::Onnx => {
                // This will be injected by the CLI crate which has access to bg-remove-onnx
                Err(BgRemovalError::invalid_config(
                    "ONNX backend not available in core. Must be injected by frontend.",
                ))
            },
            BackendType::Tract => {
                // This will be injected by the Web crate which has access to bg-remove-tract
                Err(BgRemovalError::invalid_config(
                    "Tract backend not available in core. Must be injected by frontend.",
                ))
            },
        }
    }

    fn available_backends(&self) -> Vec<BackendType> {
        vec![] // No backends available by default - must be injected
    }
}

/// Unified configuration for the background removal processor
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ProcessorConfig {
    /// Model specification (embedded or external)
    pub model_spec: ModelSpec,
    /// Backend type to use for inference
    pub backend_type: BackendType,
    /// Execution provider for the backend
    pub execution_provider: ExecutionProvider,
    /// Output format configuration
    pub output_format: OutputFormat,
    /// JPEG quality (0-100)
    pub jpeg_quality: u8,
    /// WebP quality (0-100)
    pub webp_quality: u8,
    /// Enable debug mode
    pub debug: bool,
    /// Number of intra-op threads (0 = auto)
    pub intra_threads: usize,
    /// Number of inter-op threads (0 = auto)
    pub inter_threads: usize,
    /// Preserve ICC color profiles from input images
    pub preserve_color_profiles: bool,
    /// Enable verbose progress reporting
    pub verbose_progress: bool,
    /// Disable all caches during processing
    pub disable_cache: bool,
}

impl ProcessorConfig {
    /// Create a new processor configuration builder
    #[must_use]
    pub fn builder() -> ProcessorConfigBuilder {
        ProcessorConfigBuilder::new()
    }

    /// Convert to `RemovalConfig` for backward compatibility
    #[must_use]
    pub fn to_removal_config(&self) -> RemovalConfig {
        RemovalConfig {
            execution_provider: self.execution_provider,
            output_format: self.output_format,
            jpeg_quality: self.jpeg_quality,
            webp_quality: self.webp_quality,
            debug: self.debug,
            intra_threads: self.intra_threads,
            inter_threads: self.inter_threads,
            preserve_color_profiles: self.preserve_color_profiles,
            disable_cache: self.disable_cache,
            model_spec: self.model_spec.clone(),
            format_hint: None,
        }
    }
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            model_spec: ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp32".to_string()),
            },
            backend_type: BackendType::Onnx,
            execution_provider: ExecutionProvider::Auto,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: true,
            verbose_progress: false,
            disable_cache: false,
        }
    }
}

/// Builder for `ProcessorConfig`
pub struct ProcessorConfigBuilder {
    config: ProcessorConfig,
}

impl ProcessorConfigBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ProcessorConfig::default(),
        }
    }

    #[must_use]
    pub fn model_spec(mut self, model_spec: ModelSpec) -> Self {
        self.config.model_spec = model_spec;
        self
    }

    #[must_use]
    pub fn backend_type(mut self, backend_type: BackendType) -> Self {
        self.config.backend_type = backend_type;
        self
    }

    #[must_use]
    pub fn execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.config.execution_provider = provider;
        self
    }

    #[must_use]
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    #[must_use]
    pub fn jpeg_quality(mut self, quality: u8) -> Self {
        self.config.jpeg_quality = quality.clamp(0, 100);
        self
    }

    #[must_use]
    pub fn webp_quality(mut self, quality: u8) -> Self {
        self.config.webp_quality = quality.clamp(0, 100);
        self
    }

    #[must_use]
    pub fn debug(mut self, debug: bool) -> Self {
        self.config.debug = debug;
        self
    }

    #[must_use]
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self
    }

    #[must_use]
    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.config.inter_threads = threads;
        self
    }

    #[must_use]
    pub fn preserve_color_profiles(mut self, preserve: bool) -> Self {
        self.config.preserve_color_profiles = preserve;
        self
    }

    #[must_use]
    pub fn verbose_progress(mut self, verbose: bool) -> Self {
        self.config.verbose_progress = verbose;
        self
    }

    #[must_use]
    pub fn disable_cache(mut self, disable: bool) -> Self {
        self.config.disable_cache = disable;
        self
    }

    /// Build the processor configuration
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Invalid quality values (> 100)
    /// - Configuration validation failures
    pub fn build(self) -> Result<ProcessorConfig> {
        // Validate configuration
        if self.config.jpeg_quality > 100 {
            return Err(BgRemovalError::invalid_config("JPEG quality must be 0-100"));
        }
        if self.config.webp_quality > 100 {
            return Err(BgRemovalError::invalid_config("WebP quality must be 0-100"));
        }

        Ok(self.config)
    }
}

impl Default for ProcessorConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified background removal processor that consolidates all business logic
pub struct BackgroundRemovalProcessor {
    config: ProcessorConfig,
    backend_factory: Box<dyn BackendFactory>,
    backend: Option<Box<dyn InferenceBackend>>,
    model_manager: Option<ModelManager>,
    initialized: bool,
    progress_tracker: Option<ProgressTracker>,
}

impl BackgroundRemovalProcessor {
    /// Create a new processor with the default backend factory
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Invalid processor configuration
    /// - Backend factory initialization failures
    pub fn new(config: ProcessorConfig) -> Result<Self> {
        Self::with_factory(config, Box::new(DefaultBackendFactory))
    }

    /// Create a new processor with a custom backend factory
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Invalid processor configuration
    /// - Backend factory initialization failures
    pub fn with_factory(
        config: ProcessorConfig,
        backend_factory: Box<dyn BackendFactory>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            backend_factory,
            backend: None,
            model_manager: None,
            initialized: false,
            progress_tracker: None,
        })
    }

    /// Initialize the processor with the configured model and backend
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Model loading failures
    /// - Backend initialization errors
    /// - Execution provider setup failures
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing background removal processor");
        debug!("Model spec: {:?}", self.config.model_spec);
        debug!("Backend type: {:?}", self.config.backend_type);
        debug!("Execution provider: {:?}", self.config.execution_provider);

        // Create model manager
        let model_manager = ModelManager::from_spec_with_provider(
            &self.config.model_spec,
            Some(&self.config.execution_provider),
        )?;

        // Create backend
        let mut backend = self
            .backend_factory
            .create_backend(self.config.backend_type.clone(), model_manager)?;

        // Initialize backend
        let removal_config = self.config.to_removal_config();
        let _model_load_time = backend.initialize(&removal_config)?;

        let model_manager_copy = ModelManager::from_spec_with_provider(
            &self.config.model_spec,
            Some(&self.config.execution_provider),
        )?;
        self.model_manager = Some(model_manager_copy);
        self.backend = Some(backend);
        self.initialized = true;

        info!("Background removal processor initialized successfully");
        Ok(())
    }

    /// Process an image file for background removal
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - File I/O errors when reading input
    /// - Image format parsing failures
    /// - Processing and inference errors
    pub async fn process_file<P: AsRef<Path>>(&mut self, input_path: P) -> Result<RemovalResult> {
        let input_path_ref = input_path.as_ref();

        // Report image loading progress
        if let Some(ref mut tracker) = self.progress_tracker {
            tracker.report_stage(ProcessingStage::ImageLoading);
        }

        // Extract color profile first if enabled (file-specific feature)
        let color_profile = if self.config.preserve_color_profiles {
            match crate::services::ImageIOService::extract_color_profile(input_path_ref) {
                Ok(Some(profile)) => {
                    trace_debug!(
                        "Extracted ICC color profile ({}, {} bytes) from input file",
                        profile.color_space,
                        profile.data_size()
                    );
                    Some(profile)
                },
                Ok(None) => {
                    trace_debug!("No color profile found in input file");
                    None
                },
                Err(e) => {
                    trace_debug!("Failed to extract color profile from input file: {}", e);
                    None
                },
            }
        } else {
            None
        };

        // Load the image directly for processing with color profile
        let image = image::open(input_path_ref)
            .map_err(|e| BgRemovalError::processing(format!("Failed to load image file: {}", e)))?;

        // Process the image with the extracted color profile
        self.process_image_with_profile(&image, color_profile)
    }

    /// Process a `DynamicImage` directly for background removal
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Image preprocessing failures
    /// - Inference execution errors
    /// - Mask generation and application errors
    pub fn process_image(&mut self, image: &DynamicImage) -> Result<RemovalResult> {
        self.process_image_with_profile(image, None)
    }

    /// Process a `DynamicImage` with optional color profile for background removal
    ///
    /// # Errors
    ///
    /// Returns `BgRemovalError` for:
    /// - Image preprocessing failures
    /// - Inference execution errors
    /// - Mask generation and application errors
    #[instrument(
        skip(self, image, color_profile),
        fields(
            backend = ?self.config.backend_type,
            model = %self.config.model_spec.source.display_name(),
            dimensions = %format!("{}x{}", image.width(), image.height())
        )
    )]
    pub fn process_image_with_profile(
        &mut self,
        image: &DynamicImage,
        color_profile: Option<crate::types::ColorProfile>,
    ) -> Result<RemovalResult> {
        if !self.initialized {
            self.initialize()?;
        }

        let mut timings = ProcessingTimings::default();
        let total_start = Instant::now();
        let original_dimensions = (image.width(), image.height());
        let removal_config = self.config.to_removal_config();

        trace_info!(
            backend = ?self.config.backend_type,
            model = %self.config.model_spec.source.display_name(),
            "🎯 Starting image processing"
        );

        if let Some(ref profile) = color_profile {
            trace_debug!(
                color_space = %profile.color_space,
                profile_size_bytes = %profile.data_size(),
                "🎨 Processing with ICC color profile"
            );
        }

        // Preprocess image for inference
        let input_tensor = {
            let _span = span!(
                Level::DEBUG,
                "preprocessing",
                original_width = %original_dimensions.0,
                original_height = %original_dimensions.1
            )
            .entered();
            self.preprocess_image_for_inference(image, &mut timings)?
        };

        // Perform inference
        let output_tensor = {
            let _span = span!(
                Level::INFO,
                "inference",
                backend = ?self.config.backend_type,
                model = %self.config.model_spec.source.display_name()
            )
            .entered();
            self.perform_inference(&input_tensor, &mut timings)?
        };

        // Generate mask and apply background removal
        let (mask, result_image) = {
            let _span = span!(
                Level::DEBUG,
                "background_removal",
                width = %original_dimensions.0,
                height = %original_dimensions.1
            )
            .entered();
            self.generate_mask_and_remove_background(
                &output_tensor,
                image,
                original_dimensions,
                &removal_config,
                &mut timings,
            )?
        };

        // Finalize result with format conversion and metadata
        self.finalize_processing_result(
            result_image,
            mask,
            original_dimensions,
            color_profile,
            timings,
            total_start,
        )
    }

    /// Process image data from bytes
    ///
    /// This method accepts raw image bytes and processes them for background removal,
    /// making it suitable for web servers, memory-based processing, and scenarios
    /// where files aren't available.
    ///
    /// # Arguments
    /// * `image_bytes` - Raw image data as bytes (JPEG, PNG, WebP, BMP, TIFF)
    ///
    /// # Returns
    /// A `RemovalResult` containing the processed image, mask, and metadata
    ///
    /// # Examples
    /// ```rust,no_run
    /// use imgly_bgremove::{BackgroundRemovalProcessor, ProcessorConfigBuilder, ModelSpec, ModelSource, BackendType};
    ///
    /// # async fn example(image_data: Vec<u8>) -> anyhow::Result<()> {
    /// let config = ProcessorConfigBuilder::new()
    ///     .model_spec(ModelSpec {
    ///         source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    ///         variant: None,
    ///     })
    ///     .backend_type(BackendType::Onnx)
    ///     .build()?;
    /// let mut processor = BackgroundRemovalProcessor::new(config)?;
    /// let result = processor.process_bytes(&image_data)?;
    /// let output_bytes = result.to_bytes(imgly_bgremove::OutputFormat::Png, 100)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `BgRemovalError` for:
    /// - Image decoding failures
    /// - Inference execution errors
    /// - Memory allocation failures
    pub fn process_bytes(&mut self, image_bytes: &[u8]) -> Result<RemovalResult> {
        // Load image from bytes
        let image = image::load_from_memory(image_bytes).map_err(|e| {
            BgRemovalError::processing(format!("Failed to decode image from bytes: {}", e))
        })?;

        // Process the loaded image
        self.process_image(&image)
    }

    /// Process image from an async reader stream
    ///
    /// This method accepts any async readable stream, making it suitable for processing
    /// images from network streams, large files, or any other async data source.
    ///
    /// # Arguments
    /// * `reader` - Any type implementing `AsyncRead + Unpin`
    /// * `format_hint` - Optional hint about the image format for better performance
    ///
    /// # Returns
    /// A `RemovalResult` containing the processed image, mask, and metadata
    ///
    /// # Examples
    /// ```rust,no_run
    /// use imgly_bgremove::{BackgroundRemovalProcessor, ProcessorConfigBuilder, ModelSpec, ModelSource, BackendType};
    /// use tokio::fs::File;
    /// use image::ImageFormat;
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
    ///
    /// let file = File::open("large_image.jpg").await?;
    /// let result = processor.process_reader(file, Some(ImageFormat::Jpeg)).await?;
    /// result.save_png("output.png")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `BgRemovalError` for:
    /// - Stream reading failures
    /// - Image decoding failures
    /// - Inference execution errors
    pub async fn process_reader<R: tokio::io::AsyncRead + Unpin>(
        &mut self,
        mut reader: R,
        _format_hint: Option<image::ImageFormat>,
    ) -> Result<RemovalResult> {
        use tokio::io::AsyncReadExt;

        // Read all data from the stream into memory
        // TODO: For very large images, consider streaming decode if image crate supports it
        let mut buffer = Vec::new();
        AsyncReadExt::read_to_end(&mut reader, &mut buffer)
            .await
            .map_err(|e| {
                BgRemovalError::processing(format!("Failed to read from stream: {}", e))
            })?;

        // Use the bytes-based processing
        self.process_bytes(&buffer)
    }

    /// Preprocess image for inference with timing
    fn preprocess_image_for_inference(
        &mut self,
        image: &DynamicImage,
        timings: &mut ProcessingTimings,
    ) -> Result<Array4<f32>> {
        if let Some(ref mut tracker) = self.progress_tracker {
            tracker.report_stage(ProcessingStage::Preprocessing);
        }

        let preprocess_start = Instant::now();

        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| BgRemovalError::processing("Backend not initialized"))?;

        let preprocessing_config = backend.get_preprocessing_config()?;
        let input_tensor =
            ImagePreprocessor::preprocess_for_inference(image, &preprocessing_config)?;

        timings.preprocessing_ms = preprocess_start.elapsed().as_millis() as u64;
        Ok(input_tensor)
    }

    /// Perform inference with timing
    fn perform_inference(
        &mut self,
        input_tensor: &Array4<f32>,
        timings: &mut ProcessingTimings,
    ) -> Result<Array4<f32>> {
        if let Some(ref mut tracker) = self.progress_tracker {
            tracker.report_stage(ProcessingStage::Inference);
        }

        let inference_start = Instant::now();

        let backend = self
            .backend
            .as_mut()
            .ok_or_else(|| BgRemovalError::processing("Backend not initialized"))?;

        let output_tensor = backend.infer(input_tensor)?;
        timings.inference_ms = inference_start.elapsed().as_millis() as u64;

        Ok(output_tensor)
    }

    /// Generate segmentation mask and apply background removal
    fn generate_mask_and_remove_background(
        &mut self,
        output_tensor: &Array4<f32>,
        image: &DynamicImage,
        original_dimensions: (u32, u32),
        removal_config: &RemovalConfig,
        timings: &mut ProcessingTimings,
    ) -> Result<(SegmentationMask, RgbaImage)> {
        // Generate mask
        if let Some(ref mut tracker) = self.progress_tracker {
            tracker.report_stage(ProcessingStage::MaskGeneration);
        }

        let postprocess_start = Instant::now();
        let mask = Self::tensor_to_mask(output_tensor, original_dimensions)?;

        // Apply background removal
        if let Some(ref mut tracker) = self.progress_tracker {
            tracker.report_stage(ProcessingStage::BackgroundRemoval);
        }

        let result_image = Self::apply_background_removal(image, &mask, removal_config);
        timings.postprocessing_ms = postprocess_start.elapsed().as_millis() as u64;

        Ok((mask, result_image))
    }

    /// Finalize processing result with format conversion and metadata
    fn finalize_processing_result(
        &mut self,
        result_image: RgbaImage,
        mask: SegmentationMask,
        original_dimensions: (u32, u32),
        color_profile: Option<crate::types::ColorProfile>,
        mut timings: ProcessingTimings,
        total_start: Instant,
    ) -> Result<RemovalResult> {
        // Format conversion
        if let Some(ref mut tracker) = self.progress_tracker {
            tracker.report_stage(ProcessingStage::FormatConversion);
        }

        let final_image =
            OutputFormatHandler::convert_format(result_image, self.config.output_format)?;

        // Finalize timing and metadata
        timings.total_ms = total_start.elapsed().as_millis() as u64;
        timings.image_decode_ms = 0; // Already decoded

        let mut metadata = ProcessingMetadata::new("unified_processor".to_string());
        metadata.model_precision = format!("{:?}", self.config.backend_type);
        metadata.set_detailed_timings(timings.clone());

        let mut result = RemovalResult::new(final_image, mask, original_dimensions, metadata);
        result.color_profile = color_profile;

        // Report completion
        if let Some(ref mut tracker) = self.progress_tracker {
            tracker.report_stage(ProcessingStage::Completed);
            tracker.report_completion(timings);
        }

        Ok(result)
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    /// Check if the processor is initialized
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get available backends from the factory
    #[must_use]
    pub fn available_backends(&self) -> Vec<BackendType> {
        self.backend_factory.available_backends()
    }

    /// Convert output tensor to segmentation mask with proper aspect ratio handling
    fn tensor_to_mask(
        tensor: &Array4<f32>,
        original_dimensions: (u32, u32),
    ) -> Result<SegmentationMask> {
        Self::validate_tensor_shape(tensor)?;
        let transformation = Self::calculate_inverse_transformation(tensor, original_dimensions);
        let mask_data =
            Self::extract_mask_values_from_tensor(tensor, original_dimensions, &transformation);
        Ok(SegmentationMask::new(mask_data, original_dimensions))
    }

    /// Validate tensor shape for mask generation
    #[allow(clippy::get_first)]
    fn validate_tensor_shape(tensor: &Array4<f32>) -> Result<()> {
        let shape = tensor.shape();
        if shape.len() < 4
            || shape.get(0).copied().unwrap_or(0) != 1
            || shape.get(1).copied().unwrap_or(0) != 1
        {
            return Err(BgRemovalError::processing("Invalid output tensor shape"));
        }
        Ok(())
    }

    /// Calculate transformation parameters for mapping coordinates
    fn calculate_inverse_transformation(
        tensor: &Array4<f32>,
        original_dimensions: (u32, u32),
    ) -> CoordinateTransformation {
        let shape = tensor.shape();
        let mask_height = shape.get(2).copied().unwrap_or(0);
        let mask_width = shape.get(3).copied().unwrap_or(0);
        let (orig_width, orig_height) = original_dimensions;

        // Reproduce the preprocessing calculations to get the inverse transformation
        let target_size = mask_width; // Assuming square tensor (typical for models)
        let target_size_f32 = target_size as f32;
        let orig_width_f32 = orig_width as f32;
        let orig_height_f32 = orig_height as f32;

        // Calculate the same scale factor used during preprocessing
        let scale = target_size_f32
            .min((target_size_f32 / orig_width_f32).min(target_size_f32 / orig_height_f32));

        let scaled_width = (orig_width_f32 * scale).round() as u32;
        let scaled_height = (orig_height_f32 * scale).round() as u32;

        // Calculate the centering offsets used during preprocessing
        let offset_x = (target_size as u32 - scaled_width) / 2;
        let offset_y = (target_size as u32 - scaled_height) / 2;

        CoordinateTransformation {
            scale,
            offset_x,
            offset_y,
            mask_width: mask_width as u32,
            mask_height: mask_height as u32,
        }
    }

    /// Extract mask values from tensor using coordinate transformation
    fn extract_mask_values_from_tensor(
        tensor: &Array4<f32>,
        original_dimensions: (u32, u32),
        transformation: &CoordinateTransformation,
    ) -> Vec<u8> {
        let (orig_width, orig_height) = original_dimensions;
        let mut mask_data = Vec::with_capacity((orig_width * orig_height) as usize);

        for y in 0..orig_height {
            for x in 0..orig_width {
                let mask_value = Self::get_tensor_value_at_coordinate(tensor, x, y, transformation);
                mask_data.push((mask_value.clamp(0.0, 1.0) * 255.0) as u8);
            }
        }

        mask_data
    }

    /// Get tensor value at mapped coordinates
    fn get_tensor_value_at_coordinate(
        tensor: &Array4<f32>,
        x: u32,
        y: u32,
        transformation: &CoordinateTransformation,
    ) -> f32 {
        // Map original coordinates to scaled coordinates
        let scaled_x = (x as f32 * transformation.scale).round() as u32;
        let scaled_y = (y as f32 * transformation.scale).round() as u32;

        // Map scaled coordinates to tensor coordinates (accounting for centering)
        let tensor_x = scaled_x + transformation.offset_x;
        let tensor_y = scaled_y + transformation.offset_y;

        if tensor_x < transformation.mask_width && tensor_y < transformation.mask_height {
            // Safe indexing with bounds check
            tensor
                .get([0, 0, tensor_y as usize, tensor_x as usize])
                .copied()
                .unwrap_or(0.0)
        } else {
            0.0 // Outside the model's prediction area
        }
    }

    /// Apply background removal using the segmentation mask
    fn apply_background_removal(
        image: &DynamicImage,
        mask: &SegmentationMask,
        _config: &RemovalConfig,
    ) -> RgbaImage {
        let rgba_image = image.to_rgba8();
        let (width, height) = rgba_image.dimensions();
        let mut result = ImageBuffer::new(width, height);

        for (x, y, pixel) in rgba_image.enumerate_pixels() {
            let pixel_index = (y * width + x) as usize;
            let mask_value = mask.data.get(pixel_index).copied().unwrap_or(0);
            let alpha = mask_value;

            if alpha > 0 {
                // Keep foreground pixel
                result.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], alpha]));
            } else {
                // Apply transparent background
                result.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
            }
        }

        result
    }

    /// Extract foreground segmentation mask only without applying background removal
    pub async fn segment_foreground<P: AsRef<Path>>(
        &mut self,
        input_path: P,
    ) -> Result<SegmentationMask> {
        if !self.initialized {
            self.initialize()?;
        }

        let image = image::open(&input_path)
            .map_err(|e| BgRemovalError::processing(format!("Failed to load image: {}", e)))?;

        let backend = self
            .backend
            .as_mut()
            .ok_or_else(|| BgRemovalError::processing("Backend not initialized"))?;

        // Preprocess
        let preprocessing_config = backend.get_preprocessing_config()?;
        let original_dimensions = image.dimensions();
        let input_tensor =
            ImagePreprocessor::preprocess_for_inference(&image, &preprocessing_config)?;

        // Inference
        let output_tensor = backend.infer(&input_tensor)?;

        // Convert to mask
        Self::tensor_to_mask(&output_tensor, original_dimensions)
    }

    /// Apply a pre-computed segmentation mask to an image for background removal
    pub async fn apply_mask<P: AsRef<Path>>(
        &self,
        input_path: P,
        mask: &SegmentationMask,
    ) -> Result<RemovalResult> {
        let image = image::open(&input_path)
            .map_err(|e| BgRemovalError::processing(format!("Failed to load image: {}", e)))?;

        let original_dimensions = image.dimensions();

        // Resize mask if needed
        let resized_mask = if mask.dimensions == original_dimensions {
            mask.clone()
        } else {
            mask.resize(original_dimensions.0, original_dimensions.1)?
        };

        let removal_config = self.config.to_removal_config();
        let result_image = Self::apply_background_removal(&image, &resized_mask, &removal_config);

        // Handle output format using the service
        let final_image =
            OutputFormatHandler::convert_format(result_image, self.config.output_format)?;

        let metadata = ProcessingMetadata::new("mask_application".to_string());
        Ok(RemovalResult::new(
            final_image,
            resized_mask,
            original_dimensions,
            metadata,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelSource, ModelSpec};

    #[test]
    fn test_processor_config_disable_cache() {
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: None,
        };

        // Test default (cache enabled)
        let config = ProcessorConfigBuilder::new()
            .model_spec(model_spec.clone())
            .build()
            .unwrap();
        assert!(!config.disable_cache);

        // Test disable cache
        let config = ProcessorConfigBuilder::new()
            .model_spec(model_spec.clone())
            .disable_cache(true)
            .build()
            .unwrap();
        assert!(config.disable_cache);

        // Test to_removal_config conversion
        let removal_config = config.to_removal_config();
        assert!(removal_config.disable_cache);
    }

    #[test]
    fn test_processor_config_builder_chain() {
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: None,
        };

        let config = ProcessorConfigBuilder::new()
            .model_spec(model_spec)
            .disable_cache(true)
            .debug(true)
            .jpeg_quality(95)
            .build()
            .unwrap();

        assert!(config.disable_cache);
        assert!(config.debug);
        assert_eq!(config.jpeg_quality, 95);
    }
}
