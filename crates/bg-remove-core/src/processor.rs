//! Unified background removal processor
//!
//! This module provides the main `BackgroundRemovalProcessor` that consolidates
//! all business logic for background removal operations. This processor is used
//! by both CLI and Web frontends to ensure consistent behavior.

use crate::{
    backends::MockBackend,
    config::{ExecutionProvider, OutputFormat, RemovalConfig},
    error::{BgRemovalError, Result},
    inference::InferenceBackend,
    models::{ModelManager, ModelSource, ModelSpec},
    types::{RemovalResult, ProcessingTimings, ProcessingMetadata, SegmentationMask},
    utils::ImagePreprocessor,
};
use image::{DynamicImage, ImageBuffer, RgbaImage, GenericImageView};
use log::{debug, info};
use std::path::Path;
use ndarray::Array4;
use instant::Instant;

/// Backend type enumeration for runtime selection
#[derive(Clone, Debug, PartialEq)]
pub enum BackendType {
    /// ONNX Runtime backend (supports GPU acceleration)
    Onnx,
    /// Tract backend (pure Rust, WASM compatible)
    Tract,
    /// Mock backend (for testing and debugging)
    Mock,
}

/// Factory trait for creating inference backends
pub trait BackendFactory: Send + Sync {
    /// Create a backend instance of the specified type with the given model manager
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
            BackendType::Mock => Ok(Box::new(MockBackend::new())),
            BackendType::Onnx => {
                // This will be injected by the CLI crate which has access to bg-remove-onnx
                Err(BgRemovalError::invalid_config(
                    "ONNX backend not available in core. Must be injected by frontend."
                ))
            }
            BackendType::Tract => {
                // This will be injected by the Web crate which has access to bg-remove-tract
                Err(BgRemovalError::invalid_config(
                    "Tract backend not available in core. Must be injected by frontend."
                ))
            }
        }
    }
    
    fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::Mock]
    }
}

/// Unified configuration for the background removal processor
#[derive(Debug, Clone)]
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
}

impl ProcessorConfig {
    /// Create a new processor configuration builder
    pub fn builder() -> ProcessorConfigBuilder {
        ProcessorConfigBuilder::new()
    }
    
    /// Convert to RemovalConfig for backward compatibility
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
        }
    }
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            model_spec: ModelSpec {
                source: ModelSource::Embedded("isnet-fp16".to_string()),
                variant: None,
            },
            backend_type: BackendType::Mock,
            execution_provider: ExecutionProvider::Auto,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profiles: true,
        }
    }
}

/// Builder for ProcessorConfig
pub struct ProcessorConfigBuilder {
    config: ProcessorConfig,
}

impl ProcessorConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ProcessorConfig::default(),
        }
    }
    
    pub fn model_spec(mut self, model_spec: ModelSpec) -> Self {
        self.config.model_spec = model_spec;
        self
    }
    
    pub fn backend_type(mut self, backend_type: BackendType) -> Self {
        self.config.backend_type = backend_type;
        self
    }
    
    pub fn execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.config.execution_provider = provider;
        self
    }
    
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }
    
    
    pub fn jpeg_quality(mut self, quality: u8) -> Self {
        self.config.jpeg_quality = quality.clamp(0, 100);
        self
    }
    
    pub fn webp_quality(mut self, quality: u8) -> Self {
        self.config.webp_quality = quality.clamp(0, 100);
        self
    }
    
    pub fn debug(mut self, debug: bool) -> Self {
        self.config.debug = debug;
        self
    }
    
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.config.intra_threads = threads;
        self
    }
    
    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.config.inter_threads = threads;
        self
    }
    
    pub fn preserve_color_profiles(mut self, preserve: bool) -> Self {
        self.config.preserve_color_profiles = preserve;
        self
    }
    
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
}

impl BackgroundRemovalProcessor {
    /// Create a new processor with the default backend factory
    pub fn new(config: ProcessorConfig) -> Result<Self> {
        Self::with_factory(config, Box::new(DefaultBackendFactory))
    }
    
    /// Create a new processor with a custom backend factory
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
        })
    }
    
    /// Initialize the processor with the configured model and backend
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
        let mut backend = self.backend_factory.create_backend(
            self.config.backend_type.clone(),
            model_manager,
        )?;
        
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
    pub async fn process_file<P: AsRef<Path>>(&mut self, input_path: P) -> Result<RemovalResult> {
        if !self.initialized {
            self.initialize()?;
        }
        
        // Load and process the image using the configured backend
        let image = image::open(&input_path)
            .map_err(|e| BgRemovalError::processing(format!("Failed to load image: {}", e)))?;
        
        self.process_image(image)
    }
    
    /// Process a DynamicImage directly for background removal
    pub fn process_image(&mut self, image: DynamicImage) -> Result<RemovalResult> {
        if !self.initialized {
            self.initialize()?;
        }
        
        let backend = self.backend.as_mut()
            .ok_or_else(|| BgRemovalError::processing("Backend not initialized"))?;
        
        let removal_config = self.config.to_removal_config();
        
        // Initialize timing
        let mut timings = ProcessingTimings::default();
        let total_start = Instant::now();
        
        info!("Starting processing: {} - Backend: {:?}", 
              "DynamicImage", 
              self.config.backend_type);
        
        // 1. Extract color profile
        let _color_profile: Option<crate::types::ColorProfile> = if removal_config.preserve_color_profiles {
            None // TODO: Implement color profile extraction for DynamicImage
        } else {
            None
        };
        
        // 2. Preprocessing
        let preprocess_start = Instant::now();
        let original_dimensions = (image.width(), image.height());
        
        // Get preprocessing config from backend before borrowing mutably
        let preprocessing_config = backend.get_preprocessing_config()?;
        let input_tensor = ImagePreprocessor::preprocess_for_inference(&image, &preprocessing_config)?;
        timings.preprocessing_ms = preprocess_start.elapsed().as_millis() as u64;
        
        // 3. Inference
        let inference_start = Instant::now();
        let output_tensor = backend.infer(&input_tensor)?;
        timings.inference_ms = inference_start.elapsed().as_millis() as u64;
        
        // 4. Postprocessing
        let postprocess_start = Instant::now();
        let mask = self.tensor_to_mask(&output_tensor, original_dimensions)?;
        let result_image = self.apply_background_removal(&image, &mask, &removal_config)?;
        timings.postprocessing_ms = postprocess_start.elapsed().as_millis() as u64;
        
        timings.total_ms = total_start.elapsed().as_millis() as u64;
        timings.image_decode_ms = 0; // Already decoded
        
        let mut metadata = ProcessingMetadata::new("unified_processor".to_string());
        metadata.model_precision = format!("{:?}", self.config.backend_type);
        metadata.set_detailed_timings(timings);
        
        // Handle output format
        let final_image = match self.config.output_format {
            OutputFormat::Png | OutputFormat::Rgba8 | OutputFormat::Tiff => {
                DynamicImage::ImageRgba8(result_image)
            },
            OutputFormat::Jpeg => {
                // Convert RGBA to RGB by dropping alpha channel
                let (width, height) = result_image.dimensions();
                let mut rgb_image = ImageBuffer::new(width, height);
                for (x, y, pixel) in result_image.enumerate_pixels() {
                    rgb_image.put_pixel(x, y, image::Rgb([pixel[0], pixel[1], pixel[2]]));
                }
                DynamicImage::ImageRgb8(rgb_image)
            },
            OutputFormat::WebP => {
                DynamicImage::ImageRgba8(result_image)
            },
        };
        
        let mut result = RemovalResult::new(
            final_image,
            mask,
            original_dimensions,
            metadata,
        );
        result.color_profile = _color_profile;
        Ok(result)
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }
    
    /// Check if the processor is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    /// Get available backends from the factory
    pub fn available_backends(&self) -> Vec<BackendType> {
        self.backend_factory.available_backends()
    }
    
    
    /// Convert output tensor to segmentation mask with proper aspect ratio handling
    fn tensor_to_mask(&self, tensor: &Array4<f32>, original_dimensions: (u32, u32)) -> Result<SegmentationMask> {
        let shape = tensor.shape();
        if shape[0] != 1 || shape[1] != 1 {
            return Err(BgRemovalError::processing("Invalid output tensor shape"));
        }
        
        let mask_height = shape[2];
        let mask_width = shape[3];
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
        
        // Create mask data with proper inverse transformation
        let mut mask_data = Vec::with_capacity((orig_width * orig_height) as usize);
        
        for y in 0..orig_height {
            for x in 0..orig_width {
                // Map original coordinates to scaled coordinates
                let scaled_x = (x as f32 * scale).round() as u32;
                let scaled_y = (y as f32 * scale).round() as u32;
                
                // Map scaled coordinates to tensor coordinates (accounting for centering)
                let tensor_x = scaled_x + offset_x;
                let tensor_y = scaled_y + offset_y;
                
                let mask_value = if tensor_x < mask_width as u32 && tensor_y < mask_height as u32 {
                    // Safe indexing with bounds check
                    if let Some(value) = tensor.get([0, 0, tensor_y as usize, tensor_x as usize]) {
                        *value
                    } else {
                        0.0
                    }
                } else {
                    0.0 // Outside the model's prediction area
                };
                
                mask_data.push((mask_value.clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
        
        Ok(SegmentationMask::new(mask_data, original_dimensions))
    }
    
    /// Apply background removal using the segmentation mask
    fn apply_background_removal(
        &self, 
        image: &DynamicImage, 
        mask: &SegmentationMask, 
        _config: &RemovalConfig
    ) -> Result<RgbaImage> {
        let rgba_image = image.to_rgba8();
        let (width, height) = rgba_image.dimensions();
        let mut result = ImageBuffer::new(width, height);
        
        for (x, y, pixel) in rgba_image.enumerate_pixels() {
            let pixel_index = (y * width + x) as usize;
            let mask_value = if pixel_index < mask.data.len() {
                mask.data[pixel_index]
            } else {
                0
            };
            let alpha = mask_value;
            
            if alpha > 0 {
                // Keep foreground pixel
                result.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], alpha]));
            } else {
                // Apply transparent background
                result.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
            }
        }
        
        Ok(result)
    }
    
    /// Extract foreground segmentation mask only without applying background removal
    pub async fn segment_foreground<P: AsRef<Path>>(&mut self, input_path: P) -> Result<SegmentationMask> {
        if !self.initialized {
            self.initialize()?;
        }
        
        let image = image::open(&input_path)
            .map_err(|e| BgRemovalError::processing(format!("Failed to load image: {}", e)))?;
        
        let backend = self.backend.as_mut()
            .ok_or_else(|| BgRemovalError::processing("Backend not initialized"))?;
        
        // Preprocess
        let preprocessing_config = backend.get_preprocessing_config()?;
        let original_dimensions = image.dimensions();
        let input_tensor = ImagePreprocessor::preprocess_for_inference(&image, &preprocessing_config)?;
        
        // Inference
        let output_tensor = backend.infer(&input_tensor)?;
        
        // Convert to mask
        self.tensor_to_mask(&output_tensor, original_dimensions)
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
        let result_image = self.apply_background_removal(&image, &resized_mask, &removal_config)?;
        
        // Handle output format
        let final_image = match self.config.output_format {
            OutputFormat::Png | OutputFormat::Rgba8 | OutputFormat::Tiff => {
                DynamicImage::ImageRgba8(result_image)
            },
            OutputFormat::Jpeg => {
                // Convert RGBA to RGB by dropping alpha channel
                let (width, height) = result_image.dimensions();
                let mut rgb_image = ImageBuffer::new(width, height);
                for (x, y, pixel) in result_image.enumerate_pixels() {
                    rgb_image.put_pixel(x, y, image::Rgb([pixel[0], pixel[1], pixel[2]]));
                }
                DynamicImage::ImageRgb8(rgb_image)
            },
            OutputFormat::WebP => {
                DynamicImage::ImageRgba8(result_image)
            },
        };
        
        let metadata = ProcessingMetadata::new("mask_application".to_string());
        Ok(RemovalResult::new(
            final_image,
            resized_mask,
            original_dimensions,
            metadata,
        ))
    }
}