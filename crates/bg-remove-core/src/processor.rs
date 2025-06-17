//! Unified background removal processor
//!
//! This module provides the main `BackgroundRemovalProcessor` that consolidates
//! all business logic for background removal operations. This processor is used
//! by both CLI and Web frontends to ensure consistent behavior.

use crate::{
    backends::MockBackend,
    config::{BackgroundColor, ExecutionProvider, OutputFormat, RemovalConfig},
    error::{BgRemovalError, Result},
    inference::InferenceBackend,
    models::{ModelManager, ModelSource, ModelSpec},
    types::RemovalResult,
};
use image::DynamicImage;
use log::{debug, info};
use std::path::Path;

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
    /// Background color for non-transparent formats
    pub background_color: BackgroundColor,
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
    /// Color management configuration
    pub color_management: crate::config::ColorManagementConfig,
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
            background_color: self.background_color,
            jpeg_quality: self.jpeg_quality,
            webp_quality: self.webp_quality,
            debug: self.debug,
            intra_threads: self.intra_threads,
            inter_threads: self.inter_threads,
            color_management: self.color_management.clone(),
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
            background_color: BackgroundColor::white(),
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            color_management: crate::config::ColorManagementConfig::default(),
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
    
    pub fn background_color(mut self, color: BackgroundColor) -> Self {
        self.config.background_color = color;
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
    
    pub fn color_management(mut self, color_management: crate::config::ColorManagementConfig) -> Self {
        self.config.color_management = color_management;
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
        
        let removal_config = self.config.to_removal_config();
        
        // For now, delegate to the existing remove_background function
        // TODO: Refactor to use the unified processor directly once we solve the backend sharing issue
        crate::remove_background(input_path, &removal_config).await
    }
    
    /// Process a DynamicImage directly for background removal
    pub fn process_image(&mut self, _image: DynamicImage) -> Result<RemovalResult> {
        if !self.initialized {
            self.initialize()?;
        }
        
        let removal_config = self.config.to_removal_config();
        
        // For now, delegate to the existing process_image function
        // TODO: Refactor to use the unified processor directly once we solve the backend sharing issue
        crate::process_image(_image, &removal_config)
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
}