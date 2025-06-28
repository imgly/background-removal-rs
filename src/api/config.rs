//! Library configuration API
//!
//! Provides CLI-equivalent configuration options through a builder pattern.

use crate::config::{ExecutionProvider, OutputFormat};
use crate::models::ModelSpec;
use crate::error::{BgRemovalError, Result};
use std::sync::Arc;

/// Verbose logging levels for the library
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerboseLevel {
    None,
    Info,
    Debug,
    Trace,
}

/// High-level library configuration matching CLI options
#[derive(Clone)]
pub struct LibraryConfig {
    pub model: ModelSpec,
    pub execution_provider: ExecutionProvider,
    pub output_format: OutputFormat,
    pub jpeg_quality: u8,
    pub webp_quality: u8,
    pub preserve_color_profiles: bool,
    pub threads: Option<usize>,
    pub disable_cache: bool,
    pub progress_reporter: Option<Arc<dyn super::progress::ProgressReporter>>,
    pub verbose_level: VerboseLevel,
}

impl std::fmt::Debug for LibraryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibraryConfig")
            .field("model", &self.model)
            .field("execution_provider", &self.execution_provider)
            .field("output_format", &self.output_format)
            .field("jpeg_quality", &self.jpeg_quality)
            .field("webp_quality", &self.webp_quality)
            .field("preserve_color_profiles", &self.preserve_color_profiles)
            .field("threads", &self.threads)
            .field("disable_cache", &self.disable_cache)
            .field("progress_reporter", &self.progress_reporter.as_ref().map(|_| "Some(...)"))
            .field("verbose_level", &self.verbose_level)
            .finish()
    }
}

/// Builder for LibraryConfig
pub struct LibraryConfigBuilder {
    config: LibraryConfig,
}

impl LibraryConfig {
    /// Create a new configuration builder
    pub fn new() -> LibraryConfigBuilder {
        LibraryConfigBuilder::new()
    }

    /// Create default configuration for Auto execution provider
    pub fn default_auto(model: ModelSpec) -> Self {
        Self::new().model(model).execution_provider(ExecutionProvider::Auto).build().unwrap()
    }

    /// Create default configuration for CPU execution provider
    pub fn default_cpu(model: ModelSpec) -> Self {
        Self::new().model(model).execution_provider(ExecutionProvider::Cpu).build().unwrap()
    }

    /// Create default configuration for CoreML execution provider
    pub fn default_coreml(model: ModelSpec) -> Self {
        Self::new().model(model).execution_provider(ExecutionProvider::CoreMl).build().unwrap()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.jpeg_quality > 100 {
            return Err(BgRemovalError::invalid_config("JPEG quality must be 0-100"));
        }
        if self.webp_quality > 100 {
            return Err(BgRemovalError::invalid_config("WebP quality must be 0-100"));
        }
        Ok(())
    }
}

impl Default for LibraryConfig {
    fn default() -> Self {
        use crate::models::{ModelSource, ModelSpec};
        
        LibraryConfig {
            model: ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: Some("fp16".to_string()),
            },
            execution_provider: ExecutionProvider::Auto,
            output_format: OutputFormat::Png,
            jpeg_quality: 90,
            webp_quality: 85,
            preserve_color_profiles: true,
            threads: None,
            disable_cache: false,
            progress_reporter: None,
            verbose_level: VerboseLevel::None,
        }
    }
}

impl LibraryConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: LibraryConfig::default(),
        }
    }

    /// Set the model specification
    pub fn model(mut self, model: ModelSpec) -> Self {
        self.config.model = model;
        self
    }

    /// Set the execution provider
    pub fn execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.config.execution_provider = provider;
        self
    }

    /// Set the output format
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Set JPEG quality (0-100)
    pub fn jpeg_quality(mut self, quality: u8) -> Self {
        self.config.jpeg_quality = quality;
        self
    }

    /// Set WebP quality (0-100)
    pub fn webp_quality(mut self, quality: u8) -> Self {
        self.config.webp_quality = quality;
        self
    }

    /// Enable or disable color profile preservation
    pub fn preserve_color_profiles(mut self, preserve: bool) -> Self {
        self.config.preserve_color_profiles = preserve;
        self
    }

    /// Set number of threads (None for auto-detection)
    pub fn threads(mut self, threads: usize) -> Self {
        self.config.threads = Some(threads);
        self
    }

    /// Disable caching
    pub fn disable_cache(mut self, disable: bool) -> Self {
        self.config.disable_cache = disable;
        self
    }

    /// Set progress reporter
    pub fn progress_reporter(mut self, reporter: Arc<dyn super::progress::ProgressReporter>) -> Self {
        self.config.progress_reporter = Some(reporter);
        self
    }

    /// Set verbose level
    pub fn verbose_level(mut self, level: VerboseLevel) -> Self {
        self.config.verbose_level = level;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<LibraryConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for LibraryConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}