//! WebAssembly (WASM) bindings for background removal in web browsers
//!
//! This crate provides WebAssembly bindings for the bg-remove-core library,
//! enabling background removal functionality directly in web browsers using
//! the unified processor with Tract backend.
//!
//! # Features
//!
//! - **Unified processing** - Uses the same core processor as CLI
//! - **Pure client-side processing** - No server uploads required
//! - **WebAssembly performance** - Near-native speeds in browsers
//! - **Tract backend** - Pure Rust ML inference with no external dependencies
//! - **Canvas API integration** - Direct ImageData processing
//! - **Web Workers support** - Non-blocking background processing
//! - **Cross-browser compatibility** - Works in all modern browsers
//!
//! # Example Usage
//!
//! ```javascript
//! import init, { BackgroundRemover } from './pkg/bg_remove_web.js';
//!
//! async function main() {
//!     // Initialize WASM module
//!     await init();
//!     
//!     // Create background remover instance
//!     const remover = new BackgroundRemover();
//!     await remover.initialize();
//!     
//!     // Process image from canvas
//!     const canvas = document.getElementById('myCanvas');
//!     const ctx = canvas.getContext('2d');
//!     const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
//!     
//!     // Remove background
//!     const result = await remover.removeBackground(imageData);
//!     
//!     // Display result
//!     ctx.putImageData(result, 0, 0);
//! }
//! ```

mod backend_factory;
mod config;

use wasm_bindgen::prelude::*;

// Import console.log for debugging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Macro for console logging
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

// Set up panic hook for better error messages
#[cfg(feature = "console_error_panic_hook")]
pub use console_error_panic_hook::set_once as set_panic_hook;

// Use wee_alloc for smaller WASM binary size
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use backend_factory::WebBackendFactory;
use config::WebConfigBuilder;
use bg_remove_core::{
    processor::BackgroundRemovalProcessor,
    models::get_available_embedded_models,
};
use js_sys::{Array, Promise};
use wasm_bindgen_futures::future_to_promise;
use web_sys::ImageData;

/// JavaScript-compatible error type for WASM
#[wasm_bindgen]
#[derive(Debug)]
pub struct WasmError {
    message: String,
}

impl WasmError {
    fn new(msg: &str) -> Self {
        Self {
            message: msg.to_string(),
        }
    }
}

#[wasm_bindgen]
impl WasmError {
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

impl From<anyhow::Error> for WasmError {
    fn from(err: anyhow::Error) -> Self {
        WasmError::new(&err.to_string())
    }
}

impl From<bg_remove_core::error::BgRemovalError> for WasmError {
    fn from(err: bg_remove_core::error::BgRemovalError) -> Self {
        WasmError::new(&err.to_string())
    }
}

/// Configuration options for background removal in browsers
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebRemovalConfig {
    /// Output format (png, jpeg, webp)
    output_format: String,
    /// Output format quality for JPEG (0-100)
    jpeg_quality: u8,
    /// Output format quality for WebP (0-100)  
    webp_quality: u8,
    /// Background color (RGB hex string, e.g., "#ffffff")
    background_color: String,
    /// Enable debug mode (additional logging)
    debug: bool,
    /// Number of intra-op threads for inference (0 = auto)
    intra_threads: u32,
    /// Number of inter-op threads for inference (0 = auto)
    inter_threads: u32,
    /// Whether to preserve color profiles
    preserve_color_profile: bool,
    /// Force sRGB output regardless of input profile
    force_srgb_output: bool,
    /// Fallback to sRGB when color space detection fails
    fallback_to_srgb: bool,
    /// Embed color profile in output when supported
    embed_profile_in_output: bool,
}

#[wasm_bindgen]
impl WebRemovalConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            output_format: "png".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            background_color: "#ffffff".to_string(),
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profile: true,
            force_srgb_output: false,
            fallback_to_srgb: true,
            embed_profile_in_output: true,
        }
    }

    // Output format
    #[wasm_bindgen(getter)]
    pub fn output_format(&self) -> String {
        self.output_format.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_output_format(&mut self, format: String) {
        // Validate format
        let valid_formats = ["png", "jpeg", "jpg", "webp", "tiff", "rgba8"];
        if valid_formats.contains(&format.to_lowercase().as_str()) {
            self.output_format = format.to_lowercase();
        } else {
            console_log!("Invalid output format: {}. Using 'png'", format);
            self.output_format = "png".to_string();
        }
    }

    // JPEG quality
    #[wasm_bindgen(getter)]
    pub fn jpeg_quality(&self) -> u8 {
        self.jpeg_quality
    }

    #[wasm_bindgen(setter)]
    pub fn set_jpeg_quality(&mut self, quality: u8) {
        self.jpeg_quality = quality.clamp(0, 100);
    }

    // WebP quality
    #[wasm_bindgen(getter)]
    pub fn webp_quality(&self) -> u8 {
        self.webp_quality
    }

    #[wasm_bindgen(setter)]
    pub fn set_webp_quality(&mut self, quality: u8) {
        self.webp_quality = quality.clamp(0, 100);
    }

    // Background color
    #[wasm_bindgen(getter)]
    pub fn background_color(&self) -> String {
        self.background_color.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_background_color(&mut self, color: String) {
        // Basic validation for hex color
        if color.starts_with('#') && (color.len() == 7 || color.len() == 4) {
            self.background_color = color;
        } else {
            console_log!("Invalid color format: {}. Using '#ffffff'", color);
            self.background_color = "#ffffff".to_string();
        }
    }

    // Debug mode
    #[wasm_bindgen(getter)]
    pub fn debug(&self) -> bool {
        self.debug
    }

    #[wasm_bindgen(setter)]
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    // Intra threads
    #[wasm_bindgen(getter)]
    pub fn intra_threads(&self) -> u32 {
        self.intra_threads
    }

    #[wasm_bindgen(setter)]
    pub fn set_intra_threads(&mut self, threads: u32) {
        self.intra_threads = threads;
    }

    // Inter threads
    #[wasm_bindgen(getter)]
    pub fn inter_threads(&self) -> u32 {
        self.inter_threads
    }

    #[wasm_bindgen(setter)]
    pub fn set_inter_threads(&mut self, threads: u32) {
        self.inter_threads = threads;
    }

    // Color profile preservation
    #[wasm_bindgen(getter)]
    pub fn preserve_color_profile(&self) -> bool {
        self.preserve_color_profile
    }

    #[wasm_bindgen(setter)]
    pub fn set_preserve_color_profile(&mut self, preserve: bool) {
        self.preserve_color_profile = preserve;
    }

    // Force sRGB output
    #[wasm_bindgen(getter)]
    pub fn force_srgb_output(&self) -> bool {
        self.force_srgb_output
    }

    #[wasm_bindgen(setter)]
    pub fn set_force_srgb_output(&mut self, force: bool) {
        self.force_srgb_output = force;
    }

    // Fallback to sRGB
    #[wasm_bindgen(getter)]
    pub fn fallback_to_srgb(&self) -> bool {
        self.fallback_to_srgb
    }

    #[wasm_bindgen(setter)]
    pub fn set_fallback_to_srgb(&mut self, fallback: bool) {
        self.fallback_to_srgb = fallback;
    }

    // Embed profile in output
    #[wasm_bindgen(getter)]
    pub fn embed_profile_in_output(&self) -> bool {
        self.embed_profile_in_output
    }

    #[wasm_bindgen(setter)]
    pub fn set_embed_profile_in_output(&mut self, embed: bool) {
        self.embed_profile_in_output = embed;
    }
    
    /// Convert to unified ProcessorConfig (moved to config module)
    pub(crate) fn to_processor_config(&self, model_name: Option<String>) -> Result<bg_remove_core::processor::ProcessorConfig, bg_remove_core::error::BgRemovalError> {
        WebConfigBuilder::from_web_config(self, model_name)
    }
    
    /// Validate configuration using core utilities
    pub fn validate(&self) -> Result<(), WasmError> {
        WebConfigBuilder::validate_web_config(self)
            .map_err(WasmError::from)
    }

    /// Create WebRemovalConfig from unified ProcessorConfig
    pub(crate) fn from_processor_config(config: &bg_remove_core::processor::ProcessorConfig) -> Self {
        WebConfigBuilder::to_web_config(config)
    }
}

impl Default for WebRemovalConfig {
    fn default() -> Self {
        WebRemovalConfig {
            output_format: "png".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            background_color: "#ffffff".to_string(),
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profile: true,
            force_srgb_output: false,
            fallback_to_srgb: true,
            embed_profile_in_output: true,
        }
    }
}

/// Processing progress information for long-running operations
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ProcessingProgress {
    stage: String,
    progress: f64,
    message: String,
}

#[wasm_bindgen]
impl ProcessingProgress {
    #[wasm_bindgen(getter)]
    pub fn stage(&self) -> String {
        self.stage.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn progress(&self) -> f64 {
        self.progress
    }

    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

/// Main background removal interface for web browsers
#[wasm_bindgen]
pub struct BackgroundRemover {
    processor: Option<BackgroundRemovalProcessor>,
    config: WebRemovalConfig,
    initialized: bool,
}

#[wasm_bindgen]
impl BackgroundRemover {
    /// Create a new BackgroundRemover instance with optional config
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<WebRemovalConfig>) -> Self {
        console_log!("🌐 Creating new BackgroundRemover instance");
        
        // Set up better panic messages
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        Self {
            processor: None,
            config: config.unwrap_or_default(),
            initialized: false,
        }
    }

    /// Initialize the background remover with a specific model
    /// Returns a Promise that resolves when initialization is complete
    #[allow(unsafe_code)]
    #[wasm_bindgen]
    pub fn initialize(&mut self, model_name: Option<String>) -> Promise {
        console_log!("🚀 Initializing BackgroundRemover with unified processor");
        
        let model_name = model_name.unwrap_or_else(|| {
            // Use the first available embedded model
            let available = get_available_embedded_models();
            if available.is_empty() {
                "isnet-fp16".to_string() // Default fallback
            } else {
                available[0].clone()
            }
        });

        console_log!("📦 Using model: {}", model_name);

        // Convert WebRemovalConfig to unified ProcessorConfig
        let config_result = self.config.to_processor_config(Some(model_name));
        
        let self_ptr = self as *mut Self;
        
        future_to_promise(async move {
            match config_result {
                Ok(config) => {
                    match Self::initialize_impl(config).await {
                        Ok((processor, final_config)) => {
                            console_log!("✅ BackgroundRemover initialized successfully");
                            
                            // SAFETY: We know this pointer is valid for the duration of the Promise
                            unsafe {
                                (*self_ptr).processor = Some(processor);
                                (*self_ptr).config = WebRemovalConfig::from_processor_config(&final_config);
                                (*self_ptr).initialized = true;
                            }
                            
                            Ok(JsValue::TRUE)
                        },
                        Err(e) => {
                            console_log!("❌ Failed to initialize BackgroundRemover: {}", e);
                            Err(JsValue::from(WasmError::from(e)))
                        }
                    }
                },
                Err(e) => {
                    console_log!("❌ Invalid configuration: {}", e);
                    Err(JsValue::from(WasmError::from(e)))
                }
            }
        })
    }

    /// Internal async initialization implementation using unified processor
    async fn initialize_impl(
        config: bg_remove_core::processor::ProcessorConfig,
    ) -> Result<(BackgroundRemovalProcessor, bg_remove_core::processor::ProcessorConfig), bg_remove_core::error::BgRemovalError> {
        // Create unified processor with Web backend factory
        let backend_factory = Box::new(WebBackendFactory::new());
        let processor = BackgroundRemovalProcessor::with_factory(config.clone(), backend_factory)?;
        
        console_log!("✅ Unified processor created successfully");
        
        Ok((processor, config))
    }

    /// Check if the remover is initialized and ready to process images
    #[wasm_bindgen]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the current configuration
    #[wasm_bindgen(getter)]
    pub fn config(&self) -> WebRemovalConfig {
        self.config.clone()
    }

    /// Set a new configuration
    #[wasm_bindgen(setter)]
    pub fn set_config(&mut self, config: WebRemovalConfig) {
        self.config = config;
    }

    /// Get list of available embedded models
    #[wasm_bindgen]
    pub fn get_available_models() -> Array {
        let models = get_available_embedded_models();
        let js_array = Array::new();
        
        for model in models {
            js_array.push(&JsValue::from_str(&model));
        }
        
        js_array
    }

    /// Remove background from ImageData (from Canvas)
    /// Returns a Promise that resolves to the processed ImageData
    #[wasm_bindgen]
    pub fn remove_background_from_image_data(&self, image_data: ImageData) -> Promise {
        if !self.initialized {
            return Promise::reject(&JsValue::from(WasmError::new("BackgroundRemover not initialized")));
        }

        console_log!("🖼️ Processing ImageData: {}x{} using unified processor", image_data.width(), image_data.height());

        // Clone processor reference for async operation
        let processor_ptr = self.processor.as_ref().unwrap() as *const BackgroundRemovalProcessor as *mut BackgroundRemovalProcessor;
        
        future_to_promise(async move {
            match Self::process_image_data_impl(image_data, processor_ptr).await {
                Ok(result_data) => Ok(result_data.into()),
                Err(e) => {
                    console_log!("❌ Failed to process image: {}", e);
                    Err(JsValue::from(WasmError::from(e)))
                }
            }
        })
    }

    /// Internal implementation for processing ImageData using unified processor
    #[allow(unsafe_code)]
    async fn process_image_data_impl(
        image_data: ImageData,
        processor_ptr: *mut BackgroundRemovalProcessor,
    ) -> Result<ImageData, bg_remove_core::error::BgRemovalError> {
        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data();

        console_log!("📊 Image dimensions: {}x{}, data length: {}", width, height, data.len());

        // Convert ImageData to temporary image file (WASM workaround)
        let temp_data = Self::image_data_to_temp_data(&data, width, height)?;
        
        // Create a temporary "file" in memory for processing
        let temp_path = std::path::PathBuf::from("temp_image_data.png");
        
        // SAFETY: We know this pointer is valid for the duration of the Promise
        let processor = unsafe { &mut *processor_ptr };
        
        // Process using the unified processor's process_image method
        console_log!("🔧 Processing with unified processor...");
        let result = processor.process_image(temp_data).await?;
        
        // Convert result back to ImageData
        let result_data = Self::result_to_image_data(result, width, height)?;
        
        console_log!("✅ Processing completed with unified processor");
        Ok(result_data)
    }

    /// Convert ImageData to DynamicImage and then use core preprocessing
    fn image_data_to_dynamic_image(
        data: &wasm_bindgen::Clamped<Vec<u8>>,
        width: u32,
        height: u32,
    ) -> Result<image::DynamicImage, bg_remove_core::error::BgRemovalError> {
        let data_vec: &Vec<u8> = &data;
        
        if data_vec.len() != (width * height * 4) as usize {
            return Err(bg_remove_core::error::BgRemovalError::processing(
                format!("Invalid ImageData size: expected {}, got {}", 
                        width * height * 4, data_vec.len())
            ));
        }

        // Convert RGBA ImageData to RGB image
        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
        for chunk in data_vec.chunks_exact(4) {
            rgb_data.push(chunk[0]); // R
            rgb_data.push(chunk[1]); // G  
            rgb_data.push(chunk[2]); // B
            // Ignore alpha channel (chunk[3])
        }

        // Create RGB ImageBuffer
        let rgb_image = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(width, height, rgb_data)
            .ok_or_else(|| bg_remove_core::error::BgRemovalError::processing(
                "Failed to create RGB image buffer"
            ))?;
        
        console_log!("📐 Converted ImageData to DynamicImage: {}x{}", width, height);
        Ok(image::DynamicImage::ImageRgb8(rgb_image))
    }

    /// Process image using core ImageProcessor (reuses existing preprocessing logic)
    fn process_with_core_processor(
        image: image::DynamicImage,
        backend: &mut TractBackend,
        _config: &RemovalConfig,
    ) -> Result<(Array4<f32>, image::DynamicImage), bg_remove_core::error::BgRemovalError> {
        // Create ImageProcessor with our backend - but we need to avoid double initialization
        // Since we can't easily extract the preprocessing method, let's use a simpler approach
        // and call the backend directly with proper preprocessing
        
        // Get preprocessing config from backend
        let preprocessing_config = backend.get_preprocessing_config()?;
        let target_size = preprocessing_config.target_size[0];
        
        console_log!("🔧 Model preprocessing config: target_size={}, mean={:?}, std={:?}", 
                    target_size, preprocessing_config.normalization_mean, preprocessing_config.normalization_std);

        // Apply the same preprocessing logic as ImageProcessor::preprocess_image
        let rgb_image = image.to_rgb8();
        let (orig_width, orig_height) = rgb_image.dimensions();

        // Calculate aspect ratio preserving dimensions
        let target_size_f32 = target_size as f32;
        let orig_width_f32 = orig_width as f32;
        let orig_height_f32 = orig_height as f32;

        let scale = target_size_f32
            .min((target_size_f32 / orig_width_f32).min(target_size_f32 / orig_height_f32));

        let new_width = (orig_width_f32 * scale).round() as u32;
        let new_height = (orig_height_f32 * scale).round() as u32;
        
        console_log!("📏 Scale factor: {:.3}, resized: {}x{}", scale, new_width, new_height);

        // Resize image maintaining aspect ratio
        let resized = image::imageops::resize(
            &rgb_image,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle,
        );

        // Create padded canvas with white padding (matching core default)
        let mut canvas = image::ImageBuffer::from_pixel(
            target_size,
            target_size,
            image::Rgb([255, 255, 255]), // White padding
        );

        // Calculate centering offset
        let offset_x = (target_size - new_width) / 2;
        let offset_y = (target_size - new_height) / 2;
        
        console_log!("📍 Padding offsets: x={}, y={}", offset_x, offset_y);

        // Copy resized image to center of canvas
        for (x, y, pixel) in resized.enumerate_pixels() {
            let canvas_x = x + offset_x;
            let canvas_y = y + offset_y;
            if canvas_x < target_size && canvas_y < target_size {
                canvas.put_pixel(canvas_x, canvas_y, *pixel);
            }
        }

        // Convert to tensor format (NCHW) with model-specific normalization
        let target_size_usize = target_size as usize;
        let mut tensor = Array4::<f32>::zeros((1, 3, target_size_usize, target_size_usize));

        for (y, row) in canvas.rows().enumerate() {
            for (x, pixel) in row.enumerate() {
                // Apply model-specific normalization (mean and std from preprocessing config)
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

        let preprocessed_image = image::DynamicImage::ImageRgb8(canvas);
        console_log!("✅ Preprocessed to tensor: 1x3x{}x{}", target_size, target_size);
        Ok((tensor, preprocessed_image))
    }

    /// Convert Array4<f32> back to ImageData
    fn array4_to_image_data(
        array: Array4<f32>,
        width: u32,
        height: u32,
    ) -> Result<ImageData, bg_remove_core::error::BgRemovalError> {
        let shape = array.shape();
        
        if shape[0] != 1 || shape[2] != height as usize || shape[3] != width as usize {
            return Err(bg_remove_core::error::BgRemovalError::processing(
                format!("Invalid output array shape: {:?}", shape)
            ));
        }

        let channels = shape[1];
        let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);

        // Convert from NCHW to RGBA
        for y in 0..height {
            for x in 0..width {
                let y_idx = y as usize;
                let x_idx = x as usize;

                if channels == 1 {
                    // Grayscale mask - convert to RGBA
                    let mask_value = array[[0, 0, y_idx, x_idx]];
                    let alpha = (mask_value.clamp(0.0, 1.0) * 255.0) as u8;
                    
                    rgba_data.push(0);     // R
                    rgba_data.push(0);     // G
                    rgba_data.push(0);     // B
                    rgba_data.push(alpha); // A
                } else if channels >= 3 {
                    // RGB output
                    let r = (array[[0, 0, y_idx, x_idx]].clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (array[[0, 1, y_idx, x_idx]].clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (array[[0, 2, y_idx, x_idx]].clamp(0.0, 1.0) * 255.0) as u8;
                    
                    rgba_data.push(r);     // R
                    rgba_data.push(g);     // G
                    rgba_data.push(b);     // B
                    rgba_data.push(255);   // A (fully opaque)
                } else {
                    return Err(bg_remove_core::error::BgRemovalError::processing(
                        format!("Unsupported number of channels: {}", channels)
                    ));
                }
            }
        }

        // Create Uint8ClampedArray from the data
        let clamped_array = wasm_bindgen::Clamped(rgba_data);
        
        // Create and return ImageData
        ImageData::new_with_u8_clamped_array(wasm_bindgen::Clamped(&clamped_array), width)
            .map_err(|e| bg_remove_core::error::BgRemovalError::processing(
                format!("Failed to create ImageData: {:?}", e)
            ))
    }
}

impl Default for BackgroundRemover {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Initialize the WASM module
/// This should be called once when the module is loaded
#[wasm_bindgen(start)]
pub fn init() {
    console_log!("🦀 bg-remove-web WASM module initialized");
    
    // Set up better panic messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get version information about the WASM module
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get information about available execution providers in WASM
#[wasm_bindgen]
pub fn get_wasm_providers() -> js_sys::Object {
    let info = js_sys::Object::new();
    
    // WASM only supports CPU execution with Tract backend
    js_sys::Reflect::set(&info, &"tract_cpu".into(), &true.into()).unwrap();
    js_sys::Reflect::set(&info, &"description".into(), &"Pure Rust CPU inference via Tract (WASM-compatible)".into()).unwrap();
    js_sys::Reflect::set(&info, &"wasm_compatible".into(), &true.into()).unwrap();
    
    info
}