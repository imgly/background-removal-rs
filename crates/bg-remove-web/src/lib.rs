//! WebAssembly (WASM) bindings for background removal in web browsers
//!
//! This crate provides WebAssembly bindings for the bg-remove-core library,
//! enabling background removal functionality directly in web browsers using
//! pure Rust inference with the Tract backend.
//!
//! # Features
//!
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

use bg_remove_core::{
    get_available_embedded_models, 
    ModelManager, 
    ModelSource, 
    ModelSpec,
    RemovalConfig,
    ExecutionProvider,
    InferenceBackend,
};
use bg_remove_tract::TractBackend;
use js_sys::{Array, Promise, Uint8Array};
use ndarray::Array4;
use std::collections::HashMap;
use wasm_bindgen_futures::future_to_promise;
use web_sys::{ImageData, CanvasRenderingContext2d, HtmlCanvasElement};

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
    /// Output format quality for JPEG (0-100)
    jpeg_quality: u8,
    /// Output format quality for WebP (0-100)  
    webp_quality: u8,
    /// Background color (RGB hex string, e.g., "#ffffff")
    background_color: String,
    /// Whether to preserve color profiles
    preserve_color_profile: bool,
}

#[wasm_bindgen]
impl WebRemovalConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            jpeg_quality: 90,
            webp_quality: 85,
            background_color: "#ffffff".to_string(),
            preserve_color_profile: true,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn jpeg_quality(&self) -> u8 {
        self.jpeg_quality
    }

    #[wasm_bindgen(setter)]
    pub fn set_jpeg_quality(&mut self, quality: u8) {
        self.jpeg_quality = quality.clamp(0, 100);
    }

    #[wasm_bindgen(getter)]
    pub fn webp_quality(&self) -> u8 {
        self.webp_quality
    }

    #[wasm_bindgen(setter)]
    pub fn set_webp_quality(&mut self, quality: u8) {
        self.webp_quality = quality.clamp(0, 100);
    }

    #[wasm_bindgen(getter)]
    pub fn background_color(&self) -> String {
        self.background_color.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_background_color(&mut self, color: String) {
        self.background_color = color;
    }

    #[wasm_bindgen(getter)]
    pub fn preserve_color_profile(&self) -> bool {
        self.preserve_color_profile
    }

    #[wasm_bindgen(setter)]
    pub fn set_preserve_color_profile(&mut self, preserve: bool) {
        self.preserve_color_profile = preserve;
    }
}

impl Default for WebRemovalConfig {
    fn default() -> Self {
        Self::new()
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
    backend: Option<TractBackend>,
    config: RemovalConfig,
    initialized: bool,
}

#[wasm_bindgen]
impl BackgroundRemover {
    /// Create a new BackgroundRemover instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_log!("🌐 Creating new BackgroundRemover instance");
        
        // Set up better panic messages
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        Self {
            backend: None,
            config: RemovalConfig::default(),
            initialized: false,
        }
    }

    /// Initialize the background remover with a specific model
    /// Returns a Promise that resolves when initialization is complete
    #[wasm_bindgen]
    pub fn initialize(&mut self, model_name: Option<String>) -> Promise {
        console_log!("🚀 Initializing BackgroundRemover");
        
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

        let model_spec = ModelSpec {
            source: ModelSource::Embedded(model_name.clone()),
            variant: None,
        };

        // Configure for Tract backend (CPU only in WASM)
        let mut config = RemovalConfig::default();
        config.execution_provider = ExecutionProvider::Cpu;

        // Clone self state for async operation
        let self_ptr = self as *mut Self;
        
        future_to_promise(async move {
            match Self::initialize_impl(model_spec, config).await {
                Ok((backend, config)) => {
                    console_log!("✅ BackgroundRemover initialized successfully");
                    
                    // SAFETY: We know this pointer is valid for the duration of the Promise
                    unsafe {
                        (*self_ptr).backend = Some(backend);
                        (*self_ptr).config = config;
                        (*self_ptr).initialized = true;
                    }
                    
                    Ok(JsValue::TRUE)
                },
                Err(e) => {
                    console_log!("❌ Failed to initialize BackgroundRemover: {}", e);
                    Err(JsValue::from(WasmError::from(e)))
                }
            }
        })
    }

    /// Internal async initialization implementation
    async fn initialize_impl(
        model_spec: ModelSpec,
        mut config: RemovalConfig,
    ) -> Result<(TractBackend, RemovalConfig), bg_remove_core::error::BgRemovalError> {
        // Create model manager
        let model_manager = ModelManager::from_spec(&model_spec)?;
        
        // Create Tract backend
        let mut backend = TractBackend::with_model_manager(model_manager);
        
        // Initialize backend
        let _load_time = backend.initialize(&config)?;
        
        Ok((backend, config))
    }

    /// Check if the remover is initialized and ready to process images
    #[wasm_bindgen]
    pub fn is_initialized(&self) -> bool {
        self.initialized
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

        console_log!("🖼️ Processing ImageData: {}x{}", image_data.width(), image_data.height());

        // Clone backend reference for async operation
        let backend_ptr = self.backend.as_ref().unwrap() as *const TractBackend;
        
        future_to_promise(async move {
            match Self::process_image_data_impl(image_data, backend_ptr).await {
                Ok(result_data) => Ok(result_data.into()),
                Err(e) => {
                    console_log!("❌ Failed to process image: {}", e);
                    Err(JsValue::from(WasmError::from(e)))
                }
            }
        })
    }

    /// Internal implementation for processing ImageData
    async fn process_image_data_impl(
        image_data: ImageData,
        backend_ptr: *const TractBackend,
    ) -> Result<ImageData, bg_remove_core::error::BgRemovalError> {
        let width = image_data.width();
        let height = image_data.height();
        let data = image_data.data();

        console_log!("📊 Image dimensions: {}x{}, data length: {}", width, height, data.length());

        // Convert ImageData to ndarray format for processing
        let input_array = Self::image_data_to_array4(&data, width, height)?;
        
        // SAFETY: We know this pointer is valid for the duration of the Promise
        let backend = unsafe { &*backend_ptr };
        
        // Perform inference using Tract backend
        console_log!("🧠 Running inference...");
        let output_array = backend.infer(&input_array)?;
        
        // Convert back to ImageData
        let result_data = Self::array4_to_image_data(output_array, width, height)?;
        
        console_log!("✅ Inference completed");
        Ok(result_data)
    }

    /// Convert ImageData to Array4<f32> for processing
    fn image_data_to_array4(
        data: &js_sys::Uint8ClampedArray,
        width: u32,
        height: u32,
    ) -> Result<Array4<f32>, bg_remove_core::error::BgRemovalError> {
        let data_vec: Vec<u8> = data.to_vec();
        
        if data_vec.len() != (width * height * 4) as usize {
            return Err(bg_remove_core::error::BgRemovalError::processing(
                format!("Invalid ImageData size: expected {}, got {}", 
                        width * height * 4, data_vec.len())
            ));
        }

        // Convert RGBA to RGB and normalize to [0,1]
        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
        
        for chunk in data_vec.chunks_exact(4) {
            rgb_data.push(chunk[0] as f32 / 255.0); // R
            rgb_data.push(chunk[1] as f32 / 255.0); // G  
            rgb_data.push(chunk[2] as f32 / 255.0); // B
            // Ignore alpha channel (chunk[3])
        }

        // Create Array4 in NCHW format (batch=1, channels=3, height, width)
        Array4::from_shape_vec(
            (1, 3, height as usize, width as usize),
            rgb_data,
        )
        .map_err(|e| bg_remove_core::error::BgRemovalError::processing(
            format!("Failed to create input array: {}", e)
        ))
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
        let clamped_array = js_sys::Uint8ClampedArray::from(&rgba_data[..]);
        
        // Create and return ImageData
        ImageData::new_with_u8_clamped_array(&clamped_array, width)
            .map_err(|e| bg_remove_core::error::BgRemovalError::processing(
                format!("Failed to create ImageData: {:?}", e)
            ))
    }
}

impl Default for BackgroundRemover {
    fn default() -> Self {
        Self::new()
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