//! WebAssembly (WASM) bindings for background removal in web browsers
//!
//! This crate provides WebAssembly bindings for the bg-remove-core library,
//! enabling background removal functionality directly in web browsers using
//! the unified processor with Tract backend.

mod backend_factory;
mod config;

use wasm_bindgen::prelude::*;
use backend_factory::WebBackendFactory;
use config::WebConfigBuilder;
use bg_remove_core::{
    processor::BackgroundRemovalProcessor,
    models::get_available_embedded_models,
};
use js_sys::{Array, Promise, Uint8Array};
use wasm_bindgen_futures::future_to_promise;

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
    /// Enable debug mode (additional logging)
    debug: bool,
    /// Number of intra-op threads for inference (0 = auto)
    intra_threads: u32,
    /// Number of inter-op threads for inference (0 = auto)
    inter_threads: u32,
    /// Whether to preserve color profiles
    preserve_color_profile: bool,
}

#[wasm_bindgen]
impl WebRemovalConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    // Output format getters/setters
    #[wasm_bindgen(getter)]
    pub fn output_format(&self) -> String {
        self.output_format.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_output_format(&mut self, format: String) {
        let valid_formats = ["png", "jpeg", "jpg", "webp", "tiff", "rgba8"];
        if valid_formats.contains(&format.to_lowercase().as_str()) {
            self.output_format = format.to_lowercase();
        } else {
            console_log!("Invalid output format: {}. Using 'png'", format);
            self.output_format = "png".to_string();
        }
    }


    // Additional getters/setters (simplified for brevity)
    #[wasm_bindgen(getter)]
    pub fn jpeg_quality(&self) -> u8 { self.jpeg_quality }
    
    #[wasm_bindgen(setter)]
    pub fn set_jpeg_quality(&mut self, quality: u8) { self.jpeg_quality = quality.clamp(0, 100); }

    #[wasm_bindgen(getter)]
    pub fn debug(&self) -> bool { self.debug }
    
    #[wasm_bindgen(setter)]
    pub fn set_debug(&mut self, debug: bool) { self.debug = debug; }

    /// Validate configuration using core utilities
    pub fn validate(&self) -> Result<(), WasmError> {
        WebConfigBuilder::validate_web_config(self)
            .map_err(WasmError::from)
    }

    /// Convert to unified ProcessorConfig
    pub(crate) fn to_processor_config(&self, model_name: Option<String>) -> Result<bg_remove_core::processor::ProcessorConfig, bg_remove_core::error::BgRemovalError> {
        WebConfigBuilder::from_web_config(self, model_name)
    }

    /// Create WebRemovalConfig from unified ProcessorConfig
    pub(crate) fn from_processor_config(config: &bg_remove_core::processor::ProcessorConfig) -> Self {
        WebConfigBuilder::to_web_config(config)
    }
}

impl Default for WebRemovalConfig {
    fn default() -> Self {
        Self {
            output_format: "png".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            debug: false,
            intra_threads: 0,
            inter_threads: 0,
            preserve_color_profile: true,
        }
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
    #[wasm_bindgen]
    pub fn initialize(&mut self, model_name: Option<String>) -> Promise {
        console_log!("🚀 Initializing BackgroundRemover with unified processor");
        
        let model_name = model_name.unwrap_or_else(|| {
            let available = get_available_embedded_models();
            if available.is_empty() {
                "isnet-fp16".to_string()
            } else {
                available[0].clone()
            }
        });

        console_log!("📦 Using model: {}", model_name);

        let config_result = self.config.to_processor_config(Some(model_name));
        let self_ptr = self as *mut Self;
        
        future_to_promise(async move {
            match config_result {
                Ok(config) => {
                    match Self::initialize_impl(config).await {
                        Ok((processor, final_config)) => {
                            console_log!("✅ BackgroundRemover initialized successfully");
                            
                            #[allow(unsafe_code)]
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
        let backend_factory = Box::new(WebBackendFactory::new());
        let processor = BackgroundRemovalProcessor::with_factory(config.clone(), backend_factory)?;
        
        console_log!("✅ Unified processor created successfully");
        Ok((processor, config))
    }

    /// Check if initialized
    #[wasm_bindgen]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get current configuration
    #[wasm_bindgen(getter)]
    pub fn config(&self) -> WebRemovalConfig {
        self.config.clone()
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

    /// Remove background from byte array (PNG/JPEG data)
    #[wasm_bindgen]
    pub fn remove_background_from_bytes(&self, image_bytes: Uint8Array) -> Promise {
        if !self.initialized {
            return Promise::reject(&JsValue::from(WasmError::new("BackgroundRemover not initialized")));
        }

        let bytes: Vec<u8> = image_bytes.to_vec();
        console_log!("🖼️ Processing image bytes: {} bytes", bytes.len());

        let processor_ptr = self.processor.as_ref().unwrap() as *const BackgroundRemovalProcessor as *mut BackgroundRemovalProcessor;
        
        future_to_promise(async move {
            match Self::process_bytes_impl(bytes, processor_ptr).await {
                Ok(result_bytes) => {
                    let result_array = Uint8Array::new_with_length(result_bytes.len() as u32);
                    result_array.copy_from(&result_bytes);
                    Ok(result_array.into())
                },
                Err(e) => {
                    console_log!("❌ Failed to process image: {}", e);
                    Err(JsValue::from(WasmError::from(e)))
                }
            }
        })
    }

    /// Internal implementation for processing bytes using unified processor
    #[allow(unsafe_code)]
    async fn process_bytes_impl(
        image_bytes: Vec<u8>,
        processor_ptr: *mut BackgroundRemovalProcessor,
    ) -> Result<Vec<u8>, bg_remove_core::error::BgRemovalError> {
        console_log!("🔧 Processing with unified processor...");
        
        let processor = unsafe { &mut *processor_ptr };
        
        // Decode image bytes to DynamicImage first
        let dynamic_image = image::load_from_memory(&image_bytes)
            .map_err(|e| bg_remove_core::error::BgRemovalError::processing(
                format!("Failed to decode image: {}", e)
            ))?;
        
        // Process using the unified processor
        let result = processor.process_image(dynamic_image)?;
        
        // Get result as bytes (PNG format)
        let result_bytes = result.to_bytes(bg_remove_core::config::OutputFormat::Png, 90)?;
        
        console_log!("✅ Processing completed, result: {} bytes", result_bytes.len());
        Ok(result_bytes)
    }
}

impl Default for BackgroundRemover {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    console_log!("🦀 bg-remove-web WASM module initialized");
    
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get version information
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get information about available execution providers in WASM
#[wasm_bindgen]
pub fn get_wasm_providers() -> js_sys::Object {
    let info = js_sys::Object::new();
    
    js_sys::Reflect::set(&info, &"tract_cpu".into(), &true.into()).unwrap();
    js_sys::Reflect::set(&info, &"description".into(), &"Pure Rust CPU inference via Tract (WASM-compatible)".into()).unwrap();
    js_sys::Reflect::set(&info, &"wasm_compatible".into(), &true.into()).unwrap();
    
    info
}