//! Backend implementations for different inference engines
//!
//! This module provides different inference backends for the background removal library:
//! - ONNX Runtime backend (high performance, GPU acceleration)
//! - Tract backend (pure Rust, WebAssembly compatible)

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "tract")]
pub mod tract;

// Re-export backends based on enabled features
#[cfg(feature = "onnx")]
pub use self::onnx::OnnxBackend;

#[cfg(feature = "tract")]
pub use self::tract::TractBackend;
