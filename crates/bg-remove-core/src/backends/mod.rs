//! Backend implementations for inference

pub mod mock;
pub mod onnx;

// Re-exports for easy access
pub use self::mock::MockBackend;
pub use self::onnx::OnnxBackend;
