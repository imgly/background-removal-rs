//! Utility modules for common operations
//!
//! This module consolidates utility functions that were previously
//! scattered across different crates, providing a single source of
//! truth for common operations.

pub mod color;
pub mod models;
pub mod preprocessing;
pub mod providers;
pub mod validation;

// Re-export commonly used items for convenience
pub use models::ModelSpecParser;
pub use preprocessing::{ImagePreprocessor, PreprocessingOptions};
pub use providers::{ExecutionProviderManager, ProviderInfo};
pub use validation::{ConfigValidator, ModelValidator, NumericValidator, PathValidator, TensorValidator};
