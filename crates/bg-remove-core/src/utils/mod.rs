//! Utility modules for common operations
//!
//! This module consolidates utility functions that were previously
//! scattered across different crates, providing a single source of
//! truth for common operations.

pub mod color;
pub mod models;
pub mod providers;

// Re-export commonly used items for convenience
pub use color::ColorParser;
pub use models::ModelSpecParser;
pub use providers::{ExecutionProviderManager, ProviderInfo};