//! High-level API module providing CLI-equivalent functionality
//!
//! This module contains the primary public API for the bg-remove library,
//! designed to mirror CLI usage patterns while offering separate initialization
//! and inference phases.

pub mod background_remover;
pub mod batch;
pub mod config;
pub mod models;
pub mod parallel;
pub mod progress;

// Re-export the main API types
pub use background_remover::BackgroundRemover;
pub use batch::{BatchOptions, BatchResult, FailedFile, ProcessedFile};
pub use config::{LibraryConfig, LibraryConfigBuilder, VerboseLevel};
pub use models::ModelManager;
pub use parallel::ParallelBatchProcessor;
pub use progress::{ConsoleProgressReporter, JsonProgressReporter, NoOpProgressReporter, ProgressReporter};