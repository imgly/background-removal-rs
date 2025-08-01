//! Service layer for bg-remove-core
//!
//! This module contains service classes that separate infrastructure concerns
//! from business logic, improving testability and maintainability.

pub mod format;
pub mod io;
pub mod progress;

pub use format::OutputFormatHandler;
pub use io::ImageIOService;
pub use progress::{
    create_cli_progress_reporter, BatchProcessingStats, BatchProgressUpdate,
    ConsoleProgressReporter, EnhancedProgressReporter, NoOpProgressReporter, ProcessingStage,
    ProgressReporter, ProgressTracker, ProgressUpdate,
};
