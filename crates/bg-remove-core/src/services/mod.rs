//! Service layer for bg-remove-core
//!
//! This module contains service classes that separate infrastructure concerns
//! from business logic, improving testability and maintainability.

pub mod format;
pub mod io;

pub use format::OutputFormatHandler;
pub use io::ImageIOService;
