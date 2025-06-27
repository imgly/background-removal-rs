//! CLI module for the imgly-bgremove library
//!
//! This module is only available when the "cli" feature is enabled.

mod backend_factory;
mod config;
#[path = "main.rs"]
mod main_impl;

pub use main_impl::{main, Cli, CliOutputFormat};
