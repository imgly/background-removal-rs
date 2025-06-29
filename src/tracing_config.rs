//! Tracing configuration module for structured logging and observability
//!
//! This module provides centralized configuration for tracing subscribers,
//! following Rust tracing best practices where applications configure
//! subscribers while libraries only emit trace events.

#[cfg(feature = "cli")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

/// Configuration for tracing output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    /// Human-readable console output with colors and emojis (default for CLI)
    Console,
    /// Compact console output for CI environments
    Compact,
    /// JSON structured logging for production environments
    #[cfg(feature = "tracing-json")]
    Json,
}

/// Configuration for tracing output destination
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TracingOutput {
    /// Output to stdout/stderr (default)
    Console,
    /// Output to a file
    #[cfg(feature = "tracing-files")]
    File(std::path::PathBuf),
    /// Output to both console and file
    #[cfg(feature = "tracing-files")]
    Both(std::path::PathBuf),
}

/// Tracing configuration builder
#[derive(Debug)]
pub struct TracingConfig {
    /// Verbosity level (maps to log levels)
    pub verbosity: u8,
    /// Output format
    pub format: TracingFormat,
    /// Output destination
    pub output: TracingOutput,
    /// Environment filter string (overrides verbosity if set)
    pub env_filter: Option<String>,
    /// Session ID for correlation
    pub session_id: Option<String>,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            verbosity: 0,
            format: TracingFormat::Console,
            output: TracingOutput::Console,
            env_filter: None,
            session_id: None,
        }
    }
}

impl TracingConfig {
    /// Create a new tracing configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set verbosity level (0-3+)
    pub fn with_verbosity(mut self, verbosity: u8) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Set output format
    pub fn with_format(mut self, format: TracingFormat) -> Self {
        self.format = format;
        self
    }

    /// Set output destination
    pub fn with_output(mut self, output: TracingOutput) -> Self {
        self.output = output;
        self
    }

    /// Set custom environment filter
    pub fn with_env_filter<S: Into<String>>(mut self, filter: S) -> Self {
        self.env_filter = Some(filter.into());
        self
    }

    /// Set session ID for request correlation
    pub fn with_session_id<S: Into<String>>(mut self, session_id: S) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Convert verbosity level to tracing filter string
    pub fn verbosity_to_filter(&self) -> &'static str {
        match self.verbosity {
            0 => "info",  // Default: informational messages and above
            1 => "debug", // -v: internal state and computations
            2 => "trace", // -vv: extremely detailed traces
            _ => "trace", // -vvv+: extremely detailed traces (same as -vv)
        }
    }

    /// Initialize tracing subscriber based on configuration
    #[cfg(feature = "cli")]
    pub fn init(self) -> anyhow::Result<()> {
        use tracing_subscriber::fmt;

        // Determine the filter to use
        let filter = if let Some(env_filter) = &self.env_filter {
            EnvFilter::try_new(env_filter)?
        } else {
            EnvFilter::try_new(self.verbosity_to_filter())?
        };

        // Create base subscriber
        let registry = Registry::default().with(filter);

        match (&self.format, &self.output) {
            // Console output with pretty formatting
            (TracingFormat::Console, TracingOutput::Console) => {
                let fmt_layer = fmt::layer()
                    .with_ansi(true)
                    .with_target(false)
                    .with_thread_ids(false)
                    .with_thread_names(false)
                    .with_file(false)
                    .with_line_number(false)
                    .with_level(true)
                    .compact();

                registry.with(fmt_layer).init();
            },

            // Compact console output
            (TracingFormat::Compact, TracingOutput::Console) => {
                let fmt_layer = fmt::layer()
                    .with_ansi(false)
                    .with_target(false)
                    .with_thread_ids(false)
                    .with_thread_names(false)
                    .with_file(false)
                    .with_line_number(false)
                    .compact();

                registry.with(fmt_layer).init();
            },

            #[cfg(feature = "tracing-json")]
            // JSON output for structured logging
            (TracingFormat::Json, TracingOutput::Console) => {
                let fmt_layer = fmt::layer()
                    .json()
                    .with_current_span(true)
                    .with_span_list(true);

                registry.with(fmt_layer).init();
            },

            #[cfg(feature = "tracing-files")]
            // File output
            (format, TracingOutput::File(path)) => {
                use tracing_appender::{non_blocking, rolling};

                let file_appender = rolling::never(
                    path.parent().unwrap_or_else(|| std::path::Path::new(".")),
                    path.file_name()
                        .unwrap_or_else(|| std::ffi::OsStr::new("bgremove.log")),
                );
                let (file_writer, _guard) = non_blocking(file_appender);

                let fmt_layer = match format {
                    TracingFormat::Console | TracingFormat::Compact => fmt::layer()
                        .with_ansi(false)
                        .with_writer(file_writer)
                        .compact(),
                    #[cfg(feature = "tracing-json")]
                    TracingFormat::Json => fmt::layer()
                        .json()
                        .with_writer(file_writer)
                        .with_current_span(true)
                        .with_span_list(true),
                };

                registry.with(fmt_layer).init();
            },

            #[cfg(feature = "tracing-files")]
            // Both console and file output
            (format, TracingOutput::Both(path)) => {
                use tracing_appender::{non_blocking, rolling};

                // Console layer
                let console_layer = match format {
                    TracingFormat::Console => {
                        fmt::layer().with_ansi(true).with_target(false).compact()
                    },
                    TracingFormat::Compact => {
                        fmt::layer().with_ansi(false).with_target(false).compact()
                    },
                    #[cfg(feature = "tracing-json")]
                    TracingFormat::Json => fmt::layer()
                        .json()
                        .with_current_span(true)
                        .with_span_list(true),
                };

                // File layer
                let file_appender = rolling::daily(
                    path.parent().unwrap_or_else(|| std::path::Path::new(".")),
                    path.file_stem()
                        .unwrap_or_else(|| std::ffi::OsStr::new("bgremove")),
                );
                let (file_writer, _guard) = non_blocking(file_appender);

                #[cfg(feature = "tracing-json")]
                let file_layer = fmt::layer()
                    .json()
                    .with_writer(file_writer)
                    .with_current_span(true)
                    .with_span_list(true);

                #[cfg(not(feature = "tracing-json"))]
                let file_layer = fmt::layer()
                    .with_ansi(false)
                    .with_writer(file_writer)
                    .compact();

                registry.with(console_layer).with(file_layer).init();
            },
        }

        // Set session ID as a global field if provided
        if let Some(session_id) = &self.session_id {
            tracing::info!(
                session_id = %session_id,
                "🚀 Background removal session started"
            );
        }

        Ok(())
    }
}

/// Convenience function to initialize tracing with CLI-friendly defaults
#[cfg(feature = "cli")]
pub fn init_cli_tracing(
    verbosity: u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let session_id = uuid::Uuid::new_v4().to_string();

    TracingConfig::new()
        .with_verbosity(verbosity)
        .with_format(TracingFormat::Console)
        .with_session_id(session_id)
        .init()
        .map_err(|e| {
            let boxed: Box<dyn std::error::Error + Send + Sync + 'static> = e.into();
            boxed
        })
}

/// Initialize tracing for library usage (minimal configuration)
pub fn init_library_tracing() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    // For library usage, only set up if no global subscriber is already set
    if tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )
    .is_ok()
    {
        tracing::debug!("📚 Library tracing initialized");
    }
    Ok(())
}

/// Span creation helpers for common operations
pub mod spans {
    use tracing::{Level, Span};

    /// Create a session span for the entire CLI operation
    pub fn session(session_id: &str, model_name: &str, provider: &str) -> Span {
        tracing::span!(
            Level::INFO,
            "session",
            session_id = %session_id,
            model_name = %model_name,
            provider = %provider
        )
    }

    /// Create a span for model loading operations
    pub fn model_loading(model_name: &str, provider: &str) -> Span {
        tracing::span!(
            Level::INFO,
            "model_loading",
            model_name = %model_name,
            provider = %provider
        )
    }

    /// Create a span for file processing operations
    pub fn file_processing(file_path: &std::path::Path, format: &str) -> Span {
        tracing::span!(
            Level::INFO,
            "file_processing",
            file_path = %file_path.display(),
            format = %format
        )
    }

    /// Create a span for batch processing operations
    pub fn batch_processing(file_count: usize) -> Span {
        tracing::span!(
            Level::INFO,
            "batch_processing",
            file_count = %file_count
        )
    }

    /// Create a span for inference operations
    pub fn inference(model_name: &str, dimensions: (u32, u32)) -> Span {
        tracing::span!(
            Level::DEBUG,
            "inference",
            model_name = %model_name,
            width = %dimensions.0,
            height = %dimensions.1
        )
    }

    /// Create a span for cache operations
    pub fn cache_operation(operation: &str, cache_key: &str) -> Span {
        tracing::span!(
            Level::DEBUG,
            "cache_operation",
            operation = %operation,
            cache_key = %cache_key
        )
    }

    /// Create a span for download operations
    pub fn download(url: &str, destination: &std::path::Path) -> Span {
        tracing::span!(
            Level::INFO,
            "download",
            url = %url,
            destination = %destination.display()
        )
    }

    /// Create a span for preprocessing operations
    pub fn preprocessing(original_size: (u32, u32), target_size: (u32, u32)) -> Span {
        tracing::span!(
            Level::DEBUG,
            "preprocessing",
            original_width = %original_size.0,
            original_height = %original_size.1,
            target_width = %target_size.0,
            target_height = %target_size.1
        )
    }

    /// Create a span for postprocessing operations
    pub fn postprocessing(operation: &str) -> Span {
        tracing::span!(
            Level::DEBUG,
            "postprocessing",
            operation = %operation
        )
    }
}

/// Event helpers for common logging patterns
pub mod events {
    use tracing::{debug, error, info, warn};

    /// Log a user-facing progress update
    pub fn progress(message: &str, emoji: &str) {
        info!("{} {}", emoji, message);
    }

    /// Log an error with context
    pub fn error_with_context(error: &dyn std::error::Error, context: &str) {
        error!(
            error = %error,
            context = %context,
            "❌ Operation failed"
        );
    }

    /// Log a warning with recommendation
    pub fn warning_with_recommendation(message: &str, recommendation: &str) {
        warn!(
            message = %message,
            recommendation = %recommendation,
            "⚠️  Warning"
        );
    }

    /// Log performance metrics
    pub fn performance_metric(
        operation: &str,
        duration_ms: u64,
        additional_fields: Option<&[(&str, &dyn std::fmt::Display)]>,
    ) {
        debug!(
            operation = %operation,
            duration_ms = %duration_ms,
            "⏱️  Performance metric"
        );

        // Add additional fields if provided
        if let Some(fields) = additional_fields {
            for (key, value) in fields {
                debug!(
                    operation = %operation,
                    duration_ms = %duration_ms,
                    key = %key,
                    value = %value,
                    "⏱️  Performance metric with details"
                );
                break; // Only log once with all fields - this is a simplified version
            }
        }
    }

    /// Log cache operations
    pub fn cache_hit(cache_key: &str, operation: &str) {
        debug!(
            cache_key = %cache_key,
            operation = %operation,
            "💾 Cache hit"
        );
    }

    pub fn cache_miss(cache_key: &str, operation: &str) {
        debug!(
            cache_key = %cache_key,
            operation = %operation,
            "🔍 Cache miss"
        );
    }

    /// Log download progress
    pub fn download_progress(url: &str, bytes_downloaded: u64, total_bytes: Option<u64>) {
        match total_bytes {
            Some(total) => debug!(
                url = %url,
                bytes_downloaded = %bytes_downloaded,
                total_bytes = %total,
                progress_percent = %(bytes_downloaded as f64 / total as f64 * 100.0),
                "📥 Download progress"
            ),
            None => debug!(
                url = %url,
                bytes_downloaded = %bytes_downloaded,
                "📥 Download progress"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbosity_mapping() {
        assert_eq!(TracingConfig::new().with_verbosity(0).verbosity_to_filter(), "info");
        assert_eq!(TracingConfig::new().with_verbosity(1).verbosity_to_filter(), "debug");
        assert_eq!(TracingConfig::new().with_verbosity(2).verbosity_to_filter(), "trace");
        assert_eq!(TracingConfig::new().with_verbosity(3).verbosity_to_filter(), "trace");
        assert_eq!(TracingConfig::new().with_verbosity(10).verbosity_to_filter(), "trace");
    }

    #[test]
    fn test_config_builder() {
        let config = TracingConfig::new()
            .with_verbosity(2)
            .with_format(TracingFormat::Compact)
            .with_session_id("test-session");

        assert_eq!(config.verbosity, 2);
        assert_eq!(config.format, TracingFormat::Compact);
        assert_eq!(config.session_id.as_ref().unwrap(), "test-session");
    }

    #[test]
    fn test_default_config() {
        let config = TracingConfig::default();
        assert_eq!(config.verbosity, 0);
        assert_eq!(config.format, TracingFormat::Console);
        assert_eq!(config.output, TracingOutput::Console);
        assert!(config.env_filter.is_none());
        assert!(config.session_id.is_none());
    }
}
