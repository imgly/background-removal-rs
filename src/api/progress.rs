//! Progress reporting for library operations

use std::path::Path;
use std::time::Duration;

/// Progress reporting trait for batch operations
pub trait ProgressReporter: Send + Sync {
    /// Called when batch processing starts
    fn on_start(&self, total_files: usize);
    
    /// Called when processing of a file starts
    fn on_file_start(&self, file: &Path, index: usize);
    
    /// Called when processing of a file completes successfully
    fn on_file_complete(&self, file: &Path, processing_time: Duration);
    
    /// Called when processing of a file fails
    fn on_file_error(&self, file: &Path, error: &str);
    
    /// Called when all batch processing completes
    fn on_batch_complete(&self, total_files: usize, successful: usize, failed: usize, total_time: Duration);
}

/// Console-based progress reporter
pub struct ConsoleProgressReporter {
    verbose: bool,
}

impl ConsoleProgressReporter {
    /// Create a new console progress reporter
    pub fn new() -> Self {
        Self { verbose: true }
    }

    /// Create a quiet console progress reporter
    pub fn quiet() -> Self {
        Self { verbose: false }
    }
}

impl Default for ConsoleProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for ConsoleProgressReporter {
    fn on_start(&self, total_files: usize) {
        if self.verbose {
            println!("Starting batch processing of {} files...", total_files);
        }
    }

    fn on_file_start(&self, file: &Path, index: usize) {
        if self.verbose {
            println!("[{}] Processing: {}", index + 1, file.display());
        }
    }

    fn on_file_complete(&self, file: &Path, processing_time: Duration) {
        if self.verbose {
            println!("✅ Completed: {} ({}ms)", file.display(), processing_time.as_millis());
        }
    }

    fn on_file_error(&self, file: &Path, error: &str) {
        eprintln!("❌ Failed: {} - {}", file.display(), error);
    }

    fn on_batch_complete(&self, total_files: usize, successful: usize, failed: usize, total_time: Duration) {
        println!("Batch complete: {}/{} successful in {:.2}s", 
                 successful, total_files, total_time.as_secs_f64());
        if failed > 0 {
            println!("  {} files failed processing", failed);
        }
    }
}

/// JSON-based progress reporter for programmatic use
pub struct JsonProgressReporter;

impl JsonProgressReporter {
    /// Create a new JSON progress reporter
    pub fn new() -> Self {
        Self
    }
}

impl Default for JsonProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for JsonProgressReporter {
    fn on_start(&self, total_files: usize) {
        println!(r#"{{"event":"start","total_files":{}}}"#, total_files);
    }

    fn on_file_start(&self, file: &Path, index: usize) {
        println!(r#"{{"event":"file_start","file":"{}","index":{}}}"#, 
                 file.display(), index);
    }

    fn on_file_complete(&self, file: &Path, processing_time: Duration) {
        println!(r#"{{"event":"file_complete","file":"{}","processing_time_ms":{}}}"#, 
                 file.display(), processing_time.as_millis());
    }

    fn on_file_error(&self, file: &Path, error: &str) {
        println!(r#"{{"event":"file_error","file":"{}","error":"{}"}}"#, 
                 file.display(), error);
    }

    fn on_batch_complete(&self, total_files: usize, successful: usize, failed: usize, total_time: Duration) {
        println!(r#"{{"event":"batch_complete","total_files":{},"successful":{},"failed":{},"total_time_ms":{}}}"#, 
                 total_files, successful, failed, total_time.as_millis());
    }
}

/// No-operation progress reporter that does nothing
pub struct NoOpProgressReporter;

impl NoOpProgressReporter {
    /// Create a new no-op progress reporter
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for NoOpProgressReporter {
    fn on_start(&self, _total_files: usize) {}
    fn on_file_start(&self, _file: &Path, _index: usize) {}
    fn on_file_complete(&self, _file: &Path, _processing_time: Duration) {}
    fn on_file_error(&self, _file: &Path, _error: &str) {}
    fn on_batch_complete(&self, _total_files: usize, _successful: usize, _failed: usize, _total_time: Duration) {}
}