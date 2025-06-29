//! Progress reporting service
//!
//! This module separates progress reporting concerns from business logic,
//! allowing different frontends to implement their own progress handling.

use crate::types::ProcessingTimings;
use instant::Instant;

/// Progress stages during background removal processing
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingStage {
    /// Initializing backend and loading model
    Initialization,
    /// Loading and decoding input image
    ImageLoading,
    /// Extracting color profile from input
    ColorProfileExtraction,
    /// Preprocessing image for inference
    Preprocessing,
    /// Running model inference
    Inference,
    /// Converting output tensor to mask
    MaskGeneration,
    /// Applying mask to create result image
    BackgroundRemoval,
    /// Converting to output format
    FormatConversion,
    /// Saving result to file
    FileSaving,
    /// Processing completed
    Completed,
}

impl ProcessingStage {
    /// Get a human-readable description of the processing stage
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            ProcessingStage::Initialization => "Initializing model and backend",
            ProcessingStage::ImageLoading => "Loading input image",
            ProcessingStage::ColorProfileExtraction => "Extracting color profile",
            ProcessingStage::Preprocessing => "Preprocessing image",
            ProcessingStage::Inference => "Running AI inference",
            ProcessingStage::MaskGeneration => "Generating segmentation mask",
            ProcessingStage::BackgroundRemoval => "Removing background",
            ProcessingStage::FormatConversion => "Converting output format",
            ProcessingStage::FileSaving => "Saving result",
            ProcessingStage::Completed => "Processing completed",
        }
    }

    /// Get the typical progress percentage for this stage
    #[must_use]
    pub fn progress_percentage(&self) -> u8 {
        match self {
            ProcessingStage::Initialization => 5,
            ProcessingStage::ImageLoading => 10,
            ProcessingStage::ColorProfileExtraction => 15,
            ProcessingStage::Preprocessing => 25,
            ProcessingStage::Inference => 70,
            ProcessingStage::MaskGeneration => 85,
            ProcessingStage::BackgroundRemoval => 95,
            ProcessingStage::FormatConversion => 98,
            ProcessingStage::FileSaving => 99,
            ProcessingStage::Completed => 100,
        }
    }
}

/// Progress update containing stage and timing information
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Current processing stage
    pub stage: ProcessingStage,
    /// Progress percentage (0-100)
    pub progress: u8,
    /// Human-readable stage description
    pub description: String,
    /// Elapsed time since processing started (milliseconds)
    pub elapsed_ms: u64,
    /// Estimated time remaining (milliseconds, if available)
    pub eta_ms: Option<u64>,
}

impl ProgressUpdate {
    /// Create a new progress update
    #[must_use]
    pub fn new(stage: ProcessingStage, start_time: Instant) -> Self {
        Self {
            progress: stage.progress_percentage(),
            description: stage.description().to_string(),
            elapsed_ms: start_time.elapsed().as_millis() as u64,
            eta_ms: None,
            stage,
        }
    }

    /// Create a progress update with custom description
    #[must_use]
    pub fn with_description(
        stage: ProcessingStage,
        description: String,
        start_time: Instant,
    ) -> Self {
        Self {
            progress: stage.progress_percentage(),
            elapsed_ms: start_time.elapsed().as_millis() as u64,
            eta_ms: None,
            stage,
            description,
        }
    }

    /// Add estimated time remaining
    #[must_use]
    pub fn with_eta(mut self, eta_ms: u64) -> Self {
        self.eta_ms = Some(eta_ms);
        self
    }
}

/// Trait for reporting progress during background removal operations
pub trait ProgressReporter: Send + Sync {
    /// Report a progress update
    ///
    /// # Arguments
    /// * `update` - Progress update containing stage and timing information
    fn report_progress(&self, update: ProgressUpdate);

    /// Report processing completion with final timings
    ///
    /// # Arguments
    /// * `timings` - Final processing timings
    fn report_completion(&self, timings: ProcessingTimings);

    /// Report an error during processing
    ///
    /// # Arguments
    /// * `stage` - Stage where error occurred
    /// * `error` - Error description
    fn report_error(&self, stage: ProcessingStage, error: &str);
}

/// No-op progress reporter that discards all progress updates
pub struct NoOpProgressReporter;

impl ProgressReporter for NoOpProgressReporter {
    fn report_progress(&self, _update: ProgressUpdate) {
        // Intentionally empty - discards progress updates
    }

    fn report_completion(&self, _timings: ProcessingTimings) {
        // Intentionally empty - discards completion notification
    }

    fn report_error(&self, _stage: ProcessingStage, _error: &str) {
        // Intentionally empty - discards error reports
    }
}

/// Console progress reporter that logs progress to stdout/stderr
pub struct ConsoleProgressReporter {
    verbose: bool,
}

impl ConsoleProgressReporter {
    /// Create a new console progress reporter
    ///
    /// # Arguments
    /// * `verbose` - Whether to show detailed progress information
    #[must_use]
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

impl ProgressReporter for ConsoleProgressReporter {
    fn report_progress(&self, update: ProgressUpdate) {
        if self.verbose {
            if let Some(eta) = update.eta_ms {
                log::info!(
                    "[{}%] {} ({}ms elapsed, ~{}ms remaining)",
                    update.progress,
                    update.description,
                    update.elapsed_ms,
                    eta
                );
            } else {
                log::info!(
                    "[{}%] {} ({}ms elapsed)",
                    update.progress,
                    update.description,
                    update.elapsed_ms
                );
            }
        } else {
            log::info!("[{}%] {}", update.progress, update.description);
        }
    }

    fn report_completion(&self, timings: ProcessingTimings) {
        log::info!("✅ Background removal completed in {}ms", timings.total_ms);

        if self.verbose {
            log::info!("  📊 Detailed timings:");
            log::info!("    • Image decode: {}ms", timings.image_decode_ms);
            log::info!("    • Preprocessing: {}ms", timings.preprocessing_ms);
            log::info!("    • Inference: {}ms", timings.inference_ms);
            log::info!("    • Postprocessing: {}ms", timings.postprocessing_ms);
        }
    }

    fn report_error(&self, stage: ProcessingStage, error: &str) {
        log::error!("❌ Error during {}: {}", stage.description(), error);
    }
}

/// Progress tracker that manages timing and progress reporting
pub struct ProgressTracker {
    reporter: Box<dyn ProgressReporter>,
    start_time: Instant,
    current_stage: Option<ProcessingStage>,
}

impl ProgressTracker {
    /// Create a new progress tracker with the specified reporter
    #[must_use]
    pub fn new(reporter: Box<dyn ProgressReporter>) -> Self {
        Self {
            reporter,
            start_time: Instant::now(),
            current_stage: None,
        }
    }

    /// Create a progress tracker with no-op reporter (for testing/disabled progress)
    #[must_use]
    pub fn no_op() -> Self {
        Self::new(Box::new(NoOpProgressReporter))
    }

    /// Create a progress tracker with console reporter
    #[must_use]
    pub fn console(verbose: bool) -> Self {
        Self::new(Box::new(ConsoleProgressReporter::new(verbose)))
    }

    /// Report progress for a specific stage
    pub fn report_stage(&mut self, stage: ProcessingStage) {
        self.current_stage = Some(stage.clone());
        let update = ProgressUpdate::new(stage, self.start_time);
        self.reporter.report_progress(update);
    }

    /// Report progress with custom description
    pub fn report_stage_with_description(&mut self, stage: ProcessingStage, description: String) {
        self.current_stage = Some(stage.clone());
        let update = ProgressUpdate::with_description(stage, description, self.start_time);
        self.reporter.report_progress(update);
    }

    /// Report completion with final timings
    pub fn report_completion(&self, timings: ProcessingTimings) {
        self.reporter.report_completion(timings);
    }

    /// Report an error during processing
    pub fn report_error(&self, error: &str) {
        let stage = self
            .current_stage
            .clone()
            .unwrap_or(ProcessingStage::Initialization);
        self.reporter.report_error(stage, error);
    }

    /// Get the elapsed time since tracking started
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Get the current processing stage
    #[must_use]
    pub fn current_stage(&self) -> Option<&ProcessingStage> {
        self.current_stage.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Test progress reporter that captures reports for verification
    #[derive(Default)]
    struct TestProgressReporter {
        progress_updates: Arc<Mutex<Vec<ProgressUpdate>>>,
        completions: Arc<Mutex<Vec<ProcessingTimings>>>,
        errors: Arc<Mutex<Vec<(ProcessingStage, String)>>>,
    }

    impl TestProgressReporter {
        fn new() -> Self {
            Self::default()
        }

        #[allow(dead_code)]
        fn get_progress_updates(&self) -> Vec<ProgressUpdate> {
            self.progress_updates.lock().unwrap().clone()
        }

        #[allow(dead_code)]
        fn get_completions(&self) -> Vec<ProcessingTimings> {
            self.completions.lock().unwrap().clone()
        }

        #[allow(dead_code)]
        fn get_errors(&self) -> Vec<(ProcessingStage, String)> {
            self.errors.lock().unwrap().clone()
        }
    }

    impl ProgressReporter for TestProgressReporter {
        fn report_progress(&self, update: ProgressUpdate) {
            self.progress_updates.lock().unwrap().push(update);
        }

        fn report_completion(&self, timings: ProcessingTimings) {
            self.completions.lock().unwrap().push(timings);
        }

        fn report_error(&self, stage: ProcessingStage, error: &str) {
            self.errors.lock().unwrap().push((stage, error.to_string()));
        }
    }

    #[test]
    fn test_processing_stage_descriptions() {
        assert_eq!(
            ProcessingStage::Initialization.description(),
            "Initializing model and backend"
        );
        assert_eq!(
            ProcessingStage::Inference.description(),
            "Running AI inference"
        );
        assert_eq!(
            ProcessingStage::Completed.description(),
            "Processing completed"
        );
    }

    #[test]
    fn test_processing_stage_progress_percentages() {
        assert_eq!(ProcessingStage::Initialization.progress_percentage(), 5);
        assert_eq!(ProcessingStage::Inference.progress_percentage(), 70);
        assert_eq!(ProcessingStage::Completed.progress_percentage(), 100);
    }

    #[test]
    fn test_progress_update_creation() {
        let start_time = Instant::now();
        let update = ProgressUpdate::new(ProcessingStage::Inference, start_time);

        assert_eq!(update.stage, ProcessingStage::Inference);
        assert_eq!(update.progress, 70);
        assert_eq!(update.description, "Running AI inference");
        assert!(update.elapsed_ms < 100); // Should be very small
        assert!(update.eta_ms.is_none());
    }

    #[test]
    fn test_progress_update_with_eta() {
        let start_time = Instant::now();
        let update = ProgressUpdate::new(ProcessingStage::Preprocessing, start_time).with_eta(1500);

        assert_eq!(update.eta_ms, Some(1500));
    }

    #[test]
    fn test_no_op_progress_reporter() {
        let reporter = NoOpProgressReporter;
        let update = ProgressUpdate::new(ProcessingStage::Inference, Instant::now());
        let timings = ProcessingTimings::default();

        // These should not panic and should silently discard the calls
        reporter.report_progress(update);
        reporter.report_completion(timings);
        reporter.report_error(ProcessingStage::Inference, "test error");
    }

    #[test]
    #[allow(clippy::get_first)]
    fn test_progress_tracker() {
        let test_reporter = TestProgressReporter::new();
        let progress_updates = test_reporter.progress_updates.clone();
        let completions = test_reporter.completions.clone();
        let errors = test_reporter.errors.clone();

        let mut tracker = ProgressTracker::new(Box::new(test_reporter));

        // Test stage reporting
        tracker.report_stage(ProcessingStage::Initialization);
        tracker.report_stage(ProcessingStage::Inference);

        // Test custom description
        tracker.report_stage_with_description(
            ProcessingStage::BackgroundRemoval,
            "Custom description".to_string(),
        );

        // Test completion
        let timings = ProcessingTimings::default();
        tracker.report_completion(timings.clone());

        // Test error reporting
        tracker.report_error("Test error message");

        // Verify reports were captured
        let updates = progress_updates.lock().unwrap();
        assert_eq!(updates.len(), 3);
        assert_eq!(
            updates.get(0).unwrap().stage,
            ProcessingStage::Initialization
        );
        assert_eq!(updates.get(1).unwrap().stage, ProcessingStage::Inference);
        assert_eq!(
            updates.get(2).unwrap().stage,
            ProcessingStage::BackgroundRemoval
        );
        assert_eq!(updates.get(2).unwrap().description, "Custom description");

        let completion_list = completions.lock().unwrap();
        assert_eq!(completion_list.len(), 1);

        let error_list = errors.lock().unwrap();
        assert_eq!(error_list.len(), 1);
        assert_eq!(
            error_list.get(0).unwrap().0,
            ProcessingStage::BackgroundRemoval
        );
        assert_eq!(error_list.get(0).unwrap().1, "Test error message");
    }

    #[test]
    fn test_progress_tracker_convenience_constructors() {
        let no_op_tracker = ProgressTracker::no_op();
        assert!(no_op_tracker.current_stage().is_none());

        let console_tracker = ProgressTracker::console(true);
        assert!(console_tracker.current_stage().is_none());
        assert!(console_tracker.elapsed_ms() < 100);
    }
}
