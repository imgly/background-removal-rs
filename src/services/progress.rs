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
    use instant::Instant;
    use std::sync::{Arc, Mutex};

    // Mock progress reporter for testing
    #[derive(Debug)]
    struct MockProgressReporter {
        updates: Arc<Mutex<Vec<ProgressUpdate>>>,
        completions: Arc<Mutex<Vec<ProcessingTimings>>>,
        errors: Arc<Mutex<Vec<(ProcessingStage, String)>>>,
    }

    impl ProgressReporter for MockProgressReporter {
        fn report_progress(&self, update: ProgressUpdate) {
            self.updates.lock().unwrap().push(update);
        }

        fn report_completion(&self, timings: ProcessingTimings) {
            self.completions.lock().unwrap().push(timings);
        }

        fn report_error(&self, stage: ProcessingStage, error: &str) {
            self.errors.lock().unwrap().push((stage, error.to_string()));
        }
    }

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

    #[test]
    fn test_processing_stage_all_variants() {
        let stages = vec![
            ProcessingStage::Initialization,
            ProcessingStage::ImageLoading,
            ProcessingStage::ColorProfileExtraction,
            ProcessingStage::Preprocessing,
            ProcessingStage::Inference,
            ProcessingStage::MaskGeneration,
            ProcessingStage::BackgroundRemoval,
            ProcessingStage::FormatConversion,
            ProcessingStage::FileSaving,
            ProcessingStage::Completed,
        ];

        for stage in stages {
            // Test description
            let description = stage.description();
            assert!(
                !description.is_empty(),
                "Description should not be empty for {:?}",
                stage
            );

            // Test progress percentage
            let progress = stage.progress_percentage();
            assert!(progress <= 100, "Progress should be 0-100 for {:?}", stage);

            // Test cloning
            let cloned = stage.clone();
            assert_eq!(stage, cloned);

            // Test debug formatting
            let debug_str = format!("{:?}", stage);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_processing_stage_progress_ordering() {
        // Verify that progress percentages are in ascending order
        let stages_with_progress = vec![
            (ProcessingStage::Initialization, 5),
            (ProcessingStage::ImageLoading, 10),
            (ProcessingStage::ColorProfileExtraction, 15),
            (ProcessingStage::Preprocessing, 25),
            (ProcessingStage::Inference, 70),
            (ProcessingStage::MaskGeneration, 85),
            (ProcessingStage::BackgroundRemoval, 95),
            (ProcessingStage::FormatConversion, 98),
            (ProcessingStage::FileSaving, 99),
            (ProcessingStage::Completed, 100),
        ];

        for (i, (stage, expected_progress)) in stages_with_progress.iter().enumerate() {
            assert_eq!(stage.progress_percentage(), *expected_progress);

            // Verify ascending order
            if i > 0 {
                let prev_progress = stages_with_progress[i - 1].1;
                assert!(
                    expected_progress > &prev_progress,
                    "Progress should be ascending: {:?} ({}) should be > previous ({})",
                    stage,
                    expected_progress,
                    prev_progress
                );
            }
        }
    }

    #[test]
    fn test_progress_update_creation_variants() {
        let stage = ProcessingStage::Inference;

        // Test simple creation
        let update1 = ProgressUpdate::new(stage.clone(), Instant::now());
        assert_eq!(update1.stage, stage);
        assert_eq!(update1.description, stage.description());
        assert!(update1.eta_ms.is_none());

        // Test with custom description
        let custom_desc = "Custom inference description";
        let update2 = ProgressUpdate::with_description(
            stage.clone(),
            custom_desc.to_string(),
            Instant::now(),
        );
        assert_eq!(update2.stage, stage);
        assert_eq!(update2.description, custom_desc);
        assert!(update2.eta_ms.is_none());

        // Test with ETA
        let eta_ms = 5000;
        let update3 = ProgressUpdate::new(stage.clone(), Instant::now()).with_eta(eta_ms);
        assert_eq!(update3.stage, stage);
        assert_eq!(update3.description, stage.description());
        assert_eq!(update3.eta_ms, Some(eta_ms));
    }

    #[test]
    fn test_progress_update_equality() {
        let stage = ProcessingStage::Preprocessing;

        let update1 = ProgressUpdate::new(stage.clone(), Instant::now());
        let update2 = ProgressUpdate::new(stage.clone(), Instant::now());
        let update3 = ProgressUpdate::with_description(
            stage.clone(),
            "Different".to_string(),
            Instant::now(),
        );

        assert_eq!(update1.stage, update2.stage);
        assert_ne!(update1.description, update3.description);
    }

    #[test]
    fn test_progress_update_debug_formatting() {
        let update = ProgressUpdate::new(ProcessingStage::Inference, Instant::now()).with_eta(2500);
        let debug_str = format!("{:?}", update);

        assert!(debug_str.contains("Inference"));
        assert!(debug_str.contains("2500"));
    }

    #[test]
    fn test_no_op_progress_reporter_comprehensive() {
        let reporter = NoOpProgressReporter;

        // All methods should complete without side effects
        reporter.report_progress(ProgressUpdate::new(
            ProcessingStage::Initialization,
            Instant::now(),
        ));
        reporter.report_completion(ProcessingTimings::default());
        reporter.report_error(ProcessingStage::Inference, "Test error");

        // Test with various stages
        for stage in &[
            ProcessingStage::ImageLoading,
            ProcessingStage::Preprocessing,
            ProcessingStage::Inference,
            ProcessingStage::Completed,
        ] {
            reporter.report_progress(ProgressUpdate::new(stage.clone(), Instant::now()));
        }
    }

    #[test]
    fn test_console_progress_reporter_creation() {
        let verbose_reporter = ConsoleProgressReporter::new(true);
        let non_verbose_reporter = ConsoleProgressReporter::new(false);

        // Should create without panicking
        // We can't easily test the actual output since it goes to logs
        // but we can verify the reporters handle various inputs
        let update = ProgressUpdate::new(ProcessingStage::Inference, Instant::now());
        verbose_reporter.report_progress(update.clone());
        non_verbose_reporter.report_progress(update);

        let timings = ProcessingTimings::default();
        verbose_reporter.report_completion(timings.clone());
        non_verbose_reporter.report_completion(timings);

        verbose_reporter.report_error(ProcessingStage::FileSaving, "Test error");
        non_verbose_reporter.report_error(ProcessingStage::FileSaving, "Test error");
    }

    #[test]
    fn test_progress_tracker_stage_tracking() {
        let mut tracker = ProgressTracker::no_op();

        // Initially no current stage
        assert!(tracker.current_stage().is_none());

        // Report various stages
        let stages = vec![
            ProcessingStage::Initialization,
            ProcessingStage::ImageLoading,
            ProcessingStage::Inference,
            ProcessingStage::Completed,
        ];

        for stage in stages {
            tracker.report_stage(stage.clone());
            assert_eq!(tracker.current_stage(), Some(&stage));
        }
    }

    #[test]
    fn test_progress_tracker_timing() {
        use std::thread;
        use std::time::Duration;

        let mut tracker = ProgressTracker::no_op();

        // Initial elapsed time should be small
        let initial_elapsed = tracker.elapsed_ms();
        assert!(initial_elapsed < 100);

        // Sleep briefly to ensure time passes
        thread::sleep(Duration::from_millis(10));

        // Elapsed time should have increased
        let later_elapsed = tracker.elapsed_ms();
        assert!(later_elapsed > initial_elapsed);

        // Report stage and check timing still works
        tracker.report_stage(ProcessingStage::Inference);
        let after_stage_elapsed = tracker.elapsed_ms();
        assert!(after_stage_elapsed >= later_elapsed);
    }

    #[test]
    fn test_progress_tracker_completion_reporting() {
        let progress_updates = Arc::new(Mutex::new(Vec::new()));
        let completions = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));

        let mock_reporter = MockProgressReporter {
            updates: progress_updates.clone(),
            completions: completions.clone(),
            errors: errors.clone(),
        };

        let tracker = ProgressTracker::new(Box::new(mock_reporter));

        // Create timings with specific values
        let mut timings = ProcessingTimings::default();
        timings.total_ms = 1500;
        timings.inference_ms = 800;
        timings.preprocessing_ms = 200;

        tracker.report_completion(timings.clone());

        let completion_list = completions.lock().unwrap();
        assert_eq!(completion_list.len(), 1);
        assert_eq!(completion_list[0].total_ms, 1500);
        assert_eq!(completion_list[0].inference_ms, 800);
        assert_eq!(completion_list[0].preprocessing_ms, 200);
    }

    #[test]
    fn test_progress_tracker_error_reporting_with_context() {
        let progress_updates = Arc::new(Mutex::new(Vec::new()));
        let completions = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));

        let mock_reporter = MockProgressReporter {
            updates: progress_updates,
            completions,
            errors: errors.clone(),
        };

        let mut tracker = ProgressTracker::new(Box::new(mock_reporter));

        // Set current stage
        tracker.report_stage(ProcessingStage::BackgroundRemoval);

        // Report error
        let error_message = "Failed to process image due to invalid format";
        tracker.report_error(error_message);

        let error_list = errors.lock().unwrap();
        assert_eq!(error_list.len(), 1);
        assert_eq!(error_list[0].0, ProcessingStage::BackgroundRemoval);
        assert_eq!(error_list[0].1, error_message);
    }

    #[test]
    fn test_progress_tracker_multiple_stage_transitions() {
        let progress_updates = Arc::new(Mutex::new(Vec::new()));
        let completions = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));

        let mock_reporter = MockProgressReporter {
            updates: progress_updates.clone(),
            completions,
            errors,
        };

        let mut tracker = ProgressTracker::new(Box::new(mock_reporter));

        // Simulate complete workflow
        let workflow_stages = vec![
            ProcessingStage::Initialization,
            ProcessingStage::ImageLoading,
            ProcessingStage::ColorProfileExtraction,
            ProcessingStage::Preprocessing,
            ProcessingStage::Inference,
            ProcessingStage::MaskGeneration,
            ProcessingStage::BackgroundRemoval,
            ProcessingStage::FormatConversion,
            ProcessingStage::FileSaving,
            ProcessingStage::Completed,
        ];

        for stage in &workflow_stages {
            tracker.report_stage(stage.clone());
        }

        let updates = progress_updates.lock().unwrap();
        assert_eq!(updates.len(), workflow_stages.len());

        // Verify stages are reported in correct order
        for (i, expected_stage) in workflow_stages.iter().enumerate() {
            assert_eq!(updates[i].stage, *expected_stage);
        }

        // Verify progress percentages are ascending
        for i in 1..updates.len() {
            let prev_progress = updates[i - 1].stage.progress_percentage();
            let curr_progress = updates[i].stage.progress_percentage();
            assert!(
                curr_progress >= prev_progress,
                "Progress should be ascending: {} >= {}",
                curr_progress,
                prev_progress
            );
        }
    }

    #[test]
    fn test_progress_tracker_custom_descriptions() {
        let progress_updates = Arc::new(Mutex::new(Vec::new()));
        let completions = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));

        let mock_reporter = MockProgressReporter {
            updates: progress_updates.clone(),
            completions,
            errors,
        };

        let mut tracker = ProgressTracker::new(Box::new(mock_reporter));

        // Test various custom descriptions
        let custom_stages = vec![
            (
                ProcessingStage::Initialization,
                "Loading ONNX model with GPU acceleration",
            ),
            (
                ProcessingStage::Preprocessing,
                "Resizing image to 1024x1024 and normalizing",
            ),
            (
                ProcessingStage::Inference,
                "Running AI model inference on Apple Silicon",
            ),
            (
                ProcessingStage::BackgroundRemoval,
                "Applying segmentation mask with edge smoothing",
            ),
        ];

        for (stage, description) in &custom_stages {
            tracker.report_stage_with_description(stage.clone(), description.to_string());
        }

        let updates = progress_updates.lock().unwrap();
        assert_eq!(updates.len(), custom_stages.len());

        for (i, (expected_stage, expected_description)) in custom_stages.iter().enumerate() {
            assert_eq!(updates[i].stage, *expected_stage);
            assert_eq!(updates[i].description, *expected_description);
        }
    }

    #[test]
    fn test_processing_timings_default_values() {
        let timings = ProcessingTimings::default();

        assert_eq!(timings.total_ms, 0);
        assert_eq!(timings.image_decode_ms, 0);
        assert_eq!(timings.preprocessing_ms, 0);
        assert_eq!(timings.inference_ms, 0);
        assert_eq!(timings.postprocessing_ms, 0);
    }

    #[test]
    fn test_processing_timings_debug_formatting() {
        let mut timings = ProcessingTimings::default();
        timings.total_ms = 1500;
        timings.inference_ms = 800;
        timings.preprocessing_ms = 200;

        let debug_str = format!("{:?}", timings);
        assert!(debug_str.contains("1500"));
        assert!(debug_str.contains("800"));
        assert!(debug_str.contains("200"));
    }

    #[test]
    fn test_trait_object_safety() {
        // Verify that ProgressReporter can be used as a trait object
        let reporters: Vec<Box<dyn ProgressReporter>> = vec![
            Box::new(NoOpProgressReporter),
            Box::new(ConsoleProgressReporter::new(true)),
            Box::new(ConsoleProgressReporter::new(false)),
        ];

        let update = ProgressUpdate::new(ProcessingStage::Inference, Instant::now());
        let timings = ProcessingTimings::default();

        for reporter in reporters {
            reporter.report_progress(update.clone());
            reporter.report_completion(timings.clone());
            reporter.report_error(ProcessingStage::FileSaving, "Test error");
        }
    }
}
