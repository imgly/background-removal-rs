//! Testing utilities and tools for the background removal library
//!
//! This crate provides comprehensive testing infrastructure including:
//! - Real image fixtures and loading utilities
//! - Image comparison and accuracy metrics
//! - HTML report generation with visual comparisons
//! - Performance benchmarking tools

pub mod comparison;
pub mod fixtures;
pub mod report;

pub use comparison::*;
pub use fixtures::*;
pub use report::*;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Test case definition for real image testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub category: String,
    pub input_file: String,
    pub expected_output_file: String,
    pub expected_accuracy: f64,
    pub description: String,
    pub tags: Vec<String>,
    pub complexity_level: ComplexityLevel,
}

/// Complexity level for test categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComplexityLevel {
    Simple,
    Medium,
    Complex,
    Extreme,
}

/// Validation thresholds for different metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationThresholds {
    pub pixel_accuracy: f64,
    pub ssim: f64,
    pub edge_accuracy: f64,
    pub processing_time_ms: u64,
}

impl Default for ValidationThresholds {
    fn default() -> Self {
        Self {
            pixel_accuracy: 0.30, // Lower threshold since we're using visual quality score
            ssim: 0.50,           // More forgiving SSIM threshold
            edge_accuracy: 0.60,  // More reasonable edge accuracy
            processing_time_ms: 5000,
        }
    }
}

/// Test result for a single image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_case: TestCase,
    pub passed: bool,
    pub metrics: TestMetrics,
    #[serde(with = "duration_serde")]
    pub processing_time: std::time::Duration,
    pub error_message: Option<String>,
    pub output_path: Option<PathBuf>,
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub(crate) fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

/// Accuracy and quality metrics for test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    pub pixel_accuracy: f64,
    pub ssim: f64,
    pub edge_accuracy: f64,
    pub visual_quality_score: f64,
    pub mean_squared_error: f64,
}

/// Test session containing all results and metadata
#[derive(Debug)]
pub struct TestSession {
    pub session_id: String,
    pub start_time: std::time::SystemTime,
    pub end_time: Option<std::time::SystemTime>,
    pub results: Vec<TestResult>,
    pub summary: TestSummary,
}

/// Summary statistics for test session
#[derive(Debug, Default)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub average_accuracy: f64,
    pub average_processing_time: std::time::Duration,
    pub categories_tested: Vec<String>,
}

impl Default for TestSession {
    fn default() -> Self {
        Self::new()
    }
}

impl TestSession {
    #[must_use]
    pub fn new() -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            start_time: std::time::SystemTime::now(),
            end_time: None,
            results: Vec::new(),
            summary: TestSummary::default(),
        }
    }

    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
        self.update_summary();
    }

    pub fn finalize(&mut self) {
        self.end_time = Some(std::time::SystemTime::now());
        self.update_summary();
    }

    fn update_summary(&mut self) {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();

        let avg_accuracy = if total > 0 {
            self.results
                .iter()
                .map(|r| r.metrics.pixel_accuracy)
                .sum::<f64>()
                / total as f64
        } else {
            0.0
        };

        let avg_time = if total > 0 {
            let total_ms: u64 = self
                .results
                .iter()
                .map(|r| r.processing_time.as_millis() as u64)
                .sum();
            std::time::Duration::from_millis(total_ms / total as u64)
        } else {
            std::time::Duration::from_millis(0)
        };

        let mut categories: Vec<String> = self
            .results
            .iter()
            .map(|r| r.test_case.category.clone())
            .collect();
        categories.sort();
        categories.dedup();

        self.summary = TestSummary {
            total_tests: total,
            passed_tests: passed,
            failed_tests: total - passed,
            average_accuracy: avg_accuracy,
            average_processing_time: avg_time,
            categories_tested: categories,
        };
    }
}

/// Error types for testing operations
#[derive(thiserror::Error, Debug)]
pub enum TestingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Background removal error: {0}")]
    BackgroundRemoval(#[from] imgly_bgremove::BgRemovalError),

    #[error("Test case not found: {0}")]
    TestCaseNotFound(String),

    #[error("Reference image not found: {0}")]
    ReferenceImageNotFound(String),

    #[error("Invalid test configuration: {0}")]
    InvalidConfiguration(String),
}

pub type Result<T> = std::result::Result<T, TestingError>;
