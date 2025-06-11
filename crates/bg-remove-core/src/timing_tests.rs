//! Tests for timing breakdown functionality
//!
//! These tests validate that timing measurements are working correctly
//! and that the timing breakdown calculations are accurate.

use crate::{
    types::{ProcessingTimings, RemovalResult},
    ImageProcessor, RemovalConfig,
};
use image::DynamicImage;

/// Test that basic timing structure works correctly
#[test]
fn test_processing_timings_creation() {
    let mut timings = ProcessingTimings::new();

    // Verify default values
    assert_eq!(timings.model_load_ms, 0);
    assert_eq!(timings.image_decode_ms, 0);
    assert_eq!(timings.preprocessing_ms, 0);
    assert_eq!(timings.inference_ms, 0);
    assert_eq!(timings.postprocessing_ms, 0);
    assert_eq!(timings.image_encode_ms, None);
    assert_eq!(timings.total_ms, 0);

    // Test setting values
    timings.image_decode_ms = 10;
    timings.preprocessing_ms = 50;
    timings.inference_ms = 100;
    timings.postprocessing_ms = 30;
    timings.total_ms = 200;

    assert_eq!(timings.image_decode_ms, 10);
    assert_eq!(timings.preprocessing_ms, 50);
    assert_eq!(timings.inference_ms, 100);
    assert_eq!(timings.postprocessing_ms, 30);
    assert_eq!(timings.total_ms, 200);
}

/// Test timing breakdown percentage calculations
#[test]
fn test_timing_breakdown_percentages() {
    let mut timings = ProcessingTimings::new();
    timings.image_decode_ms = 10; // 5%
    timings.preprocessing_ms = 30; // 15%
    timings.inference_ms = 140; // 70%
    timings.postprocessing_ms = 20; // 10%
    timings.total_ms = 200;

    let breakdown = timings.breakdown_percentages();

    // Test percentages (with small tolerance for floating point)
    assert!((breakdown.decode_pct - 5.0).abs() < 0.1);
    assert!((breakdown.preprocessing_pct - 15.0).abs() < 0.1);
    assert!((breakdown.inference_pct - 70.0).abs() < 0.1);
    assert!((breakdown.postprocessing_pct - 10.0).abs() < 0.1);
    assert!((breakdown.encode_pct - 0.0).abs() < 0.1);
    assert!((breakdown.other_pct - 0.0).abs() < 0.1);
}

/// Test timing breakdown with image encoding
#[test]
fn test_timing_breakdown_with_encoding() {
    let mut timings = ProcessingTimings::new();
    timings.image_decode_ms = 10; // 5%
    timings.preprocessing_ms = 30; // 15%
    timings.inference_ms = 120; // 60%
    timings.postprocessing_ms = 20; // 10%
    timings.image_encode_ms = Some(20); // 10%
    timings.total_ms = 200;

    let breakdown = timings.breakdown_percentages();

    assert!((breakdown.decode_pct - 5.0).abs() < 0.1);
    assert!((breakdown.preprocessing_pct - 15.0).abs() < 0.1);
    assert!((breakdown.inference_pct - 60.0).abs() < 0.1);
    assert!((breakdown.postprocessing_pct - 10.0).abs() < 0.1);
    assert!((breakdown.encode_pct - 10.0).abs() < 0.1);
    assert!((breakdown.other_pct - 0.0).abs() < 0.1);
}

/// Test "other" overhead calculation
#[test]
fn test_other_overhead_calculation() {
    let mut timings = ProcessingTimings::new();
    timings.image_decode_ms = 10;
    timings.preprocessing_ms = 30;
    timings.inference_ms = 100;
    timings.postprocessing_ms = 20;
    timings.image_encode_ms = Some(15);
    timings.total_ms = 200; // 25ms unaccounted for

    let other_ms = timings.other_overhead_ms();
    assert_eq!(other_ms, 25);

    let breakdown = timings.breakdown_percentages();
    assert!((breakdown.other_pct - 12.5).abs() < 0.1); // 25/200 = 12.5%
}

/// Test inference ratio calculation
#[test]
fn test_inference_ratio() {
    let mut timings = ProcessingTimings::new();
    timings.inference_ms = 150;
    timings.total_ms = 200;

    let ratio = timings.inference_ratio();
    assert!((ratio - 0.75).abs() < 0.01); // 150/200 = 0.75

    // Test zero total case
    timings.total_ms = 0;
    let zero_ratio = timings.inference_ratio();
    assert_eq!(zero_ratio, 0.0);
}

/// Test timing breakdown with zero total (edge case)
#[test]
fn test_timing_breakdown_zero_total() {
    let timings = ProcessingTimings::new(); // All zeros
    let breakdown = timings.breakdown_percentages();

    assert_eq!(breakdown.decode_pct, 0.0);
    assert_eq!(breakdown.preprocessing_pct, 0.0);
    assert_eq!(breakdown.inference_pct, 0.0);
    assert_eq!(breakdown.postprocessing_pct, 0.0);
    assert_eq!(breakdown.encode_pct, 0.0);
    assert_eq!(breakdown.other_pct, 0.0);
}

/// Test timing summary string generation
#[test]
fn test_timing_summary_generation() {
    // Create a mock removal result with timing data
    let mut timings = ProcessingTimings::new();
    timings.image_decode_ms = 10;
    timings.preprocessing_ms = 50;
    timings.inference_ms = 300;
    timings.postprocessing_ms = 40;
    timings.image_encode_ms = Some(25);
    timings.total_ms = 425;

    let mut metadata = crate::types::ProcessingMetadata::new("TestModel".to_string());
    metadata.set_detailed_timings(timings);

    // Create a simple test image
    let test_image = DynamicImage::new_rgb8(100, 100);
    let mask = crate::types::SegmentationMask::new(vec![255; 10000], (100, 100));

    let result = RemovalResult::new(test_image, mask, (100, 100), metadata);

    let summary = result.timing_summary();

    // Verify the summary contains expected information
    assert!(summary.contains("Total: 425ms"));
    assert!(summary.contains("Decode: 10ms"));
    assert!(summary.contains("Preprocess: 50ms"));
    assert!(summary.contains("Inference: 300ms"));
    assert!(summary.contains("Postprocess: 40ms"));
    assert!(summary.contains("Encode: 25ms"));
}

/// Integration test: validate timing measurements are reasonable
#[tokio::test]
async fn test_timing_measurements_integration() {
    let config = RemovalConfig::builder()
        .debug(true) // Use mock backend for consistent behavior
        .build()
        .unwrap();

    let mut processor = ImageProcessor::new(&config).unwrap();

    // Create a test image
    let test_image = DynamicImage::new_rgb8(512, 512);

    let result = processor.process_image(test_image).unwrap();
    let timings = result.timings();

    // Basic sanity checks
    assert!(timings.total_ms > 0, "Total time should be greater than 0");
    assert!(
        timings.total_ms >= timings.inference_ms,
        "Total time should be >= inference time"
    );
    assert!(
        timings.total_ms >= timings.preprocessing_ms,
        "Total time should be >= preprocessing time"
    );
    assert!(
        timings.total_ms >= timings.postprocessing_ms,
        "Total time should be >= postprocessing time"
    );

    // Verify breakdown adds up (allowing for rounding)
    let breakdown = timings.breakdown_percentages();
    let total_percentage = breakdown.decode_pct
        + breakdown.preprocessing_pct
        + breakdown.inference_pct
        + breakdown.postprocessing_pct
        + breakdown.encode_pct
        + breakdown.other_pct;

    assert!(
        (total_percentage - 100.0).abs() < 1.0,
        "Total percentage should be close to 100%, got: {total_percentage}"
    );
}

/// Test that timing measurements are monotonic and reasonable
#[test]
fn test_timing_invariants() {
    let mut timings = ProcessingTimings::new();

    // Set some realistic timing values
    timings.image_decode_ms = 5;
    timings.preprocessing_ms = 50;
    timings.inference_ms = 500;
    timings.postprocessing_ms = 25;
    timings.image_encode_ms = Some(15);

    // Total should be sum of parts plus some overhead
    let measured_sum = timings.image_decode_ms
        + timings.preprocessing_ms
        + timings.inference_ms
        + timings.postprocessing_ms
        + timings.image_encode_ms.unwrap_or(0);

    timings.total_ms = measured_sum + 5; // 5ms overhead

    // Test invariants
    assert!(
        timings.total_ms >= measured_sum,
        "Total time should be >= sum of measured phases"
    );
    assert!(
        timings.other_overhead_ms() == 5,
        "Overhead calculation should be correct"
    );

    // Inference should typically be the largest component
    assert!(timings.inference_ms > timings.preprocessing_ms);
    assert!(timings.inference_ms > timings.postprocessing_ms);
    assert!(timings.inference_ms > timings.image_decode_ms);
}

/// Performance regression test: ensure timing overhead is minimal
#[tokio::test]
async fn test_timing_overhead() {
    let config = RemovalConfig::builder().debug(true).build().unwrap();

    let test_image = DynamicImage::new_rgb8(256, 256);

    // Measure processing time without timing details
    let start = std::time::Instant::now();
    let mut processor = ImageProcessor::new(&config).unwrap();
    let result = processor.process_image(test_image.clone()).unwrap();
    let elapsed_ms = start.elapsed().as_millis() as u64;

    // Check that timing overhead is reasonable (< 5% of total time)
    let timings = result.timings();
    let overhead_ms = timings.other_overhead_ms();
    let overhead_percentage = (overhead_ms as f64 / elapsed_ms as f64) * 100.0;

    assert!(
        overhead_percentage < 5.0,
        "Timing overhead should be < 5% of total time, got {overhead_percentage:.1}%"
    );
}
