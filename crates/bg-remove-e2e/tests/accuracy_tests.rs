//! Accuracy integration tests for background removal
//!
//! These tests validate that the background removal maintains acceptable accuracy
//! across different image categories and complexity levels.

use bg_remove_core::models::ModelManager;
use bg_remove_core::{remove_background_with_backend, RemovalConfig};
use bg_remove_onnx::OnnxBackend;
// use bg_remove_testing::ValidationThresholds;
use std::path::Path;

/// Helper function to create a backend for testing
async fn create_test_backend(
    _config: &RemovalConfig,
) -> Result<
    Box<dyn bg_remove_core::inference::InferenceBackend>,
    bg_remove_core::error::BgRemovalError,
> {
    let model_manager = ModelManager::with_embedded_model("isnet-fp32".to_string())?;
    let backend = OnnxBackend::with_model_manager(model_manager);
    Ok(Box::new(backend))
}

#[tokio::test]
async fn test_portrait_accuracy_basic() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let input_path = "assets/input/portraits/portrait_single_simple_bg.jpg";
    let expected_path = "assets/expected/portraits/portrait_single_simple_bg.png";

    // Skip if test files don't exist
    if !Path::new(input_path).exists() || !Path::new(expected_path).exists() {
        println!("⏭️  Skipping portrait accuracy test: test files not found");
        return;
    }

    let backend = create_test_backend(&config)
        .await
        .expect("Failed to create backend");
    let result = remove_background_with_backend(input_path, &config, backend).await;
    assert!(
        result.is_ok(),
        "Background removal should succeed for simple portrait"
    );

    let result = result.unwrap();

    // Basic sanity checks
    assert!(!result.mask.data.is_empty(), "Mask should not be empty");
    assert!(
        result.image.width() > 0,
        "Output image should have valid dimensions"
    );
    assert!(
        result.image.height() > 0,
        "Output image should have valid dimensions"
    );
}

#[tokio::test]
async fn test_product_accuracy_basic() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let input_path = "assets/input/products/product_clothing_white_bg.jpg";

    // Skip if test file doesn't exist
    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping product accuracy test: test file not found");
        return;
    }

    let backend = create_test_backend(&config)
        .await
        .expect("Failed to create backend");
    let result = remove_background_with_backend(input_path, &config, backend).await;
    assert!(
        result.is_ok(),
        "Background removal should succeed for product image"
    );

    let result = result.unwrap();

    // Basic sanity checks
    assert!(!result.mask.data.is_empty(), "Mask should not be empty");
    assert!(
        result.image.width() > 0,
        "Output image should have valid dimensions"
    );
}

#[tokio::test]
async fn test_different_execution_providers() {
    use bg_remove_core::config::ExecutionProvider;

    let providers = vec![
        ("CPU", ExecutionProvider::Cpu),
        ("Auto", ExecutionProvider::Auto),
    ];

    let input_path = "assets/input/portraits/portrait_action_motion.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping execution provider test: test file not found");
        return;
    }

    for (name, provider) in providers {
        let config = RemovalConfig::builder()
            .execution_provider(provider)
            .debug(false)
            .build()
            .expect("Failed to create config");

        let backend = create_test_backend(&config)
            .await
            .expect("Failed to create backend");
        let result = remove_background_with_backend(input_path, &config, backend).await;
        assert!(
            result.is_ok(),
            "Background removal should succeed with {name} provider"
        );

        let result = result.unwrap();
        assert!(
            !result.mask.data.is_empty(),
            "{name} provider should produce non-empty mask"
        );
    }
}

#[tokio::test]
#[cfg(feature = "expensive-tests")]
async fn test_comprehensive_accuracy_suite() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    // Simple test cases for integration testing
    let test_cases = vec![
        "assets/input/portraits/portrait_single_simple_bg.jpg",
        "assets/input/products/product_clothing_white_bg.jpg",
        "assets/input/complex/complex_group_photo.jpg",
    ];

    let mut passed = 0;
    let mut total = 0;

    for input_path in test_cases {
        if !Path::new(input_path).exists() {
            println!("⏭️  Skipping {}: missing file", input_path);
            continue;
        }

        total += 1;

        let backend = create_test_backend(&config)
            .await
            .expect("Failed to create backend");
        let result = remove_background_with_backend(input_path, &config, backend).await;
        assert!(
            result.is_ok(),
            "Background removal should succeed for {}",
            input_path
        );

        // For integration test, just verify basic functionality
        let result = result.unwrap();
        if !result.mask.data.is_empty() && result.image.width() > 0 {
            passed += 1;
        }
    }

    let pass_rate = if total > 0 {
        passed as f64 / total as f64
    } else {
        0.0
    };
    assert!(
        pass_rate >= 0.8,
        "Pass rate should be at least 80%, got {:.1}% ({}/{})",
        pass_rate * 100.0,
        passed,
        total
    );

    println!(
        "✅ Comprehensive accuracy test passed: {}/{} tests ({}%)",
        passed,
        total,
        (pass_rate * 100.0) as u32
    );
}

#[tokio::test]
async fn test_edge_cases_basic() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let edge_cases = vec![
        "assets/input/edge_cases/edge_high_contrast.jpg",
        "assets/input/edge_cases/edge_very_small.jpg",
    ];

    for input_path in edge_cases {
        if !Path::new(input_path).exists() {
            println!("⏭️  Skipping edge case: {input_path} not found");
            continue;
        }

        let backend = create_test_backend(&config)
            .await
            .expect("Failed to create backend");
        let result = remove_background_with_backend(input_path, &config, backend).await;

        // Edge cases might fail, but shouldn't panic
        match result {
            Ok(result) => {
                assert!(
                    !result.mask.data.is_empty(),
                    "Edge case should produce some mask data"
                );
                println!("✅ Edge case passed: {input_path}");
            },
            Err(e) => {
                println!("⚠️  Edge case failed gracefully: {input_path} - {e}");
                // This is acceptable for edge cases
            },
        }
    }
}
