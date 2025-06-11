//! Performance integration tests for background removal
//!
//! These tests validate that background removal performance meets acceptable thresholds
//! and that different execution providers work correctly.

use bg_remove_core::config::ExecutionProvider;
use bg_remove_core::{remove_background, RemovalConfig};
use std::path::Path;
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_performance_threshold_basic() {
    let config = RemovalConfig::builder()
        .execution_provider(ExecutionProvider::Cpu)
        .debug(false)
        .build()
        .expect("Failed to create config");

    let input_path = "assets/input/portraits/portrait_single_simple_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping performance test: test file not found");
        return;
    }

    let start = Instant::now();
    let result = remove_background(input_path, &config).await;
    let duration = start.elapsed();

    assert!(result.is_ok(), "Background removal should succeed");

    // Performance threshold: should complete within 10 seconds for integration test
    assert!(
        duration < Duration::from_secs(10),
        "Background removal should complete within 10s, took {}ms",
        duration.as_millis()
    );

    println!("✅ Performance test passed: {}ms", duration.as_millis());
}

#[tokio::test]
async fn test_execution_provider_performance() {
    let providers = vec![
        ("CPU", ExecutionProvider::Cpu),
        ("Auto", ExecutionProvider::Auto),
    ];

    let input_path = "assets/input/portraits/portrait_action_motion.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping execution provider performance test: test file not found");
        return;
    }

    let mut results = Vec::new();

    for (name, provider) in providers {
        let config = RemovalConfig::builder()
            .execution_provider(provider)
            .debug(false)
            .build()
            .expect("Failed to create config");

        let start = Instant::now();
        let result = remove_background(input_path, &config).await;
        let duration = start.elapsed();

        assert!(result.is_ok(), "{} provider should succeed", name);

        // Each provider should complete within reasonable time
        assert!(
            duration < Duration::from_secs(15),
            "{} provider should complete within 15s, took {}ms",
            name,
            duration.as_millis()
        );

        results.push((name, duration));
        println!("✅ {} provider: {}ms", name, duration.as_millis());
    }

    // Verify we tested multiple providers
    assert!(
        results.len() >= 2,
        "Should test at least 2 execution providers"
    );
}

#[tokio::test]
async fn test_memory_usage_basic() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let input_path = "assets/input/products/product_clothing_white_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping memory test: test file not found");
        return;
    }

    // Basic memory test - ensure multiple consecutive runs work
    for i in 0..3 {
        let result = remove_background(input_path, &config).await;
        assert!(result.is_ok(), "Run {} should succeed", i + 1);

        let result = result.unwrap();
        assert!(
            !result.mask.data.is_empty(),
            "Run {} should produce valid mask",
            i + 1
        );
    }

    println!("✅ Memory test passed: 3 consecutive runs completed");
}

#[tokio::test]
async fn test_concurrent_processing() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let test_images = vec![
        "assets/input/portraits/portrait_single_simple_bg.jpg",
        "assets/input/products/product_clothing_white_bg.jpg",
    ];

    // Filter to only existing files
    let existing_images: Vec<_> = test_images
        .into_iter()
        .filter(|path| Path::new(path).exists())
        .collect();

    if existing_images.is_empty() {
        println!("⏭️  Skipping concurrent test: no test files found");
        return;
    }

    let start = Instant::now();

    // Process images sequentially (due to Send constraints)
    let mut results = Vec::new();

    for &path in &existing_images {
        let result = remove_background(path, &config).await;
        results.push(result);
    }
    let duration = start.elapsed();

    // All tasks should complete successfully
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_ok(), "Sequential task {} should succeed", i);
    }

    println!(
        "✅ Sequential test passed: {} images processed in {}ms",
        existing_images.len(),
        duration.as_millis()
    );
}

#[tokio::test]
#[cfg(feature = "regression-tests")]
async fn test_performance_regression() {
    let config = RemovalConfig::builder()
        .execution_provider(ExecutionProvider::Cpu)
        .debug(false)
        .build()
        .expect("Failed to create config");

    let test_images = vec![
        (
            "portrait",
            "assets/input/portraits/portrait_single_simple_bg.jpg",
        ),
        (
            "product",
            "assets/input/products/product_clothing_white_bg.jpg",
        ),
        ("complex", "assets/input/complex/complex_group_photo.jpg"),
    ];

    let mut all_passed = true;

    for (category, input_path) in test_images {
        if !Path::new(input_path).exists() {
            println!("⏭️  Skipping {}: file not found", category);
            continue;
        }

        let mut durations = Vec::new();

        // Run multiple times to get stable measurements
        for _ in 0..3 {
            let start = Instant::now();
            let result = remove_background(input_path, &config).await;
            let duration = start.elapsed();

            if result.is_ok() {
                durations.push(duration);
            } else {
                all_passed = false;
                println!("❌ {} failed: {:?}", category, result.err());
            }
        }

        if !durations.is_empty() {
            let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
            let max_expected = match category {
                "portrait" => Duration::from_secs(2),
                "product" => Duration::from_secs(2),
                "complex" => Duration::from_secs(5),
                _ => Duration::from_secs(3),
            };

            if avg_duration <= max_expected {
                println!(
                    "✅ {} performance: {}ms (within {}ms threshold)",
                    category,
                    avg_duration.as_millis(),
                    max_expected.as_millis()
                );
            } else {
                all_passed = false;
                println!(
                    "❌ {} performance regression: {}ms (exceeds {}ms threshold)",
                    category,
                    avg_duration.as_millis(),
                    max_expected.as_millis()
                );
            }
        }
    }

    assert!(
        all_passed,
        "Performance regression test should pass all categories"
    );
}
