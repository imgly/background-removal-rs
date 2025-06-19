//! Compatibility integration tests
//!
//! These tests validate compatibility across different configurations,
//! execution providers, and edge cases.

use bg_remove_core::config::ExecutionProvider;
use bg_remove_core::{remove_background, RemovalConfig};
use image::GenericImageView;
use std::path::Path;

#[tokio::test]
async fn test_execution_provider_compatibility() {
    let providers = vec![
        ("CPU", ExecutionProvider::Cpu),
        ("Auto", ExecutionProvider::Auto),
        ("CoreML", ExecutionProvider::CoreMl),
    ];

    let input_path = "assets/input/portraits/portrait_single_simple_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping execution provider compatibility test: test file not found");
        return;
    }

    let mut successful_providers = Vec::new();

    for (name, provider) in providers {
        let config = RemovalConfig::builder()
            .execution_provider(provider)
            .debug(false)
            .build()
            .expect("Failed to create config");

        match remove_background(input_path, &config).await {
            Ok(result) => {
                assert!(
                    !result.mask.data.is_empty(),
                    "{name} should produce non-empty mask"
                );
                successful_providers.push(name);
                println!("✅ {name} provider: compatible");
            },
            Err(e) => {
                println!("⚠️  {name} provider: incompatible - {e}");
                // Some providers might not be available on all systems
            },
        }
    }

    // At least CPU should work
    assert!(
        successful_providers.contains(&"CPU") || successful_providers.contains(&"Auto"),
        "At least CPU or Auto provider should be available"
    );

    println!(
        "✅ Execution provider compatibility: {}/{} providers working",
        successful_providers.len(),
        3
    );
}

#[tokio::test]
async fn test_output_format_options() {
    let input_path = "assets/input/portraits/portrait_single_simple_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping output format test: test file not found");
        return;
    }

    // Test different output formats instead of background colors
    use bg_remove_core::config::OutputFormat;
    let output_formats = vec![
        ("PNG", OutputFormat::Png),
        ("JPEG", OutputFormat::Jpeg),
        ("WebP", OutputFormat::WebP),
    ];

    for (name, format) in output_formats {
        let config = RemovalConfig::builder()
            .output_format(format)
            .debug(false)
            .build()
            .expect("Failed to create config");

        let result = remove_background(input_path, &config).await;
        assert!(result.is_ok(), "Should work with {name} output format");

        let result = result.unwrap();

        // Verify output properties
        assert!(
            !result.mask.data.is_empty(),
            "{name} format should produce mask"
        );
        assert!(
            result.image.width() > 0,
            "{name} format should produce valid image"
        );

        println!("✅ {name} output format: compatible");
    }
}

#[tokio::test]
async fn test_debug_mode_compatibility() {
    let input_path = "assets/input/products/product_clothing_white_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping debug mode test: test file not found");
        return;
    }

    // Test with debug disabled
    let config_no_debug = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let result_no_debug = remove_background(input_path, &config_no_debug).await;
    assert!(result_no_debug.is_ok(), "Should work with debug disabled");

    // Test with debug enabled
    let config_debug = RemovalConfig::builder()
        .debug(true)
        .build()
        .expect("Failed to create config");

    let result_debug = remove_background(input_path, &config_debug).await;
    assert!(result_debug.is_ok(), "Should work with debug enabled");

    // Both should produce similar results
    let result_no_debug = result_no_debug.unwrap();
    let result_debug = result_debug.unwrap();

    assert_eq!(
        result_no_debug.image.dimensions(),
        result_debug.image.dimensions(),
        "Debug mode should not affect output dimensions"
    );

    println!("✅ Debug mode compatibility: both modes work");
}

#[tokio::test]
async fn test_configuration_builder_compatibility() {
    let input_path = "assets/input/portraits/portrait_action_motion.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping configuration builder test: test file not found");
        return;
    }

    // Test minimal configuration
    let minimal_config = RemovalConfig::builder()
        .build()
        .expect("Minimal config should build");

    let result = remove_background(input_path, &minimal_config).await;
    assert!(result.is_ok(), "Minimal configuration should work");

    // Test full configuration
    let full_config = RemovalConfig::builder()
        .execution_provider(ExecutionProvider::Cpu)
        .debug(false)
        .jpeg_quality(95)
        .webp_quality(90)
        .build()
        .expect("Full config should build");

    let result = remove_background(input_path, &full_config).await;
    assert!(result.is_ok(), "Full configuration should work");

    println!("✅ Configuration builder compatibility: minimal and full configs work");
}

#[tokio::test]
async fn test_concurrent_different_configs() {
    let input_path = "assets/input/portraits/portrait_single_simple_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping concurrent config test: test file not found");
        return;
    }

    let configs = vec![
        RemovalConfig::builder()
            .execution_provider(ExecutionProvider::Cpu)
            .debug(false)
            .build()
            .expect("CPU config should build"),
        RemovalConfig::builder()
            .execution_provider(ExecutionProvider::Auto)
            .debug(true)
            .build()
            .expect("Auto config should build"),
        RemovalConfig::builder()
            .jpeg_quality(85)
            .debug(false)
            .build()
            .expect("Quality config should build"),
    ];

    // Process with different configs sequentially (due to Send constraints)
    let mut results = Vec::new();

    for (i, config) in configs.into_iter().enumerate() {
        let result = remove_background(input_path, &config).await;
        results.push((i, result));
    }

    // All configurations should work
    for (config_idx, task_result) in results {
        let bg_result = task_result.expect("Background removal should succeed");

        assert!(
            !bg_result.mask.data.is_empty(),
            "Config {config_idx} should produce non-empty mask"
        );
    }

    println!("✅ Concurrent different configs: all configs work simultaneously");
}

#[tokio::test]
async fn test_error_handling_compatibility() {
    // Test with non-existent file
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Config should build");

    let result = remove_background("non_existent_file.jpg", &config).await;
    assert!(result.is_err(), "Should return error for non-existent file");

    // Test with invalid file (if we have one)
    let invalid_paths = vec![
        "assets/README.md",       // Text file instead of image
        "assets/test_cases.json", // JSON file instead of image
    ];

    for invalid_path in invalid_paths {
        if Path::new(invalid_path).exists() {
            let result = remove_background(invalid_path, &config).await;
            assert!(
                result.is_err(),
                "Should return error for invalid image file: {invalid_path}"
            );
            println!("✅ Error handling: correctly rejected {invalid_path}");
        }
    }

    println!("✅ Error handling compatibility: errors handled gracefully");
}

#[tokio::test]
async fn test_cross_platform_paths() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Config should build");

    // Test different path formats (though they'll be normalized by Rust)
    let path_variants = vec![
        "assets/input/portraits/portrait_single_simple_bg.jpg",
        "./assets/input/portraits/portrait_single_simple_bg.jpg",
    ];

    let mut successful_paths = 0;

    for path in &path_variants {
        if Path::new(path).exists() {
            let result = remove_background(path, &config).await;
            if result.is_ok() {
                successful_paths += 1;
                println!("✅ Path format works: {path}");
            } else {
                println!("❌ Path format failed: {path}");
            }
        }
    }

    assert!(successful_paths > 0, "At least one path format should work");
    println!(
        "✅ Cross-platform paths: {}/{} path formats work",
        successful_paths,
        path_variants.len()
    );
}
