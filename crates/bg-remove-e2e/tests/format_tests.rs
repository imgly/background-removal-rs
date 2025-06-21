//! Format compatibility integration tests
//!
//! These tests validate that the background removal works correctly with different
//! input and output formats (JPEG, PNG, WebP).

use bg_remove_core::{remove_background_with_backend, RemovalConfig};
use bg_remove_onnx::OnnxBackend;
use bg_remove_core::models::ModelManager;
use image::GenericImageView;
use std::path::Path;

/// Helper function to create a backend for testing
async fn create_test_backend(_config: &RemovalConfig) -> Result<Box<dyn bg_remove_core::inference::InferenceBackend>, bg_remove_core::error::BgRemovalError> {
    let model_manager = ModelManager::with_embedded_model("isnet-fp32".to_string())?;
    let backend = OnnxBackend::with_model_manager(model_manager);
    Ok(Box::new(backend))
}

#[tokio::test]
async fn test_jpeg_input() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let jpeg_inputs = vec![
        "assets/input/portraits/portrait_single_simple_bg.jpg",
        "assets/input/products/product_clothing_white_bg.jpg",
    ];

    for input_path in jpeg_inputs {
        if !Path::new(input_path).exists() {
            println!("⏭️  Skipping JPEG test: {input_path} not found");
            continue;
        }

        let backend = create_test_backend(&config).await.expect("Failed to create backend");
        let result = remove_background_with_backend(input_path, &config, backend).await;
        assert!(result.is_ok(), "Should process JPEG input: {input_path}");

        let result = result.unwrap();

        // Verify output has alpha channel (RGBA)
        assert_eq!(
            result.image.color().channel_count(),
            4,
            "Output should have alpha channel for JPEG input"
        );

        println!("✅ JPEG input test passed: {input_path}");
    }
}

#[tokio::test]
async fn test_png_input() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    // Look for PNG inputs in expected directory (which contain PNG files)
    let png_inputs = vec![
        "assets/expected/portraits/portrait_single_simple_bg.png",
        "assets/expected/products/product_clothing_white_bg.png",
    ];

    for input_path in png_inputs {
        if !Path::new(input_path).exists() {
            println!("⏭️  Skipping PNG test: {input_path} not found");
            continue;
        }

        let backend = create_test_backend(&config).await.expect("Failed to create backend");
        let result = remove_background_with_backend(input_path, &config, backend).await;
        assert!(result.is_ok(), "Should process PNG input: {input_path}");

        let result = result.unwrap();

        // Verify output maintains alpha channel
        assert_eq!(
            result.image.color().channel_count(),
            4,
            "Output should maintain alpha channel for PNG input"
        );

        println!("✅ PNG input test passed: {input_path}");
    }
}

#[tokio::test]
async fn test_output_format_consistency() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let input_path = "assets/input/portraits/portrait_single_simple_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping format consistency test: test file not found");
        return;
    }

    let backend = create_test_backend(&config).await.expect("Failed to create backend");
    let result = remove_background_with_backend(input_path, &config, backend).await;
    assert!(result.is_ok(), "Background removal should succeed");

    let result = result.unwrap();

    // Test that output image has consistent properties
    assert!(result.image.width() > 0, "Output should have valid width");
    assert!(result.image.height() > 0, "Output should have valid height");
    assert_eq!(
        result.image.color().channel_count(),
        4,
        "Output should be RGBA"
    );

    // Test that mask has correct format
    assert!(!result.mask.data.is_empty(), "Mask should not be empty");
    assert_eq!(
        result.mask.dimensions.0 * result.mask.dimensions.1,
        result.mask.data.len() as u32,
        "Mask dimensions should match data length"
    );

    println!("✅ Output format consistency test passed");
}

#[tokio::test]
async fn test_image_dimensions_preservation() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let test_cases = vec![
        "assets/input/portraits/portrait_single_simple_bg.jpg",
        "assets/input/products/product_clothing_white_bg.jpg",
    ];

    for input_path in test_cases {
        if !Path::new(input_path).exists() {
            println!("⏭️  Skipping dimension test: {input_path} not found");
            continue;
        }

        // Load original image to get dimensions
        let original_image = image::open(input_path).expect("Should load original image");
        let (orig_width, orig_height) = original_image.dimensions();

        let backend = create_test_backend(&config).await.expect("Failed to create backend");
        let result = remove_background_with_backend(input_path, &config, backend).await;
        assert!(
            result.is_ok(),
            "Background removal should succeed for {input_path}"
        );

        let result = result.unwrap();
        let (out_width, out_height) = result.image.dimensions();

        // Verify dimensions are preserved
        assert_eq!(
            orig_width, out_width,
            "Output width should match input for {input_path}"
        );
        assert_eq!(
            orig_height, out_height,
            "Output height should match input for {input_path}"
        );

        // Verify mask dimensions match
        assert_eq!(
            result.mask.dimensions.0, out_width,
            "Mask width should match output for {input_path}"
        );
        assert_eq!(
            result.mask.dimensions.1, out_height,
            "Mask height should match output for {input_path}"
        );

        println!("✅ Dimension preservation test passed: {input_path} ({out_width}x{out_height})");
    }
}

#[tokio::test]
async fn test_different_aspect_ratios() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let test_cases = vec![
        (
            "square-ish",
            "assets/input/portraits/portrait_single_simple_bg.jpg",
        ),
        (
            "wide",
            "assets/input/products/product_clothing_white_bg.jpg",
        ),
        ("tall", "assets/input/portraits/portrait_action_motion.jpg"),
    ];

    for (aspect_name, input_path) in test_cases {
        if !Path::new(input_path).exists() {
            println!("⏭️  Skipping aspect ratio test: {input_path} not found");
            continue;
        }

        let backend = create_test_backend(&config).await.expect("Failed to create backend");
        let result = remove_background_with_backend(input_path, &config, backend).await;
        assert!(
            result.is_ok(),
            "Should handle {aspect_name} aspect ratio: {input_path}"
        );

        let result = result.unwrap();
        let (width, height) = result.image.dimensions();

        // Basic sanity checks
        assert!(
            width > 0 && height > 0,
            "Output should have valid dimensions"
        );
        assert!(
            width <= 4096 && height <= 4096,
            "Output dimensions should be reasonable"
        );

        println!("✅ Aspect ratio test passed: {aspect_name} ({width}x{height}) - {input_path}");
    }
}

#[tokio::test]
async fn test_image_quality_preservation() {
    let config = RemovalConfig::builder()
        .debug(false)
        .build()
        .expect("Failed to create config");

    let input_path = "assets/input/portraits/portrait_single_simple_bg.jpg";

    if !Path::new(input_path).exists() {
        println!("⏭️  Skipping quality test: test file not found");
        return;
    }

    let backend = create_test_backend(&config).await.expect("Failed to create backend");
    let result = remove_background_with_backend(input_path, &config, backend).await;
    assert!(result.is_ok(), "Background removal should succeed");

    let result = result.unwrap();

    // Test that output image maintains reasonable quality
    let (width, height) = result.image.dimensions();
    let pixel_count = (width * height) as usize;

    // Convert to RGBA for analysis
    let rgba_image = result.image.to_rgba8();
    let pixels = rgba_image.as_raw();

    // Count non-transparent pixels (alpha > 0)
    let non_transparent_pixels = pixels.chunks(4).filter(|pixel| pixel[3] > 0).count();

    // Should have some foreground content (not completely transparent)
    let foreground_ratio = non_transparent_pixels as f64 / pixel_count as f64;
    assert!(
        foreground_ratio > 0.01,
        "Should have some foreground content, got {:.1}%",
        foreground_ratio * 100.0
    );
    assert!(
        foreground_ratio < 0.99,
        "Should have removed some background, got {:.1}% foreground",
        foreground_ratio * 100.0
    );

    println!(
        "✅ Quality preservation test passed: {:.1}% foreground retained",
        foreground_ratio * 100.0
    );
}
