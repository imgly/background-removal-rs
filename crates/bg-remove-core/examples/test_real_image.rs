#!/usr/bin/env rust-script

//! Simple test script to verify our implementation works with a real image
//! 
//! Usage: cargo run --bin test_real_image

use bg_remove_core::{RemovalConfig, remove_background};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing background removal with real image...");

    // Use one of our test images
    let input_path = "tests/assets/input/portraits/portrait_single_simple_bg.jpg";
    let output_path = "test_output.png";

    if !Path::new(input_path).exists() {
        println!("❌ Test image not found: {}", input_path);
        println!("Make sure you're running from the project root directory");
        return Ok(());
    }

    // Create configuration
    let config = RemovalConfig::builder()
        .debug(false) // Use ONNX backend (though it's currently mock)
        .build()?;

    println!("📁 Input: {}", input_path);
    println!("📁 Output: {}", output_path);
    println!("⚙️  Model: Auto-optimized precision");

    // Process the image
    let start = std::time::Instant::now();
    let result = remove_background(input_path, &config).await?;
    let processing_time = start.elapsed();

    // Save result
    result.save_png(output_path)?;

    // Display results
    println!("✅ Processing completed in {:.2}s", processing_time.as_secs_f64());
    println!("📊 Image dimensions: {}x{}", result.dimensions().0, result.dimensions().1);
    println!("📊 Original dimensions: {}x{}", result.original_dimensions.0, result.original_dimensions.1);
    println!("🎯 Mask statistics: {:?}", result.mask.statistics());
    println!("💾 Output saved to: {}", output_path);

    Ok(())
}