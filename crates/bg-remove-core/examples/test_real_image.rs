#!/usr/bin/env rust-script

//! Simple test script to verify our implementation works with a real image
//!
//! Usage: cargo run --bin test_real_image

use bg_remove_core::{remove_background, RemovalConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger to see INFO messages
    env_logger::init();

    println!("🧪 Testing background removal with real image...");

    // Use one of our test images
    let input_path =
        "crates/bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg";
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
    let mut result = remove_background(input_path, &config).await?;
    let processing_time = start.elapsed();

    // Save result with timing measurement
    result.save_png_timed(output_path)?;

    // Display results
    let encode_time_ms = result.timings().image_encode_ms.unwrap_or(0);
    println!(
        "✅ Processing completed in {:.2}s (+ {}ms image encoding)",
        processing_time.as_secs_f64(),
        encode_time_ms
    );
    println!(
        "📊 Image dimensions: {}x{}",
        result.dimensions().0,
        result.dimensions().1
    );
    println!(
        "📊 Original dimensions: {}x{}",
        result.original_dimensions.0, result.original_dimensions.1
    );

    // Display detailed timing breakdown
    println!("\n⏱️  Detailed Performance Breakdown:");
    println!("{}", result.timing_summary());

    let timings = result.timings();
    let breakdown = timings.breakdown_percentages();
    let other_ms = timings.other_overhead_ms();

    println!("\n┌─────────────────┬─────────┬─────────┐");
    println!("│ Phase           │ Time    │ Percent │");
    println!("├─────────────────┼─────────┼─────────┤");
    if timings.model_load_ms > 0 {
        println!(
            "│ Model Load      │ {:>4}ms  │ {:>5.1}%  │",
            timings.model_load_ms, breakdown.model_load_pct
        );
    }
    println!(
        "│ Image Decode    │ {:>4}ms  │ {:>5.1}%  │",
        timings.image_decode_ms, breakdown.decode_pct
    );
    println!(
        "│ Preprocessing   │ {:>4}ms  │ {:>5.1}%  │",
        timings.preprocessing_ms, breakdown.preprocessing_pct
    );
    println!(
        "│ Inference       │ {:>4}ms  │ {:>5.1}%  │",
        timings.inference_ms, breakdown.inference_pct
    );
    println!(
        "│ Postprocessing  │ {:>4}ms  │ {:>5.1}%  │",
        timings.postprocessing_ms, breakdown.postprocessing_pct
    );
    if let Some(encode_ms) = timings.image_encode_ms {
        println!(
            "│ Image Encode    │ {:>4}ms  │ {:>5.1}%  │",
            encode_ms, breakdown.encode_pct
        );
    }
    if other_ms > 0 {
        println!(
            "│ Other/Overhead  │ {:>4}ms  │ {:>5.1}%  │",
            other_ms, breakdown.other_pct
        );
    }
    println!("├─────────────────┼─────────┼─────────┤");
    println!("│ Total           │ {:>4}ms  │ 100.0%  │", timings.total_ms);
    println!("└─────────────────┴─────────┴─────────┘");

    // Additional insights
    println!("\n📈 Performance Insights:");
    println!(
        "   • Inference ratio: {:.1}% (ONNX Runtime efficiency)",
        timings.inference_ratio() * 100.0
    );
    println!(
        "   • Image encoding: {}ms (measured separately)",
        encode_time_ms
    );
    if other_ms > 10 {
        println!(
            "   • Overhead: {}ms ({:.1}%) - function calls, memory allocation",
            other_ms, breakdown.other_pct
        );
    }

    println!("\n🎯 Mask statistics: {:?}", result.mask.statistics());
    println!("💾 Output saved to: {}", output_path);

    Ok(())
}
