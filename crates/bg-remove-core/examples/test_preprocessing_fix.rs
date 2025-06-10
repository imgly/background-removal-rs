//! Test the preprocessing fix for JavaScript compatibility

use bg_remove_core::{RemovalConfig, remove_background};
use bg_remove_core::config::ModelPrecision;
use std::path::Path;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing Preprocessing Fix for JavaScript Compatibility");
    println!("=========================================================");
    
    let test_image = "../../tests/assets/input/portraits/portrait_single_simple_bg.jpg";
    
    if !Path::new(test_image).exists() {
        println!("❌ Test image not found: {}", test_image);
        return Ok(());
    }

    let config = RemovalConfig::builder()
        .model_precision(ModelPrecision::Fp16)
        .debug(false)
        .build()?;

    println!("🧪 Processing with improved preprocessing...");
    let start_time = Instant::now();
    let result = remove_background(test_image, &config).await?;
    let processing_time = start_time.elapsed();
    
    // Analyze the mask values
    let mask_data = &result.mask.data;
    let mut value_histogram = std::collections::HashMap::new();
    for &value in mask_data {
        *value_histogram.entry(value).or_insert(0) += 1;
    }
    
    println!("⏱️  Processing time: {}ms", processing_time.as_millis());
    println!("📊 Mask analysis:");
    
    let min_val = *mask_data.iter().min().unwrap();
    let max_val = *mask_data.iter().max().unwrap();
    println!("   Range: {} to {}", min_val, max_val);
    
    // Show most common values
    let mut by_count: Vec<_> = value_histogram.iter().collect();
    by_count.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
    println!("   Most common values:");
    for (value, count) in by_count.iter().take(5) {
        let percentage = (**count as f64) / (mask_data.len() as f64) * 100.0;
        println!("     Value {}: {:.1}%", value, percentage);
    }
    
    // Check transparency distribution
    let transparent_pixels = mask_data.iter().filter(|&&x| x < 50).count();
    let opaque_pixels = mask_data.iter().filter(|&&x| x > 200).count();
    let partial_pixels = mask_data.len() - transparent_pixels - opaque_pixels;
    
    println!("   Transparency distribution:");
    println!("     Transparent (0-49): {:.1}%", (transparent_pixels as f64) / (mask_data.len() as f64) * 100.0);
    println!("     Opaque (200-255): {:.1}%", (opaque_pixels as f64) / (mask_data.len() as f64) * 100.0);
    println!("     Partial (50-199): {:.1}%", (partial_pixels as f64) / (mask_data.len() as f64) * 100.0);
    
    // Get foreground statistics
    let stats = result.mask.statistics();
    println!("   Foreground ratio: {:.1}%", stats.foreground_ratio * 100.0);
    
    // Save the result
    result.save_png("preprocessing_fix_test.png")?;
    println!("💾 Saved: preprocessing_fix_test.png");
    
    println!("\n🔍 Key improvements with JavaScript-compatible preprocessing:");
    println!("   • Fixed normalization: Mean=[128,128,128], Std=[256,256,256]");
    println!("   • Fixed input size: 1024x1024 (no aspect preservation)");
    println!("   • Direct tensor value to alpha conversion");
    println!("   • Should now match JavaScript quality!");

    Ok(())
}