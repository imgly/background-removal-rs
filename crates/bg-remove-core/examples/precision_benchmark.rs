//! Compare FP32 vs FP16 model performance

use bg_remove_core::{RemovalConfig, remove_background};
use bg_remove_core::config::ModelPrecision;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Model Precision Performance Comparison");
    println!("=========================================");
    
    let test_image = "../../tests/assets/input/portraits/portrait_single_simple_bg.jpg";
    
    if !std::path::Path::new(test_image).exists() {
        println!("❌ Test image not found: {}", test_image);
        return Ok(());
    }

    // Test FP16 (default)
    println!("🧪 Testing FP16 model...");
    let config_fp16 = RemovalConfig::builder()
        .model_precision(ModelPrecision::Fp16)
        .debug(false)
        .build()?;
    
    let start = Instant::now();
    let result_fp16 = remove_background(test_image, &config_fp16).await?;
    let fp16_time = start.elapsed().as_secs_f64();
    let fp16_stats = result_fp16.mask.statistics();
    
    println!("   ⏱️  Time: {:.2}s", fp16_time);
    println!("   📊 Foreground: {:.1}%", fp16_stats.foreground_ratio * 100.0);
    
    // Test FP32 (higher precision)
    println!("\n🧪 Testing FP32 model...");
    let config_fp32 = RemovalConfig::builder()
        .model_precision(ModelPrecision::Fp32)
        .debug(false)
        .build()?;
    
    let start = Instant::now();
    let result_fp32 = remove_background(test_image, &config_fp32).await?;
    let fp32_time = start.elapsed().as_secs_f64();
    let fp32_stats = result_fp32.mask.statistics();
    
    println!("   ⏱️  Time: {:.2}s", fp32_time);
    println!("   📊 Foreground: {:.1}%", fp32_stats.foreground_ratio * 100.0);
    
    // Save outputs for comparison
    result_fp16.save_png("output_fp16.png")?;
    result_fp32.save_png("output_fp32.png")?;
    
    // Comparison
    println!("\n🏆 Comparison:");
    println!("   FP16: {:.2}s ({:.0}MB model)", fp16_time, 84.0);
    println!("   FP32: {:.2}s ({:.0}MB model)", fp32_time, 167.0);
    
    let speed_diff = ((fp32_time - fp16_time) / fp16_time) * 100.0;
    if speed_diff > 0.0 {
        println!("   📈 FP32 is {:.1}% slower than FP16", speed_diff);
    } else {
        println!("   📈 FP32 is {:.1}% faster than FP16", -speed_diff);
    }
    
    let quality_diff = ((fp32_stats.foreground_ratio - fp16_stats.foreground_ratio) / fp16_stats.foreground_ratio) * 100.0;
    println!("   🎯 Segmentation difference: {:.1}%", quality_diff.abs());
    
    println!("\n💾 Outputs saved: output_fp16.png, output_fp32.png");
    
    Ok(())
}