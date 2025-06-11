//! Test which execution provider is being used

use bg_remove_core::{remove_background, RemovalConfig};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: GPU vs CPU comparison (debug mode uses mock backend)

    println!("🔍 Execution Provider Diagnostic Test");
    println!("====================================");

    let test_image =
        "../../crates/bg-remove-testing/assets/input/portraits/portrait_action_motion.jpg";

    if !std::path::Path::new(test_image).exists() {
        println!("❌ Test image not found: {}", test_image);
        return Ok(());
    }

    // Test with GPU acceleration
    println!("🧪 Testing with GPU acceleration enabled...");
    let config = RemovalConfig::builder()
        .debug(false) // Use ONNX with GPU acceleration
        .build()?;

    let start = Instant::now();
    let _result = remove_background(test_image, &config).await?;
    let gpu_time = start.elapsed().as_secs_f64();

    println!("   ⏱️  GPU-enabled time: {:.2}s", gpu_time);

    // Test with CPU only (debug mode)
    println!("\n🧪 Testing with CPU-only (debug mode)...");
    let config_cpu = RemovalConfig::builder()
        .debug(true) // Use mock backend for comparison
        .build()?;

    let start = Instant::now();
    let _result_cpu = remove_background(test_image, &config_cpu).await?;
    let cpu_time = start.elapsed().as_secs_f64();

    println!("   ⏱️  CPU-only time: {:.2}s", cpu_time);

    // Comparison
    let speedup = cpu_time / gpu_time;
    println!("\n🏆 Performance Comparison:");
    println!("   GPU-enabled: {:.2}s", gpu_time);
    println!("   CPU-only: {:.2}s", cpu_time);
    println!("   Speedup: {:.1}x", speedup);

    if speedup > 1.5 {
        println!("   ✅ GPU acceleration is working effectively!");
        if speedup > 3.0 {
            println!("   🚀 Excellent GPU performance!");
        }
    } else if speedup > 1.1 {
        println!("   📈 Some acceleration detected");
    } else {
        println!("   ⚠️  GPU acceleration may not be working");
    }

    Ok(())
}
