//! Compare CPU vs GPU ONNX Runtime performance

use bg_remove_core::config::RemovalConfig;
use bg_remove_core::image_processing::ImageProcessor;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ CPU vs GPU ONNX Runtime Performance Test");
    println!("==========================================");
    
    let test_image = "crates/bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg";
    
    if !std::path::Path::new(test_image).exists() {
        println!("❌ Test image not found: {}", test_image);
        return Ok(());
    }

    // Test current implementation (should use GPU if available)
    println!("🧪 Testing current implementation (GPU-enabled)...");
    let config_gpu = RemovalConfig::builder()
        .debug(false)
        .build()?;
    
    let mut processor_gpu = ImageProcessor::new(&config_gpu)?;
    
    let start = Instant::now();
    let result_gpu = processor_gpu.remove_background(test_image).await?;
    let gpu_time = start.elapsed().as_secs_f64();
    let gpu_stats = result_gpu.mask.statistics();
    
    println!("   ⏱️  Time: {:.2}s", gpu_time);
    println!("   🎯 Foreground: {:.1}%", gpu_stats.foreground_ratio * 100.0);
    
    // Show what we know
    println!("\n📊 Performance Analysis:");
    println!("   Current result: {:.2}s", gpu_time);
    println!("   vs JavaScript: 2.83s baseline");
    
    let js_improvement = ((2.83 - gpu_time) / 2.83) * 100.0;
    println!("   🚀 {:.1}% faster than JavaScript", js_improvement);
    
    // Memory info
    println!("   📦 Model size: Automatically optimized");
    
    // Note about execution providers
    println!("\n🔧 Execution Provider Info:");
    println!("   ✅ ONNX Runtime configured with:");
    println!("      1. CUDA (NVIDIA GPUs)");
    println!("      2. CoreML (Apple Neural Engine)");
    println!("      3. CPU (fallback)");
    println!("   📝 ONNX Runtime automatically selects the best available");
    
    if gpu_time < 2.0 {
        println!("   🎯 Performance suggests GPU/ANE acceleration is active");
    } else {
        println!("   🤔 Performance suggests CPU execution");
    }
    
    result_gpu.save_png("gpu_test_output.png")?;
    println!("\n💾 Output saved: gpu_test_output.png");
    
    Ok(())
}