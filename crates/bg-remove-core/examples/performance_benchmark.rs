//! Performance benchmarking for background removal

use bg_remove_core::{RemovalConfig, remove_background};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Background Removal Performance Benchmark");
    println!("============================================");
    
    let test_images = [
        ("Portrait Simple", "crates/bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg"),
        ("Portrait Multiple", "crates/bg-remove-testing/assets/input/portraits/portrait_multiple_people.jpg"),
        ("Product White BG", "crates/bg-remove-testing/assets/input/products/product_clothing_white_bg.jpg"),
        ("Complex Group", "crates/bg-remove-testing/assets/input/complex/complex_group_photo.jpg"),
    ];
    
    let config = RemovalConfig::builder()
        .debug(false) // Use ONNX with GPU acceleration
        .build()?;

    let mut total_time = 0.0;
    let mut successful_tests = 0;

    for (name, path) in &test_images {
        if !std::path::Path::new(path).exists() {
            println!("⏭️  Skipping {}: File not found", name);
            continue;
        }

        print!("🧪 Testing {}... ", name);
        
        let start = Instant::now();
        match remove_background(path, &config).await {
            Ok(result) => {
                let processing_time = start.elapsed().as_secs_f64();
                let dimensions = result.dimensions();
                let total_pixels = (dimensions.0 as f64) * (dimensions.1 as f64);
                let megapixels = total_pixels / 1_000_000.0;
                
                println!("✅ {:.2}s ({:.1}MP, {:.1}MP/s)", 
                    processing_time, 
                    megapixels,
                    megapixels / processing_time
                );
                
                total_time += processing_time;
                successful_tests += 1;
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }
    
    if successful_tests > 0 {
        let average_time = total_time / successful_tests as f64;
        println!("\n📊 Performance Summary:");
        println!("   Tests completed: {}", successful_tests);
        println!("   Average time: {:.2}s", average_time);
        println!("   Total time: {:.2}s", total_time);
        
        // Compare with JavaScript baseline
        let js_baseline = 2.83; // seconds from benchmark data
        let improvement = ((js_baseline - average_time) / js_baseline) * 100.0;
        
        println!("\n🏆 vs JavaScript Baseline:");
        println!("   JavaScript: {:.2}s", js_baseline);
        println!("   Rust (GPU): {:.2}s", average_time);
        if improvement > 0.0 {
            println!("   🚀 Rust is {:.1}% faster!", improvement);
        } else {
            println!("   📈 Rust is {:.1}% slower", -improvement);
        }
    }
    
    Ok(())
}