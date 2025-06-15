//! Performance benchmarking for background removal

use bg_remove_core::{remove_background, RemovalConfig};
use std::time::Instant;

#[tokio::main]
#[allow(clippy::too_many_lines)] // Performance benchmarking with comprehensive timing analysis
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Background Removal Performance Benchmark");
    println!("============================================");

    let test_images = [
        (
            "Portrait Simple",
            "../bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg",
        ),
        (
            "Portrait Multiple",
            "../bg-remove-testing/assets/input/portraits/portrait_multiple_people.jpg",
        ),
        (
            "Product White BG",
            "../bg-remove-testing/assets/input/products/product_clothing_white_bg.jpg",
        ),
        (
            "Complex Group",
            "../bg-remove-testing/assets/input/complex/complex_group_photo.jpg",
        ),
    ];

    let config = RemovalConfig::builder()
        .debug(false) // Use ONNX with GPU acceleration
        .build()?;

    let mut total_time = 0.0;
    let mut successful_tests = 0;

    for (name, path) in &test_images {
        if !std::path::Path::new(path).exists() {
            println!("⏭️  Skipping {name}: File not found");
            continue;
        }

        print!("🧪 Testing {name}... ");

        let start = Instant::now();
        match remove_background(path, &config).await {
            Ok(result) => {
                let processing_time = start.elapsed().as_secs_f64();
                let dimensions = result.dimensions();
                let total_pixels = f64::from(dimensions.0) * f64::from(dimensions.1);
                let megapixels = total_pixels / 1_000_000.0;

                // Get detailed timing breakdown
                let timings = result.timings();
                let breakdown = timings.breakdown_percentages();

                println!(
                    "✅ {:.2}s ({:.1}MP, {:.1}MP/s)",
                    processing_time,
                    megapixels,
                    megapixels / processing_time
                );

                // Show detailed timing breakdown
                println!("   🔍 Timing Breakdown:");
                println!(
                    "      • Decode: {}ms ({:.1}%)",
                    timings.image_decode_ms, breakdown.decode_pct
                );
                println!(
                    "      • Preprocess: {}ms ({:.1}%)",
                    timings.preprocessing_ms, breakdown.preprocessing_pct
                );
                println!(
                    "      • Inference: {}ms ({:.1}%)",
                    timings.inference_ms, breakdown.inference_pct
                );
                println!(
                    "      • Postprocess: {}ms ({:.1}%)",
                    timings.postprocessing_ms, breakdown.postprocessing_pct
                );
                if let Some(encode_ms) = timings.image_encode_ms {
                    println!(
                        "      • Encode: {}ms ({:.1}%)",
                        encode_ms, breakdown.encode_pct
                    );
                }
                let other_ms = timings.other_overhead_ms();
                if other_ms > 0 {
                    println!(
                        "      • Other: {}ms ({:.1}%)",
                        other_ms, breakdown.other_pct
                    );
                }

                total_time += processing_time;
                successful_tests += 1;
            },
            Err(e) => {
                println!("❌ Error: {e}");
            },
        }
    }

    if successful_tests > 0 {
        let average_time = total_time / f64::from(successful_tests);
        println!("\n📊 Performance Summary:");
        println!("   Tests completed: {successful_tests}");
        println!("   Average time: {average_time:.2}s");
        println!("   Total time: {total_time:.2}s");

        // Compare with JavaScript baseline
        let js_baseline = 2.83; // seconds from benchmark data
        let improvement = ((js_baseline - average_time) / js_baseline) * 100.0;

        println!("\n🏆 vs JavaScript Baseline:");
        println!("   JavaScript: {js_baseline:.2}s");
        println!("   Rust (GPU): {average_time:.2}s");
        if improvement > 0.0 {
            println!("   🚀 Rust is {improvement:.1}% faster!");
        } else {
            println!("   📈 Rust is {:.1}% slower", -improvement);
        }
    }

    Ok(())
}
