//! Validate Rust outputs against JavaScript reference results

use bg_remove_core::{remove_background, RemovalConfig};
use serde_json;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct JavaScriptBenchmark {
    results: JavaScriptResults,
}

#[derive(Debug, serde::Deserialize)]
struct JavaScriptResults {
    portraits: JavaScriptCategory,
    products: JavaScriptCategory,
    complex: JavaScriptCategory,
    edge_cases: JavaScriptCategory,
}

#[derive(Debug, serde::Deserialize)]
struct JavaScriptCategory {
    processed: u32,
    #[allow(dead_code)] // May be used for detailed reporting
    failed: u32,
    avg_processing_time_ms: f64,
    images: std::collections::HashMap<String, JavaScriptImageResult>,
}

#[derive(Debug, serde::Deserialize)]
struct JavaScriptImageResult {
    success: bool,
    processing_time_ms: f64,
    #[allow(dead_code)] // Reserved for memory analysis
    memory_usage_mb: f64,
    #[allow(dead_code)] // Reserved for output validation
    outputs: JavaScriptOutputs,
}

#[derive(Debug, serde::Deserialize)]
struct JavaScriptOutputs {
    #[allow(dead_code)] // Reserved for format comparison
    png: String,
    #[allow(dead_code)] // Reserved for format comparison
    jpeg: String,
    #[allow(dead_code)] // Reserved for format comparison
    webp: String,
    #[allow(dead_code)] // Reserved for format comparison
    mask: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Validating Rust Outputs vs JavaScript Reference");
    println!("==================================================");

    // Load JavaScript benchmark data
    let js_benchmark_path =
        "crates/bg-remove-testing/assets/expected/benchmarks/javascript_baseline.json";
    if !Path::new(js_benchmark_path).exists() {
        println!(
            "❌ JavaScript benchmark file not found: {}",
            js_benchmark_path
        );
        return Ok(());
    }

    let js_data: JavaScriptBenchmark =
        serde_json::from_str(&std::fs::read_to_string(js_benchmark_path)?)?;

    println!("📊 JavaScript Baseline Summary:");
    println!(
        "   Portraits: {} images, avg {:.1}ms",
        js_data.results.portraits.processed, js_data.results.portraits.avg_processing_time_ms
    );
    println!(
        "   Products: {} images, avg {:.1}ms",
        js_data.results.products.processed, js_data.results.products.avg_processing_time_ms
    );
    println!(
        "   Complex: {} images, avg {:.1}ms",
        js_data.results.complex.processed, js_data.results.complex.avg_processing_time_ms
    );
    println!(
        "   Edge cases: {} images, avg {:.1}ms\n",
        js_data.results.edge_cases.processed, js_data.results.edge_cases.avg_processing_time_ms
    );

    let config = RemovalConfig::builder()
        .debug(false) // Use real ONNX with GPU
        .build()?;

    let mut total_tests = 0;
    let mut successful_tests = 0;
    let mut total_rust_time = 0.0;
    let mut total_js_time = 0.0;

    // Test each category
    let categories = [
        ("portraits", &js_data.results.portraits),
        ("products", &js_data.results.products),
        ("complex", &js_data.results.complex),
        ("edge_cases", &js_data.results.edge_cases),
    ];

    for (category_name, category_data) in &categories {
        println!("🧪 Testing {} images...", category_name);

        for (image_name, js_result) in &category_data.images {
            if !js_result.success {
                continue; // Skip failed JS tests
            }

            let input_path = format!(
                "crates/bg-remove-testing/assets/input/{}/{}",
                category_name, image_name
            );

            if !Path::new(&input_path).exists() {
                println!("   ⏭️  Skipping {}: Input file not found", image_name);
                continue;
            }

            print!("   📸 {}... ", image_name);
            total_tests += 1;

            let start = Instant::now();
            match remove_background(&input_path, &config).await {
                Ok(rust_result) => {
                    let rust_time_ms = start.elapsed().as_millis() as f64;
                    let js_time_ms = js_result.processing_time_ms;

                    // Save Rust output for comparison
                    let rust_output_path = format!(
                        "validation_output/rust_{}_{}.png",
                        category_name,
                        image_name.trim_end_matches(".jpg")
                    );

                    // Create output directory
                    std::fs::create_dir_all("validation_output").ok();

                    rust_result.save_png(&rust_output_path)?;

                    // Save mask for comparison
                    let rust_mask_path = format!(
                        "validation_output/rust_{}_{}_mask.png",
                        category_name,
                        image_name.trim_end_matches(".jpg")
                    );
                    rust_result.mask.save_png(&rust_mask_path)?;

                    // Calculate statistics
                    let rust_stats = rust_result.mask.statistics();
                    let dimensions = rust_result.dimensions();

                    println!(
                        "✅ {:.0}ms (JS: {:.0}ms) | {:.1}% fg | {}x{}",
                        rust_time_ms,
                        js_time_ms,
                        rust_stats.foreground_ratio * 100.0,
                        dimensions.0,
                        dimensions.1
                    );

                    successful_tests += 1;
                    total_rust_time += rust_time_ms;
                    total_js_time += js_time_ms;
                },
                Err(e) => {
                    println!("❌ Error: {}", e);
                },
            }
        }
        println!();
    }

    if successful_tests > 0 {
        let avg_rust_time = total_rust_time / successful_tests as f64;
        let avg_js_time = total_js_time / successful_tests as f64;
        let speedup = avg_js_time / avg_rust_time;

        println!("📊 Validation Summary:");
        println!(
            "   ✅ Successful tests: {}/{}",
            successful_tests, total_tests
        );
        println!("   ⏱️  Average Rust time: {:.0}ms", avg_rust_time);
        println!("   ⏱️  Average JavaScript time: {:.0}ms", avg_js_time);
        println!("   🚀 Rust speedup: {:.1}x", speedup);

        if speedup > 1.5 {
            println!("   🎯 Excellent performance advantage!");
        }

        println!("\n📁 Outputs saved to validation_output/ directory");
        println!("   Compare rust_*_mask.png with JS reference masks");
        println!("   Compare rust_*.png with JS reference outputs");
    }

    Ok(())
}
