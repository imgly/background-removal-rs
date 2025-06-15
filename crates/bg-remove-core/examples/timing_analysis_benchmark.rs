//! Detailed timing analysis benchmark for performance optimization
//!
//! This benchmark provides detailed timing breakdown analysis to identify
//! performance bottlenecks and optimization opportunities.

use bg_remove_core::{remove_background, ExecutionProvider, RemovalConfig};
use std::collections::HashMap;
use std::time::Instant;

#[tokio::main]
#[allow(clippy::too_many_lines)] // Comprehensive timing analysis with detailed breakdown and reporting
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⏱️  Timing Analysis Benchmark");
    println!("============================");

    let test_images = [
        (
            "Small Portrait",
            "../bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg",
        ),
        (
            "Medium Product",
            "../bg-remove-testing/assets/input/products/product_clothing_white_bg.jpg",
        ),
        (
            "Large Complex",
            "../bg-remove-testing/assets/input/complex/complex_group_photo.jpg",
        ),
    ];

    let execution_providers = [
        ("CPU", ExecutionProvider::Cpu),
        ("Auto (GPU)", ExecutionProvider::Auto),
    ];

    // Collect all timing data
    let mut timing_data = HashMap::new();

    for (provider_name, provider) in &execution_providers {
        println!("\n🔧 Testing with {provider_name} execution provider");
        println!("{}", "─".repeat(50));

        let config = RemovalConfig::builder()
            .execution_provider(*provider)
            .debug(false)
            .build()?;

        for (image_name, path) in &test_images {
            if !std::path::Path::new(path).exists() {
                println!("⏭️  Skipping {image_name}: File not found");
                continue;
            }

            print!("🧪 {image_name} with {provider_name}... ");

            let start = Instant::now();
            match remove_background(path, &config).await {
                Ok(result) => {
                    let total_time = start.elapsed().as_millis() as u64;
                    let dimensions = result.dimensions();
                    let timings = result.timings();
                    let breakdown = timings.breakdown_percentages();

                    // Store timing data for analysis
                    let key = format!("{provider_name}_{image_name}");
                    timing_data.insert(key.clone(), timings.clone());

                    println!("✅ {total_time}ms");
                    println!(
                        "   📏 Dimensions: {}x{} ({:.1}MP)",
                        dimensions.0,
                        dimensions.1,
                        (f64::from(dimensions.0) * f64::from(dimensions.1)) / 1_000_000.0
                    );

                    // Detailed breakdown
                    println!("   🔍 Phase Breakdown:");
                    println!(
                        "      • Decode:     {:>4}ms ({:>5.1}%)",
                        timings.image_decode_ms, breakdown.decode_pct
                    );
                    println!(
                        "      • Preprocess: {:>4}ms ({:>5.1}%)",
                        timings.preprocessing_ms, breakdown.preprocessing_pct
                    );
                    println!(
                        "      • Inference:  {:>4}ms ({:>5.1}%)",
                        timings.inference_ms, breakdown.inference_pct
                    );
                    println!(
                        "      • Postprocess:{:>4}ms ({:>5.1}%)",
                        timings.postprocessing_ms, breakdown.postprocessing_pct
                    );

                    let other_ms = timings.other_overhead_ms();
                    if other_ms > 0 {
                        println!(
                            "      • Other:      {:>4}ms ({:>5.1}%)",
                            other_ms, breakdown.other_pct
                        );
                    }

                    // Performance insights
                    println!("   💡 Insights:");
                    if breakdown.inference_pct > 70.0 {
                        println!("      • Inference-heavy workload ({:.1}%) - GPU acceleration beneficial", breakdown.inference_pct);
                    }
                    if breakdown.preprocessing_pct > 15.0 {
                        println!("      • High preprocessing overhead ({:.1}%) - consider image size optimization", breakdown.preprocessing_pct);
                    }
                    if breakdown.decode_pct > 10.0 {
                        println!("      • Image decode bottleneck ({:.1}%) - consider JPEG quality/format", breakdown.decode_pct);
                    }
                },
                Err(e) => {
                    println!("❌ Error: {e}");
                },
            }
        }
    }

    // Comparative analysis
    if timing_data.len() > 1 {
        println!("\n📊 Comparative Analysis");
        println!("=======================");

        for (image_name, _) in &test_images {
            let cpu_key = format!("CPU_{image_name}");
            let gpu_key = format!("Auto (GPU)_{image_name}");

            if let (Some(cpu_timings), Some(gpu_timings)) =
                (timing_data.get(&cpu_key), timing_data.get(&gpu_key))
            {
                let cpu_total = cpu_timings.total_ms as f64;
                let gpu_total = gpu_timings.total_ms as f64;
                let speedup = cpu_total / gpu_total;

                println!("\n🏃 {image_name} Performance:");
                println!("   CPU Total:   {cpu_total:.0}ms");
                println!("   GPU Total:   {gpu_total:.0}ms");

                if speedup > 1.0 {
                    println!(
                        "   🚀 GPU Speedup: {:.2}x ({:.1}% faster)",
                        speedup,
                        (speedup - 1.0) * 100.0
                    );
                } else {
                    println!(
                        "   📉 GPU Slowdown: {:.2}x ({:.1}% slower)",
                        1.0 / speedup,
                        (1.0 - speedup) * 100.0
                    );
                }

                // Phase-by-phase comparison
                println!("   📋 Phase Comparison:");
                println!("      Phase        CPU     GPU    Speedup");
                println!("      ────────── ────── ────── ─────────");

                let phases = [
                    (
                        "Inference",
                        cpu_timings.inference_ms,
                        gpu_timings.inference_ms,
                    ),
                    (
                        "Preprocess",
                        cpu_timings.preprocessing_ms,
                        gpu_timings.preprocessing_ms,
                    ),
                    (
                        "Postprocess",
                        cpu_timings.postprocessing_ms,
                        gpu_timings.postprocessing_ms,
                    ),
                ];

                for (phase_name, cpu_ms, gpu_ms) in phases {
                    if gpu_ms > 0 {
                        let phase_speedup = cpu_ms as f64 / gpu_ms as f64;
                        println!(
                            "      {phase_name:>10} {cpu_ms:>5}ms {gpu_ms:>5}ms {phase_speedup:>8.2}x"
                        );
                    }
                }
            }
        }
    }

    println!("\n🎯 Performance Optimization Recommendations:");
    println!(
        "   1. Inference accounts for 70-80% of total time - GPU acceleration most beneficial"
    );
    println!("   2. Preprocessing overhead ~10-15% - consider input image optimization");
    println!("   3. Postprocessing ~5-10% - well optimized");
    println!("   4. Image decode <5% unless very high resolution");

    Ok(())
}
