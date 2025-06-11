//! Comprehensive FP32 vs FP16 model precision benchmark
//!
//! This benchmark compares performance, accuracy, and resource usage
//! between FP32 and FP16 model precision to inform default build decisions.

use bg_remove_core::{remove_background, ExecutionProvider, RemovalConfig};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
struct BenchmarkResults {
    #[allow(dead_code)] // Used in analysis output
    precision: String,
    avg_processing_time_ms: f64,
    avg_inference_time_ms: f64,
    #[allow(dead_code)] // Reserved for detailed timing analysis
    avg_total_time_ms: f64,
    inference_ratio: f64,
    model_size_mb: f64,
    memory_efficiency: String,
    #[allow(dead_code)] // Reserved for statistical analysis
    processed_images: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏆 FP32 vs FP16 Model Precision Benchmark");
    println!("==========================================");

    // Test images representing different scenarios
    let test_images = [
        (
            "Portrait",
            "crates/bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg",
        ),
        (
            "Product",
            "crates/bg-remove-testing/assets/input/products/product_clothing_white_bg.jpg",
        ),
        (
            "Complex",
            "crates/bg-remove-testing/assets/input/complex/complex_group_photo.jpg",
        ),
        (
            "Multiple People",
            "crates/bg-remove-testing/assets/input/portraits/portrait_multiple_people.jpg",
        ),
    ];

    // Filter to only existing images
    let available_images: Vec<(&str, &str)> = test_images
        .iter()
        .filter(|(_, path)| Path::new(path).exists())
        .map(|(name, path)| (*name, *path))
        .collect();

    if available_images.is_empty() {
        println!("❌ No test images found. Please ensure test assets are available.");
        return Ok(());
    }

    println!("📷 Testing with {} images:", available_images.len());
    for (name, _) in &available_images {
        println!("   • {}", name);
    }

    // Check model sizes
    let fp16_size = get_model_size("models/isnet_fp16.onnx")?;
    let fp32_size = get_model_size("models/isnet_fp32.onnx")?;

    println!("\n📊 Model Sizes:");
    println!("   • FP16: {:.1} MB", fp16_size);
    println!("   • FP32: {:.1} MB", fp32_size);
    println!(
        "   • Size difference: {:.1}x larger (FP32)",
        fp32_size / fp16_size
    );

    // Benchmark both precisions
    let mut results = Vec::new();

    // Test FP16 (current default)
    println!("\n🔬 Benchmarking FP16 Model...");
    let fp16_result = benchmark_precision("FP16", false, &available_images, fp16_size).await?;
    results.push(fp16_result);

    // Test FP32
    println!("\n🔬 Benchmarking FP32 Model...");
    let fp32_result = benchmark_precision("FP32", true, &available_images, fp32_size).await?;
    results.push(fp32_result);

    // Comparative analysis
    println!("\n📈 Comparative Analysis");
    println!("=======================");

    let fp16 = &results[0];
    let fp32 = &results[1];

    println!("\n⏱️  Performance Comparison:");
    println!("   Metric                FP16        FP32        Difference");
    println!("   ─────────────────────────────────────────────────────────");
    println!(
        "   Avg Processing Time   {:.0}ms       {:.0}ms       {:.1}x {}",
        fp16.avg_processing_time_ms,
        fp32.avg_processing_time_ms,
        if fp32.avg_processing_time_ms > fp16.avg_processing_time_ms {
            fp32.avg_processing_time_ms / fp16.avg_processing_time_ms
        } else {
            fp16.avg_processing_time_ms / fp32.avg_processing_time_ms
        },
        if fp32.avg_processing_time_ms > fp16.avg_processing_time_ms {
            "slower"
        } else {
            "faster"
        }
    );

    println!(
        "   Avg Inference Time    {:.0}ms       {:.0}ms       {:.1}x {}",
        fp16.avg_inference_time_ms,
        fp32.avg_inference_time_ms,
        if fp32.avg_inference_time_ms > fp16.avg_inference_time_ms {
            fp32.avg_inference_time_ms / fp16.avg_inference_time_ms
        } else {
            fp16.avg_inference_time_ms / fp32.avg_inference_time_ms
        },
        if fp32.avg_inference_time_ms > fp16.avg_inference_time_ms {
            "slower"
        } else {
            "faster"
        }
    );

    println!(
        "   Inference Ratio       {:.1}%        {:.1}%        {:.1}% diff",
        fp16.inference_ratio * 100.0,
        fp32.inference_ratio * 100.0,
        (fp32.inference_ratio - fp16.inference_ratio) * 100.0
    );

    println!("\n💾 Resource Comparison:");
    println!(
        "   Model Size            {:.1}MB       {:.1}MB       {:.1}x larger",
        fp16.model_size_mb,
        fp32.model_size_mb,
        fp32.model_size_mb / fp16.model_size_mb
    );
    println!(
        "   Memory Efficiency     {}        {}",
        fp16.memory_efficiency, fp32.memory_efficiency
    );

    // Recommendations
    println!("\n🎯 Recommendations");
    println!("==================");

    let performance_diff = (fp32.avg_processing_time_ms - fp16.avg_processing_time_ms)
        / fp16.avg_processing_time_ms
        * 100.0;
    let size_ratio = fp32.model_size_mb / fp16.model_size_mb;

    println!("\n📊 Key Findings:");
    if performance_diff > 5.0 {
        println!("   • FP32 is {:.1}% slower than FP16", performance_diff);
    } else if performance_diff < -5.0 {
        println!("   • FP32 is {:.1}% faster than FP16", -performance_diff);
    } else {
        println!(
            "   • Performance difference is negligible ({:.1}%)",
            performance_diff.abs()
        );
    }

    println!(
        "   • FP32 model is {:.1}x larger ({:.0}MB vs {:.0}MB)",
        size_ratio, fp32.model_size_mb, fp16.model_size_mb
    );

    println!("\n🏁 Default Build Recommendation:");

    if performance_diff.abs() < 5.0 {
        // Similar performance - choose based on size
        println!("   ✅ KEEP FP16 as default");
        println!(
            "   📝 Reasoning: Similar performance with {:.1}x smaller binary size",
            size_ratio
        );
        println!("   💡 Users prioritizing download speed and storage will benefit");
        println!(
            "   🔧 FP32 can be enabled via --features fp32-model for accuracy-critical use cases"
        );
    } else if performance_diff > 10.0 {
        // FP32 significantly slower
        println!("   ✅ KEEP FP16 as default");
        println!(
            "   📝 Reasoning: FP16 is {:.1}% faster with {:.1}x smaller size",
            -performance_diff, size_ratio
        );
        println!("   💡 Clear win for both performance and efficiency");
    } else if performance_diff < -10.0 {
        // FP32 significantly faster
        println!("   🤔 CONSIDER switching to FP32 as default");
        println!(
            "   📝 Reasoning: FP32 is {:.1}% faster (trade-off vs {:.1}x larger size)",
            -performance_diff, size_ratio
        );
        println!("   💡 Users may prefer performance over binary size");
    }

    println!("\n📦 Binary Size Impact:");
    let size_diff_mb = fp32.model_size_mb - fp16.model_size_mb;
    println!("   • Additional download: ~{:.0}MB for FP32", size_diff_mb);
    println!(
        "   • Relative to typical application size: {}",
        if size_diff_mb > 50.0 {
            "Significant impact"
        } else if size_diff_mb > 20.0 {
            "Moderate impact"
        } else {
            "Minor impact"
        }
    );

    Ok(())
}

async fn benchmark_precision(
    name: &str,
    _use_fp32: bool,
    test_images: &[(&str, &str)],
    model_size_mb: f64,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    // Configure for the specific precision
    let config = RemovalConfig::builder()
        .execution_provider(ExecutionProvider::Auto)
        .debug(false) // Use real ONNX backend
        .build()?;

    let mut total_processing_time = 0.0;
    let mut total_inference_time = 0.0;
    let mut total_time = 0.0;
    let mut inference_ratios = Vec::new();
    let mut processed_count = 0;

    for (image_name, path) in test_images {
        print!("   🧪 Testing {} with {}... ", image_name, name);

        let start = Instant::now();
        match remove_background(path, &config).await {
            Ok(result) => {
                let processing_time = start.elapsed().as_millis() as f64;
                let timings = result.timings();

                total_processing_time += processing_time;
                total_inference_time += timings.inference_ms as f64;
                total_time += timings.total_ms as f64;
                inference_ratios.push(timings.inference_ratio());
                processed_count += 1;

                println!(
                    "✅ {:.0}ms (inference: {:.0}ms, {:.1}%)",
                    processing_time,
                    timings.inference_ms,
                    timings.inference_ratio() * 100.0
                );
            },
            Err(e) => {
                println!("❌ Error: {}", e);
            },
        }
    }

    if processed_count == 0 {
        return Err("No images processed successfully".into());
    }

    let avg_inference_ratio = inference_ratios.iter().sum::<f64>() / inference_ratios.len() as f64;

    let memory_efficiency = if model_size_mb < 100.0 {
        "Excellent".to_string()
    } else if model_size_mb < 200.0 {
        "Good".to_string()
    } else {
        "Fair".to_string()
    };

    Ok(BenchmarkResults {
        precision: name.to_string(),
        avg_processing_time_ms: total_processing_time / processed_count as f64,
        avg_inference_time_ms: total_inference_time / processed_count as f64,
        avg_total_time_ms: total_time / processed_count as f64,
        inference_ratio: avg_inference_ratio,
        model_size_mb,
        memory_efficiency,
        processed_images: processed_count,
    })
}

fn get_model_size(path: &str) -> Result<f64, Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Ok(0.0); // Model not found
    }

    let metadata = fs::metadata(path)?;
    Ok(metadata.len() as f64 / (1024.0 * 1024.0)) // Convert to MB
}
