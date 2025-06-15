//! Comprehensive benchmark for execution provider performance
//!
//! Tests execution providers:
//! - CPU vs `CoreML` (Apple Silicon GPU)
//! - Various image sizes and types
//! - Uses the model precision compiled into the binary

use anyhow::{Context, Result};
use bg_remove_core::{ExecutionProvider, OutputFormat, RemovalConfig};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    provider: ExecutionProvider,
    name: String,
}

#[derive(Debug)]
struct BenchmarkResult {
    config: BenchmarkConfig,
    image_size: (u32, u32),
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    #[allow(dead_code)] // Reserved for future detailed reporting
    iterations: usize,
    success_rate: f64,
}

#[derive(Debug)]
struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    fn print_summary(&self) {
        println!("\n🏆 BENCHMARK RESULTS SUMMARY");
        println!("═══════════════════════════════════════════════════════════════");

        // Group by image size
        let mut by_size: HashMap<(u32, u32), Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            by_size.entry(result.image_size).or_default().push(result);
        }

        for (size, results) in by_size {
            println!("\n📏 Image Size: {}x{}", size.0, size.1);
            println!("───────────────────────────────────────────────────────────────");

            // Sort by average time for ranking
            let mut sorted_results = results;
            sorted_results.sort_by(|a, b| a.avg_time_ms.partial_cmp(&b.avg_time_ms).unwrap());

            for (rank, result) in sorted_results.iter().enumerate() {
                let medal = match rank {
                    0 => "🥇",
                    1 => "🥈",
                    2 => "🥉",
                    _ => "  ",
                };

                println!(
                    "{} {:20} | {:6.1}ms | {:6.1}-{:6.1}ms | {:5.1}% success",
                    medal,
                    result.config.name,
                    result.avg_time_ms,
                    result.min_time_ms,
                    result.max_time_ms,
                    result.success_rate * 100.0
                );
            }
        }

        // Overall winner analysis
        self.print_analysis();
    }

    fn print_analysis(&self) {
        println!("\n📊 PERFORMANCE ANALYSIS");
        println!("═══════════════════════════════════════════════════════════════");

        // Find fastest overall
        if let Some(fastest) = self
            .results
            .iter()
            .min_by(|a, b| a.avg_time_ms.partial_cmp(&b.avg_time_ms).unwrap())
        {
            println!(
                "🚀 Fastest Overall: {} ({:.1}ms average)",
                fastest.config.name, fastest.avg_time_ms
            );
        }

        // Compare by provider
        let cpu_avg = self
            .results
            .iter()
            .filter(|r| matches!(r.config.provider, ExecutionProvider::Cpu))
            .map(|r| r.avg_time_ms)
            .collect::<Vec<_>>();
        let coreml_avg = self
            .results
            .iter()
            .filter(|r| matches!(r.config.provider, ExecutionProvider::CoreMl))
            .map(|r| r.avg_time_ms)
            .collect::<Vec<_>>();

        if !cpu_avg.is_empty() && !coreml_avg.is_empty() {
            let cpu_mean = cpu_avg.iter().sum::<f64>() / cpu_avg.len() as f64;
            let coreml_mean = coreml_avg.iter().sum::<f64>() / coreml_avg.len() as f64;
            let provider_diff = ((cpu_mean - coreml_mean) / cpu_mean) * 100.0;
            println!("🍎 CoreML vs CPU: {provider_diff:.1}% faster on average");
        }
    }
}

fn create_test_image(width: u32, height: u32, pattern: &str) -> DynamicImage {
    let mut img = ImageBuffer::new(width, height);

    match pattern {
        "gradient" => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let r = (x * 255 / width) as u8;
                let g = (y * 255 / height) as u8;
                let b = ((x + y) * 255 / (width + height)) as u8;
                *pixel = Rgb([r, g, b]);
            }
        },
        "checkerboard" => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let checker = ((x / 32) + (y / 32)) % 2;
                let color = if checker == 0 { 255 } else { 0 };
                *pixel = Rgb([color, color, color]);
            }
        },
        "portrait" => {
            // Create a simple portrait-like pattern
            let center_x = width / 2;
            let center_y = height / 2;
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let dx = (x as i32 - center_x as i32).unsigned_abs();
                let dy = (y as i32 - center_y as i32).unsigned_abs();
                let dist = f64::from(dx * dx + dy * dy).sqrt();
                let face_radius = f64::from(width.min(height) / 3);

                if dist < face_radius {
                    // Face area - skin tone
                    *pixel = Rgb([222, 184, 135]);
                } else {
                    // Background - varied colors
                    let bg_r = (x * 255 / width) as u8;
                    let bg_g = 100;
                    let bg_b = (y * 255 / height) as u8;
                    *pixel = Rgb([bg_r, bg_g, bg_b]);
                }
            }
        },
        _ => {
            // Default solid color
            for pixel in img.pixels_mut() {
                *pixel = Rgb([128, 128, 128]);
            }
        },
    }

    DynamicImage::ImageRgb8(img)
}

async fn benchmark_configuration(
    config: &BenchmarkConfig,
    test_image: &DynamicImage,
    iterations: usize,
) -> Result<BenchmarkResult> {
    println!(
        "🔄 Testing: {} ({}x{})",
        config.name,
        test_image.width(),
        test_image.height()
    );

    let removal_config = RemovalConfig::builder()
        .execution_provider(config.provider)
        .output_format(OutputFormat::Png)
        .build()
        .context("Failed to build removal config")?;

    let mut times = Vec::new();
    let mut successes = 0;

    for i in 0..iterations {
        print!("  Iteration {}/{}\r", i + 1, iterations);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let start = Instant::now();

        match bg_remove_core::process_image(test_image.clone(), &removal_config) {
            Ok(_) => {
                let duration = start.elapsed();
                times.push(duration.as_millis() as f64);
                successes += 1;
            },
            Err(e) => {
                eprintln!("  ❌ Error in iteration {}: {}", i + 1, e);
            },
        }
    }

    if times.is_empty() {
        return Err(anyhow::anyhow!(
            "No successful iterations for {}",
            config.name
        ));
    }

    let avg_time = times.iter().sum::<f64>() / times.len() as f64;
    let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time = times.iter().fold(0.0f64, |a, &b| a.max(b));
    let success_rate = f64::from(successes) / iterations as f64;

    println!(
        "  ✅ Completed: {:.1}ms avg ({:.1}% success)",
        avg_time,
        success_rate * 100.0
    );

    Ok(BenchmarkResult {
        config: config.clone(),
        image_size: (test_image.width(), test_image.height()),
        avg_time_ms: avg_time,
        min_time_ms: min_time,
        max_time_ms: max_time,
        iterations,
        success_rate,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Background Removal Benchmark Suite");
    println!("Testing execution providers with embedded model");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create benchmark configurations (precision determined at compile time)
    let configs = vec![
        BenchmarkConfig {
            provider: ExecutionProvider::Cpu,
            name: "CPU".to_string(),
        },
        BenchmarkConfig {
            provider: ExecutionProvider::CoreMl,
            name: "CoreML".to_string(),
        },
    ];

    // Create test images of different sizes
    let test_cases = vec![
        (512, 512, "portrait"),
        (1024, 1024, "gradient"),
        (256, 256, "checkerboard"),
    ];

    let mut suite = BenchmarkSuite::new();
    let iterations = 5; // Number of iterations per test

    for (width, height, pattern) in test_cases {
        println!("📸 Generating test image: {width}x{height} ({pattern})");
        let test_image = create_test_image(width, height, pattern);

        for config in &configs {
            match benchmark_configuration(config, &test_image, iterations).await {
                Ok(result) => suite.add_result(result),
                Err(e) => eprintln!("❌ Failed to benchmark {}: {}", config.name, e),
            }
        }
        println!();
    }

    suite.print_summary();

    Ok(())
}
