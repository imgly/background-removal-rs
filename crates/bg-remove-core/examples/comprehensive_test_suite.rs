//! Comprehensive test suite with improved preprocessing

use bg_remove_core::{remove_background, RemovalConfig};
use std::path::Path;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Comprehensive Test Suite - Improved Preprocessing");
    println!("===================================================");

    let test_cases = [
        ("portraits", "portrait_single_simple_bg.jpg"),
        ("portraits", "portrait_multiple_people.jpg"),
        ("products", "product_clothing_white_bg.jpg"),
        ("complex", "complex_group_photo.jpg"),
        ("edge_cases", "edge_high_contrast.jpg"),
    ];

    let config = RemovalConfig::builder().debug(false).build()?;

    let mut total_time = 0u64;
    let mut successful_tests = 0;
    let mut quality_metrics = Vec::new();

    for (category, image_name) in &test_cases {
        let input_path = format!(
            "crates/bg-remove-testing/assets/input/{}/{}",
            category, image_name
        );

        if !Path::new(&input_path).exists() {
            println!("⏭️  Skipping {}: File not found", image_name);
            continue;
        }

        print!("🔬 Testing {}... ", image_name);

        let start_time = Instant::now();
        match remove_background(&input_path, &config).await {
            Ok(result) => {
                let processing_time = start_time.elapsed().as_millis() as u64;
                total_time += processing_time;
                successful_tests += 1;

                // Analyze mask quality
                let mask_stats = result.mask.statistics();
                let mask_data = &result.mask.data;

                // Check value distribution
                let min_val = *mask_data.iter().min().unwrap();
                let max_val = *mask_data.iter().max().unwrap();
                let transparent_pixels = mask_data.iter().filter(|&&x| x < 50).count();
                let opaque_pixels = mask_data.iter().filter(|&&x| x > 200).count();

                let quality_score = QualityMetrics {
                    processing_time,
                    foreground_ratio: mask_stats.foreground_ratio,
                    value_range: (min_val, max_val),
                    transparency_ratio: transparent_pixels as f32 / mask_data.len() as f32,
                    opacity_ratio: opaque_pixels as f32 / mask_data.len() as f32,
                };

                println!(
                    "✅ {}ms | FG: {:.1}% | Range: {}-{} | Trans: {:.1}% | Opaque: {:.1}%",
                    processing_time,
                    mask_stats.foreground_ratio * 100.0,
                    min_val,
                    max_val,
                    quality_score.transparency_ratio * 100.0,
                    quality_score.opacity_ratio * 100.0
                );

                // Save improved output
                let output_name = format!(
                    "improved_{}_{}.png",
                    category,
                    image_name.trim_end_matches(".jpg")
                );
                result.save_png(&output_name)?;

                quality_metrics.push(quality_score);
            },
            Err(e) => {
                println!("❌ Failed: {}", e);
            },
        }
    }

    if successful_tests > 0 {
        println!("\n📊 Overall Results:");
        println!(
            "   ✅ Successful tests: {}/{}",
            successful_tests,
            test_cases.len()
        );
        println!(
            "   ⏱️  Average processing time: {}ms",
            total_time / successful_tests as u64
        );

        // Aggregate quality metrics
        let avg_fg_ratio = quality_metrics
            .iter()
            .map(|m| m.foreground_ratio)
            .sum::<f32>()
            / quality_metrics.len() as f32;
        let avg_transparency = quality_metrics
            .iter()
            .map(|m| m.transparency_ratio)
            .sum::<f32>()
            / quality_metrics.len() as f32;
        let avg_opacity = quality_metrics.iter().map(|m| m.opacity_ratio).sum::<f32>()
            / quality_metrics.len() as f32;

        println!(
            "   🎯 Average foreground ratio: {:.1}%",
            avg_fg_ratio * 100.0
        );
        println!(
            "   🔍 Average transparency: {:.1}%",
            avg_transparency * 100.0
        );
        println!("   🔍 Average opacity: {:.1}%", avg_opacity * 100.0);

        // Check for full value range usage
        let uses_full_range = quality_metrics
            .iter()
            .any(|m| m.value_range.0 < 50 && m.value_range.1 > 200);

        println!("\n🏆 Quality Assessment:");
        if uses_full_range && avg_transparency > 0.1 && avg_opacity > 0.1 {
            println!("   ✅ EXCELLENT - Full value range, good transparency distribution");
        } else if uses_full_range {
            println!("   🎯 GOOD - Using full value range");
        } else {
            println!("   ⚠️  NEEDS REVIEW - Limited value range detected");
        }

        println!("\n💾 Improved outputs saved with 'improved_' prefix");
        println!("🔬 Key improvements:");
        println!("   • ISNet normalization: Mean=[128,128,128], Std=[256,256,256]");
        println!("   • Fixed 1024x1024 input resolution");
        println!("   • Direct tensor-to-alpha conversion");
    }

    Ok(())
}

#[derive(Debug)]
struct QualityMetrics {
    #[allow(dead_code)] // Reserved for performance analysis
    processing_time: u64,
    foreground_ratio: f32,
    value_range: (u8, u8),
    transparency_ratio: f32,
    opacity_ratio: f32,
}
