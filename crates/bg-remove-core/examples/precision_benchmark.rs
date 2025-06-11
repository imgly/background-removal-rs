//! Performance benchmark with different execution providers

use bg_remove_core::config::ExecutionProvider;
use bg_remove_core::{remove_background, RemovalConfig};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Execution Provider Performance Comparison");
    println!("==============================================");

    let test_image =
        "crates/bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg";

    if !std::path::Path::new(test_image).exists() {
        println!("❌ Test image not found: {}", test_image);
        return Ok(());
    }

    let providers = [
        ("Auto (Default)", ExecutionProvider::Auto),
        ("CPU Only", ExecutionProvider::Cpu),
        ("CoreML (Apple)", ExecutionProvider::CoreMl),
    ];

    let mut results = Vec::new();

    for (name, provider) in &providers {
        println!("🧪 Testing {} execution provider...", name);
        let config = RemovalConfig::builder()
            .execution_provider(*provider)
            .debug(false)
            .build()?;

        let start = Instant::now();
        match remove_background(test_image, &config).await {
            Ok(result) => {
                let elapsed_time = start.elapsed().as_secs_f64();
                let stats = result.mask.statistics();

                println!("   ⏱️  Time: {:.2}s", elapsed_time);
                println!("   📊 Foreground: {:.1}%", stats.foreground_ratio * 100.0);

                // Save output
                let output_name = format!(
                    "output_{}.png",
                    name.to_lowercase()
                        .replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                );
                result.save_png(&output_name)?;

                results.push((name, elapsed_time, stats.foreground_ratio));
            },
            Err(e) => {
                println!("   ❌ Error: {}", e);
            },
        }
        println!();
    }

    // Comparison
    if results.len() > 1 {
        println!("🏆 Performance Comparison:");
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (rank, (name, time, fg_ratio)) in results.iter().enumerate() {
            let medal = match rank {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "  ",
            };
            println!(
                "   {} {}: {:.2}s ({:.1}% foreground)",
                medal,
                name,
                time,
                fg_ratio * 100.0
            );
        }
    }

    println!("\n📝 Note: Model precision is now automatically optimized.");
    println!("💾 Outputs saved with provider-specific names.");

    Ok(())
}
