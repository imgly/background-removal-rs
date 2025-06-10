//! Demonstrate manual execution provider selection

use bg_remove_core::{RemovalConfig, remove_background};
use bg_remove_core::config::{ModelPrecision, ExecutionProvider};
use std::path::Path;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Execution Provider Choice Demo");
    println!("=================================");
    
    let test_image = "../../tests/assets/input/portraits/portrait_single_simple_bg.jpg";
    
    if !Path::new(test_image).exists() {
        println!("❌ Test image not found: {}", test_image);
        return Ok(());
    }

    let providers = [
        ("Auto (Default)", ExecutionProvider::Auto),
        ("CPU Only", ExecutionProvider::Cpu),
        ("CUDA GPU", ExecutionProvider::Cuda),
        ("CoreML (Apple)", ExecutionProvider::CoreMl),
    ];

    for (name, provider) in &providers {
        println!("\n🔧 Testing {} execution provider...", name);
        
        let config = RemovalConfig::builder()
            .model_precision(ModelPrecision::Fp16)
            .execution_provider(*provider)
            .debug(false)
            .build()?;

        let start_time = Instant::now();
        
        match remove_background(test_image, &config).await {
            Ok(result) => {
                let duration = start_time.elapsed();
                
                println!("   ✅ Success!");
                println!("   ⏱️  Processing time: {}ms", duration.as_millis());
                println!("   📊 Foreground ratio: {:.1}%", 
                    result.mask.statistics().foreground_ratio * 100.0);
                
                // Save output with provider name
                let provider_name = format!("{:?}", provider).to_lowercase();
                let output_path = format!("provider_test_{}.png", provider_name);
                result.save_png(&output_path)?;
                println!("   💾 Saved: {}", output_path);
            },
            Err(e) => {
                println!("   ❌ Failed: {}", e);
                match provider {
                    ExecutionProvider::Cuda => {
                        println!("   💡 CUDA may not be available on this system");
                    },
                    ExecutionProvider::CoreMl => {
                        println!("   💡 CoreML may not be available on this system");
                    },
                    _ => {}
                }
            }
        }
    }
    
    println!("\n📋 Summary:");
    println!("   • Auto: Tries CUDA → CoreML → CPU (recommended)");
    println!("   • CPU: Forces CPU execution (slower but always works)");
    println!("   • CUDA: Forces NVIDIA GPU (if available)");
    println!("   • CoreML: Forces Apple GPU/Neural Engine (if available)");
    println!("\n💡 Use .execution_provider() in your config to manually choose!");

    Ok(())
}