//! Complete example demonstrating library usage with model downloading and caching
//!
//! This example shows how to use the imgly-bgremove library without the CLI,
//! demonstrating that all core functionality (downloading, caching, processing)
//! is available to library users.

use anyhow::Result;
use imgly_bgremove::{
    remove_background_with_model, BackendType, BackgroundRemovalProcessor, ExecutionProvider,
    ModelCache, ModelDownloader, ModelSource, ModelSpec, OutputFormat, ProcessorConfigBuilder,
    RemovalConfig,
};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (optional)
    env_logger::init();

    println!("🚀 IMG.LY Background Removal Library Example");
    println!("============================================");

    // 1. Initialize model cache and downloader
    println!("\n📦 Initializing model cache and downloader...");
    let cache = ModelCache::new()?;
    let downloader = ModelDownloader::new()?;

    // 2. Check if we have any cached models
    println!("🔍 Scanning for cached models...");
    let cached_models = cache.scan_cached_models()?;

    if !cached_models.is_empty() {
        println!("✅ Found {} cached model(s):", cached_models.len());
        for model in &cached_models {
            println!(
                "  • {} (variants: {})",
                model.model_id,
                model.variants.join(", ")
            );
        }
    } else {
        println!("📭 No cached models found");
    }

    // 3. Download a model if needed
    let model_url = "https://huggingface.co/imgly/isnet-general-onnx";
    let model_id = ModelCache::url_to_model_id(model_url);

    if !cache.is_model_cached(&model_id) {
        println!("\n⬇️ Downloading model: {}", model_url);
        println!("Model ID: {}", model_id);
        downloader.download_model(model_url, true).await?;
        println!("✅ Model downloaded successfully!");
    } else {
        println!("✅ Model already cached: {}", model_id);
    }

    // 4. Configure processing with different options
    println!("\n🎛️ Configuring background removal...");

    // Example 1: High-quality PNG output
    let config_png = RemovalConfig::builder()
        .execution_provider(ExecutionProvider::Auto) // Auto-detect best provider
        .output_format(OutputFormat::Png)
        .preserve_color_profiles(true)
        .build()?;

    let model_spec = ModelSpec {
        source: ModelSource::Downloaded(model_id.clone()),
        variant: None, // Auto-select based on execution provider
    };

    // 5. Process an image (if input exists)
    let input_path = "input.jpg";
    if Path::new(input_path).exists() {
        println!("🖼️ Processing image: {}", input_path);

        // Method 1: Use the high-level API
        let result = remove_background_with_model(input_path, &config_png, &model_spec).await?;
        result.save_png("output_highlevel.png")?;
        println!("✅ High-level API result saved to: output_highlevel.png");

        // Method 2: Use the unified processor for more control
        let processor_config = ProcessorConfigBuilder::new()
            .model_spec(model_spec.clone())
            .backend_type(BackendType::Onnx)
            .execution_provider(ExecutionProvider::Auto)
            .output_format(OutputFormat::Jpeg)
            .jpeg_quality(95)
            .preserve_color_profiles(true)
            .disable_cache(false) // Enable session caching for performance
            .build()?;

        let mut processor = BackgroundRemovalProcessor::new(processor_config)?;
        let result2 = processor.process_file(input_path).await?;
        result2.save("output_processor.jpg", OutputFormat::Jpeg, 95)?;
        println!("✅ Processor API result saved to: output_processor.jpg");

        // Display processing metadata
        println!("\n📊 Processing Metadata:");
        println!("  • Model: {}", result.metadata.model_name);
        if let Some(total_time) = result.metadata.total_time_ms {
            println!("  • Total time: {:.2}ms", total_time);
        }
        if let Some(inference_time) = result.metadata.inference_time_ms {
            println!("  • Inference time: {:.2}ms", inference_time);
        }
        println!(
            "  • Image dimensions: {}x{}",
            result.original_dimensions.0, result.original_dimensions.1
        );

        // Show mask statistics
        let mask_stats = result.mask.statistics();
        println!(
            "  • Foreground ratio: {:.1}%",
            mask_stats.foreground_ratio * 100.0
        );
    } else {
        println!(
            "⚠️ Input image '{}' not found. Create this file to test processing.",
            input_path
        );
        println!("   Example: cp /path/to/your/image.jpg {}", input_path);
    }

    // 6. Demonstrate different execution providers
    println!("\n🔧 Available execution providers:");
    #[cfg(feature = "onnx")]
    {
        use imgly_bgremove::backends::OnnxBackend;
        let providers = OnnxBackend::list_providers();
        for (name, available, description) in providers {
            let status = if available { "✅" } else { "❌" };
            println!("  {} {}: {}", status, name, description);
        }
    }

    // 7. Show cache information
    println!("\n💾 Cache Information:");
    println!(
        "  • Cache directory: {}",
        cache.get_current_cache_dir().display()
    );

    let final_models = cache.scan_cached_models()?;
    let total_size: u64 = final_models.iter().map(|m| m.size_bytes).sum();
    println!("  • Cached models: {}", final_models.len());
    println!(
        "  • Total cache size: {}",
        imgly_bgremove::format_size(total_size)
    );

    // 8. Demonstrate session caching benefits
    #[cfg(feature = "onnx")]
    {
        use imgly_bgremove::SessionCache;
        if let Ok(session_cache) = SessionCache::new() {
            let stats = session_cache.get_stats();
            println!("\n⚡ Session Cache Statistics:");
            println!("  • Total sessions: {}", stats.total_sessions);
            println!("  • Cache hit ratio: {:.1}%", session_cache.get_hit_ratio());
            println!(
                "  • Total cache size: {}",
                imgly_bgremove::format_cache_size(stats.total_size_bytes)
            );
        }
    }

    println!("\n🎉 Library example completed successfully!");
    println!("\nKey Benefits Demonstrated:");
    println!("  ✅ Model downloading and caching available to library users");
    println!("  ✅ Session caching for improved performance");
    println!("  ✅ Multiple processing APIs (high-level and processor)");
    println!("  ✅ Flexible configuration options");
    println!("  ✅ Automatic execution provider selection");
    println!("  ✅ Rich metadata and statistics");

    Ok(())
}
