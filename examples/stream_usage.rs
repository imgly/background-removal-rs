//! Stream-based API usage examples
//!
//! This example demonstrates the new stream-based APIs that allow processing
//! images from memory, network streams, and other sources without requiring files.

use anyhow::Result;
use imgly_bgremove::{
    remove_background_from_reader, remove_background_simple_bytes,
    remove_background_with_model_bytes, BackendType, BackgroundRemovalProcessor, ExecutionProvider,
    ModelDownloader, ModelSource, ModelSpec, OutputFormat, ProcessorConfigBuilder, RemovalConfig,
};
use std::io::Cursor;
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (optional)
    env_logger::init();

    println!("🌊 Stream-Based Background Removal Examples");
    println!("==========================================");

    // Ensure we have a model available
    let downloader = ModelDownloader::new()?;
    let model_url = "https://huggingface.co/imgly/isnet-general-onnx";
    let model_id = downloader.download_model(model_url, false).await?;
    println!("✅ Model ready: {model_id}");

    // Example 1: Ultra-simple bytes processing
    println!("\n📝 Example 1: Ultra-simple bytes processing");
    if let Ok(sample_data) = create_sample_image() {
        match remove_background_simple_bytes(&sample_data).await {
            Ok(png_bytes) => {
                tokio::fs::write("stream_example_1.png", png_bytes).await?;
                println!("✅ Processed with ultra-simple API -> stream_example_1.png");
            },
            Err(e) => println!("❌ Error: {e}"),
        }
    }

    // Example 2: Bytes processing with custom configuration
    println!("\n🎛️ Example 2: Bytes processing with custom format");
    if let Ok(sample_data) = create_sample_image() {
        let config = RemovalConfig::builder()
            .output_format(OutputFormat::WebP)
            .webp_quality(85)
            .execution_provider(ExecutionProvider::Auto)
            .build()?;

        let model_spec = ModelSpec {
            source: ModelSource::Downloaded(model_id.clone()),
            variant: None,
        };

        match remove_background_with_model_bytes(&sample_data, &config, &model_spec).await {
            Ok(webp_bytes) => {
                tokio::fs::write("stream_example_2.webp", webp_bytes).await?;
                println!("✅ Processed with custom config -> stream_example_2.webp");
            },
            Err(e) => println!("❌ Error: {e}"),
        }
    }

    // Example 3: Stream processing from reader
    println!("\n📁 Example 3: Stream processing from file reader");
    if std::path::Path::new("input.jpg").exists() {
        let file = File::open("input.jpg").await?;
        let config = RemovalConfig::default();
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded(model_id.clone()),
            variant: None,
        };

        match remove_background_from_reader(file, None, &config, &model_spec).await {
            Ok(result) => {
                result.save_png("stream_example_3.png")?;
                println!("✅ Processed from file stream -> stream_example_3.png");
            },
            Err(e) => println!("❌ Error: {e}"),
        }
    } else {
        println!("⚠️ Skipped: input.jpg not found");
    }

    // Example 4: Advanced processor usage with streams
    println!("\n🔧 Example 4: Advanced processor with stream output");
    if let Ok(sample_data) = create_sample_image() {
        let processor_config = ProcessorConfigBuilder::new()
            .model_spec(ModelSpec {
                source: ModelSource::Downloaded(model_id.clone()),
                variant: None,
            })
            .backend_type(BackendType::Onnx)
            .execution_provider(ExecutionProvider::Auto)
            .output_format(OutputFormat::Jpeg)
            .jpeg_quality(95)
            .build()?;

        let mut processor = BackgroundRemovalProcessor::new(processor_config)?;

        match processor.process_bytes(&sample_data) {
            Ok(result) => {
                // Stream output to file
                let output_file = File::create("stream_example_4.jpg").await?;
                let bytes_written = result.write_to(output_file, OutputFormat::Jpeg, 95).await?;
                println!("✅ Streamed {bytes_written} bytes to stream_example_4.jpg");

                // Also demonstrate in-memory processing
                let png_bytes = result.to_bytes(OutputFormat::Png, 100)?;
                println!(
                    "📊 Generated {} bytes of PNG data in memory",
                    png_bytes.len()
                );

                // Show processing metadata
                println!("📈 Processing stats:");
                if let Some(inference_time) = result.metadata.inference_time_ms {
                    println!("   Inference: {inference_time:.2}ms");
                }
                if let Some(total_time) = result.metadata.total_time_ms {
                    println!("   Total: {total_time:.2}ms");
                }
            },
            Err(e) => println!("❌ Error: {e}"),
        }
    }

    // Example 5: Memory cursor processing
    println!("\n💾 Example 5: Memory cursor processing");
    if let Ok(sample_data) = create_sample_image() {
        let cursor = Cursor::new(sample_data);
        let config = RemovalConfig::default();
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded(model_id),
            variant: None,
        };

        match remove_background_from_reader(cursor, None, &config, &model_spec).await {
            Ok(result) => {
                let output_bytes = result.to_bytes(OutputFormat::Png, 100)?;
                tokio::fs::write("stream_example_5.png", output_bytes).await?;
                println!("✅ Processed from memory cursor -> stream_example_5.png");
            },
            Err(e) => println!("❌ Error: {e}"),
        }
    }

    println!("\n🎉 Stream processing examples completed!");
    println!("\nKey Benefits Demonstrated:");
    println!("  ✅ Memory-based processing (no temp files needed)");
    println!("  ✅ Stream input/output for network usage");
    println!("  ✅ Flexible format handling");
    println!("  ✅ Advanced processor control");
    println!("  ✅ Backwards compatibility with file-based APIs");

    Ok(())
}

/// Create a minimal sample image for testing
/// In real usage, you'd load actual image data from files, network, etc.
fn create_sample_image() -> Result<Vec<u8>> {
    use image::{ImageBuffer, Rgb};

    // Create a simple 64x64 test image
    let img = ImageBuffer::from_fn(64, 64, |x, y| {
        let r = (x * 4) as u8;
        let g = (y * 4) as u8;
        let b = ((x + y) * 2) as u8;
        Rgb([r, g, b])
    });

    // Encode to PNG bytes
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    img.write_to(&mut cursor, image::ImageFormat::Png)?;

    Ok(buffer)
}
