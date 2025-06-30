//! Concise example: Remove background with public APIs
//!
//! This example shows both single-image and session-based usage patterns.
//! Assumes you have already downloaded a model using the CLI.

use anyhow::Result;
use imgly_bgremove::{remove_background_from_reader, RemovalSession, ModelSource, ModelSpec, RemovalConfig};
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<()> {
    let model_spec = ModelSpec {
        source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
        variant: None,
    };
    let config = RemovalConfig::builder().model_spec(model_spec).build()?;

    // Single image: Use convenience function
    println!("🖼️ Processing single image with convenience API...");
    let file = File::open("input.jpg").await?;
    remove_background_from_reader(file, &config)
        .await?
        .save_png("output_single.png")?;
    println!("✅ Single image processed!");

    // Multiple images: Use session for efficiency
    println!("🖼️ Processing multiple images with session API...");
    let mut session = RemovalSession::new(config)?;
    for i in 1..=3 {
        if std::path::Path::new("input.jpg").exists() {
            let file = File::open("input.jpg").await?;
            let result = session.remove_background_from_reader(file).await?;
            result.save_png(&format!("output_batch_{}.png", i))?;
            println!("  ✅ Processed image {} (model cached)", i);
        }
    }

    println!("🎉 Done! Demonstrates both single and batch processing patterns.");
    Ok(())
}
