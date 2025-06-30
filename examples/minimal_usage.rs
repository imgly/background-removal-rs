//! Minimal example showing the simplest way to remove backgrounds
//!
//! This demonstrates simple usage with the new unified API
//! after the model is downloaded once.

use anyhow::Result;
use imgly_bgremove::{
    remove_background_from_reader, ModelDownloader, ModelSource, ModelSpec, RemovalConfig,
};
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<()> {
    // One-time setup: Download model (only needed once)
    let downloader = ModelDownloader::new()?;
    let model_id = downloader
        .download_model("https://huggingface.co/imgly/isnet-general-onnx", false)
        .await?;

    // Setup unified configuration
    let model_spec = ModelSpec {
        source: ModelSource::Downloaded(model_id),
        variant: None,
    };
    let config = RemovalConfig::builder()
        .model_spec(model_spec)
        .build()?;

    // Simple usage: reader-based API
    let file = File::open("input.jpg").await?;
    let result = remove_background_from_reader(file, &config).await?;

    result.save_png("output.png")?;
    println!("✅ Background removed! Saved to output.png");

    Ok(())
}
