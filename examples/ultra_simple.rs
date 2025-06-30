//! Simple example: Remove background with the reader-based API
//!
//! This demonstrates a simple way to remove a background using the new unified API.
//! Requires a model to be downloaded/cached.

use anyhow::Result;
use imgly_bgremove::{remove_background_from_reader, ModelDownloader, ModelSource, ModelSpec, RemovalConfig};
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<()> {
    // Download a model if needed
    let downloader = ModelDownloader::new()?;
    let model_url = "https://huggingface.co/imgly/isnet-general-onnx";
    let model_id = downloader.download_model(model_url, false).await?;

    // Setup configuration with model
    let model_spec = ModelSpec {
        source: ModelSource::Downloaded(model_id),
        variant: None,
    };
    let config = RemovalConfig::builder()
        .model_spec(model_spec)
        .build()?;

    // Process image: file reader -> processed result
    let file = File::open("input.jpg").await?;
    let result = remove_background_from_reader(file, &config).await?;
    result.save_png("output.png")?;

    println!("✅ Done!");
    Ok(())
}
