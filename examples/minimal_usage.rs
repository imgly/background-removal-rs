//! Minimal example showing the shortest possible way to remove backgrounds
//!
//! This demonstrates the absolute simplest usage - just one function call
//! after the model is downloaded once.

use anyhow::Result;
use imgly_bgremove::{
    remove_background_with_model, ModelDownloader, ModelSource, ModelSpec, RemovalConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // One-time setup: Download model (only needed once)
    let downloader = ModelDownloader::new()?;
    let model_id = downloader
        .download_model("https://huggingface.co/imgly/isnet-general-onnx", false)
        .await?;

    // Minimal usage: One function call to remove background
    let result = remove_background_with_model(
        "input.jpg",
        &RemovalConfig::default(),
        &ModelSpec {
            source: ModelSource::Downloaded(model_id),
            variant: None,
        },
    )
    .await?;

    result.save_png("output.png")?;
    println!("✅ Background removed! Saved to output.png");

    Ok(())
}
