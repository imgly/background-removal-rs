//! Concise example: Remove background with the new unified API
//!
//! This example assumes you have already downloaded a model using the CLI or
//! another example. It shows concise code to remove a background.

use anyhow::Result;
use imgly_bgremove::{remove_background_from_reader, ModelSource, ModelSpec, RemovalConfig};
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<()> {
    // Setup unified config with model
    let model_spec = ModelSpec {
        source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
        variant: None,
    };
    let config = RemovalConfig::builder()
        .model_spec(model_spec)
        .build()?;

    // Process and save in just two lines
    let file = File::open("input.jpg").await?;
    remove_background_from_reader(file, &config).await?.save_png("output.png")?;

    println!("✅ Done! Background removed with the new unified API.");
    Ok(())
}
