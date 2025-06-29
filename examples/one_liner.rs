//! One-liner example: Remove background with just one function call
//!
//! This example assumes you have already downloaded a model using the CLI or
//! another example. It shows the absolute shortest code to remove a background.

use anyhow::Result;
use imgly_bgremove::{remove_background_with_model, ModelSource, ModelSpec, RemovalConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // ONE LINE: Remove background and save result
    remove_background_with_model(
        "input.jpg",
        &RemovalConfig::default(),
        &ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: None,
        },
    )
    .await?
    .save_png("output.png")?;

    println!("✅ Done! Background removed in one line of code.");
    Ok(())
}
