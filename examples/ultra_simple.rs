//! Ultra-simple example: Remove background with just one function call
//!
//! This is the absolute shortest possible way to remove a background.
//! Requires a model to be cached (download once with CLI or ModelDownloader).

use anyhow::Result;
use imgly_bgremove::remove_background_simple;

#[tokio::main]
async fn main() -> Result<()> {
    // ONE FUNCTION CALL: Remove background
    remove_background_simple("input.jpg", "output.png").await?;

    println!("✅ Done!");
    Ok(())
}
