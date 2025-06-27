//! IMG.LY Background Removal CLI Tool
//!
//! Command-line interface for removing backgrounds from images using the consolidated
//! imgly-bgremove library with support for ONNX Runtime and Tract backends.

#[cfg(feature = "cli")]
use imgly_bgremove::cli;

#[cfg(feature = "cli")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::main().await
}

#[cfg(not(feature = "cli"))]
fn main() {
    panic!("CLI feature not enabled. Please rebuild with --features cli");
}
