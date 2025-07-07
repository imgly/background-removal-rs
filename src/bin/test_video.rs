use imgly_bgremove::{remove_background_from_video_file, RemovalConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Testing Video Processing...");

    let input_path = Path::new("/tmp/test_small.mp4");
    let output_path = Path::new("/tmp/output_with_bg_removed.mp4");

    if !input_path.exists() {
        println!(
            "❌ Input video file does not exist: {}",
            input_path.display()
        );
        println!("Please create a test video first with:");
        println!(
            "   ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=10 /tmp/test_small.mp4"
        );
        return Ok(());
    }

    println!("✅ Input file exists: {}", input_path.display());
    println!("🎯 Output will be: {}", output_path.display());

    let config = RemovalConfig::default();

    println!("🚀 Starting video processing...");
    match remove_background_from_video_file(input_path, &config).await {
        Ok(result) => {
            println!("✅ Video processing successful!");
            println!(
                "📊 Processed {} frames",
                result.frame_stats.frames_processed
            );

            // Write the result to output file
            std::fs::write(output_path, result.video_data)?;
            println!("💾 Output written to: {}", output_path.display());

            // Show file info
            let metadata = std::fs::metadata(output_path)?;
            println!("📏 Output file size: {} bytes", metadata.len());
        },
        Err(e) => {
            println!("❌ Video processing failed: {}", e);
        },
    }

    Ok(())
}
