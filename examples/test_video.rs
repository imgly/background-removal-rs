use imgly_bgremove::{remove_background_from_video_file, RemovalConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing video processing...");

    let input_path = Path::new("/tmp/test_small.mp4");
    let output_path = Path::new("/tmp/test_output.mp4");

    if !input_path.exists() {
        println!("Input video file does not exist: {}", input_path.display());
        return Ok(());
    }

    println!("Input file exists: {}", input_path.display());

    let config = RemovalConfig::default();

    match remove_background_from_video_file(input_path, &config).await {
        Ok(result) => {
            println!("Video processing successful!");
            println!("Processed {} frames", result.frame_stats.frames_processed);

            // Write the result to output file
            std::fs::write(output_path, result.video_data)?;
            println!("Output written to: {}", output_path.display());
        },
        Err(e) => {
            println!("Video processing failed: {}", e);
        },
    }

    Ok(())
}
