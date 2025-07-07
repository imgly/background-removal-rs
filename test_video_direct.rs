// Quick test of video processing functionality
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Testing Video Processing Backend...");
    
    // Test if video support is compiled in
    #[cfg(feature = "video-support")]
    {
        use imgly_bgremove::backends::video::{FFmpegBackend, VideoBackend};
        use imgly_bgremove::RemovalConfig;
        
        let backend = FFmpegBackend::new()?;
        let input_path = Path::new("/tmp/test_simple.mp4");
        
        if input_path.exists() {
            println!("✅ Input video found: {}", input_path.display());
            
            // Test metadata extraction
            let metadata = backend.get_metadata(input_path).await?;
            println!("📊 Video metadata:");
            println!("   Duration: {:.2}s", metadata.duration);
            println!("   Resolution: {}x{}", metadata.width, metadata.height);
            println!("   FPS: {:.2}", metadata.fps);
            println!("   Format: {:?}", metadata.format);
            println!("   Has Audio: {}", metadata.has_audio);
            
            // Test frame extraction (just a few frames)
            println!("🎞️ Testing frame extraction...");
            let mut frame_stream = backend.extract_frames(input_path).await?;
            let mut frame_count = 0;
            
            use futures::StreamExt;
            while let Some(frame_result) = frame_stream.next().await {
                match frame_result {
                    Ok(frame) => {
                        frame_count += 1;
                        println!("   Frame {}: {}x{} at {:.2}s", 
                            frame_count, 
                            frame.image.width(), 
                            frame.image.height(),
                            frame.timestamp.as_secs_f64());
                        
                        if frame_count >= 3 { break; } // Just test a few frames
                    }
                    Err(e) => {
                        println!("❌ Frame extraction error: {}", e);
                        break;
                    }
                }
            }
            
            println!("✅ Video backend test successful!");
        } else {
            println!("❌ No test video found. Create one with:");
            println!("   ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=10 /tmp/test_simple.mp4");
        }
    }
    
    #[cfg(not(feature = "video-support"))]
    {
        println!("❌ Video support not compiled in");
        println!("Build with: cargo build --features video-support");
    }
    
    Ok(())
}