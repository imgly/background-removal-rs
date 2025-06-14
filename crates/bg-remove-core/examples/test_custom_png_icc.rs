//! Test Custom PNG ICC Profile Embedding
//! 
//! This example tests the new custom PNG iCCP chunk embedding implementation.

use bg_remove_core::color_profile::ProfileExtractor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Custom PNG ICC Embedding");
    println!("===================================\n");
    
    let test_file = "phase4_icc_results/custom_png_with_icc.png";
    
    if !std::path::Path::new(test_file).exists() {
        println!("❌ Test file not found: {}", test_file);
        return Ok(());
    }
    
    // Get file size
    let metadata = std::fs::metadata(test_file)?;
    println!("📁 File: {}", test_file);
    println!("📊 Size: {:.1} MB", metadata.len() as f64 / 1_048_576.0);
    
    match ProfileExtractor::extract_from_image(test_file) {
        Ok(Some(profile)) => {
            println!("✅ SUCCESS: Custom PNG ICC Embedding Working!");
            println!("🎨 Color Space: {}", profile.color_space);
            println!("📊 Profile Size: {} bytes", profile.data_size());
            println!("💾 Has ICC Data: {}", profile.has_color_profile());
            println!("🚀 Custom iCCP chunk implementation: WORKING!");
            println!("\n🎯 PNG ICC Embedding Status: ✅ COMPLETE");
        },
        Ok(None) => {
            println!("❌ FAILED: No ICC profile found in custom PNG");
            println!("🐛 Custom iCCP chunk implementation may have issues");
            println!("\n🎯 PNG ICC Embedding Status: ❌ FAILED");
        },
        Err(e) => {
            println!("⚠️  ERROR: Failed to extract ICC profile: {}", e);
            println!("\n🎯 PNG ICC Embedding Status: ⚠️  ERROR");
        }
    }
    
    Ok(())
}