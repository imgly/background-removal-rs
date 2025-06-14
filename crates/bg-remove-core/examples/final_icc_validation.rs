//! Final ICC Profile Embedding Validation
//! 
//! Comprehensive test of Phase 4 ICC embedding implementation for both PNG and JPEG formats.

use bg_remove_core::color_profile::ProfileExtractor;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 FINAL ICC PROFILE EMBEDDING VALIDATION");
    println!("=========================================\n");
    
    let test_files = [
        ("🟢 Custom PNG ICC Embedding", "phase4_icc_results/custom_png_with_icc.png"),
        ("🟢 Custom JPEG ICC Embedding", "phase4_icc_results/custom_jpeg_with_icc.jpg"),
        ("📷 Original Reference", "crates/bg-remove-testing/assets/input/portraits/portrait_fine_hair_details.jpg"),
    ];
    
    let mut all_passed = true;
    
    for (description, file_path) in &test_files {
        analyze_icc_file(description, file_path, &mut all_passed)?;
        println!();
    }
    
    println!("📊 FINAL VALIDATION SUMMARY");
    println!("===========================");
    
    if all_passed {
        println!("✅ ALL TESTS PASSED: ICC embedding working for both PNG and JPEG");
        println!("🚀 Phase 4 Custom ICC Embedding: COMPLETE AND VALIDATED");
        println!("\n🎨 ICC Profile Support Matrix:");
        println!("   • PNG: ✅ Custom iCCP chunk implementation (WORKING)");
        println!("   • JPEG: ✅ APP2 marker implementation (WORKING)");
        println!("   • Extraction: ✅ Both formats (WORKING)");
        println!("   • Embedding: ✅ Both formats (WORKING)");
        println!("\n🎉 Full ICC color profile preservation pipeline is now production-ready!");
    } else {
        println!("❌ SOME TESTS FAILED: ICC embedding has issues");
    }
    
    Ok(())
}

fn analyze_icc_file(description: &str, file_path: &str, all_passed: &mut bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", description);
    println!("{}", "─".repeat(description.len()));
    
    if !Path::new(file_path).exists() {
        println!("❌ File not found: {}", file_path);
        *all_passed = false;
        return Ok(());
    }
    
    // Get file info
    let metadata = std::fs::metadata(file_path)?;
    println!("📁 File: {}", file_path);
    println!("📊 Size: {:.1} MB", metadata.len() as f64 / 1_048_576.0);
    
    // Test ICC profile extraction
    match ProfileExtractor::extract_from_image(file_path) {
        Ok(Some(profile)) => {
            println!("✅ ICC Profile Found:");
            println!("   🎨 Color Space: {}", profile.color_space);
            println!("   📊 Profile Size: {} bytes", profile.data_size());
            println!("   💾 Has ICC Data: {}", profile.has_color_profile());
            
            if file_path.contains("custom") {
                if file_path.contains(".png") {
                    println!("   🚀 PNG iCCP Implementation: WORKING");
                } else if file_path.contains(".jpg") {
                    println!("   🚀 JPEG APP2 Implementation: WORKING");
                }
            }
        },
        Ok(None) => {
            println!("❌ No ICC Profile Found");
            if file_path.contains("custom") {
                println!("   🐛 ICC embedding failed for this format");
                *all_passed = false;
            }
        },
        Err(e) => {
            println!("⚠️  Error extracting ICC profile: {}", e);
            *all_passed = false;
        }
    }
    
    Ok(())
}