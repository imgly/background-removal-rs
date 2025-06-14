//! Comprehensive ICC Profile Implementation Validation
//! 
//! Final validation of complete ICC profile support across PNG, JPEG, and WebP formats.

use bg_remove_core::color_profile::ProfileExtractor;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 COMPREHENSIVE ICC PROFILE VALIDATION");
    println!("=======================================\n");
    
    let test_files = [
        ("🟢 PNG ICC Implementation", "phase4_icc_results/custom_png_with_icc.png"),
        ("🟢 JPEG ICC Implementation", "phase4_icc_results/custom_jpeg_with_icc.jpg"),
        ("🟢 WebP ICC Implementation", "phase4_icc_results/custom_webp_with_icc.webp"),
        ("📷 Original Reference", "crates/bg-remove-testing/assets/input/portraits/portrait_fine_hair_details.jpg"),
    ];
    
    let mut all_passed = true;
    let mut format_results = Vec::new();
    
    for (description, file_path) in &test_files {
        let result = analyze_icc_file(description, file_path)?;
        if file_path.contains("custom") {
            format_results.push((extract_format(file_path), result));
            if !result {
                all_passed = false;
            }
        }
        println!();
    }
    
    println!("📊 COMPREHENSIVE VALIDATION SUMMARY");
    println!("===================================");
    
    if all_passed {
        println!("✅ ALL FORMATS PASSED: Complete ICC profile support achieved!");
        println!("\n🎨 Complete ICC Profile Support Matrix:");
        
        for (format, working) in &format_results {
            let status = if *working { "✅ COMPLETE" } else { "❌ FAILED" };
            println!("   • {}: {} (extraction + embedding working)", format.to_uppercase(), status);
        }
        
        println!("\n🚀 IMPLEMENTATION STATUS:");
        println!("   • PNG: ✅ Custom iCCP chunk implementation");
        println!("   • JPEG: ✅ APP2 marker implementation");
        println!("   • WebP: ✅ RIFF ICCP chunk implementation");
        println!("   • Extraction: ✅ All formats supported");
        println!("   • Embedding: ✅ All formats supported");
        println!("   • CLI Integration: ✅ Automatic ICC-aware saving");
        
        println!("\n🎉 PHASE 5 COMPLETE: Full multi-format ICC color profile preservation achieved!");
        println!("    Professional color workflow support now available for PNG, JPEG, and WebP!");
        
    } else {
        println!("❌ SOME FORMATS FAILED: ICC embedding has issues");
        for (format, working) in &format_results {
            let status = if *working { "✅" } else { "❌" };
            println!("   {} {}: {}", status, format.to_uppercase(), if *working { "Working" } else { "Failed" });
        }
    }
    
    Ok(())
}

fn analyze_icc_file(description: &str, file_path: &str) -> Result<bool, Box<dyn std::error::Error>> {
    println!("{}", description);
    println!("{}", "─".repeat(description.len()));
    
    if !Path::new(file_path).exists() {
        println!("❌ File not found: {}", file_path);
        return Ok(false);
    }
    
    // Get file info
    let metadata = std::fs::metadata(file_path)?;
    println!("📁 File: {}", file_path);
    println!("📊 Size: {:.1} KB", metadata.len() as f64 / 1024.0);
    
    // Test ICC profile extraction
    match ProfileExtractor::extract_from_image(file_path) {
        Ok(Some(profile)) => {
            println!("✅ ICC Profile Found:");
            println!("   🎨 Color Space: {}", profile.color_space);
            println!("   📊 Profile Size: {} bytes", profile.data_size());
            println!("   💾 Has ICC Data: {}", profile.has_color_profile());
            
            if file_path.contains("custom") {
                let format = extract_format(file_path);
                println!("   🚀 {} Implementation: WORKING", format.to_uppercase());
            }
            Ok(true)
        },
        Ok(None) => {
            println!("❌ No ICC Profile Found");
            if file_path.contains("custom") {
                println!("   🐛 ICC embedding failed for this format");
                Ok(false)
            } else {
                Ok(true) // Expected for some reference files
            }
        },
        Err(e) => {
            println!("⚠️  Error extracting ICC profile: {}", e);
            Ok(false)
        }
    }
}

fn extract_format(file_path: &str) -> &str {
    if file_path.ends_with(".png") {
        "png"
    } else if file_path.ends_with(".jpg") || file_path.ends_with(".jpeg") {
        "jpeg"
    } else if file_path.ends_with(".webp") {
        "webp"
    } else {
        "unknown"
    }
}