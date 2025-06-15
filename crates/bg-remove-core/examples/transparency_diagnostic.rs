//! Diagnose and fix transparency issues

use bg_remove_core::config::OutputFormat;
use bg_remove_core::{remove_background, RemovalConfig};
use image::DynamicImage;
use std::path::Path;

#[tokio::main]
#[allow(clippy::too_many_lines)] // Transparency diagnostic tool with detailed testing and output generation
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Transparency Diagnostic Tool");
    println!("==============================");

    let test_image =
        "../bg-remove-testing/assets/input/portraits/portrait_single_simple_bg.jpg";
    let js_reference =
        "../bg-remove-testing/assets/expected/portraits/portrait_single_simple_bg.png";

    if !Path::new(test_image).exists() {
        println!("❌ Test image not found: {test_image}");
        return Ok(());
    }

    let config = RemovalConfig::builder()
        .output_format(OutputFormat::Png)
        .debug(false)
        .build()?;

    // Process with our implementation
    println!("🧪 Processing with Rust implementation...");
    let result = remove_background(test_image, &config).await?;

    // Save various outputs for analysis
    result.save_png("transparency_debug_rust_original.png")?;
    result.mask.save_png("transparency_debug_mask_raw.png")?;

    // Analyze mask statistics
    let mask_stats = result.mask.statistics();
    println!("📊 Mask Statistics:");
    println!("   Total pixels: {total}", total = mask_stats.total_pixels);
    println!(
        "   Foreground: {} ({:.1}%)",
        mask_stats.foreground_pixels,
        mask_stats.foreground_ratio * 100.0
    );
    println!(
        "   Background: {} ({:.1}%)",
        mask_stats.background_pixels,
        mask_stats.background_ratio * 100.0
    );

    // Analyze mask value distribution
    let mask_data = &result.mask.data;
    let mut value_counts = std::collections::HashMap::new();
    for &value in mask_data {
        *value_counts.entry(value).or_insert(0) += 1;
    }

    println!("\n📈 Mask Value Distribution (first 10):");
    let mut sorted_values: Vec<_> = value_counts.iter().collect();
    sorted_values.sort_by_key(|&(value, _)| value);
    for (value, count) in sorted_values.iter().take(10) {
        let percentage = f64::from(**count) / (mask_data.len() as f64) * 100.0;
        println!("   Value {value}: {count} pixels ({percentage:.1}%)");
    }

    // Create binary mask version (threshold at 127)
    println!("\n🔧 Creating binary mask (threshold=127)...");
    let mut binary_mask_data = mask_data.clone();
    for value in &mut binary_mask_data {
        *value = if *value > 127 { 255 } else { 0 };
    }

    // Create binary mask version
    let binary_mask =
        bg_remove_core::types::SegmentationMask::new(binary_mask_data, result.mask.dimensions);

    // Apply binary mask to create pure transparency
    println!("🎨 Creating binary transparency output...");
    let input_image = image::open(test_image)?;
    let mut rgba_image = input_image.to_rgba8();

    // Apply binary mask
    binary_mask.apply_to_image(&mut rgba_image)?;
    let binary_result = DynamicImage::ImageRgba8(rgba_image);
    binary_result.save("transparency_debug_rust_binary.png")?;
    binary_mask.save_png("transparency_debug_mask_binary.png")?;

    // Compare with JavaScript if available
    if Path::new(js_reference).exists() {
        println!("\n📊 JavaScript Reference Analysis:");
        let js_image = image::open(js_reference)?;

        if let DynamicImage::ImageRgba8(js_rgba) = js_image {
            // Count transparent vs opaque pixels in JS output
            let mut transparent_count = 0;
            let mut opaque_count = 0;

            for pixel in js_rgba.pixels() {
                if pixel[3] == 0 {
                    transparent_count += 1;
                } else if pixel[3] == 255 {
                    opaque_count += 1;
                }
            }

            let total_pixels = js_rgba.pixels().len();
            println!(
                "   Transparent pixels: {} ({:.1}%)",
                transparent_count,
                (transparent_count as f64) / (total_pixels as f64) * 100.0
            );
            println!(
                "   Opaque pixels: {} ({:.1}%)",
                opaque_count,
                (opaque_count as f64) / (total_pixels as f64) * 100.0
            );
            println!(
                "   Partial transparency: {} ({:.1}%)",
                total_pixels - transparent_count - opaque_count,
                ((total_pixels - transparent_count - opaque_count) as f64) / (total_pixels as f64)
                    * 100.0
            );
        }
    }

    println!("\n💾 Debug outputs saved:");
    println!("   transparency_debug_rust_original.png - Original Rust output");
    println!("   transparency_debug_rust_binary.png - Binary threshold version");
    println!("   transparency_debug_mask_raw.png - Raw mask");
    println!("   transparency_debug_mask_binary.png - Binary mask");

    Ok(())
}
