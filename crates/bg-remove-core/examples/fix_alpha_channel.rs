//! Fix alpha channel application to match JavaScript version

use bg_remove_core::{RemovalConfig, remove_background};
use bg_remove_core::config::ModelPrecision;
use std::path::Path;
use image::{DynamicImage, RgbaImage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Fixing Alpha Channel Application");
    println!("==================================");
    
    let test_image = "../tests/assets/input/portraits/portrait_single_simple_bg.jpg";
    let js_reference = "../tests/assets/expected/javascript_output/portrait_single_simple_bg.png";
    
    if !Path::new(test_image).exists() {
        println!("❌ Test image not found: {}", test_image);
        return Ok(());
    }

    let config = RemovalConfig::builder()
        .model_precision(ModelPrecision::Fp16)
        .debug(false)
        .build()?;

    // Process with our current implementation
    println!("🧪 Processing with current implementation...");
    let result = remove_background(test_image, &config).await?;
    
    // Analyze current mask values
    let mask_data = &result.mask.data;
    let mut value_histogram = std::collections::HashMap::new();
    for &value in mask_data {
        *value_histogram.entry(value).or_insert(0) += 1;
    }
    
    println!("📊 Current mask value analysis:");
    let mut sorted_values: Vec<_> = value_histogram.iter().collect();
    sorted_values.sort_by_key(|&(value, _)| value);
    
    let min_val = *sorted_values.first().unwrap().0;
    let max_val = *sorted_values.last().unwrap().0;
    println!("   Range: {} to {}", min_val, max_val);
    
    // Show most common values
    let mut by_count: Vec<_> = value_histogram.iter().collect();
    by_count.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
    println!("   Most common values:");
    for (value, count) in by_count.iter().take(5) {
        let percentage = (**count as f64) / (mask_data.len() as f64) * 100.0;
        println!("     Value {}: {:.1}%", value, percentage);
    }
    
    // Create corrected alpha application
    println!("\n🔧 Creating corrected alpha channel version...");
    
    // Load original image
    let original_image = image::open(test_image)?;
    let mut rgba_image = original_image.to_rgba8();
    
    // Apply mask values directly as alpha (this should already be correct)
    result.mask.apply_to_image(&mut rgba_image)?;
    
    // Save current version
    let current_output = DynamicImage::ImageRgba8(rgba_image.clone());
    current_output.save("alpha_fix_current.png")?;
    
    // Try alternative interpretation: What if we need to invert the mask?
    println!("🔄 Testing inverted mask interpretation...");
    let original_image2 = image::open(test_image)?;
    let mut rgba_image2 = original_image2.to_rgba8();
    
    // Apply inverted mask
    for (i, pixel) in rgba_image2.pixels_mut().enumerate() {
        if i < mask_data.len() {
            let inverted_alpha = 255 - mask_data[i]; // Invert the mask
            pixel[3] = inverted_alpha;
        }
    }
    
    let inverted_output = DynamicImage::ImageRgba8(rgba_image2);
    inverted_output.save("alpha_fix_inverted.png")?;
    
    // Analyze JavaScript reference if available
    if Path::new(js_reference).exists() {
        println!("\n📊 Analyzing JavaScript reference...");
        let js_image = image::open(js_reference)?;
        
        if let DynamicImage::ImageRgba8(js_rgba) = js_image {
            let js_alpha_values: Vec<u8> = js_rgba.pixels().map(|p| p[3]).collect();
            
            let mut js_histogram = std::collections::HashMap::new();
            for &alpha in &js_alpha_values {
                *js_histogram.entry(alpha).or_insert(0) += 1;
            }
            
            let js_min = *js_alpha_values.iter().min().unwrap();
            let js_max = *js_alpha_values.iter().max().unwrap();
            println!("   JS Alpha range: {} to {}", js_min, js_max);
            
            let mut js_by_count: Vec<_> = js_histogram.iter().collect();
            js_by_count.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
            println!("   JS Most common alpha values:");
            for (alpha, count) in js_by_count.iter().take(5) {
                let percentage = (**count as f64) / (js_alpha_values.len() as f64) * 100.0;
                println!("     Alpha {}: {:.1}%", alpha, percentage);
            }
            
            // Compare distributions
            let transparent_pixels = js_alpha_values.iter().filter(|&&x| x == 0).count();
            let opaque_pixels = js_alpha_values.iter().filter(|&&x| x == 255).count();
            let partial_pixels = js_alpha_values.len() - transparent_pixels - opaque_pixels;
            
            println!("   JS Transparency distribution:");
            println!("     Transparent (0): {:.1}%", (transparent_pixels as f64) / (js_alpha_values.len() as f64) * 100.0);
            println!("     Opaque (255): {:.1}%", (opaque_pixels as f64) / (js_alpha_values.len() as f64) * 100.0);
            println!("     Partial: {:.1}%", (partial_pixels as f64) / (js_alpha_values.len() as f64) * 100.0);
        }
    }
    
    println!("\n💾 Outputs saved:");
    println!("   alpha_fix_current.png - Current implementation");
    println!("   alpha_fix_inverted.png - Inverted mask interpretation");
    println!("   Compare these with the JavaScript reference");
    
    Ok(())
}