//! Quantitative mask comparison with JavaScript reference

use bg_remove_core::{RemovalConfig, remove_background};
use std::path::Path;
use image::{DynamicImage, GrayImage};

#[derive(Debug)]
struct ComparisonMetrics {
    pixel_accuracy: f64,
    mean_absolute_error: f64,
    structural_similarity: f64,
    intersection_over_union: f64,
    dice_coefficient: f64,
    foreground_ratio_diff: f64,
}

fn load_js_mask(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("JavaScript mask not found: {}", path).into());
    }
    
    let js_mask_img = image::open(path)?;
    let gray_img = js_mask_img.to_luma8();
    Ok(gray_img.into_raw())
}

fn calculate_metrics(rust_mask: &[u8], js_mask: &[u8]) -> ComparisonMetrics {
    assert_eq!(rust_mask.len(), js_mask.len());
    
    let total_pixels = rust_mask.len() as f64;
    
    // Pixel accuracy
    let correct_pixels = rust_mask.iter()
        .zip(js_mask.iter())
        .filter(|(&r, &j)| (r > 127) == (j > 127))
        .count() as f64;
    let pixel_accuracy = correct_pixels / total_pixels;
    
    // Mean Absolute Error
    let mae = rust_mask.iter()
        .zip(js_mask.iter())
        .map(|(&r, &j)| (r as i32 - j as i32).abs() as f64)
        .sum::<f64>() / total_pixels;
    
    // Binary masks for IoU and Dice
    let rust_binary: Vec<bool> = rust_mask.iter().map(|&x| x > 127).collect();
    let js_binary: Vec<bool> = js_mask.iter().map(|&x| x > 127).collect();
    
    // Intersection over Union (IoU)
    let intersection = rust_binary.iter()
        .zip(js_binary.iter())
        .filter(|(&r, &j)| r && j)
        .count() as f64;
    let union = rust_binary.iter()
        .zip(js_binary.iter())
        .filter(|(&r, &j)| r || j)
        .count() as f64;
    let iou = if union > 0.0 { intersection / union } else { 1.0 };
    
    // Dice Coefficient
    let rust_positives = rust_binary.iter().filter(|&&x| x).count() as f64;
    let js_positives = js_binary.iter().filter(|&&x| x).count() as f64;
    let dice = if rust_positives + js_positives > 0.0 {
        (2.0 * intersection) / (rust_positives + js_positives)
    } else {
        1.0
    };
    
    // Foreground ratio difference
    let rust_fg_ratio = rust_positives / total_pixels;
    let js_fg_ratio = js_positives / total_pixels;
    let fg_ratio_diff = (rust_fg_ratio - js_fg_ratio).abs();
    
    // Simple structural similarity (correlation coefficient)
    let rust_mean = rust_mask.iter().map(|&x| x as f64).sum::<f64>() / total_pixels;
    let js_mean = js_mask.iter().map(|&x| x as f64).sum::<f64>() / total_pixels;
    
    let numerator = rust_mask.iter()
        .zip(js_mask.iter())
        .map(|(&r, &j)| (r as f64 - rust_mean) * (j as f64 - js_mean))
        .sum::<f64>();
    
    let rust_var = rust_mask.iter()
        .map(|&x| (x as f64 - rust_mean).powi(2))
        .sum::<f64>();
    let js_var = js_mask.iter()
        .map(|&x| (x as f64 - js_mean).powi(2))
        .sum::<f64>();
    
    let structural_similarity = if rust_var > 0.0 && js_var > 0.0 {
        numerator / (rust_var * js_var).sqrt()
    } else {
        0.0
    };
    
    ComparisonMetrics {
        pixel_accuracy,
        mean_absolute_error: mae,
        structural_similarity,
        intersection_over_union: iou,
        dice_coefficient: dice,
        foreground_ratio_diff: fg_ratio_diff,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Quantitative Mask Comparison with JavaScript Reference");
    println!("========================================================");
    
    let test_cases = [
        ("portraits", "portrait_single_simple_bg.jpg"),
        ("portraits", "portrait_multiple_people.jpg"),
        ("products", "product_clothing_white_bg.jpg"),
        ("complex", "complex_group_photo.jpg"),
        ("edge_cases", "edge_high_contrast.jpg"),
    ];
    
    let config = RemovalConfig::builder()
        .debug(false)
        .build()?;
    
    let mut total_metrics = ComparisonMetrics {
        pixel_accuracy: 0.0,
        mean_absolute_error: 0.0,
        structural_similarity: 0.0,
        intersection_over_union: 0.0,
        dice_coefficient: 0.0,
        foreground_ratio_diff: 0.0,
    };
    
    let mut successful_comparisons = 0;
    
    for (category, image_name) in &test_cases {
        let input_path = format!("tests/assets/input/{}/{}", category, image_name);
        let js_mask_path = format!("tests/assets/expected/masks/js_{}_mask.png", 
            image_name.trim_end_matches(".jpg"));
        
        if !Path::new(&input_path).exists() || !Path::new(&js_mask_path).exists() {
            println!("⏭️  Skipping {}: Missing files", image_name);
            continue;
        }
        
        print!("🧪 Analyzing {}... ", image_name);
        
        // Process with Rust
        let result = remove_background(&input_path, &config).await?;
        let rust_mask_data = &result.mask.data;
        
        // Load JavaScript reference mask
        let js_mask_data = load_js_mask(&js_mask_path)?;
        
        if rust_mask_data.len() != js_mask_data.len() {
            println!("❌ Size mismatch: Rust {} vs JS {}", rust_mask_data.len(), js_mask_data.len());
            continue;
        }
        
        // Calculate metrics
        let metrics = calculate_metrics(rust_mask_data, &js_mask_data);
        
        println!("✅");
        println!("   📊 Pixel Accuracy: {:.1}%", metrics.pixel_accuracy * 100.0);
        println!("   📊 IoU: {:.3}", metrics.intersection_over_union);
        println!("   📊 Dice: {:.3}", metrics.dice_coefficient);
        println!("   📊 MAE: {:.1}", metrics.mean_absolute_error);
        println!("   📊 Structural Sim: {:.3}", metrics.structural_similarity);
        println!("   📊 FG Ratio Diff: {:.1}%", metrics.foreground_ratio_diff * 100.0);
        
        // Accumulate for averages
        total_metrics.pixel_accuracy += metrics.pixel_accuracy;
        total_metrics.mean_absolute_error += metrics.mean_absolute_error;
        total_metrics.structural_similarity += metrics.structural_similarity;
        total_metrics.intersection_over_union += metrics.intersection_over_union;
        total_metrics.dice_coefficient += metrics.dice_coefficient;
        total_metrics.foreground_ratio_diff += metrics.foreground_ratio_diff;
        
        successful_comparisons += 1;
        println!();
    }
    
    if successful_comparisons > 0 {
        let n = successful_comparisons as f64;
        println!("📈 Average Metrics Across {} Images:", successful_comparisons);
        println!("   🎯 Pixel Accuracy: {:.1}%", (total_metrics.pixel_accuracy / n) * 100.0);
        println!("   🎯 IoU (Intersection over Union): {:.3}", total_metrics.intersection_over_union / n);
        println!("   🎯 Dice Coefficient: {:.3}", total_metrics.dice_coefficient / n);
        println!("   🎯 Mean Absolute Error: {:.1}", total_metrics.mean_absolute_error / n);
        println!("   🎯 Structural Similarity: {:.3}", total_metrics.structural_similarity / n);
        println!("   🎯 Foreground Ratio Difference: {:.1}%", (total_metrics.foreground_ratio_diff / n) * 100.0);
        
        // Quality assessment
        let avg_iou = total_metrics.intersection_over_union / n;
        let avg_dice = total_metrics.dice_coefficient / n;
        let avg_accuracy = total_metrics.pixel_accuracy / n;
        
        println!("\n🏆 Quality Assessment:");
        if avg_iou > 0.9 && avg_dice > 0.95 && avg_accuracy > 0.95 {
            println!("   ✅ EXCELLENT - Near-perfect match with JavaScript");
        } else if avg_iou > 0.8 && avg_dice > 0.85 && avg_accuracy > 0.90 {
            println!("   🎯 VERY GOOD - High quality match");
        } else if avg_iou > 0.7 && avg_dice > 0.8 && avg_accuracy > 0.85 {
            println!("   📈 GOOD - Acceptable quality");
        } else {
            println!("   ⚠️  NEEDS IMPROVEMENT - Significant differences detected");
        }
    }
    
    Ok(())
}