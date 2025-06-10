use image::{DynamicImage, ImageFormat};
use std::path::Path;
use std::collections::HashMap;
use serde_json;

/// Compatibility test result
#[derive(Debug, Clone)]
pub struct CompatibilityResult {
    pub test_id: String,
    pub structural_similarity: f64,
    pub mean_pixel_difference: f64,
    pub dimensions_match: bool,
    pub format_compatible: bool,
    pub compatible: bool,
    pub error_message: Option<String>,
}

/// JavaScript output comparison configuration
#[derive(Debug, serde::Deserialize)]
pub struct JSComparisonConfig {
    pub comparison_method: String,
    pub tolerance: f64,
    pub ignore_compression_artifacts: bool,
    pub structural_similarity_threshold: f64,
}

/// Cross-platform compatibility configuration
#[derive(Debug, serde::Deserialize)]
pub struct CrossPlatformConfig {
    pub platforms: Vec<String>,
    pub deterministic_output: bool,
    pub floating_point_tolerance: f64,
}

/// Model version compatibility configuration
#[derive(Debug, serde::Deserialize)]
pub struct ModelVersionConfig {
    pub model_file: String,
    pub expected_accuracy: f64,
    pub performance_baseline: f64,
}

/// Validate JavaScript compatibility by comparing outputs
pub fn validate_js_compatibility(
    rust_output: &[u8],
    js_reference: &[u8],
    tolerance: f64,
    config: &JSComparisonConfig,
) -> CompatibilityResult {
    let test_id = "js_compatibility".to_string();
    
    // Load images
    let (rust_image, js_image) = match (
        image::load_from_memory(rust_output),
        image::load_from_memory(js_reference)
    ) {
        (Ok(rust_img), Ok(js_img)) => (rust_img, js_img),
        (Err(e), _) => return CompatibilityResult {
            test_id,
            structural_similarity: 0.0,
            mean_pixel_difference: 1000.0,
            dimensions_match: false,
            format_compatible: false,
            compatible: false,
            error_message: Some(format!("Failed to load Rust output: {}", e)),
        },
        (_, Err(e)) => return CompatibilityResult {
            test_id,
            structural_similarity: 0.0,
            mean_pixel_difference: 1000.0,
            dimensions_match: false,
            format_compatible: false,
            compatible: false,
            error_message: Some(format!("Failed to load JS reference: {}", e)),
        },
    };
    
    // Check dimensions
    let dimensions_match = rust_image.dimensions() == js_image.dimensions();
    
    if !dimensions_match {
        return CompatibilityResult {
            test_id,
            structural_similarity: 0.0,
            mean_pixel_difference: 1000.0,
            dimensions_match: false,
            format_compatible: false,
            compatible: false,
            error_message: Some(format!(
                "Dimension mismatch: Rust {:?} vs JS {:?}",
                rust_image.dimensions(),
                js_image.dimensions()
            )),
        };
    }
    
    // Convert to RGBA for consistent comparison
    let rust_rgba = rust_image.to_rgba8();
    let js_rgba = js_image.to_rgba8();
    
    // Calculate similarity metrics
    let structural_similarity = calculate_structural_similarity_detailed(&rust_rgba, &js_rgba);
    let mean_pixel_difference = calculate_mean_pixel_difference_detailed(&rust_rgba, &js_rgba, config);
    
    // Check compatibility based on thresholds
    let compatible = structural_similarity >= config.structural_similarity_threshold
        && mean_pixel_difference <= tolerance;
    
    CompatibilityResult {
        test_id,
        structural_similarity,
        mean_pixel_difference,
        dimensions_match: true,
        format_compatible: true,
        compatible,
        error_message: None,
    }
}

/// Calculate detailed structural similarity
fn calculate_structural_similarity_detailed(
    img1: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    img2: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> f64 {
    let (width, height) = img1.dimensions();
    let window_size = 11u32;
    let k1 = 0.01;
    let k2 = 0.03;
    let data_range = 255.0;
    
    let c1 = (k1 * data_range).powi(2);
    let c2 = (k2 * data_range).powi(2);
    
    let mut ssim_sum = 0.0;
    let mut window_count = 0;
    
    let half_window = window_size / 2;
    
    for y in half_window..(height - half_window) {
        for x in half_window..(width - half_window) {
            let (mu1, mu2, sigma1_sq, sigma2_sq, sigma12) = 
                calculate_window_statistics_detailed(img1, img2, x, y, window_size);
            
            let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
            let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);
            
            if denominator != 0.0 {
                ssim_sum += numerator / denominator;
                window_count += 1;
            }
        }
    }
    
    if window_count > 0 {
        ssim_sum / window_count as f64
    } else {
        0.0
    }
}

/// Calculate window statistics for SSIM
fn calculate_window_statistics_detailed(
    img1: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    img2: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    center_x: u32,
    center_y: u32,
    window_size: u32,
) -> (f64, f64, f64, f64, f64) {
    let half_window = window_size / 2;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum12 = 0.0;
    let mut count = 0;
    
    for y in (center_y - half_window)..=(center_y + half_window) {
        for x in (center_x - half_window)..=(center_x + half_window) {
            // Use alpha channel for transparency comparison
            let pixel1 = img1.get_pixel(x, y);
            let pixel2 = img2.get_pixel(x, y);
            
            // Weighted average of all channels
            let val1 = (pixel1[0] as f64 * 0.299 + pixel1[1] as f64 * 0.587 + pixel1[2] as f64 * 0.114) * (pixel1[3] as f64 / 255.0);
            let val2 = (pixel2[0] as f64 * 0.299 + pixel2[1] as f64 * 0.587 + pixel2[2] as f64 * 0.114) * (pixel2[3] as f64 / 255.0);
            
            sum1 += val1;
            sum2 += val2;
            sum1_sq += val1 * val1;
            sum2_sq += val2 * val2;
            sum12 += val1 * val2;
            count += 1;
        }
    }
    
    let n = count as f64;
    let mu1 = sum1 / n;
    let mu2 = sum2 / n;
    let sigma1_sq = (sum1_sq / n) - mu1.powi(2);
    let sigma2_sq = (sum2_sq / n) - mu2.powi(2);
    let sigma12 = (sum12 / n) - mu1 * mu2;
    
    (mu1, mu2, sigma1_sq, sigma2_sq, sigma12)
}

/// Calculate mean pixel difference with compression artifact handling
fn calculate_mean_pixel_difference_detailed(
    img1: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    img2: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    config: &JSComparisonConfig,
) -> f64 {
    let (width, height) = img1.dimensions();
    let mut sum_diff = 0.0;
    let mut count = 0;
    
    for y in 0..height {
        for x in 0..width {
            let pixel1 = img1.get_pixel(x, y);
            let pixel2 = img2.get_pixel(x, y);
            
            // Calculate difference for each channel
            for i in 0..4 {
                let mut diff = (pixel1[i] as f64 - pixel2[i] as f64).abs();
                
                // Apply compression artifact filtering if enabled
                if config.ignore_compression_artifacts && i < 3 { // RGB channels only
                    diff = filter_compression_artifacts(diff, pixel1[i], pixel2[i]);
                }
                
                sum_diff += diff;
                count += 1;
            }
        }
    }
    
    if count > 0 {
        sum_diff / count as f64
    } else {
        0.0
    }
}

/// Filter out compression artifacts from pixel differences
fn filter_compression_artifacts(diff: f64, val1: u8, val2: u8) -> f64 {
    // If both values are very dark or very bright, compression artifacts are more likely
    let min_val = val1.min(val2);
    let max_val = val1.max(val2);
    
    // Reduce weight of differences in extreme ranges where compression artifacts are common
    if min_val < 10 || max_val > 245 {
        diff * 0.5 // Reduce impact of differences in extreme ranges
    } else {
        diff
    }
}

/// Test cross-platform deterministic output
pub fn test_cross_platform_compatibility<P: AsRef<Path>>(
    outputs_by_platform: &HashMap<String, P>,
    config: &CrossPlatformConfig,
) -> Vec<CompatibilityResult> {
    let mut results = Vec::new();
    let platforms: Vec<_> = outputs_by_platform.keys().collect();
    
    // Compare each platform pair
    for i in 0..platforms.len() {
        for j in (i + 1)..platforms.len() {
            let platform1 = platforms[i];
            let platform2 = platforms[j];
            
            let output1_path = outputs_by_platform.get(platform1).unwrap().as_ref();
            let output2_path = outputs_by_platform.get(platform2).unwrap().as_ref();
            
            let result = compare_platform_outputs(
                platform1,
                platform2,
                output1_path,
                output2_path,
                config,
            );
            
            results.push(result);
        }
    }
    
    results
}

/// Compare outputs between two platforms
fn compare_platform_outputs(
    platform1: &str,
    platform2: &str,
    output1: &Path,
    output2: &Path,
    config: &CrossPlatformConfig,
) -> CompatibilityResult {
    let test_id = format!("cross_platform_{}_{}", platform1, platform2);
    
    // Load images
    let (img1, img2) = match (image::open(output1), image::open(output2)) {
        (Ok(img1), Ok(img2)) => (img1, img2),
        (Err(e), _) => return CompatibilityResult {
            test_id,
            structural_similarity: 0.0,
            mean_pixel_difference: 1000.0,
            dimensions_match: false,
            format_compatible: false,
            compatible: false,
            error_message: Some(format!("Failed to load {} output: {}", platform1, e)),
        },
        (_, Err(e)) => return CompatibilityResult {
            test_id,
            structural_similarity: 0.0,
            mean_pixel_difference: 1000.0,
            dimensions_match: false,
            format_compatible: false,
            compatible: false,
            error_message: Some(format!("Failed to load {} output: {}", platform2, e)),
        },
    };
    
    // Check deterministic output requirement
    if config.deterministic_output {
        // For deterministic output, images should be identical
        let img1_rgba = img1.to_rgba8();
        let img2_rgba = img2.to_rgba8();
        
        let dimensions_match = img1.dimensions() == img2.dimensions();
        
        if !dimensions_match {
            return CompatibilityResult {
                test_id,
                structural_similarity: 0.0,
                mean_pixel_difference: 1000.0,
                dimensions_match: false,
                format_compatible: false,
                compatible: false,
                error_message: Some("Dimensions don't match for deterministic output".to_string()),
            };
        }
        
        // Check if images are pixel-perfect identical
        let is_identical = img1_rgba.pixels().zip(img2_rgba.pixels()).all(|(p1, p2)| {
            p1.0.iter().zip(p2.0.iter()).all(|(c1, c2)| {
                (*c1 as f64 - *c2 as f64).abs() <= config.floating_point_tolerance
            })
        });
        
        if is_identical {
            CompatibilityResult {
                test_id,
                structural_similarity: 1.0,
                mean_pixel_difference: 0.0,
                dimensions_match: true,
                format_compatible: true,
                compatible: true,
                error_message: None,
            }
        } else {
            let mean_diff = calculate_mean_pixel_difference_simple(&img1_rgba, &img2_rgba);
            CompatibilityResult {
                test_id,
                structural_similarity: 0.0,
                mean_pixel_difference: mean_diff,
                dimensions_match: true,
                format_compatible: true,
                compatible: false,
                error_message: Some(format!(
                    "Outputs not deterministic: mean diff = {:.3}",
                    mean_diff
                )),
            }
        }
    } else {
        // For non-deterministic output, allow some tolerance
        let img1_rgba = img1.to_rgba8();
        let img2_rgba = img2.to_rgba8();
        
        let dimensions_match = img1.dimensions() == img2.dimensions();
        let mean_diff = calculate_mean_pixel_difference_simple(&img1_rgba, &img2_rgba);
        let ssim = calculate_structural_similarity_simple(&img1_rgba, &img2_rgba);
        
        let compatible = dimensions_match && mean_diff <= 5.0 && ssim >= 0.95;
        
        CompatibilityResult {
            test_id,
            structural_similarity: ssim,
            mean_pixel_difference: mean_diff,
            dimensions_match,
            format_compatible: true,
            compatible,
            error_message: None,
        }
    }
}

/// Simple pixel difference calculation
fn calculate_mean_pixel_difference_simple(
    img1: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    img2: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> f64 {
    if img1.dimensions() != img2.dimensions() {
        return 1000.0;
    }
    
    let mut sum_diff = 0.0;
    let mut count = 0;
    
    for (p1, p2) in img1.pixels().zip(img2.pixels()) {
        for i in 0..4 {
            sum_diff += (p1[i] as f64 - p2[i] as f64).abs();
            count += 1;
        }
    }
    
    if count > 0 {
        sum_diff / count as f64
    } else {
        0.0
    }
}

/// Simple structural similarity calculation
fn calculate_structural_similarity_simple(
    img1: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    img2: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> f64 {
    if img1.dimensions() != img2.dimensions() {
        return 0.0;
    }
    
    // Simplified SSIM using correlation coefficient
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum12 = 0.0;
    let mut count = 0;
    
    for (p1, p2) in img1.pixels().zip(img2.pixels()) {
        // Use alpha-weighted luminance
        let val1 = (p1[0] as f64 * 0.299 + p1[1] as f64 * 0.587 + p1[2] as f64 * 0.114) * (p1[3] as f64 / 255.0);
        let val2 = (p2[0] as f64 * 0.299 + p2[1] as f64 * 0.587 + p2[2] as f64 * 0.114) * (p2[3] as f64 / 255.0);
        
        sum1 += val1;
        sum2 += val2;
        sum1_sq += val1 * val1;
        sum2_sq += val2 * val2;
        sum12 += val1 * val2;
        count += 1;
    }
    
    if count == 0 {
        return 0.0;
    }
    
    let n = count as f64;
    let mean1 = sum1 / n;
    let mean2 = sum2 / n;
    
    let var1 = (sum1_sq / n) - mean1.powi(2);
    let var2 = (sum2_sq / n) - mean2.powi(2);
    let cov = (sum12 / n) - mean1 * mean2;
    
    let denom = (var1 * var2).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        1.0 // Perfect correlation if no variance
    }
}

/// Test model version compatibility
pub fn test_model_version_compatibility<P: AsRef<Path>>(
    test_images_dir: P,
    binary_path: P,
    model_configs: &HashMap<String, ModelVersionConfig>,
) -> HashMap<String, Vec<CompatibilityResult>> {
    let mut results = HashMap::new();
    
    for (model_name, model_config) in model_configs {
        println!("Testing model version: {}", model_name);
        
        let mut model_results = Vec::new();
        
        // Test each image with this model version
        let test_images = find_test_images(test_images_dir.as_ref());
        
        for test_image in test_images {
            let output_file = format!("{}_{}_output.png", 
                test_image.file_stem().unwrap().to_str().unwrap(),
                model_name);
            
            // Run with specific model (hypothetical CLI flag)
            let mut cmd = std::process::Command::new(binary_path.as_ref());
            cmd.arg(&test_image)
               .arg("--output").arg(&output_file)
               .arg("--model").arg(&model_config.model_file);
            
            match cmd.output() {
                Ok(output) => {
                    let success = output.status.success() && Path::new(&output_file).exists();
                    
                    let result = CompatibilityResult {
                        test_id: format!("{}_{}", model_name, test_image.file_stem().unwrap().to_str().unwrap()),
                        structural_similarity: if success { 1.0 } else { 0.0 },
                        mean_pixel_difference: if success { 0.0 } else { 1000.0 },
                        dimensions_match: success,
                        format_compatible: success,
                        compatible: success,
                        error_message: if success { None } else { 
                            Some(String::from_utf8_lossy(&output.stderr).to_string())
                        },
                    };
                    
                    model_results.push(result);
                }
                Err(e) => {
                    model_results.push(CompatibilityResult {
                        test_id: format!("{}_{}", model_name, test_image.file_stem().unwrap().to_str().unwrap()),
                        structural_similarity: 0.0,
                        mean_pixel_difference: 1000.0,
                        dimensions_match: false,
                        format_compatible: false,
                        compatible: false,
                        error_message: Some(format!("Failed to run command: {}", e)),
                    });
                }
            }
        }
        
        results.insert(model_name.clone(), model_results);
    }
    
    results
}

/// Find test images in directory
fn find_test_images(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut images = Vec::new();
    
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "jpg" || ext == "png" || ext == "jpeg" {
                    images.push(path);
                }
            }
        }
    }
    
    // Also check subdirectories
    for category in &["portraits", "products", "complex", "edge_cases"] {
        let category_dir = dir.join(category);
        if category_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&category_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if ext == "jpg" || ext == "png" || ext == "jpeg" {
                            images.push(path);
                        }
                    }
                }
            }
        }
    }
    
    images
}

/// Generate compatibility report
pub fn generate_compatibility_report(
    js_results: &[CompatibilityResult],
    cross_platform_results: &[CompatibilityResult],
    model_results: &HashMap<String, Vec<CompatibilityResult>>,
    output_file: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    
    let mut report = String::new();
    report.push_str("# Compatibility Test Report\n\n");
    
    // JavaScript compatibility results
    report.push_str("## JavaScript Compatibility\n\n");
    let js_success_rate = js_results.iter().filter(|r| r.compatible).count() as f64 / js_results.len() as f64 * 100.0;
    report.push_str(&format!("**Success Rate:** {:.1}%\n\n", js_success_rate));
    
    report.push_str("| Test ID | SSIM | Mean Diff | Compatible | Error |\n");
    report.push_str("|---------|------|-----------|------------|-------|\n");
    
    for result in js_results {
        let error = result.error_message.as_deref().unwrap_or("None");
        report.push_str(&format!(
            "| {} | {:.3} | {:.1} | {} | {} |\n",
            result.test_id,
            result.structural_similarity,
            result.mean_pixel_difference,
            if result.compatible { "✓" } else { "✗" },
            error
        ));
    }
    
    // Cross-platform compatibility
    if !cross_platform_results.is_empty() {
        report.push_str("\n## Cross-Platform Compatibility\n\n");
        let cp_success_rate = cross_platform_results.iter().filter(|r| r.compatible).count() as f64 / cross_platform_results.len() as f64 * 100.0;
        report.push_str(&format!("**Success Rate:** {:.1}%\n\n", cp_success_rate));
        
        report.push_str("| Platform Pair | SSIM | Mean Diff | Compatible | Error |\n");
        report.push_str("|---------------|------|-----------|------------|-------|\n");
        
        for result in cross_platform_results {
            let error = result.error_message.as_deref().unwrap_or("None");
            report.push_str(&format!(
                "| {} | {:.3} | {:.1} | {} | {} |\n",
                result.test_id,
                result.structural_similarity,
                result.mean_pixel_difference,
                if result.compatible { "✓" } else { "✗" },
                error
            ));
        }
    }
    
    // Model version compatibility
    if !model_results.is_empty() {
        report.push_str("\n## Model Version Compatibility\n\n");
        
        for (model_name, results) in model_results {
            let success_rate = results.iter().filter(|r| r.compatible).count() as f64 / results.len() as f64 * 100.0;
            report.push_str(&format!("### {} (Success Rate: {:.1}%)\n\n", model_name, success_rate));
            
            report.push_str("| Test | Compatible | Error |\n");
            report.push_str("|------|------------|-------|\n");
            
            for result in results {
                let error = result.error_message.as_deref().unwrap_or("None");
                report.push_str(&format!(
                    "| {} | {} | {} |\n",
                    result.test_id,
                    if result.compatible { "✓" } else { "✗" },
                    error
                ));
            }
            
            report.push_str("\n");
        }
    }
    
    // Write report
    let mut file = std::fs::File::create(output_file)?;
    file.write_all(report.as_bytes())?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    
    #[test]
    fn test_identical_images_compatibility() {
        let img = ImageBuffer::from_fn(100, 100, |x, y| {
            if x < 50 { Rgba([255, 255, 255, 255]) } else { Rgba([0, 0, 0, 0]) }
        });
        
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        
        // Save as PNG bytes
        img.save_with_format(&mut std::io::Cursor::new(&mut buf1), ImageFormat::Png).unwrap();
        img.save_with_format(&mut std::io::Cursor::new(&mut buf2), ImageFormat::Png).unwrap();
        
        let config = JSComparisonConfig {
            comparison_method: "pixel_by_pixel".to_string(),
            tolerance: 5.0,
            ignore_compression_artifacts: false,
            structural_similarity_threshold: 0.95,
        };
        
        let result = validate_js_compatibility(&buf1, &buf2, 5.0, &config);
        
        assert!(result.compatible);
        assert!(result.structural_similarity > 0.99);
        assert!(result.mean_pixel_difference < 1.0);
    }
    
    #[test]
    fn test_compression_artifact_filtering() {
        // Test with extreme values where compression artifacts are common
        let diff_dark = filter_compression_artifacts(10.0, 5, 15);
        let diff_bright = filter_compression_artifacts(10.0, 245, 255);
        let diff_normal = filter_compression_artifacts(10.0, 100, 110);
        
        // Differences in extreme ranges should be reduced
        assert!(diff_dark < 10.0);
        assert!(diff_bright < 10.0);
        assert_eq!(diff_normal, 10.0); // Normal range unchanged
    }
}