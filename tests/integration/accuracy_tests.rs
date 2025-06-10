use image::{DynamicImage, ImageBuffer, Rgba};
use std::path::Path;
use serde_json;
use std::collections::HashMap;

/// Accuracy metrics for validation
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    pub pixel_accuracy: f32,
    pub edge_accuracy: f32,
    pub structural_similarity: f32,
    pub mean_pixel_difference: f32,
    pub acceptable: bool,
}

/// Test case specification loaded from metadata
#[derive(Debug, serde::Deserialize)]
pub struct TestCaseSpec {
    pub id: String,
    pub name: String,
    pub input_file: String,
    pub validation_criteria: ValidationCriteria,
    pub expected_outputs: HashMap<String, String>,
}

/// Validation criteria for test cases
#[derive(Debug, serde::Deserialize)]
pub struct ValidationCriteria {
    pub pixel_accuracy_min: f32,
    pub edge_accuracy_min: f32,
    pub processing_time_max_ms: u64,
}

/// Test configuration loaded from metadata
#[derive(Debug, serde::Deserialize)]
pub struct TestConfig {
    pub categories: HashMap<String, CategoryConfig>,
}

/// Category configuration
#[derive(Debug, serde::Deserialize)]
pub struct CategoryConfig {
    pub description: String,
    pub target_accuracy: f32,
    pub edge_accuracy_threshold: f32,
    pub performance_baseline_ms: u64,
    pub test_cases: Vec<TestCaseSpec>,
}

/// Main accuracy validation function
pub fn validate_mask_accuracy(
    result_mask: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    ground_truth: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    tolerance: f32,
) -> AccuracyMetrics {
    let mut correct_pixels = 0;
    let mut total_pixels = 0;
    let mut edge_accuracy_sum = 0.0;
    let mut edge_pixel_count = 0;
    
    // Ensure images have same dimensions
    if result_mask.dimensions() != ground_truth.dimensions() {
        panic!("Image dimensions mismatch: {:?} vs {:?}", 
               result_mask.dimensions(), ground_truth.dimensions());
    }
    
    let (width, height) = result_mask.dimensions();
    
    for y in 0..height {
        for x in 0..width {
            total_pixels += 1;
            
            let result_pixel = result_mask.get_pixel(x, y);
            let truth_pixel = ground_truth.get_pixel(x, y);
            
            // Compare alpha channel (assuming RGBA format)
            let result_alpha = result_pixel[3] as f32;
            let truth_alpha = truth_pixel[3] as f32;
            
            let diff = (result_alpha - truth_alpha).abs();
            
            if diff <= (tolerance * 255.0) {
                correct_pixels += 1;
            }
            
            // Calculate edge accuracy for edge pixels
            if is_edge_pixel(ground_truth, x, y) {
                edge_pixel_count += 1;
                edge_accuracy_sum += calculate_edge_accuracy_at_pixel(result_pixel, truth_pixel, tolerance);
            }
        }
    }
    
    let pixel_accuracy = correct_pixels as f32 / total_pixels as f32;
    let edge_accuracy = if edge_pixel_count > 0 {
        edge_accuracy_sum / edge_pixel_count as f32
    } else {
        pixel_accuracy // Fallback if no edges detected
    };
    
    // Calculate structural similarity
    let ssim = calculate_structural_similarity(result_mask, ground_truth);
    
    // Calculate mean pixel difference
    let mean_diff = calculate_mean_pixel_difference(result_mask, ground_truth);
    
    AccuracyMetrics {
        pixel_accuracy,
        edge_accuracy,
        structural_similarity: ssim,
        mean_pixel_difference: mean_diff,
        acceptable: pixel_accuracy > 0.95 && edge_accuracy > 0.90,
    }
}

/// Check if a pixel is an edge pixel by examining neighbors
fn is_edge_pixel(image: &ImageBuffer<Rgba<u8>, Vec<u8>>, x: u32, y: u32) -> bool {
    let (width, height) = image.dimensions();
    
    // Skip border pixels to avoid bounds checking
    if x == 0 || y == 0 || x >= width - 1 || y >= height - 1 {
        return false;
    }
    
    let center_alpha = image.get_pixel(x, y)[3];
    let threshold = 50u8; // Alpha difference threshold for edge detection
    
    // Check 8-connected neighbors
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            
            let nx = (x as i32 + dx) as u32;
            let ny = (y as i32 + dy) as u32;
            
            let neighbor_alpha = image.get_pixel(nx, ny)[3];
            
            if (center_alpha as i32 - neighbor_alpha as i32).abs() > threshold as i32 {
                return true;
            }
        }
    }
    
    false
}

/// Calculate edge accuracy at a specific pixel
fn calculate_edge_accuracy_at_pixel(
    result_pixel: &Rgba<u8>,
    truth_pixel: &Rgba<u8>,
    tolerance: f32,
) -> f32 {
    let result_alpha = result_pixel[3] as f32;
    let truth_alpha = truth_pixel[3] as f32;
    
    let diff = (result_alpha - truth_alpha).abs();
    
    if diff <= (tolerance * 255.0) {
        1.0
    } else {
        // Gradual falloff for near-misses
        let normalized_diff = diff / 255.0;
        (1.0 - normalized_diff).max(0.0)
    }
}

/// Calculate structural similarity between two images
fn calculate_structural_similarity(
    img1: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    img2: &ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> f32 {
    // Simplified SSIM calculation focusing on alpha channel
    let (width, height) = img1.dimensions();
    let window_size = 11u32;
    let k1 = 0.01f32;
    let k2 = 0.03f32;
    let data_range = 255.0f32;
    
    let c1 = (k1 * data_range).powi(2);
    let c2 = (k2 * data_range).powi(2);
    
    let mut ssim_sum = 0.0f32;
    let mut window_count = 0;
    
    for y in (window_size / 2)..(height - window_size / 2) {
        for x in (window_size / 2)..(width - window_size / 2) {
            let (mu1, mu2, sigma1_sq, sigma2_sq, sigma12) = 
                calculate_window_statistics(img1, img2, x, y, window_size);
            
            let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
            let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);
            
            ssim_sum += numerator / denominator;
            window_count += 1;
        }
    }
    
    if window_count > 0 {
        ssim_sum / window_count as f32
    } else {
        1.0 // Fallback for very small images
    }
}

/// Calculate statistics for SSIM window
fn calculate_window_statistics(
    img1: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    img2: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    center_x: u32,
    center_y: u32,
    window_size: u32,
) -> (f32, f32, f32, f32, f32) {
    let half_window = window_size / 2;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum1_sq = 0.0f32;
    let mut sum2_sq = 0.0f32;
    let mut sum12 = 0.0f32;
    let mut count = 0;
    
    for y in (center_y - half_window)..=(center_y + half_window) {
        for x in (center_x - half_window)..=(center_x + half_window) {
            let val1 = img1.get_pixel(x, y)[3] as f32;
            let val2 = img2.get_pixel(x, y)[3] as f32;
            
            sum1 += val1;
            sum2 += val2;
            sum1_sq += val1 * val1;
            sum2_sq += val2 * val2;
            sum12 += val1 * val2;
            count += 1;
        }
    }
    
    let n = count as f32;
    let mu1 = sum1 / n;
    let mu2 = sum2 / n;
    let sigma1_sq = (sum1_sq / n) - mu1.powi(2);
    let sigma2_sq = (sum2_sq / n) - mu2.powi(2);
    let sigma12 = (sum12 / n) - mu1 * mu2;
    
    (mu1, mu2, sigma1_sq, sigma2_sq, sigma12)
}

/// Calculate mean pixel difference between images
fn calculate_mean_pixel_difference(
    img1: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    img2: &ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> f32 {
    let mut sum_diff = 0.0f32;
    let mut count = 0;
    
    let (width, height) = img1.dimensions();
    
    for y in 0..height {
        for x in 0..width {
            let pixel1 = img1.get_pixel(x, y);
            let pixel2 = img2.get_pixel(x, y);
            
            // Calculate difference across all channels
            for i in 0..4 {
                let diff = (pixel1[i] as f32 - pixel2[i] as f32).abs();
                sum_diff += diff;
                count += 1;
            }
        }
    }
    
    if count > 0 {
        sum_diff / count as f32
    } else {
        0.0
    }
}

/// Load test configuration from JSON file
pub fn load_test_config<P: AsRef<Path>>(config_path: P) -> Result<TestConfig, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(config_path)?;
    let config: TestConfig = serde_json::from_str(&content)?;
    Ok(config)
}

/// Validate JavaScript compatibility by comparing outputs
pub fn validate_js_compatibility(
    rust_output: &[u8],
    js_reference: &[u8],
    tolerance: f32,
) -> Result<AccuracyMetrics, Box<dyn std::error::Error>> {
    let rust_image = image::load_from_memory(rust_output)?;
    let js_image = image::load_from_memory(js_reference)?;
    
    // Ensure dimensions match
    if rust_image.dimensions() != js_image.dimensions() {
        return Err(format!(
            "Dimension mismatch: Rust {:?} vs JS {:?}",
            rust_image.dimensions(),
            js_image.dimensions()
        ).into());
    }
    
    // Convert to RGBA for consistent comparison
    let rust_rgba = rust_image.to_rgba8();
    let js_rgba = js_image.to_rgba8();
    
    Ok(validate_mask_accuracy(&rust_rgba, &js_rgba, tolerance))
}

/// Run comprehensive accuracy tests for a category
pub fn run_accuracy_tests_for_category(
    category: &str,
    category_config: &CategoryConfig,
    rust_output_dir: &Path,
    reference_dir: &Path,
) -> Vec<(String, AccuracyMetrics)> {
    let mut results = Vec::new();
    
    for test_case in &category_config.test_cases {
        println!("Running accuracy test: {} - {}", test_case.id, test_case.name);
        
        // Find Rust output file
        let rust_output_path = rust_output_dir.join(format!("{}_alpha.png", 
            Path::new(&test_case.input_file).file_stem().unwrap().to_str().unwrap()));
        
        // Find reference output file
        let reference_path = reference_dir.join("javascript_output")
            .join(format!("{}.png", 
                Path::new(&test_case.input_file).file_stem().unwrap().to_str().unwrap()));
        
        if !rust_output_path.exists() {
            eprintln!("Rust output not found: {:?}", rust_output_path);
            continue;
        }
        
        if !reference_path.exists() {
            eprintln!("Reference output not found: {:?}", reference_path);
            continue;
        }
        
        // Load images and compare
        match (image::open(&rust_output_path), image::open(&reference_path)) {
            (Ok(rust_img), Ok(ref_img)) => {
                let rust_rgba = rust_img.to_rgba8();
                let ref_rgba = ref_img.to_rgba8();
                
                let tolerance = 0.02; // 2% tolerance
                let metrics = validate_mask_accuracy(&rust_rgba, &ref_rgba, tolerance);
                
                println!("  Pixel accuracy: {:.3}", metrics.pixel_accuracy);
                println!("  Edge accuracy: {:.3}", metrics.edge_accuracy);
                println!("  SSIM: {:.3}", metrics.structural_similarity);
                println!("  Acceptable: {}", metrics.acceptable);
                
                results.push((test_case.id.clone(), metrics));
            },
            (Err(e1), _) => eprintln!("Failed to load Rust output: {}", e1),
            (_, Err(e2)) => eprintln!("Failed to load reference output: {}", e2),
        }
    }
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    
    #[test]
    fn test_perfect_match() {
        let img1 = ImageBuffer::from_fn(100, 100, |x, y| {
            if x < 50 { Rgba([255, 255, 255, 255]) } else { Rgba([0, 0, 0, 0]) }
        });
        let img2 = img1.clone();
        
        let metrics = validate_mask_accuracy(&img1, &img2, 0.0);
        
        assert_eq!(metrics.pixel_accuracy, 1.0);
        assert!(metrics.acceptable);
    }
    
    #[test]
    fn test_tolerance_handling() {
        let img1 = ImageBuffer::from_fn(100, 100, |x, y| {
            if x < 50 { Rgba([255, 255, 255, 255]) } else { Rgba([0, 0, 0, 0]) }
        });
        let img2 = ImageBuffer::from_fn(100, 100, |x, y| {
            if x < 50 { Rgba([255, 255, 255, 250]) } else { Rgba([0, 0, 0, 5]) }
        });
        
        let metrics = validate_mask_accuracy(&img1, &img2, 0.02); // 2% tolerance
        
        assert!(metrics.pixel_accuracy > 0.9);
    }
    
    #[test]
    fn test_edge_detection() {
        let img = ImageBuffer::from_fn(10, 10, |x, y| {
            if x == 5 { Rgba([255, 255, 255, 255]) } else { Rgba([0, 0, 0, 0]) }
        });
        
        // Test pixels near the edge
        assert!(is_edge_pixel(&img, 4, 5));
        assert!(is_edge_pixel(&img, 6, 5));
        
        // Test pixels far from edge
        assert!(!is_edge_pixel(&img, 2, 5));
        assert!(!is_edge_pixel(&img, 8, 5));
    }
    
    #[test]
    fn test_mean_pixel_difference() {
        let img1 = ImageBuffer::from_fn(10, 10, |x, y| Rgba([255, 255, 255, 255]));
        let img2 = ImageBuffer::from_fn(10, 10, |x, y| Rgba([250, 250, 250, 250]));
        
        let diff = calculate_mean_pixel_difference(&img1, &img2);
        assert_eq!(diff, 5.0); // 5 unit difference across all channels
    }
}