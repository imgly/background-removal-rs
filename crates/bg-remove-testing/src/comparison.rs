//! Image comparison and accuracy metrics

use crate::{TestMetrics, TestingError, Result};
use image::{DynamicImage, GrayImage, Luma, RgbaImage, GenericImageView};
use std::cmp;

/// Image comparison utilities for testing accuracy
pub struct ImageComparison;

impl ImageComparison {
    /// Calculate pixel-level accuracy between two images using premultiplied alpha
    /// Returns percentage of pixels that match within threshold
    pub fn pixel_accuracy(
        actual: &DynamicImage,
        expected: &DynamicImage,
        threshold: f64,
    ) -> Result<f64> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for pixel accuracy comparison".to_string(),
            ));
        }

        let actual_rgba = actual.to_rgba8();
        let expected_rgba = expected.to_rgba8();

        let total_pixels = (actual.width() * actual.height()) as f64;
        let mut matching_pixels = 0.0;

        for (actual_pixel, expected_pixel) in actual_rgba.pixels().zip(expected_rgba.pixels()) {
            if Self::pixels_match_premultiplied(actual_pixel, expected_pixel, threshold) {
                matching_pixels += 1.0;
            }
        }

        Ok(matching_pixels / total_pixels)
    }

    /// Check if two pixels match within threshold using premultiplied alpha
    fn pixels_match_premultiplied(
        actual: &image::Rgba<u8>,
        expected: &image::Rgba<u8>,
        threshold: f64,
    ) -> bool {
        let actual_alpha = actual[3] as f64 / 255.0;
        let expected_alpha = expected[3] as f64 / 255.0;
        
        // Premultiply RGB by alpha
        let actual_r = (actual[0] as f64) * actual_alpha;
        let actual_g = (actual[1] as f64) * actual_alpha;
        let actual_b = (actual[2] as f64) * actual_alpha;
        
        let expected_r = (expected[0] as f64) * expected_alpha;
        let expected_g = (expected[1] as f64) * expected_alpha;
        let expected_b = (expected[2] as f64) * expected_alpha;
        
        // Calculate differences in premultiplied space
        let r_diff = (actual_r - expected_r).abs();
        let g_diff = (actual_g - expected_g).abs();
        let b_diff = (actual_b - expected_b).abs();
        let a_diff = (actual_alpha - expected_alpha).abs() * 255.0;
        
        let threshold_scaled = threshold * 255.0;
        
        r_diff <= threshold_scaled && g_diff <= threshold_scaled && 
        b_diff <= threshold_scaled && a_diff <= threshold_scaled
    }

    /// Calculate Structural Similarity Index (SSIM) between two images using premultiplied alpha
    pub fn structural_similarity(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<f64> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for SSIM calculation".to_string(),
            ));
        }

        // Convert to premultiplied grayscale for SSIM calculation
        let actual_gray = Self::to_premultiplied_grayscale(actual);
        let expected_gray = Self::to_premultiplied_grayscale(expected);

        Self::calculate_ssim(&actual_gray, &expected_gray)
    }

    /// Convert image to grayscale using premultiplied alpha
    fn to_premultiplied_grayscale(image: &DynamicImage) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut gray_image = GrayImage::new(width, height);
        let rgba = image.to_rgba8();

        for (x, y, pixel) in gray_image.enumerate_pixels_mut() {
            let rgba_pixel = rgba.get_pixel(x, y);
            let alpha = rgba_pixel[3] as f64 / 255.0;
            
            // Premultiply RGB by alpha before converting to grayscale
            let r = (rgba_pixel[0] as f64) * alpha;
            let g = (rgba_pixel[1] as f64) * alpha;
            let b = (rgba_pixel[2] as f64) * alpha;
            
            // Convert to grayscale using standard luminance formula
            let gray_value = (0.299 * r + 0.587 * g + 0.114 * b).round().min(255.0) as u8;
            *pixel = image::Luma([gray_value]);
        }

        gray_image
    }

    /// Calculate SSIM for grayscale images
    fn calculate_ssim(actual: &GrayImage, expected: &GrayImage) -> Result<f64> {
        let (_width, _height) = actual.dimensions();
        
        // Calculate means
        let mean_actual = Self::calculate_mean(actual);
        let mean_expected = Self::calculate_mean(expected);

        // Calculate variances and covariance
        let (var_actual, var_expected, covar) = Self::calculate_statistics(
            actual, expected, mean_actual, mean_expected
        );

        // SSIM constants
        let k1 = 0.01_f64;
        let k2 = 0.03_f64;
        let l = 255.0_f64; // Dynamic range for 8-bit images
        let c1 = (k1 * l).powi(2);
        let c2 = (k2 * l).powi(2);

        // SSIM formula
        let numerator = (2.0 * mean_actual * mean_expected + c1) * (2.0 * covar + c2);
        let denominator = (mean_actual.powi(2) + mean_expected.powi(2) + c1) * (var_actual + var_expected + c2);

        Ok(numerator / denominator)
    }

    /// Calculate mean pixel value for grayscale image
    fn calculate_mean(image: &GrayImage) -> f64 {
        let sum: u64 = image.pixels().map(|p| p[0] as u64).sum();
        let total_pixels = (image.width() * image.height()) as u64;
        sum as f64 / total_pixels as f64
    }

    /// Calculate variance and covariance for SSIM
    fn calculate_statistics(
        actual: &GrayImage,
        expected: &GrayImage,
        mean_actual: f64,
        mean_expected: f64,
    ) -> (f64, f64, f64) {
        let total_pixels = (actual.width() * actual.height()) as f64;
        
        let mut var_actual = 0.0;
        let mut var_expected = 0.0;
        let mut covar = 0.0;

        for (actual_pixel, expected_pixel) in actual.pixels().zip(expected.pixels()) {
            let diff_actual = actual_pixel[0] as f64 - mean_actual;
            let diff_expected = expected_pixel[0] as f64 - mean_expected;

            var_actual += diff_actual.powi(2);
            var_expected += diff_expected.powi(2);
            covar += diff_actual * diff_expected;
        }

        var_actual /= total_pixels - 1.0;
        var_expected /= total_pixels - 1.0;
        covar /= total_pixels - 1.0;

        (var_actual, var_expected, covar)
    }

    /// Calculate edge accuracy by comparing edge pixels specifically using premultiplied alpha
    pub fn edge_accuracy(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<f64> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for edge accuracy".to_string(),
            ));
        }

        // Convert to premultiplied grayscale and detect edges
        let actual_gray = Self::to_premultiplied_grayscale(actual);
        let expected_gray = Self::to_premultiplied_grayscale(expected);
        
        let actual_edges = Self::detect_edges(&actual_gray);
        let expected_edges = Self::detect_edges(&expected_gray);

        // Calculate accuracy only on edge pixels
        let mut edge_pixels = 0;
        let mut matching_edge_pixels = 0;

        for (x, y, expected_pixel) in expected_edges.enumerate_pixels() {
            if expected_pixel[0] > 128 { // Edge pixel in expected
                edge_pixels += 1;
                if let Some(actual_pixel) = actual_edges.get_pixel_checked(x, y) {
                    if actual_pixel[0] > 128 { // Edge detected in actual
                        matching_edge_pixels += 1;
                    }
                }
            }
        }

        if edge_pixels == 0 {
            return Ok(1.0); // No edges to compare
        }

        Ok(matching_edge_pixels as f64 / edge_pixels as f64)
    }

    /// Simple edge detection using Sobel operator
    fn detect_edges(image: &GrayImage) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut edges = GrayImage::new(width, height);

        // Sobel kernels
        let sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
        let sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut gx = 0i32;
                let mut gy = 0i32;

                // Apply Sobel kernels
                for i in 0..3 {
                    for j in 0..3 {
                        let pixel_value = image.get_pixel(x + j - 1, y + i - 1)[0] as i32;
                        gx += sobel_x[i as usize][j as usize] * pixel_value;
                        gy += sobel_y[i as usize][j as usize] * pixel_value;
                    }
                }

                // Calculate gradient magnitude
                let magnitude = ((gx * gx + gy * gy) as f64).sqrt();
                let edge_value = cmp::min(255, magnitude as u32) as u8;
                
                edges.put_pixel(x, y, Luma([edge_value]));
            }
        }

        edges
    }

    /// Calculate Mean Squared Error between two images using premultiplied alpha
    pub fn mean_squared_error(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<f64> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for MSE calculation".to_string(),
            ));
        }

        let actual_rgba = actual.to_rgba8();
        let expected_rgba = expected.to_rgba8();

        let total_pixels = (actual.width() * actual.height()) as f64;
        let mut sum_squared_error = 0.0;

        for (actual_pixel, expected_pixel) in actual_rgba.pixels().zip(expected_rgba.pixels()) {
            let actual_alpha = actual_pixel[3] as f64 / 255.0;
            let expected_alpha = expected_pixel[3] as f64 / 255.0;
            
            // Premultiply RGB by alpha
            let actual_r = (actual_pixel[0] as f64) * actual_alpha;
            let actual_g = (actual_pixel[1] as f64) * actual_alpha;
            let actual_b = (actual_pixel[2] as f64) * actual_alpha;
            
            let expected_r = (expected_pixel[0] as f64) * expected_alpha;
            let expected_g = (expected_pixel[1] as f64) * expected_alpha;
            let expected_b = (expected_pixel[2] as f64) * expected_alpha;
            
            // Calculate squared differences in premultiplied space
            let r_diff = actual_r - expected_r;
            let g_diff = actual_g - expected_g;
            let b_diff = actual_b - expected_b;
            let a_diff = (actual_alpha - expected_alpha) * 255.0; // Scale alpha back to 0-255 range
            
            sum_squared_error += r_diff * r_diff + g_diff * g_diff + b_diff * b_diff + a_diff * a_diff;
        }

        Ok(sum_squared_error / (total_pixels * 4.0)) // 4 channels
    }

    /// Generate a visual difference image for debugging
    pub fn generate_diff_image(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<RgbaImage> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for diff generation".to_string(),
            ));
        }

        let (width, height) = actual.dimensions();
        let mut diff_image = RgbaImage::new(width, height);

        let actual_rgba = actual.to_rgba8();
        let expected_rgba = expected.to_rgba8();

        for (x, y, pixel) in diff_image.enumerate_pixels_mut() {
            let actual_pixel = actual_rgba.get_pixel(x, y);
            let expected_pixel = expected_rgba.get_pixel(x, y);

            // Calculate absolute difference
            let r_diff = (actual_pixel[0] as i16 - expected_pixel[0] as i16).abs() as u8;
            let g_diff = (actual_pixel[1] as i16 - expected_pixel[1] as i16).abs() as u8;
            let b_diff = (actual_pixel[2] as i16 - expected_pixel[2] as i16).abs() as u8;
            let a_diff = (actual_pixel[3] as i16 - expected_pixel[3] as i16).abs() as u8;

            // Color-code differences: red for major differences, yellow for minor
            let max_diff = cmp::max(cmp::max(r_diff, g_diff), cmp::max(b_diff, a_diff));
            
            if max_diff > 50 {
                // Major difference - red
                *pixel = image::Rgba([255, 0, 0, 255]);
            } else if max_diff > 10 {
                // Minor difference - yellow
                *pixel = image::Rgba([255, 255, 0, 200]);
            } else {
                // No significant difference - transparent
                *pixel = image::Rgba([0, 0, 0, 0]);
            }
        }

        Ok(diff_image)
    }

    /// Generate an enhanced diff heatmap with color-coded intensity levels
    pub fn generate_enhanced_diff_heatmap(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<RgbaImage> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for heatmap generation".to_string(),
            ));
        }

        let (width, height) = actual.dimensions();
        let mut heatmap = RgbaImage::new(width, height);

        let actual_rgba = actual.to_rgba8();
        let expected_rgba = expected.to_rgba8();

        for (x, y, pixel) in heatmap.enumerate_pixels_mut() {
            let actual_pixel = actual_rgba.get_pixel(x, y);
            let expected_pixel = expected_rgba.get_pixel(x, y);

            // Convert to premultiplied alpha for accurate visual comparison
            let actual_alpha = actual_pixel[3] as f64 / 255.0;
            let expected_alpha = expected_pixel[3] as f64 / 255.0;
            
            // Premultiply RGB by alpha - this gives us the actual visual contribution
            let actual_r = (actual_pixel[0] as f64) * actual_alpha;
            let actual_g = (actual_pixel[1] as f64) * actual_alpha;
            let actual_b = (actual_pixel[2] as f64) * actual_alpha;
            
            let expected_r = (expected_pixel[0] as f64) * expected_alpha;
            let expected_g = (expected_pixel[1] as f64) * expected_alpha;
            let expected_b = (expected_pixel[2] as f64) * expected_alpha;
            
            // Calculate differences in premultiplied space
            let r_diff = (actual_r - expected_r).abs();
            let g_diff = (actual_g - expected_g).abs();
            let b_diff = (actual_b - expected_b).abs();
            let a_diff = (actual_alpha - expected_alpha).abs() * 255.0; // Scale alpha diff back to 0-255 range
            
            // Combine differences - weight alpha heavily since transparency is critical
            let rgb_diff = (r_diff + g_diff + b_diff) / 3.0; // Average RGB difference
            let total_diff = (rgb_diff + a_diff * 2.0) / 3.0; // Weight alpha 2x
            
            let diff_intensity = (total_diff / 255.0).min(1.0);

            // Create heatmap colors: transparent (no diff) -> blue -> green -> yellow -> red (max diff)
            let color = Self::intensity_to_heatmap_color(diff_intensity);
            *pixel = image::Rgba(color);
        }

        Ok(heatmap)
    }

    /// Convert intensity (0.0-1.0) to heatmap color [R, G, B, A]
    fn intensity_to_heatmap_color(intensity: f64) -> [u8; 4] {
        let intensity = intensity.max(0.0).min(1.0);
        
        if intensity < 0.01 {
            // Very small differences - transparent
            [0, 0, 0, 0]
        } else if intensity < 0.25 {
            // Small differences - blue to cyan
            let t = intensity / 0.25;
            [0, (t * 255.0) as u8, 255, 200]
        } else if intensity < 0.5 {
            // Medium differences - cyan to green  
            let t = (intensity - 0.25) / 0.25;
            [0, 255, (255.0 * (1.0 - t)) as u8, 220]
        } else if intensity < 0.75 {
            // Large differences - green to yellow
            let t = (intensity - 0.5) / 0.25;
            [(t * 255.0) as u8, 255, 0, 240]
        } else {
            // Very large differences - yellow to red
            let t = (intensity - 0.75) / 0.25;
            [255, (255.0 * (1.0 - t)) as u8, 0, 255]
        }
    }

    /// Calculate comprehensive test metrics for two images
    pub fn calculate_metrics(
        actual: &DynamicImage,
        expected: &DynamicImage,
        pixel_threshold: f64,
    ) -> Result<TestMetrics> {
        let pixel_accuracy = Self::pixel_accuracy(actual, expected, pixel_threshold)?;
        let ssim = Self::structural_similarity(actual, expected)?;
        let edge_accuracy = Self::edge_accuracy(actual, expected)?;
        let mse = Self::mean_squared_error(actual, expected)?;
        let visual_quality_score = Self::calculate_visual_quality_score(actual, expected)?;

        Ok(TestMetrics {
            pixel_accuracy,
            ssim,
            edge_accuracy,
            visual_quality_score,
            mean_squared_error: mse,
        })
    }

    /// Calculate visual quality score - a perceptual assessment that's more forgiving than pixel accuracy
    /// This combines SSIM, edge preservation, and background separation quality
    pub fn calculate_visual_quality_score(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<f64> {
        let ssim = Self::structural_similarity(actual, expected)?;
        let edge_accuracy = Self::edge_accuracy(actual, expected)?;
        
        // Assess background separation quality
        let bg_separation_score = Self::assess_background_separation_quality(actual)?;
        
        // Weight the scores - SSIM is most important for visual similarity
        let visual_score = (ssim * 0.6) + (edge_accuracy * 0.25) + (bg_separation_score * 0.15);
        
        Ok(visual_score.min(1.0).max(0.0))
    }
    
    /// Assess the quality of background separation regardless of exact pixel matching
    /// This checks if the image has good foreground/background separation
    fn assess_background_separation_quality(image: &DynamicImage) -> Result<f64> {
        let rgba = image.to_rgba8();
        let total_pixels = (image.width() * image.height()) as f64;
        
        let mut fully_transparent = 0;
        let mut fully_opaque = 0;
        let mut partial_alpha = 0;
        
        for pixel in rgba.pixels() {
            match pixel[3] { // Alpha channel
                0 => fully_transparent += 1,           // Perfect background removal
                255 => fully_opaque += 1,              // Clear foreground
                _ => partial_alpha += 1,               // Smooth edges
            }
        }
        
        let transparency_ratio = fully_transparent as f64 / total_pixels;
        let opacity_ratio = fully_opaque as f64 / total_pixels;
        let smooth_edges_ratio = partial_alpha as f64 / total_pixels;
        
        // Good background removal should have:
        // - Significant transparent areas (background removed)
        // - Significant opaque areas (foreground preserved) 
        // - Some smooth edges for natural transitions
        let has_background_removal = transparency_ratio > 0.1;
        let has_foreground = opacity_ratio > 0.1;
        let has_smooth_edges = smooth_edges_ratio > 0.01;
        
        // Score based on separation quality
        let separation_score = if has_background_removal && has_foreground {
            let balance_score = (transparency_ratio.min(opacity_ratio) * 2.0).min(1.0);
            let edge_bonus = if has_smooth_edges { 0.1 } else { 0.0 };
            balance_score + edge_bonus
        } else {
            0.2 // Poor separation
        };
        
        Ok(separation_score.min(1.0))
    }

    /// Generate a side-by-side comparison image
    pub fn create_comparison_image(
        original: &DynamicImage,
        expected: &DynamicImage,
        actual: &DynamicImage,
    ) -> Result<RgbaImage> {
        let (width, height) = original.dimensions();
        
        // Create a wide image to fit all three side by side
        let comparison_width = width * 3 + 20; // 10px padding between images
        let comparison_height = height;
        let mut comparison = RgbaImage::new(comparison_width, comparison_height);

        // Fill with white background
        for pixel in comparison.pixels_mut() {
            *pixel = image::Rgba([255, 255, 255, 255]);
        }

        // Copy original image
        let original_rgba = original.to_rgba8();
        for (x, y, pixel) in original_rgba.enumerate_pixels() {
            comparison.put_pixel(x, y, *pixel);
        }

        // Copy expected image (offset by width + padding)
        let expected_rgba = expected.to_rgba8();
        for (x, y, pixel) in expected_rgba.enumerate_pixels() {
            comparison.put_pixel(x + width + 10, y, *pixel);
        }

        // Copy actual image (offset by 2*width + 2*padding)
        let actual_rgba = actual.to_rgba8();
        for (x, y, pixel) in actual_rgba.enumerate_pixels() {
            comparison.put_pixel(x + 2 * width + 20, y, *pixel);
        }

        Ok(comparison)
    }
}

/// Helper functions for image analysis
pub struct ImageAnalysis;

impl ImageAnalysis {
    /// Analyze mask quality (for alpha channel analysis)
    pub fn analyze_mask_quality(mask: &GrayImage) -> MaskQualityMetrics {
        let total_pixels = (mask.width() * mask.height()) as f64;
        let mut transparent_pixels = 0;
        let mut opaque_pixels = 0;
        let mut partial_transparency = 0;
        let mut edge_pixels = 0;

        for pixel in mask.pixels() {
            let alpha = pixel[0];
            match alpha {
                0 => transparent_pixels += 1,
                255 => opaque_pixels += 1,
                _ => partial_transparency += 1,
            }

            // Simple edge detection for mask quality
            if alpha > 0 && alpha < 255 {
                edge_pixels += 1;
            }
        }

        MaskQualityMetrics {
            transparency_ratio: transparent_pixels as f64 / total_pixels,
            opacity_ratio: opaque_pixels as f64 / total_pixels,
            partial_transparency_ratio: partial_transparency as f64 / total_pixels,
            edge_softness: edge_pixels as f64 / total_pixels,
        }
    }

    /// Check if image has realistic proportions and content
    pub fn validate_output_realism(image: &DynamicImage) -> RealismScore {
        let rgba = image.to_rgba8();
        let total_pixels = (image.width() * image.height()) as f64;
        
        let mut fully_transparent = 0;
        let mut fully_opaque = 0;
        
        for pixel in rgba.pixels() {
            match pixel[3] {
                0 => fully_transparent += 1,
                255 => fully_opaque += 1,
                _ => {}
            }
        }

        let transparency_ratio = fully_transparent as f64 / total_pixels;
        let opacity_ratio = fully_opaque as f64 / total_pixels;

        // Realistic images should have significant foreground and background separation
        let has_realistic_separation = transparency_ratio > 0.1 && opacity_ratio > 0.1;
        let not_overly_transparent = transparency_ratio < 0.9;
        let not_overly_opaque = opacity_ratio < 0.9;

        RealismScore {
            has_realistic_separation,
            not_overly_transparent,
            not_overly_opaque,
            overall_score: if has_realistic_separation && not_overly_transparent && not_overly_opaque {
                1.0
            } else {
                0.5
            },
        }
    }
}

/// Metrics for mask quality analysis
#[derive(Debug, Clone)]
pub struct MaskQualityMetrics {
    pub transparency_ratio: f64,
    pub opacity_ratio: f64,
    pub partial_transparency_ratio: f64,
    pub edge_softness: f64,
}

/// Realism score for output validation
#[derive(Debug, Clone)]
pub struct RealismScore {
    pub has_realistic_separation: bool,
    pub not_overly_transparent: bool,
    pub not_overly_opaque: bool,
    pub overall_score: f64,
}