//! Image comparison and accuracy metrics

use crate::{TestMetrics, TestingError, Result};
use image::{DynamicImage, GrayImage, Luma, RgbaImage, GenericImageView};
use std::cmp;

/// Image comparison utilities for testing accuracy
pub struct ImageComparison;

impl ImageComparison {
    /// Calculate pixel-level accuracy between two images
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
            if Self::pixels_match(actual_pixel, expected_pixel, threshold) {
                matching_pixels += 1.0;
            }
        }

        Ok(matching_pixels / total_pixels)
    }

    /// Check if two pixels match within threshold
    fn pixels_match(
        actual: &image::Rgba<u8>,
        expected: &image::Rgba<u8>,
        threshold: f64,
    ) -> bool {
        let threshold = (threshold * 255.0) as u8;
        
        let r_diff = (actual[0] as i16 - expected[0] as i16).abs() as u8;
        let g_diff = (actual[1] as i16 - expected[1] as i16).abs() as u8;
        let b_diff = (actual[2] as i16 - expected[2] as i16).abs() as u8;
        let a_diff = (actual[3] as i16 - expected[3] as i16).abs() as u8;

        r_diff <= threshold && g_diff <= threshold && b_diff <= threshold && a_diff <= threshold
    }

    /// Calculate Structural Similarity Index (SSIM) between two images
    pub fn structural_similarity(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<f64> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for SSIM calculation".to_string(),
            ));
        }

        // Convert to grayscale for SSIM calculation
        let actual_gray = actual.to_luma8();
        let expected_gray = expected.to_luma8();

        Self::calculate_ssim(&actual_gray, &expected_gray)
    }

    /// Calculate SSIM for grayscale images
    fn calculate_ssim(actual: &GrayImage, expected: &GrayImage) -> Result<f64> {
        let (width, height) = actual.dimensions();
        
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

    /// Calculate edge accuracy by comparing edge pixels specifically
    pub fn edge_accuracy(
        actual: &DynamicImage,
        expected: &DynamicImage,
    ) -> Result<f64> {
        if actual.dimensions() != expected.dimensions() {
            return Err(TestingError::InvalidConfiguration(
                "Images must have same dimensions for edge accuracy".to_string(),
            ));
        }

        // Convert to grayscale and detect edges
        let actual_edges = Self::detect_edges(&actual.to_luma8());
        let expected_edges = Self::detect_edges(&expected.to_luma8());

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

    /// Calculate Mean Squared Error between two images
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
            for i in 0..4 { // RGBA channels
                let diff = actual_pixel[i] as f64 - expected_pixel[i] as f64;
                sum_squared_error += diff * diff;
            }
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

        Ok(TestMetrics {
            pixel_accuracy,
            ssim,
            edge_accuracy,
            mean_squared_error: mse,
        })
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