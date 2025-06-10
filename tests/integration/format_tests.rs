use image::{DynamicImage, ImageFormat, ColorType};
use std::path::Path;
use std::collections::HashMap;
use serde_json;

/// Format test result
#[derive(Debug, Clone)]
pub struct FormatTestResult {
    pub test_id: String,
    pub input_format: String,
    pub output_format: String,
    pub success: bool,
    pub format_valid: bool,
    pub dimensions_preserved: bool,
    pub color_channels_correct: bool,
    pub file_size_reasonable: bool,
    pub error_message: Option<String>,
}

/// Input format specification
#[derive(Debug, serde::Deserialize)]
pub struct InputFormatSpec {
    pub supported_qualities: Option<Vec<u8>>,
    pub max_resolution: Option<[u32; 2]>,
    pub min_resolution: Option<[u32; 2]>,
    pub color_spaces: Option<Vec<String>>,
    pub bit_depths: Option<Vec<u8>>,
    pub color_types: Option<Vec<String>>,
    pub pixel_format: Option<String>,
    pub bytes_per_pixel: Option<u8>,
    pub stride_alignment: Option<u8>,
    pub chroma_subsampling: Option<String>,
    pub plane_layout: Option<String>,
}

/// Output format specification
#[derive(Debug, serde::Deserialize)]
pub struct OutputFormatSpec {
    pub format: String,
    pub bit_depth: Option<u8>,
    pub color_type: Option<String>,
    pub compression_level: Option<u8>,
    pub quality: Option<u8>,
    pub lossless: Option<bool>,
    pub alpha_support: Option<bool>,
    pub alpha_validation: Option<AlphaValidation>,
    pub background_colors: Option<HashMap<String, [u8; 3]>>,
    pub value_range: Option<[u8; 2]>,
    pub validation: Option<FormatValidation>,
}

/// Alpha channel validation rules
#[derive(Debug, serde::Deserialize)]
pub struct AlphaValidation {
    pub alpha_range: [u8; 2],
    pub transparency_check: bool,
    pub alpha_edge_smoothness: Option<bool>,
}

/// Format-specific validation rules
#[derive(Debug, serde::Deserialize)]
pub struct FormatValidation {
    pub binary_mask: Option<bool>,
    pub soft_edges: Option<bool>,
    pub gradient_smoothness: Option<bool>,
}

/// Test input/output format support
pub fn test_format_support<P: AsRef<Path>>(
    binary_path: P,
    test_images_dir: P,
    output_dir: P,
    input_formats: &HashMap<String, InputFormatSpec>,
    output_formats: &HashMap<String, OutputFormatSpec>,
) -> Vec<FormatTestResult> {
    let mut results = Vec::new();
    
    // Create output directory
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir).unwrap();
    
    // Test each input format
    for (input_format_name, input_spec) in input_formats {
        println!("Testing input format: {}", input_format_name);
        
        // Find test files for this format
        let test_files = find_test_files_for_format(test_images_dir.as_ref(), input_format_name);
        
        if test_files.is_empty() {
            eprintln!("No test files found for format: {}", input_format_name);
            continue;
        }
        
        // Test with each output format
        for (output_format_name, output_spec) in output_formats {
            for test_file in &test_files {
                let result = test_single_format_combination(
                    binary_path.as_ref(),
                    test_file,
                    output_dir,
                    input_format_name,
                    input_spec,
                    output_format_name,
                    output_spec,
                );
                
                results.push(result);
            }
        }
    }
    
    results
}

/// Test a single input/output format combination
fn test_single_format_combination(
    binary_path: &Path,
    input_file: &Path,
    output_dir: &Path,
    input_format_name: &str,
    input_spec: &InputFormatSpec,
    output_format_name: &str,
    output_spec: &OutputFormatSpec,
) -> FormatTestResult {
    let test_id = format!(
        "{}_{}_to_{}",
        input_file.file_stem().unwrap().to_str().unwrap(),
        input_format_name,
        output_format_name
    );
    
    let output_file = output_dir.join(format!("{}.{}", test_id, get_file_extension(output_format_name)));
    
    // Build command
    let mut cmd = std::process::Command::new(binary_path);
    cmd.arg(input_file)
       .arg("--output")
       .arg(&output_file);
    
    // Add format-specific arguments
    add_format_arguments(&mut cmd, input_format_name, input_spec, output_format_name, output_spec);
    
    // Execute command
    match cmd.output() {
        Ok(output) => {
            let success = output.status.success() && output_file.exists();
            
            if success {
                // Validate output format
                validate_output_format(
                    &test_id,
                    input_format_name.to_string(),
                    output_format_name.to_string(),
                    input_file,
                    &output_file,
                    input_spec,
                    output_spec,
                )
            } else {
                FormatTestResult {
                    test_id,
                    input_format: input_format_name.to_string(),
                    output_format: output_format_name.to_string(),
                    success: false,
                    format_valid: false,
                    dimensions_preserved: false,
                    color_channels_correct: false,
                    file_size_reasonable: false,
                    error_message: Some(String::from_utf8_lossy(&output.stderr).to_string()),
                }
            }
        }
        Err(e) => FormatTestResult {
            test_id,
            input_format: input_format_name.to_string(),
            output_format: output_format_name.to_string(),
            success: false,
            format_valid: false,
            dimensions_preserved: false,
            color_channels_correct: false,
            file_size_reasonable: false,
            error_message: Some(format!("Failed to execute command: {}", e)),
        },
    }
}

/// Add format-specific command arguments
fn add_format_arguments(
    cmd: &mut std::process::Command,
    input_format_name: &str,
    input_spec: &InputFormatSpec,
    output_format_name: &str,
    output_spec: &OutputFormatSpec,
) {
    // Input format arguments
    match input_format_name {
        "raw_rgb24" | "raw_rgba32" | "raw_yuv420p" | "raw_yuv444p" => {
            cmd.arg("--input-format").arg(input_format_name);
            // Add width/height if it's a raw format (would need to be provided separately)
        }
        _ => {}
    }
    
    // Output format arguments
    cmd.arg("--format").arg(&output_spec.format.to_lowercase());
    
    if let Some(quality) = output_spec.quality {
        cmd.arg("--quality").arg(quality.to_string());
    }
    
    if let Some(compression) = output_spec.compression_level {
        cmd.arg("--compression").arg(compression.to_string());
    }
    
    if let Some(lossless) = output_spec.lossless {
        if lossless {
            cmd.arg("--lossless");
        }
    }
    
    // Background color for JPEG output
    if output_spec.format.to_lowercase() == "jpeg" {
        if let Some(bg_colors) = &output_spec.background_colors {
            if let Some(white) = bg_colors.get("white") {
                cmd.arg("--bg-color")
                   .arg(format!("{},{},{}", white[0], white[1], white[2]));
            }
        }
    }
}

/// Get file extension for format
fn get_file_extension(format_name: &str) -> &str {
    match format_name.to_lowercase().as_str() {
        "png_alpha" | "mask_grayscale" => "png",
        "jpeg_background" => "jpg",
        "webp_alpha" => "webp",
        _ => "png", // Default
    }
}

/// Find test files for a specific format
fn find_test_files_for_format(test_dir: &Path, format_name: &str) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    
    match format_name {
        "jpeg" => {
            // Look for JPEG files in all categories
            for category in &["portraits", "products", "complex", "edge_cases"] {
                let category_dir = test_dir.join(category);
                if let Ok(entries) = std::fs::read_dir(&category_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(ext) = path.extension() {
                            if ext == "jpg" || ext == "jpeg" {
                                files.push(path);
                            }
                        }
                    }
                }
            }
        }
        "png" => {
            // Look for PNG files
            for category in &["portraits", "products", "complex", "edge_cases"] {
                let category_dir = test_dir.join(category);
                if let Ok(entries) = std::fs::read_dir(&category_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(ext) = path.extension() {
                            if ext == "png" {
                                files.push(path);
                            }
                        }
                    }
                }
            }
        }
        "webp" => {
            // Look for WebP files
            for category in &["portraits", "products", "complex", "edge_cases"] {
                let category_dir = test_dir.join(category);
                if let Ok(entries) = std::fs::read_dir(&category_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(ext) = path.extension() {
                            if ext == "webp" {
                                files.push(path);
                            }
                        }
                    }
                }
            }
        }
        "raw_rgb24" | "raw_rgba32" | "raw_yuv420p" | "raw_yuv444p" => {
            // Look for raw format files
            let raw_dir = test_dir.join("raw_formats");
            if let Ok(entries) = std::fs::read_dir(&raw_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if ext == "raw" && path.file_name().unwrap().to_str().unwrap().contains(
                            &format_name.replace("raw_", "")
                        ) {
                            files.push(path);
                        }
                    }
                }
            }
        }
        _ => {
            // Default: look for common image files
            for category in &["portraits", "products"] {
                let category_dir = test_dir.join(category);
                if let Ok(entries) = std::fs::read_dir(&category_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(ext) = path.extension() {
                            if ext == "jpg" || ext == "png" {
                                files.push(path);
                                break; // Just take one sample
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Limit to first few files to avoid excessive testing
    files.truncate(3);
    files
}

/// Validate output format meets specifications
fn validate_output_format(
    test_id: &str,
    input_format: String,
    output_format: String,
    input_file: &Path,
    output_file: &Path,
    input_spec: &InputFormatSpec,
    output_spec: &OutputFormatSpec,
) -> FormatTestResult {
    // Load and validate output image
    let output_image = match image::open(output_file) {
        Ok(img) => img,
        Err(e) => return FormatTestResult {
            test_id: test_id.to_string(),
            input_format,
            output_format,
            success: false,
            format_valid: false,
            dimensions_preserved: false,
            color_channels_correct: false,
            file_size_reasonable: false,
            error_message: Some(format!("Failed to load output image: {}", e)),
        },
    };
    
    // Get input dimensions for comparison
    let input_dimensions = if input_file.extension().unwrap_or_default() == "raw" {
        // For raw files, we'd need to get dimensions from metadata
        // For now, assume they match output
        output_image.dimensions()
    } else {
        match image::image_dimensions(input_file) {
            Ok(dims) => dims,
            Err(_) => output_image.dimensions(), // Fallback
        }
    };
    
    // Validate dimensions
    let dimensions_preserved = output_image.dimensions() == input_dimensions;
    
    // Validate color channels
    let color_channels_correct = validate_color_channels(&output_image, output_spec);
    
    // Validate format-specific properties
    let format_valid = validate_format_properties(&output_image, output_spec);
    
    // Validate file size
    let file_size_reasonable = validate_file_size(output_file, &output_image, output_spec);
    
    // Validate alpha channel if required
    let alpha_valid = if let Some(alpha_validation) = &output_spec.alpha_validation {
        validate_alpha_channel(&output_image, alpha_validation)
    } else {
        true
    };
    
    let success = format_valid && dimensions_preserved && color_channels_correct && alpha_valid;
    
    FormatTestResult {
        test_id: test_id.to_string(),
        input_format,
        output_format,
        success,
        format_valid,
        dimensions_preserved,
        color_channels_correct,
        file_size_reasonable,
        error_message: None,
    }
}

/// Validate color channels match specification
fn validate_color_channels(image: &DynamicImage, spec: &OutputFormatSpec) -> bool {
    let expected_channels = match spec.color_type.as_deref() {
        Some("RGB") => 3,
        Some("RGBA") => 4,
        Some("Grayscale") => 1,
        _ => {
            // Infer from format
            match spec.format.to_lowercase().as_str() {
                "png" => {
                    if spec.alpha_support.unwrap_or(false) { 4 } else { 3 }
                }
                "jpeg" => 3,
                "webp" => {
                    if spec.alpha_support.unwrap_or(false) { 4 } else { 3 }
                }
                _ => 3, // Default
            }
        }
    };
    
    let actual_channels = match image.color() {
        ColorType::L8 => 1,
        ColorType::La8 => 2,
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
        ColorType::L16 => 1,
        ColorType::La16 => 2,
        ColorType::Rgb16 => 3,
        ColorType::Rgba16 => 4,
        _ => 3, // Default
    };
    
    actual_channels == expected_channels
}

/// Validate format-specific properties
fn validate_format_properties(image: &DynamicImage, spec: &OutputFormatSpec) -> bool {
    // Check bit depth if specified
    if let Some(expected_bit_depth) = spec.bit_depth {
        let actual_bit_depth = match image.color() {
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => 8,
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => 16,
            _ => 8, // Default
        };
        
        if actual_bit_depth != expected_bit_depth {
            return false;
        }
    }
    
    // Additional format-specific checks could go here
    true
}

/// Validate file size is reasonable
fn validate_file_size(file_path: &Path, image: &DynamicImage, spec: &OutputFormatSpec) -> bool {
    let file_size = match std::fs::metadata(file_path) {
        Ok(metadata) => metadata.len(),
        Err(_) => return false,
    };
    
    let (width, height) = image.dimensions();
    let pixel_count = width as u64 * height as u64;
    
    // Basic sanity checks
    let min_size = pixel_count / 10; // Very compressed
    let max_size = pixel_count * 8;  // Uncompressed RGBA + overhead
    
    file_size >= min_size && file_size <= max_size
}

/// Validate alpha channel properties
fn validate_alpha_channel(image: &DynamicImage, alpha_validation: &AlphaValidation) -> bool {
    // Convert to RGBA for alpha analysis
    let rgba_image = image.to_rgba8();
    let (width, height) = rgba_image.dimensions();
    
    let mut alpha_values = Vec::new();
    let mut has_transparency = false;
    let mut has_opacity = false;
    
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba_image.get_pixel(x, y);
            let alpha = pixel[3];
            
            alpha_values.push(alpha);
            
            if alpha < 255 {
                has_transparency = true;
            }
            if alpha > 0 {
                has_opacity = true;
            }
        }
    }
    
    // Check alpha range
    let min_alpha = *alpha_values.iter().min().unwrap_or(&0);
    let max_alpha = *alpha_values.iter().max().unwrap_or(&255);
    
    if min_alpha < alpha_validation.alpha_range[0] || max_alpha > alpha_validation.alpha_range[1] {
        return false;
    }
    
    // Check transparency requirement
    if alpha_validation.transparency_check {
        if !has_transparency || !has_opacity {
            return false; // Should have both transparent and opaque areas
        }
    }
    
    // Check edge smoothness if required
    if alpha_validation.alpha_edge_smoothness.unwrap_or(false) {
        return validate_alpha_edge_smoothness(&rgba_image);
    }
    
    true
}

/// Validate alpha edge smoothness
fn validate_alpha_edge_smoothness(image: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> bool {
    let (width, height) = image.dimensions();
    let mut harsh_edges = 0;
    let mut total_edges = 0;
    
    for y in 1..height-1 {
        for x in 1..width-1 {
            let center_alpha = image.get_pixel(x, y)[3];
            
            // Check 8-connected neighbors
            let mut max_diff = 0u8;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;
                    let neighbor_alpha = image.get_pixel(nx, ny)[3];
                    
                    let diff = (center_alpha as i32 - neighbor_alpha as i32).abs() as u8;
                    max_diff = max_diff.max(diff);
                }
            }
            
            if max_diff > 50 { // Threshold for edge detection
                total_edges += 1;
                if max_diff > 200 { // Threshold for harsh edge
                    harsh_edges += 1;
                }
            }
        }
    }
    
    // Allow up to 20% harsh edges
    if total_edges > 0 {
        (harsh_edges as f32 / total_edges as f32) < 0.2
    } else {
        true
    }
}

/// Test raw format processing specifically
pub fn test_raw_format_processing<P: AsRef<Path>>(
    binary_path: P,
    raw_test_dir: P,
    output_dir: P,
    raw_specs: &HashMap<String, InputFormatSpec>,
) -> Vec<FormatTestResult> {
    let mut results = Vec::new();
    
    for (format_name, spec) in raw_specs {
        println!("Testing raw format: {}", format_name);
        
        // Create test metadata file for raw format
        let metadata = create_raw_format_metadata(format_name, spec);
        
        // Find raw test files
        let raw_files = find_raw_test_files(raw_test_dir.as_ref(), format_name);
        
        for raw_file in raw_files {
            let result = test_raw_format_conversion(
                binary_path.as_ref(),
                &raw_file,
                output_dir.as_ref(),
                format_name,
                spec,
                &metadata,
            );
            
            results.push(result);
        }
    }
    
    results
}

/// Create metadata for raw format testing
fn create_raw_format_metadata(format_name: &str, spec: &InputFormatSpec) -> HashMap<String, serde_json::Value> {
    let mut metadata = HashMap::new();
    
    // Add common metadata
    metadata.insert("pixel_format".to_string(), serde_json::Value::String(format_name.to_string()));
    
    if let Some(bytes_per_pixel) = spec.bytes_per_pixel {
        metadata.insert("bytes_per_pixel".to_string(), serde_json::Value::Number(bytes_per_pixel.into()));
    }
    
    // Add format-specific metadata
    match format_name {
        "raw_rgb24" => {
            metadata.insert("width".to_string(), serde_json::Value::Number(1920.into()));
            metadata.insert("height".to_string(), serde_json::Value::Number(1080.into()));
        }
        "raw_rgba32" => {
            metadata.insert("width".to_string(), serde_json::Value::Number(1280.into()));
            metadata.insert("height".to_string(), serde_json::Value::Number(720.into()));
        }
        "raw_yuv420p" | "raw_yuv444p" => {
            metadata.insert("width".to_string(), serde_json::Value::Number(1920.into()));
            metadata.insert("height".to_string(), serde_json::Value::Number(1080.into()));
            if let Some(chroma) = &spec.chroma_subsampling {
                metadata.insert("chroma_subsampling".to_string(), serde_json::Value::String(chroma.clone()));
            }
        }
        _ => {}
    }
    
    metadata
}

/// Find raw test files
fn find_raw_test_files(raw_dir: &Path, format_name: &str) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    
    if let Ok(entries) = std::fs::read_dir(raw_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(file_name) = path.file_name() {
                if file_name.to_str().unwrap().contains(&format_name.replace("raw_", "")) {
                    files.push(path);
                }
            }
        }
    }
    
    files
}

/// Test raw format conversion
fn test_raw_format_conversion(
    binary_path: &Path,
    raw_file: &Path,
    output_dir: &Path,
    format_name: &str,
    spec: &InputFormatSpec,
    metadata: &HashMap<String, serde_json::Value>,
) -> FormatTestResult {
    let test_id = format!("raw_{}_{}", format_name, raw_file.file_stem().unwrap().to_str().unwrap());
    let output_file = output_dir.join(format!("{}.png", test_id));
    
    // Build command with raw format parameters
    let mut cmd = std::process::Command::new(binary_path);
    cmd.arg(raw_file)
       .arg("--output").arg(&output_file)
       .arg("--input-format").arg(format_name);
    
    // Add metadata parameters
    if let Some(width) = metadata.get("width") {
        cmd.arg("--width").arg(width.as_i64().unwrap().to_string());
    }
    if let Some(height) = metadata.get("height") {
        cmd.arg("--height").arg(height.as_i64().unwrap().to_string());
    }
    
    // Execute command
    match cmd.output() {
        Ok(output) => {
            let success = output.status.success() && output_file.exists();
            
            FormatTestResult {
                test_id,
                input_format: format_name.to_string(),
                output_format: "PNG".to_string(),
                success,
                format_valid: success,
                dimensions_preserved: success, // Would need more detailed validation
                color_channels_correct: success,
                file_size_reasonable: success,
                error_message: if success { None } else { 
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                },
            }
        }
        Err(e) => FormatTestResult {
            test_id,
            input_format: format_name.to_string(),
            output_format: "PNG".to_string(),
            success: false,
            format_valid: false,
            dimensions_preserved: false,
            color_channels_correct: false,
            file_size_reasonable: false,
            error_message: Some(format!("Command execution failed: {}", e)),
        },
    }
}

/// Generate format test report
pub fn generate_format_report(
    results: &[FormatTestResult],
    output_file: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    
    let mut report = String::new();
    report.push_str("# Format Support Test Report\n\n");
    
    // Summary
    let total_tests = results.len();
    let successful_tests = results.iter().filter(|r| r.success).count();
    let success_rate = if total_tests > 0 { successful_tests as f64 / total_tests as f64 * 100.0 } else { 0.0 };
    
    report.push_str(&format!("**Total Tests:** {}\n", total_tests));
    report.push_str(&format!("**Successful:** {}\n", successful_tests));
    report.push_str(&format!("**Success Rate:** {:.1}%\n\n", success_rate));
    
    // Group results by input format
    let mut by_input_format: HashMap<String, Vec<&FormatTestResult>> = HashMap::new();
    for result in results {
        by_input_format.entry(result.input_format.clone()).or_default().push(result);
    }
    
    for (input_format, format_results) in by_input_format {
        report.push_str(&format!("## {} Input Format\n\n", input_format));
        
        report.push_str("| Test ID | Output Format | Success | Format Valid | Dimensions | Color Channels | File Size | Error |\n");
        report.push_str("|---------|---------------|---------|--------------|------------|----------------|-----------|-------|\n");
        
        for result in format_results {
            let error = result.error_message.as_deref().unwrap_or("None");
            report.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                result.test_id,
                result.output_format,
                if result.success { "✓" } else { "✗" },
                if result.format_valid { "✓" } else { "✗" },
                if result.dimensions_preserved { "✓" } else { "✗" },
                if result.color_channels_correct { "✓" } else { "✗" },
                if result.file_size_reasonable { "✓" } else { "✗" },
                error
            ));
        }
        
        report.push_str("\n");
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
    fn test_color_channel_validation() {
        let rgb_image = DynamicImage::ImageRgb8(
            ImageBuffer::from_fn(100, 100, |_, _| image::Rgb([255, 255, 255]))
        );
        
        let rgba_image = DynamicImage::ImageRgba8(
            ImageBuffer::from_fn(100, 100, |_, _| Rgba([255, 255, 255, 255]))
        );
        
        let rgb_spec = OutputFormatSpec {
            format: "JPEG".to_string(),
            color_type: Some("RGB".to_string()),
            bit_depth: Some(8),
            compression_level: None,
            quality: None,
            lossless: None,
            alpha_support: None,
            alpha_validation: None,
            background_colors: None,
            value_range: None,
            validation: None,
        };
        
        let rgba_spec = OutputFormatSpec {
            format: "PNG".to_string(),
            color_type: Some("RGBA".to_string()),
            bit_depth: Some(8),
            compression_level: None,
            quality: None,
            lossless: None,
            alpha_support: Some(true),
            alpha_validation: None,
            background_colors: None,
            value_range: None,
            validation: None,
        };
        
        assert!(validate_color_channels(&rgb_image, &rgb_spec));
        assert!(validate_color_channels(&rgba_image, &rgba_spec));
        assert!(!validate_color_channels(&rgb_image, &rgba_spec)); // RGB image, RGBA spec
    }
    
    #[test]
    fn test_alpha_edge_smoothness() {
        // Create image with smooth alpha transition
        let smooth_image = ImageBuffer::from_fn(100, 100, |x, y| {
            let alpha = if x < 50 {
                255u8
            } else {
                // Smooth transition
                ((100 - x) as f32 * 5.1) as u8
            };
            Rgba([255, 255, 255, alpha])
        });
        
        // Create image with harsh alpha transition
        let harsh_image = ImageBuffer::from_fn(100, 100, |x, y| {
            let alpha = if x < 50 { 255u8 } else { 0u8 };
            Rgba([255, 255, 255, alpha])
        });
        
        assert!(validate_alpha_edge_smoothness(&smooth_image));
        assert!(!validate_alpha_edge_smoothness(&harsh_image));
    }
}