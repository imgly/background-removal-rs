//! End-to-end test for ICC color profile preservation
//!
//! This test verifies that ICC color profiles are correctly extracted from input images
//! and embedded in output images when color profile preservation is enabled.

use anyhow::{Context, Result};
use bg_remove_core::{
    color_profile::ProfileExtractor,
    config::{ExecutionProvider, OutputFormat},
    inference::InferenceBackend,
    models::ModelManager,
    processor::{BackendFactory, BackendType, BackgroundRemovalProcessor, ProcessorConfig},
};
use clap::Parser;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::{Path, PathBuf};

/// Test backend factory that creates ONNX backends for testing
struct TestBackendFactory;

impl TestBackendFactory {
    fn new() -> Self {
        Self
    }
}

impl BackendFactory for TestBackendFactory {
    fn create_backend(
        &self,
        backend_type: BackendType,
        model_manager: ModelManager,
    ) -> bg_remove_core::error::Result<Box<dyn InferenceBackend>> {
        match backend_type {
            BackendType::Onnx => {
                let backend = bg_remove_onnx::OnnxBackend::with_model_manager(model_manager);
                Ok(Box::new(backend))
            },
            BackendType::Tract => {
                let backend = bg_remove_tract::TractBackend::with_model_manager(model_manager);
                Ok(Box::new(backend))
            },
        }
    }

    fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::Onnx, BackendType::Tract]
    }
}

#[derive(Parser)]
#[command(name = "test-color-profile")]
#[command(about = "Test ICC color profile preservation in background removal")]
struct Args {
    /// Input image with ICC profile to test (optional - will create test image if not provided)
    #[arg(long)]
    input: Option<PathBuf>,

    /// Output directory for test results
    #[arg(long, default_value = "target/color-profile-test")]
    output_dir: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .format_timestamp_secs()
            .init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format_timestamp_secs()
            .init();
    }

    println!("🧪 Testing ICC Color Profile Preservation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create output directory
    std::fs::create_dir_all(&args.output_dir).context("Failed to create output directory")?;

    // Get or create test image with ICC profile
    let test_image_path = if let Some(input) = args.input {
        println!("📁 Using provided input image: {}", input.display());
        input
    } else {
        println!("🎨 Creating test image with sRGB ICC profile...");
        create_test_image_with_profile(&args.output_dir).await?
    };

    // Test color profile preservation
    let test_results = run_color_profile_tests(&test_image_path, &args.output_dir).await?;

    // Display results
    display_test_results(&test_results);

    if test_results.all_passed() {
        println!("✅ All color profile tests passed!");
        Ok(())
    } else {
        anyhow::bail!("❌ Some color profile tests failed");
    }
}

/// Create a test image with an embedded ICC profile
async fn create_test_image_with_profile(output_dir: &Path) -> Result<PathBuf> {
    use bg_remove_core::{color_profile::ProfileEmbedder, types::ColorProfile};

    // Create a simple test image with color gradients
    let width = 400;
    let height = 300;
    let mut img = ImageBuffer::new(width, height);

    // Create color gradients to test color profile handling
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = ((x as f32 / width as f32) * 255.0) as u8;
        let g = ((y as f32 / height as f32) * 255.0) as u8;
        let b = 128; // Fixed blue component
        *pixel = Rgb([r, g, b]);
    }

    let dynamic_img = DynamicImage::ImageRgb8(img);
    let test_image_path = output_dir.join("test_input_with_profile.jpg");

    // Create a basic sRGB ICC profile for testing
    // This is a minimal sRGB ICC profile for testing purposes
    let srgb_icc_data = create_minimal_srgb_profile();
    let color_profile = ColorProfile::from_icc_data(srgb_icc_data);

    println!("  ├─ Generating color gradient image...");
    println!("  ├─ Embedding sRGB ICC profile...");

    // Use ProfileEmbedder to save with ICC profile
    ProfileEmbedder::embed_in_output(
        &dynamic_img,
        &color_profile,
        &test_image_path,
        image::ImageFormat::Jpeg,
        90,
    )
    .context("Failed to save test image with ICC profile")?;

    println!(
        "  └─ Test image created: {} (with {} bytes ICC profile)",
        test_image_path.display(),
        color_profile.data_size()
    );

    Ok(test_image_path)
}

/// Create a minimal sRGB ICC profile for testing
fn create_minimal_srgb_profile() -> Vec<u8> {
    // This is a minimal ICC profile that identifies as sRGB
    // In a real implementation, you'd use a proper sRGB profile
    let mut profile = Vec::new();

    // ICC profile header (simplified)
    profile.extend_from_slice(b"ADSP"); // Profile CMM type
    profile.extend_from_slice(b"sRGB"); // Color space signature
    profile.extend_from_slice(b"mntr"); // Device class (monitor)
    profile.extend_from_slice(b"RGB "); // Data color space
    profile.extend_from_slice(b"XYZ "); // PCS color space

    // Add some identifier text
    profile.extend_from_slice(b"Test sRGB Profile for background removal");

    // Pad to reasonable size (ICC profiles need minimum size)
    while profile.len() < 200 {
        profile.push(0);
    }

    profile
}

/// Run comprehensive color profile preservation tests
async fn run_color_profile_tests(
    input_path: &Path,
    output_dir: &Path,
) -> Result<ColorProfileTestResults> {
    let mut results = ColorProfileTestResults::new();

    println!("\n🔍 Testing Color Profile Extraction");
    println!("─────────────────────────────────────");

    // Test 1: Extract color profile from input image
    let input_profile = ProfileExtractor::extract_from_image(input_path)
        .context("Failed to extract color profile from input")?;

    match &input_profile {
        Some(profile) => {
            println!("  ✅ Input image has ICC profile:");
            println!("     Color Space: {}", profile.color_space);
            println!("     Profile Size: {} bytes", profile.data_size());
            results.input_has_profile = true;
        },
        None => {
            println!("  ℹ️  Input image has no ICC profile");
            results.input_has_profile = false;
        },
    }

    println!("\n🎯 Testing Background Removal with Color Profile Preservation");
    println!("──────────────────────────────────────────────────────────────");

    // Configure background removal with color profile preservation
    let config = ProcessorConfig::builder()
        .backend_type(BackendType::Onnx)
        .execution_provider(ExecutionProvider::Auto)
        .output_format(OutputFormat::Png)
        .preserve_color_profiles(true)
        .build()
        .context("Failed to build removal config")?;

    println!("  ├─ Config: preserve_color_profiles = true");
    println!("  ├─ Backend: {:?}", config.backend_type);
    println!("  └─ Output format: {:?}", config.output_format);

    // Create processor
    let backend_factory = Box::new(TestBackendFactory::new());
    let mut processor = BackgroundRemovalProcessor::with_factory(config, backend_factory)
        .context("Failed to create background removal processor")?;

    // Process the image
    println!("\n  🚀 Processing image...");
    let result = processor
        .process_file(input_path)
        .await
        .context("Failed to process image")?;

    // Check if result has color profile
    if let Some(ref profile) = result.get_color_profile() {
        println!("  ✅ Result contains color profile:");
        println!("     Color Space: {}", profile.color_space);
        println!("     Profile Size: {} bytes", profile.data_size());
        results.result_has_profile = true;
    } else {
        println!("  ❌ Result missing color profile");
        results.result_has_profile = false;
    }

    println!("\n💾 Testing Output with Color Profile Embedding");
    println!("───────────────────────────────────────────────");

    // Test output formats
    let test_formats = vec![(OutputFormat::Png, "png"), (OutputFormat::Jpeg, "jpg")];

    for (format, ext) in test_formats {
        let output_path = output_dir.join(format!("test_output_with_profile.{}", ext));

        println!("  ├─ Testing {} format...", ext.to_uppercase());

        // Save with color profile preservation
        (&result)
            .save_with_color_profile(&output_path, format, 90)
            .context(format!("Failed to save {} with color profile", ext))?;

        // Verify the profile was embedded
        let output_profile = ProfileExtractor::extract_from_image(&output_path)
            .context(format!("Failed to extract profile from {} output", ext))?;

        match output_profile {
            Some(profile) => {
                println!(
                    "     ✅ {} output has ICC profile ({}, {} bytes)",
                    ext.to_uppercase(),
                    profile.color_space,
                    profile.data_size()
                );
                results.add_format_result(format, true);
            },
            None => {
                println!("     ❌ {} output missing ICC profile", ext.to_uppercase());
                results.add_format_result(format, false);
            },
        }
    }

    // Test without color profile preservation for comparison
    println!("\n🔄 Testing without Color Profile Preservation (Control)");
    println!("───────────────────────────────────────────────────────");

    let config_no_profile = ProcessorConfig::builder()
        .backend_type(BackendType::Onnx)
        .execution_provider(ExecutionProvider::Auto)
        .output_format(OutputFormat::Png)
        .preserve_color_profiles(false)
        .build()
        .context("Failed to build config without profile preservation")?;

    let mut processor_no_profile = BackgroundRemovalProcessor::with_factory(
        config_no_profile,
        Box::new(TestBackendFactory::new()),
    )
    .context("Failed to create processor without profile preservation")?;

    let result_no_profile = processor_no_profile
        .process_file(input_path)
        .await
        .context("Failed to process image without profile preservation")?;

    let control_output_path = output_dir.join("test_output_no_profile.png");
    result_no_profile
        .save(&control_output_path, OutputFormat::Png, 0)
        .context("Failed to save control output")?;

    let control_profile = ProfileExtractor::extract_from_image(&control_output_path)
        .context("Failed to extract profile from control output")?;

    match control_profile {
        Some(_) => {
            println!("  ❌ Control output unexpectedly has ICC profile");
            results.control_missing_profile = false;
        },
        None => {
            println!("  ✅ Control output correctly has no ICC profile");
            results.control_missing_profile = true;
        },
    }

    Ok(results)
}

/// Display comprehensive test results
fn display_test_results(results: &ColorProfileTestResults) {
    println!("\n📊 Color Profile Test Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("Input Analysis:");
    println!(
        "  └─ Input has ICC profile: {}",
        if results.input_has_profile {
            "✅ Yes"
        } else {
            "❌ No"
        }
    );

    println!("Processing Analysis:");
    println!(
        "  └─ Result preserves profile: {}",
        if results.result_has_profile {
            "✅ Yes"
        } else {
            "❌ No"
        }
    );

    println!("Format-Specific Tests:");
    for (format, success) in &results.format_results {
        println!(
            "  ├─ {:?}: {}",
            format,
            if *success {
                "✅ Profile preserved"
            } else {
                "❌ Profile missing"
            }
        );
    }

    println!("Control Test:");
    println!(
        "  └─ No-profile mode works: {}",
        if results.control_missing_profile {
            "✅ Yes"
        } else {
            "❌ No"
        }
    );

    let total_tests = 2 + results.format_results.len() + 1;
    let passed_tests = results.count_passed();

    println!("\nSummary: {}/{} tests passed", passed_tests, total_tests);

    if results.all_passed() {
        println!("🎉 All color profile preservation tests successful!");
    } else {
        println!("⚠️  Some tests failed - check implementation");
    }
}

/// Test results tracking
#[derive(Debug)]
struct ColorProfileTestResults {
    input_has_profile: bool,
    result_has_profile: bool,
    format_results: Vec<(OutputFormat, bool)>,
    control_missing_profile: bool,
}

impl ColorProfileTestResults {
    fn new() -> Self {
        Self {
            input_has_profile: false,
            result_has_profile: false,
            format_results: Vec::new(),
            control_missing_profile: false,
        }
    }

    fn add_format_result(&mut self, format: OutputFormat, success: bool) {
        self.format_results.push((format, success));
    }

    fn count_passed(&self) -> usize {
        let mut passed = 0;

        // These tests pass regardless of whether input has profile
        passed += 1; // Input analysis always passes

        if self.result_has_profile == self.input_has_profile {
            passed += 1; // Result preservation matches input
        }

        for (_, success) in &self.format_results {
            if *success {
                passed += 1;
            }
        }

        if self.control_missing_profile {
            passed += 1;
        }

        passed
    }

    fn all_passed(&self) -> bool {
        // If input has profile, result should preserve it and format outputs should have it
        // If input has no profile, result shouldn't have one and format outputs shouldn't either
        // Control should never have profile

        let core_logic_correct = self.result_has_profile == self.input_has_profile;
        let formats_correct = self.format_results.iter().all(|(_, success)| {
            // Formats should preserve profile if input had one
            *success == self.input_has_profile
        });
        let control_correct = self.control_missing_profile;

        core_logic_correct && formats_correct && control_correct
    }
}
