//! Comprehensive integration tests for the unified BackgroundRemovalProcessor

use bg_remove_core::{
    config::RemovalConfig,
    error::Result,
    inference::InferenceBackend,
    models::{ModelInfo, ModelManager, PreprocessingConfig},
    processor::BackendFactory,
    BackendType, BackgroundRemovalProcessor, ExecutionProvider, OutputFormat,
    ProcessorConfigBuilder,
};
use image::DynamicImage;
use instant::Duration;
use ndarray::Array4;
use std::path::PathBuf;

/// Create a test image
fn create_test_image() -> DynamicImage {
    // Create a 100x100 RGB image with a red square in the center
    let mut img = image::ImageBuffer::new(100, 100);

    // Fill with white background
    for pixel in img.pixels_mut() {
        *pixel = image::Rgb([255, 255, 255]);
    }

    // Add red square in center (30x30)
    for x in 35..65 {
        for y in 35..65 {
            img.put_pixel(x, y, image::Rgb([255, 0, 0]));
        }
    }

    DynamicImage::ImageRgb8(img)
}

/// Save test image to a temporary file
fn save_test_image() -> PathBuf {
    let test_image = create_test_image();
    let temp_dir = std::env::temp_dir();
    let test_path = temp_dir.join(format!("test_image_unified_{}.png", std::process::id()));
    test_image
        .save(&test_path)
        .expect("Failed to save test image");
    test_path
}

/// Mock backend for testing
struct TestBackend {
    initialized: bool,
}

impl TestBackend {
    fn new() -> Self {
        Self { initialized: false }
    }
}

impl InferenceBackend for TestBackend {
    fn initialize(&mut self, _config: &RemovalConfig) -> Result<Option<Duration>> {
        self.initialized = true;
        Ok(Some(Duration::from_millis(10)))
    }

    fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
        // Return a simple mock mask (same shape as input but single channel)
        let (batch, _channels, height, width) = input.dim();
        // Create a simple circular mask in the center
        let mut output = Array4::<f32>::zeros((batch, 1, height, width));
        let center_x = width / 2;
        let center_y = height / 2;
        let radius = (width.min(height) / 3) as f32;

        for y in 0..height {
            for x in 0..width {
                let dist = ((x as f32 - center_x as f32).powi(2)
                    + (y as f32 - center_y as f32).powi(2))
                .sqrt();
                if dist < radius {
                    output[[0, 0, y, x]] = 1.0;
                }
            }
        }
        Ok(output)
    }

    fn input_shape(&self) -> (usize, usize, usize, usize) {
        (1, 3, 1024, 1024)
    }

    fn output_shape(&self) -> (usize, usize, usize, usize) {
        (1, 1, 1024, 1024)
    }

    fn get_preprocessing_config(&self) -> Result<PreprocessingConfig> {
        Ok(PreprocessingConfig {
            target_size: [1024, 1024],
            normalization_mean: [0.485, 0.456, 0.406],
            normalization_std: [0.229, 0.224, 0.225],
        })
    }

    fn get_model_info(&self) -> Result<ModelInfo> {
        Ok(ModelInfo {
            name: "test_model".to_string(),
            precision: "fp32".to_string(),
            size_bytes: 1024,
            input_shape: (1, 3, 1024, 1024),
            output_shape: (1, 1, 1024, 1024),
        })
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Test backend factory
struct TestBackendFactory;

impl BackendFactory for TestBackendFactory {
    fn create_backend(
        &self,
        _backend_type: BackendType,
        _model_manager: ModelManager,
    ) -> Result<Box<dyn InferenceBackend>> {
        Ok(Box::new(TestBackend::new()))
    }

    fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::Onnx] // Report that we support Onnx for test purposes
    }
}

/// Helper function to create a test processor with mock backend
fn create_test_processor(
    config: bg_remove_core::processor::ProcessorConfig,
) -> Result<BackgroundRemovalProcessor> {
    let factory = Box::new(TestBackendFactory);
    BackgroundRemovalProcessor::with_factory(config, factory)
}

#[tokio::test]
async fn test_unified_processor_basic_workflow() -> Result<()> {
    // Create processor config
    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .execution_provider(ExecutionProvider::Cpu)
        .output_format(OutputFormat::Png)
        .build()?;

    // Create processor with test backend factory
    let mut processor = create_test_processor(config)?;

    // Process test image
    let test_path = save_test_image();
    let result = processor.process_file(&test_path).await?;

    // Verify result
    assert_eq!(result.original_dimensions, (100, 100));
    assert_eq!(result.mask.dimensions, (100, 100));
    assert!(result.metadata.model_name.contains("unified_processor"));

    // Clean up
    std::fs::remove_file(test_path).ok();

    Ok(())
}

#[test]
fn test_unified_processor_direct_image_processing() -> Result<()> {
    // Create processor config
    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .execution_provider(ExecutionProvider::Cpu)
        .output_format(OutputFormat::Png)
        .build()?;

    // Create processor with test backend factory
    let mut processor = create_test_processor(config)?;

    // Process image directly
    let test_image = create_test_image();
    let result = processor.process_image(&test_image)?;

    // Verify result
    assert_eq!(result.original_dimensions, (100, 100));
    assert_eq!(result.mask.dimensions, (100, 100));
    assert!(result.metadata.model_name.contains("unified_processor"));

    Ok(())
}

#[test]
fn test_processor_config_builder() -> Result<()> {
    // Test various configurations
    let configs = vec![
        ProcessorConfigBuilder::new()
            .backend_type(BackendType::Onnx)
            .execution_provider(ExecutionProvider::Cpu)
            .output_format(OutputFormat::Png)
            .jpeg_quality(95)
            .build()?,
        ProcessorConfigBuilder::new()
            .backend_type(BackendType::Onnx)
            .execution_provider(ExecutionProvider::Auto)
            .output_format(OutputFormat::Jpeg)
            .jpeg_quality(85)
            .build()?,
        ProcessorConfigBuilder::new()
            .backend_type(BackendType::Onnx)
            .execution_provider(ExecutionProvider::Cpu)
            .output_format(OutputFormat::WebP)
            .webp_quality(90)
            .build()?,
    ];

    for config in configs {
        let processor = create_test_processor(config)?;
        // Processor with factory should be ready to initialize
        assert_eq!(processor.config().backend_type, BackendType::Onnx);
    }

    Ok(())
}

#[test]
fn test_processor_initialization() -> Result<()> {
    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .build()?;

    let processor = create_test_processor(config)?;

    // Test processor should be ready to initialize
    assert_eq!(processor.config().backend_type, BackendType::Onnx);

    Ok(())
}

#[tokio::test]
async fn test_output_format_handling() -> Result<()> {
    // Create and save test image
    let test_image = create_test_image();
    let temp_dir = std::env::temp_dir();
    let test_path = temp_dir.join(format!("test_output_format_{}.png", std::process::id()));
    test_image
        .save(&test_path)
        .expect("Failed to save test image");

    // Ensure file exists and is readable
    assert!(
        test_path.exists(),
        "Test image was not created at {:?}",
        test_path
    );
    let _metadata = std::fs::metadata(&test_path).expect("Cannot read test file metadata");

    // Test PNG output
    let config_png = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .output_format(OutputFormat::Png)
        .build()?;
    let mut processor_png = create_test_processor(config_png)?;

    println!("Processing PNG with file at: {:?}", test_path);
    let result_png = processor_png.process_file(&test_path).await?;

    // PNG should have alpha channel
    match &result_png.image {
        DynamicImage::ImageRgba8(_) => {},
        _ => panic!("Expected RGBA8 for PNG output"),
    }

    // Test JPEG output
    let config_jpeg = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .output_format(OutputFormat::Jpeg)
        .build()?;
    let mut processor_jpeg = create_test_processor(config_jpeg)?;

    println!("Processing JPEG with file at: {:?}", test_path);
    let result_jpeg = processor_jpeg.process_file(&test_path).await?;

    // JPEG should be RGB (no alpha)
    match &result_jpeg.image {
        DynamicImage::ImageRgb8(_) => {},
        _ => panic!("Expected RGB8 for JPEG output"),
    }

    // Clean up
    std::fs::remove_file(&test_path).expect("Failed to remove test file");

    Ok(())
}

#[tokio::test]
async fn test_segment_foreground() -> Result<()> {
    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .build()?;

    let mut processor = create_test_processor(config)?;

    // Create and save test image
    let test_path = save_test_image();

    // Extract mask only
    let mask = processor.segment_foreground(&test_path).await?;

    // Verify mask
    assert_eq!(mask.dimensions, (100, 100));
    let stats = mask.statistics();
    assert!(stats.foreground_ratio > 0.0);
    assert!(stats.foreground_ratio < 1.0);

    // Clean up
    std::fs::remove_file(test_path).ok();

    Ok(())
}

#[tokio::test]
async fn test_apply_mask() -> Result<()> {
    // Create and save test image
    let test_image = create_test_image();
    let temp_dir = std::env::temp_dir();
    let test_path = temp_dir.join(format!("test_apply_mask_{}.png", std::process::id()));
    test_image
        .save(&test_path)
        .expect("Failed to save test image");

    // Ensure file exists and is readable
    assert!(
        test_path.exists(),
        "Test image was not created at {:?}",
        test_path
    );
    let _metadata = std::fs::metadata(&test_path).expect("Cannot read test file metadata");

    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .output_format(OutputFormat::Png)
        .build()?;

    let mut processor = create_test_processor(config)?;

    println!("Extracting mask from file at: {:?}", test_path);
    // First extract mask
    let mask = processor.segment_foreground(&test_path).await?;

    println!("Applying mask to file at: {:?}", test_path);
    // Apply mask to same image
    let result = processor.apply_mask(&test_path, &mask).await?;

    // Verify result
    assert_eq!(result.original_dimensions, (100, 100));
    assert_eq!(result.mask.dimensions, (100, 100));
    assert_eq!(result.metadata.model_name, "mask_application");

    // Clean up
    std::fs::remove_file(&test_path).expect("Failed to remove test file");

    Ok(())
}

#[test]
fn test_available_backends() -> Result<()> {
    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .build()?;

    let processor = create_test_processor(config)?;
    let backends = processor.available_backends();

    // Test factory should provide Onnx backend
    assert!(backends.contains(&BackendType::Onnx));

    Ok(())
}

#[test]
fn test_config_validation() {
    // Test invalid JPEG quality
    let result = ProcessorConfigBuilder::new()
        .jpeg_quality(150) // Over 100, should be clamped
        .build();

    // Should succeed but clamp value
    assert!(result.is_ok());
    let config = result.unwrap();
    assert_eq!(config.jpeg_quality, 100);

    // Test invalid WebP quality
    let result = ProcessorConfigBuilder::new()
        .webp_quality(200) // Over 100, should be clamped
        .build();

    // Should succeed but clamp value
    assert!(result.is_ok());
    let config = result.unwrap();
    assert_eq!(config.webp_quality, 100);
}

#[test]
fn test_processor_debug_mode() -> Result<()> {
    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .debug(true)
        .build()?;

    let processor = create_test_processor(config)?;
    assert!(processor.config().debug);

    Ok(())
}

#[test]
fn test_processor_thread_configuration() -> Result<()> {
    let config = ProcessorConfigBuilder::new()
        .backend_type(BackendType::Onnx)
        .intra_threads(4)
        .inter_threads(2)
        .build()?;

    let processor = create_test_processor(config)?;
    assert_eq!(processor.config().intra_threads, 4);
    assert_eq!(processor.config().inter_threads, 2);

    Ok(())
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    #[tokio::test]
    async fn test_processing_timings() -> Result<()> {
        let config = ProcessorConfigBuilder::new()
            .backend_type(BackendType::Onnx)
            .build()?;

        let mut processor = create_test_processor(config)?;
        let test_path = save_test_image();

        let result = processor.process_file(&test_path).await?;

        // Verify timing metadata is populated
        let timings = &result.metadata.timings;
        assert!(timings.total_ms > 0);
        // Timing fields exist (u64 so always >= 0)
        let _ = timings.preprocessing_ms;
        let _ = timings.inference_ms;
        let _ = timings.postprocessing_ms;

        // Total should be sum of parts (approximately)
        let sum = timings.preprocessing_ms + timings.inference_ms + timings.postprocessing_ms;
        assert!(timings.total_ms >= sum);

        // Clean up
        std::fs::remove_file(test_path).ok();

        Ok(())
    }
}
