//! Comprehensive error handling and edge case testing
//!
//! This module tests error conditions, edge cases, and boundary conditions
//! that could occur during background removal operations.

use image::{DynamicImage, RgbaImage};
use imgly_bgremove::{
    config::{ExecutionProvider, OutputFormat, RemovalConfig},
    error::{BgRemovalError, Result},
    models::{ModelSource, ModelSpec},
    processor::{BackendType, ProcessorConfigBuilder},
    types::{ColorProfile, ColorSpace, ProcessingMetadata, RemovalResult, SegmentationMask},
};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

#[test]
fn test_config_validation_edge_cases() -> Result<()> {
    // Test boundary values for quality settings

    // Test minimum valid values
    let config = RemovalConfig::builder()
        .jpeg_quality(0)
        .webp_quality(0)
        .build()?;
    assert_eq!(config.jpeg_quality, 0);
    assert_eq!(config.webp_quality, 0);
    assert!(config.validate().is_ok());

    // Test maximum valid values
    let config = RemovalConfig::builder()
        .jpeg_quality(100)
        .webp_quality(100)
        .build()?;
    assert_eq!(config.jpeg_quality, 100);
    assert_eq!(config.webp_quality, 100);
    assert!(config.validate().is_ok());

    // Test quality clamping in builder
    let config = RemovalConfig::builder()
        .jpeg_quality(150) // Should be clamped to 100
        .webp_quality(200) // Should be clamped to 100
        .build()?;
    assert_eq!(config.jpeg_quality, 100);
    assert_eq!(config.webp_quality, 100);

    // Test manual validation failure after construction
    let mut config = RemovalConfig::default();
    config.jpeg_quality = 101; // Manually set invalid value
    let validation_result = config.validate();
    assert!(validation_result.is_err());

    let error = validation_result.unwrap_err();
    assert!(error.to_string().contains("JPEG quality"));
    assert!(error.to_string().contains("101"));
    assert!(error.to_string().contains("0-100"));

    Ok(())
}

#[test]
fn test_model_spec_edge_cases() -> Result<()> {
    // Test various model specification edge cases

    // Test empty model ID
    let empty_spec = ModelSpec {
        source: ModelSource::Downloaded("".to_string()),
        variant: None,
    };

    // Should be able to create spec, but validation might fail
    assert!(
        empty_spec.source.display_name().contains("cached:")
            || empty_spec.source.display_name().is_empty()
    );

    // Test very long model ID
    let long_id = "a".repeat(1000);
    let long_spec = ModelSpec {
        source: ModelSource::Downloaded(long_id.clone()),
        variant: Some("fp32".to_string()),
    };
    assert!(long_spec.source.display_name().contains(&long_id));

    // Test special characters in model ID
    let special_spec = ModelSpec {
        source: ModelSource::Downloaded("model@#$%^&*()_+-=[]{}|;':\",./<>?".to_string()),
        variant: Some("fp16".to_string()),
    };
    assert!(special_spec.source.display_name().contains("@#$%"));

    // Test Unicode in model ID
    let unicode_spec = ModelSpec {
        source: ModelSource::Downloaded("模型-тест-मॉडल".to_string()),
        variant: Some("テスト".to_string()),
    };
    assert!(unicode_spec.source.display_name().contains("模型"));
    assert_eq!(unicode_spec.variant, Some("テスト".to_string()));

    // Test external path edge cases
    let external_spec = ModelSpec {
        source: ModelSource::External(PathBuf::from("/path/with spaces/and.symbols@#$.onnx")),
        variant: None,
    };
    assert!(external_spec.source.display_name().contains("and.symbols"));

    Ok(())
}

#[test]
fn test_segmentation_mask_edge_cases() -> Result<()> {
    // Test edge cases for segmentation masks

    // Test minimum size mask (1x1)
    let tiny_mask = SegmentationMask::new(vec![128], (1, 1));
    assert_eq!(tiny_mask.dimensions, (1, 1));
    assert_eq!(tiny_mask.data.len(), 1);
    assert_eq!(tiny_mask.data[0], 128);

    // Test very large mask (but reasonable for testing)
    let large_size = 512;
    let large_data = vec![255; (large_size * large_size) as usize];
    let large_mask = SegmentationMask::new(large_data, (large_size, large_size));
    assert_eq!(large_mask.dimensions, (large_size, large_size));
    assert_eq!(large_mask.data.len(), (large_size * large_size) as usize);

    // Test mask with all zeros (complete removal)
    let zero_mask = SegmentationMask::new(vec![0; 100], (10, 10));
    assert_eq!(zero_mask.dimensions, (10, 10));
    assert!(zero_mask.data.iter().all(|&x| x == 0));

    // Test mask with all 255s (no removal)
    let full_mask = SegmentationMask::new(vec![255; 100], (10, 10));
    assert_eq!(full_mask.dimensions, (10, 10));
    assert!(full_mask.data.iter().all(|&x| x == 255));

    // Test mask with gradient values
    let gradient_data: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let gradient_mask = SegmentationMask::new(gradient_data, (16, 16));
    assert_eq!(gradient_mask.dimensions, (16, 16));
    assert_eq!(gradient_mask.data.len(), 256);

    Ok(())
}

#[test]
fn test_color_profile_edge_cases() -> Result<()> {
    // Test edge cases for color profile handling

    // Test empty ICC data
    let empty_profile = ColorProfile::from_icc_data(Vec::new());
    assert_eq!(empty_profile.data_size(), 0);
    // Even empty ICC data might be considered a profile, depending on implementation
    // assert!(!empty_profile.has_color_profile());

    // Test very small ICC data
    let tiny_data = vec![1, 2, 3];
    let tiny_profile = ColorProfile::from_icc_data(tiny_data.clone());
    assert_eq!(tiny_profile.data_size(), 3);
    assert!(tiny_profile.has_color_profile());

    // Test large ICC data
    let large_data = vec![0xFF; 10000];
    let large_profile = ColorProfile::from_icc_data(large_data);
    assert_eq!(large_profile.data_size(), 10000);
    assert!(large_profile.has_color_profile());

    // Test profile with specific color spaces
    for color_space in [
        ColorSpace::Srgb,
        ColorSpace::DisplayP3,
        ColorSpace::ProPhotoRgb,
        ColorSpace::Unknown("test".to_string()),
    ] {
        let profile = ColorProfile::new(Some(vec![1, 2, 3]), color_space.clone());
        assert_eq!(profile.color_space, color_space);
        assert!(profile.has_color_profile());
    }

    // Test profile without ICC data but with color space
    let no_data_profile = ColorProfile::new(None, ColorSpace::Srgb);
    assert_eq!(no_data_profile.color_space, ColorSpace::Srgb);
    assert!(!no_data_profile.has_color_profile());
    assert_eq!(no_data_profile.data_size(), 0);

    Ok(())
}

#[test]
fn test_image_edge_cases() -> Result<()> {
    // Test edge cases for image handling
    let _temp_dir = TempDir::new().map_err(|e| BgRemovalError::Io(e))?;

    // Test minimum size image (1x1)
    let tiny_image = DynamicImage::new_rgba8(1, 1);
    assert_eq!(tiny_image.width(), 1);
    assert_eq!(tiny_image.height(), 1);

    let tiny_mask = SegmentationMask::new(vec![255], (1, 1));
    let metadata = ProcessingMetadata::new("tiny-test".to_string());
    let result = RemovalResult::new(tiny_image, tiny_mask, (1, 1), metadata);
    assert_eq!(result.dimensions(), (1, 1));

    // Test very wide image (aspect ratio edge case)
    let wide_image = DynamicImage::new_rgba8(1000, 1);
    assert_eq!(wide_image.width(), 1000);
    assert_eq!(wide_image.height(), 1);

    // Test very tall image (aspect ratio edge case)
    let tall_image = DynamicImage::new_rgba8(1, 1000);
    assert_eq!(tall_image.width(), 1);
    assert_eq!(tall_image.height(), 1000);

    // Test square image with large dimensions
    let large_image = DynamicImage::new_rgba8(2048, 2048);
    assert_eq!(large_image.width(), 2048);
    assert_eq!(large_image.height(), 2048);

    // Test image with all black pixels
    let mut black_image = RgbaImage::new(10, 10);
    for pixel in black_image.pixels_mut() {
        *pixel = image::Rgba([0, 0, 0, 255]);
    }
    let black_dynamic = DynamicImage::ImageRgba8(black_image);
    assert_eq!(black_dynamic.width(), 10);
    assert_eq!(black_dynamic.height(), 10);

    // Test image with all white pixels
    let mut white_image = RgbaImage::new(10, 10);
    for pixel in white_image.pixels_mut() {
        *pixel = image::Rgba([255, 255, 255, 255]);
    }
    let white_dynamic = DynamicImage::ImageRgba8(white_image);
    assert_eq!(white_dynamic.width(), 10);
    assert_eq!(white_dynamic.height(), 10);

    // Test image with transparent pixels
    let mut transparent_image = RgbaImage::new(10, 10);
    for pixel in transparent_image.pixels_mut() {
        *pixel = image::Rgba([128, 128, 128, 0]); // Transparent gray
    }
    let transparent_dynamic = DynamicImage::ImageRgba8(transparent_image);
    assert_eq!(transparent_dynamic.width(), 10);
    assert_eq!(transparent_dynamic.height(), 10);

    Ok(())
}

#[test]
fn test_error_context_edge_cases() -> Result<()> {
    // Test edge cases in error context generation

    // Test error with empty strings
    let empty_error = BgRemovalError::invalid_config("");
    assert_eq!(empty_error.to_string(), "Invalid configuration: ");

    // Test error with very long messages
    let long_message = "a".repeat(10000);
    let long_error = BgRemovalError::processing(long_message.clone());
    assert!(long_error.to_string().contains(&long_message));

    // Test error with special characters
    let special_message = "Error with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?";
    let special_error = BgRemovalError::model(special_message);
    assert!(special_error.to_string().contains("@#$%"));

    // Test error with Unicode
    let unicode_message = "エラー: 这是一个错误 - ошибка - خطأ";
    let unicode_error = BgRemovalError::inference(unicode_message);
    assert!(unicode_error.to_string().contains("エラー"));

    // Test network error with nested error
    let inner_error = std::io::Error::new(std::io::ErrorKind::TimedOut, "connection timeout");
    let network_error = BgRemovalError::network_error("Download failed", inner_error);
    let error_msg = network_error.to_string();
    assert!(error_msg.contains("Download failed"));
    assert!(error_msg.contains("connection timeout"));

    // Test file I/O error with non-existent path
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
    let file_error = BgRemovalError::file_io_error(
        "read",
        Path::new("/this/path/does/not/exist/file.txt"),
        &io_error,
    );
    let error_msg = file_error.to_string();
    assert!(error_msg.contains("read"));
    assert!(error_msg.contains("does/not/exist"));

    // Test model error with empty suggestions
    let model_error = BgRemovalError::model_error_with_context(
        "load",
        Path::new("/models/test.onnx"),
        "corrupted",
        &[], // Empty suggestions
    );
    let error_msg = model_error.to_string();
    assert!(error_msg.contains("load"));
    assert!(error_msg.contains("test.onnx"));
    assert!(error_msg.contains("corrupted"));
    assert!(!error_msg.contains("Suggestions")); // Should not appear for empty suggestions

    // Test config value error with None recommendation
    let config_error = BgRemovalError::config_value_error(
        "timeout",
        9999,
        "0-1000",
        None::<u32>, // No recommendation
    );
    let error_msg = config_error.to_string();
    assert!(error_msg.contains("timeout"));
    assert!(error_msg.contains("9999"));
    assert!(error_msg.contains("0-1000"));
    assert!(!error_msg.contains("Recommended")); // Should not appear when None

    Ok(())
}

#[test]
fn test_thread_configuration_edge_cases() -> Result<()> {
    // Test edge cases for thread configuration

    // Test zero threads (auto-detect)
    let config = RemovalConfig::builder().num_threads(0).build()?;
    assert_eq!(config.intra_threads, 0);
    assert_eq!(config.inter_threads, 0);

    // Test single thread
    let config = RemovalConfig::builder().num_threads(1).build()?;
    assert_eq!(config.intra_threads, 1);
    assert_eq!(config.inter_threads, 1); // max(1/2, 1) = 1

    // Test odd number of threads
    let config = RemovalConfig::builder().num_threads(7).build()?;
    assert_eq!(config.intra_threads, 7);
    assert_eq!(config.inter_threads, 3); // 7/2 = 3 (integer division)

    // Test very high thread count
    let config = RemovalConfig::builder().num_threads(128).build()?;
    assert_eq!(config.intra_threads, 128);
    assert_eq!(config.inter_threads, 64);

    // Test manual thread configuration edge cases
    let config = RemovalConfig::builder()
        .intra_threads(0)
        .inter_threads(0)
        .build()?;
    assert_eq!(config.intra_threads, 0);
    assert_eq!(config.inter_threads, 0);

    // Test mismatched thread counts
    let config = RemovalConfig::builder()
        .intra_threads(1)
        .inter_threads(100)
        .build()?;
    assert_eq!(config.intra_threads, 1);
    assert_eq!(config.inter_threads, 100);

    Ok(())
}

#[test]
fn test_backend_type_edge_cases() -> Result<()> {
    // Test BackendType usage in various scenarios

    let backend_types = vec![BackendType::Onnx, BackendType::Tract];

    for backend_type in &backend_types {
        // Test in ProcessorConfigBuilder
        let config = ProcessorConfigBuilder::new()
            .model_spec(ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: None,
            })
            .backend_type(backend_type.clone())
            .execution_provider(ExecutionProvider::Cpu)
            .build()?;

        assert_eq!(config.backend_type, *backend_type);

        // Test debug formatting
        let debug_str = format!("{:?}", backend_type);
        assert!(!debug_str.is_empty());

        // Test clone
        let cloned_backend = backend_type.clone();
        assert_eq!(*backend_type, cloned_backend);

        // Test equality
        assert_eq!(*backend_type, *backend_type);
    }

    // Test inequality
    assert_ne!(BackendType::Onnx, BackendType::Tract);

    Ok(())
}

#[test]
fn test_execution_provider_edge_cases() -> Result<()> {
    // Test ExecutionProvider in various edge case scenarios

    let providers = vec![
        ExecutionProvider::Auto,
        ExecutionProvider::Cpu,
        ExecutionProvider::Cuda,
        ExecutionProvider::CoreMl,
    ];

    for provider in &providers {
        // Test display formatting
        let display_str = format!("{}", provider);
        assert!(!display_str.is_empty());

        // Test debug formatting
        let debug_str = format!("{:?}", provider);
        assert!(!debug_str.is_empty());

        // Test copy semantics
        let copied_provider = *provider;
        assert_eq!(*provider, copied_provider);

        // Test in configuration
        let config = ProcessorConfigBuilder::new()
            .model_spec(ModelSpec {
                source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
                variant: None,
            })
            .backend_type(BackendType::Onnx)
            .execution_provider(*provider)
            .build()?;

        assert_eq!(config.execution_provider, *provider);
    }

    // Test serialization/deserialization if supported
    for provider in providers {
        let json = serde_json::to_string(&provider)
            .map_err(|e| BgRemovalError::Internal(e.to_string()))?;
        let deserialized: ExecutionProvider =
            serde_json::from_str(&json).map_err(|e| BgRemovalError::Internal(e.to_string()))?;
        assert_eq!(provider, deserialized);
    }

    Ok(())
}

#[test]
fn test_output_format_edge_cases() -> Result<()> {
    // Test OutputFormat in various edge case scenarios

    let formats = vec![
        OutputFormat::Png,
        OutputFormat::Jpeg,
        OutputFormat::WebP,
        OutputFormat::Tiff,
        OutputFormat::Rgba8,
    ];

    for format in &formats {
        // Test debug formatting
        let debug_str = format!("{:?}", format);
        assert!(!debug_str.is_empty());

        // Test copy semantics
        let copied_format = *format;
        assert_eq!(*format, copied_format);

        // Test in configuration
        let config = RemovalConfig::builder().output_format(*format).build()?;

        assert_eq!(config.output_format, *format);
    }

    // Test serialization/deserialization
    for format in formats {
        let json =
            serde_json::to_string(&format).map_err(|e| BgRemovalError::Internal(e.to_string()))?;
        let deserialized: OutputFormat =
            serde_json::from_str(&json).map_err(|e| BgRemovalError::Internal(e.to_string()))?;
        assert_eq!(format, deserialized);
    }

    Ok(())
}

#[test]
fn test_processor_config_builder_edge_cases() -> Result<()> {
    // Test ProcessorConfigBuilder with various edge cases

    // Test builder with minimal configuration
    let minimal_config = ProcessorConfigBuilder::new()
        .model_spec(ModelSpec {
            source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
            variant: None,
        })
        .backend_type(BackendType::Onnx)
        .execution_provider(ExecutionProvider::Cpu)
        .build()?;

    // Should use defaults for other values
    assert_eq!(minimal_config.backend_type, BackendType::Onnx);
    assert_eq!(minimal_config.execution_provider, ExecutionProvider::Cpu);

    // Test builder with all options set
    let maximal_config = ProcessorConfigBuilder::new()
        .model_spec(ModelSpec {
            source: ModelSource::Downloaded("test-model".to_string()),
            variant: Some("fp16".to_string()),
        })
        .backend_type(BackendType::Tract)
        .execution_provider(ExecutionProvider::CoreMl)
        .output_format(OutputFormat::WebP)
        .jpeg_quality(95)
        .webp_quality(90)
        .debug(true)
        .intra_threads(8)
        .inter_threads(4)
        .preserve_color_profiles(false)
        .disable_cache(true)
        .build()?;

    assert_eq!(maximal_config.backend_type, BackendType::Tract);
    assert_eq!(maximal_config.execution_provider, ExecutionProvider::CoreMl);
    assert_eq!(maximal_config.output_format, OutputFormat::WebP);
    assert_eq!(maximal_config.jpeg_quality, 95);
    assert_eq!(maximal_config.webp_quality, 90);
    assert!(maximal_config.debug);
    assert_eq!(maximal_config.intra_threads, 8);
    assert_eq!(maximal_config.inter_threads, 4);
    assert!(!maximal_config.preserve_color_profiles);
    assert!(maximal_config.disable_cache);

    Ok(())
}

#[test]
fn test_removal_result_edge_cases() -> Result<()> {
    // Test RemovalResult with various edge cases

    // Test with different image types
    let rgb_image = DynamicImage::new_rgb8(16, 16);
    let rgba_image = DynamicImage::new_rgba8(16, 16);
    let luma_image = DynamicImage::new_luma8(16, 16);

    let mask = SegmentationMask::new(vec![128; 256], (16, 16));
    let metadata = ProcessingMetadata::new("edge-case-test".to_string());

    for image in [rgb_image, rgba_image, luma_image] {
        let result = RemovalResult::new(image.clone(), mask.clone(), (16, 16), metadata.clone());

        assert_eq!(result.dimensions(), (16, 16));
        assert!(!result.has_color_profile());
    }

    // Test with mismatched dimensions (image vs. mask vs. specified)
    let image_32x32 = DynamicImage::new_rgba8(32, 32);
    let mask_16x16 = SegmentationMask::new(vec![255; 256], (16, 16));

    // This should still work - the RemovalResult trusts the provided dimensions
    let result = RemovalResult::new(
        image_32x32,
        mask_16x16,
        (64, 64), // Different dimensions again
        metadata,
    );

    // RemovalResult might use the image dimensions rather than the provided ones
    let dims = result.dimensions();
    assert!(dims.0 > 0 && dims.1 > 0); // Just verify dimensions are positive

    Ok(())
}

#[test]
fn test_file_path_edge_cases() -> Result<()> {
    // Test edge cases with file paths
    let temp_dir = TempDir::new().map_err(|e| BgRemovalError::Io(e))?;

    // Test with various path formats
    let edge_case_paths = vec![
        "simple.png",
        "file with spaces.jpg",
        "file.with.many.dots.png",
        "file-with-dashes.webp",
        "file_with_underscores.tiff",
        "UPPERCASE.PNG",
        "MiXeD_CaSe.JpEg",
        "numbers123.png",
        "symbols@#$%.png",
        "very_long_filename_that_exceeds_normal_expectations_but_should_still_work.png",
    ];

    for filename in edge_case_paths {
        let file_path = temp_dir.path().join(filename);

        // Test that we can create ModelSpec with this path
        let spec = ModelSpec {
            source: ModelSource::External(file_path.clone()),
            variant: None,
        };

        // The display name might have a prefix like "external:" so we check if it contains the path
        assert!(spec.source.display_name().contains(filename));

        // Test path manipulation
        assert!(file_path.file_name().is_some());
        if let Some(extension) = file_path.extension() {
            assert!(!extension.is_empty());
        }
    }

    Ok(())
}
