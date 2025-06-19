use bg_remove_core::config::ExecutionProvider;
use bg_remove_core::{BackgroundRemovalProcessor, ProcessorConfigBuilder, ModelSpec, ModelSource, BackendType, BackendFactory};
use bg_remove_onnx::OnnxBackend;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{DynamicImage, RgbImage};
use std::time::Duration;

/// ONNX backend factory for benchmarks
struct OnnxBackendFactory;

impl BackendFactory for OnnxBackendFactory {
    fn create_backend(
        &self,
        backend_type: BackendType,
        model_manager: bg_remove_core::ModelManager,
    ) -> bg_remove_core::Result<Box<dyn bg_remove_core::InferenceBackend>> {
        match backend_type {
            BackendType::Onnx => {
                let mut backend = Box::new(OnnxBackend::new());
                backend.set_model_manager(model_manager);
                Ok(backend as Box<dyn bg_remove_core::InferenceBackend>)
            }
            BackendType::Tract => Err(bg_remove_core::BgRemovalError::invalid_config(
                "Tract backend not available in benchmarks",
            )),
        }
    }

    fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::Onnx]
    }
}

fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);

    // Create a simple gradient pattern that resembles a portrait
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = ((x as f32 / width as f32) * 255.0) as u8;
        let g = ((y as f32 / height as f32) * 255.0) as u8;
        let b = 128; // Constant blue for simplicity
        *pixel = image::Rgb([r, g, b]);
    }

    DynamicImage::ImageRgb8(img)
}

fn benchmark_providers(c: &mut Criterion) {
    let mut group = c.benchmark_group("background_removal/provider");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    let test_image = create_test_image(512, 512);

    // Test different execution providers
    let providers = [
        (ExecutionProvider::Cpu, "cpu"),
        (ExecutionProvider::Auto, "auto"),
        (ExecutionProvider::CoreMl, "coreml"),
    ];

    for (provider, name) in providers {

        group.bench_function(name, |b| {
            b.iter(|| {
                let model_spec = ModelSpec {
                    source: ModelSource::Embedded("isnet-fp32".to_string()),
                    variant: None,
                };
                
                let processor_config = ProcessorConfigBuilder::new()
                    .model_spec(model_spec)
                    .backend_type(BackendType::Onnx)
                    .execution_provider(provider)
                    .build()
                    .unwrap();

                let backend_factory = Box::new(OnnxBackendFactory);
                let mut processor = BackgroundRemovalProcessor::with_factory(processor_config, backend_factory).unwrap();
                let result = processor.process_image(&test_image);
                black_box(result.unwrap())
            });
        });
    }
    group.finish();
}

fn benchmark_image_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_sizes/category");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let execution_provider = ExecutionProvider::Auto;

    // Test different image size categories
    let sizes = [
        ("portrait", 512, 768),   // Typical portrait ratio
        ("product", 1024, 1024),  // Square product image
        ("complex", 1920, 1080),  // Full HD landscape
    ];

    for (name, width, height) in sizes {
        let test_image = create_test_image(width, height);

        group.bench_function(name, |b| {
            b.iter(|| {
                let model_spec = ModelSpec {
                    source: ModelSource::Embedded("isnet-fp32".to_string()),
                    variant: None,
                };
                
                let processor_config = ProcessorConfigBuilder::new()
                    .model_spec(model_spec)
                    .backend_type(BackendType::Onnx)
                    .execution_provider(execution_provider)
                    .build()
                    .unwrap();

                let backend_factory = Box::new(OnnxBackendFactory);
                let mut processor = BackgroundRemovalProcessor::with_factory(processor_config, backend_factory).unwrap();
                let result = processor.process_image(&test_image);
                black_box(result.unwrap())
            });
        });
    }
    group.finish();
}

fn benchmark_model_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_variants");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let test_image = create_test_image(512, 512);

    // Test different model precisions (if available)
    let models = [
        ("isnet-fp32", "isnet-fp32"),
        // Add more models when available
        // ("isnet-fp16", "isnet-fp16"),
    ];

    for (name, model_name) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let model_spec = ModelSpec {
                    source: ModelSource::Embedded(model_name.to_string()),
                    variant: None,
                };
                
                let processor_config = ProcessorConfigBuilder::new()
                    .model_spec(model_spec)
                    .backend_type(BackendType::Onnx)
                    .execution_provider(ExecutionProvider::Auto)
                    .build()
                    .unwrap();

                let backend_factory = Box::new(OnnxBackendFactory);
                let mut processor = BackgroundRemovalProcessor::with_factory(processor_config, backend_factory).unwrap();
                let result = processor.process_image(&test_image);
                black_box(result.unwrap())
            });
        });
    }
    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    group.sample_size(10); // Keep minimum required samples
    group.measurement_time(Duration::from_secs(60)); // Longer measurement time for batch processing

    // Create 10 test images of different sizes to simulate real batch processing
    let test_images: Vec<DynamicImage> = (0..10)
        .map(|i| {
            let size = 512 + (i * 32); // Varying sizes from 512 to 800
            create_test_image(size, size)
        })
        .collect();

    let providers = [
        (ExecutionProvider::Cpu, "cpu"),
        (ExecutionProvider::Auto, "auto"), 
        (ExecutionProvider::CoreMl, "coreml"),
    ];

    for (provider, name) in providers {
        group.bench_function(&format!("{}_batch_10_images", name), |b| {
            b.iter(|| {
                // Create and initialize processor once
                let model_spec = ModelSpec {
                    source: ModelSource::Embedded("isnet-fp32".to_string()),
                    variant: None,
                };
                
                let processor_config = ProcessorConfigBuilder::new()
                    .model_spec(model_spec)
                    .backend_type(BackendType::Onnx)
                    .execution_provider(provider)
                    .build()
                    .unwrap();

                let backend_factory = Box::new(OnnxBackendFactory);
                let mut processor = BackgroundRemovalProcessor::with_factory(processor_config, backend_factory).unwrap();
                
                // Initialize the processor once (this includes model loading/compilation)
                processor.initialize().unwrap();
                
                // Process all 10 images with the same initialized processor
                let mut results = Vec::with_capacity(10);
                for image in &test_images {
                    let result = processor.process_image(image).unwrap();
                    results.push(result);
                }
                
                black_box(results)
            });
        });
        
        // Also benchmark per-image throughput (total time / 10 images)
        group.bench_function(&format!("{}_per_image_in_batch", name), |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::new(0, 0);
                
                for _ in 0..iters {
                    let start = std::time::Instant::now();
                    
                    // Create and initialize processor once
                    let model_spec = ModelSpec {
                        source: ModelSource::Embedded("isnet-fp32".to_string()),
                        variant: None,
                    };
                    
                    let processor_config = ProcessorConfigBuilder::new()
                        .model_spec(model_spec)
                        .backend_type(BackendType::Onnx)
                        .execution_provider(provider)
                        .build()
                        .unwrap();

                    let backend_factory = Box::new(OnnxBackendFactory);
                    let mut processor = BackgroundRemovalProcessor::with_factory(processor_config, backend_factory).unwrap();
                    processor.initialize().unwrap();
                    
                    // Process all 10 images
                    let mut results = Vec::with_capacity(10);
                    for image in &test_images {
                        let result = processor.process_image(image).unwrap();
                        results.push(result);
                    }
                    
                    let elapsed = start.elapsed();
                    total_duration += elapsed;
                    black_box(results);
                }
                
                // Return average time per image (total time / 10 images)
                Duration::from_nanos(total_duration.as_nanos() as u64 / 10)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_providers,
    benchmark_image_sizes,
    benchmark_model_variants,
    benchmark_batch_processing
);
criterion_main!(benches);