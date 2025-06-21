use bg_remove_core::config::ExecutionProvider;
use bg_remove_core::{
    BackendFactory, BackendType, BackgroundRemovalProcessor, ModelSource, ModelSpec,
    ProcessorConfigBuilder,
};
use bg_remove_onnx::OnnxBackend;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{DynamicImage, RgbImage};
use std::time::{Duration, Instant};

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
            },
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

fn benchmark_batch_vs_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_single");
    group.measurement_time(Duration::from_secs(60));

    // Create 10 test images
    let test_images: Vec<DynamicImage> = (0..10)
        .map(|i| {
            let size = 512 + (i * 16); // Varying sizes from 512 to 656
            create_test_image(size, size)
        })
        .collect();

    let providers = [
        (ExecutionProvider::Cpu, "cpu"),
        (ExecutionProvider::Auto, "auto"),
        (ExecutionProvider::CoreMl, "coreml"),
    ];

    for (provider, name) in providers {
        // Benchmark: Single processor instance processing 10 images (batch processing)
        group.bench_function(&format!("{}_batch_reuse_processor", name), |b| {
            b.iter_batched(
                || {
                    // Setup: Create and initialize processor once
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
                    let mut processor =
                        BackgroundRemovalProcessor::with_factory(processor_config, backend_factory)
                            .unwrap();
                    processor.initialize().unwrap(); // Model loading/compilation happens here
                    (processor, test_images.clone())
                },
                |(mut processor, images)| {
                    // Benchmark: Process all 10 images with the same processor
                    let mut results = Vec::with_capacity(10);
                    for image in images {
                        let result = processor.process_image(&image).unwrap();
                        results.push(result);
                    }
                    black_box(results)
                },
                criterion::BatchSize::SmallInput,
            )
        });

        // Benchmark: Create new processor for each image (individual processing)
        group.bench_function(&format!("{}_individual_new_processor", name), |b| {
            b.iter_batched(
                || test_images.clone(),
                |images| {
                    // Benchmark: Create new processor for each image (includes model loading overhead)
                    let mut results = Vec::with_capacity(10);
                    for image in images {
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
                        let mut processor = BackgroundRemovalProcessor::with_factory(
                            processor_config,
                            backend_factory,
                        )
                        .unwrap();
                        processor.initialize().unwrap(); // Model loading happens for each image

                        let result = processor.process_image(&image).unwrap();
                        results.push(result);
                    }
                    black_box(results)
                },
                criterion::BatchSize::SmallInput,
            )
        });

        // Benchmark: Calculate per-image time in batch processing
        group.bench_function(&format!("{}_per_image_time", name), |b| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::new(0, 0);

                for _ in 0..iters {
                    // Setup processor once
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
                    let mut processor =
                        BackgroundRemovalProcessor::with_factory(processor_config, backend_factory)
                            .unwrap();
                    processor.initialize().unwrap();

                    // Time only the inference part (excluding model loading)
                    let start = Instant::now();
                    let mut results = Vec::with_capacity(10);
                    for image in &test_images {
                        let result = processor.process_image(image).unwrap();
                        results.push(result);
                    }
                    let batch_time = start.elapsed();

                    total_time += batch_time;
                    black_box(results);
                }

                // Return average time per image in the batch
                Duration::from_nanos(total_time.as_nanos() as u64 / (10 * iters as u64))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_batch_vs_single);
criterion_main!(benches);
