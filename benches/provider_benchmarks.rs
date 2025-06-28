use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use imgly_bgremove::{
    backends::{OnnxBackend, TractBackend},
    config::{ExecutionProvider, OutputFormat},
    error::Result,
    inference::InferenceBackend,
    models::{ModelManager, ModelSource, ModelSpec},
    processor::{BackendFactory, BackendType, BackgroundRemovalProcessor, ProcessorConfigBuilder},
};
use std::path::PathBuf;
use tokio::runtime::Runtime;

const TEST_IMAGE_SMALL: &[u8] =
    include_bytes!("../tests/assets/input/portraits/portrait_single_simple_bg.jpg");
const TEST_IMAGE_MEDIUM: &[u8] =
    include_bytes!("../tests/assets/input/portraits/portrait_outdoor_natural.jpg");
const TEST_IMAGE_LARGE: &[u8] =
    include_bytes!("../tests/assets/input/complex/complex_group_photo.jpg");

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    provider: ExecutionProvider,
    backend: &'static str,
}

async fn ensure_model_downloaded(model_name: &str) -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| {
            imgly_bgremove::BgRemovalError::invalid_config("Failed to determine cache directory")
        })?
        .join("imgly-bgremove")
        .join("models");

    let model_dir = cache_dir.join(model_name);

    if !model_dir.exists() {
        println!(
            "Model {} not found. Please download it first using:",
            model_name
        );
        println!(
            "imgly-bgremove --only-download --model https://huggingface.co/imgly/{}",
            model_name.replace("--", "/")
        );
        return Err(imgly_bgremove::BgRemovalError::model(format!(
            "Model {} not found",
            model_name
        )));
    }

    Ok(model_dir)
}

fn setup_benchmark_configs() -> Vec<BenchmarkConfig> {
    let mut configs = Vec::new();

    // Test common execution providers
    let providers = vec![
        ExecutionProvider::Cpu,
        ExecutionProvider::CoreMl,
        ExecutionProvider::Cuda,
    ];

    // Create configurations for each provider
    for provider in providers {
        // ONNX backend configurations
        configs.push(BenchmarkConfig {
            provider: provider.clone(),
            backend: "onnx",
        });

        // Tract backend configurations (CPU only)
        if matches!(provider, ExecutionProvider::Cpu) {
            configs.push(BenchmarkConfig {
                provider: ExecutionProvider::Cpu,
                backend: "tract",
            });
        }
    }

    configs
}

struct BenchmarkBackendFactory;

impl BackendFactory for BenchmarkBackendFactory {
    fn create_backend(
        &self,
        backend_type: BackendType,
        model_manager: ModelManager,
    ) -> Result<Box<dyn InferenceBackend>> {
        match backend_type {
            BackendType::Onnx => {
                #[cfg(feature = "onnx")]
                {
                    let backend = OnnxBackend::with_model_manager(model_manager);
                    Ok(Box::new(backend))
                }
                #[cfg(not(feature = "onnx"))]
                Err(imgly_bgremove::BgRemovalError::invalid_config(
                    "ONNX feature not enabled",
                ))
            },
            BackendType::Tract => {
                #[cfg(feature = "tract")]
                {
                    let backend = TractBackend::with_model_manager(model_manager);
                    Ok(Box::new(backend))
                }
                #[cfg(not(feature = "tract"))]
                Err(imgly_bgremove::BgRemovalError::invalid_config(
                    "Tract feature not enabled",
                ))
            },
        }
    }

    fn available_backends(&self) -> Vec<BackendType> {
        let mut backends = Vec::new();
        #[cfg(feature = "onnx")]
        backends.push(BackendType::Onnx);
        #[cfg(feature = "tract")]
        backends.push(BackendType::Tract);
        backends
    }
}

fn create_processor(
    config: &BenchmarkConfig,
    model_path: PathBuf,
) -> Result<BackgroundRemovalProcessor> {
    let model_spec = ModelSpec {
        source: ModelSource::External(model_path),
        variant: Some("fp32".to_string()),
    };

    let backend_type = match config.backend {
        "onnx" => BackendType::Onnx,
        "tract" => BackendType::Tract,
        _ => {
            return Err(imgly_bgremove::BgRemovalError::invalid_config(format!(
                "Unknown backend: {}",
                config.backend
            )))
        },
    };

    let processor_config = ProcessorConfigBuilder::new()
        .model_spec(model_spec)
        .backend_type(backend_type)
        .execution_provider(config.provider.clone())
        .output_format(OutputFormat::Png)
        .build()?;

    let backend_factory = Box::new(BenchmarkBackendFactory);

    BackgroundRemovalProcessor::with_factory(processor_config, backend_factory)
}

fn benchmark_single_image_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let model_name = "imgly--isnet-general-onnx";

    // Ensure model is downloaded
    let model_path = rt.block_on(async { ensure_model_downloaded(model_name).await.unwrap() });

    let configs = setup_benchmark_configs();

    let mut group = c.benchmark_group("single_image_processing");
    group.sample_size(10); // Reduce sample size for faster benchmarks

    for config in configs {
        let config_name = format!(
            "{}_{}",
            config.backend,
            format!("{:?}", config.provider).to_lowercase()
        );

        // Skip if backend creation fails (provider not available)
        let mut processor = match create_processor(&config, model_path.clone()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Test with small image
        group.bench_with_input(
            BenchmarkId::new(&config_name, "small_512x512"),
            &TEST_IMAGE_SMALL,
            |b, image_data| {
                b.iter(|| {
                    rt.block_on(async {
                        // Save test image to temp file for processing
                        let temp_dir = tempfile::tempdir().unwrap();
                        let input_path = temp_dir.path().join("test.jpg");
                        std::fs::write(&input_path, image_data).unwrap();

                        processor
                            .process_file(black_box(&input_path))
                            .await
                            .unwrap()
                    })
                });
            },
        );

        // Test with medium image
        group.bench_with_input(
            BenchmarkId::new(&config_name, "medium_1024x1024"),
            &TEST_IMAGE_MEDIUM,
            |b, image_data| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = tempfile::tempdir().unwrap();
                        let input_path = temp_dir.path().join("test.jpg");
                        std::fs::write(&input_path, image_data).unwrap();

                        processor
                            .process_file(black_box(&input_path))
                            .await
                            .unwrap()
                    })
                });
            },
        );

        // Test with large image
        group.bench_with_input(
            BenchmarkId::new(&config_name, "large_2048x2048"),
            &TEST_IMAGE_LARGE,
            |b, image_data| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = tempfile::tempdir().unwrap();
                        let input_path = temp_dir.path().join("test.jpg");
                        std::fs::write(&input_path, image_data).unwrap();

                        processor
                            .process_file(black_box(&input_path))
                            .await
                            .unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let model_name = "imgly--isnet-general-onnx";

    // Ensure model is downloaded
    let model_path = rt.block_on(async { ensure_model_downloaded(model_name).await.unwrap() });

    let configs = setup_benchmark_configs();
    let batch_sizes = vec![5, 10, 20];

    let mut group = c.benchmark_group("batch_processing");
    group.sample_size(10);

    for config in configs {
        let config_name = format!(
            "{}_{}",
            config.backend,
            format!("{:?}", config.provider).to_lowercase()
        );

        // Skip if backend creation fails
        let mut processor = match create_processor(&config, model_path.clone()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        for batch_size in &batch_sizes {
            group.bench_with_input(
                BenchmarkId::new(&config_name, format!("batch_{}", batch_size)),
                batch_size,
                |b, &size| {
                    b.iter(|| {
                        rt.block_on(async {
                            let temp_dir = tempfile::tempdir().unwrap();
                            for i in 0..size {
                                let input_path = temp_dir.path().join(format!("test_{}.jpg", i));
                                std::fs::write(&input_path, TEST_IMAGE_SMALL).unwrap();
                                processor
                                    .process_file(black_box(&input_path))
                                    .await
                                    .unwrap();
                            }
                        })
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_model_variants(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let model_configs = vec![
        ("imgly--isnet-general-onnx", "fp16"),
        ("imgly--isnet-general-onnx", "fp32"),
    ];

    // Ensure all models are downloaded
    let model_paths: Vec<_> = rt.block_on(async {
        let mut paths = Vec::new();
        for (model_name, _) in &model_configs {
            match ensure_model_downloaded(model_name).await {
                Ok(path) => paths.push(Some(path)),
                Err(_) => paths.push(None),
            }
        }
        paths
    });

    let mut group = c.benchmark_group("model_variants");
    group.sample_size(10);

    // Test only with CPU for fair comparison
    let config = BenchmarkConfig {
        provider: ExecutionProvider::Cpu,
        backend: "onnx",
    };

    for ((model_name, variant), model_path) in model_configs.iter().zip(model_paths.iter()) {
        let Some(path) = model_path else {
            continue;
        };

        let model_spec = ModelSpec {
            source: ModelSource::External(path.clone()),
            variant: Some(variant.to_string()),
        };

        let processor_config = ProcessorConfigBuilder::new()
            .model_spec(model_spec)
            .backend_type(BackendType::Onnx)
            .execution_provider(config.provider.clone())
            .output_format(OutputFormat::Png)
            .build()
            .unwrap();

        let backend_factory = Box::new(BenchmarkBackendFactory);

        let mut processor =
            match BackgroundRemovalProcessor::with_factory(processor_config, backend_factory) {
                Ok(p) => p,
                Err(_) => continue,
            };

        group.bench_with_input(
            BenchmarkId::new("model_comparison", format!("{}_{}", model_name, variant)),
            &TEST_IMAGE_MEDIUM,
            |b, image_data| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = tempfile::tempdir().unwrap();
                        let input_path = temp_dir.path().join("test.jpg");
                        std::fs::write(&input_path, image_data).unwrap();
                        processor
                            .process_file(black_box(&input_path))
                            .await
                            .unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_image_processing,
    benchmark_batch_processing,
    benchmark_model_variants
);
criterion_main!(benches);
