use criterion::{black_box, criterion_group, criterion_main, Criterion};
use imgly_bgremove::{
    backends::OnnxBackend,
    config::{ExecutionProvider, OutputFormat},
    error::Result,
    inference::InferenceBackend,
    models::{ModelManager, ModelSource, ModelSpec},
    processor::{BackendFactory, BackendType, BackgroundRemovalProcessor, ProcessorConfigBuilder},
};
use std::path::PathBuf;
use tokio::runtime::Runtime;

const TEST_IMAGE: &[u8] =
    include_bytes!("../tests/assets/input/portraits/portrait_single_simple_bg.jpg");

#[derive(Debug, Clone)]
struct CacheBenchmarkConfig {
    provider: ExecutionProvider,
    use_cache: bool,
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

struct CacheBenchmarkBackendFactory;

impl BackendFactory for CacheBenchmarkBackendFactory {
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
            BackendType::Tract => Err(imgly_bgremove::BgRemovalError::invalid_config(
                "Tract doesn't support session caching",
            )),
        }
    }

    fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::Onnx]
    }
}

fn create_processor_with_cache_control(
    config: &CacheBenchmarkConfig,
    model_path: PathBuf,
) -> Result<BackgroundRemovalProcessor> {
    let model_spec = ModelSpec {
        source: ModelSource::External(model_path),
        variant: Some("fp32".to_string()),
    };

    let processor_config = ProcessorConfigBuilder::new()
        .model_spec(model_spec)
        .backend_type(BackendType::Onnx)
        .execution_provider(config.provider.clone())
        .output_format(OutputFormat::Png)
        .disable_cache(!config.use_cache)  // Invert because disable_cache = !use_cache
        .build()?;

    let backend_factory = Box::new(CacheBenchmarkBackendFactory);

    BackgroundRemovalProcessor::with_factory(processor_config, backend_factory)
}

fn benchmark_cache_cold_start(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let model_name = "imgly--isnet-general-onnx";

    // Ensure model is downloaded
    let model_path = rt.block_on(async { ensure_model_downloaded(model_name).await.unwrap() });

    // Clear any existing session cache first
    let cache_dir = dirs::cache_dir()
        .unwrap()
        .join("imgly-bgremove")
        .join("sessions");

    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    let providers = vec![ExecutionProvider::Cpu, ExecutionProvider::CoreMl];

    let mut group = c.benchmark_group("cache_cold_start");
    group.sample_size(10); // Minimum required samples

    for provider in providers {
        // Test with cache enabled - cold start
        let config = CacheBenchmarkConfig {
            provider: provider.clone(),
            use_cache: true,
        };

        let config_name = format!("{}_{}", "cached", format!("{:?}", provider).to_lowercase());

        let mut processor = match create_processor_with_cache_control(&config, model_path.clone()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        group.bench_function(&config_name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let temp_dir = tempfile::tempdir().unwrap();
                    let input_path = temp_dir.path().join("test.jpg");
                    std::fs::write(&input_path, TEST_IMAGE).unwrap();

                    processor
                        .process_file(black_box(&input_path))
                        .await
                        .unwrap()
                })
            });
        });
    }

    group.finish();
}

fn benchmark_cache_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let model_name = "imgly--isnet-general-onnx";

    // Ensure model is downloaded
    let model_path = rt.block_on(async { ensure_model_downloaded(model_name).await.unwrap() });

    // Warm up the cache by running one inference first
    rt.block_on(async {
        let warm_config = CacheBenchmarkConfig {
            provider: ExecutionProvider::Cpu,
            use_cache: true,
        };

        if let Ok(mut processor) =
            create_processor_with_cache_control(&warm_config, model_path.clone())
        {
            let temp_dir = tempfile::tempdir().unwrap();
            let input_path = temp_dir.path().join("warmup.jpg");
            std::fs::write(&input_path, TEST_IMAGE).unwrap();
            let _ = processor.process_file(&input_path).await;
        }
    });

    let providers = vec![ExecutionProvider::Cpu, ExecutionProvider::CoreMl];

    let mut group = c.benchmark_group("cache_comparison");
    group.sample_size(10);

    for provider in providers {
        for use_cache in [true, false] {
            let config = CacheBenchmarkConfig {
                provider: provider.clone(),
                use_cache,
            };

            let config_name = format!(
                "{}_{}",
                if use_cache { "cached" } else { "uncached" },
                format!("{:?}", provider).to_lowercase()
            );

            let mut processor =
                match create_processor_with_cache_control(&config, model_path.clone()) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

            group.bench_function(&config_name, |b| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = tempfile::tempdir().unwrap();
                        let input_path = temp_dir.path().join("test.jpg");
                        std::fs::write(&input_path, TEST_IMAGE).unwrap();

                        processor
                            .process_file(black_box(&input_path))
                            .await
                            .unwrap()
                    })
                });
            });
        }
    }

    group.finish();
}

fn benchmark_repeated_inference(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let model_name = "imgly--isnet-general-onnx";

    // Ensure model is downloaded
    let model_path = rt.block_on(async { ensure_model_downloaded(model_name).await.unwrap() });

    let providers = vec![ExecutionProvider::Cpu, ExecutionProvider::CoreMl];

    let mut group = c.benchmark_group("repeated_inference");
    group.sample_size(10);

    for provider in providers {
        for use_cache in [true, false] {
            let config = CacheBenchmarkConfig {
                provider: provider.clone(),
                use_cache,
            };

            let config_name = format!(
                "{}_{}_10x",
                if use_cache { "cached" } else { "uncached" },
                format!("{:?}", provider).to_lowercase()
            );

            group.bench_function(&config_name, |b| {
                b.iter(|| {
                    rt.block_on(async {
                        // Create fresh processor for each iteration to test cache behavior
                        let mut processor =
                            create_processor_with_cache_control(&config, model_path.clone())
                                .unwrap();

                        // Run 10 inferences to see cache effect
                        for i in 0..10 {
                            let temp_dir = tempfile::tempdir().unwrap();
                            let input_path = temp_dir.path().join(format!("test_{}.jpg", i));
                            std::fs::write(&input_path, TEST_IMAGE).unwrap();

                            processor
                                .process_file(black_box(&input_path))
                                .await
                                .unwrap();
                        }
                    })
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    cache_benches,
    benchmark_cache_cold_start,
    benchmark_cache_comparison,
    benchmark_repeated_inference
);
criterion_main!(cache_benches);
