use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use imgly_bgremove::{
    config::{ExecutionProvider, OutputFormat},
    error::Result,
    models::{ModelSource, ModelSpec},
    RemovalConfig, RemovalSession,
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
        println!("Model {model_name} not found. Please download it first using:");
        println!(
            "imgly-bgremove --only-download --model https://huggingface.co/imgly/{}",
            model_name.replace("--", "/")
        );
        return Err(imgly_bgremove::BgRemovalError::model(format!(
            "Model {model_name} not found"
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
            provider,
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

fn create_session(config: &BenchmarkConfig, model_path: PathBuf) -> Result<RemovalSession> {
    let model_spec = ModelSpec {
        source: ModelSource::External(model_path),
        variant: Some("fp32".to_string()),
    };

    let session_config = RemovalConfig::builder()
        .model_spec(model_spec)
        .execution_provider(config.provider)
        .output_format(OutputFormat::Png)
        .build()?;

    RemovalSession::new(session_config)
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

        // Skip if session creation fails (provider not available)
        let mut session = match create_session(&config, model_path.clone()) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Test with small image
        group.bench_with_input(
            BenchmarkId::new(&config_name, "small_512x512"),
            &TEST_IMAGE_SMALL,
            |b, image_data| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(session.remove_background_from_bytes(image_data).unwrap())
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
                        black_box(session.remove_background_from_bytes(image_data).unwrap())
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
                        black_box(session.remove_background_from_bytes(image_data).unwrap())
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

        // Skip if session creation fails (provider not available)
        let mut session = match create_session(&config, model_path.clone()) {
            Ok(s) => s,
            Err(_) => continue,
        };

        for batch_size in &batch_sizes {
            group.bench_with_input(
                BenchmarkId::new(&config_name, format!("batch_{batch_size}")),
                batch_size,
                |b, &size| {
                    b.iter(|| {
                        rt.block_on(async {
                            for _i in 0..size {
                                black_box(
                                    session
                                        .remove_background_from_bytes(TEST_IMAGE_SMALL)
                                        .unwrap(),
                                );
                            }
                        });
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
            variant: Some((*variant).to_string()),
        };

        let session_config = RemovalConfig::builder()
            .model_spec(model_spec)
            .execution_provider(config.provider)
            .output_format(OutputFormat::Png)
            .build()
            .unwrap();

        let mut session = match RemovalSession::new(session_config) {
            Ok(s) => s,
            Err(_) => continue,
        };

        group.bench_with_input(
            BenchmarkId::new("model_comparison", format!("{model_name}_{variant}")),
            &TEST_IMAGE_MEDIUM,
            |b, image_data| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(session.remove_background_from_bytes(image_data).unwrap())
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
