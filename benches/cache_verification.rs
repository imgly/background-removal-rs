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

async fn ensure_model_downloaded(model_name: &str) -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| {
            imgly_bgremove::BgRemovalError::invalid_config("Failed to determine cache directory")
        })?
        .join("imgly-bgremove")
        .join("models");

    let model_dir = cache_dir.join(model_name);

    if !model_dir.exists() {
        return Err(imgly_bgremove::BgRemovalError::model(format!(
            "Model {model_name} not found"
        )));
    }

    Ok(model_dir)
}

struct CacheVerificationBackendFactory;

impl BackendFactory for CacheVerificationBackendFactory {
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

fn benchmark_cache_verification(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let model_name = "imgly--isnet-general-onnx";

    // Ensure model is downloaded
    let model_path = rt.block_on(async { ensure_model_downloaded(model_name).await.unwrap() });

    // Clear any existing session cache to ensure clean start
    let cache_dir = dirs::cache_dir()
        .unwrap()
        .join("imgly-bgremove")
        .join("sessions");

    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
        println!("🧹 Cleared session cache for clean benchmark");
    }

    let providers = vec![
        (ExecutionProvider::Cpu, "CPU"),
        (ExecutionProvider::CoreMl, "CoreML"),
    ];

    let mut group = c.benchmark_group("cache_verification");
    group.sample_size(10);

    for (provider, provider_name) in providers {
        // Test with cache ENABLED
        println!("\n🔍 Testing {provider_name} with CACHE ENABLED");

        let model_spec = ModelSpec {
            source: ModelSource::External(model_path.clone()),
            variant: Some("fp32".to_string()),
        };

        let processor_config_cached = ProcessorConfigBuilder::new()
            .model_spec(model_spec.clone())
            .backend_type(BackendType::Onnx)
            .execution_provider(provider)
            .output_format(OutputFormat::Png)
            .disable_cache(false)  // Cache ENABLED
            .build()
            .unwrap();

        let backend_factory = Box::new(CacheVerificationBackendFactory);

        let mut processor_cached = match BackgroundRemovalProcessor::with_factory(
            processor_config_cached,
            backend_factory,
        ) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let config_name_cached = format!("{}_cache_enabled", provider_name.to_lowercase());

        group.bench_function(&config_name_cached, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let temp_dir = tempfile::tempdir().unwrap();
                    let input_path = temp_dir.path().join("test_cached.jpg");
                    std::fs::write(&input_path, TEST_IMAGE).unwrap();

                    processor_cached
                        .process_file(black_box(&input_path))
                        .await
                        .unwrap()
                })
            });
        });

        // Test with cache DISABLED
        println!("🔍 Testing {provider_name} with CACHE DISABLED");

        let processor_config_uncached = ProcessorConfigBuilder::new()
            .model_spec(model_spec)
            .backend_type(BackendType::Onnx)
            .execution_provider(provider)
            .output_format(OutputFormat::Png)
            .disable_cache(true)   // Cache DISABLED
            .build()
            .unwrap();

        let backend_factory = Box::new(CacheVerificationBackendFactory);

        let mut processor_uncached = match BackgroundRemovalProcessor::with_factory(
            processor_config_uncached,
            backend_factory,
        ) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let config_name_uncached = format!("{}_cache_disabled", provider_name.to_lowercase());

        group.bench_function(&config_name_uncached, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let temp_dir = tempfile::tempdir().unwrap();
                    let input_path = temp_dir.path().join("test_uncached.jpg");
                    std::fs::write(&input_path, TEST_IMAGE).unwrap();

                    processor_uncached
                        .process_file(black_box(&input_path))
                        .await
                        .unwrap()
                })
            });
        });

        // Check if cache directory was created for enabled case
        if cache_dir.exists() {
            let entries = std::fs::read_dir(&cache_dir).unwrap().count();
            println!(
                "✅ Session cache directory contains {entries} entries after cache_enabled test"
            );
        } else {
            println!("❌ Session cache directory not created");
        }
    }

    group.finish();
}

criterion_group!(cache_verification_benches, benchmark_cache_verification);
criterion_main!(cache_verification_benches);
