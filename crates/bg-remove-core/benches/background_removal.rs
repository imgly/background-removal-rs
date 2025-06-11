use bg_remove_core::config::ExecutionProvider;
use bg_remove_core::{remove_background, RemovalConfig};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

fn bench_background_removal(c: &mut Criterion) {
    // Test image path (relative to workspace root)
    let test_image =
        "../../crates/bg-remove-testing/assets/input/portraits/portrait_action_motion.jpg";

    // Skip benchmark if test image doesn't exist
    if !std::path::Path::new(test_image).exists() {
        eprintln!(
            "⚠️  Skipping benchmarks: test image not found at {}",
            test_image
        );
        return;
    }

    let mut group = c.benchmark_group("background_removal");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10); // Minimum sample size for Criterion

    // Benchmark different execution providers
    let providers = vec![
        ("cpu", ExecutionProvider::Cpu),
        ("auto", ExecutionProvider::Auto),
        ("coreml", ExecutionProvider::CoreMl),
    ];

    for (name, provider) in providers {
        let config = RemovalConfig::builder()
            .execution_provider(provider)
            .debug(false)
            .build()
            .unwrap();

        group.bench_with_input(BenchmarkId::new("provider", name), &config, |b, config| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let result = remove_background(black_box(test_image), black_box(config)).await;
                    black_box(result.unwrap())
                })
            });
        });
    }

    group.finish();
}

fn bench_image_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_sizes");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    // Test different image categories/sizes (relative to workspace root)
    let test_images = vec![
        (
            "portrait",
            "../../crates/bg-remove-testing/assets/input/portraits/portrait_action_motion.jpg",
        ),
        (
            "product",
            "../../crates/bg-remove-testing/assets/input/products/product_clothing_white_bg.jpg",
        ),
        (
            "complex",
            "../../crates/bg-remove-testing/assets/input/complex/complex_group_photo.jpg",
        ),
    ];

    let config = RemovalConfig::builder()
        .execution_provider(ExecutionProvider::Auto)
        .debug(false)
        .build()
        .unwrap();

    for (category, image_path) in test_images {
        if !std::path::Path::new(image_path).exists() {
            eprintln!("⚠️  Skipping {}: image not found", category);
            continue;
        }

        group.bench_with_input(
            BenchmarkId::new("category", category),
            image_path,
            |b, path| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let result = remove_background(black_box(path), black_box(&config)).await;
                        black_box(result.unwrap())
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_background_removal, bench_image_sizes);
criterion_main!(benches);
