use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bg_remove_core::{ImageProcessor, RemovalConfig};
use bg_remove_core::config::ExecutionProvider;
use image::{DynamicImage, RgbImage};
use std::time::Duration;

fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    
    // Create a simple gradient pattern
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (x as f32 / width as f32 * 255.0) as u8;
        let g = (y as f32 / height as f32 * 255.0) as u8;
        let b = ((x + y) as f32 / (width + height) as f32 * 255.0) as u8;
        *pixel = image::Rgb([r, g, b]);
    }
    
    DynamicImage::ImageRgb8(img)
}

fn bench_execution_providers(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_providers");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    
    // Create a test image
    let test_image = create_test_image(512, 512);
    
    let providers = vec![
        ("cpu", ExecutionProvider::Cpu),
        ("auto", ExecutionProvider::Auto),
    ];
    
    for (name, provider) in providers {
        let config = RemovalConfig::builder()
            .execution_provider(provider)
            .debug(false)
            .build()
            .unwrap();
            
        group.bench_function(name, |b| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mut processor = ImageProcessor::new(black_box(&config)).unwrap();
                    let result = processor.process_image(black_box(test_image.clone()));
                    black_box(result.unwrap())
                })
            });
        });
    }
    
    group.finish();
}

fn bench_image_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_sizes");
    group.measurement_time(Duration::from_secs(12));
    group.sample_size(10);
    
    let config = RemovalConfig::builder()
        .execution_provider(ExecutionProvider::Auto)
        .debug(false)
        .build()
        .unwrap();
    
    let sizes = vec![
        ("256x256", 256, 256),
        ("512x512", 512, 512),
        ("1024x1024", 1024, 1024),
    ];
    
    for (name, width, height) in sizes {
        let test_image = create_test_image(width, height);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mut processor = ImageProcessor::new(black_box(&config)).unwrap();
                    let result = processor.process_image(black_box(test_image.clone()));
                    black_box(result.unwrap())
                })
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_execution_providers, bench_image_sizes);
criterion_main!(benches);