# Performance Benchmarks

This directory contains comprehensive performance benchmarks for bg_remove-rs across different configurations:

## Benchmark Overview

The benchmarks test various combinations of:
- **Execution Providers**: CPU, CoreML (Apple Silicon), CUDA (NVIDIA GPUs)
- **Backends**: ONNX Runtime, Tract (pure Rust)
- **Image Sizes**: Small (512x512), Medium (1024x1024), Large (2048x2048)
- **Batch Sizes**: 5, 10, 20 images
- **Model Variants**: ISNet FP16/FP32, BiRefNet Lite FP16/FP32

## Prerequisites

### 1. Download Models

**CRITICAL**: Models must be downloaded before running benchmarks. The benchmarks exclude download time to measure pure inference performance.

```bash
# Download ISNet model (required for basic benchmarks)
imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx

# Download BiRefNet Lite model (for model comparison benchmarks)
imgly-bgremove --only-download --model https://huggingface.co/imgly/birefnet-lite-onnx

# Verify models are downloaded
imgly-bgremove --list-models
```

### 2. Enable Features

Ensure you have the required features enabled:

```bash
# For ONNX benchmarks (default)
cargo bench --features "onnx,tract"

# For Tract-only benchmarks
cargo bench --features "tract" --no-default-features
```

## Running Benchmarks

### Full Benchmark Suite

```bash
# Run all benchmarks with HTML reports
cargo bench --bench provider_benchmarks

# Results will be saved to target/criterion/
```

### Specific Benchmark Groups

```bash
# Single image processing only
cargo bench --bench provider_benchmarks -- single_image_processing

# Batch processing only  
cargo bench --bench provider_benchmarks -- batch_processing

# Model comparison only
cargo bench --bench provider_benchmarks -- model_variants
```

### Platform-Specific Execution

```bash
# CPU-only benchmarks (works on all platforms)
cargo bench --bench provider_benchmarks -- "cpu"

# CoreML benchmarks (macOS with Apple Silicon)
cargo bench --bench provider_benchmarks -- "coreml"

# CUDA benchmarks (systems with NVIDIA GPUs)
cargo bench --bench provider_benchmarks -- "cuda"
```

## Benchmark Structure

### 1. Single Image Processing (`benchmark_single_image_processing`)

Tests inference performance on individual images:
- **Small images**: ~512x512 pixels
- **Medium images**: ~1024x1024 pixels  
- **Large images**: ~2048x2048 pixels

**Metrics**: Time per image processing operation

### 2. Batch Processing (`benchmark_batch_processing`)

Tests performance when processing multiple images sequentially:
- **Batch sizes**: 5, 10, 20 images
- **Image type**: Small images for consistency

**Metrics**: Total time for batch processing

### 3. Model Variants (`benchmark_model_variants`)

Compares performance across different models:
- **ISNet**: FP16 vs FP32 variants
- **BiRefNet Lite**: FP16 vs FP32 variants
- **Fixed configuration**: CPU provider for fair comparison

**Metrics**: Time per image across model variants

## Performance Expectations

### Typical Performance Ranges

#### CPU (Intel/AMD)
- **Small images**: 1.5-3.0 seconds
- **Medium images**: 3.0-6.0 seconds
- **Large images**: 6.0-12.0 seconds

#### Apple Silicon (CoreML)
- **Small images**: 100-300ms
- **Medium images**: 200-600ms
- **Large images**: 400-1200ms

#### NVIDIA GPU (CUDA)
- **Small images**: 50-200ms
- **Medium images**: 100-400ms
- **Large images**: 200-800ms

### Model Performance Comparison
- **ISNet FP32**: Highest quality, slower inference
- **ISNet FP16**: Good quality, faster inference
- **BiRefNet Lite FP32**: Portrait-optimized, moderate speed
- **BiRefNet Lite FP16**: Portrait-optimized, faster inference

## Understanding Results

### Benchmark Output

Criterion generates detailed reports including:
- **Mean execution time** with confidence intervals
- **Throughput** (operations per second)
- **Performance regression detection**
- **Statistical significance analysis**

### Result Interpretation

```
benchmark_name/config/size    time: [X.XXXms X.XXXms X.XXXms]
```

- **First value**: Lower bound of confidence interval
- **Second value**: Best estimate (median)
- **Third value**: Upper bound of confidence interval

### HTML Reports

Open `target/criterion/report/index.html` for interactive visualizations:
- Performance trends over time
- Detailed statistics and distributions
- Comparison charts between configurations

## Configuration Matrix

| Provider | Backend | Available On | Acceleration |
|----------|---------|--------------|--------------|
| CPU | ONNX | All platforms | None |
| CPU | Tract | All platforms | None |
| CoreML | ONNX | macOS (Apple Silicon) | GPU |
| CUDA | ONNX | NVIDIA GPUs | GPU |

## Troubleshooting

### Model Not Found Errors

```
Model isnet-general-onnx not found. Please download it first using:
imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx
```

**Solution**: Download the required model before running benchmarks.

### Provider Unavailable

Benchmarks will automatically skip configurations for unavailable providers:
- CUDA on systems without NVIDIA GPUs
- CoreML on non-Apple Silicon systems

### Memory Issues

For large batch benchmarks:
```bash
# Reduce sample size for memory-constrained systems
export CRITERION_SAMPLE_SIZE=5
cargo bench --bench provider_benchmarks
```

## Adding Session Cache Benchmarks

The current implementation focuses on inference performance. To benchmark session caching impact:

1. **Extend BenchmarkConfig** to include cache settings
2. **Implement cache control** in backend factory
3. **Add cache warmup** before timing measurements
4. **Compare cold vs warm** cache performance

This would provide insights into:
- Model loading time impact
- Cache hit/miss ratios
- Memory usage differences
- Startup performance optimization

## Performance Optimization Tips

### For Benchmarking
- Close unnecessary applications
- Use consistent power settings
- Run multiple iterations for statistical significance
- Ensure adequate disk space for model caching

### For Production
- Use FP16 models when quality permits
- Leverage hardware acceleration (CoreML/CUDA)
- Implement session caching for repeated usage
- Consider batch processing for multiple images

## Contributing

When adding new benchmarks:

1. **Follow naming conventions**: `benchmark_feature_name`
2. **Include proper error handling**: Skip unavailable configurations
3. **Document expected performance**: Update this README
4. **Test across platforms**: Verify on different hardware configurations
5. **Maintain statistical rigor**: Use appropriate sample sizes