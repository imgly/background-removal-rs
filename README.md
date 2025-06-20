# bg-remove

[![CI](https://github.com/imgly/background-removal-rust/workflows/CI/badge.svg)](https://github.com/imgly/background-removal-rust/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Version](https://img.shields.io/crates/v/bg-remove-cli.svg)](https://crates.io/crates/bg-remove-cli)
[![Documentation](https://docs.rs/bg-remove-core/badge.svg)](https://docs.rs/bg-remove-core)

**High-performance background removal CLI built in Rust**

Remove backgrounds from images using state-of-the-art deep learning models with hardware acceleration support across multiple platforms.

## ✨ Features

- 🚀 **Dual Backend Architecture**: ONNX Runtime for hardware acceleration + Tract for pure Rust deployment
- 🎯 **Intelligent Provider Selection**: Automatic detection with manual override (CPU, CUDA, CoreML, Auto)
- ⚡ **Universal Caching**: Provider-specific optimizations for improved performance
- 🎨 **Multiple Models**: ISNet and BiRefNet variants with FP16/FP32 precision options
- 📁 **Batch Processing**: Process single files, directories, or use stdin/stdout for pipelines
- 🖼️ **Format Support**: PNG, JPEG, WebP, TIFF input/output with color profile preservation
- 🔧 **Production Ready**: Comprehensive error handling, logging, and monitoring

## 🚀 Quick Start

### Installation

#### From Pre-built Binaries
Download the latest release for your platform from [GitHub Releases](https://github.com/imgly/background-removal-rust/releases).

#### Using Cargo
```bash
cargo install bg-remove-cli
```

#### Build from Source
```bash
git clone https://github.com/imgly/background-removal-rust.git
cd background-removal-rust
cargo build --release
```

### Basic Usage

```bash
# Remove background from a single image
bg-remove input.jpg output.png

# Process with specific execution provider
bg-remove input.jpg output.png --execution-provider onnx:coreml

# Batch process a directory
bg-remove photos/ --recursive --format png

# Use in a pipeline
cat image.jpg | bg-remove - - > output.png
```

## 📚 Usage Guide

### Single File Processing

```bash
# Basic usage with auto-detection
bg-remove portrait.jpg result.png

# Specify output format and quality
bg-remove photo.jpg result.jpg --format jpeg --jpeg-quality 95

# Use specific model variant
bg-remove input.jpg output.png --model birefnet --variant fp16
```

### Batch Processing

```bash
# Process all images in a directory
bg-remove photos/ --output-dir results/

# Recursive processing with file patterns
bg-remove . --recursive --pattern "*.jpg" --format webp

# Process multiple specific files
bg-remove img1.jpg img2.png img3.webp
```

### Pipeline Integration

```bash
# Read from stdin, write to stdout
curl -s https://example.com/image.jpg | bg-remove - - | upload-tool

# Process and pipe to another tool
bg-remove input.jpg - --format png | image-optimizer --stdin
```

### Execution Provider Selection

```bash
# Auto-select best available provider (default)
bg-remove input.jpg output.png --execution-provider onnx:auto

# Force specific providers
bg-remove input.jpg output.png --execution-provider onnx:coreml  # Apple CoreML
bg-remove input.jpg output.png --execution-provider onnx:cuda    # NVIDIA CUDA
bg-remove input.jpg output.png --execution-provider onnx:cpu     # CPU only
bg-remove input.jpg output.png --execution-provider tract:cpu    # Pure Rust backend
```

### Advanced Options

```bash
# Enable verbose logging for debugging
bg-remove input.jpg output.png -vv

# Configure threading
bg-remove input.jpg output.png --threads 8

# Disable color profile preservation
bg-remove input.jpg output.png --preserve-color-profiles false

# Show provider diagnostics
bg-remove --show-providers
```

## 🏗️ Architecture

### Backend Systems

**ONNX Runtime Backend** (`onnx:*`)
- Hardware acceleration support (CUDA, CoreML, CPU)
- Optimized model caching with `with_optimized_model_path()`
- Provider-specific optimizations
- Production-grade performance

**Tract Backend** (`tract:*`)
- Pure Rust implementation
- No external dependencies
- WebAssembly compatible
- Consistent cross-platform behavior

### Caching System

The universal caching system provides significant performance improvements:

- **CoreML Provider**: Native Apple caching via `with_model_cache_dir()`
- **CPU/CUDA Providers**: Optimized model caching via `with_optimized_model_path()`
- **Cache Isolation**: Provider-specific cache keys prevent conflicts
- **Automatic Management**: Cache validation and cleanup

### Model Variants

| Model | Precision | Binary Size | Use Case |
|-------|-----------|-------------|----------|
| ISNet FP16 | Half | ~90MB | Balanced performance (default) |
| ISNet FP32 | Full | ~175MB | Maximum accuracy |
| BiRefNet FP16 | Half | ~467MB | Portrait optimization |
| BiRefNet FP32 | Full | ~928MB | Portrait maximum quality |
| BiRefNet Lite FP16 | Half | ~45MB | Lightweight portraits |
| BiRefNet Lite FP32 | Full | ~90MB | Lightweight high quality |

## 🔧 Installation Options

### Feature Flags

Control which models are embedded at build time:

```bash
# Default: ISNet FP32 model
cargo install bg-remove-cli

# Lightweight build with FP16 model
cargo install bg-remove-cli --no-default-features --features embed-isnet-fp16

# Portrait-optimized build
cargo install bg-remove-cli --no-default-features --features embed-birefnet-fp16

# All models (large binary)
cargo install bg-remove-cli --features embed-all
```

### Docker Usage

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/bg-remove /usr/local/bin/
ENTRYPOINT ["bg-remove"]
```

```bash
# Build and run
docker build -t bg-remove .
docker run --rm -v $(pwd):/workspace bg-remove /workspace/input.jpg /workspace/output.png
```

## 🛠️ Development

### Library Integration

Use bg-remove as a Rust library:

```rust
use bg_remove_core::{
    BackgroundRemovalProcessor, 
    RemovalConfig, 
    BackendType, 
    ExecutionProvider
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the processor
    let config = RemovalConfig::builder()
        .backend_type(BackendType::Onnx)
        .execution_provider(ExecutionProvider::Auto)
        .build()?;
    
    // Create processor with embedded models
    let mut processor = BackgroundRemovalProcessor::new(config)?;
    
    // Process an image
    let result = processor.process_file("input.jpg").await?;
    result.save("output.png", bg_remove_core::OutputFormat::Png, 90)?;
    
    Ok(())
}
```

### Testing

The project includes comprehensive testing:

```bash
# Fast development tests
cargo test

# Comprehensive test suite
cargo test --features expensive-tests

# Performance regression tests
cargo test --features regression-tests

# Full test coverage
cargo test --all-features
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`cargo test`)
5. Run linting (`cargo clippy`)
6. Format code (`cargo fmt`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📋 CLI Reference

```
bg-remove [OPTIONS] [INPUT]...

ARGUMENTS:
  [INPUT]...  Input image files or directories (use "-" for stdin)

OPTIONS:
  -o, --output <OUTPUT>                    Output file or directory (use "-" for stdout)
  -f, --format <FORMAT>                    Output format [default: png] [possible values: png, jpeg, webp, tiff, rgba8]
  -p, --execution-provider <PROVIDER>      Execution provider [default: onnx:auto]
      --jpeg-quality <QUALITY>             JPEG quality (0-100) [default: 90]
      --webp-quality <QUALITY>             WebP quality (0-100) [default: 85]
  -t, --threads <THREADS>                  Number of threads (0 = auto-detect) [default: 0]
  -d, --debug                              Enable debug mode
  -v, --verbose                            Enable verbose logging (-v: INFO, -vv: DEBUG, -vvv: TRACE)
  -r, --recursive                          Process directory recursively
      --pattern <PATTERN>                  Pattern for batch processing (e.g., "*.jpg")
      --show-providers                     Show execution provider diagnostics and exit
  -m, --model <MODEL>                      Model name or path to model folder
      --variant <VARIANT>                  Model variant (fp16, fp32)
      --preserve-color-profiles <BOOL>     Preserve ICC color profiles [default: true]
  -h, --help                               Print help
  -V, --version                            Print version
```

## 🚨 Troubleshooting

### Common Issues

**CUDA Provider Not Available**
```bash
# Check CUDA installation
nvidia-smi
# Verify CUDA toolkit version compatibility
# Try CPU fallback: --execution-provider onnx:cpu
```

**CoreML Provider Issues**
```bash
# Check system compatibility (macOS only)
bg-remove --show-providers
# Verify Apple Silicon or Intel Mac with CoreML support
```

**Model Loading Errors**
```bash
# Check embedded models
bg-remove --debug --show-providers
# Verify sufficient disk space and memory
# Try different model variant: --variant fp16
```

**Performance Issues**
```bash
# Enable caching diagnostics
bg-remove input.jpg output.png -vv
# Check cache directory permissions
# Monitor system resources during processing
```

### Debug Mode

Enable comprehensive diagnostics:

```bash
# Debug with verbose logging
bg-remove input.jpg output.png --debug -vv

# Show provider information
bg-remove --show-providers

# Test with different backends
bg-remove input.jpg output.png --execution-provider tract:cpu --debug
```

## 📊 CI/CD Integration

### GitHub Actions

```yaml
name: Background Removal Pipeline
on: [push, pull_request]

jobs:
  process-images:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install bg-remove
        run: cargo install bg-remove-cli
      - name: Process marketing images
        run: |
          bg-remove assets/raw/ --output-dir assets/processed/ \
            --recursive --format webp --execution-provider onnx:cpu
```

### Docker Pipeline

```yaml
version: '3.8'
services:
  bg-remove:
    build: .
    volumes:
      - ./input:/input:ro
      - ./output:/output
    command: bg-remove /input --output-dir /output --recursive
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE-MIT](LICENSE-MIT) file for details.

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 🔗 Links

- [Documentation](https://docs.rs/bg-remove-core)
- [Crates.io](https://crates.io/crates/bg-remove-cli)
- [Issues](https://github.com/imgly/background-removal-rust/issues)
- [Releases](https://github.com/imgly/background-removal-rust/releases)

---

Made with ❤️ by the [IMG.LY](https://img.ly) team