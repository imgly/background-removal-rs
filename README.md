# imgly-bgremove

[![CI](https://github.com/imgly/background-removal-rust/workflows/CI/badge.svg)](https://github.com/imgly/background-removal-rust/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Version](https://img.shields.io/crates/v/imgly-bgremove.svg)](https://crates.io/crates/imgly-bgremove)
[![Documentation](https://docs.rs/imgly-bgremove/badge.svg)](https://docs.rs/imgly-bgremove)

**A high-performance Rust library for AI-powered background removal**

Remove backgrounds from images using state-of-the-art deep learning models with hardware acceleration support. Built for speed, accuracy, and production use.

## Why imgly-bgremove?

- **Fast**: 2-5x faster than JavaScript implementations with hardware acceleration
- **Flexible**: Works as both CLI tool and Rust library 
- **Smart**: Automatic hardware detection with manual override support
- **Production-ready**: Comprehensive error handling and logging

## Supported Platforms

- **macOS**: Apple Silicon (M1/M2/M3) with CoreML acceleration, Intel Macs
- **Linux**: NVIDIA GPUs with CUDA, CPU fallback
- **Windows**: NVIDIA GPUs with CUDA, CPU fallback

## Supported Formats

**Input**: JPEG, PNG, WebP, TIFF, BMP  
**Output**: PNG (with transparency), JPEG, WebP, TIFF, raw RGBA8

## Getting Started

### CLI Usage

Install the command-line tool:

```bash
cargo install imgly-bgremove
```

Remove background from an image:

```bash
imgly-bgremove input.jpg output.png
```

### Library Usage

Add to your Rust project:

```bash
cargo add imgly-bgremove
```

Use in your code:

```rust
use imgly_bgremove::remove_background_simple;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Remove background with default settings
    remove_background_simple("input.jpg", "output.png").await?;
    Ok(())
}
```

For advanced usage, see the [documentation](https://docs.rs/imgly-bgremove).

## Key Features

### Multiple AI Models
- **ISNet**: General-purpose background removal (FP16/FP32)
- **BiRefNet**: Portrait-optimized models (FP16/FP32)  
- **BiRefNet Lite**: Lightweight variant for faster processing
- **Custom Models**: Any compatible ONNX background removal model

### Dual Backend Architecture
- **ONNX Runtime**: Hardware acceleration (CUDA, CoreML, CPU)
- **Tract**: Pure Rust backend for maximum compatibility

### Execution Providers
- **Auto**: Automatically selects best available provider
- **CUDA**: NVIDIA GPU acceleration
- **CoreML**: Apple Silicon GPU acceleration  
- **CPU**: Universal fallback

### Advanced Features
- **Model Caching**: Automatic download and caching from HuggingFace
- **Color Profiles**: ICC color profile preservation
- **Batch Processing**: Process directories and multiple files
- **Pipeline Support**: stdin/stdout for shell integration

## CLI Examples

```bash
# Basic usage
imgly-bgremove photo.jpg result.png

# Batch process a directory
imgly-bgremove photos/ --output-dir results/ --recursive

# Use specific execution provider
imgly-bgremove input.jpg output.png --execution-provider onnx:coreml

# Pipeline usage
curl -s https://example.com/image.jpg | imgly-bgremove - - | upload-tool

# Force specific model
imgly-bgremove input.jpg output.png --model birefnet --variant fp16
```

## Library Examples

### Basic Usage

```rust
use imgly_bgremove::remove_background_simple;

// One-line background removal
remove_background_simple("input.jpg", "output.png").await?;
```

### Advanced Configuration

```rust
use imgly_bgremove::{
    ModelDownloader, ModelSpec, ModelSource,
    BackgroundRemovalProcessor, RemovalConfig,
    ExecutionProvider, OutputFormat
};

// Download and cache a model from HuggingFace
let downloader = ModelDownloader::new();

// Example: ISNet general-purpose model
let model_url = "https://huggingface.co/imgly/isnet-general-onnx";
// Example: BiRefNet portrait-optimized model
// let model_url = "https://huggingface.co/imgly/birefnet-portrait-onnx";
// Example: Any compatible ONNX background removal model
// let model_url = "https://huggingface.co/your-org/your-model-onnx";

let model_id = downloader.download_model(model_url, true).await?;

// Configure processor with downloaded model
let model_spec = ModelSpec {
    source: ModelSource::Downloaded(model_id),
    variant: None, // Auto-select best variant
};

let config = RemovalConfig::builder()
    .model_spec(model_spec)
    .execution_provider(ExecutionProvider::Auto)
    .build()?;

// Process image
let mut processor = BackgroundRemovalProcessor::new(config)?;
let result = processor.process_image_file("input.jpg").await?;
result.save("output.png", OutputFormat::Png, 90)?;
```

## CLI Reference

```
imgly-bgremove [OPTIONS] [INPUT]...

ARGUMENTS:
  [INPUT]...  Input image files or directories

OPTIONS:
  -o, --output <OUTPUT>                    Output file or directory
  -f, --format <FORMAT>                    Output format [default: png]
  -p, --execution-provider <PROVIDER>      Execution provider [default: onnx:auto]
      --jpeg-quality <QUALITY>             JPEG quality (0-100) [default: 90]
      --webp-quality <QUALITY>             WebP quality (0-100) [default: 85]
  -t, --threads <THREADS>                  Number of threads [default: 0]
  -r, --recursive                          Process directory recursively
      --pattern <PATTERN>                  File pattern for batch processing
  -m, --model <MODEL>                      Model name or path
      --variant <VARIANT>                  Model variant (fp16, fp32)
      --preserve-color-profiles <BOOL>     Preserve ICC color profiles [default: true]
  -v, --verbose                            Enable verbose logging
  -h, --help                               Print help
  -V, --version                            Print version
```

## Troubleshooting

**CUDA Provider Not Available**
```bash
# Check CUDA installation and try CPU fallback
imgly-bgremove input.jpg output.png --execution-provider onnx:cpu
```

**CoreML Provider Issues** 
```bash
# Check provider diagnostics
imgly-bgremove --show-providers
```

**Model Loading Errors**
```bash
# Enable debug logging
imgly-bgremove input.jpg output.png --debug -vv
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under either of

 * Apache License, Version 2.0, (http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (http://opensource.org/licenses/MIT)

at your option.

## Links

- [Documentation](https://docs.rs/imgly-bgremove)
- [Crates.io](https://crates.io/crates/imgly-bgremove)
- [Issues](https://github.com/imgly/background-removal-rust/issues)
- [Releases](https://github.com/imgly/background-removal-rust/releases)

---

Made with ❤️ by the [IMG.LY](https://img.ly) team