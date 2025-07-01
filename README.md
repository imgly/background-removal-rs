# imgly-bgremove

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

**A high-performance Rust library for AI-powered background removal**

Remove backgrounds from images using state-of-the-art deep learning models with hardware acceleration support. Built for speed, accuracy, and production use.

## Why imgly-bgremove?

- **Fast**: Hardware acceleration with automatic provider detection
- **Flexible**: Works as both CLI tool and Rust library 
- **Multi-model**: Support for multiple AI models and any compatible ONNX model
- **Multi-backend**: ONNX Runtime and pure Rust Tract backends
- **Model and Session Caching**: Fast startup with cached models and sessions
- **Batching**: Efficient batch processing of multiple images
- **Color Profile Support**: Preserves ICC color profiles from input images

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
# Install from Git repository
cargo install --git https://github.com/imgly/background-removal-rs
```

Remove background from an image:

```bash
imgly-bgremove input.jpg --output output.png
```

### Library Usage

Add to your Rust project:

```bash
# Install from Git repository
cargo add --git https://github.com/imgly/background-removal-rs
```

Use in your code:

```rust
use imgly_bgremove::{remove_background_from_reader, RemovalConfig, ModelSpec, ModelSource};
use tokio::fs::File;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure with a downloaded model (assumes model already cached)
    let model_spec = ModelSpec {
        source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
        variant: None,
    };
    let config = RemovalConfig::builder()
        .model_spec(model_spec)
        .build()?;

    // Remove background and save result
    let file = File::open("input.jpg").await?;
    remove_background_from_reader(file, &config).await?.save_png("output.png")?;
    Ok(())
}
```

For advanced usage, see the [documentation](https://docs.rs/imgly-bgremove).

## API Choice: Single vs Multiple Images

### Single Image Processing
Use the high-level convenience functions:
```rust
remove_background_from_reader(file, &config).await?
```

### Multiple Image Processing  
Use `RemovalSession` for efficiency:
```rust
let mut session = RemovalSession::new(config)?;
session.remove_background_from_reader(file).await?  // Model stays loaded
```

**Performance Impact**: `RemovalSession` loads the model once and reuses it, while convenience functions reload the model on each call.

## Documentation

View the full API documentation locally:

```bash
# Open documentation in your browser
cargo doc --open

# Or generate docs without opening
cargo doc --no-deps
```

This will build and display the complete API documentation with examples, type information, and detailed usage guides.

## Models and HuggingFace Integration

### Using Pre-trained Models from HuggingFace

The library automatically downloads and caches models from HuggingFace repositories. All models use the standard HuggingFace format with `config.json` and `preprocessor_config.json`.

#### Default Model: ISNet General

**ISNet** is the default general-purpose background removal model:

```bash
# Download and use ISNet (happens automatically on first use)
imgly-bgremove input.jpg --output output.png

# Or explicitly specify the ISNet model
imgly-bgremove input.jpg --output output.png --model https://huggingface.co/imgly/isnet-general-onnx
```

#### BiRefNet Models

**BiRefNet** offers portrait-optimized models for better human subject handling:

```bash
# Use BiRefNet standard model
imgly-bgremove portrait.jpg --output result.png --model https://huggingface.co/onnx-community/BiRefNet-ONNX

# Use BiRefNet Lite for faster processing
imgly-bgremove portrait.jpg --output result.png --model https://huggingface.co/onnx-community/BiRefNet_lite-ONNX
```

#### Custom HuggingFace Models

Any compatible ONNX background removal model on HuggingFace can be used:

```bash
# Use any HuggingFace model repository
imgly-bgremove input.jpg --output output.png --model https://huggingface.co/username/your-model-name

# Download model without processing (useful for pre-caching)
imgly-bgremove --only-download --model https://huggingface.co/username/your-model-name
```

### Model Variants and Precision

Models typically include multiple precision variants:

- **FP16**: Half-precision, smaller size, faster on compatible hardware
- **FP32**: Full-precision, larger size, better accuracy

```bash
# Explicitly choose model precision
imgly-bgremove input.jpg --output output.png --model https://huggingface.co/imgly/isnet-general-onnx --variant fp16
imgly-bgremove input.jpg --output output.png --model https://huggingface.co/imgly/isnet-general-onnx --variant fp32

# Auto-selection (default) chooses best variant for your hardware:
# - macOS: FP32 for CoreML optimization
# - Linux/Windows: FP16 for better performance
```

### Model Cache Management

Downloaded models are cached locally for fast reuse:

```bash
# List all cached models
imgly-bgremove --list-models

# Show cache directory location
imgly-bgremove --show-cache-dir

# Clear all cached models
imgly-bgremove --clear-cache

# Clear specific model from cache
imgly-bgremove --clear-cache --model imgly--isnet-general-onnx

# Use custom cache directory
imgly-bgremove input.jpg --cache-dir /custom/cache/path
```

### Available Models

- **ISNet General** (`https://huggingface.co/imgly/isnet-general-onnx`): Default general-purpose model
- **BiRefNet** (`https://huggingface.co/onnx-community/BiRefNet-ONNX`): Portrait-optimized for human subjects  
- **BiRefNet Lite** (`https://huggingface.co/onnx-community/BiRefNet_lite-ONNX`): Lightweight variant for faster processing
- **Custom Models**: Any compatible HuggingFace ONNX background removal model

## Execution Providers

- **Auto**: Automatically selects best available provider
- **CUDA**: NVIDIA GPU acceleration
- **CoreML**: Apple Silicon GPU acceleration  
- **CPU**: Universal fallback

## CLI Examples

```bash
# Basic usage
imgly-bgremove photo.jpg --output result.png

# Batch process a directory
imgly-bgremove photos/ --output results/ --recursive

# Use specific execution provider
imgly-bgremove input.jpg --output output.png --execution-provider onnx:coreml

# Pipeline usage
curl -s https://example.com/image.jpg | imgly-bgremove - --output - | upload-tool

# Force specific model
imgly-bgremove input.jpg --output output.png --model birefnet --variant fp16
```

## Library Examples

### Basic Usage

```rust
use imgly_bgremove::{remove_background_from_reader, RemovalConfig, ModelSpec, ModelSource};
use tokio::fs::File;

// Simple background removal with cached model
let model_spec = ModelSpec {
    source: ModelSource::Downloaded("imgly--isnet-general-onnx".to_string()),
    variant: None,
};
let config = RemovalConfig::builder().model_spec(model_spec).build()?;
let file = File::open("input.jpg").await?;
remove_background_from_reader(file, &config).await?.save_png("output.png")?;

// For processing multiple images efficiently (model loaded once):
// use imgly_bgremove::RemovalSession;
// let mut session = RemovalSession::new(RemovalConfig::builder().model_spec(model_spec).build()?)?;
// for image_path in ["image1.jpg", "image2.jpg", "image3.jpg"] {
//     let file = File::open(image_path).await?;
//     let result = session.remove_background_from_reader(file).await?;
//     let output_name = format!("output_{}", image_path.replace(".jpg", ".png"));
//     result.save_png(&output_name)?;
// }
```

### Advanced Configuration

```rust
use imgly_bgremove::{
    ModelDownloader, ModelSpec, ModelSource, remove_background_from_reader,
    RemovalConfig, ExecutionProvider, OutputFormat
};
use tokio::fs::File;

// Download and cache a model from HuggingFace
let downloader = ModelDownloader::new()?;

// Example: ISNet general-purpose model
let model_url = "https://huggingface.co/imgly/isnet-general-onnx";
// Example: BiRefNet portrait-optimized model
// let model_url = "https://huggingface.co/onnx-community/BiRefNet-ONNX";
// Example: BiRefNet Lite for faster processing
// let model_url = "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX";
// Example: Any compatible ONNX background removal model
// let model_url = "https://huggingface.co/your-org/your-model-onnx";

let model_id = downloader.download_model(model_url, true).await?;

// Configure processing with downloaded model
let model_spec = ModelSpec {
    source: ModelSource::Downloaded(model_id),
    variant: None, // Auto-select best variant
};

let config = RemovalConfig::builder()
    .model_spec(model_spec)
    .execution_provider(ExecutionProvider::Auto)
    .output_format(OutputFormat::Png)
    .jpeg_quality(95)
    .preserve_color_profiles(true)
    .build()?;

// Process image with the unified API
let file = File::open("input.jpg").await?;
let result = remove_background_from_reader(file, &config).await?;
result.save_png("output.png")?;

// For batch processing: initialize model once, process multiple images efficiently  
// use imgly_bgremove::RemovalSession;
// let session_config = RemovalConfig::builder()
//     .model_spec(model_spec)
//     .execution_provider(ExecutionProvider::Auto)
//     .output_format(OutputFormat::Png)
//     .jpeg_quality(95)
//     .preserve_color_profiles(true)
//     .build()?;
// let mut session = RemovalSession::new(session_config)?;
// let image_paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"];
// for (i, image_path) in image_paths.iter().enumerate() {
//     let file = File::open(image_path).await?;
//     let result = session.remove_background_from_reader(file).await?;
//     result.save_png(&format!("processed_{}.png", i))?;
//     println!("Processed: {} -> processed_{}.png", image_path, i);
// }
```

## CLI Reference

```
imgly-bgremove [OPTIONS] [INPUT]...

ARGUMENTS:
  [INPUT]...  Input image files or directories

OPTIONS:
  -o, --output <OUTPUT>                    Output file or directory (use "-" for stdout)
  -f, --format <FORMAT>                    Output format [default: png]
  -e, --execution-provider <PROVIDER>      Execution provider [default: onnx:auto]
      --jpeg-quality <QUALITY>             JPEG quality (0-100) [default: 90]
      --webp-quality <QUALITY>             WebP quality (0-100) [default: 85]
  -t, --threads <THREADS>                  Number of threads [default: 0]
  -v, --verbose...                         Enable verbose logging (-v: INFO, -vv: DEBUG, -vvv: TRACE)
  -r, --recursive                          Process directory recursively
      --pattern <PATTERN>                  Pattern for batch processing (e.g., "*.jpg")
      --show-providers                     Show execution provider diagnostics and exit
  -m, --model <MODEL>                      Model name, URL, or path to model folder
      --variant <VARIANT>                  Model variant (fp16, fp32) [default: fp16]
      --preserve-color-profiles            Preserve ICC color profiles [default: true]
      --list-models                        List cached models available for processing
      --clear-cache                        Clear cached models
      --no-cache                           Disable all caches during processing
  -h, --help                               Print help
  -V, --version                            Print version
```

## Troubleshooting

**CUDA Provider Not Available**
```bash
# Check CUDA installation and try CPU fallback
imgly-bgremove input.jpg --output output.png --execution-provider onnx:cpu
```

**CoreML Provider Issues** 
```bash
# Check provider diagnostics
imgly-bgremove --show-providers
```

**Model Loading Errors**
```bash
# Enable debug logging
imgly-bgremove input.jpg --output output.png -vv
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under either of

 * Apache License, Version 2.0, (<http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license (<http://opensource.org/licenses/MIT>)

at your option.

## Links

- [Documentation](https://docs.rs/imgly-bgremove)
- [Issues](https://github.com/imgly/background-removal-rs/issues)
- [Releases](https://github.com/imgly/background-removal-rs/releases)

---

Made with ❤️ by the [IMG.LY](https://img.ly) team