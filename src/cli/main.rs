//! Background Removal CLI Tool - Refactored
//!
//! Command-line interface for removing backgrounds from images using the unified processor.

use super::backend_factory::CliBackendFactory;
use super::config::CliConfigBuilder;
use crate::{processor::BackgroundRemovalProcessor, utils::ExecutionProviderManager};
use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Background removal CLI tool
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "imgly-bgremove")]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// Input image files or directories (use "-" for stdin)
    #[arg(value_name = "INPUT", required_unless_present_any = &["show_providers", "only_download", "list_models", "clear_cache", "show_cache_dir"])]
    pub input: Vec<String>,

    /// Output file or directory (use "-" for stdout)
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: Option<String>,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = CliOutputFormat::Png)]
    pub format: CliOutputFormat,

    /// Execution provider in format backend:provider (e.g., onnx:auto, onnx:coreml, tract:cpu)
    #[arg(short, long, default_value = "onnx:auto")]
    pub execution_provider: String,

    /// JPEG quality (0-100)
    #[arg(long, default_value_t = 90)]
    pub jpeg_quality: u8,

    /// WebP quality (0-100)
    #[arg(long, default_value_t = 85)]
    pub webp_quality: u8,

    /// Number of threads (0 = auto-detect optimal threading)
    #[arg(short, long, default_value_t = 0)]
    pub threads: usize,

    /// Enable verbose logging (-v: INFO, -vv: DEBUG, -vvv: TRACE)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Process directory recursively
    #[arg(short, long)]
    pub recursive: bool,

    /// Pattern for batch processing (e.g., "*.jpg")
    #[arg(long)]
    pub pattern: Option<String>,

    /// Show execution provider diagnostics and exit
    #[arg(long)]
    pub show_providers: bool,

    /// Model name, URL, or path to model folder [default: first downloaded model or embedded model]
    #[arg(short, long)]
    pub model: Option<String>,

    /// Model variant (fp16, fp32) [default: fp16]
    #[arg(long)]
    pub variant: Option<String>,

    /// Preserve ICC color profiles from input images [default: true]
    #[arg(long, default_value_t = true)]
    pub preserve_color_profiles: bool,

    /// Download model from URL but don't process any images [default: <https://huggingface.co/imgly/isnet-general-onnx>]
    #[arg(long)]
    pub only_download: bool,

    /// List cached models available for processing and exit
    #[arg(long)]
    pub list_models: bool,

    /// Clear cached models (combine with --model to clear specific model)
    #[arg(long)]
    pub clear_cache: bool,

    /// Show current cache directory
    #[arg(long)]
    pub show_cache_dir: bool,

    /// Use custom cache directory
    #[arg(long, value_name = "PATH")]
    pub cache_dir: Option<String>,

    /// Disable all caches during processing (forces fresh session loading)
    #[arg(long)]
    pub no_cache: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum CliOutputFormat {
    Png,
    Jpeg,
    Webp,
    Tiff,
    Rgba8,
}

pub async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose);

    // Handle special flags that don't require inputs
    if cli.show_providers {
        show_provider_diagnostics();
        return Ok(());
    }

    if cli.list_models {
        return list_cached_models().await;
    }

    if cli.only_download {
        return download_model_only(&cli).await;
    }

    if cli.clear_cache {
        return clear_cache_models(&cli).await;
    }

    if cli.show_cache_dir {
        return show_current_cache_dir();
    }

    if cli.input.is_empty() {
        anyhow::bail!("At least one input is required");
    }

    // Validate CLI arguments
    CliConfigBuilder::validate_cli(&cli).context("Invalid CLI arguments")?;

    // Convert CLI arguments to unified configuration
    let config = CliConfigBuilder::from_cli(&cli).context("Failed to build configuration")?;

    info!("Starting background removal CLI");
    if cli.verbose >= 2 {
        info!("🐛 DEBUG MODE: Using configured backend for testing");
    }
    info!("Input(s): {}", cli.input.join(", "));
    info!(
        "Backend: {:?}, Provider: {:?}",
        config.backend_type, config.execution_provider
    );
    info!("Model: {:?}", config.model_spec);

    // Ensure model is available (auto-download if needed)
    ensure_model_available(&config.model_spec)
        .await
        .context("Failed to ensure model is available")?;

    // Create unified processor with CLI backend factory
    let backend_factory = Box::new(CliBackendFactory::new());
    let mut processor = BackgroundRemovalProcessor::with_factory(config, backend_factory)
        .context("Failed to create background removal processor")?;

    // Process inputs
    let start_time = Instant::now();
    let processed_count = process_inputs(&cli, &mut processor).await?;

    let total_time = start_time.elapsed();
    info!(
        "Processed {} image(s) in {:.2}s",
        processed_count,
        total_time.as_secs_f64()
    );

    Ok(())
}

/// Ensure model is available in cache, auto-download if needed
async fn ensure_model_available(model_spec: &crate::models::ModelSpec) -> Result<()> {
    use crate::cache::ModelCache;
    use crate::download::ModelDownloader;
    use crate::models::ModelSource;

    if let ModelSource::Downloaded(model_id) = &model_spec.source {
        let cache = ModelCache::new().context("Failed to create model cache")?;

        // Check if model is already cached
        if !cache.is_model_cached(model_id) {
            // Auto-download for default model only
            if *model_id == ModelCache::get_default_model_id() {
                println!("📦 Model not cached. Auto-downloading default model...");

                let default_url = ModelCache::get_default_model_url();
                let downloader =
                    ModelDownloader::new().context("Failed to create model downloader")?;

                let downloaded_id = downloader
                    .download_model(default_url, false)
                    .await
                    .context("Failed to download default model")?;

                if downloaded_id != *model_id {
                    anyhow::bail!(
                        "Downloaded model ID '{}' doesn't match expected '{}'",
                        downloaded_id,
                        model_id
                    );
                }

                println!("✅ Model downloaded successfully!");
            } else {
                anyhow::bail!(
                    "Model '{}' not found in cache. Use --only-download to download it first, or use --list-models to see available models.",
                    model_id
                );
            }
        }
    }

    Ok(())
}

/// Initialize logging based on verbosity level
fn init_logging(verbose_count: u8) {
    let log_level = match verbose_count {
        0 => "warn",  // Default: only warnings and errors
        1 => "info",  // -v: user-actionable information
        2 => "debug", // -vv: internal state and computations
        _ => "trace", // -vvv+: extremely detailed traces
    };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .init();

    if verbose_count > 0 {
        match verbose_count {
            1 => log::info!("📋 Verbose mode: Showing user-actionable information"),
            2 => log::debug!("🔧 Debug mode: Showing internal state and computations"),
            _ => log::trace!("🔍 Trace mode: Showing extremely detailed traces"),
        }
        log::debug!("Log level: {log_level}");
    }
}

/// Display execution provider diagnostics using core utilities
fn show_provider_diagnostics() {
    println!("🔍 Backend and Execution Provider Diagnostics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // System information
    let cpu_count = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    println!("💻 System: {cpu_count} CPU cores detected");

    println!("\n🔧 Available Backends:");
    println!("  • onnx: ONNX Runtime backend (default) - Full hardware acceleration support");
    println!("  • tract: Pure Rust backend - No external dependencies, WebAssembly compatible");
    println!("  • mock: Mock backend for testing and debugging");

    // Get provider information from core utilities
    let all_providers = ExecutionProviderManager::list_all_providers();

    println!("\n🚀 Execution Providers:");
    for provider_info in all_providers {
        let status = if provider_info.available {
            "✅ Available"
        } else {
            "❌ Not Available"
        };
        println!(
            "  • {}: {} - {}",
            provider_info.name, status, provider_info.description
        );
    }

    // Try to get actual availability from backends
    println!("\n🔍 Checking Backend-Specific Availability:");

    // Check ONNX providers
    #[cfg(feature = "onnx")]
    {
        use crate::backends::OnnxBackend;
        let providers = OnnxBackend::list_providers();
        println!("ONNX Runtime providers:");
        for (name, available, description) in providers {
            let status = if available {
                "✅ Available"
            } else {
                "❌ Not Available"
            };
            println!(
                "  • onnx:{}: {} - {}",
                name.to_lowercase(),
                status,
                description
            );
        }
    }

    // Check Tract providers
    #[cfg(feature = "tract")]
    {
        use crate::backends::TractBackend;
        let tract_providers = TractBackend::list_providers();
        println!("Tract providers:");
        for (name, available, description) in tract_providers {
            let status = if available {
                "✅ Available"
            } else {
                "❌ Not Available"
            };
            println!(
                "  • tract:{}: {} - {}",
                name.to_lowercase(),
                status,
                description
            );
        }
    }

    println!("\n💡 Usage Examples:");
    println!("  --execution-provider onnx:auto    # Auto-select best ONNX provider (default)");
    println!("  --execution-provider onnx:coreml  # Use Apple CoreML (macOS)");
    println!("  --execution-provider onnx:cuda    # Use NVIDIA CUDA");
    println!("  --execution-provider onnx:cpu     # Force ONNX CPU execution");
    println!("  --execution-provider onnx         # Same as onnx:auto");
    println!("  --execution-provider tract:cpu    # Use pure Rust Tract backend");
    println!("  --execution-provider tract        # Same as tract:cpu");
    println!("  --execution-provider mock:cpu     # Use mock backend for testing");

    println!("\n📋 Notes:");
    println!("  • Default backend is 'onnx' if none specified");
    println!("  • ONNX backend provides GPU acceleration with compatible hardware/drivers");
    println!("  • Tract backend is pure Rust with no external dependencies");
    println!("  • Mock backend is for testing and debugging purposes");
    println!("  • CPU provider is always available as fallback for all backends");
}

/// List cached models available for processing
async fn list_cached_models() -> Result<()> {
    use crate::cache::ModelCache;

    let cache = ModelCache::new().context("Failed to initialize model cache")?;
    let models = cache
        .scan_cached_models()
        .context("Failed to list cached models")?;

    println!("📦 Cached Models");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if models.is_empty() {
        println!("No cached models found.");
        println!("\n💡 To download a model, use:");
        println!("  imgly-bgremove --only-download --model https://huggingface.co/imgly/isnet-general-onnx");
        return Ok(());
    }

    for model in models {
        println!("📁 Model ID: {}", model.model_id);
        println!("  └─ Cache location: {}", model.path.display());

        // List available variants
        if !model.variants.is_empty() {
            println!("  └─ Variants: {}", model.variants.join(", "));
        }

        // Show model size
        if model.size_bytes > 0 {
            let size_mb = model.size_bytes as f64 / 1_048_576.0; // Convert to MB
            println!("  └─ Size: {:.2} MB", size_mb);
        }

        // Show configuration status
        let config_status = match (model.has_config, model.has_preprocessor) {
            (true, true) => "✅ Complete",
            (true, false) => "⚠️  Missing preprocessor config",
            (false, true) => "⚠️  Missing model config",
            (false, false) => "❌ Missing configs",
        };
        println!("  └─ Configuration: {}", config_status);

        println!();
    }

    println!("💡 To use a cached model:");
    println!("  imgly-bgremove --model MODEL_ID input.jpg");

    Ok(())
}

/// Download model only without processing images
async fn download_model_only(cli: &Cli) -> Result<()> {
    use crate::download::{validate_model_url, ModelDownloader};

    let model_url = match &cli.model {
        Some(model) => {
            // Check if it's a URL
            if model.starts_with("http") {
                model.clone()
            } else {
                anyhow::bail!("--only-download requires a URL. Use --model with a URL like https://huggingface.co/imgly/isnet-general-onnx");
            }
        },
        None => {
            // Use default model URL
            "https://huggingface.co/imgly/isnet-general-onnx".to_string()
        },
    };

    // Validate URL format
    validate_model_url(&model_url).context("Invalid model URL")?;

    println!("📦 Downloading model from: {}", model_url);

    let downloader = ModelDownloader::new().context("Failed to create model downloader")?;

    match downloader.download_model(&model_url, true).await {
        Ok(model_id) => {
            println!("✅ Successfully downloaded model!");
            println!("   Model ID: {}", model_id);
            println!(
                "   Cache location: {}",
                downloader.cache().get_model_path(&model_id).display()
            );
            println!("\n💡 To use this model:");
            println!("   imgly-bgremove --model {} input.jpg", model_id);
        },
        Err(e) => {
            anyhow::bail!("Failed to download model: {}", e);
        },
    }

    Ok(())
}

/// Clear cached models
async fn clear_cache_models(cli: &Cli) -> Result<()> {
    use crate::cache::ModelCache;

    // Get cache instance (with custom directory if specified)
    let cache = if let Some(cache_dir_str) = &cli.cache_dir {
        let cache_dir_path = PathBuf::from(cache_dir_str);
        ModelCache::with_custom_cache_dir(cache_dir_path.as_path())
            .context("Failed to create cache with custom directory")?
    } else {
        ModelCache::new().context("Failed to create model cache")?
    };

    if let Some(model_id) = &cli.model {
        // Clear specific model
        println!("🗑️  Clearing specific model: {}", model_id);

        match cache.clear_specific_model(model_id) {
            Ok(true) => {
                println!("✅ Successfully removed model: {}", model_id);
                println!(
                    "   Cache location: {}",
                    cache.get_current_cache_dir().display()
                );
            },
            Ok(false) => {
                println!("⚠️  Model '{}' not found in cache", model_id);
                println!("   Use --list-models to see available models");
            },
            Err(e) => {
                anyhow::bail!("Failed to clear model '{}': {}", model_id, e);
            },
        }
    } else {
        // Clear entire cache
        println!("🗑️  Clearing entire model cache...");

        match cache.clear_all_models() {
            Ok(removed_models) => {
                if removed_models.is_empty() {
                    println!("💡 Cache was already empty");
                } else {
                    println!("✅ Successfully removed {} model(s):", removed_models.len());
                    for model_id in &removed_models {
                        println!("   • {}", model_id);
                    }
                }
                println!(
                    "   Cache location: {}",
                    cache.get_current_cache_dir().display()
                );
            },
            Err(e) => {
                anyhow::bail!("Failed to clear cache: {}", e);
            },
        }
    }

    Ok(())
}

/// Show the current cache directory
fn show_current_cache_dir() -> Result<()> {
    use crate::cache::ModelCache;

    match ModelCache::new() {
        Ok(cache) => {
            println!("📁 Current cache directory:");
            println!("   Path: {}", cache.get_current_cache_dir().display());

            // Show platform-specific info with actual base directory
            let cache_path = cache.get_current_cache_dir();
            let base_cache_dir = cache_path.parent().and_then(|p| p.parent());

            if let Some(base_dir) = base_cache_dir {
                #[cfg(target_os = "macos")]
                println!(
                    "   Platform: macOS (using {}/imgly-bgremove/models/)",
                    base_dir.display()
                );

                #[cfg(target_os = "linux")]
                println!(
                    "   Platform: Linux (using {}/imgly-bgremove/models/)",
                    base_dir.display()
                );

                #[cfg(target_os = "windows")]
                println!(
                    "   Platform: Windows (using {}\\imgly-bgremove\\models\\)",
                    base_dir.display()
                );
            } else {
                // Fallback if we can't determine base directory
                #[cfg(target_os = "macos")]
                println!("   Platform: macOS");

                #[cfg(target_os = "linux")]
                println!("   Platform: Linux");

                #[cfg(target_os = "windows")]
                println!("   Platform: Windows");
            }

            // Show environment variable info
            if std::env::var("IMGLY_BGREMOVE_CACHE_DIR").is_ok() {
                println!("   Source: IMGLY_BGREMOVE_CACHE_DIR environment variable");
            } else {
                println!("   Source: XDG cache directory specification");
            }

            println!("\n💡 To use a custom cache directory:");
            println!("   imgly-bgremove --cache-dir /path/to/custom/cache");
            println!("   or set IMGLY_BGREMOVE_CACHE_DIR environment variable");
        },
        Err(e) => {
            anyhow::bail!("Failed to access cache directory: {}", e);
        },
    }

    Ok(())
}

/// Process multiple inputs efficiently using the unified processor
async fn process_inputs(cli: &Cli, processor: &mut BackgroundRemovalProcessor) -> Result<usize> {
    // Handle stdin specially (single input)
    if cli.input.len() == 1 && cli.input.first().is_some_and(|s| s == "-") {
        return process_stdin(cli.output.as_ref(), processor).await;
    }

    // Collect all image files from inputs (files and directories)
    let mut all_files = Vec::new();

    for input in &cli.input {
        let path = PathBuf::from(input);

        if path.is_file() {
            // Single file - validate it's an image
            if is_image_file(&path, &["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"]) {
                all_files.push(path);
            } else {
                warn!("Skipping non-image file: {}", path.display());
            }
        } else if path.is_dir() {
            // Directory - find all image files
            let dir_files = find_image_files(&path, cli.recursive, cli.pattern.as_deref())?;
            all_files.extend(dir_files);
        } else {
            anyhow::bail!(
                "Input path does not exist or is not accessible: {}",
                path.display()
            );
        }
    }

    if all_files.is_empty() {
        warn!("No image files found in the provided inputs");
        return Ok(0);
    }

    info!("Found {} image file(s) to process", all_files.len());

    // For multiple files, show progress bar
    let show_progress = all_files.len() > 1;
    let progress = if show_progress {
        let pb = ProgressBar::new(all_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    let mut processed_count = 0;
    let mut failed_count = 0;
    let file_count = all_files.len();
    let batch_start_time = Instant::now();

    for input_file in all_files {
        if let Some(ref pb) = progress {
            pb.set_message(format!("Processing {}", input_file.display()));
        }

        let output_path = if file_count == 1 {
            // Single file - use specified output or generate default
            cli.output.clone()
        } else {
            // Multiple files - always generate default names (ignore --output)
            None
        };

        match process_single_file(processor, &input_file, output_path.as_ref()).await {
            Ok(_) => {
                processed_count += 1;
                if cli.verbose > 1 {
                    log::debug!("✅ Processed: {}", input_file.display());
                }
            },
            Err(e) => {
                error!("❌ Failed to process {}: {}", input_file.display(), e);
                failed_count += 1;
            },
        }

        if let Some(ref pb) = progress {
            pb.inc(1);
        }
    }

    if let Some(pb) = progress {
        pb.finish_with_message(format!(
            "Completed! Processed: {processed_count}, Failed: {failed_count}"
        ));
    }

    if failed_count > 0 {
        warn!("Some files failed to process. Processed: {processed_count}, Failed: {failed_count}");
    }

    // For batch processing, show a summary
    let batch_total_time = batch_start_time.elapsed();
    if file_count > 1 {
        info!("📊 Batch processing summary:");
        info!("  ├─ Files processed: {}", processed_count);
        info!("  ├─ Files failed: {}", failed_count);
        info!("  ├─ Total time: {:.2}s", batch_total_time.as_secs_f64());
        info!(
            "  └─ Average per file: {:.2}s",
            if processed_count > 0 {
                batch_total_time.as_secs_f64() / (processed_count as f64)
            } else {
                0.0
            }
        );
    }

    Ok(processed_count)
}

/// Process image from stdin using the unified processor
async fn process_stdin(
    output_target: Option<&String>,
    processor: &mut BackgroundRemovalProcessor,
) -> Result<usize> {
    info!("Reading image from stdin");

    let image_data = read_stdin()?;
    let start_time = Instant::now();

    // Detect format from binary data and create temp file with proper extension
    let temp_dir = std::env::temp_dir();
    let detected_format = detect_image_format(&image_data);

    let temp_file = if let Some(ext) = detected_format {
        info!("Detected image format: {}", ext.to_uppercase());
        temp_dir.join(format!("stdin_input.{}", ext))
    } else {
        warn!("Could not detect image format from stdin data, using generic extension. This may cause format detection issues.");
        temp_dir.join("stdin_input.tmp")
    };

    std::fs::write(&temp_file, &image_data).with_context(|| {
        format!(
            "Failed to write stdin data to temporary file: {}",
            temp_file.display()
        )
    })?;

    let result = processor
        .process_file(&temp_file)
        .await
        .with_context(|| {
            if detected_format.is_none() {
                "Failed to remove background. The image format could not be detected from stdin data. Supported formats: PNG, JPEG, WebP, TIFF, BMP, GIF"
            } else {
                "Failed to remove background"
            }
        })?;

    // Clean up temp file
    if let Err(e) = std::fs::remove_file(&temp_file) {
        warn!("Failed to clean up temporary file: {}", e);
    }

    let processing_time = start_time.elapsed();
    info!(
        "Processed stdin image in {:.2}s",
        processing_time.as_secs_f64()
    );

    // Handle output
    match output_target {
        Some(target) if target == "-" => {
            // Output to stdout
            let config = processor.config();
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            info!("Image written to stdout");
        },
        Some(target) => {
            // Output to file
            let output_path = PathBuf::from(target);
            let config = processor.config();
            if config.preserve_color_profiles {
                result
                    .save_with_color_profile(
                        &output_path,
                        config.output_format,
                        config.jpeg_quality,
                    )
                    .context("Failed to save result with color profile")?;
            } else {
                result
                    .save(&output_path, config.output_format, config.jpeg_quality)
                    .context("Failed to save result")?;
            }
            info!("Image saved to: {}", output_path.display());
        },
        None => {
            // Default: output to stdout for stdin input
            let config = processor.config();
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            info!("Image written to stdout");
        },
    }

    Ok(1)
}

/// Process a single image file using the unified processor
async fn process_single_file(
    processor: &mut BackgroundRemovalProcessor,
    input_path: &Path,
    output_path: Option<&String>,
) -> Result<usize> {
    let mut result = processor
        .process_file(input_path)
        .await
        .context("Failed to remove background")?;

    // Show detailed timing breakdown
    let timings = result.timings();
    let breakdown = timings.breakdown_percentages();

    info!("📊 Processing breakdown for {}:", input_path.display());

    if timings.model_load_ms > 0 {
        info!(
            "  ├─ Model Load: {}ms ({:.1}%)",
            timings.model_load_ms, breakdown.model_load_pct
        );
    }

    info!(
        "  ├─ Image Decode: {}ms ({:.1}%)",
        timings.image_decode_ms, breakdown.decode_pct
    );
    info!(
        "  ├─ Preprocessing: {}ms ({:.1}%)",
        timings.preprocessing_ms, breakdown.preprocessing_pct
    );
    info!(
        "  ├─ Inference: {}ms ({:.1}%)",
        timings.inference_ms, breakdown.inference_pct
    );
    info!(
        "  ├─ Postprocessing: {}ms ({:.1}%)",
        timings.postprocessing_ms, breakdown.postprocessing_pct
    );

    // Handle output
    let config = processor.config();
    match output_path {
        Some(target) if target == "-" => {
            // Output to stdout
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            info!(
                "  └─ Total: {}ms ({:.2}s) - output to stdout",
                result.timings().total_ms,
                result.timings().total_ms as f64 / 1000.0
            );
        },
        Some(target) => {
            // Output to specific file
            let output_path = PathBuf::from(target);
            if config.preserve_color_profiles {
                result
                    .save_with_color_profile_timed(
                        &output_path,
                        config.output_format,
                        config.jpeg_quality,
                    )
                    .context("Failed to save result with color profile")?;
            } else {
                result
                    .save_timed(&output_path, config.output_format, config.jpeg_quality)
                    .context("Failed to save result")?;
            }

            if let Some(encode_ms) = result.timings().image_encode_ms {
                let encode_breakdown = result.timings().breakdown_percentages();
                info!(
                    "  ├─ Image Encode: {}ms ({:.1}%)",
                    encode_ms, encode_breakdown.encode_pct
                );
            }

            info!(
                "  └─ Total: {}ms ({:.2}s)",
                result.timings().total_ms,
                result.timings().total_ms as f64 / 1000.0
            );
        },
        None => {
            // Generate default output filename
            let output_path = generate_output_path(input_path, config.output_format);
            if config.preserve_color_profiles {
                result
                    .save_with_color_profile_timed(
                        &output_path,
                        config.output_format,
                        config.jpeg_quality,
                    )
                    .context("Failed to save result with color profile")?;
            } else {
                result
                    .save_timed(&output_path, config.output_format, config.jpeg_quality)
                    .context("Failed to save result")?;
            }

            if let Some(encode_ms) = result.timings().image_encode_ms {
                let encode_breakdown = result.timings().breakdown_percentages();
                info!(
                    "  ├─ Image Encode: {}ms ({:.1}%)",
                    encode_ms, encode_breakdown.encode_pct
                );
            }

            info!(
                "  └─ Total: {}ms ({:.2}s)",
                result.timings().total_ms,
                result.timings().total_ms as f64 / 1000.0
            );
        },
    }

    Ok(1)
}

/// Read image data from stdin
fn read_stdin() -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    io::stdin()
        .read_to_end(&mut buffer)
        .context("Failed to read image data from stdin")?;

    if buffer.is_empty() {
        anyhow::bail!("No data received from stdin");
    }

    Ok(buffer)
}

/// Detect image format from binary data by examining magic bytes
fn detect_image_format(data: &[u8]) -> Option<&'static str> {
    if data.len() < 4 {
        return None;
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if data.len() >= 8 && data.get(0..8).map_or(false, |slice| slice == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some("png");
    }

    // JPEG: FF D8 FF
    if data.len() >= 3 && data.get(0..3).map_or(false, |slice| slice == [0xFF, 0xD8, 0xFF]) {
        return Some("jpg");
    }

    // WebP: RIFF....WEBP
    if data.len() >= 12
        && data.get(0..4).map_or(false, |slice| slice == [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        && data.get(8..12).map_or(false, |slice| slice == [0x57, 0x45, 0x42, 0x50])
    // "WEBP"
    {
        return Some("webp");
    }

    // TIFF (Little Endian): 49 49 2A 00
    if data.len() >= 4 && data.get(0..4).map_or(false, |slice| slice == [0x49, 0x49, 0x2A, 0x00]) {
        return Some("tiff");
    }

    // TIFF (Big Endian): 4D 4D 00 2A
    if data.len() >= 4 && data.get(0..4).map_or(false, |slice| slice == [0x4D, 0x4D, 0x00, 0x2A]) {
        return Some("tiff");
    }

    // BMP: 42 4D (check at least 2 bytes but ensure we have 4 bytes for consistent logic)
    if data.len() >= 2 && data.get(0..2).map_or(false, |slice| slice == [0x42, 0x4D]) {
        return Some("bmp");
    }

    // GIF: 47 49 46 38 (GIF8)
    if data.len() >= 4 && data.get(0..4).map_or(false, |slice| slice == [0x47, 0x49, 0x46, 0x38]) {
        return Some("gif");
    }

    None
}

/// Write image data to stdout
fn write_stdout(data: &[u8]) -> Result<()> {
    io::stdout()
        .write_all(data)
        .context("Failed to write image data to stdout")?;
    io::stdout().flush().context("Failed to flush stdout")?;
    Ok(())
}

/// Find image files in directory (moved from old main.rs)
fn find_image_files(dir: &Path, recursive: bool, pattern: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let image_extensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"];

    if recursive {
        for entry in walkdir::WalkDir::new(dir) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let path = entry.path();
                if is_image_file(path, &image_extensions) && matches_pattern(path, pattern) {
                    files.push(path.to_path_buf());
                }
            }
        }
    } else {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let path = entry.path();
                if is_image_file(&path, &image_extensions) && matches_pattern(&path, pattern) {
                    files.push(path);
                }
            }
        }
    }

    Ok(files)
}

/// Check if file is an image based on extension
fn is_image_file(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| extensions.contains(&ext.to_lowercase().as_str()))
}

/// Check if file matches the given pattern
fn matches_pattern(path: &Path, pattern: Option<&str>) -> bool {
    match pattern {
        Some(pat) => {
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                glob::Pattern::new(pat)
                    .map(|p| p.matches(filename))
                    .unwrap_or(false)
            } else {
                false
            }
        },
        None => true,
    }
}

/// Generate output path with correct extension
fn generate_output_path(input_path: &Path, format: crate::OutputFormat) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or_default();
    let dir = input_path.parent().unwrap_or(Path::new("."));

    let extension = match format {
        crate::OutputFormat::Png => "png",
        crate::OutputFormat::Jpeg => "jpg",
        crate::OutputFormat::WebP => "webp",
        crate::OutputFormat::Tiff => "tiff",
        crate::OutputFormat::Rgba8 => "rgba8",
    };

    dir.join(format!(
        "{}_bg_removed.{}",
        stem.to_string_lossy(),
        extension
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_image_format_png() {
        let png_magic = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_image_format(&png_magic), Some("png"));
    }

    #[test]
    fn test_detect_image_format_jpeg() {
        let jpeg_magic = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(detect_image_format(&jpeg_magic), Some("jpg"));
    }

    #[test]
    fn test_detect_image_format_webp() {
        let webp_magic = [
            0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
        ];
        assert_eq!(detect_image_format(&webp_magic), Some("webp"));
    }

    #[test]
    fn test_detect_image_format_tiff_le() {
        let tiff_le_magic = [0x49, 0x49, 0x2A, 0x00];
        assert_eq!(detect_image_format(&tiff_le_magic), Some("tiff"));
    }

    #[test]
    fn test_detect_image_format_tiff_be() {
        let tiff_be_magic = [0x4D, 0x4D, 0x00, 0x2A];
        assert_eq!(detect_image_format(&tiff_be_magic), Some("tiff"));
    }

    #[test]
    fn test_detect_image_format_bmp() {
        let bmp_magic = [0x42, 0x4D, 0x00, 0x00];
        assert_eq!(detect_image_format(&bmp_magic), Some("bmp"));
    }

    #[test]
    fn test_detect_image_format_gif() {
        let gif_magic = [0x47, 0x49, 0x46, 0x38];
        assert_eq!(detect_image_format(&gif_magic), Some("gif"));
    }

    #[test]
    fn test_detect_image_format_unknown() {
        let unknown_magic = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_image_format(&unknown_magic), None);
    }

    #[test]
    fn test_detect_image_format_too_short() {
        let short_data = [0x89, 0x50];
        assert_eq!(detect_image_format(&short_data), None);
    }

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file(Path::new("test.jpg"), &["jpg", "png"]));
        assert!(is_image_file(Path::new("test.PNG"), &["jpg", "png"]));
        assert!(!is_image_file(Path::new("test.txt"), &["jpg", "png"]));
        assert!(!is_image_file(Path::new("test"), &["jpg", "png"]));
    }
}
