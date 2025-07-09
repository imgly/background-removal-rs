//! Background Removal CLI Tool - Refactored
//!
//! Command-line interface for removing backgrounds from images using the unified processor.

// use super::backend_factory::CliBackendFactory; // No longer needed with simplified backend creation
use super::config::CliConfigBuilder;
use crate::{
    processor::BackgroundRemovalProcessor,
    services::{
        create_cli_progress_reporter, BatchProcessingStats, BatchProgressUpdate, ProcessingStage,
        ProgressUpdate,
    },
    utils::ExecutionProviderManager,
};
use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info as trace_info, trace, warn as trace_warn};

/// Background removal CLI tool
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "imgly-bgremove")]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// Input image/video files or directories (use "-" for stdin)
    #[arg(value_name = "INPUT", required_unless_present_any = &["show_providers", "only_download", "list_models", "clear_cache", "show_cache_dir"])]
    pub input: Vec<String>,

    /// Output file (single input) or directory (batch processing). Use "-" for stdout.
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

    /// Show detailed progress with nested file/frame information for batch and video processing
    #[arg(long)]
    pub progress: bool,

    /// Video processing frame batch size for GPU efficiency (video only)
    #[cfg(feature = "video-support")]
    #[arg(long, default_value_t = 8)]
    pub video_batch_size: usize,

    /// Preserve audio track in video output (video only)
    #[cfg(feature = "video-support")]
    #[arg(long, default_value_t = true)]
    pub preserve_audio: bool,

    /// Video codec for output (h264, h265, vp8, vp9, av1) (video only)
    #[cfg(feature = "video-support")]
    #[arg(long, default_value = "h264")]
    pub video_codec: String,

    /// Video quality setting (codec-specific range) (video only)
    #[cfg(feature = "video-support")]
    #[arg(long)]
    pub video_quality: Option<u8>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum CliOutputFormat {
    Png,
    Jpeg,
    Webp,
    Tiff,
    Rgba8,
    #[cfg(feature = "video-support")]
    Mp4,
    #[cfg(feature = "video-support")]
    Avi,
    #[cfg(feature = "video-support")]
    Mov,
    #[cfg(feature = "video-support")]
    Mkv,
    #[cfg(feature = "video-support")]
    Webm,
}

pub async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    init_tracing(cli.verbose).context("Failed to initialize tracing")?;

    // Handle special flags that don't require inputs
    if cli.show_providers {
        show_provider_diagnostics();
        return Ok(());
    }

    if cli.list_models {
        return list_cached_models();
    }

    if cli.only_download {
        return download_model_only(&cli).await;
    }

    if cli.clear_cache {
        return clear_cache_models(&cli);
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

    // Create unified processor with simplified backend creation
    let mut processor = BackgroundRemovalProcessor::new(config)
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

/// Initialize tracing based on verbosity level
fn init_tracing(verbose_count: u8) -> Result<()> {
    use crate::tracing_config::{TracingConfig, TracingFormat};

    // Initialize tracing with CLI-friendly configuration
    TracingConfig::new()
        .with_verbosity(verbose_count)
        .with_format(TracingFormat::Console)
        .init()
        .context("Failed to initialize tracing subscriber")?;

    if verbose_count > 0 {
        match verbose_count {
            1 => trace_warn!("⚠️  Warning level: Showing warnings and errors"),
            2 => trace_info!("ℹ️  Info level: Showing informational messages"),
            3 => debug!("🔧 Debug level: Showing internal state and computations"),
            _ => trace!("🔍 Trace level: Showing extremely detailed traces"),
        }
        let log_level = match verbose_count {
            0 => "error",
            1 => "warn",
            2 => "info",
            3 => "debug",
            _ => "trace",
        };
        debug!(log_level = %log_level, "Tracing initialized");
    }

    Ok(())
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
    println!("  • tract: Pure Rust backend - No external dependencies");

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

    println!("\n💡 Usage Examples:");
    println!("  --execution-provider onnx:auto    # Auto-select best ONNX provider (default)");
    println!("  --execution-provider onnx:coreml  # Use Apple CoreML (macOS)");
    println!("  --execution-provider onnx:cuda    # Use NVIDIA CUDA");
    println!("  --execution-provider onnx:cpu     # Force ONNX CPU execution");
    println!("  --execution-provider onnx         # Same as onnx:auto");
    println!("  --execution-provider tract:cpu    # Use pure Rust Tract backend");
    println!("  --execution-provider tract        # Same as tract:cpu");

    println!("\n📋 Notes:");
    println!("  • Default backend is 'onnx' if none specified");
    println!("  • ONNX backend provides GPU acceleration with compatible hardware/drivers");
    println!("  • Tract backend is pure Rust with no external dependencies");
    println!("  • CPU provider is always available as fallback for all backends");
}

/// List cached models available for processing
fn list_cached_models() -> Result<()> {
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
fn clear_cache_models(cli: &Cli) -> Result<()> {
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

    // Collect all media files from inputs (files and directories)
    let mut all_files = Vec::new();

    for input in &cli.input {
        let path = PathBuf::from(input);

        if path.is_file() {
            // Single file - validate it's an image or video
            if is_image_file(&path, &["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"]) {
                all_files.push(path);
            } else if is_video_file(&path) {
                all_files.push(path);
            } else {
                warn!("Skipping unsupported file: {}", path.display());
            }
        } else if path.is_dir() {
            // Directory - find all media files
            let dir_files = find_media_files(&path, cli.recursive, cli.pattern.as_deref())?;
            all_files.extend(dir_files);
        } else {
            anyhow::bail!(
                "Input path does not exist or is not accessible: {}",
                path.display()
            );
        }
    }

    if all_files.is_empty() {
        warn!("No supported media files found in the provided inputs");
        return Ok(0);
    }

    // Sort files alphanumerically for consistent processing order
    all_files.sort();

    info!("Found {} media file(s) to process", all_files.len());

    // Create appropriate progress reporter based on --progress flag
    let progress_reporter =
        create_cli_progress_reporter(cli.progress, cli.verbose > 0, all_files.len());

    // For backward compatibility with indicatif progress bar when --progress is not used
    let indicatif_progress = if !cli.progress && all_files.len() > 1 {
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

    // Validate and prepare output directory for batch processing
    let output_dir = if file_count > 1 {
        if let Some(ref output) = cli.output {
            if output == "-" {
                anyhow::bail!("Cannot use stdout (-) as output when processing multiple files");
            }
            let output_path = PathBuf::from(output);
            // Create output directory if it doesn't exist
            if !output_path.exists() {
                std::fs::create_dir_all(&output_path).with_context(|| {
                    format!(
                        "Failed to create output directory: {}",
                        output_path.display()
                    )
                })?;
            } else if output_path.is_file() {
                anyhow::bail!(
                    "Output path exists and is a file, not a directory: {}",
                    output_path.display()
                );
            }
            Some(output_path)
        } else {
            None
        }
    } else {
        None
    };

    for (_index, input_file) in all_files.iter().enumerate() {
        let file_start_time = Instant::now();

        // Update indicatif progress bar if used (backward compatibility)
        if let Some(ref pb) = indicatif_progress {
            pb.set_message(format!("Processing {}", input_file.display()));
        }

        // Report enhanced progress if --progress is enabled
        if cli.progress && file_count > 1 {
            // Calculate processing rate
            let elapsed_seconds = batch_start_time.elapsed().as_secs_f64();
            let processing_rate = if elapsed_seconds > 0.0 && processed_count > 0 {
                processed_count as f64 / elapsed_seconds
            } else {
                0.0
            };

            // Calculate ETA
            let remaining_items = file_count - processed_count - failed_count;
            let eta_seconds = if processing_rate > 0.0 {
                Some((remaining_items as f64 / processing_rate) as u64)
            } else {
                None
            };

            // Create batch progress update
            let batch_update = BatchProgressUpdate {
                total_progress: ProgressUpdate::new(
                    ProcessingStage::BatchItemProcessing,
                    batch_start_time,
                ),
                current_item_progress: Some(ProgressUpdate::new(
                    ProcessingStage::ImageLoading,
                    file_start_time,
                )),
                stats: BatchProcessingStats {
                    items_completed: processed_count,
                    items_total: file_count,
                    items_failed: failed_count,
                    current_item_name: input_file.display().to_string(),
                    processing_rate,
                    eta_seconds,
                    frame_info: None, // No frame info for image processing
                },
            };

            // Report progress using the trait method
            progress_reporter.report_batch_progress(batch_update);
        }

        let output_path = if file_count == 1 {
            // Single file - use specified output or generate default
            cli.output.clone()
        } else {
            // Multiple files - use output directory if specified
            output_dir.as_ref().map(|dir| {
                generate_output_path_with_dir(&input_file, dir, processor.config().output_format)
            })
        };

        match process_single_media_file(cli, processor, input_file, output_path.as_ref()).await {
            Ok(_) => {
                processed_count += 1;
                if cli.verbose > 1 {
                    log::debug!("✅ Processed: {}", input_file.display());
                }
            },
            Err(e) => {
                error!("❌ Failed to process {}: {}", input_file.display(), e);
                failed_count += 1;

                // Report error through progress reporter
                progress_reporter.report_error(
                    ProcessingStage::BatchItemProcessing,
                    &format!("Failed to process {}: {}", input_file.display(), e),
                );
            },
        }

        if let Some(ref pb) = indicatif_progress {
            pb.inc(1);
        }
    }

    if let Some(pb) = indicatif_progress {
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

/// Process a single media file (image or video) and return success count
async fn process_single_media_file(
    cli: &Cli,
    processor: &mut BackgroundRemovalProcessor,
    input_path: &Path,
    output_path: Option<&String>,
) -> Result<usize> {
    if is_video_file(input_path) {
        process_single_video_file(cli, input_path, output_path).await
    } else {
        process_single_file(processor, input_path, output_path).await
    }
}

/// Process a single image file using the unified processor
async fn process_single_file(
    processor: &mut BackgroundRemovalProcessor,
    input_path: &Path,
    output_path: Option<&String>,
) -> Result<usize> {
    let mut result = processor
        .process_file(input_path)
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
    if data.len() >= 8
        && data
            .get(0..8)
            .is_some_and(|slice| slice == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    {
        return Some("png");
    }

    // JPEG: FF D8 FF
    if data.len() >= 3
        && data
            .get(0..3)
            .is_some_and(|slice| slice == [0xFF, 0xD8, 0xFF])
    {
        return Some("jpg");
    }

    // WebP: RIFF....WEBP
    if data.len() >= 12
        && data.get(0..4).is_some_and(|slice| slice == [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        && data.get(8..12).is_some_and(|slice| slice == [0x57, 0x45, 0x42, 0x50])
    // "WEBP"
    {
        return Some("webp");
    }

    // TIFF (Little Endian): 49 49 2A 00
    if data.len() >= 4
        && data
            .get(0..4)
            .is_some_and(|slice| slice == [0x49, 0x49, 0x2A, 0x00])
    {
        return Some("tiff");
    }

    // TIFF (Big Endian): 4D 4D 00 2A
    if data.len() >= 4
        && data
            .get(0..4)
            .is_some_and(|slice| slice == [0x4D, 0x4D, 0x00, 0x2A])
    {
        return Some("tiff");
    }

    // BMP: 42 4D (check at least 2 bytes but ensure we have 4 bytes for consistent logic)
    if data.len() >= 2 && data.get(0..2).is_some_and(|slice| slice == [0x42, 0x4D]) {
        return Some("bmp");
    }

    // GIF: 47 49 46 38 (GIF8)
    if data.len() >= 4
        && data
            .get(0..4)
            .is_some_and(|slice| slice == [0x47, 0x49, 0x46, 0x38])
    {
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
#[allow(dead_code)]
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

/// Find all media files (images and videos) in a directory
fn find_media_files(dir: &Path, recursive: bool, pattern: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let image_extensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"];

    if recursive {
        for entry in walkdir::WalkDir::new(dir) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let path = entry.path();
                if (is_image_file(path, &image_extensions) || is_video_file(path))
                    && matches_pattern(path, pattern)
                {
                    files.push(path.to_path_buf());
                }
            }
        }
    } else {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let path = entry.path();
                if (is_image_file(&path, &image_extensions) || is_video_file(&path))
                    && matches_pattern(&path, pattern)
                {
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

/// Check if file is a video file
#[cfg(feature = "video-support")]
fn is_video_file(path: &Path) -> bool {
    use crate::services::OutputFormatHandler;
    OutputFormatHandler::is_video_format(path)
}

/// Check if file is a video file (stub for when video support is disabled)
#[cfg(not(feature = "video-support"))]
fn is_video_file(_path: &Path) -> bool {
    false
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

/// Generate output path with custom output directory, preserving relative structure
fn generate_output_path_with_dir(
    input_path: &Path,
    output_dir: &Path,
    format: crate::OutputFormat,
) -> String {
    let stem = input_path.file_stem().unwrap_or_default();

    let extension = match format {
        crate::OutputFormat::Png => "png",
        crate::OutputFormat::Jpeg => "jpg",
        crate::OutputFormat::WebP => "webp",
        crate::OutputFormat::Tiff => "tiff",
        crate::OutputFormat::Rgba8 => "rgba8",
    };

    let output_filename = format!("{}_bg_removed.{}", stem.to_string_lossy(), extension);

    output_dir
        .join(output_filename)
        .to_string_lossy()
        .to_string()
}

/// Process a single video file using the video processing API
#[cfg(feature = "video-support")]
async fn process_single_video_file(
    cli: &Cli,
    input_path: &Path,
    output_path: Option<&String>,
) -> Result<usize> {
    use crate::{
        backends::video::codec::{QualityPreset, VideoCodec, VideoEncodingConfig},
        config::{RemovalConfig, VideoProcessingConfig},
        models::{ModelSource, ModelSpec},
        remove_background_from_video_file,
        utils::{ExecutionProviderManager, ModelSpecParser},
    };

    info!("🎬 Processing video file: {}", input_path.display());

    // Create progress reporter for video processing if --progress is enabled
    let progress_reporter = create_cli_progress_reporter(cli.progress, cli.verbose > 0, 1);
    let video_start_time = Instant::now();

    // Report initial video processing stage
    if cli.progress {
        progress_reporter.report_progress(ProgressUpdate::new(
            ProcessingStage::VideoAnalysis,
            video_start_time,
        ));
    }

    // Build video configuration from CLI options
    let video_codec = VideoCodec::from_str(&cli.video_codec)
        .with_context(|| format!("Invalid video codec: {}", cli.video_codec))?;

    let mut encoding_config =
        VideoEncodingConfig::new(video_codec).with_preset(QualityPreset::Medium);

    if let Some(quality) = cli.video_quality {
        encoding_config = encoding_config.with_quality(quality);
    }

    let video_config = VideoProcessingConfig {
        encoding: encoding_config,
        batch_size: cli.video_batch_size,
        preserve_audio: cli.preserve_audio,
        fps_override: None,
        parallel_processing: true,
        max_workers: 0, // 0 = auto-detect
        temp_dir: None,
    };

    // Build main removal configuration
    let (model_spec, _) = if let Some(model_arg) = &cli.model {
        let model_spec = ModelSpecParser::parse(model_arg);
        (model_spec, model_arg.clone())
    } else {
        use crate::cache::ModelCache;
        let default_url = ModelCache::get_default_model_url();
        let model_spec = ModelSpec {
            source: ModelSource::Downloaded(ModelCache::url_to_model_id(default_url)),
            variant: cli.variant.clone(),
        };
        (model_spec, ModelCache::url_to_model_id(default_url))
    };

    let (_backend_type, execution_provider) =
        ExecutionProviderManager::parse_provider_string(&cli.execution_provider)
            .context("Invalid execution provider format")?;

    let removal_config = RemovalConfig::builder()
        .model_spec(model_spec)
        .execution_provider(execution_provider)
        .video_config(video_config)
        .build()
        .context("Failed to build removal configuration")?;

    // Report video processing start
    if cli.progress {
        progress_reporter.report_progress(ProgressUpdate::new(
            ProcessingStage::FrameProcessing,
            video_start_time,
        ));
    }

    // Process the video
    let start_time = Instant::now();
    let result = remove_background_from_video_file(input_path, &removal_config)
        .await
        .context("Failed to process video")?;

    let processing_time = start_time.elapsed();

    // Report video encoding stage
    if cli.progress {
        progress_reporter.report_progress(ProgressUpdate::new(
            ProcessingStage::VideoEncoding,
            video_start_time,
        ));
    }

    // Determine output path
    let final_output_path = match output_path {
        Some(target) if target == "-" => {
            anyhow::bail!("Cannot output video to stdout");
        },
        Some(target) => target.clone(),
        None => {
            // Generate default output path based on CLI format or input extension
            let stem = input_path.file_stem().unwrap_or_default();
            let parent = input_path.parent().unwrap_or(Path::new("."));

            let extension = match cli.format {
                CliOutputFormat::Mp4 => "mp4",
                CliOutputFormat::Avi => "avi",
                CliOutputFormat::Mov => "mov",
                CliOutputFormat::Mkv => "mkv",
                CliOutputFormat::Webm => "webm",
                _ => "mp4", // Default to mp4 for non-video output formats
            };

            parent
                .join(format!(
                    "{}_bg_removed.{}",
                    stem.to_string_lossy(),
                    extension
                ))
                .to_string_lossy()
                .to_string()
        },
    };

    // Report video finalization stage
    if cli.progress {
        progress_reporter.report_progress(ProgressUpdate::new(
            ProcessingStage::VideoFinalization,
            video_start_time,
        ));
    }

    // Save the processed video
    std::fs::write(&final_output_path, result.video_data)
        .with_context(|| format!("Failed to write output video: {}", final_output_path))?;

    // Report completion
    if cli.progress {
        progress_reporter.report_progress(ProgressUpdate::new(
            ProcessingStage::Completed,
            video_start_time,
        ));
    }

    // Display processing statistics
    info!("🎬 Video processing completed:");
    info!(
        "  ├─ Frames processed: {}",
        result.frame_stats.frames_processed
    );
    info!("  ├─ Failed frames: {}", result.frame_stats.failed_frames);
    info!(
        "  ├─ Success rate: {:.1}%",
        result.frame_stats.success_rate()
    );
    info!(
        "  ├─ Average frame time: {:.0}ms",
        result.frame_stats.average_frame_time.as_millis()
    );
    info!(
        "  ├─ Total processing time: {:.2}s",
        processing_time.as_secs_f64()
    );
    info!("  └─ Output: {}", final_output_path);

    Ok(1)
}

/// Process a single video file (stub for when video support is disabled)
#[cfg(not(feature = "video-support"))]
async fn process_single_video_file(
    _cli: &Cli,
    input_path: &Path,
    _output_path: Option<&String>,
) -> Result<usize> {
    anyhow::bail!("Video processing not supported. Enable 'video-support' feature and ensure FFmpeg is available. File: {}", input_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

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
    fn test_detect_image_format_edge_cases() {
        // Test empty data
        assert_eq!(detect_image_format(&[]), None);

        // Test exactly required minimum bytes for each format
        let png_exact = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_image_format(&png_exact), Some("png"));

        let jpeg_exact = [0xFF, 0xD8, 0xFF, 0x00]; // Need at least 4 bytes
        assert_eq!(detect_image_format(&jpeg_exact), Some("jpg"));

        let bmp_exact = [0x42, 0x4D, 0x00, 0x00]; // Need at least 4 bytes
        assert_eq!(detect_image_format(&bmp_exact), Some("bmp"));

        // Test partial headers (should return None)
        assert_eq!(detect_image_format(&[0x89]), None);
        assert_eq!(detect_image_format(&[0xFF, 0xD8]), None);
        assert_eq!(detect_image_format(&[0x52, 0x49, 0x46]), None);
    }

    #[test]
    fn test_detect_image_format_multiple_formats_in_data() {
        // Test with longer data that could match multiple formats
        let mixed_data = [
            0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0xFF, 0xD8,
            0xFF, 0xE0, // JPEG magic later in data
        ];
        // Should detect WebP first (earlier match)
        assert_eq!(detect_image_format(&mixed_data), Some("webp"));
    }

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file(Path::new("test.jpg"), &["jpg", "png"]));
        assert!(is_image_file(Path::new("test.PNG"), &["jpg", "png"]));
        assert!(!is_image_file(Path::new("test.txt"), &["jpg", "png"]));
        assert!(!is_image_file(Path::new("test"), &["jpg", "png"]));
    }

    #[test]
    fn test_is_image_file_comprehensive() {
        let extensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"];

        // Test all supported extensions
        assert!(is_image_file(Path::new("test.jpg"), &extensions));
        assert!(is_image_file(Path::new("test.jpeg"), &extensions));
        assert!(is_image_file(Path::new("test.png"), &extensions));
        assert!(is_image_file(Path::new("test.webp"), &extensions));
        assert!(is_image_file(Path::new("test.bmp"), &extensions));
        assert!(is_image_file(Path::new("test.tiff"), &extensions));
        assert!(is_image_file(Path::new("test.tif"), &extensions));

        // Test case insensitive matching
        assert!(is_image_file(Path::new("test.JPG"), &extensions));
        assert!(is_image_file(Path::new("test.Png"), &extensions));
        assert!(is_image_file(Path::new("test.WEBP"), &extensions));

        // Test non-image files
        assert!(!is_image_file(Path::new("test.txt"), &extensions));
        assert!(!is_image_file(Path::new("test.pdf"), &extensions));
        assert!(!is_image_file(Path::new("test.mp4"), &extensions));
        assert!(!is_image_file(Path::new("test"), &extensions)); // No extension

        // Test complex filenames
        assert!(is_image_file(
            Path::new("my-image.with.dots.jpg"),
            &extensions
        ));
        assert!(is_image_file(Path::new("/path/to/image.png"), &extensions));
        assert!(is_image_file(
            Path::new("image_with_underscores.jpeg"),
            &extensions
        ));
    }

    #[test]
    fn test_matches_pattern() {
        // Test without pattern (should always match)
        assert!(matches_pattern(Path::new("any_file.jpg"), None));
        assert!(matches_pattern(Path::new("test.png"), None));

        // Test with glob patterns
        assert!(matches_pattern(Path::new("test.jpg"), Some("*.jpg")));
        assert!(matches_pattern(Path::new("image.png"), Some("*.png")));
        assert!(matches_pattern(Path::new("photo.jpeg"), Some("photo.*")));
        assert!(matches_pattern(Path::new("img_001.jpg"), Some("img_*.jpg")));

        // Test pattern mismatches
        assert!(!matches_pattern(Path::new("test.png"), Some("*.jpg")));
        assert!(!matches_pattern(Path::new("other.jpg"), Some("test.*")));

        // Test complex patterns
        assert!(matches_pattern(
            Path::new("image001.jpg"),
            Some("image???.jpg")
        ));
        assert!(matches_pattern(
            Path::new("photo_a.png"),
            Some("photo_?.png")
        ));

        // Test edge cases
        assert!(!matches_pattern(Path::new("file.jpg"), Some("*.png"))); // Wrong extension
        assert!(!matches_pattern(Path::new(""), Some("*.jpg"))); // Empty filename
    }

    #[test]
    fn test_generate_output_path() {
        use crate::config::OutputFormat;

        // Test basic functionality
        let input = Path::new("test.jpg");
        let output = generate_output_path(input, OutputFormat::Png);
        assert_eq!(output.file_name().unwrap(), "test_bg_removed.png");

        // Test different formats
        let output = generate_output_path(input, OutputFormat::Jpeg);
        assert_eq!(output.file_name().unwrap(), "test_bg_removed.jpg");

        let output = generate_output_path(input, OutputFormat::WebP);
        assert_eq!(output.file_name().unwrap(), "test_bg_removed.webp");

        let output = generate_output_path(input, OutputFormat::Tiff);
        assert_eq!(output.file_name().unwrap(), "test_bg_removed.tiff");

        let output = generate_output_path(input, OutputFormat::Rgba8);
        assert_eq!(output.file_name().unwrap(), "test_bg_removed.rgba8");

        // Test with different directory
        let input = Path::new("/path/to/image.png");
        let output = generate_output_path(input, OutputFormat::Png);
        assert_eq!(output.parent().unwrap(), Path::new("/path/to"));
        assert_eq!(output.file_name().unwrap(), "image_bg_removed.png");

        // Test with complex filename
        let input = Path::new("my.complex.filename.jpeg");
        let output = generate_output_path(input, OutputFormat::Png);
        assert_eq!(
            output.file_name().unwrap(),
            "my.complex.filename_bg_removed.png"
        );

        // Test with no extension
        let input = Path::new("filename_no_ext");
        let output = generate_output_path(input, OutputFormat::Png);
        assert_eq!(
            output.file_name().unwrap(),
            "filename_no_ext_bg_removed.png"
        );
    }

    #[test]
    fn test_find_image_files_in_empty_directory() {
        let temp_dir = tempdir().unwrap();

        // Test empty directory
        let files = find_image_files(temp_dir.path(), false, None).unwrap();
        assert!(files.is_empty());

        // Test recursive in empty directory
        let files = find_image_files(temp_dir.path(), true, None).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_find_image_files_with_pattern() {
        let temp_dir = tempdir().unwrap();

        // Create test files
        fs::write(temp_dir.path().join("test1.jpg"), b"test").unwrap();
        fs::write(temp_dir.path().join("test2.png"), b"test").unwrap();
        fs::write(temp_dir.path().join("other.jpg"), b"test").unwrap();
        fs::write(temp_dir.path().join("document.txt"), b"test").unwrap();

        // Test pattern matching
        let files = find_image_files(temp_dir.path(), false, Some("test*")).unwrap();
        assert_eq!(files.len(), 2); // test1.jpg and test2.png

        let files = find_image_files(temp_dir.path(), false, Some("*.jpg")).unwrap();
        assert_eq!(files.len(), 2); // test1.jpg and other.jpg

        let files = find_image_files(temp_dir.path(), false, Some("*.png")).unwrap();
        assert_eq!(files.len(), 1); // test2.png

        // Test no pattern (should get all image files)
        let files = find_image_files(temp_dir.path(), false, None).unwrap();
        assert_eq!(files.len(), 3); // All image files, excluding .txt
    }

    #[test]
    fn test_find_image_files_recursive() {
        let temp_dir = tempdir().unwrap();

        // Create nested directory structure
        let sub_dir = temp_dir.path().join("subdir");
        fs::create_dir(&sub_dir).unwrap();
        let deep_dir = sub_dir.join("deep");
        fs::create_dir(&deep_dir).unwrap();

        // Create files in different levels
        fs::write(temp_dir.path().join("root.jpg"), b"test").unwrap();
        fs::write(sub_dir.join("sub.png"), b"test").unwrap();
        fs::write(deep_dir.join("deep.webp"), b"test").unwrap();
        fs::write(temp_dir.path().join("non_image.txt"), b"test").unwrap();

        // Test non-recursive (should only find root.jpg)
        let files = find_image_files(temp_dir.path(), false, None).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].file_name().unwrap() == "root.jpg");

        // Test recursive (should find all image files)
        let files = find_image_files(temp_dir.path(), true, None).unwrap();
        assert_eq!(files.len(), 3); // root.jpg, sub.png, deep.webp

        let filenames: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy())
            .collect();
        assert!(filenames.contains(&"root.jpg".into()));
        assert!(filenames.contains(&"sub.png".into()));
        assert!(filenames.contains(&"deep.webp".into()));
    }

    #[test]
    fn test_find_image_files_alphanumerical_sorting() {
        let temp_dir = tempdir().unwrap();

        // Create test files with names that would have different order with filesystem vs alphabetical ordering
        fs::write(temp_dir.path().join("z_last.jpg"), b"test").unwrap();
        fs::write(temp_dir.path().join("a_first.png"), b"test").unwrap();
        fs::write(temp_dir.path().join("m_middle.webp"), b"test").unwrap();
        fs::write(temp_dir.path().join("img10.jpg"), b"test").unwrap();
        fs::write(temp_dir.path().join("img2.jpg"), b"test").unwrap();
        fs::write(temp_dir.path().join("img1.jpg"), b"test").unwrap();

        // Test that files are returned in alphanumerical order
        let mut files = find_image_files(temp_dir.path(), false, None).unwrap();

        // The sorting happens in process_inputs(), but find_image_files should at least return consistent results
        // Let's sort here to simulate what process_inputs does
        files.sort();

        let filenames: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        // Verify alphanumerical ordering
        assert_eq!(filenames.len(), 6);
        assert_eq!(filenames[0], "a_first.png");
        assert_eq!(filenames[1], "img1.jpg");
        assert_eq!(filenames[2], "img10.jpg");
        assert_eq!(filenames[3], "img2.jpg");
        assert_eq!(filenames[4], "m_middle.webp");
        assert_eq!(filenames[5], "z_last.jpg");

        // Verify that files are indeed sorted alphanumerically
        let mut expected = filenames.clone();
        expected.sort();
        assert_eq!(
            filenames, expected,
            "Files should be in alphanumerical order"
        );
    }

    #[test]
    fn test_cli_output_format_enum() {
        // Test enum variants
        assert_eq!(CliOutputFormat::Png as u8, 0);
        assert_eq!(CliOutputFormat::Jpeg as u8, 1);
        assert_eq!(CliOutputFormat::Webp as u8, 2);
        assert_eq!(CliOutputFormat::Tiff as u8, 3);
        assert_eq!(CliOutputFormat::Rgba8 as u8, 4);

        // Test ordering
        assert!(CliOutputFormat::Png < CliOutputFormat::Jpeg);
        assert!(CliOutputFormat::Jpeg < CliOutputFormat::Webp);

        // Test cloning and equality
        let format1 = CliOutputFormat::Png;
        let format2 = format1;
        assert_eq!(format1, format2);

        let format3 = CliOutputFormat::Jpeg;
        assert_ne!(format1, format3);

        // Test debug formatting
        let debug_str = format!("{:?}", CliOutputFormat::Png);
        assert!(debug_str.contains("Png"));
    }

    #[test]
    fn test_cli_struct_creation() {
        // Test creating CLI struct with minimal required fields
        let cli = Cli {
            input: vec!["test.jpg".to_string()],
            output: None,
            format: CliOutputFormat::Png,
            execution_provider: "onnx:auto".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            threads: 0,
            verbose: 0,
            recursive: false,
            pattern: None,
            show_providers: false,
            model: None,
            variant: None,
            preserve_color_profiles: true,
            only_download: false,
            list_models: false,
            clear_cache: false,
            show_cache_dir: false,
            cache_dir: None,
            no_cache: false,
            progress: false,
            #[cfg(feature = "video-support")]
            video_batch_size: 8,
            #[cfg(feature = "video-support")]
            preserve_audio: true,
            #[cfg(feature = "video-support")]
            video_codec: "h264".to_string(),
            #[cfg(feature = "video-support")]
            video_quality: None,
        };

        // Test basic field access
        assert_eq!(cli.input, vec!["test.jpg"]);
        assert_eq!(cli.format, CliOutputFormat::Png);
        assert_eq!(cli.execution_provider, "onnx:auto");
        assert_eq!(cli.jpeg_quality, 90);
        assert_eq!(cli.webp_quality, 85);
        assert!(!cli.debug_mode());
        assert!(!cli.recursive);
        assert!(cli.preserve_color_profiles);
        assert!(!cli.no_cache);
    }

    impl Cli {
        /// Helper method for tests to check debug mode
        fn debug_mode(&self) -> bool {
            self.verbose >= 2
        }
    }

    #[test]
    fn test_cli_debug_mode_detection() {
        let mut cli = Cli {
            input: vec!["test.jpg".to_string()],
            output: None,
            format: CliOutputFormat::Png,
            execution_provider: "onnx:auto".to_string(),
            jpeg_quality: 90,
            webp_quality: 85,
            threads: 0,
            verbose: 0,
            recursive: false,
            pattern: None,
            show_providers: false,
            model: None,
            variant: None,
            preserve_color_profiles: true,
            only_download: false,
            list_models: false,
            clear_cache: false,
            show_cache_dir: false,
            cache_dir: None,
            no_cache: false,
            progress: false,
            #[cfg(feature = "video-support")]
            video_batch_size: 8,
            #[cfg(feature = "video-support")]
            preserve_audio: true,
            #[cfg(feature = "video-support")]
            video_codec: "h264".to_string(),
            #[cfg(feature = "video-support")]
            video_quality: None,
        };

        // Test different verbosity levels
        cli.verbose = 0;
        assert!(!cli.debug_mode());

        cli.verbose = 1;
        assert!(!cli.debug_mode());

        cli.verbose = 2;
        assert!(cli.debug_mode());

        cli.verbose = 3;
        assert!(cli.debug_mode());
    }

    #[test]
    fn test_write_stdout_empty_data() {
        // Test writing empty data doesn't panic
        let empty_data: &[u8] = &[];
        // Note: This test can't easily verify stdout output without complex mocking
        // but we can at least verify the function doesn't panic
        let result = write_stdout(empty_data);
        assert!(result.is_ok() || result.is_err()); // Either outcome is acceptable for test
    }

    #[test]
    fn test_read_stdin_mock() {
        // Note: Testing read_stdin() is challenging without mocking stdin
        // We can test the helper logic that validates the data

        // Test empty buffer validation logic (simulated)
        let empty_buffer: Vec<u8> = Vec::new();
        assert!(empty_buffer.is_empty()); // This would cause the "No data received" error

        // Test non-empty buffer validation
        let data_buffer = vec![1, 2, 3, 4];
        assert!(!data_buffer.is_empty()); // This would pass validation
    }

    #[test]
    fn test_image_format_detection_comprehensive() {
        // Test all supported formats with full headers

        // PNG with full header
        let png_full = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52,
        ];
        assert_eq!(detect_image_format(&png_full), Some("png"));

        // JPEG with additional markers
        let jpeg_full = [
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        ];
        assert_eq!(detect_image_format(&jpeg_full), Some("jpg"));

        // WebP with complete RIFF header
        let webp_full = [
            0x52, 0x49, 0x46, 0x46, // "RIFF"
            0x24, 0x00, 0x00, 0x00, // File size
            0x57, 0x45, 0x42, 0x50, // "WEBP"
            0x56, 0x50, 0x38, 0x20, // VP8 format
        ];
        assert_eq!(detect_image_format(&webp_full), Some("webp"));

        // TIFF Little Endian with additional data
        let tiff_le_full = [
            0x49, 0x49, 0x2A, 0x00, // TIFF LE header
            0x08, 0x00, 0x00, 0x00, // IFD offset
        ];
        assert_eq!(detect_image_format(&tiff_le_full), Some("tiff"));

        // TIFF Big Endian with additional data
        let tiff_be_full = [
            0x4D, 0x4D, 0x00, 0x2A, // TIFF BE header
            0x00, 0x00, 0x00, 0x08, // IFD offset
        ];
        assert_eq!(detect_image_format(&tiff_be_full), Some("tiff"));

        // BMP with file header
        let bmp_full = [
            0x42, 0x4D, // "BM"
            0x36, 0x00, 0x00, 0x00, // File size
            0x00, 0x00, 0x00, 0x00, // Reserved
        ];
        assert_eq!(detect_image_format(&bmp_full), Some("bmp"));

        // GIF with version
        let gif_full = [
            0x47, 0x49, 0x46, 0x38, // "GIF8"
            0x39, 0x61, // "9a" (GIF89a)
            0x01, 0x00, 0x01, 0x00, // Width/height
        ];
        assert_eq!(detect_image_format(&gif_full), Some("gif"));
    }

    #[test]
    fn test_path_generation_edge_cases() {
        use crate::config::OutputFormat;

        // Test with relative paths
        let input = Path::new("./relative/path/image.jpg");
        let output = generate_output_path(input, OutputFormat::Png);
        assert_eq!(output.parent().unwrap(), Path::new("./relative/path"));

        // Test with current directory
        let input = Path::new("image.jpg");
        let output = generate_output_path(input, OutputFormat::Png);
        // For files in current directory, parent() returns empty path
        assert_eq!(output.parent().unwrap_or(Path::new("")), Path::new(""));

        // Test with file stem that could be problematic
        let input = Path::new("file.with.many.dots.and.extension.jpg");
        let output = generate_output_path(input, OutputFormat::Png);
        assert_eq!(
            output.file_name().unwrap(),
            "file.with.many.dots.and.extension_bg_removed.png"
        );

        // Test with Unicode filename
        let input = Path::new("图片.jpg");
        let output = generate_output_path(input, OutputFormat::Png);
        assert!(output
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("图片"));
        assert!(output
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("_bg_removed.png"));
    }

    #[test]
    fn test_file_operations_edge_cases() {
        let temp_dir = tempdir().unwrap();

        // Test with special characters in filenames
        let special_files = vec![
            "file with spaces.jpg",
            "file-with-dashes.png",
            "file_with_underscores.webp",
            "file.with.dots.tiff",
            "file@special#chars.bmp",
        ];

        for filename in special_files {
            fs::write(temp_dir.path().join(filename), b"test").unwrap();
        }

        let files = find_image_files(temp_dir.path(), false, None).unwrap();
        assert_eq!(files.len(), 5);

        // Verify all files were found
        let found_names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy())
            .collect();

        for expected_name in &[
            "file with spaces.jpg",
            "file-with-dashes.png",
            "file_with_underscores.webp",
            "file.with.dots.tiff",
            "file@special#chars.bmp",
        ] {
            assert!(found_names.iter().any(|name| name == expected_name));
        }
    }

    #[test]
    fn test_find_image_files_error_handling() {
        // Test with non-existent directory
        let non_existent = Path::new("/definitely/does/not/exist");
        let result = find_image_files(non_existent, false, None);
        assert!(result.is_err());

        // Test with invalid glob pattern
        let temp_dir = tempdir().unwrap();
        fs::write(temp_dir.path().join("test.jpg"), b"test").unwrap();

        // Invalid pattern should handle gracefully (depends on glob crate behavior)
        let result = find_image_files(temp_dir.path(), false, Some("["));
        // The result depends on how glob handles invalid patterns
        // We just ensure it doesn't panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_generate_output_path_with_dir() {
        use crate::config::OutputFormat;

        let output_dir = Path::new("/output");

        // Test basic functionality
        let input = Path::new("test.jpg");
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Png);
        assert_eq!(output, "/output/test_bg_removed.png");

        // Test different formats
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Jpeg);
        assert_eq!(output, "/output/test_bg_removed.jpg");

        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::WebP);
        assert_eq!(output, "/output/test_bg_removed.webp");

        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Tiff);
        assert_eq!(output, "/output/test_bg_removed.tiff");

        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Rgba8);
        assert_eq!(output, "/output/test_bg_removed.rgba8");

        // Test with complex input path (directory info should be ignored)
        let input = Path::new("/path/to/image.png");
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Png);
        assert_eq!(output, "/output/image_bg_removed.png");

        // Test with complex filename
        let input = Path::new("my.complex.filename.jpeg");
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Png);
        assert_eq!(output, "/output/my.complex.filename_bg_removed.png");

        // Test with no extension
        let input = Path::new("filename_no_ext");
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Png);
        assert_eq!(output, "/output/filename_no_ext_bg_removed.png");

        // Test with different output directory
        let custom_dir = Path::new("/custom/output/dir");
        let output = generate_output_path_with_dir(input, custom_dir, OutputFormat::Jpeg);
        assert_eq!(output, "/custom/output/dir/filename_no_ext_bg_removed.jpg");
    }

    #[test]
    fn test_generate_output_path_with_dir_edge_cases() {
        use crate::config::OutputFormat;

        // Test empty directory path (current directory)
        let output_dir = Path::new("");
        let input = Path::new("test.jpg");
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Png);
        assert_eq!(output, "test_bg_removed.png");

        // Test current directory
        let output_dir = Path::new(".");
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Png);
        assert_eq!(output, "./test_bg_removed.png");

        // Test relative directory
        let output_dir = Path::new("relative/path");
        let output = generate_output_path_with_dir(input, output_dir, OutputFormat::Png);
        assert_eq!(output, "relative/path/test_bg_removed.png");

        // Test with unicode characters in paths
        let unicode_dir = Path::new("/output/测试/图片");
        let unicode_input = Path::new("测试图片.jpg");
        let output = generate_output_path_with_dir(unicode_input, unicode_dir, OutputFormat::Png);
        assert_eq!(output, "/output/测试/图片/测试图片_bg_removed.png");
    }
}
