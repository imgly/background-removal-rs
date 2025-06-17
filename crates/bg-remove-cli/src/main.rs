//! Background Removal CLI Tool - Refactored
//!
//! Command-line interface for removing backgrounds from images using the unified processor.

mod backend_factory;
mod config;

use anyhow::{Context, Result};
use backend_factory::CliBackendFactory;
use bg_remove_core::{
    processor::BackgroundRemovalProcessor,
    utils::ExecutionProviderManager,
};
use clap::{Parser, ValueEnum};
use config::CliConfigBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Background removal CLI tool
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "bg-remove")]
pub struct Cli {
    /// Input image files or directories (use "-" for stdin)
    #[arg(value_name = "INPUT", required_unless_present = "show_providers")]
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

    /// Background color (hex format, e.g., #ffffff)
    #[arg(long, default_value = "#ffffff")]
    pub background_color: String,

    /// Number of intra-op threads (0 = auto, optimal for compute)
    #[arg(long, default_value_t = 0)]
    pub intra_threads: usize,

    /// Number of inter-op threads (0 = auto, optimal for coordination)  
    #[arg(long, default_value_t = 0)]
    pub inter_threads: usize,

    /// Number of threads - sets both intra and inter optimally (0 = auto)
    #[arg(short, long, default_value_t = 0)]
    pub threads: usize,

    /// Enable debug mode
    #[arg(short, long)]
    pub debug: bool,

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

    /// Model name or path to model folder (optional if embedded models available)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Model variant (fp16, fp32). Defaults to fp16
    #[arg(long)]
    pub variant: Option<String>,

    /// Preserve ICC color profiles from input images (enabled by default)
    #[arg(long, default_value_t = true)]
    pub preserve_color_profile: bool,

    /// Disable ICC color profile preservation
    #[arg(long, conflicts_with = "preserve_color_profile")]
    pub no_preserve_color_profile: bool,

    /// Force sRGB output regardless of input color profile
    #[arg(long)]
    pub force_srgb: bool,

    /// Embed color profile in output when supported by format (enabled by default)
    #[arg(long, default_value_t = true)]
    pub embed_profile: bool,

    /// Disable embedding color profiles in output
    #[arg(long, conflicts_with = "embed_profile")]
    pub no_embed_profile: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum CliOutputFormat {
    Png,
    Jpeg,
    Webp,
    Tiff,
    Rgba8,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose);

    // Handle provider diagnostics flag
    if cli.show_providers {
        return show_provider_diagnostics();
    }

    if cli.input.is_empty() {
        anyhow::bail!("At least one input is required");
    }

    // Validate CLI arguments
    CliConfigBuilder::validate_cli(&cli).context("Invalid CLI arguments")?;

    // Convert CLI arguments to unified configuration
    let config = CliConfigBuilder::from_cli(&cli).context("Failed to build configuration")?;

    info!("Starting background removal CLI");
    if cli.debug {
        info!("🐛 DEBUG MODE: Using configured backend for testing");
    }
    info!("Input(s): {}", cli.input.join(", "));
    info!("Backend: {:?}, Provider: {:?}", config.backend_type, config.execution_provider);
    info!("Model: {:?}", config.model_spec);

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

/// Initialize logging based on verbosity level
fn init_logging(verbose_count: u8) {
    let log_level = match verbose_count {
        0 => "warn",        // Default: only warnings and errors
        1 => "info",        // -v: user-actionable information
        2 => "debug",       // -vv: internal state and computations  
        _ => "trace",       // -vvv+: extremely detailed traces
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
fn show_provider_diagnostics() -> Result<()> {
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
            "❓ Unknown availability"
        };
        println!("  • {}: {} - {}", provider_info.name, status, provider_info.description);
    }

    // Try to get actual availability from backends
    println!("\n🔍 Checking Backend-Specific Availability:");
    
    // Check ONNX providers
    match bg_remove_onnx::OnnxBackend::list_providers() {
        providers => {
            println!("ONNX Runtime providers:");
            for (name, available, description) in providers {
                let status = if available {
                    "✅ Available"
                } else {
                    "❌ Not Available"
                };
                println!("  • onnx:{}: {} - {}", name.to_lowercase(), status, description);
            }
        }
    }

    // Check Tract providers
    let tract_providers = bg_remove_tract::TractBackend::list_providers();
    println!("Tract providers:");
    for (name, available, description) in tract_providers {
        let status = if available {
            "✅ Available"
        } else {
            "❌ Not Available"
        };
        println!("  • tract:{}: {} - {}", name.to_lowercase(), status, description);
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

    Ok(())
}

/// Process multiple inputs efficiently using the unified processor
async fn process_inputs(
    cli: &Cli,
    processor: &mut BackgroundRemovalProcessor,
) -> Result<usize> {
    // Handle stdin specially (single input)
    if cli.input.len() == 1 && cli.input[0] == "-" {
        return process_stdin(&cli.output, processor).await;
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
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
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

        match process_single_file(processor, &input_file, &output_path).await {
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
        info!("  └─ Average per file: {:.2}s", 
              if processed_count > 0 { batch_total_time.as_secs_f64() / (processed_count as f64) } else { 0.0 });
    }

    Ok(processed_count)
}

/// Process image from stdin using the unified processor
async fn process_stdin(
    output_target: &Option<String>,
    processor: &mut BackgroundRemovalProcessor,
) -> Result<usize> {
    info!("Reading image from stdin");

    let image_data = read_stdin()?;
    let start_time = Instant::now();

    // Save to temporary file for processing
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("stdin_input.tmp");
    std::fs::write(&temp_file, &image_data)?;

    let result = processor.process_file(&temp_file).await
        .context("Failed to remove background")?;

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_file);

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
            if config.color_management.preserve_color_profile {
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
    output_path: &Option<String>,
) -> Result<usize> {
    let mut result = processor.process_file(input_path).await
        .context("Failed to remove background")?;

    // Show detailed timing breakdown
    let timings = result.timings();
    let breakdown = timings.breakdown_percentages();
    
    info!("📊 Processing breakdown for {}:", input_path.display());
    
    if timings.model_load_ms > 0 {
        info!("  ├─ Model Load: {}ms ({:.1}%)", timings.model_load_ms, breakdown.model_load_pct);
    }
    
    info!("  ├─ Image Decode: {}ms ({:.1}%)", timings.image_decode_ms, breakdown.decode_pct);
    info!("  ├─ Preprocessing: {}ms ({:.1}%)", timings.preprocessing_ms, breakdown.preprocessing_pct);
    info!("  ├─ Inference: {}ms ({:.1}%)", timings.inference_ms, breakdown.inference_pct);
    info!("  ├─ Postprocessing: {}ms ({:.1}%)", timings.postprocessing_ms, breakdown.postprocessing_pct);

    // Handle output
    let config = processor.config();
    match output_path {
        Some(target) if target == "-" => {
            // Output to stdout
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            info!("  └─ Total: {}ms ({:.2}s) - output to stdout", result.timings().total_ms, result.timings().total_ms as f64 / 1000.0);
        },
        Some(target) => {
            // Output to specific file
            let output_path = PathBuf::from(target);
            if config.color_management.preserve_color_profile {
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
                info!("  ├─ Image Encode: {}ms ({:.1}%)", encode_ms, encode_breakdown.encode_pct);
            }
            
            info!("  └─ Total: {}ms ({:.2}s)", result.timings().total_ms, result.timings().total_ms as f64 / 1000.0);
        },
        None => {
            // Generate default output filename
            let output_path = generate_output_path(input_path, config.output_format);
            if config.color_management.preserve_color_profile {
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
                info!("  ├─ Image Encode: {}ms ({:.1}%)", encode_ms, encode_breakdown.encode_pct);
            }
            
            info!("  └─ Total: {}ms ({:.2}s)", result.timings().total_ms, result.timings().total_ms as f64 / 1000.0);
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
fn generate_output_path(input_path: &Path, format: bg_remove_core::OutputFormat) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or_default();
    let dir = input_path.parent().unwrap_or(Path::new("."));

    let extension = match format {
        bg_remove_core::OutputFormat::Png => "png",
        bg_remove_core::OutputFormat::Jpeg => "jpg",
        bg_remove_core::OutputFormat::WebP => "webp",
        bg_remove_core::OutputFormat::Tiff => "tiff",
        bg_remove_core::OutputFormat::Rgba8 => "rgba8",
    };

    dir.join(format!(
        "{}_bg_removed.{}",
        stem.to_string_lossy(),
        extension
    ))
}