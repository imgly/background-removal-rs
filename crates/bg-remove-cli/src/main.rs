//! Background Removal CLI Tool
//!
//! Command-line interface for removing backgrounds from images using `ISNet` models.

use anyhow::{Context, Result};
use bg_remove_core::config::{BackgroundColor, ColorManagementConfig};
use bg_remove_core::{
    get_available_embedded_models, remove_background_with_model, ExecutionProvider, ModelManager,
    ModelSource, ModelSpec, OutputFormat, RemovalConfig,
};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Background removal CLI tool
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "bg-remove")]
struct Cli {
    /// Input image files or directories (use "-" for stdin)
    #[arg(value_name = "INPUT", required_unless_present = "show_providers")]
    input: Vec<String>,

    /// Output file or directory (use "-" for stdout)
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<String>,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = CliOutputFormat::Png)]
    format: CliOutputFormat,

    /// Execution provider for ONNX Runtime
    #[arg(short, long, value_enum, default_value_t = CliExecutionProvider::Auto)]
    execution_provider: CliExecutionProvider,

    /// JPEG quality (0-100)
    #[arg(long, default_value_t = 90)]
    jpeg_quality: u8,

    /// WebP quality (0-100)
    #[arg(long, default_value_t = 85)]
    webp_quality: u8,

    /// Background color (hex format, e.g., #ffffff)
    #[arg(long, default_value = "#ffffff")]
    background_color: String,

    /// Number of intra-op threads (0 = auto, optimal for compute)
    #[arg(long, default_value_t = 0)]
    intra_threads: usize,

    /// Number of inter-op threads (0 = auto, optimal for coordination)  
    #[arg(long, default_value_t = 0)]
    inter_threads: usize,

    /// Number of threads - sets both intra and inter optimally (0 = auto)
    #[arg(short, long, default_value_t = 0)]
    threads: usize,

    /// Enable debug mode
    #[arg(short, long)]
    debug: bool,

    /// Enable verbose logging (-v: INFO, -vv: DEBUG, -vvv: TRACE)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Process directory recursively
    #[arg(short, long)]
    recursive: bool,

    /// Pattern for batch processing (e.g., "*.jpg")
    #[arg(long)]
    pattern: Option<String>,

    /// Show execution provider diagnostics and exit
    #[arg(long)]
    show_providers: bool,

    /// Model name or path to model folder (optional if embedded models available)
    #[arg(short, long)]
    model: Option<String>,

    /// Model variant (fp16, fp32). Defaults to fp16
    #[arg(long)]
    variant: Option<String>,

    /// Preserve ICC color profiles from input images (enabled by default)
    #[arg(long, default_value_t = true)]
    preserve_color_profile: bool,

    /// Disable ICC color profile preservation
    #[arg(long, conflicts_with = "preserve_color_profile")]
    no_preserve_color_profile: bool,

    /// Force sRGB output regardless of input color profile
    #[arg(long)]
    force_srgb: bool,

    /// Embed color profile in output when supported by format (enabled by default)
    #[arg(long, default_value_t = true)]
    embed_profile: bool,

    /// Disable embedding color profiles in output
    #[arg(long, conflicts_with = "embed_profile")]
    no_embed_profile: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CliOutputFormat {
    Png,
    Jpeg,
    Webp,
    Tiff,
    Rgba8,
}

impl From<CliOutputFormat> for OutputFormat {
    fn from(cli_format: CliOutputFormat) -> Self {
        match cli_format {
            CliOutputFormat::Png => OutputFormat::Png,
            CliOutputFormat::Jpeg => OutputFormat::Jpeg,
            CliOutputFormat::Webp => OutputFormat::WebP,
            CliOutputFormat::Tiff => OutputFormat::Tiff,
            CliOutputFormat::Rgba8 => OutputFormat::Rgba8,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CliExecutionProvider {
    /// Auto-detect best available provider (CUDA > `CoreML` > CPU)
    Auto,
    /// CPU execution (always available)
    Cpu,
    /// NVIDIA CUDA GPU acceleration
    Cuda,
    /// Apple Silicon GPU acceleration (Metal Performance Shaders)
    CoreMl,
}

impl From<CliExecutionProvider> for ExecutionProvider {
    fn from(cli_provider: CliExecutionProvider) -> Self {
        match cli_provider {
            CliExecutionProvider::Auto => ExecutionProvider::Auto,
            CliExecutionProvider::Cpu => ExecutionProvider::Cpu,
            CliExecutionProvider::Cuda => ExecutionProvider::Cuda,
            CliExecutionProvider::CoreMl => ExecutionProvider::CoreMl,
        }
    }
}

/// Parse model argument into ModelSpec with optional variant suffix
fn parse_model_spec(model_arg: &str) -> ModelSpec {
    // Check for suffix syntax: "model:variant"
    if let Some((path_part, variant_part)) = model_arg.split_once(':') {
        let source = if Path::new(path_part).exists() {
            ModelSource::External(PathBuf::from(path_part))
        } else {
            ModelSource::Embedded(path_part.to_string())
        };

        return ModelSpec {
            source,
            variant: Some(variant_part.to_string()),
        };
    }

    // No suffix - determine source type
    let source = if Path::new(model_arg).exists() {
        ModelSource::External(PathBuf::from(model_arg))
    } else {
        ModelSource::Embedded(model_arg.to_string())
    };

    ModelSpec {
        source,
        variant: None,
    }
}

/// Resolve the final variant to use based on precedence rules
#[allow(dead_code)] // TODO: Remove when Phase 2+ is implemented
fn resolve_variant(
    model_spec: &ModelSpec,
    cli_variant: Option<&str>,
    available_variants: &[String],
) -> Result<String> {
    // Precedence: CLI param > suffix > auto-detection > default

    // 1. CLI parameter has highest precedence
    if let Some(variant) = cli_variant {
        if available_variants.contains(&variant.to_string()) {
            return Ok(variant.to_string());
        } else {
            return Err(anyhow::anyhow!(
                "Variant '{}' not available. Available variants: {:?}",
                variant,
                available_variants
            ));
        }
    }

    // 2. Suffix syntax has medium precedence
    if let Some(variant) = &model_spec.variant {
        if available_variants.contains(variant) {
            return Ok(variant.clone());
        } else {
            return Err(anyhow::anyhow!(
                "Variant '{}' not available. Available variants: {:?}",
                variant,
                available_variants
            ));
        }
    }

    // 3. Auto-detection: prefer fp16, fallback to available
    if available_variants.contains(&"fp16".to_string()) {
        return Ok("fp16".to_string());
    }

    if available_variants.contains(&"fp32".to_string()) {
        return Ok("fp32".to_string());
    }

    // 4. Use first available variant
    if let Some(first) = available_variants.first() {
        return Ok(first.clone());
    }

    Err(anyhow::anyhow!("No variants available for model"))
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
        return Err(anyhow::anyhow!("At least one input is required"));
    }

    // Parse model parameter or use default embedded model
    let (model_spec, model_arg) = if let Some(model_arg) = &cli.model {
        // User specified a model
        let model_spec = parse_model_spec(model_arg);
        (model_spec, model_arg.clone())
    } else {
        // No model specified - try to use first available embedded model
        let available_embedded = get_available_embedded_models();
        if available_embedded.is_empty() {
            return Err(anyhow::anyhow!(
                "No model specified and no embedded models available. Use --model to specify a model name or path, or build with embed-* features."
            ));
        }

        let default_model = &available_embedded[0];
        let model_spec = ModelSpec {
            source: ModelSource::Embedded(default_model.clone()),
            variant: None,
        };
        (model_spec, default_model.clone())
    };

    info!("Starting background removal CLI");
    if cli.debug {
        info!("🐛 DEBUG MODE: Using mock backend for testing");
    }
    // Verbose mode information is already shown in init_logging()
    info!("Input(s): {}", cli.input.join(", "));
    info!(
        "Model: {} ({})",
        model_arg,
        match &model_spec.source {
            ModelSource::Embedded(name) => format!("embedded: {name}"),
            ModelSource::External(path) => format!("external: {}", path.display()),
        }
    );

    // Resolve final variant based on CLI parameter precedence
    let final_variant = cli.variant.clone().or_else(|| model_spec.variant.clone());

    // Parse background color
    let background_color =
        parse_color(&cli.background_color).context("Invalid background color format")?;

    // Build configuration first so we can use it for model optimization
    let config = RemovalConfig::builder()
        .execution_provider(cli.execution_provider.into())
        .output_format(cli.format.into())
        .background_color(background_color)
        .jpeg_quality(cli.jpeg_quality)
        .webp_quality(cli.webp_quality)
        .debug(cli.debug)
        .num_threads(cli.threads)
        .intra_threads(if cli.intra_threads > 0 {
            cli.intra_threads
        } else {
            0
        })
        .inter_threads(if cli.inter_threads > 0 {
            cli.inter_threads
        } else {
            0
        })
        .color_management(ColorManagementConfig {
            preserve_color_profile: cli.preserve_color_profile && !cli.no_preserve_color_profile,
            force_srgb_output: cli.force_srgb,
            fallback_to_srgb: true, // Always fallback to sRGB for safety
            embed_profile_in_output: cli.embed_profile && !cli.no_embed_profile,
        })
        .build()
        .context("Invalid configuration")?;

    // Create final model spec with resolved variant
    let final_model_spec = ModelSpec {
        source: model_spec.source.clone(),
        variant: final_variant.clone(),
    };

    // Validate model by attempting to create ModelManager with execution provider optimization
    let _model_manager =
        ModelManager::from_spec_with_provider(&final_model_spec, Some(&config.execution_provider))
            .context("Failed to load specified model")?;

    match &final_model_spec.source {
        ModelSource::Embedded(name) => {
            info!("Using embedded model: {name}");
            if let Some(variant) = &final_variant {
                info!("Using variant: {variant}");
            }
        },
        ModelSource::External(path) => {
            info!("Using external model from: {}", path.display());
            if let Some(variant) = &final_variant {
                info!("Using variant: {variant}");
            }
        },
    }

    if cli.verbose > 1 {
        log::debug!(
            "Configuration: execution_provider={:?}, output_format={:?}, debug={}",
            config.execution_provider,
            config.output_format,
            config.debug
        );
        log::debug!(
            "Threading: intra_threads={}, inter_threads={}",
            config.intra_threads,
            config.inter_threads
        );
        log::debug!(
            "Background color: RGB({}, {}, {})",
            config.background_color.r,
            config.background_color.g,
            config.background_color.b
        );
        log::debug!(
            "Quality settings: jpeg={}, webp={}",
            config.jpeg_quality,
            config.webp_quality
        );
        log::debug!(
            "Color management: preserve={}, force_srgb={}, embed={}",
            config.color_management.preserve_color_profile,
            config.color_management.force_srgb_output,
            config.color_management.embed_profile_in_output
        );
    }

    // Process inputs (can be files, directories, or stdin)
    let start_time = Instant::now();
    let processed_count = process_inputs(&cli, &config, &final_model_spec).await?;

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

/// Read image data from stdin
fn read_stdin() -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    io::stdin()
        .read_to_end(&mut buffer)
        .context("Failed to read image data from stdin")?;

    if buffer.is_empty() {
        return Err(anyhow::anyhow!("No data received from stdin"));
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

/// Display execution provider diagnostics
fn show_provider_diagnostics() -> Result<()> {
    use bg_remove_core::backends::OnnxBackend;

    println!("🔍 ONNX Runtime Execution Provider Diagnostics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // System information
    let cpu_count = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    println!("💻 System: {cpu_count} CPU cores detected");

    // Check provider availability
    let providers = OnnxBackend::list_providers();

    for (name, available, description) in providers {
        let status = if available {
            "✅ Available"
        } else {
            "❌ Not Available"
        };
        println!("🚀 {name}: {status} - {description}");
    }

    println!("\n💡 Tips:");
    println!("  • Use --execution-provider auto for best performance");
    println!("  • GPU acceleration requires compatible hardware/drivers");
    println!("  • CPU provider is always available as fallback");

    Ok(())
}

/// Parse hex color string to `BackgroundColor`
fn parse_color(color_str: &str) -> Result<BackgroundColor> {
    let hex = color_str.trim_start_matches('#');

    if hex.len() != 6 {
        return Err(anyhow::anyhow!("Color must be in #RRGGBB format"));
    }

    let r = u8::from_str_radix(&hex[0..2], 16).context("Invalid red component")?;
    let g = u8::from_str_radix(&hex[2..4], 16).context("Invalid green component")?;
    let b = u8::from_str_radix(&hex[4..6], 16).context("Invalid blue component")?;

    Ok(BackgroundColor::new(r, g, b))
}

/// Process multiple inputs efficiently with shared model loading
async fn process_inputs(
    cli: &Cli,
    config: &RemovalConfig,
    model_spec: &ModelSpec,
) -> Result<usize> {
    // Handle stdin specially (single input)
    if cli.input.len() == 1 && cli.input[0] == "-" {
        return process_stdin(&cli.output, config, model_spec).await;
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
            return Err(anyhow::anyhow!(
                "Input path does not exist or is not accessible: {}",
                path.display()
            ));
        }
    }

    if all_files.is_empty() {
        warn!("No image files found in the provided inputs");
        return Ok(0);
    }

    info!("Found {} image file(s) to process", all_files.len());

    // Create processor once with model pre-loaded for efficiency (massive speedup for batch processing)
    let model_manager = ModelManager::from_spec_with_provider(model_spec, Some(&config.execution_provider))
        .context("Failed to load model for batch processing")?;
    let mut processor = bg_remove_core::ImageProcessor::with_model_manager(config, model_manager)
        .context("Failed to create image processor for batch processing")?;

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

        match process_single_file_with_processor(&mut processor, &input_file, &output_path, config).await {
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
        log::debug!("Note: Individual file timing breakdowns shown above include model load (first file only), decode, preprocessing, inference, postprocessing, and encode phases.");
    }

    Ok(processed_count)
}

/// Process image from stdin
async fn process_stdin(
    output_target: &Option<String>,
    config: &RemovalConfig,
    model_spec: &ModelSpec,
) -> Result<usize> {
    info!("Reading image from stdin");

    let image_data = read_stdin()?;
    let start_time = Instant::now();

    // Use the existing API for consistency - we'll need to save to a temporary file and use remove_background_with_model
    // For now, use a simpler approach that matches the file-based API
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("stdin_input.tmp");
    std::fs::write(&temp_file, &image_data)?;

    let result = remove_background_with_model(&temp_file, config, model_spec)
        .await
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
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            info!("Image written to stdout");
        },
        Some(target) => {
            // Output to file
            let output_path = PathBuf::from(target);
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
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            info!("Image written to stdout");
        },
    }

    Ok(1)
}

/// Process a single image file with existing processor (for efficiency)
async fn process_single_file_with_processor(
    processor: &mut bg_remove_core::ImageProcessor,
    input_path: &Path,
    output_path: &Option<String>,
    config: &RemovalConfig,
) -> Result<usize> {
    let mut result = processor.remove_background(input_path)
        .await
        .context("Failed to remove background")?;

    // Show detailed timing breakdown at INFO level for single file processing
    let timings = result.timings();
    let breakdown = timings.breakdown_percentages();
    
    info!("📊 Processing breakdown for {}:", input_path.display());
    
    // Show model load timing if present (first file only)
    if timings.model_load_ms > 0 {
        info!("  ├─ Model Load: {}ms ({:.1}%)", timings.model_load_ms, breakdown.model_load_pct);
    }
    
    info!("  ├─ Image Decode: {}ms ({:.1}%)", timings.image_decode_ms, breakdown.decode_pct);
    info!("  ├─ Preprocessing: {}ms ({:.1}%)", timings.preprocessing_ms, breakdown.preprocessing_pct);
    info!("  ├─ Inference: {}ms ({:.1}%)", timings.inference_ms, breakdown.inference_pct);
    info!("  ├─ Postprocessing: {}ms ({:.1}%)", timings.postprocessing_ms, breakdown.postprocessing_pct);

    // Handle output
    match output_path {
        Some(target) if target == "-" => {
            // Output to stdout
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            
            // Show total without encoding (since we don't save to file)
            info!("  └─ Total: {}ms ({:.2}s) - output to stdout", result.timings().total_ms, result.timings().total_ms as f64 / 1000.0);
        },
        Some(target) => {
            // Output to specific file - use timed save for detailed logging
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
            
            // Show encoding timing if it was captured
            if let Some(encode_ms) = result.timings().image_encode_ms {
                let encode_breakdown = result.timings().breakdown_percentages();
                info!("  ├─ Image Encode: {}ms ({:.1}%)", encode_ms, encode_breakdown.encode_pct);
            }
            
            // Show total with final breakdown
            info!("  └─ Total: {}ms ({:.2}s)", result.timings().total_ms, result.timings().total_ms as f64 / 1000.0);
        },
        None => {
            // Generate default output filename - use timed save for detailed logging
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
            
            // Show encoding timing if it was captured
            if let Some(encode_ms) = result.timings().image_encode_ms {
                let encode_breakdown = result.timings().breakdown_percentages();
                info!("  ├─ Image Encode: {}ms ({:.1}%)", encode_ms, encode_breakdown.encode_pct);
            }
            
            // Show total with final breakdown
            info!("  └─ Total: {}ms ({:.2}s)", result.timings().total_ms, result.timings().total_ms as f64 / 1000.0);
        },
    }

    Ok(1)
}



/// Find image files in directory
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
fn generate_output_path(input_path: &Path, format: OutputFormat) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or_default();
    let dir = input_path.parent().unwrap_or(Path::new("."));

    let extension = match format {
        OutputFormat::Png => "png",
        OutputFormat::Jpeg => "jpg",
        OutputFormat::WebP => "webp",
        OutputFormat::Tiff => "tiff",
        OutputFormat::Rgba8 => "rgba8",
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
    fn test_parse_color() {
        let white = parse_color("#ffffff").unwrap();
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);

        let black = parse_color("#000000").unwrap();
        assert_eq!(black.r, 0);
        assert_eq!(black.g, 0);
        assert_eq!(black.b, 0);

        let red = parse_color("#ff0000").unwrap();
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);

        // Test without # prefix
        let blue = parse_color("0000ff").unwrap();
        assert_eq!(blue.r, 0);
        assert_eq!(blue.g, 0);
        assert_eq!(blue.b, 255);

        // Test invalid format
        assert!(parse_color("#fff").is_err());
        assert!(parse_color("#gggggg").is_err());
    }

    #[test]
    fn test_generate_output_path() {
        let input = Path::new("/path/to/image.jpg");

        let png_output = generate_output_path(input, OutputFormat::Png);
        assert_eq!(png_output, Path::new("/path/to/image_bg_removed.png"));

        let jpeg_output = generate_output_path(input, OutputFormat::Jpeg);
        assert_eq!(jpeg_output, Path::new("/path/to/image_bg_removed.jpg"));

        let tiff_output = generate_output_path(input, OutputFormat::Tiff);
        assert_eq!(tiff_output, Path::new("/path/to/image_bg_removed.tiff"));

        let rgba8_output = generate_output_path(input, OutputFormat::Rgba8);
        assert_eq!(rgba8_output, Path::new("/path/to/image_bg_removed.rgba8"));
    }

    #[test]
    fn test_is_image_file() {
        let extensions = ["jpg", "jpeg", "png", "webp"];

        assert!(is_image_file(Path::new("test.jpg"), &extensions));
        assert!(is_image_file(Path::new("test.PNG"), &extensions));
        assert!(!is_image_file(Path::new("test.txt"), &extensions));
        assert!(!is_image_file(Path::new("test"), &extensions));
    }

    #[test]
    fn test_parse_model_spec() {
        // Test embedded model without variant
        let spec = parse_model_spec("isnet");
        match spec.source {
            ModelSource::Embedded(name) => assert_eq!(name, "isnet"),
            _ => panic!("Expected embedded model"),
        }
        assert_eq!(spec.variant, None);

        // Test embedded model with variant suffix
        let spec = parse_model_spec("birefnet:fp32");
        match spec.source {
            ModelSource::Embedded(name) => assert_eq!(name, "birefnet"),
            _ => panic!("Expected embedded model"),
        }
        assert_eq!(spec.variant, Some("fp32".to_string()));

        // Test non-existent path (should be treated as embedded)
        let spec = parse_model_spec("/non/existent/path");
        match spec.source {
            ModelSource::Embedded(name) => assert_eq!(name, "/non/existent/path"),
            _ => panic!("Expected embedded model for non-existent path"),
        }
    }

    #[test]
    fn test_resolve_variant() {
        let available = vec!["fp16".to_string(), "fp32".to_string()];

        // Test CLI parameter precedence
        let spec = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: Some("fp32".to_string()),
        };
        let result = resolve_variant(&spec, Some("fp16"), &available).unwrap();
        assert_eq!(result, "fp16"); // CLI param wins

        // Test suffix precedence when no CLI param
        let result = resolve_variant(&spec, None, &available).unwrap();
        assert_eq!(result, "fp32"); // Suffix wins

        // Test auto-detection (prefers fp16)
        let spec_no_variant = ModelSpec {
            source: ModelSource::Embedded("test".to_string()),
            variant: None,
        };
        let result = resolve_variant(&spec_no_variant, None, &available).unwrap();
        assert_eq!(result, "fp16"); // Auto-detection prefers fp16

        // Test invalid variant error
        let result = resolve_variant(&spec, Some("invalid"), &available);
        assert!(result.is_err());
    }
}
