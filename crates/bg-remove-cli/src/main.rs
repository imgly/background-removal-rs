//! Background Removal CLI Tool
//!
//! Command-line interface for removing backgrounds from images using `ISNet` models.

use anyhow::{Context, Result};
use bg_remove_core::config::BackgroundColor;
use bg_remove_core::{ExecutionProvider, OutputFormat, RemovalConfig};
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
    /// Input image file or directory (use "-" for stdin)
    #[arg(value_name = "INPUT", required_unless_present = "show_providers")]
    input: Option<String>,

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

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Process directory recursively
    #[arg(short, long)]
    recursive: bool,

    /// Pattern for batch processing (e.g., "*.jpg")
    #[arg(long)]
    pattern: Option<String>,

    /// Show execution provider diagnostics and exit
    #[arg(long)]
    show_providers: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CliOutputFormat {
    Png,
    Jpeg,
    Webp,
    Rgba8,
}

impl From<CliOutputFormat> for OutputFormat {
    fn from(cli_format: CliOutputFormat) -> Self {
        match cli_format {
            CliOutputFormat::Png => OutputFormat::Png,
            CliOutputFormat::Jpeg => OutputFormat::Jpeg,
            CliOutputFormat::Webp => OutputFormat::WebP,
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose);

    // Handle provider diagnostics flag
    if cli.show_providers {
        return show_provider_diagnostics();
    }

    let input = cli
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Input is required"))?;

    info!("Starting background removal CLI");
    info!("Input: {input}");

    // Parse background color
    let background_color =
        parse_color(&cli.background_color).context("Invalid background color format")?;

    // Build configuration
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
        .build()
        .context("Invalid configuration")?;

    // Process input
    let start_time = Instant::now();
    let processed_count = if input == "-" {
        // Read from stdin
        process_stdin(&cli.output, &config).await?
    } else {
        let input_path = PathBuf::from(input);
        if input_path.is_file() {
            process_single_file(&input_path, &cli.output, &config).await?
        } else if input_path.is_dir() {
            process_directory(&cli, &config).await?
        } else {
            return Err(anyhow::anyhow!(
                "Input path does not exist or is not accessible"
            ));
        }
    };

    let total_time = start_time.elapsed();
    info!(
        "Processed {} image(s) in {:.2}s",
        processed_count,
        total_time.as_secs_f64()
    );

    Ok(())
}

/// Initialize logging based on verbosity level
fn init_logging(verbose: bool) {
    let log_level = if verbose { "debug" } else { "info" };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .init();
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
    use bg_remove_core::inference::check_provider_availability;

    println!("🔍 ONNX Runtime Execution Provider Diagnostics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // System information
    let cpu_count = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    println!("💻 System: {cpu_count} CPU cores detected");

    // Check provider availability
    let providers = check_provider_availability();

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

/// Process image from stdin
async fn process_stdin(output_target: &Option<String>, config: &RemovalConfig) -> Result<usize> {
    info!("Reading image from stdin");

    let image_data = read_stdin()?;
    let start_time = Instant::now();

    // Load image from bytes
    let image =
        image::load_from_memory(&image_data).context("Failed to decode image from stdin")?;

    // Process image through bg-remove-core
    let result =
        bg_remove_core::process_image(image, config).context("Failed to remove background")?;

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
            result
                .save(&output_path, config.output_format, config.jpeg_quality)
                .context("Failed to save result")?;
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

/// Process a single image file
async fn process_single_file(
    input_path: &Path,
    output_path: &Option<String>,
    config: &RemovalConfig,
) -> Result<usize> {
    let start_time = Instant::now();
    let mut result = bg_remove_core::remove_background(input_path, config)
        .await
        .context("Failed to remove background")?;

    let processing_time = start_time.elapsed();

    // Handle output
    match output_path {
        Some(target) if target == "-" => {
            // Output to stdout
            let output_data = result.to_bytes(config.output_format, config.jpeg_quality)?;
            write_stdout(&output_data)?;
            info!(
                "Processed {} and wrote to stdout in {:.2}s",
                input_path.display(),
                processing_time.as_secs_f64()
            );
        },
        Some(target) => {
            // Output to specific file - use timed save for detailed logging
            let output_path = PathBuf::from(target);
            match config.output_format {
                OutputFormat::Png => {
                    result
                        .save_png_timed(&output_path)
                        .context("Failed to save result")?;
                },
                _ => {
                    result
                        .save(&output_path, config.output_format, config.jpeg_quality)
                        .context("Failed to save result")?;
                },
            }
        },
        None => {
            // Generate default output filename - use timed save for detailed logging
            let output_path = generate_output_path(input_path, config.output_format);
            match config.output_format {
                OutputFormat::Png => {
                    result
                        .save_png_timed(&output_path)
                        .context("Failed to save result")?;
                },
                _ => {
                    result
                        .save(&output_path, config.output_format, config.jpeg_quality)
                        .context("Failed to save result")?;
                },
            }
        },
    }

    Ok(1)
}

/// Process all images in a directory
async fn process_directory(cli: &Cli, config: &RemovalConfig) -> Result<usize> {
    let input = cli
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Input directory required"))?;
    let input_dir = PathBuf::from(input);
    let output_dir = cli
        .output
        .as_ref().map_or_else(|| input_dir.clone(), PathBuf::from);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir).context("Failed to create output directory")?;

    // Find image files
    let image_files = find_image_files(&input_dir, cli.recursive, cli.pattern.as_deref())?;

    if image_files.is_empty() {
        warn!("No image files found in directory");
        return Ok(0);
    }

    info!("Found {} image files to process", image_files.len());

    // Create progress bar
    let progress = ProgressBar::new(image_files.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut processed_count = 0;
    let mut failed_count = 0;

    for input_file in image_files {
        let relative_path = input_file.strip_prefix(&input_dir).unwrap_or(&input_file);
        let output_file = output_dir.join(relative_path);

        // Ensure output directory exists
        if let Some(parent) = output_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Generate output filename with correct extension
        let output_file = generate_output_path(&output_file, config.output_format);

        progress.set_message(format!("Processing {}", input_file.display()));

        match bg_remove_core::remove_background(&input_file, config).await {
            Ok(mut result) => {
                let save_result = match config.output_format {
                    OutputFormat::Png => result.save_png_timed(&output_file),
                    _ => result.save(&output_file, config.output_format, config.jpeg_quality),
                };

                match save_result {
                    Ok(()) => {
                        processed_count += 1;
                        if cli.verbose {
                            info!("Processed: {}", input_file.display());
                        }
                    },
                    Err(e) => {
                        error!("Failed to save {}: {}", output_file.display(), e);
                        failed_count += 1;
                    },
                }
            },
            Err(e) => {
                error!("Failed to process {}: {}", input_file.display(), e);
                failed_count += 1;
            },
        }

        progress.inc(1);
    }

    progress.finish_with_message(format!(
        "Completed! Processed: {processed_count}, Failed: {failed_count}"
    ));

    Ok(processed_count)
}

/// Find image files in directory
fn find_image_files(dir: &Path, recursive: bool, pattern: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let image_extensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

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
}
