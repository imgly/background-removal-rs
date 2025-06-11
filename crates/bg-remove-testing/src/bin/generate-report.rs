//! Generate HTML reports from test results

use bg_remove_testing::{ReportGenerator, TestCase, TestMetrics, TestResult, TestSession};
use clap::Parser;
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "generate-report")]
#[command(about = "Generate HTML comparison report from test outputs")]
#[command(version = "1.0")]
struct Args {
    /// Output directory for the generated report
    #[arg(long, default_value = "test_results")]
    output_dir: String,

    /// Directory containing Rust outputs
    #[arg(long, default_value = "test_results/outputs")]
    rust_outputs: String,

    /// Directory containing expected outputs
    #[arg(long, default_value = "crates/bg-remove-testing/assets/expected")]
    expected_outputs: String,

    /// Directory containing original input images
    #[arg(long, default_value = "crates/bg-remove-testing/assets/input")]
    input_dir: String,

    /// Test cases metadata file
    #[arg(
        long,
        default_value = "crates/bg-remove-testing/assets/test_cases.json"
    )]
    test_cases_file: String,

    /// Report title
    #[arg(long, default_value = "Background Removal Test Report")]
    title: String,

    /// Generate comparison images (side-by-side original, expected, actual)
    #[arg(long)]
    generate_comparisons: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("📊 Generating HTML Report");
    println!("========================");
    println!();

    let report_generator = ReportBuilder::new(args)?;
    let _session = report_generator.build_session_from_outputs().await?;

    println!("✅ Report generated successfully!");
    println!(
        "📄 Open: {}/comparison_report.html",
        report_generator.output_dir.display()
    );

    Ok(())
}

/// Report builder that reconstructs test session from output files
struct ReportBuilder {
    args: Args,
    output_dir: PathBuf,
    test_cases: Vec<TestCase>,
}

impl ReportBuilder {
    fn new(args: Args) -> Result<Self, Box<dyn std::error::Error>> {
        let output_dir = PathBuf::from(&args.output_dir);
        std::fs::create_dir_all(&output_dir)?;

        // Load test cases metadata
        let test_cases_content = std::fs::read_to_string(&args.test_cases_file)?;
        let test_cases: Vec<TestCase> = serde_json::from_str(&test_cases_content)?;

        println!("📋 Loaded {} test case definitions", test_cases.len());

        Ok(Self {
            args,
            output_dir,
            test_cases,
        })
    }

    async fn build_session_from_outputs(&self) -> Result<TestSession, Box<dyn std::error::Error>> {
        let mut session = TestSession::new();

        println!("🔍 Scanning output directories...");

        // Find all output files
        let rust_outputs = self.find_output_files(&self.args.rust_outputs)?;
        println!("Found {} Rust outputs", rust_outputs.len());

        // Process each output file
        for (test_id, rust_output_path) in rust_outputs {
            if let Some(test_case) = self.find_test_case(&test_id) {
                println!("Processing: {}", test_case.id);

                let result = self
                    .process_test_output(test_case, &rust_output_path)
                    .await?;
                session.add_result(result);
            } else {
                eprintln!("⚠️  No test case found for output: {}", test_id);
            }
        }

        session.finalize();

        // Generate the HTML report
        println!("📝 Generating HTML report...");
        let generator = ReportGenerator::new(&self.output_dir)?;
        generator.generate_comprehensive_report(&session)?;

        // Generate comparison images if requested
        if self.args.generate_comparisons {
            println!("🖼️  Generating comparison images...");
            self.generate_comparison_images(&session).await?;
        }

        Ok(session)
    }

    fn find_output_files(
        &self,
        output_dir: &str,
    ) -> Result<Vec<(String, PathBuf)>, Box<dyn std::error::Error>> {
        let mut outputs = Vec::new();
        let output_path = PathBuf::from(output_dir);

        if !output_path.exists() {
            return Err(format!("Output directory does not exist: {}", output_dir).into());
        }

        for entry in WalkDir::new(&output_path)
            .min_depth(1)
            .max_depth(2)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if let Some(extension) = path.extension() {
                if extension == "png" || extension == "jpg" || extension == "jpeg" {
                    if let Some(file_stem) = path.file_stem() {
                        let test_id =
                            self.extract_test_id_from_filename(&file_stem.to_string_lossy());
                        outputs.push((test_id, path.to_path_buf()));
                    }
                }
            }
        }

        Ok(outputs)
    }

    fn extract_test_id_from_filename(&self, filename: &str) -> String {
        // Handle different naming patterns:
        // category_test_id.png -> test_id
        // test_id.png -> test_id
        // category_test_id_alpha.png -> test_id

        let without_alpha = filename.trim_end_matches("_alpha");

        // Try to match against known test case IDs
        for test_case in &self.test_cases {
            if without_alpha.contains(&test_case.id) {
                return test_case.id.clone();
            }

            // Try category_id pattern
            let pattern = format!("{}_{}", test_case.category, test_case.id);
            if without_alpha == pattern {
                return test_case.id.clone();
            }
        }

        // Fallback: return the filename as is
        without_alpha.to_string()
    }

    fn find_test_case(&self, test_id: &str) -> Option<&TestCase> {
        self.test_cases.iter().find(|tc| tc.id == test_id)
    }

    async fn process_test_output(
        &self,
        test_case: &TestCase,
        rust_output_path: &PathBuf,
    ) -> Result<TestResult, Box<dyn std::error::Error>> {
        use bg_remove_testing::ImageComparison;

        // Load images
        let rust_output = image::open(rust_output_path)?;

        let expected_path =
            PathBuf::from(&self.args.expected_outputs).join(&test_case.expected_output_file);

        let _input_path = PathBuf::from(&self.args.input_dir).join(&test_case.input_file);

        // Check if expected output exists
        let (metrics, passed) = if expected_path.exists() {
            let expected_output = image::open(&expected_path)?;
            let metrics = ImageComparison::calculate_metrics(&rust_output, &expected_output, 0.05)?;
            let passed = metrics.pixel_accuracy >= test_case.expected_accuracy;
            (metrics, passed)
        } else {
            eprintln!("⚠️  Expected output not found: {}", expected_path.display());
            // Create default metrics for missing expected output
            let metrics = TestMetrics {
                pixel_accuracy: 0.0,
                ssim: 0.0,
                edge_accuracy: 0.0,
                visual_quality_score: 0.0,
                mean_squared_error: f64::MAX,
            };
            (metrics, false)
        };

        Ok(TestResult {
            test_case: test_case.clone(),
            passed,
            metrics,
            processing_time: std::time::Duration::from_millis(0), // Unknown from file-based analysis
            error_message: None,
            output_path: Some(rust_output_path.clone()),
        })
    }

    async fn generate_comparison_images(
        &self,
        session: &TestSession,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use bg_remove_testing::ImageComparison;

        let comparison_dir = self.output_dir.join("comparisons");
        std::fs::create_dir_all(&comparison_dir)?;

        for result in &session.results {
            let input_path = PathBuf::from(&self.args.input_dir).join(&result.test_case.input_file);
            let expected_path = PathBuf::from(&self.args.expected_outputs)
                .join(&result.test_case.expected_output_file);

            if let Some(ref rust_output_path) = result.output_path {
                if input_path.exists() && expected_path.exists() && rust_output_path.exists() {
                    let original = image::open(&input_path)?;
                    let expected = image::open(&expected_path)?;
                    let actual = image::open(rust_output_path)?;

                    // Generate side-by-side comparison
                    let comparison =
                        ImageComparison::create_comparison_image(&original, &expected, &actual)?;
                    let comparison_file =
                        comparison_dir.join(format!("{}_comparison.png", result.test_case.id));
                    comparison.save(&comparison_file)?;

                    // Generate difference image
                    let diff = ImageComparison::generate_diff_image(&actual, &expected)?;
                    let diff_file =
                        comparison_dir.join(format!("{}_diff.png", result.test_case.id));
                    diff.save(&diff_file)?;
                }
            }
        }

        println!(
            "✅ Comparison images saved to: {}",
            comparison_dir.display()
        );
        Ok(())
    }
}
