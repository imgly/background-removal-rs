//! Main test suite runner for background removal testing

use bg_remove_core::{process_image, ExecutionProvider, RemovalConfig};
use bg_remove_e2e::{
    ImageComparison, ReportGenerator, TestFixtures, TestResult, TestSession, TestingError,
    ValidationThresholds,
};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "test-suite")]
#[command(about = "Run the background removal test suite with real images")]
#[command(version = "1.0")]
struct Args {
    /// Test categories to run (comma-separated)
    #[arg(long, value_delimiter = ',')]
    categories: Option<Vec<TestCategory>>,

    /// Output directory for test results
    #[arg(long, default_value = "test_results")]
    output_dir: String,

    /// Generate HTML report after tests
    #[arg(long)]
    generate_report: bool,

    /// Path to background removal binary (optional, uses library directly if not provided)
    #[arg(long)]
    binary_path: Option<String>,

    /// Assets directory containing test images
    #[arg(long, default_value = "crates/bg-remove-testing/assets")]
    assets_dir: String,

    /// Execution provider to test (auto, cpu, cuda, coreml)
    #[arg(long, default_value = "auto")]
    provider: String,

    /// Validation thresholds (JSON file)
    #[arg(long)]
    thresholds_file: Option<String>,

    /// Number of iterations for each test (for performance measurement)
    #[arg(long, default_value = "1")]
    iterations: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, ValueEnum, PartialEq)]
enum TestCategory {
    Portraits,
    Products,
    Complex,
    EdgeCases,
    All,
}

impl TestCategory {
    fn to_string(&self) -> String {
        match self {
            TestCategory::Portraits => "portraits".to_string(),
            TestCategory::Products => "products".to_string(),
            TestCategory::Complex => "complex".to_string(),
            TestCategory::EdgeCases => "edge_cases".to_string(),
            TestCategory::All => "all".to_string(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🧪 Background Removal Test Suite");
    println!("================================");
    println!();

    // Initialize test environment
    let test_runner = TestRunner::new(args)?;

    // Run the test suite
    let session = test_runner.run().await?;

    // Print summary
    test_runner.print_summary(&session);

    Ok(())
}

/// Main test runner
struct TestRunner {
    args: Args,
    fixtures: TestFixtures,
    output_dir: PathBuf,
    thresholds: ValidationThresholds,
}

impl TestRunner {
    fn new(args: Args) -> Result<Self, Box<dyn std::error::Error>> {
        // Load test fixtures
        let fixtures = TestFixtures::new(&args.assets_dir)?;

        // Create output directory
        let output_dir = PathBuf::from(&args.output_dir);
        std::fs::create_dir_all(&output_dir)?;
        std::fs::create_dir_all(output_dir.join("outputs"))?;

        // Load validation thresholds
        let thresholds = if let Some(ref thresholds_file) = args.thresholds_file {
            let content = std::fs::read_to_string(thresholds_file)?;
            serde_json::from_str(&content)?
        } else {
            ValidationThresholds::default()
        };

        // Validate assets
        if args.verbose {
            println!("🔍 Validating test assets...");
            let validation_report = fixtures.validate_assets()?;
            if validation_report.is_valid() {
                println!("✅ All test assets validated");
            } else {
                eprintln!(
                    "⚠️  Found {} issues with test assets:",
                    validation_report.total_issues()
                );
                for missing in &validation_report.missing_input_files {
                    eprintln!("  Missing input: {}", missing.display());
                }
                for missing in &validation_report.missing_expected_files {
                    eprintln!("  Missing expected: {}", missing.display());
                }
            }
            println!();
        }

        Ok(Self {
            args,
            fixtures,
            output_dir,
            thresholds,
        })
    }

    async fn run(&self) -> Result<TestSession, Box<dyn std::error::Error>> {
        let mut session = TestSession::new();

        // Determine which test cases to run
        let test_cases = self.get_test_cases_to_run()?;

        if test_cases.is_empty() {
            println!("⚠️  No test cases found for specified categories");
            return Ok(session);
        }

        println!("📋 Running {} test cases", test_cases.len());
        println!("🎯 Provider: {}", self.args.provider);
        println!("📁 Output: {}", self.output_dir.display());
        println!();

        // Create progress bar
        let progress_count = test_cases
            .len()
            .try_into()
            .map_err(|_| anyhow::anyhow!("Too many test cases for progress bar (>u64::MAX)"))?;
        let progress = ProgressBar::new(progress_count);
        progress.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        // Run each test case
        for test_case in test_cases {
            progress.set_message(format!("Testing {}", test_case.id));

            let result = self.run_single_test(test_case).await?;

            if self.args.verbose {
                self.print_test_result(&result);
            }

            session.add_result(result);
            progress.inc(1);
        }

        progress.finish_with_message("All tests completed!");
        println!();

        // Finalize session
        session.finalize();

        // Save timing data for report generation
        let timing_data_path = self.output_dir.join("timing_data.json");
        let timing_data = serde_json::to_string_pretty(&session.results)?;
        std::fs::write(&timing_data_path, timing_data)?;

        if self.args.verbose {
            println!("📊 Saved timing data to: {}", timing_data_path.display());
        }

        // Generate report if requested
        if self.args.generate_report {
            println!("📊 Generating HTML report...");
            let reporter = ReportGenerator::new(&self.output_dir)?;
            let report_path = reporter.generate_comprehensive_report(&session)?;
            println!("✅ Report generated: {}", report_path.display());
            println!();
        }

        Ok(session)
    }

    fn get_test_cases_to_run(&self) -> Result<Vec<&bg_remove_e2e::TestCase>, TestingError> {
        let default_categories = vec![TestCategory::All];
        let categories = self.args.categories.as_ref().unwrap_or(&default_categories);

        if categories.contains(&TestCategory::All) {
            Ok(self.fixtures.get_test_cases().iter().collect())
        } else {
            let mut test_cases = Vec::new();
            for category in categories {
                let category_str = category.to_string();
                let category_cases = self.fixtures.get_test_cases_for_category(&category_str);
                test_cases.extend(category_cases);
            }
            Ok(test_cases)
        }
    }

    async fn run_single_test(
        &self,
        test_case: &bg_remove_e2e::TestCase,
    ) -> Result<TestResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Load input image
        let input_image = self.fixtures.load_input_image(test_case)?;
        let expected_image = self.fixtures.load_expected_image(test_case)?;

        // Parse execution provider
        let provider = match self.args.provider.to_lowercase().as_str() {
            "auto" => ExecutionProvider::Auto,
            "cpu" => ExecutionProvider::Cpu,
            "cuda" => ExecutionProvider::Cuda,
            "coreml" => ExecutionProvider::CoreMl,
            _ => ExecutionProvider::Auto,
        };

        // Create removal configuration
        let config = RemovalConfig::builder()
            .execution_provider(provider)
            .build()?;

        // Run background removal (with iterations for performance measurement)
        let mut processing_times = Vec::new();
        let mut removal_result = None;

        for _ in 0..self.args.iterations {
            let iter_start = Instant::now();
            let result = process_image(input_image.clone(), &config)?;
            let iter_time = iter_start.elapsed();
            processing_times.push(iter_time);

            if removal_result.is_none() {
                removal_result = Some(result);
            }
        }

        let removal_result = removal_result.unwrap();
        let result_dynamic = removal_result.image.clone();

        // Save output image
        let output_filename = format!("{}_{}.png", test_case.category, test_case.id);
        let output_path = self.output_dir.join("outputs").join(&output_filename);
        removal_result.save_png(&output_path)?;

        // Calculate metrics
        let metrics = ImageComparison::calculate_metrics(
            &result_dynamic,
            &expected_image,
            0.05, // 5% pixel threshold
        )?;

        // Determine if test passed
        let passed = metrics.visual_quality_score >= test_case.expected_accuracy
            && metrics.ssim >= self.thresholds.ssim
            && metrics.edge_accuracy >= self.thresholds.edge_accuracy;

        // Calculate average processing time
        let processing_times_count: u32 = processing_times
            .len()
            .try_into()
            .map_err(|_| anyhow::anyhow!("Too many processing time samples for u32 division"))?;
        let avg_processing_time =
            processing_times.iter().sum::<std::time::Duration>() / processing_times_count;

        let _total_time = start_time.elapsed();

        Ok(TestResult {
            test_case: test_case.clone(),
            passed,
            metrics,
            processing_time: avg_processing_time,
            error_message: None,
            output_path: Some(output_path),
        })
    }

    fn print_test_result(&self, result: &TestResult) {
        let status = if result.passed { "✅" } else { "❌" };
        let accuracy = result.metrics.pixel_accuracy * 100.0;
        let time = result.processing_time.as_millis();

        println!(
            "  {} {} - {:.1}% accuracy, {}ms",
            status, result.test_case.id, accuracy, time
        );

        if !result.passed {
            println!(
                "    Expected: {:.1}%, Got: {:.1}%",
                result.test_case.expected_accuracy * 100.0,
                accuracy
            );
        }
    }

    fn print_summary(&self, session: &TestSession) {
        println!("📈 Test Results Summary");
        println!("======================");
        println!();

        let summary = &session.summary;
        let pass_rate = if summary.total_tests > 0 {
            (summary.passed_tests as f64 / summary.total_tests as f64) * 100.0
        } else {
            0.0
        };

        println!("Total Tests:     {}", summary.total_tests);
        println!(
            "Passed:          {} ({:.1}%)",
            summary.passed_tests, pass_rate
        );
        println!("Failed:          {}", summary.failed_tests);
        println!("Average Accuracy: {:.1}%", summary.average_accuracy * 100.0);
        println!(
            "Average Time:    {}ms",
            summary.average_processing_time.as_millis()
        );
        println!();

        // Category breakdown
        println!("📊 By Category:");
        for category in &summary.categories_tested {
            let category_results: Vec<_> = session
                .results
                .iter()
                .filter(|r| r.test_case.category == *category)
                .collect();

            let category_passed = category_results.iter().filter(|r| r.passed).count();
            let category_total = category_results.len();
            let category_rate = (category_passed as f64 / category_total as f64) * 100.0;

            println!("  {category}: {category_passed}/{category_total} ({category_rate:.1}%)");
        }
        println!();

        // Overall status
        if pass_rate >= 90.0 {
            println!("🎉 Excellent! Test suite passed with high accuracy");
        } else if pass_rate >= 75.0 {
            println!("✅ Good! Test suite passed with acceptable accuracy");
        } else if pass_rate >= 50.0 {
            println!("⚠️  Warning: Test suite passed but with low accuracy");
        } else {
            println!("❌ Test suite failed - significant accuracy issues detected");
        }

        println!("📄 Output directory: {}", self.output_dir.display());
        if self.args.generate_report {
            println!(
                "🌐 View detailed report: {}/comparison_report.html",
                self.output_dir.display()
            );
        }
    }
}
