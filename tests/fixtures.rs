//! Real image fixtures and loading utilities

// Import from the common test utilities
mod common;
use common::{ComplexityLevel, Result, TestCase, TestingError};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Manager for real test image assets
pub struct TestFixtures {
    assets_dir: PathBuf,
    test_cases: Vec<TestCase>,
}

impl TestFixtures {
    /// Create a new `TestFixtures` instance
    pub fn new<P: AsRef<Path>>(assets_dir: P) -> Result<Self> {
        let assets_dir = assets_dir.as_ref().to_path_buf();

        if !assets_dir.exists() {
            return Err(TestingError::InvalidConfiguration(format!(
                "Assets directory does not exist: {}",
                assets_dir.display()
            )));
        }

        let test_cases = Self::load_test_cases(&assets_dir)?;

        Ok(Self {
            assets_dir,
            test_cases,
        })
    }

    /// Load test case definitions from JSON metadata
    fn load_test_cases(assets_dir: &Path) -> Result<Vec<TestCase>> {
        let test_cases_file = assets_dir.join("test_cases.json");

        if test_cases_file.exists() {
            let content = std::fs::read_to_string(&test_cases_file)?;
            let test_cases: Vec<TestCase> = serde_json::from_str(&content)?;
            Ok(test_cases)
        } else {
            // Generate test cases from directory structure if no metadata file exists
            Self::generate_test_cases_from_directory(assets_dir)
        }
    }

    /// Generate test cases by scanning directory structure
    fn generate_test_cases_from_directory(assets_dir: &Path) -> Result<Vec<TestCase>> {
        let mut test_cases = Vec::new();
        let input_dir = assets_dir.join("input");
        let expected_dir = assets_dir.join("expected");

        if !input_dir.exists() || !expected_dir.exists() {
            return Err(TestingError::InvalidConfiguration(
                "Missing input/ or expected/ directories in assets".to_string(),
            ));
        }

        // Scan each category directory
        for category_entry in std::fs::read_dir(&input_dir)? {
            let category_entry = category_entry?;
            if !category_entry.file_type()?.is_dir() {
                continue;
            }

            let category_name = category_entry.file_name().to_string_lossy().to_string();
            let category_path = category_entry.path();

            // Find all images in this category
            for image_entry in WalkDir::new(&category_path)
                .min_depth(1)
                .max_depth(1)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
            {
                let input_file = image_entry.file_name().to_string_lossy().to_string();

                if !Self::is_image_file(&input_file) {
                    continue;
                }

                // Look for corresponding expected output
                let expected_file =
                    Self::find_expected_output(&expected_dir, &category_name, &input_file)?;

                let test_case = TestCase {
                    id: format!("{}_{}", category_name, Self::get_file_stem(&input_file)),
                    category: category_name.clone(),
                    input_file: format!("{category_name}/{input_file}"),
                    expected_output_file: expected_file,
                    expected_accuracy: Self::default_accuracy_for_category(&category_name),
                    description: format!("Test {category_name} image: {input_file}"),
                    tags: vec![category_name.clone()],
                    complexity_level: Self::infer_complexity(&category_name),
                };

                test_cases.push(test_case);
            }
        }

        Ok(test_cases)
    }

    /// Find the corresponding expected output file
    fn find_expected_output(
        expected_dir: &Path,
        category: &str,
        input_file: &str,
    ) -> Result<String> {
        let stem = Self::get_file_stem(input_file);
        let category_dir = expected_dir.join(category);

        // Try different possible output formats
        let possible_extensions = ["png", "jpg", "jpeg"];

        for ext in &possible_extensions {
            let expected_file = format!("{stem}.{ext}");
            let expected_path = category_dir.join(&expected_file);

            if expected_path.exists() {
                return Ok(format!("{category}/{expected_file}"));
            }
        }

        Err(TestingError::ReferenceImageNotFound(format!(
            "No expected output found for {input_file} in category {category}"
        )))
    }

    /// Check if file is a supported image format
    fn is_image_file(filename: &str) -> bool {
        let ext = filename.split('.').next_back().unwrap_or("").to_lowercase();
        matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp")
    }

    /// Get file stem (filename without extension)
    fn get_file_stem(filename: &str) -> String {
        Path::new(filename)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    }

    /// Infer complexity level from category name
    fn infer_complexity(category: &str) -> ComplexityLevel {
        match category {
            "portraits" => ComplexityLevel::Medium,
            "products" => ComplexityLevel::Simple,
            "complex" => ComplexityLevel::Complex,
            "edge_cases" => ComplexityLevel::Extreme,
            _ => ComplexityLevel::Medium,
        }
    }

    /// Default accuracy threshold for category
    fn default_accuracy_for_category(category: &str) -> f64 {
        match category {
            "portraits" => 0.92,
            "products" => 0.97,
            "complex" => 0.85,
            "edge_cases" => 0.75,
            _ => 0.90,
        }
    }

    /// Get all test cases
    #[must_use]
    pub fn get_test_cases(&self) -> &[TestCase] {
        &self.test_cases
    }

    /// Get test cases for a specific category
    #[must_use]
    pub fn get_test_cases_for_category(&self, category: &str) -> Vec<&TestCase> {
        self.test_cases
            .iter()
            .filter(|tc| tc.category == category)
            .collect()
    }

    /// Load an input image
    pub fn load_input_image(&self, test_case: &TestCase) -> Result<DynamicImage> {
        let image_path = self.assets_dir.join("input").join(&test_case.input_file);
        let image = image::open(&image_path)?;
        Ok(image)
    }

    /// Load the expected output image
    pub fn load_expected_image(&self, test_case: &TestCase) -> Result<DynamicImage> {
        let image_path = self
            .assets_dir
            .join("expected")
            .join(&test_case.expected_output_file);
        let image = image::open(&image_path)?;
        Ok(image)
    }

    /// Get input image path
    #[must_use]
    pub fn get_input_path(&self, test_case: &TestCase) -> PathBuf {
        self.assets_dir.join("input").join(&test_case.input_file)
    }

    /// Get expected output image path
    #[must_use]
    pub fn get_expected_path(&self, test_case: &TestCase) -> PathBuf {
        self.assets_dir
            .join("expected")
            .join(&test_case.expected_output_file)
    }

    /// Get all available categories
    #[must_use]
    pub fn get_categories(&self) -> Vec<String> {
        let mut categories: Vec<String> = self
            .test_cases
            .iter()
            .map(|tc| tc.category.clone())
            .collect();
        categories.sort();
        categories.dedup();
        categories
    }

    /// Get test case by ID
    #[must_use]
    pub fn get_test_case(&self, id: &str) -> Option<&TestCase> {
        self.test_cases.iter().find(|tc| tc.id == id)
    }

    /// Validate that all referenced files exist
    pub fn validate_assets(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        for test_case in &self.test_cases {
            let input_path = self.get_input_path(test_case);
            let expected_path = self.get_expected_path(test_case);

            if !input_path.exists() {
                report.missing_input_files.push(input_path.clone());
            }

            if !expected_path.exists() {
                report.missing_expected_files.push(expected_path.clone());
            }

            // Try to load images to verify they're valid
            if input_path.exists() {
                if let Err(e) = image::open(&input_path) {
                    report.invalid_input_files.push((input_path, e.to_string()));
                }
            }

            if expected_path.exists() {
                if let Err(e) = image::open(&expected_path) {
                    report
                        .invalid_expected_files
                        .push((expected_path, e.to_string()));
                }
            }
        }

        Ok(report)
    }

    /// Get summary of available test data
    #[must_use]
    pub fn get_summary(&self) -> TestDataSummary {
        let categories = self.get_categories();
        let mut category_counts = HashMap::new();

        for category in &categories {
            let count = self.get_test_cases_for_category(category).len();
            category_counts.insert(category.clone(), count);
        }

        TestDataSummary {
            total_test_cases: self.test_cases.len(),
            categories: categories.clone(),
            category_counts,
            assets_directory: self.assets_dir.clone(),
        }
    }
}

/// Validation report for test assets
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub missing_input_files: Vec<PathBuf>,
    pub missing_expected_files: Vec<PathBuf>,
    pub invalid_input_files: Vec<(PathBuf, String)>,
    pub invalid_expected_files: Vec<(PathBuf, String)>,
}

impl ValidationReport {
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.missing_input_files.is_empty()
            && self.missing_expected_files.is_empty()
            && self.invalid_input_files.is_empty()
            && self.invalid_expected_files.is_empty()
    }

    #[must_use]
    pub fn total_issues(&self) -> usize {
        self.missing_input_files.len()
            + self.missing_expected_files.len()
            + self.invalid_input_files.len()
            + self.invalid_expected_files.len()
    }
}

/// Summary of test data availability
#[derive(Debug)]
pub struct TestDataSummary {
    pub total_test_cases: usize,
    pub categories: Vec<String>,
    pub category_counts: HashMap<String, usize>,
    pub assets_directory: PathBuf,
}
