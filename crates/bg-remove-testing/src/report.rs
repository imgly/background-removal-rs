//! HTML report generation with image comparison tables

use crate::{Result, TestResult, TestSession, TestingError};
use std::io::Write;
use std::path::{Path, PathBuf};

/// HTML report generator for test results
pub struct ReportGenerator {
    output_dir: PathBuf,
    template_dir: Option<PathBuf>,
}

impl ReportGenerator {
    /// Create a new report generator
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir)?;
        std::fs::create_dir_all(output_dir.join("images"))?;

        Ok(Self {
            output_dir,
            template_dir: None,
        })
    }

    /// Set custom template directory
    pub fn with_template_dir<P: AsRef<Path>>(mut self, template_dir: P) -> Self {
        self.template_dir = Some(template_dir.as_ref().to_path_buf());
        self
    }

    /// Generate comprehensive HTML report with image comparison table
    pub fn generate_comprehensive_report(&self, session: &TestSession) -> Result<PathBuf> {
        let report_path = self.output_dir.join("comparison_report.html");
        let mut file = std::fs::File::create(&report_path)?;

        // Write HTML document
        writeln!(file, "{}", self.generate_html_header(session))?;
        writeln!(file, "{}", self.generate_summary_section(session))?;
        writeln!(file, "{}", self.generate_image_comparison_table(session)?)?;
        writeln!(file, "{}", self.generate_metrics_section(session))?;
        writeln!(file, "{}", self.generate_html_footer())?;

        Ok(report_path)
    }

    /// Generate HTML header with CSS styles
    fn generate_html_header(&self, session: &TestSession) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Background Removal Test Report - {}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header .session-info {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 0;
        }}
        
        .value.success {{ color: #4CAF50; }}
        .value.warning {{ color: #FF9800; }}
        .value.error {{ color: #f44336; }}
        
        .comparison-table {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            margin-bottom: 30px;
        }}
        
        .table-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .table-header h2 {{
            margin: 0;
            color: #495057;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .image-cell {{
            text-align: center;
            padding: 10px;
        }}
        
        .test-image {{
            max-width: 150px;
            max-height: 150px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }}
        
        .test-image:hover {{
            transform: scale(1.05);
        }}
        
        .status-passed {{
            color: #4CAF50;
            font-weight: bold;
        }}
        
        .status-failed {{
            color: #f44336;
            font-weight: bold;
        }}
        
        .metrics {{
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.85em;
        }}
        
        .metric-good {{ color: #4CAF50; }}
        .metric-warning {{ color: #FF9800; }}
        .metric-poor {{ color: #f44336; }}
        
        .category-section {{
            margin-bottom: 40px;
        }}
        
        .category-header {{
            background: #e9ecef;
            padding: 15px 20px;
            margin: 0;
            color: #495057;
            font-size: 1.1em;
            font-weight: 600;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        
        .modal-content {{
            margin: auto;
            display: block;
            width: 80%;
            max-width: 900px;
            margin-top: 50px;
        }}
        
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        
        .footer {{
            text-align: center;
            padding: 40px 20px;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            margin-top: 40px;
        }}
        
        @media (max-width: 768px) {{
            .test-image {{
                max-width: 100px;
                max-height: 100px;
            }}
            
            .summary {{
                grid-template-columns: 1fr;
            }}
            
            table {{
                font-size: 0.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Background Removal Test Report</h1>
        <div class="session-info">
            Session ID: {}<br>
            Generated: {}
        </div>
    </div>"#,
            session.session_id,
            session.session_id,
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        )
    }

    /// Generate summary statistics section
    fn generate_summary_section(&self, session: &TestSession) -> String {
        let pass_rate = if session.summary.total_tests > 0 {
            (session.summary.passed_tests as f64 / session.summary.total_tests as f64) * 100.0
        } else {
            0.0
        };

        let avg_accuracy = session.summary.average_accuracy * 100.0;
        let avg_time = session.summary.average_processing_time.as_millis();

        let pass_rate_class = if pass_rate >= 90.0 {
            "success"
        } else if pass_rate >= 75.0 {
            "warning"
        } else {
            "error"
        };

        let accuracy_class = if avg_accuracy >= 90.0 {
            "success"
        } else if avg_accuracy >= 75.0 {
            "warning"
        } else {
            "error"
        };

        format!(
            r#"    <div class="summary">
        <div class="summary-card">
            <h3>Total Tests</h3>
            <p class="value">{}</p>
        </div>
        <div class="summary-card">
            <h3>Pass Rate</h3>
            <p class="value {}">{:.1}%</p>
        </div>
        <div class="summary-card">
            <h3>Average Accuracy</h3>
            <p class="value {}">{:.1}%</p>
        </div>
        <div class="summary-card">
            <h3>Average Time</h3>
            <p class="value">{}ms</p>
        </div>
        <div class="summary-card">
            <h3>Categories</h3>
            <p class="value">{}</p>
        </div>
    </div>"#,
            session.summary.total_tests,
            pass_rate_class,
            pass_rate,
            accuracy_class,
            avg_accuracy,
            avg_time,
            session.summary.categories_tested.len()
        )
    }

    /// Generate the main image comparison table
    fn generate_image_comparison_table(&self, session: &TestSession) -> Result<String> {
        let mut html = String::new();

        // Group results by category
        let mut categories: std::collections::BTreeMap<String, Vec<&TestResult>> =
            std::collections::BTreeMap::new();
        for result in &session.results {
            categories
                .entry(result.test_case.category.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (category, results) in categories {
            html.push_str(&format!(
                r#"    <div class="category-section">
        <div class="comparison-table">
            <div class="table-header">
                <h2>📂 {} ({} tests)</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Original Image</th>
                        <th>Expected Output</th>
                        <th>Current Output</th>
                        <th>Diff Heatmap</th>
                        <th>Status</th>
                        <th>Metrics</th>
                    </tr>
                </thead>
                <tbody>"#,
                category.to_uppercase(),
                results.len()
            ));

            for result in results {
                html.push_str(&self.generate_test_row(result)?);
            }

            html.push_str(
                "                </tbody>\n            </table>\n        </div>\n    </div>",
            );
        }

        Ok(html)
    }

    /// Generate a single test result row
    fn generate_test_row(&self, result: &TestResult) -> Result<String> {
        let status_class = if result.passed {
            "status-passed"
        } else {
            "status-failed"
        };
        let status_text = if result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        };

        // Resolve full paths for input and expected images
        let assets_dir = Path::new("crates/bg-remove-testing/assets");
        let input_path = assets_dir.join("input").join(&result.test_case.input_file);
        let expected_path = assets_dir
            .join("expected")
            .join(&result.test_case.expected_output_file);

        // Copy images to output directory and get relative paths
        let original_image_path =
            self.copy_image_to_output(&input_path.to_string_lossy(), "original")?;
        let expected_image_path =
            self.copy_image_to_output(&expected_path.to_string_lossy(), "expected")?;
        let current_image_path = if let Some(ref output_path) = result.output_path {
            self.copy_image_to_output(&output_path.to_string_lossy(), "current")?
        } else {
            "images/placeholder.png".to_string()
        };

        // Generate diff heatmap
        let diff_heatmap_path = if let Some(ref output_path) = result.output_path {
            if expected_path.exists() && output_path.exists() {
                self.generate_diff_heatmap(&expected_path, output_path, &result.test_case.id)?
            } else {
                "images/placeholder.png".to_string()
            }
        } else {
            "images/placeholder.png".to_string()
        };

        let metrics_html = self.format_metrics(&result.metrics);

        Ok(format!(
            r#"                    <tr>
                        <td>
                            <strong>{}</strong><br>
                            <small>{}</small>
                        </td>
                        <td class="image-cell">
                            <img src="{}" alt="Original" class="test-image" onclick="openModal(this.src)">
                            <br><small>Original</small>
                        </td>
                        <td class="image-cell">
                            <img src="{}" alt="Expected" class="test-image" onclick="openModal(this.src)">
                            <br><small>Expected</small>
                        </td>
                        <td class="image-cell">
                            <img src="{}" alt="Current" class="test-image" onclick="openModal(this.src)">
                            <br><small>Current</small>
                        </td>
                        <td class="image-cell">
                            <img src="{}" alt="Diff Heatmap" class="test-image" onclick="openModal(this.src)">
                            <br><small>Diff Heatmap</small>
                        </td>
                        <td class="{}">
                            {}<br>
                            <small>{}ms</small>
                        </td>
                        <td class="metrics">
                            {}
                        </td>
                    </tr>"#,
            result.test_case.id,
            result.test_case.description,
            original_image_path,
            expected_image_path,
            current_image_path,
            diff_heatmap_path,
            status_class,
            status_text,
            result.processing_time.as_millis(),
            metrics_html
        ))
    }

    /// Copy image to output directory and return relative path
    fn copy_image_to_output(&self, source_path: &str, prefix: &str) -> Result<String> {
        let source = Path::new(source_path);
        let filename = source
            .file_name()
            .ok_or_else(|| TestingError::InvalidConfiguration("Invalid source path".to_string()))?;

        let output_filename = format!("{}_{}", prefix, filename.to_string_lossy());
        let output_path = self.output_dir.join("images").join(&output_filename);

        if source.exists() {
            std::fs::copy(source, &output_path)?;
        } else {
            // Create a placeholder if source doesn't exist
            self.create_placeholder_image(&output_path)?;
        }

        Ok(format!("images/{}", output_filename))
    }

    /// Create a placeholder image for missing files
    fn create_placeholder_image(&self, output_path: &Path) -> Result<()> {
        use image::{Rgba, RgbaImage};

        let placeholder = RgbaImage::from_fn(150, 150, |x, y| {
            if (x + y) % 20 < 10 {
                Rgba([200, 200, 200, 255])
            } else {
                Rgba([220, 220, 220, 255])
            }
        });

        placeholder.save(output_path)?;
        Ok(())
    }

    /// Generate a diff heatmap image showing differences between expected and current output
    fn generate_diff_heatmap(
        &self,
        expected_path: &Path,
        current_path: &Path,
        test_id: &str,
    ) -> Result<String> {
        use image::GenericImageView;

        // Load images
        let expected = image::open(expected_path)?;
        let current = image::open(current_path)?;

        // Ensure images have the same dimensions
        let (width, height) = expected.dimensions();
        let current_resized = if current.dimensions() != (width, height) {
            current.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        } else {
            current
        };

        // Generate heatmap using the existing diff generation logic from ImageComparison
        let diff_image =
            crate::ImageComparison::generate_enhanced_diff_heatmap(&current_resized, &expected)?;

        // Save the heatmap
        let heatmap_filename = format!("heatmap_{}.png", test_id);
        let heatmap_path = self.output_dir.join("images").join(&heatmap_filename);
        diff_image.save(&heatmap_path)?;

        Ok(format!("images/{}", heatmap_filename))
    }

    /// Format metrics for display
    fn format_metrics(&self, metrics: &crate::TestMetrics) -> String {
        let pixel_class = self.get_metric_class(metrics.pixel_accuracy, 0.9, 0.8);
        let ssim_class = self.get_metric_class(metrics.ssim, 0.85, 0.7);
        let edge_class = self.get_metric_class(metrics.edge_accuracy, 0.85, 0.7);

        let visual_class = self.get_metric_class(metrics.visual_quality_score, 0.8, 0.6);

        format!(
            r#"Visual: <span class="{}">{:.1}%</span><br>
SSIM: <span class="{}">{:.3}</span><br>
Edge: <span class="{}">{:.1}%</span><br>
Pixel: <span class="{}">{:.1}%</span><br>
MSE: {:.2}"#,
            visual_class,
            metrics.visual_quality_score * 100.0,
            ssim_class,
            metrics.ssim,
            edge_class,
            metrics.edge_accuracy * 100.0,
            pixel_class,
            metrics.pixel_accuracy * 100.0,
            metrics.mean_squared_error
        )
    }

    /// Get CSS class based on metric thresholds
    fn get_metric_class(
        &self,
        value: f64,
        good_threshold: f64,
        warning_threshold: f64,
    ) -> &'static str {
        if value >= good_threshold {
            "metric-good"
        } else if value >= warning_threshold {
            "metric-warning"
        } else {
            "metric-poor"
        }
    }

    /// Generate detailed metrics section
    fn generate_metrics_section(&self, session: &TestSession) -> String {
        // Calculate category-wise statistics
        let mut category_stats: std::collections::BTreeMap<String, CategoryStats> =
            std::collections::BTreeMap::new();

        for result in &session.results {
            let stats = category_stats
                .entry(result.test_case.category.clone())
                .or_insert_with(CategoryStats::default);

            stats.total += 1;
            if result.passed {
                stats.passed += 1;
            }
            stats.total_accuracy += result.metrics.pixel_accuracy;
            stats.total_ssim += result.metrics.ssim;
            stats.total_time += result.processing_time.as_millis() as u64;
        }

        let mut metrics_html = String::from(
            r#"    <div class="comparison-table">
        <div class="table-header">
            <h2>📊 Category Statistics</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Tests</th>
                    <th>Pass Rate</th>
                    <th>Avg Accuracy</th>
                    <th>Avg SSIM</th>
                    <th>Avg Time</th>
                </tr>
            </thead>
            <tbody>"#,
        );

        for (category, stats) in category_stats {
            let pass_rate = (stats.passed as f64 / stats.total as f64) * 100.0;
            let avg_accuracy = (stats.total_accuracy / stats.total as f64) * 100.0;
            let avg_ssim = stats.total_ssim / stats.total as f64;
            let avg_time = stats.total_time / stats.total as u64;

            metrics_html.push_str(&format!(
                r#"                <tr>
                    <td><strong>{}</strong></td>
                    <td>{}</td>
                    <td>{:.1}%</td>
                    <td>{:.1}%</td>
                    <td>{:.3}</td>
                    <td>{}ms</td>
                </tr>"#,
                category, stats.total, pass_rate, avg_accuracy, avg_ssim, avg_time
            ));
        }

        metrics_html.push_str("            </tbody>\n        </table>\n    </div>");
        metrics_html
    }

    /// Generate HTML footer with JavaScript for image modal
    fn generate_html_footer(&self) -> String {
        r#"    
    <!-- Image Modal -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <div class="footer">
        <p>Generated by bg-remove-testing | Rust Background Removal Library</p>
    </div>

    <script>
        function openModal(imageSrc) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = imageSrc;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal when clicking outside the image
        window.onclick = function(event) {
            const modal = document.getElementById('imageModal');
            if (event.target == modal) {
                closeModal();
            }
        }

        // Close modal with escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>"#
            .to_string()
    }
}

/// Statistics for a category of tests
#[derive(Debug, Default)]
struct CategoryStats {
    total: usize,
    passed: usize,
    total_accuracy: f64,
    total_ssim: f64,
    total_time: u64,
}
