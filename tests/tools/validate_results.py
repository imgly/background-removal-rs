#!/usr/bin/env python3
"""
Result validation script for background removal testing.

This script validates Rust implementation outputs against reference data
and generates detailed comparison reports.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from PIL import Image
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a single validation test."""
    test_id: str
    test_name: str
    pixel_accuracy: float
    edge_accuracy: float
    structural_similarity: float
    mean_pixel_difference: float
    processing_time_ms: float
    memory_usage_mb: float
    passed: bool
    error_message: Optional[str] = None

@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_pixel_accuracy: float
    average_edge_accuracy: float
    average_processing_time: float
    average_memory_usage: float
    categories: Dict[str, Dict]

class ResultValidator:
    """Validates Rust implementation results against reference data."""
    
    def __init__(self, test_assets_dir: Path, rust_output_dir: Path):
        """
        Initialize the result validator.
        
        Args:
            test_assets_dir: Path to test assets directory
            rust_output_dir: Path to Rust implementation outputs
        """
        self.assets_dir = test_assets_dir
        self.rust_output_dir = rust_output_dir
        self.expected_dir = test_assets_dir / "expected"
        self.metadata_dir = test_assets_dir / "metadata"
        
        # Load validation rules and test cases
        self.validation_rules = self._load_json(self.metadata_dir / "validation_rules.json")
        self.test_cases = self._load_json(self.metadata_dir / "test_cases.json")
        
    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Required file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise
    
    def validate_all_results(self) -> ValidationSummary:
        """
        Validate all test results.
        
        Returns:
            ValidationSummary with overall results
        """
        logger.info("Starting comprehensive result validation...")
        
        all_results = []
        category_results = {}
        
        # Validate each test category
        for category_name, category_data in self.test_cases["categories"].items():
            logger.info(f"Validating category: {category_name}")
            
            category_results[category_name] = {
                "tests": [],
                "passed": 0,
                "failed": 0,
                "average_accuracy": 0.0
            }
            
            for test_case in category_data.get("test_cases", []):
                try:
                    result = self._validate_single_test(test_case, category_name)
                    all_results.append(result)
                    category_results[category_name]["tests"].append(result)
                    
                    if result.passed:
                        category_results[category_name]["passed"] += 1
                    else:
                        category_results[category_name]["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to validate test {test_case.get('id', 'unknown')}: {e}")
                    error_result = ValidationResult(
                        test_id=test_case.get('id', 'unknown'),
                        test_name=test_case.get('name', 'Unknown'),
                        pixel_accuracy=0.0,
                        edge_accuracy=0.0,
                        structural_similarity=0.0,
                        mean_pixel_difference=1000.0,
                        processing_time_ms=0.0,
                        memory_usage_mb=0.0,
                        passed=False,
                        error_message=str(e)
                    )
                    all_results.append(error_result)
                    category_results[category_name]["tests"].append(error_result)
                    category_results[category_name]["failed"] += 1
            
            # Calculate category averages
            category_tests = category_results[category_name]["tests"]
            if category_tests:
                category_results[category_name]["average_accuracy"] = np.mean([
                    t.pixel_accuracy for t in category_tests if t.passed
                ])
        
        # Generate summary
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = len(all_results) - passed_tests
        
        summary = ValidationSummary(
            total_tests=len(all_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_pixel_accuracy=np.mean([r.pixel_accuracy for r in all_results if r.passed]),
            average_edge_accuracy=np.mean([r.edge_accuracy for r in all_results if r.passed]),
            average_processing_time=np.mean([r.processing_time_ms for r in all_results if r.passed]),
            average_memory_usage=np.mean([r.memory_usage_mb for r in all_results if r.passed]),
            categories=category_results
        )
        
        logger.info(f"Validation completed: {passed_tests}/{len(all_results)} tests passed")
        return summary
    
    def _validate_single_test(self, test_case: Dict, category: str) -> ValidationResult:
        """
        Validate a single test case.
        
        Args:
            test_case: Test case specification
            category: Test category name
            
        Returns:
            ValidationResult for this test
        """
        test_id = test_case["id"]
        test_name = test_case["name"]
        input_file = test_case["input_file"]
        
        logger.debug(f"Validating test {test_id}: {test_name}")
        
        # Find Rust output files
        rust_outputs = self._find_rust_outputs(input_file)
        if not rust_outputs:
            raise FileNotFoundError(f"No Rust outputs found for {input_file}")
        
        # Load reference outputs
        reference_outputs = self._load_reference_outputs(input_file)
        
        # Validate each output format
        pixel_accuracies = []
        edge_accuracies = []
        ssim_scores = []
        pixel_differences = []
        
        for output_type, rust_file in rust_outputs.items():
            if output_type in reference_outputs:
                ref_file = reference_outputs[output_type]
                
                # Load images
                rust_img = self._load_image(rust_file)
                ref_img = self._load_image(ref_file)
                
                # Calculate metrics
                pixel_acc = self._calculate_pixel_accuracy(rust_img, ref_img)
                edge_acc = self._calculate_edge_accuracy(rust_img, ref_img) 
                ssim = self._calculate_structural_similarity(rust_img, ref_img)
                pixel_diff = self._calculate_mean_pixel_difference(rust_img, ref_img)
                
                pixel_accuracies.append(pixel_acc)
                edge_accuracies.append(edge_acc)
                ssim_scores.append(ssim)
                pixel_differences.append(pixel_diff)
        
        # Calculate overall metrics
        avg_pixel_accuracy = np.mean(pixel_accuracies) if pixel_accuracies else 0.0
        avg_edge_accuracy = np.mean(edge_accuracies) if edge_accuracies else 0.0
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
        avg_pixel_diff = np.mean(pixel_differences) if pixel_differences else 1000.0
        
        # Load performance data (would come from Rust implementation)
        performance_data = self._load_performance_data(test_id)
        
        # Check if test passes validation criteria
        criteria = test_case.get("validation_criteria", {})
        passes_validation = (
            avg_pixel_accuracy >= criteria.get("pixel_accuracy_min", 0.8) and
            avg_edge_accuracy >= criteria.get("edge_accuracy_min", 0.7) and
            performance_data["processing_time_ms"] <= criteria.get("processing_time_max_ms", 5000)
        )
        
        return ValidationResult(
            test_id=test_id,
            test_name=test_name,
            pixel_accuracy=avg_pixel_accuracy,
            edge_accuracy=avg_edge_accuracy,
            structural_similarity=avg_ssim,
            mean_pixel_difference=avg_pixel_diff,
            processing_time_ms=performance_data["processing_time_ms"],
            memory_usage_mb=performance_data["memory_usage_mb"],
            passed=passes_validation
        )
    
    def _find_rust_outputs(self, input_file: str) -> Dict[str, Path]:
        """Find Rust implementation output files for given input."""
        base_name = Path(input_file).stem
        outputs = {}
        
        # Expected output patterns
        patterns = {
            "png_alpha": f"{base_name}_alpha.png",
            "jpeg_white_bg": f"{base_name}_white.jpg",
            "webp_alpha": f"{base_name}_alpha.webp", 
            "mask_only": f"{base_name}_mask.png"
        }
        
        for output_type, pattern in patterns.items():
            output_file = self.rust_output_dir / pattern
            if output_file.exists():
                outputs[output_type] = output_file
        
        return outputs
    
    def _load_reference_outputs(self, input_file: str) -> Dict[str, Path]:
        """Load reference output files for given input."""
        base_name = Path(input_file).stem
        outputs = {}
        
        # Reference output patterns
        patterns = {
            "png_alpha": f"{base_name}.png",
            "jpeg_white_bg": f"{base_name}.jpg",
            "webp_alpha": f"{base_name}.webp",
            "mask_only": f"js_{base_name}_mask.png"
        }
        
        js_output_dir = self.expected_dir / "javascript_output"
        masks_dir = self.expected_dir / "masks"
        
        for output_type, pattern in patterns.items():
            if output_type == "mask_only":
                output_file = masks_dir / pattern
            else:
                output_file = js_output_dir / pattern
                
            if output_file.exists():
                outputs[output_type] = output_file
        
        return outputs
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load image as numpy array."""
        try:
            img = Image.open(file_path)
            return np.array(img)
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            raise
    
    def _calculate_pixel_accuracy(self, img1: np.ndarray, img2: np.ndarray, tolerance: int = 5) -> float:
        """
        Calculate pixel-level accuracy between two images.
        
        Args:
            img1: First image
            img2: Second image  
            tolerance: Pixel value tolerance (0-255)
            
        Returns:
            Accuracy as float between 0 and 1
        """
        if img1.shape != img2.shape:
            logger.warning(f"Image shape mismatch: {img1.shape} vs {img2.shape}")
            # Resize to match
            from PIL import Image
            img1_pil = Image.fromarray(img1)
            img2_pil = Image.fromarray(img2)
            
            target_size = (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0]))
            img1_pil = img1_pil.resize(target_size)
            img2_pil = img2_pil.resize(target_size)
            
            img1 = np.array(img1_pil)
            img2 = np.array(img2_pil)
        
        # Calculate absolute difference
        if len(img1.shape) == 3:
            diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
            diff = np.max(diff, axis=2)  # Max difference across channels
        else:
            diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
        
        # Count pixels within tolerance
        correct_pixels = np.sum(diff <= tolerance)
        total_pixels = diff.size
        
        return correct_pixels / total_pixels
    
    def _calculate_edge_accuracy(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate edge detection accuracy between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Edge accuracy as float between 0 and 1
        """
        try:
            from scipy import ndimage
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1_gray = np.mean(img1, axis=2)
            else:
                img1_gray = img1
                
            if len(img2.shape) == 3:
                img2_gray = np.mean(img2, axis=2)
            else:
                img2_gray = img2
            
            # Resize if needed
            if img1_gray.shape != img2_gray.shape:
                from PIL import Image
                img2_pil = Image.fromarray(img2_gray.astype(np.uint8))
                img2_pil = img2_pil.resize((img1_gray.shape[1], img1_gray.shape[0]))
                img2_gray = np.array(img2_pil)
            
            # Sobel edge detection
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            edges1_x = ndimage.convolve(img1_gray, sobel_x)
            edges1_y = ndimage.convolve(img1_gray, sobel_y)
            edges1 = np.sqrt(edges1_x**2 + edges1_y**2)
            
            edges2_x = ndimage.convolve(img2_gray, sobel_x)
            edges2_y = ndimage.convolve(img2_gray, sobel_y)
            edges2 = np.sqrt(edges2_x**2 + edges2_y**2)
            
            # Threshold to binary edge maps
            threshold = 50
            edges1_binary = edges1 > threshold
            edges2_binary = edges2 > threshold
            
            # Calculate accuracy
            correct = np.sum(edges1_binary == edges2_binary)
            total = edges1_binary.size
            
            return correct / total
            
        except ImportError:
            logger.warning("scipy not available, using simplified edge calculation")
            return self._calculate_pixel_accuracy(img1, img2, tolerance=10)
    
    def _calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate structural similarity (SSIM) between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            SSIM value between -1 and 1
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1_gray = np.mean(img1, axis=2)
            else:
                img1_gray = img1
                
            if len(img2.shape) == 3:
                img2_gray = np.mean(img2, axis=2)
            else:
                img2_gray = img2
            
            # Resize if needed
            if img1_gray.shape != img2_gray.shape:
                from PIL import Image
                img2_pil = Image.fromarray(img2_gray.astype(np.uint8))
                img2_pil = img2_pil.resize((img1_gray.shape[1], img1_gray.shape[0]))
                img2_gray = np.array(img2_pil)
            
            return ssim(img1_gray, img2_gray, data_range=255)
            
        except ImportError:
            logger.warning("skimage not available, using simplified similarity calculation")
            return 1.0 - (self._calculate_mean_pixel_difference(img1, img2) / 255.0)
    
    def _calculate_mean_pixel_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate mean pixel difference between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Mean absolute pixel difference (0-255)
        """
        if img1.shape != img2.shape:
            # Resize to match
            from PIL import Image
            img1_pil = Image.fromarray(img1)
            img2_pil = Image.fromarray(img2)
            
            target_size = (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0]))
            img1_pil = img1_pil.resize(target_size)
            img2_pil = img2_pil.resize(target_size)
            
            img1 = np.array(img1_pil)
            img2 = np.array(img2_pil)
        
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)  # Average across channels
        
        return np.mean(diff)
    
    def _load_performance_data(self, test_id: str) -> Dict:
        """
        Load performance data for test.
        
        This would typically come from the Rust implementation's benchmark output.
        For now, return placeholder data.
        """
        # In a real implementation, this would load actual performance metrics
        # from the Rust test runner
        return {
            "processing_time_ms": 1000.0,  # Placeholder
            "memory_usage_mb": 150.0       # Placeholder
        }
    
    def generate_report(self, summary: ValidationSummary, output_file: Path):
        """
        Generate detailed validation report.
        
        Args:
            summary: Validation summary
            output_file: Output file path
        """
        logger.info(f"Generating validation report: {output_file}")
        
        # Generate HTML report
        html_content = self._generate_html_report(summary)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        # Also generate JSON summary
        json_file = output_file.with_suffix('.json')
        self._save_json_summary(summary, json_file)
        
        logger.info(f"Report generated: {output_file}")
        logger.info(f"JSON summary: {json_file}")
    
    def _generate_html_report(self, summary: ValidationSummary) -> str:
        """Generate HTML validation report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Background Removal Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .category {{ margin-bottom: 30px; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .passed {{ border-left-color: #4CAF50; }}
        .failed {{ border-left-color: #f44336; }}
        .metrics {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .metric {{ background: white; padding: 10px; border-radius: 3px; min-width: 120px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Background Removal Validation Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Total Tests:</strong> {summary.total_tests}
            </div>
            <div class="metric">
                <strong>Passed:</strong> {summary.passed_tests}
            </div>
            <div class="metric">
                <strong>Failed:</strong> {summary.failed_tests}
            </div>
            <div class="metric">
                <strong>Success Rate:</strong> {summary.passed_tests/summary.total_tests*100:.1f}%
            </div>
            <div class="metric">
                <strong>Avg Pixel Accuracy:</strong> {summary.average_pixel_accuracy:.3f}
            </div>
            <div class="metric">
                <strong>Avg Edge Accuracy:</strong> {summary.average_edge_accuracy:.3f}
            </div>
            <div class="metric">
                <strong>Avg Processing Time:</strong> {summary.average_processing_time:.1f}ms
            </div>
            <div class="metric">
                <strong>Avg Memory Usage:</strong> {summary.average_memory_usage:.1f}MB
            </div>
        </div>
    </div>
    
    <h2>Category Results</h2>
        """
        
        for category_name, category_data in summary.categories.items():
            html += f"""
    <div class="category">
        <h3>{category_name.title()}</h3>
        <p>Passed: {category_data['passed']}/{category_data['passed'] + category_data['failed']} 
           (Avg Accuracy: {category_data['average_accuracy']:.3f})</p>
        
        <table>
            <tr>
                <th>Test ID</th>
                <th>Test Name</th>
                <th>Pixel Accuracy</th>
                <th>Edge Accuracy</th>
                <th>SSIM</th>
                <th>Processing Time (ms)</th>
                <th>Status</th>
            </tr>
            """
            
            for test in category_data['tests']:
                status_class = "passed" if test.passed else "failed"
                status_text = "PASS" if test.passed else "FAIL"
                
                html += f"""
            <tr class="{status_class}">
                <td>{test.test_id}</td>
                <td>{test.test_name}</td>
                <td>{test.pixel_accuracy:.3f}</td>
                <td>{test.edge_accuracy:.3f}</td>
                <td>{test.structural_similarity:.3f}</td>
                <td>{test.processing_time_ms:.1f}</td>
                <td>{status_text}</td>
            </tr>
                """
            
            html += "</table></div>"
        
        html += """
</body>
</html>
        """
        
        return html
    
    def _save_json_summary(self, summary: ValidationSummary, output_file: Path):
        """Save validation summary as JSON."""
        summary_dict = {
            "total_tests": summary.total_tests,
            "passed_tests": summary.passed_tests,
            "failed_tests": summary.failed_tests,
            "success_rate": summary.passed_tests / summary.total_tests if summary.total_tests > 0 else 0,
            "average_pixel_accuracy": summary.average_pixel_accuracy,
            "average_edge_accuracy": summary.average_edge_accuracy,
            "average_processing_time": summary.average_processing_time,
            "average_memory_usage": summary.average_memory_usage,
            "categories": {}
        }
        
        for category_name, category_data in summary.categories.items():
            summary_dict["categories"][category_name] = {
                "passed": category_data["passed"],
                "failed": category_data["failed"],
                "average_accuracy": category_data["average_accuracy"],
                "tests": [
                    {
                        "test_id": test.test_id,
                        "test_name": test.test_name,
                        "pixel_accuracy": test.pixel_accuracy,
                        "edge_accuracy": test.edge_accuracy,
                        "structural_similarity": test.structural_similarity,
                        "processing_time_ms": test.processing_time_ms,
                        "passed": test.passed,
                        "error_message": test.error_message
                    }
                    for test in category_data["tests"]
                ]
            }
        
        with open(output_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)


def main():
    """Main entry point for result validation."""
    parser = argparse.ArgumentParser(description="Validate background removal results")
    parser.add_argument("--assets-dir", type=Path,
                       default=Path(__file__).parent.parent / "assets",
                       help="Test assets directory")
    parser.add_argument("--rust-output-dir", type=Path, required=True,
                       help="Directory containing Rust implementation outputs")
    parser.add_argument("--report-file", type=Path,
                       default=Path("validation_report.html"),
                       help="Output report file")
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = ResultValidator(args.assets_dir, args.rust_output_dir)
        
        # Run validation
        summary = validator.validate_all_results()
        
        # Generate report
        validator.generate_report(summary, args.report_file)
        
        # Print summary
        success_rate = summary.passed_tests / summary.total_tests * 100 if summary.total_tests > 0 else 0
        logger.info(f"Validation complete: {summary.passed_tests}/{summary.total_tests} tests passed ({success_rate:.1f}%)")
        
        return 0 if summary.failed_tests == 0 else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())