#!/usr/bin/env python3
"""
Performance benchmark runner for background removal testing.

This script runs comprehensive performance benchmarks and generates
detailed performance analysis reports.
"""

import subprocess
import time
import psutil
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import argparse
import platform
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    test_id: str
    test_name: str
    category: str
    image_resolution: Tuple[int, int]
    processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class PerformanceProfile:
    """Performance profile for a test category."""
    category: str
    total_tests: int
    successful_tests: int
    avg_processing_time: float
    median_processing_time: float
    p95_processing_time: float
    avg_memory_usage: float
    peak_memory_usage: float
    avg_cpu_usage: float
    throughput_images_per_second: float

@dataclass
class BenchmarkSummary:
    """Overall benchmark summary."""
    platform_info: Dict
    total_runtime_seconds: float
    profiles: Dict[str, PerformanceProfile]
    comparison_data: Optional[Dict] = None

class PerformanceBenchmark:
    """Runs performance benchmarks for background removal."""
    
    def __init__(self, test_assets_dir: Path, rust_binary: Path):
        """
        Initialize the benchmark runner.
        
        Args:
            test_assets_dir: Path to test assets directory
            rust_binary: Path to Rust implementation binary
        """
        self.assets_dir = test_assets_dir
        self.rust_binary = rust_binary
        self.input_dir = test_assets_dir / "input"
        self.metadata_dir = test_assets_dir / "metadata"
        self.output_dir = Path("benchmark_outputs")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load test specifications
        self.image_specs = self._load_json(self.metadata_dir / "image_specs.json")
        self.test_cases = self._load_json(self.metadata_dir / "test_cases.json")
        
        # System information
        self.platform_info = self._collect_platform_info()
        
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
    
    def _collect_platform_info(self) -> Dict:
        """Collect system platform information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "rust_binary": str(self.rust_binary)
        }
    
    def run_comprehensive_benchmark(self, iterations: int = 5) -> BenchmarkSummary:
        """
        Run comprehensive performance benchmarks.
        
        Args:
            iterations: Number of iterations per test
            
        Returns:
            BenchmarkSummary with results
        """
        logger.info(f"Starting comprehensive benchmark with {iterations} iterations per test")
        start_time = time.time()
        
        all_results = []
        category_profiles = {}
        
        # Warm up the system
        logger.info("Warming up system...")
        self._warmup_run()
        
        # Run benchmarks for each category
        for category_name in ["portraits", "products", "complex", "edge_cases"]:
            logger.info(f"Benchmarking category: {category_name}")
            
            category_results = self._benchmark_category(category_name, iterations)
            all_results.extend(category_results)
            
            # Generate performance profile
            profile = self._generate_performance_profile(category_name, category_results)
            category_profiles[category_name] = profile
        
        # Run raw format benchmarks if available
        if "raw_formats" in self.image_specs:
            logger.info("Benchmarking raw formats...")
            raw_results = self._benchmark_raw_formats(iterations)
            all_results.extend(raw_results)
            category_profiles["raw_formats"] = self._generate_performance_profile("raw_formats", raw_results)
        
        total_runtime = time.time() - start_time
        
        # Load comparison data if available
        comparison_data = self._load_comparison_data()
        
        summary = BenchmarkSummary(
            platform_info=self.platform_info,
            total_runtime_seconds=total_runtime,
            profiles=category_profiles,
            comparison_data=comparison_data
        )
        
        logger.info(f"Benchmark completed in {total_runtime:.1f} seconds")
        return summary
    
    def _warmup_run(self):
        """Perform warmup runs to stabilize performance."""
        # Find a small test image for warmup
        warmup_image = None
        for category in ["portraits", "products"]:
            category_dir = self.input_dir / category
            if category_dir.exists():
                for img_file in category_dir.glob("*.jpg"):
                    warmup_image = img_file
                    break
                if warmup_image:
                    break
        
        if warmup_image:
            logger.info(f"Running warmup with {warmup_image.name}")
            for _ in range(3):
                try:
                    self._run_single_benchmark(warmup_image, "warmup")
                except Exception as e:
                    logger.warning(f"Warmup run failed: {e}")
        else:
            logger.warning("No warmup image found")
    
    def _benchmark_category(self, category: str, iterations: int) -> List[BenchmarkResult]:
        """Benchmark all tests in a category."""
        results = []
        category_dir = self.input_dir / category
        
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            return results
        
        # Get image specifications for this category
        category_specs = self.image_specs.get(category, [])
        
        for spec in category_specs:
            filename = spec["filename"]
            image_file = category_dir / filename
            
            if not image_file.exists():
                logger.warning(f"Test image not found: {image_file}")
                continue
            
            logger.info(f"Benchmarking {filename} ({iterations} iterations)")
            
            # Run multiple iterations
            iteration_results = []
            for i in range(iterations):
                try:
                    result = self._run_single_benchmark(image_file, category, spec)
                    iteration_results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark iteration {i+1} failed for {filename}: {e}")
                    # Create error result
                    error_result = BenchmarkResult(
                        test_id=f"{category}_{Path(filename).stem}",
                        test_name=filename,
                        category=category,
                        image_resolution=tuple(spec.get("resolution", [0, 0])),
                        processing_time_ms=0.0,
                        memory_usage_mb=0.0,
                        cpu_usage_percent=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    iteration_results.append(error_result)
            
            # Calculate statistics from iterations
            if iteration_results:
                successful_results = [r for r in iteration_results if r.success]
                if successful_results:
                    # Use median values for stable results
                    median_result = BenchmarkResult(
                        test_id=f"{category}_{Path(filename).stem}",
                        test_name=filename,
                        category=category,
                        image_resolution=tuple(spec.get("resolution", [0, 0])),
                        processing_time_ms=statistics.median([r.processing_time_ms for r in successful_results]),
                        memory_usage_mb=statistics.median([r.memory_usage_mb for r in successful_results]),
                        cpu_usage_percent=statistics.median([r.cpu_usage_percent for r in successful_results]),
                        success=True
                    )
                    results.append(median_result)
                else:
                    # All iterations failed
                    results.append(iteration_results[0])  # Use first error result
        
        return results
    
    def _run_single_benchmark(self, image_file: Path, category: str, spec: Optional[Dict] = None) -> BenchmarkResult:
        """Run benchmark for a single image."""
        test_id = f"{category}_{image_file.stem}"
        output_file = self.output_dir / f"{test_id}_output.png"
        
        # Prepare command
        cmd = [
            str(self.rust_binary),
            str(image_file),
            "--output", str(output_file),
            "--format", "png"
        ]
        
        # Add raw format parameters if needed
        if spec and "format" in spec and spec["format"] in ["RGB24", "RGBA32", "YUV420P", "YUV444P"]:
            cmd.extend([
                "--input-format", spec["format"].lower(),
                "--width", str(spec["resolution"][0]),
                "--height", str(spec["resolution"][1])
            ])
            if "color_space" in spec:
                cmd.extend(["--color-space", spec["color_space"]])
        
        # Start monitoring
        start_time = time.time()
        memory_samples = []
        cpu_samples = []
        
        try:
            # Start process
            process = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Monitor resource usage
            while process.poll() is None:
                try:
                    # Memory usage
                    memory_info = process.memory_info()
                    memory_samples.append(memory_info.rss / 1024 / 1024)  # MB
                    
                    # CPU usage (approximate)
                    cpu_percent = process.cpu_percent()
                    cpu_samples.append(cpu_percent)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                
                time.sleep(0.01)  # 10ms sampling
            
            # Get final result
            stdout, stderr = process.communicate()
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # ms
            peak_memory = max(memory_samples) if memory_samples else 0.0
            avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0.0
            
            success = process.returncode == 0 and output_file.exists()
            error_message = stderr.decode() if not success else None
            
            return BenchmarkResult(
                test_id=test_id,
                test_name=image_file.name,
                category=category,
                image_resolution=tuple(spec.get("resolution", [0, 0])) if spec else (0, 0),
                processing_time_ms=processing_time,
                memory_usage_mb=peak_memory,
                cpu_usage_percent=avg_cpu,
                success=success,
                error_message=error_message
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_id=test_id,
                test_name=image_file.name,
                category=category,
                image_resolution=tuple(spec.get("resolution", [0, 0])) if spec else (0, 0),
                processing_time_ms=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _benchmark_raw_formats(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark raw format processing."""
        results = []
        raw_formats_dir = self.input_dir / "raw_formats"
        
        if not raw_formats_dir.exists():
            logger.info("No raw formats directory found, skipping raw format benchmarks")
            return results
        
        raw_specs = self.image_specs.get("raw_formats", [])
        
        for spec in raw_specs:
            filename = spec["filename"]
            raw_file = raw_formats_dir / filename
            
            if not raw_file.exists():
                logger.warning(f"Raw format file not found: {raw_file}")
                continue
            
            logger.info(f"Benchmarking raw format {filename} ({iterations} iterations)")
            
            # Run iterations
            iteration_results = []
            for i in range(iterations):
                try:
                    result = self._run_single_benchmark(raw_file, "raw_formats", spec)
                    iteration_results.append(result)
                except Exception as e:
                    logger.error(f"Raw format benchmark iteration {i+1} failed for {filename}: {e}")
            
            # Calculate median result
            if iteration_results:
                successful_results = [r for r in iteration_results if r.success]
                if successful_results:
                    median_result = BenchmarkResult(
                        test_id=f"raw_{Path(filename).stem}",
                        test_name=filename,
                        category="raw_formats",
                        image_resolution=tuple(spec.get("resolution", [0, 0])),
                        processing_time_ms=statistics.median([r.processing_time_ms for r in successful_results]),
                        memory_usage_mb=statistics.median([r.memory_usage_mb for r in successful_results]),
                        cpu_usage_percent=statistics.median([r.cpu_usage_percent for r in successful_results]),
                        success=True
                    )
                    results.append(median_result)
        
        return results
    
    def _generate_performance_profile(self, category: str, results: List[BenchmarkResult]) -> PerformanceProfile:
        """Generate performance profile for a category."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return PerformanceProfile(
                category=category,
                total_tests=len(results),
                successful_tests=0,
                avg_processing_time=0.0,
                median_processing_time=0.0,
                p95_processing_time=0.0,
                avg_memory_usage=0.0,
                peak_memory_usage=0.0,
                avg_cpu_usage=0.0,
                throughput_images_per_second=0.0
            )
        
        # Calculate statistics
        processing_times = [r.processing_time_ms for r in successful_results]
        memory_usages = [r.memory_usage_mb for r in successful_results]
        cpu_usages = [r.cpu_usage_percent for r in successful_results]
        
        avg_processing_time = statistics.mean(processing_times)
        median_processing_time = statistics.median(processing_times)
        p95_processing_time = self._percentile(processing_times, 95)
        
        avg_memory_usage = statistics.mean(memory_usages)
        peak_memory_usage = max(memory_usages)
        
        avg_cpu_usage = statistics.mean(cpu_usages)
        
        # Calculate throughput (images per second)
        throughput = 1000.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        return PerformanceProfile(
            category=category,
            total_tests=len(results),
            successful_tests=len(successful_results),
            avg_processing_time=avg_processing_time,
            median_processing_time=median_processing_time,
            p95_processing_time=p95_processing_time,
            avg_memory_usage=avg_memory_usage,
            peak_memory_usage=peak_memory_usage,
            avg_cpu_usage=avg_cpu_usage,
            throughput_images_per_second=throughput
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index < len(sorted_data):
                fraction = index - lower_index
                return sorted_data[lower_index] + fraction * (sorted_data[upper_index] - sorted_data[lower_index])
            else:
                return sorted_data[lower_index]
    
    def _load_comparison_data(self) -> Optional[Dict]:
        """Load comparison data from JavaScript baseline."""
        baseline_file = self.assets_dir / "expected" / "benchmarks" / "javascript_baseline.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load comparison data: {e}")
        
        return None
    
    def generate_report(self, summary: BenchmarkSummary, output_file: Path):
        """Generate detailed benchmark report."""
        logger.info(f"Generating benchmark report: {output_file}")
        
        # Generate HTML report
        html_content = self._generate_html_report(summary)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        # Generate JSON summary
        json_file = output_file.with_suffix('.json')
        self._save_json_summary(summary, json_file)
        
        logger.info(f"Benchmark report generated: {output_file}")
        logger.info(f"JSON summary: {json_file}")
    
    def _generate_html_report(self, summary: BenchmarkSummary) -> str:
        """Generate HTML benchmark report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Background Removal Performance Benchmark</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .category {{ margin-bottom: 30px; }}
        .metrics {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .metric {{ background: white; padding: 10px; border-radius: 3px; min-width: 120px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .platform-info {{ background: #e8f4f8; padding: 10px; border-radius: 3px; margin-bottom: 20px; }}
        .comparison {{ background: #fff3cd; padding: 10px; border-radius: 3px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Background Removal Performance Benchmark</h1>
    
    <div class="platform-info">
        <h3>Platform Information</h3>
        <div class="metrics">
            <div class="metric"><strong>Platform:</strong> {summary.platform_info['platform']}</div>
            <div class="metric"><strong>CPU Count:</strong> {summary.platform_info['cpu_count']}</div>
            <div class="metric"><strong>Total Memory:</strong> {summary.platform_info['memory_total_gb']:.1f} GB</div>
            <div class="metric"><strong>Runtime:</strong> {summary.total_runtime_seconds:.1f}s</div>
        </div>
    </div>
    
    <div class="summary">
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Success Rate</th>
                <th>Avg Time (ms)</th>
                <th>Median Time (ms)</th>
                <th>P95 Time (ms)</th>
                <th>Avg Memory (MB)</th>
                <th>Peak Memory (MB)</th>
                <th>Throughput (img/s)</th>
            </tr>
        """
        
        for category, profile in summary.profiles.items():
            success_rate = profile.successful_tests / profile.total_tests * 100 if profile.total_tests > 0 else 0
            html += f"""
            <tr>
                <td>{category.title()}</td>
                <td>{success_rate:.1f}%</td>
                <td>{profile.avg_processing_time:.1f}</td>
                <td>{profile.median_processing_time:.1f}</td>
                <td>{profile.p95_processing_time:.1f}</td>
                <td>{profile.avg_memory_usage:.1f}</td>
                <td>{profile.peak_memory_usage:.1f}</td>
                <td>{profile.throughput_images_per_second:.2f}</td>
            </tr>
            """
        
        html += "</table></div>"
        
        # Add comparison with JavaScript baseline if available
        if summary.comparison_data:
            html += self._generate_comparison_section(summary)
        
        html += "</body></html>"
        return html
    
    def _generate_comparison_section(self, summary: BenchmarkSummary) -> str:
        """Generate comparison section with JavaScript baseline."""
        html = """
    <div class="comparison">
        <h2>Performance Comparison vs JavaScript Baseline</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Rust Time (ms)</th>
                <th>JS Baseline (ms)</th>
                <th>Speedup</th>
                <th>Rust Memory (MB)</th>
                <th>JS Memory (MB)</th>
                <th>Memory Efficiency</th>
            </tr>
        """
        
        comparison_data = summary.comparison_data
        
        for category, profile in summary.profiles.items():
            if category in comparison_data:
                js_time = comparison_data[category].get("avg_processing_time_ms", 0)
                js_memory = comparison_data[category].get("peak_memory_mb", 0)
                
                rust_time = profile.avg_processing_time
                rust_memory = profile.avg_memory_usage
                
                speedup = js_time / rust_time if rust_time > 0 else 0
                memory_efficiency = js_memory / rust_memory if rust_memory > 0 else 0
                
                html += f"""
            <tr>
                <td>{category.title()}</td>
                <td>{rust_time:.1f}</td>
                <td>{js_time:.1f}</td>
                <td>{speedup:.1f}x</td>
                <td>{rust_memory:.1f}</td>
                <td>{js_memory:.1f}</td>
                <td>{memory_efficiency:.1f}x</td>
            </tr>
                """
        
        html += "</table></div>"
        return html
    
    def _save_json_summary(self, summary: BenchmarkSummary, output_file: Path):
        """Save benchmark summary as JSON."""
        summary_dict = {
            "platform_info": summary.platform_info,
            "total_runtime_seconds": summary.total_runtime_seconds,
            "profiles": {}
        }
        
        for category, profile in summary.profiles.items():
            summary_dict["profiles"][category] = {
                "total_tests": profile.total_tests,
                "successful_tests": profile.successful_tests,
                "success_rate": profile.successful_tests / profile.total_tests if profile.total_tests > 0 else 0,
                "avg_processing_time_ms": profile.avg_processing_time,
                "median_processing_time_ms": profile.median_processing_time,
                "p95_processing_time_ms": profile.p95_processing_time,
                "avg_memory_usage_mb": profile.avg_memory_usage,
                "peak_memory_usage_mb": profile.peak_memory_usage,
                "avg_cpu_usage_percent": profile.avg_cpu_usage,
                "throughput_images_per_second": profile.throughput_images_per_second
            }
        
        if summary.comparison_data:
            summary_dict["comparison_data"] = summary.comparison_data
        
        with open(output_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="Run background removal performance benchmarks")
    parser.add_argument("--assets-dir", type=Path,
                       default=Path(__file__).parent.parent / "assets",
                       help="Test assets directory")
    parser.add_argument("--rust-binary", type=Path, required=True,
                       help="Path to Rust implementation binary")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations per test")
    parser.add_argument("--report-file", type=Path,
                       default=Path("benchmark_report.html"),
                       help="Output report file")
    
    args = parser.parse_args()
    
    if not args.rust_binary.exists():
        logger.error(f"Rust binary not found: {args.rust_binary}")
        return 1
    
    try:
        # Initialize benchmark runner
        benchmark = PerformanceBenchmark(args.assets_dir, args.rust_binary)
        
        # Run benchmarks
        summary = benchmark.run_comprehensive_benchmark(args.iterations)
        
        # Generate report
        benchmark.generate_report(summary, args.report_file)
        
        # Print summary
        total_successful = sum(p.successful_tests for p in summary.profiles.values())
        total_tests = sum(p.total_tests for p in summary.profiles.values())
        
        logger.info(f"Benchmark complete: {total_successful}/{total_tests} tests successful")
        logger.info(f"Total runtime: {summary.total_runtime_seconds:.1f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())