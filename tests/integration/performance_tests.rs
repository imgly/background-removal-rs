use std::time::{Duration, Instant};
use std::path::Path;
use std::process::Command;
use sysinfo::{System, SystemExt, ProcessExt, Pid};
use serde_json;
use std::collections::HashMap;

/// Performance measurement for a single test
#[derive(Debug, Clone)]
pub struct BenchmarkMeasurement {
    pub test_id: String,
    pub image_category: String,
    pub image_size: (u32, u32),
    pub processing_time: Duration,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Performance requirements from test configuration
#[derive(Debug, serde::Deserialize)]
pub struct PerformanceRequirements {
    pub memory_usage_max_mb: u64,
    pub cpu_usage_max_percent: f64,
    pub concurrent_processing_supported: bool,
    pub timeout_seconds: u64,
}

/// Benchmark results for a category or overall
#[derive(Debug)]
pub struct BenchmarkResults {
    pub measurements: Vec<BenchmarkMeasurement>,
    pub avg_processing_time: Duration,
    pub median_processing_time: Duration,
    pub p95_processing_time: Duration,
    pub avg_memory_usage: f64,
    pub peak_memory_usage: f64,
    pub avg_cpu_usage: f64,
    pub success_rate: f64,
    pub throughput_images_per_second: f64,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            avg_processing_time: Duration::ZERO,
            median_processing_time: Duration::ZERO,
            p95_processing_time: Duration::ZERO,
            avg_memory_usage: 0.0,
            peak_memory_usage: 0.0,
            avg_cpu_usage: 0.0,
            success_rate: 0.0,
            throughput_images_per_second: 0.0,
        }
    }
    
    pub fn add_measurement(&mut self, measurement: BenchmarkMeasurement) {
        self.measurements.push(measurement);
        self.calculate_statistics();
    }
    
    fn calculate_statistics(&mut self) {
        if self.measurements.is_empty() {
            return;
        }
        
        let successful_measurements: Vec<_> = self.measurements.iter()
            .filter(|m| m.success)
            .collect();
        
        if successful_measurements.is_empty() {
            self.success_rate = 0.0;
            return;
        }
        
        // Processing time statistics
        let mut processing_times: Vec<Duration> = successful_measurements.iter()
            .map(|m| m.processing_time)
            .collect();
        processing_times.sort();
        
        self.avg_processing_time = Duration::from_nanos(
            processing_times.iter().map(|d| d.as_nanos()).sum::<u128>() / processing_times.len() as u128
        );
        
        self.median_processing_time = if processing_times.len() % 2 == 0 {
            let mid = processing_times.len() / 2;
            Duration::from_nanos((processing_times[mid - 1].as_nanos() + processing_times[mid].as_nanos()) / 2)
        } else {
            processing_times[processing_times.len() / 2]
        };
        
        // 95th percentile
        let p95_index = ((processing_times.len() as f64) * 0.95) as usize;
        self.p95_processing_time = processing_times.get(p95_index.min(processing_times.len() - 1))
            .copied()
            .unwrap_or(Duration::ZERO);
        
        // Memory statistics
        self.avg_memory_usage = successful_measurements.iter()
            .map(|m| m.memory_usage_mb)
            .sum::<f64>() / successful_measurements.len() as f64;
        
        self.peak_memory_usage = successful_measurements.iter()
            .map(|m| m.memory_usage_mb)
            .fold(0.0, f64::max);
        
        // CPU statistics
        self.avg_cpu_usage = successful_measurements.iter()
            .map(|m| m.cpu_usage_percent)
            .sum::<f64>() / successful_measurements.len() as f64;
        
        // Success rate
        self.success_rate = successful_measurements.len() as f64 / self.measurements.len() as f64;
        
        // Throughput (images per second)
        if self.avg_processing_time.as_secs_f64() > 0.0 {
            self.throughput_images_per_second = 1.0 / self.avg_processing_time.as_secs_f64();
        }
    }
}

/// Benchmark a single image processing operation
pub fn benchmark_single_image<P: AsRef<Path>>(
    binary_path: P,
    input_image: P,
    output_image: P,
    timeout: Duration,
    additional_args: &[String],
) -> Result<BenchmarkMeasurement, Box<dyn std::error::Error>> {
    let binary_path = binary_path.as_ref();
    let input_image = input_image.as_ref();
    let output_image = output_image.as_ref();
    
    // Prepare command
    let mut cmd = Command::new(binary_path);
    cmd.arg(input_image)
       .arg("--output")
       .arg(output_image);
    
    // Add additional arguments
    for arg in additional_args {
        cmd.arg(arg);
    }
    
    // Initialize system monitoring
    let mut system = System::new_all();
    system.refresh_all();
    
    let start_time = Instant::now();
    let mut memory_samples = Vec::new();
    let mut cpu_samples = Vec::new();
    
    // Start the process
    let mut child = cmd.spawn()?;
    let child_pid = Pid::from(child.id() as i32);
    
    // Monitor resource usage
    let monitoring_thread = std::thread::spawn(move || {
        let mut local_system = System::new_all();
        let mut local_memory_samples = Vec::new();
        let mut local_cpu_samples = Vec::new();
        
        loop {
            local_system.refresh_process(child_pid);
            
            if let Some(process) = local_system.process(child_pid) {
                // Memory usage in MB
                let memory_mb = process.memory() as f64 / 1024.0 / 1024.0;
                local_memory_samples.push(memory_mb);
                
                // CPU usage percentage
                local_cpu_samples.push(process.cpu_usage() as f64);
            } else {
                // Process finished or not found
                break;
            }
            
            std::thread::sleep(Duration::from_millis(10)); // 10ms sampling rate
        }
        
        (local_memory_samples, local_cpu_samples)
    });
    
    // Wait for process with timeout
    let exit_status = match wait_with_timeout(&mut child, timeout) {
        Ok(status) => status,
        Err(e) => {
            let _ = child.kill();
            return Err(format!("Process timeout or error: {}", e).into());
        }
    };
    
    let processing_time = start_time.elapsed();
    
    // Get monitoring results
    let (memory_samples, cpu_samples) = monitoring_thread.join()
        .map_err(|_| "Failed to join monitoring thread")?;
    
    // Calculate statistics
    let peak_memory = memory_samples.iter().fold(0.0, |acc, &x| acc.max(x));
    let avg_cpu = if cpu_samples.is_empty() {
        0.0
    } else {
        cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64
    };
    
    let success = exit_status.success() && output_image.exists();
    let error_message = if !success {
        Some(format!("Exit code: {}", exit_status.code().unwrap_or(-1)))
    } else {
        None
    };
    
    // Get image dimensions if successful
    let image_size = if success {
        match image::image_dimensions(input_image) {
            Ok(dims) => dims,
            Err(_) => (0, 0),
        }
    } else {
        (0, 0)
    };
    
    Ok(BenchmarkMeasurement {
        test_id: input_image.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
        image_category: "unknown".to_string(), // Will be set by caller
        image_size,
        processing_time,
        memory_usage_mb: peak_memory,
        cpu_usage_percent: avg_cpu,
        success,
        error_message,
    })
}

/// Wait for process with timeout
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<std::process::ExitStatus, Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    loop {
        match child.try_wait()? {
            Some(status) => return Ok(status),
            None => {
                if start.elapsed() > timeout {
                    child.kill()?;
                    return Err("Process timeout".into());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Benchmark multiple images in parallel
pub fn benchmark_batch_processing<P: AsRef<Path>>(
    binary_path: P,
    input_images: &[P],
    output_dir: P,
    concurrency: usize,
    timeout_per_image: Duration,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let binary_path = binary_path.as_ref();
    let output_dir = output_dir.as_ref();
    
    // Ensure output directory exists
    std::fs::create_dir_all(output_dir)?;
    
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    
    // Process images in batches
    for chunk in input_images.chunks(concurrency) {
        for input_path in chunk {
            let input_path = input_path.as_ref().to_path_buf();
            let binary_path = binary_path.to_path_buf();
            let output_dir = output_dir.to_path_buf();
            let results_clone = Arc::clone(&results);
            
            let handle = thread::spawn(move || {
                let output_file = output_dir.join(format!(
                    "{}_output.png",
                    input_path.file_stem().unwrap().to_str().unwrap()
                ));
                
                match benchmark_single_image(
                    &binary_path,
                    &input_path,
                    &output_file,
                    timeout_per_image,
                    &[],
                ) {
                    Ok(measurement) => {
                        let mut results = results_clone.lock().unwrap();
                        results.push(measurement);
                    }
                    Err(e) => {
                        eprintln!("Failed to benchmark {}: {}", input_path.display(), e);
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for current batch to complete before starting next
        for handle in handles.drain(..) {
            handle.join().map_err(|_| "Thread join failed")?;
        }
    }
    
    // Calculate final results
    let measurements = Arc::try_unwrap(results)
        .map_err(|_| "Failed to unwrap results")?
        .into_inner()
        .unwrap();
    
    let mut benchmark_results = BenchmarkResults::new();
    for measurement in measurements {
        benchmark_results.add_measurement(measurement);
    }
    
    Ok(benchmark_results)
}

/// Benchmark processing speed across different image categories
pub fn benchmark_processing_speed<P: AsRef<Path>>(
    binary_path: P,
    test_images_dir: P,
    output_dir: P,
    categories: &[String],
) -> Result<HashMap<String, BenchmarkResults>, Box<dyn std::error::Error>> {
    let test_images_dir = test_images_dir.as_ref();
    let output_dir = output_dir.as_ref();
    let binary_path = binary_path.as_ref();
    
    let mut category_results = HashMap::new();
    
    for category in categories {
        println!("Benchmarking category: {}", category);
        
        let category_dir = test_images_dir.join(category);
        if !category_dir.exists() {
            eprintln!("Category directory not found: {:?}", category_dir);
            continue;
        }
        
        // Find all test images in category
        let test_images: Vec<_> = std::fs::read_dir(&category_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()?.to_str()? == "jpg" || path.extension()?.to_str()? == "png" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        
        if test_images.is_empty() {
            eprintln!("No test images found in category: {}", category);
            continue;
        }
        
        // Create category output directory
        let category_output_dir = output_dir.join(category);
        std::fs::create_dir_all(&category_output_dir)?;
        
        // Benchmark each image
        let mut results = BenchmarkResults::new();
        
        for test_image in &test_images {
            let output_file = category_output_dir.join(format!(
                "{}_output.png",
                test_image.file_stem().unwrap().to_str().unwrap()
            ));
            
            match benchmark_single_image(
                binary_path,
                test_image,
                &output_file,
                Duration::from_secs(30),
                &[],
            ) {
                Ok(mut measurement) => {
                    measurement.image_category = category.clone();
                    println!("  {}: {:.1}ms, {:.1}MB", 
                            measurement.test_id,
                            measurement.processing_time.as_millis(),
                            measurement.memory_usage_mb);
                    results.add_measurement(measurement);
                }
                Err(e) => {
                    eprintln!("Failed to benchmark {}: {}", test_image.display(), e);
                }
            }
        }
        
        category_results.insert(category.clone(), results);
    }
    
    Ok(category_results)
}

/// Benchmark memory usage patterns
pub fn benchmark_memory_usage<P: AsRef<Path>>(
    binary_path: P,
    test_images: &[P],
    output_dir: P,
) -> Result<Vec<BenchmarkMeasurement>, Box<dyn std::error::Error>> {
    let mut measurements = Vec::new();
    
    for test_image in test_images {
        let output_file = output_dir.as_ref().join(format!(
            "{}_memory_test.png",
            test_image.as_ref().file_stem().unwrap().to_str().unwrap()
        ));
        
        // Run with memory profiling enabled
        let measurement = benchmark_single_image(
            &binary_path,
            test_image,
            &output_file,
            Duration::from_secs(60),
            &["--profile-memory".to_string()], // Hypothetical flag for memory profiling
        )?;
        
        measurements.push(measurement);
    }
    
    Ok(measurements)
}

/// Compare performance against baseline
pub fn compare_against_baseline(
    current_results: &HashMap<String, BenchmarkResults>,
    baseline_file: &Path,
) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
    let baseline_content = std::fs::read_to_string(baseline_file)?;
    let baseline: serde_json::Value = serde_json::from_str(&baseline_content)?;
    
    let mut comparisons = HashMap::new();
    
    for (category, results) in current_results {
        if let Some(baseline_category) = baseline.get(category) {
            if let Some(baseline_time) = baseline_category.get("avg_processing_time_ms") {
                let baseline_time = baseline_time.as_f64().unwrap_or(0.0);
                let current_time = results.avg_processing_time.as_millis() as f64;
                
                if baseline_time > 0.0 {
                    let speedup = baseline_time / current_time;
                    comparisons.insert(category.clone(), speedup);
                }
            }
        }
    }
    
    Ok(comparisons)
}

/// Generate performance report
pub fn generate_performance_report(
    results: &HashMap<String, BenchmarkResults>,
    baseline_comparisons: Option<&HashMap<String, f64>>,
    output_file: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    
    let mut report = String::new();
    report.push_str("# Performance Benchmark Report\n\n");
    
    // Summary table
    report.push_str("## Summary\n\n");
    report.push_str("| Category | Success Rate | Avg Time (ms) | Median Time (ms) | P95 Time (ms) | Avg Memory (MB) | Peak Memory (MB) | Throughput (img/s) |\n");
    report.push_str("|----------|--------------|---------------|------------------|---------------|-----------------|------------------|--------------------|\n");
    
    for (category, result) in results {
        report.push_str(&format!(
            "| {} | {:.1}% | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |\n",
            category,
            result.success_rate * 100.0,
            result.avg_processing_time.as_millis(),
            result.median_processing_time.as_millis(),
            result.p95_processing_time.as_millis(),
            result.avg_memory_usage,
            result.peak_memory_usage,
            result.throughput_images_per_second
        ));
    }
    
    // Baseline comparisons
    if let Some(comparisons) = baseline_comparisons {
        report.push_str("\n## Baseline Comparisons\n\n");
        report.push_str("| Category | Speedup vs Baseline |\n");
        report.push_str("|----------|---------------------|\n");
        
        for (category, speedup) in comparisons {
            report.push_str(&format!("| {} | {:.1}x |\n", category, speedup));
        }
    }
    
    // Detailed results
    report.push_str("\n## Detailed Results\n\n");
    for (category, result) in results {
        report.push_str(&format!("### {}\n\n", category));
        report.push_str(&format!("- **Tests run:** {}\n", result.measurements.len()));
        report.push_str(&format!("- **Success rate:** {:.1}%\n", result.success_rate * 100.0));
        report.push_str(&format!("- **Average processing time:** {:.1}ms\n", result.avg_processing_time.as_millis()));
        report.push_str(&format!("- **Median processing time:** {:.1}ms\n", result.median_processing_time.as_millis()));
        report.push_str(&format!("- **95th percentile time:** {:.1}ms\n", result.p95_processing_time.as_millis()));
        report.push_str(&format!("- **Average memory usage:** {:.1}MB\n", result.avg_memory_usage));
        report.push_str(&format!("- **Peak memory usage:** {:.1}MB\n", result.peak_memory_usage));
        report.push_str(&format!("- **Throughput:** {:.2} images/second\n\n", result.throughput_images_per_second));
    }
    
    // Write report
    let mut file = std::fs::File::create(output_file)?;
    file.write_all(report.as_bytes())?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_benchmark_results_calculation() {
        let mut results = BenchmarkResults::new();
        
        // Add some test measurements
        results.add_measurement(BenchmarkMeasurement {
            test_id: "test1".to_string(),
            image_category: "test".to_string(),
            image_size: (1920, 1080),
            processing_time: Duration::from_millis(1000),
            memory_usage_mb: 150.0,
            cpu_usage_percent: 50.0,
            success: true,
            error_message: None,
        });
        
        results.add_measurement(BenchmarkMeasurement {
            test_id: "test2".to_string(),
            image_category: "test".to_string(),
            image_size: (1280, 720),
            processing_time: Duration::from_millis(500),
            memory_usage_mb: 100.0,
            cpu_usage_percent: 30.0,
            success: true,
            error_message: None,
        });
        
        // Check calculated statistics
        assert_eq!(results.success_rate, 1.0);
        assert_eq!(results.avg_processing_time, Duration::from_millis(750));
        assert_eq!(results.peak_memory_usage, 150.0);
        assert!(results.throughput_images_per_second > 0.0);
    }
    
    #[test]
    fn test_benchmark_with_failures() {
        let mut results = BenchmarkResults::new();
        
        results.add_measurement(BenchmarkMeasurement {
            test_id: "success".to_string(),
            image_category: "test".to_string(),
            image_size: (1920, 1080),
            processing_time: Duration::from_millis(1000),
            memory_usage_mb: 150.0,
            cpu_usage_percent: 50.0,
            success: true,
            error_message: None,
        });
        
        results.add_measurement(BenchmarkMeasurement {
            test_id: "failure".to_string(),
            image_category: "test".to_string(),
            image_size: (1920, 1080),
            processing_time: Duration::from_millis(0),
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            success: false,
            error_message: Some("Test error".to_string()),
        });
        
        // Success rate should be 50%
        assert_eq!(results.success_rate, 0.5);
        
        // Statistics should only consider successful measurements
        assert_eq!(results.avg_processing_time, Duration::from_millis(1000));
    }
}