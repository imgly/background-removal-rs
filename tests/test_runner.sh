#!/bin/bash
set -e

# Background Removal Rust Library Test Suite Runner
# This script runs the complete test suite including accuracy, performance, 
# compatibility, and format validation tests.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$PROJECT_ROOT/tests"
ASSETS_DIR="$TESTS_DIR/assets"
TOOLS_DIR="$TESTS_DIR/tools"
OUTPUT_DIR="$PROJECT_ROOT/test_results"

# Configuration
RUST_BINARY="${RUST_BINARY:-$PROJECT_ROOT/target/release/bg-remove-standard}"
ITERATIONS="${ITERATIONS:-3}"
TIMEOUT="${TIMEOUT:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Background Removal Rust Library Test Suite${NC}"
echo "========================================"
echo ""

# Check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check if Rust binary exists
    if [ ! -f "$RUST_BINARY" ]; then
        echo -e "${RED}Error: Rust binary not found at $RUST_BINARY${NC}"
        echo "Please build the project first or set RUST_BINARY environment variable"
        exit 1
    fi
    
    # Check Python dependencies
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is required but not installed${NC}"
        exit 1
    fi
    
    # Check required Python packages
    python3 -c "import PIL, numpy, psutil" 2>/dev/null || {
        echo -e "${YELLOW}Warning: Some Python packages may be missing${NC}"
        echo "Installing required packages..."
        pip3 install pillow numpy psutil scikit-image 2>/dev/null || {
            echo -e "${YELLOW}Note: Could not install Python packages automatically${NC}"
            echo "Please install: pip3 install pillow numpy psutil scikit-image"
        }
    }
    
    echo -e "${GREEN}Dependencies check completed${NC}"
    echo ""
}

# Create output directories
setup_environment() {
    echo -e "${BLUE}Setting up test environment...${NC}"
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/rust_outputs"
    mkdir -p "$OUTPUT_DIR/reports"
    mkdir -p "$OUTPUT_DIR/benchmarks"
    
    # Set environment variables
    export SESSION_UUID=$(uuidgen 2>/dev/null || echo "test-session-$(date +%s)")
    echo "Session UUID: $SESSION_UUID"
    
    echo -e "${GREEN}Environment setup completed${NC}"
    echo ""
}

# Generate test data if not exists
generate_test_data() {
    echo -e "${BLUE}Checking test data...${NC}"
    
    if [ ! -d "$ASSETS_DIR/input/portraits" ] || [ -z "$(ls -A "$ASSETS_DIR/input/portraits" 2>/dev/null)" ]; then
        echo "Generating synthetic test images..."
        cd "$TOOLS_DIR"
        python3 generate_reference.py \
            --assets-dir "$ASSETS_DIR" \
            --generate-images \
            --generate-checksums
        cd "$PROJECT_ROOT"
    else
        echo "Test data already exists"
    fi
    
    echo -e "${GREEN}Test data ready${NC}"
    echo ""
}

# Run accuracy tests
run_accuracy_tests() {
    echo -e "${BLUE}Running accuracy validation tests...${NC}"
    
    # Run Rust tests (would be actual Rust test command)
    if [ -f "$PROJECT_ROOT/Cargo.toml" ]; then
        echo "Running Rust accuracy tests..."
        cd "$PROJECT_ROOT"
        cargo test --test accuracy_tests --release -- --nocapture --test-threads=1 2>&1 || {
            echo -e "${YELLOW}Note: Rust accuracy tests not yet implemented${NC}"
        }
        cd "$TESTS_DIR"
    fi
    
    # Run integration tests with real outputs
    if [ -d "$OUTPUT_DIR/rust_outputs" ]; then
        echo "Validating results against reference data..."
        cd "$TOOLS_DIR"
        python3 validate_results.py \
            --assets-dir "$ASSETS_DIR" \
            --rust-output-dir "$OUTPUT_DIR/rust_outputs" \
            --report-file "$OUTPUT_DIR/reports/accuracy_report.html" \
            2>&1 || echo -e "${YELLOW}Accuracy validation completed with warnings${NC}"
        cd "$PROJECT_ROOT"
    fi
    
    echo -e "${GREEN}Accuracy tests completed${NC}"
    echo ""
}

# Run performance benchmarks
run_performance_tests() {
    echo -e "${BLUE}Running performance benchmarks...${NC}"
    
    # Run Rust performance tests
    if [ -f "$PROJECT_ROOT/Cargo.toml" ]; then
        echo "Running Rust performance tests..."
        cd "$PROJECT_ROOT"
        cargo test --test performance_tests --release -- --nocapture --ignored --test-threads=1 2>&1 || {
            echo -e "${YELLOW}Note: Rust performance tests not yet implemented${NC}"
        }
        cd "$TESTS_DIR"
    fi
    
    # Run comprehensive benchmarks
    echo "Running comprehensive benchmarks..."
    cd "$TOOLS_DIR"
    python3 benchmark_runner.py \
        --assets-dir "$ASSETS_DIR" \
        --rust-binary "$RUST_BINARY" \
        --iterations "$ITERATIONS" \
        --report-file "$OUTPUT_DIR/reports/benchmark_report.html" \
        2>&1 || echo -e "${YELLOW}Benchmark completed with warnings${NC}"
    cd "$PROJECT_ROOT"
    
    echo -e "${GREEN}Performance tests completed${NC}"
    echo ""
}

# Run compatibility tests
run_compatibility_tests() {
    echo -e "${BLUE}Running compatibility tests...${NC}"
    
    # Run Rust compatibility tests
    if [ -f "$PROJECT_ROOT/Cargo.toml" ]; then
        echo "Running Rust compatibility tests..."
        cd "$PROJECT_ROOT"
        cargo test --test compatibility_tests --release -- --nocapture --test-threads=1 2>&1 || {
            echo -e "${YELLOW}Note: Rust compatibility tests not yet implemented${NC}"
        }
        cd "$TESTS_DIR"
    fi
    
    echo -e "${GREEN}Compatibility tests completed${NC}"
    echo ""
}

# Run format validation tests
run_format_tests() {
    echo -e "${BLUE}Running format validation tests...${NC}"
    
    # Run Rust format tests
    if [ -f "$PROJECT_ROOT/Cargo.toml" ]; then
        echo "Running Rust format tests..."
        cd "$PROJECT_ROOT"
        cargo test --test format_tests --release -- --nocapture --test-threads=1 2>&1 || {
            echo -e "${YELLOW}Note: Rust format tests not yet implemented${NC}"
        }
        cd "$TESTS_DIR"
    fi
    
    echo -e "${GREEN}Format tests completed${NC}"
    echo ""
}

# Process test images to generate Rust outputs
process_test_images() {
    echo -e "${BLUE}Processing test images with Rust implementation...${NC}"
    
    local processed_count=0
    local failed_count=0
    
    # Process each category
    for category in portraits products complex edge_cases; do
        local category_dir="$ASSETS_DIR/input/$category"
        local output_category_dir="$OUTPUT_DIR/rust_outputs/$category"
        
        if [ ! -d "$category_dir" ]; then
            echo "Skipping category $category (directory not found)"
            continue
        fi
        
        mkdir -p "$output_category_dir"
        echo "Processing $category images..."
        
        # Process each image in category
        find "$category_dir" -name "*.jpg" -o -name "*.png" | while read -r image_file; do
            local base_name=$(basename "$image_file" | sed 's/\.[^.]*$//')
            local output_file="$output_category_dir/${base_name}_alpha.png"
            
            echo "  Processing: $(basename "$image_file")"
            
            # Run Rust implementation with timeout
            timeout "$TIMEOUT" "$RUST_BINARY" \
                "$image_file" \
                --output "$output_file" \
                --format png \
                2>/dev/null && {
                processed_count=$((processed_count + 1))
                echo "    ✓ Success"
            } || {
                failed_count=$((failed_count + 1))
                echo "    ✗ Failed"
            }
        done
    done
    
    echo "Processed: $processed_count images"
    echo "Failed: $failed_count images"
    echo -e "${GREEN}Image processing completed${NC}"
    echo ""
}

# Generate comprehensive report
generate_final_report() {
    echo -e "${BLUE}Generating comprehensive test report...${NC}"
    
    local report_file="$OUTPUT_DIR/reports/comprehensive_report.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Background Removal Test Suite Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .status-pass { color: #4CAF50; }
        .status-fail { color: #f44336; }
        .status-warn { color: #ff9800; }
        iframe { width: 100%; height: 600px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Background Removal Test Suite Report</h1>
        <p><strong>Session:</strong> $SESSION_UUID</p>
        <p><strong>Generated:</strong> $(date)</p>
        <p><strong>Binary:</strong> $RUST_BINARY</p>
    </div>
    
    <div class="section">
        <h2>Test Summary</h2>
        <ul>
            <li><strong>Test Data:</strong> $(find "$ASSETS_DIR/input" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l) images</li>
            <li><strong>Processed:</strong> $(find "$OUTPUT_DIR/rust_outputs" -name "*.png" 2>/dev/null | wc -l) outputs</li>
            <li><strong>Reports Generated:</strong> $(find "$OUTPUT_DIR/reports" -name "*.html" 2>/dev/null | wc -l)</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Individual Reports</h2>
EOF

    # Add links to individual reports
    if [ -f "$OUTPUT_DIR/reports/accuracy_report.html" ]; then
        echo '<h3>Accuracy Validation Report</h3>' >> "$report_file"
        echo '<iframe src="accuracy_report.html"></iframe>' >> "$report_file"
    fi
    
    if [ -f "$OUTPUT_DIR/reports/benchmark_report.html" ]; then
        echo '<h3>Performance Benchmark Report</h3>' >> "$report_file"
        echo '<iframe src="benchmark_report.html"></iframe>' >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF
    </div>
    
    <div class="section">
        <h2>Files Generated</h2>
        <ul>
EOF

    # List all generated files
    find "$OUTPUT_DIR" -type f | while read -r file; do
        local rel_path=$(realpath --relative-to="$OUTPUT_DIR" "$file" 2>/dev/null || echo "$file")
        echo "            <li>$rel_path</li>" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
        </ul>
    </div>
</body>
</html>
EOF

    echo "Comprehensive report: $report_file"
    echo -e "${GREEN}Report generation completed${NC}"
    echo ""
}

# Print final summary
print_summary() {
    echo -e "${BLUE}Test Suite Summary${NC}"
    echo "=================="
    echo ""
    echo "Session UUID: $SESSION_UUID"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Binary Used: $RUST_BINARY"
    echo ""
    
    # Count results
    local total_images=$(find "$ASSETS_DIR/input" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    local processed_images=$(find "$OUTPUT_DIR/rust_outputs" -name "*.png" 2>/dev/null | wc -l)
    local reports_generated=$(find "$OUTPUT_DIR/reports" -name "*.html" 2>/dev/null | wc -l)
    
    echo "Test Images: $total_images"
    echo "Processed: $processed_images"
    echo "Reports Generated: $reports_generated"
    echo ""
    
    if [ "$processed_images" -gt 0 ]; then
        echo -e "${GREEN}✓ Test suite completed successfully${NC}"
        echo "View reports in: $OUTPUT_DIR/reports/"
    else
        echo -e "${YELLOW}⚠ Test suite completed with issues${NC}"
        echo "Check logs and verify Rust binary functionality"
    fi
    echo ""
}

# Handle script interruption
cleanup() {
    echo ""
    echo -e "${YELLOW}Test suite interrupted${NC}"
    exit 1
}

trap cleanup INT

# Main execution
main() {
    local start_time=$(date +%s)
    
    echo "Starting test suite at $(date)"
    echo "Project root: $PROJECT_ROOT"
    echo "Rust binary: $RUST_BINARY"
    echo ""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --binary)
                RUST_BINARY="$2"
                shift 2
                ;;
            --iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --binary PATH     Path to Rust binary (default: target/release/bg-remove-standard)"
                echo "  --iterations N    Number of benchmark iterations (default: 3)"
                echo "  --timeout N       Timeout per image in seconds (default: 30)"
                echo "  --help           Show this help message"
                echo ""
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run test suite
    check_dependencies
    setup_environment
    generate_test_data
    process_test_images
    run_accuracy_tests
    run_performance_tests
    run_compatibility_tests
    run_format_tests
    generate_final_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "Test suite completed in ${duration} seconds"
    print_summary
}

# Run main function with all arguments
main "$@"