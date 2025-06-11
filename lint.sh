#!/bin/bash
# Zero-Warning Policy Lint Check Script
#
# This script enforces the zero-warning policy across all compilation targets
# in the workspace, ensuring code quality and consistency.

set -e

echo "🔍 Running Zero-Warning Policy Lint Checks"
echo "==========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a check and report status
run_check() {
    local description="$1"
    local command="$2"
    
    echo -n "⏳ $description... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "   Command: $command"
        eval "$command" 2>&1 | sed 's/^/   /'
        return 1
    fi
}

# Track overall success
overall_success=true

echo ""
echo "📦 Workspace-wide checks:"

# Check main build (library + CLI)
if ! run_check "Library compilation (dev)" "cargo check --workspace"; then
    overall_success=false
fi

if ! run_check "Library compilation (release)" "cargo check --workspace --release"; then
    overall_success=false
fi

# Check all targets (including examples, benches, tests)
if ! run_check "All targets compilation" "cargo check --workspace --all-targets"; then
    overall_success=false
fi

# Run tests
if ! run_check "Unit tests" "cargo test --workspace"; then
    overall_success=false
fi

# Check different feature combinations
echo ""
echo "🎛️  Feature configuration checks:"

if ! run_check "FP16 model (default)" "cargo check --workspace --features fp16-model"; then
    overall_success=false
fi

if ! run_check "FP32 model" "cargo check --workspace --features fp32-model"; then
    overall_success=false
fi

# Run Clippy with workspace lints
echo ""
echo "📎 Clippy linting:"

if ! run_check "Clippy warnings" "cargo clippy --workspace --all-targets -- -D warnings"; then
    overall_success=false
fi

# Check documentation
echo ""
echo "📚 Documentation checks:"

if ! run_check "Doc generation" "cargo doc --workspace --no-deps"; then
    overall_success=false
fi

# Format check
echo ""
echo "🎨 Code formatting:"

if ! run_check "Code formatting" "cargo fmt --all -- --check"; then
    overall_success=false
fi

# Final result
echo ""
echo "=========================================="
if [ "$overall_success" = true ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED! Zero-warning policy maintained.${NC}"
    exit 0
else
    echo -e "${RED}❌ CHECKS FAILED! Please fix the issues above.${NC}"
    echo ""
    echo "💡 Common fixes:"
    echo "   • Run 'cargo clippy --fix' to auto-fix some issues"
    echo "   • Run 'cargo fmt' to fix formatting"
    echo "   • Add #[allow(dead_code)] for intentionally unused items"
    echo "   • Remove unused imports and variables"
    exit 1
fi