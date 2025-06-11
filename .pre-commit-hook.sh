#!/bin/bash
# Pre-commit hook for zero-warning policy
#
# To install this hook:
#   cp .pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit

echo "🔍 Running pre-commit zero-warning checks..."

# Run quick checks only (to keep commits fast)
if ! cargo check --workspace > /dev/null 2>&1; then
    echo "❌ Compilation failed. Please fix errors before committing."
    exit 1
fi

if ! cargo clippy --workspace --all-targets -- -D warnings > /dev/null 2>&1; then
    echo "❌ Clippy warnings detected. Please fix before committing."
    echo "💡 Run 'cargo clippy --workspace --all-targets' to see details"
    exit 1
fi

if ! cargo fmt --all -- --check > /dev/null 2>&1; then
    echo "❌ Code formatting issues detected. Please run 'cargo fmt' before committing."
    exit 1
fi

echo "✅ Pre-commit checks passed!"