#!/bin/bash
# Pre-commit hook for zero-warning policy with automatic formatting
#
# To install this hook:
#   cp bin/pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Options:
#   Set PRE_COMMIT_AUTO_FORMAT=true to automatically format code
#   Set PRE_COMMIT_AUTO_FIX=true to automatically fix clippy issues

echo "🔍 Running pre-commit zero-warning checks..."

# Check for auto-format environment variable
if [ "${PRE_COMMIT_AUTO_FORMAT:-false}" = "true" ]; then
    echo "🎨 Auto-formatting code..."
    if ! cargo fmt --all; then
        echo "❌ Auto-formatting failed. Please check your code manually."
        exit 1
    fi
    echo "✅ Code formatted automatically."
fi

# Check for auto-fix environment variable  
if [ "${PRE_COMMIT_AUTO_FIX:-false}" = "true" ]; then
    echo "🔧 Auto-fixing clippy issues..."
    if cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged > /dev/null 2>&1; then
        echo "✅ Auto-fixes applied."
    else
        echo "⚠️ Some issues need manual fixing."
    fi
fi

# Run quick checks only (to keep commits fast)
if ! cargo check --workspace > /dev/null 2>&1; then
    echo "❌ Compilation failed. Please fix errors before committing."
    exit 1
fi

if ! cargo clippy --workspace --all-targets -- -D warnings > /dev/null 2>&1; then
    echo "❌ Clippy warnings detected. Please fix before committing."
    echo "💡 Run './bin/lint.sh --fix' to auto-fix issues"
    echo "💡 Run 'cargo clippy --workspace --all-targets' to see details"
    exit 1
fi

if ! cargo fmt --all -- --check > /dev/null 2>&1; then
    echo "❌ Code formatting issues detected."
    echo "💡 Run './bin/lint.sh --format' to format code"
    echo "💡 Run 'cargo fmt --all' to format manually"
    echo "💡 Set PRE_COMMIT_AUTO_FORMAT=true to format automatically"
    exit 1
fi

echo "✅ Pre-commit checks passed!"