#!/bin/bash
# Quick formatting script for bg-remove-rs
#
# This script provides quick access to common formatting operations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo "Code Formatting Script for bg-remove-rs"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  format, fmt        Format all Rust code"
    echo "  check              Check formatting without changing files"
    echo "  fix                Format code and auto-fix Clippy issues"
    echo "  imports            Organize imports only (requires nightly)"
    echo "  diff               Show formatting differences"
    echo "  config             Show current rustfmt configuration"
    echo "  help, -h, --help   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 format          # Format all code"
    echo "  $0 check           # Check if code is formatted"
    echo "  $0 fix             # Format and fix issues"
    echo "  $0 diff            # Show what would change"
}

format_code() {
    echo -e "${BLUE}🎨 Formatting Rust code...${NC}"
    if cargo fmt --all; then
        echo -e "${GREEN}✅ Code formatted successfully${NC}"
    else
        echo -e "${RED}❌ Formatting failed${NC}"
        exit 1
    fi
}

check_formatting() {
    echo -e "${BLUE}🔍 Checking code formatting...${NC}"
    if cargo fmt --all -- --check; then
        echo -e "${GREEN}✅ All code is properly formatted${NC}"
    else
        echo -e "${RED}❌ Code formatting issues found${NC}"
        echo -e "${YELLOW}💡 Run '$0 format' to fix formatting${NC}"
        exit 1
    fi
}

fix_all() {
    echo -e "${BLUE}🔧 Formatting and fixing issues...${NC}"
    
    # Format first
    format_code
    
    # Then fix Clippy issues
    echo -e "${BLUE}🔧 Auto-fixing Clippy issues...${NC}"
    if cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged; then
        echo -e "${GREEN}✅ Auto-fixes applied${NC}"
    else
        echo -e "${YELLOW}⚠️ Some issues need manual fixing${NC}"
    fi
}

show_diff() {
    echo -e "${BLUE}📋 Showing formatting differences...${NC}"
    
    # Create a temporary directory for the diff
    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    # Copy current code to temp directory
    cp -r . "$temp_dir/original"
    cp -r . "$temp_dir/formatted"
    
    # Format the copy
    cd "$temp_dir/formatted"
    cargo fmt --all >/dev/null 2>&1 || true
    cd - >/dev/null
    
    # Show differences
    if diff -r "$temp_dir/original" "$temp_dir/formatted" --exclude target --exclude .git; then
        echo -e "${GREEN}✅ No formatting changes needed${NC}"
    else
        echo -e "${YELLOW}📝 The above changes would be applied${NC}"
    fi
}

organize_imports() {
    echo -e "${BLUE}📦 Organizing imports...${NC}"
    echo -e "${YELLOW}⚠️ This requires nightly Rust for some features${NC}"
    
    if cargo +nightly fmt --all; then
        echo -e "${GREEN}✅ Imports organized${NC}"
    else
        echo -e "${YELLOW}⚠️ Using stable rustfmt instead${NC}"
        cargo fmt --all
    fi
}

show_config() {
    echo -e "${BLUE}⚙️ Current rustfmt configuration:${NC}"
    echo ""
    if [ -f "rustfmt.toml" ]; then
        echo -e "${GREEN}📄 rustfmt.toml configuration:${NC}"
        cat rustfmt.toml
        echo ""
        echo -e "${BLUE}🔧 Effective configuration:${NC}"
        rustfmt --print-config current_dir
    else
        echo -e "${RED}❌ No rustfmt.toml found${NC}"
        echo -e "${BLUE}🔧 Default configuration:${NC}"
        rustfmt --print-config minimal
    fi
}

# Parse command line arguments
case "${1:-format}" in
    format|fmt)
        format_code
        ;;
    check)
        check_formatting
        ;;
    fix)
        fix_all
        ;;
    imports)
        organize_imports
        ;;
    diff)
        show_diff
        ;;
    config)
        show_config
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac