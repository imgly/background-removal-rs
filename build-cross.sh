#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}🚀 Cross-platform build script for bg-remove${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] <target-triplet>"
    echo ""
    echo "Available target triplets:"
    echo -e "  ${GREEN}aarch64-unknown-linux-gnu${NC}   - ARM64 Linux (glibc)"
    echo -e "  ${GREEN}x86_64-unknown-linux-gnu${NC}    - x86_64 Linux (glibc)"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -l, --list              List available target triplets"
    echo "  -a, --all               Build all available targets"
    echo "  --clean                 Clean build artifacts before building"
    echo ""
    echo "Examples:"
    echo "  $0 x86_64-unknown-linux-gnu"
    echo "  $0 --all"
    echo "  $0 --clean x86_64-unknown-linux-gnu"
}

list_targets() {
    echo -e "${BLUE}📋 Available target triplets:${NC}"
    for dockerfile in machines/Dockerfile.*; do
        if [[ -f "$dockerfile" ]]; then
            target=$(basename "$dockerfile" | sed 's/Dockerfile\.//')
            echo -e "  ${GREEN}$target${NC}"
        fi
    done
}

build_target() {
    local target="$1"
    local dockerfile="machines/Dockerfile.$target"
    
    if [[ ! -f "$dockerfile" ]]; then
        echo -e "${RED}❌ Error: Dockerfile for target '$target' not found at $dockerfile${NC}"
        echo -e "${YELLOW}💡 Use '$0 --list' to see available targets${NC}"
        return 1
    fi
    
    echo -e "${BLUE}🐳 Building bg-remove for target: ${GREEN}$target${NC}"
    echo -e "${BLUE}📦 Using Dockerfile: ${GREEN}$dockerfile${NC}"
    
    # Create output directory
    local output_dir="target/$target"
    mkdir -p "$output_dir"
    
    # Build the Docker image
    echo -e "${BLUE}🔨 Building Docker image...${NC}"
    
    # Determine platform based on target
    local platform=""
    case "$target" in
        aarch64-*)
            platform="linux/arm64"
            ;;
        x86_64-*)
            platform="linux/amd64"
            ;;
        *)
            echo -e "${YELLOW}⚠️  Warning: Unknown architecture for target '$target', building for host platform${NC}"
            ;;
    esac
    
    # Build with platform specification if determined
    if [[ -n "$platform" ]]; then
        echo -e "${BLUE}🏗️  Building for platform: ${GREEN}$platform${NC}"
        docker build --platform "$platform" -f "$dockerfile" -t "bg-remove-$target" .
    else
        docker build -f "$dockerfile" -t "bg-remove-$target" .
    fi
    
    # Extract the binary from the container
    echo -e "${BLUE}📤 Extracting binary...${NC}"
    
    # Create a temporary container and copy the binary
    local container_id=$(docker create "bg-remove-$target")
    
    # Try different binary paths depending on target type
    if [[ "$target" == *"musl"* ]]; then
        # For musl targets, binary is at root
        docker cp "$container_id:/bg-remove" "$output_dir/bg-remove" 2>/dev/null || {
            echo -e "${RED}❌ Failed to extract binary from musl container${NC}"
            docker rm "$container_id"
            return 1
        }
    else
        # For glibc targets, binary is in /usr/local/bin
        docker cp "$container_id:/usr/local/bin/bg-remove" "$output_dir/bg-remove" 2>/dev/null || {
            echo -e "${RED}❌ Failed to extract binary from glibc container${NC}"
            docker rm "$container_id"
            return 1
        }
    fi
    
    docker rm "$container_id" > /dev/null
    
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}🔍 Binary info:${NC}"
    file "$output_dir/bg-remove"
    ls -lh "$output_dir/bg-remove"
    echo ""
    echo -e "${BLUE}📁 Output location: ${GREEN}$output_dir/bg-remove${NC}"
}

clean_artifacts() {
    echo -e "${YELLOW}🧹 Cleaning build artifacts...${NC}"
    rm -rf target/aarch64-unknown-linux-*
    rm -rf target/x86_64-unknown-linux-*
    echo -e "${GREEN}✅ Build artifacts cleaned${NC}"
}

# Parse command line arguments
CLEAN=false
BUILD_ALL=false
TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--list)
            list_targets
            exit 0
            ;;
        -a|--all)
            BUILD_ALL=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -*)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
        *)
            if [[ -n "$TARGET" ]]; then
                echo -e "${RED}❌ Multiple targets specified. Use --all to build all targets.${NC}"
                exit 1
            fi
            TARGET="$1"
            shift
            ;;
    esac
done

# Clean if requested
if [[ "$CLEAN" == true ]]; then
    clean_artifacts
fi

# Build logic
if [[ "$BUILD_ALL" == true ]]; then
    echo -e "${BLUE}🚀 Building all available targets...${NC}"
    echo ""
    
    success_count=0
    total_count=0
    failed_targets=()
    
    for dockerfile in machines/Dockerfile.*; do
        if [[ -f "$dockerfile" ]]; then
            target=$(basename "$dockerfile" | sed 's/Dockerfile\.//')
            echo -e "${YELLOW}📦 Building target: $target${NC}"
            echo "────────────────────────────────────────────────"
            
            total_count=$((total_count + 1))
            if build_target "$target"; then
                success_count=$((success_count + 1))
                echo -e "${GREEN}✅ Target $target completed successfully${NC}"
            else
                failed_targets+=("$target")
                echo -e "${RED}❌ Target $target failed${NC}"
            fi
            echo ""
        fi
    done
    
    echo "========================================"
    echo -e "${BLUE}📊 Build Summary:${NC}"
    echo -e "  ${GREEN}Successful: $success_count/$total_count${NC}"
    
    if [[ ${#failed_targets[@]} -gt 0 ]]; then
        echo -e "  ${RED}Failed: ${failed_targets[*]}${NC}"
        exit 1
    else
        echo -e "${GREEN}🎉 All targets built successfully!${NC}"
    fi
    
elif [[ -n "$TARGET" ]]; then
    build_target "$TARGET"
else
    echo -e "${RED}❌ No target specified${NC}"
    echo ""
    show_help
    exit 1
fi