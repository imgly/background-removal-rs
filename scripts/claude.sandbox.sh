#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}🐳 Claude Sandbox for bg-remove${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --no-auth-check         Skip Claude authentication verification"
    echo "  --api-key KEY           Use specific API key (overrides environment)"
    echo "  --rebuild               Force rebuild of Docker image"
    echo "  --dry-run               Show what would be executed without running"
    echo ""
    echo "Environment Variables:"
    echo "  ANTHROPIC_API_KEY       Your Anthropic API key"
    echo "  CLAUDE_CONFIG_DIR       Custom Claude config directory (default: ~/.claude)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Standard run with authentication check"
    echo "  $0 --no-auth-check      # Skip authentication verification"
    echo "  $0 --rebuild            # Force rebuild Docker image and run"
}

check_claude_auth() {
    local skip_check="$1"
    
    if [[ "$skip_check" == true ]]; then
        echo -e "${YELLOW}⚠️  Skipping Claude authentication check${NC}"
        return 0
    fi
    
    echo -e "${BLUE}🔐 Checking Claude authentication...${NC}"
    
    # Check if Claude CLI is available
    if ! command -v claude &> /dev/null; then
        echo -e "${RED}❌ Claude CLI not found. Please install Claude Code first.${NC}"
        echo -e "${YELLOW}💡 Visit: https://claude.ai/download${NC}"
        return 1
    fi
    
    # Check if Claude is logged in
    if claude status &>/dev/null; then
        echo -e "${GREEN}✅ Claude authentication verified${NC}"
        return 0
    else
        echo -e "${RED}❌ Claude Code is not logged in on the host system${NC}"
        echo -e "${YELLOW}💡 Please run 'claude login' first to authenticate with Claude Pro${NC}"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
        echo -e "${YELLOW}⚠️  Continuing without verified authentication${NC}"
        return 0
    fi
}

check_environment() {
    echo -e "${BLUE}🌍 Checking environment variables...${NC}"
    
    # Check for API key
    if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
        echo -e "${GREEN}✅ ANTHROPIC_API_KEY found in environment${NC}"
    else
        echo -e "${YELLOW}⚠️  ANTHROPIC_API_KEY not found in environment${NC}"
        echo -e "${BLUE}ℹ️  Will rely on Claude CLI authentication${NC}"
    fi
    
    # Check Claude config directory
    local claude_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
    if [[ -d "$claude_dir" ]]; then
        echo -e "${GREEN}✅ Claude config directory found: $claude_dir${NC}"
    else
        echo -e "${YELLOW}⚠️  Claude config directory not found: $claude_dir${NC}"
    fi
}

build_docker_image() {
    local rebuild="$1"
    
    echo -e "${BLUE}🐳 Checking Docker image...${NC}"
    
    # Check if image exists and if rebuild is requested
    if [[ "$rebuild" == true ]] || ! docker image inspect bg-remove-claude &>/dev/null; then
        if [[ "$rebuild" == true ]]; then
            echo -e "${BLUE}🔨 Force rebuilding Docker image (ignoring cache)...${NC}"
            docker build --no-cache -t bg-remove-claude .
        else
            echo -e "${BLUE}🔨 Building Docker image...${NC}"
            docker build -t bg-remove-claude .
        fi
        echo -e "${GREEN}✅ Docker image built successfully${NC}"
    else
        echo -e "${GREEN}✅ Docker image already exists${NC}"
    fi
}

run_container() {
    local api_key="$1"
    local dry_run="$2"
    
    # Set up volume mounts
    local claude_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
    local project_dir="$(pwd)"
    
    # Build docker run command
    local docker_cmd=(
        "docker" "run" "-it"
        "-v" "$claude_dir:/home/claude/.claude"
        "-v" "$project_dir:/home/claude/app"
        "-e" "ANTHROPIC_API_KEY=${api_key}"
        "bg-remove-claude"
    )
    
    if [[ "$dry_run" == true ]]; then
        echo -e "${BLUE}🔍 Dry run - would execute:${NC}"
        echo "${docker_cmd[*]}"
        return 0
    fi
    
    echo -e "${BLUE}🚀 Starting Claude sandbox container...${NC}"
    echo -e "${BLUE}📁 Project directory: $project_dir${NC}"
    echo -e "${BLUE}🔧 Claude config: $claude_dir${NC}"
    echo ""
    
    # Check if we're in an interactive terminal
    if [ -t 0 ] && [ -t 1 ]; then
        "${docker_cmd[@]}"
    else
        echo -e "${RED}❌ Error: This script must be run in an interactive terminal${NC}"
        return 1
    fi
}

# Parse command line arguments
SKIP_AUTH_CHECK=false
REBUILD_IMAGE=false
DRY_RUN=false
CUSTOM_API_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-auth-check)
            SKIP_AUTH_CHECK=true
            shift
            ;;
        --api-key)
            CUSTOM_API_KEY="$2"
            shift 2
            ;;
        --rebuild)
            REBUILD_IMAGE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${BLUE}🚀 Claude Sandbox for bg-remove${NC}"
echo ""

# Check environment variables
check_environment

# Check Claude authentication
if ! check_claude_auth "$SKIP_AUTH_CHECK"; then
    exit 1
fi

# Build Docker image
build_docker_image "$REBUILD_IMAGE"

# Determine API key to use
API_KEY="${CUSTOM_API_KEY:-${ANTHROPIC_API_KEY}}"

# Run the container
if ! run_container "$API_KEY" "$DRY_RUN"; then
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Claude sandbox session completed${NC}"