#!/bin/bash

# Build the Docker image
docker build -t claude-dev .

# Check if we're in an interactive terminal
if [ -t 0 ] && [ -t 1 ]; then
    # Run the container with volume mounts in interactive mode
    docker run -it \
      -v ~/.claude:/root/.claude \
      -v $(pwd):/app \
      claude-dev
else
    echo "Error: This script must be run in an interactive terminal"
    exit 1
fi