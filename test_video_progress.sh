#!/bin/bash

# Test script for demonstrating video progress with --progress flag

echo "Building the project..."
cargo build --release

echo -e "\n=== Testing video processing with --progress flag ==="
echo "The progress bars will show:"
echo "1. Top bar: Overall progress (1 file)"
echo "2. Bottom bar: Video processing stages with spinner"
echo

# Check if a test video exists
if [ -f "test.mp4" ] || [ -f "test.mov" ]; then
    VIDEO_FILE=$(ls test.mp4 test.mov 2>/dev/null | head -1)
    echo "Using existing video file: $VIDEO_FILE"
elif command -v ffmpeg &> /dev/null; then
    echo "Creating a test video..."
    # Create a simple test video
    ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=30 -pix_fmt yuv420p test_video.mp4 -y 2>/dev/null
    VIDEO_FILE="test_video.mp4"
else
    echo "No test video found and ffmpeg not available to create one."
    echo "Please provide a test video file."
    exit 1
fi

# Run the CLI with --progress flag on video
echo -e "\nProcessing video with --progress flag..."
./target/release/imgly-bgremove --progress "$VIDEO_FILE" -o output_video.mp4

echo -e "\nTest completed!"