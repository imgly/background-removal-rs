#!/bin/bash

# Test script for demonstrating the --progress flag with stacked progress bars

echo "Building the project..."
cargo build --release

echo -e "\n=== Creating test images ==="
mkdir -p test_images
for i in {1..5}; do
    # Create a simple test image using ImageMagick convert if available, otherwise create empty files
    if command -v convert &> /dev/null; then
        convert -size 100x100 xc:blue "test_images/test_image_$i.png"
    else
        touch "test_images/test_image_$i.png"
    fi
done

echo -e "\n=== Testing batch processing with --progress flag ==="
echo "This will show two stacked progress bars:"
echo "1. Top bar: Total files progress"
echo "2. Bottom bar: Current file processing stages"
echo

# Run the CLI with --progress flag
./target/release/imgly-bgremove --progress test_images/*.png -o output_images/

echo -e "\n=== Testing single file with --progress flag ==="
./target/release/imgly-bgremove --progress test_images/test_image_1.png -o output_single.png

echo -e "\nTest completed!"