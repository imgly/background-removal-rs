#!/bin/bash

echo "🚀 bg-remove-web Demo Runner"
echo "============================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required to run the demo server"
    echo "   Please install Python 3 and try again"
    exit 1
fi

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is required to build the WASM module"
    echo ""
    echo "To install wasm-pack, run:"
    echo "  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    echo ""
    echo "Or on macOS with Homebrew:"
    echo "  brew install wasm-pack"
    exit 1
fi

echo "✅ Dependencies found"
echo ""

# Options
echo "Build options:"
echo "1) Build without embedded models (fast, ~3MB)"
echo "2) Build with ISNet FP16 model (slower, ~90MB)"
echo "3) Skip build (use existing build)"
echo ""
read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "🔨 Building WASM module without models..."
        wasm-pack build --target web --out-dir pkg --no-default-features
        ;;
    2)
        echo "🔨 Building WASM module with ISNet FP16..."
        echo "⚠️  Note: This requires the model file at ../../models/isnet/model_fp16.onnx"
        wasm-pack build --target web --out-dir pkg --features embed-isnet-fp16
        ;;
    3)
        echo "⏭️  Skipping build, using existing files..."
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Check if build succeeded
if [ ! -d "pkg" ]; then
    echo ""
    echo "❌ Build failed or pkg directory not found"
    echo ""
    echo "Common issues:"
    echo "- Model files not found: Use option 1 to build without models"
    echo "- WebP compilation errors: Make sure you have the latest code"
    echo ""
    exit 1
fi

echo ""
echo "✅ Build complete!"
echo ""
echo "🌐 Starting demo server..."
echo ""
echo "Demo will be available at:"
echo "  http://localhost:8000"
echo "  http://localhost:8000/test.html (simple test page)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd demo && python3 serve.py