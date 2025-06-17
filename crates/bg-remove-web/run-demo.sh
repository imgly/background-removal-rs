#!/bin/bash

echo "🚀 bg-remove-web Demo Runner"
echo "============================"
echo ""

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

# Check for npx/npm
if ! command -v npx &> /dev/null; then
    echo "❌ npx is required to run the demo server"
    echo "   Please install Node.js and npm"
    exit 1
fi

echo "✅ Dependencies found"
echo ""

# Always build with embedded ISNet FP16 model
echo "🔨 Building WASM module with embedded ISNet FP16 model..."
echo "⏳ This may take a minute as it embeds the 84MB model..."
echo ""

# Build the WASM module
if ! wasm-pack build --target web --out-dir pkg --features embed-isnet-fp16,console_error_panic_hook; then
    echo ""
    echo "❌ Build failed!"
    echo ""
    echo "Common issues:"
    echo "- Model file not found at ../../models/isnet/model_fp16.onnx"
    echo "- Rust/wasm-pack not properly installed"
    echo ""
    exit 1
fi

echo ""
echo "✅ Build complete!"
echo ""

# Ensure demo/pkg is symlinked to the built pkg directory
echo "🔗 Setting up pkg symlink in demo directory..."
cd demo
if [ -L "pkg" ] || [ -d "pkg" ]; then
    rm -rf pkg
fi
ln -s ../pkg pkg

echo ""
echo "🌐 Starting demo server with npx serve..."
echo ""
echo "Demo will be available at:"
echo "  http://localhost:3000"
echo "  http://localhost:3000/working-demo.html (full featured demo)"
echo "  http://localhost:3000/test.html (simple test page)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server with npx serve (using serve.json for CORS headers)
npx serve -l 3000 -c serve.json