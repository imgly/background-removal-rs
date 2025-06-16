# bg-remove-web

WebAssembly (WASM) bindings for background removal in web browsers using pure Rust inference.

## Testing the Web Version

### Prerequisites

1. **Install wasm-pack** (if not already installed):
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. **Python 3** for the demo server

### Build and Run

1. **Build the WASM module**:
   ```bash
   # From the bg-remove-web directory
   cd crates/bg-remove-web
   
   # Build without embedded models (for testing)
   wasm-pack build --target web --out-dir pkg --no-default-features
   
   # OR build with embedded ISNet model (larger bundle)
   # wasm-pack build --target web --out-dir pkg --features embed-isnet-fp16
   ```

2. **Start the demo server**:
   ```bash
   # From the bg-remove-web directory
   cd demo
   python3 serve.py
   ```

3. **Open in browser**:
   Navigate to `http://localhost:8000`

### Alternative: Using a Simple HTTP Server

If you prefer not to use the Python server:

```bash
# Using Node.js http-server
npx http-server demo -p 8000 --cors

# Using Python's built-in server
cd demo
python3 -m http.server 8000
```

### Testing Without Models

For quick testing without embedded models, you can:

1. Build with `--no-default-features`
2. The demo will load but won't be able to process images
3. You'll see the UI and can test the drag-and-drop functionality

### Troubleshooting

**Build Errors**:
- If you see WebP-related errors, ensure you're using the latest code
- Model embedding errors: Use `--no-default-features` for testing

**CORS Errors**:
- Use the provided `serve.py` script which sets proper CORS headers
- Or use a server that supports CORS headers for WASM

**Module Loading Errors**:
- Ensure the `pkg` directory was created by wasm-pack
- Check that paths in `demo/app.js` point to the correct locations

### Development Workflow

1. Make changes to Rust code in `src/lib.rs`
2. Rebuild: `wasm-pack build --target web --out-dir pkg`
3. Refresh browser to see changes

### Bundle Size

- Without models: ~2-3 MB
- With ISNet FP16: ~90 MB
- With ISNet FP32: ~175 MB

Choose the appropriate build based on your needs.