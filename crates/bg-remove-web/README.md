# bg-remove-web

WebAssembly (WASM) bindings for background removal in web browsers using pure Rust inference.

## Quick Start

### Prerequisites

1. **Install wasm-pack** (if not already installed):
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. **Node.js and npm** for the demo server

### Run the Demo

The simplest way to build and run the demo:

```bash
# From the bg-remove-web directory
cd crates/bg-remove-web
./run-demo.sh
```

This will:
- Build the WASM module with embedded ISNet FP16 model (84MB)
- Create a symlink from `demo/pkg` to the built `pkg` directory  
- Start a development server on `http://localhost:3000`

### Manual Build and Run

If you prefer manual control:

1. **Build the WASM module**:
   ```bash
   # Always builds with embedded ISNet FP16 model
   wasm-pack build --target web --features embed-isnet-fp16,console_error_panic_hook
   ```

2. **Start the demo server**:
   ```bash
   cd demo
   ln -s ../pkg pkg  # Create symlink to pkg directory
   npx serve -l 3000 -c serve.json  # Uses CORS configuration
   ```

3. **Open in browser**:
   Navigate to `http://localhost:3000/working-demo.html`

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

The demo uses embedded ISNet FP16 model:
- **WASM bundle**: ~96 MB (includes 84MB model)
- **First load**: ~30-60 seconds (model initialization)  
- **Subsequent loads**: Fast (cached by browser)

The large size is due to the embedded AI model, which enables offline background removal.