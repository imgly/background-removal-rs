# Changelog

All notable changes to `bg-remove-tract` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of bg-remove-tract
- Pure Rust inference backend using Tract
- CPU-only deployment option
- No external dependencies
- Cross-platform compatibility

## v0.1.0 (2025-06-27)

### New Features

 - <csr-id-a7384e2b17b20dcf8711917dca61f9c2bb8e5456/> extract output format handling to service layer
   - Create OutputFormatHandler service separating format conversion logic from business logic
   - Update BackgroundRemovalProcessor to use OutputFormatHandler::convert_format()
   - Replace duplicated RGBA->RGB conversion logic with service method
   - Add comprehensive tests for format conversion, extension mapping, and transparency support
   - Export OutputFormatHandler in public API for reuse
   
   This continues Phase 2 business logic separation, reducing processor complexity
   and improving testability.
   
   🤖 Generated with [Claude Code](https://claude.ai/code)
 - <csr-id-410b84d2a40e302d92d35bc32a22aebfdae262c1/> implement conditional optimization for WASM vs native builds
   - Add conditional compilation for WASM target in Tract backend
   - Use .into_typed() for WASM builds instead of .into_optimized()
   - Native builds continue using .into_optimized() for performance
   - This addresses dimension conflict errors specific to WASM environment
   - Tract's optimization behaves differently in WASM vs native contexts
   
   Resolves the "Clashing resolution for expression. 1024=1024 \!= 682" error
   that occurs during WASM model initialization while maintaining full
   optimization for native CLI builds.
   
   🤖 Generated with [Claude Code](https://claude.ai/code)
 - <csr-id-1269008e7a202a4761838bc339059cfaf7810030/> simplify demo workflow with embedded model and npx serve
   - Streamline run-demo.sh to always build with embedded ISNet FP16 model
   - Replace Python server with npx serve for modern Node.js-based development
   - Add serve.json configuration for proper CORS headers and WASM MIME types
   - Create automatic symlink from demo/pkg to parent pkg directory
   - Update README with simplified quick start instructions
   - Remove serve.py in favor of npx serve with proper CORS configuration
   
   The demo now provides a one-command build and serve experience:
   ./run-demo.sh builds WASM with embedded model and starts server on localhost:3000
 - <csr-id-0d8638a5d8b39e82a3d98bb067b1f9d45ae3ff76/> implement pure Rust Tract backend with CLI integration
   Add a complete Tract backend implementation providing a pure Rust alternative to ONNX Runtime:
   
   ## Tract Backend Features
   - Pure Rust implementation using tract-onnx and tract-core
   - No external C++ dependencies or FFI boundaries
   - WebAssembly compatible for browser deployments
   - Apple Silicon optimizations (AMX, ARMv8.2 SIMD)
   - Complete InferenceBackend trait implementation
   
   ## CLI Integration
   - New execution provider format: tract:cpu or tract
   - Updated provider diagnostics showing both ONNX and Tract backends
   - Runtime backend selection based on execution provider string
   - Comprehensive help text and usage examples
   
   ## Architecture Changes
   - Added bg-remove-tract workspace member
   - Enhanced CLI with BackendType enum for runtime selection
   - Graceful error handling for provider diagnostics
   - Updated test coverage for new provider parsing
   
   ## Performance
   - Successfully tested with isnet-fp32 model (167.99 MB)
   - Model initialization: 243ms with Apple Silicon optimizations
   - Inference working correctly with detailed timing breakdowns
   - Memory safe inference without external runtime dependencies
   
   The Tract backend provides an ideal solution for deployments requiring:
   - Simple builds without external dependencies
   - WebAssembly compatibility
   - Enhanced memory safety
   - Faster compilation times
   
   🤖 Generated with [Claude Code](https://claude.ai/code)

### Bug Fixes

 - <csr-id-26302f252588f5fdee94d123d007d817222e5b2c/> implement WASM-compatible time handling and resolve build issues
   - Replace conditional std::time compilation with universal instant crate usage
   - Fix build script to use workspace root paths instead of fragile relative paths
   - Add explicit input shape specification to prevent Tract dimension conflicts
   - Update WebRemovalConfig to include all CLI configuration parameters
   - Create comprehensive working demo with embedded ISNet FP16 model support
   
   This resolves "time not implemented on this platform" errors and ensures
   both CLI and WASM builds work correctly with the same codebase.
   
   🤖 Generated with [Claude Code](https://claude.ai/code)

### New Features (BREAKING)

 - <csr-id-624861c5b965e5eb796f4e967980469e66586416/> migrate to positional inputs and HuggingFace model format
   - Switch ONNX Runtime from named to positional inputs using ort::inputs\! macro
   - Eliminates tensor name dependencies for more robust inference
   - Migrate all models from custom format to HuggingFace format
   - Add deprecation notices to get_input_name/get_output_name methods
   - Remove unused tensor name calls from Tract backend
   - Update build script to support HuggingFace config structure
   - Both ONNX and Tract backends now use positional inputs

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 9 commits contributed to the release over the course of 11 calendar days.
 - 6 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Bump bg-remove-core v0.1.0 (07b5ff4)
    - Migrate to positional inputs and HuggingFace model format (624861c)
    - Merge branch 'feat/simplify-configuration' (3b26f20)
    - Extract output format handling to service layer (a7384e2)
    - Implement conditional optimization for WASM vs native builds (410b84d)
    - Simplify demo workflow with embedded model and npx serve (1269008)
    - Implement WASM-compatible time handling and resolve build issues (26302f2)
    - Merge branch 'feat/tract-backend' (2c0c462)
    - Implement pure Rust Tract backend with CLI integration (0d8638a)
</details>

