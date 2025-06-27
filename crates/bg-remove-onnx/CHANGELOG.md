# Changelog

All notable changes to `bg-remove-onnx` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of bg-remove-onnx
- ONNX Runtime inference backend implementation
- Hardware acceleration support (CUDA, CoreML, DirectML)
- Dynamic execution provider selection
- Optimized tensor operations

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

### Refactor

 - <csr-id-abda84cc9b3399a42aed0aac3876c76cbbecf1a9/> move ONNX backend into separate crate
   - Create new bg-remove-onnx crate with ONNX Runtime implementation
   - Remove ONNX dependencies from core crate for better modularity
   - Update CLI crate to use bg-remove-onnx for provider diagnostics
   - Add error conversion from ort::Error to BgRemovalError::Inference
   - Update workspace to include new bg-remove-onnx member
   - Maintain backward compatibility for CLI functionality
   - Core crate now focuses on abstractions and mock backend
   - Enables future addition of alternative inference backends (Candle, Tract, WASI-NN)

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

 - 6 commits contributed to the release over the course of 11 calendar days.
 - 3 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Bump bg-remove-core v0.1.0 (07b5ff4)
    - Migrate to positional inputs and HuggingFace model format (624861c)
    - Merge branch 'feat/simplify-configuration' (3b26f20)
    - Extract output format handling to service layer (a7384e2)
    - Merge feat/onnx-backend-crate: Implement modular ONNX backend architecture (14974bd)
    - Move ONNX backend into separate crate (abda84c)
</details>

