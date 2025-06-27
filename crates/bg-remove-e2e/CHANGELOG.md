# Changelog

All notable changes to `bg-remove-e2e` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of bg-remove-e2e
- End-to-end testing framework
- Accuracy tests for model outputs
- Cross-backend compatibility tests
- Format preservation tests
- Visual report generation
- Performance benchmarks

## v0.1.0 (2025-06-27)

### New Features

 - <csr-id-23beffb36fbe868e8e24682114b60bc29ebccd1f/> comprehensive codebase cleanup and refactoring
   Remove unnecessary web components, examples, and documentation:
   - Delete bg-remove-web crate and all web-related files
   - Remove all core examples that are covered by e2e tests
   - Clean up documentation files and TODOs
   - Fix unified_processor_tests.rs to use proper backend injection pattern
   
   Core improvements:
   - Update unified_processor_tests to use test backend factory pattern
   - Fix MockBackend references to use proper test backend implementation
   - Ensure all tests pass with new backend injection architecture
   - Maintain cross-platform build support with machines/ directory
   
   🤖 Generated with [Claude Code](https://claude.ai/code)
 - <csr-id-3fd6e5c7da62dc0779103680f9f7dcc79ddd0952/> complete ICC color profile preservation implementation
   - Implement comprehensive color profile extraction and embedding
   - Add color profile test binary for validation
   - Update processor to handle color profiles in pipeline
   - Add ProfileExtractor and ProfileEmbedder integration
   - Support for JPEG, PNG, WebP, and TIFF format color profiles
   - Add graceful error handling for missing profiles
   - Update benchmarks and e2e test infrastructure
   - Clean up test artifacts and add development TODOs
   
   🤖 Generated with [Claude Code](https://claude.ai/code)
 - <csr-id-1e23d9b54d8e10b494d3ab5802270c16a462998d/> remove mock backend and add comprehensive benchmarking
   This commit completes the configuration simplification by removing the misleading mock backend
   and implementing realistic performance benchmarking with ONNX backend.
   
   ## Major Changes
   
   ### Mock Backend Removal
   - Removed `crates/bg-remove-core/src/backends/` directory entirely
   - Eliminated `BackendType::Mock` from processor configuration
   - Updated all references to use `BackendType::Onnx` for realistic processing
   - Fixed compilation errors in tests and examples
   
   ### Core Library Updates
   - Updated `process_image()` to use ONNX backend instead of mock (lib.rs:474)
   - Removed mock backend imports from `inference.rs`
   - Updated `DefaultBackendFactory` to remove mock backend support
   - Enhanced error messages with better context
   - Updated provider utilities to remove mock references
   
   ### Comprehensive Benchmarking
   - Moved `bg-remove-testing` to `bg-remove-e2e` for end-to-end testing
   - Created realistic ONNX backend benchmarks replacing 30ms mock results
   - Added batch processing benchmarks to demonstrate GPU acceleration benefits
   - Implemented provider comparison benchmarks (CPU, Auto, CoreML)
   - Added image size category benchmarks (portrait, product, complex)
   
   ## Performance Results
   
   ### Single Image Processing
   - **CPU**: 680-707ms (optimal for single images)
   - **Auto/CoreML**: 6.9-7.1s (includes model compilation overhead)
   
   ### Key Insights
   - Mock backend removal reveals real performance: 680ms-7s vs previous 30ms
   - GPU providers suffer from per-session compilation overhead
   - Batch processing essential for GPU acceleration benefits
   - CPU provider provides consistent sub-second performance
   
   ## Breaking Changes
   - `BackendType::Mock` enum variant removed
   - Mock backend no longer available in any configuration
   - All benchmarks now reflect realistic ONNX performance
   
   🤖 Generated with [Claude Code](https://claude.ai/code)

### Other

 - <csr-id-d8e44708654dfa5401ebdb4fab8747870b06d7d2/> integrate feat/readme-update with color profile implementation
   Merge the color profile implementation and README updates from feature branch.
   Resolved conflicts by keeping main branch license format and adding only
   useful binary tools (generate-report and test-color-profile).
   
   🤖 Generated with [Claude Code](https://claude.ai/code)

### Refactor

 - <csr-id-134acacb1becef4417988c3a0042aa495bda7115/> clean up testing architecture and remove redundant tools
   - Remove unimplemented binary stubs (benchmark-runner, download-images, validate-outputs)
   - Remove redundant test-suite CLI tool that duplicated cargo test functionality
   - Remove performance tests that belong in benchmarks instead
   - Keep only functional tests (accuracy, compatibility, format) and useful tools (generate-report)
   - Fix all e2e tests to use proper backend injection with remove_background_with_backend
   - Achieve 100% test pass rate (17/17 tests passing)
   
   Results in cleaner, more maintainable testing architecture following Rust conventions.
   
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

 - 8 commits contributed to the release over the course of 7 calendar days.
 - 6 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Bump bg-remove-core v0.1.0 (07b5ff4)
    - Migrate to positional inputs and HuggingFace model format (624861c)
    - Comprehensive codebase cleanup and refactoring (23beffb)
    - Integrate feat/readme-update with color profile implementation (d8e4470)
    - Complete ICC color profile preservation implementation (3fd6e5c)
    - Clean up testing architecture and remove redundant tools (134acac)
    - Merge branch 'feat/simplify-configuration' (3b26f20)
    - Remove mock backend and add comprehensive benchmarking (1e23d9b)
</details>

