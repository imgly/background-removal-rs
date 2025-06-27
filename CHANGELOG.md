# Changelog

All notable changes to the bg_remove-rs workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial workspace setup with multiple crates
- Core library for background removal (bg-remove-core)
- ONNX Runtime backend (bg-remove-onnx)
- Tract pure-Rust backend (bg-remove-tract)
- Command-line interface (bg-remove-cli)
- End-to-end testing framework (bg-remove-e2e)
- Support for ISNet and BiRefNet models
- Hardware acceleration capabilities
- Comprehensive testing and benchmarking tools

### Workspace Crates
- `bg-remove-core`: Core library with unified processor API
- `bg-remove-onnx`: ONNX Runtime inference backend
- `bg-remove-tract`: Pure Rust inference backend
- `bg-remove-cli`: Command-line tool
- `bg-remove-e2e`: Testing and benchmarking framework