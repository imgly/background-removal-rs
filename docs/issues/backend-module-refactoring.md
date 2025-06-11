# Backend Module Refactoring Implementation Plan

## Overview

This document outlines the implementation plan for extracting InferenceBackend implementations into separate files. Currently, all backend implementations are located in a single `inference.rs` file. We want to extract just the backend implementations into their own module for better organization.

## Current State Analysis

### Existing Backend Implementations

**Location:** `crates/bg-remove-core/src/inference.rs` (423 lines)

1. **InferenceBackend Trait** (lines 48-60)
   - Core interface: `initialize()`, `infer()`, `input_shape()`, `output_shape()`
   - Well-defined abstraction for all backends

2. **OnnxBackend Struct** (lines 63-289) 
   - Production ONNX Runtime backend
   - Execution provider support (CPU, CUDA, CoreML)
   - Threading configuration
   - Model loading via ModelManager

3. **MockBackend Struct** (lines 292-356)
   - Testing/debugging backend with edge detection mock
   - Used when `config.debug = true`

4. **BackendRegistry Struct** (lines 359-389)
   - Factory pattern for backend management
   - Pre-registers "onnx" and "mock" backends

### Dependencies

- `config.rs` - ExecutionProvider enum, RemovalConfig
- `models.rs` - ModelManager, ModelProvider trait  
- `error.rs` - BgRemovalError types
- External: `ort`, `ndarray`

## Target Structure

```
crates/bg-remove-core/src/
├── lib.rs                       (unchanged)
├── inference.rs                 (trait, registry, utils - keep main logic)
├── backends/
│   ├── mod.rs                  (backend module exports)
│   ├── onnx.rs                 (OnnxBackend implementation)
│   └── mock.rs                 (MockBackend implementation)
├── config.rs                   (unchanged)
├── models.rs                   (unchanged)
├── image_processing.rs         (unchanged)
├── types.rs                    (unchanged)
└── error.rs                    (unchanged)
```

## Implementation Steps

### Phase 1: Create Backend Module (15 minutes)

**Task 1.1: Create backends directory and module files**
```bash
mkdir crates/bg-remove-core/src/backends
touch crates/bg-remove-core/src/backends/mod.rs
touch crates/bg-remove-core/src/backends/onnx.rs
touch crates/bg-remove-core/src/backends/mock.rs
```

### Phase 2: Extract Backend Implementations (30 minutes)

**Task 2.1: Extract ONNX backend**
- Move `OnnxBackend` struct and impl blocks to `backends/onnx.rs`
- Include execution provider configuration logic
- Maintain ModelManager integration
- Preserve threading configuration

**Task 2.2: Extract Mock backend**
- Move `MockBackend` struct and impl to `backends/mock.rs`
- Keep edge detection mock functionality
- Preserve debug mode integration

### Phase 3: Update Module Structure (15 minutes)

**Task 3.1: Configure backends module**
- Set up `backends/mod.rs` with proper exports
- Add backend module to `lib.rs`

**Task 3.2: Update inference.rs**
- Remove backend implementations from `inference.rs`
- Keep InferenceBackend trait, BackendRegistry, and utilities
- Add imports from backends module

### Phase 4: Testing and Validation (15 minutes)

**Task 4.1: Validate existing functionality**
- Run all existing tests to ensure no regressions
- Test CLI functionality with different backends
- Validate execution provider selection
- Check model loading and inference pipeline

**Task 4.2: Update documentation**
- Add module-level documentation to backend files
- Ensure imports are properly documented

## Implementation Details

### File Content Structure

**`backends/mod.rs`:**
```rust
//! Backend implementations for inference

pub mod mock;
pub mod onnx;

// Re-exports for easy access
pub use self::mock::MockBackend;
pub use self::onnx::OnnxBackend;
```

**Updated `inference.rs`:**
- Keep InferenceBackend trait definition
- Keep BackendRegistry implementation  
- Keep check_provider_availability function
- Add: `use crate::backends::{MockBackend, OnnxBackend};`

### Backward Compatibility Strategy

1. **Public API preservation** - All current public exports remain available
2. **Import path compatibility** - Existing code continues to work
3. **Behavior preservation** - No functional changes to backend behavior
4. **Configuration compatibility** - All existing config options work

### Benefits of Refactoring

1. **Separation of Concerns** - Each backend is self-contained in its own file
2. **Improved Maintainability** - Smaller, focused files are easier to understand and modify
3. **Enhanced Extensibility** - Adding new backends (TensorRT, DirectML, Candle) becomes straightforward
4. **Better Testing** - Module-specific tests improve coverage and clarity
5. **Plugin Architecture** - Registry pattern enables runtime backend discovery
6. **Code Organization** - Clear module hierarchy makes navigation easier

### Risk Mitigation

1. **Incremental approach** - Move one component at a time
2. **Comprehensive testing** - Validate each step with existing test suite
3. **Backward compatibility** - Maintain all existing API entry points
4. **Documentation updates** - Keep docs synchronized with changes

## Future Extensions Enabled

This refactoring enables easy addition of:

1. **Additional backends**: TensorRT, DirectML, Candle, WGPU
2. **Backend plugins**: Runtime loading of backend implementations
3. **Backend-specific optimizations**: Per-backend configuration options
4. **Performance monitoring**: Backend-specific metrics and profiling
5. **Dynamic backend selection**: Automatic backend selection based on hardware

## Acceptance Criteria

- [ ] All existing tests pass without modification
- [ ] CLI functionality unchanged for end users  
- [ ] Public API maintains backward compatibility
- [ ] New module structure is properly documented
- [ ] Each backend can be tested independently
- [ ] Registry supports adding new backends
- [ ] Code follows project formatting standards
- [ ] Zero-warning policy maintained

## Estimated Timeline

**Total time:** ~75 minutes
- Phase 1: 15 minutes
- Phase 2: 30 minutes  
- Phase 3: 15 minutes
- Phase 4: 15 minutes

This simple refactoring extracts backend implementations for better organization while keeping the main inference logic together.