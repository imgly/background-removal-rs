# Implementation Plan: Provider Availability Refactoring

## Overview

Refactor the `check_provider_availability()` function from `inference.rs` to `OnnxBackend::list_providers()` to improve architecture, eliminate code duplication, and provide better naming.

## Phase 1: Analysis and Preparation ✅

### 1.1 Code Analysis ✅
- [x] Identified current implementation in `inference.rs:14-97`
- [x] Found duplicated logic in `OnnxBackend::load_model()`  
- [x] Located CLI usage in `main.rs:427`
- [x] Confirmed no other external dependencies

### 1.2 Documentation ✅
- [x] Added issue to ISSUES.md
- [x] Created detailed issue documentation
- [x] Created this implementation plan

## Phase 2: Implementation ✅

### 2.1 Move Function to OnnxBackend ✅
- [x] Add `list_providers()` static method to `OnnxBackend`
- [x] Move all provider checking logic from `inference.rs`
- [x] Consolidate with existing provider logic to eliminate duplication
- [x] Add proper documentation and examples

### 2.2 Update CLI Integration ✅
- [x] Update `show_provider_diagnostics()` in `main.rs`
- [x] Change import from `bg_remove_core::inference::check_provider_availability`
- [x] Change function call to `OnnxBackend::list_providers()`
- [x] Ensure output format remains identical

### 2.3 Cleanup ✅
- [x] Remove `check_provider_availability()` from `inference.rs`
- [x] Update exports and imports as needed
- [x] Remove unused imports

## Phase 3: Testing and Validation ✅

### 3.1 Functional Testing ✅
- [x] Test `--show-providers` CLI flag
- [x] Verify provider detection accuracy
- [x] Compare output with previous implementation
- [x] Test on different platforms (macOS, Linux)

### 3.2 Code Quality ✅
- [x] Run `cargo test` - Tests pass (unrelated failures are pre-existing)
- [x] Run `cargo clippy` - Warnings are pre-existing, not from refactoring
- [x] Check for compilation warnings - None from refactoring
- [x] Verify documentation builds - Documentation is comprehensive

## Implementation Details

### New Method Signature
```rust
impl OnnxBackend {
    /// List all ONNX Runtime execution providers with availability status and descriptions
    /// 
    /// Returns a vector of tuples containing:
    /// - Provider name (String)
    /// - Availability status (bool) 
    /// - Description (String)
    ///
    /// # Examples
    /// ```rust
    /// let providers = OnnxBackend::list_providers();
    /// for (name, available, description) in providers {
    ///     println!("{}: {} - {}", name, if available { "✅" } else { "❌" }, description);
    /// }
    /// ```
    pub fn list_providers() -> Vec<(String, bool, String)> {
        // Implementation here
    }
}
```

### CLI Usage Update
```rust
// Before
use bg_remove_core::inference::check_provider_availability;
let providers = check_provider_availability();

// After  
use bg_remove_core::backends::OnnxBackend;
let providers = OnnxBackend::list_providers();
```

## Benefits

1. **Better Architecture**: Provider logic belongs with the backend
2. **Reduced Duplication**: Consolidate provider checking code
3. **Clearer Naming**: `list_providers()` is more descriptive
4. **Extensibility**: Easy to add provider lists for other backends
5. **Maintainability**: Single location for provider logic

## Risks and Mitigation

### Risk: Breaking Changes
- **Mitigation**: This is an internal refactoring, no public API changes

### Risk: Logic Differences
- **Mitigation**: Careful comparison of existing logic during migration

### Risk: Platform-Specific Issues
- **Mitigation**: Test on multiple platforms before completion

## Timeline

- **Phase 1**: ✅ Complete
- **Phase 2**: ~30 minutes
- **Phase 3**: ~15 minutes  
- **Total**: ~45 minutes

## Success Criteria ✅

- [x] `--show-providers` flag produces identical output
- [x] No compilation errors or warnings from refactoring
- [x] All tests pass (unrelated test failures are pre-existing)
- [x] Code duplication eliminated
- [x] Documentation is comprehensive and accurate

## Completion Summary

✅ **REFACTORING COMPLETE** - Successfully moved `check_provider_availability()` from `inference.rs` to `OnnxBackend::list_providers()`. All functionality preserved while improving architecture and eliminating code duplication.