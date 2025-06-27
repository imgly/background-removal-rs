# Package Consolidation Implementation Plan

## Goal
Consolidate all workspace crates (`bg-remove-core`, `bg-remove-onnx`, `bg-remove-tract`, `bg-remove-cli`, `bg-remove-e2e`) into a single `imgly-bgremove` package.

## Requirements
- **Package name**: `imgly-bgremove`
- **Backend handling**: Feature flags with `onnx` default, but include all backends by default
- **CLI integration**: Always include CLI functionality (integrated)
- **Testing**: Move e2e tests to dev-dependencies in main crate

## Implementation Steps

### Phase 1: Create New Consolidated Crate Structure ✅ COMPLETED
- [x] Create new `imgly-bgremove` crate directory
- [x] Set up Cargo.toml with appropriate feature flags
- [x] Define feature structure:
  - `default = ["onnx", "tract", "cli", "webp-support", "embed-isnet-fp32"]`
  - `onnx` - ONNX Runtime backend
  - `tract` - Pure Rust backend
  - `cli` - Command-line interface
  - Individual model embedding features
  - WebP support and convenience feature groups

### Phase 2: Consolidate Core Functionality ✅ COMPLETED
- [x] Move `bg-remove-core` source code to main lib
- [x] Integrate core traits and types as primary API
- [x] Preserve existing public API structure
- [x] Update internal module organization
- [x] Fix build script path references for model loading

### Phase 3: Integrate Backend Implementations ✅ COMPLETED
- [x] Move `bg-remove-onnx` code under `backends::onnx` module (feature-gated)
- [x] Move `bg-remove-tract` code under `backends::tract` module (feature-gated)
- [x] Ensure both backends implement core traits
- [x] Add runtime backend selection logic
- [x] Update import paths to use consolidated crate structure

### Phase 4: Integrate CLI Functionality ✅ COMPLETED
- [x] Move `bg-remove-cli` code to `cli` module
- [x] Create `src/bin/imgly-bgremove.rs` binary
- [x] Preserve all existing CLI functionality and flags
- [x] Update CLI to use consolidated library
- [x] Fix CLI module organization and re-exports

### Phase 5: Consolidate Testing ✅ COMPLETED
- [x] Move e2e test utilities to `tests/` directory
- [x] Convert e2e crate to dev-dependencies
- [x] Ensure all tests work with new structure
- [x] Add integration tests for feature combinations
- [x] Fix test import paths and add missing uuid dependency

### Phase 6: Update Configuration ✅ COMPLETED
- [x] Update root Cargo.toml to single crate (remove workspace)
- [x] Update all CHANGELOG.md files (consolidate into single file)
- [x] Update documentation and README references
- [x] Update llms.txt to reflect new structure
- [x] Resolve dependency version conflicts (env_logger)

### Phase 7: Validation ✅ COMPLETED
- [x] Run `cargo check` with all feature combinations
- [x] Run `cargo test` for comprehensive testing (106 tests pass)
- [x] Run `cargo fmt` for consistent formatting
- [x] Verify CLI binary works correctly
- [x] Test model loading and inference
- [x] Validate provider diagnostics functionality

### Phase 8: Cleanup ✅ COMPLETED
- [x] Remove old crate directories
- [x] Update any remaining references
- [x] Commit changes with appropriate changelog entries

### Phase 9: Provider Availability Enhancement ✅ COMPLETED
- [x] Fix generic provider availability detection
- [x] Integrate with backend-specific availability checks
- [x] Update display logic to show definitive status
- [x] Test and validate accurate provider reporting

## Potential Risks and Considerations

### API Compatibility
- **Risk**: Breaking existing import paths
- **Mitigation**: Preserve public API through re-exports at crate root
- **Example**: `pub use backends::onnx::*;` when onnx feature enabled

### Feature Flag Complexity
- **Risk**: Complex feature interactions and compilation issues
- **Mitigation**: Test all feature combinations, use clear feature dependencies
- **Strategy**: Default to "batteries included" but allow minimal builds

### Binary Size
- **Risk**: Larger binaries when all backends included
- **Mitigation**: Feature flags allow users to choose specific backends
- **Default**: Include both for better user experience

### Testing Complexity  
- **Risk**: More complex CI/CD with feature combinations
- **Mitigation**: Comprehensive feature matrix testing
- **Strategy**: Test default features + individual backend features

## Breaking Changes
This consolidation will require users to:

1. **Update dependencies**:
   ```toml
   # Old
   bg-remove-core = "0.1.0"
   bg-remove-onnx = "0.1.0"
   
   # New
   imgly-bgremove = "0.2.0"
   ```

2. **Update imports**:
   ```rust
   // Old
   use bg_remove_core::BackgroundRemover;
   use bg_remove_onnx::OnnxBackgroundRemover;
   
   // New  
   use imgly_bgremove::{BackgroundRemover, OnnxBackgroundRemover};
   ```

3. **Update CLI usage**:
   ```bash
   # Old
   bg-remove-cli input.jpg output.png
   
   # New
   imgly-bgremove input.jpg output.png
   ```

## Migration Guide
- Provide clear migration documentation
- Include examples for common use cases
- Document feature flag usage
- Provide backwards compatibility where possible

## Questions for Clarification
- Should we maintain backwards compatibility imports?
- What version number should the consolidated crate use?
- Any specific module organization preferences within the consolidated crate?

## Success Criteria ✅ ALL COMPLETED
- [x] Single `imgly-bgremove` crate replaces all workspace crates
- [x] All existing functionality preserved
- [x] Feature flags work correctly for backend selection
- [x] CLI integration seamless
- [x] All tests pass (106 unit tests passing)
- [x] Documentation updated
- [x] Migration path clear for users

## Final Results

### 🎯 Package Consolidation Complete
The package consolidation has been **successfully completed** with all objectives met:

#### ✅ Structural Changes
- **5 separate crates** → **1 unified `imgly-bgremove` package**
- **Workspace configuration** → **Single crate configuration**  
- **Complex dependencies** → **Feature-gated optional components**

#### ✅ Functional Validation
- **106 unit tests passing** - All functionality preserved
- **CLI binary working** - Command-line interface fully functional
- **Provider detection accurate** - Both generic and backend-specific availability reporting
- **Build successful** - All feature combinations compile correctly

#### ✅ Quality Improvements
- **Simplified project structure** - Easier navigation and maintenance
- **Better feature organization** - Clear separation of optional components
- **Accurate provider diagnostics** - Users get definitive availability information
- **Consolidated documentation** - Single changelog and unified documentation

#### ✅ User Benefits
- **Single dependency** - `imgly-bgremove = "0.2.0"` replaces multiple crates
- **Optional features** - Users can choose minimal builds or full functionality
- **Unified CLI** - Single `imgly-bgremove` binary with all capabilities
- **Better diagnostics** - Clear provider availability reporting

The package consolidation successfully transforms a complex multi-crate workspace into a streamlined, feature-rich single package while preserving all functionality and improving the user experience.