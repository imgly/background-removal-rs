# API Refactoring Implementation Plan

## Feature Description and Goals

Refactor the bg_remove-rs API to simplify the public interface by:
1. Making `remove_background_from_reader` the primary/default function
2. Reducing function parameters from 4 to 2 (reader + config)
3. Consolidating model specification into RemovalConfig
4. Removing redundant/duplicate API functions

This will create a cleaner, more intuitive API that's easier to use and maintain.

## Step-by-Step Implementation Tasks

### Phase 1: Extend RemovalConfig ✅
- [ ] Add `model_spec: ModelSpec` field to RemovalConfig struct
- [ ] Add `format_hint: Option<ImageFormat>` field to RemovalConfig struct
- [ ] Update RemovalConfigBuilder to support new fields with sensible defaults
- [ ] Ensure backward compatibility where possible

### Phase 2: Refactor Core Function
- [ ] Update `remove_background_from_reader` signature to accept only (reader, config)
- [ ] Extract model_spec and format_hint from config within the function
- [ ] Ensure all functionality remains intact

### Phase 3: Update Dependent Functions
- [ ] Refactor `remove_background_from_bytes` to use new reader API
- [ ] Refactor `remove_background_from_image` to use new reader API
- [ ] Update `remove_background_simple_bytes` to use new API

### Phase 4: Remove Deprecated Functions
- [ ] Remove `remove_background_with_backend`
- [ ] Remove `remove_background_with_model`
- [ ] Remove `remove_background_simple`
- [ ] Remove `remove_background_with_model_bytes`

### Phase 5: Testing and Documentation
- [ ] Update all tests to use new API
- [ ] Update documentation examples
- [ ] Update README if needed
- [ ] Run full test suite

### Phase 6: Finalization
- [ ] Update CHANGELOG.md with breaking changes
- [ ] Run cargo fmt and cargo check
- [ ] Ensure all tests pass

## Potential Risks or Impacts

1. **Breaking API Changes**: This is a major breaking change that will require all users to update their code
2. **Default Model Selection**: Need to ensure sensible defaults for model_spec in RemovalConfig
3. **Migration Path**: Should provide clear migration guide in changelog

## Questions for Clarification

1. ✅ Default model_spec: What should be the default if not specified? (Use first cached model or require explicit specification?)
2. ✅ format_hint: Should this be optional in config with None as default?
3. ✅ Should we provide a migration guide or compatibility layer?

## Functionality to be Modified or Removed

### Functions to be Removed:
- `remove_background_with_backend` - Redundant with new unified API
- `remove_background_with_model` - Functionality merged into reader-based API
- `remove_background_simple` - Simplified API no longer needed
- `remove_background_with_model_bytes` - Redundant with new API

### Functions to be Modified:
- `remove_background_from_reader` - Signature change to (reader, config)
- `remove_background_from_bytes` - Will internally use new reader API
- `remove_background_from_image` - Will internally use new reader API
- `remove_background_simple_bytes` - Will use new API internally

## Planned Worktree Workflow

1. All development happens in `worktree/feat-api-refactor`
2. Incremental commits after each phase
3. Full test validation before merge
4. Merge back to main when complete
5. Update changelog as part of feature branch

## Success Criteria

- [ ] All tests pass with new API
- [ ] API is simplified to 2-parameter functions
- [ ] Model specification is part of config
- [ ] No duplicate/redundant functions remain
- [ ] Documentation is updated
- [ ] Breaking changes are clearly documented

## Implementation Progress

### Phase 1: Extend RemovalConfig ✅
- [✅] Added model_spec field to RemovalConfig
- [✅] Added format_hint field to RemovalConfig  
- [✅] Updated RemovalConfigBuilder
- [✅] Added Default implementation for RemovalConfig

### Phase 2: Refactor Core Function ✅
- [✅] Updated remove_background_from_reader signature
- [✅] Extracted model_spec from config
- [✅] All functionality preserved

### Phase 3: Update Dependent Functions ✅
- [✅] Refactored remove_background_from_bytes
- [✅] Refactored remove_background_from_image
- [✅] Updated remove_background_simple_bytes

### Phase 4: Remove Deprecated Functions ✅
- [✅] Removed remove_background_with_backend
- [✅] Removed remove_background_with_model
- [✅] Removed remove_background_simple
- [✅] Removed remove_background_with_model_bytes

### Phase 5: Testing and Documentation ✅
- [✅] Updated lib.rs tests
- [✅] Fixed compilation errors in tests
- [✅] Updated inline documentation
- [✅] Updated main library documentation examples
- [✅] All unit tests pass (129 tests)

### Phase 6: Finalization ✅
- [✅] Updated CHANGELOG.md with breaking changes
- [✅] Ran cargo fmt
- [✅] Ran cargo check - all passing
- [✅] All tests pass

## Final Results

The API refactoring has been successfully implemented:

1. **Simplified API**: The main entry point is now `remove_background_from_reader` with just 2 parameters
2. **Consolidated Configuration**: `RemovalConfig` now includes `model_spec` and `format_hint`
3. **Removed Redundancy**: 4 redundant functions have been removed
4. **Maintained Compatibility**: All existing functionality is preserved through the simplified API
5. **Updated Documentation**: All function documentation has been updated to reflect the changes

### Key Improvements:
- Reduced cognitive load for users (fewer functions to choose from)
- More consistent API design (everything goes through config)
- Easier to extend in the future (just add to config)
- Cleaner codebase with less duplication

### Migration Guide for Users:

**Before:**
```rust
let result = remove_background_with_model("input.jpg", &config, &model_spec).await?;
```

**After:**
```rust
let config = RemovalConfig::builder()
    .model_spec(model_spec)
    .build()?;
let file = tokio::fs::File::open("input.jpg").await?;
let result = remove_background_from_reader(file, &config).await?;
```

The refactoring is complete and ready for release as a major version bump due to breaking API changes.