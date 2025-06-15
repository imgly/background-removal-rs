# Provider Availability Refactoring

## Issue Description

The `check_provider_availability()` function in `inference.rs` is misplaced and contains code duplication. It should be moved to the `OnnxBackend` and renamed to `list_providers()` for better architecture and naming clarity.

## Current Problems

1. **Wrong Location**: The function is in the general `inference.rs` module but only checks ONNX Runtime providers
2. **Code Duplication**: The same provider availability checking logic exists in `OnnxBackend::load_model()`
3. **Poor Naming**: "check_provider_availability" doesn't clearly indicate it returns a list
4. **Separation of Concerns**: Provider availability should be handled by the backend that uses those providers

## Current Implementation

```rust
// In inference.rs
pub fn check_provider_availability() -> Vec<(String, bool, String)> {
    // ONNX-specific provider checking logic
    // Duplicates logic already in OnnxBackend
}
```

## Proposed Solution

Move the function to `OnnxBackend` and rename it:

```rust
impl OnnxBackend {
    /// List all ONNX Runtime execution providers with availability status
    pub fn list_providers() -> Vec<(String, bool, String)> {
        // Consolidated provider checking logic
    }
}
```

## Benefits

- **Single Responsibility**: ONNX backend owns its provider logic
- **No Duplication**: Reuse the same availability checking code
- **Better Naming**: `list_providers` is more descriptive
- **Extensibility**: Other backends can have their own provider lists
- **Cleaner API**: General inference module doesn't need ONNX-specific details

## Implementation Plan

1. Move `check_provider_availability` to `OnnxBackend` as `list_providers`
2. Update CLI to use `OnnxBackend::list_providers()`
3. Remove old function from `inference.rs`
4. Update any tests that reference the old function
5. Consolidate provider checking logic to avoid duplication

## Files to Modify

- `crates/bg-remove-core/src/inference.rs` - Remove old function
- `crates/bg-remove-core/src/backends/onnx.rs` - Add new method
- `crates/bg-remove-cli/src/main.rs` - Update CLI usage
- Tests that reference the old function

## Testing

- Ensure `--show-providers` CLI flag still works
- Verify no functionality is lost
- Check that provider detection logic is consistent

## Priority

High - This is an architectural improvement that reduces code duplication and improves maintainability.