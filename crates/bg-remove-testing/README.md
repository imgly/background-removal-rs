# Background Removal Testing Suite

This crate provides comprehensive testing infrastructure for the background removal library, including both standard integration tests and specialized tooling binaries.

## Integration Tests

The crate includes standard Rust integration tests that can be run with different feature flags to control test coverage and execution time.

### Test Categories

#### Fast Tests (Default)
Run with `cargo test --package bg-remove-testing`:
- **Accuracy tests**: Basic functionality validation (4 tests)
- **Performance tests**: Basic performance thresholds (4 tests) 
- **Format tests**: Image format compatibility (6 tests)
- **Compatibility tests**: Configuration and error handling (7 tests)

**Total: 21 fast tests** in ~25 seconds

#### Expensive Tests 
Run with `cargo test --package bg-remove-testing --features expensive-tests`:
- **Comprehensive accuracy suite**: Multi-image validation with complex scenes
- Processes multiple test images across different categories
- Includes all fast tests + 1 additional expensive test

**Total: 22 tests** in ~30 seconds

#### Regression Tests
Run with `cargo test --package bg-remove-testing --features regression-tests`:
- **Performance regression testing**: Multi-run performance validation
- Tests performance thresholds across different image types
- Validates performance doesn't degrade over time

**Total: 1 additional regression test** in ~8 seconds

### Usage Examples

```bash
# Fast development testing (default)
cargo test --package bg-remove-testing

# Pre-commit comprehensive testing
cargo test --package bg-remove-testing --features expensive-tests

# Nightly CI performance validation  
cargo test --package bg-remove-testing --features regression-tests

# Full test suite (all features)
cargo test --package bg-remove-testing --features expensive-tests,regression-tests

# Workspace-wide testing
cargo test --workspace  # Includes all crates
cargo test --workspace --features expensive-tests  # With expensive tests
```

### CI/CD Integration

#### Pull Request CI
```yaml
- name: Fast Tests
  run: cargo test --workspace
```

#### Nightly CI
```yaml
- name: Comprehensive Tests
  run: cargo test --workspace --features expensive-tests,regression-tests
```

## Testing Binaries

The crate also provides specialized testing binaries for comprehensive analysis:

```bash
# Run comprehensive test suite with detailed reporting
cargo run --package bg-remove-testing --bin test-suite

# Generate HTML reports with visual comparisons  
cargo run --package bg-remove-testing --bin generate-report

# Validate outputs against expected results
cargo run --package bg-remove-testing --bin validate-outputs

# Run performance benchmarks
cargo run --package bg-remove-testing --bin benchmark-runner
```

## Test Asset Management

Test assets are organized in the `assets/` directory:
- `assets/input/` - Original test images by category
- `assets/expected/` - Expected output images for validation  
- `assets/test_cases.json` - Test case definitions and metadata

## Benefits of This Architecture

✅ **No ignored tests** - All tests are discoverable and runnable  
✅ **Feature-based control** - Explicit opt-in to expensive tests  
✅ **CI optimization** - Different test suites for different scenarios  
✅ **Comprehensive coverage** - Both fast validation and thorough analysis  
✅ **Standard patterns** - Uses Rust feature flags instead of custom ignore logic