# Mandatory Doc Test Execution

## CRITICAL RULE: Doc Tests Must Always Be Run

Doc tests are an integral part of the testing strategy and MUST be executed during all testing phases. Skipping doc tests is **strictly prohibited** and considered a serious testing violation.

## When Doc Tests Are Required

### ✅ ALWAYS Required
- **Any testing workflow** - unit, integration, or comprehensive testing
- **Before committing changes** to any Rust source files or documentation
- **After modifying documentation** with code examples
- **When fixing failing tests** of any type
- **Before merging feature branches** to main
- **During CI/CD pipeline execution**
- **When adding new public APIs** or functions
- **After changing function signatures** that appear in examples

### ❌ NO Exceptions
There are **NO exceptions** to running doc tests. Unlike some other test types, doc tests validate that:
- **Documentation examples actually compile**
- **Public API usage examples are correct**
- **Code examples stay in sync with implementation**
- **Users can rely on documentation examples**

## Mandatory Testing Commands

### 1. Full Test Suite (REQUIRED)
```bash
# This command MUST be used for comprehensive testing
cargo test
```
This runs ALL test types including:
- Unit tests (`cargo test --lib`)
- Integration tests (`cargo test --test '*'`)
- Doc tests (`cargo test --doc`)

### 2. Doc Test Only (For Focused Testing)
```bash
# Use this when specifically working on documentation
cargo test --doc
```

### 3. Verification Commands
```bash
# Verify all test types pass
cargo test --lib && cargo test --doc && cargo test --test '*'
```

## Doc Test Requirements

### Documentation Quality Standards
- **All public functions** MUST have doc tests or `no_run` examples
- **Code examples** MUST compile successfully  
- **Examples MUST be realistic** and demonstrate actual usage
- **Async examples** MUST have proper async function wrappers
- **File operations** MUST use `no_run` or mock data
- **External dependencies** MUST be handled appropriately

### Example Patterns

#### ✅ Good Doc Test Example
```rust
/// Process an image to remove background
///
/// # Examples
///
/// ```rust,no_run
/// use my_crate::{process_image, Config};
/// 
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let config = Config::default();
///     let result = process_image("input.jpg", &config).await?;
///     Ok(())
/// }
/// ```
pub async fn process_image(path: &str, config: &Config) -> Result<Output> {
    // implementation
}
```

#### ❌ Bad Doc Test Example
```rust
/// Process an image
///
/// # Examples
///
/// ```rust
/// // This will fail - no async wrapper, file doesn't exist
/// let result = process_image("input.jpg", &config).await?;
/// ```
pub async fn process_image(path: &str, config: &Config) -> Result<Output> {
    // implementation
}
```

## Enforcement and Validation

### Pre-Commit Requirements
Before ANY commit that modifies:
- Rust source files (`.rs`)
- Documentation files with code examples
- README.md with Rust examples

**MANDATORY checks:**
1. ✅ `cargo test --doc` passes with 0 failures
2. ✅ `cargo test` passes completely (includes doc tests)
3. ✅ All examples compile and execute properly

### Feature Development Workflow
```bash
# 1. During development - run doc tests frequently
cargo test --doc

# 2. Before committing - run full test suite
cargo test

# 3. Verify doc test count (should not decrease unexpectedly)
cargo test --doc -- --show-output | grep "test result:"
```

### CI/CD Integration
Continuous integration MUST include:
```yaml
# Example GitHub Actions step
- name: Run all tests including doc tests
  run: cargo test
  
- name: Verify doc tests specifically
  run: cargo test --doc
```

## Common Doc Test Issues and Solutions

### Issue: Async Functions Without Wrappers
❌ **Problem:**
```rust
/// ```rust
/// let result = async_function().await?;
/// ```
```

✅ **Solution:**
```rust
/// ```rust,no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let result = async_function().await?;
/// # Ok(())
/// # }
/// ```
```

### Issue: File Operations
❌ **Problem:**
```rust
/// ```rust
/// let file = File::open("input.jpg")?;
/// ```
```

✅ **Solution:**
```rust
/// ```rust,no_run
/// use std::fs::File;
/// let file = File::open("input.jpg")?;
/// ```
```

### Issue: Missing Imports
❌ **Problem:**
```rust
/// ```rust
/// let config = Config::default();
/// ```
```

✅ **Solution:**
```rust
/// ```rust,no_run
/// use my_crate::Config;
/// let config = Config::default();
/// ```
```

## Doc Test Workflow Integration

### With Unit Tests
```bash
# Run unit tests first, then doc tests
cargo test --lib
cargo test --doc
```

### With Integration Tests
```bash
# Run all test types in sequence
cargo test --lib
cargo test --doc  
cargo test --test '*'
```

### Complete Validation
```bash
# Single command for complete testing
cargo test
```

## Error Reporting Requirements

When doc tests fail:

### 1. IMMEDIATE Action Required
- **STOP all other work** until doc tests pass
- **Do NOT commit** with failing doc tests
- **Fix the failing examples** before proceeding

### 2. Failure Analysis
- **Identify the specific doc test** that failed
- **Determine root cause** (syntax, imports, async, file access)
- **Apply appropriate fix** from solutions above
- **Re-run doc tests** to verify fix

### 3. Prevention
- **Review all code examples** when changing function signatures
- **Test examples manually** if they seem complex
- **Use `no_run` liberally** for examples that access external resources

## Monitoring and Metrics

### Doc Test Coverage Tracking
```bash
# Check doc test count trends
cargo test --doc 2>&1 | grep "running [0-9]* tests"

# Verify no decrease in doc test coverage
# (Should be monitored over time)
```

### Quality Metrics
- **Doc test pass rate**: Must be 100%
- **Doc test count**: Should increase with new public APIs
- **Example quality**: Regular review for realistic usage patterns

## Summary

**Doc tests are not optional extras—they are mandatory quality gates.**

- **ALWAYS run doc tests** as part of any testing workflow
- **NEVER skip doc tests** when validating code changes
- **IMMEDIATELY fix** any failing doc tests
- **INCLUDE doc tests** in all CI/CD pipelines
- **MAINTAIN high-quality** examples that actually help users

**Violation of this rule compromises:**
- User experience with documentation
- Code example reliability  
- API usage guidance quality
- Overall project professionalism

Doc tests ensure that documentation stays synchronized with code and that users can trust the examples provided. This is critical for library adoption and user success.