# Cache Disabled Testing Rule

## MANDATORY: Additional Testing with All Caches Disabled

In addition to standard testing procedures, you MUST run tests with all caches disabled to ensure that caching is not hiding errors or masking issues in the underlying functionality.

## When Cache-Disabled Testing Is Required

### ✅ MANDATORY For:
- **Feature completion** - before marking any feature as complete
- **Bug fixes** - to ensure fixes work without cache assistance
- **Cache-related changes** - any modifications to caching logic
- **Performance optimizations** - to verify optimizations don't rely solely on cache
- **Model loading changes** - to ensure models load correctly without cache
- **Configuration changes** - that might affect caching behavior
- **Release preparation** - before any release or deployment
- **CI/CD validation** - as part of comprehensive testing pipelines

### 🔄 Integration with Existing Testing Rules

This rule **extends** but does not replace existing testing requirements:
1. **Run standard tests first** following existing cargo_test_after_completion.md
2. **Then run cache-disabled tests** as additional validation
3. **Both must pass** for feature completion

## Cache Disabling Methods

### Environment Variables
Set these environment variables to disable various caches:

```bash
# Disable all model caching
export BG_REMOVE_DISABLE_MODEL_CACHE=true

# Disable session caching  
export BG_REMOVE_DISABLE_SESSION_CACHE=true

# Use temporary directories (forces fresh downloads)
export BG_REMOVE_CACHE_DIR=$(mktemp -d)

# Disable any compiler/build caches
export CARGO_TARGET_DIR=$(mktemp -d)
```

### Cargo Command Modifications
Use specific cargo flags to disable caches:

```bash
# Clear target directory before testing
cargo clean

# Run tests with fresh compilation
cargo test --offline=false
```

### Comprehensive Cache-Disabled Test Sequence

```bash
# 1. Set up cache-disabled environment
export BG_REMOVE_DISABLE_MODEL_CACHE=true
export BG_REMOVE_DISABLE_SESSION_CACHE=true
export BG_REMOVE_CACHE_DIR=$(mktemp -d)
export CARGO_TARGET_DIR=$(mktemp -d)

# 2. Clean any existing build artifacts
cargo clean

# 3. Run full test suite without caches
cargo test

# 4. Run specific cache-related tests
cargo test cache

# 5. Run doc tests to ensure examples work without cache
cargo test --doc

# 6. Clean up temporary directories
rm -rf "$BG_REMOVE_CACHE_DIR" "$CARGO_TARGET_DIR"
```

## MANDATORY Test Execution Workflow

### Phase 1: Standard Testing (Existing Rules)
1. **Run standard cargo test** following existing rules
2. **Ensure all tests pass** with normal caching enabled
3. **Verify benchmarks run** with cached performance

### Phase 2: Cache-Disabled Testing (This Rule)
1. **Set cache-disabled environment** using variables above
2. **Clean build artifacts** with `cargo clean`
3. **Run full test suite** without any caching assistance
4. **Verify all tests still pass** in cache-disabled mode
5. **Check performance degradation** is within expected bounds
6. **Clean up temporary directories** after testing

### Phase 3: Validation and Documentation
1. **Document cache-disabled test results** in implementation plan
2. **Note any issues discovered** during cache-disabled testing
3. **Verify cache provides performance benefit** but not correctness

## What Cache-Disabled Testing Reveals

### Hidden Dependencies
- **Model loading issues** masked by cached models
- **Network connectivity problems** hidden by cache hits
- **File permission issues** not caught with cached files
- **Memory leaks** that only occur during fresh loading

### Correctness Issues
- **Logic errors** that only manifest without cache shortcuts
- **Race conditions** in cache vs. non-cache code paths
- **Initialization problems** bypassed by warm caches
- **Error handling gaps** in cache-miss scenarios

### Performance Understanding
- **True performance characteristics** without cache benefits
- **Cache effectiveness measurement** by comparing with/without
- **Memory usage patterns** during fresh operations
- **Realistic user experience** for first-time or cache-cleared usage

## Common Issues Found by Cache-Disabled Testing

### Model Loading Problems
```bash
# Test model download when cache is empty
export BG_REMOVE_DISABLE_MODEL_CACHE=true
cargo test test_model_loading

# Common issues found:
# - Network timeout handling
# - Corrupt download recovery
# - Permission issues in temp directories
# - Missing model validation
```

### Session Management Issues
```bash
# Test session handling without session cache
export BG_REMOVE_DISABLE_SESSION_CACHE=true
cargo test test_session_workflows

# Common issues found:
# - Resource cleanup problems
# - Memory leaks in session creation
# - State initialization errors
# - Configuration loading failures
```

### Integration Workflow Problems
```bash
# Test end-to-end workflows without any caches
export BG_REMOVE_DISABLE_MODEL_CACHE=true
export BG_REMOVE_DISABLE_SESSION_CACHE=true
cargo test integration

# Common issues found:
# - Slow performance without revealing bugs
# - Error paths not exercised with cache hits
# - Resource contention in fresh setups
# - Configuration precedence issues
```

## Error Reporting and Resolution

### When Cache-Disabled Tests Fail

#### 1. IMMEDIATE Action Required
- **STOP feature completion** until cache-disabled tests pass
- **Do NOT merge or commit** features that fail cache-disabled tests
- **Analyze the specific failure** to understand root cause
- **Fix underlying issue** not cache-dependency

#### 2. Common Failure Categories

**Cache Dependency Errors:**
```bash
# Error: Feature only works with warm cache
# Solution: Fix feature to work from cold start
# Example: Model loading that assumes cached models exist
```

**Performance Timeout Errors:**
```bash
# Error: Tests timeout without cache acceleration
# Solution: Increase test timeouts for cache-disabled mode
# Example: Integration tests that assume fast model access
```

**Resource Access Errors:**
```bash
# Error: Cannot access resources without cache layer
# Solution: Fix resource access patterns
# Example: File permission issues in fresh directories
```

**Configuration Loading Errors:**
```bash
# Error: Configuration not found without cache
# Solution: Fix configuration discovery logic  
# Example: Config files not found in fresh environment
```

#### 3. Resolution Process
1. **Identify if error is cache-dependency** or legitimate bug
2. **Fix underlying issue** rather than working around it
3. **Re-run cache-disabled tests** to verify fix
4. **Re-run standard tests** to ensure no regression
5. **Document the issue and resolution** in implementation plan

## Integration with Other Rules

### Works with Existing Testing Rules
- **Extends cargo_test_after_completion.md** with additional requirements
- **Enhances feature_finalization_testing.md** with cache validation
- **Supports mandatory_doc_test_execution.md** by testing docs without cache

### Enforcement Alignment
- **Follows mandatory_worktree_usage.md** by being created in feature worktree
- **Supports code_quality.md** by revealing hidden quality issues
- **Enhances git_workflow.md** by ensuring complete feature validation

## Documentation Requirements

### Implementation Plan Updates
Include cache-disabled testing results:

```markdown
## Cache-Disabled Testing Validation

### Standard Test Results (With Cache)
- All tests passing: ✅
- Performance benchmarks: [X ms with cache]
- Test execution time: [Y seconds with cache]

### Cache-Disabled Test Results  
- All tests passing: ✅
- Performance benchmarks: [Z ms without cache] 
- Test execution time: [A seconds without cache]
- Cache performance benefit: [X-Z ms improvement]

### Issues Found and Resolved
- [Any cache-dependency issues discovered]
- [Performance bottlenecks revealed]
- [Error handling gaps identified]

### Cache Effectiveness Metrics
- Model loading: [B% faster with cache]
- Session creation: [C% faster with cache] 
- Overall workflow: [D% faster with cache]
```

## Automation and CI/CD

### GitHub Actions Integration
Add cache-disabled testing to CI pipeline:

```yaml
- name: Run tests with cache disabled
  run: |
    export BG_REMOVE_DISABLE_MODEL_CACHE=true
    export BG_REMOVE_DISABLE_SESSION_CACHE=true
    export BG_REMOVE_CACHE_DIR=$(mktemp -d)
    cargo clean
    cargo test
  env:
    CARGO_TARGET_DIR: /tmp/cargo-target-no-cache
```

### Pre-Commit Hook Integration
Include in pre-commit validation:

```bash
#!/bin/bash
echo "Running standard tests..."
cargo test

echo "Running cache-disabled tests..."
export BG_REMOVE_DISABLE_MODEL_CACHE=true
export BG_REMOVE_DISABLE_SESSION_CACHE=true
export BG_REMOVE_CACHE_DIR=$(mktemp -d)
cargo clean
cargo test
echo "Cache-disabled tests passed ✅"
```

## Summary

**Cache-disabled testing is mandatory** for ensuring robust, reliable software that works correctly regardless of cache state.

**Key Benefits:**
- **Reveals hidden dependencies** on caching for correctness
- **Validates error handling** in cache-miss scenarios  
- **Measures true performance** characteristics
- **Ensures user experience** consistency across cache states
- **Prevents cache-dependent bugs** from reaching production

**Enforcement:**
- **ALWAYS run** cache-disabled tests after standard tests
- **NEVER complete features** without cache-disabled validation
- **IMMEDIATELY fix** any cache-dependency issues discovered
- **DOCUMENT results** in implementation plans and commit messages

**Cache should provide performance benefits, not correctness benefits.** If functionality breaks without cache, the underlying implementation has bugs that must be fixed.