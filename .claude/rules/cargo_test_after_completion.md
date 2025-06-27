# Cargo Test After Task Completion

After completing any task or phase of development, you MUST run comprehensive tests to ensure everything is working correctly.

## When to run tests:
- After implementing a new feature
- After fixing a bug
- After refactoring code
- After completing any task from the todo list
- Before committing significant changes
- After merging or integrating changes

## MANDATORY Test Sequence:

### 1. Unit Tests First
Run `cargo test --lib` to execute all unit tests:
- Fastest feedback on core functionality
- Tests individual modules and functions
- Must pass before proceeding to integration tests

### 2. Integration Tests Second
After unit tests pass, run `cargo test` (full test suite):
- Includes both unit tests and integration tests
- Tests end-to-end functionality and component interactions
- Validates real-world usage scenarios
- Tests in `tests/` directory that verify public APIs

### 3. If Any Tests Fail:
1. **Unit test failures**: Fix immediately - indicates broken core functionality
2. **Integration test failures**: Analyze the failure type:
   - **Import/compilation errors**: Usually structural issues that need fixing
   - **Logic failures**: May indicate breaking changes to public APIs
   - **Pre-existing failures**: Document and note if unrelated to current changes
3. Fix the issues causing test failures
4. Re-run the complete test sequence
5. Only consider the task complete when ALL tests pass OR failing tests are confirmed as pre-existing and unrelated

## Test execution commands:
- **Unit tests only**: `cargo test --lib`
- **Integration tests only**: `cargo test --test '*'`
- **Full test suite**: `cargo test` (includes both unit and integration)
- **Specific crate**: `cargo test -p crate-name`
- **Specific test**: `cargo test test_name`

## Special Cases:
- If integration tests have structural issues (import errors, missing dependencies), document these as separate issues
- If integration tests fail due to unrelated pre-existing problems, note this but don't block completion
- If integration tests fail due to your changes, they MUST be fixed before task completion

This comprehensive testing approach ensures that changes don't break existing functionality and maintains both code reliability and API compatibility throughout development.