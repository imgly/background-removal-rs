# Cargo Test After Task Completion

After completing any task or phase of development, you MUST run `cargo test` to ensure everything is working correctly.

## When to run cargo test:
- After implementing a new feature
- After fixing a bug
- After refactoring code
- After completing any task from the todo list
- Before committing significant changes
- After merging or integrating changes

## If tests fail:
1. Analyze the failing tests to understand what broke
2. Fix the issues causing test failures
3. Run `cargo test` again to verify fixes
4. Only consider the task complete when all tests pass

## Test execution:
- Run `cargo test` for the entire workspace
- For specific crate testing, use `cargo test -p crate-name`
- Include both unit tests and integration tests

This ensures that changes don't break existing functionality and maintains code reliability throughout development.