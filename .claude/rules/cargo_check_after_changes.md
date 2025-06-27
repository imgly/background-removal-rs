# Cargo Check After Changes

After each change to any Rust source file (*.rs) or Cargo.toml file, you MUST run `cargo check` to ensure that the changes are valid.

If `cargo check` fails with errors or warnings:
1. Fix all errors first
2. Fix all warnings (as the workspace has a zero-warning policy with `warnings = "deny"`)
3. Run `cargo check` again to verify the fixes
4. Only proceed with further changes after `cargo check` passes successfully

This ensures code integrity is maintained throughout the development process.