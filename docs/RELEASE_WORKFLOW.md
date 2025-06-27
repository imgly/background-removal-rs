# Release Workflow

This document describes the release workflow for the bg_remove-rs workspace using `cargo-smart-release`.

## Overview

The bg_remove-rs project uses `cargo-smart-release` for automated versioning, changelog management, and publishing to crates.io. This tool is specifically designed for Rust workspaces and handles complex dependency relationships between crates.

## Tools Used

- **cargo-smart-release**: Main release automation tool
- **cargo-changelog**: Changelog management (bundled with cargo-smart-release)
- **GitHub Actions**: Automated CI/CD workflows

## Workflow Steps

### 1. Development and Changes

During development, update the appropriate CHANGELOG.md files:

- For changes to `bg-remove-core`: Update `crates/bg-remove-core/CHANGELOG.md`
- For changes to `bg-remove-onnx`: Update `crates/bg-remove-onnx/CHANGELOG.md`
- For changes to `bg-remove-tract`: Update `crates/bg-remove-tract/CHANGELOG.md`  
- For changes to `bg-remove-cli`: Update `crates/bg-remove-cli/CHANGELOG.md`
- For changes to `bg-remove-e2e`: Update `crates/bg-remove-e2e/CHANGELOG.md`
- For workspace-wide changes: Update root `CHANGELOG.md`

Add entries under the `[Unreleased]` section following the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format:

```markdown
## [Unreleased]

### Added
- New feature description

### Changed
- Changed functionality description

### Fixed
- Bug fix description

### Removed
- Removed functionality description
```

### 2. Pre-Release Checks

Before releasing, ensure:

- All tests pass: `cargo test`
- Code is properly formatted: `cargo fmt`
- No warnings: `cargo check`
- Documentation builds: `cargo doc --no-deps`

### 3. Release Process

#### Option A: Manual Release

1. **Dry Run**: Test the release process without making changes:
   ```bash
   cargo smart-release --update-crates-index --no-publish --no-push --allow-dirty --execute
   ```

2. **Execute Release**: Perform the actual release:
   ```bash
   cargo smart-release --update-crates-index
   ```

This will:
- Analyze the workspace and determine which crates need updates
- Bump version numbers appropriately based on changes
- Update CHANGELOG.md files by moving unreleased entries to versioned sections
- Create git commits and tags
- Publish to crates.io (if you have `CARGO_REGISTRY_TOKEN` set)

#### Option B: Automated GitHub Release

1. **Push changes** to the main branch
2. **Trigger workflow** via GitHub Actions:
   - Go to Actions → Release workflow
   - Click "Run workflow"
   - Specify the version to release

### 4. Post-Release

After a successful release:

- Verify packages are available on [crates.io](https://crates.io/)
- Check that GitHub releases are created with proper changelog entries
- Update any external documentation if needed

## Versioning Strategy

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.y.z): Breaking changes
- **MINOR** (x.Y.z): New features, backwards compatible
- **PATCH** (x.y.Z): Bug fixes, backwards compatible

cargo-smart-release automatically determines the appropriate version bump based on:
- Conventional commit messages
- CHANGELOG.md entries
- Dependency analysis

## Crate Dependencies

The workspace crates have the following dependency relationships:

```
bg-remove-cli → bg-remove-core → {bg-remove-onnx, bg-remove-tract}
bg-remove-e2e → bg-remove-core → {bg-remove-onnx, bg-remove-tract}
```

cargo-smart-release handles these dependencies automatically, ensuring:
- Dependencies are published before dependents
- Version numbers are updated consistently across the workspace
- No circular dependency issues

## Configuration

### Repository Setup

Required repository secrets (for GitHub Actions):
- `CARGO_REGISTRY_TOKEN`: Token for publishing to crates.io

### Local Setup

Install required tools:
```bash
cargo install cargo-smart-release
```

Configure git (if not already done):
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

Login to crates.io:
```bash
cargo login YOUR_CRATES_IO_TOKEN
```

## Troubleshooting

### Common Issues

1. **"No changes detected"**: Ensure CHANGELOG.md files have unreleased entries
2. **"Dependency version mismatch"**: Run `cargo update` to sync dependency versions
3. **"Authentication failed"**: Verify `CARGO_REGISTRY_TOKEN` is set correctly
4. **"Git working directory dirty"**: Commit or stash changes before release

### Recovery

If a release fails partway through:
1. Check what was published to crates.io
2. Manually fix any git state issues
3. Re-run the release process (cargo-smart-release is idempotent)

## Best Practices

1. **Regular releases**: Release frequently to avoid large changelog accumulations
2. **Clear changelogs**: Write user-focused changelog entries, not implementation details
3. **Test before release**: Always run the dry-run option first
4. **Coordinate workspace changes**: Consider impact on dependent crates
5. **Review generated changes**: Check version bumps and changelog formatting before pushing

## References

- [cargo-smart-release documentation](https://github.com/Byron/gitoxide/tree/main/cargo-smart-release)
- [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
- [Semantic Versioning](https://semver.org/)
- [Cargo Publishing Guide](https://doc.rust-lang.org/cargo/reference/publishing.html)