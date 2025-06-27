# Changelog Update Rule

You MUST update the appropriate CHANGELOG.md file(s) whenever making changes that affect users or the public API.

## When to Update Changelog

Update changelog for ANY of the following changes:
- **New features** or functionality
- **Bug fixes** that affect user experience
- **Breaking changes** to public APIs
- **Performance improvements** or optimizations
- **Configuration changes** that affect usage
- **Dependency updates** that impact functionality
- **Documentation changes** that affect user workflows
- **CLI interface changes** (new flags, changed behavior)
- **Model additions** or changes
- **Format support** additions or modifications

## Which CHANGELOG.md to Update

Determine the appropriate changelog based on the affected component:

### Core Library Changes
- **File**: `crates/bg-remove-core/CHANGELOG.md`
- **When**: API changes, new inference features, model support, configuration options

### CLI Application Changes  
- **File**: `crates/bg-remove-cli/CHANGELOG.md`
- **When**: New CLI flags, command behavior changes, output format changes

### ONNX Backend Changes
- **File**: `crates/bg-remove-onnx/CHANGELOG.md`
- **When**: Provider support changes, performance optimizations, ONNX Runtime updates

### Tract Backend Changes
- **File**: `crates/bg-remove-tract/CHANGELOG.md`
- **When**: Pure Rust implementation changes, optimization improvements

### Testing Framework Changes
- **File**: `crates/bg-remove-e2e/CHANGELOG.md`
- **When**: New test types, framework changes, validation improvements

### Workspace-Wide Changes
- **File**: `CHANGELOG.md` (root)
- **When**: Cross-crate changes, build system updates, overall project changes

## Changelog Format

Add entries under the `[Unreleased]` section using Keep a Changelog format:

```markdown
## [Unreleased]

### Added
- New feature that users can benefit from
- New CLI flag `--example` for demonstration purposes

### Changed
- Existing functionality that has been modified
- Updated default model from ISNet FP16 to FP32

### Fixed
- Bug fixes that resolve user-reported issues
- Fixed memory leak in model loading (#123)

### Removed
- Features that have been removed (breaking changes)
- Deprecated CLI flag `--old-flag` removed

### Security
- Security-related improvements or fixes
```

## Entry Guidelines

### Writing Style
- Write from the **user's perspective**, not implementation details
- Use **present tense** ("Add support for..." not "Added support for...")
- Be **specific and actionable** - what changed and why it matters
- Include **issue/PR numbers** when relevant: `Fix memory leak (#123)`

### Good Examples
```markdown
### Added
- Support for WebP format with ICC color profile preservation
- New `--preserve-color-profiles` CLI flag (enabled by default)
- BiRefNet Lite model variant for portrait-focused background removal

### Changed
- Default execution provider changed from CPU to Auto for better performance
- Improved error messages with troubleshooting suggestions and context

### Fixed
- Resolved aspect ratio distortion in output images
- Fixed WebP transparency corruption with large images
```

### Bad Examples
```markdown
### Added
- Updated some stuff in the processor (too vague)
- Fixed a bug (no context or impact)
- Refactored internal code structure (internal detail, not user-facing)
```

## Automation

### GitHub Actions Enforcement
- The `changelog-check.yml` workflow enforces changelog updates for Rust file changes
- PRs modifying `.rs` or `.toml` files MUST include changelog updates
- The workflow will comment on PRs missing changelog updates

### Release Process
- `cargo smart-release` automatically processes changelog entries during releases
- Unreleased entries are moved to versioned sections with proper commit attribution
- Changelog entries drive automatic version bump decisions (patch/minor/major)

## Commit Integration

When committing changes that require changelog updates:

1. **Update changelog first** before committing code changes
2. **Include changelog file** in the same commit as the related changes
3. **Reference changelog** in commit messages when appropriate

Example commit workflow:
```bash
# 1. Make code changes
# 2. Update appropriate CHANGELOG.md
# 3. Commit both together
git add crates/bg-remove-core/src/new_feature.rs crates/bg-remove-core/CHANGELOG.md
git commit -m "feat(core): add new background removal feature

- Add support for custom processing modes
- Updated CHANGELOG.md with feature description"
```

## Special Cases

### Breaking Changes
For breaking changes, ALWAYS update changelog with:
- Clear description of what changed
- Migration instructions or alternatives
- Version impact (will trigger major version bump)

### Dependencies
For significant dependency updates:
- Note user-visible impacts (performance, compatibility)
- Include version numbers when relevant
- Document any behavior changes

### Internal Changes
DO NOT include in changelog:
- Pure refactoring with no user impact
- Test-only changes (unless they affect testing workflows)
- Documentation fixes that don't change usage
- Code formatting or style changes

## Validation

Before submitting PR:
1. ✅ Changelog updated for all user-facing changes
2. ✅ Correct changelog file(s) modified
3. ✅ Entries follow Keep a Changelog format
4. ✅ Entries are user-focused and actionable
5. ✅ No internal implementation details included

## Examples by Change Type

### New Feature
```markdown
### Added
- BiRefNet portrait model support for human-focused background removal
- CLI `--model birefnet-portrait` option for model selection
- Automatic model variant selection based on execution provider
```

### Bug Fix
```markdown
### Fixed
- Resolved output dimension mismatch where 1024x768 input became 800x600 output
- Fixed WebP transparency corruption with images larger than 2MB
- Corrected ICC color profile preservation in JPEG files
```

### Performance Improvement
```markdown
### Changed
- Improved inference performance by 44% for CoreML provider with FP32 models
- Optimized memory usage in batch processing mode
- Reduced model loading time from 2.1s to 0.8s on Apple Silicon
```

This rule ensures consistent, user-focused changelog maintenance that supports automated release management and clear communication of changes to users.