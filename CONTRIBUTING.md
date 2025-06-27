# Contributing to bg_remove-rs

Thank you for your interest in contributing to bg_remove-rs! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/bg_remove-rs.git
   cd bg_remove-rs
   ```
3. **Set up the development environment**:
   ```bash
   cargo build
   cargo test
   ```

## Development Workflow

### Branch Strategy

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

3. Push to your fork and create a pull request

### Code Style

- Follow the existing code style in the project
- Run `cargo fmt` before committing
- Ensure `cargo clippy` passes without warnings
- Add tests for new functionality

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation changes
- `style`: formatting, no code change
- `refactor`: code change that neither fixes a bug nor adds a feature
- `test`: adding missing tests
- `chore`: maintenance tasks

Examples:
```
feat(core): add support for new model format
fix(cli): resolve crash when processing large images
docs: update README with installation instructions
```

## Changelog Management

When making changes that affect users, update the appropriate CHANGELOG.md file:

### Which CHANGELOG to Update

- **Core library changes**: `crates/bg-remove-core/CHANGELOG.md`
- **ONNX backend changes**: `crates/bg-remove-onnx/CHANGELOG.md`
- **Tract backend changes**: `crates/bg-remove-tract/CHANGELOG.md`
- **CLI changes**: `crates/bg-remove-cli/CHANGELOG.md`
- **Testing framework changes**: `crates/bg-remove-e2e/CHANGELOG.md`
- **Workspace-wide changes**: `CHANGELOG.md`

### Changelog Format

Add entries under the `[Unreleased]` section:

```markdown
## [Unreleased]

### Added
- New feature that users can benefit from

### Changed
- Existing functionality that has been modified

### Fixed
- Bug fixes that resolve user-reported issues

### Removed
- Features that have been removed (breaking changes)
```

### Changelog Guidelines

- Write from the user's perspective
- Focus on behavior changes, not implementation details
- Use present tense ("Add support for..." not "Added support for...")
- Include relevant PR or issue numbers: `Fix memory leak in model loading (#123)`

## Pull Request Process

1. **Ensure tests pass**: Run `cargo test` locally
2. **Update documentation**: Update relevant documentation if needed
3. **Update changelog**: Add entry to appropriate CHANGELOG.md file
4. **Create pull request**: Provide a clear description of your changes
5. **Address feedback**: Respond to code review comments

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code is formatted with `cargo fmt`
- [ ] No clippy warnings
- [ ] Changelog updated (if applicable)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventional format

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p bg-remove-core

# Run with logging
RUST_LOG=debug cargo test
```

### Writing Tests

- Add unit tests for new functions
- Add integration tests for new features
- Include edge cases and error conditions
- Use descriptive test names

## Code Guidelines

### General Principles

- Follow the [SOLID principles](https://en.wikipedia.org/wiki/SOLID)
- Keep functions small and focused
- Use descriptive variable and function names
- Handle errors explicitly with `Result` types
- Add documentation comments for public APIs

### Rust-Specific Guidelines

- Use `#[must_use]` on functions whose return values should not be ignored
- Prefer `&str` over `String` for function parameters
- Use `#[derive(Debug)]` on structs and enums
- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Error Handling

- Use `anyhow::Result` for application errors
- Use `thiserror` for library errors
- Provide meaningful error messages
- Don't use `unwrap()` or `expect()` in library code

## Architecture

### Workspace Structure

```
bg_remove-rs/
├── crates/
│   ├── bg-remove-core/     # Core library and API
│   ├── bg-remove-onnx/     # ONNX Runtime backend
│   ├── bg-remove-tract/    # Tract backend
│   ├── bg-remove-cli/      # Command-line interface
│   └── bg-remove-e2e/      # End-to-end testing
├── models/                 # Model configurations
├── docs/                   # Documentation
└── .github/               # GitHub workflows
```

### Key Concepts

- **Unified Processor**: Central API for background removal
- **Backend Abstraction**: Pluggable inference backends
- **Model Management**: Automatic model downloading and caching
- **Format Support**: Multiple image formats with ICC profile preservation

## Release Process

Releases are managed using `cargo-smart-release`. See [RELEASE_WORKFLOW.md](docs/RELEASE_WORKFLOW.md) for details.

### For Contributors

- Contributors don't need to worry about versioning
- The maintainers handle releases and version bumps
- Focus on updating changelogs with your changes

## Getting Help

### Resources

- [Project README](README.md)
- [Release Workflow](docs/RELEASE_WORKFLOW.md)
- [API Documentation](https://docs.rs/bg-remove-core)

### Community

- Open an issue for bug reports or feature requests
- Join discussions in existing issues
- Ask questions in pull request comments

## License

By contributing to bg_remove-rs, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).