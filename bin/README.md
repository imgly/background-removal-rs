# Development Scripts

This directory contains development and maintenance scripts for the bg-remove-rs project.

## Scripts

### 🔍 `lint.sh` - Zero-Warning Policy Enforcement
Comprehensive linting and quality checks across the entire workspace.

```bash
# Run all checks (read-only)
./bin/lint.sh

# Format code and run checks
./bin/lint.sh --format

# Format code and auto-fix issues
./bin/lint.sh --fix
```

**What it checks:**
- Workspace compilation (dev and release)
- All targets (lib, bin, examples, tests, benches)
- Unit tests
- Feature flag combinations (FP16/FP32)
- Clippy warnings (zero-warning policy)
- Code formatting
- Documentation generation

### 🎨 `format.sh` - Code Formatting
Quick access to Rust code formatting operations.

```bash
# Format all code
./bin/format.sh format

# Check formatting without changes
./bin/format.sh check

# Format code and auto-fix Clippy issues
./bin/format.sh fix

# Show formatting differences
./bin/format.sh diff

# Display configuration
./bin/format.sh config
```

**Features:**
- Uses project's `rustfmt.toml` configuration
- Supports diff preview before applying changes
- Integrates with Clippy auto-fixes
- Shows effective configuration

### 🪝 `pre-commit-hook.sh` - Pre-commit Validation
Git pre-commit hook for automated quality checks.

```bash
# Install the hook
cp bin/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Enable automatic formatting (optional)
export PRE_COMMIT_AUTO_FORMAT=true
export PRE_COMMIT_AUTO_FIX=true
```

**Validation:**
- Fast compilation check
- Clippy warnings detection
- Code formatting verification
- Optional auto-formatting and auto-fixing

## Usage Patterns

### Daily Development
```bash
# Quick format and check
./bin/format.sh format && ./bin/lint.sh

# Full quality check with auto-fixes
./bin/lint.sh --fix
```

### Before Committing
```bash
# Ensure everything is clean
./bin/lint.sh

# Or install the pre-commit hook for automation
cp bin/pre-commit-hook.sh .git/hooks/pre-commit
```

### CI/CD Integration
The GitHub Actions workflow uses these scripts for:
- Continuous quality validation
- Cross-platform consistency checks
- Automated formatting in PRs

## Configuration

All scripts respect the project's configuration files:
- `rustfmt.toml` - Code formatting rules
- `Cargo.toml` - Workspace linting configuration
- `.editorconfig` - Editor consistency settings

## Troubleshooting

If scripts fail to run:
```bash
# Ensure they're executable
chmod +x bin/*.sh

# Run from project root
cd /path/to/bg-remove-rs
./bin/script-name.sh
```

For detailed help on any script:
```bash
./bin/script-name.sh --help
```