# Code Formatting Guidelines

This document outlines the automatic code formatting standards for the bg-remove-rs project using `rustfmt`.

## Overview

We use automatic code formatting to ensure consistent code style across the entire codebase. This eliminates debates about style and allows developers to focus on functionality.

## Configuration

### rustfmt.toml

The project uses a comprehensive `rustfmt.toml` configuration file that enforces:

- **Line length**: 100 characters maximum
- **Import organization**: Grouped and sorted automatically  
- **Consistent formatting**: Structs, functions, and expressions
- **Comment formatting**: Wrapped and normalized
- **Documentation**: Code blocks in docs are formatted

### EditorConfig

The `.editorconfig` file ensures consistent settings across different editors:

- UTF-8 encoding
- LF line endings
- 4-space indentation for Rust files
- Trailing whitespace removal
- Final newline insertion

## Key Formatting Rules

### Import Organization
```rust
// Standard library imports
use std::collections::HashMap;
use std::path::Path;

// External crate imports  
use anyhow::Result;
use serde::{Deserialize, Serialize};

// Local crate imports
use crate::config::RemovalConfig;
use crate::types::ProcessingTimings;
```

### Function Formatting
```rust
// Single-line functions are avoided
pub fn process_image(
    input: &DynamicImage,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    // Function body
}

// Parameters use "Tall" layout
pub fn complex_function_with_many_parameters(
    first_parameter: String,
    second_parameter: usize,
    third_parameter: &Config,
    fourth_parameter: Option<&str>,
) -> Result<ComplexReturnType> {
    // Function body
}
```

### Struct and Enum Formatting
```rust
// Struct formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub timings: ProcessingTimings,
    pub model_name: String,
    pub model_precision: String,
    pub input_format: String,
}

// Enum formatting with trailing commas
#[derive(Debug, Clone)]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,
    Auto,
}
```

### Comment and Documentation
```rust
/// Process an image to remove its background
///
/// This function takes an input image and configuration, then uses
/// the ISNet model to generate a segmentation mask and apply it
/// to create a transparent background result.
///
/// # Arguments
///
/// * `input_path` - Path to the input image file
/// * `config` - Configuration for the removal process
///
/// # Returns
///
/// Returns a `RemovalResult` containing the processed image with
/// transparent background and associated metadata.
///
/// # Examples
///
/// ```rust
/// use bg_remove_core::{remove_background, RemovalConfig};
///
/// let config = RemovalConfig::builder().build()?;
/// let result = remove_background("image.jpg", &config).await?;
/// result.save_png("output.png")?;
/// ```
pub async fn remove_background(
    input_path: &str,
    config: &RemovalConfig,
) -> Result<RemovalResult> {
    // Implementation
}
```

## Tools and Scripts

### Local Development

```bash
# Format all code
cargo fmt --all

# Check formatting without changing files
cargo fmt --all -- --check

# Use the lint script with formatting
./bin/lint.sh --format        # Format and run all checks
./bin/lint.sh --fix           # Format and auto-fix issues
```

### Pre-commit Hooks

Install the pre-commit hook for automatic validation:

```bash
# Install the hook
cp bin/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Enable automatic formatting (optional)
export PRE_COMMIT_AUTO_FORMAT=true
export PRE_COMMIT_AUTO_FIX=true
```

### CI/CD Integration

The GitHub Actions workflow automatically:

1. **Validates formatting** on all PRs and pushes
2. **Auto-formats PRs** when formatting issues are detected
3. **Provides formatting diffs** in PR comments when issues exist
4. **Blocks merging** if formatting is incorrect

## IDE Integration

### VS Code

Add to your `.vscode/settings.json`:

```json
{
  "rust-analyzer.rustfmt.extraArgs": ["--config-path", "./rustfmt.toml"],
  "editor.formatOnSave": true,
  "editor.formatOnPaste": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

### JetBrains IntelliJ IDEA / CLion

1. Go to **Settings** → **Languages & Frameworks** → **Rust** → **Rustfmt**
2. Enable "Use rustfmt instead of built-in formatter"
3. Set the path to your `rustfmt.toml` file
4. Enable "Run rustfmt on Save"

### Vim/Neovim

Add to your configuration:

```vim
" Auto-format on save
autocmd BufWritePre *.rs lua vim.lsp.buf.formatting_sync(nil, 200)

" Or with vim-rust plugin
let g:rustfmt_autosave = 1
let g:rustfmt_command = 'cargo fmt --'
```

## Formatting Philosophy

### Why Automatic Formatting?

1. **Consistency**: Eliminates style debates and ensures uniform code
2. **Focus**: Developers can focus on logic instead of style
3. **Review Quality**: Code reviews focus on functionality, not formatting
4. **Onboarding**: New contributors don't need to learn style guidelines
5. **Maintenance**: Reduces formatting-related churn in git history

### Configuration Rationale

- **100-character line limit**: Balances readability with modern screen sizes
- **Vertical import layout**: Easier to scan and reduces merge conflicts
- **Tall parameter layout**: Improves readability for complex function signatures
- **Trailing commas**: Cleaner diffs when adding items to lists
- **Comment wrapping**: Ensures documentation stays readable

## Troubleshooting

### Common Issues

1. **Formatting fails in CI**
   ```bash
   # Run locally to see the issue
   cargo fmt --all -- --check --verbose
   
   # Fix formatting
   cargo fmt --all
   ```

2. **Rustfmt not found**
   ```bash
   # Install rustfmt component
   rustup component add rustfmt
   ```

3. **Configuration not applied**
   ```bash
   # Verify configuration is loaded
   rustfmt --print-config current_dir
   ```

4. **IDE not using project config**
   - Ensure your IDE is pointed to the project's `rustfmt.toml`
   - Restart your IDE after configuration changes

### Debug Commands

```bash
# Show effective configuration
rustfmt --print-config current_dir

# Format with verbose output
cargo fmt --all -- --verbose

# Check specific file
rustfmt --check src/lib.rs

# Format specific file
rustfmt src/lib.rs
```

## Maintenance

The formatting configuration should be reviewed periodically:

1. **Rustfmt updates**: New rustfmt versions may add new options
2. **Team feedback**: Adjust rules based on developer experience
3. **Codebase evolution**: Rules may need adjustment as code patterns change

To propose formatting changes:

1. Update `rustfmt.toml`
2. Run `cargo fmt --all` to apply to entire codebase
3. Create PR with rationale for changes
4. Get team consensus before merging

## Resources

- [rustfmt Documentation](https://rust-lang.github.io/rustfmt/)
- [rustfmt Configuration Options](https://rust-lang.github.io/rustfmt/?version=master&search=#Configuration)
- [EditorConfig Specification](https://editorconfig.org/)
- [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/README.html)