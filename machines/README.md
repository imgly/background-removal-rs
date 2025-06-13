# Cross-Platform Build Configurations

This directory contains Docker configurations for building bg-remove for different target triplets using Rust's target architecture naming.

## Available Targets

### Linux GNU (glibc)
- **aarch64-unknown-linux-gnu**: ARM64 Linux with glibc (Debian Bookworm)
- **x86_64-unknown-linux-gnu**: x86_64 Linux with glibc (Debian Bookworm)

### Unsupported Targets
- **musl targets**: Currently unsupported due to ONNX Runtime not providing prebuilt binaries for musl libc. This would require compiling ONNX Runtime from source.

## Usage

### Single Target Build
```bash
# Build for specific target
./build-cross.sh x86_64-unknown-linux-gnu
./build-cross.sh aarch64-unknown-linux-gnu
```

### Multiple Target Build
```bash
# Build all available targets
./build-cross.sh --all

# List available targets
./build-cross.sh --list

# Clean and build
./build-cross.sh --clean x86_64-unknown-linux-gnu
```

## Output Structure

Binaries are placed in target-specific directories:
```
target/
├── aarch64-unknown-linux-gnu/bg-remove
└── x86_64-unknown-linux-gnu/bg-remove
```

## Target Details

| Target | Architecture | Libc | Base Image | Binary Size | Dependencies |
|--------|-------------|------|------------|-------------|--------------|
| `aarch64-unknown-linux-gnu` | ARM64 | glibc | Debian Bookworm | ~21MB | Requires glibc 2.36+ |
| `x86_64-unknown-linux-gnu` | x86_64 | glibc | Debian Bookworm | ~21MB | Requires glibc 2.36+ |

## Adding New Targets

To add support for a new target triplet:

1. Create `Dockerfile.{target-triplet}` in this directory
2. Follow the naming convention: `{arch}-{vendor}-{sys}-{abi}`
3. The build script will automatically detect new Dockerfiles

## Requirements

- Docker installed and running with BuildKit support
- Multi-platform Docker support for cross-architecture builds
- Sufficient disk space for build artifacts (~2GB per target)

## Platform Notes

- **glibc targets** require compatible glibc version (2.36+) on target system
- **Cross-architecture builds** require Docker BuildKit and QEMU emulation
- **musl static binaries** are not currently supported due to ONNX Runtime dependency limitations