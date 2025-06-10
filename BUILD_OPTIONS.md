# Build Configuration Options

This document describes the available build-time feature flags for controlling which ONNX model is embedded in the binary.

## Model Precision Features

**Important**: Exactly one model is always embedded. You cannot build with both models or no models.

### Default Configuration (FP16)
```bash
cargo build  # Equivalent to --features fp16-model
```
- **Binary Size**: ~90MB
- **Model**: FP16 ISNet model only
- **Performance**: Balanced speed and memory usage
- **Use Case**: General purpose, recommended for most users

### FP32 Model Only
```bash
cargo build --no-default-features --features fp32-model
```
- **Binary Size**: ~175MB  
- **Model**: FP32 ISNet model only
- **Performance**: Higher precision, slightly better quality
- **Use Case**: Maximum quality requirements

## Legacy Compatibility

For backward compatibility with existing build scripts:

```bash
# Legacy FP16 (equivalent to fp16-model)
cargo build --features isnet-fp16

# Legacy FP32 (equivalent to fp32-model)  
cargo build --features isnet-standard
```

## Performance Comparison

Based on benchmark results on Apple Silicon:

| Configuration | Binary Size | 512x512 Performance | Use Case |
|--------------|-------------|---------------------|----------|
| `fp16-model` | ~90MB       | 619ms (CoreML)      | **Recommended** |
| `fp32-model` | ~175MB      | 609ms (CoreML)      | Quality-focused |

## Examples

### Development (Fast builds)
```bash
cargo build  # Default FP16 model
```

### Production (Optimized)
```bash
cargo build --release  # FP16 model by default
```

### Quality-focused Production
```bash
cargo build --release --no-default-features --features fp32-model
```

## Model File Locations

Models are embedded from:
- `models/isnet_fp16.onnx` (~84MB) - when using `fp16-model` feature
- `models/isnet_fp32.onnx` (~168MB) - when using `fp32-model` feature

The model is compiled directly into the binary at build time.