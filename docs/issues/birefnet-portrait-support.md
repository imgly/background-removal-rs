# BiRefNet-Portrait Model Support

## Objective
Add support for the BiRefNet-portrait model, a state-of-the-art model specialized for portrait background removal and dichotomous image segmentation.

## Model Information

### Source
- **Repository**: https://huggingface.co/onnx-community/BiRefNet-portrait-ONNX
- **Base Model**: ZhengPeng7/BiRefNet-portrait
- **License**: MIT
- **Paper**: "Bilateral Reference for High-Resolution Dichotomous Image Segmentation" (CAAI AIR'24)

### Model Capabilities
- **Primary Use**: Portrait background removal
- **Specialization**: Dichotomous image segmentation 
- **Advanced Features**: Camouflaged object detection, salient object detection
- **Quality**: State-of-the-art results for portrait segmentation

## Technical Specifications

### Model Variants Available
- **FP32 Model**: `model.onnx` (928 MB)
- **FP16 Model**: `model_fp16.onnx` (467 MB)

### Input/Output Specification
- **Input Tensor Name**: `"input_image"`
- **Output Tensor Name**: `"output_image"`
- **Input Shape**: `[1, 3, 1024, 1024]` (NCHW format)
- **Output Shape**: `[1, 1, 1024, 1024]` (single-channel mask)

### Preprocessing Configuration (from preprocessor_config.json)
```json
{
  "target_size": [1024, 1024],
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "rescale_factor": 0.00392156862745098,
  "do_rescale": true,
  "do_resize": true,
  "do_normalize": true
}
```

### Key Differences from ISNet
| Aspect | ISNet | BiRefNet-Portrait |
|--------|--------|-------------------|
| **Normalization Mean** | [128.0, 128.0, 128.0] | [0.485, 0.456, 0.406] |
| **Normalization Std** | [256.0, 256.0, 256.0] | [0.229, 0.224, 0.225] |
| **Input Tensor Name** | `"input"` | `"input_image"` |
| **Output Tensor Name** | `"output"` | `"output_image"` |
| **Target Size** | 1024x1024 | 1024x1024 (same) |
| **Rescaling** | Direct pixel values | Rescale factor: 1/255 |
| **Specialization** | General segmentation | Portrait-optimized |

## Implementation Plan

### Phase 1: Model Integration Setup
1. **Create model directory structure**:
   ```
   models/birefnet_portrait/
   ├── model.json
   ├── model_fp16.onnx  (467 MB)
   └── model_fp32.onnx  (928 MB)
   ```

2. **Create model.json configuration**:
   ```json
   {
     "name": "BiRefNet-Portrait",
     "variants": {
       "fp16": {
         "input_shape": [1, 3, 1024, 1024],
         "output_shape": [1, 1, 1024, 1024],
         "input_name": "input_image",
         "output_name": "output_image"
       },
       "fp32": {
         "input_shape": [1, 3, 1024, 1024],
         "output_shape": [1, 1, 1024, 1024],
         "input_name": "input_image",
         "output_name": "output_image"
       }
     },
     "preprocessing": {
       "target_size": [1024, 1024],
       "normalization": {
         "mean": [0.485, 0.456, 0.406],
         "std": [0.229, 0.224, 0.225]
       },
       "rescale_factor": 0.00392156862745098
     }
   }
   ```

### Phase 2: Feature Flag Integration
3. **Add new Cargo features**:
   ```toml
   birefnet-fp16 = ["model-birefnet", "precision-fp16"]
   birefnet-fp32 = ["model-birefnet", "precision-fp32"]
   model-birefnet = []
   ```

4. **Update build script** to support BiRefNet model selection and rescaling factor

### Phase 3: Preprocessing Updates
5. **Extend preprocessing pipeline** to handle rescale factor
6. **Update normalization** to use ImageNet-style normalization for BiRefNet
7. **Ensure compatibility** with both ISNet and BiRefNet preprocessing modes

### Phase 4: Testing and Validation
8. **Download model files** from HuggingFace repository
9. **Create test suite** comparing BiRefNet vs ISNet results on portrait images
10. **Performance benchmarking** between models
11. **Validate preprocessing** matches reference implementation

## Expected Benefits

### Performance Advantages
- **Specialized for portraits**: Better edge detection around hair and fine details
- **Improved accuracy**: State-of-the-art results for human subject segmentation
- **Modern architecture**: BiRefNet uses more advanced segmentation techniques

### Use Case Optimization
- **Portrait photography**: Professional portrait background removal
- **Video conferencing**: Real-time background replacement
- **E-commerce**: Product photos with people
- **Social media**: Automated background effects

## Implementation Considerations

### Build Script Changes
- **Multi-model support**: Extend existing build.rs to handle different normalization schemes
- **Rescale factor**: Add support for preprocessing rescale_factor parameter
- **Model detection**: Auto-detect model type from feature flags

### Backward Compatibility
- **No breaking changes**: Existing ISNet functionality remains unchanged
- **Feature flag isolation**: BiRefNet is completely separate feature set
- **Configuration isolation**: Each model has independent preprocessing config

### Performance Impact
- **Model size**: BiRefNet FP16 (467MB) vs ISNet FP16 (84MB) - significantly larger
- **Inference speed**: Expected to be similar to ISNet for 1024x1024 input
- **Memory usage**: Higher due to larger model size

## Success Criteria

### Technical Validation
- [ ] BiRefNet models build successfully with new feature flags
- [ ] Preprocessing matches reference ImageNet normalization
- [ ] Input/output tensor shapes correctly configured
- [ ] Model inference produces valid segmentation masks

### Quality Validation  
- [ ] BiRefNet shows improved portrait segmentation vs ISNet
- [ ] Fine details (hair, edges) are better preserved
- [ ] No regression in build time or existing ISNet functionality
- [ ] Documentation and examples updated

### Integration Validation
- [ ] CLI accepts new `--features birefnet-fp16` flag
- [ ] Zero runtime overhead maintained
- [ ] Generated constants include rescale_factor support
- [ ] model.json schema handles all BiRefNet-specific parameters

## Future Extensions

### Additional BiRefNet Variants
- **BiRefNet-general**: General purpose segmentation
- **BiRefNet-lite**: Lightweight version for mobile/edge deployment
- **BiRefNet-HR**: High-resolution 2048x2048 support

### Advanced Features
- **Multi-resolution support**: Dynamic input sizes
- **Batch processing**: Multiple images in single inference
- **Post-processing**: Edge refinement and smoothing options

This implementation will position the background removal system as a comprehensive solution supporting both general-purpose (ISNet) and specialized portrait (BiRefNet) segmentation models.