# Test Assets Directory

This directory contains real images for comprehensive testing of the background removal library.

## Structure

```
assets/
├── input/                    # Input test images
│   ├── portraits/           # Human portraits (4 images)
│   ├── products/            # E-commerce products (3 images) 
│   ├── complex/             # Complex scenes (3 images)
│   └── edge_cases/          # Edge cases (4 images)
├── expected/                # Reference outputs from JS implementation
│   ├── portraits/           # Expected portrait outputs
│   ├── products/            # Expected product outputs
│   ├── complex/             # Expected complex scene outputs
│   └── edge_cases/          # Expected edge case outputs
├── test_cases.json          # Test case metadata
└── README.md               # This file
```

## Image Categories

### Portraits (4 images)
- **business_headshot.jpg**: Professional headshot with clean background
- **fine_hair_details.jpg**: Portrait with intricate hair details and flyaways  
- **outdoor_natural.jpg**: Outdoor portrait with natural lighting
- **multiple_people.jpg**: Group photo with multiple people

### Products (3 images)  
- **white_background_shoe.jpg**: Product on white background (high accuracy expected)
- **electronics_gradient.jpg**: Electronic device with gradient background
- **clothing_transparent.jpg**: Clothing with transparent/translucent elements

### Complex Scenes (3 images)
- **pet_with_person.jpg**: Person with pet, multiple subjects with fur details
- **glass_objects.jpg**: Scene with transparent glass objects
- **overlapping_subjects.jpg**: Multiple overlapping subjects

### Edge Cases (4 images)
- **very_high_resolution.jpg**: Very high resolution image (4K+) 
- **very_low_resolution.jpg**: Very low resolution image (<200px)
- **high_noise.jpg**: Image with noise and compression artifacts
- **similar_colors.jpg**: Subject and background with similar colors

## Adding New Test Images

1. Place input images in appropriate category subdirectory under `input/`
2. Generate reference outputs using the JavaScript implementation
3. Place reference outputs in corresponding subdirectory under `expected/`
4. Update `test_cases.json` with new test case definitions
5. Run validation: `cargo run --bin validate-assets`

## Image Requirements

- **Formats**: JPEG, PNG, WebP
- **Sizes**: Various (64x64 to 4096x4096)
- **Content**: Real-world images covering diverse scenarios
- **Quality**: High-quality reference images for accurate comparison

## Expected Outputs

Reference outputs should be:
- Generated using the current JavaScript implementation
- PNG format with alpha transparency
- Same dimensions as input images
- Manually verified for quality

## Usage

```bash
# Validate all test assets exist and are valid
cargo run --bin validate-assets

# Download curated test dataset
cargo run --bin download-images --dataset portraits

# Run tests on specific category  
cargo run --bin test-suite --categories portraits

# Generate comparison report
cargo run --bin generate-report --output-dir test_results/
```

## Notes

- Keep images under 10MB each for reasonable test execution time
- Include copyright-free images or properly licensed content
- Maintain consistent naming conventions
- Test images should represent real-world use cases