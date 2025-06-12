# Implementation Plan: Runtime Model Selection with Multiple Embedded Models

## Objective
Enable CLI `--model` parameter to:
1. **Runtime model selection** from embedded models by name
2. **External model loading** from folder paths with model.json
3. **Multiple embedded models** via feature flags (not mutually exclusive)
4. **No embedded models by default** - models loaded at runtime only

## Current vs Proposed Architecture

### Current System
- **Single embedded model** at compile time via mutually exclusive features
- **Compile-time model selection** only
- **No runtime model choice**

### Proposed System
- **Multiple embedded models** via additive feature flags
- **Runtime model selection** via `--model` parameter
- **External model loading** from filesystem
- **Default: no embedded models** - pure runtime loading

## CLI Usage Design

```bash
# Use embedded model by name
./bg-remove input.jpg --model isnet-fp16

# Use external model folder
./bg-remove input.jpg --model /path/to/custom-model/

# Use embedded model (if available) or error
./bg-remove input.jpg --model birefnet-fp32

# Default behavior - error if no --model specified
./bg-remove input.jpg  # Error: --model parameter required
```

## Variant Selection for External Model Paths

### CLI Arguments
```rust
#[derive(Parser)]
pub struct Args {
    #[clap(long, help = "Model name or path to model folder")]
    model: String,
    
    #[clap(long, help = "Model variant (fp16, fp32). Defaults to fp16")]
    variant: Option<String>,
}
```

### Variant Selection Methods (Hybrid Approach)
Support multiple methods with precedence rules:

```bash
# Method 1: Separate parameter (highest precedence)
./bg-remove input.jpg --model /path/to/model/ --variant fp32

# Method 2: Suffix syntax (medium precedence)  
./bg-remove input.jpg --model /path/to/model/:fp16
./bg-remove input.jpg --model isnet:fp32

# Method 3: Auto-detection (lowest precedence)
./bg-remove input.jpg --model /path/to/model/  # Auto-detects available variants
```

### Model Source Parsing
```rust
#[derive(Debug)]
pub struct ModelSpec {
    pub source: ModelSource,
    pub variant: Option<String>,
}

#[derive(Debug)]
pub enum ModelSource {
    Embedded(String),      // "isnet", "birefnet"
    External(PathBuf),     // "/path/to/model/"
}

fn parse_model_spec(model_arg: &str) -> ModelSpec {
    // Check for suffix syntax: "model:variant"
    if let Some((path_part, variant_part)) = model_arg.split_once(':') {
        let source = if Path::new(path_part).exists() {
            ModelSource::External(PathBuf::from(path_part))
        } else {
            ModelSource::Embedded(path_part.to_string())
        };
        
        return ModelSpec {
            source,
            variant: Some(variant_part.to_string()),
        };
    }
    
    // No suffix - determine source type
    let source = if Path::new(model_arg).exists() {
        ModelSource::External(PathBuf::from(model_arg))
    } else {
        ModelSource::Embedded(model_arg.to_string())
    };
    
    ModelSpec {
        source,
        variant: None,
    }
}
```

### Variant Resolution Logic
```rust
fn resolve_variant(
    model_spec: &ModelSpec, 
    cli_variant: Option<&str>,
    available_variants: &[String]
) -> Result<String> {
    // Precedence: CLI param > suffix > auto-detection > default
    
    // 1. CLI parameter has highest precedence
    if let Some(variant) = cli_variant {
        if available_variants.contains(&variant.to_string()) {
            return Ok(variant.to_string());
        } else {
            return Err(Error::VariantNotAvailable {
                requested: variant.to_string(),
                available: available_variants.to_vec(),
            });
        }
    }
    
    // 2. Suffix syntax has medium precedence
    if let Some(variant) = &model_spec.variant {
        if available_variants.contains(variant) {
            return Ok(variant.clone());
        } else {
            return Err(Error::VariantNotAvailable {
                requested: variant.clone(),
                available: available_variants.to_vec(),
            });
        }
    }
    
    // 3. Auto-detection: prefer fp16, fallback to available
    if available_variants.contains(&"fp16".to_string()) {
        return Ok("fp16".to_string());
    }
    
    if available_variants.contains(&"fp32".to_string()) {
        return Ok("fp32".to_string());
    }
    
    // 4. Use first available variant
    if let Some(first) = available_variants.first() {
        return Ok(first.clone());
    }
    
    Err(Error::NoVariantsAvailable)
}
```

## Feature Flag Design

### New Additive Feature System
```toml
[features]
# Default: no embedded models
default = []

# Individual model embedding (additive, not mutually exclusive)
embed-isnet-fp16 = []
embed-isnet-fp32 = []
embed-birefnet-fp16 = []
embed-birefnet-fp32 = []

# Convenience groups
embed-all-isnet = ["embed-isnet-fp16", "embed-isnet-fp32"]
embed-all-birefnet = ["embed-birefnet-fp16", "embed-birefnet-fp32"]
embed-all = ["embed-all-isnet", "embed-all-birefnet"]
```

## Implementation Phases

### Phase 1: CLI Parameter Design
**Goal:** Add `--model` parameter with validation

**Changes:**
1. **Update CLI args** in `main.rs`:
   ```rust
   #[clap(long, help = "Model name or path to model folder")]
   model: String,
   
   #[clap(long, help = "Model variant (fp16, fp32). Defaults to fp16")]
   variant: Option<String>,
   ```

2. **Model identifier parsing**:
   ```rust
   enum ModelSource {
       Embedded(String),      // "isnet-fp16"
       External(PathBuf),     // "/path/to/model/"
   }
   
   fn parse_model_source(model_arg: &str) -> ModelSource {
       if Path::new(model_arg).exists() {
           ModelSource::External(PathBuf::from(model_arg))
       } else {
           ModelSource::Embedded(model_arg.to_string())
       }
   }
   ```

### Phase 2: Multiple Embedded Models Support
**Goal:** Allow multiple models to be embedded simultaneously

**Changes:**
1. **Update build.rs** for additive features:
   ```rust
   fn generate_embedded_models() -> String {
       let mut embedded_models = Vec::new();
       
       if cfg!(feature = "embed-isnet-fp16") {
           embedded_models.push(("isnet-fp16", "isnet", "fp16"));
       }
       if cfg!(feature = "embed-birefnet-fp32") {
           embedded_models.push(("birefnet-fp32", "birefnet_portrait", "fp32"));
       }
       // ... etc
       
       // Generate registry of all embedded models
   }
   ```

2. **Embedded model registry**:
   ```rust
   // Generated by build.rs
   pub struct EmbeddedModelRegistry;
   
   impl EmbeddedModelRegistry {
       pub fn get_model(name: &str) -> Option<EmbeddedModelData> {
           match name {
               "isnet-fp16" => Some(load_isnet_fp16()),
               "birefnet-fp32" => Some(load_birefnet_fp32()),
               _ => None,
           }
       }
       
       pub fn list_available() -> &'static [&'static str] {
           &["isnet-fp16", "birefnet-fp32"] // Generated list
       }
   }
   ```

### Phase 3: Runtime Model Loading
**Goal:** Load models from external paths at runtime

**Changes:**
1. **External model provider**:
   ```rust
   pub struct ExternalModelProvider {
       model_path: PathBuf,
       config: ModelConfig,
   }
   
   impl ExternalModelProvider {
       pub fn from_path(path: PathBuf) -> Result<Self> {
           let config_path = path.join("model.json");
           let config: ModelConfig = serde_json::from_str(
               &std::fs::read_to_string(config_path)?
           )?;
           
           Ok(Self { model_path: path, config })
       }
   }
   
   impl ModelProvider for ExternalModelProvider {
       fn load_model_data(&self, variant: &str) -> Result<Vec<u8>> {
           let model_file = format!("model_{}.onnx", variant);
           let model_path = self.model_path.join(model_file);
           std::fs::read(model_path).map_err(Into::into)
       }
   }
   ```

2. **Runtime model.json parsing**:
   ```rust
   #[derive(Deserialize)]
   pub struct ModelConfig {
       pub name: String,
       pub variants: HashMap<String, VariantConfig>,
       pub preprocessing: PreprocessingConfig,
   }
   
   // Move from compile-time constants to runtime values
   ```

3. **Auto-detection for External Models**:
   ```rust
   impl ExternalModelProvider {
       pub fn detect_available_variants(path: &Path) -> Result<Vec<String>> {
           let mut variants = Vec::new();
           
           // Check for standard variant files
           for variant in &["fp16", "fp32"] {
               let model_file = path.join(format!("model_{}.onnx", variant));
               if model_file.exists() {
                   variants.push(variant.to_string());
               }
           }
           
           // Also check model.json for defined variants
           let config_path = path.join("model.json");
           if config_path.exists() {
               let config: ModelConfig = serde_json::from_str(
                   &std::fs::read_to_string(config_path)?
               )?;
               
               for variant_name in config.variants.keys() {
                   if !variants.contains(variant_name) {
                       variants.push(variant_name.clone());
                   }
               }
           }
           
           Ok(variants)
       }
   }
   ```

### Phase 4: Unified ModelManager
**Goal:** Single interface for embedded and external models

**Changes:**
1. **Enhanced ModelManager**:
   ```rust
   pub enum ModelManager {
       Embedded {
           name: String,
           data: EmbeddedModelData,
       },
       External {
           provider: ExternalModelProvider,
           variant: String,
       },
   }
   
   impl ModelManager {
       pub fn from_cli_args(model_arg: &str, variant_arg: Option<&str>) -> Result<Self> {
           let model_spec = parse_model_spec(model_arg);
           
           match model_spec.source {
               ModelSource::Embedded(name) => {
                   let data = EmbeddedModelRegistry::get_model(&name)
                       .ok_or_else(|| Error::ModelNotFound(name))?;
                   Ok(ModelManager::Embedded { name, data })
               },
               ModelSource::External(path) => {
                   let provider = ExternalModelProvider::from_path(path)?;
                   let available_variants = provider.detect_available_variants()?;
                   let variant = resolve_variant(&model_spec, variant_arg, &available_variants)?;
                   Ok(ModelManager::External { provider, variant })
               }
           }
       }
   }
   ```

### Phase 5: Configuration Abstraction
**Goal:** Unified preprocessing pipeline for both embedded and external models

**Changes:**
1. **Runtime configuration**:
   ```rust
   pub trait ModelConfiguration {
       fn get_preprocessing_config(&self) -> &PreprocessingConfig;
       fn get_input_name(&self) -> &str;
       fn get_output_name(&self) -> &str;
       fn get_input_shape(&self) -> [usize; 4];
   }
   
   impl ModelConfiguration for ModelManager {
       fn get_preprocessing_config(&self) -> &PreprocessingConfig {
           match self {
               ModelManager::Embedded { data, .. } => &data.config.preprocessing,
               ModelManager::External { provider, .. } => &provider.config.preprocessing,
           }
       }
   }
   ```

2. **Update ImageProcessor**:
   ```rust
   impl ImageProcessor {
       pub fn with_model_config(config: &dyn ModelConfiguration) -> Self {
           // Use runtime config instead of compile-time constants
       }
   }
   ```

## Error Handling
```rust
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found: {name}")]
    ModelNotFound { name: String },
    
    #[error("Variant '{requested}' not available. Available variants: {available:?}")]
    VariantNotAvailable { 
        requested: String, 
        available: Vec<String> 
    },
    
    #[error("No variants available for model")]
    NoVariantsAvailable,
    
    #[error("Invalid model path: {path}")]
    InvalidModelPath { path: String },
}
```

## Usage Examples

```bash
# External model examples
./bg-remove input.jpg --model /models/custom-isnet/                    # Auto fp16
./bg-remove input.jpg --model /models/custom-isnet/ --variant fp32     # Explicit fp32
./bg-remove input.jpg --model /models/custom-isnet/:fp32              # Suffix fp32

# Embedded model examples  
./bg-remove input.jpg --model isnet                                   # Auto fp16
./bg-remove input.jpg --model isnet --variant fp32                    # Explicit fp32
./bg-remove input.jpg --model isnet:fp32                             # Suffix fp32

# Mixed examples
./bg-remove input.jpg --model isnet:fp16 --variant fp32              # CLI wins: fp32
```

## Breaking Changes

### Breaking Changes
1. **Default behavior**: No model embedded by default - `--model` required
2. **Feature flag names**: Complete redesign with `embed-` prefix
3. **CLI requirement**: `--model` parameter now mandatory
4. **No legacy compatibility**: Clean slate approach

### New Usage Pattern
```bash
# Build with embedded models (optional)
cargo build --features embed-isnet-fp16,embed-birefnet-fp32

# Use embedded model at runtime
./bg-remove input.jpg --model isnet-fp16

# Use external model at runtime  
./bg-remove input.jpg --model /path/to/model/ --variant fp32
```

## Implementation Priority

```
Phase 1 (CLI) → Phase 2 (Multi-embed) → Phase 3 (External) → Phase 4 (Unified) → Phase 5 (Config)
   [1 day]         [2 days]              [3 days]           [2 days]         [1 day]
```

## Benefits

### User Benefits
- **Flexibility**: Choose model at runtime, not compile time
- **External models**: Use custom/updated models without recompilation
- **Model comparison**: Easy A/B testing between models
- **Deployment options**: Ship with multiple models or load externally

### Developer Benefits
- **Simplified builds**: No mutually exclusive feature complexity
- **Easier testing**: Test multiple models in single binary
- **Model development**: External model development workflow
- **Distribution flexibility**: Separate model distribution from binary

## Success Criteria

- [ ] CLI accepts `--model` parameter for embedded and external models
- [ ] Multiple models can be embedded simultaneously via feature flags
- [ ] External model folders load correctly with model.json validation
- [ ] Variant selection works with CLI parameter, suffix syntax, and auto-detection
- [ ] Performance overhead < 5% for model selection logic
- [ ] All existing functionality preserved with new `--model` parameter
- [ ] Clear error messages for invalid model names/paths
- [ ] Precedence rules work correctly: CLI param > suffix > auto-detection

This plan transforms the system from compile-time model binding to flexible runtime model selection while maintaining performance and adding powerful new capabilities.