# Issues



This is a list of open issues. The syntax is as follows
- "[ ]" marks an *open* issues
- "[x]" marks a *done* issue
- "[@]" marks a *delegated* issues, "[@USERNAME] marks a *delegated* issue to a specific agent or user
- "[o]" marks a *on hold* issues
- "[-]" marks a *won't fix* issue


There are also some modifies that are placed in the task description in the description
 - 📅 YYYY-MM-DD HH:SS defines the due date
 - 🔃 every XYZ defines the task to be recurring
 - 🐞 defines the task to be a bug
-
Additional we may use Hashtags "#" to tag, group and categorize Issues.
If an issue needs further information we can create an issue markdown for this particular issue in the docs/issues folder and link it in the issue description


## Open Issues

- [ ] Backends like ort should move to a Wasi NN compat style
- [ ] Idea – Compile with MCP tool/call interface and create platform specific language wrappers with MCP client (even cli)
- [ ] Implement C-lib as universal interface for all platforms
- [ ] Ort support more backends like candle, tract etc., these my be implemented.
- [ ] JSON Schema for the User facing API
- [ ] MCP Interface
- [ ] Emscripten Platform build
- [ ] Wasi1p Platform Build
- [ ] iOS Platform Build
- [ ] Android Platform Build
- [ ] NodeJS Platform build
- [-] Compile individual binaries for each model in the models directory instead of feature flags - Runtime model selection with feature flags is more flexible 📅 2025-06-12 10:35
- [ ] Is there a difference if we apply a sigmoid to the alpha channel estimate?
- [ ] We might want to jailroot the AI and thus let it run without us having to accept everything #YOLOMODE
- [ ] Birefnet fp16 variant is missing (https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/tree/main/onnx)


## Closed Issues

- [x] Docker-based cross-compilation system - COMPLETED: Implemented comprehensive cross-compilation system with target triplet naming, supporting aarch64-unknown-linux-gnu and x86_64-unknown-linux-gnu. Fixed Docker platform warnings and added multi-target build support. 📅 2025-06-13 10:20
- [x] Add comprehensive documentation after multi-model refactoring - COMPLETED: Added comprehensive docs to all public functions with examples, performance metrics, and usage patterns 📅 2025-06-12 23:30
- [x] Allow to specify the model folder to use in the cli as alternative and allow embedding multiple models and choosing one. 📅 2025-06-12 10:30
- [x] Metal MPS performance is way to slow. Are we sure we are running on the GPU on Apple Macs? - RESOLVED: CoreML works well with FP32 models (43% faster) but poorly with FP16 models (7% faster). Need to default to FP32 for CoreML. 📅 2025-06-12 22:08
- [x] Not sure if core-ml is used without feature flag 📅 2025-06-11 13:31
- [x] the tests seem to reference wrong files 📅 2025-06-11 13:46
- [x] Benchmarks skip all 📅 2025-06-11 13:45
- [x] Image is not resized to the original size 📅 2025-06-11 13:31
- [x] Default build seems FP16 📅 2025-06-11 19:20
- [x] InferenceBackend implementations should be in their own file in a module [📋 backend-module-refactoring](./issues/backend-module-refactoring.md)
- [x] Integration tests should use standard Rust patterns instead of separate testing crate binaries 📅 2025-06-11 14:00
- [x] Output timings for "load", "decode", "inference", "encode" seperately [📋 output-timing-breakdown](./issues/output-timing-breakdown.md) 📅 2025-06-11 15:10
- [x] Implement zero-warning policy for all compilation targets 🐞 [📋 zero-warning-policy](./issues/zero-warning-policy.md) [📊 implementation-status](./issues/zero-warning-policy-implementation.md)
- [x] Multi-model support with compile-time model.json configuration [📋 multi-model-support](./issues/multi-model-support.md)
- [x] Add BiRefNet-portrait model support for specialized portrait background removal [📋 birefnet-portrait-support](./issues/birefnet-portrait-support.md)
