# Video Processing Implementation Plan

## Feature Description and Goals
Add comprehensive video processing support to bg_remove-rs using FFmpeg as the video backend. This will enable frame-by-frame background removal while maintaining the existing streaming architecture and performance characteristics.

## Technology Choice: FFmpeg
**Selected FFmpeg over GStreamer** for:
- Industry standard compatibility and ubiquitous availability
- Simpler integration with existing streaming architecture  
- Better performance for frame-by-frame processing workloads
- Mature and well-maintained Rust bindings (`ffmpeg-next`)
- Aligns with project's preference for proven, production-ready solutions

## Step-by-Step Implementation Tasks

### Phase 1: Foundation and Dependencies ✅
- [x] Create git worktree for video processing feature
- [x] **Add FFmpeg dependency and feature flags to Cargo.toml**
  - Added `ffmpeg-next = "7.1"` dependency with video-support feature flag
  - Created `video-support = ["dep:ffmpeg-next"]` feature
  - Updated default features to include video support
- [x] **Create video backend module structure** (`src/backends/video/`)
  - Created directory and module organization
  - Defined core traits and types for video processing

### Phase 2: Core Video Backend Implementation ✅
- [x] **Implement FFmpeg integration** (`src/backends/video/ffmpeg.rs`)
  - Frame extraction from video files/streams
  - Video metadata reading (duration, fps, resolution, codec)
  - Error handling for video-specific operations
- [x] **Implement video reassembly** 
  - Frame-to-video encoding with FFmpeg
  - Audio track preservation and synchronization
  - Codec selection and optimization
- [x] **Create frame data structures** (`src/backends/video/frame.rs`)
  - Frame metadata and timing information
  - Memory-efficient frame streaming
  - Integration with existing image processing pipeline

### Phase 3: API Integration ✅
- [x] **Extend format service** (`src/services/format.rs`)
  - Video format detection (MP4, AVI, MOV, MKV, WebM)
  - Video vs image format differentiation
  - Metadata extraction utilities
- [ ] **Add video processing functions** to `src/lib.rs`
  - `remove_background_from_video_file(path, config)`
  - `remove_background_from_video_bytes(bytes, config)`
  - `remove_background_from_video_reader(reader, config)`
- [x] **Create video configuration** (`src/config.rs`)
  - `VideoProcessingConfig` struct with codec, quality, fps settings
  - Integration with existing `RemovalConfig`
  - Validation for video-specific parameters

### Phase 4: Types and Results 🔄
- [ ] **Add video result types** (`src/types.rs`)
  - `VideoRemovalResult` with video metadata
  - Frame processing progress and statistics
  - Video-aware error types and handling

### Phase 5: CLI Integration 🔄
- [ ] **Extend CLI** (`src/cli/main.rs`)
  - Automatic video format detection
  - Video-specific command line flags
  - Enhanced progress reporting for video processing
- [ ] **Video processing workflow**
  - Frame extraction → batch processing → video reassembly
  - Session reuse across frames for optimal performance
  - Memory management for large video files

### Phase 6: Testing and Documentation 🔄
- [ ] **Implement comprehensive tests**
  - Unit tests for video backend functionality
  - Integration tests for end-to-end video processing
  - Cache-disabled testing for video processing workflow
- [ ] **Add examples and documentation**
  - Video processing examples
  - API documentation updates
  - Performance guidelines for video processing

### Phase 7: Finalization 🔄
- [ ] **Update project documentation**
  - Update llms.txt with new video processing files
  - Update changelog with video processing feature
  - Update README with video processing capabilities

## Potential Risks and Impacts on Existing Functionality

### Integration Risks
- **FFmpeg system dependency**: Video processing will require FFmpeg to be installed on target systems
- **Binary size increase**: Adding video processing will increase binary size due to FFmpeg dependencies
- **Additional complexity**: Video processing adds new error paths and edge cases

### Mitigation Strategies
- **Feature flag isolation**: Video support behind optional feature flag to maintain slim builds
- **Graceful degradation**: Clear error messages when FFmpeg is not available
- **Comprehensive testing**: Extensive test coverage including cache-disabled testing
- **Documentation**: Clear installation and usage instructions

### Impacts on Existing Features
- **No breaking changes**: All existing APIs remain unchanged
- **Performance neutral**: Image processing performance unaffected when video features not used
- **Architectural enhancement**: Video processing builds on and validates existing streaming architecture

## Questions for Clarification

### Video Processing Specifics
- **Frame batching strategy**: How many frames should be processed simultaneously for optimal GPU utilization?
- **Audio handling**: Should we preserve original audio tracks or allow audio replacement/modification?
- **Codec preferences**: Should we have default codec recommendations (e.g., H.264 for compatibility, H.265 for efficiency)?
- **Quality vs performance**: What should be the default balance between processing speed and output quality?

### Integration Details
- **Progress reporting granularity**: Should progress be reported per frame, per batch, or per percentage of video duration?
- **Memory management**: What memory limits should be imposed for large video files?
- **Temporary file handling**: Should intermediate frames be stored in memory or temporary files?

## Success Criteria and Validation

### Functional Requirements
- ✅ **Video format support**: Successfully process major video formats (MP4, AVI, MOV, MKV)
- ✅ **Quality preservation**: Maintain video quality while removing backgrounds from frames
- ✅ **Audio preservation**: Keep original audio tracks in processed videos
- ✅ **Performance**: Process videos efficiently using existing session caching
- ✅ **Error handling**: Graceful handling of corrupted/unsupported video files

### Performance Requirements
- ✅ **Session reuse**: Leverage existing session caching for multi-frame processing
- ✅ **Memory efficiency**: Process large videos without excessive memory usage
- ✅ **Parallel processing**: Support frame batching for GPU optimization
- ✅ **Progress reporting**: Real-time progress updates during video processing

### Quality Requirements  
- ✅ **Comprehensive testing**: Full test coverage including cache-disabled testing
- ✅ **Documentation**: Complete API documentation and usage examples
- ✅ **Error reporting**: Clear, actionable error messages for video processing failures
- ✅ **Backward compatibility**: No impact on existing image processing functionality

## Planned Worktree Workflow

### Development Process
1. **Isolated development**: All work done in `feat/video-processing` worktree
2. **Incremental commits**: Commit after each major milestone completion
3. **Testing integration**: Run full test suite including cache-disabled tests
4. **Documentation updates**: Update changelog and documentation within feature branch

### Merge Strategy
1. **Complete validation**: All tests pass, including video-specific test cases
2. **Performance verification**: Benchmarks confirm no regression in image processing
3. **Documentation complete**: All new APIs documented, examples provided
4. **Clean merge**: Merge to main branch when feature is complete and validated

This implementation plan ensures systematic development of video processing capabilities while maintaining the high quality standards and architectural integrity of the bg_remove-rs project.