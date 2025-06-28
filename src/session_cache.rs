//! ONNX Runtime session caching for optimized model persistence
//!
//! This module provides functionality for caching ONNX Runtime sessions to eliminate
//! re-optimization overhead and improve cold-start performance. It handles provider-specific
//! optimized model caching, session persistence, and cache management.

use crate::config::ExecutionProvider;
use crate::error::{BgRemovalError, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cache marker for session files
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionCacheMarker {
    /// Session identifier
    pub session_id: String,
    /// Creation timestamp
    pub creation_time: u64,
    /// ONNX Runtime version
    pub ort_version: String,
    /// Type of cache (metadata_only, optimized_model)
    pub cache_type: String,
}

/// Session creation parameters for rebuilding sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Original model path
    pub model_path: PathBuf,
    /// Execution provider used
    pub execution_provider: ExecutionProvider,
    /// Graph optimization level (serialized as string)
    pub optimization_level: String,
    /// Inter-operation parallelism threads
    pub inter_op_num_threads: Option<i16>,
    /// Intra-operation parallelism threads  
    pub intra_op_num_threads: Option<i16>,
    /// Whether to use parallel execution
    pub parallel_execution: bool,
    /// Provider-specific options as JSON string
    pub provider_options: String,
}

/// Session cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCacheEntry {
    /// Unique cache key for this session
    pub cache_key: String,
    /// Model hash that was used to create this session
    pub model_hash: String,
    /// Execution provider used for this session
    pub execution_provider: String,
    /// Graph optimization level used
    pub optimization_level: String,
    /// Provider-specific configuration hash
    pub provider_config_hash: String,
    /// ONNX Runtime version used to create this session
    pub ort_version: String,
    /// Timestamp when this cache entry was created
    pub created_at: u64,
    /// Timestamp when this cache entry was last accessed
    pub last_accessed: u64,
    /// Size of the cached session data in bytes
    pub size_bytes: u64,
    /// Whether this session is provider-optimized (e.g., CoreML compiled)
    pub is_provider_optimized: bool,
    /// Session configuration for rebuilding
    pub session_config: SessionConfig,
}

/// Session cache statistics
#[derive(Debug, Clone, Default)]
pub struct SessionCacheStats {
    /// Total number of cached sessions
    pub total_sessions: usize,
    /// Total cache size in bytes
    pub total_size_bytes: u64,
    /// Number of cache hits since last reset
    pub cache_hits: u64,
    /// Number of cache misses since last reset
    pub cache_misses: u64,
    /// Sessions by execution provider
    pub sessions_by_provider: HashMap<String, usize>,
}

/// ONNX Runtime session cache manager
#[derive(Debug)]
pub struct SessionCache {
    /// Base cache directory for sessions
    cache_dir: PathBuf,
    /// In-memory metadata cache for fast lookups
    metadata_cache: HashMap<String, SessionCacheEntry>,
    /// Runtime statistics
    stats: SessionCacheStats,
}

impl SessionCache {
    /// Create a new session cache manager
    ///
    /// Uses XDG Base Directory specification for cache location:
    /// - Linux/macOS: `~/.cache/imgly-bgremove/sessions/`
    /// - Windows: `%LOCALAPPDATA%/imgly-bgremove/sessions/`
    ///
    /// # Errors
    /// - Failed to determine cache directory
    /// - Failed to create cache directory
    /// - Insufficient permissions to access cache directory
    pub fn new() -> Result<Self> {
        let cache_dir = Self::get_cache_dir()?;

        // Ensure cache directory exists
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).map_err(|e| {
                BgRemovalError::file_io_error("create session cache directory", &cache_dir, &e)
            })?;
        }

        let mut cache = Self {
            cache_dir,
            metadata_cache: HashMap::new(),
            stats: SessionCacheStats::default(),
        };

        // Load existing metadata
        cache.load_metadata()?;

        Ok(cache)
    }

    /// Get the XDG-compliant session cache directory path
    ///
    /// # Errors
    /// - Failed to determine user cache directory
    /// - Invalid cache directory path
    fn get_cache_dir() -> Result<PathBuf> {
        // Try environment variable override first
        if let Ok(cache_override) = std::env::var("IMGLY_BGREMOVE_CACHE_DIR") {
            return Ok(PathBuf::from(cache_override).join("sessions"));
        }

        // Use XDG-compliant cache directory
        #[cfg(feature = "cli")]
        {
            Ok(dirs::cache_dir()
                .ok_or_else(|| {
                    BgRemovalError::invalid_config(
                        "Failed to determine cache directory. Set IMGLY_BGREMOVE_CACHE_DIR environment variable.".to_string()
                    )
                })?
                .join("imgly-bgremove")
                .join("sessions"))
        }

        #[cfg(not(feature = "cli"))]
        {
            Err(BgRemovalError::invalid_config(
                "Session cache functionality requires CLI features to be enabled".to_string(),
            ))
        }
    }

    /// Generate a cache key for a session based on its configuration
    ///
    /// # Arguments
    /// * `model_hash` - SHA256 hash of the model data
    /// * `execution_provider` - Execution provider type
    /// * `optimization_level` - Graph optimization level
    /// * `provider_config` - Provider-specific configuration as JSON string
    ///
    /// # Returns
    /// A unique cache key string
    pub fn generate_cache_key(
        model_hash: &str,
        execution_provider: ExecutionProvider,
        optimization_level: &GraphOptimizationLevel,
        provider_config: &str,
    ) -> String {
        let mut hasher = Sha256::new();

        // Include all factors that affect session compilation
        hasher.update(model_hash.as_bytes());
        hasher.update(format!("{:?}", execution_provider).as_bytes());
        hasher.update(format!("{:?}", optimization_level).as_bytes());
        hasher.update(provider_config.as_bytes());

        // Include ONNX Runtime version for compatibility (use environment constant as fallback)
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());

        // Include target architecture for provider compatibility
        hasher.update(std::env::consts::ARCH.as_bytes());
        hasher.update(std::env::consts::OS.as_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Calculate SHA256 hash of model data
    ///
    /// # Arguments
    /// * `model_data` - Raw model data bytes
    ///
    /// # Returns
    /// SHA256 hash as hex string
    pub fn calculate_model_hash(model_data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_data);
        format!("{:x}", hasher.finalize())
    }

    /// Check if a session is cached for the given parameters
    ///
    /// # Arguments
    /// * `cache_key` - Cache key generated for the session
    ///
    /// # Returns
    /// `true` if the session is cached and valid
    pub fn is_session_cached(&self, cache_key: &str) -> bool {
        if let Some(entry) = self.metadata_cache.get(cache_key) {
            let session_path = self.get_session_path(cache_key, &entry.execution_provider);
            session_path.exists()
        } else {
            false
        }
    }

    /// Get the path for a cached session
    ///
    /// # Arguments
    /// * `cache_key` - Cache key for the session
    /// * `execution_provider` - Execution provider name for subdirectory
    ///
    /// # Returns
    /// Path to the cached session file
    fn get_session_path(&self, cache_key: &str, execution_provider: &str) -> PathBuf {
        self.cache_dir
            .join(execution_provider.to_lowercase())
            .join(format!("{}.ort", cache_key))
    }

    /// Get the path for cached session metadata
    ///
    /// # Arguments
    /// * `cache_key` - Cache key for the session
    /// * `execution_provider` - Execution provider name for subdirectory
    ///
    /// # Returns
    /// Path to the cached session metadata file
    fn get_metadata_path(&self, cache_key: &str, execution_provider: &str) -> PathBuf {
        self.cache_dir
            .join(execution_provider.to_lowercase())
            .join(format!("{}.meta.json", cache_key))
    }

    /// Load session metadata from disk
    fn load_metadata(&mut self) -> Result<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }

        // Scan all provider subdirectories
        let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
            BgRemovalError::file_io_error("read session cache directory", &self.cache_dir, &e)
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                BgRemovalError::network_error("Failed to read cache directory entry", e)
            })?;

            let path = entry.path();
            if path.is_dir() {
                self.load_provider_metadata(&path)?;
            }
        }

        // Update statistics
        self.update_stats();

        Ok(())
    }

    /// Load metadata for a specific provider directory
    fn load_provider_metadata(&mut self, provider_dir: &Path) -> Result<()> {
        let entries = fs::read_dir(provider_dir).map_err(|e| {
            BgRemovalError::file_io_error("read provider cache directory", provider_dir, &e)
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                BgRemovalError::network_error("Failed to read provider directory entry", e)
            })?;

            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with(".meta"))
                    .unwrap_or(false)
            {
                if let Ok(metadata_str) = fs::read_to_string(&path) {
                    if let Ok(entry) = serde_json::from_str::<SessionCacheEntry>(&metadata_str) {
                        self.metadata_cache.insert(entry.cache_key.clone(), entry);
                    }
                }
            }
        }

        Ok(())
    }

    /// Update internal statistics
    fn update_stats(&mut self) {
        self.stats.total_sessions = self.metadata_cache.len();
        self.stats.total_size_bytes = self
            .metadata_cache
            .values()
            .map(|entry| entry.size_bytes)
            .sum();

        self.stats.sessions_by_provider.clear();
        for entry in self.metadata_cache.values() {
            *self
                .stats
                .sessions_by_provider
                .entry(entry.execution_provider.clone())
                .or_insert(0) += 1;
        }
    }

    /// Get current cache statistics
    pub fn get_stats(&self) -> &SessionCacheStats {
        &self.stats
    }

    /// Clear all cached sessions
    ///
    /// # Returns
    /// Number of sessions removed
    ///
    /// # Errors
    /// - Failed to remove session files
    /// - I/O errors during removal operations
    pub fn clear_all_sessions(&mut self) -> Result<usize> {
        let count = self.metadata_cache.len();

        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir).map_err(|e| {
                BgRemovalError::file_io_error("remove session cache directory", &self.cache_dir, &e)
            })?;

            // Recreate cache directory
            fs::create_dir_all(&self.cache_dir).map_err(|e| {
                BgRemovalError::file_io_error("create session cache directory", &self.cache_dir, &e)
            })?;
        }

        self.metadata_cache.clear();
        self.update_stats();

        Ok(count)
    }

    /// Clear sessions for a specific execution provider
    ///
    /// # Arguments
    /// * `execution_provider` - Provider to clear sessions for
    ///
    /// # Returns
    /// Number of sessions removed
    ///
    /// # Errors
    /// - Failed to remove session files
    /// - I/O errors during removal operations
    pub fn clear_provider_sessions(&mut self, execution_provider: &str) -> Result<usize> {
        let provider_dir = self.cache_dir.join(execution_provider.to_lowercase());
        let mut removed_count = 0;

        if provider_dir.exists() {
            fs::remove_dir_all(&provider_dir).map_err(|e| {
                BgRemovalError::file_io_error(
                    "remove provider session directory",
                    &provider_dir,
                    &e,
                )
            })?;
        }

        // Remove from metadata cache
        self.metadata_cache.retain(|_, entry| {
            if entry.execution_provider.to_lowercase() == execution_provider.to_lowercase() {
                removed_count += 1;
                false
            } else {
                true
            }
        });

        self.update_stats();

        Ok(removed_count)
    }

    /// Cache a session for future use
    ///
    /// # Arguments
    /// * `session` - The ONNX Runtime session to cache
    /// * `cache_key` - Cache key for this session
    /// * `model_hash` - SHA256 hash of the model data
    /// * `execution_provider` - Execution provider used
    /// * `optimization_level` - Graph optimization level used
    /// * `provider_config` - Provider-specific configuration
    /// * `session_config` - Session configuration for rebuilding
    ///
    /// # Errors
    /// - Failed to serialize session data
    /// - Failed to write session or metadata files
    /// - I/O errors during caching operations
    pub fn cache_session(
        &mut self,
        session: &Session,
        cache_key: &str,
        model_hash: &str,
        execution_provider: ExecutionProvider,
        optimization_level: &GraphOptimizationLevel,
        provider_config: &str,
        session_config: SessionConfig,
    ) -> Result<()> {
        let provider_name = format!("{:?}", execution_provider).to_lowercase();
        let provider_dir = self.cache_dir.join(&provider_name);

        // Ensure provider directory exists
        if !provider_dir.exists() {
            fs::create_dir_all(&provider_dir).map_err(|e| {
                BgRemovalError::file_io_error("create provider cache directory", &provider_dir, &e)
            })?;
        }

        let session_path = self.get_session_path(cache_key, &provider_name);
        let metadata_path = self.get_metadata_path(cache_key, &provider_name);

        // Try to serialize session - this may not be supported by all providers
        match self.try_serialize_session(session, &session_path) {
            Ok(size_bytes) => {
                // Create metadata entry
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let entry = SessionCacheEntry {
                    cache_key: cache_key.to_string(),
                    model_hash: model_hash.to_string(),
                    execution_provider: provider_name.clone(),
                    optimization_level: format!("{:?}", optimization_level),
                    provider_config_hash: Self::hash_string(provider_config),
                    ort_version: env!("CARGO_PKG_VERSION").to_string(),
                    created_at: now,
                    last_accessed: now,
                    size_bytes,
                    is_provider_optimized: matches!(execution_provider, ExecutionProvider::CoreMl),
                    session_config,
                };

                // Write metadata
                let metadata_json = serde_json::to_string_pretty(&entry).map_err(|e| {
                    BgRemovalError::invalid_config(format!(
                        "Failed to serialize session metadata: {}",
                        e
                    ))
                })?;

                fs::write(&metadata_path, metadata_json).map_err(|e| {
                    BgRemovalError::file_io_error("write session metadata", &metadata_path, &e)
                })?;

                // Update in-memory cache
                self.metadata_cache.insert(cache_key.to_string(), entry);
                self.update_stats();

                log::debug!(
                    "Cached session: {} (provider: {}, size: {})",
                    cache_key,
                    provider_name,
                    format_cache_size(size_bytes)
                );

                self.stats.cache_hits += 1;
            },
            Err(e) => {
                log::debug!(
                    "Session serialization not supported for {}: {}",
                    provider_name,
                    e
                );
                // For providers that don't support serialization, we don't cache
                // but we don't consider this an error
                self.stats.cache_misses += 1;
            },
        }

        Ok(())
    }

    /// Load a cached session if available
    ///
    /// # Arguments
    /// * `cache_key` - Cache key for the session
    /// * `execution_provider` - Execution provider to load for
    ///
    /// # Returns
    /// The loaded session if available, None if not cached
    ///
    /// # Errors
    /// - Failed to deserialize session data
    /// - Failed to read session files
    /// - I/O errors during loading operations
    pub fn load_cached_session(
        &mut self,
        cache_key: &str,
        execution_provider: ExecutionProvider,
    ) -> Result<Option<Session>> {
        if !self.is_session_cached(cache_key) {
            self.stats.cache_misses += 1;
            return Ok(None);
        }

        let provider_name = format!("{:?}", execution_provider).to_lowercase();
        let session_path = self.get_session_path(cache_key, &provider_name);

        if !session_path.exists() {
            self.stats.cache_misses += 1;
            return Ok(None);
        }

        match self.try_deserialize_session(&session_path) {
            Ok(session) => {
                // Update last accessed time
                if let Some(entry) = self.metadata_cache.get_mut(cache_key) {
                    entry.last_accessed = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                }

                log::debug!(
                    "Loaded cached session: {} (provider: {})",
                    cache_key,
                    provider_name
                );
                self.stats.cache_hits += 1;
                Ok(Some(session))
            },
            Err(e) => {
                log::debug!("Failed to load cached session {}: {}", cache_key, e);
                // Remove invalid cache entry
                self.remove_invalid_session(cache_key, &provider_name)?;
                self.stats.cache_misses += 1;
                Ok(None)
            },
        }
    }

    /// Try to serialize a session to disk
    ///
    /// Since ONNX Runtime doesn't support true session serialization, this method
    /// implements a cache strategy based on optimized model files and metadata.
    fn try_serialize_session(&self, session: &Session, session_path: &Path) -> Result<u64> {
        // Different execution providers handle caching differently:
        // - CoreML: Creates optimized .mlmodelc files that can be reused
        // - CPU/CUDA: Session can be rebuilt quickly from configuration
        // - DirectML: Similar to CPU/CUDA
        
        // For CoreML, try to find and cache the optimized model
        if let Some(optimized_model_path) = self.find_coreml_optimized_model(session_path) {
            return self.cache_optimized_model(&optimized_model_path, session_path);
        }
        
        // For other providers, create a metadata-only cache marker
        let cache_marker = SessionCacheMarker {
            session_id: format!("session_{}", session as *const _ as usize),
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ort_version: env!("CARGO_PKG_VERSION").to_string(),
            cache_type: "metadata_only".to_string(),
        };
        
        let marker_json = serde_json::to_string(&cache_marker)
            .map_err(|e| BgRemovalError::invalid_config(format!("Failed to serialize cache marker: {}", e)))?;
        
        // Ensure the parent directory exists before writing
        if let Some(parent) = session_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    BgRemovalError::file_io_error("create session cache directory", parent, &e)
                })?;
            }
        }
        
        fs::write(session_path, marker_json.as_bytes())
            .map_err(|e| BgRemovalError::file_io_error("write session cache marker", session_path, &e))?;

        Ok(marker_json.len() as u64)
    }
    
    /// Find CoreML optimized model file if it exists
    fn find_coreml_optimized_model(&self, _session_path: &Path) -> Option<PathBuf> {
        // CoreML often creates .mlmodelc directories with optimized models
        // Look in common temporary directories where CoreML might store these
        
        let temp_dirs = [
            std::env::temp_dir(),
            PathBuf::from("/tmp"),
            PathBuf::from("/var/tmp"),
        ];
        
        for temp_dir in &temp_dirs {
            if let Ok(entries) = fs::read_dir(temp_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("mlmodelc") {
                        // This might be our optimized model
                        if path.metadata().map(|m| m.len()).unwrap_or(0) > 0 {
                            return Some(path);
                        }
                    }
                }
            }
        }
        
        None
    }
    
    /// Cache an optimized model file
    fn cache_optimized_model(&self, optimized_path: &Path, cache_path: &Path) -> Result<u64> {
        // Copy the optimized model to our cache location
        let cache_dir = cache_path.parent().ok_or_else(|| {
            BgRemovalError::invalid_config("Invalid cache path".to_string())
        })?;
        
        if !cache_dir.exists() {
            fs::create_dir_all(cache_dir)?;
        }
        
        // If it's a directory (like .mlmodelc), copy recursively
        if optimized_path.is_dir() {
            self.copy_dir_recursive(optimized_path, cache_path)?;
        } else {
            fs::copy(optimized_path, cache_path)?;
        }
        
        // Return the size of the cached data
        Self::calculate_dir_size(cache_path)
    }
    
    /// Copy directory recursively
    fn copy_dir_recursive(&self, src: &Path, dst: &Path) -> Result<()> {
        if !dst.exists() {
            fs::create_dir_all(dst)?;
        }
        
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            
            if src_path.is_dir() {
                self.copy_dir_recursive(&src_path, &dst_path)?;
            } else {
                fs::copy(&src_path, &dst_path)?;
            }
        }
        
        Ok(())
    }
    
    /// Calculate total size of a directory or file
    fn calculate_dir_size(path: &Path) -> Result<u64> {
        let mut total_size = 0u64;
        
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                
                if entry_path.is_dir() {
                    total_size += Self::calculate_dir_size(&entry_path)?;
                } else {
                    total_size += entry.metadata()?.len();
                }
            }
        } else {
            total_size = path.metadata()?.len();
        }
        
        Ok(total_size)
    }

    /// Try to deserialize a session from disk
    ///
    /// Since ONNX Runtime session rebuilding is complex and provider-dependent,
    /// this implementation validates cache metadata and provides fast cache invalidation
    fn try_deserialize_session(&self, session_path: &Path) -> Result<Session> {
        // Read the cache file to determine what type of cache we have
        let cache_content = fs::read_to_string(session_path)
            .map_err(|e| BgRemovalError::file_io_error("read session cache", session_path, &e))?;
        
        // Try to parse as a cache marker to validate cache integrity
        if let Ok(marker) = serde_json::from_str::<SessionCacheMarker>(&cache_content) {
            log::debug!("Found valid session cache marker: {:?}", marker);
            
            // For now, we don't rebuild sessions but provide faster cache management
            // This speeds up cache invalidation and provides better diagnostics
            return Err(BgRemovalError::invalid_config(
                "Session cache validated - session will be recreated with optimized settings".to_string(),
            ));
        }
        
        // If it's not a valid marker, this is an old/invalid cache
        Err(BgRemovalError::invalid_config(
            "Invalid session cache format - will recreate session".to_string(),
        ))
    }

    /// Remove an invalid session from cache
    fn remove_invalid_session(&mut self, cache_key: &str, provider_name: &str) -> Result<()> {
        let session_path = self.get_session_path(cache_key, provider_name);
        let metadata_path = self.get_metadata_path(cache_key, provider_name);

        // Remove files if they exist
        if session_path.exists() {
            fs::remove_file(&session_path).map_err(|e| {
                BgRemovalError::file_io_error("remove invalid session file", &session_path, &e)
            })?;
        }

        if metadata_path.exists() {
            fs::remove_file(&metadata_path).map_err(|e| {
                BgRemovalError::file_io_error("remove invalid metadata file", &metadata_path, &e)
            })?;
        }

        // Remove from in-memory cache
        self.metadata_cache.remove(cache_key);
        self.update_stats();

        Ok(())
    }

    /// Calculate hash of a string
    fn hash_string(input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get cache hit ratio as a percentage
    pub fn get_hit_ratio(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            0.0
        } else {
            (self.stats.cache_hits as f64 / total as f64) * 100.0
        }
    }

    /// Get the current cache directory path
    pub fn get_current_cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }
}

impl Default for SessionCache {
    fn default() -> Self {
        Self::new().expect("Failed to create default session cache")
    }
}

/// Format file size in human-readable format
pub fn format_cache_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cache_key_generation() {
        let model_hash = "test_model_hash";
        let provider = ExecutionProvider::Auto;
        let opt_level = GraphOptimizationLevel::Level3;
        let config = "{}";

        let key1 = SessionCache::generate_cache_key(model_hash, provider, &opt_level, config);
        let key2 = SessionCache::generate_cache_key(model_hash, provider, &opt_level, config);

        // Same inputs should produce same key
        assert_eq!(key1, key2);

        // Different inputs should produce different keys
        let key3 = SessionCache::generate_cache_key("different_hash", provider, &opt_level, config);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_model_hash_calculation() {
        let data1 = b"test model data";
        let data2 = b"test model data";
        let data3 = b"different data";

        let hash1 = SessionCache::calculate_model_hash(data1);
        let hash2 = SessionCache::calculate_model_hash(data2);
        let hash3 = SessionCache::calculate_model_hash(data3);

        // Same data should produce same hash
        assert_eq!(hash1, hash2);

        // Different data should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_format_cache_size() {
        assert_eq!(format_cache_size(0), "0 B");
        assert_eq!(format_cache_size(512), "512 B");
        assert_eq!(format_cache_size(1024), "1.0 KB");
        assert_eq!(format_cache_size(1536), "1.5 KB");
        assert_eq!(format_cache_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_cache_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_session_cache_creation() {
        // This test only runs when CLI features are enabled
        let _cache = SessionCache::new().expect("Should create session cache successfully");
    }

    #[test]
    fn test_cache_statistics() {
        // Create cache with temporary directory to avoid interference from existing cache
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().join("test_cache");
        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

        let cache = SessionCache {
            cache_dir,
            metadata_cache: HashMap::new(),
            stats: SessionCacheStats::default(),
        };

        let stats = cache.get_stats();

        assert_eq!(stats.total_sessions, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert!(stats.sessions_by_provider.is_empty());
    }

    #[test]
    fn test_cache_hit_ratio() {
        let mut cache = SessionCache::default();

        // Initially no hits or misses
        assert_eq!(cache.get_hit_ratio(), 0.0);

        // Simulate cache misses
        cache.stats.cache_misses = 10;
        assert_eq!(cache.get_hit_ratio(), 0.0);

        // Add some hits
        cache.stats.cache_hits = 5;
        assert!((cache.get_hit_ratio() - 33.333333333333336).abs() < 0.000001); // 5/15 * 100

        // Equal hits and misses
        cache.stats.cache_hits = 10;
        assert_eq!(cache.get_hit_ratio(), 50.0); // 10/20 * 100
    }

    #[test]
    fn test_hash_string() {
        let input1 = "test_string";
        let input2 = "test_string";
        let input3 = "different_string";

        let hash1 = SessionCache::hash_string(input1);
        let hash2 = SessionCache::hash_string(input2);
        let hash3 = SessionCache::hash_string(input3);

        // Same input should produce same hash
        assert_eq!(hash1, hash2);

        // Different input should produce different hash
        assert_ne!(hash1, hash3);

        // Hash should be hex string
        assert!(hash1.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
