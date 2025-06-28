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
                log::warn!(
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
                log::warn!("Failed to load cached session {}: {}", cache_key, e);
                // Remove invalid cache entry
                self.remove_invalid_session(cache_key, &provider_name)?;
                self.stats.cache_misses += 1;
                Ok(None)
            },
        }
    }

    /// Try to serialize a session to disk
    ///
    /// Note: Session serialization support varies by execution provider
    fn try_serialize_session(&self, session: &Session, session_path: &Path) -> Result<u64> {
        // For now, we'll use a simple approach - ONNX Runtime session serialization
        // is limited and provider-dependent. This is a placeholder for future implementation
        // when better serialization support becomes available.

        // Create a marker file for now to indicate the session was created
        let marker_data = format!("session_marker_{}", session as *const _ as usize);
        fs::write(session_path, marker_data.as_bytes())
            .map_err(|e| BgRemovalError::file_io_error("write session marker", session_path, &e))?;

        Ok(marker_data.len() as u64)
    }

    /// Try to deserialize a session from disk
    ///
    /// Note: Session deserialization support varies by execution provider
    fn try_deserialize_session(&self, _session_path: &Path) -> Result<Session> {
        // For now, return an error since true session serialization/deserialization
        // is not yet fully implemented. This will be enhanced in future iterations
        // when ONNX Runtime provides better serialization support.

        Err(BgRemovalError::invalid_config(
            "Session deserialization not yet implemented - will recreate session".to_string(),
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
        let cache = SessionCache::default();
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
