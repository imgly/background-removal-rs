//! Execution provider parsing and management utilities
//!
//! Consolidates execution provider logic that was previously in the CLI crate.

use crate::{
    config::ExecutionProvider,
    error::{BgRemovalError, Result},
    processor::BackendType,
};

/// Information about an execution provider
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: String,
    pub backend_type: BackendType,
    pub execution_provider: ExecutionProvider,
    pub available: bool,
    pub description: String,
}

/// Utility for parsing and managing execution providers
pub struct ExecutionProviderManager;

impl ExecutionProviderManager {
    /// Parse execution provider string in format "backend:provider"
    ///
    /// # Arguments
    /// * `provider_str` - String in format "backend:provider" (e.g., "onnx:auto", "tract:cpu")
    ///
    /// # Returns
    /// Tuple of (BackendType, ExecutionProvider)
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::utils::ExecutionProviderManager;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let (backend, provider) = ExecutionProviderManager::parse_provider_string("onnx:auto")?;
    /// let (backend, provider) = ExecutionProviderManager::parse_provider_string("tract:cpu")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn parse_provider_string(provider_str: &str) -> Result<(BackendType, ExecutionProvider)> {
        if let Some((backend, provider)) = provider_str.split_once(':') {
            match backend {
                "onnx" => {
                    let execution_provider = match provider {
                        "auto" => ExecutionProvider::Auto,
                        "cpu" => ExecutionProvider::Cpu,
                        "cuda" => ExecutionProvider::Cuda,
                        "coreml" => ExecutionProvider::CoreMl,
                        _ => {
                            return Err(BgRemovalError::invalid_config(&format!(
                                "Unknown ONNX provider: {}. Supported: auto, cpu, cuda, coreml",
                                provider
                            )));
                        },
                    };
                    Ok((BackendType::Onnx, execution_provider))
                },
                "tract" => {
                    let execution_provider = match provider {
                        "cpu" => ExecutionProvider::Cpu, // Tract only supports CPU
                        _ => {
                            return Err(BgRemovalError::invalid_config(&format!(
                                "Unknown Tract provider: {}. Tract only supports 'cpu'",
                                provider
                            )));
                        },
                    };
                    Ok((BackendType::Tract, execution_provider))
                },
                _ => Err(BgRemovalError::invalid_config(&format!(
                    "Unknown backend: {}. Supported backends: onnx, tract, mock",
                    backend
                ))),
            }
        } else {
            // If no colon, assume it's just a backend name and use defaults
            match provider_str {
                "onnx" => Ok((BackendType::Onnx, ExecutionProvider::Auto)),
                "tract" => Ok((BackendType::Tract, ExecutionProvider::Cpu)),
                _ => Err(BgRemovalError::invalid_config(
                    "Invalid provider format. Use backend:provider (e.g., onnx:auto, tract:cpu)",
                )),
            }
        }
    }

    /// Get a list of all theoretical provider combinations
    ///
    /// Note: This returns all possible combinations, not necessarily available ones.
    /// Use backend-specific listing functions to check actual availability.
    pub fn list_all_providers() -> Vec<ProviderInfo> {
        vec![
            // ONNX providers
            ProviderInfo {
                name: "onnx:auto".to_string(),
                backend_type: BackendType::Onnx,
                execution_provider: ExecutionProvider::Auto,
                available: false, // Unknown without checking
                description: "ONNX Runtime with auto-selected provider".to_string(),
            },
            ProviderInfo {
                name: "onnx:cpu".to_string(),
                backend_type: BackendType::Onnx,
                execution_provider: ExecutionProvider::Cpu,
                available: false, // Unknown without checking
                description: "ONNX Runtime CPU execution".to_string(),
            },
            ProviderInfo {
                name: "onnx:cuda".to_string(),
                backend_type: BackendType::Onnx,
                execution_provider: ExecutionProvider::Cuda,
                available: false, // Unknown without checking
                description: "ONNX Runtime CUDA GPU acceleration".to_string(),
            },
            ProviderInfo {
                name: "onnx:coreml".to_string(),
                backend_type: BackendType::Onnx,
                execution_provider: ExecutionProvider::CoreMl,
                available: false, // Unknown without checking
                description: "ONNX Runtime CoreML (Apple Silicon) acceleration".to_string(),
            },
            // Tract providers
            ProviderInfo {
                name: "tract:cpu".to_string(),
                backend_type: BackendType::Tract,
                execution_provider: ExecutionProvider::Cpu,
                available: true, // Tract CPU is always available
                description: "Pure Rust CPU inference via Tract".to_string(),
            },
        ]
    }

    /// Validate a provider string without parsing
    pub fn is_valid_provider_string(provider_str: &str) -> bool {
        Self::parse_provider_string(provider_str).is_ok()
    }

    /// Get the default provider for a given backend type
    pub fn default_provider_for_backend(backend_type: &BackendType) -> ExecutionProvider {
        match backend_type {
            BackendType::Onnx => ExecutionProvider::Auto,
            BackendType::Tract => ExecutionProvider::Cpu,
        }
    }

    /// Convert backend type and execution provider back to string
    pub fn provider_to_string(backend_type: &BackendType, provider: &ExecutionProvider) -> String {
        let backend_str = match backend_type {
            BackendType::Onnx => "onnx",
            BackendType::Tract => "tract",
        };

        let provider_str = match provider {
            ExecutionProvider::Auto => "auto",
            ExecutionProvider::Cpu => "cpu",
            ExecutionProvider::Cuda => "cuda",
            ExecutionProvider::CoreMl => "coreml",
        };

        format!("{}:{}", backend_str, provider_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_onnx_providers() {
        let (backend, provider) =
            ExecutionProviderManager::parse_provider_string("onnx:auto").unwrap();
        assert_eq!(backend, BackendType::Onnx);
        assert_eq!(provider, ExecutionProvider::Auto);

        let (backend, provider) =
            ExecutionProviderManager::parse_provider_string("onnx:cpu").unwrap();
        assert_eq!(backend, BackendType::Onnx);
        assert_eq!(provider, ExecutionProvider::Cpu);

        let (backend, provider) =
            ExecutionProviderManager::parse_provider_string("onnx:cuda").unwrap();
        assert_eq!(backend, BackendType::Onnx);
        assert_eq!(provider, ExecutionProvider::Cuda);

        let (backend, provider) =
            ExecutionProviderManager::parse_provider_string("onnx:coreml").unwrap();
        assert_eq!(backend, BackendType::Onnx);
        assert_eq!(provider, ExecutionProvider::CoreMl);
    }

    #[test]
    fn test_parse_tract_providers() {
        let (backend, provider) =
            ExecutionProviderManager::parse_provider_string("tract:cpu").unwrap();
        assert_eq!(backend, BackendType::Tract);
        assert_eq!(provider, ExecutionProvider::Cpu);

        // Tract doesn't support other providers
        assert!(ExecutionProviderManager::parse_provider_string("tract:cuda").is_err());
        assert!(ExecutionProviderManager::parse_provider_string("tract:auto").is_err());
    }

    #[test]
    fn test_parse_backend_only() {
        let (backend, provider) = ExecutionProviderManager::parse_provider_string("onnx").unwrap();
        assert_eq!(backend, BackendType::Onnx);
        assert_eq!(provider, ExecutionProvider::Auto);

        let (backend, provider) = ExecutionProviderManager::parse_provider_string("tract").unwrap();
        assert_eq!(backend, BackendType::Tract);
        assert_eq!(provider, ExecutionProvider::Cpu);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(ExecutionProviderManager::parse_provider_string("invalid").is_err());
        assert!(ExecutionProviderManager::parse_provider_string("onnx:invalid").is_err());
        assert!(ExecutionProviderManager::parse_provider_string("invalid:auto").is_err());
    }

    #[test]
    fn test_is_valid_provider_string() {
        assert!(ExecutionProviderManager::is_valid_provider_string(
            "onnx:auto"
        ));
        assert!(ExecutionProviderManager::is_valid_provider_string(
            "tract:cpu"
        ));
        assert!(!ExecutionProviderManager::is_valid_provider_string(
            "invalid"
        ));
        assert!(!ExecutionProviderManager::is_valid_provider_string(
            "onnx:invalid"
        ));
    }

    #[test]
    fn test_provider_to_string() {
        assert_eq!(
            ExecutionProviderManager::provider_to_string(
                &BackendType::Onnx,
                &ExecutionProvider::Auto
            ),
            "onnx:auto"
        );
        assert_eq!(
            ExecutionProviderManager::provider_to_string(
                &BackendType::Tract,
                &ExecutionProvider::Cpu
            ),
            "tract:cpu"
        );
    }

    #[test]
    fn test_list_all_providers() {
        let providers = ExecutionProviderManager::list_all_providers();
        assert!(!providers.is_empty());

        // Check that we have expected providers
        let names: Vec<&String> = providers.iter().map(|p| &p.name).collect();
        assert!(names.contains(&&"onnx:auto".to_string()));
        assert!(names.contains(&&"tract:cpu".to_string()));
        assert!(names.contains(&&"mock:cpu".to_string()));
    }
}
