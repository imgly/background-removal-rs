//! Integration tests for ICC color profile preservation

#[cfg(test)]
mod tests {
    use crate::{
        color_profile::ProfileExtractor,
        config::{ColorManagementConfig, RemovalConfig},
        types::{ColorProfile, ColorSpace},
    };

    #[test]
    fn test_color_profile_creation() {
        let profile = ColorProfile::new(None, ColorSpace::Srgb);
        assert_eq!(profile.color_space, ColorSpace::Srgb);
        assert_eq!(profile.data_size(), 0);
        assert!(!profile.has_color_profile());
    }

    #[test]
    fn test_color_profile_from_icc_data() {
        let icc_data = b"fake sRGB profile data".to_vec();
        let profile = ColorProfile::from_icc_data(icc_data.clone());
        assert_eq!(profile.data_size(), icc_data.len());
        assert!(profile.has_color_profile());
        // Should detect as sRGB due to "sRGB" in the fake data
        assert_eq!(profile.color_space, ColorSpace::Srgb);
    }

    #[test]
    fn test_color_space_detection() {
        // Test sRGB detection
        let srgb_data = b"this is an sRGB color profile".to_vec();
        let profile = ColorProfile::from_icc_data(srgb_data);
        assert_eq!(profile.color_space, ColorSpace::Srgb);

        // Test Adobe RGB detection
        let adobe_data = b"Adobe RGB color profile".to_vec();
        let profile = ColorProfile::from_icc_data(adobe_data);
        assert_eq!(profile.color_space, ColorSpace::AdobeRgb);

        // Test Display P3 detection
        let p3_data = b"Display P3 profile".to_vec();
        let profile = ColorProfile::from_icc_data(p3_data);
        assert_eq!(profile.color_space, ColorSpace::DisplayP3);

        // Test unknown profile
        let unknown_data = b"some random profile data".to_vec();
        let profile = ColorProfile::from_icc_data(unknown_data);
        assert_eq!(profile.color_space, ColorSpace::Unknown("ICC Present".to_string()));
    }

    #[test]
    fn test_color_space_display() {
        assert_eq!(ColorSpace::Srgb.to_string(), "sRGB");
        assert_eq!(ColorSpace::AdobeRgb.to_string(), "Adobe RGB");
        assert_eq!(ColorSpace::DisplayP3.to_string(), "Display P3");
        assert_eq!(ColorSpace::ProPhotoRgb.to_string(), "ProPhoto RGB");
        assert_eq!(
            ColorSpace::Unknown("Test".to_string()).to_string(),
            "Unknown (Test)"
        );
    }

    #[test]
    fn test_color_management_config_presets() {
        let preserve_config = ColorManagementConfig::preserve();
        assert!(preserve_config.preserve_color_profile);
        assert!(!preserve_config.force_srgb_output);
        assert!(preserve_config.fallback_to_srgb);
        assert!(preserve_config.embed_profile_in_output);

        let ignore_config = ColorManagementConfig::ignore();
        assert!(!ignore_config.preserve_color_profile);
        assert!(!ignore_config.force_srgb_output);
        assert!(ignore_config.fallback_to_srgb);
        assert!(!ignore_config.embed_profile_in_output);

        let force_srgb_config = ColorManagementConfig::force_srgb();
        assert!(force_srgb_config.preserve_color_profile);
        assert!(force_srgb_config.force_srgb_output);
        assert!(force_srgb_config.fallback_to_srgb);
        assert!(force_srgb_config.embed_profile_in_output);
    }

    #[test]
    fn test_removal_config_with_color_management() {
        let config = RemovalConfig::builder()
            .color_management(ColorManagementConfig::preserve())
            .preserve_color_profile(true)
            .force_srgb_output(false)
            .embed_profile_in_output(true)
            .build()
            .unwrap();

        assert!(config.color_management.preserve_color_profile);
        assert!(!config.color_management.force_srgb_output);
        assert!(config.color_management.embed_profile_in_output);
    }

    #[test]
    fn test_profile_extractor_unsupported_format() {
        // Test with a non-existent file with unsupported extension
        let result = ProfileExtractor::extract_from_image("test.bmp");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_profile_extractor_nonexistent_file() {
        // Test with non-existent JPEG file
        let result = ProfileExtractor::extract_from_image("nonexistent.jpg");
        assert!(result.is_err());
    }

    // Integration test demonstrating the full workflow
    #[test]
    fn test_color_profile_workflow() {
        // Create a mock color profile
        let icc_data = b"sRGB IEC61966-2.1".to_vec();
        let original_profile = ColorProfile::from_icc_data(icc_data);

        // Verify profile properties
        assert_eq!(original_profile.color_space, ColorSpace::Srgb);
        assert!(original_profile.has_color_profile());
        assert!(original_profile.data_size() > 0);

        // Test color management configuration
        let config = ColorManagementConfig::preserve();
        assert!(config.preserve_color_profile);
        assert!(config.embed_profile_in_output);

        // Verify the profile can be accessed
        if let Some(profile_ref) = Some(&original_profile) {
            assert_eq!(profile_ref.color_space, ColorSpace::Srgb);
            assert!(profile_ref.data_size() > 0);
        }
    }
}