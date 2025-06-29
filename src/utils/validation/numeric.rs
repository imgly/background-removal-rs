//! Numeric validation utilities
//!
//! Provides safe numeric conversions and range validation to prevent
//! overflow, underflow, and other numeric errors.

use crate::error::{BgRemovalError, Result};

/// Validator for numeric operations and conversions
pub struct NumericValidator;

impl NumericValidator {
    /// Safely convert f32 to u32 with bounds checking
    pub fn validate_f32_to_u32(value: f32) -> Result<u32> {
        if !value.is_finite() {
            return Err(BgRemovalError::processing(format!(
                "Cannot convert non-finite value {} to u32",
                value
            )));
        }

        if value < 0.0 {
            return Err(BgRemovalError::processing(format!(
                "Cannot convert negative value {} to u32",
                value
            )));
        }

        if value > u32::MAX as f32 {
            return Err(BgRemovalError::processing(format!(
                "Value {} exceeds u32::MAX ({})",
                value,
                u32::MAX
            )));
        }

        Ok(value as u32)
    }

    /// Safely convert u64 to usize with bounds checking
    pub fn validate_u64_to_usize(value: u64) -> Result<usize> {
        if value > usize::MAX as u64 {
            return Err(BgRemovalError::processing(format!(
                "Value {} exceeds usize::MAX on this platform ({})",
                value,
                usize::MAX
            )));
        }
        Ok(value as usize)
    }

    /// Validate percentage value (0.0 to 1.0)
    pub fn validate_percentage(value: f32) -> Result<f32> {
        if !value.is_finite() {
            return Err(BgRemovalError::invalid_config(format!(
                "Percentage value must be finite, got {}",
                value
            )));
        }

        if !(0.0..=1.0).contains(&value) {
            return Err(BgRemovalError::invalid_config(format!(
                "Percentage value must be between 0.0 and 1.0, got {}",
                value
            )));
        }

        Ok(value)
    }

    /// Validate quality setting (0-100)
    pub fn validate_quality(value: u8) -> Result<u8> {
        if value > 100 {
            return Err(BgRemovalError::invalid_config(format!(
                "Quality must be between 0 and 100, got {}",
                value
            )));
        }
        Ok(value)
    }

    /// Validate thread count
    pub fn validate_thread_count(value: usize) -> Result<usize> {
        const MAX_THREADS: usize = 256; // Reasonable upper limit

        if value > MAX_THREADS {
            return Err(BgRemovalError::invalid_config(format!(
                "Thread count {} exceeds maximum allowed ({})",
                value, MAX_THREADS
            )));
        }

        Ok(value)
    }

    /// Validate numeric range (inclusive)
    pub fn validate_range<T>(value: T, min: T, max: T, name: &str) -> Result<T>
    where
        T: PartialOrd + std::fmt::Display + Copy,
    {
        if value < min || value > max {
            return Err(BgRemovalError::invalid_config(format!(
                "{} must be between {} and {}, got {}",
                name, min, max, value
            )));
        }
        Ok(value)
    }

    /// Validate that a value is positive
    pub fn validate_positive<T>(value: T, name: &str) -> Result<T>
    where
        T: PartialOrd + std::fmt::Display + Copy + Default,
    {
        if value <= T::default() {
            return Err(BgRemovalError::invalid_config(format!(
                "{} must be positive, got {}",
                name, value
            )));
        }
        Ok(value)
    }

    /// Validate and clamp a value to a range
    pub fn clamp_to_range<T>(value: T, min: T, max: T) -> T
    where
        T: PartialOrd + Copy,
    {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }

    /// Safely multiply two u32 values checking for overflow
    pub fn safe_multiply_u32(a: u32, b: u32) -> Result<u32> {
        a.checked_mul(b).ok_or_else(|| {
            BgRemovalError::processing(format!("Multiplication overflow: {} * {}", a, b))
        })
    }

    /// Safely add two u32 values checking for overflow
    pub fn safe_add_u32(a: u32, b: u32) -> Result<u32> {
        a.checked_add(b)
            .ok_or_else(|| BgRemovalError::processing(format!("Addition overflow: {} + {}", a, b)))
    }

    /// Validate normalization parameters (mean and std arrays)
    pub fn validate_normalization_params(
        mean: &[f64],
        std: &[f64],
        expected_channels: usize,
    ) -> Result<()> {
        if mean.len() != expected_channels {
            return Err(BgRemovalError::invalid_config(format!(
                "Mean array length {} doesn't match expected channels {}",
                mean.len(),
                expected_channels
            )));
        }

        if std.len() != expected_channels {
            return Err(BgRemovalError::invalid_config(format!(
                "Std array length {} doesn't match expected channels {}",
                std.len(),
                expected_channels
            )));
        }

        // Validate mean values are reasonable
        for (i, &value) in mean.iter().enumerate() {
            if !value.is_finite() {
                return Err(BgRemovalError::invalid_config(format!(
                    "Mean value at index {} is not finite: {}",
                    i, value
                )));
            }
        }

        // Validate std values are positive and reasonable
        for (i, &value) in std.iter().enumerate() {
            if !value.is_finite() {
                return Err(BgRemovalError::invalid_config(format!(
                    "Std value at index {} is not finite: {}",
                    i, value
                )));
            }
            if value <= 0.0 {
                return Err(BgRemovalError::invalid_config(format!(
                    "Std value at index {} must be positive: {}",
                    i, value
                )));
            }
            if value > 10.0 {
                log::warn!(
                    "Unusually large std value at index {}: {} (typical range: 0.1-2.0)",
                    i,
                    value
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_f32_to_u32() {
        // Valid conversions
        assert_eq!(NumericValidator::validate_f32_to_u32(0.0).unwrap(), 0);
        assert_eq!(NumericValidator::validate_f32_to_u32(100.5).unwrap(), 100);
        assert_eq!(
            NumericValidator::validate_f32_to_u32(u32::MAX as f32).unwrap(),
            u32::MAX
        );

        // Invalid conversions
        assert!(NumericValidator::validate_f32_to_u32(-1.0).is_err());
        assert!(NumericValidator::validate_f32_to_u32(f32::NAN).is_err());
        assert!(NumericValidator::validate_f32_to_u32(f32::INFINITY).is_err());
    }

    #[test]
    fn test_validate_percentage() {
        // Valid percentages
        assert!(NumericValidator::validate_percentage(0.0).is_ok());
        assert!(NumericValidator::validate_percentage(0.5).is_ok());
        assert!(NumericValidator::validate_percentage(1.0).is_ok());

        // Invalid percentages
        assert!(NumericValidator::validate_percentage(-0.1).is_err());
        assert!(NumericValidator::validate_percentage(1.1).is_err());
        assert!(NumericValidator::validate_percentage(f32::NAN).is_err());
    }

    #[test]
    fn test_validate_quality() {
        // Valid quality values
        assert!(NumericValidator::validate_quality(0).is_ok());
        assert!(NumericValidator::validate_quality(50).is_ok());
        assert!(NumericValidator::validate_quality(100).is_ok());

        // Invalid quality values
        assert!(NumericValidator::validate_quality(101).is_err());
    }

    #[test]
    fn test_validate_range() {
        // Valid range
        assert!(NumericValidator::validate_range(50, 0, 100, "test").is_ok());
        assert!(NumericValidator::validate_range(0, 0, 100, "test").is_ok());
        assert!(NumericValidator::validate_range(100, 0, 100, "test").is_ok());

        // Out of range
        assert!(NumericValidator::validate_range(-1, 0, 100, "test").is_err());
        assert!(NumericValidator::validate_range(101, 0, 100, "test").is_err());
    }

    #[test]
    fn test_validate_positive() {
        // Valid positive values
        assert!(NumericValidator::validate_positive(1, "test").is_ok());
        assert!(NumericValidator::validate_positive(100, "test").is_ok());

        // Invalid values
        assert!(NumericValidator::validate_positive(0, "test").is_err());
        assert!(NumericValidator::validate_positive(-1, "test").is_err());
    }

    #[test]
    fn test_clamp_to_range() {
        assert_eq!(NumericValidator::clamp_to_range(50, 0, 100), 50);
        assert_eq!(NumericValidator::clamp_to_range(-10, 0, 100), 0);
        assert_eq!(NumericValidator::clamp_to_range(150, 0, 100), 100);
    }

    #[test]
    fn test_safe_arithmetic() {
        // Valid operations
        assert_eq!(
            NumericValidator::safe_multiply_u32(100, 200).unwrap(),
            20000
        );
        assert_eq!(NumericValidator::safe_add_u32(100, 200).unwrap(), 300);

        // Overflow
        assert!(NumericValidator::safe_multiply_u32(u32::MAX, 2).is_err());
        assert!(NumericValidator::safe_add_u32(u32::MAX, 1).is_err());
    }

    #[test]
    fn test_validate_normalization_params() {
        // Valid parameters
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        assert!(NumericValidator::validate_normalization_params(&mean, &std, 3).is_ok());

        // Wrong length
        assert!(NumericValidator::validate_normalization_params(&mean, &std, 1).is_err());

        // Invalid std values
        let invalid_std = [0.0, 0.224, 0.225];
        assert!(NumericValidator::validate_normalization_params(&mean, &invalid_std, 3).is_err());

        // Non-finite values
        let nan_mean = [f64::NAN, 0.456, 0.406];
        assert!(NumericValidator::validate_normalization_params(&nan_mean, &std, 3).is_err());
    }
}
