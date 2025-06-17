//! Color parsing and conversion utilities
//!
//! Consolidates color handling logic that was previously duplicated
//! across CLI and Web crates.

use crate::{config::BackgroundColor, error::Result, error::BgRemovalError};

/// Utility for parsing and converting colors
pub struct ColorParser;

impl ColorParser {
    /// Parse a hex color string to BackgroundColor
    ///
    /// Supports both #RRGGBB and #RGB formats.
    ///
    /// # Arguments
    /// * `hex` - Hex color string (with or without # prefix)
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::utils::ColorParser;
    /// 
    /// let white = ColorParser::parse_hex("#ffffff")?;
    /// let red = ColorParser::parse_hex("#f00")?;
    /// let blue = ColorParser::parse_hex("0000ff")?;
    /// ```
    pub fn parse_hex(hex: &str) -> Result<BackgroundColor> {
        let hex = hex.trim_start_matches('#');
        
        if hex.len() == 6 {
            // Parse #RRGGBB format
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| BgRemovalError::configuration("Invalid red component in hex color"))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| BgRemovalError::configuration("Invalid green component in hex color"))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| BgRemovalError::configuration("Invalid blue component in hex color"))?;
            
            Ok(BackgroundColor::new(r, g, b))
        } else if hex.len() == 3 {
            // Parse #RGB format (expand to #RRGGBB)
            let r = u8::from_str_radix(&hex[0..1], 16)
                .map_err(|_| BgRemovalError::configuration("Invalid red component in hex color"))? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16)
                .map_err(|_| BgRemovalError::configuration("Invalid green component in hex color"))? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16)
                .map_err(|_| BgRemovalError::configuration("Invalid blue component in hex color"))? * 17;
            
            Ok(BackgroundColor::new(r, g, b))
        } else {
            Err(BgRemovalError::configuration(
                "Color must be in #RRGGBB or #RGB format"
            ))
        }
    }
    
    /// Convert BackgroundColor to hex string
    ///
    /// # Arguments
    /// * `color` - BackgroundColor to convert
    /// * `include_hash` - Whether to include # prefix
    ///
    /// # Examples
    /// ```rust
    /// use bg_remove_core::{utils::ColorParser, BackgroundColor};
    /// 
    /// let color = BackgroundColor::new(255, 0, 128);
    /// let hex = ColorParser::to_hex(&color, true); // "#ff0080"
    /// let hex_no_hash = ColorParser::to_hex(&color, false); // "ff0080"
    /// ```
    pub fn to_hex(color: &BackgroundColor, include_hash: bool) -> String {
        if include_hash {
            format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b)
        } else {
            format!("{:02x}{:02x}{:02x}", color.r, color.g, color.b)
        }
    }
    
    /// Validate hex color format without parsing
    ///
    /// # Arguments
    /// * `hex` - Hex color string to validate
    ///
    /// # Returns
    /// true if the format is valid, false otherwise
    pub fn is_valid_hex(hex: &str) -> bool {
        let hex = hex.trim_start_matches('#');
        
        if hex.len() != 3 && hex.len() != 6 {
            return false;
        }
        
        hex.chars().all(|c| c.is_ascii_hexdigit())
    }
    
    /// Parse color with fallback to default
    ///
    /// # Arguments
    /// * `hex` - Hex color string to parse
    /// * `fallback` - Default color to use if parsing fails
    ///
    /// # Returns
    /// Parsed color or fallback if parsing fails
    pub fn parse_hex_or_default(hex: &str, fallback: BackgroundColor) -> BackgroundColor {
        Self::parse_hex(hex).unwrap_or(fallback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_6_digit() {
        let white = ColorParser::parse_hex("#ffffff").unwrap();
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);

        let black = ColorParser::parse_hex("#000000").unwrap();
        assert_eq!(black.r, 0);
        assert_eq!(black.g, 0);
        assert_eq!(black.b, 0);

        let red = ColorParser::parse_hex("#ff0000").unwrap();
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);
    }

    #[test]
    fn test_parse_hex_3_digit() {
        let white = ColorParser::parse_hex("#fff").unwrap();
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);

        let red = ColorParser::parse_hex("#f00").unwrap();
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);
    }

    #[test]
    fn test_parse_hex_without_hash() {
        let blue = ColorParser::parse_hex("0000ff").unwrap();
        assert_eq!(blue.r, 0);
        assert_eq!(blue.g, 0);
        assert_eq!(blue.b, 255);
    }

    #[test]
    fn test_parse_hex_invalid() {
        assert!(ColorParser::parse_hex("#fff").is_ok());
        assert!(ColorParser::parse_hex("#gggggg").is_err());
        assert!(ColorParser::parse_hex("#ff").is_err());
        assert!(ColorParser::parse_hex("#fffffff").is_err());
    }

    #[test]
    fn test_to_hex() {
        let color = BackgroundColor::new(255, 128, 0);
        assert_eq!(ColorParser::to_hex(&color, true), "#ff8000");
        assert_eq!(ColorParser::to_hex(&color, false), "ff8000");
    }

    #[test]
    fn test_is_valid_hex() {
        assert!(ColorParser::is_valid_hex("#ffffff"));
        assert!(ColorParser::is_valid_hex("#fff"));
        assert!(ColorParser::is_valid_hex("ffffff"));
        assert!(ColorParser::is_valid_hex("fff"));
        assert!(!ColorParser::is_valid_hex("#gggggg"));
        assert!(!ColorParser::is_valid_hex("#ff"));
        assert!(!ColorParser::is_valid_hex("#fffffff"));
    }

    #[test]
    fn test_parse_hex_or_default() {
        let fallback = BackgroundColor::white();
        
        let valid = ColorParser::parse_hex_or_default("#ff0000", fallback);
        assert_eq!(valid.r, 255);
        assert_eq!(valid.g, 0);
        assert_eq!(valid.b, 0);
        
        let invalid = ColorParser::parse_hex_or_default("#invalid", fallback);
        assert_eq!(invalid.r, 255);
        assert_eq!(invalid.g, 255);
        assert_eq!(invalid.b, 255);
    }
}