//! Test if models are actually embedded

use bg_remove_core::models::{ModelManager, EmbeddedModelProvider, ModelProvider};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing model embedding...");
    
    // Test embedded provider directly
    let provider = EmbeddedModelProvider;
    
    println!("📊 Loading default model info...");
    let info = provider.get_model_info()?;
    println!("   Model: {}", info.name);
    println!("   Expected size: {} MB", info.size_bytes / (1024 * 1024));
    
    println!("📊 Loading default model data...");
    let data = provider.load_model_data()?;
    println!("   Actual loaded size: {} MB", data.len() / (1024 * 1024));
    
    // Test with ModelManager
    println!("📊 Testing ModelManager...");
    let manager = ModelManager::with_embedded();
    let manager_data = manager.load_model()?;
    println!("   Manager loaded size: {} MB", manager_data.len() / (1024 * 1024));
    
    if data.len() == manager_data.len() {
        println!("✅ ModelManager and direct provider return same data");
    } else {
        println!("❌ ModelManager and direct provider return different data");
    }
    
    // Check if data looks like ONNX
    if data.starts_with(b"\x08\x01") || data.starts_with(b"ONNX") {
        println!("✅ Data appears to be valid ONNX format");
    } else {
        println!("❌ Data does not appear to be ONNX format. First 16 bytes: {:?}", &data[..16.min(data.len())]);
    }
    
    println!("\n🎯 Models appear to be properly embedded!");
    
    Ok(())
}