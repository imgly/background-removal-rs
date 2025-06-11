use ort::execution_providers::{CoreMLExecutionProvider, CUDAExecutionProvider, ExecutionProvider as OrtExecutionProvider};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Checking ONNX Runtime Execution Provider Availability");
    println!("========================================================");
    
    // Check CPU (always available)
    println!("CPU: ✅ Always available");
    
    // Check CUDA availability
    let cuda_provider = CUDAExecutionProvider::default();
    let cuda_available = OrtExecutionProvider::is_available(&cuda_provider).unwrap_or(false);
    println!("CUDA: {} {}", 
        if cuda_available { "✅" } else { "❌" },
        if cuda_available { "Available" } else { "Not available" }
    );
    
    // Check CoreML availability  
    let coreml_provider = CoreMLExecutionProvider::default();
    let coreml_available = OrtExecutionProvider::is_available(&coreml_provider).unwrap_or(false);
    println!("CoreML: {} {}", 
        if coreml_available { "✅" } else { "❌" },
        if coreml_available { "Available" } else { "Not available" }
    );
    
    println!("\n🔧 System Information:");
    println!("OS: {}", std::env::consts::OS);
    println!("Architecture: {}", std::env::consts::ARCH);
    
    if std::env::consts::OS == "macos" {
        println!("\n💡 On macOS, CoreML should be available if:");
        println!("   • You're running on Apple Silicon (M1/M2/M3) or Intel with macOS 10.13+");
        println!("   • ONNX Runtime was compiled with CoreML support");
        println!("   • The 'coreml' feature flag is enabled");
    }
    
    Ok(())
}