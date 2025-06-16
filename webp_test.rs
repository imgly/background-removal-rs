use std::fs;

fn main() {
    // Create a simple test image
    let width = 100u32;
    let height = 100u32;
    
    // RGBA version
    let rgba_data: Vec<u8> = (0..width*height*4).map(|i| (i % 256) as u8).collect();
    let encoder = webp::Encoder::from_rgba(&rgba_data, width, height);
    let rgba_webp = encoder.encode(80.0);
    fs::write("test_rgba.webp", &*rgba_webp).unwrap();
    
    // RGB version  
    let rgb_data: Vec<u8> = (0..width*height*3).map(|i| (i % 256) as u8).collect();
    let encoder = webp::Encoder::from_rgb(&rgb_data, width, height);
    let rgb_webp = encoder.encode(80.0);
    fs::write("test_rgb.webp", &*rgb_webp).unwrap();
    
    println!("RGBA WebP size: {}", rgba_webp.len());
    println!("RGB WebP size: {}", rgb_webp.len());
}