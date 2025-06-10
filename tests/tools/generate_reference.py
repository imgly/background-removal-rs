#!/usr/bin/env python3
"""
Reference data generation script for background removal testing.

This script generates reference outputs using the existing JavaScript implementation
to establish ground truth for the Rust implementation validation.
"""

import subprocess
import json
import os
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReferenceGenerator:
    """Generates reference test data for validation."""
    
    def __init__(self, test_assets_dir: Path, js_runner_path: Optional[Path] = None):
        """
        Initialize the reference generator.
        
        Args:
            test_assets_dir: Path to test assets directory
            js_runner_path: Path to JavaScript reference runner (optional)
        """
        self.assets_dir = test_assets_dir
        self.input_dir = test_assets_dir / "input"
        self.expected_dir = test_assets_dir / "expected"
        self.metadata_dir = test_assets_dir / "metadata"
        
        # JavaScript runner path (would need to be implemented separately)
        self.js_runner = js_runner_path or Path("js_reference_runner.js")
        
        # Load test specifications
        self.image_specs = self._load_json(self.metadata_dir / "image_specs.json")
        self.test_cases = self._load_json(self.metadata_dir / "test_cases.json")
        
    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Required file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise
    
    def generate_test_images(self) -> bool:
        """
        Generate synthetic test images based on specifications.
        
        This is a placeholder implementation. In a real scenario, you would either:
        1. Use a dataset of real images
        2. Generate synthetic images using tools like PIL/OpenCV
        3. Download images from a curated dataset
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Generating synthetic test images...")
        
        try:
            # Generate portrait images
            self._generate_portrait_images()
            
            # Generate product images  
            self._generate_product_images()
            
            # Generate complex scene images
            self._generate_complex_images()
            
            # Generate edge case images
            self._generate_edge_case_images()
            
            # Generate raw format samples
            self._generate_raw_format_samples()
            
            logger.info("Test image generation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate test images: {e}")
            return False
    
    def _generate_portrait_images(self):
        """Generate portrait test images."""
        from PIL import Image, ImageDraw
        import numpy as np
        
        portraits_dir = self.input_dir / "portraits"
        portraits_dir.mkdir(exist_ok=True)
        
        for spec in self.image_specs["portraits"]:
            width, height = spec["resolution"]
            filename = spec["filename"]
            
            # Create a simple synthetic portrait
            img = Image.new('RGB', (width, height), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Simple "person" shape (oval for head, rectangle for body)
            head_x = width // 2
            head_y = height // 3
            head_radius = min(width, height) // 8
            
            # Draw head (oval)
            draw.ellipse([
                head_x - head_radius, head_y - head_radius,
                head_x + head_radius, head_y + head_radius
            ], fill='peachpuff', outline='black')
            
            # Draw body (rectangle)
            body_width = head_radius
            body_height = height // 3
            draw.rectangle([
                head_x - body_width//2, head_y + head_radius,
                head_x + body_width//2, head_y + head_radius + body_height
            ], fill='darkblue', outline='black')
            
            # Add complexity based on specification
            if spec.get("background_complexity") == "complex":
                # Add random noise/patterns to background
                self._add_complex_background(draw, width, height)
            
            # Save image
            output_path = portraits_dir / filename
            img.save(output_path, "JPEG", quality=85)
            logger.info(f"Generated portrait: {filename}")
    
    def _generate_product_images(self):
        """Generate product test images."""
        from PIL import Image, ImageDraw
        
        products_dir = self.input_dir / "products"
        products_dir.mkdir(exist_ok=True)
        
        for spec in self.image_specs["products"]:
            width, height = spec["resolution"]
            filename = spec["filename"]
            
            # Create product image based on type
            if spec["background_type"] == "white":
                bg_color = 'white'
            elif spec["background_type"] == "gradient":
                bg_color = 'lightgray'
            else:
                bg_color = 'beige'
            
            img = Image.new('RGB', (width, height), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            # Simple product shape (rectangle for most products)
            product_w = width // 3
            product_h = height // 3
            product_x = (width - product_w) // 2
            product_y = (height - product_h) // 2
            
            if spec["product_type"] == "clothing":
                # Draw t-shirt like shape
                draw.rectangle([
                    product_x, product_y,
                    product_x + product_w, product_y + product_h
                ], fill='darkred', outline='black')
            elif spec["product_type"] == "electronics":
                # Draw phone/device shape
                draw.rectangle([
                    product_x, product_y,
                    product_x + product_w//2, product_y + product_h
                ], fill='black', outline='gray')
            else:
                # Generic product
                draw.rectangle([
                    product_x, product_y,
                    product_x + product_w, product_y + product_h
                ], fill='brown', outline='black')
            
            # Save image
            output_path = products_dir / filename
            img.save(output_path, "JPEG", quality=90)
            logger.info(f"Generated product: {filename}")
    
    def _generate_complex_images(self):
        """Generate complex scene test images."""
        from PIL import Image, ImageDraw
        
        complex_dir = self.input_dir / "complex"
        complex_dir.mkdir(exist_ok=True)
        
        for spec in self.image_specs["complex"]:
            width, height = spec["resolution"]
            filename = spec["filename"]
            
            img = Image.new('RGB', (width, height), color='lightsteelblue')
            draw = ImageDraw.Draw(img)
            
            # Add multiple overlapping elements
            for i in range(3):
                x = width // 4 + i * (width // 6)
                y = height // 4 + i * (height // 8)
                w = width // 6
                h = height // 6
                
                colors = ['red', 'green', 'blue']
                draw.ellipse([x, y, x + w, y + h], fill=colors[i], outline='black')
            
            # Save image
            output_path = complex_dir / filename
            img.save(output_path, "JPEG", quality=85)
            logger.info(f"Generated complex scene: {filename}")
    
    def _generate_edge_case_images(self):
        """Generate edge case test images."""
        from PIL import Image, ImageDraw
        import numpy as np
        
        edge_cases_dir = self.input_dir / "edge_cases"
        edge_cases_dir.mkdir(exist_ok=True)
        
        for spec in self.image_specs["edge_cases"]:
            width, height = spec["resolution"]
            filename = spec["filename"]
            
            if spec["challenge_type"] == "resolution":
                # Very small or very large images
                img = Image.new('RGB', (width, height), color='blue')
                draw = ImageDraw.Draw(img)
                # Small subject
                draw.ellipse([width//4, height//4, 3*width//4, 3*height//4], fill='red')
                
            elif spec["challenge_type"] == "quality":
                # High noise image
                img = Image.new('RGB', (width, height), color='gray')
                # Add noise
                noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
                img_array = np.array(img)
                noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(noisy_array)
                
            elif spec["challenge_type"] == "color":
                # Monochrome image
                img = Image.new('L', (width, height), color=128)
                img = img.convert('RGB')
                
            else:
                # Default edge case
                img = Image.new('RGB', (width, height), color='white')
                draw = ImageDraw.Draw(img)
                draw.rectangle([width//3, height//3, 2*width//3, 2*height//3], fill='lightgray')
            
            # Save image
            output_path = edge_cases_dir / filename
            img.save(output_path, "JPEG", quality=70 if spec.get("quality") == "poor" else 85)
            logger.info(f"Generated edge case: {filename}")
    
    def _generate_raw_format_samples(self):
        """Generate raw format test samples."""
        import numpy as np
        
        raw_formats_dir = self.input_dir / "raw_formats" 
        raw_formats_dir.mkdir(exist_ok=True)
        
        for spec in self.image_specs["raw_formats"]:
            width, height = spec["resolution"]
            filename = spec["filename"]
            pixel_format = spec["format"]
            
            if pixel_format == "RGB24":
                # Generate RGB24 data
                data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                
            elif pixel_format == "RGBA32":
                # Generate RGBA32 data
                data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
                
            elif pixel_format in ["YUV420P", "YUV444P"]:
                # Generate YUV data (simplified)
                y_plane = np.random.randint(16, 236, (height, width), dtype=np.uint8)
                
                if pixel_format == "YUV420P":
                    u_plane = np.random.randint(16, 240, (height//2, width//2), dtype=np.uint8)
                    v_plane = np.random.randint(16, 240, (height//2, width//2), dtype=np.uint8)
                else:  # YUV444P
                    u_plane = np.random.randint(16, 240, (height, width), dtype=np.uint8)
                    v_plane = np.random.randint(16, 240, (height, width), dtype=np.uint8)
                
                # Combine planes
                data = np.concatenate([y_plane.flatten(), u_plane.flatten(), v_plane.flatten()])
            
            # Save raw data
            output_path = raw_formats_dir / filename
            data.tobytes() if hasattr(data, 'tobytes') else data.tostring()
            with open(output_path, 'wb') as f:
                f.write(data.tobytes() if hasattr(data, 'tobytes') else data.tostring())
            
            logger.info(f"Generated raw format: {filename}")
    
    def _add_complex_background(self, draw, width: int, height: int):
        """Add complex background patterns."""
        import random
        
        # Add random shapes to make background complex
        for _ in range(20):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(x1, min(x1 + 50, width))
            y2 = random.randint(y1, min(y1 + 50, height))
            
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    def generate_js_reference(self) -> Dict:
        """
        Generate reference outputs using JavaScript implementation.
        
        NOTE: This requires the JavaScript implementation to be available.
        In a real scenario, you would need to implement js_reference_runner.js
        """
        logger.info("Generating JavaScript reference outputs...")
        
        results = {}
        
        if not self.js_runner.exists():
            logger.warning(f"JavaScript runner not found at {self.js_runner}")
            logger.warning("Creating placeholder reference data...")
            return self._create_placeholder_reference()
        
        # Process each category
        for category in ["portraits", "products", "complex", "edge_cases"]:
            category_dir = self.input_dir / category
            if not category_dir.exists():
                continue
                
            results[category] = {}
            
            for image_file in category_dir.glob("*.jpg"):
                logger.info(f"Processing {image_file.name}...")
                
                try:
                    result = self._run_js_reference(image_file)
                    results[category][image_file.name] = result
                except Exception as e:
                    logger.error(f"Failed to process {image_file.name}: {e}")
                    results[category][image_file.name] = {"error": str(e)}
        
        # Save results
        benchmark_file = self.expected_dir / "benchmarks" / "javascript_baseline.json"
        benchmark_file.parent.mkdir(exist_ok=True)
        
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"JavaScript reference data saved to {benchmark_file}")
        return results
    
    def _run_js_reference(self, image_path: Path) -> Dict:
        """Run JavaScript reference implementation on single image."""
        output_base = self.expected_dir / "javascript_output" / image_path.stem
        
        cmd = [
            'node', str(self.js_runner),
            '--input', str(image_path),
            '--output-png', f"{output_base}.png",
            '--output-jpg', f"{output_base}.jpg", 
            '--output-webp', f"{output_base}.webp",
            '--output-mask', f"{output_base}_mask.png"
        ]
        
        # Measure performance
        start_time = time.time()
        process = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor memory usage
        memory_samples = []
        try:
            while process.poll() is None:
                try:
                    memory_info = process.memory_info()
                    memory_samples.append(memory_info.rss / 1024 / 1024)  # MB
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.1)
        except Exception:
            pass
        
        stdout, stderr = process.communicate()
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # ms
        peak_memory = max(memory_samples) if memory_samples else 0
        
        return {
            'success': process.returncode == 0,
            'processing_time_ms': processing_time,
            'peak_memory_mb': peak_memory,
            'stdout': stdout,
            'stderr': stderr,
            'output_files': self._list_output_files(output_base)
        }
    
    def _list_output_files(self, output_base: Path) -> List[str]:
        """List generated output files."""
        extensions = ['.png', '.jpg', '.webp', '_mask.png']
        files = []
        
        for ext in extensions:
            file_path = Path(str(output_base) + ext)
            if file_path.exists():
                files.append(file_path.name)
        
        return files
    
    def _create_placeholder_reference(self) -> Dict:
        """Create placeholder reference data when JS implementation unavailable."""
        logger.info("Creating placeholder reference data...")
        
        placeholder = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "Placeholder data - JavaScript implementation not available",
            "portraits": {
                "avg_processing_time_ms": 1250,
                "peak_memory_mb": 180,
                "total_images": 10
            },
            "products": {
                "avg_processing_time_ms": 950,
                "peak_memory_mb": 165,
                "total_images": 8
            },
            "complex": {
                "avg_processing_time_ms": 1800,
                "peak_memory_mb": 220,
                "total_images": 6
            },
            "edge_cases": {
                "avg_processing_time_ms": 2000,
                "peak_memory_mb": 200,
                "total_images": 8
            }
        }
        
        # Create placeholder output files
        self._create_placeholder_outputs()
        
        return placeholder
    
    def _create_placeholder_outputs(self):
        """Create placeholder output files for testing."""
        from PIL import Image
        
        js_output_dir = self.expected_dir / "javascript_output"
        masks_dir = self.expected_dir / "masks"
        
        for dir_path in [js_output_dir, masks_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Create placeholder images for each test image
        for category in ["portraits", "products", "complex", "edge_cases"]:
            category_specs = self.image_specs.get(category, [])
            
            for spec in category_specs:
                base_name = Path(spec["filename"]).stem
                width, height = spec["resolution"]
                
                # Create placeholder processed image (transparent background)
                img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
                img.save(js_output_dir / f"{base_name}.png")
                
                # Create placeholder mask
                mask = Image.new('L', (width, height), color=255)
                mask.save(masks_dir / f"js_{base_name}_mask.png")
        
        logger.info("Created placeholder output files")
    
    def generate_checksums(self) -> Dict[str, str]:
        """Generate checksums for all test files."""
        logger.info("Generating file checksums...")
        
        checksums = {}
        
        for root, dirs, files in os.walk(self.assets_dir):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.assets_dir)
                
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    checksums[str(relative_path)] = file_hash
        
        # Save checksums
        checksum_file = self.metadata_dir / "file_checksums.json"
        with open(checksum_file, 'w') as f:
            json.dump(checksums, f, indent=2, sort_keys=True)
        
        logger.info(f"Checksums saved to {checksum_file}")
        return checksums


def main():
    """Main entry point for reference generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reference test data")
    parser.add_argument("--assets-dir", type=Path, 
                       default=Path(__file__).parent.parent / "assets",
                       help="Test assets directory")
    parser.add_argument("--js-runner", type=Path,
                       help="Path to JavaScript reference runner")
    parser.add_argument("--generate-images", action="store_true",
                       help="Generate synthetic test images")
    parser.add_argument("--generate-reference", action="store_true", 
                       help="Generate JavaScript reference outputs")
    parser.add_argument("--generate-checksums", action="store_true",
                       help="Generate file checksums")
    parser.add_argument("--all", action="store_true",
                       help="Generate everything")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ReferenceGenerator(args.assets_dir, args.js_runner)
    
    try:
        success = True
        
        if args.all or args.generate_images:
            success &= generator.generate_test_images()
        
        if args.all or args.generate_reference:
            generator.generate_js_reference()
        
        if args.all or args.generate_checksums:
            generator.generate_checksums()
        
        if success:
            logger.info("Reference generation completed successfully")
        else:
            logger.error("Reference generation completed with errors")
            return 1
            
    except Exception as e:
        logger.error(f"Reference generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())