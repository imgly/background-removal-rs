#!/usr/bin/env python3
"""
Real image downloader for background removal testing.

This script downloads real images from the internet to create a comprehensive
test dataset for validating the background removal implementation.
"""

import requests
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
from urllib.parse import urlparse
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealImageDownloader:
    """Downloads real images for background removal testing."""
    
    def __init__(self, test_assets_dir: Path):
        """
        Initialize the image downloader.
        
        Args:
            test_assets_dir: Path to test assets directory
        """
        self.assets_dir = test_assets_dir
        self.input_dir = test_assets_dir / "input"
        self.metadata_dir = test_assets_dir / "metadata"
        
        # Create directories
        for category in ["portraits", "products", "complex", "edge_cases", "raw_formats"]:
            (self.input_dir / category).mkdir(parents=True, exist_ok=True)
        
        # Image sources - using freely available images
        self.image_sources = self._get_image_sources()
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _get_image_sources(self) -> Dict[str, List[Dict]]:
        """Get curated list of free image sources for testing."""
        return {
            "portraits": [
                {
                    "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80",
                    "filename": "portrait_single_simple_bg.jpg",
                    "description": "Professional headshot with clean background",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1494790108755-2616c829daf5?w=800&q=80",
                    "filename": "portrait_single_complex_bg.jpg", 
                    "description": "Portrait with detailed background",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=800&q=80",
                    "filename": "portrait_fine_hair_details.jpg",
                    "description": "Portrait with fine hair details",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1552058544-f2b08422138a?w=800&q=80",
                    "filename": "portrait_outdoor_natural.jpg",
                    "description": "Outdoor portrait with natural lighting",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=800&q=80",
                    "filename": "portrait_side_profile.jpg",
                    "description": "Side profile portrait",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=800&q=80",
                    "filename": "portrait_studio_lighting.jpg",
                    "description": "Studio portrait with professional lighting",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=800&q=80",
                    "filename": "portrait_multiple_people.jpg",
                    "description": "Multiple people in frame",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=800&q=80",
                    "filename": "portrait_partial_figure.jpg",
                    "description": "Partial figure portrait",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=800&q=80",
                    "filename": "portrait_low_contrast.jpg",
                    "description": "Low contrast portrait",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&q=80",
                    "filename": "portrait_action_motion.jpg",
                    "description": "Portrait with motion/action",
                    "source": "Unsplash - Free License"
                }
            ],
            "products": [
                {
                    "url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=800&q=80",
                    "filename": "product_clothing_white_bg.jpg",
                    "description": "Clothing item on white background",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=800&q=80",
                    "filename": "product_electronics_gradient.jpg",
                    "description": "Electronic device with gradient background",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&q=80",
                    "filename": "product_furniture_textured.jpg",
                    "description": "Furniture with textured background",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800&q=80",
                    "filename": "product_accessories_glass.jpg",
                    "description": "Watch or accessory with reflective surfaces",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&q=80",
                    "filename": "product_multiple_items.jpg",
                    "description": "Multiple products together",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800&q=80",
                    "filename": "product_irregular_shape.jpg",
                    "description": "Product with complex irregular shape",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=800&q=80",
                    "filename": "product_shadow_handling.jpg",
                    "description": "Product with strong shadows",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800&q=80",
                    "filename": "product_clothing_transparent.jpg",
                    "description": "Clothing with transparent elements",
                    "source": "Unsplash - Free License"
                }
            ],
            "complex": [
                {
                    "url": "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=800&q=80",
                    "filename": "complex_group_photo.jpg",
                    "description": "Group of people in complex scene",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1601758228041-f3b2795255f1?w=800&q=80",
                    "filename": "complex_pet_with_person.jpg",
                    "description": "Person with pet",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=800&q=80",
                    "filename": "complex_overlapping_objects.jpg",
                    "description": "Multiple overlapping objects",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?w=800&q=80",
                    "filename": "complex_fine_details.jpg",
                    "description": "Scene with intricate fine details",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=800&q=80",
                    "filename": "complex_mixed_lighting.jpg",
                    "description": "Scene with mixed lighting sources",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=800&q=80",
                    "filename": "complex_transparent_elements.jpg",
                    "description": "Scene with glass and transparent objects",
                    "source": "Unsplash - Free License"
                }
            ],
            "edge_cases": [
                {
                    "url": "https://images.unsplash.com/photo-1508921912186-1d1a45ebb3c1?w=64&q=30",
                    "filename": "edge_very_small.jpg",
                    "description": "Very small resolution test image",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=2048&q=100",
                    "filename": "edge_very_large.jpg",
                    "description": "Very large resolution test image",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1519336056116-bc0f1771dec8?w=800&q=20",
                    "filename": "edge_high_noise.jpg",
                    "description": "High noise/low quality image",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&q=80&sat=-100",
                    "filename": "edge_monochrome.jpg",
                    "description": "Monochrome/grayscale image",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1494548162494-384bba4ab999?w=800&q=80",
                    "filename": "edge_high_contrast.jpg",
                    "description": "High contrast image",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800&q=80",
                    "filename": "edge_similar_colors.jpg",
                    "description": "Subject with similar background colors",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?w=800&q=80",
                    "filename": "edge_minimal_subject.jpg",
                    "description": "Very small subject in large frame",
                    "source": "Unsplash - Free License"
                },
                {
                    "url": "https://images.unsplash.com/photo-1519904981063-b0cf448d479e?w=800&q=10",
                    "filename": "edge_corrupted_data.jpg",
                    "description": "Heavily compressed/low quality image",
                    "source": "Unsplash - Free License"
                }
            ]
        }
    
    def download_image(self, url: str, output_path: Path, filename: str) -> bool:
        """
        Download a single image from URL.
        
        Args:
            url: Image URL
            output_path: Directory to save image
            filename: Filename for saved image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading {filename} from {url}")
            
            # Add delay to be respectful to servers
            time.sleep(1)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Validate it's actually an image
            try:
                img = Image.open(io.BytesIO(response.content))
                img.verify()
            except Exception as e:
                logger.error(f"Invalid image data for {filename}: {e}")
                return False
            
            # Save the image
            file_path = output_path / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Verify file was saved correctly
            if file_path.exists() and file_path.stat().st_size > 0:
                logger.info(f"Successfully downloaded {filename} ({file_path.stat().st_size} bytes)")
                return True
            else:
                logger.error(f"Failed to save {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def download_category_images(self, category: str) -> Dict[str, bool]:
        """
        Download all images for a specific category.
        
        Args:
            category: Category name (portraits, products, etc.)
            
        Returns:
            Dictionary mapping filename to success status
        """
        if category not in self.image_sources:
            logger.error(f"Unknown category: {category}")
            return {}
        
        category_dir = self.input_dir / category
        results = {}
        
        for image_info in self.image_sources[category]:
            filename = image_info["filename"]
            url = image_info["url"]
            
            success = self.download_image(url, category_dir, filename)
            results[filename] = success
            
            if not success:
                logger.warning(f"Failed to download {filename}")
        
        return results
    
    def download_all_images(self) -> Dict[str, Dict[str, bool]]:
        """
        Download all test images.
        
        Returns:
            Nested dictionary with category -> filename -> success status
        """
        logger.info("Starting download of all test images...")
        
        all_results = {}
        
        for category in self.image_sources.keys():
            logger.info(f"Downloading {category} images...")
            category_results = self.download_category_images(category)
            all_results[category] = category_results
            
            # Log category summary
            successful = sum(1 for success in category_results.values() if success)
            total = len(category_results)
            logger.info(f"Category {category}: {successful}/{total} images downloaded successfully")
        
        return all_results
    
    def create_image_metadata(self, download_results: Dict[str, Dict[str, bool]]) -> Dict:
        """
        Create updated image metadata based on downloaded images.
        
        Args:
            download_results: Results from image downloads
            
        Returns:
            Updated metadata dictionary
        """
        metadata = {}
        
        for category, category_results in download_results.items():
            metadata[category] = []
            
            for filename, success in category_results.items():
                if not success:
                    continue
                
                # Get image info from sources
                image_info = next(
                    (info for info in self.image_sources[category] if info["filename"] == filename),
                    None
                )
                
                if not image_info:
                    continue
                
                # Get actual image dimensions
                image_path = self.input_dir / category / filename
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        format_name = img.format
                except Exception as e:
                    logger.warning(f"Could not read image info for {filename}: {e}")
                    width, height = 800, 600  # Default
                    format_name = "JPEG"
                
                # Create metadata entry
                entry = {
                    "filename": filename,
                    "resolution": [width, height],
                    "format": format_name,
                    "description": image_info["description"],
                    "source": image_info["source"],
                    "url": image_info["url"],
                    "downloaded": True
                }
                
                # Add category-specific metadata
                if category == "portraits":
                    entry.update({
                        "subject_type": self._infer_subject_type(filename),
                        "background_complexity": self._infer_background_complexity(filename),
                        "edge_complexity": self._infer_edge_complexity(filename),
                        "lighting": self._infer_lighting(filename),
                        "expected_difficulty": self._infer_difficulty(filename)
                    })
                elif category == "products":
                    entry.update({
                        "product_type": self._infer_product_type(filename),
                        "material": self._infer_material(filename),
                        "background_type": self._infer_background_type(filename),
                        "transparency": self._infer_transparency(filename),
                        "expected_difficulty": self._infer_difficulty(filename)
                    })
                elif category == "complex":
                    entry.update({
                        "subjects": self._infer_subjects(filename),
                        "background_complexity": "complex",
                        "occlusion": True,
                        "lighting": self._infer_lighting(filename),
                        "expected_difficulty": "hard"
                    })
                elif category == "edge_cases":
                    entry.update({
                        "challenge_type": self._infer_challenge_type(filename),
                        "quality": self._infer_quality(filename),
                        "expected_difficulty": self._infer_difficulty(filename)
                    })
                
                metadata[category].append(entry)
        
        return metadata
    
    def _infer_subject_type(self, filename: str) -> str:
        """Infer subject type from filename."""
        if "multiple" in filename:
            return "multiple_people"
        elif "partial" in filename:
            return "partial_person"
        else:
            return "single_person"
    
    def _infer_background_complexity(self, filename: str) -> str:
        """Infer background complexity from filename."""
        if "simple" in filename or "white" in filename:
            return "simple"
        elif "complex" in filename:
            return "complex"
        elif "outdoor" in filename:
            return "natural"
        else:
            return "moderate"
    
    def _infer_edge_complexity(self, filename: str) -> str:
        """Infer edge complexity from filename."""
        if "hair" in filename:
            return "very_fine"
        elif "motion" in filename:
            return "motion_blur"
        elif "low_contrast" in filename:
            return "low_contrast"
        else:
            return "clean"
    
    def _infer_lighting(self, filename: str) -> str:
        """Infer lighting type from filename."""
        if "studio" in filename:
            return "studio_professional"
        elif "outdoor" in filename:
            return "outdoor_natural"
        elif "side" in filename:
            return "side_lit"
        elif "mixed" in filename:
            return "mixed_sources"
        else:
            return "natural"
    
    def _infer_difficulty(self, filename: str) -> str:
        """Infer expected difficulty from filename."""
        if any(word in filename for word in ["simple", "studio", "white", "clean"]):
            return "easy"
        elif any(word in filename for word in ["complex", "hair", "motion", "contrast", "noise", "similar"]):
            return "hard"
        elif any(word in filename for word in ["transparent", "glass", "fine", "small", "large"]):
            return "very_hard"
        else:
            return "medium"
    
    def _infer_product_type(self, filename: str) -> str:
        """Infer product type from filename."""
        if "clothing" in filename:
            return "clothing"
        elif "electronics" in filename:
            return "electronics"
        elif "furniture" in filename:
            return "furniture"
        elif "accessories" in filename:
            return "accessories"
        else:
            return "mixed"
    
    def _infer_material(self, filename: str) -> str:
        """Infer material type from filename."""
        if "glass" in filename:
            return "glass_metal"
        elif "transparent" in filename:
            return "transparent_fabric"
        elif "clothing" in filename:
            return "fabric"
        else:
            return "various"
    
    def _infer_background_type(self, filename: str) -> str:
        """Infer background type from filename."""
        if "white" in filename:
            return "white"
        elif "gradient" in filename:
            return "gradient"
        elif "textured" in filename:
            return "textured"
        else:
            return "neutral"
    
    def _infer_transparency(self, filename: str) -> bool:
        """Infer if product has transparency from filename."""
        return "transparent" in filename or "glass" in filename
    
    def _infer_subjects(self, filename: str) -> str:
        """Infer subject types for complex scenes."""
        if "pet" in filename:
            return "person_and_pet"
        elif "group" in filename:
            return "multiple_people"
        elif "overlapping" in filename:
            return "multiple_objects"
        elif "transparent" in filename:
            return "transparent_objects"
        else:
            return "person_and_objects"
    
    def _infer_challenge_type(self, filename: str) -> str:
        """Infer challenge type for edge cases."""
        if "small" in filename:
            return "resolution"
        elif "large" in filename:
            return "resolution"
        elif "noise" in filename or "corrupted" in filename:
            return "quality"
        elif "monochrome" in filename:
            return "color"
        elif "contrast" in filename:
            return "contrast"
        elif "similar" in filename:
            return "contrast"
        elif "minimal" in filename:
            return "size"
        else:
            return "quality"
    
    def _infer_quality(self, filename: str) -> str:
        """Infer image quality from filename."""
        if "noise" in filename or "corrupted" in filename:
            return "poor"
        else:
            return "normal"
    
    def generate_checksums(self) -> Dict[str, str]:
        """Generate checksums for all downloaded images."""
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
    """Main entry point for real image download."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real test images")
    parser.add_argument("--assets-dir", type=Path,
                       default=Path(__file__).parent.parent / "assets",
                       help="Test assets directory")
    parser.add_argument("--category", type=str,
                       help="Download only specific category")
    parser.add_argument("--update-metadata", action="store_true",
                       help="Update metadata files")
    parser.add_argument("--generate-checksums", action="store_true",
                       help="Generate file checksums")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = RealImageDownloader(args.assets_dir)
    
    try:
        if args.category:
            # Download specific category
            results = {args.category: downloader.download_category_images(args.category)}
        else:
            # Download all images
            results = downloader.download_all_images()
        
        # Summary
        total_attempted = sum(len(category_results) for category_results in results.values())
        total_successful = sum(
            sum(1 for success in category_results.values() if success)
            for category_results in results.values()
        )
        
        logger.info(f"Download completed: {total_successful}/{total_attempted} images successful")
        
        # Update metadata if requested
        if args.update_metadata:
            logger.info("Updating image metadata...")
            metadata = downloader.create_image_metadata(results)
            
            metadata_file = args.assets_dir / "metadata" / "image_specs.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata updated: {metadata_file}")
        
        # Generate checksums if requested
        if args.generate_checksums:
            downloader.generate_checksums()
        
        if total_successful < total_attempted:
            logger.warning(f"{total_attempted - total_successful} images failed to download")
            return 1
        
        logger.info("All images downloaded successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())