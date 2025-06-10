#!/usr/bin/env node

/**
 * JavaScript Reference Runner
 * 
 * This script runs the Node.js background removal implementation on all test images
 * to generate reference outputs for validating the Rust implementation.
 */

// Use the built reference implementation directly
const referencePath = require('path').resolve(__dirname, '../../reference/packages/node/dist/index.cjs');
const {
  removeBackground,
  segmentForeground,
  applySegmentationMask
} = require(referencePath);

const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');

class JSReferenceRunner {
  constructor(assetsDir) {
    this.assetsDir = assetsDir;
    this.inputDir = path.join(assetsDir, 'input');
    this.outputDir = path.join(assetsDir, 'expected', 'javascript_output');
    this.masksDir = path.join(assetsDir, 'expected', 'masks');
    this.benchmarksDir = path.join(assetsDir, 'expected', 'benchmarks');
    
    this.results = {
      timestamp: new Date().toISOString(),
      platform: process.platform,
      node_version: process.version,
      results: {}
    };
  }
  
  async ensureDirectories() {
    await fs.mkdir(this.outputDir, { recursive: true });
    await fs.mkdir(this.masksDir, { recursive: true });
    await fs.mkdir(this.benchmarksDir, { recursive: true });
  }
  
  async processAllImages() {
    console.log('🚀 Starting JavaScript reference generation...');
    console.log(`Input directory: ${this.inputDir}`);
    console.log(`Output directory: ${this.outputDir}`);
    
    await this.ensureDirectories();
    
    // Process each category
    const categories = ['portraits', 'products', 'complex', 'edge_cases'];
    
    for (const category of categories) {
      console.log(`\n📂 Processing ${category} category...`);
      await this.processCategory(category);
    }
    
    // Save benchmark results
    await this.saveBenchmarkResults();
    
    console.log('\n✅ JavaScript reference generation completed!');
    console.log(`Results saved to: ${this.outputDir}`);
    console.log(`Benchmarks saved to: ${this.benchmarksDir}`);
  }
  
  async processCategory(category) {
    const categoryDir = path.join(this.inputDir, category);
    const categoryResults = {
      processed: 0,
      failed: 0,
      avg_processing_time_ms: 0,
      peak_memory_mb: 0,
      total_processing_time: 0,
      images: {}
    };
    
    try {
      const files = await fs.readdir(categoryDir);
      const imageFiles = files.filter(file => file.toLowerCase().endsWith('.jpg'));
      
      console.log(`  Found ${imageFiles.length} images`);
      
      for (const imageFile of imageFiles) {
        const imagePath = path.join(categoryDir, imageFile);
        const baseName = path.parse(imageFile).name;
        
        console.log(`  Processing: ${imageFile}`);
        
        try {
          const result = await this.processImage(imagePath, baseName, category);
          categoryResults.images[imageFile] = result;
          categoryResults.processed++;
          categoryResults.total_processing_time += result.processing_time_ms;
          categoryResults.peak_memory_mb = Math.max(categoryResults.peak_memory_mb, result.memory_usage_mb);
          
          console.log(`    ✅ Success (${result.processing_time_ms.toFixed(1)}ms, ${result.memory_usage_mb.toFixed(1)}MB)`);
        } catch (error) {
          console.log(`    ❌ Failed: ${error.message}`);
          categoryResults.images[imageFile] = {
            success: false,
            error: error.message,
            processing_time_ms: 0,
            memory_usage_mb: 0
          };
          categoryResults.failed++;
        }
      }
      
      // Calculate averages
      if (categoryResults.processed > 0) {
        categoryResults.avg_processing_time_ms = categoryResults.total_processing_time / categoryResults.processed;
      }
      
      this.results.results[category] = categoryResults;
      
      console.log(`  📊 Category summary: ${categoryResults.processed}/${categoryResults.processed + categoryResults.failed} successful`);
      console.log(`     Average time: ${categoryResults.avg_processing_time_ms.toFixed(1)}ms`);
      console.log(`     Peak memory: ${categoryResults.peak_memory_mb.toFixed(1)}MB`);
      
    } catch (error) {
      console.error(`  ❌ Failed to process category ${category}: ${error.message}`);
      categoryResults.error = error.message;
      this.results.results[category] = categoryResults;
    }
  }
  
  async processImage(imagePath, baseName, category) {
    const startTime = performance.now();
    const startMemory = process.memoryUsage();
    
    // Configuration for background removal
    const config = {
      debug: false,
      model: 'isnet', // Use the same model as will be used in Rust
      output: {
        quality: 0.9,
        format: 'image/png' // PNG for alpha channel
      },
      progress: undefined // Disable progress for batch processing
    };
    
    try {
      // Generate different output formats
      const outputs = {};
      
      // 1. PNG with alpha channel (main output)
      const pngBlob = await removeBackground(imagePath, {
        ...config,
        output: { quality: 0.9, format: 'image/png' }
      });
      const pngBuffer = Buffer.from(await pngBlob.arrayBuffer());
      const pngOutputPath = path.join(this.outputDir, `${baseName}.png`);
      await fs.writeFile(pngOutputPath, pngBuffer);
      outputs.png = pngOutputPath;
      
      // 2. JPEG with white background
      const jpegBlob = await removeBackground(imagePath, {
        ...config,
        output: { quality: 0.9, format: 'image/jpeg' }
      });
      const jpegBuffer = Buffer.from(await jpegBlob.arrayBuffer());
      const jpegOutputPath = path.join(this.outputDir, `${baseName}.jpg`);
      await fs.writeFile(jpegOutputPath, jpegBuffer);
      outputs.jpeg = jpegOutputPath;
      
      // 3. WebP with alpha channel
      const webpBlob = await removeBackground(imagePath, {
        ...config,
        output: { quality: 0.85, format: 'image/webp' }
      });
      const webpBuffer = Buffer.from(await webpBlob.arrayBuffer());
      const webpOutputPath = path.join(this.outputDir, `${baseName}.webp`);
      await fs.writeFile(webpOutputPath, webpBuffer);
      outputs.webp = webpOutputPath;
      
      // 4. Segmentation mask
      const maskBlob = await segmentForeground(imagePath, config);
      const maskBuffer = Buffer.from(await maskBlob.arrayBuffer());
      const maskOutputPath = path.join(this.masksDir, `js_${baseName}_mask.png`);
      await fs.writeFile(maskOutputPath, maskBuffer);
      outputs.mask = maskOutputPath;
      
      const endTime = performance.now();
      const endMemory = process.memoryUsage();
      
      const processingTime = endTime - startTime;
      const memoryUsage = (endMemory.heapUsed - startMemory.heapUsed) / 1024 / 1024; // MB
      
      return {
        success: true,
        processing_time_ms: processingTime,
        memory_usage_mb: Math.max(0, memoryUsage), // Ensure non-negative
        output_files: Object.values(outputs).map(p => path.basename(p)),
        outputs: outputs
      };
      
    } catch (error) {
      throw new Error(`Processing failed: ${error.message}`);
    }
  }
  
  async saveBenchmarkResults() {
    const benchmarkFile = path.join(this.benchmarksDir, 'javascript_baseline.json');
    
    // Add overall summary
    this.results.summary = {
      total_categories: Object.keys(this.results.results).length,
      total_images_processed: Object.values(this.results.results).reduce((sum, cat) => sum + cat.processed, 0),
      total_images_failed: Object.values(this.results.results).reduce((sum, cat) => sum + cat.failed, 0),
      overall_avg_time_ms: this.calculateOverallAverageTime(),
      overall_peak_memory_mb: Math.max(...Object.values(this.results.results).map(cat => cat.peak_memory_mb || 0))
    };
    
    await fs.writeFile(benchmarkFile, JSON.stringify(this.results, null, 2));
    console.log(`\n📊 Benchmark results saved to: ${benchmarkFile}`);
  }
  
  calculateOverallAverageTime() {
    const categories = Object.values(this.results.results);
    const totalTime = categories.reduce((sum, cat) => sum + (cat.total_processing_time || 0), 0);
    const totalImages = categories.reduce((sum, cat) => sum + cat.processed, 0);
    return totalImages > 0 ? totalTime / totalImages : 0;
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);
  
  // Parse command line arguments
  let assetsDir = path.join(__dirname, '..', 'assets');
  let category = null;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--assets-dir' && i + 1 < args.length) {
      assetsDir = args[i + 1];
      i++;
    } else if (args[i] === '--category' && i + 1 < args.length) {
      category = args[i + 1];
      i++;
    } else if (args[i] === '--help') {
      console.log(`
JavaScript Reference Runner

Usage: node js_reference_runner.js [options]

Options:
  --assets-dir PATH    Path to test assets directory (default: ../assets)
  --category NAME      Process only specific category (portraits, products, complex, edge_cases)
  --help              Show this help message

Examples:
  node js_reference_runner.js
  node js_reference_runner.js --category portraits
  node js_reference_runner.js --assets-dir /path/to/assets
      `);
      process.exit(0);
    }
  }
  
  try {
    const runner = new JSReferenceRunner(assetsDir);
    
    if (category) {
      console.log(`Processing single category: ${category}`);
      await runner.ensureDirectories();
      await runner.processCategory(category);
      await runner.saveBenchmarkResults();
    } else {
      await runner.processAllImages();
    }
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Handle process termination gracefully
process.on('SIGINT', () => {
  console.log('\n⏹️  Process interrupted by user');
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

if (require.main === module) {
  main();
}

module.exports = { JSReferenceRunner };