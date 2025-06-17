/**
 * High-level JavaScript API for bg-remove-web WASM module
 * 
 * This module provides a convenient JavaScript interface for background removal
 * in web browsers using WebAssembly and the pure Rust Tract backend.
 * 
 * @example
 * ```javascript
 * import { BackgroundRemovalAPI } from './index.js';
 * 
 * const api = new BackgroundRemovalAPI();
 * await api.initialize();
 * 
 * const canvas = document.getElementById('myCanvas');
 * const result = await api.removeBackground(canvas);
 * ```
 */

/**
 * High-level API for background removal operations
 */
export class BackgroundRemovalAPI {
    constructor() {
        this.wasmModule = null;
        this.remover = null;
        this.initialized = false;
    }

    /**
     * Initialize the background removal API
     * @param {Object} options - Configuration options
     * @param {string} options.modelName - Name of the model to use (optional)
     * @param {string} options.outputFormat - Output format (optional)
     * @param {number} options.jpegQuality - JPEG quality 0-100 (optional)
     * @param {number} options.webpQuality - WebP quality 0-100 (optional)
     * @param {string} options.backgroundColor - Background color hex (optional)
     * @param {boolean} options.debug - Enable debug mode (optional)
     * @param {number} options.intraThreads - Number of intra-op threads (optional)
     * @param {number} options.interThreads - Number of inter-op threads (optional)
     * @param {boolean} options.preserveColorProfile - Preserve ICC profiles (optional)
     * @param {boolean} options.forceSrgbOutput - Force sRGB output (optional)
     * @param {boolean} options.fallbackToSrgb - Fallback to sRGB (optional)
     * @param {boolean} options.embedProfileInOutput - Embed profile in output (optional)
     * @param {Function} options.onProgress - Progress callback (optional)
     * @returns {Promise<void>}
     */
    async initialize(options = {}) {
        console.log('🌐 Initializing bg-remove-web API...');
        
        try {
            // Import the WASM module
            this.wasmModule = await import('../pkg/bg_remove_web.js');
            await this.wasmModule.default();
            
            console.log('✅ WASM module loaded');
            console.log('📦 Version:', this.wasmModule.get_version());
            console.log('🔧 Providers:', this.wasmModule.get_wasm_providers());
            
            // Create config
            const config = new this.wasmModule.WebRemovalConfig();
            
            // Apply configuration options
            if (options.outputFormat) config.outputFormat = options.outputFormat;
            if (options.jpegQuality !== undefined) config.jpegQuality = options.jpegQuality;
            if (options.webpQuality !== undefined) config.webpQuality = options.webpQuality;
            if (options.backgroundColor) config.backgroundColor = options.backgroundColor;
            if (options.debug !== undefined) config.debug = options.debug;
            if (options.intraThreads !== undefined) config.intraThreads = options.intraThreads;
            if (options.interThreads !== undefined) config.interThreads = options.interThreads;
            if (options.preserveColorProfile !== undefined) config.preserveColorProfile = options.preserveColorProfile;
            if (options.forceSrgbOutput !== undefined) config.forceSrgbOutput = options.forceSrgbOutput;
            if (options.fallbackToSrgb !== undefined) config.fallbackToSrgb = options.fallbackToSrgb;
            if (options.embedProfileInOutput !== undefined) config.embedProfileInOutput = options.embedProfileInOutput;
            
            // Create remover instance with config
            this.remover = new this.wasmModule.BackgroundRemover(config);
            
            // Initialize with model
            const modelName = options.modelName || null;
            await this.remover.initialize(modelName);
            
            this.initialized = true;
            console.log('🎯 BackgroundRemovalAPI ready');
            
        } catch (error) {
            console.error('❌ Failed to initialize bg-remove-web:', error);
            throw new Error(`Initialization failed: ${error.message}`);
        }
    }

    /**
     * Check if the API is initialized and ready
     * @returns {boolean}
     */
    isInitialized() {
        return this.initialized && this.remover && this.remover.is_initialized();
    }

    /**
     * Get list of available models
     * @returns {string[]} Array of model names
     */
    getAvailableModels() {
        if (!this.wasmModule) {
            throw new Error('API not initialized');
        }
        
        const models = this.wasmModule.BackgroundRemover.get_available_models();
        return Array.from(models);
    }

    /**
     * Remove background from a canvas element
     * @param {HTMLCanvasElement} canvas - Source canvas
     * @param {Object} options - Processing options
     * @returns {Promise<HTMLCanvasElement>} Canvas with background removed
     */
    async removeBackgroundFromCanvas(canvas, options = {}) {
        if (!this.isInitialized()) {
            throw new Error('API not initialized');
        }

        try {
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            console.log(`🖼️ Processing canvas: ${canvas.width}x${canvas.height}`);
            
            const resultImageData = await this.remover.remove_background_from_image_data(imageData);
            
            // Create result canvas
            const resultCanvas = document.createElement('canvas');
            resultCanvas.width = canvas.width;
            resultCanvas.height = canvas.height;
            
            const resultCtx = resultCanvas.getContext('2d');
            resultCtx.putImageData(resultImageData, 0, 0);
            
            console.log('✅ Background removal completed');
            return resultCanvas;
            
        } catch (error) {
            console.error('❌ Background removal failed:', error);
            throw new Error(`Processing failed: ${error.message}`);
        }
    }

    /**
     * Remove background from an image element
     * @param {HTMLImageElement} image - Source image
     * @param {Object} options - Processing options
     * @returns {Promise<HTMLCanvasElement>} Canvas with background removed
     */
    async removeBackgroundFromImage(image, options = {}) {
        if (!this.isInitialized()) {
            throw new Error('API not initialized');
        }

        // Create canvas from image
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth || image.width;
        canvas.height = image.naturalHeight || image.height;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        
        return this.removeBackgroundFromCanvas(canvas, options);
    }

    /**
     * Remove background from a File object (uploaded image)
     * @param {File} file - Image file
     * @param {Object} options - Processing options
     * @returns {Promise<HTMLCanvasElement>} Canvas with background removed
     */
    async removeBackgroundFromFile(file, options = {}) {
        if (!this.isInitialized()) {
            throw new Error('API not initialized');
        }

        return new Promise((resolve, reject) => {
            const image = new Image();
            
            image.onload = async () => {
                try {
                    const result = await this.removeBackgroundFromImage(image, options);
                    URL.revokeObjectURL(image.src); // Clean up
                    resolve(result);
                } catch (error) {
                    reject(error);
                }
            };
            
            image.onerror = () => {
                reject(new Error('Failed to load image file'));
            };
            
            image.src = URL.createObjectURL(file);
        });
    }

    /**
     * Convert canvas to various output formats
     * @param {HTMLCanvasElement} canvas - Source canvas
     * @param {string} format - Output format ('png', 'jpeg', 'webp')
     * @param {number} quality - Quality (0-1, for lossy formats)
     * @returns {Promise<Blob>} Image data as Blob
     */
    async canvasToBlob(canvas, format = 'png', quality = 0.9) {
        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to convert canvas to blob'));
                }
            }, `image/${format}`, quality);
        });
    }

    /**
     * Download canvas as image file
     * @param {HTMLCanvasElement} canvas - Source canvas
     * @param {string} filename - Download filename
     * @param {string} format - Image format ('png', 'jpeg', 'webp')
     * @param {number} quality - Quality (0-1, for lossy formats)
     */
    async downloadCanvas(canvas, filename = 'background-removed.png', format = 'png', quality = 0.9) {
        const blob = await this.canvasToBlob(canvas, format, quality);
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

/**
 * Simple function-based API for quick usage
 * @param {HTMLCanvasElement|HTMLImageElement|File} input - Input image
 * @param {Object} options - Configuration options
 * @returns {Promise<HTMLCanvasElement>} Canvas with background removed
 */
export async function removeBackground(input, options = {}) {
    const api = new BackgroundRemovalAPI();
    await api.initialize(options);
    
    if (input instanceof HTMLCanvasElement) {
        return api.removeBackgroundFromCanvas(input, options);
    } else if (input instanceof HTMLImageElement) {
        return api.removeBackgroundFromImage(input, options);
    } else if (input instanceof File) {
        return api.removeBackgroundFromFile(input, options);
    } else {
        throw new Error('Unsupported input type. Use HTMLCanvasElement, HTMLImageElement, or File');
    }
}

// Export the WASM module for advanced usage
export { BackgroundRemovalAPI as default };