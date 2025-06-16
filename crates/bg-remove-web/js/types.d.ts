/**
 * TypeScript type definitions for bg-remove-web
 * 
 * This module provides type definitions for the WebAssembly background removal library,
 * enabling type-safe usage in TypeScript projects.
 */

declare module 'bg-remove-web' {
    /**
     * Configuration options for background removal operations
     */
    export interface RemovalOptions {
        /** Name of the model to use (optional, defaults to available model) */
        modelName?: string;
        /** JPEG quality (0-100) */
        jpegQuality?: number;
        /** WebP quality (0-100) */
        webpQuality?: number;
        /** Background color (hex string, e.g., "#ffffff") */
        backgroundColor?: string;
        /** Whether to preserve color profiles */
        preserveColorProfile?: boolean;
        /** Progress callback function */
        onProgress?: (progress: ProcessingProgress) => void;
    }

    /**
     * Processing progress information
     */
    export interface ProcessingProgress {
        /** Current processing stage */
        readonly stage: string;
        /** Progress percentage (0-100) */
        readonly progress: number;
        /** Descriptive message */
        readonly message: string;
    }

    /**
     * WebAssembly error type
     */
    export interface WasmError extends Error {
        /** Error message */
        readonly message: string;
    }

    /**
     * Configuration for web-specific removal settings
     */
    export class WebRemovalConfig {
        constructor();
        
        jpegQuality: number;
        webpQuality: number;
        backgroundColor: string;
        preserveColorProfile: boolean;
    }

    /**
     * Main background removal class (WebAssembly binding)
     */
    export class BackgroundRemover {
        constructor();
        
        /**
         * Initialize the background remover with a specific model
         * @param modelName Optional model name
         * @returns Promise that resolves when initialization is complete
         */
        initialize(modelName?: string): Promise<void>;
        
        /**
         * Check if the remover is initialized
         */
        isInitialized(): boolean;
        
        /**
         * Remove background from ImageData
         * @param imageData Canvas ImageData
         * @returns Promise that resolves to processed ImageData
         */
        removeBackgroundFromImageData(imageData: ImageData): Promise<ImageData>;
        
        /**
         * Get list of available embedded models
         */
        static getAvailableModels(): string[];
    }

    /**
     * High-level JavaScript API for background removal
     */
    export class BackgroundRemovalAPI {
        constructor();
        
        /**
         * Initialize the API
         * @param options Configuration options
         */
        initialize(options?: RemovalOptions): Promise<void>;
        
        /**
         * Check if the API is initialized and ready
         */
        isInitialized(): boolean;
        
        /**
         * Get list of available models
         */
        getAvailableModels(): string[];
        
        /**
         * Remove background from a canvas element
         * @param canvas Source canvas
         * @param options Processing options
         * @returns Canvas with background removed
         */
        removeBackgroundFromCanvas(canvas: HTMLCanvasElement, options?: RemovalOptions): Promise<HTMLCanvasElement>;
        
        /**
         * Remove background from an image element
         * @param image Source image
         * @param options Processing options
         * @returns Canvas with background removed
         */
        removeBackgroundFromImage(image: HTMLImageElement, options?: RemovalOptions): Promise<HTMLCanvasElement>;
        
        /**
         * Remove background from a File object
         * @param file Image file
         * @param options Processing options
         * @returns Canvas with background removed
         */
        removeBackgroundFromFile(file: File, options?: RemovalOptions): Promise<HTMLCanvasElement>;
        
        /**
         * Convert canvas to Blob
         * @param canvas Source canvas
         * @param format Output format
         * @param quality Quality (0-1)
         */
        canvasToBlob(canvas: HTMLCanvasElement, format?: string, quality?: number): Promise<Blob>;
        
        /**
         * Download canvas as image file
         * @param canvas Source canvas
         * @param filename Download filename
         * @param format Image format
         * @param quality Quality (0-1)
         */
        downloadCanvas(canvas: HTMLCanvasElement, filename?: string, format?: string, quality?: number): Promise<void>;
    }

    /**
     * Simple function-based API for quick usage
     * @param input Input image (Canvas, Image, or File)
     * @param options Configuration options
     * @returns Canvas with background removed
     */
    export function removeBackground(
        input: HTMLCanvasElement | HTMLImageElement | File,
        options?: RemovalOptions
    ): Promise<HTMLCanvasElement>;

    /**
     * Get version information
     */
    export function getVersion(): string;

    /**
     * Get information about available execution providers
     */
    export function getWasmProviders(): Record<string, any>;

    /**
     * Initialize the WASM module (called automatically)
     */
    export function init(): void;

    // Export the main API class as default
    export default BackgroundRemovalAPI;
}

// Module augmentation for direct WASM usage
declare module 'bg-remove-web/pkg' {
    export * from 'bg-remove-web';
}

// Global type declarations for browser environments
declare global {
    interface Window {
        bgRemoveWeb?: {
            BackgroundRemovalAPI: typeof import('bg-remove-web').BackgroundRemovalAPI;
            removeBackground: typeof import('bg-remove-web').removeBackground;
        };
    }
}