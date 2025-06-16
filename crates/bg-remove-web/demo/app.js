/**
 * Demo application for bg-remove-web
 * 
 * This script provides the interactive functionality for the HTML demo page,
 * handling file uploads, drag-and-drop, processing, and results display.
 */

import { BackgroundRemovalAPI } from '../js/index.js';

// Global state
let api = null;
let currentFile = null;
let processingStats = {
    modelName: '',
    startTime: 0,
    endTime: 0,
    imageSize: ''
};

// DOM elements
const elements = {
    initStatus: document.getElementById('init-status'),
    uploadSection: document.querySelector('.upload-section'),
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    processingSection: document.querySelector('.processing-section'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    resultsSection: document.querySelector('.results-section'),
    errorSection: document.querySelector('.error-section'),
    errorMessage: document.getElementById('error-message'),
    originalCanvas: document.getElementById('original-canvas'),
    resultCanvas: document.getElementById('result-canvas'),
    originalInfo: document.getElementById('original-info'),
    downloadBtn: document.getElementById('download-btn'),
    downloadJpgBtn: document.getElementById('download-jpg-btn'),
    processAnotherBtn: document.getElementById('process-another-btn'),
    retryBtn: document.getElementById('retry-btn'),
    modelName: document.getElementById('model-name'),
    processingTime: document.getElementById('processing-time'),
    imageSize: document.getElementById('image-size'),
    availableModels: document.getElementById('available-models'),
    version: document.getElementById('version')
};

/**
 * Initialize the application
 */
async function initializeApp() {
    console.log('🚀 Initializing bg-remove-web demo...');
    
    try {
        updateInitStatus('Initializing WASM module...', 'Loading WebAssembly bindings');
        
        // Create API instance
        api = new BackgroundRemovalAPI();
        
        updateInitStatus('Loading AI model...', 'Preparing neural network for inference');
        
        // Initialize with progress tracking
        await api.initialize({
            onProgress: (progress) => {
                updateInitStatus(progress.stage, progress.message);
            }
        });
        
        console.log('✅ Initialization complete');
        
        // Update UI with model information
        updateModelInfo();
        
        // Show upload section
        showSection('upload');
        
        // Set up event listeners
        setupEventListeners();
        
    } catch (error) {
        console.error('❌ Initialization failed:', error);
        showError(`Failed to initialize: ${error.message}`);
    }
}

/**
 * Update initialization status display
 */
function updateInitStatus(stage, message) {
    if (elements.initStatus) {
        const statusText = elements.initStatus.querySelector('p');
        const statusDetail = elements.initStatus.querySelector('small');
        
        if (statusText) statusText.textContent = stage;
        if (statusDetail) statusDetail.textContent = message;
    }
}

/**
 * Update model information in the UI
 */
function updateModelInfo() {
    try {
        // Update version
        if (elements.version) {
            elements.version.textContent = api.wasmModule?.get_version() || '1.0.0';
        }
        
        // Update available models list
        if (elements.availableModels) {
            const models = api.getAvailableModels();
            elements.availableModels.innerHTML = '';
            
            models.forEach(model => {
                const li = document.createElement('li');
                li.textContent = model;
                elements.availableModels.appendChild(li);
            });
        }
        
    } catch (error) {
        console.warn('Could not update model info:', error);
    }
}

/**
 * Set up event listeners for user interactions
 */
function setupEventListeners() {
    // File input change
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Drag and drop
    if (elements.uploadArea) {
        elements.uploadArea.addEventListener('dragover', handleDragOver);
        elements.uploadArea.addEventListener('drop', handleFileDrop);
        elements.uploadArea.addEventListener('dragleave', handleDragLeave);
        elements.uploadArea.addEventListener('click', () => {
            elements.fileInput?.click();
        });
    }
    
    // Download buttons
    if (elements.downloadBtn) {
        elements.downloadBtn.addEventListener('click', () => {
            downloadResult('png');
        });
    }
    
    if (elements.downloadJpgBtn) {
        elements.downloadJpgBtn.addEventListener('click', () => {
            downloadResult('jpeg');
        });
    }
    
    // Process another button
    if (elements.processAnotherBtn) {
        elements.processAnotherBtn.addEventListener('click', () => {
            resetToUpload();
        });
    }
    
    // Retry button
    if (elements.retryBtn) {
        elements.retryBtn.addEventListener('click', () => {
            if (currentFile) {
                processImage(currentFile);
            } else {
                resetToUpload();
            }
        });
    }
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && isValidImageFile(file)) {
        processImage(file);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    elements.uploadArea?.classList.add('drag-over');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    event.preventDefault();
    elements.uploadArea?.classList.remove('drag-over');
}

/**
 * Handle file drop event
 */
function handleFileDrop(event) {
    event.preventDefault();
    elements.uploadArea?.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isValidImageFile(file)) {
            processImage(file);
        } else {
            showError('Please select a valid image file (JPG, PNG, WebP)');
        }
    }
}

/**
 * Check if file is a valid image
 */
function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    return validTypes.includes(file.type);
}

/**
 * Process an image file
 */
async function processImage(file) {
    if (!api || !api.isInitialized()) {
        showError('API not initialized');
        return;
    }
    
    currentFile = file;
    
    try {
        // Show processing section
        showSection('processing');
        updateProgress(0, 'Loading image...');
        
        // Record processing start time
        processingStats.startTime = performance.now();
        processingStats.imageSize = `${Math.round(file.size / 1024)} KB`;
        
        // Load and display original image
        const originalImage = await loadImageFromFile(file);
        displayOriginalImage(originalImage);
        
        updateProgress(25, 'Preparing image data...');
        
        // Process with background removal
        const resultCanvas = await api.removeBackgroundFromFile(file, {
            onProgress: (progress) => {
                updateProgress(25 + (progress.progress * 0.7), progress.message);
            }
        });
        
        // Record processing end time
        processingStats.endTime = performance.now();
        
        updateProgress(100, 'Processing complete!');
        
        // Display results
        displayResults(originalImage, resultCanvas);
        
    } catch (error) {
        console.error('❌ Processing failed:', error);
        showError(`Processing failed: ${error.message}`);
    }
}

/**
 * Load image from file as HTMLImageElement
 */
function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            URL.revokeObjectURL(image.src);
            resolve(image);
        };
        image.onerror = () => {
            URL.revokeObjectURL(image.src);
            reject(new Error('Failed to load image'));
        };
        image.src = URL.createObjectURL(file);
    });
}

/**
 * Display original image on canvas
 */
function displayOriginalImage(image) {
    if (!elements.originalCanvas) return;
    
    const canvas = elements.originalCanvas;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match image
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    
    // Draw image
    ctx.drawImage(image, 0, 0);
    
    // Update image info
    if (elements.originalInfo) {
        elements.originalInfo.textContent = `${image.naturalWidth} × ${image.naturalHeight} pixels`;
    }
}

/**
 * Display processing results
 */
function displayResults(originalImage, resultCanvas) {
    // Copy result to result canvas
    if (elements.resultCanvas) {
        const canvas = elements.resultCanvas;
        const ctx = canvas.getContext('2d');
        
        canvas.width = resultCanvas.width;
        canvas.height = resultCanvas.height;
        
        ctx.drawImage(resultCanvas, 0, 0);
    }
    
    // Update processing stats
    const processingTime = processingStats.endTime - processingStats.startTime;
    
    if (elements.modelName) {
        elements.modelName.textContent = 'ISNet (embedded)';
    }
    
    if (elements.processingTime) {
        elements.processingTime.textContent = `${Math.round(processingTime)}ms`;
    }
    
    if (elements.imageSize) {
        elements.imageSize.textContent = processingStats.imageSize;
    }
    
    // Show results section
    showSection('results');
}

/**
 * Update processing progress
 */
function updateProgress(percentage, message) {
    if (elements.progressFill) {
        elements.progressFill.style.width = `${percentage}%`;
    }
    
    if (elements.progressText) {
        elements.progressText.textContent = message;
    }
}

/**
 * Download result image
 */
async function downloadResult(format) {
    if (!elements.resultCanvas) return;
    
    try {
        const filename = `background-removed-${Date.now()}.${format}`;
        await api.downloadCanvas(elements.resultCanvas, filename, format);
        
        console.log(`✅ Downloaded: ${filename}`);
        
    } catch (error) {
        console.error('❌ Download failed:', error);
        showError(`Download failed: ${error.message}`);
    }
}

/**
 * Show a specific section and hide others
 */
function showSection(sectionName) {
    // Hide all sections
    elements.initStatus?.parentElement?.style.setProperty('display', 'none');
    elements.uploadSection?.style.setProperty('display', 'none');
    elements.processingSection?.style.setProperty('display', 'none');
    elements.resultsSection?.style.setProperty('display', 'none');
    elements.errorSection?.style.setProperty('display', 'none');
    
    // Show requested section
    switch (sectionName) {
        case 'upload':
            elements.uploadSection?.style.setProperty('display', 'block');
            break;
        case 'processing':
            elements.processingSection?.style.setProperty('display', 'block');
            break;
        case 'results':
            elements.resultsSection?.style.setProperty('display', 'block');
            break;
        case 'error':
            elements.errorSection?.style.setProperty('display', 'block');
            break;
    }
}

/**
 * Show error message
 */
function showError(message) {
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
    }
    
    showSection('error');
}

/**
 * Reset to upload state
 */
function resetToUpload() {
    currentFile = null;
    
    // Clear file input
    if (elements.fileInput) {
        elements.fileInput.value = '';
    }
    
    // Reset progress
    updateProgress(0, '');
    
    // Show upload section
    showSection('upload');
}

// Initialize the application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Export for debugging
window.bgRemoveDemo = {
    api,
    resetToUpload,
    processingStats
};