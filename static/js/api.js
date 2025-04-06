/**
 * Manga Translator API - Frontend JavaScript
 * Handles UI interactions and API requests for manga translation
 */

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const processForm = document.getElementById('process-form');
    const imageUpload = document.getElementById('image-upload');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error-message');
    const originalImageEl = document.getElementById('original-image');
    const processedImageEl = document.getElementById('processed-image');
    const textResultsEl = document.getElementById('text-results');
    const processingTimesEl = document.getElementById('processing-times');
    
    // Provider dropdowns
    const ocrProviderSelect = document.getElementById('ocr-provider');
    const inpaintingModelSelect = document.getElementById('inpainting-model');
    const translatorSelect = document.getElementById('translator');
    const sourceLanguageSelect = document.getElementById('source-language');
    const targetLanguageSelect = document.getElementById('target-language');
    
    // Get available providers on page load
    fetchProviders();
    
    // Form submission
    if (processForm) {
        processForm.addEventListener('submit', function(e) {
            e.preventDefault();
            processImage();
        });
    }
    
    /**
     * Fetch available providers for OCR, inpainting, and translation
     */
    function fetchProviders() {
        fetch('/api/providers')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch providers');
                }
                return response.json();
            })
            .then(data => {
                // Populate OCR providers
                if (data.ocr && data.ocr.length > 0) {
                    populateSelect(ocrProviderSelect, data.ocr);
                }
                
                // Populate inpainting models
                if (data.inpainting && data.inpainting.length > 0) {
                    populateSelect(inpaintingModelSelect, data.inpainting);
                }
                
                // Populate translators
                if (data.translation && data.translation.length > 0) {
                    populateSelect(translatorSelect, data.translation);
                }
            })
            .catch(error => {
                console.error('Error fetching providers:', error);
                showError('Failed to fetch available providers. Please refresh the page.');
            });
    }
    
    /**
     * Populate a select dropdown with options
     */
    function populateSelect(selectElement, options) {
        // Clear existing options
        selectElement.innerHTML = '';
        
        // Add new options
        options.forEach(option => {
            const optionEl = document.createElement('option');
            optionEl.value = option;
            optionEl.textContent = formatProviderName(option);
            selectElement.appendChild(optionEl);
        });
    }
    
    /**
     * Format provider name for display
     */
    function formatProviderName(name) {
        // Convert snake_case or kebab-case to Title Case
        return name
            .replace(/[-_]/g, ' ')
            .replace(/\w\S*/g, function(txt) {
                return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
            });
    }
    
    /**
     * Process the uploaded image
     */
    function processImage() {
        const file = imageUpload.files[0];
        if (!file) {
            showError('Please select an image to process');
            return;
        }
        
        // Show loading state
        showLoading(true);
        hideError();
        hideResults();
        
        // Read the file as base64
        const reader = new FileReader();
        reader.onload = function(e) {
            const base64Image = e.target.result.split(',')[1]; // Remove data URL prefix
            
            // Prepare request payload
            const payload = {
                image: base64Image,
                ocr_provider: ocrProviderSelect.value,
                inpainting_model: inpaintingModelSelect.value,
                translator: translatorSelect.value,
                source_language: sourceLanguageSelect.value,
                target_language: targetLanguageSelect.value
            };
            
            // Send API request
            fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Failed to process image');
                    });
                }
                return response.json();
            })
            .then(data => {
                displayResults(data, e.target.result);
            })
            .catch(error => {
                console.error('Error processing image:', error);
                showError(error.message || 'An error occurred while processing the image');
            })
            .finally(() => {
                showLoading(false);
            });
        };
        
        reader.onerror = function() {
            showError('Failed to read the image file');
            showLoading(false);
        };
        
        reader.readAsDataURL(file);
    }
    
    /**
     * Display processing results
     */
    function displayResults(data, originalImageSrc) {
        // Set images
        originalImageEl.src = originalImageSrc;
        processedImageEl.src = 'data:image/jpeg;base64,' + data.processed_image;
        
        // Clear previous text results
        textResultsEl.innerHTML = '';
        
        // Add text results
        if (data.detected_texts && data.detected_texts.length > 0) {
            data.detected_texts.forEach(textItem => {
                const row = document.createElement('tr');
                
                const originalCell = document.createElement('td');
                originalCell.textContent = textItem.text || '(No text)';
                
                const translatedCell = document.createElement('td');
                translatedCell.textContent = textItem.translated_text || '(No translation)';
                
                row.appendChild(originalCell);
                row.appendChild(translatedCell);
                textResultsEl.appendChild(row);
            });
        } else {
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 2;
            cell.textContent = 'No text detected in the image';
            cell.className = 'text-center';
            row.appendChild(cell);
            textResultsEl.appendChild(row);
        }
        
        // Display processing times
        processingTimesEl.innerHTML = '';
        if (data.processing_time) {
            for (const [key, value] of Object.entries(data.processing_time)) {
                const li = document.createElement('li');
                li.textContent = `${formatTimeName(key)}: ${value.toFixed(2)}s`;
                processingTimesEl.appendChild(li);
            }
        }
        
        // Show results
        showResults();
    }
    
    /**
     * Format processing time name for display
     */
    function formatTimeName(name) {
        return name.charAt(0).toUpperCase() + name.slice(1);
    }
    
    /**
     * Show loading indicator
     */
    function showLoading(show) {
        loadingDiv.classList.toggle('d-none', !show);
    }
    
    /**
     * Show results container
     */
    function showResults() {
        resultsDiv.classList.remove('d-none');
    }
    
    /**
     * Hide results container
     */
    function hideResults() {
        resultsDiv.classList.add('d-none');
    }
    
    /**
     * Show error message
     */
    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('d-none');
    }
    
    /**
     * Hide error message
     */
    function hideError() {
        errorDiv.classList.add('d-none');
    }
    
    // Add try-it-now functionality for docs page
    setupApiExamples();
});

/**
 * Set up interactive API examples in the documentation
 */
function setupApiExamples() {
    const tryButtons = document.querySelectorAll('.try-endpoint');
    
    tryButtons.forEach(button => {
        button.addEventListener('click', function() {
            const endpoint = this.dataset.endpoint;
            const method = this.dataset.method;
            const exampleId = this.dataset.example;
            
            const exampleElement = document.getElementById(exampleId);
            if (!exampleElement) return;
            
            let payload;
            try {
                payload = JSON.parse(exampleElement.textContent);
            } catch (error) {
                console.error('Invalid JSON in example:', error);
                return;
            }
            
            // Get the response container
            const responseContainer = document.getElementById(`${exampleId}-response`);
            if (!responseContainer) return;
            
            // Show loading
            responseContainer.innerHTML = '<div class="loading-overlay"><div class="spinner-border text-light" role="status"></div><p class="mt-2">Processing request...</p></div>';
            
            // Make API request
            fetch(`/api/${endpoint}`, {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                },
                body: method === 'GET' ? undefined : JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                // Format and display the response
                const formattedJson = JSON.stringify(data, null, 2);
                responseContainer.innerHTML = `<pre class="code-block">${formattedJson}</pre>`;
            })
            .catch(error => {
                console.error('API request error:', error);
                responseContainer.innerHTML = `<div class="alert alert-danger">Error: ${error.message || 'Failed to send request'}</div>`;
            });
        });
    });
}
