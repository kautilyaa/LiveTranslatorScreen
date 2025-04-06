# Manga Translator API

A modular Flask-based API for manga/text processing with multiple inpainting models, OCR providers (local and cloud-based), and translation options.

## Features

- **Multiple OCR Providers**: Support for Tesseract, EasyOCR, Azure Computer Vision, and AWS Textract
- **Advanced Inpainting**: LaMa and Stable Diffusion-based inpainting for text removal
- **Comprehensive Translation Options**: 
  - OpenAI LLM-based translation (GPT-3.5/GPT-4)
  - Google Translate API integration
  - DeepL API integration
  - Local Ollama LLM integration
  - Hugging Face models fallback
- **YOLO Integration**: Compatible with custom YOLO models for text detection
- **Segmentation**: Precise mask generation using segmentation models
- **Asynchronous Processing**: Parallel processing for better performance
- **RESTful API**: Simple JSON API for all functionality

## Getting Started

### Prerequisites

- Python 3.11+
- OpenCV
- PyTorch (for YOLO models)
- Flask and Gunicorn

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/manga-translator-api.git
   cd manga-translator-api
   ```

2. Install dependencies using pip:
   ```bash
   pip install -e .
   ```
   This will install all dependencies defined in the pyproject.toml file.

3. Configure environment variables for translation services (optional):
   ```bash
   # OpenAI API for LLM translation
   export OPENAI_API_KEY=your_openai_api_key
   
   # Google Cloud Translation (optional)
   export GOOGLE_API_KEY=your_google_api_key
   export GOOGLE_PROJECT_ID=your_google_project_id
   
   # DeepL Translation (optional)
   export DEEPL_API_KEY=your_deepl_api_key
   
   # Ollama (optional, for local LLM translation)
   export OLLAMA_API_HOST=http://localhost:11434  # Default Ollama endpoint
   ```

4. Run the application:
   ```bash
   # Development mode with auto-reload
   python main.py
   
   # Production mode with Gunicorn
   gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
   ```

### Docker Deployment

Build and run using Docker:

```bash
# Build the Docker image
docker build -t manga-translator .

# Run the container with environment variables
docker run -p 5000:5000 \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e GOOGLE_API_KEY=your_google_api_key \
  -e GOOGLE_PROJECT_ID=your_google_project_id \
  -e DEEPL_API_KEY=your_deepl_api_key \
  manga-translator
```

The Docker image is based on Python 3.11-slim and includes all necessary system dependencies for OpenCV and other components.

### Requirements

The application depends on the following Python packages:

- **Web Framework**: Flask, Gunicorn
- **Image Processing**: OpenCV, Pillow, NumPy
- **Machine Learning**: PyTorch
- **OCR**: EasyOCR (simplified implementation)
- **Translation**: OpenAI, Google Cloud Translate, DeepL
- **Utilities**: Requests, aiohttp, PyYAML
- **Database**: Flask-SQLAlchemy, psycopg2-binary (for PostgreSQL support)

All dependencies are specified in the pyproject.toml file. The application is designed to work even without certain API keys, falling back to simpler models when needed.

## API Endpoints

### Process Image
```
POST /api/process
```
Process a manga image with text detection, OCR, inpainting, and translation.

#### Request Body
```json
{
    "image": "base64_encoded_image",
    "ocr_provider": "tesseract|easyocr|azure|aws",
    "inpainting_model": "lama|stable_diffusion",
    "translator": "huggingface|llm|ollama|google|deepl",
    "source_language": "ja|ko|en|zh|...",
    "target_language": "en|ja|ko|zh|fr|de|es|...",
    "detection_model": "path/to/yolo_model.pt",
    "segmentation_model": "path/to/segmentation_model.pt"
}
```

#### Response
```json
{
    "processed_image": "base64_encoded_processed_image",
    "detected_texts": [
        {
            "text": "original_text",
            "translated_text": "translated_text",
            "bounding_box": [x, y, width, height],
            "confidence": 0.95
        }
    ],
    "processing_time": {
        "detection": 0.5,
        "segmentation": 0.3,
        "ocr": 0.3,
        "inpainting": 0.8,
        "translation": 0.4,
        "total": 2.0
    }
}
```

### OCR
```
POST /api/ocr
```
Perform OCR on an image with specified provider.

### Translate
```
POST /api/translate
```
Translate text using specified provider.

### Inpaint
```
POST /api/inpaint
```
Perform inpainting on an image with specified model.

### Detect
```
POST /api/detect
```
Detect text regions in an image.

### Segment
```
POST /api/segment
```
Generate precise segmentation mask for text regions.

### Providers
```
GET /api/providers
```
Get all available providers for OCR, translation, and inpainting.

## Configuration

Edit `core/config.py` to change default settings.

## External Models

To use custom YOLO models for detection or segmentation, provide the model paths in the API requests or configure them in `core/config.py`.

## Cloud Services

### OCR Services

To use cloud-based OCR providers:

1. For Azure Computer Vision:
   - Set `AZURE_VISION_KEY` and `AZURE_VISION_ENDPOINT` environment variables

2. For AWS Textract:
   - Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables

### Translation Services

The system supports multiple translation services with proper fallback mechanisms:

1. **OpenAI LLM Translation**:
   - Set `OPENAI_API_KEY` environment variable
   - Configure the model name in `translation/llm_translator.py` (default: `gpt-3.5-turbo`)

2. **Google Cloud Translation**:
   - Set `GOOGLE_API_KEY` and `GOOGLE_PROJECT_ID` environment variables
   - Requires Google Cloud project with Translation API enabled

3. **DeepL Translation**:
   - Set `DEEPL_API_KEY` environment variable
   - Provides high-quality translations for supported language pairs

4. **Ollama (Local LLM)**:
   - Set `OLLAMA_API_HOST` environment variable (default: `http://localhost:11434`)
   - Requires Ollama server running locally or in another container

5. **Hugging Face (Fallback)**:
   - Always available without API keys
   - Uses simplified translations for demonstration purposes

## License

This project is licensed under the MIT License - see the LICENSE file for details.