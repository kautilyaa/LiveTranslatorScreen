# LiveTranslatorScreen

A Python application for real-time manga page translation. The current phase includes an end-to-end pipeline that integrates OCR, translation, and image overlay functionality.

## Features

- **Full Conversion Pipeline:**  
  Combines OCR, translation, and image drawing into a single streamlined process.
  
- **Modular Architecture:**  
  Easily swap or configure the translator, OCR, and drawing components by selecting from available modules and providing JSON-formatted arguments.

- **Batch Processing:**  
  Process single images or batches of images with simple function calls.

- **Image Preprocessing:**  
  Includes utilities for slicing images, resizing with padding, and annotation conversions (e.g., COCO to YOLO formats).

- **GPU Support:**  
  Optionally leverage CUDA (via PyTorch) to accelerate computations if available.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/LiveTranslatorScreen.git
   cd LiveTranslatorScreen
   ```

2. **Create a Virtual Environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Tesseract OCR Setup:**
   - Install Tesseract OCR on your system.
   - Ensure that `pytesseract.pytesseract.tesseract_cmd` points to the correct Tesseract executable.

## Usage

The conversion pipeline is managed via the `do_convert` function. This function accepts a list of image file paths along with configuration arguments for the translator, OCR, and drawing modules.

### Example
```python
from conversion_module import do_convert  # Adjust the import path as needed

data = do_convert(
    files=["/path/to/Sample/Naruto_jp.png"],
    translator=1,
    translator_args="{}",  # JSON configuration for the translator
    ocr=2,
    ocr_args="{}",         # JSON configuration for OCR
    drawer=0,
    drawer_args="{}"       # JSON configuration for the drawer
)

# 'data' is a dictionary mapping converted filenames to their processed images.
print(data)
```

## Project Structure

```
LiveTranslatorScreen/
├── LiveTranslatorScreen.ipynb         # Main notebook for live translation
├── conversion_module.py               # Contains the conversion pipeline logic
├── image_utils.py                     # Image slicing, resizing, and preprocessing utilities
├── annotation_utils.py                # Annotation conversion utilities, e.g., COCO to YOLO
├── requirements.txt                   # List of dependencies
├── README.md                          # This file
└── ...
```

## Configuration

- **Module Arguments:**  
  The translator, OCR, and drawer modules are configurable via JSON strings passed as arguments. For example, if your translator module requires a model path or language setting, include those in the JSON:
  ```json
  translator_args='{"model_path": "/path/to/model", "src_lang": "ja", "tgt_lang": "en"}'
  ```

- **Pipeline Adjustments:**  
  Customize the pipeline by modifying the module selections in `get_translators()`, `get_ocr()`, and `get_drawers()`.

## Contributing

Contributions are welcome

