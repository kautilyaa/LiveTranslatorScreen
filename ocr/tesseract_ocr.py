import os
import sys
import logging
import numpy as np
import cv2
import random
from typing import List, Dict, Any, Optional

from ocr.base import BaseOcr, OcrResult, simplify_lang_code, lang_code_to_name

logger = logging.getLogger(__name__)

# Define platform-specific paths (kept for reference)
TESSERACT_PATHS = {
    'darwin': '/opt/homebrew/bin/tesseract',  # macOS
    'linux': '/usr/bin/tesseract',            # Linux
    'win32': r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
}

class TesseractOcr(BaseOcr):
    """
    Simplified OCR provider that simulates Tesseract for development.
    """
    
    # Default language to use
    DEFAULT_LANGUAGE = "eng"
    
    # Reference to platform-specific path (not used in this implementation)
    TESSERACT_PATH = TESSERACT_PATHS.get(sys.platform)
    
    # Dictionary of supported languages
    SUPPORTED_LANGUAGES = [
        "eng", "jpn", "deu", "fra", "spa", "ita", "por", "chi_sim", "chi_tra", 
        "ara", "hin", "kor", "rus", "vie", "nld", "swe", "tur"
    ]
    
    # Sample texts for demo mode
    DEMO_TEXTS = {
        "eng": ["Sample text", "Hello world", "Testing OCR", "Document processing"],
        "jpn": ["サンプルテキスト", "こんにちは世界", "OCRのテスト", "文書処理"],
        "deu": ["Beispieltext", "Hallo Welt", "OCR-Test", "Dokumentenverarbeitung"],
        "fra": ["Exemple de texte", "Bonjour le monde", "Test d'OCR", "Traitement de documents"],
    }
    
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        """
        Initialize simplified Tesseract OCR provider.
        
        Args:
            language: Language code for OCR
        """
        super().__init__(language)
        
        # Map language code
        self.lang_code = self._map_language_code(language)
        logger.info(f"Initialized simplified Tesseract OCR with language: {self.lang_code}")
    
    def _map_language_code(self, language: str) -> str:
        """Map language code to standard format."""
        # Simple mapping for common language codes
        mapping = {
            "en": "eng",
            "ja": "jpn",
            "de": "deu",
            "fr": "fra",
            "es": "spa"
        }
        return mapping.get(language, language)
    
    def _text_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions using MSER feature detector.
        
        Args:
            image: Input image
            
        Returns:
            List of text regions with bounding boxes
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply preprocessing
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use MSER feature detector to find text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(blur)
        
        # Filter and merge regions
        boxes = []
        for region in regions:
            # Get bounding box for region
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter out noise
            if w < 10 or h < 10 or w / h > 10 or h / w > 10:
                continue
                
            # Add to boxes with random confidence
            confidence = random.uniform(0.75, 0.95)
            boxes.append({
                "bounding_box": [x, y, w, h],
                "confidence": confidence
            })
            
        # Merge overlapping boxes
        merged_boxes = []
        while boxes:
            base = boxes.pop(0)
            
            x1, y1, w1, h1 = base["bounding_box"]
            merged = False
            
            i = 0
            while i < len(boxes):
                x2, y2, w2, h2 = boxes[i]["bounding_box"]
                
                # Check if boxes overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                if overlap_x > 0 and overlap_y > 0:
                    # Merge boxes
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    
                    # Update base box
                    base["bounding_box"] = [x, y, w, h]
                    base["confidence"] = max(base["confidence"], boxes[i]["confidence"])
                    
                    # Remove the merged box
                    boxes.pop(i)
                    merged = True
                else:
                    i += 1
            
            if not merged:
                merged_boxes.append(base)
        
        return merged_boxes
    
    def process_batch(self, batch: List[np.ndarray]) -> List[OcrResult]:
        """
        Process a batch of images using simplified text detection.
        
        Args:
            batch: List of numpy arrays containing images
            
        Returns:
            List of OcrResult objects
        """
        results = []
        
        for image in batch:
            try:
                # Detect text regions
                regions = self._text_detection(image)
                
                # Get demo texts for the current language
                demo_texts = self.DEMO_TEXTS.get(
                    self.lang_code, 
                    self.DEMO_TEXTS.get("eng")
                )
                
                # Process regions
                boxes = []
                full_text = []
                
                for i, region in enumerate(regions):
                    # Pick demo text
                    text = demo_texts[i % len(demo_texts)]
                    full_text.append(text)
                    
                    # Get bounding box
                    x, y, w, h = region["bounding_box"]
                    confidence = region["confidence"]
                    
                    boxes.append({
                        "text": text,
                        "bounding_box": [x, y, w, h],
                        "confidence": confidence
                    })
                
                # Create result
                result = OcrResult(
                    text="\n".join(full_text),
                    language=simplify_lang_code(self.language),
                    bounding_boxes=boxes
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in simplified Tesseract OCR: {str(e)}")
                # Add empty result on error
                results.append(OcrResult("", self.language))
        
        return results
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """
        Get list of languages supported by this OCR provider.
        
        Returns:
            List of language codes
        """
        return cls.SUPPORTED_LANGUAGES
    
    @classmethod
    def is_available(cls) -> bool:
        """
        This simplified version is always available.
        
        Returns:
            True
        """
        return True
    
    @classmethod
    def get_name(cls) -> str:
        return "Tesseract OCR (Demo)"
