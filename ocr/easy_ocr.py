import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

from ocr.base import BaseOcr, OcrResult, simplify_lang_code

logger = logging.getLogger(__name__)

class EasyOcr(BaseOcr):
    """
    Simplified OCR provider that simulates EasyOCR using OpenCV text detection.
    """
    
    # Default language to use
    DEFAULT_LANGUAGE = "ja"
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        "ja", "ko", "en", "ch_sim", "ch_tra", "fr", "de", "es", 
        "ar", "ru", "pt", "it", "th", "vi"
    ]
    
    # Simple demo texts for common languages
    DEMO_TEXTS = {
        "ja": ["こんにちは", "さようなら", "ありがとう", "マンガ", "テキスト"],
        "en": ["Hello", "Goodbye", "Thank you", "Manga", "Text"],
        "ko": ["안녕하세요", "안녕히 가세요", "감사합니다", "만화", "텍스트"],
        "fr": ["Bonjour", "Au revoir", "Merci", "BD", "Texte"],
    }
    
    def __init__(self, language: str = DEFAULT_LANGUAGE, gpu: bool = False):
        """
        Initialize simplified EasyOCR-like provider.
        
        Args:
            language: Language code for OCR
            gpu: Not used in this implementation
        """
        super().__init__(language)
        self.gpu = False  # Force CPU mode in simplified version
        
        # Map language code if needed
        self.lang_code = self._map_language_code(language)
        logger.info(f"Initialized simplified EasyOCR-like OCR with language: {self.lang_code}")
    
    def _map_language_code(self, language: str) -> str:
        """
        Map language code to standard format.
        
        Args:
            language: Language code to map
            
        Returns:
            Mapped language code
        """
        # Mapping for common language codes
        mapping = {
            "jpn": "ja",
            "jpn_vert": "ja",
            "eng": "en",
            "chi_sim": "ch_sim",
            "chi_tra": "ch_tra",
            "kor": "ko",
            "fra": "fr",
            "deu": "de",
            "spa": "es"
        }
        
        return mapping.get(language, language)
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions using OpenCV's simple contour-based approach.
        
        Args:
            image: Input image
            
        Returns:
            List of detected regions with bounding boxes
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply some preprocessing
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        regions = []
        min_area = 100
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area < min_area or w < 10 or h < 10:
                continue
                
            # Filter based on aspect ratio (avoid extreme rectangles)
            aspect_ratio = w / h
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
                
            # Add region
            regions.append({
                "bounding_box": [x, y, w, h],
                "confidence": min(0.9, area / 10000)  # Simple confidence score
            })
            
        return regions
    
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
                regions = self._detect_text_regions(image)
                
                # Generate placeholder text based on language
                demo_texts = self.DEMO_TEXTS.get(
                    self.lang_code, 
                    self.DEMO_TEXTS.get("en", ["Text"])
                )
                
                # Generate boxes with text
                boxes = []
                full_text = []
                
                for i, region in enumerate(regions):
                    # Pick a demo text based on region index
                    text = demo_texts[i % len(demo_texts)]
                    
                    # Add to full text
                    full_text.append(text)
                    
                    # Get bounding box
                    x, y, w, h = region["bounding_box"]
                    confidence = region["confidence"]
                    
                    boxes.append({
                        "text": text,
                        "bounding_box": [int(x), int(y), int(w), int(h)],
                        "confidence": float(confidence)
                    })
                
                # Create result
                result = OcrResult(
                    text="\n".join(full_text),
                    language=simplify_lang_code(self.language),
                    bounding_boxes=boxes
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image with simplified OCR: {str(e)}")
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
        return "EasyOCR (Demo)"
