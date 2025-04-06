import numpy as np
from typing import List, Dict, Optional, Any

class OcrResult:
    """Class to store OCR result data"""
    def __init__(self, text: str = "", language: str = "en", 
                 bounding_boxes: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize OCR result.
        
        Args:
            text: Detected text
            language: Detected text language
            bounding_boxes: Optional list of bounding boxes for multi-region results
        """
        self.text = text
        self.language = language
        self.bounding_boxes = bounding_boxes or []

class BaseOcr:
    """Base class for OCR providers"""
    
    def __init__(self, language: str = "en"):
        """
        Initialize OCR provider.
        
        Args:
            language: Language code for OCR
        """
        self.language = language
    
    def __call__(self, batch: List[np.ndarray]) -> List[OcrResult]:
        """
        Process a batch of images for OCR.
        
        Args:
            batch: List of numpy arrays containing images
            
        Returns:
            List of OcrResult objects
        """
        return self.process_batch(batch)
    
    def process_batch(self, batch: List[np.ndarray]) -> List[OcrResult]:
        """
        Process a batch of images.
        
        Args:
            batch: List of numpy arrays containing images
            
        Returns:
            List of OcrResult objects
        """
        raise NotImplementedError("OCR providers must implement process_batch method")
    
    @classmethod
    def get_name(cls) -> str:
        """Get name of OCR provider"""
        return cls.__name__
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages"""
        return ["en"]  # Default supports English only
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if the OCR provider is available on the system"""
        return True

class NoOcr(BaseOcr):
    """OCR provider that returns empty results, useful for testing"""
    
    def process_batch(self, batch: List[np.ndarray]) -> List[OcrResult]:
        """Return empty results for all images"""
        return [OcrResult("", "") for _ in batch]
    
    @classmethod
    def get_name(cls) -> str:
        return "No OCR"

# Helper functions
def simplify_lang_code(lang_code: str) -> str:
    """
    Simplify complex language codes to base language.
    E.g., 'en_US' -> 'en', 'jpn_vert' -> 'ja'
    
    Args:
        lang_code: Language code to simplify
        
    Returns:
        Simplified language code
    """
    mapping = {
        "jpn": "ja", 
        "jpn_vert": "ja",
        "eng": "en",
        "chi_sim": "zh-cn",
        "chi_tra": "zh-tw"
    }
    
    # Use mapping if available
    if lang_code in mapping:
        return mapping[lang_code]
    
    # Otherwise, take first part before underscore or dash
    for separator in ['_', '-']:
        if separator in lang_code:
            return lang_code.split(separator)[0]
    
    return lang_code

def lang_code_to_name(lang_code: str) -> str:
    """
    Convert language code to readable name.
    
    Args:
        lang_code: Language code
        
    Returns:
        Human-readable language name
    """
    language_names = {
        "en": "English",
        "ja": "Japanese",
        "jpn": "Japanese",
        "jpn_vert": "Japanese (Vertical)",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "chi_sim": "Chinese (Simplified)",
        "chi_tra": "Chinese (Traditional)",
        "ar": "Arabic",
        "hi": "Hindi",
        "th": "Thai",
        "vi": "Vietnamese"
    }
    
    simplified = simplify_lang_code(lang_code)
    return language_names.get(simplified, lang_code)
