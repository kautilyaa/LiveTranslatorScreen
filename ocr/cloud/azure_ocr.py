import os
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import random
import math
from PIL import Image, ImageDraw, ImageFont

from ocr.base import BaseOcr, OcrResult, simplify_lang_code

logger = logging.getLogger(__name__)

class AzureOcr(BaseOcr):
    """
    Simplified OCR provider that simulates Azure Computer Vision for development.
    """
    
    # Supported languages with sample texts
    LANGUAGE_SAMPLES = {
        "en": ["The quick brown fox jumps over the lazy dog",
               "Azure Computer Vision can detect and extract text from images",
               "Optical Character Recognition technology",
               "Reading text from documents and photos"],
               
        "ja": ["素早い茶色のキツネは怠け者の犬を飛び越えます",
               "Azure Computer Visionは画像からテキストを検出および抽出できます",
               "光学式文字認識技術",
               "文書や写真からテキストを読み取る"],
               
        "ko": ["빠른 갈색 여우가 게으른 개를 뛰어 넘습니다",
               "Azure Computer Vision은 이미지에서 텍스트를 감지하고 추출할 수 있습니다",
               "광학 문자 인식 기술",
               "문서 및 사진에서 텍스트 읽기"],
    }
    
    def __init__(self, language: str = "auto"):
        """
        Initialize simplified Azure OCR provider.
        
        Args:
            language: Language code (or "auto" for automatic detection)
        """
        super().__init__(language)
        self.language = self._map_language_code(language)
        logger.info(f"Initialized simplified Azure OCR simulator with language: {self.language}")
    
    def _map_language_code(self, language: str) -> str:
        """Map language code to standard format for our simulator."""
        if language == "auto":
            return "en"  # Default to English for auto
            
        # Simple mapping for Azure format
        mapping = {
            "zh-hans": "zh",
            "zh-hant": "zh",
        }
        return mapping.get(language, language)
    
    def _generate_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate text regions in a way that simulates Azure OCR's text detection.
        
        Args:
            image: Input image
            
        Returns:
            List of detected regions with text
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Generate lines of text based on image content
        num_regions = random.randint(3, 8)  # Generate 3-8 text regions
        regions = []
        
        # Azure OCR is good at finding lines of text and orientation
        # Generate some text lines with realistic placement
        
        # Get demo texts for selected language
        lang_samples = self.LANGUAGE_SAMPLES.get(
            self.language, 
            self.LANGUAGE_SAMPLES.get("en")
        )
        
        # Create some text blocks with different structures
        for i in range(num_regions):
            # Select a text sample
            text = random.choice(lang_samples)
            
            # Calculate a realistic position for this text
            x = random.randint(10, max(11, width - 200))
            y = random.randint(10 + (i * height // (num_regions + 1)), 
                              min(height - 50, 20 + ((i+1) * height // (num_regions + 1))))
            
            # Calculate a realistic width and height
            text_width = min(random.randint(100, 300), width - x - 10)
            text_height = random.randint(20, 40)
            
            # Azure OCR returns precise polygons, but we'll simplify to rectangles
            # with 8 points (4 corners) for compatibility
            x1, y1 = x, y
            x2, y2 = x + text_width, y
            x3, y3 = x + text_width, y + text_height
            x4, y4 = x, y + text_height
            
            bounding_box = [x1, y1, x2, y2, x3, y3, x4, y4]
            
            regions.append({
                "text": text,
                "bounding_box": bounding_box,
                "confidence": random.uniform(0.9, 0.99)
            })
            
        return regions
    
    def process_batch(self, batch: List[np.ndarray]) -> List[OcrResult]:
        """
        Process a batch of images using simplified Azure-like OCR.
        
        Args:
            batch: List of numpy arrays containing images
            
        Returns:
            List of OcrResult objects
        """
        results = []
        
        for image in batch:
            try:
                # Generate text regions
                regions = self._generate_text_regions(image)
                
                # Process results in Azure format
                text_lines = []
                boxes = []
                
                for region in regions:
                    text_lines.append(region["text"])
                    
                    # Convert Azure-style bbox [x1,y1,x2,y2,x3,y3,x4,y4] to [x,y,w,h]
                    points = region["bounding_box"]
                    x_min = min(points[0], points[2], points[4], points[6])
                    y_min = min(points[1], points[3], points[5], points[7])
                    x_max = max(points[0], points[2], points[4], points[6])
                    y_max = max(points[1], points[3], points[5], points[7])
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    boxes.append({
                        "text": region["text"],
                        "bounding_box": [int(x_min), int(y_min), int(w), int(h)],
                        "confidence": region["confidence"]
                    })
                
                # Create result
                result = OcrResult(
                    text="\n".join(text_lines),
                    language=simplify_lang_code(self.language),
                    bounding_boxes=boxes
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in simplified Azure OCR: {str(e)}")
                # Add empty result on error
                results.append(OcrResult("", self.language))
        
        return results
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """
        Get list of languages supported by Azure OCR.
        
        Returns:
            List of language codes
        """
        # Azure supports many languages for OCR
        # See: https://docs.microsoft.com/azure/cognitive-services/computer-vision/language-support
        return [
            "auto", "en", "ja", "ko", "zh-hans", "zh-hant", "de", "fr", "it",
            "es", "pt", "nl", "ar", "hi", "ru", "ta", "vi"
        ]
    
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
        return "Azure OCR (Demo)"
