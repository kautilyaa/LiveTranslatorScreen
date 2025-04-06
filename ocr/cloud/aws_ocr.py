import os
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import random

from ocr.base import BaseOcr, OcrResult, simplify_lang_code

logger = logging.getLogger(__name__)

class AwsOcr(BaseOcr):
    """
    Simplified OCR provider that simulates AWS Textract for development.
    """
    
    # Sample business document texts
    SAMPLE_TEXTS = [
        "INVOICE", "Receipt", "Order #12345", "Total Amount", "Customer ID",
        "Date", "Payment Due", "Subtotal", "Tax", "Shipping",
        "Purchase Order", "Contract", "Agreement", "Terms and Conditions",
        "Authorized Signature", "Form W-9", "CONFIDENTIAL", "Page 1 of 2",
        "Contact Information", "Reference Number"
    ]
    
    def __init__(self, language: str = "auto"):
        """
        Initialize simplified AWS Textract OCR provider.
        
        Note: AWS Textract does not support language selection directly,
        but we keep the language parameter for consistency with other providers.
        
        Args:
            language: Language code (ignored for AWS)
        """
        super().__init__(language)
        logger.info("Initialized simplified AWS Textract OCR simulator")
    
    def _detect_document_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions using a simplified method that mimics AWS Textract's
        document detection behavior.
        
        Args:
            image: Input image
            
        Returns:
            List of detected regions with text
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find connected components
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        regions = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter small regions
            if area < 200 or w < 20 or h < 10:
                continue
            
            # Filter extreme aspect ratios
            aspect_ratio = w / h
            if aspect_ratio > 20 or aspect_ratio < 0.05:
                continue
                
            # AWS Textract is good at detecting form fields, table cells, etc.
            # Pick a random text from samples
            text = random.choice(self.SAMPLE_TEXTS)
            confidence = random.uniform(0.85, 0.99)
            
            regions.append({
                "text": text,
                "bounding_box": [x, y, w, h],
                "confidence": confidence
            })
            
        return regions
    
    def process_batch(self, batch: List[np.ndarray]) -> List[OcrResult]:
        """
        Process a batch of images using simplified document text detection.
        
        Args:
            batch: List of numpy arrays containing images
            
        Returns:
            List of OcrResult objects
        """
        results = []
        
        for image in batch:
            try:
                # Detect document regions
                regions = self._detect_document_regions(image)
                
                # Process results
                text_blocks = []
                boxes = []
                
                for region in regions:
                    text_blocks.append(region["text"])
                    boxes.append(region)
                
                # Create result
                result = OcrResult(
                    text="\n".join(text_blocks),
                    language="en",  # Textract doesn't specify language
                    bounding_boxes=boxes
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in simplified AWS Textract: {str(e)}")
                # Add empty result on error
                results.append(OcrResult("", "en"))
        
        return results
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """
        Get list of languages supported by AWS Textract.
        
        Returns:
            List of language codes
        """
        # Textract primarily supports English, but can work with other languages
        # Note: Language selection is not a parameter in Textract API
        return ["en", "auto"]
    
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
        return "AWS Textract (Demo)"
