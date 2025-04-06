import logging
import os
import numpy as np
import cv2
from typing import Optional

from inpainting.base import BaseInpainting, OpenCVInpainting

logger = logging.getLogger(__name__)

class StableDiffusionInpainting(BaseInpainting):
    """
    Simplified inpainting model that would normally use Stable Diffusion.
    Currently using OpenCV inpainting with enhanced processing for development.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize a simulated Stable Diffusion inpainting model.
        
        Args:
            model_path: Path to model weights (not used in this implementation)
            device: Device to use for inference (not used in this implementation)
        """
        # Initialize with base parameters
        super().__init__(model_path, device)
        
        # Use OpenCV inpainting as base
        self.cv_inpainter = OpenCVInpainting(method="ns")
        
        # Set model as ready (since we're using a simplified version)
        self.ready = True
        logger.info("Initialized simplified Stable Diffusion inpainting (OpenCV-based)")
    
    def _preprocess(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        """
        Preprocess image and mask for enhanced inpainting.
        
        Args:
            image: Image to inpaint (HxWxC, BGR format)
            mask: Binary mask with 255 for areas to inpaint (HxW)
            
        Returns:
            Preprocessed image and mask
        """
        from PIL import Image
        
        # Ensure mask is binary (0 or 1)
        mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        # Apply some dilation to expand mask slightly
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask_binary, kernel, iterations=1)
        
        # Return processed versions
        return image, dilated_mask
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the masked regions using enhanced OpenCV inpainting.
        
        Args:
            image: Image to inpaint (HxWxC, BGR format)
            mask: Binary mask with 255 for areas to inpaint (HxW)
            
        Returns:
            Inpainted image (HxWxC, BGR format)
        """
        try:
            # Preprocess image and mask
            processed_image, processed_mask = self._preprocess(image, mask)
            
            # First, use standard OpenCV inpainting
            result = self.cv_inpainter.inpaint(processed_image, processed_mask)
            
            # Apply some post-processing to improve results
            # Blend original with inpainted result using the mask
            mask_3ch = np.stack([processed_mask / 255.0] * 3, axis=2)
            blended = image * (1 - mask_3ch) + result * mask_3ch
            
            # Apply slight blur to the inpainted regions for better blending
            kernel_size = 3
            blurred_result = cv2.GaussianBlur(blended, (kernel_size, kernel_size), 0)
            final_result = image * (1 - mask_3ch) + blurred_result * mask_3ch
            
            logger.info("Completed enhanced inpainting")
            return final_result.astype(np.uint8)
                
        except Exception as e:
            logger.error(f"Error during enhanced inpainting: {str(e)}")
            logger.warning("Falling back to basic OpenCV inpainting")
            return self.cv_inpainter.inpaint(image, mask)
    
    @classmethod
    def is_available(cls) -> bool:
        """This simplified version is always available"""
        return True
    
    @classmethod
    def get_name(cls) -> str:
        return "Stable Diffusion Inpainting (Demo)"
