import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseInpainting:
    """
    Base class for inpainting models.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize inpainting model.
        
        Args:
            model_path: Path to model weights, if None uses default
            device: Device to use for inference ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.device = device
        
        # Check if GPU is available if device is cuda
        if device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the masked regions in the image.
        
        Args:
            image: Image to inpaint (HxWxC, BGR format)
            mask: Binary mask with 255 for areas to inpaint (HxW)
            
        Returns:
            Inpainted image (HxWxC, BGR format)
        """
        raise NotImplementedError("Inpainting models must implement inpaint method")
    
    @classmethod
    def get_name(cls) -> str:
        """Get name of inpainting model"""
        return cls.__name__
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if the inpainting model is available"""
        return True

class OpenCVInpainting(BaseInpainting):
    """
    Simple inpainting using OpenCV algorithms.
    Useful as a fallback when other models fail.
    """
    
    def __init__(self, method: str = "ns", device: str = "cpu"):
        """
        Initialize OpenCV inpainting.
        
        Args:
            method: Inpainting method ("ns" for INPAINT_NS or "telea" for INPAINT_TELEA)
            device: Ignored (OpenCV doesn't use GPU by default)
        """
        super().__init__(None, "cpu")
        self.method = method.lower()
        
        # Import OpenCV
        import cv2
        self.cv2 = cv2
        
        # Set inpainting method
        if self.method == "ns":
            self.inpaint_method = cv2.INPAINT_NS
        elif self.method == "telea":
            self.inpaint_method = cv2.INPAINT_TELEA
        else:
            logger.warning(f"Unknown inpainting method '{method}', using INPAINT_NS")
            self.inpaint_method = cv2.INPAINT_NS
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the masked regions using OpenCV.
        
        Args:
            image: Image to inpaint (HxWxC, BGR format)
            mask: Binary mask with 255 for areas to inpaint (HxW)
            
        Returns:
            Inpainted image (HxWxC, BGR format)
        """
        # Ensure mask is of the right type
        mask = mask.astype(np.uint8)
        
        # Apply inpainting
        inpainted = self.cv2.inpaint(image, mask, 3, self.inpaint_method)
        
        return inpainted
    
    @classmethod
    def get_name(cls) -> str:
        return "OpenCV Inpainting"
