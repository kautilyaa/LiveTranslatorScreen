import logging
import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class SegmentationModel:
    """
    Text segmentation model for more precise manga text masking.
    Supports YOLO-based segmentation models with OpenCV fallback.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.5):
        """
        Initialize segmentation model.
        
        Args:
            model_path: Path to segmentation model weights
            confidence: Confidence threshold for segmentation
        """
        self.confidence = confidence
        self.model_path = model_path
        self.use_yolo = False
        self.model = None
        
        # Try to load YOLO if model_path is provided and dependencies are available
        if model_path and os.path.exists(model_path):
            try:
                # Dynamically import torch and YOLO to avoid hard dependency
                import torch
                from models.yolo_loader import load_yolo_model
                
                # Determine device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Load the model
                self.model = load_yolo_model(model_path, device)
                if self.model:
                    self.use_yolo = True
                    logger.info(f"Initialized YOLO segmentation model using: {model_path}")
                else:
                    logger.warning(f"Failed to load YOLO segmentation model from {model_path}")
            except ImportError:
                logger.info("PyTorch or YOLO not available, using OpenCV-based segmentation")
            except Exception as e:
                logger.error(f"Error loading YOLO segmentation model: {str(e)}")
        
        if not self.use_yolo:
            logger.info("Initializing fallback segmentation (OpenCV-based)")
    
    def segment(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate precise segmentation mask for text regions.
        
        Args:
            image: Original image
            regions: List of detected regions from TextDetector
            
        Returns:
            Binary mask with text regions segmented precisely
        """
        if self.use_yolo and self.model:
            try:
                return self._segment_with_yolo(image, regions)
            except Exception as e:
                logger.error(f"YOLO segmentation failed: {str(e)}, falling back to OpenCV")
                return self._segment_with_opencv(image, regions)
        else:
            return self._segment_with_opencv(image, regions)
    
    def _segment_with_yolo(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate precise segmentation mask using YOLO.
        
        Args:
            image: Original image
            regions: List of detected regions
            
        Returns:
            Binary mask with text regions segmented precisely
        """
        # Create empty mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Run YOLO segmentation model
        results = self.model(image, size=640)
        
        # Check if the model produced segmentation masks
        if hasattr(results, 'masks') and results.masks is not None:
            # Get segmentation masks
            for i, m in enumerate(results.masks.data):
                # Check confidence
                if results.boxes.conf[i] < self.confidence:
                    continue
                    
                # Convert mask to numpy and resize to original image dimensions
                seg_mask = m.cpu().numpy()
                seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]))
                
                # Add to final mask
                mask = np.logical_or(mask, seg_mask > 0.5).astype(np.uint8) * 255
        else:
            # If the model doesn't produce masks, use the detection boxes
            for i, box in enumerate(results.boxes.xyxy):
                # Check confidence
                if results.boxes.conf[i] < self.confidence:
                    continue
                
                # Get box coordinates
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                
                # Fill the region in the mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        logger.info(f"Generated text segmentation mask using YOLO")
        return mask
    
    def _segment_with_opencv(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate segmentation mask using OpenCV techniques.
        
        Args:
            image: Original image
            regions: List of detected regions
            
        Returns:
            Binary mask with text regions
        """
        # Create empty mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Process each detected region
        for region in regions:
            bbox = region["bounding_box"]
            x, y, w, h = bbox
            
            # Extract ROI
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:  # Skip empty regions
                continue
                
            # Apply adaptive threshold to find text
            try:
                # Try adaptive threshold
                binary = cv2.adaptiveThreshold(
                    roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Clean up noise
                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Find contours within the region
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Filter contours by area
                min_area = 10
                for contour in contours:
                    if cv2.contourArea(contour) > min_area:
                        # Shift contour coordinates to global image space
                        shifted_contour = contour + np.array([x, y])
                        # Draw filled contour on mask
                        cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
            except Exception as e:
                logger.warning(f"Error processing region {bbox}: {str(e)}")
                # Fallback: just use the bounding box
                mask[y:y+h, x:x+w] = 255
        
        # Final cleanup - remove very small isolated components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 20:  # Minimum area threshold
                mask[labels == i] = 0
                
        logger.info(f"Generated text segmentation mask using OpenCV")
        return mask