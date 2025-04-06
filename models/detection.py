import logging
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class TextDetector:
    """
    Text region detector for manga images.
    Supports YOLO models with fallback to OpenCV-based detection.
    """
    
    # Model URLs for different YOLO models
    YOLO_MODELS = {
        "manga_text_detection": {
            "url": None,  # To be provided by user
            "md5": None   # Optional validation hash
        },
        "manga_text_segmentation": {
            "url": None,  # To be provided by user
            "md5": None   # Optional validation hash
        }
    }
    
    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.5):
        """
        Initialize text detector with YOLO when available, fallback to OpenCV.
        
        Args:
            model_path: Path to YOLO model weights file
            confidence: Confidence threshold for detections (0-1)
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
                    logger.info(f"Initialized YOLO text detector using model: {model_path}")
                else:
                    logger.warning(f"Failed to load YOLO model from {model_path}")
            except ImportError:
                logger.info("PyTorch or YOLO not available, using OpenCV-based detector")
            except Exception as e:
                logger.error(f"Error loading YOLO model: {str(e)}")
        
        if not self.use_yolo:
            logger.info("Initializing simplified TextDetector (OpenCV-based)")
        
    def detect(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect text regions in the given image using YOLO or fallback to OpenCV.
        
        Args:
            image: numpy array containing the image (BGR format)
            
        Returns:
            Tuple containing:
            - List of detected regions as dictionaries with bounding_box and confidence
            - Visualization image with bounding boxes drawn
        """
        if self.use_yolo and self.model:
            try:
                return self._detect_with_yolo(image)
            except Exception as e:
                logger.error(f"YOLO detection failed: {str(e)}, falling back to OpenCV")
                return self._detect_with_opencv(image)
        else:
            return self._detect_with_opencv(image)
    
    def _detect_with_yolo(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect text regions using YOLO model.
        
        Args:
            image: numpy array containing the image (BGR format)
            
        Returns:
            Tuple containing detected regions and visualization image
        """
        # Perform inference
        results = self.model(image, size=640)
        
        # Process results
        detections = []
        
        # Extract detections from YOLO output
        predictions = results.xyxy[0].cpu().numpy()  # xmin, ymin, xmax, ymax, confidence, class
        
        for pred in predictions:
            x1, y1, x2, y2, conf, cls_id = pred
            
            # Skip if below confidence threshold
            if conf < self.confidence:
                continue
                
            # Calculate width and height
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            detections.append({
                "bounding_box": [int(x1), int(y1), width, height],
                "confidence": float(conf),
                "class_id": int(cls_id)
            })
        
        # Use YOLO's built-in rendering for visualization
        vis_image = results.render()[0]
        
        logger.info(f"Detected {len(detections)} text regions using YOLO")
        return detections, vis_image
    
    def _detect_with_opencv(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect text regions using OpenCV-based method (fallback).
        
        Args:
            image: numpy array containing the image (BGR format)
            
        Returns:
            Tuple containing detected regions and visualization image
        """
        try:
            # Make a copy for visualization
            vis_image = image.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and process contours
            detections = []
            for contour in contours:
                # Get bounding rectangle
                x, y, width, height = cv2.boundingRect(contour)
                
                # Filter small contours
                if width < 20 or height < 20:
                    continue
                
                # Calculate area
                area = cv2.contourArea(contour)
                rect_area = width * height
                
                # Skip if area is too small or ratio is unrealistic
                if area < 100 or float(area) / rect_area < 0.1:
                    continue
                
                # Simple confidence score based on contour area
                confidence = min(0.9, area / 10000)
                
                detections.append({
                    "bounding_box": [x, y, width, height],
                    "confidence": confidence
                })
                
                # Draw on visualization image
                cv2.rectangle(vis_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(vis_image, f"{confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            logger.info(f"Detected {len(detections)} text regions using OpenCV")
            return detections, vis_image
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {str(e)}")
            # Return empty results and original image
            return [], image
    
    def generate_mask(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate binary mask for the detected text regions.
        
        Args:
            image: Original image
            regions: List of detected regions
            
        Returns:
            Binary mask with text regions masked
        """
        # Create empty mask with same dimensions as image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Fill the mask with detected regions
        for region in regions:
            x, y, w, h = region["bounding_box"]
            # Add padding
            pad = 2
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(mask.shape[1], x + w + pad)
            y_max = min(mask.shape[0], y + h + pad)
            
            mask[y_min:y_max, x_min:x_max] = 255
        
        return mask
    
    @classmethod
    def set_model_url(cls, model_name, url, md5=None):
        """
        Set the URL for a specific model to enable downloading.
        
        Args:
            model_name: Name of the model (e.g., 'manga_text_detection')
            url: Direct download URL for the model weights
            md5: Optional MD5 hash for verification
        """
        if model_name in cls.YOLO_MODELS:
            cls.YOLO_MODELS[model_name]['url'] = url
            if md5:
                cls.YOLO_MODELS[model_name]['md5'] = md5
            return True
        return False
