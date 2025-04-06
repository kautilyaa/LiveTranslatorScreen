import os
import sys
import logging
import importlib
from typing import Optional, Any

logger = logging.getLogger(__name__)

def load_yolo_model(model_path: str, device: str = 'cpu') -> Optional[Any]:
    """
    Load a YOLO model from the given path, with dynamic import to avoid hard dependencies.
    
    Args:
        model_path: Path to the YOLO model weights (.pt file)
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded YOLO model or None if loading fails
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        # Try to import torch and check availability
        torch = importlib.import_module('torch')
        
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = 'cpu'
        
        # First try to load using ultralytics YOLOv8 (if available)
        try:
            ultralytics = importlib.import_module('ultralytics')
            YOLO = getattr(ultralytics, 'YOLO')
            
            # Load the model
            model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model from {model_path} using ultralytics")
            return model
        except (ImportError, AttributeError):
            logger.info("ultralytics not available, trying YOLOv5")
        
        # If ultralytics fails, try YOLOv5
        try:
            # Load using torch hub
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=model_path, device=device)
            model.conf = 0.5  # Default confidence threshold
            logger.info(f"Loaded YOLOv5 model from {model_path} using torch hub")
            return model
        except Exception as e:
            logger.error(f"Failed to load using torch hub: {str(e)}")
            
            # Try direct import as a last resort
            try:
                sys.path.append(os.path.dirname(model_path))
                from models.yolov5 import attempt_load
                model = attempt_load(model_path, device=device)
                logger.info(f"Loaded YOLOv5 model from {model_path} using direct import")
                return model
            except Exception as e2:
                logger.error(f"Failed to load using direct import: {str(e2)}")
        
        # If we get here, all loading attempts failed
        logger.error("All YOLO loading methods failed")
        return None
        
    except ImportError:
        logger.error("PyTorch not available, cannot load YOLO model")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading YOLO model: {str(e)}")
        return None


def download_yolo_models(download_dir: str = 'models/weights') -> dict:
    """
    Download pre-trained YOLO models from official sources.
    
    Args:
        download_dir: Directory to save models
        
    Returns:
        Dictionary with model names and paths
    """
    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    models = {}
    
    try:
        # Try to import torch
        import torch
        
        # Try to download YOLOv5 models
        try:
            # Text detection model
            model_path = os.path.join(download_dir, 'yolov5s.pt')
            if not os.path.exists(model_path):
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                torch.save(model.state_dict(), model_path)
                models['yolov5s'] = model_path
                logger.info(f"Downloaded YOLOv5s model to {model_path}")
            else:
                models['yolov5s'] = model_path
                logger.info(f"Using existing YOLOv5s model at {model_path}")
        except Exception as e:
            logger.error(f"Failed to download YOLOv5 model: {str(e)}")
            
    except ImportError:
        logger.error("PyTorch not available, cannot download YOLO models")
    
    return models