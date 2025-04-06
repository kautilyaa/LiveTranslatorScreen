import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from typing import Union, List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

def resize_image(image: Union[np.ndarray, Image.Image], max_size: int = 1024) -> Union[np.ndarray, Image.Image]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image (numpy array or PIL Image)
        max_size: Maximum size for any dimension
        
    Returns:
        Resized image of same type as input
    """
    # Handle different input types
    if isinstance(image, np.ndarray):
        # OpenCV/numpy array
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        if height > max_size or width > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            
            # Resize using OpenCV
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        else:
            return image
    
    elif isinstance(image, Image.Image):
        # PIL Image
        width, height = image.size
        
        # Calculate new dimensions
        if height > max_size or width > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            
            # Resize using PIL
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            return resized
        else:
            return image
    
    else:
        raise TypeError("Input must be numpy.ndarray or PIL.Image.Image")

def draw_boxes_on_image(image: Union[np.ndarray, Image.Image], 
                       boxes: List[Dict[str, Any]], 
                       show_text: bool = True,
                       color: Union[Tuple[int, int, int], str] = (0, 255, 0)) -> Union[np.ndarray, Image.Image]:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (numpy array or PIL Image)
        boxes: List of dictionaries with 'bounding_box' and optional 'text' and 'confidence'
        show_text: Whether to show text and confidence
        color: Box color (BGR tuple for numpy, RGB tuple or string for PIL)
        
    Returns:
        Image with drawn boxes of same type as input
    """
    # Handle different input types
    if isinstance(image, np.ndarray):
        # Create a copy to avoid modifying the original
        result = image.copy()
        
        # Draw boxes on OpenCV/numpy image
        for box in boxes:
            # Get box coordinates
            x, y, w, h = box['bounding_box']
            
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw text if requested
            if show_text and 'text' in box:
                text = box['text']
                confidence = box.get('confidence', None)
                
                # Combine text and confidence if available
                display_text = text
                if confidence is not None:
                    display_text = f"{text} ({confidence:.2f})"
                
                # Add text
                cv2.putText(result, display_text, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    elif isinstance(image, Image.Image):
        # Create a copy to avoid modifying the original
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw boxes on PIL image
        for box in boxes:
            # Get box coordinates
            x, y, w, h = box['bounding_box']
            
            # Draw rectangle
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
            
            # Draw text if requested
            if show_text and 'text' in box:
                text = box['text']
                confidence = box.get('confidence', None)
                
                # Combine text and confidence if available
                display_text = text
                if confidence is not None:
                    display_text = f"{text} ({confidence:.2f})"
                
                # Add text
                draw.text((x, y - 15), display_text, fill=color, font=font)
        
        return result
    
    else:
        raise TypeError("Input must be numpy.ndarray or PIL.Image.Image")

def encode_image_to_base64(image: Union[np.ndarray, Image.Image], format: str = "JPEG") -> str:
    """
    Encode image to base64 string.
    
    Args:
        image: Input image (numpy array or PIL Image)
        format: Image format to encode to ("JPEG" or "PNG")
        
    Returns:
        Base64 encoded string
    """
    # Handle different input types
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, Image.Image):
        # Use PIL Image directly
        image_pil = image
    else:
        raise TypeError("Input must be numpy.ndarray or PIL.Image.Image")
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    image_pil.save(buffer, format=format)
    buffer.seek(0)
    
    # Encode to base64
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return encoded

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        # Extract the base64 part after the comma
        base64_string = base64_string.split(',', 1)[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Open as PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    return image

def create_text_mask(image: np.ndarray, boxes: List[Dict[str, Any]], padding: int = 2) -> np.ndarray:
    """
    Create binary mask for text regions.
    
    Args:
        image: Input image
        boxes: List of dictionaries with 'bounding_box'
        padding: Padding around boxes in pixels
        
    Returns:
        Binary mask with 255 for text regions
    """
    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Fill mask with box regions
    for box in boxes:
        x, y, w, h = box['bounding_box']
        
        # Add padding
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(mask.shape[1], x + w + padding)
        y_max = min(mask.shape[0], y + h + padding)
        
        # Fill region
        mask[y_min:y_max, x_min:x_max] = 255
    
    return mask
