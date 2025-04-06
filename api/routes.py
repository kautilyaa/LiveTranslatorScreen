import base64
import io
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify

from core.pipeline import MangaProcessingPipeline
from core.config import Config
from utils.image_processing import encode_image_to_base64, decode_base64_to_image

# Initialize logger
logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Global pipeline instance
config = Config()
pipeline = MangaProcessingPipeline(config)

@api_bp.route('/process', methods=['POST'])
def process_image():
    """
    Process a manga image with text detection, OCR, inpainting, and translation
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "ocr_provider": "tesseract|easyocr|azure|aws",
        "inpainting_model": "lama|stable_diffusion",
        "translator": "huggingface|llm|ollama|google|deepl",
        "source_language": "ja|ko|en|...",
        "target_language": "en|ja|ko|...",
        "detection_model": "path/to/yolo_model.pt", (optional),
        "segmentation_model": "path/to/segmentation_model.pt" (optional)
    }
    
    Returns:
    {
        "processed_image": "base64_encoded_processed_image",
        "detected_texts": [
            {
                "text": "original_text",
                "translated_text": "translated_text",
                "bounding_box": [x, y, width, height],
                "confidence": 0.95
            },
            ...
        ],
        "processing_time": {
            "detection": 0.5,
            "segmentation": 0.3,
            "ocr": 0.3,
            "inpainting": 0.8,
            "translation": 0.4,
            "total": 2.0
        }
    }
    """
    from flask import request, jsonify
    
    try:
        # Parse request data
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing required parameter: image"}), 400
        
        # Decode base64 image
        try:
            image = decode_base64_to_image(data['image'])
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Configure pipeline with request parameters
        pipeline.configure(
            ocr_provider=data.get('ocr_provider'),
            inpainting_model=data.get('inpainting_model'),
            translator=data.get('translator'),
            source_language=data.get('source_language'),
            target_language=data.get('target_language'),
            detection_model=data.get('detection_model'),
            segmentation_model=data.get('segmentation_model')
        )
        
        # Process the image
        processed_image, detected_texts, metrics = pipeline.process(image)
        
        # Encode processed image to base64
        processed_image_b64 = encode_image_to_base64(processed_image)
        
        # Prepare response
        response = {
            "processed_image": processed_image_b64,
            "detected_texts": detected_texts,
            "processing_time": metrics
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/ocr', methods=['POST'])
def perform_ocr():
    """
    Perform OCR on an image with specified provider
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "provider": "tesseract|easyocr|azure|aws",
        "language": "ja|ko|en|..."
    }
    
    Returns:
    {
        "text_results": [
            {
                "text": "detected_text",
                "bounding_box": [x, y, width, height],
                "confidence": 0.95
            },
            ...
        ],
        "processing_time": 0.5
    }
    """
    from flask import request, jsonify
    
    try:
        # Parse request data
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing required parameter: image"}), 400
        
        # Get OCR parameters
        provider = data.get('provider', pipeline.current_ocr)
        language = data.get('language', pipeline.current_source_lang)
        
        # Decode base64 image
        try:
            image = decode_base64_to_image(data['image'])
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Perform OCR
        start_time = time.time()
        ocr_provider = pipeline.get_ocr_provider(provider, language)
        ocr_results = ocr_provider.process_batch([image_np])
        processing_time = time.time() - start_time
        
        # Format results
        text_results = []
        for result in ocr_results:
            if result.bounding_boxes:
                for box in result.bounding_boxes:
                    text_results.append({
                        "text": box.get('text', ''),
                        "bounding_box": box.get('bounding_box', [0, 0, 0, 0]),
                        "confidence": box.get('confidence', 0.0)
                    })
            else:
                text_results.append({
                    "text": result.text,
                    "bounding_box": [0, 0, image.width, image.height],
                    "confidence": 1.0
                })
        
        return jsonify({
            "text_results": text_results,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/translate', methods=['POST'])
def translate_text():
    """
    Translate text using specified provider
    
    Expected JSON payload:
    {
        "text": "text_to_translate",
        "provider": "huggingface|llm|ollama|google|deepl",
        "source_language": "ja|ko|en|...",
        "target_language": "en|ja|ko|..."
    }
    
    Returns:
    {
        "translated_text": "translated_text",
        "processing_time": 0.5
    }
    """
    from flask import request, jsonify
    
    try:
        # Parse request data
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400
        
        # Get translation parameters
        text = data.get('text', '')
        provider = data.get('provider', pipeline.current_translator)
        source_lang = data.get('source_language', pipeline.current_source_lang)
        target_lang = data.get('target_language', pipeline.current_target_lang)
        
        # Perform translation
        start_time = time.time()
        translator = pipeline.get_translator(provider, source_lang, target_lang)
        translated_text = translator.translate(text)
        processing_time = time.time() - start_time
        
        return jsonify({
            "translated_text": translated_text,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/inpaint', methods=['POST'])
def inpaint_image():
    """
    Perform inpainting on an image with specified model
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "mask": "base64_encoded_mask_image",
        "model": "lama|stable_diffusion"
    }
    
    Returns:
    {
        "inpainted_image": "base64_encoded_inpainted_image",
        "processing_time": 0.8
    }
    """
    from flask import request, jsonify
    
    try:
        # Parse request data
        data = request.json
        
        if not data or 'image' not in data or 'mask' not in data:
            return jsonify({"error": "Missing required parameters: image, mask"}), 400
        
        # Get inpainting parameters
        model_name = data.get('model', pipeline.current_inpainting)
        
        # Decode base64 images
        try:
            image = decode_base64_to_image(data['image'])
            image_np = np.array(image)
            
            mask = decode_base64_to_image(data['mask'])
            # Convert mask to grayscale if it's RGB
            if len(mask.getbands()) > 1:
                mask = mask.convert('L')
            mask_np = np.array(mask)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Perform inpainting
        start_time = time.time()
        inpainter = pipeline.get_inpainting_model(model_name)
        inpainted_image = inpainter.inpaint(image_np, mask_np)
        processing_time = time.time() - start_time
        
        # Encode inpainted image to base64
        inpainted_image_pil = Image.fromarray(inpainted_image)
        inpainted_image_b64 = encode_image_to_base64(inpainted_image_pil)
        
        return jsonify({
            "inpainted_image": inpainted_image_b64,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error inpainting image: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/detect', methods=['POST'])
def detect_text_regions():
    """
    Detect text regions in an image
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image"
    }
    
    Returns:
    {
        "regions": [
            {
                "bounding_box": [x, y, width, height],
                "confidence": 0.95
            },
            ...
        ],
        "visualization": "base64_encoded_visualization_image",
        "processing_time": 0.5
    }
    """
    from flask import request, jsonify
    
    try:
        # Parse request data
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing required parameter: image"}), 400
        
        # Decode base64 image
        try:
            image = decode_base64_to_image(data['image'])
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Detect text regions
        start_time = time.time()
        detector = pipeline.get_detector()
        regions, visualization = detector.detect(image_np)
        processing_time = time.time() - start_time
        
        # Encode visualization image to base64
        visualization_b64 = encode_image_to_base64(Image.fromarray(visualization))
        
        return jsonify({
            "regions": regions,
            "visualization": visualization_b64,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error detecting text regions: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/segment', methods=['POST'])
def segment_text_regions():
    """
    Generate precise segmentation mask for text regions
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "regions": [{"bounding_box": [x, y, width, height], "confidence": 0.95}, ...],
        "segmentation_model": "path/to/segmentation_model.pt" (optional)
    }
    
    Returns:
    {
        "mask": "base64_encoded_mask_image",
        "processing_time": 0.5
    }
    """
    from flask import request, jsonify
    
    try:
        # Parse request data
        data = request.json
        
        if not data or 'image' not in data or 'regions' not in data:
            return jsonify({"error": "Missing required parameters: image, regions"}), 400
        
        # Decode base64 image
        try:
            image = decode_base64_to_image(data['image'])
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Get regions
        regions = data.get('regions', [])
        if not regions:
            return jsonify({"error": "No regions provided for segmentation"}), 400
        
        # Update segmentation model if provided
        if 'segmentation_model' in data:
            from models.segmentation import SegmentationModel
            pipeline.segmenter = SegmentationModel(model_path=data['segmentation_model'])
        
        # Generate segmentation mask
        start_time = time.time()
        segmenter = pipeline.get_segmenter()
        mask = segmenter.segment(image_np, regions)
        processing_time = time.time() - start_time
        
        # Encode mask to base64
        mask_pil = Image.fromarray(mask)
        mask_b64 = encode_image_to_base64(mask_pil)
        
        return jsonify({
            "mask": mask_b64,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error segmenting text regions: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/providers', methods=['GET'])
def get_providers():
    """Get all available providers for OCR, translation, and inpainting"""
    from flask import jsonify
    
    ocr_providers = pipeline.list_ocr_providers()
    inpainting_models = pipeline.list_inpainting_models()
    translators = pipeline.list_translators()
    
    # Get supported languages for each provider
    ocr_languages = {}
    for provider in ocr_providers:
        provider_class = pipeline._ocr_providers[provider]
        ocr_languages[provider] = provider_class.get_supported_languages()
    
    translator_languages = {}
    for translator in translators:
        translator_class = pipeline._translators[translator]
        translator_languages[translator] = translator_class.get_supported_languages()
    
    return jsonify({
        "ocr_providers": ocr_providers,
        "inpainting_models": inpainting_models,
        "translators": translators,
        "ocr_languages": ocr_languages,
        "translator_languages": translator_languages
    })