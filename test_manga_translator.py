#!/usr/bin/env python3
"""
Test script for the manga translator system.
This will test each component independently and then the full pipeline.
"""

import os
import cv2
import numpy as np
import logging
from PIL import Image
import time
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_manga_translator")

# Import system components
from core.pipeline import MangaProcessingPipeline
from core.config import Config
from models.detection import TextDetector
from ocr.base import BaseOcr, OcrResult
from translation.base import BaseTranslator
from inpainting.base import BaseInpainting
from utils.image_processing import encode_image_to_base64, decode_base64_to_image

def test_text_detection(image_path):
    """Test the text detection module"""
    logger.info("Testing text detection...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    # Initialize detector
    detector = TextDetector()
    
    # Detect text regions
    start_time = time.time()
    regions, viz_image = detector.detect(image)
    end_time = time.time()
    
    # Log results
    logger.info(f"Detected {len(regions)} text regions in {end_time - start_time:.2f} seconds")
    
    # Save visualization
    output_path = "output_detection.jpg"
    cv2.imwrite(output_path, viz_image)
    logger.info(f"Saved visualization to {output_path}")
    
    return True

def test_ocr(image_path, provider_name="EasyOCR (Demo)", language="en"):
    """Test OCR with different providers"""
    logger.info(f"Testing OCR with {provider_name}...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    # Initialize pipeline for OCR provider lookup
    config = Config()
    pipeline = MangaProcessingPipeline(config)
    
    # Get available OCR providers
    available_providers = pipeline.list_ocr_providers()
    logger.info(f"Available OCR providers: {available_providers}")
    
    # Initialize requested OCR provider
    ocr_provider = pipeline.get_ocr_provider(provider_name, language)
    
    # Process image
    start_time = time.time()
    results = ocr_provider.process_batch([image])
    end_time = time.time()
    
    # Log results
    logger.info(f"OCR completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Detected text: {results[0].text[:200]}...")  # Show first 200 chars
    logger.info(f"Detected {len(results[0].bounding_boxes)} text regions")
    
    return True

def test_translation(text="Hello, world!", source_lang="en", target_lang="ja"):
    """Test translation with different providers"""
    logger.info(f"Testing translation from {source_lang} to {target_lang}...")
    
    # Initialize pipeline for translator lookup
    config = Config()
    pipeline = MangaProcessingPipeline(config)
    
    # Get available translators
    available_translators = pipeline.list_translators()
    logger.info(f"Available translators: {available_translators}")
    
    # Test each available translator
    for translator_name in available_translators:
        logger.info(f"Testing {translator_name}...")
        
        # Initialize translator
        translator = pipeline.get_translator(translator_name, source_lang, target_lang)
        
        # Translate text
        start_time = time.time()
        translated_text = translator.translate(text)
        end_time = time.time()
        
        # Log results
        logger.info(f"Translation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Original text: {text}")
        logger.info(f"Translated text: {translated_text}")
    
    return True

def test_inpainting(image_path, model_name="OpenCV Inpainting"):
    """Test inpainting with different models"""
    logger.info(f"Testing inpainting with {model_name}...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    # Create a simple mask for testing
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add some text-like regions to mask
    for _ in range(5):
        x = np.random.randint(width // 4, 3 * width // 4)
        y = np.random.randint(height // 4, 3 * height // 4)
        w = np.random.randint(50, 200)
        h = np.random.randint(20, 50)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    # Initialize pipeline for inpainting model lookup
    config = Config()
    pipeline = MangaProcessingPipeline(config)
    
    # Get available inpainting models
    available_models = pipeline.list_inpainting_models()
    logger.info(f"Available inpainting models: {available_models}")
    
    # Initialize requested inpainting model
    inpainting_model = pipeline.get_inpainting_model(model_name)
    
    # Process image
    start_time = time.time()
    inpainted_image = inpainting_model.inpaint(image, mask)
    end_time = time.time()
    
    # Log results
    logger.info(f"Inpainting completed in {end_time - start_time:.2f} seconds")
    
    # Save output
    output_path = "output_inpainting.jpg"
    cv2.imwrite(output_path, inpainted_image)
    logger.info(f"Saved inpainted image to {output_path}")
    
    # Save mask for reference
    mask_path = "output_mask.jpg"
    cv2.imwrite(mask_path, mask)
    logger.info(f"Saved mask to {mask_path}")
    
    return True

def test_full_pipeline(image_path, source_lang="ja", target_lang="en", 
                      ocr_provider="easyocr", 
                      inpainting_model="lama",
                      translator="huggingface"):
    """Test the full manga processing pipeline"""
    logger.info("Testing full manga processing pipeline...")
    
    # Initialize pipeline
    config = Config()
    pipeline = MangaProcessingPipeline(config)
    
    # Configure pipeline
    pipeline.configure(
        ocr_provider=ocr_provider,
        inpainting_model=inpainting_model,
        translator=translator,
        source_language=source_lang,
        target_language=target_lang
    )
    
    # Load image
    pil_image = Image.open(image_path)
    
    # Process image
    start_time = time.time()
    processed_image, text_results, metrics = pipeline.process(pil_image)
    end_time = time.time()
    
    # Log results
    logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Processing metrics: {metrics}")
    logger.info(f"Detected and translated {len(text_results)} text regions")
    
    # Log some sample results
    if text_results:
        logger.info("Sample text results:")
        for i, result in enumerate(text_results[:3]):  # Show first 3 results
            logger.info(f"  {i+1}. Original: {result.get('text', '')}")
            logger.info(f"     Translated: {result.get('translated_text', '')}")
    
    # Save output
    output_path = "output_pipeline.jpg"
    processed_image.save(output_path)
    logger.info(f"Saved processed image to {output_path}")
    
    return True

def create_sample_image(output_path="sample_manga.jpg", text=True):
    """Create a simple test image if none is provided"""
    logger.info(f"Creating sample test image at {output_path}...")
    
    # Create a simple image with text-like regions
    width, height = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some background elements
    for _ in range(20):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        color = np.random.randint(200, 255, 3).tolist()
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
    
    # Add some random shapes
    for _ in range(10):
        x = np.random.randint(0, width - 100)
        y = np.random.randint(0, height - 100)
        w = np.random.randint(50, 150)
        h = np.random.randint(50, 150)
        color = np.random.randint(180, 240, 3).tolist()
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    
    # Add some text-like elements
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = ["こんにちは", "マンガ", "テキスト", "翻訳", "Hello", "Manga", "Text"]
        
        for _ in range(8):
            x = np.random.randint(50, width - 200)
            y = np.random.randint(50, height - 50)
            text = np.random.choice(texts)
            cv2.putText(image, text, (x, y), font, 1, (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite(output_path, image)
    logger.info(f"Created sample image: {output_path}")
    
    return output_path

def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description="Test the manga translator system")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--tests", type=str, default="all", 
                       help="Tests to run (all, detection, ocr, translation, inpainting, pipeline)")
    parser.add_argument("--source-lang", type=str, default="ja", help="Source language code")
    parser.add_argument("--target-lang", type=str, default="en", help="Target language code")
    
    args = parser.parse_args()
    
    # Create a sample image if none is provided
    image_path = args.image
    if not image_path or not os.path.exists(image_path):
        image_path = create_sample_image()
    
    # Run the requested tests
    tests = args.tests.lower().split(",")
    
    if "all" in tests or "detection" in tests:
        test_text_detection(image_path)
        print()
    
    if "all" in tests or "ocr" in tests:
        # Test with different OCR providers
        providers = ["easyocr", "tesseract", "aws", "azure"]
        for provider in providers:
            test_ocr(image_path, provider, args.source_lang)
            print()
    
    if "all" in tests or "translation" in tests:
        test_translation("Hello, world!", args.source_lang, args.target_lang)
        if args.source_lang != "en" and args.target_lang != "en":
            test_translation("こんにちは、世界！", "ja", "en")
        print()
    
    if "all" in tests or "inpainting" in tests:
        # Test with different inpainting models
        models = ["lama", "stable_diffusion"]
        for model in models:
            test_inpainting(image_path, model)
            print()
    
    if "all" in tests or "pipeline" in tests:
        test_full_pipeline(image_path, args.source_lang, args.target_lang)
        print()
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()