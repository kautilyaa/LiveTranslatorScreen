import logging
import time
import numpy as np
import asyncio
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union

from models.detection import TextDetector
from models.segmentation import SegmentationModel
from utils.image_processing import draw_boxes_on_image
from core.async_processor import AsyncProcessor, async_processor, process_images_concurrently

logger = logging.getLogger(__name__)

class MangaProcessingPipeline:
    """
    Main pipeline for manga processing.
    Handles the complete flow from text detection to OCR, inpainting and translation.
    With support for asynchronous processing and advanced segmentation.
    """
    
    def __init__(self, config):
        """
        Initialize the manga processing pipeline.
        
        Args:
            config: Configuration object containing default settings
        """
        self.config = config
        self.detector = TextDetector()
        
        # Initialize segmentation model - defaults to OpenCV-based if no model provided
        self.segmenter = SegmentationModel()
        
        # Initialize provider registries
        self._ocr_providers = {}
        self._inpainting_models = {}
        self._translators = {}
        
        # Load available providers
        self._load_ocr_providers()
        self._load_inpainting_models()
        self._load_translators()
        
        # Set default providers
        self.current_ocr = config.default_ocr_provider
        self.current_inpainting = config.default_inpainting_model
        self.current_translator = config.default_translator
        self.current_source_lang = config.default_source_language
        self.current_target_lang = config.default_target_language
        
        # Initialize async processor for concurrent operations
        self.async_processor = AsyncProcessor()
        
        logger.info("Manga processing pipeline initialized")
    
    def _load_ocr_providers(self):
        """Load all available OCR providers"""
        # Load local OCR providers
        from ocr.tesseract_ocr import TesseractOcr
        from ocr.easy_ocr import EasyOcr
        
        # Register providers
        self._ocr_providers["tesseract"] = TesseractOcr
        self._ocr_providers["easyocr"] = EasyOcr
        
        # Try to load cloud providers if available
        try:
            from ocr.cloud.azure_ocr import AzureOcr
            self._ocr_providers["azure"] = AzureOcr
        except ImportError:
            logger.warning("Azure OCR not available")
        
        try:
            from ocr.cloud.aws_ocr import AwsOcr
            self._ocr_providers["aws"] = AwsOcr
        except ImportError:
            logger.warning("AWS OCR not available")
        
        logger.info(f"Loaded OCR providers: {', '.join(self._ocr_providers.keys())}")
    
    def _load_inpainting_models(self):
        """Load all available inpainting models"""
        # Load inpainting models
        from inpainting.lama import LamaInpainting
        from inpainting.stable_diffusion import StableDiffusionInpainting
        
        # Register models
        self._inpainting_models["lama"] = LamaInpainting
        self._inpainting_models["stable_diffusion"] = StableDiffusionInpainting
        
        logger.info(f"Loaded inpainting models: {', '.join(self._inpainting_models.keys())}")
    
    def _load_translators(self):
        """Load all available translators"""
        # Load translators
        from translation.huggingface import HuggingfaceTranslator
        
        # Register base translators
        self._translators["huggingface"] = HuggingfaceTranslator
        
        # Try to load LLM translator if OpenAI is available
        try:
            from translation.llm_translator import LLMTranslator
            self._translators["llm"] = LLMTranslator
            logger.info("LLM translator (OpenAI) loaded successfully")
        except ImportError:
            logger.warning("OpenAI not available, LLM translator will not be available")
            
        # Try to load Ollama translator if available
        try:
            from translation.ollama_translator import OllamaTranslator
            self._translators["ollama"] = OllamaTranslator
            logger.info("Ollama translator loaded successfully")
        except ImportError:
            logger.warning("Requests package not available, Ollama translator will not be available")
        
        # Try to load Google Translate if available
        try:
            from translation.google_translator import GoogleTranslator
            self._translators["google"] = GoogleTranslator
            logger.info("Google Translate loaded successfully")
        except ImportError:
            logger.warning("Google Cloud Translation package not available, Google Translate will not be available")
        
        # Try to load DeepL translator if available
        try:
            from translation.deepl_translator import DeepLTranslator
            self._translators["deepl"] = DeepLTranslator
            logger.info("DeepL translator loaded successfully")
        except ImportError:
            logger.warning("DeepL package not available, DeepL translator will not be available")
        
        logger.info(f"Loaded translators: {', '.join(self._translators.keys())}")
    
    def configure(self, ocr_provider=None, inpainting_model=None, 
                 translator=None, source_language=None, target_language=None,
                 detection_model=None, segmentation_model=None):
        """
        Configure the pipeline with specific providers and settings.
        
        Args:
            ocr_provider: Name of OCR provider to use
            inpainting_model: Name of inpainting model to use
            translator: Name of translator to use
            source_language: Source language code
            target_language: Target language code
            detection_model: Path to text detection model
            segmentation_model: Path to text segmentation model
        """
        if ocr_provider:
            if ocr_provider in self._ocr_providers:
                self.current_ocr = ocr_provider
            else:
                logger.warning(f"OCR provider '{ocr_provider}' not found. Using default.")
        
        if inpainting_model:
            if inpainting_model in self._inpainting_models:
                self.current_inpainting = inpainting_model
            else:
                logger.warning(f"Inpainting model '{inpainting_model}' not found. Using default.")
        
        if translator:
            if translator in self._translators:
                self.current_translator = translator
            else:
                logger.warning(f"Translator '{translator}' not found. Using default.")
        
        if source_language:
            self.current_source_lang = source_language
        
        if target_language:
            self.current_target_lang = target_language
            
        # Re-initialize detection and segmentation models if paths provided
        if detection_model:
            self.detector = TextDetector(model_path=detection_model)
            
        if segmentation_model:
            self.segmenter = SegmentationModel(model_path=segmentation_model)
        
        logger.info(f"Pipeline configured: OCR={self.current_ocr}, Inpainting={self.current_inpainting}, "
                   f"Translator={self.current_translator}, Languages={self.current_source_lang}->{self.current_target_lang}")
    
    def get_ocr_provider(self, provider_name, language):
        """
        Get an instance of an OCR provider.
        
        Args:
            provider_name: Name of the OCR provider
            language: Language code for OCR
            
        Returns:
            Initialized OCR provider instance
        """
        if provider_name not in self._ocr_providers:
            logger.warning(f"OCR provider '{provider_name}' not found. Using default: {self.current_ocr}")
            provider_name = self.current_ocr
        
        provider_class = self._ocr_providers[provider_name]
        return provider_class(language)
    
    def get_inpainting_model(self, model_name):
        """
        Get an instance of an inpainting model.
        
        Args:
            model_name: Name of the inpainting model
            
        Returns:
            Initialized inpainting model instance
        """
        if model_name not in self._inpainting_models:
            logger.warning(f"Inpainting model '{model_name}' not found. Using default: {self.current_inpainting}")
            model_name = self.current_inpainting
        
        model_class = self._inpainting_models[model_name]
        return model_class()
    
    def get_translator(self, translator_name, source_lang, target_lang):
        """
        Get an instance of a translator.
        
        Args:
            translator_name: Name of the translator
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Initialized translator instance
        """
        if translator_name not in self._translators:
            logger.warning(f"Translator '{translator_name}' not found. Using default: {self.current_translator}")
            translator_name = self.current_translator
        
        translator_class = self._translators[translator_name]
        return translator_class(source_lang, target_lang)
    
    def get_detector(self):
        """Get the text detector instance"""
        return self.detector
    
    def get_segmenter(self):
        """Get the segmentation model instance"""
        return self.segmenter
    
    def list_ocr_providers(self):
        """List all available OCR providers"""
        return list(self._ocr_providers.keys())
    
    def list_inpainting_models(self):
        """List all available inpainting models"""
        return list(self._inpainting_models.keys())
    
    def list_translators(self):
        """List all available translators"""
        return list(self._translators.keys())
    
    async def process_async(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, float]]:
        """
        Process a manga image through the complete pipeline asynchronously.
        
        Args:
            image: PIL Image or numpy array to process
            
        Returns:
            Tuple containing:
            - Processed image (PIL Image)
            - List of detected texts with translations
            - Dictionary of processing metrics
        """
        metrics = {}
        
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # 1. Detect text regions
        logger.info("Detecting text regions...")
        start_time = time.time()
        detect_task = await self.async_processor.run_async(self.detector.detect, image_np)
        regions, visualization = detect_task
        metrics["detection"] = time.time() - start_time
        logger.info(f"Detected {len(regions)} text regions in {metrics['detection']:.2f}s")
        
        # 2. Perform segmentation for more precise masks
        logger.info("Generating precise text masks with segmentation...")
        start_time = time.time()
        segmentation_task = await self.async_processor.run_async(self.segmenter.segment, image_np, regions)
        precise_mask = segmentation_task
        metrics["segmentation"] = time.time() - start_time
        logger.info(f"Segmentation completed in {metrics['segmentation']:.2f}s")
        
        # 3. Extract text regions for OCR in parallel
        text_regions = []
        for region in regions:
            x, y, w, h = region["bounding_box"]
            # Extract region with some padding
            pad = 5
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(image_np.shape[1], x + w + pad)
            y_max = min(image_np.shape[0], y + h + pad)
            
            region_img = image_np[y_min:y_max, x_min:x_max]
            text_regions.append(region_img)
        
        # 4. Perform OCR - process all regions in parallel
        logger.info(f"Performing OCR with provider: {self.current_ocr}")
        start_time = time.time()
        ocr_provider = self.get_ocr_provider(self.current_ocr, self.current_source_lang)
        
        # Use batch processing if available, otherwise process each region in parallel
        if hasattr(ocr_provider, 'process_batch'):
            ocr_results = await self.async_processor.run_async(ocr_provider.process_batch, text_regions)
        else:
            # Process each region concurrently
            async def process_single(region_img):
                return ocr_provider([region_img])[0]
            
            ocr_tasks = [process_single(region) for region in text_regions]
            ocr_results = await asyncio.gather(*ocr_tasks)
            
        metrics["ocr"] = time.time() - start_time
        logger.info(f"OCR completed in {metrics['ocr']:.2f}s")
        
        # 5. Translate detected text
        logger.info(f"Translating text with provider: {self.current_translator}")
        start_time = time.time()
        translator = self.get_translator(
            self.current_translator, 
            self.current_source_lang, 
            self.current_target_lang
        )
        
        # Get all texts to translate
        texts_to_translate = [
            ocr_result.text for ocr_result in ocr_results 
            if ocr_result and ocr_result.text and ocr_result.text.strip()
        ]
        
        # Batch translate all texts at once if possible
        if texts_to_translate:
            if hasattr(translator, 'batch_translate'):
                translated_texts = await self.async_processor.run_async(
                    translator.batch_translate, texts_to_translate
                )
            else:
                # Translate each text concurrently
                async def translate_single(text):
                    return translator.translate(text)
                
                translate_tasks = [translate_single(text) for text in texts_to_translate]
                translated_texts = await asyncio.gather(*translate_tasks)
                
            # Map translations back to regions
            translate_idx = 0
            for i, (region, ocr_result) in enumerate(zip(regions, ocr_results)):
                if ocr_result and ocr_result.text and ocr_result.text.strip():
                    region["text"] = ocr_result.text
                    region["translated_text"] = translated_texts[translate_idx]
                    translate_idx += 1
                else:
                    region["text"] = ""
                    region["translated_text"] = ""
        else:
            # No text to translate
            for i, (region, ocr_result) in enumerate(zip(regions, ocr_results)):
                region["text"] = ocr_result.text if ocr_result and ocr_result.text else ""
                region["translated_text"] = ""
                
        metrics["translation"] = time.time() - start_time
        logger.info(f"Translation completed in {metrics['translation']:.2f}s")
        
        # 6. Use the precise mask from segmentation for inpainting
        # Or fall back to bounding box mask if segmentation produced empty mask
        if np.sum(precise_mask) > 0:
            mask = precise_mask
            logger.info("Using precise segmentation mask for inpainting")
        else:
            # Create mask from bounding boxes as fallback
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            for region in regions:
                if region["text"].strip():  # Only inpaint regions with text
                    x, y, w, h = region["bounding_box"]
                    # Add padding
                    pad = 2
                    x_min = max(0, x - pad)
                    y_min = max(0, y - pad)
                    x_max = min(mask.shape[1], x + w + pad)
                    y_max = min(mask.shape[0], y + h + pad)
                    
                    mask[y_min:y_max, x_min:x_max] = 255
            logger.info("Falling back to bounding box mask for inpainting")
        
        # 7. Perform inpainting
        logger.info(f"Inpainting with model: {self.current_inpainting}")
        start_time = time.time()
        inpainter = self.get_inpainting_model(self.current_inpainting)
        inpainted_image = await self.async_processor.run_async(inpainter.inpaint, image_np, mask)
        metrics["inpainting"] = time.time() - start_time
        logger.info(f"Inpainting completed in {metrics['inpainting']:.2f}s")
        
        # 8. Draw translated text on inpainted image
        result_image = Image.fromarray(inpainted_image)
        # This would require complex text rendering, simplified version:
        # In a real implementation, we would render the translated text with appropriate
        # styling and positioning based on the original text layout
        
        # For the prototype, we'll return the detected texts for the frontend to render
        detected_texts = []
        for region in regions:
            if "text" in region and region["text"].strip():
                detected_texts.append({
                    "text": region["text"],
                    "translated_text": region["translated_text"],
                    "bounding_box": region["bounding_box"],
                    "confidence": region.get("confidence", 0.0)
                })
        
        # Calculate total processing time
        metrics["total"] = sum(metrics.values())
        logger.info(f"Total processing time: {metrics['total']:.2f}s")
        
        return result_image, detected_texts, metrics
    
    def process(self, image: Image.Image) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, float]]:
        """
        Process a manga image through the complete pipeline.
        This is a synchronous wrapper around the async process method.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple containing:
            - Processed image (PIL Image)
            - List of detected texts with translations
            - Dictionary of processing metrics
        """
        # Run the async method in the event loop
        try:
            # Get the event loop or create one if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Run the async method and wait for the result
            return loop.run_until_complete(self.process_async(image))
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            # Return original image and empty results in case of error
            return image, [], {"error": str(e)}
