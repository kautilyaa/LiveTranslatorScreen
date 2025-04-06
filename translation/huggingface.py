import logging
import os
from typing import List, Dict, Optional
import random

from translation.base import BaseTranslator, DummyTranslator

logger = logging.getLogger(__name__)

class HuggingfaceTranslator(BaseTranslator):
    """
    Simplified translator using dummy translation for development.
    This version does not require the transformers package.
    """
    
    # Default model configurations for language pairs (kept for compatibility)
    DEFAULT_MODELS = {
        "ja-en": "Helsinki-NLP/opus-mt-ja-en",
        "en-ja": "Helsinki-NLP/opus-mt-en-ja",
        "ko-en": "Helsinki-NLP/opus-mt-ko-en",
        "en-ko": "Helsinki-NLP/opus-mt-en-ko",
        "zh-en": "Helsinki-NLP/opus-mt-zh-en",
        "en-zh": "Helsinki-NLP/opus-mt-en-zh",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "de-en": "Helsinki-NLP/opus-mt-de-en",
        "en-de": "Helsinki-NLP/opus-mt-en-de",
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
        "es-en": "Helsinki-NLP/opus-mt-es-en",
        "en-es": "Helsinki-NLP/opus-mt-en-es"
    }
    
    # Simple translations for demo mode
    DEMO_TRANSLATIONS = {
        "ja-en": {
            "こんにちは": "Hello",
            "さようなら": "Goodbye",
            "ありがとう": "Thank you",
            "お元気ですか": "How are you?",
            "マンガ": "Manga",
            "テキスト": "Text",
            "翻訳": "Translation"
        },
        "en-ja": {
            "Hello": "こんにちは",
            "Goodbye": "さようなら",
            "Thank you": "ありがとう",
            "How are you?": "お元気ですか",
            "Manga": "マンガ",
            "Text": "テキスト",
            "Translation": "翻訳"
        }
    }
    
    def __init__(self, source_language: str = "ja", target_language: str = "en", model_name: Optional[str] = None):
        """
        Initialize simplified translator.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            model_name: Not used in this implementation
        """
        super().__init__(source_language, target_language)
        
        # Language pair key
        self.lang_pair = f"{source_language}-{target_language}"
        logger.info(f"Initializing simplified translator for {self.lang_pair}")
        
        # Fallback translator that returns original text
        self.fallback = DummyTranslator(source_language, target_language)
    
    def translate(self, text: str) -> str:
        """
        Translate text using demo translations or append "[Translated]" for demonstration.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # Check if text is empty
        if not text or not text.strip():
            return ""
        
        try:
            # Check if we have a demo translation for this text
            demo_dict = self.DEMO_TRANSLATIONS.get(self.lang_pair, {})
            if text in demo_dict:
                return demo_dict[text]
            
            # Otherwise, append [Translated] for demonstration
            if self.source_language != self.target_language:
                return f"{text} [Translated]"
            else:
                return text
            
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            return self.fallback.translate(text)
    
    def batch_translate(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            
        Returns:
            List of translated texts
        """
        # Translate each text individually
        return [self.translate(text) for text in texts]
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, List[str]]:
        """
        Get supported language pairs.
        
        Returns:
            Dictionary with source languages as keys and lists of target languages as values
        """
        # Extract unique source and target languages from DEFAULT_MODELS
        supported = {}
        
        for lang_pair in cls.DEFAULT_MODELS.keys():
            source, target = lang_pair.split("-")
            
            if source not in supported:
                supported[source] = []
                
            if target not in supported[source]:
                supported[source].append(target)
        
        return supported
    
    @classmethod
    def is_available(cls) -> bool:
        """This simplified version is always available"""
        return True
    
    @classmethod
    def get_name(cls) -> str:
        return "Hugging Face Translator (Demo)"
