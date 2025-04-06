"""
DeepL translator implementation for the manga translation system.
Provides translation using DeepL's API or a demo version.
"""

import os
import logging
from typing import Dict, List, Optional, Any

from translation.base import BaseTranslator

logger = logging.getLogger(__name__)

class DeepLTranslator(BaseTranslator):
    """
    Translator using DeepL API for high-quality translation.
    """

    DEFAULT_KEY_ENV = "DEEPL_API_KEY"
    
    # For demo/fallback purposes
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

    # Language code mapping between ISO and DeepL format
    LANGUAGE_MAPPING = {
        'en': 'EN',
        'de': 'DE',
        'fr': 'FR',
        'es': 'ES',
        'it': 'IT',
        'nl': 'NL',
        'pl': 'PL',
        'pt': 'PT',
        'ru': 'RU',
        'ja': 'JA',
        'zh': 'ZH',
        'ko': 'KO'
    }

    def __init__(self, source_language: str = "ja", target_language: str = "en", 
                api_key: Optional[str] = None):
        """
        Initialize DeepL translator.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            api_key: DeepL API key (optional, will use env var if not provided)
        """
        super().__init__(source_language, target_language)
        self.api_key = api_key or os.environ.get(self.DEFAULT_KEY_ENV)
        self._client = None
        
        # Initialize DeepL client if API key is available
        if self.is_available():
            try:
                self._initialize_client()
            except Exception as e:
                logger.warning(f"Failed to initialize DeepL client: {str(e)}")
    
    def _initialize_client(self):
        """Initialize DeepL Translation client if possible."""
        try:
            import deepl
            
            if self.api_key:
                self._client = deepl.Translator(self.api_key)
            else:
                logger.warning("No DeepL API key provided")
        except ImportError:
            logger.warning("DeepL library not installed")
        except Exception as e:
            logger.error(f"Error initializing DeepL client: {str(e)}")
    
    def _map_language_code(self, lang_code: str) -> str:
        """Map ISO language code to DeepL format."""
        return self.LANGUAGE_MAPPING.get(lang_code, lang_code.upper())
    
    def _get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name from code."""
        language_names = {
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "ru": "Russian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "pl": "Polish"
        }
        return language_names.get(lang_code, lang_code)

    def translate(self, text: str) -> str:
        """
        Translate text using DeepL API or fallback to demo.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        if not text.strip():
            return text
            
        pair_key = f"{self.source_language}-{self.target_language}"
        
        # Try to use the actual DeepL API
        if self._client:
            try:
                source_lang = self._map_language_code(self.source_language)
                target_lang = self._map_language_code(self.target_language)
                
                result = self._client.translate_text(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                return result.text
            except Exception as e:
                logger.error(f"Error using DeepL API: {str(e)}")
                # Fall back to demo mode
        
        # Demo mode for development/testing
        if pair_key in self.DEMO_TRANSLATIONS and text in self.DEMO_TRANSLATIONS[pair_key]:
            return self.DEMO_TRANSLATIONS[pair_key][text]
        
        # If we don't have a demo translation, append a marker
        return f"{text} [DeepL Translated to {self._get_language_name(self.target_language)}]"

    def batch_translate(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts efficiently.
        
        Args:
            texts: List of texts to translate
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
            
        # If we have an actual DeepL client and API key, use batch translation
        if self._client:
            try:
                source_lang = self._map_language_code(self.source_language)
                target_lang = self._map_language_code(self.target_language)
                
                results = self._client.translate_text(
                    texts,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                return [r.text for r in results]
            except Exception as e:
                logger.error(f"Error using DeepL API for batch translation: {str(e)}")
                # Fall back to individual translation
        
        # Otherwise translate one by one
        return [self.translate(text) for text in texts]

    @classmethod
    def get_name(cls) -> str:
        return "DeepL"
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, List[str]]:
        """Get supported language pairs."""
        languages = list(cls.LANGUAGE_MAPPING.keys())
        return {lang: [l for l in languages if l != lang] for lang in languages}
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if DeepL API is available."""
        return bool(os.environ.get(cls.DEFAULT_KEY_ENV))