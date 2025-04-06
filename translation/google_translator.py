"""
Google Translate implementation for the manga translation system.
Provides translation using Google's Cloud Translation API or a demo version.
"""

import os
import logging
from typing import Dict, List, Optional, Any

from translation.base import BaseTranslator

logger = logging.getLogger(__name__)

class GoogleTranslator(BaseTranslator):
    """
    Translator using Google Translate API for high-quality translation.
    """

    DEFAULT_KEY_ENV = "GOOGLE_API_KEY"
    DEFAULT_PROJECT_ENV = "GOOGLE_PROJECT_ID"
    
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

    def __init__(self, source_language: str = "ja", target_language: str = "en", 
                api_key: Optional[str] = None, project_id: Optional[str] = None):
        """
        Initialize Google translator.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            api_key: Google API key (optional, will use env var if not provided)
            project_id: Google Cloud project ID (optional, will use env var if not provided)
        """
        super().__init__(source_language, target_language)
        self.api_key = api_key or os.environ.get(self.DEFAULT_KEY_ENV)
        self.project_id = project_id or os.environ.get(self.DEFAULT_PROJECT_ENV)
        self._client = None
        self.api_version = None
        self.parent = None
        
        # Initialize Google Translate client if API key is available
        if self.is_available():
            try:
                self._initialize_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translate client: {str(e)}")
    
    def _initialize_client(self):
        """Initialize Google Cloud Translation client if possible."""
        try:
            # Try to use the newer Translation API (v3)
            from google.cloud import translate_v3
            from google.oauth2 import service_account
            
            if self.project_id:
                # If we have a project ID, we can use the v3 API
                try:
                    self._client = translate_v3.TranslationServiceClient()
                    self.api_version = 'v3'
                    self.parent = f"projects/{self.project_id}"
                    logger.info(f"Initialized Google Translate v3 with project: {self.project_id}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize Google Translate v3: {str(e)}")
            
            # Fall back to the v2 API if v3 is not available or project_id is missing
            from google.cloud import translate_v2 as translate
            
            if self.api_key:
                self._client = translate.Client(key=self.api_key)
                self.api_version = 'v2'
                logger.info("Initialized Google Translate v2 with API key")
            elif self.project_id:
                self._client = translate.Client(project=self.project_id)
                self.api_version = 'v2'
                logger.info(f"Initialized Google Translate v2 with project: {self.project_id}")
            else:
                logger.warning("No Google API key or project ID provided")
        except ImportError:
            logger.warning("Google Cloud Translation library not installed")
        except Exception as e:
            logger.error(f"Error initializing Google Translate client: {str(e)}")
    
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
            "ar": "Arabic",
            "hi": "Hindi",
            "th": "Thai",
            "vi": "Vietnamese"
        }
        return language_names.get(lang_code, lang_code)

    def translate(self, text: str) -> str:
        """
        Translate text using Google Translate API or fallback to demo.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        if not text.strip():
            return text
            
        pair_key = f"{self.source_language}-{self.target_language}"
        
        # Try to use the actual Google Translate API
        if self._client:
            try:
                if self.api_version == 'v3':
                    # Use the v3 API
                    from google.cloud import translate_v3
                    
                    request = translate_v3.TranslateTextRequest(
                        parent=self.parent,
                        contents=[text],
                        mime_type="text/plain",
                        source_language_code=self.source_language,
                        target_language_code=self.target_language
                    )
                    response = self._client.translate_text(request=request)
                    return response.translations[0].translated_text
                else:
                    # Use the v2 API
                    result = self._client.translate(
                        text,
                        target_language=self.target_language,
                        source_language=self.source_language
                    )
                    return result["translatedText"]
            except Exception as e:
                logger.error(f"Error using Google Translate API: {str(e)}")
                # Fall back to demo mode
        
        # Demo mode for development/testing
        if pair_key in self.DEMO_TRANSLATIONS and text in self.DEMO_TRANSLATIONS[pair_key]:
            return self.DEMO_TRANSLATIONS[pair_key][text]
        
        # If we don't have a demo translation, append a marker
        return f"{text} [Google Translated to {self._get_language_name(self.target_language)}]"

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
            
        # If we have an actual Google Translate client, use batch translation
        if self._client:
            try:
                if self.api_version == 'v3':
                    # Use the v3 API for batch translation
                    from google.cloud import translate_v3
                    
                    request = translate_v3.TranslateTextRequest(
                        parent=self.parent,
                        contents=texts,
                        mime_type="text/plain",
                        source_language_code=self.source_language,
                        target_language_code=self.target_language
                    )
                    response = self._client.translate_text(request=request)
                    return [translation.translated_text for translation in response.translations]
                else:
                    # Use the v2 API for batch translation
                    results = self._client.translate(
                        texts,
                        target_language=self.target_language,
                        source_language=self.source_language
                    )
                    return [result["translatedText"] for result in results]
            except Exception as e:
                logger.error(f"Error using Google Translate API for batch translation: {str(e)}")
                # Fall back to individual translation
                
        # Otherwise translate one by one
        return [self.translate(text) for text in texts]

    @classmethod
    def get_name(cls) -> str:
        return "Google Translate"
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, List[str]]:
        """Get supported language pairs."""
        languages = [
            "en", "ja", "ko", "zh", "fr", "de", "es", "it", "ru", 
            "pt", "ar", "hi", "th", "vi"
        ]
        
        return {lang: [l for l in languages if l != lang] for lang in languages}
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Google Translate API is available."""
        return bool(os.environ.get(cls.DEFAULT_KEY_ENV) or os.environ.get(cls.DEFAULT_PROJECT_ENV))