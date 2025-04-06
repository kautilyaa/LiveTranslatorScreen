"""
Translator using Ollama for local LLM-based translation.
Provides high-quality translation without dependency on cloud services.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
import requests

from translation.base import BaseTranslator

logger = logging.getLogger(__name__)

class OllamaTranslator(BaseTranslator):
    """
    Translator using Ollama for high-quality translation using local LLMs.
    """

    DEFAULT_MODEL = "mistral"  # Default model, can be changed
    DEFAULT_HOST = "http://localhost:11434"  # Default Ollama API endpoint
    
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
                model_name: Optional[str] = None, api_host: Optional[str] = None):
        """
        Initialize Ollama translator.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            model_name: Ollama model name (defaults to mistral)
            api_host: Ollama API host URL (defaults to localhost:11434)
        """
        super().__init__(source_language, target_language)
        self.model = model_name or self.DEFAULT_MODEL
        self.api_host = api_host or os.environ.get("OLLAMA_API_HOST", self.DEFAULT_HOST)
        self._available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.api_host}/api/version", timeout=2)
            if response.status_code == 200:
                logger.info(f"Ollama server found at {self.api_host}")
                return True
        except Exception as e:
            logger.warning(f"Ollama server not available at {self.api_host}: {str(e)}")
        return False

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
            "pt": "Portuguese"
        }
        return language_names.get(lang_code, lang_code)

    def translate(self, text: str) -> str:
        """
        Translate text using Ollama LLM.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        if not text.strip():
            return text
            
        pair_key = f"{self.source_language}-{self.target_language}"
        
        # Try to use the actual Ollama API if available
        if self._available:
            try:
                prompt = f"""Translate the following text from {self._get_language_name(self.source_language)} to {self._get_language_name(self.target_language)}. 
                
Return ONLY the translated text, without any additional comments, explanations, or quotation marks.

Text to translate: {text}"""
                
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.api_host}/api/generate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["response"].strip()
                else:
                    logger.error(f"Error using Ollama API: {response.status_code} - {response.text}")
                    # Fall back to demo mode
            except Exception as e:
                logger.error(f"Error using Ollama API: {str(e)}")
                # Fall back to demo mode
        
        # Demo mode for development/testing
        if pair_key in self.DEMO_TRANSLATIONS and text in self.DEMO_TRANSLATIONS[pair_key]:
            return self.DEMO_TRANSLATIONS[pair_key][text]
        
        # If we don't have a demo translation, append a marker
        return f"{text} [Ollama Translated to {self._get_language_name(self.target_language)}]"

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
            
        # If Ollama is not available, translate one by one using demo mode
        if not self._available:
            return [self.translate(text) for text in texts]
            
        # Otherwise, we can try to batch process efficiently
        try:
            combined_text = "\n---\n".join(texts)
            batch_prompt = f"""Translate the following texts from {self._get_language_name(self.source_language)} to {self._get_language_name(self.target_language)}. 
                
Each text is separated by '---'. Return ONLY the translated texts, separated by '---', preserving the original order.

Texts to translate:
{combined_text}"""
            
            payload = {
                "model": self.model,
                "prompt": batch_prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.api_host}/api/generate",
                json=payload,
                timeout=60  # Longer timeout for batch processing
            )
            
            if response.status_code == 200:
                result = response.json()
                # Split the result back into individual translations
                translations = result["response"].strip().split("---")
                # Ensure we have the same number of translations as inputs
                if len(translations) == len(texts):
                    return [t.strip() for t in translations]
                # If not, fall back to individual translations
            
        except Exception as e:
            logger.error(f"Error in batch translation with Ollama: {str(e)}")
        
        # Fallback: translate one by one
        return [self.translate(text) for text in texts]

    @classmethod
    def get_name(cls) -> str:
        return "Ollama (Local LLM)"
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, List[str]]:
        """Get supported language pairs."""
        # Since Ollama can theoretically handle any language pair with the right prompt,
        # we'll list the most common ones
        languages = [
            "en", "ja", "ko", "zh", "fr", "de", "es", "it", "ru", "pt"
        ]
        
        return {lang: [l for l in languages if l != lang] for lang in languages}
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Ollama is running and available."""
        try:
            api_host = os.environ.get("OLLAMA_API_HOST", cls.DEFAULT_HOST)
            response = requests.get(f"{api_host}/api/version", timeout=2)
            return response.status_code == 200
        except:
            return False