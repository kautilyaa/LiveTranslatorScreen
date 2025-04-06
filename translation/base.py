import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseTranslator:
    """
    Base class for text translators.
    """
    
    def __init__(self, source_language: str = "ja", target_language: str = "en"):
        """
        Initialize translator.
        
        Args:
            source_language: Source language code
            target_language: Target language code
        """
        self.source_language = source_language
        self.target_language = target_language
    
    def translate(self, text: str) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        raise NotImplementedError("Translators must implement translate method")
    
    def batch_translate(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            
        Returns:
            List of translated texts
        """
        # Default implementation translates each text individually
        # Subclasses can override for more efficient batch translation
        return [self.translate(text) for text in texts]
    
    @classmethod
    def get_name(cls) -> str:
        """Get name of translator"""
        return cls.__name__
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, List[str]]:
        """
        Get supported language pairs.
        
        Returns:
            Dictionary with source languages as keys and lists of target languages as values
        """
        return {"en": ["en"]}  # Default supports only English to English
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if the translator is available"""
        return True

class DummyTranslator(BaseTranslator):
    """
    Dummy translator that returns the original text.
    Useful for testing or as a fallback.
    """
    
    def translate(self, text: str) -> str:
        """Return the input text unchanged"""
        return text
    
    @classmethod
    def get_name(cls) -> str:
        return "No Translation"
