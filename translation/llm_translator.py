import os
import logging
import time
from typing import Dict, List, Optional, Any

import openai
from openai import OpenAI

from translation.base import BaseTranslator

logger = logging.getLogger(__name__)

class LLMTranslator(BaseTranslator):
    """
    Translator using LLM (OpenAI) APIs for high-quality translation.
    """
    
    DEFAULT_MODEL = "gpt-3.5-turbo"
    
    def __init__(self, source_language: str = "ja", target_language: str = "en", 
                model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LLM translator.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            model_name: LLM model name (defaults to gpt-3.5-turbo)
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        super().__init__(source_language, target_language)
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. LLM translation will not work.")
            self._client = None
        else:
            self._client = OpenAI(api_key=self.api_key)
            
        # Cache for translations to avoid repeated API calls
        self._translation_cache = {}
        
        # Map language codes to human-readable names
        self.language_names = {
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
    
    def _get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name from code."""
        return self.language_names.get(lang_code, lang_code)
    
    def translate(self, text: str) -> str:
        """
        Translate text using LLM.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
            
        # Check cache
        cache_key = (text, self.source_language, self.target_language)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]
            
        if not self._client:
            logger.warning("LLM translation failed: No API key provided")
            return f"{text} [LLM Translation Error: No API key]"
        
        try:
            source_lang = self._get_language_name(self.source_language)
            target_lang = self._get_language_name(self.target_language)
            
            # Create prompt for translation
            system_prompt = f"You are a professional translator specialized in manga and comics translation from {source_lang} to {target_lang}."
            user_prompt = f"Translate the following text from {source_lang} to {target_lang}. Preserve the meaning, tone, and style appropriate for manga/comics context:\n\n{text}"
            
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=1024
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Cache the result
            self._translation_cache[cache_key] = translated_text
            
            return translated_text
            
        except Exception as e:
            logger.error(f"LLM translation error: {str(e)}")
            return f"{text} [LLM Translation Error]"
    
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
            
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text and text.strip()]
        if not non_empty_texts:
            return [""] * len(texts)
        
        # Prepare result list with same length as input
        results = [""] * len(texts)
        
        if not self._client:
            logger.warning("LLM batch translation failed: No API key provided")
            for i, text in enumerate(texts):
                if text and text.strip():
                    results[i] = f"{text} [LLM Translation Error: No API key]"
            return results
        
        try:
            source_lang = self._get_language_name(self.source_language)
            target_lang = self._get_language_name(self.target_language)
            
            # Create system prompt
            system_prompt = f"You are a professional translator specialized in manga and comics translation from {source_lang} to {target_lang}."
            
            # If we have more than 5 texts, batch them together
            # This reduces the number of API calls and is more efficient
            if len(non_empty_texts) > 5:
                # Combine texts with markers for batch processing
                batch_texts = [f"[{i}] {text}" for i, text in enumerate(non_empty_texts)]
                batch_input = "\n\n".join(batch_texts)
                
                user_prompt = f"Translate the following texts from {source_lang} to {target_lang}. Keep the same format with numbered markers like [0], [1], etc. Preserve the meaning, tone, and style appropriate for manga/comics context:\n\n{batch_input}"
                
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2048  # Increased for batch processing
                )
                
                translated_batch = response.choices[0].message.content.strip()
                
                # Parse the batch response
                parts = translated_batch.split("\n\n")
                for part in parts:
                    if not part.strip():
                        continue
                        
                    # Extract index and translated text
                    try:
                        # Format should be [index] translated_text
                        marker_end = part.find("]")
                        if marker_end > 0:
                            marker = part[:marker_end+1].strip()
                            index_str = marker.replace("[", "").replace("]", "").strip()
                            index = int(index_str)
                            
                            if 0 <= index < len(non_empty_texts):
                                translated_text = part[marker_end+1:].strip()
                                
                                # Find the original position of this text
                                original_indices = [i for i, text in enumerate(texts) 
                                                  if text and text.strip() and text.strip() == non_empty_texts[index]]
                                
                                for orig_idx in original_indices:
                                    results[orig_idx] = translated_text
                                    
                                    # Cache the result
                                    cache_key = (non_empty_texts[index], self.source_language, self.target_language)
                                    self._translation_cache[cache_key] = translated_text
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error parsing batch translation: {str(e)}")
            else:
                # Translate each text individually for smaller batches
                for i, text in enumerate(non_empty_texts):
                    # Check cache first
                    cache_key = (text, self.source_language, self.target_language)
                    if cache_key in self._translation_cache:
                        translated = self._translation_cache[cache_key]
                    else:
                        user_prompt = f"Translate the following text from {source_lang} to {target_lang}. Preserve the meaning, tone, and style appropriate for manga/comics context:\n\n{text}"
                        
                        response = self._client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=1024
                        )
                        
                        translated = response.choices[0].message.content.strip()
                        self._translation_cache[cache_key] = translated
                    
                    # Find the original position of this text
                    original_indices = [j for j, t in enumerate(texts) 
                                      if t and t.strip() and t.strip() == text]
                    
                    for orig_idx in original_indices:
                        results[orig_idx] = translated
            
            # Final pass to ensure all inputs have outputs
            for i, (text, result) in enumerate(zip(texts, results)):
                if text and text.strip() and not result:
                    results[i] = text  # Fallback to original if translation failed
            
            return results
                    
        except Exception as e:
            logger.error(f"LLM batch translation error: {str(e)}")
            # Return original texts with error indicator
            for i, text in enumerate(texts):
                if text and text.strip():
                    results[i] = f"{text} [LLM Translation Error]"
            return results
    
    @classmethod
    def get_name(cls) -> str:
        return "LLM Translator"
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, List[str]]:
        """Get supported language pairs."""
        languages = ["en", "ja", "ko", "zh", "fr", "de", "es", "it", "ru", "pt", "ar", "hi", "th", "vi"]
        
        # All languages can be translated to any other language
        supported_pairs = {}
        for source in languages:
            supported_pairs[source] = [target for target in languages if target != source]
            
        return supported_pairs
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenAI API key is available."""
        return bool(os.environ.get("OPENAI_API_KEY"))