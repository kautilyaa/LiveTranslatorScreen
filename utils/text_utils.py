import logging
import re
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

def clean_ocr_text(text: str) -> str:
    """
    Clean OCR-recognized text by removing noise and fixing common errors.
    
    Args:
        text: OCR text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove isolated special characters
    text = re.sub(r'\s[^\w\s]\s', ' ', text)
    
    # Remove lines with only special characters
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.search(r'[a-zA-Z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', line):
            cleaned_lines.append(line)
    
    # Join lines
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Trim whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def detect_language(text: str) -> str:
    """
    Detect the language of the text.
    
    Args:
        text: Text to detect language
        
    Returns:
        ISO language code
    """
    try:
        from langdetect import detect
        
        if not text or len(text.strip()) < 5:
            return "unknown"
        
        # Simple language detection with langdetect
        lang = detect(text)
        return lang
    except:
        # Fallback to character set based detection
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
            return "zh"
        elif re.search(r'[\u3040-\u309f]', text):  # Hiragana
            return "ja"
        elif re.search(r'[\u30a0-\u30ff]', text):  # Katakana
            return "ja"
        elif re.search(r'[\uac00-\ud7af]', text):  # Korean
            return "ko"
        elif re.search(r'[а-яА-Я]', text):  # Cyrillic
            return "ru"
        elif re.search(r'[a-zA-Z]', text):  # Latin alphabet
            return "en"
        else:
            return "unknown"

def fit_text_to_box(text: str, box_width: int, box_height: int, font_size: int = 12) -> Dict[str, Any]:
    """
    Calculate how to fit text in a box with proper font size and line breaks.
    
    Args:
        text: Text to fit
        box_width: Width of the box in pixels
        box_height: Height of the box in pixels
        font_size: Initial font size
        
    Returns:
        Dictionary with fitted text, font size, and line count
    """
    try:
        # Rough estimate: each character takes about 0.6 * font_size pixels in width
        # and each line takes about 1.2 * font_size pixels in height
        char_width = 0.6 * font_size
        line_height = 1.2 * font_size
        
        # Split text into words
        words = text.split()
        
        # Try to fit with current font size
        lines = []
        current_line = []
        
        for word in words:
            # Check if adding this word exceeds the box width
            test_line = current_line + [word]
            test_line_width = len(' '.join(test_line)) * char_width
            
            if test_line_width <= box_width:
                # Word fits, add it to the current line
                current_line = test_line
            else:
                # Word doesn't fit, start a new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add the last line if it's not empty
        if current_line:
            lines.append(' '.join(current_line))
        
        # Check if all lines fit in the box height
        total_height = len(lines) * line_height
        
        if total_height > box_height:
            # Text doesn't fit with current font size, try smaller size
            return fit_text_to_box(text, box_width, box_height, font_size - 1)
        
        return {
            "text": '\n'.join(lines),
            "font_size": font_size,
            "line_count": len(lines)
        }
        
    except Exception as e:
        logger.error(f"Error fitting text to box: {str(e)}")
        # Fallback: return original text with reduced font size
        return {
            "text": text,
            "font_size": max(8, font_size - 4),
            "line_count": text.count('\n') + 1
        }

def hyphenate_text(text: str, language: str = "en_US") -> str:
    """
    Add hyphenation to text for better line breaks.
    
    Args:
        text: Text to hyphenate
        language: Language code for hyphenation
        
    Returns:
        Hyphenated text with soft hyphens
    """
    try:
        from hyphen import Hyphenator
        
        # Map language code to hyphenation language
        lang_map = {
            "en": "en_US",
            "fr": "fr_FR",
            "de": "de_DE",
            "es": "es_ES",
            "it": "it_IT",
            "nl": "nl_NL",
            "ru": "ru_RU"
        }
        
        # Use mapped language or fallback to provided language
        hyphen_lang = lang_map.get(language, language)
        
        try:
            h = Hyphenator(hyphen_lang)
        except:
            # Fallback to English
            h = Hyphenator("en_US")
        
        # Split into words and hyphenate each word
        words = text.split()
        hyphenated_words = [h.inserted(word, '\u00AD') for word in words]
        
        # Join words back
        return ' '.join(hyphenated_words)
        
    except ImportError:
        # If hyphen package is not available, return original text
        return text
    except Exception as e:
        logger.error(f"Error hyphenating text: {str(e)}")
        return text
