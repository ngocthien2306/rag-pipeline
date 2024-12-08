from functools import lru_cache
import logging
from hanziconv import HanziConv
import os
class ChineseConverter:
    """
    A utility class for Chinese text conversion with caching and error handling
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.logger = logging.getLogger(__name__)

    @staticmethod
    @lru_cache(maxsize=1000)
    def _traditional_to_simplified(text: str) -> str:
        """
        Convert traditional Chinese characters to simplified Chinese.
        Uses caching to improve performance for repeated conversions.
        
        Args:
            text (str): Text in traditional Chinese
            
        Returns:
            str: Text converted to simplified Chinese
        """
        try:
            # Convert to simplified Chinese using HanziConv
            simplified = HanziConv.toSimplified(text)
            
            # Additional processing for special cases
            special_chars = {
                '著': '着',  # Special case for "zhe"
                '藉': '借',  # Special case for "jie"
                '讓': '让',  # Special case for "rang"
                '佔': '占',  # Special case for "zhan"
                '隻': '只',  # Special case for "zhi"
                # Add more special cases as needed
            }
            
            for trad, simp in special_chars.items():
                simplified = simplified.replace(trad, simp)
                
            return simplified
            
        except Exception as e:
            logging.error(f"Error converting text: {str(e)}")
            return text  # Return original text if conversion fails

    @staticmethod
    def is_traditional(text: str) -> bool:
        """
        Check if the text contains traditional Chinese characters.
        
        Args:
            text (str): Input text to check
            
        Returns:
            bool: True if text contains traditional characters
        """
        try:
            return text != HanziConv.toSimplified(text)
        except:
            return False

    @staticmethod
    def contains_chinese(text: str) -> bool:
        """
        Check if the text contains Chinese characters.
        
        Args:
            text (str): Input text to check
            
        Returns:
            bool: True if text contains Chinese characters
        """
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
