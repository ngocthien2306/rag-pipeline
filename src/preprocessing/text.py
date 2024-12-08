import re
import jieba
import logging
from typing import List, Dict
import cn2an
import os
from .chinese import ChineseConverter
from .synonyms import SynonymExpander
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(self._load_stopwords())
        self.chinese_converter = ChineseConverter()
        self.synonym_expander = SynonymExpander()
        self.patterns = self._compile_patterns()
        jieba.initialize()

    @staticmethod
    def _load_stopwords():
        # Add your stopwords file path here
        stopwords_file = "../assets/path_to_stopwords.txt"
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        except FileNotFoundError:
            logger.warning("Stopwords file not found. Using empty stopwords list.")
            return []

    def _compile_patterns(self):
        """Compile regex patterns once during initialization"""
        return {
            # Email pattern
            'email': re.compile(r'\S+@\S+\.\S+'),
            
            # URL pattern 
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            
            # Numbers pattern (includes decimals)
            'numbers': re.compile(r'\d+\.?\d*'),
            
            # English words pattern
            'english': re.compile(r'[a-zA-Z]+'),
            
            # Special characters pattern 
            'special_chars': re.compile(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,!?，。！？、:：()（）《》""0-9a-zA-Z\s]'),
            
            # Multiple spaces
            'spaces': re.compile(r'\s+'),
            
            # HTML tags
            'html': re.compile(r'<[^>]+>'),
            
            # Chinese punctuation
            'cn_punc': re.compile(r'[，。！？、：；''""（）【】《》]'),
            
            # English punctuation
            'en_punc': re.compile(r'[,.!?:;\'"()\[\]<>]'),
            
            # Duplicate punctuation
            'dup_punc': re.compile(r'([,.!?:;])\1+'),
            
            # Phone numbers
            'phone': re.compile(r'\d{3}[-.]?\d{3,4}[-.]?\d{4}'),
            
            # Dates
            'date': re.compile(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?'),
            
            # Currency symbols
            'currency': re.compile(r'[$¥€£]'),
            
            # Emoji and special symbols
            'emoji': re.compile(r'[\U0001F300-\U0001F9FF]'),
        }

    def preprocess(self, chunks: List[str], config: Dict = None) -> List[str]:
        """
        Preprocess a list of text chunks
        
        Args:
            chunks: List of text chunks
            config: Dictionary of preprocessing options
                - remove_english: bool (default=False)
                - remove_numbers: bool (default=False) 
                - remove_punctuation: bool (default=False)
                - convert_numbers: bool (default=True)
                - expand_synonyms: bool (default=False)
                - normalize_chinese: bool (default=True)
        """
        if not chunks:
            return []

        # Default config optimized for chunks
        
        default_config = {
            'remove_english': True,
            'remove_numbers': False,
            'remove_punctuation': True,
            'convert_numbers': True,
            'expand_synonyms': True,
            'normalize_chinese': True,
        }

        if config:
            default_config.update(config)
        config = default_config

        processed_chunks = []

        for chunk in chunks:
            if not chunk:
                continue

            try:
                # Core preprocessing steps
                text = chunk
                
                # Remove HTML and unwanted elements
                text = self.patterns['html'].sub(' ', text)
                text = self.patterns['url'].sub(' ', text)
                text = self.patterns['email'].sub(' ', text)
                text = self.patterns['emoji'].sub(' ', text)

                # Convert traditional to simplified Chinese
                if config['normalize_chinese'] and self.chinese_converter.contains_chinese(text):
                    text = self.chinese_converter._traditional_to_simplified(text)

                # Handle numbers
                if config['remove_numbers']:
                    text = self.patterns['numbers'].sub(' ', text)
                elif config['convert_numbers']:
                    text = self._normalize_numbers(text)

                # Handle punctuation
                if config['remove_punctuation']:
                    text = self.patterns['cn_punc'].sub(' ', text)
                    text = self.patterns['en_punc'].sub(' ', text)
                else:
                    # Just normalize duplicate punctuation
                    text = self.patterns['dup_punc'].sub(r'\1', text)

                # Handle English
                if config['remove_english']:
                    text = self.patterns['english'].sub(' ', text)

                # Clean and normalize
                text = self.patterns['special_chars'].sub(' ', text)
                text = self.patterns['spaces'].sub(' ', text)
                text = re.sub(r'(\d+)', r' \1 ', text)

                # Optional synonym expansion
                if config['expand_synonyms']:
                    text = self.synonym_expander.expand_text(text)

                # Tokenization while preserving structure
                tokens = jieba.cut(text, cut_all=False)  # Use precise mode
                
                # Keep important words even if they're stopwords
                important_words = {'是', '的', '了', '在', '有', '和', '与', '或'}
                tokens = [t for t in tokens if t.strip() and (
                    t not in self.stopwords or t in important_words
                )]

                processed_text = ' '.join(tokens).strip()
                
                if processed_text:
                    processed_chunks.append(processed_text)

            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                # Keep original chunk if processing fails
                processed_chunks.append(chunk)

        return processed_chunks



    def _normalize_dates(self, text: str) -> str:
        """Convert various date formats to standard format"""
        def date_replacer(match):
            date_str = match.group(0)
            # Remove Chinese characters and standardize separators
            date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
            return date_str
            
        return self.patterns['date'].sub(date_replacer, text)
        
    def contains_sensitive_content(self, text: str) -> bool:
        """Check for potentially sensitive content"""
        sensitive_patterns = {
            'id_card': re.compile(r'\d{17}[\dXx]'),
            'bank_card': re.compile(r'\d{16,19}'),
            'secret_key': re.compile(r'[a-zA-Z0-9]{32,}'),
        }
        
        for pattern in sensitive_patterns.values():
            if pattern.search(text):
                return True
        return False

    def mask_sensitive_info(self, text: str) -> str:
        """Mask sensitive information in text"""
        # Mask email addresses
        text = self.patterns['email'].sub('[EMAIL]', text)
        
        # Mask phone numbers
        text = self.patterns['phone'].sub('[PHONE]', text)
        
        # Mask numbers that look like ID cards
        text = re.sub(r'\d{17}[\dXx]', '[ID_CARD]', text)
        
        # Mask numbers that look like bank cards
        text = re.sub(r'\d{16,19}', '[BANK_CARD]', text)
        
        return text

    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract dates from text in various formats"""
        date_patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?',  # YYYY-MM-DD
            r'\d{1,2}[-/月]\d{1,2}[日]?',             # MM-DD
            r'[今明后前昨][天日年月]',                  # Relative dates
            r'[上下这下个]个?[周月季年]',              # Relative periods
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        return dates

    @staticmethod
    def _traditional_to_simplified(text: str) -> str:
        """Convert traditional Chinese characters to simplified Chinese"""
        # Implement conversion using appropriate library
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
            }
            
            for trad, simp in special_chars.items():
                simplified = simplified.replace(trad, simp)
                
            return simplified
            
        except Exception as e:
            logging.error(f"Error converting text: {str(e)}")
            return text  # Return original text if conversion fails
        return text  # Placeholder for actual implementation

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        """Convert Chinese numbers to Arabic numbers"""
        try:
            return cn2an.transform(text, "cn2an")
        except:
            return text

    def _expand_synonyms(self, tokens: List[str]) -> List[str]:
        """Add synonyms for important terms"""
        expanded = []
        for token in tokens:
            expanded.append(token)
            try:
                syns = synonyms.nearby(token)[0][:2]  # Get top 2 synonyms
                expanded.extend(syns)
            except:
                continue
        return expanded
