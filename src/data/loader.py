import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging
import pdfplumber
from tqdm import tqdm
import re
from config.config import CONFIG

logger = logging.getLogger(__name__)

class CachedCorpusLoader:
    """
    A class for loading and caching processed document corpora
    """
    
    def __init__(self, base_path: Path, cache_dir: Optional[Path] = None):
        self.base_path = Path(base_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_path / '.cache'
        self.logger = logging.getLogger(__name__)
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
            self.logger.info(f"Created cache directory: {self.cache_dir}")

    def _get_cache_path(self, category: str) -> Path:
        """Generate cache file path for a category"""
        return self.cache_dir / f"{category}_corpus_cache.pkl"

    def _get_source_hash(self, directory: Path) -> str:
        """
        Calculate hash of source files to detect changes
        """
        hasher = hashlib.md5()
        
        # Get all files and their modification times
        for file_path in sorted(directory.glob('**/*')):
            if file_path.is_file() and not str(file_path).startswith('.'):
                hasher.update(str(file_path).encode())
                hasher.update(str(os.path.getmtime(file_path)).encode())

        return hasher.hexdigest()

    def _save_cache(self, category: str, corpus: Dict[int, str], source_hash: str):
        """Save processed corpus and metadata to cache"""
        cache_data = {
            'corpus': corpus,
            'source_hash': source_hash,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'  # For future compatibility
        }
        
        cache_path = self._get_cache_path(category)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            self.logger.info(f"Cached {category} corpus saved to {cache_path}")
        except Exception as e:
            self.logger.error(f"Error saving cache for {category}: {str(e)}")

    def _create_chunks(self, text: str, chunk_size: int = 5, overlap: int = 2, max_tokens: int = 512) -> List[str]:
        """
        Create overlapping chunks from text with token length control.
        
        Args:
            text: Input text to chunk
            chunk_size: Target number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
            max_tokens: Maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = re.split('[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        def count_tokens(text: str) -> int:
            """Estimate token count (rough estimation: 1 token ≈ 4 chars)"""
            return len(text) // 4
        
        def create_subchunks(long_sentence: str) -> List[str]:
            """Split a long sentence into sub-chunks based on punctuation or size"""
            # Try splitting by common Chinese punctuation first
            subsentences = re.split('[，；：、]', long_sentence)
            subsentences = [s.strip() for s in subsentences if s.strip()]
            
            # If still too long, split by size
            final_subchunks = []
            for subsent in subsentences:
                if count_tokens(subsent) > max_tokens:
                    # Split into roughly equal parts
                    chars = list(subsent)
                    target_length = max_tokens * 4
                    for i in range(0, len(chars), target_length):
                        subchunk = ''.join(chars[i:i + target_length])
                        if subchunk:
                            final_subchunks.append(subchunk)
                else:
                    final_subchunks.append(subsent)
                    
            return final_subchunks

        for sentence in sentences:
            # Check if single sentence is too long
            if count_tokens(sentence) > max_tokens:
                # If current chunk has content, save it
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    # Keep overlap sentences for next chunk
                    current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                    current_length = sum(count_tokens(s) for s in current_chunk)
                
                # Split long sentence into sub-chunks
                subchunks = create_subchunks(sentence)
                for subchunk in subchunks:
                    chunks.append(subchunk)
                continue
                
            # Check if adding sentence would exceed max_tokens
            sentence_tokens = count_tokens(sentence)
            if current_length + sentence_tokens > max_tokens:
                # Save current chunk
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    # Keep overlap sentences for next chunk
                    current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                    current_length = sum(count_tokens(s) for s in current_chunk)
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_tokens
            
            # Check if reached target chunk size
            if len(current_chunk) >= chunk_size:
                chunks.append("".join(current_chunk))
                # Keep overlap sentences for next chunk
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_length = sum(count_tokens(s) for s in current_chunk)
        
        # Add remaining sentences
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks

    def _load_cache(self, category: str) -> Optional[Dict[int, List[str]]]:
        """Load corpus from cache and split into chunks if valid"""
        cache_path = self._get_cache_path(category)
        
        if not cache_path.exists():
            return None
            
        try: 
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Get current source hash  
            source_dir = self.base_path / category
            current_hash = self._get_source_hash(source_dir)
            
            # Check if cache is valid
            if cache_data['source_hash'] == current_hash:
                self.logger.info(f"Loaded {category} corpus from cache")
                
                # Create chunks from cached documents
                chunked_corpus = {}
                for doc_id, text in tqdm(cache_data['corpus'].items(), "Processing Chunk ..."):
                    if category == 'faq':
                        chunks = self._create_chunks(
                            text,
                            chunk_size=CONFIG['preprocessing'][category]['chunk_size'], 
                            overlap=CONFIG['preprocessing'][category]['overlap']  
                        )
                    elif category == 'finance':
                        chunks = self._create_chunks(
                            text,
                            chunk_size=CONFIG['preprocessing'][category]['chunk_size'],
                            overlap=CONFIG['preprocessing'][category]['overlap']    
                        )
                    else:
                        chunks = self._create_chunks(
                            text,
                            chunk_size=CONFIG['preprocessing'][category]['chunk_size'],
                            overlap=CONFIG['preprocessing'][category]['overlap']   
                        )
                    chunked_corpus[doc_id] = chunks
                    
                return chunked_corpus
            else:
                self.logger.info(f"Cache invalid for {category} - source files changed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading cache for {category}: {str(e)}")
            return None

    def load_pdf_corpus(self, category: str) -> Dict[int, str]:
        """
        Load PDF corpus with caching
        """
        # Try to load from cache first
        cached_corpus = self._load_cache(category)
        if cached_corpus is not None:
            return cached_corpus

            
        # If cache miss or invalid, process files
        self.logger.info(f"Processing {category} PDFs...")
        corpus_path = self.base_path / category
        corpus_dict = {}
        
        try:
            for pdf_file in tqdm(list(corpus_path.glob('*.pdf')), 
                                desc=f"Loading {category} PDFs"):
                try:
                    file_id = int(pdf_file.stem)
                    with pdfplumber.open(pdf_file) as pdf:
                        text = ' '.join(page.extract_text() or '' for page in pdf.pages)
                    corpus_dict[file_id] = text
                except Exception as e:
                    self.logger.error(f"Error loading {pdf_file}: {str(e)}")
                    
            # Save to cache
            source_hash = self._get_source_hash(corpus_path)
            self._save_cache(category, corpus_dict, source_hash)
            
            return corpus_dict
            
        except Exception as e:
            self.logger.error(f"Error processing {category} corpus: {str(e)}")
            raise

    def load_faq_corpus(self) -> Dict[int, str]:
        """
        Load FAQ corpus with caching
        """
        # Try to load from cache first
        cached_corpus = self._load_cache('faq')
        if cached_corpus is not None:
            return cached_corpus
            
        # If cache miss or invalid, process files
        self.logger.info("Processing FAQ corpus...")
        try:
            faq_path = self.base_path / 'faq/pid_map_content.json'
            with open(faq_path, 'rb') as f:
                corpus_dict = {int(k): str(v) for k, v in json.load(f).items()}
                
            # Save to cache
            source_hash = self._get_source_hash(self.base_path / 'faq')
            self._save_cache('faq', corpus_dict, source_hash)
            
            return corpus_dict
            
        except Exception as e:
            self.logger.error(f"Error processing FAQ corpus: {str(e)}")
            raise

    def clear_cache(self, category: Optional[str] = None):
        """
        Clear cache for specific category or all categories
        """
        try:
            if category:
                cache_path = self._get_cache_path(category)
                if cache_path.exists():
                    cache_path.unlink()
                    self.logger.info(f"Cleared cache for {category}")
            else:
                for cache_file in self.cache_dir.glob('*_corpus_cache.pkl'):
                    cache_file.unlink()
                self.logger.info("Cleared all cache files")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
