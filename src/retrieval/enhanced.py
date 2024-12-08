import json
import pickle
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import os
from preprocessing.text import TextPreprocessor
from embedding.embedder import TextEmbedder

logger = logging.getLogger(__name__)

class EnhancedRetriever:
    def __init__(self, corpus_dict: Dict[int, List[str]], category: str, cache_dir: str = "../.cache"):
        """
        Initialize retriever with corpus of chunks
        
        Args:
            corpus_dict: Dictionary mapping document IDs to lists of text chunks
            category: Document category ('finance', 'faq', 'insurance')
            cache_dir: Directory to store cached embeddings and models
        """
        self.corpus_dict = corpus_dict
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = TextPreprocessor()
        self.embedder = TextEmbedder()
        
        # Initialize or load cached data for category
        self.initialize_retrievers(category)

    def _get_corpus_hash(self) -> str:
        """Generate hash of corpus to detect changes"""
        corpus_str = json.dumps(self.corpus_dict, sort_keys=True)
        return hashlib.md5(corpus_str.encode()).hexdigest()

    def _get_cache_paths(self, category: str) -> Dict[str, Path]:
        """
        Get paths for all cached files based on category
        
        Args:
            category: Document category ('finance', 'faq', 'insurance')
        """
        category_dir = self.cache_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'embeddings': category_dir / f'{category}_chunk_embeddings.npy',
            'doc_map': category_dir / f'{category}_doc_chunk_map.json',
            'processed_chunks': category_dir / f'{category}_processed_chunks.pkl',
            'corpus_hash': category_dir / f'{category}_corpus_hash.txt',
            'tfidf': category_dir / f'{category}_tfidf.pkl',
            'bm25': category_dir / f'{category}_bm25.pkl'
        }


    def save_cache(self, category: str):
        """
        Save all model data and embeddings to cache for specific category
        
        Args:
            category: Document category to save cache for
        """
        cache_paths = self._get_cache_paths(category)
        
        try:
            # Save corpus hash
            with open(cache_paths['corpus_hash'], 'w') as f:
                f.write(self._get_corpus_hash())

            # Save embeddings as numpy array with category
            np.save(cache_paths['embeddings'], self.chunk_embeddings)
            
            # Save document to chunk mapping
            with open(cache_paths['doc_map'], 'w') as f:
                json.dump(self.doc_to_chunk_map, f)
            
            # Save processed chunks
            with open(cache_paths['processed_chunks'], 'wb') as f:
                pickle.dump(self.processed_chunks, f)

            # Save TF-IDF vectorizer
            with open(cache_paths['tfidf'], 'wb') as f:
                pickle.dump((self.tfidf, self.tfidf_matrix), f)

            # Save BM25 model
            with open(cache_paths['bm25'], 'wb') as f:
                pickle.dump((self.bm25, self.tokenized_chunks), f)

            logger.info(f"Successfully saved cache for category {category} to {self.cache_dir}")
            
        except Exception as e:
            logger.error(f"Error saving cache for category {category}: {str(e)}")
            raise

    def load_cache(self, category: str) -> bool:
        """
        Load cached data for specific category if available and valid
        
        Args:
            category: Document category to load cache for
            
        Returns:
            bool: True if cache was loaded successfully
        """
        cache_paths = self._get_cache_paths(category)
        
        # Check if all cache files exist
        if not all(path.exists() for path in cache_paths.values()):
            return False
            
        try:
            # Verify corpus hasn't changed
            with open(cache_paths['corpus_hash'], 'r') as f:
                cached_hash = f.read()
            if cached_hash != self._get_corpus_hash():
                logger.info(f"Corpus has changed for category {category}, cache invalid")
                return False

            # Load embeddings
            self.chunk_embeddings = np.load(cache_paths['embeddings'])
            
            # Load document mapping
            with open(cache_paths['doc_map'], 'r') as f:
                self.doc_to_chunk_map = json.load(f)
                # Convert string keys back to integers
                self.doc_to_chunk_map = {int(k): v for k, v in self.doc_to_chunk_map.items()}
            
            # Load processed chunks
            with open(cache_paths['processed_chunks'], 'rb') as f:
                self.processed_chunks = pickle.load(f)

            # Load TF-IDF
            with open(cache_paths['tfidf'], 'rb') as f:
                self.tfidf, self.tfidf_matrix = pickle.load(f)

            # Load BM25
            with open(cache_paths['bm25'], 'rb') as f:
                self.bm25, self.tokenized_chunks = pickle.load(f)

            logger.info(f"Successfully loaded cache for category {category} from {self.cache_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading cache for category {category}: {str(e)}")
            return False

    def initialize_retrievers(self, category: str):
        """
        Initialize or load retrieval methods for chunks of specific category
        
        Args:
            category: Document category to initialize
        """
        # Try to load from cache first
        if self.load_cache(category):
            return

        # If cache missing or invalid, process everything
        self.doc_to_chunk_map = {}
        flattened_chunks = []
        current_index = 0
        
        for doc_id, chunks in self.corpus_dict.items():
            chunk_indices = []
            for chunk in chunks:
                flattened_chunks.append(chunk)
                chunk_indices.append(current_index)
                current_index += 1
            self.doc_to_chunk_map[doc_id] = chunk_indices

        # Preprocess chunks
        self.processed_chunks = self.preprocessor.preprocess(flattened_chunks)
        
        # Initialize BM25
        self.tokenized_chunks = [text.split() for text in self.processed_chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        # Initialize TF-IDF
        self.tfidf = TfidfVectorizer(analyzer='word')
        self.tfidf_matrix = self.tfidf.fit_transform(self.processed_chunks)
        
        # Generate embeddings
        self.chunk_embeddings = self.embedder.get_embeddings(self.processed_chunks)
        
        # Save everything to cache with category
        self.save_cache(category)
        
    def get_combined_scores(self, query: str, source_files: List[str], category: str) -> Tuple[int, List[float]]:
        """
        Combine multiple ranking methods for better retrieval
        
        Args:
            query: Query text
            source_files: List of possible source document IDs
            category: Document category ('finance', 'faq', etc.)
            
        Returns:
            Tuple of (best_doc_id, scores)
        """
        # Preprocess query
        processed_query = self.preprocessor.preprocess([query])[0]
        
        # Get relevant chunk indices for source files
        source_chunk_indices = []
        for file_id in source_files:
            source_chunk_indices.extend(self.doc_to_chunk_map[int(file_id)])
        
        # Get BM25 scores for chunks
        bm25_scores = self.bm25.get_scores(processed_query.split())
        bm25_scores = bm25_scores[source_chunk_indices]
        
        # Get TF-IDF similarity scores for chunks
        query_tfidf = self.tfidf.transform([processed_query])
        tfidf_scores = cosine_similarity(query_tfidf, 
                                       self.tfidf_matrix[source_chunk_indices]).flatten()
        
        # Get embedding similarity scores for chunks
        query_embedding = self.embedder.get_embeddings([processed_query])
        embedding_scores = cosine_similarity(query_embedding, 
                                          self.chunk_embeddings[source_chunk_indices]).flatten()
        
        # Normalize scores
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        

        chunk_scores = (
            0.3 * bm25_scores + 
            0.3 * tfidf_scores +
            0.4 * embedding_scores
        )

        # Aggregate chunk scores to document scores
        doc_scores = {}
        for idx, score in enumerate(chunk_scores):
            chunk_idx = source_chunk_indices[idx]
            doc_id = next(doc_id for doc_id, indices in self.doc_to_chunk_map.items() 
                         if chunk_idx in indices)
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), score)
        
        # Get best matching document
        best_doc_id = max(doc_scores.items(), key=lambda x: x[1])[0]
        
        # Return scores for all source documents
        all_scores = [doc_scores.get(int(file_id), 0) for file_id in source_files]
        
        return best_doc_id, all_scores
