from pathlib import Path
from typing import Optional
import logging
import os
from data.loader import CachedCorpusLoader
from data.models import Question, RetrievalResult
from .enhanced import EnhancedRetriever
logger = logging.getLogger(__name__)

class DocumentRetrievalSystem:
    def __init__(self, base_path: Path, cache_dir: Optional[Path] = None):
        self.base_path = base_path
        self.retrievers = {}
        self.corpus_loader = CachedCorpusLoader(base_path, cache_dir)
        
    def initialize(self) -> None:
        """Initialize retrievers for each category"""
        categories = ['insurance', 'finance', 'faq']
        
        for category in categories:
            logger.info(f"Loading {category} documents...")
            if category in ['insurance', 'finance']:
                corpus = self.corpus_loader.load_pdf_corpus(category)
            else:  # FAQ
                corpus = self.corpus_loader.load_faq_corpus()
            self.retrievers[category] = EnhancedRetriever(corpus, category)

    def process_question(self, question: Question) -> RetrievalResult:
        """Process a single question using the appropriate retriever"""
        if question.category not in self.retrievers:
            raise ValueError(f"Invalid category: {question.category}")
            
        retrieved, scores = self.retrievers[question.category].get_combined_scores(
            question.query, question.source, question.category
        )
        
        return RetrievalResult(
            qid=question.qid,
            retrieved=retrieved,
            scores=scores,
            category=question.category,
            source_files=question.source
        )
