import torch
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
        
    @torch.no_grad()
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        return embeddings