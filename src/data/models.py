from dataclasses import dataclass
from typing import List

@dataclass
class Question:
    qid: str
    query: str  
    category: str
    source: List[str]

@dataclass  
class RetrievalResult:
    qid: str
    retrieved: int
    scores: List[float]
    category: str
    source_files: List[int]