from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from typing import List
import logging
from data.models import RetrievalResult
import os

logger = logging.getLogger(__name__)

def save_results(results: List[RetrievalResult], output_path: Path) -> None:
    """Save retrieval results and scores."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save answer dictionary
    answer_dict = {
        "answers": [
            {"qid": result.qid, "retrieve": result.retrieved}
            for result in results
        ]
    }
    
    answer_file = output_path.with_name(f"{output_path.stem}_result_{timestamp}.json")
    with open(answer_file, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
    logger.info(f"Results saved to: {answer_file}")
    
    # Save scores
    score_data = [
        {
            "qid": result.qid,
            "category": result.category, 
            "source_file": source_file,
            "score": score
        }
        for result in results
        for source_file, score in zip(result.source_files, result.scores)
    ]
    
    scores_df = pd.DataFrame(score_data)
    score_file = output_path.with_name(f"{output_path.stem}_scores_{timestamp}.csv")
    scores_df.to_csv(score_file, index=False, encoding='utf8')
    logger.info(f"Scores saved to: {score_file}")