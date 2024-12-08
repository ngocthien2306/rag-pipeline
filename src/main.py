import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import json

from data.models import Question
from retrieval.system import DocumentRetrievalSystem
from utils.io import save_results

def main():
    parser = argparse.ArgumentParser(description='Enhanced document retrieval system')
    parser.add_argument('--question_path', type=Path, default='/root/nguyen/research/NLP/data/dataset/preliminary/questions_example.json')
    parser.add_argument('--source_path', type=Path, default='/root/nguyen/research/NLP/data/dataset/reference') 
    parser.add_argument('--output_path', type=Path, default='/root/nguyen/research/NLP/AI_CUP/results')
    args = parser.parse_args()

    # Load questions
    with open(args.question_path, 'rb') as f:
        questions = [Question(**q) for q in json.load(f)['questions']]

    cache_dir = Path("../.cache")  # Optional
    system = DocumentRetrievalSystem(args.source_path, cache_dir)
    system.initialize()

    results = []
    for question in tqdm(questions, desc="Processing questions"):
        try:
            result = system.process_question(question)
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing question {question.qid}: {str(e)}")

    # Save results  
    save_results(results, args.output_path)

if __name__ == "__main__":
    main()