import json
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_category_accuracy(pred_retrieve_path, ground_truth_path):
    # Load data
    pred_retrieve = load_json(pred_retrieve_path)
    ground_truth = load_json(ground_truth_path)
    
    # Create dictionaries for ground truth data
    ground_truth_dict = {}
    category_dict = {}
    for item in ground_truth['ground_truths']:
        qid = item['qid']
        ground_truth_dict[qid] = item['retrieve']
        category_dict[qid] = item['category']
    
    # Initialize counters for each category
    category_totals = defaultdict(int)
    category_correct = defaultdict(int)
    
    # Compare results
    for answer in pred_retrieve['answers']:
        qid = answer['qid']
        predicted_retrieve = answer['retrieve']
        
        if qid in ground_truth_dict:
            category = category_dict[qid]
            category_totals[category] += 1
            
            if predicted_retrieve == ground_truth_dict[qid]:
                category_correct[category] += 1
    
    # Calculate accuracy for each category
    category_accuracy = {}
    for category in category_totals:
        accuracy = (category_correct[category] / category_totals[category]) * 100
        category_accuracy[category] = {
            'total': category_totals[category],
            'correct': category_correct[category],
            'accuracy': round(accuracy, 2)
        }
    
    # Calculate overall accuracy
    total_questions = sum(category_totals.values())
    total_correct = sum(category_correct.values())
    overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    
    return {
        'overall': {
            'total': total_questions,
            'correct': total_correct,
            'accuracy': round(overall_accuracy, 2)
        },
        'by_category': category_accuracy
    }

# Example usage
pred_retrieve_path = '/home/nguyen/research/NLP/AI_CUP/results_result_20241208_224013.json'
ground_truth_path = '/home/nguyen/research/NLP/data/dataset/preliminary/ground_truths_example.json'

results = calculate_category_accuracy(pred_retrieve_path, ground_truth_path)

# Print results
print("\nOverall Results:")
print(f"Total Questions: {results['overall']['total']}")
print(f"Correct Predictions: {results['overall']['correct']}")
print(f"Overall Accuracy: {results['overall']['accuracy']}%")

print("\nResults by Category:")
for category, stats in results['by_category'].items():
    print(f"\nCategory: {category}")
    print(f"Total Questions: {stats['total']}")
    print(f"Correct Predictions: {stats['correct']}")
    print(f"Accuracy: {stats['accuracy']}%")