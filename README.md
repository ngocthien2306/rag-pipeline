# Chinese Document Retrieval System

## Overview

This project implements a sophisticated document retrieval system specifically designed for Chinese language documents. It combines multiple retrieval methods including BM25, TF-IDF, and neural embeddings to effectively search and retrieve relevant documents from a mixed corpus of financial reports, insurance documents, and FAQs.

## Key Features

- Multiple retrieval methods (BM25, TF-IDF, Neural embeddings)  
- Support for traditional and simplified Chinese text
- Intelligent text preprocessing and chunking
- Synonym expansion for better semantic matching
- Cache system for improved performance
- Comprehensive testing suite
- Support for multiple document formats (PDF, JSON)

## Project Structure
```
project_root/
├── src/
│   ├── config/          # Configuration files
│   ├── data/            # Data loading and models
│   ├── preprocessing/   # Text preprocessing 
│   ├── embedding/       # Text embedding
│   ├── retrieval/       # Retrieval system
│   └── utils/           # Utility functions
├── tests/               # Test files
├── requirements.txt
└── README.md
```

## Requirements

### Hardware Requirements

- RAM: Minimum 16GB recommended
- GPU: NVIDIA GPU with at least 8GB VRAM (for embedding model)
- Storage: 10GB minimum for models and cache

### Software Requirements

- Python 3.8 or higher
- CUDA 11.4 or higher (for GPU support)
- Ubuntu 20.04 or higher (recommended)

### Dependencies

```txt
pdfplumber==0.10.3
tqdm==4.66.2
sentence-transformers==2.5.1
numpy>=1.24.0
pandas>=2.0.0
jieba==0.42.1
scikit-learn==1.4.0
rank_bm25==0.2.2
torch>=2.0.0
transformers>=4.36.0
cn2an==0.5.22
hanziconv==0.3.2
gensim>=4.3.0
pytest>=7.4.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ngocthien2306/rag-pipeline.git
cd crag-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Prepare your directory structure:
```bash
mkdir -p data/{finance,insurance,faq}
```

2. Place your documents in appropriate directories:
- PDF financial documents in `data/finance/`
- PDF insurance documents in `data/insurance/`
- FAQ JSON file in `data/faq/`

3. FAQ JSON format example:
```json
{
    "1": [
        {
            "question": "问题1",
            "answers": ["答案1"]
        }
    ],
    "2": [
        {
            "question": "问题2", 
            "answers": ["答案2"]
        }
    ]
}
```

## Usage

### Basic Usage

```python
from pathlib import Path
from src.retrieval.system import DocumentRetrievalSystem
from src.data.models import Question

# Initialize system
system = DocumentRetrievalSystem(
    base_path=Path("data"),
    cache_dir=Path(".cache")
)
system.initialize()

# Create a question
question = Question(
    qid="1",
    query="投资理财相关的问题",
    category="finance", 
    source=["1", "2", "3"]
)

# Get retrieval result
result = system.process_question(question)
print(f"Retrieved document: {result.retrieved}")
print(f"Confidence scores: {result.scores}")
```

### Command Line Usage

Run the main script:
```bash
python src/main.py \
    --question_path data/questions.json \
    --source_path data/reference \
    --output_path results/output
```

### Running Tests

Run all tests:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chinese_converter.py

# Run with coverage report
pytest --cov=src tests/
```

## Configuration

Modify configuration in `src/config/config.py`:

```python
CONFIG = {
    # Embedding model settings
    'embedding': {
        'model_name': 'BAAI/bge-large-zh-v1.5',
        'batch_size': 128,
        'max_length': 512
    },
    
    # Cache settings
    'cache': {
        'enabled': True,
        'dir': '.cache',
        'max_size': '10GB'
    },
    
    # Preprocessing settings 
    'preprocessing': {
        'finance': {
            'chunk_size': 20,
            'overlap': 2
        },
        'faq': {
            'chunk_size': 30,
            'overlap': 2
        },
        'insurance': {
            'chunk_size': 50,
            'overlap': 2
        },
        'remove_english': True,
        'expand_synonyms': True,
        'remove_numbers': False,
        'remove_punctuation': True,
        'convert_numbers': True,
        'normalize_chinese': True,
    }
    
}
```

## Example

Input question format (`questions.json`):
```json
{
    "questions": [
        {
            "qid": "1",
            "query": "关于投资的问题",
            "category": "finance",
            "source": ["1", "2", "3"]
        }
    ]
}
```

Output format:
```json
{
    "answers": [
        {
            "qid": "1", 
            "retrieve": 2
        }
    ]
}
```

Score output format (`scores.csv`):
```csv
qid,category,source_file,score
1,finance,1,0.85
1,finance,2,0.92
1,finance,3,0.78
```

## Performance Optimization

1. Enable GPU acceleration:
```python
import torch
from src.embedding.embedder import TextEmbedder

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize embedder with GPU
embedder = TextEmbedder(device=device)
```

2. Use caching:
```python
# Enable caching in system
system = DocumentRetrievalSystem(
    base_path=Path("data"),
    cache_dir=Path(".cache")
)

# Clear cache if needed
system.corpus_loader.clear_cache()
```

## Troubleshooting

1. CUDA Out of Memory
```python
# Reduce batch size
embedder = TextEmbedder(batch_size=32)  # Default is 128

# Use smaller chunks
preprocessor = TextPreprocessor(chunk_size=3)  # Default is 5
```

2. Slow Processing 
```python
# Increase chunk size for faster processing
preprocessor = TextPreprocessor(
    chunk_size=10,
    overlap=1
)

# Enable multiprocessing
system.initialize(num_workers=4)
```

3. PDF Extraction Issues
```python
# Custom PDF extraction
import pdfplumber

def custom_pdf_reader(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            try:
                text += page.extract_text() or ""
            except Exception as e:
                print(f"Error on page {page.page_number}: {e}")
        return text
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please create an issue in the repository or contact:
- Email: ngocthien.dev23@gmail.com
- GitHub: @ngocthien23

## Acknowledgments

- BAAI for the BGE embedding model
- Hugging Face for transformer models 
- PyMuPDF team for PDF processing capabilities