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