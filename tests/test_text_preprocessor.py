import os
import sys
sys.path.append(os.getcwd())
from src.preprocessing.text import TextPreprocessor

def test_preprocess_basic(preprocessor, sample_texts):
    processed = preprocessor.preprocess(sample_texts)
    
    
    processed_no_english = preprocessor.preprocess(
        sample_texts, 
        config={'remove_english': True}
    )

    print("test_preprocess_basic: ", processed_no_english)

def test_normalize_numbers(preprocessor):
    text = "一百二十三"
    result = preprocessor._normalize_numbers(text)
    print("test_normalize_numbers: ", result)


def test_mask_sensitive_info(preprocessor):
    text = "邮箱是test@example.com，电话是123-456-7890"
    masked = preprocessor.mask_sensitive_info(text)
    print("test_mask_sensitive_info: ", masked)
    


def test_extract_dates(preprocessor):
    text = "日期是2024年3月15日，另一个日期是2024-03-15"
    dates = preprocessor.extract_dates(text)
    print("test_extract_dates: ", dates)
    

def sample_texts():
    """Sample texts for testing"""
    return [
        "这是第一个测试文本。",
        "This is an English text.",
        "這是繁體中文文本。",
        "包含数字123和符号!@#的文本"
    ]

if __name__ == "__main__":
    texts = sample_texts()
    preprocessor = TextPreprocessor()

    test_preprocess_basic(preprocessor, texts)
    test_normalize_numbers(preprocessor)
    test_mask_sensitive_info(preprocessor)
    test_extract_dates(preprocessor)
