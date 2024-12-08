import os
import sys
sys.path.append(os.getcwd())
from src.preprocessing.chinese import ChineseConverter

def test_traditional_to_simplified():
    converter = ChineseConverter()
    
    print(converter._traditional_to_simplified("本保險契約包含哪些構成部分？"))
    print(converter._traditional_to_simplified("要保人隨時終止本契約時，其終止生效的時間是何時?"))
    print(converter._traditional_to_simplified("要保人或受益人應於知悉本公司應負保險責任之事故後幾日內通知本公司？"))

def test_is_traditional():
    converter = ChineseConverter()
    print(converter.is_traditional("這是繁體字"))
    print(converter.is_traditional("这是简体字"))
    print(converter.is_traditional("English text"))

    assert converter.is_traditional("這是繁體字") == True
    assert converter.is_traditional("这是简体字") == False
    assert converter.is_traditional("English text") == False

if __name__ == "__main__":
    test_traditional_to_simplified()
    test_is_traditional()
