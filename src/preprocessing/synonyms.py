import logging
from typing import Dict, Set
from collections import defaultdict
from gensim.models import KeyedVectors
import os
import jieba

class SynonymExpander:
    """
    A class for expanding Chinese words with their synonyms using Word2Vec
    and custom dictionary
    """
    
    def __init__(self, threshold: float = 0.75):
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.word2vec = None
        self.custom_synonyms = self._load_custom_synonyms()
        self.domain_specific = self._load_domain_specific()
        self._initialize_word2vec()

    def _initialize_word2vec(self):
        """Initialize Word2Vec model"""
        try:
            # You can download a pre-trained Chinese word2vec model
            # For example: https://github.com/Embedding/Chinese-Word-Vectors
            model_path = "path_to_your_word2vec_model.txt"  # Change this to your model path
            if os.path.exists(model_path):
                self.word2vec = KeyedVectors.load_word2vec_format(model_path, binary=False)
                self.logger.info("Word2Vec model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load Word2Vec model: {str(e)}")

    def _load_custom_synonyms(self) -> Dict[str, Set[str]]:
        """Load custom synonym dictionary"""
        custom_synonyms = defaultdict(set)
        
        # Basic financial terms
        financial_terms = {
            "股票": {"证券", "股份", "股权", "个股", "股本", "筹码"},
            "投资": {"理财", "投入", "出资", "资金投放", "融资", "募资"},
            "收益": {"回报", "利润", "盈利", "收入", "盈余", "利益", "报酬"},
            "风险": {"危险", "隐患", "风险性", "不确定性", "波动", "损失可能"},
            "市场": {"市面", "行情", "市况", "交易市场", "金融市场", "资本市场"},
            "基金": {"基金产品", "公募基金", "资金", "私募基金", "投资基金", "互惠基金"},
            "证券": {"股票", "债券", "有价证券", "证券产品", "金融证券", "市场证券"},
            "交易": {"买卖", "成交", "交易额", "市场交易", "证券交易", "期货交易"},
            "股价": {"股票价格", "市价", "价位", "交易价格", "市场价格", "证券价格"},
            "银行": {"银行机构", "金融机构", "银行业", "商业银行", "储蓄银行", "投资银行"},
            
            # Additional Market Terms
            "期货": {"期货合约", "商品期货", "金融期货", "衍生品", "期货交易"},
            "债券": {"公司债", "国债", "地方债", "企业债", "可转债", "债券市场"},
            "指数": {"股指", "市场指数", "指数点位", "大盘指数", "板块指数"},
            "波动": {"市场波动", "价格波动", "震荡", "涨跌", "波动率"},
            "趋势": {"市场趋势", "走势", "行情走向", "发展方向", "市场方向"},
            
            # Investment Terms
            "投资组合": {"资产配置", "投资搭配", "组合投资", "资金组合"},
            "分散投资": {"分散风险", "多元化投资", "投资多样化", "风险分散"},
            "长期投资": {"价值投资", "战略投资", "持有投资", "长线投资"},
            "短期投资": {"投机", "短线交易", "快速交易", "短线投资"},
            "回报率": {"投资回报", "收益率", "投资收益", "回报比率"},
            
            # Risk Management Terms
            "风险管理": {"风险控制", "风险防范", "风险规避", "风险评估"},
            "止损": {"损失控制", "风险止损", "止损点", "止损限制"},
            "杠杆": {"财务杠杆", "借贷", "融资融券", "杠杆率"},
            "流动性": {"市场流动性", "资金流动性", "变现能力", "流动性风险"},
            "对冲": {"风险对冲", "套期保值", "对冲策略", "风险对冲"},
            
            # Trading Terms
            "做多": {"买入持有", "看涨", "多头", "做多策略"},
            "做空": {"卖空", "看跌", "空头", "做空策略"},
            "成交量": {"交易量", "市场成交", "成交额", "交易规模"},
            "委托": {"交易委托", "买卖委托", "委托单", "交易指令"},
            "盘面": {"市场盘面", "交易盘面", "盘势", "盘口"},
            
            # Market Analysis Terms
            "技术分析": {"图表分析", "行情分析", "技术指标", "走势分析"},
            "基本面": {"基本分析", "基础分析", "公司基本面", "市场基本面"},
            "估值": {"价值评估", "市场估值", "估价", "评估价值"},
            "趋势线": {"支撑线", "压力线", "趋势通道", "技术线"},
            "量价": {"量价关系", "成交量价", "量价分析", "交易量价"}
        }
        
        # Insurance terms
        insurance_terms = {
            # Basic Insurance Concepts
            "保险": {"保障", "保单", "保费", "保险产品", "保险服务", "保险计划"},
            "理赔": {"赔付", "赔偿", "理赔金", "保险赔付", "损失赔偿", "理赔服务"},
            "保费": {"保险费", "缴费", "保险金", "保费金额", "保险费用", "缴费金额"},
            "保单": {"保险合同", "合约", "保险单", "保险协议", "保障合同", "保险证明"},
            "投保": {"购买保险", "参保", "保险投保", "投保申请", "保险申请", "投保手续"},
            "保障": {"保护", "担保", "保险保障", "风险保障", "保障范围", "保险保护"},
            
            # Insurance Types
            "寿险": {"人寿保险", "生命保险", "人身保险", "死亡保险", "终身寿险"},
            "健康险": {"医疗保险", "疾病保险", "医保", "健康保障", "医疗保障"},
            "意外险": {"意外保险", "人身意外", "意外伤害", "意外保障", "意外理赔"},
            "财产险": {"财险", "物品保险", "资产保险", "财产保障", "财产理赔"},
            "车险": {"汽车保险", "机动车险", "交通保险", "车辆保险", "汽车理赔"},
            
            # Claims Process
            "索赔": {"申请理赔", "理赔申请", "赔偿申请", "索赔程序", "理赔流程"},
            "理赔调查": {"损失评估", "理赔评估", "索赔调查", "赔付调查", "理赔勘察"},
            "理赔材料": {"索赔资料", "理赔文件", "索赔证明", "理赔单证", "理赔凭证"},
            "理赔金额": {"赔付金额", "赔偿金额", "理赔费用", "保险赔款", "赔付款项"},
            "拒赔": {"拒绝理赔", "拒绝赔付", "拒赔原因", "不予理赔", "理赔拒绝"},
            
            # Policy Terms
            "承保": {"保险承保", "承保范围", "承保风险", "承保条件", "保险责任"},
            "免赔": {"免赔额", "自付", "免赔率", "免赔金额", "自付额度"},
            "等待期": {"观察期", "等候期", "保险等待", "等待时间", "等待周期"},
            "保险期限": {"保障期限", "保险有效期", "保险期间", "保障时间", "保险时效"},
            "续保": {"继续投保", "保险续期", "续约", "保单续保", "保险延期"},
            
            # Insurance Beneficiaries
            "受益人": {"保险金受益人", "受益者", "受益对象", "保险受益", "理赔受益"},
            "第三者": {"第三方", "涉外方", "相关方", "第三者责任", "第三方责任"},
            "被保险人": {"投保对象", "保险对象", "受保人", "被保障人", "保险客体"},
            "投保人": {"保险购买人", "保单持有人", "保险合同人", "投保主体", "保险申请人"},
            "保险人": {"保险公司", "承保人", "保险机构", "承保方", "保险提供方"},
            
            # Additional Terms
            "保险责任": {"保障责任", "承保责任", "保险义务", "保障义务", "保险保障责任"},
            "除外责任": {"责任免除", "除外风险", "免责事项", "保险除外", "责任豁免"},
            "保险利益": {"可保利益", "保险权益", "保险收益", "保险价值", "保险好处"},
            "保险标的": {"保险物", "承保标的", "保险客体", "承保对象", "保险目标"},
            "保险事故": {"保险案件", "理赔事件", "保险灾害", "保险意外", "理赔案件"}
        }
        # Add all terms to custom_synonyms
        for terms_dict in [financial_terms, insurance_terms]:
            for word, synonyms in terms_dict.items():
                custom_synonyms[word].update(synonyms)
                # Add reverse mappings
                for synonym in synonyms:
                    custom_synonyms[synonym].add(word)

        return custom_synonyms

    def _load_domain_specific(self) -> Dict[str, Set[str]]:
        """Load domain-specific terminology mappings"""
        return {
            # Financial specific mappings
            "牛市": {"上涨市场", "多头市场"},
            "熊市": {"下跌市场", "空头市场"},
            "止损": {"停损", "损失控制"},
            "做多": {"看涨", "买入持有"},
            "做空": {"看跌", "卖出"},
            
            # Insurance specific mappings
            "身故": {"死亡", "逝世"},
            "伤残": {"残疾", "残障"},
            "保额": {"保险金额", "承保金额"},
            "出险": {"发生保险事故", "报案"},
            "续期": {"续保", "继续投保"}
        }

    def get_synonyms(self, word: str, top_n: int = 3) -> Set[str]:
        """
        Get synonyms for a word using multiple methods
        """
        synonyms = set()
        
        # Check custom dictionary first
        if word in self.custom_synonyms:
            synonyms.update(self.custom_synonyms[word])
        
        # Check domain-specific terms
        if word in self.domain_specific:
            synonyms.update(self.domain_specific[word])
        
        # Use Word2Vec if available
        if self.word2vec and word in self.word2vec.key_to_index:
            try:
                similar_words = self.word2vec.most_similar(word, topn=top_n)
                for similar_word, score in similar_words:
                    if score >= self.threshold:
                        synonyms.add(similar_word)
            except Exception as e:
                self.logger.debug(f"Word2Vec lookup failed for {word}: {str(e)}")
        
        return synonyms

    def expand_text(self, text: str, max_expansions: int = 2) -> str:
        """
        Expand text with synonyms
        """
        if not text:
            return ""
            
        # Tokenize text
        words = list(jieba.cut_for_search(text))
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            
            # Get synonyms for words longer than 1 character
            if len(word) > 1:
                synonyms = self.get_synonyms(word)
                # Add top synonyms up to max_expansions
                expanded_words.extend(list(synonyms)[:max_expansions])
        
        return " ".join(expanded_words)
