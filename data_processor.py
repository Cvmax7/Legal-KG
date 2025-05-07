import os
import jieba
import json
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import jieba.posseg as pseg
from tqdm import tqdm


class LegalDataProcessor:
    """统一的法律文本数据处理器"""

    def __init__(self):
        # 修改领域映射关系
        self.domain_mapping = {
            "commercial": "commercial_law",  # 商法
            "civil": "civil_law",           # 民法
            "criminal": "criminal_law",      # 刑法
            "administrative": "administrative_law",  # 行政法
            "procedure": "procedure_law",    # 诉讼法
            "constitutional": "constitutional_law",  # 宪法
            "economic": "economic_law",      # 经济法
            "social": "social_law",          # 社会法
        }
        
        # 可能需要合并处理的相关目录
        self.related_dirs = {
            "administrative": ["administrative_regulation"],  # 行政法相关的行政法规
            "constitutional": ["constitutional_related_law"], # 宪法相关法律
        }
        
        # 初始化jieba
        jieba.setLogLevel(20)
        
        # 加载停用词和法律词典
        self.stopwords = self._load_stopwords()
        self.legal_terms = self._load_legal_terms()
        
        # 将法律术语添加到jieba词典
        for term in self.legal_terms:
            jieba.add_word(term)

    def _load_stopwords(self):
        """加载停用词"""
        stopwords = set()
        stopwords_path = "./Config/stopwords/stopwords"
        if os.path.exists(stopwords_path):
            with open(stopwords_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
        return stopwords

    def _load_legal_terms(self):
        """加载法律词典"""
        terms = set()
        dict_path = "./Config/dict/legal_terms.txt"
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    term = line.strip()
                    if term and not term.startswith("#"):
                        terms.add(term)
        return terms

    def extract_keywords_from_text(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        
        # 收集候选关键词
        keywords = []
        for word, flag in words:
            # 选择名词、动词等可能的关键词
            if len(word) >= 2 and word not in self.stopwords:
                if (flag.startswith('n') or  # 名词
                    flag.startswith('v') or  # 动词
                    word in self.legal_terms):  # 法律术语
                    keywords.append(word)
        
        # 去重并限制数量
        keywords = list(set(keywords))[:100]  # 限制每个文档最多100个关键词
        return keywords

    def load_data(self):
        """加载数据并按领域组织"""
        all_data = {domain: {"texts": [], "keywords": []} 
                   for domain in self.domain_mapping.keys()}
        
        print("\n=== 开始加载法律语料库 ===")
        base_path = "./legal_corpus"
        
        # 读取每个领域的数据
        for domain, main_folder in self.domain_mapping.items():
            print(f"\n处理 {domain} 领域文档...")
            
            # 获取需要处理的所有目录
            folders_to_process = [main_folder]
            if domain in self.related_dirs:
                folders_to_process.extend(self.related_dirs[domain])
            
            total_files = 0
            for folder in folders_to_process:
                folder_path = os.path.join(base_path, folder)
                if not os.path.exists(folder_path):
                    print(f"警告: {folder_path} 目录不存在，已跳过")
                    continue
                
                files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
                print(f"在 {folder} 中发现 {len(files)} 个文件")
                
                for file in tqdm(files, desc=f"处理{folder}文件"):
                    try:
                        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                            text = f.read().strip()
                            # 跳过空文件
                            if not text:
                                continue
                            keywords = self.extract_keywords_from_text(text)
                            all_data[domain]["texts"].append(text)
                            all_data[domain]["keywords"].append(keywords)
                            total_files += 1
                    except Exception as e:
                        print(f"处理文件 {file} 时出错: {str(e)}")
                        continue
            
            print(f"共处理 {total_files} 个文件")
            
        # 打印统计信息
        print("\n=== 数据集统计 ===")
        total_docs = 0
        for domain, data in all_data.items():
            n_docs = len(data["texts"])
            total_docs += n_docs
            if n_docs > 0:
                avg_len = sum(len(t) for t in data["texts"]) / n_docs
                avg_kw = sum(len(k) for k in data["keywords"]) / n_docs
                print(f"\n{domain} 领域:")
                print(f"文档数: {n_docs}")
                print(f"平均文档长度: {avg_len:.0f} 字符")
                print(f"平均关键词数: {avg_kw:.1f}")
        
        print(f"\n总文档数: {total_docs}")
        return all_data

    def prepare_domain_data(self, domain_data, train_ratio=0.7, val_ratio=0.15):
        """为单个领域准备数据"""
        texts = domain_data["texts"]
        keywords = domain_data["keywords"]
        
        if not texts:  # 如果没有数据，返回空的数据集
            return {
                "train": {"texts": [], "keywords": []},
                "val": {"texts": [], "keywords": []},
                "test": {"texts": [], "keywords": []}
            }
        
        # 首先划分训练集
        train_texts, temp_texts, train_keywords, temp_keywords = train_test_split(
            texts, keywords, 
            test_size=(1-train_ratio),
            random_state=42,
            shuffle=True
        )
        
        # 然后划分验证集和测试集
        val_size = val_ratio / (1-train_ratio)
        val_texts, test_texts, val_keywords, test_keywords = train_test_split(
            temp_texts, temp_keywords,
            test_size=(1-val_size),
            random_state=42,
            shuffle=True
        )
        
        return {
            "train": {"texts": train_texts, "keywords": train_keywords},
            "val": {"texts": val_texts, "keywords": val_keywords},
            "test": {"texts": test_texts, "keywords": test_keywords}
        }

    def save_processed_data(self, data: Dict, output_dir: str = "./data/processed"):
        """保存处理后的数据集"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存每个领域的数据
        for domain, splits in data.items():
            domain_dir = os.path.join(output_dir, domain)
            if not os.path.exists(domain_dir):
                os.makedirs(domain_dir)
                
            for split_name, split_data in splits.items():
                split_path = os.path.join(domain_dir, f"{split_name}_data.json")
                with open(split_path, "w", encoding="utf-8") as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=2)

        print(f"处理后的数据集已保存到: {output_dir}")
