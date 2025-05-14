import os
import json
import datetime
from keyword_extractors import (
    TFIDFKeywordExtractor,
    YAKEKeywordExtractor,
    TextRankKeywordExtractor,
    RAKEKeywordExtractor,
    EmbeddingKeywordExtractor,
    MDKRankKeywordExtractor,
    Word2VecKeywordExtractor,
    KeyBERTKeywordExtractor
)
from tqdm import tqdm
from collections import defaultdict

def load_corpus(corpus_dir: str) -> dict:
    """加载语料库，返回文件夹名到文件内容的映射"""
    corpus = defaultdict(dict)
    print(f"正在从 {corpus_dir} 加载语料库...")
    
    # 只处理指定的文件夹
    target_folders = ["constitutional_law", "civil_law"]
    
    for root, _, files in os.walk(corpus_dir):
        folder_name = os.path.basename(root)
        if folder_name == "legal_corpus":  # 跳过根目录
            continue
            
        # 只处理指定的文件夹
        if folder_name not in target_folders:
            continue
            
        print(f"处理文件夹: {folder_name}")
        for file in tqdm(files, desc=f"加载 {folder_name} 文件夹"):
            if file.endswith('.md'):
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:  # 只添加非空文本
                            corpus[folder_name][file] = text
                except Exception as e:
                    print(f"警告：无法读取文件 {file}: {str(e)}")
    return corpus

def extract_keywords_from_text(text: str, extractors: dict) -> dict:
    """使用所有提取器从文本中提取关键词，返回并集和交集"""
    all_keywords = set()  # 所有关键词的并集
    method_keywords = {}  # 每个方法提取的关键词
    
    # 首先获取每个方法的关键词
    for name, extractor in extractors.items():
        try:
            # 提取更多关键词以增加交集的可能性
            extracted_keywords = set(extractor.extract_keywords(text, top_k=15))
            method_keywords[name] = extracted_keywords
            all_keywords.update(extracted_keywords)
        except Exception as e:
            print(f"警告：{name} 提取器出错: {str(e)}")
            method_keywords[name] = set()
    
    # 过滤掉空集，只考虑有结果的提取器
    non_empty_sets = [keywords for keywords in method_keywords.values() if keywords]
    
    # 采用多数投票机制，对多种算法的结果进行智能融合
    # 统计每个关键词被不同提取器识别的次数
    keyword_votes = {}
    for keywords in non_empty_sets:
        for keyword in keywords:
            keyword_votes[keyword] = keyword_votes.get(keyword, 0) + 1
    
    # 设置投票阈值，根据提取器数量动态调整
    total_extractors = len(non_empty_sets)
    if total_extractors >= 5:
        # 若提取器较多，设置更高阈值以保证质量
        vote_threshold = max(2, total_extractors // 3)
    else:
        # 若提取器较少，使用较低阈值以保证召回
        vote_threshold = 2 if total_extractors >= 2 else 1
    
    # 根据投票阈值筛选高置信度关键词
    common_keywords = {keyword for keyword, votes in keyword_votes.items() 
                      if votes >= vote_threshold}
    
    # 返回结果
    result = {
        "union": list(all_keywords),  # 所有关键词的并集
        "intersection": list(common_keywords),  # 通过投票机制融合的高置信度关键词
        "methods": {}  # 每种方法提取的关键词
    }
    
    # 添加每种方法提取的关键词
    for name, keywords in method_keywords.items():
        result["methods"][name] = list(keywords)
    
    return result

def main():
    # 创建输出目录
    output_dir = "./keyword_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加时间戳，避免覆盖之前的结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载语料库
    corpus_dir = "./legal_corpus"
    if not os.path.exists(corpus_dir):
        print(f"错误：未找到语料库目录 {corpus_dir}")
        return
        
    corpus = load_corpus(corpus_dir)
    if not corpus:
        print("错误：语料库为空")
        return
    
    print(f"已加载 {sum(len(files) for files in corpus.values())} 个文件，从 {len(corpus)} 个文件夹")
    for folder_name, files in corpus.items():
        print(f"  - {folder_name}: {len(files)} 个文件")
    
    # 初始化关键词提取器
    print("初始化关键词提取器...")
    extractors = {
        "TF-IDF": TFIDFKeywordExtractor(corpus=[text for files in corpus.values() for text in files.values()]),
        "YAKE": YAKEKeywordExtractor(),
        "TextRank": TextRankKeywordExtractor(),
        "RAKE": RAKEKeywordExtractor(),
        "Embedding": EmbeddingKeywordExtractor(),
        "MDKRank": MDKRankKeywordExtractor(),
        "Word2Vec": Word2VecKeywordExtractor(
            vector_size=100,
            window=5,
            min_count=1,
            n_clusters=10
        ),
        "KeyBERT": KeyBERTKeywordExtractor(
            model_name="bert-base-chinese",
            top_n=20
        )
    }
    print("关键词提取器初始化完成！")
    
    # 处理每个文件夹中的文件
    results = defaultdict(dict)
    total_files = sum(len(files) for files in corpus.values())
    processed_files = 0
    
    for folder_name, files in tqdm(corpus.items(), desc="处理文件夹"):
        print(f"\n处理文件夹：{folder_name}，共 {len(files)} 个文件")
        
        for file_name, text in tqdm(files.items(), desc=f"处理 {folder_name} 文件夹"):
            processed_files += 1
            print(f"\n[{processed_files}/{total_files}] 处理文件：{file_name}")
            
            # 提取关键词
            print("提取关键词中...")
            file_keywords = extract_keywords_from_text(text, extractors)
            
            # 存储结果
            results[folder_name][file_name] = file_keywords
            
            # 打印进度
            print(f"已提取关键词：")
            print(f"并集（所有方法提取的关键词）：{len(file_keywords['union'])}个")
            print(f"交集（常见关键词）：{len(file_keywords['intersection'])}个")
            if file_keywords['intersection']:
                print(f"  交集关键词: {', '.join(file_keywords['intersection'])}")
            
            # 打印各方法提取的关键词数量
            print("各方法提取关键词数量：")
            for method_name, keywords in file_keywords['methods'].items():
                print(f"  - {method_name}: {len(keywords)}个")
    
    # 保存结果到JSON文件
    output_file = os.path.join(output_dir, f"keyword_results_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到：{output_file}")
    print(f"总共处理了 {processed_files} 个文件，来自 {len(corpus)} 个文件夹")

if __name__ == "__main__":
    main() 