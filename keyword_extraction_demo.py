import os
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
 
def load_corpus(corpus_dir: str) -> list:
    """加载语料库"""
    corpus = []
    print(f"正在从 {corpus_dir} 加载语料库...")
    for root, _, files in os.walk(corpus_dir):
        for file in tqdm(files, desc="加载文件"):
            if file.endswith('.md'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:  # 只添加非空文本
                            corpus.append(text)
                except Exception as e:
                    print(f"警告：无法读取文件 {file}: {str(e)}")
    return corpus

def main():
    # 示例文本
    text = """
    最高人民法院关于审理民间借贷案件适用法律若干问题的规定
    为正确审理民间借贷纠纷案件，根据《中华人民共和国民法通则》《中华人民共和国合同法》《中华人民共和国民事诉讼法》等相关法律之规定，结合审判实践，制定本规定。
    第一条 本规定所称的民间借贷，是指自然人、法人、其他组织之间及其相互之间进行资金融通的行为。
    经金融监管部门批准设立的从事贷款业务的金融机构及其分支机构，因发放贷款等相关金融业务引发的纠纷，不适用本规定。
    """
    
    print("=== 关键词提取演示 ===\n")
    print("原文：")
    print(text)
    print("\n=== 不同方法提取的关键词 ===\n")
    
    # 加载语料库
    corpus_dir = "./legal_corpus"
    if os.path.exists(corpus_dir):
        corpus = load_corpus(corpus_dir)
        print(f"已加载 {len(corpus)} 篇文档")
    else:
        print("警告：未找到语料库目录，将使用示例文本作为语料库")
        corpus = [text]
    
    try:
        # 1. TF-IDF方法
        print("\n1. TF-IDF方法：")
        tfidf_extractor = TFIDFKeywordExtractor(corpus=corpus)
        tfidf_keywords = tfidf_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(tfidf_keywords)}\n")
        
        # 2. YAKE方法
        print("2. YAKE方法：")
        yake_extractor = YAKEKeywordExtractor()
        yake_keywords = yake_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(yake_keywords)}\n")
        
        # 3. TextRank方法
        print("3. TextRank方法：")
        textrank_extractor = TextRankKeywordExtractor()
        textrank_keywords = textrank_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(textrank_keywords)}\n")
        
        # 4. RAKE方法
        print("4. RAKE方法：")
        rake_extractor = RAKEKeywordExtractor()
        rake_keywords = rake_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(rake_keywords)}\n")
        
        # 5. 基于嵌入的方法
        print("5. 基于嵌入的方法：")
        embedding_extractor = EmbeddingKeywordExtractor()
        embedding_keywords = embedding_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(embedding_keywords)}\n")
        
        # 6. MDKRank方法
        print("6. MDKRank方法：")
        mdkrank_extractor = MDKRankKeywordExtractor()
        mdkrank_keywords = mdkrank_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(mdkrank_keywords)}\n")
        
        # 7. Word2Vec方法
        print("7. Word2Vec方法：")
        word2vec_extractor = Word2VecKeywordExtractor(
            vector_size=100,  # 词向量维度
            window=5,         # 上下文窗口大小
            min_count=1,      # 最小词频
            n_clusters=10     # 聚类数量
        )
        word2vec_keywords = word2vec_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(word2vec_keywords)}\n")
        
        # 8. KeyBERT方法
        print("8. KeyBERT方法：")
        keybert_extractor = KeyBERTKeywordExtractor(
            model_name="bert-base-chinese",  # 使用中文BERT模型
            top_n=20       # 候选关键词数量
        )
        keybert_keywords = keybert_extractor.extract_keywords(text, top_k=10)
        print(f"关键词：{', '.join(keybert_keywords)}\n")
        
    except Exception as e:
        print(f"错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 