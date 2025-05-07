import numpy as np
from typing import List, Dict, Set, Tuple
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
import re
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import math
import os
from summa import keywords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from keybert import KeyBERT
from keybert.backend import SentenceTransformerBackend

class BaseKeywordExtractor:
    """关键词提取器基类"""
    def __init__(self):
        self.stopwords = self._load_stopwords()
        
    def _load_stopwords(self) -> Set[str]:
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
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词的抽象方法"""
        raise NotImplementedError

class TFIDFKeywordExtractor(BaseKeywordExtractor):
    """基于TF-IDF的关键词提取器"""
    def __init__(self, corpus: List[str] = None):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: [w for w in jieba.cut(x) 
                               if w not in self.stopwords 
                               and len(w) >= 2 
                               and not re.match(r'[^\w\s]', w)],  # 过滤标点符号
            max_features=1000
        )
        if corpus:
            self.vectorizer.fit(corpus)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        # 分词
        words = [w for w in jieba.cut(text) 
                if w not in self.stopwords 
                and len(w) >= 2 
                and not re.match(r'[^\w\s]', w)]  # 过滤标点符号
        
        # 计算TF-IDF
        tfidf_matrix = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 获取每个词的TF-IDF分数
        word_scores = {}
        for word in words:
            if word in feature_names:
                idx = np.where(feature_names == word)[0][0]
                score = tfidf_matrix[0, idx]
                word_scores[word] = score
        
        # 按分数排序并返回top_k个关键词
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]

class YAKEKeywordExtractor(BaseKeywordExtractor):
    """基于YAKE的关键词提取器"""
    def __init__(self):
        super().__init__()
        
    def _calculate_case_score(self, word: str, text: str) -> float:
        """计算大小写得分"""
        if not word:
            return 0
        # 计算单词在文本中出现大写或作为首字母缩略词的次数
        upper_count = len(re.findall(r'\b' + word.upper() + r'\b', text))
        return upper_count / (text.count(word) + 1)
    
    def _calculate_position_score(self, word: str, text: str) -> float:
        """计算位置得分"""
        if not word:
            return 0
        # 计算单词在文本中的中间位置
        positions = [m.start() for m in re.finditer(word, text)]
        if not positions:
            return 0
        return 1 - (sum(positions) / len(positions)) / len(text)
    
    def _calculate_frequency_score(self, word: str, text: str) -> float:
        """计算词频得分"""
        if not word:
            return 0
        # 计算归一化词频
        freq = text.count(word)
        return freq / (len(text) + 1)
    
    def _calculate_context_score(self, word: str, text: str) -> float:
        """计算上下文相关性得分"""
        if not word:
            return 0
        # 计算与单词共同出现的不同词的数量
        window_size = 5
        words = list(jieba.cut(text))
        word_indices = [i for i, w in enumerate(words) if w == word]
        context_words = set()
        for idx in word_indices:
            start = max(0, idx - window_size)
            end = min(len(words), idx + window_size + 1)
            context_words.update(words[start:end])
        return 1 / (len(context_words) + 1)
    
    def _calculate_sentence_score(self, word: str, text: str) -> float:
        """计算句子得分"""
        if not word:
            return 0
        # 计算单词在不同句子中出现的次数
        sentences = re.split(r'[。！？]', text)
        sentence_count = sum(1 for s in sentences if word in s)
        return sentence_count / (len(sentences) + 1)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        # 分词
        words = [w for w in jieba.cut(text) if w not in self.stopwords]
        
        # 计算每个词的得分
        word_scores = {}
        for word in words:
            if len(word) < 2:  # 跳过单字词
                continue
                
            # 计算各个特征得分
            case_score = self._calculate_case_score(word, text)
            position_score = self._calculate_position_score(word, text)
            frequency_score = self._calculate_frequency_score(word, text)
            context_score = self._calculate_context_score(word, text)
            sentence_score = self._calculate_sentence_score(word, text)
            
            # 综合得分
            score = (case_score + position_score + frequency_score + 
                    context_score + sentence_score) / 5
            
            word_scores[word] = score
        
        # 按分数排序并返回top_k个关键词
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]

class TextRankKeywordExtractor(BaseKeywordExtractor):
    """基于TextRank的关键词提取器"""
    def __init__(self, window_size: int = 2, damping: float = 0.85):
        super().__init__()
        self.window_size = window_size
        self.damping = damping
        
    def _build_graph(self, words: List[str]) -> nx.Graph:
        """构建词共现图"""
        graph = nx.Graph()
        
        # 添加节点
        for word in words:
            if word not in self.stopwords and len(word) >= 2:
                graph.add_node(word)
        
        # 添加边
        for i in range(len(words)):
            if words[i] in self.stopwords or len(words[i]) < 2:
                continue
                
            for j in range(max(0, i - self.window_size), 
                          min(len(words), i + self.window_size + 1)):
                if i != j and words[j] not in self.stopwords and len(words[j]) >= 2:
                    graph.add_edge(words[i], words[j])
        
        return graph
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        # 分词
        words = list(jieba.cut(text))
        
        # 构建图
        graph = self._build_graph(words)
        
        # 计算PageRank
        scores = nx.pagerank(graph, alpha=self.damping)
        
        # 按分数排序并返回top_k个关键词
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]

class RAKEKeywordExtractor(BaseKeywordExtractor):
    """基于RAKE的关键词提取器"""
    def __init__(self):
        super().__init__()
        self.text = None
        
    def _extract_candidate_keywords(self, text: str) -> List[str]:
        """提取候选关键词"""
        # 使用停用词和标点符号分割文本
        sentences = re.split(r'[。！？]', text)
        candidates = []
        
        for sentence in sentences:
            words = list(jieba.cut(sentence))
            current_phrase = []
            
            for word in words:
                if word in self.stopwords or re.match(r'[^\w\s]', word):
                    if current_phrase:
                        candidates.append(''.join(current_phrase))
                        current_phrase = []
                else:
                    current_phrase.append(word)
            
            if current_phrase:
                candidates.append(''.join(current_phrase))
        
        return candidates
    
    def _build_cooccurrence_graph(self, candidates: List[str]) -> nx.Graph:
        """构建共现图"""
        graph = nx.Graph()
        
        # 添加节点和边
        for candidate in candidates:
            words = list(jieba.cut(candidate))
            for word in words:
                if word not in self.stopwords and len(word) >= 2:
                    graph.add_node(word)
                    for other_word in words:
                        if other_word != word and other_word not in self.stopwords and len(other_word) >= 2:
                            if graph.has_edge(word, other_word):
                                graph[word][other_word]['weight'] += 1
                            else:
                                graph.add_edge(word, other_word, weight=1)
        
        return graph
    
    def _calculate_word_scores(self, graph: nx.Graph) -> Dict[str, float]:
        """计算词分数"""
        word_scores = {}
        
        for node in graph.nodes():
            # 计算词频和词度
            freq = sum(edge['weight'] for _, _, edge in graph.edges(node, data=True))
            deg = graph.degree(node, weight='weight')
            
            # 计算分数
            if deg > 0:
                word_scores[node] = freq / deg
            else:
                word_scores[node] = 0
                
        return word_scores
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        self.text = text  # 保存文本
        
        # 提取候选关键词
        candidates = self._extract_candidate_keywords(text)
        
        # 构建共现图
        graph = self._build_cooccurrence_graph(candidates)
        
        # 计算词分数
        word_scores = self._calculate_word_scores(graph)
        
        # 计算短语分数
        phrase_scores = {}
        for candidate in candidates:
            words = list(jieba.cut(candidate))
            score = sum(word_scores.get(word, 0) for word in words 
                       if word not in self.stopwords and len(word) >= 2)
            if score > 0:
                phrase_scores[candidate] = score
        
        # 按分数排序并返回top_k个关键词
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases[:top_k]]

class EmbeddingKeywordExtractor(BaseKeywordExtractor):
    """基于嵌入的关键词提取器"""
    def __init__(self, model_name: str = "bert-base-chinese"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _get_document_embedding(self, text: str) -> torch.Tensor:
        """获取文档嵌入"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的嵌入作为文档表示
            return outputs.last_hidden_state[:, 0, :]
    
    def _get_keyword_embedding(self, keyword: str) -> torch.Tensor:
        """获取关键词嵌入"""
        inputs = self.tokenizer(keyword, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        # 获取文档嵌入
        doc_embedding = self._get_document_embedding(text)
        
        # 提取候选关键词
        words = [w for w in jieba.cut(text) if w not in self.stopwords and len(w) >= 2]
        
        # 计算每个候选关键词的相似度
        keyword_scores = {}
        for word in words:
            keyword_embedding = self._get_keyword_embedding(word)
            similarity = cosine_similarity(
                doc_embedding.cpu().numpy(),
                keyword_embedding.cpu().numpy()
            )[0][0]
            keyword_scores[word] = similarity
        
        # 按相似度排序并返回top_k个关键词
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_keywords[:top_k]] 


class MDKRankKeywordExtractor(BaseKeywordExtractor):
    """基于MDKRank的关键词提取器"""
    def __init__(self, model_name: str = "bert-base-chinese", layer_num: int = -1):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.layer_num = layer_num
        
    def _extract_candidates(self, text: str) -> List[str]:
        """提取候选关键词"""
        try:
            # 使用jieba进行分词和词性标注
            words = list(jieba.posseg.cut(text))
            candidates = []
            
            # 提取名词和形容词短语
            current_phrase = []
            for word, flag in words:
                if flag.startswith('n') or flag.startswith('a'):  # 名词或形容词
                    current_phrase.append(word)
                else:
                    if current_phrase:
                        candidates.append(''.join(current_phrase))
                        current_phrase = []
            
            # 添加最后一个短语
            if current_phrase:
                candidates.append(''.join(current_phrase))
            
            # 过滤停用词和短词
            candidates = [c for c in candidates 
                         if c not in self.stopwords 
                         and len(c) >= 2]
            
            return candidates
        except Exception as e:
            print(f"提取候选关键词时出错: {str(e)}")
            return []
    
    def _get_document_embedding(self, text: str) -> torch.Tensor:
        """获取文档嵌入"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if self.layer_num == -1:
                # 使用最后一层的[CLS]标记的嵌入
                return outputs.last_hidden_state[:, 0, :]
            else:
                # 使用指定层的[CLS]标记的嵌入
                return outputs.hidden_states[self.layer_num][:, 0, :]
    
    def _get_masked_embedding(self, text: str, candidate: str) -> torch.Tensor:
        """获取掩码后的文档嵌入"""
        # 将候选关键词替换为[MASK]
        mask = ' '.join(['[MASK]'] * len(self.tokenizer.tokenize(candidate)))
        masked_text = text.replace(candidate, mask)
        
        inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if self.layer_num == -1:
                return outputs.last_hidden_state[:, 0, :]
            else:
                return outputs.hidden_states[self.layer_num][:, 0, :]
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        # 提取候选关键词
        candidates = self._extract_candidates(text)
        if not candidates:
            return []
        
        # 获取原始文档嵌入
        doc_embedding = self._get_document_embedding(text)
        
        # 计算每个候选关键词的得分
        keyword_scores = {}
        for candidate in candidates:
            # 获取掩码后的文档嵌入
            masked_embedding = self._get_masked_embedding(text, candidate)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(doc_embedding, masked_embedding, dim=1).cpu().item()
            keyword_scores[candidate] = similarity
        
        # 按得分排序并返回top_k个关键词
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_keywords[:top_k]]
    
    def extract_keywords_with_scores(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """提取关键词及其分数"""
        # 提取候选关键词
        candidates = self._extract_candidates(text)
        if not candidates:
            return []
        
        # 获取原始文档嵌入
        doc_embedding = self._get_document_embedding(text)
        
        # 计算每个候选关键词的得分
        keyword_scores = {}
        for candidate in candidates:
            # 获取掩码后的文档嵌入
            masked_embedding = self._get_masked_embedding(text, candidate)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(doc_embedding, masked_embedding, dim=1).cpu().item()
            keyword_scores[candidate] = similarity
        
        # 按得分排序并返回top_k个关键词及其分数
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'en_model'):
            self.en_model.close()

class InputTextObj:
    """表示输入文本的对象"""
    def __init__(self, en_model, text=""):
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extract_candidates(self.tokens_tagged)

def extract_candidates(tokens_tagged, no_subset=False):
    """提取候选短语"""
    # 使用jieba词性标注
    words_pos = pseg.cut(tokens_tagged)
    candidates = []
    current_candidate = []
    
    for word, flag in words_pos:
        # 如果是名词或形容词
        if flag.startswith('n') or flag.startswith('a'):
            current_candidate.append(word)
        else:
            if current_candidate:
                candidates.append(''.join(current_candidate))
                current_candidate = []
    
    # 添加最后一个候选短语
    if current_candidate:
        candidates.append(''.join(current_candidate))
    
    return candidates

class SummaKeywordExtractor(BaseKeywordExtractor):
    """基于Summa的关键词提取器"""
    def __init__(self):
        super().__init__()
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 使用jieba分词，并过滤停用词
        words = [w for w in jieba.cut(text) if w not in self.stopwords]
        return ' '.join(words)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 使用summa提取关键词
        extracted_keywords = keywords.keywords(processed_text, words=top_k)
        
        # 将结果转换为列表
        keyword_list = extracted_keywords.split('\n') if extracted_keywords else []
        
        return keyword_list[:top_k]
    

class Word2VecKeywordExtractor(BaseKeywordExtractor):
    """基于Word2Vec词聚类的关键词提取器"""
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, n_clusters: int = 5):
        super().__init__()
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.n_clusters = n_clusters
        self.model = None
        
    def _preprocess_text(self, text: str) -> List[str]:
        """预处理文本"""
        # 分词并过滤停用词
        words = []
        for word, flag in pseg.cut(text):
            # 只保留名词、动词、形容词
            if (flag.startswith('n') or flag.startswith('v') or flag.startswith('a')) and \
               word not in self.stopwords and len(word) >= 2:
                words.append(word)
        return words
    
    def _train_word2vec(self, sentences: List[List[str]]):
        """训练Word2Vec模型"""
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
    
    def _get_word_vectors(self, words: List[str]) -> Tuple[List[str], np.ndarray]:
        """获取词向量"""
        valid_words = []
        vectors = []
        
        for word in words:
            try:
                if word in self.model.wv:
                    valid_words.append(word)
                    vectors.append(self.model.wv[word])
            except KeyError:
                continue
                
        return valid_words, np.array(vectors)
    
    def _cluster_words(self, words: List[str], vectors: np.ndarray) -> List[str]:
        """对词向量进行聚类"""
        if len(words) <= self.n_clusters:
            return words
            
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)
        
        # 获取每个簇的中心词
        cluster_centers = kmeans.cluster_centers_
        keywords = []
        
        for i in range(self.n_clusters):
            # 计算簇内所有词到簇中心的距离
            cluster_words = [words[j] for j in range(len(words)) if labels[j] == i]
            cluster_vectors = vectors[labels == i]
            
            if len(cluster_words) > 0:
                # 计算每个词到簇中心的距离
                distances = np.linalg.norm(cluster_vectors - cluster_centers[i], axis=1)
                # 选择距离最小的词作为该簇的代表词
                closest_word_idx = np.argmin(distances)
                keywords.append(cluster_words[closest_word_idx])
        
        return keywords
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        # 预处理文本
        words = self._preprocess_text(text)
        
        # 如果没有足够的词，直接返回
        if len(words) < 2:
            return words
            
        # 训练Word2Vec模型
        self._train_word2vec([words])
        
        # 获取词向量
        valid_words, vectors = self._get_word_vectors(words)
        
        # 如果没有有效的词向量，返回空列表
        if len(valid_words) == 0:
            return []
            
        # 聚类并获取关键词
        keywords = self._cluster_words(valid_words, vectors)
        
        # 如果关键词数量超过top_k，使用TF-IDF分数进行排序
        if len(keywords) > top_k:
            # 计算每个关键词的TF-IDF分数
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform([text])
            feature_names = tfidf.get_feature_names_out()
            
            keyword_scores = {}
            for keyword in keywords:
                if keyword in feature_names:
                    idx = np.where(feature_names == keyword)[0][0]
                    score = tfidf_matrix[0, idx]
                    keyword_scores[keyword] = score
            
            # 按分数排序
            sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_keywords[:top_k]]
        
        return keywords[:top_k]
    
    def extract_keywords_with_scores(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """提取关键词及其分数"""
        # 预处理文本
        words = self._preprocess_text(text)
        
        # 如果没有足够的词，直接返回
        if len(words) < 2:
            return [(word, 1.0) for word in words]
            
        # 训练Word2Vec模型
        self._train_word2vec([words])
        
        # 获取词向量
        valid_words, vectors = self._get_word_vectors(words)
        
        # 如果没有有效的词向量，返回空列表
        if len(valid_words) == 0:
            return []
            
        # 聚类并获取关键词
        keywords = self._cluster_words(valid_words, vectors)
        
        # 计算每个关键词的TF-IDF分数
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([text])
        feature_names = tfidf.get_feature_names_out()
        
        keyword_scores = {}
        for keyword in keywords:
            if keyword in feature_names:
                idx = np.where(feature_names == keyword)[0][0]
                score = tfidf_matrix[0, idx]
                keyword_scores[keyword] = score
        
        # 按分数排序
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]

class KeyBERTKeywordExtractor(BaseKeywordExtractor):
    """基于KeyBERT的关键词提取器"""
    def __init__(self, model_name: str = "bert-base-chinese", 
                 top_n: int = 20):
        super().__init__()
        # 设置离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        try:
            self.model = KeyBERT(model=SentenceTransformerBackend(model_name))
        except Exception as e:
            print(f"警告：无法加载KeyBERT模型，将使用备用模型: {str(e)}")
            # 使用备用模型
            self.model = KeyBERT(model=SentenceTransformerBackend("bert-base-chinese"))
        self.top_n = top_n
        
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 使用jieba分词，并过滤停用词
        words = [w for w in jieba.cut(text) if w not in self.stopwords]
        return ' '.join(words)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 使用KeyBERT提取关键词
        keywords = self.model.extract_keywords(
            processed_text,
            keyphrase_ngram_range=(1, 2),  # 支持1-2个词的短语
            stop_words=list(self.stopwords),
            top_n=self.top_n,
            use_mmr=True,  # 使用MMR算法增加多样性
            diversity=0.5   # 多样性参数
        )
        
        # 提取关键词文本
        keyword_list = [keyword for keyword, _ in keywords]
        
        return keyword_list[:top_k]
    
    def extract_keywords_with_scores(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """提取关键词及其分数"""
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 使用KeyBERT提取关键词及其分数
        keywords = self.model.extract_keywords(
            processed_text,
            keyphrase_ngram_range=(1, 2),  # 支持1-2个词的短语
            stop_words=list(self.stopwords),
            top_n=self.top_n,
            use_mmr=True,  # 使用MMR算法增加多样性
            diversity=0.5   # 多样性参数
        )
        
        return keywords[:top_k]

    