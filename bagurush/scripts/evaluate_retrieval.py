"""
evaluate_retrieval.py — 构建检索测试集 + 计算 Recall@K / Top-3 命中率

用法:
    cd bagurush/
    /mnt/d/ForWSL/env/bagurush/bin/python scripts/evaluate_retrieval.py

流程:
    1. 定义 50 条检索 query（覆盖所有 topic）+ 每条对应的 ground truth 关键词
    2. 用 FAISS 和 BM25 分别检索
    3. 计算指标：Recall@5, Recall@10, Top-3 命中率, MRR
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FAISS_INDEX_DIR = PROJECT_DIR / "faiss_index" / "tech_knowledge"
BM25_INDEX_DIR = PROJECT_DIR / "bm25_index"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ------------------------------------------------------------------ #
#  检索测试集：50 条 query + ground truth（通过关键词匹配判定）
# ------------------------------------------------------------------ #
# 每条包含：query, expected_topic, ground_truth_keywords（只需其中任一关键词出现在召回文本中即认为命中）
TEST_SET = [
    # cs_408 — 数据结构/OS/网络/数据库 (13 条)
    {"query": "Python GIL 全局解释器锁是什么", "topic": "cs_408", "keywords": ["GIL", "全局解释器锁", "Global Interpreter Lock"]},
    {"query": "TCP 三次握手过程", "topic": "cs_408", "keywords": ["三次握手", "SYN", "ACK"]},
    {"query": "TCP 和 UDP 的区别", "topic": "cs_408", "keywords": ["TCP", "UDP", "面向连接", "无连接"]},
    {"query": "操作系统进程和线程的区别", "topic": "cs_408", "keywords": ["进程", "线程", "process", "thread"]},
    {"query": "死锁的四个必要条件", "topic": "cs_408", "keywords": ["死锁", "互斥", "deadlock", "必要条件"]},
    {"query": "虚拟内存的概念和作用", "topic": "cs_408", "keywords": ["虚拟内存", "virtual memory", "页表"]},
    {"query": "HTTP 和 HTTPS 的区别", "topic": "cs_408", "keywords": ["HTTPS", "SSL", "TLS", "加密"]},
    {"query": "二叉搜索树的特性", "topic": "cs_408", "keywords": ["二叉搜索树", "BST", "左子树", "右子树"]},
    {"query": "快速排序的时间复杂度", "topic": "cs_408", "keywords": ["快速排序", "quicksort", "O(nlogn)", "分治"]},
    {"query": "数据库事务的 ACID 特性", "topic": "cs_408", "keywords": ["ACID", "原子性", "一致性", "隔离性", "持久性"]},
    {"query": "B+树和B树的区别", "topic": "cs_408", "keywords": ["B+树", "B树", "叶子节点", "索引"]},
    {"query": "DNS 解析过程", "topic": "cs_408", "keywords": ["DNS", "域名解析", "递归", "迭代"]},
    {"query": "页面置换算法 LRU", "topic": "cs_408", "keywords": ["LRU", "页面置换", "最近最少使用"]},

    # java_backend — Java/Spring/中间件 (12 条)
    {"query": "HashMap 底层实现原理", "topic": "java_backend", "keywords": ["HashMap", "数组", "链表", "红黑树", "hash"]},
    {"query": "Spring Boot 自动配置原理", "topic": "java_backend", "keywords": ["自动配置", "AutoConfiguration", "SpringBoot", "spring.factories"]},
    {"query": "JVM 垃圾回收机制", "topic": "java_backend", "keywords": ["垃圾回收", "GC", "标记清除", "新生代", "老年代"]},
    {"query": "Redis 持久化 RDB 和 AOF", "topic": "java_backend", "keywords": ["RDB", "AOF", "持久化", "快照"]},
    {"query": "MySQL 索引优化", "topic": "java_backend", "keywords": ["索引", "MySQL", "B+树", "explain"]},
    {"query": "Kafka 消息队列的基本架构", "topic": "java_backend", "keywords": ["Kafka", "broker", "partition", "消息队列", "topic"]},
    {"query": "Spring IOC 容器", "topic": "java_backend", "keywords": ["IOC", "控制反转", "依赖注入", "Spring"]},
    {"query": "Java 多线程 synchronized 和 Lock", "topic": "java_backend", "keywords": ["synchronized", "Lock", "ReentrantLock", "锁"]},
    {"query": "Netty 的线程模型", "topic": "java_backend", "keywords": ["Netty", "EventLoop", "NIO", "线程模型"]},
    {"query": "分布式系统 CAP 理论", "topic": "java_backend", "keywords": ["CAP", "一致性", "可用性", "分区容错"]},
    {"query": "SpringCloud 微服务组件", "topic": "java_backend", "keywords": ["SpringCloud", "微服务", "Eureka", "Feign", "网关"]},
    {"query": "Redis 数据结构类型", "topic": "java_backend", "keywords": ["Redis", "String", "Hash", "List", "Set", "ZSet"]},

    # ml_dl — 机器学习/深度学习 (13 条)
    {"query": "什么是 Transformer 的自注意力机制", "topic": "ml_dl", "keywords": ["自注意力", "self-attention", "Transformer", "Q K V"]},
    {"query": "梯度消失和梯度爆炸", "topic": "ml_dl", "keywords": ["梯度消失", "梯度爆炸", "vanishing gradient"]},
    {"query": "Batch Normalization 的原理", "topic": "ml_dl", "keywords": ["Batch Normalization", "BN", "归一化"]},
    {"query": "过拟合的解决方法", "topic": "ml_dl", "keywords": ["过拟合", "overfitting", "正则化", "dropout"]},
    {"query": "卷积神经网络 CNN 的基本结构", "topic": "ml_dl", "keywords": ["CNN", "卷积", "池化", "全连接"]},
    {"query": "LSTM 和 GRU 的区别", "topic": "ml_dl", "keywords": ["LSTM", "GRU", "门控", "遗忘门"]},
    {"query": "逻辑回归的损失函数", "topic": "ml_dl", "keywords": ["逻辑回归", "交叉熵", "sigmoid", "损失函数"]},
    {"query": "SVM 支持向量机", "topic": "ml_dl", "keywords": ["SVM", "支持向量", "核函数", "超平面"]},
    {"query": "随机森林和 XGBoost 区别", "topic": "ml_dl", "keywords": ["随机森林", "XGBoost", "集成学习", "bagging", "boosting"]},
    {"query": "Adam 优化器", "topic": "ml_dl", "keywords": ["Adam", "优化器", "动量", "自适应"]},
    {"query": "注意力机制 Attention 原理", "topic": "ml_dl", "keywords": ["注意力", "Attention", "Query", "Key", "Value"]},
    {"query": "目标检测 YOLO 算法", "topic": "ml_dl", "keywords": ["YOLO", "目标检测", "anchor", "bounding box"]},
    {"query": "残差网络 ResNet 的核心思想", "topic": "ml_dl", "keywords": ["ResNet", "残差", "skip connection", "shortcut"]},

    # llm_rag_agent — LLM/RAG/Agent (12 条)
    {"query": "RAG 检索增强生成", "topic": "llm_rag_agent", "keywords": ["RAG", "检索增强", "Retrieval Augmented"]},
    {"query": "LangChain Agent 工作流程", "topic": "llm_rag_agent", "keywords": ["Agent", "LangChain", "工具调用", "tool"]},
    {"query": "Prompt Engineering 提示工程", "topic": "llm_rag_agent", "keywords": ["Prompt", "提示工程", "few-shot", "chain of thought"]},
    {"query": "大模型微调方法 LoRA", "topic": "llm_rag_agent", "keywords": ["LoRA", "微调", "fine-tune", "低秩"]},
    {"query": "向量检索和 FAISS", "topic": "llm_rag_agent", "keywords": ["FAISS", "向量检索", "向量数据库", "embedding"]},
    {"query": "Embedding 模型的作用", "topic": "llm_rag_agent", "keywords": ["Embedding", "嵌入", "向量化", "语义"]},
    {"query": "LLM 幻觉问题及解决方案", "topic": "llm_rag_agent", "keywords": ["幻觉", "hallucination", "事实性", "RAG"]},
    {"query": "大模型的 RLHF 训练", "topic": "llm_rag_agent", "keywords": ["RLHF", "人类反馈", "强化学习", "对齐"]},
    {"query": "LangGraph 状态图", "topic": "llm_rag_agent", "keywords": ["LangGraph", "状态图", "StateGraph", "节点"]},
    {"query": "向量数据库 Milvus 或 Chroma", "topic": "llm_rag_agent", "keywords": ["Milvus", "Chroma", "向量数据库", "向量存储"]},
    {"query": "大模型推理优化 KV Cache", "topic": "llm_rag_agent", "keywords": ["KV Cache", "推理优化", "inference", "加速"]},
    {"query": "多模态大模型架构", "topic": "llm_rag_agent", "keywords": ["多模态", "multimodal", "视觉语言", "图文"]},
]


def check_hit(text: str, keywords: List[str]) -> bool:
    """检查文本中是否包含任一关键词（不区分大小写）"""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate_faiss(test_set: List[Dict]) -> Dict:
    """评估 FAISS 检索"""
    from langchain_community.vectorstores import FAISS
    from rag.embeddings import get_embeddings

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )

    results = {"recall_5": 0, "recall_10": 0, "top3_hit": 0, "mrr": 0.0, "topic_hit_5": 0}
    total = len(test_set)

    for tq in test_set:
        docs = vectorstore.similarity_search(tq["query"], k=10)
        texts = [d.page_content for d in docs]
        topics = [d.metadata.get("topic", "") for d in docs]

        # Recall@5: 前 5 中是否有命中
        hit_5 = any(check_hit(t, tq["keywords"]) for t in texts[:5])
        # Recall@10: 前 10 中是否有命中
        hit_10 = any(check_hit(t, tq["keywords"]) for t in texts[:10])
        # Top-3 命中率
        hit_3 = any(check_hit(t, tq["keywords"]) for t in texts[:3])
        # Topic 命中 @5
        topic_hit = tq["topic"] in topics[:5]

        # MRR: 第一个命中的位置
        mrr = 0.0
        for rank, t in enumerate(texts[:10], 1):
            if check_hit(t, tq["keywords"]):
                mrr = 1.0 / rank
                break

        results["recall_5"] += int(hit_5)
        results["recall_10"] += int(hit_10)
        results["top3_hit"] += int(hit_3)
        results["mrr"] += mrr
        results["topic_hit_5"] += int(topic_hit)

    # 转为比率
    for key in results:
        results[key] = results[key] / total

    return results


def evaluate_bm25(test_set: List[Dict]) -> Dict:
    """评估 BM25 检索"""
    import jieba
    from rank_bm25 import BM25Okapi

    with open(BM25_INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_INDEX_DIR / "chunks_meta.pkl", "rb") as f:
        chunks_meta = pickle.load(f)

    results = {"recall_5": 0, "recall_10": 0, "top3_hit": 0, "mrr": 0.0, "topic_hit_5": 0}
    total = len(test_set)

    for tq in test_set:
        query_tokens = list(jieba.cut(tq["query"]))
        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:10]
        top_results = [chunks_meta[j] for j in top_indices]

        texts = [r["text"] for r in top_results]
        topics = [r["topic"] for r in top_results]

        hit_5 = any(check_hit(t, tq["keywords"]) for t in texts[:5])
        hit_10 = any(check_hit(t, tq["keywords"]) for t in texts[:10])
        hit_3 = any(check_hit(t, tq["keywords"]) for t in texts[:3])
        topic_hit = tq["topic"] in topics[:5]

        mrr = 0.0
        for rank, t in enumerate(texts[:10], 1):
            if check_hit(t, tq["keywords"]):
                mrr = 1.0 / rank
                break

        results["recall_5"] += int(hit_5)
        results["recall_10"] += int(hit_10)
        results["top3_hit"] += int(hit_3)
        results["mrr"] += mrr
        results["topic_hit_5"] += int(topic_hit)

    for key in results:
        results[key] = results[key] / total

    return results


def evaluate_hybrid(test_set: List[Dict]) -> Dict:
    """评估混合检索（FAISS + BM25 + RRF 融合）"""
    from langchain_community.vectorstores import FAISS
    from rag.embeddings import get_embeddings
    import jieba

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )

    with open(BM25_INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_INDEX_DIR / "chunks_meta.pkl", "rb") as f:
        chunks_meta = pickle.load(f)

    results = {"recall_5": 0, "recall_10": 0, "top3_hit": 0, "mrr": 0.0, "topic_hit_5": 0}
    total = len(test_set)

    for tq in test_set:
        # FAISS 召回 top 20
        faiss_docs = vectorstore.similarity_search(tq["query"], k=20)

        # BM25 召回 top 20
        query_tokens = list(jieba.cut(tq["query"]))
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda j: bm25_scores[j], reverse=True)[:20]

        # RRF 融合
        rrf_scores = {}  # text_hash -> (score, text, topic)
        RRF_K = 60

        for rank, doc in enumerate(faiss_docs):
            key = hash(doc.page_content[:200])
            old_score = rrf_scores.get(key, (0, "", ""))[0]
            rrf_scores[key] = (old_score + 1.0 / (RRF_K + rank + 1), doc.page_content, doc.metadata.get("topic", ""))

        for rank, idx in enumerate(bm25_top):
            meta = chunks_meta[idx]
            key = hash(meta["text"][:200])
            old_score = rrf_scores.get(key, (0, "", ""))[0]
            rrf_scores[key] = (old_score + 1.0 / (RRF_K + rank + 1), meta["text"], meta["topic"])

        # 按 RRF 分数排序
        ranked = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)[:10]
        texts = [r[1] for r in ranked]
        topics = [r[2] for r in ranked]

        hit_5 = any(check_hit(t, tq["keywords"]) for t in texts[:5])
        hit_10 = any(check_hit(t, tq["keywords"]) for t in texts[:10])
        hit_3 = any(check_hit(t, tq["keywords"]) for t in texts[:3])
        topic_hit = tq["topic"] in topics[:5]

        mrr = 0.0
        for rank, t in enumerate(texts[:10], 1):
            if check_hit(t, tq["keywords"]):
                mrr = 1.0 / rank
                break

        results["recall_5"] += int(hit_5)
        results["recall_10"] += int(hit_10)
        results["top3_hit"] += int(hit_3)
        results["mrr"] += mrr
        results["topic_hit_5"] += int(topic_hit)

    for key in results:
        results[key] = results[key] / total

    return results


def print_results(name: str, results: Dict):
    """打印评估结果"""
    print(f"\n  [{name}]")
    print(f"    Recall@5:       {results['recall_5']:.2%}")
    print(f"    Recall@10:      {results['recall_10']:.2%}")
    print(f"    Top-3 命中率:   {results['top3_hit']:.2%}")
    print(f"    MRR@10:         {results['mrr']:.4f}")
    print(f"    Topic 命中@5:   {results['topic_hit_5']:.2%}")


def main():
    print("=" * 60)
    print(f"[Evaluate] 检索质量评估（{len(TEST_SET)} 条测试 query）")
    print("=" * 60)

    print("\n>>> 评估 FAISS 语义检索...")
    faiss_results = evaluate_faiss(TEST_SET)
    print_results("FAISS", faiss_results)

    print("\n>>> 评估 BM25 关键词检索...")
    bm25_results = evaluate_bm25(TEST_SET)
    print_results("BM25", bm25_results)

    print("\n>>> 评估 Hybrid（FAISS + BM25 + RRF）...")
    hybrid_results = evaluate_hybrid(TEST_SET)
    print_results("Hybrid (RRF)", hybrid_results)

    # 汇总表格
    print(f"\n{'='*60}")
    print(f"{'指标':<20} {'FAISS':>8} {'BM25':>8} {'Hybrid':>8} {'目标':>8}")
    print(f"{'-'*60}")
    for metric, target in [("recall_5", "≥85%"), ("recall_10", "≥85%"), ("top3_hit", "≥80%"), ("mrr", "—"), ("topic_hit_5", "—")]:
        label = {"recall_5": "Recall@5", "recall_10": "Recall@10", "top3_hit": "Top-3命中率", "mrr": "MRR@10", "topic_hit_5": "Topic命中@5"}[metric]
        print(f"{label:<20} {faiss_results[metric]:>7.2%} {bm25_results[metric]:>7.2%} {hybrid_results[metric]:>7.2%} {target:>8}")
    print(f"{'='*60}")

    # 保存结果
    report = {
        "test_set_size": len(TEST_SET),
        "faiss": faiss_results,
        "bm25": bm25_results,
        "hybrid_rrf": hybrid_results,
    }
    report_path = PROJECT_DIR / "knowledge_base" / "chunks" / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n评估报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
