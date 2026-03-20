# RAG 检索架构说明

## 概述

BaguRush 的 RAG 系统采用**四级混合检索 Pipeline**，从技术知识库中检索相关内容，供 Interviewer 出题和 Evaluator 评分使用。

## 知识库构建

### 数据来源
- 11 个 GitHub 仓库（`data_sources/` 目录，517MB 原始数据）
- 经 `scripts/clean_and_chunk.py` 清洗 → 切分为 chunk → 自动推断 metadata

### 数据清洗规则

`clean_and_chunk.py` 的 `clean_text()` 函数执行以下清洗：

| 清洗规则 | 说明 |
|---------|------|
| 去除 HTML 标签 | `<h2>`、`<div>`、`<img>`、`<br>` 等 |
| 去除 HTML 注释 | `<!-- ... -->` |
| 去除目录锚点行 | `- [问题](#user-content-xxx)` 格式的目录行 |
| 去除纯 URL 行 | 不在 markdown 链接中的裸 URL |
| 去除图片引用 | `![alt](url)` → 只保留 `[alt]` |
| 去除 HTML 实体 | `&amp;`、`&nbsp;` 等 |
| 合并连续空行 | 最多保留 2 个换行 |

### 切分策略：Structure-Aware Chunking

采用**按标题边界切分**策略（`chunk_by_sections()`），区分 Markdown 和非 Markdown 文件：

#### Markdown 文件（`.md`）— `chunk_by_sections()`

```
原始文档
    │
    ▼
第1步：按 # / ## / ### / #### 标题拆分为独立 section
    │
    ├── 短 section（≤1500字）→ 整个作为 1 个 chunk，不与相邻 section 合并
    │
    └── 长 section（>1500字）→ 按段落二次切分为 ~1000 字子块
                                每个子块保留 section 标题前缀
                                相邻子块间 200 字 overlap
    │
    ▼
第2步：过滤纯链接/目录 section 和 <100 字碎片
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_section_size` | 1500 字符 | 超过此长度的 section 才二次切分 |
| `sub_chunk_size` | 1000 字符 | 二次切分目标块大小，对齐 BGE 模型 512 token 窗口 |
| `overlap` | 200 字符 | 二次切分时的重叠量 |
| `min_chunk_size` | 100 字符 | 过滤太短的碎片 |

#### 非 Markdown 文件（`.txt`、`.py`、`.java` 等）— `chunk_text()`

| 参数 | 值 |
|------|-----|
| `chunk_size` | 1000 字符 |
| `chunk_overlap` | 200 字符 |

#### 切分效果

| 指标 | 值 |
|------|-----|
| 总 chunk 数 | 11145 |
| 平均长度 | 511 字符 |
| 最长 chunk | 1498 字符 |
| >1500 字 chunk | 0（全部在 embedding 模型窗口内） |

### 索引文件

| 索引 | 文件位置 | 说明 |
|------|---------|------|
| FAISS 向量索引 | `faiss_index/tech_knowledge/index.faiss` + `index.pkl` | BGE-small-zh-v1.5 编码的 512 维向量 |
| BM25 关键词索引 | `bm25_index/bm25.pkl` + `chunks_meta.pkl` | jieba 分词后的倒排索引（pickle 序列化） |

### 构建命令

```bash
# 步骤 1：清洗 + 切分（生成 all_chunks.jsonl）
python scripts/clean_and_chunk.py

# 步骤 2：构建 FAISS + BM25 索引（推荐用 CUDA 加速）
EMBEDDING_DEVICE=cuda python scripts/build_index.py

# 快捷方式：用预置知识库直接建索引（跳过清洗）
python -m rag.vector_store --init
```

## 四级检索 Pipeline

```
用户 query（如 "什么是死锁"）
    │
    ├── 第1路：FAISS 语义检索 → Top-20
    │     代码：hybrid_retriever.py L219-225
    │     模型：BAAI/bge-small-zh-v1.5（512 维，支持 CUDA）
    │     原理：query 和 chunk 都变成向量，计算余弦相似度
    │     擅长：理解语义（"死锁" ≈ "两个进程互相等待"）
    │
    ├── 第2路：BM25 关键词检索 → Top-20
    │     代码：hybrid_retriever.py L228-230, BM25Searcher.search()
    │     分词：jieba 中文分词
    │     原理：按词频+逆文档频率打分（TF-IDF 变体）
    │     擅长：精确匹配术语（"GIL"、"-Xmx"、"B+树"）
    │
    ▼
第3步：RRF 融合
    代码：hybrid_retriever.py L236, reciprocal_rank_fusion()
    公式：score(doc) = Σ 1/(k + rank_i), k=60
    效果：两路都排名靠前的 chunk 得分最高
    │
    ▼
第3.5步：同源去重
    代码：hybrid_retriever.py L239, _deduplicate_by_source()
    规则：同一 source_file 最多保留 2 条
    效果：避免结果被单个文件霸占
    │
    ▼
第4步：BGE Reranker 精排 → Top-K
    代码：hybrid_retriever.py L242-243
    模型：BAAI/bge-reranker-base（Cross-Encoder，sentence-transformers）
    原理：(query, chunk) 拼在一起过 Transformer，输出精确相关性分数
    速度：比 embedding 慢（逐对计算），但精度显著更高
    │
    ▼
返回最终 Top-K Document
```

## 统一的检索参数

| 调用者 | 代码位置 | final_k | 截断 | 说明 |
|--------|---------|---------|------|------|
| **Interviewer** | `knowledge_rag.py` L63 | **k=3** | 由 LLM 看全文 | 出题时检索知识库 |
| **Evaluator 题目检索** | `evaluator.py` L101 | **k=3** | 2500 字 | 用面试题搜标准知识点 |
| **Evaluator 回答检索** | `evaluator.py` L106 | **k=2** | 1500 字 | 用候选人回答验证准确性 |

## 降级兜底策略

| 故障场景 | 降级行为 | 代码位置 |
|---------|---------|---------|
| BM25 索引加载失败 | 跳过 BM25，只用 FAISS 单路 | `hybrid_retriever.py` L179-182 |
| Reranker 模型加载失败 | 跳过精排，返回 RRF 融合结果 | `hybrid_retriever.py` L185-189 |
| 整个混合检索异常 | 回退到纯 FAISS 语义检索 | `hybrid_retriever.py` L207-210 |
| 连 FAISS 索引都不存在 | 返回错误提示字符串 | `knowledge_rag.py` L102-106 |

## 智能离线模式

`rag/embeddings.py` 在 import 时自动检测本地缓存：
- 如果 `bge-small-zh-v1.5` 和 `bge-reranker-base` 都已缓存 → 自动启用离线模式
- 如果缓存不全 → 允许联网下载（首次运行时）
- 用户手动设置 `HF_HUB_OFFLINE` 环境变量 → 尊重用户设置

## 调用入口

所有 RAG 调用统一走 `tools/knowledge_rag.py` 的 `search_tech_knowledge` 工具：

```python
# knowledge_rag.py 核心逻辑
hybrid = _get_hybrid_retriever()          # 拿 HybridRetriever 单例
if hybrid is not None:
    results = hybrid.retrieve(query, final_k=k)  # 四级 Pipeline
else:
    results = store.search(query, k=k)           # 降级到纯 FAISS
```

## RAG 在面试中的两个作用层次

| 层次 | 使用者 | 调用方式 | 代码位置 |
|------|--------|---------|---------|
| **出题引导** | Interviewer | Tool Calling（LLM 自主决定 query） | `agents/interviewer.py` L108-125 |
| **评分参考** | Evaluator | 直接调用，双重检索（题目 k=3 + 回答 k=2） | `agents/evaluator.py` L97-118 |

## 交互式测试

```bash
cd bagurush
python rag/demo_rag.py
```

输入 query 可查看四级 Pipeline 每一步的检索结果和耗时。

## 相关文件索引

| 文件 | 职责 |
|------|------|
| `rag/hybrid_retriever.py` | 四级混合检索器 + 同源去重 + Reranker |
| `rag/vector_store.py` | FAISS 向量存储管理（构建/加载/检索/持久化） |
| `rag/embeddings.py` | BGE 嵌入模型加载 + 智能离线检测 |
| `rag/document_loader.py` | 文档加载 + 切分（运行时文档用） |
| `rag/demo_rag.py` | 四级 Pipeline 交互式测试 Demo |
| `tools/knowledge_rag.py` | RAG 工具入口（@tool 定义，默认 k=3） |
| `scripts/build_index.py` | 从 data_sources 构建 FAISS + BM25 索引 |
| `scripts/clean_and_chunk.py` | 数据清洗 + Structure-Aware Chunking |
