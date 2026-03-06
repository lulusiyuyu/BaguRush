# BaguRush — AI 模拟面试多 Agent 系统 · 项目上下文文档

> **文档版本**：v1.0  
> **最后更新**：2026-03-05  
> **用途**：任何无上下文的开发 Agent 阅读本文档后即可独立实现整个项目。

---

## 一、项目概述

### 1.1 项目名称与定位

**BaguRush**（锻造面试能力）是一个基于 **LangGraph** 的多 Agent 协作面试模拟平台。用户上传简历，选择目标岗位，系统自动进行一场完整的模拟面试，包括：简历分析 → 智能出题 → 多轮追问 → 实时评估 → 生成报告。

### 1.2 核心价值

| 维度 | 描述 |
|---|---|
| **简历驱动** | 解析用户简历，针对其背景弱点和强项进行差异化出题 |
| **多 Agent 编排** | 4 个专职 Agent（Planner / Interviewer / Evaluator / Reporter）通过 LangGraph 状态图协作 |
| **RAG 知识检索** | FAISS 向量数据库 + LangChain Document Loaders，检索简历和技术知识库 |
| **Tool Calling** | 5 个工具函数，Agent 通过 Function Calling 按需调用 |
| **动态追问** | Router 节点根据评分决定追问/换题/结束，模拟真实面试递进式追问 |
| **智能评分报告** | 分维度评分 + 强弱项分析 + 针对性改进建议 |

### 1.3 与竞品的差异化

本项目同时超越了两个开源项目的不足：

| 维度 | rakia/ai-interviewer（教学 Demo） | TechPrep AI（全栈产品） | **BaguRush（本项目）** |
|---|---|---|---|
| Agent 框架 | ✅ LangGraph（单节点） | ❌ 只用 LangChain | ✅ LangGraph 多节点状态图 |
| Tool Calling | ✅ 有（2 个简单工具） | ❌ 无 | ✅ 有（5 个工具函数） |
| RAG | ❌ 无（硬编码 dict） | ✅ 有（向量嵌入） | ✅ 有（简历 + 技术知识库） |
| 多 Agent | ❌ 单节点 | ❌ 单节点 | ✅ 4 Agent 协作 |
| 简历分析 | ❌ 无 | ❌ 无 | ✅ 上传简历 → 向量化 → 针对弱点提问 |
| Memory | ✅ MemorySaver | 不明确 | ✅ 对话记忆 + 跨轮次追问 |
| 前端 | ❌ 命令行 | ✅ Next.js | ✅ FastAPI + Web UI |
| 评分报告 | ❌ 简单关键词 | ✅ 多维度 | ✅ LLM 多维度评分 + 报告 |

---

## 二、技术栈

### 2.1 全技术栈清单

| 层 | 技术 | 版本要求 | 用途 |
|---|---|---|---|
| **Agent 编排** | LangGraph | `>=0.2` | 多 Agent 状态图、条件路由、Memory |
| **LLM 框架** | LangChain | `>=0.3` | Document Loaders、Prompt Templates、Tool 定义 |
| **LLM 提供者** | DeepSeek API | — | 主力 LLM（OpenAI-compatible 接口） |
| **向量数据库** | FAISS（faiss-cpu） | — | 本地向量索引，RAG 检索 |
| **嵌入模型** | BGE系列 (BAAI/bge-small-zh-v1.5) | — | **国内最强中文嵌入模型**（首选），本地运行方案 |
| **嵌入模型 (API)** | 智谱AI / 阿里通义 / 百度 | — | 国内 API 平替方案（可选） |
| **后端框架** | FastAPI | `>=0.100` | RESTful API、WebSocket 支持 |
| **Python** | Python | `>=3.10` | 核心运行时 |
| **前端** | 纯 HTML + CSS + JavaScript | — | LeetCode 风格 Web 交互界面（白灰配色，质感设计） |
| **文档解析** | LangChain Document Loaders | — | 解析 PDF / Markdown 简历 |
| **依赖管理** | pip + requirements.txt | — | 依赖管理 |

### 2.2 环境变量

项目需要以下环境变量（通过 `.env` 文件管理）：

```bash
# LLM 配置
DEEPSEEK_API_KEY=sk-xxx         # DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com  # DeepSeek API Base URL
DEEPSEEK_MODEL=deepseek-chat     # 模型名称

# 也可以用 OpenAI 兼容接口的其他供应商
# OPENAI_API_KEY=sk-xxx
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini

# 嵌入模型配置
# 方案1: 使用本地 BGE 模型 (完全免费，最适合中文)
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu

# 方案2: 使用国内 API (如智谱 AI)
# ZHIPUAI_API_KEY=your_key
# ZHIPUAI_MODEL=embedding-3

# 服务配置
HOST=0.0.0.0
PORT=8000
```

### 2.3 核心依赖（requirements.txt）

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-core>=0.3.0
faiss-cpu>=1.7.0
fastapi>=0.100.0
uvicorn>=0.20.0
python-dotenv>=1.0.0
python-multipart>=0.0.6
pydantic>=2.0.0
pypdf>=3.0.0
unstructured>=0.10.0
sentence-transformers>=2.0.0
```

---

## 三、项目目录结构

```
bagurush/
├── .env                          # 环境变量（不上传 git）
├── .env.example                  # 环境变量模板
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                       # FastAPI 应用入口
│
├── agents/                       # Agent 定义目录
│   ├── __init__.py
│   ├── state.py                  # InterviewState 状态定义
│   ├── graph.py                  # LangGraph 核心状态图构建
│   ├── planner.py                # Planner Agent — 面试规划
│   ├── interviewer.py            # Interviewer Agent — 提问与追问
│   ├── evaluator.py              # Evaluator Agent — 回答评估
│   ├── reporter.py               # Reporter Agent — 报告生成
│   └── router.py                 # Router 节点 — 条件路由逻辑
│
├── tools/                        # Tool 定义目录
│   ├── __init__.py
│   ├── resume_parser.py          # 工具 1：简历解析
│   ├── job_search.py             # 工具 2：岗位要求检索
│   ├── knowledge_rag.py          # 工具 3：技术知识 RAG 检索
│   ├── answer_evaluator.py       # 工具 4：回答多维评估
│   └── code_analyzer.py          # 工具 5：代码质量分析
│
├── rag/                          # RAG 知识库管理目录
│   ├── __init__.py
│   ├── vector_store.py           # FAISS 向量存储管理
│   ├── document_loader.py        # 文档加载与切分
│   └── embeddings.py             # 嵌入模型封装
│
├── prompts/                      # Prompt 模板目录
│   ├── planner_prompt.py         # Planner Agent 的 System Prompt
│   ├── interviewer_prompt.py     # Interviewer Agent 的 System Prompt
│   ├── evaluator_prompt.py       # Evaluator Agent 的 System Prompt
│   └── reporter_prompt.py        # Reporter Agent 的 System Prompt
│
├── api/                          # API 路由目录
│   ├── __init__.py
│   ├── routes.py                 # FastAPI 路由定义
│   └── schemas.py                # Pydantic 请求/响应模型
│
├── knowledge_base/               # 预置知识库内容
│   ├── tech/                     # 技术知识（Markdown 文件）
│   │   ├── python_basics.md
│   │   ├── data_structures.md
│   │   ├── system_design.md
│   │   ├── machine_learning.md
│   │   └── recommender_systems.md
│   └── jobs/                     # 岗位要求模板（JSON 文件）
│       ├── backend_engineer.json
│       ├── ml_engineer.json
│       ├── ai_agent_developer.json
│       └── recsys_engineer.json
│
├── uploads/                      # 用户上传文件（简历等）
│   └── .gitkeep
│
├── frontend/                     # 前端文件（LeetCode 风格 HTML+JS+CSS）
│   ├── index.html                # 主页面结构
│   ├── style.css                 # 全局样式（设计系统 + 组件样式）
│   ├── app.js                    # 核心交互逻辑 + API 对接
│   └── markdown-it.min.js        # Markdown 渲染库（CDN 备用本地拷贝）
│
└── tests/                        # 测试目录
    ├── __init__.py
    ├── test_tools.py
    ├── test_agents.py
    └── test_graph.py
```

---

## 四、核心架构设计

### 4.1 LangGraph 多 Agent 状态图

这是整个项目的核心架构。所有 Agent 通过 LangGraph 的 `StateGraph` 编排，共享一个全局状态 `InterviewState`。

```
                    ┌─────────────┐
                    │    START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Planner   │ ← 读取简历 + JD，规划面试大纲
                    │    Agent    │   Tools: parse_resume, search_job_requirements
                    └──────┬──────┘
                           │ 输出：interview_plan (面试大纲)
                    ┌──────▼──────┐
              ┌────►│ Interviewer │ ← 根据大纲出题，等待用户回答
              │     │    Agent    │   Tools: search_tech_knowledge
              │     └──────┬──────┘
              │            │ 等待用户输入回答 (human-in-the-loop)
              │     ┌──────▼──────┐
              │     │  Evaluator  │ ← 评估当前回答质量
              │     │    Agent    │   Tools: evaluate_answer, evaluate_code
              │     └──────┬──────┘
              │            │ 输出：evaluation_result
              │     ┌──────▼──────┐
              └─────┤   Router    │ ← 条件路由：追问 → Interviewer
                    │   (节点)    │               下一题 → Interviewer
                    └──────┬──────┘               结束 → Reporter
                           │ 面试结束
                    ┌──────▼──────┐
                    │  Reporter   │ ← 生成最终面试报告
                    │    Agent    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │     END     │
                    └─────────────┘
```

### 4.2 InterviewState 状态定义

`InterviewState` 是 LangGraph 状态图的全局共享状态，定义了所有 Agent 需要读写的数据：

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

class InterviewState(TypedDict):
    # --- 输入信息 ---
    messages: Annotated[list, add_messages]  # 对话历史（LangGraph 消息列表）
    resume_text: str                          # 原始简历文本
    job_role: str                             # 目标岗位（如 "后端开发"）
    candidate_name: str                       # 候选人姓名
    
    # --- Planner 输出 ---
    resume_analysis: str                      # 简历分析结果（强项、弱点、关键技能）
    interview_plan: list[dict]                # 面试大纲：[{topic, difficulty, questions_count, focus_areas}]
    
    # --- Interviewer 状态 ---
    current_topic_index: int                  # 当前题目大纲索引
    current_question: str                     # 当前提问内容
    follow_up_count: int                      # 当前题目已追问次数
    max_follow_ups: int                       # 每题最大追问次数（默认 2）
    
    # --- Evaluator 输出 ---
    current_evaluation: dict                  # 当前回答评估：{completeness, accuracy, depth, expression, score, feedback}
    all_evaluations: list[dict]               # 所有评估记录
    
    # --- Router 状态 ---
    next_action: Literal["follow_up", "next_question", "end_interview"]  # 下一步动作
    
    # --- Reporter 输出 ---
    final_report: str                         # 最终面试报告（Markdown 格式）
    
    # --- 全局控制 ---
    total_questions_asked: int                # 已提问总数
    max_questions: int                        # 最大提问数（默认 8）
    interview_status: Literal["planning", "interviewing", "evaluating", "reporting", "completed"]
```

### 4.3 各 Agent 职责详解

#### 4.3.1 Planner Agent

**输入**：`resume_text`, `job_role`  
**输出**：`resume_analysis`, `interview_plan`  
**使用工具**：`parse_resume`, `search_job_requirements`

**职责**：
1. 调用 `parse_resume` 解析简历，提取技能树、项目经历、教育背景
2. 调用 `search_job_requirements` 获取目标岗位的技术要求
3. 对比简历和岗位要求，找出匹配项和差距
4. 生成面试大纲 `interview_plan`，规划 5~8 个面试话题，每个话题包含：
   - `topic`：话题名称（如 "Python 基础"、"系统设计"）
   - `difficulty`：难度级别（easy / medium / hard）
   - `questions_count`：该话题计划提问数
   - `focus_areas`：重点考察方向
   - `from_resume_weakness`：是否来自简历薄弱环节（用于针对性出题）

**System Prompt 核心要求**：
```
你是一位资深的面试规划师。你的任务是：
1. 分析候选人的简历，提取关键技能、项目经历和教育背景
2. 将简历信息与目标岗位要求进行对比
3. 制定一份面试大纲，包含 5~8 个面试话题
4. 对简历中的薄弱环节加大提问权重
5. 输出结构化的面试计划 JSON
```

#### 4.3.2 Interviewer Agent

**输入**：`interview_plan`, `current_topic_index`, `messages`  
**输出**：`current_question`, `messages`（追加新提问）  
**使用工具**：`search_tech_knowledge`

**职责**：
1. 根据面试大纲的当前话题，生成具体的面试问题
2. 如果是追问（`next_action == "follow_up"`），根据之前的回答和评估反馈进行递进式追问
3. 调用 `search_tech_knowledge` 检索技术知识库，确保出题有据可依
4. 将问题以自然的面试官口吻提出

**System Prompt 核心要求**：
```
你是一位专业的技术面试官。你的风格是：
1. 先从基础概念开始，根据回答质量逐步深入
2. 对模糊的回答会追问细节
3. 注意倾听候选人的回答，问题要有逻辑连贯性
4. 面试语气专业但友善
5. 如果候选人提到简历中的项目经历，可以围绕实际经验深入提问
```

#### 4.3.3 Evaluator Agent

**输入**：`current_question`, `messages`（包含用户最新回答）, 知识库参考答案  
**输出**：`current_evaluation`, `all_evaluations`（追加）  
**使用工具**：`evaluate_answer`, `evaluate_code`

**职责**：
1. 对用户的回答进行多维度评估
2. 调用 `evaluate_answer` 工具，输出标准化评分
3. 如果回答中包含代码，调用 `evaluate_code` 进行代码质量分析
4. 生成评估结果：

```python
evaluation = {
    "question": "...",
    "answer": "...",
    "scores": {
        "completeness": 8,     # 完整度（0-10）：是否覆盖了问题的所有要点
        "accuracy": 7,         # 准确度（0-10）：技术细节是否正确
        "depth": 6,            # 深度（0-10）：是否有深入的理解和分析
        "expression": 8        # 表达（0-10）：是否条理清晰、逻辑通顺
    },
    "overall_score": 7.25,     # 综合评分
    "feedback": "...",         # 简要反馈
    "follow_up_suggestion": "建议追问..."  # 追问建议
}
```

**System Prompt 核心要求**：
```
你是一位严格但公正的面试评审官。你的评估标准是：
1. 完整度：回答是否覆盖了问题的所有关键要点
2. 准确度：技术细节是否正确，有无知识性错误
3. 深度：是否展示了对底层原理的理解，而非只是表面知识
4. 表达：是否条理清晰，有逻辑地组织答案
5. 你需要同时给出评分数字和文字反馈
6. 你需要建议是否值得追问以及追问方向
```

#### 4.3.4 Router 节点（条件路由）

**输入**：`current_evaluation`, `follow_up_count`, `current_topic_index`, `total_questions_asked`  
**输出**：`next_action`

Router 不是 LLM Agent，而是一个**纯逻辑函数**，根据规则决定下一步：

```python
def route_decision(state: InterviewState) -> str:
    eval_score = state["current_evaluation"]["overall_score"]
    follow_ups = state["follow_up_count"]
    topic_idx = state["current_topic_index"]
    total_asked = state["total_questions_asked"]
    plan = state["interview_plan"]
    
    # 规则 1：达到最大提问数 → 结束
    if total_asked >= state["max_questions"]:
        return "end_interview"
    
    # 规则 2：回答评分低且还没追问满 → 追问
    if eval_score < 6.0 and follow_ups < state["max_follow_ups"]:
        return "follow_up"
    
    # 规则 3：还有更多话题 → 下一题
    if topic_idx + 1 < len(plan):
        return "next_question"
    
    # 规则 4：所有话题问完 → 结束
    return "end_interview"
```

#### 4.3.5 Reporter Agent

**输入**：`all_evaluations`, `resume_analysis`, `interview_plan`, `messages`  
**输出**：`final_report`

**职责**：
1. 汇总所有评估结果
2. 计算各维度平均分
3. 生成结构化面试报告（Markdown 格式），包含：
   - 面试概览（岗位、时间、题目数）
   - 分维度得分雷达分析
   - 每道题目的表现回顾
   - 强项分析（做得好的方面）
   - 弱项分析（需要改进的方面）
   - 针对性改进建议（基于弱项给出具体学习建议）
   - 总体评语和推荐等级（A / B / C / D）

**System Prompt 核心要求**：
```
你是一位面试总结官。请根据以下面试全过程，生成一份详细的面试评估报告：
1. 用 Markdown 格式输出
2. 包含各维度评分的总结
3. 分析候选人的强项和弱项
4. 给出具体的改进建议，建议要可操作
5. 给出最终评级（A=优秀，B=良好，C=合格，D=需改进）
6. 整体报告控制在 800-1200 字
```

---

## 五、5 个工具函数详细设计

### 5.1 Tool 1: parse_resume（简历解析）

```python
@tool
def parse_resume(file_path: str) -> str:
    """
    解析用户简历文件（支持 PDF、Markdown、TXT），提取结构化信息。
    
    返回 JSON 字符串，包含：
    - name: 候选人姓名
    - education: 教育背景列表
    - skills: 技能列表
    - projects: 项目经历列表
    - experience: 工作/研究经历列表
    - competitions: 竞赛荣誉列表
    """
```

**实现要点**：
- 使用 LangChain 的 `PyPDFLoader` 或 `UnstructuredMarkdownLoader` 加载文件
- 将加载的文档内容发送给 LLM，由 LLM 提取结构化信息
- 同时将简历内容向量化存入 FAISS，供后续 RAG 检索

### 5.2 Tool 2: search_job_requirements（岗位要求检索）

```python
@tool
def search_job_requirements(role: str) -> str:
    """
    从知识库检索指定岗位的技术要求和常见面试题。
    
    参数：
    - role: 目标岗位名称（如 "后端开发"、"推荐系统"、"AI Agent"）
    
    返回：岗位要求描述，包含必备技能、加分项、常见面试方向。
    """
```

**实现要点**：
- 从 `knowledge_base/jobs/` 目录读取预置的岗位要求 JSON
- 如果精确匹配不到岗位名称，使用 FAISS 向量检索找最相近的岗位
- 返回格式化的岗位要求文本

### 5.3 Tool 3: search_tech_knowledge（技术知识 RAG 检索）

```python
@tool
def search_tech_knowledge(query: str) -> str:
    """
    从技术知识库进行语义检索，返回与查询最相关的知识片段。
    用于 Interviewer 出题时参考和 Evaluator 评估时对照参考答案。
    
    参数：
    - query: 检索查询（如 "Python GIL 原理"、"B+树和哈希索引的区别"）
    
    返回：最相关的 top-3 知识片段，包含来源标注。
    """
```

**实现要点**：
- 使用 FAISS 向量库进行相似度检索
- 返回 top-3 相关文档片段
- 知识库来源为 `knowledge_base/tech/` 目录下的 Markdown 文件

### 5.4 Tool 4: evaluate_answer（回答多维评估）

```python
@tool
def evaluate_answer(question: str, answer: str, reference: str = "") -> str:
    """
    对候选人的回答进行多维度评估。
    
    参数：
    - question: 面试问题
    - answer: 候选人回答
    - reference: 参考答案（可选，来自 RAG 检索）
    
    返回 JSON 字符串，包含 completeness/accuracy/depth/expression 四个维度评分（0-10）和反馈。
    """
```

**实现要点**：
- 内部调用 LLM 进行评估
- 传入评分标准（rubric）让 LLM 输出结构化评分
- 使用 JSON mode 或 Output Parser 确保输出格式

### 5.5 Tool 5: evaluate_code（代码质量分析）

```python
@tool
def evaluate_code(code: str, language: str = "python") -> str:
    """
    分析代码质量，评估正确性、效率和可读性。
    
    参数：
    - code: 候选人提供的代码
    - language: 编程语言
    
    返回 JSON 字符串，包含正确性评估、时间复杂度分析、代码质量建议。
    """
```

**实现要点**：
- 调用 LLM 分析代码
- 如果检测到代码中有明显的语法错误，直接指出
- 分析时间/空间复杂度
- 给出改进建议

---

## 六、RAG 知识库设计

### 6.1 知识库架构

```
知识库分为 3 层：

1. 简历库（动态）
   └── 用户上传的简历 → 向量化后存入 FAISS session 级索引
   
2. 技术知识库（预置 + 可扩展）
   └── knowledge_base/tech/ 下的 Markdown 文件
   └── 覆盖：Python 基础、数据结构、系统设计、机器学习、推荐系统等
   
3. 岗位要求库（预置）
   └── knowledge_base/jobs/ 下的 JSON 文件
   └── 覆盖：后端工程师、ML 工程师、AI Agent 开发、推荐系统工程师
```

### 6.2 向量化流程

```python
# 文档加载 → 文本切分 → 向量化 → 存入 FAISS

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载文档
docs = load_documents_from_directory("knowledge_base/tech/")

# 2. 文本切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量化 (使用国内最强开源 BGE 模型)
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示用于检索相关文章："
)

# 4. 存入 FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

### 6.3 预置知识库内容指引

开发时需要为 `knowledge_base/tech/` 创建以下 Markdown 文件（每个 2000~5000 字）：

| 文件名 | 内容范围 |
|---|---|
| `python_basics.md` | Python 核心机制（GIL、内存管理、装饰器、生成器、异步等） |
| `data_structures.md` | 常见数据结构与算法（数组、链表、树、图、排序、搜索、动态规划） |
| `system_design.md` | 系统设计基础（负载均衡、缓存、消息队列、数据库选型、微服务） |
| `machine_learning.md` | 机器学习基础（监督/非监督、常见算法、评价指标、过拟合） |
| `recommender_systems.md` | 推荐系统（协同过滤、序列推荐、多模态推荐、冷启动） |

岗位要求 JSON 格式：

```json
{
  "role": "后端开发工程师",
  "required_skills": ["Python", "SQL", "HTTP", "Linux", "Git"],
  "preferred_skills": ["Redis", "Docker", "消息队列", "微服务"],
  "interview_topics": ["语言基础", "数据库", "系统设计", "网络", "并发"],
  "difficulty_distribution": {"easy": 2, "medium": 4, "hard": 2}
}
```

---

## 七、API 设计

### 7.1 RESTful API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/interview/start` | 上传简历 + 选择岗位，启动面试 |
| `POST` | `/api/interview/{session_id}/answer` | 提交回答 |
| `GET` | `/api/interview/{session_id}/status` | 查询面试状态 |
| `GET` | `/api/interview/{session_id}/report` | 获取面试报告 |
| `GET` | `/api/interview/{session_id}/report/export?format=pdf` | 导出报告为 PDF 文件 |
| `GET` | `/api/interview/{session_id}/report/export?format=md` | 导出报告为 Markdown 文件 |
| `GET` | `/api/interview/{session_id}/history` | 获取对话历史 |
| `POST` | `/api/knowledge/upload` | 上传额外知识库文档（可选） |

### 7.2 请求/响应模型

```python
# --- 启动面试 ---
class StartInterviewRequest(BaseModel):
    candidate_name: str
    job_role: str               # 目标岗位
    max_questions: int = 8      # 最大提问数
    max_follow_ups: int = 2     # 每题最大追问次数
    # 简历通过 multipart/form-data 上传 (UploadFile)

class StartInterviewResponse(BaseModel):
    session_id: str
    message: str                # Planner 的初始分析摘要
    interview_plan: list[dict]  # 面试大纲
    first_question: str         # 第一个面试问题

# --- 提交回答 ---
class AnswerRequest(BaseModel):
    answer: str                 # 用户的回答

class AnswerResponse(BaseModel):
    evaluation: dict            # 当前回答评估（仅调试模式返回）
    next_question: str | None   # 下一个问题（如果面试未结束）
    is_follow_up: bool          # 是否是追问
    interview_ended: bool       # 面试是否结束
    progress: str               # 进度信息（如 "3/8"）

# --- 面试报告 ---
class ReportResponse(BaseModel):
    report: str                 # Markdown 格式的面试报告
    overall_score: float        # 总分
    grade: str                  # 等级 A/B/C/D
    evaluations: list[dict]     # 所有题目的评估详情
```

### 7.3 面试交互流程（时序图）

```
用户                      FastAPI                    LangGraph
 │                          │                           │
 │  POST /start (简历+岗位)  │                           │
 │ ────────────────────────► │                           │
 │                          │   invoke(Planner)          │
 │                          │ ─────────────────────────► │
 │                          │   ◄───── plan + question   │
 │  ◄──── first_question    │                           │
 │                          │                           │
 │  POST /answer (回答)      │                           │
 │ ────────────────────────► │                           │
 │                          │   invoke(Evaluator)        │
 │                          │ ─────────────────────────► │
 │                          │   invoke(Router)           │
 │                          │ ─────────────────────────► │
 │                          │   invoke(Interviewer)      │ (if not ended)
 │                          │ ─────────────────────────► │
 │  ◄──── next_question     │                           │
 │                          │                           │
 │  ... (多轮对话) ...       │                           │
 │                          │                           │
 │                          │   invoke(Reporter)         │ (interview ended)
 │                          │ ─────────────────────────► │
 │  GET /report              │                           │
 │ ────────────────────────► │                           │
 │  ◄──── final_report      │                           │
```

---

## 八、LangGraph 状态图构建代码骨架

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_interview_graph():
    """构建面试流程的 LangGraph 状态图"""
    
    graph = StateGraph(InterviewState)
    
    # 1. 添加节点
    graph.add_node("planner", planner_node)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("router", router_node)
    graph.add_node("reporter", reporter_node)
    
    # 2. 设置入口
    graph.set_entry_point("planner")
    
    # 3. 添加边
    graph.add_edge("planner", "interviewer")          # Planner 完成后进入 Interviewer
    graph.add_edge("interviewer", "__interrupt__")     # Interviewer 提问后中断等待用户输入
    # 用户回答后从 evaluator 继续
    graph.add_edge("evaluator", "router")              # 评估完进路由
    
    # 4. 条件路由
    graph.add_conditional_edges(
        "router",
        route_decision,  # 路由函数
        {
            "follow_up": "interviewer",     # 追问 → 回到 Interviewer
            "next_question": "interviewer", # 下一题 → 回到 Interviewer
            "end_interview": "reporter"     # 结束 → 进入 Reporter
        }
    )
    
    graph.add_edge("reporter", END)  # Reporter 完成 → 结束
    
    # 5. 编译
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
```

### 8.1 Human-in-the-Loop（等待用户输入）

LangGraph 支持 `interrupt` 机制来实现 human-in-the-loop。关键实现方式：

```python
from langgraph.types import interrupt

def interviewer_node(state: InterviewState):
    """Interviewer Agent 节点：提问后使用 interrupt 等待用户回答"""
    # ... 生成问题逻辑 ...
    question = generate_question(state)
    
    # 使用 interrupt 暂停图执行，等待外部输入
    user_answer = interrupt({"question": question})
    
    # 用户回答后继续执行
    return {
        "messages": [AIMessage(content=question), HumanMessage(content=user_answer)],
        "current_question": question,
    }
```

在 FastAPI 层：
```python
# 启动面试时
config = {"configurable": {"thread_id": session_id}}
result = graph.invoke(initial_state, config)

# 用户回答时
result = graph.invoke(Command(resume=user_answer), config)
```

---

## 九、前端设计（LeetCode 风格 · 白灰配色）

> 设计灵感来源于 LeetCode 的专业、干净、有质感的 UI 风格。
> 技术方案：纯 HTML + CSS + JavaScript，无框架依赖，由 FastAPI 提供静态文件服务。

### 9.1 设计系统（Design Tokens）

#### 配色方案（白灰为主，偏白）

```css
:root {
  /* 主色调 */
  --color-bg-primary: #FFFFFF;           /* 页面主背景：纯白 */
  --color-bg-secondary: #F7F8FA;         /* 面板/卡片背景：极浅灰（LeetCode 风） */
  --color-bg-tertiary: #EFF0F2;          /* 输入框/代码块背景 */
  --color-bg-hover: #F0F1F3;             /* 悬停态背景 */

  /* 边框 */
  --color-border-primary: #E3E5E8;       /* 主分隔线 */
  --color-border-secondary: #EBEDF0;     /* 次级分隔线 */
  --color-border-focus: #3C8CFF;         /* 聚焦态边框（蓝色） */

  /* 文字 */
  --color-text-primary: #262626;         /* 主文字：深灰近黑 */
  --color-text-secondary: #595959;       /* 次级文字 */
  --color-text-tertiary: #8C8C8C;        /* 辅助/占位文字 */
  --color-text-inverse: #FFFFFF;         /* 反色文字（在深色按钮上） */

  /* 功能色 */
  --color-accent: #3C8CFF;               /* 主强调色：LeetCode 蓝 */
  --color-accent-hover: #2B7AE8;         /* 强调色悬停 */
  --color-success: #52C41A;              /* 成功/通过：绿 */
  --color-warning: #FAAD14;              /* 警告/中等：橙黄 */
  --color-danger: #FF4D4F;               /* 错误/低分：红 */
  --color-info: #3C8CFF;                 /* 信息色（同主强调） */

  /* 阴影 */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.06);
  --shadow-md: 0 2px 8px rgba(0,0,0,0.08);
  --shadow-lg: 0 4px 16px rgba(0,0,0,0.10);

  /* 圆角 */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;

  /* 字体 */
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
  --font-mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;

  /* 尺寸 */
  --sidebar-width: 300px;
  --header-height: 56px;
}
```

#### 字体规范

| 用途 | 字号 | 字重 | 颜色 |
|---|---|---|---|
| 页面标题 | 20px | 600 | `--color-text-primary` |
| 面板标题 | 14px | 600 | `--color-text-primary` |
| 正文/消息 | 14px | 400 | `--color-text-primary` |
| 辅助文字 | 12px | 400 | `--color-text-tertiary` |
| 代码块 | 13px | 400 | `--font-mono` |
| 按钮 | 14px | 500 | 视背景色 |

### 9.2 页面布局（三栏 LeetCode 风格）

整体采用 LeetCode 经典的三栏布局，上方固定导航栏：

```
┌────────────────────────────────────────────────────────────────────┐
│  Header: Logo + BaguRush + 面试进度 (3/8) + 结束面试按钮     │
├──────────────┬─────────────────────────────┬───────────────────────┤
│              │                             │                       │
│  左侧面板     │     中央主区域               │    右侧面板            │
│  (300px)     │     (flex-1, 自适应)         │    (320px)            │
│              │                             │                       │
│  ┌────────┐  │  ┌─────────────────────┐    │  ┌───────────────┐    │
│  │简历上传 │  │  │                     │    │  │ 面试大纲/话题  │    │
│  │        │  │  │   聊天对话区域        │    │  │ 列表          │    │
│  ├────────┤  │  │                     │    │  │ (当前高亮)     │    │
│  │岗位选择 │  │  │  面试官 💬           │    │  ├───────────────┤    │
│  │        │  │  │                     │    │  │ 当前评估       │    │
│  ├────────┤  │  │       候选人 💬      │    │  │ (实时更新)     │    │
│  │面试参数 │  │  │                     │    │  ├───────────────┤    │
│  │(题目数) │  │  │  面试官 💬           │    │  │ 四维度分数     │    │
│  │        │  │  │                     │    │  │ 条形图        │    │
│  ├────────┤  │  │       ...           │    │  │               │    │
│  │开始面试 │  │  │                     │    │  │               │    │
│  │[按钮]   │  │  └─────────────────────┘    │  └───────────────┘    │
│  │        │  │  ┌─────────────────────┐    │                       │
│  └────────┘  │  │ 输入区域 + 提交按钮   │    │                       │
│              │  └─────────────────────┘    │                       │
├──────────────┴─────────────────────────────┴───────────────────────┤
│  （面试结束后：中央区域切换为「面试报告」全屏展示）                      │
└────────────────────────────────────────────────────────────────────┘
```

### 9.3 三栏详细设计

#### 左侧面板（Setup Panel）

- **背景**：`--color-bg-secondary`，右边框 `--color-border-primary`
- **面试前**（可交互）：
  - 📄 简历上传区：虚线边框拖拽区 + 点击上传，支持 PDF/MD
  - 🎯 岗位选择：`<select>` 下拉框，选项从后端 `/api/jobs` 获取或硬编码
  - ⚙️ 面试参数：题目数量 slider（3~10，默认 8）、每题追问次数（1~3，默认 2）
  - 🚀 开始面试按钮：`--color-accent` 蓝色实心按钮，全宽
- **面试中**（只读信息展示）：
  - 已上传简历文件名
  - 当前岗位
  - 面试进度：`3 / 8 题`
  - 一个「结束面试」危险按钮（`--color-danger` 边框按钮）

- **⭐ Agent 流水线进度指示器**（左侧面板底部，`#agent-pipeline`）：

  在点击「开始面试」后（Planner 运行期间）和每次用户回答后（Evaluator/Router 运行期间），左侧面板底部动态展示当前后端 Agent 的工作进度。当 Agent 全部空闲时（等待用户回答），该区域显示 "✅ 等待你的回答"。

  ```
  ┌─────────────────────────┐
  │  🤖 Agent 工作状态       │
  ├─────────────────────────┤
  │  ✅ 简历解析完成          │
  │  ✅ 岗位需求检索完成      │
  │  ⏳ 正在生成面试大纲...   │  ← 当前步骤（蓝色 + 动态省略号）
  │  ○ 生成第一题            │  ← 未开始（灰色）
  │                         │
  │  ▓▓▓▓▓▓▓▓░░░░  65%     │  ← 总进度条
  └─────────────────────────┘
  ```

  **实现方式**：
  - 后端 API 在 `StartInterviewResponse` 和 `AnswerResponse` 中增加 `pipeline_steps` 字段，返回当前 Agent 执行链的各步骤状态
  - 或者前端在发起请求后，按固定阶段展示模拟进度（无需后端额外支持）：
    1. 面试启动时：「解析简历 → 检索岗位需求 → 生成面试大纲 → 生成第一题」
    2. 每次回答后：「评估回答 → 分析路由 → 准备下一题」
  - 每个步骤有三种状态：✅ 已完成（绿色）、⏳ 进行中（蓝色 + 动画）、○ 未开始（灰色）
  - 底部有一个总进度条，`width` 动画过渡 `500ms ease-in-out`
  - 进度步骤文字使用 `12px` 辅助字体大小，紧凑排列

  **CSS 样式要点**：
  ```css
  .pipeline-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 12px;
    color: var(--color-text-tertiary);
  }
  .pipeline-step.active {
    color: var(--color-accent);
    font-weight: 500;
  }
  .pipeline-step.done {
    color: var(--color-success);
  }
  .pipeline-progress-bar {
    height: 4px;
    background: var(--color-bg-tertiary);
    border-radius: 2px;
    margin-top: 8px;
  }
  .pipeline-progress-fill {
    height: 100%;
    background: var(--color-accent);
    border-radius: 2px;
    transition: width 500ms ease-in-out;
  }
  ```

#### 中央主区域（Chat Area）

- **背景**：`--color-bg-primary`（纯白）
- **聊天消息样式**（关键视觉元素）：

  > ⚠️ **重要：消息对齐规则**  
  > 每条消息需要用 `.msg-row` 包装层控制水平对齐方向。  
  > **面试官消息 → 左对齐**（`justify-content: flex-start`）  
  > **候选人消息 → 右对齐**（`justify-content: flex-end`）  
  > 消息气泡本身不负责对齐，对齐由外层 `.msg-row` 容器负责。

  **消息行容器**（控制左右对齐）：
  ```css
  .msg-row {
    display: flex;
    width: 100%;
    margin-bottom: 16px;
  }
  .msg-row.interviewer {
    justify-content: flex-start;   /* 面试官消息靠左 */
  }
  .msg-row.candidate {
    justify-content: flex-end;     /* 候选人消息靠右 */
  }
  ```

  **面试官消息气泡**（左对齐，浅灰背景）：
  ```css
  .msg-interviewer {
    background: var(--color-bg-secondary);  /* 浅灰背景 */
    border: 1px solid var(--color-border-secondary);
    border-radius: 2px 12px 12px 12px;       /* 左上直角，其余圆角 */
    padding: 12px 16px;
    max-width: 75%;
    color: var(--color-text-primary);
    font-size: 14px;
    line-height: 1.7;
  }
  ```
  - 左侧有一个小圆形头像（蓝色圆圈 + "IF" 字母，代表 BaguRush）
  - 消息下方显示时间戳 `12px --color-text-tertiary`

  **候选人消息气泡**（右对齐，蓝色背景）：
  ```css
  .msg-candidate {
    background: var(--color-accent);           /* 蓝色背景 */
    color: var(--color-text-inverse);          /* 白色文字 */
    border-radius: 12px 2px 12px 12px;         /* 右上直角 */
    padding: 12px 16px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.7;
  }
  ```

  **HTML 结构示例**（每条消息的 DOM 嵌套）：
  ```html
  <!-- 面试官消息 —— 靠左 -->
  <div class="msg-row interviewer">
    <div class="avatar">IF</div>
    <div class="msg-content">
      <div class="msg-interviewer">请介绍一下 Python 的 GIL...</div>
      <span class="msg-time">14:32</span>
    </div>
  </div>

  <!-- 候选人消息 —— 靠右 -->
  <div class="msg-row candidate">
    <div class="msg-content">
      <div class="msg-candidate">GIL 是全局解释器锁...</div>
      <span class="msg-time">14:33</span>
    </div>
  </div>
  ```

  **系统消息**（居中，灰色小字）：
  ```css
  .msg-system {
    text-align: center;
    color: var(--color-text-tertiary);
    font-size: 12px;
    padding: 8px 0;
  }
  /* 例如: "Planner 已完成面试规划，共 6 个话题" */
  ```

- **输入区域**（底部固定）：
  - `<textarea>` 多行输入框，背景 `--color-bg-tertiary`，边框 `--color-border-primary`
  - 聚焦时边框变 `--color-border-focus`（蓝色）+ 浅蓝阴影
  - 右侧「提交回答」按钮，蓝色实心
  - 支持 `Ctrl+Enter` 快捷提交
  - Placeholder: "请输入你的回答... (Ctrl+Enter 提交)"

- **加载状态**：
  - 面试官思考中显示三个跳动的灰色圆点动画（typing indicator）
  - 消息区域底部平滑滚动到最新消息（`scrollIntoView({ behavior: 'smooth' })`）

#### 右侧面板（Evaluation Panel）

- **背景**：`--color-bg-secondary`，左边框 `--color-border-primary`

- **面试大纲卡片**：
  ```
  面试大纲                    ▾
  ─────────────────────
  ● Python 基础        ✅ (已完成)
  ● 数据结构与算法      ⬜ (当前，蓝色高亮)
  ○ 系统设计           ○ (未开始)
  ○ 推荐系统           ○
  ○ 项目经历           ○
  ```
  - 当前话题行：左边框 3px `--color-accent` 蓝色条 + 背景微蓝 `rgba(60,140,255,0.06)`
  - 已完成话题：`--color-success` 绿色勾号
  - 未开始话题：灰色空心圆

- **实时评估卡片**（每次回答后更新）：
  - 标题："上一题评估" + 总分 badge
  - 四维度水平进度条：
    ```
    完整度  ████████░░  8/10
    准确度  ███████░░░  7/10
    深度    ██████░░░░  6/10
    表达    ████████░░  8/10
    ```
  - 条形颜色规则：
    - 8-10：`--color-success`（绿）
    - 5-7：`--color-warning`（橙黄）
    - 0-4：`--color-danger`（红）
  - 下方显示简短反馈文字

### 9.4 面试报告页面

面试结束后，中央区域切换为全宽报告展示页（左右面板可收起或覆盖）：

```
┌────────────────────────────────────────────┐
│  🏆 面试报告                                │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 总分      │  │ 等级      │  │ 题目数    │  │
│  │  7.5/10   │  │   B+      │  │  6 题     │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │ 各维度平均分 (水平条形图)              │  │
│  │ 完整度  ████████░░  7.8              │  │
│  │ 准确度  ███████░░░  7.2              │  │
│  │ 深度    ██████░░░░  6.1              │  │
│  │ 表达    ████████░░  8.0              │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ## 💪 强项                                │
│  - 表达能力突出，回答条理清晰              │
│  - Python 基础扎实...                     │
│                                            │
│  ## ⚠️ 待改进                              │
│  - 系统设计部分深度不足...                 │
│                                            │
│  ## 📚 改进建议                            │
│  1. 建议阅读《DDIA》...                    │
│                                            │
│  ## 📝 逐题回顾                            │
│  ┌─────────────────────────────────────┐   │
│  │ Q1: Python 的 GIL 是什么？          │   │
│  │ 得分: 8.0  |  ✅ 回答完整            │   │
│  ├─────────────────────────────────────┤   │
│  │ Q2: 解释 B+ 树的结构                │   │
│  │ 得分: 6.5  |  ⚠️ 深度略有不足        │   │
│  └─────────────────────────────────────┘   │
│                                            │
│  [导出 PDF ⬇]  [导出 Markdown ⬇]  [重新开始面试]  │
└────────────────────────────────────────────┘
```

- 报告内容由后端返回 Markdown，前端使用 `markdown-it` 库渲染为 HTML
- 顶部统计卡片使用 `display: flex; gap: 16px;` 水平排列
- 逐题回顾使用可折叠的 `<details>` 元素或手风琴组件

**报告导出功能**：
- 支持两种导出格式：**PDF** 和 **Markdown**
- **Markdown 导出**（前端实现）：
  - 直接将后端返回的原始 Markdown 报告文本创建为 `.md` 文件下载
  - 使用 `Blob` + `URL.createObjectURL` + `<a download>` 实现纯前端下载
  ```javascript
  function exportMarkdown(reportMd) {
    const blob = new Blob([reportMd], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `BaguRush_Report_${state.sessionId.slice(0,8)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }
  ```
- **PDF 导出**（后端实现）：
  - 后端新增 API：`GET /api/interview/{session_id}/report/export?format=pdf`
  - 使用 Python 库（如 `markdown` + `weasyprint` 或 `fpdf2`）将 Markdown 转为 PDF
  - 返回 `Content-Type: application/pdf` 的文件流
  - 前端通过 `window.open()` 或 `fetch` + `Blob` 触发浏览器下载
  ```python
  # requirements.txt 新增
  # weasyprint>=60.0  或  fpdf2>=2.7.0
  
  @app.get("/api/interview/{session_id}/report/export")
  async def export_report(session_id: str, format: str = "pdf"):
      report_md = get_report_markdown(session_id)
      if format == "md":
          return Response(content=report_md, media_type="text/markdown",
                          headers={"Content-Disposition": f"attachment; filename=report.md"})
      elif format == "pdf":
          pdf_bytes = markdown_to_pdf(report_md)
          return Response(content=pdf_bytes, media_type="application/pdf",
                          headers={"Content-Disposition": f"attachment; filename=report.pdf"})
  ```
- **导出按钮样式**：
  - 两个按钮并排，使用 `--color-accent` 边框样式（outline 按钮），hover 时填充蓝色
  - 「重新开始面试」按钮使用灰色 outline 样式，与导出按钮有视觉区分

### 9.5 交互状态与动效

| 状态 | 视觉表现 |
|---|---|
| **空闲**（面试前） | 左侧面板可交互，中央区域显示欢迎语 + 使用说明 |
| **规划中**（Planner 运行） | 中央显示 "AI 正在分析您的简历..." + 加载动画 |
| **提问中**（等待用户回答） | 输入框激活，显示当前题号 |
| **思考中**（Agent 处理） | 左下角 typing indicator 跳动圆点 + 输入框禁用 |
| **面试结束** | 切换到报告视图 |

**微动效**：
- 新消息出现：`opacity 0→1 + translateY(8px→0)` 过渡动画，`300ms ease-out`
- 按钮悬停：`background-color` 过渡 `150ms` + 轻微 `translateY(-1px)` 上浮
- 进度条更新：`width` 过渡 `500ms ease-in-out`
- 面板切换（聊天→报告）：`opacity` 交叉淡出 `200ms`

### 9.6 前端文件结构

```
frontend/
├── index.html              # 页面结构（语义化 HTML5）
│
├── style.css               # 样式文件，包含：
│   ├── /* === 设计系统 === */     # CSS 变量、reset、字体
│   ├── /* === 布局 === */         # 三栏布局、header
│   ├── /* === 左侧面板 === */     # setup-panel 组件样式
│   ├── /* === 聊天区域 === */     # 消息气泡、输入框、typing indicator
│   ├── /* === 右侧面板 === */     # 大纲、评估卡片、进度条
│   ├── /* === 报告页 === */       # 报告卡片、分数图表、手风琴
│   └── /* === 动效 === */         # transitions、keyframes
│
├── app.js                  # 交互逻辑，包含：
│   ├── // === 状态管理 ===        # 全局 state 对象
│   ├── // === API 对接 ===        # startInterview(), submitAnswer(), getReport()
│   ├── // === DOM 渲染 ===        # renderMessage(), renderPlan(), renderEval()
│   ├── // === 事件绑定 ===        # 按钮点击、文件拖拽、键盘快捷键
│   └── // === 工具函数 ===        # formatTime(), scrollToBottom(), togglePanel()
│
└── markdown-it.min.js      # Markdown 渲染库（约 50KB）
```

### 9.7 关键 JavaScript API 调用示例

```javascript
// 全局状态
const state = {
  sessionId: null,
  messages: [],
  plan: [],
  currentTopicIndex: 0,
  isInterviewing: false,
  isLoading: false,
};

// 开始面试
async function startInterview() {
  const formData = new FormData();
  formData.append('resume', document.getElementById('resume-input').files[0]);
  formData.append('candidate_name', document.getElementById('name-input').value);
  formData.append('job_role', document.getElementById('role-select').value);
  formData.append('max_questions', document.getElementById('questions-slider').value);

  const res = await fetch('/api/interview/start', { method: 'POST', body: formData });
  const data = await res.json();

  state.sessionId = data.session_id;
  state.plan = data.interview_plan;
  state.isInterviewing = true;

  renderSystemMessage('面试规划完成，共 ' + state.plan.length + ' 个话题');
  renderInterviewerMessage(data.first_question);
  renderPlan(state.plan);
  switchToInterviewMode();
}

// 提交回答
async function submitAnswer() {
  const answer = document.getElementById('answer-textarea').value.trim();
  if (!answer || state.isLoading) return;

  renderCandidateMessage(answer);
  document.getElementById('answer-textarea').value = '';
  state.isLoading = true;
  showTypingIndicator();

  const res = await fetch(`/api/interview/${state.sessionId}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer }),
  });
  const data = await res.json();

  hideTypingIndicator();
  state.isLoading = false;

  if (data.evaluation) renderEvaluation(data.evaluation);
  updateProgress(data.progress);

  if (data.interview_ended) {
    renderSystemMessage('面试结束，正在生成报告...');
    await showReport();
  } else {
    renderInterviewerMessage(data.next_question);
  }
}

// 获取并展示报告
async function showReport() {
  const res = await fetch(`/api/interview/${state.sessionId}/report`);
  const data = await res.json();
  const md = window.markdownit();
  document.getElementById('report-content').innerHTML = md.render(data.report);
  switchToReportView();
}
```

---

## 十、关键实现注意事项

### 10.1 LLM 调用配置

所有 Agent 统一使用 `ChatOpenAI` 类（OpenAI 兼容接口）：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.7,
    max_tokens=2000,
)
```

### 10.2 Tool Binding

Agent 绑定工具的方式：

```python
from langchain_core.tools import tool

# 定义工具
@tool
def parse_resume(file_path: str) -> str:
    """解析简历..."""
    ...

# 绑定到 LLM
planner_llm = llm.bind_tools([parse_resume, search_job_requirements])
```

### 10.3 Prompt Engineering 原则

- 所有 System Prompt 使用**中文**编写（因为用户和面试模拟场景都是中文）
- Prompt 中明确输出格式要求（JSON schema 或 Markdown 结构）
- 使用 Few-shot 示例提升输出质量
- 每个 Agent 的 Prompt 独立存放在 `prompts/` 目录

### 10.4 错误处理

- LLM API 调用失败时使用重试机制（3 次 exponential backoff）
- JSON 解析失败时使用 LangChain 的 `OutputFixingParser`
- 文件上传失败时返回明确的错误信息
- 向量检索结果为空时使用默认知识进行补充

### 10.5 会话管理

- 每个面试会话使用 UUID 作为 `session_id`
- LangGraph 的 `MemorySaver` 保存每个会话的状态检查点
- API 层通过 `session_id` 映射到 LangGraph 的 `thread_id`
- 面试结束后保留报告，但可清理中间状态

---

## 十一、运行方式

### 11.1 本地开发

```bash
# 1. 克隆项目
git clone https://github.com/lulusiyuyu/bagurush.git
cd bagurush

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 5. 初始化知识库向量索引（首次运行）
python -m rag.vector_store --init

# 6. 启动服务（同时提供 API + 前端静态文件）
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 7. 打开浏览器访问前端
# http://localhost:8000/frontend/index.html
```

### 11.2 API 测试

```bash
# 启动面试
curl -X POST http://localhost:8000/api/interview/start \
  -F "resume=@resume.pdf" \
  -F "candidate_name=张三" \
  -F "job_role=后端开发"

# 提交回答
curl -X POST http://localhost:8000/api/interview/{session_id}/answer \
  -H "Content-Type: application/json" \
  -d '{"answer": "Python 的 GIL 是全局解释器锁..."}'

# 获取报告
curl http://localhost:8000/api/interview/{session_id}/report
```

---

## 十二、预期产出物

完成后，项目应包含：

1. ✅ 可运行的 FastAPI 后端服务
2. ✅ 基于 LangGraph 的 4 Agent 状态图
3. ✅ 5 个 Tool 函数（简历解析、岗位检索、知识 RAG、回答评估、代码分析）
4. ✅ FAISS 向量知识库（包含预置技术知识和岗位要求）
5. ✅ 前端交互界面（LeetCode 风格 HTML+CSS+JS，白灰配色）
6. ✅ 完整的面试流程（上传简历 → 提问 → 追问 → 评分 → 报告）
7. ✅ README.md（含项目介绍、架构图、运行方式、Demo 截图）
8. ✅ 单元测试覆盖核心逻辑
