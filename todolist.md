# 📋 BaguRush — 开发任务清单 (TodoList)

> **项目**：BaguRush — AI 模拟面试多 Agent 系统  
> **最后更新**：2026-03-06  
> **配套文档**：[ProjectContext.md](./ProjectContext.md) — 完整项目上下文（架构、设计、API 等）  
> **状态图例**：✅ 已完成 | ⏳ 进行中 | ⬜ 未开始 | 🔴 阻塞

---

## 开发总览

本项目分为 **6 个 Phase**，建议按顺序执行。每个 Phase 完成后应可独立验证。

| Phase | 名称 | 预计耗时 | 核心产出 |
|---|---|---|---|
| Phase 0 | 项目初始化 | 0.5h | 目录结构、依赖、环境变量 |
| Phase 1 | RAG 知识库 | 2~3h | FAISS 向量库 + 文档加载 + 预置知识内容 |
| Phase 2 | 工具函数 | 2~3h | 5 个 Tool 全部可独立调用并测试通过 |
| Phase 3 | Agent + 状态图 | 4~5h | 4 Agent + LangGraph 状态图跑通完整面试流程 |
| Phase 4 | FastAPI 后端 | 2~3h | RESTful API 可接收请求并返回面试结果 |
| Phase 5 | 前端 UI | 2~3h | 可交互的 Web 面试界面 |
| Phase 6 | 测试与完善 | 2~3h | 测试、Prompt 调优、README、Demo |

---

## Phase 0: 项目初始化

> 目标：搭建完整的项目骨架，所有依赖装好，确保环境可用。

- [x] **0.1** 创建项目根目录 `bagurush/`
- [x] **0.2** 创建完整目录结构（参考 ProjectContext.md 第三节）：
  ```
  bagurush/
  ├── agents/          # Agent 定义
  ├── tools/           # Tool 定义
  ├── rag/             # RAG 知识库管理
  ├── prompts/         # Prompt 模板
  ├── api/             # FastAPI 路由
  ├── knowledge_base/  # 预置知识库
  │   ├── tech/
  │   └── jobs/
  ├── uploads/         # 用户上传文件
  ├── frontend/        # 前端文件
  └── tests/           # 测试
  ```
- [x] **0.3** 创建 `requirements.txt`，包含以下核心依赖：
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
  sentence-transformers>=2.0.0
  ```
- [x] **0.4** 创建虚拟环境并安装依赖：
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- [x] **0.5** 创建 `.env.example` 和 `.env` 文件，配置 LLM API：
  ```bash
  DEEPSEEK_API_KEY=sk-xxx
  DEEPSEEK_BASE_URL=https://api.deepseek.com
  DEEPSEEK_MODEL=deepseek-chat
  EMBEDDING_MODEL=text-embedding-ada-002
  HOST=0.0.0.0
  PORT=8000
  ```
- [x] **0.6** 创建 `.gitignore`（排除 `.env`, `venv/`, `__pycache__/`, `uploads/`, `faiss_index/`）
- [x] **0.7** 创建所有 `__init__.py` 文件
- [x] **0.8** ✅ **验证点**：`python -c "import langgraph; import langchain; import faiss; import fastapi; print('OK')"` 通过

---

## Phase 1: RAG 知识库搭建

> 目标：完成向量知识库的搭建，包括嵌入模型封装、文档加载切分、FAISS 索引构建、预置知识内容编写。

### 1.1 嵌入模型封装 (`rag/embeddings.py`)

- [x] **1.1.1** 创建 `get_embeddings()` 函数，实现国内平替方案：
  - **首选方案**：使用 `HuggingFaceBgeEmbeddings` 加载本地 `BAAI/bge-small-zh-v1.5`（中文检索最强）。
  - **API 方案**：对接 `ZhipuAIEmbeddings`（智谱 AI）或通义千问嵌入。
- [x] **1.1.2** 支持通过环境变量 `EMBEDDING_PROVIDER` 切换：`local` 或 `api`。

### 1.2 文档加载器 (`rag/document_loader.py`)

- [x] **1.2.1** 创建 `load_document(file_path)` 函数，根据文件后缀选择 Loader：
  - `.pdf` → `PyPDFLoader`
  - `.md` → `UnstructuredMarkdownLoader` 或直接读取文本
  - `.txt` → `TextLoader`
- [x] **1.2.2** 创建 `load_directory(dir_path)` 函数，批量加载目录下所有文档
- [x] **1.2.3** 创建 `split_documents(docs, chunk_size=500, chunk_overlap=50)` 函数，使用 `RecursiveCharacterTextSplitter` 切分文档

### 1.3 向量存储管理 (`rag/vector_store.py`)

- [x] **1.3.1** 创建 `VectorStoreManager` 类，封装 FAISS 操作：
  - `__init__(index_path)` — 初始化，加载或创建索引
  - `build_index(documents)` — 从文档列表构建索引
  - `add_documents(documents)` — 追加文档到已有索引
  - `search(query, k=3)` — 语义检索 top-k 结果
  - `save()` / `load()` — 持久化索引到磁盘
- [x] **1.3.2** 实现 `--init` 命令行模式，运行 `python -m rag.vector_store --init` 可一次性构建预置知识库的索引
- [x] **1.3.3** 支持为每个面试会话创建临时的简历索引（session-level）

### 1.4 预置知识库内容编写 (`knowledge_base/`)

- [x] **1.4.1** 创建 `knowledge_base/tech/python_basics.md`（2000~3000 字）：
  - 内容覆盖：GIL、内存管理（引用计数 + 垃圾回收）、装饰器、生成器/迭代器、上下文管理器、异步（asyncio）、多线程 vs 多进程、常见数据类型底层实现（list/dict/set）
- [x] **1.4.2** 创建 `knowledge_base/tech/data_structures.md`（2000~3000 字）：
  - 内容覆盖：数组、链表、栈/队列、哈希表、二叉树/BST/AVL/红黑树、B+树、堆、图（BFS/DFS）、排序算法（快排/归并/堆排）、动态规划、贪心、二分查找
- [x] **1.4.3** 创建 `knowledge_base/tech/system_design.md`（2000~3000 字）：
  - 内容覆盖：负载均衡、缓存策略（Redis）、消息队列（Kafka/RabbitMQ）、数据库选型（SQL vs NoSQL）、分布式系统（CAP/BASE）、微服务、API 设计、限流/熔断
- [x] **1.4.4** 创建 `knowledge_base/tech/machine_learning.md`（1500~2000 字）：
  - 内容覆盖：监督/无监督/强化学习、线性回归/逻辑回归、决策树/随机森林、SVM、神经网络基础、CNN/RNN/Transformer、过拟合/正则化、评价指标
- [x] **1.4.5** 创建 `knowledge_base/tech/recommender_systems.md`（1500~2000 字）：
  - 内容覆盖：协同过滤（UserCF/ItemCF）、矩阵分解、序列推荐（SASRec/BERT4Rec）、多模态推荐、冷启动问题、评估指标（HR/NDCG/MRR）
- [x] **1.4.6** 创建 `knowledge_base/jobs/backend_engineer.json` — 后端开发工程师岗位要求
- [x] **1.4.7** 创建 `knowledge_base/jobs/ml_engineer.json` — 机器学习工程师岗位要求
- [x] **1.4.8** 创建 `knowledge_base/jobs/ai_agent_developer.json` — AI Agent 开发者岗位要求
- [x] **1.4.9** 创建 `knowledge_base/jobs/recsys_engineer.json` — 推荐系统工程师岗位要求
- [x] **1.4.10** 岗位要求 JSON 格式参考：
  ```json
  {
    "role": "后端开发工程师",
    "required_skills": ["Python", "SQL", "HTTP/HTTPS", "Linux", "Git"],
    "preferred_skills": ["Redis", "Docker", "Kubernetes", "消息队列"],
    "interview_topics": [
      {"topic": "语言基础", "weight": 0.2},
      {"topic": "数据库", "weight": 0.2},
      {"topic": "系统设计", "weight": 0.25},
      {"topic": "网络协议", "weight": 0.15},
      {"topic": "并发与性能", "weight": 0.2}
    ],
    "difficulty_distribution": {"easy": 2, "medium": 4, "hard": 2}
  }
  ```

### 1.5 Phase 1 验证

- [x] **1.5.1** ✅ **验证点**：运行 `python -m rag.vector_store --init`，成功构建索引，输出索引大小和文档数量
- [x] **1.5.2** ✅ **验证点**：执行查询 `search("Python GIL 是什么")`，返回相关的知识片段
- [x] **1.5.3** ✅ **验证点**：加载一个 Markdown 简历文件，切分后添加到临时索引，可检索简历内容

---

## Phase 2: 工具函数实现

> 目标：实现 5 个 Tool，每个工具可独立调用并通过测试。

### 2.1 Tool 1: 简历解析 (`tools/resume_parser.py`)

- [x] **2.1.1** 使用 `@tool` 装饰器定义 `parse_resume(file_path: str) -> str`
- [x] **2.1.2** 实现文件加载逻辑（PDF → `PyPDFLoader`，MD → 直接读取）
- [x] **2.1.3** 调用 LLM 提取结构化信息（姓名、教育、技能、项目、经历、竞赛）
- [x] **2.1.4** 将简历文本向量化并存入 session-level FAISS 索引
- [x] **2.1.5** 返回 JSON 字符串格式的结构化简历信息
- [x] **2.1.6** ✅ **验证点**：传入测试简历文件，输出完整的结构化分析

### 2.2 Tool 2: 岗位要求检索 (`tools/job_search.py`)

- [x] **2.2.1** 使用 `@tool` 装饰器定义 `search_job_requirements(role: str) -> str`
- [x] **2.2.2** 从 `knowledge_base/jobs/` 加载岗位 JSON 文件
- [x] **2.2.3** 精确匹配岗位名称 → 返回完整要求
- [x] **2.2.4** 模糊匹配（FAISS 语义检索）→ 返回最相近岗位的要求
- [x] **2.2.5** ✅ **验证点**：`search_job_requirements("后端开发")` 返回正确的岗位要求

### 2.3 Tool 3: 技术知识 RAG 检索 (`tools/knowledge_rag.py`)

- [x] **2.3.1** 使用 `@tool` 装饰器定义 `search_tech_knowledge(query: str) -> str`
- [x] **2.3.2** 调用 `VectorStoreManager.search(query, k=3)` 进行语义检索
- [x] **2.3.3** 格式化返回结果（包含知识片段内容 + 来源标注）
- [x] **2.3.4** ✅ **验证点**：`search_tech_knowledge("B+树和哈希索引的区别")` 返回相关知识片段

### 2.4 Tool 4: 回答多维评估 (`tools/answer_evaluator.py`)

- [x] **2.4.1** 使用 `@tool` 装饰器定义 `evaluate_answer(question: str, answer: str, reference: str = "") -> str`
- [x] **2.4.2** 构建评估 Prompt，包含 rubric（评分标准）
- [x] **2.4.3** 调用 LLM 进行评估，要求输出 JSON 格式：
  ```json
  {
    "completeness": 8,
    "accuracy": 7,
    "depth": 6,
    "expression": 8,
    "overall_score": 7.25,
    "feedback": "...",
    "follow_up_suggestion": "..."
  }
  ```
- [x] **2.4.4** 使用 `JsonOutputParser` 或 `PydanticOutputParser` 确保输出格式正确
- [x] **2.4.5** 异常处理：LLM 返回非 JSON 时进行重试或使用 `OutputFixingParser`
- [x] **2.4.6** ✅ **验证点**：传入一个问题和回答，输出结构化的评分结果

### 2.5 Tool 5: 代码质量分析 (`tools/code_analyzer.py`)

- [x] **2.5.1** 使用 `@tool` 装饰器定义 `evaluate_code(code: str, language: str = "python") -> str`
- [x] **2.5.2** 构建代码分析 Prompt，要求 LLM 评估：正确性、时间/空间复杂度、代码风格、改进建议
- [x] **2.5.3** 返回 JSON 格式的分析结果
- [x] **2.5.4** ✅ **验证点**：传入一段 Python 代码，输出分析结果

### 2.6 Phase 2 集成验证

- [x] **2.6.1** 编写 `tests/test_tools.py`，为每个 Tool 写至少 2 个测试用例
- [x] **2.6.2** ✅ **验证点**：`pytest tests/test_tools.py` 全部通过（22/22）

---

## Phase 3: Agent + LangGraph 状态图

> 目标：实现 4 个 Agent 和 Router 节点，构建完整的 LangGraph 状态图，跑通从开始到报告的完整流程。

### 3.1 状态定义 (`agents/state.py`)

- [x] **3.1.1** 定义 `InterviewState(TypedDict)`，包含以下字段：
  - 输入信息：`messages`, `resume_text`, `job_role`, `candidate_name`
  - Planner 输出：`resume_analysis`, `interview_plan`
  - Interviewer 状态：`current_topic_index`, `current_question`, `follow_up_count`, `max_follow_ups`
  - Evaluator 输出：`current_evaluation`, `all_evaluations`
  - Router 状态：`next_action`
  - Reporter 输出：`final_report`
  - 全局控制：`total_questions_asked`, `max_questions`, `interview_status`
- [x] **3.1.2** 为 `messages` 字段使用 `Annotated[list, add_messages]` 注解（LangGraph 消息累加器）

### 3.2 Prompt 模板 (`prompts/`)

- [x] **3.2.1** 编写 `prompts/planner_prompt.py`：
  - System Prompt：定义 Planner 的角色、目标、输出格式
  - 要求输出结构化的面试大纲 JSON
  - 包含 few-shot 示例
- [x] **3.2.2** 编写 `prompts/interviewer_prompt.py`：
  - System Prompt：定义面试官的提问风格、追问策略
  - 区分「新题目」和「追问」两种场景的 Prompt
  - 要求自然、专业的面试语气
- [x] **3.2.3** 编写 `prompts/evaluator_prompt.py`：
  - System Prompt：定义评估标准（completeness / accuracy / depth / expression）
  - 包含评分 rubric（每个维度 0-10 分的评分参考）
  - 要求输出 JSON 格式的评估结果
- [x] **3.2.4** 编写 `prompts/reporter_prompt.py`：
  - System Prompt：定义报告格式（Markdown）
  - 要求包含：概览、各维度得分、每题回顾、强弱项分析、改进建议、总评级
  - 包含报告模板示例

### 3.3 Planner Agent (`agents/planner.py`)

- [x] **3.3.1** 创建 `planner_node(state: InterviewState) -> dict` 节点函数
- [x] **3.3.2** 实现逻辑：
  1. 绑定 `parse_resume` 和 `search_job_requirements` 工具到 LLM
  2. 将简历文本和目标岗位传入 Planner Prompt
  3. 让 LLM 调用工具分析简历和岗位需求
  4. 解析 LLM 输出，提取 `resume_analysis` 和 `interview_plan`
- [x] **3.3.3** 处理 Tool Call 循环（Agent 可能多次调用工具后才给出最终答案）
- [x] **3.3.4** ✅ **验证点**：单独调用 `planner_node`，传入简历和岗位，输出合理的面试大纲

### 3.4 Interviewer Agent (`agents/interviewer.py`)

- [x] **3.4.1** 创建 `interviewer_node(state: InterviewState) -> dict` 节点函数
- [x] **3.4.2** 实现逻辑：
  1. 绑定 `search_tech_knowledge` 工具到 LLM
  2. 根据 `next_action` 判断是提新题还是追问：
     - `next_question`：从 `interview_plan[current_topic_index]` 获取话题，生成新问题
     - `follow_up`：根据 `current_evaluation.follow_up_suggestion` 生成追问
  3. 更新 `current_question` 和对话历史
  4. 使用 LangGraph 的 `interrupt` 机制暂停等待用户回答
- [x] **3.4.3** 实现 human-in-the-loop 交互点：
  ```python
  from langgraph.types import interrupt
  user_answer = interrupt({"question": question_text})
  ```
- [x] **3.4.4** ✅ **验证点**：单独调用能生成合理的面试问题

### 3.5 Evaluator Agent (`agents/evaluator.py`)

- [x] **3.5.1** 创建 `evaluator_node(state: InterviewState) -> dict` 节点函数
- [x] **3.5.2** 实现逻辑：
  1. 绑定 `evaluate_answer` 和 `evaluate_code` 工具到 LLM
  2. 从 `messages` 提取最新的用户回答
  3. 使用 `search_tech_knowledge` 获取参考答案（可在节点内直接调用，不通过 Tool Calling）
  4. 调用评估工具，获取结构化评分
  5. 将评估结果追加到 `all_evaluations` 列表
  6. 更新 `total_questions_asked` 计数
- [x] **3.5.3** ✅ **验证点**：传入问题和回答，输出包含四维度评分的评估结果

### 3.6 Router 节点 (`agents/router.py`)

- [x] **3.6.1** 创建 `router_node(state: InterviewState) -> dict` 节点函数（纯逻辑，不调用 LLM）
- [x] **3.6.2** 实现路由规则：
  ```
  规则 1: total_questions_asked >= max_questions → "end_interview"
  规则 2: overall_score < 6.0 AND follow_up_count < max_follow_ups → "follow_up"  
  规则 3: current_topic_index + 1 < len(interview_plan) → "next_question"
  规则 4: 所有话题已问完 → "end_interview"
  ```
- [x] **3.6.3** 更新状态：
  - `follow_up` → `follow_up_count += 1`
  - `next_question` → `current_topic_index += 1`, `follow_up_count = 0`
  - `end_interview` → `interview_status = "reporting"`
- [x] **3.6.4** 创建 `route_decision(state) -> str` 函数，返回路由方向（供 `add_conditional_edges` 使用）
- [x] **3.6.5** ✅ **验证点**：不同评分和状态组合下路由方向正确

### 3.7 Reporter Agent (`agents/reporter.py`)

- [x] **3.7.1** 创建 `reporter_node(state: InterviewState) -> dict` 节点函数
- [x] **3.7.2** 实现逻辑：
  1. 汇总 `all_evaluations` 中的所有评估数据
  2. 计算各维度平均分和总平均分
  3. 使用 Reporter Prompt 让 LLM 生成 Markdown 格式的面试报告
  4. 报告内容包含：
     - 📊 面试概览（候选人、岗位、题目数、总时长估算）
     - 📈 分数总结（各维度平均分、总分、等级）
     - 📝 逐题回顾（每道题的表现简评）
     - 💪 强项分析（表现突出的领域）
     - ⚠️ 弱项分析（需要改进的领域）
     - 📚 改进建议（针对弱项给出具体、可操作的学习建议）
     - 🏆 总体评语与等级（A/B/C/D）
  5. 更新 `interview_status = "completed"`
- [x] **3.7.3** ✅ **验证点**：传入模拟的评估数据，生成完整的面试报告

### 3.8 状态图组装 (`agents/graph.py`)

- [x] **3.8.1** 创建 `build_interview_graph()` 函数
- [x] **3.8.2** 注册所有节点：`planner`, `interviewer`, `evaluator`, `router`, `reporter`
- [x] **3.8.3** 设置入口点：`set_entry_point("planner")`
- [x] **3.8.4** 添加固定边：
  - `planner → interviewer`
  - `evaluator → router`
  - `reporter → END`
- [x] **3.8.5** 添加 Interviewer 节点的 interrupt（human-in-the-loop）
- [x] **3.8.6** 添加条件边：
  ```python
  graph.add_conditional_edges(
      "router",
      route_decision,
      {
          "follow_up": "interviewer",
          "next_question": "interviewer",
          "end_interview": "reporter"
      }
  )
  ```
- [x] **3.8.7** 使用 `MemorySaver` 编译图：
  ```python
  memory = MemorySaver()
  compiled_graph = graph.compile(checkpointer=memory, interrupt_before=["interviewer"])
  ```
  > 注意：此处 `interrupt_before=["interviewer"]` 配合 human-in-the-loop 使用。第一次进入 interviewer 前也会中断，需要在 API 层处理好第一次 invoke 和后续 resume 的区别。也可以选择在 interviewer 节点内部使用 `interrupt()` 函数，而非 `interrupt_before`，具体取决于 LangGraph 版本和 API 设计需求。
- [x] **3.8.8** ✅ **验证点（命令行全流程测试）**：
  ```python
  # test_full_flow.py
  graph = build_interview_graph()
  config = {"configurable": {"thread_id": "test-001"}}
  
  # 1. 启动面试
  result = graph.invoke({
      "resume_text": "...(测试简历)...",
      "job_role": "后端开发",
      "candidate_name": "测试用户",
      "max_questions": 3,  # 测试用，减少题数
      "max_follow_ups": 1,
  }, config)
  
  # 2. 获取第一个问题
  print(result["current_question"])
  
  # 3. 模拟用户回答
  result = graph.invoke(Command(resume="我的回答是..."), config)
  
  # 4. 循环直到面试结束
  # ...
  
  # 5. 查看报告
  print(result["final_report"])
  ```
  跑通完整的面试流程，从 Planner 到 Reporter 全部正常工作。

---

## Phase 4: FastAPI 后端

> 目标：将 LangGraph 图封装为 RESTful API，支持通过 HTTP 请求进行面试交互。

### 4.1 数据模型 (`api/schemas.py`)

- [x] **4.1.1** 定义 Pydantic 请求/响应模型：
  - `StartInterviewRequest`：candidate_name, job_role, max_questions, max_follow_ups
  - `StartInterviewResponse`：session_id, message, interview_plan, first_question
  - `AnswerRequest`：answer
  - `AnswerResponse`：evaluation(可选), next_question, is_follow_up, interview_ended, progress
  - `ReportResponse`：report, overall_score, grade, evaluations

### 4.2 API 路由 (`api/routes.py`)

- [x] **4.2.1** 实现 `POST /api/interview/start`：
  1. 接收简历文件（multipart/form-data）和面试参数
  2. 保存简历到 `uploads/` 目录
  3. 创建 session_id (UUID)
  4. 调用 LangGraph 图 invoke，运行 Planner + 第一次 Interviewer
  5. 返回 session_id、面试大纲、第一个问题
- [x] **4.2.2** 实现 `POST /api/interview/{session_id}/answer`：
  1. 接收用户回答
  2. 使用 `Command(resume=answer)` 恢复 LangGraph 图执行
  3. 图运行 Evaluator → Router → (Interviewer 或 Reporter)
  4. 如果面试未结束，返回下一个问题
  5. 如果面试结束，返回结束信号
- [x] **4.2.3** 实现 `GET /api/interview/{session_id}/status`：
  1. 查询当前面试状态（planning/interviewing/completed）
  2. 返回已问题数/总题数的进度
- [x] **4.2.4** 实现 `GET /api/interview/{session_id}/report`：
  1. 检查面试是否已结束
  2. 返回 `final_report` 内容
- [x] **4.2.5** 实现 `GET /api/interview/{session_id}/history`：
  1. 返回完整的对话历史（messages 列表）
- [x] **4.2.6** 实现 `GET /api/interview/{session_id}/report/export`：
  1. 接收查询参数 `format`（`pdf` 或 `md`）
  2. `format=md`：直接返回原始 Markdown 文本，`Content-Type: text/markdown`，带 `Content-Disposition: attachment`
  3. `format=pdf`：使用 Python 库（`weasyprint` 或 `fpdf2`）将 Markdown 转为 PDF 返回
  4. 在 `requirements.txt` 中新增 PDF 生成依赖

### 4.3 FastAPI 主应用 (`main.py`)

- [x] **4.3.1** 创建 FastAPI app 实例
- [x] **4.3.2** 注册路由
- [x] **4.3.3** 配置 CORS（允许前端跨域请求）
- [x] **4.3.4** 在 app 启动时初始化 LangGraph 图和向量知识库
- [x] **4.3.5** 静态文件服务（`/frontend` 目录）
- [x] **4.3.6** ✅ **验证点**：使用 curl 或 Postman 完整走通一次面试流程（启动 → 回答 3 题 → 获取报告）

---

## Phase 5: 前端 UI（LeetCode 风格 · 白灰配色 · HTML+CSS+JS）

> 目标：构建一个类 LeetCode 风格的专业级 Web 面试界面。  
> 技术方案：纯 HTML + CSS + JavaScript，无框架依赖。  
> 设计风格：白灰配色为主（偏白），干净、有质感、专业。  
> 详细设计规范请参考 **ProjectContext.md 第九节**。

### 5.1 页面结构 (`frontend/index.html`)

- [x] **5.1.1** 创建语义化 HTML5 页面骨架：
  - `<header>` — 顶部导航栏（Logo + 项目名 + 进度指示 + 结束面试按钮）
  - `<main>` — 三栏布局容器（左侧面板 + 中央聊天区 + 右侧评估面板）
  - 引入 `style.css`、`app.js`、`markdown-it`（CDN 或本地）
- [x] **5.1.2** 左侧面板 HTML（`#setup-panel`）：
  - 简历上传区：`<div class="upload-zone">` 虚线拖拽区 + `<input type="file">`
  - 岗位选择：`<select id="role-select">`（后端开发 / 推荐系统 / ML 工程师 / AI Agent）
  - 面试参数：题目数量 `<input type="range">`（3~10）、追问次数 `<select>`（1~3）
  - 开始面试按钮：`<button id="start-btn">`
  - 面试中信息区（默认隐藏）：简历文件名、当前岗位、进度、结束按钮
  - **Agent 流水线进度区**（`#agent-pipeline`，默认隐藏，面板底部）：步骤列表 + 总进度条
- [x] **5.1.3** 中央聊天区 HTML（`#chat-area`）：
  - 消息容器：`<div id="messages-container">`（flex-column, overflow-y: auto）
  - 欢迎页（面试前默认显示）：Logo + "上传简历，选择岗位，开始你的 AI 模拟面试"
  - typing indicator：`<div id="typing-indicator">` 三个跳动圆点（默认隐藏）
  - 底部输入栏：`<textarea id="answer-textarea">` + `<button id="submit-btn">`
- [x] **5.1.4** 右侧评估面板 HTML（`#eval-panel`）：
  - 面试大纲列表：`<div id="plan-list">`（话题列表，带状态图标）
  - 实时评估卡片：`<div id="eval-card">`（四维度进度条 + 反馈文字）
- [x] **5.1.5** 报告视图（默认隐藏）：`<div id="report-view">`（全宽布局，统计卡片 + Markdown 渲染区 + 导出按钮组）

### 5.2 样式设计 (`frontend/style.css`)

> 完整的 Design Tokens（CSS 变量、配色、字体、阴影、圆角）参考 ProjectContext.md 9.1 节。

- [x] **5.2.1** CSS Reset + Design Tokens：
  - `:root` 定义所有 CSS 变量（配色、阴影、圆角、字体）
  - 全局 `box-sizing: border-box`、`margin: 0`、`font-family: var(--font-sans)`
  - `body` 背景 `#FFFFFF`
- [x] **5.2.2** 三栏布局样式：
  - Header 高度 `56px`，下边框 `--color-border-primary`，`position: sticky; top: 0`
  - `.layout` — `display: flex; height: calc(100vh - 56px)`
  - 左侧面板 `width: 300px`，右侧面板 `width: 320px`，中央 `flex: 1`
  - 面板背景 `--color-bg-secondary`，面板间 1px 边框分隔
- [x] **5.2.3** 左侧面板样式：
  - `.upload-zone`：虚线边框 `2px dashed --color-border-primary`，hover 变蓝
  - 下拉框、slider：统一样式，圆角 `--radius-sm`
  - 开始按钮：`background: var(--color-accent)`，`color: white`，`border-radius: var(--radius-md)`，hover 加深 + 上浮 1px
- [x] **5.2.3b** ⭐ Agent 流水线进度指示器样式（`#agent-pipeline`）：
  - `.pipeline-step`：`font-size: 12px`，三种状态色（`.done` 绿 / `.active` 蓝 / 默认灰）
  - `.pipeline-progress-bar`：高度 4px，背景 `--color-bg-tertiary`，圆角 2px
  - `.pipeline-progress-fill`：蓝色填充，`transition: width 500ms ease-in-out`
  - 「⏳ 正在...」动态省略号动画：`@keyframes ellipsis`，尾部省略号循环变化
- [x] **5.2.4** 聊天消息样式：
  - `.msg-row`：`display: flex; width: 100%`，控制消息左右对齐
  - `.msg-row.interviewer`：`justify-content: flex-start`（**面试官消息靠左**）
  - `.msg-row.candidate`：`justify-content: flex-end`（**候选人消息靠右**）
  - `.msg-interviewer`：浅灰背景 `--color-bg-secondary`，左上直角圆角 `2px 12px 12px 12px`，左侧蓝色圆形头像（IF）
  - `.msg-candidate`：蓝色背景 `--color-accent`，白色文字，右上直角圆角 `12px 2px 12px 12px`
  - `.msg-system`：居中灰色小字 `12px`
  - 消息出现动画：`opacity 0→1 + translateY(8px→0)`，`300ms ease-out`
- [x] **5.2.5** Typing indicator 样式：
  - 三个灰色圆点，`@keyframes` 逐个上下跳动 `0.6s infinite`
- [x] **5.2.6** 输入区域样式：
  - `textarea`：背景 `--color-bg-tertiary`，聚焦时蓝色边框 + 浅蓝阴影
  - 提交按钮：蓝色实心，与 textarea 等高，右侧紧贴
- [x] **5.2.7** 右侧面板样式：
  - 面试大纲列表：当前话题行左边框 3px 蓝色条 + 微蓝背景
  - 四维度进度条：高度 8px，圆角，颜色按分数分段（绿/橙/红）
  - 评估卡片：白色背景 `--color-bg-primary`，`box-shadow: var(--shadow-sm)`
- [x] **5.2.8** 报告页样式：
  - 顶部统计卡片：`display: flex; gap: 16px`，白色卡片 + 阴影
  - Markdown 渲染区：标题、列表、代码块样式覆盖
  - 逐题回顾：可折叠卡片，点击展开
  - 底部操作按钮：「导出 PDF」「导出 Markdown」「重新开始」三个按钮并排
  - 导出按钮：`--color-accent` outline 样式，hover 填充蓝色
  - 重新开始按钮：灰色 outline 样式，与导出按钮视觉区分
- [x] **5.2.9** 过渡动效：
  - 按钮 hover：`background-color 150ms` + `translateY(-1px)`
  - 进度条更新：`width transition 500ms ease-in-out`
  - 面板切换（聊天→报告）：`opacity 交叉淡出 200ms`

### 5.3 交互逻辑 (`frontend/app.js`)

- [x] **5.3.1** 全局状态管理：
  ```javascript
  const state = {
    sessionId: null,
    messages: [],
    plan: [],
    currentTopicIndex: 0,
    isInterviewing: false,
    isLoading: false,
  };
  ```
- [x] **5.3.2** API 对接函数：
  - `startInterview()` — POST `/api/interview/start`（FormData 含简历文件）
  - `submitAnswer(answer)` — POST `/api/interview/{sessionId}/answer`
  - `getReport()` — GET `/api/interview/{sessionId}/report`
  - `getStatus()` — GET `/api/interview/{sessionId}/status`
  - 所有函数包含 `try/catch` 错误处理 + 用户友好提示
- [x] **5.3.3** DOM 渲染函数：
  - `renderInterviewerMessage(text)` — 创建面试官消息气泡，**包裹在 `.msg-row.interviewer` 中靠左展示**，带头像和时间戳
  - `renderCandidateMessage(text)` — 创建候选人消息气泡，**包裹在 `.msg-row.candidate` 中靠右展示**
  - `renderSystemMessage(text)` — 创建居中系统消息
  - `renderPlan(planList)` — 渲染右侧面试大纲（高亮当前话题）
  - `renderEvaluation(evalData)` — 渲染四维度进度条 + 总分 badge + 反馈
  - `renderReport(reportHtml)` — 切换到报告视图，渲染 Markdown HTML
  - `updateProgress(progressStr)` — 更新 header 进度指示器
  - `renderPipelineProgress(steps)` — 渲染左侧面板 Agent 流水线进度
- [x] **5.3.4** 事件绑定：
  - 开始面试按钮 `click` 事件
  - 提交回答按钮 `click` 事件
  - `Ctrl+Enter` 键盘快捷键提交
  - 文件拖拽上传（`dragover` / `drop` 事件）
  - 结束面试按钮（弹确认框）
  - 导出 PDF 按钮 / 导出 Markdown 按钮 / 重新开始按钮
- [x] **5.3.5** UI 状态切换：
  - `switchToInterviewMode()` — 隐藏欢迎页，显示聊天区，锁定左侧面板，**显示 Agent 流水线进度**
  - `switchToReportView()` — 隐藏聊天 + 右侧面板，显示报告全宽
  - `showTypingIndicator()` / `hideTypingIndicator()` — 控制跳动圆点
  - `scrollToBottom()` — 平滑滚动消息区到底部
  - `updatePipeline(currentStep)` — 更新 Agent 流水线各步骤状态（已完成/进行中/未开始）
- [x] **5.3.6** 工具函数：
  - `formatTime()` — 格式化消息时间戳
  - `getScoreColor(score)` — 根据分数返回颜色（绿/橙/红）
  - `escapeHtml(str)` — 防 XSS 转义
  - `exportMarkdown(reportMd)` — 纯前端 Blob 下载 Markdown 报告
  - `exportPdf(sessionId)` — 调用后端导出 API 下载 PDF 报告

### 5.4 Markdown 渲染

- [x] **5.4.1** 引入 `markdown-it` 库（CDN: `https://cdn.jsdelivr.net/npm/markdown-it/dist/markdown-it.min.js`）
- [x] **5.4.2** 同时下载一份本地拷贝 `frontend/markdown-it.min.js` 作为 CDN 备用
- [x] **5.4.3** 报告页面使用 `md.render()` 将后端返回的 Markdown 报告渲染为 HTML
- [x] **5.4.4** 为渲染后的 HTML 添加样式覆盖（标题、列表、代码块、表格）

### 5.5 报告导出功能

- [x] **5.5.1** Markdown 导出（前端实现）：
  - `exportMarkdown()` — 使用 `Blob` + `URL.createObjectURL` + `<a download>` 导出 `.md` 文件
  - 文件名格式：`BaguRush_Report_{sessionId前8位}.md`
- [x] **5.5.2** PDF 导出（后端实现）：
  - 后端 API `GET /api/interview/{session_id}/report/export?format=pdf`
  - 使用 `weasyprint` 或 `fpdf2` 将 Markdown 转 PDF
  - 前端通过 `window.open()` 或 `fetch` + `Blob` 触发下载
- [x] **5.5.3** 在 `requirements.txt` 中新增 PDF 生成依赖（`weasyprint>=60.0` 或 `fpdf2>=2.7.0`）

### 5.6 Phase 5 验证

- [x] **5.6.1** ✅ **验证点**：`python main.py` 启动后，浏览器访问 `http://localhost:8000/frontend/index.html`，页面正常加载
- [x] **5.6.2** ✅ **验证点**：上传简历 + 选择岗位 → 点击开始 → 左侧面板底部显示 Agent 流水线进度 → 收到第一个面试问题
- [x] **5.6.3** ✅ **验证点**：面试官消息靠左展示（浅灰气泡 + IF 头像），候选人消息靠右展示（蓝色气泡）
- [x] **5.6.4** ✅ **验证点**：输入回答 → 提交 → 右侧面板实时显示评估分数
- [x] **5.6.5** ✅ **验证点**：完整面试结束后 → 自动切换到报告视图 → Markdown 报告正常渲染
- [x] **5.6.6** ✅ **验证点**：报告页点击「导出 PDF」成功下载 PDF 文件，点击「导出 Markdown」成功下载 .md 文件
- [x] **5.6.7** ✅ **验证点**：UI 视觉质感达标（白灰配色、消息气泡样式、进度条动画、typing indicator 动效）

---

## Phase 6: 测试、完善与发布

> 目标：确保项目稳定可用，编写文档，准备 Demo。

### 6.1 测试

- [x] **6.1.1** 编写 `tests/test_tools.py` — 每个 Tool 至少 2 个测试用例
- [x] **6.1.2** 编写 `tests/test_agents.py` — 每个 Agent 节点至少 1 个独立测试
- [x] **6.1.3** 编写 `tests/test_graph.py` — 完整面试流程端到端测试
- [x] **6.1.4** 运行所有测试：`pytest tests/ -v`

### 6.2 Prompt 调优

- [ ] **6.2.1** 使用真实简历（自己的简历）进行 3~5 次完整面试测试
- [ ] **6.2.2** 根据面试质量调整各 Agent 的 System Prompt：
  - Planner 是否能准确识别简历弱点？
  - Interviewer 的问题难度和自然度如何？
  - Evaluator 的评分是否合理公正？
  - Reporter 的报告是否有价值？
- [ ] **6.2.3** 调整 Router 的路由阈值（评分低于多少才追问）

### 6.3 README 编写

- [x] **6.3.1** 编写 `README.md`，包含：
  - 项目介绍和功能展示（GIF 或截图）
  - 架构图（LangGraph 状态图）
  - 技术栈说明
  - 快速启动指南
  - API 文档概要
  - 项目结构
  - 未来计划 (Roadmap)
- [x] **6.3.2** 添加架构图（Mermaid 或手绘图）
- [ ] **6.3.3** 添加 Demo 截图或 GIF

### 6.4 代码质量

- [ ] **6.4.1** 所有函数添加 docstring
- [ ] **6.4.2** 类型标注完整
- [ ] **6.4.3** 错误处理覆盖主要异常场景（API 超时、JSON 解析失败、文件不存在）
- [ ] **6.4.4** 日志记录关键节点的执行信息

### 6.5 发布

- [ ] **6.5.1** 初始化 Git 仓库，推送到 GitHub
- [ ] **6.5.2** 确保 `.env` 不在 Git 中
- [ ] **6.5.3** 可选：部署到 VPS（`154.193.217.104`）或其他平台
- [ ] **6.5.4** 更新简历中 BaguRush 项目的描述和链接

---

## 🔗 关键依赖关系

```
Phase 0 (初始化)
    │
    ▼
Phase 1 (RAG 知识库)  ←── Phase 2 (工具函数) 中的 Tool 3/4/5 需要
    │                       Phase 1 的向量库
    ▼
Phase 2 (工具函数)
    │
    ▼
Phase 3 (Agent + 状态图)  ←── 依赖 Phase 2 的所有工具
    │
    ▼
Phase 4 (FastAPI 后端)  ←── 封装 Phase 3 的状态图
    │
    ▼
Phase 5 (前端 UI)  ←── 对接 Phase 4 的 API
    │
    ▼
Phase 6 (测试与完善)
```

---

## ⚠️ 开发注意事项

1. **LLM API Key 管理**：所有 API Key 通过 `.env` 文件管理，**绝对不要硬编码在代码中**
2. **Token 成本控制**：开发测试时使用较小的 `max_questions`（如 3 题），避免消耗过多 Token
3. **嵌入模型选择**：如果没有 OpenAI Embedding API，使用 `sentence-transformers/all-MiniLM-L6-v2` 本地模型（免费且效果不错）
4. **FAISS 索引持久化**：首次构建索引后保存到磁盘，避免每次启动重建
5. **LangGraph 版本兼容**：`interrupt` 和 `Command` API 在不同版本间有差异，以实际安装版本的文档为准
6. **JSON 输出稳定性**：LLM 输出 JSON 时可能格式不稳定，务必使用 Output Parser 和异常处理
7. **编码**：所有 Prompt 和知识库内容使用 UTF-8 编码，支持中文
8. **并发**：FastAPI 天然支持异步，但 LangGraph 图的 invoke 是同步的。对于长时间运行的面试，考虑使用 `asyncio.to_thread` 或后台任务

---

## 📎 参考资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangChain Tool Calling](https://python.langchain.com/docs/concepts/tool_calling/)
- [FAISS 文档](https://github.com/facebookresearch/faiss)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
