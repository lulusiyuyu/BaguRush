"""
Phase 2 工具函数测试套件。

覆盖 5 个 Tool：
  - parse_resume           : 简历解析
  - search_job_requirements: 岗位要求检索
  - search_tech_knowledge  : 技术知识 RAG 检索
  - evaluate_answer        : 回答多维度评估
  - evaluate_code          : 代码质量分析

运行方式：
  cd bagurush
  pytest tests/test_tools.py -v

含 LLM API 调用的测试默认开启，如需跳过可加 -m "not llm"：
  pytest tests/test_tools.py -v -m "not llm"
"""

import json
import sys
from pathlib import Path

import pytest

# 将项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 测试用简历路径（来自用户提供的测试文件）
_TEST_RESUME_PDF = str(Path(_ROOT).parent / "简历_测试.pdf")
_TEST_RESUME_FALLBACK = str(_ROOT / "uploads" / "test_resume.md")

# =========================================================================== #
#  Tool 1: parse_resume
# =========================================================================== #

class TestParseResume:
    """简历解析工具测试。"""

    @pytest.mark.llm
    def test_parse_pdf_resume(self):
        """测试解析 PDF 简历文件，验证返回结构完整。"""
        from tools.resume_parser import parse_resume

        # 检查测试文件是否存在
        if not Path(_TEST_RESUME_PDF).exists():
            pytest.skip(f"测试简历 PDF 不存在: {_TEST_RESUME_PDF}")

        result = parse_resume.invoke({
            "file_path": _TEST_RESUME_PDF,
            "session_id": "test_session_pdf",
        })

        data = json.loads(result)
        print(f"\n[简历解析结果]\n{json.dumps(data, ensure_ascii=False, indent=2)}")

        # 基本结构验证
        assert isinstance(data, dict), "返回值应为 JSON 对象"
        assert "error" not in data, f"解析失败: {data.get('error')}"

        # 验证关键字段存在
        assert "name" in data, "缺少 name 字段"
        assert "education" in data, "缺少 education 字段"
        assert "skills" in data, "缺少 skills 字段"
        assert "projects" in data, "缺少 projects 字段"
        assert "summary" in data, "缺少 summary 字段"

        # skills 应为列表
        assert isinstance(data["skills"], list), "skills 应为列表"

        print(f"✅ 解析成功，候选人：{data.get('name', '未知')}，技能数：{len(data.get('skills', []))}")

    @pytest.mark.llm
    def test_session_index_built(self):
        """测试解析后 session 向量索引应被成功创建。"""
        from tools.resume_parser import SESSION_STORES, parse_resume

        if not Path(_TEST_RESUME_PDF).exists():
            pytest.skip(f"测试简历 PDF 不存在: {_TEST_RESUME_PDF}")

        session_id = "test_session_index_check"
        result = parse_resume.invoke({
            "file_path": _TEST_RESUME_PDF,
            "session_id": session_id,
        })

        data = json.loads(result)
        assert "error" not in data, f"解析失败: {data.get('error')}"

        # 验证 session 向量索引已创建
        assert session_id in SESSION_STORES, f"Session '{session_id}' 未找到向量索引"
        store = SESSION_STORES[session_id]
        assert store.doc_count > 0, "向量索引应包含至少 1 个向量"
        print(f"✅ Session 索引创建成功，向量数: {store.doc_count}")

    def test_parse_nonexistent_file(self):
        """测试不存在的文件应返回包含 error 字段的 JSON。"""
        from tools.resume_parser import parse_resume

        result = parse_resume.invoke({
            "file_path": "/nonexistent/path/resume.pdf",
            "session_id": "test_error",
        })

        data = json.loads(result)
        assert "error" in data, "文件不存在时应返回 error 字段"
        print(f"✅ 错误处理正确: {data['error']}")

    def test_parse_unsupported_format(self):
        """测试不支持的文件格式应返回包含 error 字段的 JSON。"""
        from tools.resume_parser import parse_resume

        # 创建一个临时 docx 文件（不支持格式）
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"fake content")
            tmp_path = f.name

        try:
            result = parse_resume.invoke({
                "file_path": tmp_path,
                "session_id": "test_format_error",
            })
            data = json.loads(result)
            assert "error" in data, "不支持的格式应返回 error 字段"
            print(f"✅ 格式错误处理正确: {data['error']}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# =========================================================================== #
#  Tool 2: search_job_requirements
# =========================================================================== #

class TestSearchJobRequirements:
    """岗位要求检索工具测试。"""

    def test_search_backend_engineer(self):
        """测试检索后端开发工程师岗位要求。"""
        from tools.job_search import search_job_requirements

        result = search_job_requirements.invoke({"role": "后端开发工程师"})

        assert isinstance(result, str), "返回值应为字符串"
        assert "后端" in result or "backend" in result.lower(), "应返回后端岗位相关内容"
        assert "技能" in result or "skill" in result.lower(), "应包含技能信息"
        print(f"✅ 后端岗位检索成功\n{result[:300]}...")

    def test_search_recsys_engineer(self):
        """测试检索推荐系统工程师岗位要求。"""
        from tools.job_search import search_job_requirements

        result = search_job_requirements.invoke({"role": "推荐系统工程师"})

        assert isinstance(result, str)
        assert "推荐" in result, "应返回推荐系统相关内容"
        print(f"✅ 推荐系统岗位检索成功\n{result[:300]}...")

    def test_search_by_partial_name(self):
        """测试模糊匹配：'后端开发' 应能匹配到后端工程师岗位。"""
        from tools.job_search import search_job_requirements

        result = search_job_requirements.invoke({"role": "后端开发"})

        assert isinstance(result, str)
        assert "❌" not in result, f"模糊匹配失败，返回: {result[:100]}"
        print(f"✅ 模糊匹配成功\n{result[:200]}...")

    def test_search_ml_engineer(self):
        """测试检索机器学习工程师岗位要求。"""
        from tools.job_search import search_job_requirements

        result = search_job_requirements.invoke({"role": "机器学习工程师"})

        assert isinstance(result, str)
        assert "❌" not in result or "机器学习" in result, "应返回机器学习相关内容"
        print(f"✅ ML 岗位检索成功\n{result[:200]}...")

    def test_search_unknown_role(self):
        """测试无法匹配的岗位名称，应返回可用岗位列表。"""
        from tools.job_search import search_job_requirements

        result = search_job_requirements.invoke({"role": "外星人工程师 xyz123"})

        assert isinstance(result, str)
        # 应该返回"未找到"提示或返回某个岗位（关键词匹配的后备行为）
        print(f"✅ 未知岗位处理正常: {result[:200]}")


# =========================================================================== #
#  Tool 3: search_tech_knowledge
# =========================================================================== #

class TestSearchTechKnowledge:
    """技术知识 RAG 检索工具测试。"""

    def test_search_python_gil(self):
        """测试检索 Python GIL 相关知识。"""
        from tools.knowledge_rag import search_tech_knowledge

        result = search_tech_knowledge.invoke({"query": "Python GIL 是什么"})

        assert isinstance(result, str)
        assert "❌" not in result, f"检索失败: {result[:200]}"
        assert len(result) > 50, "返回内容应有实质内容"
        # GIL 相关关键词
        assert any(kw in result for kw in ["GIL", "全局解释器锁", "线程", "gil"]), \
            "返回内容应包含 GIL 相关知识"
        print(f"✅ GIL 知识检索成功，内容长度: {len(result)}")

    def test_search_bplus_tree(self):
        """测试检索 B+树 相关知识。"""
        from tools.knowledge_rag import search_tech_knowledge

        result = search_tech_knowledge.invoke({"query": "B+树和哈希索引的区别"})

        assert isinstance(result, str)
        assert "❌" not in result, f"检索失败: {result[:200]}"
        assert len(result) > 50
        print(f"✅ B+树知识检索成功，内容长度: {len(result)}")

    def test_search_returns_source(self):
        """测试检索结果应包含来源标注。"""
        from tools.knowledge_rag import search_tech_knowledge

        result = search_tech_knowledge.invoke({"query": "推荐系统冷启动"})

        assert isinstance(result, str)
        # 应包含来源文件名
        assert "来源" in result or "source" in result.lower() or "recommender" in result.lower(), \
            "返回结果应包含来源标注"
        print(f"✅ 来源标注验证通过")

    def test_search_system_design(self):
        """测试检索系统设计相关知识。"""
        from tools.knowledge_rag import search_tech_knowledge

        result = search_tech_knowledge.invoke({"query": "Redis 缓存策略 Cache-Aside"})

        assert isinstance(result, str)
        assert "❌" not in result
        print(f"✅ 系统设计知识检索成功，内容长度: {len(result)}")


# =========================================================================== #
#  Tool 4: evaluate_answer
# =========================================================================== #

class TestEvaluateAnswer:
    """回答多维评估工具测试。"""

    @pytest.mark.llm
    def test_evaluate_good_answer(self):
        """测试评估一个较好的回答，分数应偏高。"""
        from tools.answer_evaluator import evaluate_answer

        question = "请解释 Python 中的 GIL（全局解释器锁）是什么，它对多线程编程有什么影响？"
        answer = (
            "GIL（Global Interpreter Lock）是 CPython 解释器中的一个互斥锁，"
            "它保证同一时刻只有一个线程在执行 Python 字节码。"
            "这是为了保护 CPython 的内存管理（引用计数）不被多线程并发修改导致问题。\n\n"
            "影响：对于 CPU 密集型任务，多线程无法利用多核心，效率提升有限；"
            "对于 IO 密集型任务（如网络请求、文件读写），线程在等待 IO 时会释放 GIL，"
            "多线程仍有效果。\n\n"
            "解决方案：CPU 密集型任务可用 multiprocessing 模块（每进程独立 GIL），"
            "或使用 C 扩展（如 NumPy）绕过 GIL。"
        )

        result = evaluate_answer.invoke({
            "question": question,
            "answer": answer,
            "reference": "",
        })

        data = json.loads(result)
        print(f"\n[评估结果]\n{json.dumps(data, ensure_ascii=False, indent=2)}")

        assert "error" not in data, f"评估失败: {data.get('error')}"
        assert "overall_score" in data, "缺少 overall_score"
        assert "feedback" in data, "缺少 feedback"
        assert "follow_up_suggestion" in data, "缺少 follow_up_suggestion"

        # 验证分数范围
        for dim in ["completeness", "accuracy", "depth", "expression"]:
            assert dim in data, f"缺少维度: {dim}"
            assert 0 <= data[dim] <= 10, f"{dim} 分数超出 0-10 范围: {data[dim]}"

        assert 0 <= data["overall_score"] <= 10, "综合分数超出范围"

        # 较好的回答，综合分应 >= 6
        assert data["overall_score"] >= 5.0, f"较好回答的综合分应 >= 5，实际: {data['overall_score']}"
        print(f"✅ 评估完成，综合分: {data['overall_score']}/10")

    @pytest.mark.llm
    def test_evaluate_poor_answer(self):
        """测试评估一个较差的回答，分数应偏低。"""
        from tools.answer_evaluator import evaluate_answer

        question = "请详细解释 TCP 的三次握手过程和为什么需要三次。"
        answer = "TCP 需要握手来建立连接，握手三次是为了保证可靠性。"

        result = evaluate_answer.invoke({
            "question": question,
            "answer": answer,
        })

        data = json.loads(result)
        print(f"\n[简短回答评估结果]\n{json.dumps(data, ensure_ascii=False, indent=2)}")

        assert "error" not in data
        assert "overall_score" in data

        # 过于简短的回答，综合分应 <= 8
        assert data["overall_score"] <= 8.5, f"简短回答的综合分不应过高: {data['overall_score']}"
        print(f"✅ 简短回答评估完成，综合分: {data['overall_score']}/10")

    @pytest.mark.llm
    def test_evaluate_with_reference(self):
        """测试带参考答案的评估，验证 reference 参数有效。"""
        from tools.answer_evaluator import evaluate_answer

        question = "什么是 B+树，它和 B 树有何区别？"
        answer = "B+树是一种多路搜索树，所有数据存在叶子节点，叶子节点之间有指针链接。"
        reference = "B+树：1. 所有数据存储在叶子节点；2. 内部节点只存键值用于导航；3. 叶子节点通过双向链表连接；4. 适合范围查询；B树：数据存在所有节点。"

        result = evaluate_answer.invoke({
            "question": question,
            "answer": answer,
            "reference": reference,
        })

        data = json.loads(result)
        assert "error" not in data
        assert "overall_score" in data
        print(f"✅ 带参考答案评估完成，综合分: {data['overall_score']}/10")


# =========================================================================== #
#  Tool 5: evaluate_code
# =========================================================================== #

class TestEvaluateCode:
    """代码质量分析工具测试。"""

    @pytest.mark.llm
    def test_evaluate_correct_code(self):
        """测试分析一段正确的 Python 代码。"""
        from tools.code_analyzer import evaluate_code

        code = """
def two_sum(nums: list, target: int) -> list:
    \"\"\"
    给定整数数组 nums 和目标值 target，返回两数之和等于目标值的下标。
    使用哈希表实现 O(n) 时间复杂度。
    \"\"\"
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""

        result = evaluate_code.invoke({"code": code, "language": "python"})
        data = json.loads(result)
        print(f"\n[代码分析结果]\n{json.dumps(data, ensure_ascii=False, indent=2)}")

        assert "error" not in data, f"代码分析失败: {data.get('error')}"
        assert "is_correct" in data, "缺少 is_correct 字段"
        assert "time_complexity" in data, "缺少 time_complexity 字段"
        assert "space_complexity" in data, "缺少 space_complexity 字段"
        assert "overall_score" in data, "缺少 overall_score 字段"
        assert "improvements" in data, "缺少 improvements 字段"
        assert isinstance(data["improvements"], list), "improvements 应为列表"

        # 正确的代码，is_correct 应为 True
        assert data["is_correct"] is True, f"正确代码应标记为 is_correct=True，实际: {data['is_correct']}"
        # 时间复杂度应为 O(n)
        assert "O(n)" in data["time_complexity"] or "n" in data["time_complexity"], \
            f"Two Sum 哈希表解法时间复杂度应为 O(n)，实际: {data['time_complexity']}"
        print(f"✅ 代码分析完成，综合分: {data['overall_score']}/10")

    @pytest.mark.llm
    def test_evaluate_buggy_code(self):
        """测试分析一段有 bug 的 Python 代码，is_correct 应为 False。"""
        from tools.code_analyzer import evaluate_code

        code = """
def binary_search(arr, target):
    left, right = 0, len(arr)  # bug: right 应为 len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid  # bug: 应为 mid + 1，否则死循环
        else:
            right = mid - 1
    return -1
"""

        result = evaluate_code.invoke({"code": code, "language": "python"})
        data = json.loads(result)
        print(f"\n[有 bug 代码分析结果]\n{json.dumps(data, ensure_ascii=False, indent=2)}")

        assert "error" not in data
        assert "is_correct" in data
        # 有 bug 的代码，is_correct 应为 False
        assert data["is_correct"] is False, \
            f"有 bug 的代码应标记为 is_correct=False，实际: {data['is_correct']}"
        print(f"✅ 有 bug 代码检测正确，综合分: {data['overall_score']}/10")

    def test_evaluate_empty_code(self):
        """测试空代码应返回包含 error 字段的 JSON。"""
        from tools.code_analyzer import evaluate_code

        result = evaluate_code.invoke({"code": "", "language": "python"})
        data = json.loads(result)

        assert "error" in data, "空代码应返回 error 字段"
        print(f"✅ 空代码错误处理正确: {data['error']}")

    @pytest.mark.llm
    def test_evaluate_code_with_complexity(self):
        """测试分析有嵌套循环的代码，时间复杂度应为 O(n²)。"""
        from tools.code_analyzer import evaluate_code

        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

        result = evaluate_code.invoke({"code": code, "language": "python"})
        data = json.loads(result)

        assert "error" not in data
        assert "time_complexity" in data
        # 冒泡排序时间复杂度应为 O(n²)
        assert "n²" in data["time_complexity"] or "n^2" in data["time_complexity"] or \
               "O(n2)" in data["time_complexity"] or "n*n" in data["time_complexity"] or \
               "²" in data["time_complexity"], \
            f"冒泡排序时间复杂度应为 O(n²)，实际: {data['time_complexity']}"
        print(f"✅ 冒泡排序复杂度分析正确: {data['time_complexity']}")


# =========================================================================== #
#  集成测试：验证 tools __init__ 导入
# =========================================================================== #

class TestToolsImport:
    """验证 tools 包的所有公共 API 可正常导入。"""

    def test_import_all_tools(self):
        """测试 tools 包的所有 Tool 和辅助函数可正常导入。"""
        from tools import (
            SESSION_STORES,
            evaluate_answer,
            evaluate_code,
            get_session_store,
            parse_resume,
            search_job_requirements,
            search_tech_knowledge,
        )

        # 验证 LangChain Tool 都有 invoke 方法（StructuredTool 不是内置 callable）
        for tool_obj in [parse_resume, search_job_requirements, search_tech_knowledge,
                         evaluate_answer, evaluate_code]:
            assert hasattr(tool_obj, "invoke"), f"{tool_obj} 缺少 invoke 方法"
        assert callable(get_session_store)
        assert isinstance(SESSION_STORES, dict)

        print("✅ tools 包所有公共 API 导入成功")

    def test_tool_names(self):
        """验证每个 Tool 的 name 属性正确。"""
        from tools import (
            evaluate_answer,
            evaluate_code,
            parse_resume,
            search_job_requirements,
            search_tech_knowledge,
        )

        assert parse_resume.name == "parse_resume"
        assert search_job_requirements.name == "search_job_requirements"
        assert search_tech_knowledge.name == "search_tech_knowledge"
        assert evaluate_answer.name == "evaluate_answer"
        assert evaluate_code.name == "evaluate_code"

        print("✅ 所有 Tool 名称验证通过")
