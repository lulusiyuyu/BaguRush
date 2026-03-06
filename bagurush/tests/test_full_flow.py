"""
BaguRush 全流程集成测试。

用法：
  # 标准模拟面试（自动回答）
  python tests/test_full_flow.py

  # 通过 pytest（标记为 llm，默认跳过）
  pytest tests/test_full_flow.py -m llm -v

功能验证：
  1. 图编译和节点注册
  2. Planner 制定面试大纲（需要 LLM）
  3. Interviewer 生成问题 + interrupt
  4. Evaluator 评估回答
  5. Router 路由决策
  6. Reporter 生成报告
  7. 完整状态流转检验
"""

import os
import sys
import time
import uuid
from pathlib import Path

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# ------------------------------------------------------------------ #
#  静态测试（不需要 LLM）
# ------------------------------------------------------------------ #


def test_graph_compiles():
    """图能被正确编译。"""
    from agents.graph import build_interview_graph

    g = build_interview_graph()
    nodes = list(g.nodes)
    assert "planner" in nodes
    assert "interviewer" in nodes
    assert "evaluator" in nodes
    assert "router" in nodes
    assert "reporter" in nodes


def test_state_typeddict():
    """InterviewState 字段完整性检查。"""
    from agents.state import InterviewState

    required_keys = [
        "messages",
        "resume_file_path",
        "job_role",
        "session_id",
        "resume_analysis",
        "interview_plan",
        "current_topic_index",
        "current_question",
        "follow_up_count",
        "max_follow_ups",
        "current_evaluation",
        "all_evaluations",
        "next_action",
        "final_report",
        "total_questions_asked",
        "max_questions",
        "interview_status",
    ]
    annotations = InterviewState.__annotations__
    for key in required_keys:
        assert key in annotations, f"InterviewState 缺少字段: {key}"


def test_router_rules():
    """Router 逻辑验证（无 LLM 调用）。"""
    from agents.router import router_node

    # 规则 1：达到最大题数
    state = {
        "total_questions_asked": 5, "max_questions": 5,
        "follow_up_count": 0, "max_follow_ups": 1,
        "current_topic_index": 0, "interview_plan": [{"topic": "A"}],
        "current_evaluation": {"overall_score": 9.0},
    }
    result = router_node(state)
    assert result["next_action"] == "end_interview"

    # 规则 2：低分触发追问
    state2 = {
        "total_questions_asked": 1, "max_questions": 8,
        "follow_up_count": 0, "max_follow_ups": 1,
        "current_topic_index": 0, "interview_plan": [{"topic": "A"}, {"topic": "B"}],
        "current_evaluation": {"overall_score": 4.5},
    }
    result2 = router_node(state2)
    assert result2["next_action"] == "follow_up"
    assert result2["follow_up_count"] == 1

    # 规则 3：高分换题
    state3 = {
        "total_questions_asked": 1, "max_questions": 8,
        "follow_up_count": 0, "max_follow_ups": 1,
        "current_topic_index": 0, "interview_plan": [{"topic": "A"}, {"topic": "B"}],
        "current_evaluation": {"overall_score": 8.0},
    }
    result3 = router_node(state3)
    assert result3["next_action"] == "next_question"
    assert result3["current_topic_index"] == 1

    # 规则 4：所有话题完成
    state4 = {
        "total_questions_asked": 2, "max_questions": 8,
        "follow_up_count": 1, "max_follow_ups": 1,
        "current_topic_index": 0, "interview_plan": [{"topic": "A"}],
        "current_evaluation": {"overall_score": 8.0},
    }
    result4 = router_node(state4)
    assert result4["next_action"] == "end_interview"


def test_reporter_fallback():
    """Reporter 兜底报告生成（无 LLM）。"""
    from agents.reporter import _generate_fallback_report

    report = _generate_fallback_report(
        candidate_name="张三",
        job_role="后端开发",
        avgs={"completeness": 7.5, "accuracy": 8.0, "depth": 6.5, "expression": 7.0, "overall": 7.25},
        evaluations=[
            {"topic": "Python基础", "question": "解释 GIL", "answer": "GIL 是...", "overall_score": 7.5, "feedback": "回答较准确"},
        ],
    )
    assert "张三" in report
    assert "后端开发" in report
    assert "7.25" in report or "7.3" in report  # 浮点显示


# ------------------------------------------------------------------ #
#  LLM 集成测试（需要 DeepSeek API，标记为 llm）
# ------------------------------------------------------------------ #

RESUME_PATH = "/home/lsy/project_set/playground/BaguRush/简历_测试.pdf"


@pytest.mark.llm
def test_full_interview_flow():
    """
    完整面试流程测试（2 题自动回答）。

    测试步骤：
    1. 启动图 → Planner 制定大纲
    2. Interviewer 提问 → interrupt
    3. 候选人回答（自动模拟）→ Evaluator 评估
    4. Router 路由
    5. 第 2 题 → interrupt → 回答
    6. Reporter 生成报告
    """
    from agents.graph import build_interview_graph

    # 跳过条件
    if not os.path.exists(RESUME_PATH):
        pytest.skip(f"测试简历不存在: {RESUME_PATH}")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("未设置 DEEPSEEK_API_KEY")

    session_id = f"test-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}

    graph = build_interview_graph()

    # ---- 启动面试 ---- #
    print("\n" + "=" * 60)
    print("🚀 启动面试流程")
    print("=" * 60)

    initial_state = {
        "resume_file_path": RESUME_PATH,
        "job_role": "Python 后端开发工程师",
        "candidate_name": "张三",
        "session_id": session_id,
        "max_questions": 2,       # 仅测 2 题，加速测试
        "max_follow_ups": 1,
        "interview_status": "planning",
        "messages": [],
    }

    # 第一次 invoke：Planner 执行 → interviewer 前暂停
    result1 = graph.invoke(initial_state, config)

    # 验证 Planner 输出
    assert result1.get("interview_plan"), "❌ interview_plan 不应为空"
    assert result1.get("interview_status") == "interviewing", "❌ 状态应为 interviewing"
    print(f"✅ Planner 完成，大纲话题数: {len(result1['interview_plan'])}")

    # 此时图在 interviewer 前暂停（interrupt_before=["interviewer"]）
    # 图的 pending interrupts 为空（还没进入 interviewer 所以没有 interrupt values）
    # 继续执行让 interviewer 产生问题
    from langgraph.types import Command

    # ---- 面试轮次循环 ---- #
    mock_answers = [
        "Python 的 GIL 是全局解释器锁，它保证了在同一时刻只有一个线程在执行 Python 字节码，这是 CPython 的实现细节，可以通过多进程或异步IO来绕过。",
        "在我参与的项目中，我负责设计了一个高并发的 API 服务，使用了 FastAPI + Redis + PostgreSQL 的技术栈，QPS 达到了 5000 以上。",
    ]

    question_count = 0
    max_rounds = 5  # 防止死循环

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # 恢复图执行（或传入答案）
        if round_num == 0:
            # 第一次：不传答案，让 interviewer 先提问
            result = graph.invoke(None, config)
        else:
            # 后续：传入候选人回答
            answer = mock_answers[min(question_count - 1, len(mock_answers) - 1)]
            print(f"💬 候选人回答: {answer[:60]}...")
            result = graph.invoke(Command(resume=answer), config)

        # 检查是否有 pending interrupts（interviewer 刚提了问题）
        state = graph.get_state(config)

        if state.next and "interviewer" in state.next:
            # 图在 interviewer 前暂停，需要让 interviewer 提问
            result = graph.invoke(None, config)
            state = graph.get_state(config)

        # 检查 interrupt 值（面试官的问题）
        interrupts = state.tasks[0].interrupts if state.tasks else []
        if interrupts:
            question_data = interrupts[0].value
            question_text = question_data.get("question", "") if isinstance(question_data, dict) else str(question_data)
            question_count += 1
            print(f"❓ 面试官问题 [{question_count}]: {question_text[:80]}...")
        elif result.get("interview_status") == "completed":
            print("✅ 面试已完成，跳出循环")
            break
        elif result.get("final_report"):
            print("✅ 报告已生成，跳出循环")
            break

        # 达到预定题数后提交最后一个答案并结束
        if question_count >= initial_state["max_questions"]:
            answer = mock_answers[min(question_count - 1, len(mock_answers) - 1)]
            print(f"💬 最后回答: {answer[:60]}...")
            result = graph.invoke(Command(resume=answer), config)
            break

    # ---- 验证输出 ---- #
    final_result = graph.get_state(config).values

    assert final_result.get("interview_status") == "completed", \
        f"❌ 最终状态应为 completed，实际: {final_result.get('interview_status')}"
    assert final_result.get("final_report"), "❌ final_report 不应为空"
    assert final_result.get("all_evaluations"), "❌ all_evaluations 不应为空"

    print("\n" + "=" * 60)
    print("🎉 全流程测试通过！")
    print(f"  话题数: {len(final_result.get('interview_plan', []))}")
    print(f"  实际题数: {final_result.get('total_questions_asked', 0)}")
    print(f"  评估记录数: {len(final_result.get('all_evaluations', []))}")
    print(f"  报告长度: {len(final_result.get('final_report', ''))} 字符")
    print("=" * 60)
    print("\n📄 面试报告（前 500 字）:")
    print(final_result["final_report"][:500])


# ------------------------------------------------------------------ #
#  独立运行入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("BaguRush 全流程测试（独立运行模式）")
    print("=" * 60)

    # 静态测试
    print("\n[1/4] 测试图编译...")
    test_graph_compiles()
    print("✅ 通过")

    print("\n[2/4] 测试 InterviewState 字段...")
    test_state_typeddict()
    print("✅ 通过")

    print("\n[3/4] 测试 Router 逻辑...")
    test_router_rules()
    print("✅ 通过")

    print("\n[4/4] 测试 Reporter 兜底...")
    test_reporter_fallback()
    print("✅ 通过")

    print("\n" + "=" * 60)
    print("✅ 所有静态测试通过！")
    print("📌 要运行 LLM 集成测试：pytest tests/test_full_flow.py -m llm -v -s")
    print("=" * 60)

    # 如果命令行参数包含 --llm，运行 LLM 测试
    if "--llm" in sys.argv:
        print("\n🚀 运行 LLM 集成测试...")
        test_full_interview_flow()
