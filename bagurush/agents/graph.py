"""
BaguRush LangGraph 状态图定义。

图结构：
  START
    ↓
  planner          （制定面试大纲）
    ↓
  interviewer      （提问 + interrupt 等待回答）
    ↓
  evaluator        （评估候选人回答）
    ↓
  router           （纯逻辑路由）
    ↓
  ┌─────────────────────────────────────┐
  │  follow_up  → interviewer（追问）    │
  │  next_question → interviewer（换题） │
  │  end_interview → reporter           │
  └─────────────────────────────────────┘
    ↓
  reporter         （生成面试报告）
    ↓
  END

关键配置：
  - checkpointer = MemorySaver()  支持多轮对话恢复
  - interrupt_before = ["interviewer"]  每次提问前暂停，等待候选人输入

使用方式：
  graph = build_interview_graph()

  config = {"configurable": {"thread_id": "session-001"}}

  # 第一次调用（启动面试）
  result = graph.invoke({
      "resume_file_path": "简历.pdf",
      "job_role": "后端开发",
      "max_questions": 5,
      "session_id": "session-001",
  }, config)

  # 之后循环：面试官发出问题后图暂停，外部收到问题并获取候选人回答
  from langgraph.types import Command
  result = graph.invoke(Command(resume="候选人的回答"), config)
"""

import sys
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.evaluator import evaluator_node
from agents.interviewer import interviewer_node
from agents.planner import planner_node
from agents.reporter import reporter_node
from agents.router import route_decision, router_node
from agents.state import InterviewState


def build_interview_graph():
    """
    构建并编译 BaguRush 状态图。

    Returns:
        CompiledStateGraph: 带 MemorySaver checkpointer 的可执行图
    """
    builder = StateGraph(InterviewState)

    # 注册节点
    builder.add_node("planner", planner_node)
    builder.add_node("interviewer", interviewer_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("router", router_node)
    builder.add_node("reporter", reporter_node)

    # 固定边
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "interviewer")
    builder.add_edge("interviewer", "evaluator")
    builder.add_edge("evaluator", "router")
    builder.add_edge("reporter", END)

    # 条件边：router 根据 next_action 决定下一节点
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "follow_up": "interviewer",
            "next_question": "interviewer",
            "end_interview": "reporter",
        },
    )

    # 编译：使用 MemorySaver 支持多轮持久化，在 interviewer 前暂停等待候选人输入
    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["interviewer"],
    )

    return graph


# 模块级单例（延迟初始化）
_graph = None


def get_graph():
    """获取全局单例图（线程不安全，生产应用应为每个会话创建独立实例）。"""
    global _graph
    if _graph is None:
        _graph = build_interview_graph()
    return _graph


if __name__ == "__main__":
    # 快速验证图结构
    g = build_interview_graph()
    print("✅ 图编译成功")
    print(f"节点: {list(g.nodes)}")
    try:
        # 尝试生成 Mermaid 图
        print("\n图结构 (Mermaid):")
        print(g.get_graph().draw_mermaid())
    except Exception as e:
        print(f"Mermaid 图生成失败: {e}")
