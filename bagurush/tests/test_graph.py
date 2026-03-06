"""
LangGraph 图 + FastAPI 端到端集成测试。

- 图编译/结构测试（静态）
- FastAPI 端点冒烟测试（使用 TestClient，不依赖 LLM）
- 完整面试流程测试（@pytest.mark.llm）
"""

import io
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================ #
#  图结构静态测试
# ============================================================ #
class TestGraphStructure:

    def test_graph_compiles(self):
        from agents.graph import build_interview_graph
        g = build_interview_graph()
        nodes = list(g.nodes)
        assert "planner" in nodes
        assert "interviewer" in nodes
        assert "evaluator" in nodes
        assert "router" in nodes
        assert "reporter" in nodes

    def test_graph_has_five_nodes(self):
        from agents.graph import build_interview_graph
        g = build_interview_graph()
        # __start__ 和 __end__ 也算节点
        real_nodes = [n for n in g.nodes if not n.startswith("__")]
        assert len(real_nodes) == 5

    def test_get_graph_singleton(self):
        """get_graph 应返回可复用的图实例。"""
        import agents.graph as gmod
        # Reset singleton for test isolation
        gmod._graph = None
        g1 = gmod.get_graph()
        g2 = gmod.get_graph()
        assert g1 is g2
        gmod._graph = None  # cleanup

    def test_state_has_all_fields(self):
        from agents.state import InterviewState
        expected = [
            "messages", "resume_text", "job_role", "candidate_name",
            "resume_file_path", "session_id",
            "resume_analysis", "interview_plan",
            "current_topic_index", "current_question", "follow_up_count", "max_follow_ups",
            "current_evaluation", "all_evaluations",
            "next_action", "final_report",
            "total_questions_asked", "max_questions", "interview_status",
        ]
        annotations = InterviewState.__annotations__
        for key in expected:
            assert key in annotations, f"缺少字段: {key}"


# ============================================================ #
#  FastAPI 端点冒烟测试（不需要 LLM）
# ============================================================ #
class TestAPIEndpoints:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "BaguRush"

    def test_root_redirect(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 307
        assert "/frontend/index.html" in r.headers.get("location", "")

    def test_frontend_index(self, client):
        r = client.get("/frontend/index.html")
        assert r.status_code == 200
        assert "BaguRush" in r.text

    def test_frontend_css(self, client):
        r = client.get("/frontend/style.css")
        assert r.status_code == 200
        assert "--color-accent" in r.text

    def test_frontend_js(self, client):
        r = client.get("/frontend/app.js")
        assert r.status_code == 200
        assert "startInterview" in r.text

    def test_frontend_markdown_it(self, client):
        r = client.get("/frontend/markdown-it.min.js")
        assert r.status_code == 200
        assert len(r.text) > 10000

    def test_status_404_no_session(self, client):
        r = client.get("/api/interview/nonexistent123/status")
        assert r.status_code == 404

    def test_report_404_no_session(self, client):
        r = client.get("/api/interview/nonexistent123/report")
        assert r.status_code == 404

    def test_history_404_no_session(self, client):
        r = client.get("/api/interview/nonexistent123/history")
        assert r.status_code == 404

    def test_answer_404_no_session(self, client):
        r = client.post(
            "/api/interview/nonexistent123/answer",
            json={"answer": "test"},
        )
        assert r.status_code == 404

    def test_start_requires_file(self, client):
        """没有上传文件应返回 422。"""
        r = client.post("/api/interview/start")
        assert r.status_code == 422


# ============================================================ #
#  完整 API 面试流程（需要 LLM）
# ============================================================ #
class TestFullAPIFlow:

    RESUME_PATH = Path("/home/lsy/project_set/playground/BaguRush/简历_测试.pdf")

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    @pytest.mark.llm
    def test_full_api_interview(self, client):
        """通过 HTTP API 完整走完一轮面试（3 题）。"""
        import os
        if not self.RESUME_PATH.exists():
            pytest.skip(f"测试简历不存在: {self.RESUME_PATH}")
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("未设置 DEEPSEEK_API_KEY")

        # --- 1. 启动面试 ---
        with open(self.RESUME_PATH, "rb") as f:
            r = client.post(
                "/api/interview/start",
                files={"resume": ("resume.pdf", f, "application/pdf")},
                data={
                    "candidate_name": "测试用户",
                    "job_role": "后端开发工程师",
                    "max_questions": "3",
                    "max_follow_ups": "1",
                },
            )
        assert r.status_code == 200, f"启动失败: {r.text}"
        start_data = r.json()
        session_id = start_data["session_id"]
        assert session_id
        assert start_data.get("first_question")
        print(f"\n[Test] Session: {session_id}")
        print(f"[Test] 第一个问题: {start_data['first_question'][:80]}")

        # --- 2. 查询状态 ---
        r = client.get(f"/api/interview/{session_id}/status")
        assert r.status_code == 200
        status_data = r.json()
        assert status_data["status"] == "interviewing"

        # --- 3. 回答问题循环 ---
        mock_answers = [
            "Python 的 GIL 是全局解释器锁，保证同一时刻只有一个线程执行 Python 字节码。"
            "对 CPU 密集型任务不利，可用 multiprocessing 绕过。",
            "在项目中使用 FastAPI + Redis + PostgreSQL 架构，"
            "Redis 做缓存和分布式锁，PostgreSQL 做持久化。QPS 约 5000。",
            "分布式场景下用消息队列（如 Kafka）解耦服务，"
            "数据一致性通过最终一致性方案解决，辅以补偿事务。",
        ]

        interview_ended = False
        for i, answer in enumerate(mock_answers):
            if interview_ended:
                break
            r = client.post(
                f"/api/interview/{session_id}/answer",
                json={"answer": answer},
            )
            assert r.status_code == 200, f"回答 {i+1} 失败: {r.text}"
            ans_data = r.json()
            print(f"[Test] 回答 {i+1} | 进度: {ans_data.get('progress')} | 结束: {ans_data.get('interview_ended')}")

            if ans_data.get("interview_ended"):
                interview_ended = True
            elif ans_data.get("next_question"):
                print(f"[Test] 下一题: {ans_data['next_question'][:80]}")

        # --- 4. 获取报告 ---
        r = client.get(f"/api/interview/{session_id}/report")
        assert r.status_code == 200
        report_data = r.json()
        assert report_data.get("report")
        assert report_data.get("overall_score", 0) > 0
        assert report_data.get("grade") in ("A", "B", "C", "D")
        print(f"[Test] 报告评级: {report_data['grade']} | 综合分: {report_data['overall_score']:.2f}")

        # --- 5. 获取历史 ---
        r = client.get(f"/api/interview/{session_id}/history")
        assert r.status_code == 200
        history_data = r.json()
        assert len(history_data["messages"]) > 0
        print(f"[Test] 历史消息数: {len(history_data['messages'])}")
