"""
FastAPI 路由定义。

端点：
  POST /api/interview/start              启动面试
  POST /api/interview/{session_id}/answer 提交回答
  POST /api/interview/{session_id}/end    手动结束面试并生成报告
  GET  /api/interview/{session_id}/status 查询状态
  GET  /api/interview/{session_id}/report 获取报告
  GET  /api/interview/{session_id}/history 对话历史
"""

import os
import sys
import uuid
import json
import asyncio
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

# 确保项目根在 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agents.graph import build_interview_graph
from utils.llm_config import set_runtime_config, clear_runtime_config, set_stream_callbacks, clear_stream_callbacks
from utils.llm_events import get_store, remove_store, LLMStreamHandler
from api.schemas import (
    AnswerRequest,
    AnswerResponse,
    ErrorResponse,
    HistoryResponse,
    MessageItem,
    ReportResponse,
    StartInterviewResponse,
    StatusResponse,
)

router = APIRouter(prefix="/api/interview", tags=["interview"])

# ------------------------------------------------------------------ #
#  每个会话独立的 graph 实例（MemorySaver 不能跨 graph 共享 thread）
# ------------------------------------------------------------------ #
_graphs: Dict[str, Any] = {}


def _get_or_create_graph(session_id: str):
    """获取或创建某个 session 专属的 graph 实例。"""
    if session_id not in _graphs:
        _graphs[session_id] = build_interview_graph()
    return _graphs[session_id]


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _apply_llm_config(request: Request):
    """从请求头读取前端传入的 LLM 配置并设置到运行时。"""
    api_key = request.headers.get("x-llm-api-key")
    base_url = request.headers.get("x-llm-base-url")
    model = request.headers.get("x-llm-model")
    if api_key:
        set_runtime_config(api_key=api_key, base_url=base_url or None, model=model or None)
    else:
        clear_runtime_config()


def _setup_stream(request: Request):
    """如果请求带 X-Stream-Id 头，建立流式回调并返回 (store, handler)。"""
    stream_id = request.headers.get("x-stream-id", "").strip()
    if not stream_id:
        return None, None
    store = get_store(stream_id)
    handler = LLMStreamHandler(store)
    set_stream_callbacks([handler])
    return store, handler


def _teardown_stream(store):
    """请求结束时清理流式回调。"""
    clear_stream_callbacks()
    if store:
        store.close()


# ------------------------------------------------------------------ #
#  辅助函数
# ------------------------------------------------------------------ #

def _extract_question_from_interrupts(graph, config: dict) -> Optional[str]:
    """从 graph 的 pending interrupts 中提取面试官问题文本。"""
    state = graph.get_state(config)
    if state.tasks:
        for task in state.tasks:
            if task.interrupts:
                val = task.interrupts[0].value
                if isinstance(val, dict):
                    return val.get("question", str(val))
                return str(val)
    return None


def _get_state_values(graph, config: dict) -> dict:
    """安全获取 graph 当前状态值。"""
    try:
        s = graph.get_state(config)
        return s.values if s else {}
    except Exception:
        return {}


def _compute_grade(score: float) -> str:
    if score >= 8.5:
        return "A"
    elif score >= 7.0:
        return "B"
    elif score >= 5.5:
        return "C"
    return "D"


# ------------------------------------------------------------------ #
#  POST /api/interview/start
# ------------------------------------------------------------------ #

@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(
    request: Request,
    resume: UploadFile = File(...),
    candidate_name: str = Form("候选人"),
    job_role: str = Form("后端开发工程师"),
    max_questions: int = Form(8),
    max_follow_ups: int = Form(1),
):
    """
    上传简历、选择岗位，启动面试。
    返回 session_id、面试大纲和第一个问题。
    """
    _apply_llm_config(request)
    stream_store, _ = _setup_stream(request)

    # 1. 保存简历文件
    session_id = uuid.uuid4().hex[:12]
    upload_dir = _ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)

    suffix = Path(resume.filename).suffix if resume.filename else ".pdf"
    saved_path = upload_dir / f"{session_id}{suffix}"
    content = await resume.read()
    saved_path.write_bytes(content)
    print(f"[API] 简历已保存: {saved_path} ({len(content)} bytes)")

    # 2. 构建初始状态
    initial_state = {
        "resume_file_path": str(saved_path),
        "job_role": job_role,
        "candidate_name": candidate_name,
        "session_id": session_id,
        "max_questions": max_questions,
        "max_follow_ups": max_follow_ups,
        "interview_status": "planning",
        "messages": [],
    }

    graph = _get_or_create_graph(session_id)
    config = _config(session_id)

    try:
        # 3. 第一次 invoke → Planner 执行，在 interviewer 前暂停
        result = await asyncio.to_thread(graph.invoke, initial_state, config)

        # 4. 继续执行让 interviewer 提问并 interrupt
        result = await asyncio.to_thread(graph.invoke, None, config)

        # 5. 提取第一个问题
        first_question = _extract_question_from_interrupts(graph, config)

        # 6. 获取完整状态
        vals = _get_state_values(graph, config)
        plan = vals.get("interview_plan", [])

        return StartInterviewResponse(
            session_id=session_id,
            message=f"面试规划完成，共 {len(plan)} 个话题。",
            interview_plan=plan,
            first_question=first_question or "请简要介绍一下你自己。",
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"启动面试失败: {e}")
    finally:
        _teardown_stream(stream_store)


# ------------------------------------------------------------------ #
#  POST /api/interview/{session_id}/answer
# ------------------------------------------------------------------ #

@router.post("/{session_id}/answer", response_model=AnswerResponse)
async def submit_answer(session_id: str, body: AnswerRequest, request: Request):
    """
    提交候选人回答。
    图恢复执行 → Evaluator → Router → (Interviewer 或 Reporter)。
    """
    if session_id not in _graphs:
        raise HTTPException(status_code=404, detail="会话不存在")

    _apply_llm_config(request)
    stream_store, _ = _setup_stream(request)

    graph = _graphs[session_id]
    config = _config(session_id)

    try:
        # 使用 Command(resume=answer) 恢复图执行
        result = await asyncio.to_thread(
            graph.invoke, Command(resume=body.answer), config
        )

        vals = _get_state_values(graph, config)
        interview_status = vals.get("interview_status", "")
        total_asked = vals.get("total_questions_asked", 0)
        max_q = vals.get("max_questions", 8)
        current_eval = vals.get("current_evaluation")
        next_action = vals.get("next_action", "")
        interview_plan = vals.get("interview_plan", [])
        topic_idx = vals.get("current_topic_index", 0)
        current_topic = interview_plan[topic_idx]["topic"] if topic_idx < len(interview_plan) else ""

        # 面试已结束 → reporter 已生成报告
        if interview_status == "completed":
            return AnswerResponse(
                evaluation=current_eval,
                next_question=None,
                is_follow_up=False,
                interview_ended=True,
                progress=f"{total_asked}/{max_q}",
                topic=current_topic,
            )

        # 面试未结束 → 图在 interviewer 前暂停，需让 interviewer 提问
        # 先检查是否已经有 interrupt（interviewer 内部 interrupt）
        next_question = _extract_question_from_interrupts(graph, config)

        if not next_question:
            # 可能图停在 interrupt_before，继续执行
            await asyncio.to_thread(graph.invoke, None, config)
            next_question = _extract_question_from_interrupts(graph, config)

        # 更新 vals（可能有变化）
        vals = _get_state_values(graph, config)
        total_asked = vals.get("total_questions_asked", total_asked)

        return AnswerResponse(
            evaluation=current_eval,
            next_question=next_question,
            is_follow_up=(next_action == "follow_up"),
            interview_ended=False,
            progress=f"{total_asked}/{max_q}",
            topic=current_topic,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理回答失败: {e}")
    finally:
        _teardown_stream(stream_store)


# ------------------------------------------------------------------ #
#  GET /api/interview/{session_id}/status
# ------------------------------------------------------------------ #

@router.get("/{session_id}/status", response_model=StatusResponse)
async def get_status(session_id: str):
    """查询面试状态和进度。"""
    if session_id not in _graphs:
        raise HTTPException(status_code=404, detail="会话不存在")

    graph = _graphs[session_id]
    config = _config(session_id)
    vals = _get_state_values(graph, config)

    total = vals.get("total_questions_asked", 0)
    max_q = vals.get("max_questions", 8)
    plan = vals.get("interview_plan", [])
    idx = vals.get("current_topic_index", 0)
    ct = plan[idx]["topic"] if idx < len(plan) else None

    return StatusResponse(
        session_id=session_id,
        status=vals.get("interview_status", "unknown"),
        total_questions_asked=total,
        max_questions=max_q,
        progress=f"{total}/{max_q}",
        current_topic=ct,
    )


# ------------------------------------------------------------------ #
#  POST /api/interview/{session_id}/end — 手动结束面试
# ------------------------------------------------------------------ #

@router.post("/{session_id}/end", response_model=ReportResponse)
async def end_interview(session_id: str, request: Request):
    """
    手动结束面试并生成报告。
    即使面试还没走完所有题目，也会基于已有的评估数据生成报告。
    """
    if session_id not in _graphs:
        raise HTTPException(status_code=404, detail="会话不存在")

    graph = _graphs[session_id]
    config = _config(session_id)
    vals = _get_state_values(graph, config)

    # 如果已经完成了，直接返回报告
    if vals.get("interview_status") == "completed" and vals.get("final_report"):
        evals = vals.get("all_evaluations", [])
        scores = [e.get("overall_score", 0) for e in evals if isinstance(e.get("overall_score"), (int, float))]
        avg = sum(scores) / len(scores) if scores else 0.0
        return ReportResponse(
            report=vals.get("final_report", ""),
            overall_score=round(avg, 2),
            grade=_compute_grade(avg),
            evaluations=evals,
        )

    _apply_llm_config(request)
    stream_store, _ = _setup_stream(request)

    try:
        # 直接调用 reporter_node 生成报告
        from agents.reporter import reporter_node

        report_result = await asyncio.to_thread(reporter_node, vals)

        # 手动更新 graph 状态
        graph.update_state(config, report_result)

        # 从更新后的状态获取报告
        updated_vals = _get_state_values(graph, config)
        final_report = updated_vals.get("final_report") or report_result.get("final_report", "")
        evals = updated_vals.get("all_evaluations") or vals.get("all_evaluations", [])
        scores = [e.get("overall_score", 0) for e in evals if isinstance(e.get("overall_score"), (int, float))]
        avg = sum(scores) / len(scores) if scores else 0.0

        print(f"[API] 面试已手动结束，报告已生成 | session={session_id}")

        return ReportResponse(
            report=final_report,
            overall_score=round(avg, 2),
            grade=_compute_grade(avg),
            evaluations=evals,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成报告失败: {e}")
    finally:
        _teardown_stream(stream_store)


# ------------------------------------------------------------------ #
#  GET /api/interview/{session_id}/report
# ------------------------------------------------------------------ #

@router.get("/{session_id}/report", response_model=ReportResponse)
async def get_report(session_id: str):
    """获取面试报告。"""
    if session_id not in _graphs:
        raise HTTPException(status_code=404, detail="会话不存在")

    graph = _graphs[session_id]
    config = _config(session_id)
    vals = _get_state_values(graph, config)

    if vals.get("interview_status") != "completed":
        raise HTTPException(status_code=400, detail="面试尚未完成")

    evals = vals.get("all_evaluations", [])
    scores = [e.get("overall_score", 0) for e in evals if isinstance(e.get("overall_score"), (int, float))]
    avg = sum(scores) / len(scores) if scores else 0.0

    return ReportResponse(
        report=vals.get("final_report", ""),
        overall_score=round(avg, 2),
        grade=_compute_grade(avg),
        evaluations=evals,
    )


# ------------------------------------------------------------------ #
#  GET /api/interview/{session_id}/report/export
# ------------------------------------------------------------------ #

@router.get("/{session_id}/report/export")
async def export_report(session_id: str):
    """导出面试报告为 Markdown 文件。"""
    if session_id not in _graphs:
        raise HTTPException(status_code=404, detail="会话不存在")

    graph = _graphs[session_id]
    config = _config(session_id)
    vals = _get_state_values(graph, config)

    if vals.get("interview_status") != "completed":
        raise HTTPException(status_code=400, detail="面试尚未完成")

    report_md = vals.get("final_report", "")
    filename_base = f"BaguRush_Report_{session_id[:8]}"

    return Response(
        content=report_md.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename_base}.md"},
    )


# ------------------------------------------------------------------ #
#  GET /api/interview/stream/{stream_id}  — SSE LLM 数据流
# ------------------------------------------------------------------ #

@router.get("/stream/{stream_id}")
async def llm_stream(stream_id: str):
    """Server-Sent Events 端点，实时推送 LLM 调用/token/tool 事件。"""
    store = get_store(stream_id)

    async def event_generator():
        while not store.closed:
            events = store.drain()
            for evt in events:
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            if not events:
                await asyncio.sleep(0.1)
        # 排空剩余事件
        for evt in store.drain():
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        remove_store(stream_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ #
#  GET /api/interview/{session_id}/history
# ------------------------------------------------------------------ #

@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """获取对话历史。"""
    if session_id not in _graphs:
        raise HTTPException(status_code=404, detail="会话不存在")

    graph = _graphs[session_id]
    config = _config(session_id)
    vals = _get_state_values(graph, config)
    raw_messages = vals.get("messages", [])

    items = []
    for msg in raw_messages:
        if isinstance(msg, AIMessage):
            role = "interviewer"
        elif isinstance(msg, HumanMessage):
            role = "candidate"
        else:
            role = "system"
        items.append(MessageItem(
            role=role,
            content=msg.content if hasattr(msg, "content") else str(msg),
            name=getattr(msg, "name", None),
        ))

    return HistoryResponse(session_id=session_id, messages=items)
