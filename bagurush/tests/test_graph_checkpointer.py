"""graph.py checkpointer 选择逻辑测试。"""

import importlib
import os
from unittest.mock import patch


def test_get_checkpointer_sqlite_default():
    """默认应返回 SqliteSaver 实例（使用临时文件）。"""
    import tempfile, sqlite3
    from agents.graph import _get_checkpointer

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name

    try:
        with patch.dict(os.environ, {"SQLITE_DB_PATH": f"sqlite:///{tmp_path}"}):
            cp = _get_checkpointer()
        from langgraph.checkpoint.sqlite import SqliteSaver
        assert isinstance(cp, SqliteSaver)
    finally:
        os.unlink(tmp_path)


def test_get_checkpointer_fallback():
    """当 SqliteSaver 导入失败时应回退到 MemorySaver。"""
    from agents.graph import _get_checkpointer
    from langgraph.checkpoint.memory import MemorySaver

    with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": None}):
        cp = _get_checkpointer()
    assert isinstance(cp, MemorySaver)
