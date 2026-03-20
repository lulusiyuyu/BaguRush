#!/usr/bin/env bash
# BaguRush 关闭脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/server.pid"

# 1. 先杀 PID 文件记录的进程
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo "已停止 PID 文件记录的进程（PID: $PID）"
    fi
    rm -f "$PID_FILE"
fi

# 2. 再杀所有残留的 uvicorn main:app 进程（包括手动启动的）
REMAINING=$(pgrep -f "uvicorn main:app" 2>/dev/null)
if [ -n "$REMAINING" ]; then
    echo "清理残留进程: $REMAINING"
    pkill -f "uvicorn main:app" 2>/dev/null
    sleep 1
    # 如果还没死，强制杀
    STILL=$(pgrep -f "uvicorn main:app" 2>/dev/null)
    if [ -n "$STILL" ]; then
        echo "强制终止: $STILL"
        pkill -9 -f "uvicorn main:app" 2>/dev/null
    fi
fi

echo "BaguRush 已停止"
