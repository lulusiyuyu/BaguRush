#!/usr/bin/env bash
# BaguRush 关闭脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "未找到 PID 文件，尝试按进程名查找..."
    pkill -f "uvicorn main:app" 2>/dev/null && echo "服务已停止" || echo "未发现运行中的服务"
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    rm -f "$PID_FILE"
    echo "BaguRush 已停止（PID: $PID）"
else
    echo "进程 $PID 已不存在，清理 PID 文件"
    rm -f "$PID_FILE"
fi
