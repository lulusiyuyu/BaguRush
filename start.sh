#!/usr/bin/env bash
# BaguRush 启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/bagurush"
LOG_FILE="$SCRIPT_DIR/server.log"
PID_FILE="$SCRIPT_DIR/server.pid"

# 检查是否已在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "BaguRush 已在运行（PID: $PID），请先执行 ./stop.sh"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# 检查 .env 文件
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "未找到 $PROJECT_DIR/.env，请先复制并填写 API Key："
    echo "  cp $PROJECT_DIR/.env.example $PROJECT_DIR/.env"
    exit 1
fi

# 激活 conda 环境
CONDA_ENV="/mnt/d/ForWSL/env/bagurush"
if [ -d "$CONDA_ENV" ]; then
    source /home/lsy/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    echo "已激活 conda 环境：$CONDA_ENV"
else
    echo "⚠️  未找到 conda 环境：$CONDA_ENV，使用系统 Python"
fi

echo "正在启动 BaguRush..."
cd "$PROJECT_DIR"

nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "启动成功！PID: $(cat $PID_FILE)"
echo "访问地址：http://localhost:8000"
echo "日志文件：$LOG_FILE"
echo "停止服务：./stop.sh"
