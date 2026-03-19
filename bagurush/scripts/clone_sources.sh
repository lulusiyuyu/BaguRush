#!/bin/bash
# clone_sources.sh — 拉取所有数据源仓库到 data_sources/ 目录
# 用法: bash scripts/clone_sources.sh
# 注意: 使用 --depth 1 浅克隆节省空间和时间

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data_sources"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "========================================="
echo "  BaguRush 数据源拉取脚本"
echo "  目标目录: $DATA_DIR"
echo "========================================="

# ---------- 必选数据源 ----------
echo ""
echo ">>> [必选] 拉取核心数据源..."

# 1. Byron4j/CookBook — Java/JVM/中间件/算法
if [ ! -d "CookBook" ]; then
    echo "[1/6] Cloning CookBook..."
    git clone --depth 1 https://github.com/Byron4j/CookBook.git
else
    echo "[1/6] CookBook 已存在，跳过"
fi

# 2. adongwanai/AgentGuide — Agent/RAG/LangGraph/面试题库
if [ ! -d "AgentGuide" ]; then
    echo "[2/6] Cloning AgentGuide..."
    git clone --depth 1 https://github.com/adongwanai/AgentGuide.git
else
    echo "[2/6] AgentGuide 已存在，跳过"
fi

# 3. CyC2018/CS-Notes — OS/CN/系统设计/DB/LeetCode
if [ ! -d "CS-Notes" ]; then
    echo "[3/6] Cloning CS-Notes..."
    git clone --depth 1 https://github.com/CyC2018/CS-Notes.git
else
    echo "[3/6] CS-Notes 已存在，跳过"
fi

# 4. workattech/core-cs-os-networks-dbms — OS/CN/DBMS 短问短答
if [ ! -d "core-cs-os-networks-dbms" ]; then
    echo "[4/6] Cloning core-cs-os-networks-dbms..."
    git clone --depth 1 https://github.com/workattech/core-cs-os-networks-dbms.git
else
    echo "[4/6] core-cs-os-networks-dbms 已存在，跳过"
fi

# 5. alirezadir/machine-learning-interviews — ML/AI 面试
if [ ! -d "machine-learning-interviews" ]; then
    echo "[5/6] Cloning machine-learning-interviews..."
    git clone --depth 1 https://github.com/alirezadir/machine-learning-interviews.git
else
    echo "[5/6] machine-learning-interviews 已存在，跳过"
fi

# 6. llmgenai/LLMInterviewQuestions — LLM 面试题
if [ ! -d "LLMInterviewQuestions" ]; then
    echo "[6/6] Cloning LLMInterviewQuestions..."
    git clone --depth 1 https://github.com/llmgenai/LLMInterviewQuestions.git
else
    echo "[6/6] LLMInterviewQuestions 已存在，跳过"
fi

# ---------- 推荐数据源 ----------
echo ""
echo ">>> [推荐] 拉取推荐数据源..."

# 7. anxkhn/LastMinuteNotes
if [ ! -d "LastMinuteNotes" ]; then
    echo "[7/10] Cloning LastMinuteNotes..."
    git clone --depth 1 https://github.com/anxkhn/LastMinuteNotes.git
else
    echo "[7/10] LastMinuteNotes 已存在，跳过"
fi

# 8. KalyanKS-NLP/RAG-Interview-Questions-and-Answers-Hub
if [ ! -d "RAG-Interview-Questions-and-Answers-Hub" ]; then
    echo "[8/10] Cloning RAG-Interview-Questions-and-Answers-Hub..."
    git clone --depth 1 https://github.com/KalyanKS-NLP/RAG-Interview-Questions-and-Answers-Hub.git
else
    echo "[8/10] RAG-Interview-Questions-and-Answers-Hub 已存在，跳过"
fi

# 9. zixian2021/AI-interview-cards
if [ ! -d "AI-interview-cards" ]; then
    echo "[9/10] Cloning AI-interview-cards..."
    git clone --depth 1 https://github.com/zixian2021/AI-interview-cards.git
else
    echo "[9/10] AI-interview-cards 已存在，跳过"
fi

# 10. WeThinkIn/AIGC-Interview-Book
if [ ! -d "AIGC-Interview-Book" ]; then
    echo "[10/10] Cloning AIGC-Interview-Book..."
    git clone --depth 1 https://github.com/WeThinkIn/AIGC-Interview-Book.git
else
    echo "[10/10] AIGC-Interview-Book 已存在，跳过"
fi

echo ""
echo "========================================="
echo "  全部数据源拉取完成！"
echo "  目录: $DATA_DIR"
echo "========================================="
echo ""
ls -la "$DATA_DIR"
