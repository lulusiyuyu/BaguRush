"""
BaguRush — FastAPI 应用入口。

启动方式：
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

或直接运行：
  python main.py
"""

import os
import sys
from pathlib import Path

# 确保项目根在 sys.path（main.py 就在 bagurush/ 下）
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

# ------------------------------------------------------------------ #
#  创建 FastAPI 应用
# ------------------------------------------------------------------ #

app = FastAPI(
    title="BaguRush",
    description="AI 模拟面试多 Agent 系统",
    version="1.0.0",
)

# ------------------------------------------------------------------ #
#  CORS
# ------------------------------------------------------------------ #

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  注册 API 路由
# ------------------------------------------------------------------ #

from api.routes import router as interview_router  # noqa: E402

app.include_router(interview_router)

# ------------------------------------------------------------------ #
#  静态文件服务（前端）
# ------------------------------------------------------------------ #

frontend_dir = _ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


# ------------------------------------------------------------------ #
#  根路由 → 重定向到前端
# ------------------------------------------------------------------ #

from fastapi.responses import RedirectResponse  # noqa: E402


@app.get("/")
async def root():
    return RedirectResponse(url="/frontend/index.html")


# ------------------------------------------------------------------ #
#  健康检查
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {"status": "ok", "service": "BaguRush"}


# ------------------------------------------------------------------ #
#  启动
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 BaguRush 启动中 → http://{host}:{port}")
    print(f"📂 前端页面 → http://{host}:{port}/frontend/index.html")
    uvicorn.run("main:app", host=host, port=port, reload=True)
