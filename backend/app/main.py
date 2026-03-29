from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import engine, init_db
from app.services.knowledge_service import knowledge_service


# 这是整个后端应用的入口文件。
# 它负责：
# 1. 创建 FastAPI 应用
# 2. 配置跨域
# 3. 注册 API 路由
# 4. 在启动时初始化数据库和本地知识库
settings = get_settings()

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    """应用启动时执行一次的初始化逻辑。"""
    init_db()
    with Session(engine) as session:
        knowledge_service.ensure_initialized(session)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    """最简单的健康检查接口，用来确认服务本身是否正常启动。"""
    return {"status": "ok"}
