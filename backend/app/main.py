from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import engine, init_db
from app.services.knowledge_service import knowledge_service


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
    init_db()
    with Session(engine) as session:
        knowledge_service.ensure_initialized(session)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
