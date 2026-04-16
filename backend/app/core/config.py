from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """项目的统一配置对象。

    所有环境变量都会在这里收口，业务代码通过 `get_settings()`
    获取配置，而不是直接到处读环境变量。
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Medical Checkup Agent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(default="sqlite:///../data/sqlite/medical_agent.db", alias="DATABASE_URL")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")
    database_pool_timeout_seconds: int = Field(default=30, alias="DATABASE_POOL_TIMEOUT_SECONDS")
    database_pool_recycle_seconds: int = Field(default=1800, alias="DATABASE_POOL_RECYCLE_SECONDS")
    report_queue_poll_interval_seconds: float = Field(default=1.0, alias="REPORT_QUEUE_POLL_INTERVAL_SECONDS")
    report_queue_lease_seconds: int = Field(default=300, alias="REPORT_QUEUE_LEASE_SECONDS")
    report_queue_run_embedded_worker: bool = Field(default=True, alias="REPORT_QUEUE_RUN_EMBEDDED_WORKER")
    upload_dir: str = Field(default="../data/uploads", alias="UPLOAD_DIR")
    output_dir: str = Field(default="../data/output", alias="OUTPUT_DIR")
    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_BASE_URL",
    )
    qwen_chat_model: str = Field(default="qwen3.5-plus", alias="QWEN_CHAT_MODEL")
    qwen_fast_model: str = Field(default="qwen3.5-flash", alias="QWEN_FAST_MODEL")
    qwen_max_model: str = Field(default="qwen3-max", alias="QWEN_MAX_MODEL")
    qwen_vl_model: str = Field(default="qwen3-vl-flash", alias="QWEN_VL_MODEL")
    short_term_context_turns: int = Field(default=4, alias="SHORT_TERM_CONTEXT_TURNS")
    agent_response_cache_ttl_seconds: int = Field(default=1800, alias="AGENT_RESPONSE_CACHE_TTL_SECONDS")
    who_client_id: str = Field(default="", alias="WHO_CLIENT_ID")
    who_client_secret: str = Field(default="", alias="WHO_CLIENT_SECRET")

    @property
    def is_sqlite(self) -> bool:
        """判断当前数据库是否仍然使用 SQLite。"""
        return self.database_url.startswith("sqlite:///") or self.database_url.startswith("sqlite+pysqlite:///")

    @property
    def resolved_database_url(self) -> str:
        """返回真正用于建连的数据库 URL。"""
        if not self.is_sqlite:
            return self.database_url
        prefix = "sqlite+pysqlite:///" if self.database_url.startswith("sqlite+pysqlite:///") else "sqlite:///"
        raw_path = self.database_url.removeprefix(prefix)
        resolved_path = (BASE_DIR / raw_path).resolve()
        return f"{prefix}{resolved_path.as_posix()}"

    @property
    def upload_path(self) -> Path:
        """上传文件最终落盘的目录。"""
        return (BASE_DIR / self.upload_dir).resolve()

    @property
    def output_path(self) -> Path:
        """导出文件目录，当前主要用于保存 PDF。"""
        return (BASE_DIR / self.output_dir).resolve()

    @property
    def sqlite_path(self) -> Path | None:
        """把 sqlite URL 解析成磁盘上的实际路径。"""
        if not self.is_sqlite:
            return None
        if self.resolved_database_url.startswith("sqlite+pysqlite:///"):
            return Path(self.resolved_database_url.removeprefix("sqlite+pysqlite:///"))
        return Path(self.resolved_database_url.removeprefix("sqlite:///"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取并缓存配置，同时确保运行所需目录存在。"""
    settings = Settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    (settings.output_path / "pdf").mkdir(parents=True, exist_ok=True)
    if settings.sqlite_path is not None:
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
