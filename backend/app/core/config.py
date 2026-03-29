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
    who_client_id: str = Field(default="", alias="WHO_CLIENT_ID")
    who_client_secret: str = Field(default="", alias="WHO_CLIENT_SECRET")

    @property
    def upload_path(self) -> Path:
        """上传文件最终落盘的目录。"""
        return (BASE_DIR / self.upload_dir).resolve()

    @property
    def output_path(self) -> Path:
        """导出文件目录，当前主要用于保存 PDF。"""
        return (BASE_DIR / self.output_dir).resolve()

    @property
    def sqlite_path(self) -> Path:
        """把 sqlite URL 解析成磁盘上的实际路径。"""
        if self.database_url.startswith("sqlite:///"):
            return (BASE_DIR / self.database_url.removeprefix("sqlite:///")).resolve()
        raise ValueError("Only sqlite database_url is supported in this project.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取并缓存配置，同时确保运行所需目录存在。"""
    settings = Settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    (settings.output_path / "pdf").mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
