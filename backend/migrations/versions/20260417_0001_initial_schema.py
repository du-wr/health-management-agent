"""initial schema

Revision ID: 20260417_0001
Revises:
Create Date: 2026-04-17 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlmodel import SQLModel

from app.models import entities  # noqa: F401


# revision identifiers, used by Alembic.
revision = "20260417_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """创建当前项目的初始表结构。"""
    bind = op.get_bind()
    SQLModel.metadata.create_all(bind=bind)

    if bind.dialect.name == "sqlite":
        # 仅 SQLite 需要单独创建 FTS5 虚拟表；MySQL 不支持这套语法。
        bind.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_doc_fts
            USING fts5(
                doc_id UNINDEXED,
                title,
                snippet,
                body_text,
                trust_tier UNINDEXED,
                source_domain UNINDEXED
            );
            """
        )


def downgrade() -> None:
    """按依赖逆序删除初始表结构。"""
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        bind.exec_driver_sql("DROP TABLE IF EXISTS knowledge_doc_fts;")

    # 按显式顺序删除，避免部分数据库在存在索引或依赖时失败。
    table_names = [
        "summaryartifact",
        "knowledgedoc",
        "chatmessage",
        "chatsession",
        "labitem",
        "agentanswercache",
        "reportparsetask",
        "report",
    ]
    for table_name in table_names:
        op.drop_table(table_name)
