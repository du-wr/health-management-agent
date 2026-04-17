"""add agent memory tables

Revision ID: 20260417_0004
Revises: 20260417_0003
Create Date: 2026-04-17 02:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = "20260417_0004"
down_revision = "20260417_0003"
branch_labels = None
depends_on = None


def _text_type():
    bind = op.get_bind()
    if bind.dialect.name == "mysql":
        return mysql.LONGTEXT()
    return sa.Text()


def _inspector():
    return sa.inspect(op.get_bind())


def _has_table(table_name: str) -> bool:
    return table_name in _inspector().get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    return any(index.get("name") == index_name for index in _inspector().get_indexes(table_name))


def _create_index_if_missing(index_name: str, table_name: str, columns: list[str]) -> None:
    if not _has_index(table_name, index_name):
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    """新增会话摘要记忆和报告长期洞察表。"""

    text_type = _text_type()

    if not _has_table("sessionmemory"):
        op.create_table(
            "sessionmemory",
            sa.Column("session_id", sa.String(length=255), primary_key=True, nullable=False),
            sa.Column("report_id", sa.String(length=255), nullable=True),
            sa.Column("latest_run_id", sa.String(length=255), nullable=True),
            sa.Column("summary_text", text_type, nullable=False),
            sa.Column("focus_points_json", text_type, nullable=False),
            sa.Column("latest_intent", sa.String(length=255), nullable=True),
            sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    _create_index_if_missing("ix_sessionmemory_report_id", "sessionmemory", ["report_id"])
    _create_index_if_missing("ix_sessionmemory_latest_run_id", "sessionmemory", ["latest_run_id"])
    _create_index_if_missing("ix_sessionmemory_latest_intent", "sessionmemory", ["latest_intent"])
    _create_index_if_missing("ix_sessionmemory_updated_at", "sessionmemory", ["updated_at"])

    if not _has_table("reportinsight"):
        op.create_table(
            "reportinsight",
            sa.Column("report_id", sa.String(length=255), primary_key=True, nullable=False),
            sa.Column("parse_status", sa.String(length=255), nullable=False, server_default="uploaded"),
            sa.Column("abnormal_item_names_json", text_type, nullable=False),
            sa.Column("key_findings_json", text_type, nullable=False),
            sa.Column("monitoring_summary", text_type, nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    _create_index_if_missing("ix_reportinsight_parse_status", "reportinsight", ["parse_status"])
    _create_index_if_missing("ix_reportinsight_updated_at", "reportinsight", ["updated_at"])


def downgrade() -> None:
    """回退 Agent memory 相关表。"""

    for index_name in [
        "ix_reportinsight_updated_at",
        "ix_reportinsight_parse_status",
    ]:
        op.drop_index(index_name, table_name="reportinsight")
    op.drop_table("reportinsight")

    for index_name in [
        "ix_sessionmemory_updated_at",
        "ix_sessionmemory_latest_intent",
        "ix_sessionmemory_latest_run_id",
        "ix_sessionmemory_report_id",
    ]:
        op.drop_index(index_name, table_name="sessionmemory")
    op.drop_table("sessionmemory")
