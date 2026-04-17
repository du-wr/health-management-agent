"""add session report link table

Revision ID: 20260417_0005
Revises: 20260417_0004
Create Date: 2026-04-17 10:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260417_0005"
down_revision = "20260417_0004"
branch_labels = None
depends_on = None


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
    """新增会话-报告历史关联表。"""

    if not _has_table("sessionreportlink"):
        op.create_table(
            "sessionreportlink",
            sa.Column("id", sa.String(length=255), primary_key=True, nullable=False),
            sa.Column("session_id", sa.String(length=255), nullable=False),
            sa.Column("report_id", sa.String(length=255), nullable=False),
            sa.Column("linked_at", sa.DateTime(timezone=True), nullable=False),
        )
    _create_index_if_missing("ix_sessionreportlink_session_id", "sessionreportlink", ["session_id"])
    _create_index_if_missing("ix_sessionreportlink_report_id", "sessionreportlink", ["report_id"])
    _create_index_if_missing("ix_sessionreportlink_linked_at", "sessionreportlink", ["linked_at"])


def downgrade() -> None:
    """回退会话-报告历史关联表。"""

    for index_name in [
        "ix_sessionreportlink_linked_at",
        "ix_sessionreportlink_report_id",
        "ix_sessionreportlink_session_id",
    ]:
        op.drop_index(index_name, table_name="sessionreportlink")
    op.drop_table("sessionreportlink")
