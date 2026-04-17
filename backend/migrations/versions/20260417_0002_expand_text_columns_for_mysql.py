"""expand text columns for mysql

Revision ID: 20260417_0002
Revises: 20260417_0001
Create Date: 2026-04-17 00:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = "20260417_0002"
down_revision = "20260417_0001"
branch_labels = None
depends_on = None


def _text_type():
    bind = op.get_bind()
    if bind.dialect.name == "mysql":
        return mysql.LONGTEXT()
    return sa.Text()


def upgrade() -> None:
    """把易超长的文本字段扩成 TEXT/LONGTEXT。"""
    text_type = _text_type()

    columns_to_expand = {
        "report": [("raw_text", False), ("parse_warnings_json", False)],
        "reportparsetask": [("last_error", True)],
        "agentanswercache": [("normalized_message", False), ("response_json", False), ("answer_text", False)],
        "labitem": [("clinical_note", True)],
        "chatmessage": [("content", False), ("citations_json", False)],
        "knowledgedoc": [("snippet", False), ("body_text", False)],
        "summaryartifact": [("markdown", False)],
    }

    for table_name, columns in columns_to_expand.items():
        with op.batch_alter_table(table_name) as batch_op:
            for column_name, nullable in columns:
                batch_op.alter_column(
                    column_name,
                    existing_type=sa.String(length=255),
                    type_=text_type,
                    existing_nullable=nullable,
                )


def downgrade() -> None:
    """回退到旧的 VARCHAR(255) 文本字段。"""
    columns_to_shrink = {
        "report": [("raw_text", False), ("parse_warnings_json", False)],
        "reportparsetask": [("last_error", True)],
        "agentanswercache": [("normalized_message", False), ("response_json", False), ("answer_text", False)],
        "labitem": [("clinical_note", True)],
        "chatmessage": [("content", False), ("citations_json", False)],
        "knowledgedoc": [("snippet", False), ("body_text", False)],
        "summaryartifact": [("markdown", False)],
    }

    for table_name, columns in columns_to_shrink.items():
        with op.batch_alter_table(table_name) as batch_op:
            for column_name, nullable in columns:
                batch_op.alter_column(
                    column_name,
                    existing_type=sa.Text(),
                    type_=sa.String(length=255),
                    existing_nullable=nullable,
                )
