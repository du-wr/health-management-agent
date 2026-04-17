"""add agent runtime tables

Revision ID: 20260417_0003
Revises: 20260417_0002
Create Date: 2026-04-17 01:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = "20260417_0003"
down_revision = "20260417_0002"
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


def _has_column(table_name: str, column_name: str) -> bool:
    return any(column.get("name") == column_name for column in _inspector().get_columns(table_name))


def _create_index_if_missing(index_name: str, table_name: str, columns: list[str]) -> None:
    if not _has_index(table_name, index_name):
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    """新增 Agent 目标、运行记录、轨迹表，并给消息表补运行关联字段。"""

    text_type = _text_type()

    if not _has_table("agentgoal"):
        op.create_table(
            "agentgoal",
            sa.Column("id", sa.String(length=255), primary_key=True, nullable=False),
            sa.Column("session_id", sa.String(length=255), nullable=False),
            sa.Column("report_id", sa.String(length=255), nullable=True),
            sa.Column("goal_type", sa.String(length=255), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=255), nullable=False, server_default="active"),
            sa.Column("source_intent", sa.String(length=255), nullable=True),
            sa.Column("latest_user_message", text_type, nullable=False),
            sa.Column("summary_json", text_type, nullable=False),
            sa.Column("last_run_id", sa.String(length=255), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    _create_index_if_missing("ix_agentgoal_session_id", "agentgoal", ["session_id"])
    _create_index_if_missing("ix_agentgoal_report_id", "agentgoal", ["report_id"])
    _create_index_if_missing("ix_agentgoal_goal_type", "agentgoal", ["goal_type"])
    _create_index_if_missing("ix_agentgoal_status", "agentgoal", ["status"])
    _create_index_if_missing("ix_agentgoal_source_intent", "agentgoal", ["source_intent"])
    _create_index_if_missing("ix_agentgoal_last_run_id", "agentgoal", ["last_run_id"])
    _create_index_if_missing("ix_agentgoal_created_at", "agentgoal", ["created_at"])
    _create_index_if_missing("ix_agentgoal_updated_at", "agentgoal", ["updated_at"])

    if not _has_table("agenttaskrun"):
        op.create_table(
            "agenttaskrun",
            sa.Column("id", sa.String(length=255), primary_key=True, nullable=False),
            sa.Column("session_id", sa.String(length=255), nullable=False),
            sa.Column("goal_id", sa.String(length=255), nullable=True),
            sa.Column("report_id", sa.String(length=255), nullable=True),
            sa.Column("user_message", text_type, nullable=False),
            sa.Column("normalized_message", text_type, nullable=False),
            sa.Column("status", sa.String(length=255), nullable=False, server_default="running"),
            sa.Column("intent", sa.String(length=255), nullable=True),
            sa.Column("response_mode", sa.String(length=255), nullable=False, server_default="stream"),
            sa.Column("cache_status", sa.String(length=255), nullable=False, server_default="miss"),
            sa.Column("handoff_required", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("answer_excerpt", text_type, nullable=False),
            sa.Column("used_tools_json", text_type, nullable=False),
            sa.Column("debug_json", text_type, nullable=False),
            sa.Column("error_message", text_type, nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        )
    _create_index_if_missing("ix_agenttaskrun_session_id", "agenttaskrun", ["session_id"])
    _create_index_if_missing("ix_agenttaskrun_goal_id", "agenttaskrun", ["goal_id"])
    _create_index_if_missing("ix_agenttaskrun_report_id", "agenttaskrun", ["report_id"])
    _create_index_if_missing("ix_agenttaskrun_status", "agenttaskrun", ["status"])
    _create_index_if_missing("ix_agenttaskrun_intent", "agenttaskrun", ["intent"])
    _create_index_if_missing("ix_agenttaskrun_response_mode", "agenttaskrun", ["response_mode"])
    _create_index_if_missing("ix_agenttaskrun_cache_status", "agenttaskrun", ["cache_status"])
    _create_index_if_missing("ix_agenttaskrun_started_at", "agenttaskrun", ["started_at"])
    _create_index_if_missing("ix_agenttaskrun_finished_at", "agenttaskrun", ["finished_at"])

    if not _has_table("agenttraceevent"):
        op.create_table(
            "agenttraceevent",
            sa.Column("id", sa.String(length=255), primary_key=True, nullable=False),
            sa.Column("run_id", sa.String(length=255), nullable=False),
            sa.Column("sequence_no", sa.Integer(), nullable=False),
            sa.Column("phase", sa.String(length=255), nullable=False),
            sa.Column("step_name", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=255), nullable=False, server_default="completed"),
            sa.Column("payload_json", text_type, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _create_index_if_missing("ix_agenttraceevent_run_id", "agenttraceevent", ["run_id"])
    _create_index_if_missing("ix_agenttraceevent_sequence_no", "agenttraceevent", ["sequence_no"])
    _create_index_if_missing("ix_agenttraceevent_phase", "agenttraceevent", ["phase"])
    _create_index_if_missing("ix_agenttraceevent_step_name", "agenttraceevent", ["step_name"])
    _create_index_if_missing("ix_agenttraceevent_status", "agenttraceevent", ["status"])
    _create_index_if_missing("ix_agenttraceevent_created_at", "agenttraceevent", ["created_at"])

    if not _has_column("chatmessage", "agent_run_id"):
        with op.batch_alter_table("chatmessage") as batch_op:
            batch_op.add_column(sa.Column("agent_run_id", sa.String(length=255), nullable=True))
    _create_index_if_missing("ix_chatmessage_agent_run_id", "chatmessage", ["agent_run_id"])


def downgrade() -> None:
    """回退 Agent runtime 相关结构。"""

    with op.batch_alter_table("chatmessage") as batch_op:
        batch_op.drop_index("ix_chatmessage_agent_run_id")
        batch_op.drop_column("agent_run_id")

    for index_name in [
        "ix_agenttraceevent_created_at",
        "ix_agenttraceevent_status",
        "ix_agenttraceevent_step_name",
        "ix_agenttraceevent_phase",
        "ix_agenttraceevent_sequence_no",
        "ix_agenttraceevent_run_id",
    ]:
        op.drop_index(index_name, table_name="agenttraceevent")
    op.drop_table("agenttraceevent")

    for index_name in [
        "ix_agenttaskrun_finished_at",
        "ix_agenttaskrun_started_at",
        "ix_agenttaskrun_cache_status",
        "ix_agenttaskrun_response_mode",
        "ix_agenttaskrun_intent",
        "ix_agenttaskrun_status",
        "ix_agenttaskrun_report_id",
        "ix_agenttaskrun_goal_id",
        "ix_agenttaskrun_session_id",
    ]:
        op.drop_index(index_name, table_name="agenttaskrun")
    op.drop_table("agenttaskrun")

    for index_name in [
        "ix_agentgoal_updated_at",
        "ix_agentgoal_created_at",
        "ix_agentgoal_last_run_id",
        "ix_agentgoal_source_intent",
        "ix_agentgoal_status",
        "ix_agentgoal_goal_type",
        "ix_agentgoal_report_id",
        "ix_agentgoal_session_id",
    ]:
        op.drop_index(index_name, table_name="agentgoal")
    op.drop_table("agentgoal")
