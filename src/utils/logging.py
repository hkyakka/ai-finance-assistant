from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Optional

# Context variables for structured logging
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")
turn_id_var: ContextVar[str] = ContextVar("turn_id", default="-")
agent_var: ContextVar[str] = ContextVar("agent", default="-")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        record.turn_id = turn_id_var.get()
        record.agent = agent_var.get()
        return True


class SimpleStructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        msg = record.getMessage()
        return (
            f"{ts} level={record.levelname} logger={record.name} "
            f"request_id={getattr(record,'request_id','-')} session_id={getattr(record,'session_id','-')} "
            f"turn_id={getattr(record,'turn_id','-')} agent={getattr(record,'agent','-')} "
            f"msg={msg}"
        )


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)

    # Replace handlers (avoid duplicate logs in Streamlit reloads)
    root.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(lvl)
    handler.addFilter(ContextFilter())
    handler.setFormatter(SimpleStructuredFormatter())

    root.addHandler(handler)


def set_log_context(*, request_id: str, session_id: Optional[str] = None, turn_id: Optional[str] = None, agent: Optional[str] = None) -> None:
    request_id_var.set(request_id)
    if session_id is not None:
        session_id_var.set(session_id)
    if turn_id is not None:
        turn_id_var.set(str(turn_id))
    if agent is not None:
        agent_var.set(agent)


def set_agent(agent_name: str) -> None:
    agent_var.set(agent_name)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
