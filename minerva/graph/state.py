from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from minerva.core.state import RuntimeState


class TodoItem(TypedDict):
    id: str
    content: str
    status: str
    note: str


class VerificationResult(TypedDict):
    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class MinervaGraphState(TypedDict, total=False):
    task: str
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]
    plan_summary: str
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    verification_results: list[VerificationResult]
    passed: bool
    attempts: int
    max_attempts: int
    final_answer: str
    last_actor_summary: str
    last_error: str
    metadata: dict[str, Any]
