from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.config import get_stream_writer

from minerva.agent.code_agent import run_code_agent
from minerva.agent.search_agent import run_search_agent
from minerva.graph.state import MinervaGraphState, TodoItem, VerificationCheck
from minerva.tools import get_read_only_tools
from minerva.tools.todo_tool import write_todos
from minerva.providers.ollama import create_model
from minerva.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
    persist_history_summary,
)
from .const import AMIYA_COMMANDS


def estimate_context_tokens(state: MinervaGraphState) -> int:
    messages = list(state.get("messages", []))
    payload = build_layered_memory(state, node="context_monitor")
    payload_message = HumanMessage(
        content=json.dumps(payload, ensure_ascii=False, default=str)
    )
    try:
        model = create_model("ornith:9b")
        return int(model.get_num_tokens_from_messages(messages + [payload_message]))
    except Exception:
        text = "\n".join(_message_text(message) for message in messages)
        text += "\n" + payload_message.content
        return max(1, len(text) // 4)


def _context_payload(state: MinervaGraphState) -> dict[str, Any]:
    return build_layered_memory(state, node="graph")


def _todo_write_tool(
    state: MinervaGraphState,
    writer,
    todos: Any,
    acceptance_criteria: Any,
    verification_commands: Any,
    plan_summary: str = "",
) -> dict[str, Any]:
    result = write_todos(todos, acceptance_criteria, verification_commands)
    if result.get("ok"):
        state["plan_summary"] = (
            plan_summary or state.get("plan_summary") or "MultiAgent plan"
        )
        state["todos"] = _todo_items(result["todos"], existing=state.get("todos", []))
        state["acceptance_criteria"] = result["acceptance_criteria"]
        state["verification_commands"] = result["verification_commands"]
        writer(
            {
                "type": "plan_snapshot",
                "node": "planner",
                "plan_summary": state.get("plan_summary", ""),
                "todos": state.get("todos", []),
                "verification_commands": state.get("verification_commands", []),
                "acceptance_criteria": state.get("acceptance_criteria", []),
            }
        )
    return {
        **result,
        "plan_summary": state.get("plan_summary", ""),
        "todo_items": state.get("todos", []),
    }


def _call_search_agent_tool(
    state: MinervaGraphState, writer, instruction: str
) -> dict[str, Any]:
    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "searchAgent",
            "instruction": instruction,
        }
    )
    result = run_search_agent(state, instruction, writer=writer)
    existing_sources = list(state.get("sources", []))
    state["research_notes"] = _join_notes(
        state.get("research_notes", ""), result.get("summary", "")
    )
    state["sources"] = _dedupe_sources(
        existing_sources + list(result.get("sources", []))
    )
    handoff = {
        "from_agent": "planner",
        "to_agent": "searchAgent",
        "instruction": instruction,
        "result": result.get("summary", ""),
    }
    state["agent_handoffs"] = list(state.get("agent_handoffs", [])) + [handoff]
    writer(
        {
            "type": "handoff_result",
            "from": "searchAgent",
            "to": "planner",
            "result": result.get("summary", ""),
        }
    )
    return {
        "ok": True,
        "summary": result.get("summary", ""),
        "sources": state.get("sources", []),
        "queries": result.get("queries", []),
    }


def _call_code_agent_tool(
    state: MinervaGraphState, writer, instruction: str
) -> dict[str, Any]:
    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "codeAgent",
            "instruction": instruction,
        }
    )
    result = run_code_agent(state, instruction, writer=writer)
    state["todos"] = result.get("todos", state.get("todos", []))
    state["code_agent_summary"] = result.get("summary", "")
    state["last_actor_summary"] = result.get("summary", "")
    handoff = {
        "from_agent": "planner",
        "to_agent": "codeAgent",
        "instruction": instruction,
        "result": result.get("summary", ""),
    }
    state["agent_handoffs"] = list(state.get("agent_handoffs", [])) + [handoff]
    writer(
        {
            "type": "handoff_result",
            "from": "codeAgent",
            "to": "planner",
            "result": result.get("summary", ""),
        }
    )
    return {
        "ok": True,
        "summary": result.get("summary", ""),
        "todos": state.get("todos", []),
    }


def _execute_read_only_tool(
    state: MinervaGraphState, call: dict[str, Any]
) -> ToolMessage:
    name = call.get("name", "")
    args = call.get("args") or {}
    tools = {tool.name: tool for tool in get_read_only_tools(state["runtime"])}
    tool = tools.get(name)
    if tool is None:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return ToolMessage(
        content=json.dumps(result, ensure_ascii=False),
        name=name,
        tool_call_id=call.get("id") or f"{name}-call",
    )


def _fallback_compression(
    state: MinervaGraphState, *, error: str = ""
) -> dict[str, Any]:
    return {
        "summary": _short_text(
            "\n\n".join(
                [
                    state.get("context_summary", ""),
                    state.get("research_notes", ""),
                    state.get("code_agent_summary", ""),
                    state.get("verifier_summary", ""),
                    state.get("last_error", ""),
                ]
            ),
            2400,
        ),
        "active_goal": state.get("task", ""),
        "completed_work": state.get("code_agent_summary", ""),
        "open_todos": [
            todo.get("content", "")
            for todo in state.get("todos", [])
            if todo.get("status") != "completed"
        ],
        "important_files": _important_files_from_state(state),
        "tool_findings": _short_text(state.get("last_error", ""), 1200),
        "sources": [
            {"title": source.get("title", ""), "url": source.get("url", "")}
            for source in state.get("sources", [])
        ],
        "next_steps": state.get("context_next_node", ""),
        "risks": error,
    }


def _message_snapshot(message: Any) -> dict[str, str]:
    return {
        "type": type(message).__name__,
        "name": str(getattr(message, "name", "") or ""),
        "content": _short_text(_message_text(message), 2000),
    }


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _important_files_from_state(state: MinervaGraphState) -> list[str]:
    files: list[str] = []
    for command in state.get("verification_commands", []):
        files.extend(re.findall(r"[\w./\\-]+\.(?:py|html|css|js|json|md|txt)", command))
    for text in [state.get("code_agent_summary", ""), state.get("last_error", "")]:
        files.extend(re.findall(r"[\w./\\-]+\.(?:py|html|css|js|json|md|txt)", text))
    seen: set[str] = set()
    deduped = []
    for item in files:
        normalized = item.strip("\"'")
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _verification_commands_for_task(task: str, parsed: dict[str, Any]) -> list[str]:
    if _is_amiya_task(task):
        return AMIYA_COMMANDS
    return [str(item) for item in parsed.get("verification_commands") or []]


def _todo_items(
    todos: list[str], *, existing: list[dict[str, Any]] | None = None
) -> list[TodoItem]:
    existing_by_content = {todo.get("content", ""): todo for todo in existing or []}
    items: list[TodoItem] = []
    for idx, todo in enumerate(todos, start=1):
        previous = existing_by_content.get(todo, {})
        items.append(
            {
                "id": str(previous.get("id") or f"todo-{idx}"),
                "content": todo,
                "status": str(previous.get("status") or "pending"),
                "note": str(previous.get("note") or ""),
            }
        )
    return items


def _extract_json(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_result_event(tool_message: ToolMessage, *, node: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        parsed = tool_message.content
    return {
        "type": "tool_result",
        "node": node,
        "name": tool_message.name,
        "result": parsed,
    }


def _tool_events_to_verification_results(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for event in events:
        result = event.get("result", {})
        if not isinstance(result, dict):
            continue
        results.append(
            {
                "command": result.get("command") or event.get("name", ""),
                "ok": bool(result.get("ok")),
                "exit_code": result.get("exit_code"),
                "stdout": str(result.get("stdout", "")),
                "stderr": str(result.get("stderr") or result.get("error", "")),
            }
        )
    return results


def _normalize_checks(raw: Any) -> list[VerificationCheck]:
    if not isinstance(raw, list):
        return []
    checks: list[VerificationCheck] = []
    for item in raw:
        if isinstance(item, dict):
            checks.append(
                {
                    "name": str(item.get("name") or "check"),
                    "passed": bool(item.get("passed")),
                    "detail": str(item.get("detail") or ""),
                }
            )
    return checks


def _join_notes(existing: str, new: str) -> str:
    if not existing:
        return new
    if not new:
        return existing
    return existing + "\n\n" + new


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for source in sources:
        url = str(source.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(source)
    return deduped


def _trim_handoffs(handoffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = []
    for handoff in handoffs[-6:]:
        trimmed.append(
            {
                "from_agent": handoff.get("from_agent", ""),
                "to_agent": handoff.get("to_agent", ""),
                "instruction": _short_text(str(handoff.get("instruction", "")), 500),
                "result": _short_text(str(handoff.get("result", "")), 700),
            }
        )
    return trimmed


def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""


def _todos_text(todos: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {todo.get('id', '')} [{todo.get('status', '')}] {todo.get('content', '')} {todo.get('note', '')}".strip()
        for todo in todos
    )


def _list_text(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _is_amiya_task(task: str) -> bool:
    lowered = task.lower()
    return (
        "阿米娅" in task
        or "amiya" in lowered
        or "arknights" in lowered
        or "明日方舟" in task
    )


def _get_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
