from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from minerva.graph.state import MinervaGraphState
from minerva.prompts.stage3 import VERIFIER_PROMPT
from minerva.providers.ollama import create_model
from minerva.tools import get_read_only_tools
from minerva.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
    persist_history_summary,
)
from ._common import (
    _get_writer,
    _execute_read_only_tool,
    _tool_result_event,
    _extract_json,
    _last_ai_content,
    _normalize_checks,
    _tool_events_to_verification_results,
    _todos_text,
    _list_text,
)
from .route import verifier_route


def verifier_node(state: MinervaGraphState) -> dict[str, Any]:
    writer = _get_writer()
    memory = build_layered_memory(state, node="verifier")
    writer(memory_event(memory, node="verifier"))
    writer(
        {
            "type": "plan_snapshot",
            "node": "verifier",
            "plan_summary": state.get("plan_summary", ""),
            "todos": state.get("todos", []),
            "verification_commands": state.get("verification_commands", []),
        }
    )

    model = create_model("ornith:9b")
    verifier = model.bind_tools(get_read_only_tools(state["runtime"]))
    messages: list[Any] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=_verifier_input(state, memory)),
    ]
    produced_messages: list[Any] = []
    tool_events: list[dict[str, Any]] = []

    for _ in range(8):
        response = verifier.invoke(messages)
        produced_messages.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            writer(
                {
                    "type": "tool_call",
                    "node": "verifier",
                    "name": call.get("name"),
                    "args": call.get("args", {}),
                }
            )
            tool_message = _execute_read_only_tool(state, call)
            event = _tool_result_event(tool_message, node="verifier")
            tool_events.append(event)
            writer(event)
            produced_messages.append(tool_message)
            messages.append(tool_message)
    else:
        produced_messages.append(
            AIMessage(
                content=json.dumps(
                    {
                        "passed": False,
                        "reason": "Verifier stopped after the maximum tool loop count.",
                        "checks": [],
                        "recommended_next_instruction": "Inspect the workspace and complete the unfinished task.",
                    },
                    ensure_ascii=False,
                )
            )
        )

    parsed = _extract_json(_last_ai_content(produced_messages)) or {
        "passed": False,
        "reason": "Verifier did not return valid JSON.",
        "checks": [
            {
                "name": "verifier_json",
                "passed": False,
                "detail": _last_ai_content(produced_messages)[:800],
            }
        ],
        "recommended_next_instruction": "Return valid verifier JSON after inspecting the result.",
    }
    checks = _normalize_checks(parsed.get("checks"))
    passed = bool(parsed.get("passed"))
    reason = str(parsed.get("reason") or "")
    recommended = str(parsed.get("recommended_next_instruction") or "")
    attempts = state.get("attempts", 0) + 1
    todos = [dict(todo) for todo in state.get("todos", [])]
    if passed:
        todos = [
            {
                **todo,
                "status": "completed"
                if todo.get("status") != "blocked"
                else todo.get("status", "blocked"),
                "note": todo.get("note") or "verified",
            }
            for todo in todos
        ]
        writer(
            {
                "type": "todo_update",
                "node": "verifier",
                "plan_summary": state.get("plan_summary", ""),
                "todos": todos,
                "verification_commands": state.get("verification_commands", []),
            }
        )
    last_error = (
        "" if passed else _format_verifier_error(reason, recommended, tool_events)
    )

    return {
        "messages": produced_messages,
        "verification_results": _tool_events_to_verification_results(tool_events),
        "verification_checks": checks,
        "verifier_summary": reason,
        "passed": passed,
        "attempts": attempts,
        "last_error": last_error,
        "todos": todos,
        "memory_snapshot": memory,
        "history_summary": memory.get("history_summary_store", {}).get(
            "history_summary", ""
        ),
        "context_next_node": verifier_route(
            {**state, "passed": passed, "attempts": attempts}
        ),
    }


def _format_verifier_error(
    reason: str, recommended: str, tool_events: list[dict[str, Any]]
) -> str:
    event_text = json.dumps(tool_events[-3:], ensure_ascii=False, default=str)[:1600]
    return (
        f"Verifier failed: {reason}\n"
        f"Recommended next instruction: {recommended}\n"
        f"Recent verifier tool events:\n{event_text}"
    )


def _verifier_input(state: MinervaGraphState, memory: dict[str, Any]) -> str:
    source_text = "\n".join(
        f"- {source.get('title', '')}: {source.get('url', '')}"
        for source in state.get("sources", [])
    )
    return (
        f"Task: {state['task']}\n\n"
        "Layered memory snapshot:\n"
        f"{format_layered_memory_for_prompt(memory)}\n\n"
        "Inspect the workspace with tools and return only verifier JSON."
    )
