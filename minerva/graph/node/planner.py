from __future__ import annotations

import json
from typing import Any


from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from minerva.graph.state import MinervaGraphState
from minerva.prompts.stage3 import PLANNER_PROMPT
from minerva.providers.ollama import create_model
from minerva.tools.todo_tool import persist_todos, write_todos
from minerva.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
    persist_history_summary,
)

from ._common import (
    _get_writer,
    _last_ai_content,
    _todo_write_tool,
    _call_search_agent_tool,
    _call_code_agent_tool,
    _tool_result_event,
    _todo_items,
    _verification_commands_for_task,
    _todos_text,
    _list_text,
    _is_amiya_task,
)
from .const import (
    AMIYA_TODOS,
    AMIYA_CRITERIA,
    AMIYA_COMMANDS,
    DEFAULT_TODOS,
)


def planner_node(state: MinervaGraphState) -> dict[str, Any]:
    writer = _get_writer()
    working_state: MinervaGraphState = {**state}
    if not working_state.get("todos"):
        _apply_plan(working_state, _default_plan(working_state["task"]))
        persist_todos(
            working_state["runtime"],
            working_state.get("todos", []),
            working_state.get("acceptance_criteria", []),
            working_state.get("verification_commands", []),
            working_state.get("plan_summary", ""),
        )

    memory = build_layered_memory(working_state, node="planner")
    writer(memory_event(memory, node="planner"))

    model = create_model("ornith:9b")

    planner = model.bind_tools(_get_planner_tools(working_state, writer))
    messages: list[Any] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=_planner_input(working_state, memory)),
    ]
    produced_messages: list[Any] = []

    writer(
        {
            "type": "plan_snapshot",
            "node": "planner",
            "plan_summary": working_state.get("plan_summary", ""),
            "todos": working_state.get("todos", []),
            "verification_commands": working_state.get("verification_commands", []),
            "attempts": working_state.get("attempts", 0),
        }
    )

    for _ in range(8):
        response = planner.invoke(messages)
        produced_messages.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            tool_message = _execute_planner_tool(working_state, writer, call)
            produced_messages.append(tool_message)
            messages.append(tool_message)
    else:
        produced_messages.append(
            AIMessage(
                content="planner stopped after the maximum supervisor tool loop count."
            )
        )

    metadata = dict(working_state.get("metadata", {}))
    metadata["planner_raw"] = _last_ai_content(produced_messages)
    return {
        "plan_summary": working_state.get("plan_summary", ""),
        "todos": working_state.get("todos", []),
        "acceptance_criteria": working_state.get("acceptance_criteria", []),
        "verification_commands": working_state.get("verification_commands", []),
        "research_notes": working_state.get("research_notes", ""),
        "sources": working_state.get("sources", []),
        "agent_handoffs": working_state.get("agent_handoffs", []),
        "code_agent_summary": working_state.get("code_agent_summary", ""),
        "last_actor_summary": working_state.get("code_agent_summary", ""),
        "messages": produced_messages,
        "metadata": metadata,
        "context_next_node": "verifier",
    }


def _get_planner_tools(state: MinervaGraphState, writer) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="TodoWriteTool",
            func=lambda todos,
            acceptance_criteria,
            verification_commands,
            plan_summary="": _todo_write_tool(
                state,
                writer,
                todos,
                acceptance_criteria,
                verification_commands,
                plan_summary,
            ),
            description=(
                "Publish or revise plan state. Args: todos, acceptance_criteria, "
                "verification_commands, optional plan_summary."
            ),
        ),
        StructuredTool.from_function(
            name="CallSearchAgentTool",
            func=lambda instruction: _call_search_agent_tool(
                state, writer, instruction
            ),
            description="Delegate research work to searchAgent. Args: instruction.",
        ),
        StructuredTool.from_function(
            name="CallCodeAgentTool",
            func=lambda instruction: _call_code_agent_tool(state, writer, instruction),
            description="Delegate implementation work to codeAgent. Args: instruction.",
        ),
    ]


def _execute_planner_tool(
    state: MinervaGraphState, writer, call: dict[str, Any]
) -> ToolMessage:
    name = call.get("name", "")
    args = call.get("args") or {}
    writer({"type": "tool_call", "node": "planner", "name": name, "args": args})
    tools = {tool.name: tool for tool in _get_planner_tools(state, writer)}
    tool = tools.get(name)
    if tool is None:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    tool_message = ToolMessage(
        content=json.dumps(result, ensure_ascii=False),
        name=name,
        tool_call_id=call.get("id") or f"{name}-call",
    )
    writer(_tool_result_event(tool_message, node="planner"))
    return tool_message


def _apply_plan(state: MinervaGraphState, plan: dict[str, Any]) -> None:
    state["plan_summary"] = str(plan.get("plan_summary", ""))
    state["todos"] = _todo_items(
        [str(item) for item in plan.get("todos", [])], existing=state.get("todos", [])
    )
    state["acceptance_criteria"] = [
        str(item) for item in plan.get("acceptance_criteria", [])
    ]
    state["verification_commands"] = _verification_commands_for_task(
        state["task"], plan
    )


def _planner_input(state: MinervaGraphState, memory: dict[str, Any]) -> str:
    return (
        f"Task: {state['task']}\n"
        f"Attempt: {state.get('attempts', 0) + 1}\n\n"
        "Layered memory snapshot:\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def _default_plan(task: str) -> dict[str, Any]:
    if _is_amiya_task(task):
        return {
            "plan_summary": "Research Amiya from Arknights and build a sourced HTML character profile.",
            "todos": AMIYA_TODOS,
            "acceptance_criteria": AMIYA_CRITERIA,
            "verification_commands": AMIYA_COMMANDS,
        }
    return {
        "plan_summary": "Coordinate specialist agents to complete and verify the requested deliverable.",
        "todos": DEFAULT_TODOS,
        "acceptance_criteria": [
            "The requested deliverable exists.",
            "The verifier model confirms completion.",
        ],
        "verification_commands": [],
    }
