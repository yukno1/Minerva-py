from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from minerva.core.path import default_workspace
from minerva.core.state import RuntimeState
from minerva.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from minerva.providers.ollama import create_model
from minerva.tools import get_tools


def create_runtime(workspace: Path | None = None) -> RuntimeState:
    selected = workspace or default_workspace()
    selected.mkdir(parents=True, exist_ok=True)
    return RuntimeState(workspace=selected)


def create_code_agent(state: RuntimeState):
    model = create_model("ornith:9b")
    return create_agent(
        model=model,
        tools=get_tools(state),
        system_prompt=STAGE1_SYSTEM_PROMPT,
    )


def stream_agent_events(
    task: str, *, workspace: Path | None = None
) -> Iterator[dict[str, Any]]:
    state = create_runtime(workspace)
    agent = create_code_agent(state)
    yield {"type": "workspace", "path": str(state.workspace)}

    inputs = {"messages": [HumanMessage(content=task)]}
    for event in agent.stream(inputs, stream_mode="updates"):
        yield {"type": "agent_event", "event": event}
