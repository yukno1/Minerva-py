from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from minerva.core.path import default_workspace
from minerva.core.state import RuntimeState
from minerva.graph.workflow import build_workflow


def create_runtime(workspace: Path | None = None) -> RuntimeState:
    selected = workspace or default_workspace()
    selected.mkdir(parents=True, exist_ok=True)
    return RuntimeState(workspace=selected)


def stream_agent_events(
    task: str,
    *,
    workspace: Path | None = None,
    max_attempts: int = 3,
) -> Iterator[dict[str, Any]]:
    state = create_runtime(workspace)
    workflow = build_workflow()
    yield {"type": "workspace", "path": str(state.workspace)}

    inputs: dict[str, Any] = {
        "task": task,
        "runtime": state,
        "messages": [],
        "attempts": 0,
        "max_attempts": max_attempts,
    }
    for mode, event in workflow.stream(inputs, stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield {"type": "custom_event", "event": event}
        else:
            yield {"type": "graph_event", "event": event}
