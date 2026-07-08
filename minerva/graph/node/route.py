from __future__ import annotations

from minerva.graph.state import MinervaGraphState


def context_monitor_route(state: MinervaGraphState) -> str:
    if state.get("context_should_compress"):
        return "context_compressor"
    return state.get("context_next_node") or "verifier"


def context_compressor_route(state: MinervaGraphState) -> str:
    return state.get("context_next_node") or "verifier"


def verifier_route(state: MinervaGraphState) -> str:
    if state.get("passed"):
        return "final"
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return "final"
    return "planner"
