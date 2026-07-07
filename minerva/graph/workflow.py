from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from minerva.graph.nodes import (
    actor_node,
    final_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from minerva.graph.state import MinervaGraphState


def build_workflow():
    graph = StateGraph(MinervaGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_route, {"planner": "planner", "final": "final"}
    )
    graph.add_edge("final", END)
    return graph.compile()
