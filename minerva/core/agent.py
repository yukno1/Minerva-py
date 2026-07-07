from langchain.agents import create_agent
from pydantic import BaseModel
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path
from typing import Any, Iterator

from minerva.core.path import default_workspace
from minerva.core.state import RuntimeState
from minerva.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from minerva.providers.ollama import create_model
from minerva.tools import get_tools


llm = create_model("qwen3:4b")

# system prompt
system_prompt = "You are a helpful assistant. Be concise and accurate."


# structured output
class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=system_prompt,
    response_format=Answer,
)


def create_runtime(workspace: Path | None = None) -> RuntimeState:
    selected = workspace or default_workspace()
    selected.mkdir(parents=True, exist_ok=True)
    return RuntimeState(workspace=selected)
