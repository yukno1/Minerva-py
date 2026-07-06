from langchain.agents import create_agent
from pydantic import BaseModel
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from minerva.providers.ollama import create_model


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
