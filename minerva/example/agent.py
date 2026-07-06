from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


llm = ChatOllama(model="qwen3:4b", base_url="http://localhost:11434")


# tool
# Python callable, LangChain tool, or tool dict
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# system prompt
system_prompt = "You are a helpful assistant. Be concise and accurate."


# structured output
class Answer(BaseModel):
    summary: str
    confidence: float


# invocation
config = {"configurable": {"thread_id": str(uuid7())}}

# model
# Pass a model identifier string ("provider:model") or an initialized model instance
agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt=system_prompt,
    response_format=Answer,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)
print(result["structured_response"])  # Answer(summary=..., confidence=...)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
print(result["structured_response"])
