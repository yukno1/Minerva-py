from pydantic import BaseModel, Field
from .llm import llm


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


if __name__ == "__main__":
    # Augment the LLM with schema for structured output
    structured_llm = llm.with_structured_output(SearchQuery)

    # Invoke the augmented LLM
    output = structured_llm.invoke(
        "How does Calcium CT score relate to high cholesterol?"
    )

    # Define a tool
    def multiply(a: int, b: int) -> int:
        return a * b

    # Augment the LLM with tools
    llm_with_tools = llm.bind_tools([multiply])

    # Invoke the LLM with input that triggers the tool call
    msg = llm_with_tools.invoke("What is 2 times 3?")

    # Get the tool call
    msg.tool_calls
