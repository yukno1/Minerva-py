from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b", base_url="http://localhost:11434")

if __name__ == "__main__":
    print(llm.invoke("Hello, world!"))
