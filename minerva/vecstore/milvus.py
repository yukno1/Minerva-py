from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from uuid import uuid4

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")


URI = "./milvus-store/ddgs_vector_collection.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

# def add_document(doc: str):
