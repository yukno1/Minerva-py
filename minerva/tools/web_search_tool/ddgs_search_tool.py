from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
import trafilatura
from langchain_chroma import Chroma

from pathlib import Path


def _project_root() -> Path:
    start = Path(__file__).resolve().parent
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b", base_url="http://localhost:11434"
)

ROOT = _project_root()  # 项目根目录


def web_search(
    query: str,
    max_results: int | str = 5,
    # include_answer: bool | str = True,
) -> dict[str, Any]:
    """Search the web with DDGS and return a compact structured result."""

    vector_store = _connect_chroma()

    try:
        from ddgs import DDGS
    except ImportError as exc:
        return {"ok": False, "error": f"ddgs is not installed: {exc}"}

    try:
        max_value = int(max_results)
    except (TypeError, ValueError):
        max_value = 5
    max_value = max(1, min(max_value, 10))
    # answer_value = _coerce_bool(include_answer)

    try:
        # client = TavilyClient(api_key=api_key)
        # response = client.search(
        #     query=query,
        #     search_depth="basic",
        #     max_results=max_value,
        #     include_answer=answer_value,
        # )
        with DDGS() as ddgs:
            responses = ddgs.text(query=query, max_results=max_value)
    except Exception as exc:
        return {"ok": False, "query": query, "error": f"{type(exc).__name__}: {exc}"}

    results = []
    docs = []
    ids = []
    for item in responses:
        title = item["title"]
        url = item["href"]
        try:
            content = _fetch(url)
            if content and len(content) > 200:
                # 向量数据库不适合存一整本书，通常按 500-1000 字切分
                chunks = [
                    content[i : i + 800] for i in range(0, len(content), 600)
                ]  # 每块800字，重叠200字

                for chunk in chunks:
                    doc = Document(
                        page_content=chunk, metadata={"title": title, "url": url}
                    )
                    doc_id = str(uuid4())
                    docs.append(doc)
                    ids.append(doc_id)
                    print(chunk)

        except Exception as e:
            print(f"抓取失败: {e}")

    vector_store.add_documents(documents=docs, ids=ids)

    results = vector_store.similarity_search_with_score(query, k=max_value)

    res = []
    for doc, score in results:
        res.append(
            {
                "title": doc.metadata["title"],
                "url": doc.metadata["url"],
                "content": doc.page_content,
                "score": score,
            }
        )

    return {
        "ok": True,
        "query": query,
        "answer": "",
        "results": res,
    }


def ddgs_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="WebSearchTool",
        func=web_search,
        description=(
            "Search the web with ddgs. Args: query, optional max_results. "
            "Returns result sources with title, url, content, and score."
        ),
    )


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


def _fetch(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    # 提取纯正文 (自动去噪)
    # output_format='txt' 返回纯文本，'xml' 返回带结构的 XML
    text = trafilatura.extract(downloaded, output_format="txt", include_comments=False)
    return text


def _connect_chroma() -> Chroma:
    URI = str(ROOT / "chroma-store" / "ddgs_vector_collection.db")
    Path(URI).parent.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=URI,
    )


def _connect_milvus():
    """现在langchain的集成有bug"""
    from langchain_milvus import Milvus

    URI = str(ROOT / "milvus-store" / "ddgs_vector_collection.db")
    Path(URI).parent.mkdir(parents=True, exist_ok=True)

    return Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )


if __name__ == "__main__":
    from pprint import pprint

    results = web_search("帮我查阅明日方舟阿米娅")

    pprint(results)
