from pathlib import Path

from src.retriever import retrieve_documents


def _display_source(source):
    return Path(str(source).replace("\\", "/")).name or "未知来源"

def knowledge_search(
    question,
    documents,
    vector_store,
):
    retrieved_docs = retrieve_documents(
        question=question,
        documents=documents,
        vector_store=vector_store,
    )

    results = []

    for index, doc in enumerate(
        retrieved_docs,
        start=1,
    ):
        results.append(
            {
                "index": index,
                "content": doc.page_content,
                "source": doc.metadata.get(
                    "source",
                    "未知来源",
                ),
            }
        )

        results[-1]["source"] = _display_source(results[-1]["source"])

    return results

def list_knowledge_sources(documents):
    sources = []

    for doc in documents:
        source = doc.metadata.get(
            "source",
            "未知来源",
        )

        display_source = _display_source(source)
        if display_source not in sources:
            sources.append(display_source)

    return sources
