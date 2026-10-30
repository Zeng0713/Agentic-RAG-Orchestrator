from src.retriever import retrieve_documents

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

    return results

def list_knowledge_sources(documents):
    sources = []

    for doc in documents:
        source = doc.metadata.get(
            "source",
            "未知来源",
        )

        if source not in sources:
            sources.append(source)

    return sources