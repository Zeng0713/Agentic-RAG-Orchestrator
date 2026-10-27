def find_title_match(question, documents):
    """
    根据用户问题中的关键词，
    寻找标题最匹配的文档。
    """

    question_lower = question.lower().strip()

    best_doc = None
    best_score = 0

    for doc in documents:
        title = doc.page_content.splitlines()[0].strip()

        if not title.startswith("## "):
            continue

        clean_title = title.replace(
            "## ",
            "",
            1,
        ).strip()

        topic = clean_title.replace(
            "什么是",
            "",
            1,
        ).strip()

        topic_lower = topic.lower()

        keywords = topic_lower.split("的")

        score = 0

        for keyword in keywords:
            keyword = keyword.strip()

            if keyword and keyword in question_lower:
                score += len(keyword)

        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc
def retrieve_documents(
    question,
    documents,
    vector_store,
    k=4,
    max_results=2,
):
    """
    使用标题关键词匹配 + 向量检索，
    返回最终用于 RAG 的参考文档。
    """

    # 1. 标题关键词匹配
    matched_doc = find_title_match(
        question,
        documents,
    )

    if matched_doc:
        print(
            "\n命中标题关键词：",
            matched_doc.page_content.splitlines()[0],
        )

    # 2. 向量相似度检索
    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    # 3. 保存最终检索结果
    retrieved_docs = []

    # 标题匹配结果优先加入
    if matched_doc:
        retrieved_docs.append(matched_doc)

    # 4. 再加入向量检索结果
    for doc, score in results:
        if len(retrieved_docs) >= max_results:
            break

        # 防止同一个 Chunk 重复加入
        if all(
            existing.page_content != doc.page_content
            for existing in retrieved_docs
        ):
            retrieved_docs.append(doc)

    return retrieved_docs