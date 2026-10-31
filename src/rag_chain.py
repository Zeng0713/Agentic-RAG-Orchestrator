from langchain_openai import ChatOpenAI

from src.config import (
    DEEPSEEK_API_KEY,
    LLM_MODEL,
    LLM_BASE_URL,
)
from src.exceptions import LLMServiceError
from src.logger import setup_logger
logger = setup_logger()
def create_llm():
    """
    创建并返回用于 RAG 问答的 DeepSeek 模型。
    """

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    )

    return llm
def generate_answer(
    question,
    retrieved_docs,
    llm,
    on_token=None,
):
    """
    根据检索到的参考资料，
    调用大语言模型生成回答。
    """

    context_parts = []

    for index, doc in enumerate(
        retrieved_docs,
        start=1,
    ):
        source = doc.metadata.get(
            "source",
            "未知来源",
        )

        context_part = f"""
[资料 {index}]
来源：{source}
内容：
{doc.page_content}
"""

        context_parts.append(context_part)

    context = "\n".join(context_parts)

    prompt = f"""
你是一个严谨的知识库问答助手。

请严格遵守以下规则：
1. 优先依据参考资料回答，不要把参考资料中没有的信息当成事实。
2. 如果参考资料直接包含答案，请准确、清晰地总结。
3. 如果参考资料只包含部分答案，请回答有依据的部分，并明确指出缺少哪些信息。
4. 如果参考资料完全无法回答问题，请明确回答：
   "抱歉，当前知识库中的资料不足以回答这个问题。"
5. 不要为了给出完整答案而自行编造事实。
6. 请使用清晰、易懂的中文回答。
7. 只有当某条参考资料直接支持回答中的事实或结论时，才使用 [资料 1]、[资料 2] 这样的编号进行引用。
   不要引用与问题无关、没有提供有效答案、或仅被用来说明“资料不足”的参考资料。
8. 不要自行输出文件路径或“参考来源”部分，文件来源将由程序统一处理。

参考资料：
{context}
用户问题：
{question}

请根据以上规则回答：
"""
    try:
        if on_token is not None and hasattr(llm, "stream"):
            parts = []
            for chunk in llm.stream(prompt):
                content = chunk.content
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                else:
                    text = ""
                if text:
                    parts.append(text)
                    on_token(text)
            return "".join(parts)

        response = llm.invoke(prompt)
        if on_token is not None and isinstance(response.content, str):
            on_token(response.content)
        return response.content

    except Exception as error:
        logger.error(
            "DeepSeek API 调用失败（%s）",
            type(error).__name__,
        )

        raise LLMServiceError(
            "大语言模型服务暂时不可用"
        ) from error
def get_citations(
    answer,
    retrieved_docs,
):
    """
    根据回答中实际使用的 [资料 X]，
    返回对应的真实文件来源。
    """
    citations = []

    for index, doc in enumerate(
        retrieved_docs,
        start=1,
    ):
        citation = f"[资料 {index}]"

        if citation in answer:
            source = doc.metadata.get(
                "source",
                "未知来源",
            )

            citations.append(
                f"- {citation} {source}"
            )
    return citations
