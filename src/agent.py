import json
from src.tool_schemas import TOOL_SCHEMAS
from src.tool_registry import get_tool


def select_tool(question, llm):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个知识库助手。"
                "请根据用户问题判断是否需要调用工具。"
                "需要查询具体知识时使用 knowledge_search；"
                "需要查看知识库有哪些资料来源时使用 "
                "list_knowledge_sources。"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = llm.invoke(
        messages,
        tools=TOOL_SCHEMAS,
    )

    return response

def execute_tool(
    tool_call,
    documents,
    vector_store,
):
    tool_name = tool_call["name"]
    arguments = tool_call["args"]

    tool = get_tool(tool_name)

    if tool_name == "knowledge_search":
        return tool(
            question=arguments["question"],
            documents=documents,
            vector_store=vector_store,
        )

    if tool_name == "list_knowledge_sources":
        return tool(
            documents=documents,
        )

    raise ValueError(
        f"无法执行工具：{tool_name}"
    )
def run_agent(
    question,
    llm,
    documents,
    vector_store,
):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个知识库助手。"
                "回答知识库相关问题时，"
                "请根据需要调用提供的工具。"
                "需要查询具体知识时使用 "
                "knowledge_search；"
                "需要查看知识库有哪些资料来源时使用 "
                "list_knowledge_sources。"
                "工具返回结果后，请根据工具结果"
                "生成最终回答，不要编造资料中没有的信息。"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = llm.invoke(
        messages,
        tools=TOOL_SCHEMAS,
    )

    if not response.tool_calls:
        return response.content

    messages.append(response)

    for tool_call in response.tool_calls:
        result = execute_tool(
            tool_call=tool_call,
            documents=documents,
            vector_store=vector_store,
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            }
        )

    final_response = llm.invoke(messages)

    return final_response.content