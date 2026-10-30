import json
from src.tool_schemas import TOOL_SCHEMAS
from src.tool_registry import get_tool

MAX_TOOL_ROUNDS = 6
MAX_CONVERSATION_TURNS = 6


def add_conversation_turn(history, question, answer):
    """Append one completed exchange and retain only the latest turns."""
    history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])
    del history[:-2 * MAX_CONVERSATION_TURNS]


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
    if not isinstance(tool_call, dict):
        raise ValueError("工具调用必须是对象")

    tool_name = tool_call.get("name")
    arguments = tool_call.get("args")

    tool = get_tool(tool_name)
    schema = next(
        item["function"]
        for item in TOOL_SCHEMAS
        if item["function"]["name"] == tool_name
    )
    parameters = schema["parameters"]

    if not isinstance(arguments, dict):
        raise ValueError(f"工具 {tool_name} 的参数必须是对象")

    properties = parameters.get("properties", {})
    required = parameters.get("required", [])

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(
            f"工具 {tool_name} 缺少必填参数：{', '.join(missing)}"
        )

    unexpected = set(arguments) - set(properties)
    if unexpected:
        raise ValueError(
            f"工具 {tool_name} 包含未知参数：{', '.join(sorted(unexpected))}"
        )

    for key, value in arguments.items():
        expected_type = properties[key].get("type")
        valid = {
            "string": lambda item: isinstance(item, str),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
        }.get(expected_type)
        if valid and not valid(value):
            raise ValueError(
                f"工具 {tool_name} 参数 {key} 类型错误，预期 {expected_type}"
            )

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
    history=None,
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
                "如果用户使用‘它’、‘这个’等指代或提出追问，"
                "请结合对话历史理解其含义；调用知识搜索时，"
                "将问题整理为独立、完整的检索问题。"
                "工具返回结果后，请根据工具结果"
                "生成最终回答，不要编造资料中没有的信息。"
            ),
        },
    ]
    if history:
        messages.extend(history[-2 * MAX_CONVERSATION_TURNS:])
    messages.append({"role": "user", "content": question})

    for _ in range(MAX_TOOL_ROUNDS):
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

    messages.append(
        {
            "role": "system",
            "content": (
                "工具调用轮数已达到上限。请勿再调用或输出工具调用标记；"
                "请直接使用已有工具结果回答用户，并说明资料不足之处。"
            ),
        }
    )
    final_response = llm.invoke(messages)
    return final_response.content
