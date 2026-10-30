from src.tools import (
    knowledge_search,
    list_knowledge_sources,
)


TOOL_REGISTRY = {
    "knowledge_search": knowledge_search,
    "list_knowledge_sources": list_knowledge_sources,
}


def get_tool(tool_name):
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"未知工具：{tool_name}"
        )

    return TOOL_REGISTRY[tool_name]