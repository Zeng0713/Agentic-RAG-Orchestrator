import pytest

from src.tool_registry import get_tool
from src.tools import (
    knowledge_search,
    list_knowledge_sources,
)


def test_get_knowledge_search_tool():
    tool = get_tool("knowledge_search")

    assert tool is knowledge_search


def test_get_list_knowledge_sources_tool():
    tool = get_tool(
        "list_knowledge_sources"
    )

    assert tool is list_knowledge_sources


def test_unknown_tool():
    with pytest.raises(
        ValueError,
        match="未知工具",
    ):
        get_tool("unknown_tool")