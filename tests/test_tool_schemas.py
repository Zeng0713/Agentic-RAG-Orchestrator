from src.tool_schemas import TOOL_SCHEMAS


def test_tool_schema_names():
    tool_names = [
        tool["function"]["name"]
        for tool in TOOL_SCHEMAS
    ]

    assert "knowledge_search" in tool_names

    assert (
        "list_knowledge_sources"
        in tool_names
    )


def test_knowledge_search_requires_question():
    knowledge_search_schema = (
        TOOL_SCHEMAS[0]["function"]
    )

    required = (
        knowledge_search_schema[
            "parameters"
        ]["required"]
    )

    assert "question" in required