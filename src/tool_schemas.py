TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": (
                "搜索本地知识库中的相关内容。"
                "当用户询问 Agent、RAG、Embedding、"
                "向量数据库等知识库中的具体知识时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户需要查询的问题",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_knowledge_sources",
            "description": (
                "查看当前知识库包含哪些文档来源。"
                "当用户询问知识库里有哪些资料、"
                "文档或数据来源时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]