from src.config import (
    SILICONFLOW_API_KEY,
    DEEPSEEK_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    LLM_MODEL,
    LLM_BASE_URL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
)


def test_api_keys_exist():
    assert SILICONFLOW_API_KEY
    assert DEEPSEEK_API_KEY


def test_embedding_config():
    assert EMBEDDING_MODEL == "BAAI/bge-m3"
    assert (
        EMBEDDING_BASE_URL
        == "https://api.siliconflow.cn/v1"
    )


def test_llm_config():
    assert LLM_MODEL == "deepseek-chat"
    assert (
        LLM_BASE_URL
        == "https://api.deepseek.com"
    )


def test_chroma_config():
    assert CHROMA_DB_PATH == "./chroma_db"
    assert (
        CHROMA_COLLECTION_NAME
        == "agent_knowledge"
    )