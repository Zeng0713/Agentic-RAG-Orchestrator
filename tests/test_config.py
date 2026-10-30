import src.config as config


def test_embedding_model():
    assert config.EMBEDDING_MODEL == "BAAI/bge-m3"


def test_embedding_base_url():
    assert (
        config.EMBEDDING_BASE_URL
        == "https://api.siliconflow.cn/v1"
    )


def test_llm_model():
    assert config.LLM_MODEL == "deepseek-chat"


def test_chroma_config():
    assert config.CHROMA_COLLECTION_NAME == "agent_knowledge"