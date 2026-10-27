import os

from dotenv import load_dotenv


# 加载 .env 文件
load_dotenv()


# ==============================
# SiliconFlow / Embedding 配置
# ==============================

SILICONFLOW_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY"
)

EMBEDDING_MODEL = "BAAI/bge-m3"

EMBEDDING_BASE_URL = (
    "https://api.siliconflow.cn/v1"
)


# ==============================
# DeepSeek / LLM 配置
# ==============================

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)

LLM_MODEL = "deepseek-chat"

LLM_BASE_URL = (
    "https://api.deepseek.com"
)


# ==============================
# Chroma 配置
# ==============================

CHROMA_DB_PATH = "./chroma_db"

CHROMA_COLLECTION_NAME = (
    "agent_knowledge"
)
def validate_config():
    """
    检查项目运行所需的重要配置。
    如果缺少配置，则立即抛出明确异常。
    """

    missing_configs = []

    if not SILICONFLOW_API_KEY:
        missing_configs.append(
            "SILICONFLOW_API_KEY"
        )

    if not DEEPSEEK_API_KEY:
        missing_configs.append(
            "DEEPSEEK_API_KEY"
        )

    if missing_configs:
        missing_text = ", ".join(
            missing_configs
        )

        raise ValueError(
            f"缺少必要配置：{missing_text}。"
            "请检查项目根目录下的 .env 文件。"
        )