import logging

def setup_logger():
    """
    创建项目统一日志器。
    """
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )
    logger = logging.getLogger(
        "rag_project"
    )

    return logger