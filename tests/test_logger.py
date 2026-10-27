from src.logger import setup_logger


logger = setup_logger()


logger.info(
    "这是一条 INFO 日志"
)

logger.warning(
    "这是一条 WARNING 日志"
)

logger.error(
    "这是一条 ERROR 日志"
)