"""Configuration and safe Redis connections for shared application state."""

import logging
import os
import math
from dataclasses import dataclass
from urllib.parse import urlsplit

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

logger = logging.getLogger("rag_project")


class StateError(Exception):
    status_code = 503
    code = "STATE_UNAVAILABLE"

    def __init__(self, message="会话或限流服务暂时不可用，请稍后重试"):
        super().__init__(message)


class SessionBusy(StateError):
    status_code = 409
    code = "SESSION_BUSY"

    def __init__(self):
        super().__init__("当前会话正在处理问题，请等待完成后再提交")


class LeaseLost(StateError):
    code = "SESSION_LOCK_LOST"

    def __init__(self):
        super().__init__("会话执行权已失效，请重新提问")


@dataclass(frozen=True)
class StateSettings:
    session_backend: str = "memory"
    rate_limit_backend: str = "memory"
    redis_url: str | None = None
    prefix: str = "rag"
    session_ttl: int = 86400
    lease_seconds: float = 60
    renew_seconds: float = 15
    chat_deadline: float = 300

    @classmethod
    def from_env(cls):
        settings = cls(
            session_backend=os.getenv("SESSION_BACKEND", "memory"),
            rate_limit_backend=os.getenv("RATE_LIMIT_BACKEND", "memory"),
            redis_url=os.getenv("REDIS_URL"),
            prefix=os.getenv("REDIS_KEY_PREFIX", "rag"),
            session_ttl=int(os.getenv("SESSION_TTL_SECONDS", "86400")),
            lease_seconds=float(os.getenv("SESSION_LOCK_LEASE_SECONDS", "60")),
            renew_seconds=float(os.getenv("SESSION_LOCK_RENEW_SECONDS", "15")),
            chat_deadline=float(os.getenv("CHAT_DEADLINE_SECONDS", "300")),
        )
        settings.validate()
        return settings

    @property
    def uses_redis(self):
        return "redis" in (self.session_backend, self.rate_limit_backend)

    def validate(self):
        if any(value not in {"memory", "redis"} for value in
               (self.session_backend, self.rate_limit_backend)):
            raise ValueError("状态后端仅支持 memory 或 redis")
        if self.uses_redis:
            try:
                parsed = urlsplit(self.redis_url or "")
                valid = (parsed.scheme in {"redis", "rediss"} and bool(parsed.hostname)
                         and not parsed.query and not parsed.fragment)
                _ = parsed.port
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("Redis 模式必须配置有效的 REDIS_URL")
        if not self.prefix or any(c in self.prefix for c in "{}\r\n"):
            raise ValueError("REDIS_KEY_PREFIX 无效")
        if not all(math.isfinite(n) for n in (self.renew_seconds, self.lease_seconds, self.chat_deadline)):
            raise ValueError("时间配置必须是有限数值")
        if not 0 < self.renew_seconds < self.lease_seconds / 2:
            raise ValueError("续租间隔必须大于 0 且小于租约的一半")
        if self.session_ttl <= 0 or self.chat_deadline <= 0:
            raise ValueError("会话过期时间和请求预算必须大于 0")


def create_redis_client(settings):
    # Never log URL or raw SDK exceptions: either may contain credentials.
    client = redis.Redis.from_url(
        settings.redis_url, decode_responses=True,
        socket_connect_timeout=0.5, socket_timeout=1,
        max_connections=32, health_check_interval=30,
        retry=Retry(NoBackoff(), 0),
    )
    try:
        client.ping()
    except redis.RedisError as exc:
        client.close()
        logger.error("redis_startup_failed error_type=%s", type(exc).__name__)
        raise StateError() from None
    return client
