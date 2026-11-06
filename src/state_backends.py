"""Dependency factories; instantiated once per API process, not per request."""

from src.rate_limiter import MemoryRateLimiter, RedisRateLimiter
from src.session_store import MemorySessionStore, RedisSessionStore
from src.state_runtime import StateSettings, create_redis_client


class StateBackends:
    def __init__(self, settings=None, client=None):
        self.settings = settings or StateSettings.from_env()
        self.settings.validate()
        self.client = client
        if self.settings.uses_redis and client is None:
            self.client = create_redis_client(self.settings)
        self.sessions = (RedisSessionStore(self.client, self.settings)
                         if self.settings.session_backend == "redis"
                         else MemorySessionStore(ttl=self.settings.session_ttl))
        self.limiter = (RedisRateLimiter(self.client, self.settings.prefix)
                        if self.settings.rate_limit_backend == "redis" else MemoryRateLimiter())

    def ready(self):
        return self.sessions.ready() and self.limiter.ready()

    def close(self):
        if self.client:
            self.client.close()
