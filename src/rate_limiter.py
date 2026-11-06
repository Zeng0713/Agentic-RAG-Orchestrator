"""Atomic sliding-window admission across one or several request dimensions."""

import hashlib
import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import redis

from src.state_runtime import StateError

logger = logging.getLogger("rag_project")


@dataclass
class RateDecision:
    allowed: bool
    retry_after: int = 0


class MemoryRateLimiter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._requests = {}
        self._lock = threading.Lock()

    def check(self, limits, window=60):
        with self._lock:
            now = self.clock()
            # Bound cardinality without silently evicting live counters.
            self._requests = {key: values for key, values in self._requests.items()
                              if values and values[-1] > now - window}
            if len(self._requests) + sum(key not in self._requests for key, _ in limits) > 10000:
                raise StateError("限流服务容量已满，请稍后再试")
            retry = 0
            for key, limit in limits:
                recent = self._requests.setdefault(key, deque())
                while recent and recent[0] <= now - window:
                    recent.popleft()
                if len(recent) >= limit:
                    retry = max(retry, math.ceil(recent[0] + window - now))
            if retry:
                logger.info("rate_limit_hit backend=memory retry_after=%d", retry)
                return RateDecision(False, retry)
            for key, _ in limits:
                self._requests[key].append(now)
            return RateDecision(True)

    def allow(self, key, limit, window=60):
        return self.check([(key, limit)], window).allowed

    def ready(self):
        return True


SLIDING_WINDOW = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local window = tonumber(ARGV[1])
local retry = 0
-- Validate all types and limits before making changes (Lua has no rollback).
for i, key in ipairs(KEYS) do
    local kind = redis.call('TYPE', key).ok
    if kind ~= 'none' and kind ~= 'zset' then return {-1, 0} end
    if not tonumber(ARGV[i+2]) or tonumber(ARGV[i+2]) <= 0 then return {-1, 0} end
end
for i, key in ipairs(KEYS) do
    redis.call('ZREMRANGEBYSCORE', key, '-inf', now-window)
    if redis.call('ZCARD', key) >= tonumber(ARGV[i+2]) then
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        retry = math.max(retry, tonumber(oldest[2]) + window - now)
    end
end
if retry > 0 then return {0, math.ceil(retry/1000)} end
for _, key in ipairs(KEYS) do
    redis.call('ZADD', key, now, ARGV[2])
    redis.call('PEXPIRE', key, window)
end
return {1, 0}
"""


class RedisRateLimiter:
    def __init__(self, client, prefix="rag"):
        self.client, self.prefix = client, prefix

    def check(self, limits, window=60):
        keys = [f"{self.prefix}:rate:{{limits}}:{hashlib.sha256(key.encode()).hexdigest()}"
                for key, _ in limits]
        started = time.monotonic()
        try:
            result = self.client.eval(SLIDING_WINDOW, len(keys), *keys,
                                      int(window * 1000), uuid.uuid4().hex,
                                      *(limit for _, limit in limits))
            if result[0] < 0:
                raise StateError("限流存储格式异常")
        except redis.RedisError as exc:
            logger.warning("rate_limit_failed error_type=%s", type(exc).__name__)
            raise StateError() from None
        logger.info("rate_check duration_ms=%.1f allowed=%s",
                    (time.monotonic() - started) * 1000, bool(result[0]))
        return RateDecision(bool(result[0]), int(result[1]))

    def ready(self):
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            raise StateError() from None
