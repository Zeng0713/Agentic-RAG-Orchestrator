"""Bound context and provider calls without assuming an OpenAI tokenizer."""

import json
import logging
import time

from src.exceptions import LLMServiceError
from src.state_runtime import StateError

INPUT_BUDGET = 8000
logger = logging.getLogger("rag_project")


def size(value):
    """UTF-8 byte count is a conservative engineering estimate, not exact tokens."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return len(value.encode("utf-8"))


def clip(text, budget):
    return str(text).encode("utf-8")[:max(0, budget)].decode("utf-8", errors="ignore")


def trim_history(history, budget=2000):
    turns = [[dict(message) for message in history[i:i + 2]]
             for i in range(0, len(history or []), 2)][-6:]
    while len(turns) > 1 and size([message for turn in turns for message in turn]) > budget:
        turns.pop(0)
    result = [message for turn in turns for message in turn]
    while result and size(result) > budget:
        longest = max(result, key=lambda message: size(message.get("content", "")))
        content = longest.get("content", "")
        if not content:
            return []
        longest["content"] = clip(content, max(0, size(content) - (size(result) - budget)))
    return result


class BudgetedLLM:
    """Per-request wrapper: no mutation of the shared model client."""

    def __init__(self, llm, check, deadline):
        self.llm, self.check, self.deadline = llm, check, deadline

    def _kwargs(self, kwargs):
        self.check()
        return {**kwargs, "timeout": min(kwargs.get("timeout", 60), 60, max(0.1, self.deadline - time.monotonic())),
                "max_tokens": kwargs.get("max_tokens", 2048)}

    def invoke(self, messages, **kwargs):
        options = self._kwargs(kwargs)
        try:
            response = self.llm.invoke(messages, **options)
        except StateError:
            raise
        except Exception as exc:
            logger.warning("model_invoke_failed error_type=%s", type(exc).__name__)
            raise LLMServiceError("大语言模型服务暂时不可用") from None
        self.check()
        return response

    def stream(self, messages, **kwargs):
        if not hasattr(self.llm, "stream"):
            yield self.invoke(messages, **kwargs)
            return
        options = self._kwargs(kwargs)
        stream = None
        try:
            stream = self.llm.stream(messages, **options)
            for chunk in stream:
                self.check()
                yield chunk
        except StateError:
            raise
        except Exception as exc:
            logger.warning("model_stream_failed error_type=%s", type(exc).__name__)
            raise LLMServiceError("大语言模型服务暂时不可用") from None
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()


def rewrite_question(question, history, llm):
    if not history:
        return question
    messages = [
        {"role": "system", "content": (
            "根据聊天历史，将用户本次问题改写成可独立检索的完整问题。"
            "只消解指代、补充已有上下文，不回答、不添加事实。仅输出改写后的问题。"
        )},
        *trim_history(history),
        {"role": "user", "content": question},
    ]
    try:
        response = llm.invoke(messages, timeout=10, max_tokens=256, temperature=0)
        result = response.content
        if not isinstance(result, str) or not result.strip():
            raise ValueError("empty rewrite")
        return clip(result.strip(), 2000)
    except StateError:
        raise
    except Exception:
        raise LLMServiceError("追问理解服务暂时不可用，请稍后重试") from None
