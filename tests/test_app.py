from pathlib import Path
import json
import pytest

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest


def test_streamlit_page_renders_primary_controls(monkeypatch):
    class FakeResponse:
        ok = True

        @staticmethod
        def json():
            return {"success": True, "data": []}

    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: FakeResponse())
    app_path = Path(__file__).parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=15).run()

    assert not app_test.exception
    assert any("Agentic RAG Orchestrator" in element.proto.body for element in app_test.get("html"))
    assert any("上传 Markdown 或 TXT 文档" in element.label for element in app_test.get("file_uploader"))
    assert any("在这里输入你的问题" in element.placeholder for element in app_test.get("chat_input"))
    assert any(button.label.startswith("混合检索是什么") for button in app_test.button)


def test_clicking_suggested_question_uses_existing_chat_flow(monkeypatch):
    requested = []

    class FakeResponse:
        ok = True

        @staticmethod
        def json():
            return {"success": True, "data": []}

    class FakeStreamResponse:
        ok = True

        @staticmethod
        def iter_lines(decode_unicode=True):
            yield 'data: {"type": "token", "text": "混合检索结合关键词与向量检索。"}'
            yield 'data: {"type": "done", "sources": [], "chunks": [], "mode": "agent"}'

        @staticmethod
        def close():
            pass

    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: FakeResponse())

    def fake_post(_url, json=None, **_kwargs):
        requested.append(json["question"])
        return FakeStreamResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    app_path = Path(__file__).parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=15).run()
    prompt_button = next(
        button for button in app_test.button
        if button.label.startswith("混合检索是什么")
    )
    app_test = prompt_button.click().run()

    assert requested == ["混合检索是什么？关键词检索和向量检索如何互补？"]
    assert any("混合检索结合关键词与向量检索" in item.value for item in app_test.markdown)
    assert not any(button.label.startswith("混合检索是什么") for button in app_test.button)


def test_streamlit_chat_displays_answer_sources_and_tool_trace(monkeypatch):
    class FakeResponse:
        ok = True

        def __init__(self, path):
            self.path = path

        def json(self):
            if self.path.endswith("/documents"):
                data = []
            elif self.path.endswith("/chat"):
                data = {
                    "answer": "检索增强生成。",
                    "sources": ["rag_intro.md"],
                    "tool_events": [{
                        "tool": "knowledge_search",
                        "result": [{"source": "rag_intro.md", "content": "RAG 内容"}],
                    }],
                }
            else:
                data = {"cleared": True}
            return {"success": True, "data": data}

    monkeypatch.setattr(
        requests,
        "request",
        lambda method, url, **kwargs: FakeResponse(url),
    )

    class FakeStreamResponse:
        ok = True

        @staticmethod
        def iter_lines(decode_unicode=True):
            yield 'data: {"type": "token", "text": "检索增强生成。"}'
            yield 'data: {"type": "tool", "tool": "knowledge_search", "arguments": {"question": "RAG"}, "result_count": 1, "sources": ["rag_intro.md"]}'
            yield 'data: {"type": "done", "sources": ["rag_intro.md"], "chunks": [{"source": "rag_intro.md", "content": "RAG 原始 chunk"}], "mode": "agent"}'

        @staticmethod
        def close():
            pass

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeStreamResponse())
    app_path = Path(__file__).parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=15).run()
    app_test.chat_input[0].set_value("什么是 RAG？").run()

    assert any("检索增强生成" in element.value for element in app_test.markdown)
    assert not any("RAG 原始 chunk" in element.value for element in app_test.code)
    next(button for button in app_test.button if button.label.startswith("引用来源与检索片段")).click().run()
    assert not app_test.exception
    assert any("RAG 原始 chunk" in element.value for element in app_test.code)
    next(button for button in app_test.button if button.label == "检索与工具调用").click().run()
    assert not app_test.exception
    assert any("知识库检索" in element.value for element in app_test.markdown)
    assert any("rag_intro.md" in element.value for element in app_test.caption)
    # Repeated answers with the same source counts still need distinct action keys.
    app_test.chat_input[0].set_value("再解释一下 RAG").run()
    assert not app_test.exception
    assert len([button for button in app_test.button if button.label.startswith("引用来源与检索片段")]) == 2


def test_streamlit_document_list_is_read_only(monkeypatch):
    class FakeResponse:
        ok = True

        def __init__(self, path):
            self.path = path

        def json(self):
            data = [{
                "document_id": "doc-1",
                "filename": "rag.md",
                "file_type": "md",
                "chunk_count": 2,
                "origin": "上传",
            }]
            if self.path.endswith("/documents/doc-1"):
                data = {
                    "document_id": "doc-1",
                    "filename": "rag.md",
                    "file_type": "md",
                    "content": "## RAG\n可预览的正文",
                }
            return {"success": True, "data": data}

    monkeypatch.setattr(
        requests,
        "request",
        lambda method, url, **kwargs: FakeResponse(url),
    )
    app_path = Path(__file__).parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=15).run()
    assert any(button.label == "rag.md" for button in app_test.button)
    assert not any(button.label == "删除" for button in app_test.button)
    preview_button = next(button for button in app_test.button if button.label == "rag.md")
    preview_button.click().run()

    assert any("可预览的正文" in element.value for element in app_test.code)


def test_streamlit_toggle_selects_traditional_rag_mode(monkeypatch):
    requested_modes = []

    class FakeResponse:
        ok = True

        @staticmethod
        def json():
            return {"success": True, "data": []}

    class FakeStreamResponse:
        ok = True

        @staticmethod
        def iter_lines(decode_unicode=True):
            yield 'data: {"type": "token", "text": "RAG 回答"}'
            yield 'data: {"type": "done", "sources": [], "mode": "rag"}'

        @staticmethod
        def close():
            pass

    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: FakeResponse())

    def fake_post(_url, json=None, **_kwargs):
        requested_modes.append(json["mode"])
        return FakeStreamResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    app_path = Path(__file__).parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=15).run()
    app_test.toggle[0].set_value(False).run()
    app_test.chat_input[0].set_value("什么是 RAG？").run()

    assert requested_modes == ["rag"]
    assert any("RAG 回答" in element.value for element in app_test.markdown)


@pytest.mark.parametrize("complete", [False, True])
def test_streamlit_detects_missing_done_and_displays_save_warning(monkeypatch, complete):
    class Response:
        ok = True

        def json(self):
            return {"success": True, "data": []}

        def iter_lines(self, **_kwargs):
            yield 'data: {"type":"token","text":"answer"}'
            if complete:
                yield "data: " + json.dumps({"type": "done", "sources": [], "chunks": [],
                                             "history_status": "unconfirmed", "warning": "保存状态未确认"})

        def close(self):
            pass

    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: Response())
    app_test = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    app_test.chat_input[0].set_value("q").run()
    assert not app_test.exception
    if complete:
        assert any("保存状态未确认" in item.value for item in app_test.warning)
    else:
        assert any("提前中断" in item.value for item in app_test.error)


@pytest.mark.parametrize("fails", [False, True])
def test_generation_locks_mode_before_stream_and_restores_controls(monkeypatch, fails):
    toggle_states = []
    requests_seen = []
    original_toggle = st.toggle

    def record_toggle(*args, **kwargs):
        toggle_states.append(kwargs.get("disabled", False))
        return original_toggle(*args, **kwargs)

    class Response:
        ok = True

        def json(self):
            return {"success": True, "data": []}

        def iter_lines(self, **_kwargs):
            yield 'data: {"type":"token","text":"回答"}'
            if fails:
                raise requests.ConnectionError("interrupted")
            yield 'data: {"type":"done","sources":[],"chunks":[],"mode":"agent"}'

        def close(self):
            pass

    def post(_url, json, **_kwargs):
        # At the start of network I/O the disabled toggle must already be rendered.
        assert toggle_states[-1] is True
        assert st.session_state.is_generating is True
        requests_seen.append(json["mode"])
        return Response()

    monkeypatch.setattr(st, "toggle", record_toggle)
    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(requests, "post", post)
    page = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    page.chat_input[0].set_value("测试问题").run()
    assert not page.exception
    assert requests_seen == ["agent"]
    assert toggle_states[-1] is False
    assert page.session_state.is_generating is False
    assert page.toggle[0].disabled is False
    page.toggle[0].set_value(False).run()
    assert not page.exception
    assert requests_seen == ["agent"], "Changing mode after completion must not replay the request"
    assert page.toggle[0].value is False


def test_related_question_click_reuses_session_and_chat_pipeline(monkeypatch):
    sent = []

    class Response:
        ok = True

        def json(self):
            return {"success": True, "data": [{"document_id": "doc", "filename": "rag_build_intro.md",
                    "chunk_count": 1, "origin": "内置"}]}

        def iter_lines(self, **_kwargs):
            yield 'data: {"type":"token","text":"RAG 文档上传切分入库流程"}'
            yield 'data: {"type":"done","sources":["rag_build_intro.md"],"chunks":[],"mode":"agent"}'

        def close(self):
            pass

    def post(_url, json, **_kwargs):
        sent.append(json)
        return Response()

    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(requests, "post", post)
    page = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    page.chat_input[0].set_value("RAG 搭建流程").run()
    followup = next(button for button in page.button if (button.key or "").startswith("followup-"))
    prompt = followup.label
    followup.click().run()
    assert not page.exception
    assert len(sent) == 2
    assert sent[1]["question"] == prompt
    assert sent[1]["session_id"] == sent[0]["session_id"]
    assert sent[1]["mode"] == sent[0]["mode"]
    assert prompt not in [button.label for button in page.button if (button.key or "").startswith("followup-")]


def test_library_filter_matches_chinese_topic_and_restores_documents(monkeypatch):
    class Response:
        ok = True

        def json(self):
            return {"success": True, "data": [
                {"document_id": "a", "filename": "rag_intro.md", "chunk_count": 5, "origin": "内置"},
                {"document_id": "b", "filename": "retrieval_quality_guide.md", "chunk_count": 16, "origin": "内置"},
            ]}

    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: Response())
    page = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    page.text_input[0].set_value("质量").run()
    assert not page.exception
    assert any(button.label == "检索质量与评估" for button in page.button)
    assert not any(button.label == "RAG 基础入门" for button in page.button)
    page.text_input[0].set_value("").run()
    assert any(button.label == "RAG 基础入门" for button in page.button)
