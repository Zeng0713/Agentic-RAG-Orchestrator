from pathlib import Path

import requests
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
    assert any("Agentic RAG Orchestrator" in element.value for element in app_test.markdown)
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
    assert any("rag_intro.md" in element.value for element in app_test.caption)
    assert any("知识库检索" in element.value for element in app_test.markdown)
    assert any("RAG 原始 chunk" in element.value for element in app_test.code)


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
    assert any(button.label.startswith("📄 rag.md") for button in app_test.button)
    assert not any(button.label == "删除" for button in app_test.button)
    preview_button = next(button for button in app_test.button if button.label.startswith("📄 rag.md"))
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
