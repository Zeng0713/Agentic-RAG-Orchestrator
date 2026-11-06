"""Exercise both public CLI entries without touching models or production data."""

from pathlib import Path
import runpy

from langchain_core.documents import Document
import pytest


@pytest.mark.parametrize("entry", ["rag_qa.py", "agent_qa.py"])
def test_cli_accepts_question_and_exits_offline(entry, monkeypatch, capsys):
    import src.agent as agent
    import src.config as config
    import src.document_loader as loader
    import src.embeddings as embeddings
    import src.rag_chain as rag
    import src.retriever as retriever
    import src.vector_store as vector

    docs = [Document(page_content="## RAG\nOffline knowledge", metadata={"source": "rag.md"})]
    answers = iter(["What is RAG?", "exit"])
    indexed = []
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))
    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(loader, "load_documents", lambda: docs)
    monkeypatch.setattr(embeddings, "create_embeddings", lambda: object())
    monkeypatch.setattr(vector, "create_vector_store", lambda *_args: object())
    monkeypatch.setattr(vector, "sync_documents", lambda store, chunks: indexed.extend(chunks))
    monkeypatch.setattr(rag, "create_llm", lambda: object())
    monkeypatch.setattr(retriever, "retrieve_documents", lambda *_args: docs)
    monkeypatch.setattr(rag, "generate_answer", lambda *_args: "Offline answer [资料 1]")
    monkeypatch.setattr(agent, "run_agent", lambda **_kwargs: "Offline answer")
    runpy.run_path(str(Path(__file__).parents[1] / entry), run_name="__main__")
    output = capsys.readouterr().out
    assert "Offline answer" in output
    assert "程序已退出" in output
    assert indexed == docs
