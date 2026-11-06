"""Offline tests must never accidentally spend provider API quota."""

import pytest
from openai.resources.chat.completions import Completions
from openai.resources.embeddings import Embeddings


@pytest.fixture(autouse=True)
def forbid_real_model_calls(request, monkeypatch):
    if request.node.get_closest_marker("integration"):
        return

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Offline tests must mock provider calls")

    monkeypatch.setattr(Completions, "create", forbidden)
    monkeypatch.setattr(Embeddings, "create", forbidden)
