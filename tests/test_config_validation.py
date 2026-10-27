import pytest
import src.config as config
from src.config import validate_config


def test_validate_config():
    validate_config()


def test_missing_deepseek_api_key(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "DEEPSEEK_API_KEY",
        None,
    )

    with pytest.raises(
        ValueError,
        match="DEEPSEEK_API_KEY",
    ):
        validate_config()