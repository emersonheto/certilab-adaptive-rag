from __future__ import annotations
from unittest.mock import MagicMock
import pytest

@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.openai_api_key = "test-key"
    s.openai_chat_model = "gpt-4o-mini"
    s.openai_temperature = 0.0
    s.default_top_k = 4
    return s
