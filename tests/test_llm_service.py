import sys
import types

import pytest

from app.services.llm_service import LLMService


def test_openai_service_uses_provider_and_configured_request(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="إجابة"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    service = LLMService(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="sk-test",
        temperature=0.2,
        max_tokens=123,
    )

    assert service.generate("سؤال") == "إجابة"
    assert calls["client"] == {"api_key": "sk-test", "timeout": 30.0, "max_retries": 0}
    assert calls["model"] == "gpt-4.1-mini"
    assert calls["temperature"] == 0.2
    assert calls["max_tokens"] == 123
    assert calls["timeout"] == 30.0


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, LLMService.LLM_AUTHENTICATION_FAILED),
        (402, LLMService.LLM_BILLING_FAILED),
        (404, LLMService.LLM_MODEL_UNAVAILABLE),
    ],
)
def test_openai_provider_errors_are_safe(monkeypatch, status_code, expected):
    class FakeError(Exception):
        def __init__(self):
            super().__init__("provider error")
            self.status_code = status_code

    class FakeCompletions:
        def create(self, **kwargs):
            raise FakeError()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    service = LLMService(provider="openai", model="gpt-4.1-mini", api_key="sk-test")

    assert service.generate("سؤال") == expected


def test_missing_api_key_does_not_claim_success():
    service = LLMService(provider="openai", model="gpt-4.1-mini", api_key="")

    assert service.generate("سؤال") == LLMService.LLM_API_UNAVAILABLE
