import pytest

from agent.llm import LLMClient


def test_hugging_face_chat_response_parser() -> None:
    client = LLMClient(provider="huggingface", openai_model="unused")

    text = client._extract_text_from_hf_response(
        {"choices": [{"message": {"content": "grounded answer"}}]}
    )

    assert text == "grounded answer"


def test_hugging_face_generation_response_parser() -> None:
    client = LLMClient(provider="huggingface", openai_model="unused")

    text = client._extract_text_from_hf_response([{"generated_text": "generated answer"}])

    assert text == "generated answer"


def test_hugging_face_missing_endpoint_fallback() -> None:
    client = LLMClient(provider="huggingface", openai_model="unused")

    response = client.complete("question")

    assert "HF_ENDPOINT_URL" in response.text


def test_hugging_face_chat_completions_url() -> None:
    client = LLMClient(provider="huggingface", openai_model="unused")

    url = client._chat_completions_url("https://example.endpoints.huggingface.cloud")

    assert url == "https://example.endpoints.huggingface.cloud/v1/chat/completions"


def test_hugging_face_text_generation_url() -> None:
    client = LLMClient(provider="huggingface", openai_model="unused")

    url = client._text_generation_url("https://example.endpoints.huggingface.cloud")

    assert url == "https://example.endpoints.huggingface.cloud/generate"


def test_hugging_face_placeholder_endpoint_raises() -> None:
    client = LLMClient(
        provider="huggingface",
        openai_model="unused",
        hf_endpoint_url="https://your-endpoint.endpoints.huggingface.cloud",
    )

    with pytest.raises(RuntimeError, match="placeholder"):
        client.complete("question")


def test_openai_missing_key_fallback() -> None:
    client = LLMClient(provider="openai", openai_model="gpt-4.1-mini")

    response = client.complete("question")

    assert "supported LLM provider" in response.text
