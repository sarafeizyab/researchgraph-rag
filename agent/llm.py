from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM response payload with telemetry."""

    text: str
    latency_ms: float
    token_usage: dict[str, int]


class LLMClient:
    """Provider-aware chat completion client for scientific answer synthesis."""

    system_prompt = "You are a precise scientific RAG assistant."

    def __init__(
        self,
        provider: str,
        openai_model: str,
        openai_api_key: str | None = None,
        hf_token: str | None = None,
        hf_endpoint_url: str | None = None,
        hf_endpoint_mode: str = "text-generation",
        hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
        hf_max_new_tokens: int = 768,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b",
    ) -> None:
        self.provider = provider
        self.openai_model = openai_model
        self.hf_token = hf_token
        self.hf_endpoint_url = hf_endpoint_url
        self.hf_endpoint_mode = hf_endpoint_mode
        self.hf_model = hf_model
        self.hf_max_new_tokens = hf_max_new_tokens
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self._client = None

        if provider == "openai" and openai_api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=openai_api_key)

    def complete(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        if self.provider == "openai":
            return self._complete_openai(prompt=prompt, temperature=temperature)

        if self.provider == "huggingface":
            return self._complete_huggingface(prompt=prompt, temperature=temperature)

        if self.provider == "ollama":
            return self._complete_ollama(prompt=prompt, temperature=temperature)

        LOGGER.warning("Unknown LLM_PROVIDER=%s; using deterministic fallback", self.provider)
        return LLMResponse(text=self._fallback_text(prompt), latency_ms=0.0, token_usage={})

    def _complete_openai(self, prompt: str, temperature: float) -> LLMResponse:
        if self._client is None:
            return LLMResponse(text=self._fallback_text(prompt), latency_ms=0.0, token_usage={})

        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.openai_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        }
        return LLMResponse(text=text, latency_ms=latency_ms, token_usage=usage)

    def _complete_huggingface(self, prompt: str, temperature: float) -> LLMResponse:
        if not self.hf_endpoint_url:
            return LLMResponse(
                text="Insufficient Hugging Face configuration. Please set HF_ENDPOINT_URL.",
                latency_ms=0.0,
                token_usage={},
            )

        if "your-endpoint" in self.hf_endpoint_url:
            raise RuntimeError("Hugging Face endpoint request failed: HF_ENDPOINT_URL is still a placeholder.")

        import httpx

        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0) as client:
                if self.hf_endpoint_mode == "chat-completions":
                    url = self._chat_completions_url(self.hf_endpoint_url)
                    payload: dict[str, Any] = {
                        "model": self.hf_model,
                        "temperature": temperature,
                        "max_tokens": self.hf_max_new_tokens,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    }
                else:
                    url = self._text_generation_url(self.hf_endpoint_url)
                    payload = {
                        "inputs": self._prompt_for_text_generation(prompt),
                        "parameters": {
                            "temperature": max(temperature, 0.01),
                            "max_new_tokens": self.hf_max_new_tokens,
                            "return_full_text": False,
                        },
                    }

                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Hugging Face endpoint request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        data = resp.json()
        return LLMResponse(
            text=self._extract_text_from_hf_response(data),
            latency_ms=latency_ms,
            token_usage=self._extract_usage(data),
        )

    def _complete_ollama(self, prompt: str, temperature: float) -> LLMResponse:
        import httpx

        payload = {
            "model": self.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature},
        }

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        data = resp.json()
        message = data.get("message", {})
        text = str(message.get("content") or data.get("response") or "")
        usage = {
            "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "completion_tokens": int(data.get("eval_count", 0) or 0),
            "total_tokens": int(data.get("prompt_eval_count", 0) or 0)
            + int(data.get("eval_count", 0) or 0),
        }
        return LLMResponse(text=text, latency_ms=latency_ms, token_usage=usage)

    def complete_json(self, prompt: str, temperature: float = 0.0) -> tuple[dict[str, Any], LLMResponse]:
        response = self.complete(prompt=prompt, temperature=temperature)
        parsed = self._parse_json(response.text)
        return parsed, response

    def _parse_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            LOGGER.warning("Could not parse JSON from LLM output")
            return {}

    def _prompt_for_text_generation(self, prompt: str) -> str:
        return f"System: {self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    def _chat_completions_url(self, endpoint_url: str) -> str:
        normalized = endpoint_url.rstrip("/")
        if normalized.endswith("/v1/chat/completions"):
            return normalized
        return f"{normalized}/v1/chat/completions"

    def _text_generation_url(self, endpoint_url: str) -> str:
        normalized = endpoint_url.rstrip("/")
        if normalized.endswith("/generate"):
            return normalized
        return f"{normalized}/generate"

    def _extract_text_from_hf_response(self, data: Any) -> str:
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict) and message.get("content"):
                        return str(message["content"])
                    if first.get("text"):
                        return str(first["text"])

            if data.get("generated_text"):
                return str(data["generated_text"])

            if data.get("error"):
                raise RuntimeError(str(data["error"]))

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and first.get("generated_text"):
                return str(first["generated_text"])
            if isinstance(first, str):
                return first

        return str(data)

    def _extract_usage(self, data: Any) -> dict[str, int]:
        if not isinstance(data, dict):
            return {}

        usage = data.get("usage")
        if not isinstance(usage, dict):
            return {}

        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    def _fallback_text(self, prompt: str) -> str:
        # Deterministic fallback for environments without API keys.
        if "sub_queries" in prompt:
            return '{"sub_queries": ["' + prompt.split("Question:")[-1].strip().replace('"', "") + '"]}'
        if "sufficient" in prompt:
            return '{"sufficient": true, "follow_up_query": null, "rationale": "fallback"}'
        return "Insufficient LLM configuration. Please configure a supported LLM provider."
