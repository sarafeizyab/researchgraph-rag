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
    """Small wrapper around OpenAI chat completions with safe fallbacks."""

    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self._client = None
        if api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

    def complete(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        if self._client is None:
            return LLMResponse(text=self._fallback_text(prompt), latency_ms=0.0, token_usage={})

        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a precise scientific RAG assistant."},
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

    def _fallback_text(self, prompt: str) -> str:
        # Deterministic fallback for environments without API keys.
        if "sub_queries" in prompt:
            return '{"sub_queries": ["' + prompt.split("Question:")[-1].strip().replace('"', "") + '"]}'
        if "sufficient" in prompt:
            return '{"sufficient": true, "follow_up_query": null, "rationale": "fallback"}'
        return "Insufficient LLM configuration. Please set OPENAI_API_KEY for generative outputs."
