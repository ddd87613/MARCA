from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Mapping, Optional

from marca_reproduction.schemas import LLMCallRecord, MarcaConfig


class BaseLLMClient(ABC):
    """LLM adapter used by MARCA agents.

    This is the explicit LLM integration point. Replace NoOpLLMClient with
    OpenAICompatibleLLMClient, a LangChain adapter, or a local vLLM wrapper.
    """

    @abstractmethod
    def complete_json(self, role: str, prompt: str, config: MarcaConfig) -> LLMCallRecord:
        raise NotImplementedError


class NoOpLLMClient(BaseLLMClient):
    """Default client: records that LLM reasoning is disabled."""

    def complete_json(self, role: str, prompt: str, config: MarcaConfig) -> LLMCallRecord:
        return LLMCallRecord(
            role=role,
            prompt=prompt,
            response_text="",
            parsed_json=None,
            error="LLM disabled. Deterministic MARCA fallback was used.",
        )


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Minimal OpenAI-compatible client for local vLLM or hosted APIs.

    Expected endpoint:
      POST {base_url}/chat/completions

    For local vLLM, run a server exposing the OpenAI-compatible API and set:
      config.llm_api_base = "http://localhost:8000/v1"
      config.model_name = "your-local-model"

    For hosted OpenAI-compatible providers, set config.llm_api_base and the
    environment variable named by config.llm_api_key_env.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete_json(self, role: str, prompt: str, config: MarcaConfig) -> LLMCallRecord:
        base_url = (self.base_url or config.llm_api_base).rstrip("/")
        api_key = self.api_key or os.environ.get(config.llm_api_key_env, "")
        payload = {
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a MARCA RCA agent. Return only valid JSON "
                        "that follows the schema in the prompt."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return LLMCallRecord(
                role=role,
                prompt=prompt,
                response_text=content,
                parsed_json=_parse_json_object(content),
            )
        except Exception as exc:  # pragma: no cover - depends on external server
            return LLMCallRecord(
                role=role,
                prompt=prompt,
                response_text="",
                parsed_json=None,
                error=f"LLM call failed: {exc}",
            )


def _parse_json_object(text: str) -> Optional[Mapping[str, object]]:
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, Mapping) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            loaded = json.loads(text[start : end + 1])
            return loaded if isinstance(loaded, Mapping) else None
        except json.JSONDecodeError:
            return None
