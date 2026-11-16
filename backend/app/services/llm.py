"""Minimal OpenAI-compatible chat-completions client behind a provider interface."""
import json
from typing import Protocol

import httpx


class LLMProvider(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


class LLMNotConfigured(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self, api_key: str | None, base_url: str, model: str, timeout: int = 120):
        self.api_key, self.base_url, self.model, self.timeout = api_key, base_url, model, timeout

    def complete_json(self, system: str, user: str) -> dict:
        if not self.api_key:
            raise LLMNotConfigured("Configure LLM_API_KEY to generate answers and run LLM verification.")
        payload = {
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise LLMError(
                    "The LLM provider rejected the configured API key. Check LLM_API_KEY and "
                    "confirm it is valid for the selected provider."
                ) from exc
            raise LLMError(
                f"The configured LLM service returned HTTP {exc.response.status_code}. "
                "Check the provider URL, model, and account configuration."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                "The LLM provider timed out before returning a response. Retry or check provider status and limits."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError("Could not connect to the configured LLM service.") from exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError("The LLM returned malformed structured output.") from exc

