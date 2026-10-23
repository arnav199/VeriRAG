"""Minimal OpenAI-compatible chat-completions client behind a provider interface."""
import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMProvider(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


class LLMNotConfigured(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self, api_key: str | None, base_url: str, model: str, timeout: int = 60):
        self.api_key, self.base_url, self.model, self.timeout = api_key, base_url, model, timeout

    def complete_json(self, system: str, user: str) -> dict:
        if not self.api_key:
            raise LLMNotConfigured("Configure LLM_API_KEY to generate answers and run LLM verification.")
        payload = json.dumps({
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions", data=payload, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except (HTTPError, URLError, TimeoutError) as exc:
            raise LLMError("The configured LLM service could not complete the request.") from exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError("The LLM returned malformed structured output.") from exc

