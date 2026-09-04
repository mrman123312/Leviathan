"""Small OpenAI-compatible inference client used by Leviathan prompt tools.

This module deliberately uses only the Python standard library so a remote or
locally served DeepSeek V4 checkpoint can be prompted without pulling a second
client stack into the core package.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence
from urllib import error, request


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: Sequence[Mapping[str, str]]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": list(self.messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"inference server returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"could not reach inference server at {url}: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"inference server returned invalid JSON: {raw[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("inference server returned a non-object JSON response")
    return parsed


def extract_completion_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("completion choice is not an object")
    text = first.get("text")
    if not isinstance(text, str):
        raise ValueError("completion choice has no text")
    return text


def extract_chat_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("chat choice is not an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("chat choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("chat message has no text content")
    return content


def complete(
    base_url: str,
    request_data: CompletionRequest,
    *,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> tuple[str, dict[str, Any]]:
    endpoint = normalize_base_url(base_url) + "/v1/completions"
    response = _post_json(
        endpoint,
        request_data.as_dict(),
        api_key=api_key,
        timeout=timeout,
    )
    return extract_completion_text(response), response


def chat(
    base_url: str,
    request_data: ChatRequest,
    *,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> tuple[str, dict[str, Any]]:
    endpoint = normalize_base_url(base_url) + "/v1/chat/completions"
    response = _post_json(
        endpoint,
        request_data.as_dict(),
        api_key=api_key,
        timeout=timeout,
    )
    return extract_chat_text(response), response
