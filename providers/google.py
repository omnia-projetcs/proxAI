"""Adapter Google Gemini (format OpenAI ↔ generateContent)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from proxai.providers.base import BaseProvider, ProviderError


def _openai_to_gemini(body: dict[str, Any]) -> dict[str, Any]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: list[dict[str, Any]] = []

        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})

        contents.append({"role": gemini_role, "parts": parts})

    payload: dict[str, Any] = {"contents": contents}
    generation_config: dict[str, Any] = {}
    if "temperature" in body:
        generation_config["temperature"] = body["temperature"]
    if "top_p" in body:
        generation_config["topP"] = body["top_p"]
    if body.get("max_tokens"):
        generation_config["maxOutputTokens"] = body["max_tokens"]
    if generation_config:
        payload["generationConfig"] = generation_config
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    return payload


def _gemini_to_openai(data: dict[str, Any], model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])

    usage = data.get("usageMetadata", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }


def _stream_chunk(model: str, delta: str, finish_reason: str | None = None) -> bytes:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class GoogleProvider(BaseProvider):
    name = "google"

    def _require_key(self) -> str:
        if not self.api_key:
            raise ProviderError(401, "GOOGLE_API_KEY manquante")
        return self.api_key

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        key = self._require_key()
        model = body["model"]
        payload = _openai_to_gemini(body)
        url = f"{self.base_url}/models/{model}:generateContent"
        response = await self._request(
            "POST",
            url,
            headers={"Content-Type": "application/json"},
            params={"key": key},
            json=payload,
        )
        return _gemini_to_openai(response.json(), model)

    async def chat_completions_stream(
        self, body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        key = self._require_key()
        model = body["model"]
        payload = _openai_to_gemini(body)
        url = f"{self.base_url}/models/{model}:streamGenerateContent"
        params = {"key": key, "alt": "sse"}

        async with self.client.stream(
            "POST",
            url,
            headers={"Content-Type": "application/json"},
            params=params,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                content = await response.aread()
                try:
                    detail = json.loads(content)
                except Exception:
                    detail = content.decode(errors="replace")
                raise ProviderError(response.status_code, "Erreur google", detail)

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            yield _stream_chunk(model, part["text"])
            yield _stream_chunk(model, "", "stop")
            yield b"data: [DONE]\n\n"

    async def list_models(self) -> dict[str, Any]:
        key = self._require_key()
        response = await self._request(
            "GET",
            f"{self.base_url}/models",
            params={"key": key},
        )
        raw = response.json()
        return {
            "object": "list",
            "data": [
                {
                    "id": m.get("name", "").replace("models/", ""),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "google",
                }
                for m in raw.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ],
        }