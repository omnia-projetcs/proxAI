"""Adapter Anthropic (format OpenAI ↔ Messages API)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from proxai.providers.base import BaseProvider, ProviderError


def _openai_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages", [])
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
            continue

        if role == "assistant":
            anthropic_role = "assistant"
        else:
            anthropic_role = "user"

        if isinstance(content, str):
            converted.append({"role": anthropic_role, "content": content})
        elif isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media = header.split(";")[0].split(":")[1]
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media,
                                    "data": data,
                                },
                            }
                        )
            converted.append({"role": anthropic_role, "content": blocks or ""})

    payload: dict[str, Any] = {
        "model": body["model"],
        "messages": converted,
        "max_tokens": body.get("max_tokens", 4096),
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    if "top_p" in body:
        payload["top_p"] = body["top_p"]
    if body.get("stop"):
        payload["stop_sequences"] = (
            body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
        )
    return payload


def _anthropic_to_openai(data: dict[str, Any], model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    usage = data.get("usage", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(text_parts),
                },
                "finish_reason": _map_stop_reason(data.get("stop_reason")),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        },
    }


def _map_stop_reason(reason: str | None) -> str:
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    return mapping.get(reason or "", "stop")


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


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError(401, "ANTHROPIC_API_KEY manquante")
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = _openai_to_anthropic(body)
        response = await self._request(
            "POST",
            f"{self.base_url}/v1/messages",
            headers=self._headers(),
            json=payload,
        )
        return _anthropic_to_openai(response.json(), body["model"])

    async def chat_completions_stream(
        self, body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        payload = {**_openai_to_anthropic(body), "stream": True}
        headers = self._headers()
        headers["Accept"] = "text/event-stream"

        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                content = await response.aread()
                try:
                    detail = json.loads(content)
                except Exception:
                    detail = content.decode(errors="replace")
                raise ProviderError(response.status_code, "Erreur anthropic", detail)

            event_type = ""
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield _stream_chunk(body["model"], delta.get("text", ""))
                elif event_type == "message_stop":
                    yield _stream_chunk(body["model"], "", "stop")
                    yield b"data: [DONE]\n\n"

    async def list_models(self) -> dict[str, Any]:
        models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "anthropic",
                }
                for m in models
            ],
        }