"""Provider générique OpenAI-compatible (Ollama, vLLM, Groq, etc.)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from proxai.providers.base import BaseProvider, ProviderError


class OpenAICompatibleProvider(BaseProvider):
    """Passe les requêtes telles quelles vers une API OpenAI-compatible."""

    name = "openai_compat"
    uses_bearer_auth: bool = True
    openai_path_prefix: str = "/v1"

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.uses_bearer_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        prefix = self.openai_path_prefix.rstrip("/")
        return f"{self.base_url}{prefix}{path}"

    def _extra_params(self) -> dict[str, str]:
        return {}

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(
            "POST",
            self._url("/chat/completions"),
            headers=self._auth_headers(),
            json=body,
            params=self._extra_params(),
        )
        return response.json()

    async def chat_completions_stream(
        self, body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        body = {**body, "stream": True}
        headers = self._auth_headers()
        headers["Accept"] = "text/event-stream"

        async with self.client.stream(
            "POST",
            self._url("/chat/completions"),
            headers=headers,
            json=body,
            params=self._extra_params(),
        ) as response:
            if response.status_code >= 400:
                content = await response.aread()
                detail: Any
                try:
                    detail = json.loads(content)
                except Exception:
                    detail = content.decode(errors="replace")
                raise ProviderError(
                    response.status_code, f"Erreur {self.name}", detail
                )
            async for chunk in response.aiter_bytes():
                yield chunk

    async def list_models(self) -> dict[str, Any]:
        response = await self._request(
            "GET",
            self._url("/models"),
            headers=self._auth_headers(),
            params=self._extra_params(),
        )
        return response.json()

    async def embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(
            "POST",
            self._url("/embeddings"),
            headers=self._auth_headers(),
            json=body,
            params=self._extra_params(),
        )
        return response.json()


class AzureProvider(OpenAICompatibleProvider):
    name = "azure"
    openai_path_prefix = ""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        client: httpx.AsyncClient,
        *,
        api_version: str,
    ):
        super().__init__(api_key, base_url, client)
        self.api_version = api_version

    def _extra_params(self) -> dict[str, str]:
        return {"api-version": self.api_version}


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"
    uses_bearer_auth = False

    async def list_models(self) -> dict[str, Any]:
        try:
            return await super().list_models()
        except ProviderError:
            response = await self._request(
                "GET",
                f"{self.base_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            raw = response.json()
            return {
                "object": "list",
                "data": [
                    {
                        "id": m.get("name", ""),
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ollama",
                    }
                    for m in raw.get("models", [])
                ],
            }


class VLLMProvider(OpenAICompatibleProvider):
    name = "vllm"
    uses_bearer_auth = False