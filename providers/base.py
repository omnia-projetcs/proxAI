"""Interface de base des providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx


class ProviderError(Exception):
    def __init__(self, status_code: int, message: str, detail: Any = None):
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(message)


class BaseProvider(ABC):
    name: str

    def __init__(self, api_key: str | None, base_url: str, client: httpx.AsyncClient):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client

    @abstractmethod
    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    async def chat_completions_stream(
        self, body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        ...

    @abstractmethod
    async def list_models(self) -> dict[str, Any]:
        ...

    async def embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        raise ProviderError(501, f"Embeddings non supportés par {self.name}")

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = await self.client.request(
                method, url, headers=headers, json=json, params=params
            )
        except httpx.RequestError as exc:
            raise ProviderError(502, f"Erreur réseau vers {self.name}: {exc}") from exc

        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ProviderError(response.status_code, f"Erreur {self.name}", detail)
        return response