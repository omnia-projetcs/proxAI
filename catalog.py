"""Catalogue de modèles — énumération parallèle et cache thread-safe."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from proxai.config import PROVIDER_NAMES, Settings
from proxai.providers import get_provider
from proxai.providers.base import ProviderError
from proxai.providers.registry import active_providers

logger = logging.getLogger(__name__)

_CATALOG: ModelCatalog | None = None


class ModelCatalog:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: dict[str, Any] | None = None
        self._cached_at: float = 0.0
        self._lock = asyncio.Lock()

    async def list_all(self, *, refresh: bool = False) -> dict[str, Any]:
        ttl = self._settings.model_cache_ttl
        if ttl > 0 and not refresh and self._cache is not None:
            if (time.monotonic() - self._cached_at) < ttl:
                return self._cache

        async with self._lock:
            if ttl > 0 and not refresh and self._cache is not None:
                if (time.monotonic() - self._cached_at) < ttl:
                    return self._cache

            providers = active_providers(self._settings)
            results = await asyncio.gather(
                *[self._fetch_provider(name) for name in providers],
                return_exceptions=True,
            )

            all_models: list[dict[str, Any]] = []
            for name, result in zip(providers, results):
                if isinstance(result, Exception):
                    logger.warning("Énumération %s échouée: %s", name, result)
                    continue
                all_models.extend(result)

            payload = self._build_response(all_models)
            self._cache = payload
            self._cached_at = time.monotonic()
            return payload

    async def list_provider(self, provider: str) -> dict[str, Any]:
        name = provider.strip().lower()
        if name not in PROVIDER_NAMES:
            raise ProviderError(400, f"Provider inconnu: {provider}")
        models = await self._fetch_provider(name)
        return self._build_response(models)

    async def get_model(self, model_id: str) -> dict[str, Any] | None:
        catalog = await self.list_all()
        for item in catalog.get("data", []):
            if item.get("id") == model_id:
                return item
        return None

    async def _fetch_provider(self, provider: str) -> list[dict[str, Any]]:
        prov = get_provider(provider, self._settings)
        result = await prov.list_models()
        tagged: list[dict[str, Any]] = []
        for item in result.get("data", []):
            raw_id = item.get("id", "")
            if "/" not in raw_id:
                full_id = f"{provider}/{raw_id}"
            else:
                full_id = raw_id
            tagged.append(
                {
                    **item,
                    "id": full_id,
                    "owned_by": provider,
                    "proxai_provider": provider,
                    "proxai_model": raw_id.split("/", 1)[-1],
                }
            )
        return tagged

    def _build_response(self, models: list[dict[str, Any]]) -> dict[str, Any]:
        data = list(models)

        if self._settings.default_model:
            route = self._settings.resolved_default_route()
            data.insert(
                0,
                {
                    "id": "default",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "proxai",
                    "proxai_provider": route.provider,
                    "proxai_model": route.model,
                    "proxai_target": self._settings.default_model,
                },
            )

        return {"object": "list", "data": data, "proxai_count": len(data)}


def init_catalog(settings: Settings) -> ModelCatalog:
    global _CATALOG
    _CATALOG = ModelCatalog(settings)
    return _CATALOG


def get_catalog() -> ModelCatalog:
    if _CATALOG is None:
        from proxai.config import get_settings

        return init_catalog(get_settings())
    return _CATALOG