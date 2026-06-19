"""Serveur FastAPI — API OpenAI-compatible transparente."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from proxai.catalog import get_catalog, init_catalog
from proxai.concurrency import (
    acquire_request_slot,
    active_request_count,
    init_concurrency,
    wrap_stream_with_slot,
)
from proxai.config import get_settings
from proxai.providers import get_provider
from proxai.providers.base import ProviderError
from proxai.providers.registry import close_http_client
from proxai.router import resolve_route

logger = logging.getLogger(__name__)


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[dict[str, Any]]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class EmbeddingRequest(BaseModel):
    model: str = "default"
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None

    model_config = {"extra": "allow"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_concurrency(settings)
    catalog = init_catalog(settings)

    async def _warm_cache():
        try:
            await catalog.list_all()
            logger.info("Catalogue de modèles initialisé")
        except Exception as exc:
            logger.warning("Pré-chargement du catalogue échoué: %s", exc)

    asyncio.create_task(_warm_cache())
    yield
    await close_http_client()


def create_app() -> FastAPI:
    app = FastAPI(
        title="proxAI",
        description="Proxy transparent OpenAI-compatible pour Ollama, vLLM et providers cloud",
        version="0.2.1",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ProviderError)
    async def provider_error_handler(_request: Request, exc: ProviderError):
        content: dict[str, Any] = {
            "error": {
                "message": exc.message,
                "type": "provider_error",
                "code": exc.status_code,
            }
        }
        if exc.detail is not None:
            content["error"]["detail"] = exc.detail
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("Erreur non gérée sur %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Erreur interne du proxy",
                    "type": "internal_error",
                    "code": 500,
                }
            },
        )

    @app.get("/health")
    async def health():
        settings = get_settings()
        route = settings.resolved_default_route()
        return {
            "status": "ok",
            "service": "proxai",
            "active_requests": await active_request_count(),
            "max_concurrent_requests": settings.max_concurrent_requests,
            "workers": settings.workers,
            "default_model": settings.default_model or "default",
            "default_route": {
                "provider": route.provider,
                "model": route.model,
            },
        }

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models(provider: str | None = None, refresh: bool = False):
        catalog = get_catalog()
        if provider:
            return await catalog.list_provider(provider)
        return await catalog.list_all(refresh=refresh)

    @app.get("/v1/models/{model_id:path}")
    @app.get("/models/{model_id:path}")
    async def get_model(model_id: str):
        catalog = get_catalog()
        model = await catalog.get_model(model_id)
        if model is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Modèle introuvable: {model_id}",
                        "type": "not_found",
                    }
                },
            )
        return model

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        settings = get_settings()
        route = resolve_route(request.model, settings)
        provider = get_provider(route.provider, settings)
        body = request.model_dump(exclude_none=True)
        body["model"] = route.model

        logger.info(
            "chat client_model=%s -> provider=%s backend_model=%s stream=%s",
            request.model,
            route.provider,
            route.model,
            request.stream,
        )

        headers = {
            "X-ProxAI-Provider": route.provider,
            "X-ProxAI-Model": route.model,
        }

        if request.stream:
            stream = wrap_stream_with_slot(provider.chat_completions_stream(body))
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={
                    **headers,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        async with acquire_request_slot():
            result = await provider.chat_completions(body)
            return JSONResponse(content=result, headers=headers)

    @app.post("/v1/embeddings")
    @app.post("/embeddings")
    async def embeddings(request: EmbeddingRequest):
        settings = get_settings()
        route = resolve_route(request.model, settings)

        async with acquire_request_slot():
            provider = get_provider(route.provider, settings)
            body = request.model_dump(exclude_none=True)
            body["model"] = route.model

            result = await provider.embeddings(body)
            return JSONResponse(
                content=result,
                headers={
                    "X-ProxAI-Provider": route.provider,
                    "X-ProxAI-Model": route.model,
                },
            )

    return app