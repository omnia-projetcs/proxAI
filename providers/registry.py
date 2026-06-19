"""Fabrique des instances de providers."""

from __future__ import annotations

import httpx

from proxai.config import Settings
from proxai.providers.anthropic import AnthropicProvider
from proxai.providers.base import BaseProvider, ProviderError
from proxai.providers.google import GoogleProvider
from proxai.providers.openai_compat import (
    AzureProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    VLLMProvider,
)

_CLIENT: httpx.AsyncClient | None = None

OPENAI_COMPAT_PROVIDERS = frozenset(
    {
        "openai",
        "groq",
        "together",
        "deepseek",
        "mistral",
        "openrouter",
        "fireworks",
        "perplexity",
        "xai",
        "cohere",
        "nvidia",
        "qwen",
    }
)

LOCAL_PROVIDERS = ("ollama", "vllm")

KEYED_PROVIDERS: list[tuple[str, str]] = [
    ("openai", "openai_api_key"),
    ("anthropic", "anthropic_api_key"),
    ("google", "google_api_key"),
    ("groq", "groq_api_key"),
    ("together", "together_api_key"),
    ("deepseek", "deepseek_api_key"),
    ("mistral", "mistral_api_key"),
    ("openrouter", "openrouter_api_key"),
    ("azure", "azure_api_key"),
    ("cohere", "cohere_api_key"),
    ("fireworks", "fireworks_api_key"),
    ("perplexity", "perplexity_api_key"),
    ("xai", "xai_api_key"),
    ("nvidia", "nvidia_api_key"),
    ("qwen", "qwen_api_key"),
]


def get_http_client(settings: Settings) -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive,
            ),
        )
    return _CLIENT


async def close_http_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
        _CLIENT = None


def active_providers(settings: Settings) -> list[str]:
    if settings.enabled_providers:
        known = _all_known_providers(settings)
        return [p for p in settings.enabled_providers if p in known]

    providers: list[str] = []
    if settings.ollama_base_url:
        providers.append("ollama")
    if settings.vllm_base_url:
        providers.append("vllm")

    for name, attr in KEYED_PROVIDERS:
        if getattr(settings, attr, None):
            providers.append(name)
    return providers


def _is_local_backend(base_url: str) -> bool:
    if not base_url:
        return False
    host = base_url.lower()
    return any(token in host for token in ("localhost", "127.0.0.1", "0.0.0.0"))


def _all_known_providers(settings: Settings) -> set[str]:
    known = set()
    if settings.ollama_base_url:
        known.add("ollama")
    if settings.vllm_base_url:
        known.add("vllm")
    for name, attr in KEYED_PROVIDERS:
        if getattr(settings, attr, None):
            known.add(name)
    return known


def get_provider(provider_name: str, settings: Settings) -> BaseProvider:
    name = provider_name.lower()
    api_key, base_url = settings.provider_credentials(name)
    client = get_http_client(settings)

    if name == "ollama":
        if not settings.ollama_base_url:
            raise ProviderError(401, "Ollama n'est pas configuré. Définissez OLLAMA_BASE_URL dans .env")
        return OllamaProvider(None, settings.ollama_base_url, client)
    if name == "vllm":
        if not settings.vllm_base_url:
            raise ProviderError(401, "vLLM n'est pas configuré. Définissez VLLM_BASE_URL dans .env")
        return VLLMProvider(None, settings.vllm_base_url, client)
    if name == "anthropic":
        return AnthropicProvider(api_key, settings.anthropic_base_url, client)
    if name == "google":
        return GoogleProvider(api_key, settings.google_base_url, client)
    if name == "azure":
        if not api_key:
            raise ProviderError(401, "Clé API manquante pour azure. Définissez AZURE_API_KEY dans .env")
        if not base_url:
            raise ProviderError(
                500,
                "AZURE_BASE_URL manquant (ex: https://<resource>.openai.azure.com/openai/deployments/<deployment>)",
            )
        return AzureProvider(
            api_key, base_url, client, api_version=settings.azure_api_version
        )
    if name in OPENAI_COMPAT_PROVIDERS:
        if not api_key and not _is_local_backend(base_url):
            raise ProviderError(
                401,
                f"Clé API manquante pour {name}. Définissez {name.upper()}_API_KEY dans .env",
            )
        provider = OpenAICompatibleProvider(api_key, base_url, client)
        provider.name = name
        if not api_key:
            provider.uses_bearer_auth = False
        return provider

    raise ProviderError(400, f"Provider inconnu: {name}")