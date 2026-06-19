"""Routage transparent des modèles vers le bon provider."""

from __future__ import annotations

import re
from dataclasses import dataclass

from proxai.config import PROVIDER_NAMES, Settings

DEFAULT_MODEL_SENTINELS = frozenset({"", "default", "auto", "proxy-default"})

MODEL_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^gpt-", re.I), "openai"),
    (re.compile(r"^o[134](-mini)?$", re.I), "openai"),
    (re.compile(r"^text-embedding-", re.I), "openai"),
    (re.compile(r"^claude-", re.I), "anthropic"),
    (re.compile(r"^gemini-", re.I), "google"),
    (re.compile(r"^llama-\d", re.I), "groq"),
    (re.compile(r"^gemma2-", re.I), "groq"),
    (re.compile(r"^mistral", re.I), "mistral"),
    (re.compile(r"^mixtral", re.I), "mistral"),
    (re.compile(r"^codestral", re.I), "mistral"),
    (re.compile(r"^(meta|mistralai|nvidia|microsoft)/", re.I), "nvidia"),
    (re.compile(r"^nemotron", re.I), "nvidia"),
    (re.compile(r"^llama", re.I), "ollama"),
    (re.compile(r"^deepseek", re.I), "deepseek"),
    (re.compile(r"^command-", re.I), "cohere"),
    (re.compile(r"^grok-", re.I), "xai"),
    (re.compile(r"^sonar", re.I), "perplexity"),
]


@dataclass(frozen=True)
class RouteTarget:
    provider: str
    model: str


def _parse_model_ref(model: str, settings: Settings) -> RouteTarget:
    raw = model.strip()

    for sep in ("/", ":"):
        if sep in raw:
            provider_part, model_part = raw.split(sep, 1)
            provider = provider_part.lower()
            if provider in PROVIDER_NAMES and model_part:
                return RouteTarget(provider=provider, model=model_part)

    if raw in settings.model_routes:
        provider = settings.model_routes[raw].lower()
        if provider in PROVIDER_NAMES:
            return RouteTarget(provider=provider, model=raw)

    for pattern, provider in MODEL_HINTS:
        if pattern.search(raw):
            return RouteTarget(provider=provider, model=raw)

    default = settings.default_provider.lower()
    if default not in PROVIDER_NAMES:
        default = "ollama"
    return RouteTarget(provider=default, model=raw)


def resolve_route(model: str, settings: Settings) -> RouteTarget:
    """Détermine provider + nom de modèle effectif.

    Si ``model`` est vide ou vaut ``default``/``auto``, utilise ``DEFAULT_MODEL``
    du fichier ``.env``.
    """
    raw = (model or "").strip()

    if raw.lower() in DEFAULT_MODEL_SENTINELS:
        if settings.default_model:
            return _parse_model_ref(settings.default_model, settings)
        default = settings.default_provider.lower()
        if default not in PROVIDER_NAMES:
            default = "ollama"
        return RouteTarget(provider=default, model="default")

    return _parse_model_ref(raw, settings)