"""Configuration via variables d'environnement (.env)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROVIDER_NAMES = frozenset(
    {
        "ollama",
        "vllm",
        "openai",
        "anthropic",
        "google",
        "groq",
        "together",
        "deepseek",
        "mistral",
        "openrouter",
        "azure",
        "cohere",
        "fireworks",
        "perplexity",
        "xai",
        "nvidia",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    host: str = Field(default="0.0.0.0", alias="PROXY_HOST")
    port: int = Field(default=8080, alias="PROXY_PORT")
    workers: int = Field(default=4, alias="WORKERS", ge=1)
    default_provider: str = Field(default="ollama", alias="DEFAULT_PROVIDER")
    default_model: str | None = Field(default=None, alias="DEFAULT_MODEL")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    max_concurrent_requests: int = Field(
        default=200, alias="MAX_CONCURRENT_REQUESTS", ge=1
    )
    http_max_connections: int = Field(default=500, alias="HTTP_MAX_CONNECTIONS", ge=1)
    http_max_keepalive: int = Field(default=100, alias="HTTP_MAX_KEEPALIVE", ge=1)
    model_cache_ttl: int = Field(default=60, alias="MODEL_CACHE_TTL", ge=0)

    enabled_providers: list[str] = Field(
        default_factory=list, alias="ENABLED_PROVIDERS"
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    vllm_base_url: str = Field(default="http://localhost:8000", alias="VLLM_BASE_URL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com", alias="ANTHROPIC_BASE_URL"
    )

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="GOOGLE_BASE_URL",
    )

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL"
    )

    together_api_key: str | None = Field(default=None, alias="TOGETHER_API_KEY")
    together_base_url: str = Field(
        default="https://api.together.xyz/v1", alias="TOGETHER_BASE_URL"
    )

    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL"
    )

    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    mistral_base_url: str = Field(
        default="https://api.mistral.ai/v1", alias="MISTRAL_BASE_URL"
    )

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )

    azure_api_key: str | None = Field(default=None, alias="AZURE_API_KEY")
    azure_base_url: str | None = Field(default=None, alias="AZURE_BASE_URL")
    azure_api_version: str = Field(default="2024-02-15-preview", alias="AZURE_API_VERSION")

    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    cohere_base_url: str = Field(
        default="https://api.cohere.com/compatibility/v1",
        alias="COHERE_BASE_URL",
    )

    fireworks_api_key: str | None = Field(default=None, alias="FIREWORKS_API_KEY")
    fireworks_base_url: str = Field(
        default="https://api.fireworks.ai/inference/v1", alias="FIREWORKS_BASE_URL"
    )

    perplexity_api_key: str | None = Field(default=None, alias="PERPLEXITY_API_KEY")
    perplexity_base_url: str = Field(
        default="https://api.perplexity.ai", alias="PERPLEXITY_BASE_URL"
    )

    xai_api_key: str | None = Field(default=None, alias="XAI_API_KEY")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")

    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )

    # JSON: {"gpt-4o": "openai", "llama3": "ollama", "claude-3-5-sonnet": "anthropic"}
    model_routes: dict[str, str] = Field(default_factory=dict, alias="MODEL_ROUTES")

    @field_validator("default_provider", mode="before")
    @classmethod
    def validate_default_provider(cls, value: Any) -> str:
        provider = str(value).strip().lower()
        if provider not in PROVIDER_NAMES:
            raise ValueError(
                f"DEFAULT_PROVIDER invalide: {value}. Valeurs: {', '.join(sorted(PROVIDER_NAMES))}"
            )
        return provider

    @field_validator("enabled_providers", mode="before")
    @classmethod
    def parse_enabled_providers(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(v).strip().lower() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [p.strip().lower() for p in value.split(",") if p.strip()]
        return []

    @field_validator("model_routes", mode="before")
    @classmethod
    def parse_model_routes(cls, value: Any) -> dict[str, str]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass
            routes: dict[str, str] = {}
            for part in value.split(","):
                part = part.strip()
                if not part or ":" not in part:
                    continue
                model, provider = part.rsplit(":", 1)
                routes[model.strip()] = provider.strip()
            return routes
        return {}

    def resolved_default_route(self):
        from proxai.router import RouteTarget, _parse_model_ref

        if self.default_model:
            return _parse_model_ref(self.default_model, self)
        return RouteTarget(
            provider=self.default_provider.lower(), model="default"
        )

    def provider_credentials(self, provider: str) -> tuple[str | None, str]:
        mapping: dict[str, tuple[str | None, str]] = {
            "ollama": (None, self.ollama_base_url),
            "vllm": (None, self.vllm_base_url),
            "openai": (self.openai_api_key, self.openai_base_url),
            "anthropic": (self.anthropic_api_key, self.anthropic_base_url),
            "google": (self.google_api_key, self.google_base_url),
            "groq": (self.groq_api_key, self.groq_base_url),
            "together": (self.together_api_key, self.together_base_url),
            "deepseek": (self.deepseek_api_key, self.deepseek_base_url),
            "mistral": (self.mistral_api_key, self.mistral_base_url),
            "openrouter": (self.openrouter_api_key, self.openrouter_base_url),
            "azure": (self.azure_api_key, self.azure_base_url or ""),
            "cohere": (self.cohere_api_key, self.cohere_base_url),
            "fireworks": (self.fireworks_api_key, self.fireworks_base_url),
            "perplexity": (self.perplexity_api_key, self.perplexity_base_url),
            "xai": (self.xai_api_key, self.xai_base_url),
            "nvidia": (self.nvidia_api_key, self.nvidia_base_url),
        }
        return mapping.get(provider, (None, ""))


@lru_cache
def get_settings() -> Settings:
    return Settings()