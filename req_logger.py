"""Journalisation des requêtes (entrées, sorties, consommation de tokens, IP source)."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from proxai.config import Settings

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """Extrait l'adresse IP source du client, gérant les proxys (X-Forwarded-For, X-Real-IP)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Premier élément de la liste d'adresses IP fournies par le proxy
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def estimate_prompt_tokens(messages: list[dict[str, Any]] | str) -> int:
    """Estimation grossière des tokens de prompt si non retournés par l'API (1 token ~ 4 chars)."""
    if isinstance(messages, str):
        return max(1, len(messages) // 4)
    
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
    return max(1, total_chars // 4)


def _write_log_entry(file_path: str, entry: dict[str, Any]) -> None:
    """Écriture synchrone d'une ligne de log au format JSON Lines (JSONL)."""
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Impossible d'écrire la requête dans le fichier de log %s: %s", file_path, exc)


async def async_write_log_entry(settings: Settings, entry: dict[str, Any]) -> None:
    """Écrit une entrée de log de manière asynchrone sans bloquer l'event loop."""
    log_file = settings.log_requests_file or "requests.log"
    await asyncio.to_thread(_write_log_entry, log_file, entry)


def log_chat_request(
    settings: Settings,
    ip: str,
    client_model: str,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    response: dict[str, Any],
) -> None:
    """Journalise une requête de chat-completion standard (non stream)."""
    # Extraction de la réponse texte
    output_text = ""
    choices = response.get("choices", [])
    if choices and len(choices) > 0:
        message = choices[0].get("message", {})
        if message:
            output_text = message.get("content", "")

    # Extraction / estimation de la consommation
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    if prompt_tokens == 0:
        prompt_tokens = estimate_prompt_tokens(messages)
    if completion_tokens == 0:
        completion_tokens = max(1, len(output_text) // 4)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ip": ip,
        "type": "chat",
        "client_model": client_model,
        "provider": provider,
        "model": model,
        "input": messages,
        "output": output_text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    # Lancement de l'écriture en arrière-plan
    asyncio.create_task(async_write_log_entry(settings, entry))


async def wrap_stream_for_logging(
    stream: AsyncIterator[bytes],
    settings: Settings,
    ip: str,
    client_model: str,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
) -> AsyncIterator[bytes]:
    """Intercepte et accumule les chunks d'une réponse streamée pour la journaliser à la fin."""
    accumulated_content: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        async for chunk in stream:
            yield chunk

            # Extraction du texte
            lines = chunk.decode("utf-8", errors="ignore").split("\n")
            for line in lines:
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "content" in delta:
                            accumulated_content.append(delta["content"])

                    # Recherche d'usage si présent dans les chunks
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        total_tokens = usage.get("total_tokens", total_tokens)
                except Exception:
                    pass
    finally:
        full_content = "".join(accumulated_content)

        # Calculer/estimer si non reçus
        if prompt_tokens == 0:
            prompt_tokens = estimate_prompt_tokens(messages)
        if completion_tokens == 0:
            completion_tokens = max(1, len(full_content) // 4)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ip": ip,
            "type": "chat_stream",
            "client_model": client_model,
            "provider": provider,
            "model": model,
            "input": messages,
            "output": full_content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        await async_write_log_entry(settings, entry)


def log_embeddings_request(
    settings: Settings,
    ip: str,
    client_model: str,
    provider: str,
    model: str,
    input_data: str | list[str] | list[int] | list[list[int]],
    response: dict[str, Any],
) -> None:
    """Journalise une requête d'embeddings."""
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    if prompt_tokens == 0:
        if isinstance(input_data, str):
            prompt_tokens = estimate_prompt_tokens(input_data)
        elif isinstance(input_data, list):
            prompt_tokens = 0
            for item in input_data:
                if isinstance(item, str):
                    prompt_tokens += estimate_prompt_tokens(item)
                elif isinstance(item, int):
                    prompt_tokens += 1
                elif isinstance(item, list):
                    prompt_tokens += len(item)
            prompt_tokens = max(1, prompt_tokens)
        total_tokens = prompt_tokens

    # On évite de logger les vecteurs complets pour ne pas surcharger le fichier
    data_list = response.get("data", [])
    embedding_count = len(data_list)
    dimensions = len(data_list[0].get("embedding", [])) if data_list else 0
    output_summary = {
        "embedding_count": embedding_count,
        "dimensions": dimensions,
    }

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ip": ip,
        "type": "embedding",
        "client_model": client_model,
        "provider": provider,
        "model": model,
        "input": input_data,
        "output": output_summary,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": total_tokens,
        },
    }

    asyncio.create_task(async_write_log_entry(settings, entry))
