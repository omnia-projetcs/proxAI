"""Point d'entrée CLI."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from proxai.config import get_settings


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="proxAI — proxy transparent OpenAI-compatible"
    )
    parser.add_argument("--host", default=None, help="Adresse d'écoute")
    parser.add_argument("--port", type=int, default=None, help="Port d'écoute")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de workers (processus) pour multi-clients",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Rechargement automatique (développement, force 1 worker)",
    )
    args = parser.parse_args()

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    workers = 1 if args.reload else (args.workers or settings.workers)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "proxai.server:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        reload=args.reload,
        log_level=settings.log_level,
        limit_concurrency=settings.max_concurrent_requests,
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    cli()