"""Security helpers for the public FastAPI service."""

import hmac
import os
from typing import Final

from fastapi import HTTPException, Request, status

PUBLIC_PATHS: Final = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def auth_enabled() -> bool:
    """Return whether API-key authentication is enabled."""
    return _as_bool(os.getenv("API_AUTH_ENABLED"), default=False)


def cors_origins() -> list[str]:
    """Read the comma-separated CORS allowlist."""
    configured = os.getenv("CORS_ORIGINS", "http://localhost:8501")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def validate_api_configuration() -> None:
    """Fail closed when production authentication is enabled incorrectly."""
    if not auth_enabled():
        return

    secret = os.getenv("API_SECRET_KEY", "").strip()
    if not secret or secret.lower().startswith(("your_", "change_")):
        raise RuntimeError(
            "API_AUTH_ENABLED=true requires a real API_SECRET_KEY."
        )


def enforce_api_key(request: Request) -> None:
    """Require the configured API key for non-public routes."""
    if not auth_enabled() or request.url.path in PUBLIC_PATHS:
        return

    configured_key = os.getenv("API_SECRET_KEY", "").strip()
    provided_key = request.headers.get("X-API-Key", "")

    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if not provided_key or not hmac.compare_digest(provided_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
