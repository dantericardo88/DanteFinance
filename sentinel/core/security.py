"""Secret store wrapper — prevents accidental logging of credentials."""
from __future__ import annotations
import re
from sentinel.core.config import get_settings


class _Redacted(str):
    """String subclass that redacts itself in repr/str."""
    def __repr__(self) -> str:
        return "'***REDACTED***'"
    def __str__(self) -> str:
        return "***REDACTED***"


def get_secret(name: str) -> str:
    """
    Fetch a secret from settings. Returns the raw value for use in code.
    Never log the return value of this function.
    """
    settings = get_settings()
    value = getattr(settings, name.lower(), "")
    if not value:
        raise ValueError(
            f"Secret '{name}' is not configured. "
            f"Add it to your .env file (see .env.example)."
        )
    return value


def mask_key(value: str) -> str:
    """Return a masked version safe to log: 'sk-abc...xyz'."""
    if not value or len(value) < 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def scrub_dict(d: dict) -> dict:
    """Return a copy of dict with secret-looking values masked."""
    _SECRET_PATTERN = re.compile(
        r"(key|token|secret|password|credential|auth|api)", re.IGNORECASE
    )
    return {
        k: mask_key(str(v)) if _SECRET_PATTERN.search(k) else v
        for k, v in d.items()
    }


def assert_live_trading_enabled() -> None:
    """Gate that must pass before any live order is submitted."""
    settings = get_settings()
    if not settings.sentinel_live_trading:
        raise PermissionError(
            "Live trading is disabled. "
            "Set SENTINEL_LIVE_TRADING=true in .env to enable it. "
            "This is a safety gate — paper trading is always available."
        )
