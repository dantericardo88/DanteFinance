"""SDS — SENTINEL Data Spine. Adapter registry with tiered fallback chain."""
from __future__ import annotations
from typing import Optional
from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger
from sentinel.sds.base_adapter import BaseAdapter

logger = get_logger(__name__)

_registry: dict[str, BaseAdapter] = {}


def register_adapter(adapter: BaseAdapter) -> None:
    _registry[adapter.name] = adapter
    logger.info("SDS adapter registered", name=adapter.name)


def get_adapter(name: str) -> Optional[BaseAdapter]:
    return _registry.get(name)


def get_all_adapters() -> dict[str, BaseAdapter]:
    return dict(_registry)


def build_default_adapters() -> dict[str, BaseAdapter]:
    """Instantiate all adapters from settings. Returns registry dict."""
    from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
    from sentinel.sds.adapters.fred_adapter import FREDAdapter
    from sentinel.sds.adapters.finnhub_adapter import FinnhubAdapter
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.sds.adapters.ccxt_adapter import CCXTAdapter
    from sentinel.sds.adapters.polygon_adapter import PolygonAdapter
    from sentinel.sds.adapters.alpaca_adapter import AlpacaAdapter

    s = get_settings()
    adapters: list[BaseAdapter] = [
        YFinanceAdapter(),
        FREDAdapter(api_key=s.fred_api_key),
        FinnhubAdapter(api_key=s.finnhub_api_key),
        EDGARAdapter(user_agent=f"SENTINEL {s.edgar_user_agent}"),
        PolygonAdapter(api_key=s.polygon_api_key),
        AlpacaAdapter(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key),
    ]

    # CCXT: add default exchange (Binance) for crypto
    try:
        adapters.append(CCXTAdapter(exchange_id="binance"))
    except Exception:
        pass

    # Congressional trades — always available (no API key required)
    try:
        from sentinel.sds.adapters.congress_adapter import CongressAdapter
        adapters.append(CongressAdapter())
    except Exception:
        pass

    for a in adapters:
        register_adapter(a)

    return _registry


# Tiered fallback chain for OHLCV: yfinance → alpaca → polygon → ccxt:binance
OHLCV_CHAIN: list[str] = ["yfinance", "alpaca", "polygon", "ccxt"]

# Chain for equities fundamentals
FUNDAMENTALS_CHAIN: list[str] = ["finnhub", "yfinance"]

# Chain for macro data
MACRO_CHAIN: list[str] = ["fred"]

# Chain for filing metadata
FILING_CHAIN: list[str] = ["edgar"]
