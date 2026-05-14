"""Pytest configuration — stub optional heavy dependencies before any test imports.

Packages like yfinance, fredapi, finnhub, ccxt, duckdb, and transformers are
not installed in the CI/test environment. Stubbing them here lets every module
import cleanly so that mock.patch() targets resolve via pkgutil.resolve_name.

After stubbing, sentinel.sds.adapters is explicitly imported to populate
sentinel.sds.adapters.<submodule> attributes (required by pkgutil.resolve_name
which uses getattr, not __import__, to walk dotted patch targets).
"""
import sys
from unittest.mock import MagicMock

# ── Stub all optional packages ────────────────────────────────────────────────

_STUBS = [
    "yfinance",
    "fredapi",
    "finnhub",
    "websockets",
    "ccxt",
    "ccxt.async_support",
    "duckdb",
    "transformers",
    "torch",
    "pandas",
    "numpy",
    "fastmcp",
    "alpaca",
    "alpaca.data",
    "alpaca.data.historical",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
    "alpaca.trading",
    "alpaca.trading.client",
    "alpaca.trading.requests",
    "alpaca.trading.enums",
    "alpaca_trade_api",
    "alpaca_trade_api.rest",
    "polygon",
]

for _pkg in _STUBS:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = MagicMock()

# Specific attributes that are used in isinstance / subclass checks
sys.modules["fredapi"].Fred = MagicMock()
sys.modules["yfinance"].Ticker = MagicMock()
sys.modules["yfinance"].download = MagicMock(return_value=MagicMock())
sys.modules["pandas"].DataFrame = type("DataFrame", (), {})
sys.modules["pandas"].Series = type("Series", (), {})

# alpaca subpackage attributes must be re-exported on parent stubs
sys.modules["alpaca"].data = sys.modules["alpaca.data"]
sys.modules["alpaca.data"].historical = sys.modules["alpaca.data.historical"]
sys.modules["alpaca.data"].requests = sys.modules["alpaca.data.requests"]
sys.modules["alpaca.data"].timeframe = sys.modules["alpaca.data.timeframe"]

# ── Force-import the adapters package so submodule attributes are populated ──
# This populates sentinel.sds.adapters.fred_adapter etc. which pkgutil needs.
import sentinel.sds.adapters  # noqa: E402, F401
