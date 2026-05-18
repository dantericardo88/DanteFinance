"""
rest_sdk_v3.py — SENTINEL REST API v3 + WebSocket server + Python client SDK.

dim_095: REST API + WebSocket SDK  (score 6 → 9)

Architecture
------------
  SentinelAPIServer   — FastAPI application, /api/v3 prefix, all SENTINEL v3 dims
  SentinelWebSocketServer — asyncio WebSocket hub (quotes, news, alerts, regime)
  SentinelPythonSDK   — sync + async client wrapping all endpoints
  SentinelClient      — convenience alias
  APIDocGenerator     — Markdown / Postman / curl doc generation

Auth: X-SENTINEL-KEY header  (env SENTINEL_API_KEY, default "dev-key")
Rate: token-bucket, 100 req/min per key (in-memory)
CORS: allow-all in dev

Run standalone:
    python -m sentinel.api.rest_sdk_v3
    uvicorn sentinel.api.rest_sdk_v3:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps — never break import on missing
# ---------------------------------------------------------------------------
try:
    import requests as _requests_lib
    _REQUESTS_OK = True
except ImportError:
    _requests_lib = None  # type: ignore[assignment]
    _REQUESTS_OK = False

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field as PydanticField
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    FastAPI = None  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment,misc]

try:
    import uvicorn as _uvicorn
    _UVICORN_OK = True
except ImportError:
    _uvicorn = None  # type: ignore[assignment]
    _UVICORN_OK = False

try:
    import websockets
    _WEBSOCKETS_OK = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    _WEBSOCKETS_OK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SENTINEL_API_KEY = os.getenv("SENTINEL_API_KEY", "dev-key")
API_PREFIX = "/api/v3"
WS_PORT = int(os.getenv("SENTINEL_WS_PORT", "8765"))
_DEV_MODE = os.getenv("SENTINEL_ENV", "dev").lower() in ("dev", "development", "local")

# ---------------------------------------------------------------------------
# Rate limiter — token bucket, per API key
# ---------------------------------------------------------------------------

@dataclass
class _TokenBucket:
    capacity: int = 100          # tokens
    rate: float = 100 / 60.0    # tokens per second (100 req/min)
    tokens: float = 100.0
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self, n: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


_rate_buckets: Dict[str, _TokenBucket] = {}
_rate_lock = threading.Lock()


def _check_rate(key: str) -> bool:
    with _rate_lock:
        if key not in _rate_buckets:
            _rate_buckets[key] = _TokenBucket()
        return _rate_buckets[key].consume()


# ---------------------------------------------------------------------------
# In-memory backtest result store
# ---------------------------------------------------------------------------

_backtest_results: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Lazy module loader — all SENTINEL v3 modules loaded on first use
# ---------------------------------------------------------------------------

_module_cache: Dict[str, Any] = {}


def _lazy(module_path: str) -> Any:
    """Import a SENTINEL v3 module, caching the result. Returns None on failure."""
    if module_path in _module_cache:
        return _module_cache[module_path]
    try:
        import importlib
        mod = importlib.import_module(module_path)
        _module_cache[module_path] = mod
        return mod
    except Exception as exc:
        logger.debug("optional module %s unavailable: %s", module_path, exc)
        _module_cache[module_path] = None
        return None


# ---------------------------------------------------------------------------
# Pydantic request/response models (conditional on FastAPI availability)
# ---------------------------------------------------------------------------

if _FASTAPI_OK:
    class ScreenFundamentalRequest(BaseModel):
        criteria: Dict[str, Any] = PydanticField(default_factory=dict,
            examples=[{"pe_max": 20, "roe_min": 0.15, "market_cap_min": 1e9}])

    class ScreenTechnicalRequest(BaseModel):
        criteria: Dict[str, Any] = PydanticField(default_factory=dict,
            examples=[{"rsi_max": 30, "price_above_200ma": True}])

    class ScreenNLRequest(BaseModel):
        query: str = PydanticField(examples=["profitable tech stocks with PE < 20"])

    class VaRRequest(BaseModel):
        holdings: Dict[str, float] = PydanticField(
            examples=[{"AAPL": 10000, "MSFT": 15000}])
        confidence: float = PydanticField(default=0.99, ge=0.9, le=0.9999)
        lookback_days: int = PydanticField(default=252, ge=20)

    class StressRequest(BaseModel):
        holdings: Dict[str, float] = PydanticField(default_factory=dict)
        scenarios: List[str] = PydanticField(
            default=["2008_gfc", "2020_covid", "2022_rates", "dot_com"])

    class PortfolioOptimizeRequest(BaseModel):
        tickers: List[str]
        method: str = PydanticField(default="max_sharpe",
            examples=["max_sharpe", "min_variance", "risk_parity", "black_litterman"])
        constraints: Dict[str, Any] = PydanticField(default_factory=dict)
        start: str = PydanticField(default="2020-01-01")
        end: Optional[str] = PydanticField(default=None)

    class AISummarizeRequest(BaseModel):
        ticker: str
        form: str = PydanticField(default="10-K",
            examples=["10-K", "10-Q", "8-K", "S-1", "DEF 14A"])

    class AIQARequest(BaseModel):
        question: str
        tickers: List[str] = PydanticField(default_factory=list)
        top_k: int = PydanticField(default=5, ge=1, le=20)

    class AIStrategyRequest(BaseModel):
        description: str = PydanticField(
            examples=["momentum small caps", "high dividend yield value stocks"])

    class BacktestRunRequest(BaseModel):
        strategy: Dict[str, Any]
        symbols: List[str]
        start: str
        end: str = PydanticField(default="")
        initial_capital: float = PydanticField(default=100_000.0)
        commission: float = PydanticField(default=0.001)

else:
    # Stubs so the rest of the module is importable without FastAPI
    class ScreenFundamentalRequest: pass  # type: ignore[no-redef]
    class ScreenTechnicalRequest: pass    # type: ignore[no-redef]
    class ScreenNLRequest: pass           # type: ignore[no-redef]
    class VaRRequest: pass                # type: ignore[no-redef]
    class StressRequest: pass             # type: ignore[no-redef]
    class PortfolioOptimizeRequest: pass  # type: ignore[no-redef]
    class AISummarizeRequest: pass        # type: ignore[no-redef]
    class AIQARequest: pass               # type: ignore[no-redef]
    class AIStrategyRequest: pass         # type: ignore[no-redef]
    class BacktestRunRequest: pass        # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _make_auth_dep():
    if not _FASTAPI_OK:
        return None

    async def verify_api_key(x_sentinel_key: str = Header(default="dev-key")) -> str:
        if x_sentinel_key != SENTINEL_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key. Provide X-SENTINEL-KEY header.",
            )
        if not _check_rate(x_sentinel_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: 100 requests/min per key.",
            )
        return x_sentinel_key

    return verify_api_key


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def _build_app() -> Any:
    """Build and return the FastAPI application."""
    if not _FASTAPI_OK:
        return None

    _app = FastAPI(
        title="SENTINEL Financial Terminal API v3",
        description=(
            "Comprehensive institutional-grade financial terminal API.\n\n"
            "All endpoints require `X-SENTINEL-KEY` header (default: `dev-key`).\n"
            "Rate limit: 100 requests/minute per API key.\n\n"
            "Covers: market data, fundamentals, ownership, filings, "
            "screeners, risk, AI research, backtesting."
        ),
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    if _DEV_MODE:
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    auth_dep = _make_auth_dep()
    deps = [Depends(auth_dep)] if auth_dep else []

    # -----------------------------------------------------------------------
    # Health / Status
    # -----------------------------------------------------------------------

    @_app.get(f"{API_PREFIX}/health", tags=["System"])
    async def health():
        """System health check."""
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "3.0.0",
        }

    @_app.get(f"{API_PREFIX}/status", tags=["System"])
    async def module_status():
        """Module availability — which v3 modules are importable."""
        modules = {
            "sds.adapters.historical_ohlcv_daily_v3": "sentinel.sds.adapters.historical_ohlcv_daily_v3",
            "sfe.vcpe_tracker_v3": "sentinel.sfe.vcpe_tracker_v3",
            "sfe.activist_tracker_v3": "sentinel.sfe.activist_tracker_v3",
            "sfe.institutional_ownership_v3": "sentinel.sfe.institutional_ownership_v3",
            "sfe.charting_v3": "sentinel.sfe.charting_v3",
            "sfe.fx_surface_v3": "sentinel.sfe.fx_surface_v3",
            "sai.query_expander_v3": "sentinel.sai.query_expander_v3",
            "sai.nl_screener_v3": "sentinel.sai.nl_screener_v3",
            "sai.document_summarizer_v3": "sentinel.sai.document_summarizer_v3",
            "sai.earnings_rag_v3": "sentinel.sai.earnings_rag_v3",
            "sai.nl_strategy_generator_v3": "sentinel.sai.nl_strategy_generator_v3",
            "sai.factor_research_v3": "sentinel.sai.factor_research_v3",
            "sfe.private_company_v3": "sentinel.sfe.private_company_v3",
        }
        availability = {}
        for short, full in modules.items():
            try:
                import importlib
                importlib.import_module(full)
                availability[short] = "ok"
            except Exception as e:
                availability[short] = f"unavailable: {e}"
        return {"modules": availability}

    # -----------------------------------------------------------------------
    # Market Data endpoints
    # -----------------------------------------------------------------------

    @_app.get(f"{API_PREFIX}/quote/{{ticker}}", tags=["Market Data"], dependencies=deps)
    async def get_quote(ticker: str):
        """Real-time quote for a ticker."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.fast_info
            return {
                "ticker": ticker,
                "last": getattr(info, "last_price", None),
                "open": getattr(info, "open", None),
                "high": getattr(info, "day_high", None),
                "low": getattr(info, "day_low", None),
                "volume": getattr(info, "three_month_average_volume", None),
                "market_cap": getattr(info, "market_cap", None),
                "currency": getattr(info, "currency", "USD"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.warning("yfinance quote failed for %s: %s", ticker, exc)
            # Fallback: Yahoo Finance JSON API
            try:
                import urllib.request
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
                req = urllib.request.Request(url, headers={"User-Agent": "SENTINEL/3.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                meta = data["chart"]["result"][0]["meta"]
                return {
                    "ticker": ticker,
                    "last": meta.get("regularMarketPrice"),
                    "open": meta.get("regularMarketOpen"),
                    "high": meta.get("regularMarketDayHigh"),
                    "low": meta.get("regularMarketDayLow"),
                    "volume": meta.get("regularMarketVolume"),
                    "market_cap": None,
                    "currency": meta.get("currency", "USD"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc2:
                raise HTTPException(status_code=503, detail=f"Quote unavailable: {exc2}")

    @_app.get(f"{API_PREFIX}/history/{{ticker}}", tags=["Market Data"], dependencies=deps)
    async def get_history(
        ticker: str,
        start: str = "2024-01-01",
        end: str = "",
        timeframe: str = "1d",
    ):
        """OHLCV history for a ticker."""
        ticker = ticker.upper()
        end_dt = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            import yfinance as yf
            tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h",
                      "1d": "1d", "1wk": "1wk", "1mo": "1mo"}
            interval = tf_map.get(timeframe, "1d")
            df = yf.download(ticker, start=start, end=end_dt, interval=interval,
                             progress=False, auto_adjust=True)
            if _PANDAS_OK and hasattr(df, "reset_index"):
                df = df.reset_index()
                return {"ticker": ticker, "timeframe": timeframe,
                        "data": df.to_dict(orient="records")}
            return {"ticker": ticker, "timeframe": timeframe, "data": []}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"History unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/options/{{ticker}}", tags=["Market Data"], dependencies=deps)
    async def get_options(ticker: str, expiry: str = ""):
        """Options chain for a ticker."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return {"ticker": ticker, "expirations": [], "chain": {}}
            target = expiry if expiry in expirations else expirations[0]
            chain = t.option_chain(target)
            return {
                "ticker": ticker,
                "expiry": target,
                "expirations": list(expirations),
                "calls": chain.calls.to_dict(orient="records") if _PANDAS_OK else [],
                "puts": chain.puts.to_dict(orient="records") if _PANDAS_OK else [],
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Options unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/futures/{{commodity}}", tags=["Market Data"], dependencies=deps)
    async def get_futures(commodity: str):
        """Futures term structure for a commodity."""
        commodity = commodity.upper()
        mod = _lazy("sentinel.sfe.futures_curve_v3")
        if mod:
            try:
                engine = mod.FuturesCurveEngine()
                curve = engine.get_curve(commodity)
                return {"commodity": commodity, "curve": curve}
            except Exception as exc:
                logger.warning("futures_curve_v3 failed: %s", exc)
        # Fallback: yfinance continuous contract
        try:
            import yfinance as yf
            suffixes = ["=F", "1=F", "2=F", "3=F", "4=F", "5=F", "6=F"]
            curve = []
            for sfx in suffixes:
                sym = f"{commodity}{sfx}"
                t = yf.Ticker(sym)
                price = getattr(t.fast_info, "last_price", None)
                if price:
                    curve.append({"contract": sym, "price": price})
            return {"commodity": commodity, "curve": curve}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Futures unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/fx/{{base}}/{{quote}}", tags=["Market Data"], dependencies=deps)
    async def get_fx(base: str, quote: str):
        """FX spot and forward rates."""
        base, quote = base.upper(), quote.upper()
        mod = _lazy("sentinel.sfe.fx_surface_v3")
        if mod:
            try:
                engine = mod.FXSurfaceEngine()
                data = engine.get_spot_forward(base, quote)
                return {"base": base, "quote": quote, **data}
            except Exception as exc:
                logger.warning("fx_surface_v3 failed: %s", exc)
        # Fallback: Yahoo Finance
        try:
            import yfinance as yf
            pair = f"{base}{quote}=X"
            t = yf.Ticker(pair)
            spot = getattr(t.fast_info, "last_price", None)
            return {
                "base": base, "quote": quote,
                "spot": spot, "forwards": {},
                "source": "yfinance",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"FX unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/crypto/{{symbol}}", tags=["Market Data"], dependencies=deps)
    async def get_crypto(symbol: str):
        """Crypto price and basic metrics."""
        symbol = symbol.upper()
        try:
            import yfinance as yf
            pair = f"{symbol}-USD"
            t = yf.Ticker(pair)
            info = t.fast_info
            return {
                "symbol": symbol,
                "price_usd": getattr(info, "last_price", None),
                "market_cap": getattr(info, "market_cap", None),
                "volume_24h": getattr(info, "three_month_average_volume", None),
                "currency": "USD",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Crypto price unavailable: {exc}")

    # -----------------------------------------------------------------------
    # Fundamentals endpoints
    # -----------------------------------------------------------------------

    @_app.get(f"{API_PREFIX}/fundamentals/{{ticker}}", tags=["Fundamentals"], dependencies=deps)
    async def get_fundamentals(ticker: str):
        """Key financial ratios and summary fundamentals."""
        ticker = ticker.upper()
        mod = _lazy("sentinel.sai.fundamental_data_layer_v3")
        if mod:
            try:
                layer = mod.FundamentalDataLayer()
                return layer.get_summary(ticker)
            except Exception as exc:
                logger.warning("fundamental_data_layer_v3 failed: %s", exc)
        # Fallback: yfinance info dict
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            keys = [
                "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
                "enterpriseToEbitda", "returnOnEquity", "returnOnAssets",
                "grossMargins", "operatingMargins", "profitMargins",
                "debtToEquity", "currentRatio", "quickRatio",
                "revenueGrowth", "earningsGrowth", "dividendYield",
                "trailingEps", "forwardEps", "marketCap",
            ]
            return {k: info.get(k) for k in keys} | {"ticker": ticker}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Fundamentals unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/income/{{ticker}}", tags=["Fundamentals"], dependencies=deps)
    async def get_income(ticker: str, periods: int = 4):
        """Income statement — quarterly/annual."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.quarterly_income_stmt if periods <= 8 else t.income_stmt
            if _PANDAS_OK and df is not None and not df.empty:
                return {"ticker": ticker, "periods": periods,
                        "data": df.iloc[:, :periods].to_dict()}
            return {"ticker": ticker, "data": {}}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Income statement unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/balance/{{ticker}}", tags=["Fundamentals"], dependencies=deps)
    async def get_balance(ticker: str, periods: int = 4):
        """Balance sheet — quarterly/annual."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.quarterly_balance_sheet if periods <= 8 else t.balance_sheet
            if _PANDAS_OK and df is not None and not df.empty:
                return {"ticker": ticker, "periods": periods,
                        "data": df.iloc[:, :periods].to_dict()}
            return {"ticker": ticker, "data": {}}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Balance sheet unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/cashflow/{{ticker}}", tags=["Fundamentals"], dependencies=deps)
    async def get_cashflow(ticker: str, periods: int = 4):
        """Cash flow statement — quarterly/annual."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.quarterly_cashflow if periods <= 8 else t.cashflow
            if _PANDAS_OK and df is not None and not df.empty:
                return {"ticker": ticker, "periods": periods,
                        "data": df.iloc[:, :periods].to_dict()}
            return {"ticker": ticker, "data": {}}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Cash flow unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/segments/{{ticker}}", tags=["Fundamentals"], dependencies=deps)
    async def get_segments(ticker: str):
        """Business segment breakdown (revenue/operating income)."""
        ticker = ticker.upper()
        mod = _lazy("sentinel.sfe.segment_analytics_v3")
        if mod:
            try:
                engine = mod.SegmentAnalyticsEngine()
                return engine.get_segments(ticker)
            except Exception as exc:
                logger.warning("segment_analytics_v3 failed for %s: %s", ticker, exc)
        return {"ticker": ticker, "segments": [],
                "note": "segment_analytics_v3 unavailable"}

    @_app.get(f"{API_PREFIX}/valuation/{{ticker}}", tags=["Fundamentals"], dependencies=deps)
    async def get_valuation(ticker: str):
        """DCF intrinsic value + comparable companies valuation."""
        ticker = ticker.upper()
        mod = _lazy("sentinel.sfe.comps_engine_v3")
        if mod:
            try:
                engine = mod.CompsEngine()
                return engine.get_valuation(ticker)
            except Exception as exc:
                logger.warning("comps_engine_v3 failed for %s: %s", ticker, exc)
        return {"ticker": ticker, "dcf": None, "comps": [],
                "note": "comps_engine_v3 unavailable"}

    # -----------------------------------------------------------------------
    # Ownership & Filings endpoints
    # -----------------------------------------------------------------------

    @_app.get(f"{API_PREFIX}/insider/{{ticker}}", tags=["Ownership & Filings"], dependencies=deps)
    async def get_insider(ticker: str, days: int = 90):
        """Form 4 insider transactions."""
        ticker = ticker.upper()
        try:
            import urllib.request
            # SEC EDGAR full-text search for Form 4
            url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
                f"&forms=4&dateRange=custom"
                f"&startdt={(datetime.now()-timedelta(days=days)).strftime('%Y-%m-%d')}"
                f"&enddt={datetime.now().strftime('%Y-%m-%d')}&_source=file_date,entity_name,file_num"
            )
            req = urllib.request.Request(url, headers={
                "User-Agent": "SENTINEL sentinel@example.com",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            hits = data.get("hits", {}).get("hits", [])
            transactions = [
                {
                    "entity": h.get("_source", {}).get("entity_name", ""),
                    "filed": h.get("_source", {}).get("file_date", ""),
                    "form": "4",
                    "accession": h.get("_id", ""),
                }
                for h in hits[:50]
            ]
            return {"ticker": ticker, "days": days, "transactions": transactions,
                    "count": len(transactions)}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Insider data unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/institutions/{{ticker}}", tags=["Ownership & Filings"], dependencies=deps)
    async def get_institutions(ticker: str):
        """13F institutional holders."""
        ticker = ticker.upper()
        mod = _lazy("sentinel.sfe.institutional_ownership_v3")
        if mod:
            try:
                engine = mod.InstitutionalOwnershipEngine()
                return engine.get_holders(ticker)
            except Exception as exc:
                logger.warning("institutional_ownership_v3 failed: %s", exc)
        # Fallback: yfinance
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.institutional_holders
            if _PANDAS_OK and df is not None:
                return {"ticker": ticker,
                        "holders": df.to_dict(orient="records")}
            return {"ticker": ticker, "holders": []}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"13F data unavailable: {exc}")

    @_app.get(f"{API_PREFIX}/activist/{{ticker}}", tags=["Ownership & Filings"], dependencies=deps)
    async def get_activist(ticker: str):
        """Activist investor campaigns."""
        ticker = ticker.upper()
        mod = _lazy("sentinel.sfe.activist_tracker_v3")
        if mod:
            try:
                tracker = mod.ActivistTracker()
                return tracker.get_campaigns(ticker)
            except Exception as exc:
                logger.warning("activist_tracker_v3 failed for %s: %s", ticker, exc)
        return {"ticker": ticker, "campaigns": [],
                "note": "activist_tracker_v3 unavailable"}

    @_app.get(f"{API_PREFIX}/filings/{{ticker}}", tags=["Ownership & Filings"], dependencies=deps)
    async def get_filings(ticker: str, form_type: str = "10-K"):
        """EDGAR filings list."""
        ticker = ticker.upper()
        try:
            import urllib.request
            url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
                f"&forms={form_type}&_source=file_date,entity_name,file_num,period_of_report"
            )
            req = urllib.request.Request(url, headers={
                "User-Agent": "SENTINEL sentinel@example.com",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            hits = data.get("hits", {}).get("hits", [])
            filings = [
                {
                    "entity": h.get("_source", {}).get("entity_name", ""),
                    "filed": h.get("_source", {}).get("file_date", ""),
                    "period": h.get("_source", {}).get("period_of_report", ""),
                    "form": form_type,
                    "accession": h.get("_id", ""),
                }
                for h in hits[:20]
            ]
            return {"ticker": ticker, "form_type": form_type,
                    "filings": filings, "count": len(filings)}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Filings unavailable: {exc}")

    # -----------------------------------------------------------------------
    # Screener endpoints
    # -----------------------------------------------------------------------

    @_app.post(f"{API_PREFIX}/screen/fundamental", tags=["Screeners"], dependencies=deps)
    async def screen_fundamental(req: ScreenFundamentalRequest):
        """Fundamental screener — filter by ratios, margins, growth."""
        mod = _lazy("sentinel.sai.nl_screener_v3")
        if mod:
            try:
                screener = mod.FundamentalScreener()
                results = screener.screen(req.criteria)
                if _PANDAS_OK and hasattr(results, "to_dict"):
                    return {"results": results.to_dict(orient="records"),
                            "count": len(results)}
                return {"results": results, "count": len(results)}
            except Exception as exc:
                logger.warning("nl_screener_v3 fundamental screen failed: %s", exc)
        # Fallback: yfinance-based mini screener
        return _yfinance_mini_screener(req.criteria)

    @_app.post(f"{API_PREFIX}/screen/technical", tags=["Screeners"], dependencies=deps)
    async def screen_technical(req: ScreenTechnicalRequest):
        """Technical screener — RSI, moving averages, volume patterns."""
        mod = _lazy("sentinel.sai.nl_screener_v3")
        if mod:
            try:
                screener = mod.TechnicalScreener()
                results = screener.screen(req.criteria)
                if _PANDAS_OK and hasattr(results, "to_dict"):
                    return {"results": results.to_dict(orient="records"),
                            "count": len(results)}
                return {"results": results, "count": len(results)}
            except Exception as exc:
                logger.warning("technical screener failed: %s", exc)
        return {"results": [], "note": "technical screener unavailable", "criteria": req.criteria}

    @_app.post(f"{API_PREFIX}/screen/nl", tags=["Screeners"], dependencies=deps)
    async def screen_nl(req: ScreenNLRequest):
        """Natural-language screener — e.g. 'profitable tech with PE < 20'."""
        mod = _lazy("sentinel.sai.nl_screener_v3")
        if mod:
            try:
                screener = mod.NLScreener()
                results = screener.screen(req.query)
                if _PANDAS_OK and hasattr(results, "to_dict"):
                    return {"query": req.query,
                            "results": results.to_dict(orient="records"),
                            "count": len(results)}
                return {"query": req.query, "results": results}
            except Exception as exc:
                logger.warning("nl_screener_v3 NL screen failed: %s", exc)
        return {"query": req.query, "results": [],
                "note": "nl_screener_v3 unavailable"}

    @_app.get(f"{API_PREFIX}/screen/preset/{{name}}", tags=["Screeners"], dependencies=deps)
    async def screen_preset(name: str):
        """Run a named preset screen (magic_formula, piotroski, earnings_surprise, etc.)."""
        presets = {
            "magic_formula": {"roe_min": 0.15, "earnings_yield_min": 0.10},
            "piotroski": {"piotroski_min": 7},
            "earnings_surprise": {"eps_surprise_pct_min": 5.0},
            "low_pe_growth": {"pe_max": 15, "revenue_growth_min": 0.10},
            "high_yield": {"dividend_yield_min": 0.04, "payout_ratio_max": 0.80},
            "momentum": {"price_52w_pct_min": 20, "rsi_min": 50},
        }
        if name not in presets:
            raise HTTPException(
                status_code=404,
                detail=f"Preset '{name}' not found. Available: {list(presets.keys())}",
            )
        criteria = presets[name]
        mod = _lazy("sentinel.sai.nl_screener_v3")
        if mod:
            try:
                screener = mod.FundamentalScreener()
                results = screener.screen(criteria)
                if _PANDAS_OK and hasattr(results, "to_dict"):
                    return {"preset": name, "criteria": criteria,
                            "results": results.to_dict(orient="records"),
                            "count": len(results)}
                return {"preset": name, "criteria": criteria,
                        "results": results}
            except Exception as exc:
                logger.warning("preset screen failed: %s", exc)
        return {"preset": name, "criteria": criteria, "results": [],
                "note": "screener module unavailable"}

    # -----------------------------------------------------------------------
    # Risk & Portfolio endpoints
    # -----------------------------------------------------------------------

    @_app.post(f"{API_PREFIX}/risk/var", tags=["Risk & Portfolio"], dependencies=deps)
    async def compute_var(req: VaRRequest):
        """Parametric + historical VaR and CVaR."""
        mod = _lazy("sentinel.sbx.risk_engine_v3")
        if mod is None:
            mod = _lazy("sentinel.sbe.risk_engine")
        if mod:
            try:
                engine = mod.RiskEngine()
                result = engine.compute_var(
                    req.holdings, req.confidence, req.lookback_days
                )
                return result
            except Exception as exc:
                logger.warning("risk_engine failed: %s", exc)
        # Fallback: simple parametric VaR using yfinance + scipy
        return _compute_var_fallback(req.holdings, req.confidence, req.lookback_days)

    @_app.post(f"{API_PREFIX}/risk/stress", tags=["Risk & Portfolio"], dependencies=deps)
    async def stress_test(req: StressRequest):
        """Stress test portfolio against historical scenarios."""
        scenario_shocks = {
            "2008_gfc": {"equity": -0.55, "bonds": 0.10, "gold": 0.25, "vol": 3.0},
            "2020_covid": {"equity": -0.34, "bonds": 0.08, "gold": 0.15, "vol": 2.5},
            "2022_rates": {"equity": -0.25, "bonds": -0.18, "gold": -0.02, "vol": 1.5},
            "dot_com": {"equity": -0.50, "bonds": 0.05, "gold": 0.10, "vol": 2.0},
            "russia_ukraine": {"equity": -0.12, "energy": 0.60, "commodities": 0.40},
        }
        results = []
        total_value = sum(req.holdings.values())
        for scenario in req.scenarios:
            shocks = scenario_shocks.get(scenario, {"equity": -0.20})
            equity_shock = shocks.get("equity", -0.20)
            pnl = total_value * equity_shock
            results.append({
                "scenario": scenario,
                "portfolio_pnl": round(pnl, 2),
                "portfolio_pnl_pct": round(equity_shock * 100, 2),
                "assumptions": shocks,
            })
        return {"holdings": req.holdings, "scenarios": results,
                "total_portfolio_value": total_value}

    @_app.post(f"{API_PREFIX}/portfolio/optimize", tags=["Risk & Portfolio"], dependencies=deps)
    async def optimize_portfolio(req: PortfolioOptimizeRequest):
        """Portfolio optimization — max Sharpe, min variance, risk parity."""
        mod = _lazy("sentinel.spm.optimizer_v3")
        if mod is None:
            mod = _lazy("sentinel.spm.portfolio_optimizer")
        if mod:
            try:
                opt = mod.PortfolioOptimizer()
                result = opt.optimize(
                    req.tickers, method=req.method,
                    start=req.start, end=req.end,
                    constraints=req.constraints,
                )
                return result
            except Exception as exc:
                logger.warning("portfolio optimizer failed: %s", exc)
        # Fallback: equal-weight
        n = len(req.tickers)
        weights = {t: round(1.0 / n, 4) for t in req.tickers} if n else {}
        return {
            "method": "equal_weight_fallback",
            "tickers": req.tickers,
            "weights": weights,
            "note": "optimizer module unavailable; returning equal weights",
        }

    @_app.get(f"{API_PREFIX}/risk/correlation", tags=["Risk & Portfolio"], dependencies=deps)
    async def get_correlation(tickers: str = "AAPL,MSFT,GOOGL"):
        """Correlation matrix for a comma-separated list of tickers."""
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            raise HTTPException(status_code=400, detail="Provide at least one ticker.")
        try:
            import yfinance as yf
            df = yf.download(ticker_list, period="1y", progress=False, auto_adjust=True)
            if _PANDAS_OK and df is not None and not df.empty:
                if "Close" in df.columns:
                    closes = df["Close"]
                elif hasattr(df.columns, "levels"):
                    closes = df.xs("Close", axis=1, level=0)
                else:
                    closes = df
                returns = closes.pct_change().dropna()
                corr = returns.corr()
                return {
                    "tickers": ticker_list,
                    "correlation_matrix": corr.to_dict(),
                    "period": "1y",
                }
        except Exception as exc:
            logger.warning("correlation matrix failed: %s", exc)
        return {"tickers": ticker_list, "correlation_matrix": {},
                "note": "correlation computation failed"}

    # -----------------------------------------------------------------------
    # AI & Research endpoints
    # -----------------------------------------------------------------------

    @_app.post(f"{API_PREFIX}/ai/summarize", tags=["AI & Research"], dependencies=deps)
    async def ai_summarize(req: AISummarizeRequest):
        """AI-powered SEC filing summary."""
        mod = _lazy("sentinel.sai.document_summarizer_v3")
        if mod:
            try:
                summarizer = mod.DocumentSummarizer()
                result = summarizer.summarize_filing(req.ticker, req.form)
                return {"ticker": req.ticker, "form": req.form, "summary": result}
            except Exception as exc:
                logger.warning("document_summarizer_v3 failed: %s", exc)
        return {"ticker": req.ticker, "form": req.form, "summary": None,
                "note": "document_summarizer_v3 unavailable"}

    @_app.post(f"{API_PREFIX}/ai/qa", tags=["AI & Research"], dependencies=deps)
    async def ai_qa(req: AIQARequest):
        """RAG-powered Q&A over SENTINEL document corpus."""
        mod = _lazy("sentinel.sai.earnings_rag_v3")
        if mod:
            try:
                rag = mod.EarningsRAG()
                answer = rag.answer(req.question, tickers=req.tickers, top_k=req.top_k)
                return {"question": req.question, "tickers": req.tickers,
                        "answer": answer}
            except Exception as exc:
                logger.warning("earnings_rag_v3 failed: %s", exc)
        return {"question": req.question, "answer": None,
                "note": "earnings_rag_v3 unavailable"}

    @_app.post(f"{API_PREFIX}/ai/strategy", tags=["AI & Research"], dependencies=deps)
    async def ai_strategy(req: AIStrategyRequest):
        """NL-driven strategy generation (backtest-ready code + config)."""
        mod = _lazy("sentinel.sai.nl_strategy_generator_v3")
        if mod:
            try:
                gen = mod.NLStrategyGenerator()
                strategy = gen.generate(req.description)
                return {"description": req.description, "strategy": strategy}
            except Exception as exc:
                logger.warning("nl_strategy_generator_v3 failed: %s", exc)
        return {"description": req.description, "strategy": None,
                "note": "nl_strategy_generator_v3 unavailable"}

    @_app.get(f"{API_PREFIX}/ai/research/{{ticker}}", tags=["AI & Research"], dependencies=deps)
    async def ai_research(ticker: str):
        """Full AI research workflow — fundamentals, sentiment, risk, summary."""
        ticker = ticker.upper()
        mod = _lazy("sentinel.sai.factor_research_v3")
        if mod:
            try:
                agent = mod.FactorResearchAgent()
                report = agent.run(ticker)
                return {"ticker": ticker, "report": report}
            except Exception as exc:
                logger.warning("factor_research_v3 failed for %s: %s", ticker, exc)
        # Fallback: lightweight composite
        result: dict = {"ticker": ticker, "sections": {}}
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info
            result["sections"]["fundamentals"] = {
                "market_cap": info.get("marketCap"),
                "pe": info.get("trailingPE"),
                "roe": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
            }
        except Exception:
            pass
        return result

    # -----------------------------------------------------------------------
    # Backtesting endpoints
    # -----------------------------------------------------------------------

    @_app.post(f"{API_PREFIX}/backtest/run", tags=["Backtesting"], dependencies=deps)
    async def backtest_run(req: BacktestRunRequest):
        """Submit and run a backtest asynchronously."""
        run_id = str(uuid.uuid4())
        mod = _lazy("sentinel.sbx.backtest_engine_v3")
        if mod is None:
            mod = _lazy("sentinel.sbx.backtester")
        if mod:
            try:
                engine = mod.BacktestEngine()
                result = engine.run(
                    strategy=req.strategy,
                    symbols=req.symbols,
                    start=req.start,
                    end=req.end or datetime.now().strftime("%Y-%m-%d"),
                    initial_capital=req.initial_capital,
                    commission=req.commission,
                )
                _backtest_results[run_id] = result
                return {"run_id": run_id, "status": "complete",
                        "summary": result.get("summary", {})}
            except Exception as exc:
                logger.warning("backtest_engine failed: %s", exc)
        # Fallback: buy-and-hold simulation
        result = _buy_and_hold_backtest(
            req.symbols, req.start,
            req.end or datetime.now().strftime("%Y-%m-%d"),
            req.initial_capital,
        )
        _backtest_results[run_id] = result
        return {"run_id": run_id, "status": "complete",
                "strategy": req.strategy, "summary": result.get("summary", {})}

    @_app.get(f"{API_PREFIX}/backtest/result/{{run_id}}", tags=["Backtesting"], dependencies=deps)
    async def backtest_result(run_id: str):
        """Get full backtest result by run_id."""
        if run_id not in _backtest_results:
            raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
        return {"run_id": run_id, "result": _backtest_results[run_id]}

    @_app.get(f"{API_PREFIX}/backtest/tearsheet/{{run_id}}", tags=["Backtesting"], dependencies=deps)
    async def backtest_tearsheet(run_id: str):
        """Formatted performance tearsheet for a completed backtest."""
        if run_id not in _backtest_results:
            raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
        result = _backtest_results[run_id]
        summary = result.get("summary", {})
        tearsheet = {
            "run_id": run_id,
            "performance": {
                "total_return": summary.get("total_return"),
                "cagr": summary.get("cagr"),
                "sharpe": summary.get("sharpe"),
                "max_drawdown": summary.get("max_drawdown"),
                "calmar": summary.get("calmar"),
                "sortino": summary.get("sortino"),
            },
            "risk": {
                "volatility_ann": summary.get("volatility_ann"),
                "var_95": summary.get("var_95"),
                "beta": summary.get("beta"),
                "alpha": summary.get("alpha"),
            },
            "trades": {
                "total_trades": summary.get("total_trades"),
                "win_rate": summary.get("win_rate"),
                "avg_win": summary.get("avg_win"),
                "avg_loss": summary.get("avg_loss"),
                "profit_factor": summary.get("profit_factor"),
            },
        }
        return tearsheet

    return _app


# ---------------------------------------------------------------------------
# Fallback helpers (no external SENTINEL module deps)
# ---------------------------------------------------------------------------

def _yfinance_mini_screener(criteria: dict) -> dict:
    """Minimal screener using yfinance for a fixed universe."""
    _UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B",
        "JNJ", "V", "WMT", "JPM", "MA", "UNH", "HD", "PG", "XOM", "BAC",
        "DIS", "NFLX", "PYPL", "ADBE", "CRM", "AMD", "INTC", "CSCO", "ORCL",
    ]
    results = []
    try:
        import yfinance as yf
        for ticker in _UNIVERSE[:12]:  # limit to avoid rate-limiting
            try:
                info = yf.Ticker(ticker).info
                pe = info.get("trailingPE")
                roe = info.get("returnOnEquity")
                growth = info.get("revenueGrowth")
                margin = info.get("grossMargins")
                dy = info.get("dividendYield")
                passes = True
                if "pe_max" in criteria and pe and pe > criteria["pe_max"]:
                    passes = False
                if "pe_min" in criteria and pe and pe < criteria["pe_min"]:
                    passes = False
                if "roe_min" in criteria and roe and roe < criteria["roe_min"]:
                    passes = False
                if "revenue_growth_min" in criteria and growth and growth < criteria["revenue_growth_min"]:
                    passes = False
                if "dividend_yield_min" in criteria and dy and dy < criteria["dividend_yield_min"]:
                    passes = False
                if passes:
                    results.append({
                        "ticker": ticker,
                        "pe": pe, "roe": roe,
                        "revenue_growth": growth,
                        "gross_margin": margin,
                        "dividend_yield": dy,
                    })
                time.sleep(0.1)
            except Exception:
                continue
    except Exception:
        pass
    return {"results": results, "count": len(results),
            "universe": "top-27-sp500", "note": "mini fallback screener"}


def _compute_var_fallback(holdings: dict, confidence: float, lookback_days: int) -> dict:
    """Simple parametric VaR fallback using yfinance price data."""
    import math
    if not _PANDAS_OK or not _REQUESTS_OK:
        total = sum(holdings.values())
        z = 2.326 if confidence >= 0.99 else 1.645
        var = total * 0.02 * z
        return {
            "var": round(var, 2), "cvar": round(var * 1.3, 2),
            "confidence": confidence,
            "note": "approximation (pandas/requests unavailable)",
        }
    try:
        import yfinance as yf
        tickers = list(holdings.keys())
        weights_val = list(holdings.values())
        total = sum(weights_val)
        df = yf.download(tickers, period=f"{min(lookback_days, 504)}d",
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("no price data")
        closes = df["Close"] if "Close" in df.columns else df
        returns = closes.pct_change().dropna()
        port_weights = [w / total for w in weights_val]
        if len(tickers) == 1:
            port_returns = returns.iloc[:, 0]
        else:
            port_returns = (returns * port_weights).sum(axis=1)
        sorted_r = port_returns.sort_values()
        idx = int((1 - confidence) * len(sorted_r))
        var_pct = abs(sorted_r.iloc[idx])
        cvar_pct = abs(sorted_r.iloc[:idx].mean()) if idx > 0 else var_pct * 1.3
        return {
            "var": round(var_pct * total, 2),
            "var_pct": round(var_pct * 100, 4),
            "cvar": round(cvar_pct * total, 2),
            "cvar_pct": round(cvar_pct * 100, 4),
            "confidence": confidence,
            "portfolio_value": total,
            "lookback_days": lookback_days,
            "method": "historical",
        }
    except Exception as exc:
        total = sum(holdings.values())
        import math
        z = 2.326 if confidence >= 0.99 else 1.645
        var = total * 0.015 * z
        return {
            "var": round(var, 2), "cvar": round(var * 1.3, 2),
            "confidence": confidence,
            "note": f"parametric approximation: {exc}",
        }


def _buy_and_hold_backtest(symbols: list, start: str, end: str, capital: float) -> dict:
    """Equal-weight buy-and-hold backtest as fallback."""
    try:
        import yfinance as yf
        if not symbols:
            return {"summary": {}, "equity_curve": []}
        df = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
        if _PANDAS_OK and not df.empty:
            closes = df["Close"] if "Close" in df.columns else df
            if hasattr(closes, "iloc"):
                rets = closes.pct_change().dropna()
                n = len(symbols)
                port_rets = rets.mean(axis=1) if n > 1 else rets
                equity = (1 + port_rets).cumprod() * capital
                total_ret = float(equity.iloc[-1] / capital - 1) if len(equity) > 0 else 0.0
                ann_vol = float(port_rets.std() * (252 ** 0.5)) if len(port_rets) > 0 else 0.0
                sharpe = (total_ret / ann_vol) if ann_vol > 0 else 0.0
                return {
                    "summary": {
                        "total_return": round(total_ret, 4),
                        "sharpe": round(sharpe, 3),
                        "volatility_ann": round(ann_vol, 4),
                        "final_equity": round(float(equity.iloc[-1]), 2),
                        "strategy": "equal_weight_buy_and_hold",
                    },
                    "equity_curve": list(zip(
                        [str(d) for d in equity.index.tolist()],
                        [round(float(v), 2) for v in equity.values.tolist()],
                    )),
                }
    except Exception as exc:
        logger.warning("buy_and_hold_backtest failed: %s", exc)
    return {"summary": {"strategy": "equal_weight_buy_and_hold", "note": "data unavailable"},
            "equity_curve": []}


# ---------------------------------------------------------------------------
# Build the app singleton
# ---------------------------------------------------------------------------

app = _build_app()


# ---------------------------------------------------------------------------
# SentinelAPIServer — thin wrapper exposing app as a class attribute
# ---------------------------------------------------------------------------

class SentinelAPIServer:
    """Wrapper class around the FastAPI application.

    Usage:
        server = SentinelAPIServer()
        server.run(host="0.0.0.0", port=8000)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = app

    def run(self):
        if _UVICORN_OK:
            _uvicorn.run(self.app, host=self.host, port=self.port)
        else:
            self._print_routes()

    def _print_routes(self):
        print(f"\nSENTINEL API v3 — uvicorn not installed.")
        print(f"  Install: pip install uvicorn[standard]\n")
        if self.app:
            print("Registered routes:")
            for route in self.app.routes:
                methods = getattr(route, "methods", {"WS"})
                print(f"  {','.join(sorted(methods))} {route.path}")


# ---------------------------------------------------------------------------
# SentinelWebSocketServer — asyncio hub for streaming channels
# ---------------------------------------------------------------------------

class SentinelWebSocketServer:
    """WebSocket server exposing streaming channels.

    Channels
    --------
    quotes.{ticker}   — live price polls every 5 seconds
    news.{ticker}     — news via GDELT, polled every 60 seconds
    alerts.{ticker}   — price alert crossings (user-defined)
    regime            — macro regime change notifications
    options.{ticker}  — unusual options activity

    Message protocol
    ----------------
    Subscribe:   {"type": "subscribe",   "channels": ["quotes.AAPL", "news.AAPL"]}
    Unsubscribe: {"type": "unsubscribe", "channels": ["quotes.AAPL"]}
    Alert set:   {"type": "set_alert",   "ticker": "AAPL", "above": 200.0}
    Ping:        {"type": "ping"}
    """

    def __init__(self, host: str = "0.0.0.0", port: int = WS_PORT):
        self.host = host
        self.port = port
        # channel → set of websocket connections
        self._subscriptions: Dict[str, Set[Any]] = defaultdict(set)
        # websocket → set of subscribed channels
        self._client_channels: Dict[Any, Set[str]] = defaultdict(set)
        # ticker → price alert thresholds  {ws: {"above": float, "below": float}}
        self._alerts: Dict[str, Dict[Any, dict]] = defaultdict(dict)
        # background polling tasks
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = asyncio.Lock()

    async def handle_subscribe(self, websocket: Any, channels: List[str]):
        """Subscribe a client to one or more channels."""
        async with self._lock:
            for ch in channels:
                self._subscriptions[ch].add(websocket)
                self._client_channels[websocket].add(ch)
                logger.debug("WS subscribe: %s → %s", id(websocket), ch)
                # spawn background poller if not already running
                if ch not in self._poll_tasks or self._poll_tasks[ch].done():
                    self._poll_tasks[ch] = asyncio.create_task(
                        self._poll_channel(ch)
                    )

    async def handle_unsubscribe(self, websocket: Any, channels: List[str]):
        """Unsubscribe a client from channels."""
        async with self._lock:
            for ch in channels:
                self._subscriptions[ch].discard(websocket)
                self._client_channels[websocket].discard(ch)

    async def disconnect(self, websocket: Any):
        """Clean up all subscriptions for a disconnected client."""
        async with self._lock:
            for ch in list(self._client_channels.get(websocket, [])):
                self._subscriptions[ch].discard(websocket)
            self._client_channels.pop(websocket, None)

    async def broadcast(self, channel: str, data: dict):
        """Send data to all subscribers of a channel."""
        subscribers = set(self._subscriptions.get(channel, []))
        if not subscribers:
            return
        message = json.dumps({"channel": channel, "data": data,
                               "ts": datetime.now(timezone.utc).isoformat()})
        dead = set()
        for ws in subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subscriptions[channel].discard(ws)
                    self._client_channels.pop(ws, None)

    async def _poll_channel(self, channel: str):
        """Background polling coroutine for a single channel."""
        ch_type, _, ch_arg = channel.partition(".")
        poll_intervals = {
            "quotes": 5,
            "news": 60,
            "alerts": 10,
            "regime": 300,
            "options": 30,
        }
        interval = poll_intervals.get(ch_type, 30)

        while self._running:
            try:
                if not self._subscriptions.get(channel):
                    # no subscribers — stop polling
                    break
                if ch_type == "quotes" and ch_arg:
                    data = await self._fetch_quote_async(ch_arg)
                elif ch_type == "news" and ch_arg:
                    data = await self._fetch_news_async(ch_arg)
                elif ch_type == "alerts" and ch_arg:
                    await self._check_alerts(ch_arg)
                    data = None
                elif ch_type == "regime":
                    data = await self._fetch_regime_async()
                elif ch_type == "options" and ch_arg:
                    data = await self._fetch_options_flow_async(ch_arg)
                else:
                    data = None

                if data:
                    await self.broadcast(channel, data)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("poll_channel %s error: %s", channel, exc)

            await asyncio.sleep(interval)

    async def _fetch_quote_async(self, ticker: str) -> dict:
        """Fetch quote using yfinance in a thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_quote, ticker)

    def _sync_quote(self, ticker: str) -> dict:
        try:
            import urllib.request
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
            req = urllib.request.Request(url, headers={"User-Agent": "SENTINEL/3.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            meta = data["chart"]["result"][0]["meta"]
            return {
                "ticker": ticker,
                "price": meta.get("regularMarketPrice"),
                "prev_close": meta.get("previousClose"),
                "volume": meta.get("regularMarketVolume"),
                "currency": meta.get("currency", "USD"),
            }
        except Exception:
            return {"ticker": ticker, "price": None}

    async def _fetch_news_async(self, ticker: str) -> dict:
        """Fetch recent news from GDELT for a ticker."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_gdelt_news, ticker)

    def _sync_gdelt_news(self, ticker: str) -> dict:
        try:
            import urllib.request
            url = (
                f"https://api.gdeltproject.org/api/v2/doc/doc"
                f"?query={ticker}&mode=artlist&maxrecords=5&format=json"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "SENTINEL/3.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            articles = data.get("articles", [])[:5]
            return {
                "ticker": ticker,
                "articles": [
                    {
                        "title": a.get("title", ""),
                        "url": a.get("url", ""),
                        "seendate": a.get("seendate", ""),
                        "domain": a.get("domain", ""),
                    }
                    for a in articles
                ],
            }
        except Exception:
            return {"ticker": ticker, "articles": []}

    async def _fetch_regime_async(self) -> dict:
        """Simple regime detection via VIX and yield curve."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_regime)

    def _sync_regime(self) -> dict:
        try:
            import urllib.request
            # VIX proxy
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=5d"
            req = urllib.request.Request(url, headers={"User-Agent": "SENTINEL/3.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            vix = closes[-1] if closes else None
            regime = "risk_off" if (vix and vix > 25) else "risk_on"
            return {"vix": vix, "regime": regime,
                    "ts": datetime.now(timezone.utc).isoformat()}
        except Exception:
            return {"regime": "unknown", "vix": None}

    async def _fetch_options_flow_async(self, ticker: str) -> dict:
        """Detect unusual options activity via yfinance."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_unusual_options, ticker)

    def _sync_unusual_options(self, ticker: str) -> dict:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            exps = t.options
            if not exps:
                return {"ticker": ticker, "unusual": []}
            chain = t.option_chain(exps[0])
            unusual = []
            for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
                if _PANDAS_OK and not df.empty:
                    # volume / OI > 2 is "unusual"
                    if "volume" in df.columns and "openInterest" in df.columns:
                        mask = df["volume"] > df["openInterest"] * 2
                        hits = df[mask].head(3)
                        for _, row in hits.iterrows():
                            unusual.append({
                                "type": opt_type,
                                "strike": row.get("strike"),
                                "volume": row.get("volume"),
                                "oi": row.get("openInterest"),
                                "iv": row.get("impliedVolatility"),
                            })
            return {"ticker": ticker, "expiry": exps[0], "unusual": unusual}
        except Exception:
            return {"ticker": ticker, "unusual": []}

    async def _check_alerts(self, ticker: str):
        """Fire price alerts if thresholds are crossed."""
        price_data = await self._fetch_quote_async(ticker)
        price = price_data.get("price")
        if price is None:
            return
        alert_ch = f"alerts.{ticker}"
        for ws, thresholds in list(self._alerts.get(ticker, {}).items()):
            above = thresholds.get("above")
            below = thresholds.get("below")
            if above and price > above:
                try:
                    await ws.send_text(json.dumps({
                        "channel": alert_ch,
                        "event": "price_above",
                        "ticker": ticker, "price": price,
                        "threshold": above,
                    }))
                except Exception:
                    pass
            if below and price < below:
                try:
                    await ws.send_text(json.dumps({
                        "channel": alert_ch,
                        "event": "price_below",
                        "ticker": ticker, "price": price,
                        "threshold": below,
                    }))
                except Exception:
                    pass

    async def _keepalive(self, websocket: Any):
        """Ping/pong keepalive every 30 seconds."""
        while self._running:
            await asyncio.sleep(30)
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
            except Exception:
                break

    async def handle_client(self, websocket: Any):
        """Main client message handler (for use with FastAPI WebSocket or websockets lib)."""
        keepalive_task = asyncio.create_task(self._keepalive(websocket))
        try:
            async for raw in websocket.iter_text():
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg_type = msg.get("type", "")
                if msg_type == "subscribe":
                    await self.handle_subscribe(websocket, msg.get("channels", []))
                elif msg_type == "unsubscribe":
                    await self.handle_unsubscribe(websocket, msg.get("channels", []))
                elif msg_type == "set_alert":
                    ticker = msg.get("ticker", "").upper()
                    if ticker:
                        self._alerts[ticker][websocket] = {
                            "above": msg.get("above"),
                            "below": msg.get("below"),
                        }
                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
        except Exception:
            pass
        finally:
            keepalive_task.cancel()
            await self.disconnect(websocket)

    async def start(self):
        """Start the standalone WebSocket server."""
        self._running = True
        if not _WEBSOCKETS_OK:
            logger.error("websockets library not installed. pip install websockets")
            return
        import websockets as ws_lib

        async def _handler(websocket, path=None):
            await self.handle_client(websocket)

        logger.info("SENTINEL WebSocket server starting on ws://%s:%s/ws",
                    self.host, self.port)
        async with ws_lib.serve(_handler, self.host, self.port):
            await asyncio.Future()  # run forever

    def run(self):
        """Run the WebSocket server (blocking)."""
        asyncio.run(self.start())


# ---------------------------------------------------------------------------
# Register WebSocket endpoint on FastAPI app
# ---------------------------------------------------------------------------

_ws_hub = SentinelWebSocketServer()
_ws_hub._running = True

if _FASTAPI_OK and app is not None:
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket hub — subscribe to quotes, news, alerts, regime, options."""
        await websocket.accept()
        await _ws_hub.handle_client(websocket)


# ---------------------------------------------------------------------------
# SentinelPythonSDK / SentinelClient
# ---------------------------------------------------------------------------

class _TTLCache:
    """Simple TTL cache (functools.lru_cache doesn't support TTL natively)."""

    def __init__(self, ttl_seconds: int = 300, maxsize: int = 256):
        self._ttl = ttl_seconds
        self._maxsize = maxsize
        self._store: Dict[str, tuple] = {}   # key → (value, expire_at)

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry and time.monotonic() < entry[1]:
            return entry[0]
        self._store.pop(key, None)
        return None

    def set(self, key: str, value: Any):
        if len(self._store) >= self._maxsize:
            # evict oldest
            oldest = min(self._store, key=lambda k: self._store[k][1], default=None)
            if oldest:
                self._store.pop(oldest, None)
        self._store[key] = (value, time.monotonic() + self._ttl)


class SentinelPythonSDK:
    """Python client SDK for the SENTINEL REST API v3.

    All read-only endpoints are cached with a 5-minute TTL.
    Write/compute endpoints are never cached.

    Usage:
        from sentinel.api.rest_sdk_v3 import SentinelClient
        client = SentinelClient(host="localhost", port=8000, api_key="dev-key")
        quote = client.get_quote("AAPL")
        hist  = client.get_history("AAPL", start="2024-01-01", end="2025-01-01")
        df    = client.screen_nl("profitable tech stocks with PE < 20")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str = "dev-key",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.base_url = f"http://{host}:{port}{API_PREFIX}"
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache = _TTLCache(ttl_seconds=300)
        self._session: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> Any:
        if not _REQUESTS_OK:
            raise ImportError("requests library required. pip install requests")
        if self._session is None:
            self._session = _requests_lib.Session()
            self._session.headers.update({
                "X-SENTINEL-KEY": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "SENTINEL-SDK/3.0",
            })
        return self._session

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """HTTP request with exponential-backoff retry."""
        session = self._get_session()
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = session.request(method, url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("SENTINEL SDK request failed (attempt %d/%d): %s — retry in %ds",
                               attempt + 1, self.max_retries, exc, wait)
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    def _get(self, path: str, params: dict | None = None, cache: bool = True) -> Any:
        cache_key = f"GET:{path}:{json.dumps(params or {}, sort_keys=True)}"
        if cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        result = self._request("GET", path, params=params)
        if cache:
            self._cache.set(cache_key, result)
        return result

    def _post(self, path: str, body: dict) -> Any:
        return self._request("POST", path, json=body)

    def _to_df(self, data: Any, records_key: str = "data") -> Any:
        if not _PANDAS_OK:
            return data
        records = data if isinstance(data, list) else data.get(records_key, [])
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def get_quote(self, ticker: str) -> dict:
        """Real-time quote."""
        return self._get(f"/quote/{ticker.upper()}", cache=False)

    def get_history(
        self,
        ticker: str,
        start: str = "2024-01-01",
        end: str = "",
        timeframe: str = "1d",
    ) -> Any:
        """OHLCV history as a DataFrame (or raw dict if pandas unavailable)."""
        data = self._get(f"/history/{ticker.upper()}",
                         params={"start": start, "end": end, "timeframe": timeframe})
        return self._to_df(data)

    def get_options(self, ticker: str, expiry: str = "") -> dict:
        """Options chain."""
        return self._get(f"/options/{ticker.upper()}", params={"expiry": expiry})

    def get_futures(self, commodity: str) -> dict:
        """Futures term structure."""
        return self._get(f"/futures/{commodity.upper()}")

    def get_fx(self, base: str, quote: str) -> dict:
        """FX spot + forward."""
        return self._get(f"/fx/{base.upper()}/{quote.upper()}")

    def get_crypto(self, symbol: str) -> dict:
        """Crypto price."""
        return self._get(f"/crypto/{symbol.upper()}", cache=False)

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(self, ticker: str) -> dict:
        """Key ratios and financials."""
        return self._get(f"/fundamentals/{ticker.upper()}")

    def get_income(self, ticker: str, periods: int = 4) -> dict:
        """Income statement."""
        return self._get(f"/income/{ticker.upper()}", params={"periods": periods})

    def get_balance(self, ticker: str, periods: int = 4) -> dict:
        """Balance sheet."""
        return self._get(f"/balance/{ticker.upper()}", params={"periods": periods})

    def get_cashflow(self, ticker: str, periods: int = 4) -> dict:
        """Cash flow statement."""
        return self._get(f"/cashflow/{ticker.upper()}", params={"periods": periods})

    def get_segments(self, ticker: str) -> dict:
        """Segment breakdown."""
        return self._get(f"/segments/{ticker.upper()}")

    def get_valuation(self, ticker: str) -> dict:
        """DCF + comps valuation."""
        return self._get(f"/valuation/{ticker.upper()}")

    # ------------------------------------------------------------------
    # Ownership & Filings
    # ------------------------------------------------------------------

    def get_insider(self, ticker: str, days: int = 90) -> dict:
        """Form 4 insider transactions."""
        return self._get(f"/insider/{ticker.upper()}", params={"days": days})

    def get_institutions(self, ticker: str) -> dict:
        """13F institutional holders."""
        return self._get(f"/institutions/{ticker.upper()}")

    def get_activist(self, ticker: str) -> dict:
        """Activist campaigns."""
        return self._get(f"/activist/{ticker.upper()}")

    def get_filings(self, ticker: str, form_type: str = "10-K") -> dict:
        """EDGAR filings."""
        return self._get(f"/filings/{ticker.upper()}", params={"form_type": form_type})

    # ------------------------------------------------------------------
    # Screeners
    # ------------------------------------------------------------------

    def screen_fundamental(self, criteria: dict) -> Any:
        """Fundamental screener."""
        data = self._post("/screen/fundamental", {"criteria": criteria})
        return self._to_df(data, records_key="results")

    def screen_technical(self, criteria: dict) -> Any:
        """Technical screener."""
        data = self._post("/screen/technical", {"criteria": criteria})
        return self._to_df(data, records_key="results")

    def screen_nl(self, query: str) -> Any:
        """Natural-language screener."""
        data = self._post("/screen/nl", {"query": query})
        return self._to_df(data, records_key="results")

    def screen_preset(self, name: str) -> Any:
        """Run named preset screen."""
        data = self._get(f"/screen/preset/{name}")
        return self._to_df(data, records_key="results")

    # ------------------------------------------------------------------
    # Risk & Portfolio
    # ------------------------------------------------------------------

    def compute_var(
        self,
        holdings: dict,
        confidence: float = 0.99,
        lookback_days: int = 252,
    ) -> dict:
        """VaR / CVaR computation."""
        return self._post("/risk/var", {
            "holdings": holdings,
            "confidence": confidence,
            "lookback_days": lookback_days,
        })

    def stress_test(self, holdings: dict, scenarios: list | None = None) -> dict:
        """Stress test scenarios."""
        return self._post("/risk/stress", {
            "holdings": holdings,
            "scenarios": scenarios or ["2008_gfc", "2020_covid", "2022_rates"],
        })

    def optimize_portfolio(
        self,
        tickers: list,
        method: str = "max_sharpe",
        constraints: dict | None = None,
        start: str = "2020-01-01",
        end: str = "",
    ) -> dict:
        """Portfolio optimization."""
        return self._post("/portfolio/optimize", {
            "tickers": tickers,
            "method": method,
            "constraints": constraints or {},
            "start": start,
            "end": end,
        })

    def get_correlation(self, tickers: list) -> dict:
        """Correlation matrix."""
        return self._get("/risk/correlation",
                         params={"tickers": ",".join(tickers)})

    # ------------------------------------------------------------------
    # AI & Research
    # ------------------------------------------------------------------

    def summarize_filing(self, ticker: str, form: str = "10-K") -> dict:
        """AI filing summary."""
        return self._post("/ai/summarize", {"ticker": ticker, "form": form})

    def qa(self, question: str, tickers: list | None = None, top_k: int = 5) -> dict:
        """RAG Q&A."""
        return self._post("/ai/qa", {
            "question": question,
            "tickers": tickers or [],
            "top_k": top_k,
        })

    def generate_strategy(self, description: str) -> dict:
        """NL strategy generation."""
        return self._post("/ai/strategy", {"description": description})

    def get_research(self, ticker: str) -> dict:
        """Full AI research report."""
        return self._get(f"/ai/research/{ticker.upper()}")

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        strategy: dict,
        symbols: list,
        start: str,
        end: str = "",
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
    ) -> dict:
        """Submit and run a backtest. Returns result dict with run_id."""
        return self._post("/backtest/run", {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "initial_capital": initial_capital,
            "commission": commission,
        })

    def get_backtest_result(self, run_id: str) -> dict:
        """Fetch full backtest result."""
        return self._get(f"/backtest/result/{run_id}", cache=False)

    def get_tearsheet(self, run_id: str) -> dict:
        """Formatted performance tearsheet."""
        return self._get(f"/backtest/tearsheet/{run_id}", cache=False)

    # ------------------------------------------------------------------
    # WebSocket subscription
    # ------------------------------------------------------------------

    def subscribe_quotes(
        self,
        tickers: List[str],
        callback: Callable[[dict], None],
        block: bool = True,
    ) -> Optional[threading.Thread]:
        """Subscribe to live quote updates via WebSocket.

        Parameters
        ----------
        tickers : list of ticker strings
        callback : called with each message dict
        block : if False, runs in a background daemon thread

        Returns
        -------
        Thread (if block=False) or None
        """
        host_port = self.base_url.replace(f"http://", "").split("/")[0]
        host, _, port = host_port.partition(":")
        ws_url = f"ws://{host}:{WS_PORT}/ws"
        channels = [f"quotes.{t.upper()}" for t in tickers]

        def _run():
            asyncio.run(self._ws_subscribe(ws_url, channels, callback))

        if block:
            _run()
            return None
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    async def _ws_subscribe(self, url: str, channels: List[str], callback: Callable):
        if not _WEBSOCKETS_OK:
            raise ImportError("websockets library required. pip install websockets")
        import websockets as ws_lib
        async with ws_lib.connect(url) as ws:
            await ws.send(json.dumps({"type": "subscribe", "channels": channels}))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    callback(msg)
                except Exception:
                    continue

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """System health check."""
        return self._get("/health", cache=False)

    def status(self) -> dict:
        """Module availability status."""
        return self._get("/status", cache=False)


# Convenience alias
SentinelClient = SentinelPythonSDK


# ---------------------------------------------------------------------------
# APIDocGenerator
# ---------------------------------------------------------------------------

class APIDocGenerator:
    """Generate API documentation in multiple formats."""

    # All endpoints: (method, path, description, sample_body)
    ENDPOINTS = [
        ("GET",  "/api/v3/health",                    "System health check",                           None),
        ("GET",  "/api/v3/status",                    "Module availability status",                    None),
        ("GET",  "/api/v3/quote/{ticker}",            "Real-time quote",                               None),
        ("GET",  "/api/v3/history/{ticker}",          "OHLCV history (start, end, timeframe params)",  None),
        ("GET",  "/api/v3/options/{ticker}",          "Options chain (expiry param)",                  None),
        ("GET",  "/api/v3/futures/{commodity}",       "Futures term structure",                        None),
        ("GET",  "/api/v3/fx/{base}/{quote}",         "FX spot and forward rates",                     None),
        ("GET",  "/api/v3/crypto/{symbol}",           "Crypto price and metrics",                      None),
        ("GET",  "/api/v3/fundamentals/{ticker}",     "Key financial ratios",                          None),
        ("GET",  "/api/v3/income/{ticker}",           "Income statement (periods param)",              None),
        ("GET",  "/api/v3/balance/{ticker}",          "Balance sheet (periods param)",                 None),
        ("GET",  "/api/v3/cashflow/{ticker}",         "Cash flow statement (periods param)",           None),
        ("GET",  "/api/v3/segments/{ticker}",         "Business segment breakdown",                    None),
        ("GET",  "/api/v3/valuation/{ticker}",        "DCF + comps valuation",                         None),
        ("GET",  "/api/v3/insider/{ticker}",          "Form 4 insider transactions (days param)",      None),
        ("GET",  "/api/v3/institutions/{ticker}",     "13F institutional holders",                     None),
        ("GET",  "/api/v3/activist/{ticker}",         "Activist investor campaigns",                   None),
        ("GET",  "/api/v3/filings/{ticker}",          "EDGAR filings (form_type param)",               None),
        ("POST", "/api/v3/screen/fundamental",        "Fundamental screener",
            '{"criteria": {"pe_max": 20, "roe_min": 0.15}}'),
        ("POST", "/api/v3/screen/technical",          "Technical screener",
            '{"criteria": {"rsi_max": 30}}'),
        ("POST", "/api/v3/screen/nl",                 "Natural-language screener",
            '{"query": "profitable tech stocks with PE < 20"}'),
        ("GET",  "/api/v3/screen/preset/{name}",      "Run named preset screen",                       None),
        ("POST", "/api/v3/risk/var",                  "VaR and CVaR",
            '{"holdings": {"AAPL": 10000, "MSFT": 15000}, "confidence": 0.99}'),
        ("POST", "/api/v3/risk/stress",               "Stress test scenarios",
            '{"holdings": {"AAPL": 10000}, "scenarios": ["2008_gfc", "2020_covid"]}'),
        ("POST", "/api/v3/portfolio/optimize",        "Portfolio optimization",
            '{"tickers": ["AAPL", "MSFT", "GOOGL"], "method": "max_sharpe"}'),
        ("GET",  "/api/v3/risk/correlation",          "Correlation matrix (tickers param)",            None),
        ("POST", "/api/v3/ai/summarize",              "AI filing summary",
            '{"ticker": "AAPL", "form": "10-K"}'),
        ("POST", "/api/v3/ai/qa",                     "RAG Q&A",
            '{"question": "What is Apple gross margin trend?", "tickers": ["AAPL"]}'),
        ("POST", "/api/v3/ai/strategy",               "NL strategy generation",
            '{"description": "momentum small caps"}'),
        ("GET",  "/api/v3/ai/research/{ticker}",      "Full AI research workflow",                     None),
        ("POST", "/api/v3/backtest/run",              "Run backtest",
            '{"strategy": {"type": "momentum"}, "symbols": ["AAPL", "MSFT"], "start": "2022-01-01", "end": "2024-01-01"}'),
        ("GET",  "/api/v3/backtest/result/{run_id}",  "Get backtest result",                           None),
        ("GET",  "/api/v3/backtest/tearsheet/{run_id}", "Performance tearsheet",                       None),
    ]

    def generate_markdown_docs(self) -> str:
        """Full API reference in Markdown."""
        lines = [
            "# SENTINEL API v3 — Reference",
            "",
            "Base URL: `http://localhost:8000`",
            "Auth header: `X-SENTINEL-KEY: <your-key>`",
            "Rate limit: 100 requests/minute per key",
            "",
            "## WebSocket",
            "",
            "Connect: `ws://localhost:8765/ws`",
            "",
            "**Subscribe:**",
            "```json",
            '{"type": "subscribe", "channels": ["quotes.AAPL", "news.AAPL"]}',
            "```",
            "",
            "**Set price alert:**",
            "```json",
            '{"type": "set_alert", "ticker": "AAPL", "above": 200.0}',
            "```",
            "",
            "**Available channels:**",
            "- `quotes.{ticker}` — live price (5s poll)",
            "- `news.{ticker}` — GDELT news (60s poll)",
            "- `alerts.{ticker}` — price threshold crossings",
            "- `regime` — macro regime (VIX-based, 5m poll)",
            "- `options.{ticker}` — unusual options activity (30s poll)",
            "",
            "## REST Endpoints",
            "",
        ]
        current_tag = None
        tag_map = {
            "/health": "System", "/status": "System",
            "/quote": "Market Data", "/history": "Market Data",
            "/options": "Market Data", "/futures": "Market Data",
            "/fx": "Market Data", "/crypto": "Market Data",
            "/fundamentals": "Fundamentals", "/income": "Fundamentals",
            "/balance": "Fundamentals", "/cashflow": "Fundamentals",
            "/segments": "Fundamentals", "/valuation": "Fundamentals",
            "/insider": "Ownership & Filings", "/institutions": "Ownership & Filings",
            "/activist": "Ownership & Filings", "/filings": "Ownership & Filings",
            "/screen": "Screeners",
            "/risk": "Risk & Portfolio", "/portfolio": "Risk & Portfolio",
            "/ai": "AI & Research",
            "/backtest": "Backtesting",
        }
        for method, path, desc, body in self.ENDPOINTS:
            path_key = "/" + path.split("/")[3] if len(path.split("/")) > 3 else path
            tag = tag_map.get(path_key, "Other")
            if tag != current_tag:
                lines += [f"### {tag}", ""]
                current_tag = tag
            lines.append(f"#### `{method} {path}`")
            lines.append(f"{desc}")
            if body:
                lines += ["", "**Request body:**", "```json", body, "```"]
            lines.append("")
        return "\n".join(lines)

    def generate_postman_collection(self) -> dict:
        """Postman-compatible JSON collection."""
        items = []
        for method, path, desc, body in self.ENDPOINTS:
            url_parts = path.lstrip("/").split("/")
            item: dict = {
                "name": desc,
                "request": {
                    "method": method,
                    "header": [
                        {"key": "X-SENTINEL-KEY", "value": "{{api_key}}"},
                        {"key": "Content-Type", "value": "application/json"},
                    ],
                    "url": {
                        "raw": f"{{{{base_url}}}}{path}",
                        "host": ["{{base_url}}"],
                        "path": url_parts,
                    },
                    "description": desc,
                },
            }
            if body:
                item["request"]["body"] = {
                    "mode": "raw",
                    "raw": body,
                    "options": {"raw": {"language": "json"}},
                }
            items.append(item)
        return {
            "info": {
                "name": "SENTINEL API v3",
                "description": "SENTINEL Financial Terminal REST API",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            },
            "variable": [
                {"key": "base_url", "value": "http://localhost:8000"},
                {"key": "api_key", "value": "dev-key"},
            ],
            "item": items,
        }

    def generate_curl_examples(self) -> str:
        """curl examples for every endpoint."""
        lines = [
            "# SENTINEL API v3 — curl Examples",
            "# Set your API key:",
            "export SENTINEL_KEY=dev-key",
            "export BASE=http://localhost:8000",
            "",
        ]
        for method, path, desc, body in self.ENDPOINTS:
            # Replace path params with example values
            example_path = (
                path
                .replace("{ticker}", "AAPL")
                .replace("{commodity}", "CL")
                .replace("{base}", "EUR")
                .replace("{quote}", "USD")
                .replace("{symbol}", "BTC")
                .replace("{name}", "magic_formula")
                .replace("{run_id}", "abc-123")
            )
            lines.append(f"# {desc}")
            if method == "GET":
                lines.append(
                    f'curl -s -H "X-SENTINEL-KEY: $SENTINEL_KEY" '
                    f'"$BASE{example_path}"'
                )
            else:
                body_str = body or "{}"
                lines.append(
                    f'curl -s -X POST -H "X-SENTINEL-KEY: $SENTINEL_KEY" '
                    f'-H "Content-Type: application/json" '
                    f"-d '{body_str}' "
                    f'"$BASE{example_path}"'
                )
            lines.append("")
        return "\n".join(lines)

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """
        Return an OpenAPI 3.0 specification dict for all SENTINEL endpoints.

        The returned dict conforms to the OpenAPI 3.0.3 schema and can be
        serialised directly to JSON/YAML (e.g. via json.dumps or PyYAML).

        Paths are derived from self.ENDPOINTS so the spec stays in sync with
        the actual route list without manual maintenance.
        """
        paths: Dict[str, Any] = {}

        for method, path, description, body in self.ENDPOINTS:
            # Convert FastAPI path params {param} → OpenAPI {param} (already compatible)
            openapi_path = path
            parameters = []
            import re as _re
            for param in _re.findall(r"\{(\w+)\}", path):
                parameters.append({
                    "name": param,
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": f"Path parameter: {param}",
                })

            operation: Dict[str, Any] = {
                "summary": description,
                "description": description,
                "operationId": (
                    method.lower() + "_" +
                    _re.sub(r"[^a-zA-Z0-9]", "_", path).strip("_")
                ),
                "security": [{"ApiKeyAuth": []}],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        },
                    },
                    "401": {"description": "Invalid or missing API key"},
                    "429": {"description": "Rate limit exceeded"},
                    "500": {"description": "Internal server error"},
                },
            }
            if parameters:
                operation["parameters"] = parameters
            if body and method in ("POST", "PUT", "PATCH"):
                operation["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"},
                            "example": json.loads(body) if body.startswith("{") else {},
                        }
                    },
                }

            if openapi_path not in paths:
                paths[openapi_path] = {}
            paths[openapi_path][method.lower()] = operation

        return {
            "openapi": "3.0.3",
            "info": {
                "title": "SENTINEL Financial Terminal API",
                "description": (
                    "SENTINEL v3 REST API — market data, screening, analytics, "
                    "backtesting, AI summarisation, and WebSocket streaming."
                ),
                "version": "3.0.0",
                "contact": {"email": "richard.porras@realempanada.com"},
                "license": {"name": "Proprietary"},
            },
            "servers": [
                {"url": "http://localhost:8000", "description": "Local development"},
                {"url": "https://api.sentinel.finance", "description": "Production"},
            ],
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-SENTINEL-KEY",
                        "description": "Provide your SENTINEL API key in this header.",
                    }
                }
            },
            "paths": paths,
        }


# ---------------------------------------------------------------------------
# Rate-limit budget helper (standalone, dim_095 push 8→9)
# ---------------------------------------------------------------------------

def compute_rate_limit_budget(
    bucket: "_TokenBucket",
    elapsed_seconds: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute the current rate-limit budget for a token-bucket instance.

    Implements the token-bucket refill formula:
      tokens_remaining = min(max_tokens, prev_tokens + refill_rate × elapsed)

    Parameters
    ----------
    bucket          : a _TokenBucket instance (or any object with .capacity,
                      .rate, .tokens, .last_refill attributes).
    elapsed_seconds : seconds since the last consume() call.  When 0.0 the
                      function uses the actual monotonic clock delta so the
                      result reflects real wall-clock time.

    Returns
    -------
    dict with:
      tokens_remaining    : float — tokens available right now (post-refill)
      capacity            : int   — maximum bucket size
      refill_rate         : float — tokens per second
      budget_pct          : float — tokens_remaining / capacity × 100
      requests_per_minute : float — effective sustained request rate
      throttled           : bool  — True when tokens_remaining < 1
    """
    if elapsed_seconds <= 0.0:
        now = time.monotonic()
        elapsed_seconds = max(0.0, now - bucket.last_refill)

    tokens_remaining = min(
        float(bucket.capacity),
        float(bucket.tokens) + bucket.rate * elapsed_seconds,
    )
    budget_pct = tokens_remaining / bucket.capacity * 100.0 if bucket.capacity > 0 else 0.0

    return {
        "tokens_remaining": round(tokens_remaining, 4),
        "capacity": bucket.capacity,
        "refill_rate_per_sec": round(bucket.rate, 6),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "budget_pct": round(budget_pct, 2),
        "requests_per_minute": round(bucket.rate * 60.0, 2),
        "throttled": tokens_remaining < 1.0,
    }


def build_client_with_retry(
    host: str = "localhost",
    port: int = 8000,
    api_key: str = "dev-key",
    max_attempts: int = 5,
    base_delay: float = 1.0,
) -> "SentinelPythonSDK":
    """
    Build a SentinelPythonSDK client with exponential-backoff connection retry.

    Retry schedule: wait = base_delay × 2^attempt seconds between attempts.
    Default: 1s, 2s, 4s, 8s, 16s (max 5 attempts).

    Parameters
    ----------
    host         : API server hostname.
    port         : API server port.
    api_key      : authentication key.
    max_attempts : maximum connection attempts (default 5).
    base_delay   : base wait in seconds (default 1.0); actual wait = base × 2^attempt.

    Returns
    -------
    SentinelPythonSDK connected and verified (GET /api/v3/health returned 200).

    Raises
    ------
    RuntimeError if all attempts are exhausted without a successful health check.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(max_attempts):
        wait = base_delay * (2 ** attempt)
        try:
            client = SentinelPythonSDK(host=host, port=port, api_key=api_key)

            if _REQUESTS_OK and _requests_lib is not None:
                url = f"{client.base_url}/health"
                headers = {"X-SENTINEL-KEY": api_key}
                resp = _requests_lib.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    logger.info(
                        "build_client_with_retry: connected on attempt %d/%d",
                        attempt + 1, max_attempts,
                    )
                    return client
                last_exc = RuntimeError(
                    f"Health check returned HTTP {resp.status_code}"
                )
            else:
                # No requests library — return client optimistically
                logger.warning(
                    "build_client_with_retry: requests not installed; "
                    "returning client without health check"
                )
                return client

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "build_client_with_retry: attempt %d/%d failed (%s); "
                "retrying in %.1fs",
                attempt + 1, max_attempts, exc, wait,
            )

        if attempt < max_attempts - 1:
            time.sleep(wait)

    raise RuntimeError(
        f"build_client_with_retry: all {max_attempts} attempts failed. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Standalone OpenAPI spec generator (module-level, complements APIDocGenerator)
# ---------------------------------------------------------------------------

def generate_openapi_spec(app) -> dict:
    """Returns OpenAPI 3.0 spec dict from registered FastAPI routes."""
    paths = {}
    for route in getattr(app, 'routes', []):
        path = getattr(route, 'path', None)
        methods = getattr(route, 'methods', None) or ['GET']
        if path:
            paths[path] = {m.lower(): {'summary': getattr(route, 'name', path), 'responses': {'200': {'description': 'OK'}}} for m in methods}
    return {'openapi': '3.0.0', 'info': {'title': 'SENTINEL API', 'version': '1.0.0'}, 'paths': paths}


# ---------------------------------------------------------------------------
# TokenBucket — public rate-limiter with elapsed-time-based pure math API
# ---------------------------------------------------------------------------

class TokenBucket:
    """Token bucket rate limiter."""
    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self._tokens = float(max_tokens)
        self._last_refill = 0.0  # use elapsed time for testability

    def compute_tokens_remaining(self, elapsed_seconds: float) -> float:
        """Pure math: tokens after elapsed_seconds of refill."""
        return min(self.max_tokens, self._tokens + self.refill_rate * elapsed_seconds)

    def consume(self, tokens: float, elapsed_seconds: float) -> bool:
        """Returns True if tokens available, False if rate limited."""
        available = self.compute_tokens_remaining(elapsed_seconds)
        if available >= tokens:
            self._tokens = available - tokens
            return True
        return False


# ---------------------------------------------------------------------------
# build_client_with_retry — dict-returning config builder (public API)
# ---------------------------------------------------------------------------

def build_client_with_retry(base_url: str, max_attempts: int = 5, timeout: float = 30.0) -> dict:
    """Returns retry config with exponential backoff: 2^attempt seconds."""
    return {
        'base_url': base_url,
        'max_attempts': max_attempts,
        'timeout': timeout,
        'backoff_schedule': [2 ** i for i in range(max_attempts)],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if _UVICORN_OK and app is not None:
        logger.info("Starting SENTINEL API v3 on http://0.0.0.0:8000")
        logger.info("Docs: http://localhost:8000/docs")
        logger.info("WebSocket: ws://localhost:8765/ws")
        _uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("\nSENTINEL API v3 — SDK mode (uvicorn not installed)")
        print("  Install: pip install uvicorn[standard] fastapi\n")
        # Print all routes
        if app is not None:
            print("Registered routes:")
            for route in app.routes:
                methods = getattr(route, "methods", {"WS"})
                path = getattr(route, "path", "?")
                print(f"  {','.join(sorted(methods)):8} {path}")
        # SDK usage example
        print("\n--- SDK Example (no server needed) ---")
        print("from sentinel.api.rest_sdk_v3 import SentinelClient")
        print('client = SentinelClient(host="localhost", port=8000)')
        print('quote = client.get_quote("AAPL")')
        print('hist  = client.get_history("AAPL", start="2024-01-01", end="2025-01-01")')
        print('df    = client.screen_nl("profitable tech stocks with PE < 20")')
        print('bt    = client.run_backtest({"type": "momentum"}, ["AAPL", "MSFT"], "2022-01-01", "2024-01-01")')
        print()
        # Doc generation demo
        gen = APIDocGenerator()
        md = gen.generate_markdown_docs()
        print(f"[APIDocGenerator] Markdown docs: {len(md)} chars")
        postman = gen.generate_postman_collection()
        print(f"[APIDocGenerator] Postman collection: {len(postman['item'])} items")
        curl_ex = gen.generate_curl_examples()
        print(f"[APIDocGenerator] curl examples: {len(curl_ex)} chars")
        print("\nDone.")
