"""SENTINEL Python SDK — Dimension #095 (target 9+).

Provides sync, async, and WebSocket clients for the SENTINEL API,
plus DataFrame serialization helpers and a command-line interface.

Quick start
-----------
    from sentinel.api.sdk import SentinelClient

    client = SentinelClient(base_url="http://localhost:8000", api_key="sk-...")

    quote  = client.get_quote("AAPL")
    hist   = client.get_history("AAPL", start="2024-01-01", end="2025-01-01")
    screen = client.screen_fundamental({"pe_max": 20, "roe_min": 0.15})

Async
-----
    from sentinel.api.sdk import AsyncSentinelClient

    async with AsyncSentinelClient() as client:
        quote = await client.get_quote("AAPL")

WebSocket streaming
-------------------
    from sentinel.api.sdk import SentinelWebSocketClient

    def on_quote(msg):
        print(msg["ticker"], msg["data"]["last"])

    ws = SentinelWebSocketClient()
    ws.subscribe_quotes(["AAPL", "MSFT"], callback=on_quote)
    ws.start()      # blocks — use ws.start(block=False) for background thread

CLI
---
    sentinel quote AAPL
    sentinel history AAPL --start 2024-01-01 --output prices.csv
    sentinel screen --preset magic_formula --output results.csv
    sentinel ask "What is Apple's gross margin trend?" --ticker AAPL
    sentinel financials AAPL --statement income --periods 5
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — gracefully absent so the SDK can be imported
# in environments that only have the stdlib.
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    _PD_OK = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PD_OK = False

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    _HTTPX_OK = False

try:
    import websockets
    _WS_OK = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    _WS_OK = False

try:
    import openpyxl  # noqa: F401 — needed for pd.ExcelWriter
    _EXCEL_OK = True
except ImportError:
    _EXCEL_OK = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SentinelAPIError(Exception):
    """Raised when the SENTINEL API returns a non-2xx HTTP status."""

    def __init__(self, message: str, status_code: int = 0, endpoint: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message     = message
        self.endpoint    = endpoint

    def __str__(self) -> str:
        return f"[{self.status_code}] {self.endpoint} — {self.message}"


# ---------------------------------------------------------------------------
# DataFrameSerializer
# ---------------------------------------------------------------------------

class DataFrameSerializer:
    """Convert DataFrames and dicts to various on-disk or in-memory formats.

    All methods are synchronous and do not depend on a live API connection.
    """

    @staticmethod
    def to_csv(df: "pd.DataFrame", path: str = None) -> str:
        """Serialize DataFrame to CSV.

        Args:
            df:   Input DataFrame.
            path: If provided, write to file and return the path; otherwise
                  return the CSV as a string.

        Returns:
            CSV string when path is None, else the file path.
        """
        if not _PD_OK:
            raise RuntimeError("pandas is required for DataFrameSerializer")

        buf = io.StringIO()
        df.to_csv(buf)
        csv_str = buf.getvalue()

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(csv_str)
            return path
        return csv_str

    @staticmethod
    def to_parquet(df: "pd.DataFrame", path: str) -> str:
        """Save DataFrame to Parquet format.

        Args:
            df:   Input DataFrame.
            path: Destination file path (must end in .parquet).

        Returns:
            The file path.
        """
        if not _PD_OK:
            raise RuntimeError("pandas is required for DataFrameSerializer")
        df.to_parquet(path, index=True)
        return path

    @staticmethod
    def to_excel(data: dict[str, "pd.DataFrame"], path: str) -> str:
        """Write multiple DataFrames to separate sheets in an Excel file.

        Typical usage for financials::

            serializer.to_excel(
                {"Income Statement": is_df, "Balance Sheet": bs_df, "Cash Flow": cf_df},
                path="AAPL_financials.xlsx",
            )

        Args:
            data: Mapping of sheet name → DataFrame.
            path: Destination .xlsx file path.

        Returns:
            The file path.
        """
        if not _PD_OK:
            raise RuntimeError("pandas is required for DataFrameSerializer")
        if not _EXCEL_OK:
            raise RuntimeError("openpyxl is required: pip install openpyxl")

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name[:31])  # Excel 31-char limit
        return path

    @staticmethod
    def to_json(df: "pd.DataFrame") -> str:
        """Serialize DataFrame to JSON records string.

        Returns:
            JSON array string, one record per row.
        """
        if not _PD_OK:
            raise RuntimeError("pandas is required for DataFrameSerializer")
        return df.to_json(orient="records", date_format="iso", indent=2)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class SentinelClient:
    """Synchronous REST client for the SENTINEL API.

    Uses httpx under the hood (requests-compatible interface).
    Thread-safe: each method creates a short-lived httpx.Client.

    Args:
        base_url: Base URL of the SENTINEL API (default: http://localhost:8000).
        api_key:  Bearer token for authenticated endpoints.
        timeout:  Request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        if not _HTTPX_OK:
            raise RuntimeError("httpx is required: pip install httpx")

        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.timeout  = timeout
        self.serializer = DataFrameSerializer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params, headers=self._headers())
        if resp.status_code >= 400:
            raise SentinelAPIError(
                message=resp.text or resp.reason_phrase,
                status_code=resp.status_code,
                endpoint=path,
            )
        return resp.json()

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=body, headers=self._headers())
        if resp.status_code >= 400:
            raise SentinelAPIError(
                message=resp.text or resp.reason_phrase,
                status_code=resp.status_code,
                endpoint=path,
            )
        return resp.json()

    @staticmethod
    def _to_df(data: Any, index_col: Optional[str] = None) -> "pd.DataFrame":
        if not _PD_OK:
            return data
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data]) if not any(isinstance(v, list) for v in data.values()) \
                 else pd.DataFrame(data)
        else:
            return pd.DataFrame()
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        return df

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def get_quote(self, ticker: str) -> dict:
        """Fetch a real-time Level 1 quote for a single ticker.

        Returns:
            dict with keys: ticker, last, bid, ask, change, change_pct,
            volume, vwap, market_cap, session, source.
        """
        return self._get(f"/api/quotes/{ticker.upper()}")

    def get_bulk_quotes(self, tickers: list[str]) -> dict:
        """Fetch quotes for multiple tickers in a single request.

        Args:
            tickers: List of ticker symbols (max 100).

        Returns:
            dict: {"count": N, "quotes": {AAPL: {...}, ...}, "as_of": "..."}
        """
        symbols = ",".join(t.upper() for t in tickers)
        return self._get("/api/quotes", params={"symbols": symbols})

    def get_history(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> "pd.DataFrame":
        """Fetch OHLCV history for a ticker.

        Args:
            ticker:   Symbol (e.g. "AAPL").
            start:    Start date "YYYY-MM-DD".
            end:      End date "YYYY-MM-DD" (default: today).
            interval: Bar interval: 1d|1wk|1mo|1h|30m|15m|5m|1m.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume,
            indexed by date.
        """
        params: dict = {"start": start, "interval": interval}
        if end:
            params["end"] = end
        data = self._get(f"/v1/ohlcv/{ticker.upper()}", params=params)
        if not _PD_OK or not isinstance(data, dict):
            return data
        dates = data.get("date", [])
        df = pd.DataFrame({
            "Date":   dates,
            "Open":   data.get("open",   []),
            "High":   data.get("high",   []),
            "Low":    data.get("low",    []),
            "Close":  data.get("close",  []),
            "Volume": data.get("volume", []),
        })
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")

    def get_options_chain(self, ticker: str, expiry: Optional[str] = None) -> dict:
        """Fetch near-term options chain summary.

        Args:
            ticker: Symbol.
            expiry: Specific expiry date "YYYY-MM-DD" (default: nearest).

        Returns:
            dict: atm_iv, put_call_ratio, max_pain_strike, call_volume, put_volume.
        """
        params = {}
        if expiry:
            params["expiry"] = expiry
        return self._get(f"/api/quotes/{ticker.upper()}/options-summary", params=params)

    # ------------------------------------------------------------------
    # Financials
    # ------------------------------------------------------------------

    def _get_statement(self, ticker: str, statement: str, periods: int) -> "pd.DataFrame":
        params = {"statement": statement, "period": "annual"}
        data = self._get(f"/v1/financials/{ticker.upper()}", params=params)
        if not _PD_OK or not isinstance(data, dict):
            return data
        dates_raw = data.get("dates", [])
        records   = data.get("data", {})
        df = pd.DataFrame.from_dict(records, orient="index")
        df.index = pd.to_datetime(dates_raw) if dates_raw else df.index
        return df.iloc[:, :periods] if periods < len(df.columns) else df

    def get_income_statement(self, ticker: str, periods: int = 5) -> "pd.DataFrame":
        """Fetch annual income statement.

        Returns:
            DataFrame with financial line items as columns and dates as rows.
        """
        return self._get_statement(ticker, "income", periods)

    def get_balance_sheet(self, ticker: str, periods: int = 5) -> "pd.DataFrame":
        """Fetch annual balance sheet."""
        return self._get_statement(ticker, "balance", periods)

    def get_cash_flow(self, ticker: str, periods: int = 5) -> "pd.DataFrame":
        """Fetch annual cash flow statement."""
        return self._get_statement(ticker, "cashflow", periods)

    def get_ratios(self, ticker: str) -> dict:
        """Fetch key financial ratios and valuation metrics.

        Returns:
            dict: P/E, P/B, EV/EBITDA, ROE, margins, debt ratios, etc.
        """
        return self._get(f"/v1/fundamentals/{ticker.upper()}")

    # ------------------------------------------------------------------
    # Screeners
    # ------------------------------------------------------------------

    def screen_fundamental(self, criteria: dict) -> "pd.DataFrame":
        """Run a fundamental screener with custom criteria.

        Args:
            criteria: dict of filter rules, e.g.:
                {"pe_max": 20, "roe_min": 0.15, "market_cap_min": 1e9}

        Returns:
            DataFrame of matching stocks with key metrics.
        """
        data = self._post("/api/v1/screen/fundamental", body={"criteria": criteria})
        return self._to_df(data.get("results", data))

    def screen_technical(self, criteria: dict) -> "pd.DataFrame":
        """Run a technical screener.

        Args:
            criteria: dict, e.g. {"rsi_max": 30, "above_200ma": True}

        Returns:
            DataFrame of matching stocks.
        """
        data = self._post("/api/v1/screen/technical", body={"criteria": criteria})
        return self._to_df(data.get("results", data))

    def run_preset_screen(self, name: str, type: str = "fundamental") -> "pd.DataFrame":
        """Run a named preset screen.

        Built-in presets:
            magic_formula, deep_value, dividend_aristocrats,
            momentum_52w, growth_at_value, low_vol_quality

        Args:
            name: Preset name.
            type: "fundamental" or "technical".

        Returns:
            DataFrame of matching stocks.
        """
        data = self._get(f"/api/v1/screen/preset/{name}", params={"type": type})
        return self._to_df(data.get("results", data))

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_ownership(self, ticker: str) -> dict:
        """Fetch institutional and insider ownership breakdown.

        Returns:
            dict: top_holders, institutional_pct, insider_pct, short_interest.
        """
        return self._get(f"/api/v1/intelligence/ownership/{ticker.upper()}")

    def get_insider_trades(self, ticker: str, lookback_days: int = 90) -> "pd.DataFrame":
        """Fetch insider Form 4 filings.

        Args:
            ticker:        Symbol.
            lookback_days: How many days back to look (default: 90).

        Returns:
            DataFrame with columns: date, insider, title, transaction, shares, value.
        """
        params = {"lookback_days": lookback_days}
        data   = self._get(f"/api/v1/intelligence/insider-trades/{ticker.upper()}", params=params)
        return self._to_df(data.get("trades", data))

    def get_congressional_trades(self, ticker: str) -> "pd.DataFrame":
        """Fetch congressional trading disclosures (STOCK Act filings).

        Returns:
            DataFrame: date, member, transaction, amount_range, party.
        """
        data = self._get(f"/api/v1/intelligence/congressional-trades/{ticker.upper()}")
        return self._to_df(data.get("trades", data))

    def get_sentiment(self, ticker: str) -> dict:
        """Fetch aggregated sentiment score (news + social).

        Returns:
            dict: score [-1, 1], label, news_articles, reddit_mentions.
        """
        return self._get(f"/api/v1/intelligence/sentiment/{ticker.upper()}")

    def get_governance_score(self, ticker: str) -> dict:
        """Fetch governance / ESG score.

        Returns:
            dict: overall_score, board_score, audit_score, shareholder_rights_score.
        """
        return self._get(f"/api/v1/intelligence/governance/{ticker.upper()}")

    # ------------------------------------------------------------------
    # AI / NLP
    # ------------------------------------------------------------------

    def ask(self, question: str, ticker: Optional[str] = None) -> dict:
        """Ask a natural-language question about a company or market.

        Uses SENTINEL's RAG pipeline over SEC filings and financial data.

        Args:
            question: Free-text question, e.g. "What is Apple's main revenue risk?"
            ticker:   Optional ticker context (narrows retrieval).

        Returns:
            dict: {"answer": "...", "sources": [...], "confidence": 0.0-1.0}
        """
        body: dict = {"question": question}
        if ticker:
            body["ticker"] = ticker.upper()
        return self._post("/api/v1/intelligence/ask", body=body)

    def summarize_filing(self, ticker: str, form_type: str = "10-K") -> dict:
        """Summarize the most recent SEC filing for a ticker.

        Args:
            ticker:    Symbol.
            form_type: SEC form: 10-K|10-Q|8-K|DEF 14A (default: 10-K).

        Returns:
            dict: {"summary": "...", "key_risks": [...], "highlights": [...]}
        """
        params = {"form_type": form_type}
        return self._get(f"/api/v1/intelligence/summarize/{ticker.upper()}", params=params)

    # ------------------------------------------------------------------
    # Paper Trading
    # ------------------------------------------------------------------

    def get_positions(self) -> "pd.DataFrame":
        """Fetch current paper-trading positions.

        Returns:
            DataFrame: ticker, qty, avg_price, current_price, pnl, pnl_pct.
        """
        data = self._get("/api/v1/orders/positions")
        return self._to_df(data.get("positions", data))

    def submit_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        """Submit a paper-trading order.

        Args:
            ticker:      Symbol.
            qty:         Number of shares (fractional allowed).
            side:        "buy" or "sell".
            order_type:  "market" or "limit" (default: "market").
            limit_price: Required when order_type="limit".

        Returns:
            dict: order_id, status, filled_qty, filled_price, timestamp.
        """
        body: dict = {
            "ticker":     ticker.upper(),
            "qty":        qty,
            "side":       side.lower(),
            "order_type": order_type,
        }
        if limit_price is not None:
            body["limit_price"] = limit_price
        return self._post("/api/v1/orders/submit", body=body)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class AsyncSentinelClient:
    """Async REST client using httpx.AsyncClient with connection pooling.

    Usage::

        async with AsyncSentinelClient(base_url="http://localhost:8000") as client:
            quote = await client.get_quote("AAPL")
            hist  = await client.get_history("AAPL", start="2024-01-01")

    All methods mirror :class:`SentinelClient` but are ``async def``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key:  Optional[str] = None,
        timeout:  int = 30,
    ) -> None:
        if not _HTTPX_OK:
            raise RuntimeError("httpx is required: pip install httpx")

        self.base_url   = base_url.rstrip("/")
        self.api_key    = api_key
        self.timeout    = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self.serializer = DataFrameSerializer()

    async def __aenter__(self) -> "AsyncSentinelClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._headers(),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        client = self._client
        standalone = False
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._headers(),
            )
            standalone = True
        try:
            resp = await client.get(path, params=params)
        finally:
            if standalone:
                await client.aclose()

        if resp.status_code >= 400:
            raise SentinelAPIError(
                message=resp.text or str(resp.status_code),
                status_code=resp.status_code,
                endpoint=path,
            )
        return resp.json()

    async def _post(self, path: str, body: dict) -> Any:
        client = self._client
        standalone = False
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._headers(),
            )
            standalone = True
        try:
            resp = await client.post(path, json=body)
        finally:
            if standalone:
                await client.aclose()

        if resp.status_code >= 400:
            raise SentinelAPIError(
                message=resp.text or str(resp.status_code),
                status_code=resp.status_code,
                endpoint=path,
            )
        return resp.json()

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    async def get_quote(self, ticker: str) -> dict:
        """Async: fetch Level 1 quote."""
        return await self._get(f"/api/quotes/{ticker.upper()}")

    async def get_bulk_quotes(self, tickers: list[str]) -> dict:
        """Async: fetch quotes for multiple tickers."""
        symbols = ",".join(t.upper() for t in tickers)
        return await self._get("/api/quotes", params={"symbols": symbols})

    async def get_history(
        self, ticker: str, start: str, end: Optional[str] = None, interval: str = "1d"
    ) -> "pd.DataFrame":
        """Async: fetch OHLCV history."""
        params: dict = {"start": start, "interval": interval}
        if end:
            params["end"] = end
        data = await self._get(f"/v1/ohlcv/{ticker.upper()}", params=params)
        if not _PD_OK or not isinstance(data, dict):
            return data
        df = pd.DataFrame({
            "Date":   data.get("date", []),
            "Open":   data.get("open", []),
            "High":   data.get("high", []),
            "Low":    data.get("low", []),
            "Close":  data.get("close", []),
            "Volume": data.get("volume", []),
        })
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")

    async def get_options_chain(self, ticker: str, expiry: Optional[str] = None) -> dict:
        """Async: fetch options chain summary."""
        params = {"expiry": expiry} if expiry else {}
        return await self._get(f"/api/quotes/{ticker.upper()}/options-summary", params=params)

    async def get_income_statement(self, ticker: str, periods: int = 5) -> "pd.DataFrame":
        """Async: fetch income statement."""
        data = await self._get(f"/v1/financials/{ticker.upper()}", params={"statement": "income"})
        if not _PD_OK:
            return data
        return pd.DataFrame.from_dict(data.get("data", {}), orient="index")

    async def get_balance_sheet(self, ticker: str, periods: int = 5) -> "pd.DataFrame":
        """Async: fetch balance sheet."""
        data = await self._get(f"/v1/financials/{ticker.upper()}", params={"statement": "balance"})
        if not _PD_OK:
            return data
        return pd.DataFrame.from_dict(data.get("data", {}), orient="index")

    async def get_cash_flow(self, ticker: str, periods: int = 5) -> "pd.DataFrame":
        """Async: fetch cash flow statement."""
        data = await self._get(f"/v1/financials/{ticker.upper()}", params={"statement": "cashflow"})
        if not _PD_OK:
            return data
        return pd.DataFrame.from_dict(data.get("data", {}), orient="index")

    async def get_ratios(self, ticker: str) -> dict:
        """Async: fetch key financial ratios."""
        return await self._get(f"/v1/fundamentals/{ticker.upper()}")

    async def screen_fundamental(self, criteria: dict) -> "pd.DataFrame":
        """Async: run fundamental screener."""
        data = await self._post("/api/v1/screen/fundamental", body={"criteria": criteria})
        if not _PD_OK:
            return data
        return pd.DataFrame(data.get("results", data))

    async def screen_technical(self, criteria: dict) -> "pd.DataFrame":
        """Async: run technical screener."""
        data = await self._post("/api/v1/screen/technical", body={"criteria": criteria})
        if not _PD_OK:
            return data
        return pd.DataFrame(data.get("results", data))

    async def run_preset_screen(self, name: str, type: str = "fundamental") -> "pd.DataFrame":
        """Async: run a named preset screen."""
        data = await self._get(f"/api/v1/screen/preset/{name}", params={"type": type})
        if not _PD_OK:
            return data
        return pd.DataFrame(data.get("results", data))

    async def get_ownership(self, ticker: str) -> dict:
        """Async: fetch ownership data."""
        return await self._get(f"/api/v1/intelligence/ownership/{ticker.upper()}")

    async def get_insider_trades(self, ticker: str, lookback_days: int = 90) -> "pd.DataFrame":
        """Async: fetch insider trades."""
        data = await self._get(
            f"/api/v1/intelligence/insider-trades/{ticker.upper()}",
            params={"lookback_days": lookback_days},
        )
        if not _PD_OK:
            return data
        return pd.DataFrame(data.get("trades", data))

    async def get_congressional_trades(self, ticker: str) -> "pd.DataFrame":
        """Async: fetch congressional trades."""
        data = await self._get(f"/api/v1/intelligence/congressional-trades/{ticker.upper()}")
        if not _PD_OK:
            return data
        return pd.DataFrame(data.get("trades", data))

    async def get_sentiment(self, ticker: str) -> dict:
        """Async: fetch sentiment score."""
        return await self._get(f"/api/v1/intelligence/sentiment/{ticker.upper()}")

    async def get_governance_score(self, ticker: str) -> dict:
        """Async: fetch governance score."""
        return await self._get(f"/api/v1/intelligence/governance/{ticker.upper()}")

    async def ask(self, question: str, ticker: Optional[str] = None) -> dict:
        """Async: RAG Q&A."""
        body: dict = {"question": question}
        if ticker:
            body["ticker"] = ticker.upper()
        return await self._post("/api/v1/intelligence/ask", body=body)

    async def summarize_filing(self, ticker: str, form_type: str = "10-K") -> dict:
        """Async: summarize SEC filing."""
        return await self._get(
            f"/api/v1/intelligence/summarize/{ticker.upper()}",
            params={"form_type": form_type},
        )

    async def get_positions(self) -> "pd.DataFrame":
        """Async: fetch paper-trading positions."""
        data = await self._get("/api/v1/orders/positions")
        if not _PD_OK:
            return data
        return pd.DataFrame(data.get("positions", data))

    async def submit_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        """Async: submit paper-trading order."""
        body: dict = {
            "ticker": ticker.upper(), "qty": qty,
            "side": side.lower(), "order_type": order_type,
        }
        if limit_price is not None:
            body["limit_price"] = limit_price
        return await self._post("/api/v1/orders/submit", body=body)


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

class SentinelWebSocketClient:
    """WebSocket client for streaming real-time quotes and bars.

    Usage::

        def on_quote(msg):
            print(msg["ticker"], msg["data"]["last"])

        ws = SentinelWebSocketClient(ws_url="ws://localhost:8000")
        ws.subscribe_quotes(["AAPL", "MSFT"], callback=on_quote)
        ws.start()    # blocks; use start(block=False) for background thread

    Reconnects automatically on disconnect with exponential backoff (up to 5 attempts).
    """

    _MAX_RECONNECT_ATTEMPTS = 5
    _RECONNECT_BASE_DELAY   = 1.0   # seconds; doubled on each attempt

    def __init__(self, ws_url: str = "ws://localhost:8000") -> None:
        if not _WS_OK:
            raise RuntimeError("websockets is required: pip install websockets")

        self.ws_url          = ws_url.rstrip("/")
        self._subscriptions: dict[str, list[Callable]] = {}   # ticker → [callback, ...]
        self._bar_subs:      dict[str, dict]            = {}   # key → {resolution, callback}
        self._global_callbacks: list[Callable] = []
        self._running        = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread]        = None
        self._ws_task: Optional[asyncio.Task]           = None

    # ------------------------------------------------------------------
    def subscribe_quotes(self, tickers: list[str], callback: Callable) -> None:
        """Register a callback for real-time quote updates.

        Args:
            tickers:  List of ticker symbols.
            callback: Called with each message dict: {type, ticker, data, ts}.
        """
        for t in tickers:
            self._subscriptions.setdefault(t.upper(), []).append(callback)

    def subscribe_bars(
        self, tickers: list[str], resolution: str, callback: Callable
    ) -> None:
        """Register a callback for live bar (OHLCV) updates.

        Args:
            tickers:    List of ticker symbols.
            resolution: Bar resolution (1|5|15|30|60|D).
            callback:   Called with each bar message dict.
        """
        for t in tickers:
            key = f"{t.upper()}:{resolution}"
            self._bar_subs[key] = {"resolution": resolution, "callback": callback}

    def unsubscribe(self, tickers: list[str]) -> None:
        """Remove all subscriptions for the given tickers."""
        for t in tickers:
            self._subscriptions.pop(t.upper(), None)
            for key in list(self._bar_subs.keys()):
                if key.startswith(t.upper() + ":"):
                    del self._bar_subs[key]

    def start(self, block: bool = True) -> None:
        """Connect and start receiving messages.

        Args:
            block: If True (default), block the calling thread.
                   If False, run in a background thread.
        """
        self._running = True
        if block:
            asyncio.run(self._run())
        else:
            self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Disconnect and stop the WebSocket client."""
        self._running = False
        if self._loop and self._ws_task:
            self._loop.call_soon_threadsafe(self._ws_task.cancel)

    def _run_in_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())

    async def _run(self) -> None:
        """Main WebSocket event loop with reconnect logic."""
        self._loop = asyncio.get_event_loop()
        attempt = 0

        while self._running and attempt <= self._MAX_RECONNECT_ATTEMPTS:
            try:
                await self._connect_and_listen()
                attempt = 0  # reset on clean disconnect
            except Exception as exc:
                attempt += 1
                if attempt > self._MAX_RECONNECT_ATTEMPTS:
                    logger.error("ws_max_reconnects_exceeded: %s", exc)
                    break
                delay = self._RECONNECT_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("ws_reconnect: attempt=%d delay=%.1fs error=%s", attempt, delay, exc)
                await asyncio.sleep(delay)

        self._running = False

    async def _connect_and_listen(self) -> None:
        """Open WebSocket, send subscriptions, and dispatch messages."""
        uri = f"{self.ws_url}/ws/quotes"
        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
            logger.info("ws_connected: %s", uri)

            # Send subscribe messages for all registered tickers
            all_tickers = list(self._subscriptions.keys())
            if all_tickers:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "tickers": all_tickers,
                }))

            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")
                ticker   = msg.get("ticker", "").upper()

                if msg_type == "quote" and ticker:
                    callbacks = self._subscriptions.get(ticker, [])
                    for cb in callbacks:
                        try:
                            cb(msg)
                        except Exception as exc:
                            logger.error("ws_callback_error: %s", exc)

                elif msg_type == "bar" and ticker:
                    resolution = msg.get("resolution", "")
                    key = f"{ticker}:{resolution}"
                    entry = self._bar_subs.get(key)
                    if entry:
                        try:
                            entry["callback"](msg)
                        except Exception as exc:
                            logger.error("ws_bar_callback_error: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class SentinelCLI:
    """Command-line interface for the SENTINEL API.

    Entry points:
        sentinel quote AAPL
        sentinel history AAPL --start 2024-01-01 --end 2025-01-01 --output data.csv
        sentinel screen --preset magic_formula --output results.csv
        sentinel ask "What is Apple's gross margin trend?" --ticker AAPL
        sentinel financials AAPL --statement income --periods 5
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None) -> None:
        self.base_url = base_url
        self.api_key  = api_key
        self._client: Optional[SentinelClient] = None

    def _get_client(self) -> SentinelClient:
        if self._client is None:
            self._client = SentinelClient(base_url=self.base_url, api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------
    def run(self, argv: Optional[list[str]] = None) -> int:
        """Parse CLI arguments and dispatch to sub-commands.

        Returns:
            Exit code (0 = success, 1 = error).
        """
        parser = argparse.ArgumentParser(
            prog="sentinel",
            description="SENTINEL Financial Terminal CLI",
        )
        parser.add_argument(
            "--url",
            default="http://localhost:8000",
            help="SENTINEL API base URL (default: http://localhost:8000)",
        )
        parser.add_argument("--api-key", default=None, help="API bearer token")

        sub = parser.add_subparsers(dest="command", required=True)

        # quote
        p_quote = sub.add_parser("quote", help="Fetch a real-time quote")
        p_quote.add_argument("ticker", help="Ticker symbol")

        # history
        p_hist = sub.add_parser("history", help="Fetch OHLCV history")
        p_hist.add_argument("ticker", help="Ticker symbol")
        p_hist.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
        p_hist.add_argument("--end",   default=None,  help="End date YYYY-MM-DD")
        p_hist.add_argument("--interval", default="1d",
                             help="Bar interval: 1d|1wk|1mo|1h|30m|15m|5m|1m")
        p_hist.add_argument("--output", default=None, help="Output CSV file path")

        # screen
        p_screen = sub.add_parser("screen", help="Run a stock screener")
        p_screen.add_argument("--preset",   default="magic_formula",
                               help="Preset screen name (default: magic_formula)")
        p_screen.add_argument("--type",     default="fundamental",
                               choices=["fundamental", "technical"])
        p_screen.add_argument("--output",   default=None, help="Output CSV file path")

        # ask
        p_ask = sub.add_parser("ask", help="Ask a natural-language question")
        p_ask.add_argument("question", help="Question text")
        p_ask.add_argument("--ticker", default=None, help="Optional ticker context")

        # financials
        p_fin = sub.add_parser("financials", help="Fetch financial statements")
        p_fin.add_argument("ticker", help="Ticker symbol")
        p_fin.add_argument("--statement", default="income",
                            choices=["income", "balance", "cashflow"])
        p_fin.add_argument("--periods", type=int, default=5,
                            help="Number of annual periods (default: 5)")
        p_fin.add_argument("--output", default=None, help="Output CSV file path")

        args = parser.parse_args(argv)
        self.base_url = args.url
        if hasattr(args, "api_key") and args.api_key:
            self.api_key = args.api_key
        self._client = None  # reset so new base_url is used

        try:
            if args.command == "quote":
                return self._cmd_quote(args)
            elif args.command == "history":
                return self._cmd_history(args)
            elif args.command == "screen":
                return self._cmd_screen(args)
            elif args.command == "ask":
                return self._cmd_ask(args)
            elif args.command == "financials":
                return self._cmd_financials(args)
        except SentinelAPIError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Unexpected error: {exc}", file=sys.stderr)
            return 1

        return 0

    def _cmd_quote(self, args: argparse.Namespace) -> int:
        client = self._get_client()
        data   = client.get_quote(args.ticker)
        print(json.dumps(data, indent=2, default=str))
        return 0

    def _cmd_history(self, args: argparse.Namespace) -> int:
        client = self._get_client()
        df     = client.get_history(
            args.ticker, start=args.start, end=args.end, interval=args.interval
        )
        if args.output:
            DataFrameSerializer.to_csv(df, path=args.output)
            print(f"Saved {len(df)} rows to {args.output}")
        else:
            if _PD_OK and isinstance(df, pd.DataFrame):
                print(df.to_string())
            else:
                print(json.dumps(df, indent=2, default=str))
        return 0

    def _cmd_screen(self, args: argparse.Namespace) -> int:
        client = self._get_client()
        df     = client.run_preset_screen(args.preset, type=args.type)
        if args.output:
            DataFrameSerializer.to_csv(df, path=args.output)
            print(f"Saved {len(df)} results to {args.output}")
        else:
            if _PD_OK and isinstance(df, pd.DataFrame):
                print(df.to_string())
            else:
                print(json.dumps(df, indent=2, default=str))
        return 0

    def _cmd_ask(self, args: argparse.Namespace) -> int:
        client = self._get_client()
        result = client.ask(args.question, ticker=args.ticker)
        answer  = result.get("answer", result)
        sources = result.get("sources", [])
        print(f"\n{answer}\n")
        if sources:
            print("Sources:")
            for src in sources[:5]:
                print(f"  - {src}")
        return 0

    def _cmd_financials(self, args: argparse.Namespace) -> int:
        client = self._get_client()
        if args.statement == "income":
            df = client.get_income_statement(args.ticker, periods=args.periods)
        elif args.statement == "balance":
            df = client.get_balance_sheet(args.ticker, periods=args.periods)
        else:
            df = client.get_cash_flow(args.ticker, periods=args.periods)

        if args.output:
            DataFrameSerializer.to_csv(df, path=args.output)
            print(f"Saved to {args.output}")
        else:
            if _PD_OK and isinstance(df, pd.DataFrame):
                print(df.to_string())
            else:
                print(json.dumps(df, indent=2, default=str))
        return 0


# ---------------------------------------------------------------------------
# Module-level CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the ``sentinel`` CLI command."""
    import os
    cli = SentinelCLI(
        base_url=os.environ.get("SENTINEL_URL", "http://localhost:8000"),
        api_key=os.environ.get("SENTINEL_API_KEY"),
    )
    sys.exit(cli.run(argv))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SentinelClient",
    "AsyncSentinelClient",
    "SentinelWebSocketClient",
    "SentinelCLI",
    "DataFrameSerializer",
    "SentinelAPIError",
    "main",
]
