"""
Autonomous AI research agent: multi-step financial research with Claude API.
Orchestrates: news gathering, fundamental analysis, technical analysis,
sentiment scoring → synthesizes into investment research report.

dim_060 — Autonomous AI research agent (target: 9)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "research_agent.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 4096
_TOOL_RATE_LIMIT_SEC = 1.0          # 1 req/sec across all tools
_CACHE_TTL_SECONDS = 3600           # 1 hour tool result cache

# ---------------------------------------------------------------------------
# Lazy Anthropic client
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic_module
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic_sdk_not_installed", hint="pip install anthropic")

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False


def _get_anthropic_client():
    """Return Anthropic client or None if API key absent / SDK missing."""
    if not _ANTHROPIC_AVAILABLE:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY") or get_settings().anthropic_api_key
    if not api_key:
        return None
    return _anthropic_module.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, type]
    required: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    tool_name: str
    params: Dict[str, Any]
    data: Any
    success: bool
    error: Optional[str] = None
    cached: bool = False
    latency_ms: float = 0.0


@dataclass
class ResearchReport:
    ticker: str
    generated_at: datetime
    executive_summary: str
    investment_thesis: str
    key_metrics: Dict[str, Any]
    technical_outlook: str
    sentiment_score: float
    risks: List[str]
    catalysts: List[str]
    recommendation: str   # BUY | HOLD | SELL
    price_target: Optional[float]
    confidence: float
    sources: List[str]
    steps_taken: int = 0
    tokens_used: int = 0
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


# Serialise to dict for JSON/SQLite
def _report_to_dict(r: ResearchReport) -> Dict[str, Any]:
    return {
        "ticker": r.ticker,
        "generated_at": r.generated_at.isoformat(),
        "executive_summary": r.executive_summary,
        "investment_thesis": r.investment_thesis,
        "key_metrics": r.key_metrics,
        "technical_outlook": r.technical_outlook,
        "sentiment_score": r.sentiment_score,
        "risks": r.risks,
        "catalysts": r.catalysts,
        "recommendation": r.recommendation,
        "price_target": r.price_target,
        "confidence": r.confidence,
        "sources": r.sources,
        "steps_taken": r.steps_taken,
        "tokens_used": r.tokens_used,
        "job_id": r.job_id,
    }


# ---------------------------------------------------------------------------
# ── 1. Research Tool Registry ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

RESEARCH_TOOLS: List[Tool] = [
    Tool(
        name="fetch_news",
        description="Get recent news headlines and summaries for a ticker",
        params={"ticker": str, "days": int},
        required=["ticker"],
    ),
    Tool(
        name="fetch_fundamentals",
        description="Get key financial metrics: PE, EPS, revenue growth, margins, debt",
        params={"ticker": str},
        required=["ticker"],
    ),
    Tool(
        name="fetch_technicals",
        description="Get price/volume/indicator data: RSI, MACD, MA, Bollinger Bands",
        params={"ticker": str, "period": str},
        required=["ticker"],
    ),
    Tool(
        name="fetch_sentiment",
        description="Get market sentiment score from news and social media (-1 to +1)",
        params={"ticker": str},
        required=["ticker"],
    ),
    Tool(
        name="fetch_peers",
        description="Get peer companies for comparison and relative valuation",
        params={"ticker": str},
        required=["ticker"],
    ),
    Tool(
        name="fetch_filing",
        description="Get recent 10-K or 10-Q summary from EDGAR",
        params={"ticker": str, "form": str},
        required=["ticker"],
    ),
    Tool(
        name="fetch_ownership",
        description="Get institutional and insider ownership data",
        params={"ticker": str},
        required=["ticker"],
    ),
    Tool(
        name="compute_dcf",
        description="Run a simplified DCF valuation with user-specified assumptions",
        params={"ticker": str, "growth_rate": float, "discount_rate": float},
        required=["ticker"],
    ),
]

_TOOL_SCHEMA_MAP: Dict[str, Dict[str, Any]] = {
    t.name: {
        "name": t.name,
        "description": t.description,
        "input_schema": {
            "type": "object",
            "properties": {
                k: {"type": "string" if v == str else "number" if v in (int, float) else "string"}
                for k, v in t.params.items()
            },
            "required": t.required,
        },
    }
    for t in RESEARCH_TOOLS
}


class ResearchToolRegistry:
    """Registry for research tools available to the agent."""

    def get_tool(self, name: str) -> Optional[Tool]:
        return next((t for t in RESEARCH_TOOLS if t.name == name), None)

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Return tool list in Anthropic API tool_use format."""
        return list(_TOOL_SCHEMA_MAP.values())

    def list_tools(self) -> List[str]:
        return [t.name for t in RESEARCH_TOOLS]


# ---------------------------------------------------------------------------
# ── 2. Tool Executor ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Execute research tools with caching, rate-limiting, and error handling."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._last_call_time: float = 0.0
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key TEXT PRIMARY KEY,
                    tool_name TEXT,
                    params_json TEXT,
                    result_json TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

    def _cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        import hashlib
        payload = json.dumps({"tool": tool_name, "params": params}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _get_cached(self, key: str) -> Optional[Any]:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            row = conn.execute(
                "SELECT result_json, created_at FROM tool_cache WHERE cache_key=?", (key,)
            ).fetchone()
        if row:
            result_json, created_at = row
            age = time.time() - created_at
            if age < _CACHE_TTL_SECONDS:
                return json.loads(result_json)
        return None

    def _set_cached(self, key: str, tool_name: str, params: Dict[str, Any], data: Any) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tool_cache
                   (cache_key, tool_name, params_json, result_json, created_at)
                   VALUES (?,?,?,?,?)""",
                (key, tool_name, json.dumps(params), json.dumps(data, default=str), time.time()),
            )
            conn.commit()

    def _rate_limit(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_call_time
            if elapsed < _TOOL_RATE_LIMIT_SEC:
                time.sleep(_TOOL_RATE_LIMIT_SEC - elapsed)
            self._last_call_time = time.time()

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Dispatch a tool call with caching, rate limiting, and error handling."""
        key = self._cache_key(tool_name, params)
        cached = self._get_cached(key)
        if cached is not None:
            return ToolResult(tool_name=tool_name, params=params, data=cached, success=True, cached=True)

        self._rate_limit()
        t0 = time.time()
        try:
            data = self._dispatch(tool_name, params)
            latency = (time.time() - t0) * 1000
            self._set_cached(key, tool_name, params, data)
            return ToolResult(tool_name=tool_name, params=params, data=data, success=True, latency_ms=latency)
        except Exception as exc:
            latency = (time.time() - t0) * 1000
            logger.warning("tool_execution_failed", tool=tool_name, error=str(exc))
            partial = {"error": str(exc), "tool": tool_name, "params": params}
            return ToolResult(
                tool_name=tool_name, params=params, data=partial,
                success=False, error=str(exc), latency_ms=latency,
            )

    def _dispatch(self, tool_name: str, params: Dict[str, Any]) -> Any:
        dispatch_map = {
            "fetch_news": self._fetch_news,
            "fetch_fundamentals": self._fetch_fundamentals,
            "fetch_technicals": self._fetch_technicals,
            "fetch_sentiment": self._fetch_sentiment,
            "fetch_peers": self._fetch_peers,
            "fetch_filing": self._fetch_filing,
            "fetch_ownership": self._fetch_ownership,
            "compute_dcf": self._compute_dcf,
        }
        fn = dispatch_map.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return fn(**params)

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #

    def _fetch_news(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Fetch recent news via Finnhub free API, fall back to yfinance news."""
        settings = get_settings()
        articles = []

        # Try Finnhub
        if settings.finnhub_api_key:
            try:
                since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
                to_date = datetime.utcnow().strftime("%Y-%m-%d")
                url = (
                    f"https://finnhub.io/api/v1/company-news"
                    f"?symbol={ticker}&from={since}&to={to_date}"
                    f"&token={settings.finnhub_api_key}"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    raw = resp.json()
                    for item in raw[:15]:
                        articles.append({
                            "headline": item.get("headline", ""),
                            "summary": item.get("summary", "")[:300],
                            "source": item.get("source", ""),
                            "datetime": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d"),
                            "sentiment": item.get("sentiment", {}).get("compound", 0.0),
                        })
            except Exception as exc:
                logger.debug("finnhub_news_failed", ticker=ticker, error=str(exc))

        # Fallback: yfinance
        if not articles and _YFINANCE_AVAILABLE:
            try:
                tkr = yf.Ticker(ticker)
                news = tkr.news or []
                for item in news[:10]:
                    articles.append({
                        "headline": item.get("title", ""),
                        "summary": item.get("summary", item.get("title", ""))[:300],
                        "source": item.get("publisher", ""),
                        "datetime": datetime.fromtimestamp(item.get("providerPublishTime", time.time())).strftime("%Y-%m-%d"),
                        "sentiment": 0.0,
                    })
            except Exception as exc:
                logger.debug("yfinance_news_failed", ticker=ticker, error=str(exc))

        if not articles:
            articles = [{"headline": f"No news available for {ticker}", "summary": "", "source": "mock", "datetime": datetime.utcnow().strftime("%Y-%m-%d"), "sentiment": 0.0}]

        return {"ticker": ticker, "days": days, "articles": articles, "count": len(articles)}

    def _fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get key financial metrics via yfinance."""
        if not _YFINANCE_AVAILABLE:
            return self._mock_fundamentals(ticker)

        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info or {}
            fins = {}

            # Income statement TTM
            try:
                is_df = tkr.income_stmt
                if is_df is not None and not is_df.empty:
                    col = is_df.columns[0]
                    fins["revenue_ttm"] = float(is_df.loc["Total Revenue", col]) if "Total Revenue" in is_df.index else None
                    fins["net_income_ttm"] = float(is_df.loc["Net Income", col]) if "Net Income" in is_df.index else None
                    fins["ebitda_ttm"] = float(is_df.loc["EBITDA", col]) if "EBITDA" in is_df.index else None
            except Exception:
                pass

            return {
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "analyst_target_price": info.get("targetMeanPrice"),
                "analyst_recommendation": info.get("recommendationMean"),
                "eps_ttm": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "free_cash_flow": info.get("freeCashflow"),
                "shares_outstanding": info.get("sharesOutstanding"),
                **fins,
            }
        except Exception as exc:
            logger.warning("fundamentals_fetch_failed", ticker=ticker, error=str(exc))
            return self._mock_fundamentals(ticker)

    def _mock_fundamentals(self, ticker: str) -> Dict[str, Any]:
        import random
        rng = random.Random(hash(ticker) % 2**31)
        return {
            "ticker": ticker, "company_name": f"{ticker} Corp", "sector": "Technology",
            "market_cap": rng.randint(1, 500) * 1e9, "pe_ratio": rng.uniform(10, 40),
            "forward_pe": rng.uniform(8, 35), "price_to_book": rng.uniform(1, 10),
            "roe": rng.uniform(0.05, 0.30), "profit_margin": rng.uniform(0.05, 0.25),
            "revenue_growth": rng.uniform(-0.05, 0.30), "debt_to_equity": rng.uniform(0.1, 2.0),
            "beta": rng.uniform(0.5, 1.8), "dividend_yield": rng.uniform(0, 0.04),
            "eps_ttm": rng.uniform(1, 15), "note": "mock_data",
        }

    def _fetch_technicals(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Compute technical indicators from price history."""
        if not _YFINANCE_AVAILABLE:
            return self._mock_technicals(ticker)

        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period=period, auto_adjust=True)
            if hist.empty:
                return self._mock_technicals(ticker)

            close = hist["Close"].squeeze()
            volume = hist["Volume"].squeeze()

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

            # MAs
            sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.iloc[-1])

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_val = float(macd_line.iloc[-1])
            signal_val = float(signal_line.iloc[-1])

            # Bollinger Bands (20, 2)
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = float((bb_mid + 2 * bb_std).iloc[-1]) if len(close) >= 20 else float(close.iloc[-1]) * 1.05
            bb_lower = float((bb_mid - 2 * bb_std).iloc[-1]) if len(close) >= 20 else float(close.iloc[-1]) * 0.95

            current_price = float(close.iloc[-1])
            pct_change_1m = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0
            pct_change_3m = float((close.iloc[-1] / close.iloc[-63] - 1) * 100) if len(close) >= 63 else 0.0
            pct_change_1y = float((close.iloc[-1] / close.iloc[0] - 1) * 100)

            avg_volume_20d = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
            latest_volume = float(volume.iloc[-1])

            trend = "BULLISH" if current_price > sma200 and sma50 > sma200 else ("BEARISH" if current_price < sma200 else "NEUTRAL")

            return {
                "ticker": ticker,
                "period": period,
                "current_price": round(current_price, 2),
                "rsi_14": round(rsi, 1),
                "sma_20": round(sma20, 2),
                "sma_50": round(sma50, 2),
                "sma_200": round(sma200, 2),
                "macd": round(macd_val, 4),
                "macd_signal": round(signal_val, 4),
                "macd_histogram": round(macd_val - signal_val, 4),
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2),
                "pct_change_1m": round(pct_change_1m, 2),
                "pct_change_3m": round(pct_change_3m, 2),
                "pct_change_1y": round(pct_change_1y, 2),
                "volume_latest": int(latest_volume),
                "volume_avg_20d": int(avg_volume_20d),
                "volume_ratio": round(latest_volume / avg_volume_20d, 2) if avg_volume_20d > 0 else 1.0,
                "above_sma200": current_price > sma200,
                "above_sma50": current_price > sma50,
                "trend": trend,
                "52_week_high": round(float(close.max()), 2),
                "52_week_low": round(float(close.min()), 2),
            }
        except Exception as exc:
            logger.warning("technicals_fetch_failed", ticker=ticker, error=str(exc))
            return self._mock_technicals(ticker)

    def _mock_technicals(self, ticker: str) -> Dict[str, Any]:
        import random
        rng = random.Random(hash(ticker + "tech") % 2**31)
        price = rng.uniform(50, 500)
        return {
            "ticker": ticker, "current_price": round(price, 2),
            "rsi_14": round(rng.uniform(30, 70), 1),
            "sma_50": round(price * rng.uniform(0.95, 1.05), 2),
            "sma_200": round(price * rng.uniform(0.90, 1.10), 2),
            "macd": round(rng.uniform(-2, 2), 3),
            "trend": rng.choice(["BULLISH", "NEUTRAL", "BEARISH"]),
            "note": "mock_data",
        }

    def _fetch_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Compute sentiment score from news headlines."""
        news = self._fetch_news(ticker, days=7)
        articles = news.get("articles", [])

        # Simple keyword-based scoring if no sentiment field
        positive_words = {"beat", "surge", "growth", "strong", "record", "upgrade", "buy", "outperform", "profit", "rally", "gain"}
        negative_words = {"miss", "drop", "loss", "decline", "downgrade", "sell", "underperform", "cut", "crash", "warning", "risk"}

        scores = []
        for article in articles:
            headline = (article.get("headline", "") + " " + article.get("summary", "")).lower()
            pos = sum(1 for w in positive_words if w in headline)
            neg = sum(1 for w in negative_words if w in headline)
            total = pos + neg
            if total == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / total)

        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        avg_score = max(-1.0, min(1.0, avg_score))

        label = "BULLISH" if avg_score > 0.15 else ("BEARISH" if avg_score < -0.15 else "NEUTRAL")
        return {
            "ticker": ticker,
            "sentiment_score": round(avg_score, 3),
            "label": label,
            "article_count": len(articles),
            "positive_articles": sum(1 for s in scores if s > 0),
            "negative_articles": sum(1 for s in scores if s < 0),
            "neutral_articles": sum(1 for s in scores if s == 0),
        }

    def _fetch_peers(self, ticker: str) -> Dict[str, Any]:
        """Get peer companies via Finnhub, fall back to sector-based heuristic."""
        settings = get_settings()
        peers = []

        if settings.finnhub_api_key:
            try:
                url = f"https://finnhub.io/api/v1/stock/peers?symbol={ticker}&token={settings.finnhub_api_key}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    raw = resp.json()
                    peers = [p for p in raw if isinstance(p, str) and p != ticker][:8]
            except Exception as exc:
                logger.debug("finnhub_peers_failed", ticker=ticker, error=str(exc))

        # Fallback: hardcoded sector peers for common tickers
        if not peers:
            _PEER_MAP = {
                "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
                "MSFT": ["AAPL", "GOOGL", "ORCL", "IBM"],
                "GOOGL": ["MSFT", "META", "AAPL", "AMZN"],
                "TSLA": ["GM", "F", "RIVN", "NIO"],
                "JPM": ["BAC", "C", "WFC", "GS"],
                "XOM": ["CVX", "COP", "BP", "SHEL"],
                "SPY": ["QQQ", "IWM", "DIA", "VTI"],
            }
            peers = _PEER_MAP.get(ticker.upper(), ["SPY", "QQQ"])

        # Gather basic metrics for each peer
        peer_data = []
        for peer in peers[:5]:
            try:
                if _YFINANCE_AVAILABLE:
                    info = yf.Ticker(peer).info or {}
                    peer_data.append({
                        "ticker": peer,
                        "pe": info.get("trailingPE"),
                        "market_cap": info.get("marketCap"),
                        "revenue_growth": info.get("revenueGrowth"),
                        "profit_margin": info.get("profitMargins"),
                    })
                else:
                    peer_data.append({"ticker": peer})
            except Exception:
                peer_data.append({"ticker": peer})

        return {"ticker": ticker, "peers": peer_data, "peer_tickers": peers}

    def _fetch_filing(self, ticker: str, form: str = "10-K") -> Dict[str, Any]:
        """Get recent EDGAR filing summary for the ticker."""
        try:
            # Map ticker → CIK via EDGAR tickers JSON
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            headers = {"User-Agent": "SENTINEL richard.porras@realempanada.com"}
            resp = requests.get(tickers_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                raise RuntimeError(f"EDGAR tickers fetch failed: {resp.status_code}")

            tickers_data = resp.json()
            cik = None
            for _, entry in tickers_data.items():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    break

            if not cik:
                return {"ticker": ticker, "form": form, "error": "CIK not found", "filings": []}

            # Get submissions
            sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp2 = requests.get(sub_url, headers=headers, timeout=15)
            if resp2.status_code != 200:
                return {"ticker": ticker, "form": form, "error": "submissions fetch failed"}

            sub_data = resp2.json()
            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            descriptions = recent.get("primaryDocument", [])

            filings_found = []
            for i, f in enumerate(forms):
                if f == form:
                    filings_found.append({
                        "form": f,
                        "filing_date": dates[i] if i < len(dates) else "",
                        "accession": accessions[i] if i < len(accessions) else "",
                        "primary_doc": descriptions[i] if i < len(descriptions) else "",
                    })
                if len(filings_found) >= 3:
                    break

            company_name = sub_data.get("name", ticker)
            return {
                "ticker": ticker,
                "cik": cik,
                "company_name": company_name,
                "form": form,
                "filings": filings_found,
                "fiscal_year_end": sub_data.get("fiscalYearEnd", ""),
            }
        except Exception as exc:
            logger.warning("edgar_filing_failed", ticker=ticker, form=form, error=str(exc))
            return {"ticker": ticker, "form": form, "error": str(exc), "filings": []}

    def _fetch_ownership(self, ticker: str) -> Dict[str, Any]:
        """Get institutional and insider ownership via yfinance."""
        if not _YFINANCE_AVAILABLE:
            return {"ticker": ticker, "error": "yfinance not available", "institutional": [], "insider": []}

        try:
            tkr = yf.Ticker(ticker)
            inst_holders = []
            try:
                inst_df = tkr.institutional_holders
                if inst_df is not None and not inst_df.empty:
                    for _, row in inst_df.head(10).iterrows():
                        inst_holders.append({
                            "holder": str(row.get("Holder", "")),
                            "shares": int(row.get("Shares", 0)),
                            "pct_out": float(row.get("% Out", 0)),
                            "value": float(row.get("Value", 0)),
                        })
            except Exception:
                pass

            insider = []
            try:
                ins_df = tkr.insider_purchases
                if ins_df is not None and not ins_df.empty:
                    for _, row in ins_df.head(5).iterrows():
                        insider.append({
                            "insider": str(row.get("Insider Trading", "")),
                            "relationship": str(row.get("Relationship", "")),
                            "transaction": str(row.get("Transaction", "")),
                            "shares": int(row.get("#Shares", 0)),
                        })
            except Exception:
                pass

            info = tkr.info or {}
            return {
                "ticker": ticker,
                "institutional_holders": inst_holders,
                "insider_transactions": insider,
                "institutional_ownership_pct": info.get("heldPercentInstitutions"),
                "insider_ownership_pct": info.get("heldPercentInsiders"),
                "shares_short_pct": info.get("shortPercentOfFloat"),
                "float_shares": info.get("floatShares"),
            }
        except Exception as exc:
            return {"ticker": ticker, "error": str(exc)}

    def _compute_dcf(
        self,
        ticker: str,
        growth_rate: float = 0.10,
        discount_rate: float = 0.10,
    ) -> Dict[str, Any]:
        """Run a simplified Gordon Growth / DCF model."""
        funds = self._fetch_fundamentals(ticker)

        fcf = funds.get("free_cash_flow")
        shares = funds.get("shares_outstanding")
        if not fcf or not shares or shares == 0:
            # Use net income as proxy
            fcf = funds.get("net_income_ttm")
        if not fcf or not shares:
            return {"ticker": ticker, "error": "Insufficient financial data for DCF", "dcf_value": None}

        # Project FCF for 10 years, then terminal value
        terminal_growth = min(0.025, growth_rate * 0.25)
        projected_fcf = []
        current_fcf = float(fcf)
        for yr in range(1, 11):
            current_fcf *= (1 + growth_rate)
            pv = current_fcf / ((1 + discount_rate) ** yr)
            projected_fcf.append({"year": yr, "fcf": round(current_fcf, 0), "pv": round(pv, 0)})

        terminal_fcf = current_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** 10)

        intrinsic_value = (sum(p["pv"] for p in projected_fcf) + pv_terminal) / float(shares)

        if _YFINANCE_AVAILABLE:
            try:
                current_price = yf.Ticker(ticker).info.get("currentPrice") or yf.Ticker(ticker).info.get("regularMarketPrice")
            except Exception:
                current_price = None
        else:
            current_price = None

        margin_of_safety = None
        if current_price and intrinsic_value > 0:
            margin_of_safety = round((intrinsic_value - current_price) / intrinsic_value * 100, 1)

        return {
            "ticker": ticker,
            "dcf_value": round(intrinsic_value, 2),
            "current_price": current_price,
            "margin_of_safety_pct": margin_of_safety,
            "assumptions": {
                "growth_rate": growth_rate,
                "discount_rate": discount_rate,
                "terminal_growth": terminal_growth,
                "years": 10,
            },
            "pv_fcf_sum": round(sum(p["pv"] for p in projected_fcf), 0),
            "pv_terminal": round(pv_terminal, 0),
            "projected_fcf": projected_fcf,
        }


# ---------------------------------------------------------------------------
# ── 3. Agent Memory ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class AgentMemory:
    """Persistent research memory — ticker-scoped, cross-session."""

    def __init__(self) -> None:
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    question TEXT,
                    summary TEXT,
                    key_findings TEXT,
                    recommendation TEXT,
                    sentiment_score REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    job_id TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_ticker ON research_memory(ticker)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_jobs (
                    job_id TEXT PRIMARY KEY,
                    ticker TEXT,
                    question TEXT,
                    status TEXT DEFAULT 'pending',
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT
                )
            """)
            conn.commit()

    def store(self, report: ResearchReport, question: str) -> None:
        findings = json.dumps({
            "risks": report.risks,
            "catalysts": report.catalysts,
            "key_metrics": report.key_metrics,
        })
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                """INSERT INTO research_memory
                   (ticker, question, summary, key_findings, recommendation, sentiment_score, job_id)
                   VALUES (?,?,?,?,?,?,?)""",
                (report.ticker, question, report.executive_summary,
                 findings, report.recommendation, report.sentiment_score, report.job_id),
            )
            conn.commit()

    def get_prior_research(self, ticker: str, days_back: int = 90) -> List[Dict[str, Any]]:
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        with sqlite3.connect(str(_DB_PATH)) as conn:
            rows = conn.execute(
                """SELECT ticker, question, summary, key_findings, recommendation, sentiment_score, created_at
                   FROM research_memory
                   WHERE ticker=? AND created_at >= ?
                   ORDER BY created_at DESC LIMIT 10""",
                (ticker.upper(), cutoff),
            ).fetchall()
        result = []
        for row in rows:
            result.append({
                "ticker": row[0], "question": row[1], "summary": row[2],
                "key_findings": json.loads(row[3]) if row[3] else {},
                "recommendation": row[4], "sentiment_score": row[5], "date": row[6],
            })
        return result

    def has_recent_research(self, ticker: str, question: str, hours: int = 24) -> Optional[str]:
        """Return job_id if we have recent identical research, else None."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(str(_DB_PATH)) as conn:
            row = conn.execute(
                "SELECT job_id FROM research_memory WHERE ticker=? AND question=? AND created_at>=? LIMIT 1",
                (ticker.upper(), question, cutoff),
            ).fetchone()
        return row[0] if row else None

    # ---- Job tracking -------------------------------------------------------

    def create_job(self, job_id: str, ticker: str, question: str) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_jobs (job_id, ticker, question, status) VALUES (?,?,?,?)",
                (job_id, ticker, question, "running"),
            )
            conn.commit()

    def complete_job(self, job_id: str, report: ResearchReport) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "UPDATE research_jobs SET status=?, result_json=?, completed_at=datetime('now') WHERE job_id=?",
                ("completed", json.dumps(_report_to_dict(report), default=str), job_id),
            )
            conn.commit()

    def fail_job(self, job_id: str, error: str) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "UPDATE research_jobs SET status=?, error=?, completed_at=datetime('now') WHERE job_id=?",
                ("failed", error, job_id),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            row = conn.execute(
                "SELECT job_id, ticker, question, status, result_json, error, created_at, completed_at FROM research_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "job_id": row[0], "ticker": row[1], "question": row[2], "status": row[3],
            "result": json.loads(row[4]) if row[4] else None,
            "error": row[5], "created_at": row[6], "completed_at": row[7],
        }

    def get_history(self, ticker: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            if ticker:
                rows = conn.execute(
                    "SELECT job_id, ticker, question, status, created_at, completed_at FROM research_jobs WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
                    (ticker.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT job_id, ticker, question, status, created_at, completed_at FROM research_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {"job_id": r[0], "ticker": r[1], "question": r[2], "status": r[3], "created_at": r[4], "completed_at": r[5]}
            for r in rows
        ]


# ---------------------------------------------------------------------------
# ── 4. Report Generator ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Structure tool results into a ResearchReport."""

    def build_mock_report(
        self,
        ticker: str,
        question: str,
        tool_results: Dict[str, ToolResult],
    ) -> ResearchReport:
        """Build a report from raw tool data without calling the LLM."""
        funds = tool_results.get("fetch_fundamentals", ToolResult("", {}, {}, False)).data or {}
        tech = tool_results.get("fetch_technicals", ToolResult("", {}, {}, False)).data or {}
        news_data = tool_results.get("fetch_news", ToolResult("", {}, {}, False)).data or {}
        sent = tool_results.get("fetch_sentiment", ToolResult("", {}, {}, False)).data or {}
        dcf = tool_results.get("compute_dcf", ToolResult("", {}, {}, False)).data or {}

        ticker_upper = ticker.upper()
        pe = funds.get("pe_ratio")
        rev_growth = funds.get("revenue_growth")
        margin = funds.get("profit_margin")
        rsi = tech.get("rsi_14", 50)
        trend = tech.get("trend", "NEUTRAL")
        sentiment_score = float(sent.get("sentiment_score", 0.0))
        dcf_val = dcf.get("dcf_value")
        current_price = tech.get("current_price") or funds.get("52_week_high")
        mos = dcf.get("margin_of_safety_pct")

        # Heuristic recommendation
        bull_signals = 0
        bear_signals = 0
        if rsi < 40:
            bull_signals += 1
        elif rsi > 65:
            bear_signals += 1
        if trend == "BULLISH":
            bull_signals += 2
        elif trend == "BEARISH":
            bear_signals += 2
        if sentiment_score > 0.1:
            bull_signals += 1
        elif sentiment_score < -0.1:
            bear_signals += 1
        if mos and mos > 20:
            bull_signals += 2
        elif mos and mos < -20:
            bear_signals += 2
        if rev_growth and rev_growth > 0.10:
            bull_signals += 1

        score = bull_signals - bear_signals
        if score >= 3:
            recommendation = "BUY"
            confidence = min(0.85, 0.55 + score * 0.05)
        elif score <= -2:
            recommendation = "SELL"
            confidence = min(0.80, 0.50 + abs(score) * 0.05)
        else:
            recommendation = "HOLD"
            confidence = 0.55

        price_target = None
        if dcf_val and dcf_val > 0:
            price_target = round(dcf_val, 2)
        elif current_price and pe and pe > 0:
            # Simple forward PE estimate
            eps = funds.get("eps_forward") or funds.get("eps_ttm", 0)
            if eps:
                price_target = round(float(pe) * float(eps) * 1.1, 2)

        articles = news_data.get("articles", [])
        news_summary = "; ".join(a["headline"] for a in articles[:3]) if articles else "No recent news."

        executive_summary = (
            f"{ticker_upper} is trading at ${current_price or 'N/A'} with a "
            f"{'strong' if trend == 'BULLISH' else 'weak' if trend == 'BEARISH' else 'neutral'} technical trend. "
            f"RSI at {rsi:.0f} indicates {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'} conditions. "
            f"Sentiment is {sent.get('label', 'NEUTRAL')} based on {news_data.get('count', 0)} recent articles."
        )

        investment_thesis = (
            f"Bull case: {trend} trend, "
            f"{'growing revenues (' + str(round(rev_growth * 100, 1)) + '% YoY)' if rev_growth else 'revenue data unavailable'}, "
            f"{'DCF upside of ' + str(mos) + '%' if mos and mos > 0 else 'fair valued by DCF'}. "
            f"Bear case: {'' if pe is None else f'PE of {pe:.1f}x may be stretched at'} current levels; "
            f"macro headwinds could compress multiples."
        )

        risks = [
            "Macro interest rate sensitivity may compress valuations",
            "Revenue growth deceleration risk if consumer spending slows",
            "Competitive pressure from sector peers",
            "Regulatory and geopolitical risks affecting operations",
            f"Technical breakdown risk if price falls below 200-day SMA (${tech.get('sma_200', 'N/A')})",
        ]

        catalysts = [
            "Upcoming earnings report — EPS beat could re-rate stock higher",
            f"DCF-implied {'upside' if mos and mos > 0 else 'downside'} of {abs(mos or 0):.1f}% vs intrinsic value",
            "Institutional ownership changes or activist involvement",
            "New product launches or market expansion",
            "Share buyback program acceleration",
        ]

        key_metrics = {
            "pe_ratio": pe,
            "forward_pe": funds.get("forward_pe"),
            "revenue_growth": rev_growth,
            "profit_margin": margin,
            "debt_to_equity": funds.get("debt_to_equity"),
            "rsi_14": rsi,
            "trend": trend,
            "52w_high": funds.get("52_week_high"),
            "52w_low": funds.get("52_week_low"),
            "dcf_value": dcf_val,
            "margin_of_safety_pct": mos,
        }

        sources_used = [k for k, v in tool_results.items() if v.success]

        return ResearchReport(
            ticker=ticker_upper,
            generated_at=datetime.utcnow(),
            executive_summary=executive_summary,
            investment_thesis=investment_thesis,
            key_metrics=key_metrics,
            technical_outlook=f"Trend: {trend} | RSI: {rsi:.0f} | MACD: {'Bullish' if tech.get('macd', 0) > tech.get('macd_signal', 0) else 'Bearish'} | Price vs 200MA: {'Above' if tech.get('above_sma200') else 'Below'}",
            sentiment_score=sentiment_score,
            risks=risks,
            catalysts=catalysts,
            recommendation=recommendation,
            price_target=price_target,
            confidence=round(confidence, 2),
            sources=sources_used,
        )

    def build_from_llm_response(
        self,
        ticker: str,
        llm_text: str,
        tool_results: Dict[str, ToolResult],
        tokens_used: int = 0,
    ) -> ResearchReport:
        """Parse LLM synthesis text into a structured report."""
        # Attempt to extract structured fields from LLM output
        def extract(pattern: str, text: str, default: str = "") -> str:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else default

        import re

        exec_summary = extract(r"(?:executive summary|summary)[:：]\s*(.+?)(?:\n\n|\Z)", llm_text) or llm_text[:300]
        thesis = extract(r"(?:investment thesis|thesis)[:：]\s*(.+?)(?:\n\n|\Z)", llm_text) or ""
        recommendation_raw = extract(r"(?:recommendation|rating)[:：]\s*(BUY|HOLD|SELL|buy|hold|sell)", llm_text) or "HOLD"
        recommendation = recommendation_raw.upper()[:4]
        if recommendation not in {"BUY", "HOLD", "SELL"}:
            recommendation = "HOLD"

        price_target_raw = extract(r"price target[:：]\s*\$?([\d,]+(?:\.\d+)?)", llm_text)
        price_target = float(price_target_raw.replace(",", "")) if price_target_raw else None

        confidence_raw = extract(r"confidence[:：]\s*([\d.]+)", llm_text)
        confidence = float(confidence_raw) if confidence_raw else 0.60
        confidence = max(0.0, min(1.0, confidence))

        # Extract bullet-list risks and catalysts
        risks = re.findall(r"(?:risk[s]?[:：]\s*[-•]\s*|[-•]\s*)([^\n]+)", llm_text)[:5] or [
            "Macro headwinds", "Competitive risks", "Execution risk", "Valuation risk", "Regulatory risk"
        ]
        catalysts = re.findall(r"(?:catalyst[s]?[:：]\s*[-•]\s*|[+]\s*)([^\n]+)", llm_text)[:5] or [
            "Earnings beat", "Revenue acceleration", "New product cycle", "Buybacks", "Analyst upgrades"
        ]

        sent = tool_results.get("fetch_sentiment", ToolResult("", {}, {}, False)).data or {}
        tech = tool_results.get("fetch_technicals", ToolResult("", {}, {}, False)).data or {}
        funds = tool_results.get("fetch_fundamentals", ToolResult("", {}, {}, False)).data or {}
        dcf = tool_results.get("compute_dcf", ToolResult("", {}, {}, False)).data or {}

        sources_used = [k for k, v in tool_results.items() if v.success]

        return ResearchReport(
            ticker=ticker.upper(),
            generated_at=datetime.utcnow(),
            executive_summary=exec_summary,
            investment_thesis=thesis or llm_text[:500],
            key_metrics={
                "pe_ratio": funds.get("pe_ratio"),
                "revenue_growth": funds.get("revenue_growth"),
                "rsi_14": tech.get("rsi_14"),
                "trend": tech.get("trend"),
                "dcf_value": dcf.get("dcf_value"),
            },
            technical_outlook=f"Trend: {tech.get('trend', 'N/A')} | RSI: {tech.get('rsi_14', 'N/A')}",
            sentiment_score=float(sent.get("sentiment_score", 0.0)),
            risks=risks,
            catalysts=catalysts,
            recommendation=recommendation,
            price_target=price_target,
            confidence=confidence,
            sources=sources_used,
            tokens_used=tokens_used,
        )


# ---------------------------------------------------------------------------
# ── 5. Research Workflow Templates ───────────────────────────────────────────
# ---------------------------------------------------------------------------

WORKFLOW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "quick_screen": {
        "description": "Fast 5-step screen: fundamentals + sentiment → brief report",
        "tools": ["fetch_fundamentals", "fetch_technicals", "fetch_sentiment"],
        "max_steps": 5,
        "estimated_cost_usd": 0.03,
    },
    "full_deep_dive": {
        "description": "Comprehensive 8-tool research report",
        "tools": ["fetch_news", "fetch_fundamentals", "fetch_technicals", "fetch_sentiment",
                  "fetch_peers", "fetch_filing", "fetch_ownership", "compute_dcf"],
        "max_steps": 12,
        "estimated_cost_usd": 0.12,
    },
    "pre_earnings": {
        "description": "Pre-earnings analysis: estimates, guidance, options positioning",
        "tools": ["fetch_fundamentals", "fetch_news", "fetch_sentiment", "fetch_technicals"],
        "max_steps": 8,
        "estimated_cost_usd": 0.06,
    },
    "risk_assessment": {
        "description": "Risk focus: debt, covenant risk, default metrics, stress test",
        "tools": ["fetch_fundamentals", "fetch_filing", "fetch_ownership"],
        "max_steps": 6,
        "estimated_cost_usd": 0.05,
    },
    "ma_target": {
        "description": "M&A target analysis: valuation, balance sheet, peer multiples",
        "tools": ["fetch_fundamentals", "fetch_peers", "compute_dcf", "fetch_ownership"],
        "max_steps": 8,
        "estimated_cost_usd": 0.07,
    },
}


# ---------------------------------------------------------------------------
# ── 6. Research Agent Orchestrator ───────────────────────────────────────────
# ---------------------------------------------------------------------------

class ResearchAgentOrchestrator:
    """Main autonomous research agent: plan → execute tools → synthesize."""

    def __init__(self) -> None:
        self._executor = ToolExecutor()
        self._memory = AgentMemory()
        self._report_gen = ReportGenerator()
        self._registry = ResearchToolRegistry()
        self._client = _get_anthropic_client()

    def run_research(
        self,
        ticker: str,
        question: str,
        max_steps: int = 10,
        workflow: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> ResearchReport:
        """
        Orchestrate a full research session.
        If ANTHROPIC_API_KEY is set, use Claude for planning + synthesis.
        Otherwise fall back to rule-based mock pipeline.
        """
        ticker = ticker.upper()
        job_id = job_id or str(uuid.uuid4())[:8]
        self._memory.create_job(job_id, ticker, question)

        try:
            # Determine tools to use
            if workflow and workflow in WORKFLOW_TEMPLATES:
                tool_names = WORKFLOW_TEMPLATES[workflow]["tools"]
                max_steps = WORKFLOW_TEMPLATES[workflow]["max_steps"]
            else:
                tool_names = self._plan_tools(ticker, question)

            tool_names = tool_names[:max_steps]

            # Gather prior context
            prior = self._memory.get_prior_research(ticker, days_back=90)
            prior_context = ""
            if prior:
                prior_context = f"\nPrior research on {ticker}: {prior[0].get('summary', '')}"

            # Execute tools
            tool_results: Dict[str, ToolResult] = {}
            for tool_name in tool_names:
                logger.info("executing_tool", tool=tool_name, ticker=ticker)
                result = self._executor.execute_tool(tool_name, {"ticker": ticker})
                tool_results[tool_name] = result

            # Synthesize
            if self._client:
                report = self._run_with_claude(
                    ticker, question, tool_results, prior_context, max_steps, job_id
                )
            else:
                report = self._run_offline(ticker, question, tool_results, job_id)

            report.steps_taken = len(tool_names)
            self._memory.complete_job(job_id, report)
            self._memory.store(report, question)
            return report

        except Exception as exc:
            logger.error("research_agent_failed", ticker=ticker, error=str(exc))
            self._memory.fail_job(job_id, str(exc))
            raise

    def _plan_tools(self, ticker: str, question: str) -> List[str]:
        """Heuristically select tools based on question keywords."""
        q_lower = question.lower()
        tools = ["fetch_fundamentals", "fetch_technicals", "fetch_sentiment"]

        if any(kw in q_lower for kw in ["news", "recent", "latest", "event"]):
            tools.insert(0, "fetch_news")
        if any(kw in q_lower for kw in ["valuation", "dcf", "intrinsic", "value", "worth"]):
            tools.append("compute_dcf")
        if any(kw in q_lower for kw in ["peer", "competitor", "relative", "compare"]):
            tools.append("fetch_peers")
        if any(kw in q_lower for kw in ["filing", "10k", "10q", "annual", "report"]):
            tools.append("fetch_filing")
        if any(kw in q_lower for kw in ["ownership", "insider", "institution", "holder"]):
            tools.append("fetch_ownership")

        # Deduplicate while preserving order
        seen = set()
        return [t for t in tools if not (t in seen or seen.add(t))]

    def _run_offline(
        self,
        ticker: str,
        question: str,
        tool_results: Dict[str, ToolResult],
        job_id: str,
    ) -> ResearchReport:
        """Rule-based report when Claude API is unavailable."""
        logger.info("research_agent_offline_mode", ticker=ticker)
        report = self._report_gen.build_mock_report(ticker, question, tool_results)
        report.job_id = job_id
        return report

    def _run_with_claude(
        self,
        ticker: str,
        question: str,
        initial_tool_results: Dict[str, ToolResult],
        prior_context: str,
        max_steps: int,
        job_id: str,
    ) -> ResearchReport:
        """
        Agentic loop using Claude's tool_use API:
        1. Provide gathered data + question to Claude
        2. Claude may request additional tool calls
        3. Execute those calls and feed results back
        4. Ask Claude to synthesize a final report
        """
        all_tool_results = dict(initial_tool_results)

        # Build initial context from already-fetched tools
        tool_summaries = []
        for tool_name, result in initial_tool_results.items():
            summary = json.dumps(result.data, default=str)[:1500]
            tool_summaries.append(f"### {tool_name}\n{summary}")

        system_prompt = (
            "You are SENTINEL, an institutional-grade financial research AI. "
            "Analyze the provided data and answer the research question. "
            "Structure your response with: Executive Summary, Investment Thesis, "
            "Key Metrics, Technical Outlook, Risks (3-5 bullets), Catalysts (3-5 bullets), "
            "Recommendation (BUY/HOLD/SELL), Price Target, Confidence (0-1). "
            "Be specific, data-driven, and concise."
        )

        initial_message = (
            f"Research question: {question}\n"
            f"Ticker: {ticker}\n"
            f"{prior_context}\n\n"
            f"Data gathered so far:\n\n" + "\n\n".join(tool_summaries)
        )

        messages = [{"role": "user", "content": initial_message}]
        anthropic_tools = self._registry.get_anthropic_tools()
        total_tokens = 0
        step = 0

        while step < max_steps:
            step += 1
            try:
                response = self._client.messages.create(
                    model=_ANTHROPIC_MODEL,
                    max_tokens=_MAX_TOKENS,
                    system=system_prompt,
                    tools=anthropic_tools,
                    messages=messages,
                )
                total_tokens += response.usage.input_tokens + response.usage.output_tokens

                # Check stop reason
                if response.stop_reason == "end_turn":
                    # Extract final text synthesis
                    text_content = " ".join(
                        block.text for block in response.content if hasattr(block, "text")
                    )
                    report = self._report_gen.build_from_llm_response(
                        ticker, text_content, all_tool_results, total_tokens
                    )
                    report.job_id = job_id
                    report.tokens_used = total_tokens
                    return report

                elif response.stop_reason == "tool_use":
                    # Execute requested tools
                    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results_for_claude = []
                    for tool_block in tool_use_blocks:
                        t_name = tool_block.name
                        t_params = tool_block.input or {}
                        # Always inject ticker if missing
                        if "ticker" not in t_params:
                            t_params["ticker"] = ticker

                        logger.info("claude_requested_tool", tool=t_name, ticker=ticker)
                        t_result = self._executor.execute_tool(t_name, t_params)
                        all_tool_results[t_name] = t_result

                        tool_results_for_claude.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": json.dumps(t_result.data, default=str)[:2000],
                        })

                    messages.append({"role": "user", "content": tool_results_for_claude})

                else:
                    # Unexpected stop — synthesize from what we have
                    break

            except Exception as exc:
                logger.error("claude_api_error", error=str(exc), step=step)
                # Fall back to offline report
                report = self._report_gen.build_mock_report(ticker, question, all_tool_results)
                report.job_id = job_id
                report.tokens_used = total_tokens
                return report

        # Max steps reached — build from accumulated data
        report = self._report_gen.build_mock_report(ticker, question, all_tool_results)
        report.job_id = job_id
        report.steps_taken = step
        report.tokens_used = total_tokens
        return report


# ---------------------------------------------------------------------------
# ── 7. FastAPI Router ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

research_router = APIRouter(prefix="/research", tags=["research"])

_orchestrator = ResearchAgentOrchestrator()
_memory = _orchestrator._memory


class ResearchRunRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    question: str = Field("Provide a comprehensive investment analysis", description="Research question")
    max_steps: int = Field(10, ge=1, le=15)
    async_mode: bool = Field(False, description="Run in background and return job_id")


class WorkflowRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    question: Optional[str] = Field(None, description="Override default question")


class ResearchRunResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    report: Optional[Dict[str, Any]] = None
    message: str = ""


def _run_research_job(job_id: str, ticker: str, question: str, max_steps: int) -> None:
    """Background task executor."""
    try:
        _orchestrator.run_research(ticker, question, max_steps=max_steps, job_id=job_id)
    except Exception as exc:
        logger.error("background_research_failed", job_id=job_id, error=str(exc))


@research_router.post("/run", response_model=ResearchRunResponse)
def run_research(req: ResearchRunRequest, background_tasks: BackgroundTasks) -> ResearchRunResponse:
    """Launch a research session. Returns immediately with job_id if async_mode=True."""
    job_id = str(uuid.uuid4())[:8]

    if req.async_mode:
        _memory.create_job(job_id, req.ticker.upper(), req.question)
        background_tasks.add_task(
            _run_research_job, job_id, req.ticker.upper(), req.question, req.max_steps
        )
        return ResearchRunResponse(
            job_id=job_id, ticker=req.ticker.upper(),
            status="running", message="Research started in background. Poll /research/status/{job_id}",
        )

    # Synchronous
    try:
        report = _orchestrator.run_research(
            req.ticker, req.question, max_steps=req.max_steps, job_id=job_id
        )
        return ResearchRunResponse(
            job_id=job_id, ticker=req.ticker.upper(),
            status="completed", report=_report_to_dict(report),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@research_router.get("/status/{job_id}", response_model=ResearchRunResponse)
def get_status(job_id: str) -> ResearchRunResponse:
    """Poll the status of an async research job."""
    job = _memory.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return ResearchRunResponse(
        job_id=job_id,
        ticker=job["ticker"],
        status=job["status"],
        report=job.get("result"),
        message=job.get("error") or "",
    )


@research_router.get("/report/{ticker}")
def get_latest_report(ticker: str) -> Dict[str, Any]:
    """Return the most recent completed research report for a ticker."""
    history = _memory.get_history(ticker=ticker.upper(), limit=5)
    completed = [h for h in history if h["status"] == "completed"]
    if not completed:
        raise HTTPException(status_code=404, detail=f"No completed research found for {ticker.upper()}")
    job = _memory.get_job(completed[0]["job_id"])
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Report data unavailable")
    return job["result"]


@research_router.get("/history")
def get_history(
    ticker: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return research job history, optionally filtered by ticker."""
    history = _memory.get_history(ticker=ticker.upper() if ticker else None, limit=limit)
    return {"history": history, "count": len(history)}


@research_router.post("/workflow/{template}")
def run_workflow(
    template: str,
    req: WorkflowRequest,
    background_tasks: BackgroundTasks,
) -> ResearchRunResponse:
    """Run a pre-defined research workflow template."""
    if template not in WORKFLOW_TEMPLATES:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template}' not found. Available: {list(WORKFLOW_TEMPLATES)}",
        )
    wf = WORKFLOW_TEMPLATES[template]
    question = req.question or f"{wf['description']} for {req.ticker.upper()}"
    job_id = str(uuid.uuid4())[:8]

    _memory.create_job(job_id, req.ticker.upper(), question)
    background_tasks.add_task(
        lambda: _orchestrator.run_research(
            req.ticker, question,
            max_steps=wf["max_steps"],
            workflow=template,
            job_id=job_id,
        )
    )
    return ResearchRunResponse(
        job_id=job_id, ticker=req.ticker.upper(),
        status="running",
        message=f"Workflow '{template}' started. Estimated cost ${wf['estimated_cost_usd']:.2f}. Poll /research/status/{job_id}",
    )


@research_router.get("/workflows")
def list_workflows() -> Dict[str, Any]:
    """List available research workflow templates."""
    return {
        "workflows": [
            {"name": k, **{kk: vv for kk, vv in v.items() if kk != "tools"}}
            for k, v in WORKFLOW_TEMPLATES.items()
        ]
    }


@research_router.get("/tools")
def list_tools() -> Dict[str, Any]:
    """List all research tools available to the agent."""
    return {
        "tools": [
            {"name": t.name, "description": t.description, "params": list(t.params.keys())}
            for t in RESEARCH_TOOLS
        ]
    }


# ---------------------------------------------------------------------------
# ── CLI convenience ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def quick_research(ticker: str, question: str = "What is the investment outlook?") -> ResearchReport:
    """Top-level convenience function for quick research."""
    orch = ResearchAgentOrchestrator()
    return orch.run_research(ticker, question, max_steps=5, workflow="quick_screen")


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Provide a comprehensive investment analysis"

    print(f"\nResearching {ticker}: {question}\n{'='*60}")
    orch = ResearchAgentOrchestrator()
    report = orch.run_research(ticker, question, max_steps=8)

    print(f"Recommendation   : {report.recommendation} (confidence {report.confidence:.0%})")
    print(f"Price Target     : ${report.price_target or 'N/A'}")
    print(f"Sentiment Score  : {report.sentiment_score:+.2f}")
    print(f"Steps Taken      : {report.steps_taken}")
    print(f"Tokens Used      : {report.tokens_used}")
    print(f"\nExecutive Summary:\n{report.executive_summary}")
    print(f"\nInvestment Thesis:\n{report.investment_thesis}")
    print(f"\nTechnical Outlook:\n{report.technical_outlook}")
    print(f"\nTop Risks:")
    for r in report.risks[:3]:
        print(f"  - {r}")
    print(f"\nKey Catalysts:")
    for c in report.catalysts[:3]:
        print(f"  + {c}")
    print(f"\nKey Metrics: {json.dumps(report.key_metrics, default=str, indent=2)}")
    print(f"\nData Sources: {', '.join(report.sources)}")
