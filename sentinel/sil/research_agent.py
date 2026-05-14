"""
SIL Research Agent — SENTINEL.

Autonomous financial research agent using Claude. Gathers data in parallel then
synthesises a structured ResearchMemo answering a research question. Dimension 60.
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import anthropic
import httpx
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ── Pydantic Models ───────────────────────────────────────────────────────────

class SourcedFact(BaseModel):
    fact: str
    source: str    # "EDGAR", "FRED", "Polygon", "Form4", etc.
    confidence: str  # "high", "medium", "low"


class ValuationView(BaseModel):
    method: str    # "DCF", "Comps", "Asset-based"
    implied_value: float | None = None
    current_price: float | None = None
    upside_pct: float | None = None
    assumptions: str = ""


class ResearchMemo(BaseModel):
    ticker: str
    question: str
    executive_summary: str
    key_facts: list[SourcedFact]
    bull_case: list[str]
    bear_case: list[str]
    valuation: list[ValuationView]
    data_quality: str    # "high" / "medium" / "low"
    recommendation: str  # "Buy" / "Sell" / "Hold" / "Insufficient data"
    confidence: float
    sources_used: list[str]
    generated_at: datetime


# ── Constants ─────────────────────────────────────────────────────────────────

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FRED_BASE = "https://api.stlouisfed.org/fred"
_DEFAULT_UA = "SENTINEL-Research sentinel@example.com"

_XBRL_CONCEPTS: dict[str, str] = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "NetIncomeLoss": "net_income",
    "EarningsPerShareDiluted": "eps_diluted",
    "LongTermDebt": "long_term_debt",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "GrossProfit": "gross_profit",
    "Assets": "total_assets",
    "StockholdersEquity": "stockholders_equity",
}

_MACRO_SERIES: dict[str, str] = {
    "T10Y2Y": "10Y-2Y Yield Spread",
    "VIXCLS": "VIX",
    "FEDFUNDS": "Fed Funds Rate",
    "UNRATE": "Unemployment Rate",
    "T10YIE": "10Y Breakeven Inflation",
}

_MEMO_SCHEMA = (
    '{"executive_summary":"2-3 sentences","key_facts":[{"fact":"...","source":"EDGAR|FRED|yfinance","confidence":"high|medium|low"}],'
    '"bull_case":["bullet"],"bear_case":["bullet"],'
    '"valuation":[{"method":"Comps|DCF","implied_value":null,"current_price":null,"upside_pct":null,"assumptions":"..."}],'
    '"data_quality":"high|medium|low","recommendation":"Buy|Sell|Hold|Insufficient data","confidence":0.0}'
)


# ── Data Fetchers ─────────────────────────────────────────────────────────────

async def _resolve_cik(ticker: str) -> str | None:
    headers = {"User-Agent": _DEFAULT_UA, "Accept-Encoding": "gzip, deflate"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(_EDGAR_TICKERS_URL, headers={**headers, "Host": "www.sec.gov"})
            r.raise_for_status()
        for e in r.json().values():
            if e.get("ticker", "").upper() == ticker.upper():
                return str(e["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("research_agent._resolve_cik", ticker=ticker, error=str(exc))
    return None


async def _fetch_fundamentals(ticker: str, settings: Any) -> dict[str, Any]:
    """Fetch EDGAR XBRL companyfacts — revenue, EPS, debt, margins."""
    result: dict[str, Any] = {"source": "EDGAR", "ticker": ticker}
    ua = getattr(settings, "edgar_user_agent", _DEFAULT_UA)
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
    try:
        cik = await _resolve_cik(ticker)
        if not cik:
            result["error"] = f"CIK not found for {ticker}"
            return result
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json", headers=headers)
            r.raise_for_status()
            data = r.json()
        result["entity_name"] = data.get("entityName", ticker)
        result["cik"] = cik
        gaap = data.get("facts", {}).get("us-gaap", {})
        extracted: dict[str, Any] = {}
        for concept, fname in _XBRL_CONCEPTS.items():
            if fname in extracted:
                continue
            for unit in ("USD", "USD/shares", "shares"):
                entries = [e for e in gaap.get(concept, {}).get("units", {}).get(unit, [])
                           if e.get("form") in ("10-K", "10-K/A") and e.get("end")]
                if entries:
                    e = max(entries, key=lambda x: x["end"])
                    extracted[fname] = {"value": e.get("val"), "period_end": e.get("end"), "unit": unit}
                    break
        result["fundamentals"] = extracted
        # Derived ratios
        rev = (extracted.get("revenue") or {}).get("value")
        gp = (extracted.get("gross_profit") or {}).get("value")
        ni = (extracted.get("net_income") or {}).get("value")
        equity = (extracted.get("stockholders_equity") or {}).get("value")
        if rev and gp and rev > 0:
            result["gross_margin"] = round(gp / rev, 4)
        if rev and ni and rev > 0:
            result["net_margin"] = round(ni / rev, 4)
        if equity and ni and equity > 0:
            result["roe"] = round(ni / equity, 4)
        # YoY revenue growth from up to 2 annual filings
        rev_entries = sorted(
            [e for e in gaap.get("Revenues", {}).get("units", {}).get("USD", [])
             if e.get("form") in ("10-K", "10-K/A") and e.get("end")],
            key=lambda x: x["end"], reverse=True,
        )[:2]
        if len(rev_entries) == 2:
            r0, r1 = rev_entries[0].get("val", 0), rev_entries[1].get("val", 0)
            if r1 and r1 > 0:
                result["revenue_growth_yoy"] = round((r0 - r1) / r1, 4)
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("research_agent._fetch_fundamentals", ticker=ticker, error=str(exc))
    return result


async def _fetch_news(ticker: str, settings: Any) -> list[dict[str, Any]]:
    """Fetch recent Finnhub news with keyword-based sentiment."""
    key = getattr(settings, "finnhub_api_key", "")
    if not key:
        return []
    try:
        today = datetime.now(timezone.utc).date()
        params = {"symbol": ticker.upper(), "from": (today - timedelta(days=30)).isoformat(),
                  "to": today.isoformat(), "token": key}
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get("https://finnhub.io/api/v1/company-news", params=params)
            r.raise_for_status()
        pos_words = {"strong", "beat", "record", "growth", "upgrade", "raised", "profit", "exceed"}
        neg_words = {"miss", "loss", "cut", "downgrade", "decline", "lawsuit", "fraud", "weak"}
        results = []
        for a in r.json()[:10]:
            tokens = (a.get("headline", "") + " " + a.get("summary", "")).lower().split()
            p, n = sum(1 for t in tokens if t in pos_words), sum(1 for t in tokens if t in neg_words)
            results.append({"headline": a.get("headline", ""), "source": a.get("source", ""),
                            "sentiment": "positive" if p > n else ("negative" if n > p else "neutral")})
        return results
    except Exception as exc:
        logger.warning("research_agent._fetch_news", ticker=ticker, error=str(exc))
        return []


async def _fetch_insider_trades(ticker: str) -> dict[str, Any]:
    """Query EDGAR full-text search for Form 4 filings in last 180 days."""
    result: dict[str, Any] = {"source": "Form4/EDGAR", "ticker": ticker}
    try:
        today = datetime.now(timezone.utc).date()
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker.upper()}%22"
               f"&dateRange=custom&startdt={(today - timedelta(days=180)).isoformat()}"
               f"&enddt={today.isoformat()}&forms=4")
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _DEFAULT_UA}) as c:
            r = await c.get(url)
            r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        result["filing_count_180d"] = len(hits)
        result["note"] = (f"{len(hits)} Form 4 filings in 180 days. Full XML parsing needed for buy/sell breakdown.")
    except Exception as exc:
        logger.warning("research_agent._fetch_insider_trades", ticker=ticker, error=str(exc))
        result["filing_count_180d"] = 0
        result["error"] = str(exc)
    return result


async def _fetch_price_history(ticker: str) -> dict[str, Any]:
    """1-year return, volatility, 52w range, valuation ratios via yfinance."""
    result: dict[str, Any] = {"source": "yfinance", "ticker": ticker}
    try:
        import yfinance as yf
        loop = asyncio.get_event_loop()

        def _dl() -> dict[str, Any]:
            t = yf.Ticker(ticker.upper())
            hist = t.history(period="1y")
            if hist.empty:
                return {"error": "No yfinance data"}
            cur, y1 = float(hist["Close"].iloc[-1]), float(hist["Close"].iloc[0])
            ret = hist["Close"].pct_change().dropna()
            info = t.info
            return {
                "current_price": round(cur, 2),
                "return_1y": round((cur - y1) / y1, 4) if y1 > 0 else 0.0,
                "volatility_ann": round(float(ret.std() * math.sqrt(252)), 4),
                "high_52w": round(float(hist["High"].max()), 2),
                "low_52w": round(float(hist["Low"].min()), 2),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
            }

        result.update(await loop.run_in_executor(None, _dl))
    except ImportError:
        result["error"] = "yfinance not installed"
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("research_agent._fetch_price_history", ticker=ticker, error=str(exc))
    return result


async def _fetch_macro_context() -> dict[str, Any]:
    """Key FRED macro series + qualitative regime assessment."""
    result: dict[str, Any] = {"source": "FRED", "series": {}}
    try:
        from sentinel.core.config import get_settings
        fred_key = get_settings().fred_api_key
    except Exception:
        fred_key = ""

    for sid, name in _MACRO_SERIES.items():
        try:
            params: dict[str, Any] = {"series_id": sid, "sort_order": "desc", "limit": 1, "file_type": "json"}
            if fred_key:
                params["api_key"] = fred_key
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{_FRED_BASE}/series/observations", params=params)
                r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs and obs[0].get("value") != ".":
                result["series"][sid] = {"name": name, "value": float(obs[0]["value"]), "date": obs[0].get("date")}
        except Exception:
            continue

    series, notes = result["series"], []
    spread = (series.get("T10Y2Y") or {}).get("value")
    vix = (series.get("VIXCLS") or {}).get("value")
    fed = (series.get("FEDFUNDS") or {}).get("value")
    if spread is not None:
        notes.append("Yield curve inverted — recessionary signal." if spread < 0 else
                     ("Yield curve flat — late-cycle." if spread < 0.5 else f"Yield curve {spread:.2f}% — supportive."))
    if vix is not None:
        notes.append(f"VIX {vix:.1f} — {'elevated fear' if vix > 30 else ('moderate caution' if vix > 20 else 'risk-on')}.")
    if fed is not None:
        notes.append(f"Fed funds {fed:.2f}% — {'restrictive' if fed > 4 else ('neutral' if fed > 2 else 'accommodative')}.")
    result["qualitative_assessment"] = notes
    result["data_available"] = bool(series)
    return result


# ── Context Builder ───────────────────────────────────────────────────────────

def _build_context(ticker: str, fundamentals: dict, news: list, insider: dict,
                   price: dict, macro: dict, current_price: float | None) -> str:
    parts = [f"# Data for {ticker}"]

    parts.append("\n## Price & Market Data")
    if price.get("error"):
        parts.append(f"Unavailable: {price['error']}")
    else:
        cp = current_price or price.get("current_price")
        if cp:
            parts.append(f"- Price: ${cp}")
        for k, label in [("return_1y", "1Y return"), ("volatility_ann", "Ann vol"),
                         ("high_52w", "52w high"), ("low_52w", "52w low")]:
            if price.get(k) is not None:
                v = price[k]
                parts.append(f"- {label}: {v:.1%}" if "return" in k or "vol" in k else f"- {label}: ${v}")
        mc = price.get("market_cap")
        if mc:
            parts.append(f"- Market cap: ${mc/1e9:.1f}B" if mc > 1e9 else f"- Market cap: ${mc/1e6:.0f}M")
        for k, label in [("pe_ratio", "P/E"), ("forward_pe", "Fwd P/E"),
                         ("ev_ebitda", "EV/EBITDA"), ("pb_ratio", "P/B"), ("beta", "Beta")]:
            if price.get(k):
                parts.append(f"- {label}: {price[k]:.2f}x")
        if price.get("sector"):
            parts.append(f"- Sector: {price['sector']} / {price.get('industry','')}")

    parts.append("\n## EDGAR Fundamentals (Latest Annual)")
    if fundamentals.get("error"):
        parts.append(f"Unavailable: {fundamentals['error']}")
    else:
        for fname, fdata in (fundamentals.get("fundamentals") or {}).items():
            v = fdata.get("value")
            if v is None:
                continue
            unit = fdata.get("unit", "USD")
            s = f"${v/1e9:.2f}B" if unit == "USD" and abs(v) > 1e9 else (f"${v/1e6:.1f}M" if unit == "USD" and abs(v) > 1e6 else str(v))
            parts.append(f"- {fname}: {s} ({fdata.get('period_end','')})")
        for k, label in [("gross_margin", "Gross margin"), ("net_margin", "Net margin"),
                         ("roe", "ROE"), ("revenue_growth_yoy", "Rev growth YoY")]:
            if fundamentals.get(k) is not None:
                parts.append(f"- {label}: {fundamentals[k]:.1%}")

    parts.append("\n## Recent News (30d)")
    if not news:
        parts.append("Unavailable (no Finnhub key).")
    else:
        pos = sum(1 for n in news if n["sentiment"] == "positive")
        neg = sum(1 for n in news if n["sentiment"] == "negative")
        parts.append(f"- {len(news)} articles: {pos} positive, {neg} negative, {len(news)-pos-neg} neutral")
        for n in news[:5]:
            m = {"positive": "+", "negative": "-"}.get(n["sentiment"], "~")
            parts.append(f"  [{m}] {n['headline'][:120]} ({n['source']})")

    parts.append("\n## Insider Activity (180d Form 4)")
    parts.append(f"- Filings: {insider.get('filing_count_180d', 0)}")
    if insider.get("note"):
        parts.append(f"- {insider['note']}")

    parts.append("\n## Macro Context")
    if not macro.get("data_available"):
        parts.append("FRED data unavailable.")
    else:
        for note in macro.get("qualitative_assessment", []):
            parts.append(f"- {note}")
        for sid, sd in (macro.get("series") or {}).items():
            parts.append(f"  [{sid}] {sd['name']}: {sd['value']} ({sd['date']})")

    return "\n".join(parts)


# ── Response Parser ───────────────────────────────────────────────────────────

def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _parse_response(ticker: str, question: str, raw: str, sources: list[str]) -> ResearchMemo:
    text = raw.strip()
    for fence in ("```json", "```"):
        if fence in text:
            s = text.find(fence) + len(fence)
            e = text.find("```", s)
            if e > s:
                text = text[s:e].strip()
                break
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ResearchMemo(
            ticker=ticker, question=question,
            executive_summary=raw[:400] or "Response parse failed.",
            key_facts=[SourcedFact(fact="JSON parse failure.", source="system", confidence="low")],
            bull_case=["Insufficient data."], bear_case=["JSON parse failure."],
            valuation=[ValuationView(method="Insufficient data")],
            data_quality="low", recommendation="Insufficient data", confidence=0.1,
            sources_used=sources, generated_at=datetime.now(timezone.utc),
        )

    key_facts = [SourcedFact(fact=str(f.get("fact", "")), source=str(f.get("source", "")),
                              confidence=str(f.get("confidence", "medium")))
                 for f in data.get("key_facts", []) if isinstance(f, dict)]
    valuation = [ValuationView(method=str(v.get("method", "Comps")),
                               implied_value=_safe_float(v.get("implied_value")),
                               current_price=_safe_float(v.get("current_price")),
                               upside_pct=_safe_float(v.get("upside_pct")),
                               assumptions=str(v.get("assumptions", "")))
                 for v in data.get("valuation", []) if isinstance(v, dict)]

    bull = [str(b) for b in data.get("bull_case", []) if b] or ["No bull case identified."]
    bear = [str(b) for b in data.get("bear_case", []) if b] or ["No bear case identified."]
    if not valuation:
        valuation = [ValuationView(method="Insufficient data", assumptions="Not enough data for valuation.")]

    conf = max(0.0, min(1.0, _safe_float(data.get("confidence")) or 0.5))
    return ResearchMemo(
        ticker=ticker, question=question,
        executive_summary=str(data.get("executive_summary", "")),
        key_facts=key_facts, bull_case=bull, bear_case=bear, valuation=valuation,
        data_quality=str(data.get("data_quality", "medium")),
        recommendation=str(data.get("recommendation", "Insufficient data")),
        confidence=conf, sources_used=sources, generated_at=datetime.now(timezone.utc),
    )


# ── Fallback & Null helpers ───────────────────────────────────────────────────

def _fallback_memo(ticker: str, question: str, reason: str) -> ResearchMemo:
    return ResearchMemo(
        ticker=ticker, question=question,
        executive_summary=(
            f"Unable to generate research for {ticker}. {reason} "
            "ANTHROPIC_API_KEY + data source keys required."
        ),
        key_facts=[SourcedFact(fact=f"Generation failed: {reason}", source="system", confidence="high")],
        bull_case=["Insufficient data."], bear_case=["Generation failed."],
        valuation=[ValuationView(method="Insufficient data",
                                 assumptions="API keys required: ANTHROPIC_API_KEY + FINNHUB_API_KEY.")],
        data_quality="low", recommendation="Insufficient data", confidence=0.1,
        sources_used=[], generated_at=datetime.now(timezone.utc),
    )


class _NullSettings:
    edgar_user_agent: str = _DEFAULT_UA
    finnhub_api_key: str = ""
    polygon_api_key: str = ""
    fred_api_key: str = ""


# ── Main Function ─────────────────────────────────────────────────────────────

async def research_ticker(
    ticker: str,
    question: str,
    anthropic_api_key: str,
    current_price: float | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> ResearchMemo:
    """
    Gather financial data in parallel, then call Claude to produce a ResearchMemo.
    Always returns a valid ResearchMemo — never raises.
    """
    ticker = ticker.upper().strip()
    if not anthropic_api_key:
        logger.warning("research_agent: no API key")
        return _fallback_memo(ticker, question, "No ANTHROPIC_API_KEY provided.")

    try:
        from sentinel.core.config import get_settings
        settings = get_settings()
    except Exception:
        settings = _NullSettings()

    # Parallel data gather
    logger.info("research_agent: gathering data", ticker=ticker)
    gathered = await asyncio.gather(
        _fetch_fundamentals(ticker, settings),
        _fetch_news(ticker, settings),
        _fetch_insider_trades(ticker),
        _fetch_price_history(ticker),
        _fetch_macro_context(),
        return_exceptions=True,
    )

    sources: list[str] = []
    names = ["EDGAR", "Finnhub", "Form4/EDGAR", "yfinance", "FRED"]
    fundamentals, news, insider, price, macro = {}, [], {}, {}, {}
    for i, (name, res) in enumerate(zip(names, gathered)):
        if isinstance(res, Exception):
            logger.warning("research_agent: fetcher failed", source=name, error=str(res))
        else:
            sources.append(name)
            if i == 0: fundamentals = res  # type: ignore[assignment]
            elif i == 1: news = res        # type: ignore[assignment]
            elif i == 2: insider = res     # type: ignore[assignment]
            elif i == 3: price = res       # type: ignore[assignment]
            elif i == 4: macro = res       # type: ignore[assignment]

    context = _build_context(ticker, fundamentals, news, insider, price, macro, current_price)
    logger.info("research_agent: context ready", chars=len(context), sources=sources)

    system = (
        f"You are an institutional equity research analyst. You have data about {ticker}. "
        f"Answer the research question '{question}' using ONLY the provided data. "
        f"Be quantitative, specific, honest about gaps. Never invent numbers. "
        f"Output ONLY a JSON object matching this schema: {_MEMO_SCHEMA}"
    )
    user_msg = f"Research question: {question}\n\nTicker: {ticker}\n\n{context}\n\nOutput JSON only."

    try:
        client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        resp = await client.messages.create(model=model, max_tokens=4096,
                                            system=system, messages=[{"role": "user", "content": user_msg}])
        raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
    except anthropic.APIError as exc:
        logger.error("research_agent: API error", error=str(exc))
        return _fallback_memo(ticker, question, f"API error: {exc}")
    except Exception as exc:
        logger.error("research_agent: unexpected error", error=str(exc))
        return _fallback_memo(ticker, question, f"Unexpected: {exc}")

    return _parse_response(ticker, question, raw, sources)


def research_ticker_sync(
    ticker: str,
    question: str,
    anthropic_api_key: str,
    current_price: float | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> ResearchMemo:
    """Synchronous wrapper for non-async callers."""
    return asyncio.run(research_ticker(ticker, question, anthropic_api_key, current_price, model))
