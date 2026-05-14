"""Intelligence routes — insider trades, institutional holdings, news/sentiment,
congressional trades, COT signals, and macro regime.

All endpoints are DB-first: query the data lake, fall back to live adapters
on cache miss, and persist the result so the next call is served from DB.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.sds.db import get_session

router = APIRouter()


# ── Insider Transactions (Form 4) ─────────────────────────────────────────────

@router.get("/insider/{ticker}")
async def insider_transactions(
    ticker: str,
    days: int = Query(365, ge=1, le=1825, description="Lookback window in days"),
    is_derivative: Optional[bool] = Query(None, description="Filter derivatives (options/warrants)"),
    session: AsyncSession = Depends(get_session),
):
    """Recent Form 4 insider transactions. DB-first; triggers EDGAR ingest on miss.

    Returns purchases and sales by officers, directors, and 10% owners.
    Disposals are shown as negative share counts.
    """
    from sentinel.sds import repository
    from sentinel.sds.ingest import ingest_insider_transactions
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.core.config import get_settings

    since_dt = datetime.utcnow() - timedelta(days=days)
    rows = await repository.get_insider_transactions(
        session=session,
        ticker=ticker.upper(),
        since=since_dt,
        is_derivative=is_derivative,
        limit=200,
    )
    source = "db"

    if not rows:
        # Try to ingest from EDGAR — requires CIK lookup
        try:
            s = get_settings()
            edgar = EDGARAdapter(user_agent=s.edgar_user_agent)
            await edgar.load_company_tickers()
            cik = edgar.ticker_to_cik(ticker.upper())
            if cik:
                result = await ingest_insider_transactions(
                    cik=cik, ticker=ticker.upper(), session=session,
                    since=since_dt.date(), limit=40,
                )
                if result.ok:
                    rows = await repository.get_insider_transactions(
                        session=session,
                        ticker=ticker.upper(),
                        since=since_dt,
                        is_derivative=is_derivative,
                        limit=200,
                    )
                    source = "edgar_live"
        except Exception:
            pass

    # Aggregate: buys vs sells by role
    buys = [r for r in rows if r.get("shares") and r["shares"] > 0]
    sells = [r for r in rows if r.get("shares") and r["shares"] < 0]
    buy_value = sum(r.get("value") or 0 for r in buys)
    sell_value = sum(abs(r.get("value") or 0) for r in sells)

    return {
        "ticker": ticker.upper(),
        "days": days,
        "source": source,
        "summary": {
            "total": len(rows),
            "buys": len(buys),
            "sells": len(sells),
            "buy_value_usd": round(buy_value, 0),
            "sell_value_usd": round(sell_value, 0),
            "net_sentiment": "bullish" if buy_value > sell_value else ("bearish" if sell_value > buy_value else "neutral"),
        },
        "transactions": [
            {
                "owner": r["owner_name"],
                "role": r["role"],
                "date": r["tx_date"].isoformat() if hasattr(r["tx_date"], "isoformat") else str(r["tx_date"]),
                "code": r["tx_code"],
                "shares": float(r["shares"]) if r.get("shares") is not None else None,
                "price": float(r["price_per_share"]) if r.get("price_per_share") else None,
                "value": float(r["value"]) if r.get("value") else None,
                "security": r.get("security_title"),
                "is_derivative": r.get("is_derivative"),
            }
            for r in rows[:100]
        ],
    }


# ── Institutional Holdings (13F-HR) ──────────────────────────────────────────

@router.get("/institutional/{ticker}")
async def institutional_holdings(
    ticker: str,
    limit: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Institutional ownership from 13F-HR filings. DB-first, latest period shown."""
    from sentinel.sds import repository

    rows = await repository.get_institutional_holdings(
        session=session, ticker=ticker.upper(), limit=limit
    )

    if not rows:
        return {
            "ticker": ticker.upper(),
            "holdings": [],
            "note": "No institutional holdings data. Run: make backfill-institutional",
        }

    # Latest period is first (ORDER BY period DESC in repository)
    latest_period = rows[0]["period_of_report"] if rows else None
    total_shares = sum(r.get("shares") or 0 for r in rows if r.get("shares"))
    total_value = sum(r.get("market_value") or 0 for r in rows if r.get("market_value"))

    return {
        "ticker": ticker.upper(),
        "period": latest_period.isoformat() if hasattr(latest_period, "isoformat") else str(latest_period),
        "summary": {
            "institutions": len(rows),
            "total_shares": round(total_shares, 0),
            "total_value_usd": round(total_value, 0),
        },
        "holdings": [
            {
                "manager_cik": r["manager_cik"],
                "issuer_name": r["issuer_name"],
                "cusip": r["cusip"],
                "market_value": float(r["market_value"]) if r.get("market_value") else None,
                "shares": float(r["shares"]) if r.get("shares") else None,
                "share_type": r.get("share_type"),
                "put_call": r.get("put_call"),
                "discretion": r.get("investment_discretion"),
                "period": r["period_of_report"].isoformat() if hasattr(r.get("period_of_report"), "isoformat") else str(r.get("period_of_report")),
                "filed": r["filed_date"].isoformat() if r.get("filed_date") and hasattr(r["filed_date"], "isoformat") else None,
            }
            for r in rows
        ],
    }


# ── News & Sentiment ──────────────────────────────────────────────────────────

@router.get("/news/{ticker}")
async def news_sentiment(
    ticker: str,
    days: int = Query(7, ge=1, le=90),
    sentiment: Optional[str] = Query(None, description="Filter: positive / negative / neutral"),
    session: AsyncSession = Depends(get_session),
):
    """News with FinBERT sentiment. DB-first; live Finnhub ingest on miss.

    FinBERT model is lazy-loaded on first request (~2s cold start).
    """
    from sentinel.sds import repository
    from sentinel.sds.ingest import ingest_news

    since_dt = datetime.utcnow() - timedelta(days=days)
    rows = await repository.get_news_articles(
        session=session,
        ticker=ticker.upper(),
        since=since_dt,
        sentiment=sentiment,
        limit=100,
    )
    source = "db"

    if not rows:
        from_date = (datetime.utcnow() - timedelta(days=days)).date()
        result = await ingest_news(
            ticker=ticker.upper(), session=session,
            from_date=from_date, to_date=date.today(),
        )
        if result.ok:
            rows = await repository.get_news_articles(
                session=session,
                ticker=ticker.upper(),
                since=since_dt,
                sentiment=sentiment,
                limit=100,
            )
            source = "finnhub_live"

    if not rows:
        note = "No news data."
        if source == "finnhub_live":
            note = "No Finnhub API key — set FINNHUB_API_KEY in .env"
        return {"ticker": ticker.upper(), "days": days, "articles": [], "note": note}

    pos = sum(1 for r in rows if r.get("sentiment_label") == "positive")
    neg = sum(1 for r in rows if r.get("sentiment_label") == "negative")
    total = len(rows)

    return {
        "ticker": ticker.upper(),
        "days": days,
        "source": source,
        "aggregate": {
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": total - pos - neg,
            "positive_pct": round(pos / total * 100, 1) if total else 0,
            "negative_pct": round(neg / total * 100, 1) if total else 0,
            "net_sentiment": round((pos - neg) / total, 3) if total else 0,
        },
        "articles": [
            {
                "headline": r["headline"],
                "source": r.get("source"),
                "url": r.get("url"),
                "published": r["published_at"].isoformat() if hasattr(r.get("published_at"), "isoformat") else str(r.get("published_at")),
                "sentiment": r.get("sentiment_label"),
                "score": float(r["sentiment_score"]) if r.get("sentiment_score") is not None else None,
            }
            for r in rows[:50]
        ],
    }


@router.get("/sentiment/{ticker}")
async def sentiment_summary(
    ticker: str,
    days: int = Query(7, ge=1, le=90),
    session: AsyncSession = Depends(get_session),
):
    """Aggregate sentiment summary only (no article list). Lighter than /news/{ticker}."""
    from sentinel.sds import repository

    since_dt = datetime.utcnow() - timedelta(days=days)
    rows = await repository.get_news_articles(
        session=session, ticker=ticker.upper(), since=since_dt, limit=200
    )

    if not rows:
        # Lightweight fallback: call Finnhub directly without persisting
        try:
            from sentinel.sds.adapters.finnhub_adapter import FinnhubAdapter
            from sentinel.sil.sentiment import score_sentiment_batch, aggregate_sentiment
            from sentinel.core.config import get_settings
            s = get_settings()
            fh = FinnhubAdapter(api_key=s.finnhub_api_key)
            from_str = (date.today() - timedelta(days=days)).isoformat()
            to_str = date.today().isoformat()
            news = await fh.fetch_news(ticker, from_str, to_str)
            headlines = [n.get("headline", "") for n in news]
            sentiments = await score_sentiment_batch(headlines)
            agg = aggregate_sentiment(sentiments)
            return {"ticker": ticker.upper(), "days": days, "aggregate": agg, "source": "live"}
        except Exception:
            return {"ticker": ticker.upper(), "days": days, "aggregate": {}}

    pos = sum(1 for r in rows if r.get("sentiment_label") == "positive")
    neg = sum(1 for r in rows if r.get("sentiment_label") == "negative")
    total = len(rows)
    avg_score = sum(r.get("sentiment_score") or 0 for r in rows) / total if total else 0

    return {
        "ticker": ticker.upper(),
        "days": days,
        "source": "db",
        "aggregate": {
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": total - pos - neg,
            "positive_pct": round(pos / total * 100, 1) if total else 0,
            "negative_pct": round(neg / total * 100, 1) if total else 0,
            "net_sentiment": round((pos - neg) / total, 3) if total else 0,
            "avg_score": round(avg_score, 3),
        },
    }


# ── Congressional Trades ──────────────────────────────────────────────────────

@router.get("/congressional")
async def congressional_trades(
    ticker: Optional[str] = None,
    days: int = Query(90, ge=1, le=730),
    chamber: Optional[str] = Query(None, description="house / senate"),
    session: AsyncSession = Depends(get_session),
):
    """STOCK Act congressional trade disclosures. DB-first with live fallback."""
    from sentinel.sds import repository
    from sentinel.sds.ingest import ingest_congressional_trades

    since_dt = datetime.utcnow() - timedelta(days=days)
    rows = await repository.get_congressional_trades(
        session=session,
        ticker=ticker.upper() if ticker else None,
        since=since_dt,
        chamber=chamber,
        limit=500,
    )
    source = "db"

    if not rows:
        since_date = since_dt.date()
        result = await ingest_congressional_trades(session=session, since=since_date)
        if result.ok:
            rows = await repository.get_congressional_trades(
                session=session,
                ticker=ticker.upper() if ticker else None,
                since=since_dt,
                chamber=chamber,
                limit=500,
            )
            source = "live"

    # Signal: politicians who bought most in last 90 days
    buys = [r for r in rows if (r.get("tx_code") or "") in ("P", "purchase")]
    sells = [r for r in rows if (r.get("tx_code") or "") in ("S", "sale")]

    from collections import Counter
    top_buyers = Counter(r["politician_name"] for r in buys).most_common(10)
    top_sellers = Counter(r["politician_name"] for r in sells).most_common(10)

    return {
        "source": source,
        "days": days,
        "ticker_filter": ticker,
        "chamber_filter": chamber,
        "summary": {
            "total": len(rows),
            "buys": len(buys),
            "sells": len(sells),
            "late_filings": sum(1 for r in rows if r.get("late_filing")),
        },
        "top_buyers": [{"name": n, "trades": c} for n, c in top_buyers],
        "top_sellers": [{"name": n, "trades": c} for n, c in top_sellers],
        "trades": [
            {
                "politician": r["politician_name"],
                "chamber": r.get("chamber"),
                "party": r.get("party"),
                "state": r.get("state"),
                "ticker": r.get("ticker"),
                "date": r["tx_date"].isoformat() if hasattr(r["tx_date"], "isoformat") else str(r["tx_date"]),
                "type": r.get("tx_code"),
                "amount_low": float(r["amount_low"]) if r.get("amount_low") else None,
                "amount_high": float(r["amount_high"]) if r.get("amount_high") else None,
                "lag_days": r.get("filing_lag_days"),
                "late": r.get("late_filing"),
            }
            for r in rows[:200]
        ],
    }


# ── COT Signals ───────────────────────────────────────────────────────────────

@router.get("/cot")
async def cot_signals(
    market: Optional[str] = Query(None, description="Partial market name filter (e.g. 'GOLD')"),
    session: AsyncSession = Depends(get_session),
):
    """CFTC COT signals — latest managed money positioning for 16 futures markets."""
    from sentinel.sds import repository

    rows = await repository.get_cot_signals(session=session, market_name=market)

    if not rows:
        return {
            "signals": [],
            "note": "No COT data in DB. Run: make backfill-cot",
        }

    return {
        "count": len(rows),
        "market_filter": market,
        "signals": [
            {
                "market": r["market_name"],
                "report_date": r["report_date"].isoformat() if hasattr(r.get("report_date"), "isoformat") else str(r.get("report_date")),
                "net_speculator": r.get("net_speculator"),
                "cot_index": float(r["cot_index"]) if r.get("cot_index") is not None else None,
                "signal": r.get("signal"),
                "open_interest": r.get("open_interest"),
            }
            for r in rows
        ],
    }


# ── Macro Regime ──────────────────────────────────────────────────────────────

@router.get("/regime")
async def macro_regime():
    """Current macro regime from MCP intelligence layer."""
    from sentinel.sil.mcp_server import get_macro_regime
    return await get_macro_regime()


# ─── Options Analytics ────────────────────────────────────────────────────────
@router.get("/options/{ticker}")
async def get_options_analytics(
    ticker: str,
    underlying_price: float = Query(..., description="Current stock price"),
    expiry_days: int = Query(90, ge=7, le=365, description="Expiration window in days"),
):
    """Get IV surface, skew, gamma exposure, max pain for a ticker."""
    from sentinel.core.config import get_settings
    settings = get_settings()

    if not settings.polygon_api_key:
        return {
            "ticker": ticker,
            "underlying_price": underlying_price,
            "contract_count": 0,
            "error": "POLYGON_API_KEY not configured — options analytics unavailable",
        }

    from datetime import date as _date, timedelta as _td
    from sentinel.sds.adapters.polygon_adapter import PolygonAdapter
    from sentinel.sbx.options_analytics import get_options_summary

    today = _date.today()
    exp_max = today + _td(days=expiry_days)

    adapter = PolygonAdapter(api_key=settings.polygon_api_key)
    contracts_raw = await adapter.fetch_options_chain(
        underlying=ticker.upper(),
        expiration_date_gte=today.isoformat(),
        expiration_date_lte=exp_max.isoformat(),
    )

    summary = get_options_summary(ticker.upper(), contracts_raw, underlying_price)

    def _ser(v):
        if hasattr(v, "model_dump"):
            d = v.model_dump()
            return {k: _ser(val) for k, val in d.items()}
        if isinstance(v, list):
            return [_ser(i) for i in v]
        if isinstance(v, dict):
            return {k: _ser(val) for k, val in v.items()}
        return v

    return _ser(summary)

# ─── Social Sentiment ─────────────────────────────────────────────────────────
@router.get("/social/{ticker}")
async def get_social_sentiment(ticker: str):
    """Social media sentiment from Reddit and StockTwits."""
    from sentinel.snm.social_sentiment import get_social_sentiment as _social
    result = await _social(ticker)
    return result.model_dump()

# ─── Economic Calendar ────────────────────────────────────────────────────────
@router.get("/calendar")
async def get_economic_calendar(days_ahead: int = Query(14, ge=1, le=90)):
    """Upcoming macro economic releases with importance scoring."""
    from sentinel.sma.economic_calendar import get_calendar
    from sentinel.core.config import get_settings
    settings = get_settings()
    calendar = await get_calendar(api_key=settings.fred_api_key, days_ahead=days_ahead)
    return calendar.model_dump()

# ─── CB Speech Analysis ───────────────────────────────────────────────────────
@router.get("/cb-speech/{bank}")
async def get_cb_speech_summary(bank: str = "FED"):
    """Central bank speech tone analysis — hawkish/dovish signal."""
    from sentinel.sma.cb_speech import get_cb_summary
    result = await get_cb_summary(bank=bank.upper())
    return result.model_dump()


# ── Semantic News Search ──────────────────────────────────────────────────────

@router.get("/search")
async def semantic_news_search(
    q: str = Query(..., description="Natural language query (e.g. 'Fed rate hike inflation')"),
    ticker: Optional[str] = Query(None, description="Optional ticker filter"),
    limit: int = Query(10, ge=1, le=50),
    threshold: float = Query(0.3, ge=0.0, le=1.0, description="Min cosine similarity"),
    session: AsyncSession = Depends(get_session),
):
    """Semantic similarity search over news articles using pgvector embeddings.

    Returns articles ranked by cosine similarity to the query. Requires
    sentence-transformers and populated embeddings (run: make backfill-embeddings).
    """
    from sentinel.sil.news_embeddings import semantic_search

    results = await semantic_search(
        query=q,
        session=session,
        ticker=ticker.upper() if ticker else None,
        limit=limit,
        similarity_threshold=threshold,
    )

    return {
        "query": q,
        "ticker_filter": ticker,
        "threshold": threshold,
        "count": len(results),
        "results": results,
    }

# ─── Document RAG Search ──────────────────────────────────────────────────────
@router.get("/rag")
async def rag_search(
    q: str = Query(..., description="Natural language query"),
    ticker: str | None = Query(None),
    doc_type: str | None = Query(None),
    synthesize: bool = Query(False),
):
    """Semantic search over financial documents with optional Claude synthesis."""
    from sentinel.sil.rag import query as rag_query
    from sentinel.core.config import get_settings
    settings = get_settings()
    result = await rag_query(
        db_url=settings.database_url,
        query_text=q,
        ticker=ticker,
        doc_type=doc_type,
        synthesize=synthesize,
        anthropic_api_key=settings.anthropic_api_key,
    )
    return result.model_dump()
