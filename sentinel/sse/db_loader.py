"""Screener DB loader — populates DuckDB screener engine from PostgreSQL."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


async def populate_screener_from_db(session: AsyncSession) -> int:
    """Pull PostgreSQL instruments + OHLCV + XBRL facts + insider data into the
    DuckDB screener engine.  Returns number of tickers loaded."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logger.error("pandas/numpy not available — screener population skipped")
        return 0

    # ── 1. Instrument universe ────────────────────────────────────────────────
    inst_result = await session.execute(text("""
        SELECT figi, ticker, name, sector, industry, exchange
        FROM instruments
        WHERE asset_class = 'equity'
        ORDER BY ticker
    """))
    instruments = [dict(r) for r in inst_result.mappings()]
    if not instruments:
        logger.warning("No instruments in DB — screener will be empty")
        return 0

    # ── 2. Price + 52-week stats + SMAs from OHLCV ───────────────────────────
    cutoff_252 = datetime.utcnow() - timedelta(days=380)   # ~252 trading days
    price_result = await session.execute(text("""
        WITH ranked AS (
            SELECT
                i.ticker,
                i.figi,
                o.close,
                o.volume,
                o.time,
                ROW_NUMBER() OVER (PARTITION BY i.ticker ORDER BY o.time DESC) AS rn,
                MAX(o.close)  OVER (PARTITION BY i.ticker) AS high_52w,
                MIN(o.close)  OVER (PARTITION BY i.ticker) AS low_52w,
                AVG(o.close)  OVER (
                    PARTITION BY i.ticker ORDER BY o.time
                    ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                ) AS sma_50,
                AVG(o.close)  OVER (
                    PARTITION BY i.ticker ORDER BY o.time
                    ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
                ) AS sma_200,
                AVG(o.volume) OVER (
                    PARTITION BY i.ticker ORDER BY o.time
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS avg_vol_20
            FROM ohlcv o
            JOIN instruments i ON o.figi = i.figi
            WHERE o.interval = '1d'
              AND o.time >= :cutoff
        )
        SELECT ticker, figi,
               close AS price, high_52w, low_52w,
               sma_50, sma_200, avg_vol_20
        FROM ranked
        WHERE rn = 1
    """), {"cutoff": cutoff_252})
    price_rows = [dict(r) for r in price_result.mappings()]
    price_df = pd.DataFrame(price_rows) if price_rows else pd.DataFrame()

    # ── 3. RSI-14: fetch last 30 closes per ticker, compute Wilder's in Python ─
    rsi_result = await session.execute(text("""
        WITH ranked AS (
            SELECT
                i.ticker,
                o.close,
                ROW_NUMBER() OVER (PARTITION BY i.ticker ORDER BY o.time DESC) AS rn
            FROM ohlcv o
            JOIN instruments i ON o.figi = i.figi
            WHERE o.interval = '1d' AND o.time >= :cutoff
        )
        SELECT ticker, close FROM ranked WHERE rn <= 30 ORDER BY ticker, rn DESC
    """), {"cutoff": cutoff_252})
    rsi_rows = [dict(r) for r in rsi_result.mappings()]

    rsi_map: dict[str, float] = {}
    if rsi_rows:
        rsi_df = pd.DataFrame(rsi_rows)
        for ticker, grp in rsi_df.groupby("ticker"):
            closes = grp["close"].values.astype(float)
            if len(closes) >= 15:
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0.0)
                losses = np.where(deltas < 0, -deltas, 0.0)
                avg_gain = float(gains[-14:].mean())
                avg_loss = float(losses[-14:].mean())
                rsi_map[str(ticker)] = (
                    100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
                    if avg_loss > 0 else 100.0
                )

    # ── 4. Financial facts — most recent annual value per concept per FIGI ───
    CONCEPTS = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "NetIncomeLoss",
        "GrossProfit",
        "OperatingIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "LongTermDebt",
        "CashAndCashEquivalentsAtCarryingValue",
        "EarningsPerShareBasic",
        "CommonStockSharesOutstanding",
        "InterestExpense",
    ]
    placeholders = ", ".join(f"'{c}'" for c in CONCEPTS)
    facts_result = await session.execute(text(f"""
        WITH ranked AS (
            SELECT
                figi, concept, value,
                ROW_NUMBER() OVER (
                    PARTITION BY figi, concept
                    ORDER BY period_end DESC, filed DESC
                ) AS rn
            FROM financial_facts
            WHERE concept IN ({placeholders})
              AND unit IN ('USD', 'shares', 'USD/shares')
              AND figi IS NOT NULL AND figi != ''
        )
        SELECT figi, concept, value FROM ranked WHERE rn = 1
    """))
    facts_map: dict[str, dict[str, float]] = {}
    for row in facts_result.mappings():
        figi = row["figi"]
        if figi not in facts_map:
            facts_map[figi] = {}
        if row["value"] is not None:
            facts_map[figi][row["concept"]] = float(row["value"])

    # ── 5. Insider transaction counts (last 30 days) ─────────────────────────
    insider_cutoff = datetime.utcnow() - timedelta(days=30)
    insider_result = await session.execute(text("""
        SELECT
            ticker,
            COUNT(*) FILTER (WHERE tx_type = 'BUY')  AS buys,
            COUNT(*) FILTER (WHERE tx_type = 'SELL') AS sells
        FROM insider_transactions
        WHERE tx_date >= :cutoff
        GROUP BY ticker
    """), {"cutoff": insider_cutoff})
    insider_map: dict[str, dict] = {
        row["ticker"]: {"buys": int(row["buys"]), "sells": int(row["sells"])}
        for row in insider_result.mappings()
    }

    # ── 6. Institutional ownership pct (latest within 90 days) ───────────────
    inst_pct_result = await session.execute(text("""
        SELECT ticker, MAX(pct_outstanding) AS inst_pct
        FROM institutional_holdings
        WHERE as_of >= NOW() - INTERVAL '90 days'
        GROUP BY ticker
    """))
    inst_pct_map: dict[str, float] = {
        row["ticker"]: float(row["inst_pct"])
        for row in inst_pct_result.mappings()
        if row["inst_pct"] is not None
    }

    # ── 7. Assemble screener rows ─────────────────────────────────────────────
    price_by_ticker: dict[str, dict] = {}
    if not price_df.empty:
        for _, row in price_df.iterrows():
            price_by_ticker[row["ticker"]] = row.to_dict()

    rows = []
    for inst in instruments:
        ticker = inst["ticker"]
        figi = inst["figi"]
        pr = price_by_ticker.get(ticker)
        facts = facts_map.get(figi, {})
        insider = insider_map.get(ticker, {"buys": 0, "sells": 0})

        def _fv(key: str) -> Optional[float]:
            v = pr.get(key) if pr else None
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

        price_val = _fv("price")
        high_52w  = _fv("high_52w")

        revenue   = _first(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
        net_inc   = facts.get("NetIncomeLoss")
        gross_p   = facts.get("GrossProfit")
        op_inc    = facts.get("OperatingIncomeLoss")
        assets    = facts.get("Assets")
        equity    = facts.get("StockholdersEquity")
        ltd       = facts.get("LongTermDebt")
        interest  = facts.get("InterestExpense")
        shares    = facts.get("CommonStockSharesOutstanding")
        eps       = facts.get("EarningsPerShareBasic")

        mkt_cap   = (price_val * shares)            if (price_val and shares)                      else None
        pe        = (price_val / eps)               if (price_val and eps and eps > 0)              else None
        p2b       = (mkt_cap / equity)              if (mkt_cap and equity and equity > 0)          else None
        p2s       = (mkt_cap / revenue)             if (mkt_cap and revenue and revenue > 0)        else None
        gm        = (gross_p / revenue)             if (gross_p is not None and revenue and revenue > 0) else None
        om        = (op_inc / revenue)              if (op_inc is not None and revenue and revenue > 0)  else None
        nm        = (net_inc / revenue)             if (net_inc is not None and revenue and revenue > 0) else None
        roe_val   = (net_inc / equity)              if (net_inc is not None and equity and equity > 0)   else None
        roa_val   = (net_inc / assets)              if (net_inc is not None and assets and assets > 0)   else None
        d2e       = (ltd / equity)                  if (ltd and equity and equity > 0)              else None
        icr       = (op_inc / interest)             if (op_inc is not None and interest and interest > 0) else None

        rows.append({
            "ticker": ticker,
            "figi": figi,
            "name": inst.get("name", ""),
            "sector": inst.get("sector"),
            "industry": inst.get("industry"),
            "exchange": inst.get("exchange", ""),
            "market_cap": mkt_cap,
            "pe_ratio": pe,
            "forward_pe": None,
            "peg_ratio": None,
            "price_to_book": p2b,
            "price_to_sales": p2s,
            "ev_to_ebitda": None,
            "ev_to_revenue": None,
            "revenue_growth_yoy": None,
            "earnings_growth_yoy": None,
            "revenue_growth_3y": None,
            "gross_margin": gm,
            "operating_margin": om,
            "net_margin": nm,
            "roe": roe_val,
            "roa": roa_val,
            "roic": None,
            "debt_to_equity": d2e,
            "current_ratio": None,
            "quick_ratio": None,
            "interest_coverage": icr,
            "dividend_yield": None,
            "payout_ratio": None,
            "price": price_val,
            "price_52w_high": high_52w,
            "price_52w_low": _fv("low_52w"),
            "price_vs_52w_high": (price_val / high_52w) if (price_val and high_52w) else None,
            "sma_50": _fv("sma_50"),
            "sma_200": _fv("sma_200"),
            "rsi_14": rsi_map.get(ticker),
            "avg_volume_20d": _fv("avg_vol_20"),
            "insider_buy_30d": insider.get("buys", 0),
            "insider_sell_30d": insider.get("sells", 0),
            "institutional_pct": inst_pct_map.get(ticker),
            "short_interest_pct": None,
        })

    if not rows:
        logger.warning("populate_screener_from_db: no rows assembled")
        return 0

    df = pd.DataFrame(rows)

    from sentinel.sse.screener import ScreenerEngine
    import sentinel.api.routes.screen as _screen_route

    engine = ScreenerEngine()
    engine.load_from_dataframe(df)
    _screen_route._engine = engine          # hot-swap the global singleton

    logger.info("Screener populated from DB", tickers=len(rows))
    return len(rows)


def _first(d: dict, keys: list[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return float(v)
    return None
