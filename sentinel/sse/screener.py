"""SSE screener — DuckDB-backed multi-criteria screener with fundamental + technical criteria."""
from __future__ import annotations
import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional, Any
try:
    import duckdb
    import pandas as pd
    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    _DUCKDB_AVAILABLE = False
from sentinel.core.types import ScreenResult
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Universe of S&P 500 + Russell 2000 as starting point
# Full universe built by SIM → instrument_master at startup
DEFAULT_UNIVERSE_SQL = """
    SELECT ticker, figi, name, sector, industry, exchange
    FROM instruments
    WHERE asset_class = 'equity' AND exchange IN ('US', 'NYSE', 'NASDAQ')
"""


class ScreenerEngine:
    """
    Columnar screener backed by DuckDB for fast in-process SQL execution.
    Fundamental data loaded from EDGAR, technical from price history.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = duckdb.connect(db_path)
        self._initialized = False

    def initialize_tables(self) -> None:
        """Create in-memory screening tables."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS screen_universe (
                ticker VARCHAR PRIMARY KEY,
                figi VARCHAR,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                exchange VARCHAR,
                -- Fundamental metrics
                market_cap DOUBLE,
                pe_ratio DOUBLE,
                forward_pe DOUBLE,
                peg_ratio DOUBLE,
                price_to_book DOUBLE,
                price_to_sales DOUBLE,
                ev_to_ebitda DOUBLE,
                ev_to_revenue DOUBLE,
                -- Growth
                revenue_growth_yoy DOUBLE,
                earnings_growth_yoy DOUBLE,
                revenue_growth_3y DOUBLE,
                -- Profitability
                gross_margin DOUBLE,
                operating_margin DOUBLE,
                net_margin DOUBLE,
                roe DOUBLE,
                roa DOUBLE,
                roic DOUBLE,
                -- Debt
                debt_to_equity DOUBLE,
                current_ratio DOUBLE,
                quick_ratio DOUBLE,
                interest_coverage DOUBLE,
                -- Dividends
                dividend_yield DOUBLE,
                payout_ratio DOUBLE,
                -- Technical
                price DOUBLE,
                price_52w_high DOUBLE,
                price_52w_low DOUBLE,
                price_vs_52w_high DOUBLE,
                sma_50 DOUBLE,
                sma_200 DOUBLE,
                rsi_14 DOUBLE,
                avg_volume_20d DOUBLE,
                -- Ownership signals
                insider_buy_30d INTEGER DEFAULT 0,
                insider_sell_30d INTEGER DEFAULT 0,
                institutional_pct DOUBLE,
                short_interest_pct DOUBLE,
                -- Timestamp
                updated_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        self._initialized = True
        logger.info("Screener tables initialized")

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Bulk-load screening universe from a pandas DataFrame."""
        if not self._initialized:
            self.initialize_tables()
        self._conn.execute("DELETE FROM screen_universe")
        self._conn.execute("INSERT INTO screen_universe SELECT * FROM df")
        count = self._conn.execute("SELECT COUNT(*) FROM screen_universe").fetchone()[0]
        logger.info("Screener universe loaded", rows=count)

    def screen(self, criteria: dict) -> list[ScreenResult]:
        """
        Run a screening pass with the given criteria dict.

        Criteria keys (all optional):
          pe_ratio_lt, pe_ratio_gt, market_cap_min, market_cap_max,
          dividend_yield_gt, revenue_growth_gt, net_margin_gt,
          roe_gt, debt_to_equity_lt, rsi_lt, rsi_gt,
          price_vs_52w_high_gt, insider_buying, sectors, exchanges
        """
        if not self._initialized:
            self.initialize_tables()

        where_clauses = ["1=1"]
        params = []

        # Fundamental criteria
        if "pe_ratio_lt" in criteria and criteria["pe_ratio_lt"] is not None:
            where_clauses.append("pe_ratio < ? AND pe_ratio > 0")
            params.append(float(criteria["pe_ratio_lt"]))

        if "pe_ratio_gt" in criteria:
            where_clauses.append("pe_ratio > ?")
            params.append(float(criteria["pe_ratio_gt"]))

        if "market_cap_min" in criteria:
            where_clauses.append("market_cap >= ?")
            params.append(float(criteria["market_cap_min"]))

        if "market_cap_max" in criteria:
            where_clauses.append("market_cap <= ?")
            params.append(float(criteria["market_cap_max"]))

        if "dividend_yield_gt" in criteria:
            where_clauses.append("dividend_yield > ?")
            params.append(float(criteria["dividend_yield_gt"]))

        if criteria.get("has_dividend"):
            where_clauses.append("dividend_yield > 0")

        if "revenue_growth_gt" in criteria:
            where_clauses.append("revenue_growth_yoy > ?")
            params.append(float(criteria["revenue_growth_gt"]))

        if "net_margin_gt" in criteria:
            where_clauses.append("net_margin > ?")
            params.append(float(criteria["net_margin_gt"]))

        if "roe_gt" in criteria:
            where_clauses.append("roe > ?")
            params.append(float(criteria["roe_gt"]))

        if "debt_to_equity_lt" in criteria:
            where_clauses.append("debt_to_equity < ? AND debt_to_equity >= 0")
            params.append(float(criteria["debt_to_equity_lt"]))

        if "gross_margin_gt" in criteria:
            where_clauses.append("gross_margin > ?")
            params.append(float(criteria["gross_margin_gt"]))

        if "ev_to_ebitda_lt" in criteria:
            where_clauses.append("ev_to_ebitda < ? AND ev_to_ebitda > 0")
            params.append(float(criteria["ev_to_ebitda_lt"]))

        # Technical criteria
        if "rsi_lt" in criteria:
            where_clauses.append("rsi_14 < ?")
            params.append(float(criteria["rsi_lt"]))

        if "rsi_gt" in criteria:
            where_clauses.append("rsi_14 > ?")
            params.append(float(criteria["rsi_gt"]))

        if "price_vs_52w_high_gt" in criteria:
            where_clauses.append("price_vs_52w_high > ?")
            params.append(float(criteria["price_vs_52w_high_gt"]))

        if "above_sma200" in criteria and criteria["above_sma200"]:
            where_clauses.append("price > sma_200 AND sma_200 > 0")

        if "above_sma50" in criteria and criteria["above_sma50"]:
            where_clauses.append("price > sma_50 AND sma_50 > 0")

        # Ownership signals
        if criteria.get("insider_buying_30d"):
            where_clauses.append("insider_buy_30d > 0")

        if "short_interest_lt" in criteria:
            where_clauses.append("short_interest_pct < ?")
            params.append(float(criteria["short_interest_lt"]))

        # Sector filter
        if "sectors" in criteria and criteria["sectors"]:
            placeholders = ",".join("?" for _ in criteria["sectors"])
            where_clauses.append(f"sector IN ({placeholders})")
            params.extend(criteria["sectors"])

        limit = int(criteria.get("limit", 50))
        sort_col = criteria.get("sort_by", "market_cap")
        sort_dir = "DESC" if criteria.get("sort_desc", True) else "ASC"

        sql = f"""
            SELECT ticker, figi, name, sector, industry,
                   market_cap, pe_ratio, dividend_yield, revenue_growth_yoy,
                   net_margin, roe, debt_to_equity, rsi_14, price,
                   price_vs_52w_high, insider_buy_30d, short_interest_pct
            FROM screen_universe
            WHERE {' AND '.join(where_clauses)}
            ORDER BY {sort_col} {sort_dir}
            LIMIT {limit}
        """

        try:
            rows = self._conn.execute(sql, params).fetchdf()
        except Exception as exc:
            logger.error("Screener SQL error", error=str(exc))
            return []

        results = []
        for _, row in rows.iterrows():
            results.append(ScreenResult(
                ticker=row["ticker"],
                figi=row.get("figi", ""),
                name=row.get("name", ""),
                sector=row.get("sector"),
                industry=row.get("industry"),
                market_cap=Decimal(str(row["market_cap"])) if pd.notna(row.get("market_cap")) else None,
                pe_ratio=Decimal(str(row["pe_ratio"])) if pd.notna(row.get("pe_ratio")) else None,
                dividend_yield=Decimal(str(row["dividend_yield"])) if pd.notna(row.get("dividend_yield")) else None,
                revenue_growth_yoy=Decimal(str(row["revenue_growth_yoy"])) if pd.notna(row.get("revenue_growth_yoy")) else None,
                net_margin=Decimal(str(row["net_margin"])) if pd.notna(row.get("net_margin")) else None,
                roe=Decimal(str(row["roe"])) if pd.notna(row.get("roe")) else None,
                screened_at=datetime.utcnow(),
            ))
        logger.info("Screen complete", results=len(results), criteria=criteria)
        return results

    def get_universe_count(self) -> int:
        if not self._initialized:
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM screen_universe").fetchone()[0]

    def get_sector_breakdown(self) -> list[dict]:
        if not self._initialized:
            return []
        rows = self._conn.execute("""
            SELECT sector, COUNT(*) as count, AVG(market_cap) as avg_market_cap
            FROM screen_universe
            WHERE sector IS NOT NULL
            GROUP BY sector
            ORDER BY count DESC
        """).fetchdf()
        return rows.to_dict("records")
