"""Point-in-time financial data engine — EDGAR XBRL as authoritative source.

All data selection is gated by filed_date <= as_of_date, which means every
query returns ONLY information that was publicly available on the requested
date.  This prevents look-ahead bias in backtesting and satisfies dim_020 /
dim_022.

Key design decisions
--------------------
* Single company-facts JSON (~5-15 MB) is fetched once per CIK and cached in
  memory for the lifetime of the engine instance.  Subsequent calls for the
  same ticker — including full backtest panels with hundreds of dates — pay
  only the parsing cost.
* TTM is computed from the *earliest* filed version of each quarterly report
  (as-originally-reported), not the latest revision, preserving the original
  information set.
* Balance-sheet items (assets, debt, cash, equity) use the latest *filed*
  value for a period rather than the sum of four quarters.
* Flow items (revenue, net income, CFO, capex) are summed across the four
  most recent non-overlapping quarters to produce TTM.
"""
from __future__ import annotations

import asyncio
import re
from datetime import date, timedelta
from typing import Literal, Optional

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_SLEEP = 0.12  # 8 req/sec → well within the 10 req/sec EDGAR cap

# Forms that represent annual filings
_ANNUAL_FORMS = {"10-K", "10-KT", "20-F", "40-F"}
# Forms that represent quarterly filings
_QUARTERLY_FORMS = {"10-Q", "10-QT"}
# Balance-sheet concepts: take latest value, do NOT sum for TTM
_BALANCE_SHEET_CONCEPTS = {
    "total_assets",
    "total_liabilities",
    "equity",
    "long_term_debt",
    "short_term_debt",
    "cash",
    "shares_diluted",
    "inventory",
    "receivables",
}
# Approximate durations (in days) used to classify period types
_ANNUAL_MIN_DAYS = 340
_QUARTERLY_MIN_DAYS = 70
_QUARTERLY_MAX_DAYS = 110

# Standard financial concepts in preference order (fallback chain)
CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomer",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "NetIncome",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "gross_profit": ["GrossProfit"],
    "ebit": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short_term_debt": ["ShortTermBorrowings", "NotesPayableCurrent", "DebtCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalents",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "shares_diluted": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "rd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "inventory": ["InventoryNet", "Inventories"],
    "receivables": ["AccountsReceivableNetCurrent", "AccountsReceivableNet"],
    "dividends_per_share": ["CommonStockDividendsPerShareCashPaid"],
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FinancialFact(BaseModel):
    """A single XBRL data point with full PIT metadata."""

    concept: str  # canonical name e.g. "revenue"
    xbrl_concept: str  # e.g. "RevenueFromContractWithCustomerExcludingAssessedTax"
    value: float
    unit: str  # "USD", "shares", "USD/shares"
    period_start: Optional[date] = None
    period_end: date
    filed_date: date  # THE key PIT field — when SEC received the filing
    form: str  # "10-K", "10-Q", "8-K", …
    accession: str
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None  # "Q1" | "Q2" | "Q3" | "FY"
    is_annual: bool = False
    is_ttm: bool = False  # trailing twelve months (computed)


class PointInTimeFinancials(BaseModel):
    """Full PIT financial snapshot for a ticker on a given as_of_date."""

    ticker: str
    cik: str
    company_name: str
    as_of_date: date  # the date we simulate knowledge "as of"
    fiscal_period_end: date  # period_end of the most recent available filing
    filed_date: date  # when EDGAR received that filing

    # ── Income Statement ─────────────────────────────────────────────────────
    revenue: Optional[float] = None  # most recent annual (10-K) or quarterly
    revenue_ttm: Optional[float] = None
    net_income: Optional[float] = None
    net_income_ttm: Optional[float] = None
    operating_income: Optional[float] = None
    gross_profit: Optional[float] = None
    ebitda: Optional[float] = None  # operating_income + D&A
    ebitda_ttm: Optional[float] = None
    eps_diluted: Optional[float] = None
    rd: Optional[float] = None
    sga: Optional[float] = None

    # ── Balance Sheet ────────────────────────────────────────────────────────
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    short_term_debt: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    net_debt: Optional[float] = None
    equity: Optional[float] = None
    inventory: Optional[float] = None
    receivables: Optional[float] = None

    # ── Cash Flow ────────────────────────────────────────────────────────────
    capex: Optional[float] = None
    capex_ttm: Optional[float] = None
    cfo: Optional[float] = None
    cfo_ttm: Optional[float] = None
    free_cash_flow: Optional[float] = None  # cfo_ttm - capex_ttm
    da: Optional[float] = None

    # ── Per Share ────────────────────────────────────────────────────────────
    shares_diluted: Optional[float] = None
    dividends_per_share: Optional[float] = None

    # ── Margins / Growth ─────────────────────────────────────────────────────
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None

    # ── Valuation (requires price input) ────────────────────────────────────
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_to_fcf: Optional[float] = None

    # ── Data quality ─────────────────────────────────────────────────────────
    filing_lag_days: int = 0  # days from period_end to filed_date
    n_concepts_found: int = 0


class FinancialTimeSeries(BaseModel):
    """Complete PIT time series for a single concept."""

    ticker: str
    cik: str
    concept: str  # e.g. "revenue"
    annual_series: list[FinancialFact]  # 10-K only, sorted by period_end asc
    quarterly_series: list[FinancialFact]  # 10-Q only, sorted by period_end asc
    ttm_series: list[FinancialFact]  # computed TTM at each quarter end


class BacktestPITData(BaseModel):
    """PIT snapshots for a single ticker across multiple backtest dates."""

    ticker: str
    dates: list[date]
    financials: list[PointInTimeFinancials]  # parallel to dates


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PointInTimeEngine:
    """Async engine that pulls EDGAR XBRL facts and applies PIT filters.

    Usage::

        engine = PointInTimeEngine()
        pit = await engine.get_as_of("AAPL", date(2020, 6, 30))
    """

    def __init__(
        self,
        timeout: float = 30.0,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._timeout = timeout
        self._cache_dir = cache_dir
        # In-memory caches — facts JSON is ~5-15 MB each; keep per session
        self._cik_cache: dict[str, tuple[str, str]] = {}  # ticker → (cik, company_name)
        self._facts_cache: dict[str, dict] = {}  # cik → raw companyfacts JSON

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def get_as_of(
        self,
        ticker: str,
        as_of_date: date,
        price: Optional[float] = None,
    ) -> PointInTimeFinancials:
        """Return PIT financial snapshot for *ticker* as of *as_of_date*.

        Only data with filed_date <= as_of_date is considered.  TTM figures
        aggregate the four most-recent non-overlapping quarterly reports that
        were available on that date.

        Parameters
        ----------
        ticker:
            Upper-case US exchange ticker symbol.
        as_of_date:
            Simulate knowledge available on this date (no look-ahead).
        price:
            If provided, compute valuation multiples (P/E, P/B, EV/EBITDA).
        """
        cik, company_name = await self._get_cik(ticker)
        facts = await self._load_company_facts(cik)

        n_found = 0

        # ── Revenue ─────────────────────────────────────────────────────
        rev_fact = self._extract_concept_pit(facts, "revenue", as_of_date, "any")
        revenue = rev_fact.value if rev_fact else None
        if revenue is not None:
            n_found += 1
        revenue_ttm = self._compute_ttm_value(facts, "revenue", as_of_date)

        # ── Net Income ───────────────────────────────────────────────────
        ni_fact = self._extract_concept_pit(facts, "net_income", as_of_date, "any")
        net_income = ni_fact.value if ni_fact else None
        if net_income is not None:
            n_found += 1
        net_income_ttm = self._compute_ttm_value(facts, "net_income", as_of_date)

        # ── Operating Income ─────────────────────────────────────────────
        oi_fact = self._extract_concept_pit(facts, "operating_income", as_of_date, "any")
        operating_income = oi_fact.value if oi_fact else None
        if operating_income is not None:
            n_found += 1

        # ── Gross Profit ─────────────────────────────────────────────────
        gp_fact = self._extract_concept_pit(facts, "gross_profit", as_of_date, "any")
        gross_profit = gp_fact.value if gp_fact else None
        if gross_profit is not None:
            n_found += 1

        # ── D&A ──────────────────────────────────────────────────────────
        da_fact = self._extract_concept_pit(facts, "da", as_of_date, "any")
        da = da_fact.value if da_fact else None

        # EBITDA = operating income + D&A
        ebitda: Optional[float] = None
        if operating_income is not None and da is not None:
            ebitda = operating_income + da

        ebitda_ttm: Optional[float] = None
        oi_ttm = self._compute_ttm_value(facts, "operating_income", as_of_date)
        da_ttm = self._compute_ttm_value(facts, "da", as_of_date)
        if oi_ttm is not None and da_ttm is not None:
            ebitda_ttm = oi_ttm + da_ttm

        # ── EPS ──────────────────────────────────────────────────────────
        eps_fact = self._extract_concept_pit(facts, "eps_diluted", as_of_date, "any")
        eps_diluted = eps_fact.value if eps_fact else None
        if eps_diluted is not None:
            n_found += 1

        # ── R&D / SG&A ───────────────────────────────────────────────────
        rd_fact = self._extract_concept_pit(facts, "rd", as_of_date, "any")
        rd = rd_fact.value if rd_fact else None

        sga_fact = self._extract_concept_pit(facts, "sga", as_of_date, "any")
        sga = sga_fact.value if sga_fact else None

        # ── Balance Sheet ────────────────────────────────────────────────
        assets_fact = self._extract_concept_pit(facts, "total_assets", as_of_date, "any")
        total_assets = assets_fact.value if assets_fact else None
        if total_assets is not None:
            n_found += 1

        liab_fact = self._extract_concept_pit(facts, "total_liabilities", as_of_date, "any")
        total_liabilities = liab_fact.value if liab_fact else None

        eq_fact = self._extract_concept_pit(facts, "equity", as_of_date, "any")
        equity = eq_fact.value if eq_fact else None
        if equity is not None:
            n_found += 1

        ltd_fact = self._extract_concept_pit(facts, "long_term_debt", as_of_date, "any")
        long_term_debt = ltd_fact.value if ltd_fact else None

        std_fact = self._extract_concept_pit(facts, "short_term_debt", as_of_date, "any")
        short_term_debt = std_fact.value if std_fact else None

        cash_fact = self._extract_concept_pit(facts, "cash", as_of_date, "any")
        cash = cash_fact.value if cash_fact else None
        if cash is not None:
            n_found += 1

        inv_fact = self._extract_concept_pit(facts, "inventory", as_of_date, "any")
        inventory = inv_fact.value if inv_fact else None

        rec_fact = self._extract_concept_pit(facts, "receivables", as_of_date, "any")
        receivables = rec_fact.value if rec_fact else None

        # Derived debt / net-debt
        total_debt: Optional[float] = None
        if long_term_debt is not None or short_term_debt is not None:
            total_debt = (long_term_debt or 0.0) + (short_term_debt or 0.0)

        net_debt: Optional[float] = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash

        # ── Cash Flow ────────────────────────────────────────────────────
        capex_fact = self._extract_concept_pit(facts, "capex", as_of_date, "any")
        capex = capex_fact.value if capex_fact else None
        if capex is not None:
            # EDGAR stores capex as a negative outflow in some periods
            capex = abs(capex)

        capex_ttm_raw = self._compute_ttm_value(facts, "capex", as_of_date)
        capex_ttm = abs(capex_ttm_raw) if capex_ttm_raw is not None else None

        cfo_fact = self._extract_concept_pit(facts, "cfo", as_of_date, "any")
        cfo = cfo_fact.value if cfo_fact else None
        if cfo is not None:
            n_found += 1

        cfo_ttm = self._compute_ttm_value(facts, "cfo", as_of_date)

        # FCF = CFO (TTM) - capex (TTM)
        free_cash_flow: Optional[float] = None
        if cfo_ttm is not None and capex_ttm is not None:
            free_cash_flow = cfo_ttm - capex_ttm

        # ── Shares ───────────────────────────────────────────────────────
        shares_fact = self._extract_concept_pit(facts, "shares_diluted", as_of_date, "any")
        shares_diluted = shares_fact.value if shares_fact else None

        # ── Dividends per share ──────────────────────────────────────────
        div_fact = self._extract_concept_pit(facts, "dividends_per_share", as_of_date, "any")
        dividends_per_share = div_fact.value if div_fact else None

        # ── Margins ──────────────────────────────────────────────────────
        gross_margin: Optional[float] = None
        operating_margin: Optional[float] = None
        net_margin: Optional[float] = None

        rev_base = revenue_ttm or revenue
        if rev_base and rev_base != 0.0:
            gp_base = self._compute_ttm_value(facts, "gross_profit", as_of_date) or gross_profit
            if gp_base is not None:
                gross_margin = gp_base / rev_base
            oi_base = oi_ttm or operating_income
            if oi_base is not None:
                operating_margin = oi_base / rev_base
            ni_base = net_income_ttm or net_income
            if ni_base is not None:
                net_margin = ni_base / rev_base

        # ── YoY Revenue Growth ───────────────────────────────────────────
        revenue_growth_yoy: Optional[float] = None
        prior_year_date = date(as_of_date.year - 1, as_of_date.month, as_of_date.day)
        rev_prior = self._compute_ttm_value(facts, "revenue", prior_year_date)
        if rev_prior is None:
            rev_prior_fact = self._extract_concept_pit(
                facts, "revenue", prior_year_date, "any"
            )
            rev_prior = rev_prior_fact.value if rev_prior_fact else None
        rev_current = revenue_ttm or revenue
        if rev_current is not None and rev_prior and rev_prior != 0.0:
            revenue_growth_yoy = (rev_current - rev_prior) / abs(rev_prior)

        # ── PIT anchor — most recent filing available ────────────────────
        # Pick the fact with the latest filed_date across all concepts
        anchor_fact = self._most_recent_fact(facts, as_of_date)
        fiscal_period_end = anchor_fact.period_end if anchor_fact else as_of_date
        filed_date_anchor = anchor_fact.filed_date if anchor_fact else as_of_date
        filing_lag = (filed_date_anchor - fiscal_period_end).days if anchor_fact else 0

        # ── Valuation ratios (price required) ────────────────────────────
        pe_ratio: Optional[float] = None
        pb_ratio: Optional[float] = None
        ps_ratio: Optional[float] = None
        ev_ebitda: Optional[float] = None
        price_to_fcf: Optional[float] = None

        if price is not None and price > 0.0 and shares_diluted and shares_diluted > 0.0:
            market_cap = price * shares_diluted

            eps_for_pe = eps_diluted
            if eps_for_pe is None and net_income_ttm is not None and shares_diluted > 0.0:
                eps_for_pe = net_income_ttm / shares_diluted
            if eps_for_pe and eps_for_pe > 0.0:
                pe_ratio = price / eps_for_pe

            if equity is not None and equity > 0.0:
                book_per_share = equity / shares_diluted
                if book_per_share > 0.0:
                    pb_ratio = price / book_per_share

            rev_for_ps = revenue_ttm or revenue
            if rev_for_ps and rev_for_ps > 0.0:
                ps_ratio = market_cap / rev_for_ps

            if ebitda_ttm and ebitda_ttm > 0.0 and net_debt is not None:
                enterprise_value = market_cap + net_debt
                ev_ebitda = enterprise_value / ebitda_ttm

            if free_cash_flow and free_cash_flow > 0.0:
                price_to_fcf = market_cap / free_cash_flow

        return PointInTimeFinancials(
            ticker=ticker.upper(),
            cik=cik,
            company_name=company_name,
            as_of_date=as_of_date,
            fiscal_period_end=fiscal_period_end,
            filed_date=filed_date_anchor,
            revenue=revenue,
            revenue_ttm=revenue_ttm,
            net_income=net_income,
            net_income_ttm=net_income_ttm,
            operating_income=operating_income,
            gross_profit=gross_profit,
            ebitda=ebitda,
            ebitda_ttm=ebitda_ttm,
            eps_diluted=eps_diluted,
            rd=rd,
            sga=sga,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            long_term_debt=long_term_debt,
            short_term_debt=short_term_debt,
            total_debt=total_debt,
            cash=cash,
            net_debt=net_debt,
            equity=equity,
            inventory=inventory,
            receivables=receivables,
            capex=capex,
            capex_ttm=capex_ttm,
            cfo=cfo,
            cfo_ttm=cfo_ttm,
            free_cash_flow=free_cash_flow,
            da=da,
            shares_diluted=shares_diluted,
            dividends_per_share=dividends_per_share,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            revenue_growth_yoy=revenue_growth_yoy,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            ps_ratio=ps_ratio,
            ev_ebitda=ev_ebitda,
            price_to_fcf=price_to_fcf,
            filing_lag_days=max(0, filing_lag),
            n_concepts_found=n_found,
        )

    async def get_time_series(
        self, ticker: str, concept: str, start_year: int = 2005
    ) -> FinancialTimeSeries:
        """Return complete PIT history for a single financial concept.

        Results are sorted by *filed_date* (chronological PIT order), not by
        period_end, so backtests can iterate the list without skipping ahead.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.
        concept:
            Canonical concept name from CONCEPT_MAP (e.g. "revenue", "cfo").
        start_year:
            Exclude facts whose period_end falls before this calendar year.
        """
        if concept not in CONCEPT_MAP:
            raise ValueError(
                f"Unknown concept '{concept}'. Valid: {sorted(CONCEPT_MAP)}"
            )

        cik, _ = await self._get_cik(ticker)
        facts = await self._load_company_facts(cik)

        cutoff = date(start_year, 1, 1)
        annual: list[FinancialFact] = []
        quarterly: list[FinancialFact] = []

        for xbrl_concept in CONCEPT_MAP[concept]:
            raw_list = self._get_raw_observations(facts, xbrl_concept)
            for obs in raw_list:
                filed_str = obs.get("filed")
                end_str = obs.get("end") or obs.get("instant")
                if not filed_str or not end_str:
                    continue
                try:
                    filed = date.fromisoformat(filed_str)
                    period_end = date.fromisoformat(end_str)
                except ValueError:
                    continue
                if period_end < cutoff:
                    continue

                start_str = obs.get("start")
                period_start = (
                    date.fromisoformat(start_str) if start_str else None
                )
                form = obs.get("form", "")
                val = obs.get("val")
                if val is None:
                    continue

                unit = self._detect_unit(facts, xbrl_concept)
                fact = FinancialFact(
                    concept=concept,
                    xbrl_concept=xbrl_concept,
                    value=float(val),
                    unit=unit,
                    period_start=period_start,
                    period_end=period_end,
                    filed_date=filed,
                    form=form,
                    accession=obs.get("accn", ""),
                    fiscal_year=obs.get("fy"),
                    fiscal_period=obs.get("fp"),
                    is_annual=self._is_annual(obs),
                )
                if fact.is_annual:
                    annual.append(fact)
                elif self._is_quarterly(obs, period_start, period_end):
                    quarterly.append(fact)

        # Deduplicate: keep one entry per (period_end, filed_date) tuple,
        # preferring the xbrl_concept that appears first in CONCEPT_MAP
        annual = _deduplicate_facts(annual)
        quarterly = _deduplicate_facts(quarterly)

        # Sort chronologically by period_end (filed_date is secondary)
        annual.sort(key=lambda f: (f.period_end, f.filed_date))
        quarterly.sort(key=lambda f: (f.period_end, f.filed_date))

        ttm = await self.compute_ttm(quarterly)

        return FinancialTimeSeries(
            ticker=ticker.upper(),
            cik=cik,
            concept=concept,
            annual_series=annual,
            quarterly_series=quarterly,
            ttm_series=ttm,
        )

    async def get_backtest_panel(
        self,
        ticker: str,
        dates: list[date],
        include_price: bool = False,
    ) -> BacktestPITData:
        """Produce PIT snapshots for *ticker* across every date in *dates*.

        The company-facts JSON is fetched once and reused for all dates —
        this makes bulk backtest runs efficient even with hundreds of dates.

        Parameters
        ----------
        dates:
            Sorted (ascending) list of evaluation dates.
        include_price:
            Reserved for future integration with the price feed; valuation
            ratios are omitted when False.
        """
        cik, company_name = await self._get_cik(ticker)
        # Warm the cache with a single HTTP fetch
        await self._load_company_facts(cik)

        sorted_dates = sorted(dates)
        financials: list[PointInTimeFinancials] = []

        for eval_date in sorted_dates:
            try:
                pit = await self.get_as_of(ticker, eval_date)
                financials.append(pit)
            except Exception as exc:
                logger.warning(
                    "PIT snapshot failed",
                    ticker=ticker,
                    date=str(eval_date),
                    error=str(exc),
                )
                # Insert a placeholder so output stays aligned with input dates
                financials.append(
                    PointInTimeFinancials(
                        ticker=ticker.upper(),
                        cik=cik,
                        company_name=company_name,
                        as_of_date=eval_date,
                        fiscal_period_end=eval_date,
                        filed_date=eval_date,
                        n_concepts_found=0,
                    )
                )

        return BacktestPITData(
            ticker=ticker.upper(),
            dates=sorted_dates,
            financials=financials,
        )

    async def get_multi_ticker_pit(
        self,
        tickers: list[str],
        as_of_date: date,
    ) -> list[PointInTimeFinancials]:
        """Fetch PIT snapshots for many tickers on the same date concurrently.

        Uses asyncio.Semaphore(5) so we never fire more than 5 parallel
        EDGAR requests and stay well within the 10 req/sec limit.
        """
        sem = asyncio.Semaphore(5)

        async def _fetch(ticker: str) -> Optional[PointInTimeFinancials]:
            async with sem:
                await asyncio.sleep(_SLEEP)
                try:
                    return await self.get_as_of(ticker, as_of_date)
                except Exception as exc:
                    logger.warning(
                        "Multi-ticker PIT failed",
                        ticker=ticker,
                        error=str(exc),
                    )
                    return None

        results = await asyncio.gather(*[_fetch(t) for t in tickers])
        return [r for r in results if r is not None]

    async def get_revision_history(
        self, ticker: str, concept: str
    ) -> pd.DataFrame:
        """Show how a metric evolved across successive amendments.

        Returns a DataFrame with columns::

            period_end | initial_filed | initial_value | latest_filed |
            latest_value | n_revisions | revision_pct

        Rows where n_revisions == 0 were never restated.
        """
        if concept not in CONCEPT_MAP:
            raise ValueError(f"Unknown concept '{concept}'.")

        cik, _ = await self._get_cik(ticker)
        facts = await self._load_company_facts(cik)

        # Collect all raw observations for the concept across all xbrl names
        all_obs: list[dict] = []
        for xbrl_concept in CONCEPT_MAP[concept]:
            for obs in self._get_raw_observations(facts, xbrl_concept):
                if obs.get("filed") and (obs.get("end") or obs.get("instant")):
                    all_obs.append({**obs, "_xbrl": xbrl_concept})

        if not all_obs:
            return pd.DataFrame()

        rows: dict[date, list[dict]] = {}
        for obs in all_obs:
            try:
                period_end = date.fromisoformat(obs.get("end") or obs.get("instant", ""))
            except ValueError:
                continue
            rows.setdefault(period_end, []).append(obs)

        records = []
        for period_end, obs_list in sorted(rows.items()):
            obs_list_sorted = sorted(obs_list, key=lambda o: o.get("filed", ""))
            first = obs_list_sorted[0]
            last = obs_list_sorted[-1]
            initial_value = float(first.get("val", 0) or 0)
            latest_value = float(last.get("val", 0) or 0)
            n_revisions = len(obs_list_sorted) - 1
            revision_pct = (
                (latest_value - initial_value) / abs(initial_value)
                if initial_value != 0.0
                else None
            )
            records.append(
                {
                    "period_end": period_end,
                    "initial_filed": first.get("filed"),
                    "initial_value": initial_value,
                    "latest_filed": last.get("filed"),
                    "latest_value": latest_value,
                    "n_revisions": n_revisions,
                    "revision_pct": revision_pct,
                    "form": first.get("form", ""),
                }
            )

        return pd.DataFrame(records)

    async def compute_ttm(
        self, quarterly_facts: list[FinancialFact]
    ) -> list[FinancialFact]:
        """Compute TTM values at each quarterly filing date.

        For each quarterly fact *q*, find the four most-recent non-overlapping
        quarters whose *filed_date* is <= *q.filed_date*.  Sum them and return
        a synthetic FinancialFact with is_ttm=True.

        Balance-sheet concepts are excluded from summation — they return the
        latest point value instead.
        """
        if not quarterly_facts:
            return []

        sorted_qs = sorted(quarterly_facts, key=lambda f: (f.period_end, f.filed_date))
        is_balance = quarterly_facts[0].concept in _BALANCE_SHEET_CONCEPTS
        ttm_facts: list[FinancialFact] = []

        for i, anchor in enumerate(sorted_qs):
            # All quarters available on anchor.filed_date (PIT)
            available = [
                f
                for f in sorted_qs[: i + 1]
                if f.filed_date <= anchor.filed_date
            ]
            # Take the four most recent non-overlapping quarters
            selected = _select_non_overlapping_quarters(available, n=4)
            if len(selected) < 4:
                continue

            if is_balance:
                ttm_value = selected[0].value  # latest balance sheet value
            else:
                ttm_value = sum(f.value for f in selected)

            ttm_facts.append(
                FinancialFact(
                    concept=anchor.concept,
                    xbrl_concept=anchor.xbrl_concept,
                    value=ttm_value,
                    unit=anchor.unit,
                    period_start=selected[-1].period_end,
                    period_end=anchor.period_end,
                    filed_date=anchor.filed_date,
                    form=anchor.form,
                    accession=anchor.accession,
                    fiscal_year=anchor.fiscal_year,
                    fiscal_period="TTM",
                    is_annual=False,
                    is_ttm=True,
                )
            )

        return ttm_facts

    async def get_pit_ratios(
        self,
        ticker: str,
        as_of_date: date,
        price_history: Optional[pd.Series] = None,
    ) -> dict:
        """Return PIT valuation ratios for *ticker* on *as_of_date*.

        Parameters
        ----------
        price_history:
            Pandas Series indexed by date with closing prices.  The price on
            (or before) *as_of_date* is used.  When None, price-dependent
            ratios are omitted.
        """
        price: Optional[float] = None
        if price_history is not None and not price_history.empty:
            idx = price_history.index
            mask = idx <= as_of_date
            if mask.any():
                price = float(price_history[mask].iloc[-1])

        pit = await self.get_as_of(ticker, as_of_date, price=price)
        result: dict = {
            "ticker": ticker,
            "as_of_date": str(as_of_date),
            "price": price,
            "pe": pit.pe_ratio,
            "pb": pit.pb_ratio,
            "ps": pit.ps_ratio,
            "ev_ebitda": pit.ev_ebitda,
            "price_to_fcf": pit.price_to_fcf,
            "gross_margin": pit.gross_margin,
            "operating_margin": pit.operating_margin,
            "net_margin": pit.net_margin,
            "revenue_growth_yoy": pit.revenue_growth_yoy,
            "net_debt": pit.net_debt,
            "filing_lag_days": pit.filing_lag_days,
        }
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    async def _load_company_facts(self, cik: str) -> dict:
        """Fetch (and cache) the full companyfacts JSON for *cik*.

        The file is large (5-15 MB) and rarely changes intra-day, so we cache
        it for the lifetime of this engine instance.
        """
        padded = cik.zfill(10)
        if padded in self._facts_cache:
            return self._facts_cache[padded]

        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{padded}.json"
        logger.info("EDGAR companyfacts fetch", cik=padded, url=url)

        async with httpx.AsyncClient(
            headers=_HEADERS,
            timeout=self._timeout,
            follow_redirects=True,
        ) as client:
            await asyncio.sleep(_SLEEP)
            resp = await client.get(url)
            resp.raise_for_status()
            data: dict = resp.json()

        self._facts_cache[padded] = data
        logger.info(
            "EDGAR companyfacts cached",
            cik=padded,
            entity=data.get("entityName", "unknown"),
        )
        return data

    async def _get_cik(self, ticker: str) -> tuple[str, str]:
        """Resolve ticker → (zero-padded CIK, company_name).

        Results are cached in ``_cik_cache`` so repeated lookups hit only the
        in-memory dict.
        """
        key = ticker.upper()
        if key in self._cik_cache:
            return self._cik_cache[key]

        logger.info("EDGAR ticker lookup", ticker=key)
        async with httpx.AsyncClient(
            headers={**_HEADERS, "Host": "www.sec.gov"},
            timeout=self._timeout,
            follow_redirects=True,
        ) as client:
            await asyncio.sleep(_SLEEP)
            resp = await client.get(EDGAR_TICKERS)
            resp.raise_for_status()
            data: dict = resp.json()

        for entry in data.values():
            t = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            name = str(entry.get("title", ""))
            if t and cik:
                self._cik_cache[t] = (cik, name)

        if key not in self._cik_cache:
            raise LookupError(
                f"Ticker '{key}' not found in SEC company_tickers.json"
            )
        return self._cik_cache[key]

    def _extract_concept_pit(
        self,
        facts: dict,
        concept_key: str,
        as_of_date: date,
        period_type: Literal["annual", "quarterly", "any"] = "any",
    ) -> Optional[FinancialFact]:
        """Return the most appropriate PIT fact for *concept_key*.

        Selection algorithm
        -------------------
        1. Try each XBRL concept in CONCEPT_MAP[concept_key] in priority order.
        2. Filter observations: filed_date <= as_of_date.
        3. Further filter by period_type if specified.
        4. Among remaining, pick the observation with the latest period_end.
        5. If multiple filings cover the same period_end, prefer the one with
           the *earliest* filed_date (original as-reported, not a restatement).
        """
        if concept_key not in CONCEPT_MAP:
            return None

        candidates: list[FinancialFact] = []
        found_xbrl: Optional[str] = None

        for xbrl_concept in CONCEPT_MAP[concept_key]:
            raw_list = self._get_raw_observations(facts, xbrl_concept)
            if not raw_list:
                continue
            found_xbrl = xbrl_concept

            for obs in raw_list:
                filed_str = obs.get("filed")
                end_str = obs.get("end") or obs.get("instant")
                if not filed_str or not end_str:
                    continue
                try:
                    filed = date.fromisoformat(filed_str)
                    period_end = date.fromisoformat(end_str)
                except ValueError:
                    continue

                # PIT gate: only data that was publicly available
                if filed > as_of_date:
                    continue

                val = obs.get("val")
                if val is None:
                    continue

                start_str = obs.get("start")
                period_start = (
                    date.fromisoformat(start_str) if start_str else None
                )
                form = obs.get("form", "")
                is_ann = self._is_annual(obs)
                is_qtr = self._is_quarterly(obs, period_start, period_end)

                if period_type == "annual" and not is_ann:
                    continue
                if period_type == "quarterly" and not is_qtr:
                    continue

                unit = self._detect_unit(facts, xbrl_concept)
                candidates.append(
                    FinancialFact(
                        concept=concept_key,
                        xbrl_concept=xbrl_concept,
                        value=float(val),
                        unit=unit,
                        period_start=period_start,
                        period_end=period_end,
                        filed_date=filed,
                        form=form,
                        accession=obs.get("accn", ""),
                        fiscal_year=obs.get("fy"),
                        fiscal_period=obs.get("fp"),
                        is_annual=is_ann,
                    )
                )

            # Stop iterating through fallback concepts once we found data
            if candidates:
                break

        if not candidates:
            return None

        # Sort: latest period_end first; for ties, earliest filed_date first
        # (as-originally-reported takes priority over amendments)
        candidates.sort(key=lambda f: (-f.period_end.toordinal(), f.filed_date.toordinal()))
        return candidates[0]

    def _compute_ttm_value(
        self, facts: dict, concept_key: str, as_of_date: date
    ) -> Optional[float]:
        """Compute a trailing-twelve-months value for *concept_key*.

        For flow items (revenue, net_income, cfo, capex, …): sum the four
        most-recent non-overlapping quarterly observations filed <= as_of_date.

        For balance-sheet items (assets, equity, …): return the latest point
        value available on as_of_date (no summation).

        Returns None when fewer than four quarters are available.
        """
        is_balance = concept_key in _BALANCE_SHEET_CONCEPTS

        all_quarterly: list[FinancialFact] = []

        for xbrl_concept in CONCEPT_MAP[concept_key]:
            raw_list = self._get_raw_observations(facts, xbrl_concept)
            if not raw_list:
                continue

            for obs in raw_list:
                filed_str = obs.get("filed")
                end_str = obs.get("end") or obs.get("instant")
                if not filed_str or not end_str:
                    continue
                try:
                    filed = date.fromisoformat(filed_str)
                    period_end = date.fromisoformat(end_str)
                except ValueError:
                    continue

                if filed > as_of_date:
                    continue

                val = obs.get("val")
                if val is None:
                    continue

                start_str = obs.get("start")
                period_start = (
                    date.fromisoformat(start_str) if start_str else None
                )

                if not self._is_quarterly(obs, period_start, period_end):
                    continue

                unit = self._detect_unit(facts, xbrl_concept)
                all_quarterly.append(
                    FinancialFact(
                        concept=concept_key,
                        xbrl_concept=xbrl_concept,
                        value=float(val),
                        unit=unit,
                        period_start=period_start,
                        period_end=period_end,
                        filed_date=filed,
                        form=obs.get("form", ""),
                        accession=obs.get("accn", ""),
                        fiscal_year=obs.get("fy"),
                        fiscal_period=obs.get("fp"),
                        is_annual=False,
                    )
                )
            # Use first xbrl_concept that yields data
            if all_quarterly:
                break

        if not all_quarterly:
            return None

        if is_balance:
            # Latest available point value
            all_quarterly.sort(key=lambda f: (f.period_end, f.filed_date), reverse=True)
            return all_quarterly[0].value

        # Select four most-recent non-overlapping quarters (PIT order)
        all_quarterly.sort(key=lambda f: f.period_end, reverse=True)
        selected = _select_non_overlapping_quarters(all_quarterly, n=4)
        if len(selected) < 4:
            return None

        return sum(f.value for f in selected)

    # ------------------------------------------------------------------ #
    # Low-level helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_raw_observations(self, facts: dict, xbrl_concept: str) -> list[dict]:
        """Return the raw observation list for an XBRL concept across all units."""
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        concept_data = us_gaap.get(xbrl_concept, {})
        units = concept_data.get("units", {})
        # Prefer USD; fall back to shares, USD/shares, pure
        for unit_key in ("USD", "shares", "USD/shares", "pure"):
            if unit_key in units:
                return units[unit_key]
        # Take the first available unit
        for unit_key, obs_list in units.items():
            return obs_list
        return []

    def _detect_unit(self, facts: dict, xbrl_concept: str) -> str:
        """Return the best-fit unit string for an XBRL concept."""
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        concept_data = us_gaap.get(xbrl_concept, {})
        units = concept_data.get("units", {})
        for unit_key in ("USD", "shares", "USD/shares", "pure"):
            if unit_key in units:
                return unit_key
        return next(iter(units), "USD")

    def _is_annual(self, obs: dict) -> bool:
        """True when the observation comes from an annual filing form."""
        return obs.get("form", "") in _ANNUAL_FORMS

    def _is_quarterly(
        self,
        obs: dict,
        period_start: Optional[date],
        period_end: date,
    ) -> bool:
        """True when the observation represents a single fiscal quarter.

        Criteria (in priority order):
        1. form ∈ {10-Q, 10-QT} → definitely quarterly.
        2. Duration between start and end in [70, 110] days → quarterly.
        """
        if obs.get("form", "") in _QUARTERLY_FORMS:
            return True
        if period_start is not None:
            dur = (period_end - period_start).days
            if _QUARTERLY_MIN_DAYS <= dur <= _QUARTERLY_MAX_DAYS:
                return True
        return False

    def _duration_days(self, obs: dict) -> int:
        """Days between start and end dates of an observation."""
        start_str = obs.get("start")
        end_str = obs.get("end") or obs.get("instant")
        if not start_str or not end_str:
            return 0
        try:
            return (
                date.fromisoformat(end_str) - date.fromisoformat(start_str)
            ).days
        except ValueError:
            return 0

    def _most_recent_fact(
        self, facts: dict, as_of_date: date
    ) -> Optional[FinancialFact]:
        """Return the fact with the latest filed_date across all concepts."""
        best: Optional[FinancialFact] = None
        for concept_key, xbrl_concepts in CONCEPT_MAP.items():
            for xbrl_concept in xbrl_concepts:
                for obs in self._get_raw_observations(facts, xbrl_concept):
                    filed_str = obs.get("filed")
                    end_str = obs.get("end") or obs.get("instant")
                    if not filed_str or not end_str:
                        continue
                    try:
                        filed = date.fromisoformat(filed_str)
                        period_end = date.fromisoformat(end_str)
                    except ValueError:
                        continue
                    if filed > as_of_date:
                        continue
                    val = obs.get("val")
                    if val is None:
                        continue
                    start_str = obs.get("start")
                    period_start = (
                        date.fromisoformat(start_str) if start_str else None
                    )
                    fact = FinancialFact(
                        concept=concept_key,
                        xbrl_concept=xbrl_concept,
                        value=float(val),
                        unit=self._detect_unit(facts, xbrl_concept),
                        period_start=period_start,
                        period_end=period_end,
                        filed_date=filed,
                        form=obs.get("form", ""),
                        accession=obs.get("accn", ""),
                        fiscal_year=obs.get("fy"),
                        fiscal_period=obs.get("fp"),
                        is_annual=self._is_annual(obs),
                    )
                    if best is None or filed > best.filed_date:
                        best = fact
                break  # first successful xbrl_concept per canonical key
        return best


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------


def _deduplicate_facts(facts: list[FinancialFact]) -> list[FinancialFact]:
    """Keep one record per (period_end, form) pair.

    When the same period appears multiple times (e.g. from different XBRL
    concept names), keep the one with the earliest filed_date so we preserve
    the as-originally-reported value.
    """
    seen: dict[tuple, FinancialFact] = {}
    for f in facts:
        key = (f.period_end, f.form)
        if key not in seen or f.filed_date < seen[key].filed_date:
            seen[key] = f
    return list(seen.values())


def _select_non_overlapping_quarters(
    facts: list[FinancialFact], n: int = 4
) -> list[FinancialFact]:
    """Greedily select *n* non-overlapping quarterly observations.

    Input should be sorted by period_end descending (most recent first).
    Two observations overlap when their period ranges intersect.
    """
    selected: list[FinancialFact] = []
    for f in facts:
        if len(selected) >= n:
            break
        if f.period_start is None:
            # Estimate start from period_end (assume ~91-day quarter)
            est_start = f.period_end - timedelta(days=91)
            f = f.model_copy(update={"period_start": est_start})
        overlaps = any(
            not (f.period_end <= s.period_start or f.period_start >= s.period_end)
            for s in selected
            if s.period_start is not None
        )
        if not overlaps:
            selected.append(f)
    return selected


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


async def pit_financials(
    ticker: str, as_of_date: date, price: Optional[float] = None
) -> PointInTimeFinancials:
    """One-shot helper: PIT financial snapshot with a fresh engine instance."""
    engine = PointInTimeEngine()
    return await engine.get_as_of(ticker, as_of_date, price=price)


async def pit_time_series(
    ticker: str,
    concept: str,
    start_year: int = 2005,
) -> FinancialTimeSeries:
    """One-shot helper: full PIT time series for a single concept."""
    engine = PointInTimeEngine()
    return await engine.get_time_series(ticker, concept, start_year)


async def pit_panel(
    ticker: str,
    dates: list[date],
) -> BacktestPITData:
    """One-shot helper: PIT panel for backtesting — loads facts once."""
    engine = PointInTimeEngine()
    return await engine.get_backtest_panel(ticker, dates)
