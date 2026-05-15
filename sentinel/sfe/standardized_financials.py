"""
Standardized Financial Statements — normalize EDGAR XBRL data across 10,000+ companies.

Targets:
  dim_013  Income statement standardized (10K+ companies)   → 9+
  dim_014  Balance sheet standardized                        → 9+
  dim_015  Cash flow statement standardized                  → 9+

Public API
----------
FinancialStatementStandardizer   — core XBRL extraction and normalization
FinancialRatioEngine             — profitability / leverage / valuation ratios
CrossSectionalFinancials         — sector-wide comparison and screening
FinancialsCache                  — SQLite-backed TTL cache
financials_router                — FastAPI APIRouter

Helpers
-------
resolve_cik(ticker)  — EDGAR company search → CIK
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "INCOME_STATEMENT_MAP",
    "BALANCE_SHEET_MAP",
    "CASH_FLOW_MAP",
    "FinancialStatementStandardizer",
    "FinancialRatioEngine",
    "CrossSectionalFinancials",
    "FinancialsCache",
    "financials_router",
    "resolve_cik",
]

# ---------------------------------------------------------------------------
# EDGAR API constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
EDGAR_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_DELAY = 0.12   # 120 ms between requests — stay under 10 req/s
_TIMEOUT    = 30.0
_MAX_RETRY  = 3

# ---------------------------------------------------------------------------
# XBRL concept maps  (3-6 synonymous concepts per line item, priority-ordered)
# ---------------------------------------------------------------------------

INCOME_STATEMENT_MAP: dict[str, list[str] | None] = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenueNotFromContractWithCustomer",
        "SalesRevenueGoodsGross",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfGoodsSoldAndServicesSold",
        "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
        "CostOfServices",
        "CostOfGoodsAndServicesSold",
        "CostOfSalesPolicyTextBlock",
    ],
    "gross_profit": [
        "GrossProfit",
        "GrossProfitLoss",
    ],
    "r_and_d": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
        "SellingAndMarketingExpense",
        "SellingExpense",
        "MarketingAndAdvertisingExpense",
    ],
    "ebit": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
        "FinanceCostsNet",
        "InterestExpenseRelatedParty",
    ],
    "interest_income": [
        "InvestmentIncomeInterest",
        "InterestAndDividendIncomeOperating",
        "InterestIncomeOperating",
    ],
    "ebt": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "PretaxIncomeLoss",
    ],
    "income_tax": [
        "IncomeTaxExpenseBenefit",
        "CurrentIncomeTaxExpenseBenefit",
        "DeferredIncomeTaxExpenseBenefit",
        "IncomeTaxesPaidNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLossAttributableToParent",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
        "IncomeLossFromContinuingOperationsPerBasicShare",
        "BasicEarningsLossPerShare",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
        "DilutedEarningsLossPerShare",
    ],
    "shares_basic": [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
        "WeightedAverageBasicSharesOutstanding",
    ],
    "shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "DilutedWeightedAverageSharesOutstanding",
    ],
    "depreciation": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "Depreciation",
        "DepreciationAmortizationAndAccretionNet",
        "AmortizationOfIntangibleAssets",
    ],
    "operating_expenses": [
        "OperatingExpenses",
        "CostsAndExpenses",
        "NoninterestExpense",
    ],
    "ebitda": None,   # computed: ebit + depreciation
    "gross_margin": None,   # computed
    "ebitda_margin": None,  # computed
    "net_margin": None,     # computed
}

BALANCE_SHEET_MAP: dict[str, list[str] | None] = {
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashAndDueFromBanks",
        "CashEquivalentsAtCarryingValue",
        "CashAndCashEquivalentsAtFairValue",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
        "TradingSecurities",
        "ShortTermInvestmentsAndMarketableSecurities",
    ],
    "cash_and_short_term": [
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAndShortTermInvestments",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsReceivableNet",
        "TradeAndOtherReceivablesNetCurrent",
        "NotesAndLoansReceivableNetCurrent",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryGross",
        "FIFOInventoryAmount",
        "LIFOInventoryAmount",
        "RetailRelatedInventoryMerchandise",
    ],
    "other_current_assets": [
        "OtherAssetsCurrent",
        "PrepaidExpenseAndOtherAssetsCurrent",
        "DeferredIncomeTaxAssetsNet",
    ],
    "current_assets": [
        "AssetsCurrent",
        "CurrentAssets",
    ],
    "ppe_gross": [
        "PropertyPlantAndEquipmentGross",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization",
    ],
    "ppe_net": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        "PropertyPlantAndEquipmentNetOfAccumulatedDepreciation",
    ],
    "goodwill": [
        "Goodwill",
        "GoodwillGross",
        "GoodwillPeriodIncreaseDecrease",
    ],
    "intangibles": [
        "FiniteLivedIntangibleAssetsNet",
        "IntangibleAssetsNetExcludingGoodwill",
        "IndefiniteLivedIntangibleAssetsExcludingGoodwill",
        "IntangibleAssetsNetIncludingGoodwill",
    ],
    "long_term_investments": [
        "LongTermInvestments",
        "EquityMethodInvestments",
        "AvailableForSaleSecuritiesNoncurrent",
    ],
    "other_noncurrent_assets": [
        "OtherAssetsNoncurrent",
        "DeferredIncomeTaxAssetsNet",
    ],
    "total_assets": [
        "Assets",
        "TotalAssets",
    ],
    "accounts_payable": [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
        "AccountsPayableTradeCurrent",
    ],
    "accrued_liabilities": [
        "AccruedLiabilitiesCurrent",
        "AccruedExpensesAndOtherCurrentLiabilities",
        "EmployeeRelatedLiabilitiesCurrent",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "DebtCurrent",
        "NotesPayableCurrent",
        "ShortTermDebtAndCurrentPortionOfLongTermDebt",
    ],
    "deferred_revenue_current": [
        "DeferredRevenueCurrent",
        "ContractWithCustomerLiabilityCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
        "CurrentLiabilities",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermNotesPayable",
        "SeniorLongTermNotes",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "deferred_tax_liability": [
        "DeferredIncomeTaxLiabilitiesNet",
        "DeferredTaxLiabilitiesNoncurrent",
    ],
    "other_noncurrent_liabilities": [
        "OtherLiabilitiesNoncurrent",
        "OtherNoncurrentLiabilities",
    ],
    "total_liabilities": [
        "Liabilities",
        "TotalLiabilities",
        "LiabilitiesAndStockholdersEquity",
    ],
    "common_stock": [
        "CommonStockValue",
        "CommonStocksIncludingAdditionalPaidInCapital",
    ],
    "additional_paid_in_capital": [
        "AdditionalPaidInCapital",
        "AdditionalPaidInCapitalCommonStock",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarningsUnappropriated",
    ],
    "treasury_stock": [
        "TreasuryStockValue",
        "TreasuryStockCommonValue",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquityAttributableToParent",
        "LimitedLiabilityCompanyLlcMembersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssuedNet",
    ],
    "book_value_per_share": None,   # computed: equity / shares_outstanding
    "net_debt": None,               # computed: LTD + STD - cash
    "debt_to_equity": None,         # computed
    "current_ratio": None,          # computed
    "quick_ratio": None,            # computed
}

CASH_FLOW_MAP: dict[str, list[str] | None] = {
    "net_income_cf": [
        "NetIncomeLoss",
        "ProfitLoss",
        "IncomeLossFromContinuingOperations",
    ],
    "d_and_a": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ],
    "stock_based_comp": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
        "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue",
        "EmployeeBenefitsAndShareBasedCompensation",
    ],
    "working_capital_change": [
        "IncreaseDecreaseInOperatingCapital",
        "IncreaseDecreaseInOperatingLiabilities",
        "IncreaseDecreaseInOperatingAssets",
    ],
    "ar_change": [
        "IncreaseDecreaseInAccountsReceivable",
        "IncreaseDecreaseInReceivables",
    ],
    "inventory_change": [
        "IncreaseDecreaseInInventories",
        "IncreaseDecreaseInRetailRelatedInventories",
    ],
    "ap_change": [
        "IncreaseDecreaseInAccountsPayable",
        "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",
    ],
    "deferred_revenue_change": [
        "IncreaseDecreaseInDeferredRevenue",
        "IncreaseDecreaseInContractWithCustomerLiability",
    ],
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashGeneratedFromOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditureDiscontinuedOperations",
        "PaymentsForCapitalImprovements",
        "AcquisitionsNetOfCashAcquiredAndPurchasesOfBusinesses",
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
    ],
    "acquisitions": [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
        "BusinessCombinationConsiderationTransferred1",
    ],
    "proceeds_from_sales": [
        "ProceedsFromSaleOfPropertyPlantAndEquipment",
        "ProceedsFromDivestitureOfBusinesses",
        "ProceedsFromSaleOfAvailableForSaleSecurities",
    ],
    "investing_cf": [
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
        "CashFlowsFromUsedInInvestingActivities",
    ],
    "debt_issuance": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromIssuanceOfDebt",
        "ProceedsFromDebtNetOfIssuanceCosts",
    ],
    "debt_repayment": [
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
        "RepaymentsOfLongTermDebtAndCapitalSecurities",
    ],
    "dividends_paid": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfOrdinaryDividends",
        "PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits",
    ],
    "buybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "StockRepurchasedAndRetiredDuringPeriodValue",
        "PaymentsForRepurchaseOfEquity",
        "TreasuryStockValueAcquiredCostMethod",
    ],
    "stock_issuance": [
        "ProceedsFromIssuanceOfCommonStock",
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
    ],
    "financing_cf": [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
        "CashFlowsFromUsedInFinancingActivities",
    ],
    "net_change_cash": [
        "CashAndCashEquivalentsPeriodIncreaseDecrease",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
        "EffectOfExchangeRateOnCashAndCashEquivalents",
    ],
    "fcf": None,        # computed: operating_cf - capex
    "fcf_margin": None, # computed
}

# ---------------------------------------------------------------------------
# Ticker → CIK resolution
# ---------------------------------------------------------------------------

_CIK_CACHE: dict[str, str] = {}


def resolve_cik(ticker: str) -> str:
    """
    Resolve a ticker symbol to a zero-padded 10-digit CIK string.
    Uses EDGAR company_tickers.json (cached in-process).

    Raises
    ------
    ValueError  if the ticker is not found.
    """
    ticker_upper = ticker.upper()
    if ticker_upper in _CIK_CACHE:
        return _CIK_CACHE[ticker_upper]

    try:
        r = httpx.get(
            EDGAR_COMPANY_TICKERS,
            headers=EDGAR_HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"EDGAR company_tickers fetch failed: {exc}") from exc

    for entry in data.values():
        tick = str(entry.get("ticker", "")).upper()
        cik  = str(entry.get("cik_str", ""))
        if tick:
            _CIK_CACHE[tick] = cik.zfill(10)

    if ticker_upper not in _CIK_CACHE:
        raise ValueError(f"Ticker '{ticker}' not found in EDGAR company_tickers.")
    return _CIK_CACHE[ticker_upper]


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

class FinancialsCache:
    """
    SQLite-backed TTL cache for EDGAR company facts and derived statements.

    Tables
    ------
    company_facts  — raw companyfacts JSON, TTL 24 h
    statements     — serialized DataFrames, TTL 1 h
    """

    _FACTS_TTL_H = 24
    _STMT_TTL_H  = 1

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            cache_dir = Path(".sentinel") / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "financials.db")
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS company_facts (
                    cik        TEXT PRIMARY KEY,
                    json_blob  TEXT NOT NULL,
                    fetched_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS statements (
                    cik         TEXT NOT NULL,
                    period_type TEXT NOT NULL,
                    table_name  TEXT NOT NULL,
                    data_json   TEXT NOT NULL,
                    computed_at REAL NOT NULL,
                    PRIMARY KEY (cik, period_type, table_name)
                )
            """)
            conn.commit()

    # company_facts ──────────────────────────────────────────────────────────

    def get_facts(self, cik: str) -> dict | None:
        cutoff = time.time() - self._FACTS_TTL_H * 3600
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT json_blob, fetched_at FROM company_facts WHERE cik=?",
                (cik,),
            ).fetchone()
        if row and row[1] >= cutoff:
            return json.loads(row[0])
        return None

    def set_facts(self, cik: str, facts: dict) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO company_facts (cik, json_blob, fetched_at) VALUES (?,?,?)",
                (cik, json.dumps(facts), time.time()),
            )
            conn.commit()

    # statements ─────────────────────────────────────────────────────────────

    def get_statement(
        self, cik: str, period_type: str, table_name: str
    ) -> pd.DataFrame | None:
        cutoff = time.time() - self._STMT_TTL_H * 3600
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                """SELECT data_json, computed_at FROM statements
                   WHERE cik=? AND period_type=? AND table_name=?""",
                (cik, period_type, table_name),
            ).fetchone()
        if row and row[1] >= cutoff:
            return pd.read_json(row[0], orient="split")
        return None

    def set_statement(
        self,
        cik: str,
        period_type: str,
        table_name: str,
        df: pd.DataFrame,
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO statements
                   (cik, period_type, table_name, data_json, computed_at)
                   VALUES (?,?,?,?,?)""",
                (
                    cik,
                    period_type,
                    table_name,
                    df.to_json(orient="split", date_format="iso"),
                    time.time(),
                ),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Core standardizer
# ---------------------------------------------------------------------------

class FinancialStatementStandardizer:
    """
    Pull and normalize EDGAR XBRL data into consistent DataFrames.

    Parameters
    ----------
    pg_conn_string : str, optional
        PostgreSQL DSN.  Not yet used for storage but reserved for future
        integration with the SENTINEL SDS layer.
    cache_path : str, optional
        Path for the SQLite cache file (passed to FinancialsCache).
    """

    def __init__(
        self,
        pg_conn_string: str | None = None,
        cache_path: str | None = None,
    ) -> None:
        self._pg = pg_conn_string
        self._cache = FinancialsCache(db_path=cache_path)
        self._http = httpx.Client(
            headers=EDGAR_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    # ── EDGAR fetch ──────────────────────────────────────────────────────────

    def get_company_facts(self, cik: str) -> dict:
        """
        Fetch the full EDGAR companyfacts JSON for a CIK.
        Returns cached version if < 24 h old, otherwise re-fetches.
        """
        cik_padded = cik.zfill(10)
        cached = self._cache.get_facts(cik_padded)
        if cached is not None:
            logger.debug("company_facts cache hit", cik=cik_padded)
            return cached

        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRY):
            try:
                time.sleep(_RATE_DELAY)
                resp = self._http.get(url)
                resp.raise_for_status()
                facts = resp.json()
                self._cache.set_facts(cik_padded, facts)
                logger.info("company_facts fetched", cik=cik_padded)
                return facts
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise ValueError(
                        f"CIK {cik_padded} not found on EDGAR."
                    ) from exc
                last_exc = exc
                time.sleep(2 ** attempt)
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Failed to fetch company facts for CIK {cik_padded}: {last_exc}"
        )

    # ── Concept extraction ───────────────────────────────────────────────────

    def extract_metric(
        self,
        facts: dict,
        concept_list: list[str],
        period_type: str = "annual",
        units: str = "USD",
    ) -> pd.DataFrame:
        """
        Try each concept in priority order; return the first non-empty series.

        Parameters
        ----------
        facts       : raw EDGAR companyfacts JSON
        concept_list: list of XBRL concept names (without namespace prefix)
        period_type : "annual" (10-K) | "quarterly" (10-Q) | "all"
        units       : USD, shares, pure, etc.

        Returns
        -------
        DataFrame with columns: period_end, value, filed_date, accession, form
        Empty DataFrame if nothing found.
        """
        form_filter: set[str]
        if period_type == "annual":
            form_filter = {"10-K", "10-K/A", "20-F", "40-F"}
        elif period_type == "quarterly":
            form_filter = {"10-Q", "10-Q/A"}
        else:
            form_filter = set()

        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei  = facts.get("facts", {}).get("dei", {})

        for concept in concept_list:
            concept_data = gaap.get(concept) or dei.get(concept)
            if concept_data is None:
                continue

            unit_data = concept_data.get("units", {})
            rows_raw = unit_data.get(units) or unit_data.get("shares") or []

            rows = []
            seen_periods: set[str] = set()
            for r in rows_raw:
                form = r.get("form", "")
                if form_filter and form not in form_filter:
                    continue
                # annual: only include full-year (no startDate offset trick —
                # use filed accession to pick the most recent filing per period)
                end = r.get("end", "")
                if end in seen_periods:
                    continue
                seen_periods.add(end)
                rows.append(
                    {
                        "period_end":  end,
                        "value":       r.get("val"),
                        "filed_date":  r.get("filed", ""),
                        "accession":   r.get("accn", ""),
                        "form":        form,
                        "concept":     concept,
                    }
                )

            if rows:
                df = pd.DataFrame(rows)
                df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
                df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
                df = df.dropna(subset=["period_end", "value"])
                df = df.sort_values("period_end")
                return df

        return pd.DataFrame(
            columns=["period_end", "value", "filed_date", "accession", "form", "concept"]
        )

    # ── Income Statement ─────────────────────────────────────────────────────

    def get_income_statement(
        self,
        cik: str,
        periods: int = 5,
        period_type: str = "annual",
    ) -> pd.DataFrame:
        """
        Return a wide DataFrame (period_end as index, line items as columns).
        Derived: ebitda, gross_margin, ebitda_margin, net_margin.
        """
        cached = self._cache.get_statement(cik, period_type, "income_statement")
        if cached is not None:
            return cached.tail(periods)

        facts = self.get_company_facts(cik)
        rows: dict[str, dict] = {}

        for metric, concepts in INCOME_STATEMENT_MAP.items():
            if concepts is None:
                continue
            series = self.extract_metric(facts, concepts, period_type)
            if series.empty:
                continue
            for _, r in series.tail(periods * 2).iterrows():
                key = str(r["period_end"].date())
                if key not in rows:
                    rows[key] = {}
                if metric not in rows[key]:
                    rows[key][metric] = float(r["value"]) if r["value"] is not None else None

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "period_end"
        df = df.sort_index()

        # Derived metrics
        if "ebit" in df.columns and "depreciation" in df.columns:
            df["ebitda"] = df["ebit"].fillna(0) + df["depreciation"].fillna(0)
        if "gross_profit" in df.columns and "revenue" in df.columns:
            df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, float("nan"))
        if "ebitda" in df.columns and "revenue" in df.columns:
            df["ebitda_margin"] = df["ebitda"] / df["revenue"].replace(0, float("nan"))
        if "net_income" in df.columns and "revenue" in df.columns:
            df["net_margin"] = df["net_income"] / df["revenue"].replace(0, float("nan"))

        df = df.tail(periods)
        self._cache.set_statement(cik, period_type, "income_statement", df)
        return df

    # ── Balance Sheet ────────────────────────────────────────────────────────

    def get_balance_sheet(
        self,
        cik: str,
        periods: int = 5,
        period_type: str = "annual",
    ) -> pd.DataFrame:
        """
        Wide balance sheet DataFrame with computed:
        book_value_per_share, net_debt, debt_to_equity, current_ratio, quick_ratio.
        """
        cached = self._cache.get_statement(cik, period_type, "balance_sheet")
        if cached is not None:
            return cached.tail(periods)

        facts = self.get_company_facts(cik)
        rows: dict[str, dict] = {}

        for metric, concepts in BALANCE_SHEET_MAP.items():
            if concepts is None:
                continue
            series = self.extract_metric(facts, concepts, period_type)
            if series.empty:
                continue
            for _, r in series.tail(periods * 2).iterrows():
                key = str(r["period_end"].date())
                if key not in rows:
                    rows[key] = {}
                if metric not in rows[key]:
                    rows[key][metric] = float(r["value"]) if r["value"] is not None else None

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "period_end"
        df = df.sort_index()

        # Derived
        if "total_equity" in df.columns and "shares_outstanding" in df.columns:
            shares = df["shares_outstanding"].replace(0, float("nan"))
            df["book_value_per_share"] = df["total_equity"] / shares

        if "long_term_debt" in df.columns:
            ltd = df["long_term_debt"].fillna(0)
            std = df.get("short_term_debt", pd.Series(0, index=df.index)).fillna(0)
            cash = df.get("cash", pd.Series(0, index=df.index)).fillna(0)
            df["net_debt"] = ltd + std - cash

        if "total_liabilities" in df.columns and "total_equity" in df.columns:
            equity = df["total_equity"].replace(0, float("nan"))
            df["debt_to_equity"] = df["total_liabilities"] / equity

        if "current_assets" in df.columns and "current_liabilities" in df.columns:
            cl = df["current_liabilities"].replace(0, float("nan"))
            df["current_ratio"] = df["current_assets"] / cl
            cash_plus_ar = (
                df.get("cash", pd.Series(0, index=df.index)).fillna(0)
                + df.get("short_term_investments", pd.Series(0, index=df.index)).fillna(0)
                + df.get("accounts_receivable", pd.Series(0, index=df.index)).fillna(0)
            )
            df["quick_ratio"] = cash_plus_ar / cl

        df = df.tail(periods)
        self._cache.set_statement(cik, period_type, "balance_sheet", df)
        return df

    # ── Cash Flow Statement ──────────────────────────────────────────────────

    def get_cash_flow_statement(
        self,
        cik: str,
        periods: int = 5,
        period_type: str = "annual",
        market_cap: float | None = None,
    ) -> pd.DataFrame:
        """
        Wide cash flow DataFrame with computed:
        fcf = operating_cf - capex, fcf_margin (if market_cap provided: fcf_yield).
        """
        cached = self._cache.get_statement(cik, period_type, "cash_flow")
        if cached is not None:
            return cached.tail(periods)

        facts = self.get_company_facts(cik)
        rows: dict[str, dict] = {}

        for metric, concepts in CASH_FLOW_MAP.items():
            if concepts is None:
                continue
            series = self.extract_metric(facts, concepts, period_type)
            if series.empty:
                continue
            for _, r in series.tail(periods * 2).iterrows():
                key = str(r["period_end"].date())
                if key not in rows:
                    rows[key] = {}
                if metric not in rows[key]:
                    rows[key][metric] = float(r["value"]) if r["value"] is not None else None

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "period_end"
        df = df.sort_index()

        # capex is typically reported as negative in cash flow; normalise to negative
        if "capex" in df.columns:
            df["capex"] = df["capex"].apply(
                lambda v: -abs(v) if v is not None and not math.isnan(v) else v
            )

        if "operating_cf" in df.columns and "capex" in df.columns:
            df["fcf"] = df["operating_cf"].fillna(0) + df["capex"].fillna(0)

        if "fcf" in df.columns and market_cap and market_cap > 0:
            df["fcf_yield"] = df["fcf"] / market_cap

        df = df.tail(periods)
        self._cache.set_statement(cik, period_type, "cash_flow", df)
        return df

    # ── TTM ─────────────────────────────────────────────────────────────────

    def get_ttm_financials(self, cik: str) -> dict[str, pd.DataFrame]:
        """
        Trailing 12 months financials built from the last 4 quarterly filings.
        Sum flow items (IS, CF); take most recent period-end snapshot for BS.
        """
        is_q = self.get_income_statement(cik, periods=8, period_type="quarterly")
        bs_q = self.get_balance_sheet(cik, periods=8, period_type="quarterly")
        cf_q = self.get_cash_flow_statement(cik, periods=8, period_type="quarterly")

        ttm_is  = _sum_last_n(is_q, 4)
        ttm_cf  = _sum_last_n(cf_q, 4)
        ttm_bs  = bs_q.iloc[-1:] if not bs_q.empty else pd.DataFrame()

        # Recompute derived on TTM totals
        if not ttm_is.empty:
            _compute_is_derived(ttm_is)
        if not ttm_cf.empty:
            _compute_cf_derived(ttm_cf)

        return {
            "income_statement": ttm_is,
            "balance_sheet":    ttm_bs,
            "cash_flow":        ttm_cf,
        }

    # ── Full financials ──────────────────────────────────────────────────────

    def get_full_financials(
        self,
        cik: str,
        ticker: str | None = None,
        periods: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """
        Pull all three statements + compute a ratios DataFrame.

        Returns
        -------
        {income_statement, balance_sheet, cash_flow, ratios}
        """
        is_df = self.get_income_statement(cik, periods=periods)
        bs_df = self.get_balance_sheet(cik, periods=periods)
        cf_df = self.get_cash_flow_statement(cik, periods=periods)

        ratio_engine = FinancialRatioEngine()
        ratios = ratio_engine.compute_ratios(is_df, bs_df, cf_df)

        return {
            "income_statement": is_df,
            "balance_sheet":    bs_df,
            "cash_flow":        cf_df,
            "ratios":           ratios,
        }


# ---------------------------------------------------------------------------
# Helpers for TTM aggregation
# ---------------------------------------------------------------------------

def _sum_last_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sum the last n rows (quarters) into a single-row DataFrame."""
    if df.empty or len(df) < 1:
        return df
    subset = df.tail(n)
    numeric = subset.select_dtypes(include="number")
    total = numeric.sum().to_frame().T
    total.index = [subset.index[-1]]
    total.index.name = "period_end"
    return total


def _compute_is_derived(df: pd.DataFrame) -> None:
    """In-place derived metrics for IS DataFrame."""
    if "ebit" in df.columns and "depreciation" in df.columns:
        df["ebitda"] = df["ebit"].fillna(0) + df["depreciation"].fillna(0)
    if "gross_profit" in df.columns and "revenue" in df.columns:
        df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, float("nan"))
    if "ebitda" in df.columns and "revenue" in df.columns:
        df["ebitda_margin"] = df["ebitda"] / df["revenue"].replace(0, float("nan"))
    if "net_income" in df.columns and "revenue" in df.columns:
        df["net_margin"] = df["net_income"] / df["revenue"].replace(0, float("nan"))


def _compute_cf_derived(df: pd.DataFrame) -> None:
    """In-place derived metrics for CF DataFrame."""
    if "operating_cf" in df.columns and "capex" in df.columns:
        df["fcf"] = df["operating_cf"].fillna(0) + df["capex"].fillna(0)


# ---------------------------------------------------------------------------
# Ratio Engine
# ---------------------------------------------------------------------------

class FinancialRatioEngine:
    """
    Compute institutional-grade financial ratios from normalized statements.
    Accepts wide DataFrames from FinancialStatementStandardizer.
    """

    def compute_ratios(
        self,
        is_df: pd.DataFrame,
        bs_df: pd.DataFrame,
        cf_df: pd.DataFrame,
        price: float | None = None,
        market_cap: float | None = None,
        shares_outstanding: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute the full suite of financial ratios, period by period.

        Valuation     — P/E, P/S, P/B, EV/EBITDA, EV/Revenue
        Profitability — ROE, ROA, ROIC, margins (gross, EBITDA, net, FCF)
        Liquidity     — current_ratio, quick_ratio, cash_ratio
        Leverage      — D/E, net_debt/EBITDA, interest_coverage
        Efficiency    — asset_turnover, inventory_turns, DSO, DPO, CCC
        Growth (YoY)  — revenue, EBITDA, EPS, FCF
        """
        # Align indices (period_end)
        all_idx = sorted(
            set(is_df.index).union(bs_df.index).union(cf_df.index)
        )
        result: dict[str, dict] = {}

        for idx in all_idx:
            r: dict[str, float | None] = {}

            is_row = _safe_row(is_df, idx)
            bs_row = _safe_row(bs_df, idx)
            cf_row = _safe_row(cf_df, idx)

            rev      = is_row.get("revenue")
            ni       = is_row.get("net_income")
            ebitda   = is_row.get("ebitda")
            ebit     = is_row.get("ebit")
            eps_d    = is_row.get("eps_diluted")
            gross_p  = is_row.get("gross_profit")
            int_exp  = is_row.get("interest_expense")

            ta       = bs_row.get("total_assets")
            equity   = bs_row.get("total_equity")
            ca       = bs_row.get("current_assets")
            cl       = bs_row.get("current_liabilities")
            inv      = bs_row.get("inventory")
            ar       = bs_row.get("accounts_receivable")
            ap       = bs_row.get("accounts_payable")
            cash_bs  = bs_row.get("cash")
            ltd      = bs_row.get("long_term_debt")
            std      = bs_row.get("short_term_debt")
            net_debt = bs_row.get("net_debt")
            bvps     = bs_row.get("book_value_per_share")

            op_cf    = cf_row.get("operating_cf")
            capex    = cf_row.get("capex")
            fcf      = cf_row.get("fcf")

            # ── Valuation ───────────────────────────────────────────────────
            if price and shares_outstanding and ni and ni != 0:
                mc = price * shares_outstanding
                r["market_cap"] = mc
                r["pe"]   = mc / ni if ni and ni > 0 else None
                r["ps"]   = _safe_div(mc, rev)
                r["pb"]   = _safe_div(price, bvps) if bvps else None
                gross_debt = (ltd or 0) + (std or 0)
                ev    = mc + gross_debt - (cash_bs or 0)
                r["ev"]          = ev
                r["ev_ebitda"]   = _safe_div(ev, ebitda)
                r["ev_revenue"]  = _safe_div(ev, rev)
                r["ev_ebit"]     = _safe_div(ev, ebit)

            # ── Profitability ───────────────────────────────────────────────
            r["gross_margin"]   = _safe_div(gross_p, rev)
            r["ebitda_margin"]  = _safe_div(ebitda, rev)
            r["net_margin"]     = _safe_div(ni, rev)
            r["fcf_margin"]     = _safe_div(fcf, rev)
            r["roe"]            = _safe_div(ni, equity)
            r["roa"]            = _safe_div(ni, ta)
            r["roic"]           = _compute_roic(ebit, ta, cl)
            r["return_on_equity"] = r["roe"]

            # ── Liquidity ───────────────────────────────────────────────────
            r["current_ratio"]  = _safe_div(ca, cl)
            r["quick_ratio"]    = _safe_div((ca or 0) - (inv or 0), cl)
            r["cash_ratio"]     = _safe_div(cash_bs, cl)

            # ── Leverage ────────────────────────────────────────────────────
            r["debt_to_equity"]    = _safe_div(ltd, equity)
            r["net_debt_ebitda"]   = _safe_div(net_debt, ebitda)
            r["interest_coverage"] = _safe_div(ebit, int_exp)
            r["debt_to_assets"]    = _safe_div((ltd or 0) + (std or 0), ta)

            # ── Efficiency ──────────────────────────────────────────────────
            r["asset_turnover"]    = _safe_div(rev, ta)
            r["inventory_turnover"] = _safe_div(rev, inv)
            r["dso"]               = _days_outstanding(ar, rev)
            r["dpo"]               = _days_outstanding(ap, rev)
            r["ccc"]               = _compute_ccc(ar, inv, ap, rev)

            result[idx] = r

        if not result:
            return pd.DataFrame()

        out = pd.DataFrame.from_dict(result, orient="index")
        out.index.name = "period_end"
        out = out.sort_index()

        # ── Growth rates (YoY) ───────────────────────────────────────────────
        _add_growth(out, is_df, "revenue",    "revenue_growth")
        _add_growth(out, is_df, "ebitda",     "ebitda_growth")
        _add_growth(out, is_df, "eps_diluted", "eps_growth")
        _add_growth(out, cf_df, "fcf",        "fcf_growth")

        return out

    def compute_dupont(
        self,
        is_df: pd.DataFrame,
        bs_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        3-factor DuPont: ROE = net_margin × asset_turnover × equity_multiplier
        5-factor DuPont: ROE = tax_burden × interest_burden × ebit_margin × asset_turnover × equity_multiplier

        Returns DataFrame with both decompositions, period_end as index.
        """
        all_idx = sorted(set(is_df.index).union(bs_df.index))
        rows = []

        for idx in all_idx:
            is_row = _safe_row(is_df, idx)
            bs_row = _safe_row(bs_df, idx)

            rev    = is_row.get("revenue")
            ni     = is_row.get("net_income")
            ebit   = is_row.get("ebit")
            ebt    = is_row.get("ebt")
            ta     = bs_row.get("total_assets")
            equity = bs_row.get("total_equity")

            net_margin  = _safe_div(ni, rev)
            asset_turn  = _safe_div(rev, ta)
            eq_mult     = _safe_div(ta, equity)

            roe_3f = _mul(net_margin, asset_turn, eq_mult)

            # 5-factor
            tax_burden       = _safe_div(ni, ebt)
            interest_burden  = _safe_div(ebt, ebit)
            ebit_margin      = _safe_div(ebit, rev)

            roe_5f = _mul(tax_burden, interest_burden, ebit_margin, asset_turn, eq_mult)

            rows.append(
                {
                    "period_end":      idx,
                    "roe_3factor":     roe_3f,
                    "net_margin":      net_margin,
                    "asset_turnover":  asset_turn,
                    "equity_multiplier": eq_mult,
                    "roe_5factor":     roe_5f,
                    "tax_burden":      tax_burden,
                    "interest_burden": interest_burden,
                    "ebit_margin":     ebit_margin,
                }
            )

        df = pd.DataFrame(rows).set_index("period_end")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df


# ---------------------------------------------------------------------------
# Ratio helpers
# ---------------------------------------------------------------------------

def _safe_row(df: pd.DataFrame, idx) -> dict:
    """Return row as dict; return {} if index not present."""
    if df.empty or idx not in df.index:
        return {}
    row = df.loc[idx]
    return row.to_dict() if hasattr(row, "to_dict") else {}


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0 or math.isnan(b):
        return None
    return a / b


def _mul(*args: float | None) -> float | None:
    result = 1.0
    for v in args:
        if v is None:
            return None
        result *= v
    return result


def _compute_roic(
    ebit: float | None,
    total_assets: float | None,
    current_liabilities: float | None,
) -> float | None:
    """ROIC = EBIT(1-t) / (total_assets - current_liabilities); assumes 21% tax."""
    if ebit is None or total_assets is None:
        return None
    invested_capital = total_assets - (current_liabilities or 0)
    if invested_capital == 0:
        return None
    nopat = ebit * (1 - 0.21)
    return nopat / invested_capital


def _days_outstanding(balance: float | None, revenue: float | None) -> float | None:
    """balance / (revenue / 365) → days."""
    if balance is None or revenue is None or revenue == 0:
        return None
    return (balance / revenue) * 365


def _compute_ccc(
    ar: float | None,
    inv: float | None,
    ap: float | None,
    rev: float | None,
) -> float | None:
    """CCC = DSO + DIO - DPO."""
    dso = _days_outstanding(ar, rev)
    dio = _days_outstanding(inv, rev)
    dpo = _days_outstanding(ap, rev)
    if dso is None or dio is None or dpo is None:
        return None
    return dso + dio - dpo


def _add_growth(
    ratios: pd.DataFrame,
    source: pd.DataFrame,
    col: str,
    result_col: str,
) -> None:
    """Append YoY growth rate to ratios DataFrame."""
    if col not in source.columns:
        return
    growth = source[col].pct_change()
    for idx in ratios.index:
        if idx in growth.index:
            ratios.loc[idx, result_col] = growth.loc[idx]


# ---------------------------------------------------------------------------
# Cross-sectional financials
# ---------------------------------------------------------------------------

class CrossSectionalFinancials:
    """
    Sector-wide financial comparison using EDGAR company search + facts API.

    Pull the latest annual income statement for all companies in a sector,
    compute percentile ranks, and run configurable screens.
    """

    _SIC_SEARCH = "https://www.sec.gov/cgi-bin/browse-edgar"
    _COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, cache_path: str | None = None) -> None:
        self._standardizer = FinancialStatementStandardizer(cache_path=cache_path)

    def _get_ciks_for_sic(self, sic_code: str, limit: int = 100) -> list[str]:
        """
        Return a list of CIKs filing under a given SIC code via EDGAR browse.
        Paginates until `limit` is reached.
        """
        url = self._SIC_SEARCH
        params = {
            "action":  "getcompany",
            "SIC":     sic_code,
            "type":    "10-K",
            "dateb":   "",
            "owner":   "include",
            "count":   min(limit, 40),
            "search_text": "",
            "output":  "atom",
        }
        try:
            time.sleep(_RATE_DELAY)
            r = httpx.get(url, params=params, headers=EDGAR_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("SIC search failed", sic=sic_code, error=str(exc))
            return []

        import re
        ciks = re.findall(r"CIK=(\d+)", r.text)
        return list(dict.fromkeys(ciks))[:limit]   # deduplicate, preserve order

    def get_sector_financials(
        self,
        sic_code: str | None = None,
        gics_sector: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Pull latest annual IS for all companies in sector.

        Parameters
        ----------
        sic_code    : SIC code (preferred — EDGAR native)
        gics_sector : GICS sector name (used as fallback keyword search)
        limit       : max companies to include

        Returns
        -------
        DataFrame  rows=companies, columns=key metrics
        """
        if sic_code:
            ciks = self._get_ciks_for_sic(sic_code, limit)
        elif gics_sector:
            ciks = self._search_ciks_by_sector(gics_sector, limit)
        else:
            raise ValueError("Provide sic_code or gics_sector.")

        logger.info("cross_sectional pull", n_companies=len(ciks))
        records = []
        for cik in ciks:
            try:
                is_df = self._standardizer.get_income_statement(cik, periods=1)
                bs_df = self._standardizer.get_balance_sheet(cik, periods=1)
                cf_df = self._standardizer.get_cash_flow_statement(cik, periods=1)
                if is_df.empty:
                    continue
                row = {"cik": cik}
                for col in is_df.columns:
                    row[col] = is_df.iloc[-1][col]
                for col in ["total_assets", "total_equity", "net_debt",
                            "current_ratio", "debt_to_equity"]:
                    if col in bs_df.columns:
                        row[col] = bs_df.iloc[-1][col]
                for col in ["operating_cf", "fcf", "capex"]:
                    if col in cf_df.columns:
                        row[col] = cf_df.iloc[-1][col]
                records.append(row)
            except Exception as exc:
                logger.warning("cross_sectional skip", cik=cik, error=str(exc))
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("cik")
        return df

    def _search_ciks_by_sector(self, sector: str, limit: int) -> list[str]:
        """Keyword search via EFTS — less precise than SIC but handles GICS names."""
        try:
            params = {
                "q":         f'"{sector}"',
                "dateRange": "custom",
                "forms":     "10-K",
                "hits.hits.total.value": limit,
                "_source":   "entity_id",
            }
            time.sleep(_RATE_DELAY)
            r = httpx.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                headers=EDGAR_HEADERS,
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            hits = r.json().get("hits", {}).get("hits", [])
            ciks = [
                str(h.get("_source", {}).get("entity_id", "")).zfill(10)
                for h in hits
                if h.get("_source", {}).get("entity_id")
            ]
            return list(dict.fromkeys(ciks))[:limit]
        except httpx.HTTPError:
            return []

    def compute_sector_percentiles(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Percentile-rank each company within its cohort for every numeric column.
        Returns DataFrame of same shape with values in [0, 1].
        """
        numeric = metrics_df.select_dtypes(include="number")
        return numeric.rank(pct=True)

    def screen(
        self,
        df: pd.DataFrame | None = None,
        sic_code: str | None = None,
        min_revenue: float | None = None,
        min_ebitda_margin: float | None = None,
        max_net_debt_ebitda: float | None = None,
        min_roic: float | None = None,
        min_fcf: float | None = None,
        min_gross_margin: float | None = None,
    ) -> pd.DataFrame:
        """
        Screen companies by fundamental criteria.

        If `df` is not provided, pull sector financials first
        (requires sic_code or gics_sector).
        """
        if df is None:
            if sic_code:
                df = self.get_sector_financials(sic_code=sic_code)
            else:
                raise ValueError("Provide df or sic_code.")

        mask = pd.Series(True, index=df.index)

        if min_revenue is not None and "revenue" in df.columns:
            mask &= df["revenue"].fillna(0) >= min_revenue

        if min_ebitda_margin is not None and "ebitda_margin" in df.columns:
            mask &= df["ebitda_margin"].fillna(0) >= min_ebitda_margin

        if max_net_debt_ebitda is not None:
            if "net_debt" in df.columns and "ebitda" in df.columns:
                nd_ebitda = df["net_debt"] / df["ebitda"].replace(0, float("nan"))
                mask &= nd_ebitda.fillna(float("inf")) <= max_net_debt_ebitda

        if min_roic is not None and "roic" in df.columns:
            mask &= df["roic"].fillna(-1) >= min_roic

        if min_fcf is not None and "fcf" in df.columns:
            mask &= df["fcf"].fillna(0) >= min_fcf

        if min_gross_margin is not None and "gross_margin" in df.columns:
            mask &= df["gross_margin"].fillna(0) >= min_gross_margin

        return df[mask]


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

financials_router = APIRouter(prefix="/api/financials", tags=["standardized-financials"])

_standardizer: FinancialStatementStandardizer | None = None
_ratio_engine: FinancialRatioEngine = FinancialRatioEngine()
_xsectional: CrossSectionalFinancials | None = None


def _get_standardizer() -> FinancialStatementStandardizer:
    global _standardizer
    if _standardizer is None:
        _standardizer = FinancialStatementStandardizer()
    return _standardizer


def _get_xsectional() -> CrossSectionalFinancials:
    global _xsectional
    if _xsectional is None:
        _xsectional = CrossSectionalFinancials()
    return _xsectional


def _df_to_response(df: pd.DataFrame) -> dict:
    """Serialize DataFrame to JSON-safe dict."""
    if df.empty:
        return {"rows": [], "columns": []}
    reset = df.reset_index()
    reset.columns = [str(c) for c in reset.columns]
    for col in reset.columns:
        if pd.api.types.is_datetime64_any_dtype(reset[col]):
            reset[col] = reset[col].dt.strftime("%Y-%m-%d")
    return reset.where(pd.notnull(reset), other=None).to_dict(orient="records")


def _resolve_to_cik(ticker: str) -> str:
    try:
        return resolve_cik(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@financials_router.get("/{ticker}/income-statement")
def income_statement(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    period_type: str = Query(default="annual", pattern="^(annual|quarterly)$"),
):
    """Standardized income statement for a given ticker."""
    cik = _resolve_to_cik(ticker)
    try:
        df = _get_standardizer().get_income_statement(cik, periods=periods, period_type=period_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ticker":      ticker,
        "cik":         cik,
        "period_type": period_type,
        "data":        _df_to_response(df),
    }


@financials_router.get("/{ticker}/balance-sheet")
def balance_sheet(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    period_type: str = Query(default="annual", pattern="^(annual|quarterly)$"),
):
    """Standardized balance sheet for a given ticker."""
    cik = _resolve_to_cik(ticker)
    try:
        df = _get_standardizer().get_balance_sheet(cik, periods=periods, period_type=period_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ticker":      ticker,
        "cik":         cik,
        "period_type": period_type,
        "data":        _df_to_response(df),
    }


@financials_router.get("/{ticker}/cash-flow")
def cash_flow(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    period_type: str = Query(default="annual", pattern="^(annual|quarterly)$"),
):
    """Standardized cash flow statement for a given ticker."""
    cik = _resolve_to_cik(ticker)
    try:
        df = _get_standardizer().get_cash_flow_statement(cik, periods=periods, period_type=period_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ticker":      ticker,
        "cik":         cik,
        "period_type": period_type,
        "data":        _df_to_response(df),
    }


@financials_router.get("/{ticker}/ratios")
def financial_ratios(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    price: float | None = Query(default=None),
    market_cap: float | None = Query(default=None),
    shares_outstanding: float | None = Query(default=None),
):
    """Full ratio suite for a ticker (valuation requires price + shares_outstanding)."""
    cik = _resolve_to_cik(ticker)
    try:
        std = _get_standardizer()
        is_df = std.get_income_statement(cik, periods=periods)
        bs_df = std.get_balance_sheet(cik, periods=periods)
        cf_df = std.get_cash_flow_statement(cik, periods=periods)
        ratios = _ratio_engine.compute_ratios(
            is_df, bs_df, cf_df,
            price=price,
            market_cap=market_cap,
            shares_outstanding=shares_outstanding,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ticker": ticker,
        "cik":    cik,
        "data":   _df_to_response(ratios),
    }


@financials_router.get("/{ticker}/full")
def full_financials(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
):
    """Income statement + balance sheet + cash flow + ratios in one call."""
    cik = _resolve_to_cik(ticker)
    try:
        bundle = _get_standardizer().get_full_financials(cik, ticker=ticker, periods=periods)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ticker": ticker,
        "cik":    cik,
        "income_statement": _df_to_response(bundle["income_statement"]),
        "balance_sheet":    _df_to_response(bundle["balance_sheet"]),
        "cash_flow":        _df_to_response(bundle["cash_flow"]),
        "ratios":           _df_to_response(bundle["ratios"]),
    }


@financials_router.get("/{ticker}/ttm")
def ttm_financials(ticker: str):
    """Trailing 12-month financials assembled from last 4 quarterly filings."""
    cik = _resolve_to_cik(ticker)
    try:
        bundle = _get_standardizer().get_ttm_financials(cik)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ticker": ticker,
        "cik":    cik,
        "period": "TTM",
        "income_statement": _df_to_response(bundle["income_statement"]),
        "balance_sheet":    _df_to_response(bundle["balance_sheet"]),
        "cash_flow":        _df_to_response(bundle["cash_flow"]),
    }


@financials_router.get("/sector/{sector}/summary")
def sector_summary(
    sector: str,
    sic_code: str | None = Query(default=None),
    limit: int = Query(default=50, ge=5, le=200),
    min_revenue: float | None = Query(default=None),
    min_ebitda_margin: float | None = Query(default=None),
    max_net_debt_ebitda: float | None = Query(default=None),
):
    """
    Cross-sectional sector summary.
    Pass sic_code (preferred) or use sector as GICS keyword.
    Optionally apply fundamental screens.
    """
    xs = _get_xsectional()
    try:
        df = xs.get_sector_financials(
            sic_code=sic_code,
            gics_sector=sector if not sic_code else None,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if df.empty:
        return {"sector": sector, "count": 0, "data": []}

    screened = xs.screen(
        df=df,
        min_revenue=min_revenue,
        min_ebitda_margin=min_ebitda_margin,
        max_net_debt_ebitda=max_net_debt_ebitda,
    )
    pctiles = xs.compute_sector_percentiles(screened)

    return {
        "sector":      sector,
        "sic_code":    sic_code,
        "count":       len(screened),
        "data":        _df_to_response(screened),
        "percentiles": _df_to_response(pctiles),
    }
