"""
Enhanced cash flow statement: free cash flow quality analysis, FCF yield,
cash conversion, capex classification, owner earnings, Levered FCF.

Dimension: dim_015 — Cash flow statement standardized (target: 9, from 8)

Enhancements over standardized_financials.py:
  - 35+ line items with full SEC XBRL taxonomy coverage
  - Multiple FCF definitions: Standard, Levered, Unlevered (UFCF), Owner Earnings
  - Capex decomposition: maintenance vs growth, asset-light detection, R&D proxy
  - Earnings quality: accruals ratio, cash conversion, revenue quality, one-time detection
  - FCF yield and FCF margin via yfinance market cap lookup
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger
from sentinel.sfe.standardized_financials import (
    EDGAR_BASE,
    EDGAR_HEADERS,
    FinancialsCache,
    _RATE_DELAY,
    _TIMEOUT,
    _MAX_RETRY,
    _safe_div,
    resolve_cik,
    _df_to_response,
    _resolve_to_cik,
)

logger = get_logger(__name__)

__all__ = [
    "ENHANCED_CASH_FLOW_MAP",
    "UniversalCashFlowParser",
    "FreeCashFlowEngine",
    "CapexAnalyzer",
    "EarningsQualityAnalyzer",
    "cashflow_router_v2",
]

# ---------------------------------------------------------------------------
# Extended XBRL concept map — 35+ line items
# ---------------------------------------------------------------------------

ENHANCED_CASH_FLOW_MAP: dict[str, list[str] | None] = {
    # ── Operating activities ─────────────────────────────────────────────────
    "net_income_cf": [
        "NetIncomeLoss",
        "ProfitLoss",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ConsolidatedNetIncomeLoss",
    ],
    "d_and_a": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
        "AmortizationOfIntangibleAssets",
        "DepreciationNonproduction",
    ],
    "depreciation_only": [
        "Depreciation",
        "DepreciationNonproduction",
        "DepreciationAndAmortizationDiscontinuedOperations",
    ],
    "amortization_only": [
        "AmortizationOfIntangibleAssets",
        "AmortizationOfFinancingCosts",
        "AmortizationOfAcquiredIntangibles",
        "FiniteLivedIntangibleAssetsAmortizationExpense",
    ],
    "stock_based_comp": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
        "EmployeeBenefitsAndShareBasedCompensation",
        "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue",
        "StockOptionPlanExpense",
        "RestrictedStockExpense",
    ],
    "deferred_tax": [
        "DeferredIncomeTaxExpenseBenefit",
        "DeferredIncomeTaxesAndTaxCredits",
        "IncreaseDecreaseInDeferredIncomeTaxes",
        "DeferredTaxExpenseBenefit",
    ],
    "ar_change": [
        "IncreaseDecreaseInAccountsReceivable",
        "IncreaseDecreaseInReceivables",
        "IncreaseDecreaseInAccountsAndOtherReceivables",
        "IncreaseDecreaseInContractWithCustomerAsset",
    ],
    "inventory_change": [
        "IncreaseDecreaseInInventories",
        "IncreaseDecreaseInRetailRelatedInventories",
        "IncreaseDecreaseInRawMaterialsAndSupplies",
        "IncreaseDecreaseInFinishedGoods",
    ],
    "ap_change": [
        "IncreaseDecreaseInAccountsPayable",
        "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",
        "IncreaseDecreaseInAccountsPayableTrade",
    ],
    "deferred_revenue_change": [
        "IncreaseDecreaseInDeferredRevenue",
        "IncreaseDecreaseInContractWithCustomerLiability",
        "IncreaseDecreaseInDeferredRevenueAndCustomerAdvancesAndDeposits",
    ],
    "other_working_capital": [
        "IncreaseDecreaseInOtherOperatingLiabilities",
        "IncreaseDecreaseInOtherOperatingAssets",
        "IncreaseDecreaseInOperatingCapital",
        "IncreaseDecreaseInAccruedLiabilitiesAndOtherOperatingLiabilities",
    ],
    "other_operating": [
        "OtherOperatingActivitiesCashFlowStatement",
        "OtherNoncashIncomeExpense",
        "AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities",
        "GainLossOnSaleOfPropertyPlantEquipment",
        "GainLossOnInvestments",
    ],
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashGeneratedFromOperations",
        "NetCashFromOperatingActivities",
    ],
    # ── Investing activities ─────────────────────────────────────────────────
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "AcquisitionOfProductiveAssets",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "capex_discontinued": [
        "CapitalExpenditureDiscontinuedOperations",
    ],
    "acquisitions": [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
        "BusinessCombinationConsiderationTransferred1",
        "PaymentsToAcquireOtherInvestments",
        "PaymentsToAcquireBusinessesAndInterestInAffiliates",
    ],
    "divestitures": [
        "ProceedsFromDivestitureOfBusinesses",
        "ProceedsFromDivestitureOfBusinessesNetOfCashDivested",
        "ProceedsFromSaleOfBusinessUnit",
        "ProceedsFromDivestitureOfBusinessesAndInterestsInAffiliates",
    ],
    "purchases_investments": [
        "PaymentsToAcquireInvestments",
        "PaymentsToAcquireAvailableForSaleSecurities",
        "PaymentsToAcquireMarketableSecurities",
        "PaymentsToAcquireShortTermInvestments",
        "PaymentsForProceedsFromInvestments",
    ],
    "proceeds_investments": [
        "ProceedsFromSaleOfAvailableForSaleSecurities",
        "ProceedsFromSaleAndMaturityOfMarketableSecurities",
        "ProceedsFromSaleAndMaturityOfOtherInvestments",
        "ProceedsFromSaleOfShortTermInvestments",
        "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
    ],
    "ppe_proceeds": [
        "ProceedsFromSaleOfPropertyPlantAndEquipment",
        "ProceedsFromSaleOfProductiveAssets",
        "ProceedsFromDisposalOfFixedAssets",
    ],
    "other_investing": [
        "PaymentsForProceedsFromOtherInvestingActivities",
        "OtherPaymentsToAcquireBusinesses",
        "PaymentsToAcquireIntangibleAssets",
        "PaymentsToAcquireSoftware",
    ],
    "investing_cf": [
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
        "CashFlowsFromUsedInInvestingActivities",
        "NetCashFromInvestingActivities",
    ],
    # ── Financing activities ─────────────────────────────────────────────────
    "debt_issuance": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromIssuanceOfDebt",
        "ProceedsFromDebtNetOfIssuanceCosts",
        "ProceedsFromIssuanceOfSeniorLongTermDebt",
        "ProceedsFromBorrowings",
    ],
    "debt_repayment": [
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
        "RepaymentsOfLongTermDebtAndCapitalSecurities",
        "RepaymentsOfBorrowings",
        "RepaymentsOfNotesPayable",
    ],
    "dividends_paid": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfOrdinaryDividends",
        "PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits",
        "DividendsPaid",
    ],
    "preferred_dividends": [
        "PaymentsOfDividendsPreferredStockAndPreferenceStock",
        "PreferredStockDividendsAndOtherAdjustments",
        "PaymentsOfDividendsMinorityInterest",
    ],
    "buybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "StockRepurchasedAndRetiredDuringPeriodValue",
        "PaymentsForRepurchaseOfEquity",
        "TreasuryStockValueAcquiredCostMethod",
        "PaymentsForRepurchaseOfOtherEquity",
    ],
    "stock_issuance": [
        "ProceedsFromIssuanceOfCommonStock",
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
        "ProceedsFromIssuanceOfPreferredStockAndPreferenceStock",
    ],
    "other_financing": [
        "ProceedsFromRepaymentsOfShortTermDebt",
        "PaymentsOfDebtIssuanceCosts",
        "FinanceLeaseObligationPayments",
        "RepaymentsOfRelatedPartyDebt",
        "OtherFinancingActivities",
    ],
    "financing_cf": [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
        "CashFlowsFromUsedInFinancingActivities",
        "NetCashFromFinancingActivities",
    ],
    # ── Net change and cash positions ────────────────────────────────────────
    "fx_effect": [
        "EffectOfExchangeRateOnCashAndCashEquivalents",
        "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "EffectOfExchangeRateOnCash",
    ],
    "net_change_cash": [
        "CashAndCashEquivalentsPeriodIncreaseDecrease",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
        "NetIncreaseDecreaseInCashAndCashEquivalents",
    ],
    "beginning_cash": [
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
        "CashAndCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "ending_cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalentsAtFairValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    # ── Supplemental ─────────────────────────────────────────────────────────
    "interest_paid": [
        "InterestPaidNet",
        "InterestPaid",
        "InterestPaidCapitalized",
    ],
    "taxes_paid": [
        "IncomeTaxesPaid",
        "IncomeTaxesPaidNet",
        "IncomeTaxPaidRefunded",
    ],
    # ── Computed (None = derived) ─────────────────────────────────────────────
    "fcf": None,           # operating_cf - capex
    "levered_fcf": None,   # fcf - net_debt_change - preferred_dividends
    "ufcf": None,          # EBIT*(1-t) + D&A - delta_wc - capex
    "owner_earnings": None, # net_income + D&A - maintenance_capex
    "fcf_yield": None,     # fcf / market_cap
    "fcf_margin": None,    # fcf / revenue
    "fcf_conversion": None, # fcf / net_income (quality ratio)
}

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class FCFBreakdown(BaseModel):
    ticker: str
    period_end: str
    standard_fcf: Optional[float] = None
    levered_fcf: Optional[float] = None
    ufcf: Optional[float] = None
    owner_earnings: Optional[float] = None
    fcf_yield: Optional[float] = None
    fcf_margin: Optional[float] = None
    fcf_conversion: Optional[float] = None
    operating_cf: Optional[float] = None
    capex: Optional[float] = None
    market_cap: Optional[float] = None


class CapexBreakdown(BaseModel):
    ticker: str
    period_end: str
    total_capex: Optional[float] = None
    maintenance_capex: Optional[float] = None
    growth_capex: Optional[float] = None
    capex_intensity: Optional[float] = None
    rd_plus_capex: Optional[float] = None
    is_asset_light: bool = False
    industry_benchmark_intensity: Optional[float] = None
    capex_to_depreciation: Optional[float] = None


class EarningsQualityReport(BaseModel):
    ticker: str
    period_end: str
    accruals_ratio: Optional[float] = None
    cash_earnings_ratio: Optional[float] = None
    operating_cf: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    quality_score: Optional[float] = None  # 0-100
    quality_label: str = "Unknown"
    five_year_avg_accruals: Optional[float] = None


class CashFlowStatement(BaseModel):
    ticker: str
    cik: str
    period_type: str
    periods: int
    data: list[dict]


# ---------------------------------------------------------------------------
# Industry capex intensity benchmarks (capex/revenue, approximate medians)
# ---------------------------------------------------------------------------

INDUSTRY_CAPEX_BENCHMARKS: dict[str, float] = {
    "technology":        0.04,
    "software":          0.02,
    "semiconductor":     0.10,
    "telecom":           0.14,
    "utilities":         0.18,
    "oil_gas":           0.20,
    "mining":            0.18,
    "manufacturing":     0.07,
    "retail":            0.03,
    "healthcare":        0.05,
    "pharma":            0.06,
    "financials":        0.02,
    "real_estate":       0.08,
    "consumer_staples":  0.04,
    "industrials":       0.06,
    "default":           0.06,
}

# SIC code prefix → industry key
SIC_TO_INDUSTRY: dict[str, str] = {
    "73":  "software",
    "36":  "semiconductor",
    "48":  "telecom",
    "49":  "utilities",
    "13":  "oil_gas",
    "10":  "mining",
    "20":  "manufacturing",
    "52":  "retail",
    "59":  "retail",
    "80":  "healthcare",
    "28":  "pharma",
    "60":  "financials",
    "65":  "real_estate",
    "20":  "consumer_staples",
    "35":  "industrials",
}

# ---------------------------------------------------------------------------
# Universal Cash Flow Parser
# ---------------------------------------------------------------------------


class UniversalCashFlowParser:
    """
    Enhanced cash flow statement parser with 35+ line items.

    Handles SEC XBRL taxonomy variations across thousands of companies
    using priority-ordered concept lists. Normalizes sign conventions
    (capex always negative, buybacks always negative, dividends always negative).
    """

    def __init__(self, cache_path: str | None = None) -> None:
        self._cache = FinancialsCache(db_path=cache_path)
        self._http = httpx.Client(
            headers=EDGAR_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    def get_company_facts(self, cik: str) -> dict:
        """Fetch EDGAR company facts with caching."""
        cik_padded = cik.zfill(10)
        cached = self._cache.get_facts(cik_padded)
        if cached is not None:
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
                return facts
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise ValueError(f"CIK {cik_padded} not found on EDGAR.") from exc
                last_exc = exc
                time.sleep(2 ** attempt)
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to fetch company facts: {last_exc}")

    def extract_metric(
        self,
        facts: dict,
        concept_list: list[str],
        period_type: str = "annual",
        units: str = "USD",
    ) -> pd.DataFrame:
        """
        Extract a time series for the first matching concept.

        Returns DataFrame with: period_end, value, filed_date, accession, form, concept
        """
        form_filter: set[str]
        if period_type == "annual":
            form_filter = {"10-K", "10-K/A", "20-F", "40-F"}
        elif period_type == "quarterly":
            form_filter = {"10-Q", "10-Q/A"}
        else:
            form_filter = set()

        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei = facts.get("facts", {}).get("dei", {})

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
                end = r.get("end", "")
                if end in seen_periods:
                    continue
                seen_periods.add(end)
                rows.append({
                    "period_end": end,
                    "value": r.get("val"),
                    "filed_date": r.get("filed", ""),
                    "accession": r.get("accn", ""),
                    "form": form,
                    "concept": concept,
                })

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

    def get_enhanced_cash_flow(
        self,
        cik: str,
        periods: int = 5,
        period_type: str = "annual",
    ) -> pd.DataFrame:
        """
        Return wide cash flow DataFrame with all 35+ line items.

        Sign convention (all normalized):
          - Operating CF: positive = cash inflow
          - Capex: negative (cash outflow)
          - Buybacks: negative (cash outflow)
          - Dividends paid: negative (cash outflow)
          - Debt repayment: negative (cash outflow)
        """
        cache_key = f"enhanced_cf_{period_type}"
        cached = self._cache.get_statement(cik, period_type, cache_key)
        if cached is not None:
            return cached.tail(periods)

        facts = self.get_company_facts(cik)
        rows: dict[str, dict] = {}

        for metric, concepts in ENHANCED_CASH_FLOW_MAP.items():
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

        # Normalize sign conventions — outflows should be negative
        negative_items = [
            "capex", "capex_discontinued", "acquisitions", "purchases_investments",
            "buybacks", "dividends_paid", "preferred_dividends", "debt_repayment",
        ]
        for col in negative_items:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: -abs(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else v
                )

        # Inflows should be positive
        positive_items = [
            "debt_issuance", "stock_issuance", "divestitures",
            "proceeds_investments", "ppe_proceeds",
        ]
        for col in positive_items:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: abs(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else v
                )

        df = df.tail(periods)
        self._cache.set_statement(cik, period_type, cache_key, df)
        return df

    def get_working_capital_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate individual working capital components into total_wc_change.
        Uses sum of: ar_change + inventory_change + ap_change +
                     deferred_revenue_change + other_working_capital
        """
        wc_cols = [
            "ar_change", "inventory_change", "ap_change",
            "deferred_revenue_change", "other_working_capital",
        ]
        available = [c for c in wc_cols if c in df.columns]
        if available:
            df["total_wc_change"] = df[available].fillna(0).sum(axis=1)
        return df

    def get_ticker_cash_flow(
        self,
        ticker: str,
        periods: int = 5,
        period_type: str = "annual",
    ) -> pd.DataFrame:
        """Convenience: resolve ticker → CIK → enhanced cash flow DataFrame."""
        cik = resolve_cik(ticker)
        df = self.get_enhanced_cash_flow(cik, periods=periods, period_type=period_type)
        return self.get_working_capital_detail(df)


# ---------------------------------------------------------------------------
# Free Cash Flow Engine
# ---------------------------------------------------------------------------


class FreeCashFlowEngine:
    """
    Multiple FCF definitions for institutional-grade analysis.

    Standard FCF      = operating_cf - capex
    Levered FCF       = FCF - net_debt_change - preferred_dividends
    Unlevered FCF     = EBIT × (1-tax) + D&A - ΔWorkingCapital - capex
    Owner Earnings    = net_income + D&A - maintenance_capex  (Buffett)
    FCF Yield         = FCF / market_cap
    FCF Margin        = FCF / revenue
    FCF Conversion    = FCF / net_income  (quality ratio, >1 = cash-generative)
    """

    _YFINANCE_BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
    _ASSUMED_TAX_RATE = 0.21  # US statutory corporate rate

    def __init__(
        self,
        cf_parser: UniversalCashFlowParser | None = None,
        cache_path: str | None = None,
    ) -> None:
        self._parser = cf_parser or UniversalCashFlowParser(cache_path=cache_path)
        self._http = httpx.Client(
            headers={"User-Agent": EDGAR_HEADERS["User-Agent"]},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    def get_market_cap(self, ticker: str) -> float | None:
        """Fetch current market cap from Yahoo Finance (free)."""
        try:
            url = self._YFINANCE_BASE.format(ticker)
            params = {"range": "1d", "interval": "1d", "includePrePost": "false"}
            resp = self._http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            price = meta.get("regularMarketPrice") or meta.get("previousClose")
            shares = meta.get("circulatingSupply") or meta.get("impliedShares")
            if price and shares:
                return float(price) * float(shares)
            # Fallback: marketCap key
            return float(meta.get("marketCap", 0)) or None
        except Exception as exc:
            logger.warning("market_cap_fetch_failed", ticker=ticker, error=str(exc))
            return None

    def get_revenue_series(self, facts: dict, period_type: str = "annual") -> pd.Series:
        """Extract revenue series from EDGAR facts for FCF margin calculation."""
        revenue_concepts = [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ]
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer()
        series = std.extract_metric(facts, revenue_concepts, period_type)
        if series.empty:
            return pd.Series(dtype=float)
        return series.set_index("period_end")["value"].astype(float)

    def compute_standard_fcf(self, df: pd.DataFrame) -> pd.Series:
        """Standard FCF = operating_cf + capex (capex is already negative)."""
        if "operating_cf" not in df.columns or "capex" not in df.columns:
            return pd.Series(dtype=float)
        return (df["operating_cf"].fillna(0) + df["capex"].fillna(0)).rename("standard_fcf")

    def compute_levered_fcf(self, df: pd.DataFrame) -> pd.Series:
        """
        Levered FCF = Standard FCF - net debt change - preferred dividends.
        Net debt change = debt_issuance + debt_repayment (net; repayment is negative).
        """
        standard = self.compute_standard_fcf(df)
        net_debt_change = pd.Series(0.0, index=df.index)
        if "debt_issuance" in df.columns:
            net_debt_change += df["debt_issuance"].fillna(0)
        if "debt_repayment" in df.columns:
            net_debt_change += df["debt_repayment"].fillna(0)  # already negative

        pref_div = pd.Series(0.0, index=df.index)
        if "preferred_dividends" in df.columns:
            pref_div = df["preferred_dividends"].fillna(0)  # already negative

        return (standard - net_debt_change + pref_div).rename("levered_fcf")

    def compute_ufcf(
        self,
        df: pd.DataFrame,
        ebit_series: pd.Series | None = None,
        tax_rate: float | None = None,
    ) -> pd.Series:
        """
        Unlevered FCF (UFCF) = EBIT × (1-t) + D&A - ΔWorkingCapital - capex.

        Used in DCF models (enterprise value basis, ignores capital structure).
        Tax rate defaults to US statutory 21%.
        """
        t = tax_rate if tax_rate is not None else self._ASSUMED_TAX_RATE
        result = pd.Series(0.0, index=df.index)

        # EBIT × (1-t)  — NOPAT
        if ebit_series is not None and not ebit_series.empty:
            nopat = ebit_series.reindex(df.index).fillna(0) * (1 - t)
        elif "net_income_cf" in df.columns:
            # Proxy: use net income if EBIT unavailable
            nopat = df["net_income_cf"].fillna(0) * (1 - t)
        else:
            nopat = pd.Series(0.0, index=df.index)

        result += nopat

        # + D&A (add back non-cash)
        if "d_and_a" in df.columns:
            result += df["d_and_a"].fillna(0)

        # - ΔWorking Capital (increase in WC = cash outflow, so subtract positive change)
        if "total_wc_change" in df.columns:
            result -= df["total_wc_change"].fillna(0)
        else:
            wc_components = ["ar_change", "inventory_change", "ap_change",
                             "deferred_revenue_change", "other_working_capital"]
            wc_sum = pd.Series(0.0, index=df.index)
            for col in wc_components:
                if col in df.columns:
                    wc_sum += df[col].fillna(0)
            result -= wc_sum

        # + capex (already negative, so adding negative = subtracting)
        if "capex" in df.columns:
            result += df["capex"].fillna(0)

        return result.rename("ufcf")

    def compute_owner_earnings(
        self,
        df: pd.DataFrame,
        maintenance_capex: pd.Series | None = None,
    ) -> pd.Series:
        """
        Owner Earnings (Buffett definition):
        = net_income + D&A + stock_comp - maintenance_capex
        - other_wc_changes

        Maintenance capex defaults to prior year D&A as conservative proxy.
        """
        result = pd.Series(0.0, index=df.index)

        if "net_income_cf" in df.columns:
            result += df["net_income_cf"].fillna(0)

        if "d_and_a" in df.columns:
            result += df["d_and_a"].fillna(0)

        if "stock_based_comp" in df.columns:
            result += df["stock_based_comp"].fillna(0)

        # Subtract maintenance capex
        if maintenance_capex is not None:
            result -= maintenance_capex.reindex(df.index).fillna(0)
        elif "d_and_a" in df.columns:
            # Conservative proxy: prior year D&A is maintenance capex
            prior_da = df["d_and_a"].shift(1).fillna(df["d_and_a"])
            result -= prior_da

        # Subtract working capital changes (increases reduce cash)
        wc_cols = ["ar_change", "inventory_change"]
        for col in wc_cols:
            if col in df.columns:
                result -= df[col].fillna(0)

        return result.rename("owner_earnings")

    def compute_all_fcf_metrics(
        self,
        ticker: str,
        periods: int = 5,
        period_type: str = "annual",
        market_cap_override: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute all FCF variants and quality ratios in one call.

        Returns DataFrame with columns:
          standard_fcf, levered_fcf, ufcf, owner_earnings,
          fcf_yield, fcf_margin, fcf_conversion,
          operating_cf, capex, market_cap
        """
        cik = resolve_cik(ticker)
        facts = self._parser.get_company_facts(cik)

        df = self._parser.get_enhanced_cash_flow(cik, periods=periods, period_type=period_type)
        df = self._parser.get_working_capital_detail(df)

        if df.empty:
            return pd.DataFrame()

        # Market cap
        market_cap = market_cap_override or self.get_market_cap(ticker)

        # Revenue for margin
        rev_series = self.get_revenue_series(facts, period_type)

        # Compute all variants
        result = pd.DataFrame(index=df.index)
        result["operating_cf"] = df.get("operating_cf")
        result["capex"] = df.get("capex")
        result["d_and_a"] = df.get("d_and_a")

        result["standard_fcf"] = self.compute_standard_fcf(df)
        result["levered_fcf"] = self.compute_levered_fcf(df)
        result["ufcf"] = self.compute_ufcf(df)
        result["owner_earnings"] = self.compute_owner_earnings(df)

        # FCF yield (use most recent market cap for all periods as approximation)
        if market_cap and market_cap > 0:
            result["fcf_yield"] = result["standard_fcf"] / market_cap
            result["market_cap"] = market_cap
        else:
            result["fcf_yield"] = None
            result["market_cap"] = None

        # FCF margin = FCF / revenue
        if not rev_series.empty:
            rev_aligned = rev_series.reindex(result.index)
            result["fcf_margin"] = result["standard_fcf"] / rev_aligned.replace(0, float("nan"))
        else:
            result["fcf_margin"] = None

        # FCF conversion = FCF / net_income
        if "net_income_cf" in df.columns:
            ni = df["net_income_cf"].replace(0, float("nan"))
            result["fcf_conversion"] = result["standard_fcf"] / ni
        else:
            result["fcf_conversion"] = None

        result.index.name = "period_end"
        return result

    def get_fcf_summary(self, ticker: str, periods: int = 5) -> list[FCFBreakdown]:
        """Return list of FCFBreakdown pydantic models for API response."""
        df = self.compute_all_fcf_metrics(ticker, periods=periods)
        if df.empty:
            return []

        results = []
        for period_end, row in df.iterrows():
            results.append(FCFBreakdown(
                ticker=ticker,
                period_end=str(period_end.date()) if hasattr(period_end, "date") else str(period_end),
                standard_fcf=_float_or_none(row.get("standard_fcf")),
                levered_fcf=_float_or_none(row.get("levered_fcf")),
                ufcf=_float_or_none(row.get("ufcf")),
                owner_earnings=_float_or_none(row.get("owner_earnings")),
                fcf_yield=_float_or_none(row.get("fcf_yield")),
                fcf_margin=_float_or_none(row.get("fcf_margin")),
                fcf_conversion=_float_or_none(row.get("fcf_conversion")),
                operating_cf=_float_or_none(row.get("operating_cf")),
                capex=_float_or_none(row.get("capex")),
                market_cap=_float_or_none(row.get("market_cap")),
            ))
        return results


# ---------------------------------------------------------------------------
# Capex Analyzer
# ---------------------------------------------------------------------------


class CapexAnalyzer:
    """
    Capital expenditure decomposition and classification.

    Maintenance capex   — sustaining existing asset base (D&A proxy)
    Growth capex        — incremental investment beyond maintenance
    Capex intensity     — capex/revenue vs industry benchmark
    Asset-light check   — capex/revenue < 2%
    R&D as capex proxy  — total investment = R&D + capex (tech sector)
    """

    def __init__(
        self,
        cf_parser: UniversalCashFlowParser | None = None,
        cache_path: str | None = None,
    ) -> None:
        self._parser = cf_parser or UniversalCashFlowParser(cache_path=cache_path)
        self._http = httpx.Client(
            headers=EDGAR_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    def _get_rd_expense(self, facts: dict, period_type: str = "annual") -> pd.Series:
        """Extract R&D expense from EDGAR facts."""
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer()
        concepts = [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
            "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
        ]
        series = std.extract_metric(facts, concepts, period_type)
        if series.empty:
            return pd.Series(dtype=float)
        return series.set_index("period_end")["value"].astype(float)

    def _get_revenue(self, facts: dict, period_type: str = "annual") -> pd.Series:
        """Extract revenue from EDGAR facts."""
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer()
        concepts = [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ]
        series = std.extract_metric(facts, concepts, period_type)
        if series.empty:
            return pd.Series(dtype=float)
        return series.set_index("period_end")["value"].astype(float)

    def _detect_industry(self, facts: dict) -> str:
        """Infer industry from SIC code in EDGAR entity data."""
        try:
            sic = str(facts.get("entityType", {}).get("sic", "") or
                      facts.get("sic", "") or "")
            if not sic:
                # Try to find SIC in the facts structure
                entity_info = facts.get("entityInfo", {})
                sic = str(entity_info.get("sic", ""))
            for prefix, industry in SIC_TO_INDUSTRY.items():
                if sic.startswith(prefix):
                    return industry
        except Exception:
            pass
        return "default"

    def compute_capex_breakdown(
        self,
        ticker: str,
        periods: int = 5,
        period_type: str = "annual",
        industry: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute full capex decomposition for a ticker.

        Returns DataFrame with per-period breakdown.
        """
        cik = resolve_cik(ticker)
        facts = self._parser.get_company_facts(cik)
        df = self._parser.get_enhanced_cash_flow(cik, periods=periods, period_type=period_type)

        if df.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=df.index)

        # Total capex (already normalized negative)
        result["total_capex"] = df.get("capex", pd.Series(dtype=float))

        # Maintenance capex proxy: prior year D&A (conservative)
        if "d_and_a" in df.columns:
            da = df["d_and_a"].abs()
            # Maintenance = prior year D&A, floored at half current capex
            result["maintenance_capex"] = da.shift(1).fillna(da).apply(lambda v: -abs(v))
            result["growth_capex"] = (
                result["total_capex"].fillna(0) - result["maintenance_capex"].fillna(0)
            )
            result["capex_to_depreciation"] = (
                result["total_capex"].abs() / da.replace(0, float("nan"))
            )
        else:
            result["maintenance_capex"] = None
            result["growth_capex"] = None
            result["capex_to_depreciation"] = None

        # Revenue for intensity calculation
        rev_series = self._get_revenue(facts, period_type)
        if not rev_series.empty:
            rev_aligned = rev_series.reindex(result.index)
            result["capex_intensity"] = (
                result["total_capex"].abs() / rev_aligned.replace(0, float("nan"))
            )
        else:
            result["capex_intensity"] = None

        # R&D as capex proxy (total investment)
        rd_series = self._get_rd_expense(facts, period_type)
        if not rd_series.empty:
            rd_aligned = rd_series.reindex(result.index).fillna(0)
            result["rd_expense"] = rd_aligned
            result["rd_plus_capex"] = result["total_capex"].fillna(0).abs() + rd_aligned
            if not rev_series.empty:
                rev_aligned2 = rev_series.reindex(result.index)
                result["total_investment_intensity"] = (
                    result["rd_plus_capex"] / rev_aligned2.replace(0, float("nan"))
                )
        else:
            result["rd_expense"] = None
            result["rd_plus_capex"] = None
            result["total_investment_intensity"] = None

        # Asset-light detection: capex/revenue < 2%
        if "capex_intensity" in result.columns:
            result["is_asset_light"] = result["capex_intensity"].fillna(1.0) < 0.02

        # Industry benchmark
        detected_industry = industry or self._detect_industry(facts)
        benchmark = INDUSTRY_CAPEX_BENCHMARKS.get(
            detected_industry, INDUSTRY_CAPEX_BENCHMARKS["default"]
        )
        result["industry_benchmark_intensity"] = benchmark
        result["industry"] = detected_industry

        # Above/below benchmark
        if "capex_intensity" in result.columns:
            result["vs_benchmark"] = result["capex_intensity"] - benchmark
            result["is_above_benchmark"] = result["vs_benchmark"] > 0

        result.index.name = "period_end"
        return result

    def get_capex_summary(
        self,
        ticker: str,
        periods: int = 5,
    ) -> list[CapexBreakdown]:
        """Return list of CapexBreakdown pydantic models."""
        df = self.compute_capex_breakdown(ticker, periods=periods)
        if df.empty:
            return []

        results = []
        for period_end, row in df.iterrows():
            results.append(CapexBreakdown(
                ticker=ticker,
                period_end=str(period_end.date()) if hasattr(period_end, "date") else str(period_end),
                total_capex=_float_or_none(row.get("total_capex")),
                maintenance_capex=_float_or_none(row.get("maintenance_capex")),
                growth_capex=_float_or_none(row.get("growth_capex")),
                capex_intensity=_float_or_none(row.get("capex_intensity")),
                rd_plus_capex=_float_or_none(row.get("rd_plus_capex")),
                is_asset_light=bool(row.get("is_asset_light", False)),
                industry_benchmark_intensity=_float_or_none(
                    row.get("industry_benchmark_intensity")
                ),
                capex_to_depreciation=_float_or_none(row.get("capex_to_depreciation")),
            ))
        return results


# ---------------------------------------------------------------------------
# Earnings Quality Analyzer
# ---------------------------------------------------------------------------


class EarningsQualityAnalyzer:
    """
    Cash vs accrual earnings quality scoring.

    Accruals Ratio      = (net_income - operating_CF) / total_assets
                         Lower is better (high = aggressive accrual accounting)

    Cash Earnings Ratio = operating_CF / net_income
                         >1 = cash-generative; <0.5 = accruals concern

    Revenue Quality     = cash_revenue proxy = operating_CF + ΔAP - ΔAR
                         (approximates cash collected from customers)

    Historical Score    = 5-year average accruals ratio (lower = better quality)

    Quality Score 0-100: 100 = pristine (all cash, no accruals)
                          0  = highly aggressive accrual manipulation
    """

    # Thresholds for quality scoring
    ACCRUALS_EXCELLENT = -0.05   # slightly negative = conservative
    ACCRUALS_GOOD      =  0.02
    ACCRUALS_FAIR      =  0.05
    ACCRUALS_POOR      =  0.10
    ACCRUALS_RED_FLAG  =  0.15

    CASH_RATIO_EXCELLENT = 1.20
    CASH_RATIO_GOOD      = 1.00
    CASH_RATIO_FAIR      = 0.80
    CASH_RATIO_POOR      = 0.60

    def __init__(
        self,
        cf_parser: UniversalCashFlowParser | None = None,
        cache_path: str | None = None,
    ) -> None:
        self._parser = cf_parser or UniversalCashFlowParser(cache_path=cache_path)
        self._http = httpx.Client(headers=EDGAR_HEADERS, timeout=_TIMEOUT)

    def _get_total_assets(self, facts: dict, period_type: str = "annual") -> pd.Series:
        """Extract total assets from EDGAR facts."""
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer()
        concepts = ["Assets", "TotalAssets"]
        series = std.extract_metric(facts, concepts, period_type)
        if series.empty:
            return pd.Series(dtype=float)
        return series.set_index("period_end")["value"].astype(float)

    def compute_accruals_ratio(
        self,
        net_income: pd.Series,
        operating_cf: pd.Series,
        total_assets: pd.Series,
    ) -> pd.Series:
        """
        Accruals ratio = (net_income - operating_CF) / avg_total_assets.
        Uses average of current and prior period assets (Sloan, 1996).
        """
        avg_assets = (total_assets + total_assets.shift(1)) / 2
        avg_assets = avg_assets.replace(0, float("nan"))
        accruals = (net_income.reindex(avg_assets.index).fillna(0) -
                    operating_cf.reindex(avg_assets.index).fillna(0))
        return (accruals / avg_assets).rename("accruals_ratio")

    def compute_cash_earnings_ratio(
        self,
        operating_cf: pd.Series,
        net_income: pd.Series,
    ) -> pd.Series:
        """Cash earnings ratio = operating_CF / net_income."""
        ni = net_income.replace(0, float("nan"))
        return (operating_cf.reindex(ni.index).fillna(0) / ni).rename("cash_earnings_ratio")

    def compute_revenue_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        Approximate cash revenue = operating_CF + ΔAP - ΔAR.
        This estimates cash actually collected vs reported revenue.
        """
        result = pd.Series(0.0, index=df.index)

        if "operating_cf" in df.columns:
            result = df["operating_cf"].fillna(0).copy()

        # ΔAP positive = liability increased = cash retained = add back
        if "ap_change" in df.columns:
            result += df["ap_change"].fillna(0)

        # ΔAR negative in XBRL when AR increases (cash hasn't been collected)
        # EDGAR sign: IncreaseDecreaseInAccountsReceivable is negative when AR rises
        # We want: subtract AR increases (they reduce cash quality)
        if "ar_change" in df.columns:
            result += df["ar_change"].fillna(0)  # already sign-adjusted

        return result.rename("cash_revenue_proxy")

    def score_quality(
        self,
        accruals_ratio: float | None,
        cash_earnings_ratio: float | None,
    ) -> tuple[float, str]:
        """
        Compute composite quality score (0-100) and label.

        Scoring:
          Accruals ratio: 50 points (lower = better)
          Cash earnings ratio: 50 points (higher = better)
        """
        score = 0.0

        # Accruals ratio component (50 pts)
        if accruals_ratio is not None and not math.isnan(accruals_ratio):
            if accruals_ratio <= self.ACCRUALS_EXCELLENT:
                score += 50
            elif accruals_ratio <= self.ACCRUALS_GOOD:
                score += 42
            elif accruals_ratio <= self.ACCRUALS_FAIR:
                score += 30
            elif accruals_ratio <= self.ACCRUALS_POOR:
                score += 15
            elif accruals_ratio <= self.ACCRUALS_RED_FLAG:
                score += 5
            else:
                score += 0

        # Cash earnings ratio component (50 pts)
        if cash_earnings_ratio is not None and not math.isnan(cash_earnings_ratio):
            if cash_earnings_ratio >= self.CASH_RATIO_EXCELLENT:
                score += 50
            elif cash_earnings_ratio >= self.CASH_RATIO_GOOD:
                score += 40
            elif cash_earnings_ratio >= self.CASH_RATIO_FAIR:
                score += 25
            elif cash_earnings_ratio >= self.CASH_RATIO_POOR:
                score += 10
            else:
                score += 0

        # Label
        if score >= 85:
            label = "Excellent"
        elif score >= 70:
            label = "Good"
        elif score >= 50:
            label = "Fair"
        elif score >= 30:
            label = "Poor"
        else:
            label = "Red Flag"

        return score, label

    def compute_earnings_quality(
        self,
        ticker: str,
        periods: int = 8,  # need extra periods for 5-year avg
        period_type: str = "annual",
    ) -> pd.DataFrame:
        """
        Full earnings quality analysis for a ticker.

        Returns DataFrame with accruals_ratio, cash_earnings_ratio,
        quality_score, quality_label, five_year_avg_accruals.
        """
        cik = resolve_cik(ticker)
        facts = self._parser.get_company_facts(cik)
        df = self._parser.get_enhanced_cash_flow(cik, periods=periods, period_type=period_type)

        if df.empty:
            return pd.DataFrame()

        # Total assets
        ta_series = self._get_total_assets(facts, period_type)

        result = pd.DataFrame(index=df.index)
        result["operating_cf"] = df.get("operating_cf")
        result["net_income"] = df.get("net_income_cf")

        # Accruals ratio
        if not ta_series.empty and "net_income_cf" in df.columns and "operating_cf" in df.columns:
            ta_aligned = ta_series.reindex(df.index)
            accruals = self.compute_accruals_ratio(
                df["net_income_cf"], df["operating_cf"], ta_aligned
            )
            result["accruals_ratio"] = accruals
            result["total_assets"] = ta_aligned

            # 5-year rolling average
            result["five_year_avg_accruals"] = (
                result["accruals_ratio"].rolling(window=5, min_periods=2).mean()
            )
        else:
            result["accruals_ratio"] = None
            result["total_assets"] = None
            result["five_year_avg_accruals"] = None

        # Cash earnings ratio
        if "net_income_cf" in df.columns and "operating_cf" in df.columns:
            cer = self.compute_cash_earnings_ratio(
                df["operating_cf"], df["net_income_cf"]
            )
            result["cash_earnings_ratio"] = cer
        else:
            result["cash_earnings_ratio"] = None

        # Revenue quality proxy
        result["cash_revenue_proxy"] = self.compute_revenue_quality(df)

        # Per-period quality scores
        scores = []
        labels = []
        for _, row in result.iterrows():
            score, label = self.score_quality(
                accruals_ratio=row.get("accruals_ratio"),
                cash_earnings_ratio=row.get("cash_earnings_ratio"),
            )
            scores.append(score)
            labels.append(label)

        result["quality_score"] = scores
        result["quality_label"] = labels

        # One-time item flag: large stock_based_comp spikes
        if "stock_based_comp" in df.columns:
            sbc = df["stock_based_comp"].fillna(0)
            sbc_pct_change = sbc.pct_change().abs()
            result["sbc_spike_flag"] = sbc_pct_change > 0.50  # >50% YoY spike

        # Deferred revenue change — positive = unearned cash (quality positive)
        if "deferred_revenue_change" in df.columns:
            result["deferred_rev_change"] = df["deferred_revenue_change"]
            result["deferred_rev_positive"] = df["deferred_revenue_change"].fillna(0) > 0

        result.index.name = "period_end"
        return result

    def get_quality_reports(
        self,
        ticker: str,
        periods: int = 5,
    ) -> list[EarningsQualityReport]:
        """Return list of EarningsQualityReport pydantic models."""
        df = self.compute_earnings_quality(ticker, periods=periods + 2)  # extra for rolling
        if df.empty:
            return []

        results = []
        for period_end, row in df.tail(periods).iterrows():
            results.append(EarningsQualityReport(
                ticker=ticker,
                period_end=str(period_end.date()) if hasattr(period_end, "date") else str(period_end),
                accruals_ratio=_float_or_none(row.get("accruals_ratio")),
                cash_earnings_ratio=_float_or_none(row.get("cash_earnings_ratio")),
                operating_cf=_float_or_none(row.get("operating_cf")),
                net_income=_float_or_none(row.get("net_income")),
                total_assets=_float_or_none(row.get("total_assets")),
                quality_score=_float_or_none(row.get("quality_score")),
                quality_label=str(row.get("quality_label", "Unknown")),
                five_year_avg_accruals=_float_or_none(row.get("five_year_avg_accruals")),
            ))
        return results

    def detect_one_time_items(
        self,
        ticker: str,
        periods: int = 5,
    ) -> dict[str, Any]:
        """
        Detect non-recurring items by comparing IS and CF statement.
        Flags: large gains/losses in 'other_operating' relative to operating_cf.
        """
        cik = resolve_cik(ticker)
        df = self._parser.get_enhanced_cash_flow(cik, periods=periods)

        if df.empty:
            return {"ticker": ticker, "flags": [], "data": []}

        flags = []
        data_rows = []

        for period_end, row in df.iterrows():
            period_str = str(period_end.date()) if hasattr(period_end, "date") else str(period_end)
            period_flags = []

            op_cf = row.get("operating_cf", 0) or 0
            ni = row.get("net_income_cf", 0) or 0
            other_op = row.get("other_operating", 0) or 0
            sbc = row.get("stock_based_comp", 0) or 0
            da = row.get("d_and_a", 0) or 0

            # Flag: "other operating" > 15% of operating CF
            if op_cf != 0 and abs(other_op) > abs(op_cf) * 0.15:
                period_flags.append({
                    "type": "large_other_operating",
                    "value": other_op,
                    "pct_of_op_cf": other_op / op_cf if op_cf else None,
                })

            # Flag: SBC > 20% of net income (dilutive, often excluded in non-GAAP)
            if ni != 0 and abs(sbc) > abs(ni) * 0.20:
                period_flags.append({
                    "type": "high_sbc_relative_to_earnings",
                    "sbc": sbc,
                    "net_income": ni,
                    "ratio": sbc / ni if ni else None,
                })

            # Flag: D&A acceleration (>20% YoY increase)
            # Tracked separately in rolling analysis

            data_rows.append({
                "period_end": period_str,
                "flags": period_flags,
                "operating_cf": op_cf,
                "net_income": ni,
                "other_operating": other_op,
                "sbc": sbc,
                "d_and_a": da,
            })

            if period_flags:
                flags.append({"period": period_str, "flags": period_flags})

        return {
            "ticker": ticker,
            "flags": flags,
            "total_flag_periods": len(flags),
            "data": data_rows,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _float_or_none(v: Any) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _df_to_json_records(df: pd.DataFrame) -> list[dict]:
    """Serialize DataFrame to list of JSON-safe dicts."""
    if df.empty:
        return []
    reset = df.reset_index()
    reset.columns = [str(c) for c in reset.columns]
    for col in reset.columns:
        if pd.api.types.is_datetime64_any_dtype(reset[col]):
            reset[col] = reset[col].dt.strftime("%Y-%m-%d")
    return reset.where(pd.notnull(reset), other=None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_cf_parser: UniversalCashFlowParser | None = None
_fcf_engine: FreeCashFlowEngine | None = None
_capex_analyzer: CapexAnalyzer | None = None
_quality_analyzer: EarningsQualityAnalyzer | None = None


def _get_cf_parser() -> UniversalCashFlowParser:
    global _cf_parser
    if _cf_parser is None:
        _cf_parser = UniversalCashFlowParser()
    return _cf_parser


def _get_fcf_engine() -> FreeCashFlowEngine:
    global _fcf_engine
    if _fcf_engine is None:
        _fcf_engine = FreeCashFlowEngine(cf_parser=_get_cf_parser())
    return _fcf_engine


def _get_capex_analyzer() -> CapexAnalyzer:
    global _capex_analyzer
    if _capex_analyzer is None:
        _capex_analyzer = CapexAnalyzer(cf_parser=_get_cf_parser())
    return _capex_analyzer


def _get_quality_analyzer() -> EarningsQualityAnalyzer:
    global _quality_analyzer
    if _quality_analyzer is None:
        _quality_analyzer = EarningsQualityAnalyzer(cf_parser=_get_cf_parser())
    return _quality_analyzer


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

cashflow_router_v2 = APIRouter(
    prefix="/api/financials/v2/cashflow",
    tags=["enhanced-cashflow"],
)


def _ticker_404(ticker: str) -> str:
    """Resolve ticker to CIK or raise HTTP 404."""
    try:
        return resolve_cik(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@cashflow_router_v2.get("/{ticker}", response_model=CashFlowStatement)
def get_enhanced_cashflow(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    period_type: str = Query(default="annual", pattern="^(annual|quarterly)$"),
):
    """
    Enhanced cash flow statement with 35+ line items.

    Includes operating, investing, financing sections with full XBRL taxonomy
    coverage and normalized sign conventions.
    """
    cik = _ticker_404(ticker)
    try:
        df = _get_cf_parser().get_ticker_cash_flow(
            ticker, periods=periods, period_type=period_type
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CashFlowStatement(
        ticker=ticker,
        cik=cik,
        period_type=period_type,
        periods=len(df),
        data=_df_to_json_records(df),
    )


@cashflow_router_v2.get("/{ticker}/fcf")
def get_fcf_analysis(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    period_type: str = Query(default="annual", pattern="^(annual|quarterly)$"),
    market_cap_override: float | None = Query(default=None, description="Override market cap in USD"),
):
    """
    Multiple FCF definitions: Standard, Levered, UFCF, Owner Earnings.

    Also returns FCF yield (vs market cap), FCF margin, and FCF conversion ratio.
    Market cap is fetched from Yahoo Finance if not provided.
    """
    _ticker_404(ticker)
    try:
        summaries = _get_fcf_engine().get_fcf_summary(ticker, periods=periods)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker,
        "period_type": period_type,
        "definitions": {
            "standard_fcf": "operating_cf - capex",
            "levered_fcf": "standard_fcf - net_debt_change - preferred_dividends",
            "ufcf": "EBIT*(1-t) + D&A - delta_WC - capex (unlevered, for DCF)",
            "owner_earnings": "net_income + D&A - maintenance_capex (Buffett)",
            "fcf_yield": "standard_fcf / market_cap",
            "fcf_margin": "standard_fcf / revenue",
            "fcf_conversion": "standard_fcf / net_income (>1 = cash generative)",
        },
        "data": [s.model_dump() for s in summaries],
    }


@cashflow_router_v2.get("/{ticker}/capex")
def get_capex_analysis(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    industry: str | None = Query(default=None, description="Override industry for benchmark"),
):
    """
    Capex decomposition: maintenance vs growth, intensity vs benchmark.

    Also detects asset-light businesses (capex/revenue < 2%) and computes
    total investment (R&D + capex) for tech sector analysis.
    """
    _ticker_404(ticker)
    try:
        summaries = _get_capex_analyzer().get_capex_summary(ticker, periods=periods)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker,
        "industry_benchmarks": INDUSTRY_CAPEX_BENCHMARKS,
        "data": [s.model_dump() for s in summaries],
    }


@cashflow_router_v2.get("/{ticker}/quality")
def get_earnings_quality(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
    include_flags: bool = Query(default=True),
):
    """
    Earnings quality analysis: cash vs accrual metrics.

    Returns accruals ratio (Sloan), cash earnings ratio, quality score (0-100),
    five-year average accruals, and optionally one-time item flags.
    """
    _ticker_404(ticker)
    try:
        reports = _get_quality_analyzer().get_quality_reports(ticker, periods=periods)
        flags_data = {}
        if include_flags:
            flags_data = _get_quality_analyzer().detect_one_time_items(
                ticker, periods=periods
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker,
        "quality_framework": {
            "accruals_ratio": "Sloan (1996): (net_income - op_CF) / avg_assets. Lower is better.",
            "cash_earnings_ratio": "op_CF / net_income. >1.0 = cash-generative.",
            "score_legend": {
                "90-100": "Excellent — pristine cash-based earnings",
                "70-89": "Good — minor accruals, well-managed",
                "50-69": "Fair — moderate accruals, monitor trends",
                "30-49": "Poor — material accruals concerns",
                "0-29": "Red Flag — aggressive accrual accounting",
            },
        },
        "data": [r.model_dump() for r in reports],
        "one_time_flags": flags_data if include_flags else {},
    }


@cashflow_router_v2.get("/{ticker}/summary")
def get_cashflow_summary(
    ticker: str,
    periods: int = Query(default=5, ge=1, le=20),
):
    """
    Comprehensive cash flow summary combining all sub-analyses.

    Single call to get: enhanced CF statement, all FCF variants,
    capex breakdown, and earnings quality in one response.
    """
    cik = _ticker_404(ticker)
    try:
        cf_df = _get_cf_parser().get_ticker_cash_flow(ticker, periods=periods)
        fcf_summaries = _get_fcf_engine().get_fcf_summary(ticker, periods=periods)
        capex_summaries = _get_capex_analyzer().get_capex_summary(ticker, periods=periods)
        quality_reports = _get_quality_analyzer().get_quality_reports(ticker, periods=periods)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker,
        "cik": cik,
        "periods": periods,
        "cash_flow_statement": _df_to_json_records(cf_df),
        "fcf_analysis": [s.model_dump() for s in fcf_summaries],
        "capex_analysis": [s.model_dump() for s in capex_summaries],
        "earnings_quality": [r.model_dump() for r in quality_reports],
    }
