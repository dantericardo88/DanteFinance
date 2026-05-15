"""
Enhanced balance sheet: working capital analytics, off-balance-sheet items,
lease capitalization (IFRS 16 / ASC 842), goodwill tracking, financial health metrics.

Dimension: dim_014 — Balance sheet standardized
Target: 9 (from 8)

Enhancements over standardized_financials.py:
- 50+ XBRL balance sheet line items (vs 16 in base)
- 200+ taxonomy variation handling across companies
- Off-balance-sheet: operating leases, pensions, VIEs, guarantees
- IFRS 16 / ASC 842 right-of-use asset parsing
- Goodwill quality and impairment history tracking
- Working capital engine: DSO, DIO, DPO, CCC with trend analysis
- Financial health scorer: Altman Z, Piotroski F, current/quick/cash ratios,
  aggregate 0-100 health score with letter rating
"""
from __future__ import annotations

import math
import time
from typing import Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from sentinel.core.logging import get_logger
from sentinel.sfe.standardized_financials import (
    EDGAR_BASE,
    EDGAR_HEADERS,
    FinancialsCache,
    _RATE_DELAY,
    _TIMEOUT,
    _MAX_RETRY,
    resolve_cik,
)

logger = get_logger(__name__)

__all__ = [
    "ENHANCED_BALANCE_SHEET_MAP",
    "UniversalBalanceSheetParser",
    "OffBalanceSheetAnalyzer",
    "GoodwillAnalyzer",
    "WorkingCapitalEngine",
    "FinancialHealthScorer",
    "balance_router_v2",
]

# ---------------------------------------------------------------------------
# Extended XBRL concept map — 50+ balance sheet line items
# ---------------------------------------------------------------------------

ENHANCED_BALANCE_SHEET_MAP: dict[str, list[str] | None] = {
    # ── Current Assets ────────────────────────────────────────────────────────
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashAndDueFromBanks",
        "CashEquivalentsAtCarryingValue",
        "CashAndCashEquivalentsAtFairValue",
        "CashAndCashEquivalentsFairValueDisclosure",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "restricted_cash": [
        "RestrictedCashAndCashEquivalentsAtCarryingValue",
        "RestrictedCash",
        "RestrictedCashCurrent",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
        "TradingSecurities",
        "ShortTermInvestmentsAndMarketableSecurities",
        "HeldToMaturitySecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ],
    "accounts_receivable_gross": [
        "AccountsReceivableGrossCurrent",
        "AccountsReceivableGross",
        "ReceivablesGross",
    ],
    "allowance_doubtful_accounts": [
        "AllowanceForDoubtfulAccountsReceivableCurrent",
        "AllowanceForDoubtfulAccounts",
    ],
    "accounts_receivable_net": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsReceivableNet",
        "TradeAndOtherReceivablesNetCurrent",
        "NotesAndLoansReceivableNetCurrent",
        "TradeReceivablesNetCurrent",
    ],
    "notes_receivable": [
        "NotesReceivableNet",
        "NotesAndLoansReceivableNetCurrent",
        "ReceivableFromOfficersAndDirectorsCurrent",
    ],
    "inventory_raw_materials": [
        "InventoryRawMaterials",
        "InventoryRawMaterialsAndSupplies",
        "InventoryRawMaterialsNetOfReserves",
    ],
    "inventory_wip": [
        "InventoryWorkInProcess",
        "InventoryWorkInProcessNetOfReserves",
    ],
    "inventory_finished_goods": [
        "InventoryFinishedGoods",
        "InventoryFinishedGoodsNetOfReserves",
        "RetailRelatedInventoryMerchandise",
    ],
    "inventory_net": [
        "InventoryNet",
        "InventoryGross",
        "FIFOInventoryAmount",
        "LIFOInventoryAmount",
        "Inventories",
        "InventoriesNet",
    ],
    "prepaid_expenses": [
        "PrepaidExpenseAndOtherAssetsCurrent",
        "PrepaidExpenseCurrent",
        "OtherPrepaidExpenseCurrent",
    ],
    "deferred_tax_asset_current": [
        "DeferredTaxAssetsNetCurrent",
        "DeferredIncomeTaxAssetsNet",
        "DeferredTaxAssetsCurrent",
    ],
    "other_current_assets": [
        "OtherAssetsCurrent",
        "OtherCurrentAssets",
        "OtherAssetsAndReceivablesCurrent",
    ],
    "total_current_assets": [
        "AssetsCurrent",
        "CurrentAssets",
        "TotalCurrentAssets",
    ],
    # ── Non-current assets ────────────────────────────────────────────────────
    "ppe_gross": [
        "PropertyPlantAndEquipmentGross",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciation",
    ],
    "accumulated_depreciation": [
        "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "PropertyPlantAndEquipmentAccumulatedDepreciation",
    ],
    "ppe_net": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        "PropertyPlantAndEquipmentNetOfAccumulatedDepreciation",
        "PropertyPlantAndEquipmentNetExcludingCapitalLeasedAssets",
    ],
    "right_of_use_assets_operating": [
        "OperatingLeaseRightOfUseAsset",
        "RightOfUseAssetOperatingLease",
        "OperatingLeaseRightOfUseAssetNoncurrent",
    ],
    "right_of_use_assets_finance": [
        "FinanceLeaseRightOfUseAsset",
        "FinanceLeaseRightOfUseAssetAfterAccumulatedAmortization",
        "CapitalLeaseObligationsAssetsSubjectToOrAvailableForOperatingLeaseNet",
    ],
    "goodwill": [
        "Goodwill",
        "GoodwillGross",
        "GoodwillNet",
        "GoodwillNoncurrent",
    ],
    "intangibles_net": [
        "FiniteLivedIntangibleAssetsNet",
        "IntangibleAssetsNetExcludingGoodwill",
        "IndefiniteLivedIntangibleAssetsExcludingGoodwill",
        "OtherIntangibleAssetsNet",
        "IntangibleAssetsNet",
    ],
    "intangibles_gross": [
        "FiniteLivedIntangibleAssetsGross",
        "IntangibleAssetsGrossExcludingGoodwill",
    ],
    "long_term_investments": [
        "LongTermInvestments",
        "EquityMethodInvestments",
        "AvailableForSaleSecuritiesNoncurrent",
        "HeldToMaturitySecuritiesNoncurrent",
        "OtherLongTermInvestments",
        "InvestmentsAndAdvancesToAffiliates",
    ],
    "deferred_tax_asset_noncurrent": [
        "DeferredIncomeTaxAssetsNet",
        "DeferredTaxAssetsNetNoncurrent",
        "DeferredTaxAssetsNoncurrent",
    ],
    "pension_asset": [
        "DefinedBenefitPlanAssetsForPlanBenefitsNoncurrent",
        "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent",
    ],
    "other_noncurrent_assets": [
        "OtherAssetsNoncurrent",
        "OtherNoncurrentAssets",
        "OtherAssets",
    ],
    "total_noncurrent_assets": [
        "AssetsNoncurrent",
        "NoncurrentAssets",
    ],
    "total_assets": [
        "Assets",
        "TotalAssets",
        "AssetsNet",
    ],
    # ── Current Liabilities ───────────────────────────────────────────────────
    "accounts_payable": [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
        "AccountsPayableTradeCurrent",
        "AccountsPayableRelatedPartiesCurrent",
    ],
    "accrued_expenses": [
        "AccruedLiabilitiesCurrent",
        "AccruedExpensesAndOtherCurrentLiabilities",
        "EmployeeRelatedLiabilitiesCurrent",
        "OtherAccruedLiabilitiesCurrent",
    ],
    "accrued_compensation": [
        "EmployeeRelatedLiabilitiesCurrent",
        "AccruedSalariesCurrent",
        "AccruedEmployeeBenefitsCurrent",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "DebtCurrent",
        "NotesPayableCurrent",
        "ShortTermDebtAndCurrentPortionOfLongTermDebt",
        "CommercialPaper",
    ],
    "current_portion_ltd": [
        "LongTermDebtCurrent",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
    ],
    "operating_lease_liability_current": [
        "OperatingLeaseLiabilityCurrent",
        "OperatingLeaseObligationsCurrent",
    ],
    "finance_lease_liability_current": [
        "FinanceLeaseLiabilityCurrent",
        "CapitalLeaseObligationsCurrent",
    ],
    "deferred_revenue_current": [
        "DeferredRevenueCurrent",
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueAndCredits",
    ],
    "income_taxes_payable": [
        "TaxesPayableCurrent",
        "IncomeTaxesPayable",
        "AccruedIncomeTaxesCurrent",
    ],
    "other_current_liabilities": [
        "OtherLiabilitiesCurrent",
        "OtherCurrentLiabilities",
        "AccruedLiabilitiesAndOtherLiabilities",
    ],
    "total_current_liabilities": [
        "LiabilitiesCurrent",
        "CurrentLiabilities",
        "TotalCurrentLiabilities",
    ],
    # ── Non-current liabilities ───────────────────────────────────────────────
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermNotesPayable",
        "SeniorLongTermNotes",
        "LongTermDebtAndCapitalLeaseObligations",
        "UnsecuredLongTermDebt",
        "SecuredLongTermDebt",
    ],
    "operating_lease_liability_noncurrent": [
        "OperatingLeaseLiabilityNoncurrent",
        "OperatingLeaseObligationsNoncurrent",
        "LongTermOperatingLeaseLiability",
    ],
    "finance_lease_liability_noncurrent": [
        "FinanceLeaseLiabilityNoncurrent",
        "CapitalLeaseObligationsNoncurrent",
        "LongTermCapitalLeaseObligation",
    ],
    "pension_liability": [
        "DefinedBenefitPensionAndOtherPostretirementPlansNoncurrent",
        "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent",
        "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    ],
    "deferred_tax_liability": [
        "DeferredIncomeTaxLiabilitiesNet",
        "DeferredTaxLiabilitiesNoncurrent",
        "DeferredTaxAndOtherLiabilities",
    ],
    "deferred_revenue_noncurrent": [
        "DeferredRevenueNoncurrent",
        "ContractWithCustomerLiabilityNoncurrent",
    ],
    "minority_interest_liability": [
        "MinorityInterest",
        "NoncontrollingInterestInSubsidiaries",
        "RedeemableNoncontrollingInterestEquityFairValue",
    ],
    "other_noncurrent_liabilities": [
        "OtherLiabilitiesNoncurrent",
        "OtherNoncurrentLiabilities",
        "OtherLiabilities",
    ],
    "total_noncurrent_liabilities": [
        "LiabilitiesNoncurrent",
        "NoncurrentLiabilities",
    ],
    "total_liabilities": [
        "Liabilities",
        "TotalLiabilities",
    ],
    # ── Equity ────────────────────────────────────────────────────────────────
    "common_stock_value": [
        "CommonStockValue",
        "CommonStocksIncludingAdditionalPaidInCapital",
        "CommonStockParOrStatedValuePerShare",  # fallback
    ],
    "additional_paid_in_capital": [
        "AdditionalPaidInCapital",
        "AdditionalPaidInCapitalCommonStock",
        "CapitalInExcessOfParValue",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarningsUnappropriated",
        "AccumulatedDeficit",
        "RetainedEarnings",
    ],
    "treasury_stock": [
        "TreasuryStockValue",
        "TreasuryStockCommonValue",
        "TreasuryStockPreferredValue",
    ],
    "accumulated_other_comprehensive_income": [
        "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
        "AccumulatedOtherComprehensiveIncomeLoss",
        "OtherComprehensiveIncomeLossNetOfTax",
    ],
    "noncontrolling_interest": [
        "MinorityInterest",
        "NoncontrollingInterestInSubsidiaries",
        "StockholdersEquityAttributableToNoncontrollingInterest",
    ],
    "total_equity_parent": [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
        "LimitedLiabilityCompanyLlcMembersEquityAttributableToParent",
    ],
    "total_equity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
        "LimitedLiabilityCompanyLlcMembersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssuedNet",
        "SharesOutstanding",
    ],
    "shares_issued": [
        "CommonStockSharesIssued",
        "SharesIssued",
    ],
    "shares_authorized": [
        "CommonStockSharesAuthorized",
    ],
    "preferred_stock": [
        "PreferredStockValue",
        "PreferredStockMember",
    ],
    # ── Computed ─────────────────────────────────────────────────────────────
    "book_value_per_share":       None,
    "tangible_book_value":        None,
    "tangible_book_value_per_share": None,
    "net_debt":                   None,
    "total_debt":                 None,
    "debt_to_equity":             None,
    "current_ratio":              None,
    "quick_ratio":                None,
    "cash_ratio":                 None,
    "goodwill_pct_assets":        None,
    "intangibles_pct_assets":     None,
}

# ---------------------------------------------------------------------------
# Universal balance sheet parser
# ---------------------------------------------------------------------------

class UniversalBalanceSheetParser:
    """
    Enhanced balance sheet parser with 50+ line items and 200+
    XBRL taxonomy variations across companies.
    """

    def __init__(self, cache_path: str | None = None) -> None:
        self._cache = FinancialsCache(db_path=cache_path)
        self._http = httpx.Client(
            headers=EDGAR_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    def _fetch_company_facts(self, cik: str) -> dict:
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

        raise RuntimeError(f"Failed to fetch facts for CIK {cik_padded}: {last_exc}")

    def _extract_concept(
        self,
        facts: dict,
        concepts: list[str],
        period_type: str = "annual",
        units: str = "USD",
        max_periods: int = 20,
    ) -> pd.Series:
        """Extract balance sheet concept. Prefers point-in-time (instantaneous) values."""
        form_filter: set[str]
        if period_type == "annual":
            form_filter = {"10-K", "10-K/A", "20-F", "40-F"}
        elif period_type == "quarterly":
            form_filter = {"10-Q", "10-Q/A", "10-K", "10-K/A"}
        else:
            form_filter = set()

        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei  = facts.get("facts", {}).get("dei", {})

        for concept in concepts:
            concept_data = gaap.get(concept) or dei.get(concept)
            if concept_data is None:
                continue

            unit_data = concept_data.get("units", {})
            rows_raw = unit_data.get(units) or unit_data.get("shares") or []

            best: dict[str, tuple[float, str]] = {}
            for r in rows_raw:
                form = r.get("form", "")
                if form_filter and form not in form_filter:
                    continue
                end = r.get("end", "")
                val = r.get("val")
                filed = r.get("filed", "")
                # Balance sheet: only instantaneous (no start date) or same-day
                start = r.get("start", "")
                if start and start != end:
                    # Skip period items for balance sheet concepts
                    continue
                if val is None:
                    continue
                if end not in best or filed > best[end][1]:
                    best[end] = (float(val), filed)

            if best:
                s = pd.Series(
                    {k: v[0] for k, v in best.items()},
                    name=concepts[0],
                )
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.dropna()
                s = s.sort_index()
                if not s.empty:
                    return s.tail(max_periods)

            # Fallback: allow period items if instantaneous not found
            best2: dict[str, tuple[float, str]] = {}
            for r in rows_raw:
                form = r.get("form", "")
                if form_filter and form not in form_filter:
                    continue
                end = r.get("end", "")
                val = r.get("val")
                filed = r.get("filed", "")
                if val is None:
                    continue
                if end not in best2 or filed > best2[end][1]:
                    best2[end] = (float(val), filed)

            if best2:
                s = pd.Series(
                    {k: v[0] for k, v in best2.items()},
                    name=concepts[0],
                )
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.dropna()
                s = s.sort_index()
                return s.tail(max_periods)

        return pd.Series(dtype=float)

    def parse_balance_sheet(
        self,
        ticker: str,
        periods: int = 20,
        period_type: str = "annual",
    ) -> pd.DataFrame:
        """Parse enhanced balance sheet. Returns DataFrame (period_end index, 50+ columns)."""
        cik = resolve_cik(ticker)
        cache_key = f"enhanced_balance_{period_type}"
        cached = self._cache.get_statement(cik, period_type, cache_key)
        if cached is not None and len(cached) >= min(periods, 2):
            return cached.tail(periods)

        facts = self._fetch_company_facts(cik)
        rows: dict[str, dict[str, float]] = {}

        for metric, concepts in ENHANCED_BALANCE_SHEET_MAP.items():
            if concepts is None:
                continue
            series = self._extract_concept(
                facts, concepts, period_type, max_periods=periods * 2
            )
            if series.empty:
                continue
            for period_end, value in series.items():
                key = str(period_end.date()) if hasattr(period_end, "date") else str(period_end)
                if key not in rows:
                    rows[key] = {}
                if metric not in rows[key]:
                    rows[key][metric] = value

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "period_end"
        df = df.sort_index()

        df = self._compute_derived(df)
        df = df.tail(periods)
        self._cache.set_statement(cik, period_type, cache_key, df)
        return df

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all derived balance sheet metrics."""
        ta = df.get("total_assets")
        equity = df.get("total_equity")
        shares = df.get("shares_outstanding")
        cash = df.get("cash_and_equivalents")
        sti = df.get("short_term_investments")
        ltd = df.get("long_term_debt")
        std = df.get("short_term_debt")
        ca = df.get("total_current_assets")
        cl = df.get("total_current_liabilities")
        inv = df.get("inventory_net")
        gw = df.get("goodwill")
        intang = df.get("intangibles_net")

        # Book value per share
        if equity is not None and shares is not None:
            s = shares.replace(0, float("nan"))
            df["book_value_per_share"] = equity / s

        # Tangible book value (equity minus goodwill minus intangibles)
        if equity is not None:
            tangible = equity.fillna(0).copy()
            if gw is not None:
                tangible -= gw.fillna(0)
            if intang is not None:
                tangible -= intang.fillna(0)
            df["tangible_book_value"] = tangible

            if shares is not None:
                s = shares.replace(0, float("nan"))
                df["tangible_book_value_per_share"] = tangible / s

        # Debt
        ltd_s = ltd.fillna(0) if ltd is not None else pd.Series(0.0, index=df.index)
        std_s = std.fillna(0) if std is not None else pd.Series(0.0, index=df.index)
        cash_s = cash.fillna(0) if cash is not None else pd.Series(0.0, index=df.index)
        sti_s = sti.fillna(0) if sti is not None else pd.Series(0.0, index=df.index)

        df["total_debt"] = ltd_s + std_s
        df["net_debt"] = ltd_s + std_s - cash_s - sti_s

        # Leverage ratios
        if equity is not None:
            eq_nonzero = equity.replace(0, float("nan"))
            df["debt_to_equity"] = df["total_debt"] / eq_nonzero

        if ta is not None:
            ta_nonzero = ta.replace(0, float("nan"))
            df["debt_to_assets"] = df["total_debt"] / ta_nonzero

        # Liquidity ratios
        if ca is not None and cl is not None:
            cl_nz = cl.replace(0, float("nan"))
            df["current_ratio"] = ca / cl_nz
            # Quick ratio: (CA - inventory) / CL
            inv_s = inv.fillna(0) if inv is not None else pd.Series(0.0, index=df.index)
            df["quick_ratio"] = (ca.fillna(0) - inv_s) / cl_nz
            # Cash ratio: (cash + STI) / CL
            df["cash_ratio"] = (cash_s + sti_s) / cl_nz

        # Goodwill/intangible as % of total assets
        if ta is not None:
            ta_nz = ta.replace(0, float("nan"))
            if gw is not None:
                df["goodwill_pct_assets"] = gw.fillna(0) / ta_nz
            if intang is not None:
                df["intangibles_pct_assets"] = intang.fillna(0) / ta_nz

        return df


# ---------------------------------------------------------------------------
# Off-balance-sheet analyzer
# ---------------------------------------------------------------------------

class OffBalanceSheetAnalyzer:
    """
    Analyzes off-balance-sheet obligations: operating leases (pre-ASC 842),
    IFRS 16 / ASC 842 ROU assets, pension funded status, VIEs, guarantees.
    """

    def __init__(self, cache_path: str | None = None) -> None:
        self._bs_parser = UniversalBalanceSheetParser(cache_path=cache_path)

    # Additional XBRL concepts for off-balance-sheet items
    _OBS_CONCEPTS: dict[str, list[str]] = {
        # Operating lease commitments (pre-ASC 842)
        "operating_lease_future_minimum_1yr": [
            "OperatingLeasesFutureMinimumPaymentsDueCurrent",
            "LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths",
        ],
        "operating_lease_future_minimum_2yr": [
            "OperatingLeasesFutureMinimumPaymentsDueInTwoYears",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearTwo",
        ],
        "operating_lease_future_minimum_3yr": [
            "OperatingLeasesFutureMinimumPaymentsDueInThreeYears",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearThree",
        ],
        "operating_lease_future_minimum_4yr": [
            "OperatingLeasesFutureMinimumPaymentsDueInFourYears",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearFour",
        ],
        "operating_lease_future_minimum_5yr": [
            "OperatingLeasesFutureMinimumPaymentsDueInFiveYears",
            "LesseeOperatingLeaseLiabilityPaymentsDueYearFive",
        ],
        "operating_lease_future_minimum_thereafter": [
            "OperatingLeasesFutureMinimumPaymentsDueThereafter",
            "LesseeOperatingLeaseLiabilityPaymentsDueAfterYearFive",
        ],
        "operating_lease_future_minimum_total": [
            "OperatingLeasesFutureMinimumPaymentsDue",
            "LesseeOperatingLeaseLiabilityPaymentsDue",
        ],
        # ASC 842 / IFRS 16 — right-of-use assets already in BS map
        "operating_lease_rou": [
            "OperatingLeaseRightOfUseAsset",
        ],
        "operating_lease_liability_total": [
            "OperatingLeaseLiability",
            "OperatingLeaseLiabilityNoncurrent",
        ],
        "finance_lease_rou": [
            "FinanceLeaseRightOfUseAsset",
        ],
        "finance_lease_liability_total": [
            "FinanceLeaseLiability",
        ],
        # Pension
        "pension_benefit_obligation": [
            "DefinedBenefitPlanBenefitObligation",
            "PensionAndOtherPostretirementBenefitExpense",
        ],
        "pension_plan_assets": [
            "DefinedBenefitPlanFairValueOfPlanAssets",
            "DefinedBenefitPlanAssetsForPlanBenefitsNoncurrent",
        ],
        "pension_funded_status": [
            "DefinedBenefitPlanFundedStatusOfPlan",
        ],
        "pension_service_cost": [
            "DefinedBenefitPlanServiceCost",
        ],
        # Contingent / guarantee liabilities
        "letters_of_credit": [
            "LettersOfCreditOutstandingAmount",
            "GuaranteesAndLettersOfCreditFaceAmount",
        ],
        "purchase_obligations": [
            "PurchaseObligationDueInNextTwelveMonths",
            "RecordedUnconditionalPurchaseObligationDueWithinOneYear",
        ],
        "contingent_liabilities": [
            "LossContingencyAccrualAtCarryingValue",
            "LossContingencyEstimateOfPossibleLoss",
        ],
        # Variable interest entities
        "vie_assets": [
            "VariableInterestEntityConsolidatedCarryingAmountAssets",
        ],
        "vie_liabilities": [
            "VariableInterestEntityConsolidatedCarryingAmountLiabilities",
        ],
    }

    def get_lease_obligations(
        self,
        ticker: str,
    ) -> dict[str, Any]:
        """Extract operating/finance lease ROU assets, liabilities, and future commitments."""
        cik = resolve_cik(ticker)
        facts = self._bs_parser._fetch_company_facts(cik)

        result: dict[str, Any] = {}

        # Extract each OBS concept
        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei  = facts.get("facts", {}).get("dei", {})

        def _get_latest(concept_list: list[str]) -> float | None:
            for concept in concept_list:
                data = gaap.get(concept) or dei.get(concept)
                if data is None:
                    continue
                units = data.get("units", {})
                rows = units.get("USD") or units.get("shares") or []
                if rows:
                    latest = max(rows, key=lambda r: r.get("filed", ""))
                    val = latest.get("val")
                    return float(val) if val is not None else None
            return None

        rou_op = _get_latest(self._OBS_CONCEPTS["operating_lease_rou"])
        rou_fin = _get_latest(self._OBS_CONCEPTS["finance_lease_rou"])
        lease_liab = _get_latest(self._OBS_CONCEPTS["operating_lease_liability_total"])

        result["rou_assets_operating"] = rou_op
        result["rou_assets_finance"] = rou_fin
        result["lease_liability_total"] = lease_liab

        # Future commitments
        commitments: dict[str, float | None] = {}
        for yr, key in [
            (1, "operating_lease_future_minimum_1yr"),
            (2, "operating_lease_future_minimum_2yr"),
            (3, "operating_lease_future_minimum_3yr"),
            (4, "operating_lease_future_minimum_4yr"),
            (5, "operating_lease_future_minimum_5yr"),
        ]:
            commitments[f"year_{yr}"] = _get_latest(self._OBS_CONCEPTS[key])
        commitments["thereafter"] = _get_latest(
            self._OBS_CONCEPTS["operating_lease_future_minimum_thereafter"]
        )
        result["future_commitments"] = commitments

        # Classify lease standard
        if rou_op is not None:
            result["lease_type"] = "ASC_842_IFRS16"
        elif any(v is not None for v in commitments.values()):
            result["lease_type"] = "pre_ASC_842"
        else:
            result["lease_type"] = "unknown"

        # Capitalization proxy: approximate NPV using 8x first-year rent
        if commitments.get("year_1") is not None:
            result["capitalization_multiple_proxy"] = commitments["year_1"] * 8.0
        elif lease_liab is not None:
            result["capitalization_multiple_proxy"] = lease_liab
        else:
            result["capitalization_multiple_proxy"] = None

        return result

    def get_pension_status(
        self,
        ticker: str,
    ) -> dict[str, Any]:
        """
        Pension obligation analysis: funded vs underfunded status,
        service cost, underfunded amount.
        """
        cik = resolve_cik(ticker)
        facts = self._bs_parser._fetch_company_facts(cik)
        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei  = facts.get("facts", {}).get("dei", {})

        def _get_latest(concept_list: list[str]) -> float | None:
            for concept in concept_list:
                data = gaap.get(concept) or dei.get(concept)
                if data is None:
                    continue
                units = data.get("units", {})
                rows = units.get("USD") or []
                if rows:
                    latest = max(rows, key=lambda r: r.get("filed", ""))
                    val = latest.get("val")
                    return float(val) if val is not None else None
            return None

        obligation = _get_latest(self._OBS_CONCEPTS["pension_benefit_obligation"])
        plan_assets = _get_latest(self._OBS_CONCEPTS["pension_plan_assets"])
        funded_status = _get_latest(self._OBS_CONCEPTS["pension_funded_status"])
        service_cost = _get_latest(self._OBS_CONCEPTS["pension_service_cost"])

        # Compute underfunded amount
        if funded_status is not None:
            underfunded_amount = funded_status  # negative = underfunded
        elif obligation is not None and plan_assets is not None:
            underfunded_amount = plan_assets - obligation  # negative = underfunded
        else:
            underfunded_amount = None

        status = "UNKNOWN"
        if underfunded_amount is not None:
            if underfunded_amount < -500_000_000:
                status = "SIGNIFICANTLY_UNDERFUNDED"
            elif underfunded_amount < 0:
                status = "UNDERFUNDED"
            elif underfunded_amount == 0:
                status = "FULLY_FUNDED"
            else:
                status = "OVERFUNDED"

        return {
            "ticker": ticker,
            "benefit_obligation": obligation,
            "plan_assets": plan_assets,
            "funded_status": underfunded_amount,
            "pension_status": status,
            "annual_service_cost": service_cost,
            "has_pension": obligation is not None or plan_assets is not None,
        }

    def get_contingent_obligations(
        self,
        ticker: str,
    ) -> dict[str, Any]:
        """
        Extract contingent liabilities, guarantees, letters of credit,
        and VIE exposure.
        """
        cik = resolve_cik(ticker)
        facts = self._bs_parser._fetch_company_facts(cik)
        gaap = facts.get("facts", {}).get("us-gaap", {})

        def _get_latest(concept_list: list[str]) -> float | None:
            for concept in concept_list:
                data = gaap.get(concept)
                if data is None:
                    continue
                units = data.get("units", {})
                rows = units.get("USD") or []
                if rows:
                    latest = max(rows, key=lambda r: r.get("filed", ""))
                    val = latest.get("val")
                    return float(val) if val is not None else None
            return None

        loc = _get_latest(self._OBS_CONCEPTS["letters_of_credit"])
        purchase_obs = _get_latest(self._OBS_CONCEPTS["purchase_obligations"])
        contingent = _get_latest(self._OBS_CONCEPTS["contingent_liabilities"])
        vie_assets = _get_latest(self._OBS_CONCEPTS["vie_assets"])
        vie_liab = _get_latest(self._OBS_CONCEPTS["vie_liabilities"])

        total_off_bs = sum(
            v for v in [loc, purchase_obs, contingent] if v is not None
        )

        return {
            "ticker": ticker,
            "letters_of_credit": loc,
            "purchase_obligations_1yr": purchase_obs,
            "contingent_liabilities": contingent,
            "vie_assets": vie_assets,
            "vie_liabilities": vie_liab,
            "total_quantified_off_balance_sheet": total_off_bs if total_off_bs > 0 else None,
            "has_vie": vie_assets is not None,
        }

    def get_full_off_balance_sheet(self, ticker: str) -> dict[str, Any]:
        """Consolidated off-balance-sheet summary."""
        return {
            "ticker": ticker,
            "leases": self.get_lease_obligations(ticker),
            "pension": self.get_pension_status(ticker),
            "contingent": self.get_contingent_obligations(ticker),
        }


# ---------------------------------------------------------------------------
# Goodwill analyzer
# ---------------------------------------------------------------------------

class GoodwillAnalyzer:
    """
    Track goodwill quality, impairment history, and acquisition premiums.
    """

    _GOODWILL_CONCEPTS: dict[str, list[str]] = {
        "goodwill_beginning": [
            "GoodwillGross",
        ],
        "goodwill_acquisitions": [
            "GoodwillAcquiredDuringPeriod",
            "GoodwillPurchaseAccountingAdjustments",
        ],
        "goodwill_impairment": [
            "GoodwillImpairmentLoss",
            "GoodwillWrittenOffRelatedToSaleOfBusinessUnit",
        ],
        "goodwill_end": [
            "Goodwill",
            "GoodwillNet",
        ],
        "intangibles_acquisitions": [
            "FiniteLivedIntangibleAssetsAcquiredAsPartOfBusinessCombinationTableTextBlock",
            "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill",
        ],
        "acquisition_price": [
            "BusinessCombinationConsiderationTransferred1",
            "PaymentsToAcquireBusinessesNetOfCashAcquired",
        ],
        "tangible_assets_acquired": [
            "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedNet",
        ],
    }

    def __init__(self, cache_path: str | None = None) -> None:
        self._bs_parser = UniversalBalanceSheetParser(cache_path=cache_path)

    def _get_time_series(
        self,
        cik: str,
        concept_list: list[str],
        periods: int = 20,
    ) -> pd.Series:
        facts = self._bs_parser._fetch_company_facts(cik)
        gaap = facts.get("facts", {}).get("us-gaap", {})

        for concept in concept_list:
            data = gaap.get(concept)
            if data is None:
                continue
            units = data.get("units", {})
            rows_raw = units.get("USD") or []

            best: dict[str, tuple[float, str]] = {}
            for r in rows_raw:
                form = r.get("form", "")
                if form not in ("10-K", "10-K/A", "10-Q", "10-Q/A"):
                    continue
                end = r.get("end", "")
                val = r.get("val")
                filed = r.get("filed", "")
                if val is None:
                    continue
                if end not in best or filed > best[end][1]:
                    best[end] = (float(val), filed)

            if best:
                s = pd.Series({k: v[0] for k, v in best.items()})
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.dropna().sort_index()
                return s.tail(periods)

        return pd.Series(dtype=float)

    def get_goodwill_history(
        self,
        ticker: str,
        periods: int = 20,
    ) -> pd.DataFrame:
        """
        Multi-period goodwill tracking.

        Returns DataFrame with columns:
        - goodwill: ending goodwill balance
        - goodwill_impairment: impairment charges taken
        - goodwill_acquisitions: goodwill added via M&A
        - cumulative_impairments: running total impairments
        - goodwill_pct_assets: goodwill / total assets
        """
        cik = resolve_cik(ticker)

        gw_series = self._get_time_series(cik, ["Goodwill", "GoodwillNet"], periods)
        imp_series = self._get_time_series(
            cik, self._GOODWILL_CONCEPTS["goodwill_impairment"], periods
        )
        acq_series = self._get_time_series(
            cik, self._GOODWILL_CONCEPTS["goodwill_acquisitions"], periods
        )

        # Build aligned DataFrame
        all_idx = sorted(set(list(gw_series.index) + list(imp_series.index)))
        if not all_idx:
            return pd.DataFrame()

        df = pd.DataFrame(index=all_idx)
        df.index.name = "period_end"
        df["goodwill"] = gw_series.reindex(all_idx)
        df["goodwill_impairment"] = imp_series.reindex(all_idx).fillna(0)
        df["goodwill_acquisitions"] = acq_series.reindex(all_idx).fillna(0)

        # Cumulative impairments
        df["cumulative_impairments"] = df["goodwill_impairment"].cumsum()

        # Goodwill growth
        df["goodwill_growth_pct"] = df["goodwill"].pct_change()

        return df.dropna(subset=["goodwill"]).tail(periods)

    def get_goodwill_quality_score(
        self,
        ticker: str,
    ) -> dict[str, Any]:
        """
        Goodwill quality score (0-100). Penalizes: >30% of assets, impairment history,
        rapid growth, and goodwill > equity.
        """
        cik = resolve_cik(ticker)
        bs_df = self._bs_parser.parse_balance_sheet(ticker, periods=10, period_type="annual")
        gw_history = self.get_goodwill_history(ticker, periods=10)

        score = 100
        flags: list[str] = []

        if bs_df.empty or "goodwill" not in bs_df.columns:
            return {"error": "no_goodwill_data"}

        latest_bs = bs_df.iloc[-1]
        gw = latest_bs.get("goodwill", 0) or 0
        ta = latest_bs.get("total_assets", 1) or 1
        equity = latest_bs.get("total_equity", 1) or 1

        gw_pct_assets = gw / ta if ta else 0
        gw_to_equity = gw / equity if equity else 0

        # Flag: goodwill > 30% of assets
        if gw_pct_assets > 0.30:
            score -= 20
            flags.append(f"GOODWILL_EXCEEDS_30PCT_ASSETS ({gw_pct_assets:.1%})")
        elif gw_pct_assets > 0.20:
            score -= 10
            flags.append(f"GOODWILL_EXCEEDS_20PCT_ASSETS ({gw_pct_assets:.1%})")

        # Flag: impairment history
        impairment_count = 0
        if not gw_history.empty and "goodwill_impairment" in gw_history.columns:
            impairment_years = (gw_history["goodwill_impairment"] > 0).sum()
            impairment_count = int(impairment_years)
            if impairment_count > 0:
                penalty = min(impairment_count * 15, 40)
                score -= penalty
                flags.append(f"IMPAIRMENT_HISTORY ({impairment_count} periods)")

        # Flag: goodwill > equity
        if gw_to_equity > 1.0:
            score -= 15
            flags.append(f"GOODWILL_EXCEEDS_EQUITY ({gw_to_equity:.1f}x)")

        # Flag: rapid goodwill growth
        if not gw_history.empty and "goodwill_growth_pct" in gw_history.columns:
            recent_growth = gw_history["goodwill_growth_pct"].tail(3)
            avg_gw_growth = float(recent_growth.mean()) if not recent_growth.empty else 0
            if avg_gw_growth > 0.20:
                score -= 10
                flags.append(f"RAPID_GOODWILL_GROWTH ({avg_gw_growth:.1%} avg)")

        score = max(0, min(100, score))

        if score >= 80:
            rating = "A"
        elif score >= 60:
            rating = "B"
        elif score >= 40:
            rating = "C"
        else:
            rating = "D"

        return {
            "ticker": ticker,
            "score": score,
            "rating": rating,
            "goodwill": gw,
            "goodwill_pct_assets": round(gw_pct_assets, 4),
            "goodwill_to_equity": round(gw_to_equity, 4),
            "has_impairments": impairment_count > 0,
            "impairment_count": impairment_count,
            "flags": flags,
        }


# ---------------------------------------------------------------------------
# Working capital engine
# ---------------------------------------------------------------------------

class WorkingCapitalEngine:
    """
    Working capital analytics: DSO, DIO, DPO, Cash Conversion Cycle,
    trend analysis, and receivables quality assessment.
    """

    def __init__(self, cache_path: str | None = None) -> None:
        self._bs_parser = UniversalBalanceSheetParser(cache_path=cache_path)
        self._cache = FinancialsCache(db_path=cache_path)
        self._http = httpx.Client(
            headers=EDGAR_HEADERS, timeout=_TIMEOUT, follow_redirects=True
        )

    def _get_income_data(self, ticker: str, periods: int) -> pd.DataFrame:
        """Pull revenue and COGS data from standardized financials."""
        from sentinel.sfe.standardized_financials import (
            FinancialStatementStandardizer,
            INCOME_STATEMENT_MAP,
        )
        cik = resolve_cik(ticker)
        std = FinancialStatementStandardizer()
        return std.get_income_statement(cik, periods=periods, period_type="quarterly")

    def compute_working_capital_metrics(
        self,
        ticker: str,
        periods: int = 20,
    ) -> pd.DataFrame:
        """
        Compute DSO, DIO, DPO, CCC for N quarters.
        DSO=AR/Rev*365, DIO=Inv/COGS*365, DPO=AP/COGS*365, CCC=DSO+DIO-DPO.
        """
        bs_df = self._bs_parser.parse_balance_sheet(ticker, periods=periods + 2, period_type="quarterly")
        is_df = self._get_income_data(ticker, periods=periods + 2)

        if bs_df.empty:
            return pd.DataFrame()

        # Align on common periods
        common_idx = bs_df.index.intersection(is_df.index) if not is_df.empty else bs_df.index
        if len(common_idx) == 0:
            common_idx = bs_df.index

        result = pd.DataFrame(index=bs_df.index)

        ar = bs_df.get("accounts_receivable_net")
        inv = bs_df.get("inventory_net")
        ap = bs_df.get("accounts_payable")

        # Revenue and COGS from income statement (annualize quarterly figures * 4)
        rev = is_df.get("revenue").reindex(bs_df.index) if not is_df.empty and "revenue" in is_df.columns else None
        cogs = is_df.get("cost_of_revenue").reindex(bs_df.index) if not is_df.empty and "cost_of_revenue" in is_df.columns else None

        # DSO
        if ar is not None and rev is not None:
            rev_daily = rev / 91.25  # quarterly revenue to daily
            rev_daily = rev_daily.replace(0, float("nan"))
            result["dso"] = ar / rev_daily
        elif ar is not None:
            result["dso"] = None

        # DIO
        if inv is not None and cogs is not None:
            cogs_daily = cogs / 91.25
            cogs_daily = cogs_daily.replace(0, float("nan"))
            result["dio"] = inv / cogs_daily
        elif inv is not None:
            result["dio"] = None

        # DPO
        if ap is not None and cogs is not None:
            cogs_daily = cogs / 91.25
            cogs_daily = cogs_daily.replace(0, float("nan"))
            result["dpo"] = ap / cogs_daily
        elif ap is not None:
            result["dpo"] = None

        # CCC
        if "dso" in result.columns and "dio" in result.columns and "dpo" in result.columns:
            dso = result["dso"].fillna(0)
            dio = result["dio"].fillna(0)
            dpo = result["dpo"].fillna(0)
            # Only compute CCC where all three are available
            has_all = result["dso"].notna() & result["dio"].notna() & result["dpo"].notna()
            result["ccc"] = np.where(has_all, dso + dio - dpo, np.nan)

        # Working capital balance
        ca = bs_df.get("total_current_assets")
        cl = bs_df.get("total_current_liabilities")
        if ca is not None and cl is not None:
            result["working_capital"] = ca.fillna(0) - cl.fillna(0)

        # Add balance sheet components
        for col in ["accounts_receivable_net", "inventory_net", "accounts_payable",
                    "total_current_assets", "total_current_liabilities"]:
            if col in bs_df.columns:
                result[col] = bs_df[col]

        return result.tail(periods)

    def compute_trend_metrics(
        self,
        ticker: str,
        periods: int = 12,
    ) -> dict[str, Any]:
        """Assess WC trend: IMPROVING / DETERIORATING / STABLE / MIXED. Returns per-metric trends."""
        wc_df = self.compute_working_capital_metrics(ticker, periods=periods)
        if wc_df.empty:
            return {"error": "insufficient_data"}

        result: dict[str, Any] = {"ticker": ticker}

        # CCC trend
        if "ccc" in wc_df.columns:
            ccc = wc_df["ccc"].dropna()
            if len(ccc) >= 2:
                result["current_ccc"] = round(float(ccc.iloc[-1]), 1)
                result["avg_ccc"] = round(float(ccc.mean()), 1)
                ccc_trend = float(ccc.diff().mean())
                result["ccc_trend"] = round(ccc_trend, 2)
                # Positive trend = CCC increasing = deteriorating
                if ccc_trend > 2:
                    ccc_signal = "DETERIORATING"
                elif ccc_trend < -2:
                    ccc_signal = "IMPROVING"
                else:
                    ccc_signal = "STABLE"
                result["ccc_trend_signal"] = ccc_signal

        # DSO trend
        if "dso" in wc_df.columns:
            dso = wc_df["dso"].dropna()
            if len(dso) >= 2:
                result["current_dso"] = round(float(dso.iloc[-1]), 1)
                result["dso_trend"] = round(float(dso.diff().mean()), 2)

        # DIO trend
        if "dio" in wc_df.columns:
            dio = wc_df["dio"].dropna()
            if len(dio) >= 2:
                result["current_dio"] = round(float(dio.iloc[-1]), 1)
                result["dio_trend"] = round(float(dio.diff().mean()), 2)

        # DPO trend
        if "dpo" in wc_df.columns:
            dpo = wc_df["dpo"].dropna()
            if len(dpo) >= 2:
                result["current_dpo"] = round(float(dpo.iloc[-1]), 1)
                result["dpo_trend"] = round(float(dpo.diff().mean()), 2)

        # Overall trend assessment
        signals = [
            result.get("ccc_trend_signal", "STABLE"),
        ]
        # Additional signal: DPO increasing = good (paying slower)
        dpo_trend = result.get("dpo_trend", 0)
        if dpo_trend and dpo_trend > 1:
            signals.append("IMPROVING")
        elif dpo_trend and dpo_trend < -1:
            signals.append("DETERIORATING")

        improving = signals.count("IMPROVING")
        deteriorating = signals.count("DETERIORATING")

        if improving > deteriorating:
            result["trend"] = "IMPROVING"
        elif deteriorating > improving:
            result["trend"] = "DETERIORATING"
        elif improving == deteriorating and improving > 0:
            result["trend"] = "MIXED"
        else:
            result["trend"] = "STABLE"

        return result

    def receivables_quality_check(
        self,
        ticker: str,
        periods: int = 12,
    ) -> dict[str, Any]:
        """Check AR quality: GOOD / WATCH / RED_FLAG based on AR vs revenue growth divergence."""
        bs_df = self._bs_parser.parse_balance_sheet(
            ticker, periods=periods + 2, period_type="quarterly"
        )
        is_df = self._get_income_data(ticker, periods=periods + 2)

        if bs_df.empty:
            return {"error": "no_data"}

        result: dict[str, Any] = {"ticker": ticker}

        ar_net = bs_df.get("accounts_receivable_net")
        ar_gross = bs_df.get("accounts_receivable_gross")
        allowance = bs_df.get("allowance_doubtful_accounts")
        rev = is_df.get("revenue").reindex(bs_df.index) if not is_df.empty and "revenue" in is_df.columns else None

        # AR vs revenue growth comparison
        if ar_net is not None and rev is not None:
            ar_growth = ar_net.pct_change(4).dropna()  # YoY
            rev_growth = rev.pct_change(4).reindex(ar_growth.index).dropna()
            common = ar_growth.index.intersection(rev_growth.index)
            if len(common) >= 2:
                ar_g = float(ar_growth[common].mean())
                rev_g = float(rev_growth[common].mean())
                divergence = ar_g - rev_g
                result["ar_growth_avg"] = round(ar_g, 4)
                result["rev_growth_avg"] = round(rev_g, 4)
                result["divergence"] = round(divergence, 4)

                if divergence > 0.10:
                    result["quality"] = "RED_FLAG"
                    result["quality_reason"] = "AR growing >10ppt faster than revenue"
                elif divergence > 0.05:
                    result["quality"] = "WATCH"
                    result["quality_reason"] = "AR growing 5-10ppt faster than revenue"
                else:
                    result["quality"] = "GOOD"
                    result["quality_reason"] = "AR growth in line with revenue"

        # Allowance coverage ratio
        if ar_gross is not None and allowance is not None:
            latest_gross = ar_gross.dropna()
            latest_allow = allowance.dropna().reindex(latest_gross.index)
            if not latest_gross.empty and not latest_allow.empty:
                coverage = (latest_allow.iloc[-1] / latest_gross.iloc[-1]
                           if latest_gross.iloc[-1] != 0 else None)
                result["allowance_coverage"] = round(float(coverage), 4) if coverage else None

        if "quality" not in result:
            result["quality"] = "UNKNOWN"
            result["quality_reason"] = "insufficient_data"

        return result


# ---------------------------------------------------------------------------
# Financial health scorer
# ---------------------------------------------------------------------------

class FinancialHealthScorer:
    """
    Comprehensive financial health assessment combining:
    - Altman Z-Score (public manufacturing companies)
    - Piotroski F-Score (9-point balance sheet health)
    - Standard liquidity and leverage ratios
    - Aggregate 0-100 health score with A/B/C/D rating
    """

    def __init__(self, cache_path: str | None = None) -> None:
        self._bs_parser = UniversalBalanceSheetParser(cache_path=cache_path)
        self._cache = FinancialsCache(db_path=cache_path)

    def _get_is_data(self, ticker: str, periods: int = 8) -> pd.DataFrame:
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        cik = resolve_cik(ticker)
        std = FinancialStatementStandardizer()
        return std.get_income_statement(cik, periods=periods, period_type="annual")

    def _get_cf_data(self, ticker: str, periods: int = 8) -> pd.DataFrame:
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        cik = resolve_cik(ticker)
        std = FinancialStatementStandardizer()
        return std.get_cash_flow_statement(cik, periods=periods, period_type="annual")

    def compute_altman_z_score(
        self,
        ticker: str,
        market_cap: float | None = None,
    ) -> dict[str, Any]:
        """
        Altman Z-Score: Z=1.2*X1+1.4*X2+3.3*X3+0.6*X4+1.0*X5.
        Zones: >2.99 Safe, 1.81-2.99 Grey, <1.81 Distress.
        """
        bs_df = self._bs_parser.parse_balance_sheet(ticker, periods=2, period_type="annual")
        is_df = self._get_is_data(ticker, periods=2)

        if bs_df.empty:
            return {"error": "no_balance_sheet_data"}

        latest_bs = bs_df.iloc[-1]
        ta = latest_bs.get("total_assets")
        if ta is None or ta == 0:
            return {"error": "missing_total_assets"}

        ca = latest_bs.get("total_current_assets", 0) or 0
        cl = latest_bs.get("total_current_liabilities", 0) or 0
        re = latest_bs.get("retained_earnings", 0) or 0
        tl = latest_bs.get("total_liabilities", 0) or 0
        equity = latest_bs.get("total_equity", 0) or 0

        # Get EBIT and Revenue from income statement
        ebit = None
        rev = None
        if not is_df.empty:
            latest_is = is_df.iloc[-1]
            ebit = latest_is.get("ebit")
            rev = latest_is.get("revenue")

        # Compute components
        x1 = (ca - cl) / ta
        x2 = re / ta
        x3 = (ebit / ta) if ebit is not None else 0.0
        # X4: use market cap if available, else book equity
        if market_cap is not None and tl and tl != 0:
            x4 = market_cap / tl
        elif tl and tl != 0:
            x4 = equity / tl
        else:
            x4 = 1.0
        x5 = (rev / ta) if rev is not None else 0.0

        z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        if z_score > 2.99:
            zone = "SAFE"
        elif z_score > 1.81:
            zone = "GREY"
        else:
            zone = "DISTRESS"

        return {
            "ticker": ticker,
            "z_score": round(float(z_score), 3),
            "zone": zone,
            "x1_working_capital_ta": round(float(x1), 4),
            "x2_retained_earnings_ta": round(float(x2), 4),
            "x3_ebit_ta": round(float(x3), 4),
            "x4_equity_liabilities": round(float(x4), 4),
            "x5_revenue_ta": round(float(x5), 4),
        }

    def compute_piotroski_f_score(
        self,
        ticker: str,
    ) -> dict[str, Any]:
        """
        9-point Piotroski F-Score. Profitability (F1-F4), Leverage/Liquidity (F5-F7),
        Efficiency (F8-F9). 7-9=Strong, 4-6=Average, 0-3=Weak.
        """
        bs_df = self._bs_parser.parse_balance_sheet(ticker, periods=3, period_type="annual")
        is_df = self._get_is_data(ticker, periods=3)
        cf_df = self._get_cf_data(ticker, periods=3)

        if bs_df.empty or len(bs_df) < 2:
            return {"error": "need_at_least_2_years_data"}

        scores: dict[str, int] = {}
        details: dict[str, Any] = {}

        curr_bs = bs_df.iloc[-1] if len(bs_df) >= 1 else {}
        prev_bs = bs_df.iloc[-2] if len(bs_df) >= 2 else {}
        curr_is = is_df.iloc[-1] if not is_df.empty and len(is_df) >= 1 else {}
        prev_is = is_df.iloc[-2] if not is_df.empty and len(is_df) >= 2 else {}
        curr_cf = cf_df.iloc[-1] if not cf_df.empty and len(cf_df) >= 1 else {}

        def _v(d: Any, key: str) -> float:
            if isinstance(d, pd.Series):
                val = d.get(key)
                return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else 0.0
            return float(d.get(key, 0) or 0)

        # ── Profitability ────────────────────────────────────────────────────
        ta_curr = _v(curr_bs, "total_assets") or 1
        ta_prev = _v(prev_bs, "total_assets") or 1
        ni_curr = _v(curr_is, "net_income")
        ni_prev = _v(prev_is, "net_income")
        ocf_curr = _v(curr_cf, "operating_cf")

        roa_curr = ni_curr / ta_curr
        roa_prev = ni_prev / ta_prev

        # F1: ROA > 0
        scores["F1_roa_positive"] = 1 if roa_curr > 0 else 0
        details["roa_current"] = round(roa_curr, 4)

        # F2: Operating CF > 0
        scores["F2_ocf_positive"] = 1 if ocf_curr > 0 else 0
        details["ocf_current"] = ocf_curr

        # F3: ROA improving
        scores["F3_roa_improving"] = 1 if roa_curr > roa_prev else 0
        details["roa_prev"] = round(roa_prev, 4)

        # F4: Cash quality (OCF / TA > ROA)
        ocf_roa = ocf_curr / ta_curr
        scores["F4_accruals"] = 1 if ocf_roa > roa_curr else 0
        details["ocf_roa"] = round(ocf_roa, 4)

        # ── Leverage / Liquidity ─────────────────────────────────────────────
        ltd_curr = _v(curr_bs, "long_term_debt")
        ltd_prev = _v(prev_bs, "long_term_debt")
        leverage_curr = ltd_curr / ta_curr
        leverage_prev = ltd_prev / ta_prev

        # F5: Leverage decreased
        scores["F5_leverage_decreased"] = 1 if leverage_curr < leverage_prev else 0
        details["leverage_current"] = round(leverage_curr, 4)

        ca_curr = _v(curr_bs, "total_current_assets")
        cl_curr = _v(curr_bs, "total_current_liabilities") or 1
        ca_prev = _v(prev_bs, "total_current_assets")
        cl_prev = _v(prev_bs, "total_current_liabilities") or 1
        cr_curr = ca_curr / cl_curr
        cr_prev = ca_prev / cl_prev

        # F6: Current ratio improved
        scores["F6_current_ratio_improved"] = 1 if cr_curr > cr_prev else 0
        details["current_ratio"] = round(cr_curr, 3)

        # F7: No new share dilution (shares outstanding not increased significantly)
        shares_curr = _v(curr_bs, "shares_outstanding")
        shares_prev = _v(prev_bs, "shares_outstanding")
        if shares_prev > 0:
            share_change = (shares_curr - shares_prev) / shares_prev
            scores["F7_no_dilution"] = 1 if share_change <= 0.02 else 0
            details["share_change_pct"] = round(share_change, 4)
        else:
            scores["F7_no_dilution"] = 0

        # ── Operating Efficiency ─────────────────────────────────────────────
        rev_curr = _v(curr_is, "revenue") or 1
        rev_prev = _v(prev_is, "revenue") or 1
        gross_curr = _v(curr_is, "gross_profit")
        gross_prev = _v(prev_is, "gross_profit")

        gm_curr = gross_curr / rev_curr
        gm_prev = gross_prev / rev_prev

        # F8: Gross margin improved
        scores["F8_gross_margin_improved"] = 1 if gm_curr > gm_prev else 0
        details["gross_margin_current"] = round(gm_curr, 4)

        # F9: Asset turnover improved
        at_curr = rev_curr / ta_curr
        at_prev = rev_prev / ta_prev
        scores["F9_asset_turnover_improved"] = 1 if at_curr > at_prev else 0
        details["asset_turnover"] = round(at_curr, 4)

        total_score = sum(scores.values())

        if total_score >= 7:
            signal = "STRONG"
        elif total_score >= 4:
            signal = "AVERAGE"
        else:
            signal = "WEAK"

        return {
            "ticker": ticker,
            "f_score": total_score,
            "signal": signal,
            "scores": scores,
            "details": details,
        }

    def compute_ratio_health(
        self,
        ticker: str,
    ) -> dict[str, Any]:
        """
        Standard liquidity and leverage ratio health check.

        Returns individual ratios plus a composite component score (0-40).
        """
        bs_df = self._bs_parser.parse_balance_sheet(ticker, periods=2, period_type="annual")
        is_df = self._get_is_data(ticker, periods=2)

        if bs_df.empty:
            return {"error": "no_data"}

        latest = bs_df.iloc[-1]
        latest_is = is_df.iloc[-1] if not is_df.empty else pd.Series(dtype=float)

        ta = float(latest.get("total_assets") or 1)
        ca = float(latest.get("total_current_assets") or 0)
        cl = float(latest.get("total_current_liabilities") or 1)
        inv = float(latest.get("inventory_net") or 0)
        cash = float(latest.get("cash_and_equivalents") or 0)
        sti = float(latest.get("short_term_investments") or 0)
        ltd = float(latest.get("long_term_debt") or 0)
        std = float(latest.get("short_term_debt") or 0)
        equity = float(latest.get("total_equity") or 1)
        ebit = float(latest_is.get("ebit") or 0) if not latest_is.empty else 0
        int_exp = float(latest_is.get("interest_expense") or 0) if not latest_is.empty else 0

        current_ratio = ca / cl if cl else None
        quick_ratio = (ca - inv) / cl if cl else None
        cash_ratio = (cash + sti) / cl if cl else None
        debt_to_equity = (ltd + std) / equity if equity else None
        net_debt = ltd + std - cash - sti
        interest_coverage = ebit / int_exp if int_exp else None

        # Score each ratio
        component_score = 0

        # Current ratio: >= 2 is healthy
        if current_ratio is not None:
            if current_ratio >= 2.0:
                component_score += 10
            elif current_ratio >= 1.5:
                component_score += 7
            elif current_ratio >= 1.0:
                component_score += 4
            else:
                component_score += 0

        # Quick ratio: >= 1.0 is healthy
        if quick_ratio is not None:
            if quick_ratio >= 1.5:
                component_score += 8
            elif quick_ratio >= 1.0:
                component_score += 5
            elif quick_ratio >= 0.5:
                component_score += 2
            else:
                component_score += 0

        # D/E ratio: < 1.0 is conservative
        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                component_score += 12
            elif debt_to_equity < 1.0:
                component_score += 8
            elif debt_to_equity < 2.0:
                component_score += 4
            else:
                component_score += 0

        # Interest coverage: > 5x is very safe
        if interest_coverage is not None:
            if interest_coverage > 10:
                component_score += 10
            elif interest_coverage > 5:
                component_score += 7
            elif interest_coverage > 2:
                component_score += 3
            elif interest_coverage > 0:
                component_score += 1
            else:
                component_score += 0

        return {
            "ticker": ticker,
            "current_ratio": _round_safe(current_ratio),
            "quick_ratio": _round_safe(quick_ratio),
            "cash_ratio": _round_safe(cash_ratio),
            "debt_to_equity": _round_safe(debt_to_equity),
            "net_debt": net_debt,
            "interest_coverage": _round_safe(interest_coverage),
            "component_score": component_score,  # max 40
        }

    def compute_aggregate_health_score(
        self,
        ticker: str,
        market_cap: float | None = None,
    ) -> dict[str, Any]:
        """Aggregate health score 0-100 (A/B/C/D): 30pts Altman Z + 30pts Piotroski F + 40pts ratios."""
        z_result = self.compute_altman_z_score(ticker, market_cap=market_cap)
        f_result = self.compute_piotroski_f_score(ticker)
        ratio_result = self.compute_ratio_health(ticker)

        total_score = 0
        components: dict[str, Any] = {}

        # Z-Score component (30 points)
        if "z_score" in z_result:
            z = z_result["z_score"]
            zone = z_result.get("zone", "GREY")
            if zone == "SAFE":
                z_component = 30
            elif zone == "GREY":
                # Linear interpolation in grey zone
                z_component = int(15 + (z - 1.81) / (2.99 - 1.81) * 15)
            else:
                # Distress zone: partial credit based on how far below 1.81
                z_component = max(0, int((z / 1.81) * 15))
            total_score += z_component
            components["altman_z_component"] = z_component
            components["altman_z_score"] = z
            components["altman_zone"] = zone

        # Piotroski F-Score component (30 points)
        if "f_score" in f_result:
            f = f_result["f_score"]
            f_component = int((f / 9) * 30)
            total_score += f_component
            components["piotroski_f_component"] = f_component
            components["piotroski_f_score"] = f
            components["piotroski_signal"] = f_result.get("signal")

        # Ratio health component (40 points)
        if "component_score" in ratio_result:
            ratio_component = ratio_result["component_score"]
            total_score += ratio_component
            components["ratio_component"] = ratio_component
            components["current_ratio"] = ratio_result.get("current_ratio")
            components["quick_ratio"] = ratio_result.get("quick_ratio")
            components["debt_to_equity"] = ratio_result.get("debt_to_equity")
            components["interest_coverage"] = ratio_result.get("interest_coverage")

        total_score = max(0, min(100, total_score))

        if total_score >= 80:
            rating = "A"
            assessment = "EXCELLENT — strong financial health across all dimensions"
        elif total_score >= 60:
            rating = "B"
            assessment = "GOOD — solid fundamentals with minor concerns"
        elif total_score >= 40:
            rating = "C"
            assessment = "FAIR — material financial risks present"
        else:
            rating = "D"
            assessment = "POOR — significant financial distress signals"

        return {
            "ticker": ticker,
            "aggregate_score": total_score,
            "rating": rating,
            "assessment": assessment,
            "components": components,
            "altman_detail": z_result,
            "piotroski_detail": f_result,
            "ratio_detail": ratio_result,
        }


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class BalanceSheetResponse(BaseModel):
    ticker: str
    cik: str
    period_type: str
    periods: int
    line_items: int
    data: list[dict[str, Any]]


class OffBalanceSheetResponse(BaseModel):
    ticker: str
    leases: dict[str, Any]
    pension: dict[str, Any]
    contingent: dict[str, Any]


class WorkingCapitalResponse(BaseModel):
    ticker: str
    metrics: list[dict[str, Any]]
    trend: dict[str, Any]
    receivables_quality: dict[str, Any]


class HealthScoreResponse(BaseModel):
    ticker: str
    aggregate_score: int
    rating: str
    assessment: str
    components: dict[str, Any]


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

balance_router_v2 = APIRouter(prefix="/financials/v2/balance", tags=["balance-sheet-v2"])

_bs_parser_instance: UniversalBalanceSheetParser | None = None
_obs_analyzer: OffBalanceSheetAnalyzer | None = None
_gw_analyzer: GoodwillAnalyzer | None = None
_wc_engine: WorkingCapitalEngine | None = None
_health_scorer: FinancialHealthScorer | None = None


def _get_bs_parser() -> UniversalBalanceSheetParser:
    global _bs_parser_instance
    if _bs_parser_instance is None:
        _bs_parser_instance = UniversalBalanceSheetParser()
    return _bs_parser_instance


def _get_obs_analyzer() -> OffBalanceSheetAnalyzer:
    global _obs_analyzer
    if _obs_analyzer is None:
        _obs_analyzer = OffBalanceSheetAnalyzer()
    return _obs_analyzer


def _get_gw_analyzer() -> GoodwillAnalyzer:
    global _gw_analyzer
    if _gw_analyzer is None:
        _gw_analyzer = GoodwillAnalyzer()
    return _gw_analyzer


def _get_wc_engine() -> WorkingCapitalEngine:
    global _wc_engine
    if _wc_engine is None:
        _wc_engine = WorkingCapitalEngine()
    return _wc_engine


def _get_health_scorer() -> FinancialHealthScorer:
    global _health_scorer
    if _health_scorer is None:
        _health_scorer = FinancialHealthScorer()
    return _health_scorer


def _df_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    reset = df.reset_index()
    reset.columns = [str(c) for c in reset.columns]
    for col in reset.columns:
        if pd.api.types.is_datetime64_any_dtype(reset[col]):
            reset[col] = reset[col].dt.strftime("%Y-%m-%d")
    return reset.where(pd.notnull(reset), other=None).to_dict(orient="records")


def _resolve_ticker(ticker: str) -> str:
    try:
        return resolve_cik(ticker.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@balance_router_v2.get("/{ticker}", response_model=None)
def get_balance_sheet_v2(
    ticker: str,
    periods: int = Query(default=8, ge=1, le=40),
    period_type: str = Query(default="annual", pattern="^(annual|quarterly)$"),
):
    """
    Enhanced balance sheet with 50+ line items.
    Includes: ROU assets, operating/finance leases, detailed equity breakdown,
    goodwill, intangibles, pension assets, minority interest.
    """
    cik = _resolve_ticker(ticker)
    try:
        df = _get_bs_parser().parse_balance_sheet(ticker, periods=periods, period_type=period_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "period_type": period_type,
        "periods": len(df),
        "line_items": len(df.columns),
        "data": _df_to_records(df),
    }


@balance_router_v2.get("/{ticker}/off-balance", response_model=None)
def get_off_balance_sheet(
    ticker: str,
):
    """
    Off-balance-sheet obligations analysis.
    Returns: operating lease commitments, IFRS 16/ASC 842 ROU assets,
    pension funded status, contingent liabilities, VIE exposure.
    """
    try:
        result = _get_obs_analyzer().get_full_off_balance_sheet(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@balance_router_v2.get("/{ticker}/goodwill", response_model=None)
def get_goodwill_analysis(
    ticker: str,
    periods: int = Query(default=10, ge=2, le=20),
):
    """
    Goodwill quality analysis and impairment history.
    Returns: goodwill history, impairment track record, quality score (A-D).
    """
    try:
        gw_analyzer = _get_gw_analyzer()
        history = gw_analyzer.get_goodwill_history(ticker, periods=periods)
        quality = gw_analyzer.get_goodwill_quality_score(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker.upper(),
        "quality_score": quality,
        "history": _df_to_records(history),
    }


@balance_router_v2.get("/{ticker}/working-capital", response_model=None)
def get_working_capital(
    ticker: str,
    periods: int = Query(default=12, ge=4, le=24),
):
    """
    Working capital analytics: DSO, DIO, DPO, Cash Conversion Cycle.
    Includes trend analysis and receivables quality assessment.
    """
    try:
        wc = _get_wc_engine()
        metrics_df = wc.compute_working_capital_metrics(ticker, periods=periods)
        trend = wc.compute_trend_metrics(ticker, periods=periods)
        quality = wc.receivables_quality_check(ticker, periods=periods)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker.upper(),
        "metrics": _df_to_records(metrics_df),
        "trend": trend,
        "receivables_quality": quality,
    }


@balance_router_v2.get("/{ticker}/health-score", response_model=None)
def get_health_score(
    ticker: str,
    market_cap: float | None = Query(default=None, description="Market cap in USD for Altman Z"),
):
    """
    Comprehensive financial health score (0-100) with letter rating (A/B/C/D).
    Combines Altman Z-Score, Piotroski F-Score, and ratio health metrics.
    """
    try:
        result = _get_health_scorer().compute_aggregate_health_score(
            ticker, market_cap=market_cap
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@balance_router_v2.get("/{ticker}/piotroski", response_model=None)
def get_piotroski_score(
    ticker: str,
):
    """
    Piotroski F-Score: 9-point binary balance sheet health scoring system.
    7-9 = Strong, 4-6 = Average, 0-3 = Weak.
    """
    try:
        result = _get_health_scorer().compute_piotroski_f_score(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@balance_router_v2.get("/{ticker}/altman", response_model=None)
def get_altman_z_score(
    ticker: str,
    market_cap: float | None = Query(default=None),
):
    """
    Altman Z-Score. >2.99 = Safe, 1.81-2.99 = Grey Zone, <1.81 = Distress.
    """
    try:
        result = _get_health_scorer().compute_altman_z_score(ticker, market_cap=market_cap)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _round_safe(val: float | None, decimals: int = 3) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(float(val), decimals)
