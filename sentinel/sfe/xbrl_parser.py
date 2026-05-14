"""XBRL parser — extracts standardized financial facts from EDGAR companyfacts JSON."""
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, Iterator
from sentinel.core.types import FinancialFact
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Standard GAAP/IFRS concepts mapped to canonical labels
CONCEPT_LABELS: dict[str, str] = {
    # Income Statement
    "us-gaap:Revenues": "revenue",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "us-gaap:GrossProfit": "gross_profit",
    "us-gaap:OperatingIncomeLoss": "operating_income",
    "us-gaap:NetIncomeLoss": "net_income",
    "us-gaap:EarningsPerShareBasic": "eps_basic",
    "us-gaap:EarningsPerShareDiluted": "eps_diluted",
    "us-gaap:ResearchAndDevelopmentExpense": "rd_expense",
    "us-gaap:SellingGeneralAndAdministrativeExpense": "sga_expense",
    "us-gaap:DepreciationAndAmortization": "da",
    # Balance Sheet
    "us-gaap:Assets": "total_assets",
    "us-gaap:Liabilities": "total_liabilities",
    "us-gaap:StockholdersEquity": "stockholders_equity",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": "cash",
    "us-gaap:LongTermDebt": "long_term_debt",
    "us-gaap:CommonStockSharesOutstanding": "shares_outstanding",
    "us-gaap:RetainedEarningsAccumulatedDeficit": "retained_earnings",
    # Cash Flow
    "us-gaap:NetCashProvidedByUsedInOperatingActivities": "cfo",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities": "cfi",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities": "cff",
    "us-gaap:CapitalExpenditures": "capex",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "us-gaap:DividendsPaid": "dividends_paid",
    # Other
    "us-gaap:InterestExpense": "interest_expense",
    "us-gaap:IncomeTaxExpenseBenefit": "income_tax",
    "us-gaap:GoodwillAndIntangibleAssetsDisclosureTextBlock": "goodwill",
}


def extract_facts(
    companyfacts: dict,
    cik: str,
    figi: Optional[str] = None,
    concept_filter: Optional[set[str]] = None,
) -> list[FinancialFact]:
    """
    Extract FinancialFact records from an EDGAR companyfacts payload.
    companyfacts: raw JSON from /api/xbrl/companyfacts/CIK{cik}.json
    concept_filter: if provided, only extract these concept names.
    """
    facts: list[FinancialFact] = []
    facts_data = companyfacts.get("facts", {})

    for taxonomy, concepts in facts_data.items():
        for concept, concept_data in concepts.items():
            full_concept = f"{taxonomy}:{concept}"
            if concept_filter and concept not in concept_filter and full_concept not in concept_filter:
                continue

            label = CONCEPT_LABELS.get(full_concept, concept.lower())
            units_data = concept_data.get("units", {})

            for unit, observations in units_data.items():
                for obs in observations:
                    fact = _parse_observation(
                        obs, cik=cik, figi=figi,
                        concept=full_concept, label=label, unit=unit
                    )
                    if fact:
                        facts.append(fact)

    logger.info("XBRL facts extracted", cik=cik, count=len(facts))
    return facts


def _parse_observation(
    obs: dict,
    cik: str,
    figi: Optional[str],
    concept: str,
    label: str,
    unit: str,
) -> Optional[FinancialFact]:
    val = obs.get("val")
    if val is None:
        return None
    end_str = obs.get("end") or obs.get("instant")
    if not end_str:
        return None
    try:
        period_end = date.fromisoformat(end_str)
    except ValueError:
        return None

    start_str = obs.get("start")
    period_start = date.fromisoformat(start_str) if start_str else None

    form = obs.get("form", "")
    filed_str = obs.get("filed")
    filed = date.fromisoformat(filed_str) if filed_str else None

    return FinancialFact(
        cik=cik,
        figi=figi or "",
        concept=concept,
        label=label,
        value=Decimal(str(val)),
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        form_type=form,
        filed_date=filed,       # immutable timestamp — never backfilled after ingest
        accession=obs.get("accn"),
        frame=obs.get("frame"),
    )


def get_annual_facts(facts: list[FinancialFact], label: str) -> list[FinancialFact]:
    """Filter to annual (10-K) facts for a specific label, sorted by period_end desc."""
    return sorted(
        [f for f in facts if f.label == label and f.form in ("10-K", "20-F")],
        key=lambda f: f.period_end,
        reverse=True,
    )


def get_quarterly_facts(facts: list[FinancialFact], label: str) -> list[FinancialFact]:
    """Filter to quarterly (10-Q) facts for a specific label, sorted by period_end desc."""
    return sorted(
        [f for f in facts if f.label == label and f.form == "10-Q"],
        key=lambda f: f.period_end,
        reverse=True,
    )


def build_time_series(
    facts: list[FinancialFact], label: str, annual: bool = True
) -> list[tuple[date, Decimal]]:
    """Return (period_end, value) pairs for charting/analysis."""
    selected = get_annual_facts(facts, label) if annual else get_quarterly_facts(facts, label)
    return [(f.period_end, f.value) for f in selected]


def compute_ttm(quarterly_facts: list[FinancialFact]) -> Optional[Decimal]:
    """Sum last 4 quarterly observations for trailing-twelve-months."""
    last4 = sorted(quarterly_facts, key=lambda f: f.period_end, reverse=True)[:4]
    if len(last4) < 4:
        return None
    return sum(f.value for f in last4)
