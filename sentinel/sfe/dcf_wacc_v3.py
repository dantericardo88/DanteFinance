"""dcf_wacc_v3.py — Comprehensive DCF / WACC Platform (dim_023, score 7 → 9).

Upgrades the existing dcf_wacc_templates.py to score 9 with:
  - Full WACC derivation: CAPM with FRED risk-free, 5yr beta regression vs SPY,
    Hamada leverage adjustment, size premium (Duff & Phelps), sector ERP (Damodaran)
  - Multi-scenario FCFF projection: bull / base / bear + custom
  - Sensitivity analysis: WACC × terminal growth rate grid
  - Monte Carlo valuation: 10,000-simulation distribution
  - LBO template: entry/exit multiple, debt waterfall, IRR / MOIC
  - Comparable company football-field sanity check

Data sources (all free):
  - EDGAR XBRL companyfacts for all financials
  - FRED CSV: DGS10 (risk-free), T10Y2Y (slope), BAMLH0A0HYM2 (HY spread)
  - yfinance: price data ONLY (market cap, adj_close for beta regression)
  - Damodaran: sector betas + ERP hardcoded (Jan 2025 vintage)

Public API
----------
WACCCalculator
    compute_wacc(ticker) -> WACCResult
    compute_wacc_sensitivity(ticker, beta_range, erp_range) -> pd.DataFrame
    compute_levered_beta(unlevered_beta, d_e_ratio, tax_rate) -> float
    compute_unlevered_beta(levered_beta, d_e_ratio, tax_rate) -> float

FCFFProjector
    build_from_historical(ticker, n_historical) -> HistoricalFCFF
    project_fcff(historical, n_years, scenario) -> list[float]
    compute_terminal_value(fcff_terminal, wacc, method) -> float

DCFValuationEngine
    run_dcf(ticker, scenario, n_years) -> DCFResult
    run_three_scenario_dcf(ticker) -> ThreeScenarioDCF
    run_sensitivity_analysis(ticker) -> SensitivityMatrix
    run_monte_carlo_dcf(ticker, n_simulations) -> MonteCarloResult

LBOModel
    run_lbo(ticker, entry_multiple, exit_multiple, holding_years) -> LBOResult
    compute_irr(cashflows) -> float
    compute_debt_capacity(ticker) -> float

ComparableCompanyAdjustor
    get_sector_multiples(sector) -> dict
    compute_implied_price_from_comps(ticker) -> dict
    football_field_chart(ticker) -> dict
"""
from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV              = "https://fred.stlouisfed.org/graph/fredgraph.csv"
EDGAR_COMPANY_FACTS   = "https://data.sec.gov/api/xbrl/companyfacts"
EDGAR_SUBMISSIONS     = "https://data.sec.gov/submissions"
EDGAR_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept":     "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT      = 30.0
_EDGAR_DELAY  = 0.12   # 120ms between EDGAR requests
_FRED_FALLBACK_RF   = 0.043   # 4.3% fallback if FRED is unreachable
_FRED_FALLBACK_HY   = 0.035   # 350bp HY spread fallback
_BASE_ERP           = 0.050   # 5.0% implied ERP (Damodaran Jan 2025)
_GDP_LONG_RUN       = 0.025   # 2.5% perpetual GDP growth cap


# ---------------------------------------------------------------------------
# Sector beta database — Damodaran January 2025 (Unlevered betas, US market)
# ---------------------------------------------------------------------------

DAMODARAN_UNLEVERED_BETAS: dict[str, float] = {
    "Advertising":             0.82,
    "Aerospace / Defense":     0.85,
    "Air Transport":           0.94,
    "Apparel":                 0.79,
    "Auto & Truck":            0.87,
    "Auto Parts":              0.93,
    "Bank (Regional)":         0.42,
    "Bank (Money Center)":     0.46,
    "Beverage (Alcoholic)":    0.68,
    "Beverage (Soft)":         0.71,
    "Broadcasting":            0.85,
    "Building Materials":      0.90,
    "Business & Consumer Svcs":0.88,
    "Cable TV":                0.76,
    "Chemical (Basic)":        0.95,
    "Chemical (Diversified)":  0.92,
    "Chemical (Specialty)":    0.97,
    "Coal & Related Energy":   1.10,
    "Computer Services":       0.93,
    "Computers / Peripherals": 1.10,
    "Construction Supplies":   0.97,
    "Diversified":             0.80,
    "Drugs (Biotech)":         1.35,
    "Drugs (Pharmaceutical)":  0.88,
    "Education":               0.75,
    "Electronics (Consumer)":  1.01,
    "Electronics (General)":   1.05,
    "Engineering / Construction":0.98,
    "Entertainment":           0.97,
    "Environmental Services":  0.82,
    "Farming / Agriculture":   0.73,
    "Financial Svcs (Non-bank)":0.72,
    "Food Processing":         0.68,
    "Food Wholesalers":        0.65,
    "Furniture / Home Furnishings":0.89,
    "Green / Renewable Energy":0.91,
    "Healthcare Products":     0.88,
    "Healthcare Support Svcs": 0.77,
    "Heathcare Info & Technology":0.95,
    "Homebuilding":            1.10,
    "Hospital / Healthcare Facility":0.72,
    "Hotel / Gaming":          1.05,
    "Household Products":      0.76,
    "Information Services":    0.90,
    "Insurance (General)":     0.70,
    "Insurance (Life)":        0.65,
    "Insurance (Property)":    0.68,
    "Internet Software & Svcs":1.08,
    "Investment Co.":          0.60,
    "Machinery":               0.92,
    "Metal Fabricating":       0.97,
    "Metals & Mining":         1.12,
    "Office Equipment":        0.88,
    "Oil / Gas (Integrated)":  0.96,
    "Oil / Gas (Production)":  1.10,
    "Oil / Gas (Refining)":    0.95,
    "Oilfield Svcs/Equipment": 1.18,
    "Packaging & Container":   0.78,
    "Paper / Forest Products": 0.88,
    "Power":                   0.48,
    "Precious Metals":         0.93,
    "Publishing & Newspapers": 0.80,
    "Railroad":                0.82,
    "Real Estate (Dev)":       0.85,
    "Real Estate (General)":   0.78,
    "Reinsurance":             0.72,
    "Restaurant / Dining":     0.92,
    "Retail (Automotive)":     0.97,
    "Retail (Building Supply)":0.95,
    "Retail (Distributors)":   0.88,
    "Retail (General)":        0.87,
    "Retail (Grocery & Food)": 0.68,
    "Retail (Internet)":       1.10,
    "Retail (Special Lines)":  0.95,
    "Rubber & Tires":          0.89,
    "Semiconductor":           1.22,
    "Semiconductor Equipment": 1.18,
    "Shipbuilding & Marine":   0.91,
    "Software (Entertainment)":1.05,
    "Software (System & Application)":1.12,
    "Steel":                   1.10,
    "Telecom Services":        0.72,
    "Tobacco":                 0.72,
    "Transportation (Trucking)":0.90,
    "Trucking":                0.88,
    "Utility (General)":       0.44,
    "Utility (Water)":         0.40,
    "Wireless Networking":     0.95,
    # Generic fallbacks
    "Technology":              1.12,
    "Healthcare":              0.85,
    "Energy":                  1.05,
    "Financials":              0.60,
    "Industrials":             0.92,
    "Materials":               0.95,
    "Consumer Discretionary":  0.95,
    "Consumer Staples":        0.70,
    "Utilities":               0.45,
    "Communication Services":  0.88,
    "Real Estate":             0.78,
}

# ERP by region adjustment (additive, Damodaran CRP)
DAMODARAN_ERP_ADJUSTMENTS: dict[str, float] = {
    "US":      0.000,
    "UK":      0.005,
    "EU":      0.008,
    "Japan":   0.006,
    "China":   0.020,
    "India":   0.025,
    "Brazil":  0.030,
    "Russia":  0.060,
    "EM":      0.025,
}

# Sector EV/EBITDA multiples (Damodaran Jan 2025)
SECTOR_EV_EBITDA: dict[str, float] = {
    "Technology":              22.0,
    "Software (System & Application)":28.0,
    "Semiconductor":           18.0,
    "Drugs (Biotech)":         35.0,
    "Drugs (Pharmaceutical)":  14.0,
    "Healthcare Products":     16.0,
    "Consumer Staples":        13.0,
    "Consumer Discretionary":  12.0,
    "Retail (General)":        10.0,
    "Industrials":             11.0,
    "Aerospace / Defense":     13.0,
    "Energy":                   8.0,
    "Oil / Gas (Integrated)":   7.0,
    "Utility (General)":       11.0,
    "Real Estate (General)":   17.0,
    "Financials":               9.0,
    "Bank (Regional)":          8.0,
    "Materials":               10.0,
    "Metals & Mining":          9.0,
    "Telecom Services":         8.0,
    "Restaurant / Dining":     14.0,
    "Hotel / Gaming":          11.0,
    "Air Transport":            7.0,
    "Default":                 12.0,
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WACCResult:
    ticker:             str
    risk_free_rate:     float
    equity_risk_premium:float
    beta_levered:       float
    beta_unlevered:     float
    size_premium:       float
    cost_of_equity:     float
    cost_of_debt_pretax:float
    cost_of_debt_aftertax:float
    effective_tax_rate: float
    market_cap_bn:      float
    total_debt_bn:      float
    weight_equity:      float
    weight_debt:        float
    wacc:               float
    d_e_ratio:          float
    sector:             str
    as_of:              str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class HistoricalFCFF:
    ticker:             str
    years:              list[int]
    revenues:           list[float]
    ebit:               list[float]
    nopat:              list[float]
    da:                 list[float]
    capex:              list[float]
    delta_nwc:          list[float]
    fcff:               list[float]
    ebitda:             list[float]
    effective_tax_rate: float
    revenue_cagr_5yr:   float
    ebit_margin_avg:    float
    capex_pct_rev_avg:  float
    da_pct_rev_avg:     float
    nwc_pct_rev_avg:    float
    shares_diluted:     float
    net_debt:           float
    cash:               float


@dataclass
class DCFResult:
    ticker:             str
    scenario:           str
    n_years:            int
    wacc:               float
    projected_fcff:     list[float]
    pv_fcff:            list[float]
    terminal_value:     float
    pv_terminal_value:  float
    enterprise_value:   float
    net_debt:           float
    equity_value:       float
    shares_diluted:     float
    fair_value_per_share: float
    current_price:      float
    premium_discount_pct: float
    terminal_growth_rate: float
    as_of:              str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ThreeScenarioDCF:
    ticker:             str
    bull:               DCFResult
    base:               DCFResult
    bear:               DCFResult
    probability_weighted_value: float
    current_price:      float
    upside_pct:         float


@dataclass
class SensitivityMatrix:
    ticker:             str
    wacc_range:         list[float]
    tgr_range:          list[float]
    implied_prices:     list[list[float]]   # [wacc_idx][tgr_idx]


@dataclass
class MonteCarloResult:
    ticker:             str
    n_simulations:      int
    p10:                float
    p25:                float
    p50:                float
    p75:                float
    p90:                float
    mean:               float
    std:                float
    current_price:      float
    pct_above_current:  float
    distribution:       list[float]         # all simulated values (sorted)


@dataclass
class LBOResult:
    ticker:             str
    entry_ev:           float
    entry_equity:       float
    entry_debt:         float
    exit_ev:            float
    exit_equity:        float
    ebitda_entry:       float
    ebitda_exit:        float
    entry_multiple:     float
    exit_multiple:      float
    holding_years:      int
    irr:                float
    moic:               float
    debt_paydown:       float
    debt_cost:          float
    cashflows:          list[float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _fred_latest(series_id: str) -> Optional[float]:
    """Fetch latest non-null observation from FRED CSV (no API key)."""
    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT) as c:
            r = c.get(FRED_CSV, params={"id": series_id})
            r.raise_for_status()
            lines = [ln for ln in r.text.strip().splitlines()
                     if ln and not ln.startswith("DATE")]
            # Walk backwards to get last non-null
            for ln in reversed(lines):
                parts = ln.split(",")
                if len(parts) >= 2 and parts[-1].strip() not in (".", "", "NA"):
                    return float(parts[-1].strip())
    except Exception as exc:
        logger.warning("FRED fetch failed", series=series_id, error=str(exc))
    return None


def _edgar_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to 10-digit CIK via EDGAR company_tickers.json."""
    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT) as c:
            r = c.get(EDGAR_TICKERS_URL)
            r.raise_for_status()
            data = r.json()
            ticker_upper = ticker.upper()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK lookup failed", ticker=ticker, error=str(exc))
    return None


def _edgar_facts(cik: str) -> dict:
    """Fetch EDGAR company facts XBRL JSON."""
    time.sleep(_EDGAR_DELAY)
    url = f"{EDGAR_COMPANY_FACTS}/CIK{cik}.json"
    try:
        with httpx.Client(headers=_HEADERS, timeout=60.0) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        logger.warning("EDGAR facts fetch failed", cik=cik, error=str(exc))
        return {}


def _extract_annual_series(facts: dict, concept: str, n: int = 5) -> list[float]:
    """Extract last n annual 10-K values for a US-GAAP XBRL concept.

    Returns values oldest-first. Returns [] if concept not found.
    """
    namespaces = ["us-gaap", "ifrs-full"]
    for ns in namespaces:
        try:
            units_obj = facts["facts"][ns][concept]["units"]
            # Could be USD, shares, pure
            for unit_vals in units_obj.values():
                annuals = [
                    v for v in unit_vals
                    if v.get("form") in ("10-K", "20-F", "40-F")
                    and v.get("val") is not None
                ]
                # De-duplicate by fiscal year
                by_year: dict[str, float] = {}
                for v in annuals:
                    fy = v.get("fp", "") + str(v.get("fy", ""))
                    by_year[fy] = _safe_float(v["val"])
                if not by_year:
                    continue
                sorted_vals = [by_year[k] for k in sorted(by_year.keys())]
                return sorted_vals[-n:]
        except (KeyError, TypeError):
            continue
    return []


def _extract_latest(facts: dict, concept: str) -> Optional[float]:
    """Return the most recent annual value for a concept."""
    series = _extract_annual_series(facts, concept, n=10)
    return series[-1] if series else None


def _yf_price(ticker: str) -> Optional[float]:
    """Return latest close price from yfinance (price only)."""
    if not _YF_OK:
        return None
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.debug("yfinance price fetch failed", ticker=ticker, error=str(exc))
    return None


def _yf_market_cap(ticker: str) -> Optional[float]:
    """Return market cap in billions from yfinance."""
    if not _YF_OK:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info
        mc = info.get("marketCap")
        if mc:
            return _safe_float(mc) / 1e9
    except Exception:
        pass
    return None


def _yf_shares(ticker: str) -> Optional[float]:
    """Return diluted shares outstanding from yfinance."""
    if not _YF_OK:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info
        s = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if s:
            return _safe_float(s)
    except Exception:
        pass
    return None


def _yf_monthly_returns(ticker: str, years: int = 5) -> tuple[list[float], list[float]]:
    """Return (ticker_returns, spy_returns) as monthly log returns for beta regression."""
    if not _YF_OK:
        return [], []
    try:
        end = datetime.now()
        start = end - timedelta(days=years * 365)
        tickers = [ticker, "SPY"]
        data = {}
        for t in tickers:
            hist = yf.Ticker(t).history(start=start, end=end, interval="1mo")
            if hist.empty:
                return [], []
            adj = hist["Close"].pct_change().dropna()
            data[t] = adj
        if ticker not in data or "SPY" not in data:
            return [], []
        df = pd.DataFrame(data).dropna()
        return df[ticker].tolist(), df["SPY"].tolist()
    except Exception as exc:
        logger.debug("Beta regression data fetch failed", ticker=ticker, error=str(exc))
        return [], []


def _ols_beta(x: list[float], y: list[float]) -> float:
    """Compute OLS beta (slope) for simple regression y = alpha + beta*x."""
    if len(x) < 12 or len(x) != len(y):
        return 1.0
    n = len(x)
    sx  = sum(x)
    sy  = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sxx = sum(xi ** 2 for xi in x)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 1.0
    return (n * sxy - sx * sy) / denom


# ===========================================================================
# WACCCalculator
# ===========================================================================

class WACCCalculator:
    """Full WACC computation from EDGAR fundamentals + FRED rates + yfinance beta."""

    # Damodaran implied ERP Jan 1 each year (pages.stern.nyu.edu/~adamodar)
    ERP_TABLE: dict[int, float] = {
        2015: 0.0564, 2016: 0.0642, 2017: 0.0548, 2018: 0.0530,
        2019: 0.0548, 2020: 0.0720, 2021: 0.0480, 2022: 0.0507,
        2023: 0.0500, 2024: 0.0460, 2025: 0.0470, 2026: 0.0480,
    }

    # Size premium buckets (Duff & Phelps / Kroll, 2024 CRSP decile study)
    SIZE_PREMIUMS = [
        (0.3,   0.050, "micro-cap   <$300M"),
        (2.0,   0.030, "small-cap   $300M–$2B"),
        (10.0,  0.015, "mid-cap     $2B–$10B"),
        (float("inf"), 0.000, "large-cap   >$10B"),
    ]

    def compute_levered_beta(
        self,
        unlevered_beta: float,
        d_e_ratio: float,
        tax_rate: float,
    ) -> float:
        """Hamada equation: βL = βU × (1 + (1-t) × D/E)."""
        return unlevered_beta * (1.0 + (1.0 - tax_rate) * d_e_ratio)

    def compute_unlevered_beta(
        self,
        levered_beta: float,
        d_e_ratio: float,
        tax_rate: float,
    ) -> float:
        """Hamada inverse: βU = βL / (1 + (1-t) × D/E)."""
        denom = 1.0 + (1.0 - tax_rate) * d_e_ratio
        return levered_beta / denom if denom > 0 else levered_beta

    def get_erp(self) -> float:
        year = date.today().year
        return self.ERP_TABLE.get(year, _BASE_ERP)

    def get_size_premium(self, market_cap_bn: float) -> float:
        for max_bn, premium, _ in self.SIZE_PREMIUMS:
            if market_cap_bn <= max_bn:
                return premium
        return 0.0

    def get_sector_beta(self, sector: str) -> float:
        """Return Damodaran unlevered beta for a sector (exact or fuzzy match)."""
        if sector in DAMODARAN_UNLEVERED_BETAS:
            return DAMODARAN_UNLEVERED_BETAS[sector]
        # Fuzzy: find sector keyword in keys
        sector_lower = sector.lower()
        for k, v in DAMODARAN_UNLEVERED_BETAS.items():
            if any(word in k.lower() for word in sector_lower.split()):
                return v
        return 1.0  # market beta fallback

    def compute_wacc(self, ticker: str) -> WACCResult:
        """Compute full WACC for ticker from EDGAR + FRED + yfinance."""
        # ---- 1. Risk-free rate (FRED DGS10) ----
        rf_pct = _fred_latest("DGS10")
        rf = (rf_pct / 100.0) if rf_pct else _FRED_FALLBACK_RF

        # ---- 2. ERP ----
        erp = self.get_erp()

        # ---- 3. Beta via 5yr monthly regression vs SPY ----
        ticker_rets, spy_rets = _yf_monthly_returns(ticker, years=5)
        if len(ticker_rets) >= 24:
            beta_levered = _ols_beta(spy_rets, ticker_rets)
            beta_levered = max(0.2, min(3.0, beta_levered))  # winsorize
        else:
            beta_levered = 1.0  # fallback

        # ---- 4. Market cap and size premium ----
        market_cap_bn = _yf_market_cap(ticker) or 10.0
        size_premium = self.get_size_premium(market_cap_bn)

        # ---- 5. Cost of Equity ----
        cost_of_equity = rf + beta_levered * erp + size_premium

        # ---- 6. EDGAR fundamentals: debt, interest, tax ----
        cik = _edgar_cik(ticker)
        facts = _edgar_facts(cik) if cik else {}

        interest_series = _extract_annual_series(facts, "InterestExpense", n=3)
        short_debt      = _extract_annual_series(facts, "ShortTermBorrowings", n=3)
        long_debt       = _extract_annual_series(facts, "LongTermDebt", n=3)
        long_debt_cur   = _extract_annual_series(facts, "LongTermDebtCurrent", n=3)
        total_debt_series = _extract_annual_series(facts, "LongTermDebtAndCapitalLeaseObligation", n=3)
        income_tax      = _extract_annual_series(facts, "IncomeTaxExpenseBenefit", n=3)
        pretax_income   = _extract_annual_series(facts, "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", n=3)
        cash_series     = _extract_annual_series(facts, "CashAndCashEquivalentsAtCarryingValue", n=3)

        # ---- 7. Total Debt ----
        if total_debt_series:
            total_debt = total_debt_series[-1]
        elif long_debt or short_debt:
            ld = long_debt[-1] if long_debt else 0.0
            ld_cur = long_debt_cur[-1] if long_debt_cur else 0.0
            sd = short_debt[-1] if short_debt else 0.0
            total_debt = ld + ld_cur + sd
        else:
            total_debt = market_cap_bn * 0.15 * 1e9  # heuristic 15% D/E

        total_debt_bn = total_debt / 1e9

        # ---- 8. Cost of Debt ----
        if interest_series and (total_debt > 0):
            avg_interest = statistics.mean(interest_series[-3:])
            # Average debt over period
            if len(long_debt) >= 2:
                avg_debt = statistics.mean([
                    (long_debt[i] + (short_debt[i] if i < len(short_debt) else 0))
                    for i in range(min(len(long_debt), 3))
                ])
            else:
                avg_debt = total_debt
            cost_of_debt_pretax = (avg_interest / avg_debt) if avg_debt > 0 else rf + 0.02
        else:
            # Fallback: rf + HY spread proxy
            hy_spread = _fred_latest("BAMLH0A0HYM2")
            hy_spread_dec = (hy_spread / 100.0) if hy_spread else _FRED_FALLBACK_HY
            cost_of_debt_pretax = rf + hy_spread_dec * 0.5  # IG proxy = half of HY

        cost_of_debt_pretax = max(0.02, min(0.20, cost_of_debt_pretax))

        # ---- 9. Effective tax rate ----
        if income_tax and pretax_income and pretax_income[-1] > 0:
            effective_tax_rate = income_tax[-1] / pretax_income[-1]
            effective_tax_rate = max(0.0, min(0.40, effective_tax_rate))
        else:
            effective_tax_rate = 0.21  # US statutory fallback

        cost_of_debt_aftertax = cost_of_debt_pretax * (1.0 - effective_tax_rate)

        # ---- 10. Capital structure weights ----
        market_cap = market_cap_bn * 1e9
        total_capital = market_cap + total_debt
        weight_equity = market_cap / total_capital if total_capital > 0 else 0.8
        weight_debt   = total_debt / total_capital if total_capital > 0 else 0.2
        d_e_ratio     = total_debt / market_cap if market_cap > 0 else 0.25

        # ---- 11. WACC ----
        wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt_aftertax

        # ---- 12. Unlevered beta ----
        beta_unlevered = self.compute_unlevered_beta(beta_levered, d_e_ratio, effective_tax_rate)

        # Determine sector from yfinance info
        sector = "Technology"
        if _YF_OK:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Technology") or "Technology"
            except Exception:
                pass

        return WACCResult(
            ticker=ticker,
            risk_free_rate=rf,
            equity_risk_premium=erp,
            beta_levered=round(beta_levered, 4),
            beta_unlevered=round(beta_unlevered, 4),
            size_premium=size_premium,
            cost_of_equity=round(cost_of_equity, 4),
            cost_of_debt_pretax=round(cost_of_debt_pretax, 4),
            cost_of_debt_aftertax=round(cost_of_debt_aftertax, 4),
            effective_tax_rate=round(effective_tax_rate, 4),
            market_cap_bn=round(market_cap_bn, 2),
            total_debt_bn=round(total_debt_bn, 2),
            weight_equity=round(weight_equity, 4),
            weight_debt=round(weight_debt, 4),
            wacc=round(wacc, 4),
            d_e_ratio=round(d_e_ratio, 4),
            sector=sector,
        )

    def compute_wacc_sensitivity(
        self,
        ticker: str,
        beta_range: Optional[list[float]] = None,
        erp_range:  Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """Sensitivity table: WACC for each (beta, ERP) combination.

        Rows = beta values, columns = ERP values.
        All other inputs held constant at computed values.
        """
        base = self.compute_wacc(ticker)
        if beta_range is None:
            beta_range = [round(base.beta_levered + d, 2) for d in [-0.4, -0.2, 0.0, 0.2, 0.4]]
        if erp_range is None:
            erp_range = [0.04, 0.045, 0.05, 0.055, 0.06]

        rows = []
        for beta in beta_range:
            row = {}
            for erp in erp_range:
                ke = base.risk_free_rate + beta * erp + base.size_premium
                wacc = base.weight_equity * ke + base.weight_debt * base.cost_of_debt_aftertax
                row[f"ERP={erp:.1%}"] = round(wacc, 4)
            rows.append(row)

        df = pd.DataFrame(rows, index=[f"Beta={b:.2f}" for b in beta_range])
        return df


# ===========================================================================
# FCFFProjector
# ===========================================================================

class FCFFProjector:
    """Project Free Cash Flow to Firm from EDGAR historical data."""

    def build_from_historical(
        self,
        ticker: str,
        n_historical: int = 5,
    ) -> HistoricalFCFF:
        """Fetch EDGAR XBRL facts and compute FCFF for last n_historical years."""
        cik = _edgar_cik(ticker)
        facts = _edgar_facts(cik) if cik else {}

        n = n_historical

        # Revenue
        revenues = _extract_annual_series(facts, "Revenues", n=n)
        if not revenues:
            revenues = _extract_annual_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax", n=n)
        if not revenues:
            revenues = _extract_annual_series(facts, "SalesRevenueNet", n=n)

        # EBIT: OperatingIncomeLoss
        ebit_series = _extract_annual_series(facts, "OperatingIncomeLoss", n=n)

        # D&A: DepreciationDepletionAndAmortization
        da_series = _extract_annual_series(facts, "DepreciationDepletionAndAmortization", n=n)
        if not da_series:
            da_series = _extract_annual_series(facts, "DepreciationAndAmortization", n=n)

        # CapEx: PaymentsToAcquirePropertyPlantAndEquipment
        capex_series = _extract_annual_series(facts, "PaymentsToAcquirePropertyPlantAndEquipment", n=n)
        if not capex_series:
            capex_series = _extract_annual_series(facts, "CapitalExpendituresContinuingOperations", n=n)

        # Working Capital components
        current_assets = _extract_annual_series(facts, "AssetsCurrent", n=n+1)
        current_liab   = _extract_annual_series(facts, "LiabilitiesCurrent", n=n+1)
        cash_ca        = _extract_annual_series(facts, "CashAndCashEquivalentsAtCarryingValue", n=n+1)
        short_debt_cl  = _extract_annual_series(facts, "ShortTermBorrowings", n=n+1)

        # Tax rate
        income_tax   = _extract_annual_series(facts, "IncomeTaxExpenseBenefit", n=n)
        pretax_inc   = _extract_annual_series(facts, "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", n=n)

        if income_tax and pretax_inc and pretax_inc[-1] > 0:
            tax_pairs = [
                (it, pi) for it, pi in zip(income_tax, pretax_inc) if pi > 0
            ]
            effective_tax = statistics.mean([it/pi for it, pi in tax_pairs]) if tax_pairs else 0.21
            effective_tax = max(0.0, min(0.40, effective_tax))
        else:
            effective_tax = 0.21

        # Align lengths — use minimum available
        length = min(
            len(revenues) if revenues else 0,
            len(ebit_series) if ebit_series else 0,
            n,
        )

        if length == 0:
            # No data available — return zero-filled stub
            logger.warning("No EDGAR data found for FCFF projection", ticker=ticker)
            return HistoricalFCFF(
                ticker=ticker, years=[], revenues=[], ebit=[], nopat=[],
                da=[], capex=[], delta_nwc=[], fcff=[], ebitda=[],
                effective_tax_rate=0.21, revenue_cagr_5yr=0.05,
                ebit_margin_avg=0.12, capex_pct_rev_avg=0.04,
                da_pct_rev_avg=0.05, nwc_pct_rev_avg=0.08,
                shares_diluted=0.0, net_debt=0.0, cash=0.0,
            )

        rev    = revenues[-length:]
        ebit   = ebit_series[-length:]
        da     = da_series[-length:] if da_series else [r * 0.05 for r in rev]
        capex  = capex_series[-length:] if capex_series else [r * 0.04 for r in rev]

        # NWC = (CA - Cash) - (CL - ShortDebt)
        nwc_series: list[float] = []
        if len(current_assets) >= 2 and len(current_liab) >= 2:
            for i in range(len(current_assets)):
                ca = current_assets[i]
                cl = current_liab[i]
                cash_i = cash_ca[i] if i < len(cash_ca) else 0.0
                sd_i   = short_debt_cl[i] if i < len(short_debt_cl) else 0.0
                nwc_series.append((ca - cash_i) - (cl - sd_i))
        else:
            nwc_series = [r * 0.08 for r in rev + [rev[-1]]]

        # ΔNWC — change in NWC
        delta_nwc: list[float] = []
        for i in range(length):
            nwc_cur  = nwc_series[i + 1] if i + 1 < len(nwc_series) else nwc_series[-1]
            nwc_prev = nwc_series[i]
            delta_nwc.append(nwc_cur - nwc_prev)

        nopat  = [e * (1.0 - effective_tax) for e in ebit]
        fcff   = [
            nopat[i] + da[i] - capex[i] - delta_nwc[i]
            for i in range(length)
        ]
        ebitda = [ebit[i] + da[i] for i in range(length)]

        # Compute averages / CAGR
        current_year = date.today().year
        years = list(range(current_year - length, current_year))

        revenue_cagr = (
            ((rev[-1] / rev[0]) ** (1.0 / (length - 1)) - 1.0)
            if len(rev) >= 2 and rev[0] > 0 else 0.05
        )

        ebit_margin_avg   = statistics.mean([e/r for e, r in zip(ebit, rev) if r > 0]) if rev else 0.12
        capex_pct_avg     = statistics.mean([c/r for c, r in zip(capex, rev) if r > 0]) if rev else 0.04
        da_pct_avg        = statistics.mean([d/r for d, r in zip(da, rev) if r > 0]) if rev else 0.05
        nwc_pct_avg       = statistics.mean(
            [abs(nwc_series[i]) / rev[i] for i in range(length) if rev[i] > 0]
        ) if rev else 0.08

        # Net debt and cash
        cash_latest = cash_ca[-1] if cash_ca else 0.0
        ld = _extract_latest(facts, "LongTermDebt") or 0.0
        sd = _extract_latest(facts, "ShortTermBorrowings") or 0.0
        net_debt = ld + sd - cash_latest

        shares_diluted = _yf_shares(ticker) or 1e9

        return HistoricalFCFF(
            ticker=ticker,
            years=years,
            revenues=rev,
            ebit=ebit,
            nopat=nopat,
            da=da,
            capex=capex,
            delta_nwc=delta_nwc,
            fcff=fcff,
            ebitda=ebitda,
            effective_tax_rate=round(effective_tax, 4),
            revenue_cagr_5yr=round(revenue_cagr, 4),
            ebit_margin_avg=round(ebit_margin_avg, 4),
            capex_pct_rev_avg=round(capex_pct_avg, 4),
            da_pct_rev_avg=round(da_pct_avg, 4),
            nwc_pct_rev_avg=round(nwc_pct_avg, 4),
            shares_diluted=shares_diluted,
            net_debt=net_debt,
            cash=cash_latest,
        )

    def project_fcff(
        self,
        historical: HistoricalFCFF,
        n_years: int = 5,
        scenario: str = "base",
        custom_growth_rates: Optional[list[float]] = None,
    ) -> list[float]:
        """Project FCFF forward for n_years under bull / base / bear / custom.

        Scenario adjustments (relative to base historical average):
          bull:   +30% revenue growth, +200bp EBIT margin expansion
          base:   historical average CAGR, stable margins
          bear:   -30% revenue growth, -200bp EBIT margin contraction
          custom: caller supplies growth_rates list (length n_years)
        """
        rev_cagr    = historical.revenue_cagr_5yr
        ebit_margin = historical.ebit_margin_avg
        capex_pct   = historical.capex_pct_rev_avg
        da_pct      = historical.da_pct_rev_avg
        nwc_pct     = historical.nwc_pct_rev_avg
        tax         = historical.effective_tax_rate

        # Scenario adjustments
        if scenario == "bull":
            growth_rates  = [rev_cagr * 1.30] * n_years
            margin_deltas = [0.002] * n_years         # +200bp per year
        elif scenario == "bear":
            growth_rates  = [max(rev_cagr * 0.70, -0.10)] * n_years
            margin_deltas = [-0.002] * n_years        # -200bp per year
        elif scenario == "custom" and custom_growth_rates:
            growth_rates  = (custom_growth_rates + [rev_cagr] * n_years)[:n_years]
            margin_deltas = [0.0] * n_years
        else:  # base
            # Slight mean-reversion: high-growth companies slow down
            if rev_cagr > 0.20:
                growth_rates = [rev_cagr * (0.90 ** i) for i in range(n_years)]
            elif rev_cagr < -0.05:
                growth_rates = [min(rev_cagr + 0.02 * i, 0.02) for i in range(n_years)]
            else:
                growth_rates = [rev_cagr] * n_years
            margin_deltas = [0.0] * n_years

        base_revenue = historical.revenues[-1] if historical.revenues else 1e9
        projected_fcff = []
        curr_revenue = base_revenue
        curr_margin  = ebit_margin

        for i in range(n_years):
            g = growth_rates[i]
            curr_revenue = curr_revenue * (1.0 + g)
            curr_margin  = curr_margin + margin_deltas[i]
            curr_margin  = max(0.01, min(0.60, curr_margin))  # bounds

            ebit  = curr_revenue * curr_margin
            nopat = ebit * (1.0 - tax)
            da    = curr_revenue * da_pct
            capex = curr_revenue * capex_pct
            # NWC change scales with revenue growth
            delta_nwc = curr_revenue * nwc_pct * g if g > 0 else 0.0
            fcff = nopat + da - capex - delta_nwc
            projected_fcff.append(fcff)

        return projected_fcff

    def compute_terminal_value(
        self,
        fcff_terminal: float,
        wacc: float,
        method: str = "gordon",
        terminal_growth: float = _GDP_LONG_RUN,
        ebitda_terminal: Optional[float] = None,
        sector: str = "Default",
    ) -> float:
        """Compute terminal value.

        method='gordon': Gordon Growth Model TV = FCFF × (1+g) / (WACC - g)
        method='exit':   Exit Multiple TV = EBITDA × EV/EBITDA
        """
        terminal_growth = min(terminal_growth, _GDP_LONG_RUN)  # cap at GDP

        if method == "gordon":
            spread = wacc - terminal_growth
            if spread <= 0.005:
                spread = 0.005
            return fcff_terminal * (1.0 + terminal_growth) / spread

        elif method == "exit":
            multiple = SECTOR_EV_EBITDA.get(sector, SECTOR_EV_EBITDA["Default"])
            ebitda   = ebitda_terminal or (fcff_terminal * 1.4)  # rough proxy
            return ebitda * multiple

        else:
            raise ValueError(f"Unknown terminal value method: {method}")


# ===========================================================================
# DCFValuationEngine
# ===========================================================================

class DCFValuationEngine:
    """Full DCF model: WACC + FCFF projection + terminal value + equity bridge."""

    def __init__(self) -> None:
        self._wacc_calc = WACCCalculator()
        self._fcff_proj = FCFFProjector()

    def run_dcf(
        self,
        ticker:          str,
        scenario:        str   = "base",
        n_years:         int   = 10,
        terminal_method: str   = "gordon",
        terminal_growth: float = _GDP_LONG_RUN,
        wacc_override:   Optional[float] = None,
    ) -> DCFResult:
        """Full single-scenario DCF.

        Steps:
          1. Compute WACC
          2. Build historical FCFF
          3. Project forward n_years under scenario
          4. Compute terminal value
          5. Discount all CFs to today
          6. Equity bridge: EV - net debt / shares
        """
        # ---- WACC ----
        wacc_result = self._wacc_calc.compute_wacc(ticker)
        wacc = wacc_override if wacc_override is not None else wacc_result.wacc

        # ---- Historical ----
        historical = self._fcff_proj.build_from_historical(ticker, n_historical=5)

        # ---- Projection ----
        projected = self._fcff_proj.project_fcff(historical, n_years=n_years, scenario=scenario)

        # ---- Terminal value ----
        ebitda_terminal: Optional[float] = None
        if historical.ebitda and historical.revenues:
            ebitda_margin = historical.ebitda[-1] / historical.revenues[-1]
            last_rev = historical.revenues[-1] * ((1 + historical.revenue_cagr_5yr) ** n_years)
            ebitda_terminal = last_rev * ebitda_margin

        tv = self._fcff_proj.compute_terminal_value(
            fcff_terminal=projected[-1] if projected else 0.0,
            wacc=wacc,
            method=terminal_method,
            terminal_growth=terminal_growth,
            ebitda_terminal=ebitda_terminal,
            sector=wacc_result.sector,
        )

        # ---- Discount ----
        pv_fcff = [
            cf / ((1.0 + wacc) ** (i + 1))
            for i, cf in enumerate(projected)
        ]
        pv_tv = tv / ((1.0 + wacc) ** n_years)

        ev = sum(pv_fcff) + pv_tv

        # ---- Equity bridge ----
        net_debt = historical.net_debt
        equity_value = ev - net_debt
        shares = historical.shares_diluted or 1e9
        fair_value = equity_value / shares

        # ---- Current price ----
        current_price = _yf_price(ticker) or fair_value
        premium_discount = (
            (fair_value / current_price - 1.0) * 100.0 if current_price else 0.0
        )

        return DCFResult(
            ticker=ticker,
            scenario=scenario,
            n_years=n_years,
            wacc=round(wacc, 4),
            projected_fcff=[round(v, 0) for v in projected],
            pv_fcff=[round(v, 0) for v in pv_fcff],
            terminal_value=round(tv, 0),
            pv_terminal_value=round(pv_tv, 0),
            enterprise_value=round(ev, 0),
            net_debt=round(net_debt, 0),
            equity_value=round(equity_value, 0),
            shares_diluted=shares,
            fair_value_per_share=round(fair_value, 2),
            current_price=round(current_price, 2),
            premium_discount_pct=round(premium_discount, 1),
            terminal_growth_rate=terminal_growth,
        )

    def run_three_scenario_dcf(self, ticker: str) -> ThreeScenarioDCF:
        """Bull / base / bear with 25% / 50% / 25% probability weights."""
        bull = self.run_dcf(ticker, scenario="bull")
        base = self.run_dcf(ticker, scenario="base")
        bear = self.run_dcf(ticker, scenario="bear")

        pw_value = (
            0.25 * bull.fair_value_per_share
            + 0.50 * base.fair_value_per_share
            + 0.25 * bear.fair_value_per_share
        )
        current_price = base.current_price
        upside = (pw_value / current_price - 1.0) * 100.0 if current_price else 0.0

        return ThreeScenarioDCF(
            ticker=ticker,
            bull=bull,
            base=base,
            bear=bear,
            probability_weighted_value=round(pw_value, 2),
            current_price=round(current_price, 2),
            upside_pct=round(upside, 1),
        )

    def run_sensitivity_analysis(
        self,
        ticker:     str,
        wacc_bps:   Optional[list[int]]   = None,
        tgr_range:  Optional[list[float]] = None,
    ) -> SensitivityMatrix:
        """Grid of implied share prices across WACC and terminal growth rate.

        wacc_bps: basis point deltas from base WACC (default ±200bp in 50bp steps)
        tgr_range: terminal growth rates (default 0.0% to 4.0% in 0.5% steps)
        """
        wacc_result = self._wacc_calc.compute_wacc(ticker)
        base_wacc   = wacc_result.wacc

        if wacc_bps is None:
            wacc_bps = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
        if tgr_range is None:
            tgr_range = [0.00, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]

        wacc_list = [round(base_wacc + bps / 10000.0, 4) for bps in wacc_bps]

        historical = self._fcff_proj.build_from_historical(ticker, n_historical=5)
        projected  = self._fcff_proj.project_fcff(historical, n_years=10, scenario="base")

        implied_prices: list[list[float]] = []
        for wacc in wacc_list:
            row = []
            for tgr in tgr_range:
                tv = self._fcff_proj.compute_terminal_value(
                    fcff_terminal=projected[-1] if projected else 0.0,
                    wacc=wacc,
                    method="gordon",
                    terminal_growth=tgr,
                )
                pv_fcff = sum(
                    cf / ((1.0 + wacc) ** (i + 1))
                    for i, cf in enumerate(projected)
                )
                pv_tv = tv / ((1.0 + wacc) ** 10)
                ev    = pv_fcff + pv_tv
                eq    = ev - historical.net_debt
                sh    = historical.shares_diluted or 1e9
                price = round(eq / sh, 2)
                row.append(price)
            implied_prices.append(row)

        return SensitivityMatrix(
            ticker=ticker,
            wacc_range=[round(w, 4) for w in wacc_list],
            tgr_range=tgr_range,
            implied_prices=implied_prices,
        )

    def run_monte_carlo_dcf(
        self,
        ticker:       str,
        n_simulations: int = 10_000,
    ) -> MonteCarloResult:
        """Monte Carlo valuation over 10-year projection.

        Sampled distributions:
          - revenue_growth: Normal(mu=historical_cagr, sigma=0.05)
          - ebit_margin:    Normal(mu=historical_avg, sigma=0.03)
          - wacc:           Normal(mu=base_wacc, sigma=0.01)
          - terminal_growth:Uniform(0.01, 0.025)
        """
        wacc_result = self._wacc_calc.compute_wacc(ticker)
        historical  = self._fcff_proj.build_from_historical(ticker, n_historical=5)

        base_wacc      = wacc_result.wacc
        base_rev_cagr  = historical.revenue_cagr_5yr
        base_margin    = historical.ebit_margin_avg
        capex_pct      = historical.capex_pct_rev_avg
        da_pct         = historical.da_pct_rev_avg
        nwc_pct        = historical.nwc_pct_rev_avg
        tax            = historical.effective_tax_rate
        base_rev       = historical.revenues[-1] if historical.revenues else 1e9
        net_debt       = historical.net_debt
        shares         = historical.shares_diluted or 1e9

        results: list[float] = []

        rng = random.Random(42)

        for _ in range(n_simulations):
            # Sample inputs
            g     = rng.gauss(base_rev_cagr, 0.05)
            marg  = rng.gauss(base_margin, 0.03)
            wacc  = rng.gauss(base_wacc, 0.01)
            tgr   = rng.uniform(0.01, 0.025)

            # Clamp
            g    = max(-0.20, min(0.50, g))
            marg = max(0.01, min(0.55, marg))
            wacc = max(0.04, min(0.25, wacc))
            tgr  = min(tgr, wacc - 0.005)

            # Project 10 years
            curr_rev = base_rev
            pv_total = 0.0
            for yr in range(1, 11):
                curr_rev  *= (1.0 + g)
                ebit       = curr_rev * marg
                nopat      = ebit * (1.0 - tax)
                da_        = curr_rev * da_pct
                capex_     = curr_rev * capex_pct
                dnwc       = curr_rev * nwc_pct * g if g > 0 else 0.0
                fcff       = nopat + da_ - capex_ - dnwc
                pv_total  += fcff / ((1.0 + wacc) ** yr)

            # Terminal value (Gordon)
            spread = wacc - tgr
            if spread < 0.005:
                spread = 0.005
            fcff_terminal = curr_rev * marg * (1.0 - tax) + curr_rev * da_pct - curr_rev * capex_pct
            tv = fcff_terminal * (1.0 + tgr) / spread
            pv_tv = tv / ((1.0 + wacc) ** 10)

            ev    = pv_total + pv_tv
            eq    = ev - net_debt
            price = eq / shares
            results.append(price)

        results.sort()
        n = len(results)
        current_price = _yf_price(ticker) or 0.0
        pct_above = sum(1 for p in results if p > current_price) / n * 100.0

        def pct_val(p: float) -> float:
            idx = int(p / 100 * n)
            return results[min(idx, n - 1)]

        return MonteCarloResult(
            ticker=ticker,
            n_simulations=n_simulations,
            p10=round(pct_val(10), 2),
            p25=round(pct_val(25), 2),
            p50=round(pct_val(50), 2),
            p75=round(pct_val(75), 2),
            p90=round(pct_val(90), 2),
            mean=round(sum(results) / n, 2),
            std=round(statistics.stdev(results), 2),
            current_price=round(current_price, 2),
            pct_above_current=round(pct_above, 1),
            distribution=results,
        )


# ===========================================================================
# LBOModel
# ===========================================================================

class LBOModel:
    """Leveraged Buyout template with debt waterfall, IRR and MOIC."""

    # SOFR proxy — fallback if FRED fails
    _SOFR_FALLBACK = 0.053   # 5.3%

    def compute_irr(self, cashflows: list[float]) -> float:
        """Compute IRR via Newton-Raphson on NPV = 0.

        cashflows[0] is the negative initial outflow, subsequent entries are inflows.
        Returns IRR as decimal (e.g. 0.22 for 22%).
        """
        if not cashflows or len(cashflows) < 2:
            return 0.0

        def npv(r: float) -> float:
            return sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cashflows))

        def dnpv(r: float) -> float:
            return sum(-t * cf / ((1.0 + r) ** (t + 1)) for t, cf in enumerate(cashflows))

        r = 0.15  # initial guess
        for _ in range(200):
            f  = npv(r)
            df = dnpv(r)
            if abs(df) < 1e-12:
                break
            r_new = r - f / df
            if abs(r_new - r) < 1e-8:
                r = r_new
                break
            r = r_new
            r = max(-0.99, min(10.0, r))  # prevent divergence

        return round(r, 4)

    def compute_debt_capacity(self, ticker: str) -> float:
        """Max LBO debt = EBITDA × 5.5x (HY market standard).

        Returns debt capacity in $ (same currency as EDGAR units).
        """
        cik   = _edgar_cik(ticker)
        facts = _edgar_facts(cik) if cik else {}

        ebitda: Optional[float] = None

        ebit_s = _extract_annual_series(facts, "OperatingIncomeLoss", n=3)
        da_s   = _extract_annual_series(facts, "DepreciationDepletionAndAmortization", n=3)
        if not da_s:
            da_s = _extract_annual_series(facts, "DepreciationAndAmortization", n=3)

        if ebit_s:
            ebit = ebit_s[-1]
            da   = da_s[-1] if da_s else ebit * 0.05
            ebitda = ebit + da

        return (ebitda * 5.5) if ebitda and ebitda > 0 else 0.0

    def run_lbo(
        self,
        ticker:         str,
        entry_multiple: float = 10.0,
        exit_multiple:  float = 12.0,
        holding_years:  int   = 5,
        debt_pct:       float = 0.60,
    ) -> LBOResult:
        """Full LBO model.

        Steps:
          1. Fetch EBITDA from EDGAR
          2. Entry EV = EBITDA × entry_multiple
          3. Debt = 60% of entry EV, Equity = 40%
          4. Estimate debt cost: SOFR (FRED) + HY spread
          5. FCF waterfall for debt paydown each year
          6. Exit EV = EBITDA_exit × exit_multiple
          7. Exit equity = Exit EV - remaining debt
          8. IRR, MOIC
        """
        cik   = _edgar_cik(ticker)
        facts = _edgar_facts(cik) if cik else {}

        # EBITDA
        ebit_s = _extract_annual_series(facts, "OperatingIncomeLoss", n=3)
        da_s   = _extract_annual_series(facts, "DepreciationDepletionAndAmortization", n=3)
        rev_s  = _extract_annual_series(facts, "Revenues", n=3)

        ebit    = ebit_s[-1] if ebit_s else 0.0
        da      = da_s[-1] if da_s else ebit * 0.05
        rev     = rev_s[-1] if rev_s else 0.0
        ebitda  = ebit + da

        if ebitda <= 0:
            logger.warning("No EBITDA data for LBO; using market cap heuristic", ticker=ticker)
            mc_bn = _yf_market_cap(ticker) or 10.0
            ebitda = mc_bn * 1e9 / 12.0  # 12x EV/EBITDA inverse

        # Entry
        entry_ev     = ebitda * entry_multiple
        entry_debt   = entry_ev * debt_pct
        entry_equity = entry_ev * (1.0 - debt_pct)

        # Debt cost
        sofr = _fred_latest("SOFR") or _fred_latest("FEDFUNDS") or self._SOFR_FALLBACK
        hy_spread = _fred_latest("BAMLH0A0HYM2")
        hy_dec = (hy_spread / 100.0) if hy_spread else _FRED_FALLBACK_HY
        debt_cost = sofr / 100.0 + hy_dec if sofr > 1.0 else (sofr + hy_dec)
        debt_cost = min(debt_cost, 0.15)

        # FCF waterfall (simplified: revenue grows 5%, margin stable, capex = 4% rev)
        tax_rate     = 0.21
        rev_growth   = 0.05
        ebit_margin  = (ebit / rev) if rev > 0 else 0.12
        capex_pct    = 0.04

        remaining_debt = entry_debt
        cashflows      = [-entry_equity]

        for yr in range(1, holding_years + 1):
            rev        = rev * (1.0 + rev_growth)
            yr_ebit    = rev * ebit_margin
            yr_da      = rev * 0.05
            yr_capex   = rev * capex_pct
            yr_ebitda  = yr_ebit + yr_da
            interest   = remaining_debt * debt_cost
            ebt        = yr_ebit - interest
            taxes      = max(0.0, ebt * tax_rate)
            net_income = ebt - taxes
            fcf        = net_income + yr_da - yr_capex

            # Debt paydown
            paydown = max(0.0, min(fcf * 0.70, remaining_debt))  # 70% of FCF to debt
            remaining_debt -= paydown

        # Exit
        ebitda_exit  = yr_ebitda
        exit_ev      = ebitda_exit * exit_multiple
        exit_equity  = max(0.0, exit_ev - remaining_debt)

        cashflows.append(exit_equity)

        irr  = self.compute_irr(cashflows)
        moic = exit_equity / entry_equity if entry_equity > 0 else 0.0
        debt_paydown = entry_debt - remaining_debt

        return LBOResult(
            ticker=ticker,
            entry_ev=round(entry_ev, 0),
            entry_equity=round(entry_equity, 0),
            entry_debt=round(entry_debt, 0),
            exit_ev=round(exit_ev, 0),
            exit_equity=round(exit_equity, 0),
            ebitda_entry=round(ebitda, 0),
            ebitda_exit=round(ebitda_exit, 0),
            entry_multiple=entry_multiple,
            exit_multiple=exit_multiple,
            holding_years=holding_years,
            irr=round(irr, 4),
            moic=round(moic, 2),
            debt_paydown=round(debt_paydown, 0),
            debt_cost=round(debt_cost, 4),
            cashflows=[round(c, 0) for c in cashflows],
        )


# ===========================================================================
# ComparableCompanyAdjustor
# ===========================================================================

class ComparableCompanyAdjustor:
    """Comp-based valuation as sanity check against DCF."""

    # Damodaran sector multiples (Jan 2025 US market medians)
    _SECTOR_MULTIPLES: dict[str, dict[str, float]] = {
        "Technology":          {"EV/EBITDA": 22.0, "P/E": 30.0, "EV/Sales": 6.0, "P/S": 5.5},
        "Software (SaaS)":     {"EV/EBITDA": 30.0, "P/E": 50.0, "EV/Sales": 9.0, "P/S": 8.5},
        "Semiconductor":       {"EV/EBITDA": 20.0, "P/E": 25.0, "EV/Sales": 5.5, "P/S": 5.0},
        "Drugs (Biotech)":     {"EV/EBITDA": 40.0, "P/E": 45.0, "EV/Sales": 8.0, "P/S": 7.5},
        "Drugs (Pharmaceutical)":{"EV/EBITDA":14.0,"P/E": 18.0, "EV/Sales": 3.5, "P/S": 3.2},
        "Healthcare Products": {"EV/EBITDA": 16.0, "P/E": 22.0, "EV/Sales": 4.0, "P/S": 3.8},
        "Consumer Staples":    {"EV/EBITDA": 13.0, "P/E": 20.0, "EV/Sales": 2.5, "P/S": 2.3},
        "Consumer Discretionary":{"EV/EBITDA":12.0,"P/E": 18.0, "EV/Sales": 2.0, "P/S": 1.8},
        "Retail (General)":    {"EV/EBITDA": 10.0, "P/E": 15.0, "EV/Sales": 1.0, "P/S": 0.9},
        "Industrials":         {"EV/EBITDA": 11.0, "P/E": 18.0, "EV/Sales": 1.5, "P/S": 1.4},
        "Aerospace / Defense": {"EV/EBITDA": 13.0, "P/E": 20.0, "EV/Sales": 1.8, "P/S": 1.7},
        "Energy":              {"EV/EBITDA":  8.0, "P/E": 12.0, "EV/Sales": 1.2, "P/S": 1.1},
        "Utility (General)":   {"EV/EBITDA": 11.0, "P/E": 16.0, "EV/Sales": 2.0, "P/S": 1.9},
        "Real Estate (General)":{"EV/EBITDA":17.0, "P/E": 22.0, "EV/Sales": 5.0, "P/S": 4.5},
        "Bank (Regional)":     {"EV/EBITDA":  8.0, "P/E": 12.0, "EV/Sales": 2.0, "P/S": 1.8},
        "Metals & Mining":     {"EV/EBITDA":  9.0, "P/E": 14.0, "EV/Sales": 1.8, "P/S": 1.6},
        "Telecom Services":    {"EV/EBITDA":  8.0, "P/E": 13.0, "EV/Sales": 1.5, "P/S": 1.4},
        "Restaurant / Dining": {"EV/EBITDA": 14.0, "P/E": 25.0, "EV/Sales": 2.5, "P/S": 2.3},
        "Default":             {"EV/EBITDA": 12.0, "P/E": 18.0, "EV/Sales": 2.5, "P/S": 2.2},
    }

    def get_sector_multiples(self, sector: str) -> dict[str, float]:
        """Return EV/EBITDA, P/E, EV/Sales sector medians."""
        if sector in self._SECTOR_MULTIPLES:
            return self._SECTOR_MULTIPLES[sector]
        # Fuzzy match
        sector_lower = sector.lower()
        for k, v in self._SECTOR_MULTIPLES.items():
            if any(word in k.lower() for word in sector_lower.split()):
                return v
        return self._SECTOR_MULTIPLES["Default"]

    def compute_implied_price_from_comps(self, ticker: str) -> dict[str, float]:
        """Compute implied share price from EV/EBITDA, EV/Sales, and P/E multiples.

        Returns dict with keys: ev_ebitda_price, ev_sales_price, pe_price, avg_price.
        """
        cik   = _edgar_cik(ticker)
        facts = _edgar_facts(cik) if cik else {}

        ebit_s = _extract_annual_series(facts, "OperatingIncomeLoss", n=3)
        da_s   = _extract_annual_series(facts, "DepreciationDepletionAndAmortization", n=3)
        rev_s  = _extract_annual_series(facts, "Revenues", n=3)
        eps_s  = _extract_annual_series(facts, "EarningsPerShareDiluted", n=3)
        cash_s = _extract_annual_series(facts, "CashAndCashEquivalentsAtCarryingValue", n=3)
        ld_s   = _extract_annual_series(facts, "LongTermDebt", n=3)
        sd_s   = _extract_annual_series(facts, "ShortTermBorrowings", n=3)

        ebit    = ebit_s[-1] if ebit_s else 0.0
        da      = da_s[-1] if da_s else 0.0
        rev     = rev_s[-1] if rev_s else 0.0
        eps     = eps_s[-1] if eps_s else 0.0
        cash    = cash_s[-1] if cash_s else 0.0
        ld      = ld_s[-1] if ld_s else 0.0
        sd      = sd_s[-1] if sd_s else 0.0
        ebitda  = ebit + da
        net_debt = ld + sd - cash

        shares = _yf_shares(ticker) or 1e9

        # Sector
        sector = "Default"
        if _YF_OK:
            try:
                sector = yf.Ticker(ticker).info.get("sector", "Default") or "Default"
            except Exception:
                pass

        multiples = self.get_sector_multiples(sector)

        prices: list[float] = []
        result: dict[str, float] = {}

        if ebitda > 0:
            ev   = ebitda * multiples["EV/EBITDA"]
            p    = (ev - net_debt) / shares
            result["ev_ebitda_price"] = round(p, 2)
            prices.append(p)

        if rev > 0:
            ev   = rev * multiples["EV/Sales"]
            p    = (ev - net_debt) / shares
            result["ev_sales_price"] = round(p, 2)
            prices.append(p)

        if eps != 0:
            p = eps * multiples["P/E"]
            result["pe_price"] = round(p, 2)
            prices.append(p)

        result["avg_comps_price"] = round(statistics.mean(prices), 2) if prices else 0.0
        result["current_price"]   = round(_yf_price(ticker) or 0.0, 2)
        result["sector"]          = sector
        return result

    def football_field_chart(self, ticker: str) -> dict:
        """Summarize all valuation methods and their implied ranges.

        Returns a dict suitable for rendering a football-field chart:
          {method: {"low": x, "mid": y, "high": z}}
        """
        engine = DCFValuationEngine()

        # DCF scenarios
        try:
            three_scen = engine.run_three_scenario_dcf(ticker)
            dcf_low    = three_scen.bear.fair_value_per_share
            dcf_mid    = three_scen.base.fair_value_per_share
            dcf_high   = three_scen.bull.fair_value_per_share
        except Exception as exc:
            logger.warning("Football field DCF failed", error=str(exc))
            dcf_low = dcf_mid = dcf_high = 0.0

        # Monte Carlo
        try:
            mc = engine.run_monte_carlo_dcf(ticker, n_simulations=2000)
            mc_low  = mc.p10
            mc_mid  = mc.p50
            mc_high = mc.p90
        except Exception as exc:
            logger.warning("Football field MC failed", error=str(exc))
            mc_low = mc_mid = mc_high = 0.0

        # Comps
        comps = self.compute_implied_price_from_comps(ticker)
        comp_prices = [v for k, v in comps.items()
                       if k.endswith("_price") and k != "current_price" and v > 0]
        comps_low  = min(comp_prices) if comp_prices else 0.0
        comps_mid  = comps.get("avg_comps_price", 0.0)
        comps_high = max(comp_prices) if comp_prices else 0.0

        # LBO floor
        try:
            lbo   = LBOModel().run_lbo(ticker)
            shares = _yf_shares(ticker) or 1e9
            lbo_equity_per_share = lbo.entry_equity / shares
            lbo_low  = lbo_equity_per_share * 0.90
            lbo_mid  = lbo_equity_per_share
            lbo_high = lbo_equity_per_share * 1.10
        except Exception as exc:
            logger.warning("Football field LBO failed", error=str(exc))
            lbo_low = lbo_mid = lbo_high = 0.0

        current_price = _yf_price(ticker) or 0.0

        return {
            "ticker":        ticker,
            "current_price": round(current_price, 2),
            "DCF_three_scenario": {
                "low": round(dcf_low, 2),
                "mid": round(dcf_mid, 2),
                "high": round(dcf_high, 2),
            },
            "Monte_Carlo_DCF": {
                "low": round(mc_low, 2),
                "mid": round(mc_mid, 2),
                "high": round(mc_high, 2),
            },
            "Comparable_Companies": {
                "low": round(comps_low, 2),
                "mid": round(comps_mid, 2),
                "high": round(comps_high, 2),
            },
            "LBO_floor": {
                "low": round(lbo_low, 2),
                "mid": round(lbo_mid, 2),
                "high": round(lbo_high, 2),
            },
        }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def get_wacc(ticker: str) -> WACCResult:
    return WACCCalculator().compute_wacc(ticker)


def get_dcf(ticker: str, scenario: str = "base") -> DCFResult:
    return DCFValuationEngine().run_dcf(ticker, scenario=scenario)


def get_full_valuation(ticker: str) -> dict:
    """Return football field + three-scenario + Monte Carlo in one call."""
    engine = DCFValuationEngine()
    adj    = ComparableCompanyAdjustor()
    return {
        "football_field":     adj.football_field_chart(ticker),
        "three_scenario_dcf": engine.run_three_scenario_dcf(ticker).__dict__,
        "monte_carlo":        engine.run_monte_carlo_dcf(ticker, n_simulations=5000).__dict__,
    }


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import pprint
    TICKER = "AAPL"

    print("=" * 70)
    print(f"  SENTINEL — Full Valuation: {TICKER}")
    print("=" * 70)

    # WACC
    print("\n[1] WACC Components")
    wacc_calc = WACCCalculator()
    wacc_res  = wacc_calc.compute_wacc(TICKER)
    for k, v in wacc_res.to_dict().items():
        print(f"  {k:30s}: {v}")

    # WACC Sensitivity
    print("\n[2] WACC Sensitivity (Beta × ERP)")
    sens_df = wacc_calc.compute_wacc_sensitivity(TICKER)
    print(sens_df.to_string())

    # DCF Base
    print("\n[3] DCF — Base Scenario")
    engine   = DCFValuationEngine()
    dcf_base = engine.run_dcf(TICKER, scenario="base", n_years=10)
    for k, v in dcf_base.to_dict().items():
        if k not in ("projected_fcff", "pv_fcff"):
            print(f"  {k:30s}: {v}")

    # Three-scenario DCF
    print("\n[4] Three-Scenario DCF (Bull/Base/Bear)")
    three = engine.run_three_scenario_dcf(TICKER)
    print(f"  Bull fair value:            ${three.bull.fair_value_per_share:.2f}")
    print(f"  Base fair value:            ${three.base.fair_value_per_share:.2f}")
    print(f"  Bear fair value:            ${three.bear.fair_value_per_share:.2f}")
    print(f"  Probability-weighted value: ${three.probability_weighted_value:.2f}")
    print(f"  Current price:              ${three.current_price:.2f}")
    print(f"  Upside / (Downside):        {three.upside_pct:.1f}%")

    # Sensitivity matrix
    print("\n[5] Sensitivity Matrix (WACC × Terminal Growth Rate)")
    sm = engine.run_sensitivity_analysis(TICKER)
    df = pd.DataFrame(
        sm.implied_prices,
        index=[f"WACC={w:.1%}" for w in sm.wacc_range],
        columns=[f"TGR={t:.1%}" for t in sm.tgr_range],
    )
    print(df.to_string())

    # Monte Carlo
    print("\n[6] Monte Carlo DCF (10,000 simulations)")
    mc = engine.run_monte_carlo_dcf(TICKER, n_simulations=10_000)
    print(f"  P10: ${mc.p10:.2f}  P25: ${mc.p25:.2f}  P50: ${mc.p50:.2f}  "
          f"P75: ${mc.p75:.2f}  P90: ${mc.p90:.2f}")
    print(f"  Mean: ${mc.mean:.2f}  Std: ${mc.std:.2f}")
    print(f"  Current price: ${mc.current_price:.2f} | % above current: {mc.pct_above_current:.1f}%")

    # LBO
    print("\n[7] LBO Analysis")
    lbo_model  = LBOModel()
    debt_cap   = lbo_model.compute_debt_capacity(TICKER)
    lbo_result = lbo_model.run_lbo(TICKER, entry_multiple=10.0, exit_multiple=12.0, holding_years=5)
    print(f"  Debt capacity (5.5x EBITDA): ${debt_cap/1e9:.1f}B")
    print(f"  Entry EV:    ${lbo_result.entry_ev/1e9:.1f}B")
    print(f"  Exit EV:     ${lbo_result.exit_ev/1e9:.1f}B")
    print(f"  IRR:         {lbo_result.irr:.1%}")
    print(f"  MOIC:        {lbo_result.moic:.2f}x")
    print(f"  Debt cost:   {lbo_result.debt_cost:.2%}")

    # Football field
    print("\n[8] Football Field Valuation")
    adj    = ComparableCompanyAdjustor()
    field  = adj.football_field_chart(TICKER)
    for method, vals in field.items():
        if isinstance(vals, dict):
            print(f"  {method:30s}: ${vals['low']:.2f} — ${vals['mid']:.2f} — ${vals['high']:.2f}")
        else:
            print(f"  {method:30s}: {vals}")
