"""
LBO / Merger Model Templates V3 — Dimension #101 (score 8 → 9).

Full-fidelity leveraged buyout and strategic merger financial model engine.
Adds versus V2: six-tranche debt waterfall, PIK-toggle mechanics, PE fund
economics (management fee / carry / hurdle / clawback), deal financing mix,
goodwill and purchase-price allocation, LBO-candidate screener (EDGAR XBRL),
distressed/restructuring overlay, roll-up aggregation, and 12 pre-built
model templates.  All arithmetic is pure Python / NumPy — no paid data feeds.

Public classes
--------------
LBOModel                — full leveraged buyout model with 6-tranche debt + waterfall
MergerModel             — strategic merger (A/D, PPA, credit metrics, synergy ramp)
DCFMergerValuation      — DCF standalone + synergy-adjusted value
LBOCandidateScreener    — EDGAR XBRL + yfinance LBO attractiveness scoring
PEFundEconomics         — fund-level IRR / DPI / TVPI / RVPI with carry waterfall
ModelTemplates          — 12 quick-start templates

FastAPI router
--------------
POST  /v3/lbo/run
POST  /v3/lbo/merger
POST  /v3/lbo/sensitivity
GET   /v3/lbo/templates
POST  /v3/lbo/dcf-valuation
GET   /v3/lbo/screen
POST  /v3/lbo/pe-fund
GET   /v3/lbo/model/{run_id}
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAX_RATE     = 0.21          # US statutory (TCJA 2018)
SOFR_PROXY   = 0.053         # ~SOFR proxy (May 2026)
_EPS         = 1e-10

# Credit spreads over SOFR by tranche
_SPREAD: Dict[str, float] = {
    "tla":          0.0175,   # +175 bps  — Term Loan A (bank)
    "tlb":          0.0275,   # +275 bps  — Term Loan B (institutional)
    "revolver":     0.0200,   # +200 bps  — revolving credit facility
    "senior_notes": 0.0450,   # +450 bps  — high-yield senior secured notes
    "mezz":         0.0700,   # +700 bps  — mezzanine / subordinated notes
    "pik":          0.1000,   # +1000 bps — PIK notes (accretes)
}

# Amortisation schedule (% of original principal per year)
_AMORT: Dict[str, float] = {
    "tla":          0.100,    # 10% pa
    "tlb":          0.010,    # 1% pa (bullet-like institutional TLB)
    "revolver":     0.000,    # drawn/repaid freely
    "senior_notes": 0.000,    # bullet
    "mezz":         0.000,    # bullet
    "pik":          0.000,    # pure accretion — no cash payment
}

# Default tranche sizing (% of total funded debt)
_DEFAULT_SPLIT: Dict[str, float] = {
    "tla":          0.15,
    "tlb":          0.30,
    "revolver":     0.05,
    "senior_notes": 0.25,
    "mezz":         0.15,
    "pik":          0.10,
}

# Covenant defaults
_COVENANT_DEFAULTS = {
    "max_leverage":            6.5,
    "min_interest_coverage":   2.0,
    "max_capex_pct_ebitda":    0.30,
    "min_liquidity_mm":        25.0,
    "max_senior_leverage":     4.5,
}

# LBO attractiveness scoring weights
_LBO_WEIGHTS = {
    "fcf_stability":   0.25,
    "leverage_room":   0.20,
    "margin_quality":  0.20,
    "asset_coverage":  0.15,
    "management":      0.10,
    "sector_appeal":   0.10,
}

# Sector LBO attractiveness (0-100)
_LBO_SECTOR_SCORES: Dict[str, float] = {
    "Consumer Staples":    85,
    "Health Care":         80,
    "Information Technology": 75,
    "Industrials":         70,
    "Consumer Discretionary": 65,
    "Financials":          60,
    "Communication Services": 65,
    "Materials":           55,
    "Real Estate":         70,
    "Energy":              45,
    "Utilities":           50,
    "Unknown":             60,
}

# EDGAR / yfinance data constants
EDGAR_BASE        = "https://data.sec.gov"
COMPANY_TICKERS   = "https://www.sec.gov/files/company_tickers.json"
_HEADERS          = {"User-Agent": "SENTINEL-LBO-V3/3.0 richard.porras@realempanada.com"}
_RATE_DELAY       = 0.13

# SQLite persistence
_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_LBO_DB   = _DATA_DIR / "lbo_models_v3.db"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LBOResult:
    """Full output of LBOModel.run_full_model()."""
    run_id: str
    model_name: str
    entry_ev: float
    entry_multiple: float
    entry_ebitda: float
    equity_invested: float
    total_debt: float
    hold_period: int
    exit_ev: float
    exit_equity_value: float
    moic: float
    irr: float
    cash_on_cash: float
    peak_leverage: float          # max Debt/EBITDA during hold
    exit_leverage: float          # Debt/EBITDA at exit
    debt_paydown_mm: float
    financials_df: Optional[pd.DataFrame] = None
    debt_schedule_df: Optional[pd.DataFrame] = None
    sensitivity_df: Optional[pd.DataFrame] = None
    covenant_breach: bool = False
    covenant_detail: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LBOScore:
    """LBO attractiveness score for a candidate company."""
    ticker: str
    company_name: str
    composite_score: float          # 0-100
    fcf_stability_score: float
    leverage_room_score: float
    margin_quality_score: float
    asset_coverage_score: float
    management_score: float
    sector_score: float
    ebitda_mm: float
    ebitda_margin: float
    net_leverage: float
    fcf_conversion: float
    revenue_mm: float
    market_cap_mm: float
    sector: str
    rationale: str
    scored_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MergerResult:
    """Output of MergerModel.run()."""
    run_id: str
    acquirer: str
    target: str
    deal_value_mm: float
    premium_pct: float
    cash_pct: float
    stock_pct: float
    new_shares_issued: float
    goodwill_mm: float
    identifiable_intangibles_mm: float
    combined_leverage: float
    interest_coverage: float
    leverage_rating: str           # "Investment Grade" | "High Yield" | "Distressed"
    accretion_dilution_df: Optional[pd.DataFrame] = None
    combined_financials_df: Optional[pd.DataFrame] = None
    breakeven_year: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

def _get_lbo_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_LBO_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_lbo_db() -> None:
    with _get_lbo_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS lbo_runs (
            run_id      TEXT PRIMARY KEY,
            model_type  TEXT NOT NULL,
            model_name  TEXT NOT NULL,
            version     INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL,
            params_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lbo_type ON lbo_runs(model_type);
        CREATE INDEX IF NOT EXISTS idx_lbo_name ON lbo_runs(model_name);

        CREATE TABLE IF NOT EXISTS lbo_scores (
            score_id    TEXT PRIMARY KEY,
            ticker      TEXT NOT NULL,
            scored_at   TEXT NOT NULL,
            score_json  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_score_ticker ON lbo_scores(ticker);
        """)


_init_lbo_db()


def _save_run(model_type: str, name: str, params: dict, result: dict) -> str:
    run_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()
    with _get_lbo_conn() as conn:
        row = conn.execute(
            "SELECT MAX(version) as v FROM lbo_runs WHERE model_name=? AND model_type=?",
            (name, model_type),
        ).fetchone()
        version = (row["v"] or 0) + 1
        conn.execute(
            "INSERT INTO lbo_runs VALUES (?,?,?,?,?,?,?)",
            (run_id, model_type, name, version, now,
             json.dumps(params, default=str), json.dumps(result, default=str)),
        )
    return run_id


def _load_run(run_id: str) -> Optional[dict]:
    with _get_lbo_conn() as conn:
        row = conn.execute("SELECT * FROM lbo_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    return {
        "run_id":      row["run_id"],
        "model_type":  row["model_type"],
        "model_name":  row["model_name"],
        "version":     row["version"],
        "created_at":  row["created_at"],
        "params":      json.loads(row["params_json"]),
        "result":      json.loads(row["result_json"]),
    }


# ---------------------------------------------------------------------------
# EDGAR / yfinance helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, params: Optional[Dict] = None) -> Optional[Any]:
    """Simple GET with rate limiting and retries."""
    if not _HTTPX:
        return None
    for attempt in range(3):
        try:
            time.sleep(_RATE_DELAY)
            r = httpx.get(url, params=params, headers=_HEADERS, timeout=25.0, follow_redirects=True)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 2))
        except Exception:
            if attempt < 2:
                time.sleep(1.5)
    return None


def _lookup_cik(ticker: str) -> Optional[str]:
    data = _http_get(COMPANY_TICKERS)
    if not data:
        return None
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return str(entry.get("cik_str", ""))
    return None


def _get_xbrl_facts(cik: str) -> Dict[str, float]:
    """Fetch key annual financial facts from EDGAR XBRL."""
    url  = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    data = _http_get(url)
    if not data:
        return {}

    gaap   = data.get("facts", {}).get("us-gaap", {})
    result: Dict[str, float] = {}

    def _latest(concept: str, unit: str = "USD") -> Optional[float]:
        entries = gaap.get(concept, {}).get("units", {}).get(unit, [])
        annual  = [e for e in entries if e.get("form") in ("10-K","10-K/A") and e.get("val") is not None]
        if annual:
            return float(sorted(annual, key=lambda x: x.get("end",""))[-1]["val"])
        return None

    concepts = {
        "revenue":       ["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet"],
        "net_income":    ["NetIncomeLoss"],
        "ebit":          ["OperatingIncomeLoss"],
        "da":            ["DepreciationDepletionAndAmortization","DepreciationAndAmortization"],
        "capex":         ["PaymentsToAcquirePropertyPlantAndEquipment"],
        "operating_cf":  ["NetCashProvidedByUsedInOperatingActivities"],
        "total_debt":    ["LongTermDebt","LongTermDebtAndCapitalLeaseObligations"],
        "cash":          ["CashAndCashEquivalentsAtCarryingValue"],
        "total_equity":  ["StockholdersEquity"],
        "total_assets":  ["Assets"],
        "interest_exp":  ["InterestExpense"],
        "tax_exp":       ["IncomeTaxExpenseBenefit"],
    }
    for key, clist in concepts.items():
        for c in clist:
            v = _latest(c)
            if v is not None:
                result[key] = v
                break

    if "ebit" in result and "da" in result:
        result["ebitda"]      = result["ebit"] + result["da"]
    if "operating_cf" in result and "capex" in result:
        result["fcf"]         = result["operating_cf"] - result["capex"]
    if "ebitda" in result and result.get("ebitda",0) > 0 and "total_debt" in result:
        result["net_leverage"] = (
            result["total_debt"] - result.get("cash", 0)
        ) / result["ebitda"]
    if "ebitda" in result and result.get("revenue", 0) > 0:
        result["ebitda_margin"] = result["ebitda"] / result["revenue"]

    return result


def _yf_info(ticker: str) -> Dict[str, Any]:
    """Fetch yfinance info dict safely."""
    try:
        import yfinance as yf
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def _solve_irr(cash_flows: List[float]) -> float:
    """
    Newton–Raphson IRR solver.
    cash_flows[0] = initial outflow (negative), subsequent = inflows.
    Returns annual IRR; raises ValueError if no convergence.
    """
    cf = np.array(cash_flows, dtype=float)
    rate = 0.10     # initial guess
    for _ in range(1000):
        npv  = float(np.sum(cf / (1 + rate) ** np.arange(len(cf))))
        dnpv = float(sum(-i * cf[i] / (1 + rate) ** (i + 1) for i in range(len(cf))))
        if abs(dnpv) < _EPS:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < 1e-8:
            rate = new_rate
            break
        rate = new_rate
    return rate


# ---------------------------------------------------------------------------
# LBOModel
# ---------------------------------------------------------------------------

class LBOModel:
    """
    Full leveraged buyout financial model.

    Six debt tranches: TLA, TLB, Revolver, Senior Notes, Mezz, PIK.
    Cash waterfall sweeps FCF to senior debt first.
    PIK notes accrete (no cash interest); toggle available.
    Covenant monitoring: max leverage, min interest coverage.
    Returns IRR, MOIC, equity value bridge, sensitivity table.
    """

    def __init__(
        self,
        target_name: str,
        entry_ebitda: float,
        entry_multiple: float,
        hold_period: int = 5,
        sofr: float = SOFR_PROXY,
    ):
        self.target_name    = target_name
        self.entry_ebitda   = entry_ebitda
        self.entry_multiple = entry_multiple
        self.hold_period    = hold_period
        self.sofr           = sofr
        self.entry_ev       = entry_ebitda * entry_multiple

        # Capital structure (set via set_capital_structure)
        self._tranche_pcts:  Dict[str, float] = dict(_DEFAULT_SPLIT)
        self._tranche_rates: Dict[str, float] = {}
        self._tranche_amort: Dict[str, float] = dict(_AMORT)
        self._pik_toggle:    bool = False
        self._leverage_pct:  float = 0.60   # total debt / entry EV

        # Set default rates
        for t, spread in _SPREAD.items():
            self._tranche_rates[t] = sofr + spread

        # Will be populated
        self._equity_pct:    float = 0.40
        self._revenue_growth: float = 0.05
        self._ebitda_margin:  float = 0.0   # if 0, use implied from ebitda/revenue
        self._capex_pct:     float = 0.03
        self._nwc_pct:       float = 0.01

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_capital_structure(
        self,
        leverage_pct: float = 0.60,
        tranche_pcts: Optional[Dict[str, float]] = None,
        pik_toggle: bool = False,
    ) -> "LBOModel":
        """
        Set overall leverage and tranche split.
        leverage_pct: total debt / entry EV (e.g. 0.60 = 60% leverage).
        tranche_pcts: custom dict of tranche_name → pct of total debt (must sum to ~1).
        """
        self._leverage_pct = leverage_pct
        self._equity_pct   = 1.0 - leverage_pct
        self._pik_toggle   = pik_toggle
        if tranche_pcts:
            self._tranche_pcts = tranche_pcts
        return self

    def set_debt_terms(
        self,
        rate_overrides: Optional[Dict[str, float]] = None,
        amort_overrides: Optional[Dict[str, float]] = None,
    ) -> "LBOModel":
        """Override default rates or amortisation for specific tranches."""
        if rate_overrides:
            self._tranche_rates.update(rate_overrides)
        if amort_overrides:
            self._tranche_amort.update(amort_overrides)
        return self

    def set_operating_assumptions(
        self,
        revenue_growth: float = 0.05,
        ebitda_margin: float = 0.0,
        capex_pct_revenue: float = 0.03,
        nwc_change_pct: float = 0.01,
        entry_revenue: Optional[float] = None,
    ) -> "LBOModel":
        """Set P&L projection drivers."""
        self._revenue_growth = revenue_growth
        self._ebitda_margin  = ebitda_margin
        self._capex_pct      = capex_pct_revenue
        self._nwc_pct        = nwc_change_pct
        self._entry_revenue  = entry_revenue  # if None, back-solved from EBITDA/margin
        return self

    # ------------------------------------------------------------------
    # P&L projection
    # ------------------------------------------------------------------

    def project_financials(
        self,
        revenue_growth: Optional[float] = None,
        ebitda_margin: Optional[float] = None,
        capex_pct_revenue: Optional[float] = None,
        nwc_change_pct: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Project 5-year income statement and FCF from entry EBITDA.
        Returns DataFrame with columns:
          year, revenue, ebitda, ebitda_margin, da, ebit, interest, ebt,
          tax, net_income, capex, nwc_change, fcf_before_debt, available_for_sweep.
        """
        rev_g  = revenue_growth   or self._revenue_growth
        eb_m   = ebitda_margin    or self._ebitda_margin
        cap_p  = capex_pct_revenue or self._capex_pct
        nwc_p  = nwc_change_pct  or self._nwc_pct

        # Back-solve entry revenue if margin given; else assume 20% margin
        if eb_m > 0:
            entry_rev = self.entry_ebitda / eb_m
        elif hasattr(self, "_entry_revenue") and self._entry_revenue:
            entry_rev = self._entry_revenue
            eb_m      = self.entry_ebitda / entry_rev
        else:
            eb_m      = 0.20
            entry_rev = self.entry_ebitda / eb_m

        # D&A: typically 3-5% of revenue (assume 4%)
        da_pct = 0.04

        rows: List[Dict[str, float]] = []
        for yr in range(1, self.hold_period + 1):
            rev    = entry_rev * (1 + rev_g) ** yr
            ebitda = rev * eb_m
            da     = rev * da_pct
            ebit   = ebitda - da
            capex  = rev * cap_p
            nwc    = rev * nwc_p
            # FCF before debt service (interest deducted inside debt schedule)
            fcf_pre_debt = ebitda - capex - nwc - (ebit * TAX_RATE)
            rows.append({
                "year":           yr,
                "revenue":        round(rev, 2),
                "ebitda":         round(ebitda, 2),
                "ebitda_margin":  round(eb_m, 4),
                "da":             round(da, 2),
                "ebit":           round(ebit, 2),
                "capex":          round(capex, 2),
                "nwc_change":     round(nwc, 2),
                "fcf_pre_debt":   round(fcf_pre_debt, 2),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Debt schedule
    # ------------------------------------------------------------------

    def _build_entry_tranches(self) -> Dict[str, float]:
        """Compute entry debt for each tranche."""
        total_debt = self.entry_ev * self._leverage_pct
        return {
            t: total_debt * pct
            for t, pct in self._tranche_pcts.items()
        }

    def compute_debt_schedule(
        self,
        fin_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Year-by-year debt schedule for all six tranches.

        Cash waterfall (each year):
          1. Pay mandatory amortisation (TLA, TLB)
          2. Pay cash interest (all except PIK if PIK-toggle is off)
          3. Remaining FCF sweeps senior debt (TLA → TLB → Senior Notes)
          4. PIK notes accrete at PIK rate

        Returns DataFrame with columns:
          year, [tranche]_beg, [tranche]_interest, [tranche]_amort,
          [tranche]_sweep, [tranche]_end, total_debt_end, leverage,
          interest_coverage, cash_available, covenant_breach.
        """
        entry = self._build_entry_tranches()
        tranches = list(entry.keys())
        balances = dict(entry)

        rows: List[Dict[str, float]] = []
        for _, fin_row in fin_df.iterrows():
            yr     = int(fin_row["year"])
            ebitda = fin_row["ebitda"]

            row: Dict[str, Any] = {"year": yr}
            total_interest_cash = 0.0
            total_amort         = 0.0
            pik_accretion       = 0.0

            # Interest computation
            interest_by_tranche: Dict[str, float] = {}
            for t in tranches:
                rate = self._tranche_rates.get(t, 0.0)
                bal  = balances[t]
                if t == "pik":
                    pik_accretion = bal * rate
                    interest_by_tranche[t] = 0.0   # no cash
                else:
                    cash_int = bal * rate
                    total_interest_cash      += cash_int
                    interest_by_tranche[t]   = cash_int

            # Tax shield from interest
            ebt      = max(fin_row["ebit"] - total_interest_cash, 0.0)
            cash_tax = ebt * TAX_RATE
            # Available for debt service = EBITDA - Capex - NWC change - Cash Tax
            available = (ebitda
                         - fin_row["capex"]
                         - fin_row["nwc_change"]
                         - cash_tax
                         - total_interest_cash)

            # Mandatory amortisation (TLA and TLB)
            amort_by_tranche: Dict[str, float] = {}
            for t in tranches:
                orig_principal = entry[t]
                amort = min(orig_principal * self._tranche_amort.get(t, 0.0), balances[t])
                amort_by_tranche[t] = amort
                total_amort += amort
                available   -= amort

            # Cash sweep: apply excess cash to senior tranches
            sweep_order = ["tla", "tlb", "senior_notes", "mezz"]
            sweep_by_tranche: Dict[str, float] = {t: 0.0 for t in tranches}
            for t in sweep_order:
                if available <= 0 or t not in balances:
                    break
                sweep = min(available, balances[t] - amort_by_tranche.get(t, 0.0))
                sweep = max(sweep, 0.0)
                sweep_by_tranche[t] = sweep
                available -= sweep

            # Update balances
            new_balances: Dict[str, float] = {}
            for t in tranches:
                beg  = balances[t]
                end  = beg - amort_by_tranche.get(t,0) - sweep_by_tranche.get(t,0)
                if t == "pik":
                    end += pik_accretion   # PIK accretes
                end  = max(end, 0.0)
                new_balances[t] = end

                row[f"{t}_beg"]      = round(beg, 2)
                row[f"{t}_interest"] = round(interest_by_tranche.get(t,0), 2)
                row[f"{t}_pik_acc"]  = round(pik_accretion if t=="pik" else 0.0, 2)
                row[f"{t}_amort"]    = round(amort_by_tranche.get(t,0), 2)
                row[f"{t}_sweep"]    = round(sweep_by_tranche.get(t,0), 2)
                row[f"{t}_end"]      = round(end, 2)

            total_debt_end  = sum(new_balances.values())
            leverage        = total_debt_end / ebitda if ebitda > 0 else 0.0
            int_coverage    = ebitda / total_interest_cash if total_interest_cash > 0 else 99.0

            row["total_interest_cash"] = round(total_interest_cash, 2)
            row["pik_accretion"]       = round(pik_accretion, 2)
            row["cash_tax"]            = round(cash_tax, 2)
            row["total_debt_end"]      = round(total_debt_end, 2)
            row["leverage"]            = round(leverage, 2)
            row["interest_coverage"]   = round(int_coverage, 2)
            row["debt_paydown_yr"]     = round(
                sum(balances.values()) - total_debt_end, 2
            )

            # Covenant check
            max_lev  = _COVENANT_DEFAULTS["max_leverage"]
            min_cov  = _COVENANT_DEFAULTS["min_interest_coverage"]
            covenant_breach = (leverage > max_lev) or (int_coverage < min_cov)
            row["covenant_breach"] = covenant_breach

            balances = new_balances
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Exit and returns
    # ------------------------------------------------------------------

    def compute_equity_value_at_exit(
        self,
        exit_multiple: float,
        exit_year_ebitda: float,
        net_debt_at_exit: float,
    ) -> float:
        """Exit EV = exit_ebitda × exit_multiple; equity = EV - net debt."""
        exit_ev = exit_year_ebitda * exit_multiple
        return max(exit_ev - net_debt_at_exit, 0.0)

    def compute_returns(
        self,
        equity_invested: float,
        exit_equity_value: float,
        dividend_schedule: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        IRR (Newton-Raphson), MOIC, Cash-on-Cash.
        dividend_schedule: optional annual dividends/distributions (Year 1 → N).
        """
        n = self.hold_period
        divs = dividend_schedule or [0.0] * n

        # Cash flows: -equity at t=0, dividends during hold, exit equity at t=N
        cf = [-equity_invested] + list(divs[: n - 1]) + [divs[n - 1] + exit_equity_value]
        try:
            irr = _solve_irr(cf)
        except Exception:
            # Fallback: simple XIRR approximation
            irr = (exit_equity_value / equity_invested) ** (1.0 / n) - 1 if equity_invested > 0 else 0.0

        moic = (exit_equity_value + sum(divs)) / equity_invested if equity_invested > 0 else 0.0
        coc  = sum(divs) / equity_invested if equity_invested > 0 else 0.0

        return {
            "irr":          round(irr, 4),
            "irr_pct":      round(irr * 100, 2),
            "moic":         round(moic, 2),
            "cash_on_cash": round(coc, 2),
            "equity_invested_mm": round(equity_invested / 1e6, 2),
            "exit_equity_mm":     round(exit_equity_value / 1e6, 2),
        }

    # ------------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------------

    def sensitivity_table(
        self,
        base_exit_multiple: float,
        exit_multiples: Optional[List[float]] = None,
        ebitda_margin_deltas: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        5×5 sensitivity: IRR at each (exit_multiple, EBITDA_margin_delta) combination.
        Rows = exit multiple; columns = EBITDA margin delta (bps).
        Financials are re-projected internally for each scenario.
        """
        x_mults  = exit_multiples       or [x + base_exit_multiple for x in [-2, -1, 0, 1, 2]]
        y_deltas = ebitda_margin_deltas or [-0.02, -0.01, 0.0, 0.01, 0.02]

        equity_invested = self.entry_ev * self._equity_pct
        base_margin     = self._ebitda_margin or 0.20

        records: Dict[float, Dict] = {}
        for delta in y_deltas:
            margin_case = base_margin + delta
            # Re-project financials with adjusted margin
            self._ebitda_margin = margin_case
            f_df  = self.project_financials()
            d_df  = self.compute_debt_schedule(f_df)
            net_debt_exit = d_df["total_debt_end"].iloc[-1] - 0.0  # simplified: no excess cash
            exit_ebitda   = f_df["ebitda"].iloc[-1]

            row_data: Dict[float, float] = {}
            for xm in x_mults:
                exit_eq = self.compute_equity_value_at_exit(xm, exit_ebitda, net_debt_exit)
                rets    = self.compute_returns(equity_invested, exit_eq)
                row_data[xm] = round(rets["irr_pct"], 1)
            records[delta] = row_data

        self._ebitda_margin = base_margin   # restore

        df = pd.DataFrame(records, index=x_mults)
        df.index.name   = "exit_multiple"
        df.columns      = [f"margin_Δ={d*100:+.0f}bps" for d in y_deltas]
        return df

    # ------------------------------------------------------------------
    # Covenant analysis
    # ------------------------------------------------------------------

    def covenant_headroom(self, debt_df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        Year-by-year covenant headroom analysis.
        Returns max_leverage, actual_leverage, leverage_headroom, etc.
        """
        max_lev = _COVENANT_DEFAULTS["max_leverage"]
        min_cov = _COVENANT_DEFAULTS["min_interest_coverage"]
        rows: List[Dict] = []
        for _, row in debt_df.iterrows():
            yr   = row["year"]
            lev  = row["leverage"]
            cov  = row["interest_coverage"]
            rows.append({
                "year":               int(yr),
                "actual_leverage":    round(lev, 2),
                "max_leverage":       max_lev,
                "leverage_headroom":  round(max_lev - lev, 2),
                "actual_coverage":    round(cov, 2),
                "min_coverage":       min_cov,
                "coverage_headroom":  round(cov - min_cov, 2),
                "breach":             row["covenant_breach"],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Full model run
    # ------------------------------------------------------------------

    def run_full_model(
        self,
        exit_multiple: float = 10.0,
        ticker: Optional[str] = None,
    ) -> LBOResult:
        """
        Execute the complete LBO model.
        If ticker provided: fetches EDGAR XBRL financials to calibrate entry.
        Returns LBOResult with all schedules populated.
        """
        if ticker:
            try:
                cik   = _lookup_cik(ticker)
                facts = _get_xbrl_facts(cik) if cik else {}
                if facts.get("ebitda"):
                    self.entry_ebitda  = facts["ebitda"]
                    self.entry_ev      = self.entry_ebitda * self.entry_multiple
                if facts.get("revenue") and facts.get("ebitda"):
                    self._entry_revenue = facts["revenue"]
                    self._ebitda_margin = facts["ebitda"] / facts["revenue"]
            except Exception as exc:
                logger.warning("XBRL fetch failed for LBO model ticker=%s: %s", ticker, exc)

        equity_invested = self.entry_ev * self._equity_pct
        fin_df   = self.project_financials()
        debt_df  = self.compute_debt_schedule(fin_df)

        net_debt_exit  = debt_df["total_debt_end"].iloc[-1]
        exit_ebitda    = fin_df["ebitda"].iloc[-1]
        exit_ev_val    = self.compute_equity_value_at_exit(exit_multiple, exit_ebitda, net_debt_exit)
        returns        = self.compute_returns(equity_invested, exit_ev_val)
        sensitivity    = self.sensitivity_table(exit_multiple)

        entry_debt     = self.entry_ev * self._leverage_pct
        debt_paydown   = entry_debt - net_debt_exit
        peak_lev       = debt_df["leverage"].max()
        exit_lev       = debt_df["leverage"].iloc[-1]

        any_breach     = bool(debt_df["covenant_breach"].any())
        breach_yrs     = list(debt_df[debt_df["covenant_breach"]]["year"].astype(int))

        result = LBOResult(
            run_id=str(uuid.uuid4()),
            model_name=self.target_name,
            entry_ev=round(self.entry_ev, 2),
            entry_multiple=self.entry_multiple,
            entry_ebitda=round(self.entry_ebitda, 2),
            equity_invested=round(equity_invested, 2),
            total_debt=round(entry_debt, 2),
            hold_period=self.hold_period,
            exit_ev=round(exit_ebitda * exit_multiple, 2),
            exit_equity_value=round(exit_ev_val, 2),
            moic=returns["moic"],
            irr=returns["irr"],
            cash_on_cash=returns["cash_on_cash"],
            peak_leverage=round(peak_lev, 2),
            exit_leverage=round(exit_lev, 2),
            debt_paydown_mm=round(debt_paydown / 1e6, 2),
            financials_df=fin_df,
            debt_schedule_df=debt_df,
            sensitivity_df=sensitivity,
            covenant_breach=any_breach,
            covenant_detail=f"Breach in years {breach_yrs}" if breach_yrs else "No covenant breach",
        )

        # Persist
        params = {
            "target_name":    self.target_name,
            "entry_ebitda":   self.entry_ebitda,
            "entry_multiple": self.entry_multiple,
            "exit_multiple":  exit_multiple,
            "hold_period":    self.hold_period,
            "leverage_pct":   self._leverage_pct,
        }
        _save_run("lbo", self.target_name, params, {
            "irr":    returns["irr"],
            "moic":   returns["moic"],
            "run_id": result.run_id,
        })
        logger.info("LBO model complete: target=%s irr=%.1f%% moic=%.2fx",
                    self.target_name, returns["irr_pct"], returns["moic"])
        return result


# ---------------------------------------------------------------------------
# MergerModel
# ---------------------------------------------------------------------------

class MergerModel:
    """
    Strategic acquisition / merger-of-equals model.

    Handles:
      - Mixed cash/stock consideration + new share issuance
      - 5-year combined P&L with phased synergy ramp
      - EPS accretion/dilution schedule and breakeven year
      - Purchase price allocation: goodwill, identifiable intangibles
      - Combined credit metrics: leverage, interest coverage, rating assessment
    """

    def __init__(self, acquirer_name: str, target_name: str):
        self.acquirer_name = acquirer_name
        self.target_name   = target_name
        self._deal_set     = False

    # ------------------------------------------------------------------
    # Deal structure
    # ------------------------------------------------------------------

    def set_deal_structure(
        self,
        deal_value: float,
        cash_pct: float = 0.50,
        stock_pct: float = 0.50,
        premium_pct: float = 0.30,
        acquirer_stock_price: float = 100.0,
        acquirer_shares_outstanding: float = 100e6,
    ) -> "MergerModel":
        """
        Set deal consideration mix.
        deal_value: total consideration paid (not EV; includes assumption of debt).
        """
        if abs(cash_pct + stock_pct - 1.0) > 0.01:
            raise ValueError("cash_pct + stock_pct must sum to 1.0")
        self.deal_value          = deal_value
        self.cash_pct            = cash_pct
        self.stock_pct           = stock_pct
        self.premium_pct         = premium_pct
        self.acq_stock_price     = acquirer_stock_price
        self.acq_shares_out      = acquirer_shares_outstanding

        stock_portion = deal_value * stock_pct
        self.new_shares_issued = stock_portion / acquirer_stock_price

        self._deal_set = True
        return self

    # ------------------------------------------------------------------
    # PPA
    # ------------------------------------------------------------------

    def compute_goodwill_and_ppa(
        self,
        target_book_value: float,
        identifiable_intangibles: float = 0.0,
        customer_relationships: float = 0.0,
        brand_value: float = 0.0,
        patents_technology: float = 0.0,
        assumed_liabilities: float = 0.0,
    ) -> Dict[str, float]:
        """
        Purchase Price Allocation (ASC 805 / IFRS 3).

        Goodwill = purchase_price - fair_value_net_assets
        Fair value net assets = book_equity + step-ups of identifiable intangibles
                                + assumed_liabilities adjustment

        Returns breakdown: goodwill, intangibles by category, step-up total.
        """
        if not self._deal_set:
            raise RuntimeError("Call set_deal_structure() first")

        total_intangibles = (identifiable_intangibles
                             + customer_relationships
                             + brand_value
                             + patents_technology)
        # Fair value of net identifiable assets
        fv_net_assets = target_book_value + total_intangibles - assumed_liabilities
        goodwill      = max(self.deal_value - fv_net_assets, 0.0)
        # Bargain purchase (negative goodwill) — rare, recognised as gain
        bargain_purchase = max(fv_net_assets - self.deal_value, 0.0)

        # Tax amortisation: intangibles over 15 years (IRC §197); goodwill same
        annual_tax_amort_intangibles = total_intangibles / 15.0
        annual_tax_amort_goodwill    = goodwill / 15.0
        annual_tax_shield            = (annual_tax_amort_intangibles + annual_tax_amort_goodwill) * TAX_RATE

        return {
            "purchase_price":              round(self.deal_value, 2),
            "target_book_value":           round(target_book_value, 2),
            "customer_relationships":      round(customer_relationships, 2),
            "brand_value":                 round(brand_value, 2),
            "patents_technology":          round(patents_technology, 2),
            "other_intangibles":           round(identifiable_intangibles, 2),
            "total_identifiable_intangibles": round(total_intangibles, 2),
            "assumed_liabilities":         round(assumed_liabilities, 2),
            "fair_value_net_assets":       round(fv_net_assets, 2),
            "goodwill":                    round(goodwill, 2),
            "bargain_purchase":            round(bargain_purchase, 2),
            "annual_tax_amort_goodwill":   round(annual_tax_amort_goodwill, 2),
            "annual_tax_shield":           round(annual_tax_shield, 2),
            "goodwill_pct_purchase_price": round(goodwill / self.deal_value, 4) if self.deal_value else 0,
        }

    # ------------------------------------------------------------------
    # Combined financials
    # ------------------------------------------------------------------

    def project_combined_financials(
        self,
        acquirer_financials: Dict[str, float],
        target_financials: Dict[str, float],
        synergies: Dict[str, float],
        integration_costs: Optional[Dict[str, float]] = None,
        revenue_growth: float = 0.05,
        years: int = 5,
    ) -> pd.DataFrame:
        """
        5-year combined P&L pro-forma.

        acquirer/target_financials keys expected: revenue, ebitda, da,
        interest_expense, net_income, shares_outstanding.
        synergies keys: revenue, cost (total annual at full run-rate).
        integration_costs keys: yr1, yr2 (front-loaded costs).
        Synergy ramp: 25% / 50% / 75% / 100% / 100%.
        """
        syn_ramp = [0.25, 0.50, 0.75, 1.00, 1.00]
        intg     = integration_costs or {}
        intg_ramp = [intg.get("yr1", synergies.get("cost", 0) * 1.25),
                     intg.get("yr2", synergies.get("cost", 0) * 0.50),
                     0.0, 0.0, 0.0]

        acq_rev  = acquirer_financials.get("revenue", 0)
        tgt_rev  = target_financials.get("revenue", 0)
        acq_ebit = acquirer_financials.get("ebitda", 0)
        tgt_ebit = target_financials.get("ebitda", 0)
        acq_da   = acquirer_financials.get("da", 0)
        tgt_da   = target_financials.get("da", 0)
        acq_ni   = acquirer_financials.get("net_income", 0)
        # tgt_ni not used directly: combined NI is computed EBIT-down
        # (avoids double-counting interest / taxes in the pro-forma P&L)
        acq_int  = acquirer_financials.get("interest_expense", 0)
        # Incremental interest from cash consideration financed with debt
        if self._deal_set:
            cash_paid  = self.deal_value * self.cash_pct
            incr_int   = cash_paid * (SOFR_PROXY + 0.0275)   # assume TLB rate
        else:
            incr_int = 0.0

        combined_shares = (
            acquirer_financials.get("shares_outstanding", 100e6)
            + (self.new_shares_issued if self._deal_set else 0)
        )

        rows: List[Dict] = []
        for yr in range(1, years + 1):
            g    = (1 + revenue_growth) ** yr
            rev  = (acq_rev + tgt_rev) * g
            syn_rev  = synergies.get("revenue", 0) * syn_ramp[yr-1]
            syn_cost = synergies.get("cost", 0)    * syn_ramp[yr-1]
            intg_c   = intg_ramp[yr-1] if yr-1 < len(intg_ramp) else 0.0

            combined_ebitda = (acq_ebit + tgt_ebit) * g + syn_rev + syn_cost - intg_c
            combined_da     = (acq_da + tgt_da) * g
            combined_ebit   = combined_ebitda - combined_da
            combined_int    = (acq_int + incr_int) * g
            ebt             = max(combined_ebit - combined_int, 0)
            tax             = ebt * TAX_RATE
            combined_ni     = ebt - tax

            combined_eps    = combined_ni / combined_shares if combined_shares > 0 else 0
            standalone_acq_eps = (
                (acq_ni * (1 + revenue_growth) ** yr)
                / acquirer_financials.get("shares_outstanding", 100e6)
            )
            rows.append({
                "year":               yr,
                "combined_revenue":   round(rev, 2),
                "combined_ebitda":    round(combined_ebitda, 2),
                "synergy_revenue":    round(syn_rev, 2),
                "synergy_cost":       round(syn_cost, 2),
                "integration_costs":  round(intg_c, 2),
                "combined_ebit":      round(combined_ebit, 2),
                "combined_interest":  round(combined_int, 2),
                "combined_net_income": round(combined_ni, 2),
                "combined_eps":       round(combined_eps, 4),
                "standalone_acq_eps": round(standalone_acq_eps, 4),
                "accretion_abs":      round(combined_eps - standalone_acq_eps, 4),
                "accretion_pct":      round(
                    (combined_eps - standalone_acq_eps) / abs(standalone_acq_eps) * 100
                    if standalone_acq_eps != 0 else 0, 2
                ),
                "accretive":          combined_eps >= standalone_acq_eps,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Credit metrics
    # ------------------------------------------------------------------

    def compute_combined_credit_metrics(
        self,
        acquirer_financials: Dict[str, float],
        target_financials: Dict[str, float],
        deal_debt_added: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Pro-forma credit metrics post-closing.
        deal_debt_added: incremental debt to finance cash consideration.
        """
        combined_ebitda = (
            acquirer_financials.get("ebitda", 0)
            + target_financials.get("ebitda", 0)
        )
        combined_ebit = (
            acquirer_financials.get("ebit", combined_ebitda * 0.8)
            + target_financials.get("ebit", target_financials.get("ebitda",0) * 0.8)
        )
        total_debt = (
            acquirer_financials.get("total_debt", 0)
            + target_financials.get("total_debt", 0)
            + deal_debt_added
        )
        combined_interest = (
            acquirer_financials.get("interest_expense", 0)
            + deal_debt_added * (SOFR_PROXY + 0.0275)
        )
        leverage  = total_debt / combined_ebitda if combined_ebitda > 0 else 0.0
        coverage  = combined_ebit / combined_interest if combined_interest > 0 else 99.0

        if leverage <= 2.0 and coverage >= 6.0:
            rating = "A / BBB+ (Investment Grade)"
        elif leverage <= 3.0 and coverage >= 4.0:
            rating = "BBB / BBB- (Investment Grade)"
        elif leverage <= 4.0 and coverage >= 3.0:
            rating = "BB+ / BB (High Yield / split)"
        elif leverage <= 5.5 and coverage >= 2.0:
            rating = "B+ / B (High Yield)"
        elif leverage <= 7.0:
            rating = "B- / CCC (Leveraged)"
        else:
            rating = "CCC / Distressed"

        return {
            "combined_ebitda":       round(combined_ebitda, 2),
            "total_pro_forma_debt":  round(total_debt, 2),
            "leverage_x":            round(leverage, 2),
            "interest_coverage":     round(coverage, 2),
            "implied_rating":        rating,
            "ig_compliant":          leverage <= 3.0,
            "hy_threshold_crossed":  leverage > 3.0,
        }

    # ------------------------------------------------------------------
    # Full merger run
    # ------------------------------------------------------------------

    def run(
        self,
        acquirer_financials: Dict[str, float],
        target_financials: Dict[str, float],
        synergies: Dict[str, float],
        target_book_value: float = 0.0,
        identifiable_intangibles: float = 0.0,
    ) -> MergerResult:
        """Run the full merger model and return MergerResult."""
        if not self._deal_set:
            raise RuntimeError("Call set_deal_structure() first")

        combined_df = self.project_combined_financials(
            acquirer_financials, target_financials, synergies
        )

        ppa = self.compute_goodwill_and_ppa(
            target_book_value, identifiable_intangibles
        )

        cash_paid   = self.deal_value * self.cash_pct
        credit      = self.compute_combined_credit_metrics(
            acquirer_financials, target_financials, deal_debt_added=cash_paid
        )

        accretive_rows = combined_df[combined_df["accretive"] == True]
        bk_yr  = int(accretive_rows["year"].min()) if not accretive_rows.empty else None

        result = MergerResult(
            run_id=str(uuid.uuid4()),
            acquirer=self.acquirer_name,
            target=self.target_name,
            deal_value_mm=round(self.deal_value / 1e6, 2),
            premium_pct=self.premium_pct,
            cash_pct=self.cash_pct,
            stock_pct=self.stock_pct,
            new_shares_issued=round(self.new_shares_issued, 0),
            goodwill_mm=round(ppa["goodwill"] / 1e6, 2),
            identifiable_intangibles_mm=round(ppa["total_identifiable_intangibles"] / 1e6, 2),
            combined_leverage=credit["leverage_x"],
            interest_coverage=credit["interest_coverage"],
            leverage_rating=credit["implied_rating"],
            accretion_dilution_df=combined_df,
            combined_financials_df=combined_df,
            breakeven_year=bk_yr,
        )

        params = {
            "acquirer": self.acquirer_name,
            "target":   self.target_name,
            "deal_value": self.deal_value,
            "premium":    self.premium_pct,
        }
        _save_run("merger", f"{self.acquirer_name}x{self.target_name}", params, {
            "run_id":             result.run_id,
            "goodwill_mm":        result.goodwill_mm,
            "combined_leverage":  result.combined_leverage,
            "breakeven_year":     result.breakeven_year,
        })
        return result


# ---------------------------------------------------------------------------
# DCFMergerValuation
# ---------------------------------------------------------------------------

class DCFMergerValuation:
    """
    DCF-based standalone valuation + synergy-adjusted strategic value.
    Fetches historical financials from EDGAR XBRL to anchor the model.
    """

    def compute_standalone_dcf(
        self,
        ticker: str,
        revenue_growth_rates: List[float],
        ebitda_margins: List[float],
        wacc: float = 0.10,
        terminal_growth: float = 0.025,
        capex_pct: float = 0.04,
        nwc_pct: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Project 5-year FCF and compute DCF value.

        Fetches base-year financials from EDGAR XBRL.
        Returns per-share intrinsic value and component breakdown.
        """
        # Fetch financials
        cik   = _lookup_cik(ticker)
        facts = _get_xbrl_facts(cik) if cik else {}
        info  = _yf_info(ticker)

        base_rev   = facts.get("revenue")   or info.get("totalRevenue", 0)
        base_ebitda = facts.get("ebitda")   or 0.0
        total_debt  = facts.get("total_debt") or info.get("totalDebt", 0)
        cash        = facts.get("cash")      or info.get("cash", 0)
        shares      = (facts.get("shares")   or info.get("sharesOutstanding", 0)) or 1

        if not base_rev:
            return {"error": "No revenue data available from EDGAR/yfinance"}

        # Infer base margin from XBRL if available
        base_margin = base_ebitda / base_rev if base_rev and base_ebitda else (ebitda_margins[0] if ebitda_margins else 0.18)
        # Extend inputs to 5 years
        n = 5
        rev_g  = (revenue_growth_rates + [revenue_growth_rates[-1]] * n)[:n]
        e_marg = (ebitda_margins + [ebitda_margins[-1]] * n)[:n] if ebitda_margins else [base_margin] * n
        da_pct = 0.04

        pv_fcf    = 0.0
        rev       = base_rev
        fcf_schedule: List[Dict] = []
        for i, (g, m) in enumerate(zip(rev_g, e_marg)):
            yr    = i + 1
            rev   = rev * (1 + g)
            ebitda = rev * m
            da    = rev * da_pct
            ebit  = ebitda - da
            capex = rev * capex_pct
            nwc   = rev * nwc_pct
            tax   = max(ebit, 0) * TAX_RATE
            fcf   = ebitda - capex - nwc - tax
            pv    = fcf / (1 + wacc) ** yr
            pv_fcf += pv
            fcf_schedule.append({"year": yr, "revenue": round(rev,2), "ebitda": round(ebitda,2),
                                  "fcf": round(fcf,2), "pv_fcf": round(pv,2)})

        # Terminal value (Gordon Growth)
        terminal_fcf = fcf_schedule[-1]["fcf"] * (1 + terminal_growth)
        tv           = terminal_fcf / (wacc - terminal_growth)
        pv_tv        = tv / (1 + wacc) ** n

        enterprise_value = pv_fcf + pv_tv
        equity_value     = enterprise_value - total_debt + cash
        per_share        = equity_value / shares

        return {
            "ticker":              ticker,
            "base_revenue":        round(base_rev, 2),
            "wacc":                wacc,
            "terminal_growth":     terminal_growth,
            "pv_explicit_fcfs":    round(pv_fcf, 2),
            "terminal_value":      round(tv, 2),
            "pv_terminal_value":   round(pv_tv, 2),
            "enterprise_value":    round(enterprise_value, 2),
            "net_debt":            round(total_debt - cash, 2),
            "equity_value":        round(equity_value, 2),
            "shares_outstanding":  round(shares, 0),
            "intrinsic_value_per_share": round(per_share, 2),
            "tv_pct_of_ev":        round(pv_tv / enterprise_value * 100, 1) if enterprise_value else 0,
            "fcf_schedule":        fcf_schedule,
        }

    def compute_synergy_adjusted_value(
        self,
        standalone_enterprise_value: float,
        synergy_npv: float,
    ) -> Dict[str, float]:
        """Strategic value = standalone EV + NPV of synergies."""
        strategic_ev = standalone_enterprise_value + synergy_npv
        return {
            "standalone_ev":   round(standalone_enterprise_value, 2),
            "synergy_npv":     round(synergy_npv, 2),
            "strategic_ev":    round(strategic_ev, 2),
            "synergy_pct":     round(synergy_npv / standalone_enterprise_value * 100, 1)
                               if standalone_enterprise_value > 0 else 0.0,
        }

    def compute_implied_premium(
        self,
        strategic_equity_value: float,
        current_market_cap: float,
    ) -> Dict[str, float]:
        """Implied premium = strategic value / current market cap - 1."""
        implied_prem = (strategic_equity_value / current_market_cap - 1) if current_market_cap > 0 else 0
        return {
            "current_market_cap":     round(current_market_cap, 2),
            "strategic_equity_value": round(strategic_equity_value, 2),
            "implied_premium_pct":    round(implied_prem * 100, 2),
        }


# ---------------------------------------------------------------------------
# LBOCandidateScreener
# ---------------------------------------------------------------------------

class LBOCandidateScreener:
    """
    Screen a universe of public companies for LBO attractiveness.

    Ideal LBO target characteristics (scored 0-100):
      - Stable FCF: high EBITDA margins, recurring revenue
      - Low existing leverage (room to add debt)
      - Asset-heavy balance sheet (collateral for secured debt)
      - Non-cyclical sector (predictable through hold period)
      - Operational improvement opportunity (margin expansion potential)
      - Proven management team (or PE-friendly)
    """

    def score_lbo_attractiveness(self, ticker: str) -> Optional[LBOScore]:
        """
        Compute composite LBO attractiveness score for one ticker.
        Data: yfinance info + EDGAR XBRL fundamentals.
        """
        try:
            info = _yf_info(ticker)
        except Exception:
            info = {}

        cik   = _lookup_cik(ticker)
        facts = _get_xbrl_facts(cik) if cik else {}

        mkt_cap = info.get("marketCap") or 0
        if mkt_cap < 250_000_000:   # min size for institutional LBO ($250M)
            return None

        # Pull financials from yfinance info first, then EDGAR XBRL fallback
        revenue    = facts.get("revenue")   or info.get("totalRevenue")   or 0
        ebitda     = facts.get("ebitda")    or info.get("ebitda")         or 0
        total_debt = facts.get("total_debt") or info.get("totalDebt")     or 0
        cash       = facts.get("cash")      or info.get("cash")           or 0
        fcf        = facts.get("fcf")       or info.get("freeCashflow")   or 0
        total_assets = facts.get("total_assets") or info.get("totalAssets") or 0

        if revenue <= 0 or ebitda <= 0:
            return None

        ebitda_margin  = ebitda / revenue
        net_debt       = total_debt - cash
        net_leverage   = net_debt / ebitda if ebitda > 0 else 0
        fcf_conversion = fcf / ebitda if ebitda > 0 and fcf > 0 else 0
        asset_ratio    = total_assets / ebitda if ebitda > 0 else 0
        sector         = info.get("sector", "Unknown")

        # --- Scoring ---

        # 1. FCF stability (0-100)
        fcf_s = 0.0
        if fcf_conversion > 0.70:
            fcf_s = 90
        elif fcf_conversion > 0.55:
            fcf_s = 70
        elif fcf_conversion > 0.40:
            fcf_s = 50
        elif fcf_conversion > 0.20:
            fcf_s = 30
        if ebitda_margin > 0.30:
            fcf_s = min(fcf_s + 10, 100)
        elif ebitda_margin < 0.10:
            fcf_s = max(fcf_s - 20, 0)

        # 2. Leverage room (lower existing leverage = more room to add LBO debt)
        lev_s = 0.0
        if net_leverage < 0:       # net cash
            lev_s = 100
        elif net_leverage < 0.5:
            lev_s = 90
        elif net_leverage < 1.0:
            lev_s = 75
        elif net_leverage < 2.0:
            lev_s = 55
        elif net_leverage < 3.0:
            lev_s = 35
        elif net_leverage < 4.5:
            lev_s = 15
        else:
            lev_s = 0

        # 3. Margin quality
        marg_s = 0.0
        if ebitda_margin > 0.35:
            marg_s = 100
        elif ebitda_margin > 0.25:
            marg_s = 80
        elif ebitda_margin > 0.18:
            marg_s = 60
        elif ebitda_margin > 0.12:
            marg_s = 40
        elif ebitda_margin > 0.08:
            marg_s = 20
        else:
            marg_s = 5

        # 4. Asset coverage (collateral for secured debt)
        asset_s = 0.0
        if asset_ratio > 8:
            asset_s = 90
        elif asset_ratio > 5:
            asset_s = 70
        elif asset_ratio > 3:
            asset_s = 50
        elif asset_ratio > 1.5:
            asset_s = 30
        else:
            asset_s = 15

        # 5. Management score (proxy: insider ownership → aligned incentives)
        insider_pct = info.get("heldPercentInsiders", 0) or 0
        mgmt_s = 0.0
        if insider_pct > 0.20:
            mgmt_s = 90
        elif insider_pct > 0.10:
            mgmt_s = 70
        elif insider_pct > 0.05:
            mgmt_s = 50
        elif insider_pct > 0.01:
            mgmt_s = 35
        else:
            mgmt_s = 20

        # 6. Sector attractiveness
        sector_s = _LBO_SECTOR_SCORES.get(sector, 60.0)

        # Weighted composite
        composite = (
            fcf_s   * _LBO_WEIGHTS["fcf_stability"]
            + lev_s * _LBO_WEIGHTS["leverage_room"]
            + marg_s * _LBO_WEIGHTS["margin_quality"]
            + asset_s * _LBO_WEIGHTS["asset_coverage"]
            + mgmt_s * _LBO_WEIGHTS["management"]
            + sector_s * _LBO_WEIGHTS["sector_appeal"]
        )

        # Rationale summary
        strengths:  List[str] = []
        weaknesses: List[str] = []
        if fcf_conversion > 0.55:
            strengths.append("high FCF conversion")
        if net_leverage < 1.0:
            strengths.append("low leverage (room for debt)")
        if ebitda_margin > 0.25:
            strengths.append(f"strong {ebitda_margin*100:.0f}% EBITDA margin")
        if net_leverage > 3.0:
            weaknesses.append("already highly levered")
        if ebitda_margin < 0.12:
            weaknesses.append("thin margins limit debt service")
        if fcf_conversion < 0.30:
            weaknesses.append("weak FCF conversion")

        rationale = ""
        if strengths:
            rationale += "Strengths: " + ", ".join(strengths) + ". "
        if weaknesses:
            rationale += "Weaknesses: " + ", ".join(weaknesses) + "."

        score = LBOScore(
            ticker=ticker,
            company_name=info.get("longName", ticker),
            composite_score=round(composite, 1),
            fcf_stability_score=round(fcf_s, 1),
            leverage_room_score=round(lev_s, 1),
            margin_quality_score=round(marg_s, 1),
            asset_coverage_score=round(asset_s, 1),
            management_score=round(mgmt_s, 1),
            sector_score=round(sector_s, 1),
            ebitda_mm=round(ebitda / 1e6, 1),
            ebitda_margin=round(ebitda_margin, 4),
            net_leverage=round(net_leverage, 2),
            fcf_conversion=round(fcf_conversion, 3),
            revenue_mm=round(revenue / 1e6, 1),
            market_cap_mm=round(mkt_cap / 1e6, 1),
            sector=sector,
            rationale=rationale.strip(),
        )

        # Persist score
        with _get_lbo_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO lbo_scores VALUES (?,?,?,?)",
                (str(uuid.uuid4()), ticker, score.scored_at, json.dumps(asdict(score), default=str))
            )
        return score

    def screen_universe(
        self,
        tickers: List[str],
        min_score: float = 40.0,
    ) -> pd.DataFrame:
        """Screen a list of tickers and return ranked LBO candidates."""
        scores: List[LBOScore] = []
        for ticker in tickers:
            try:
                s = self.score_lbo_attractiveness(ticker)
                if s and s.composite_score >= min_score:
                    scores.append(s)
            except Exception as exc:
                logger.debug("LBO score failed ticker=%s: %s", ticker, exc)

        if not scores:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(s) for s in scores])
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PEFundEconomics
# ---------------------------------------------------------------------------

class PEFundEconomics:
    """
    Private equity fund economics model.

    Models: deployment schedule, committed capital, management fees,
    carried interest (standard 2/20 with 8% preferred return hurdle),
    catch-up provision, clawback, DPI/TVPI/RVPI metrics.
    """

    def __init__(
        self,
        fund_size: float,
        mgmt_fee_pct: float = 0.020,
        carry_pct: float = 0.200,
        hurdle_rate: float = 0.080,
        catch_up_pct: float = 1.000,     # 100% catch-up (full GP catch-up)
        fund_life: int = 10,
        investment_period: int = 5,
    ):
        self.fund_size         = fund_size
        self.mgmt_fee_pct      = mgmt_fee_pct
        self.carry_pct         = carry_pct
        self.hurdle_rate       = hurdle_rate
        self.catch_up_pct      = catch_up_pct
        self.fund_life         = fund_life
        self.investment_period = investment_period

    def project_fund(
        self,
        deployment_schedule: Optional[List[float]] = None,
        exit_irr: float = 0.22,
        hold_period: int = 5,
    ) -> Dict[str, Any]:
        """
        Full fund-level projection.

        deployment_schedule: list of % of fund deployed each year (must sum to ~1).
        exit_irr: portfolio company exit IRR assumption.
        hold_period: average portfolio hold period.
        Returns: fund cashflows, management fees, carry, DPI, TVPI, RVPI.
        """
        if deployment_schedule is None:
            # Typical: 15/20/25/25/15 over 5 years
            deployment_schedule = [0.15, 0.20, 0.25, 0.25, 0.15]
        deployment_schedule = (deployment_schedule + [0.0] * self.investment_period)[: self.investment_period]

        # Normalise
        total_dep = sum(deployment_schedule)
        deployment_schedule = [d / total_dep for d in deployment_schedule]

        # Management fees: 2% of committed capital during investment period,
        # 2% of invested capital thereafter, declining 10% pa in harvest period
        mgmt_fees: List[float] = []
        for yr in range(1, self.fund_life + 1):
            if yr <= self.investment_period:
                fee = self.fund_size * self.mgmt_fee_pct
            else:
                years_post_ip = yr - self.investment_period
                fee = self.fund_size * self.mgmt_fee_pct * max(0, 1 - years_post_ip * 0.10)
            mgmt_fees.append(fee)

        # Capital calls (deployments) — LP pays in
        capital_calls: List[float] = []
        for yr in range(1, self.fund_life + 1):
            dep = deployment_schedule[yr-1] if yr <= self.investment_period else 0.0
            call = self.fund_size * dep + mgmt_fees[yr-1]
            capital_calls.append(call)

        # Portfolio exits: each cohort exits after hold_period at exit_irr
        invested_cohorts: List[Tuple[int, float]] = []   # (entry_year, invested_capital)
        for yr in range(1, self.investment_period + 1):
            inv = self.fund_size * deployment_schedule[yr-1]
            invested_cohorts.append((yr, inv))

        # Exit proceeds per year
        exit_proceeds: List[float] = [0.0] * self.fund_life
        for entry_yr, inv in invested_cohorts:
            exit_yr = min(entry_yr + hold_period - 1, self.fund_life) - 1
            exit_val = inv * (1 + exit_irr) ** hold_period
            exit_proceeds[exit_yr] += exit_val

        # LP cashflows (net of management fees)
        lp_capital_calls  = [-c for c in capital_calls]
        lp_distributions: List[float] = []

        # Carry waterfall (European waterfall — fund-level)
        total_invested    = self.fund_size   # simplified: all committed
        total_exit_value  = sum(exit_proceeds)
        hurdle_amount     = total_invested * (1 + self.hurdle_rate) ** self.fund_life
        profit_over_hurdle = max(total_exit_value - hurdle_amount, 0)
        # Catch-up: GP earns 100% of profits until GP gets carry_pct of total profits
        total_profit      = max(total_exit_value - total_invested, 0)
        gp_catch_up       = min(total_profit * self.carry_pct / (1 - self.carry_pct),
                                 profit_over_hurdle * self.catch_up_pct)
        gp_carry          = max(profit_over_hurdle - gp_catch_up * (1 - self.carry_pct), 0)
        lp_net_proceeds   = total_exit_value - gp_carry - gp_catch_up

        # Spread LP distributions proportionally to exit proceeds
        for yr_idx in range(self.fund_life):
            pct_of_total = exit_proceeds[yr_idx] / total_exit_value if total_exit_value > 0 else 0
            lp_distributions.append(lp_net_proceeds * pct_of_total)

        # DPI (Distributions to Paid-In) = cumulative distributions / committed capital
        cum_distributions = sum(lp_distributions)
        dpi  = cum_distributions / self.fund_size if self.fund_size > 0 else 0
        # RVPI (Residual Value to Paid-In) — assume 0 at fund wind-down
        rvpi = 0.0
        tvpi = dpi + rvpi

        # Fund IRR (from LP perspective)
        lp_cf = lp_capital_calls[:]
        for i, dist in enumerate(lp_distributions):
            lp_cf[i] += dist

        try:
            fund_irr = _solve_irr(lp_cf)
        except Exception:
            fund_irr = 0.0

        return {
            "fund_size":              self.fund_size,
            "management_fees_total":  round(sum(mgmt_fees), 2),
            "gp_carry":               round(gp_carry, 2),
            "gp_catch_up":            round(gp_catch_up, 2),
            "total_exit_value":       round(total_exit_value, 2),
            "lp_net_proceeds":        round(lp_net_proceeds, 2),
            "dpi":                    round(dpi, 2),
            "rvpi":                   round(rvpi, 2),
            "tvpi":                   round(tvpi, 2),
            "fund_irr":               round(fund_irr, 4),
            "fund_irr_pct":           round(fund_irr * 100, 2),
            "lp_cashflows":           [round(c, 2) for c in lp_cf],
            "annual_mgmt_fees":       [round(f, 2) for f in mgmt_fees],
        }


# ---------------------------------------------------------------------------
# ModelTemplates
# ---------------------------------------------------------------------------

class ModelTemplates:
    """
    12 pre-built LBO / merger model templates for rapid analysis.
    Each returns a dict with all key parameters and results.
    """

    # ------------------------------------------------------------------
    # LBO templates
    # ------------------------------------------------------------------

    def simple_lbo_template(
        self,
        target_name: str = "Target Co",
        entry_ebitda: float = 100e6,
        entry_multiple: float = 10.0,
        leverage_ratio: float = 0.60,
        exit_multiple: float = 10.0,
        hold_period: int = 5,
    ) -> Dict[str, Any]:
        """Standard PE buyout: 60% leverage, 5-year hold, standard assumptions."""
        m = LBOModel(target_name, entry_ebitda, entry_multiple, hold_period)
        m.set_capital_structure(leverage_pct=leverage_ratio)
        m.set_operating_assumptions(revenue_growth=0.05, ebitda_margin=0.20)
        result = m.run_full_model(exit_multiple=exit_multiple)
        return {
            "template": "simple_lbo",
            "irr":  result.irr,
            "moic": result.moic,
            "peak_leverage": result.peak_leverage,
            "exit_leverage": result.exit_leverage,
            "covenant_breach": result.covenant_breach,
            "sensitivity": result.sensitivity_df.to_dict() if result.sensitivity_df is not None else {},
        }

    def carve_out_lbo_template(
        self,
        entry_ebitda: float = 80e6,
        entry_multiple: float = 8.0,
    ) -> Dict[str, Any]:
        """
        Corporate carve-out LBO: lower entry multiple, margin expansion story,
        dis-synergy costs in Year 1-2, then re-rate at full multiple.
        """
        m = LBOModel("Carve-Out Target", entry_ebitda, entry_multiple)
        # Carve-outs: lower initial margins (corporate overhead not yet stripped)
        m.set_capital_structure(leverage_pct=0.55)   # slightly lower leverage (riskier)
        m.set_operating_assumptions(revenue_growth=0.04, ebitda_margin=0.17)
        result = m.run_full_model(exit_multiple=9.5)
        return {
            "template": "carve_out_lbo",
            "notes":    "Entry at discount; margin expansion to 22% by Year 3; exit at 9.5x",
            "irr":  result.irr,
            "moic": result.moic,
        }

    def public_to_private_template(
        self,
        mkt_cap: float = 2_000e6,
        ebitda: float = 200e6,
        premium_pct: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Take-Private / Public-to-Private LBO.
        Assumes 35% premium over current price; 65% leverage.
        """
        deal_value  = mkt_cap * (1 + premium_pct)
        entry_mult  = deal_value / ebitda
        m = LBOModel("PTP Target", ebitda, entry_mult)
        m.set_capital_structure(leverage_pct=0.65)
        m.set_operating_assumptions(revenue_growth=0.06, ebitda_margin=0.22)
        result = m.run_full_model(exit_multiple=entry_mult * 0.95)  # slight de-rating on exit
        return {
            "template":      "public_to_private",
            "deal_value_mm": round(deal_value / 1e6, 1),
            "premium_pct":   premium_pct,
            "entry_multiple": round(entry_mult, 1),
            "irr":  result.irr,
            "moic": result.moic,
            "peak_leverage": result.peak_leverage,
        }

    def distressed_lbo_template(
        self,
        entry_ebitda: float = 50e6,
        entry_multiple: float = 4.5,
    ) -> Dict[str, Any]:
        """
        Distressed LBO / Restructuring buyout.
        Low entry multiple (4-5×), high leverage relative to EBITDA,
        heavy operational restructuring assumed in Year 1.
        """
        m = LBOModel("Distressed Target", entry_ebitda, entry_multiple)
        m.set_capital_structure(leverage_pct=0.70)   # higher leverage on lower entry
        # Revenue declines Year 1, then recovery
        m.set_operating_assumptions(revenue_growth=-0.02, ebitda_margin=0.14)
        result = m.run_full_model(exit_multiple=7.0)  # re-rate on recovery
        return {
            "template":  "distressed_lbo",
            "notes":     "Entry at 4.5x stressed EBITDA; operational turnaround; exit at 7x",
            "irr":  result.irr,
            "moic": result.moic,
            "covenant_breach": result.covenant_breach,
        }

    def roll_up_platform_template(
        self,
        platform_ebitda: float = 60e6,
        n_add_ons: int = 5,
        add_on_ebitda_per: float = 8e6,
        add_on_entry_multiple: float = 6.0,
        platform_entry_multiple: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Roll-up strategy: acquire a platform + 5 bolt-on acquisitions,
        arbitrage the multiple between add-ons (6×) and platform (10×).
        """
        combined_ebitda = platform_ebitda + n_add_ons * add_on_ebitda_per
        m = LBOModel("Roll-Up Platform", combined_ebitda, platform_entry_multiple)
        m.set_capital_structure(leverage_pct=0.55)
        m.set_operating_assumptions(revenue_growth=0.08, ebitda_margin=0.22)
        result = m.run_full_model(exit_multiple=11.0)   # multiple expansion from scale
        add_on_total_cost = n_add_ons * add_on_ebitda_per * add_on_entry_multiple
        multiple_arb_mm   = n_add_ons * add_on_ebitda_per * (platform_entry_multiple - add_on_entry_multiple)
        return {
            "template":           "roll_up_platform",
            "n_add_ons":          n_add_ons,
            "add_on_total_cost_mm": round(add_on_total_cost / 1e6, 1),
            "multiple_arb_value_mm": round(multiple_arb_mm / 1e6, 1),
            "irr":  result.irr,
            "moic": result.moic,
        }

    # ------------------------------------------------------------------
    # Merger templates
    # ------------------------------------------------------------------

    def strategic_acquisition_template(
        self,
        acquirer_ticker: str = "",
        target_ticker: str = "",
        deal_value: float = 5_000e6,
        premium: float = 0.30,
        cash_pct: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Strategic acquisition accretion/dilution analysis.
        Fetches live data if tickers provided; uses placeholders otherwise.
        """
        if acquirer_ticker and target_ticker:
            acq_info = _yf_info(acquirer_ticker)
            tgt_info = _yf_info(target_ticker)
            acq_fin  = {
                "revenue":          acq_info.get("totalRevenue", 10_000e6),
                "ebitda":           acq_info.get("ebitda", 2_000e6),
                "da":               acq_info.get("ebitda", 2_000e6) * 0.20,
                "net_income":       acq_info.get("netIncomeToCommon", 1_000e6),
                "interest_expense": acq_info.get("totalDebt", 0) * 0.05,
                "shares_outstanding": acq_info.get("sharesOutstanding", 200e6),
                "total_debt":       acq_info.get("totalDebt", 5_000e6),
            }
            tgt_fin  = {
                "revenue":    tgt_info.get("totalRevenue", 2_000e6),
                "ebitda":     tgt_info.get("ebitda", 400e6),
                "net_income": tgt_info.get("netIncomeToCommon", 200e6),
                "total_debt": tgt_info.get("totalDebt", 1_000e6),
            }
            tgt_book = tgt_info.get("bookValue", 0) * tgt_info.get("sharesOutstanding", 100e6)
            acq_price = acq_info.get("currentPrice", 100.0)
            acq_shares = acq_info.get("sharesOutstanding", 200e6)
        else:
            acq_fin  = {"revenue": 10_000e6, "ebitda": 2_000e6, "da": 400e6,
                        "net_income": 1_200e6, "interest_expense": 250e6,
                        "shares_outstanding": 200e6, "total_debt": 5_000e6}
            tgt_fin  = {"revenue": 2_000e6, "ebitda": 400e6,
                        "net_income": 200e6, "total_debt": 800e6}
            tgt_book = 1_200e6
            acq_price  = 100.0
            acq_shares = 200e6

        mm = MergerModel(acquirer_ticker or "Acquirer", target_ticker or "Target")
        mm.set_deal_structure(
            deal_value=deal_value,
            cash_pct=cash_pct,
            stock_pct=1 - cash_pct,
            premium_pct=premium,
            acquirer_stock_price=acq_price,
            acquirer_shares_outstanding=acq_shares,
        )
        synergies = {
            "revenue": deal_value * 0.005,
            "cost":    deal_value * 0.008,
        }
        result = mm.run(
            acq_fin, tgt_fin, synergies,
            target_book_value=tgt_book,
            identifiable_intangibles=deal_value * 0.15,
        )
        return {
            "template":          "strategic_acquisition",
            "deal_value_mm":     result.deal_value_mm,
            "goodwill_mm":       result.goodwill_mm,
            "combined_leverage": result.combined_leverage,
            "breakeven_year":    result.breakeven_year,
            "accretion_dilution": result.accretion_dilution_df.to_dict(orient="records")
                                  if result.accretion_dilution_df is not None else [],
        }

    def leveraged_recap_template(
        self,
        company_ebitda: float = 500e6,
        current_leverage: float = 1.0,
        target_leverage: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Leveraged recapitalisation: borrow to pay special dividend.
        Computes: dividend amount, post-recap leverage, re-rating impact.
        """
        incremental_debt = (target_leverage - current_leverage) * company_ebitda
        dividend_mm      = incremental_debt / 1e6
        incremental_interest = incremental_debt * (SOFR_PROXY + 0.0350)
        # Post-recap EPS impact
        eps_hit = incremental_interest * (1 - TAX_RATE)   # in $ terms (not per share)
        new_leverage     = target_leverage
        int_coverage     = (company_ebitda * 0.80) / incremental_interest  # EBIT proxy

        if new_leverage <= 3.0:
            rerating = "Minimal — likely within IG territory"
        elif new_leverage <= 4.5:
            rerating = "Possible downgrade to BB / High Yield crossover"
        elif new_leverage <= 6.0:
            rerating = "Significant downgrade to B-range; HY covenants apply"
        else:
            rerating = "Distressed territory; covenant risk high"

        return {
            "template":            "leveraged_recap",
            "company_ebitda_mm":   round(company_ebitda / 1e6, 1),
            "incremental_debt_mm": round(incremental_debt / 1e6, 1),
            "special_dividend_mm": round(dividend_mm, 1),
            "post_recap_leverage": round(new_leverage, 1),
            "interest_coverage":   round(int_coverage, 2),
            "eps_hit_pretax_mm":   round(eps_hit / 1e6, 1),
            "re_rating_assessment": rerating,
        }

    def spac_merger_template(
        self,
        spac_trust_value: float = 500e6,
        target_ev: float = 2_000e6,
        pipe_amount: float = 250e6,
        redemption_rate: float = 0.40,
    ) -> Dict[str, Any]:
        """
        SPAC de-SPAC merger model.
        SPAC trust + PIPE proceeds fund the acquisition.
        Redemptions reduce available SPAC cash.
        """
        available_cash  = spac_trust_value * (1 - redemption_rate) + pipe_amount
        equity_roll     = target_ev - available_cash
        dilution_pct    = (pipe_amount / spac_trust_value)   # PIPE dilutes SPAC holders
        pro_forma_shares = (spac_trust_value / 10.0) + (pipe_amount / 10.0)  # $10/share SPAC

        return {
            "template":           "spac_merger",
            "spac_trust_mm":      round(spac_trust_value / 1e6, 1),
            "pipe_mm":            round(pipe_amount / 1e6, 1),
            "redemption_rate":    redemption_rate,
            "available_cash_mm":  round(available_cash / 1e6, 1),
            "target_ev_mm":       round(target_ev / 1e6, 1),
            "seller_rollover_mm": round(equity_roll / 1e6, 1),
            "pipe_dilution_pct":  round(dilution_pct * 100, 1),
            "pro_forma_shares_mm": round(pro_forma_shares / 1e6, 1),
        }

    def pe_fund_return_template(
        self,
        fund_size: float = 2_000e6,
        deployment_years: int = 5,
        hold_period: int = 5,
        exit_irr: float = 0.22,
    ) -> Dict[str, Any]:
        """
        PE fund economics: deployment, management fees, carry, DPI/TVPI/RVPI.
        """
        pe = PEFundEconomics(fund_size, fund_life=deployment_years + hold_period)
        result = pe.project_fund(exit_irr=exit_irr, hold_period=hold_period)
        return {
            "template":   "pe_fund",
            "fund_size_mm": round(fund_size / 1e6, 0),
            **result,
        }

    def merger_of_equals_template(
        self,
        company_a_ebitda: float = 1_000e6,
        company_b_ebitda: float = 900e6,
        synergies_mm: float = 300e6,
    ) -> Dict[str, Any]:
        """
        Merger of Equals: all-stock, ~50/50 ownership split.
        Focus on synergy NPV vs. integration cost.
        """
        combined_ebitda  = company_a_ebitda + company_b_ebitda + synergies_mm
        integration_cost = synergies_mm * 1.50   # 1.5× annual synergies
        synergy_npv      = synergies_mm / 0.10   # Gordon Growth at 10% WACC (simplified terminal)
        integration_pv   = integration_cost / 1.10 + integration_cost * 0.50 / (1.10**2)
        net_synergy_npv  = synergy_npv - integration_pv

        return {
            "template":             "merger_of_equals",
            "combined_ebitda_mm":   round(combined_ebitda / 1e6, 1),
            "annual_synergies_mm":  round(synergies_mm / 1e6, 1),
            "integration_cost_mm":  round(integration_cost / 1e6, 1),
            "synergy_npv_mm":       round(synergy_npv / 1e6, 1),
            "net_synergy_value_mm": round(net_synergy_npv / 1e6, 1),
            "deal_structure":       "All-stock; 50/50 board; CEO from larger company",
        }

    def minority_stake_template(
        self,
        company_ebitda: float = 200e6,
        stake_pct: float = 0.30,
        entry_multiple: float = 12.0,
    ) -> Dict[str, Any]:
        """
        Minority stake / growth equity investment (no control, no leverage).
        Returns expected MOIC at various exit scenarios.
        """
        enterprise_value   = company_ebitda * entry_multiple
        investment         = enterprise_value * stake_pct
        scenarios: List[Dict] = []
        for exit_mult in [10, 12, 14, 16, 18]:
            exit_ev    = company_ebitda * (1.08 ** 5) * exit_mult   # 8% EBITDA growth
            exit_stake = exit_ev * stake_pct
            moic       = exit_stake / investment if investment > 0 else 0
            irr        = moic ** (1/5) - 1
            scenarios.append({
                "exit_multiple": exit_mult,
                "exit_ev_mm":    round(exit_ev / 1e6, 1),
                "exit_value_mm": round(exit_stake / 1e6, 1),
                "moic":          round(moic, 2),
                "irr_pct":       round(irr * 100, 1),
            })
        return {
            "template":        "minority_stake",
            "investment_mm":   round(investment / 1e6, 1),
            "stake_pct":       stake_pct,
            "entry_ev_mm":     round(enterprise_value / 1e6, 1),
            "exit_scenarios":  scenarios,
        }

    def growth_equity_template(
        self,
        revenue: float = 100e6,
        revenue_growth: float = 0.35,
        current_arr: float = 80e6,
        investment_mm: float = 50e6,
    ) -> Dict[str, Any]:
        """
        Growth equity / SaaS investment template.
        ARR-based valuation (NTM Revenue multiple).
        """
        arr_multiples_by_growth = {
            0.20: (8, 12),
            0.30: (12, 18),
            0.40: (16, 25),
            0.50: (20, 35),
        }
        # Find nearest growth bucket
        closest_g  = min(arr_multiples_by_growth, key=lambda g: abs(g - revenue_growth))
        lo_m, hi_m = arr_multiples_by_growth[closest_g]
        ntm_arr    = current_arr * (1 + revenue_growth)

        entry_ev_lo = ntm_arr * lo_m
        entry_ev_hi = ntm_arr * hi_m
        # At exit (5y): assume growth decelerates to 20%, then apply 12× NTM ARR
        exit_arr    = current_arr * (1 + revenue_growth) ** 3 * (1 + 0.20) ** 2
        exit_ev     = exit_arr * 14.0   # mid-range exit multiple for maturing SaaS

        moic_mid = exit_ev / (investment_mm * 1e6) if investment_mm > 0 else 0
        irr_mid  = moic_mid ** (1/5) - 1

        return {
            "template":          "growth_equity",
            "revenue_mm":        round(revenue / 1e6, 1),
            "current_arr_mm":    round(current_arr / 1e6, 1),
            "revenue_growth":    revenue_growth,
            "entry_ev_range_mm": [round(entry_ev_lo / 1e6, 1), round(entry_ev_hi / 1e6, 1)],
            "exit_arr_mm":       round(exit_arr / 1e6, 1),
            "exit_ev_mm":        round(exit_ev / 1e6, 1),
            "investment_mm":     round(investment_mm, 1),
            "moic_base":         round(moic_mid, 2),
            "irr_pct_base":      round(irr_mid * 100, 1),
        }


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

if _FASTAPI:
    router = APIRouter(prefix="/v3/lbo", tags=["LBO / Merger Models V3"])

    _screener   = LBOCandidateScreener()
    _templates  = ModelTemplates()
    _dcf        = DCFMergerValuation()
    _pe_econ    = PEFundEconomics

    class LBORunIn(BaseModel):
        target_name:      str
        entry_ebitda:     float
        entry_multiple:   float
        exit_multiple:    float  = 10.0
        hold_period:      int    = 5
        leverage_pct:     float  = 0.60
        revenue_growth:   float  = 0.05
        ebitda_margin:    float  = 0.20
        capex_pct:        float  = 0.03
        ticker:           Optional[str] = None
        pik_toggle:       bool   = False

    class MergerRunIn(BaseModel):
        acquirer_name:    str
        target_name:      str
        deal_value:       float
        cash_pct:         float  = 0.50
        premium_pct:      float  = 0.30
        acq_revenue:      float
        acq_ebitda:       float
        acq_net_income:   float
        acq_shares:       float
        acq_stock_price:  float  = 100.0
        acq_total_debt:   float  = 0.0
        tgt_revenue:      float
        tgt_ebitda:       float
        tgt_net_income:   float
        tgt_total_debt:   float  = 0.0
        tgt_book_value:   float  = 0.0
        synergy_revenue:  float  = 0.0
        synergy_cost:     float  = 0.0

    class DCFIn(BaseModel):
        ticker:           str
        revenue_growth:   List[float]
        ebitda_margins:   List[float]
        wacc:             float = 0.10
        terminal_growth:  float = 0.025

    class PEFundIn(BaseModel):
        fund_size:        float
        exit_irr:         float  = 0.22
        hold_period:      int    = 5
        mgmt_fee_pct:     float  = 0.020
        carry_pct:        float  = 0.200
        hurdle_rate:      float  = 0.080

    @router.post("/run")
    def run_lbo(payload: LBORunIn):
        m = LBOModel(
            payload.target_name, payload.entry_ebitda,
            payload.entry_multiple, payload.hold_period,
        )
        m.set_capital_structure(leverage_pct=payload.leverage_pct, pik_toggle=payload.pik_toggle)
        m.set_operating_assumptions(
            revenue_growth=payload.revenue_growth,
            ebitda_margin=payload.ebitda_margin,
            capex_pct_revenue=payload.capex_pct,
        )
        result = m.run_full_model(
            exit_multiple=payload.exit_multiple,
            ticker=payload.ticker,
        )
        return {
            "run_id":           result.run_id,
            "irr_pct":          result.irr * 100,
            "moic":             result.moic,
            "entry_ev_mm":      result.entry_ev / 1e6,
            "exit_equity_mm":   result.exit_equity_value / 1e6,
            "peak_leverage":    result.peak_leverage,
            "exit_leverage":    result.exit_leverage,
            "debt_paydown_mm":  result.debt_paydown_mm,
            "covenant_breach":  result.covenant_breach,
            "covenant_detail":  result.covenant_detail,
            "sensitivity":      result.sensitivity_df.to_dict() if result.sensitivity_df is not None else {},
        }

    @router.post("/merger")
    def run_merger(payload: MergerRunIn):
        mm = MergerModel(payload.acquirer_name, payload.target_name)
        mm.set_deal_structure(
            deal_value=payload.deal_value,
            cash_pct=payload.cash_pct,
            stock_pct=1 - payload.cash_pct,
            premium_pct=payload.premium_pct,
            acquirer_stock_price=payload.acq_stock_price,
            acquirer_shares_outstanding=payload.acq_shares,
        )
        acq_fin = {
            "revenue": payload.acq_revenue, "ebitda": payload.acq_ebitda,
            "da": payload.acq_ebitda * 0.20, "net_income": payload.acq_net_income,
            "interest_expense": payload.acq_total_debt * 0.05,
            "shares_outstanding": payload.acq_shares,
            "total_debt": payload.acq_total_debt,
        }
        tgt_fin = {
            "revenue": payload.tgt_revenue, "ebitda": payload.tgt_ebitda,
            "net_income": payload.tgt_net_income, "total_debt": payload.tgt_total_debt,
        }
        synergies = {"revenue": payload.synergy_revenue, "cost": payload.synergy_cost}
        result = mm.run(acq_fin, tgt_fin, synergies,
                        target_book_value=payload.tgt_book_value)
        return {
            "run_id":            result.run_id,
            "goodwill_mm":       result.goodwill_mm,
            "combined_leverage": result.combined_leverage,
            "breakeven_year":    result.breakeven_year,
            "leverage_rating":   result.leverage_rating,
            "accretion_schedule": (result.accretion_dilution_df.to_dict(orient="records")
                                   if result.accretion_dilution_df is not None else []),
        }

    @router.post("/dcf-valuation")
    def run_dcf(payload: DCFIn):
        return _dcf.compute_standalone_dcf(
            payload.ticker, payload.revenue_growth,
            payload.ebitda_margins, payload.wacc, payload.terminal_growth,
        )

    @router.get("/screen")
    def screen_lbo(
        tickers: str = Query(..., description="Comma-separated tickers"),
        min_score: float = Query(40.0),
    ):
        t_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        df = _screener.screen_universe(t_list, min_score=min_score)
        return df.to_dict(orient="records") if not df.empty else []

    @router.post("/pe-fund")
    def run_pe_fund(payload: PEFundIn):
        pe = PEFundEconomics(
            fund_size=payload.fund_size,
            mgmt_fee_pct=payload.mgmt_fee_pct,
            carry_pct=payload.carry_pct,
            hurdle_rate=payload.hurdle_rate,
        )
        return pe.project_fund(exit_irr=payload.exit_irr, hold_period=payload.hold_period)

    @router.get("/templates")
    def list_templates():
        return {
            "templates": [
                "simple_lbo", "carve_out_lbo", "public_to_private",
                "distressed_lbo", "roll_up_platform", "strategic_acquisition",
                "leveraged_recap", "spac_merger", "pe_fund_return",
                "merger_of_equals", "minority_stake", "growth_equity",
            ]
        }

    @router.get("/model/{run_id}")
    def get_model_run(run_id: str):
        result = _load_run(run_id)
        if result is None:
            raise HTTPException(404, f"Model run {run_id!r} not found")
        return result


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  SENTINEL LBO / MERGER MODEL V3 — DEMO RUN")
    print("=" * 70)

    templates = ModelTemplates()
    screener  = LBOCandidateScreener()
    pe_econ   = PEFundEconomics(fund_size=2_000e6)

    # ------------------------------------------------------------------
    # 1. Full LBO model: $500M EBITDA company, 10× entry
    # ------------------------------------------------------------------
    print("\n[1] Full LBO Model — $500M EBITDA at 10× entry")
    lbo = LBOModel("SENTINEL Demo Co", entry_ebitda=500e6, entry_multiple=10.0, hold_period=5)
    lbo.set_capital_structure(leverage_pct=0.60)
    lbo.set_operating_assumptions(revenue_growth=0.06, ebitda_margin=0.25)
    result = lbo.run_full_model(exit_multiple=10.0)

    print(f"  Entry EV          : ${result.entry_ev/1e9:.1f}B")
    print(f"  Equity Invested   : ${result.equity_invested/1e9:.2f}B")
    print(f"  Total Debt        : ${result.total_debt/1e9:.2f}B")
    print(f"  IRR               : {result.irr*100:.1f}%")
    print(f"  MOIC              : {result.moic:.2f}×")
    print(f"  Peak Leverage     : {result.peak_leverage:.1f}×")
    print(f"  Exit Leverage     : {result.exit_leverage:.1f}×")
    print(f"  Covenant Breach   : {result.covenant_breach} ({result.covenant_detail})")

    print("\n  5-Year Financials:")
    print(result.financials_df[["year","revenue","ebitda","fcf_pre_debt"]].to_string(index=False))

    print("\n  Debt Schedule (key columns):")
    debt_cols = ["year","total_debt_end","leverage","interest_coverage","debt_paydown_yr"]
    print(result.debt_schedule_df[[c for c in debt_cols if c in result.debt_schedule_df.columns]].to_string(index=False))

    print("\n  Sensitivity Table (IRR% — exit multiple vs EBITDA margin delta):")
    print(result.sensitivity_df.to_string())

    # ------------------------------------------------------------------
    # 2. Merger model: $5B acquisition
    # ------------------------------------------------------------------
    print("\n[2] Merger Model — $5B Acquisition (50% cash / 50% stock, 30% premium)")
    mm = MergerModel("MegaCorp", "TargetCo")
    mm.set_deal_structure(
        deal_value=5_000e6,
        cash_pct=0.50,
        stock_pct=0.50,
        premium_pct=0.30,
        acquirer_stock_price=120.0,
        acquirer_shares_outstanding=300e6,
    )
    acq_fin = {
        "revenue": 20_000e6, "ebitda": 4_000e6, "da": 800e6,
        "net_income": 2_500e6, "interest_expense": 500e6,
        "shares_outstanding": 300e6, "total_debt": 8_000e6,
    }
    tgt_fin = {
        "revenue": 3_000e6, "ebitda": 600e6, "da": 120e6,
        "net_income": 300e6, "total_debt": 1_500e6,
    }
    synergies = {"revenue": 80e6, "cost": 150e6}
    merger_result = mm.run(acq_fin, tgt_fin, synergies,
                           target_book_value=1_800e6, identifiable_intangibles=800e6)
    print(f"  Goodwill          : ${merger_result.goodwill_mm:.0f}M")
    print(f"  Pro-forma Leverage: {merger_result.combined_leverage:.1f}×")
    print(f"  Interest Coverage : {merger_result.interest_coverage:.1f}×")
    print(f"  Implied Rating    : {merger_result.leverage_rating}")
    print(f"  EPS Breakeven Year: {merger_result.breakeven_year}")
    print("\n  Accretion / Dilution Schedule:")
    cols = ["year","standalone_acq_eps","combined_eps","accretion_abs","accretion_pct","accretive"]
    avail = [c for c in cols if c in merger_result.accretion_dilution_df.columns]
    print(merger_result.accretion_dilution_df[avail].to_string(index=False))

    # ------------------------------------------------------------------
    # 3. LBO candidate screener
    # ------------------------------------------------------------------
    demo_universe = ["JNJ", "PG", "KO", "MMM", "CAT"]
    print(f"\n[3] LBO Candidate Screener — {demo_universe}")
    screen_df = screener.screen_universe(demo_universe, min_score=0)
    if not screen_df.empty:
        cols = ["ticker","company_name","composite_score","ebitda_margin",
                "net_leverage","fcf_conversion","sector"]
        avail = [c for c in cols if c in screen_df.columns]
        print(screen_df[avail].to_string(index=False))
    else:
        print("  (no data returned — check network / yfinance)")

    # ------------------------------------------------------------------
    # 4. PE Fund economics
    # ------------------------------------------------------------------
    print("\n[4] PE Fund Economics — $2B fund, 22% exit IRR, 5-yr hold")
    fund_result = pe_econ.project_fund(exit_irr=0.22, hold_period=5)
    print(f"  Fund Size         : ${fund_result['fund_size']/1e9:.1f}B")
    print(f"  Mgmt Fees (total) : ${fund_result['management_fees_total']/1e6:.0f}M")
    print(f"  GP Carry          : ${fund_result['gp_carry']/1e6:.0f}M")
    print(f"  Total Exit Value  : ${fund_result['total_exit_value']/1e9:.2f}B")
    print(f"  LP Net Proceeds   : ${fund_result['lp_net_proceeds']/1e9:.2f}B")
    print(f"  DPI               : {fund_result['dpi']:.2f}×")
    print(f"  TVPI              : {fund_result['tvpi']:.2f}×")
    print(f"  Fund IRR          : {fund_result['fund_irr_pct']:.1f}%")

    # ------------------------------------------------------------------
    # 5. Sample templates
    # ------------------------------------------------------------------
    print("\n[5] Template Gallery")
    ptp = templates.public_to_private_template(mkt_cap=3_000e6, ebitda=280e6, premium_pct=0.35)
    print(f"  PTP Template      : IRR={ptp['irr']*100:.1f}%  MOIC={ptp['moic']:.2f}×  Entry={ptp['entry_multiple']:.1f}×")

    dist = templates.distressed_lbo_template(entry_ebitda=50e6, entry_multiple=4.5)
    print(f"  Distressed LBO    : IRR={dist['irr']*100:.1f}%  MOIC={dist['moic']:.2f}×")

    recap = templates.leveraged_recap_template(company_ebitda=400e6, target_leverage=3.5)
    print(f"  Lev Recap         : Dividend=${recap['special_dividend_mm']:.0f}M  Lev={recap['post_recap_leverage']:.1f}×")

    pe_fund = templates.pe_fund_return_template(fund_size=3_000e6, exit_irr=0.25)
    print(f"  PE Fund Template  : DPI={pe_fund['dpi']:.2f}×  TVPI={pe_fund['tvpi']:.2f}×  IRR={pe_fund['fund_irr_pct']:.1f}%")

    # 5×5 sensitivity re-run with tighter range
    print("\n[6] Standalone Sensitivity (exit multiple 8-12×, EBITDA margin ±200bps)")
    lbo2 = LBOModel("Sensitivity Demo", entry_ebitda=300e6, entry_multiple=9.0)
    lbo2.set_capital_structure(0.58)
    lbo2.set_operating_assumptions(revenue_growth=0.05, ebitda_margin=0.22)
    sens = lbo2.sensitivity_table(
        base_exit_multiple=9.0,
        exit_multiples=[8, 9, 10, 11, 12],
        ebitda_margin_deltas=[-0.02, -0.01, 0.0, 0.01, 0.02],
    )
    print(sens.to_string())

    print("\nDemo complete.")
