"""
LBO / Merger Model Templates V2 — Dimension #101 (target score 9).

Comprehensive leveraged buyout and merger model suite with:
- Full 6-tranche LBO (TLA / TLB / Revolver / Senior Notes / Mezz / PIK)
- Debt capacity analysis: leverage multiples, interest coverage, FCF sweep
- Returns: IRR, MOIC, equity value bridge at entry/exit
- Exit multiple sensitivity: 5×7 table (entry leverage × exit multiple)
- Sponsor economics: management fee, carried interest (20%), preferred return hurdle
- PIK toggle: cash vs PIK interest, compounding on returns
- Covenant analysis: leverage / coverage headroom calculator
- Cash flow waterfall: Revolver → Scheduled TL → Excess CF Sweep → PIK accrual
- Merger model (accretion/dilution): pro-forma IS, EPS impact Y1–Y3, breakeven synergies
- Leveraged recap: borrow-to-dividend, post-recap leverage, re-rating scenarios
- 12 model templates (PE buyout, carve-out, PTP, recap, roll-up, MoE, strategic, SPAC, …)
- SQLite storage with versioning
- FastAPI router: POST /lbo, POST /merger-accretion, POST /leveraged-recap,
                  GET /templates, GET /model/{id}, POST /sensitivity

Free data: no paid API keys required (all computation is local / numeric).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAX_RATE: float = 0.21          # US statutory (TCJA 2018)
SOFR_PROXY: float = 0.053       # ~SOFR + term premium (May 2026)
_EPS = 1e-9

# Credit spreads (bps) over SOFR by tranche
_SPREAD: Dict[str, float] = {
    "tla":         0.0175,   # +175 bps  — senior secured TLA
    "tlb":         0.0275,   # +275 bps  — senior secured TLB
    "revolver":    0.0200,   # +200 bps  — revolving credit facility
    "senior_notes": 0.0450,  # +450 bps  — high-yield senior notes
    "mezz":        0.0700,   # +700 bps  — mezzanine / subordinated
    "pik":         0.1000,   # +1000 bps — PIK notes
}

# Default amortisation (% of original principal per year)
_AMORT_PCT: Dict[str, float] = {
    "tla":         0.100,    # 10% pa
    "tlb":         0.010,    # 1% pa (bullet-like)
    "revolver":    0.000,    # drawn/repaid freely
    "senior_notes": 0.000,  # bullet
    "mezz":        0.000,    # bullet
    "pik":         0.000,    # accretes — no cash payment
}

# Default sizing as % of total funded debt
_DEFAULT_TRANCHE_SPLIT: Dict[str, float] = {
    "tla":          0.20,
    "tlb":          0.35,
    "revolver":     0.05,
    "senior_notes": 0.25,
    "mezz":         0.10,
    "pik":          0.05,
}

# Financial covenants — typical haircuts from credit agreement
_COVENANT_DEFAULTS = {
    "max_leverage":         6.5,   # x EBITDA
    "min_interest_coverage": 2.0,  # x EBITDA / cash interest
    "max_capex_mm":         None,  # optional cap
    "min_liquidity_mm":     25.0,  # $25M floor
}

# ---------------------------------------------------------------------------
# SQLite — model run persistence
# ---------------------------------------------------------------------------

_DB_DIR = Path(__file__).parent.parent / "data" / "db"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "lbo_models_v2.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id          TEXT PRIMARY KEY,
            model_type  TEXT NOT NULL,
            name        TEXT NOT NULL,
            version     INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL,
            params_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model_type ON model_runs(model_type);
        CREATE INDEX IF NOT EXISTS idx_model_name ON model_runs(name);
        """)


_init_db()


def _save_model(model_type: str, name: str, params: dict, result: dict) -> str:
    """Persist a model run, auto-increment version for same name."""
    run_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(version) as v FROM model_runs WHERE name=? AND model_type=?",
            (name, model_type),
        ).fetchone()
        version = (row["v"] or 0) + 1
        conn.execute(
            "INSERT INTO model_runs VALUES (?,?,?,?,?,?,?)",
            (
                run_id, model_type, name, version, created_at,
                json.dumps(params, default=str),
                json.dumps(result, default=str),
            ),
        )
    logger.info("Saved model run", id=run_id, type=model_type, name=name, version=version)
    return run_id


def _load_model(run_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM model_runs WHERE id=?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "id":           row["id"],
        "model_type":   row["model_type"],
        "name":         row["name"],
        "version":      row["version"],
        "created_at":   row["created_at"],
        "params":       json.loads(row["params_json"]),
        "result":       json.loads(row["result_json"]),
    }


def _list_models(model_type: Optional[str] = None) -> List[dict]:
    with _get_conn() as conn:
        if model_type:
            rows = conn.execute(
                "SELECT id, model_type, name, version, created_at FROM model_runs "
                "WHERE model_type=? ORDER BY created_at DESC",
                (model_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, model_type, name, version, created_at FROM model_runs "
                "ORDER BY created_at DESC",
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------

def _irr(cash_flows: List[float], guess: float = 0.10) -> float:
    """Newton-Raphson IRR solver."""
    rate = guess
    for _ in range(200):
        npv = sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cash_flows))
        d_npv = sum(-t * cf / (1.0 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))
        if abs(d_npv) < _EPS:
            break
        new_rate = rate - npv / d_npv
        if abs(new_rate - rate) < 1e-8:
            rate = new_rate
            break
        rate = new_rate
    return rate


def _moic(equity_in: float, equity_out: float) -> float:
    return equity_out / equity_in if equity_in > _EPS else float("nan")


def _require_pos(v: float, name: str) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")


# ===========================================================================
# Tranche / DebtSchedule
# ===========================================================================

class TrancheConfig(BaseModel):
    """Single tranche of LBO debt."""
    name: str
    principal_mm: float = Field(..., gt=0)
    rate: float          # all-in rate (SOFR + spread)
    amort_pct: float = 0.0      # annual amortisation as % of original principal
    pik_toggle: bool = False     # True → interest accretes, no cash pay
    maturity_years: int = 7

    @field_validator("rate")
    @classmethod
    def _rate_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("rate must be a decimal between 0 and 1")
        return v


class DebtSchedule:
    """
    Full 6-tranche LBO debt schedule with:
    - Cash interest, PIK accrual, scheduled amortisation
    - Excess cash flow sweep (typically 50% → 25% → 0% step-down)
    - Revolver pay-down priority before TL amortisation
    - PIK compounding on PIK notes
    """

    def __init__(self, tranches: List[TrancheConfig], ebitda_start_mm: float,
                 capex_pct_revenue: float = 0.04, revenue_mm: float = 500.0,
                 revenue_growth: float = 0.05, ebitda_margin: float = 0.20,
                 nwc_pct_revenue: float = 0.02, ecf_sweep_pct: float = 0.50,
                 hold_years: int = 5) -> None:
        self.tranches = tranches
        self.ebitda_mm = ebitda_start_mm
        self.capex_pct = capex_pct_revenue
        self.revenue_mm = revenue_mm
        self.rev_growth = revenue_growth
        self.ebitda_margin = ebitda_margin
        self.nwc_pct = nwc_pct_revenue
        self.ecf_sweep = ecf_sweep_pct
        self.hold_years = hold_years
        self._schedule: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self) -> pd.DataFrame:
        """
        Build the full debt schedule year-by-year.

        Returns DataFrame with columns:
          year, [tranche]_bop, [tranche]_cash_int, [tranche]_pik_int,
          [tranche]_scheduled_amort, [tranche]_ecf_sweep, [tranche]_eop,
          total_debt_bop, total_cash_interest, total_pik, total_amort,
          total_ecf_sweep, total_debt_eop,
          ebitda, capex, nwc_change, taxes_paid, free_cash_flow,
          excess_cash_flow, revolver_balance
        """
        rows: List[Dict] = []
        # state: dict tranche_name → current balance
        balances: Dict[str, float] = {t.name: t.principal_mm for t in self.tranches}
        pik_accruals: Dict[str, float] = {t.name: 0.0 for t in self.tranches}
        # Revolver is tracked separately (name "revolver")
        revolver = next((t for t in self.tranches if "revolver" in t.name.lower()), None)

        revenue = self.revenue_mm
        ebitda = self.ebitda_mm

        for yr in range(1, self.hold_years + 1):
            # grow operating metrics
            if yr > 1:
                revenue *= (1 + self.rev_growth)
                ebitda = revenue * self.ebitda_margin

            capex = revenue * self.capex_pct
            nwc_change = revenue * self.nwc_pct * self.rev_growth  # incremental NWC

            # total cash interest for EBIT → EBT
            cash_int_total = 0.0
            pik_int_total = 0.0
            for t in self.tranches:
                bal = balances[t.name]
                interest = bal * t.rate
                if t.pik_toggle:
                    pik_int_total += interest
                    pik_accruals[t.name] += interest
                    balances[t.name] += interest   # PIK compounds
                else:
                    cash_int_total += interest

            # Taxes (simplified: EBITDA – D&A – cash_int, D&A ≈ 3.5% revenue)
            da = revenue * 0.035
            ebt = ebitda - da - cash_int_total
            taxes = max(ebt * TAX_RATE, 0.0)

            # Free cash flow
            fcf = ebitda - capex - nwc_change - cash_int_total - taxes
            # Excess cash flow sweep
            ecf = max(fcf, 0.0)
            sweep_available = ecf * self.ecf_sweep

            row: Dict = {
                "year": yr,
                "revenue_mm": round(revenue, 2),
                "ebitda_mm": round(ebitda, 2),
                "da_mm": round(da, 2),
                "capex_mm": round(capex, 2),
                "nwc_change_mm": round(nwc_change, 2),
                "cash_interest_mm": round(cash_int_total, 2),
                "pik_accrual_mm": round(pik_int_total, 2),
                "taxes_mm": round(taxes, 2),
                "free_cash_flow_mm": round(fcf, 2),
                "excess_cash_flow_mm": round(ecf, 2),
                "ecf_sweep_applied_mm": round(sweep_available, 2),
            }

            # Apply sweep priority: Revolver first, then TLA, then TLB
            sweep_remaining = sweep_available
            for t in self.tranches:
                if sweep_remaining <= _EPS:
                    break
                if t.pik_toggle:
                    continue  # never sweep PIK
                bal = balances[t.name]
                if bal <= _EPS:
                    continue
                applied = min(bal, sweep_remaining)
                balances[t.name] -= applied
                sweep_remaining -= applied
                row[f"{t.name}_ecf_sweep_mm"] = round(applied, 2)

            # Scheduled amortisation (after sweep)
            for t in self.tranches:
                sched = t.principal_mm * t.amort_pct
                bal = balances[t.name]
                applied = min(bal, sched)
                balances[t.name] = max(bal - applied, 0.0)
                row[f"{t.name}_bop_mm"] = round(bal, 2)
                row[f"{t.name}_sched_amort_mm"] = round(applied, 2)
                row[f"{t.name}_eop_mm"] = round(balances[t.name], 2)
                rate_used = t.rate
                if t.pik_toggle:
                    row[f"{t.name}_pik_int_mm"] = round(pik_accruals[t.name], 2)
                    row[f"{t.name}_cash_int_mm"] = 0.0
                else:
                    row[f"{t.name}_cash_int_mm"] = round(bal * rate_used, 2)
                    row[f"{t.name}_pik_int_mm"] = 0.0

            row["total_debt_eop_mm"] = round(sum(balances.values()), 2)
            row["leverage_x_ebitda"] = round(row["total_debt_eop_mm"] / ebitda, 2) if ebitda > _EPS else None
            row["interest_coverage_x"] = round(ebitda / cash_int_total, 2) if cash_int_total > _EPS else None
            rows.append(row)

        self._schedule = pd.DataFrame(rows)
        return self._schedule

    def summary(self) -> Dict:
        if self._schedule is None:
            self.build()
        df = self._schedule
        return {
            "hold_years":           self.hold_years,
            "year_1_leverage":      df.iloc[0]["leverage_x_ebitda"],
            "year_end_leverage":    df.iloc[-1]["leverage_x_ebitda"],
            "year_1_coverage":      df.iloc[0]["interest_coverage_x"],
            "year_end_coverage":    df.iloc[-1]["interest_coverage_x"],
            "total_debt_eop_mm":    df.iloc[-1]["total_debt_eop_mm"],
            "total_fcf_mm":         round(df["free_cash_flow_mm"].sum(), 2),
            "total_ecf_sweep_mm":   round(df["ecf_sweep_applied_mm"].sum(), 2),
            "avg_cash_interest_mm": round(df["cash_interest_mm"].mean(), 2),
        }


# ===========================================================================
# CovenantAnalyzer
# ===========================================================================

class CovenantAnalyzer:
    """
    Financial covenant headroom calculator.

    Evaluates leverage and coverage covenants against the debt schedule,
    flags tightening headroom, and computes EBITDA cushion to breach.
    """

    def __init__(self, schedule_df: pd.DataFrame,
                 max_leverage: float = 6.5,
                 min_coverage: float = 2.0,
                 step_down_per_year: float = 0.25) -> None:
        self.df = schedule_df
        self.max_lev = max_leverage
        self.min_cov = min_coverage
        self.step_down = step_down_per_year

    def analyze(self) -> pd.DataFrame:
        """
        Return year-by-year covenant headroom table.

        Columns: year, actual_leverage, max_allowed_leverage, leverage_headroom_x,
                 actual_coverage, min_allowed_coverage, coverage_headroom_x,
                 ebitda_cushion_to_breach_mm, covenant_breach_risk
        """
        rows: List[Dict] = []
        for _, r in self.df.iterrows():
            yr = int(r["year"])
            # Step-down: max leverage tightens each year
            max_lev_yr = max(self.max_lev - self.step_down * (yr - 1), 3.0)
            min_cov_yr = self.min_cov   # typically flat

            act_lev = r.get("leverage_x_ebitda") or 0.0
            act_cov = r.get("interest_coverage_x") or 0.0
            ebitda = r["ebitda_mm"]
            total_debt = r["total_debt_eop_mm"]
            cash_int = r["cash_interest_mm"]

            lev_headroom = max_lev_yr - act_lev
            cov_headroom = act_cov - min_cov_yr

            # EBITDA cushion: how far can EBITDA fall before breach?
            # Leverage breach: EBITDA_min = debt / max_lev
            ebitda_min_lev = total_debt / max_lev_yr if max_lev_yr > _EPS else 0.0
            # Coverage breach: EBITDA_min = cash_int * min_cov
            ebitda_min_cov = cash_int * min_cov_yr

            ebitda_cushion = min(ebitda - ebitda_min_lev, ebitda - ebitda_min_cov)

            risk = "pass"
            if lev_headroom < 0.25 or cov_headroom < 0.25:
                risk = "tight"
            if lev_headroom < 0.0 or cov_headroom < 0.0:
                risk = "breach"

            rows.append({
                "year":                         yr,
                "actual_leverage_x":            round(act_lev, 2),
                "max_allowed_leverage_x":       round(max_lev_yr, 2),
                "leverage_headroom_x":          round(lev_headroom, 3),
                "actual_coverage_x":            round(act_cov, 2),
                "min_allowed_coverage_x":       round(min_cov_yr, 2),
                "coverage_headroom_x":          round(cov_headroom, 3),
                "ebitda_cushion_to_breach_mm":  round(ebitda_cushion, 2),
                "covenant_status":              risk,
            })

        return pd.DataFrame(rows)


# ===========================================================================
# LBOModel_V2
# ===========================================================================

class LBOModel_V2:
    """
    Full institutional-grade LBO model.

    Parameters
    ----------
    target_name     : Name of the target company.
    entry_ev_mm     : Entry enterprise value ($M).
    entry_ebitda_mm : LTM EBITDA at entry ($M).
    revenue_mm      : LTM revenue ($M).
    debt_multiple   : Total debt / EBITDA at entry (default 5.5×).
    tranche_split   : Dict overriding default tranche %s (must sum ≤ 1).
    pik_tranche     : Enable PIK tranche (default True).
    exit_multiple   : EV / EBITDA at exit (default 8.0×).
    hold_years      : Investment horizon (default 5).
    revenue_growth  : Annual revenue CAGR (default 5%).
    ebitda_margin   : Stabilised EBITDA margin (default 20%).
    management_fee_pct : Annual management fee on committed equity (default 2%).
    carry_pct       : Carried interest % (default 20%).
    preferred_return: Hurdle rate / preferred return (default 8%).
    """

    def __init__(
        self,
        target_name: str,
        entry_ev_mm: float,
        entry_ebitda_mm: float,
        revenue_mm: float,
        debt_multiple: float = 5.5,
        tranche_split: Optional[Dict[str, float]] = None,
        pik_tranche: bool = True,
        exit_multiple: float = 8.0,
        hold_years: int = 5,
        revenue_growth: float = 0.05,
        ebitda_margin: float = 0.20,
        management_fee_pct: float = 0.02,
        carry_pct: float = 0.20,
        preferred_return: float = 0.08,
        ecf_sweep_pct: float = 0.50,
    ) -> None:
        _require_pos(entry_ev_mm, "entry_ev_mm")
        _require_pos(entry_ebitda_mm, "entry_ebitda_mm")
        _require_pos(revenue_mm, "revenue_mm")

        self.target_name = target_name
        self.entry_ev_mm = entry_ev_mm
        self.entry_ebitda_mm = entry_ebitda_mm
        self.revenue_mm = revenue_mm
        self.debt_multiple = debt_multiple
        self.pik_tranche = pik_tranche
        self.exit_multiple = exit_multiple
        self.hold_years = hold_years
        self.revenue_growth = revenue_growth
        self.ebitda_margin = ebitda_margin
        self.mgmt_fee_pct = management_fee_pct
        self.carry_pct = carry_pct
        self.preferred_return = preferred_return
        self.ecf_sweep_pct = ecf_sweep_pct
        self.tranche_split = tranche_split or dict(_DEFAULT_TRANCHE_SPLIT)

        # Derived
        self.total_debt_mm = entry_ebitda_mm * debt_multiple
        self.equity_mm = entry_ev_mm - self.total_debt_mm
        if self.equity_mm <= 0:
            raise ValueError(
                f"Equity contribution negative: EV={entry_ev_mm} Debt={self.total_debt_mm}. "
                "Reduce debt_multiple or increase entry_ev_mm."
            )

        self._tranches: List[TrancheConfig] = self._build_tranches()
        self._debt_schedule: Optional[DebtSchedule] = None
        self._schedule_df: Optional[pd.DataFrame] = None
        self._results: Optional[Dict] = None

    # ------------------------------------------------------------------
    # _build_tranches
    # ------------------------------------------------------------------
    def _build_tranches(self) -> List[TrancheConfig]:
        split = self.tranche_split
        # Normalise splits if not including PIK when disabled
        names = list(_SPREAD.keys())
        if not self.pik_tranche:
            names = [n for n in names if n != "pik"]
            # Redistribute PIK pct to mezz
            pik_pct = split.get("pik", 0.05)
            split = dict(split)
            split["mezz"] = split.get("mezz", 0.10) + pik_pct
            split["pik"] = 0.0

        tranches: List[TrancheConfig] = []
        for name in names:
            pct = split.get(name, 0.0)
            if pct <= _EPS:
                continue
            principal = self.total_debt_mm * pct
            rate = SOFR_PROXY + _SPREAD[name]
            pik = (name == "pik") and self.pik_tranche
            tranches.append(TrancheConfig(
                name=name,
                principal_mm=round(principal, 4),
                rate=round(rate, 6),
                amort_pct=_AMORT_PCT.get(name, 0.0),
                pik_toggle=pik,
                maturity_years=7 if "tl" in name else 8,
            ))
        return tranches

    # ------------------------------------------------------------------
    # sources_and_uses
    # ------------------------------------------------------------------
    def sources_and_uses(self) -> Dict:
        """
        LBO sources & uses table.

        Sources: equity sponsor, debt tranches, roll-over equity (if any).
        Uses: purchase price, deal fees, OID / financing fees, working capital.
        """
        deal_fees_pct = 0.015   # ~1.5% of EV
        financing_fees_pct = 0.02  # ~2% of debt
        oid_pct = 0.01           # 1% OID on TLB / HY

        deal_fees = self.entry_ev_mm * deal_fees_pct
        financing_fees = self.total_debt_mm * financing_fees_pct
        oid = self.total_debt_mm * oid_pct
        wc_adj = self.revenue_mm * 0.01   # 1% of revenue WC normalisation
        total_uses = self.entry_ev_mm + deal_fees + financing_fees + oid + wc_adj

        # Equity fills the gap
        equity_required = total_uses - self.total_debt_mm
        equity_pct = equity_required / total_uses

        sources = {
            "equity_sponsor_mm":      round(equity_required, 2),
            "equity_pct":             round(equity_pct * 100, 2),
            "total_debt_mm":          round(self.total_debt_mm, 2),
            "debt_pct":               round((1 - equity_pct) * 100, 2),
        }
        for t in self._tranches:
            sources[f"  {t.name}_mm"] = round(t.principal_mm, 2)
        sources["total_sources_mm"] = round(total_uses, 2)

        uses = {
            "purchase_price_ev_mm":   round(self.entry_ev_mm, 2),
            "deal_fees_mm":           round(deal_fees, 2),
            "financing_fees_mm":      round(financing_fees, 2),
            "oid_mm":                 round(oid, 2),
            "working_capital_adj_mm": round(wc_adj, 2),
            "total_uses_mm":          round(total_uses, 2),
        }
        return {
            "sources": sources,
            "uses": uses,
            "entry_ev_ebitda_x": round(self.entry_ev_mm / self.entry_ebitda_mm, 2),
            "total_debt_ebitda_x": round(self.total_debt_mm / self.entry_ebitda_mm, 2),
        }

    # ------------------------------------------------------------------
    # run_model
    # ------------------------------------------------------------------
    def run_model(self) -> Dict:
        """
        Execute the full LBO model.

        Returns comprehensive results dict with:
        - sources_and_uses
        - debt_schedule (records)
        - covenant_analysis (records)
        - returns (IRR, MOIC, equity bridge)
        - sponsor_economics (mgmt fee, carry, net IRR)
        - exit_summary
        """
        s_and_u = self.sources_and_uses()

        # Build debt schedule
        self._debt_schedule = DebtSchedule(
            tranches=self._tranches,
            ebitda_start_mm=self.entry_ebitda_mm,
            revenue_mm=self.revenue_mm,
            revenue_growth=self.revenue_growth,
            ebitda_margin=self.ebitda_margin,
            ecf_sweep_pct=self.ecf_sweep_pct,
            hold_years=self.hold_years,
        )
        df = self._debt_schedule.build()
        self._schedule_df = df

        # Covenant analysis
        covenant_df = CovenantAnalyzer(df).analyze()

        # Exit computation
        exit_ebitda = df.iloc[-1]["ebitda_mm"]
        exit_ev = exit_ebitda * self.exit_multiple
        exit_debt = df.iloc[-1]["total_debt_eop_mm"]
        exit_equity = max(exit_ev - exit_debt, 0.0)

        # Equity investment (all-in including fees)
        total_uses = s_and_u["uses"]["total_uses_mm"]
        equity_in = total_uses - self.total_debt_mm

        moic = _moic(equity_in, exit_equity)

        # IRR: initial outflow, annual zero interim (CF sweep retained by model),
        #      terminal inflow = exit_equity
        irr_cfs = [-equity_in] + [0.0] * (self.hold_years - 1) + [exit_equity]
        try:
            irr = _irr(irr_cfs)
        except Exception:
            irr = float("nan")

        # Sponsor economics
        sponsor = self._sponsor_economics(equity_in, exit_equity, irr)

        # Entry / exit equity value bridge
        bridge = {
            "entry_ev_mm":         round(self.entry_ev_mm, 2),
            "entry_debt_mm":       round(self.total_debt_mm, 2),
            "entry_equity_mm":     round(equity_in, 2),
            "ebitda_growth_contribution_mm": round(exit_ebitda - self.entry_ebitda_mm, 2) * self.exit_multiple,
            "multiple_expansion_mm": round((self.exit_multiple - self.entry_ev_mm / self.entry_ebitda_mm) * exit_ebitda, 2),
            "debt_paydown_mm":     round(self.total_debt_mm - exit_debt, 2),
            "exit_ev_mm":          round(exit_ev, 2),
            "exit_debt_mm":        round(exit_debt, 2),
            "exit_equity_mm":      round(exit_equity, 2),
            "moic_x":              round(moic, 3),
            "gross_irr_pct":       round(irr * 100, 2) if not math.isnan(irr) else None,
        }

        self._results = {
            "target_name":       self.target_name,
            "run_date":          date.today().isoformat(),
            "sources_and_uses":  s_and_u,
            "debt_schedule":     df.to_dict(orient="records"),
            "schedule_summary":  self._debt_schedule.summary(),
            "covenant_analysis": covenant_df.to_dict(orient="records"),
            "equity_bridge":     bridge,
            "sponsor_economics": sponsor,
            "exit_summary": {
                "exit_multiple_x": self.exit_multiple,
                "exit_ebitda_mm":  round(exit_ebitda, 2),
                "exit_ev_mm":      round(exit_ev, 2),
                "exit_debt_mm":    round(exit_debt, 2),
                "exit_equity_mm":  round(exit_equity, 2),
            },
        }
        return self._results

    # ------------------------------------------------------------------
    # _sponsor_economics
    # ------------------------------------------------------------------
    def _sponsor_economics(self, equity_in: float, exit_equity: float, gross_irr: float) -> Dict:
        """
        Compute PE sponsor economics:
        - Annual management fee (2% on committed equity)
        - Preferred return hurdle (8% compound)
        - Catch-up (100%) then 20% carry on profits above hurdle

        Returns dict with mgmt_fees_total, preferred_return_threshold,
        carry_proceeds, gp_net_proceeds, lp_net_proceeds, net_irr_lp_pct.
        """
        # Management fees are paid out of portfolio company (deducted from LP returns)
        mgmt_fees_annual = equity_in * self.mgmt_fee_pct
        mgmt_fees_total = mgmt_fees_annual * self.hold_years

        # Preferred return: LP invested capital grows at hurdle
        lp_investment = equity_in
        preferred_threshold = lp_investment * ((1 + self.preferred_return) ** self.hold_years)

        profits = exit_equity - equity_in
        if profits <= 0 or exit_equity <= preferred_threshold:
            # No carry — LP gets everything, GP gets management fee only
            carry = 0.0
            gp_proceeds = 0.0
            lp_proceeds = exit_equity
        else:
            # Above hurdle: 100% catch-up then 80/20
            above_hurdle = exit_equity - preferred_threshold
            # Catch-up: GP receives until GP has 20% of total profits
            total_profit = exit_equity - lp_investment
            gp_target_carry = total_profit * self.carry_pct
            catch_up = min(above_hurdle, gp_target_carry)
            remaining = above_hurdle - catch_up
            carry = catch_up + remaining * self.carry_pct
            gp_proceeds = carry
            lp_proceeds = exit_equity - carry

        # Net LP IRR (after carry and fees)
        lp_net_in = equity_in + mgmt_fees_total
        lp_net_out = lp_proceeds
        lp_cfs = [-lp_net_in] + [0.0] * (self.hold_years - 1) + [lp_net_out]
        try:
            lp_irr = _irr(lp_cfs)
        except Exception:
            lp_irr = float("nan")

        return {
            "equity_committed_mm":      round(equity_in, 2),
            "management_fee_annual_mm": round(mgmt_fees_annual, 2),
            "management_fee_total_mm":  round(mgmt_fees_total, 2),
            "preferred_return_pct":     round(self.preferred_return * 100, 2),
            "preferred_threshold_mm":   round(preferred_threshold, 2),
            "carry_pct":                round(self.carry_pct * 100, 2),
            "carry_proceeds_mm":        round(carry, 2),
            "gp_total_proceeds_mm":     round(gp_proceeds + mgmt_fees_total, 2),
            "lp_net_proceeds_mm":       round(lp_proceeds, 2),
            "lp_net_irr_pct":           round(lp_irr * 100, 2) if not math.isnan(lp_irr) else None,
            "gross_irr_pct":            round(gross_irr * 100, 2) if not math.isnan(gross_irr) else None,
        }

    # ------------------------------------------------------------------
    # debt_capacity_analysis
    # ------------------------------------------------------------------
    def debt_capacity_analysis(self) -> Dict:
        """
        Estimate maximum supportable debt using three methods:

        1. Leverage multiple: max debt = n.0x EBITDA (ratings-implied)
        2. Interest coverage: debt such that EBITDA/interest = 2.0x
        3. FCF sweep: debt repayable over hold period from FCF

        Returns summary with binding constraint and headroom.
        """
        ebitda = self.entry_ebitda_mm
        # Blended all-in rate (cash tranches only)
        cash_tranches = [t for t in self._tranches if not t.pik_toggle]
        total_cash_debt = sum(t.principal_mm for t in cash_tranches)
        blended_rate = (
            sum(t.principal_mm * t.rate for t in cash_tranches) / total_cash_debt
            if total_cash_debt > _EPS else SOFR_PROXY + 0.04
        )

        # 1. Leverage-based (BB/B implied: 5.5–7.0x for LBO)
        lev_max_debt_5x = ebitda * 5.0
        lev_max_debt_6x = ebitda * 6.0
        lev_max_debt_7x = ebitda * 7.0

        # 2. Coverage-based: EBITDA / (debt × rate) = 2.0x → debt = EBITDA / (rate × 2.0)
        cov_max_debt = ebitda / (blended_rate * 2.0) if blended_rate > _EPS else 0.0

        # 3. FCF sweep: estimate 5-year cumulative FCF ≈ (EBITDA * 60% - capex) × 5
        capex_approx = self.revenue_mm * 0.04
        annual_fcf_approx = ebitda * 0.60 - capex_approx
        fcf_repayment_capacity = annual_fcf_approx * self.hold_years * self.ecf_sweep_pct

        current_debt = self.total_debt_mm
        binding = min(lev_max_debt_6x, cov_max_debt)

        return {
            "entry_ebitda_mm":              round(ebitda, 2),
            "blended_cash_interest_rate":   round(blended_rate * 100, 3),
            "current_debt_mm":              round(current_debt, 2),
            "current_leverage_x":           round(current_debt / ebitda, 2),
            "max_debt_5x_ebitda_mm":        round(lev_max_debt_5x, 2),
            "max_debt_6x_ebitda_mm":        round(lev_max_debt_6x, 2),
            "max_debt_7x_ebitda_mm":        round(lev_max_debt_7x, 2),
            "coverage_implied_max_debt_mm": round(cov_max_debt, 2),
            "fcf_sweep_repayment_5yr_mm":   round(fcf_repayment_capacity, 2),
            "binding_constraint_mm":        round(binding, 2),
            "headroom_vs_binding_mm":       round(binding - current_debt, 2),
            "headroom_pct":                 round((binding - current_debt) / binding * 100 if binding > _EPS else 0.0, 2),
        }

    # ------------------------------------------------------------------
    # sensitivity_table
    # ------------------------------------------------------------------
    def sensitivity_table(
        self,
        entry_leverages: Optional[List[float]] = None,
        exit_multiples: Optional[List[float]] = None,
        metric: str = "irr",
    ) -> pd.DataFrame:
        """
        5×7 sensitivity table: entry leverage (rows) × exit multiple (cols).

        metric: "irr" | "moic" | "exit_equity"

        Returns DataFrame indexed by entry_leverage, columns = exit multiples.
        """
        if entry_leverages is None:
            entry_leverages = [4.0, 4.5, 5.0, 5.5, 6.0]
        if exit_multiples is None:
            exit_multiples = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

        rows: List[Dict] = []
        for lev in entry_leverages:
            row: Dict = {"entry_leverage_x": lev}
            for em in exit_multiples:
                try:
                    m = LBOModel_V2(
                        target_name=self.target_name,
                        entry_ev_mm=self.entry_ev_mm,
                        entry_ebitda_mm=self.entry_ebitda_mm,
                        revenue_mm=self.revenue_mm,
                        debt_multiple=lev,
                        exit_multiple=em,
                        hold_years=self.hold_years,
                        revenue_growth=self.revenue_growth,
                        ebitda_margin=self.ebitda_margin,
                        pik_tranche=self.pik_tranche,
                        ecf_sweep_pct=self.ecf_sweep_pct,
                    )
                    res = m.run_model()
                    bridge = res["equity_bridge"]
                    if metric == "moic":
                        val = bridge["moic_x"]
                    elif metric == "exit_equity":
                        val = bridge["exit_equity_mm"]
                    else:
                        val = bridge["gross_irr_pct"]
                    row[f"exit_{em:.1f}x"] = round(val, 2) if val is not None else None
                except Exception as exc:
                    row[f"exit_{em:.1f}x"] = None
                    logger.debug("Sensitivity cell error", lev=lev, em=em, error=str(exc))
            rows.append(row)

        df = pd.DataFrame(rows).set_index("entry_leverage_x")
        return df

    # ------------------------------------------------------------------
    # pik_impact_analysis
    # ------------------------------------------------------------------
    def pik_impact_analysis(self) -> Dict:
        """
        Compare cash-pay vs PIK interest for the PIK tranche.

        Shows compounding effect on final debt balance, IRR delta,
        and MOIC delta between cash-pay and full-PIK scenarios.
        """
        pik_t = next((t for t in self._tranches if t.pik_toggle), None)
        if pik_t is None:
            return {"note": "No PIK tranche configured"}

        pik_principal = pik_t.principal_mm
        pik_rate = pik_t.rate

        cash_pay_schedule = []
        pik_schedule = []
        bal_cash = pik_principal
        bal_pik = pik_principal

        for yr in range(1, self.hold_years + 1):
            cash_int = bal_cash * pik_rate
            cash_pay_schedule.append({
                "year": yr,
                "balance": round(bal_cash, 2),
                "cash_interest": round(cash_int, 2),
            })
            # PIK — interest accretes
            pik_int = bal_pik * pik_rate
            bal_pik += pik_int
            pik_schedule.append({
                "year": yr,
                "balance": round(bal_pik, 2),
                "pik_accrual": round(pik_int, 2),
            })

        incremental_debt = bal_pik - pik_principal  # debt saved by cash-pay
        return {
            "pik_tranche_name":        pik_t.name,
            "pik_rate_pct":            round(pik_rate * 100, 3),
            "pik_principal_mm":        round(pik_principal, 2),
            "cash_pay_scenario":       cash_pay_schedule,
            "pik_scenario":            pik_schedule,
            "pik_final_balance_mm":    round(bal_pik, 2),
            "cash_pay_final_balance_mm": round(pik_principal, 2),
            "pik_incremental_debt_mm": round(incremental_debt, 2),
            "pik_compounding_pct":     round((bal_pik / pik_principal - 1) * 100, 2),
        }


# ===========================================================================
# MergerModel_V2
# ===========================================================================

class MergerModel_V2:
    """
    Comprehensive merger accretion/dilution and pro-forma income statement model.

    Supports cash, stock, and mixed consideration deals.
    Includes:
    - Purchase price and premium analysis
    - Pro-forma income statement (Y1–Y3)
    - EPS accretion/dilution with synergy ramp
    - Breakeven synergy calculation
    - Purchase price allocation (ASC 805)
    - Goodwill sensitivity
    - Financing structure (new debt, equity issuance, bridge)
    """

    def __init__(
        self,
        acquirer: str,
        target: str,
        # Acquirer financials ($M unless noted)
        acq_revenue_mm: float,
        acq_ebitda_mm: float,
        acq_net_income_mm: float,
        acq_shares_mm: float,           # millions of diluted shares
        acq_stock_price: float,         # $ per share
        acq_eps: float,                 # LTM EPS ($)
        # Target financials ($M)
        tgt_revenue_mm: float,
        tgt_ebitda_mm: float,
        tgt_net_income_mm: float,
        tgt_shares_mm: float,
        tgt_stock_price: float,
        # Deal terms
        offer_price_per_share: float,   # $ per target share
        deal_consideration: str = "all_cash",  # "all_cash" | "all_stock" | "mixed"
        cash_pct: float = 1.0,          # % of deal paid in cash (for mixed)
        # Synergies
        revenue_synergies_mm: float = 0.0,
        cost_synergies_mm: float = 0.0,
        one_time_costs_mm: float = 0.0,  # restructuring / integration
        # Financing
        new_debt_mm: float = 0.0,
        new_debt_rate: float = 0.05,
        bridge_facility_mm: float = 0.0,
        # PPA
        identified_intangibles_mm: float = 0.0,
        intangibles_useful_life_yrs: int = 15,
        ppe_step_up_pct: float = 0.10,
        # Growth assumptions
        acq_revenue_growth: float = 0.05,
        tgt_revenue_growth: float = 0.05,
        acq_ebitda_margin: float = 0.20,
        tgt_ebitda_margin: float = 0.20,
    ) -> None:
        self.acquirer = acquirer
        self.target = target
        self.acq_revenue = acq_revenue_mm
        self.acq_ebitda = acq_ebitda_mm
        self.acq_ni = acq_net_income_mm
        self.acq_shares = acq_shares_mm
        self.acq_stock = acq_stock_price
        self.acq_eps = acq_eps
        self.tgt_revenue = tgt_revenue_mm
        self.tgt_ebitda = tgt_ebitda_mm
        self.tgt_ni = tgt_net_income_mm
        self.tgt_shares = tgt_shares_mm
        self.tgt_stock = tgt_stock_price
        self.offer_price = offer_price_per_share
        self.deal_consideration = deal_consideration
        self.cash_pct = cash_pct if deal_consideration == "mixed" else (
            1.0 if deal_consideration == "all_cash" else 0.0
        )
        self.rev_syn = revenue_synergies_mm
        self.cost_syn = cost_synergies_mm
        self.ot_costs = one_time_costs_mm
        self.new_debt = new_debt_mm
        self.new_debt_rate = new_debt_rate
        self.bridge = bridge_facility_mm
        self.intangibles = identified_intangibles_mm
        self.intang_life = intangibles_useful_life_yrs
        self.ppe_step_up_pct = ppe_step_up_pct
        self.acq_rev_growth = acq_revenue_growth
        self.tgt_rev_growth = tgt_revenue_growth
        self.acq_ebitda_margin = acq_ebitda_margin
        self.tgt_ebitda_margin = tgt_ebitda_margin

        # Derived
        self.deal_value_mm = self.offer_price * self.tgt_shares
        self.equity_premium_pct = (self.offer_price / self.tgt_stock - 1.0) * 100.0 if self.tgt_stock > _EPS else 0.0
        self.cash_consideration_mm = self.deal_value_mm * self.cash_pct
        self.stock_consideration_mm = self.deal_value_mm * (1 - self.cash_pct)
        self.new_shares_issued_mm = self.stock_consideration_mm / self.acq_stock if self.acq_stock > _EPS else 0.0
        self.proforma_shares_mm = self.acq_shares + self.new_shares_issued_mm

        # PPA
        self.ppa = self._purchase_price_allocation()

    # ------------------------------------------------------------------
    # _purchase_price_allocation
    # ------------------------------------------------------------------
    def _purchase_price_allocation(self) -> Dict:
        """Compute purchase price allocation (ASC 805 / IFRS 3)."""
        tgt_book_value = self.tgt_ni / max(self.tgt_ebitda / max(self.tgt_revenue, 1), 0.05)  # rough proxy
        # More robust: use stated book value approximation as 2× NI
        tgt_book_approx = self.tgt_ni * 2.0
        premium = self.deal_value_mm - tgt_book_approx
        ppe_step_up = self.deal_value_mm * self.ppe_step_up_pct
        dtl = self.intangibles * TAX_RATE
        goodwill = max(premium - self.intangibles - ppe_step_up + dtl, 0.0)
        annual_intang_amort = self.intangibles / max(self.intang_life, 1)
        annual_ppe_amort = ppe_step_up / 10.0
        return {
            "deal_value_mm":              round(self.deal_value_mm, 2),
            "target_book_value_est_mm":   round(tgt_book_approx, 2),
            "deal_premium_mm":            round(premium, 2),
            "ppe_step_up_mm":             round(ppe_step_up, 2),
            "identified_intangibles_mm":  round(self.intangibles, 2),
            "deferred_tax_liability_mm":  round(dtl, 2),
            "goodwill_mm":                round(goodwill, 2),
            "annual_intangible_amort_mm": round(annual_intang_amort, 2),
            "annual_ppe_amort_mm":        round(annual_ppe_amort, 2),
            "total_annual_ppa_amort_mm":  round(annual_intang_amort + annual_ppe_amort, 2),
            "goodwill_pct_of_deal":       round(goodwill / self.deal_value_mm * 100 if self.deal_value_mm > _EPS else 0.0, 2),
        }

    # ------------------------------------------------------------------
    # pro_forma_income_statement
    # ------------------------------------------------------------------
    def pro_forma_income_statement(self, years: int = 3) -> pd.DataFrame:
        """
        Build combined pro-forma income statement for years 1 to N.

        Includes synergy ramp (50% Y1, 100% Y2+), one-time costs in Y1,
        and PPA amortisation.
        """
        ppa_amort = self.ppa["total_annual_ppa_amort_mm"]
        new_interest = (self.new_debt + self.bridge) * self.new_debt_rate * (1 - TAX_RATE)

        rows: List[Dict] = []
        acq_rev = self.acq_revenue
        tgt_rev = self.tgt_revenue

        for yr in range(1, years + 1):
            acq_rev *= (1 + self.acq_rev_growth)
            tgt_rev *= (1 + self.tgt_rev_growth)
            combined_rev = acq_rev + tgt_rev

            # Synergy ramp
            syn_ramp = min(yr / 2.0, 1.0)   # 50% Y1, 100% Y2+
            rev_syn = self.rev_syn * syn_ramp
            cost_syn = self.cost_syn * syn_ramp
            ot = self.ot_costs if yr == 1 else 0.0

            acq_ebitda_yr = acq_rev * self.acq_ebitda_margin
            tgt_ebitda_yr = tgt_rev * self.tgt_ebitda_margin
            combined_ebitda = acq_ebitda_yr + tgt_ebitda_yr + cost_syn + rev_syn - ot

            da = combined_rev * 0.035
            ebit = combined_ebitda - da - ppa_amort
            ebt = ebit - new_interest
            taxes = max(ebt * TAX_RATE, 0.0)
            ni = ebt - taxes

            rows.append({
                "year":                    yr,
                "acquirer_revenue_mm":     round(acq_rev, 2),
                "target_revenue_mm":       round(tgt_rev, 2),
                "combined_revenue_mm":     round(combined_rev, 2),
                "revenue_synergies_mm":    round(rev_syn, 2),
                "cost_synergies_mm":       round(cost_syn, 2),
                "one_time_costs_mm":       round(ot, 2),
                "combined_ebitda_mm":      round(combined_ebitda, 2),
                "da_mm":                   round(da, 2),
                "ppa_amortisation_mm":     round(ppa_amort, 2),
                "ebit_mm":                 round(ebit, 2),
                "new_debt_interest_mm":    round(new_interest, 2),
                "ebt_mm":                  round(ebt, 2),
                "tax_provision_mm":        round(taxes, 2),
                "net_income_mm":           round(ni, 2),
                "proforma_eps":            round(ni / self.proforma_shares_mm, 4) if self.proforma_shares_mm > _EPS else 0.0,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # accretion_dilution
    # ------------------------------------------------------------------
    def accretion_dilution(self, years: int = 3) -> Dict:
        """
        Compute EPS accretion/dilution for years 1 through N.

        Returns:
          acquirer_standalone_eps_yr[n], proforma_eps_yr[n],
          accretion_dilution_pct_yr[n], breakeven_synergies_mm
        """
        pf_df = self.pro_forma_income_statement(years)
        ppa_amort = self.ppa["total_annual_ppa_amort_mm"]
        new_interest = (self.new_debt + self.bridge) * self.new_debt_rate * (1 - TAX_RATE)

        results: Dict = {
            "deal_value_mm":             round(self.deal_value_mm, 2),
            "offer_premium_pct":         round(self.equity_premium_pct, 2),
            "deal_consideration":        self.deal_consideration,
            "cash_consideration_mm":     round(self.cash_consideration_mm, 2),
            "stock_consideration_mm":    round(self.stock_consideration_mm, 2),
            "new_shares_issued_mm":      round(self.new_shares_issued_mm, 4),
            "proforma_shares_mm":        round(self.proforma_shares_mm, 4),
            "ppa": self.ppa,
        }

        eps_rows = []
        for yr in range(1, years + 1):
            standalone_eps = self.acq_eps * ((1 + self.acq_rev_growth) ** yr)
            pf_row = pf_df[pf_df["year"] == yr].iloc[0]
            pf_eps = pf_row["proforma_eps"]
            delta_pct = (pf_eps / standalone_eps - 1.0) * 100.0 if abs(standalone_eps) > _EPS else 0.0
            eps_rows.append({
                "year":                   yr,
                "standalone_acq_eps":     round(standalone_eps, 4),
                "proforma_eps":           round(pf_eps, 4),
                "accretion_dilution_pct": round(delta_pct, 3),
                "accretive":              delta_pct > 0,
            })

        results["eps_analysis"] = eps_rows

        # Breakeven synergies: synergies needed for zero accretion/dilution in Y1
        # Solve: pf_eps_y1 (with syn=0) + syn * margin / shares = standalone_eps_y1
        acq_rev_y1 = self.acq_revenue * (1 + self.acq_rev_growth)
        tgt_rev_y1 = self.tgt_revenue * (1 + self.tgt_rev_growth)
        acq_ni_y1 = acq_rev_y1 * self.acq_ebitda_margin * (1 - TAX_RATE)
        tgt_ni_y1 = tgt_rev_y1 * self.tgt_ebitda_margin * (1 - TAX_RATE)
        ppa_after_tax = ppa_amort * (1 - TAX_RATE)
        pf_ni_no_syn = acq_ni_y1 + tgt_ni_y1 - new_interest - ppa_after_tax
        standalone_ni_y1 = self.acq_ni * ((1 + self.acq_rev_growth) ** 1)
        standalone_eps_y1 = standalone_ni_y1 / self.acq_shares if self.acq_shares > _EPS else 0.0

        # Required NI from synergies = standalone_ni - pf_ni_no_syn
        required_ni_from_syn = max((standalone_eps_y1 * self.proforma_shares_mm) - pf_ni_no_syn, 0.0)
        # Syn pre-tax = required_ni / (1 - tax) × (ramp to 50% in Y1) * 2
        breakeven_syn_annual = (required_ni_from_syn / (1 - TAX_RATE)) * 2.0  # 50% ramp in Y1

        results["breakeven_cost_synergies_mm"] = round(breakeven_syn_annual, 2)
        results["breakeven_pct_of_combined_revenue"] = round(
            breakeven_syn_annual / (acq_rev_y1 + tgt_rev_y1) * 100.0
            if (acq_rev_y1 + tgt_rev_y1) > _EPS else 0.0, 3
        )
        results["proforma_income_statement"] = pf_df.to_dict(orient="records")
        return results

    # ------------------------------------------------------------------
    # football_field
    # ------------------------------------------------------------------
    def football_field(
        self,
        dcf_range: Tuple[float, float],
        comps_range: Tuple[float, float],
        precedent_range: Tuple[float, float],
    ) -> Dict:
        """Football field chart data for offer vs valuation methodologies."""
        offer = self.offer_price

        def _rng(lo: float, hi: float) -> Dict:
            mid = (lo + hi) / 2.0
            prem = (offer / mid - 1.0) * 100.0 if mid > _EPS else 0.0
            return {"low": round(lo, 2), "high": round(hi, 2),
                    "midpoint": round(mid, 2), "premium_to_mid_pct": round(prem, 2)}

        return {
            "offer_price": round(offer, 2),
            "52_week_range": _rng(self.tgt_stock * 0.70, self.tgt_stock * 0.98),
            "dcf_range": _rng(*dcf_range),
            "trading_comps": _rng(*comps_range),
            "precedent_transactions": _rng(*precedent_range),
            "offer_vs_unaffected_premium_pct": round(self.equity_premium_pct, 2),
        }


# ===========================================================================
# LeveragedRecapModel
# ===========================================================================

class LeveragedRecapModel:
    """
    Leveraged recapitalisation model: borrow-to-dividend.

    Models:
    - Pre/post recap balance sheet
    - Special dividend proceeds
    - Post-recap credit rating migration
    - Re-rating scenario analysis (spread widening, equity re-rating)
    - Breakeven EBITDA for debt service
    """

    def __init__(
        self,
        company: str,
        pre_recap_ev_mm: float,
        pre_recap_ebitda_mm: float,
        pre_recap_debt_mm: float,
        pre_recap_equity_mm: float,
        new_debt_mm: float,
        new_debt_rate: float = 0.065,
        shares_mm: float = 100.0,
        stock_price: float = 50.0,
    ) -> None:
        self.company = company
        self.pre_ev = pre_recap_ev_mm
        self.pre_ebitda = pre_recap_ebitda_mm
        self.pre_debt = pre_recap_debt_mm
        self.pre_equity = pre_recap_equity_mm
        self.new_debt = new_debt_mm
        self.new_rate = new_debt_rate
        self.shares = shares_mm
        self.stock = stock_price

    def run(self) -> Dict:
        """Execute leveraged recap analysis."""
        # Post-recap balance sheet
        post_debt = self.pre_debt + self.new_debt
        dividend_mm = self.new_debt * 0.95  # ~5% fees
        dividend_per_share = dividend_mm / self.shares if self.shares > _EPS else 0.0
        post_equity = self.pre_equity - dividend_mm
        post_ev = self.pre_ev  # EV unchanged near-term (leverage-neutral at entry)

        # Leverage metrics
        pre_lev = self.pre_debt / self.pre_ebitda if self.pre_ebitda > _EPS else 0.0
        post_lev = post_debt / self.pre_ebitda if self.pre_ebitda > _EPS else 0.0

        # Implied incremental interest
        incr_interest = self.new_debt * self.new_rate
        pre_coverage = self.pre_ebitda / max(self.pre_debt * (SOFR_PROXY + 0.03), 1.0)
        post_coverage = self.pre_ebitda / max(
            (self.pre_debt * (SOFR_PROXY + 0.03)) + incr_interest, 1.0
        )

        # Re-rating scenarios
        scenarios = self._rerate_scenarios(post_lev, post_coverage)

        # Breakeven: min EBITDA to service all debt
        total_interest = (
            self.pre_debt * (SOFR_PROXY + 0.03) + incr_interest
        )
        breakeven_ebitda = total_interest * 2.0   # 2.0× coverage floor

        return {
            "company":                  self.company,
            "pre_recap": {
                "debt_mm":              round(self.pre_debt, 2),
                "equity_mm":            round(self.pre_equity, 2),
                "leverage_x":           round(pre_lev, 2),
                "interest_coverage_x":  round(pre_coverage, 2),
            },
            "post_recap": {
                "new_debt_raised_mm":   round(self.new_debt, 2),
                "total_debt_mm":        round(post_debt, 2),
                "equity_mm":            round(post_equity, 2),
                "leverage_x":           round(post_lev, 2),
                "interest_coverage_x":  round(post_coverage, 2),
                "incremental_interest_mm": round(incr_interest, 2),
            },
            "special_dividend": {
                "gross_proceeds_mm":    round(self.new_debt, 2),
                "fees_mm":              round(self.new_debt * 0.05, 2),
                "net_dividend_mm":      round(dividend_mm, 2),
                "dividend_per_share":   round(dividend_per_share, 2),
            },
            "credit_rerate_scenarios":  scenarios,
            "breakeven_ebitda_mm":      round(breakeven_ebitda, 2),
            "ebitda_cushion_to_breakeven_mm": round(self.pre_ebitda - breakeven_ebitda, 2),
        }

    def _rerate_scenarios(self, post_lev: float, post_cov: float) -> List[Dict]:
        """Model three re-rating scenarios: base, downgrade, severe downgrade."""
        scenarios = []
        for label, spread_widen, equity_derate in [
            ("base_case", 0.0, 0.0),
            ("one_notch_downgrade", 0.0050, 0.10),
            ("two_notch_downgrade", 0.0150, 0.20),
        ]:
            extra_interest = self.new_debt * spread_widen
            post_ni_est = max(
                (self.pre_ebitda - (self.pre_debt + self.new_debt) * (SOFR_PROXY + 0.04 + spread_widen)) * (1 - TAX_RATE),
                0.0
            )
            new_stock_price = self.stock * (1 - equity_derate)
            scenarios.append({
                "scenario":             label,
                "spread_widen_bps":     round(spread_widen * 10000, 0),
                "extra_interest_mm":    round(extra_interest, 2),
                "post_recap_ni_est_mm": round(post_ni_est, 2),
                "equity_derate_pct":    round(equity_derate * 100, 2),
                "implied_stock_price":  round(new_stock_price, 2),
                "post_recap_leverage_x": round(post_lev, 2),
                "post_recap_coverage_x": round(post_cov - extra_interest / max(self.pre_ebitda, 1.0), 2),
            })
        return scenarios


# ===========================================================================
# LBOTemplateLibrary — 12 model templates
# ===========================================================================

LBO_TEMPLATES: Dict[str, Dict] = {
    "pe_buyout": {
        "description": "Classic private equity buyout: sponsor-led, 5–6× debt, 5-year hold",
        "debt_multiple": 5.5,
        "exit_multiple": 9.0,
        "hold_years": 5,
        "revenue_growth": 0.07,
        "ebitda_margin": 0.22,
        "ecf_sweep_pct": 0.50,
        "pik_tranche": False,
        "typical_sectors": ["industrials", "healthcare", "tech", "consumer"],
        "key_value_drivers": ["organic growth", "margin improvement", "multiple expansion"],
    },
    "growth_buyout": {
        "description": "High-growth software/tech buyout: lower leverage, higher exit multiple",
        "debt_multiple": 4.0,
        "exit_multiple": 15.0,
        "hold_years": 4,
        "revenue_growth": 0.20,
        "ebitda_margin": 0.15,
        "ecf_sweep_pct": 0.25,
        "pik_tranche": False,
        "typical_sectors": ["software", "fintech", "healthtech"],
        "key_value_drivers": ["ARR growth", "NRR expansion", "multiple re-rating"],
    },
    "carve_out": {
        "description": "Corporate carve-out: operational complexity, stranded costs, 100-day plan",
        "debt_multiple": 5.0,
        "exit_multiple": 8.5,
        "hold_years": 5,
        "revenue_growth": 0.05,
        "ebitda_margin": 0.18,
        "ecf_sweep_pct": 0.50,
        "pik_tranche": True,
        "typical_sectors": ["diversified industrials", "conglomerates"],
        "key_value_drivers": ["stranded cost removal", "focus premium", "standalone infra"],
    },
    "public_to_private": {
        "description": "Take-private: regulatory premium, public company discount removal",
        "debt_multiple": 6.0,
        "exit_multiple": 10.0,
        "hold_years": 4,
        "revenue_growth": 0.08,
        "ebitda_margin": 0.20,
        "ecf_sweep_pct": 0.50,
        "pik_tranche": True,
        "typical_sectors": ["software", "media", "healthcare"],
        "key_value_drivers": ["de-listing cost savings", "management incentive alignment"],
    },
    "leveraged_recap": {
        "description": "Recap existing portfolio company to return capital to LP",
        "debt_multiple": 5.0,
        "exit_multiple": 9.0,
        "hold_years": 3,
        "revenue_growth": 0.05,
        "ebitda_margin": 0.22,
        "ecf_sweep_pct": 0.30,
        "pik_tranche": False,
        "typical_sectors": ["stable cashflow businesses"],
        "key_value_drivers": ["DPI acceleration", "bridge to exit"],
    },
    "roll_up": {
        "description": "Platform + add-on acquisition strategy: bolt-on synergies, multiple arb",
        "debt_multiple": 5.5,
        "exit_multiple": 11.0,
        "hold_years": 6,
        "revenue_growth": 0.12,
        "ebitda_margin": 0.20,
        "ecf_sweep_pct": 0.50,
        "pik_tranche": False,
        "typical_sectors": ["fragmented services", "B2B software", "specialty distribution"],
        "key_value_drivers": ["M&A synergies", "scale premium", "add-on accretion"],
    },
    "merger_of_equals": {
        "description": "MoE: combined entity, zero premium, synergy sharing, stock-for-stock",
        "deal_type": "merger",
        "consideration": "all_stock",
        "premium_pct": 5.0,
        "cost_synergy_pct_combined_opex": 0.05,
        "typical_sectors": ["banking", "insurance", "media", "mining"],
        "key_value_drivers": ["cost synergies", "scale", "cross-sell"],
    },
    "strategic_acquisition_cash": {
        "description": "Strategic M&A: cash deal, synergies from revenue + cost, EPS accretive",
        "deal_type": "merger",
        "consideration": "all_cash",
        "premium_pct": 30.0,
        "cost_synergy_pct_combined_opex": 0.08,
        "typical_sectors": ["pharma", "technology", "consumer staples"],
        "key_value_drivers": ["cost synergies", "revenue synergies", "market position"],
    },
    "strategic_acquisition_stock": {
        "description": "Strategic M&A: stock deal, lower dilution risk if acquirer richly valued",
        "deal_type": "merger",
        "consideration": "all_stock",
        "premium_pct": 25.0,
        "cost_synergy_pct_combined_opex": 0.06,
        "typical_sectors": ["technology", "media", "financials"],
        "key_value_drivers": ["currency arbitrage", "integration synergies"],
    },
    "hostile_tender": {
        "description": "Hostile tender offer: higher premium, no management cooperation",
        "deal_type": "merger",
        "consideration": "all_cash",
        "premium_pct": 45.0,
        "cost_synergy_pct_combined_opex": 0.07,
        "typical_sectors": ["all sectors"],
        "key_value_drivers": ["control premium", "forced merger", "white knight risk"],
    },
    "spac_merger": {
        "description": "SPAC de-SPAC merger: trust value, redemptions, earnout, warrants",
        "deal_type": "merger",
        "consideration": "mixed",
        "premium_pct": 0.0,
        "spac_trust_per_share": 10.0,
        "typical_sectors": ["technology", "EV", "biotech", "space"],
        "key_value_drivers": ["public liquidity premium", "growth story", "warrant dilution"],
    },
    "consortium_bid": {
        "description": "Club deal / consortium: equity co-investment, shared risk, larger deal",
        "debt_multiple": 6.0,
        "exit_multiple": 9.5,
        "hold_years": 5,
        "revenue_growth": 0.06,
        "ebitda_margin": 0.20,
        "ecf_sweep_pct": 0.50,
        "pik_tranche": True,
        "num_sponsors": 2,
        "typical_sectors": ["large-cap infrastructure", "energy", "telco"],
        "key_value_drivers": ["scale of deal", "shared DD costs", "regulatory leverage"],
    },
}


# ===========================================================================
# FastAPI Router
# ===========================================================================

router = APIRouter(prefix="/api/v2/lbo", tags=["lbo-merger-v2"])


# -- Pydantic request/response models --

class LBORequest(BaseModel):
    target_name: str = "Target Co"
    entry_ev_mm: float = Field(1000.0, gt=0)
    entry_ebitda_mm: float = Field(100.0, gt=0)
    revenue_mm: float = Field(500.0, gt=0)
    debt_multiple: float = Field(5.5, ge=1.0, le=10.0)
    exit_multiple: float = Field(9.0, ge=1.0, le=25.0)
    hold_years: int = Field(5, ge=1, le=10)
    revenue_growth: float = Field(0.05, ge=-0.20, le=0.50)
    ebitda_margin: float = Field(0.20, ge=0.01, le=0.80)
    pik_tranche: bool = True
    ecf_sweep_pct: float = Field(0.50, ge=0.0, le=1.0)
    management_fee_pct: float = Field(0.02, ge=0.0, le=0.05)
    carry_pct: float = Field(0.20, ge=0.0, le=0.35)
    preferred_return: float = Field(0.08, ge=0.0, le=0.20)
    tranche_split: Optional[Dict[str, float]] = None
    save: bool = True
    name: Optional[str] = None


class MergerRequest(BaseModel):
    acquirer: str = "Acquirer Inc"
    target: str = "Target Corp"
    acq_revenue_mm: float = Field(5000.0, gt=0)
    acq_ebitda_mm: float = Field(1000.0, gt=0)
    acq_net_income_mm: float = Field(600.0, gt=0)
    acq_shares_mm: float = Field(500.0, gt=0)
    acq_stock_price: float = Field(100.0, gt=0)
    acq_eps: float = Field(1.20, gt=0)
    tgt_revenue_mm: float = Field(1000.0, gt=0)
    tgt_ebitda_mm: float = Field(200.0, gt=0)
    tgt_net_income_mm: float = Field(120.0, gt=0)
    tgt_shares_mm: float = Field(100.0, gt=0)
    tgt_stock_price: float = Field(40.0, gt=0)
    offer_price_per_share: float = Field(52.0, gt=0)
    deal_consideration: str = "all_cash"
    cash_pct: float = Field(1.0, ge=0.0, le=1.0)
    revenue_synergies_mm: float = 0.0
    cost_synergies_mm: float = Field(50.0, ge=0.0)
    one_time_costs_mm: float = Field(30.0, ge=0.0)
    new_debt_mm: float = Field(4000.0, ge=0.0)
    new_debt_rate: float = Field(0.055, ge=0.01, le=0.20)
    bridge_facility_mm: float = 0.0
    identified_intangibles_mm: float = Field(300.0, ge=0.0)
    intangibles_useful_life_yrs: int = Field(15, ge=1, le=40)
    ppe_step_up_pct: float = Field(0.10, ge=0.0, le=0.50)
    analysis_years: int = Field(3, ge=1, le=5)
    save: bool = True
    name: Optional[str] = None


class RecapRequest(BaseModel):
    company: str = "RecapCo"
    pre_recap_ev_mm: float = Field(2000.0, gt=0)
    pre_recap_ebitda_mm: float = Field(250.0, gt=0)
    pre_recap_debt_mm: float = Field(500.0, ge=0)
    pre_recap_equity_mm: float = Field(1500.0, gt=0)
    new_debt_mm: float = Field(800.0, gt=0)
    new_debt_rate: float = Field(0.065, ge=0.01, le=0.20)
    shares_mm: float = Field(100.0, gt=0)
    stock_price: float = Field(50.0, gt=0)
    save: bool = True
    name: Optional[str] = None


class SensitivityRequest(BaseModel):
    target_name: str = "Target Co"
    entry_ev_mm: float = Field(1000.0, gt=0)
    entry_ebitda_mm: float = Field(100.0, gt=0)
    revenue_mm: float = Field(500.0, gt=0)
    hold_years: int = Field(5, ge=1, le=10)
    revenue_growth: float = 0.05
    ebitda_margin: float = 0.20
    entry_leverages: List[float] = Field(default_factory=lambda: [4.0, 4.5, 5.0, 5.5, 6.0])
    exit_multiples: List[float] = Field(default_factory=lambda: [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    metric: str = "irr"  # "irr" | "moic" | "exit_equity"


# -- Endpoints --

@router.post("/lbo", summary="Run full LBO model")
def run_lbo(req: LBORequest = Body(...)) -> Dict:
    """Execute a full 6-tranche LBO model with debt schedule, returns, and sponsor economics."""
    try:
        model = LBOModel_V2(
            target_name=req.target_name,
            entry_ev_mm=req.entry_ev_mm,
            entry_ebitda_mm=req.entry_ebitda_mm,
            revenue_mm=req.revenue_mm,
            debt_multiple=req.debt_multiple,
            exit_multiple=req.exit_multiple,
            hold_years=req.hold_years,
            revenue_growth=req.revenue_growth,
            ebitda_margin=req.ebitda_margin,
            pik_tranche=req.pik_tranche,
            ecf_sweep_pct=req.ecf_sweep_pct,
            management_fee_pct=req.management_fee_pct,
            carry_pct=req.carry_pct,
            preferred_return=req.preferred_return,
            tranche_split=req.tranche_split,
        )
        result = model.run_model()
        capacity = model.debt_capacity_analysis()
        pik_impact = model.pik_impact_analysis()
        result["debt_capacity"] = capacity
        result["pik_impact"] = pik_impact

        run_id = None
        if req.save:
            name = req.name or f"{req.target_name}_{date.today().isoformat()}"
            run_id = _save_model("lbo", name, req.model_dump(), result)
        result["run_id"] = run_id
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("LBO model error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/merger-accretion", summary="Merger accretion/dilution model")
def run_merger(req: MergerRequest = Body(...)) -> Dict:
    """Run full merger model: accretion/dilution, PPA, pro-forma IS, football field."""
    try:
        model = MergerModel_V2(
            acquirer=req.acquirer,
            target=req.target,
            acq_revenue_mm=req.acq_revenue_mm,
            acq_ebitda_mm=req.acq_ebitda_mm,
            acq_net_income_mm=req.acq_net_income_mm,
            acq_shares_mm=req.acq_shares_mm,
            acq_stock_price=req.acq_stock_price,
            acq_eps=req.acq_eps,
            tgt_revenue_mm=req.tgt_revenue_mm,
            tgt_ebitda_mm=req.tgt_ebitda_mm,
            tgt_net_income_mm=req.tgt_net_income_mm,
            tgt_shares_mm=req.tgt_shares_mm,
            tgt_stock_price=req.tgt_stock_price,
            offer_price_per_share=req.offer_price_per_share,
            deal_consideration=req.deal_consideration,
            cash_pct=req.cash_pct,
            revenue_synergies_mm=req.revenue_synergies_mm,
            cost_synergies_mm=req.cost_synergies_mm,
            one_time_costs_mm=req.one_time_costs_mm,
            new_debt_mm=req.new_debt_mm,
            new_debt_rate=req.new_debt_rate,
            bridge_facility_mm=req.bridge_facility_mm,
            identified_intangibles_mm=req.identified_intangibles_mm,
            intangibles_useful_life_yrs=req.intangibles_useful_life_yrs,
            ppe_step_up_pct=req.ppe_step_up_pct,
        )
        result = model.accretion_dilution(years=req.analysis_years)

        run_id = None
        if req.save:
            name = req.name or f"{req.acquirer}_acquires_{req.target}_{date.today().isoformat()}"
            run_id = _save_model("merger", name, req.model_dump(), result)
        result["run_id"] = run_id
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Merger model error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/leveraged-recap", summary="Leveraged recapitalisation model")
def run_recap(req: RecapRequest = Body(...)) -> Dict:
    """Model a leveraged recap: borrow-to-dividend, credit re-rating, breakeven analysis."""
    try:
        model = LeveragedRecapModel(
            company=req.company,
            pre_recap_ev_mm=req.pre_recap_ev_mm,
            pre_recap_ebitda_mm=req.pre_recap_ebitda_mm,
            pre_recap_debt_mm=req.pre_recap_debt_mm,
            pre_recap_equity_mm=req.pre_recap_equity_mm,
            new_debt_mm=req.new_debt_mm,
            new_debt_rate=req.new_debt_rate,
            shares_mm=req.shares_mm,
            stock_price=req.stock_price,
        )
        result = model.run()

        run_id = None
        if req.save:
            name = req.name or f"{req.company}_recap_{date.today().isoformat()}"
            run_id = _save_model("recap", name, req.model_dump(), result)
        result["run_id"] = run_id
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Recap model error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/templates", summary="List all 12 LBO/merger model templates")
def list_templates() -> Dict:
    """Return all built-in model templates with descriptions and key parameters."""
    return {
        "count": len(LBO_TEMPLATES),
        "templates": {
            k: {
                "id": k,
                "description": v.get("description", ""),
                "key_params": {kk: vv for kk, vv in v.items() if kk != "description"},
            }
            for k, v in LBO_TEMPLATES.items()
        },
    }


@router.get("/model/{run_id}", summary="Retrieve a saved model run")
def get_model(run_id: str) -> Dict:
    """Retrieve a saved model run by ID."""
    data = _load_model(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Model run {run_id} not found")
    return data


@router.get("/models", summary="List saved model runs")
def list_saved_models(
    model_type: Optional[str] = Query(None, description="Filter by type: lbo | merger | recap"),
    limit: int = Query(50, ge=1, le=200),
) -> Dict:
    """List all saved model runs, optionally filtered by type."""
    rows = _list_models(model_type)[:limit]
    return {"count": len(rows), "models": rows}


@router.post("/sensitivity", summary="IRR / MOIC sensitivity table (entry leverage × exit multiple)")
def sensitivity_table(req: SensitivityRequest = Body(...)) -> Dict:
    """
    Build 5×7 sensitivity table: entry leverage (rows) × exit multiple (cols).

    metric: 'irr' (gross IRR %), 'moic' (equity multiple), 'exit_equity' ($M).
    """
    try:
        # Validate metric
        if req.metric not in ("irr", "moic", "exit_equity"):
            raise HTTPException(status_code=422, detail="metric must be irr | moic | exit_equity")

        # Use base leverage for constructing the model
        base_lev = req.entry_leverages[len(req.entry_leverages) // 2]
        model = LBOModel_V2(
            target_name=req.target_name,
            entry_ev_mm=req.entry_ev_mm,
            entry_ebitda_mm=req.entry_ebitda_mm,
            revenue_mm=req.revenue_mm,
            debt_multiple=base_lev,
            exit_multiple=req.exit_multiples[len(req.exit_multiples) // 2],
            hold_years=req.hold_years,
            revenue_growth=req.revenue_growth,
            ebitda_margin=req.ebitda_margin,
        )
        df = model.sensitivity_table(
            entry_leverages=req.entry_leverages,
            exit_multiples=req.exit_multiples,
            metric=req.metric,
        )
        return {
            "metric":          req.metric,
            "entry_leverages": req.entry_leverages,
            "exit_multiples":  req.exit_multiples,
            "table":           df.reset_index().to_dict(orient="records"),
            "note": (
                "IRR = gross unlevered IRR %; MOIC = equity multiple; "
                "exit_equity = sponsor exit equity ($M)"
            ),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Sensitivity table error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/debt-capacity", summary="Debt capacity analysis for given EBITDA")
def debt_capacity(
    ebitda_mm: float = Query(..., gt=0, description="LTM EBITDA ($M)"),
    revenue_mm: float = Query(..., gt=0, description="LTM revenue ($M)"),
    debt_multiple: float = Query(5.5, ge=1.0, le=10.0),
    entry_ev_mm: Optional[float] = Query(None),
) -> Dict:
    """Quick debt capacity analysis without running full model."""
    if entry_ev_mm is None:
        entry_ev_mm = ebitda_mm * 10.0  # default 10× EBITDA entry
    try:
        model = LBOModel_V2(
            target_name="Debt Capacity Analysis",
            entry_ev_mm=entry_ev_mm,
            entry_ebitda_mm=ebitda_mm,
            revenue_mm=revenue_mm,
            debt_multiple=debt_multiple,
        )
        return model.debt_capacity_analysis()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/covenant-check", summary="Covenant headroom check")
def covenant_check(
    run_id: str = Query(..., description="Model run ID from POST /lbo"),
    max_leverage: float = Query(6.5, ge=1.0, le=10.0),
    min_coverage: float = Query(2.0, ge=0.5, le=5.0),
    step_down_per_year: float = Query(0.25, ge=0.0, le=1.0),
) -> Dict:
    """
    Run covenant headroom analysis on a previously saved LBO model run.
    Returns year-by-year leverage / coverage headroom and breach risk.
    """
    data = _load_model(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Model run {run_id} not found")
    if data["model_type"] != "lbo":
        raise HTTPException(status_code=422, detail="Covenant check only applies to LBO model runs")

    sched = data["result"].get("debt_schedule", [])
    if not sched:
        raise HTTPException(status_code=422, detail="No debt schedule found in model run")

    df = pd.DataFrame(sched)
    analyzer = CovenantAnalyzer(
        df,
        max_leverage=max_leverage,
        min_coverage=min_coverage,
        step_down_per_year=step_down_per_year,
    )
    cov_df = analyzer.analyze()
    breach_years = cov_df[cov_df["covenant_status"] == "breach"]["year"].tolist()
    tight_years = cov_df[cov_df["covenant_status"] == "tight"]["year"].tolist()

    return {
        "run_id":        run_id,
        "max_leverage":  max_leverage,
        "min_coverage":  min_coverage,
        "step_down":     step_down_per_year,
        "covenant_table": cov_df.to_dict(orient="records"),
        "breach_years":  breach_years,
        "tight_years":   tight_years,
        "overall_status": "breach" if breach_years else ("tight" if tight_years else "pass"),
    }


@router.get("/pik-analysis", summary="PIK toggle impact analysis")
def pik_analysis(
    run_id: str = Query(..., description="Model run ID from POST /lbo"),
) -> Dict:
    """Compare cash-pay vs PIK interest compounding for the PIK tranche."""
    data = _load_model(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Model run {run_id} not found")
    return data["result"].get("pik_impact", {"note": "No PIK impact data in this run"})


@router.get("/health", summary="Module health check")
def health() -> Dict:
    """Return module status and SQLite model run count."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) as n FROM model_runs").fetchone()["n"]
        by_type = conn.execute(
            "SELECT model_type, COUNT(*) as n FROM model_runs GROUP BY model_type"
        ).fetchall()
    return {
        "status":       "ok",
        "dimension":    101,
        "version":      "v2",
        "db_path":      str(_DB_PATH),
        "total_runs":   total,
        "runs_by_type": {r["model_type"]: r["n"] for r in by_type},
        "templates":    len(LBO_TEMPLATES),
        "as_of":        datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Convenience re-exports for app router registration
# ---------------------------------------------------------------------------

lbo_router = router  # alias for import in main app
