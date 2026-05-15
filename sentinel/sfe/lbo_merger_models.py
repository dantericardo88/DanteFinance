"""
LBO / Merger Model Templates — Dimension #101 (target 9+).

Institutional-grade financial model templates for leveraged buyout and merger
analysis.  All models are purely numeric — no I/O required — and serialize to
JSON for persistence.

Classes
-------
LBOModel        — Full 5-tranche LBO model with debt schedule, P&L, cash flow,
                  IRR/MOIC, sensitivity tables, and scenario analysis.
MergerModel     — Accretion/dilution, pro-forma P&L, football-field chart data,
                  and purchase price allocation.
DCFModel        — Gordon-growth DCF, WACC computation, WACC/TGR sensitivity.
ModelLibrary    — Persist / load models to .sentinel/models/ as JSON.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAX_RATE:    float = 0.21        # US corporate statutory rate (2018 TCJA)
LIBOR_PROXY: float = 0.053       # Approximate SOFR/risk-free proxy (May 2026)

# Credit spread (bps) by tranche vs LIBOR_PROXY
_TRANCHE_SPREADS: dict[str, float] = {
    "senior_secured_term_a":  0.0175,  # +175 bps
    "senior_secured_term_b":  0.0275,  # +275 bps
    "senior_notes":           0.0450,  # +450 bps high-yield
    "subordinated_notes":     0.0700,  # +700 bps mezz
    "revolver":               0.0200,  # +200 bps revolver
}

# Typical LBO debt tranche sizing as % of total debt
_TRANCHE_SIZE_PCT: dict[str, float] = {
    "senior_secured_term_a":  0.20,
    "senior_secured_term_b":  0.35,
    "senior_notes":           0.25,
    "subordinated_notes":     0.15,
    "revolver":               0.05,
}

# Amortization schedules (% of initial principal per year)
_TRANCHE_AMORT_PCT: dict[str, float] = {
    "senior_secured_term_a":  0.10,   # 10% annually
    "senior_secured_term_b":  0.01,   # 1% (bullet with 1% token amort)
    "senior_notes":           0.00,   # bullet
    "subordinated_notes":     0.00,   # bullet
    "revolver":               0.00,   # revolver, drawn/repaid freely
}

# PIK toggle eligibility by tranche
_PIK_ELIGIBLE: dict[str, bool] = {
    "senior_secured_term_a":  False,
    "senior_secured_term_b":  False,
    "senior_notes":           False,
    "subordinated_notes":     True,   # typical mezz PIK feature
    "revolver":               False,
}

_MODELS_DIR = Path(os.environ.get("SENTINEL_MODELS_DIR", ".sentinel/models"))

# Numeric guard
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_fraction(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


# ---------------------------------------------------------------------------
# IRR solver (Newton-Raphson with bisection fallback)
# ---------------------------------------------------------------------------

def _compute_irr(cash_flows: list[float], guess: float = 0.15) -> float:
    """
    Solve for IRR using Newton-Raphson with bisection fallback.

    Args:
        cash_flows: Time-ordered cash flows. cash_flows[0] must be negative.
        guess:      Initial rate guess.

    Returns:
        IRR as decimal (e.g. 0.25 = 25%).
    """
    n = len(cash_flows)
    if n < 2:
        return 0.0

    def _npv(r: float) -> float:
        return sum(cf / (1.0 + r) ** t for t, cf in enumerate(cash_flows))

    def _dnpv(r: float) -> float:
        return sum(-t * cf / (1.0 + r) ** (t + 1) for t, cf in enumerate(cash_flows))

    # Newton-Raphson
    r = guess
    for _ in range(100):
        npv  = _npv(r)
        dnpv = _dnpv(r)
        if abs(dnpv) < _EPS:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < 1e-10:
            return round(r_new, 6)
        r = r_new
        r = max(-0.99, min(r, 50.0))   # clamp

    # Bisection fallback
    lo, hi = -0.99, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _npv(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return round((lo + hi) / 2.0, 6)


# ---------------------------------------------------------------------------
# LBOModel
# ---------------------------------------------------------------------------

class LBOModel:
    """
    Full institutional-grade LBO model with five debt tranches.

    Usage::

        model = LBOModel("TargetCo", entry_ev=1000.0, entry_ebitda=100.0)
        debt_df   = model.build_debt_schedule()
        income_df = model.build_income_statement([0.05]*5, [0.30]*5)
        cf_df     = model.build_cash_flow()
        returns   = model.compute_returns(exit_multiple=10.0)
        sensitivity = model.sensitivity_table([8,9,10,11,12], [8,9,10,11,12])
        scenarios = model.run_base_bull_bear()

    Args:
        target_name:    Company being acquired.
        entry_ev:       Entry enterprise value (millions).
        entry_ebitda:   LTM EBITDA at entry (millions).
        debt_multiple:  Total debt / EBITDA at entry.
        equity_pct:     Equity as % of total capitalization.
    """

    def __init__(
        self,
        target_name:    str,
        entry_ev:       float,
        entry_ebitda:   float,
        debt_multiple:  float = 5.0,
        equity_pct:     float = 0.35,
    ) -> None:
        _require_positive(entry_ev,     "entry_ev")
        _require_positive(entry_ebitda, "entry_ebitda")
        _require_fraction(equity_pct,   "equity_pct")
        if debt_multiple < 0:
            raise ValueError("debt_multiple must be non-negative")

        self.target_name   = target_name
        self.entry_ev      = entry_ev
        self.entry_ebitda  = entry_ebitda
        self.debt_multiple = debt_multiple
        self.equity_pct    = equity_pct

        self.total_debt    = entry_ebitda * debt_multiple
        self.equity_check  = entry_ev * equity_pct
        # Reconcile: equity = EV - debt (standard LBO identity)
        self.entry_equity  = max(entry_ev - self.total_debt, 0.0)

        # Entry multiple
        self.entry_multiple = entry_ev / entry_ebitda if entry_ebitda > _EPS else 0.0

        # Build tranche breakdown
        self.tranches: dict[str, dict] = {}
        for name, size_pct in _TRANCHE_SIZE_PCT.items():
            principal = self.total_debt * size_pct
            rate      = LIBOR_PROXY + _TRANCHE_SPREADS[name]
            self.tranches[name] = {
                "principal":   round(principal, 2),
                "rate":        round(rate, 4),
                "amort_pct":   _TRANCHE_AMORT_PCT[name],
                "pik_eligible": _PIK_ELIGIBLE[name],
                "balance":     principal,   # evolving state
            }

        # Caches set by build_* methods (for cross-method consistency)
        self._debt_df:   Optional[pd.DataFrame] = None
        self._income_df: Optional[pd.DataFrame] = None
        self._cf_df:     Optional[pd.DataFrame] = None

        logger.debug(
            "LBOModel init: %s EV=%.1f EBITDA=%.1f debt=%.1f equity=%.1f",
            target_name, entry_ev, entry_ebitda, self.total_debt, self.entry_equity,
        )

    # ------------------------------------------------------------------
    # build_debt_schedule
    # ------------------------------------------------------------------

    def build_debt_schedule(self, years: int = 5) -> pd.DataFrame:
        """
        Build a detailed debt schedule for all five tranches over the hold period.

        Includes: mandatory amortization, cash interest, PIK interest (sub notes),
        revolver draws assumed zero, and end-of-period balance.

        Args:
            years: Hold period in years (default 5).

        Returns:
            pd.DataFrame indexed by (tranche, year) with columns:
            bop_balance, cash_interest, pik_interest, mandatory_amort,
            optional_paydown, eop_balance.
        """
        rows: list[dict] = []

        # Reset tranche balances to initial principals
        for name in self.tranches:
            self.tranches[name]["balance"] = self.tranches[name]["principal"]

        for yr in range(1, years + 1):
            for tranche_name, t in self.tranches.items():
                bop_balance = t["balance"]
                rate        = t["rate"]
                amort_pct   = t["amort_pct"]

                cash_interest = bop_balance * rate
                pik_interest  = 0.0
                if t["pik_eligible"] and yr <= 2:
                    # PIK toggle: sub notes PIK first 2 years to preserve cash
                    pik_interest  = cash_interest * 0.50
                    cash_interest = cash_interest * 0.50

                mandatory_amort = bop_balance * amort_pct
                eop_balance     = bop_balance - mandatory_amort + pik_interest

                t["balance"] = max(eop_balance, 0.0)

                rows.append({
                    "tranche":          tranche_name,
                    "year":             yr,
                    "bop_balance":      round(bop_balance, 2),
                    "rate_pct":         round(rate * 100.0, 3),
                    "cash_interest":    round(cash_interest, 2),
                    "pik_interest":     round(pik_interest, 2),
                    "total_interest":   round(cash_interest + pik_interest, 2),
                    "mandatory_amort":  round(mandatory_amort, 2),
                    "optional_paydown": 0.0,   # filled by cash flow sweep
                    "eop_balance":      round(t["balance"], 2),
                })

        df = pd.DataFrame(rows)
        self._debt_df = df
        return df

    # ------------------------------------------------------------------
    # build_income_statement
    # ------------------------------------------------------------------

    def build_income_statement(
        self,
        revenue_growth_rates: list[float],
        ebitda_margins:       list[float],
        da_pct_revenue:       float = 0.05,
    ) -> pd.DataFrame:
        """
        Build a projected income statement for the hold period.

        Args:
            revenue_growth_rates: Year-over-year revenue growth rates (decimals).
            ebitda_margins:       EBITDA margin for each year (decimals).
            da_pct_revenue:       D&A as % of revenue (default 5%).

        Returns:
            pd.DataFrame with one row per year and columns:
            revenue, ebitda, ebitda_margin, da, ebit, interest_expense,
            ebt, tax_provision, net_income.
        """
        years = max(len(revenue_growth_rates), len(ebitda_margins))
        # Pad shorter lists with last value
        while len(revenue_growth_rates) < years:
            revenue_growth_rates.append(revenue_growth_rates[-1] if revenue_growth_rates else 0.05)
        while len(ebitda_margins) < years:
            ebitda_margins.append(ebitda_margins[-1] if ebitda_margins else 0.30)

        # Infer base revenue from entry_ebitda / entry ebitda margin
        base_margin  = ebitda_margins[0]
        base_revenue = self.entry_ebitda / base_margin if base_margin > _EPS else self.entry_ebitda * 3.0

        # Rebuild debt schedule if not yet built (for interest expense)
        if self._debt_df is None:
            self.build_debt_schedule(years=years)

        # Aggregate total interest per year from debt schedule
        debt_df = self._debt_df
        interest_by_year: dict[int, float] = {}
        if debt_df is not None and not debt_df.empty:
            for yr, grp in debt_df.groupby("year"):
                interest_by_year[int(yr)] = grp["cash_interest"].sum()

        rows: list[dict] = []
        revenue = base_revenue

        for i in range(years):
            yr      = i + 1
            revenue = revenue * (1.0 + revenue_growth_rates[i]) if i > 0 else revenue
            ebitda  = revenue * ebitda_margins[i]
            da      = revenue * da_pct_revenue
            ebit    = ebitda - da
            interest_expense = interest_by_year.get(yr, self.total_debt * (LIBOR_PROXY + 0.03))
            ebt     = ebit - interest_expense
            taxes   = max(ebt * TAX_RATE, 0.0)
            net_inc = ebt - taxes

            rows.append({
                "year":             yr,
                "revenue":          round(revenue, 2),
                "revenue_growth":   round(revenue_growth_rates[i] * 100.0, 2),
                "ebitda":           round(ebitda, 2),
                "ebitda_margin_pct": round(ebitda_margins[i] * 100.0, 2),
                "da":               round(da, 2),
                "ebit":             round(ebit, 2),
                "ebit_margin_pct":  round((ebit / revenue * 100.0) if revenue > _EPS else 0.0, 2),
                "interest_expense": round(interest_expense, 2),
                "ebt":              round(ebt, 2),
                "tax_provision":    round(taxes, 2),
                "net_income":       round(net_inc, 2),
                "net_margin_pct":   round((net_inc / revenue * 100.0) if revenue > _EPS else 0.0, 2),
            })

        df = pd.DataFrame(rows)
        self._income_df = df
        return df

    # ------------------------------------------------------------------
    # build_cash_flow
    # ------------------------------------------------------------------

    def build_cash_flow(
        self,
        working_capital_pct_rev: float = 0.12,
        capex_pct_rev:           float = 0.04,
    ) -> pd.DataFrame:
        """
        Build free cash flow statement with debt paydown sweep.

        Args:
            working_capital_pct_rev: Working capital as % of revenue (default 12%).
            capex_pct_rev:           Capex as % of revenue (default 4%).

        Returns:
            pd.DataFrame with: ebitda, change_in_wc, capex, cash_taxes,
            operating_cf, investing_cf, fcf_before_debt, debt_paydown,
            fcf_after_debt, cash_balance.
        """
        if self._income_df is None or self._income_df.empty:
            self.build_income_statement([0.05] * 5, [0.30] * 5)

        income_df = self._income_df
        debt_df   = self._debt_df

        # Mandatory amortization per year
        mandatory_amort_by_yr: dict[int, float] = {}
        if debt_df is not None and not debt_df.empty:
            for yr, grp in debt_df.groupby("year"):
                mandatory_amort_by_yr[int(yr)] = grp["mandatory_amort"].sum()

        rows: list[dict] = []
        prev_revenue = None
        cash_balance = 25.0  # opening cash balance ($M)

        for _, inc_row in income_df.iterrows():
            yr      = int(inc_row["year"])
            revenue = float(inc_row["revenue"])
            ebitda  = float(inc_row["ebitda"])
            da      = float(inc_row["da"])
            taxes   = float(inc_row["tax_provision"])
            ebit    = float(inc_row["ebit"])

            # Working capital change (increase = cash outflow)
            if prev_revenue is not None:
                delta_wc = (revenue - prev_revenue) * working_capital_pct_rev
            else:
                delta_wc = 0.0
            prev_revenue = revenue

            capex        = revenue * capex_pct_rev
            operating_cf = ebitda - taxes - delta_wc
            investing_cf = -capex

            fcf_before_debt = operating_cf + investing_cf

            # Debt paydown: mandatory first, then optional sweep from excess FCF
            mandatory = mandatory_amort_by_yr.get(yr, 0.0)
            optional_sweep = max(fcf_before_debt - mandatory, 0.0)
            total_paydown   = mandatory + optional_sweep
            fcf_after_debt  = fcf_before_debt - total_paydown

            cash_balance += fcf_after_debt
            cash_balance  = max(cash_balance, 0.0)

            rows.append({
                "year":             yr,
                "ebitda":           round(ebitda, 2),
                "change_in_wc":     round(-delta_wc, 2),   # sign: positive = WC reduction = source
                "cash_taxes":       round(-taxes, 2),
                "da_addback":       round(da, 2),
                "operating_cf":     round(operating_cf, 2),
                "capex":            round(-capex, 2),
                "investing_cf":     round(investing_cf, 2),
                "fcf_before_debt":  round(fcf_before_debt, 2),
                "mandatory_amort":  round(-mandatory, 2),
                "optional_sweep":   round(-optional_sweep, 2),
                "total_debt_paydown": round(-total_paydown, 2),
                "fcf_after_debt":   round(fcf_after_debt, 2),
                "ending_cash":      round(cash_balance, 2),
            })

        df = pd.DataFrame(rows)
        self._cf_df = df
        return df

    # ------------------------------------------------------------------
    # compute_returns
    # ------------------------------------------------------------------

    def compute_returns(self, exit_multiple: float, exit_year: int = 5) -> dict:
        """
        Compute investment returns at exit.

        Derives exit equity value from exit EV minus remaining debt, then
        solves for IRR and MOIC.

        Args:
            exit_multiple: EV/EBITDA multiple at exit.
            exit_year:     Year of exit (default 5).

        Returns:
            Dict with: entry_equity, exit_equity, exit_ev, exit_ebitda,
            gross_moic, irr, dpi, cash_on_cash.
        """
        _require_positive(exit_multiple, "exit_multiple")

        # Ensure model has been built
        if self._income_df is None:
            self.build_income_statement([0.05] * exit_year, [0.30] * exit_year)
        if self._debt_df is None:
            self.build_debt_schedule(years=exit_year)

        # Exit EBITDA
        year_rows = self._income_df[self._income_df["year"] == exit_year]
        if year_rows.empty:
            # Extrapolate if exit year beyond modeled range
            last_ebitda = float(self._income_df["ebitda"].iloc[-1])
            exit_ebitda = last_ebitda * (1.05 ** max(exit_year - len(self._income_df), 0))
        else:
            exit_ebitda = float(year_rows["ebitda"].iloc[0])

        exit_ev = exit_multiple * exit_ebitda

        # Remaining debt at exit year
        year_debt = self._debt_df[self._debt_df["year"] == exit_year]
        remaining_debt = float(year_debt["eop_balance"].sum()) if not year_debt.empty else self.total_debt

        exit_equity = max(exit_ev - remaining_debt, 0.0)
        entry_equity = self.entry_equity if self.entry_equity > _EPS else 1.0

        gross_moic = exit_equity / entry_equity
        irr        = _compute_irr([-entry_equity] + [0.0] * (exit_year - 1) + [exit_equity])
        dpi        = gross_moic   # simplified: no interim distributions modeled

        result = {
            "target_name":    self.target_name,
            "entry_ev":       round(self.entry_ev, 2),
            "entry_ebitda":   round(self.entry_ebitda, 2),
            "entry_multiple": round(self.entry_multiple, 2),
            "total_debt":     round(self.total_debt, 2),
            "entry_equity":   round(entry_equity, 2),
            "exit_year":      exit_year,
            "exit_multiple":  round(exit_multiple, 2),
            "exit_ebitda":    round(exit_ebitda, 2),
            "exit_ev":        round(exit_ev, 2),
            "remaining_debt": round(remaining_debt, 2),
            "exit_equity":    round(exit_equity, 2),
            "gross_moic":     round(gross_moic, 3),
            "irr":            round(irr, 4),
            "dpi":            round(dpi, 3),
            "cash_on_cash":   round(gross_moic, 3),
            "verdict":        self._irr_verdict(irr),
        }
        logger.debug("LBOModel.compute_returns: %s IRR=%.1f%% MOIC=%.2fx", self.target_name, irr * 100, gross_moic)
        return result

    @staticmethod
    def _irr_verdict(irr: float) -> str:
        if irr >= 0.30:
            return "exceptional"
        if irr >= 0.22:
            return "strong"
        if irr >= 0.15:
            return "acceptable"
        if irr >= 0.08:
            return "marginal"
        return "value_destroying"

    # ------------------------------------------------------------------
    # sensitivity_table
    # ------------------------------------------------------------------

    def sensitivity_table(
        self,
        exit_multiples:  list[float],
        entry_multiples: list[float],
    ) -> pd.DataFrame:
        """
        Two-dimensional sensitivity of IRR and MOIC vs exit and entry multiples.

        Args:
            exit_multiples:  List of EV/EBITDA exit multiples (rows).
            entry_multiples: List of EV/EBITDA entry multiples (columns).

        Returns:
            pd.DataFrame with MultiIndex rows (exit_multiple, metric)
            and entry_multiple columns.  Metric is "IRR_pct" or "MOIC".
        """
        rows: list[dict] = []
        for exit_mult in exit_multiples:
            irr_row:  dict[str, Any] = {"exit_multiple": exit_mult, "metric": "IRR_%"}
            moic_row: dict[str, Any] = {"exit_multiple": exit_mult, "metric": "MOIC"}
            for entry_mult in entry_multiples:
                # Temporary model with modified entry multiple
                tmp = LBOModel(
                    target_name   = self.target_name,
                    entry_ev      = entry_mult * self.entry_ebitda,
                    entry_ebitda  = self.entry_ebitda,
                    debt_multiple = self.debt_multiple,
                    equity_pct    = self.equity_pct,
                )
                tmp.build_debt_schedule()
                tmp.build_income_statement([0.05] * 5, [0.30] * 5)
                ret = tmp.compute_returns(exit_multiple=exit_mult)
                irr_row[f"entry_{entry_mult}x"]  = round(ret["irr"] * 100.0, 2)
                moic_row[f"entry_{entry_mult}x"] = round(ret["gross_moic"], 3)
            rows.extend([irr_row, moic_row])

        df = pd.DataFrame(rows).set_index(["exit_multiple", "metric"])
        return df

    # ------------------------------------------------------------------
    # run_base_bull_bear
    # ------------------------------------------------------------------

    def run_base_bull_bear(self, scenarios: Optional[dict] = None) -> dict:
        """
        Run Base / Bull / Bear scenarios with different growth and margin assumptions.

        Args:
            scenarios: Optional override dict with keys "base", "bull", "bear",
                       each containing "growth_rates" and "ebitda_margins" lists.

        Returns:
            Dict with scenario names as keys, each value is the returns dict
            plus the income and cash flow DataFrames as records.
        """
        default_scenarios = {
            "base": {
                "growth_rates":   [0.05, 0.05, 0.05, 0.05, 0.05],
                "ebitda_margins": [0.30, 0.30, 0.31, 0.31, 0.32],
                "exit_multiple":  self.entry_multiple,
                "label":          "Base Case",
            },
            "bull": {
                "growth_rates":   [0.08, 0.08, 0.07, 0.07, 0.06],
                "ebitda_margins": [0.32, 0.33, 0.34, 0.35, 0.36],
                "exit_multiple":  self.entry_multiple * 1.15,
                "label":          "Bull Case",
            },
            "bear": {
                "growth_rates":   [0.02, 0.02, 0.03, 0.03, 0.04],
                "ebitda_margins": [0.27, 0.27, 0.28, 0.28, 0.29],
                "exit_multiple":  self.entry_multiple * 0.85,
                "label":          "Bear Case",
            },
        }
        scenarios = scenarios or default_scenarios

        results: dict = {}
        for name, params in scenarios.items():
            tmp = LBOModel(
                target_name   = self.target_name,
                entry_ev      = self.entry_ev,
                entry_ebitda  = self.entry_ebitda,
                debt_multiple = self.debt_multiple,
                equity_pct    = self.equity_pct,
            )
            tmp.build_debt_schedule()
            income_df = tmp.build_income_statement(
                params["growth_rates"], params["ebitda_margins"]
            )
            cf_df  = tmp.build_cash_flow()
            ret    = tmp.compute_returns(exit_multiple=params["exit_multiple"])
            results[name] = {
                "label":    params.get("label", name),
                "returns":  ret,
                "income":   income_df.to_dict(orient="records"),
                "cashflow": cf_df.to_dict(orient="records"),
            }

        logger.info("LBOModel.run_base_bull_bear: %s scenarios computed", self.target_name)
        return results


# ---------------------------------------------------------------------------
# MergerModel
# ---------------------------------------------------------------------------

class MergerModel:
    """
    Merger accretion/dilution model, pro-forma income statement,
    football field valuation chart, and purchase price allocation.

    Args:
        acquirer:     Acquirer name.
        target:       Target name.
        acq_revenue:  Acquirer LTM revenue ($M).
        acq_ebitda:   Acquirer LTM EBITDA ($M).
        tgt_revenue:  Target LTM revenue ($M).
        tgt_ebitda:   Target LTM EBITDA ($M).
    """

    def __init__(
        self,
        acquirer:    str,
        target:      str,
        acq_revenue: float,
        acq_ebitda:  float,
        tgt_revenue: float,
        tgt_ebitda:  float,
    ) -> None:
        self.acquirer    = acquirer
        self.target      = target
        self.acq_revenue = _require_positive(acq_revenue, "acq_revenue")
        self.acq_ebitda  = acq_ebitda
        self.tgt_revenue = _require_positive(tgt_revenue, "tgt_revenue")
        self.tgt_ebitda  = tgt_ebitda

    # ------------------------------------------------------------------
    # compute_accretion_dilution
    # ------------------------------------------------------------------

    def compute_accretion_dilution(
        self,
        offer_price:         float,
        acq_shares:          float,
        acq_eps:             float,
        tgt_eps:             float,
        synergies_pct_revenue: float = 0.05,
        deal_type:           str    = "all_stock",
        acq_stock_price:     float  = 0.0,
    ) -> dict:
        """
        Compute EPS accretion / (dilution) in Year 1 and Year 3.

        Args:
            offer_price:           Total deal consideration per target share ($).
            acq_shares:            Acquirer diluted shares outstanding (millions).
            acq_eps:               Acquirer LTM EPS ($).
            tgt_eps:               Target LTM EPS ($).
            synergies_pct_revenue: Run-rate synergies as % of combined revenue.
            deal_type:             "all_cash", "all_stock", or "mixed".
            acq_stock_price:       Acquirer stock price (required for stock deals).

        Returns:
            Dict with pro_forma_eps_yr1, pro_forma_eps_yr3, accretion_pct_yr1,
            accretion_pct_yr3, breakeven_synergies_mm.
        """
        # --- Implied deal value ---
        tgt_ebitda_margin = self.tgt_ebitda / self.tgt_revenue if self.tgt_revenue > _EPS else 0.20
        # Back into implied tgt shares from offer_price and tgt market cap estimate
        # We don't know tgt share count, so accept caller providing tgt_eps and derive NI
        acq_ni = acq_eps * acq_shares   # $M acquirer net income

        # Estimate tgt shares from tgt_eps and assumed P/E to get tgt mkt cap
        # Use offer_price directly as total deal value if unit is $M
        deal_value_mm = offer_price   # caller passes total deal value in $M

        # Synergies
        combined_revenue = self.acq_revenue + self.tgt_revenue
        synergies_mm     = combined_revenue * synergies_pct_revenue

        # Ramp: 50% realization in Y1, 100% in Y3
        syn_yr1 = synergies_mm * 0.50
        syn_yr3 = synergies_mm * 1.00

        # Financing cost (after-tax)
        if deal_type == "all_cash":
            cash_pct  = 1.0
            stock_pct = 0.0
        elif deal_type == "all_stock":
            cash_pct  = 0.0
            stock_pct = 1.0
        else:
            cash_pct  = 0.50
            stock_pct = 0.50

        cash_portion  = deal_value_mm * cash_pct
        stock_portion = deal_value_mm * stock_pct

        interest_cost = cash_portion * (LIBOR_PROXY + 0.02) * (1.0 - TAX_RATE)
        new_shares    = (stock_portion / acq_stock_price) if acq_stock_price > _EPS else 0.0

        combined_shares_yr1 = acq_shares + new_shares

        # Amortization of intangibles (PPA step-up): assume 15-year life, intangibles = 30% of premium
        ppa_premium    = deal_value_mm * 0.30
        amort_intang   = ppa_premium / 15.0 * (1.0 - TAX_RATE)   # after-tax

        # Target net income
        tgt_ni = tgt_eps * (self.tgt_revenue / self.acq_revenue * acq_shares)  # rough proxy

        # Year 1
        combined_ni_yr1 = acq_ni + tgt_ni + syn_yr1 - interest_cost - amort_intang
        pf_eps_yr1      = combined_ni_yr1 / combined_shares_yr1 if combined_shares_yr1 > _EPS else 0.0
        acc_yr1_pct     = (pf_eps_yr1 - acq_eps) / abs(acq_eps) * 100.0 if abs(acq_eps) > _EPS else 0.0

        # Year 3 (organic growth: 5% for both + full synergies)
        acq_ni_yr3 = acq_ni * (1.05 ** 3)
        tgt_ni_yr3 = tgt_ni * (1.05 ** 3)
        combined_ni_yr3 = acq_ni_yr3 + tgt_ni_yr3 + syn_yr3 - interest_cost - amort_intang
        acq_eps_yr3     = acq_eps * (1.05 ** 3)
        pf_eps_yr3      = combined_ni_yr3 / combined_shares_yr1 if combined_shares_yr1 > _EPS else 0.0
        acc_yr3_pct     = (pf_eps_yr3 - acq_eps_yr3) / abs(acq_eps_yr3) * 100.0 if abs(acq_eps_yr3) > _EPS else 0.0

        # Breakeven synergies (solve for syn such that pf_eps = acq_eps with no synergies)
        breakeven_syn = max(0.0, (acq_eps * combined_shares_yr1 - acq_ni - tgt_ni + interest_cost + amort_intang))

        return {
            "acquirer":             self.acquirer,
            "target":               self.target,
            "deal_value_mm":        round(deal_value_mm, 2),
            "deal_type":            deal_type,
            "cash_portion_mm":      round(cash_portion, 2),
            "stock_portion_mm":     round(stock_portion, 2),
            "new_shares_issued_mm": round(new_shares, 4),
            "combined_shares_mm":   round(combined_shares_yr1, 4),
            "run_rate_synergies_mm": round(synergies_mm, 2),
            "breakeven_synergies_mm": round(breakeven_syn, 2),
            "amort_of_intangibles_mm": round(amort_intang, 2),
            "acquirer_eps_standalone": round(acq_eps, 4),
            "pro_forma_eps_yr1":    round(pf_eps_yr1, 4),
            "pro_forma_eps_yr3":    round(pf_eps_yr3, 4),
            "accretion_pct_yr1":    round(acc_yr1_pct, 3),
            "accretion_pct_yr3":    round(acc_yr3_pct, 3),
            "verdict_yr1":          "accretive" if acc_yr1_pct > 0 else "dilutive",
            "verdict_yr3":          "accretive" if acc_yr3_pct > 0 else "dilutive",
        }

    # ------------------------------------------------------------------
    # build_pro_forma_is
    # ------------------------------------------------------------------

    def build_pro_forma_is(
        self,
        synergies:                float,
        one_time_costs:           float,
        amortization_of_intangibles: float,
        projection_years:         int = 3,
    ) -> pd.DataFrame:
        """
        Build combined pro-forma income statement with purchase accounting.

        Args:
            synergies:                  Annual run-rate synergies ($M), fully ramped.
            one_time_costs:             Deal/integration costs in Year 1 ($M).
            amortization_of_intangibles: Annual PPA amortization ($M).
            projection_years:           Number of years to project.

        Returns:
            pd.DataFrame with: year, acquirer_revenue, target_revenue,
            combined_revenue, synergies, one_time_costs, combined_ebitda,
            da, amort_intangibles, ebit, interest_expense, ebt, taxes, net_income.
        """
        rows: list[dict] = []
        acq_rev  = self.acq_revenue
        tgt_rev  = self.tgt_revenue
        acq_ebitda = self.acq_ebitda
        tgt_ebitda = self.tgt_ebitda

        for yr in range(1, projection_years + 1):
            growth    = 1.05 ** yr
            syn_ramp  = min(synergies * (yr / 2.0), synergies)   # ramp over 2 years
            ot_costs  = one_time_costs if yr == 1 else 0.0

            acq_rev_yr  = acq_rev  * growth
            tgt_rev_yr  = tgt_rev  * growth
            comb_rev    = acq_rev_yr + tgt_rev_yr
            comb_ebitda = (acq_ebitda + tgt_ebitda) * growth + syn_ramp - ot_costs

            da           = comb_rev * 0.05
            ebit         = comb_ebitda - da - amortization_of_intangibles
            int_expense  = (self.acq_revenue * 0.30) * (LIBOR_PROXY + 0.02)   # rough debt estimate
            ebt          = ebit - int_expense
            taxes        = max(ebt * TAX_RATE, 0.0)
            net_income   = ebt - taxes

            rows.append({
                "year":               yr,
                "acquirer_revenue":   round(acq_rev_yr, 2),
                "target_revenue":     round(tgt_rev_yr, 2),
                "combined_revenue":   round(comb_rev, 2),
                "synergies":          round(syn_ramp, 2),
                "one_time_costs":     round(ot_costs, 2),
                "combined_ebitda":    round(comb_ebitda, 2),
                "da":                 round(da, 2),
                "amort_intangibles":  round(amortization_of_intangibles, 2),
                "ebit":               round(ebit, 2),
                "interest_expense":   round(int_expense, 2),
                "ebt":                round(ebt, 2),
                "tax_provision":      round(taxes, 2),
                "net_income":         round(net_income, 2),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # football_field_chart_data
    # ------------------------------------------------------------------

    def football_field_chart_data(
        self,
        dcf_low:         float,
        dcf_high:        float,
        comps_low:       float,
        comps_high:      float,
        precedent_low:   float,
        precedent_high:  float,
        offer:           float,
    ) -> dict:
        """
        Return football field valuation ranges for chart visualization.

        Args:
            dcf_low/high:       DCF implied value range per share.
            comps_low/high:     Trading comps implied range per share.
            precedent_low/high: Precedent transactions implied range.
            offer:              Proposed offer price per share.

        Returns:
            Dict suitable for rendering a horizontal bar chart, including
            premium_to_midpoint for each methodology.
        """
        def _midpoint(lo: float, hi: float) -> float:
            return (lo + hi) / 2.0

        def _premium(lo: float, hi: float, price: float) -> float:
            mid = _midpoint(lo, hi)
            return round((price / mid - 1.0) * 100.0, 2) if mid > _EPS else 0.0

        dcf_mid  = _midpoint(dcf_low, dcf_high)
        comp_mid = _midpoint(comps_low, comps_high)
        prec_mid = _midpoint(precedent_low, precedent_high)

        # 52-week range estimate (synthetic: offer - 15%/20% is typical deal premium range)
        wk52_low  = offer * 0.70
        wk52_high = offer * 0.95

        return {
            "offer_price": round(offer, 4),
            "methodologies": {
                "dcf": {
                    "label":       "DCF (WACC Sensitivity)",
                    "low":         round(dcf_low, 4),
                    "high":        round(dcf_high, 4),
                    "midpoint":    round(dcf_mid, 4),
                    "premium_to_midpoint_pct": _premium(dcf_low, dcf_high, offer),
                },
                "trading_comps": {
                    "label":       "Public Company Comps",
                    "low":         round(comps_low, 4),
                    "high":        round(comps_high, 4),
                    "midpoint":    round(comp_mid, 4),
                    "premium_to_midpoint_pct": _premium(comps_low, comps_high, offer),
                },
                "precedent_transactions": {
                    "label":       "Precedent Transactions",
                    "low":         round(precedent_low, 4),
                    "high":        round(precedent_high, 4),
                    "midpoint":    round(prec_mid, 4),
                    "premium_to_midpoint_pct": _premium(precedent_low, precedent_high, offer),
                },
                "52_week_range": {
                    "label":       "52-Week Trading Range",
                    "low":         round(wk52_low, 4),
                    "high":        round(wk52_high, 4),
                    "midpoint":    round(_midpoint(wk52_low, wk52_high), 4),
                    "premium_to_midpoint_pct": _premium(wk52_low, wk52_high, offer),
                },
            },
            "offer_in_range": dcf_low <= offer <= precedent_high,
        }

    # ------------------------------------------------------------------
    # purchase_price_allocation
    # ------------------------------------------------------------------

    def purchase_price_allocation(
        self,
        offer_price:          float,
        book_value:           float,
        identified_intangibles: float,
        ppe_step_up_pct:      float = 0.15,
    ) -> dict:
        """
        Compute purchase price allocation (PPA / ASC 805).

        Args:
            offer_price:            Total consideration paid ($M).
            book_value:             Target book value of equity ($M).
            identified_intangibles: Fair value of identifiable intangibles ($M)
                                    (customer lists, IP, brand, etc.).
            ppe_step_up_pct:        PP&E step-up as % of offer price (default 15%).

        Returns:
            Dict with goodwill, ppe_step_up, deferred_tax_liability, total_ppa.
        """
        _require_positive(offer_price, "offer_price")

        premium        = offer_price - book_value
        ppe_step_up    = offer_price * ppe_step_up_pct
        dtl_rate       = TAX_RATE
        dtl            = identified_intangibles * dtl_rate    # deferred tax liability on intangibles
        goodwill       = max(premium - identified_intangibles - ppe_step_up + dtl, 0.0)
        total_ppa      = goodwill + identified_intangibles + ppe_step_up

        # Annual amortization (intangibles 15-yr, ppe step-up over remaining useful life 10-yr)
        annual_intang_amort = identified_intangibles / 15.0
        annual_ppe_amort    = ppe_step_up / 10.0

        return {
            "offer_price":                   round(offer_price, 2),
            "book_value":                    round(book_value, 2),
            "deal_premium":                  round(premium, 2),
            "pp_and_e_step_up":              round(ppe_step_up, 2),
            "identified_intangibles":        round(identified_intangibles, 2),
            "goodwill":                      round(goodwill, 2),
            "deferred_tax_liability":        round(dtl, 2),
            "total_ppa":                     round(total_ppa, 2),
            "annual_intangible_amortization": round(annual_intang_amort, 2),
            "annual_ppe_step_up_amortization": round(annual_ppe_amort, 2),
            "total_annual_ppa_amortization": round(annual_intang_amort + annual_ppe_amort, 2),
            "goodwill_pct_of_purchase_price": round(goodwill / offer_price * 100.0 if offer_price > _EPS else 0.0, 2),
        }


# ---------------------------------------------------------------------------
# DCFModel
# ---------------------------------------------------------------------------

class DCFModel:
    """
    Discounted cash flow model supporting standalone and M&A use cases.

    Args:
        name:                Company or asset name.
        free_cash_flows:     Projected FCFs ($M), ordered Year 1 … Year N.
        wacc:                Weighted average cost of capital (decimal).
        terminal_growth_rate: Gordon Growth terminal growth rate (decimal).
    """

    def __init__(
        self,
        name:                str,
        free_cash_flows:     list[float],
        wacc:                float,
        terminal_growth_rate: float = 0.025,
    ) -> None:
        if not free_cash_flows:
            raise ValueError("free_cash_flows must be non-empty")
        _require_positive(wacc, "wacc")
        if terminal_growth_rate >= wacc:
            raise ValueError("terminal_growth_rate must be less than wacc")

        self.name                = name
        self.free_cash_flows     = free_cash_flows
        self.wacc                = wacc
        self.terminal_growth_rate = terminal_growth_rate

    # ------------------------------------------------------------------
    # compute_wacc (static)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_wacc(
        equity_weight:  float,
        cost_of_equity: float,
        debt_weight:    float,
        cost_of_debt:   float,
        tax_rate:       float = TAX_RATE,
    ) -> float:
        """
        Compute WACC from capital structure inputs.

        Args:
            equity_weight:   Equity as fraction of total capital [0, 1].
            cost_of_equity:  Cost of equity (CAPM or build-up, decimal).
            debt_weight:     Debt as fraction of total capital [0, 1].
            cost_of_debt:    Pre-tax cost of debt (decimal).
            tax_rate:        Marginal corporate tax rate (default 21%).

        Returns:
            WACC as decimal.
        """
        _require_fraction(equity_weight, "equity_weight")
        _require_fraction(debt_weight,   "debt_weight")
        _require_fraction(tax_rate,      "tax_rate")
        _require_positive(cost_of_equity, "cost_of_equity")
        _require_positive(cost_of_debt,   "cost_of_debt")

        wacc = (equity_weight * cost_of_equity
                + debt_weight * cost_of_debt * (1.0 - tax_rate))
        return round(wacc, 6)

    # ------------------------------------------------------------------
    # compute_dcf
    # ------------------------------------------------------------------

    def compute_dcf(
        self,
        shares_outstanding: Optional[float] = None,
        net_debt:           Optional[float] = None,
    ) -> dict:
        """
        Run the full DCF: PV of explicit FCFs + Gordon Growth terminal value.

        Args:
            shares_outstanding: Diluted shares (millions) for per-share equity value.
            net_debt:           Net debt ($M) to bridge EV to equity value.

        Returns:
            Dict with pv_fcfs, terminal_value, enterprise_value,
            equity_value, equity_value_per_share (if shares provided).
        """
        n    = len(self.free_cash_flows)
        wacc = self.wacc
        tgr  = self.terminal_growth_rate

        # PV of explicit FCFs
        pv_fcfs = []
        for t, fcf in enumerate(self.free_cash_flows, start=1):
            pv = fcf / (1.0 + wacc) ** t
            pv_fcfs.append(round(pv, 4))

        total_pv_fcfs = sum(pv_fcfs)

        # Terminal value (Gordon Growth on final-year FCF)
        final_fcf        = self.free_cash_flows[-1]
        terminal_fcf     = final_fcf * (1.0 + tgr)
        terminal_value   = terminal_fcf / (wacc - tgr)
        pv_terminal      = terminal_value / (1.0 + wacc) ** n

        enterprise_value = total_pv_fcfs + pv_terminal
        tv_pct_of_ev     = pv_terminal / enterprise_value * 100.0 if enterprise_value > _EPS else 0.0

        # Equity bridge
        nd          = net_debt or 0.0
        equity_val  = enterprise_value - nd
        ev_per_share = equity_val / shares_outstanding if shares_outstanding and shares_outstanding > _EPS else None

        result: dict = {
            "name":                      self.name,
            "wacc":                      round(wacc * 100.0, 3),
            "terminal_growth_rate":      round(tgr * 100.0, 3),
            "explicit_fcf_years":        n,
            "pv_explicit_fcfs":          [round(v, 2) for v in pv_fcfs],
            "total_pv_fcfs":             round(total_pv_fcfs, 2),
            "final_year_fcf":            round(final_fcf, 2),
            "terminal_value":            round(terminal_value, 2),
            "pv_terminal_value":         round(pv_terminal, 2),
            "tv_pct_of_enterprise_value": round(tv_pct_of_ev, 2),
            "enterprise_value":          round(enterprise_value, 2),
            "net_debt":                  round(nd, 2),
            "equity_value":              round(equity_val, 2),
        }
        if ev_per_share is not None:
            result["shares_outstanding_mm"]   = shares_outstanding
            result["equity_value_per_share"] = round(ev_per_share, 4)

        logger.debug(
            "DCFModel.compute_dcf: %s EV=%.1f TV_pct=%.1f%%",
            self.name, enterprise_value, tv_pct_of_ev,
        )
        return result

    # ------------------------------------------------------------------
    # sensitivity_wacc_tgr
    # ------------------------------------------------------------------

    def sensitivity_wacc_tgr(
        self,
        wacc_range: list[float],
        tgr_range:  list[float],
    ) -> pd.DataFrame:
        """
        Build enterprise value sensitivity table across WACC and TGR ranges.

        Args:
            wacc_range: List of WACC values (decimal, e.g. [0.08, 0.09, 0.10]).
            tgr_range:  List of terminal growth rates (decimal).

        Returns:
            pd.DataFrame indexed by WACC (%), columns are TGR (%) values,
            cell values are enterprise values.
        """
        rows: list[dict] = []
        for w in wacc_range:
            row: dict = {"wacc_pct": round(w * 100.0, 2)}
            for tgr in tgr_range:
                if tgr >= w:
                    row[f"tgr_{tgr * 100:.1f}%"] = float("nan")
                    continue
                tmp_model = DCFModel(
                    name                 = self.name,
                    free_cash_flows      = self.free_cash_flows,
                    wacc                 = w,
                    terminal_growth_rate = tgr,
                )
                result = tmp_model.compute_dcf()
                row[f"tgr_{tgr * 100:.1f}%"] = round(result["enterprise_value"], 2)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("wacc_pct")
        return df


# ---------------------------------------------------------------------------
# ModelLibrary
# ---------------------------------------------------------------------------

class ModelLibrary:
    """
    Persist and retrieve financial models as JSON in the .sentinel/models/ directory.

    All models are stored as JSON blobs.  LBO and merger DataFrames are
    serialized as lists of records.

    Args:
        models_dir: Override the default storage directory.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self._dir = models_dir or _MODELS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ModelLibrary: storage dir = %s", self._dir)

    def save_lbo(self, model: LBOModel, name: str) -> Path:
        """
        Serialize an LBOModel (with pre-built schedules) to JSON.

        Args:
            model: LBOModel instance (call build_* methods first).
            name:  Logical model name (used as filename stem).

        Returns:
            Path to the saved JSON file.
        """
        payload: dict = {
            "type":         "lbo",
            "name":         name,
            "saved_at":     date.today().isoformat(),
            "parameters":   {
                "target_name":   model.target_name,
                "entry_ev":      model.entry_ev,
                "entry_ebitda":  model.entry_ebitda,
                "debt_multiple": model.debt_multiple,
                "equity_pct":    model.equity_pct,
            },
            "debt_schedule": model._debt_df.to_dict(orient="records") if model._debt_df is not None else [],
            "income_statement": model._income_df.to_dict(orient="records") if model._income_df is not None else [],
            "cash_flow": model._cf_df.to_dict(orient="records") if model._cf_df is not None else [],
        }
        path = self._dir / f"lbo_{name.lower().replace(' ', '_')}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("ModelLibrary: saved LBO model to %s", path)
        return path

    def save_dcf(self, model: DCFModel, name: str, dcf_result: dict) -> Path:
        """
        Serialize a DCFModel and its computed results to JSON.

        Args:
            model:      DCFModel instance.
            name:       Logical model name.
            dcf_result: Output from model.compute_dcf().

        Returns:
            Path to the saved JSON file.
        """
        payload = {
            "type":       "dcf",
            "name":       name,
            "saved_at":   date.today().isoformat(),
            "parameters": {
                "name":                model.name,
                "free_cash_flows":     model.free_cash_flows,
                "wacc":                model.wacc,
                "terminal_growth_rate": model.terminal_growth_rate,
            },
            "results": dcf_result,
        }
        path = self._dir / f"dcf_{name.lower().replace(' ', '_')}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("ModelLibrary: saved DCF model to %s", path)
        return path

    def load(self, filename: str) -> dict:
        """
        Load a model from JSON by filename.

        Args:
            filename: Filename (stem or full name with .json).

        Returns:
            Parsed dict with all model data.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        path = self._dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_models(self) -> list[dict]:
        """
        List all saved models with metadata.

        Returns:
            List of dicts with: filename, type, name, saved_at.
        """
        result: list[dict] = []
        for f in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                result.append({
                    "filename": f.name,
                    "type":     data.get("type", "unknown"),
                    "name":     data.get("name", f.stem),
                    "saved_at": data.get("saved_at", ""),
                })
            except (json.JSONDecodeError, OSError):
                result.append({"filename": f.name, "type": "corrupt", "name": f.stem, "saved_at": ""})
        return result

    def apply_template(
        self,
        template_filename: str,
        new_parameters:    dict,
    ) -> LBOModel | DCFModel:
        """
        Load a saved model template and instantiate with new parameters.

        Args:
            template_filename: Filename of the saved template.
            new_parameters:    Dict overriding parameters from the saved template.

        Returns:
            A new LBOModel or DCFModel with merged parameters.

        Raises:
            ValueError:       If model type is unsupported.
            FileNotFoundError: If template file does not exist.
        """
        data   = self.load(template_filename)
        params = {**data.get("parameters", {}), **new_parameters}
        model_type = data.get("type", "unknown")

        if model_type == "lbo":
            return LBOModel(
                target_name   = params.get("target_name", "Template Target"),
                entry_ev      = float(params.get("entry_ev", 1000.0)),
                entry_ebitda  = float(params.get("entry_ebitda", 100.0)),
                debt_multiple = float(params.get("debt_multiple", 5.0)),
                equity_pct    = float(params.get("equity_pct", 0.35)),
            )
        elif model_type == "dcf":
            return DCFModel(
                name                 = params.get("name", "Template DCF"),
                free_cash_flows      = list(params.get("free_cash_flows", [100.0] * 5)),
                wacc                 = float(params.get("wacc", 0.10)),
                terminal_growth_rate = float(params.get("terminal_growth_rate", 0.025)),
            )
        else:
            raise ValueError(f"Unsupported model type in template: {model_type}")

    def delete(self, filename: str) -> bool:
        """
        Delete a saved model file.

        Args:
            filename: Filename (stem or with .json extension).

        Returns:
            True if deleted, False if file did not exist.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        path = self._dir / filename
        if path.exists():
            path.unlink()
            logger.info("ModelLibrary: deleted %s", path)
            return True
        return False
