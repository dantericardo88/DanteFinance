"""DCF / WACC / comparable company valuation templates — Damodaran methodology."""
from __future__ import annotations

import statistics
from typing import Any

import numpy as np
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class DCFAssumptions(BaseModel):
    ticker: str
    revenue_base: float                      # trailing 12M revenue
    revenue_growth_rates: list[float]        # per-year growth, e.g. [0.15, 0.12, 0.10, 0.08, 0.06]
    terminal_growth_rate: float = 0.025
    ebit_margin: float                       # stabilized EBIT margin
    tax_rate: float = 0.21
    capex_pct_revenue: float = 0.05
    da_pct_revenue: float = 0.04             # D&A as % of revenue
    nwc_change_pct_revenue: float = 0.02     # change in NWC as % of revenue
    wacc: float                              # discount rate
    net_debt: float = 0.0                    # debt - cash (positive = net debt)
    shares_outstanding: float               # in millions


class WACCInputs(BaseModel):
    equity_beta: float                       # levered beta
    risk_free_rate: float = 0.045            # 10Y Treasury
    equity_risk_premium: float = 0.055       # Damodaran ERP
    cost_of_debt: float = 0.06
    tax_rate: float = 0.21
    debt_to_equity: float = 0.3
    equity_weight: float                     # E/(D+E)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class DCFResult(BaseModel):
    ticker: str
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    upside_pct: float                        # vs current_price
    current_price: float
    terminal_value: float
    pv_fcf_sum: float
    year_fcfs: list[float]
    sensitivity: dict[str, dict[str, float]]  # WACC × terminal_growth sensitivity table


class CompsTable(BaseModel):
    target_ticker: str
    peers: list[dict]                        # peer → {pe, pb, ev_ebitda, ps, …}
    implied_values: dict[str, float]         # multiple → implied price for target


# ---------------------------------------------------------------------------
# WACC computation
# ---------------------------------------------------------------------------

def compute_wacc(inputs: WACCInputs) -> float:
    """
    WACC = Ke * We + Kd * (1-t) * Wd.
    CAPM for Ke: Rf + beta * ERP.
    equity_weight is E/(D+E); debt_weight = 1 - equity_weight.
    """
    ke = inputs.risk_free_rate + inputs.equity_beta * inputs.equity_risk_premium
    debt_weight = 1.0 - inputs.equity_weight
    kd_after_tax = inputs.cost_of_debt * (1.0 - inputs.tax_rate)
    wacc = ke * inputs.equity_weight + kd_after_tax * debt_weight
    logger.info(
        "WACC computed",
        ke=round(ke, 4),
        kd_after_tax=round(kd_after_tax, 4),
        wacc=round(wacc, 4),
        equity_weight=inputs.equity_weight,
    )
    return wacc


def compute_levered_beta(
    unlevered_beta: float,
    tax_rate: float,
    de_ratio: float,
) -> float:
    """
    Hamada equation: beta_L = beta_U * (1 + (1 - t) * D/E).
    Used to re-lever industry betas at a firm's specific capital structure.
    """
    return unlevered_beta * (1.0 + (1.0 - tax_rate) * de_ratio)


# ---------------------------------------------------------------------------
# FCF projection helper
# ---------------------------------------------------------------------------

def _project_fcf(assumptions: DCFAssumptions) -> list[float]:
    """
    Project free cash flow for each year in revenue_growth_rates.
    FCF = EBIT*(1-t) + D&A - Capex - ΔNWC
    where each line item is expressed as % of that year's projected revenue.
    """
    revenue = assumptions.revenue_base
    fcfs: list[float] = []

    for g in assumptions.revenue_growth_rates:
        revenue = revenue * (1.0 + g)
        ebit = revenue * assumptions.ebit_margin
        nopat = ebit * (1.0 - assumptions.tax_rate)
        da = revenue * assumptions.da_pct_revenue
        capex = revenue * assumptions.capex_pct_revenue
        delta_nwc = revenue * assumptions.nwc_change_pct_revenue
        fcf = nopat + da - capex - delta_nwc
        fcfs.append(fcf)

    return fcfs


# ---------------------------------------------------------------------------
# Sensitivity table builder
# ---------------------------------------------------------------------------

def _build_sensitivity(
    assumptions: DCFAssumptions,
    year_fcfs: list[float],
    current_price: float,
) -> dict[str, dict[str, float]]:
    """
    5×5 sensitivity of intrinsic value per share across WACC ±2% and terminal growth ±1%.
    Returns a dict of {wacc_label: {tg_label: value_per_share}}.
    """
    base_wacc = assumptions.wacc
    base_tg = assumptions.terminal_growth_rate
    n = len(year_fcfs)

    wacc_deltas = [-0.02, -0.01, 0.0, 0.01, 0.02]
    tg_deltas = [-0.01, -0.005, 0.0, 0.005, 0.01]

    table: dict[str, dict[str, float]] = {}

    for wd in wacc_deltas:
        wacc = base_wacc + wd
        wacc_label = f"WACC={wacc:.1%}"
        table[wacc_label] = {}
        for td in tg_deltas:
            tg = base_tg + td
            tg_label = f"g={tg:.2%}"
            if wacc <= tg:
                # Gordon Growth Model undefined when g >= WACC
                table[wacc_label][tg_label] = float("nan")
                continue
            # PV of explicit FCFs at this WACC
            pv_sum = sum(fcf / (1.0 + wacc) ** (i + 1) for i, fcf in enumerate(year_fcfs))
            # Terminal value based on last FCF
            last_fcf = year_fcfs[-1] if year_fcfs else 0.0
            tv = last_fcf * (1.0 + tg) / (wacc - tg)
            pv_tv = tv / (1.0 + wacc) ** n
            ev = pv_sum + pv_tv - assumptions.net_debt
            per_share = ev / assumptions.shares_outstanding if assumptions.shares_outstanding > 0 else 0.0
            table[wacc_label][tg_label] = round(per_share, 2)

    return table


# ---------------------------------------------------------------------------
# Main DCF runner
# ---------------------------------------------------------------------------

def run_dcf(
    assumptions: DCFAssumptions,
    current_price: float,
) -> DCFResult:
    """
    5-stage DCF (Damodaran methodology):

    1. Project FCF for each year: EBIT*(1-t) + D&A - Capex - ΔNWC
    2. Discount each year's FCF to PV using WACC
    3. Terminal value = FCF_year_N * (1+g) / (WACC - g)
    4. PV of TV = TV / (1+WACC)^N
    5. EV = sum(PV FCFs) + PV(TV) - net_debt
    6. Intrinsic value = EV / shares_outstanding (millions → price per share)
    7. Build 5×5 sensitivity table: WACC ±2% × terminal_growth ±1%
    """
    if assumptions.wacc <= assumptions.terminal_growth_rate:
        raise ValueError(
            f"WACC ({assumptions.wacc:.2%}) must exceed terminal growth rate "
            f"({assumptions.terminal_growth_rate:.2%}) for a finite terminal value."
        )
    if not assumptions.revenue_growth_rates:
        raise ValueError("revenue_growth_rates must contain at least one period.")
    if assumptions.shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive.")

    n = len(assumptions.revenue_growth_rates)
    year_fcfs = _project_fcf(assumptions)

    # Discount explicit-period FCFs
    pv_fcfs = [
        fcf / (1.0 + assumptions.wacc) ** (i + 1)
        for i, fcf in enumerate(year_fcfs)
    ]
    pv_fcf_sum = sum(pv_fcfs)

    # Terminal value (Gordon Growth on last-year FCF)
    last_fcf = year_fcfs[-1]
    terminal_value = last_fcf * (1.0 + assumptions.terminal_growth_rate) / (
        assumptions.wacc - assumptions.terminal_growth_rate
    )
    pv_terminal_value = terminal_value / (1.0 + assumptions.wacc) ** n

    # Enterprise value → equity value → per-share
    enterprise_value = pv_fcf_sum + pv_terminal_value
    equity_value = enterprise_value - assumptions.net_debt
    intrinsic_value_per_share = equity_value / assumptions.shares_outstanding

    upside_pct = (intrinsic_value_per_share - current_price) / current_price if current_price > 0 else 0.0

    sensitivity = _build_sensitivity(assumptions, year_fcfs, current_price)

    logger.info(
        "DCF complete",
        ticker=assumptions.ticker,
        intrinsic_value=round(intrinsic_value_per_share, 2),
        current_price=current_price,
        upside_pct=round(upside_pct, 4),
        pv_fcf_sum=round(pv_fcf_sum, 0),
        terminal_value=round(terminal_value, 0),
    )

    return DCFResult(
        ticker=assumptions.ticker,
        intrinsic_value_per_share=round(intrinsic_value_per_share, 2),
        enterprise_value=round(enterprise_value, 2),
        equity_value=round(equity_value, 2),
        upside_pct=round(upside_pct, 4),
        current_price=current_price,
        terminal_value=round(terminal_value, 2),
        pv_fcf_sum=round(pv_fcf_sum, 2),
        year_fcfs=[round(f, 2) for f in year_fcfs],
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# Comparable company analysis
# ---------------------------------------------------------------------------

_SUPPORTED_MULTIPLES = ("pe", "pb", "ev_ebitda", "ps")


def _safe_median(values: list[float]) -> float | None:
    """Median of a list, ignoring None and NaN."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return statistics.median(clean)


def build_comps_table(
    target: dict[str, Any],
    peers: list[dict[str, Any]],
) -> CompsTable:
    """
    Comparable company analysis using median peer multiples.

    target schema: {ticker, pe, pb, ev_ebitda, ps, revenue, ebitda, book_value, shares}
    peers schema: same as target (each peer must include 'ticker').

    Returns CompsTable with:
    - peers: the raw peer inputs
    - implied_values: {multiple_name: implied_price_per_share_for_target}

    Implied price derivation per multiple:
      P/E:       median(peer P/E) * target_EPS
      P/B:       median(peer P/B) * target_book_per_share
      EV/EBITDA: median(peer EV/EBITDA) * target_EBITDA → EV → price
      P/S:       median(peer P/S) * target_revenue_per_share
    """
    target_ticker = target.get("ticker", "TARGET")
    shares = float(target.get("shares", 1.0))  # millions
    if shares <= 0:
        raise ValueError("target shares must be positive")

    # Per-share metrics for target
    revenue_per_share = float(target.get("revenue", 0.0)) / shares if shares else 0.0
    ebitda = float(target.get("ebitda", 0.0))
    book_value_per_share = float(target.get("book_value", 0.0)) / shares if shares else 0.0
    # EPS implied from P/E — if not given, approximate from ebitda*(1-0.21)/shares
    target_eps = float(target.get("eps", ebitda * 0.79 / shares if shares else 0.0))

    implied_values: dict[str, float] = {}

    # P/E implied price
    peer_pes = [float(p["pe"]) for p in peers if p.get("pe") is not None]
    median_pe = _safe_median(peer_pes)
    if median_pe is not None and target_eps > 0:
        implied_values["pe"] = round(median_pe * target_eps, 2)
    else:
        implied_values["pe"] = float("nan")

    # P/B implied price
    peer_pbs = [float(p["pb"]) for p in peers if p.get("pb") is not None]
    median_pb = _safe_median(peer_pbs)
    if median_pb is not None and book_value_per_share > 0:
        implied_values["pb"] = round(median_pb * book_value_per_share, 2)
    else:
        implied_values["pb"] = float("nan")

    # EV/EBITDA implied price
    peer_ev_ebitdas = [float(p["ev_ebitda"]) for p in peers if p.get("ev_ebitda") is not None]
    median_ev_ebitda = _safe_median(peer_ev_ebitdas)
    if median_ev_ebitda is not None and ebitda > 0:
        implied_ev = median_ev_ebitda * ebitda
        # EV → equity value → price (assume zero net debt unless specified)
        net_debt = float(target.get("net_debt", 0.0))
        implied_equity = implied_ev - net_debt
        implied_values["ev_ebitda"] = round(implied_equity / shares, 2)
    else:
        implied_values["ev_ebitda"] = float("nan")

    # P/S implied price
    peer_pss = [float(p["ps"]) for p in peers if p.get("ps") is not None]
    median_ps = _safe_median(peer_pss)
    if median_ps is not None and revenue_per_share > 0:
        implied_values["ps"] = round(median_ps * revenue_per_share, 2)
    else:
        implied_values["ps"] = float("nan")

    # Composite: equal-weight average of available implied values
    valid_values = [v for v in implied_values.values() if not np.isnan(v)]
    if valid_values:
        implied_values["composite_avg"] = round(statistics.mean(valid_values), 2)
        implied_values["composite_median"] = round(statistics.median(valid_values), 2)

    logger.info(
        "Comps table built",
        target=target_ticker,
        peer_count=len(peers),
        implied_pe=implied_values.get("pe"),
        implied_ev_ebitda=implied_values.get("ev_ebitda"),
        composite=implied_values.get("composite_avg"),
    )

    return CompsTable(
        target_ticker=target_ticker,
        peers=peers,
        implied_values=implied_values,
    )
