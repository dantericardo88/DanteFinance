"""LBO and merger/accretion-dilution model templates — PE and M&A advisory grade."""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class LBOAssumptions(BaseModel):
    purchase_price: float
    ebitda: float
    ebitda_growth_rate: float
    leverage_multiple: float
    interest_rate: float
    hold_years: int
    exit_multiple: float
    tax_rate: float = 0.25
    amortization_pct: float = 0.05


class MergerAssumptions(BaseModel):
    acquirer_eps: float
    acquirer_shares_mm: float
    acquirer_price: float
    target_eps: float
    target_shares_mm: float
    acquisition_price_per_share: float
    pct_stock: float = 0.0
    synergies_after_tax_mm: float = 0.0
    cost_of_debt: float = 0.08
    tax_rate: float = 0.25


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class LBOYear(BaseModel):
    year: int
    ebitda: float
    interest_expense: float
    debt_balance: float
    free_cash_flow: float
    cumulative_debt_paid: float


class LBOResult(BaseModel):
    assumptions: LBOAssumptions
    equity_invested: float
    total_debt: float
    years: list[LBOYear]
    exit_ebitda: float
    exit_ev: float
    exit_equity: float
    moic: float
    irr: float
    cash_yield: float
    verdict: str
    warnings: list[str] = Field(default_factory=list)


class MergerResult(BaseModel):
    assumptions: MergerAssumptions
    deal_value_mm: float
    equity_consideration_mm: float
    cash_consideration_mm: float
    new_shares_issued_mm: float
    combined_eps: float
    acquirer_eps: float
    accretion_pct: float
    verdict: str
    premium_paid_pct: float
    warnings: list[str] = Field(default_factory=list)


class LBOScreenResult(BaseModel):
    ticker: str
    entity_name: str
    lbo_result: LBOResult
    market_data_used: dict
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# IRR bisection
# ---------------------------------------------------------------------------

def _compute_irr(cash_flows: list[float]) -> float:
    """Bisection search for IRR. cash_flows[0] is negative (investment)."""
    lo, hi = -0.5, 5.0
    mid = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        npv = sum(cf / (1 + mid) ** t for t, cf in enumerate(cash_flows))
        if abs(npv) < 1e-6:
            break
        if npv > 0:
            lo = mid
        else:
            hi = mid
    return mid


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def _lbo_verdict(irr: float) -> str:
    if irr >= 0.25:
        return "strong return"
    if irr >= 0.18:
        return "acceptable"
    if irr >= 0.10:
        return "marginal"
    return "value-destroying"


def _merger_verdict(accretion_pct: float) -> str:
    if accretion_pct > 0.5:
        return "accretive"
    if accretion_pct < -0.5:
        return "dilutive"
    return "neutral"


# ---------------------------------------------------------------------------
# LBO model
# ---------------------------------------------------------------------------

def run_lbo_model(assumptions: LBOAssumptions) -> LBOResult:
    """Pure synchronous LBO model — no I/O, fully deterministic."""
    warnings: list[str] = []

    total_debt = assumptions.leverage_multiple * assumptions.ebitda
    equity_invested = assumptions.purchase_price - total_debt

    if equity_invested <= 0:
        warnings.append(
            f"Leverage implies negative equity ({equity_invested:,.0f}); "
            "deal is over-levered at entry."
        )
        equity_invested = max(equity_invested, 1.0)

    # Estimate revenue from EBITDA assuming ~25% EBITDA margin
    revenue_est = assumptions.ebitda / 0.25
    # D&A and capex both 5% of revenue, net = 0
    da = revenue_est * 0.05

    debt_balance = total_debt
    amort_per_year = total_debt * assumptions.amortization_pct
    cumulative_debt_paid = 0.0

    years: list[LBOYear] = []
    ebitda = assumptions.ebitda

    # Cash flows for IRR: outflow at t=0, inflow at t=hold_years
    irr_flows: list[float] = [-equity_invested]

    for yr in range(1, assumptions.hold_years + 1):
        ebitda = ebitda * (1.0 + assumptions.ebitda_growth_rate)
        revenue_est = ebitda / 0.25
        da = revenue_est * 0.05
        capex = revenue_est * 0.05  # capex = D&A → net zero maintenance assumption

        interest_expense = debt_balance * assumptions.interest_rate
        ebit = ebitda - da
        net_income = (ebit - interest_expense) * (1.0 - assumptions.tax_rate)

        # Mandatory amortization capped at remaining debt
        actual_amort = min(amort_per_year, debt_balance)
        fcf = net_income + da - capex - actual_amort

        # Any positive FCF beyond mandatory amort applied to additional paydown
        extra_paydown = max(fcf, 0.0)
        debt_balance = max(debt_balance - actual_amort - extra_paydown, 0.0)
        cumulative_debt_paid += actual_amort + extra_paydown

        years.append(
            LBOYear(
                year=yr,
                ebitda=round(ebitda, 2),
                interest_expense=round(interest_expense, 2),
                debt_balance=round(debt_balance, 2),
                free_cash_flow=round(fcf, 2),
                cumulative_debt_paid=round(cumulative_debt_paid, 2),
            )
        )

    exit_ebitda = ebitda
    exit_ev = assumptions.exit_multiple * exit_ebitda
    exit_equity = max(exit_ev - debt_balance, 0.0)

    if exit_equity <= 0:
        warnings.append("Exit equity is zero — debt exceeds exit EV. Total loss scenario.")

    irr_flows.append(exit_equity)
    irr = _compute_irr(irr_flows)
    moic = exit_equity / equity_invested if equity_invested > 0 else 0.0

    # Cash yield: total cash received (exit proceeds only in this template) / invested
    cash_yield = exit_equity / equity_invested if equity_invested > 0 else 0.0

    if assumptions.leverage_multiple > 7.0:
        warnings.append(f"Leverage of {assumptions.leverage_multiple}x is above typical PE limit of 7x.")
    if assumptions.interest_rate > 0.12:
        warnings.append(f"Interest rate {assumptions.interest_rate:.1%} is elevated; stress-test coverage.")

    return LBOResult(
        assumptions=assumptions,
        equity_invested=round(equity_invested, 2),
        total_debt=round(total_debt, 2),
        years=years,
        exit_ebitda=round(exit_ebitda, 2),
        exit_ev=round(exit_ev, 2),
        exit_equity=round(exit_equity, 2),
        moic=round(moic, 3),
        irr=round(irr, 4),
        cash_yield=round(cash_yield, 3),
        verdict=_lbo_verdict(irr),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Merger accretion/dilution model
# ---------------------------------------------------------------------------

def run_merger_model(assumptions: MergerAssumptions) -> MergerResult:
    """Pure synchronous merger accretion/dilution model."""
    warnings: list[str] = []

    deal_value_mm = assumptions.acquisition_price_per_share * assumptions.target_shares_mm
    equity_consideration_mm = deal_value_mm * assumptions.pct_stock
    cash_consideration_mm = deal_value_mm * (1.0 - assumptions.pct_stock)

    new_shares_issued_mm = (
        equity_consideration_mm / assumptions.acquirer_price
        if assumptions.acquirer_price > 0
        else 0.0
    )

    acquirer_ni_mm = assumptions.acquirer_eps * assumptions.acquirer_shares_mm
    target_ni_mm = assumptions.target_eps * assumptions.target_shares_mm

    # After-tax interest cost on cash portion
    interest_cost_mm = (
        cash_consideration_mm
        * assumptions.cost_of_debt
        * (1.0 - assumptions.tax_rate)
    )

    combined_ni_mm = (
        acquirer_ni_mm
        + target_ni_mm
        + assumptions.synergies_after_tax_mm
        - interest_cost_mm
    )
    combined_shares_mm = assumptions.acquirer_shares_mm + new_shares_issued_mm
    combined_eps = combined_ni_mm / combined_shares_mm if combined_shares_mm > 0 else 0.0

    accretion_pct = (
        (combined_eps - assumptions.acquirer_eps) / assumptions.acquirer_eps * 100.0
        if assumptions.acquirer_eps != 0
        else 0.0
    )

    # Premium: acquisition price vs implied market price (target EPS × acquirer P/E)
    acquirer_pe = (
        assumptions.acquirer_price / assumptions.acquirer_eps
        if assumptions.acquirer_eps > 0
        else 0.0
    )
    target_market_price = assumptions.target_eps * acquirer_pe if acquirer_pe > 0 else 0.0
    premium_paid_pct = (
        (assumptions.acquisition_price_per_share / target_market_price - 1.0) * 100.0
        if target_market_price > 0
        else 0.0
    )

    if assumptions.pct_stock < 0.0 or assumptions.pct_stock > 1.0:
        warnings.append("pct_stock must be between 0 and 1.")
    if premium_paid_pct > 50.0:
        warnings.append(f"Premium of {premium_paid_pct:.1f}% is elevated; synergy hurdle is high.")
    if combined_shares_mm <= 0:
        warnings.append("Combined share count is zero or negative; check inputs.")

    return MergerResult(
        assumptions=assumptions,
        deal_value_mm=round(deal_value_mm, 2),
        equity_consideration_mm=round(equity_consideration_mm, 2),
        cash_consideration_mm=round(cash_consideration_mm, 2),
        new_shares_issued_mm=round(new_shares_issued_mm, 4),
        combined_eps=round(combined_eps, 4),
        acquirer_eps=round(assumptions.acquirer_eps, 4),
        accretion_pct=round(accretion_pct, 4),
        verdict=_merger_verdict(accretion_pct),
        premium_paid_pct=round(premium_paid_pct, 2),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Live-data LBO screen
# ---------------------------------------------------------------------------

async def screen_lbo_candidate(ticker: str) -> LBOScreenResult:
    """Fetch live financials for ticker via yfinance, auto-populate LBO assumptions, run model."""
    warnings: list[str] = []

    def _fetch() -> dict[str, Any]:
        import yfinance as yf  # lazy import — heavy optional dep
        return yf.Ticker(ticker).info

    info: dict[str, Any] = await asyncio.to_thread(_fetch)

    enterprise_value: float | None = info.get("enterpriseValue")
    ebitda: float | None = info.get("ebitda")
    entity_name: str = info.get("longName") or info.get("shortName") or ticker

    market_data_used: dict[str, Any] = {
        "enterpriseValue": enterprise_value,
        "ebitda": ebitda,
        "longName": entity_name,
        "forwardPE": info.get("forwardPE"),
        "trailingPE": info.get("trailingPE"),
    }

    if not enterprise_value or enterprise_value <= 0:
        warnings.append("enterpriseValue unavailable from yfinance; using fallback 1 000 000 000.")
        enterprise_value = 1_000_000_000.0

    if not ebitda or ebitda <= 0:
        warnings.append("ebitda unavailable from yfinance; using fallback 100 000 000.")
        ebitda = 100_000_000.0

    entry_multiple = enterprise_value / ebitda

    assumptions = LBOAssumptions(
        purchase_price=enterprise_value,
        ebitda=ebitda,
        ebitda_growth_rate=0.05,
        leverage_multiple=5.0,
        interest_rate=0.085,
        hold_years=5,
        exit_multiple=entry_multiple,
        tax_rate=0.25,
        amortization_pct=0.05,
    )

    if entry_multiple > 20.0:
        warnings.append(
            f"Entry EV/EBITDA of {entry_multiple:.1f}x is very high; LBO is unlikely to pencil."
        )

    result = run_lbo_model(assumptions)

    logger.info(
        "LBO screen complete",
        ticker=ticker,
        entity_name=entity_name,
        irr=result.irr,
        moic=result.moic,
        verdict=result.verdict,
    )

    return LBOScreenResult(
        ticker=ticker.upper(),
        entity_name=entity_name,
        lbo_result=result,
        market_data_used=market_data_used,
        warnings=warnings + result.warnings,
    )
