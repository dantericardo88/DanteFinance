"""Credit analytics — Merton structural model, credit score, CDS spread proxy."""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_RECOVERY_RATE = 0.40
_DEFAULT_RISK_FREE = 0.05
_MERTON_ITERS = 100


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class MertonOutput(BaseModel):
    asset_value: Optional[float] = None
    asset_volatility: Optional[float] = None
    distance_to_default: Optional[float] = None
    probability_of_default: Optional[float] = None  # 0–1
    implied_cds_spread_bps: Optional[float] = None  # basis points
    recovery_rate: float = _RECOVERY_RATE
    converged: bool = False


class CreditFactors(BaseModel):
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None
    current_ratio: Optional[float] = None
    fcf_positive: Optional[bool] = None
    altman_z: Optional[float] = None


class CreditAnalytics(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    credit_score: float          # 1–10 (10 = best)
    credit_tier: str             # "AAA/AA" | "A/BBB" | "BB/B" | "CCC/CC" | "D"
    merton: MertonOutput
    factors: CreditFactors
    equity_vol_annualized: Optional[float] = None
    market_cap: Optional[float] = None
    total_debt: Optional[float] = None
    enterprise_value: Optional[float] = None
    verdict: str                 # "strong" | "adequate" | "speculative" | "distressed"
    warnings: list[str] = Field(default_factory=list)
    as_of: str


class CreditScreenResult(BaseModel):
    tickers: list[str]
    analytics: list[CreditAnalytics]
    avg_credit_score: float
    strongest: Optional[str] = None
    weakest: Optional[str] = None
    distressed_count: int
    as_of: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merton_model(
    E: float,
    D: float,
    sigma_E: float,
    r: float = _DEFAULT_RISK_FREE,
    T: float = 1.0,
) -> MertonOutput:
    """Iterative Merton structural model.  E = market cap, D = total debt."""
    from scipy.stats import norm  # lazy import — only used here

    if D <= 0 or E <= 0 or sigma_E <= 0:
        return MertonOutput()

    # Initial guesses
    sigma_A = sigma_E * E / (E + D)
    A = E + D

    converged = False
    for _ in range(_MERTON_ITERS):
        denom = sigma_A * np.sqrt(T)
        if denom < 1e-12 or A <= 0:
            break
        d1 = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / denom
        d2 = d1 - denom

        n_d1 = norm.cdf(d1)
        if n_d1 < 1e-12:
            break

        # Update volatility and asset value
        sigma_A_new = sigma_E * E / (A * n_d1)
        A_new = E / n_d1 + D * np.exp(-r * T) * norm.cdf(-d2) / n_d1

        if abs(A_new - A) < 0.001 and abs(sigma_A_new - sigma_A) < 0.0001:
            A, sigma_A = A_new, sigma_A_new
            converged = True
            break
        A, sigma_A = A_new, sigma_A_new

    # Final metrics
    denom = sigma_A * np.sqrt(T)
    if denom < 1e-12 or A <= 0:
        return MertonOutput()

    d1 = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / denom
    d2 = d1 - denom
    pd_val = float(norm.cdf(-d2))
    dd = float(d2)
    cds_bps = float(pd_val / (1.0 - _RECOVERY_RATE) * 10_000)

    return MertonOutput(
        asset_value=float(A),
        asset_volatility=float(sigma_A),
        distance_to_default=dd,
        probability_of_default=pd_val,
        implied_cds_spread_bps=cds_bps,
        recovery_rate=_RECOVERY_RATE,
        converged=converged,
    )


def _score_factors(factors: CreditFactors) -> float:
    """Composite credit score 0–10 from individual factor sub-scores."""
    score = 0.0

    # Debt/Equity: max 3 pts
    de = factors.debt_to_equity
    if de is not None:
        if de < 0.3:
            score += 3
        elif de < 0.7:
            score += 2
        elif de < 1.5:
            score += 1
        # else 0

    # Interest coverage: max 3 pts
    ic = factors.interest_coverage
    if ic is not None:
        if ic > 10:
            score += 3
        elif ic > 5:
            score += 2
        elif ic > 2:
            score += 1

    # Current ratio: max 2 pts
    cr = factors.current_ratio
    if cr is not None:
        if cr > 2:
            score += 2
        elif cr > 1:
            score += 1

    # FCF positive: max 1 pt
    if factors.fcf_positive:
        score += 1

    # Altman Z: max 1 pt
    if factors.altman_z is not None and factors.altman_z > 2.99:
        score += 1

    # Clamp to 1–10
    return float(max(1.0, min(10.0, score)))


def _credit_tier(score: float) -> str:
    if score >= 9:
        return "AAA/AA"
    if score >= 7:
        return "A/BBB"
    if score >= 5:
        return "BB/B"
    if score >= 3:
        return "CCC/CC"
    return "D"


def _verdict(score: float) -> str:
    if score >= 7:
        return "strong"
    if score >= 5:
        return "adequate"
    if score >= 3:
        return "speculative"
    return "distressed"


# ---------------------------------------------------------------------------
# yfinance data fetch (runs inside asyncio.to_thread)
# ---------------------------------------------------------------------------


def _fetch_credit_data(ticker: str) -> dict:
    """Blocking yfinance fetch — call via asyncio.to_thread."""
    import yfinance as yf  # lazy import

    result: dict = {
        "info": {}, "sigma_E": None, "market_cap": None,
        "total_debt": None, "warnings": [],
    }

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        result["info"] = info

        # -- Equity volatility from 1-year daily history --
        try:
            hist = t.history(period="1y", auto_adjust=True)
            if hist is not None and not hist.empty and len(hist) > 10:
                returns = hist["Close"].pct_change().dropna()
                result["sigma_E"] = float(returns.std() * np.sqrt(252))
            else:
                result["warnings"].append("Insufficient price history for vol estimate")
        except Exception as exc:
            result["warnings"].append(f"History fetch failed: {exc}")

        # -- Market cap --
        mc = info.get("marketCap")
        result["market_cap"] = float(mc) if mc else None

        # -- Total debt: try balance sheet first, then info --
        total_debt = None
        try:
            balance = t.balance_sheet
            if balance is not None and not balance.empty:
                for row_key in ("Total Liabilities Net Minority Interest", "Total Debt", "Total Liabilities"):
                    if row_key in balance.index:
                        val = balance.loc[row_key].iloc[0]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            total_debt = float(val)
                            break
        except Exception as exc:
            result["warnings"].append(f"Balance sheet parse failed: {exc}")

        if total_debt is None:
            td = info.get("totalDebt")
            total_debt = float(td) if td else None

        result["total_debt"] = total_debt
        if total_debt is None:
            result["warnings"].append("Total debt not found; Merton model skipped")

        # -- Interest expense for coverage ratio --
        interest_exp = None
        try:
            fins = t.financials
            if fins is not None and not fins.empty:
                for row_key in ("Interest Expense", "Interest Expense Non Operating"):
                    if row_key in fins.index:
                        val = fins.loc[row_key].iloc[0]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            interest_exp = abs(float(val))
                            break
        except Exception as exc:
            result["warnings"].append(f"Financials parse failed: {exc}")

        result["interest_expense"] = interest_exp

    except Exception as exc:
        result["warnings"].append(f"yfinance error: {exc}")

    return result


# ---------------------------------------------------------------------------
# Core analytics builder
# ---------------------------------------------------------------------------


def _build_analytics(ticker: str, raw: dict) -> CreditAnalytics:
    info = raw.get("info", {})
    warnings = list(raw.get("warnings", []))
    as_of = date.today().isoformat()

    company_name = info.get("longName") or info.get("shortName")
    market_cap = raw.get("market_cap")
    total_debt = raw.get("total_debt")
    sigma_E = raw.get("sigma_E")

    # Enterprise value
    ev = info.get("enterpriseValue")
    enterprise_value = float(ev) if ev else (
        (market_cap or 0) + (total_debt or 0)
    )

    # -- Merton model --
    merton = MertonOutput()
    if market_cap and total_debt and sigma_E:
        merton = _merton_model(market_cap, total_debt, sigma_E)
        if not merton.converged:
            warnings.append("Merton model did not fully converge")
    else:
        missing = [k for k, v in [("market_cap", market_cap), ("total_debt", total_debt), ("sigma_E", sigma_E)] if not v]
        warnings.append(f"Merton skipped: missing {', '.join(missing)}")

    # -- Credit factors --
    de_raw = info.get("debtToEquity")
    debt_to_equity = float(de_raw) / 100.0 if de_raw is not None else (
        (total_debt / market_cap) if (total_debt and market_cap) else None
    )

    ebitda = info.get("ebitda")
    interest_exp = raw.get("interest_expense") or info.get("totalInterestExpense")
    if ebitda and interest_exp and interest_exp > 0:
        interest_coverage = float(ebitda) / float(interest_exp)
    else:
        interest_coverage = None

    cr_raw = info.get("currentRatio")
    current_ratio = float(cr_raw) if cr_raw is not None else None

    fcf_raw = info.get("freeCashflow")
    fcf_positive = (float(fcf_raw) > 0) if fcf_raw is not None else None

    # Altman Z: approximate from available info fields
    altman_z: Optional[float] = None
    try:
        # Z' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4 (Altman private-firm variant)
        total_assets = info.get("totalAssets")
        total_liab = info.get("totalLiabilities") or total_debt
        retained = info.get("retainedEarnings")
        ebit = info.get("ebit") or ebitda
        if total_assets and total_assets > 0 and total_liab is not None and market_cap:
            x1 = ((info.get("currentAssets", 0) or 0) - (info.get("currentLiabilities", 0) or 0)) / total_assets
            x2 = (retained or 0) / total_assets
            x3 = (ebit or 0) / total_assets
            x4 = market_cap / max(float(total_liab), 1.0)
            altman_z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    except Exception:
        pass  # altman_z stays None

    factors = CreditFactors(
        debt_to_equity=debt_to_equity,
        interest_coverage=interest_coverage,
        current_ratio=current_ratio,
        fcf_positive=fcf_positive,
        altman_z=altman_z,
    )

    credit_score = _score_factors(factors)
    tier = _credit_tier(credit_score)
    verdict = _verdict(credit_score)

    logger.info(
        "credit analytics built",
        ticker=ticker,
        credit_score=credit_score,
        tier=tier,
        pd=merton.probability_of_default,
        dd=merton.distance_to_default,
    )

    return CreditAnalytics(
        ticker=ticker.upper(),
        company_name=company_name,
        credit_score=credit_score,
        credit_tier=tier,
        merton=merton,
        factors=factors,
        equity_vol_annualized=sigma_E,
        market_cap=market_cap,
        total_debt=total_debt,
        enterprise_value=enterprise_value,
        verdict=verdict,
        warnings=warnings,
        as_of=as_of,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def get_credit_analytics(ticker: str) -> CreditAnalytics:
    """Full credit analytics: Merton model, credit score, CDS proxy, tier classification."""
    logger.info("fetching credit analytics", ticker=ticker)
    raw = await asyncio.to_thread(_fetch_credit_data, ticker)
    return _build_analytics(ticker, raw)


async def screen_credit(
    tickers: list[str],
    max_credit_score: float | None = None,
    min_credit_score: float | None = None,
) -> CreditScreenResult:
    """Screen multiple tickers for credit quality. Uses asyncio.gather for parallel fetch."""
    as_of = date.today().isoformat()
    results: list[CreditAnalytics] = await asyncio.gather(
        *[get_credit_analytics(t) for t in tickers]
    )

    # Apply filters
    filtered = [
        a for a in results
        if (max_credit_score is None or a.credit_score <= max_credit_score)
        and (min_credit_score is None or a.credit_score >= min_credit_score)
    ]

    scores = [a.credit_score for a in filtered]
    avg = float(np.mean(scores)) if scores else 0.0
    strongest = max(filtered, key=lambda a: a.credit_score).ticker if filtered else None
    weakest = min(filtered, key=lambda a: a.credit_score).ticker if filtered else None
    distressed = sum(1 for a in filtered if a.verdict == "distressed")

    logger.info(
        "credit screen complete",
        tickers=len(tickers),
        filtered=len(filtered),
        avg_score=avg,
        distressed=distressed,
    )

    return CreditScreenResult(
        tickers=[a.ticker for a in filtered],
        analytics=filtered,
        avg_credit_score=round(avg, 2),
        strongest=strongest,
        weakest=weakest,
        distressed_count=distressed,
        as_of=as_of,
    )
