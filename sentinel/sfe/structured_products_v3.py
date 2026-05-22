"""
MBS/ABS/CLO structured product analytics.

Dimension: dim_040 — Structured products MBS/ABS/CLO (target score: 9)

Pure quantitative analytics — no live data, no network calls.
Implements PSA prepayment model, OAS, WAL, CLO waterfall with OC/IC tests,
CLO equity IRR, and ABS yield/spread analytics.

Public API
----------
psa_cpr(month, speed)                        -> float
smm_from_cpr(cpr)                            -> float
mbs_cash_flows(pool)                         -> List[MBSCashFlow]
mbs_price(pool, discount_rate)               -> float
mbs_wal(pool)                                -> float
clo_equity_irr(waterfall, equity_investment) -> float

MBSPricer
    cash_flows(pool)                         -> List[MBSCashFlow]
    price(pool, discount_rate)               -> float
    wal(pool)                                -> float
    duration(pool, discount_rate)            -> float
    oas(pool, market_price, spot_curve, maturities) -> float
    psa_sensitivity(pool, discount_rate, psa_speeds) -> np.ndarray

CLOAnalytics
    period_cashflows(n_periods)              -> Dict[str, np.ndarray]
    oc_ratio(period)                         -> Dict[str, float]
    equity_irr(equity_investment, n_periods) -> float
    tranche_yield(tranche_name, market_price_pct, n_periods) -> float

ABSPricer
    cash_flows()                             -> np.ndarray
    price(discount_rate)                     -> float
    yield_to_maturity(market_price)          -> float
    spread(market_price, benchmark_rate)     -> float
    wal()                                    -> float
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MBSPool:
    balance: float           # current outstanding principal ($)
    wac: float               # weighted avg coupon rate (annual, decimal e.g. 0.06)
    wam: int                 # weighted avg maturity (months)
    psa_speed: float = 1.0   # 1.0 = 100% PSA, 1.5 = 150% PSA


@dataclass
class MBSCashFlow:
    month: int
    scheduled_interest: float
    scheduled_principal: float
    prepayment: float
    total_cashflow: float
    remaining_balance: float


@dataclass
class CLOTranche:
    name: str
    rating: str              # 'AAA', 'AA', 'A', 'BBB', 'BB', 'Equity'
    par_amount: float
    coupon: float            # annual coupon rate (decimal e.g. 0.015)
    oc_trigger: float = 1.20  # OC test threshold


@dataclass
class CLOWaterfall:
    tranches: List[CLOTranche]
    collateral_balance: float
    collateral_coupon: float     # portfolio weighted avg spread (annual, decimal)
    default_rate: float = 0.02   # annual CDR (conditional default rate)
    recovery_rate: float = 0.40


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def psa_cpr(month: int, speed: float = 1.0) -> float:
    """
    Return the CPR at a given month under a given PSA speed.

    100% PSA ramps linearly from 0% CPR at month 0 to 6% CPR at month 30,
    then stays flat at 6%.  Other speeds scale proportionally.

    Parameters
    ----------
    month : int   calendar month (1-indexed)
    speed : float PSA multiple (1.0 = 100 PSA, 1.5 = 150 PSA)

    Returns
    -------
    float CPR (annual, decimal)
    """
    cpr_plateau = 0.06  # 6% at 100 PSA
    if month <= 0:
        return 0.0
    base_cpr = min(cpr_plateau * month / 30.0, cpr_plateau)
    return base_cpr * speed


def smm_from_cpr(cpr: float) -> float:
    """
    Convert annualized CPR to Single Monthly Mortality (SMM).

    SMM = 1 - (1 - CPR)^(1/12)
    """
    if cpr <= 0.0:
        return 0.0
    if cpr >= 1.0:
        return 1.0
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def _fixed_payment(balance: float, monthly_rate: float, n_months: int) -> float:
    """Standard fixed mortgage payment formula."""
    if monthly_rate == 0.0:
        return balance / n_months
    return balance * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-n_months))


def mbs_cash_flows(pool: MBSPool) -> List[MBSCashFlow]:
    """
    Generate monthly cash flows for an MBS pool using the PSA prepayment model.

    The scheduled payment is computed from the *original* balance and WAM,
    then each month we also apply the PSA prepayment.
    """
    monthly_rate = pool.wac / 12.0
    remaining = pool.balance
    n = pool.wam

    # Fixed scheduled payment for fully amortizing loan (no prepayment)
    fixed_pmt = _fixed_payment(pool.balance, monthly_rate, n)

    cash_flows: List[MBSCashFlow] = []

    for t in range(1, n + 1):
        if remaining <= 0.0:
            # Pool fully paid
            cf = MBSCashFlow(
                month=t,
                scheduled_interest=0.0,
                scheduled_principal=0.0,
                prepayment=0.0,
                total_cashflow=0.0,
                remaining_balance=0.0,
            )
            cash_flows.append(cf)
            continue

        # Scheduled interest and principal (standard amortization)
        sched_interest = remaining * monthly_rate
        sched_principal = min(fixed_pmt - sched_interest, remaining)
        sched_principal = max(sched_principal, 0.0)

        # PSA prepayment applied to balance *after* scheduled principal
        balance_after_sched = remaining - sched_principal
        cpr = psa_cpr(t, pool.psa_speed)
        smm = smm_from_cpr(cpr)
        prepayment = smm * balance_after_sched

        total_principal = sched_principal + prepayment
        total_cf = sched_interest + total_principal

        remaining -= total_principal
        remaining = max(remaining, 0.0)

        cash_flows.append(MBSCashFlow(
            month=t,
            scheduled_interest=sched_interest,
            scheduled_principal=sched_principal,
            prepayment=prepayment,
            total_cashflow=total_cf,
            remaining_balance=remaining,
        ))

    return cash_flows


def mbs_price(pool: MBSPool, discount_rate: float) -> float:
    """
    Price an MBS pool by discounting cash flows at a flat discount rate.

    Uses monthly discounting: df_t = 1 / (1 + r/12)^t
    Returns price as percentage of par (100 = par).
    """
    cfs = mbs_cash_flows(pool)
    monthly_disc = discount_rate / 12.0
    pv = 0.0
    for cf in cfs:
        df = (1.0 + monthly_disc) ** (-cf.month)
        pv += cf.total_cashflow * df
    return pv / pool.balance * 100.0


def mbs_wal(pool: MBSPool) -> float:
    """
    Weighted Average Life of the MBS pool in years.

    WAL = sum(t * principal_t) / (total_principal * 12)
    """
    cfs = mbs_cash_flows(pool)
    total_principal = sum(cf.scheduled_principal + cf.prepayment for cf in cfs)
    if total_principal <= 0.0:
        return 0.0
    wal_months = sum(
        cf.month * (cf.scheduled_principal + cf.prepayment) for cf in cfs
    )
    return wal_months / total_principal / 12.0


def clo_equity_irr(waterfall: CLOWaterfall, equity_investment: float) -> float:
    """
    Compute the IRR on CLO equity cash flows over the CLO life (default 5 periods).

    Thin wrapper around CLOAnalytics.equity_irr().
    """
    analytics = CLOAnalytics(waterfall)
    return analytics.equity_irr(equity_investment=equity_investment, n_periods=5)


# ---------------------------------------------------------------------------
# MBSPricer class
# ---------------------------------------------------------------------------

class MBSPricer:
    """
    Full MBS analytics: cash flows, price, WAL, duration, OAS, PSA sensitivity.
    """

    def cash_flows(self, pool: MBSPool) -> List[MBSCashFlow]:
        """Generate monthly MBS cash flows using PSA prepayment model."""
        return mbs_cash_flows(pool)

    def price(self, pool: MBSPool, discount_rate: float) -> float:
        """
        Price MBS as % of par using flat discount rate (monthly compounding).
        """
        return mbs_price(pool, discount_rate)

    def wal(self, pool: MBSPool) -> float:
        """Weighted Average Life in years."""
        return mbs_wal(pool)

    def duration(self, pool: MBSPool, discount_rate: float) -> float:
        """
        Modified duration in years (Macaulay / (1 + r/12)).

        Uses monthly cash flows and monthly discounting.
        """
        cfs = mbs_cash_flows(pool)
        monthly_disc = discount_rate / 12.0
        pv_total = 0.0
        pv_t = 0.0
        for cf in cfs:
            t = cf.month
            df = (1.0 + monthly_disc) ** (-t)
            pv = cf.total_cashflow * df
            pv_total += pv
            pv_t += (t / 12.0) * pv  # time in years

        if pv_total <= 0.0:
            return 0.0
        macaulay = pv_t / pv_total
        modified = macaulay / (1.0 + monthly_disc)
        return modified

    def oas(
        self,
        pool: MBSPool,
        market_price: float,
        spot_curve: np.ndarray,
        maturities: np.ndarray,
    ) -> float:
        """
        Option-Adjusted Spread given market price and a spot rate curve.

        The spot curve uses semiannual compounding:
          Price = sum(CF_t / (1 + spot(t)/2 + OAS/2)^(2*t))
        where t is in years.

        Parameters
        ----------
        pool         : MBSPool
        market_price : float  market price as % of par (e.g. 98.5)
        spot_curve   : np.ndarray  spot rates (annual, decimal) at given maturities
        maturities   : np.ndarray  maturities in years (same length as spot_curve)

        Returns
        -------
        float  OAS in decimal (e.g. 0.0050 = 50 bps)
        """
        cfs = mbs_cash_flows(pool)
        target_pv = market_price / 100.0 * pool.balance

        def price_given_oas(oas_val: float) -> float:
            pv = 0.0
            for cf in cfs:
                t_years = cf.month / 12.0
                # interpolate spot rate
                s = float(np.interp(t_years, maturities, spot_curve))
                # semiannual compounding
                df = (1.0 + s / 2.0 + oas_val / 2.0) ** (-2.0 * t_years)
                pv += cf.total_cashflow * df
            return pv - target_pv

        # Search OAS in range [-500bps, +2000bps]
        try:
            oas_val = brentq(price_given_oas, -0.05, 0.20, xtol=1e-8, maxiter=200)
        except ValueError:
            # Fallback: return simple yield spread
            ytm = self._flat_ytm(pool, market_price)
            benchmark = float(np.interp(self.wal(pool), maturities, spot_curve))
            oas_val = ytm - benchmark
        return oas_val

    def _flat_ytm(self, pool: MBSPool, market_price_pct: float) -> float:
        """Internal helper: flat YTM from monthly cash flows."""
        cfs = mbs_cash_flows(pool)
        target_pv = market_price_pct / 100.0 * pool.balance

        def obj(r_monthly: float) -> float:
            pv = sum(
                cf.total_cashflow / (1.0 + r_monthly) ** cf.month
                for cf in cfs
            )
            return pv - target_pv

        try:
            r_m = brentq(obj, 1e-8, 0.10, xtol=1e-10, maxiter=300)
        except ValueError:
            r_m = pool.wac / 12.0
        return r_m * 12.0

    def psa_sensitivity(
        self,
        pool: MBSPool,
        discount_rate: float,
        psa_speeds: np.ndarray,
    ) -> np.ndarray:
        """
        Compute WAL at each PSA speed.

        Parameters
        ----------
        pool          : MBSPool (base pool; psa_speed will be overridden)
        discount_rate : float   (unused here, kept for API symmetry)
        psa_speeds    : np.ndarray  array of PSA multiples (e.g. [0.5, 1.0, 1.5, 2.0])

        Returns
        -------
        np.ndarray of WAL values (years) corresponding to each PSA speed
        """
        wals = np.zeros(len(psa_speeds))
        for i, speed in enumerate(psa_speeds):
            new_pool = MBSPool(
                balance=pool.balance,
                wac=pool.wac,
                wam=pool.wam,
                psa_speed=float(speed),
            )
            wals[i] = mbs_wal(new_pool)
        return wals


# ---------------------------------------------------------------------------
# CLOAnalytics class
# ---------------------------------------------------------------------------

class CLOAnalytics:
    """
    CLO waterfall analytics: period cash flows, OC ratio, equity IRR,
    tranche yield.

    Simplified annual-period model:
      - Collateral generates interest each period.
      - CDR reduces the collateral balance (defaults), recovery is received.
      - Waterfall distributes interest (and principal recovery) in priority order.
      - OC tests divert interest to pay down senior tranches if breached.
    """

    def __init__(self, waterfall: CLOWaterfall) -> None:
        self.wf = waterfall

    def _senior_tranches(self) -> List[CLOTranche]:
        """Non-equity tranches ordered senior-first."""
        return [t for t in self.wf.tranches if t.rating != "Equity"]

    def _equity_tranche(self) -> Optional[CLOTranche]:
        for t in self.wf.tranches:
            if t.rating == "Equity":
                return t
        return None

    def period_cashflows(self, n_periods: int = 5) -> Dict[str, np.ndarray]:
        """
        Simulate the CLO waterfall over n_periods annual periods.

        Returns a dict mapping tranche name to arrays of length n_periods:
          {
            "AAA_interest": np.ndarray,
            "AAA_principal": np.ndarray,
            ... (per tranche)
          }
        """
        wf = self.wf
        senior = self._senior_tranches()
        equity = self._equity_tranche()

        # Outstanding balances (mutable)
        tranche_balance = {t.name: t.par_amount for t in wf.tranches}
        collateral_bal = wf.collateral_balance

        result: Dict[str, List[float]] = {}
        for t in wf.tranches:
            result[f"{t.name}_interest"] = []
            result[f"{t.name}_principal"] = []

        for period in range(n_periods):
            # Defaults this period
            defaults = collateral_bal * wf.default_rate
            recovery = defaults * wf.recovery_rate
            collateral_bal = collateral_bal - defaults

            # Interest income from collateral
            interest_income = collateral_bal * wf.collateral_coupon + recovery

            # OC test: collateral par / AAA par outstanding
            # (simplified: check against the senior-most tranche)
            aaa_tranche = next((t for t in senior if t.rating == "AAA"), None)
            oc_test_passes = True
            if aaa_tranche is not None:
                aaa_bal = tranche_balance[aaa_tranche.name]
                if aaa_bal > 0:
                    oc = collateral_bal / aaa_bal
                    oc_test_passes = oc >= aaa_tranche.oc_trigger

            # If OC fails, divert interest to pay down AAA
            diverted_to_aaa = 0.0
            if not oc_test_passes and aaa_tranche is not None:
                # Divert excess interest after paying AAA coupon
                aaa_coupon_due = tranche_balance[aaa_tranche.name] * aaa_tranche.coupon
                diverted_to_aaa = max(interest_income - aaa_coupon_due, 0.0)
                # Cap at outstanding AAA balance
                diverted_to_aaa = min(diverted_to_aaa, tranche_balance[aaa_tranche.name])
                interest_income -= diverted_to_aaa
                tranche_balance[aaa_tranche.name] -= diverted_to_aaa

            # Waterfall: pay senior tranches in order
            remaining_interest = interest_income
            # Principal from final period paydown
            principal_available = collateral_bal if period == n_periods - 1 else 0.0

            for t in senior:
                bal = tranche_balance[t.name]
                if bal <= 0.0:
                    result[f"{t.name}_interest"].append(0.0)
                    result[f"{t.name}_principal"].append(0.0)
                    continue

                coupon_due = bal * t.coupon
                interest_paid = min(coupon_due, remaining_interest)
                remaining_interest = max(remaining_interest - interest_paid, 0.0)

                # Principal repayment (only at maturity in this simplified model)
                principal_paid = min(bal, principal_available)
                principal_available -= principal_paid
                tranche_balance[t.name] -= principal_paid

                # Include diverted AAA principal in AAA principal received
                if t.name == (aaa_tranche.name if aaa_tranche else ""):
                    principal_paid += diverted_to_aaa

                result[f"{t.name}_interest"].append(interest_paid)
                result[f"{t.name}_principal"].append(principal_paid)

            # Equity gets residual
            if equity is not None:
                eq_interest = remaining_interest
                eq_principal = principal_available
                result[f"{equity.name}_interest"].append(eq_interest)
                result[f"{equity.name}_principal"].append(eq_principal)

        # Convert to numpy arrays
        return {k: np.array(v) for k, v in result.items()}

    def oc_ratio(self, period: int = 0) -> Dict[str, float]:
        """
        OC (Overcollateralization) ratio for each tranche at a given period.

        OC_tranche = collateral_par_at_period / tranche_par_outstanding_at_period

        For period=0 this is simply collateral_balance / tranche_par_amount
        (before any defaults).
        """
        wf = self.wf
        # Collateral balance after defaults
        collateral_bal = wf.collateral_balance * (1.0 - wf.default_rate) ** period

        oc: Dict[str, float] = {}
        for t in wf.tranches:
            if t.rating == "Equity":
                continue
            if t.par_amount > 0:
                oc[t.name] = collateral_bal / t.par_amount
            else:
                oc[t.name] = float("inf")
        return oc

    def equity_irr(
        self,
        equity_investment: float,
        n_periods: int = 5,
    ) -> float:
        """
        Compute equity IRR.

        Cash flows: -equity_investment at t=0, then annual equity cash flows.
        IRR: solve NPV(r) = 0.
        """
        cfs = self.period_cashflows(n_periods=n_periods)
        equity = self._equity_tranche()
        if equity is None:
            return float("nan")

        eq_int = cfs.get(f"{equity.name}_interest", np.zeros(n_periods))
        eq_pri = cfs.get(f"{equity.name}_principal", np.zeros(n_periods))
        equity_cfs = eq_int + eq_pri  # annual cash flows (periods 1..n)

        # t=0: -equity_investment; t=1..n: equity_cfs
        def npv(r: float) -> float:
            pv = -equity_investment
            for t, cf in enumerate(equity_cfs, start=1):
                pv += cf / (1.0 + r) ** t
            return pv

        # Bracket search
        try:
            irr = brentq(npv, -0.999, 10.0, xtol=1e-8, maxiter=500)
        except ValueError:
            # If no sign change, return approximate
            irr = float("nan")
        return irr

    def tranche_yield(
        self,
        tranche_name: str,
        market_price_pct: float,
        n_periods: int = 5,
    ) -> float:
        """
        Compute yield to maturity for a given tranche at a market price.

        market_price_pct: % of par (e.g. 95.0 = 95% of par)
        """
        tranche = next(
            (t for t in self.wf.tranches if t.name == tranche_name), None
        )
        if tranche is None:
            raise ValueError(f"Tranche '{tranche_name}' not found")

        purchase_price = market_price_pct / 100.0 * tranche.par_amount
        cfs = self.period_cashflows(n_periods=n_periods)

        int_cfs = cfs.get(f"{tranche_name}_interest", np.zeros(n_periods))
        pri_cfs = cfs.get(f"{tranche_name}_principal", np.zeros(n_periods))
        annual_cfs = int_cfs + pri_cfs

        def npv(r: float) -> float:
            pv = -purchase_price
            for t, cf in enumerate(annual_cfs, start=1):
                pv += cf / (1.0 + r) ** t
            return pv

        try:
            ytm = brentq(npv, -0.999, 10.0, xtol=1e-8, maxiter=500)
        except ValueError:
            ytm = float("nan")
        return ytm


# ---------------------------------------------------------------------------
# ABSPricer class
# ---------------------------------------------------------------------------

class ABSPricer:
    """
    ABS (auto loan / student loan) analytics.

    Standard fully-amortizing structure with constant CPR prepayment.
    """

    def __init__(
        self,
        balance: float,
        coupon: float,
        term_months: int,
        cpr: float = 0.06,
    ) -> None:
        """
        Parameters
        ----------
        balance      : float  initial pool balance ($)
        coupon       : float  annual coupon rate (decimal, e.g. 0.05)
        term_months  : int    original term in months
        cpr          : float  constant annual prepayment rate (decimal)
        """
        self.balance = balance
        self.coupon = coupon
        self.term_months = term_months
        self.cpr = cpr

    def cash_flows(self) -> np.ndarray:
        """
        Generate monthly total cash flows (interest + principal + prepayment).

        Returns np.ndarray of shape (term_months,) with total cash flow per month.
        """
        monthly_rate = self.coupon / 12.0
        smm = smm_from_cpr(self.cpr)
        remaining = self.balance
        fixed_pmt = _fixed_payment(self.balance, monthly_rate, self.term_months)

        cfs = np.zeros(self.term_months)
        for t in range(self.term_months):
            if remaining <= 0.0:
                break
            interest = remaining * monthly_rate
            sched_principal = min(fixed_pmt - interest, remaining)
            sched_principal = max(sched_principal, 0.0)
            balance_after_sched = remaining - sched_principal
            prepayment = smm * balance_after_sched
            total = interest + sched_principal + prepayment
            cfs[t] = total
            remaining -= (sched_principal + prepayment)
            remaining = max(remaining, 0.0)

        return cfs

    def price(self, discount_rate: float) -> float:
        """
        Price ABS as % of par (100 = par) using flat monthly discount rate.
        """
        cfs = self.cash_flows()
        monthly_disc = discount_rate / 12.0
        pv = 0.0
        for t, cf in enumerate(cfs, start=1):
            pv += cf / (1.0 + monthly_disc) ** t
        return pv / self.balance * 100.0

    def yield_to_maturity(self, market_price: float) -> float:
        """
        Compute YTM given market price as % of par.

        Returns annual YTM (decimal).
        """
        cfs = self.cash_flows()
        target_pv = market_price / 100.0 * self.balance

        def obj(r_monthly: float) -> float:
            pv = sum(
                cf / (1.0 + r_monthly) ** (t + 1)
                for t, cf in enumerate(cfs)
            )
            return pv - target_pv

        try:
            r_m = brentq(obj, 1e-8, 0.10, xtol=1e-10, maxiter=300)
        except ValueError:
            r_m = self.coupon / 12.0
        return r_m * 12.0

    def spread(self, market_price: float, benchmark_rate: float) -> float:
        """
        Spread over benchmark (e.g. swap rate) at market price.

        spread = YTM - benchmark_rate
        """
        return self.yield_to_maturity(market_price) - benchmark_rate

    def wal(self) -> float:
        """
        Weighted Average Life in years.

        WAL = sum(t * principal_t) / (total_principal * 12)
        """
        monthly_rate = self.coupon / 12.0
        smm = smm_from_cpr(self.cpr)
        remaining = self.balance
        fixed_pmt = _fixed_payment(self.balance, monthly_rate, self.term_months)

        total_principal = 0.0
        wal_months = 0.0
        for t in range(1, self.term_months + 1):
            if remaining <= 0.0:
                break
            interest = remaining * monthly_rate
            sched_principal = min(fixed_pmt - interest, remaining)
            sched_principal = max(sched_principal, 0.0)
            balance_after_sched = remaining - sched_principal
            prepayment = smm * balance_after_sched
            principal = sched_principal + prepayment
            total_principal += principal
            wal_months += t * principal
            remaining -= principal
            remaining = max(remaining, 0.0)

        if total_principal <= 0.0:
            return 0.0
        return wal_months / total_principal / 12.0
