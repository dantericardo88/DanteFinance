"""
sentinel/spm/exec_compensation_v3.py
dim_126: Executive Compensation Benchmarking (score target: 9)

Implements CEO pay-ratio analysis, peer-group benchmarking, pay-for-performance
alignment, Say-on-Pay (SOP) risk scoring, and equity dilution analytics.

Key mathematics
---------------
CEO Pay Ratio (SEC Reg S-K §953(b)):
    pay_ratio = total_CEO_compensation / median_employee_pay

Pay-for-performance alignment:
    r_pay_perf  = Pearson correlation of (CEO pay growth %, TSR %) over N years
    excess_pay  = CEO_pay - (b0 + b1*TSR + b2*EBITDA_growth)  -- OLS residual

Peer benchmarking:
    percentile  = rank(company_pay) / n_peers * 100
    pay_premium = (company_pay / peer_median - 1) * 100   [%]

Say-on-Pay risk score [0, 1]:
    sop_risk = 0.25 * excess_pay_z
             + 0.25 * high_CEO_ratio
             + 0.20 * poor_tsr
             + 0.15 * equity_concentration
             + 0.15 * lack_of_disclosure
    ISS/Glass Lewis proxy: score > 0.6 → "AGAINST" recommendation likely.

Equity dilution:
    dilution = (options + restricted) / (shares_outstanding + options + restricted)
    overhang  = (options + unvested_equity) / diluted_shares
    burn_rate = new_grants / shares_outstanding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


# ===========================================================================
# Data structures
# ===========================================================================


@dataclass
class ExecComp:
    """One year of executive compensation detail."""

    name: str
    role: str           # 'CEO', 'CFO', 'COO', …
    year: int
    base_salary: float
    annual_bonus: float
    stock_awards: float
    option_awards: float
    other: float = 0.0

    @property
    def total(self) -> float:
        """Total compensation = base + bonus + stock + options + other."""
        return (
            self.base_salary
            + self.annual_bonus
            + self.stock_awards
            + self.option_awards
            + self.other
        )

    @property
    def equity_fraction(self) -> float:
        """Share of total comp delivered as equity (stock + options)."""
        t = self.total
        if t == 0:
            return 0.0
        return (self.stock_awards + self.option_awards) / t

    @property
    def at_risk_fraction(self) -> float:
        """Share of total comp that is variable/at-risk (bonus + equity)."""
        t = self.total
        if t == 0:
            return 0.0
        return (self.annual_bonus + self.stock_awards + self.option_awards) / t


@dataclass
class PeerGroup:
    """Compensation and performance data for a peer group."""

    companies: List[str]
    ceo_pays: List[float]           # total CEO comp per company ($)
    tsr_5yr: List[float]            # 5-year total shareholder return (%)
    ebitda_growth: List[float]      # annualised EBITDA growth (%)
    median_employee_pay: float = 50_000.0   # median employee compensation ($)


@dataclass
class CompBenchmark:
    """Full benchmarking result for a single executive."""

    pay_percentile: float           # 0–100
    pay_premium_pct: float          # % above (+) or below (–) peer median
    pay_performance_r: float        # Pearson r of pay series vs TSR series
    excess_pay: float               # OLS residual vs peers ($)
    say_on_pay_risk: float          # 0–1
    ceo_pay_ratio: float            # CEO total / median employee pay
    sop_recommendation: str         # 'FOR' or 'AGAINST'


# ===========================================================================
# Compensation Benchmarker
# ===========================================================================


class CompensationBenchmarker:
    """
    Benchmarks a single executive's compensation against a peer group.

    Parameters
    ----------
    company_comp : ExecComp
        The executive's compensation for the evaluation year.
    peers : PeerGroup
        Peer-group data including pay, TSR, and EBITDA growth arrays.
    """

    def __init__(self, company_comp: ExecComp, peers: PeerGroup) -> None:
        self.comp = company_comp
        self.peers = peers
        self._peer_pays = np.array(peers.ceo_pays, dtype=float)
        self._peer_tsr = np.array(peers.tsr_5yr, dtype=float)
        self._peer_ebitda = np.array(peers.ebitda_growth, dtype=float)

    def percentile(self) -> float:
        """
        Percentile rank of company pay within the peer group.

        Uses scipy.stats.percentileofscore with 'rank' interpolation
        so ties are averaged.  Returns value in [0, 100].
        """
        return float(
            stats.percentileofscore(self._peer_pays, self.comp.total, kind="rank")
        )

    def pay_premium(self) -> float:
        """
        Percentage premium (+) or discount (–) vs peer median.

        pay_premium = (company_pay / peer_median - 1) * 100
        """
        peer_median = float(np.median(self._peer_pays))
        if peer_median == 0:
            return 0.0
        return (self.comp.total / peer_median - 1.0) * 100.0

    def pay_for_performance(self, company_tsr_5yr: float) -> dict:
        """
        Analyse pay-for-performance alignment.

        We run an OLS regression of peer CEO pay on TSR (and EBITDA growth
        if variation exists) to determine whether the company's CEO is
        over- or under-paid given performance.

        Returns
        -------
        dict with keys:
            r_squared       — R² of pay vs TSR across peer group
            excess_pay      — OLS residual ($) for this company
            excess_pay_pct  — excess pay as % of predicted pay
            aligned         — True if |excess_pay| < 20% of predicted pay
        """
        tsr = self._peer_tsr
        pays = self._peer_pays

        # Pearson r between peer pay and peer TSR
        if np.std(tsr) < 1e-10 or np.std(pays) < 1e-10:
            r = 0.0
        else:
            r, _ = stats.pearsonr(tsr, pays)

        r_sq = r ** 2

        # OLS: pay ~ b0 + b1*TSR + b2*EBITDA_growth
        # Build design matrix with peer data
        X = np.column_stack(
            [np.ones(len(pays)), tsr, self._peer_ebitda]
        )
        # Handle rank-deficient designs
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, pays, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.array([float(np.mean(pays)), 0.0, 0.0])

        b0, b1, b2 = coeffs

        # Predict pay for this company given its TSR and assumed avg EBITDA
        ebitda_proxy = float(np.mean(self._peer_ebitda))
        predicted_pay = b0 + b1 * company_tsr_5yr + b2 * ebitda_proxy
        excess = self.comp.total - predicted_pay

        if predicted_pay != 0:
            excess_pct = excess / abs(predicted_pay) * 100.0
        else:
            excess_pct = 0.0

        aligned = bool(abs(excess_pct) < 20.0)

        return {
            "r_squared": r_sq,
            "excess_pay": excess,
            "excess_pay_pct": excess_pct,
            "aligned": aligned,
            "predicted_pay": predicted_pay,
            "pearson_r": r,
        }

    def say_on_pay_risk(self, company_tsr_5yr: float) -> float:
        """
        Estimate Say-on-Pay (SOP) risk score in [0, 1].

        Delegates to the module-level function after computing the necessary
        z-scores relative to the peer group.
        """
        # Excess pay z-score within peer distribution
        peer_std = float(np.std(self._peer_pays))
        peer_mean = float(np.mean(self._peer_pays))
        excess_pay_z = (
            (self.comp.total - peer_mean) / peer_std if peer_std > 0 else 0.0
        )

        # TSR z-score — negative z means poor TSR relative to peers
        tsr_std = float(np.std(self._peer_tsr))
        tsr_mean = float(np.mean(self._peer_tsr))
        # We convert to a "poor_tsr" signal: inverted z (worse TSR → higher signal)
        tsr_z = (
            -(company_tsr_5yr - tsr_mean) / tsr_std if tsr_std > 0 else 0.0
        )

        # Pay-ratio z-score
        ratio = ceo_pay_ratio(self.comp.total, self.peers.median_employee_pay)
        peer_ratios = self._peer_pays / self.peers.median_employee_pay
        ratio_std = float(np.std(peer_ratios))
        ratio_mean = float(np.mean(peer_ratios))
        ratio_z = (ratio - ratio_mean) / ratio_std if ratio_std > 0 else 0.0

        return say_on_pay_risk(excess_pay_z, tsr_z, ratio_z)

    def full_benchmark(self, company_tsr_5yr: float) -> CompBenchmark:
        """Run all benchmarking analyses and return a CompBenchmark."""
        pf = self.pay_for_performance(company_tsr_5yr)
        sop = self.say_on_pay_risk(company_tsr_5yr)
        ratio = ceo_pay_ratio(self.comp.total, self.peers.median_employee_pay)
        pctile = self.percentile()
        premium = self.pay_premium()

        # Pearson r of pay vs TSR at the peer level
        if np.std(self._peer_tsr) < 1e-10 or np.std(self._peer_pays) < 1e-10:
            r = 0.0
        else:
            r, _ = stats.pearsonr(self._peer_tsr, self._peer_pays)

        return CompBenchmark(
            pay_percentile=pctile,
            pay_premium_pct=premium,
            pay_performance_r=r,
            excess_pay=pf["excess_pay"],
            say_on_pay_risk=sop,
            ceo_pay_ratio=ratio,
            sop_recommendation="AGAINST" if sop > 0.6 else "FOR",
        )


# ===========================================================================
# Equity Dilution Analyzer
# ===========================================================================


class EquityDilutionAnalyzer:
    """Utility class for equity-related dilution metrics."""

    def dilution(
        self,
        options: float,
        restricted: float,
        shares_outstanding: float,
    ) -> float:
        """
        Basic dilution from equity awards.

        dilution = (options + restricted) / (shares_outstanding + options + restricted)
        """
        numerator = options + restricted
        denominator = shares_outstanding + options + restricted
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def overhang(self, unvested: float, diluted_shares: float) -> float:
        """
        Equity overhang.

        overhang = unvested_equity / diluted_shares
        """
        if diluted_shares == 0:
            return 0.0
        return unvested / diluted_shares

    def burn_rate(self, new_grants: float, shares_outstanding: float) -> float:
        """
        Annual equity burn rate.

        burn_rate = new_grants / shares_outstanding
        """
        if shares_outstanding == 0:
            return 0.0
        return new_grants / shares_outstanding


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def ceo_pay_ratio(ceo_total: float, median_employee: float) -> float:
    """
    CEO Pay Ratio as defined by SEC Regulation S-K §953(b).

    pay_ratio = total_CEO_compensation / median_employee_pay
    """
    if median_employee == 0:
        return float("inf")
    return ceo_total / median_employee


def pay_percentile(company_pay: float, peer_pays: List[float]) -> float:
    """
    Percentile rank of company_pay within the peer_pays distribution.

    Uses 'rank' interpolation so the return value is in [0, 100].
    """
    if not peer_pays:
        return 0.0
    return float(stats.percentileofscore(peer_pays, company_pay, kind="rank"))


def pay_performance_correlation(
    pay_series: np.ndarray, tsr_series: np.ndarray
) -> float:
    """
    Pearson correlation between a pay time series and a TSR time series.

    Returns 0.0 if either series has zero variance.
    """
    pay = np.asarray(pay_series, dtype=float)
    tsr = np.asarray(tsr_series, dtype=float)
    if pay.std() < 1e-12 or tsr.std() < 1e-12:
        return 0.0
    r, _ = stats.pearsonr(pay, tsr)
    return float(r)


def say_on_pay_risk(
    excess_pay_z: float,
    tsr_z: float,
    pay_ratio_z: float,
    equity_concentration: float = 0.5,
    lack_of_disclosure: float = 0.5,
) -> float:
    """
    Say-on-Pay risk score in [0, 1].

    Modelled after ISS/Glass Lewis proxy advisory methodology:

        sop_risk = 0.25 * excess_pay_signal
                 + 0.25 * high_CEO_ratio_signal
                 + 0.20 * poor_tsr_signal
                 + 0.15 * equity_concentration
                 + 0.15 * lack_of_disclosure

    z-scores are sigmoid-transformed to [0, 1]:
        signal(z) = 1 / (1 + exp(-z))

    Parameters
    ----------
    excess_pay_z      : z-score of CEO pay vs peer distribution (high → risky)
    tsr_z             : poor-TSR signal z-score (high → poor relative TSR → risky)
    pay_ratio_z       : z-score of CEO pay ratio vs peers (high → risky)
    equity_concentration : share of comp in equity [0,1]; default 0.5 (neutral)
    lack_of_disclosure   : disclosure quality penalty [0,1]; default 0.5 (neutral)
    """

    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-float(z)))

    excess_pay_signal = _sigmoid(excess_pay_z)
    high_ratio_signal = _sigmoid(pay_ratio_z)
    poor_tsr_signal = _sigmoid(tsr_z)

    score = (
        0.25 * excess_pay_signal
        + 0.25 * high_ratio_signal
        + 0.20 * poor_tsr_signal
        + 0.15 * float(np.clip(equity_concentration, 0.0, 1.0))
        + 0.15 * float(np.clip(lack_of_disclosure, 0.0, 1.0))
    )

    return float(np.clip(score, 0.0, 1.0))
