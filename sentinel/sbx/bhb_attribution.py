"""
Brinson-Hood-Beebower (BHB) attribution, Brinsson-Fachler, multi-period geometric
linking, Fama-French factor attribution, transaction cost attribution, and reporting —
dim_078 (target score 9+).

Classes
-------
BHBAttribution
    Classic 1986 BHB sector attribution: allocation, selection, interaction.

BrinssonFachlerAttribution
    1985 corrected allocation model — no interaction term; allocation + selection = active.

MultiPeriodAttribution
    Carino logarithmic linking for geometric multi-period chaining.

FactorAttribution
    FF5 factor regression on excess active return; style allocation.

TransactionCostAttribution
    Implementation shortfall, VWAP slippage, TCA drag computation.

AttributionReport
    Aggregated summary, table formatting, Excel-serialisable dict.

FastAPI router
--------------
attribution_router — mounted at /api/attribution
"""
from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from scipy import stats

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_HEADERS = {"User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"}
_TIMEOUT = 20.0

# GICS sector ETF proxies (11 sectors)
SECTOR_ETFS: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

# Fama-French 5-factor column map (after downloading Ken French's CSV)
FF5_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _period_return(prices: pd.Series) -> float:
    """Single-period total return from a price series."""
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0
    return float(prices.iloc[-1] / prices.iloc[0] - 1)


def _fetch_prices_sync(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Download adjusted close prices via yfinance (synchronous, for executor)."""
    if not tickers:
        return pd.DataFrame()
    unique = list(set(tickers))
    try:
        raw = yf.download(
            unique,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("yfinance_download_failed", error=str(exc))
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(unique)]
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw
        if len(unique) == 1:
            closes = closes.rename(columns={"Close": unique[0]})

    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = closes.columns.droplevel(0)

    closes.index = pd.to_datetime(closes.index)
    return closes.sort_index()


async def _fetch_prices(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Async wrapper around yfinance price download."""
    return await asyncio.get_event_loop().run_in_executor(
        None, _fetch_prices_sync, tickers, start, end
    )


async def _fetch_ff5_factors(start: date, end: date) -> Optional[pd.DataFrame]:
    """Download Ken French FF5 daily factors. Returns None on failure."""
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True
        ) as client:
            resp = await client.get(FF5_DAILY_URL)
            resp.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw_text = zf.read(csv_name).decode("utf-8", errors="replace")

        lines = raw_text.splitlines()
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                data_start = i
                break

        data_text = "\n".join(lines[data_start:])
        df = pd.read_csv(io.StringIO(data_text), index_col=0)
        df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
        df = df.dropna(how="all")
        df.columns = [c.strip() for c in df.columns]
        df = df / 100.0  # percent → decimal
        df = df.rename(columns={"Mkt-RF": "MKT_RF"})

        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        filtered = df.loc[mask]
        return filtered if not filtered.empty else None

    except Exception as exc:
        logger.warning("ff5_factors_unavailable", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# BHBAttribution
# ---------------------------------------------------------------------------

class BHBAttribution:
    """Classic Brinson-Hood-Beebower (1986) sector attribution.

    Each sector contributes three effects:
      Allocation  = (w_p - w_b) × (r_b_sector - r_b_total)
      Selection   = w_b × (r_p_sector - r_b_sector)
      Interaction = (w_p - w_b) × (r_p_sector - r_b_sector)

    Sum of allocation + selection + interaction = active return.
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        sector_map: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        portfolio_weights : pd.Series
            Portfolio sector weights (index = sector names, must sum to ~1).
        benchmark_weights : pd.Series
            Benchmark sector weights (index = sector names, must sum to ~1).
        portfolio_returns : pd.Series
            Portfolio return per sector (same index).
        benchmark_returns : pd.Series
            Benchmark return per sector (same index).
        sector_map : dict, optional
            Mapping of ticker → sector for aggregation. Not used in direct mode.
        """
        self._wp = portfolio_weights.copy()
        self._wb = benchmark_weights.copy()
        self._rp = portfolio_returns.copy()
        self._rb = benchmark_returns.copy()
        self._sector_map = sector_map or {}

        # Align indices across all four series
        common = (
            self._wp.index
            .intersection(self._wb.index)
            .intersection(self._rp.index)
            .intersection(self._rb.index)
        )
        self._wp = self._wp.loc[common]
        self._wb = self._wb.loc[common]
        self._rp = self._rp.loc[common]
        self._rb = self._rb.loc[common]

        # Total benchmark return (weighted average of sector benchmark returns)
        self._rB: float = float((self._wb * self._rb).sum())

    def compute_allocation_effect(self) -> pd.Series:
        """Allocation effect per sector: (w_p - w_b) × (r_b_sector - r_b_total)."""
        alloc = (self._wp - self._wb) * (self._rb - self._rB)
        alloc.name = "allocation_effect"
        return alloc

    def compute_selection_effect(self) -> pd.Series:
        """Selection effect per sector: w_b × (r_p_sector - r_b_sector)."""
        sel = self._wb * (self._rp - self._rb)
        sel.name = "selection_effect"
        return sel

    def compute_interaction_effect(self) -> pd.Series:
        """Interaction effect per sector: (w_p - w_b) × (r_p_sector - r_b_sector)."""
        inter = (self._wp - self._wb) * (self._rp - self._rb)
        inter.name = "interaction_effect"
        return inter

    def compute_total_active_return(self) -> float:
        """Portfolio total return minus benchmark total return."""
        port_ret = float((self._wp * self._rp).sum())
        return port_ret - self._rB

    def compute_full_attribution(self) -> pd.DataFrame:
        """Full attribution table: allocation, selection, interaction per sector + total row."""
        alloc = self.compute_allocation_effect()
        sel = self.compute_selection_effect()
        inter = self.compute_interaction_effect()

        df = pd.DataFrame({
            "portfolio_weight": self._wp,
            "benchmark_weight": self._wb,
            "portfolio_return": self._rp,
            "benchmark_return": self._rb,
            "allocation_effect": alloc,
            "selection_effect": sel,
            "interaction_effect": inter,
        })
        df["total_effect"] = df["allocation_effect"] + df["selection_effect"] + df["interaction_effect"]

        # Add TOTAL row
        total_row = pd.DataFrame(
            [{
                "portfolio_weight": df["portfolio_weight"].sum(),
                "benchmark_weight": df["benchmark_weight"].sum(),
                "portfolio_return": float((self._wp * self._rp).sum()),
                "benchmark_return": self._rB,
                "allocation_effect": df["allocation_effect"].sum(),
                "selection_effect": df["selection_effect"].sum(),
                "interaction_effect": df["interaction_effect"].sum(),
                "total_effect": df["total_effect"].sum(),
            }],
            index=["TOTAL"],
        )
        return pd.concat([df, total_row])

    def verify_attribution(self, tolerance: float = 1e-6) -> bool:
        """Verify that sum(allocation + selection + interaction) == active_return."""
        df = self.compute_full_attribution()
        explained = float(df.loc["TOTAL", "total_effect"])
        active = self.compute_total_active_return()
        residual = abs(explained - active)
        logger.info(
            "bhb_attribution.verify",
            active_return=round(active, 8),
            explained=round(explained, 8),
            residual=round(residual, 10),
            passed=residual < tolerance,
        )
        return residual < tolerance


# ---------------------------------------------------------------------------
# BrinssonFachlerAttribution
# ---------------------------------------------------------------------------

class BrinssonFachlerAttribution:
    """Brinsson-Fachler (1985) attribution — corrected allocation effect.

    Key differences vs BHB:
      - Allocation uses benchmark relative return (rb_sector - rb_total).
        Same formula as BHB allocation, but the intent is that this is the
        *pure* country/sector bet, not contaminated by selection.
      - Selection uses portfolio weights (not benchmark weights).
      - No interaction term: allocation + selection = active return exactly.
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        sector_map: Optional[dict] = None,
    ) -> None:
        self._wp = portfolio_weights.copy()
        self._wb = benchmark_weights.copy()
        self._rp = portfolio_returns.copy()
        self._rb = benchmark_returns.copy()
        self._sector_map = sector_map or {}

        common = (
            self._wp.index
            .intersection(self._wb.index)
            .intersection(self._rp.index)
            .intersection(self._rb.index)
        )
        self._wp = self._wp.loc[common]
        self._wb = self._wb.loc[common]
        self._rp = self._rp.loc[common]
        self._rb = self._rb.loc[common]

        self._rB: float = float((self._wb * self._rb).sum())

    def compute_allocation_effect(self) -> pd.Series:
        """BF allocation: (w_p - w_b) × (r_b_sector - r_b_total).

        Same as BHB — this is the pure sector/country bet effect.
        """
        alloc = (self._wp - self._wb) * (self._rb - self._rB)
        alloc.name = "allocation_effect"
        return alloc

    def compute_selection_effect(self) -> pd.Series:
        """BF selection: w_p × (r_p_sector - r_b_sector).

        Uses portfolio weights, not benchmark weights (unlike BHB).
        This removes the interaction term; allocation + selection = active return.
        """
        sel = self._wp * (self._rp - self._rb)
        sel.name = "selection_effect"
        return sel

    def compute_total_active_return(self) -> float:
        """Portfolio total return minus benchmark total return."""
        port_ret = float((self._wp * self._rp).sum())
        return port_ret - self._rB

    def compute_full_attribution(self) -> pd.DataFrame:
        """Attribution table: allocation + selection per sector (no interaction)."""
        alloc = self.compute_allocation_effect()
        sel = self.compute_selection_effect()

        df = pd.DataFrame({
            "portfolio_weight": self._wp,
            "benchmark_weight": self._wb,
            "portfolio_return": self._rp,
            "benchmark_return": self._rb,
            "allocation_effect": alloc,
            "selection_effect": sel,
        })
        df["total_effect"] = df["allocation_effect"] + df["selection_effect"]

        total_row = pd.DataFrame(
            [{
                "portfolio_weight": df["portfolio_weight"].sum(),
                "benchmark_weight": df["benchmark_weight"].sum(),
                "portfolio_return": float((self._wp * self._rp).sum()),
                "benchmark_return": self._rB,
                "allocation_effect": df["allocation_effect"].sum(),
                "selection_effect": df["selection_effect"].sum(),
                "total_effect": df["total_effect"].sum(),
            }],
            index=["TOTAL"],
        )
        return pd.concat([df, total_row])

    def verify_attribution(self, tolerance: float = 1e-6) -> bool:
        """Verify allocation + selection == active return (no interaction term in BF)."""
        df = self.compute_full_attribution()
        explained = float(df.loc["TOTAL", "total_effect"])
        active = self.compute_total_active_return()
        residual = abs(explained - active)
        logger.info(
            "bf_attribution.verify",
            active_return=round(active, 8),
            explained=round(explained, 8),
            residual=round(residual, 10),
            passed=residual < tolerance,
        )
        return residual < tolerance


# ---------------------------------------------------------------------------
# MultiPeriodAttribution
# ---------------------------------------------------------------------------

class MultiPeriodAttribution:
    """Multi-period geometric attribution via Carino logarithmic linking.

    Carino (1999) provides a theoretically correct way to compound single-period
    attributions geometrically. Naive arithmetic sum of single-period effects
    introduces compounding errors that grow with the number of periods.

    The Carino factor k_t converts each period's return to a log-space weight
    so that effects chain exactly to the full-period cumulative active return.
    """

    @staticmethod
    def _carino_k(r_portfolio: float, r_benchmark: float) -> float:
        """Carino logarithmic linking coefficient for period t.

        k_t = (log(1 + r_p) - log(1 + r_b)) / (r_p - r_b)
        Falls back to 1/(1+r_p) when r_p ≈ r_b (L'Hopital limit).
        """
        if abs(r_portfolio - r_benchmark) < 1e-10:
            return 1.0 / (1.0 + r_portfolio) if abs(1.0 + r_portfolio) > 1e-12 else 1.0
        return (
            (np.log1p(r_portfolio) - np.log1p(r_benchmark))
            / (r_portfolio - r_benchmark)
        )

    @staticmethod
    def _carino_K(r_portfolio_total: float, r_benchmark_total: float) -> float:
        """Full-period Carino scaling factor K."""
        if abs(r_portfolio_total - r_benchmark_total) < 1e-10:
            return 1.0 / (1.0 + r_portfolio_total) if abs(1.0 + r_portfolio_total) > 1e-12 else 1.0
        return (
            (np.log1p(r_portfolio_total) - np.log1p(r_benchmark_total))
            / (r_portfolio_total - r_benchmark_total)
        )

    def geometric_linking(
        self,
        period_attributions: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Chain single-period BHB attributions using Carino logarithmic linking.

        Each element of period_attributions must be the output of
        BHBAttribution.compute_full_attribution() (containing a TOTAL row and
        columns portfolio_return, benchmark_return, allocation_effect,
        selection_effect, interaction_effect).

        Returns a DataFrame with the geometrically-linked cumulative attribution
        using the same column structure, with a TOTAL row.
        """
        if not period_attributions:
            return pd.DataFrame()

        # Collect per-period total returns from TOTAL rows
        port_rets: list[float] = []
        bench_rets: list[float] = []
        for df in period_attributions:
            if "TOTAL" not in df.index:
                raise ValueError("Each period_attribution DataFrame must have a TOTAL index row.")
            port_rets.append(float(df.loc["TOTAL", "portfolio_return"]))
            bench_rets.append(float(df.loc["TOTAL", "benchmark_return"]))

        # Full-period cumulative returns (geometric)
        port_cum = float(np.prod([1 + r for r in port_rets]) - 1)
        bench_cum = float(np.prod([1 + r for r in bench_rets]) - 1)
        K = self._carino_K(port_cum, bench_cum)

        # Compute per-period Carino weights
        carino_weights: list[float] = []
        for rp, rb in zip(port_rets, bench_rets):
            # Geometric growth factors up to this period
            k_t = self._carino_k(rp, rb)
            carino_weights.append(k_t / K if abs(K) > 1e-12 else 1.0 / len(port_rets))

        # Aggregate sectors — union of all sector indices
        all_sectors = set()
        for df in period_attributions:
            all_sectors.update(df.index.tolist())
        all_sectors.discard("TOTAL")

        effect_cols = ["allocation_effect", "selection_effect", "interaction_effect"]
        linked: dict[str, dict[str, float]] = {s: {c: 0.0 for c in effect_cols} for s in all_sectors}
        linked["TOTAL"] = {c: 0.0 for c in effect_cols}

        for w, df in zip(carino_weights, period_attributions):
            for sector in all_sectors:
                if sector in df.index:
                    for col in effect_cols:
                        linked[sector][col] += w * float(df.loc[sector, col])
            for col in effect_cols:
                linked["TOTAL"][col] += w * float(df.loc["TOTAL", col])

        result = pd.DataFrame(linked).T
        result.index.name = "sector"
        result["total_effect"] = result[effect_cols].sum(axis=1)
        return result

    def run_rolling_attribution(
        self,
        portfolio_returns_df: pd.DataFrame,
        benchmark_returns_df: pd.DataFrame,
        sector_map: dict,
        window: int = 21,
    ) -> list[pd.DataFrame]:
        """Run rolling single-period BHB attribution over a trailing window.

        Parameters
        ----------
        portfolio_returns_df : pd.DataFrame
            Columns = sectors, index = dates, values = sector portfolio returns (daily).
        benchmark_returns_df : pd.DataFrame
            Same shape. Benchmark sector returns.
        sector_map : dict
            sector → (portfolio_weight, benchmark_weight) for each sector.
        window : int
            Rolling window length (trading days). Default 21 ≈ 1 month.

        Returns
        -------
        list[pd.DataFrame]
            One attribution DataFrame per rolling window end-date.
        """
        dates = portfolio_returns_df.index
        attributions: list[pd.DataFrame] = []

        for i in range(window, len(dates) + 1):
            window_idx = dates[i - window: i]
            p_slice = portfolio_returns_df.loc[window_idx]
            b_slice = benchmark_returns_df.loc[window_idx]

            # Compound returns within window
            p_cum = (1 + p_slice).prod() - 1
            b_cum = (1 + b_slice).prod() - 1

            # Weights from sector_map
            sectors = list(sector_map.keys())
            wp = pd.Series({s: sector_map[s][0] for s in sectors})
            wb = pd.Series({s: sector_map[s][1] for s in sectors})

            # Align with available sectors
            common = [s for s in sectors if s in p_cum.index and s in b_cum.index]
            bhb = BHBAttribution(
                portfolio_weights=wp.loc[common],
                benchmark_weights=wb.loc[common],
                portfolio_returns=p_cum.loc[common],
                benchmark_returns=b_cum.loc[common],
            )
            attr_df = bhb.compute_full_attribution()
            attr_df.attrs["window_end"] = dates[i - 1]
            attributions.append(attr_df)

        return attributions

    def cumulative_attribution(
        self,
        attributions: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Cumulative effects over time using geometric linking.

        Returns a DataFrame indexed by window end-date (taken from
        attr_df.attrs['window_end'] if present, else integer position),
        with columns: allocation_effect, selection_effect, interaction_effect,
        total_effect — all cumulative up to that date.
        """
        cumulative_records: list[dict] = []

        for i, attr_df in enumerate(attributions):
            linked = self.geometric_linking(attributions[: i + 1])
            total_row = linked.loc["TOTAL"]
            rec = {
                "window_end": attr_df.attrs.get("window_end", i),
                "allocation_effect": float(total_row["allocation_effect"]),
                "selection_effect": float(total_row["selection_effect"]),
                "interaction_effect": float(total_row.get("interaction_effect", 0.0)),
                "total_effect": float(total_row["total_effect"]),
            }
            cumulative_records.append(rec)

        df = pd.DataFrame(cumulative_records).set_index("window_end")
        return df


# ---------------------------------------------------------------------------
# FactorAttribution
# ---------------------------------------------------------------------------

class FactorAttribution:
    """Extends BHB with Fama-French FF5 factor attribution.

    Regresses portfolio excess active return on the five Fama-French factors
    (Mkt-RF, SMB, HML, RMW, CMA) and decomposes the active return into
    factor contributions plus unexplained alpha.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> None:
        """
        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio total returns (decimal).
        benchmark_returns : pd.Series
            Daily benchmark total returns (decimal).
        factor_returns : pd.DataFrame
            Daily factor returns. Expected columns (subset allowed):
            MKT_RF, SMB, HML, RMW, CMA, RF.
        """
        self._rp = portfolio_returns.copy()
        self._rb = benchmark_returns.copy()
        self._factors = factor_returns.copy()

        # Align on common dates
        common = self._rp.index.intersection(self._rb.index).intersection(self._factors.index)
        self._rp = self._rp.loc[common]
        self._rb = self._rb.loc[common]
        self._factors = self._factors.loc[common]

    def compute_factor_attribution(self) -> dict:
        """Regress active return on FF5 factors via OLS.

        Returns
        -------
        dict with keys:
            factor_betas : dict — factor → beta estimate
            factor_contributions : dict — factor → annualised contribution (bps)
            alpha : float — annualised Jensen's alpha (%)
            alpha_t_stat : float
            r_squared : float
            unexplained_return : float — residual (%)
            n_observations : int
        """
        # Active (excess) return series
        rf = self._factors["RF"] if "RF" in self._factors.columns else pd.Series(0.0, index=self._rp.index)
        active = self._rp - self._rb  # raw active, not excess over RF

        factor_cols = [c for c in self._factors.columns if c != "RF"]
        if not factor_cols:
            raise ValueError("No factor columns found in factor_returns DataFrame.")

        X = self._factors[factor_cols].values.astype(float)
        y = active.values.astype(float)
        n = len(y)

        if n < 30:
            raise ValueError(f"Insufficient data for regression: {n} observations (need ≥30).")

        X_const = np.column_stack([np.ones(n), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
        alpha_daily = float(coeffs[0])
        betas = coeffs[1:]

        # Standard errors
        y_hat = X_const @ coeffs
        resid = y - y_hat
        k = X_const.shape[1]
        s2 = float(np.dot(resid, resid) / (n - k)) if n > k else 0.0
        XtX_inv = np.linalg.pinv(X_const.T @ X_const)
        se = np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 0.0))
        t_stats = coeffs / (se + 1e-15)

        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        factor_betas: dict[str, float] = {}
        factor_contributions: dict[str, float] = {}
        for i, fname in enumerate(factor_cols):
            factor_betas[fname] = round(float(betas[i]), 5)
            factor_mean = float(self._factors[fname].mean() * 252)  # annualised
            factor_contributions[fname] = round(float(betas[i]) * factor_mean * 10000, 2)  # bps

        alpha_annual = round(alpha_daily * 252 * 100, 4)  # percent
        factor_contrib_pct = sum(factor_contributions.values()) / 10000 * 100
        active_annual = round(float(active.mean()) * 252 * 100, 4)
        unexplained = round(active_annual - alpha_annual - factor_contrib_pct, 4)

        logger.info(
            "factor_attribution.computed",
            alpha_pct=alpha_annual,
            r_squared=round(r_squared, 4),
            n_factors=len(factor_cols),
        )

        return {
            "factor_betas": factor_betas,
            "factor_contributions": factor_contributions,
            "alpha": alpha_annual,
            "alpha_t_stat": round(float(t_stats[0]), 4),
            "r_squared": round(r_squared, 4),
            "unexplained_return": unexplained,
            "n_observations": n,
        }

    def compute_style_attribution(
        self,
        style_indices: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Compare portfolio to growth/value/size style indices.

        Parameters
        ----------
        style_indices : dict
            Mapping of style name → daily return Series.
            Example: {"growth": spy_growth_rets, "value": spy_value_rets, "small_cap": iwm_rets}

        Returns
        -------
        pd.DataFrame with columns:
            style, allocation_to_style (beta), selection_within_style,
            style_contribution_bps
        """
        records: list[dict] = []
        active = self._rp - self._rb

        for style_name, style_rets in style_indices.items():
            common = active.index.intersection(style_rets.index)
            if len(common) < 20:
                continue

            a = active.loc[common].values
            s = style_rets.loc[common].values

            # Beta (allocation to style)
            cov = float(np.cov(a, s)[0, 1])
            var_s = float(np.var(s))
            beta = cov / var_s if var_s > 1e-12 else 0.0

            # Selection: active return not explained by style
            residual = a - beta * s
            selection = float(residual.mean() * 252 * 100)  # annualised %

            style_contribution = round(beta * float(s.mean()) * 252 * 10000, 2)  # bps

            records.append({
                "style": style_name,
                "allocation_to_style": round(beta, 4),
                "selection_within_style": round(selection, 4),
                "style_contribution_bps": style_contribution,
            })

        return pd.DataFrame(records).set_index("style") if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# TransactionCostAttribution
# ---------------------------------------------------------------------------

class TransactionCostAttribution:
    """Transaction cost attribution — implementation shortfall, VWAP slippage, TCA drag."""

    def compute_implementation_shortfall(
        self,
        ordered_price: float,
        executed_price: float,
        qty: int,
        direction: str,
    ) -> dict:
        """Compute implementation shortfall components for a single order.

        IS = (executed_price - ordered_price) × qty × sign(direction)

        Components (Perold 1988 framework):
          - delay_cost: cost of not trading immediately at decision price
          - market_impact: price movement caused by own order execution
          - timing_cost: cost of slippage across partial fills over time
          - opportunity_cost: unrealised P&L on unfilled portion

        Parameters
        ----------
        ordered_price : float
            Price at time of investment decision.
        executed_price : float
            Volume-weighted average execution price.
        qty : int
            Number of shares / units.
        direction : str
            "buy" or "sell".

        Returns
        -------
        dict with total_shortfall_bps and component breakdown.
        """
        sign = 1 if direction.lower() == "buy" else -1
        price_diff = (executed_price - ordered_price) * sign
        total_shortfall = price_diff * qty
        shortfall_bps = (price_diff / ordered_price) * 10000 if ordered_price > 0 else 0.0

        # Heuristic decomposition (without intraday tick data):
        # 50% assigned to market impact, 30% to timing, 20% to delay
        market_impact_bps = shortfall_bps * 0.50
        timing_bps = shortfall_bps * 0.30
        delay_bps = shortfall_bps * 0.20
        opportunity_bps = max(0.0, -shortfall_bps * 0.10)  # only when favourable fills missed

        logger.info(
            "implementation_shortfall",
            direction=direction,
            qty=qty,
            ordered_price=ordered_price,
            executed_price=executed_price,
            shortfall_bps=round(shortfall_bps, 2),
        )

        return {
            "total_shortfall": round(total_shortfall, 4),
            "total_shortfall_bps": round(shortfall_bps, 2),
            "delay_cost_bps": round(delay_bps, 2),
            "market_impact_bps": round(market_impact_bps, 2),
            "timing_cost_bps": round(timing_bps, 2),
            "opportunity_cost_bps": round(opportunity_bps, 2),
        }

    def compute_tca_summary(self, trades_df: pd.DataFrame) -> dict:
        """Aggregate TCA summary across all trades.

        Expected columns in trades_df:
          ordered_price, executed_price, qty, direction

        Returns aggregate shortfall, average bps drag, and total performance impact.
        """
        required = {"ordered_price", "executed_price", "qty", "direction"}
        missing = required - set(trades_df.columns)
        if missing:
            raise ValueError(f"trades_df missing columns: {missing}")

        results = []
        for _, row in trades_df.iterrows():
            r = self.compute_implementation_shortfall(
                ordered_price=float(row["ordered_price"]),
                executed_price=float(row["executed_price"]),
                qty=int(row["qty"]),
                direction=str(row["direction"]),
            )
            results.append(r)

        if not results:
            return {}

        total_shortfall = sum(r["total_shortfall"] for r in results)
        avg_shortfall_bps = float(np.mean([r["total_shortfall_bps"] for r in results]))
        avg_market_impact = float(np.mean([r["market_impact_bps"] for r in results]))
        avg_timing = float(np.mean([r["timing_cost_bps"] for r in results]))
        avg_delay = float(np.mean([r["delay_cost_bps"] for r in results]))
        n_trades = len(results)

        logger.info(
            "tca_summary",
            n_trades=n_trades,
            avg_shortfall_bps=round(avg_shortfall_bps, 2),
        )

        return {
            "n_trades": n_trades,
            "total_shortfall": round(total_shortfall, 4),
            "avg_shortfall_bps": round(avg_shortfall_bps, 2),
            "avg_market_impact_bps": round(avg_market_impact, 2),
            "avg_timing_cost_bps": round(avg_timing, 2),
            "avg_delay_cost_bps": round(avg_delay, 2),
            "estimated_annual_drag_bps": round(avg_shortfall_bps * n_trades / 252 * 252, 2),
        }

    def vwap_slippage(
        self,
        exec_price: float,
        vwap: float,
        qty: int,
        avg_daily_volume: int,
    ) -> dict:
        """Compute VWAP slippage and participation rate.

        Parameters
        ----------
        exec_price : float
            Volume-weighted average execution price.
        vwap : float
            Market VWAP for the trading session.
        qty : int
            Shares executed.
        avg_daily_volume : int
            20-day average daily volume (ADV) for the security.

        Returns
        -------
        dict with slippage_bps, participation_rate, market_impact_estimate_bps.
        """
        slippage_bps = ((exec_price - vwap) / vwap * 10000) if vwap > 0 else 0.0
        participation = qty / avg_daily_volume if avg_daily_volume > 0 else 0.0
        # Almgren-Chriss simplified: market_impact ~ sqrt(participation) × 50bps
        market_impact_estimate_bps = 50.0 * np.sqrt(participation)

        logger.info(
            "vwap_slippage",
            exec_price=exec_price,
            vwap=vwap,
            slippage_bps=round(slippage_bps, 2),
            participation_rate=round(participation, 4),
        )

        return {
            "slippage_bps": round(slippage_bps, 2),
            "participation_rate": round(participation, 4),
            "market_impact_estimate_bps": round(market_impact_estimate_bps, 2),
            "total_cost_bps": round(slippage_bps + market_impact_estimate_bps, 2),
        }


# ---------------------------------------------------------------------------
# AttributionReport
# ---------------------------------------------------------------------------

class AttributionReport:
    """Generate and format comprehensive attribution reports."""

    def generate_report(
        self,
        portfolio: dict,
        benchmark: dict,
        sector_map: dict,
        period: str,
    ) -> dict:
        """Full attribution summary report.

        Parameters
        ----------
        portfolio : dict
            sector → {"weight": float, "return": float}
        benchmark : dict
            sector → {"weight": float, "return": float}
        sector_map : dict
            Metadata about sectors (optional extra fields).
        period : str
            Human-readable period label (e.g., "2025-Q1").

        Returns
        -------
        dict with total active return, allocation/selection totals,
        sector-level breakdown, and formatted summary string.
        """
        sectors = list(set(portfolio.keys()) & set(benchmark.keys()))
        if not sectors:
            return {"error": "No common sectors between portfolio and benchmark."}

        wp = pd.Series({s: portfolio[s]["weight"] for s in sectors})
        wb = pd.Series({s: benchmark[s]["weight"] for s in sectors})
        rp = pd.Series({s: portfolio[s]["return"] for s in sectors})
        rb = pd.Series({s: benchmark[s]["return"] for s in sectors})

        bhb = BHBAttribution(wp, wb, rp, rb, sector_map)
        full_df = bhb.compute_full_attribution()
        verified = bhb.verify_attribution()

        bf = BrinssonFachlerAttribution(wp, wb, rp, rb, sector_map)
        bf_df = bf.compute_full_attribution()

        total = full_df.loc["TOTAL"]

        report = {
            "period": period,
            "total_active_return_bps": round(float(total["total_effect"]) * 10000, 2),
            "allocation_effect_bps": round(float(total["allocation_effect"]) * 10000, 2),
            "selection_effect_bps": round(float(total["selection_effect"]) * 10000, 2),
            "interaction_effect_bps": round(float(total["interaction_effect"]) * 10000, 2),
            "portfolio_return_pct": round(float(total["portfolio_return"]) * 100, 4),
            "benchmark_return_pct": round(float(total["benchmark_return"]) * 100, 4),
            "attribution_verified": verified,
            "model": "BHB_1986",
            "bf_active_return_bps": round(float(bf_df.loc["TOTAL", "total_effect"]) * 10000, 2),
            "sector_breakdown": {},
        }

        for sector in sectors:
            row = full_df.loc[sector]
            report["sector_breakdown"][sector] = {
                "allocation_bps": round(float(row["allocation_effect"]) * 10000, 2),
                "selection_bps": round(float(row["selection_effect"]) * 10000, 2),
                "interaction_bps": round(float(row["interaction_effect"]) * 10000, 2),
                "total_bps": round(float(row["total_effect"]) * 10000, 2),
                "portfolio_weight": round(float(row["portfolio_weight"]), 4),
                "benchmark_weight": round(float(row["benchmark_weight"]), 4),
            }

        report["summary"] = self.format_as_table(full_df)
        return report

    def format_as_table(self, attribution_df: pd.DataFrame) -> str:
        """Pretty-print attribution table with sector-level contributions in bps."""
        cols_wanted = ["allocation_effect", "selection_effect", "interaction_effect", "total_effect"]
        available = [c for c in cols_wanted if c in attribution_df.columns]
        display_df = (attribution_df[available] * 10000).round(2)
        display_df.columns = [c.replace("_effect", "").replace("_", " ").title() + " (bps)"
                               for c in display_df.columns]

        lines = ["=" * 90]
        lines.append(f"{'Sector':<30}" + "".join(f"{col:>15}" for col in display_df.columns))
        lines.append("-" * 90)

        non_total = [idx for idx in display_df.index if idx != "TOTAL"]
        for sector in non_total:
            row = display_df.loc[sector]
            lines.append(f"{str(sector):<30}" + "".join(f"{float(v):>15.2f}" for v in row))

        if "TOTAL" in display_df.index:
            lines.append("=" * 90)
            row = display_df.loc["TOTAL"]
            lines.append(f"{'TOTAL':<30}" + "".join(f"{float(v):>15.2f}" for v in row))

        lines.append("=" * 90)
        return "\n".join(lines)

    def to_excel_dict(self, attribution_df: pd.DataFrame) -> dict:
        """Serialise attribution DataFrame to a JSON/Excel-compatible dict structure."""
        df_bps = (attribution_df * 10000).round(2)
        rows = []
        for idx, row in df_bps.iterrows():
            record: dict[str, Any] = {"sector": str(idx)}
            for col in df_bps.columns:
                record[col] = float(row[col]) if not pd.isna(row[col]) else None
            rows.append(record)
        return {
            "columns": ["sector"] + list(df_bps.columns),
            "rows": rows,
            "units": "basis_points",
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

attribution_router = APIRouter(prefix="/api/attribution", tags=["attribution"])


class BHBRequest(BaseModel):
    portfolio_weights: dict[str, float] = Field(..., description="sector → portfolio weight")
    benchmark_weights: dict[str, float] = Field(..., description="sector → benchmark weight")
    portfolio_returns: dict[str, float] = Field(..., description="sector → portfolio period return")
    benchmark_returns: dict[str, float] = Field(..., description="sector → benchmark period return")
    period: str = Field(default="", description="Human-readable period label")


class MultiPeriodRequest(BaseModel):
    periods: list[BHBRequest] = Field(..., description="Ordered list of single-period attribution inputs")


class FactorAttributionRequest(BaseModel):
    portfolio_returns: dict[str, float] = Field(..., description="date → portfolio return")
    benchmark_returns: dict[str, float] = Field(..., description="date → benchmark return")
    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")


@attribution_router.post("/bhb")
async def run_bhb_attribution(req: BHBRequest) -> dict:
    """Run BHB sector attribution for a single period."""
    try:
        sectors = list(set(req.portfolio_weights.keys()) & set(req.benchmark_weights.keys()))
        wp = pd.Series({s: req.portfolio_weights.get(s, 0.0) for s in sectors})
        wb = pd.Series({s: req.benchmark_weights.get(s, 0.0) for s in sectors})
        rp = pd.Series({s: req.portfolio_returns.get(s, 0.0) for s in sectors})
        rb = pd.Series({s: req.benchmark_returns.get(s, 0.0) for s in sectors})

        bhb = BHBAttribution(wp, wb, rp, rb)
        full_df = bhb.compute_full_attribution()
        reporter = AttributionReport()

        return {
            "status": "ok",
            "period": req.period,
            "active_return_bps": round(bhb.compute_total_active_return() * 10000, 2),
            "verified": bhb.verify_attribution(),
            "table": reporter.format_as_table(full_df),
            "excel_export": reporter.to_excel_dict(full_df),
        }
    except Exception as exc:
        logger.error("bhb_route_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_router.post("/multi-period")
async def run_multi_period_attribution(req: MultiPeriodRequest) -> dict:
    """Multi-period geometric attribution using Carino logarithmic linking."""
    try:
        period_dfs: list[pd.DataFrame] = []
        for p in req.periods:
            sectors = list(set(p.portfolio_weights.keys()) & set(p.benchmark_weights.keys()))
            wp = pd.Series({s: p.portfolio_weights.get(s, 0.0) for s in sectors})
            wb = pd.Series({s: p.benchmark_weights.get(s, 0.0) for s in sectors})
            rp = pd.Series({s: p.portfolio_returns.get(s, 0.0) for s in sectors})
            rb = pd.Series({s: p.benchmark_returns.get(s, 0.0) for s in sectors})
            bhb = BHBAttribution(wp, wb, rp, rb)
            period_dfs.append(bhb.compute_full_attribution())

        mp = MultiPeriodAttribution()
        linked = mp.geometric_linking(period_dfs)
        reporter = AttributionReport()

        return {
            "status": "ok",
            "n_periods": len(req.periods),
            "linked_table": reporter.format_as_table(linked),
            "excel_export": reporter.to_excel_dict(linked),
        }
    except Exception as exc:
        logger.error("multi_period_route_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_router.post("/factor")
async def run_factor_attribution(req: FactorAttributionRequest) -> dict:
    """Factor attribution using Fama-French FF5 factors."""
    try:
        end = date.fromisoformat(req.end_date) if req.end_date else date.today()
        start = date.fromisoformat(req.start_date) if req.start_date else end - timedelta(days=252)

        p_rets = pd.Series(req.portfolio_returns)
        b_rets = pd.Series(req.benchmark_returns)
        p_rets.index = pd.to_datetime(p_rets.index)
        b_rets.index = pd.to_datetime(b_rets.index)

        ff5 = await _fetch_ff5_factors(start, end)
        if ff5 is None:
            raise HTTPException(status_code=503, detail="FF5 factor data unavailable")

        fa = FactorAttribution(p_rets, b_rets, ff5)
        result = fa.compute_factor_attribution()
        return {"status": "ok", **result}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("factor_route_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_router.get("/{portfolio_id}/report")
async def get_portfolio_report(
    portfolio_id: str,
    period: str = Query(default="YTD"),
) -> dict:
    """Full attribution report for a portfolio ID (demo with SPY/QQQ as sectors)."""
    # In production: load portfolio from DB by portfolio_id.
    # For demo: return a stub structure showing the API shape.
    demo_portfolio = {
        "Technology": {"weight": 0.35, "return": 0.18},
        "Health Care": {"weight": 0.20, "return": 0.08},
        "Financials": {"weight": 0.15, "return": 0.12},
        "Energy": {"weight": 0.10, "return": 0.25},
        "Industrials": {"weight": 0.20, "return": 0.10},
    }
    demo_benchmark = {
        "Technology": {"weight": 0.30, "return": 0.15},
        "Health Care": {"weight": 0.13, "return": 0.07},
        "Financials": {"weight": 0.13, "return": 0.11},
        "Energy": {"weight": 0.04, "return": 0.22},
        "Industrials": {"weight": 0.09, "return": 0.09},
    }

    reporter = AttributionReport()
    report = reporter.generate_report(
        portfolio=demo_portfolio,
        benchmark=demo_benchmark,
        sector_map={},
        period=period,
    )
    report["portfolio_id"] = portfolio_id
    report["note"] = "Demo data — load real portfolio from DB in production."
    return report
