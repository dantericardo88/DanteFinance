"""Brinson-Hood-Beebower (BHB) portfolio attribution, Fama-French factor attribution,
and risk attribution for the SENTINEL financial terminal."""
from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import date, datetime, timedelta
from typing import Optional, Literal

import numpy as np
import pandas as pd
import httpx
import yfinance as yf
from pydantic import BaseModel, Field
from scipy import stats

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FF_FACTOR_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
_HEADERS = {"User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"}

# GICS sector ETF proxies
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

# Factor proxy ETFs (for when Fama-French CSV is unavailable)
FACTOR_ETF_PROXIES: dict[str, tuple[str, str | None, float]] = {
    # factor: (long_etf, short_etf_or_None, scaling)
    "market": ("SPY", None, 1.0),
    "smb":    ("IWM", "SPY", 1.0),   # small – large cap
    "hml":    ("IVE", "IVW", 1.0),   # value – growth
    "mom":    ("MTUM", "SPY", 1.0),  # momentum
}


# ── Pydantic models ────────────────────────────────────────────────────────────

class Holding(BaseModel):
    ticker: str
    weight: float            # portfolio weight (0-1)
    benchmark_weight: float  # benchmark weight (0-1)
    sector: Optional[str] = None
    return_period: Optional[float] = None  # pre-computed return if available


class AttributionEffect(BaseModel):
    allocation: float    # sector allocation effect
    selection: float     # security selection effect
    interaction: float   # interaction effect
    total: float         # sum of all effects


class SectorAttribution(BaseModel):
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float


class BHBAttribution(BaseModel):
    portfolio_name: str
    start_date: date
    end_date: date
    portfolio_return: float
    benchmark_return: float
    active_return: float          # portfolio - benchmark
    allocation_total: float       # sum of allocation effects
    selection_total: float        # sum of selection effects
    interaction_total: float      # sum of interaction effects
    residual: float               # should be near 0
    sector_attribution: list[SectorAttribution]
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


class FactorExposure(BaseModel):
    factor: str
    beta: float
    t_stat: float
    p_value: float
    contribution_to_return: float  # bps


class FactorAttribution(BaseModel):
    portfolio_name: str
    start_date: date
    end_date: date
    portfolio_return: float
    risk_free_rate: float
    alpha: float            # Jensen's alpha (annualised, %)
    alpha_t_stat: float
    factor_exposures: list[FactorExposure]
    r_squared: float
    specific_return: float  # return not explained by factors


class RiskAttribution(BaseModel):
    portfolio_name: str
    as_of: date
    portfolio_volatility: float   # annualised %
    benchmark_volatility: float
    tracking_error: float
    beta: float
    correlation_to_benchmark: float
    var_95_1d: float              # 1-day VaR at 95% (% of portfolio)
    cvar_95_1d: float
    max_drawdown: float
    market_risk_pct: float        # % of total variance from market factor
    factor_risk_pct: float        # % from style factors
    specific_risk_pct: float      # % from idiosyncratic (remainder)


# ── Core attributor ────────────────────────────────────────────────────────────

class PortfolioAttributor:
    """Computes BHB attribution, Fama-French factor attribution, and risk attribution."""

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        self._ff_factors: Optional[pd.DataFrame] = None

    # ── BHB Attribution ────────────────────────────────────────────────────────

    async def compute_bhb_attribution(
        self,
        holdings: list[Holding],
        benchmark_ticker: str = "SPY",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        portfolio_name: str = "Portfolio",
    ) -> BHBAttribution:
        """Full Brinson-Hood-Beebower sector attribution.

        Steps:
          1. Fetch period returns for every holding + benchmark + sector ETFs.
          2. Aggregate holdings into sectors (portfolio weight & return).
          3. Apply BHB formulas per sector.
          4. Sum effects; compute residual and tracking error.
        """
        end = end_date or date.today()
        start = start_date or (end - timedelta(days=252))

        # ── 1. Gather tickers ──
        all_tickers = list({h.ticker for h in holdings})
        sector_tickers = list(SECTOR_ETFS.values())
        all_tickers_ext = list(set(all_tickers + sector_tickers + [benchmark_ticker]))

        logger.info(
            "bhb_attribution.fetching_returns",
            n_holdings=len(holdings),
            benchmark=benchmark_ticker,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        price_df = await self._fetch_returns(all_tickers_ext, start, end)

        # ── 2. Compute single-period total returns ──
        def _period_return(ticker: str) -> float:
            if ticker not in price_df.columns:
                return 0.0
            col = price_df[ticker].dropna()
            if len(col) < 2:
                return 0.0
            return float((col.iloc[-1] / col.iloc[0]) - 1)

        benchmark_return = _period_return(benchmark_ticker)

        # ── 3. Sector ETF returns (benchmark for each GICS sector) ──
        sector_bench_returns: dict[str, float] = {
            sector: _period_return(etf) for sector, etf in SECTOR_ETFS.items()
        }

        # ── 4. Map holdings → sectors ──
        sector_map: dict[str, dict] = {}
        for h in holdings:
            sector = h.sector or "Unclassified"
            if sector not in sector_map:
                sector_map[sector] = {
                    "portfolio_weight": 0.0,
                    "benchmark_weight": 0.0,
                    "weighted_return_numerator": 0.0,
                    "holding_returns": [],
                }
            sm = sector_map[sector]
            sm["portfolio_weight"] += h.weight
            sm["benchmark_weight"] += h.benchmark_weight
            # Use pre-computed return if available, else fetch
            r = h.return_period if h.return_period is not None else _period_return(h.ticker)
            sm["weighted_return_numerator"] += h.weight * r
            sm["holding_returns"].append((h.weight, r))

        # ── 5. Sector-level portfolio returns (weight-average within sector) ──
        sector_attributions: list[SectorAttribution] = []
        alloc_total = sel_total = inter_total = 0.0
        port_return = 0.0

        for sector, sm in sector_map.items():
            wp = sm["portfolio_weight"]
            wb = sm["benchmark_weight"]
            # Portfolio return for this sector
            if wp > 1e-9:
                rp = sm["weighted_return_numerator"] / wp
            else:
                rp = 0.0
            # Benchmark return for this sector (GICS ETF proxy)
            rb = sector_bench_returns.get(sector, benchmark_return)
            rB = benchmark_return

            alloc, sel, inter = self._bhb_effects(wp, wb, rp, rb, rB)

            sector_attributions.append(SectorAttribution(
                sector=sector,
                portfolio_weight=round(wp, 6),
                benchmark_weight=round(wb, 6),
                portfolio_return=round(rp, 6),
                benchmark_return=round(rb, 6),
                allocation_effect=round(alloc, 6),
                selection_effect=round(sel, 6),
                interaction_effect=round(inter, 6),
                total_effect=round(alloc + sel + inter, 6),
            ))

            alloc_total += alloc
            sel_total += sel
            inter_total += inter
            port_return += wp * rp

        active_return = port_return - benchmark_return
        residual = active_return - (alloc_total + sel_total + inter_total)

        # ── 6. Tracking error and IR from daily returns ──
        tracking_error: Optional[float] = None
        information_ratio: Optional[float] = None

        # Reconstruct daily portfolio returns using equal-weight of holdings in price_df
        available = [h.ticker for h in holdings if h.ticker in price_df.columns]
        if available and benchmark_ticker in price_df.columns:
            weights = np.array([
                next(h.weight for h in holdings if h.ticker == t) for t in available
            ])
            daily_prices = price_df[available].dropna()
            daily_rets = daily_prices.pct_change().dropna()
            port_daily = daily_rets.values @ weights / weights.sum()
            bench_daily = price_df[benchmark_ticker].pct_change().dropna()
            aligned = bench_daily.reindex(daily_rets.index).dropna()
            port_aligned = pd.Series(port_daily, index=daily_rets.index).reindex(aligned.index)
            active_daily = port_aligned - aligned
            te = float(active_daily.std() * np.sqrt(252) * 100)
            tracking_error = round(te, 4)
            if te > 0:
                ir_daily = float(active_daily.mean() / active_daily.std() * np.sqrt(252))
                information_ratio = round(ir_daily, 4)

        logger.info(
            "bhb_attribution.complete",
            active_return_bps=round(active_return * 10000, 1),
            alloc_bps=round(alloc_total * 10000, 1),
            sel_bps=round(sel_total * 10000, 1),
        )

        return BHBAttribution(
            portfolio_name=portfolio_name,
            start_date=start,
            end_date=end,
            portfolio_return=round(port_return, 6),
            benchmark_return=round(benchmark_return, 6),
            active_return=round(active_return, 6),
            allocation_total=round(alloc_total, 6),
            selection_total=round(sel_total, 6),
            interaction_total=round(inter_total, 6),
            residual=round(residual, 8),
            sector_attribution=sector_attributions,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
        )

    # ── Factor Attribution ─────────────────────────────────────────────────────

    async def compute_factor_attribution(
        self,
        holdings: list[Holding],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        factors: list[str] = None,
        portfolio_name: str = "Portfolio",
    ) -> FactorAttribution:
        """Fama-French style multi-factor regression attribution.

        Attempts to download Ken French's daily factor CSV. Falls back to
        ETF-based factor proxies if the academic server is unreachable.

        Factors: market, smb (size), hml (value), mom (momentum).
        """
        if factors is None:
            factors = ["market", "smb", "hml", "mom"]

        end = end_date or date.today()
        start = start_date or (end - timedelta(days=252))

        logger.info(
            "factor_attribution.start",
            factors=factors,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        # ── 1. Build portfolio daily returns ──
        available_tickers = list({h.ticker for h in holdings})
        price_df = await self._fetch_returns(
            available_tickers + ["SPY", "IWM", "IVE", "IVW", "MTUM"], start, end
        )

        valid_tickers = [t for t in available_tickers if t in price_df.columns]
        weights = np.array([
            next((h.weight for h in holdings if h.ticker == t), 0.0)
            for t in valid_tickers
        ])
        if weights.sum() > 0:
            weights /= weights.sum()

        daily_prices = price_df[valid_tickers].dropna()
        daily_rets = daily_prices.pct_change().dropna()
        port_daily = pd.Series(
            daily_rets.values @ weights,
            index=daily_rets.index,
            name="portfolio",
        )

        # ── 2. Fetch Fama-French factors (academic or ETF proxy) ──
        ff_df = await self._fetch_ff_factors(start, end)

        if ff_df is not None:
            # Use academic daily factors (already in decimal)
            ff_df.index = pd.to_datetime(ff_df.index)
            common_idx = port_daily.index.intersection(ff_df.index)
            port_excess = port_daily.loc[common_idx] - ff_df.loc[common_idx, "RF"]
            factor_matrix = ff_df.loc[common_idx, [f.upper() for f in factors
                                                     if f.upper() in ff_df.columns]]
            rf_annual = float(ff_df["RF"].mean() * 252)
        else:
            # ETF proxy fallback
            factor_matrix, rf_series = self._build_etf_factors(price_df, factors)
            common_idx = port_daily.index.intersection(factor_matrix.index)
            port_excess = port_daily.loc[common_idx] - rf_series.loc[common_idx]
            factor_matrix = factor_matrix.loc[common_idx]
            rf_annual = float(rf_series.mean() * 252)

        if len(common_idx) < 30:
            raise ValueError(
                f"Insufficient overlapping data points for regression: {len(common_idx)}"
            )

        # ── 3. OLS regression ──
        X = factor_matrix.values
        y = port_excess.values
        X_const = np.column_stack([np.ones(len(y)), X])

        result = stats.linregress  # Use scipy for single-factor, numpy for multi
        # Multi-factor OLS via numpy lstsq
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_const, y, rcond=None)
        alpha_daily = coeffs[0]
        betas = coeffs[1:]

        # Standard errors
        n, k = X_const.shape
        y_hat = X_const @ coeffs
        resid = y - y_hat
        s2 = float(np.dot(resid, resid) / (n - k))
        XtX_inv = np.linalg.pinv(X_const.T @ X_const)
        se = np.sqrt(s2 * np.diag(XtX_inv))
        t_stats = coeffs / se

        # p-values (two-tailed)
        p_values = [2 * (1 - stats.t.cdf(abs(t), df=n - k)) for t in t_stats]

        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # ── 4. Factor contributions (annualised bps) ──
        factor_cols = list(factor_matrix.columns)
        factor_exposures: list[FactorExposure] = []
        for i, fname in enumerate(factor_cols):
            factor_mean_return = float(factor_matrix[fname].mean() * 252)
            contrib_annual = float(betas[i] * factor_mean_return * 10000)
            factor_exposures.append(FactorExposure(
                factor=fname.lower(),
                beta=round(float(betas[i]), 4),
                t_stat=round(float(t_stats[i + 1]), 4),
                p_value=round(float(p_values[i + 1]), 4),
                contribution_to_return=round(contrib_annual, 2),
            ))

        alpha_annual = float(alpha_daily * 252 * 100)
        port_return_total = float(port_daily.mean() * 252 * 100)

        # Specific return = total return - (alpha + sum of factor contributions/10000)
        factor_contrib_total = sum(fe.contribution_to_return for fe in factor_exposures) / 10000 * 100
        specific_return = port_return_total - alpha_annual - factor_contrib_total

        logger.info(
            "factor_attribution.complete",
            alpha_pct=round(alpha_annual, 3),
            r_squared=round(r_squared, 4),
        )

        return FactorAttribution(
            portfolio_name=portfolio_name,
            start_date=start,
            end_date=end,
            portfolio_return=round(port_return_total, 4),
            risk_free_rate=round(rf_annual * 100, 4),
            alpha=round(alpha_annual, 4),
            alpha_t_stat=round(float(t_stats[0]), 4),
            factor_exposures=factor_exposures,
            r_squared=round(r_squared, 4),
            specific_return=round(specific_return, 4),
        )

    # ── Risk Attribution ───────────────────────────────────────────────────────

    async def compute_risk_attribution(
        self,
        holdings: list[Holding],
        benchmark_ticker: str = "SPY",
        lookback_days: int = 252,
        portfolio_name: str = "Portfolio",
    ) -> RiskAttribution:
        """Covariance-matrix risk decomposition.

        Portfolio variance = w'Σw.
        Decomposes into market, style-factor, and idiosyncratic risk.
        """
        end = date.today()
        start = end - timedelta(days=lookback_days + 30)

        tickers = list({h.ticker for h in holdings})
        all_t = tickers + [benchmark_ticker, "SPY", "IWM", "IVE", "IVW"]
        price_df = await self._fetch_returns(all_t, start, end)

        valid = [t for t in tickers if t in price_df.columns]
        weights = np.array([
            next((h.weight for h in holdings if h.ticker == t), 0.0) for t in valid
        ])
        if weights.sum() > 0:
            weights /= weights.sum()

        rets = price_df[valid].pct_change().dropna().tail(lookback_days)
        bench_rets = price_df[benchmark_ticker].pct_change().dropna().tail(lookback_days)

        port_rets = rets.values @ weights
        common = min(len(port_rets), len(bench_rets))
        port_rets = port_rets[-common:]
        bench_arr = bench_rets.values[-common:]

        # ── Annualised stats ──
        port_vol = float(port_rets.std() * np.sqrt(252) * 100)
        bench_vol = float(bench_arr.std() * np.sqrt(252) * 100)

        # Tracking error
        active = port_rets - bench_arr
        te = float(active.std() * np.sqrt(252) * 100)

        # Beta + correlation
        cov_matrix = np.cov(port_rets, bench_arr)
        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 1.0
        corr = float(np.corrcoef(port_rets, bench_arr)[0, 1])

        # VaR / CVaR (historical)
        var_95 = float(-np.percentile(port_rets, 5) * 100)
        cvar_95 = float(-port_rets[port_rets <= np.percentile(port_rets, 5)].mean() * 100)

        # Max drawdown
        cumulative = (1 + port_rets).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(drawdowns.min() * 100)

        # ── Risk decomposition via covariance ──
        # Portfolio variance
        sigma = np.cov(rets.values.T)
        port_var = float(weights @ sigma @ weights)

        # Market factor variance contribution: beta^2 * bench_var
        bench_var = float(bench_arr.var())
        market_var = (beta ** 2) * bench_var
        market_risk_pct = min(market_var / port_var * 100, 100.0) if port_var > 0 else 0.0

        # Style factor variance: use SMB and HML proxies
        style_tickers = ["IWM", "IVE"]
        style_cols = [t for t in style_tickers if t in price_df.columns]
        if style_cols and len(valid) > 0:
            style_rets = price_df[style_cols].pct_change().dropna().tail(common)
            style_var_sum = 0.0
            for sc in style_cols:
                sc_arr = style_rets[sc].values[-common:] if sc in style_rets.columns else np.zeros(common)
                cov_ps = float(np.cov(port_rets, sc_arr)[0, 1])
                factor_beta = cov_ps / float(np.var(sc_arr)) if np.var(sc_arr) > 0 else 0.0
                style_var_sum += (factor_beta ** 2) * float(np.var(sc_arr))
            factor_risk_pct = min(style_var_sum / port_var * 100, 100.0 - market_risk_pct) if port_var > 0 else 0.0
        else:
            factor_risk_pct = 0.0

        specific_risk_pct = max(0.0, 100.0 - market_risk_pct - factor_risk_pct)

        logger.info(
            "risk_attribution.complete",
            portfolio_vol=round(port_vol, 2),
            tracking_error=round(te, 2),
            beta=round(beta, 3),
        )

        return RiskAttribution(
            portfolio_name=portfolio_name,
            as_of=end,
            portfolio_volatility=round(port_vol, 4),
            benchmark_volatility=round(bench_vol, 4),
            tracking_error=round(te, 4),
            beta=round(beta, 4),
            correlation_to_benchmark=round(corr, 4),
            var_95_1d=round(var_95, 4),
            cvar_95_1d=round(cvar_95, 4),
            max_drawdown=round(max_dd, 4),
            market_risk_pct=round(market_risk_pct, 2),
            factor_risk_pct=round(factor_risk_pct, 2),
            specific_risk_pct=round(specific_risk_pct, 2),
        )

    # ── Sector returns helper ──────────────────────────────────────────────────

    async def get_sector_returns(
        self,
        sector_etfs: Optional[dict[str, str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> dict[str, float]:
        """Fetch period total returns for GICS sector ETFs.

        Returns mapping of sector name → total return over period.
        Default sector ETFs: XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU.
        """
        mapping = sector_etfs or SECTOR_ETFS
        end = end or date.today()
        start = start or (end - timedelta(days=252))

        tickers = list(mapping.values())
        price_df = await self._fetch_returns(tickers, start, end)

        result: dict[str, float] = {}
        for sector, etf in mapping.items():
            if etf not in price_df.columns:
                continue
            col = price_df[etf].dropna()
            if len(col) < 2:
                continue
            result[sector] = round(float(col.iloc[-1] / col.iloc[0] - 1), 6)

        return result

    # ── Rolling attribution ────────────────────────────────────────────────────

    async def rolling_attribution(
        self,
        holdings: list[Holding],
        benchmark_ticker: str = "SPY",
        window_days: int = 63,
        portfolio_name: str = "Portfolio",
    ) -> pd.DataFrame:
        """Compute rolling BHB attribution over time.

        Returns DataFrame with columns:
          date, portfolio_return, benchmark_return, active_return,
          allocation_effect, selection_effect, interaction_effect.
        """
        end = date.today()
        start = end - timedelta(days=window_days * 4 + 60)

        all_tickers = list({h.ticker for h in holdings})
        sector_tickers = list(SECTOR_ETFS.values())
        all_ext = list(set(all_tickers + sector_tickers + [benchmark_ticker]))

        price_df = await self._fetch_returns(all_ext, start, end)

        valid = [t for t in all_tickers if t in price_df.columns]
        weights = np.array([
            next((h.weight for h in holdings if h.ticker == t), 0.0) for t in valid
        ])
        if weights.sum() > 0:
            weights /= weights.sum()

        rets_df = price_df.pct_change().dropna()
        trading_dates = rets_df.index

        records = []
        for i in range(window_days, len(trading_dates)):
            window_idx = trading_dates[i - window_days: i]
            window_rets = rets_df.loc[window_idx]

            def _cum_ret(col: str) -> float:
                if col not in window_rets.columns:
                    return 0.0
                return float((1 + window_rets[col]).prod() - 1)

            bench_ret = _cum_ret(benchmark_ticker)

            sector_bench: dict[str, float] = {
                sector: _cum_ret(etf) for sector, etf in SECTOR_ETFS.items()
            }

            # Portfolio sector aggregation
            sector_map: dict[str, dict] = {}
            for h in holdings:
                if h.ticker not in valid:
                    continue
                sector = h.sector or "Unclassified"
                sm = sector_map.setdefault(sector, {
                    "wp": 0.0, "wb": 0.0, "weighted_ret": 0.0
                })
                sm["wp"] += h.weight
                sm["wb"] += h.benchmark_weight
                r = _cum_ret(h.ticker)
                sm["weighted_ret"] += h.weight * r

            alloc_t = sel_t = inter_t = port_ret = 0.0
            for sector, sm in sector_map.items():
                wp = sm["wp"]
                wb = sm["wb"]
                rp = sm["weighted_ret"] / wp if wp > 1e-9 else 0.0
                rb = sector_bench.get(sector, bench_ret)
                a, s, inter = self._bhb_effects(wp, wb, rp, rb, bench_ret)
                alloc_t += a
                sel_t += s
                inter_t += inter
                port_ret += wp * rp

            records.append({
                "date": trading_dates[i],
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
                "active_return": port_ret - bench_ret,
                "allocation_effect": alloc_t,
                "selection_effect": sel_t,
                "interaction_effect": inter_t,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _fetch_returns(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Download adjusted close prices via yfinance (batch download).

        Returns a DataFrame of adjusted close prices indexed by date.
        Auto-adjusted=True ensures total return (dividends reinvested).
        """
        if not tickers:
            return pd.DataFrame()

        unique = list(set(tickers))
        start_str = start.strftime("%Y-%m-%d")
        end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.download(
                    unique,
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                ),
            )
        except Exception as exc:
            logger.warning("yfinance_download_failed", error=str(exc))
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        # yfinance returns MultiIndex columns when > 1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw
            if len(unique) == 1:
                close = close.rename(columns={"Close": unique[0]})

        close.index = pd.to_datetime(close.index)
        return close.sort_index()

    async def _fetch_ff_factors(
        self, start: date, end: date
    ) -> Optional[pd.DataFrame]:
        """Attempt to download Ken French daily factor returns.

        Returns a DataFrame with columns: Mkt-RF, SMB, HML, Mom, RF (all decimal).
        Returns None if the download fails (uses ETF proxies instead).
        """
        if self._ff_factors is not None:
            df = self._ff_factors
            # Filter to requested range
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            filtered = df.loc[mask]
            return filtered if not filtered.empty else None

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, headers=_HEADERS, follow_redirects=True
            ) as client:
                resp = await client.get(FF_FACTOR_URL)
                resp.raise_for_status()

            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = next(n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")

            # Skip header lines (Kenneth French format: copyright text then blank line)
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
            # Convert from percent to decimal
            df = df / 100.0

            # Rename for consistent access
            rename = {"Mkt-RF": "MARKET", "SMB": "SMB", "HML": "HML", "RF": "RF"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

            self._ff_factors = df
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            filtered = df.loc[mask]
            return filtered if not filtered.empty else None

        except Exception as exc:
            logger.warning(
                "ff_factors_unavailable",
                error=str(exc),
                fallback="etf_proxies",
            )
            return None

    def _build_etf_factors(
        self, price_df: pd.DataFrame, factors: list[str]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build factor return series from ETF proxies.

        Returns (factor_DataFrame, rf_Series). RF is set to 0 (conservative).
        """
        rets = price_df.pct_change().dropna()
        factor_data: dict[str, pd.Series] = {}

        for factor in factors:
            factor_upper = factor.upper()
            if factor in ("market", "MARKET"):
                if "SPY" in rets.columns:
                    factor_data[factor_upper] = rets["SPY"]
            elif factor in ("smb", "SMB"):
                if "IWM" in rets.columns and "SPY" in rets.columns:
                    factor_data[factor_upper] = rets["IWM"] - rets["SPY"]
            elif factor in ("hml", "HML"):
                if "IVE" in rets.columns and "IVW" in rets.columns:
                    factor_data[factor_upper] = rets["IVE"] - rets["IVW"]
            elif factor in ("mom", "MOM"):
                if "MTUM" in rets.columns and "SPY" in rets.columns:
                    factor_data[factor_upper] = rets["MTUM"] - rets["SPY"]

        factor_df = pd.DataFrame(factor_data).dropna()
        # RF = 0 when using ETF proxies (conservative)
        rf_series = pd.Series(0.0, index=factor_df.index)
        return factor_df, rf_series

    @staticmethod
    def _bhb_effects(
        wp: float, wb: float, rp: float, rb: float, rB: float
    ) -> tuple[float, float, float]:
        """Brinson-Hood-Beebower attribution decomposition for one sector.

        Args:
            wp: Portfolio sector weight.
            wb: Benchmark sector weight.
            rp: Portfolio return in sector (weighted average of holdings).
            rb: Benchmark sector return (GICS ETF proxy).
            rB: Overall benchmark return.

        Returns:
            (allocation_effect, selection_effect, interaction_effect)

        Formulas (Brinson, Hood & Beebower 1986):
            Allocation  = (wp - wb) × (rb - rB)
            Selection   = wb × (rp - rb)
            Interaction = (wp - wb) × (rp - rb)
        """
        allocation = (wp - wb) * (rb - rB)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)
        return allocation, selection, interaction


# ── Module-level convenience helpers ──────────────────────────────────────────

async def bhb_attribution(
    holdings: list[dict],
    benchmark: str = "SPY",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    portfolio_name: str = "Portfolio",
) -> BHBAttribution:
    """Convenience wrapper: accepts list of dicts, returns BHBAttribution."""
    attributor = PortfolioAttributor()
    parsed = [Holding(**h) for h in holdings]
    return await attributor.compute_bhb_attribution(
        parsed, benchmark_ticker=benchmark,
        start_date=start_date, end_date=end_date,
        portfolio_name=portfolio_name,
    )


async def factor_attribution(
    holdings: list[dict],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    factors: Optional[list[str]] = None,
    portfolio_name: str = "Portfolio",
) -> FactorAttribution:
    """Convenience wrapper: accepts list of dicts, returns FactorAttribution."""
    attributor = PortfolioAttributor()
    parsed = [Holding(**h) for h in holdings]
    return await attributor.compute_factor_attribution(
        parsed,
        start_date=start_date,
        end_date=end_date,
        factors=factors or ["market", "smb", "hml", "mom"],
        portfolio_name=portfolio_name,
    )


async def risk_attribution(
    holdings: list[dict],
    benchmark: str = "SPY",
    lookback_days: int = 252,
    portfolio_name: str = "Portfolio",
) -> RiskAttribution:
    """Convenience wrapper: accepts list of dicts, returns RiskAttribution."""
    attributor = PortfolioAttributor()
    parsed = [Holding(**h) for h in holdings]
    return await attributor.compute_risk_attribution(
        parsed,
        benchmark_ticker=benchmark,
        lookback_days=lookback_days,
        portfolio_name=portfolio_name,
    )
