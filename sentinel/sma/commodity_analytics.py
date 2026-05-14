"""
Commodity Analytics — Dimension #48.

Multi-sector commodity dashboard (FRED + yfinance). Covers Energy, Metals,
Agriculture, futures spread structure, and regime detection via CPI/PPI.

Extended (score-9 additions):
  - FuturesTermStructure  : term structure curve, contango score, roll dates, basis
  - ContinuousContract    : Panama backward-adjusted series + constant-maturity
  - CrossCommodityAnalysis: correlation matrix + spread ratios (crack, gold/silver, corn/wheat)
  - get_futures_dashboard : module-level entry point
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 20.0
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# (name, fred_series_or_None, yf_symbol, unit)
_ENERGY: list[tuple[str, Optional[str], str, str]] = [
    ("Crude Oil (WTI)", "DCOILWTICO", "CL=F", "$/barrel"),
    ("Natural Gas",     "DHHNGSP",   "NG=F", "$/MMBtu"),
    ("Gasoline (RBOB)", None,        "RB=F", "$/gallon"),
    ("Heating Oil",     None,        "HO=F", "$/gallon"),
]
_METALS: list[tuple[str, Optional[str], str, str]] = [
    ("Gold",     "GOLDAMGBD228NLBM", "GC=F", "$/troy oz"),
    ("Silver",   "SLVPRUSD",         "SI=F", "$/troy oz"),
    ("Copper",   "PCOPPUSDM",        "HG=F", "$/metric ton"),
    ("Platinum", None,               "PL=F", "$/troy oz"),
]
_AGRICULTURE: list[tuple[str, Optional[str], str, str]] = [
    ("Corn",     "CORN",  "ZC=F", "$/bushel"),
    ("Wheat",    "WHEAT", "ZW=F", "$/bushel"),
    ("Soybeans", None,    "ZS=F", "$/bushel"),
    ("Coffee",   None,    "KC=F", "$/lb"),
    ("Sugar",    None,    "SB=F", "$/lb"),
]

# commodity key → (name, fred, yf_symbol, unit)
_KEY_MAP: dict[str, tuple[str, Optional[str], str, str]] = {
    "crude_oil":   ("Crude Oil (WTI)", "DCOILWTICO",          "CL=F", "$/barrel"),
    "natural_gas": ("Natural Gas",     "DHHNGSP",             "NG=F", "$/MMBtu"),
    "gasoline":    ("Gasoline (RBOB)", None,                  "RB=F", "$/gallon"),
    "heating_oil": ("Heating Oil",     None,                  "HO=F", "$/gallon"),
    "gold":        ("Gold",     "GOLDAMGBD228NLBM", "GC=F", "$/troy oz"),
    "silver":      ("Silver",   "SLVPRUSD",         "SI=F", "$/troy oz"),
    "copper":      ("Copper",   "PCOPPUSDM",        "HG=F", "$/metric ton"),
    "platinum":    ("Platinum", None,               "PL=F", "$/troy oz"),
    "corn":        ("Corn",     "CORN",  "ZC=F", "$/bushel"),
    "wheat":       ("Wheat",    "WHEAT", "ZW=F", "$/bushel"),
    "soybeans":    ("Soybeans", None,    "ZS=F", "$/bushel"),
    "coffee":      ("Coffee",   None,    "KC=F", "$/lb"),
    "sugar":       ("Sugar",    None,    "SB=F", "$/lb"),
}

# (name, front_contract, next_contract)
_SPREAD_PAIRS: list[tuple[str, str, str]] = [
    ("Crude Oil", "CL=F", "CLM=F"),
    ("Gold",      "GC=F", "GCM=F"),
    ("Corn",      "ZC=F", "ZCH=F"),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CommodityPrice(BaseModel):
    name: str
    symbol: str
    price: Optional[float] = None
    price_unit: str
    change_1d: Optional[float] = None
    change_1m: Optional[float] = None
    change_ytd: Optional[float] = None
    data_source: str
    as_of: str


class CommoditySector(BaseModel):
    sector: str
    commodities: list[CommodityPrice]


class FuturesSpread(BaseModel):
    name: str
    front_month: Optional[float] = None
    next_month: Optional[float] = None
    spread: Optional[float] = None
    structure: str  # "contango" | "backwardation" | "flat"


class CommodityDashboard(BaseModel):
    sectors: list[CommoditySector]
    futures_spreads: list[FuturesSpread]
    commodity_regime: str
    cpi_vs_ppi_gap: Optional[float] = None
    as_of: str
    warnings: list[str] = Field(default_factory=list)


class CommodityHistory(BaseModel):
    commodity: str
    prices: list[dict]
    data_source: str
    unit: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

async def _fred_series(
    client: httpx.AsyncClient, series_id: str, limit_obs: int = 5,
) -> list[tuple[str, float]]:
    """Fetch FRED CSV → last *limit_obs* (date, value) pairs; dots excluded."""
    try:
        r = await client.get(_FRED_CSV, params={"id": series_id}, timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning("FRED non-200", series=series_id, status=r.status_code)
            return []
        result: list[tuple[str, float]] = []
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            dt, val = parts[0].strip(), parts[1].strip()
            if val in (".", "", "NA"):
                continue
            try:
                result.append((dt, float(val)))
            except ValueError:
                continue
        return result[-limit_obs:] if len(result) > limit_obs else result
    except Exception as exc:
        logger.warning("FRED fetch failed", series=series_id, error=str(exc))
        return []


async def _fred_history(
    client: httpx.AsyncClient, series_id: str, days_back: int = 365,
) -> list[tuple[str, float]]:
    """Fetch full FRED series filtered to last *days_back* calendar days."""
    try:
        r = await client.get(_FRED_CSV, params={"id": series_id}, timeout=_TIMEOUT)
        if r.status_code != 200:
            return []
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        result: list[tuple[str, float]] = []
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            dt, val = parts[0].strip(), parts[1].strip()
            if val in (".", "", "NA") or dt < cutoff:
                continue
            try:
                result.append((dt, float(val)))
            except ValueError:
                continue
        return result
    except Exception as exc:
        logger.warning("FRED history failed", series=series_id, error=str(exc))
        return []


# ---------------------------------------------------------------------------
# yfinance helpers (always called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _yf_commodity(symbol: str) -> dict:
    """Return price + pct changes for *symbol* using 1y of history."""
    try:
        import yfinance as yf  # noqa: PLC0415

        hist = yf.Ticker(symbol).history(period="1y", auto_adjust=True)
        if hist.empty:
            return {}
        last = float(hist["Close"].iloc[-1])
        as_of = hist.index[-1].strftime("%Y-%m-%d")

        def _pct(old: float, new: float) -> Optional[float]:
            return round((new - old) / old * 100, 4) if old else None

        c1d = _pct(float(hist["Close"].iloc[-2]), last) if len(hist) >= 2 else None
        c1m = _pct(float(hist["Close"].iloc[-21]), last) if len(hist) >= 21 else None

        ytd = hist[hist.index.year == hist.index[-1].year]
        cytd = _pct(float(ytd["Close"].iloc[0]), last) if not ytd.empty else None

        return {"price": round(last, 4), "change_1d": c1d, "change_1m": c1m,
                "change_ytd": cytd, "as_of": as_of}
    except Exception as exc:
        logger.warning("yf_commodity failed", symbol=symbol, error=str(exc))
        return {}


def _yf_last_close(symbol: str) -> Optional[float]:
    """Return most recent close for *symbol*, or None."""
    try:
        import yfinance as yf  # noqa: PLC0415

        hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception as exc:
        logger.warning("yf_last_close failed", symbol=symbol, error=str(exc))
        return None


def _yf_history(symbol: str, days_back: int = 365) -> list[tuple[str, float]]:
    """Return (date_str, close) pairs from yfinance for the lookback window."""
    try:
        import yfinance as yf  # noqa: PLC0415

        period = (
            "1mo" if days_back <= 30 else "3mo" if days_back <= 90 else
            "6mo" if days_back <= 180 else "1y" if days_back <= 365 else "2y"
        )
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist.empty:
            return []
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        return [
            (ts.strftime("%Y-%m-%d"), round(float(row["Close"]), 4))
            for ts, row in hist.iterrows()
            if ts.strftime("%Y-%m-%d") >= cutoff
        ]
    except Exception as exc:
        logger.warning("yf_history failed", symbol=symbol, error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Price builder (FRED primary, yfinance fallback)
# ---------------------------------------------------------------------------

async def _build_commodity_price(
    client: httpx.AsyncClient,
    name: str,
    fred_series: Optional[str],
    yf_symbol: str,
    unit: str,
) -> CommodityPrice:
    today = date.today().isoformat()

    if fred_series:
        try:
            obs = await _fred_history(client, fred_series, days_back=400)
            if obs:
                obs = sorted(obs, key=lambda x: x[0])
                last_dt, last_px = obs[-1]

                def _pct(old: float, new: float) -> Optional[float]:
                    return round((new - old) / old * 100, 4) if old else None

                c1d = _pct(obs[-2][1], last_px) if len(obs) >= 2 else None

                cutoff_1m = (
                    datetime.fromisoformat(last_dt) - timedelta(days=30)
                ).date().isoformat()
                prev_1m = [o for o in obs if o[0] <= cutoff_1m]
                c1m = _pct(prev_1m[-1][1], last_px) if prev_1m else None

                ytd = [o for o in obs if o[0].startswith(last_dt[:4])]
                cytd = _pct(ytd[0][1], last_px) if ytd else None

                return CommodityPrice(
                    name=name, symbol=fred_series, price=round(last_px, 4),
                    price_unit=unit, change_1d=c1d, change_1m=c1m,
                    change_ytd=cytd, data_source="FRED", as_of=last_dt,
                )
        except Exception as exc:
            logger.warning("FRED price build failed", series=fred_series, error=str(exc))

    # yfinance fallback
    try:
        d = await asyncio.to_thread(_yf_commodity, yf_symbol)
        return CommodityPrice(
            name=name, symbol=yf_symbol,
            price=d.get("price"), price_unit=unit,
            change_1d=d.get("change_1d"), change_1m=d.get("change_1m"),
            change_ytd=d.get("change_ytd"),
            data_source="yfinance", as_of=d.get("as_of", today),
        )
    except Exception as exc:
        logger.warning("yfinance price build failed", symbol=yf_symbol, error=str(exc))

    return CommodityPrice(
        name=name, symbol=yf_symbol, price=None, price_unit=unit,
        data_source="yfinance", as_of=today,
    )


# ---------------------------------------------------------------------------
# Futures spread builder
# ---------------------------------------------------------------------------

async def _build_futures_spread(
    name: str, front_sym: str, next_sym: str
) -> FuturesSpread:
    try:
        front_px, next_px = await asyncio.gather(
            asyncio.to_thread(_yf_last_close, front_sym),
            asyncio.to_thread(_yf_last_close, next_sym),
        )
        if front_px is not None and next_px is not None:
            spread = round(next_px - front_px, 4)
            structure = (
                "contango" if spread > 0.01 else
                "backwardation" if spread < -0.01 else "flat"
            )
            return FuturesSpread(
                name=name,
                front_month=round(front_px, 4),
                next_month=round(next_px, 4),
                spread=spread,
                structure=structure,
            )
    except Exception as exc:
        logger.warning("Spread build failed", name=name, error=str(exc))
    return FuturesSpread(name=name, structure="flat")


# ---------------------------------------------------------------------------
# Regime detection (CPI + PPI YoY via FRED)
# ---------------------------------------------------------------------------

async def _build_regime(
    client: httpx.AsyncClient,
) -> tuple[str, Optional[float], list[str]]:
    """
    Returns (regime, cpi_vs_ppi_gap, warnings).
    inflationary = CPI YoY > 3% AND PPI YoY > 5%
    deflationary = CPI YoY < 1% AND PPI YoY < 1%
    cpi_vs_ppi_gap = ppi_yoy - cpi_yoy (positive → producer squeeze)
    """
    warnings: list[str] = []
    cpi_obs, ppi_obs = await asyncio.gather(
        _fred_series(client, "CPIAUCSL", limit_obs=14),
        _fred_series(client, "PPIACO",   limit_obs=14),
    )

    def _yoy(obs: list[tuple[str, float]]) -> Optional[float]:
        if len(obs) < 2:
            return None
        latest = obs[-1][1]
        year_ago = obs[-13][1] if len(obs) >= 13 else obs[0][1]
        return round((latest - year_ago) / year_ago * 100, 4) if year_ago else None

    cpi_yoy, ppi_yoy = _yoy(cpi_obs), _yoy(ppi_obs)
    if cpi_yoy is None:
        warnings.append("CPI data unavailable — regime detection degraded")
    if ppi_yoy is None:
        warnings.append("PPI data unavailable — regime detection degraded")

    regime = "neutral"
    if cpi_yoy is not None and ppi_yoy is not None:
        if cpi_yoy > 3.0 and ppi_yoy > 5.0:
            regime = "inflationary"
        elif cpi_yoy < 1.0 and ppi_yoy < 1.0:
            regime = "deflationary"

    gap = round(ppi_yoy - cpi_yoy, 4) if (cpi_yoy is not None and ppi_yoy is not None) else None
    return regime, gap, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_commodity_dashboard() -> CommodityDashboard:
    """Fetch all sectors, spreads, and regime in parallel; return dashboard."""
    today = date.today().isoformat()
    all_warnings: list[str] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        energy_tasks      = [_build_commodity_price(client, *row) for row in _ENERGY]
        metals_tasks      = [_build_commodity_price(client, *row) for row in _METALS]
        agriculture_tasks = [_build_commodity_price(client, *row) for row in _AGRICULTURE]
        spread_tasks      = [_build_futures_spread(*row) for row in _SPREAD_PAIRS]

        energy_px, metals_px, agri_px, spreads, regime_result = await asyncio.gather(
            asyncio.gather(*energy_tasks),
            asyncio.gather(*metals_tasks),
            asyncio.gather(*agriculture_tasks),
            asyncio.gather(*spread_tasks),
            _build_regime(client),
        )

    regime, gap, regime_warnings = regime_result
    all_warnings.extend(regime_warnings)

    for cp in [*energy_px, *metals_px, *agri_px]:
        if cp.price is None:
            all_warnings.append(f"No price for {cp.name} ({cp.symbol})")

    return CommodityDashboard(
        sectors=[
            CommoditySector(sector="Energy",      commodities=list(energy_px)),
            CommoditySector(sector="Metals",      commodities=list(metals_px)),
            CommoditySector(sector="Agriculture", commodities=list(agri_px)),
        ],
        futures_spreads=list(spreads),
        commodity_regime=regime,
        cpi_vs_ppi_gap=gap,
        as_of=today,
        warnings=all_warnings,
    )


async def get_commodity_history(
    commodity: str,
    days_back: int = 365,
) -> CommodityHistory:
    """
    Return daily price history for a single commodity.

    *commodity* must be one of: crude_oil, natural_gas, gasoline, heating_oil,
    gold, silver, copper, platinum, corn, wheat, soybeans, coffee, sugar.
    Tries FRED first; falls back to yfinance.
    """
    entry = _KEY_MAP.get(commodity.lower())
    if entry is None:
        return CommodityHistory(
            commodity=commodity, prices=[], data_source="none", unit="",
            warnings=[
                f"Unknown commodity '{commodity}'. "
                f"Available: {', '.join(sorted(_KEY_MAP))}"
            ],
        )

    name, fred_series, yf_symbol, unit = entry
    warnings: list[str] = []

    if fred_series:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                obs = await _fred_history(client, fred_series, days_back=days_back)
            if obs:
                return CommodityHistory(
                    commodity=name,
                    prices=[{"date": dt, "price": val} for dt, val in obs],
                    data_source="FRED", unit=unit, warnings=warnings,
                )
            warnings.append(f"FRED empty for {fred_series}; using yfinance")
        except Exception as exc:
            warnings.append(f"FRED history failed for {fred_series}: {exc}")

    try:
        obs = await asyncio.to_thread(_yf_history, yf_symbol, days_back)
        return CommodityHistory(
            commodity=name,
            prices=[{"date": dt, "price": val} for dt, val in obs],
            data_source="yfinance", unit=unit, warnings=warnings,
        )
    except Exception as exc:
        warnings.append(f"yfinance history failed for {yf_symbol}: {exc}")
        return CommodityHistory(
            commodity=name, prices=[], data_source="yfinance",
            unit=unit, warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Futures term structure helpers
# ---------------------------------------------------------------------------

# Front-4 contract symbols for major commodities (yfinance naming).
# yfinance doesn't expose deferred months as separate tickers in a reliable way,
# so we use the front contract + approximate quarterly expiries via the known
# CME suffix convention (F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul,
# Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec).
_MONTH_CODES = "FGHJKMNQUVXZ"  # index 0 → January

# Key → (name, front yf symbol, spot yf symbol or None)
_FUTURES_MAP: dict[str, tuple[str, str, Optional[str]]] = {
    "crude_oil":   ("Crude Oil (WTI)",  "CL=F",  None),
    "natural_gas": ("Natural Gas",       "NG=F",  None),
    "gold":        ("Gold",              "GC=F",  "GLD"),
    "silver":      ("Silver",            "SI=F",  "SLV"),
    "copper":      ("Copper",            "HG=F",  "CPER"),
    "corn":        ("Corn",              "ZC=F",  None),
    "wheat":       ("Wheat",             "ZW=F",  None),
    "soybeans":    ("Soybeans",          "ZS=F",  None),
    "sp500":       ("S&P 500",           "ES=F",  "SPY"),
    "nasdaq":      ("Nasdaq 100",        "NQ=F",  "QQQ"),
}

# 10 major futures tickers for cross-commodity correlation
_CROSS_COMMODITY_TICKERS: list[tuple[str, str]] = [
    ("Crude Oil",    "CL=F"),
    ("Natural Gas",  "NG=F"),
    ("Gold",         "GC=F"),
    ("Silver",       "SI=F"),
    ("Copper",       "HG=F"),
    ("Corn",         "ZC=F"),
    ("Wheat",        "ZW=F"),
    ("Soybeans",     "ZS=F"),
    ("S&P 500",      "ES=F"),
    ("30yr Bond",    "ZB=F"),
]


def _next_contract_symbols(root: str, n: int = 4) -> list[str]:
    """
    Generate the next *n* quarterly contract tickers for *root* (e.g. 'CL').
    Uses CME quarterly cycle (Mar/Jun/Sep/Dec) rolling forward from today.
    Returns yfinance-style symbols like 'CLM25'.
    """
    quarterly = {2, 5, 8, 11}  # Mar=2, Jun=5, Sep=8, Dec=11 (0-indexed months)
    today = date.today()
    symbols: list[str] = []
    # Iterate over the next 24 calendar months from today
    start_month = today.month
    start_year  = today.year
    for offset in range(24):
        m = (start_month - 1 + offset) % 12  # 0-indexed
        y = start_year + (start_month - 1 + offset) // 12
        if m in quarterly:
            code = _MONTH_CODES[m]
            yr2 = str(y % 100).zfill(2)
            symbols.append(f"{root}{code}{yr2}")
        if len(symbols) >= n:
            break
    return symbols


def _yf_term_structure(root: str, front_sym: str) -> pd.DataFrame:
    """
    Fetch front contract + next 3 quarterly contracts.
    Returns DataFrame with columns [tenor_months, symbol, last_price, volume].
    """
    import yfinance as yf  # noqa: PLC0415

    contract_syms = _next_contract_symbols(root, n=3)
    all_syms = [front_sym] + contract_syms
    rows: list[dict] = []
    for i, sym in enumerate(all_syms):
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d", auto_adjust=True)
            if hist.empty:
                continue
            last_px = float(hist["Close"].iloc[-1])
            vol = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0.0
            rows.append({
                "tenor_months": i * 3,
                "symbol": sym,
                "last_price": round(last_px, 4),
                "volume": vol,
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["tenor_months", "symbol", "last_price", "volume"])
    return pd.DataFrame(rows).sort_values("tenor_months").reset_index(drop=True)


def _yf_full_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV history from yfinance as a DataFrame indexed by date.
    Returns empty DataFrame on failure.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return hist
    except Exception as exc:
        logger.warning("yf_full_history failed", symbol=symbol, error=str(exc))
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Nelson-Siegel curve fitting
# ---------------------------------------------------------------------------

def _nelson_siegel(tau: np.ndarray, beta0: float, beta1: float,
                   beta2: float, lam: float) -> np.ndarray:
    """Nelson-Siegel yield curve formula applied to futures term (months)."""
    t = tau / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        factor1 = np.where(t == 0, 1.0, (1 - np.exp(-t)) / t)
        factor2 = factor1 - np.exp(-t)
    return beta0 + beta1 * factor1 + beta2 * factor2


def _fit_ns(tenors: np.ndarray, prices: np.ndarray) -> dict:
    """
    Fit Nelson-Siegel to (tenor_months, price) pairs.
    Returns dict with beta0/beta1/beta2/lambda and r_squared.
    Falls back to linear fit if scipy unavailable or optimisation fails.
    """
    try:
        from scipy.optimize import curve_fit  # noqa: PLC0415

        p0 = [float(prices[-1]), float(prices[0] - prices[-1]), 0.0, 12.0]
        bounds = ([-np.inf, -np.inf, -np.inf, 0.01], [np.inf, np.inf, np.inf, 120.0])
        popt, _ = curve_fit(
            _nelson_siegel, tenors, prices,
            p0=p0, bounds=bounds, maxfev=5000,
        )
        fitted = _nelson_siegel(tenors, *popt)
        ss_res = float(np.sum((prices - fitted) ** 2))
        ss_tot = float(np.sum((prices - prices.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return {
            "method": "nelson_siegel",
            "beta0": round(float(popt[0]), 6),
            "beta1": round(float(popt[1]), 6),
            "beta2": round(float(popt[2]), 6),
            "lambda": round(float(popt[3]), 6),
            "r_squared": round(r2, 6),
        }
    except Exception:
        pass

    # Linear fallback
    if len(tenors) >= 2:
        slope = float(np.polyfit(tenors, prices, 1)[0])
    else:
        slope = 0.0
    return {
        "method": "linear_fallback",
        "slope_per_month": round(slope, 6),
        "r_squared": None,
    }


# ---------------------------------------------------------------------------
# FuturesTermStructure
# ---------------------------------------------------------------------------

class FuturesTermStructure:
    """
    Term structure analysis for a single commodity futures curve.

    Parameters
    ----------
    commodity : str
        Key from _FUTURES_MAP (e.g. 'crude_oil', 'gold').

    Usage
    -----
        fts = FuturesTermStructure("crude_oil")
        df  = fts.fetch_term_structure()
        score = fts.compute_contango_score()
        ns    = fts.fit_term_structure_curve()
        rolls = fts.detect_roll_dates()
        basis = fts.compute_basis()
    """

    def __init__(self, commodity: str) -> None:
        entry = _FUTURES_MAP.get(commodity.lower())
        if entry is None:
            raise ValueError(
                f"Unknown commodity '{commodity}'. "
                f"Available: {sorted(_FUTURES_MAP)}"
            )
        self.commodity = commodity.lower()
        self.name, self.front_sym, self.spot_sym = entry
        # root = letters before '=' (e.g. 'CL' from 'CL=F')
        self.root = self.front_sym.split("=")[0]
        self._curve: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def fetch_term_structure(self, commodity: Optional[str] = None) -> pd.DataFrame:  # noqa: ARG002
        """
        Fetch front 4 contracts and return a DataFrame:
        [tenor_months, symbol, last_price, volume].

        *commodity* parameter is accepted for API compatibility but ignored
        (the instance is already bound to one commodity).
        """
        df = _yf_term_structure(self.root, self.front_sym)
        self._curve = df
        return df

    # ------------------------------------------------------------------
    def compute_contango_score(self) -> float:
        """
        Roll-adjusted annualised roll yield (%).

        Defined as the annualised return of rolling the front contract into
        the next:  ((P_next / P_front) - 1) * (12 / tenor_gap_months) * 100

        Positive → backwardation (roll yield positive for longs).
        Negative → contango (roll yield negative for longs).
        Returns 0.0 if term structure has fewer than 2 points.
        """
        df = self._curve if self._curve is not None else self.fetch_term_structure()
        if len(df) < 2:
            return 0.0
        front_px  = float(df.iloc[0]["last_price"])
        next_px   = float(df.iloc[1]["last_price"])
        tenor_gap = float(df.iloc[1]["tenor_months"] - df.iloc[0]["tenor_months"])
        if front_px <= 0 or tenor_gap <= 0:
            return 0.0
        roll_yield = ((next_px / front_px) - 1.0) * (12.0 / tenor_gap) * 100.0
        return round(roll_yield, 4)

    # ------------------------------------------------------------------
    def fit_term_structure_curve(self) -> dict:
        """
        Fit a Nelson-Siegel curve to the futures term structure.

        Returns a dict with curve parameters and R².
        Falls back to linear fit if scipy is unavailable or curve has < 3 pts.
        """
        df = self._curve if self._curve is not None else self.fetch_term_structure()
        if df.empty:
            return {"method": "none", "error": "no term structure data"}
        tenors = df["tenor_months"].to_numpy(dtype=float)
        prices = df["last_price"].to_numpy(dtype=float)
        result = _fit_ns(tenors, prices)
        result["commodity"] = self.name
        result["n_points"]  = len(df)
        return result

    # ------------------------------------------------------------------
    def detect_roll_dates(self, lookback_days: int = 365) -> list[date]:  # noqa: ARG002
        """
        Identify contract roll dates from volume data.

        A roll date is detected when daily volume of the front contract drops
        by more than 50 % relative to its 5-day trailing average — indicating
        traders have migrated to the next contract.

        *lookback_days* is reserved for future use; the underlying yfinance
        call always fetches 1 year of history.

        Returns a list of date objects (may be empty if data unavailable).
        """
        hist = _yf_full_history(self.front_sym, period="1y")
        if hist.empty or "Volume" not in hist.columns:
            return []
        vol = hist["Volume"].dropna()
        if len(vol) < 6:
            return []
        roll_dates: list[date] = []
        rolling_avg = vol.rolling(5).mean()
        for i in range(5, len(vol)):
            avg = float(rolling_avg.iloc[i - 1])
            cur = float(vol.iloc[i])
            if avg > 0 and cur < avg * 0.50:
                roll_dates.append(vol.index[i].date())
        return roll_dates

    # ------------------------------------------------------------------
    def compute_basis(self, spot_ticker: Optional[str] = None) -> float:
        """
        Compute basis = futures_front_price - spot_price.

        Uses *spot_ticker* if provided, otherwise falls back to the instance's
        configured spot symbol.  Returns 0.0 if spot data unavailable.
        """
        sym = spot_ticker or self.spot_sym
        if sym is None:
            logger.warning("No spot ticker for basis computation", commodity=self.commodity)
            return 0.0
        try:
            import yfinance as yf  # noqa: PLC0415

            spot_hist = yf.Ticker(sym).history(period="5d", auto_adjust=True)
            if spot_hist.empty:
                return 0.0
            spot_px = float(spot_hist["Close"].iloc[-1])

            df = self._curve if self._curve is not None else self.fetch_term_structure()
            if df.empty:
                return 0.0
            front_px = float(df.iloc[0]["last_price"])
            return round(front_px - spot_px, 4)
        except Exception as exc:
            logger.warning("compute_basis failed", commodity=self.commodity, error=str(exc))
            return 0.0


# ---------------------------------------------------------------------------
# ContinuousContract
# ---------------------------------------------------------------------------

class ContinuousContract:
    """
    Build backward-adjusted (Panama-stitched) continuous price series and
    constant-maturity interpolated series for a commodity futures contract.

    Parameters
    ----------
    commodity : str
        Key from _FUTURES_MAP.
    """

    def __init__(self, commodity: str) -> None:
        entry = _FUTURES_MAP.get(commodity.lower())
        if entry is None:
            raise ValueError(f"Unknown commodity '{commodity}'.")
        self.commodity = commodity.lower()
        self.name, self.front_sym, _ = entry
        self.root = self.front_sym.split("=")[0]

    # ------------------------------------------------------------------
    def panama_stitch(self, lookback_days: int = 730) -> pd.DataFrame:
        """
        Construct a backward-adjusted (Panama method) continuous price series.

        Algorithm:
          1. Fetch front-contract OHLCV.
          2. Identify roll dates (volume-based, same logic as FuturesTermStructure).
          3. At each roll, compute the price gap between expiring and new front
             and apply a cumulative backward adjustment so the series is
             gap-free at the roll point.

        Returns DataFrame indexed by date with columns:
          [raw_close, adjusted_close, cumulative_adjustment].
        """
        hist = _yf_full_history(self.front_sym, period="2y")
        if hist.empty:
            return pd.DataFrame(columns=["raw_close", "adjusted_close", "cumulative_adjustment"])

        hist = hist[["Close"]].copy()
        hist.columns = ["raw_close"]
        hist = hist.sort_index()

        # Volume-based roll detection
        fts = FuturesTermStructure(self.commodity)
        roll_dates_list = fts.detect_roll_dates()
        roll_date_set: set[date] = set(roll_dates_list)

        # Build cumulative Panama adjustment (backward, so future prices unchanged)
        adj = hist["raw_close"].copy()
        cumulative_shift = 0.0
        adjustments: list[float] = [0.0] * len(adj)

        idx_list = list(adj.index)
        for i in range(len(idx_list) - 1, 0, -1):
            dt = idx_list[i].date()
            prev_dt = idx_list[i - 1].date()
            if dt in roll_date_set or prev_dt in roll_date_set:
                gap = float(adj.iloc[i]) - float(adj.iloc[i - 1])
                # If gap is large relative to price, treat as roll gap
                if abs(gap) / (float(adj.iloc[i - 1]) + 1e-9) > 0.005:
                    cumulative_shift -= gap
            adjustments[i - 1] = cumulative_shift

        hist["cumulative_adjustment"] = adjustments
        hist["adjusted_close"] = hist["raw_close"] + hist["cumulative_adjustment"]

        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        mask = hist.index.strftime("%Y-%m-%d") >= cutoff
        return hist[mask].round(4)

    # ------------------------------------------------------------------
    def constant_maturity_series(
        self, maturities: list[int] = None, lookback_days: int = 365  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Construct constant-maturity futures price series by linear interpolation
        between available contract tenors.

        Parameters
        ----------
        maturities : list[int]
            Target maturities in days (default: [30, 60, 90]).
        lookback_days : int
            Not used directly (interpolation is point-in-time); kept for API
            consistency. The term structure is fetched fresh each call.

        Returns DataFrame with columns for each maturity:
          [price_30d, price_60d, price_90d] (or whichever maturities requested).
        """
        if maturities is None:
            maturities = [30, 60, 90]

        fts = FuturesTermStructure(self.commodity)
        df = fts.fetch_term_structure()
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        # Convert tenor_months to days for interpolation
        tenors_days = (df["tenor_months"] * 30.44).to_numpy(dtype=float)
        prices      = df["last_price"].to_numpy(dtype=float)

        today_str = date.today().isoformat()
        row: dict[str, Any] = {"date": today_str}
        for mat in maturities:
            price = float(np.interp(float(mat), tenors_days, prices))
            row[f"price_{mat}d"] = round(price, 4)

        return pd.DataFrame([row]).set_index("date")


# ---------------------------------------------------------------------------
# CrossCommodityAnalysis
# ---------------------------------------------------------------------------

class CrossCommodityAnalysis:
    """
    Cross-commodity correlation matrix and spread ratio analysis for 10 major
    futures markets.

    Usage
    -----
        cca = CrossCommodityAnalysis()
        corr = cca.correlation_matrix(lookback_days=252)
        spreads = cca.spread_ratios()
    """

    def __init__(self) -> None:
        self._prices: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def _fetch_all_prices(self, lookback_days: int = 252) -> pd.DataFrame:
        """
        Fetch closing prices for all 10 tickers and return a wide DataFrame
        indexed by date.  Missing dates are forward-filled then dropped.
        """
        if self._prices is not None:
            return self._prices

        try:
            import yfinance as yf  # noqa: PLC0415

            tickers = [sym for _, sym in _CROSS_COMMODITY_TICKERS]
            period  = "1y" if lookback_days <= 365 else "2y"
            raw     = yf.download(tickers, period=period, auto_adjust=True,
                                  progress=False, threads=True)
            if raw.empty:
                return pd.DataFrame()

            # yf.download with multiple tickers → MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = raw[["Close"]]

            closes.columns = [
                name for name, _ in _CROSS_COMMODITY_TICKERS
                if _ in (closes.columns.tolist() if not isinstance(raw.columns, pd.MultiIndex)
                         else raw["Close"].columns.tolist())
            ]
            # Re-map column names robustly
            ticker_to_name = {sym: name for name, sym in _CROSS_COMMODITY_TICKERS}
            closes = closes.rename(columns=ticker_to_name)
            closes = closes.ffill().dropna(how="all")

            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            closes = closes[closes.index.strftime("%Y-%m-%d") >= cutoff]

            self._prices = closes
            return closes
        except Exception as exc:
            logger.warning("CrossCommodityAnalysis fetch failed", error=str(exc))
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def correlation_matrix(self, lookback_days: int = 252) -> dict:
        """
        Compute pairwise Pearson correlation matrix on daily log returns
        for the 10 major futures.

        Parameters
        ----------
        lookback_days : int
            Trading-day lookback window (default 252 ≈ 1 year).

        Returns
        -------
        dict with keys:
          'labels'  : list of commodity names
          'matrix'  : list[list[float]] (row-major, same order as labels)
          'as_of'   : ISO date string
          'warnings': list of any degradation notes
        """
        warnings: list[str] = []
        closes = self._fetch_all_prices(lookback_days)
        if closes.empty:
            return {"labels": [], "matrix": [], "as_of": date.today().isoformat(),
                    "warnings": ["No price data fetched"]}

        log_ret = np.log(closes / closes.shift(1)).dropna()
        corr_df = log_ret.corr()

        labels = list(corr_df.columns)
        matrix = [[round(float(corr_df.loc[r, c]), 4) for c in labels] for r in labels]

        if len(labels) < len(_CROSS_COMMODITY_TICKERS):
            missing = set(n for n, _ in _CROSS_COMMODITY_TICKERS) - set(labels)
            warnings.append(f"Missing tickers (data unavailable): {sorted(missing)}")

        return {
            "labels":   labels,
            "matrix":   matrix,
            "as_of":    date.today().isoformat(),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    def spread_ratios(self) -> dict:
        """
        Compute key commodity spread ratios:
          - crack_spread_321   : 3-2-1 crack spread (3×crude → 2×gasoline + 1×heating oil)
          - gold_silver_ratio  : gold price / silver price
          - corn_wheat_ratio   : corn price / wheat price
          - crude_natgas_ratio : crude oil price / natural gas price (energy equivalence)

        All prices fetched live from yfinance (5-day window, last close).

        Returns dict with each ratio and its constituent prices.
        """
        tickers = {
            "crude":       "CL=F",
            "gasoline":    "RB=F",
            "heating_oil": "HO=F",
            "gold":        "GC=F",
            "silver":      "SI=F",
            "corn":        "ZC=F",
            "wheat":       "ZW=F",
            "natgas":      "NG=F",
        }
        px: dict[str, Optional[float]] = {}
        for key, sym in tickers.items():
            px[key] = _yf_last_close(sym)

        warnings: list[str] = []

        def _safe_ratio(num: Optional[float], den: Optional[float],
                        label: str) -> Optional[float]:
            if num is None or den is None or den == 0:
                warnings.append(f"Insufficient data for {label}")
                return None
            return round(num / den, 4)

        # 3-2-1 crack spread: (2 × gasoline + 1 × heating oil - 3 × crude) / 3
        # Prices in $/gallon → multiply by 42 to convert to $/barrel
        crack: Optional[float] = None
        if all(px[k] is not None for k in ("crude", "gasoline", "heating_oil")):
            crack = round(
                (2 * px["gasoline"] * 42 + 1 * px["heating_oil"] * 42  # type: ignore[operator]
                 - 3 * px["crude"]) / 3,
                4,
            )
        else:
            warnings.append("Insufficient data for crack_spread_321")

        return {
            "crack_spread_321": {
                "value": crack,
                "description": "3-2-1 crack spread ($/barrel)",
                "crude_px":       px["crude"],
                "gasoline_px":    px["gasoline"],
                "heating_oil_px": px["heating_oil"],
            },
            "gold_silver_ratio": {
                "value": _safe_ratio(px["gold"], px["silver"], "gold_silver_ratio"),
                "description": "Gold / Silver price ratio",
                "gold_px":   px["gold"],
                "silver_px": px["silver"],
            },
            "corn_wheat_ratio": {
                "value": _safe_ratio(px["corn"], px["wheat"], "corn_wheat_ratio"),
                "description": "Corn / Wheat price ratio",
                "corn_px":  px["corn"],
                "wheat_px": px["wheat"],
            },
            "crude_natgas_ratio": {
                "value": _safe_ratio(px["crude"], px["natgas"], "crude_natgas_ratio"),
                "description": "Crude Oil / Natural Gas ratio",
                "crude_px":  px["crude"],
                "natgas_px": px["natgas"],
            },
            "as_of":    date.today().isoformat(),
            "warnings": warnings,
        }


# ---------------------------------------------------------------------------
# Module-level entry point: get_futures_dashboard
# ---------------------------------------------------------------------------

async def get_futures_dashboard(commodity: str = "crude_oil") -> dict:
    """
    Comprehensive futures term structure dashboard for a single commodity.

    Runs in parallel:
      1. Term structure fetch (front 4 contracts)
      2. Contango score
      3. Nelson-Siegel curve fit
      4. Roll date detection
      5. Basis vs spot
      6. Panama-stitched continuous contract (last 5 rows summary)
      7. Constant-maturity series (30/60/90-day)
      8. Cross-commodity correlation matrix (252-day)
      9. Spread ratios (crack, gold/silver, corn/wheat, crude/natgas)

    Parameters
    ----------
    commodity : str
        Any key from _FUTURES_MAP (default 'crude_oil').

    Returns
    -------
    dict with all sub-results and a top-level 'warnings' list.
    """
    warnings: list[str] = []

    # ---- term structure + derivatives (sync work in thread) ----
    def _term_structure_block() -> dict:
        fts = FuturesTermStructure(commodity)
        curve_df = fts.fetch_term_structure()
        contango  = fts.compute_contango_score()
        ns_fit    = fts.fit_term_structure_curve()
        rolls     = [str(d) for d in fts.detect_roll_dates()]
        basis     = fts.compute_basis()
        curve_records = curve_df.to_dict(orient="records") if not curve_df.empty else []
        return {
            "term_structure":    curve_records,
            "contango_score":    contango,
            "curve_fit":         ns_fit,
            "roll_dates":        rolls,
            "basis_vs_spot":     basis,
        }

    def _continuous_block() -> dict:
        cc = ContinuousContract(commodity)
        panama_df = cc.panama_stitch(lookback_days=365)
        cm_df     = cc.constant_maturity_series([30, 60, 90])
        # Return last 5 rows of panama series to keep payload small
        panama_tail = (
            panama_df.tail(5).reset_index().rename(columns={"index": "date"})
            .to_dict(orient="records")
            if not panama_df.empty else []
        )
        cm_records = cm_df.reset_index().to_dict(orient="records") if not cm_df.empty else []
        return {
            "panama_series_tail":       panama_tail,
            "constant_maturity_series": cm_records,
        }

    def _cross_commodity_block() -> dict:
        cca = CrossCommodityAnalysis()
        corr   = cca.correlation_matrix(lookback_days=252)
        ratios = cca.spread_ratios()
        return {"correlation_matrix": corr, "spread_ratios": ratios}

    # Run all three blocks concurrently in threads
    ts_result, cont_result, cross_result = await asyncio.gather(
        asyncio.to_thread(_term_structure_block),
        asyncio.to_thread(_continuous_block),
        asyncio.to_thread(_cross_commodity_block),
    )

    # Collect sub-warnings
    for block in (ts_result, cont_result, cross_result):
        for sub in block.values():
            if isinstance(sub, dict) and "warnings" in sub:
                warnings.extend(sub["warnings"])

    return {
        "commodity":                ts_result["term_structure"] and commodity or commodity,
        "term_structure":           ts_result["term_structure"],
        "contango_score":           ts_result["contango_score"],
        "curve_fit":                ts_result["curve_fit"],
        "roll_dates":               ts_result["roll_dates"],
        "basis_vs_spot":            ts_result["basis_vs_spot"],
        "panama_series_tail":       cont_result["panama_series_tail"],
        "constant_maturity_series": cont_result["constant_maturity_series"],
        "correlation_matrix":       cross_result["correlation_matrix"],
        "spread_ratios":            cross_result["spread_ratios"],
        "as_of":                    date.today().isoformat(),
        "warnings":                 warnings,
    }
