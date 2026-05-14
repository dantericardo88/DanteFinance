"""
Commodity Analytics — Dimension #48.

Multi-sector commodity dashboard (FRED + yfinance). Covers Energy, Metals,
Agriculture, futures spread structure, and regime detection via CPI/PPI.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
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
