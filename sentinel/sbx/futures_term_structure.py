"""Futures term structure analytics — contango/backwardation detection, roll yield,
continuous contract stitching, and trading signals for commodities, VIX, and rates."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, Literal

import numpy as np
import pandas as pd
import httpx
import yfinance as yf
from pydantic import BaseModel, Field
from scipy.interpolate import CubicSpline

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
REVERSE_CODES = {v: k for k, v in MONTH_CODES.items()}

COMMODITY_CONFIG = {
    "wti":      {"prefix": "CL",  "exchange": "NYM", "months": [1, 2, 3, 4, 5, 6, 9, 12]},
    "gold":     {"prefix": "GC",  "exchange": "CMX", "months": [2, 4, 6, 8, 10, 12]},
    "silver":   {"prefix": "SI",  "exchange": "CMX", "months": [3, 5, 7, 9, 12]},
    "corn":     {"prefix": "ZC",  "exchange": "CBT", "months": [3, 5, 7, 9, 12]},
    "wheat":    {"prefix": "ZW",  "exchange": "CBT", "months": [3, 5, 7, 9, 12]},
    "soybeans": {"prefix": "ZS",  "exchange": "CBT", "months": [1, 3, 5, 7, 8, 9, 11]},
    "copper":   {"prefix": "HG",  "exchange": "CMX", "months": [3, 5, 7, 9, 12]},
    "nat_gas":  {"prefix": "NG",  "exchange": "NYM", "months": list(range(1, 13))},
}

CBOE_VIX_FUTURES_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VX_History.csv"

_SLOPE_CONTANGO_THRESHOLD = 0.005
_SLOPE_BACKWARDATION_THRESHOLD = -0.005
_STRONG_CONTANGO_THRESHOLD = 0.015
_STRONG_BACKWARDATION_THRESHOLD = -0.015


class FuturesContract(BaseModel):
    symbol: str
    commodity: str
    expiry_month: int
    expiry_year: int
    days_to_expiry: int
    price: float
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    basis: Optional[float] = None


class TermStructure(BaseModel):
    commodity: str
    as_of: date
    spot_price: Optional[float] = None
    contracts: list[FuturesContract]
    structure: Literal["contango", "backwardation", "flat"]
    front_back_spread: float
    annualized_roll_cost_pct: float
    term_structure_slope: float


class RollYield(BaseModel):
    commodity: str
    roll_date: date
    front_price: float
    next_price: float
    days_to_roll: int
    spot_return: float
    roll_return: float
    roll_yield_annualized: float


class ContinuousSeries(BaseModel):
    commodity: str
    method: Literal["panama", "ratio", "nearest"]
    start_date: date
    end_date: date
    prices: dict[str, float]
    adjustment_factors: dict[str, float]


class TermStructureSignal(BaseModel):
    commodity: str
    as_of: date
    signal: float
    signal_label: str
    roll_cost_bps: float
    percentile_30d: Optional[float] = None


class FuturesTermStructure:
    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout

    async def get_vix_futures_curve(self) -> TermStructure:
        today = date.today()
        raw = await self._download_vix_csv()

        raw["Trade Date"] = pd.to_datetime(raw["Trade Date"], errors="coerce")
        raw = raw.dropna(subset=["Trade Date"])
        latest_date = raw["Trade Date"].max()
        latest = raw[raw["Trade Date"] == latest_date].copy()

        contracts: list[FuturesContract] = []
        for _, row in latest.iterrows():
            futures_label = str(row.get("Futures", "")).strip()
            if not futures_label or futures_label == "nan":
                continue
            try:
                settle = float(row.get("Settle", row.get("Close", np.nan)))
                if np.isnan(settle) or settle <= 0:
                    continue
            except (TypeError, ValueError):
                continue

            month_num, year_num = _parse_vix_futures_label(futures_label)
            if month_num is None or year_num is None:
                continue

            expiry_date = date(year_num, month_num, 1) + timedelta(days=14)
            dte = max(0, (expiry_date - today).days)

            try:
                oi = int(row.get("Open Interest", 0) or 0)
            except (TypeError, ValueError):
                oi = None
            try:
                vol = int(row.get("Total Volume", 0) or 0)
            except (TypeError, ValueError):
                vol = None

            contracts.append(FuturesContract(
                symbol=f"VX_{futures_label}",
                commodity="vix",
                expiry_month=month_num,
                expiry_year=year_num,
                days_to_expiry=dte,
                price=settle,
                open_interest=oi,
                volume=vol,
            ))

        contracts.sort(key=lambda c: c.days_to_expiry)
        contracts = contracts[:8]

        if not contracts:
            return TermStructure(
                commodity="vix",
                as_of=today,
                contracts=[],
                structure="flat",
                front_back_spread=0.0,
                annualized_roll_cost_pct=0.0,
                term_structure_slope=0.0,
            )

        metrics = self.compute_contango_backwardation(contracts)

        spot_ticker = yf.Ticker("^VIX")
        spot_price: Optional[float] = None
        try:
            hist = spot_ticker.history(period="1d")
            if not hist.empty:
                spot_price = float(hist["Close"].iloc[-1])
        except Exception:
            pass

        if spot_price is not None:
            for c in contracts:
                c.basis = c.price - spot_price

        return TermStructure(
            commodity="vix",
            as_of=today,
            spot_price=spot_price,
            contracts=contracts,
            structure=metrics["structure"],
            front_back_spread=metrics["front_back_spread"],
            annualized_roll_cost_pct=metrics["annualized_roll_cost_pct"],
            term_structure_slope=metrics["slope"],
        )

    async def get_commodity_curve(
        self, commodity: str, n_contracts: int = 6
    ) -> TermStructure:
        today = date.today()
        if commodity not in COMMODITY_CONFIG:
            raise ValueError(f"Unknown commodity: {commodity}. Valid: {list(COMMODITY_CONFIG)}")

        symbols_info = self._build_symbols(commodity, n_contracts)

        prices_coros = [self._fetch_price(sym) for sym, _, _ in symbols_info]
        prices_raw = await asyncio.gather(*prices_coros, return_exceptions=True)

        contracts: list[FuturesContract] = []
        for (sym, month, year), price_result in zip(symbols_info, prices_raw):
            if isinstance(price_result, Exception) or price_result is None:
                continue
            price = price_result
            if price <= 0:
                continue
            expiry_date = date(year, month, 1) + timedelta(days=14)
            dte = max(0, (expiry_date - today).days)
            contracts.append(FuturesContract(
                symbol=sym,
                commodity=commodity,
                expiry_month=month,
                expiry_year=year,
                days_to_expiry=dte,
                price=price,
            ))

        contracts.sort(key=lambda c: c.days_to_expiry)

        spot_price: Optional[float] = None
        cfg = COMMODITY_CONFIG[commodity]
        spot_sym = cfg["prefix"] + "=F"
        spot_price = await self._fetch_price(spot_sym)

        if spot_price is not None and spot_price > 0:
            for c in contracts:
                c.basis = c.price - spot_price
        elif contracts:
            spot_price = contracts[0].price

        if not contracts:
            return TermStructure(
                commodity=commodity,
                as_of=today,
                spot_price=spot_price,
                contracts=[],
                structure="flat",
                front_back_spread=0.0,
                annualized_roll_cost_pct=0.0,
                term_structure_slope=0.0,
            )

        metrics = self.compute_contango_backwardation(contracts)

        return TermStructure(
            commodity=commodity,
            as_of=today,
            spot_price=spot_price,
            contracts=contracts,
            structure=metrics["structure"],
            front_back_spread=metrics["front_back_spread"],
            annualized_roll_cost_pct=metrics["annualized_roll_cost_pct"],
            term_structure_slope=metrics["slope"],
        )

    def compute_contango_backwardation(self, contracts: list[FuturesContract]) -> dict:
        if not contracts:
            return {
                "structure": "flat",
                "front_back_spread": 0.0,
                "annualized_roll_cost_pct": 0.0,
                "slope": 0.0,
                "r_squared": 0.0,
                "n_contracts": 0,
            }

        sorted_c = sorted(contracts, key=lambda c: c.days_to_expiry)
        front = sorted_c[0]
        back = sorted_c[-1]

        front_back_spread = front.price - back.price

        annualized_roll_cost_pct = 0.0
        if len(sorted_c) >= 2:
            c1 = sorted_c[0]
            c2 = sorted_c[1]
            days_diff = max(1, c2.days_to_expiry - c1.days_to_expiry)
            if c1.price > 0:
                roll_diff = (c2.price - c1.price) / c1.price
                annualized_roll_cost_pct = roll_diff * (365.0 / days_diff)

        slope = 0.0
        r_squared = 0.0
        if len(sorted_c) >= 2:
            x = np.array([c.days_to_expiry for c in sorted_c], dtype=float)
            y = np.array([c.price for c in sorted_c], dtype=float)
            x_mean = x.mean()
            y_mean = y.mean()
            ss_xy = np.sum((x - x_mean) * (y - y_mean))
            ss_xx = np.sum((x - x_mean) ** 2)
            if ss_xx > 0:
                slope = ss_xy / ss_xx
                y_pred = y_mean + slope * (x - x_mean)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y_mean) ** 2)
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if slope > _SLOPE_CONTANGO_THRESHOLD:
            structure: Literal["contango", "backwardation", "flat"] = "contango"
        elif slope < _SLOPE_BACKWARDATION_THRESHOLD:
            structure = "backwardation"
        else:
            structure = "flat"

        return {
            "structure": structure,
            "front_back_spread": front_back_spread,
            "annualized_roll_cost_pct": annualized_roll_cost_pct,
            "slope": slope,
            "r_squared": r_squared,
            "n_contracts": len(sorted_c),
        }

    def compute_roll_yield(
        self, front: FuturesContract, next_contract: FuturesContract
    ) -> RollYield:
        days_to_roll = max(1, front.days_to_expiry)
        roll_yield_annualized = 0.0
        roll_return = 0.0
        if next_contract.price > 0:
            ratio = front.price / next_contract.price
            roll_return = ratio - 1.0
            roll_yield_annualized = roll_return * (365.0 / days_to_roll)

        spot_return = 0.0

        return RollYield(
            commodity=front.commodity,
            roll_date=date.today(),
            front_price=front.price,
            next_price=next_contract.price,
            days_to_roll=days_to_roll,
            spot_return=spot_return,
            roll_return=roll_return,
            roll_yield_annualized=roll_yield_annualized,
        )

    def continuous_contract_stitch(
        self,
        prices_by_contract: dict[str, pd.Series],
        method: Literal["panama", "ratio"] = "ratio",
    ) -> pd.Series:
        if not prices_by_contract:
            return pd.Series(dtype=float)

        sorted_symbols = sorted(prices_by_contract.keys())
        series_list = [prices_by_contract[s].dropna().sort_index() for s in sorted_symbols]
        series_list = [s for s in series_list if not s.empty]

        if not series_list:
            return pd.Series(dtype=float)

        if len(series_list) == 1:
            return series_list[0].copy()

        combined = series_list[-1].copy()

        for i in range(len(series_list) - 2, -1, -1):
            near = series_list[i]
            far = series_list[i + 1]

            overlap_dates = near.index.intersection(far.index)

            if len(overlap_dates) == 0:
                combined = pd.concat([near, combined])
                combined = combined[~combined.index.duplicated(keep="last")]
                continue

            roll_date = overlap_dates[-1]
            near_price_at_roll = near.loc[roll_date]
            far_price_at_roll = far.loc[roll_date]

            prior_dates = near.index[near.index <= roll_date]
            prior_near = near.loc[prior_dates]

            if method == "panama":
                adjustment = float(far_price_at_roll) - float(near_price_at_roll)
                adjusted_prior = prior_near + adjustment
            else:
                if float(near_price_at_roll) != 0:
                    ratio = float(far_price_at_roll) / float(near_price_at_roll)
                else:
                    ratio = 1.0
                adjusted_prior = prior_near * ratio

            combined = pd.concat([adjusted_prior, combined])
            combined = combined[~combined.index.duplicated(keep="last")]

        return combined.sort_index()

    def term_structure_slope_signal(self, ts: TermStructure) -> TermStructureSignal:
        slope = ts.term_structure_slope

        if slope >= _STRONG_CONTANGO_THRESHOLD:
            raw_signal = 1.0
            label = "strong_contango"
        elif slope >= _SLOPE_CONTANGO_THRESHOLD:
            raw_signal = slope / _STRONG_CONTANGO_THRESHOLD
            raw_signal = min(raw_signal, 0.99)
            label = "mild_contango"
        elif slope <= _STRONG_BACKWARDATION_THRESHOLD:
            raw_signal = -1.0
            label = "strong_backwardation"
        elif slope <= _SLOPE_BACKWARDATION_THRESHOLD:
            raw_signal = slope / abs(_STRONG_BACKWARDATION_THRESHOLD)
            raw_signal = max(raw_signal, -0.99)
            label = "mild_backwardation"
        else:
            raw_signal = slope / _SLOPE_CONTANGO_THRESHOLD * 0.1
            label = "flat"

        signal = float(np.clip(raw_signal, -1.0, 1.0))
        roll_cost_bps = ts.annualized_roll_cost_pct * 10_000.0

        return TermStructureSignal(
            commodity=ts.commodity,
            as_of=ts.as_of,
            signal=signal,
            signal_label=label,
            roll_cost_bps=roll_cost_bps,
            percentile_30d=None,
        )

    async def get_multi_commodity_snapshot(
        self, commodities: Optional[list[str]] = None
    ) -> dict[str, TermStructure]:
        if commodities is None:
            commodities = list(COMMODITY_CONFIG.keys())

        async def _safe_fetch(c: str) -> tuple[str, Optional[TermStructure]]:
            try:
                ts = await self.get_commodity_curve(c)
                return c, ts
            except Exception as exc:
                logger.warning("multi_snapshot_error", commodity=c, error=str(exc))
                return c, None

        results = await asyncio.gather(*[_safe_fetch(c) for c in commodities])
        return {c: ts for c, ts in results if ts is not None}

    async def get_futures_curve_history(
        self,
        commodity: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = date.today()

        if commodity not in COMMODITY_CONFIG:
            raise ValueError(f"Unknown commodity: {commodity}")

        cfg = COMMODITY_CONFIG[commodity]
        prefix = cfg["prefix"]
        exchange = cfg["exchange"]

        symbols_info = self._build_symbols(commodity, 3)
        symbols = [sym for sym, _, _ in symbols_info]

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

        rows: list[dict] = []
        try:
            hist_data = yf.download(
                tickers=symbols,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
            )
            close = hist_data["Close"] if "Close" in hist_data.columns else hist_data
            if isinstance(close, pd.Series):
                close = close.to_frame(name=symbols[0])

            for idx_date, row in close.iterrows():
                valid_prices = row.dropna()
                if len(valid_prices) < 2:
                    continue
                sorted_vals = [(sym, float(p)) for sym, p in valid_prices.items() if float(p) > 0]
                sorted_vals.sort(key=lambda x: symbols.index(x[0]) if x[0] in symbols else 999)

                if len(sorted_vals) < 2:
                    continue

                front_price = sorted_vals[0][1]
                m2_price = sorted_vals[1][1] if len(sorted_vals) > 1 else None
                m3_price = sorted_vals[2][1] if len(sorted_vals) > 2 else None
                front_back = front_price - sorted_vals[-1][1]
                structure = "contango" if front_back < 0 else ("backwardation" if front_back > 0 else "flat")

                rows.append({
                    "date": idx_date.date() if hasattr(idx_date, "date") else idx_date,
                    "front_price": front_price,
                    "m2_price": m2_price,
                    "m3_price": m3_price,
                    "front_back_spread": front_back,
                    "structure": structure,
                })
        except Exception as exc:
            logger.warning("curve_history_error", commodity=commodity, error=str(exc))

        return pd.DataFrame(rows)

    def _build_symbols(self, commodity: str, n: int) -> list[tuple[str, int, int]]:
        cfg = COMMODITY_CONFIG[commodity]
        prefix = cfg["prefix"]
        exchange = cfg["exchange"]
        allowed_months: list[int] = cfg["months"]

        today = date.today()
        current_year = today.year
        current_month = today.month

        results: list[tuple[str, int, int]] = []
        year = current_year

        candidates: list[tuple[int, int]] = []
        for y_offset in range(3):
            y = current_year + y_offset
            for m in allowed_months:
                expiry_approx = date(y, m, 1) + timedelta(days=14)
                if expiry_approx > today:
                    candidates.append((y, m))

        candidates.sort()
        candidates = candidates[:n]

        for y, m in candidates:
            month_code = MONTH_CODES[m]
            year_suffix = str(y)[-2:]
            symbol = f"{prefix}{month_code}{year_suffix}.{exchange}"
            results.append((symbol, m, y))

        return results

    async def _fetch_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            last = getattr(info, "last_price", None)
            if last is not None and last > 0:
                return float(last)
            hist = ticker.history(period="5d")
            if not hist.empty:
                close_val = hist["Close"].dropna()
                if not close_val.empty:
                    val = float(close_val.iloc[-1])
                    if val > 0:
                        return val
        except Exception as exc:
            logger.debug("fetch_price_error", symbol=symbol, error=str(exc))
        return None

    async def _download_vix_csv(self) -> pd.DataFrame:
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            resp = await client.get(CBOE_VIX_FUTURES_URL)
            resp.raise_for_status()
            content = resp.text

        from io import StringIO
        df = pd.read_csv(StringIO(content))
        df.columns = [c.strip() for c in df.columns]
        return df


def _parse_vix_futures_label(label: str) -> tuple[Optional[int], Optional[int]]:
    label = label.strip()
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    for abbr, num in month_map.items():
        if abbr in label:
            parts = label.replace(abbr, "").strip().split()
            for p in parts:
                if p.isdigit() and len(p) in (2, 4):
                    year = int(p) if len(p) == 4 else 2000 + int(p)
                    return num, year
    if len(label) >= 3 and label[0] in REVERSE_CODES:
        month_code = label[0]
        year_part = label[1:]
        if year_part.isdigit():
            month_num = REVERSE_CODES[month_code]
            year = 2000 + int(year_part) if len(year_part) == 2 else int(year_part)
            return month_num, year
    return None, None


async def vix_curve() -> TermStructure:
    fts = FuturesTermStructure()
    return await fts.get_vix_futures_curve()


async def commodity_curve(commodity: str) -> TermStructure:
    fts = FuturesTermStructure()
    return await fts.get_commodity_curve(commodity)


async def multi_curve_snapshot() -> dict[str, TermStructure]:
    fts = FuturesTermStructure()
    return await fts.get_multi_commodity_snapshot()
