"""
On-chain crypto metrics from free sources — Dimension #108.

Uses CoinGecko free API (no key) and Alternative.me Fear & Greed Index.
Approximates NVT ratio, MVRV, volume-price divergence, and a fear/greed
proxy from publicly available price/volume/market-cap data.

Full Glassnode quality (true UTXOs, realised cap) requires a paid key; this
module reaches dimension score ~3-4 from free signals only — still better
than most retail terminals.

Base URL: https://api.coingecko.com/api/v3
Fear/Greed: https://api.alternative.me/fng/?limit=7
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CG_BASE = "https://api.coingecko.com/api/v3"
FNG_URL = "https://api.alternative.me/fng/?limit=7"

_TIMEOUT = 30.0
_CACHE_TTL = 300       # 5 minutes
_RATE_LIMIT_SLEEP = 1.2  # CoinGecko free tier

# NVT thresholds (empirically derived, proxy values)
_NVT_OVERVALUED = 65.0
_NVT_UNDERVALUED = 27.0

# Symbol → CoinGecko coin ID map
_SYMBOL_TO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "DOGE": "dogecoin",
    "MATIC": "matic-network",
    "POL": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "LTC": "litecoin",
    "ATOM": "cosmos",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "INJ": "injective-protocol",
    "SUI": "sui",
    "TRX": "tron",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
    "TON": "the-open-network",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OnChainMetric(BaseModel):
    symbol: str
    date: date
    price_usd: float
    market_cap: float
    total_volume_24h: float
    nvt_proxy: Optional[float] = None
    mvrv_proxy: Optional[float] = None
    volume_price_divergence: Optional[float] = None
    dominance_pct: Optional[float] = None


class OnChainSignal(BaseModel):
    symbol: str
    nvt_signal: str           # "overvalued" | "fair" | "undervalued"
    nvt_value: Optional[float]
    mvrv_signal: str          # "overvalued" | "fair" | "undervalued"
    mvrv_value: Optional[float]
    fear_greed_proxy: Optional[float]  # 0-100
    trend_30d: str            # "up" | "down" | "sideways"
    signal_strength: str      # "strong_bullish" … "strong_bearish"


class GlobalCryptoData(BaseModel):
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: float
    eth_dominance: float
    defi_market_cap: Optional[float] = None
    market_cap_change_24h: float
    active_cryptocurrencies: int


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if datetime.now(timezone.utc).timestamp() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value: object) -> None:
    _cache[key] = (datetime.now(timezone.utc).timestamp(), value)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _get(client: httpx.AsyncClient, url: str, params: Optional[dict] = None) -> Optional[object]:
    cache_key = url + str(sorted((params or {}).items()))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    for attempt in range(1, 4):
        try:
            resp = await client.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code == 429:
                wait = _RATE_LIMIT_SLEEP * (attempt * 2)
                logger.warning("onchain_metrics 429 on %s, sleeping %.1fs", url, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            _cache_set(cache_key, data)
            return data
        except httpx.TimeoutException:
            logger.warning("onchain_metrics timeout %s (attempt %d)", url, attempt)
        except httpx.HTTPStatusError as exc:
            logger.error("onchain_metrics HTTP %d on %s", exc.response.status_code, url)
            return None
        except Exception as exc:
            logger.error("onchain_metrics error %s: %s", url, exc)
            return None
        if attempt < 3:
            await asyncio.sleep(_RATE_LIMIT_SLEEP)

    return None


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _nvt_signal(nvt: Optional[float]) -> str:
    if nvt is None:
        return "fair"
    if nvt > _NVT_OVERVALUED:
        return "overvalued"
    if nvt < _NVT_UNDERVALUED:
        return "undervalued"
    return "fair"


def _mvrv_signal(mvrv: Optional[float]) -> str:
    """
    Approximated MVRV: ratio of current market cap to 30-day-ago market cap
    (proxy for realised value).  MVRV > 3.5 = overvalued, < 1.0 = undervalued.
    """
    if mvrv is None:
        return "fair"
    if mvrv > 3.5:
        return "overvalued"
    if mvrv < 1.0:
        return "undervalued"
    return "fair"


def _overall_strength(
    nvt_sig: str,
    mvrv_sig: str,
    trend: str,
    fg_proxy: Optional[float],
) -> str:
    score = 0
    # Trend contribution
    if trend == "up":
        score += 1
    elif trend == "down":
        score -= 1
    # Valuation: undervalued = bullish signal
    if nvt_sig == "undervalued":
        score += 1
    elif nvt_sig == "overvalued":
        score -= 1
    if mvrv_sig == "undervalued":
        score += 1
    elif mvrv_sig == "overvalued":
        score -= 1
    # Fear/greed proxy: > 60 = greedy (bearish contrarian), < 40 = fearful (bullish contrarian)
    if fg_proxy is not None:
        if fg_proxy < 25:
            score += 2   # extreme fear → strong buy signal
        elif fg_proxy < 40:
            score += 1
        elif fg_proxy > 75:
            score -= 2   # extreme greed → strong sell signal
        elif fg_proxy > 60:
            score -= 1

    if score >= 3:
        return "strong_bullish"
    if score >= 1:
        return "bullish"
    if score <= -3:
        return "strong_bearish"
    if score <= -1:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# OnChainClient
# ---------------------------------------------------------------------------

class OnChainClient:
    CG_BASE = CG_BASE

    def _resolve_id(self, symbol: str) -> str:
        """Map a common ticker symbol to a CoinGecko coin ID."""
        return _SYMBOL_TO_ID.get(symbol.upper(), symbol.lower())

    # ------------------------------------------------------------------
    async def get_coin_metrics(self, symbol: str, days: int = 90) -> list[OnChainMetric]:
        """
        Fetch OHLCV + market-cap history from CoinGecko and compute derived
        on-chain proxies:

        - NVT proxy: market_cap / rolling_30d_avg_volume
        - Volume-price divergence: correlation(price_change, volume_change) over 14d
          (negative correlation = divergence)
        """
        coin_id = self._resolve_id(symbol)
        url = f"{self.CG_BASE}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}

        async with httpx.AsyncClient() as client:
            await asyncio.sleep(_RATE_LIMIT_SLEEP)
            data = await _get(client, url, params)

        if not isinstance(data, dict):
            logger.error("get_coin_metrics: no data for %s", symbol)
            return []

        prices = data.get("prices", [])        # [[timestamp_ms, price], ...]
        market_caps = data.get("market_caps", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            return []

        import pandas as pd

        def _to_series(raw: list) -> "pd.Series":
            idx = pd.to_datetime([r[0] for r in raw], unit="ms", utc=True)
            vals = [r[1] for r in raw]
            return pd.Series(vals, index=idx, dtype=float)

        price_s = _to_series(prices)
        mcap_s = _to_series(market_caps) if market_caps else pd.Series(dtype=float)
        vol_s = _to_series(volumes) if volumes else pd.Series(dtype=float)

        # Rolling 30d average volume for NVT proxy
        vol_30d_avg = vol_s.rolling(30, min_periods=7).mean()

        # Volume-price divergence: pearson correlation of 14d pct changes
        price_change = price_s.pct_change()
        vol_change = vol_s.pct_change()
        # Rolling 14-day correlation
        vp_div = price_change.rolling(14).corr(vol_change)

        result: list[OnChainMetric] = []
        common_idx = price_s.index

        for ts in common_idx:
            try:
                price = float(price_s.get(ts, float("nan")))
                mcap = float(mcap_s.get(ts, float("nan"))) if not mcap_s.empty else float("nan")
                vol = float(vol_s.get(ts, float("nan"))) if not vol_s.empty else float("nan")

                if any(v != v for v in (price, mcap, vol)):  # nan check
                    continue

                nvt = None
                avg_vol = vol_30d_avg.get(ts)
                if avg_vol and avg_vol > 0:
                    nvt = round(mcap / avg_vol, 4)

                corr = vp_div.get(ts)
                vp_divergence = round(float(corr), 4) if corr is not None and corr == corr else None

                result.append(OnChainMetric(
                    symbol=symbol.upper(),
                    date=ts.date(),
                    price_usd=round(price, 8),
                    market_cap=round(mcap, 2),
                    total_volume_24h=round(vol, 2),
                    nvt_proxy=nvt,
                    volume_price_divergence=vp_divergence,
                ))
            except Exception as exc:
                logger.debug("onchain metric row error: %s", exc)

        return result

    # ------------------------------------------------------------------
    async def get_signal(self, symbol: str) -> OnChainSignal:
        """
        Compute on-chain signals from 90 days of free CoinGecko data.

        NVT proxy: current_mcap / avg_daily_volume_30d
          > 65 = overvalued, < 27 = undervalued

        MVRV proxy: current_price / price_30d_ago
          (stand-in for realised price; imperfect but directionally useful)
          > 3.5 = overvalued, < 1.0 = undervalued

        Volume-price divergence: 14d rolling correlation of price_change ~ volume_change
          Negative = volume rising while price falls (or vice versa) = divergence

        Fear/greed proxy: 50 + (14d_return * 100) - (14d_vol * 100), clamped 0-100
        """
        metrics = await self.get_coin_metrics(symbol, days=90)
        if not metrics:
            return OnChainSignal(
                symbol=symbol.upper(),
                nvt_signal="fair",
                nvt_value=None,
                mvrv_signal="fair",
                mvrv_value=None,
                fear_greed_proxy=None,
                trend_30d="sideways",
                signal_strength="neutral",
            )

        latest = metrics[-1]

        # NVT from latest data point
        nvt_val = latest.nvt_proxy
        nvt_sig = _nvt_signal(nvt_val)

        # MVRV proxy: current price / price 30 days ago
        mvrv_val = None
        if len(metrics) >= 30:
            price_30d_ago = metrics[-30].price_usd
            if price_30d_ago > 0:
                mvrv_val = round(latest.price_usd / price_30d_ago, 4)
        mvrv_sig = _mvrv_signal(mvrv_val)

        # 30d trend
        trend_30d = "sideways"
        if len(metrics) >= 30:
            p_now = latest.price_usd
            p_30 = metrics[-30].price_usd
            ret_30 = (p_now - p_30) / max(p_30, 1e-9)
            if ret_30 > 0.05:
                trend_30d = "up"
            elif ret_30 < -0.05:
                trend_30d = "down"

        # Fear/greed proxy from 14d return and 14d volatility
        fg_proxy = None
        if len(metrics) >= 15:
            prices_14 = [m.price_usd for m in metrics[-15:]]
            ret_14 = (prices_14[-1] - prices_14[0]) / max(prices_14[0], 1e-9)
            rets = np.diff(np.log(np.array(prices_14) + 1e-9))
            vol_14 = float(np.std(rets)) if len(rets) > 1 else 0.0
            raw = 50 + (ret_14 * 100) - (vol_14 * 100)
            fg_proxy = round(float(np.clip(raw, 0, 100)), 2)

        strength = _overall_strength(nvt_sig, mvrv_sig, trend_30d, fg_proxy)

        return OnChainSignal(
            symbol=symbol.upper(),
            nvt_signal=nvt_sig,
            nvt_value=nvt_val,
            mvrv_signal=mvrv_sig,
            mvrv_value=mvrv_val,
            fear_greed_proxy=fg_proxy,
            trend_30d=trend_30d,
            signal_strength=strength,
        )

    # ------------------------------------------------------------------
    async def get_global_metrics(self) -> GlobalCryptoData:
        """GET /global — total market cap, dominance, volume."""
        url = f"{self.CG_BASE}/global"
        async with httpx.AsyncClient() as client:
            await asyncio.sleep(_RATE_LIMIT_SLEEP)
            data = await _get(client, url)

        if not isinstance(data, dict):
            raise RuntimeError("get_global_metrics: unexpected response")

        gd = data.get("data", {})
        market_caps: dict = gd.get("market_cap_percentage", {})
        total_mcap_map: dict = gd.get("total_market_cap", {})
        total_vol_map: dict = gd.get("total_volume", {})

        total_mcap = float(total_mcap_map.get("usd", 0))
        total_vol = float(total_vol_map.get("usd", 0))
        btc_dom = float(market_caps.get("btc", 0))
        eth_dom = float(market_caps.get("eth", 0))
        defi_mcap_map = gd.get("market_cap_defi")
        defi_mcap = float(defi_mcap_map.get("usd", 0)) if isinstance(defi_mcap_map, dict) else None
        mcap_change_24h = float(gd.get("market_cap_change_percentage_24h_usd", 0))
        active_coins = int(gd.get("active_cryptocurrencies", 0))

        return GlobalCryptoData(
            total_market_cap=round(total_mcap, 2),
            total_volume_24h=round(total_vol, 2),
            btc_dominance=round(btc_dom, 4),
            eth_dominance=round(eth_dom, 4),
            defi_market_cap=round(defi_mcap, 2) if defi_mcap else None,
            market_cap_change_24h=round(mcap_change_24h, 4),
            active_cryptocurrencies=active_coins,
        )

    # ------------------------------------------------------------------
    async def get_fear_greed_index(self) -> dict:
        """
        GET https://api.alternative.me/fng/?limit=7 — Fear & Greed Index.
        Free, no API key. Returns last 7 days of readings.

        Response format:
        {
            "name": "Fear and Greed Index",
            "data": [
                {"value": "72", "value_classification": "Greed", "timestamp": "...", "time_until_update": "..."},
                ...
            ]
        }
        """
        async with httpx.AsyncClient() as client:
            data = await _get(client, FNG_URL)

        if not isinstance(data, dict):
            return {"error": "Failed to fetch Fear & Greed Index"}

        raw_list = data.get("data", [])
        if not isinstance(raw_list, list):
            return {"error": "Unexpected response structure"}

        readings: list[dict] = []
        for entry in raw_list:
            try:
                readings.append({
                    "value": int(entry.get("value", 0)),
                    "classification": entry.get("value_classification", ""),
                    "date": datetime.utcfromtimestamp(
                        int(entry["timestamp"])
                    ).date().isoformat(),
                })
            except Exception as exc:
                logger.debug("fng parse error: %s", exc)

        if not readings:
            return {"error": "No readings parsed"}

        latest = readings[0]
        return {
            "current_value": latest["value"],
            "current_classification": latest["classification"],
            "last_7_days": readings,
        }
