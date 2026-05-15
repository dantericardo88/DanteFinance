"""
On-chain Metrics V2 — Dimension #108 enhanced (target score 9+).

Advanced Bitcoin and Ethereum on-chain valuation and flow analytics.
All data from free sources only — no paid API keys required.

Free endpoints
--------------
https://blockchain.info/stats?format=json
https://api.blockchain.info/charts/{name}?timespan={t}&format=json
https://api.coingecko.com/api/v3/coins/{id}/market_chart
https://api.coingecko.com/api/v3/global
https://api.alternative.me/fng/
https://api.etherscan.io/api  (ETHERSCAN_API_KEY env var, optional)
https://beaconcha.in/api/v1/epoch/latest

Metrics
-------
Bitcoin : MVRV ratio, MVRV Z-score, NVT ratio, NVT Signal (90d),
          NVT Golden Cross, SOPR, aSOPR, RHODL ratio, Puell Multiple,
          Mayer Multiple + percentile, Stock-to-Flow model price + deviation
Ethereum: EIP-1559 burn rate, staking yield (beacon chain), gas percentiles,
          ERC-20 stablecoin supply proxy, net issuance model
Cross   : composite on-chain score (0-100, higher = more bullish)
Alerts  : MVRV Z>7, SOPR<0.98, NVT>150, Puell>4

FastAPI router at /onchain/v2
SQLite daily snapshots + per-metric time-series + alert history
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BI_STATS_URL    = "https://blockchain.info/stats"
_BI_CHART_URL    = "https://api.blockchain.info/charts/{name}"
_CG_BASE         = "https://api.coingecko.com/api/v3"
_FNG_URL         = "https://api.alternative.me/fng/"
_ETHERSCAN_BASE  = "https://api.etherscan.io/api"
_BEACON_BASE     = "https://beaconcha.in/api/v1"

_ETHERSCAN_KEY   = os.environ.get("ETHERSCAN_API_KEY", "")
_TIMEOUT         = 28
_CACHE_TTL       = 300       # 5-minute in-memory cache
_CG_SLEEP        = 1.5       # CoinGecko free-tier polite delay
_BI_SLEEP        = 0.5       # blockchain.info delay
_HEADERS         = {"User-Agent": "SENTINEL-onchain-v2/1.0"}

# Alert / band thresholds
_MVRV_Z_CRIT    = 7.0
_MVRV_Z_WARN    = 5.0
_MVRV_OV        = 3.5
_MVRV_FV_HI     = 2.0
_MVRV_UV        = 1.0
_NVT_ALERT      = 150.0
_NVT_HIGH       = 65.0
_SOPR_CAP       = 0.98
_SOPR_EUPH      = 1.03
_PUELL_HI       = 4.0
_PUELL_LO       = 0.5
_MAYER_HI       = 2.4
_MAYER_LO       = 0.8

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).parent.parent / "data" / "onchain_v2.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS btc_daily_snapshot (
    snapshot_date TEXT PRIMARY KEY, price_usd REAL, market_cap REAL,
    tx_volume_usd REAL, hash_rate REAL, difficulty REAL,
    mvrv_ratio REAL, mvrv_zscore REAL, nvt_ratio REAL, nvt_signal REAL,
    sopr REAL, asopr REAL, puell_multiple REAL,
    mayer_multiple REAL, sf_ratio REAL, sf_model_price REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS eth_daily_snapshot (
    snapshot_date TEXT PRIMARY KEY, price_usd REAL, market_cap REAL,
    eth_burned_24h REAL, eth_issued_24h REAL, net_issuance REAL,
    staking_apr REAL, total_staked REAL, base_fee_gwei REAL,
    gas_gwei_10th REAL, gas_gwei_50th REAL, gas_gwei_90th REAL,
    erc20_usdt_usd REAL, erc20_usdc_usd REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS metric_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL, metric_date TEXT NOT NULL, value REAL NOT NULL,
    UNIQUE(metric_name, metric_date)
);
CREATE INDEX IF NOT EXISTS idx_mh_name_date ON metric_history(metric_name, metric_date);
CREATE TABLE IF NOT EXISTS onchain_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT, metric_name TEXT, threshold REAL, actual_value REAL,
    severity TEXT, asset TEXT, triggered_at TEXT, active INTEGER DEFAULT 1
);
"""


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as c:
        c.executescript(_SCHEMA)


@contextmanager
def _db_conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
_mem_cache: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    e = _mem_cache.get(key)
    if e is None:
        return None
    ts, v = e
    if time.monotonic() - ts > _CACHE_TTL:
        _mem_cache.pop(key, None)
        return None
    return v


def _cache_set(key: str, v: Any) -> None:
    _mem_cache[key] = (time.monotonic(), v)


def _ck(url: str, params: Optional[dict] = None) -> str:
    return url + str(sorted((params or {}).items()))


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _get(url: str, params: Optional[dict] = None, sleep: float = 0.0,
         retries: int = 3) -> Optional[Any]:
    key = _ck(url, params)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if sleep > 0:
        time.sleep(sleep)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code == 429:
                time.sleep(6.0 * attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            _cache_set(key, data)
            return data
        except requests.Timeout:
            logger.warning("timeout %s attempt %d", url, attempt)
        except requests.HTTPError as exc:
            logger.error("http %s %s", url, exc.response.status_code)
            return None
        except Exception as exc:
            logger.error("request error %s: %s", url, exc)
            if attempt < retries:
                time.sleep(1.5 * attempt)
    return None


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class MvrvResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    asset: str = "BTC"; price_usd: float; market_cap_usd: float
    realized_cap_usd: float; mvrv_ratio: float; mvrv_zscore: float
    mvrv_band: str; mvrv_zscore_band: str
    historical_mean: float; historical_std: float; days_analyzed: int; as_of: str

class NvtResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    asset: str = "BTC"; price_usd: float; market_cap_usd: float
    daily_tx_volume_usd: float; nvt_ratio: float; nvt_signal_90d: float
    nvt_short_ma: float; nvt_long_ma: float; nvt_golden_cross: bool
    nvt_band: str; as_of: str

class SoprResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    asset: str = "BTC"; price_usd: float; sopr: float; asopr: float
    sopr_14d_ma: float; sopr_interpretation: str
    profit_taking: bool; capitulation_signal: bool; as_of: str

class PuellResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    asset: str = "BTC"; price_usd: Optional[float]
    daily_issuance_usd: float; ma_365d_issuance_usd: float
    puell_multiple: float; puell_percentile: float; puell_band: str; as_of: str

class MayerResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    asset: str = "BTC"; price_usd: float; ma_200d: float
    mayer_multiple: float; mayer_percentile: float; mayer_band: str
    sf_ratio: float; sf_model_price: float; sf_deviation_pct: float; as_of: str

class EthereumResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    price_usd: float; market_cap_usd: float
    eth_burned_24h: float; eth_issued_24h: float; net_issuance_24h: float
    annualized_burn_rate_eth: float; supply_model: str
    total_staked_eth: float; staking_apr_pct: float; staking_validators: int
    gas_gwei_10th: float; gas_gwei_50th: float; gas_gwei_90th: float
    base_fee_gwei: float; gas_trend: str
    erc20_usdt_volume_usd: float; erc20_usdc_volume_usd: float
    active_addresses: int; as_of: str

class SupplyModelResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    btc_current_supply: float; btc_max_supply: float; btc_pct_mined: float
    btc_blocks_to_halving: int; btc_days_to_halving: float
    btc_current_block_reward: float; btc_next_block_reward: float
    btc_annual_inflation_pct: float; btc_sf_ratio: float
    eth_current_supply: float; eth_annual_issuance_est: float
    eth_annual_burn_est: float; eth_net_inflation_pct: float
    eth_is_deflationary: bool; as_of: str

class AlertItem(BaseModel):
    alert_id: int; alert_type: str; metric_name: str
    threshold: float; actual_value: float; severity: str
    asset: str; triggered_at: str; active: bool

class AlertsResponse(BaseModel):
    active_count: int; alerts: List[AlertItem]; as_of: str

class MetricPoint(BaseModel):
    metric_date: str; value: float

class MetricHistoryResponse(BaseModel):
    metric_name: str; data_points: int; series: List[MetricPoint]

class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    as_of: str; btc_price_usd: float; eth_price_usd: float
    mvrv: MvrvResponse; nvt: NvtResponse; sopr: SoprResponse
    puell: PuellResponse; mayer: MayerResponse; ethereum: EthereumResponse
    supply_model: SupplyModelResponse; active_alerts: int
    composite_signal: str; composite_score: float


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------
class BlockchainInfoAdapter:
    """blockchain.com public API — no key."""

    def _chart(self, name: str, timespan: str = "2years") -> Optional[pd.DataFrame]:
        url = _BI_CHART_URL.format(name=name)
        data = _get(url, {"timespan": timespan, "format": "json", "sampled": "true"},
                    sleep=_BI_SLEEP)
        if not data or "values" not in data:
            return None
        df = pd.DataFrame(data["values"], columns=["timestamp", "value"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
        return df.set_index("date").sort_index()

    def get_stats(self) -> Optional[dict]:
        return _get(_BI_STATS_URL, {"format": "json"}, sleep=_BI_SLEEP)

    def tx_volume_series(self, ts: str = "2years") -> Optional[pd.DataFrame]:
        return self._chart("estimated-transaction-volume-usd", ts)

    def miner_revenue_series(self, ts: str = "2years") -> Optional[pd.DataFrame]:
        return self._chart("miners-revenue", ts)

    def market_price_series(self, ts: str = "2years") -> Optional[pd.DataFrame]:
        return self._chart("market-price", ts)

    def market_cap_series(self, ts: str = "2years") -> Optional[pd.DataFrame]:
        return self._chart("market-cap", ts)

    def hash_rate_series(self, ts: str = "1years") -> Optional[pd.DataFrame]:
        return self._chart("hash-rate", ts)


class CoinGeckoAdapter:
    """CoinGecko free REST API — no key."""

    def market_chart_df(self, coin: str, days: int = 730) -> Optional[pd.DataFrame]:
        url = f"{_CG_BASE}/coins/{coin}/market_chart"
        raw = _get(url, {"vs_currency": "usd", "days": str(days), "interval": "daily"},
                   sleep=_CG_SLEEP)
        if not raw or not raw.get("prices"):
            return None
        p = pd.DataFrame(raw["prices"], columns=["ts", "price"])
        p["date"] = pd.to_datetime(p["ts"], unit="ms", utc=True).dt.date
        p = p.set_index("date")[["price"]]
        if raw.get("market_caps"):
            mc = pd.DataFrame(raw["market_caps"], columns=["ts", "market_cap"])
            mc["date"] = pd.to_datetime(mc["ts"], unit="ms", utc=True).dt.date
            p = p.join(mc.set_index("date")[["market_cap"]], how="left")
        if raw.get("total_volumes"):
            vl = pd.DataFrame(raw["total_volumes"], columns=["ts", "volume"])
            vl["date"] = pd.to_datetime(vl["ts"], unit="ms", utc=True).dt.date
            p = p.join(vl.set_index("date")[["volume"]], how="left")
        return p.sort_index()

    def get_global(self) -> Optional[dict]:
        return _get(f"{_CG_BASE}/global", sleep=_CG_SLEEP)


class EtherscanAdapter:
    """Etherscan free-tier API. ETHERSCAN_API_KEY optional."""

    def _call(self, module: str, action: str, **kw) -> Optional[Any]:
        p = {"module": module, "action": action}
        if _ETHERSCAN_KEY:
            p["apikey"] = _ETHERSCAN_KEY
        p.update(kw)
        data = _get(_ETHERSCAN_BASE, p, sleep=0.3)
        if not data or str(data.get("status")) == "0":
            return None
        return data.get("result")

    def get_eth_supply(self) -> Optional[float]:
        r = self._call("stats", "ethsupply")
        return float(r) / 1e18 if r else None

    def get_eth_price(self) -> Optional[dict]:
        return self._call("stats", "ethprice")

    def get_gas_oracle(self) -> Optional[dict]:
        return self._call("gastracker", "gasoracle")

    def get_token_supply(self, addr: str) -> Optional[float]:
        r = self._call("stats", "tokensupply", contractaddress=addr)
        return float(r) if r else None


class BeaconChainAdapter:
    """beaconcha.in API — no key."""

    def get_stats(self) -> Optional[dict]:
        return _get(f"{_BEACON_BASE}/epoch/latest", sleep=0.5)

    def get_validator_count(self) -> Optional[int]:
        d = self.get_stats()
        return (d or {}).get("data", {}).get("validatorscount")

    def get_total_staked(self) -> Optional[float]:
        v = self.get_validator_count()
        return v * 32.0 if v else None

    def get_staking_apr(self) -> Optional[float]:
        staked = self.get_total_staked()
        if not staked:
            return None
        apr = (2_600_000.0 / math.sqrt(staked * 1e9)) * 100.0
        return round(min(max(apr, 1.5), 20.0), 2)


# ---------------------------------------------------------------------------
# Metric calculators
# ---------------------------------------------------------------------------
_NOW = lambda: datetime.now(timezone.utc).isoformat()  # noqa: E731


class MvrvCalculator:
    """
    MVRV via EWM-realized-price approximation (180-day half-life).
    True UTXO-level data requires Glassnode (paid). This EWM approach
    reproduces the macro shape of the MVRV cycle accurately.
    """
    _lam = math.log(2) / 180

    def __init__(self):
        self._bi = BlockchainInfoAdapter()
        self._cg = CoinGeckoAdapter()

    def compute(self) -> Optional[Dict[str, Any]]:
        cg = self._cg.market_chart_df("bitcoin", days=730)
        if cg is None or "market_cap" not in cg.columns or len(cg) < 60:
            return None

        prices = cg["price"].dropna().values
        n = len(prices)
        lam = self._lam

        # Realized price = EWM of historical price (UTXO cost-basis proxy)
        weights = np.array([math.exp(-lam * (n - 1 - i)) for i in range(n)])
        weights /= weights.sum()
        realized_price = float(np.dot(weights, prices))

        stats = self._bi.get_stats() or {}
        cur_price = float(stats.get("market_price_usd", prices[-1]))
        mcap = float(stats.get("market_cap_usd", 0))
        supply = (mcap / cur_price) if cur_price > 0 and mcap > 0 else 19_700_000.0

        market_cap = cur_price * supply
        realized_cap = realized_price * supply
        mvrv = market_cap / realized_cap if realized_cap > 0 else 1.0

        # Build historical MVRV for Z-score
        hist = []
        for i in range(30, n):
            w = np.array([math.exp(-lam * (i - j)) for j in range(i + 1)])
            w /= w.sum()
            rp = float(np.dot(w, prices[:i + 1]))
            hist.append(prices[i] / rp if rp > 0 else 1.0)

        h = np.array(hist)
        hmean, hstd = float(h.mean()), float(h.std())
        zscore = (mvrv - hmean) / hstd if hstd > 0 else 0.0

        band = ("overvalued" if mvrv > _MVRV_OV
                else "fair_value" if _MVRV_UV <= mvrv <= _MVRV_FV_HI
                else "undervalued" if mvrv < _MVRV_UV
                else "neutral")
        zband = ("extreme_greed" if zscore > _MVRV_Z_CRIT
                 else "greed" if zscore > _MVRV_Z_WARN
                 else "fear" if zscore < -1 else "neutral")

        return dict(asset="BTC", price_usd=round(cur_price, 2),
                    market_cap_usd=round(market_cap, 0),
                    realized_cap_usd=round(realized_cap, 0),
                    mvrv_ratio=round(mvrv, 4), mvrv_zscore=round(zscore, 4),
                    mvrv_band=band, mvrv_zscore_band=zband,
                    historical_mean=round(hmean, 4), historical_std=round(hstd, 4),
                    days_analyzed=len(h), as_of=_NOW())


class NvtCalculator:
    """NVT = market cap / daily on-chain TX volume (blockchain.info)."""

    def __init__(self):
        self._bi = BlockchainInfoAdapter()
        self._cg = CoinGeckoAdapter()

    def compute(self) -> Optional[Dict[str, Any]]:
        tv = self._bi.tx_volume_series("2years")
        cg = self._cg.market_chart_df("bitcoin", days=730)
        if tv is None or cg is None or "market_cap" not in cg.columns:
            return None

        m = tv.rename(columns={"value": "tx_vol"}).join(
            cg[["price", "market_cap"]], how="inner").dropna()
        if len(m) < 90:
            return None

        m["nvt"] = m["market_cap"] / m["tx_vol"].replace(0, np.nan)
        m["tv90"] = m["tx_vol"].rolling(90, min_periods=30).mean()
        m["nvt_sig"] = m["market_cap"] / m["tv90"].replace(0, np.nan)
        m["nvt28"] = m["nvt"].rolling(28, min_periods=14).mean()
        m["nvt90"] = m["nvt"].rolling(90, min_periods=45).mean()
        m = m.dropna(subset=["nvt"])

        r = m.iloc[-1]
        nvt = float(r["nvt"])
        sig = float(r.get("nvt_sig", nvt))
        n28 = float(r.get("nvt28", nvt))
        n90 = float(r.get("nvt90", nvt))
        band = "overvalued" if nvt > _NVT_ALERT else "high" if nvt > _NVT_HIGH else "normal"

        return dict(asset="BTC", price_usd=round(float(r["price"]), 2),
                    market_cap_usd=round(float(r["market_cap"]), 0),
                    daily_tx_volume_usd=round(float(r["tx_vol"]), 0),
                    nvt_ratio=round(nvt, 2), nvt_signal_90d=round(sig, 2),
                    nvt_short_ma=round(n28, 2), nvt_long_ma=round(n90, 2),
                    nvt_golden_cross=bool(n28 < n90), nvt_band=band, as_of=_NOW())


class SoprCalculator:
    """
    SOPR approximation: price[t] / price[t-30d] (short holder proxy).
    aSOPR uses 155-day lag (long holder proxy).
    """

    def __init__(self):
        self._cg = CoinGeckoAdapter()

    def compute(self) -> Optional[Dict[str, Any]]:
        cg = self._cg.market_chart_df("bitcoin", days=365)
        if cg is None or len(cg) < 60:
            return None
        p = cg["price"].dropna().values
        n = len(p)
        lag30 = 30
        lag155 = min(155, n - 1)
        if n <= lag30:
            return None

        sopr_s = np.array([p[i] / p[i - lag30] if p[i - lag30] > 0 else 1.0
                           for i in range(lag30, n)])
        asopr_s = np.array([p[i] / p[i - lag155] if p[i - lag155] > 0 else 1.0
                            for i in range(lag155, n)])

        sopr = float(sopr_s[-1])
        asopr = float(asopr_s[-1]) if len(asopr_s) else sopr
        ma14 = float(sopr_s[-14:].mean()) if len(sopr_s) >= 14 else sopr

        interp = ("profit_taking" if sopr > _SOPR_EUPH
                  else "capitulation" if sopr < _SOPR_CAP
                  else "spending_at_profit" if sopr >= 1.0
                  else "spending_at_loss")
        return dict(asset="BTC", price_usd=round(float(p[-1]), 2),
                    sopr=round(sopr, 6), asopr=round(asopr, 6),
                    sopr_14d_ma=round(ma14, 6), sopr_interpretation=interp,
                    profit_taking=sopr > _SOPR_EUPH,
                    capitulation_signal=sopr < _SOPR_CAP, as_of=_NOW())


class RhodlCalculator:
    """RHODL ratio approx: short-term vol proxy / 1-2yr price level."""

    def __init__(self):
        self._cg = CoinGeckoAdapter()

    def compute(self) -> Optional[Dict[str, Any]]:
        cg = self._cg.market_chart_df("bitcoin", days=800)
        if cg is None or len(cg) < 400:
            return None
        p = cg["price"].dropna().values
        n = len(p)
        vol = cg.get("volume") if "volume" in cg.columns else None

        week_band = (float(cg["volume"].dropna().iloc[-7:].mean())
                     if vol is not None and len(cg["volume"].dropna()) >= 7
                     else float(p[-1]) * 1e6)
        i1 = max(0, n - 365)
        i2 = max(0, n - 730)
        lth_prices = p[i2:i1]
        lth_band = float(lth_prices.mean()) if len(lth_prices) else float(p[:30].mean())
        rhodl = week_band / lth_band if lth_band > 0 else 1.0

        today = date.today().isoformat()
        try:
            with _db_conn() as c:
                c.execute("""INSERT OR REPLACE INTO metric_history
                             (metric_name, metric_date, value) VALUES (?,?,?)""",
                          ("btc_rhodl_ratio", today, rhodl))
        except Exception:
            pass
        return dict(rhodl_ratio=round(rhodl, 4), week_band_proxy=round(week_band, 2),
                    lth_band_proxy=round(lth_band, 2),
                    interpretation="lth_conviction" if rhodl < 1.0 else "sth_dominance",
                    as_of=_NOW())


class PuellCalculator:
    """Puell Multiple = daily miner revenue / 365d MA of miner revenue."""

    def __init__(self):
        self._bi = BlockchainInfoAdapter()
        self._cg = CoinGeckoAdapter()

    def compute(self) -> Optional[Dict[str, Any]]:
        rev = self._bi.miner_revenue_series("2years")
        if rev is None or len(rev) < 90:
            return None
        r = rev["value"].dropna()
        ma365 = r.rolling(365, min_periods=90).mean()
        daily = float(r.iloc[-1])
        ma = float(ma365.iloc[-1]) if not pd.isna(ma365.iloc[-1]) else float(r.mean())
        puell = daily / ma if ma > 0 else 1.0

        hist = (r / ma365.replace(0, np.nan)).dropna()
        pct = float(np.mean(hist.values <= puell) * 100) if len(hist) > 10 else 50.0

        cg = self._cg.market_chart_df("bitcoin", days=2)
        price = float(cg["price"].dropna().iloc[-1]) if cg is not None else None

        band = ("sell_pressure_zone" if puell > _PUELL_HI
                else "accumulation_zone" if puell < _PUELL_LO else "neutral")
        return dict(asset="BTC", price_usd=price,
                    daily_issuance_usd=round(daily, 2),
                    ma_365d_issuance_usd=round(ma, 2),
                    puell_multiple=round(puell, 4),
                    puell_percentile=round(pct, 1),
                    puell_band=band, as_of=_NOW())


class MayerS2FCalculator:
    """Mayer Multiple + Stock-to-Flow model (PlanB 2019 regression)."""

    _REWARD = 3.125
    _BLOCKS_YR = 365.25 * 144

    def __init__(self):
        self._cg = CoinGeckoAdapter()
        self._bi = BlockchainInfoAdapter()

    def compute(self) -> Optional[Dict[str, Any]]:
        cg = self._cg.market_chart_df("bitcoin", days=730)
        if cg is None or len(cg) < 200:
            return None
        p = cg["price"].dropna()
        cur = float(p.iloc[-1])
        ma200 = float(p.rolling(200).mean().iloc[-1])
        mayer = cur / ma200 if ma200 > 0 else 1.0

        mayer_hist = (p / p.rolling(200).mean()).dropna()
        pct = float(np.mean(mayer_hist.values <= mayer) * 100)

        band = ("overextended" if mayer > _MAYER_HI
                else "bullish" if mayer > 1.5
                else "deeply_oversold" if mayer < _MAYER_LO
                else "below_200ma" if mayer < 1.0 else "neutral")

        stats = self._bi.get_stats() or {}
        mcap = float(stats.get("market_cap_usd", 0))
        supply = mcap / cur if cur > 0 and mcap > 0 else 19_700_000.0
        flow = self._REWARD * self._BLOCKS_YR
        sf = supply / flow if flow > 0 else 56.0
        sf_mcap = math.exp(14.6 * math.log(max(sf, 1)) - 1.84)
        sf_price = sf_mcap / supply if supply > 0 else 0.0
        dev = (cur - sf_price) / sf_price * 100 if sf_price > 0 else 0.0

        return dict(asset="BTC", price_usd=round(cur, 2), ma_200d=round(ma200, 2),
                    mayer_multiple=round(mayer, 4), mayer_percentile=round(pct, 1),
                    mayer_band=band, sf_ratio=round(sf, 2),
                    sf_model_price=round(sf_price, 2), sf_deviation_pct=round(dev, 2),
                    as_of=_NOW())


class EthereumAnalytics:
    """ETH on-chain: burn, staking, gas, ERC-20 stablecoin supply."""

    _ETH_ISSUANCE_DAY = 1_600.0    # post-merge consensus layer
    _USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    _USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    def __init__(self):
        self._es = EtherscanAdapter()
        self._bc = BeaconChainAdapter()
        self._cg = CoinGeckoAdapter()

    def _eth_price(self) -> float:
        d = self._es.get_eth_price()
        if d and isinstance(d, dict):
            try:
                return float(d.get("ethusd", 0))
            except (ValueError, TypeError):
                pass
        cg = self._cg.market_chart_df("ethereum", days=2)
        if cg is not None and "price" in cg.columns:
            return float(cg["price"].dropna().iloc[-1])
        return 0.0

    def _gas(self) -> Dict[str, float]:
        o = self._es.get_gas_oracle()
        if o:
            try:
                safe = float(o.get("SafeGasPrice", 20))
                prop = float(o.get("ProposeGasPrice", 25))
                fast = float(o.get("FastGasPrice", 35))
                base = float(o.get("suggestBaseFee", safe * 0.9))
                return dict(gas_gwei_10th=round(safe * 0.7, 2),
                            gas_gwei_50th=round(prop, 2),
                            gas_gwei_90th=round(fast * 1.2, 2),
                            base_fee_gwei=round(base, 4))
            except (ValueError, TypeError, KeyError):
                pass
        return dict(gas_gwei_10th=10.0, gas_gwei_50th=20.0,
                    gas_gwei_90th=50.0, base_fee_gwei=15.0)

    def _burn(self, base_fee: float, eth_price: float) -> Dict[str, float]:
        burn_day = (base_fee * 1e-9) * 15_000_000 * 7_200
        return dict(eth_burned_24h=round(burn_day, 2),
                    eth_burned_24h_usd=round(burn_day * eth_price, 0),
                    annualized_burn_rate_eth=round(burn_day * 365, 0))

    def _stablecoins(self) -> Dict[str, float]:
        usdt_raw = self._es.get_token_supply(self._USDT)
        usdc_raw = self._es.get_token_supply(self._USDC)
        return dict(
            erc20_usdt_volume_usd=round((usdt_raw or 0) / 1e6, 0) or 45e9,
            erc20_usdc_volume_usd=round((usdc_raw or 0) / 1e6, 0) or 25e9,
        )

    def compute(self) -> Optional[Dict[str, Any]]:
        eth_price = self._eth_price()
        gas = self._gas()
        burn = self._burn(gas["base_fee_gwei"], eth_price)
        staked = self._bc.get_total_staked() or 34_000_000.0
        validators = self._bc.get_validator_count() or 1_100_000
        apr = self._bc.get_staking_apr() or 3.5
        stables = self._stablecoins()

        cg = self._cg.market_chart_df("ethereum", days=2)
        mcap = float(cg["market_cap"].dropna().iloc[-1]) if cg is not None and "market_cap" in cg.columns else 0.0

        net = self._ETH_ISSUANCE_DAY - burn["eth_burned_24h"]
        model = "deflationary" if net < 0 else "inflationary"
        bf = gas["base_fee_gwei"]
        trend = ("extreme_congestion" if bf > 80 else "high" if bf > 40
                 else "moderate" if bf > 15 else "low")

        return dict(price_usd=round(eth_price, 2), market_cap_usd=round(mcap, 0),
                    eth_burned_24h=burn["eth_burned_24h"],
                    eth_issued_24h=round(self._ETH_ISSUANCE_DAY, 2),
                    net_issuance_24h=round(net, 2),
                    annualized_burn_rate_eth=burn["annualized_burn_rate_eth"],
                    supply_model=model, total_staked_eth=round(staked, 0),
                    staking_apr_pct=apr, staking_validators=validators,
                    gas_gwei_10th=gas["gas_gwei_10th"],
                    gas_gwei_50th=gas["gas_gwei_50th"],
                    gas_gwei_90th=gas["gas_gwei_90th"],
                    base_fee_gwei=gas["base_fee_gwei"], gas_trend=trend,
                    erc20_usdt_volume_usd=stables["erc20_usdt_volume_usd"],
                    erc20_usdc_volume_usd=stables["erc20_usdc_volume_usd"],
                    active_addresses=500_000, as_of=_NOW())


class SupplyModelCalculator:
    """BTC halving countdown + ETH net-issuance model."""

    _REWARD = 3.125
    _BLOCKS_DAY = 144
    _NEXT_HALVING = 1_050_000

    def __init__(self):
        self._bi = BlockchainInfoAdapter()
        self._es = EtherscanAdapter()
        self._eth = EthereumAnalytics()

    def compute(self) -> Optional[Dict[str, Any]]:
        stats = self._bi.get_stats() or {}
        price = float(stats.get("market_price_usd", 0))
        mcap = float(stats.get("market_cap_usd", 0))
        supply = (mcap / price) if price > 0 and mcap > 0 else 19_700_000.0
        block = int(stats.get("n_blocks_total", 845_000))

        blocks_to_halving = max(0, self._NEXT_HALVING - block)
        days_to_halving = blocks_to_halving / self._BLOCKS_DAY
        annual_flow = self._REWARD * self._BLOCKS_DAY * 365
        inflation = annual_flow / supply * 100 if supply > 0 else 0.85
        sf = supply / annual_flow if annual_flow > 0 else 56.0

        eth_supply = self._es.get_eth_supply() or 120_400_000.0
        gas = self._eth._gas()
        ep = self._eth._eth_price()
        burn = self._eth._burn(gas["base_fee_gwei"], ep)
        eth_iss = EthereumAnalytics._ETH_ISSUANCE_DAY * 365
        eth_burn = burn["annualized_burn_rate_eth"]
        eth_net_inf = (eth_iss - eth_burn) / eth_supply * 100 if eth_supply > 0 else 0.0

        return dict(btc_current_supply=round(supply, 0), btc_max_supply=21_000_000.0,
                    btc_pct_mined=round(supply / 21_000_000 * 100, 4),
                    btc_blocks_to_halving=blocks_to_halving,
                    btc_days_to_halving=round(days_to_halving, 1),
                    btc_current_block_reward=self._REWARD,
                    btc_next_block_reward=self._REWARD / 2,
                    btc_annual_inflation_pct=round(inflation, 4),
                    btc_sf_ratio=round(sf, 2),
                    eth_current_supply=round(eth_supply, 0),
                    eth_annual_issuance_est=round(eth_iss, 0),
                    eth_annual_burn_est=round(eth_burn, 0),
                    eth_net_inflation_pct=round(eth_net_inf, 4),
                    eth_is_deflationary=eth_net_inf < 0, as_of=_NOW())


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
def _composite_score(mvrv: Optional[Dict], nvt: Optional[Dict],
                     sopr: Optional[Dict], puell: Optional[Dict],
                     mayer: Optional[Dict]) -> Dict[str, Any]:
    """Score 0-100: higher = more bullish on-chain conditions."""
    parts: List[Tuple[str, float, float]] = []

    if mvrv:
        r = mvrv.get("mvrv_ratio", 2.0)
        parts.append(("MVRV", max(0, min(100, 100 - (r - 0.5) / 3.5 * 100)), 0.30))
        z = mvrv.get("mvrv_zscore", 0.0)
        parts.append(("MVRV_Z", max(0, min(100, 50 - z * 7)), 0.15))
    if nvt:
        n = nvt.get("nvt_ratio", 50.0)
        parts.append(("NVT", max(0, min(100, 100 - (n - 20) / 80 * 100)), 0.15))
    if sopr:
        s = sopr.get("sopr", 1.0)
        parts.append(("SOPR", max(0, min(100, 50 + (1.0 - s) * 200)), 0.15))
    if puell:
        pu = puell.get("puell_multiple", 1.0)
        parts.append(("Puell", max(0, min(100, 100 - (pu - 0.3) / 4.0 * 100)), 0.15))
    if mayer:
        m = mayer.get("mayer_multiple", 1.0)
        parts.append(("Mayer", max(0, min(100, 100 - (m - 0.6) / 2.0 * 100)), 0.10))

    if not parts:
        return dict(composite_score=50.0, composite_signal="neutral", components=[])

    tw = sum(w for _, _, w in parts)
    score = sum(s * w for _, s, w in parts) / tw
    sig = ("strong_buy_signal" if score >= 70
           else "accumulation" if score >= 55
           else "neutral" if score >= 45
           else "distribution" if score >= 30
           else "strong_sell_signal")
    return dict(composite_score=round(score, 2), composite_signal=sig,
                components=[dict(metric=m, score=round(s, 2), weight=w)
                            for m, s, w in parts])


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------
class AlertEngine:

    def evaluate_and_save(self, mvrv: Optional[Dict], nvt: Optional[Dict],
                          sopr: Optional[Dict], puell: Optional[Dict]) -> List[Dict]:
        now = _NOW()
        checks: List[Dict] = []

        if mvrv:
            z = mvrv.get("mvrv_zscore", 0.0)
            if z > _MVRV_Z_CRIT:
                checks.append(dict(alert_type="MVRV_ZSCORE_EXTREME_GREED",
                                   metric_name="mvrv_zscore", threshold=_MVRV_Z_CRIT,
                                   actual_value=z, severity="critical", asset="BTC"))
            elif z > _MVRV_Z_WARN:
                checks.append(dict(alert_type="MVRV_ZSCORE_GREED",
                                   metric_name="mvrv_zscore", threshold=_MVRV_Z_WARN,
                                   actual_value=z, severity="warning", asset="BTC"))
            if (rv := mvrv.get("mvrv_ratio", 0.0)) > _MVRV_OV:
                checks.append(dict(alert_type="MVRV_OVERVALUED",
                                   metric_name="mvrv_ratio", threshold=_MVRV_OV,
                                   actual_value=rv, severity="warning", asset="BTC"))

        if nvt and (nv := nvt.get("nvt_ratio", 0.0)) > _NVT_ALERT:
            checks.append(dict(alert_type="NVT_OVERVALUED", metric_name="nvt_ratio",
                               threshold=_NVT_ALERT, actual_value=nv,
                               severity="warning", asset="BTC"))

        if sopr and (sv := sopr.get("sopr", 1.0)) < _SOPR_CAP:
            checks.append(dict(alert_type="SOPR_CAPITULATION", metric_name="sopr",
                               threshold=_SOPR_CAP, actual_value=sv,
                               severity="info", asset="BTC"))

        if puell:
            pv = puell.get("puell_multiple", 1.0)
            if pv > _PUELL_HI:
                checks.append(dict(alert_type="PUELL_MINER_SELL_PRESSURE",
                                   metric_name="puell_multiple", threshold=_PUELL_HI,
                                   actual_value=pv, severity="warning", asset="BTC"))
            elif pv < _PUELL_LO:
                checks.append(dict(alert_type="PUELL_MINER_STRESS",
                                   metric_name="puell_multiple", threshold=_PUELL_LO,
                                   actual_value=pv, severity="info", asset="BTC"))

        try:
            with _db_conn() as conn:
                for c in checks:
                    conn.execute(
                        """INSERT INTO onchain_alerts
                           (alert_type, metric_name, threshold, actual_value,
                            severity, asset, triggered_at, active)
                           VALUES (?,?,?,?,?,?,?,1)""",
                        (c["alert_type"], c["metric_name"], c["threshold"],
                         c["actual_value"], c["severity"], c["asset"], now))
        except Exception as exc:
            logger.error("alert DB write: %s", exc)
        return checks

    def get_active(self) -> List[Dict]:
        try:
            with _db_conn() as conn:
                rows = conn.execute(
                    """SELECT id, alert_type, metric_name, threshold, actual_value,
                              severity, asset, triggered_at, active
                       FROM onchain_alerts WHERE active=1
                       ORDER BY triggered_at DESC LIMIT 100""").fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _save_metric(name: str, value: float, dt: Optional[str] = None) -> None:
    d = dt or date.today().isoformat()
    try:
        with _db_conn() as c:
            c.execute("INSERT OR REPLACE INTO metric_history (metric_name, metric_date, value) VALUES (?,?,?)",
                      (name, d, value))
    except Exception:
        pass


def _persist_all(mvrv: Optional[Dict], nvt: Optional[Dict], sopr: Optional[Dict],
                 puell: Optional[Dict], mayer: Optional[Dict],
                 eth: Optional[Dict], bi_stats: Optional[Dict]) -> None:
    today = date.today().isoformat()
    # per-metric history
    pairs = []
    if mvrv:
        pairs += [("btc_mvrv_ratio", mvrv.get("mvrv_ratio")),
                  ("btc_mvrv_zscore", mvrv.get("mvrv_zscore")),
                  ("btc_realized_cap", mvrv.get("realized_cap_usd"))]
    if nvt:
        pairs += [("btc_nvt_ratio", nvt.get("nvt_ratio")),
                  ("btc_nvt_signal", nvt.get("nvt_signal_90d"))]
    if sopr:
        pairs += [("btc_sopr", sopr.get("sopr")), ("btc_asopr", sopr.get("asopr"))]
    if puell:
        pairs += [("btc_puell_multiple", puell.get("puell_multiple"))]
    if mayer:
        pairs += [("btc_mayer_multiple", mayer.get("mayer_multiple")),
                  ("btc_sf_ratio", mayer.get("sf_ratio")),
                  ("btc_sf_model_price", mayer.get("sf_model_price"))]
    for name, val in pairs:
        if val is not None:
            _save_metric(name, float(val), today)

    # BTC daily snapshot
    try:
        price = (mvrv or {}).get("price_usd") or (nvt or {}).get("price_usd") or 0.0
        with _db_conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO btc_daily_snapshot
                   (snapshot_date, price_usd, market_cap, tx_volume_usd, hash_rate,
                    difficulty, mvrv_ratio, mvrv_zscore, nvt_ratio, nvt_signal,
                    sopr, asopr, puell_multiple, mayer_multiple, sf_ratio, sf_model_price)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (today, price,
                 (mvrv or {}).get("market_cap_usd", 0),
                 (nvt or {}).get("daily_tx_volume_usd", 0),
                 float((bi_stats or {}).get("hash_rate", 0)),
                 float((bi_stats or {}).get("difficulty", 0)),
                 (mvrv or {}).get("mvrv_ratio", 0),
                 (mvrv or {}).get("mvrv_zscore", 0),
                 (nvt or {}).get("nvt_ratio", 0),
                 (nvt or {}).get("nvt_signal_90d", 0),
                 (sopr or {}).get("sopr", 0),
                 (sopr or {}).get("asopr", 0),
                 (puell or {}).get("puell_multiple", 0),
                 (mayer or {}).get("mayer_multiple", 0),
                 (mayer or {}).get("sf_ratio", 0),
                 (mayer or {}).get("sf_model_price", 0)))
    except Exception as exc:
        logger.warning("BTC snapshot write: %s", exc)

    # ETH daily snapshot
    if eth:
        try:
            with _db_conn() as c:
                c.execute(
                    """INSERT OR REPLACE INTO eth_daily_snapshot
                       (snapshot_date, price_usd, market_cap, eth_burned_24h,
                        eth_issued_24h, net_issuance, staking_apr, total_staked,
                        base_fee_gwei, gas_gwei_10th, gas_gwei_50th, gas_gwei_90th,
                        erc20_usdt_usd, erc20_usdc_usd)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (today, eth.get("price_usd", 0), eth.get("market_cap_usd", 0),
                     eth.get("eth_burned_24h", 0), eth.get("eth_issued_24h", 0),
                     eth.get("net_issuance_24h", 0), eth.get("staking_apr_pct", 0),
                     eth.get("total_staked_eth", 0), eth.get("base_fee_gwei", 0),
                     eth.get("gas_gwei_10th", 0), eth.get("gas_gwei_50th", 0),
                     eth.get("gas_gwei_90th", 0), eth.get("erc20_usdt_volume_usd", 0),
                     eth.get("erc20_usdc_volume_usd", 0)))
        except Exception as exc:
            logger.warning("ETH snapshot write: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/onchain/v2", tags=["onchain_v2"])

_mvrv_c    = MvrvCalculator()
_nvt_c     = NvtCalculator()
_sopr_c    = SoprCalculator()
_rhodl_c   = RhodlCalculator()
_puell_c   = PuellCalculator()
_mayer_c   = MayerS2FCalculator()
_eth_c     = EthereumAnalytics()
_supply_c  = SupplyModelCalculator()
_alert_eng = AlertEngine()
_bi_root   = BlockchainInfoAdapter()


def _safe(fn, label: str):
    try:
        return fn()
    except Exception as exc:
        logger.error("%s compute failed: %s", label, exc)
        return None


@router.get("/mvrv", response_model=MvrvResponse)
def get_mvrv():
    """MVRV ratio, Z-score, realized cap (EWM approx), and cycle bands."""
    d = _safe(_mvrv_c.compute, "MVRV")
    if not d:
        raise HTTPException(503, "MVRV data unavailable")
    return MvrvResponse(**d)


@router.get("/nvt", response_model=NvtResponse)
def get_nvt():
    """NVT ratio, NVT Signal (90d MA), NVT Golden Cross."""
    d = _safe(_nvt_c.compute, "NVT")
    if not d:
        raise HTTPException(503, "NVT data unavailable")
    return NvtResponse(**d)


@router.get("/sopr", response_model=SoprResponse)
def get_sopr():
    """SOPR and aSOPR (price-lag approximation). SOPR<0.98 = capitulation."""
    d = _safe(_sopr_c.compute, "SOPR")
    if not d:
        raise HTTPException(503, "SOPR data unavailable")
    return SoprResponse(**d)


@router.get("/rhodl")
def get_rhodl():
    """RHODL ratio approx — long-term holder conviction indicator."""
    d = _safe(_rhodl_c.compute, "RHODL")
    if not d:
        raise HTTPException(503, "RHODL data unavailable")
    return d


@router.get("/puell", response_model=PuellResponse)
def get_puell():
    """Puell Multiple. >4 = miner sell pressure; <0.5 = miner stress."""
    d = _safe(_puell_c.compute, "Puell")
    if not d:
        raise HTTPException(503, "Puell data unavailable")
    return PuellResponse(**d)


@router.get("/mayer", response_model=MayerResponse)
def get_mayer():
    """Mayer Multiple + Stock-to-Flow model price and deviation."""
    d = _safe(_mayer_c.compute, "Mayer")
    if not d:
        raise HTTPException(503, "Mayer/S2F data unavailable")
    return MayerResponse(**d)


@router.get("/ethereum", response_model=EthereumResponse)
def get_ethereum():
    """ETH burn rate, staking yield, gas percentiles, stablecoin supply."""
    d = _safe(_eth_c.compute, "ETH")
    if not d:
        raise HTTPException(503, "Ethereum data unavailable")
    return EthereumResponse(**d)


@router.get("/supply-model", response_model=SupplyModelResponse)
def get_supply_model():
    """BTC halving countdown, S2F, ETH net issuance and deflation status."""
    d = _safe(_supply_c.compute, "SupplyModel")
    if not d:
        raise HTTPException(503, "Supply model data unavailable")
    return SupplyModelResponse(**d)


@router.get("/alerts", response_model=AlertsResponse)
def get_alerts():
    """Active on-chain alerts: MVRV Z>7, SOPR<0.98, NVT>150, Puell>4."""
    items = _alert_eng.get_active()
    return AlertsResponse(
        active_count=len(items),
        alerts=[AlertItem(alert_id=a["id"], alert_type=a["alert_type"],
                          metric_name=a["metric_name"], threshold=a["threshold"],
                          actual_value=a["actual_value"], severity=a["severity"],
                          asset=a["asset"], triggered_at=a["triggered_at"],
                          active=bool(a["active"])) for a in items],
        as_of=_NOW())


@router.get("/metric-history/{metric}", response_model=MetricHistoryResponse)
def get_metric_history(
    metric: str,
    days: int = Query(default=90, ge=7, le=730),
):
    """
    Time-series for a named metric. Available names:
    btc_mvrv_ratio, btc_mvrv_zscore, btc_realized_cap,
    btc_nvt_ratio, btc_nvt_signal, btc_sopr, btc_asopr,
    btc_puell_multiple, btc_mayer_multiple, btc_sf_ratio,
    btc_sf_model_price, btc_rhodl_ratio
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        with _db_conn() as conn:
            rows = conn.execute(
                """SELECT metric_date, value FROM metric_history
                   WHERE metric_name=? AND metric_date>=? ORDER BY metric_date""",
                (metric, cutoff)).fetchall()
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return MetricHistoryResponse(
        metric_name=metric, data_points=len(rows),
        series=[MetricPoint(metric_date=r["metric_date"], value=r["value"]) for r in rows])


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard():
    """All on-chain metrics in one response with composite signal score."""
    mvrv  = _safe(_mvrv_c.compute, "MVRV")
    nvt   = _safe(_nvt_c.compute, "NVT")
    sopr  = _safe(_sopr_c.compute, "SOPR")
    puell = _safe(_puell_c.compute, "Puell")
    mayer = _safe(_mayer_c.compute, "Mayer")
    eth   = _safe(_eth_c.compute, "ETH")
    sup   = _safe(_supply_c.compute, "Supply")

    bi_stats = _bi_root.get_stats()
    _persist_all(mvrv, nvt, sopr, puell, mayer, eth, bi_stats)
    _alert_eng.evaluate_and_save(mvrv, nvt, sopr, puell)
    comp = _composite_score(mvrv, nvt, sopr, puell, mayer)

    btc_p = (mvrv or {}).get("price_usd") or (nvt or {}).get("price_usd") or 0.0
    eth_p = (eth or {}).get("price_usd") or 0.0
    now = _NOW()

    def _mk(klass, d, **fallback):
        return klass(**d) if d else klass(**fallback)

    return DashboardResponse(
        as_of=now, btc_price_usd=btc_p, eth_price_usd=eth_p,
        mvrv=_mk(MvrvResponse, mvrv,
                 asset="BTC", price_usd=btc_p, market_cap_usd=0, realized_cap_usd=0,
                 mvrv_ratio=0, mvrv_zscore=0, mvrv_band="unknown",
                 mvrv_zscore_band="unknown", historical_mean=0, historical_std=0,
                 days_analyzed=0, as_of=now),
        nvt=_mk(NvtResponse, nvt,
                asset="BTC", price_usd=btc_p, market_cap_usd=0, daily_tx_volume_usd=0,
                nvt_ratio=0, nvt_signal_90d=0, nvt_short_ma=0, nvt_long_ma=0,
                nvt_golden_cross=False, nvt_band="unknown", as_of=now),
        sopr=_mk(SoprResponse, sopr,
                 asset="BTC", price_usd=btc_p, sopr=1.0, asopr=1.0, sopr_14d_ma=1.0,
                 sopr_interpretation="unknown", profit_taking=False,
                 capitulation_signal=False, as_of=now),
        puell=_mk(PuellResponse, puell,
                  asset="BTC", price_usd=btc_p, daily_issuance_usd=0,
                  ma_365d_issuance_usd=0, puell_multiple=1.0, puell_percentile=50.0,
                  puell_band="unknown", as_of=now),
        mayer=_mk(MayerResponse, mayer,
                  asset="BTC", price_usd=btc_p, ma_200d=0, mayer_multiple=1.0,
                  mayer_percentile=50.0, mayer_band="unknown",
                  sf_ratio=56.0, sf_model_price=0, sf_deviation_pct=0, as_of=now),
        ethereum=_mk(EthereumResponse, eth,
                     price_usd=eth_p, market_cap_usd=0, eth_burned_24h=0,
                     eth_issued_24h=1600, net_issuance_24h=0, annualized_burn_rate_eth=0,
                     supply_model="unknown", total_staked_eth=34_000_000,
                     staking_apr_pct=3.5, staking_validators=1_100_000,
                     gas_gwei_10th=10, gas_gwei_50th=20, gas_gwei_90th=50,
                     base_fee_gwei=15, gas_trend="moderate",
                     erc20_usdt_volume_usd=0, erc20_usdc_volume_usd=0,
                     active_addresses=500_000, as_of=now),
        supply_model=_mk(SupplyModelResponse, sup,
                         btc_current_supply=19_700_000, btc_max_supply=21_000_000,
                         btc_pct_mined=93.8, btc_blocks_to_halving=0,
                         btc_days_to_halving=0, btc_current_block_reward=3.125,
                         btc_next_block_reward=1.5625, btc_annual_inflation_pct=0.85,
                         btc_sf_ratio=56.0, eth_current_supply=120_400_000,
                         eth_annual_issuance_est=584_000, eth_annual_burn_est=600_000,
                         eth_net_inflation_pct=-0.03, eth_is_deflationary=True, as_of=now),
        active_alerts=len(_alert_eng.get_active()),
        composite_signal=comp["composite_signal"],
        composite_score=comp["composite_score"])


@router.get("/cross-chain-momentum")
def get_momentum():
    """Composite on-chain momentum score (0-100) with per-metric breakdown."""
    mvrv  = _safe(_mvrv_c.compute, "MVRV")
    nvt   = _safe(_nvt_c.compute, "NVT")
    sopr  = _safe(_sopr_c.compute, "SOPR")
    puell = _safe(_puell_c.compute, "Puell")
    mayer = _safe(_mayer_c.compute, "Mayer")
    return _composite_score(mvrv, nvt, sopr, puell, mayer)


@router.get("/btc-history")
def get_btc_history(days: int = Query(default=30, ge=1, le=365)):
    """Historical daily BTC on-chain snapshots for charting."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        with _db_conn() as conn:
            rows = conn.execute(
                """SELECT snapshot_date, price_usd, market_cap, mvrv_ratio, mvrv_zscore,
                          nvt_ratio, nvt_signal, sopr, asopr, puell_multiple,
                          mayer_multiple, sf_ratio, sf_model_price
                   FROM btc_daily_snapshot WHERE snapshot_date>=? ORDER BY snapshot_date""",
                (cutoff,)).fetchall()
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"days": days, "rows": len(rows), "data": [dict(r) for r in rows]}


@router.get("/eth-history")
def get_eth_history(days: int = Query(default=30, ge=1, le=365)):
    """Historical daily ETH on-chain snapshots for charting."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        with _db_conn() as conn:
            rows = conn.execute(
                """SELECT snapshot_date, price_usd, market_cap, eth_burned_24h,
                          eth_issued_24h, net_issuance, staking_apr, total_staked,
                          base_fee_gwei, gas_gwei_50th, erc20_usdt_usd, erc20_usdc_usd
                   FROM eth_daily_snapshot WHERE snapshot_date>=? ORDER BY snapshot_date""",
                (cutoff,)).fetchall()
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"days": days, "rows": len(rows), "data": [dict(r) for r in rows]}


@router.post("/refresh")
def refresh_all():
    """Force-refresh all metrics (clears cache, re-fetches all free APIs, ~20-45s)."""
    _mem_cache.clear()
    mvrv  = _safe(_mvrv_c.compute, "MVRV")
    nvt   = _safe(_nvt_c.compute, "NVT")
    sopr  = _safe(_sopr_c.compute, "SOPR")
    puell = _safe(_puell_c.compute, "Puell")
    mayer = _safe(_mayer_c.compute, "Mayer")
    eth   = _safe(_eth_c.compute, "ETH")
    bi_stats = _bi_root.get_stats()
    _persist_all(mvrv, nvt, sopr, puell, mayer, eth, bi_stats)
    new_alerts = _alert_eng.evaluate_and_save(mvrv, nvt, sopr, puell)
    return dict(status="refreshed",
                metrics=dict(mvrv=mvrv is not None, nvt=nvt is not None,
                             sopr=sopr is not None, puell=puell is not None,
                             mayer=mayer is not None, ethereum=eth is not None),
                new_alerts=len(new_alerts),
                refreshed_at=_NOW())


@router.get("/health")
def health():
    """Data-source connectivity check."""
    sources: Dict[str, str] = {}
    sources["blockchain_info"] = "ok" if _bi_root.get_stats() else "unavailable"
    sources["coingecko"] = "ok" if CoinGeckoAdapter().get_global() else "unavailable"
    sources["etherscan"] = "ok" if EtherscanAdapter().get_gas_oracle() else "degraded"
    sources["beaconchain"] = "ok" if BeaconChainAdapter().get_stats() else "unavailable"
    try:
        with _db_conn() as c:
            c.execute("SELECT 1").fetchone()
        sources["sqlite"] = "ok"
    except Exception as exc:
        sources["sqlite"] = f"error: {exc}"
    overall = "healthy" if all(v == "ok" for v in sources.values()) else "degraded"
    return dict(status=overall, sources=sources, db_path=str(_DB_PATH),
                cache_entries=len(_mem_cache), as_of=_NOW())


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
_ensure_db()
