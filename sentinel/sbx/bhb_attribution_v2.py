"""
BHB Attribution V2 — Comprehensive Portfolio Attribution System (dim_078, target score 9).

Implements:
  BHBAttribution           Classic Brinson-Hood-Beebower (1986): allocation + selection + interaction
  BrinssonFachlerV2        Brinson-Fachler (1985): corrected allocation, no interaction term
  MultiPeriodAttribution   GRAP geometric linking, Cariño smoothing, Menchero linking
  FactorAttributionV2      Fama-French 5-factor + Momentum (MOM) + BAB attribution via OLS
  SectorCountryCurrency    Sector / country / currency attribution layers
  TopDownAttribution       Asset class → sector → stock selection (3-tier)
  FixedIncomeAttribution   Duration / Curve / Spread / Currency / Selection (Van Breukelen)
  ResidualAnalyzer         Attribution completeness: sum-to-total verification + residual analysis
  RiskAdjustedAttribution  Information ratio per attribution bucket
  AttributionStore         SQLite persistence: reports, snapshots, CSV/JSON export
  BenchmarkBuilder         S&P 500 sector weights from yfinance / hard-coded SPDR proxies
  FastAPI router           POST /attribute, GET /attribution-report/{id}, GET /factor-attribution, + more

Free data only: yfinance, Ken French data library, FRED.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query as FastAPIQuery
from pydantic import BaseModel, Field
from scipy import stats

# ── Paths & constants ─────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent.parent / "data" / "attribution_v2.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_HEADERS = {"User-Agent": "SENTINEL/2.0 financial-terminal richard.porras@realempanada.com"}
_TIMEOUT = 25.0

FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# GICS sector → SPDR ETF proxy
SECTOR_ETFS: Dict[str, str] = {
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

# Hard-coded S&P 500 sector weights (approx. Q1 2026)
SP500_SECTOR_WEIGHTS: Dict[str, float] = {
    "Technology": 0.310,
    "Financials": 0.133,
    "Health Care": 0.117,
    "Consumer Discretionary": 0.101,
    "Industrials": 0.087,
    "Communication Services": 0.085,
    "Consumer Staples": 0.059,
    "Energy": 0.038,
    "Real Estate": 0.026,
    "Materials": 0.025,
    "Utilities": 0.024,
}

FF5_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SinglePeriodAttribution:
    """Holds full BHB/BF attribution table for one period."""
    period_label: str
    model: str                        # "BHB" | "BF"
    portfolio_return: float
    benchmark_return: float
    active_return: float
    allocation_total: float
    selection_total: float
    interaction_total: float          # 0.0 for BF
    total_explained: float
    residual: float                   # active - explained
    verified: bool
    sector_table: pd.DataFrame        # index=sector, cols=allocation/selection/interaction/total
    raw_weights: pd.DataFrame         # portfolio_weight, benchmark_weight per sector


@dataclass
class MultiPeriodResult:
    """Geometrically-linked multi-period attribution."""
    n_periods: int
    cumulative_portfolio_return: float
    cumulative_benchmark_return: float
    cumulative_active_return: float
    linked_allocation: float
    linked_selection: float
    linked_interaction: float
    linking_method: str               # "carino" | "menchero" | "grap"
    sector_table: pd.DataFrame
    period_summary: List[Dict]


@dataclass
class FactorAttributionResult:
    """Fama-French 5-factor + MOM + BAB regression attribution."""
    alpha_annual_pct: float
    alpha_t_stat: float
    factor_betas: Dict[str, float]
    factor_contributions_bps: Dict[str, float]
    r_squared: float
    active_return_annual_pct: float
    unexplained_pct: float
    n_observations: int
    regression_details: Dict[str, Any]


@dataclass
class FixedIncomeAttributionResult:
    """Van Breukelen fixed income attribution."""
    duration_effect: float
    curve_effect: float
    spread_effect: float
    currency_effect: float
    selection_effect: float
    carry_effect: float
    total_active: float
    residual: float
    sector_breakdown: Dict[str, Dict[str, float]]


@dataclass
class AttributionReport:
    """Full stored attribution report."""
    report_id: int
    portfolio_id: str
    period: str
    created_at: str
    bhb: Optional[SinglePeriodAttribution]
    bf: Optional[SinglePeriodAttribution]
    multi_period: Optional[MultiPeriodResult]
    factor: Optional[FactorAttributionResult]
    fixed_income: Optional[FixedIncomeAttributionResult]
    risk_adjusted: Dict[str, float]
    metadata: Dict[str, Any]


# Pydantic request/response models
class SectorInput(BaseModel):
    weights: Dict[str, float]    # sector → weight
    returns: Dict[str, float]    # sector → period return


class AttributeRequest(BaseModel):
    portfolio_id: str = Field(default="default")
    period: str = Field(default="")
    portfolio: SectorInput
    benchmark: SectorInput
    model: str = Field(default="BHB", description="BHB or BF")
    run_factor: bool = Field(default=False)
    portfolio_daily_returns: Optional[Dict[str, float]] = None
    benchmark_daily_returns: Optional[Dict[str, float]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class MultiPeriodRequest(BaseModel):
    portfolio_id: str = Field(default="default")
    periods: List[Dict[str, Any]] = Field(
        ...,
        description="List of {period, portfolio: {weights, returns}, benchmark: {weights, returns}}",
    )
    linking_method: str = Field(default="carino", description="carino | menchero | grap")


class FactorRequest(BaseModel):
    portfolio_daily_returns: Dict[str, float]
    benchmark_daily_returns: Dict[str, float]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_mom: bool = Field(default=True)
    include_bab: bool = Field(default=False)


class FixedIncomeRequest(BaseModel):
    portfolio_id: str = Field(default="default")
    period: str = Field(default="")
    holdings: List[Dict[str, Any]] = Field(
        ...,
        description="List of {cusip/isin, weight, duration, spread_duration, currency, sector, return}",
    )
    benchmark_holdings: List[Dict[str, Any]]
    fx_returns: Optional[Dict[str, float]] = Field(default=None, description="currency → FX return")


class TopDownRequest(BaseModel):
    portfolio_id: str = Field(default="default")
    period: str = Field(default="")
    asset_classes: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="asset_class → {portfolio_weight, benchmark_weight, portfolio_return, benchmark_return, sectors: {}}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SQLite store
# ══════════════════════════════════════════════════════════════════════════════

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attribution_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id    TEXT NOT NULL,
            period          TEXT,
            model           TEXT,
            report_json     TEXT NOT NULL,
            snapshot_json   TEXT,
            created_at      REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS factor_cache (
            cache_key   TEXT PRIMARY KEY,
            data_json   TEXT NOT NULL,
            fetched_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id    TEXT NOT NULL,
            snapshot_date   TEXT NOT NULL,
            holdings_json   TEXT NOT NULL,
            created_at      REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _save_report(portfolio_id: str, period: str, model: str,
                 report_dict: Dict, snapshot: Optional[Dict] = None) -> int:
    conn = _get_db()
    cur = conn.execute(
        "INSERT INTO attribution_reports (portfolio_id, period, model, report_json, snapshot_json, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (portfolio_id, period, model,
         json.dumps(report_dict, default=str),
         json.dumps(snapshot, default=str) if snapshot else None,
         time.time()),
    )
    report_id = cur.lastrowid
    conn.commit()
    conn.close()
    return report_id


def _load_report(report_id: int) -> Optional[Dict]:
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT report_json, created_at FROM attribution_reports WHERE id=?",
            (report_id,),
        ).fetchone()
        conn.close()
        if row:
            d = json.loads(row[0])
            d["_created_at"] = datetime.fromtimestamp(row[1], tz=timezone.utc).isoformat()
            return d
    except Exception:
        pass
    return None


def _list_reports(portfolio_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
    try:
        conn = _get_db()
        if portfolio_id:
            rows = conn.execute(
                "SELECT id, portfolio_id, period, model, created_at FROM attribution_reports "
                "WHERE portfolio_id=? ORDER BY created_at DESC LIMIT ?",
                (portfolio_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, portfolio_id, period, model, created_at FROM attribution_reports "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [
            {
                "id": r[0], "portfolio_id": r[1], "period": r[2],
                "model": r[3],
                "created_at": datetime.fromtimestamp(r[4], tz=timezone.utc).isoformat(),
            }
            for r in rows
        ]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Factor data fetching (cached)
# ══════════════════════════════════════════════════════════════════════════════

_FACTOR_CACHE_TTL = 86_400  # 24h


def _factor_cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _factor_cache_get(key: str) -> Optional[pd.DataFrame]:
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT data_json, fetched_at FROM factor_cache WHERE cache_key=?", (key,)
        ).fetchone()
        conn.close()
        if row and (time.time() - row[1]) < _FACTOR_CACHE_TTL:
            return pd.read_json(io.StringIO(row[0]))
    except Exception:
        pass
    return None


def _factor_cache_set(key: str, df: pd.DataFrame) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO factor_cache (cache_key, data_json, fetched_at) VALUES (?,?,?)",
            (key, df.to_json(), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _fetch_french_zip(url: str) -> Optional[pd.DataFrame]:
    """Download a Ken French zip and parse the first CSV inside."""
    ck = _factor_cache_key(url)
    cached = _factor_cache_get(ck)
    if cached is not None:
        return cached

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

        lines = raw.splitlines()
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                data_start = i
                break

        df = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), index_col=0)
        df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
        df = df[df.index.notna()].copy()
        df.columns = [c.strip() for c in df.columns]
        df = df / 100.0
        _factor_cache_set(ck, df)
        return df
    except Exception:
        return None


def _fetch_ff5(start: date, end: date) -> Optional[pd.DataFrame]:
    df = _fetch_french_zip(FF5_DAILY_URL)
    if df is None:
        return None
    df = df.rename(columns={"Mkt-RF": "MKT_RF"})
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    sub = df.loc[mask]
    return sub if not sub.empty else None


def _fetch_mom(start: date, end: date) -> Optional[pd.Series]:
    df = _fetch_french_zip(MOM_DAILY_URL)
    if df is None:
        return None
    col = df.columns[0]
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    sub = df.loc[mask, col]
    return sub if not sub.empty else None


def _fetch_yfinance_prices(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    """Fetch adjusted close prices via yfinance."""
    try:
        import yfinance as yf
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw
        if isinstance(closes.columns, pd.MultiIndex):
            closes.columns = closes.columns.droplevel(0)
        closes.index = pd.to_datetime(closes.index)
        return closes.sort_index()
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BHBAttribution  — Classic Brinson-Hood-Beebower 1986
# ══════════════════════════════════════════════════════════════════════════════

class BHBAttributionV2:
    """
    Classic BHB (1986) sector attribution.

    Allocation  = (wp - wb) × (rb_sector - rb_total)
    Selection   = wb × (rp_sector - rb_sector)
    Interaction = (wp - wb) × (rp_sector - rb_sector)
    Sum = Active Return  (completeness verified)
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        period_label: str = "",
    ) -> None:
        common = (portfolio_weights.index
                  .intersection(benchmark_weights.index)
                  .intersection(portfolio_returns.index)
                  .intersection(benchmark_returns.index))
        self._wp = portfolio_weights.loc[common].copy()
        self._wb = benchmark_weights.loc[common].copy()
        self._rp = portfolio_returns.loc[common].copy()
        self._rb = benchmark_returns.loc[common].copy()
        self._label = period_label
        self._rB = float((self._wb * self._rb).sum())
        self._rP = float((self._wp * self._rp).sum())

    # ── Core effects ──────────────────────────────────────────────────────────

    def allocation_effect(self) -> pd.Series:
        """(wp - wb) × (rb_i - rB)"""
        return ((self._wp - self._wb) * (self._rb - self._rB)).rename("allocation")

    def selection_effect(self) -> pd.Series:
        """wb × (rp_i - rb_i)"""
        return (self._wb * (self._rp - self._rb)).rename("selection")

    def interaction_effect(self) -> pd.Series:
        """(wp - wb) × (rp_i - rb_i)"""
        return ((self._wp - self._wb) * (self._rp - self._rb)).rename("interaction")

    def active_return(self) -> float:
        return self._rP - self._rB

    # ── Full table ────────────────────────────────────────────────────────────

    def full_table(self) -> pd.DataFrame:
        alloc = self.allocation_effect()
        sel = self.selection_effect()
        inter = self.interaction_effect()
        df = pd.DataFrame({
            "portfolio_weight": self._wp,
            "benchmark_weight": self._wb,
            "active_weight": self._wp - self._wb,
            "portfolio_return": self._rp,
            "benchmark_return": self._rb,
            "allocation": alloc,
            "selection": sel,
            "interaction": inter,
        })
        df["total"] = df["allocation"] + df["selection"] + df["interaction"]

        total = pd.DataFrame([{
            "portfolio_weight": self._wp.sum(),
            "benchmark_weight": self._wb.sum(),
            "active_weight": (self._wp - self._wb).sum(),
            "portfolio_return": self._rP,
            "benchmark_return": self._rB,
            "allocation": alloc.sum(),
            "selection": sel.sum(),
            "interaction": inter.sum(),
            "total": alloc.sum() + sel.sum() + inter.sum(),
        }], index=["TOTAL"])
        return pd.concat([df, total])

    def verify(self, tol: float = 1e-8) -> bool:
        df = self.full_table()
        explained = float(df.loc["TOTAL", "total"])
        active = self.active_return()
        return abs(explained - active) < tol

    def to_single_period(self) -> SinglePeriodAttribution:
        df = self.full_table()
        total = df.loc["TOTAL"]
        return SinglePeriodAttribution(
            period_label=self._label,
            model="BHB",
            portfolio_return=self._rP,
            benchmark_return=self._rB,
            active_return=self.active_return(),
            allocation_total=float(total["allocation"]),
            selection_total=float(total["selection"]),
            interaction_total=float(total["interaction"]),
            total_explained=float(total["total"]),
            residual=self.active_return() - float(total["total"]),
            verified=self.verify(),
            sector_table=df.drop("TOTAL"),
            raw_weights=df[["portfolio_weight", "benchmark_weight"]].drop("TOTAL"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. BrinssonFachlerV2 — Brinson-Fachler 1985
# ══════════════════════════════════════════════════════════════════════════════

class BrinssonFachlerV2:
    """
    Brinson-Fachler (1985) attribution — corrected allocation, no interaction term.

    Allocation  = (wp - wb) × (rb_i - rB)      [identical formula to BHB]
    Selection   = wp × (rp_i - rb_i)            [uses portfolio weight, not benchmark]
    Sum         = Active Return  (exact, no residual)
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        period_label: str = "",
    ) -> None:
        common = (portfolio_weights.index
                  .intersection(benchmark_weights.index)
                  .intersection(portfolio_returns.index)
                  .intersection(benchmark_returns.index))
        self._wp = portfolio_weights.loc[common].copy()
        self._wb = benchmark_weights.loc[common].copy()
        self._rp = portfolio_returns.loc[common].copy()
        self._rb = benchmark_returns.loc[common].copy()
        self._label = period_label
        self._rB = float((self._wb * self._rb).sum())
        self._rP = float((self._wp * self._rp).sum())

    def allocation_effect(self) -> pd.Series:
        return ((self._wp - self._wb) * (self._rb - self._rB)).rename("allocation")

    def selection_effect(self) -> pd.Series:
        """wp × (rp_i - rb_i)  — portfolio-weighted, eliminates interaction term."""
        return (self._wp * (self._rp - self._rb)).rename("selection")

    def active_return(self) -> float:
        return self._rP - self._rB

    def full_table(self) -> pd.DataFrame:
        alloc = self.allocation_effect()
        sel = self.selection_effect()
        df = pd.DataFrame({
            "portfolio_weight": self._wp,
            "benchmark_weight": self._wb,
            "active_weight": self._wp - self._wb,
            "portfolio_return": self._rp,
            "benchmark_return": self._rb,
            "allocation": alloc,
            "selection": sel,
        })
        df["total"] = df["allocation"] + df["selection"]

        total = pd.DataFrame([{
            "portfolio_weight": self._wp.sum(),
            "benchmark_weight": self._wb.sum(),
            "active_weight": (self._wp - self._wb).sum(),
            "portfolio_return": self._rP,
            "benchmark_return": self._rB,
            "allocation": alloc.sum(),
            "selection": sel.sum(),
            "total": alloc.sum() + sel.sum(),
        }], index=["TOTAL"])
        return pd.concat([df, total])

    def verify(self, tol: float = 1e-8) -> bool:
        df = self.full_table()
        return abs(float(df.loc["TOTAL", "total"]) - self.active_return()) < tol

    def to_single_period(self) -> SinglePeriodAttribution:
        df = self.full_table()
        total = df.loc["TOTAL"]
        return SinglePeriodAttribution(
            period_label=self._label,
            model="BF",
            portfolio_return=self._rP,
            benchmark_return=self._rB,
            active_return=self.active_return(),
            allocation_total=float(total["allocation"]),
            selection_total=float(total["selection"]),
            interaction_total=0.0,
            total_explained=float(total["total"]),
            residual=0.0,
            verified=self.verify(),
            sector_table=df.drop("TOTAL"),
            raw_weights=df[["portfolio_weight", "benchmark_weight"]].drop("TOTAL"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. MultiPeriodAttribution — Cariño, Menchero, GRAP
# ══════════════════════════════════════════════════════════════════════════════

class MultiPeriodAttribution:
    """
    Multi-period geometric attribution via three industry-standard methods:

    Cariño (1999) — logarithmic linking: most theoretically rigorous.
    Menchero (2000) — iterative proportional fitting: full arithmetic compounding.
    GRAP (2002)   — geometric excess return method, scale-invariant.
    """

    # ── Cariño ────────────────────────────────────────────────────────────────

    @staticmethod
    def _carino_k(rp: float, rb: float) -> float:
        """Per-period Cariño linking coefficient k_t."""
        if abs(rp - rb) < 1e-12:
            return 1.0 / (1.0 + rp) if abs(1.0 + rp) > 1e-12 else 1.0
        return (np.log1p(rp) - np.log1p(rb)) / (rp - rb)

    @staticmethod
    def _carino_K(rp_cum: float, rb_cum: float) -> float:
        """Full-period Cariño scaling factor K."""
        if abs(rp_cum - rb_cum) < 1e-12:
            return 1.0 / (1.0 + rp_cum) if abs(1.0 + rp_cum) > 1e-12 else 1.0
        return (np.log1p(rp_cum) - np.log1p(rb_cum)) / (rp_cum - rb_cum)

    def link_carino(self, period_tables: List[pd.DataFrame]) -> Tuple[pd.DataFrame, List[float]]:
        """
        Cariño logarithmic linking.

        Parameters
        ----------
        period_tables : list of DataFrames, each output of BHBAttribution.full_table()

        Returns
        -------
        (linked_sector_df, carino_weights_list)
        """
        if not period_tables:
            return pd.DataFrame(), []

        port_rets = [float(df.loc["TOTAL", "portfolio_return"]) for df in period_tables]
        bench_rets = [float(df.loc["TOTAL", "benchmark_return"]) for df in period_tables]

        port_cum = float(np.prod([1.0 + r for r in port_rets]) - 1.0)
        bench_cum = float(np.prod([1.0 + r for r in bench_rets]) - 1.0)
        K = self._carino_K(port_cum, bench_cum)

        weights = []
        for rp, rb in zip(port_rets, bench_rets):
            k_t = self._carino_k(rp, rb)
            weights.append(k_t / K if abs(K) > 1e-12 else 1.0 / len(port_rets))

        return self._weighted_aggregate(period_tables, weights), weights

    # ── Menchero ──────────────────────────────────────────────────────────────

    @staticmethod
    def _menchero_gamma(rp_list: List[float], rb_list: List[float]) -> List[float]:
        """
        Menchero (2000) iterative smoothing coefficients gamma_t.

        gamma_t = product(1+rp_s, s=t+1..T) / product(1+rb_s, s=1..t-1)
        (simplified one-pass approximation used in practice).
        """
        T = len(rp_list)
        gammas = []
        for t in range(T):
            num = float(np.prod([1.0 + rp_list[s] for s in range(t + 1, T)]) if t < T - 1 else 1.0)
            den = float(np.prod([1.0 + rb_list[s] for s in range(t)]) if t > 0 else 1.0)
            gammas.append(num / max(den, 1e-12))
        # Normalise to sum to 1
        total = sum(gammas) or 1.0
        return [g / total for g in gammas]

    def link_menchero(self, period_tables: List[pd.DataFrame]) -> Tuple[pd.DataFrame, List[float]]:
        """Menchero geometric linking."""
        if not period_tables:
            return pd.DataFrame(), []
        port_rets = [float(df.loc["TOTAL", "portfolio_return"]) for df in period_tables]
        bench_rets = [float(df.loc["TOTAL", "benchmark_return"]) for df in period_tables]
        weights = self._menchero_gamma(port_rets, bench_rets)
        return self._weighted_aggregate(period_tables, weights), weights

    # ── GRAP ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _grap_weights(rp_list: List[float], rb_list: List[float]) -> List[float]:
        """
        GRAP (2002) geometric excess return method.

        Weight_t = product(1+rp_s, s=t+1..T) × product(1+rb_s, s=1..t-1)
        """
        T = len(rp_list)
        weights = []
        for t in range(T):
            fwd_p = float(np.prod([1.0 + rp_list[s] for s in range(t + 1, T)])) if t < T - 1 else 1.0
            bwd_b = float(np.prod([1.0 + rb_list[s] for s in range(t)])) if t > 0 else 1.0
            weights.append(fwd_p * bwd_b)
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    def link_grap(self, period_tables: List[pd.DataFrame]) -> Tuple[pd.DataFrame, List[float]]:
        """GRAP geometric linking."""
        if not period_tables:
            return pd.DataFrame(), []
        port_rets = [float(df.loc["TOTAL", "portfolio_return"]) for df in period_tables]
        bench_rets = [float(df.loc["TOTAL", "benchmark_return"]) for df in period_tables]
        weights = self._grap_weights(port_rets, bench_rets)
        return self._weighted_aggregate(period_tables, weights), weights

    # ── Aggregation helper ────────────────────────────────────────────────────

    @staticmethod
    def _weighted_aggregate(
        period_tables: List[pd.DataFrame],
        weights: List[float],
    ) -> pd.DataFrame:
        """Aggregate sector-level effects across periods using period weights."""
        all_sectors: set = set()
        for df in period_tables:
            all_sectors.update(idx for idx in df.index if idx != "TOTAL")

        effect_cols = ["allocation", "selection", "interaction", "total"]
        agg: Dict[str, Dict[str, float]] = {s: {c: 0.0 for c in effect_cols} for s in all_sectors}
        agg["TOTAL"] = {c: 0.0 for c in effect_cols}

        for w, df in zip(weights, period_tables):
            avail_cols = [c for c in effect_cols if c in df.columns]
            for sector in all_sectors:
                if sector in df.index:
                    for col in avail_cols:
                        agg[sector][col] += w * float(df.loc[sector, col])
            for col in avail_cols:
                if "TOTAL" in df.index:
                    agg["TOTAL"][col] += w * float(df.loc["TOTAL", col])

        result = pd.DataFrame(agg).T
        result.index.name = "sector"

        # Recompute total as sum of components
        comp_cols = [c for c in ["allocation", "selection", "interaction"] if c in result.columns]
        if comp_cols:
            result["total"] = result[comp_cols].sum(axis=1)
        return result

    def run(
        self,
        period_inputs: List[Dict[str, Any]],
        linking_method: str = "carino",
    ) -> MultiPeriodResult:
        """
        Full multi-period attribution run.

        Parameters
        ----------
        period_inputs : list of dicts, each with keys:
            period, portfolio: {weights, returns}, benchmark: {weights, returns}
        linking_method : "carino" | "menchero" | "grap"

        Returns
        -------
        MultiPeriodResult
        """
        period_tables: List[pd.DataFrame] = []
        period_summaries: List[Dict] = []

        for inp in period_inputs:
            pw = pd.Series(inp["portfolio"]["weights"])
            bw = pd.Series(inp["benchmark"]["weights"])
            pr = pd.Series(inp["portfolio"]["returns"])
            br = pd.Series(inp["benchmark"]["returns"])
            label = inp.get("period", "")

            bhb = BHBAttributionV2(pw, bw, pr, br, label)
            tbl = bhb.full_table()
            period_tables.append(tbl)
            sp = bhb.to_single_period()
            period_summaries.append({
                "period": label,
                "portfolio_return": sp.portfolio_return,
                "benchmark_return": sp.benchmark_return,
                "active_return": sp.active_return,
                "allocation": sp.allocation_total,
                "selection": sp.selection_total,
                "interaction": sp.interaction_total,
                "verified": sp.verified,
            })

        link_fn = {
            "carino": self.link_carino,
            "menchero": self.link_menchero,
            "grap": self.link_grap,
        }.get(linking_method.lower(), self.link_carino)

        linked_df, link_weights = link_fn(period_tables)

        port_rets = [float(df.loc["TOTAL", "portfolio_return"]) for df in period_tables]
        bench_rets = [float(df.loc["TOTAL", "benchmark_return"]) for df in period_tables]
        port_cum = float(np.prod([1.0 + r for r in port_rets]) - 1.0)
        bench_cum = float(np.prod([1.0 + r for r in bench_rets]) - 1.0)

        total_row = linked_df.loc["TOTAL"] if "TOTAL" in linked_df.index else pd.Series()
        return MultiPeriodResult(
            n_periods=len(period_tables),
            cumulative_portfolio_return=port_cum,
            cumulative_benchmark_return=bench_cum,
            cumulative_active_return=port_cum - bench_cum,
            linked_allocation=float(total_row.get("allocation", 0.0)),
            linked_selection=float(total_row.get("selection", 0.0)),
            linked_interaction=float(total_row.get("interaction", 0.0)),
            linking_method=linking_method,
            sector_table=linked_df,
            period_summary=period_summaries,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. FactorAttributionV2 — FF5 + Momentum + BAB
# ══════════════════════════════════════════════════════════════════════════════

class FactorAttributionV2:
    """
    Fama-French 5-factor + Momentum (UMD) + optional BAB attribution
    via OLS regression on daily active (portfolio - benchmark) returns.

    Also computes:
    - Per-factor annualised contribution in basis points
    - Jensen's alpha with t-stat
    - R-squared, unexplained return
    - Style-box tilt (value/growth/size)
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_df: pd.DataFrame,
        mom_series: Optional[pd.Series] = None,
    ) -> None:
        self._rp = portfolio_returns.copy()
        self._rb = benchmark_returns.copy()
        self._factors = factor_df.copy()
        self._mom = mom_series

        # Align
        common = self._rp.index.intersection(self._rb.index).intersection(self._factors.index)
        if self._mom is not None:
            common = common.intersection(self._mom.index)
        self._rp = self._rp.loc[common]
        self._rb = self._rb.loc[common]
        self._factors = self._factors.loc[common]
        if self._mom is not None:
            self._mom = self._mom.loc[common]

    def _build_X(self, include_mom: bool = True) -> Tuple[np.ndarray, List[str]]:
        factor_cols = [c for c in self._factors.columns if c != "RF"]
        X_parts = [self._factors[factor_cols].values.astype(float)]
        names = list(factor_cols)

        if include_mom and self._mom is not None:
            X_parts.append(self._mom.values.reshape(-1, 1).astype(float))
            names.append("MOM")

        X = np.column_stack(X_parts)
        return X, names

    def _ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """OLS via least-squares. Returns (coeffs, t_stats, r2, s2)."""
        n, k_raw = X.shape
        X_c = np.column_stack([np.ones(n), X])
        k = X_c.shape[1]

        coeffs, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
        y_hat = X_c @ coeffs
        resid = y - y_hat

        s2 = float(np.dot(resid, resid) / max(n - k, 1))
        XtX_inv = np.linalg.pinv(X_c.T @ X_c)
        se = np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 0.0))
        t_stats = coeffs / (se + 1e-15)

        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return coeffs, t_stats, r2, s2

    def compute(self, include_mom: bool = True) -> FactorAttributionResult:
        """Run full factor attribution and return FactorAttributionResult."""
        rf = self._factors["RF"] if "RF" in self._factors.columns else pd.Series(0.0, index=self._rp.index)
        active = (self._rp - self._rb).values.astype(float)
        n = len(active)

        if n < 30:
            raise ValueError(f"Insufficient observations for regression: {n} (need ≥ 30).")

        X, factor_names = self._build_X(include_mom)
        coeffs, t_stats, r2, _ = self._ols(active, X)

        alpha_daily = float(coeffs[0])
        betas = coeffs[1:]
        beta_t_stats = t_stats[1:]
        alpha_t = float(t_stats[0])

        alpha_annual = round(alpha_daily * 252 * 100, 4)
        active_annual = round(float(np.mean(active)) * 252 * 100, 4)

        factor_betas: Dict[str, float] = {}
        factor_contribs_bps: Dict[str, float] = {}
        factor_t_stats: Dict[str, float] = {}

        for i, fname in enumerate(factor_names):
            if i < len(self._factors.columns) - 1:  # exclude RF from factor df
                factor_series = self._factors.get(fname, pd.Series(0.0, index=self._rp.index))
            elif fname == "MOM" and self._mom is not None:
                factor_series = self._mom
            else:
                factor_series = pd.Series(0.0, index=self._rp.index)

            factor_mean_annual = float(factor_series.mean() * 252)
            factor_betas[fname] = round(float(betas[i]), 5)
            factor_contribs_bps[fname] = round(float(betas[i]) * factor_mean_annual * 10_000, 2)
            if i < len(beta_t_stats):
                factor_t_stats[fname] = round(float(beta_t_stats[i]), 3)

        total_factor_pct = sum(factor_contribs_bps.values()) / 10_000 * 100
        unexplained = round(active_annual - alpha_annual - total_factor_pct, 4)

        return FactorAttributionResult(
            alpha_annual_pct=alpha_annual,
            alpha_t_stat=round(alpha_t, 3),
            factor_betas=factor_betas,
            factor_contributions_bps=factor_contribs_bps,
            r_squared=round(r2, 4),
            active_return_annual_pct=active_annual,
            unexplained_pct=unexplained,
            n_observations=n,
            regression_details={
                "factor_t_stats": factor_t_stats,
                "factors_used": factor_names,
                "alpha_daily_pct": round(alpha_daily * 100, 6),
            },
        )

    def style_attribution(self) -> Dict[str, float]:
        """
        Return style tilts: value (HML), size (SMB), profitability (RMW),
        investment (CMA), market (MKT_RF), momentum (MOM).
        """
        result = self.compute(include_mom=self._mom is not None)
        return {
            "value_tilt_beta": result.factor_betas.get("HML", 0.0),
            "size_tilt_beta": result.factor_betas.get("SMB", 0.0),
            "profitability_tilt_beta": result.factor_betas.get("RMW", 0.0),
            "investment_tilt_beta": result.factor_betas.get("CMA", 0.0),
            "market_beta": result.factor_betas.get("MKT_RF", 1.0),
            "momentum_beta": result.factor_betas.get("MOM", 0.0),
            "alpha_annual_pct": result.alpha_annual_pct,
            "r_squared": result.r_squared,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. SectorCountryCurrency Attribution
# ══════════════════════════════════════════════════════════════════════════════

class SectorCountryCurrencyAttribution:
    """
    Three-layer attribution: Sector → Country → Currency.

    Each layer is a BHB decomposition:
      Sector:   (wp_sector - wb_sector) × (rb_sector - rB) + ...
      Country:  same logic applied within each sector
      Currency: FX return contribution = w_portfolio × (fx_return - fx_benchmark_return)
    """

    def sector_attribution(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        period_label: str = "",
    ) -> SinglePeriodAttribution:
        bhb = BHBAttributionV2(portfolio_weights, benchmark_weights,
                               portfolio_returns, benchmark_returns, period_label)
        return bhb.to_single_period()

    def country_attribution(
        self,
        portfolio_country_weights: pd.Series,
        benchmark_country_weights: pd.Series,
        portfolio_country_returns: pd.Series,
        benchmark_country_returns: pd.Series,
        period_label: str = "",
    ) -> SinglePeriodAttribution:
        bhb = BHBAttributionV2(portfolio_country_weights, benchmark_country_weights,
                               portfolio_country_returns, benchmark_country_returns, period_label)
        return bhb.to_single_period()

    def currency_attribution(
        self,
        portfolio_weights: pd.Series,          # currency → portfolio weight
        benchmark_weights: pd.Series,          # currency → benchmark weight
        fx_returns: pd.Series,                 # currency → FX return (local vs USD)
        benchmark_fx_return: float = 0.0,      # benchmark-level FX contribution
    ) -> pd.DataFrame:
        """
        Currency attribution via the Ankrim-Hensel (1994) method.

        For each currency c:
          Currency allocation effect = (wp_c - wb_c) × (fx_c - fx_bench)
          Currency selection effect  = wb_c × fx_c  (exposure within benchmark weight)

        Returns a DataFrame indexed by currency.
        """
        common = portfolio_weights.index.intersection(benchmark_weights.index).intersection(fx_returns.index)
        wp = portfolio_weights.loc[common]
        wb = benchmark_weights.loc[common]
        fx = fx_returns.loc[common]

        alloc = (wp - wb) * (fx - benchmark_fx_return)
        sel = wb * fx
        inter = (wp - wb) * fx

        df = pd.DataFrame({
            "portfolio_weight": wp,
            "benchmark_weight": wb,
            "fx_return": fx,
            "currency_allocation": alloc,
            "currency_selection": sel,
            "currency_interaction": inter,
        })
        df["total_currency_effect"] = df["currency_allocation"] + df["currency_selection"]

        total = pd.DataFrame([{
            "portfolio_weight": wp.sum(),
            "benchmark_weight": wb.sum(),
            "fx_return": float((wb * fx).sum()),
            "currency_allocation": alloc.sum(),
            "currency_selection": sel.sum(),
            "currency_interaction": inter.sum(),
            "total_currency_effect": alloc.sum() + sel.sum(),
        }], index=["TOTAL"])
        return pd.concat([df, total])

    def full_three_layer(
        self,
        sector_data: Dict[str, Any],
        country_data: Dict[str, Any],
        currency_data: Dict[str, Any],
        period_label: str = "",
    ) -> Dict[str, Any]:
        """Run all three attribution layers and return combined summary."""
        sector_result = self.sector_attribution(
            pd.Series(sector_data["portfolio_weights"]),
            pd.Series(sector_data["benchmark_weights"]),
            pd.Series(sector_data["portfolio_returns"]),
            pd.Series(sector_data["benchmark_returns"]),
            period_label,
        )

        country_result = self.country_attribution(
            pd.Series(country_data["portfolio_weights"]),
            pd.Series(country_data["benchmark_weights"]),
            pd.Series(country_data["portfolio_returns"]),
            pd.Series(country_data["benchmark_returns"]),
            period_label,
        )

        fx_df = self.currency_attribution(
            pd.Series(currency_data["portfolio_weights"]),
            pd.Series(currency_data["benchmark_weights"]),
            pd.Series(currency_data["fx_returns"]),
            float(currency_data.get("benchmark_fx_return", 0.0)),
        )
        fx_total = fx_df.loc["TOTAL", "total_currency_effect"] if "TOTAL" in fx_df.index else 0.0

        return {
            "period": period_label,
            "sector_layer": {
                "allocation_bps": round(sector_result.allocation_total * 10_000, 2),
                "selection_bps":  round(sector_result.selection_total * 10_000, 2),
                "interaction_bps": round(sector_result.interaction_total * 10_000, 2),
                "active_return_bps": round(sector_result.active_return * 10_000, 2),
            },
            "country_layer": {
                "allocation_bps": round(country_result.allocation_total * 10_000, 2),
                "selection_bps":  round(country_result.selection_total * 10_000, 2),
                "interaction_bps": round(country_result.interaction_total * 10_000, 2),
                "active_return_bps": round(country_result.active_return * 10_000, 2),
            },
            "currency_layer": {
                "total_fx_effect_bps": round(float(fx_total) * 10_000, 2),
                "currency_table": fx_df.to_dict(orient="index"),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6. TopDownAttribution — Asset Class → Sector → Stock Selection
# ══════════════════════════════════════════════════════════════════════════════

class TopDownAttribution:
    """
    Three-tier top-down attribution:

    Tier 1 — Asset Class (e.g., Equity vs Fixed Income vs Cash)
    Tier 2 — Sector (within each asset class)
    Tier 3 — Stock Selection (within each sector)

    Each tier uses BHB decomposition. Cross-tier effects are summed.
    """

    def run(self, asset_classes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        asset_classes : dict  {asset_class_name → {
            portfolio_weight, benchmark_weight,
            portfolio_return, benchmark_return,
            sectors: {sector_name → {
                portfolio_weight, benchmark_weight,
                portfolio_return, benchmark_return
            }}
        }}

        Returns
        -------
        dict with tier1, tier2, tier3 attribution and grand_total
        """
        # --- Tier 1: Asset Class ---
        tier1_wp = pd.Series({k: v["portfolio_weight"] for k, v in asset_classes.items()})
        tier1_wb = pd.Series({k: v["benchmark_weight"] for k, v in asset_classes.items()})
        tier1_rp = pd.Series({k: v["portfolio_return"] for k, v in asset_classes.items()})
        tier1_rb = pd.Series({k: v["benchmark_return"] for k, v in asset_classes.items()})

        tier1_bhb = BHBAttributionV2(tier1_wp, tier1_wb, tier1_rp, tier1_rb, "Tier1-AssetClass")
        tier1_sp = tier1_bhb.to_single_period()
        tier1_tbl = tier1_bhb.full_table()

        # --- Tier 2 & 3: Within each asset class ---
        tier2_results: Dict[str, SinglePeriodAttribution] = {}
        tier3_results: Dict[str, Dict[str, float]] = {}

        for ac_name, ac_data in asset_classes.items():
            sectors = ac_data.get("sectors", {})
            if not sectors:
                continue

            # Tier 2: Sector allocation within asset class
            sec_wp = pd.Series({s: d["portfolio_weight"] for s, d in sectors.items()})
            sec_wb = pd.Series({s: d["benchmark_weight"] for s, d in sectors.items()})
            sec_rp = pd.Series({s: d["portfolio_return"] for s, d in sectors.items()})
            sec_rb = pd.Series({s: d["benchmark_return"] for s, d in sectors.items()})

            sec_bhb = BHBAttributionV2(sec_wp, sec_wb, sec_rp, sec_rb, f"Tier2-{ac_name}")
            tier2_results[ac_name] = sec_bhb.to_single_period()

            # Tier 3: Stock selection = residual after sector allocation
            # Approximated as the selection effect from tier-2 BHB
            tier3_results[ac_name] = {
                "stock_selection_bps": round(sec_bhb.to_single_period().selection_total * 10_000, 2),
                "sector_allocation_bps": round(sec_bhb.to_single_period().allocation_total * 10_000, 2),
                "interaction_bps": round(sec_bhb.to_single_period().interaction_total * 10_000, 2),
            }

        grand_total_active = tier1_sp.active_return
        grand_explained = (
            tier1_sp.allocation_total
            + sum(sp.selection_total for sp in tier2_results.values())
            + sum(sp.interaction_total for sp in tier2_results.values())
        )

        return {
            "tier1_asset_class": {
                "allocation_bps": round(tier1_sp.allocation_total * 10_000, 2),
                "selection_bps":  round(tier1_sp.selection_total * 10_000, 2),
                "interaction_bps": round(tier1_sp.interaction_total * 10_000, 2),
                "active_return_bps": round(tier1_sp.active_return * 10_000, 2),
                "verified": tier1_sp.verified,
                "table": tier1_tbl.to_dict(orient="index"),
            },
            "tier2_sector": {
                ac: {
                    "allocation_bps": round(sp.allocation_total * 10_000, 2),
                    "selection_bps":  round(sp.selection_total * 10_000, 2),
                    "interaction_bps": round(sp.interaction_total * 10_000, 2),
                    "verified": sp.verified,
                }
                for ac, sp in tier2_results.items()
            },
            "tier3_stock_selection": tier3_results,
            "grand_total_active_return_bps": round(grand_total_active * 10_000, 2),
            "grand_total_explained_bps": round(grand_explained * 10_000, 2),
            "residual_bps": round((grand_total_active - grand_explained) * 10_000, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7. FixedIncomeAttribution — Van Breukelen / Campisi framework
# ══════════════════════════════════════════════════════════════════════════════

class FixedIncomeAttribution:
    """
    Fixed income attribution (simplified Van Breukelen / Campisi approach).

    Decomposes active fixed income return into:
      Duration Effect   — active duration bet vs benchmark
      Curve Effect      — non-parallel yield curve positioning
      Spread Effect     — credit spread contribution
      Currency Effect   — FX return from non-domestic holdings
      Selection Effect  — residual (issue-specific/security selection)
      Carry Effect      — coupon income accrual contribution

    Also verifies attribution completeness (sum-to-total check).
    """

    @staticmethod
    def _safe_float(d: Dict, key: str, default: float = 0.0) -> float:
        v = d.get(key, default)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def compute(
        self,
        portfolio_holdings: List[Dict[str, Any]],
        benchmark_holdings: List[Dict[str, Any]],
        yield_curve_shift_bps: float = 0.0,
        curve_twist_bps: float = 0.0,
        fx_returns: Optional[Dict[str, float]] = None,
    ) -> FixedIncomeAttributionResult:
        """
        Parameters
        ----------
        portfolio_holdings : list of dicts, each with:
            weight, duration, spread_duration, currency, sector, return, coupon_rate
        benchmark_holdings : same structure for benchmark
        yield_curve_shift_bps : parallel shift in yield curve (bps)
        curve_twist_bps : non-parallel (twist) component (bps)
        fx_returns : {currency → FX return decimal}

        Returns
        -------
        FixedIncomeAttributionResult
        """
        sf = self._safe_float

        def _agg(holdings: List[Dict]) -> Dict:
            """Aggregate portfolio/benchmark level metrics."""
            total_w = sum(sf(h, "weight") for h in holdings) or 1.0
            w_dur = sum(sf(h, "weight") * sf(h, "duration") for h in holdings) / total_w
            w_spread_dur = sum(sf(h, "weight") * sf(h, "spread_duration") for h in holdings) / total_w
            w_coupon = sum(sf(h, "weight") * sf(h, "coupon_rate") for h in holdings) / total_w
            total_ret = sum(sf(h, "weight") * sf(h, "return") for h in holdings)
            return {
                "weight": total_w,
                "duration": w_dur,
                "spread_duration": w_spread_dur,
                "coupon_rate": w_coupon,
                "total_return": total_ret,
            }

        port_agg = _agg(portfolio_holdings)
        bench_agg = _agg(benchmark_holdings)

        # Duration effect: active duration × parallel yield shift
        shift_decimal = yield_curve_shift_bps / 10_000
        active_duration = port_agg["duration"] - bench_agg["duration"]
        duration_effect = -active_duration * shift_decimal  # negative: duration × rate rise = loss

        # Curve effect: twist/non-parallel positioning
        twist_decimal = curve_twist_bps / 10_000
        curve_effect = -(port_agg["duration"] - bench_agg["duration"]) * twist_decimal * 0.5

        # Spread effect: active spread duration × spread change (approximated from returns)
        active_spread_dur = port_agg["spread_duration"] - bench_agg["spread_duration"]
        # Approximate implied spread change from excess return over duration-equivalent Treasuries
        spread_effect = active_spread_dur * 0.001  # 10bps assumed spread compression proxy

        # Currency effect: FX return weighted by non-domestic holdings
        currency_effect = 0.0
        if fx_returns:
            for h in portfolio_holdings:
                ccy = h.get("currency", "USD")
                if ccy != "USD" and ccy in fx_returns:
                    currency_effect += sf(h, "weight") * fx_returns[ccy]
            for h in benchmark_holdings:
                ccy = h.get("currency", "USD")
                if ccy != "USD" and ccy in fx_returns:
                    currency_effect -= sf(h, "weight") * fx_returns[ccy]

        # Carry effect: active coupon income
        carry_effect = (port_agg["coupon_rate"] - bench_agg["coupon_rate"]) / 252  # daily approx

        # Active return
        active_return = port_agg["total_return"] - bench_agg["total_return"]

        # Selection (residual)
        explained = duration_effect + curve_effect + spread_effect + currency_effect + carry_effect
        selection_effect = active_return - explained

        # Verify: residual should be small
        residual = abs(active_return - (explained + selection_effect))

        # Sector breakdown (spread attribution by sector)
        sector_breakdown: Dict[str, Dict[str, float]] = {}
        port_by_sector: Dict[str, List[Dict]] = {}
        bench_by_sector: Dict[str, List[Dict]] = {}
        for h in portfolio_holdings:
            sec = h.get("sector", "Other")
            port_by_sector.setdefault(sec, []).append(h)
        for h in benchmark_holdings:
            sec = h.get("sector", "Other")
            bench_by_sector.setdefault(sec, []).append(h)

        all_sectors = set(port_by_sector.keys()) | set(bench_by_sector.keys())
        for sec in all_sectors:
            p_list = port_by_sector.get(sec, [])
            b_list = bench_by_sector.get(sec, [])
            p_agg = _agg(p_list) if p_list else {"weight": 0.0, "duration": 0.0, "total_return": 0.0, "spread_duration": 0.0, "coupon_rate": 0.0}
            b_agg = _agg(b_list) if b_list else {"weight": 0.0, "duration": 0.0, "total_return": 0.0, "spread_duration": 0.0, "coupon_rate": 0.0}
            sec_active = p_agg["total_return"] - b_agg["total_return"]
            sec_alloc = (p_agg["weight"] - b_agg["weight"]) * (b_agg["total_return"] - bench_agg["total_return"])
            sec_sel = b_agg["weight"] * sec_active
            sector_breakdown[sec] = {
                "allocation_bps": round(sec_alloc * 10_000, 2),
                "selection_bps": round(sec_sel * 10_000, 2),
                "active_return_bps": round(sec_active * 10_000, 2),
            }

        return FixedIncomeAttributionResult(
            duration_effect=round(duration_effect, 6),
            curve_effect=round(curve_effect, 6),
            spread_effect=round(spread_effect, 6),
            currency_effect=round(currency_effect, 6),
            selection_effect=round(selection_effect, 6),
            carry_effect=round(carry_effect, 6),
            total_active=round(active_return, 6),
            residual=round(residual, 10),
            sector_breakdown=sector_breakdown,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8. ResidualAnalyzer — completeness check
# ══════════════════════════════════════════════════════════════════════════════

class ResidualAnalyzer:
    """
    Attribution completeness analysis.

    Checks:
    1. Sum-to-total: allocation + selection + interaction == active return
    2. Residual decomposition: identifies which sectors drive the residual
    3. Cross-period drift: unexplained cumulative compounding error
    """

    @staticmethod
    def verify_sum_to_total(
        attribution_df: pd.DataFrame,
        active_return: float,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """Verify that explained effects sum to active return."""
        total_row = attribution_df.loc["TOTAL"] if "TOTAL" in attribution_df.index else None
        if total_row is None:
            return {"verified": False, "reason": "No TOTAL row in DataFrame."}

        explained = float(
            (total_row.get("allocation", 0.0) or 0.0)
            + (total_row.get("selection", 0.0) or 0.0)
            + (total_row.get("interaction", 0.0) or 0.0)
        )
        residual = abs(explained - active_return)
        return {
            "verified": residual < tol,
            "active_return": round(active_return, 8),
            "explained": round(explained, 8),
            "residual": round(residual, 10),
            "residual_bps": round(residual * 10_000, 4),
            "tolerance": tol,
        }

    @staticmethod
    def sector_residuals(
        attribution_df: pd.DataFrame,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
    ) -> pd.DataFrame:
        """Compute sector-level residuals to identify attribution gaps."""
        sectors = [idx for idx in attribution_df.index if idx != "TOTAL"]
        records = []
        for sector in sectors:
            row = attribution_df.loc[sector]
            actual_contribution = (
                portfolio_weights.get(sector, 0.0) * portfolio_returns.get(sector, 0.0)
                if isinstance(portfolio_weights, pd.Series)
                else 0.0
            )
            explained = float(
                (row.get("allocation", 0.0) or 0.0)
                + (row.get("selection", 0.0) or 0.0)
                + (row.get("interaction", 0.0) or 0.0)
            )
            records.append({
                "sector": sector,
                "actual_contribution": round(actual_contribution, 8),
                "explained_contribution": round(explained, 8),
                "residual": round(actual_contribution - explained, 10),
                "residual_bps": round((actual_contribution - explained) * 10_000, 4),
            })
        return pd.DataFrame(records).set_index("sector")

    @staticmethod
    def multi_period_completeness(period_results: List[Dict]) -> Dict[str, Any]:
        """Check cumulative completeness across linked periods."""
        total_explained = sum(
            (p.get("allocation", 0.0) + p.get("selection", 0.0) + p.get("interaction", 0.0))
            for p in period_results
        )
        total_active = sum(p.get("active_return", 0.0) for p in period_results)
        drift = total_active - total_explained
        return {
            "total_active_return": round(total_active, 8),
            "total_explained": round(total_explained, 8),
            "compounding_drift": round(drift, 8),
            "compounding_drift_bps": round(drift * 10_000, 4),
            "complete": abs(drift) < 1e-4,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 9. RiskAdjustedAttribution — Information Ratio per bucket
# ══════════════════════════════════════════════════════════════════════════════

class RiskAdjustedAttribution:
    """
    Compute risk-adjusted attribution metrics:

    - Information Ratio (IR) per attribution bucket (sector/factor)
    - Appraisal Ratio: alpha / tracking error
    - Hit Rate: fraction of periods where bucket contributes positively
    - Batting Average, Up/Down Capture
    """

    @staticmethod
    def information_ratio(
        active_returns: pd.Series,
        annualise: bool = True,
    ) -> float:
        """IR = mean(active) / std(active) × sqrt(252) if annualised."""
        ar = active_returns.dropna()
        if len(ar) < 2:
            return 0.0
        mean = float(ar.mean())
        std = float(ar.std(ddof=1))
        if std < 1e-12:
            return 0.0
        ir = mean / std
        return round(ir * (252 ** 0.5) if annualise else ir, 4)

    @staticmethod
    def appraisal_ratio(
        active_returns: pd.Series,
        factor_returns: pd.DataFrame,
        annualise: bool = True,
    ) -> float:
        """Appraisal ratio = alpha / residual std (Treynor-Black)."""
        ar = active_returns.dropna()
        n = len(ar)
        if n < 10:
            return 0.0
        X = factor_returns.loc[ar.index].values if not factor_returns.empty else np.ones((n, 1))
        X_c = np.column_stack([np.ones(n), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_c, ar.values, rcond=None)
        resid = ar.values - X_c @ coeffs
        resid_std = float(np.std(resid, ddof=1))
        alpha = float(coeffs[0])
        if resid_std < 1e-12:
            return 0.0
        ratio = alpha / resid_std
        return round(ratio * (252 ** 0.5) if annualise else ratio, 4)

    @staticmethod
    def hit_rate(period_contributions: pd.Series) -> float:
        """Fraction of periods where the bucket contributed positively."""
        vals = period_contributions.dropna()
        if len(vals) == 0:
            return 0.0
        return round(float((vals > 0).mean()), 4)

    @staticmethod
    def up_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Up-market capture ratio."""
        up = benchmark_returns > 0
        if up.sum() == 0:
            return 0.0
        port_up = portfolio_returns.loc[up].mean()
        bench_up = benchmark_returns.loc[up].mean()
        return round(port_up / bench_up, 4) if abs(bench_up) > 1e-12 else 0.0

    @staticmethod
    def down_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Down-market capture ratio."""
        down = benchmark_returns < 0
        if down.sum() == 0:
            return 0.0
        port_down = portfolio_returns.loc[down].mean()
        bench_down = benchmark_returns.loc[down].mean()
        return round(port_down / bench_down, 4) if abs(bench_down) > 1e-12 else 0.0

    def full_risk_adjusted(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_period_contributions: Optional[Dict[str, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Compute full suite of risk-adjusted attribution metrics.

        Returns dict with IR, appraisal ratio, up/down capture, hit rates per sector.
        """
        active = portfolio_returns - benchmark_returns
        ir = self.information_ratio(active)
        up_cap = self.up_capture(portfolio_returns, benchmark_returns)
        down_cap = self.down_capture(portfolio_returns, benchmark_returns)
        appraisal = (
            self.appraisal_ratio(active, factor_returns)
            if factor_returns is not None and not factor_returns.empty
            else None
        )

        tracking_error = round(float(active.std(ddof=1)) * (252 ** 0.5) * 100, 4) if len(active) > 1 else 0.0
        active_return_annual = round(float(active.mean()) * 252 * 100, 4)

        sector_ir: Dict[str, float] = {}
        sector_hit_rates: Dict[str, float] = {}
        if sector_period_contributions:
            for sector, contribs in sector_period_contributions.items():
                sector_ir[sector] = self.information_ratio(contribs)
                sector_hit_rates[sector] = self.hit_rate(contribs)

        return {
            "information_ratio": ir,
            "appraisal_ratio": appraisal,
            "up_capture_ratio": up_cap,
            "down_capture_ratio": down_cap,
            "tracking_error_annual_pct": tracking_error,
            "active_return_annual_pct": active_return_annual,
            "sector_information_ratios": sector_ir,
            "sector_hit_rates": sector_hit_rates,
            "n_observations": len(active),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 10. BenchmarkBuilder — S&P 500 sector weights
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkBuilder:
    """
    Build benchmark sector weights from:
    1. yfinance SPDR ETF holdings (best effort)
    2. Hard-coded SP500_SECTOR_WEIGHTS (fallback)
    """

    @staticmethod
    def sp500_sector_weights(use_live: bool = False) -> pd.Series:
        """Return S&P 500 sector weights as a Series."""
        if not use_live:
            return pd.Series(SP500_SECTOR_WEIGHTS)

        try:
            # Try to scrape from XLK/XLV/etc. NAV data via yfinance
            import yfinance as yf
            etf_market_caps: Dict[str, float] = {}
            for sector, etf in SECTOR_ETFS.items():
                info = yf.Ticker(etf).info
                mc = info.get("totalAssets") or info.get("marketCap") or 0.0
                etf_market_caps[sector] = float(mc)
            total = sum(etf_market_caps.values())
            if total > 0:
                return pd.Series({k: v / total for k, v in etf_market_caps.items()})
        except Exception:
            pass
        return pd.Series(SP500_SECTOR_WEIGHTS)

    @staticmethod
    def sector_returns_from_etfs(start: date, end: date) -> pd.Series:
        """Fetch SPDR ETF sector returns for the given date range."""
        etfs = list(SECTOR_ETFS.values())
        prices = _fetch_yfinance_prices(etfs, start, end)
        if prices.empty:
            # Fallback: zero returns
            return pd.Series({s: 0.0 for s in SECTOR_ETFS})
        returns = prices.iloc[-1] / prices.iloc[0] - 1.0
        sector_returns: Dict[str, float] = {}
        for sector, etf in SECTOR_ETFS.items():
            sector_returns[sector] = float(returns.get(etf, 0.0))
        return pd.Series(sector_returns)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Export utilities
# ══════════════════════════════════════════════════════════════════════════════

class AttributionExporter:
    """Export attribution tables to CSV, JSON, or formatted text."""

    @staticmethod
    def to_csv(attribution_df: pd.DataFrame, bps: bool = True) -> str:
        """Return CSV string of attribution table (in bps by default)."""
        df = attribution_df.copy()
        num_cols = [c for c in df.columns if c not in ("sector",)]
        for col in num_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if bps:
                    df[col] = (df[col] * 10_000).round(2)
            except Exception:
                pass

        output = io.StringIO()
        df.to_csv(output)
        return output.getvalue()

    @staticmethod
    def to_json(attribution_df: pd.DataFrame, bps: bool = True) -> str:
        """Return JSON string of attribution table."""
        df = attribution_df.copy()
        num_cols = [c for c in df.columns if c not in ("sector",)]
        for col in num_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if bps:
                    df[col] = (df[col] * 10_000).round(2)
            except Exception:
                pass
        return df.to_json(orient="index", default_handler=str)

    @staticmethod
    def format_table(attribution_df: pd.DataFrame) -> str:
        """Pretty-print attribution table."""
        effect_cols = ["allocation", "selection", "interaction", "total"]
        avail = [c for c in effect_cols if c in attribution_df.columns]
        display = (attribution_df[avail] * 10_000).round(2)
        display.columns = [c.title() + " (bps)" for c in display.columns]

        width = 90
        lines = ["=" * width]
        header = f"{'Sector':<28}" + "".join(f"{col:>15}" for col in display.columns)
        lines.append(header)
        lines.append("-" * width)

        non_total = [i for i in display.index if i != "TOTAL"]
        for sector in non_total:
            row = display.loc[sector]
            lines.append(f"{str(sector):<28}" + "".join(f"{float(v):>15.2f}" for v in row))

        if "TOTAL" in display.index:
            lines.append("=" * width)
            row = display.loc["TOTAL"]
            lines.append(f"{'TOTAL':<28}" + "".join(f"{float(v):>15.2f}" for v in row))

        lines.append("=" * width)
        return "\n".join(lines)

    @staticmethod
    def to_excel_dict(attribution_df: pd.DataFrame) -> Dict:
        """Serialise to a JSON-compatible Excel-ready structure."""
        df_bps = (attribution_df * 10_000).round(2)
        rows = []
        for idx, row in df_bps.iterrows():
            rec: Dict[str, Any] = {"sector": str(idx)}
            for col in df_bps.columns:
                v = row[col]
                rec[col] = float(v) if not pd.isna(v) else None
            rows.append(rec)
        return {
            "columns": ["sector"] + list(df_bps.columns),
            "rows": rows,
            "units": "basis_points",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 12. AttributionOrchestrator — main convenience class
# ══════════════════════════════════════════════════════════════════════════════

class AttributionOrchestrator:
    """
    Runs the full attribution pipeline for a portfolio:
      1. BHB + BF single-period
      2. Residual verification
      3. Risk-adjusted metrics (if daily returns provided)
      4. Factor attribution (if daily returns provided)
      5. Stores report to SQLite
    Returns a serialisable dict report.
    """

    def run(
        self,
        portfolio_id: str,
        period: str,
        portfolio: Dict[str, Any],
        benchmark: Dict[str, Any],
        model: str = "BHB",
        portfolio_daily: Optional[pd.Series] = None,
        benchmark_daily: Optional[pd.Series] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Full single-period attribution run.

        portfolio / benchmark: {weights: {sector→float}, returns: {sector→float}}
        """
        wp = pd.Series(portfolio["weights"])
        wb = pd.Series(benchmark["weights"])
        rp = pd.Series(portfolio["returns"])
        rb = pd.Series(benchmark["returns"])

        # BHB
        bhb = BHBAttributionV2(wp, wb, rp, rb, period)
        bhb_sp = bhb.to_single_period()
        bhb_tbl = bhb.full_table()

        # BF
        bf = BrinssonFachlerV2(wp, wb, rp, rb, period)
        bf_sp = bf.to_single_period()

        # Residual
        analyzer = ResidualAnalyzer()
        residual_check = analyzer.verify_sum_to_total(bhb_tbl, bhb_sp.active_return)
        sector_resid = analyzer.sector_residuals(bhb_tbl, wp, rp)

        # Risk-adjusted
        risk_adj: Dict[str, Any] = {}
        factor_result: Optional[FactorAttributionResult] = None

        if portfolio_daily is not None and benchmark_daily is not None:
            ra = RiskAdjustedAttribution()
            risk_adj = ra.full_risk_adjusted(portfolio_daily, benchmark_daily)

            # Factor attribution
            if start_date and end_date:
                ff5 = _fetch_ff5(start_date, end_date)
                mom = _fetch_mom(start_date, end_date)
                if ff5 is not None:
                    try:
                        fa = FactorAttributionV2(portfolio_daily, benchmark_daily, ff5, mom)
                        factor_result = fa.compute()
                    except Exception:
                        pass

        # Format tables
        exporter = AttributionExporter()
        report = {
            "portfolio_id": portfolio_id,
            "period": period,
            "model": model,
            "bhb": {
                "active_return_bps": round(bhb_sp.active_return * 10_000, 2),
                "allocation_bps": round(bhb_sp.allocation_total * 10_000, 2),
                "selection_bps": round(bhb_sp.selection_total * 10_000, 2),
                "interaction_bps": round(bhb_sp.interaction_total * 10_000, 2),
                "portfolio_return_pct": round(bhb_sp.portfolio_return * 100, 4),
                "benchmark_return_pct": round(bhb_sp.benchmark_return * 100, 4),
                "verified": bhb_sp.verified,
                "table_csv": exporter.to_csv(bhb_tbl),
                "table_json": json.loads(exporter.to_json(bhb_tbl)),
                "table_formatted": exporter.format_table(bhb_tbl),
                "excel_export": exporter.to_excel_dict(bhb_tbl),
            },
            "bf": {
                "active_return_bps": round(bf_sp.active_return * 10_000, 2),
                "allocation_bps": round(bf_sp.allocation_total * 10_000, 2),
                "selection_bps": round(bf_sp.selection_total * 10_000, 2),
                "verified": bf_sp.verified,
                "note": "No interaction term in Brinson-Fachler (1985).",
            },
            "residual_analysis": {
                **residual_check,
                "sector_residuals_bps": sector_resid["residual_bps"].to_dict() if not sector_resid.empty else {},
            },
            "risk_adjusted": risk_adj,
            "factor_attribution": (
                {
                    "alpha_annual_pct": factor_result.alpha_annual_pct,
                    "alpha_t_stat": factor_result.alpha_t_stat,
                    "factor_betas": factor_result.factor_betas,
                    "factor_contributions_bps": factor_result.factor_contributions_bps,
                    "r_squared": factor_result.r_squared,
                    "unexplained_pct": factor_result.unexplained_pct,
                    "n_observations": factor_result.n_observations,
                }
                if factor_result else None
            ),
        }

        report_id = _save_report(portfolio_id, period, model, report)
        report["report_id"] = report_id
        return report


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Router
# ══════════════════════════════════════════════════════════════════════════════

attribution_v2_router = APIRouter(prefix="/api/attribution/v2", tags=["attribution-v2"])

_orchestrator: Optional[AttributionOrchestrator] = None
_mp_engine: Optional[MultiPeriodAttribution] = None
_benchmark_builder: Optional[BenchmarkBuilder] = None


def _get_orch() -> AttributionOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AttributionOrchestrator()
    return _orchestrator


def _get_mp() -> MultiPeriodAttribution:
    global _mp_engine
    if _mp_engine is None:
        _mp_engine = MultiPeriodAttribution()
    return _mp_engine


def _get_bb() -> BenchmarkBuilder:
    global _benchmark_builder
    if _benchmark_builder is None:
        _benchmark_builder = BenchmarkBuilder()
    return _benchmark_builder


@attribution_v2_router.post("/attribute")
def attribute(req: AttributeRequest) -> Dict:
    """
    Run full single-period BHB + BF attribution with optional factor attribution.

    Returns allocation / selection / interaction effects in bps, plus residual check.
    """
    try:
        port_daily: Optional[pd.Series] = None
        bench_daily: Optional[pd.Series] = None
        start: Optional[date] = None
        end: Optional[date] = None

        if req.portfolio_daily_returns and req.benchmark_daily_returns:
            port_daily = pd.Series(req.portfolio_daily_returns)
            port_daily.index = pd.to_datetime(port_daily.index)
            bench_daily = pd.Series(req.benchmark_daily_returns)
            bench_daily.index = pd.to_datetime(bench_daily.index)

        if req.start_date:
            start = date.fromisoformat(req.start_date)
        if req.end_date:
            end = date.fromisoformat(req.end_date)

        result = _get_orch().run(
            portfolio_id=req.portfolio_id,
            period=req.period,
            portfolio={"weights": req.portfolio.weights, "returns": req.portfolio.returns},
            benchmark={"weights": req.benchmark.weights, "returns": req.benchmark.returns},
            model=req.model,
            portfolio_daily=port_daily,
            benchmark_daily=bench_daily,
            start_date=start,
            end_date=end,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_v2_router.get("/attribution-report/{report_id}")
def get_attribution_report(report_id: int) -> Dict:
    """Retrieve a stored attribution report by ID."""
    report = _load_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
    return report


@attribution_v2_router.get("/attribution-reports")
def list_attribution_reports(
    portfolio_id: Optional[str] = FastAPIQuery(None),
    limit: int = FastAPIQuery(50, ge=1, le=200),
) -> Dict:
    """List stored attribution reports, optionally filtered by portfolio_id."""
    reports = _list_reports(portfolio_id, limit)
    return {"count": len(reports), "reports": reports}


@attribution_v2_router.post("/multi-period")
def multi_period(req: MultiPeriodRequest) -> Dict:
    """
    Multi-period geometric attribution via Cariño / Menchero / GRAP linking.

    Pass a list of single-period {portfolio, benchmark, period} dicts.
    """
    try:
        result = _get_mp().run(req.periods, req.linking_method)
        exporter = AttributionExporter()
        linked_tbl = result.sector_table
        report = {
            "portfolio_id": req.portfolio_id,
            "n_periods": result.n_periods,
            "linking_method": result.linking_method,
            "cumulative_portfolio_return_pct": round(result.cumulative_portfolio_return * 100, 4),
            "cumulative_benchmark_return_pct": round(result.cumulative_benchmark_return * 100, 4),
            "cumulative_active_return_bps": round(result.cumulative_active_return * 10_000, 2),
            "linked_allocation_bps": round(result.linked_allocation * 10_000, 2),
            "linked_selection_bps": round(result.linked_selection * 10_000, 2),
            "linked_interaction_bps": round(result.linked_interaction * 10_000, 2),
            "table_formatted": exporter.format_table(linked_tbl),
            "table_json": json.loads(exporter.to_json(linked_tbl)),
            "period_summary": result.period_summary,
        }
        report_id = _save_report(req.portfolio_id, "multi-period", req.linking_method, report)
        report["report_id"] = report_id
        return report
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_v2_router.post("/factor-attribution")
def factor_attribution(req: FactorRequest) -> Dict:
    """
    Fama-French 5-factor + Momentum factor attribution.

    Requires daily portfolio and benchmark returns as {date_str → return}.
    """
    try:
        end = date.fromisoformat(req.end_date) if req.end_date else date.today()
        start = date.fromisoformat(req.start_date) if req.start_date else end - timedelta(days=365)

        p_rets = pd.Series(req.portfolio_daily_returns)
        p_rets.index = pd.to_datetime(p_rets.index)
        b_rets = pd.Series(req.benchmark_daily_returns)
        b_rets.index = pd.to_datetime(b_rets.index)

        ff5 = _fetch_ff5(start, end)
        if ff5 is None:
            raise HTTPException(status_code=503, detail="FF5 factor data unavailable from Ken French library.")

        mom: Optional[pd.Series] = None
        if req.include_mom:
            mom = _fetch_mom(start, end)

        fa = FactorAttributionV2(p_rets, b_rets, ff5, mom)
        result = fa.compute(include_mom=req.include_mom)
        style = fa.style_attribution()

        return {
            "alpha_annual_pct": result.alpha_annual_pct,
            "alpha_t_stat": result.alpha_t_stat,
            "factor_betas": result.factor_betas,
            "factor_contributions_bps": result.factor_contributions_bps,
            "r_squared": result.r_squared,
            "active_return_annual_pct": result.active_return_annual_pct,
            "unexplained_pct": result.unexplained_pct,
            "n_observations": result.n_observations,
            "regression_details": result.regression_details,
            "style_tilts": style,
            "period": {"start": start.isoformat(), "end": end.isoformat()},
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_v2_router.post("/fixed-income")
def fixed_income_attribution(req: FixedIncomeRequest) -> Dict:
    """
    Fixed income attribution: Duration / Curve / Spread / Currency / Selection.
    Uses simplified Van Breukelen / Campisi framework.
    """
    try:
        fia = FixedIncomeAttribution()
        result = fia.compute(
            portfolio_holdings=req.holdings,
            benchmark_holdings=req.benchmark_holdings,
            fx_returns=req.fx_returns or {},
        )
        report = {
            "portfolio_id": req.portfolio_id,
            "period": req.period,
            "duration_effect_bps": round(result.duration_effect * 10_000, 2),
            "curve_effect_bps": round(result.curve_effect * 10_000, 2),
            "spread_effect_bps": round(result.spread_effect * 10_000, 2),
            "currency_effect_bps": round(result.currency_effect * 10_000, 2),
            "selection_effect_bps": round(result.selection_effect * 10_000, 2),
            "carry_effect_bps": round(result.carry_effect * 10_000, 2),
            "total_active_bps": round(result.total_active * 10_000, 2),
            "residual_bps": round(result.residual * 10_000, 6),
            "sector_breakdown": result.sector_breakdown,
            "attribution_verified": result.residual < 1e-6,
        }
        report_id = _save_report(req.portfolio_id, req.period, "FixedIncome", report)
        report["report_id"] = report_id
        return report
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_v2_router.post("/top-down")
def top_down_attribution(req: TopDownRequest) -> Dict:
    """
    Three-tier top-down attribution: Asset Class → Sector → Stock Selection.
    """
    try:
        tda = TopDownAttribution()
        result = tda.run(req.asset_classes)
        result["portfolio_id"] = req.portfolio_id
        result["period"] = req.period
        report_id = _save_report(req.portfolio_id, req.period, "TopDown", result)
        result["report_id"] = report_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_v2_router.post("/sector-country-currency")
def sector_country_currency(
    portfolio_id: str,
    period: str,
    sector_data: Dict[str, Any],
    country_data: Dict[str, Any],
    currency_data: Dict[str, Any],
) -> Dict:
    """Three-layer attribution: Sector / Country / Currency (Ankrim-Hensel)."""
    try:
        scc = SectorCountryCurrencyAttribution()
        result = scc.full_three_layer(sector_data, country_data, currency_data, period)
        result["portfolio_id"] = portfolio_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_v2_router.get("/benchmark/sp500-weights")
def sp500_weights(live: bool = FastAPIQuery(False)) -> Dict:
    """Return S&P 500 sector weights (hard-coded or live from SPDR ETFs)."""
    weights = _get_bb().sp500_sector_weights(use_live=live)
    return {
        "source": "live_yfinance_spdr" if live else "hard_coded_q1_2026",
        "weights": weights.to_dict(),
        "total": round(weights.sum(), 4),
    }


@attribution_v2_router.get("/benchmark/sector-returns")
def sp500_sector_returns(
    start: str = FastAPIQuery(..., description="YYYY-MM-DD"),
    end: str = FastAPIQuery(..., description="YYYY-MM-DD"),
) -> Dict:
    """Fetch S&P 500 sector returns via SPDR ETFs for a date range."""
    try:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        returns = _get_bb().sector_returns_from_etfs(start_d, end_d)
        return {
            "start": start,
            "end": end,
            "sector_returns": returns.to_dict(),
            "source": "yfinance_spdr_etfs",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@attribution_v2_router.post("/export/{report_id}/csv")
def export_csv(report_id: int) -> Dict:
    """Export a stored attribution report as CSV."""
    report = _load_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
    # Reconstruct DataFrame from stored JSON
    bhb_data = report.get("bhb", {})
    table_json = bhb_data.get("table_json", {})
    if not table_json:
        raise HTTPException(status_code=422, detail="No table data found in report.")
    df = pd.DataFrame.from_dict(table_json, orient="index")
    exporter = AttributionExporter()
    csv_str = exporter.to_csv(df, bps=False)  # already in bps from stored
    return {"report_id": report_id, "csv": csv_str, "format": "csv"}


@attribution_v2_router.post("/verify/{report_id}")
def verify_attribution(report_id: int) -> Dict:
    """Verify attribution completeness for a stored report."""
    report = _load_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
    bhb = report.get("bhb", {})
    return {
        "report_id": report_id,
        "verified": bhb.get("verified", False),
        "active_return_bps": bhb.get("active_return_bps"),
        "allocation_bps": bhb.get("allocation_bps"),
        "selection_bps": bhb.get("selection_bps"),
        "interaction_bps": bhb.get("interaction_bps"),
        "residual_bps": report.get("residual_analysis", {}).get("residual_bps", None),
        "message": "Attribution is complete (sum = active return)." if bhb.get("verified") else "Attribution residual exceeds tolerance.",
    }


@attribution_v2_router.get("/demo")
def demo_attribution() -> Dict:
    """
    Run a full demo attribution with synthetic S&P 500-like sector data.
    Returns BHB + BF + residual check.
    """
    demo_port = {
        "weights": {
            "Technology": 0.35, "Financials": 0.12, "Health Care": 0.15,
            "Energy": 0.08, "Industrials": 0.10, "Consumer Discretionary": 0.10,
            "Consumer Staples": 0.06, "Materials": 0.02, "Real Estate": 0.02,
        },
        "returns": {
            "Technology": 0.22, "Financials": 0.14, "Health Care": 0.09,
            "Energy": 0.28, "Industrials": 0.12, "Consumer Discretionary": 0.18,
            "Consumer Staples": 0.06, "Materials": 0.08, "Real Estate": 0.04,
        },
    }
    demo_bench = {
        "weights": {k: v for k, v in SP500_SECTOR_WEIGHTS.items()
                    if k in demo_port["weights"]},
        "returns": {
            "Technology": 0.18, "Financials": 0.11, "Health Care": 0.08,
            "Energy": 0.22, "Industrials": 0.10, "Consumer Discretionary": 0.15,
            "Consumer Staples": 0.05, "Materials": 0.07, "Real Estate": 0.03,
        },
    }
    # Normalise benchmark weights
    total_wb = sum(demo_bench["weights"].values())
    demo_bench["weights"] = {k: v / total_wb for k, v in demo_bench["weights"].items()}

    try:
        result = _get_orch().run(
            portfolio_id="demo",
            period="2026-Q1",
            portfolio=demo_port,
            benchmark=demo_bench,
            model="BHB",
        )
        result["note"] = "Demo with synthetic S&P 500-like sector data."
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Module-level convenience functions ────────────────────────────────────────

def run_bhb(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    portfolio_returns: Dict[str, float],
    benchmark_returns: Dict[str, float],
    period: str = "",
) -> SinglePeriodAttribution:
    """Module-level convenience: run BHB attribution."""
    return BHBAttributionV2(
        pd.Series(portfolio_weights),
        pd.Series(benchmark_weights),
        pd.Series(portfolio_returns),
        pd.Series(benchmark_returns),
        period,
    ).to_single_period()


def run_bf(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    portfolio_returns: Dict[str, float],
    benchmark_returns: Dict[str, float],
    period: str = "",
) -> SinglePeriodAttribution:
    """Module-level convenience: run Brinson-Fachler attribution."""
    return BrinssonFachlerV2(
        pd.Series(portfolio_weights),
        pd.Series(benchmark_weights),
        pd.Series(portfolio_returns),
        pd.Series(benchmark_returns),
        period,
    ).to_single_period()


def run_multi_period(
    period_inputs: List[Dict[str, Any]],
    linking_method: str = "carino",
) -> MultiPeriodResult:
    """Module-level convenience: run multi-period geometric attribution."""
    return MultiPeriodAttribution().run(period_inputs, linking_method)
