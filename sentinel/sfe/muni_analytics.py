from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional


class MuniBondAnalytics(BaseModel):
    cusip: str
    coupon: float
    ytm: float
    maturity_years: float
    price: float
    duration_modified: float
    convexity: float
    dv01: float
    taxable_equiv_yield: float
    tax_equivalent_spread: float


def duration_modified(coupon: float, ytm: float, maturity_years: float, freq: int = 2) -> float:
    n = int(round(maturity_years * freq))
    if n <= 0:
        return 0.0
    c = coupon / freq / 100.0
    y = ytm / freq / 100.0

    if coupon == 0.0:
        mac = maturity_years
        return mac / (1.0 + ytm / freq / 100.0)

    if abs(y) < 1e-12:
        weights_num = sum(t * c for t in range(1, n + 1)) + n * 1.0
        weights_den = n * c + 1.0
        mac = weights_num / weights_den / freq
        return mac / (1.0 + y)

    pv_coupons = 0.0
    mac_num = 0.0
    for t in range(1, n + 1):
        pv_t = c / (1.0 + y) ** t
        pv_coupons += pv_t
        mac_num += t * pv_t

    pv_face = 1.0 / (1.0 + y) ** n
    mac_num += n * pv_face
    total_pv = pv_coupons + pv_face

    mac = (mac_num / total_pv) / freq
    return mac / (1.0 + y)


def convexity(coupon: float, ytm: float, maturity_years: float, freq: int = 2) -> float:
    n = int(round(maturity_years * freq))
    if n <= 0:
        return 0.0
    c = coupon / freq / 100.0
    y = ytm / freq / 100.0

    if abs(y) < 1e-12:
        conv_num = sum(t * (t + 1) * c for t in range(1, n + 1)) + n * (n + 1) * 1.0
        conv_den = (n * c + 1.0) * (1.0 + y) ** 2
        return conv_num / conv_den / freq ** 2

    total_pv = 0.0
    conv_sum = 0.0
    for t in range(1, n + 1):
        pv_t = c / (1.0 + y) ** t
        total_pv += pv_t
        conv_sum += t * (t + 1) * pv_t

    pv_face = 1.0 / (1.0 + y) ** n
    total_pv += pv_face
    conv_sum += n * (n + 1) * pv_face

    return conv_sum / (total_pv * (1.0 + y) ** 2) / freq ** 2


def price_from_ytm(coupon: float, ytm: float, maturity_years: float, freq: int = 2) -> float:
    n = int(round(maturity_years * freq))
    if n <= 0:
        return 100.0
    c = coupon / freq / 100.0
    y = ytm / freq / 100.0

    if abs(y) < 1e-12:
        return (c * n + 1.0) * 100.0

    pv_coupons = c * (1.0 - (1.0 + y) ** (-n)) / y
    pv_face = 1.0 / (1.0 + y) ** n
    return (pv_coupons + pv_face) * 100.0


def ytm_from_price(coupon: float, price: float, maturity_years: float, freq: int = 2) -> float:
    n = int(round(maturity_years * freq))
    if n <= 0:
        return coupon
    c = coupon / freq / 100.0
    p = price / 100.0

    y = (c + (1.0 - p) / n) / ((1.0 + p) / 2.0)
    if y <= 0:
        y = c

    for _ in range(200):
        if abs(1.0 + y) < 1e-12:
            break
        discount = (1.0 + y) ** n
        pv = c * (1.0 - 1.0 / discount) / y + 1.0 / discount
        dpv = (
            -c * (1.0 - 1.0 / discount) / (y ** 2)
            + c * n / (y * discount * (1.0 + y))
            - n / (discount * (1.0 + y))
        )
        if abs(dpv) < 1e-15:
            break
        delta = (pv - p) / dpv
        y -= delta
        if abs(delta) < 1e-10:
            break

    return y * freq * 100.0


def taxable_equivalent_yield(muni_yield: float, marginal_tax_rate: float = 0.37) -> float:
    return muni_yield / (1.0 - marginal_tax_rate)


def relative_value_analysis(bonds: list[dict]) -> pd.DataFrame:
    if not bonds:
        return pd.DataFrame()

    rows = []
    for b in bonds:
        coupon = float(b.get("coupon", 0.0))
        ytm = float(b.get("ytm", 0.0))
        mat = float(b.get("maturity_years", 0.0))
        p = price_from_ytm(coupon, ytm, mat)
        dur = duration_modified(coupon, ytm, mat)
        conv = convexity(coupon, ytm, mat)
        dv01_val = dur * p * 0.0001 * 10_000
        tey = taxable_equivalent_yield(ytm)
        rows.append({
            "cusip": b.get("cusip", ""),
            "state": b.get("state", ""),
            "coupon": coupon,
            "ytm": ytm,
            "maturity_years": mat,
            "price": round(p, 4),
            "duration_modified": round(dur, 4),
            "convexity": round(conv, 4),
            "dv01": round(dv01_val, 2),
            "taxable_equiv_yield": round(tey, 4),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["dur_bucket"] = (df["duration_modified"] / 1.0).round(0)

    def _zscore(group: pd.DataFrame) -> pd.Series:
        if len(group) < 2:
            return pd.Series(0.0, index=group.index)
        mean = group["ytm"].mean()
        std = group["ytm"].std(ddof=1)
        if std < 1e-10:
            return pd.Series(0.0, index=group.index)
        return (group["ytm"] - mean) / std

    df["ytm_zscore"] = df.groupby("dur_bucket", group_keys=False).apply(_zscore)
    df["relative_value"] = df["ytm_zscore"].apply(
        lambda z: "cheap" if z > 1.5 else ("expensive" if z < -1.5 else "fair")
    )
    df = df.drop(columns=["dur_bucket"])
    return df.reset_index(drop=True)


class MuniAnalyticsEngine:
    def __init__(self, treasury_yields: dict[float, float]) -> None:
        self._treasury_yields = treasury_yields

    def _nearest_treasury_yield(self, maturity_years: float) -> float:
        if not self._treasury_yields:
            return 0.0
        tenors = sorted(self._treasury_yields.keys())
        if maturity_years <= tenors[0]:
            return self._treasury_yields[tenors[0]]
        if maturity_years >= tenors[-1]:
            return self._treasury_yields[tenors[-1]]
        for i in range(len(tenors) - 1):
            lo, hi = tenors[i], tenors[i + 1]
            if lo <= maturity_years <= hi:
                frac = (maturity_years - lo) / (hi - lo)
                return self._treasury_yields[lo] + frac * (self._treasury_yields[hi] - self._treasury_yields[lo])
        return self._treasury_yields[tenors[-1]]

    def analyze_bond(
        self,
        cusip: str,
        coupon: float,
        ytm: float,
        maturity_years: float,
        price: float,
    ) -> MuniBondAnalytics:
        dur = duration_modified(coupon, ytm, maturity_years)
        conv = convexity(coupon, ytm, maturity_years)
        dv01_val = dur * price * 0.0001 * 10_000
        tey = taxable_equivalent_yield(ytm)
        tsy = self._nearest_treasury_yield(maturity_years)
        tes = tey - tsy

        return MuniBondAnalytics(
            cusip=cusip,
            coupon=coupon,
            ytm=ytm,
            maturity_years=maturity_years,
            price=round(price, 4),
            duration_modified=round(dur, 4),
            convexity=round(conv, 4),
            dv01=round(dv01_val, 2),
            taxable_equiv_yield=round(tey, 4),
            tax_equivalent_spread=round(tes, 4),
        )

    def screen_for_value(
        self, bonds: list[dict], min_spread_bps: float = 50
    ) -> pd.DataFrame:
        enriched = []
        for b in bonds:
            coupon = float(b.get("coupon", 0.0))
            ytm = float(b.get("ytm", 0.0))
            mat = float(b.get("maturity_years", 0.0))
            price = b.get("price") or price_from_ytm(coupon, ytm, mat)
            analytics = self.analyze_bond(
                cusip=b.get("cusip", ""),
                coupon=coupon,
                ytm=ytm,
                maturity_years=mat,
                price=float(price),
            )
            row = analytics.model_dump()
            row["state"] = b.get("state", "")
            row["spread_to_treasury_bps"] = round(
                (ytm - self._nearest_treasury_yield(mat)) * 100, 2
            )
            enriched.append(row)

        df = pd.DataFrame(enriched)
        if df.empty:
            return df

        min_spread_pct = min_spread_bps / 100.0
        df = df[df["spread_to_treasury_bps"] >= min_spread_bps].copy()
        return df.sort_values("tax_equivalent_spread", ascending=False).reset_index(drop=True)

    def compute_mmd_curve(self, bond_trades: list[dict]) -> dict[float, float]:
        if not bond_trades:
            return {}

        rows = []
        for t in bond_trades:
            mat = float(t.get("maturity_years", 0.0))
            y = t.get("yield_pct") or t.get("ytm") or t.get("yield")
            if mat <= 0 or y is None:
                continue
            try:
                rows.append({"maturity_years": mat, "yield_pct": float(y)})
            except (TypeError, ValueError):
                continue

        if not rows:
            return {}

        df = pd.DataFrame(rows)
        bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        labels = [0.5, 1.5, 2.5, 4.0, 6.0, 8.5, 12.5, 17.5, 25.0]
        df["bin"] = pd.cut(
            df["maturity_years"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )
        grouped = df.groupby("bin", observed=True)["yield_pct"].median()
        mmd: dict[float, float] = {}
        for tenor, y_val in grouped.items():
            if pd.notna(y_val):
                mmd[float(tenor)] = round(float(y_val), 4)

        return dict(sorted(mmd.items()))
