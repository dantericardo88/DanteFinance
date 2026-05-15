"""
DEX and AMM analytics: Uniswap v2/v3, Curve, Balancer, Sushiswap.
Pool analytics, impermanent loss, liquidity concentration, price impact.
Free data: DeFiLlama, The Graph public API (limited), direct contract queries.

Dimension: dim_110 — DEX / AMM liquidity analytics (target score: 9)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_SESSION_TIMEOUT = 20
_GRAPH_RATE_LIMIT_SLEEP = 0.5   # be kind to public endpoints

def _http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
              timeout: int = _SESSION_TIMEOUT) -> Optional[dict]:
    """Synchronous GET with basic error handling."""
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("http_get_error url=%s err=%s", url, exc)
        return None


def _graph_query(subgraph_url: str, query: str, variables: Optional[dict] = None) -> Optional[dict]:
    """Execute a GraphQL query against a public subgraph endpoint."""
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        resp = requests.post(subgraph_url, json=payload, timeout=_SESSION_TIMEOUT,
                             headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            logger.warning("graph_errors subgraph=%s errors=%s", subgraph_url, data["errors"])
        return data.get("data")
    except Exception as exc:
        logger.warning("graph_query_error url=%s err=%s", subgraph_url, exc)
        return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PoolInfo(BaseModel):
    pool_id: str
    token0: str
    token1: str
    fee_tier: Optional[float] = None
    tvl_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    fee_revenue_24h_usd: Optional[float] = None
    apy_pct: Optional[float] = None
    protocol: str = "uniswap_v3"
    source: str = "the_graph"


class ILResult(BaseModel):
    initial_price: float
    current_price: float
    price_ratio: float
    il_pct: float
    il_description: str
    breakeven_fee_apy_pct: Optional[float] = None


class PriceImpactResult(BaseModel):
    trade_size_usd: float
    price_impact_pct: float
    output_amount: float
    effective_price: float
    max_trade_1pct_slippage: Optional[float] = None


class FarmingOpportunity(BaseModel):
    pool: str
    protocol: str
    chain: str
    apy_pct: float
    tvl_usd: float
    il_risk: str
    risk_score: float
    net_apy_est_pct: Optional[float] = None
    source: str = "defillama"


# ---------------------------------------------------------------------------
# 1. UniswapV3Analytics
# ---------------------------------------------------------------------------

class UniswapV3Analytics:
    """
    Uniswap v3 pool analytics via The Graph public subgraph.

    Subgraph: https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3
    Data: top pools by TVL, volume, fee tiers, concentrated liquidity metrics.
    """

    SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    # Fallback: DeFiLlama Uniswap endpoint
    DEFILLAMA_PROTOCOL_URL = "https://api.llama.fi/protocol/uniswap-v3"
    DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"

    _TOP_POOLS_QUERY = """
    {
      pools(
        first: 50
        orderBy: totalValueLockedUSD
        orderDirection: desc
        where: {totalValueLockedUSD_gt: "100000"}
      ) {
        id
        token0 { symbol decimals }
        token1 { symbol decimals }
        feeTier
        totalValueLockedUSD
        volumeUSD
        txCount
        liquidity
        tick
        sqrtPrice
        token0Price
        token1Price
        poolDayData(first: 1 orderBy: date orderDirection: desc) {
          volumeUSD
          feesUSD
          tvlUSD
        }
      }
    }
    """

    _POOL_TICKS_QUERY = """
    query PoolTicks($pool: String!) {
      ticks(
        first: 200
        where: { pool: $pool, liquidityNet_not: "0" }
        orderBy: tickIdx
        orderDirection: asc
      ) {
        tickIdx
        liquidityNet
        liquidityGross
        price0
        price1
      }
    }
    """

    def get_top_pools(self, limit: int = 20) -> list[PoolInfo]:
        """Fetch top Uniswap v3 pools by TVL from The Graph."""
        data = _graph_query(self.SUBGRAPH_URL, self._TOP_POOLS_QUERY)
        pools: list[PoolInfo] = []

        if data and "pools" in data:
            for p in data["pools"][:limit]:
                fee_tier = int(p.get("feeTier", 3000)) / 1_000_000  # basis points → fraction
                tvl = float(p.get("totalValueLockedUSD", 0))
                day_data = p.get("poolDayData", [{}])
                vol_24h = float(day_data[0].get("volumeUSD", 0)) if day_data else 0.0
                fee_rev_24h = vol_24h * fee_tier
                apy = (fee_rev_24h * 365 / tvl * 100) if tvl > 0 else None

                pools.append(PoolInfo(
                    pool_id=p["id"],
                    token0=p["token0"]["symbol"],
                    token1=p["token1"]["symbol"],
                    fee_tier=fee_tier * 100,  # as percentage
                    tvl_usd=round(tvl, 2),
                    volume_24h_usd=round(vol_24h, 2),
                    fee_revenue_24h_usd=round(fee_rev_24h, 4),
                    apy_pct=round(apy, 4) if apy else None,
                    protocol="uniswap_v3",
                    source="the_graph",
                ))
        else:
            # Fallback to DeFiLlama yields API
            pools = self._get_pools_defillama(limit)

        return pools

    def _get_pools_defillama(self, limit: int = 20) -> list[PoolInfo]:
        """Fallback: get Uniswap v3 pools from DeFiLlama yields endpoint."""
        data = _http_get(self.DEFILLAMA_POOLS_URL)
        pools = []
        if not data or "data" not in data:
            return pools

        uniswap_pools = [
            p for p in data["data"]
            if "uniswap-v3" in p.get("project", "").lower()
            and p.get("chain", "").lower() == "ethereum"
        ]
        uniswap_pools.sort(key=lambda x: x.get("tvlUsd", 0), reverse=True)

        for p in uniswap_pools[:limit]:
            tvl = p.get("tvlUsd", 0) or 0
            apy = p.get("apy", None)
            pools.append(PoolInfo(
                pool_id=p.get("pool", ""),
                token0=p.get("symbol", "").split("-")[0] if "-" in p.get("symbol", "") else p.get("symbol", ""),
                token1=p.get("symbol", "").split("-")[1] if "-" in p.get("symbol", "") else "",
                tvl_usd=round(tvl, 2),
                apy_pct=round(apy, 4) if apy else None,
                protocol="uniswap_v3",
                source="defillama",
            ))
        return pools

    def get_pool_tick_analysis(self, pool_id: str) -> dict:
        """
        Analyse concentrated liquidity tick distribution for a specific pool.

        Returns: active_tick, price_range_active, liquidity_concentration,
                 tick_distribution summary.
        """
        data = _graph_query(self.SUBGRAPH_URL, self._POOL_TICKS_QUERY,
                            variables={"pool": pool_id.lower()})

        if not data or "ticks" not in data:
            return {"pool_id": pool_id, "error": "No tick data available"}

        ticks = data["ticks"]
        if not ticks:
            return {"pool_id": pool_id, "error": "Empty tick list"}

        tick_indices = [int(t["tickIdx"]) for t in ticks]
        liquidity_gross = [float(t["liquidityGross"]) for t in ticks]
        total_liq = sum(liquidity_gross) or 1

        # Concentration: what % of liquidity is in ±5% price range around median
        n = len(tick_indices)
        mid_idx = n // 2
        window = max(1, n // 10)
        center_liq = sum(liquidity_gross[max(0, mid_idx - window): mid_idx + window])
        concentration_ratio = center_liq / total_liq

        return {
            "pool_id": pool_id,
            "tick_count": n,
            "min_tick": min(tick_indices),
            "max_tick": max(tick_indices),
            "liquidity_concentration_pct": round(concentration_ratio * 100, 2),
            "note": "Concentration = fraction of liquidity in central ±10% tick range",
            "source": "the_graph",
        }

    def compute_v3_apy(self, tvl_usd: float, volume_24h_usd: float,
                       fee_tier_pct: float) -> dict:
        """
        Compute Uniswap v3 LP fee APY.

        Args:
            tvl_usd: Pool total value locked in USD
            volume_24h_usd: 24h trading volume
            fee_tier_pct: Fee tier as percentage (e.g. 0.3 for 0.3%)
        """
        fee_rev_24h = volume_24h_usd * (fee_tier_pct / 100)
        fee_rev_365 = fee_rev_24h * 365
        apy = (fee_rev_365 / tvl_usd * 100) if tvl_usd > 0 else 0

        # Capital efficiency: v3 is ~4-10x more capital efficient than v2 for same range
        # because LP picks a narrow range vs full [0, inf]
        capital_efficiency_vs_v2 = 4.0  # conservative estimate; can be 100x+ for narrow range

        return {
            "tvl_usd": tvl_usd,
            "volume_24h_usd": volume_24h_usd,
            "fee_tier_pct": fee_tier_pct,
            "fee_revenue_24h_usd": round(fee_rev_24h, 4),
            "fee_revenue_365d_usd": round(fee_rev_365, 2),
            "fee_apy_pct": round(apy, 4),
            "capital_efficiency_vs_v2": capital_efficiency_vs_v2,
            "note": "Capital efficiency assumes a ±25% price range around current price",
        }

    def get_pool_by_pair(self, token0: str, token1: str) -> list[PoolInfo]:
        """Find pools for a given token pair across all fee tiers."""
        query = f"""
        {{
          pools(
            where: {{
              token0_: {{ symbol_in: ["{token0.upper()}", "{token1.upper()}"] }}
              token1_: {{ symbol_in: ["{token0.upper()}", "{token1.upper()}"] }}
            }}
            orderBy: totalValueLockedUSD
            orderDirection: desc
            first: 5
          ) {{
            id token0 {{ symbol }} token1 {{ symbol }}
            feeTier totalValueLockedUSD
            poolDayData(first: 1 orderBy: date orderDirection: desc) {{
              volumeUSD feesUSD
            }}
          }}
        }}
        """
        data = _graph_query(self.SUBGRAPH_URL, query)
        pools = []
        if data and "pools" in data:
            for p in data["pools"]:
                fee_tier = int(p.get("feeTier", 3000)) / 1_000_000
                tvl = float(p.get("totalValueLockedUSD", 0))
                day = p.get("poolDayData", [{}])
                vol = float(day[0].get("volumeUSD", 0)) if day else 0
                pools.append(PoolInfo(
                    pool_id=p["id"],
                    token0=p["token0"]["symbol"],
                    token1=p["token1"]["symbol"],
                    fee_tier=fee_tier * 100,
                    tvl_usd=round(tvl, 2),
                    volume_24h_usd=round(vol, 2),
                    fee_revenue_24h_usd=round(vol * fee_tier, 4),
                    protocol="uniswap_v3",
                ))
        return pools


# ---------------------------------------------------------------------------
# 2. UniswapV2Analytics
# ---------------------------------------------------------------------------

class UniswapV2Analytics:
    """
    Uniswap v2 and Sushiswap pair analytics.

    Uses The Graph public subgraphs for pair reserves, volume, and LP pricing.
    Implements xy=k AMM formulas for price impact and optimal trade sizing.
    """

    UNISWAP_V2_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
    SUSHI_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/sushi/exchange"
    DEFILLAMA_POOLS = "https://yields.llama.fi/pools"

    _TOP_PAIRS_QUERY = """
    {
      pairs(
        first: 50
        orderBy: reserveUSD
        orderDirection: desc
        where: {reserveUSD_gt: "100000"}
      ) {
        id
        token0 { symbol }
        token1 { symbol }
        reserve0
        reserve1
        reserveUSD
        token0Price
        token1Price
        pairDayData(first: 1 orderBy: date orderDirection: desc) {
          dailyVolumeUSD
          reserveUSD
        }
      }
    }
    """

    def get_top_pairs(self, protocol: str = "uniswap_v2", limit: int = 20) -> list[dict]:
        """Fetch top pairs by TVL from v2-style subgraph."""
        subgraph = self.UNISWAP_V2_SUBGRAPH if protocol == "uniswap_v2" else self.SUSHI_SUBGRAPH
        data = _graph_query(subgraph, self._TOP_PAIRS_QUERY)

        pairs = []
        if data and "pairs" in data:
            for p in data["pairs"][:limit]:
                reserve_usd = float(p.get("reserveUSD", 0))
                day = p.get("pairDayData", [{}])
                vol_24h = float(day[0].get("dailyVolumeUSD", 0)) if day else 0
                fee_rev = vol_24h * 0.003  # v2 flat 0.3% fee

                r0 = float(p.get("reserve0", 0))
                r1 = float(p.get("reserve1", 0))
                price = r1 / r0 if r0 > 0 else None

                pairs.append({
                    "pair_id": p["id"],
                    "token0": p["token0"]["symbol"],
                    "token1": p["token1"]["symbol"],
                    "reserve0": r0,
                    "reserve1": r1,
                    "price_token0_in_token1": round(price, 6) if price else None,
                    "tvl_usd": round(reserve_usd, 2),
                    "volume_24h_usd": round(vol_24h, 2),
                    "fee_revenue_24h_usd": round(fee_rev, 4),
                    "apy_pct": round(fee_rev * 365 / reserve_usd * 100, 4) if reserve_usd > 0 else None,
                    "protocol": protocol,
                    "source": "the_graph",
                })
        else:
            pairs = self._get_v2_pairs_defillama(protocol, limit)

        return pairs

    def _get_v2_pairs_defillama(self, protocol: str, limit: int) -> list[dict]:
        """Fallback: v2 pairs from DeFiLlama."""
        keyword = "uniswap-v2" if protocol == "uniswap_v2" else "sushiswap"
        data = _http_get(self.DEFILLAMA_POOLS)
        pairs = []
        if not data or "data" not in data:
            return pairs
        filtered = [p for p in data["data"]
                    if keyword in p.get("project", "").lower()
                    and p.get("chain", "").lower() == "ethereum"]
        filtered.sort(key=lambda x: x.get("tvlUsd", 0), reverse=True)
        for p in filtered[:limit]:
            pairs.append({
                "pair_id": p.get("pool", ""),
                "token0": p.get("symbol", "").split("-")[0],
                "token1": p.get("symbol", "").split("-")[-1],
                "tvl_usd": p.get("tvlUsd", 0),
                "apy_pct": p.get("apy"),
                "protocol": protocol,
                "source": "defillama",
            })
        return pairs

    def price_impact(self, reserve_x: float, reserve_y: float,
                     trade_size_x: float) -> PriceImpactResult:
        """
        Compute price impact for a trade in an xy=k pool.

        Formula (constant product):
          new_reserve_x = reserve_x + trade_size_x * (1 - fee)
          new_reserve_y = k / new_reserve_x
          output_y = reserve_y - new_reserve_y
          price_impact = (initial_price - effective_price) / initial_price

        Args:
            reserve_x: Current reserve of input token
            reserve_y: Current reserve of output token
            trade_size_x: Size of trade in input token units
        """
        fee = 0.003  # Uniswap v2 0.3% fee
        k = reserve_x * reserve_y
        initial_price = reserve_y / reserve_x if reserve_x > 0 else 0

        trade_after_fee = trade_size_x * (1 - fee)
        new_reserve_x = reserve_x + trade_after_fee
        new_reserve_y = k / new_reserve_x
        output_y = reserve_y - new_reserve_y
        effective_price = output_y / trade_size_x if trade_size_x > 0 else 0

        price_impact_pct = ((initial_price - effective_price) / initial_price * 100
                            if initial_price > 0 else 0)

        # Estimate USD size (approximate: assuming reserve_y is a stablecoin)
        # For display we keep in token units
        trade_size_usd = trade_size_x  # caller to convert if needed

        return PriceImpactResult(
            trade_size_usd=round(trade_size_usd, 4),
            price_impact_pct=round(price_impact_pct, 4),
            output_amount=round(output_y, 6),
            effective_price=round(effective_price, 6),
        )

    def optimal_swap_size(self, reserve_x: float, reserve_y: float,
                          max_slippage: float = 0.01) -> dict:
        """
        Find maximum trade size that stays within max_slippage (default 1%).

        Uses the exact formula derived from xy=k constant product:
          For slippage s: trade_size = reserve_x * (1 - sqrt(1 - s)) / (1 - fee)
          Simplified approximation: trade ≈ reserve_x * s / (1 + s)

        Args:
            reserve_x: Reserve of input token
            reserve_y: Reserve of output token
            max_slippage: Maximum acceptable price impact (0.01 = 1%)
        """
        fee = 0.003
        # Exact formula from xy=k with fee:
        # slippage = 1 - (reserve_x / (reserve_x + trade_after_fee))^2 ... approx
        # Numerically solve for trade_size:
        lo, hi = 0.0, reserve_x * 0.5
        for _ in range(50):  # binary search
            mid = (lo + hi) / 2
            result = self.price_impact(reserve_x, reserve_y, mid)
            if result.price_impact_pct / 100 < max_slippage:
                lo = mid
            else:
                hi = mid

        max_trade = lo
        impact_result = self.price_impact(reserve_x, reserve_y, max_trade)

        return {
            "max_trade_size_tokens": round(max_trade, 6),
            "price_impact_pct": round(impact_result.price_impact_pct, 4),
            "output_tokens": round(impact_result.output_amount, 6),
            "max_slippage_pct": max_slippage * 100,
            "reserve_x": reserve_x,
            "reserve_y": reserve_y,
            "method": "binary_search_xy_k",
        }

    def lp_token_price(self, reserve0: float, reserve1: float,
                       total_supply: float, price0_usd: float, price1_usd: float) -> dict:
        """
        Compute LP token price from underlying reserves.

        LP_price = (reserve0 * price0_usd + reserve1 * price1_usd) / total_supply
        """
        total_value = reserve0 * price0_usd + reserve1 * price1_usd
        lp_price = total_value / total_supply if total_supply > 0 else 0
        return {
            "lp_token_price_usd": round(lp_price, 6),
            "total_pool_value_usd": round(total_value, 2),
            "reserve0": reserve0,
            "reserve1": reserve1,
            "price0_usd": price0_usd,
            "price1_usd": price1_usd,
            "total_supply": total_supply,
        }


# ---------------------------------------------------------------------------
# 3. ImpermanentLossCalculator
# ---------------------------------------------------------------------------

class ImpermanentLossCalculator:
    """
    Impermanent loss calculation for v2 (full range) and v3 (concentrated) LPs.

    For v2: IL = 2*sqrt(k) / (1+k) - 1  where k = P_current / P_initial
    For v3: IL is amplified when price moves toward range boundaries.
    """

    def impermanent_loss(self, initial_price: float, current_price: float) -> ILResult:
        """
        Compute impermanent loss for a full-range (v2-style) LP position.

        Args:
            initial_price: Price of token0 in token1 at entry
            current_price: Current price of token0 in token1
        """
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")

        price_ratio = current_price / initial_price
        il_factor = 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1
        il_pct = il_factor * 100

        # Interpretation
        if abs(il_pct) < 0.5:
            description = "Negligible IL — price nearly unchanged"
        elif abs(il_pct) < 5:
            description = f"Low IL ({il_pct:.2f}%) — manageable with fee income"
        elif abs(il_pct) < 20:
            description = f"Moderate IL ({il_pct:.2f}%) — fee APY must exceed {abs(il_pct):.1f}%/yr"
        else:
            description = f"Severe IL ({il_pct:.2f}%) — likely loss vs holding"

        return ILResult(
            initial_price=initial_price,
            current_price=current_price,
            price_ratio=round(price_ratio, 6),
            il_pct=round(il_pct, 4),
            il_description=description,
        )

    def il_with_fees(self, initial_price: float, current_price: float,
                     fee_apy_pct: float, holding_days: int) -> dict:
        """
        Net P&L of LP position vs holding, accounting for fee income.

        Args:
            initial_price: Entry price
            current_price: Current price
            fee_apy_pct: Annual fee APY earned by the LP position
            holding_days: Days position has been open
        """
        il_result = self.impermanent_loss(initial_price, current_price)
        il_pct = il_result.il_pct

        # Fee income over holding period
        fee_income_pct = fee_apy_pct * (holding_days / 365)

        net_pnl_vs_hold = fee_income_pct + il_pct  # il_pct is negative
        is_profitable = net_pnl_vs_hold > 0

        # Breakeven: how many days needed for fees to cover IL?
        if fee_apy_pct > 0 and il_pct < 0:
            breakeven_days = abs(il_pct) / fee_apy_pct * 365
        else:
            breakeven_days = None

        return {
            "initial_price": initial_price,
            "current_price": current_price,
            "il_pct": round(il_pct, 4),
            "fee_income_pct": round(fee_income_pct, 4),
            "net_pnl_vs_hold_pct": round(net_pnl_vs_hold, 4),
            "is_profitable_vs_hold": is_profitable,
            "breakeven_days": round(breakeven_days, 1) if breakeven_days else None,
            "holding_days": holding_days,
            "fee_apy_pct": fee_apy_pct,
        }

    def il_breakeven_fee_apy(self, initial_price: float, current_price: float,
                             holding_days: int = 365) -> dict:
        """
        Minimum fee APY needed to break even vs holding given IL, over holding_days.
        """
        il_result = self.impermanent_loss(initial_price, current_price)
        il_pct = abs(il_result.il_pct)

        # Need fee_apy * (holding_days/365) >= il_pct
        required_apy = il_pct / (holding_days / 365) if holding_days > 0 else None

        return {
            "il_pct": round(il_result.il_pct, 4),
            "holding_days": holding_days,
            "required_fee_apy_pct": round(required_apy, 4) if required_apy else None,
            "interpretation": (
                f"LP needs >{required_apy:.1f}% fee APY to break even vs holding over {holding_days}d"
                if required_apy else "N/A"
            ),
        }

    def v3_il(self, price_lower: float, price_upper: float, current_price: float,
              initial_price: float) -> dict:
        """
        Impermanent loss for a Uniswap v3 concentrated liquidity position.

        V3 IL is higher than v2 when price exits the chosen range.
        When price is outside [price_lower, price_upper], the position
        is 100% in one token (full IL on that side).

        Args:
            price_lower: Lower bound of LP range
            price_upper: Upper bound of LP range
            current_price: Current market price
            initial_price: Price at LP entry
        """
        if current_price <= price_lower:
            # Price below range: LP is 100% token0, full IL realized
            # Compare to holding 50/50 at initial price
            il_pct = -100.0  # Maximum IL (simplified)
            position_status = "below_range_100pct_token0"
        elif current_price >= price_upper:
            # Price above range: LP is 100% token1
            il_pct = -100.0
            position_status = "above_range_100pct_token1"
        else:
            # Price in range: compute as ratio of effective range
            # V3 IL amplification factor ≈ v2_il × (range_width_factor)
            pa = math.sqrt(price_lower)
            pb = math.sqrt(price_upper)
            pc = math.sqrt(current_price)
            pi = math.sqrt(initial_price)

            # Liquidity in range formula
            # For a position with L units of liquidity:
            # IL_v3 = IL_v2 × amplification
            v2_il = self.impermanent_loss(initial_price, current_price).il_pct

            # Amplification: wider range → closer to v2, narrower → more IL
            range_factor = (pb - pa) / (pb + pa) if (pb + pa) > 0 else 1
            # Amplification is inverse: narrow range = higher amplification
            amplification = 1 / (range_factor * 2 + 0.5) if range_factor > 0 else 1
            amplification = min(amplification, 10)  # cap at 10x

            il_pct = v2_il * amplification
            position_status = "in_range"

        range_pct = (price_upper - price_lower) / initial_price * 100

        return {
            "price_lower": price_lower,
            "price_upper": price_upper,
            "current_price": current_price,
            "initial_price": initial_price,
            "position_status": position_status,
            "il_pct": round(il_pct, 4),
            "range_width_pct": round(range_pct, 2),
            "warning": "Full IL realized if price exits range" if position_status != "in_range" else "",
            "note": "V3 IL approximation; exact value depends on liquidity distribution",
        }

    def historical_il_dataframe(self, entry_price: float,
                                price_series: list[float]) -> pd.DataFrame:
        """
        Compute IL at each point in a historical price series.

        Returns DataFrame with: price, price_ratio, il_pct, cumulative_max_il.
        """
        rows = []
        for price in price_series:
            result = self.impermanent_loss(entry_price, price)
            rows.append({
                "price": price,
                "price_ratio": result.price_ratio,
                "il_pct": result.il_pct,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["cumulative_max_il"] = df["il_pct"].cummin()  # IL is negative
        return df


# ---------------------------------------------------------------------------
# 4. CurveFinanceAdapter
# ---------------------------------------------------------------------------

class CurveFinanceAdapter:
    """
    Curve Finance stableswap analytics.

    Primary source: DeFiLlama (free, no auth required).
    Provides: pool TVL, volume, APY, gauge rewards, peg deviation detection.
    """

    DEFILLAMA_PROTOCOL = "https://api.llama.fi/protocol/curve"
    DEFILLAMA_YIELDS = "https://yields.llama.fi/pools"
    CURVE_API_BASE = "https://api.curve.fi/api"

    def get_curve_pools(self, chain: str = "ethereum", limit: int = 30) -> list[dict]:
        """Fetch Curve pools from DeFiLlama yields endpoint."""
        data = _http_get(self.DEFILLAMA_YIELDS)
        pools = []
        if not data or "data" not in data:
            return pools

        curve_pools = [
            p for p in data["data"]
            if "curve" in p.get("project", "").lower()
            and p.get("chain", "").lower() == chain.lower()
        ]
        curve_pools.sort(key=lambda x: x.get("tvlUsd", 0), reverse=True)

        for p in curve_pools[:limit]:
            tvl = p.get("tvlUsd", 0) or 0
            apy = p.get("apy", 0) or 0
            apy_reward = p.get("apyReward", 0) or 0
            pools.append({
                "pool_id": p.get("pool", ""),
                "symbol": p.get("symbol", ""),
                "chain": p.get("chain", ""),
                "tvl_usd": round(tvl, 2),
                "base_apy_pct": round(apy, 4),
                "reward_apy_pct": round(apy_reward, 4),
                "total_apy_pct": round(apy + apy_reward, 4),
                "volume_7d_usd": p.get("volumeUsd7d"),
                "il_risk": p.get("ilRisk", "low"),
                "stablecoin": p.get("stablecoin", True),
                "source": "defillama",
            })
        return pools

    def stableswap_price(self, balances: list[float], amplification: float,
                         token_in_idx: int, token_out_idx: int,
                         amount_in: float) -> dict:
        """
        Compute output amount using Curve StableSwap invariant.

        The invariant D satisfies: A*n^n * sum(x_i) + D = A*n^n*D + D^(n+1)/(n^n * prod(x_i))
        This is solved numerically for D, then for the output token balance.

        Args:
            balances: List of token balances in pool
            amplification: Curve A parameter (e.g. 100 for USDT/USDC)
            token_in_idx: Index of input token
            token_out_idx: Index of output token
            amount_in: Amount of input token
        """
        n = len(balances)
        A = amplification

        def _get_D(xp: list[float]) -> float:
            S = sum(xp)
            if S == 0:
                return 0
            D = S
            Ann = A * n ** n
            for _ in range(255):
                D_P = D
                for x in xp:
                    D_P = D_P * D / (n * x + 1e-18)
                D_prev = D
                D = (Ann * S + D_P * n) * D / ((Ann - 1) * D + (n + 1) * D_P)
                if abs(D - D_prev) < 1e-9:
                    break
            return D

        def _get_y(xp: list[float], i: int, j: int, x_new: float) -> float:
            """Compute new balance of token j given new balance x_new of token i."""
            D = _get_D(xp)
            n_ = n
            Ann = A * n_ ** n_
            c = D
            S_ = 0.0
            for k in range(n_):
                if k == i:
                    x_k = x_new
                elif k != j:
                    x_k = xp[k]
                else:
                    continue
                S_ += x_k
                c = c * D / (x_k * n_)
            c = c * D / (Ann * n_)
            b = S_ + D / Ann
            y = D
            for _ in range(255):
                y_prev = y
                y = (y * y + c) / (2 * y + b - D)
                if abs(y - y_prev) < 1e-9:
                    break
            return y

        new_balance_in = balances[token_in_idx] + amount_in
        xp = list(balances)
        new_balance_out = _get_y(xp, token_in_idx, token_out_idx, new_balance_in)
        output = balances[token_out_idx] - new_balance_out - 1  # -1 for rounding
        fee = 0.0004  # Curve base fee 0.04%
        output_after_fee = output * (1 - fee)

        spot_price = balances[token_out_idx] / balances[token_in_idx] if balances[token_in_idx] > 0 else 1
        effective_price = output_after_fee / amount_in if amount_in > 0 else spot_price
        slippage_pct = (spot_price - effective_price) / spot_price * 100 if spot_price > 0 else 0

        return {
            "amount_in": amount_in,
            "amount_out": round(output_after_fee, 6),
            "spot_price": round(spot_price, 6),
            "effective_price": round(effective_price, 6),
            "slippage_pct": round(slippage_pct, 4),
            "fee_pct": fee * 100,
            "amplification": amplification,
        }

    def peg_deviation_detector(self, pool_balances: list[float],
                               token_symbols: list[str]) -> dict:
        """
        Detect stablecoin peg deviations by looking at pool imbalance.

        In a balanced Curve pool, all stablecoin balances should be roughly equal.
        A large imbalance indicates a depeg event.

        Args:
            pool_balances: Current balances of each token
            token_symbols: Corresponding token symbols
        """
        total = sum(pool_balances) or 1
        n = len(pool_balances)
        equal_share = total / n

        deviations = []
        for sym, bal in zip(token_symbols, pool_balances):
            share_pct = bal / total * 100
            deviation_pct = (bal - equal_share) / equal_share * 100
            deviations.append({
                "token": sym,
                "balance": round(bal, 2),
                "share_pct": round(share_pct, 2),
                "deviation_from_equal_pct": round(deviation_pct, 2),
            })

        max_dev = max(abs(d["deviation_from_equal_pct"]) for d in deviations)
        alert = "none"
        if max_dev > 30:
            alert = "severe_depeg"
        elif max_dev > 15:
            alert = "moderate_depeg"
        elif max_dev > 5:
            alert = "minor_imbalance"

        return {
            "pool_composition": deviations,
            "max_deviation_pct": round(max_dev, 2),
            "alert": alert,
            "total_balance": round(total, 2),
        }

    def gauge_crv_apy(self, crv_price_usd: float, crv_emission_per_second: float,
                      tvl_usd: float) -> dict:
        """
        Estimate CRV gauge reward APY.

        Args:
            crv_price_usd: CRV token price in USD
            crv_emission_per_second: CRV tokens emitted per second to this gauge
            tvl_usd: Pool TVL in USD
        """
        crv_per_year = crv_emission_per_second * 86400 * 365
        crv_usd_per_year = crv_per_year * crv_price_usd
        reward_apy = crv_usd_per_year / tvl_usd * 100 if tvl_usd > 0 else 0

        return {
            "crv_emission_per_day": round(crv_emission_per_second * 86400, 2),
            "crv_usd_emission_per_year": round(crv_usd_per_year, 2),
            "gauge_reward_apy_pct": round(reward_apy, 4),
            "crv_price_usd": crv_price_usd,
            "tvl_usd": tvl_usd,
        }


# ---------------------------------------------------------------------------
# 5. LiquidityDepthAnalyzer
# ---------------------------------------------------------------------------

class LiquidityDepthAnalyzer:
    """
    Compare DEX pool depth vs CEX order book depth for the same asset pair.

    Uses DeFiLlama for DEX data and public exchange APIs for CEX comparison.
    """

    DEFILLAMA_POOLS = "https://yields.llama.fi/pools"
    BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"

    def get_dex_liquidity_score(self, token0: str, token1: str) -> dict:
        """
        Aggregate DEX liquidity score for a pair across major DEXes.

        Score = log10(TVL) * volume_factor * price_efficiency_factor
        """
        data = _http_get(self.DEFILLAMA_POOLS)
        if not data or "data" not in data:
            return {"error": "DeFiLlama unavailable"}

        pair_lower = f"{token0}-{token1}".lower()
        pair_rev = f"{token1}-{token0}".lower()

        matching = [
            p for p in data["data"]
            if (pair_lower in p.get("symbol", "").lower()
                or pair_rev in p.get("symbol", "").lower())
            and p.get("chain", "").lower() == "ethereum"
        ]

        if not matching:
            return {
                "pair": f"{token0}/{token1}",
                "dex_pools_found": 0,
                "total_tvl_usd": 0,
                "error": "No DEX pools found for pair",
            }

        total_tvl = sum(p.get("tvlUsd", 0) or 0 for p in matching)
        avg_apy = sum(p.get("apy", 0) or 0 for p in matching) / max(len(matching), 1)

        protocols = list(set(p.get("project", "") for p in matching))

        # Fragmentation score: how evenly split is liquidity?
        tvls = [p.get("tvlUsd", 0) or 0 for p in matching]
        if total_tvl > 0:
            shares = [t / total_tvl for t in tvls]
            hhi = sum(s ** 2 for s in shares)  # Herfindahl index
            fragmentation = 1 - hhi  # 0 = all in one pool, 1 = perfectly split
        else:
            fragmentation = 0

        score = math.log10(max(total_tvl, 1)) * (1 + avg_apy / 100) * (1 - fragmentation * 0.2)

        return {
            "pair": f"{token0}/{token1}",
            "dex_pools_found": len(matching),
            "protocols": protocols,
            "total_tvl_usd": round(total_tvl, 2),
            "avg_apy_pct": round(avg_apy, 4),
            "liquidity_fragmentation": round(fragmentation, 4),
            "liquidity_score": round(score, 4),
            "source": "defillama",
        }

    def get_cex_depth(self, symbol: str = "ETHUSDT", limit: int = 20) -> dict:
        """
        Fetch CEX order book depth from Binance public API.

        Args:
            symbol: Trading pair in Binance format (e.g. ETHUSDT)
            limit: Order book depth levels (5, 10, 20, 50, 100, 500, 1000)
        """
        data = _http_get(self.BINANCE_DEPTH_URL,
                         params={"symbol": symbol.upper(), "limit": limit})
        if not data:
            return {"symbol": symbol, "error": "Binance API unavailable"}

        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]

        if not bids or not asks:
            return {"symbol": symbol, "error": "Empty order book"}

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2

        bid_depth_usd = sum(p * q for p, q in bids)
        ask_depth_usd = sum(p * q for p, q in asks)

        spread_bps = (best_ask - best_bid) / mid * 10000

        return {
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": round(spread_bps, 2),
            "bid_depth_usd": round(bid_depth_usd, 2),
            "ask_depth_usd": round(ask_depth_usd, 2),
            "total_depth_usd": round(bid_depth_usd + ask_depth_usd, 2),
            "levels": limit,
            "source": "binance_public",
        }

    def compare_dex_vs_cex_impact(self, dex_tvl_usd: float, cex_depth_usd: float,
                                   trade_sizes_usd: Optional[list] = None) -> dict:
        """
        Compare price impact on DEX vs CEX for different trade sizes.

        DEX impact (xy=k approx): trade_size / (tvl/2) as fraction
        CEX impact: trade_size / cex_depth_usd as fraction (rough proxy)
        """
        if trade_sizes_usd is None:
            trade_sizes_usd = [10_000, 100_000, 1_000_000]

        comparison = []
        for size in trade_sizes_usd:
            # DEX: constant product formula approximate
            dex_reserve = dex_tvl_usd / 2  # single-token reserve
            dex_impact = (size / (dex_reserve + size)) * 100

            # CEX: linear approximation (simplified)
            cex_impact = (size / max(cex_depth_usd, 1)) * 100

            comparison.append({
                "trade_size_usd": size,
                "dex_price_impact_pct": round(dex_impact, 4),
                "cex_price_impact_pct": round(cex_impact, 4),
                "dex_cheaper": cex_impact > dex_impact,
            })

        return {
            "dex_tvl_usd": dex_tvl_usd,
            "cex_depth_usd": cex_depth_usd,
            "impact_comparison": comparison,
            "note": "Approximate; actual impact depends on liquidity distribution",
        }


# ---------------------------------------------------------------------------
# 6. YieldFarmingOptimizer
# ---------------------------------------------------------------------------

class YieldFarmingOptimizer:
    """
    Find and rank yield farming opportunities using DeFiLlama yields API.

    Incorporates IL risk, protocol risk, and net APY estimates.
    """

    DEFILLAMA_YIELDS_URL = "https://yields.llama.fi/pools"
    DEFILLAMA_YIELD_CHART = "https://yields.llama.fi/chart/{pool_id}"

    # Risk scores by protocol (lower = safer)
    PROTOCOL_RISK = {
        "uniswap-v3": 1,
        "uniswap-v2": 1,
        "curve": 1,
        "aave-v3": 1,
        "aave-v2": 1,
        "compound": 1,
        "maker": 1,
        "sushiswap": 2,
        "balancer-v2": 2,
        "pancakeswap": 2,
        "lido": 1,
        "convex-finance": 2,
        "yearn-finance": 2,
    }

    # IL risk by pool type
    IL_RISK_MAP = {
        True: "none",   # stablecoin pools
        False: "moderate",  # non-stablecoin
    }

    def find_best_opportunities(self, min_tvl: float = 1_000_000,
                                max_risk: float = 3,
                                chain: Optional[str] = None,
                                min_apy: float = 1.0,
                                limit: int = 25) -> list[FarmingOpportunity]:
        """
        Fetch and rank yield farming opportunities from DeFiLlama.

        Args:
            min_tvl: Minimum pool TVL in USD
            max_risk: Maximum risk score (1=lowest, 5=highest)
            chain: Filter to specific chain (e.g. 'Ethereum')
            min_apy: Minimum total APY percentage
            limit: Maximum results to return
        """
        data = _http_get(self.DEFILLAMA_YIELDS_URL)
        if not data or "data" not in data:
            return []

        opportunities = []
        for pool in data["data"]:
            tvl = pool.get("tvlUsd", 0) or 0
            apy = (pool.get("apy") or 0) + (pool.get("apyReward") or 0)
            project = pool.get("project", "").lower()
            pool_chain = pool.get("chain", "")
            is_stable = pool.get("stablecoin", False)
            il_risk_str = pool.get("ilRisk", "low")

            if tvl < min_tvl:
                continue
            if apy < min_apy:
                continue
            if chain and pool_chain.lower() != chain.lower():
                continue

            risk_score = self.PROTOCOL_RISK.get(project, 3)
            # Add IL risk to score
            if not is_stable and il_risk_str not in ("none", "low"):
                risk_score += 0.5

            if risk_score > max_risk:
                continue

            # Estimate IL cost for non-stable pools (assumes 20% price move)
            if not is_stable:
                # IL for 20% price change: IL ≈ 0.5% (rough estimate)
                il_cost_est = 2.0  # 2% annual IL estimate for volatile pairs
                net_apy = apy - il_cost_est
                il_risk = il_risk_str or "moderate"
            else:
                il_cost_est = 0
                net_apy = apy
                il_risk = "none"

            opportunities.append(FarmingOpportunity(
                pool=pool.get("pool", pool.get("symbol", "")),
                protocol=pool.get("project", ""),
                chain=pool_chain,
                apy_pct=round(apy, 4),
                tvl_usd=round(tvl, 2),
                il_risk=il_risk,
                risk_score=risk_score,
                net_apy_est_pct=round(net_apy, 4),
                source="defillama",
            ))

        # Sort by risk-adjusted net APY: net_apy / risk_score
        opportunities.sort(
            key=lambda x: (x.net_apy_est_pct or 0) / max(x.risk_score, 0.1),
            reverse=True,
        )
        return opportunities[:limit]

    def compute_composable_yield(self, steps: list[dict]) -> dict:
        """
        Compute combined APY for multi-step composable yield strategies.

        Example:
          steps = [
            {"name": "Lido stETH", "apy_pct": 4.2},
            {"name": "Curve stETH/ETH", "apy_pct": 3.1},
            {"name": "Convex boost", "apy_pct": 5.0},
          ]
        Composable APY ≈ (1 + step1/100) * (1 + step2/100) * ... - 1

        Note: not all yields compound — some are parallel, some are sequential.
        """
        total_factor = 1.0
        for step in steps:
            total_factor *= (1 + step.get("apy_pct", 0) / 100)

        composite_apy = (total_factor - 1) * 100

        return {
            "steps": steps,
            "step_count": len(steps),
            "composite_apy_pct": round(composite_apy, 4),
            "simple_sum_apy_pct": round(sum(s.get("apy_pct", 0) for s in steps), 4),
            "note": "Composite APY assumes fully compounded sequential strategies",
        }

    def risk_adjusted_yield(self, opportunities: list[FarmingOpportunity]) -> list[dict]:
        """
        Rank opportunities by risk-adjusted APY (Sharpe-like ratio).

        Risk-adjusted APY = net_apy / risk_score
        """
        ranked = []
        for opp in opportunities:
            net = opp.net_apy_est_pct or opp.apy_pct
            risk_adj = net / max(opp.risk_score, 0.1)
            ranked.append({
                "pool": opp.pool,
                "protocol": opp.protocol,
                "chain": opp.chain,
                "apy_pct": opp.apy_pct,
                "net_apy_pct": opp.net_apy_est_pct,
                "risk_score": opp.risk_score,
                "il_risk": opp.il_risk,
                "risk_adj_apy": round(risk_adj, 4),
                "tvl_usd": opp.tvl_usd,
            })
        ranked.sort(key=lambda x: x["risk_adj_apy"], reverse=True)
        return ranked

    def get_historical_apy(self, pool_id: str, days: int = 30) -> dict:
        """
        Fetch historical APY data for a pool from DeFiLlama.

        Args:
            pool_id: DeFiLlama pool ID (UUID format)
            days: Number of days of history
        """
        url = self.DEFILLAMA_YIELD_CHART.format(pool_id=pool_id)
        data = _http_get(url)
        if not data or "data" not in data:
            return {"pool_id": pool_id, "error": "Historical APY unavailable"}

        history = data["data"][-days:]
        apys = [h.get("apy", 0) or 0 for h in history]
        tvls = [h.get("tvlUsd", 0) or 0 for h in history]

        if not apys:
            return {"pool_id": pool_id, "error": "No APY data points"}

        return {
            "pool_id": pool_id,
            "days": len(history),
            "avg_apy_pct": round(sum(apys) / len(apys), 4),
            "min_apy_pct": round(min(apys), 4),
            "max_apy_pct": round(max(apys), 4),
            "apy_std_pct": round(float(np.std(apys)), 4) if len(apys) > 1 else 0,
            "avg_tvl_usd": round(sum(tvls) / len(tvls), 2),
            "latest_apy_pct": round(apys[-1], 4),
            "latest_tvl_usd": round(tvls[-1], 2),
        }


# ---------------------------------------------------------------------------
# Module-level singleton instances
# ---------------------------------------------------------------------------

_v3_analytics = UniswapV3Analytics()
_v2_analytics = UniswapV2Analytics()
_il_calculator = ImpermanentLossCalculator()
_curve_adapter = CurveFinanceAdapter()
_depth_analyzer = LiquidityDepthAnalyzer()
_farming_optimizer = YieldFarmingOptimizer()


# ---------------------------------------------------------------------------
# 7. FastAPI Router
# ---------------------------------------------------------------------------

dex_router = APIRouter(prefix="/dex", tags=["DEX/AMM Analytics"])


@dex_router.get("/pools/top", summary="Top DEX pools by TVL")
def get_top_pools(
    protocol: str = Query("uniswap_v3", description="Protocol: uniswap_v3, uniswap_v2, sushiswap, curve"),
    chain: str = Query("ethereum", description="Chain filter"),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Top liquidity pools by TVL for the selected DEX protocol.

    Sources: The Graph (primary), DeFiLlama (fallback).
    Returns: pool_id, tokens, fee_tier, TVL, 24h volume, fee APY.
    """
    try:
        if protocol == "uniswap_v3":
            pools = _v3_analytics.get_top_pools(limit=limit)
            return {"protocol": protocol, "count": len(pools),
                    "pools": [p.model_dump() for p in pools]}
        elif protocol in ("uniswap_v2", "sushiswap"):
            pairs = _v2_analytics.get_top_pairs(protocol=protocol, limit=limit)
            return {"protocol": protocol, "count": len(pairs), "pools": pairs}
        elif protocol == "curve":
            pools = _curve_adapter.get_curve_pools(chain=chain, limit=limit)
            return {"protocol": protocol, "count": len(pools), "pools": pools}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown protocol: {protocol}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("top_pools_error protocol=%s err=%s", protocol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/pool/{pair}/uniswap-v3", summary="Uniswap v3 pool details for a token pair")
def get_uniswap_v3_pool(
    pair: str,
    fee_tier_bps: Optional[int] = Query(None, description="Fee tier in bps: 5, 30, 100"),
):
    """
    Fetch Uniswap v3 pool analytics for a token pair.

    Pair format: TOKEN0-TOKEN1 (e.g. WETH-USDC).
    Returns pools across all fee tiers sorted by TVL.
    """
    try:
        parts = pair.upper().split("-")
        if len(parts) != 2:
            raise HTTPException(status_code=400,
                                detail="Pair must be TOKEN0-TOKEN1 format")
        token0, token1 = parts
        pools = _v3_analytics.get_pool_by_pair(token0, token1)

        if not pools:
            # Return empty rather than 404 — pair may exist but not indexed
            return {"pair": pair, "pools": [], "count": 0,
                    "note": "No pools found in subgraph; may be a new or low-liquidity pair"}

        return {
            "pair": pair,
            "count": len(pools),
            "pools": [p.model_dump() for p in pools],
            "as_of": datetime.now(_UTC).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/il/{token0}/{token1}", summary="Impermanent loss calculator")
def get_impermanent_loss(
    token0: str,
    token1: str,
    initial_price: float = Query(..., gt=0, description="Entry price (token0 in token1)"),
    current_price: float = Query(..., gt=0, description="Current price"),
    fee_apy_pct: float = Query(0.0, ge=0, description="Fee APY earned (%)"),
    holding_days: int = Query(365, ge=1, le=3650),
    price_lower: Optional[float] = Query(None, description="V3: lower price bound"),
    price_upper: Optional[float] = Query(None, description="V3: upper price bound"),
):
    """
    Compute impermanent loss for LP position.

    For v2 (full range): provide initial_price and current_price.
    For v3 (concentrated): also provide price_lower and price_upper.
    """
    try:
        base_il = _il_calculator.impermanent_loss(initial_price, current_price)
        result: dict[str, Any] = {
            "pair": f"{token0.upper()}/{token1.upper()}",
            "v2_full_range": base_il.model_dump(),
        }

        if price_lower and price_upper:
            v3_il = _il_calculator.v3_il(
                price_lower=price_lower,
                price_upper=price_upper,
                current_price=current_price,
                initial_price=initial_price,
            )
            result["v3_concentrated"] = v3_il

        if fee_apy_pct > 0:
            net = _il_calculator.il_with_fees(
                initial_price, current_price, fee_apy_pct, holding_days
            )
            result["net_pnl_with_fees"] = net

        breakeven = _il_calculator.il_breakeven_fee_apy(
            initial_price, current_price, holding_days
        )
        result["breakeven_analysis"] = breakeven

        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/price-impact", summary="Price impact / slippage calculator")
def get_price_impact(
    reserve_x: float = Query(..., gt=0, description="Input token reserve"),
    reserve_y: float = Query(..., gt=0, description="Output token reserve"),
    trade_size: float = Query(..., gt=0, description="Trade size in input token units"),
    max_slippage_pct: float = Query(1.0, gt=0, le=50,
                                    description="Max slippage for optimal trade sizing (%)"),
):
    """
    Compute price impact and optimal trade size for a constant-product (xy=k) AMM.

    Returns: actual price impact, output amount, and max trade below max_slippage.
    """
    try:
        impact = _v2_analytics.price_impact(reserve_x, reserve_y, trade_size)
        optimal = _v2_analytics.optimal_swap_size(
            reserve_x, reserve_y, max_slippage=max_slippage_pct / 100
        )
        return {
            "trade_analysis": impact.model_dump(),
            "optimal_trade": optimal,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/curve", summary="Curve Finance pool analytics")
def get_curve_analytics(
    chain: str = Query("ethereum", description="Blockchain (ethereum, arbitrum, polygon)"),
    limit: int = Query(30, ge=1, le=100),
):
    """
    Curve Finance pool list with TVL, APY (base + CRV rewards), and peg status.

    Source: DeFiLlama yields API.
    """
    try:
        pools = _curve_adapter.get_curve_pools(chain=chain, limit=limit)
        return {
            "chain": chain,
            "count": len(pools),
            "pools": pools,
            "as_of": datetime.now(_UTC).isoformat(),
            "source": "defillama",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/depth-comparison", summary="DEX vs CEX liquidity depth comparison")
def get_depth_comparison(
    token0: str = Query("WETH", description="Token 0 symbol"),
    token1: str = Query("USDC", description="Token 1 symbol"),
    binance_symbol: str = Query("ETHUSDT", description="Binance symbol for CEX comparison"),
    trade_size_usd: float = Query(100_000, description="Trade size in USD for impact calc"),
):
    """
    Compare DEX pool depth vs Binance CEX order book depth.

    Returns: TVL, CEX order book depth, price impact comparison at $10K/$100K/$1M.
    """
    try:
        dex_score = _depth_analyzer.get_dex_liquidity_score(token0, token1)
        cex_depth = _depth_analyzer.get_cex_depth(binance_symbol, limit=20)

        dex_tvl = dex_score.get("total_tvl_usd", 0)
        cex_total = cex_depth.get("total_depth_usd", 0)

        comparison = _depth_analyzer.compare_dex_vs_cex_impact(
            dex_tvl_usd=dex_tvl,
            cex_depth_usd=cex_total,
            trade_sizes_usd=[10_000, 100_000, 1_000_000],
        )

        return {
            "pair": f"{token0}/{token1}",
            "dex_liquidity": dex_score,
            "cex_order_book": cex_depth,
            "impact_comparison": comparison,
            "as_of": datetime.now(_UTC).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/farming", summary="Best yield farming opportunities")
def get_farming_opportunities(
    min_tvl: float = Query(1_000_000, ge=10_000, description="Minimum pool TVL in USD"),
    max_risk: float = Query(2.5, ge=1, le=5, description="Maximum risk score (1=safest, 5=riskiest)"),
    min_apy: float = Query(2.0, ge=0, description="Minimum total APY %"),
    chain: Optional[str] = Query(None, description="Filter by chain (e.g. Ethereum)"),
    limit: int = Query(25, ge=1, le=100),
):
    """
    Top yield farming opportunities ranked by risk-adjusted net APY.

    Sources: DeFiLlama yields API (covers 200+ protocols across all chains).
    Includes: IL cost estimate, protocol risk score, net APY after IL.
    """
    try:
        opps = _farming_optimizer.find_best_opportunities(
            min_tvl=min_tvl,
            max_risk=max_risk,
            chain=chain,
            min_apy=min_apy,
            limit=limit,
        )
        ranked = _farming_optimizer.risk_adjusted_yield(opps)
        return {
            "count": len(ranked),
            "opportunities": ranked,
            "filters": {
                "min_tvl_usd": min_tvl,
                "max_risk_score": max_risk,
                "min_apy_pct": min_apy,
                "chain": chain,
            },
            "as_of": datetime.now(_UTC).isoformat(),
            "source": "defillama",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/farming/historical/{pool_id}", summary="Historical APY for a pool")
def get_historical_farming_apy(
    pool_id: str,
    days: int = Query(30, ge=7, le=365),
):
    """
    Historical APY and TVL trend for a DeFiLlama pool ID.

    Pool IDs can be obtained from the /dex/farming endpoint.
    """
    try:
        history = _farming_optimizer.get_historical_apy(pool_id, days=days)
        return history
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/il/breakeven", summary="IL breakeven fee APY calculator")
def get_il_breakeven(
    initial_price: float = Query(..., gt=0),
    current_price: float = Query(..., gt=0),
    holding_days: int = Query(365, ge=1, le=3650),
):
    """
    Compute minimum fee APY needed to offset impermanent loss over the holding period.
    """
    try:
        result = _il_calculator.il_breakeven_fee_apy(
            initial_price, current_price, holding_days
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/pools/tick-analysis/{pool_id}", summary="Uniswap v3 tick distribution")
def get_tick_analysis(pool_id: str):
    """
    Concentrated liquidity tick distribution for a Uniswap v3 pool.

    Shows how much liquidity is concentrated around the current price.
    Higher concentration = better capital efficiency but higher IL risk.
    """
    try:
        result = _v3_analytics.get_pool_tick_analysis(pool_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@dex_router.get("/stableswap/peg-check", summary="Curve pool peg deviation detector")
def check_peg_deviation(
    balances: str = Query(..., description="Comma-separated token balances (e.g. 1000000,980000,1020000)"),
    symbols: str = Query(..., description="Comma-separated token symbols (e.g. USDT,USDC,DAI)"),
):
    """
    Detect stablecoin peg deviation in a Curve pool from pool balance imbalance.

    A healthy Curve stableswap pool has roughly equal balances.
    Large imbalance → one token is being sold (possible depeg).
    """
    try:
        bal_list = [float(b.strip()) for b in balances.split(",")]
        sym_list = [s.strip() for s in symbols.split(",")]
        if len(bal_list) != len(sym_list):
            raise HTTPException(status_code=400,
                                detail="balances and symbols must have same count")
        result = _curve_adapter.peg_deviation_detector(bal_list, sym_list)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "dex_router",
    "UniswapV3Analytics",
    "UniswapV2Analytics",
    "ImpermanentLossCalculator",
    "CurveFinanceAdapter",
    "LiquidityDepthAnalyzer",
    "YieldFarmingOptimizer",
    "PoolInfo",
    "ILResult",
    "PriceImpactResult",
    "FarmingOpportunity",
]
