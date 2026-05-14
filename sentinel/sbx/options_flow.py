"""Options flow analysis — unusual activity screening, PC ratio, max pain, sentiment."""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────

class OptionContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    expiration: str
    strike: float
    option_type: str            # "call" | "put"
    last_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: int
    open_interest: int
    implied_volatility: float
    in_the_money: bool
    volume_oi_ratio: float
    dollar_premium: float       # volume × mid_price × 100
    unusual_score: float        # 0-10
    days_to_expiry: int


class OptionsFlow(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    current_price: Optional[float] = None
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int
    pc_volume_ratio: float
    pc_oi_ratio: float
    sentiment: str              # "bullish" | "bearish" | "neutral"
    iv_skew: Optional[float] = None     # put IV − call IV at ±5% strikes
    unusual_contracts: list[OptionContract]
    max_pain: Optional[float] = None
    as_of: str
    warnings: list[str] = Field(default_factory=list)


class MarketOptionsScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: list[str]
    unusual_activity: list[OptionContract]
    most_bullish: Optional[str] = None
    most_bearish: Optional[str] = None
    highest_premium: Optional[str] = None
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_contract(
    vol_oi: float,
    dollar_premium: float,
    iv: float,
    dte: int,
    strike: float,
    current_price: Optional[float],
) -> float:
    score = 0.0

    # Volume/OI component
    if vol_oi > 3.0:
        score += 3.0
    elif vol_oi >= 1.0:
        score += 2.0
    elif vol_oi >= 0.5:
        score += 1.0

    # Dollar premium component
    if dollar_premium > 5_000_000:
        score += 3.0
    elif dollar_premium > 1_000_000:
        score += 2.0
    elif dollar_premium > 100_000:
        score += 1.0

    # IV component
    if iv > 0.80:
        score += 2.0
    elif iv > 0.50:
        score += 1.0

    # Near-term expiry urgency (< 30 days)
    if dte < 30:
        score += 1.0

    # Deep OTM speculative bet (> 10% from money)
    if current_price and current_price > 0:
        otm_pct = abs(strike - current_price) / current_price
        if otm_pct > 0.10:
            score += 1.0

    return min(score, 10.0)


# ── Max Pain ──────────────────────────────────────────────────────────────────

def _compute_max_pain(
    calls_df,
    puts_df,
) -> Optional[float]:
    """Strike where total option-writer loss (holder gain) is minimised."""
    try:
        import pandas as pd  # noqa: F401 — already loaded via yfinance context

        all_strikes = sorted(
            set(np.round(calls_df["strike"].values, 2))
            | set(np.round(puts_df["strike"].values, 2))
        )
        if not all_strikes:
            return None

        call_map = {
            float(np.round(r["strike"], 2)): int(r["openInterest"] or 0)
            for _, r in calls_df.iterrows()
        }
        put_map = {
            float(np.round(r["strike"], 2)): int(r["openInterest"] or 0)
            for _, r in puts_df.iterrows()
        }

        pain = {}
        for S in all_strikes:
            call_loss = sum(
                (S - K) * oi * 100
                for K, oi in call_map.items()
                if K < S and oi > 0
            )
            put_loss = sum(
                (K - S) * oi * 100
                for K, oi in put_map.items()
                if K > S and oi > 0
            )
            pain[S] = call_loss + put_loss

        return float(min(pain, key=lambda k: pain[k]))
    except Exception as exc:
        logger.warning("max_pain computation failed", error=str(exc))
        return None


# ── IV Skew ───────────────────────────────────────────────────────────────────

def _compute_iv_skew(
    calls_df,
    puts_df,
    current_price: float,
) -> Optional[float]:
    """put IV − call IV for strikes nearest ±5% from spot."""
    try:
        target_put_strike = current_price * 0.95
        target_call_strike = current_price * 1.05

        valid_puts = puts_df[puts_df["impliedVolatility"] > 0]
        valid_calls = calls_df[calls_df["impliedVolatility"] > 0]

        if valid_puts.empty or valid_calls.empty:
            return None

        put_idx = (valid_puts["strike"] - target_put_strike).abs().idxmin()
        call_idx = (valid_calls["strike"] - target_call_strike).abs().idxmin()

        put_iv = float(valid_puts.loc[put_idx, "impliedVolatility"])
        call_iv = float(valid_calls.loc[call_idx, "impliedVolatility"])

        return round(put_iv - call_iv, 4)
    except Exception as exc:
        logger.warning("iv_skew computation failed", error=str(exc))
        return None


# ── Contract Parsing ──────────────────────────────────────────────────────────

def _parse_contracts(
    df,
    option_type: str,
    expiration: str,
    dte: int,
    current_price: Optional[float],
) -> list[OptionContract]:
    contracts: list[OptionContract] = []
    for _, row in df.iterrows():
        try:
            volume = int(row.get("volume") or 0)
            oi = int(row.get("openInterest") or 0)
            if volume == 0 or oi == 0:
                continue

            strike = float(row["strike"])
            last_price = float(row.get("lastPrice") or 0.0)
            bid = float(row["bid"]) if row.get("bid") is not None else None
            ask = float(row["ask"]) if row.get("ask") is not None else None
            iv = float(row.get("impliedVolatility") or 0.0)
            itm = bool(row.get("inTheMoney", False))
            symbol = str(row.get("contractSymbol", ""))

            # Mid price for premium calc
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            else:
                mid = last_price

            vol_oi = volume / oi
            dollar_premium = volume * mid * 100.0

            score = _score_contract(
                vol_oi=vol_oi,
                dollar_premium=dollar_premium,
                iv=iv,
                dte=dte,
                strike=strike,
                current_price=current_price,
            )

            contracts.append(OptionContract(
                symbol=symbol,
                expiration=expiration,
                strike=strike,
                option_type=option_type,
                last_price=last_price,
                bid=bid,
                ask=ask,
                volume=volume,
                open_interest=oi,
                implied_volatility=round(iv, 4),
                in_the_money=itm,
                volume_oi_ratio=round(vol_oi, 4),
                dollar_premium=round(dollar_premium, 2),
                unusual_score=round(score, 2),
                days_to_expiry=dte,
            ))
        except Exception as exc:
            logger.warning("contract parse error", option_type=option_type, error=str(exc))

    return contracts


# ── yfinance Fetch (sync, runs in thread) ─────────────────────────────────────

def _fetch_ticker_data(ticker: str, max_expirations: int) -> dict:
    """Synchronous yfinance fetch — call via asyncio.to_thread."""
    import yfinance as yf

    obj = yf.Ticker(ticker)

    current_price: Optional[float] = None
    try:
        current_price = float(obj.fast_info.last_price)
    except Exception:
        pass

    expirations: list[str] = list(obj.options or [])[:max_expirations]

    chains: list[tuple[str, object]] = []
    for exp in expirations:
        try:
            chain = obj.option_chain(exp)
            chains.append((exp, chain))
        except Exception as exc:
            logger.warning("option_chain fetch failed", ticker=ticker, expiration=exp, error=str(exc))

    return {"current_price": current_price, "chains": chains}


# ── Entry Points ──────────────────────────────────────────────────────────────

async def get_options_flow(
    ticker: str,
    min_unusual_score: float = 3.0,
    max_expirations: int = 3,
) -> OptionsFlow:
    """Fetch options chain, compute unusual activity scores, PC ratio, max pain."""
    logger.info("get_options_flow start", ticker=ticker, max_expirations=max_expirations)
    warnings: list[str] = []
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    today = date.today()

    try:
        result = await asyncio.to_thread(_fetch_ticker_data, ticker, max_expirations)
    except Exception as exc:
        msg = f"yfinance fetch failed for {ticker}: {exc}"
        logger.error("get_options_flow fetch error", ticker=ticker, error=str(exc))
        warnings.append(msg)
        return OptionsFlow(
            ticker=ticker,
            total_call_volume=0, total_put_volume=0,
            total_call_oi=0, total_put_oi=0,
            pc_volume_ratio=0.0, pc_oi_ratio=0.0,
            sentiment="neutral",
            unusual_contracts=[],
            as_of=as_of,
            warnings=warnings,
        )

    current_price: Optional[float] = result["current_price"]
    chains: list[tuple[str, object]] = result["chains"]

    if current_price is None:
        warnings.append(f"Could not retrieve current price for {ticker}")

    all_contracts: list[OptionContract] = []
    all_calls_dfs = []
    all_puts_dfs = []

    for expiration, chain in chains:
        try:
            exp_date = date.fromisoformat(expiration)
        except ValueError:
            warnings.append(f"Invalid expiration date: {expiration}")
            continue

        dte = max(0, (exp_date - today).days)
        calls_df = chain.calls
        puts_df = chain.puts

        all_calls_dfs.append(calls_df)
        all_puts_dfs.append(puts_df)

        calls = _parse_contracts(calls_df, "call", expiration, dte, current_price)
        puts = _parse_contracts(puts_df, "put", expiration, dte, current_price)
        all_contracts.extend(calls)
        all_contracts.extend(puts)

    # Aggregate call/put totals
    total_call_vol = sum(c.volume for c in all_contracts if c.option_type == "call")
    total_put_vol = sum(c.volume for c in all_contracts if c.option_type == "put")
    total_call_oi = sum(c.open_interest for c in all_contracts if c.option_type == "call")
    total_put_oi = sum(c.open_interest for c in all_contracts if c.option_type == "put")

    pc_vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0.0
    pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0

    # Sentiment from PC volume ratio
    if pc_vol_ratio > 1.2:
        sentiment = "bearish"
    elif pc_vol_ratio < 0.8:
        sentiment = "bullish"
    else:
        sentiment = "neutral"

    # IV skew — use first expiration that has both calls and puts
    iv_skew: Optional[float] = None
    if current_price and all_calls_dfs and all_puts_dfs:
        try:
            import pandas as pd
            merged_calls = pd.concat(all_calls_dfs, ignore_index=True)
            merged_puts = pd.concat(all_puts_dfs, ignore_index=True)
            iv_skew = _compute_iv_skew(merged_calls, merged_puts, current_price)
        except Exception as exc:
            warnings.append(f"IV skew calculation failed: {exc}")

    # Max pain — use first expiration
    max_pain: Optional[float] = None
    if all_calls_dfs and all_puts_dfs:
        try:
            max_pain = _compute_max_pain(all_calls_dfs[0], all_puts_dfs[0])
        except Exception as exc:
            warnings.append(f"Max pain calculation failed: {exc}")

    # Filter and sort unusual contracts
    unusual = [c for c in all_contracts if c.unusual_score >= min_unusual_score]
    unusual.sort(key=lambda c: c.unusual_score, reverse=True)
    unusual = unusual[:20]

    if not unusual:
        warnings.append(f"No unusual contracts found above score {min_unusual_score}")

    logger.info(
        "get_options_flow complete",
        ticker=ticker,
        total_contracts=len(all_contracts),
        unusual_count=len(unusual),
        sentiment=sentiment,
        pc_vol=round(pc_vol_ratio, 3),
    )

    return OptionsFlow(
        ticker=ticker,
        current_price=current_price,
        total_call_volume=total_call_vol,
        total_put_volume=total_put_vol,
        total_call_oi=total_call_oi,
        total_put_oi=total_put_oi,
        pc_volume_ratio=round(pc_vol_ratio, 4),
        pc_oi_ratio=round(pc_oi_ratio, 4),
        sentiment=sentiment,
        iv_skew=iv_skew,
        unusual_contracts=unusual,
        max_pain=max_pain,
        as_of=as_of,
        warnings=warnings,
    )


async def screen_options_flow(
    tickers: list[str],
    min_unusual_score: float = 5.0,
) -> MarketOptionsScreen:
    """Screen multiple tickers for unusual options activity."""
    logger.info("screen_options_flow start", tickers=tickers, min_score=min_unusual_score)
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    if not tickers:
        return MarketOptionsScreen(
            tickers_screened=[],
            unusual_activity=[],
            as_of=as_of,
            warnings=["No tickers provided"],
        )

    # Fetch all flows concurrently
    flows: list[OptionsFlow] = await asyncio.gather(
        *[get_options_flow(ticker, min_unusual_score=min_unusual_score) for ticker in tickers],
        return_exceptions=False,
    )

    for flow in flows:
        warnings.extend(flow.warnings)

    # Aggregate unusual contracts across all tickers
    all_unusual: list[OptionContract] = []
    for flow in flows:
        all_unusual.extend(flow.unusual_contracts)

    all_unusual.sort(key=lambda c: c.unusual_score, reverse=True)

    # Sentiment rankings
    bullish_flows = [f for f in flows if f.sentiment == "bullish"]
    bearish_flows = [f for f in flows if f.sentiment == "bearish"]

    most_bullish: Optional[str] = None
    most_bearish: Optional[str] = None

    if bullish_flows:
        # Most bullish = lowest PC ratio among bullish sentiment tickers
        most_bullish = min(bullish_flows, key=lambda f: f.pc_volume_ratio).ticker

    if bearish_flows:
        # Most bearish = highest PC ratio among bearish sentiment tickers
        most_bearish = max(bearish_flows, key=lambda f: f.pc_volume_ratio).ticker

    # Ticker with highest total dollar premium
    ticker_premium: dict[str, float] = {}
    for flow in flows:
        total = sum(c.dollar_premium for c in flow.unusual_contracts)
        ticker_premium[flow.ticker] = total

    highest_premium: Optional[str] = None
    if ticker_premium:
        candidate = max(ticker_premium, key=lambda t: ticker_premium[t])
        if ticker_premium[candidate] > 0:
            highest_premium = candidate

    logger.info(
        "screen_options_flow complete",
        tickers_screened=len(tickers),
        total_unusual=len(all_unusual),
        most_bullish=most_bullish,
        most_bearish=most_bearish,
        highest_premium=highest_premium,
    )

    return MarketOptionsScreen(
        tickers_screened=tickers,
        unusual_activity=all_unusual,
        most_bullish=most_bullish,
        most_bearish=most_bearish,
        highest_premium=highest_premium,
        as_of=as_of,
        warnings=warnings,
    )
