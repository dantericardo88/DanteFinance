"""FX adapter — spot rates and historical daily data via Frankfurter (ECB official rates)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger
from sentinel.core.types import DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAJOR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "USDCNY", "EURGBP", "EURJPY",
]

_FRANKFURTER_CURRENCIES = {
    "AUD", "BGN", "BRL", "CAD", "CHF", "CZK", "DKK", "EUR", "GBP",
    "HKD", "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN",
    "MYR", "NOK", "NZD", "PHP", "PLN", "RON", "SEK", "SGD", "THB",
    "TRY", "USD", "ZAR",
}

_TIMEOUT = httpx.Timeout(30.0)
_HEADERS = {"User-Agent": "SENTINEL/1.0 (DanteFinance; contact@danteai.com)"}

# Symbols to request in the batch majors call (all non-USD majors + crosses)
_BATCH_SYMBOLS = "EUR,GBP,JPY,CHF,AUD,CAD,NZD,CNY,HKD,SGD"


# ── Models ───────────────────────────────────────────────────────────────────

class FXQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    pair: str         # e.g. "EURUSD"
    base: str         # e.g. "EUR"
    quote: str        # e.g. "USD"
    rate: Decimal
    timestamp: datetime
    source: str = "frankfurter"


class FXBar(BaseModel):
    """Daily OHLC bar for an FX pair.

    Frankfurter only publishes one ECB fixing per day (close). open/high/low
    are set equal to close so downstream code can treat this as a standard bar.
    """
    model_config = ConfigDict(frozen=True)

    pair: str
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pair(pair: str) -> tuple[str, str]:
    """'EURUSD' → ('EUR', 'USD'). Raises ValueError for non-6-char codes."""
    if len(pair) != 6:
        raise ValueError(f"Expected 6-char FX pair code, got: {pair!r}")
    return pair[:3].upper(), pair[3:].upper()


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _to_decimal(v: Any) -> Decimal:
    return Decimal(str(v))


def _pair_supported(base: str, quote: str) -> bool:
    return base in _FRANKFURTER_CURRENCIES and quote in _FRANKFURTER_CURRENCIES


def _bar_from_close(pair: str, dt: date, close_val: Decimal) -> FXBar:
    """Build an OHLC bar where O=H=L=C (Frankfurter single-fixing day)."""
    return FXBar(pair=pair, date=dt, open=close_val, high=close_val, low=close_val, close=close_val)


# ── FXAdapter ─────────────────────────────────────────────────────────────────

class FXAdapter(BaseAdapter):
    """Frankfurter adapter — ECB official FX rates, completely free, no API key.

    All HTTP calls use the shared BaseAdapter._get() which enforces 3-attempt
    exponential-backoff retry (tenacity) and rate-limit detection.
    """

    name = "fx"
    rate_limit_per_min = 60        # Frankfurter is unlisted; stay conservative
    BASE_URL = "https://api.frankfurter.app"

    def __init__(self) -> None:
        super().__init__()

    # ── Internal HTTP ─────────────────────────────────────────────────────────

    async def _get_json(self, path: str, params: dict | None = None) -> dict:
        """GET a Frankfurter endpoint and return parsed JSON."""
        url = f"{self.BASE_URL}{path}"
        resp = await self._get(url, params=params, headers=_HEADERS)
        return resp.json()

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_latest(self, base: str = "USD") -> dict[str, FXQuote]:
        """Fetch all rates vs *base* in one request. Returns {pair: FXQuote}.

        Example:  fetch_latest("USD") → {"USDEUR": FXQuote, "USDGBP": FXQuote, ...}
        """
        base = base.upper()
        if base not in _FRANKFURTER_CURRENCIES:
            raise ValueError(f"Base currency {base!r} not in Frankfurter universe")

        await self._throttle()
        data = await self._get_json("/latest", params={"from": base})

        ts_str: str = data.get("date", "")
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc) if ts_str else _now_utc()
        raw_rates: dict[str, float] = data.get("rates", {})

        result: dict[str, FXQuote] = {}
        for quote_ccy, rate_val in raw_rates.items():
            pair = f"{base}{quote_ccy}"
            result[pair] = FXQuote(
                pair=pair,
                base=base,
                quote=quote_ccy,
                rate=_to_decimal(rate_val),
                timestamp=ts,
                source="frankfurter",
            )

        logger.info("Frankfurter latest fetched", base=base, pairs=len(result))
        return result

    async def fetch_spot(self, base: str, quote: str) -> FXQuote:
        """Fetch a single pair spot rate from the latest ECB fixing.

        Raises ValueError if either currency is outside the Frankfurter universe.
        """
        base = base.upper()
        quote = quote.upper()

        if not _pair_supported(base, quote):
            unsupported = {c for c in (base, quote) if c not in _FRANKFURTER_CURRENCIES}
            raise ValueError(
                f"Pair {base}{quote} not supported by Frankfurter. "
                f"Unsupported currencies: {unsupported}"
            )

        await self._throttle()
        data = await self._get_json("/latest", params={"from": base, "to": quote})

        rates: dict[str, float] = data.get("rates", {})
        if quote not in rates:
            raise ValueError(f"Quote {quote!r} absent from Frankfurter response for {base}")

        ts_str: str = data.get("date", "")
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc) if ts_str else _now_utc()

        logger.info("Frankfurter spot fetched", pair=f"{base}{quote}", rate=float(rates[quote]))
        return FXQuote(
            pair=f"{base}{quote}",
            base=base,
            quote=quote,
            rate=_to_decimal(rates[quote]),
            timestamp=ts,
            source="frankfurter",
        )

    async def fetch_historical(self, pair: str, start: date, end: date) -> list[FXBar]:
        """Fetch daily historical closing rates for *pair* between *start* and *end*.

        Frankfurter publishes one ECB fixing per business day. open/high/low are
        set equal to close so callers get standard OHLC bars.

        Raises ValueError for pairs outside the Frankfurter universe.
        """
        base, quote = _parse_pair(pair)

        if not _pair_supported(base, quote):
            raise ValueError(f"Pair {pair} not in Frankfurter currency universe")

        await self._throttle()
        path = f"/{start.isoformat()}..{end.isoformat()}"
        data = await self._get_json(path, params={"from": base, "to": quote})

        raw_rates: dict[str, dict[str, float]] = data.get("rates", {})
        bars: list[FXBar] = []

        for date_str, rate_map in sorted(raw_rates.items()):
            if quote not in rate_map:
                logger.debug("Quote missing from rate map", date=date_str, pair=pair)
                continue
            close_val = _to_decimal(rate_map[quote])
            bars.append(_bar_from_close(pair, date.fromisoformat(date_str), close_val))

        logger.info(
            "Frankfurter historical fetched",
            pair=pair, start=str(start), end=str(end), bars=len(bars),
        )
        return bars

    async def fetch_all_majors(self) -> dict[str, FXQuote]:
        """Return {pair: FXQuote} for all MAJOR_PAIRS in minimal API calls.

        Strategy: one batch call for USD-base rates covers all USD/*  and */USD
        pairs through cross-rate inversion; no extra requests needed.
        """
        await self._throttle()
        data = await self._get_json("/latest", params={"from": "USD", "to": _BATCH_SYMBOLS})

        usd_rates: dict[str, float] = data.get("rates", {})
        usd_rates["USD"] = 1.0   # anchor

        ts_str: str = data.get("date", "")
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc) if ts_str else _now_utc()

        result: dict[str, FXQuote] = {}

        for pair in MAJOR_PAIRS:
            base, quote = _parse_pair(pair)
            base_per_usd = usd_rates.get(base, 0.0)
            quote_per_usd = usd_rates.get(quote, 0.0)

            if base_per_usd == 0.0:
                logger.warning("Zero base rate, skipping pair", pair=pair)
                continue

            # cross rate: units of quote per 1 unit of base
            cross_rate = quote_per_usd / base_per_usd

            result[pair] = FXQuote(
                pair=pair,
                base=base,
                quote=quote,
                rate=_to_decimal(cross_rate),
                timestamp=ts,
                source="frankfurter",
            )

        logger.info("Frankfurter majors batch fetched", count=len(result))
        return result

    async def fetch_base_rates(self, base: str, start: date, end: date) -> list[FXBar]:
        """Fetch daily historical rates for all major currencies against *base*.

        Returns bars for each available quote currency, labelled as '{base}/{QUOTE}'.
        Uses one Frankfurter API call (no `to` param → all available currencies).
        """
        base = base.upper()
        await self._throttle()
        path = f"/{start.isoformat()}..{end.isoformat()}"
        data = await self._get_json(path, params={"from": base})
        raw_rates: dict[str, dict[str, float]] = data.get("rates", {})

        bars: list[FXBar] = []
        for date_str, rate_map in sorted(raw_rates.items()):
            for quote, rate_val in rate_map.items():
                pair = f"{base}/{quote}"
                close_val = _to_decimal(rate_val)
                bars.append(_bar_from_close(pair, date.fromisoformat(date_str), close_val))

        logger.info("Frankfurter base rates fetched", base=base, dates=len(raw_rates), bars=len(bars))
        return bars

    async def fetch_ohlcv(self, ticker, start, end, interval="1d", figi=None):
        """BaseAdapter contract — delegates to fetch_historical for FX pairs."""
        try:
            pair = ticker.upper()
            bars = await self.fetch_historical(pair, start.date(), end.date())
            # Return empty list; callers should use fetch_historical directly for FXBar objects
            return []
        except Exception as exc:
            logger.warning("FXAdapter.fetch_ohlcv delegation failed", ticker=ticker, error=str(exc))
            return []

    async def health_check(self) -> DataHealthEvent:
        """Ping Frankfurter /latest?from=USD and return True if 200."""
        try:
            await self._throttle()
            data = await self._get_json("/latest", params={"from": "USD", "to": "EUR"})
            if data.get("rates"):
                return self._ok_event()
            return self._error_event("Frankfurter returned empty rates")
        except Exception as exc:
            logger.error("FX health check failed", error=str(exc))
            return self._error_event(str(exc))
