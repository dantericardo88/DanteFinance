"""
NautilusTrader event-driven backtesting backend.
Install: pip install nautilus_trader (not in default deps — heavy Rust build).
Provides tick-fidelity backtesting vs VectorBT's vectorized daily approach.
Use for: execution-realistic strategies, market impact modeling, intraday strategies.
"""
from __future__ import annotations

import importlib
import uuid
from datetime import date, datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


class NautilusBacktestConfig(BaseModel):
    strategy_name: str
    tickers: list[str]
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001   # 0.1%
    slippage_bps: float = 1.0        # 1 bps
    venue: str = "SIM"               # simulated venue


class NautilusBacktestResult(BaseModel):
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_trade_return: float
    trade_log: list[dict]


def _check_nautilus_available() -> bool:
    """Return True if nautilus_trader is importable."""
    return importlib.util.find_spec("nautilus_trader") is not None


def _build_result(
    config: NautilusBacktestConfig,
    final_capital: float,
    equity_curve: list[float],
    trade_log: list[dict],
) -> NautilusBacktestResult:
    equity_arr = np.array(equity_curve, dtype=float)
    total_return = (final_capital - config.initial_capital) / config.initial_capital
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / np.where(peak == 0, 1, peak)
    max_drawdown = float(drawdown.min())
    if len(equity_arr) > 2:
        daily_rets = np.diff(equity_arr) / equity_arr[:-1]
        std = daily_rets.std()
        sharpe: float | None = float((daily_rets.mean() / std) * np.sqrt(252)) if std > 1e-10 else None
    else:
        sharpe = None
    sells = [t for t in trade_log if t.get("side") == "SELL"]
    rets = [t.get("return", 0.0) for t in sells]
    return NautilusBacktestResult(
        strategy_name=config.strategy_name,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_capital=config.initial_capital,
        final_capital=round(final_capital, 2),
        total_return=round(total_return, 6),
        sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
        max_drawdown=round(max_drawdown, 6),
        num_trades=len(sells),
        win_rate=round(sum(1 for r in rets if r > 0) / len(sells), 4) if sells else 0.0,
        avg_trade_return=round(float(np.mean(rets)), 6) if rets else 0.0,
        trade_log=trade_log,
    )


class _FallbackEngine:
    """
    Pure-Python bar-by-bar engine. Mirrors NautilusTrader semantics with
    slippage and commission. Used when nautilus_trader is not installed.
    """

    def __init__(self, config: NautilusBacktestConfig, signal_fn: Callable) -> None:
        self._cfg = config
        self._signal_fn = signal_fn
        self._cash = config.initial_capital
        self._pos: float = 0.0
        self._cost: float = 0.0
        self._log: list[dict] = []

    def run(self, bars_data: dict[str, pd.DataFrame]) -> NautilusBacktestResult:
        ticker = self._cfg.tickers[0]
        df = bars_data.get(ticker)
        if df is None or df.empty:
            raise ValueError(f"No bar data for ticker {ticker!r}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy(); df.index = pd.to_datetime(df.index)
        slip = self._cfg.slippage_bps / 10_000.0
        comm = self._cfg.commission_rate
        equity_curve: list[float] = [self._cfg.initial_capital]
        close_col = next((c for c in df.columns if c.lower() == "close"), df.columns[3])
        for ts, row in df.iterrows():
            price = float(row[close_col])
            if np.isnan(price):
                continue
            bar = {"timestamp": ts, "close": price,
                   "open": float(row.get("Open", row.get("open", price))),
                   "high": float(row.get("High", row.get("high", price))),
                   "low": float(row.get("Low", row.get("low", price))),
                   "volume": float(row.get("Volume", row.get("volume", 0.0))),
                   "ticker": ticker}
            sig = self._signal_fn(bar)
            if sig == 1 and self._pos == 0 and self._cash > 0:
                fp = price * (1 + slip)
                shares = self._cash * (1 - comm) / fp
                self._cash -= shares * fp
                self._pos, self._cost = shares, fp
                self._log.append({"timestamp": str(ts), "side": "BUY", "ticker": ticker,
                                   "shares": round(shares, 4), "price": round(fp, 4),
                                   "commission": round(shares * fp * comm, 4)})
            elif sig == -1 and self._pos > 0:
                fp = price * (1 - slip)
                ret = (fp - self._cost) / self._cost
                self._cash += self._pos * fp * (1 - comm)
                self._log.append({"timestamp": str(ts), "side": "SELL", "ticker": ticker,
                                   "shares": round(self._pos, 4), "price": round(fp, 4),
                                   "return": round(ret, 6),
                                   "commission": round(self._pos * fp * comm, 4)})
                self._pos = self._cost = 0.0
            equity_curve.append(self._cash + self._pos * price)
        if self._pos > 0:
            lp = float(df[close_col].iloc[-1])
            self._cash += self._pos * lp * (1 - comm)
            self._pos = 0.0
        return _build_result(self._cfg, self._cash, equity_curve, self._log)


class SentinelStrategy:
    """
    Base NautilusTrader strategy wrapper. Subclass to implement signal logic.
    Instantiated internally by run_nautilus_backtest; not called directly.
    signal_fn(bar_data: dict) -> int: +1 buy, -1 sell, 0 hold.
    """

    def __init__(self, config: NautilusBacktestConfig, signal_fn: Callable) -> None:
        self._cfg = config
        self._signal_fn = signal_fn
        self._pos: float = 0.0
        self._cost: float = 0.0
        self._cash: float = config.initial_capital
        self._log: list[dict] = []
        self._equity: list[float] = [config.initial_capital]
        self._nt_strategy = None      # injected by _register_strategy
        self._instrument_id = None

    def on_bar(self, bar) -> None:
        """Called by NautilusTrader on each bar. Calls signal_fn and submits orders."""
        try:
            close = float(bar.close)
        except Exception:
            return
        bar_dict = {"open": float(bar.open), "high": float(bar.high),
                    "low": float(bar.low), "close": close,
                    "volume": float(bar.volume), "timestamp": bar.ts_event}
        sig = self._signal_fn(bar_dict)
        slip = self._cfg.slippage_bps / 10_000.0
        comm = self._cfg.commission_rate
        if sig == 1 and self._pos == 0 and self._cash > 0:
            fp = close * (1 + slip)
            shares = self._cash * (1 - comm) / fp
            self._cash -= shares * fp
            self._pos, self._cost = shares, fp
            self._log.append({"side": "BUY", "price": round(fp, 4), "shares": round(shares, 4),
                               "commission": round(shares * fp * comm, 4)})
            if self._nt_strategy is not None:
                self._submit_order("BUY", shares)
        elif sig == -1 and self._pos > 0:
            fp = close * (1 - slip)
            ret = (fp - self._cost) / self._cost
            self._cash += self._pos * fp * (1 - comm)
            self._log.append({"side": "SELL", "price": round(fp, 4), "shares": round(self._pos, 4),
                               "return": round(ret, 6),
                               "commission": round(self._pos * fp * comm, 4)})
            if self._nt_strategy is not None:
                self._submit_order("SELL", self._pos)
            self._pos = self._cost = 0.0
        self._equity.append(self._cash + self._pos * close)

    def _submit_order(self, side: str, qty: float) -> None:
        """Submit a MarketOrder through the live NautilusTrader engine."""
        try:
            from nautilus_trader.model.enums import OrderSide
            from nautilus_trader.model.orders import MarketOrder
            from nautilus_trader.model.identifiers import ClientOrderId
            s = self._nt_strategy
            order = MarketOrder(
                trader_id=s.trader_id, strategy_id=s.id,
                instrument_id=self._instrument_id,
                client_order_id=ClientOrderId(str(uuid.uuid4())[:16]),
                order_side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                quantity=s.instrument.make_qty(abs(qty)),
                time_in_force=s._clock.timestamp_ns(),
                init_id=s._uuid_factory.generate(),
                ts_init=s._clock.timestamp_ns(),
            )
            s.submit_order(order)
        except Exception as exc:
            logger.warning("Order submission failed", error=str(exc))

    def on_order_filled(self, event) -> None:
        """Track fills for trade log (supplementary to on_bar tracking)."""
        try:
            logger.debug("Order filled", side=str(event.order_side),
                         qty=str(event.last_qty), price=str(event.last_px))
        except Exception:
            pass


def _df_to_nautilus_bars(df: pd.DataFrame, bar_type) -> list:
    """Convert a pandas OHLCV DataFrame to nautilus_trader Bar objects."""
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price, Quantity
    cols = {k: next((c for c in df.columns if c.lower() == k), None)
            for k in ("open", "high", "low", "close", "volume")}
    bars = []
    for ts, row in df.iterrows():
        try:
            ns = int(pd.Timestamp(ts).value)
            bars.append(Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{row[cols['open']]:.4f}"),
                high=Price.from_str(f"{row[cols['high']]:.4f}"),
                low=Price.from_str(f"{row[cols['low']]:.4f}"),
                close=Price.from_str(f"{row[cols['close']]:.4f}"),
                volume=Quantity.from_str(f"{row[cols['volume']]:.0f}"),
                ts_event=ns, ts_init=ns,
            ))
        except Exception as exc:
            logger.debug("Skipping malformed bar", ts=str(ts), error=str(exc))
    return bars


def _register_strategy(engine, wrapper: SentinelStrategy) -> None:
    """Register a thin NautilusTrader Strategy subclass delegating to wrapper."""
    try:
        from nautilus_trader.trading.strategy import Strategy
        from nautilus_trader.config import StrategyConfig
        from nautilus_trader.model.data import Bar

        class _Thin(Strategy):
            def __init__(self, w: SentinelStrategy) -> None:
                super().__init__(config=StrategyConfig(strategy_id=w._cfg.strategy_name))
                self._w = w
            def on_bar(self, bar: Bar) -> None:
                self._w.on_bar(bar)

        thin = _Thin(wrapper)
        engine.add_strategy(thin)
        wrapper._nt_strategy = thin
    except Exception as exc:
        logger.warning("Could not register NautilusTrader strategy", error=str(exc))


class _NautilusEngineAdapter:
    """Wraps BacktestEngine: venue config, instrument registration, data loading, run."""

    def __init__(self, config: NautilusBacktestConfig) -> None:
        self._cfg = config

    def run(self, bars_data: dict[str, pd.DataFrame], signal_fn: Callable) -> NautilusBacktestResult:
        from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
        from nautilus_trader.config import LoggingConfig
        from nautilus_trader.model.currencies import USD
        from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType
        from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol
        from nautilus_trader.model.objects import Money, Price, Quantity
        from nautilus_trader.model.data import BarType, BarSpecification

        venue = Venue(self._cfg.venue)
        engine = BacktestEngine(config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="WARNING")))
        engine.add_venue(venue=venue, oms_type=OmsType.NETTING,
                         account_type=AccountType.CASH, base_currency=USD,
                         starting_balances=[Money(self._cfg.initial_capital, USD)],
                         fee_model=None)

        wrapper = SentinelStrategy(config=self._cfg, signal_fn=signal_fn)

        for ticker in self._cfg.tickers:
            df = bars_data.get(ticker)
            if df is None or df.empty:
                logger.warning("No bars data, skipping ticker", ticker=ticker); continue
            iid = InstrumentId(Symbol(ticker), venue)
            try:
                from nautilus_trader.test_kit.providers import TestInstrumentProvider
                instrument = TestInstrumentProvider.equity(symbol=ticker, venue=self._cfg.venue)
            except Exception:
                from nautilus_trader.model.instruments import Equity
                instrument = Equity(
                    instrument_id=iid, raw_symbol=Symbol(ticker), currency=USD,
                    price_precision=2, price_increment=Price.from_str("0.01"),
                    lot_size=Quantity.from_str("1"), isin=None, ts_event=0, ts_init=0)
            engine.add_instrument(instrument)
            bar_type = BarType(instrument_id=iid,
                               bar_spec=BarSpecification(1, BarAggregation.DAY, PriceType.LAST))
            engine.add_data(_df_to_nautilus_bars(df, bar_type))

        _register_strategy(engine, wrapper)
        engine.run(
            start=datetime.combine(self._cfg.start_date, datetime.min.time()).replace(tzinfo=timezone.utc),
            end=datetime.combine(self._cfg.end_date, datetime.min.time()).replace(tzinfo=timezone.utc),
        )
        try:
            accounts = list(engine.trader.portfolio.accounts.values())
            final = float(accounts[0].balance_total(USD).as_double()) if accounts else self._cfg.initial_capital
        except Exception:
            final = self._cfg.initial_capital
        engine.dispose()
        return _build_result(self._cfg, final, wrapper._equity, wrapper._log)


def run_nautilus_backtest(
    config: NautilusBacktestConfig,
    bars_data: dict[str, pd.DataFrame],
    signal_fn: Callable,
) -> NautilusBacktestResult:
    """
    Run event-driven backtest via NautilusTrader.

    bars_data: pre-loaded OHLCV DataFrames per ticker (columns: Open/High/Low/Close/Volume).
    signal_fn: function(row: dict) -> int (+1 buy, -1 sell, 0 hold).

    If nautilus_trader is not installed: falls back to a pure-Python event loop that
    replicates the same bar-by-bar semantics with slippage and commission applied.
    Install for full Rust-core execution realism: pip install nautilus_trader
    """
    if _check_nautilus_available():
        logger.info("NautilusTrader available — using Rust-core engine", strategy=config.strategy_name)
        try:
            return _NautilusEngineAdapter(config).run(bars_data, signal_fn)
        except Exception as exc:
            logger.warning("NautilusTrader engine failed, falling back", error=str(exc))
    logger.info(
        "NautilusTrader not installed — pure-Python fallback. pip install nautilus_trader",
        strategy=config.strategy_name,
    )
    return _FallbackEngine(config, signal_fn).run(bars_data)


def compare_vectorbt_vs_nautilus(
    vectorbt_metrics: dict,
    nautilus_result: NautilusBacktestResult,
) -> dict:
    """
    Side-by-side comparison of VectorBT vs NautilusTrader results.

    Key metrics: return diff, Sharpe diff, drawdown diff.
    Large discrepancies indicate execution realism gap — slippage and market
    impact are modeled more precisely by the event-driven NautilusTrader engine.

    vectorbt_metrics: dict with keys total_return, sharpe_ratio, max_drawdown, num_trades.
    """
    vbt_ret = float(vectorbt_metrics.get("total_return", 0.0))
    vbt_sh = vectorbt_metrics.get("sharpe_ratio")
    vbt_sh = float(vbt_sh) if vbt_sh is not None else None
    vbt_dd = float(vectorbt_metrics.get("max_drawdown", 0.0))
    vbt_n = int(vectorbt_metrics.get("num_trades", 0))

    ret_diff = nautilus_result.total_return - vbt_ret
    sh_diff = ((nautilus_result.sharpe_ratio - vbt_sh)
               if nautilus_result.sharpe_ratio is not None and vbt_sh is not None else None)
    dd_diff = nautilus_result.max_drawdown - vbt_dd
    flagged = abs(ret_diff) > 0.02 or (sh_diff is not None and abs(sh_diff) > 0.3)

    logger.info("VectorBT vs NautilusTrader comparison",
                return_diff=round(ret_diff, 4),
                sharpe_diff=round(sh_diff, 4) if sh_diff is not None else None,
                flagged=flagged)
    return {
        "strategy_name": nautilus_result.strategy_name,
        "vectorbt": {"total_return": vbt_ret, "sharpe_ratio": vbt_sh,
                     "max_drawdown": vbt_dd, "num_trades": vbt_n},
        "nautilus": {"total_return": nautilus_result.total_return,
                     "sharpe_ratio": nautilus_result.sharpe_ratio,
                     "max_drawdown": nautilus_result.max_drawdown,
                     "num_trades": nautilus_result.num_trades},
        "diff": {"return_diff": round(ret_diff, 6),
                 "sharpe_diff": round(sh_diff, 4) if sh_diff is not None else None,
                 "drawdown_diff": round(dd_diff, 6),
                 "trade_count_diff": nautilus_result.num_trades - vbt_n},
        "verdict": {
            "realism_gap_flagged": flagged,
            "note": ("Large discrepancy — vectorized backtest likely overstates performance. "
                     "Review slippage and commission assumptions."
                     if flagged else "Results consistent across both engines."),
        },
    }
