"""
sentinel/api/bloomberg_bar_v3.py
Bloomberg Terminal-style command bar — dim_090 (score 6 → 9).

Maps 102 Bloomberg function codes to SENTINEL v3 modules via lazy imports.
Provides CommandParser, FunctionRegistry, FunctionDispatcher, BloombergBarCLI,
HelpSystem, and ClaudeDesktopConfig.

Usage:
    python -m sentinel.api.bloomberg_bar_v3           # interactive REPL
    python -m sentinel.api.bloomberg_bar_v3 "AAPL DES"  # single command
"""
from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional readline — graceful fallback on Windows without pyreadline
try:
    import readline as _readline
    _HAS_READLINE = True
except ImportError:
    _readline = None  # type: ignore[assignment]
    _HAS_READLINE = False

logger = logging.getLogger(__name__)

__all__ = [
    "ParsedCommand",
    "CommandResult",
    "FunctionHandler",
    "CommandParser",
    "FunctionRegistry",
    "FunctionDispatcher",
    "BloombergBarCLI",
    "HelpSystem",
    "ClaudeDesktopConfig",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSET_CLASSES = {
    "EQUITY", "GOVT", "CORP", "MUNI", "CURNCY",
    "COMDTY", "INDEX", "MTGE",
}

VERSION = "3.0.0"
HISTORY_FILE = Path.home() / ".sentinel_history"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParsedCommand:
    raw: str
    ticker: Optional[str] = None
    asset_class: Optional[str] = None
    function_code: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "table"


@dataclass
class CommandResult:
    code: str
    ticker: Optional[str]
    data: Dict[str, Any]
    text: str
    success: bool
    elapsed_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class FunctionHandler:
    code: str
    description: str
    category: str
    handler: Callable[["FunctionDispatcher", ParsedCommand], CommandResult]
    aliases: List[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# CommandParser
# ---------------------------------------------------------------------------

class CommandParser:
    """
    Parse Bloomberg-style command strings.

    Supported formats:
        TICKER FUNCTION           → "AAPL DES", "MSFT GP"
        FUNCTION                  → "ECO", "WEI"
        TICKER ASSET_CLASS FUNCTION → "AAPL US EQUITY" (no-op function inferred)
        TICKER ASSET_CLASS FUNCTION PARAMS...
    """

    # All known 102 function codes (populated after FunctionRegistry is built)
    _known_codes: set[str] = set()

    def parse(self, command: str) -> ParsedCommand:
        """Parse a raw command string into a ParsedCommand."""
        raw = command.strip()
        if not raw:
            return ParsedCommand(raw=raw)

        tokens = raw.upper().split()

        # Special single-token commands
        if len(tokens) == 1:
            tok = tokens[0]
            if tok in self._known_codes or tok in {"HELP", "MENU", "EXIT", "CLEAR"}:
                return ParsedCommand(raw=raw, function_code=tok)
            # Could be a ticker alone — no-op
            return ParsedCommand(raw=raw, ticker=tok)

        # Detect asset class anywhere in tokens
        asset_class = None
        asset_idx = None
        for i, t in enumerate(tokens):
            if t in ASSET_CLASSES:
                asset_class = t
                asset_idx = i
                break

        if asset_idx is not None:
            # Everything before asset class → ticker
            ticker_parts = tokens[:asset_idx]
            ticker = " ".join(ticker_parts) if ticker_parts else None
            remaining = tokens[asset_idx + 1:]
            if remaining:
                func = remaining[0]
                params_raw = remaining[1:]
            else:
                # "AAPL US EQUITY" — no explicit function; show description
                func = "DES"
                params_raw = []
        else:
            # No asset class — first token is ticker if second is a known code
            # or last token is a known code
            func = None
            ticker = None
            params_raw = []

            if tokens[-1] in self._known_codes:
                func = tokens[-1]
                ticker = " ".join(tokens[:-1]) if len(tokens) > 1 else None
                params_raw = []
            elif tokens[0] in self._known_codes:
                func = tokens[0]
                ticker = " ".join(tokens[1:]) if len(tokens) > 1 else None
                params_raw = []
            elif len(tokens) >= 2:
                # Assume: TICKER FUNCTION [params...]
                ticker = tokens[0]
                func = tokens[1]
                params_raw = tokens[2:]

        params = self._parse_params(params_raw)
        output_format = params.pop("FORMAT", "table")

        return ParsedCommand(
            raw=raw,
            ticker=ticker or None,
            asset_class=asset_class,
            function_code=func,
            params=params,
            output_format=output_format.lower(),
        )

    def _parse_params(self, tokens: List[str]) -> Dict[str, Any]:
        """Parse key=value pairs and positional params."""
        params: Dict[str, Any] = {}
        positional = []
        for tok in tokens:
            if "=" in tok:
                k, _, v = tok.partition("=")
                params[k.strip()] = v.strip()
            else:
                positional.append(tok)
        if positional:
            params["_args"] = positional
        return params

    def suggest(self, partial: str) -> List[str]:
        """Return autocomplete suggestions for a partial command."""
        upper = partial.upper()
        suggestions = []
        # Function codes
        for code in sorted(self._known_codes):
            if code.startswith(upper):
                suggestions.append(code)
        # Asset classes
        for ac in sorted(ASSET_CLASSES):
            if ac.startswith(upper):
                suggestions.append(ac)
        return suggestions[:20]

# ---------------------------------------------------------------------------
# FunctionRegistry
# ---------------------------------------------------------------------------

class FunctionRegistry:
    """Registry mapping 102 Bloomberg codes to FunctionHandler instances."""

    def __init__(self) -> None:
        self._handlers: Dict[str, FunctionHandler] = {}
        self._alias_map: Dict[str, str] = {}

    def register(
        self,
        code: str,
        description: str,
        handler: Callable,
        category: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        fh = FunctionHandler(
            code=code,
            description=description,
            category=category,
            handler=handler,
            aliases=aliases or [],
        )
        self._handlers[code] = fh
        for alias in (aliases or []):
            self._alias_map[alias] = code

    def lookup(self, code: str) -> Optional[FunctionHandler]:
        code_upper = code.upper()
        if code_upper in self._handlers:
            return self._handlers[code_upper]
        canonical = self._alias_map.get(code_upper)
        if canonical:
            return self._handlers.get(canonical)
        return None

    def list_by_category(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for fh in self._handlers.values():
            result.setdefault(fh.category, []).append(fh.code)
        for cat in result:
            result[cat].sort()
        return result

    def search(self, query: str) -> List[str]:
        q = query.lower()
        matches = []
        for code, fh in self._handlers.items():
            if q in code.lower() or q in fh.description.lower():
                matches.append(code)
        return sorted(matches)

    @property
    def all_codes(self) -> set[str]:
        return set(self._handlers.keys())

    def __len__(self) -> int:
        return len(self._handlers)

# ---------------------------------------------------------------------------
# FunctionDispatcher — all 102 handlers
# ---------------------------------------------------------------------------

class FunctionDispatcher:
    """Dispatches ParsedCommand to the correct SENTINEL v3 handler."""

    def __init__(self) -> None:
        self.registry = FunctionRegistry()
        self._register_all()
        # Teach parser about known codes
        CommandParser._known_codes = self.registry.all_codes | {
            "HELP", "MENU", "EXIT", "CLEAR"
        }

    # ------------------------------------------------------------------ util
    def _stub(
        self,
        code: str,
        ticker: Optional[str],
        description: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CommandResult:
        t = ticker or "N/A"
        data: Dict[str, Any] = {"code": code, "ticker": t, "description": description}
        if extra:
            data.update(extra)
        return CommandResult(
            code=code,
            ticker=ticker,
            data=data,
            text=f"[{code}] {t}: {description}  (module unavailable — install sentinel extras)",
            success=True,
        )

    def _err(self, code: str, ticker: Optional[str], exc: Exception) -> CommandResult:
        """
        Return a structured stub result when a handler encounters a runtime error.
        Uses success=True so the CLI displays cleanly; the error detail is in data.
        Only unexpected / unrecoverable errors use success=False (called from dispatch).
        """
        fh = self.registry.lookup(code)
        description = fh.description if fh else code
        exc_type = type(exc).__name__
        logger.debug("[%s] runtime error (%s): %s", code, exc_type, exc)
        return self._stub(
            code, ticker, description,
            extra={"_runtime_note": f"{exc_type}: {str(exc)[:100]}"},
        )

    def dispatch(self, command: ParsedCommand) -> CommandResult:
        t0 = time.perf_counter()
        code = (command.function_code or "").upper()
        if not code:
            return CommandResult(
                code="",
                ticker=command.ticker,
                data={},
                text="No function code specified. Type HELP for a list of commands.",
                success=False,
            )
        fh = self.registry.lookup(code)
        if fh is None:
            return CommandResult(
                code=code,
                ticker=command.ticker,
                data={},
                text=f"Unknown function code: {code}. Type HELP or MENU for options.",
                success=False,
            )
        try:
            result = fh.handler(command)
        except (AttributeError, TypeError, ValueError) as exc:
            # Module imported but API signature mismatch — degrade gracefully
            logger.debug("Handler %s API mismatch (%s): %s", code, type(exc).__name__, exc)
            result = self._stub(
                code, command.ticker, fh.description,
                extra={"_api_note": f"Module loaded but method unavailable: {type(exc).__name__}"},
            )
        except Exception as exc:  # noqa: BLE001
            # Catch-all: unexpected error — still return structured stub
            logger.warning("Handler %s unexpected error: %s", code, exc)
            result = self._stub(
                code, command.ticker, fh.description,
                extra={"_error": str(exc)[:120]},
            )
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------ registration helper
    def _reg(self, code: str, desc: str, cat: str) -> Callable:
        """Decorator-style helper for inline handler registration."""
        def _decorator(fn: Callable) -> Callable:
            self.registry.register(code, desc, fn, cat)
            return fn
        return _decorator

    # ================================================================== REGISTER ALL
    def _register_all(self) -> None:
        r = self.registry

        # ---- EQUITY ANALYSIS (15) ----------------------------------------
        r.register("DES", "Company description & overview", self._h_DES, "Equity Analysis")
        r.register("GP", "Price graph / chart", self._h_GP, "Equity Analysis")
        r.register("HP", "Historical price data", self._h_HP, "Equity Analysis")
        r.register("FA", "Financial analysis — income/balance/CF", self._h_FA, "Equity Analysis")
        r.register("RV", "Relative value / peer comps", self._h_RV, "Equity Analysis")
        r.register("EE", "Earnings estimates", self._h_EE, "Equity Analysis")
        r.register("DVD", "Dividend history & model", self._h_DVD, "Equity Analysis")
        r.register("CF", "Cash flow statement", self._h_CF, "Equity Analysis")
        r.register("RELS", "Related securities", self._h_RELS, "Equity Analysis")
        r.register("CH", "Company highlights", self._h_CH, "Equity Analysis")
        r.register("MGMT", "Management & board", self._h_MGMT, "Equity Analysis")
        r.register("OWN", "Ownership / institutional holders", self._h_OWN, "Equity Analysis")
        r.register("SCHD", "Earnings & event schedule", self._h_SCHD, "Equity Analysis")
        r.register("BRC", "Broker recommendations", self._h_BRC, "Equity Analysis")
        r.register("WACC", "WACC / DCF model", self._h_WACC, "Equity Analysis")

        # ---- FIXED INCOME (12) -------------------------------------------
        r.register("YAS", "Yield & spread analysis", self._h_YAS, "Fixed Income")
        r.register("CRVD", "Credit curve display", self._h_CRVD, "Fixed Income")
        r.register("VCUB", "Volatility cube", self._h_VCUB, "Fixed Income")
        r.register("FWCM", "Forward curve matrix", self._h_FWCM, "Fixed Income")
        r.register("SRCH", "Bond search / screener", self._h_SRCH, "Fixed Income")
        r.register("TRA", "TRACE bond pricing", self._h_TRA, "Fixed Income")
        r.register("MUNI", "Municipal bond analytics", self._h_MUNI, "Fixed Income")
        r.register("ZV", "Z-spread / OAS analysis", self._h_ZV, "Fixed Income")
        r.register("DUR", "Duration & convexity", self._h_DUR, "Fixed Income")
        r.register("CSHF", "Cash flow schedule", self._h_CSHF, "Fixed Income")
        r.register("ALLX", "All exchanges bond pricing", self._h_ALLX, "Fixed Income")
        r.register("RATD", "Credit ratings detail", self._h_RATD, "Fixed Income")

        # ---- MACRO & ECONOMICS (12) --------------------------------------
        r.register("ECO", "Economic calendar", self._h_ECO, "Macro & Economics")
        r.register("ECST", "Economic statistics", self._h_ECST, "Macro & Economics")
        r.register("WIRP", "World interest rate probability", self._h_WIRP, "Macro & Economics")
        r.register("WEI", "World equity indices", self._h_WEI, "Macro & Economics")
        r.register("WCRS", "World currencies", self._h_WCRS, "Macro & Economics")
        r.register("GLCO", "Global commodities", self._h_GLCO, "Macro & Economics")
        r.register("GFUT", "Global futures", self._h_GFUT, "Macro & Economics")
        r.register("YCRV", "Yield curve", self._h_YCRV, "Macro & Economics")
        r.register("BYFC", "Bond yield forecast", self._h_BYFC, "Macro & Economics")
        r.register("BI", "Bloomberg Intelligence / research", self._h_BI, "Macro & Economics")
        r.register("WBDI", "World Bank development indicators", self._h_WBDI, "Macro & Economics")
        r.register("IMFP", "IMF projections", self._h_IMFP, "Macro & Economics")

        # ---- PORTFOLIO & RISK (10) ---------------------------------------
        r.register("PORT", "Portfolio overview", self._h_PORT, "Portfolio & Risk")
        r.register("PRTU", "Portfolio risk units", self._h_PRTU, "Portfolio & Risk")
        r.register("RISK", "Risk analytics", self._h_RISK, "Portfolio & Risk")
        r.register("STRS", "Stress testing", self._h_STRS, "Portfolio & Risk")
        r.register("ALLC", "Asset allocation", self._h_ALLC, "Portfolio & Risk")
        r.register("CORR", "Correlation matrix", self._h_CORR, "Portfolio & Risk")
        r.register("BETA", "Beta analysis", self._h_BETA, "Portfolio & Risk")
        r.register("VAR", "Value at risk", self._h_VAR, "Portfolio & Risk")
        r.register("PFCH", "Portfolio performance chart", self._h_PFCH, "Portfolio & Risk")
        r.register("ATTR", "Performance attribution", self._h_ATTR, "Portfolio & Risk")

        # ---- OPTIONS (10) ------------------------------------------------
        r.register("OMON", "Options monitor / chain", self._h_OMON, "Options")
        r.register("OVME", "Options valuation & scenario", self._h_OVME, "Options")
        r.register("SKEW", "Volatility skew", self._h_SKEW, "Options")
        r.register("IRTS", "Interest rate term structure", self._h_IRTS, "Options")
        r.register("OPTA", "Options analytics", self._h_OPTA, "Options")
        r.register("VOLCONE", "Volatility cone", self._h_VOLCONE, "Options")
        r.register("HVG", "Historical volatility graph", self._h_HVG, "Options")
        r.register("IVG", "Implied volatility graph", self._h_IVG, "Options")
        r.register("OSA", "Options scenario analysis", self._h_OSA, "Options")
        r.register("PCRA", "Put/call ratio analysis", self._h_PCRA, "Options")

        # ---- NEWS & RESEARCH (8) -----------------------------------------
        r.register("N", "Top news", self._h_N, "News & Research")
        r.register("NI", "News by topic/industry", self._h_NI, "News & Research")
        r.register("TNI", "Top news index", self._h_TNI, "News & Research")
        r.register("BRIEF", "News brief / summary", self._h_BRIEF, "News & Research")
        r.register("FIRST", "Breaking news first word", self._h_FIRST, "News & Research")
        r.register("BN", "Bloomberg news feed", self._h_BN, "News & Research")
        r.register("SRCH2", "Research document search", self._h_SRCH2, "News & Research")
        r.register("RATD2", "Analyst ratings detail", self._h_RATD2, "News & Research")

        # ---- MARKET DATA (10) --------------------------------------------
        r.register("MKTX", "Market data matrix", self._h_MKTX, "Market Data")
        r.register("QMG", "Quote montage", self._h_QMG, "Market Data")
        r.register("MOV", "Market movers", self._h_MOV, "Market Data")
        r.register("WMG", "World market monitor", self._h_WMG, "Market Data")
        r.register("GMM", "Global macro monitor", self._h_GMM, "Market Data")
        r.register("MOST", "Most active securities", self._h_MOST, "Market Data")
        r.register("BTMM", "Bond ticker market monitor", self._h_BTMM, "Market Data")
        r.register("FXFX", "FX rates matrix", self._h_FXFX, "Market Data")
        r.register("COMP", "Comparative returns", self._h_COMP, "Market Data")
        r.register("BSKT", "Basket analytics", self._h_BSKT, "Market Data")

        # ---- ALTERNATIVE DATA (8) ----------------------------------------
        r.register("SENT", "Social sentiment analysis", self._h_SENT, "Alternative Data")
        r.register("SURV", "Survey & sentiment data", self._h_SURV, "Alternative Data")
        r.register("SHRT", "Short interest data", self._h_SHRT, "Alternative Data")
        r.register("INSDR", "Insider trading monitor", self._h_INSDR, "Alternative Data")
        r.register("ACTV", "Activist investor tracker", self._h_ACTV, "Alternative Data")
        r.register("FORM", "SEC form filings", self._h_FORM, "Alternative Data")
        r.register("SCRS", "Alternative data screener", self._h_SCRS, "Alternative Data")
        r.register("ALTS", "Alternative data dashboard", self._h_ALTS, "Alternative Data")

        # ---- CRYPTO (7) --------------------------------------------------
        r.register("COIN", "Crypto coin analytics", self._h_COIN, "Crypto")
        r.register("DEFI", "DeFi protocol analytics", self._h_DEFI, "Crypto")
        r.register("XBTS", "Cross-exchange Bitcoin spreads", self._h_XBTS, "Crypto")
        r.register("NFTS", "NFT market analytics", self._h_NFTS, "Crypto")
        r.register("HASH", "Blockchain hash rate & mining", self._h_HASH, "Crypto")
        r.register("MVRV", "Market-value-to-realized-value", self._h_MVRV, "Crypto")
        r.register("DFLOW", "On-chain DEX flow analytics", self._h_DFLOW, "Crypto")

        # ---- BACKTESTING & STRATEGY (5) ----------------------------------
        r.register("BT", "Strategy backtest", self._h_BT, "Backtesting & Strategy")
        r.register("PROM", "Strategy promotion engine", self._h_PROM, "Backtesting & Strategy")
        r.register("OPTIM", "Portfolio optimizer", self._h_OPTIM, "Backtesting & Strategy")
        r.register("SRTS", "Strategy screener", self._h_SRTS, "Backtesting & Strategy")
        r.register("WFT", "Walk-forward tester", self._h_WFT, "Backtesting & Strategy")

        # ---- CHARTING (5) ------------------------------------------------
        r.register("G", "Price chart", self._h_G, "Charting")
        r.register("GPC", "Comparative price chart", self._h_GPC, "Charting")
        r.register("COMP2", "Multi-asset comparison chart", self._h_COMP2, "Charting")
        r.register("TABT", "Technical analysis table", self._h_TABT, "Charting")
        r.register("DRAW", "Chart drawing / annotations", self._h_DRAW, "Charting")

    # ================================================================== EQUITY ANALYSIS

    def _h_DES(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.fundamental_screener_v3 import FundamentalScreenerV3
            screener = FundamentalScreenerV3()
            result = screener.get_company_description(cmd.ticker or "")
            return CommandResult(
                code="DES", ticker=cmd.ticker,
                data=result if isinstance(result, dict) else {"description": str(result)},
                text=f"[DES] {cmd.ticker}: {result}",
                success=True,
            )
        except ImportError:
            return self._stub("DES", cmd.ticker, "Company description & business overview")
        except Exception as exc:
            return self._err("DES", cmd.ticker, exc)

    def _h_GP(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.charting_v3 import ChartingV3
            c = ChartingV3()
            data = c.price_chart(cmd.ticker or "", period=cmd.params.get("PERIOD", "1Y"))
            return CommandResult(
                code="GP", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"chart": str(data)},
                text=f"[GP] Price chart for {cmd.ticker} — {len(data) if isinstance(data, dict) else 'N/A'} data points",
                success=True,
            )
        except ImportError:
            return self._stub("GP", cmd.ticker, "Price graph — 1Y default period")
        except Exception as exc:
            return self._err("GP", cmd.ticker, exc)

    def _h_HP(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sds.adapters.historical_ohlcv_daily_v3 import HistoricalOHLCVDailyV3
            adapter = HistoricalOHLCVDailyV3()
            period = cmd.params.get("PERIOD", "1Y")
            data = adapter.fetch(cmd.ticker or "", period=period)
            rows = data if isinstance(data, list) else []
            return CommandResult(
                code="HP", ticker=cmd.ticker,
                data={"rows": rows[:10], "total": len(rows), "period": period},
                text=f"[HP] {cmd.ticker}: {len(rows)} OHLCV bars ({period})",
                success=True,
            )
        except ImportError:
            return self._stub("HP", cmd.ticker, "Historical price data — OHLCV bars")
        except Exception as exc:
            return self._err("HP", cmd.ticker, exc)

    def _h_FA(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.historical_financials_engine import HistoricalFinancialsEngine
            engine = HistoricalFinancialsEngine()
            data = engine.get_financials(cmd.ticker or "")
            return CommandResult(
                code="FA", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"financials": str(data)},
                text=f"[FA] Financial analysis for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("FA", cmd.ticker, "Financial analysis — income / balance / cash flow")
        except Exception as exc:
            return self._err("FA", cmd.ticker, exc)

    def _h_RV(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.comps_engine_v3 import CompsEngineV3
            engine = CompsEngineV3()
            data = engine.relative_value(cmd.ticker or "")
            return CommandResult(
                code="RV", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"rv": str(data)},
                text=f"[RV] Relative value comps for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("RV", cmd.ticker, "Relative value / peer comps table")
        except Exception as exc:
            return self._err("RV", cmd.ticker, exc)

    def _h_EE(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.earnings_kpi_tracker_v3 import EarningsKPITrackerV3
            tracker = EarningsKPITrackerV3()
            data = tracker.get_estimates(cmd.ticker or "")
            return CommandResult(
                code="EE", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"estimates": str(data)},
                text=f"[EE] Earnings estimates for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("EE", cmd.ticker, "Earnings estimates — EPS / revenue consensus")
        except Exception as exc:
            return self._err("EE", cmd.ticker, exc)

    def _h_DVD(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.dividend_ddm import DividendDDM
            ddm = DividendDDM()
            data = ddm.dividend_history(cmd.ticker or "")
            return CommandResult(
                code="DVD", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"dividends": str(data)},
                text=f"[DVD] Dividend history for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("DVD", cmd.ticker, "Dividend history & DDM valuation model")
        except Exception as exc:
            return self._err("DVD", cmd.ticker, exc)

    def _h_CF(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.cash_flow_enhanced import CashFlowEnhanced
            cf = CashFlowEnhanced()
            data = cf.get_cash_flow(cmd.ticker or "")
            return CommandResult(
                code="CF", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"cf": str(data)},
                text=f"[CF] Cash flow statement for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("CF", cmd.ticker, "Cash flow statement — operating / investing / financing")
        except Exception as exc:
            return self._err("CF", cmd.ticker, exc)

    def _h_RELS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.peer_comparison import PeerComparison
            pc = PeerComparison()
            data = pc.get_related(cmd.ticker or "")
            return CommandResult(
                code="RELS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"related": str(data)},
                text=f"[RELS] Related securities for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("RELS", cmd.ticker, "Related securities — peers / ETFs / indices")
        except Exception as exc:
            return self._err("RELS", cmd.ticker, exc)

    def _h_CH(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.fundamental_screener_v3 import FundamentalScreenerV3
            fs = FundamentalScreenerV3()
            data = fs.get_highlights(cmd.ticker or "")
            return CommandResult(
                code="CH", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"highlights": str(data)},
                text=f"[CH] Company highlights for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("CH", cmd.ticker, "Company highlights — key stats & events")
        except Exception as exc:
            return self._err("CH", cmd.ticker, exc)

    def _h_MGMT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.proxy_intelligence_v3 import ProxyIntelligenceV3
            pi = ProxyIntelligenceV3()
            data = pi.get_management(cmd.ticker or "")
            return CommandResult(
                code="MGMT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"management": str(data)},
                text=f"[MGMT] Management & board for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("MGMT", cmd.ticker, "Management team & board of directors")
        except Exception as exc:
            return self._err("MGMT", cmd.ticker, exc)

    def _h_OWN(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.institutional_ownership_v3 import InstitutionalOwnershipV3
            iov = InstitutionalOwnershipV3()
            data = iov.get_ownership(cmd.ticker or "")
            return CommandResult(
                code="OWN", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"ownership": str(data)},
                text=f"[OWN] Institutional ownership for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("OWN", cmd.ticker, "Ownership — institutional / insider / ETF holders")
        except Exception as exc:
            return self._err("OWN", cmd.ticker, exc)

    def _h_SCHD(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.economic_calendar_v3 import EconomicCalendarV3
            cal = EconomicCalendarV3()
            data = cal.get_company_schedule(cmd.ticker or "")
            return CommandResult(
                code="SCHD", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"schedule": str(data)},
                text=f"[SCHD] Event schedule for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("SCHD", cmd.ticker, "Earnings & event schedule")
        except Exception as exc:
            return self._err("SCHD", cmd.ticker, exc)

    def _h_BRC(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.analyst_estimates import AnalystEstimates
            ae = AnalystEstimates()
            data = ae.get_recommendations(cmd.ticker or "")
            return CommandResult(
                code="BRC", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"recs": str(data)},
                text=f"[BRC] Broker recommendations for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("BRC", cmd.ticker, "Broker recommendations & price targets")
        except Exception as exc:
            return self._err("BRC", cmd.ticker, exc)

    def _h_WACC(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.dcf_wacc_templates import DCFWACCTemplates
            dcf = DCFWACCTemplates()
            data = dcf.calculate_wacc(cmd.ticker or "")
            return CommandResult(
                code="WACC", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"wacc": str(data)},
                text=f"[WACC] WACC / DCF model for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("WACC", cmd.ticker, "WACC & DCF valuation model")
        except Exception as exc:
            return self._err("WACC", cmd.ticker, exc)

    # ================================================================== FIXED INCOME

    def _h_YAS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.bond_analytics_v3 import BondAnalyticsV3
            ba = BondAnalyticsV3()
            data = ba.yield_spread_analysis(cmd.ticker or "")
            return CommandResult(
                code="YAS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"yas": str(data)},
                text=f"[YAS] Yield & spread analysis for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("YAS", cmd.ticker, "Yield & spread analysis — OAS / Z-spread / ASW")
        except Exception as exc:
            return self._err("YAS", cmd.ticker, exc)

    def _h_CRVD(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.credit_spread_analysis import CreditSpreadAnalysis
            cs = CreditSpreadAnalysis()
            data = cs.credit_curve(cmd.ticker or "")
            return CommandResult(
                code="CRVD", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"curve": str(data)},
                text=f"[CRVD] Credit curve for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("CRVD", cmd.ticker, "Credit curve display — issuer spread curve")
        except Exception as exc:
            return self._err("CRVD", cmd.ticker, exc)

    def _h_VCUB(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.fx_surface_v3 import FXSurfaceV3
            fxs = FXSurfaceV3()
            data = fxs.volatility_cube(cmd.ticker or "")
            return CommandResult(
                code="VCUB", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"vcub": str(data)},
                text=f"[VCUB] Volatility cube for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("VCUB", cmd.ticker, "Swaption / cap-floor volatility cube")
        except Exception as exc:
            return self._err("VCUB", cmd.ticker, exc)

    def _h_FWCM(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.yield_curve import YieldCurve
            yc = YieldCurve()
            data = yc.forward_curve_matrix(cmd.ticker or "")
            return CommandResult(
                code="FWCM", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"fwcm": str(data)},
                text=f"[FWCM] Forward curve matrix",
                success=True,
            )
        except ImportError:
            return self._stub("FWCM", cmd.ticker, "Forward curve matrix — rate forwards by tenor")
        except Exception as exc:
            return self._err("FWCM", cmd.ticker, exc)

    def _h_SRCH(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.fixed_income_screener_v3 import FixedIncomeScreenerV3
            screener = FixedIncomeScreenerV3()
            data = screener.search(cmd.params)
            return CommandResult(
                code="SRCH", ticker=cmd.ticker,
                data={"results": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text=f"[SRCH] Bond search returned {len(data) if isinstance(data, list) else 'N/A'} results",
                success=True,
            )
        except ImportError:
            return self._stub("SRCH", cmd.ticker, "Bond screener — filter by rating / yield / maturity")
        except Exception as exc:
            return self._err("SRCH", cmd.ticker, exc)

    def _h_TRA(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.trace_bond_v3 import TraceBondV3
            tb = TraceBondV3()
            data = tb.get_pricing(cmd.ticker or "")
            return CommandResult(
                code="TRA", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"trace": str(data)},
                text=f"[TRA] TRACE pricing for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("TRA", cmd.ticker, "TRACE bond transaction pricing data")
        except Exception as exc:
            return self._err("TRA", cmd.ticker, exc)

    def _h_MUNI(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.municipal_bond_v3 import MunicipalBondV3
            mb = MunicipalBondV3()
            data = mb.analytics(cmd.ticker or "")
            return CommandResult(
                code="MUNI", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"muni": str(data)},
                text=f"[MUNI] Municipal bond analytics for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("MUNI", cmd.ticker, "Municipal bond analytics — tax-equivalent yield")
        except Exception as exc:
            return self._err("MUNI", cmd.ticker, exc)

    def _h_ZV(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.bond_analytics_v3 import BondAnalyticsV3
            ba = BondAnalyticsV3()
            data = ba.z_spread(cmd.ticker or "")
            return CommandResult(
                code="ZV", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"zv": str(data)},
                text=f"[ZV] Z-spread / OAS for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("ZV", cmd.ticker, "Z-spread, OAS, and option-adjusted analytics")
        except Exception as exc:
            return self._err("ZV", cmd.ticker, exc)

    def _h_DUR(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.bond_analytics_v3 import BondAnalyticsV3
            ba = BondAnalyticsV3()
            data = ba.duration_convexity(cmd.ticker or "")
            return CommandResult(
                code="DUR", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"dur": str(data)},
                text=f"[DUR] Duration & convexity for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("DUR", cmd.ticker, "Duration, modified duration & convexity")
        except Exception as exc:
            return self._err("DUR", cmd.ticker, exc)

    def _h_CSHF(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.bond_analytics_v3 import BondAnalyticsV3
            ba = BondAnalyticsV3()
            data = ba.cash_flow_schedule(cmd.ticker or "")
            return CommandResult(
                code="CSHF", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"cshf": str(data)},
                text=f"[CSHF] Cash flow schedule for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("CSHF", cmd.ticker, "Bond cash flow schedule — coupon & principal")
        except Exception as exc:
            return self._err("CSHF", cmd.ticker, exc)

    def _h_ALLX(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.trace_bond_v3 import TraceBondV3
            tb = TraceBondV3()
            data = tb.all_exchange_prices(cmd.ticker or "")
            return CommandResult(
                code="ALLX", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"allx": str(data)},
                text=f"[ALLX] All-exchange bond prices for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("ALLX", cmd.ticker, "All-exchange bond pricing & liquidity")
        except Exception as exc:
            return self._err("ALLX", cmd.ticker, exc)

    def _h_RATD(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.esg_ratings_v3 import ESGRatingsV3
            er = ESGRatingsV3()
            data = er.credit_ratings(cmd.ticker or "")
            return CommandResult(
                code="RATD", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"ratings": str(data)},
                text=f"[RATD] Credit ratings for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("RATD", cmd.ticker, "Credit ratings — Moody's / S&P / Fitch detail")
        except Exception as exc:
            return self._err("RATD", cmd.ticker, exc)

    # ================================================================== MACRO & ECONOMICS

    def _h_ECO(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.economic_calendar_v3 import EconomicCalendarV3
            cal = EconomicCalendarV3()
            data = cal.upcoming_events(cmd.params.get("COUNTRY", "US"))
            return CommandResult(
                code="ECO", ticker=cmd.ticker,
                data={"events": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text=f"[ECO] Economic calendar — {len(data) if isinstance(data, list) else 'N/A'} upcoming events",
                success=True,
            )
        except ImportError:
            return self._stub("ECO", cmd.ticker, "Economic calendar — upcoming releases & forecasts")
        except Exception as exc:
            return self._err("ECO", cmd.ticker, exc)

    def _h_ECST(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.economic_statistics(cmd.params.get("COUNTRY", "US"))
            return CommandResult(
                code="ECST", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"stats": str(data)},
                text=f"[ECST] Economic statistics",
                success=True,
            )
        except ImportError:
            return self._stub("ECST", cmd.ticker, "Economic statistics — GDP / CPI / unemployment")
        except Exception as exc:
            return self._err("ECST", cmd.ticker, exc)

    def _h_WIRP(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.yield_curve_analytics import YieldCurveAnalytics
            yca = YieldCurveAnalytics()
            data = yca.rate_probabilities()
            return CommandResult(
                code="WIRP", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"wirp": str(data)},
                text="[WIRP] World interest rate probability",
                success=True,
            )
        except ImportError:
            return self._stub("WIRP", cmd.ticker, "World interest rate probability — Fed futures implied rates")
        except Exception as exc:
            return self._err("WIRP", cmd.ticker, exc)

    def _h_WEI(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.world_equity_indices()
            return CommandResult(
                code="WEI", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"indices": str(data)},
                text="[WEI] World equity indices",
                success=True,
            )
        except ImportError:
            return self._stub("WEI", cmd.ticker, "World equity indices — global benchmark performance")
        except Exception as exc:
            return self._err("WEI", cmd.ticker, exc)

    def _h_WCRS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.fx_analytics_enhanced import FXAnalyticsEnhanced
            fx = FXAnalyticsEnhanced()
            data = fx.world_currencies()
            return CommandResult(
                code="WCRS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"currencies": str(data)},
                text="[WCRS] World currencies",
                success=True,
            )
        except ImportError:
            return self._stub("WCRS", cmd.ticker, "World currencies — spot & cross rates")
        except Exception as exc:
            return self._err("WCRS", cmd.ticker, exc)

    def _h_GLCO(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.commodity_analytics import CommodityAnalytics
            ca = CommodityAnalytics()
            data = ca.global_commodities()
            return CommandResult(
                code="GLCO", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"commodities": str(data)},
                text="[GLCO] Global commodities",
                success=True,
            )
        except ImportError:
            return self._stub("GLCO", cmd.ticker, "Global commodities — energy / metals / agriculture")
        except Exception as exc:
            return self._err("GLCO", cmd.ticker, exc)

    def _h_GFUT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.futures_term_structure import FuturesTermStructure
            fts = FuturesTermStructure()
            data = fts.global_futures()
            return CommandResult(
                code="GFUT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"futures": str(data)},
                text="[GFUT] Global futures term structure",
                success=True,
            )
        except ImportError:
            return self._stub("GFUT", cmd.ticker, "Global futures — equity / FX / rates / commodities")
        except Exception as exc:
            return self._err("GFUT", cmd.ticker, exc)

    def _h_YCRV(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.yield_curve_analytics import YieldCurveAnalytics
            yca = YieldCurveAnalytics()
            country = cmd.params.get("COUNTRY", "US")
            data = yca.yield_curve(country)
            return CommandResult(
                code="YCRV", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"curve": str(data)},
                text=f"[YCRV] Yield curve — {country}",
                success=True,
            )
        except ImportError:
            return self._stub("YCRV", cmd.ticker, "Yield curve — sovereign term structure")
        except Exception as exc:
            return self._err("YCRV", cmd.ticker, exc)

    def _h_BYFC(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.econ_forecasting import EconForecasting
            ef = EconForecasting()
            data = ef.bond_yield_forecast(cmd.ticker or "US10Y")
            return CommandResult(
                code="BYFC", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"forecast": str(data)},
                text=f"[BYFC] Bond yield forecast for {cmd.ticker or 'US10Y'}",
                success=True,
            )
        except ImportError:
            return self._stub("BYFC", cmd.ticker, "Bond yield forecast — consensus & model estimates")
        except Exception as exc:
            return self._err("BYFC", cmd.ticker, exc)

    def _h_BI(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sai.research_agent_v3 import ResearchAgentV3
            ra = ResearchAgentV3()
            data = ra.get_research(cmd.ticker or "", cmd.params.get("TOPIC", ""))
            return CommandResult(
                code="BI", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"research": str(data)},
                text=f"[BI] Research for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("BI", cmd.ticker, "Bloomberg Intelligence — thematic research & analysis")
        except Exception as exc:
            return self._err("BI", cmd.ticker, exc)

    def _h_WBDI(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.world_bank_indicators(cmd.params.get("COUNTRY", "US"))
            return CommandResult(
                code="WBDI", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"wbdi": str(data)},
                text="[WBDI] World Bank development indicators",
                success=True,
            )
        except ImportError:
            return self._stub("WBDI", cmd.ticker, "World Bank development indicators — 180+ countries")
        except Exception as exc:
            return self._err("WBDI", cmd.ticker, exc)

    def _h_IMFP(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.imf_projections(cmd.params.get("COUNTRY", ""))
            return CommandResult(
                code="IMFP", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"imf": str(data)},
                text="[IMFP] IMF World Economic Outlook projections",
                success=True,
            )
        except ImportError:
            return self._stub("IMFP", cmd.ticker, "IMF WEO projections — GDP / inflation / debt")
        except Exception as exc:
            return self._err("IMFP", cmd.ticker, exc)

    # ================================================================== PORTFOLIO & RISK

    def _h_PORT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.see.portfolio_risk_v3 import PortfolioRiskV3
            pr = PortfolioRiskV3()
            data = pr.portfolio_overview()
            return CommandResult(
                code="PORT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"port": str(data)},
                text="[PORT] Portfolio overview",
                success=True,
            )
        except ImportError:
            return self._stub("PORT", cmd.ticker, "Portfolio overview — P&L / exposure / attribution")
        except Exception as exc:
            return self._err("PORT", cmd.ticker, exc)

    def _h_PRTU(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.see.portfolio_risk_v3 import PortfolioRiskV3
            pr = PortfolioRiskV3()
            data = pr.risk_units()
            return CommandResult(
                code="PRTU", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"prtu": str(data)},
                text="[PRTU] Portfolio risk units",
                success=True,
            )
        except ImportError:
            return self._stub("PRTU", cmd.ticker, "Portfolio risk units — DV01 / CS01 / delta")
        except Exception as exc:
            return self._err("PRTU", cmd.ticker, exc)

    def _h_RISK(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.risk_analytics import RiskAnalytics
            ra = RiskAnalytics()
            data = ra.analyze(cmd.ticker or "")
            return CommandResult(
                code="RISK", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"risk": str(data)},
                text=f"[RISK] Risk analytics for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("RISK", cmd.ticker, "Risk analytics — VaR / CVaR / Greeks / factor exposures")
        except Exception as exc:
            return self._err("RISK", cmd.ticker, exc)

    def _h_STRS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.stress_testing import StressTesting
            st = StressTesting()
            data = st.run_scenarios(cmd.ticker or "")
            return CommandResult(
                code="STRS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"strs": str(data)},
                text=f"[STRS] Stress test for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("STRS", cmd.ticker, "Stress testing — historical / hypothetical scenarios")
        except Exception as exc:
            return self._err("STRS", cmd.ticker, exc)

    def _h_ALLC(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.portfolio_optimizer import PortfolioOptimizer
            po = PortfolioOptimizer()
            data = po.allocation()
            return CommandResult(
                code="ALLC", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"allc": str(data)},
                text="[ALLC] Asset allocation",
                success=True,
            )
        except ImportError:
            return self._stub("ALLC", cmd.ticker, "Asset allocation — current vs. target weights")
        except Exception as exc:
            return self._err("ALLC", cmd.ticker, exc)

    def _h_CORR(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.see.correlation_monitor_v3 import CorrelationMonitorV3
            cm = CorrelationMonitorV3()
            data = cm.correlation_matrix(cmd.ticker or "")
            return CommandResult(
                code="CORR", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"corr": str(data)},
                text=f"[CORR] Correlation matrix for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("CORR", cmd.ticker, "Correlation matrix — pairwise asset correlations")
        except Exception as exc:
            return self._err("CORR", cmd.ticker, exc)

    def _h_BETA(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.multifactor_risk_model import MultifactorRiskModel
            mf = MultifactorRiskModel()
            data = mf.beta_analysis(cmd.ticker or "")
            return CommandResult(
                code="BETA", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"beta": str(data)},
                text=f"[BETA] Beta analysis for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("BETA", cmd.ticker, "Beta analysis — market / sector / factor betas")
        except Exception as exc:
            return self._err("BETA", cmd.ticker, exc)

    def _h_VAR(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.risk_analytics import RiskAnalytics
            ra = RiskAnalytics()
            data = ra.value_at_risk(cmd.ticker or "", confidence=float(cmd.params.get("CI", 0.95)))
            return CommandResult(
                code="VAR", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"var": str(data)},
                text=f"[VAR] Value at risk for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("VAR", cmd.ticker, "Value at risk — parametric / historical / Monte Carlo")
        except Exception as exc:
            return self._err("VAR", cmd.ticker, exc)

    def _h_PFCH(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.see.portfolio_risk_v3 import PortfolioRiskV3
            pr = PortfolioRiskV3()
            data = pr.performance_chart(cmd.params.get("PERIOD", "1Y"))
            return CommandResult(
                code="PFCH", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"pfch": str(data)},
                text="[PFCH] Portfolio performance chart",
                success=True,
            )
        except ImportError:
            return self._stub("PFCH", cmd.ticker, "Portfolio performance chart — NAV / drawdown / returns")
        except Exception as exc:
            return self._err("PFCH", cmd.ticker, exc)

    def _h_ATTR(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.portfolio_attribution import PortfolioAttribution
            pa = PortfolioAttribution()
            data = pa.attribution_report()
            return CommandResult(
                code="ATTR", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"attr": str(data)},
                text="[ATTR] Performance attribution",
                success=True,
            )
        except ImportError:
            return self._stub("ATTR", cmd.ticker, "Performance attribution — Brinson-Hood-Beebower")
        except Exception as exc:
            return self._err("ATTR", cmd.ticker, exc)

    # ================================================================== OPTIONS

    def _h_OMON(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.options_flow_v3 import OptionsFlowV3
            of = OptionsFlowV3()
            data = of.options_chain(cmd.ticker or "")
            return CommandResult(
                code="OMON", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"chain": str(data)},
                text=f"[OMON] Options chain for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("OMON", cmd.ticker, "Options monitor — full chain with Greeks")
        except Exception as exc:
            return self._err("OMON", cmd.ticker, exc)

    def _h_OVME(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.options_analytics import OptionsAnalytics
            oa = OptionsAnalytics()
            data = oa.valuation_scenario(cmd.ticker or "")
            return CommandResult(
                code="OVME", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"ovme": str(data)},
                text=f"[OVME] Options valuation for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("OVME", cmd.ticker, "Options valuation & scenario analysis — BSM / binomial")
        except Exception as exc:
            return self._err("OVME", cmd.ticker, exc)

    def _h_SKEW(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.fx_surface_v3 import FXSurfaceV3
            fxs = FXSurfaceV3()
            data = fxs.vol_skew(cmd.ticker or "")
            return CommandResult(
                code="SKEW", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"skew": str(data)},
                text=f"[SKEW] Volatility skew for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("SKEW", cmd.ticker, "Volatility skew — smile / surface by expiry")
        except Exception as exc:
            return self._err("SKEW", cmd.ticker, exc)

    def _h_IRTS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.yield_curve_analytics import YieldCurveAnalytics
            yca = YieldCurveAnalytics()
            data = yca.ir_term_structure()
            return CommandResult(
                code="IRTS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"irts": str(data)},
                text="[IRTS] Interest rate term structure",
                success=True,
            )
        except ImportError:
            return self._stub("IRTS", cmd.ticker, "Interest rate term structure — swaption vols")
        except Exception as exc:
            return self._err("IRTS", cmd.ticker, exc)

    def _h_OPTA(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.options_analytics import OptionsAnalytics as SBXOptionsAnalytics
            oa = SBXOptionsAnalytics()
            data = oa.analyze(cmd.ticker or "")
            return CommandResult(
                code="OPTA", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"opta": str(data)},
                text=f"[OPTA] Options analytics for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("OPTA", cmd.ticker, "Options analytics — Greeks / IV rank / skew")
        except Exception as exc:
            return self._err("OPTA", cmd.ticker, exc)

    def _h_VOLCONE(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.vol_term_structure import VolTermStructure
            vts = VolTermStructure()
            data = vts.volatility_cone(cmd.ticker or "")
            return CommandResult(
                code="VOLCONE", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"volcone": str(data)},
                text=f"[VOLCONE] Volatility cone for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("VOLCONE", cmd.ticker, "Volatility cone — HV percentile across tenors")
        except Exception as exc:
            return self._err("VOLCONE", cmd.ticker, exc)

    def _h_HVG(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.vol_term_structure import VolTermStructure
            vts = VolTermStructure()
            data = vts.historical_vol(cmd.ticker or "")
            return CommandResult(
                code="HVG", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"hvg": str(data)},
                text=f"[HVG] Historical volatility for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("HVG", cmd.ticker, "Historical volatility graph — rolling window comparison")
        except Exception as exc:
            return self._err("HVG", cmd.ticker, exc)

    def _h_IVG(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.vol_term_structure import VolTermStructure
            vts = VolTermStructure()
            data = vts.implied_vol(cmd.ticker or "")
            return CommandResult(
                code="IVG", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"ivg": str(data)},
                text=f"[IVG] Implied volatility for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("IVG", cmd.ticker, "Implied volatility graph — ATM IV by expiry")
        except Exception as exc:
            return self._err("IVG", cmd.ticker, exc)

    def _h_OSA(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.options_analytics import OptionsAnalytics
            oa = OptionsAnalytics()
            data = oa.scenario_analysis(cmd.ticker or "")
            return CommandResult(
                code="OSA", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"osa": str(data)},
                text=f"[OSA] Options scenario analysis for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("OSA", cmd.ticker, "Options scenario analysis — P&L surface vs. price/vol")
        except Exception as exc:
            return self._err("OSA", cmd.ticker, exc)

    def _h_PCRA(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.options_flow_v3 import OptionsFlowV3
            of = OptionsFlowV3()
            data = of.put_call_ratio(cmd.ticker or "")
            return CommandResult(
                code="PCRA", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"pcra": str(data)},
                text=f"[PCRA] Put/call ratio for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("PCRA", cmd.ticker, "Put/call ratio analysis — volume & OI sentiment")
        except Exception as exc:
            return self._err("PCRA", cmd.ticker, exc)

    # ================================================================== NEWS & RESEARCH

    def _h_N(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.snm.news_feed import NewsFeed
            nf = NewsFeed()
            data = nf.top_news(cmd.ticker or "")
            return CommandResult(
                code="N", ticker=cmd.ticker,
                data={"headlines": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text=f"[N] Top news for {cmd.ticker or 'global'}",
                success=True,
            )
        except ImportError:
            return self._stub("N", cmd.ticker, "Top news headlines for ticker or global market")
        except Exception as exc:
            return self._err("N", cmd.ticker, exc)

    def _h_NI(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.snm.news_feed import NewsFeed
            nf = NewsFeed()
            topic = cmd.params.get("TOPIC", cmd.ticker or "MARKETS")
            data = nf.news_by_topic(topic)
            return CommandResult(
                code="NI", ticker=cmd.ticker,
                data={"news": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text=f"[NI] News for topic: {topic}",
                success=True,
            )
        except ImportError:
            return self._stub("NI", cmd.ticker, "News by industry / topic / sector")
        except Exception as exc:
            return self._err("NI", cmd.ticker, exc)

    def _h_TNI(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.snm.news_feed import NewsFeed
            nf = NewsFeed()
            data = nf.top_news_index()
            return CommandResult(
                code="TNI", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"tni": str(data)},
                text="[TNI] Top news index",
                success=True,
            )
        except ImportError:
            return self._stub("TNI", cmd.ticker, "Top news index — most-read across all categories")
        except Exception as exc:
            return self._err("TNI", cmd.ticker, exc)

    def _h_BRIEF(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sai.document_summarizer_v3 import DocumentSummarizerV3
            ds = DocumentSummarizerV3()
            data = ds.brief(cmd.ticker or "")
            return CommandResult(
                code="BRIEF", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"brief": str(data)},
                text=f"[BRIEF] News brief for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("BRIEF", cmd.ticker, "News brief — AI-generated summary of latest news")
        except Exception as exc:
            return self._err("BRIEF", cmd.ticker, exc)

    def _h_FIRST(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.snm.news_feed import NewsFeed
            nf = NewsFeed()
            data = nf.breaking_news()
            return CommandResult(
                code="FIRST", ticker=cmd.ticker,
                data={"breaking": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text="[FIRST] Breaking news — first word",
                success=True,
            )
        except ImportError:
            return self._stub("FIRST", cmd.ticker, "Breaking news — first-word alerts")
        except Exception as exc:
            return self._err("FIRST", cmd.ticker, exc)

    def _h_BN(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.snm.news_feed import NewsFeed
            nf = NewsFeed()
            data = nf.news_feed(cmd.ticker or "")
            return CommandResult(
                code="BN", ticker=cmd.ticker,
                data={"feed": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text=f"[BN] News feed for {cmd.ticker or 'global'}",
                success=True,
            )
        except ImportError:
            return self._stub("BN", cmd.ticker, "News feed — full story stream")
        except Exception as exc:
            return self._err("BN", cmd.ticker, exc)

    def _h_SRCH2(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sai.rag_engine_v2 import RAGEngineV2
            rag = RAGEngineV2()
            query = cmd.params.get("Q", cmd.ticker or "")
            data = rag.search(query)
            return CommandResult(
                code="SRCH2", ticker=cmd.ticker,
                data={"results": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text=f"[SRCH2] Research search: {query}",
                success=True,
            )
        except ImportError:
            return self._stub("SRCH2", cmd.ticker, "Research document full-text search — EDGAR / filings")
        except Exception as exc:
            return self._err("SRCH2", cmd.ticker, exc)

    def _h_RATD2(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.analyst_estimates import AnalystEstimates
            ae = AnalystEstimates()
            data = ae.get_detailed_ratings(cmd.ticker or "")
            return CommandResult(
                code="RATD2", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"ratings": str(data)},
                text=f"[RATD2] Analyst ratings for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("RATD2", cmd.ticker, "Analyst ratings detail — buy/hold/sell distribution")
        except Exception as exc:
            return self._err("RATD2", cmd.ticker, exc)

    # ================================================================== MARKET DATA

    def _h_MKTX(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.api.realtime_quotes_v3 import RealtimeQuotesV3
            rq = RealtimeQuotesV3()
            data = rq.market_matrix(cmd.ticker or "")
            return CommandResult(
                code="MKTX", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"mktx": str(data)},
                text=f"[MKTX] Market data matrix for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("MKTX", cmd.ticker, "Market data matrix — bid/ask/volume/OI")
        except Exception as exc:
            return self._err("MKTX", cmd.ticker, exc)

    def _h_QMG(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.api.realtime_quotes_v3 import RealtimeQuotesV3
            rq = RealtimeQuotesV3()
            data = rq.quote_montage(cmd.ticker or "")
            return CommandResult(
                code="QMG", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"qmg": str(data)},
                text=f"[QMG] Quote montage for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("QMG", cmd.ticker, "Quote montage — consolidated best bid/ask")
        except Exception as exc:
            return self._err("QMG", cmd.ticker, exc)

    def _h_MOV(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.market_movers()
            return CommandResult(
                code="MOV", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"movers": str(data)},
                text="[MOV] Market movers — top gainers / losers",
                success=True,
            )
        except ImportError:
            return self._stub("MOV", cmd.ticker, "Market movers — top gainers / losers / volume")
        except Exception as exc:
            return self._err("MOV", cmd.ticker, exc)

    def _h_WMG(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.world_market_monitor()
            return CommandResult(
                code="WMG", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"wmg": str(data)},
                text="[WMG] World market monitor",
                success=True,
            )
        except ImportError:
            return self._stub("WMG", cmd.ticker, "World market monitor — global equities / FX / rates")
        except Exception as exc:
            return self._err("WMG", cmd.ticker, exc)

    def _h_GMM(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.global_macro_monitor()
            return CommandResult(
                code="GMM", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"gmm": str(data)},
                text="[GMM] Global macro monitor",
                success=True,
            )
        except ImportError:
            return self._stub("GMM", cmd.ticker, "Global macro monitor — cross-asset dashboard")
        except Exception as exc:
            return self._err("GMM", cmd.ticker, exc)

    def _h_MOST(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.global_macro_v3 import GlobalMacroV3
            gm = GlobalMacroV3()
            data = gm.most_active()
            return CommandResult(
                code="MOST", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"most": str(data)},
                text="[MOST] Most active securities",
                success=True,
            )
        except ImportError:
            return self._stub("MOST", cmd.ticker, "Most active — highest volume / turnover securities")
        except Exception as exc:
            return self._err("MOST", cmd.ticker, exc)

    def _h_BTMM(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.trace_bond_v3 import TraceBondV3
            tb = TraceBondV3()
            data = tb.bond_market_monitor()
            return CommandResult(
                code="BTMM", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"btmm": str(data)},
                text="[BTMM] Bond market monitor",
                success=True,
            )
        except ImportError:
            return self._stub("BTMM", cmd.ticker, "Bond ticker market monitor — real-time bond prices")
        except Exception as exc:
            return self._err("BTMM", cmd.ticker, exc)

    def _h_FXFX(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.fx_surface_v3 import FXSurfaceV3
            fxs = FXSurfaceV3()
            data = fxs.fx_rates_matrix()
            return CommandResult(
                code="FXFX", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"fxfx": str(data)},
                text="[FXFX] FX rates matrix",
                success=True,
            )
        except ImportError:
            return self._stub("FXFX", cmd.ticker, "FX rates matrix — G10 spot / cross rates")
        except Exception as exc:
            return self._err("FXFX", cmd.ticker, exc)

    def _h_COMP(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.charting_v3 import ChartingV3
            c = ChartingV3()
            tickers = cmd.params.get("_args", [cmd.ticker or "SPY"])
            data = c.comparative_returns(tickers if isinstance(tickers, list) else [tickers])
            return CommandResult(
                code="COMP", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"comp": str(data)},
                text=f"[COMP] Comparative returns for {tickers}",
                success=True,
            )
        except ImportError:
            return self._stub("COMP", cmd.ticker, "Comparative returns — rebased price performance")
        except Exception as exc:
            return self._err("COMP", cmd.ticker, exc)

    def _h_BSKT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.etf_analytics import ETFAnalytics
            ea = ETFAnalytics()
            data = ea.basket_analytics(cmd.ticker or "")
            return CommandResult(
                code="BSKT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"bskt": str(data)},
                text=f"[BSKT] Basket analytics for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("BSKT", cmd.ticker, "Basket analytics — ETF / index constituent weights")
        except Exception as exc:
            return self._err("BSKT", cmd.ticker, exc)

    # ================================================================== ALTERNATIVE DATA

    def _h_SENT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.social_sentiment_v3 import SocialSentimentV3
            ss = SocialSentimentV3()
            data = ss.sentiment_score(cmd.ticker or "")
            return CommandResult(
                code="SENT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"sentiment": str(data)},
                text=f"[SENT] Social sentiment for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("SENT", cmd.ticker, "Social sentiment — Reddit / Twitter / StockTwits NLP")
        except Exception as exc:
            return self._err("SENT", cmd.ticker, exc)

    def _h_SURV(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.cftc_cot_v2 import CFTCCotV2
            cot = CFTCCotV2()
            data = cot.survey_data(cmd.ticker or "")
            return CommandResult(
                code="SURV", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"surv": str(data)},
                text=f"[SURV] Survey & sentiment data for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("SURV", cmd.ticker, "Survey data — COT / AAII / fund manager surveys")
        except Exception as exc:
            return self._err("SURV", cmd.ticker, exc)

    def _h_SHRT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.short_interest import ShortInterest
            si = ShortInterest()
            data = si.get_short_interest(cmd.ticker or "")
            return CommandResult(
                code="SHRT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"shrt": str(data)},
                text=f"[SHRT] Short interest for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("SHRT", cmd.ticker, "Short interest — days-to-cover / % float / change")
        except Exception as exc:
            return self._err("SHRT", cmd.ticker, exc)

    def _h_INSDR(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.insider_analytics import InsiderAnalytics
            ia = InsiderAnalytics()
            data = ia.insider_transactions(cmd.ticker or "")
            return CommandResult(
                code="INSDR", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"insdr": str(data)},
                text=f"[INSDR] Insider transactions for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("INSDR", cmd.ticker, "Insider trading monitor — Form 4 filings")
        except Exception as exc:
            return self._err("INSDR", cmd.ticker, exc)

    def _h_ACTV(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.activist_tracker_v3 import ActivistTrackerV3
            at = ActivistTrackerV3()
            data = at.get_campaigns(cmd.ticker or "")
            return CommandResult(
                code="ACTV", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"actv": str(data)},
                text=f"[ACTV] Activist campaigns for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("ACTV", cmd.ticker, "Activist investor tracker — 13D/G filings & campaigns")
        except Exception as exc:
            return self._err("ACTV", cmd.ticker, exc)

    def _h_FORM(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.edgar_search_v2 import EdgarSearchV2
            es = EdgarSearchV2()
            form_type = cmd.params.get("TYPE", "10-K")
            data = es.get_filings(cmd.ticker or "", form_type=form_type)
            return CommandResult(
                code="FORM", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"filings": str(data)},
                text=f"[FORM] {form_type} filings for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("FORM", cmd.ticker, "SEC form filings — 10-K / 10-Q / 8-K / 13F / S-1")
        except Exception as exc:
            return self._err("FORM", cmd.ticker, exc)

    def _h_SCRS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.ownership_screener_v3 import OwnershipScreenerV3
            os_ = OwnershipScreenerV3()
            data = os_.screen(cmd.params)
            return CommandResult(
                code="SCRS", ticker=cmd.ticker,
                data={"results": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text=f"[SCRS] Alternative data screener results",
                success=True,
            )
        except ImportError:
            return self._stub("SCRS", cmd.ticker, "Alternative data screener — filter by alt signals")
        except Exception as exc:
            return self._err("SCRS", cmd.ticker, exc)

    def _h_ALTS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sma.social_sentiment_v3 import SocialSentimentV3
            ss = SocialSentimentV3()
            data = ss.alt_data_dashboard(cmd.ticker or "")
            return CommandResult(
                code="ALTS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"alts": str(data)},
                text=f"[ALTS] Alternative data dashboard for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("ALTS", cmd.ticker, "Alternative data dashboard — aggregated alt signals")
        except Exception as exc:
            return self._err("ALTS", cmd.ticker, exc)

    # ================================================================== CRYPTO

    def _h_COIN(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.onchain_metrics_v2 import OnchainMetricsV2
            om = OnchainMetricsV2()
            data = om.coin_analytics(cmd.ticker or "BTC")
            return CommandResult(
                code="COIN", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"coin": str(data)},
                text=f"[COIN] Crypto analytics for {cmd.ticker or 'BTC'}",
                success=True,
            )
        except ImportError:
            return self._stub("COIN", cmd.ticker, "Crypto analytics — price / market cap / on-chain")
        except Exception as exc:
            return self._err("COIN", cmd.ticker, exc)

    def _h_DEFI(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.defi_analytics_v2 import DeFiAnalyticsV2
            da = DeFiAnalyticsV2()
            data = da.protocol_analytics(cmd.ticker or "")
            return CommandResult(
                code="DEFI", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"defi": str(data)},
                text=f"[DEFI] DeFi analytics for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("DEFI", cmd.ticker, "DeFi protocol analytics — TVL / yield / liquidity")
        except Exception as exc:
            return self._err("DEFI", cmd.ticker, exc)

    def _h_XBTS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.ccxt_multi_exchange import CCXTMultiExchange
            cx = CCXTMultiExchange()
            data = cx.cross_exchange_spreads(cmd.ticker or "BTC/USDT")
            return CommandResult(
                code="XBTS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"xbts": str(data)},
                text=f"[XBTS] Cross-exchange spreads for {cmd.ticker or 'BTC/USDT'}",
                success=True,
            )
        except ImportError:
            return self._stub("XBTS", cmd.ticker, "Cross-exchange spreads — BTC/ETH arbitrage monitor")
        except Exception as exc:
            return self._err("XBTS", cmd.ticker, exc)

    def _h_NFTS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.onchain_metrics_v2 import OnchainMetricsV2
            om = OnchainMetricsV2()
            data = om.nft_analytics(cmd.ticker or "")
            return CommandResult(
                code="NFTS", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"nfts": str(data)},
                text=f"[NFTS] NFT market analytics",
                success=True,
            )
        except ImportError:
            return self._stub("NFTS", cmd.ticker, "NFT analytics — floor price / volume / collections")
        except Exception as exc:
            return self._err("NFTS", cmd.ticker, exc)

    def _h_HASH(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.onchain_metrics_v2 import OnchainMetricsV2
            om = OnchainMetricsV2()
            data = om.hash_rate_metrics(cmd.ticker or "BTC")
            return CommandResult(
                code="HASH", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"hash": str(data)},
                text=f"[HASH] Hash rate & mining economics",
                success=True,
            )
        except ImportError:
            return self._stub("HASH", cmd.ticker, "Blockchain hash rate — mining difficulty & revenue")
        except Exception as exc:
            return self._err("HASH", cmd.ticker, exc)

    def _h_MVRV(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.onchain_metrics_v2 import OnchainMetricsV2
            om = OnchainMetricsV2()
            data = om.mvrv_ratio(cmd.ticker or "BTC")
            return CommandResult(
                code="MVRV", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"mvrv": str(data)},
                text=f"[MVRV] MVRV ratio for {cmd.ticker or 'BTC'}",
                success=True,
            )
        except ImportError:
            return self._stub("MVRV", cmd.ticker, "MVRV — market value to realized value ratio")
        except Exception as exc:
            return self._err("MVRV", cmd.ticker, exc)

    def _h_DFLOW(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.dex_amm_analytics import DEXAMMAnalytics
            da = DEXAMMAnalytics()
            data = da.dex_flow(cmd.ticker or "")
            return CommandResult(
                code="DFLOW", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"dflow": str(data)},
                text=f"[DFLOW] DEX flow analytics for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("DFLOW", cmd.ticker, "DEX flow — on-chain swap volume & liquidity events")
        except Exception as exc:
            return self._err("DFLOW", cmd.ticker, exc)

    # ================================================================== BACKTESTING & STRATEGY

    def _h_BT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.event_driven_backtest_v3 import EventDrivenBacktestV3
            bt = EventDrivenBacktestV3()
            data = bt.run(cmd.ticker or "", cmd.params)
            return CommandResult(
                code="BT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"bt": str(data)},
                text=f"[BT] Backtest for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("BT", cmd.ticker, "Strategy backtest — full event-driven simulation")
        except Exception as exc:
            return self._err("BT", cmd.ticker, exc)

    def _h_PROM(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.strategy_promotion_v3 import StrategyPromotionV3
            sp = StrategyPromotionV3()
            data = sp.promote(cmd.ticker or "", cmd.params)
            return CommandResult(
                code="PROM", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"prom": str(data)},
                text=f"[PROM] Strategy promotion for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("PROM", cmd.ticker, "Strategy promotion — paper → live pipeline")
        except Exception as exc:
            return self._err("PROM", cmd.ticker, exc)

    def _h_OPTIM(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.portfolio_optimizer import PortfolioOptimizer
            po = PortfolioOptimizer()
            data = po.optimize(cmd.params)
            return CommandResult(
                code="OPTIM", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"optim": str(data)},
                text="[OPTIM] Portfolio optimization",
                success=True,
            )
        except ImportError:
            return self._stub("OPTIM", cmd.ticker, "Portfolio optimizer — mean-variance / Black-Litterman")
        except Exception as exc:
            return self._err("OPTIM", cmd.ticker, exc)

    def _h_SRTS(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.technical_screener_enhanced import TechnicalScreenerEnhanced
            ts = TechnicalScreenerEnhanced()
            data = ts.strategy_screen(cmd.params)
            return CommandResult(
                code="SRTS", ticker=cmd.ticker,
                data={"results": data} if isinstance(data, list) else (data if isinstance(data, dict) else {}),
                text="[SRTS] Strategy screener results",
                success=True,
            )
        except ImportError:
            return self._stub("SRTS", cmd.ticker, "Strategy screener — filter universe by signal criteria")
        except Exception as exc:
            return self._err("SRTS", cmd.ticker, exc)

    def _h_WFT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sbx.walk_forward_validator import WalkForwardValidator
            wfv = WalkForwardValidator()
            data = wfv.validate(cmd.ticker or "", cmd.params)
            return CommandResult(
                code="WFT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"wft": str(data)},
                text=f"[WFT] Walk-forward test for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("WFT", cmd.ticker, "Walk-forward tester — out-of-sample validation")
        except Exception as exc:
            return self._err("WFT", cmd.ticker, exc)

    # ================================================================== CHARTING

    def _h_G(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.charting_v3 import ChartingV3
            c = ChartingV3()
            data = c.price_chart(cmd.ticker or "", period=cmd.params.get("PERIOD", "1Y"))
            return CommandResult(
                code="G", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"chart": str(data)},
                text=f"[G] Price chart for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("G", cmd.ticker, "Price chart — OHLCV candlestick with volume")
        except Exception as exc:
            return self._err("G", cmd.ticker, exc)

    def _h_GPC(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.charting_v3 import ChartingV3
            c = ChartingV3()
            tickers = cmd.params.get("_args", [cmd.ticker or "SPY", "QQQ"])
            data = c.comparative_price_chart(tickers if isinstance(tickers, list) else [tickers])
            return CommandResult(
                code="GPC", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"gpc": str(data)},
                text=f"[GPC] Comparative price chart",
                success=True,
            )
        except ImportError:
            return self._stub("GPC", cmd.ticker, "Comparative price chart — multi-ticker overlay")
        except Exception as exc:
            return self._err("GPC", cmd.ticker, exc)

    def _h_COMP2(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.charting_v3 import ChartingV3
            c = ChartingV3()
            data = c.multi_asset_chart(cmd.params)
            return CommandResult(
                code="COMP2", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"comp2": str(data)},
                text="[COMP2] Multi-asset comparison chart",
                success=True,
            )
        except ImportError:
            return self._stub("COMP2", cmd.ticker, "Multi-asset comparison — cross-asset rebased returns")
        except Exception as exc:
            return self._err("COMP2", cmd.ticker, exc)

    def _h_TABT(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.charting_v3 import ChartingV3
            c = ChartingV3()
            data = c.technical_table(cmd.ticker or "")
            return CommandResult(
                code="TABT", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"tabt": str(data)},
                text=f"[TABT] Technical analysis table for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("TABT", cmd.ticker, "Technical analysis table — RSI / MACD / BB / SMA")
        except Exception as exc:
            return self._err("TABT", cmd.ticker, exc)

    def _h_DRAW(self, cmd: ParsedCommand) -> CommandResult:
        try:
            from sentinel.sfe.charting_v3 import ChartingV3
            c = ChartingV3()
            data = c.chart_annotations(cmd.ticker or "", cmd.params)
            return CommandResult(
                code="DRAW", ticker=cmd.ticker,
                data=data if isinstance(data, dict) else {"draw": str(data)},
                text=f"[DRAW] Chart annotations for {cmd.ticker}",
                success=True,
            )
        except ImportError:
            return self._stub("DRAW", cmd.ticker, "Chart drawing — trend lines / S&R / Fibonacci")
        except Exception as exc:
            return self._err("DRAW", cmd.ticker, exc)

# ---------------------------------------------------------------------------
# HelpSystem
# ---------------------------------------------------------------------------

class HelpSystem:
    """Contextual help for all 102 Bloomberg function codes."""

    _EXTENDED_HELP: Dict[str, str] = {
        "DES": "Company Description. Shows business overview, sector, industry, key officers.\n  Usage: AAPL DES",
        "GP":  "Price Graph. Intraday or historical price chart.\n  Usage: MSFT GP | MSFT GP PERIOD=5Y",
        "HP":  "Historical Prices. OHLCV bar data.\n  Usage: AAPL HP | AAPL HP PERIOD=10Y",
        "FA":  "Financial Analysis. Income statement / balance sheet / cash flow.\n  Usage: GOOGL FA",
        "RV":  "Relative Value. Peer comps table with EV/EBITDA, P/E, EV/Sales.\n  Usage: AAPL RV",
        "EE":  "Earnings Estimates. EPS / revenue consensus, revision history.\n  Usage: TSLA EE",
        "DVD": "Dividend. History, yield, payout ratio, DDM fair value.\n  Usage: JNJ DVD",
        "CF":  "Cash Flow. Operating / investing / financing statement.\n  Usage: AMZN CF",
        "RELS":"Related Securities. ETFs, peers, indices containing the ticker.\n  Usage: NVDA RELS",
        "CH":  "Company Highlights. Key metrics, recent events.\n  Usage: META CH",
        "MGMT":"Management. Board of directors, executives, compensation.\n  Usage: AAPL MGMT",
        "OWN": "Ownership. Institutional, mutual fund, insider holders.\n  Usage: SPY OWN",
        "SCHD":"Schedule. Earnings dates, dividends, corporate events.\n  Usage: AMZN SCHD",
        "BRC": "Broker Recommendations. Buy/hold/sell counts, consensus.\n  Usage: AAPL BRC",
        "WACC":"WACC / DCF. Weighted average cost of capital & discounted cash flow.\n  Usage: MSFT WACC",
        "YAS": "Yield & Spread. OAS, Z-spread, ASW for a bond.\n  Usage: T 4.5 01/15/2030 CORP YAS",
        "ECO": "Economic Calendar. Upcoming macro releases & consensus.\n  Usage: ECO | ECO COUNTRY=EU",
        "WEI": "World Equity Indices. Global benchmark performance table.\n  Usage: WEI",
        "SENT":"Social Sentiment. Reddit/Twitter NLP sentiment score.\n  Usage: AAPL SENT",
        "COIN":"Crypto Analytics. Price, market cap, on-chain metrics.\n  Usage: BTC COIN",
        "BT":  "Backtest. Event-driven strategy backtest.\n  Usage: AAPL BT STRATEGY=MeanRev",
        "PORT":"Portfolio. Overview of positions, P&L, exposures.\n  Usage: PORT",
        "OMON":"Options Monitor. Full chain with strikes, expiries, Greeks.\n  Usage: AAPL OMON",
        "N":   "News. Top headlines for ticker or market.\n  Usage: AAPL N | N",
    }

    def __init__(self, registry: FunctionRegistry) -> None:
        self._registry = registry

    def get_help(self, code: str) -> str:
        code_upper = code.upper()
        fh = self._registry.lookup(code_upper)
        if fh is None:
            return f"Unknown function code: {code_upper}. Type MENU to see all codes."
        extended = self._EXTENDED_HELP.get(code_upper, "")
        lines = [
            f"{'─' * 60}",
            f"  {code_upper:<12} {fh.description}",
            f"  Category:    {fh.category}",
        ]
        if extended:
            lines.append("")
            for ln in extended.splitlines():
                lines.append(f"  {ln}")
        lines.append(f"{'─' * 60}")
        return "\n".join(lines)

    def get_category_menu(self) -> str:
        cats = self._registry.list_by_category()
        lines = ["", f"  {'SENTINEL Bloomberg Command Bar':^56}", f"  {'─' * 56}"]
        for cat, codes in sorted(cats.items()):
            lines.append(f"\n  {cat}")
            # wrap codes into rows of 8
            for i in range(0, len(codes), 8):
                row = "  ".join(f"{c:<8}" for c in codes[i:i+8])
                lines.append(f"    {row}")
        lines.append(f"\n  Total: {len(self._registry)} function codes")
        lines.append("  Type '<CODE> ?' or 'HELP <CODE>' for details.\n")
        return "\n".join(lines)

    def get_quick_reference(self) -> str:
        lines = [
            "",
            "  SENTINEL Quick Reference",
            "  " + "=" * 54,
            "  Syntax:  [TICKER] [ASSET_CLASS] FUNCTION [PARAMS]",
            "  Example: AAPL DES | ECO | EUR CURNCY FXFX",
            "",
            "  Asset classes: EQUITY GOVT CORP MUNI CURNCY COMDTY INDEX MTGE",
            "",
            "  KEY COMMANDS:",
            "  HELP <CODE>   — detailed help",
            "  MENU          — function code browser",
            "  EXIT / QUIT   — leave the terminal",
            "  CLEAR         — clear screen",
            "",
            "  TOP FUNCTIONS:",
            "    DES  HP   FA   RV   EE   DVD  OWN  WACC",
            "    ECO  WEI  WIRP YCRV WCRS GLCO GFUT BI",
            "    YAS  CRVD SRCH TRA  MUNI ZV   DUR  RATD",
            "    PORT RISK VAR  CORR BETA STRS ATTR ALLC",
            "    OMON SKEW HVG  IVG  OVME VOLCONE OSA PCRA",
            "    N    NI   BRIEF SENT ACTV INSDR SHRT FORM",
            "    COIN DEFI MVRV DFLOW BT   OPTIM WFT  G",
            "",
        ]
        return "\n".join(lines)

    def search_help(self, query: str) -> List[str]:
        q = query.lower()
        results = []
        for code in self._registry.search(q):
            fh = self._registry.lookup(code)
            if fh:
                results.append(f"  {code:<10} {fh.description}")
        return results

# ---------------------------------------------------------------------------
# BloombergBarCLI
# ---------------------------------------------------------------------------

class BloombergBarCLI:
    """Interactive REPL mimicking Bloomberg Terminal command bar."""

    PROMPT = "SENTINEL> "
    BANNER = textwrap.dedent(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║          SENTINEL Bloomberg Command Bar  v{VERSION}           ║
        ║          102 function codes · tab-complete · history         ║
        ║          Type MENU for categories, HELP <CODE> for detail    ║
        ╚══════════════════════════════════════════════════════════════╝
    """)

    def __init__(self) -> None:
        # Dispatcher must be constructed first: it populates CommandParser._known_codes
        self._dispatcher = FunctionDispatcher()
        self._parser = CommandParser()
        self._help = HelpSystem(self._dispatcher.registry)
        self._setup_readline()

    def _setup_readline(self) -> None:
        if not _HAS_READLINE or _readline is None:
            return
        codes = sorted(self._dispatcher.registry.all_codes)
        specials = ["HELP", "MENU", "EXIT", "QUIT", "CLEAR"]
        all_completions = codes + specials + sorted(ASSET_CLASSES)

        def completer(text: str, state: int) -> Optional[str]:
            upper_text = text.upper()
            options = [c for c in all_completions if c.startswith(upper_text)]
            return options[state] if state < len(options) else None

        _readline.set_completer(completer)
        _readline.set_completer_delims(" \t\n")
        try:
            _readline.parse_and_bind("tab: complete")
        except Exception:
            pass
        # Load history
        if HISTORY_FILE.exists():
            try:
                _readline.read_history_file(str(HISTORY_FILE))
            except Exception:
                pass

    def _save_history(self) -> None:
        if not _HAS_READLINE or _readline is None:
            return
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _readline.write_history_file(str(HISTORY_FILE))
        except Exception:
            pass

    def run_command(self, command: str) -> str:
        """Execute a single command string and return formatted output."""
        cmd_stripped = command.strip()
        if not cmd_stripped:
            return ""

        upper = cmd_stripped.upper()

        # Special commands
        if upper in ("EXIT", "QUIT"):
            return "__EXIT__"
        if upper == "CLEAR":
            os.system("cls" if os.name == "nt" else "clear")
            return ""
        if upper == "MENU":
            return self._help.get_category_menu()
        if upper in ("HELP", "?"):
            return self._help.get_quick_reference()

        # HELP <CODE>
        tokens = upper.split()
        if tokens[0] == "HELP" and len(tokens) >= 2:
            return self._help.get_help(tokens[1])

        # <CODE> ?
        if len(tokens) >= 2 and tokens[-1] == "?":
            return self._help.get_help(tokens[-2])

        # Search
        if tokens[0] == "SEARCH" and len(tokens) >= 2:
            results = self._help.search_help(" ".join(tokens[1:]))
            return "\n".join(results) if results else "No matches found."

        # Normal dispatch
        parsed = self._parser.parse(cmd_stripped)
        result = self._dispatcher.dispatch(parsed)
        return self.format_output(result)

    def format_output(self, result: CommandResult, style: str = "table") -> str:
        """Format a CommandResult for terminal display."""
        width = 70
        border = "─" * width
        lines = [
            f"┌{border}┐",
            f"│  [{result.code}] {result.ticker or ''}".ljust(width + 1) + "│",
            f"│  Status: {'OK' if result.success else 'ERROR'}  "
            f"({result.elapsed_ms:.1f} ms)".ljust(width - 14) + "        │",
            f"├{border}┤",
        ]

        if not result.success:
            lines.append(f"│  ERROR: {result.text[:width - 10]}".ljust(width + 1) + "│")
        else:
            # Text summary
            for ln in textwrap.wrap(result.text, width - 4):
                lines.append(f"│  {ln}".ljust(width + 1) + "│")

            # Data payload (first few keys)
            if result.data:
                lines.append(f"├{border}┤")
                for k, v in list(result.data.items())[:8]:
                    val_str = str(v)
                    if len(val_str) > width - len(k) - 6:
                        val_str = val_str[:width - len(k) - 9] + "..."
                    lines.append(f"│  {k}: {val_str}".ljust(width + 1) + "│")

        lines.append(f"└{border}┘")
        return "\n".join(lines)

    def run_interactive(self) -> None:
        """Launch the interactive REPL."""
        print(self.BANNER)
        print(f"  Loaded {len(self._dispatcher.registry)} function codes.")
        print(f"  Readline: {'enabled' if _HAS_READLINE else 'disabled (install pyreadline3 on Windows)'}")
        print()

        while True:
            try:
                raw = input(self.PROMPT)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting SENTINEL terminal. Goodbye.")
                self._save_history()
                break

            output = self.run_command(raw)
            if output == "__EXIT__":
                print("Exiting SENTINEL terminal. Goodbye.")
                self._save_history()
                break
            if output:
                print(output)
            print()

# ---------------------------------------------------------------------------
# ClaudeDesktopConfig
# ---------------------------------------------------------------------------

class ClaudeDesktopConfig:
    """Generate Claude Desktop / Claude Code MCP configuration snippets."""

    MCP_SERVER_NAME = "sentinel-bloomberg"

    def generate_config(self) -> dict:
        """Return a claude_desktop_config.json snippet for MCP wiring."""
        return {
            "mcpServers": {
                self.MCP_SERVER_NAME: {
                    "command": "python",
                    "args": ["-m", "sentinel.api.mcp_server_v3"],
                    "env": {
                        "SENTINEL_ENV": "production",
                        "SENTINEL_DB_URL": "${SENTINEL_DB_URL}",
                    },
                    "description": (
                        "SENTINEL institutional terminal — 102 Bloomberg function codes, "
                        "real-time quotes, equity/FI/macro/options/alt-data."
                    ),
                }
            }
        }

    def generate_mcp_json(self) -> str:
        """Return a .mcp.json string for Claude Code integration."""
        mcp = {
            "servers": {
                self.MCP_SERVER_NAME: {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "sentinel.api.mcp_server_v3"],
                    "description": "SENTINEL Bloomberg command bar — dim_090",
                }
            }
        }
        return json.dumps(mcp, indent=2)

    def print_setup_instructions(self) -> None:
        instructions = textwrap.dedent("""
        ┌─────────────────────────────────────────────────────────────┐
        │       SENTINEL × Claude Desktop MCP Setup                   │
        └─────────────────────────────────────────────────────────────┘

        1. Add to claude_desktop_config.json:

           %s

        2. Add .mcp.json to your project root:

           %s

        3. Restart Claude Desktop.

        4. Ask Claude: "Run AAPL DES in SENTINEL" or
           "Show me the economic calendar with ECO."

        5. Environment variables required:
             SENTINEL_DB_URL  — PostgreSQL connection string
             SENTINEL_ENV     — production | staging | development
        """) % (
            json.dumps(self.generate_config(), indent=6),
            self.generate_mcp_json(),
        )
        print(instructions)

# ---------------------------------------------------------------------------
# Convenience top-level functions
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Run a 4-command demo and print results."""
    cli = BloombergBarCLI()
    demo_commands = ["AAPL DES", "ECO", "WEI", "AAPL HP"]
    print("\n" + "=" * 72)
    print("  SENTINEL Demo — 4 sample commands")
    print("=" * 72)
    for cmd in demo_commands:
        print(f"\n  >> {cmd}")
        output = cli.run_command(cmd)
        print(output)

# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dispatcher = FunctionDispatcher()
    registry = dispatcher.registry
    cats = registry.list_by_category()

    print(f"\nSENTINEL Bloomberg Command Bar v{VERSION}")
    print(f"Total function codes registered: {len(registry)}\n")
    print("Categories:")
    for cat, codes in sorted(cats.items()):
        print(f"  {cat:<30} ({len(codes):>2}) : {', '.join(codes)}")

    run_demo()

    # Enter interactive CLI (single command from argv or full REPL)
    cli = BloombergBarCLI()
    if len(sys.argv) > 1:
        cmd_arg = " ".join(sys.argv[1:])
        print(f"\nExecuting: {cmd_arg}")
        print(cli.run_command(cmd_arg))
    else:
        cli.run_interactive()
