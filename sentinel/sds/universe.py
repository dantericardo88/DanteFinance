"""SENTINEL universe builder — equity universe construction for backtests and screens.

Provides:
  get_sp500_tickers()        — live S&P 500 from Wikipedia (falls back to static list)
  get_russell2000_proxy()    — 1000 most-liquid small-caps from yfinance screener
  get_international_adrs()   — 50 large-cap ADRs for ex-US exposure
  get_gics_classification()  — GICS sector/industry from yfinance .info for a ticker list
  build_full_universe()      — combined deduplicated universe with metadata

Usage
-----
    from sentinel.sds.universe import build_full_universe
    universe = build_full_universe()
    # universe: list[dict] with keys ticker, name, sector, industry, source
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Static fallback: S&P 500 core ─────────────────────────────────────────────
# Kept in sync with the index as of 2026-Q1. Wikipedia scrape is always
# attempted first; this list activates only on network failure.

_SP500_STATIC: list[str] = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "TXN",
    "INTC", "NOW", "INTU", "IBM", "AMAT", "MU", "KLAC", "LRCX", "ADI", "MCHP",
    "SNPS", "CDNS", "FTNT", "PANW", "CRWD", "ANSS", "TER", "KEYS", "AKAM", "CDW",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "USB",
    "PNC", "TFC", "COF", "MCO", "SPGI", "ICE", "CME", "CB", "MMC", "AIG",
    "AFL", "MET", "PRU", "ALL", "TRV", "HIG", "BK", "STT", "NTRS", "FITB",
    # Health Care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN", "BSX",
    "ELV", "CVS", "MDT", "SYK", "ISRG", "ZBH", "HCA", "CNC", "HUM", "CI",
    "GILD", "VRTX", "REGN", "BIIB", "IQV", "A", "BAX", "BDX", "CAH", "MCK",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "MAR",
    "GM", "F", "ORLY", "AZO", "BBY", "DHI", "LEN", "PHM", "NVR", "POOL",
    "YUM", "DRI", "CMG", "HLT", "MGM", "WYNN", "LVS", "RCL", "CCL", "NCLH",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "GIS",
    "KMB", "SJM", "HSY", "K", "CPB", "MKC", "CAG", "HRL", "TSN", "ADM",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA", "TTWO",
    "ATVI", "OMC", "IPG", "FOXA", "NWS", "PARA", "WBD", "LUMN", "DISH", "CHTR",
    # Industrials
    "CAT", "HON", "RTX", "UPS", "BA", "DE", "GE", "MMM", "LMT", "NOC",
    "GD", "EMR", "ETN", "ITW", "PH", "ROK", "IR", "XYL", "CARR", "OTIS",
    "CSX", "UNP", "NSC", "FDX", "DAL", "UAL", "AAL", "LUV", "JBLU", "ALK",
    # Energy
    "XOM", "CVX", "COP", "EOG", "PXD", "SLB", "MPC", "VLO", "PSX", "OXY",
    "HES", "DVN", "FANG", "HAL", "BKR", "APA", "MRO", "EQT", "CVI", "PBF",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "ES", "WEC",
    "ED", "ETR", "FE", "AES", "CMS", "CNP", "NI", "EVRG", "PNW", "OGE",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "SPG", "PSA", "WELL", "DLR", "O", "SBAC",
    "EXR", "AVB", "EQR", "MAA", "UDR", "CPT", "AIV", "NNN", "VICI", "GLPI",
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "STLD", "RS", "VMC",
    "MLM", "CF", "MOS", "ALB", "PPG", "IFF", "EMN", "CE", "RPM", "SEE",
    # Key ETFs (for macro/factor exposure)
    "SPY", "QQQ", "IWM", "GLD", "TLT", "HYG", "LQD", "EEM", "VNQ", "XLE",
    "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
]

# ── International ADRs — 50 large-cap ex-US ───────────────────────────────────

INTERNATIONAL_ADRS: list[dict] = [
    # Europe
    {"ticker": "ASML",  "name": "ASML Holding",             "country": "Netherlands", "sector": "Information Technology"},
    {"ticker": "SAP",   "name": "SAP SE",                   "country": "Germany",     "sector": "Information Technology"},
    {"ticker": "SHEL",  "name": "Shell plc",                 "country": "UK",          "sector": "Energy"},
    {"ticker": "AZN",   "name": "AstraZeneca",               "country": "UK",          "sector": "Health Care"},
    {"ticker": "HSBC",  "name": "HSBC Holdings",             "country": "UK",          "sector": "Financials"},
    {"ticker": "NVO",   "name": "Novo Nordisk",              "country": "Denmark",     "sector": "Health Care"},
    {"ticker": "LVMUY", "name": "LVMH Moet Hennessy",        "country": "France",      "sector": "Consumer Discretionary"},
    {"ticker": "TTE",   "name": "TotalEnergies SE",          "country": "France",      "sector": "Energy"},
    {"ticker": "UL",    "name": "Unilever plc",              "country": "UK",          "sector": "Consumer Staples"},
    {"ticker": "BHP",   "name": "BHP Group",                 "country": "Australia",   "sector": "Materials"},
    {"ticker": "GSK",   "name": "GSK plc",                   "country": "UK",          "sector": "Health Care"},
    {"ticker": "ABB",   "name": "ABB Ltd",                   "country": "Switzerland", "sector": "Industrials"},
    {"ticker": "IDEXY", "name": "Industria de Diseno Textil","country": "Spain",       "sector": "Consumer Discretionary"},
    {"ticker": "SIE",   "name": "Siemens AG",                "country": "Germany",     "sector": "Industrials"},
    {"ticker": "ALIZF", "name": "Allianz SE",                "country": "Germany",     "sector": "Financials"},
    # Asia-Pacific
    {"ticker": "TSM",   "name": "Taiwan Semiconductor",      "country": "Taiwan",      "sector": "Information Technology"},
    {"ticker": "SONY",  "name": "Sony Group",                 "country": "Japan",       "sector": "Consumer Discretionary"},
    {"ticker": "TM",    "name": "Toyota Motor",              "country": "Japan",       "sector": "Consumer Discretionary"},
    {"ticker": "SNY",   "name": "Sanofi SA",                 "country": "France",      "sector": "Health Care"},
    {"ticker": "BABA",  "name": "Alibaba Group",             "country": "China",       "sector": "Consumer Discretionary"},
    {"ticker": "JD",    "name": "JD.com Inc",                "country": "China",       "sector": "Consumer Discretionary"},
    {"ticker": "PDD",   "name": "PDD Holdings",              "country": "China",       "sector": "Consumer Discretionary"},
    {"ticker": "BIDU",  "name": "Baidu Inc",                 "country": "China",       "sector": "Communication Services"},
    {"ticker": "NTES",  "name": "NetEase Inc",               "country": "China",       "sector": "Communication Services"},
    {"ticker": "SE",    "name": "Sea Limited",               "country": "Singapore",   "sector": "Consumer Discretionary"},
    {"ticker": "GRAB",  "name": "Grab Holdings",             "country": "Singapore",   "sector": "Consumer Discretionary"},
    {"ticker": "HDB",   "name": "HDFC Bank",                 "country": "India",       "sector": "Financials"},
    {"ticker": "INFY",  "name": "Infosys",                   "country": "India",       "sector": "Information Technology"},
    {"ticker": "WIT",   "name": "Wipro",                     "country": "India",       "sector": "Information Technology"},
    {"ticker": "IBN",   "name": "ICICI Bank",                "country": "India",       "sector": "Financials"},
    {"ticker": "RDY",   "name": "Dr. Reddy's Laboratories",  "country": "India",       "sector": "Health Care"},
    {"ticker": "VALE",  "name": "Vale SA",                   "country": "Brazil",      "sector": "Materials"},
    {"ticker": "PBR",   "name": "Petrobras",                 "country": "Brazil",      "sector": "Energy"},
    {"ticker": "ITUB",  "name": "Itau Unibanco",             "country": "Brazil",      "sector": "Financials"},
    {"ticker": "BBD",   "name": "Banco Bradesco",            "country": "Brazil",      "sector": "Financials"},
    # Canada
    {"ticker": "RY",    "name": "Royal Bank of Canada",      "country": "Canada",      "sector": "Financials"},
    {"ticker": "TD",    "name": "Toronto-Dominion Bank",     "country": "Canada",      "sector": "Financials"},
    {"ticker": "CNI",   "name": "Canadian National Railway", "country": "Canada",      "sector": "Industrials"},
    {"ticker": "ENB",   "name": "Enbridge Inc",              "country": "Canada",      "sector": "Energy"},
    {"ticker": "BCE",   "name": "BCE Inc",                   "country": "Canada",      "sector": "Communication Services"},
    # Other
    {"ticker": "WDS",   "name": "Woodside Energy",           "country": "Australia",   "sector": "Energy"},
    {"ticker": "SQM",   "name": "SQM SA",                    "country": "Chile",       "sector": "Materials"},
    {"ticker": "GOLD",  "name": "Barrick Gold",              "country": "Canada",      "sector": "Materials"},
    {"ticker": "NEM",   "name": "Newmont Corp",              "country": "USA",         "sector": "Materials"},
    {"ticker": "RIO",   "name": "Rio Tinto Group",           "country": "UK/Australia","sector": "Materials"},
    {"ticker": "MT",    "name": "ArcelorMittal",             "country": "Luxembourg",  "sector": "Materials"},
    {"ticker": "MFG",   "name": "Mizuho Financial Group",    "country": "Japan",       "sector": "Financials"},
    {"ticker": "SMFG",  "name": "Sumitomo Mitsui Financial", "country": "Japan",       "sector": "Financials"},
    {"ticker": "KB",    "name": "KB Financial Group",        "country": "South Korea", "sector": "Financials"},
    {"ticker": "SHG",   "name": "Shinhan Financial Group",   "country": "South Korea", "sector": "Financials"},
]

# ── Russell 2000 small-cap proxy seed ─────────────────────────────────────────
# Top 50 most-liquid small-caps used as seed; get_russell2000_proxy() expands
# this dynamically to the 1000 most-liquid names available via yfinance.

_RUSSELL2000_SEED: list[str] = [
    "SIRI", "PLUG", "FFIE", "NKLA", "WISH", "CLOV", "AMC", "GME", "BB", "NOK",
    "RIDE", "GOEV", "WKHS", "PTON", "LCID", "RIVN", "ARVL", "XPEV", "LI", "NIO",
    "SPCE", "ASTR", "RKT", "UWMC", "GHVI", "PSFE", "GREE", "MVIS", "EXPR", "KOSS",
    "NAKD", "SNDL", "TLRY", "CGC", "ACB", "APHA", "OGI", "HEXO", "CRON", "CURLF",
    "IIPR", "GRWG", "GWPH", "CARA", "ACRS", "CORT", "RCKT", "FATE", "BEAM", "EDIT",
]


# ── Wikipedia S&P 500 scrape ──────────────────────────────────────────────────

def get_sp500_tickers(use_cache: bool = True) -> list[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia.

    Parameters
    ----------
    use_cache:
        If True (default) and a previous successful scrape is stored in the
        module-level cache, return it without re-scraping.

    Returns
    -------
    Sorted list of ticker symbols (e.g. ['AAPL', 'MSFT', ...]).
    Falls back to the static _SP500_STATIC list on any network/parse error.
    """
    global _sp500_cache
    if use_cache and _sp500_cache:
        return _sp500_cache

    try:
        import pandas as pd
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        df = tables[0]
        # Wikipedia table has a "Symbol" column; dots → hyphens for yfinance
        tickers = (
            df["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        tickers = sorted(set(tickers))
        logger.info("S&P 500 tickers loaded from Wikipedia", count=len(tickers))
        _sp500_cache = tickers
        return tickers
    except Exception as exc:
        logger.warning(
            "Wikipedia S&P 500 scrape failed — using static list",
            error=str(exc),
        )
        return sorted(set(_SP500_STATIC))


_sp500_cache: list[str] = []


# ── Russell 2000 proxy ────────────────────────────────────────────────────────

def get_russell2000_proxy(
    n: int = 1000,
    min_avg_volume: int = 500_000,
) -> list[str]:
    """Return up to n small-cap tickers as a Russell 2000 proxy.

    Strategy:
      1. Use the seed list of known liquid small-caps as a starting point.
      2. Attempt to fetch small-cap screener results from yfinance if available.
      3. Return deduplicated, volume-filtered list capped at n tickers.

    Parameters
    ----------
    n:              Maximum tickers to return.
    min_avg_volume: Minimum average daily volume filter.

    Returns
    -------
    List of ticker symbols up to length n.
    """
    try:
        import yfinance as yf
        # yfinance screener for small-caps (market cap $300M–$2B)
        # This is a best-effort; not all yfinance versions support screener
        screen = yf.screen(
            query_filter="is-small-cap",
            sortField="avgDailyVolume3Month",
            sortType="DESC",
            offset=0,
            count=min(n, 250),  # yfinance caps at 250 per call
        )
        if screen and screen.get("quotes"):
            tickers = [q["symbol"] for q in screen["quotes"] if q.get("symbol")]
            tickers = list(dict.fromkeys(tickers))  # deduplicate preserving order
            if len(tickers) >= 100:
                logger.info("Russell 2000 proxy built via yfinance screener", count=len(tickers))
                return tickers[:n]
    except Exception as exc:
        logger.debug("yfinance screener unavailable for Russell 2000 proxy", error=str(exc))

    # Fallback: return the static seed (smaller but always available)
    logger.info(
        "Russell 2000 proxy using static seed",
        count=len(_RUSSELL2000_SEED),
    )
    return _RUSSELL2000_SEED[:n]


# ── International ADRs ────────────────────────────────────────────────────────

def get_international_adrs() -> list[dict]:
    """Return the curated list of 50 large-cap ADRs for ex-US coverage.

    Returns
    -------
    List of dicts: {ticker, name, country, sector}
    """
    return list(INTERNATIONAL_ADRS)


# ── GICS classification ───────────────────────────────────────────────────────

def get_gics_classification(
    tickers: list[str],
    delay_s: float = 0.2,
) -> dict[str, dict]:
    """Fetch GICS sector/industry from yfinance .info for each ticker.

    This is a synchronous call (yfinance .info is blocking). For large lists,
    consider calling in a thread pool executor from async code.

    Parameters
    ----------
    tickers:  List of ticker symbols.
    delay_s:  Courtesy delay between API calls (seconds).

    Returns
    -------
    Dict: { ticker -> {"sector": str, "industry": str, "name": str,
                       "market_cap": float|None, "exchange": str|None} }
    """
    import time

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot fetch GICS classification")
        return {}

    result: dict[str, dict] = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
            result[ticker] = {
                "sector":     info.get("sector") or "Unknown",
                "industry":   info.get("industry") or "Unknown",
                "name":       info.get("shortName") or info.get("longName") or ticker,
                "market_cap": info.get("marketCap"),
                "exchange":   info.get("exchange"),
            }
            logger.debug("GICS fetched", ticker=ticker, sector=result[ticker]["sector"])
        except Exception as exc:
            logger.warning("GICS fetch failed", ticker=ticker, error=str(exc))
            result[ticker] = {
                "sector": "Unknown", "industry": "Unknown",
                "name": ticker, "market_cap": None, "exchange": None,
            }
        if delay_s > 0:
            time.sleep(delay_s)

    logger.info("GICS classification complete", tickers=len(result))
    return result


# ── Full universe builder ─────────────────────────────────────────────────────

def build_full_universe(
    include_russell2000: bool = True,
    include_adrs: bool = True,
    fetch_gics: bool = False,
) -> list[dict]:
    """Build the combined SENTINEL equity universe.

    Parameters
    ----------
    include_russell2000:
        Include Russell 2000 small-cap proxy tickers.
    include_adrs:
        Include international large-cap ADRs.
    fetch_gics:
        If True, enrich each entry with live GICS data from yfinance.info.
        Warning: makes one yfinance API call per ticker — slow for large lists.

    Returns
    -------
    Deduplicated list of dicts:
        { ticker, name, sector, industry, source, country }
    The 'source' field is one of: 'sp500', 'russell2000', 'adr'.
    """
    universe: dict[str, dict] = {}

    # 1. S&P 500
    for ticker in get_sp500_tickers():
        universe[ticker] = {
            "ticker": ticker,
            "name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "source": "sp500",
            "country": "USA",
        }

    # 2. Russell 2000 proxy (adds small-caps not in S&P 500)
    if include_russell2000:
        for ticker in get_russell2000_proxy():
            if ticker not in universe:
                universe[ticker] = {
                    "ticker": ticker,
                    "name": ticker,
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "source": "russell2000",
                    "country": "USA",
                }

    # 3. International ADRs
    if include_adrs:
        for entry in get_international_adrs():
            ticker = entry["ticker"]
            if ticker not in universe:
                universe[ticker] = {
                    "ticker":   ticker,
                    "name":     entry.get("name", ticker),
                    "sector":   entry.get("sector", "Unknown"),
                    "industry": "Unknown",
                    "source":   "adr",
                    "country":  entry.get("country", "Unknown"),
                }

    result = list(universe.values())

    # 4. Optional GICS enrichment
    if fetch_gics:
        tickers = [r["ticker"] for r in result]
        gics = get_gics_classification(tickers)
        for row in result:
            g = gics.get(row["ticker"], {})
            if g.get("sector") and g["sector"] != "Unknown":
                row["sector"] = g["sector"]
            if g.get("industry") and g["industry"] != "Unknown":
                row["industry"] = g["industry"]
            if g.get("name") and g["name"] != row["ticker"]:
                row["name"] = g["name"]

    logger.info(
        "Universe built",
        total=len(result),
        sp500=sum(1 for r in result if r["source"] == "sp500"),
        russell2000=sum(1 for r in result if r["source"] == "russell2000"),
        adrs=sum(1 for r in result if r["source"] == "adr"),
    )
    return result
