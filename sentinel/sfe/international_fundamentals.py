"""International / IFRS fundamentals from free APIs — Dimension #21 (Wave 3).

Data sources:
  - World Bank Open Data API (no key required)
  - IMF World Economic Outlook DataMapper API (no key required)
  - yfinance for international exchange-listed tickers
  - SEC EDGAR EFTS for 20-F foreign private issuer search
  - FRED (St. Louis Fed) for FX rates (no key required for CSV endpoint)

This module extends ``sentinel.sfe.ifrs_fundamentals`` (EDGAR XBRL path) with
market-data-level international coverage: macro country profiles, real-time
international ticker financials from yfinance, 20-F discovery, global peer
comparison, and IFRS vs. US GAAP adjustment notes.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, Literal

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLDBANK_BASE = "https://api.worldbank.org/v2"
IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"
EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
}
_TIMEOUT = 30.0

# yfinance exchange suffixes by friendly market name
MARKET_SUFFIXES: dict[str, str] = {
    "london": ".L",
    "germany": ".DE",
    "france": ".PA",
    "amsterdam": ".AS",
    "switzerland": ".SW",
    "italy": ".MI",
    "spain": ".MC",
    "stockholm": ".ST",
    "hong_kong": ".HK",
    "japan": ".T",
    "australia": ".AX",
    "canada": ".TO",
    "singapore": ".SI",
    "india": ".NS",
    "brazil": ".SA",
    "mexico": ".MX",
    "south_korea": ".KS",
    "taiwan": ".TW",
}

# ISO-2 country codes for World Bank API
COUNTRY_CODES: dict[str, str] = {
    "uk": "GB",
    "germany": "DE",
    "france": "FR",
    "japan": "JP",
    "china": "CN",
    "australia": "AU",
    "canada": "CA",
    "switzerland": "CH",
    "india": "IN",
    "brazil": "BR",
    "south_korea": "KR",
    "singapore": "SG",
    "hong_kong": "HK",
    "taiwan": "TW",
    "spain": "ES",
    "italy": "IT",
    "netherlands": "NL",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "mexico": "MX",
    "south_africa": "ZA",
    "indonesia": "ID",
    "saudi_arabia": "SA",
    "uae": "AE",
    "israel": "IL",
    "poland": "PL",
    "turkey": "TR",
    "argentina": "AR",
    "chile": "CL",
}

# World Bank indicator codes
_WB_INDICATORS: dict[str, str] = {
    "gdp_usd": "NY.GDP.MKTP.CD",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "inflation": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
    "current_account_pct_gdp": "BN.CAB.XOKA.GD.ZS",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "market_cap_pct_gdp": "CM.MKT.LCAP.GD.ZS",
    "fdi_inflows": "BX.KLT.DINV.CD.WD",
    "gross_savings_pct_gdp": "NY.GNS.ICTR.ZS",
    "trade_pct_gdp": "NE.TRD.GNFS.ZS",
}

# FRED FX series: currency → FRED series ID
_FRED_FX_MAP: dict[str, str] = {
    "EUR": "DEXUSEU",   # USD per EUR
    "GBP": "DEXUSUK",   # USD per GBP
    "JPY": "DEXJPUS",   # JPY per USD (inverted)
    "CNY": "DEXCHUS",   # CNY per USD (inverted)
    "CAD": "DEXCAUS",   # CAD per USD (inverted)
    "AUD": "DEXUSAL",   # USD per AUD
    "CHF": "DEXSZUS",   # CHF per USD (inverted)
    "HKD": "DEXHKUS",   # HKD per USD (inverted)
    "SGD": "DEXSIUS",   # SGD per USD (inverted)
    "INR": "DEXINUS",   # INR per USD (inverted)
    "BRL": "DEXBZUS",   # BRL per USD (inverted)
    "MXN": "DEXMXUS",   # MXN per USD (inverted)
    "KRW": "DEXKOUS",   # KRW per USD (inverted)
    "TWD": "DEXTAUS",   # TWD per USD (inverted)
    "NOK": "DEXNOUS",   # NOK per USD (inverted)
    "SEK": "DEXSDUS",   # SEK per USD (inverted)
    "DKK": "DEXDNUS",   # DKK per USD (inverted)
}

# Series where the quote is foreign-per-USD (need to invert to get USD-per-foreign)
_INVERTED_FRED_SERIES = {
    "DEXJPUS", "DEXCHUS", "DEXCAUS", "DEXSZUS", "DEXHKUS",
    "DEXSIUS", "DEXINUS", "DEXBZUS", "DEXMXUS", "DEXKOUS",
    "DEXTAUS", "DEXNOUS", "DEXSDUS", "DEXDNUS",
}

# Key IFRS vs. US GAAP differences to flag per company
IFRS_GAAP_DIFFS: dict[str, str] = {
    "goodwill_amortization": (
        "IFRS (pre-2004) amortized goodwill; modern IFRS 3 requires impairment-only "
        "(same as US GAAP ASC 350) — check acquisition vintage"
    ),
    "development_costs": (
        "IFRS IAS 38 capitalizes qualifying development costs; US GAAP ASC 730 "
        "expenses most R&D immediately — IFRS EBITDA may appear higher"
    ),
    "inventory_lifo": (
        "IFRS IAS 2 prohibits LIFO; US GAAP allows — inventory values and COGS "
        "may not be directly comparable in inflationary environments"
    ),
    "revaluation_model": (
        "IFRS IAS 16/40 allows upward revaluation of PP&E and investment property; "
        "US GAAP uses historical cost only — book value may differ significantly"
    ),
    "lease_accounting": (
        "IFRS 16 and ASC 842 both capitalize most leases, but differ on variable "
        "lease payments and sale-leaseback accounting"
    ),
    "revenue_recognition": (
        "IFRS 15 and ASC 606 broadly converged; differences remain in licenses, "
        "principal-vs-agent, and contract modifications"
    ),
    "financial_instruments": (
        "IFRS 9 vs ASC 326 (CECL): both use expected credit loss models but differ "
        "in staging and measurement — bank comparisons need adjustment"
    ),
    "pension_accounting": (
        "IFRS IAS 19 recognises actuarial gains/losses in OCI immediately; "
        "US GAAP allows corridor amortisation — pension liabilities may appear larger under IFRS"
    ),
    "biological_assets": (
        "IFRS IAS 41 measures biological assets at fair value through P&L; "
        "US GAAP uses historical cost — relevant for agri/forestry companies"
    ),
}

# Predefined sector peer lists (US ticker → international comparables)
_SECTOR_PEERS: dict[str, list[str]] = {
    "technology": ["0700.HK", "ASML.AS", "SAP.DE", "SMSN.IL", "6758.T", "ERIC-B.ST"],
    "semiconductors": ["ASML.AS", "2330.TW", "005930.KS", "6723.T", "IFX.DE"],
    "luxury_consumer": ["MC.PA", "CFR.SW", "BRBY.L", "RMS.PA", "KER.PA"],
    "automotive": ["VOW3.DE", "BMW.DE", "MBG.DE", "7203.T", "005380.KS", "STLA.MI"],
    "banking": ["HSBA.L", "BNP.PA", "DBK.DE", "8306.T", "939.HK", "SAN.MC"],
    "pharma": ["AZN.L", "NVS", "ROG.SW", "SAN.PA", "NOVO-B.CO", "4502.T"],
    "energy": ["SHEL.L", "TTE.PA", "BP.L", "EQNR.OL", "ENI.MI", "PTT.BK"],
    "mining": ["RIO.L", "BHP.AX", "GLEN.L", "AAL.L", "VALE3.SA"],
    "telecom": ["VOD.L", "DTE.DE", "ORA.PA", "TEF.MC", "9984.T", "9432.T"],
    "insurance": ["AV.L", "ZURICH", "MUV2.DE", "G.MI", "AXA.PA"],
    "consumer_staples": ["ULVR.L", "NESN.SW", "DGE.L", "ABF.L", "DANOY"],
    "aerospace": ["AIR.PA", "BA.L", "SAF.PA", "RR.L"],
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class WorldBankIndicator(BaseModel):
    """Time series for a single World Bank indicator and country."""
    country_code: str
    country_name: str
    indicator_id: str
    indicator_name: str
    data: dict[int, Optional[float]]        # year → value
    latest_year: int
    latest_value: Optional[float]
    yoy_change_pct: Optional[float]
    source: str = "World Bank Open Data"


class MacroCountryProfile(BaseModel):
    """Macro-economic snapshot for a single country."""
    country: str
    country_code: str
    as_of_year: int
    gdp_usd: Optional[float] = None
    gdp_per_capita: Optional[float] = None
    gdp_growth_pct: Optional[float] = None
    inflation_pct: Optional[float] = None
    unemployment_pct: Optional[float] = None
    current_account_pct_gdp: Optional[float] = None
    gross_savings_pct_gdp: Optional[float] = None
    trade_pct_gdp: Optional[float] = None
    fx_rate_vs_usd: Optional[float] = None     # local-currency units per USD
    market_cap_pct_gdp: Optional[float] = None
    data_warnings: list[str] = Field(default_factory=list)


class InternationalFinancials(BaseModel):
    """Financial snapshot for a single international listed company."""
    ticker: str
    company_name: str
    exchange: str
    country: str
    currency: str
    reporting_standard: Literal["IFRS", "GAAP", "Local GAAP", "Unknown"] = "Unknown"
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap_local: Optional[float] = None
    market_cap_usd: Optional[float] = None
    # Income statement (trailing twelve months or latest annual)
    revenue: Optional[float] = None        # reporting currency
    revenue_usd: Optional[float] = None
    gross_profit: Optional[float] = None
    ebitda: Optional[float] = None
    ebitda_usd: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    net_income_usd: Optional[float] = None
    eps: Optional[float] = None
    # Balance sheet
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    net_debt: Optional[float] = None
    # Cash flow
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None
    free_cash_flow: Optional[float] = None
    free_cash_flow_usd: Optional[float] = None
    # Currency-neutral ratios
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_book: Optional[float] = None
    price_sales: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    debt_to_equity: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    fcf_yield: Optional[float] = None
    dividend_yield: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    # Metadata
    fiscal_year_end: Optional[str] = None
    data_date: Optional[str] = None
    usd_fx_rate: Optional[float] = None    # local per USD
    ifrs_gaap_notes: list[str] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)


class Form20F(BaseModel):
    """Summary of a single 20-F filing as found in EDGAR EFTS."""
    company_name: str
    cik: str
    country_of_incorporation: Optional[str] = None
    filed_date: date
    period_date: Optional[date] = None
    accession_number: str
    reporting_standard: str = "IFRS"
    us_listing_exchange: Optional[str] = None
    form_url: Optional[str] = None


class ComparablePeerSet(BaseModel):
    """Cross-border peer comparison for a given company."""
    base_ticker: str
    base_company: str
    sector: str
    peers: list[InternationalFinancials]
    # Aggregate peer stats
    avg_pe: Optional[float] = None
    median_pe: Optional[float] = None
    avg_ev_ebitda: Optional[float] = None
    median_ev_ebitda: Optional[float] = None
    avg_gross_margin: Optional[float] = None
    avg_roe: Optional[float] = None
    # Base vs. peers
    base_pe: Optional[float] = None
    base_ev_ebitda: Optional[float] = None
    base_pe_vs_peers_pct: Optional[float] = None      # premium (+) / discount (-)
    base_ev_ebitda_vs_peers_pct: Optional[float] = None
    analysis_date: str = Field(default_factory=lambda: date.today().isoformat())
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    """Coerce a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100.0, 2)


def _df_val(df: pd.DataFrame, row_key: str, col_idx: int = 0) -> Optional[float]:
    """Safely extract a value from a yfinance financials DataFrame (items as index)."""
    if df is None or df.empty:
        return None
    # yfinance index items vary; do case-insensitive partial match
    idx_lower = {k.lower(): k for k in df.index}
    needle = row_key.lower()
    matched_key = idx_lower.get(needle)
    if matched_key is None:
        # Try partial match
        for k_low, k_orig in idx_lower.items():
            if needle in k_low:
                matched_key = k_orig
                break
    if matched_key is None:
        return None
    cols = list(df.columns)
    if col_idx >= len(cols):
        return None
    return _safe_float(df.loc[matched_key, cols[col_idx]])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InternationalFundamentals:
    """
    Fetches international macro and company financial data from free sources.

    Instantiate once and reuse; maintains an FX rate cache per session.
    """

    def __init__(self, timeout: float = _TIMEOUT):
        self._timeout = timeout
        self._fx_cache: dict[str, float] = {}   # currency.upper() → USD per 1 unit
        self._http_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # HTTP session management
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        """Return a shared httpx client (created lazily)."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers=_HEADERS,
            )
        return self._http_client

    async def aclose(self) -> None:
        """Close the underlying HTTP session."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # ------------------------------------------------------------------
    # FX rate resolution (FRED CSV endpoint — no API key)
    # ------------------------------------------------------------------

    def _get_fred_fx_series(self, currency: str) -> Optional[str]:
        """Return the FRED series ID for the given ISO-4217 currency code."""
        return _FRED_FX_MAP.get(currency.upper())

    async def get_fx_rate(self, currency: str) -> float:
        """
        Return USD per 1 unit of ``currency``.

        Uses FRED CSV endpoint with a 30-day lookback to get the most recent
        business-day close.  Results are cached in ``self._fx_cache``.
        Returns 1.0 for USD or any currency we cannot resolve.
        """
        ccy = currency.upper()
        if ccy == "USD":
            return 1.0
        if ccy in self._fx_cache:
            return self._fx_cache[ccy]

        series_id = self._get_fred_fx_series(ccy)
        if series_id is None:
            logger.warning("intl_fx.no_fred_series", currency=ccy)
            return 1.0

        start_dt = (datetime.utcnow() - timedelta(days=45)).strftime("%Y-%m-%d")
        end_dt = datetime.utcnow().strftime("%Y-%m-%d")
        params = {"id": series_id, "vintage_date": end_dt, "sdate": start_dt}
        try:
            resp = await self._client().get(FRED_CSV, params=params)
            resp.raise_for_status()
            lines = [ln for ln in resp.text.strip().splitlines() if ln and not ln.startswith("DATE")]
            if not lines:
                logger.warning("intl_fx.empty_response", currency=ccy, series=series_id)
                return 1.0
            # Last non-empty line
            last_value_str = lines[-1].split(",")[-1].strip()
            raw_rate = float(last_value_str)
            # Invert if the series is quoted as foreign-per-USD
            usd_per_unit = (1.0 / raw_rate) if series_id in _INVERTED_FRED_SERIES else raw_rate
            self._fx_cache[ccy] = round(usd_per_unit, 6)
            logger.info("intl_fx.resolved", currency=ccy, series=series_id, usd_per_unit=usd_per_unit)
            return self._fx_cache[ccy]
        except Exception as exc:
            logger.warning("intl_fx.fetch_failed", currency=ccy, series=series_id, error=str(exc))
            return 1.0

    # ------------------------------------------------------------------
    # IFRS / GAAP difference flagging
    # ------------------------------------------------------------------

    def flag_ifrs_gaap_differences(
        self, financials: InternationalFinancials
    ) -> list[str]:
        """
        Return a list of IFRS vs. US GAAP adjustment notes relevant to this
        company based on its sector.  Always returns general notes plus
        sector-specific ones.
        """
        notes: list[str] = []
        sector = (financials.sector or "").lower()
        reporting = financials.reporting_standard

        if reporting not in ("IFRS", "Unknown"):
            return notes  # GAAP company: no cross-standard notes needed

        # Universal notes for any IFRS filer
        for key in ("development_costs", "lease_accounting", "revenue_recognition"):
            notes.append(IFRS_GAAP_DIFFS[key])

        # Sector-specific
        if any(k in sector for k in ("bank", "financ", "insur", "credit")):
            notes.append(IFRS_GAAP_DIFFS["financial_instruments"])
            notes.append(IFRS_GAAP_DIFFS["pension_accounting"])
        if any(k in sector for k in ("manufactur", "industri", "auto", "aerospace")):
            notes.append(IFRS_GAAP_DIFFS["inventory_lifo"])
            notes.append(IFRS_GAAP_DIFFS["revaluation_model"])
        if any(k in sector for k in ("pharma", "biotech", "health", "medic")):
            notes.append(IFRS_GAAP_DIFFS["development_costs"])
        if any(k in sector for k in ("agri", "forest", "farm", "fish")):
            notes.append(IFRS_GAAP_DIFFS["biological_assets"])
        if any(k in sector for k in ("real estate", "reit", "property")):
            notes.append(IFRS_GAAP_DIFFS["revaluation_model"])

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_notes: list[str] = []
        for n in notes:
            if n not in seen:
                seen.add(n)
                unique_notes.append(n)
        return unique_notes

    # ------------------------------------------------------------------
    # yfinance ticker financials
    # ------------------------------------------------------------------

    async def get_ticker_financials(
        self, ticker: str, convert_to_usd: bool = True
    ) -> InternationalFinancials:
        """
        Fetch financial data for an international ticker via yfinance.

        ``ticker`` should include the exchange suffix, e.g. ``"0700.HK"``,
        ``"ASML.AS"``, ``"SAP.DE"``.  US tickers (no suffix) work too.

        All monetary fields are in the reporting currency; ``*_usd`` fields
        are populated when ``convert_to_usd=True`` and an FX rate is found.
        """
        warnings: list[str] = []
        ticker_upper = ticker.upper()

        # Run yfinance in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            yf_ticker = await loop.run_in_executor(None, lambda: yf.Ticker(ticker_upper))
            info = await loop.run_in_executor(None, lambda: yf_ticker.info)
        except Exception as exc:
            logger.warning("intl.yf_info_failed", ticker=ticker_upper, error=str(exc))
            return InternationalFinancials(
                ticker=ticker_upper,
                company_name=ticker_upper,
                exchange="Unknown",
                country="Unknown",
                currency="USD",
                data_warnings=[f"yfinance info fetch failed: {exc}"],
            )

        currency = info.get("currency", "USD") or "USD"
        company_name = info.get("longName") or info.get("shortName") or ticker_upper
        exchange = info.get("exchange") or info.get("market") or "Unknown"
        country = info.get("country") or "Unknown"
        sector = info.get("sector")
        industry = info.get("industry")
        fiscal_year_end = info.get("lastFiscalYearEnd")
        if fiscal_year_end:
            try:
                fiscal_year_end = datetime.utcfromtimestamp(fiscal_year_end).strftime("%Y-%m-%d")
            except Exception:
                fiscal_year_end = str(fiscal_year_end)

        # Determine reporting standard from country + exchange
        gaap_countries = {"United States", "US"}
        if country in gaap_countries or not any(c in ticker_upper for c in [".", "-"]):
            reporting_standard: Literal["IFRS", "GAAP", "Local GAAP", "Unknown"] = "GAAP"
        elif country in {"China", "Japan"}:
            reporting_standard = "Local GAAP"
        else:
            reporting_standard = "IFRS"

        # Market cap
        market_cap_local = _safe_float(info.get("marketCap"))

        # Fetch financial statements (blocking — run in executor)
        try:
            fin_df = await loop.run_in_executor(None, lambda: yf_ticker.financials)       # annual income stmt
            bs_df = await loop.run_in_executor(None, lambda: yf_ticker.balance_sheet)     # annual balance sheet
            cf_df = await loop.run_in_executor(None, lambda: yf_ticker.cashflow)          # annual cash flow
        except Exception as exc:
            warnings.append(f"Statement fetch error: {exc}")
            fin_df = bs_df = cf_df = pd.DataFrame()

        # Income statement
        revenue = _df_val(fin_df, "Total Revenue")
        revenue_prev = _df_val(fin_df, "Total Revenue", col_idx=1)
        gross_profit = _df_val(fin_df, "Gross Profit")
        ebitda = _df_val(fin_df, "EBITDA")
        operating_income = _df_val(fin_df, "Operating Income")
        net_income = _df_val(fin_df, "Net Income")
        eps = _safe_float(info.get("trailingEps"))

        # If EBITDA not in financials, try info dict
        if ebitda is None:
            ebitda = _safe_float(info.get("ebitda"))

        # Balance sheet
        total_assets = _df_val(bs_df, "Total Assets")
        total_equity = _df_val(bs_df, "Total Stockholder Equity") or _df_val(bs_df, "Stockholders Equity")
        if total_equity is None:
            total_equity = _df_val(bs_df, "Total Equity Gross Minority Interest")
        total_debt = _df_val(bs_df, "Long Term Debt") or 0.0
        short_debt = _df_val(bs_df, "Current Debt") or _df_val(bs_df, "Short Long Term Debt") or 0.0
        if total_debt is not None and short_debt is not None:
            total_debt = total_debt + short_debt
        cash = _df_val(bs_df, "Cash And Cash Equivalents") or _df_val(bs_df, "Cash")
        net_debt = (total_debt - cash) if (total_debt is not None and cash is not None) else None

        # Cash flow
        operating_cf = _df_val(cf_df, "Total Cash From Operating Activities") or _df_val(cf_df, "Operating Cash Flow")
        capex = _df_val(cf_df, "Capital Expenditures") or _df_val(cf_df, "Purchase Of Plant Property And Equipment")
        if capex is None:
            capex = _safe_float(info.get("capitalExpenditures"))
        # FCF = operating CF - capex (capex is usually negative in yf; handle both signs)
        if operating_cf is not None and capex is not None:
            free_cash_flow = operating_cf - abs(capex)
        elif operating_cf is not None:
            free_cash_flow = operating_cf
        else:
            free_cash_flow = _safe_float(info.get("freeCashflow"))

        # FX conversion
        fx_rate = 1.0
        if convert_to_usd and currency != "USD":
            fx_rate = await self.get_fx_rate(currency)

        def to_usd(val: Optional[float]) -> Optional[float]:
            if val is None or fx_rate == 1.0:
                return val
            return _safe_float(val * fx_rate)

        # Ratios (use info dict for market-based ratios; compute fundamentals ourselves)
        pe_ratio = _safe_float(info.get("trailingPE"))
        forward_pe = _safe_float(info.get("forwardPE"))
        ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
        price_book = _safe_float(info.get("priceToBook"))
        price_sales = _safe_float(info.get("priceToSalesTrailing12Months"))
        dividend_yield = _safe_float(info.get("dividendYield"))

        gross_margin = _safe_float(info.get("grossMargins"))
        operating_margin = _safe_float(info.get("operatingMargins"))
        net_margin = _safe_float(info.get("profitMargins"))
        revenue_growth = _safe_float(info.get("revenueGrowth"))  # TTM YoY

        # Compute ratios we can't get directly from info
        roe = _safe_float(info.get("returnOnEquity"))
        roa = _safe_float(info.get("returnOnAssets"))

        roic: Optional[float] = None
        if net_income is not None and total_equity is not None and total_debt is not None:
            invested_capital = total_equity + total_debt
            if invested_capital and invested_capital != 0:
                roic = round(net_income / invested_capital * 100.0, 2)

        debt_to_equity: Optional[float] = None
        if total_debt is not None and total_equity is not None and total_equity != 0:
            debt_to_equity = round(total_debt / total_equity, 4)

        net_debt_to_ebitda: Optional[float] = None
        if net_debt is not None and ebitda is not None and ebitda != 0:
            net_debt_to_ebitda = round(net_debt / ebitda, 2)

        fcf_yield: Optional[float] = None
        if free_cash_flow is not None and market_cap_local is not None and market_cap_local != 0:
            fcf_yield = round(free_cash_flow / market_cap_local * 100.0, 2)

        revenue_growth_yoy = revenue_growth  # already as decimal from yf
        if revenue_growth_yoy is None and revenue is not None and revenue_prev is not None:
            revenue_growth_yoy = _pct_change(revenue, revenue_prev)
            if revenue_growth_yoy is not None:
                revenue_growth_yoy = revenue_growth_yoy / 100.0  # normalise to decimal

        result = InternationalFinancials(
            ticker=ticker_upper,
            company_name=company_name,
            exchange=exchange,
            country=country,
            currency=currency,
            reporting_standard=reporting_standard,
            sector=sector,
            industry=industry,
            market_cap_local=market_cap_local,
            market_cap_usd=to_usd(market_cap_local),
            revenue=revenue,
            revenue_usd=to_usd(revenue),
            gross_profit=gross_profit,
            ebitda=ebitda,
            ebitda_usd=to_usd(ebitda),
            operating_income=operating_income,
            net_income=net_income,
            net_income_usd=to_usd(net_income),
            eps=eps,
            total_assets=total_assets,
            total_equity=total_equity,
            total_debt=total_debt,
            cash=cash,
            net_debt=net_debt,
            operating_cash_flow=operating_cf,
            capex=capex,
            free_cash_flow=free_cash_flow,
            free_cash_flow_usd=to_usd(free_cash_flow),
            pe_ratio=pe_ratio,
            forward_pe=forward_pe,
            ev_ebitda=ev_ebitda,
            price_book=price_book,
            price_sales=price_sales,
            roe=roe,
            roa=roa,
            roic=roic,
            debt_to_equity=debt_to_equity,
            net_debt_to_ebitda=net_debt_to_ebitda,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            fcf_yield=fcf_yield,
            dividend_yield=dividend_yield,
            revenue_growth_yoy=revenue_growth_yoy,
            fiscal_year_end=fiscal_year_end,
            data_date=date.today().isoformat(),
            usd_fx_rate=(1.0 / fx_rate) if fx_rate and fx_rate != 0 else None,
            data_warnings=warnings,
        )

        # Add IFRS/GAAP notes
        result.ifrs_gaap_notes = self.flag_ifrs_gaap_differences(result)
        logger.info(
            "intl.ticker_done",
            ticker=ticker_upper,
            currency=currency,
            reporting_standard=reporting_standard,
            revenue=revenue,
            pe=pe_ratio,
        )
        return result

    # ------------------------------------------------------------------
    # World Bank API
    # ------------------------------------------------------------------

    async def _fetch_worldbank(
        self, country_code: str, indicator: str, years: int = 10
    ) -> dict:
        """
        Fetch a World Bank time-series for one country/indicator.

        Returns a dict mapping ``int(year)`` → ``float|None``.
        The World Bank response is: ``[{page info}, [{countryiso3code, date, value, ...}]]``
        """
        end_year = datetime.utcnow().year
        start_year = end_year - years
        url = (
            f"{WORLDBANK_BASE}/country/{country_code}/indicator/{indicator}"
            f"?format=json&date={start_year}:{end_year}&per_page=100"
        )
        try:
            resp = await self._client().get(url)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning(
                "intl.wb_fetch_failed",
                country=country_code,
                indicator=indicator,
                error=str(exc),
            )
            return {}

        # Response: [meta_dict, [data_items]] — data_items can be null
        if not isinstance(payload, list) or len(payload) < 2:
            return {}
        records = payload[1]
        if not records:
            return {}

        result: dict[int, Optional[float]] = {}
        for rec in records:
            try:
                yr = int(rec["date"])
                val = rec.get("value")
                result[yr] = float(val) if val is not None else None
            except (KeyError, TypeError, ValueError):
                continue
        return result

    async def get_world_bank_series(
        self, indicator_id: str, countries: list[str], years: int = 10
    ) -> list[WorldBankIndicator]:
        """
        Fetch a World Bank indicator for multiple countries in parallel.

        ``countries`` can be friendly keys from ``COUNTRY_CODES`` (e.g. ``"germany"``)
        or ISO-2 codes directly (e.g. ``"DE"``).
        """
        # Resolve friendly names → ISO-2 codes
        tasks = []
        resolved: list[tuple[str, str]] = []  # (friendly_name, iso2_code)
        for c in countries:
            iso2 = COUNTRY_CODES.get(c.lower()) or c.upper()
            resolved.append((c, iso2))
            tasks.append(self._fetch_worldbank(iso2, indicator_id, years))

        all_data = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[WorldBankIndicator] = []
        for (friendly, iso2), data in zip(resolved, all_data):
            if isinstance(data, Exception) or not data:
                logger.warning("intl.wb_series_empty", country=friendly, indicator=indicator_id)
                continue
            sorted_years = sorted((yr for yr, v in data.items() if v is not None), reverse=True)
            if not sorted_years:
                continue
            latest_year = sorted_years[0]
            latest_value = data.get(latest_year)
            prev_value = data.get(latest_year - 1) if (latest_year - 1) in data else None
            yoy_change = _pct_change(latest_value, prev_value)

            results.append(WorldBankIndicator(
                country_code=iso2,
                country_name=friendly.replace("_", " ").title(),
                indicator_id=indicator_id,
                indicator_name=indicator_id,   # WB name would require an extra lookup
                data=data,
                latest_year=latest_year,
                latest_value=latest_value,
                yoy_change_pct=yoy_change,
            ))
        return results

    # ------------------------------------------------------------------
    # Macro country profile (World Bank + FRED FX)
    # ------------------------------------------------------------------

    async def get_macro_country_profile(self, country: str) -> MacroCountryProfile:
        """
        Build a macro snapshot for a single country from World Bank + FRED FX.

        ``country`` is a friendly key from ``COUNTRY_CODES`` or an ISO-2 code.
        """
        friendly = country.lower()
        iso2 = COUNTRY_CODES.get(friendly) or country.upper()
        warnings: list[str] = []

        # Fetch all indicators in parallel
        indicator_keys = list(_WB_INDICATORS.keys())
        indicator_ids = list(_WB_INDICATORS.values())
        fetch_tasks = [self._fetch_worldbank(iso2, ind, years=6) for ind in indicator_ids]
        all_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        def _latest(data) -> Optional[float]:
            """Return the most recent non-None value from a year→value dict."""
            if isinstance(data, Exception) or not data:
                return None
            for yr in sorted(data.keys(), reverse=True):
                if data[yr] is not None:
                    return data[yr]
            return None

        def _latest_year(data) -> int:
            if isinstance(data, Exception) or not data:
                return datetime.utcnow().year - 1
            non_null = [yr for yr, v in data.items() if v is not None]
            return max(non_null) if non_null else datetime.utcnow().year - 1

        indicator_data: dict[str, dict] = {}
        for key, result in zip(indicator_keys, all_results):
            if isinstance(result, Exception):
                warnings.append(f"World Bank fetch failed for {key}: {result}")
                indicator_data[key] = {}
            else:
                indicator_data[key] = result

        # FX rate — need currency code; map country to currency
        _COUNTRY_CURRENCY: dict[str, str] = {
            "GB": "GBP", "DE": "EUR", "FR": "EUR", "JP": "JPY", "CN": "CNY",
            "AU": "AUD", "CA": "CAD", "CH": "CHF", "IN": "INR", "BR": "BRL",
            "KR": "KRW", "SG": "SGD", "HK": "HKD", "TW": "TWD", "ES": "EUR",
            "IT": "EUR", "NL": "EUR", "SE": "SEK", "NO": "NOK", "DK": "DKK",
            "MX": "MXN", "ZA": "ZAR", "ID": "IDR", "SA": "SAR", "AE": "AED",
            "IL": "ILS", "PL": "PLN", "TR": "TRY", "AR": "ARS", "CL": "CLP",
        }
        currency = _COUNTRY_CURRENCY.get(iso2)
        fx_rate: Optional[float] = None
        if currency:
            try:
                usd_per_unit = await self.get_fx_rate(currency)
                fx_rate = round(1.0 / usd_per_unit, 4) if usd_per_unit else None
            except Exception as exc:
                warnings.append(f"FX rate unavailable for {currency}: {exc}")

        as_of_year = _latest_year(indicator_data.get("gdp_usd", {}))

        return MacroCountryProfile(
            country=friendly.replace("_", " ").title(),
            country_code=iso2,
            as_of_year=as_of_year,
            gdp_usd=_latest(indicator_data.get("gdp_usd")),
            gdp_per_capita=_latest(indicator_data.get("gdp_per_capita")),
            gdp_growth_pct=_latest(indicator_data.get("gdp_growth")),
            inflation_pct=_latest(indicator_data.get("inflation")),
            unemployment_pct=_latest(indicator_data.get("unemployment")),
            current_account_pct_gdp=_latest(indicator_data.get("current_account_pct_gdp")),
            gross_savings_pct_gdp=_latest(indicator_data.get("gross_savings_pct_gdp")),
            trade_pct_gdp=_latest(indicator_data.get("trade_pct_gdp")),
            fx_rate_vs_usd=fx_rate,
            market_cap_pct_gdp=_latest(indicator_data.get("market_cap_pct_gdp")),
            data_warnings=warnings,
        )

    async def get_country_profiles(
        self, countries: Optional[list[str]] = None
    ) -> list[MacroCountryProfile]:
        """
        Fetch macro profiles for multiple countries in parallel.

        Defaults to all countries in ``COUNTRY_CODES`` if none specified.
        """
        target = countries if countries is not None else list(COUNTRY_CODES.keys())
        tasks = [self.get_macro_country_profile(c) for c in target]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        profiles: list[MacroCountryProfile] = []
        for country, res in zip(target, results):
            if isinstance(res, Exception):
                logger.warning("intl.country_profile_failed", country=country, error=str(res))
            else:
                profiles.append(res)
        return profiles

    # ------------------------------------------------------------------
    # EDGAR 20-F filing discovery
    # ------------------------------------------------------------------

    async def get_recent_20f_filings(
        self, days_back: int = 90, limit: int = 50
    ) -> list[Form20F]:
        """
        Search EDGAR EFTS for recent 20-F filings by foreign private issuers.

        Returns filings sorted by ``filed_date`` descending.
        """
        start_dt = (date.today() - timedelta(days=days_back)).isoformat()
        end_dt = date.today().isoformat()
        params = {
            "forms": "20-F",
            "dateRange": "custom",
            "startdt": start_dt,
            "enddt": end_dt,
            "hits.hits.total.value": limit,
        }
        try:
            resp = await self._client().get(EDGAR_EFTS, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("intl.edgar_20f_failed", error=str(exc))
            return []

        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            # Try alternative response shape
            hits = payload.get("hits", [])

        filings: list[Form20F] = []
        for hit in hits[:limit]:
            src = hit.get("_source", {})
            try:
                filed_str = src.get("file_date") or src.get("period_of_report", "")
                period_str = src.get("period_of_report")

                filed_date = date.fromisoformat(filed_str) if filed_str else date.today()
                period_date = date.fromisoformat(period_str) if period_str else None

                # Company name: may be in display_names or entity_name
                display_names = src.get("display_names", [])
                company_name = (
                    display_names[0].get("name") if display_names and isinstance(display_names[0], dict)
                    else (display_names[0] if display_names else src.get("entity_name", "Unknown"))
                )

                cik = str(src.get("entity_id") or hit.get("_id", "")).zfill(10)
                accn = src.get("accession_no") or hit.get("_id", "")
                exchange = src.get("biz_location") or None

                filings.append(Form20F(
                    company_name=str(company_name),
                    cik=cik,
                    country_of_incorporation=src.get("inc_states"),
                    filed_date=filed_date,
                    period_date=period_date,
                    accession_number=accn,
                    reporting_standard="IFRS",
                    us_listing_exchange=exchange,
                    form_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=20-F",
                ))
            except Exception as exc:
                logger.warning("intl.20f_parse_error", hit_id=hit.get("_id"), error=str(exc))
                continue

        # Sort descending by filed date
        filings.sort(key=lambda f: f.filed_date, reverse=True)
        logger.info("intl.20f_found", count=len(filings), days_back=days_back)
        return filings

    # ------------------------------------------------------------------
    # International peer comparison
    # ------------------------------------------------------------------

    async def get_international_peers(
        self, us_ticker: str, sector: str
    ) -> ComparablePeerSet:
        """
        Given a US ticker and its sector, fetch international peers from the
        predefined ``_SECTOR_PEERS`` map and build a cross-border comparison.

        ``sector`` should match one of the keys in ``_SECTOR_PEERS``, e.g.
        ``"technology"``, ``"banking"``, ``"pharma"``.
        """
        sector_key = sector.lower().replace(" ", "_")
        peer_tickers = _SECTOR_PEERS.get(sector_key, [])

        # Fetch base (US) ticker
        base_task = self.get_ticker_financials(us_ticker, convert_to_usd=True)
        # Fetch all international peers in parallel
        peer_tasks = [self.get_ticker_financials(t, convert_to_usd=True) for t in peer_tickers]

        all_tasks = [base_task] + peer_tasks
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        base_result = all_results[0]
        if isinstance(base_result, Exception):
            base_fin = InternationalFinancials(
                ticker=us_ticker.upper(),
                company_name=us_ticker.upper(),
                exchange="Unknown",
                country="US",
                currency="USD",
                data_warnings=[str(base_result)],
            )
        else:
            base_fin = base_result

        peers: list[InternationalFinancials] = []
        for t, res in zip(peer_tickers, all_results[1:]):
            if isinstance(res, Exception):
                logger.warning("intl.peer_failed", ticker=t, error=str(res))
            else:
                peers.append(res)

        # Aggregate peer stats
        peer_pes = [p.pe_ratio for p in peers if p.pe_ratio is not None and p.pe_ratio > 0]
        peer_ev_ebitdas = [p.ev_ebitda for p in peers if p.ev_ebitda is not None and p.ev_ebitda > 0]
        peer_gross_margins = [p.gross_margin for p in peers if p.gross_margin is not None]
        peer_roes = [p.roe for p in peers if p.roe is not None]

        avg_pe = round(float(np.mean(peer_pes)), 2) if peer_pes else None
        median_pe = round(float(np.median(peer_pes)), 2) if peer_pes else None
        avg_ev_ebitda = round(float(np.mean(peer_ev_ebitdas)), 2) if peer_ev_ebitdas else None
        median_ev_ebitda = round(float(np.median(peer_ev_ebitdas)), 2) if peer_ev_ebitdas else None
        avg_gross_margin = round(float(np.mean(peer_gross_margins)), 4) if peer_gross_margins else None
        avg_roe = round(float(np.mean(peer_roes)), 4) if peer_roes else None

        base_pe = base_fin.pe_ratio
        base_ev_ebitda_val = base_fin.ev_ebitda
        base_pe_vs_peers = _pct_change(base_pe, avg_pe)
        base_ev_vs_peers = _pct_change(base_ev_ebitda_val, avg_ev_ebitda)

        notes = [
            f"Sector: {sector}; {len(peers)} international peers fetched from yfinance",
            "IFRS vs. GAAP differences may affect direct ratio comparisons — see per-peer ifrs_gaap_notes",
        ]
        if base_pe_vs_peers is not None:
            direction = "premium" if base_pe_vs_peers > 0 else "discount"
            notes.append(
                f"{us_ticker} trades at {abs(base_pe_vs_peers):.1f}% {direction} to peer avg P/E"
            )

        return ComparablePeerSet(
            base_ticker=us_ticker.upper(),
            base_company=base_fin.company_name,
            sector=sector,
            peers=peers,
            avg_pe=avg_pe,
            median_pe=median_pe,
            avg_ev_ebitda=avg_ev_ebitda,
            median_ev_ebitda=median_ev_ebitda,
            avg_gross_margin=avg_gross_margin,
            avg_roe=avg_roe,
            base_pe=base_pe,
            base_ev_ebitda=base_ev_ebitda_val,
            base_pe_vs_peers_pct=base_pe_vs_peers,
            base_ev_ebitda_vs_peers_pct=base_ev_vs_peers,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Global valuation comparison
    # ------------------------------------------------------------------

    async def compare_global_valuations(self, ticker: str) -> dict:
        """
        High-level comparison of a given ticker against its global peers.

        Fetches the ticker's sector from yfinance info, looks up the closest
        ``_SECTOR_PEERS`` bucket, then returns a structured summary dict with
        premium/discount analysis and IFRS adjustment notes.
        """
        loop = asyncio.get_event_loop()
        try:
            yf_ticker = await loop.run_in_executor(None, lambda: yf.Ticker(ticker.upper()))
            info = await loop.run_in_executor(None, lambda: yf_ticker.info)
            sector = info.get("sector", "technology")
        except Exception:
            sector = "technology"

        # Map yfinance sector to our peer bucket key
        _SECTOR_MAP = {
            "Technology": "technology",
            "Financial Services": "banking",
            "Healthcare": "pharma",
            "Consumer Cyclical": "luxury_consumer",
            "Consumer Defensive": "consumer_staples",
            "Industrials": "aerospace",
            "Basic Materials": "mining",
            "Energy": "energy",
            "Communication Services": "telecom",
            "Utilities": "energy",
            "Real Estate": "banking",
        }
        bucket = _SECTOR_MAP.get(sector, sector.lower().replace(" ", "_"))

        peer_set = await self.get_international_peers(ticker, bucket)

        return {
            "ticker": ticker.upper(),
            "sector": sector,
            "peer_bucket": bucket,
            "base_pe": peer_set.base_pe,
            "base_ev_ebitda": peer_set.base_ev_ebitda,
            "peer_avg_pe": peer_set.avg_pe,
            "peer_median_pe": peer_set.median_pe,
            "peer_avg_ev_ebitda": peer_set.avg_ev_ebitda,
            "pe_premium_discount_pct": peer_set.base_pe_vs_peers_pct,
            "ev_ebitda_premium_discount_pct": peer_set.base_ev_ebitda_vs_peers_pct,
            "peers_count": len(peer_set.peers),
            "peers": [
                {
                    "ticker": p.ticker,
                    "name": p.company_name,
                    "country": p.country,
                    "currency": p.currency,
                    "reporting_standard": p.reporting_standard,
                    "pe": p.pe_ratio,
                    "ev_ebitda": p.ev_ebitda,
                    "gross_margin": p.gross_margin,
                    "roe": p.roe,
                    "market_cap_usd": p.market_cap_usd,
                }
                for p in peer_set.peers
            ],
            "notes": peer_set.notes,
            "analysis_date": date.today().isoformat(),
        }

    # ------------------------------------------------------------------
    # IMF WEO data (supplementary GDP growth)
    # ------------------------------------------------------------------

    async def get_imf_gdp_growth(
        self, countries: list[str], indicator: str = "NGDP_RPCH"
    ) -> dict[str, Optional[float]]:
        """
        Fetch IMF DataMapper GDP growth for a list of ISO-2 country codes.

        Returns dict mapping country_code → latest_value (percent).
        IMF DataMapper URL: ``/v1/{indicator}/{country1}/{country2}``
        """
        iso2_list = [COUNTRY_CODES.get(c.lower(), c.upper()) for c in countries]
        country_str = "/".join(iso2_list)
        url = f"{IMF_BASE}/{indicator}/{country_str}"
        try:
            resp = await self._client().get(url)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("intl.imf_gdp_failed", error=str(exc))
            return {}

        # IMF DataMapper response: {"values": {indicator: {country: {year: value}}}}
        values = payload.get("values", {}).get(indicator, {})
        result: dict[str, Optional[float]] = {}
        for iso2, year_data in values.items():
            if not year_data:
                result[iso2] = None
                continue
            latest_yr = max(year_data.keys())
            result[iso2] = _safe_float(year_data.get(latest_yr))
        return result


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def intl_financials(ticker: str) -> InternationalFinancials:
    """Fetch financial data for an international ticker. Creates a fresh client."""
    client = InternationalFundamentals()
    try:
        return await client.get_ticker_financials(ticker, convert_to_usd=True)
    finally:
        await client.aclose()


async def country_profile(country: str) -> MacroCountryProfile:
    """Fetch macro profile for a single country. Creates a fresh client."""
    client = InternationalFundamentals()
    try:
        return await client.get_macro_country_profile(country)
    finally:
        await client.aclose()


async def global_peers(ticker: str, sector: str) -> ComparablePeerSet:
    """Fetch cross-border peer comparison for a ticker. Creates a fresh client."""
    client = InternationalFundamentals()
    try:
        return await client.get_international_peers(ticker, sector)
    finally:
        await client.aclose()


async def global_valuation_compare(ticker: str) -> dict:
    """Full global valuation comparison. Auto-detects sector. Creates a fresh client."""
    client = InternationalFundamentals()
    try:
        return await client.compare_global_valuations(ticker)
    finally:
        await client.aclose()


async def recent_20f(days_back: int = 90, limit: int = 50) -> list[Form20F]:
    """Fetch recent 20-F filings from EDGAR. Creates a fresh client."""
    client = InternationalFundamentals()
    try:
        return await client.get_recent_20f_filings(days_back=days_back, limit=limit)
    finally:
        await client.aclose()
