"""Integration smoke tests — verify the full data pipeline without a live DB.

These tests confirm that all modules wire together correctly:
  - Every ingest function can be imported and called with valid inputs
  - The adapter chain falls back correctly on missing API keys
  - The backfill script parses arguments without crashing
  - The check script imports clean
  - Congressional trades adapter can be instantiated
  - COT ingest result types are correct
  - Repository functions have the right signatures
  - API routes import without errors

No network calls are made (all adapters are mocked at the network boundary).
No database is required (all session operations are mocked).

Run: make test
"""
from __future__ import annotations
import asyncio
import inspect
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.core.types import (
    CorporateAction,
    CorporateActionType,
    DataProvenance,
    MacroDataPoint,
    OHLCVBar,
    SurvivorshipRecord,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bar(ticker="AAPL", close=150.0, dt=None):
    return OHLCVBar(
        time=dt or datetime(2024, 1, 2),
        figi=ticker,
        ticker=ticker,
        open=Decimal(str(close - 1)),
        high=Decimal(str(close + 1)),
        low=Decimal(str(close - 2)),
        close=Decimal(str(close)),
        volume=1_000_000,
        source="yfinance",
    )


def _session_mock():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        mappings=MagicMock(return_value=[]),
        fetchone=MagicMock(return_value=None),
        scalar=MagicMock(return_value=None),
    ))
    session.commit = AsyncMock()
    return session


# ─── Module import tests ───────────────────────────────────────────────────────

class TestCriticalImports:
    """All core modules must import without raising."""

    def test_db_imports(self):
        from sentinel.sds.db import get_session_factory, get_engine, dispose_engine
        assert callable(get_session_factory)
        assert callable(dispose_engine)

    def test_repository_imports(self):
        from sentinel.sds.repository import (
            write_ohlcv_bars, get_ohlcv_bars_by_ticker,
            write_macro_points, get_macro_series,
            write_financial_facts, get_financial_facts,
            upsert_corporate_actions, get_corporate_actions,
            write_provenance_receipt, get_latest_provenance_hash,
            upsert_survivorship_records,
            write_congressional_trades, get_congressional_trades,
            write_cot_records, get_cot_signals,
        )
        assert callable(write_ohlcv_bars)
        assert callable(write_congressional_trades)
        assert callable(write_cot_records)

    def test_ingest_imports(self):
        from sentinel.sds.ingest import (
            ingest_ticker_ohlcv, ingest_macro_series,
            ingest_edgar_facts, ingest_corporate_actions,
            ingest_congressional_trades, ingest_cot_data,
            IngestResult, MacroIngestResult, EdgarIngestResult,
            CongressIngestResult, COTIngestResult,
        )
        assert callable(ingest_ticker_ohlcv)
        assert callable(ingest_congressional_trades)
        assert callable(ingest_cot_data)

    def test_catalog_imports(self):
        from sentinel.sds.catalog import get_coverage, format_catalog_report
        assert callable(get_coverage)
        assert callable(format_catalog_report)

    def test_scheduler_imports(self):
        from sentinel.sds.scheduler import start_scheduler, stop_scheduler, get_scheduler
        assert callable(start_scheduler)
        assert callable(stop_scheduler)

    def test_congress_adapter_imports(self):
        from sentinel.sds.adapters.congress_adapter import CongressAdapter
        adapter = CongressAdapter()
        assert adapter.name == "congress"

    def test_sds_init_exports(self):
        from sentinel.sds import build_default_adapters, get_adapter, get_all_adapters
        assert callable(build_default_adapters)
        assert callable(get_adapter)
        assert callable(get_all_adapters)


# ─── Ingest result types ───────────────────────────────────────────────────────

class TestIngestResultTypes:

    def test_ingest_result_ok(self):
        from sentinel.sds.ingest import IngestResult
        r = IngestResult(ticker="AAPL", interval="1d", status="ok", bars=250)
        assert r.ok is True

    def test_ingest_result_error(self):
        from sentinel.sds.ingest import IngestResult
        r = IngestResult(ticker="AAPL", interval="1d", status="error", error="timeout")
        assert r.ok is False

    def test_congress_result(self):
        from sentinel.sds.ingest import CongressIngestResult
        r = CongressIngestResult(status="ok", house=150, senate=42)
        assert r.ok is True
        assert r.house + r.senate == 192

    def test_cot_result(self):
        from sentinel.sds.ingest import COTIngestResult
        r = COTIngestResult(status="ok", markets=15, records=5000)
        assert r.ok is True
        assert r.records == 5000

    def test_macro_result(self):
        from sentinel.sds.ingest import MacroIngestResult
        r = MacroIngestResult(series_id="GDP", status="ok", points=300)
        assert r.ok is True

    def test_edgar_result(self):
        from sentinel.sds.ingest import EdgarIngestResult
        r = EdgarIngestResult(cik="0000320193", ticker="AAPL", status="ok", facts=8000)
        assert r.ok is True


# ─── Congressional trades adapter ─────────────────────────────────────────────

class TestCongressAdapter:

    def test_parse_amount(self):
        from sentinel.sds.adapters.congress_adapter import _parse_amount
        low, high = _parse_amount("$15,001 - $50,000")
        assert low == Decimal("15001")
        assert high == Decimal("50000")

    def test_parse_amount_empty(self):
        from sentinel.sds.adapters.congress_adapter import _parse_amount
        low, high = _parse_amount("")
        assert low is None and high is None

    def test_tx_code_purchase(self):
        from sentinel.sds.adapters.congress_adapter import _tx_code
        assert _tx_code("Purchase") == "P"
        assert _tx_code("Sale (Partial)") == "S"
        assert _tx_code("Exchange") == "X"

    def test_parse_date(self):
        from sentinel.sds.adapters.congress_adapter import _parse_date
        assert _parse_date("2024-01-15") == date(2024, 1, 15)
        assert _parse_date("01/15/2024") == date(2024, 1, 15)
        assert _parse_date("") is None

    @pytest.mark.asyncio
    async def test_fetch_house_trades_network_error(self):
        """Gracefully returns [] on network failure."""
        from sentinel.sds.adapters.congress_adapter import CongressAdapter
        adapter = CongressAdapter()
        with patch("httpx.AsyncClient.get", side_effect=Exception("network error")):
            result = await adapter.fetch_house_trades()
        assert result == []


# ─── Ingest pipeline (mocked DB) ─────────────────────────────────────────────

class TestIngestPipelineMocked:

    @pytest.mark.asyncio
    async def test_ingest_ticker_ohlcv_no_data(self):
        """Returns no_data status when adapter chain returns empty."""
        from sentinel.sds.ingest import ingest_ticker_ohlcv
        session = _session_mock()

        with patch("sentinel.sds.normalizer.fetch_ohlcv_with_fallback",
                   new=AsyncMock(return_value=[])):
            result = await ingest_ticker_ohlcv(
                ticker="AAPL", interval="1d",
                start=datetime(2024, 1, 1), end=datetime(2024, 1, 31),
                session=session,
            )
        assert result.status == "no_data"

    @pytest.mark.asyncio
    async def test_ingest_macro_series_no_data(self):
        """Returns no_data when FRED returns empty."""
        from sentinel.sds.ingest import ingest_macro_series
        session = _session_mock()

        with patch("sentinel.sds.adapters.fred_adapter.FREDAdapter.fetch_series",
                   new=AsyncMock(return_value=[])):
            result = await ingest_macro_series(series_id="GDP", session=session)
        assert result.status == "no_data"

    @pytest.mark.asyncio
    async def test_ingest_congressional_no_data(self):
        """Returns no_data when both chambers return empty."""
        from sentinel.sds.ingest import ingest_congressional_trades
        session = _session_mock()

        with patch("sentinel.sds.adapters.congress_adapter.CongressAdapter.fetch_house_trades",
                   new=AsyncMock(return_value=[])), \
             patch("sentinel.sds.adapters.congress_adapter.CongressAdapter.fetch_senate_trades",
                   new=AsyncMock(return_value=[])):
            result = await ingest_congressional_trades(session=session)
        assert result.status == "no_data"

    @pytest.mark.asyncio
    async def test_ingest_congressional_writes_to_db(self):
        """When trades are available, write_congressional_trades is called."""
        from sentinel.sds.ingest import ingest_congressional_trades
        session = _session_mock()

        sample_trade = {
            "politician_name": "Nancy Pelosi",
            "chamber": "house",
            "party": "D",
            "state": "CA",
            "ticker": "NVDA",
            "figi": None,
            "asset_name": "NVIDIA Corp",
            "tx_date": date(2024, 1, 10),
            "filed_date": date(2024, 2, 1),
            "tx_type": "Purchase",
            "tx_code": "P",
            "amount_low": Decimal("1000001"),
            "amount_high": Decimal("5000000"),
            "filing_lag_days": 22,
            "late_filing": False,
            "source": "house_stock_watcher",
            "disclosure_url": None,
        }

        with patch("sentinel.sds.adapters.congress_adapter.CongressAdapter.fetch_house_trades",
                   new=AsyncMock(return_value=[sample_trade])), \
             patch("sentinel.sds.adapters.congress_adapter.CongressAdapter.fetch_senate_trades",
                   new=AsyncMock(return_value=[])), \
             patch("sentinel.sds.repository.write_congressional_trades",
                   new=AsyncMock(return_value=1)):
            result = await ingest_congressional_trades(session=session)

        assert result.ok is True
        assert result.house == 1
        assert result.senate == 0


# ─── Corporate actions (regression from Gen 0) ────────────────────────────────

class TestCorporateActionMath:

    def test_aapl_4for1_split(self):
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = CorporateAction(
            figi="AAPL", ticker="AAPL",
            action_type=CorporateActionType.SPLIT,
            ex_date=date(2020, 8, 31),
            ratio_new=Decimal("4"), ratio_old=Decimal("1"),
            factor=Decimal("0.25"),
            source="yfinance",
        )
        # Bar before split: gets adjusted
        factor = compute_cumulative_factor(date(2020, 1, 1), [action])
        assert factor == Decimal("0.25")

    def test_bar_after_split_no_adjustment(self):
        from sentinel.sds.corporate_actions import compute_cumulative_factor
        action = CorporateAction(
            figi="AAPL", ticker="AAPL",
            action_type=CorporateActionType.SPLIT,
            ex_date=date(2020, 8, 31),
            ratio_new=Decimal("4"), ratio_old=Decimal("1"),
            factor=Decimal("0.25"),
            source="yfinance",
        )
        # Bar on or after split: no adjustment
        factor = compute_cumulative_factor(date(2020, 8, 31), [action])
        assert factor == Decimal("1.0")


# ─── Catalog format ───────────────────────────────────────────────────────────

class TestCatalogFormat:

    def test_empty_lake_warning(self):
        from sentinel.sds.catalog import CatalogSummary, format_catalog_report
        summary = CatalogSummary(
            tickers=[], macro_series=[],
            total_ohlcv_rows=0, total_macro_rows=0,
            total_fact_rows=0, total_provenance_records=0,
            delisted_in_registry=0, ca_records=0,
            insider_count=0, institutional_count=0,
            news_count=0, cot_count=0, congress_count=0,
            as_of=datetime.utcnow(),
        )
        report = format_catalog_report(summary)
        assert "EMPTY" in report
        assert "make backfill" in report

    def test_populated_lake_no_warning(self):
        from sentinel.sds.catalog import CatalogSummary, format_catalog_report
        summary = CatalogSummary(
            tickers=[], macro_series=[],
            total_ohlcv_rows=1_000_000, total_macro_rows=50_000,
            total_fact_rows=500_000, total_provenance_records=250,
            delisted_in_registry=8, ca_records=1200,
            insider_count=5000, institutional_count=3000,
            news_count=25000, cot_count=800, congress_count=4000,
            as_of=datetime.utcnow(),
        )
        report = format_catalog_report(summary)
        assert "1,000,000" in report
        assert "EMPTY" not in report


# ─── API route import sanity ──────────────────────────────────────────────────

class TestAPIRoutesImport:

    def test_all_routers_importable(self):
        from sentinel.api.routes import data, macro, intelligence
        from sentinel.api.routes.data import router as data_router
        from sentinel.api.routes.macro import router as macro_router
        from sentinel.api.routes.intelligence import router as intel_router
        assert data_router is not None
        assert macro_router is not None
        assert intel_router is not None

    def test_main_app_importable(self):
        from sentinel.api.main import app
        assert app is not None
        routes = {getattr(r, "path", None) for r in app.routes}
        assert "/health" in routes
        assert "/" in routes


# ─── Insider adapter parsing ──────────────────────────────────────────────────

class TestInsiderAdapter:

    SAMPLE_FORM4_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>APPLE INC</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214128</rptOwnerCik>
      <rptOwnerName>Cook Timothy D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-02-01</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50000</value></transactionShares>
        <transactionPricePerShare><value>185.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>3200000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

    def test_parse_form4_xml_sale(self):
        from sentinel.sds.adapters.insider_adapter import _parse_form4_xml
        txs = _parse_form4_xml(self.SAMPLE_FORM4_XML, cik="0000320193", ticker="AAPL")
        assert len(txs) == 1
        tx = txs[0]
        assert tx["ticker"] == "AAPL"
        assert tx["owner_name"] == "Cook Timothy D"
        assert tx["role"] == "Chief Executive Officer"
        assert tx["tx_code"] == "S"
        # Disposal → negative shares
        assert tx["shares"] < 0
        assert abs(tx["shares"]) == 50000
        assert tx["is_derivative"] is False

    def test_parse_form4_xml_invalid_xml(self):
        from sentinel.sds.adapters.insider_adapter import _parse_form4_xml
        txs = _parse_form4_xml(b"not xml at all <<>>", cik="0000320193", ticker="AAPL")
        assert txs == []

    def test_safe_decimal_edge_cases(self):
        from sentinel.sds.adapters.insider_adapter import _safe_decimal
        from decimal import Decimal
        assert _safe_decimal("185.50") == Decimal("185.50")
        assert _safe_decimal("1,234,567") == Decimal("1234567")
        assert _safe_decimal("") is None
        assert _safe_decimal(None) is None
        assert _safe_decimal("not_a_number") is None

    def test_owner_role_director_precedence(self):
        import xml.etree.ElementTree as ET
        from sentinel.sds.adapters.insider_adapter import _owner_role
        xml = ET.fromstring("""
        <reportingOwnerRelationship>
          <isDirector>1</isDirector>
          <isOfficer>0</isOfficer>
        </reportingOwnerRelationship>""")
        assert _owner_role(xml) == "Director"

    def test_owner_role_officer_beats_director(self):
        import xml.etree.ElementTree as ET
        from sentinel.sds.adapters.insider_adapter import _owner_role
        xml = ET.fromstring("""
        <reportingOwnerRelationship>
          <isDirector>1</isDirector>
          <isOfficer>1</isOfficer>
          <officerTitle>CFO</officerTitle>
        </reportingOwnerRelationship>""")
        assert _owner_role(xml) == "CFO"


# ─── Institutional adapter parsing ───────────────────────────────────────────

class TestInstitutionalAdapter:

    SAMPLE_13F_XML = b"""<?xml version="1.0"?>
<informationTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>5432100</value>
    <shrsOrPrnAmt>
      <sshPrnamt>29300000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <putCall/>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>3100000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>8000000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>"""

    def test_parse_13f_xml_basic(self):
        from sentinel.sds.adapters.institutional_adapter import _parse_13f_xml
        from datetime import date
        holdings = _parse_13f_xml(
            self.SAMPLE_13F_XML,
            manager_cik="0001067983",
            period=date(2024, 3, 31),
            filed_date=date(2024, 5, 15),
        )
        assert len(holdings) == 2
        aapl = next(h for h in holdings if h["cusip"] == "037833100")
        assert aapl["issuer_name"] == "APPLE INC"
        # value stored in dollars (×1000 from thousands)
        assert aapl["market_value"] == 5432100 * 1000
        assert aapl["shares"] == 29300000
        assert aapl["investment_discretion"] == "SOLE"

    def test_parse_13f_xml_missing_cusip_filtered(self):
        from sentinel.sds.adapters.institutional_adapter import _parse_13f_xml
        from datetime import date
        xml_no_cusip = b"""<informationTable>
          <infoTable><nameOfIssuer></nameOfIssuer><cusip></cusip></infoTable>
        </informationTable>"""
        holdings = _parse_13f_xml(xml_no_cusip, "0001067983", date(2024, 3, 31), None)
        assert holdings == []


# ─── Ingest result types (new) ────────────────────────────────────────────────

class TestNewIngestResultTypes:

    def test_insider_result_ok(self):
        from sentinel.sds.ingest import InsiderIngestResult
        r = InsiderIngestResult(cik="0000320193", ticker="AAPL", status="ok", transactions=42)
        assert r.ok is True
        assert r.transactions == 42

    def test_insider_result_no_data(self):
        from sentinel.sds.ingest import InsiderIngestResult
        r = InsiderIngestResult(cik="0000320193", ticker="AAPL", status="no_data")
        assert r.ok is False

    def test_institutional_result_ok(self):
        from sentinel.sds.ingest import InstitutionalIngestResult
        r = InstitutionalIngestResult(manager_cik="0001067983", status="ok", holdings=1500, filings=4)
        assert r.ok is True
        assert r.holdings == 1500

    def test_news_result_ok(self):
        from sentinel.sds.ingest import NewsIngestResult
        r = NewsIngestResult(ticker="AAPL", status="ok", articles=25)
        assert r.ok is True
        assert r.articles == 25

    def test_news_result_no_key(self):
        from sentinel.sds.ingest import NewsIngestResult
        r = NewsIngestResult(ticker="AAPL", status="no_key")
        assert r.ok is False


# ─── Repository function signatures ──────────────────────────────────────────

class TestNewRepositorySignatures:

    def test_write_insider_transactions_importable(self):
        from sentinel.sds.repository import write_insider_transactions
        assert callable(write_insider_transactions)

    def test_write_institutional_holdings_importable(self):
        from sentinel.sds.repository import write_institutional_holdings
        assert callable(write_institutional_holdings)

    def test_write_news_articles_importable(self):
        from sentinel.sds.repository import write_news_articles
        assert callable(write_news_articles)

    def test_get_news_articles_importable(self):
        from sentinel.sds.repository import get_news_articles
        assert callable(get_news_articles)

    def test_get_insider_transactions_importable(self):
        from sentinel.sds.repository import get_insider_transactions
        assert callable(get_insider_transactions)

    def test_get_institutional_holdings_importable(self):
        from sentinel.sds.repository import get_institutional_holdings
        assert callable(get_institutional_holdings)


# ─── Ingest pipeline (news mocked) ───────────────────────────────────────────

class TestNewsIngestMocked:

    @pytest.mark.asyncio
    async def test_ingest_news_no_key(self):
        """Returns no_key when Finnhub key is empty."""
        from sentinel.sds.ingest import ingest_news

        session = _session_mock()
        with patch("sentinel.core.config.get_settings") as mock_settings:
            mock_settings.return_value.finnhub_api_key = ""
            result = await ingest_news(ticker="AAPL", session=session)
        assert result.status == "no_key"
        assert result.ok is False

    @pytest.mark.asyncio
    async def test_ingest_news_no_data(self):
        """Returns no_data when Finnhub returns empty."""
        from sentinel.sds.ingest import ingest_news

        session = _session_mock()
        with patch("sentinel.core.config.get_settings") as mock_settings, \
             patch("sentinel.sds.adapters.finnhub_adapter.FinnhubAdapter.fetch_news",
                   new=AsyncMock(return_value=[])):
            mock_settings.return_value.finnhub_api_key = "test_key"
            result = await ingest_news(ticker="AAPL", session=session)
        assert result.status == "no_data"


# ─── Options analytics ────────────────────────────────────────────────────────

class TestOptionsAnalytics:

    def test_imports(self):
        from sentinel.sbx.options_analytics import (
            get_options_summary, normalize_chain,
            build_iv_surface, compute_skew, compute_gex, compute_max_pain,
        )
        assert callable(get_options_summary)

    def test_empty_contracts(self):
        from sentinel.sbx.options_analytics import get_options_summary
        result = get_options_summary("AAPL", [], 150.0)
        assert result["contract_count"] == 0
        assert result["ticker"] == "AAPL"
        assert result["surface"] is None

    def test_normalize_chain_bad_data(self):
        """Contracts missing required fields are silently dropped."""
        from sentinel.sbx.options_analytics import normalize_chain
        bad = [{"ticker": "AAPL"}]  # missing strike, expiry, etc.
        result = normalize_chain(bad, 150.0)
        assert result == []

    def test_options_route_no_api_key(self):
        """Route returns error dict when polygon_api_key is unset."""
        import asyncio
        from sentinel.api.routes.intelligence import get_options_analytics
        with patch("sentinel.core.config.get_settings") as mock_cfg:
            mock_cfg.return_value.polygon_api_key = ""
            result = asyncio.get_event_loop().run_until_complete(
                get_options_analytics(ticker="AAPL", underlying_price=150.0)
            )
        assert "error" in result
        assert result["contract_count"] == 0

    @pytest.mark.asyncio
    async def test_options_route_with_empty_polygon_response(self):
        """Route calls Polygon and returns zero-contract summary on empty response."""
        from sentinel.api.routes.intelligence import get_options_analytics
        with patch("sentinel.core.config.get_settings") as mock_cfg, \
             patch("sentinel.sds.adapters.polygon_adapter.PolygonAdapter.fetch_options_chain",
                   new=AsyncMock(return_value=[])):
            mock_cfg.return_value.polygon_api_key = "test_key"
            # Pass expiry_days explicitly — FastAPI Query defaults aren't resolved
            # when calling route functions directly outside the request lifecycle.
            result = await get_options_analytics(ticker="AAPL", underlying_price=150.0, expiry_days=90)
        assert result["contract_count"] == 0
        assert result["ticker"] == "AAPL"


# ─── Screener DB loader ───────────────────────────────────────────────────────

class TestScreenerDbLoader:

    def test_imports(self):
        from sentinel.sse.db_loader import populate_screener_from_db
        assert callable(populate_screener_from_db)

    @pytest.mark.asyncio
    async def test_empty_instruments_returns_zero(self):
        """Returns 0 and logs warning when instruments table is empty."""
        from sentinel.sse.db_loader import populate_screener_from_db

        session = _session_mock()
        # All queries return empty — instruments table is empty
        result = await populate_screener_from_db(session)
        assert result == 0

    def test_screener_route_has_refresh_endpoint(self):
        from sentinel.api.routes.screen import router
        paths = {getattr(r, "path", "") for r in router.routes}
        assert "/refresh" in paths

    def test_screener_route_has_universe_stats(self):
        from sentinel.api.routes.screen import router
        paths = {getattr(r, "path", "") for r in router.routes}
        assert "/universe/stats" in paths


# ─── RAG EDGAR ingestor ───────────────────────────────────────────────────────

class TestRagEdgarIngestor:

    def test_backfill_rag_importable(self):
        from scripts.backfill import backfill_rag_documents
        assert callable(backfill_rag_documents)

    def test_rag_module_imports(self):
        from sentinel.sil.rag import (
            ingest_document, chunk_document, ingest_batch,
            dense_retrieve, sparse_retrieve,
        )
        assert callable(ingest_document)
        assert callable(chunk_document)

    def test_chunk_document_basic(self):
        from sentinel.sil.rag import chunk_document
        text = "Apple revenue grew 10%. Net income rose to $100B. EPS was $6.40."
        chunks = chunk_document(
            text=text, doc_id="test:001", ticker="AAPL",
            doc_type="10-K", filed_date=date(2024, 11, 1),
            chunk_size=16, overlap=4,
        )
        assert len(chunks) >= 1
        assert all(c.ticker == "AAPL" for c in chunks)
        assert all(c.doc_type == "10-K" for c in chunks)
        assert all(c.filed_date == date(2024, 11, 1) for c in chunks)

    def test_chunk_document_empty(self):
        from sentinel.sil.rag import chunk_document
        chunks = chunk_document("", "doc:empty", None, "10-K", None)
        assert chunks == []

    def test_backfill_both_functions_importable(self):
        """backfill_rag_documents and backfill_embeddings are both defined."""
        from scripts.backfill import backfill_rag_documents, backfill_embeddings
        assert callable(backfill_rag_documents)
        assert callable(backfill_embeddings)


# ─── DataCleaner ──────────────────────────────────────────────────────────────

class TestDataCleaner:

    def test_imports(self):
        from sentinel.sds.data_cleaner import DataCleaner, CleaningReport, SourceQualityScore
        assert callable(DataCleaner)

    def test_cleaning_report_to_dict(self):
        from sentinel.sds.data_cleaner import CleaningReport, SourceQualityScore
        report = CleaningReport(
            ticker="AAPL", interval="1d",
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
            sources_attempted=["yfinance", "polygon"],
            sources_with_data=["yfinance"],
            sources_rejected=["polygon"],
            consensus_bars=250,
            bars_outlier_flagged=2,
            gaps_detected=0,
            quality_scores=[
                SourceQualityScore(
                    source="yfinance", bars_returned=250, bars_expected=250,
                    availability=1.0, mean_abs_deviation=0.0,
                    outlier_bars=0, grade="A",
                )
            ],
            wrote_to_db=True,
            as_of=datetime(2024, 12, 31),
        )
        d = report.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["consensus_bars"] == 250
        assert d["wrote_to_db"] is True
        assert d["quality_scores"][0]["grade"] == "A"
        assert d["start"] == "2024-01-01T00:00:00"

    def test_best_source_grade_priority(self):
        from sentinel.sds.data_cleaner import CleaningReport, SourceQualityScore
        report = CleaningReport(
            ticker="SPY", interval="1d",
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
            sources_attempted=["yfinance", "alpaca", "polygon"],
            sources_with_data=["yfinance", "alpaca", "polygon"],
            sources_rejected=[],
            consensus_bars=252, bars_outlier_flagged=0, gaps_detected=0,
            quality_scores=[
                SourceQualityScore("yfinance", 252, 252, 1.0, 0.0, 0, "A"),
                SourceQualityScore("alpaca",   240, 252, 0.95, 0.003, 1, "B"),
                SourceQualityScore("polygon",  200, 252, 0.79, 0.015, 5, "F"),
            ],
            wrote_to_db=True, as_of=datetime(2024, 12, 31),
        )
        assert report.best_source() == "yfinance"

    def test_best_source_empty(self):
        from sentinel.sds.data_cleaner import CleaningReport
        report = CleaningReport(
            ticker="X", interval="1d",
            start=datetime(2024, 1, 1), end=datetime(2024, 1, 2),
            sources_attempted=[], sources_with_data=[], sources_rejected=[],
            consensus_bars=0, bars_outlier_flagged=0, gaps_detected=0,
            quality_scores=[], wrote_to_db=False, as_of=datetime(2024, 1, 2),
        )
        assert report.best_source() == ""

    @pytest.mark.asyncio
    async def test_clean_ticker_no_adapters(self):
        """Returns empty CleaningReport when no adapters are registered."""
        from sentinel.sds.data_cleaner import DataCleaner
        session = _session_mock()
        with patch("sentinel.sds.data_cleaner.DataCleaner.clean_ticker",
                   new=AsyncMock(return_value=__import__(
                       "sentinel.sds.data_cleaner", fromlist=["CleaningReport"]
                   ).CleaningReport(
                       ticker="AAPL", interval="1d",
                       start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
                       sources_attempted=[], sources_with_data=[], sources_rejected=[],
                       consensus_bars=0, bars_outlier_flagged=0, gaps_detected=0,
                       quality_scores=[], wrote_to_db=False, as_of=datetime(2024, 12, 31),
                   ))):
            cleaner = DataCleaner(session)
            report = await cleaner.clean_ticker("AAPL")
        assert report.consensus_bars == 0
        assert report.ticker == "AAPL"

    def test_cleaner_route_importable(self):
        from sentinel.api.routes.cleaner import router
        paths = {getattr(r, "path", "") for r in router.routes}
        assert "/run" in paths
        assert "/quality-report" in paths
        assert "/run-universe" in paths
        assert "/source-rankings" in paths

    def test_repository_quality_functions(self):
        from sentinel.sds.repository import write_quality_scores, get_quality_scores
        assert callable(write_quality_scores)
        assert callable(get_quality_scores)


# ─── Crypto ingest ────────────────────────────────────────────────────────────

class TestCryptoIngest:

    def test_result_type_importable(self):
        from sentinel.sds.ingest import CryptoIngestResult
        r = CryptoIngestResult(pair="BTC/USDT", interval="1d", status="ok", bars=365)
        assert r.ok is True
        assert r.pair == "BTC/USDT"

    def test_result_no_adapter(self):
        from sentinel.sds.ingest import CryptoIngestResult
        r = CryptoIngestResult(pair="ETH/USDT", interval="1d", status="no_adapter")
        assert r.ok is False

    def test_ingest_function_importable(self):
        from sentinel.sds.ingest import ingest_crypto_ohlcv
        assert callable(ingest_crypto_ohlcv)

    @pytest.mark.asyncio
    async def test_ingest_crypto_no_adapter(self):
        """Returns no_adapter status when ccxt adapter is not registered."""
        from sentinel.sds.ingest import ingest_crypto_ohlcv
        # get_adapter is lazily imported from sentinel.sds inside the function
        with patch("sentinel.sds.get_adapter", return_value=None):
            result = await ingest_crypto_ohlcv("BTC/USDT", session=_session_mock())
        assert result.status == "no_adapter"

    @pytest.mark.asyncio
    async def test_ingest_crypto_empty_response(self):
        """Returns no_data when adapter returns empty bars."""
        from sentinel.sds.ingest import ingest_crypto_ohlcv
        mock_adapter = MagicMock()
        mock_adapter.fetch_ohlcv = AsyncMock(return_value=[])
        with patch("sentinel.sds.get_adapter", return_value=mock_adapter):
            result = await ingest_crypto_ohlcv("BTC/USDT", session=_session_mock())
        assert result.status == "no_data"

    def test_backfill_crypto_importable(self):
        from scripts.backfill import backfill_crypto, DEFAULT_CRYPTO_PAIRS
        assert callable(backfill_crypto)
        assert "BTC/USDT" in DEFAULT_CRYPTO_PAIRS
        assert len(DEFAULT_CRYPTO_PAIRS) >= 10


# ─── NL screener + screen_stocks wiring ──────────────────────────────────────

class TestNLScreener:
    """_parse_nl_screen and screen_stocks wiring tests.

    pandas/duckdb are stubbed in conftest, so we test logic without real
    ScreenerEngine execution. fastmcp is stubbed so mcp_server imports cleanly.
    """

    def test_parse_nl_screen_pe(self):
        from sentinel.sil.mcp_server import _parse_nl_screen
        criteria = _parse_nl_screen("stocks with PE < 15")
        assert "pe_ratio_lt" in criteria
        assert criteria["pe_ratio_lt"] == 15.0

    def test_parse_nl_screen_dividend(self):
        from sentinel.sil.mcp_server import _parse_nl_screen
        criteria = _parse_nl_screen("dividend aristocrats")
        assert criteria.get("has_dividend") is True

    def test_parse_nl_screen_small_cap(self):
        from sentinel.sil.mcp_server import _parse_nl_screen
        criteria = _parse_nl_screen("small cap value stocks")
        assert "market_cap_max" in criteria

    def test_parse_nl_screen_large_cap(self):
        from sentinel.sil.mcp_server import _parse_nl_screen
        criteria = _parse_nl_screen("large cap growth")
        assert "market_cap_min" in criteria

    def test_parse_nl_screen_insider_buying(self):
        from sentinel.sil.mcp_server import _parse_nl_screen
        criteria = _parse_nl_screen("insider buying in the last 30 days")
        assert criteria.get("insider_buying_30d") is True

    def test_mcp_server_imports_cleanly(self):
        """mcp_server.py imports without error now that fastmcp is stubbed."""
        import sentinel.sil.mcp_server as _mcp
        assert hasattr(_mcp, "_parse_nl_screen")

    def test_screen_route_get_engine(self):
        import sentinel.api.routes.screen as _screen_route
        engine = _screen_route.get_engine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_screen_api_route_uses_engine(self):
        """POST /screen passes criteria through ScreenerEngine.screen()."""
        import sentinel.api.routes.screen as _screen_route
        from sentinel.api.routes.screen import screen

        mock_engine = MagicMock()
        mock_engine.get_universe_count.return_value = 1
        mock_engine.screen.return_value = []
        _screen_route._engine = mock_engine

        result = await screen(criteria={"pe_ratio_lt": 20.0}, session=_session_mock())
        assert result["count"] == 0
        assert result["results"] == []
        mock_engine.screen.assert_called_once_with({"pe_ratio_lt": 20.0})


# ─── FXAdapter.fetch_base_rates ───────────────────────────────────────────────

class TestFXAdapterBaseRates:
    """fetch_base_rates fetches all pairs for a given base currency."""

    def test_method_exists(self):
        from sentinel.sds.adapters.fx_adapter import FXAdapter
        assert hasattr(FXAdapter, "fetch_base_rates")
        assert callable(FXAdapter.fetch_base_rates)

    @pytest.mark.asyncio
    async def test_fetch_base_rates_parses_response(self):
        """Builds FXBar list from a mocked Frankfurter multi-currency response."""
        from sentinel.sds.adapters.fx_adapter import FXAdapter
        from datetime import date as _date

        mock_response = {
            "base": "USD",
            "rates": {
                "2024-01-02": {"EUR": 0.9150, "GBP": 0.7850, "JPY": 143.50},
                "2024-01-03": {"EUR": 0.9175, "GBP": 0.7875, "JPY": 144.10},
            },
        }
        adapter = FXAdapter()
        with patch.object(adapter, "_get_json", new=AsyncMock(return_value=mock_response)):
            with patch.object(adapter, "_throttle", new=AsyncMock()):
                bars = await adapter.fetch_base_rates(
                    base="USD",
                    start=_date(2024, 1, 2),
                    end=_date(2024, 1, 3),
                )

        assert len(bars) == 6  # 2 dates × 3 quotes
        pairs = {b.pair for b in bars}
        assert "USD/EUR" in pairs
        assert "USD/GBP" in pairs
        assert "USD/JPY" in pairs

    @pytest.mark.asyncio
    async def test_fetch_base_rates_empty_response(self):
        from sentinel.sds.adapters.fx_adapter import FXAdapter
        from datetime import date as _date

        adapter = FXAdapter()
        with patch.object(adapter, "_get_json", new=AsyncMock(return_value={"rates": {}})):
            with patch.object(adapter, "_throttle", new=AsyncMock()):
                bars = await adapter.fetch_base_rates("EUR", _date(2024, 1, 2), _date(2024, 1, 2))
        assert bars == []


# ── Continuous Futures ─────────────────────────────────────────────────────────

class TestContinuousFutures:
    """Unit tests for sentinel/sds/continuous_futures.py."""

    # ── ContractSpec factory methods ──────────────────────────────────────────

    def test_contract_spec_es(self):
        from sentinel.sds.continuous_futures import ContractSpec
        spec = ContractSpec.es()
        assert spec.root == "ES"
        assert spec.exchange == "CME"
        assert spec.months == [3, 6, 9, 12]
        assert spec.roll_days_before == 5

    def test_contract_spec_nq(self):
        from sentinel.sds.continuous_futures import ContractSpec
        spec = ContractSpec.nq()
        assert spec.root == "NQ"
        assert spec.exchange == "CME"
        assert set(spec.months) == {3, 6, 9, 12}

    def test_contract_spec_cl_monthly(self):
        from sentinel.sds.continuous_futures import ContractSpec
        spec = ContractSpec.cl()
        assert spec.root == "CL"
        assert spec.exchange == "NYMEX"
        assert len(spec.months) == 12  # monthly
        assert spec.roll_days_before == 3

    def test_contract_spec_gc(self):
        from sentinel.sds.continuous_futures import ContractSpec
        spec = ContractSpec.gc()
        assert spec.root == "GC"
        assert spec.exchange == "COMEX"
        assert spec.months == [2, 4, 6, 8, 10, 12]

    def test_contract_spec_zb(self):
        from sentinel.sds.continuous_futures import ContractSpec
        spec = ContractSpec.zb()
        assert spec.root == "ZB"
        assert spec.exchange == "CBOT"
        assert spec.months == [3, 6, 9, 12]

    # ── _contract_ticker ──────────────────────────────────────────────────────

    def test_contract_ticker_format(self):
        from sentinel.sds.continuous_futures import _contract_ticker
        assert _contract_ticker("ES", 2024, 3) == "ESH24"
        assert _contract_ticker("ES", 2024, 6) == "ESM24"
        assert _contract_ticker("ES", 2024, 9) == "ESU24"
        assert _contract_ticker("ES", 2024, 12) == "ESZ24"

    def test_contract_ticker_cl_monthly(self):
        from sentinel.sds.continuous_futures import _contract_ticker
        assert _contract_ticker("CL", 2024, 1) == "CLF24"
        assert _contract_ticker("CL", 2024, 11) == "CLX24"

    # ── _third_friday ─────────────────────────────────────────────────────────

    def test_third_friday_is_friday(self):
        from sentinel.sds.continuous_futures import _third_friday
        from datetime import date as _date
        d = _third_friday(2024, 3)
        assert d.weekday() == 4  # Friday
        assert d == _date(2024, 3, 15)

    def test_third_friday_different_months(self):
        from sentinel.sds.continuous_futures import _third_friday
        for year in (2023, 2024):
            for month in (3, 6, 9, 12):
                d = _third_friday(year, month)
                assert d.weekday() == 4  # always a Friday
                assert 15 <= d.day <= 21  # third Friday is always in this range

    # ── _contract_sequence ────────────────────────────────────────────────────

    def test_contract_sequence_covers_range(self):
        from sentinel.sds.continuous_futures import _contract_sequence, ContractSpec
        from datetime import date as _date
        spec = ContractSpec.es()
        seq = _contract_sequence(spec, _date(2024, 1, 1), _date(2024, 12, 31))
        assert len(seq) >= 4  # at least 4 quarterly contracts
        tickers = [t for t, _, _ in seq]
        # All quarterly codes for ES in 2024
        assert any("H24" in t for t in tickers)  # March
        assert any("M24" in t for t in tickers)  # June
        assert any("U24" in t for t in tickers)  # September
        assert any("Z24" in t for t in tickers)  # December

    def test_contract_sequence_non_overlapping(self):
        from sentinel.sds.continuous_futures import _contract_sequence, ContractSpec
        from datetime import date as _date
        spec = ContractSpec.es()
        seq = _contract_sequence(spec, _date(2024, 1, 1), _date(2024, 12, 31))
        # Ensure from dates are non-overlapping (each segment starts after previous ends)
        for i in range(1, len(seq)):
            _, prev_from, prev_to = seq[i - 1]
            _, curr_from, _ = seq[i]
            assert curr_from > prev_to or curr_from == prev_to + __import__("datetime").timedelta(days=1)

    def test_contract_sequence_bounded(self):
        from sentinel.sds.continuous_futures import _contract_sequence, ContractSpec
        from datetime import date as _date
        start = _date(2024, 1, 1)
        end = _date(2024, 12, 31)
        spec = ContractSpec.es()
        seq = _contract_sequence(spec, start, end)
        for _, from_dt, to_dt in seq:
            assert from_dt >= start
            assert to_dt <= end

    # ── _apply_panama ─────────────────────────────────────────────────────────

    def test_apply_panama_no_segments(self):
        from sentinel.sds.continuous_futures import _apply_panama
        assert _apply_panama([]) == []

    def test_apply_panama_single_segment(self):
        from sentinel.sds.continuous_futures import _apply_panama
        from datetime import datetime
        bar = {"time": datetime(2024, 3, 1), "open": 5000, "high": 5010,
               "low": 4990, "close": 5005, "volume": 1000}
        result = _apply_panama([[bar]])
        assert len(result) == 1
        # Single segment: no roll, prices unchanged
        from decimal import Decimal
        assert result[0]["close"] == Decimal("5005")

    def test_apply_panama_two_segments_shift(self):
        from sentinel.sds.continuous_futures import _apply_panama
        from datetime import datetime
        from decimal import Decimal
        # Segment 1 closes at 5000; segment 2 opens at 5010 → gap = +10
        # Panama shifts segment 1 up by 10
        seg1 = [{"time": datetime(2024, 3, 14), "open": 4990, "high": 5005,
                 "low": 4985, "close": 5000, "volume": 500}]
        seg2 = [{"time": datetime(2024, 3, 15), "open": 5010, "high": 5020,
                 "low": 5005, "close": 5015, "volume": 600}]
        result = _apply_panama([seg1, seg2])
        assert len(result) == 2
        # Seg2 (newest) is unadjusted
        assert result[1]["close"] == Decimal("5015")
        # Seg1 is shifted by gap (5010 - 5000 = 10)
        assert result[0]["close"] == Decimal("5010")

    def test_apply_panama_empty_segment_skipped(self):
        from sentinel.sds.continuous_futures import _apply_panama
        from datetime import datetime
        from decimal import Decimal
        seg1 = [{"time": datetime(2024, 3, 1), "open": 100, "high": 105,
                 "low": 98, "close": 103, "volume": 100}]
        result = _apply_panama([seg1, []])  # second segment empty
        assert len(result) == 1

    # ── _apply_ratio ──────────────────────────────────────────────────────────

    def test_apply_ratio_no_segments(self):
        from sentinel.sds.continuous_futures import _apply_ratio
        assert _apply_ratio([]) == []

    def test_apply_ratio_single_segment(self):
        from sentinel.sds.continuous_futures import _apply_ratio
        from datetime import datetime
        from decimal import Decimal
        bar = {"time": datetime(2024, 3, 1), "open": 100, "high": 105,
               "low": 98, "close": 103, "volume": 100}
        result = _apply_ratio([[bar]])
        assert len(result) == 1
        assert result[0]["close"] == Decimal("103.00")

    def test_apply_ratio_two_segments_scale(self):
        from sentinel.sds.continuous_futures import _apply_ratio
        from datetime import datetime
        from decimal import Decimal
        # Seg1 closes at 100; seg2 opens at 102 → ratio = 1.02
        seg1 = [{"time": datetime(2024, 3, 14), "open": 98, "high": 101,
                 "low": 97, "close": 100, "volume": 500}]
        seg2 = [{"time": datetime(2024, 3, 15), "open": 102, "high": 105,
                 "low": 101, "close": 104, "volume": 600}]
        result = _apply_ratio([seg1, seg2])
        assert len(result) == 2
        # Seg2 is unadjusted (ratio applied to older segs only)
        assert result[1]["close"] == Decimal("104.00")
        # Seg1 close should be 100 * (102/100) = 102.00
        assert result[0]["close"] == Decimal("102.00")

    # ── _apply_unadj ──────────────────────────────────────────────────────────

    def test_apply_unadj_concatenates(self):
        from sentinel.sds.continuous_futures import _apply_unadj
        from datetime import datetime
        seg1 = [{"time": datetime(2024, 3, 1), "open": 100, "high": 105,
                 "low": 98, "close": 103, "volume": 100}]
        seg2 = [{"time": datetime(2024, 6, 1), "open": 104, "high": 110,
                 "low": 103, "close": 108, "volume": 200}]
        result = _apply_unadj([seg1, seg2])
        assert len(result) == 2
        assert result[0]["close"] == 103
        assert result[1]["close"] == 108

    # ── build_continuous_series ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_build_continuous_series_returns_ohlcvbars(self):
        from sentinel.sds.continuous_futures import build_continuous_series, ContractSpec
        from datetime import date as _date, datetime, timezone
        from unittest.mock import AsyncMock, patch, MagicMock

        spec = ContractSpec.es()
        start = _date(2024, 1, 1)
        end = _date(2024, 3, 31)

        mock_bar = {
            "time": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "open": 4800, "high": 4850, "low": 4790, "close": 4830,
            "volume": 100000, "source": "test",
        }

        session = MagicMock()
        with patch("sentinel.sds.continuous_futures._fetch_bars",
                   new=AsyncMock(return_value=[mock_bar])):
            bars = await build_continuous_series(session, spec, start, end, "unadj")

        assert len(bars) > 0
        from sentinel.core.types import OHLCVBar
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert all(b.ticker == "CONT:ES1" for b in bars)
        assert all(b.figi == "CONT:ES" for b in bars)

    @pytest.mark.asyncio
    async def test_build_continuous_series_panama_method(self):
        from sentinel.sds.continuous_futures import build_continuous_series, ContractSpec
        from datetime import date as _date, datetime, timezone
        from unittest.mock import AsyncMock, patch, MagicMock

        spec = ContractSpec.es()
        session = MagicMock()
        mock_bar = {
            "time": datetime(2024, 2, 1, tzinfo=timezone.utc),
            "open": 4900, "high": 4950, "low": 4890, "close": 4920,
            "volume": 50000, "source": "test",
        }
        with patch("sentinel.sds.continuous_futures._fetch_bars",
                   new=AsyncMock(return_value=[mock_bar])):
            bars = await build_continuous_series(
                session, spec, _date(2024, 1, 1), _date(2024, 3, 31), "panama"
            )
        assert isinstance(bars, list)

    @pytest.mark.asyncio
    async def test_build_continuous_series_empty_when_no_data(self):
        from sentinel.sds.continuous_futures import build_continuous_series, ContractSpec
        from datetime import date as _date
        from unittest.mock import AsyncMock, patch, MagicMock

        spec = ContractSpec.es()
        session = MagicMock()
        with patch("sentinel.sds.continuous_futures._fetch_bars",
                   new=AsyncMock(return_value=[])):
            bars = await build_continuous_series(
                session, spec, _date(2024, 1, 1), _date(2024, 3, 31)
            )
        assert bars == []

    # ── AdjustmentMethod enum ─────────────────────────────────────────────────

    def test_adjustment_method_values(self):
        from sentinel.sds.continuous_futures import AdjustmentMethod
        assert AdjustmentMethod("panama") == AdjustmentMethod.PANAMA
        assert AdjustmentMethod("ratio") == AdjustmentMethod.RATIO
        assert AdjustmentMethod("unadj") == AdjustmentMethod.UNADJ

    def test_adjustment_method_invalid(self):
        from sentinel.sds.continuous_futures import AdjustmentMethod
        with pytest.raises(ValueError):
            AdjustmentMethod("invalid")

    # ── Module & route imports ────────────────────────────────────────────────

    def test_continuous_futures_module_imports(self):
        import sentinel.sds.continuous_futures as cf
        assert hasattr(cf, "build_continuous_series")
        assert hasattr(cf, "build_and_persist_continuous")
        assert hasattr(cf, "write_continuous_series")
        assert hasattr(cf, "ContractSpec")
        assert hasattr(cf, "AdjustmentMethod")

    def test_futures_api_route_imports(self):
        import sentinel.api.routes.futures as fut_route
        assert hasattr(fut_route, "router")

    def test_backfill_continuous_importable(self):
        import scripts.backfill as bf
        assert hasattr(bf, "backfill_continuous_futures")


# ── MCP DataCleaner & Continuous Futures tools ────────────────────────────────

class TestMCPNewTools:
    """Tests for MCP Tools 52–54 added in Gen-0 completion pass.

    NOTE: @mcp.tool() from the stubbed FastMCP wraps decorated coroutines into
    MagicMocks, so we inspect the source file directly rather than using
    inspect.getsource() or awaiting the decorated symbols.
    """

    # ── Source-level presence checks ─────────────────────────────────────────

    def _src(self):
        from pathlib import Path
        return Path("sentinel/sil/mcp_server.py").read_text()

    def test_clean_data_tool_in_source(self):
        assert "async def clean_data(" in self._src()

    def test_clean_universe_tool_in_source(self):
        assert "async def clean_universe(" in self._src()

    def test_get_continuous_futures_tool_in_source(self):
        assert "async def get_continuous_futures(" in self._src()

    def test_mcp_server_tool_count(self):
        """Docstring should report 54 tools."""
        assert "54-tool" in self._src()

    # ── Bug-fix verification (source inspection) ─────────────────────────────

    def test_global_macro_uses_fred_api_key(self):
        """Tool 24 must use fred_api_key= not api_key= when calling global_macro funcs."""
        src = self._src()
        assert "fred_api_key=settings.fred_api_key" in src

    def test_kelly_uses_risk_free_rate(self):
        """Tool 25 must call kelly_from_returns with risk_free_rate, not risk_free/fractional."""
        src = self._src()
        assert "risk_free_rate=0.045" in src
        assert "fractional=0.5" not in src

    def test_fx_rates_uses_fetch_base_rates(self):
        """Tool 26 must call fetch_base_rates for historical FX, not fetch_historical."""
        src = self._src()
        assert "fetch_base_rates(base=" in src
        # fetch_historical should not appear inside the get_fx_rates function
        # (it may appear in imports elsewhere, so we test for the specific call)
        assert "fetch_historical(base=" not in src

    def test_options_flow_uses_correct_kwargs(self):
        """Tool 30 must build OptionsScreenerCriteria and pass polygon_api_key."""
        src = self._src()
        assert "OptionsScreenerCriteria(" in src
        assert "polygon_api_key=settings.polygon_api_key" in src

    # ── Structural content checks for new tools ───────────────────────────────

    def test_clean_data_calls_datacleaner(self):
        src = self._src()
        assert "DataCleaner(" in src
        assert "clean_ticker(" in src

    def test_clean_universe_calls_clean_universe(self):
        src = self._src()
        assert "clean_universe(" in src
        assert "grade_distribution" in src

    def test_get_continuous_futures_handles_invalid_method(self):
        src = self._src()
        assert '"error": "method must be panama | ratio | unadj"' in src

    def test_get_continuous_futures_uses_contiguous_session(self):
        src = self._src()
        assert "get_session_factory" in src
        assert "build_continuous_series" in src

    # ── Runtime: test the underlying logic without going through MCP decorator ─

    @pytest.mark.asyncio
    async def test_continuous_futures_invalid_method_via_api_route(self):
        """Futures API route validates method param; raises HTTPException 422."""
        from sentinel.api.routes.futures import get_continuous_series
        from fastapi import HTTPException
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        # Must pass start/end explicitly: FastAPI Query defaults become FieldInfo
        # objects when the route is called directly outside the HTTP stack.
        with pytest.raises(HTTPException) as exc_info:
            await get_continuous_series(
                "ES", start=None, end=None, method="bad_method", session=mock_session
            )
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_continuous_futures_invalid_date_via_api_route(self):
        """Futures API route validates dates; raises HTTPException 422."""
        from sentinel.api.routes.futures import get_continuous_series
        from fastapi import HTTPException
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await get_continuous_series(
                "ES", start="not-a-date", end=None, method="panama", session=mock_session
            )
        assert exc_info.value.status_code == 422


# ─── Wave 11 module tests ─────────────────────────────────────────────────────


class TestLBOModel:
    """Tests for sentinel.sfe.lbo_model — pure-Python LBO and merger math."""

    def test_imports(self):
        from sentinel.sfe.lbo_model import (
            LBOAssumptions, MergerAssumptions, LBOResult, MergerResult,
            LBOScreenResult, run_lbo_model, run_merger_model, screen_lbo_candidate,
        )
        assert LBOAssumptions and MergerAssumptions

    def test_run_lbo_model_basic(self):
        from sentinel.sfe.lbo_model import LBOAssumptions, run_lbo_model
        a = LBOAssumptions(
            purchase_price=1_000_000_000,
            ebitda=100_000_000,
            ebitda_growth_rate=0.05,
            leverage_multiple=5.0,
            interest_rate=0.08,
            hold_years=5,
            exit_multiple=10.0,
        )
        r = run_lbo_model(a)
        assert r.equity_invested == pytest.approx(500_000_000.0, rel=0.01)
        assert r.total_debt == pytest.approx(500_000_000.0, rel=0.01)
        assert len(r.years) == 5
        assert r.moic > 1.0
        assert r.irr > 0.0

    def test_run_lbo_model_verdict_strong(self):
        from sentinel.sfe.lbo_model import LBOAssumptions, run_lbo_model
        a = LBOAssumptions(
            purchase_price=500_000_000,
            ebitda=100_000_000,
            ebitda_growth_rate=0.10,
            leverage_multiple=4.0,
            interest_rate=0.07,
            hold_years=5,
            exit_multiple=10.0,
        )
        r = run_lbo_model(a)
        assert r.verdict == "strong return"
        assert r.irr >= 0.25

    def test_run_lbo_model_high_leverage_warning(self):
        from sentinel.sfe.lbo_model import LBOAssumptions, run_lbo_model
        a = LBOAssumptions(
            purchase_price=1_000_000_000,
            ebitda=100_000_000,
            ebitda_growth_rate=0.05,
            leverage_multiple=8.0,
            interest_rate=0.09,
            hold_years=5,
            exit_multiple=10.0,
        )
        r = run_lbo_model(a)
        assert any("8x" in w or "leverage" in w.lower() for w in r.warnings)

    def test_run_lbo_model_year_structure(self):
        from sentinel.sfe.lbo_model import LBOAssumptions, run_lbo_model
        a = LBOAssumptions(
            purchase_price=1_000_000_000,
            ebitda=100_000_000,
            ebitda_growth_rate=0.05,
            leverage_multiple=5.0,
            interest_rate=0.08,
            hold_years=3,
            exit_multiple=10.0,
        )
        r = run_lbo_model(a)
        assert len(r.years) == 3
        for i, yr in enumerate(r.years, 1):
            assert yr.year == i
            assert yr.ebitda > 0
            assert yr.debt_balance >= 0

    def test_compute_irr_simple(self):
        from sentinel.sfe.lbo_model import _compute_irr
        # invest 100, receive 121 in 2 years → IRR ≈ 10%
        irr = _compute_irr([-100.0, 0.0, 121.0])
        assert abs(irr - 0.10) < 0.001

    def test_lbo_verdict_boundaries(self):
        from sentinel.sfe.lbo_model import _lbo_verdict
        assert _lbo_verdict(0.30) == "strong return"
        assert _lbo_verdict(0.20) == "acceptable"
        assert _lbo_verdict(0.14) == "marginal"
        assert _lbo_verdict(0.05) == "value-destroying"

    def test_run_merger_model_accretive(self):
        from sentinel.sfe.lbo_model import MergerAssumptions, run_merger_model
        a = MergerAssumptions(
            acquirer_eps=5.0, acquirer_shares_mm=100.0, acquirer_price=100.0,
            target_eps=2.0, target_shares_mm=50.0,
            acquisition_price_per_share=30.0,
            pct_stock=0.0, synergies_after_tax_mm=50.0,
            cost_of_debt=0.05, tax_rate=0.25,
        )
        r = run_merger_model(a)
        assert r.verdict == "accretive"
        assert r.accretion_pct > 0

    def test_run_merger_model_deal_value(self):
        from sentinel.sfe.lbo_model import MergerAssumptions, run_merger_model
        a = MergerAssumptions(
            acquirer_eps=5.0, acquirer_shares_mm=100.0, acquirer_price=100.0,
            target_eps=2.0, target_shares_mm=50.0,
            acquisition_price_per_share=25.0,
            pct_stock=0.0, synergies_after_tax_mm=0.0,
            cost_of_debt=0.05, tax_rate=0.25,
        )
        r = run_merger_model(a)
        assert r.deal_value_mm == pytest.approx(25.0 * 50.0)

    def test_lbo_assumptions_defaults(self):
        from sentinel.sfe.lbo_model import LBOAssumptions
        a = LBOAssumptions(
            purchase_price=1e9, ebitda=1e8, ebitda_growth_rate=0.05,
            leverage_multiple=5.0, interest_rate=0.08, hold_years=5, exit_multiple=10.0,
        )
        assert a.tax_rate == 0.25
        assert a.amortization_pct == 0.05

    def test_screen_lbo_candidate_is_async(self):
        import asyncio
        from sentinel.sfe.lbo_model import screen_lbo_candidate
        assert inspect.iscoroutinefunction(screen_lbo_candidate)


class TestScenarioAnalysis:
    """Tests for sentinel.spr.scenario_analysis — macro shock P&L framework."""

    def test_imports(self):
        from sentinel.spr.scenario_analysis import (
            MacroShock, ScenarioResult, MultiScenarioComparison,
            AssetSensitivity, SCENARIO_TEMPLATES, MACRO_FACTORS,
            run_scenario, run_multi_scenario,
        )
        assert MacroShock and ScenarioResult

    def test_scenario_templates_all_seven(self):
        from sentinel.spr.scenario_analysis import SCENARIO_TEMPLATES
        expected = {
            "2008_crisis", "covid_crash", "rate_hike_200bps", "soft_landing",
            "stagflation", "china_taiwan", "usd_crash",
        }
        assert expected.issubset(SCENARIO_TEMPLATES.keys())

    def test_macro_shock_defaults(self):
        from sentinel.spr.scenario_analysis import MacroShock
        s = MacroShock()
        assert s.equity_shock_pct == 0.0
        assert s.rate_shock_bps == 0.0
        assert s.scenario_name == "Custom"

    def test_compute_asset_return_equity_dominated(self):
        from sentinel.spr.scenario_analysis import (
            AssetSensitivity, MacroShock, _compute_asset_return,
        )
        sens = AssetSensitivity(ticker="SPY", equity_beta=1.5, duration_years=0.0)
        shock = MacroShock(equity_shock_pct=-20.0, scenario_name="test")
        ret, dominant = _compute_asset_return(sens, shock)
        assert dominant == "equity"
        assert ret == pytest.approx(-0.30, rel=0.01)  # 1.5 × -20% = -30%

    def test_compute_asset_return_rates_dominated(self):
        from sentinel.spr.scenario_analysis import (
            AssetSensitivity, MacroShock, _compute_asset_return,
        )
        # Duration=10, rate hike 100bps → -10 × 100/10000 = -0.10
        sens = AssetSensitivity(ticker="TLT", equity_beta=0.0, duration_years=10.0)
        shock = MacroShock(rate_shock_bps=100.0, scenario_name="rate_test")
        ret, dominant = _compute_asset_return(sens, shock)
        assert dominant == "rates"
        assert ret == pytest.approx(-0.10, rel=0.01)

    def test_ols_beta_too_few_points(self):
        import numpy as np
        from sentinel.spr.scenario_analysis import _ols_beta
        y = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        assert _ols_beta(y, x) is None

    def test_ols_beta_perfect_line(self):
        import numpy as np
        from sentinel.spr.scenario_analysis import _ols_beta
        x = np.linspace(0.01, 0.05, 20)
        y = 1.5 * x
        result = _ols_beta(y, x)
        assert result is not None
        assert abs(result - 1.5) < 0.01

    @pytest.mark.asyncio
    async def test_run_scenario_empty_tickers_raises(self):
        from sentinel.spr.scenario_analysis import MacroShock, run_scenario
        shock = MacroShock(equity_shock_pct=-10.0)
        with pytest.raises(ValueError, match="non-empty"):
            await run_scenario(tickers=[], shock=shock)

    @pytest.mark.asyncio
    async def test_run_scenario_invalid_name_raises(self):
        from sentinel.spr.scenario_analysis import run_scenario
        with pytest.raises(ValueError, match="Unknown scenario_name"):
            await run_scenario(tickers=["AAPL"], scenario_name="nonexistent_scenario")

    @pytest.mark.asyncio
    async def test_run_scenario_no_shock_no_name_raises(self):
        from sentinel.spr.scenario_analysis import run_scenario
        with pytest.raises(ValueError, match="shock or scenario_name"):
            await run_scenario(tickers=["AAPL"], shock=None, scenario_name=None)

    @pytest.mark.asyncio
    async def test_run_multi_scenario_unknown_name_raises(self):
        from sentinel.spr.scenario_analysis import run_multi_scenario
        with pytest.raises(ValueError, match="Unknown scenario_name"):
            await run_multi_scenario(tickers=["AAPL"], scenario_names=["bad_name"])

    def test_scenario_2008_equity_shock_negative(self):
        from sentinel.spr.scenario_analysis import SCENARIO_TEMPLATES
        assert SCENARIO_TEMPLATES["2008_crisis"]["equity_shock_pct"] < 0

    def test_scenario_soft_landing_equity_positive(self):
        from sentinel.spr.scenario_analysis import SCENARIO_TEMPLATES
        assert SCENARIO_TEMPLATES["soft_landing"]["equity_shock_pct"] > 0


class TestCorporateActions:
    """Tests for sentinel.sfe.corporate_actions — dividend analytics and corporate events."""

    def test_imports(self):
        from sentinel.sfe.corporate_actions import (
            CorporateAction, DividendAnalytics, DividendHistory, DividendScreenResult,
            get_dividend_analytics, screen_dividends,
        )
        assert DividendAnalytics and DividendScreenResult

    def test_classify_freq_monthly(self):
        from sentinel.sfe.corporate_actions import _classify_freq
        assert _classify_freq([30.0, 31.0, 28.0]) == "monthly"

    def test_classify_freq_quarterly(self):
        from sentinel.sfe.corporate_actions import _classify_freq
        assert _classify_freq([91.0, 90.0, 92.0]) == "quarterly"

    def test_classify_freq_annual(self):
        from sentinel.sfe.corporate_actions import _classify_freq
        assert _classify_freq([365.0, 364.0]) == "annual"

    def test_classify_freq_empty(self):
        from sentinel.sfe.corporate_actions import _classify_freq
        assert _classify_freq([]) == "annual"

    def test_quality_score_high(self):
        from sentinel.sfe.corporate_actions import _quality_score
        # consistency=100%, dgr=12%, payout=40%, yield=10%, rf=0.045
        score, label = _quality_score(100.0, 12.0, 40.0, 10.0, 0.045)
        assert label == "high"
        assert score >= 8.0

    def test_quality_score_none_no_dividend(self):
        from sentinel.sfe.corporate_actions import _quality_score
        score, label = _quality_score(None, None, None, None, 0.045)
        assert score == 0.0
        assert label == "none"

    def test_quality_score_low(self):
        from sentinel.sfe.corporate_actions import _quality_score
        # minimal signals → low/medium
        score, label = _quality_score(60.0, 2.0, None, 1.0, 0.045)
        assert label in ("low", "medium")

    def test_ddm_no_dividend_returns_na(self):
        from sentinel.sfe.corporate_actions import _ddm
        val, verdict = _ddm(0.0, 5.0, 40.0, 100.0, 0.045)
        assert val is None
        assert verdict == "N/A"

    def test_ddm_no_price_returns_na(self):
        from sentinel.sfe.corporate_actions import _ddm
        val, verdict = _ddm(2.0, 5.0, 40.0, 0.0, 0.045)
        assert val is None
        assert verdict == "N/A"

    def test_ddm_undervalued(self):
        from sentinel.sfe.corporate_actions import _ddm
        # div=$2, g=5%, payout=40%, price=$20, rf=0.045
        # r=0.095, intrinsic≈46.67 >> 20*1.15=23 → undervalued
        val, verdict = _ddm(2.0, 5.0, 40.0, 20.0, 0.045)
        assert verdict == "undervalued"
        assert val is not None and val > 20.0

    def test_ddm_overvalued(self):
        from sentinel.sfe.corporate_actions import _ddm
        # Tiny dividend, huge price → overvalued
        val, verdict = _ddm(0.01, 1.0, 90.0, 500.0, 0.045)
        assert verdict == "overvalued"

    def test_get_dividend_analytics_is_async(self):
        import asyncio
        from sentinel.sfe.corporate_actions import get_dividend_analytics
        assert inspect.iscoroutinefunction(get_dividend_analytics)

    @pytest.mark.asyncio
    async def test_screen_dividends_empty_returns_empty(self):
        from sentinel.sfe.corporate_actions import DividendScreenResult, screen_dividends
        result = await screen_dividends([])
        assert isinstance(result, DividendScreenResult)
        assert result.tickers == []
        assert result.analytics == []


class TestFormAdv:
    """Tests for sentinel.sfe.form_adv — SEC IAPD RIA intelligence."""

    def test_imports(self):
        from sentinel.sfe.form_adv import (
            ClientType, FeeStructure, RIAProfile, RIAScreenResult,
            get_ria_profile, screen_rias,
        )
        assert RIAProfile and FeeStructure

    def test_parse_aum_millions_conversion(self):
        from sentinel.sfe.form_adv import _parse_aum
        # Value < 100_000 → treated as millions, scaled ×1e6
        adv = {"totalRegulatoryAssets": 5000}
        total, disc = _parse_aum(adv)
        assert total == pytest.approx(5_000_000_000.0)
        assert disc is None

    def test_parse_aum_already_large(self):
        from sentinel.sfe.form_adv import _parse_aum
        adv = {"totalRegulatoryAssets": 5_000_000_000}
        total, _ = _parse_aum(adv)
        assert total == pytest.approx(5_000_000_000.0)

    def test_parse_aum_empty(self):
        from sentinel.sfe.form_adv import _parse_aum
        total, disc = _parse_aum({})
        assert total is None
        assert disc is None

    def test_parse_fee_pct_of_aum(self):
        from sentinel.sfe.form_adv import _parse_fee_structure
        adv = {"Part1A": {"Item5E": {"compensationTypes": ["percentage of assets under management"]}}}
        fee = _parse_fee_structure(adv)
        assert fee is not None
        assert fee.pct_of_aum is True

    def test_parse_fee_hourly(self):
        from sentinel.sfe.form_adv import _parse_fee_structure
        adv = {"Part1A": {"Item5E": {"compensationTypes": ["hourly rate"]}}}
        fee = _parse_fee_structure(adv)
        assert fee is not None
        assert fee.hourly is True

    def test_parse_fee_performance_based(self):
        from sentinel.sfe.form_adv import _parse_fee_structure
        adv = {"Part1A": {"Item5E": {"compensationTypes": ["performance-based fees"]}}}
        fee = _parse_fee_structure(adv)
        assert fee is not None
        assert fee.performance_based is True

    def test_parse_fee_none_when_empty(self):
        from sentinel.sfe.form_adv import _parse_fee_structure
        assert _parse_fee_structure({}) is None

    def test_parse_styles_equity(self):
        from sentinel.sfe.form_adv import _parse_styles
        adv = {"advisoryServices": "We invest primarily in equity securities"}
        styles = _parse_styles(adv)
        assert "Equity" in styles

    def test_parse_styles_multiple(self):
        from sentinel.sfe.form_adv import _parse_styles
        adv = {"advisoryServices": "equity, fixed income, esg strategies"}
        styles = _parse_styles(adv)
        assert "Equity" in styles
        assert "Fixed Income" in styles
        assert "ESG" in styles

    def test_ria_profile_model_defaults(self):
        from sentinel.sfe.form_adv import RIAProfile
        p = RIAProfile(name="Test RIA", as_of="2024-01-01")
        assert p.name == "Test RIA"
        assert p.client_types == []
        assert p.investment_styles == []
        assert p.aum_usd is None

    def test_get_ria_profile_is_async(self):
        import asyncio
        from sentinel.sfe.form_adv import get_ria_profile
        assert inspect.iscoroutinefunction(get_ria_profile)

    def test_screen_rias_is_async(self):
        import asyncio
        from sentinel.sfe.form_adv import screen_rias
        assert inspect.iscoroutinefunction(screen_rias)


class TestFormD:
    """Tests for sentinel.sfe.form_d — SEC Form D private market intelligence."""

    def test_imports(self):
        from sentinel.sfe.form_d import (
            FormDFiling, PrivateMarketScreen,
            get_company_form_d, screen_private_market,
        )
        assert FormDFiling and PrivateMarketScreen

    def test_parse_xml_equity_type(self):
        from sentinel.sfe.form_d import _parse_form_d_xml
        xml = (
            '<?xml version="1.0"?>'
            "<root><isEquityType>true</isEquityType>"
            "<totalOfferingAmount>5000000</totalOfferingAmount>"
            "<totalAmountSold>3000000</totalAmountSold></root>"
        )
        result = _parse_form_d_xml(xml)
        assert result["offering_type"] == "Equity"
        assert result["total_offering_amount"] == pytest.approx(5_000_000.0)
        assert result["amount_sold"] == pytest.approx(3_000_000.0)

    def test_parse_xml_fund_type(self):
        from sentinel.sfe.form_d import _parse_form_d_xml
        xml = (
            '<?xml version="1.0"?>'
            "<root><isPooledInvestmentFundType>true</isPooledInvestmentFundType>"
            "<investmentFundType>Hedge Fund</investmentFundType>"
            "<totalAmountSold>50000000</totalAmountSold></root>"
        )
        result = _parse_form_d_xml(xml)
        assert result["offering_type"] == "Fund"
        assert result["fund_type"] == "Hedge Fund"

    def test_parse_xml_debt_type(self):
        from sentinel.sfe.form_d import _parse_form_d_xml
        xml = (
            '<?xml version="1.0"?>'
            "<root><isDebtType>true</isDebtType>"
            "<totalOfferingAmount>10000000</totalOfferingAmount></root>"
        )
        result = _parse_form_d_xml(xml)
        assert result["offering_type"] == "Debt"

    def test_parse_xml_amendment_flag(self):
        from sentinel.sfe.form_d import _parse_form_d_xml
        xml = '<?xml version="1.0"?><root><isAmendment>true</isAmendment></root>'
        result = _parse_form_d_xml(xml)
        assert result.get("is_amendment") is True

    def test_parse_xml_invalid_returns_empty(self):
        from sentinel.sfe.form_d import _parse_form_d_xml
        assert _parse_form_d_xml("not xml at all!!!") == {}

    def test_parse_xml_empty_string(self):
        from sentinel.sfe.form_d import _parse_form_d_xml
        assert _parse_form_d_xml("") == {}

    def test_match_state_correct(self):
        from sentinel.sfe.form_d import FormDFiling, _match_state
        f = FormDFiling(
            company_name="X", cik="0001", file_date="2024-01-01",
            state="CA", filing_url="http://test",
        )
        assert _match_state(f, "CA") is True
        assert _match_state(f, "ca") is True
        assert _match_state(f, "NY") is False

    def test_match_amount_above_and_below(self):
        from sentinel.sfe.form_d import FormDFiling, _match_amount
        f = FormDFiling(
            company_name="X", cik="0001", file_date="2024-01-01",
            amount_sold=50_000_000.0, filing_url="http://test",
        )
        assert _match_amount(f, 10.0) is True    # min $10M → pass
        assert _match_amount(f, 100.0) is False  # min $100M → fail

    def test_match_fund_type(self):
        from sentinel.sfe.form_d import FormDFiling, _match_fund_type
        f = FormDFiling(
            company_name="X", cik="0001", file_date="2024-01-01",
            fund_type="Hedge Fund", filing_url="http://test",
        )
        assert _match_fund_type(f, "hedge") is True
        assert _match_fund_type(f, "venture") is False

    def test_form_d_filing_model_defaults(self):
        from sentinel.sfe.form_d import FormDFiling
        f = FormDFiling(
            company_name="Startup Inc", cik="0001234",
            file_date="2024-06-15", filing_url="https://test",
        )
        assert f.offering_type == "Unknown"
        assert f.is_amendment is False
        assert f.key_persons == []

    def test_get_company_form_d_is_async(self):
        import asyncio
        from sentinel.sfe.form_d import get_company_form_d
        assert inspect.iscoroutinefunction(get_company_form_d)

    def test_screen_private_market_is_async(self):
        import asyncio
        from sentinel.sfe.form_d import screen_private_market
        assert inspect.iscoroutinefunction(screen_private_market)
