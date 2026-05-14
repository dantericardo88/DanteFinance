from __future__ import annotations
import asyncio
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Optional
import pandas as pd
import httpx
from pydantic import BaseModel, Field
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_SEARCH = "https://efts.sec.gov"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}

ASSET_CAT_MAP = {
    "EC": "equity_common",
    "EP": "equity_preferred",
    "DB": "debt",
    "ABS": "asset_backed",
    "MBS": "mortgage_backed",
    "MM": "money_market",
    "RA": "real_assets",
    "DER": "derivative",
    "OTH": "other",
}

_NPORT_FORMS = {"N-PORT", "N-PORT-P"}


class NPORTHolding(BaseModel):
    name: str
    cusip: Optional[str] = None
    ticker: Optional[str] = None
    lei: Optional[str] = None
    isin: Optional[str] = None
    balance: float
    units: str
    market_value_usd: float
    pct_of_nav: float
    asset_category: str
    country: Optional[str] = None
    currency: str = "USD"


class NPORTReport(BaseModel):
    fund_cik: str
    fund_name: str
    period_date: date
    filing_date: date
    total_assets: Optional[float] = None
    net_assets: Optional[float] = None
    n_holdings: int
    holdings: list[NPORTHolding]


class PositionChange(BaseModel):
    ticker: Optional[str]
    cusip: Optional[str]
    name: str
    prev_pct: Optional[float]
    curr_pct: Optional[float]
    change_pct_pts: float
    prev_value: Optional[float]
    curr_value: Optional[float]
    action: str


def _strip_ns(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _find(el: ET.Element, local: str) -> Optional[ET.Element]:
    for child in el.iter():
        if _strip_ns(child.tag) == local:
            return child
    return None


def _findall(el: ET.Element, local: str) -> list[ET.Element]:
    return [child for child in el.iter() if _strip_ns(child.tag) == local]


def _text(el: ET.Element, local: str) -> Optional[str]:
    found = _find(el, local)
    if found is not None and found.text:
        return found.text.strip()
    return None


def _float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _parse_date(val: Optional[str]) -> Optional[date]:
    if not val:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(val.strip(), fmt).date()
        except ValueError:
            continue
    return None


class NPORTParser:
    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    def _clean_accession(self, accession: str) -> str:
        return accession.replace("-", "")

    async def get_fund_filings(self, fund_cik: str, n_months: int = 12) -> list[dict]:
        cik_padded = fund_cik.zfill(10)
        url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(url, headers=_HEADERS)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.error("EDGAR submissions fetch failed", cik=fund_cik, error=str(exc))
                return []

        data = resp.json()
        fund_name = data.get("name", "")
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        period_dates = recent.get("periodOfReport", [])
        primary_docs = recent.get("primaryDocument", [])

        results: list[dict] = []
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=n_months)

        for i, form in enumerate(forms):
            if form not in _NPORT_FORMS:
                continue
            fdate_str = filing_dates[i] if i < len(filing_dates) else None
            fdate = _parse_date(fdate_str)
            if fdate and pd.Timestamp(fdate) < cutoff:
                continue
            pdate_str = period_dates[i] if i < len(period_dates) else None
            acc = accessions[i] if i < len(accessions) else ""
            pdoc = primary_docs[i] if i < len(primary_docs) else ""
            results.append({
                "accession_number": acc,
                "filing_date": fdate,
                "period_date": _parse_date(pdate_str),
                "primary_doc": pdoc,
                "fund_name": fund_name,
                "form": form,
            })

        results.sort(key=lambda x: x["filing_date"] or date.min, reverse=True)
        logger.info("N-PORT filings found", cik=fund_cik, count=len(results))
        return results

    async def _fetch_filing_index(
        self, client: httpx.AsyncClient, fund_cik: str, accession_number: str
    ) -> list[dict]:
        cik_stripped = str(int(fund_cik))
        acc_clean = self._clean_accession(accession_number)
        index_url = f"{EDGAR_ARCHIVES}/{cik_stripped}/{acc_clean}/{accession_number}-index.json"
        try:
            resp = await client.get(index_url, headers=_HEADERS)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("directory", {}).get("item", [])
        except httpx.HTTPError:
            pass

        index_url2 = f"{EDGAR_ARCHIVES}/{cik_stripped}/{acc_clean}/{acc_clean}-index.json"
        try:
            resp = await client.get(index_url2, headers=_HEADERS)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("directory", {}).get("item", [])
        except httpx.HTTPError:
            pass

        return []

    async def _fetch_xml_content(
        self,
        client: httpx.AsyncClient,
        fund_cik: str,
        accession_number: str,
        primary_doc: Optional[str] = None,
    ) -> Optional[str]:
        cik_stripped = str(int(fund_cik))
        acc_clean = self._clean_accession(accession_number)
        base = f"{EDGAR_ARCHIVES}/{cik_stripped}/{acc_clean}"

        items = await self._fetch_filing_index(client, fund_cik, accession_number)
        candidates: list[str] = []

        for item in items:
            name = item.get("name", "")
            doc_type = item.get("type", "")
            if name.lower().endswith(".xml") and "nport" in name.lower():
                candidates.insert(0, name)
            elif name.lower().endswith(".xml") and doc_type in ("N-PORT", "N-PORT-P"):
                candidates.insert(0, name)
            elif name.lower().endswith(".xml"):
                candidates.append(name)

        if primary_doc and primary_doc.lower().endswith(".xml"):
            candidates.insert(0, primary_doc)
        else:
            candidates.insert(0, "primary_doc.xml")

        seen: set[str] = set()
        ordered: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        for doc_name in ordered:
            url = f"{base}/{doc_name}"
            try:
                resp = await client.get(url, headers=_HEADERS)
                if resp.status_code == 200:
                    text = resp.text
                    if "invstOrSec" in text or "edgarSubmission" in text or "N-PORT" in text:
                        logger.debug("N-PORT XML found", cik=fund_cik, doc=doc_name)
                        return text
            except httpx.HTTPError:
                continue

        logger.warning("N-PORT XML not found", cik=fund_cik, accession=accession_number)
        return None

    async def parse_nport_filing(self, fund_cik: str, accession_number: str) -> NPORTReport:
        filings_meta = await self.get_fund_filings(fund_cik, n_months=36)
        filing_meta = next(
            (f for f in filings_meta if f["accession_number"] == accession_number), None
        )
        fund_name = filing_meta["fund_name"] if filing_meta else ""
        filing_date = filing_meta["filing_date"] if filing_meta else date.today()
        primary_doc = filing_meta.get("primary_doc") if filing_meta else None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            xml_text = await self._fetch_xml_content(
                client, fund_cik, accession_number, primary_doc
            )

        if not xml_text:
            raise ValueError(
                f"Could not retrieve N-PORT XML for CIK={fund_cik} acc={accession_number}"
            )

        period_date = (
            filing_meta["period_date"] if filing_meta and filing_meta.get("period_date")
            else date.today()
        )
        return self._parse_xml_holdings(
            xml_text, fund_cik, fund_name, period_date, filing_date
        )

    async def get_latest_holdings(self, fund_cik: str) -> NPORTReport:
        filings = await self.get_fund_filings(fund_cik, n_months=3)
        if not filings:
            raise ValueError(f"No N-PORT filings found for CIK {fund_cik}")
        return await self.parse_nport_filing(fund_cik, filings[0]["accession_number"])

    async def track_fund_changes(
        self, fund_cik: str, n_periods: int = 3
    ) -> pd.DataFrame:
        filings = await self.get_fund_filings(fund_cik, n_months=n_periods * 2)
        if not filings:
            return pd.DataFrame()

        filings = filings[:n_periods]

        reports: list[NPORTReport] = []
        for filing in filings:
            try:
                report = await self.parse_nport_filing(fund_cik, filing["accession_number"])
                reports.append(report)
            except Exception as exc:
                logger.warning(
                    "Failed to parse filing",
                    cik=fund_cik,
                    accession=filing["accession_number"],
                    error=str(exc),
                )

        if not reports:
            return pd.DataFrame()

        reports.sort(key=lambda r: r.period_date)

        rows: list[dict] = []
        for i, report in enumerate(reports):
            prev_report = reports[i - 1] if i > 0 else None
            prev_map: dict[str, NPORTHolding] = {}
            if prev_report:
                for h in prev_report.holdings:
                    key = h.cusip or h.name
                    prev_map[key] = h

            curr_keys = set()
            for holding in report.holdings:
                key = holding.cusip or holding.name
                curr_keys.add(key)
                prev = prev_map.get(key)
                if prev is None:
                    action = "new" if prev_report else "initial"
                    change = holding.pct_of_nav
                    prev_pct = None
                    prev_val = None
                else:
                    diff = holding.pct_of_nav - prev.pct_of_nav
                    if abs(diff) < 0.001:
                        action = "unchanged"
                    elif diff > 0:
                        action = "increased"
                    else:
                        action = "decreased"
                    change = diff
                    prev_pct = prev.pct_of_nav
                    prev_val = prev.market_value_usd

                rows.append({
                    "name": holding.name,
                    "cusip": holding.cusip,
                    "ticker": holding.ticker,
                    "period": report.period_date,
                    "pct_nav": holding.pct_of_nav,
                    "value_usd": holding.market_value_usd,
                    "prev_pct_nav": prev_pct,
                    "prev_value_usd": prev_val,
                    "change_pct_pts": change,
                    "action": action,
                })

            if prev_report:
                for key, prev_holding in prev_map.items():
                    if key not in curr_keys:
                        rows.append({
                            "name": prev_holding.name,
                            "cusip": prev_holding.cusip,
                            "ticker": prev_holding.ticker,
                            "period": report.period_date,
                            "pct_nav": 0.0,
                            "value_usd": 0.0,
                            "prev_pct_nav": prev_holding.pct_of_nav,
                            "prev_value_usd": prev_holding.market_value_usd,
                            "change_pct_pts": -prev_holding.pct_of_nav,
                            "action": "exited",
                        })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values(["period", "change_pct_pts"], ascending=[True, False])
        return df.reset_index(drop=True)

    async def find_funds_holding(
        self, ticker: str, n_months_back: int = 3
    ) -> pd.DataFrame:
        end_dt = date.today()
        start_dt = (pd.Timestamp.now() - pd.DateOffset(months=n_months_back)).date()

        search_url = (
            f"{EDGAR_SEARCH}/LATEST/search-index"
            f"?q=%22{ticker}%22"
            f"&dateRange=custom"
            f"&startdt={start_dt.isoformat()}"
            f"&enddt={end_dt.isoformat()}"
            f"&forms=N-PORT"
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(search_url, headers=_HEADERS)
                resp.raise_for_status()
                search_data = resp.json()
            except Exception as exc:
                logger.error("EDGAR full-text search failed", ticker=ticker, error=str(exc))
                return pd.DataFrame()

        hits = search_data.get("hits", {}).get("hits", [])
        if not hits:
            logger.info("No N-PORT search results", ticker=ticker)
            return pd.DataFrame()

        tasks = []
        meta_list = []
        for hit in hits[:20]:
            src = hit.get("_source", {})
            entity_id = src.get("entity_id", "")
            accession = src.get("file_num", "") or src.get("period_of_report", "")
            acc_raw = hit.get("_id", "")

            if not entity_id:
                continue

            acc_parts = acc_raw.split(":")
            if len(acc_parts) >= 2:
                acc_formatted = acc_parts[-1]
            else:
                acc_formatted = acc_raw

            meta_list.append({
                "cik": entity_id,
                "accession": acc_formatted,
                "period": src.get("period_of_report", ""),
                "entity_name": src.get("entity_name", ""),
            })

        rows: list[dict] = []
        for meta in meta_list:
            try:
                report = await self.parse_nport_filing(meta["cik"], meta["accession"])
                for holding in report.holdings:
                    if holding.ticker and holding.ticker.upper() == ticker.upper():
                        rows.append({
                            "fund_name": report.fund_name,
                            "fund_cik": report.fund_cik,
                            "period": report.period_date,
                            "pct_of_nav": holding.pct_of_nav,
                            "value_usd": holding.market_value_usd,
                            "cusip": holding.cusip,
                        })
                        break
            except Exception as exc:
                logger.debug(
                    "find_funds_holding parse error",
                    cik=meta["cik"],
                    error=str(exc),
                )

        if not rows:
            return pd.DataFrame(
                columns=["fund_name", "fund_cik", "period", "pct_of_nav", "value_usd", "cusip"]
            )

        df = pd.DataFrame(rows)
        return df.sort_values("pct_of_nav", ascending=False).reset_index(drop=True)

    async def smart_money_overlap(
        self, fund_ciks: list[str]
    ) -> pd.DataFrame:
        reports: list[NPORTReport] = []
        for cik in fund_ciks:
            try:
                report = await self.get_latest_holdings(cik)
                reports.append(report)
            except Exception as exc:
                logger.warning("smart_money_overlap: failed CIK", cik=cik, error=str(exc))

        if not reports:
            return pd.DataFrame()

        holding_map: dict[str, dict] = {}

        for report in reports:
            for holding in report.holdings:
                key = holding.cusip or holding.ticker or holding.name
                if not key:
                    continue
                if key not in holding_map:
                    holding_map[key] = {
                        "ticker": holding.ticker,
                        "cusip": holding.cusip,
                        "name": holding.name,
                        "fund_names": [],
                        "pct_navs": [],
                    }
                holding_map[key]["fund_names"].append(report.fund_name)
                holding_map[key]["pct_navs"].append(holding.pct_of_nav)

        rows: list[dict] = []
        for key, data in holding_map.items():
            n = len(data["fund_names"])
            rows.append({
                "ticker": data["ticker"],
                "cusip": data["cusip"],
                "name": data["name"],
                "n_funds_holding": n,
                "avg_pct_nav": sum(data["pct_navs"]) / n if n else 0.0,
                "fund_names": ", ".join(data["fund_names"]),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df.sort_values(
            ["n_funds_holding", "avg_pct_nav"], ascending=[False, False]
        ).reset_index(drop=True)

    def _parse_xml_holdings(
        self,
        xml_text: str,
        fund_cik: str,
        fund_name: str,
        period_date: date,
        filing_date: date,
    ) -> NPORTReport:
        try:
            root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
        except ET.ParseError as exc:
            logger.error("N-PORT XML parse error", cik=fund_cik, error=str(exc))
            return NPORTReport(
                fund_cik=fund_cik,
                fund_name=fund_name,
                period_date=period_date,
                filing_date=filing_date,
                n_holdings=0,
                holdings=[],
            )

        gen_info = _find(root, "generalInfo")
        if gen_info is not None:
            rep_pd = _text(gen_info, "repPdDate")
            if rep_pd:
                parsed = _parse_date(rep_pd)
                if parsed:
                    period_date = parsed

        total_assets: Optional[float] = None
        net_assets: Optional[float] = None
        if gen_info is not None:
            total_assets = _float(_text(gen_info, "totalAssets"))
            net_assets = _float(_text(gen_info, "netAssets"))

        if not fund_name:
            series_el = _find(root, "seriesLeiInfo") or _find(root, "seriesInfo")
            if series_el is not None:
                fund_name = _text(series_el, "seriesName") or fund_name
        if not fund_name:
            reg_info = _find(root, "registrantInfo")
            if reg_info is not None:
                fund_name = _text(reg_info, "regName") or fund_name

        holdings: list[NPORTHolding] = []
        for sec_el in _findall(root, "invstOrSec"):
            holding = self._parse_holding_element(sec_el)
            if holding is not None:
                holdings.append(holding)

        logger.info(
            "N-PORT parsed",
            cik=fund_cik,
            period=period_date,
            holdings=len(holdings),
        )

        return NPORTReport(
            fund_cik=fund_cik,
            fund_name=fund_name,
            period_date=period_date,
            filing_date=filing_date,
            total_assets=total_assets,
            net_assets=net_assets,
            n_holdings=len(holdings),
            holdings=holdings,
        )

    def _parse_holding_element(self, el: ET.Element) -> Optional[NPORTHolding]:
        name = _text(el, "name") or ""
        if not name:
            return None

        lei = _text(el, "lei")
        title = _text(el, "title")
        cusip = _text(el, "cusip")
        if cusip == "000000000" or cusip == "N/A":
            cusip = None

        identifiers_el = _find(el, "identifiers")
        ticker: Optional[str] = None
        isin: Optional[str] = None
        if identifiers_el is not None:
            ticker = _text(identifiers_el, "ticker")
            isin_el = _find(identifiers_el, "isin")
            if isin_el is not None:
                isin = isin_el.get("value") or _text(identifiers_el, "isin")
            if not ticker:
                ticker_el = _find(identifiers_el, "ticker")
                if ticker_el is not None:
                    ticker = ticker_el.get("value") or ticker

        balance_str = _text(el, "balance")
        balance = _float(balance_str) or 0.0

        units = _text(el, "units") or "NS"
        currency = _text(el, "curCd") or "USD"

        val_usd_str = _text(el, "valUSD")
        market_value_usd = _float(val_usd_str) or 0.0

        pct_str = _text(el, "pctVal")
        pct_of_nav = _float(pct_str) or 0.0

        asset_cat_raw = _text(el, "assetCat") or "OTH"
        asset_category = ASSET_CAT_MAP.get(asset_cat_raw, asset_cat_raw.lower())

        country = _text(el, "invCountry")

        return NPORTHolding(
            name=name,
            cusip=cusip if cusip else None,
            ticker=ticker if ticker else None,
            lei=lei if lei else None,
            isin=isin if isin else None,
            balance=balance,
            units=units,
            market_value_usd=market_value_usd,
            pct_of_nav=pct_of_nav,
            asset_category=asset_category,
            country=country,
            currency=currency,
        )


async def get_fund_holdings(fund_cik: str) -> NPORTReport:
    parser = NPORTParser()
    return await parser.get_latest_holdings(fund_cik)


async def who_owns(ticker: str) -> pd.DataFrame:
    parser = NPORTParser()
    return await parser.find_funds_holding(ticker)
