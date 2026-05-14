"""13F-HR institutional holdings adapter — parses SEC EDGAR quarterly filings.

Downloads 13F-HR XML infotables for a given investment manager CIK.
Each 13F-HR filing covers one quarter of institutional equity holdings
(positions >= $100M AUM are required to file).

EDGAR namespaces change across filing periods; we strip all namespaces
before parsing to keep the XPath selectors simple and forward-compatible.

Rate-limited to 10 req/sec (EDGAR policy).
"""
from __future__ import annotations
import asyncio
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Optional
import xml.etree.ElementTree as ET

import httpx

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
_RATE_LIMIT = 0.12  # ~8 req/sec, conservative


def _strip_ns(xml_bytes: bytes) -> bytes:
    """Remove all XML namespace declarations and prefixes for simple XPath."""
    text = xml_bytes.decode("utf-8", errors="replace")
    # Remove namespace declarations
    text = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", text)
    # Remove namespace prefixes  e.g. ns1:infoTable → infoTable
    text = re.sub(r"<(/?)[\w]+:([\w]+)", r"<\1\2", text)
    return text.encode("utf-8")


def _safe_decimal(value: str | None) -> Optional[Decimal]:
    if not value or not value.strip():
        return None
    try:
        return Decimal(str(value).strip().replace(",", ""))
    except InvalidOperation:
        return None


def _safe_date(value: str | None) -> Optional[date]:
    if not value or not value.strip():
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_13f_xml(xml_bytes: bytes, manager_cik: str, period: date, filed_date: Optional[date]) -> list[dict]:
    """Parse 13F-HR infotable XML into a list of holding dicts."""
    xml_bytes = _strip_ns(xml_bytes)
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("13F XML parse error", cik=manager_cik, error=str(exc))
        return []

    holdings: list[dict] = []

    # The infotable has <infoTable> rows (case varies across filers)
    for tag in ("infoTable", "infotable", "InfoTable"):
        tables = root.findall(f".//{tag}")
        if tables:
            break

    for table in tables:
        def _txt(t: str) -> Optional[str]:
            el = table.find(t)
            return el.text.strip() if el is not None and el.text else None

        issuer = _txt("nameOfIssuer") or _txt("nameofissuer") or ""
        cusip = _txt("cusip") or ""
        value_str = _txt("value") or ""
        # EDGAR 13F value is in thousands of dollars
        market_value_thousands = _safe_decimal(value_str)
        market_value = market_value_thousands * 1000 if market_value_thousands else None

        # Shares can be under shrsOrPrnAmt/sshPrnamt or sshPrnamtType
        shares_el = table.find("shrsOrPrnAmt") or table.find("shrsorprnamt")
        shares = None
        share_type = ""
        if shares_el is not None:
            shares = _safe_decimal(
                (shares_el.findtext("sshPrnamt") or shares_el.findtext("sshprnamt") or "").strip()
            )
            share_type = (
                shares_el.findtext("sshPrnamtType") or shares_el.findtext("sshprnamt_type") or ""
            ).strip()

        put_call = (_txt("putCall") or _txt("putcall") or "").strip() or None
        inv_disc = (_txt("investmentDiscretion") or _txt("investmentdiscretion") or "").strip()

        if not cusip and not issuer:
            continue

        holdings.append({
            "manager_cik": manager_cik,
            "issuer_name": issuer[:256],
            "cusip": cusip[:9] or None,
            "ticker": None,  # resolved later via OpenFIGI or instruments table
            "figi": None,
            "period_of_report": period,
            "filed_date": filed_date,
            "market_value": market_value,
            "shares": shares,
            "share_type": share_type[:5] or None,
            "put_call": put_call[:4] if put_call else None,
            "investment_discretion": inv_disc[:10] or None,
        })

    return holdings


class InstitutionalAdapter:
    """Fetches and parses 13F-HR filings from EDGAR for an investment manager."""

    name = "institutional"

    def __init__(self, user_agent: str = "Sentinel sentinel@example.com") -> None:
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    async def _get(self, url: str, host_header: str = "data.sec.gov") -> httpx.Response:
        headers = {**self._headers, "Host": host_header}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp

    async def fetch_13f_filings(
        self,
        manager_cik: str,
        limit: int = 8,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Get list of 13F-HR filings for a manager CIK. Default: last 8 quarters (2 years)."""
        cik = manager_cik.zfill(10)
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
        try:
            await asyncio.sleep(_RATE_LIMIT)
            resp = await self._get(url)
            data = resp.json()
        except Exception as exc:
            logger.error("13F submissions fetch error", cik=cik, error=str(exc))
            return []

        filings = data.get("filings", {}).get("recent", {})
        if not filings:
            return []

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        periods = filings.get("reportDate", [])

        results = []
        for i, form in enumerate(forms):
            if form not in ("13F-HR", "13F-HR/A"):
                continue
            filing_date_str = dates[i] if i < len(dates) else None
            if filing_date_str and since:
                try:
                    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
                    if filing_date < since:
                        continue
                except ValueError:
                    pass
            accession = accessions[i] if i < len(accessions) else None
            period_str = periods[i] if i < len(periods) else None
            if not accession:
                continue
            results.append({
                "manager_cik": cik,
                "form_type": form,
                "filing_date": filing_date_str,
                "period": period_str,
                "accession": accession,
            })
            if len(results) >= limit:
                break
        return results

    async def _find_infotable_url(
        self, cik_numeric: str, accession: str
    ) -> Optional[str]:
        """Find the infotable XML URL from the filing index page."""
        accession_nodash = accession.replace("-", "")
        index_url = f"{EDGAR_ARCHIVES}/{cik_numeric}/{accession_nodash}/{accession}-index.htm"
        try:
            await asyncio.sleep(_RATE_LIMIT)
            resp = await self._get(index_url, host_header="www.sec.gov")
            html = resp.text
            # Look for the XML infotable file
            for pattern in [
                r'href="([^"]*infotable[^"]*\.xml)"',
                r'href="([^"]*form13fInfoTable[^"]*\.xml)"',
                r'href="([^"]*13f[^"]*\.xml)"',
            ]:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    filename = match.group(1)
                    if filename.startswith("http"):
                        return filename
                    return f"{EDGAR_ARCHIVES}/{cik_numeric}/{accession_nodash}/{filename}"
        except Exception as exc:
            logger.warning("13F index fetch error", accession=accession, error=str(exc))
        return None

    async def fetch_holdings(
        self,
        manager_cik: str,
        limit: int = 8,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Download and parse 13F-HR XMLs. Returns flat list of all holdings across filings."""
        filings = await self.fetch_13f_filings(manager_cik, limit=limit, since=since)
        if not filings:
            return []

        cik_numeric = manager_cik.lstrip("0")
        all_holdings: list[dict] = []

        for filing in filings:
            accession = filing["accession"]
            period_date = _safe_date(filing.get("period"))
            filed_date = _safe_date(filing.get("filing_date"))

            if period_date is None:
                continue

            xml_url = await self._find_infotable_url(cik_numeric, accession)
            if not xml_url:
                logger.warning("No infotable XML found", accession=accession)
                continue

            try:
                await asyncio.sleep(_RATE_LIMIT)
                resp = await self._get(xml_url, host_header="www.sec.gov")
                holdings = _parse_13f_xml(resp.content, manager_cik, period_date, filed_date)
                all_holdings.extend(holdings)
                logger.info("13F parsed",
                            cik=manager_cik, period=period_date.isoformat(),
                            holdings=len(holdings))
            except Exception as exc:
                logger.warning("13F XML download error",
                               cik=manager_cik, accession=accession, error=str(exc))

        return all_holdings
