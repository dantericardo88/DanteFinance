"""Form 4 insider transaction adapter — parses SEC EDGAR XML filings.

Downloads Form 4 XML for a given CIK using the EDGAR submissions API,
parses both nonDerivative and derivative transactions, and returns
normalized dicts ready for repository.write_insider_transactions().

Rate-limited to 10 req/sec (EDGAR policy). User-Agent header required.
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
_RATE_LIMIT = 0.1   # 10 req/sec


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
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _elem_text(parent: ET.Element, tag: str) -> Optional[str]:
    """Get text of first matching child, including nested <value> sub-elements."""
    el = parent.find(tag)
    if el is None:
        return None
    val_el = el.find("value")
    if val_el is not None:
        return (val_el.text or "").strip() or None
    return (el.text or "").strip() or None


def _owner_role(relationship: ET.Element) -> str:
    """Extract highest-priority role from reportingOwnerRelationship."""
    is_director = (relationship.findtext("isDirector") or "0").strip() == "1"
    is_officer = (relationship.findtext("isOfficer") or "0").strip() == "1"
    is_tenPct = (relationship.findtext("isTenPercentOwner") or "0").strip() == "1"
    if is_officer:
        title = (relationship.findtext("officerTitle") or "Officer").strip()
        return title[:30] if title else "Officer"
    if is_director:
        return "Director"
    if is_tenPct:
        return "10% Owner"
    return "Other"


def _parse_form4_xml(xml_bytes: bytes, cik: str, ticker: str, figi: str = "") -> list[dict]:
    """Parse raw Form 4 XML bytes into a list of transaction dicts."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("Form 4 XML parse error", cik=cik, error=str(exc))
        return []

    # Issuer info
    issuer = root.find("issuer")
    issuer_symbol = ""
    if issuer is not None:
        issuer_symbol = (issuer.findtext("issuerTradingSymbol") or ticker or "").strip().upper()

    # Reporting owner
    owner_name = ""
    owner_cik = ""
    role = "Unknown"
    owner_el = root.find("reportingOwner")
    if owner_el is not None:
        oid = owner_el.find("reportingOwnerId")
        if oid is not None:
            owner_name = (oid.findtext("rptOwnerName") or "").strip()
            owner_cik = (oid.findtext("rptOwnerCik") or "").strip().zfill(10)
        rel = owner_el.find("reportingOwnerRelationship")
        if rel is not None:
            role = _owner_role(rel)

    transactions: list[dict] = []

    # Non-derivative transactions (common stock purchases/sales)
    for tx in root.findall(".//nonDerivativeTransaction"):
        security_title = _elem_text(tx, "securityTitle") or ""
        tx_date = _safe_date(_elem_text(tx, "transactionDate"))
        if tx_date is None:
            continue

        coding = tx.find("transactionCoding")
        tx_code = ""
        if coding is not None:
            tx_code = (coding.findtext("transactionCode") or "").strip()

        amounts = tx.find("transactionAmounts")
        shares = price = None
        acquired_disposed = ""
        if amounts is not None:
            shares = _safe_decimal(_elem_text(amounts, "transactionShares"))
            price = _safe_decimal(_elem_text(amounts, "transactionPricePerShare"))
            acquired_disposed = (_elem_text(amounts, "transactionAcquiredDisposedCode") or "").upper()

        post = tx.find("postTransactionAmounts")
        shares_after = None
        if post is not None:
            shares_after = _safe_decimal(_elem_text(post, "sharesOwnedFollowingTransaction"))

        if shares is None:
            continue

        # Normalize: disposals become negative share counts
        if acquired_disposed == "D" and shares and shares > 0:
            shares = -shares

        value = (abs(shares) * price) if shares and price else None

        transactions.append({
            "cik": cik,
            "figi": figi,
            "ticker": issuer_symbol or ticker,
            "owner_name": owner_name[:200],
            "owner_cik": owner_cik or None,
            "role": role[:30],
            "security_title": security_title[:100],
            "tx_date": tx_date,
            "tx_code": tx_code[:20],
            "shares": shares,
            "price_per_share": price,
            "value": value,
            "shares_owned_after": shares_after,
            "is_derivative": False,
            "exercise_price": None,
            "expiry_date": None,
        })

    # Derivative transactions (options, warrants)
    for tx in root.findall(".//derivativeTransaction"):
        security_title = _elem_text(tx, "securityTitle") or ""
        tx_date = _safe_date(_elem_text(tx, "transactionDate"))
        if tx_date is None:
            continue

        coding = tx.find("transactionCoding")
        tx_code = ""
        if coding is not None:
            tx_code = (coding.findtext("transactionCode") or "").strip()

        amounts = tx.find("transactionAmounts")
        shares = price = None
        acquired_disposed = ""
        if amounts is not None:
            shares = _safe_decimal(_elem_text(amounts, "transactionShares"))
            price = _safe_decimal(_elem_text(amounts, "transactionPricePerShare"))
            acquired_disposed = (_elem_text(amounts, "transactionAcquiredDisposedCode") or "").upper()

        exercise_price = _safe_decimal(_elem_text(tx, "conversionOrExercisePrice"))
        expiry_date = _safe_date(_elem_text(tx, "expirationDate"))

        post = tx.find("postTransactionAmounts")
        shares_after = None
        if post is not None:
            shares_after = _safe_decimal(
                _elem_text(post, "sharesOwnedFollowingTransaction")
                or _elem_text(post, "derivativeSecuritiesOwnedFollowingTransaction")
            )

        if shares is None:
            continue

        if acquired_disposed == "D" and shares and shares > 0:
            shares = -shares

        transactions.append({
            "cik": cik,
            "figi": figi,
            "ticker": issuer_symbol or ticker,
            "owner_name": owner_name[:200],
            "owner_cik": owner_cik or None,
            "role": role[:30],
            "security_title": security_title[:100],
            "tx_date": tx_date,
            "tx_code": tx_code[:20],
            "shares": shares,
            "price_per_share": price,
            "value": (abs(shares) * price) if shares and price else None,
            "shares_owned_after": shares_after,
            "is_derivative": True,
            "exercise_price": exercise_price,
            "expiry_date": expiry_date,
        })

    return transactions


class InsiderAdapter:
    """Fetches and parses Form 4 filings from EDGAR for a given issuer CIK."""

    name = "insider"

    def __init__(self, user_agent: str = "Sentinel sentinel@example.com") -> None:
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    async def _get(self, url: str, host_header: str = "data.sec.gov") -> httpx.Response:
        headers = {**self._headers, "Host": host_header}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp

    async def fetch_form4_filings(
        self,
        cik: str,
        limit: int = 40,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Get list of Form 4 filing metadata for an issuer CIK."""
        cik = cik.zfill(10)
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
        try:
            await asyncio.sleep(_RATE_LIMIT)
            resp = await self._get(url)
            data = resp.json()
        except Exception as exc:
            logger.error("Form 4 submissions fetch error", cik=cik, error=str(exc))
            return []

        filings = data.get("filings", {}).get("recent", {})
        if not filings:
            return []

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form not in ("4", "4/A"):
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
            if not accession:
                continue
            results.append({
                "cik": cik,
                "form_type": form,
                "filing_date": filing_date_str,
                "accession": accession,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
            })
            if len(results) >= limit:
                break
        return results

    async def fetch_transactions(
        self,
        cik: str,
        ticker: str,
        figi: str = "",
        limit: int = 40,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Download and parse Form 4 XMLs for all recent filings.

        Returns a flat list of transaction dicts for all Form 4s found.
        """
        filings = await self.fetch_form4_filings(cik, limit=limit, since=since)
        if not filings:
            return []

        all_transactions: list[dict] = []
        cik_numeric = cik.lstrip("0")

        for filing in filings:
            accession = filing["accession"]
            primary_doc = filing.get("primary_doc", "")
            if not primary_doc:
                continue

            accession_nodash = accession.replace("-", "")
            xml_url = f"{EDGAR_ARCHIVES}/{cik_numeric}/{accession_nodash}/{primary_doc}"

            try:
                await asyncio.sleep(_RATE_LIMIT)
                resp = await self._get(xml_url, host_header="www.sec.gov")
                content_type = resp.headers.get("content-type", "")
                # Only parse XML files; skip HTML wrappers
                if "xml" not in content_type and not primary_doc.endswith((".xml", ".XML")):
                    continue
                txs = _parse_form4_xml(resp.content, cik, ticker, figi)
                all_transactions.extend(txs)
            except Exception as exc:
                logger.warning("Form 4 XML download error",
                               cik=cik, accession=accession, error=str(exc))
                continue

        logger.info("Form 4 parsed",
                    cik=cik, ticker=ticker, filings=len(filings),
                    transactions=len(all_transactions))
        return all_transactions
