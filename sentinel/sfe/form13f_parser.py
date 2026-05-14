"""13F-HR parser — institutional holdings from EDGAR XML filings."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from datetime import date
from decimal import Decimal
from typing import Optional
import httpx
from sentinel.core.types import InstitutionalHolding
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# 13F XML namespace
NS = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}


def parse_13f_xml(
    xml_content: str,
    manager_cik: str,
    period_of_report: date,
    filed_date: Optional[date] = None,
) -> list[InstitutionalHolding]:
    """Parse 13F information table XML into InstitutionalHolding records."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        logger.error("13F XML parse error", cik=manager_cik, error=str(exc))
        return []

    holdings: list[InstitutionalHolding] = []
    for entry in root.findall(".//ns:infoTable", NS) or root.findall(".//infoTable"):
        holding = _parse_info_table_entry(
            entry, manager_cik, period_of_report, filed_date
        )
        if holding:
            holdings.append(holding)

    logger.info("13F parsed", cik=manager_cik, holdings=len(holdings), period=period_of_report)
    return holdings


def _parse_info_table_entry(
    el: ET.Element,
    manager_cik: str,
    period: date,
    filed_date: Optional[date],
) -> Optional[InstitutionalHolding]:
    def t(tag: str) -> Optional[str]:
        # Try with and without namespace
        found = el.find(f"ns:{tag}", NS) or el.find(tag)
        return found.text.strip() if found is not None and found.text else None

    name = t("nameOfIssuer") or ""
    cusip = t("cusip") or ""
    ticker_str = t("ticker") or ""

    value_str = t("value") or "0"
    shares_str = t("sshPrnamt") or "0"
    share_type = t("sshPrnamtType") or "SH"
    put_call = t("putCall")
    investment_discretion = t("investmentDiscretion") or "SOLE"
    voting_authority_sole = t("Sole") or t("votingAuthority/Sole") or "0"

    try:
        market_value = Decimal(value_str) * 1000  # 13F reports in thousands
        shares = Decimal(shares_str)
    except Exception:
        return None

    return InstitutionalHolding(
        manager_cik=manager_cik,
        issuer_name=name,
        cusip=cusip,
        ticker=ticker_str,
        figi="",  # Resolved later via SIM
        period_of_report=period,
        filed_date=filed_date or period,
        market_value=market_value,
        shares=shares,
        share_type=share_type,
        put_call=put_call,
        investment_discretion=investment_discretion,
    )


async def fetch_13f_holdings(
    manager_cik: str,
    accession: str,
    user_agent: str,
    period_of_report: date,
) -> list[InstitutionalHolding]:
    """Fetch the 13F information table XML and parse it."""
    acc_clean = accession.replace("-", "")
    cik_stripped = manager_cik.lstrip("0")

    # Try common naming conventions for the info table document
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_clean}"
    candidate_names = [
        f"{accession}-index.htm",
        "infotable.xml",
        "form13fInfoTable.xml",
        "primary_doc.xml",
    ]

    headers = {"User-Agent": user_agent}
    async with httpx.AsyncClient(timeout=60) as client:
        # First, get the index to find the actual info table document name
        try:
            index_url = f"{base_url}/{accession}-index.json"
            resp = await client.get(index_url, headers=headers)
            if resp.status_code == 200:
                index = resp.json()
                for doc in index.get("directory", {}).get("item", []):
                    name = doc.get("name", "")
                    if "infotable" in name.lower() or name.endswith(".xml"):
                        candidate_names.insert(0, name)
                        break
        except Exception:
            pass

        for doc_name in candidate_names:
            try:
                url = f"{base_url}/{doc_name}"
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200 and "<infoTable>" in resp.text or "infoTable" in resp.text:
                    return parse_13f_xml(resp.text, manager_cik, period_of_report)
            except Exception:
                continue

    logger.warning("13F info table not found", cik=manager_cik, accession=accession)
    return []


def compute_qoq_changes(
    current: list[InstitutionalHolding],
    prior: list[InstitutionalHolding],
) -> list[dict]:
    """
    Compute quarter-over-quarter position changes for a manager.
    Returns list of {cusip, ticker, current_shares, prior_shares, change_shares, change_pct, signal}.
    """
    prior_map = {h.cusip: h for h in prior if h.cusip}
    results = []
    for holding in current:
        if not holding.cusip:
            continue
        prev = prior_map.get(holding.cusip)
        if prev:
            change = holding.shares - prev.shares
            pct = float(change / prev.shares * 100) if prev.shares else 0.0
        else:
            change = holding.shares
            pct = 100.0  # New position

        signal = "new" if prev is None else ("add" if change > 0 else "trim" if change < 0 else "hold")
        results.append({
            "cusip": holding.cusip,
            "ticker": holding.ticker,
            "issuer": holding.issuer_name,
            "current_shares": float(holding.shares),
            "prior_shares": float(prev.shares) if prev else 0.0,
            "change_shares": float(change),
            "change_pct": pct,
            "current_value": float(holding.market_value),
            "signal": signal,
        })

    # Include exits (positions in prior but not current)
    current_cusips = {h.cusip for h in current}
    for holding in prior:
        if holding.cusip and holding.cusip not in current_cusips:
            results.append({
                "cusip": holding.cusip,
                "ticker": holding.ticker,
                "issuer": holding.issuer_name,
                "current_shares": 0.0,
                "prior_shares": float(holding.shares),
                "change_shares": float(-holding.shares),
                "change_pct": -100.0,
                "current_value": 0.0,
                "signal": "exit",
            })

    return sorted(results, key=lambda x: abs(x["change_pct"]), reverse=True)
