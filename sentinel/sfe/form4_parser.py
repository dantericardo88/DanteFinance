"""Form 4 parser — SEC insider transaction filings → InsiderTransaction records."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from datetime import date
from decimal import Decimal
from typing import Optional
import httpx
from sentinel.core.types import InsiderTransaction, InsiderRole, InsiderTxCode
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_ARCHIVES = "https://www.archives.sec.gov/cgi-bin/browse-edgar"
EDGAR_BASE = "https://www.sec.gov/Archives/edgar/data"


def parse_form4_xml(xml_content: str, cik: str, figi: Optional[str] = None) -> list[InsiderTransaction]:
    """Parse Form 4 XML into InsiderTransaction records."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        logger.error("Form 4 XML parse error", cik=cik, error=str(exc))
        return []

    ns = {"": ""}  # Form 4 uses no namespace

    # Reporting owner info
    owner_el = root.find(".//reportingOwner")
    if owner_el is None:
        return []

    owner_name = _text(owner_el, ".//rptOwnerName") or ""
    owner_cik = _text(owner_el, ".//rptOwnerCik") or ""
    is_director = _text(owner_el, ".//isDirector") == "1"
    is_officer = _text(owner_el, ".//isOfficer") == "1"
    is_ten_pct = _text(owner_el, ".//isTenPercentOwner") == "1"
    officer_title = _text(owner_el, ".//officerTitle") or ""

    role = _infer_role(is_director, is_officer, is_ten_pct, officer_title)

    issuer_ticker = _text(root, ".//issuerTradingSymbol") or ""
    issuer_cik = _text(root, ".//issuerCik") or cik
    period_str = _text(root, ".//periodOfReport") or ""
    period = _parse_date(period_str)

    transactions: list[InsiderTransaction] = []

    # Non-derivative transactions (open market buys/sells)
    for tx_el in root.findall(".//nonDerivativeTransaction"):
        tx = _parse_nonderivative(
            tx_el, cik=issuer_cik, figi=figi, ticker=issuer_ticker,
            owner_name=owner_name, owner_cik=owner_cik, role=role, period=period
        )
        if tx:
            transactions.append(tx)

    # Derivative transactions (options exercises, RSU vests)
    for tx_el in root.findall(".//derivativeTransaction"):
        tx = _parse_derivative(
            tx_el, cik=issuer_cik, figi=figi, ticker=issuer_ticker,
            owner_name=owner_name, owner_cik=owner_cik, role=role, period=period
        )
        if tx:
            transactions.append(tx)

    logger.info("Form 4 parsed", ticker=issuer_ticker, transactions=len(transactions))
    return transactions


def _parse_nonderivative(
    el: ET.Element, cik: str, figi: Optional[str], ticker: str,
    owner_name: str, owner_cik: str, role: InsiderRole, period: Optional[date]
) -> Optional[InsiderTransaction]:
    security_title = _text(el, ".//securityTitle/value") or "Common Stock"
    tx_date_str = _text(el, ".//transactionDate/value") or ""
    tx_date = _parse_date(tx_date_str)
    code_str = _text(el, ".//transactionCode") or ""
    code = _parse_tx_code(code_str)
    shares_str = _text(el, ".//transactionShares/value") or "0"
    price_str = _text(el, ".//transactionPricePerShare/value") or "0"
    acq_disp = _text(el, ".//transactionAcquiredDisposedCode/value") or "A"
    post_shares_str = _text(el, ".//sharesOwnedFollowingTransaction/value") or "0"

    try:
        shares = Decimal(shares_str)
        price = Decimal(price_str) if price_str else Decimal("0")
        post_shares = Decimal(post_shares_str)
    except Exception:
        return None

    if acq_disp == "D":
        shares = -shares

    return InsiderTransaction(
        cik=cik,
        figi=figi or "",
        ticker=ticker,
        owner_name=owner_name,
        owner_cik=owner_cik,
        role=role,
        security_title=security_title,
        tx_date=tx_date or date.today(),
        tx_code=code,
        shares=shares,
        price_per_share=price,
        value=abs(shares) * price,
        shares_owned_after=post_shares,
        is_derivative=False,
    )


def _parse_derivative(
    el: ET.Element, cik: str, figi: Optional[str], ticker: str,
    owner_name: str, owner_cik: str, role: InsiderRole, period: Optional[date]
) -> Optional[InsiderTransaction]:
    security_title = _text(el, ".//securityTitle/value") or ""
    tx_date_str = _text(el, ".//transactionDate/value") or ""
    tx_date = _parse_date(tx_date_str)
    code_str = _text(el, ".//transactionCode") or ""
    code = _parse_tx_code(code_str)
    underlying_shares_str = _text(el, ".//underlyingSecurityShares/value") or "0"
    exercise_price_str = _text(el, ".//exercisePrice/value") or "0"
    expiry_str = _text(el, ".//expirationDate/value") or ""

    try:
        shares = Decimal(underlying_shares_str)
        price = Decimal(exercise_price_str)
    except Exception:
        return None

    return InsiderTransaction(
        cik=cik,
        figi=figi or "",
        ticker=ticker,
        owner_name=owner_name,
        owner_cik=owner_cik,
        role=role,
        security_title=security_title,
        tx_date=tx_date or date.today(),
        tx_code=code,
        shares=shares,
        price_per_share=price,
        value=shares * price,
        shares_owned_after=Decimal("0"),
        is_derivative=True,
        exercise_price=price,
        expiry_date=_parse_date(expiry_str),
    )


async def fetch_and_parse_form4(
    cik: str, accession: str, user_agent: str, figi: Optional[str] = None
) -> list[InsiderTransaction]:
    """Fetch a Form 4 filing XML from EDGAR and parse it."""
    acc_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_clean}/{accession}.xml"
    headers = {"User-Agent": user_agent}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return parse_form4_xml(resp.text, cik=cik, figi=figi)
    except Exception as exc:
        logger.error("Form 4 fetch error", cik=cik, accession=accession, error=str(exc))
        return []


def _text(el: ET.Element, path: str) -> Optional[str]:
    found = el.find(path)
    return found.text.strip() if found is not None and found.text else None


def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _parse_tx_code(code: str) -> InsiderTxCode:
    mapping = {
        "P": InsiderTxCode.PURCHASE,
        "S": InsiderTxCode.SALE,
        "A": InsiderTxCode.GRANT,
        "M": InsiderTxCode.EXERCISE,
        "G": InsiderTxCode.GIFT,
        "F": InsiderTxCode.WITHHOLD_TAX,
        "D": InsiderTxCode.DISPOSITION,
    }
    return mapping.get(code.upper(), InsiderTxCode.OTHER)


def _infer_role(is_director: bool, is_officer: bool, is_ten_pct: bool, title: str) -> InsiderRole:
    t = title.lower()
    if "ceo" in t or "chief executive" in t:
        return InsiderRole.CEO
    if "cfo" in t or "chief financial" in t:
        return InsiderRole.CFO
    if "coo" in t or "chief operating" in t:
        return InsiderRole.COO
    if "president" in t:
        return InsiderRole.PRESIDENT
    if is_director:
        return InsiderRole.DIRECTOR
    if is_ten_pct:
        return InsiderRole.TEN_PCT_OWNER
    if is_officer:
        return InsiderRole.OTHER_OFFICER
    return InsiderRole.OTHER
