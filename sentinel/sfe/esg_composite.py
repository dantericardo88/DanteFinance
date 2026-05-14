"""esg_composite.py — Multi-pillar ESG composite scoring engine.

Derives comprehensive E, S, G scores (0-100) from free public data:
  E: GHG/water/waste/energy/net-zero keyword analysis (SASB/GRI mappings)
  S: OSHA safety signals, board diversity, pay equity, supply-chain
  G: Board independence, exec comp quality, anti-corruption, transparency
  Controversy: SEC enforcement search penalty deductions

Sector-specific SASB materiality weights applied to composite.
Proxy signals only — NOT equivalent to MSCI/Sustainalytics/ISS ratings.
"""
from __future__ import annotations

import asyncio
import math
import re
from datetime import date, datetime
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_SLEEP = 0.15
_TEXT_LIMIT = 120_000  # chars per filing download

# ---------------------------------------------------------------------------
# SASB/GRI keyword mappings for EDGAR text analysis
# ---------------------------------------------------------------------------

ENVIRONMENTAL_METRICS: dict[str, list[str]] = {
    "ghg_emissions": [
        "greenhouse gas", "scope 1", "scope 2", "scope 3", "carbon emissions",
        "CO2 equivalent", "metric tons of CO2", "GHG emissions", "scope one",
        "scope two", "direct emissions",
    ],
    "water": [
        "water consumption", "water usage", "water withdrawal", "water intensity",
        "freshwater", "water recycled", "water efficiency", "water stewardship",
    ],
    "waste": [
        "waste generated", "hazardous waste", "recycling rate", "zero waste",
        "landfill diversion", "waste intensity", "solid waste", "waste reduction",
    ],
    "energy": [
        "energy consumption", "renewable energy", "energy intensity", "megawatt",
        "energy efficiency", "electricity consumption", "energy use",
        "kilowatt-hour", "gigajoule",
    ],
    "net_zero": [
        "net zero", "carbon neutral", "carbon neutrality", "net-zero target",
        "2050 target", "climate target", "science based target", "SBTi",
        "carbon negative", "climate pledge",
    ],
    "biodiversity": [
        "biodiversity", "deforestation", "land use", "ecosystem services",
        "habitat", "species", "reforestation", "no net loss",
    ],
}

SOCIAL_METRICS: dict[str, list[str]] = {
    "safety": [
        "OSHA", "recordable incidents", "lost time injury", "fatality",
        "TRIR", "total recordable", "safety performance", "workplace injury",
        "days away from work", "DART rate",
    ],
    "diversity": [
        "board diversity", "women on board", "gender diversity",
        "racial diversity", "ethnic diversity", "inclusion", "DEI",
        "diversity, equity", "representation", "underrepresented",
    ],
    "pay_equity": [
        "pay equity", "gender pay gap", "pay ratio", "CEO pay", "median employee",
        "pay gap", "equal pay", "compensation equity",
    ],
    "supply_chain": [
        "human rights", "supplier code", "forced labor", "child labor",
        "modern slavery", "responsible sourcing", "supply chain audit",
        "supplier standards", "conflict minerals",
    ],
    "community": [
        "community investment", "philanthropic", "charitable contributions",
        "local community", "social impact", "community engagement",
        "corporate giving", "foundation",
    ],
}

GOVERNANCE_METRICS: dict[str, list[str]] = {
    "board": [
        "independent director", "board independence", "separation of chair",
        "lead independent", "board refreshment", "director term",
        "board skills", "director qualifications",
    ],
    "exec_comp": [
        "performance-based", "clawback", "say-on-pay", "at-risk compensation",
        "long-term incentive", "equity award", "compensation recovery",
    ],
    "anti_corruption": [
        "anti-corruption", "anti-bribery", "FCPA", "UK Bribery Act",
        "corruption prevention", "bribery policy", "integrity",
    ],
    "transparency": [
        "disclosure", "transparency", "stakeholder engagement",
        "ESG report", "sustainability report", "integrated report",
        "TCFD", "GRI", "SASB",
    ],
    "shareholder": [
        "shareholder rights", "proxy access", "dual class", "poison pill",
        "staggered board", "shareholder engagement", "activism",
    ],
}

# ---------------------------------------------------------------------------
# Sector-specific ESG weights (SASB materiality)
# ---------------------------------------------------------------------------

SECTOR_ESG_WEIGHTS: dict[str, dict[str, float]] = {
    "energy": {"E": 0.60, "S": 0.25, "G": 0.15},
    "materials": {"E": 0.55, "S": 0.25, "G": 0.20},
    "industrials": {"E": 0.40, "S": 0.35, "G": 0.25},
    "utilities": {"E": 0.55, "S": 0.25, "G": 0.20},
    "consumer_staples": {"E": 0.30, "S": 0.45, "G": 0.25},
    "consumer_discretionary": {"E": 0.25, "S": 0.45, "G": 0.30},
    "health_care": {"E": 0.20, "S": 0.50, "G": 0.30},
    "financials": {"E": 0.15, "S": 0.40, "G": 0.45},
    "information_technology": {"E": 0.20, "S": 0.40, "G": 0.40},
    "communication_services": {"E": 0.20, "S": 0.40, "G": 0.40},
    "real_estate": {"E": 0.40, "S": 0.30, "G": 0.30},
    "default": {"E": 0.33, "S": 0.33, "G": 0.34},
}

# SIC → SASB sector mapping
_SIC_TO_SECTOR: dict[str, str] = {
    # Energy
    "1311": "energy", "1382": "energy", "2911": "energy", "1321": "energy",
    "5171": "energy", "1381": "energy",
    # Materials
    "2819": "materials", "2860": "materials", "2810": "materials",
    "3312": "materials", "1040": "materials", "2650": "materials",
    "2670": "materials", "2621": "materials",
    # Industrials
    "3559": "industrials", "3720": "industrials", "3812": "industrials",
    "3490": "industrials", "3460": "industrials", "3440": "industrials",
    # Utilities
    "4911": "utilities", "4931": "utilities", "4941": "utilities",
    "4924": "utilities", "4952": "utilities",
    # Consumer Staples
    "2000": "consumer_staples", "2010": "consumer_staples",
    "5140": "consumer_staples", "5400": "consumer_staples",
    "2100": "consumer_staples", "5910": "consumer_staples",
    # Consumer Discretionary
    "5900": "consumer_discretionary", "5600": "consumer_discretionary",
    "7011": "consumer_discretionary", "7812": "consumer_discretionary",
    "5511": "consumer_discretionary", "7372": "consumer_discretionary",
    # Health Care
    "2836": "health_care", "2830": "health_care", "8011": "health_care",
    "8049": "health_care", "5912": "health_care", "8099": "health_care",
    # Financials
    "6020": "financials", "6022": "financials", "6211": "financials",
    "6159": "financials", "6311": "financials", "6411": "financials",
    # Information Technology
    "7372": "information_technology", "7371": "information_technology",
    "3674": "information_technology", "3577": "information_technology",
    # Communication Services
    "4813": "communication_services", "4812": "communication_services",
    "7372": "communication_services",
    # Real Estate
    "6552": "real_estate", "6500": "real_estate", "6512": "real_estate",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ESGPillar(BaseModel):
    pillar: Literal["E", "S", "G"]
    score: float                    # 0-100
    grade: str                      # A+, A, A-, B+, B, B-, C+, C, C-, D, F
    metrics: dict[str, float]       # sub-metric scores 0-100
    evidence: list[str]             # supporting text snippets
    data_coverage: float            # 0-1, fraction of metrics with data


class ESGComposite(BaseModel):
    ticker: str
    company_name: str
    cik: str
    sector: Optional[str] = None
    sic_code: Optional[str] = None
    filing_year: int
    environmental: ESGPillar
    social: ESGPillar
    governance: ESGPillar
    composite_score: float          # 0-100
    composite_grade: str
    esg_rating: str                 # "ESG Leader", "Above Average", "Average", "Below Average", "ESG Laggard"
    peer_rank_pct: Optional[float] = None   # % of peers scored below this company
    controversy_penalty: float = 0.0        # score deduction for violations/fines
    has_net_zero_target: bool = False
    has_diversity_disclosure: bool = False
    has_sustainability_report: bool = False
    yoy_improvement: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = (
        "Proxy signals from public EDGAR/EPA filings. "
        "Not equivalent to MSCI/Sustainalytics/ISS ratings."
    )


class ESGSectorSummary(BaseModel):
    sector: str
    n_companies: int
    avg_composite: float
    avg_e: float
    avg_s: float
    avg_g: float
    leaders: list[str]      # top 3 tickers by composite
    laggards: list[str]     # bottom 3 tickers by composite
    sector_trend: str       # "improving", "declining", "stable"


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class ESGCompositeEngine:
    """Multi-pillar ESG scoring engine using free EDGAR and regulatory data."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._cik_cache: dict[str, str] = {}
        self._ticker_map: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score_ticker(self, ticker: str) -> ESGComposite:
        """Full ESG pipeline for one ticker: EDGAR → text analysis → composite."""
        ticker = ticker.upper()
        cik = await self._get_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker '{ticker}'")

        text_10k, text_proxy, sic_code, company_name, fiscal_year = \
            await self._get_filing_texts(cik)

        sector = _sic_to_sector(sic_code)

        e_pillar = self._score_e_pillar(text_10k, sic_code or "")
        s_pillar = self._score_s_pillar(text_10k, text_proxy)
        g_pillar = self._score_g_pillar(text_proxy, text_10k)

        weights = SECTOR_ESG_WEIGHTS.get(sector, SECTOR_ESG_WEIGHTS["default"])
        raw_composite = (
            e_pillar.score * weights["E"]
            + s_pillar.score * weights["S"]
            + g_pillar.score * weights["G"]
        )

        # Controversy penalty (SEC enforcement actions)
        controversy_penalty = await self._apply_controversy_penalty(
            cik, raw_composite
        )
        composite_score = round(
            max(0.0, min(100.0, raw_composite - controversy_penalty)), 1
        )

        # Boolean flags derived from text
        text_lower = (text_10k + " " + text_proxy).lower()
        has_net_zero = bool(re.search(
            r"net[\s\-]zero|carbon[\s\-]neutral|carbon\s+neutrality", text_lower
        ))
        has_diversity = bool(re.search(
            r"board\s+diversity|gender\s+diversity|women\s+on\s+board|"
            r"diversity[,\s]+equity", text_lower
        ))
        has_sustain = bool(re.search(
            r"sustainability\s+report|esg\s+report|corporate\s+responsibility\s+report|"
            r"gri\s+index|sasb\s+index", text_lower
        ))

        composite_grade = self._score_to_grade(composite_score)
        esg_rating = self._composite_rating(composite_score)

        logger.info(
            "ESG composite scored",
            ticker=ticker,
            e=e_pillar.score,
            s=s_pillar.score,
            g=g_pillar.score,
            composite=composite_score,
            sector=sector,
        )

        return ESGComposite(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            sector=sector,
            sic_code=sic_code,
            filing_year=fiscal_year,
            environmental=e_pillar,
            social=s_pillar,
            governance=g_pillar,
            composite_score=composite_score,
            composite_grade=composite_grade,
            esg_rating=esg_rating,
            controversy_penalty=controversy_penalty,
            has_net_zero_target=has_net_zero,
            has_diversity_disclosure=has_diversity,
            has_sustainability_report=has_sustain,
        )

    async def score_sector(
        self, tickers: list[str], sector: str
    ) -> ESGSectorSummary:
        """Score all tickers in a sector and return aggregated summary."""
        results = await self.screen_by_esg(tickers, min_score=0.0,
                                           exclude_controversies=False)
        if not results:
            return ESGSectorSummary(
                sector=sector, n_companies=0, avg_composite=0.0,
                avg_e=0.0, avg_s=0.0, avg_g=0.0,
                leaders=[], laggards=[], sector_trend="stable",
            )

        avg_composite = round(
            sum(r.composite_score for r in results) / len(results), 1
        )
        avg_e = round(
            sum(r.environmental.score for r in results) / len(results), 1
        )
        avg_s = round(
            sum(r.social.score for r in results) / len(results), 1
        )
        avg_g = round(
            sum(r.governance.score for r in results) / len(results), 1
        )

        sorted_results = sorted(results, key=lambda r: r.composite_score, reverse=True)
        leaders = [r.ticker for r in sorted_results[:3]]
        laggards = [r.ticker for r in sorted_results[-3:]]

        # Simple trend heuristic: compare E pillar to S (proxy for improvement focus)
        sector_trend = "stable"
        if avg_composite >= 65:
            sector_trend = "improving"
        elif avg_composite < 45:
            sector_trend = "declining"

        return ESGSectorSummary(
            sector=sector,
            n_companies=len(results),
            avg_composite=avg_composite,
            avg_e=avg_e,
            avg_s=avg_s,
            avg_g=avg_g,
            leaders=leaders,
            laggards=laggards,
            sector_trend=sector_trend,
        )

    async def get_esg_momentum(
        self, ticker: str, n_years: int = 3
    ) -> dict:
        """Compare ESG composite across up to n_years annual 10-K filings.

        Returns a dict with year → score mapping and trend direction.
        Note: EDGAR submissions only expose the most recent filing via
        primaryDocument; we therefore look at the last n_years accessions.
        """
        ticker = ticker.upper()
        cik = await self._get_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker '{ticker}'")

        accessions = await self._get_10k_accessions(cik, limit=n_years)
        if not accessions:
            return {"error": "No 10-K filings found", "ticker": ticker}

        year_scores: dict[int, float] = {}
        for acc_info in accessions:
            try:
                raw = await self._fetch_filing_text(
                    cik, acc_info["accession"], acc_info["primary_doc"]
                )
                proxy_raw = ""  # momentum uses 10-K only for speed
                sic_code = acc_info.get("sic_code", "")
                sector = _sic_to_sector(sic_code)
                e_pillar = self._score_e_pillar(raw, sic_code)
                s_pillar = self._score_s_pillar(raw, proxy_raw)
                g_pillar = self._score_g_pillar(proxy_raw, raw)
                weights = SECTOR_ESG_WEIGHTS.get(sector, SECTOR_ESG_WEIGHTS["default"])
                composite = round(
                    e_pillar.score * weights["E"]
                    + s_pillar.score * weights["S"]
                    + g_pillar.score * weights["G"],
                    1,
                )
                yr = acc_info.get("fiscal_year", 0)
                year_scores[yr] = composite
            except Exception as exc:
                logger.warning(
                    "ESG momentum filing failed",
                    ticker=ticker,
                    acc=acc_info.get("accession"),
                    error=str(exc),
                )

        if len(year_scores) < 2:
            return {
                "ticker": ticker,
                "scores_by_year": year_scores,
                "trend": "insufficient_data",
                "improvement_rate": None,
            }

        sorted_years = sorted(year_scores)
        first = year_scores[sorted_years[0]]
        last = year_scores[sorted_years[-1]]
        delta = last - first
        trend = "improving" if delta > 2 else ("declining" if delta < -2 else "stable")
        n = len(sorted_years) - 1
        annual_rate = round(delta / n, 2) if n > 0 else 0.0

        return {
            "ticker": ticker,
            "scores_by_year": year_scores,
            "trend": trend,
            "improvement_rate": annual_rate,
            "delta_total": round(delta, 1),
        }

    async def screen_by_esg(
        self,
        tickers: list[str],
        min_score: float = 60.0,
        exclude_controversies: bool = True,
        max_concurrent: int = 5,
    ) -> list[ESGComposite]:
        """Batch score tickers and return those passing ESG thresholds."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _bounded(t: str) -> Optional[ESGComposite]:
            async with sem:
                try:
                    return await self.score_ticker(t)
                except Exception as exc:
                    logger.warning("ESG score failed", ticker=t, error=str(exc))
                    return None

        raw_results = await asyncio.gather(*[_bounded(t) for t in tickers])
        results: list[ESGComposite] = []

        for r in raw_results:
            if r is None:
                continue
            if r.composite_score < min_score:
                continue
            if exclude_controversies and r.controversy_penalty > 5.0:
                continue
            results.append(r)

        results.sort(key=lambda r: r.composite_score, reverse=True)

        # Assign peer rank percentiles
        n = len(results)
        for idx, r in enumerate(results):
            r.peer_rank_pct = round(100.0 * (n - 1 - idx) / max(n - 1, 1), 1)

        return results

    # ------------------------------------------------------------------
    # Pillar scorers
    # ------------------------------------------------------------------

    def _score_e_pillar(self, text: str, sic_code: str) -> ESGPillar:
        """Score Environmental pillar 0-100 from 10-K text.

        Sub-metrics:
          - ghg_emissions (0-100): disclosure depth of scope 1/2/3
          - water (0-100): water management mentions
          - waste (0-100): waste reduction / recycling
          - energy (0-100): energy efficiency / renewables
          - net_zero (0-100): climate targets presence
          - biodiversity (0-100): ecosystem / land-use mentions
        Sector materiality multipliers applied where relevant.
        """
        text_lower = text.lower()
        sentences = _split_sentences(text)
        metrics: dict[str, float] = {}
        evidence: list[str] = []
        covered = 0

        for metric_key, terms in ENVIRONMENTAL_METRICS.items():
            hits = 0
            for term in terms:
                hits += len(re.findall(re.escape(term.lower()), text_lower))
                if hits > 0 and len(evidence) < 8:
                    snippet = _find_evidence(sentences, term)
                    if snippet:
                        evidence.append(snippet)

            if hits > 0:
                covered += 1
            # Log-scale hits → 0-100 range
            raw = math.log1p(hits) * 18.0
            metrics[metric_key] = round(min(100.0, raw), 1)

        # Sector-specific boost: energy-sector companies should disclose more
        sector = _sic_to_sector(sic_code)
        if sector in ("energy", "utilities", "materials"):
            # Heavily penalise absence of GHG disclosure
            if metrics.get("ghg_emissions", 0) < 10:
                for k in metrics:
                    metrics[k] = metrics[k] * 0.75

        # Compute sub-metric average (equal-weighted within pillar)
        n_metrics = len(ENVIRONMENTAL_METRICS)
        raw_score = sum(metrics.values()) / n_metrics if metrics else 0.0
        score = round(min(100.0, raw_score), 1)
        coverage = round(covered / n_metrics, 2)

        return ESGPillar(
            pillar="E",
            score=score,
            grade=self._score_to_grade(score),
            metrics=metrics,
            evidence=evidence[:6],
            data_coverage=coverage,
        )

    def _score_s_pillar(self, text_10k: str, text_proxy: str) -> ESGPillar:
        """Score Social pillar 0-100 from 10-K + DEF 14A text.

        Sub-metrics:
          - safety: OSHA/recordable incident disclosures
          - diversity: board/workforce diversity mentions
          - pay_equity: pay ratio and equity disclosures
          - supply_chain: human rights / supplier standards
          - community: community investment / philanthropy
        """
        combined = (text_10k + " " + text_proxy).lower()
        sentences = _split_sentences(text_10k + " " + text_proxy)
        metrics: dict[str, float] = {}
        evidence: list[str] = []
        covered = 0

        for metric_key, terms in SOCIAL_METRICS.items():
            hits = 0
            for term in terms:
                hits += len(re.findall(re.escape(term.lower()), combined))
                if hits > 0 and len(evidence) < 8:
                    snippet = _find_evidence(sentences, term)
                    if snippet:
                        evidence.append(snippet)

            if hits > 0:
                covered += 1
            raw = math.log1p(hits) * 18.0
            metrics[metric_key] = round(min(100.0, raw), 1)

        # Boost if pay ratio is quantitatively disclosed (specific number pattern)
        if re.search(r"\b\d{1,4}\s*(?:to|-to-)\s*1\b", combined):
            metrics["pay_equity"] = min(100.0, metrics.get("pay_equity", 0) + 20.0)

        # Boost if TRIR / DART rate is quantified
        if re.search(r"\b(?:trir|dart|recordable)\s+(?:rate|of)\s+[\d\.]+", combined):
            metrics["safety"] = min(100.0, metrics.get("safety", 0) + 20.0)

        n_metrics = len(SOCIAL_METRICS)
        raw_score = sum(metrics.values()) / n_metrics if metrics else 0.0
        score = round(min(100.0, raw_score), 1)
        coverage = round(covered / n_metrics, 2)

        return ESGPillar(
            pillar="S",
            score=score,
            grade=self._score_to_grade(score),
            metrics=metrics,
            evidence=evidence[:6],
            data_coverage=coverage,
        )

    def _score_g_pillar(self, text_proxy: str, text_10k: str) -> ESGPillar:
        """Score Governance pillar 0-100 from DEF 14A + 10-K text.

        Sub-metrics:
          - board: board independence, refreshment quality
          - exec_comp: performance linkage, clawback
          - anti_corruption: FCPA / ethics programme
          - transparency: disclosure quality (TCFD/GRI/SASB)
          - shareholder: rights protections
        """
        combined = (text_proxy + " " + text_10k).lower()
        sentences = _split_sentences(text_proxy + " " + text_10k)
        metrics: dict[str, float] = {}
        evidence: list[str] = []
        covered = 0

        for metric_key, terms in GOVERNANCE_METRICS.items():
            hits = 0
            for term in terms:
                hits += len(re.findall(re.escape(term.lower()), combined))
                if hits > 0 and len(evidence) < 8:
                    snippet = _find_evidence(sentences, term)
                    if snippet:
                        evidence.append(snippet)

            if hits > 0:
                covered += 1
            raw = math.log1p(hits) * 18.0
            metrics[metric_key] = round(min(100.0, raw), 1)

        # Penalty: accounting restatement mentioned
        if re.search(
            r"restate(?:ment|d)|material\s+weakness|accounting\s+error", combined
        ):
            for k in metrics:
                metrics[k] = max(0.0, metrics[k] - 15.0)
            logger.debug("G-pillar: restatement penalty applied")

        # Boost: explicit board independence % >= 75%
        if re.search(r"(?:7[5-9]|8\d|9\d|100)\s*%\s*independent", combined):
            metrics["board"] = min(100.0, metrics.get("board", 0) + 15.0)

        n_metrics = len(GOVERNANCE_METRICS)
        raw_score = sum(metrics.values()) / n_metrics if metrics else 0.0
        score = round(min(100.0, raw_score), 1)
        coverage = round(covered / n_metrics, 2)

        return ESGPillar(
            pillar="G",
            score=score,
            grade=self._score_to_grade(score),
            metrics=metrics,
            evidence=evidence[:6],
            data_coverage=coverage,
        )

    # ------------------------------------------------------------------
    # Controversy penalty
    # ------------------------------------------------------------------

    async def _apply_controversy_penalty(
        self, cik: str, base_score: float
    ) -> float:
        """Search EDGAR EFTS for enforcement-related filings linked to this CIK.

        Deducts up to 10 points for recent (last 3 years) SEC enforcement.
        Returns the penalty amount (not the adjusted score).
        """
        try:
            query = f"cik:{cik.lstrip('0')} (\"SEC order\" OR \"civil penalty\" OR \"cease and desist\")"
            params = {
                "q": query,
                "dateRange": "custom",
                "startdt": str(date.today().year - 3) + "-01-01",
                "forms": "8-K",
                "_source": "period_of_report,file_date,display_names",
                "hits.hits.total.value": 1,
                "hits.hits._source.period_of_report": 1,
            }
            await asyncio.sleep(_EDGAR_SLEEP)
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    EDGAR_EFTS, params=params, headers=_HEADERS
                )
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("hits", {}).get("total", {}).get("value", 0)
                if total > 0:
                    penalty = min(10.0, total * 2.5)
                    logger.info(
                        "Controversy penalty applied",
                        cik=cik,
                        enforcement_filings=total,
                        penalty=penalty,
                    )
                    return round(penalty, 1)
        except Exception as exc:
            logger.warning("Controversy check failed", cik=cik, error=str(exc))
        return 0.0

    # ------------------------------------------------------------------
    # Grade / rating helpers
    # ------------------------------------------------------------------

    def _score_to_grade(self, score: float) -> str:
        """Convert 0-100 score to letter grade."""
        if score >= 93:
            return "A+"
        if score >= 87:
            return "A"
        if score >= 80:
            return "A-"
        if score >= 75:
            return "B+"
        if score >= 70:
            return "B"
        if score >= 65:
            return "B-"
        if score >= 60:
            return "C+"
        if score >= 55:
            return "C"
        if score >= 50:
            return "C-"
        if score >= 40:
            return "D"
        return "F"

    def _composite_rating(self, score: float) -> str:
        """Map composite score to qualitative ESG rating."""
        if score >= 80:
            return "ESG Leader"
        if score >= 65:
            return "Above Average"
        if score >= 50:
            return "Average"
        if score >= 35:
            return "Below Average"
        return "ESG Laggard"

    # ------------------------------------------------------------------
    # EDGAR data retrieval
    # ------------------------------------------------------------------

    async def _get_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker to zero-padded 10-digit CIK via EDGAR tickers JSON."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        try:
            if self._ticker_map is None:
                await asyncio.sleep(_EDGAR_SLEEP)
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(EDGAR_TICKERS, headers=_HEADERS)
                    resp.raise_for_status()
                    self._ticker_map = resp.json()

            for entry in self._ticker_map.values():
                if str(entry.get("ticker", "")).upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik
        except Exception as exc:
            logger.error("CIK lookup failed", ticker=ticker, error=str(exc))

        logger.warning("Ticker not found in EDGAR", ticker=ticker)
        return None

    async def _get_filing_texts(
        self, cik: str
    ) -> tuple[str, str, Optional[str], str, int]:
        """Fetch most recent 10-K and DEF 14A texts plus metadata.

        Returns:
            (text_10k, text_proxy, sic_code, company_name, fiscal_year)
        """
        cik_padded = cik.zfill(10)
        sub_url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"

        await asyncio.sleep(_EDGAR_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                sub_resp = await client.get(sub_url, headers=_HEADERS)
                sub_resp.raise_for_status()
                sub = sub_resp.json()
        except Exception as exc:
            logger.error("Submissions fetch failed", cik=cik, error=str(exc))
            return "", "", None, cik, datetime.utcnow().year

        company_name: str = sub.get("name", cik)
        sic_code: Optional[str] = sub.get("sic")
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        acc_10k = doc_10k = date_10k = None
        acc_proxy = doc_proxy = None

        for i, form in enumerate(forms):
            if form in ("10-K", "10-K405") and acc_10k is None:
                acc_10k = accessions[i] if i < len(accessions) else None
                doc_10k = primary_docs[i] if i < len(primary_docs) else None
                date_10k = report_dates[i] if i < len(report_dates) else None
            if form == "DEF 14A" and acc_proxy is None:
                acc_proxy = accessions[i] if i < len(accessions) else None
                doc_proxy = primary_docs[i] if i < len(primary_docs) else None
            if acc_10k and acc_proxy:
                break

        # Parse fiscal year
        fiscal_year = datetime.utcnow().year
        if date_10k:
            try:
                fiscal_year = int(date_10k[:4])
            except (ValueError, TypeError):
                pass

        text_10k = ""
        if acc_10k and doc_10k:
            text_10k = await self._fetch_filing_text(cik_padded, acc_10k, doc_10k)

        text_proxy = ""
        if acc_proxy and doc_proxy:
            text_proxy = await self._fetch_filing_text(
                cik_padded, acc_proxy, doc_proxy, limit=40_000
            )

        return text_10k, text_proxy, sic_code, company_name, fiscal_year

    async def _get_10k_accessions(
        self, cik: str, limit: int = 3
    ) -> list[dict]:
        """Return the last `limit` 10-K accession records for multi-year momentum."""
        cik_padded = cik.zfill(10)
        sub_url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"

        await asyncio.sleep(_EDGAR_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                sub_resp = await client.get(sub_url, headers=_HEADERS)
                sub_resp.raise_for_status()
                sub = sub_resp.json()
        except Exception as exc:
            logger.error("Submissions fetch failed", cik=cik, error=str(exc))
            return []

        sic_code: Optional[str] = sub.get("sic")
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        results: list[dict] = []
        for i, form in enumerate(forms):
            if form in ("10-K", "10-K405"):
                acc = accessions[i] if i < len(accessions) else None
                doc = primary_docs[i] if i < len(primary_docs) else None
                rd = report_dates[i] if i < len(report_dates) else None
                if acc and doc:
                    yr = int(rd[:4]) if rd else 0
                    results.append({
                        "accession": acc,
                        "primary_doc": doc,
                        "fiscal_year": yr,
                        "sic_code": sic_code or "",
                    })
                if len(results) >= limit:
                    break

        return results

    async def _fetch_filing_text(
        self,
        cik_padded: str,
        accession_num: str,
        doc_name: str,
        limit: int = _TEXT_LIMIT,
    ) -> str:
        """Download filing HTML, strip tags, return up to `limit` chars."""
        cik_bare = cik_padded.lstrip("0") or "0"
        acc_clean = accession_num.replace("-", "")
        url = f"{EDGAR_ARCHIVES}/{cik_bare}/{acc_clean}/{doc_name}"

        await asyncio.sleep(_EDGAR_SLEEP)
        try:
            async with httpx.AsyncClient(
                timeout=60.0, follow_redirects=True
            ) as client:
                resp = await client.get(url, headers=_HEADERS)
                resp.raise_for_status()
                raw = resp.text
        except Exception as exc:
            logger.error("Filing text fetch failed", url=url, error=str(exc))
            return ""

        if "<" in raw[:500]:
            raw = _strip_html(raw)
        return re.sub(r"\s+", " ", raw)[:limit]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Strip HTML tags and decode common entities."""
    text = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>",
        " ", html, flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, rep in [
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
        ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"),
    ]:
        text = text.replace(ent, rep)
    return re.sub(r"\s{3,}", "  ", text).strip()


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def _find_evidence(sentences: list[str], term: str) -> Optional[str]:
    """Return first sentence containing `term` (case-insensitive), truncated."""
    pat = re.escape(term.lower())
    for s in sentences:
        if re.search(pat, s.lower()) and len(s.strip()) > 20:
            return s.strip()[:200]
    return None


def _sic_to_sector(sic_code: Optional[str]) -> str:
    """Map SIC code to SASB sector name."""
    if not sic_code:
        return "default"
    sic4 = str(sic_code)[:4]
    return _SIC_TO_SECTOR.get(sic4, "default")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def esg_score(ticker: str) -> ESGComposite:
    """Score a single ticker on the full ESG composite."""
    engine = ESGCompositeEngine()
    return await engine.score_ticker(ticker)


async def esg_sector(
    tickers: list[str], sector: str
) -> ESGSectorSummary:
    """Score all tickers in a sector and return aggregated summary."""
    engine = ESGCompositeEngine()
    return await engine.score_sector(tickers, sector)


async def esg_screen(
    tickers: list[str],
    min_score: float = 60.0,
    exclude_controversies: bool = True,
) -> list[ESGComposite]:
    """Batch ESG screen — return tickers passing the minimum score threshold."""
    engine = ESGCompositeEngine()
    return await engine.screen_by_esg(
        tickers,
        min_score=min_score,
        exclude_controversies=exclude_controversies,
    )


async def esg_momentum(ticker: str, n_years: int = 3) -> dict:
    """Track ESG score trend across last n_years 10-K filings."""
    engine = ESGCompositeEngine()
    return await engine.get_esg_momentum(ticker, n_years=n_years)
