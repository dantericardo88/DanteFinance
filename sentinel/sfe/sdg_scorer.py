"""sdg_scorer.py — UN Sustainable Development Goal alignment scoring from free data.

Maps company data to SDG 1–17 alignment scores using:
  - SEC EDGAR 10-K filings (Item 1 Business + Item 7 MD&A keyword analysis)
  - SIC code industry-level positive / negative SDG exposure
  - Ticker → CIK resolution via EDGAR company_tickers.json

Proxy signals only — NOT equivalent to GRI/SASB/CDP ratings.
"""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}

# Maximum filing text to download and analyse (100 KB)
_TEXT_LIMIT = 100_000
# Rate-limiting pause between EDGAR requests (SEC fair-use: ≤10 req/s)
_EDGAR_SLEEP = 0.15

# ---------------------------------------------------------------------------
# SDG goal definitions
# ---------------------------------------------------------------------------

SDG_DEFINITIONS: dict[int, dict] = {
    1: {
        "name": "No Poverty",
        "positive_terms": [
            "financial inclusion", "microfinance", "affordable", "low-income",
            "community development", "poverty", "underserved", "unbanked",
            "economic empowerment",
        ],
        "negative_terms": ["poverty wages", "exploitative", "predatory"],
        "sector_positive": ["6020", "6022", "6159"],
        "sector_negative": [],
    },
    2: {
        "name": "Zero Hunger",
        "positive_terms": [
            "food security", "sustainable agriculture", "nutrition", "hunger",
            "crop yield", "food access", "food system", "smallholder",
            "agrobiodiversity",
        ],
        "negative_terms": ["food desert", "food waste", "food insecurity"],
        "sector_positive": ["0100", "0200", "2000", "2010", "5140"],
        "sector_negative": [],
    },
    3: {
        "name": "Good Health & Well-Being",
        "positive_terms": [
            "patient outcomes", "healthcare access", "disease prevention",
            "clinical trial", "drug discovery", "mental health", "vaccine",
            "telehealth", "preventive care", "public health",
        ],
        "negative_terms": ["opioid", "price gouging", "drug abuse", "addiction"],
        "sector_positive": ["2830", "2836", "8000", "8011", "8049", "8051", "8099"],
        "sector_negative": ["2100", "5912"],
    },
    4: {
        "name": "Quality Education",
        "positive_terms": [
            "education", "training", "workforce development", "scholarship",
            "e-learning", "skills", "literacy", "upskilling", "reskilling",
            "STEM", "tuition assistance",
        ],
        "negative_terms": ["predatory lending", "for-profit college", "student debt trap"],
        "sector_positive": ["7372", "7371", "8200", "8211", "8220", "8221", "8222"],
        "sector_negative": [],
    },
    5: {
        "name": "Gender Equality",
        "positive_terms": [
            "gender diversity", "women leadership", "equal pay", "parental leave",
            "inclusion", "diversity", "gender parity", "pay equity",
            "female directors", "women in STEM",
        ],
        "negative_terms": [
            "gender pay gap", "discriminat", "sexual harassment", "gender bias",
        ],
        "sector_positive": [],
        "sector_negative": [],
    },
    6: {
        "name": "Clean Water & Sanitation",
        "positive_terms": [
            "water treatment", "clean water", "sanitation", "water efficiency",
            "wastewater", "desalination", "water recycling", "water reuse",
            "water stewardship",
        ],
        "negative_terms": ["water pollution", "contamination", "discharge violation"],
        "sector_positive": ["4941", "4952", "3589", "3825"],
        "sector_negative": ["2911", "2819", "2860"],
    },
    7: {
        "name": "Affordable Clean Energy",
        "positive_terms": [
            "renewable energy", "solar", "wind", "clean energy", "energy efficiency",
            "electric vehicle", "battery", "net zero", "carbon neutral",
            "energy storage", "hydrogen", "geothermal", "offshore wind",
        ],
        "negative_terms": ["coal", "fossil fuel", "fracking", "tar sands", "flaring"],
        "sector_positive": ["4911", "4931", "3559", "3621", "3699", "3714"],
        "sector_negative": ["1311", "1382", "2911", "1221"],
    },
    8: {
        "name": "Decent Work & Economic Growth",
        "positive_terms": [
            "living wage", "employee benefits", "worker safety", "job creation",
            "labor rights", "fair labor", "occupational health", "employee wellbeing",
            "union", "collective bargaining",
        ],
        "negative_terms": [
            "labor violation", "child labor", "forced labor", "unsafe working",
            "wage theft", "exploitation",
        ],
        "sector_positive": [],
        "sector_negative": ["3131", "2310", "5600"],
    },
    9: {
        "name": "Industry, Innovation & Infrastructure",
        "positive_terms": [
            "research and development", "innovation", "patent", "infrastructure",
            "technology", "automation", "artificial intelligence", "digitalization",
            "broadband", "5G", "smart manufacturing",
        ],
        "negative_terms": [],
        "sector_positive": ["3559", "3674", "7372", "7371", "4813", "4812"],
        "sector_negative": [],
    },
    10: {
        "name": "Reduced Inequalities",
        "positive_terms": [
            "diversity", "inclusion", "equity", "pay equity", "community investment",
            "affordable housing", "underrepresented", "minority-owned",
            "socioeconomic mobility",
        ],
        "negative_terms": ["inequality", "discriminat", "pay gap", "wealth gap"],
        "sector_positive": ["6159", "6552"],
        "sector_negative": [],
    },
    11: {
        "name": "Sustainable Cities & Communities",
        "positive_terms": [
            "smart city", "public transit", "affordable housing", "urban planning",
            "sustainable building", "green building", "LEED", "mixed-use",
            "transit-oriented", "urban resilience",
        ],
        "negative_terms": ["urban sprawl", "displacement", "gentrification"],
        "sector_positive": ["1500", "1521", "1531", "4111", "4131", "4141"],
        "sector_negative": [],
    },
    12: {
        "name": "Responsible Consumption & Production",
        "positive_terms": [
            "circular economy", "recycling", "sustainable supply chain", "zero waste",
            "responsible sourcing", "lifecycle", "product stewardship",
            "extended producer responsibility", "take-back",
        ],
        "negative_terms": [
            "single-use plastic", "landfill", "hazardous waste", "overconsumption",
        ],
        "sector_positive": ["4953", "5093"],
        "sector_negative": ["2911", "2621", "5051"],
    },
    13: {
        "name": "Climate Action",
        "positive_terms": [
            "climate", "greenhouse gas", "emissions reduction", "carbon",
            "net zero", "paris agreement", "scope 1", "scope 2", "scope 3",
            "TCFD", "climate risk", "decarbonization", "carbon offset",
        ],
        "negative_terms": [
            "climate denier", "carbon intensive", "high emissions",
            "stranded asset", "greenwashing",
        ],
        "sector_positive": ["4911", "3674", "3559"],
        "sector_negative": ["1311", "2911", "1221"],
    },
    14: {
        "name": "Life Below Water",
        "positive_terms": [
            "ocean", "marine", "sustainable fishing", "plastic reduction",
            "water quality", "coral reef", "ocean conservation", "blue economy",
            "marine protected area",
        ],
        "negative_terms": [
            "ocean pollution", "overfishing", "plastic waste discharge",
            "deep-sea mining", "bycatch",
        ],
        "sector_positive": ["0900"],
        "sector_negative": ["2911", "2819", "5171"],
    },
    15: {
        "name": "Life on Land",
        "positive_terms": [
            "biodiversity", "sustainable forestry", "habitat", "land conservation",
            "reforestation", "ecosystem services", "no net loss", "afforestation",
        ],
        "negative_terms": [
            "deforestation", "land clearing", "habitat destruction",
            "illegal logging", "land degradation",
        ],
        "sector_positive": ["0800", "0811"],
        "sector_negative": ["2411", "0100"],
    },
    16: {
        "name": "Peace, Justice & Strong Institutions",
        "positive_terms": [
            "governance", "anti-corruption", "compliance", "whistleblower",
            "transparency", "rule of law", "ethics", "human rights due diligence",
            "FCPA", "code of conduct",
        ],
        "negative_terms": [
            "corruption", "bribery", "sanction", "fraud", "money laundering",
            "regulatory violation", "enforcement action",
        ],
        "sector_positive": [],
        "sector_negative": [],
    },
    17: {
        "name": "Partnerships for the Goals",
        "positive_terms": [
            "public-private partnership", "global initiative", "UN", "SDG",
            "sustainable development", "multilateral", "international cooperation",
            "blended finance", "impact investment",
        ],
        "negative_terms": [],
        "sector_positive": [],
        "sector_negative": [],
    },
}

# SIC codes with inherent negative SDG impact by goal number
NEGATIVE_IMPACT_SICS: dict[str, list[int]] = {
    "1311": [7, 13, 14],    # Oil & gas exploration
    "2911": [7, 12, 13, 14],  # Petroleum refining
    "1221": [7, 13],         # Bituminous coal
    "7993": [1, 8],          # Casinos / gambling
    "2100": [3, 8],          # Tobacco
    "3760": [16],            # Defence / guided missiles
    "5912": [3],             # Drug stores (price-gouging risk)
    "2819": [6, 14],         # Industrial inorganic chemicals
    "2860": [6],             # Industrial organic chemicals
    "2411": [15],            # Logging
}

# SIC codes with inherent positive SDG impact by goal number
POSITIVE_IMPACT_SICS: dict[str, list[int]] = {
    "4911": [7, 13],   # Electric services
    "7372": [9],       # Prepackaged software
    "2836": [3],       # Biological products / vaccines
    "8011": [3],       # Offices of physicians
    "8200": [4],       # Educational services
    "4941": [6],       # Water supply
    "0800": [15],      # Forestry
    "4953": [12],      # Refuse systems
    "0900": [14],      # Fishing, hunting & trapping
    "3714": [7],       # Motor vehicle parts (EVs)
}

# SDG weighting for overall composite (must sum to 1.0)
_SDG_WEIGHTS: dict[int, float] = {
    1: 0.05, 2: 0.04, 3: 0.06, 4: 0.05, 5: 0.06,
    6: 0.05, 7: 0.08, 8: 0.06, 9: 0.07, 10: 0.05,
    11: 0.05, 12: 0.06, 13: 0.10, 14: 0.04, 15: 0.04,
    16: 0.07, 17: 0.07,
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SDGScore(BaseModel):
    goal_number: int
    goal_name: str
    score: float                # 0–10
    alignment: str              # "strong_positive" | "positive" | "neutral" | "negative" | "strong_negative"
    evidence: list[str]         # key sentences containing matched keywords (max 5)
    keyword_hits: int


class SDGProfile(BaseModel):
    ticker: str
    company_name: str
    cik: str
    sic_code: Optional[str] = None
    filing_date: Optional[date] = None
    overall_sdg_score: float    # weighted average across all 17 SDGs (0–10)
    positive_goals: list[int]   # SDG numbers with score >= 7
    negative_goals: list[int]   # SDG numbers with score <= 3
    sdg_scores: list[SDGScore]
    top_aligned_goals: list[str]  # human-readable names of top 3
    esg_category: str           # "SDG Leader" | "SDG Committed" | "Neutral" | "SDG Laggard" | "SDG Risk"
    analysis_text: str          # 2–3 sentence narrative summary
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = (
        "Proxy signals from public SEC EDGAR filings. "
        "Not equivalent to GRI/SASB/CDP/MSCI ratings."
    )


class SDGScreenResult(BaseModel):
    query: dict
    total_screened: int
    results: list[SDGProfile]
    avg_score: float
    distribution: dict[str, int]  # {"SDG Leader": N, ...}


# ---------------------------------------------------------------------------
# Core scorer class
# ---------------------------------------------------------------------------

class SDGScorer:
    """Score companies against all 17 UN SDGs using free EDGAR data."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._cik_cache: dict[str, str] = {}
        self._ticker_map: dict[str, dict] | None = None  # lazy loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score_ticker(self, ticker: str) -> SDGProfile:
        """Fetch EDGAR 10-K and produce full SDG profile for one ticker."""
        ticker = ticker.upper()
        cik = await self._get_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker '{ticker}'")

        text, sic_code, filing_date, company_name = await self._get_latest_10k_text(cik)

        sdg_scores: list[SDGScore] = []
        for goal_num in range(1, 18):
            score_obj = self._score_text_for_sdg(text, goal_num, sic_code)
            sdg_scores.append(score_obj)

        overall = self._compute_overall_score(sdg_scores)
        positive_goals = [s.goal_number for s in sdg_scores if s.score >= 7.0]
        negative_goals = [s.goal_number for s in sdg_scores if s.score <= 3.0]

        sorted_scores = sorted(sdg_scores, key=lambda s: s.score, reverse=True)
        top_aligned = [s.goal_name for s in sorted_scores[:3]]

        category = self._categorize(overall)
        analysis = self._build_analysis_text(
            ticker, company_name, overall, category,
            positive_goals, negative_goals, sic_code,
        )

        return SDGProfile(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            sic_code=sic_code,
            filing_date=filing_date,
            overall_sdg_score=round(overall, 2),
            positive_goals=positive_goals,
            negative_goals=negative_goals,
            sdg_scores=sdg_scores,
            top_aligned_goals=top_aligned,
            esg_category=category,
            analysis_text=analysis,
        )

    async def score_batch(
        self,
        tickers: list[str],
        max_concurrent: int = 5,
    ) -> list[SDGProfile]:
        """Score multiple tickers with semaphore-controlled concurrency."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _bounded(ticker: str) -> SDGProfile | None:
            async with sem:
                try:
                    return await self.score_ticker(ticker)
                except Exception as exc:
                    logger.warning(
                        "SDG score failed", ticker=ticker, error=str(exc)
                    )
                    return None

        results = await asyncio.gather(*[_bounded(t) for t in tickers])
        return [r for r in results if r is not None]

    async def screen_by_sdg(
        self,
        tickers: list[str],
        target_sdg: int,
        min_score: float = 6.0,
    ) -> SDGScreenResult:
        """Screen tickers for alignment with a specific SDG goal."""
        if target_sdg not in range(1, 18):
            raise ValueError(f"target_sdg must be 1–17, got {target_sdg}")

        profiles = await self.score_batch(tickers)
        qualifying = [
            p for p in profiles
            if any(
                s.goal_number == target_sdg and s.score >= min_score
                for s in p.sdg_scores
            )
        ]
        qualifying.sort(
            key=lambda p: next(
                (s.score for s in p.sdg_scores if s.goal_number == target_sdg), 0
            ),
            reverse=True,
        )

        avg = (
            sum(
                next(
                    (s.score for s in p.sdg_scores if s.goal_number == target_sdg), 0.0
                )
                for p in qualifying
            )
            / len(qualifying)
            if qualifying
            else 0.0
        )

        dist: dict[str, int] = {}
        for p in profiles:
            dist[p.esg_category] = dist.get(p.esg_category, 0) + 1

        return SDGScreenResult(
            query={"target_sdg": target_sdg, "min_score": min_score, "tickers": tickers},
            total_screened=len(profiles),
            results=qualifying,
            avg_score=round(avg, 2),
            distribution=dist,
        )

    async def compare_sdg_profiles(
        self, ticker1: str, ticker2: str
    ) -> dict:
        """Side-by-side SDG comparison between two tickers."""
        p1, p2 = await asyncio.gather(
            self.score_ticker(ticker1),
            self.score_ticker(ticker2),
        )

        differences: list[dict] = []
        score_map1 = {s.goal_number: s for s in p1.sdg_scores}
        score_map2 = {s.goal_number: s for s in p2.sdg_scores}

        for goal_num in range(1, 18):
            s1 = score_map1.get(goal_num)
            s2 = score_map2.get(goal_num)
            if s1 and s2:
                delta = round(s1.score - s2.score, 2)
                if abs(delta) >= 1.0:
                    differences.append(
                        {
                            "goal_number": goal_num,
                            "goal_name": SDG_DEFINITIONS[goal_num]["name"],
                            ticker1: s1.score,
                            ticker2: s2.score,
                            "delta": delta,
                            "advantage": ticker1 if delta > 0 else ticker2,
                        }
                    )

        differences.sort(key=lambda d: abs(d["delta"]), reverse=True)

        return {
            ticker1: p1,
            ticker2: p2,
            "differences": differences,
            "overall_winner": (
                ticker1 if p1.overall_sdg_score >= p2.overall_sdg_score else ticker2
            ),
        }

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_text_for_sdg(
        self,
        text: str,
        sdg_num: int,
        sic_code: Optional[str] = None,
    ) -> SDGScore:
        """
        Analyse filing text for SDG alignment signals.

        Formula:
          raw_keyword_score = log1p(positive_hits) * 2.0 - log1p(negative_hits) * 2.5
          sic_adjustment    = +2 if sic strongly positive, -2 if strongly negative
          base_score        = 5.0  (neutral mid-point)
          score             = clamp(base_score + raw_keyword_score + sic_adjustment, 0, 10)
        """
        defn = SDG_DEFINITIONS[sdg_num]
        text_lower = text.lower()

        # --- positive keyword hits ---
        pos_hit_count = 0
        neg_hit_count = 0
        evidence_sentences: list[str] = []

        # Split into sentences for evidence extraction (rough sentence boundary)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentence_lower = [s.lower() for s in sentences]

        for term in defn["positive_terms"]:
            matches = re.findall(re.escape(term), text_lower)
            hits = len(matches)
            pos_hit_count += hits
            if hits > 0:
                # Find sentences containing this term (up to 2 per term)
                pat = re.escape(term)
                for i, sl in enumerate(sentence_lower):
                    if re.search(pat, sl) and len(evidence_sentences) < 5:
                        snippet = sentences[i].strip()
                        if len(snippet) > 20 and snippet not in evidence_sentences:
                            evidence_sentences.append(snippet[:200])

        for term in defn["negative_terms"]:
            matches = re.findall(re.escape(term), text_lower)
            neg_hit_count += len(matches)

        # --- SIC code adjustment ---
        sic_adj = 0.0
        if sic_code:
            sic4 = sic_code[:4]
            # Check sector-specific positives/negatives from SDG definition
            if sic4 in defn.get("sector_positive", []):
                sic_adj += 1.5
            if sic4 in defn.get("sector_negative", []):
                sic_adj -= 1.5

            # Check global positive/negative impact SIC tables
            if sic4 in POSITIVE_IMPACT_SICS and sdg_num in POSITIVE_IMPACT_SICS[sic4]:
                sic_adj += 2.0
            if sic4 in NEGATIVE_IMPACT_SICS and sdg_num in NEGATIVE_IMPACT_SICS[sic4]:
                sic_adj -= 2.0

        # --- log-scaled keyword score ---
        keyword_score = (
            math.log1p(pos_hit_count) * 2.0
            - math.log1p(neg_hit_count) * 2.5
        )

        base = 5.0
        raw = base + keyword_score + sic_adj
        score = round(max(0.0, min(10.0, raw)), 2)

        alignment = _alignment_label(score)

        return SDGScore(
            goal_number=sdg_num,
            goal_name=defn["name"],
            score=score,
            alignment=alignment,
            evidence=evidence_sentences[:5],
            keyword_hits=pos_hit_count + neg_hit_count,
        )

    def _compute_overall_score(self, scores: list[SDGScore]) -> float:
        """Weighted average — SDG 13 (climate), 7 (energy), 9 (innovation) weighted higher."""
        total_weight = 0.0
        weighted_sum = 0.0
        for s in scores:
            w = _SDG_WEIGHTS.get(s.goal_number, 1 / 17)
            weighted_sum += s.score * w
            total_weight += w
        if total_weight == 0:
            return 5.0
        return round(weighted_sum / total_weight, 2)

    def _categorize(self, score: float) -> str:
        """Map overall SDG score to category label."""
        if score >= 7.5:
            return "SDG Leader"
        if score >= 6.0:
            return "SDG Committed"
        if score >= 4.5:
            return "Neutral"
        if score >= 3.0:
            return "SDG Laggard"
        return "SDG Risk"

    def _build_analysis_text(
        self,
        ticker: str,
        company_name: str,
        overall: float,
        category: str,
        positive_goals: list[int],
        negative_goals: list[int],
        sic_code: Optional[str],
    ) -> str:
        pos_names = [SDG_DEFINITIONS[g]["name"] for g in positive_goals[:3]]
        neg_names = [SDG_DEFINITIONS[g]["name"] for g in negative_goals[:2]]

        parts = [
            f"{company_name} ({ticker}) scores {overall:.1f}/10 overall SDG alignment, "
            f"classified as '{category}' based on SEC EDGAR 10-K filing analysis."
        ]
        if pos_names:
            parts.append(
                f"Strongest alignment observed in: {', '.join(pos_names)}."
            )
        if neg_names:
            parts.append(
                f"Areas of concern or limited disclosure: {', '.join(neg_names)}."
            )
        elif not neg_names and positive_goals:
            parts.append(
                "No material negative SDG signals identified in the most recent annual filing."
            )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # EDGAR data retrieval
    # ------------------------------------------------------------------

    async def _get_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker symbol to zero-padded 10-digit CIK via EDGAR tickers JSON."""
        ticker_upper = ticker.upper()
        if ticker_upper in self._cik_cache:
            return self._cik_cache[ticker_upper]

        try:
            if self._ticker_map is None:
                await asyncio.sleep(_EDGAR_SLEEP)
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(EDGAR_TICKERS_URL, headers=_HEADERS)
                    resp.raise_for_status()
                    self._ticker_map = resp.json()

            for entry in self._ticker_map.values():
                if str(entry.get("ticker", "")).upper() == ticker_upper:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[ticker_upper] = cik
                    return cik
        except Exception as exc:
            logger.error("CIK lookup failed", ticker=ticker, error=str(exc))

        logger.warning("Ticker not found in EDGAR", ticker=ticker)
        return None

    async def _get_latest_10k_text(
        self, cik: str
    ) -> tuple[str, Optional[str], Optional[date], str]:
        """
        Fetch most recent 10-K filing text (Item 1 + Item 7) plus metadata.

        Returns:
            (text, sic_code, filing_date, company_name)
        """
        cik_padded = cik.zfill(10)

        # 1. Submissions metadata
        sub_url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
        await asyncio.sleep(_EDGAR_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                sub_resp = await client.get(sub_url, headers=_HEADERS)
                sub_resp.raise_for_status()
                sub = sub_resp.json()
        except Exception as exc:
            logger.error("Submissions fetch failed", cik=cik, error=str(exc))
            return "", None, None, cik

        company_name: str = sub.get("name", cik)
        sic_code: Optional[str] = sub.get("sic")

        # 2. Locate most recent 10-K accession
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_num: Optional[str] = None
        primary_doc: Optional[str] = None
        filing_date_str: Optional[str] = None

        for i, form in enumerate(forms):
            if form in ("10-K", "10-K405"):
                accession_list = recent.get("accessionNumber", [])
                doc_list = recent.get("primaryDocument", [])
                date_list = recent.get("filingDate", [])
                if i < len(accession_list):
                    accession_num = accession_list[i]
                if i < len(doc_list):
                    primary_doc = doc_list[i]
                if i < len(date_list):
                    filing_date_str = date_list[i]
                break

        if not accession_num:
            logger.warning("No 10-K found in submissions", cik=cik)
            return "", sic_code, None, company_name

        filing_date: Optional[date] = None
        if filing_date_str:
            try:
                filing_date = date.fromisoformat(filing_date_str)
            except ValueError:
                pass

        # 3. Fetch filing index to find the main document if primary_doc is missing
        if not primary_doc:
            primary_doc = await self._resolve_primary_doc(
                cik_padded, accession_num
            )

        if not primary_doc:
            return "", sic_code, filing_date, company_name

        # 4. Download and clean filing text
        text = await self._fetch_filing_text(cik_padded, accession_num, primary_doc)
        return text, sic_code, filing_date, company_name

    async def _resolve_primary_doc(
        self, cik_padded: str, accession_num: str
    ) -> Optional[str]:
        """Fetch filing index JSON and return the primary .htm document name."""
        acc_clean = accession_num.replace("-", "")
        index_url = (
            f"{EDGAR_BASE}/submissions/CIK{cik_padded}/filing-index/{acc_clean}.json"
        )
        # Fallback: EDGAR index endpoint
        alt_index_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik_padded}&type=10-K&dateb=&owner=include&count=1&search_text="
        )
        await asyncio.sleep(_EDGAR_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(index_url, headers=_HEADERS)
                if resp.status_code == 200:
                    data = resp.json()
                    docs = data.get("documents", [])
                    for doc in docs:
                        if doc.get("type") in ("10-K", "10-K405") and doc.get("document", "").endswith(".htm"):
                            return doc["document"]
                    # Return first htm if specific type not found
                    for doc in docs:
                        if doc.get("document", "").endswith(".htm"):
                            return doc["document"]
        except Exception as exc:
            logger.debug("Filing index fetch failed", error=str(exc))

        return None

    async def _fetch_filing_text(
        self,
        cik_padded: str,
        accession_num: str,
        doc_name: str,
    ) -> str:
        """Download 10-K HTML, strip tags, and return first _TEXT_LIMIT chars."""
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
            raw = self._strip_html(raw)

        # Collapse whitespace and truncate
        cleaned = re.sub(r"\s+", " ", raw)
        return cleaned[:_TEXT_LIMIT]

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and decode common entities; collapse whitespace."""
        # Remove script and style blocks entirely
        text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        for entity, replacement in [
            ("&nbsp;", " "),
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&quot;", '"'),
            ("&#39;", "'"),
            ("&apos;", "'"),
        ]:
            text = text.replace(entity, replacement)
        # Collapse runs of whitespace
        return re.sub(r"\s{3,}", "  ", text).strip()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _alignment_label(score: float) -> str:
    if score >= 7.5:
        return "strong_positive"
    if score >= 6.0:
        return "positive"
    if score >= 4.0:
        return "neutral"
    if score >= 2.5:
        return "negative"
    return "strong_negative"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def score_ticker(ticker: str) -> SDGProfile:
    """Score a single ticker against all 17 SDGs."""
    scorer = SDGScorer()
    return await scorer.score_ticker(ticker)


async def score_batch(
    tickers: list[str], max_concurrent: int = 5
) -> list[SDGProfile]:
    """Score multiple tickers with controlled concurrency."""
    scorer = SDGScorer()
    return await scorer.score_batch(tickers, max_concurrent=max_concurrent)


async def screen_sdg(
    tickers: list[str], sdg: int, min_score: float = 6.0
) -> SDGScreenResult:
    """Screen a universe of tickers for alignment with a specific SDG."""
    scorer = SDGScorer()
    return await scorer.screen_by_sdg(tickers, sdg, min_score=min_score)


async def compare_tickers(ticker1: str, ticker2: str) -> dict:
    """Side-by-side SDG comparison of two companies."""
    scorer = SDGScorer()
    return await scorer.compare_sdg_profiles(ticker1, ticker2)
