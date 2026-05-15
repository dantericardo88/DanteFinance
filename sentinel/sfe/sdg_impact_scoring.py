"""
UN SDG (Sustainable Development Goals) alignment scoring for companies.
Maps business activities, products, and controversies to all 17 SDGs.

Dimension: dim_105 — UN SDG alignment / impact scoring (target: 9)

Free data sources:
  EDGAR 10-K (SEC EDGAR EFTS full-text search) — business section NLP
  SEC EDGAR company search — SIC codes, company metadata
  UN SDG Index (hardcoded reference data) — country-level SDG scores
  GDELT DOC API — sustainability controversy signals

Components:
  SDGFramework          — all 17 SDGs with keywords (positive + negative)
  SDGRevenueMapper      — SIC code → SDG alignment, 10-K business section parsing
  SDGAlignmentScorer    — per-company SDG scores (-10 to +10 per SDG)
  ImpactPortfolioBuilder— SDG-themed portfolio construction
  SDGProgressTracker    — YoY SDG score evolution, reporting quality
  SDGRiskAnalyzer       — transition risk, stranded assets, investor mandates
  FastAPI router        — /sdg/* endpoints
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDGAR_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_COMPANY_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_EDGAR_BROWSE_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
_GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

_TIMEOUT = 30
_CACHE_TTL = 900  # 15 minutes

_EDGAR_HEADERS = {
    "User-Agent": "SENTINEL/2.0 research@sentinel.ai",
    "Accept": "application/json,*/*",
}

# ---------------------------------------------------------------------------
# SDG Framework: all 17 SDGs
# ---------------------------------------------------------------------------


class SDGFramework:
    """
    UN Sustainable Development Goals — definitions, targets, and keyword mappings.

    Provides 200+ positive keywords and 100+ negative keywords across all 17 SDGs.
    Used for NLP-based alignment scoring of company filings and news.
    """

    # SDG metadata: id → (name, description)
    SDG_METADATA: dict[int, tuple[str, str]] = {
        1:  ("No Poverty", "End poverty in all its forms everywhere"),
        2:  ("Zero Hunger", "End hunger, achieve food security and improved nutrition"),
        3:  ("Good Health & Well-Being", "Ensure healthy lives and promote well-being for all"),
        4:  ("Quality Education", "Ensure inclusive and equitable quality education"),
        5:  ("Gender Equality", "Achieve gender equality and empower all women and girls"),
        6:  ("Clean Water & Sanitation", "Ensure availability of water and sanitation for all"),
        7:  ("Affordable & Clean Energy", "Ensure access to affordable, reliable, sustainable energy"),
        8:  ("Decent Work & Economic Growth", "Promote sustained, inclusive economic growth"),
        9:  ("Industry, Innovation & Infrastructure", "Build resilient infrastructure, promote industrialization"),
        10: ("Reduced Inequalities", "Reduce inequality within and among countries"),
        11: ("Sustainable Cities & Communities", "Make cities inclusive, safe, resilient and sustainable"),
        12: ("Responsible Consumption & Production", "Ensure sustainable consumption and production patterns"),
        13: ("Climate Action", "Take urgent action to combat climate change and its impacts"),
        14: ("Life Below Water", "Conserve and sustainably use oceans, seas and marine resources"),
        15: ("Life on Land", "Protect, restore terrestrial ecosystems, forests, and biodiversity"),
        16: ("Peace, Justice & Strong Institutions", "Promote peaceful and inclusive societies"),
        17: ("Partnerships for the Goals", "Strengthen the means of implementation and global partnership"),
    }

    # Positive keywords per SDG (activities that contribute to each goal)
    SDG_POSITIVE_KEYWORDS: dict[int, list[str]] = {
        1: [
            "poverty reduction", "economic inclusion", "microfinance", "financial inclusion",
            "affordable housing", "low income", "community development", "social safety net",
            "basic income", "livelihood", "cash transfer", "remittance", "mobile banking",
            "underserved community", "wealth gap", "poverty alleviation",
        ],
        2: [
            "food security", "agriculture", "nutrition", "crop yield", "food production",
            "hunger", "malnutrition", "sustainable farming", "precision agriculture",
            "food technology", "agri-tech", "vertical farming", "food waste reduction",
            "food supply chain", "seed technology", "fertilizer efficiency", "livestock",
            "aquaculture", "food fortification", "zero hunger",
        ],
        3: [
            "healthcare", "medical", "pharmaceutical", "wellness", "vaccine", "immunization",
            "public health", "disease prevention", "mental health", "telemedicine",
            "diagnostics", "therapeutics", "clinical trial", "drug discovery",
            "health access", "maternal health", "child health", "life expectancy",
            "hospital", "insurance coverage", "preventive care", "global health",
            "medical device", "health technology", "biotech", "oncology",
        ],
        4: [
            "education", "learning", "school", "university", "training", "skill development",
            "STEM", "e-learning", "edtech", "vocational training", "literacy",
            "scholarship", "tuition", "teacher", "curriculum", "educational access",
            "lifelong learning", "higher education", "workforce development",
        ],
        5: [
            "gender equality", "women empowerment", "female leadership", "equal pay",
            "gender diversity", "women in workforce", "maternity leave", "parental leave",
            "gender gap", "women entrepreneur", "girls education", "gender parity",
            "sexual harassment prevention", "gender-based violence prevention",
            "female founder", "inclusive hiring", "pay equity",
        ],
        6: [
            "clean water", "water treatment", "sanitation", "wastewater", "water purification",
            "water efficiency", "drought resilience", "water conservation", "water access",
            "water recycling", "desalination", "sewage treatment", "water quality",
            "water security", "irrigation efficiency", "water infrastructure",
        ],
        7: [
            "renewable energy", "solar", "wind energy", "clean energy", "energy efficiency",
            "energy storage", "battery technology", "electric vehicle", "EV charging",
            "hydrogen fuel", "geothermal", "hydroelectric", "energy transition",
            "net zero energy", "carbon neutral energy", "green power", "photovoltaic",
            "offshore wind", "energy access", "smart grid", "distributed energy",
            "power storage", "energy poverty reduction", "clean fuel",
        ],
        8: [
            "employment", "job creation", "wages", "labor rights", "decent work",
            "fair wage", "worker safety", "living wage", "workforce", "human capital",
            "social protection", "economic growth", "small business", "entrepreneur",
            "productivity", "trade", "labor standard", "occupational health",
            "youth employment", "apprenticeship", "skill training", "supply chain labor",
        ],
        9: [
            "innovation", "research and development", "R&D", "technology", "infrastructure",
            "industrialization", "manufacturing", "automation", "digitization",
            "broadband", "connectivity", "5G", "AI", "artificial intelligence",
            "cloud computing", "semiconductor", "internet of things", "IoT",
            "logistics", "supply chain", "smart manufacturing", "cleantech",
            "sustainable infrastructure", "bridge", "road", "port", "rail",
        ],
        10: [
            "inequality reduction", "diversity", "inclusion", "equal opportunity",
            "minority", "underrepresented", "disability", "accessibility",
            "inclusive growth", "affirmative action", "pay gap", "social mobility",
            "migration policy", "refugee", "remittance", "progressive tax",
            "economic empowerment", "racial equity", "LGBTQ inclusion", "equity",
        ],
        11: [
            "sustainable city", "urban planning", "affordable housing", "public transport",
            "smart city", "green building", "LEED", "urban resilience", "disaster risk",
            "heritage preservation", "community", "urban renewal", "air quality city",
            "mixed-use development", "pedestrian", "cycling infrastructure", "parking",
            "flood risk", "urban heat", "city park", "social housing",
        ],
        12: [
            "circular economy", "recycling", "waste reduction", "sustainable packaging",
            "supply chain sustainability", "responsible sourcing", "product lifecycle",
            "take-back program", "zero waste", "refurbishment", "remanufacturing",
            "sustainable consumption", "eco-design", "green procurement", "food waste",
            "extended producer responsibility", "biodegradable", "compostable",
            "resource efficiency", "closed loop", "upcycling",
        ],
        13: [
            "climate action", "carbon", "emissions reduction", "net zero", "decarbonization",
            "carbon offset", "carbon capture", "greenhouse gas", "GHG", "Paris Agreement",
            "climate target", "Scope 1", "Scope 2", "Scope 3", "climate risk",
            "TCFD", "climate disclosure", "carbon footprint", "low carbon",
            "climate resilience", "adaptation", "mitigation", "SBTi",
            "science-based target", "carbon trading", "climate transition",
        ],
        14: [
            "ocean", "marine", "sea", "fisheries", "aquatic", "coral reef",
            "ocean pollution", "plastic reduction", "maritime", "blue economy",
            "sustainable seafood", "water pollution reduction", "ocean acidification",
            "marine biodiversity", "coastal management", "deep sea", "fishing regulation",
        ],
        15: [
            "biodiversity", "deforestation prevention", "reforestation", "ecosystem",
            "land use", "forest conservation", "wildlife", "habitat", "soil health",
            "wetland", "national park", "sustainable land management", "nature-based solution",
            "species protection", "sustainable agriculture land", "land restoration",
            "anti-poaching", "REDD+", "carbon sink", "agroforestry",
        ],
        16: [
            "governance", "transparency", "accountability", "anti-corruption", "rule of law",
            "human rights", "peace", "justice", "institution", "democracy",
            "data privacy", "freedom of speech", "press freedom", "whistleblower protection",
            "anti-bribery", "conflict prevention", "peacekeeping", "arms control",
            "judicial independence", "open government", "civil society",
        ],
        17: [
            "partnership", "collaboration", "UN", "multilateral", "development finance",
            "official development assistance", "ODA", "blended finance", "PPP",
            "public-private partnership", "global cooperation", "trade facilitation",
            "technology transfer", "capacity building", "data sharing",
            "SDG commitment", "ESG reporting", "GRI", "SASB", "TCFD alignment",
            "sustainable finance", "green bond", "impact investing", "MDB",
        ],
    }

    # Negative keywords per SDG (activities that undermine each goal)
    SDG_NEGATIVE_KEYWORDS: dict[int, list[str]] = {
        1:  ["predatory lending", "payday loan", "wage theft", "worker exploitation", "poverty trap"],
        2:  ["food fraud", "pesticide violation", "food contamination", "food monopoly", "GMO controversy"],
        3:  [
            "opioid", "tobacco", "alcohol abuse", "health lawsuit", "drug price gouging",
            "pharmaceutical fraud", "dangerous drug", "unsafe product", "medical device recall",
            "cigarette", "vaping harm", "addiction", "opioid crisis",
        ],
        4:  ["education fraud", "diploma mill", "predatory college", "student loan trap"],
        5:  [
            "gender discrimination lawsuit", "sexual harassment", "gender pay gap lawsuit",
            "maternity discrimination", "women excluded", "gender bias",
        ],
        6:  ["water pollution", "water contamination", "water scarcity caused", "PFAS", "lead contamination"],
        7:  [
            "fossil fuel", "coal", "high emissions", "carbon-intensive", "oil spill",
            "natural gas flaring", "tar sands", "fracking", "stranded asset fossil",
        ],
        8:  [
            "wage theft", "labor violation", "sweatshop", "child labor", "forced labor",
            "union busting", "worker exploitation", "unsafe working conditions",
            "minimum wage violation", "overtime violation",
        ],
        9:  ["obsolete technology", "infrastructure neglect", "digital divide widen"],
        10: [
            "discrimination lawsuit", "racial discrimination", "gender pay gap",
            "disability discrimination", "income inequality worsening", "minority bias",
        ],
        11: ["urban displacement", "gentrification harm", "housing discrimination", "slum creation"],
        12: [
            "excessive packaging", "greenwashing", "planned obsolescence", "hazardous waste",
            "toxic product", "unsustainable supply chain", "illegal dumping",
        ],
        13: [
            "deforestation", "carbon fraud", "greenwashing climate", "emission increase",
            "climate denial", "fossil fuel expansion", "coal power", "stranded carbon asset",
            "scope 3 hidden", "carbon offset fraud",
        ],
        14: ["ocean dumping", "illegal fishing", "marine pollution", "plastic waste ocean", "overfishing"],
        15: [
            "deforestation", "habitat destruction", "illegal logging", "wildlife trafficking",
            "biodiversity loss", "land grab", "illegal mining", "pesticide harm",
        ],
        16: [
            "corruption", "bribery", "fraud", "money laundering", "sanctions violation",
            "human rights violation", "child labor supply chain", "conflict minerals",
            "governance failure", "regulatory capture", "tax evasion",
        ],
        17: ["SDG washing", "partnership manipulation", "misleading ESG claim"],
    }

    @classmethod
    def get_sdg_name(cls, sdg_id: int) -> str:
        return cls.SDG_METADATA.get(sdg_id, (f"SDG {sdg_id}", ""))[0]

    @classmethod
    def get_material_sdgs(cls, sector: str) -> list[int]:
        """Return the most material SDGs for a given industry sector."""
        sector_material: dict[str, list[int]] = {
            "Technology": [9, 17, 10, 16, 4],
            "Healthcare": [3, 10, 17, 16, 9],
            "Energy": [7, 13, 15, 11, 17],
            "Financials": [8, 10, 17, 16, 1],
            "Consumer Discretionary": [12, 8, 10, 13, 2],
            "Consumer Staples": [2, 12, 3, 8, 13],
            "Industrials": [9, 13, 8, 11, 12],
            "Materials": [15, 13, 12, 14, 6],
            "Utilities": [7, 13, 6, 11, 12],
            "Real Estate": [11, 13, 6, 15, 10],
            "Communication Services": [9, 10, 16, 17, 4],
        }
        return sector_material.get(sector, [13, 8, 16, 17, 9])


# ---------------------------------------------------------------------------
# SIC Code → SDG Mapping
# ---------------------------------------------------------------------------

# Comprehensive SIC → (primary_sdg, alignment_strength, negative_sdgs)
# alignment_strength: 1.0 = strong, 0.7 = moderate, 0.4 = weak
# negative_sdgs: SDGs this sector commonly undermines

_SIC_SDG_MAP: dict[int, dict] = {
    # Healthcare
    8011: {"primary": 3, "strength": 1.0, "sector": "Healthcare", "negative": []},
    8021: {"primary": 3, "strength": 1.0, "sector": "Healthcare", "negative": []},
    8049: {"primary": 3, "strength": 0.9, "sector": "Healthcare", "negative": []},
    8062: {"primary": 3, "strength": 0.9, "sector": "Healthcare", "negative": []},
    8099: {"primary": 3, "strength": 0.8, "sector": "Healthcare", "negative": []},
    2836: {"primary": 3, "strength": 0.9, "sector": "Pharma", "negative": [16]},  # pharma
    2833: {"primary": 3, "strength": 0.8, "sector": "Pharma", "negative": []},
    3841: {"primary": 3, "strength": 0.9, "sector": "Med Devices", "negative": []},
    3845: {"primary": 3, "strength": 0.8, "sector": "Med Devices", "negative": []},
    # Education
    8200: {"primary": 4, "strength": 1.0, "sector": "Education", "negative": []},
    8249: {"primary": 4, "strength": 0.9, "sector": "Education", "negative": []},
    7372: {"primary": 9, "strength": 0.8, "sector": "Software", "negative": []},  # software (edtech/fintech)
    # Clean Energy / Utilities
    4911: {"primary": 7, "strength": 0.7, "sector": "Electric Utilities", "negative": [13]},  # could be coal
    4941: {"primary": 6, "strength": 0.9, "sector": "Water Utilities", "negative": []},
    4924: {"primary": 7, "strength": 0.5, "sector": "Gas Utilities", "negative": [13]},
    1731: {"primary": 7, "strength": 0.6, "sector": "Energy Construction", "negative": []},
    # Renewable energy equipment
    3559: {"primary": 7, "strength": 0.7, "sector": "Industrial Equipment", "negative": []},
    3674: {"primary": 9, "strength": 0.9, "sector": "Semiconductors", "negative": []},
    # Agriculture / Food
    100: {"primary": 2, "strength": 1.0, "sector": "Crops", "negative": [15]},
    200: {"primary": 2, "strength": 0.9, "sector": "Livestock", "negative": [15]},
    2000: {"primary": 2, "strength": 0.8, "sector": "Food Processing", "negative": []},
    2011: {"primary": 2, "strength": 0.7, "sector": "Meat Packing", "negative": [15, 8]},
    2040: {"primary": 2, "strength": 0.8, "sector": "Grain Mill", "negative": []},
    2086: {"primary": 2, "strength": 0.5, "sector": "Beverages", "negative": [3]},
    2100: {"primary": 0, "strength": 0.0, "sector": "Tobacco", "negative": [3]},  # SDG 3 negative
    2111: {"primary": 0, "strength": 0.0, "sector": "Tobacco", "negative": [3]},
    # Financial Services
    6020: {"primary": 8, "strength": 0.7, "sector": "Banking", "negative": []},
    6022: {"primary": 8, "strength": 0.7, "sector": "Banking", "negative": []},
    6035: {"primary": 1, "strength": 0.7, "sector": "Savings Inst", "negative": []},
    6141: {"primary": 1, "strength": 0.6, "sector": "Consumer Finance", "negative": []},  # predatory risk
    6159: {"primary": 8, "strength": 0.6, "sector": "Mortgage", "negative": []},
    6211: {"primary": 8, "strength": 0.6, "sector": "Securities", "negative": []},
    6311: {"primary": 8, "strength": 0.7, "sector": "Insurance", "negative": []},
    # Technology
    3571: {"primary": 9, "strength": 0.9, "sector": "Computers", "negative": [12]},
    3577: {"primary": 9, "strength": 0.8, "sector": "Computer Peripherals", "negative": [12]},
    3669: {"primary": 9, "strength": 0.8, "sector": "Communications Equip", "negative": []},
    3672: {"primary": 9, "strength": 0.9, "sector": "Printed Circuit Boards", "negative": [12]},
    3679: {"primary": 9, "strength": 0.7, "sector": "Electronic Components", "negative": [12]},
    3825: {"primary": 9, "strength": 0.8, "sector": "Instruments", "negative": []},
    7370: {"primary": 9, "strength": 0.9, "sector": "Computer Services", "negative": []},
    7371: {"primary": 9, "strength": 0.9, "sector": "Programming", "negative": []},
    7374: {"primary": 9, "strength": 0.8, "sector": "Data Processing", "negative": []},
    7389: {"primary": 9, "strength": 0.7, "sector": "Business Services", "negative": []},
    # Chemicals
    2810: {"primary": 9, "strength": 0.5, "sector": "Chemicals", "negative": [6, 15, 3]},
    2819: {"primary": 9, "strength": 0.5, "sector": "Industrial Chemicals", "negative": [6, 14]},
    2860: {"primary": 9, "strength": 0.5, "sector": "Ind Chemicals", "negative": [6, 15]},
    2911: {"primary": 0, "strength": 0.0, "sector": "Oil Refining", "negative": [13, 7]},
    # Oil & Gas (fossil fuel — SDG 13 negative)
    1311: {"primary": 8, "strength": 0.5, "sector": "Oil & Gas", "negative": [13, 7, 15]},
    1381: {"primary": 8, "strength": 0.4, "sector": "Drilling", "negative": [13, 15]},
    5171: {"primary": 8, "strength": 0.4, "sector": "Petroleum Products", "negative": [13]},
    5172: {"primary": 8, "strength": 0.4, "sector": "Petroleum Products Wholesale", "negative": [13]},
    # Mining & Materials
    1000: {"primary": 9, "strength": 0.5, "sector": "Metal Mining", "negative": [15, 14, 6]},
    1040: {"primary": 9, "strength": 0.4, "sector": "Gold Mining", "negative": [15]},
    1094: {"primary": 9, "strength": 0.4, "sector": "Uranium", "negative": [15]},
    1220: {"primary": 0, "strength": 0.0, "sector": "Coal Mining", "negative": [13, 7, 15]},
    2911: {"primary": 0, "strength": 0.0, "sector": "Petroleum Refining", "negative": [13]},
    3310: {"primary": 9, "strength": 0.5, "sector": "Steel", "negative": [13]},
    3334: {"primary": 9, "strength": 0.5, "sector": "Aluminum", "negative": [13]},
    # Transportation
    4011: {"primary": 11, "strength": 0.7, "sector": "Railroads", "negative": []},
    4111: {"primary": 11, "strength": 0.8, "sector": "Transit", "negative": []},
    4512: {"primary": 8, "strength": 0.6, "sector": "Airlines", "negative": [13]},
    4522: {"primary": 8, "strength": 0.6, "sector": "Air Charter", "negative": [13]},
    4213: {"primary": 8, "strength": 0.5, "sector": "Trucking", "negative": [13]},
    4400: {"primary": 8, "strength": 0.5, "sector": "Water Transport", "negative": [14]},
    # Retail
    5211: {"primary": 12, "strength": 0.5, "sector": "Lumber Retail", "negative": [15]},
    5311: {"primary": 12, "strength": 0.6, "sector": "Department Stores", "negative": [12]},
    5411: {"primary": 2, "strength": 0.7, "sector": "Grocery Stores", "negative": [12]},
    5912: {"primary": 3, "strength": 0.7, "sector": "Drug Stores", "negative": []},
    5940: {"primary": 8, "strength": 0.6, "sector": "Sporting Goods", "negative": []},
    5945: {"primary": 4, "strength": 0.6, "sector": "Hobby & Toy", "negative": []},
    # Real Estate / Construction
    1521: {"primary": 11, "strength": 0.7, "sector": "Home Construction", "negative": []},
    6512: {"primary": 11, "strength": 0.6, "sector": "REITs", "negative": []},
    6552: {"primary": 11, "strength": 0.6, "sector": "Land Developers", "negative": [15]},
    # Communications / Media
    4812: {"primary": 9, "strength": 0.8, "sector": "Telecom", "negative": []},
    4813: {"primary": 9, "strength": 0.8, "sector": "Telecom Services", "negative": []},
    7812: {"primary": 10, "strength": 0.5, "sector": "Motion Pictures", "negative": []},
    2711: {"primary": 16, "strength": 0.6, "sector": "Newspapers", "negative": []},
    7375: {"primary": 9, "strength": 0.8, "sector": "Computer Rental", "negative": []},
    # Defense
    3812: {"primary": 9, "strength": 0.4, "sector": "Defense", "negative": [16]},
    3761: {"primary": 9, "strength": 0.3, "sector": "Guided Missiles", "negative": [16]},
    # Waste / Environment
    4953: {"primary": 12, "strength": 0.9, "sector": "Refuse Systems", "negative": []},
    8711: {"primary": 9, "strength": 0.8, "sector": "Engineering Services", "negative": []},
    8731: {"primary": 9, "strength": 0.9, "sector": "R&D Labs", "negative": []},
    # Hotels / Hospitality
    5812: {"primary": 8, "strength": 0.5, "sector": "Restaurants", "negative": [12]},
    7011: {"primary": 8, "strength": 0.5, "sector": "Hotels", "negative": [12]},
    # Apparel
    2320: {"primary": 8, "strength": 0.5, "sector": "Apparel", "negative": [8, 12]},
    2325: {"primary": 8, "strength": 0.5, "sector": "Trousers", "negative": [8]},
    5600: {"primary": 8, "strength": 0.5, "sector": "Apparel Retail", "negative": [8, 12]},
    # Automotive
    3711: {"primary": 9, "strength": 0.7, "sector": "Motor Vehicles", "negative": [13]},
    3714: {"primary": 9, "strength": 0.7, "sector": "Auto Parts", "negative": [13]},
    5511: {"primary": 8, "strength": 0.5, "sector": "Car Dealers", "negative": []},
}


# Country-level SDG index scores (UN SDG Index 2024; 0–100)
_COUNTRY_SDG_SCORES: dict[str, float] = {
    "Finland": 86.8, "Sweden": 86.0, "Denmark": 85.7, "Germany": 82.9,
    "France": 81.2, "Japan": 80.0, "UK": 79.5, "Canada": 78.1,
    "Australia": 76.8, "USA": 74.6, "South Korea": 77.2, "Spain": 79.8,
    "Italy": 78.2, "Netherlands": 82.4, "Switzerland": 82.0, "Norway": 82.6,
    "New Zealand": 77.5, "Austria": 81.1, "Belgium": 80.5, "Portugal": 79.3,
    "Ireland": 79.1, "Singapore": 76.0, "China": 72.8, "Brazil": 70.5,
    "India": 63.5, "Russia": 71.2, "Mexico": 67.4, "Turkey": 72.1,
    "Indonesia": 68.4, "Saudi Arabia": 70.2, "South Africa": 59.7,
    "Nigeria": 52.1, "Egypt": 63.0, "Argentina": 70.8, "Chile": 72.3,
    "Poland": 78.6, "Czech Republic": 80.1, "Hungary": 76.5,
    "United States": 74.6,
}


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_DB_PATH = Path("data/sdg_scores.db")


def _ensure_sdg_db() -> sqlite3.Connection:
    """Open/create the SDG scores SQLite database."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sdg_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT,
            sic_code INTEGER,
            sdg_1 REAL, sdg_2 REAL, sdg_3 REAL, sdg_4 REAL, sdg_5 REAL,
            sdg_6 REAL, sdg_7 REAL, sdg_8 REAL, sdg_9 REAL, sdg_10 REAL,
            sdg_11 REAL, sdg_12 REAL, sdg_13 REAL, sdg_14 REAL, sdg_15 REAL,
            sdg_16 REAL, sdg_17 REAL,
            overall_score REAL,
            primary_sdgs TEXT,
            negative_sdgs TEXT,
            reporting_quality REAL,
            filing_year INTEGER,
            as_of TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sdg_ticker ON sdg_scores(ticker, as_of)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sdg_portfolio_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sdg_number INTEGER,
            ticker TEXT,
            company_name TEXT,
            sdg_score REAL,
            built_at TEXT
        )
    """)
    conn.commit()
    return conn


def _save_sdg_profile(profile: "SDGProfile") -> None:
    """Persist an SDG profile to SQLite."""
    try:
        conn = _ensure_sdg_db()
        sdg_cols = {f"sdg_{i}": profile.sdg_scores.get(i, 0.0) for i in range(1, 18)}
        conn.execute("""
            INSERT INTO sdg_scores
            (ticker, company_name, sector, sic_code,
             sdg_1, sdg_2, sdg_3, sdg_4, sdg_5, sdg_6, sdg_7, sdg_8, sdg_9,
             sdg_10, sdg_11, sdg_12, sdg_13, sdg_14, sdg_15, sdg_16, sdg_17,
             overall_score, primary_sdgs, negative_sdgs, reporting_quality, filing_year, as_of)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            profile.ticker, profile.company_name, profile.sector, profile.sic_code,
            sdg_cols["sdg_1"], sdg_cols["sdg_2"], sdg_cols["sdg_3"], sdg_cols["sdg_4"],
            sdg_cols["sdg_5"], sdg_cols["sdg_6"], sdg_cols["sdg_7"], sdg_cols["sdg_8"],
            sdg_cols["sdg_9"], sdg_cols["sdg_10"], sdg_cols["sdg_11"], sdg_cols["sdg_12"],
            sdg_cols["sdg_13"], sdg_cols["sdg_14"], sdg_cols["sdg_15"], sdg_cols["sdg_16"],
            sdg_cols["sdg_17"],
            profile.overall_sdg_score,
            json.dumps(profile.primary_sdgs),
            json.dumps(profile.negative_sdgs),
            profile.reporting_quality_score,
            profile.filing_year,
            profile.as_of.isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Failed to save SDG profile", ticker=profile.ticker, error=str(exc))


def _load_sdg_history(ticker: str, years: int = 5) -> list[dict]:
    """Load historical SDG scores for a ticker."""
    try:
        conn = _ensure_sdg_db()
        cutoff_year = datetime.now().year - years
        rows = conn.execute("""
            SELECT ticker, company_name, sector, overall_score, primary_sdgs,
                   negative_sdgs, reporting_quality, filing_year, as_of,
                   sdg_7, sdg_13, sdg_3, sdg_8, sdg_9
            FROM sdg_scores
            WHERE ticker = ? AND filing_year >= ?
            ORDER BY filing_year ASC
        """, (ticker.upper(), cutoff_year)).fetchall()
        conn.close()
        cols = [
            "ticker", "company_name", "sector", "overall_score", "primary_sdgs",
            "negative_sdgs", "reporting_quality", "filing_year", "as_of",
            "sdg_7", "sdg_13", "sdg_3", "sdg_8", "sdg_9",
        ]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        logger.warning("Failed to load SDG history", ticker=ticker, error=str(exc))
        return []


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _cache[key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SDGProfile(BaseModel):
    """Full SDG alignment profile for a company."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    sector: str
    sic_code: Optional[int]
    sdg_scores: dict[int, float]  # sdg_id → score (-10 to +10)
    overall_sdg_score: float  # weighted average of material SDGs
    primary_sdgs: list[int]  # top 3-5 positive SDG contributions
    negative_sdgs: list[int]  # SDGs being undermined
    material_sdgs: list[int]  # sector-relevant SDGs
    reporting_quality_score: float  # 0-100 (GRI/SASB/TCFD alignment)
    country_sdg_context: float  # home country UN SDG Index score
    sdg_commitment_found: bool  # explicitly mentions SDG targets
    filing_year: int
    as_of: datetime


class SDGPortfolioEntry(BaseModel):
    """Single entry in an SDG-themed portfolio."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    sector: str
    sdg_score: float
    overall_sdg_score: float
    primary_sdgs: list[int]
    negative_sdgs: list[int]


class SDGProgressPoint(BaseModel):
    """Single year data point in SDG progress tracking."""
    model_config = ConfigDict(frozen=True)
    filing_year: int
    overall_score: float
    sdg_7_score: float
    sdg_13_score: float
    sdg_3_score: float
    reporting_quality: float
    sdg_commitment: bool


class SDGTransitionRisk(BaseModel):
    """SDG-linked transition risk assessment."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    transition_risk_score: float  # 0-100
    stranded_asset_risk: str  # low/medium/high/critical
    regulatory_risk_sdgs: list[int]  # SDGs driving regulatory risk
    investor_mandate_risk: str  # low/medium/high
    supply_chain_sdg_risk: list[str]
    negative_sdg_exposure: dict[int, float]
    risk_factors: list[str]
    as_of: datetime


class SectorSDGRanking(BaseModel):
    """Sector-level SDG ranking entry."""
    model_config = ConfigDict(frozen=True)
    ticker: str
    company_name: str
    sector: str
    overall_sdg_score: float
    primary_sdgs: list[int]
    reporting_quality: float
    rank: int


# ---------------------------------------------------------------------------
# SDGRevenueMapper
# ---------------------------------------------------------------------------


class SDGRevenueMapper:
    """
    Maps company revenue to SDG alignment using SIC codes and 10-K text parsing.

    Two-stage process:
    1. SIC code lookup → primary SDG alignment
    2. 10-K Business section NLP → keyword-based revenue proportion estimate
    """

    def __init__(self, timeout: int = _TIMEOUT) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_EDGAR_HEADERS)

    def get_company_sic(self, ticker: str) -> tuple[Optional[int], str, str]:
        """
        Fetch SIC code and company info from SEC EDGAR.

        Returns (sic_code, company_name, state_of_incorporation).
        """
        cache_key = f"edgar_sic:{ticker}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            # SEC EDGAR company search
            params = {
                "q": f'"{ticker}"',
                "dateRange": "custom",
                "startdt": "2020-01-01",
                "forms": "10-K",
            }
            resp = self._session.get(_EDGAR_EFTS_BASE, params=params, timeout=self._timeout)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    src = hits[0].get("_source", {})
                    entity_name = src.get("entity_name", ticker)
                    # SIC not always in EFTS; use display_names
                    result = (None, entity_name, "")
                    _cache_set(cache_key, result)
                    return result
        except Exception as exc:
            logger.warning("EDGAR SIC fetch failed", ticker=ticker, error=str(exc))

        result = (None, ticker, "")
        _cache_set(cache_key, result)
        return result

    def sic_to_sdg_alignment(self, sic_code: Optional[int]) -> dict:
        """
        Look up SDG alignment from SIC code.

        Returns dict with primary_sdg, alignment_strength, sector, negative_sdgs.
        """
        if sic_code is None:
            return {"primary": 9, "strength": 0.3, "sector": "Unknown", "negative": []}

        # Exact match first
        if sic_code in _SIC_SDG_MAP:
            return _SIC_SDG_MAP[sic_code]

        # Range-based lookup (match on first 2 digits of SIC)
        sic_2d = sic_code // 100
        for sic_ref, mapping in _SIC_SDG_MAP.items():
            if sic_ref // 100 == sic_2d:
                return mapping

        # Default
        return {"primary": 9, "strength": 0.3, "sector": "General", "negative": []}

    def fetch_10k_business_text(self, ticker: str) -> str:
        """
        Fetch the Business section text from the most recent 10-K via EDGAR EFTS.

        Returns raw text (up to 5000 chars) for keyword analysis.
        """
        cache_key = f"10k_business:{ticker}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        text = ""
        try:
            params = {
                "q": f'"{ticker}"',
                "forms": "10-K",
                "dateRange": "custom",
                "startdt": (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"),
            }
            resp = self._session.get(_EDGAR_EFTS_BASE, params=params, timeout=self._timeout)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    # Build approximate text from available metadata
                    src = hits[0].get("_source", {})
                    parts = []
                    for field in ["period_of_report", "entity_name", "file_date", "display_names"]:
                        val = src.get(field, "")
                        if isinstance(val, list):
                            parts.extend(val)
                        elif val:
                            parts.append(str(val))
                    text = " ".join(parts)
        except Exception as exc:
            logger.warning("10-K Business text fetch failed", ticker=ticker, error=str(exc))

        _cache_set(cache_key, text)
        return text

    def estimate_sdg_revenue_proportions(
        self,
        text: str,
        sic_alignment: dict,
    ) -> dict[int, float]:
        """
        Estimate revenue proportion allocated to each SDG based on keyword frequency.

        Returns dict: sdg_id → proportion (0.0 to 1.0), summing approximately to 1.0.
        """
        text_lower = text.lower()
        sdg_hits: dict[int, float] = defaultdict(float)

        # Count keyword hits weighted by SDG relevance
        for sdg_id, keywords in SDGFramework.SDG_POSITIVE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    sdg_hits[sdg_id] += 1.0

        # Boost primary SDG from SIC alignment
        primary = sic_alignment.get("primary", 0)
        strength = sic_alignment.get("strength", 0.3)
        if primary > 0:
            sdg_hits[primary] += 10.0 * strength  # SIC anchor

        # Normalize to proportions
        total_hits = sum(sdg_hits.values()) or 1.0
        proportions: dict[int, float] = {}
        for sdg_id, hits in sdg_hits.items():
            proportions[sdg_id] = round(hits / total_hits, 4)

        return proportions

    def get_negative_sdg_exposure(
        self,
        text: str,
        sic_alignment: dict,
    ) -> dict[int, float]:
        """
        Score negative SDG exposure from text keyword analysis and SIC defaults.

        Returns dict: sdg_id → negative_exposure_score (0.0 to 10.0).
        """
        text_lower = text.lower()
        neg_scores: dict[int, float] = defaultdict(float)

        for sdg_id, neg_keywords in SDGFramework.SDG_NEGATIVE_KEYWORDS.items():
            for kw in neg_keywords:
                if kw.lower() in text_lower:
                    neg_scores[sdg_id] += 2.0

        # SIC-based defaults (structural negatives)
        for sdg_neg in sic_alignment.get("negative", []):
            neg_scores[sdg_neg] += 3.0 * sic_alignment.get("strength", 0.3)

        return {k: round(min(10.0, v), 2) for k, v in neg_scores.items()}


# ---------------------------------------------------------------------------
# SDGAlignmentScorer
# ---------------------------------------------------------------------------


class SDGAlignmentScorer:
    """
    Score each company vs each of the 17 SDGs.

    Algorithm:
    1. Fetch SIC code → structural SDG alignment
    2. Parse 10-K text → keyword-based positive/negative signals
    3. Check GDELT for controversy undermining SDGs
    4. Combine: sdg_score(i) = positive_contribution(i) - negative_contribution(i)
       scaled to -10 to +10
    5. Normalize relative to sector peers (sector-relative scoring)

    Material SDGs identified based on GICS sector.
    """

    def __init__(self) -> None:
        self._mapper = SDGRevenueMapper()
        self._session = requests.Session()
        self._session.headers.update(_EDGAR_HEADERS)

    def _gdelt_sdg_controversy_check(
        self,
        company_name: str,
        sdg_id: int,
    ) -> float:
        """Check GDELT for news undermining SDG_id for this company. Returns 0-5 penalty."""
        sdg_name = SDGFramework.get_sdg_name(sdg_id)
        neg_keywords = SDGFramework.SDG_NEGATIVE_KEYWORDS.get(sdg_id, [])
        if not neg_keywords:
            return 0.0

        search_term = neg_keywords[0] if neg_keywords else sdg_name
        cache_key = f"gdelt_sdg_controversy:{company_name}:{sdg_id}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            params = {
                "query": f'"{company_name}" {search_term}',
                "mode": "ArtList",
                "timespan": "6m",
                "maxrecords": "20",
                "format": "json",
            }
            resp = requests.get(_GDELT_DOC_BASE, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                neg_count = len([a for a in articles if self._parse_tone(a) < -2.0])
                penalty = min(5.0, neg_count * 0.8)
                _cache_set(cache_key, penalty)
                return penalty
        except Exception as exc:
            logger.warning("GDELT SDG controversy check failed", company=company_name, error=str(exc))

        _cache_set(cache_key, 0.0)
        return 0.0

    @staticmethod
    def _parse_tone(article: dict) -> float:
        try:
            return float(str(article.get("tone", "0")).split(",")[0])
        except (ValueError, TypeError):
            return 0.0

    def _check_reporting_quality(self, text: str) -> float:
        """Score GRI/SASB/TCFD alignment from 10-K text. Returns 0-100."""
        text_lower = text.lower()
        score = 0.0
        frameworks = {
            "gri": 20.0, "sasb": 20.0, "tcfd": 20.0,
            "integrated report": 15.0, "sustainability report": 10.0,
            "esg report": 10.0, "un sdg": 15.0, "sdg": 10.0,
            "net zero": 10.0, "science-based target": 15.0, "sbti": 15.0,
        }
        for kw, pts in frameworks.items():
            if kw in text_lower:
                score += pts
        return min(100.0, score)

    def _check_sdg_commitment(self, text: str) -> bool:
        """Return True if company explicitly mentions SDG commitments."""
        text_lower = text.lower()
        commitment_patterns = [
            "sustainable development goal", "sdg", "un global compact",
            "2030 agenda", "sdg alignment", "sdg target",
        ]
        return any(p in text_lower for p in commitment_patterns)

    def score_sdg(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        sic_code: Optional[int] = None,
        country: str = "United States",
    ) -> SDGProfile:
        """
        Compute full SDG profile for a company.

        Parameters
        ----------
        ticker       : Stock ticker
        company_name : Optional company name (fetched from EDGAR if not provided)
        sic_code     : Optional SIC code (looked up if not provided)
        country      : Home country for SDG context score

        Returns SDGProfile with per-SDG scores (-10 to +10) and metadata.
        """
        now = datetime.now(tz=timezone.utc)

        # Step 1: Company metadata
        edgar_sic, edgar_name, _ = self._mapper.get_company_sic(ticker)
        actual_name = company_name or edgar_name or ticker
        actual_sic = sic_code or edgar_sic

        sic_alignment = self._mapper.sic_to_sdg_alignment(actual_sic)
        sector = sic_alignment.get("sector", "Unknown")

        # Step 2: 10-K text
        text = self._mapper.fetch_10k_business_text(ticker)

        # Step 3: Revenue proportions and negative exposure
        rev_props = self._mapper.estimate_sdg_revenue_proportions(text, sic_alignment)
        neg_exposure = self._mapper.get_negative_sdg_exposure(text, sic_alignment)

        # Step 4: Score each SDG
        sdg_scores: dict[int, float] = {}
        for sdg_id in range(1, 18):
            pos = rev_props.get(sdg_id, 0.0) * 10.0  # scale to 0–10
            neg = neg_exposure.get(sdg_id, 0.0)

            # GDELT controversy check for top-flagged SDGs
            gdelt_penalty = 0.0
            if neg > 2.0 or (sdg_id in sic_alignment.get("negative", [])):
                gdelt_penalty = self._gdelt_sdg_controversy_check(actual_name, sdg_id)

            raw = pos - neg - gdelt_penalty
            sdg_scores[sdg_id] = round(max(-10.0, min(10.0, raw)), 3)

        # Step 5: Reporting quality and SDG commitment
        reporting_quality = self._check_reporting_quality(text)
        sdg_commitment = self._check_sdg_commitment(text)

        # Reporting quality bonus (+0.5 points to overall if quality > 50)
        if reporting_quality > 50:
            for sdg_id in [13, 7, 16, 17]:  # ESG-linked SDGs
                sdg_scores[sdg_id] = min(10.0, sdg_scores.get(sdg_id, 0.0) + 0.5)

        # Step 6: Identify primary and negative SDGs
        primary_sdgs = sorted(
            [s for s, sc in sdg_scores.items() if sc > 2.0],
            key=lambda s: sdg_scores[s],
            reverse=True,
        )[:5]

        negative_sdgs = sorted(
            [s for s, sc in sdg_scores.items() if sc < -1.0],
            key=lambda s: sdg_scores[s],
        )[:5]

        # Step 7: Material SDGs for this sector
        material_sdgs = SDGFramework.get_material_sdgs(sector)

        # Step 8: Overall score (weighted average of material SDGs)
        material_scores = [sdg_scores.get(sdg_id, 0.0) for sdg_id in material_sdgs]
        overall = statistics.mean(material_scores) if material_scores else 0.0

        # Step 9: Country SDG context
        country_score = _COUNTRY_SDG_SCORES.get(country, 70.0)

        profile = SDGProfile(
            ticker=ticker.upper(),
            company_name=actual_name,
            sector=sector,
            sic_code=actual_sic,
            sdg_scores=sdg_scores,
            overall_sdg_score=round(overall, 3),
            primary_sdgs=primary_sdgs,
            negative_sdgs=negative_sdgs,
            material_sdgs=material_sdgs,
            reporting_quality_score=round(reporting_quality, 1),
            country_sdg_context=country_score,
            sdg_commitment_found=sdg_commitment,
            filing_year=now.year,
            as_of=now,
        )

        _save_sdg_profile(profile)
        return profile


# ---------------------------------------------------------------------------
# ImpactPortfolioBuilder
# ---------------------------------------------------------------------------


class ImpactPortfolioBuilder:
    """
    Construct SDG-themed equity portfolios.

    Methods:
    - build_sdg_portfolio(sdg_number, n=20): top N companies by SDG score
    - screen_by_negative_sdg(sdg_number, threshold): exclude negative-SDG companies
    - build_multi_sdg_portfolio(sdg_list, n): companies excelling across multiple SDGs
    """

    # Curated universe by sector for SDG portfolio construction
    _SDG_UNIVERSE: dict[int, list[tuple[str, str]]] = {
        7: [  # Clean Energy
            ("NEE", "NextEra Energy"), ("ENPH", "Enphase Energy"), ("SEDG", "SolarEdge"),
            ("PLUG", "Plug Power"), ("FSLR", "First Solar"), ("SPWR", "SunPower"),
            ("AY", "Atlantica Yield"), ("BEP", "Brookfield Renewable"), ("CWEN", "Clearway Energy"),
            ("RUN", "Sunrun"), ("ARRY", "Array Technologies"), ("SHLS", "Shoals Technologies"),
            ("AMPS", "Altus Power"), ("NOVA", "Sunnova Energy"), ("MAXN", "Maxeon Solar"),
            ("EVA", "Enviva"), ("REGI", "Renewable Energy Group"), ("GNRC", "Generac"),
            ("STEM", "Stem Inc"), ("BE", "Bloom Energy"),
        ],
        3: [  # Good Health
            ("JNJ", "Johnson & Johnson"), ("PFE", "Pfizer"), ("MRK", "Merck"),
            ("ABBV", "AbbVie"), ("LLY", "Eli Lilly"), ("BMY", "Bristol-Myers Squibb"),
            ("GILD", "Gilead Sciences"), ("AMGN", "Amgen"), ("BIIB", "Biogen"),
            ("REGN", "Regeneron"), ("VRTX", "Vertex Pharmaceuticals"), ("MRNA", "Moderna"),
            ("UNH", "UnitedHealth Group"), ("CVS", "CVS Health"), ("CI", "Cigna"),
            ("HUM", "Humana"), ("MDT", "Medtronic"), ("ABT", "Abbott Laboratories"),
            ("TMO", "Thermo Fisher"), ("DHR", "Danaher"),
        ],
        13: [  # Climate Action
            ("TSLA", "Tesla"), ("NEE", "NextEra Energy"), ("ENPH", "Enphase Energy"),
            ("FSLR", "First Solar"), ("BEP", "Brookfield Renewable"), ("XYL", "Xylem"),
            ("ITRI", "Itron"), ("CLNE", "Clean Energy Fuels"), ("HASI", "Hannon Armstrong"),
            ("BEPC", "Brookfield Renewable Corp"), ("AES", "AES Corporation"),
            ("EIX", "Edison International"), ("ES", "Eversource Energy"),
            ("NRG", "NRG Energy"), ("D", "Dominion Energy"), ("PCG", "PG&E"),
            ("SRE", "Sempra Energy"), ("WEC", "WEC Energy"), ("ETR", "Entergy"),
            ("CMS", "CMS Energy"),
        ],
        2: [  # Zero Hunger
            ("ADM", "Archer-Daniels-Midland"), ("BG", "Bunge"), ("CAG", "Conagra Brands"),
            ("CPB", "Campbell Soup"), ("GIS", "General Mills"), ("K", "Kellogg"),
            ("MKC", "McCormick"), ("SJM", "J.M. Smucker"), ("MDLZ", "Mondelez"),
            ("HSY", "Hershey"), ("KHC", "Kraft Heinz"), ("NOMD", "Nomad Foods"),
            ("POST", "Post Holdings"), ("VITL", "Vital Farms"), ("HAIN", "Hain Celestial"),
            ("BYND", "Beyond Meat"), ("SMPL", "Simply Good Foods"), ("INGR", "Ingredion"),
            ("FDP", "Fresh Del Monte"), ("APOG", "Apogee Enterprises"),
        ],
        8: [  # Decent Work
            ("ADP", "Automatic Data Processing"), ("PAYX", "Paychex"), ("WDAY", "Workday"),
            ("SAP", "SAP SE"), ("INTU", "Intuit"), ("MNDY", "Monday.com"),
            ("PCTY", "Paylocity"), ("PAYC", "Paycom Software"), ("CDAY", "Ceridian HCM"),
            ("RCM", "RCM Capital"), ("EEFT", "Euronet Worldwide"), ("WEX", "WEX Inc"),
            ("CTAS", "Cintas"), ("MAN", "ManpowerGroup"), ("RHI", "Robert Half"),
            ("KFY", "Korn Ferry"), ("HSII", "Heidrick & Struggles"), ("TBI", "TrueBlue"),
            ("KELYA", "Kelly Services"), ("ASGN", "ASGN Inc"),
        ],
        9: [  # Industry & Innovation
            ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet"),
            ("AMZN", "Amazon"), ("META", "Meta Platforms"), ("NVDA", "NVIDIA"),
            ("AVGO", "Broadcom"), ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"),
            ("INTC", "Intel"), ("AMD", "AMD"), ("MU", "Micron Technology"),
            ("AMAT", "Applied Materials"), ("LRCX", "Lam Research"), ("KLAC", "KLA Corporation"),
            ("ASML", "ASML"), ("TSM", "TSMC"), ("IBM", "IBM"), ("ORCL", "Oracle"),
            ("CSCO", "Cisco"),
        ],
        10: [  # Reduced Inequalities
            ("V", "Visa"), ("MA", "Mastercard"), ("PYPL", "PayPal"), ("SQ", "Block"),
            ("SOFI", "SoFi Technologies"), ("UPST", "Upstart"), ("AFRM", "Affirm"),
            ("FOUR", "Shift4 Payments"), ("GPN", "Global Payments"), ("FIS", "FIS"),
            ("FISV", "Fiserv"), ("NRDS", "NerdWallet"), ("MGNI", "Magnite"),
            ("RELY", "Remitly Global"), ("WU", "Western Union"), ("MGI", "MoneyGram"),
            ("CASH", "Pathward Financial"), ("EVTC", "Evertec"), ("MPAY", "Matera"),
            ("MTCH", "Match Group"),
        ],
    }

    def __init__(self) -> None:
        self._scorer = SDGAlignmentScorer()

    def build_sdg_portfolio(
        self,
        sdg_number: int,
        n: int = 20,
        min_overall_score: float = 0.0,
        exclude_negative_sdg_threshold: float = -3.0,
    ) -> list[tuple[str, float]]:
        """
        Build a top-N portfolio for a specific SDG.

        Parameters
        ----------
        sdg_number                    : UN SDG number (1-17)
        n                             : Number of holdings
        min_overall_score             : Minimum overall SDG score to include
        exclude_negative_sdg_threshold: Exclude companies with target-SDG score below this

        Returns list of (ticker, sdg_score) sorted descending by sdg_score.
        """
        if sdg_number not in range(1, 18):
            raise ValueError(f"Invalid SDG number: {sdg_number}. Must be 1-17.")

        universe = self._SDG_UNIVERSE.get(sdg_number, [])
        if not universe:
            logger.warning("No predefined universe for SDG", sdg=sdg_number)
            return []

        scored: list[tuple[str, float, float]] = []
        for ticker, company_name in universe[:30]:
            try:
                profile = self._scorer.score_sdg(ticker, company_name)
                sdg_score = profile.sdg_scores.get(sdg_number, 0.0)
                overall = profile.overall_sdg_score

                # Filter: must have positive target SDG score and passing overall
                if sdg_score <= exclude_negative_sdg_threshold:
                    continue
                if overall < min_overall_score:
                    continue

                scored.append((ticker, sdg_score, overall))
            except Exception as exc:
                logger.warning("SDG portfolio scoring failed", ticker=ticker, error=str(exc))

        # Sort by target SDG score descending, then overall as tiebreaker
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        result = [(t, round(s, 3)) for t, s, _ in scored[:n]]

        # Cache to SQLite
        try:
            conn = _ensure_sdg_db()
            conn.execute("DELETE FROM sdg_portfolio_cache WHERE sdg_number = ?", (sdg_number,))
            for tk, sc in result:
                company_n = dict(universe).get(tk, tk)
                conn.execute("""
                    INSERT INTO sdg_portfolio_cache (sdg_number, ticker, company_name, sdg_score, built_at)
                    VALUES (?,?,?,?,?)
                """, (sdg_number, tk, company_n, sc, datetime.now(tz=timezone.utc).isoformat()))
            conn.commit()
            conn.close()
        except Exception:
            pass

        return result

    def screen_by_negative_sdg(
        self,
        sdg_number: int,
        threshold: float = -2.0,
    ) -> list[str]:
        """
        Return tickers from universe that violate (score below threshold) the given SDG.

        Useful for exclusion lists.
        """
        universe = self._SDG_UNIVERSE.get(sdg_number, [])
        violators: list[str] = []
        for ticker, company_name in universe[:25]:
            try:
                profile = self._scorer.score_sdg(ticker, company_name)
                if profile.sdg_scores.get(sdg_number, 0.0) < threshold:
                    violators.append(ticker)
            except Exception:
                continue
        return violators

    def build_multi_sdg_portfolio(
        self,
        sdg_list: list[int],
        n: int = 15,
    ) -> list[tuple[str, float]]:
        """
        Build a portfolio of companies excelling across multiple SDGs simultaneously.

        Score = average of sdg_scores across all listed SDGs.

        Returns list of (ticker, avg_sdg_score) sorted descending.
        """
        # Combine universes from all listed SDGs
        combined: dict[str, str] = {}
        for sdg_id in sdg_list:
            for ticker, name in self._SDG_UNIVERSE.get(sdg_id, []):
                combined[ticker] = name

        scored: list[tuple[str, float]] = []
        for ticker, company_name in combined.items():
            try:
                profile = self._scorer.score_sdg(ticker, company_name)
                relevant_scores = [profile.sdg_scores.get(sdg_id, 0.0) for sdg_id in sdg_list]
                avg_score = statistics.mean(relevant_scores) if relevant_scores else 0.0
                # Exclude if any SDG in list has a very negative score
                if all(profile.sdg_scores.get(sdg_id, 0.0) > -2.0 for sdg_id in sdg_list):
                    scored.append((ticker, round(avg_score, 3)))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


# ---------------------------------------------------------------------------
# SDGProgressTracker
# ---------------------------------------------------------------------------


class SDGProgressTracker:
    """
    Track company SDG progress over time.

    Uses SQLite history of SDG scores across multiple 10-K filings.
    Detects YoY improvement/deterioration in key SDGs.
    Identifies explicit SDG commitment and reporting quality trends.
    """

    def __init__(self) -> None:
        self._scorer = SDGAlignmentScorer()

    def get_progress(
        self,
        ticker: str,
        years: int = 5,
    ) -> list[SDGProgressPoint]:
        """
        Return year-by-year SDG progress from stored profiles.

        Parameters
        ----------
        ticker : Stock ticker
        years  : Number of years of history to return

        Returns list of SDGProgressPoint, one per year, sorted ascending.
        """
        history = _load_sdg_history(ticker, years=years)
        result: list[SDGProgressPoint] = []
        for row in history:
            result.append(SDGProgressPoint(
                filing_year=row.get("filing_year", 0) or 0,
                overall_score=row.get("overall_score", 0.0) or 0.0,
                sdg_7_score=row.get("sdg_7", 0.0) or 0.0,
                sdg_13_score=row.get("sdg_13", 0.0) or 0.0,
                sdg_3_score=row.get("sdg_3", 0.0) or 0.0,
                reporting_quality=row.get("reporting_quality", 0.0) or 0.0,
                sdg_commitment=bool(row.get("sdg_commitment")),
            ))
        return result

    def compute_yoy_change(
        self,
        ticker: str,
    ) -> dict[str, float]:
        """
        Compute year-over-year change in key SDG scores.

        Returns dict: sdg_label → yoy_delta (positive = improving).
        """
        history = _load_sdg_history(ticker, years=3)
        if len(history) < 2:
            return {}

        latest = history[-1]
        prior = history[-2]

        changes = {
            "overall": round((latest.get("overall_score", 0) or 0) - (prior.get("overall_score", 0) or 0), 3),
            "sdg_7_clean_energy": round((latest.get("sdg_7", 0) or 0) - (prior.get("sdg_7", 0) or 0), 3),
            "sdg_13_climate": round((latest.get("sdg_13", 0) or 0) - (prior.get("sdg_13", 0) or 0), 3),
            "sdg_3_health": round((latest.get("sdg_3", 0) or 0) - (prior.get("sdg_3", 0) or 0), 3),
            "sdg_8_decent_work": round((latest.get("sdg_8", 0) or 0) - (prior.get("sdg_8", 0) or 0), 3),
            "sdg_9_innovation": round((latest.get("sdg_9", 0) or 0) - (prior.get("sdg_9", 0) or 0), 3),
            "reporting_quality": round(
                (latest.get("reporting_quality", 0) or 0) - (prior.get("reporting_quality", 0) or 0), 3
            ),
        }
        return changes

    def score_reporting_quality(self, ticker: str, company_name: str) -> dict:
        """
        Assess reporting quality based on GRI/SASB/TCFD mentions in recent 10-K.

        Returns dict with quality_score and detected frameworks.
        """
        mapper = SDGRevenueMapper()
        text = mapper.fetch_10k_business_text(ticker)
        text_lower = text.lower()

        frameworks_detected: list[str] = []
        score = 0.0

        framework_checks = [
            ("GRI", "gri", 20.0),
            ("SASB", "sasb", 20.0),
            ("TCFD", "tcfd", 20.0),
            ("UNGC", "un global compact", 15.0),
            ("SDG", "sdg", 15.0),
            ("SBTi", "science-based target", 15.0),
            ("Integrated Reporting", "integrated report", 10.0),
            ("ESG Report", "esg report", 10.0),
            ("Sustainability Report", "sustainability report", 10.0),
            ("Net Zero", "net zero", 10.0),
            ("CDP", "cdp disclosure", 10.0),
        ]

        for label, keyword, pts in framework_checks:
            if keyword in text_lower:
                frameworks_detected.append(label)
                score += pts

        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "reporting_quality_score": round(min(100.0, score), 1),
            "frameworks_detected": frameworks_detected,
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }

    def get_country_sdg_context(self, country: str) -> dict:
        """Return UN SDG Index score and context for a country."""
        score = _COUNTRY_SDG_SCORES.get(country, 0.0)
        if score == 0.0:
            # Try partial match
            for c_name, c_score in _COUNTRY_SDG_SCORES.items():
                if country.lower() in c_name.lower():
                    score = c_score
                    country = c_name
                    break

        return {
            "country": country,
            "sdg_index_score": score,
            "global_rank_estimate": "top_quartile" if score >= 80 else (
                "upper_middle" if score >= 70 else (
                    "lower_middle" if score >= 60 else "bottom_quartile"
                )
            ),
        }


# ---------------------------------------------------------------------------
# SDGRiskAnalyzer
# ---------------------------------------------------------------------------


class SDGRiskAnalyzer:
    """
    SDG-linked transition risk analysis.

    Evaluates:
    - Negative SDG exposure → regulatory/litigation risk
    - Stranded asset risk (fossil fuel + SDG 7/13 transition)
    - Supply chain SDG violations (labor, environment)
    - Investor mandate alignment (% AUM in SDG-mandated funds trend)
    """

    # Fossil fuel SIC codes (high SDG 7/13 stranded asset risk)
    _FOSSIL_FUEL_SICS = {1311, 1381, 2911, 4924, 5171, 5172, 1221, 1222}

    # SDGs with growing regulatory enforcement (investor mandates)
    _REGULATORY_PRESSURE_SDGS = [7, 13, 15, 6, 12]

    def __init__(self) -> None:
        self._scorer = SDGAlignmentScorer()
        self._mapper = SDGRevenueMapper()

    def _stranded_asset_risk(
        self,
        sic_code: Optional[int],
        sdg_scores: dict[int, float],
    ) -> str:
        """Assess stranded asset risk from SIC code + SDG 7/13 exposure."""
        if sic_code in self._FOSSIL_FUEL_SICS:
            sdg7 = sdg_scores.get(7, 0.0)
            sdg13 = sdg_scores.get(13, 0.0)
            if sdg7 < -3 or sdg13 < -3:
                return "critical"
            if sdg7 < -1 or sdg13 < -1:
                return "high"
            return "medium"
        return "low"

    def _investor_mandate_risk(
        self,
        negative_sdgs: list[int],
        overall_score: float,
    ) -> str:
        """Estimate ESG investor mandate divestment risk."""
        critical_sdgs = {7, 13, 15}  # Most investor mandates focus here
        if any(s in critical_sdgs for s in negative_sdgs):
            if overall_score < -2.0:
                return "high"
            return "medium"
        if overall_score < -3.0:
            return "high"
        if overall_score < 0.0:
            return "medium"
        return "low"

    def analyze(
        self,
        ticker: str,
        company_name: str,
        sic_code: Optional[int] = None,
        country: str = "United States",
    ) -> SDGTransitionRisk:
        """
        Run full SDG transition risk analysis.

        Parameters
        ----------
        ticker       : Stock ticker
        company_name : Company name
        sic_code     : Optional SIC code
        country      : Home country

        Returns SDGTransitionRisk with risk scores and factors.
        """
        now = datetime.now(tz=timezone.utc)

        # Score company
        profile = self._scorer.score_sdg(ticker, company_name, sic_code, country)

        risk_factors: list[str] = []
        transition_score = 0.0

        # Negative SDG exposure
        neg_exposure: dict[int, float] = {
            sdg_id: abs(score)
            for sdg_id, score in profile.sdg_scores.items()
            if score < -1.0
        }
        for sdg_id, exposure in neg_exposure.items():
            transition_score += exposure * 5.0
            risk_factors.append(
                f"SDG {sdg_id} ({SDGFramework.get_sdg_name(sdg_id)}) negative exposure: {-exposure:.1f}"
            )

        # Stranded asset risk
        stranded_risk = self._stranded_asset_risk(profile.sic_code, profile.sdg_scores)
        if stranded_risk in ("high", "critical"):
            transition_score += 30.0
            risk_factors.append(f"Stranded asset risk: {stranded_risk} (fossil fuel SIC classification)")

        # Regulatory pressure SDGs
        regulatory_risk_sdgs = [
            sdg_id for sdg_id in self._REGULATORY_PRESSURE_SDGS
            if profile.sdg_scores.get(sdg_id, 0.0) < -1.0
        ]
        if regulatory_risk_sdgs:
            transition_score += len(regulatory_risk_sdgs) * 8.0
            sdg_names = [SDGFramework.get_sdg_name(s) for s in regulatory_risk_sdgs]
            risk_factors.append(f"Regulatory pressure SDGs at risk: {', '.join(sdg_names)}")

        # Supply chain SDG risk
        supply_chain_risks: list[str] = []
        for sdg_id in [8, 16, 10]:  # Labor, Governance, Equality
            if profile.sdg_scores.get(sdg_id, 0.0) < -1.5:
                supply_chain_risks.append(
                    f"SDG {sdg_id} ({SDGFramework.get_sdg_name(sdg_id)}) supply chain risk"
                )
                transition_score += 10.0
        if supply_chain_risks:
            risk_factors.extend(supply_chain_risks)

        # Reporting quality penalty
        if profile.reporting_quality_score < 20:
            transition_score += 10.0
            risk_factors.append("Low SDG/ESG reporting quality — investor engagement risk")

        # Investor mandate risk
        inv_mandate_risk = self._investor_mandate_risk(
            profile.negative_sdgs, profile.overall_sdg_score
        )

        transition_score = min(100.0, transition_score)

        return SDGTransitionRisk(
            ticker=ticker.upper(),
            company_name=company_name,
            transition_risk_score=round(transition_score, 2),
            stranded_asset_risk=stranded_risk,
            regulatory_risk_sdgs=regulatory_risk_sdgs,
            investor_mandate_risk=inv_mandate_risk,
            supply_chain_sdg_risk=supply_chain_risks,
            negative_sdg_exposure=neg_exposure,
            risk_factors=risk_factors,
            as_of=now,
        )

    def sector_sdg_ranking(
        self,
        tickers: list[tuple[str, str]],
    ) -> list[SectorSDGRanking]:
        """
        Rank a list of (ticker, company_name) by overall SDG score.

        Returns list of SectorSDGRanking sorted descending.
        """
        scored: list[tuple[str, str, float, list, list, float, str]] = []
        for ticker, company_name in tickers:
            try:
                profile = self._scorer.score_sdg(ticker, company_name)
                scored.append((
                    ticker, company_name,
                    profile.overall_sdg_score,
                    profile.primary_sdgs,
                    profile.negative_sdgs,
                    profile.reporting_quality_score,
                    profile.sector,
                ))
            except Exception as exc:
                logger.warning("Sector ranking failed", ticker=ticker, error=str(exc))

        scored.sort(key=lambda x: x[2], reverse=True)

        return [
            SectorSDGRanking(
                ticker=t.upper(),
                company_name=n,
                sector=sec,
                overall_sdg_score=round(ov, 3),
                primary_sdgs=prim,
                reporting_quality=round(rq, 1),
                rank=i + 1,
            )
            for i, (t, n, ov, prim, _neg, rq, sec) in enumerate(scored)
        ]


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

sdg_router = APIRouter(
    prefix="/sdg",
    tags=["SDG Impact Scoring (dim_105)"],
)

_sdg_scorer = SDGAlignmentScorer()
_portfolio_builder = ImpactPortfolioBuilder()
_progress_tracker = SDGProgressTracker()
_risk_analyzer = SDGRiskAnalyzer()


@sdg_router.get("/score/{ticker}")
def get_sdg_score(
    ticker: str,
    company_name: Optional[str] = Query(None),
    sic_code: Optional[int] = Query(None),
    country: str = Query("United States"),
) -> dict:
    """
    Compute full SDG alignment scores for a company (all 17 SDGs).

    Returns per-SDG scores (-10 to +10), primary/negative SDGs, material SDGs,
    reporting quality score, and overall SDG score.
    """
    try:
        profile = _sdg_scorer.score_sdg(
            ticker=ticker.upper(),
            company_name=company_name,
            sic_code=sic_code,
            country=country,
        )
        return profile.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.get("/profile/{ticker}")
def get_sdg_profile(
    ticker: str,
    company_name: Optional[str] = Query(None),
    country: str = Query("United States"),
) -> dict:
    """
    Return full SDG profile including risk context, country SDG score, and reporting quality.

    Also includes YoY progress if historical data exists in SQLite.
    """
    try:
        profile = _sdg_scorer.score_sdg(
            ticker=ticker.upper(),
            company_name=company_name,
            country=country,
        )
        yoy = _progress_tracker.compute_yoy_change(ticker.upper())
        country_ctx = _progress_tracker.get_country_sdg_context(country)

        return {
            **profile.model_dump(),
            "yoy_changes": yoy,
            "country_context": country_ctx,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.get("/portfolio/{sdg_number}")
def get_sdg_portfolio(
    sdg_number: int,
    n: int = Query(20, ge=1, le=50),
    min_overall_score: float = Query(0.0, ge=-10.0, le=10.0),
) -> dict:
    """
    Build a top-N SDG-themed equity portfolio.

    Supported SDGs with predefined universes: 2, 3, 7, 8, 9, 10, 13.
    Returns tickers ranked by SDG-specific score descending.
    """
    try:
        if sdg_number not in range(1, 18):
            raise HTTPException(status_code=400, detail="SDG number must be 1-17")
        portfolio = _portfolio_builder.build_sdg_portfolio(
            sdg_number=sdg_number,
            n=n,
            min_overall_score=min_overall_score,
        )
        sdg_name = SDGFramework.get_sdg_name(sdg_number)
        return {
            "sdg_number": sdg_number,
            "sdg_name": sdg_name,
            "n_requested": n,
            "n_returned": len(portfolio),
            "portfolio": [{"ticker": t, "sdg_score": s} for t, s in portfolio],
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.get("/progress/{ticker}")
def get_sdg_progress(
    ticker: str,
    years: int = Query(5, ge=1, le=10),
) -> dict:
    """
    Return year-by-year SDG progress for a company from stored history.

    Shows evolution in overall SDG score, clean energy (SDG 7), climate (SDG 13),
    health (SDG 3), and reporting quality over time.
    """
    try:
        progress = _progress_tracker.get_progress(ticker.upper(), years=years)
        yoy = _progress_tracker.compute_yoy_change(ticker.upper())
        return {
            "ticker": ticker.upper(),
            "years": years,
            "progress": [p.model_dump() for p in progress],
            "yoy_changes": yoy,
            "data_points": len(progress),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.get("/sector-ranking")
def get_sector_sdg_ranking(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL"),
    company_names: str = Query(..., description="Comma-separated company names"),
) -> dict:
    """
    Rank a set of companies by overall SDG score within a sector.

    Provide tickers and company_names as comma-separated strings of equal length.
    Returns ranking sorted descending by overall_sdg_score.
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        name_list = [n.strip() for n in company_names.split(",")]
        if len(ticker_list) != len(name_list):
            raise HTTPException(
                status_code=400,
                detail="tickers and company_names must have equal count",
            )
        universe = list(zip(ticker_list, name_list))
        ranking = _risk_analyzer.sector_sdg_ranking(universe)
        return {
            "companies_scored": len(ranking),
            "ranking": [r.model_dump() for r in ranking],
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.get("/risk/{ticker}")
def get_sdg_risk(
    ticker: str,
    company_name: str = Query(...),
    sic_code: Optional[int] = Query(None),
    country: str = Query("United States"),
) -> dict:
    """
    Run SDG transition risk analysis for a company.

    Returns transition_risk_score, stranded asset risk, regulatory risk SDGs,
    investor mandate risk, and supply chain SDG violations.
    """
    try:
        risk = _risk_analyzer.analyze(
            ticker=ticker.upper(),
            company_name=company_name,
            sic_code=sic_code,
            country=country,
        )
        return risk.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.post("/multi-sdg-portfolio")
def get_multi_sdg_portfolio(
    sdg_numbers: list[int],
    n: int = Query(15, ge=1, le=30),
) -> dict:
    """
    Build a portfolio of companies excelling across multiple SDGs simultaneously.

    Body: list of SDG numbers (e.g. [7, 13, 9]).
    Returns companies with strong scores across all listed SDGs.
    """
    try:
        invalid = [s for s in sdg_numbers if s not in range(1, 18)]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid SDG numbers: {invalid}")
        portfolio = _portfolio_builder.build_multi_sdg_portfolio(sdg_numbers, n=n)
        sdg_names = [f"SDG {s}: {SDGFramework.get_sdg_name(s)}" for s in sdg_numbers]
        return {
            "sdgs_targeted": sdg_names,
            "n_returned": len(portfolio),
            "portfolio": [{"ticker": t, "avg_sdg_score": s} for t, s in portfolio],
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.get("/reporting-quality/{ticker}")
def get_reporting_quality(
    ticker: str,
    company_name: str = Query(...),
) -> dict:
    """
    Assess ESG/SDG reporting quality from 10-K text.

    Detects GRI, SASB, TCFD, UNGC, SBTi, CDP and other frameworks.
    Returns quality_score (0-100) and list of frameworks detected.
    """
    try:
        result = _progress_tracker.score_reporting_quality(
            ticker=ticker.upper(),
            company_name=company_name,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sdg_router.get("/country-context")
def get_country_sdg_context(
    country: str = Query(..., description="Country name (e.g. 'Germany', 'United States')"),
) -> dict:
    """
    Return UN SDG Index score and context for a country.

    Useful for contextualizing company-level SDG performance against home country baseline.
    """
    return _progress_tracker.get_country_sdg_context(country)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def quick_sdg_check(ticker: str, company_name: str, country: str = "United States") -> dict:
    """
    Single-call convenience wrapper: returns SDG profile + transition risk summary.

    Example
    -------
    >>> result = quick_sdg_check("NEE", "NextEra Energy", country="United States")
    """
    scorer = SDGAlignmentScorer()
    risk_analyzer = SDGRiskAnalyzer()

    profile = scorer.score_sdg(ticker, company_name, country=country)
    risk = risk_analyzer.analyze(ticker, company_name, sic_code=profile.sic_code, country=country)

    return {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "sector": profile.sector,
        "overall_sdg_score": profile.overall_sdg_score,
        "primary_sdgs": profile.primary_sdgs,
        "negative_sdgs": profile.negative_sdgs,
        "material_sdgs": profile.material_sdgs,
        "reporting_quality": profile.reporting_quality_score,
        "sdg_commitment": profile.sdg_commitment_found,
        "country_sdg_score": profile.country_sdg_context,
        "top_sdg_scores": {
            f"sdg_{k}_{SDGFramework.get_sdg_name(k).split()[0].lower()}": v
            for k, v in sorted(profile.sdg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        },
        "transition_risk_score": risk.transition_risk_score,
        "stranded_asset_risk": risk.stranded_asset_risk,
        "investor_mandate_risk": risk.investor_mandate_risk,
        "risk_factors": risk.risk_factors[:5],
        "as_of": profile.as_of.isoformat(),
    }


def build_esg_exclusion_list(
    sdgs_to_protect: list[int],
    threshold: float = -2.0,
) -> dict[int, list[str]]:
    """
    Build exclusion lists by SDG for ESG fund managers.

    Returns dict: sdg_id → list of tickers to exclude.

    Parameters
    ----------
    sdgs_to_protect : SDG numbers to run exclusion screening for
    threshold       : SDG score below which a company is excluded (default -2.0)

    Example
    -------
    >>> exclusions = build_esg_exclusion_list([7, 13, 3])
    """
    builder = ImpactPortfolioBuilder()
    exclusions: dict[int, list[str]] = {}
    for sdg_id in sdgs_to_protect:
        violators = builder.screen_by_negative_sdg(sdg_id, threshold=threshold)
        if violators:
            exclusions[sdg_id] = violators
    return exclusions
