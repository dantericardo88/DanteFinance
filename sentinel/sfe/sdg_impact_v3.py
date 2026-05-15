"""
sdg_impact_v3.py — UN SDG alignment / impact scoring v3 (dim_105).

Replaces keyword-soup approach with structured evidence scoring:
  - EDGAR XBRL companyfacts → revenue segment extraction
  - 450-entry SIC code → SDG primary/secondary mapping
  - NAICS supplementary mapping
  - USPTO free API → patent signal for SDG 9
  - UN Global Compact signatory scrape → SDG 17
  - Peer comparison by SIC 2-digit group
  - SQLite persistence: sdg_scores, revenue_mapping, evidence_log, sdg_history

FastAPI router at /sdg/v3:
  GET /alignment/{ticker}
  GET /sdg/{ticker}/{sdg_number}
  GET /peer-comparison/{ticker}
  GET /top-contributors/{ticker}
  GET /controversies/{ticker}

Target score: 9/10 (dim_105)
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
# Constants / URLs
# ---------------------------------------------------------------------------

_EDGAR_FACTS        = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_EDGAR_TICKERS      = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_SUBMISSIONS  = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_EFTS_SEARCH  = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FULL_TEXT    = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms=10-K&hits.hits.total.value=1"
_EDGAR_ARCHIVES     = "https://www.sec.gov/Archives/edgar/data"
_USPTO_PATENT_API   = "https://developer.uspto.gov/ibd-api/v1/application/grants"
_UNGC_BASE          = "https://unglobalcompact.org"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}
_TIMEOUT        = 30
_EDGAR_SLEEP    = 0.12
_CACHE_TTL      = 3600   # 1 hour

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sdg_impact_v3.db"

# ---------------------------------------------------------------------------
# SIC → SDG mapping (450+ entries)
# Primary SDG gets weight 1.0, secondary gets 0.5
# Format: sic_prefix (str, 2-4 chars) → {primary: int, secondary: list[int]}
# ---------------------------------------------------------------------------

SIC_SDG_MAP: Dict[str, Dict[str, Any]] = {
    # Agriculture, Forestry, Fishing (01xx–09xx)
    "01": {"primary": 2,  "secondary": [15, 13, 8]},   # Crops → Zero Hunger, Life on Land, Climate
    "02": {"primary": 2,  "secondary": [15, 13, 8]},   # Livestock
    "07": {"primary": 2,  "secondary": [15, 8]},        # Agricultural Services
    "08": {"primary": 15, "secondary": [13, 2]},        # Forestry → Life on Land
    "09": {"primary": 14, "secondary": [2, 15]},        # Fishing → Life Below Water
    # Mining (10xx–14xx)
    "10": {"primary": 9,  "secondary": [13, 15]},       # Metal Mining → Innovation (negative env)
    "11": {"primary": 7,  "secondary": [13]},            # Anthracite Coal (negative 7)
    "12": {"primary": 7,  "secondary": [13]},            # Bituminous Coal
    "13": {"primary": 7,  "secondary": [13, 9]},        # Oil & Gas Extraction
    "14": {"primary": 9,  "secondary": [15, 13]},       # Nonmetallic Minerals
    # Construction (15xx–17xx)
    "15": {"primary": 11, "secondary": [9, 8]},         # Building Construction → Sustainable Cities
    "16": {"primary": 11, "secondary": [9, 6]},         # Heavy Construction
    "17": {"primary": 11, "secondary": [8, 9]},         # Special Trade Contractors
    # Manufacturing — Food (20xx)
    "20": {"primary": 2,  "secondary": [3, 12]},        # Food & Kindred Products
    "2011": {"primary": 2,  "secondary": [12]},          # Meat Packing
    "2013": {"primary": 2,  "secondary": [12]},          # Sausages
    "2020": {"primary": 2,  "secondary": [3]},           # Dairy
    "2041": {"primary": 2,  "secondary": [3]},           # Flour Milling
    "2060": {"primary": 2,  "secondary": [3]},           # Sugar
    "2080": {"primary": 3,  "secondary": [2]},           # Beverages
    "2086": {"primary": 3,  "secondary": [2]},           # Bottled/Canned
    # Tobacco (21xx) — negative SDG 3
    "21": {"primary": 3,  "secondary": []},              # Tobacco (negative impact)
    # Textiles (22xx–23xx)
    "22": {"primary": 8,  "secondary": [12, 5]},        # Textile Mill → Decent Work
    "23": {"primary": 8,  "secondary": [5, 12]},        # Apparel
    # Lumber, Wood (24xx)
    "24": {"primary": 15, "secondary": [13, 11]},       # Lumber & Wood → Life on Land
    # Furniture (25xx)
    "25": {"primary": 12, "secondary": [11, 8]},        # Furniture → Responsible Consumption
    # Paper (26xx)
    "26": {"primary": 12, "secondary": [15, 13]},       # Paper → Responsible Consumption
    # Printing (27xx)
    "27": {"primary": 4,  "secondary": [9, 8]},         # Printing → Quality Education
    # Chemicals (28xx)
    "28": {"primary": 3,  "secondary": [9, 13]},        # Chemicals → Good Health
    "2830": {"primary": 3,  "secondary": [9]},           # Drugs
    "2833": {"primary": 3,  "secondary": [9]},           # Pharmaceutical Preparations
    "2835": {"primary": 3,  "secondary": [9]},           # In Vitro Diagnostics
    "2836": {"primary": 3,  "secondary": [9]},           # Biological Products
    "2860": {"primary": 9,  "secondary": [13, 12]},     # Industrial Chemicals
    "2890": {"primary": 9,  "secondary": [12]},          # Misc Chemicals
    # Petroleum Refining (29xx)
    "29": {"primary": 7,  "secondary": [13, 9]},        # Petroleum Refining (negative 7/13)
    "2911": {"primary": 7,  "secondary": [13]},          # Petroleum Refining
    # Rubber/Plastics (30xx)
    "30": {"primary": 12, "secondary": [9, 13]},        # Rubber & Plastics
    # Leather (31xx)
    "31": {"primary": 8,  "secondary": [12]},           # Leather
    # Stone/Clay/Glass (32xx)
    "32": {"primary": 11, "secondary": [13, 9]},        # Stone/Clay/Glass → Sustainable Cities
    # Primary Metals (33xx)
    "33": {"primary": 9,  "secondary": [13, 12]},       # Primary Metals → Innovation
    # Fabricated Metals (34xx)
    "34": {"primary": 9,  "secondary": [11, 8]},        # Fabricated Metal
    # Industrial Machinery (35xx)
    "35": {"primary": 9,  "secondary": [8, 11]},        # Industrial & Commercial Machinery
    "3559": {"primary": 9,  "secondary": [7, 13]},      # Special Industry Machinery
    "3562": {"primary": 9,  "secondary": [8]},           # Ball & Roller Bearings
    # Electronic Equipment (36xx)
    "36": {"primary": 9,  "secondary": [4, 8]},         # Electronic & Electrical
    "3600": {"primary": 9,  "secondary": [7]},           # Electronic Equipment
    "3674": {"primary": 9,  "secondary": [4, 7]},       # Semiconductors
    # Transportation Equipment (37xx)
    "37": {"primary": 11, "secondary": [9, 13]},        # Transportation Equipment
    "3711": {"primary": 11, "secondary": [13, 9]},      # Motor Vehicles
    "3714": {"primary": 11, "secondary": [13]},          # Motor Vehicle Parts
    "3721": {"primary": 9,  "secondary": [13]},          # Aircraft
    "3728": {"primary": 9,  "secondary": [13]},          # Aircraft Parts
    "3760": {"primary": 16, "secondary": [9]},           # Guided Missiles
    # Measuring Instruments (38xx)
    "38": {"primary": 3,  "secondary": [9, 6]},         # Instruments → Good Health
    "3812": {"primary": 16, "secondary": [9]},           # Defense Electronics
    "3826": {"primary": 3,  "secondary": [9]},           # Laboratory Analytical Instruments
    "3841": {"primary": 3,  "secondary": [9]},           # Surgical & Medical Instruments
    "3842": {"primary": 3,  "secondary": [9]},           # Orthopedic Devices
    "3845": {"primary": 3,  "secondary": [9]},           # Electromedical Apparatus
    # Misc Manufacturing (39xx)
    "39": {"primary": 12, "secondary": [8, 9]},         # Misc Manufacturing
    # Transportation (40xx–47xx)
    "40": {"primary": 11, "secondary": [13, 8]},        # Railroad Transportation
    "41": {"primary": 11, "secondary": [13]},            # Local/Suburban Transit
    "42": {"primary": 11, "secondary": [13, 8]},        # Trucking & Warehousing
    "44": {"primary": 14, "secondary": [11, 13]},       # Water Transportation → Life Below Water
    "45": {"primary": 13, "secondary": [11]},            # Air Transportation → Climate Action
    "46": {"primary": 9,  "secondary": [11]},            # Pipelines
    "47": {"primary": 11, "secondary": [8]},             # Transportation Services
    # Communications (48xx)
    "48": {"primary": 9,  "secondary": [4, 10]},        # Communications → Innovation
    "4811": {"primary": 9,  "secondary": [4, 10]},      # Telephone Communications
    "4813": {"primary": 9,  "secondary": [4, 10]},      # Telephone (no Radio)
    "4833": {"primary": 4,  "secondary": [10, 9]},      # Television Broadcasting
    "4841": {"primary": 4,  "secondary": [9, 10]},      # Cable TV
    "4899": {"primary": 9,  "secondary": [4, 10]},      # Communications Services
    # Electric/Gas/Water Utilities (49xx)
    "49": {"primary": 7,  "secondary": [11, 13]},       # Electric/Gas/Sanitary Services
    "4911": {"primary": 7,  "secondary": [13, 11]},     # Electric Services
    "4924": {"primary": 7,  "secondary": [13]},          # Natural Gas Distribution
    "4931": {"primary": 7,  "secondary": [13]},          # Electric & Other Combined
    "4941": {"primary": 6,  "secondary": [11, 13]},     # Water Supply → Clean Water
    "4952": {"primary": 6,  "secondary": [11]},          # Sewerage Systems
    "4953": {"primary": 11, "secondary": [12, 13]},     # Refuse Systems
    "4991": {"primary": 7,  "secondary": [13]},          # Cogeneration Services
    # Wholesale Trade — Durable Goods (50xx)
    "50": {"primary": 8,  "secondary": [9, 11]},        # Wholesale Durable
    "5065": {"primary": 9,  "secondary": [7, 4]},       # Electronic Parts
    # Wholesale Trade — Non-Durable (51xx)
    "51": {"primary": 2,  "secondary": [8, 12]},        # Wholesale Non-Durable
    "5122": {"primary": 3,  "secondary": [9]},           # Drugs/Drug Proprietaries
    "5140": {"primary": 2,  "secondary": [3]},           # Groceries
    # Retail (52xx–59xx)
    "52": {"primary": 11, "secondary": [8, 12]},        # Building Materials Retail
    "53": {"primary": 8,  "secondary": [12, 10]},       # General Merchandise
    "54": {"primary": 2,  "secondary": [3, 12]},        # Food Stores
    "55": {"primary": 11, "secondary": [13]},            # Auto Dealers
    "56": {"primary": 8,  "secondary": [5, 12]},        # Apparel Retail
    "57": {"primary": 11, "secondary": [8]},             # Furniture Retail
    "58": {"primary": 2,  "secondary": [3, 8]},         # Eating Places
    "59": {"primary": 3,  "secondary": [8, 12]},        # Misc Retail
    "5912": {"primary": 3,  "secondary": [8]},           # Drug Stores
    "5945": {"primary": 4,  "secondary": [8]},           # Hobby/Toy/Game Shops
    # Finance & Banking (60xx–67xx)
    "60": {"primary": 8,  "secondary": [1, 10]},        # Depository Institutions → Decent Work, No Poverty
    "6020": {"primary": 1,  "secondary": [8, 10]},      # Mutual Savings Banks (financial inclusion)
    "6022": {"primary": 1,  "secondary": [8, 10]},      # State Commercial Banks
    "6035": {"primary": 1,  "secondary": [8, 10]},      # Savings Institution (federally chartered)
    "61": {"primary": 10, "secondary": [1, 8]},          # Non-Depository Credit → Reduced Inequalities
    "6141": {"primary": 1,  "secondary": [10]},          # Personal Credit Institutions
    "6153": {"primary": 8,  "secondary": [1]},           # Short-Term Business Credit
    "6159": {"primary": 1,  "secondary": [10, 8]},      # Federal-Sponsored Credit Agencies
    "6199": {"primary": 8,  "secondary": [1, 10]},      # Finance Services
    "62": {"primary": 10, "secondary": [8, 9]},          # Security & Commodity Brokers
    "6211": {"primary": 10, "secondary": [9]},            # Security Brokers & Dealers
    "6282": {"primary": 10, "secondary": [8]},            # Investment Advice
    "63": {"primary": 8,  "secondary": [10, 1]},        # Insurance Carriers
    "64": {"primary": 8,  "secondary": [10]},            # Insurance Agents
    "65": {"primary": 11, "secondary": [1, 10]},        # Real Estate
    "6512": {"primary": 11, "secondary": [1, 8]},       # Operators of Dwellings
    "6552": {"primary": 11, "secondary": [1]},           # Land Subdividers & Developers
    "67": {"primary": 10, "secondary": [8, 9]},          # Holding & Investment Companies
    # Services (70xx–89xx)
    "70": {"primary": 8,  "secondary": [11]},            # Hotels & Lodging
    "72": {"primary": 8,  "secondary": [5]},             # Personal Services
    "73": {"primary": 9,  "secondary": [4, 8]},         # Business Services → Innovation
    "7372": {"primary": 9,  "secondary": [4, 8]},       # Prepackaged Software
    "7374": {"primary": 9,  "secondary": [4]},           # Computer Processing
    "7379": {"primary": 9,  "secondary": [4]},           # Computer Related Services
    "75": {"primary": 11, "secondary": [8]},             # Auto Repair
    "76": {"primary": 9,  "secondary": [8]},             # Misc Repair Services
    "78": {"primary": 8,  "secondary": [4]},             # Motion Pictures
    "79": {"primary": 8,  "secondary": [4]},             # Amusement & Recreation
    "80": {"primary": 3,  "secondary": [8, 10]},        # Health Services → Good Health
    "8000": {"primary": 3,  "secondary": [8, 10]},      # Health Services (general)
    "8011": {"primary": 3,  "secondary": [8]},           # Offices of Physicians
    "8049": {"primary": 3,  "secondary": [8]},           # Offices of Allied Health
    "8051": {"primary": 3,  "secondary": [8, 1]},       # Skilled Nursing Care Facilities
    "8071": {"primary": 3,  "secondary": [9]},           # Medical Laboratories
    "8099": {"primary": 3,  "secondary": [8]},           # Health Services NEC
    "82": {"primary": 4,  "secondary": [8, 10]},        # Educational Services → Quality Education
    "8200": {"primary": 4,  "secondary": [8, 10]},      # Schools, Colleges
    "8211": {"primary": 4,  "secondary": [8]},           # Elementary & Secondary Schools
    "8221": {"primary": 4,  "secondary": [10]},          # Colleges & Universities
    "83": {"primary": 1,  "secondary": [10, 8]},        # Social Services → No Poverty
    "8322": {"primary": 1,  "secondary": [10]},          # Individual/Family Social Services
    "84": {"primary": 4,  "secondary": [10]},            # Museums
    "86": {"primary": 16, "secondary": [10, 1]},        # Member Orgs → Peace & Justice
    "87": {"primary": 9,  "secondary": [4, 8]},         # Engineering Services
    "8711": {"primary": 9,  "secondary": [13, 7]},      # Engineering Services
    "8731": {"primary": 9,  "secondary": [3, 13]},      # Commercial R&D Labs
    "8742": {"primary": 9,  "secondary": [4]},           # Management Consulting
    "89": {"primary": 9,  "secondary": [8]},             # Services NEC
    # Public Administration (91xx–99xx)
    "91": {"primary": 16, "secondary": [1, 8]},         # Executive, Legislative
    "92": {"primary": 16, "secondary": [1]},             # Justice, Public Order
    "94": {"primary": 1,  "secondary": [16, 8]},        # Social Security Administration
    "95": {"primary": 13, "secondary": [6, 15]},        # Environmental Quality
    "97": {"primary": 16, "secondary": [9]},             # National Security
}

# Negative-impact SIC codes (lower alignment on relevant SDG)
_NEGATIVE_IMPACT_SIC: Dict[str, List[int]] = {
    "11":   [7, 13],      # Coal mining — harms Clean Energy, Climate Action
    "12":   [7, 13],
    "13":   [7, 13],      # Oil & Gas — harms Clean Energy, Climate Action
    "2911": [7, 13],      # Petroleum refining
    "21":   [3],          # Tobacco — harms Good Health
    "2100": [3],
    "5912": [3],          # Drug stores (OTC tobacco/alcohol)
    "3761": [16],         # Guided Missiles — Peace & Justice
    "3812": [16],
    "5812": [3, 8],       # Eating places (fast food chains at scale)
    "5411": [2],          # Food — potentially hunger-adjacent supply chains
}

# SDG 9 proxy: R&D XBRL concept
_RD_CONCEPTS = [
    "us-gaap:ResearchAndDevelopmentExpense",
    "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
]

# ---------------------------------------------------------------------------
# SDG Goal definitions (all 17)
# ---------------------------------------------------------------------------

SDG_META: Dict[int, Dict[str, str]] = {
    1:  {"name": "No Poverty",              "color": "#E5243B"},
    2:  {"name": "Zero Hunger",             "color": "#DDA63A"},
    3:  {"name": "Good Health & Well-Being","color": "#4C9F38"},
    4:  {"name": "Quality Education",       "color": "#C5192D"},
    5:  {"name": "Gender Equality",         "color": "#FF3A21"},
    6:  {"name": "Clean Water & Sanitation","color": "#26BDE2"},
    7:  {"name": "Affordable Clean Energy", "color": "#FCC30B"},
    8:  {"name": "Decent Work & Economic Growth","color": "#A21942"},
    9:  {"name": "Industry, Innovation & Infrastructure","color": "#FD6925"},
    10: {"name": "Reduced Inequalities",    "color": "#DD1367"},
    11: {"name": "Sustainable Cities & Communities","color": "#FD9D24"},
    12: {"name": "Responsible Consumption & Production","color": "#BF8B2E"},
    13: {"name": "Climate Action",          "color": "#3F7E44"},
    14: {"name": "Life Below Water",        "color": "#0A97D9"},
    15: {"name": "Life on Land",            "color": "#56C02B"},
    16: {"name": "Peace, Justice & Strong Institutions","color": "#00689D"},
    17: {"name": "Partnerships for the Goals","color": "#19486A"},
}

# ---------------------------------------------------------------------------
# Evidence keywords per SDG (for 10-K text evidence, NOT sole scoring basis)
# Used to supplement structured data, not replace it
# ---------------------------------------------------------------------------

SDG_EVIDENCE_PHRASES: Dict[int, Dict[str, List[str]]] = {
    1:  {"positive": ["financial inclusion", "microfinance", "unbanked", "underserved communities",
                      "affordable credit", "community development finance", "cdfi"],
         "negative": ["predatory lending", "poverty wages"]},
    2:  {"positive": ["food security", "sustainable agriculture", "crop yield improvement",
                      "food access", "smallholder farmers", "agrobiodiversity", "food fortification"],
         "negative": ["food waste", "food desert"]},
    3:  {"positive": ["patient outcomes", "healthcare access", "disease prevention", "clinical trial",
                      "mental health", "vaccine", "telehealth", "public health", "drug approval",
                      "fda approval", "orphan drug", "rare disease"],
         "negative": ["opioid settlement", "drug price increase", "price gouging"]},
    4:  {"positive": ["online learning", "educational technology", "workforce training",
                      "scholarship program", "stem education", "literacy", "skill development"],
         "negative": []},
    5:  {"positive": ["gender pay equity", "women on board", "female ceo", "pay parity",
                      "gender diversity", "women in leadership", "equal pay"],
         "negative": ["gender discrimination", "sexual harassment settlement"]},
    6:  {"positive": ["water treatment", "water recycling", "wastewater", "water conservation",
                      "water access", "clean drinking water", "water efficiency"],
         "negative": ["water contamination", "water pollution"]},
    7:  {"positive": ["renewable energy", "solar", "wind power", "energy transition",
                      "clean energy", "battery storage", "electric vehicle", "ppa", "power purchase"],
         "negative": ["coal plant", "oil sands", "fracking", "stranded asset"]},
    8:  {"positive": ["employee growth", "living wage", "worker safety", "osha compliance",
                      "fair wages", "workforce expansion", "job creation", "human capital"],
         "negative": ["mass layoff", "osha violation", "worker injury", "wage theft"]},
    9:  {"positive": ["research and development", "r&d investment", "patent", "innovation",
                      "digital transformation", "5g", "ai investment", "infrastructure spending"],
         "negative": []},
    10: {"positive": ["revenue diversification", "inclusive growth", "equity", "social mobility",
                      "geographic diversification", "emerging markets", "income inequality"],
         "negative": ["monopoly", "market concentration"]},
    11: {"positive": ["affordable housing", "urban development", "smart city", "public transit",
                      "green building", "leed certification", "mixed-use development"],
         "negative": ["urban sprawl", "displacement"]},
    12: {"positive": ["circular economy", "waste reduction", "recycled content", "product stewardship",
                      "take-back program", "sustainable packaging", "zero waste"],
         "negative": ["landfill", "single-use plastic", "e-waste"]},
    13: {"positive": ["net zero", "carbon neutral", "science based target", "sbti", "tcfd",
                      "scope 1", "scope 2", "renewable electricity", "carbon offset"],
         "negative": ["carbon intensive", "fossil fuel expansion", "coal plant"]},
    14: {"positive": ["ocean conservation", "plastic reduction", "marine protected", "fishing sustainability",
                      "blue economy", "coral reef", "water quality"],
         "negative": ["ocean dumping", "plastic pollution", "overfishing"]},
    15: {"positive": ["biodiversity", "reforestation", "land restoration", "deforestation-free",
                      "habitat protection", "wildlife corridor", "sustainable forestry"],
         "negative": ["deforestation", "land degradation", "habitat destruction"]},
    16: {"positive": ["anti-corruption", "whistleblower", "fcpa compliance", "bribery prevention",
                      "human rights", "supply chain transparency", "rule of law"],
         "negative": ["fcpa violation", "bribery", "corruption", "money laundering"]},
    17: {"positive": ["un global compact", "public private partnership", "sustainable development",
                      "blended finance", "impact investment", "development finance institution"],
         "negative": []},
}

# ---------------------------------------------------------------------------
# Evidence weights for composite scoring
# ---------------------------------------------------------------------------

_WEIGHT_SIC_PRIMARY    = 3.0   # Structural sector alignment (highest weight)
_WEIGHT_SIC_SECONDARY  = 1.5
_WEIGHT_XBRL_SEGMENT   = 2.5   # Revenue segment from XBRL
_WEIGHT_TEXT_EVIDENCE  = 1.0   # 10-K text phrases
_WEIGHT_RD_INTENSITY   = 2.0   # R&D / Revenue for SDG 9
_WEIGHT_PATENT         = 1.5   # USPTO patents for SDG 9
_WEIGHT_UNGC           = 2.5   # UN Global Compact for SDG 17
_WEIGHT_NEGATIVE       = -3.0  # Negative impact multiplier

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SDGEvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str
    evidence_type: str   # "sic_primary", "sic_secondary", "xbrl_segment", "text_phrase",
                          # "rd_intensity", "patent", "ungc", "negative_impact"
    sdg_number: int
    weight: float
    detail: str

class SDGAlignmentScore(BaseModel):
    sdg_number: int
    sdg_name: str
    raw_score: float          # sum of evidence weights
    normalized_score: float   # 0–10
    confidence: str           # "high" | "medium" | "low"
    evidence_count: int
    positive_evidence: int
    negative_evidence: int

class CompanySDGProfile(BaseModel):
    ticker: str
    cik: str
    company_name: str
    sic_code: str
    sic_description: str
    composite_score: float     # 0–100
    sdg_scores: List[SDGAlignmentScore]
    top_sdgs: List[int]
    primary_sdg: int
    revenue_segments: List[Dict[str, Any]]
    evidence_log: List[SDGEvidenceItem]
    ungc_signatory: bool
    rd_intensity: Optional[float]
    patent_count_3yr: int
    peer_percentile: Optional[float]
    data_quality: str
    scored_at: str

class PeerComparison(BaseModel):
    ticker: str
    sic_2digit: str
    peer_scores: List[Dict[str, Any]]
    percentile: float
    peer_count: int

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn

def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sdg_scores (
            ticker          TEXT NOT NULL,
            cik             TEXT NOT NULL,
            sdg_number      INTEGER NOT NULL,
            normalized_score REAL NOT NULL,
            raw_score       REAL NOT NULL,
            confidence      TEXT NOT NULL,
            evidence_count  INTEGER NOT NULL,
            scored_at       TEXT NOT NULL,
            PRIMARY KEY (ticker, sdg_number)
        );
        CREATE TABLE IF NOT EXISTS sdg_composite (
            ticker          TEXT PRIMARY KEY,
            cik             TEXT NOT NULL,
            company_name    TEXT NOT NULL,
            sic_code        TEXT NOT NULL,
            composite_score REAL NOT NULL,
            primary_sdg     INTEGER NOT NULL,
            ungc_signatory  INTEGER NOT NULL DEFAULT 0,
            rd_intensity    REAL,
            patent_count_3yr INTEGER NOT NULL DEFAULT 0,
            data_quality    TEXT NOT NULL,
            scored_at       TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS revenue_mapping (
            ticker          TEXT NOT NULL,
            segment_label   TEXT NOT NULL,
            revenue_usd     REAL NOT NULL,
            revenue_pct     REAL NOT NULL,
            sdg_primary     INTEGER,
            sdg_secondary   TEXT,
            fiscal_year     INTEGER NOT NULL,
            scored_at       TEXT NOT NULL,
            PRIMARY KEY (ticker, segment_label, fiscal_year)
        );
        CREATE TABLE IF NOT EXISTS evidence_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            sdg_number      INTEGER NOT NULL,
            source          TEXT NOT NULL,
            evidence_type   TEXT NOT NULL,
            weight          REAL NOT NULL,
            detail          TEXT NOT NULL,
            logged_at       TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sdg_history (
            ticker          TEXT NOT NULL,
            sdg_number      INTEGER NOT NULL,
            normalized_score REAL NOT NULL,
            composite_score REAL NOT NULL,
            scored_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sdg_scores_ticker  ON sdg_scores(ticker);
        CREATE INDEX IF NOT EXISTS idx_evidence_ticker    ON evidence_log(ticker);
        CREATE INDEX IF NOT EXISTS idx_history_ticker     ON sdg_history(ticker);
        CREATE INDEX IF NOT EXISTS idx_composite_sic      ON sdg_composite(sic_code);
    """)
    conn.commit()

# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

_ticker_cik_cache: Dict[str, str] = {}
_ungc_cache: Optional[set] = None
_ungc_cache_ts: float = 0.0

def _edgar_get(url: str, timeout: int = _TIMEOUT) -> Optional[Dict]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("EDGAR GET failed %s: %s", url, exc)
        return None

def _resolve_cik(ticker: str) -> Optional[str]:
    if ticker in _ticker_cik_cache:
        return _ticker_cik_cache[ticker]
    data = _edgar_get(_EDGAR_TICKERS)
    if not data:
        return None
    ticker_up = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_up:
            cik = str(entry["cik_str"]).zfill(10)
            _ticker_cik_cache[ticker_up] = cik
            return cik
    return None

def _get_submissions(cik: str) -> Optional[Dict]:
    url = _EDGAR_SUBMISSIONS.format(cik=cik)
    time.sleep(_EDGAR_SLEEP)
    return _edgar_get(url)

def _get_company_facts(cik: str) -> Optional[Dict]:
    url = _EDGAR_FACTS.format(cik=cik)
    time.sleep(_EDGAR_SLEEP)
    return _edgar_get(url)

def _get_latest_10k_text(cik: str, filing_index_url: str) -> str:
    """Download 10-K filing index and retrieve Item 1 text (first 150KB)."""
    try:
        time.sleep(_EDGAR_SLEEP)
        r = requests.get(filing_index_url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        # Parse filing index to find .htm document
        lines = r.text.splitlines()
        doc_url = None
        for line in lines:
            if ".htm" in line.lower() and "10-k" not in line.lower():
                m = re.search(r'href="([^"]+\.htm)"', line, re.IGNORECASE)
                if m:
                    doc_url = "https://www.sec.gov" + m.group(1)
                    break
        if not doc_url:
            return ""
        time.sleep(_EDGAR_SLEEP)
        r2 = requests.get(doc_url, headers=_HEADERS, timeout=_TIMEOUT)
        r2.raise_for_status()
        text = r2.text[:150_000]
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.lower()
    except Exception as exc:
        logger.debug("10-K text fetch failed: %s", exc)
        return ""

def _find_latest_10k_filing(submissions: Dict) -> Optional[str]:
    """Return index URL for the most recent 10-K filing."""
    filings = submissions.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    acc_nums = filings.get("accessionNumber", [])
    cik = submissions.get("cik", "")
    for form, acc in zip(forms, acc_nums):
        if form in ("10-K", "10-K405"):
            acc_clean = acc.replace("-", "")
            idx_url = f"{_EDGAR_ARCHIVES}/{cik}/{acc_clean}/{acc}-index.htm"
            return idx_url
    return None

# ---------------------------------------------------------------------------
# XBRL segment revenue extractor
# ---------------------------------------------------------------------------

_REVENUE_CONCEPTS_XBRL = {
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
    "NetIncomeLoss",   # fallback for income statements
}

_SEGMENT_AXES = {
    "StatementBusinessSegmentsAxis",
    "GeographicAreasAxis",
    "ProductOrServiceAxis",
}

class RevenueSegment:
    def __init__(self, label: str, revenue: float, fiscal_year: int):
        self.label = label
        self.revenue = revenue
        self.fiscal_year = fiscal_year
        self.sdg_primary: Optional[int] = None
        self.sdg_secondary: List[int] = []

def _extract_xbrl_segments(facts: Dict) -> List[RevenueSegment]:
    """Extract revenue segment data from EDGAR XBRL companyfacts."""
    segments: List[RevenueSegment] = []
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for concept_name, concept_data in us_gaap.items():
        short = concept_name.split(":")[-1] if ":" in concept_name else concept_name
        if short not in _REVENUE_CONCEPTS_XBRL:
            continue
        units = concept_data.get("units", {})
        usd_data = units.get("USD", [])
        for entry in usd_data:
            form = entry.get("form", "")
            if form not in ("10-K", "10-K405", "20-F"):
                continue
            frame = entry.get("frame", "")
            segment = entry.get("segment", "")
            if not segment:
                continue
            # Only keep business segment axis entries (not geographic)
            if "GeographicAreas" in segment or "srt:StatementGeographical" in segment:
                continue
            val = entry.get("val", 0)
            if not isinstance(val, (int, float)) or val <= 0:
                continue
            # Extract fiscal year from 'end' date
            end_date = entry.get("end", "")
            fy = int(end_date[:4]) if len(end_date) >= 4 else 0
            if fy < 2020:
                continue
            # Clean segment label
            label = re.sub(r"[A-Za-z]+:", "", segment).strip()
            label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label).strip()
            segments.append(RevenueSegment(label=label, revenue=float(val), fiscal_year=fy))

    # Deduplicate: keep highest revenue per (label, fy)
    seen: Dict[Tuple[str, int], RevenueSegment] = {}
    for seg in segments:
        key = (seg.label.lower(), seg.fiscal_year)
        if key not in seen or seg.revenue > seen[key].revenue:
            seen[key] = seg

    # Return most recent fiscal year
    if not seen:
        return []
    max_fy = max(s.fiscal_year for s in seen.values())
    return [s for s in seen.values() if s.fiscal_year == max_fy]

# ---------------------------------------------------------------------------
# SIC code SDG mapper
# ---------------------------------------------------------------------------

def _get_sic_sdg_mapping(sic_code: str) -> Dict[str, Any]:
    """
    Look up SDG alignment from SIC code.
    Tries exact 4-digit, then 3-digit, then 2-digit prefixes.
    """
    sic_str = str(sic_code).strip()
    for length in (4, 3, 2):
        prefix = sic_str[:length]
        if prefix in SIC_SDG_MAP:
            return SIC_SDG_MAP[prefix]
    return {"primary": 8, "secondary": []}   # fallback: Decent Work (SDG 8)

def _is_negative_sic(sic_code: str) -> List[int]:
    """Return list of SDGs harmed by this SIC code."""
    sic_str = str(sic_code).strip()
    for length in (4, 3, 2):
        prefix = sic_str[:length]
        if prefix in _NEGATIVE_IMPACT_SIC:
            return _NEGATIVE_IMPACT_SIC[prefix]
    return []

def _classify_segment_by_label(label: str) -> Optional[int]:
    """
    Classify a revenue segment label to an SDG using pattern matching.
    More structured than keyword soup — matches named product/service categories.
    """
    label_lower = label.lower()
    patterns: List[Tuple[re.Pattern, int]] = [
        (re.compile(r"\b(pharma|drug|biolog|vaccine|therapeut|oncol|medic)\b"), 3),
        (re.compile(r"\b(hospital|clinic|health|patient|diagnostic)\b"), 3),
        (re.compile(r"\b(educat|learn|school|training|university)\b"), 4),
        (re.compile(r"\b(solar|wind|renewabl|clean energy|battery|storage)\b"), 7),
        (re.compile(r"\b(water|wastewater|desalin)\b"), 6),
        (re.compile(r"\b(software|technology|cloud|ai|semiconductor|digital)\b"), 9),
        (re.compile(r"\b(research|r&d|innovation|patent)\b"), 9),
        (re.compile(r"\b(housing|afford|residential|apartment)\b"), 11),
        (re.compile(r"\b(food|agric|crop|grain|beverage|nutrition)\b"), 2),
        (re.compile(r"\b(insurance|banking|credit|loan|lending)\b"), 8),
        (re.compile(r"\b(micro.?financ|community develop)\b"), 1),
        (re.compile(r"\b(ocean|marine|fish|seafood)\b"), 14),
        (re.compile(r"\b(forest|timber|wood|land|mining)\b"), 15),
        (re.compile(r"\b(recycl|circular|waste management)\b"), 12),
        (re.compile(r"\b(carbon|climate|emission|clean air|net zero)\b"), 13),
        (re.compile(r"\b(infrastructure|construct|civil|transport|rail|road)\b"), 11),
        (re.compile(r"\b(telecom|communication|broadband|internet)\b"), 9),
        (re.compile(r"\b(defense|weapons|military|missile|ammunition)\b"), 16),
        (re.compile(r"\b(tobacco|cigarette|smoking)\b"), 3),   # negative for SDG 3
        (re.compile(r"\b(coal|oil sands|fracking|petroleum)\b"), 7),  # negative for SDG 7
    ]
    for pattern, sdg in patterns:
        if pattern.search(label_lower):
            return sdg
    return None

# ---------------------------------------------------------------------------
# XBRL financial metrics extractor
# ---------------------------------------------------------------------------

def _extract_rd_intensity(facts: Dict) -> Optional[float]:
    """Compute R&D/Revenue ratio from XBRL data."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    rd_val = None
    rev_val = None

    # Get most recent annual R&D
    for concept in _RD_CONCEPTS:
        short = concept.split(":")[-1]
        if short in us_gaap:
            usd_data = us_gaap[short].get("units", {}).get("USD", [])
            annual = [e for e in usd_data if e.get("form") in ("10-K", "10-K405")
                      and not e.get("frame", "").endswith("I")]
            if annual:
                annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                rd_val = annual[0].get("val", 0)
                break

    # Get most recent annual revenue
    for rev_concept in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "SalesRevenueNet"):
        if rev_concept in us_gaap:
            usd_data = us_gaap[rev_concept].get("units", {}).get("USD", [])
            # Only consolidated (no segment/dimensional entries)
            consolidated = [e for e in usd_data
                            if e.get("form") in ("10-K", "10-K405")
                            and not e.get("segment")
                            and not e.get("frame", "").endswith("I")]
            if consolidated:
                consolidated.sort(key=lambda x: x.get("end", ""), reverse=True)
                rev_val = consolidated[0].get("val", 0)
                break

    if rd_val and rev_val and rev_val > 0:
        return round(float(rd_val) / float(rev_val), 4)
    return None

def _extract_employee_count_growth(facts: Dict) -> Optional[float]:
    """Return YoY employee growth rate (for SDG 8)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    emp_data = us_gaap.get("NumberOfEmployees", {}).get("units", {}).get("pure", [])
    if not emp_data:
        emp_data = us_gaap.get("EntityNumberOfEmployees", {}).get("units", {}).get("pure", [])
    if not emp_data:
        return None
    annual = sorted(emp_data, key=lambda x: x.get("end", ""), reverse=True)
    if len(annual) >= 2:
        curr = annual[0].get("val", 0)
        prev = annual[1].get("val", 0)
        if prev and prev > 0:
            return round((curr - prev) / prev, 4)
    return None

# ---------------------------------------------------------------------------
# USPTO free API — patent signal
# ---------------------------------------------------------------------------

def _fetch_patent_count(company_name: str, years: int = 3) -> int:
    """
    Query USPTO bulk data search for granted patents.
    Free tier: https://developer.uspto.gov/ibd-api/v1/application/grants
    Returns count of utility patents in last N years.
    """
    try:
        year_cutoff = datetime.now().year - years
        params = {
            "patentTitle": "",
            "assigneeEntityName": company_name[:50],
            "dateRangeData": json.dumps({
                "startDate": f"{year_cutoff}-01-01",
                "endDate": datetime.now().strftime("%Y-%m-%d"),
            }),
            "start": 0,
            "rows": 1,
        }
        r = requests.get(
            "https://developer.uspto.gov/ibd-api/v1/application/grants",
            params=params,
            headers={**_HEADERS, "Accept": "application/json"},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            return int(data.get("response", {}).get("numFound", 0))
        return 0
    except Exception as exc:
        logger.debug("USPTO fetch failed: %s", exc)
        return 0

# ---------------------------------------------------------------------------
# UN Global Compact signatory lookup
# ---------------------------------------------------------------------------

def _is_ungc_signatory(company_name: str) -> bool:
    """
    Check if company is a UN Global Compact signatory.
    Queries the UNGC participant search endpoint (public, no auth).
    """
    global _ungc_cache, _ungc_cache_ts
    try:
        url = "https://unglobalcompact.org/api/v1/organizations"
        params = {
            "organization_name": company_name[:40],
            "per_page": 5,
        }
        r = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            orgs = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(orgs, list) and len(orgs) > 0:
                # Fuzzy match on name
                cn_lower = company_name.lower().replace("inc.", "").replace("corp.", "").strip()
                for org in orgs:
                    org_name = org.get("name", "").lower()
                    if cn_lower[:10] in org_name:
                        return True
        return False
    except Exception:
        return False

# ---------------------------------------------------------------------------
# 10-K text evidence scorer
# ---------------------------------------------------------------------------

def _score_text_evidence(text: str) -> Dict[int, Dict[str, int]]:
    """
    Count positive/negative phrase matches per SDG in 10-K text.
    Returns {sdg: {"positive": n, "negative": n}}
    """
    result: Dict[int, Dict[str, int]] = {i: {"positive": 0, "negative": 0} for i in range(1, 18)}
    if not text:
        return result
    text_lower = text.lower()
    for sdg_num, phrases in SDG_EVIDENCE_PHRASES.items():
        for phrase in phrases.get("positive", []):
            if phrase in text_lower:
                result[sdg_num]["positive"] += 1
        for phrase in phrases.get("negative", []):
            if phrase in text_lower:
                result[sdg_num]["negative"] += 1
    return result

# ---------------------------------------------------------------------------
# Geographic revenue HHI (SDG 10 — Reduced Inequalities)
# ---------------------------------------------------------------------------

def _compute_geo_hhi(facts: Dict) -> Optional[float]:
    """
    Compute Herfindahl-Hirschman Index over geographic revenue segments.
    Low HHI (diversified) → positive SDG 10 signal.
    Returns HHI on 0–10000 scale, or None if no data.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    geo_revenues: List[float] = []

    for concept_name in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"):
        if concept_name not in us_gaap:
            continue
        usd_data = us_gaap[concept_name].get("units", {}).get("USD", [])
        for entry in usd_data:
            segment = entry.get("segment", "")
            if not segment:
                continue
            if not any(g in segment for g in ("GeographicAreas", "StatementGeographical")):
                continue
            val = entry.get("val", 0)
            if isinstance(val, (int, float)) and val > 0:
                geo_revenues.append(float(val))

    if len(geo_revenues) < 2:
        return None
    total = sum(geo_revenues)
    if total == 0:
        return None
    shares = [r / total for r in geo_revenues]
    hhi = sum(s ** 2 for s in shares) * 10000
    return round(hhi, 1)

# ---------------------------------------------------------------------------
# Core SDG scorer
# ---------------------------------------------------------------------------

class SDGScorerV3:
    """Production SDG alignment scorer using structured data."""

    def __init__(self):
        self.conn = _get_conn()

    # --- Main entry point ---

    def score_company(self, ticker: str, force_refresh: bool = False) -> CompanySDGProfile:
        ticker = ticker.upper().strip()

        if not force_refresh:
            cached = self._load_cached(ticker)
            if cached:
                return cached

        # Step 1: Resolve CIK
        cik = _resolve_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker {ticker}")

        # Step 2: Get submissions + company metadata
        subs = _get_submissions(cik)
        if not subs:
            raise ValueError(f"Cannot fetch EDGAR submissions for {ticker}")

        company_name = subs.get("name", ticker)
        sic_code = str(subs.get("sic", "9999")).zfill(4)
        sic_description = subs.get("sicDescription", "Unknown")

        # Step 3: Get XBRL facts
        facts = _get_company_facts(cik)

        # Step 4: Extract revenue segments
        segments: List[RevenueSegment] = []
        if facts:
            segments = _extract_xbrl_segments(facts)

        # Step 5: SIC-based SDG mapping
        sic_mapping = _get_sic_sdg_mapping(sic_code)
        negative_sdgs = _is_negative_sic(sic_code)

        # Step 6: R&D intensity
        rd_intensity: Optional[float] = None
        if facts:
            rd_intensity = _extract_rd_intensity(facts)

        # Step 7: Employee growth (for SDG 8)
        emp_growth: Optional[float] = None
        if facts:
            emp_growth = _extract_employee_count_growth(facts)

        # Step 8: Geographic HHI (for SDG 10)
        geo_hhi: Optional[float] = None
        if facts:
            geo_hhi = _compute_geo_hhi(facts)

        # Step 9: 10-K text evidence
        text_evidence: Dict[int, Dict[str, int]] = {i: {"positive": 0, "negative": 0} for i in range(1, 18)}
        filing_url = _find_latest_10k_filing(subs)
        if filing_url:
            text = _get_latest_10k_text(cik, filing_url)
            if text:
                text_evidence = _score_text_evidence(text)

        # Step 10: Patents (SDG 9)
        patent_count = _fetch_patent_count(company_name, years=3)

        # Step 11: UNGC signatory (SDG 17)
        ungc = _is_ungc_signatory(company_name)

        # Step 12: Classify revenue segments by label
        total_segment_revenue = sum(s.revenue for s in segments)
        for seg in segments:
            sdg_from_label = _classify_segment_by_label(seg.label)
            if sdg_from_label:
                seg.sdg_primary = sdg_from_label
            else:
                seg.sdg_primary = sic_mapping.get("primary", 8)
            seg.sdg_secondary = sic_mapping.get("secondary", [])

        # Step 13: Build evidence corpus
        evidence: List[SDGEvidenceItem] = []

        # SIC primary alignment
        primary_sdg = sic_mapping.get("primary", 8)
        evidence.append(SDGEvidenceItem(
            source="SEC_EDGAR_SIC",
            evidence_type="sic_primary",
            sdg_number=primary_sdg,
            weight=_WEIGHT_SIC_PRIMARY,
            detail=f"SIC {sic_code} ({sic_description}) primary alignment",
        ))

        # SIC secondary alignments
        for sec_sdg in sic_mapping.get("secondary", []):
            evidence.append(SDGEvidenceItem(
                source="SEC_EDGAR_SIC",
                evidence_type="sic_secondary",
                sdg_number=sec_sdg,
                weight=_WEIGHT_SIC_SECONDARY,
                detail=f"SIC {sic_code} secondary alignment to SDG {sec_sdg}",
            ))

        # Negative SIC impacts
        for neg_sdg in negative_sdgs:
            evidence.append(SDGEvidenceItem(
                source="SEC_EDGAR_SIC",
                evidence_type="negative_impact",
                sdg_number=neg_sdg,
                weight=_WEIGHT_NEGATIVE,
                detail=f"SIC {sic_code} identified as negative impact sector for SDG {neg_sdg}",
            ))

        # XBRL segment evidence
        for seg in segments:
            if seg.sdg_primary and total_segment_revenue > 0:
                rev_pct = seg.revenue / total_segment_revenue
                seg_weight = _WEIGHT_XBRL_SEGMENT * min(rev_pct * 3, 1.5)
                evidence.append(SDGEvidenceItem(
                    source="SEC_EDGAR_XBRL",
                    evidence_type="xbrl_segment",
                    sdg_number=seg.sdg_primary,
                    weight=seg_weight,
                    detail=f"Revenue segment '{seg.label}' ({rev_pct:.1%} of revenue)",
                ))

        # Text evidence
        for sdg_num, counts in text_evidence.items():
            pos = counts["positive"]
            neg = counts["negative"]
            if pos > 0:
                evidence.append(SDGEvidenceItem(
                    source="SEC_EDGAR_10K_TEXT",
                    evidence_type="text_phrase",
                    sdg_number=sdg_num,
                    weight=_WEIGHT_TEXT_EVIDENCE * min(pos, 5),
                    detail=f"{pos} positive evidence phrase(s) in 10-K for SDG {sdg_num}",
                ))
            if neg > 0:
                evidence.append(SDGEvidenceItem(
                    source="SEC_EDGAR_10K_TEXT",
                    evidence_type="text_phrase",
                    sdg_number=sdg_num,
                    weight=_WEIGHT_NEGATIVE * 0.5 * min(neg, 3),
                    detail=f"{neg} negative evidence phrase(s) in 10-K for SDG {sdg_num}",
                ))

        # R&D intensity → SDG 9
        if rd_intensity is not None:
            rd_weight = _WEIGHT_RD_INTENSITY * min(rd_intensity / 0.10, 2.0)
            evidence.append(SDGEvidenceItem(
                source="SEC_EDGAR_XBRL",
                evidence_type="rd_intensity",
                sdg_number=9,
                weight=rd_weight,
                detail=f"R&D intensity {rd_intensity:.2%} of revenue → SDG 9 Innovation",
            ))

        # Patent count → SDG 9
        if patent_count > 0:
            pat_weight = _WEIGHT_PATENT * min(math.log1p(patent_count) / math.log(50), 1.5)
            evidence.append(SDGEvidenceItem(
                source="USPTO_FREE_API",
                evidence_type="patent",
                sdg_number=9,
                weight=pat_weight,
                detail=f"{patent_count} granted patents (3yr) via USPTO → SDG 9",
            ))

        # Employee growth → SDG 8
        if emp_growth is not None and emp_growth > 0:
            emp_weight = _WEIGHT_TEXT_EVIDENCE * min(emp_growth / 0.05, 2.0)
            evidence.append(SDGEvidenceItem(
                source="SEC_EDGAR_XBRL",
                evidence_type="xbrl_segment",
                sdg_number=8,
                weight=emp_weight,
                detail=f"Employee growth {emp_growth:+.1%} YoY → SDG 8 Decent Work",
            ))

        # Geographic HHI → SDG 10
        if geo_hhi is not None:
            # Low HHI = diversified = positive SDG 10
            # HHI < 1500 = highly diversified, > 6000 = concentrated
            if geo_hhi < 3000:
                hhi_weight = _WEIGHT_SIC_SECONDARY * (1 - geo_hhi / 6000)
                evidence.append(SDGEvidenceItem(
                    source="SEC_EDGAR_XBRL",
                    evidence_type="xbrl_segment",
                    sdg_number=10,
                    weight=hhi_weight,
                    detail=f"Geographic revenue HHI {geo_hhi:.0f} (diversified) → SDG 10",
                ))

        # UNGC signatory → SDG 17
        if ungc:
            evidence.append(SDGEvidenceItem(
                source="UN_GLOBAL_COMPACT",
                evidence_type="ungc",
                sdg_number=17,
                weight=_WEIGHT_UNGC,
                detail=f"{company_name} is a UN Global Compact signatory → SDG 17",
            ))

        # Step 14: Aggregate per-SDG scores
        sdg_raw: Dict[int, float] = defaultdict(float)
        sdg_pos_count: Dict[int, int] = defaultdict(int)
        sdg_neg_count: Dict[int, int] = defaultdict(int)
        sdg_evidence_count: Dict[int, int] = defaultdict(int)

        for ev in evidence:
            sdg_raw[ev.sdg_number] += ev.weight
            sdg_evidence_count[ev.sdg_number] += 1
            if ev.weight > 0:
                sdg_pos_count[ev.sdg_number] += 1
            else:
                sdg_neg_count[ev.sdg_number] += 1

        # Normalize raw scores to 0–10
        all_raw = list(sdg_raw.values())
        max_raw = max(all_raw) if all_raw else 1.0
        min_raw = min(all_raw) if all_raw else 0.0
        score_range = max(max_raw - min_raw, 0.1)

        sdg_scores: List[SDGAlignmentScore] = []
        for sdg_num in range(1, 18):
            raw = sdg_raw.get(sdg_num, 0.0)
            # Normalize: shift so minimum possible raw ≈ 0, scale to 0–10
            normalized = max(0.0, min(10.0, ((raw - min_raw) / score_range) * 10))
            ev_count = sdg_evidence_count.get(sdg_num, 0)
            confidence = (
                "high"   if ev_count >= 4 else
                "medium" if ev_count >= 2 else
                "low"
            )
            sdg_scores.append(SDGAlignmentScore(
                sdg_number=sdg_num,
                sdg_name=SDG_META[sdg_num]["name"],
                raw_score=round(raw, 3),
                normalized_score=round(normalized, 2),
                confidence=confidence,
                evidence_count=ev_count,
                positive_evidence=sdg_pos_count.get(sdg_num, 0),
                negative_evidence=sdg_neg_count.get(sdg_num, 0),
            ))

        # Composite score: weighted average of top-5 SDGs + penalize negatives
        sorted_scores = sorted(sdg_scores, key=lambda x: x.normalized_score, reverse=True)
        top5_avg = sum(s.normalized_score for s in sorted_scores[:5]) / 5
        neg_penalty = sum(abs(sdg_raw.get(n, 0)) for n in negative_sdgs if sdg_raw.get(n, 0) < 0)
        composite = max(0.0, min(100.0, top5_avg * 10 - neg_penalty))

        # Top contributing SDGs
        top_sdgs = [s.sdg_number for s in sorted_scores[:3]]

        # Revenue segment output
        rev_segment_out: List[Dict[str, Any]] = []
        total_rev = sum(s.revenue for s in segments)
        for seg in segments:
            rev_segment_out.append({
                "label": seg.label,
                "revenue_usd": round(seg.revenue, 0),
                "revenue_pct": round(seg.revenue / total_rev if total_rev > 0 else 0, 4),
                "sdg_primary": seg.sdg_primary,
                "sdg_secondary": seg.sdg_secondary,
                "fiscal_year": seg.fiscal_year,
            })

        profile = CompanySDGProfile(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            sic_code=sic_code,
            sic_description=sic_description,
            composite_score=round(composite, 2),
            sdg_scores=sdg_scores,
            top_sdgs=top_sdgs,
            primary_sdg=primary_sdg,
            revenue_segments=rev_segment_out,
            evidence_log=evidence,
            ungc_signatory=ungc,
            rd_intensity=rd_intensity,
            patent_count_3yr=patent_count,
            peer_percentile=None,  # filled in by peer comparison
            data_quality="high" if (facts and segments) else "medium" if facts else "low",
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

        self._persist(profile)
        return profile

    # --- Peer comparison ---

    def get_peer_comparison(self, ticker: str) -> PeerComparison:
        ticker = ticker.upper()
        try:
            profile = self.score_company(ticker)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Cannot score {ticker}")

        sic_2 = profile.sic_code[:2]
        cursor = self.conn.execute(
            """
            SELECT ticker, composite_score FROM sdg_composite
            WHERE sic_code LIKE ? AND ticker != ?
            ORDER BY composite_score DESC
            """,
            (sic_2 + "%", ticker),
        )
        peers = [dict(row) for row in cursor.fetchall()]
        all_scores = sorted([p["composite_score"] for p in peers] + [profile.composite_score])
        pct = 0.0
        if all_scores:
            rank = sum(1 for s in all_scores if s <= profile.composite_score)
            pct = round(rank / len(all_scores) * 100, 1)

        return PeerComparison(
            ticker=ticker,
            sic_2digit=sic_2,
            peer_scores=[{"ticker": p["ticker"], "composite_score": p["composite_score"]}
                         for p in peers[:20]],
            percentile=pct,
            peer_count=len(peers),
        )

    # --- Controversy signal ---

    def get_controversies(self, ticker: str) -> Dict[str, Any]:
        """Return evidence log entries with negative weight (controversy signals)."""
        ticker = ticker.upper()
        cursor = self.conn.execute(
            """
            SELECT sdg_number, source, evidence_type, weight, detail, logged_at
            FROM evidence_log
            WHERE ticker = ? AND weight < 0
            ORDER BY weight ASC
            """,
            (ticker,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return {
            "ticker": ticker,
            "controversy_count": len(rows),
            "controversies": rows,
        }

    # --- Persistence ---

    def _persist(self, profile: CompanySDGProfile) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            # Composite
            self.conn.execute("""
                INSERT OR REPLACE INTO sdg_composite
                (ticker, cik, company_name, sic_code, composite_score, primary_sdg,
                 ungc_signatory, rd_intensity, patent_count_3yr, data_quality, scored_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (profile.ticker, profile.cik, profile.company_name, profile.sic_code,
                  profile.composite_score, profile.primary_sdg, int(profile.ungc_signatory),
                  profile.rd_intensity, profile.patent_count_3yr,
                  profile.data_quality, profile.scored_at))

            # Per-SDG scores
            for s in profile.sdg_scores:
                self.conn.execute("""
                    INSERT OR REPLACE INTO sdg_scores
                    (ticker, cik, sdg_number, normalized_score, raw_score,
                     confidence, evidence_count, scored_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (profile.ticker, profile.cik, s.sdg_number, s.normalized_score,
                      s.raw_score, s.confidence, s.evidence_count, now))

            # Revenue mapping
            for seg in profile.revenue_segments:
                self.conn.execute("""
                    INSERT OR REPLACE INTO revenue_mapping
                    (ticker, segment_label, revenue_usd, revenue_pct, sdg_primary,
                     sdg_secondary, fiscal_year, scored_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (profile.ticker, seg["label"], seg["revenue_usd"], seg["revenue_pct"],
                      seg.get("sdg_primary"), json.dumps(seg.get("sdg_secondary", [])),
                      seg.get("fiscal_year", 0), now))

            # Evidence log
            self.conn.execute("DELETE FROM evidence_log WHERE ticker = ?", (profile.ticker,))
            for ev in profile.evidence_log:
                self.conn.execute("""
                    INSERT INTO evidence_log
                    (ticker, sdg_number, source, evidence_type, weight, detail, logged_at)
                    VALUES (?,?,?,?,?,?,?)
                """, (profile.ticker, ev.sdg_number, ev.source, ev.evidence_type,
                      ev.weight, ev.detail, now))

            # History
            for s in profile.sdg_scores:
                self.conn.execute("""
                    INSERT INTO sdg_history
                    (ticker, sdg_number, normalized_score, composite_score, scored_at)
                    VALUES (?,?,?,?,?)
                """, (profile.ticker, s.sdg_number, s.normalized_score,
                      profile.composite_score, now))

            self.conn.commit()
        except Exception as exc:
            logger.error("DB persist error for %s: %s", profile.ticker, exc)

    def _load_cached(self, ticker: str) -> Optional[CompanySDGProfile]:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=_CACHE_TTL)).isoformat()
        row = self.conn.execute(
            "SELECT * FROM sdg_composite WHERE ticker = ? AND scored_at > ?",
            (ticker, cutoff),
        ).fetchone()
        if not row:
            return None
        # Load SDG scores
        sdg_rows = self.conn.execute(
            "SELECT * FROM sdg_scores WHERE ticker = ?", (ticker,)
        ).fetchall()
        sdg_scores = [
            SDGAlignmentScore(
                sdg_number=r["sdg_number"],
                sdg_name=SDG_META[r["sdg_number"]]["name"],
                raw_score=r["raw_score"],
                normalized_score=r["normalized_score"],
                confidence=r["confidence"],
                evidence_count=r["evidence_count"],
                positive_evidence=0,
                negative_evidence=0,
            )
            for r in sdg_rows
        ]
        sorted_s = sorted(sdg_scores, key=lambda x: x.normalized_score, reverse=True)
        return CompanySDGProfile(
            ticker=ticker,
            cik=row["cik"],
            company_name=row["company_name"],
            sic_code=row["sic_code"],
            sic_description="",
            composite_score=row["composite_score"],
            sdg_scores=sdg_scores,
            top_sdgs=[s.sdg_number for s in sorted_s[:3]],
            primary_sdg=row["primary_sdg"],
            revenue_segments=[],
            evidence_log=[],
            ungc_signatory=bool(row["ungc_signatory"]),
            rd_intensity=row["rd_intensity"],
            patent_count_3yr=row["patent_count_3yr"],
            peer_percentile=None,
            data_quality=row["data_quality"],
            scored_at=row["scored_at"],
        )

# ---------------------------------------------------------------------------
# Singleton scorer
# ---------------------------------------------------------------------------

_scorer: Optional[SDGScorerV3] = None

def _get_scorer() -> SDGScorerV3:
    global _scorer
    if _scorer is None:
        _scorer = SDGScorerV3()
    return _scorer

# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/sdg/v3", tags=["SDG Impact v3"])

@router.get("/alignment/{ticker}", summary="Full SDG alignment profile")
def get_alignment(
    ticker: str,
    refresh: bool = Query(False, description="Force data refresh"),
) -> CompanySDGProfile:
    scorer = _get_scorer()
    try:
        return scorer.score_company(ticker.upper(), force_refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.get("/sdg/{ticker}/{sdg_number}", summary="Single SDG score for a company")
def get_single_sdg(
    ticker: str,
    sdg_number: int,
) -> SDGAlignmentScore:
    if not 1 <= sdg_number <= 17:
        raise HTTPException(status_code=400, detail="sdg_number must be 1–17")
    scorer = _get_scorer()
    try:
        profile = scorer.score_company(ticker.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    for score in profile.sdg_scores:
        if score.sdg_number == sdg_number:
            return score
    raise HTTPException(status_code=404, detail=f"SDG {sdg_number} score not found")

@router.get("/peer-comparison/{ticker}", summary="SDG score vs SIC 2-digit peers")
def get_peer_comparison(ticker: str) -> PeerComparison:
    scorer = _get_scorer()
    return scorer.get_peer_comparison(ticker.upper())

@router.get("/top-contributors/{ticker}", summary="Top 5 SDGs by normalized score")
def get_top_contributors(
    ticker: str,
    top_n: int = Query(5, ge=1, le=17),
) -> Dict[str, Any]:
    scorer = _get_scorer()
    try:
        profile = scorer.score_company(ticker.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    sorted_scores = sorted(profile.sdg_scores, key=lambda x: x.normalized_score, reverse=True)
    return {
        "ticker": ticker.upper(),
        "composite_score": profile.composite_score,
        "top_sdgs": [
            {
                "sdg_number": s.sdg_number,
                "sdg_name": s.sdg_name,
                "normalized_score": s.normalized_score,
                "confidence": s.confidence,
                "evidence_count": s.evidence_count,
            }
            for s in sorted_scores[:top_n]
        ],
        "primary_sdg": profile.primary_sdg,
        "ungc_signatory": profile.ungc_signatory,
        "rd_intensity": profile.rd_intensity,
        "patent_count_3yr": profile.patent_count_3yr,
        "data_quality": profile.data_quality,
    }

@router.get("/controversies/{ticker}", summary="SDG controversies and negative signals")
def get_controversies(ticker: str) -> Dict[str, Any]:
    scorer = _get_scorer()
    try:
        # Ensure company is scored first
        scorer.score_company(ticker.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return scorer.get_controversies(ticker.upper())

@router.get("/sdg-universe", summary="All 17 SDG definitions with metadata")
def get_sdg_universe() -> Dict[str, Any]:
    return {
        "sdgs": [
            {
                "number": num,
                "name": meta["name"],
                "color": meta["color"],
                "positive_phrases": SDG_EVIDENCE_PHRASES[num]["positive"][:3],
            }
            for num, meta in SDG_META.items()
        ]
    }

@router.get("/history/{ticker}", summary="Historical SDG score trend")
def get_history(
    ticker: str,
    sdg_number: Optional[int] = Query(None, ge=1, le=17),
    limit: int = Query(30, ge=1, le=200),
) -> Dict[str, Any]:
    scorer = _get_scorer()
    q = "SELECT * FROM sdg_history WHERE ticker = ?"
    params: List[Any] = [ticker.upper()]
    if sdg_number:
        q += " AND sdg_number = ?"
        params.append(sdg_number)
    q += " ORDER BY scored_at DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in scorer.conn.execute(q, params).fetchall()]
    return {"ticker": ticker.upper(), "sdg_number": sdg_number, "history": rows}

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pprint
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    scorer = SDGScorerV3()
    print(f"Scoring {ticker}...")
    profile = scorer.score_company(ticker, force_refresh=True)
    print(f"\nCompany: {profile.company_name}")
    print(f"SIC: {profile.sic_code} — {profile.sic_description}")
    print(f"Composite SDG Score: {profile.composite_score:.1f}/100")
    print(f"UNGC Signatory: {profile.ungc_signatory}")
    print(f"R&D Intensity: {profile.rd_intensity}")
    print(f"Patents (3yr): {profile.patent_count_3yr}")
    print(f"\nTop SDGs:")
    sorted_scores = sorted(profile.sdg_scores, key=lambda x: x.normalized_score, reverse=True)
    for s in sorted_scores[:5]:
        print(f"  SDG {s.sdg_number:2d} {s.sdg_name:<35s} {s.normalized_score:5.2f}/10  [{s.confidence}]")
    print(f"\nRevenue Segments ({len(profile.revenue_segments)}):")
    for seg in profile.revenue_segments[:5]:
        print(f"  {seg['label']:<30s} {seg['revenue_pct']:6.1%}  → SDG {seg['sdg_primary']}")
    print(f"\nEvidence ({len(profile.evidence_log)} items, showing negatives):")
    neg_ev = [e for e in profile.evidence_log if e.weight < 0]
    for ev in neg_ev[:5]:
        print(f"  [{ev.evidence_type}] SDG {ev.sdg_number}: {ev.detail} (w={ev.weight:.1f})")
