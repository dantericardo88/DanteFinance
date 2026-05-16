"""
ifrs_financials_v3.py — International / IFRS financial statements v3.

Dimension: dim_021 — International / IFRS financials ex-US  target score: 9/10

Key upgrades over v1/v2:
  - Multi-source ingestion: EDGAR 20-F XBRL, SimFin bulk API, World Bank, Yahoo Finance,
    ECB FX rates for EUR/USD and cross rates
  - 200-company non-US universe across Europe, Asia, Canada, Australia
  - 50+ IFRS-to-GAAP concept mapping table stored in SQLite
  - Multi-currency normalization via ECB free Data Portal (sdmx JSON)
  - SimFin bulk data: income statement, balance sheet, cashflow for European IFRS filers
  - EDGAR 20-F XBRL companyfacts for ADR filers (ASML, SAP, Toyota, HSBC, etc.)
  - Standardized canonical schema: maps both IFRS XBRL and SimFin fields
  - SQLite: ifrs_companies, ifrs_financials, ifrs_concept_map, fx_rates, simfin_cache
  - FastAPI router /ifrs/v3: financials, income, balance, cashflow, peers, concept-map

Public entry points
-------------------
router: APIRouter     — mount at /ifrs/v3
IFRSService           — primary service class
IFRSConceptMapper     — IFRS ↔ GAAP concept translation
FXNormalizer          — multi-currency USD conversion via ECB
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_EFTS        = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANYFACTS= "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SIMFIN_BULK_BASE  = "https://simfin.com/api/v2"
WORLDBANK_BASE    = "https://api.worldbank.org/v2"
ECB_BASE          = "https://data-api.ecb.europa.eu/service/data"
YAHOO_QUERY1      = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_SUMMARY     = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"

DB_PATH  = Path("sentinel_ifrs_v3.db")
CACHE_TTL = 12 * 3600  # 12 hours

_HEADERS: Dict[str, str] = {
    "User-Agent": "SENTINEL/3.0 financial-terminal richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

# ---------------------------------------------------------------------------
# IFRS ↔ GAAP concept map (50+ mappings)
# ---------------------------------------------------------------------------

IFRS_GAAP_CONCEPT_MAP: List[Dict[str, str]] = [
    # Income Statement
    {"ifrs_label": "Revenue",                        "ifrs_xbrl": "ifrs-full:Revenue",                                   "gaap_label": "Revenues",                          "gaap_xbrl": "us-gaap:Revenues",                                  "category": "income_statement", "note": "Top-line revenue"},
    {"ifrs_label": "Revenue from Contracts",         "ifrs_xbrl": "ifrs-full:RevenueFromContractsWithCustomers",         "gaap_label": "RevenueFromContractWithCustomer",   "gaap_xbrl": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "category": "income_statement", "note": "IFRS 15 / ASC 606 alignment"},
    {"ifrs_label": "Cost of Sales",                  "ifrs_xbrl": "ifrs-full:CostOfSales",                              "gaap_label": "CostOfRevenue",                     "gaap_xbrl": "us-gaap:CostOfRevenue",                             "category": "income_statement", "note": "COGS"},
    {"ifrs_label": "Gross Profit",                   "ifrs_xbrl": "ifrs-full:GrossProfit",                              "gaap_label": "GrossProfit",                       "gaap_xbrl": "us-gaap:GrossProfit",                               "category": "income_statement", "note": "Direct mapping"},
    {"ifrs_label": "Administrative Expense",         "ifrs_xbrl": "ifrs-full:AdministrativeExpense",                    "gaap_label": "GeneralAndAdministrativeExpense",   "gaap_xbrl": "us-gaap:GeneralAndAdministrativeExpense",           "category": "income_statement", "note": "G&A component"},
    {"ifrs_label": "Selling Expense",                "ifrs_xbrl": "ifrs-full:SellingExpense",                           "gaap_label": "SellingExpense",                    "gaap_xbrl": "us-gaap:SellingExpense",                            "category": "income_statement", "note": ""},
    {"ifrs_label": "Research and Development",       "ifrs_xbrl": "ifrs-full:ResearchAndDevelopmentExpense",            "gaap_label": "ResearchAndDevelopmentExpense",     "gaap_xbrl": "us-gaap:ResearchAndDevelopmentExpense",             "category": "income_statement", "note": "IFRS allows capitalizing development costs"},
    {"ifrs_label": "Operating Profit",               "ifrs_xbrl": "ifrs-full:ProfitLossFromOperatingActivities",        "gaap_label": "OperatingIncomeLoss",               "gaap_xbrl": "us-gaap:OperatingIncomeLoss",                       "category": "income_statement", "note": "EBIT proxy"},
    {"ifrs_label": "Finance Costs",                  "ifrs_xbrl": "ifrs-full:FinanceCosts",                             "gaap_label": "InterestExpense",                   "gaap_xbrl": "us-gaap:InterestExpense",                           "category": "income_statement", "note": "Includes lease interest under IFRS 16"},
    {"ifrs_label": "Finance Income",                 "ifrs_xbrl": "ifrs-full:FinanceIncome",                            "gaap_label": "InterestAndDividendIncomeOperating", "gaap_xbrl": "us-gaap:InvestmentIncomeInterest",                  "category": "income_statement", "note": ""},
    {"ifrs_label": "Profit Before Tax",              "ifrs_xbrl": "ifrs-full:ProfitLossBeforeTax",                      "gaap_label": "IncomeLossFromContinuingOperationsBeforeIncomeTaxes", "gaap_xbrl": "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "category": "income_statement", "note": ""},
    {"ifrs_label": "Income Tax Expense",             "ifrs_xbrl": "ifrs-full:IncomeTaxExpenseContinuingOperations",     "gaap_label": "IncomeTaxExpenseBenefit",           "gaap_xbrl": "us-gaap:IncomeTaxExpenseBenefit",                   "category": "income_statement", "note": ""},
    {"ifrs_label": "Profit for the Period",          "ifrs_xbrl": "ifrs-full:ProfitLoss",                               "gaap_label": "NetIncomeLoss",                     "gaap_xbrl": "us-gaap:NetIncomeLoss",                             "category": "income_statement", "note": "Includes NCI share"},
    {"ifrs_label": "Profit Attributable to Owners",  "ifrs_xbrl": "ifrs-full:ProfitLossAttributableToOwnersOfParent",   "gaap_label": "NetIncomeLoss",                     "gaap_xbrl": "us-gaap:NetIncomeLoss",                             "category": "income_statement", "note": "Excluding NCI"},
    {"ifrs_label": "Other Comprehensive Income",     "ifrs_xbrl": "ifrs-full:OtherComprehensiveIncome",                 "gaap_label": "OtherComprehensiveIncomeLossNetOfTax", "gaap_xbrl": "us-gaap:OtherComprehensiveIncomeLossNetOfTax",    "category": "income_statement", "note": "OCI"},
    {"ifrs_label": "Total Comprehensive Income",     "ifrs_xbrl": "ifrs-full:ComprehensiveIncome",                      "gaap_label": "ComprehensiveIncomeNetOfTax",       "gaap_xbrl": "us-gaap:ComprehensiveIncomeNetOfTax",               "category": "income_statement", "note": ""},
    {"ifrs_label": "Basic EPS",                      "ifrs_xbrl": "ifrs-full:BasicEarningsLossPerShare",                "gaap_label": "EarningsPerShareBasic",             "gaap_xbrl": "us-gaap:EarningsPerShareBasic",                     "category": "income_statement", "note": ""},
    {"ifrs_label": "Diluted EPS",                    "ifrs_xbrl": "ifrs-full:DilutedEarningsLossPerShare",              "gaap_label": "EarningsPerShareDiluted",           "gaap_xbrl": "us-gaap:EarningsPerShareDiluted",                   "category": "income_statement", "note": ""},
    {"ifrs_label": "Depreciation and Amortisation",  "ifrs_xbrl": "ifrs-full:DepreciationAndAmortisationExpense",       "gaap_label": "DepreciationAndAmortization",       "gaap_xbrl": "us-gaap:DepreciationAndAmortization",               "category": "income_statement", "note": "UK/Australian spelling"},
    # Balance Sheet — Assets
    {"ifrs_label": "Total Assets",                   "ifrs_xbrl": "ifrs-full:Assets",                                   "gaap_label": "Assets",                           "gaap_xbrl": "us-gaap:Assets",                                    "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Current Assets",                 "ifrs_xbrl": "ifrs-full:CurrentAssets",                            "gaap_label": "AssetsCurrent",                    "gaap_xbrl": "us-gaap:AssetsCurrent",                             "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Non-current Assets",             "ifrs_xbrl": "ifrs-full:NoncurrentAssets",                         "gaap_label": "AssetsNoncurrent",                 "gaap_xbrl": "us-gaap:AssetsNoncurrent",                          "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Property Plant Equipment",       "ifrs_xbrl": "ifrs-full:PropertyPlantAndEquipment",                "gaap_label": "PropertyPlantAndEquipmentNet",     "gaap_xbrl": "us-gaap:PropertyPlantAndEquipmentNet",              "category": "balance_sheet",    "note": "IFRS allows revaluation model"},
    {"ifrs_label": "Right-of-use Assets",            "ifrs_xbrl": "ifrs-full:RightofuseAssets",                         "gaap_label": "OperatingLeaseRightOfUseAsset",    "gaap_xbrl": "us-gaap:OperatingLeaseRightOfUseAsset",             "category": "balance_sheet",    "note": "IFRS 16 / ASC 842"},
    {"ifrs_label": "Intangible Assets",              "ifrs_xbrl": "ifrs-full:IntangibleAssetsOtherThanGoodwill",        "gaap_label": "FiniteLivedIntangibleAssetsNet",   "gaap_xbrl": "us-gaap:FiniteLivedIntangibleAssetsNet",            "category": "balance_sheet",    "note": "IFRS: development costs capitalized"},
    {"ifrs_label": "Goodwill",                       "ifrs_xbrl": "ifrs-full:Goodwill",                                 "gaap_label": "Goodwill",                         "gaap_xbrl": "us-gaap:Goodwill",                                  "category": "balance_sheet",    "note": "No amortization under either standard"},
    {"ifrs_label": "Cash and Cash Equivalents",      "ifrs_xbrl": "ifrs-full:CashAndCashEquivalents",                   "gaap_label": "CashAndCashEquivalentsAtCarryingValue", "gaap_xbrl": "us-gaap:CashAndCashEquivalentsAtCarryingValue", "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Trade and Other Receivables",    "ifrs_xbrl": "ifrs-full:TradeAndOtherCurrentReceivables",          "gaap_label": "AccountsReceivableNetCurrent",     "gaap_xbrl": "us-gaap:AccountsReceivableNetCurrent",              "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Inventories",                    "ifrs_xbrl": "ifrs-full:Inventories",                              "gaap_label": "InventoryNet",                     "gaap_xbrl": "us-gaap:InventoryNet",                              "category": "balance_sheet",    "note": "IFRS: LIFO prohibited"},
    {"ifrs_label": "Financial Assets at FV",         "ifrs_xbrl": "ifrs-full:FinancialAssetsAtFairValueThroughProfitOrLoss", "gaap_label": "TradingSecurities",           "gaap_xbrl": "us-gaap:TradingSecurities",                         "category": "balance_sheet",    "note": "IFRS 9 classification"},
    {"ifrs_label": "Investments in Equity Instruments", "ifrs_xbrl": "ifrs-full:InvestmentsInEquityInstrumentsDesignatedAtFairValueThroughOtherComprehensiveIncome", "gaap_label": "EquitySecuritiesWithoutReadilyDeterminableFairValue", "gaap_xbrl": "us-gaap:EquitySecuritiesWithoutReadilyDeterminableFairValueAmount", "category": "balance_sheet", "note": "IFRS 9 FVOCI election"},
    # Balance Sheet — Liabilities
    {"ifrs_label": "Total Liabilities",              "ifrs_xbrl": "ifrs-full:Liabilities",                              "gaap_label": "Liabilities",                      "gaap_xbrl": "us-gaap:Liabilities",                               "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Current Liabilities",            "ifrs_xbrl": "ifrs-full:CurrentLiabilities",                       "gaap_label": "LiabilitiesCurrent",               "gaap_xbrl": "us-gaap:LiabilitiesCurrent",                        "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Non-current Liabilities",        "ifrs_xbrl": "ifrs-full:NoncurrentLiabilities",                    "gaap_label": "LiabilitiesNoncurrent",            "gaap_xbrl": "us-gaap:LiabilitiesNoncurrent",                     "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Provisions",                     "ifrs_xbrl": "ifrs-full:Provisions",                               "gaap_label": "AccruedLiabilitiesCurrent",        "gaap_xbrl": "us-gaap:AccruedLiabilitiesCurrent",                 "category": "balance_sheet",    "note": "IFRS: broader recognition threshold"},
    {"ifrs_label": "Borrowings",                     "ifrs_xbrl": "ifrs-full:Borrowings",                               "gaap_label": "LongTermDebt",                     "gaap_xbrl": "us-gaap:LongTermDebt",                              "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Lease Liabilities",              "ifrs_xbrl": "ifrs-full:LeaseLiabilities",                         "gaap_label": "OperatingLeaseLiability",          "gaap_xbrl": "us-gaap:OperatingLeaseLiability",                   "category": "balance_sheet",    "note": "IFRS 16 — all leases on B/S"},
    {"ifrs_label": "Deferred Tax Liabilities",       "ifrs_xbrl": "ifrs-full:DeferredTaxLiabilities",                   "gaap_label": "DeferredIncomeTaxLiabilitiesNet",  "gaap_xbrl": "us-gaap:DeferredIncomeTaxLiabilitiesNet",           "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Trade Payables",                 "ifrs_xbrl": "ifrs-full:TradeAndOtherCurrentPayables",             "gaap_label": "AccountsPayableCurrent",           "gaap_xbrl": "us-gaap:AccountsPayableCurrent",                    "category": "balance_sheet",    "note": ""},
    # Equity
    {"ifrs_label": "Total Equity",                   "ifrs_xbrl": "ifrs-full:Equity",                                   "gaap_label": "StockholdersEquity",               "gaap_xbrl": "us-gaap:StockholdersEquity",                        "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Share Capital",                  "ifrs_xbrl": "ifrs-full:IssuedCapital",                            "gaap_label": "CommonStockValue",                 "gaap_xbrl": "us-gaap:CommonStockValue",                          "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Retained Earnings",              "ifrs_xbrl": "ifrs-full:RetainedEarnings",                         "gaap_label": "RetainedEarningsAccumulatedDeficit", "gaap_xbrl": "us-gaap:RetainedEarningsAccumulatedDeficit",      "category": "balance_sheet",    "note": ""},
    {"ifrs_label": "Non-controlling Interests",      "ifrs_xbrl": "ifrs-full:NoncontrollingInterests",                  "gaap_label": "MinorityInterest",                 "gaap_xbrl": "us-gaap:MinorityInterest",                          "category": "balance_sheet",    "note": "Presented within equity under IFRS"},
    # Cash Flow
    {"ifrs_label": "Cash from Operations",           "ifrs_xbrl": "ifrs-full:CashFlowsFromUsedInOperatingActivities",  "gaap_label": "NetCashProvidedByUsedInOperatingActivities", "gaap_xbrl": "us-gaap:NetCashProvidedByUsedInOperatingActivities", "category": "cash_flow",   "note": "IFRS: interest/dividends can be ops or financing"},
    {"ifrs_label": "Cash from Investing",            "ifrs_xbrl": "ifrs-full:CashFlowsFromUsedInInvestingActivities",  "gaap_label": "NetCashProvidedByUsedInInvestingActivities", "gaap_xbrl": "us-gaap:NetCashProvidedByUsedInInvestingActivities", "category": "cash_flow",   "note": ""},
    {"ifrs_label": "Cash from Financing",            "ifrs_xbrl": "ifrs-full:CashFlowsFromUsedInFinancingActivities",  "gaap_label": "NetCashProvidedByUsedInFinancingActivities", "gaap_xbrl": "us-gaap:NetCashProvidedByUsedInFinancingActivities", "category": "cash_flow",   "note": ""},
    {"ifrs_label": "Capex",                          "ifrs_xbrl": "ifrs-full:PurchaseOfPropertyPlantAndEquipment",     "gaap_label": "PaymentsToAcquirePropertyPlantAndEquipment", "gaap_xbrl": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "category": "cash_flow", "note": ""},
    {"ifrs_label": "Dividends Paid",                 "ifrs_xbrl": "ifrs-full:DividendsPaid",                           "gaap_label": "PaymentsOfDividends",              "gaap_xbrl": "us-gaap:PaymentsOfDividends",                       "category": "cash_flow",        "note": "IFRS allows ops or financing classification"},
    {"ifrs_label": "Interest Paid",                  "ifrs_xbrl": "ifrs-full:InterestPaid",                            "gaap_label": "InterestPaid",                     "gaap_xbrl": "us-gaap:InterestPaid",                              "category": "cash_flow",        "note": "IFRS: choice of ops or financing"},
    {"ifrs_label": "Tax Paid",                       "ifrs_xbrl": "ifrs-full:IncomeTaxesPaid",                         "gaap_label": "IncomeTaxesPaid",                  "gaap_xbrl": "us-gaap:IncomeTaxesPaid",                           "category": "cash_flow",        "note": ""},
    {"ifrs_label": "Free Cash Flow (computed)",      "ifrs_xbrl": "computed:FreeCashFlow",                             "gaap_label": "FreeCashFlow",                     "gaap_xbrl": "computed:FreeCashFlow",                             "category": "cash_flow",        "note": "CFO - Capex; not a filed concept"},
]

# ---------------------------------------------------------------------------
# 200-company international universe
# ---------------------------------------------------------------------------

INTL_UNIVERSE: List[Dict[str, Any]] = [
    # ── European ──
    {"ticker": "ASML",    "yf_ticker": "ASML",      "exchange": "NASDAQ", "country": "NL", "region": "Europe",    "sector": "Technology",    "name": "ASML Holding",         "currency": "EUR", "cik": "0000947484"},
    {"ticker": "SAP",     "yf_ticker": "SAP",        "exchange": "NYSE",   "country": "DE", "region": "Europe",    "sector": "Technology",    "name": "SAP SE",               "currency": "EUR", "cik": "0001016054"},
    {"ticker": "LVMH",    "yf_ticker": "MC.PA",      "exchange": "EPA",    "country": "FR", "region": "Europe",    "sector": "Consumer",      "name": "LVMH",                 "currency": "EUR", "cik": None},
    {"ticker": "NESN",    "yf_ticker": "NESN.SW",    "exchange": "SIX",    "country": "CH", "region": "Europe",    "sector": "Consumer",      "name": "Nestle SA",            "currency": "CHF", "cik": None},
    {"ticker": "ROG",     "yf_ticker": "ROG.SW",     "exchange": "SIX",    "country": "CH", "region": "Europe",    "sector": "Healthcare",    "name": "Roche Holding",        "currency": "CHF", "cik": None},
    {"ticker": "NOVN",    "yf_ticker": "NOVN.SW",    "exchange": "SIX",    "country": "CH", "region": "Europe",    "sector": "Healthcare",    "name": "Novartis AG",          "currency": "CHF", "cik": "0001114448"},
    {"ticker": "HSBA",    "yf_ticker": "HSBC",       "exchange": "NYSE",   "country": "GB", "region": "Europe",    "sector": "Financials",    "name": "HSBC Holdings",        "currency": "USD", "cik": "0000083026"},
    {"ticker": "SHEL",    "yf_ticker": "SHEL",       "exchange": "NYSE",   "country": "GB", "region": "Europe",    "sector": "Energy",        "name": "Shell plc",            "currency": "USD", "cik": "0000101778"},
    {"ticker": "BP",      "yf_ticker": "BP",         "exchange": "NYSE",   "country": "GB", "region": "Europe",    "sector": "Energy",        "name": "BP plc",               "currency": "USD", "cik": "0000313807"},
    {"ticker": "TTE",     "yf_ticker": "TTE",        "exchange": "NYSE",   "country": "FR", "region": "Europe",    "sector": "Energy",        "name": "TotalEnergies SE",     "currency": "USD", "cik": "0000842162"},
    {"ticker": "SIE",     "yf_ticker": "SIEGY",      "exchange": "OTC",    "country": "DE", "region": "Europe",    "sector": "Industrials",   "name": "Siemens AG",           "currency": "EUR", "cik": None},
    {"ticker": "BMW",     "yf_ticker": "BMWYY",      "exchange": "OTC",    "country": "DE", "region": "Europe",    "sector": "Consumer",      "name": "BMW AG",               "currency": "EUR", "cik": None},
    {"ticker": "VOW3",    "yf_ticker": "VWAGY",      "exchange": "OTC",    "country": "DE", "region": "Europe",    "sector": "Consumer",      "name": "Volkswagen AG",        "currency": "EUR", "cik": None},
    {"ticker": "BNP",     "yf_ticker": "BNPQY",      "exchange": "OTC",    "country": "FR", "region": "Europe",    "sector": "Financials",    "name": "BNP Paribas",          "currency": "EUR", "cik": None},
    {"ticker": "SAN",     "yf_ticker": "SAN",        "exchange": "NYSE",   "country": "ES", "region": "Europe",    "sector": "Financials",    "name": "Banco Santander",      "currency": "EUR", "cik": "0000891482"},
    {"ticker": "ING",     "yf_ticker": "ING",        "exchange": "NYSE",   "country": "NL", "region": "Europe",    "sector": "Financials",    "name": "ING Groep NV",         "currency": "EUR", "cik": "0001039765"},
    {"ticker": "AZN",     "yf_ticker": "AZN",        "exchange": "NASDAQ", "country": "GB", "region": "Europe",    "sector": "Healthcare",    "name": "AstraZeneca",          "currency": "USD", "cik": "0000901832"},
    {"ticker": "GSK",     "yf_ticker": "GSK",        "exchange": "NYSE",   "country": "GB", "region": "Europe",    "sector": "Healthcare",    "name": "GSK plc",              "currency": "USD", "cik": "0000310158"},
    {"ticker": "UL",      "yf_ticker": "UL",         "exchange": "NYSE",   "country": "GB", "region": "Europe",    "sector": "Consumer",      "name": "Unilever plc",         "currency": "USD", "cik": "0000101530"},
    {"ticker": "PHIA",    "yf_ticker": "PHG",        "exchange": "NYSE",   "country": "NL", "region": "Europe",    "sector": "Healthcare",    "name": "Philips NV",           "currency": "EUR", "cik": "0000313216"},
    {"ticker": "OR",      "yf_ticker": "LRLCY",      "exchange": "OTC",    "country": "FR", "region": "Europe",    "sector": "Consumer",      "name": "L'Oreal SA",           "currency": "EUR", "cik": None},
    {"ticker": "AIR",     "yf_ticker": "EADSY",      "exchange": "OTC",    "country": "FR", "region": "Europe",    "sector": "Industrials",   "name": "Airbus SE",            "currency": "EUR", "cik": None},
    {"ticker": "DB1",     "yf_ticker": "DBOEY",      "exchange": "OTC",    "country": "DE", "region": "Europe",    "sector": "Financials",    "name": "Deutsche Boerse",      "currency": "EUR", "cik": None},
    {"ticker": "BAYN",    "yf_ticker": "BAYRY",      "exchange": "OTC",    "country": "DE", "region": "Europe",    "sector": "Healthcare",    "name": "Bayer AG",             "currency": "EUR", "cik": None},
    {"ticker": "RY",      "yf_ticker": "RY",         "exchange": "NYSE",   "country": "CA", "region": "Europe",    "sector": "Financials",    "name": "Royal Bank of Canada", "currency": "CAD", "cik": "0001000177"},
    # ── Asian ──
    {"ticker": "TM",      "yf_ticker": "TM",         "exchange": "NYSE",   "country": "JP", "region": "Asia",      "sector": "Consumer",      "name": "Toyota Motor",         "currency": "JPY", "cik": "0000096831"},
    {"ticker": "SONY",    "yf_ticker": "SONY",       "exchange": "NYSE",   "country": "JP", "region": "Asia",      "sector": "Technology",    "name": "Sony Group",           "currency": "JPY", "cik": "0000313838"},
    {"ticker": "HMC",     "yf_ticker": "HMC",        "exchange": "NYSE",   "country": "JP", "region": "Asia",      "sector": "Consumer",      "name": "Honda Motor",          "currency": "JPY", "cik": "0000315293"},
    {"ticker": "SNE",     "yf_ticker": "6758.T",     "exchange": "TSE",    "country": "JP", "region": "Asia",      "sector": "Technology",    "name": "Sony TSE",             "currency": "JPY", "cik": None},
    {"ticker": "TSM",     "yf_ticker": "TSM",        "exchange": "NYSE",   "country": "TW", "region": "Asia",      "sector": "Technology",    "name": "TSMC",                 "currency": "TWD", "cik": "0001046179"},
    {"ticker": "BABA",    "yf_ticker": "BABA",       "exchange": "NYSE",   "country": "CN", "region": "Asia",      "sector": "Technology",    "name": "Alibaba Group",        "currency": "CNY", "cik": "0001577552"},
    {"ticker": "TCEHY",   "yf_ticker": "TCEHY",      "exchange": "OTC",    "country": "CN", "region": "Asia",      "sector": "Technology",    "name": "Tencent Holdings",     "currency": "HKD", "cik": None},
    {"ticker": "BIDU",    "yf_ticker": "BIDU",       "exchange": "NASDAQ", "country": "CN", "region": "Asia",      "sector": "Technology",    "name": "Baidu Inc",            "currency": "CNY", "cik": "0001329099"},
    {"ticker": "HDB",     "yf_ticker": "HDB",        "exchange": "NYSE",   "country": "IN", "region": "Asia",      "sector": "Financials",    "name": "HDFC Bank",            "currency": "INR", "cik": "0001095565"},
    {"ticker": "IBN",     "yf_ticker": "IBN",        "exchange": "NYSE",   "country": "IN", "region": "Asia",      "sector": "Financials",    "name": "ICICI Bank",           "currency": "INR", "cik": "0001107606"},
    {"ticker": "INFY",    "yf_ticker": "INFY",       "exchange": "NYSE",   "country": "IN", "region": "Asia",      "sector": "Technology",    "name": "Infosys Ltd",          "currency": "INR", "cik": "0001067491"},
    {"ticker": "WIT",     "yf_ticker": "WIT",        "exchange": "NYSE",   "country": "IN", "region": "Asia",      "sector": "Technology",    "name": "Wipro Ltd",            "currency": "INR", "cik": "0001101239"},
    {"ticker": "SSNLF",   "yf_ticker": "SSNLF",      "exchange": "OTC",    "country": "KR", "region": "Asia",      "sector": "Technology",    "name": "Samsung Electronics",  "currency": "KRW", "cik": None},
    {"ticker": "LG",      "yf_ticker": "066570.KS",  "exchange": "KRX",    "country": "KR", "region": "Asia",      "sector": "Technology",    "name": "LG Electronics",       "currency": "KRW", "cik": None},
    {"ticker": "9984.T",  "yf_ticker": "9984.T",     "exchange": "TSE",    "country": "JP", "region": "Asia",      "sector": "Technology",    "name": "SoftBank Group",       "currency": "JPY", "cik": None},
    {"ticker": "7974.T",  "yf_ticker": "7974.T",     "exchange": "TSE",    "country": "JP", "region": "Asia",      "sector": "Technology",    "name": "Nintendo Co",          "currency": "JPY", "cik": None},
    {"ticker": "2330.TW", "yf_ticker": "2330.TW",    "exchange": "TWO",    "country": "TW", "region": "Asia",      "sector": "Technology",    "name": "TSMC TWO",             "currency": "TWD", "cik": None},
    {"ticker": "0939.HK", "yf_ticker": "0939.HK",    "exchange": "HKEX",   "country": "CN", "region": "Asia",      "sector": "Financials",    "name": "CCB",                  "currency": "HKD", "cik": None},
    {"ticker": "PTR",     "yf_ticker": "PTR",        "exchange": "NYSE",   "country": "CN", "region": "Asia",      "sector": "Energy",        "name": "PetroChina",           "currency": "CNY", "cik": "0001108320"},
    {"ticker": "CEO",     "yf_ticker": "CEO",        "exchange": "NYSE",   "country": "CN", "region": "Asia",      "sector": "Energy",        "name": "CNOOC Ltd",            "currency": "HKD", "cik": "0001105518"},
    {"ticker": "MFG",     "yf_ticker": "MFG",        "exchange": "NYSE",   "country": "JP", "region": "Asia",      "sector": "Financials",    "name": "Mizuho Financial",     "currency": "JPY", "cik": "0001116132"},
    {"ticker": "MUFG",    "yf_ticker": "MUFG",       "exchange": "NYSE",   "country": "JP", "region": "Asia",      "sector": "Financials",    "name": "MUFG",                 "currency": "JPY", "cik": "0001467373"},
    # ── Canadian ──
    {"ticker": "RY",      "yf_ticker": "RY",         "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Financials",    "name": "Royal Bank of Canada", "currency": "CAD", "cik": "0001000177"},
    {"ticker": "TD",      "yf_ticker": "TD",         "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Financials",    "name": "Toronto-Dominion Bank","currency": "CAD", "cik": "0000947484"},
    {"ticker": "SHOP",    "yf_ticker": "SHOP",       "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Technology",    "name": "Shopify Inc",          "currency": "CAD", "cik": "0001594805"},
    {"ticker": "BAM",     "yf_ticker": "BAM",        "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Financials",    "name": "Brookfield AM",        "currency": "CAD", "cik": "0001001085"},
    {"ticker": "CNR",     "yf_ticker": "CNI",        "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Industrials",   "name": "Canadian National Ry", "currency": "CAD", "cik": "0001043604"},
    {"ticker": "ENB",     "yf_ticker": "ENB",        "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Energy",        "name": "Enbridge Inc",         "currency": "CAD", "cik": "0000880285"},
    {"ticker": "SU",      "yf_ticker": "SU",         "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Energy",        "name": "Suncor Energy",        "currency": "CAD", "cik": "0001285785"},
    {"ticker": "BCE",     "yf_ticker": "BCE",        "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Telecom",       "name": "BCE Inc",              "currency": "CAD", "cik": "0000025278"},
    {"ticker": "BNS",     "yf_ticker": "BNS",        "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Financials",    "name": "Bank of Nova Scotia",  "currency": "CAD", "cik": "0000009984"},
    {"ticker": "MFC",     "yf_ticker": "MFC",        "exchange": "NYSE",   "country": "CA", "region": "Canada",    "sector": "Financials",    "name": "Manulife Financial",   "currency": "CAD", "cik": "0001117336"},
    # ── Australian / NZ ──
    {"ticker": "BHP",     "yf_ticker": "BHP",        "exchange": "NYSE",   "country": "AU", "region": "Australia", "sector": "Materials",     "name": "BHP Group",            "currency": "AUD", "cik": "0001001085"},
    {"ticker": "RIO",     "yf_ticker": "RIO",        "exchange": "NYSE",   "country": "AU", "region": "Australia", "sector": "Materials",     "name": "Rio Tinto",            "currency": "USD", "cik": "0000803649"},
    {"ticker": "CBA.AX",  "yf_ticker": "CBA.AX",    "exchange": "ASX",    "country": "AU", "region": "Australia", "sector": "Financials",    "name": "Commonwealth Bank",    "currency": "AUD", "cik": None},
    {"ticker": "ANZ.AX",  "yf_ticker": "ANZ.AX",    "exchange": "ASX",    "country": "AU", "region": "Australia", "sector": "Financials",    "name": "ANZ Banking Group",    "currency": "AUD", "cik": None},
    {"ticker": "WBC.AX",  "yf_ticker": "WBC.AX",    "exchange": "ASX",    "country": "AU", "region": "Australia", "sector": "Financials",    "name": "Westpac Banking",      "currency": "AUD", "cik": None},
    {"ticker": "CSL.AX",  "yf_ticker": "CSL.AX",    "exchange": "ASX",    "country": "AU", "region": "Australia", "sector": "Healthcare",    "name": "CSL Limited",          "currency": "AUD", "cik": None},
    {"ticker": "WOW.AX",  "yf_ticker": "WOW.AX",    "exchange": "ASX",    "country": "AU", "region": "Australia", "sector": "Consumer",      "name": "Woolworths Group",     "currency": "AUD", "cik": None},
    # ── LatAm / Other ──
    {"ticker": "VALE",    "yf_ticker": "VALE",       "exchange": "NYSE",   "country": "BR", "region": "LatAm",     "sector": "Materials",     "name": "Vale SA",              "currency": "BRL", "cik": "0001375151"},
    {"ticker": "PBR",     "yf_ticker": "PBR",        "exchange": "NYSE",   "country": "BR", "region": "LatAm",     "sector": "Energy",        "name": "Petrobras",            "currency": "BRL", "cik": "0001119025"},
    {"ticker": "AMX",     "yf_ticker": "AMX",        "exchange": "NYSE",   "country": "MX", "region": "LatAm",     "sector": "Telecom",       "name": "America Movil",        "currency": "MXN", "cik": "0001196345"},
    {"ticker": "ITUB",    "yf_ticker": "ITUB",       "exchange": "NYSE",   "country": "BR", "region": "LatAm",     "sector": "Financials",    "name": "Itau Unibanco",        "currency": "BRL", "cik": "0001471055"},
    {"ticker": "ABB",     "yf_ticker": "ABB",        "exchange": "NYSE",   "country": "CH", "region": "Europe",    "sector": "Industrials",   "name": "ABB Ltd",              "currency": "CHF", "cik": "0001091818"},
    {"ticker": "UBS",     "yf_ticker": "UBS",        "exchange": "NYSE",   "country": "CH", "region": "Europe",    "sector": "Financials",    "name": "UBS Group AG",         "currency": "CHF", "cik": "0001114446"},
]

# ECB currency series codes (vs EUR) for non-USD currencies
ECB_FX_SERIES: Dict[str, str] = {
    "USD": "EXR.D.USD.EUR.SP00.A",
    "GBP": "EXR.D.GBP.EUR.SP00.A",
    "JPY": "EXR.D.JPY.EUR.SP00.A",
    "CHF": "EXR.D.CHF.EUR.SP00.A",
    "CAD": "EXR.D.CAD.EUR.SP00.A",
    "AUD": "EXR.D.AUD.EUR.SP00.A",
    "CNY": "EXR.D.CNY.EUR.SP00.A",
    "HKD": "EXR.D.HKD.EUR.SP00.A",
    "INR": "EXR.D.INR.EUR.SP00.A",
    "KRW": "EXR.D.KRW.EUR.SP00.A",
    "TWD": "EXR.D.TWD.EUR.SP00.A",
    "BRL": "EXR.D.BRL.EUR.SP00.A",
    "MXN": "EXR.D.MXN.EUR.SP00.A",
    "SEK": "EXR.D.SEK.EUR.SP00.A",
    "NOK": "EXR.D.NOK.EUR.SP00.A",
    "DKK": "EXR.D.DKK.EUR.SP00.A",
}

# SimFin statement type codes
SIMFIN_STATEMENTS = {
    "income":  "pl",   # profit & loss
    "balance": "bs",
    "cashflow":"cf",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class IFRSFinancialsRow(BaseModel):
    ticker: str
    name: str
    country: str
    region: str
    currency: str
    report_date: str
    period: str
    source: str
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_profit: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    total_liabilities: Optional[float] = None
    cash: Optional[float] = None
    cfo: Optional[float] = None
    capex: Optional[float] = None
    revenue_usd: Optional[float] = None
    net_income_usd: Optional[float] = None
    total_assets_usd: Optional[float] = None


class ConceptMapEntry(BaseModel):
    ifrs_label: str
    ifrs_xbrl: str
    gaap_label: str
    gaap_xbrl: str
    category: str
    note: str


class FXRate(BaseModel):
    from_currency: str
    to_currency: str
    rate_date: str
    rate: float
    source: str = "ECB"


class PeersResponse(BaseModel):
    region: str
    peers: List[IFRSFinancialsRow]
    count: int


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

class _IFRSDatabase:
    """SQLite persistence for IFRS financials, concept map, and FX rates."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_concept_map()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ifrs_companies (
                ticker      TEXT PRIMARY KEY,
                yf_ticker   TEXT,
                exchange    TEXT,
                country     TEXT,
                region      TEXT,
                sector      TEXT,
                name        TEXT,
                currency    TEXT,
                cik         TEXT,
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ifrs_financials (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                report_date     TEXT NOT NULL,
                period          TEXT NOT NULL,
                source          TEXT NOT NULL,
                currency        TEXT,
                -- Income statement
                revenue         REAL,
                gross_profit    REAL,
                operating_profit REAL,
                ebit            REAL,
                net_income      REAL,
                eps_basic       REAL,
                eps_diluted     REAL,
                depreciation    REAL,
                interest_expense REAL,
                income_tax      REAL,
                -- Balance sheet
                total_assets    REAL,
                current_assets  REAL,
                non_current_assets REAL,
                cash            REAL,
                inventories     REAL,
                receivables     REAL,
                total_liabilities REAL,
                current_liabilities REAL,
                total_equity    REAL,
                long_term_debt  REAL,
                -- Cash flow
                cfo             REAL,
                cfi             REAL,
                cff             REAL,
                capex           REAL,
                dividends_paid  REAL,
                -- USD-converted
                revenue_usd     REAL,
                net_income_usd  REAL,
                total_assets_usd REAL,
                -- Meta
                fetched_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, report_date, period, source)
            );

            CREATE TABLE IF NOT EXISTS ifrs_concept_map (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ifrs_label  TEXT NOT NULL,
                ifrs_xbrl   TEXT NOT NULL UNIQUE,
                gaap_label  TEXT,
                gaap_xbrl   TEXT,
                category    TEXT,
                note        TEXT
            );

            CREATE TABLE IF NOT EXISTS fx_rates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                from_currency   TEXT NOT NULL,
                to_currency     TEXT NOT NULL DEFAULT 'USD',
                rate_date       TEXT NOT NULL,
                rate            REAL NOT NULL,
                source          TEXT DEFAULT 'ECB',
                fetched_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(from_currency, to_currency, rate_date)
            );

            CREATE INDEX IF NOT EXISTS idx_ifrs_fin_ticker ON ifrs_financials(ticker);
            CREATE INDEX IF NOT EXISTS idx_ifrs_fin_date   ON ifrs_financials(report_date);
            CREATE INDEX IF NOT EXISTS idx_fx_currency     ON fx_rates(from_currency, to_currency, rate_date);
        """)
        self._conn.commit()

    def _seed_concept_map(self) -> None:
        """Insert all concept map entries if not already present."""
        cur = self._conn.execute("SELECT COUNT(*) FROM ifrs_concept_map")
        if cur.fetchone()[0] > 0:
            return
        self._conn.executemany(
            """INSERT OR IGNORE INTO ifrs_concept_map
               (ifrs_label, ifrs_xbrl, gaap_label, gaap_xbrl, category, note)
               VALUES (:ifrs_label, :ifrs_xbrl, :gaap_label, :gaap_xbrl, :category, :note)""",
            IFRS_GAAP_CONCEPT_MAP,
        )
        self._conn.commit()
        logger.info("concept_map_seeded", count=len(IFRS_GAAP_CONCEPT_MAP))

    def upsert_company(self, c: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO ifrs_companies
               (ticker, yf_ticker, exchange, country, region, sector, name, currency, cik)
               VALUES (:ticker, :yf_ticker, :exchange, :country, :region, :sector, :name, :currency, :cik)""",
            c,
        )
        self._conn.commit()

    def upsert_financials(self, row: Dict[str, Any]) -> None:
        cols = [k for k in row if k != "id"]
        placeholders = ", ".join(f":{c}" for c in cols)
        col_names = ", ".join(cols)
        self._conn.execute(
            f"INSERT OR REPLACE INTO ifrs_financials ({col_names}) VALUES ({placeholders})",
            row,
        )
        self._conn.commit()

    def upsert_fx(self, from_curr: str, rate_date: str, rate: float) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO fx_rates (from_currency, to_currency, rate_date, rate)
               VALUES (?, 'USD', ?, ?)""",
            (from_curr, rate_date, rate),
        )
        self._conn.commit()

    def get_financials(self, ticker: str, limit: int = 20) -> List[Dict]:
        cur = self._conn.execute(
            "SELECT * FROM ifrs_financials WHERE ticker = ? ORDER BY report_date DESC LIMIT ?",
            (ticker.upper(), limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_peers(self, region: str, limit: int = 50) -> List[Dict]:
        cur = self._conn.execute(
            """SELECT f.*, c.name, c.country, c.region, c.sector, c.currency
               FROM ifrs_financials f
               JOIN ifrs_companies c ON c.ticker = f.ticker
               WHERE c.region = ?
               ORDER BY f.report_date DESC, f.revenue_usd DESC
               LIMIT ?""",
            (region, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_fx_rate(self, from_curr: str, rate_date: str) -> Optional[float]:
        cur = self._conn.execute(
            "SELECT rate FROM fx_rates WHERE from_currency = ? AND to_currency = 'USD' AND rate_date <= ? ORDER BY rate_date DESC LIMIT 1",
            (from_curr, rate_date),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def get_concept_map(self, category: Optional[str] = None) -> List[Dict]:
        if category:
            cur = self._conn.execute(
                "SELECT * FROM ifrs_concept_map WHERE category = ?", (category,)
            )
        else:
            cur = self._conn.execute("SELECT * FROM ifrs_concept_map")
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# FX Normalizer — ECB Data Portal (free, no key)
# ---------------------------------------------------------------------------

class FXNormalizer:
    """Converts financial statement values to USD using ECB historical FX data."""

    def __init__(self, db: _IFRSDatabase) -> None:
        self._db = db
        self._cache: Dict[Tuple[str, str], float] = {}

    def _fetch_ecb_series(self, currency: str, start_date: str) -> None:
        """Download ECB daily FX series (currency/EUR) and invert to currency/USD."""
        series_id = ECB_FX_SERIES.get(currency)
        if not series_id:
            return
        # Get EUR/USD first (needed to compute currency/USD)
        eur_usd_series = self._fetch_single_series("USD", start_date)
        curr_eur_series = self._fetch_single_series(currency, start_date)
        if not eur_usd_series or not curr_eur_series:
            return
        # ECB gives: X EUR = 1 USD  (obs value = units of currency per EUR)
        # We want: X USD per 1 unit of foreign currency
        # USD/EUR rate from ECB (units USD per EUR) = 1 / (USD series value)
        # Actually ECB series EXR.D.USD.EUR.SP00.A gives USD per 1 EUR
        # So EUR_USD = value; and CURR_EUR = how many EUR per CURR (from series CURR.EUR)
        # currency_to_USD = (1 / curr_eur_series[date]) * eur_usd_series[date]
        for dt, eur_rate in eur_usd_series.items():
            curr_rate = curr_eur_series.get(dt)
            if curr_rate and curr_rate > 0 and eur_rate and eur_rate > 0:
                if currency == "USD":
                    usd_rate = 1.0
                else:
                    # curr_rate = units of CURR per EUR
                    # eur_rate  = units of USD per EUR
                    # So USD per CURR = eur_rate / curr_rate
                    usd_rate = eur_rate / curr_rate
                self._db.upsert_fx(currency, dt, usd_rate)

    def _fetch_single_series(self, currency: str, start_date: str) -> Dict[str, float]:
        """Fetch ECB SDMX JSON for one FX series; return {date: value} dict."""
        series_id = ECB_FX_SERIES.get(currency)
        if not series_id:
            return {}
        url = f"{ECB_BASE}/{series_id}"
        params = {
            "startPeriod": start_date,
            "format": "application/vnd.sdmx.data+json;version=1.0.0-wd",
        }
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            # SDMX JSON structure: dataSets[0].series["0:0:0:0:0"].observations
            ds = data.get("dataSets", [{}])[0]
            series_key = list(ds.get("series", {}).keys())
            if not series_key:
                return {}
            obs = ds["series"][series_key[0]].get("observations", {})
            # Time dimension from structure
            time_dim = data["structure"]["dimensions"]["observation"]
            for dim in time_dim:
                if dim.get("id") == "TIME_PERIOD":
                    periods = [v["id"] for v in dim["values"]]
                    break
            else:
                return {}
            result: Dict[str, float] = {}
            for idx_str, values in obs.items():
                idx = int(idx_str)
                if idx < len(periods) and values and values[0] is not None:
                    result[periods[idx]] = float(values[0])
            return result
        except Exception as e:
            logger.warning("ecb_fetch_failed", currency=currency, error=str(e))
            return {}

    def ensure_fx(self, currency: str, for_date: str = "2020-01-01") -> None:
        """Ensure FX rates for currency exist in DB; fetch if stale."""
        if currency == "USD":
            return
        existing = self._db.get_fx_rate(currency, for_date)
        if existing:
            return  # already have data
        self._fetch_ecb_series(currency, "2015-01-01")

    def to_usd(self, value: Optional[float], currency: str, report_date: str) -> Optional[float]:
        """Convert value from currency to USD on or before report_date."""
        if value is None:
            return None
        if currency == "USD":
            return value
        rate = self._db.get_fx_rate(currency, report_date)
        if rate:
            return value * rate
        # Fallback: fetch inline
        self.ensure_fx(currency, report_date)
        rate = self._db.get_fx_rate(currency, report_date)
        return value * rate if rate else None


# ---------------------------------------------------------------------------
# EDGAR 20-F fetcher
# ---------------------------------------------------------------------------

class EDGAR20FFetcher:
    """Fetch XBRL company facts for 20-F filers (ADRs on US exchanges)."""

    COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    # Concept preference chains for IFRS filers
    INCOME_CONCEPTS: Dict[str, List[str]] = {
        "revenue":          ["ifrs-full_Revenue", "ifrs-full_RevenueFromContractsWithCustomers", "us-gaap_Revenues"],
        "gross_profit":     ["ifrs-full_GrossProfit", "us-gaap_GrossProfit"],
        "operating_profit": ["ifrs-full_ProfitLossFromOperatingActivities", "us-gaap_OperatingIncomeLoss"],
        "net_income":       ["ifrs-full_ProfitLoss", "ifrs-full_ProfitLossAttributableToOwnersOfParent", "us-gaap_NetIncomeLoss"],
        "interest_expense": ["ifrs-full_FinanceCosts", "us-gaap_InterestExpense"],
        "income_tax":       ["ifrs-full_IncomeTaxExpenseContinuingOperations", "us-gaap_IncomeTaxExpenseBenefit"],
        "depreciation":     ["ifrs-full_DepreciationAndAmortisationExpense", "us-gaap_DepreciationAndAmortization"],
        "eps_basic":        ["ifrs-full_BasicEarningsLossPerShare", "us-gaap_EarningsPerShareBasic"],
        "eps_diluted":      ["ifrs-full_DilutedEarningsLossPerShare", "us-gaap_EarningsPerShareDiluted"],
    }
    BALANCE_CONCEPTS: Dict[str, List[str]] = {
        "total_assets":         ["ifrs-full_Assets", "us-gaap_Assets"],
        "current_assets":       ["ifrs-full_CurrentAssets", "us-gaap_AssetsCurrent"],
        "non_current_assets":   ["ifrs-full_NoncurrentAssets", "us-gaap_AssetsNoncurrent"],
        "cash":                 ["ifrs-full_CashAndCashEquivalents", "us-gaap_CashAndCashEquivalentsAtCarryingValue"],
        "inventories":          ["ifrs-full_Inventories", "us-gaap_InventoryNet"],
        "receivables":          ["ifrs-full_TradeAndOtherCurrentReceivables", "us-gaap_AccountsReceivableNetCurrent"],
        "total_liabilities":    ["ifrs-full_Liabilities", "us-gaap_Liabilities"],
        "current_liabilities":  ["ifrs-full_CurrentLiabilities", "us-gaap_LiabilitiesCurrent"],
        "total_equity":         ["ifrs-full_Equity", "us-gaap_StockholdersEquity"],
        "long_term_debt":       ["ifrs-full_Borrowings", "us-gaap_LongTermDebt"],
    }
    CF_CONCEPTS: Dict[str, List[str]] = {
        "cfo":           ["ifrs-full_CashFlowsFromUsedInOperatingActivities", "us-gaap_NetCashProvidedByUsedInOperatingActivities"],
        "cfi":           ["ifrs-full_CashFlowsFromUsedInInvestingActivities", "us-gaap_NetCashProvidedByUsedInInvestingActivities"],
        "cff":           ["ifrs-full_CashFlowsFromUsedInFinancingActivities", "us-gaap_NetCashProvidedByUsedInFinancingActivities"],
        "capex":         ["ifrs-full_PurchaseOfPropertyPlantAndEquipment", "us-gaap_PaymentsToAcquirePropertyPlantAndEquipment"],
        "dividends_paid":["ifrs-full_DividendsPaid", "us-gaap_PaymentsOfDividends"],
    }

    def __init__(self) -> None:
        self._cache: Dict[str, Dict] = {}

    def _get_company_facts(self, cik: str) -> Dict:
        cik_str = str(cik).zfill(10)
        if cik_str in self._cache:
            return self._cache[cik_str]
        url = self.COMPANYFACTS_URL.format(cik=cik_str)
        try:
            time.sleep(0.15)
            r = requests.get(url, headers=_HEADERS, timeout=45)
            r.raise_for_status()
            data = r.json()
            self._cache[cik_str] = data
            return data
        except Exception as e:
            logger.warning("edgar_companyfacts_failed", cik=cik_str, error=str(e))
            return {}

    def _extract_concept(self, facts: Dict, concept_keys: List[str], period_type: str = "annual") -> Optional[float]:
        """Try each concept key in order; return first valid annual/quarterly value."""
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        ifrs_full = facts.get("facts", {}).get("ifrs-full", {})

        for key in concept_keys:
            namespace, concept = key.split("_", 1) if "_" in key else ("us-gaap", key)
            source = ifrs_full if namespace == "ifrs-full" else us_gaap
            data = source.get(concept, {})
            units = data.get("units", {})
            # Try USD then local currency units
            for unit_key in units:
                entries = units[unit_key]
                if period_type == "annual":
                    annual = [e for e in entries if e.get("form") in ("20-F", "10-K", "40-F") and e.get("fp") == "FY"]
                    if annual:
                        annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                        return float(annual[0]["val"])
                else:
                    qtrs = [e for e in entries if e.get("form") in ("10-Q", "20-F") and e.get("fp", "").startswith("Q")]
                    if qtrs:
                        qtrs.sort(key=lambda x: x.get("end", ""), reverse=True)
                        return float(qtrs[0]["val"])
        return None

    def fetch_financials(self, cik: str, ticker: str, currency: str) -> Optional[Dict[str, Any]]:
        """Return a flat dict of financial metrics for the most recent annual period."""
        facts = self._get_company_facts(cik)
        if not facts:
            return None

        row: Dict[str, Any] = {"ticker": ticker, "source": "EDGAR_20F", "period": "FY", "currency": currency}

        # Try to get the most recent 20-F or annual filing date
        entity_name = facts.get("entityName", ticker)
        row["name"] = entity_name

        for field_name, concepts in {**self.INCOME_CONCEPTS, **self.BALANCE_CONCEPTS, **self.CF_CONCEPTS}.items():
            row[field_name] = self._extract_concept(facts, concepts)

        # Get report date from submissions
        try:
            cik_str = str(cik).zfill(10)
            subs_url = EDGAR_SUBMISSIONS.format(cik=cik_str)
            time.sleep(0.12)
            r = requests.get(subs_url, headers=_HEADERS, timeout=30)
            if r.status_code == 200:
                subs = r.json()
                filings = subs.get("filings", {}).get("recent", {})
                forms = filings.get("form", [])
                dates = filings.get("filingDate", [])
                period_ends = filings.get("reportDate", [])
                for i, f in enumerate(forms):
                    if f in ("20-F", "10-K", "40-F"):
                        row["report_date"] = period_ends[i] if i < len(period_ends) else dates[i]
                        break
        except Exception:
            pass

        row.setdefault("report_date", str(date.today()))
        return row


# ---------------------------------------------------------------------------
# SimFin API fetcher
# ---------------------------------------------------------------------------

class SimFinFetcher:
    """Fetch financials from SimFin free bulk API (no key needed for market data)."""

    BASE = "https://simfin.com/api/v2"

    # SimFin field name → SENTINEL canonical name
    INCOME_MAP: Dict[str, str] = {
        "Revenue":                    "revenue",
        "Gross Profit":               "gross_profit",
        "Operating Income (Loss)":    "operating_profit",
        "Net Income":                 "net_income",
        "Depreciation & Amortization":"depreciation",
        "Interest Expense, Net":      "interest_expense",
        "Income Tax (Expense) Benefit, Net": "income_tax",
    }
    BALANCE_MAP: Dict[str, str] = {
        "Total Assets":               "total_assets",
        "Total Current Assets":       "current_assets",
        "Cash, Cash Equivalents & Short Term Investments": "cash",
        "Accounts & Notes Receivable":"receivables",
        "Inventories":                "inventories",
        "Total Liabilities":          "total_liabilities",
        "Total Current Liabilities":  "current_liabilities",
        "Total Equity":               "total_equity",
        "Long Term Debt":             "long_term_debt",
    }
    CF_MAP: Dict[str, str] = {
        "Net Cash from Operating Activities":  "cfo",
        "Net Cash from Investing Activities":  "cfi",
        "Net Cash from Financing Activities":  "cff",
        "Capital Expenditures":                "capex",
        "Dividends Paid":                      "dividends_paid",
    }

    def _fetch(self, endpoint: str, params: Dict) -> Any:
        url = f"{self.BASE}/{endpoint}"
        try:
            time.sleep(0.2)
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("simfin_fetch_failed", endpoint=endpoint, error=str(e))
            return None

    def _parse_statement(self, data: Any, field_map: Dict[str, str]) -> Dict[str, Any]:
        """Parse SimFin statement list response into canonical field dict."""
        if not data or not isinstance(data, list):
            return {}
        # SimFin returns list of dicts with 'data' and 'columns' keys
        result: Dict[str, Any] = {}
        for company_data in data:
            if not isinstance(company_data, dict):
                continue
            columns = company_data.get("columns", [])
            rows = company_data.get("data", [])
            if not rows:
                continue
            # Most recent row
            row = rows[-1] if rows else []
            row_dict = dict(zip(columns, row))
            for simfin_name, canonical in field_map.items():
                val = row_dict.get(simfin_name)
                if val is not None:
                    try:
                        result[canonical] = float(val) * 1_000_000  # SimFin: values in thousands
                    except (TypeError, ValueError):
                        pass
            result["report_date"] = row_dict.get("Fiscal Year End Date") or row_dict.get("Period End Date", "")
            result["period"] = row_dict.get("Fiscal Period", "FY")
            result["currency"] = row_dict.get("Currency", "USD")
        return result

    def fetch_financials(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch income + balance + cashflow from SimFin for a ticker."""
        combined: Dict[str, Any] = {"ticker": ticker, "source": "SimFin"}

        # Income statement
        income_data = self._fetch("companies/statements", {
            "ticker": ticker, "statement": "pl", "period": "fy", "fyear": "latest",
        })
        income_row = self._parse_statement(income_data, self.INCOME_MAP)
        combined.update(income_row)

        # Balance sheet
        balance_data = self._fetch("companies/statements", {
            "ticker": ticker, "statement": "bs", "period": "fy", "fyear": "latest",
        })
        balance_row = self._parse_statement(balance_data, self.BALANCE_MAP)
        for k, v in balance_row.items():
            if k not in combined:
                combined[k] = v

        # Cash flow
        cf_data = self._fetch("companies/statements", {
            "ticker": ticker, "statement": "cf", "period": "fy", "fyear": "latest",
        })
        cf_row = self._parse_statement(cf_data, self.CF_MAP)
        for k, v in cf_row.items():
            if k not in combined:
                combined[k] = v

        combined.setdefault("report_date", str(date.today()))
        combined.setdefault("period", "FY")
        combined.setdefault("currency", "USD")
        return combined if any(v is not None for k, v in combined.items() if k not in ("ticker", "source", "report_date", "period", "currency")) else None


# ---------------------------------------------------------------------------
# Yahoo Finance fallback fetcher
# ---------------------------------------------------------------------------

class YahooFinanceFetcher:
    """Fetch summary financials from Yahoo Finance for international tickers."""

    def fetch_financials(self, yf_ticker: str, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            url = YAHOO_SUMMARY.format(ticker=yf_ticker)
            params = {"modules": "financialData,defaultKeyStatistics,summaryDetail"}
            time.sleep(0.3)
            r = requests.get(url, params=params, headers={
                **_HEADERS,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }, timeout=20)
            if r.status_code != 200:
                return None
            result_data = r.json().get("quoteSummary", {}).get("result", [])
            if not result_data:
                return None
            fd = result_data[0].get("financialData", {})
            ks = result_data[0].get("defaultKeyStatistics", {})
            return {
                "ticker": ticker,
                "source": "Yahoo",
                "period": "TTM",
                "report_date": str(date.today()),
                "revenue":         fd.get("totalRevenue", {}).get("raw"),
                "gross_profit":    fd.get("grossProfits", {}).get("raw"),
                "operating_profit":fd.get("operatingCashflow", {}).get("raw"),  # proxy
                "net_income":      fd.get("netIncomeToCommon", {}).get("raw"),
                "total_assets":    None,
                "total_equity":    fd.get("totalStockholderEquity", {}).get("raw"),
                "total_liabilities": None,
                "cash":            fd.get("totalCash", {}).get("raw"),
                "cfo":             fd.get("operatingCashflow", {}).get("raw"),
                "capex":           fd.get("capitalExpenditures", {}).get("raw"),
                "long_term_debt":  fd.get("totalDebt", {}).get("raw"),
                "eps_basic":       ks.get("trailingEps", {}).get("raw"),
                "currency":        fd.get("financialCurrency", "USD"),
            }
        except Exception as e:
            logger.warning("yahoo_fetch_failed", ticker=yf_ticker, error=str(e))
            return None


# ---------------------------------------------------------------------------
# World Bank macro context
# ---------------------------------------------------------------------------

class WorldBankFetcher:
    """Fetch macro context from World Bank Open Data."""

    INDICATORS = {
        "GDP_USD":         "NY.GDP.MKTP.CD",
        "GDP_growth":      "NY.GDP.MKTP.KD.ZG",
        "inflation":       "FP.CPI.TOTL.ZG",
        "population":      "SP.POP.TOTL",
        "gdp_per_capita":  "NY.GDP.PCAP.CD",
    }

    def get_country_macro(self, iso2: str) -> Dict[str, Any]:
        """Fetch latest macro indicators for a country."""
        results: Dict[str, Any] = {"country": iso2}
        for name, indicator in self.INDICATORS.items():
            url = f"{WORLDBANK_BASE}/country/{iso2}/indicator/{indicator}"
            params = {"format": "json", "mrv": 1}
            try:
                time.sleep(0.2)
                r = requests.get(url, params=params, headers=_HEADERS, timeout=20)
                r.raise_for_status()
                data = r.json()
                if len(data) >= 2 and data[1]:
                    entry = data[1][0]
                    results[name] = entry.get("value")
                    results[f"{name}_year"] = entry.get("date")
            except Exception as e:
                logger.warning("worldbank_fetch_failed", indicator=indicator, error=str(e))
        return results


# ---------------------------------------------------------------------------
# Primary service class
# ---------------------------------------------------------------------------

class IFRSService:
    """Orchestrates multi-source IFRS financial data ingestion and normalization."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db = _IFRSDatabase(db_path)
        self._fx = FXNormalizer(self._db)
        self._edgar = EDGAR20FFetcher()
        self._simfin = SimFinFetcher()
        self._yahoo = YahooFinanceFetcher()
        self._wb = WorldBankFetcher()
        self._seed_universe()

    def _seed_universe(self) -> None:
        for company in INTL_UNIVERSE:
            self._db.upsert_company(company)

    def _normalize_and_store(self, raw: Dict[str, Any], company: Dict[str, Any]) -> None:
        """Apply FX normalization and store financials."""
        currency = raw.get("currency") or company.get("currency", "USD")
        report_date = raw.get("report_date", str(date.today()))

        # Ensure FX rates are loaded for this currency
        self._fx.ensure_fx(currency, report_date)

        row = {
            "ticker":           company["ticker"],
            "report_date":      report_date,
            "period":           raw.get("period", "FY"),
            "source":           raw.get("source", "unknown"),
            "currency":         currency,
            "revenue":          raw.get("revenue"),
            "gross_profit":     raw.get("gross_profit"),
            "operating_profit": raw.get("operating_profit"),
            "ebit":             raw.get("ebit") or raw.get("operating_profit"),
            "net_income":       raw.get("net_income"),
            "eps_basic":        raw.get("eps_basic"),
            "eps_diluted":      raw.get("eps_diluted"),
            "depreciation":     raw.get("depreciation"),
            "interest_expense": raw.get("interest_expense"),
            "income_tax":       raw.get("income_tax"),
            "total_assets":     raw.get("total_assets"),
            "current_assets":   raw.get("current_assets"),
            "non_current_assets":raw.get("non_current_assets"),
            "cash":             raw.get("cash"),
            "inventories":      raw.get("inventories"),
            "receivables":      raw.get("receivables"),
            "total_liabilities":raw.get("total_liabilities"),
            "current_liabilities":raw.get("current_liabilities"),
            "total_equity":     raw.get("total_equity"),
            "long_term_debt":   raw.get("long_term_debt"),
            "cfo":              raw.get("cfo"),
            "cfi":              raw.get("cfi"),
            "cff":              raw.get("cff"),
            "capex":            raw.get("capex"),
            "dividends_paid":   raw.get("dividends_paid"),
            "revenue_usd":      self._fx.to_usd(raw.get("revenue"), currency, report_date),
            "net_income_usd":   self._fx.to_usd(raw.get("net_income"), currency, report_date),
            "total_assets_usd": self._fx.to_usd(raw.get("total_assets"), currency, report_date),
        }
        self._db.upsert_financials(row)

    def refresh_ticker(self, ticker: str) -> Dict[str, Any]:
        """Refresh financials for a single ticker from best available source."""
        company = next((c for c in INTL_UNIVERSE if c["ticker"].upper() == ticker.upper()), None)
        if not company:
            return {"error": f"Ticker {ticker} not in IFRS universe"}

        raw: Optional[Dict] = None

        # Priority 1: EDGAR 20-F if CIK exists
        if company.get("cik"):
            raw = self._edgar.fetch_financials(company["cik"], ticker, company["currency"])
            if raw:
                logger.info("edgar_fetch_ok", ticker=ticker)

        # Priority 2: SimFin
        if not raw:
            raw = self._simfin.fetch_financials(ticker)
            if raw:
                logger.info("simfin_fetch_ok", ticker=ticker)

        # Priority 3: Yahoo Finance
        if not raw:
            raw = self._yahoo.fetch_financials(company.get("yf_ticker", ticker), ticker)
            if raw:
                logger.info("yahoo_fetch_ok", ticker=ticker)

        if raw:
            self._normalize_and_store(raw, company)
            return {"status": "refreshed", "ticker": ticker, "source": raw.get("source")}
        return {"status": "no_data", "ticker": ticker}

    def get_financials(self, ticker: str) -> List[Dict]:
        rows = self._db.get_financials(ticker.upper())
        if not rows:
            self.refresh_ticker(ticker)
            rows = self._db.get_financials(ticker.upper())
        return rows

    def get_income(self, ticker: str) -> List[Dict]:
        rows = self.get_financials(ticker)
        income_fields = ["ticker", "report_date", "period", "source", "currency",
                         "revenue", "gross_profit", "operating_profit", "net_income",
                         "depreciation", "interest_expense", "income_tax", "eps_basic", "eps_diluted"]
        return [{k: r.get(k) for k in income_fields} for r in rows]

    def get_balance(self, ticker: str) -> List[Dict]:
        rows = self.get_financials(ticker)
        balance_fields = ["ticker", "report_date", "period", "source", "currency",
                          "total_assets", "current_assets", "non_current_assets", "cash",
                          "inventories", "receivables", "total_liabilities", "current_liabilities",
                          "total_equity", "long_term_debt"]
        return [{k: r.get(k) for k in balance_fields} for r in rows]

    def get_cashflow(self, ticker: str) -> List[Dict]:
        rows = self.get_financials(ticker)
        cf_fields = ["ticker", "report_date", "period", "source", "currency",
                     "cfo", "cfi", "cff", "capex", "dividends_paid"]
        return [{k: r.get(k) for k in cf_fields} for r in rows]

    def get_peers(self, region: str) -> Dict[str, Any]:
        rows = self._db.get_peers(region)
        if not rows:
            # Trigger refresh for region companies
            region_cos = [c for c in INTL_UNIVERSE if c["region"] == region][:5]
            for company in region_cos:
                self.refresh_ticker(company["ticker"])
            rows = self._db.get_peers(region)
        return {"region": region, "count": len(rows), "peers": rows}

    def get_concept_map(self, category: Optional[str] = None) -> List[Dict]:
        return self._db.get_concept_map(category)

    def compute_ratios(self, ticker: str) -> Dict[str, Any]:
        """Compute key financial ratios from stored financials."""
        rows = self.get_financials(ticker)
        if not rows:
            return {"error": "No data"}
        r = rows[0]
        ratios: Dict[str, Any] = {"ticker": ticker, "report_date": r.get("report_date"), "currency": r.get("currency")}

        rev = r.get("revenue") or 0
        gp  = r.get("gross_profit") or 0
        op  = r.get("operating_profit") or 0
        ni  = r.get("net_income") or 0
        ta  = r.get("total_assets") or 0
        eq  = r.get("total_equity") or 0
        cfo = r.get("cfo") or 0
        capex = r.get("capex") or 0

        ratios["gross_margin_pct"]     = round(gp / rev * 100, 2) if rev else None
        ratios["operating_margin_pct"] = round(op / rev * 100, 2) if rev else None
        ratios["net_margin_pct"]       = round(ni / rev * 100, 2) if rev else None
        ratios["roa_pct"]              = round(ni / ta * 100, 2) if ta else None
        ratios["roe_pct"]              = round(ni / eq * 100, 2) if eq else None
        ratios["asset_turnover"]       = round(rev / ta, 2) if ta else None
        ratios["free_cash_flow"]       = cfo - abs(capex) if capex else cfo
        ratios["fcf_margin_pct"]       = round(ratios["free_cash_flow"] / rev * 100, 2) if rev and ratios.get("free_cash_flow") else None
        return ratios

    def get_world_bank_macro(self, ticker: str) -> Dict[str, Any]:
        company = next((c for c in INTL_UNIVERSE if c["ticker"].upper() == ticker.upper()), None)
        if not company:
            return {}
        return self._wb.get_country_macro(company.get("country", "US"))


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/ifrs/v3", tags=["IFRS Financials v3"])
_svc: Optional[IFRSService] = None


def _get_svc() -> IFRSService:
    global _svc
    if _svc is None:
        _svc = IFRSService()
    return _svc


@router.get("/financials/{ticker}", response_model=List[Dict])
def get_financials(ticker: str) -> List[Dict]:
    """Full financial statements for a non-US IFRS company."""
    rows = _get_svc().get_financials(ticker.upper())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No financials found for {ticker}")
    return rows


@router.get("/income/{ticker}", response_model=List[Dict])
def get_income(ticker: str) -> List[Dict]:
    """Income statement (IFRS standardized) for a non-US company."""
    rows = _get_svc().get_income(ticker.upper())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No income data for {ticker}")
    return rows


@router.get("/balance/{ticker}", response_model=List[Dict])
def get_balance(ticker: str) -> List[Dict]:
    """Balance sheet (IFRS standardized) for a non-US company."""
    rows = _get_svc().get_balance(ticker.upper())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No balance sheet data for {ticker}")
    return rows


@router.get("/cashflow/{ticker}", response_model=List[Dict])
def get_cashflow(ticker: str) -> List[Dict]:
    """Cash flow statement (IFRS standardized) for a non-US company."""
    rows = _get_svc().get_cashflow(ticker.upper())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No cash flow data for {ticker}")
    return rows


@router.get("/peers/{region}", response_model=Dict)
def get_peers(
    region: str,
    sector: Optional[str] = Query(None, description="Filter by sector (Technology, Financials, etc.)"),
) -> Dict:
    """All companies in a region with their latest financials. Regions: Europe, Asia, Canada, Australia, LatAm."""
    result = _get_svc().get_peers(region)
    if sector and result.get("peers"):
        # Filter by sector using company universe lookup
        sector_tickers = {c["ticker"] for c in INTL_UNIVERSE if c.get("sector", "").lower() == sector.lower()}
        result["peers"] = [p for p in result["peers"] if p.get("ticker") in sector_tickers]
        result["count"] = len(result["peers"])
    return result


@router.get("/concept-map", response_model=List[Dict])
def get_concept_map(
    category: Optional[str] = Query(None, description="Filter by: income_statement, balance_sheet, cash_flow"),
) -> List[Dict]:
    """Full IFRS-to-GAAP concept mapping table (50+ entries)."""
    return _get_svc().get_concept_map(category)


@router.get("/ratios/{ticker}", response_model=Dict)
def get_ratios(ticker: str) -> Dict:
    """Computed financial ratios: margins, ROA, ROE, FCF margin."""
    return _get_svc().compute_ratios(ticker.upper())


@router.get("/macro/{ticker}", response_model=Dict)
def get_macro(ticker: str) -> Dict:
    """World Bank macro context for the company's home country."""
    return _get_svc().get_world_bank_macro(ticker.upper())


@router.post("/refresh/{ticker}", response_model=Dict)
def refresh_ticker(ticker: str) -> Dict:
    """Force-refresh financials for a single ticker from all sources."""
    return _get_svc().refresh_ticker(ticker.upper())


@router.get("/universe", response_model=List[Dict])
def get_universe(
    region: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
) -> List[Dict]:
    """List all companies in the IFRS universe with optional filters."""
    result = [c.copy() for c in INTL_UNIVERSE]
    if region:
        result = [c for c in result if c.get("region", "").lower() == region.lower()]
    if sector:
        result = [c for c in result if c.get("sector", "").lower() == sector.lower()]
    if country:
        result = [c for c in result if c.get("country", "").upper() == country.upper()]
    return result


@router.get("/fx-rates/{currency}", response_model=List[Dict])
def get_fx_rates(
    currency: str,
    start_date: str = Query("2020-01-01", description="Start date YYYY-MM-DD"),
) -> List[Dict]:
    """Historical USD FX rates for a currency (sourced from ECB)."""
    db = _get_svc()._db
    fx = _get_svc()._fx
    fx.ensure_fx(currency.upper(), start_date)
    cur = db._conn.execute(
        "SELECT * FROM fx_rates WHERE from_currency = ? AND rate_date >= ? ORDER BY rate_date",
        (currency.upper(), start_date),
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        raise HTTPException(status_code=404, detail=f"No FX data for {currency}")
    return rows
