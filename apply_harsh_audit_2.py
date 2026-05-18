"""
Harsh re-audit 2026-05-18: 4 agents read every source file directly.
No trust of self-scores. Findings from actual code inspection.
"""
import json, datetime

with open('.danteforge/compete/matrix.json', encoding='utf-8') as f:
    m = json.load(f)

EXCLUDED = {10, 12, 18, 40, 41, 42, 58, 69, 87, 88, 94, 99}

# Scores from 4 parallel agents reading actual source files
harsh2 = {
    # MARKET DATA
    'dim_001': 8,   # 3-source fallback real; yfinance volume=3mo avg; holiday table expires 2026
    'dim_002': 8,   # Solid validation; Yahoo pre-2000 degrades
    'dim_003': 7,   # Free Alpaca = 2yr ceiling (docstring claims 10yr — misleading)
    'dim_004': 9,   # 5+5 Greeks, GEX, max pain — genuinely production-grade
    'dim_005': 7,   # VIX via CBOE CSV solid; commodity yfinance futures fragile/empty
    'dim_006': 7,   # Parkinson vol (not Garman-Klass as claimed); SABR fits degenerate on thin FX options
    'dim_007': 7,   # 18 exchanges not 20; defunct FTX listed; arb math real
    'dim_008': 7,   # FSM real; EDGAR text term parsing fragile; static completion probabilities
    'dim_009': 8,   # FINRA daily+monthly+SEC FTD; FTD URL scraper fragile
    'dim_011': 7,   # Real Alpaca+yfinance data; session constants correct; gap analytics genuine
    # FUNDAMENTALS
    'dim_013': 8,   # 40+ XBRL concepts; Finviz consensus estimates scraped (fragile)
    'dim_014': 8,   # 50+ XBRL; OBS items (VIEs, pensions) only partially extractable
    'dim_015': 8,   # Multiple FCF definitions correct; XBRL sign convention handled
    'dim_016': 8,   # HHI correct; dual XBRL+HTML; non-standard segment axis endpoint
    'dim_017': 8,   # EQI formula implemented; cookie-jar real; 8-K HTML parsing fragile
    'dim_019': 8,   # Beat/miss/streak real; Finviz estimates fragile; 8-K EPS regex incomplete
    'dim_020': 8,   # Correct PIT via EDGAR filingDate; Wikipedia S&P 500 fragile
    'dim_021': 8,   # 51 IFRS mappings confirmed; 200-company universe; pure-foreign non-ADR thin
    'dim_022': 8,   # Stale penalty + restatement detection; Wikipedia universe fragile
    # CORPORATE INTELLIGENCE
    'dim_023': 8,   # Hamada correct; Monte Carlo uses Python random not numpy normal
    'dim_024': 8,   # EV bridge + LBO back-solve real; SIC peer selection misses economic peers
    'dim_025': 8,   # HHI correct; smart money heuristic; institution categorization not authoritative
    'dim_026': 8,   # Form 4 + 10b5-1 footnote detection; conviction score implemented
    'dim_027': 7,   # Settlement prob real; hardcoded illustrative fallback list at line 1771
    'dim_028': 8,   # 20 governance components; pre-2021 SCT parsing fragile
    'dim_029': 7,   # Missing cluster analysis; 3-tier dollar heuristic only
    'dim_030': 7,   # IPO pop prediction is uncalibrated linear formula (not trained)
    'dim_031': 8,   # Correct Reg D thresholds (504/506b/506c/CF); no stubs
    'dim_032': 8,   # Active share = Cremers-Petajisto exactly; 200+ fund universe
    'dim_033': 8,   # EDGAR EFTS + TF-IDF NLP pipeline solid; no stubs
    'dim_034': 8,   # IAPD bulk CSV + log-scale AUM scoring; real multi-source
    # FIXED INCOME
    'dim_035': 8,   # NSS 6-param correct; FOMC dates hardcoded through 2027
    'dim_036': 8,   # Newton-Raphson YTM (200 iter, 1e-8 tol) + real CUSIPs + VWAP
    'dim_037': 8,   # TEY = muni/(1-combined_rate); EMMA TradeSearch real
    'dim_038': 8,   # Ho-Lee binomial OAS; live FRED curve (no hardcoded rates)
    'dim_039': 8,   # Merton scipy.fsolve simultaneous solve; KMV default point
    # MACRO
    'dim_043': 8,   # 200+ curated FRED series; MacroSeriesLibrary organized
    'dim_044': 7,   # Surprise index correct; consensus from bot-blocked scrapers (fragile)
    'dim_045': 8,   # Bigram lexicon + negation detection (6-token lookback) + intensity
    'dim_046': 6,   # Claims "100+ futures markets" — actual code: 44 market entries
    'dim_047': 9,   # Estrella-Mishkin Φ(-0.6002-0.5288×T10Y3M) exact; Wright (2006) exact
    'dim_048': 7,   # VIX term structure real; inflation thresholds hardcoded without calibration
    'dim_049': 7,   # 35+ countries (not 49 as claimed); World Bank+IMF+OECD real
    'dim_050': 9,   # Full Baum-Welch EM in pure numpy; Viterbi decoding; 4-state HMM
    # AI / NLP
    'dim_051': 8,   # RRF hybrid (k=60); 3-tier storage (Chroma→sqlite-vec→numpy)
    'dim_052': 8,   # FinBERT + LM lexicon + VADER + Spearman IC backtest
    'dim_053': 7,   # Claude tool-use; fallback 8 screens only; no SIC mapping
    'dim_054': 7,   # Claude tool-use; 10368 combo grid not found in code
    'dim_055': 8,   # Multi-section EDGAR summarizer; Claude + TF-IDF fallback
    'dim_056': 7,   # HyDE present; only 30-group synonym lexicon; no concept graph
    'dim_057': 8,   # Guidance window extraction correct; tone_shift; multi-source
    'dim_059': 8,   # 131 wired tools (docstring says 54); no stub bodies found
    'dim_060': 8,   # 5 parallel async sources; news sentiment is keyword-only (not FinBERT)
    # BACKTESTING
    'dim_061': 8,   # Ulcer/Pain in separate class, not wired into primary metrics dict
    'dim_062': 8,   # NautilusTrader is dead stub; beta vs SPY hardcoded
    'dim_063': 8,   # MC permutation split across two modules; integration implicit
    'dim_064': 9,   # DSR Bailey (2014) Gumbel exact; Haircut SR Harvey+Liu; CPCV correct
    'dim_065': 8,   # Almgren-Chriss enumerated but optimal trajectory not solved; VWAP profile unfilled
    'dim_066': 8,   # Full order lifecycle + PnL attribution; AC impact needs deeper verification
    'dim_067': 8,   # Promotion FSM complete; capacity is tiered flat, not liquidity-derived
    'dim_068': 7,   # IC series = single cross-section (not rolling); GP fragile XBRL; macro NaN
    # SCREENERS
    'dim_070': 7,   # Beneish M stubbed (returns None); Piotroski 2/9 signals mis-implemented
    'dim_071': 8,   # 29 criteria with Ichimoku; no intraday volume profile (yfinance can't provide)
    'dim_072': 7,   # Float squeeze absent; institutional momentum single-period delta only
    'dim_073': 8,   # Real BS Greeks; sweep detection absent (requires T&S data)
    'dim_074': 8,   # 140 bonds (not 170 as claimed); FINRA HY pricing fragile
    'dim_075': 8,   # CoinGecko real; MVRV/SOPR silently null without realised cap source
    'dim_076': 8,   # Rule-based NL pipeline complete; Claude enhancement optional
    # PORTFOLIO / RISK
    'dim_077': 8,   # GARCH(1,1) correct; EGARCH convergence fragile without scipy
    'dim_078': 9,   # Carino/Menchero/GRAP all match published papers; Ken French direct
    'dim_079': 8,   # FF5+MOM OLS correct; crowding detection is half-period heuristic
    'dim_080': 7,   # DCC-GARCH is EWMA of pseudo-residuals (not genuine 2-step MLE)
    'dim_081': 8,   # MVO/BL/HRP/LW correct; Michaud resampling is approximation
    'dim_082': 8,   # James-Stein Kelly correct; options Kelly ignores vol surface
    'dim_083': 8,   # 11 historical scenarios + Student-t MC; reverse stress test absent
    # ALT DATA
    'dim_084': 7,   # Market reaction alpha hardcoded; no OOS IC validation
    'dim_085': 7,   # Reddit/StockTwits unofficial APIs; 40% signal breaks if Reddit enforces auth
    'dim_086': 7,   # BLS+FRED real; LinkedIn hiring via Google search hack (fragile)
    'dim_089': 6,   # pytrends = unofficial Google scraper; breaks regularly; solid fallback
    # INTERFACE / API
    'dim_090': 6,   # 102 registered codes; ~80+ return _stub() silently (only HP, WACC, CDSW wired)
    'dim_091': 7,   # Save/load/export real; no complex math; solid workspace persistence
    'dim_092': 7,   # Full UDF spec; polling only (no websocket push); Alpaca requires key
    'dim_093': 7,   # openpyxl correct; Google Sheets requires service account; RTD simulated
    'dim_095': 8,   # Token bucket correct; no key refresh/rotation mechanism
    'dim_096': 8,   # TimescaleDB hypertables + compression + retention; image tag unpinned
    # PRIVATE MARKETS
    'dim_097': 7,   # EDGAR Form D real; implied valuation is sector-multiple heuristic
    'dim_098': 7,   # 50 hardcoded fund CIKs real; velocity basic; no IRR/DPI tracking
    # M&A / CORP FINANCE
    'dim_100': 8,   # 7-state FSM complete; sector premium tables static (not calibrated)
    'dim_101': 8,   # Newton-Raphson IRR + 6-tranche waterfall; tax shield simplified
    # ESG
    'dim_102': 6,   # CDP scraping fragile; EPA ECHO+OSHA real; no esg_momentum field found
    'dim_103': 7,   # 4-pillar TCFD; 3-tier GHG extraction; scope 3 text accuracy noisy
    'dim_104': 7,   # Exponential decay (90-day half-life) correct; GDELT+EDGAR real
    'dim_105': 7,   # 450+ SIC-SDG mappings; scoring weights static (not empirically derived)
    # CRYPTO / DEFI / ON-CHAIN
    'dim_106': 6,   # CCXT connectivity real; no SOR/TWAP found; 0.05% constant slippage
    'dim_107': 7,   # IL formula 2√r/(1+r)-1 correct; cascade = first-order only
    'dim_108': 4,   # NVT/MVRV are CoinGecko proxies — not actual on-chain metrics (file self-declares 4)
    'dim_109': 4,   # 20 hardcoded ERC-20 addresses; Etherscan free = no mempool (file self-declares 4)
    'dim_110': 7,   # Uniswap V3 sqrtPriceX96 correct; rugpull weights not backtested
}

changes = []
for d in m['dimensions']:
    dim_id = d['id']
    if dim_id in harsh2:
        old = d['scores'].get('self', 0)
        new = harsh2[dim_id]
        if old != new:
            direction = 'UP' if new > old else 'DOWN'
            changes.append((dim_id, old, new, direction))
            d['scores']['self'] = new

print(f"Changed {len(changes)} dims:")
for dim_id, old, new, direction in sorted(changes, key=lambda x: x[2]):
    print(f"  {dim_id}: {old} -> {new} ({direction})")

eligible = [d for d in m['dimensions']
            if int(d['id'].split('_')[1]) not in EXCLUDED and d['scores'].get('self', 0) > 0]
total_w = sum(d['weight'] for d in eligible)
ws = sum(d['scores']['self'] * d['weight'] for d in eligible)
new_comp = round(ws / total_w, 4)

old_comp = m.get('selfComposite', 0)
print(f"\nComposite: {old_comp} -> {new_comp}")
print(f"Eligible dims: {len(eligible)}, total_weight: {total_w:.1f}")

# Score distribution
nines = [d['id'] for d in eligible if d['scores']['self'] == 9]
eights = sum(1 for d in eligible if d['scores']['self'] == 8)
sevens = sum(1 for d in eligible if d['scores']['self'] == 7)
sixes = [d['id'] for d in eligible if d['scores']['self'] == 6]
fours = [d['id'] for d in eligible if d['scores']['self'] == 4]
print(f"\nScore 9 ({len(nines)}): {', '.join(nines)}")
print(f"Score 8: {eights} dims")
print(f"Score 7: {sevens} dims")
print(f"Score 6 ({len(sixes)}): {', '.join(sixes)}")
print(f"Score 4 ({len(fours)}): {', '.join(fours)}")

m['selfComposite'] = new_comp
m['overallSelfScore'] = new_comp
m['lastUpdated'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
m['harshReviewDate2'] = '2026-05-18T00:00:00.000Z'
m['harshReviewNotes2'] = (
    'Harsh re-audit 2026-05-18: 4 agents read every source file. '
    'Key downgrades: dim_046=6 (44 markets not 100+), dim_090=6 (~80 stubs), '
    'dim_108=4 (proxies not on-chain), dim_109=4 (static addresses), '
    'dim_089=6 (pytrends unofficial), dim_102=6 (CDP fragile), dim_106=6 (no real SOR). '
    'Key 9s confirmed: dim_004 (BS Greeks), dim_047 (probit exact), dim_050 (Baum-Welch), '
    'dim_064 (DSR/PBO/CPCV), dim_078 (Carino/Menchero/GRAP). '
    f'Composite {old_comp} -> {new_comp}.'
)

with open('.danteforge/compete/matrix.json', 'w', encoding='utf-8') as f:
    json.dump(m, f, indent=2)

print("\nmatrix.json updated.")

# Category averages
cats = {}
for d in m['dimensions']:
    if int(d['id'].split('_')[1]) not in EXCLUDED and d['scores'].get('self', 0) > 0:
        cat = d.get('category', 'other')
        cats.setdefault(cat, []).append(d['scores']['self'])
print("\nCategory averages (post-audit):")
for cat, scores in sorted(cats.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg = sum(scores)/len(scores)
    print(f"  {cat}: {avg:.2f} (n={len(scores)})")
